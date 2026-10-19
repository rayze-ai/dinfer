//! MXFP4 解包对拍。参考由 `scripts/verify/gen_mxfp4_refs.py` 生成（按位构造，不依赖 torch 的 fp4 dtype）。
//!
//! 运行：`MXFP4_REFS_DIR=<dir> cargo test -p dinfer-kernels --features ref-tests --test mxfp4_parity -- --ignored`
#![cfg(feature = "ref-tests")]

use dinfer_kernels::{bf16, Device, Stream};
use std::path::PathBuf;

fn root() -> PathBuf {
    let p = PathBuf::from(std::env::var("MXFP4_REFS_DIR").unwrap_or_else(|_| "test/kimik3/2026-10-19/mxfp4_refs".into()));
    if p.is_absolute() { p } else { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join(p) }
}

fn load_u8(case: &str, name: &str) -> (Vec<usize>, Vec<u8>) {
    let bytes = std::fs::read(root().join(case).join(format!("{name}.npy"))).unwrap();
    let arr = npyz::NpyFile::new(&bytes[..]).unwrap();
    let shape: Vec<usize> = arr.shape().iter().map(|&d| d as usize).collect();
    (shape, arr.into_vec::<u8>().unwrap())
}

fn load_f32(case: &str, name: &str) -> Vec<f32> {
    let bytes = std::fs::read(root().join(case).join(format!("{name}.npy"))).unwrap();
    let arr = npyz::NpyFile::new(&bytes[..]).unwrap();
    arr.into_vec::<f32>().unwrap()
}

#[test]
#[ignore]
fn mxfp4_dequant_all_cases() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let cases = ["moE_w1_small", "moE_w2_small", "group32_odd_rows"];
    for case in cases {
        let (ps, packed) = load_u8(case, "packed");
        let (ss, scale) = load_u8(case, "scale");
        let want = load_f32(case, "out");
        let rows = ps[0];
        let cols = ps[1] * 2;
        let group = cols / ss[1];
        let dp = st.htod_copy(packed).unwrap();
        let ds = st.htod_copy(scale).unwrap();
        let mut dout = st.alloc_uninit::<bf16>(rows * cols).unwrap();
        dinfer_kernels::ops::mxfp4_dequant_bf16(&st, &dp, &ds, &mut dout, rows, cols, group).unwrap();
        st.synchronize().unwrap();
        let got = st.dtoh_sync_copy(&dout).unwrap();
        assert_eq!(got.len(), want.len(), "{case}: 长度");
        let mut max_err = 0f32;
        for (g, w) in got.iter().zip(&want) {
            let e = (g.to_f32() - w).abs();
            let rel = e / w.abs().max(1e-6);
            assert!(rel <= 1e-3 || e <= 1e-6, "{case}: {} vs {w}", g.to_f32());
            max_err = max_err.max(e);
        }
        println!("  {case:<20} rows={rows} cols={cols} group={group} max_err {max_err:.3e}");
    }
}

// --------------------------------------------------------------------------- grouped GEMM

fn load_u8_3d(case: &str, name: &str) -> (Vec<usize>, Vec<u8>) {
    let bytes = std::fs::read(root().join(case).join(format!("{name}.npy"))).unwrap();
    let arr = npyz::NpyFile::new(&bytes[..]).unwrap();
    let shape: Vec<usize> = arr.shape().iter().map(|&d| d as usize).collect();
    (shape, arr.into_vec::<u8>().unwrap())
}

fn load_i32(case: &str, name: &str) -> Vec<i32> {
    let bytes = std::fs::read(root().join(case).join(format!("{name}.npy"))).unwrap();
    let arr = npyz::NpyFile::new(&bytes[..]).unwrap();
    arr.into_vec::<i32>().unwrap()
}

/// grouped GEMM 对拍：`out[r, n] = Σ_k dequant(w[e, n, k])·x[r, k]`，容差 bf16 输出 atol 1e-2 + cos 0.999。
#[test]
#[ignore]
fn mxfp4_gemm_all_cases() {
    use dinfer_kernels::ops;
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    for case in ["gemm_small", "gemm_w1_like", "gemm_w2_like"] {
        let (ps, packed) = load_u8_3d(case, "packed");   // [E, N, K/2]
        let (ss, scale) = load_u8_3d(case, "scale");     // [E, N, K/group]
        let (xs, x) = {
            let bytes = std::fs::read(root().join(case).join("x.npy")).unwrap();
            let arr = npyz::NpyFile::new(&bytes[..]).unwrap();
            let shape: Vec<usize> = arr.shape().iter().map(|&d| d as usize).collect();
            (shape, arr.into_vec::<f32>().unwrap())
        };
        let want = load_f32(case, "out");
        let off = load_i32(case, "expert_off");
        let (e, n, kh) = (ps[0], ps[1], ps[2]);
        let k = kh * 2;
        let group = k / ss[2];
        let pairs = xs[0];
        assert_eq!(xs[1], k, "{case}: x 的 K");
        assert_eq!(off.len(), e + 1, "{case}: expert_off 长度");

        let dx = st.htod_copy(x.iter().map(|&v| bf16::from_f32(v)).collect::<Vec<_>>()).unwrap();
        let dwp = st.htod_copy(packed).unwrap();
        let dws = st.htod_copy(scale).unwrap();
        let doff = st.htod_copy(off.clone()).unwrap();
        let mut dout = st.alloc_zeros::<bf16>(pairs * n).unwrap();
        let tiles = ops::Mxfp4GemmTiles::new(&st, &off, n, 16, 64).unwrap();
        ops::mxfp4_gemm_grouped(&st, &dx, &dwp, &dws, &doff, &tiles, &mut dout, k, n, group).unwrap();
        st.synchronize().unwrap();
        let got = st.dtoh_sync_copy(&dout).unwrap();

        // 相对误差口径：mx*4 的输出量级随 K 与 scale 变化（真权重下 |out| ~ 1e1，
        // bf16 输出本身的粒度就是 2^-8 相对）。逐元素比 max_err/|want|，外加 cos。
        let mut max_err = 0f32;
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        let amax = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
        for (g, w) in got.iter().zip(&want) {
            let gv = g.to_f32();
            max_err = max_err.max((gv - w).abs());
            dot += (gv as f64) * (*w as f64);
            na += (gv as f64) * (gv as f64);
            nb += (*w as f64) * (*w as f64);
        }
        let cos = dot / (na.sqrt() * nb.sqrt());
        // 口径（docs/loops/k3-moe.md）：相对 |out|max 的 max_err ≤ 1e-2 + cos ≥ 0.999。
        // 逐元素绝对 atol 不适用——输出量级随 K 与 e8m0 scale 变化，bf16 输出本身粒度是相对 2^-8。
        println!("  {case:<16} E={e} N={n} K={k} pairs={pairs} tiles={} |out|max {amax:.3} max_err {max_err:.3e} rel {:.2e} cos {cos:.7}",
                 tiles.num_tiles, max_err / amax);
        assert!(max_err / amax <= 1e-2 && cos >= 0.999, "{case}: rel {:.3e} cos {cos:.7}", max_err / amax);
    }
}

// --------------------------------------------------------------------------- 计时（真形状）

/// 计时：真实专家形状（latent MoE 的 w1/w3 与 w2）上的 grouped GEMM；口径见 `docs/loops/k3-moe.md`。
/// 两个真实形状：`[3072, 3584]`（K=3584）与 `[3584, 3072]`（K=3072）。E=8 专家 × 每专家 32 token
/// （= 256 pairs，接近一个 decode 批在单层上的量级）。
/// 同时跑一条「解包成 bf16 + cuBLASLt」的参照，用来看 v1 标量路径离 tensor core 有多远。
#[test]
#[ignore]
fn mxfp4_gemm_bench() {
    use dinfer_kernels::{ops, Gemm};
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let e = 8usize;
    let ntok = 32usize;
    let group = 32usize;

    for (name, n_out, k_in) in [("w1/w3", 3072usize, 3584usize), ("w2", 3584, 3072)] {
        let pairs = e * ntok;
        // 权重用真权重量级的 e8m0 指数 [118,126) 与全 16 种半字节（与 parity case 同口径）。
        let mut packed = vec![0u8; e * n_out * k_in / 2];
        for (i, b) in packed.iter_mut().enumerate() {
            *b = (i as u8) | ((i / 3) as u8) << 4;
        }
        let mut scale = vec![0u8; e * n_out * k_in / group];
        for (i, s) in scale.iter_mut().enumerate() {
            *s = 118 + (i % 8) as u8;
        }
        let x: Vec<bf16> = (0..pairs * k_in).map(|i| bf16::from_f32(((i % 17) as f32 - 8.0) / 64.0)).collect();
        let off: Vec<i32> = (0..=e).map(|i| (i * ntok) as i32).collect();

        let dwp = st.htod_copy(packed.clone()).unwrap();
        let dws = st.htod_copy(scale.clone()).unwrap();
        let dx = st.htod_copy(x.clone()).unwrap();
        let doff = st.htod_copy(off.clone()).unwrap();
        let mut dout = st.alloc_zeros::<bf16>(pairs * n_out).unwrap();
        let tiles = ops::Mxfp4GemmTiles::new(&st, &off, n_out, 16, 64).unwrap();
        let run = |st: &dinfer_kernels::Stream, out: &mut _| {
            ops::mxfp4_gemm_grouped(st, &dx, &dwp, &dws, &doff, &tiles, out, k_in, n_out, group).unwrap();
        };
        run(&st, &mut dout);
        st.synchronize().unwrap();
        let iters = 5;
        let t0 = std::time::Instant::now();
        for _ in 0..iters { run(&st, &mut dout); }
        st.synchronize().unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

        // 参照：解包成 bf16 权重（每专家一份）走 cuBLASLt。解包只做一次，计时单独报。
        // H2D 在计时外（只算 GPU 侧解包时间，与 mxfp4 grouped 同口径）。
        let gemm = Gemm::new(&st).unwrap();
        let dp_e: Vec<_> = (0..e)
            .map(|ei| st.htod_copy(packed[ei * n_out * k_in / 2..(ei + 1) * n_out * k_in / 2].to_vec()).unwrap())
            .collect();
        let ds_e: Vec<_> = (0..e)
            .map(|ei| st.htod_copy(scale[ei * n_out * k_in / group..(ei + 1) * n_out * k_in / group].to_vec()).unwrap())
            .collect();
        let mut wbf: Vec<_> = (0..e).map(|_| st.alloc_uninit::<bf16>(n_out * k_in).unwrap()).collect();
        let mut deq_run = |st: &dinfer_kernels::Stream| {
            for ei in 0..e {
                ops::mxfp4_dequant_bf16(st, &dp_e[ei], &ds_e[ei], &mut wbf[ei], n_out, k_in, group).unwrap();
            }
        };
        deq_run(&st);
        st.synchronize().unwrap();
        let t1 = std::time::Instant::now();
        for _ in 0..iters { deq_run(&st); }
        st.synchronize().unwrap();
        let ms_deq = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;

        // 每专家一次 bf16 GEMM（M=ntok）。x 按专家切成独立缓冲（拷贝在计时外）。
        let xs: Vec<_> = (0..e)
            .map(|ei| st.htod_copy(x[ei * ntok * k_in..(ei + 1) * ntok * k_in].to_vec()).unwrap())
            .collect();
        let mut o_e = st.alloc_uninit::<bf16>(ntok * n_out).unwrap();
        let mut g_run = |st: &dinfer_kernels::Stream| {
            for ei in 0..e {
                gemm.forward_bf16(&xs[ei], &wbf[ei], None, &mut o_e, ntok, n_out, k_in).unwrap();
            }
            let _ = st;
        };
        g_run(&st);
        st.synchronize().unwrap();
        let t2 = std::time::Instant::now();
        for _ in 0..iters { g_run(&st); }
        st.synchronize().unwrap();
        let ms_gemm = t2.elapsed().as_secs_f64() * 1e3 / iters as f64;
        let _ = &mut wbf;

        println!(
            "  {name:<6} N={n_out} K={k_in} E={e} pairs={pairs} tiles={}: mxfp4 grouped {ms:.2} ms | \
             解包 bf16 {ms_deq:.1} ms（一次） | bf16 cuBLASLt ×{e} {ms_gemm:.2} ms",
            tiles.num_tiles
        );
    }
}
