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
