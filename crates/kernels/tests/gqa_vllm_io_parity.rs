//! 用户 2026-10-24 第 2 步：喂 vLLM compile 口径真实 attention 输入 q/k/v，过我们的
//! `gqa_prefill_win_bf16`，比 vLLM 真实 attention 输出。隔离出 attention kernel 本身对不对
//! （不受上游 qkv/norm/rope 末位漂移影响）。
//!
//! 数据（`ATTN_IO_DIR`，由 `kaon3_vllm_attn_io.py` 抓，f32 存）：
//!   vllm_q.bin [T,4096]（fp8 值，已量化）  vllm_k.bin [T,2048] bf16  vllm_v.bin [T,2048] bf16
//!   vllm_out.bin [T,4096] bf16（attention 输出，o_proj 之前）
//! T 由文件大小推（4 B/f32，4096 列）。
//! 跑：`ATTN_IO_DIR=/vllm_attn_io T=22 cargo test --release -p dinfer-kernels --test gqa_vllm_io_parity -- --ignored --nocapture`
use dinfer_kernels::{bf16, ffi, Device};
use std::ffi::c_void;

fn rd(dir: &str, name: &str) -> Vec<f32> {
    let b = std::fs::read(format!("{dir}/{name}.bin")).unwrap_or_else(|_| panic!("读 {name}"));
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// CPU f64 attention 参照（因果+滑窗），输入是 bf16 量化后的 q/k/v（f32）。与 gqa_window_parity::cpu_ref 同口径。
fn cpu_ref(q: &[f32], k: &[f32], v: &[f32], t: usize, h: usize, hkv: usize, d: usize, window: usize) -> Vec<f32> {
    let mut out = vec![0f32; t * h * d];
    for i in 0..t {
        let j0 = if window > 0 { (i + 1).saturating_sub(window) } else { 0 };
        for hh in 0..h {
            let hk = hh / (h / hkv);
            let qi = &q[(i * h + hh) * d..][..d];
            let s: Vec<f64> = (j0..=i)
                .map(|j| qi.iter().zip(&k[(j * hkv + hk) * d..][..d]).map(|(a, b)| *a as f64 * *b as f64).sum())
                .collect();
            let m = s.iter().copied().fold(f64::MIN, f64::max);
            let e: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
            let z: f64 = e.iter().sum();
            for c in 0..d {
                let acc: f64 = (j0..=i).zip(&e).map(|(j, p)| p * v[(j * hkv + hk) * d + c] as f64).sum();
                out[(i * h + hh) * d + c] = (acc / z) as f32;
            }
        }
    }
    out
}

#[test]
#[ignore]
fn gqa_matches_vllm_attn_output() {
    let dir = std::env::var("ATTN_IO_DIR").unwrap_or_else(|_| "/vllm_attn_io".into());
    let t: usize = std::env::var("T").ok().and_then(|v| v.parse().ok()).unwrap_or(22);
    let (h, hkv, d, window) = (16usize, 8usize, 256usize, 1024usize);
    // vLLM q 是 fp8 值（存 f32）；k/v bf16。只取前 t 行（vLLM 可能 padding 到 24）。
    let q = &rd(&dir, "vllm_q")[..t * h * d];
    let k = &rd(&dir, "vllm_k")[..t * hkv * d];
    let v = &rd(&dir, "vllm_v")[..t * hkv * d];
    let want = &rd(&dir, "vllm_out")[..t * h * d];

    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let up = |x: &[f32]| st.htod_copy(x.iter().map(|&f| bf16::from_f32(f)).collect::<Vec<_>>()).unwrap();
    // 输入保持与 vLLM 一致：q 直接上传（vLLM 进 attn 时 q 已是 fp8 值）。
    // k/v：用我们**生产代码里真实的** device 端 fp8 往返算子（scale 1.0），复刻 vLLM 写 fp8 cache→读 dequant，
    // 不用 CPU 近似（之前的 f32_e4m3_rt 等于私自改输入，已删）。
    let dq = up(q);
    // 优先用 vLLM fp8 cache 里读回的 k/v（attention 实际用的那份字节）；没有再退回 bf16 + 我方 fp8 往返。
    let (dk, dv) = if std::path::Path::new(&format!("{dir}/vllm_kcache.bin")).exists() {
        let kc = &rd(&dir, "vllm_kcache")[..t * hkv * d];
        let vc = &rd(&dir, "vllm_vcache")[..t * hkv * d];
        // 退化检查：cache 读回的 k 应与 op 入参 bf16 k 接近（fp8 量化级），范数非零、不全等
        let (mut nk, mut nc, mut dmax) = (0f64, 0f64, 0f32);
        for i in 0..t * hkv * d { nk += (k[i] as f64).powi(2); nc += (kc[i] as f64).powi(2); dmax = dmax.max((k[i] - kc[i]).abs()); }
        println!("[输入] 用 vLLM fp8 cache 的 k/v（完全对齐）；退化检查 ‖k_bf16‖={:.2} ‖k_cache‖={:.2} max|d|={dmax:.3e}", nk.sqrt(), nc.sqrt());
        assert!(nc.sqrt() > 1.0, "k_cache 范数异常（可能全零/布局错）");
        (up(kc), up(vc))
    } else {
        let mut dk = up(k);
        let mut dv = up(v);
        dinfer_kernels::fp8_e4m3_roundtrip_bf16_n(&st, &mut dk, t * hkv * d, 1.0).unwrap();
        dinfer_kernels::fp8_e4m3_roundtrip_bf16_n(&st, &mut dv, t * hkv * d, 1.0).unwrap();
        println!("[输入] 用 bf16 k/v + 我方 fp8 往返（近似，无 cache dump 时）");
        (dk, dv)
    };
    let cu = st.htod_copy(vec![0i32, t as i32]).unwrap();

    let cfg: i32 = std::env::var("GQA_PREFILL_CFG").ok().and_then(|v| v.parse().ok()).unwrap_or(464);
    for &(p_fp8, tag) in &[(0i32, "p_fp8=0"), (1i32, "p_fp8=1")] {
        let mut out = st.alloc_zeros::<bf16>(t * h * d).unwrap();
        let rc = unsafe {
            ffi::gqa_prefill_win_bf16(
                st.ptr(&dq) as *const c_void, st.ptr(&dk) as *const c_void, st.ptr(&dv) as *const c_void,
                st.ptr_mut(&mut out) as *mut c_void, st.ptr(&cu) as *const i32,
                1, t as i64, h as i64, hkv as i64, d as i64, 1.0, 0, 0,
                cfg, window as i32, p_fp8, st.raw() as _,
            )
        };
        st.synchronize().unwrap();
        let got = st.dtoh_sync_copy(&out).unwrap();
        let (mut nd, mut mx, mut nrm_r, mut nrm_g, mut dot) = (0usize, 0f32, 0f64, 0f64, 0f64);
        for i in 0..t * h * d {
            let g = got[i].to_f32();
            let w = bf16::from_f32(want[i]).to_f32();
            if bf16::from_f32(g).to_bits() != bf16::from_f32(w).to_bits() { nd += 1; }
            mx = mx.max((g - w).abs());
            nrm_r += (w as f64) * (w as f64);
            nrm_g += (g as f64) * (g as f64);
            dot += ((g - w) as f64) * ((g - w) as f64);
        }
        let rel = dot.sqrt() / nrm_r.sqrt().max(1e-30);
        println!("gqa vs vLLM out [{tag}] rc={rc} 不同 {nd}/{} max|d|={mx:.4e} rel={rel:.3e} ‖want‖={:.2} ‖got‖={:.2}",
                 t * h * d, nrm_r.sqrt(), nrm_g.sqrt());
    }
    let _ = cpu_ref; // 留着备用，本测试不再绕 f64
}
