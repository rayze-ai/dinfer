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
    // vLLM：k/v 写 fp8 cache 再 dequant 回 bf16（scale 1.0）→ 喂 kernel 前做 fp8 往返复刻
    let fp8rt = |x: &[f32]| x.iter().map(|&f| f32_e4m3_rt(bf16::from_f32(f).to_f32())).collect::<Vec<_>>();
    let kq = fp8rt(k);
    let vq = fp8rt(v);
    // q 已是 fp8 值，直接 bf16 上传（kernel 内用 bf16 读）
    let (dq, dk, dv) = (up(q), up(&kq), up(&vq));
    let cu = st.htod_copy(vec![0i32, t as i32]).unwrap();

    let cfg: i32 = std::env::var("GQA_PREFILL_CFG").ok().and_then(|v| v.parse().ok()).unwrap_or(-1);
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
}

fn f32_e4m3_rt(x: f32) -> f32 {
    // 复刻 __nv_cvt_float_to_fp8(E4M3, SATFINITE) 往返；用 half crate 的 f16 近似不对，
    // 这里走 CPU e4m3：饱和 ±448、就近偶数。简化版（够做末位级对拍）。
    let a = x.abs();
    if a == 0.0 { return 0.0; }
    if a >= 448.0 { return x.signum() * 448.0; }
    let e = a.log2().floor();
    let step = 2f32.powf(e - 3.0); // 3 尾数位
    let q = (a / step).round() * step;
    x.signum() * q
}
