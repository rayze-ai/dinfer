//! dense MLP 支路 persistent megakernel 对拍：mega（一次 launch）vs 现有逐 op 路径。
//!
//! 判据：两条路径给同输入，输出 bf16 逐位或仅 fp8 级差（megakernel 的 fp8 量化/累加顺序与
//! `gemm_fp8_static_bf16`(cuBLASLt) 不同，允许 ~1e-3 相对差，但不能有系统性偏差）。
//!
//! 跑：`CUDA_ARCHS=120 cargo test --release -p dinfer-kernels --features sm120 --test mega_mlp_parity -- --ignored --nocapture`
use dinfer_kernels::{bf16, ffi, Device};
use std::ffi::c_void;

const H: usize = 2816;
const I: usize = 2112;
const M: usize = 4;  // decode batch
const EPS: f32 = 1e-6;

fn rand(n: usize, seed: u64) -> Vec<f32> {
  let mut s = seed | 1;
  (0..n)
    .map(|_| {
      s ^= s << 13;
      s ^= s >> 7;
      s ^= s << 17;
      ((s % 2000) as f32 / 1000.0) - 1.0
    })
    .collect()
}

/// bf16 → e4m3（饱和 ±448 + RNE），与 kernel cvt16 同口径
fn q_fp8(x: &[f32], inv: f32) -> Vec<u8> {
  x.iter()
    .map(|&v| {
      let y = (v * inv).clamp(-448.0, 448.0);
      // 用 CPU 参考：f32 → e4m3 位模式（简化：与 GPU __nv_cvt_float_to_fp8 一致的就近偶数）
      f32_to_e4m3(y)
    })
    .collect()
}

fn f32_to_e4m3(y: f32) -> u8 {
  // 与 weights.rs 的 f32_to_e4m3 同实现（RNE + 饱和）
  let s = if y < 0.0 { 0x80u8 } else { 0 };
  let a = y.abs();
  if a >= 448.0 {
    return s | 0x7E;
  }
  if a == 0.0 {
    return s;
  }
  let e = a.log2().floor() as i32;
  if e < -6 {
    // 次正规
    let m = (a / (1.0f32 / 8.0) * 2f32.powi(-6)).round_ties_even() as i32;
    return s | (m as u8).min(7);
  }
  let e = e.clamp(-6, 8);
  let mut m = ((a / 2f32.powi(e) - 1.0) * 8.0).round_ties_even() as i32;
  if m >= 8 {
    m = 0;
  } else {
    m = m.clamp(0, 7);
  }
  s | (((e + 7) as u8) << 3) | (m as u8 & 7)
}

#[test]
#[ignore]
fn mega_mlp_matches_eager_path() {
  let dev = Device::new(0).unwrap();
  let st = dev.default_stream();

  // ---- 输入与权重 ----
  let x = rand(M * H, 11);
  let pre_w = rand(H, 21);
  let post_w = rand(H, 31);
  let gate_w = rand(I * H, 41);
  let up_w = rand(I * H, 51);
  let dn_w = rand(H * I, 61);
  // 模拟真实 scale
  let (gate_ws, gate_as) = (1.2e-3f32, 0.59f32);
  let (up_ws, up_as) = (1.1e-3f32, 0.59f32);
  let (dn_ws, dn_as) = (1.7e-4f32, 0.59f32);

  // 权重按 fp8 量化（模拟 checkpoint 里的 fp8 权重）——两端用同一份字节
  let gw_q = q_fp8(&gate_w, 1.0 / gate_ws);
  let uw_q = q_fp8(&up_w, 1.0 / up_ws);
  let dw_q = q_fp8(&dn_w, 1.0 / dn_ws);

  // ---- 参考路径（现有逐 op，用 cuBLASLt fp8 GEMM） ----
  let dx = st.htod_copy(x.iter().map(|&f| bf16::from_f32(f)).collect::<Vec<_>>()).unwrap();
  let dpre = st.htod_copy(pre_w.iter().map(|&f| bf16::from_f32(f)).collect::<Vec<_>>()).unwrap();
  let dpost = st.htod_copy(post_w.iter().map(|&f| bf16::from_f32(f)).collect::<Vec<_>>()).unwrap();
  let dgw = st.htod_copy(gw_q.clone()).unwrap();
  let duw = st.htod_copy(uw_q.clone()).unwrap();
  let ddw = st.htod_copy(dw_q.clone()).unwrap();

  let mut norm1 = st.alloc_zeros::<bf16>(M * H).unwrap();
  dinfer_kernels::ops::rmsnorm_bf16_n(&st, &dx, &dpre, &mut norm1, M, H, EPS).unwrap();

  let mut gate = st.alloc_zeros::<bf16>(M * I).unwrap();
  let mut up = st.alloc_zeros::<bf16>(M * I).unwrap();
  let mut dn_out = st.alloc_zeros::<bf16>(M * H).unwrap();
  let mut act_bf = st.alloc_zeros::<bf16>(M * I).unwrap();
  let mut ref_out = st.alloc_zeros::<bf16>(M * H).unwrap();
  dinfer_kernels::ops::gemm_fp8_static_bf16(&st, &norm1, gate_as, &dgw, gate_ws, &mut gate, M, I, H).unwrap();
  dinfer_kernels::ops::gemm_fp8_static_bf16(&st, &norm1, up_as, &duw, up_ws, &mut up, M, I, H).unwrap();
  dinfer_kernels::ops::gelu_tanh_mul_round2_bf16(&st, &gate, &up, &mut act_bf, M, I).unwrap();
  dinfer_kernels::ops::gemm_fp8_static_bf16(&st, &act_bf, dn_as, &ddw, dn_ws, &mut dn_out, M, H, I).unwrap();
  dinfer_kernels::ops::rmsnorm_bf16_n(&st, &dn_out, &dpost, &mut ref_out, M, H, EPS).unwrap();
  st.synchronize().unwrap();
  let refv = st.dtoh_sync_copy(&ref_out).unwrap();

  // ---- megakernel 路径 ----
  let norm1_m = st.alloc_zeros::<bf16>(M * H).unwrap();
  let mut norm1_q = st.alloc_zeros::<u8>(M * H).unwrap();
  let mut gate_m = st.alloc_zeros::<bf16>(M * I).unwrap();
  let mut up_m = st.alloc_zeros::<bf16>(M * I).unwrap();
  let mut act_bf_m = st.alloc_zeros::<bf16>(M * I).unwrap();
  let mut act_q = st.alloc_zeros::<u8>(M * I).unwrap();
  let mut dn_m = st.alloc_zeros::<bf16>(M * H).unwrap();
  let mut out_m = st.alloc_zeros::<bf16>(M * H).unwrap();
  let mut dep = st.alloc_zeros::<i32>(8).unwrap();
  let mut cnt = st.alloc_zeros::<i32>(8).unwrap();
  let mut sta = st.alloc_zeros::<i32>(8).unwrap();
  let mut rdy = st.alloc_zeros::<i32>(8).unwrap();

  let n_gu = I.div_ceil(64) as i32;
  let n_dn = H.div_ceil(64) as i32;
  let rc = unsafe {
    ffi::mega_mlp_dense_branch(
      st.ptr(&dx) as *const c_void,
      st.ptr_mut(&mut out_m) as *mut c_void,
      st.ptr(&dpre) as *const c_void,
      st.ptr(&dpost) as *const c_void,
      st.ptr(&dgw) as *const c_void,
      gate_ws,
      1.0 / gate_as,
      st.ptr(&duw) as *const c_void,
      up_ws,
      1.0 / up_as,
      st.ptr(&ddw) as *const c_void,
      dn_ws,
      1.0 / dn_as,
      st.ptr_mut(&mut norm1_m) as *mut c_void,
      st.ptr_mut(&mut norm1_q) as *mut c_void,
      st.ptr_mut(&mut gate_m) as *mut c_void,
      st.ptr_mut(&mut up_m) as *mut c_void,
      st.ptr_mut(&mut act_bf_m) as *mut c_void,
      st.ptr_mut(&mut act_q) as *mut c_void,
      st.ptr_mut(&mut dn_m) as *mut c_void,
      M as i32,
      H as i32,
      I as i32,
      EPS,
      n_gu,
      n_dn,
      st.ptr_mut(&mut dep) as *mut i32,
      st.ptr_mut(&mut cnt) as *mut i32,
      st.ptr_mut(&mut sta) as *mut i32,
      st.ptr_mut(&mut rdy) as *mut i32,
      2,
      0,
      st.raw() as _,
    )
  };
  st.synchronize().unwrap();
  println!("mega rc={rc}（0=成功；801=本 arch 未编入）");
  assert_eq!(rc, 0, "megakernel 返回 {rc}");

  let megav = st.dtoh_sync_copy(&out_m).unwrap();

  // ---- 比较（退化检查优先）----
  let (mut nr, mut nm, mut nd, mut mx, mut dot) = (0f64, 0f64, 0usize, 0f32, 0f64);
  for i in 0..M * H {
    let r = refv[i].to_f32();
    let g = megav[i].to_f32();
    nr += (r as f64) * (r as f64);
    nm += (g as f64) * (g as f64);
    if bf16::from_f32(r).to_bits() != bf16::from_f32(g).to_bits() {
      nd += 1;
    }
    mx = mx.max((r - g).abs());
    dot += ((r - g) as f64) * ((r - g) as f64);
  }
  let rel = dot.sqrt() / nr.sqrt().max(1e-30);
  println!("[退化检查] ‖ref‖={:.4} ‖mega‖={:.4}（应同量级非零）", nr.sqrt(), nm.sqrt());
  assert!(nr.sqrt() > 1.0, "参考范数异常（可能全零）");
  assert!(nm.sqrt() > 1.0, "mega 范数异常（可能全零）");
  println!("[对比] 不同 {nd}/{} max|d|={mx:.4e} rel={rel:.3e}", M * H);
  println!("VERDICT: {}", if nd == 0 { "逐位一致" } else if rel < 5e-3 { "fp8 级差（可接受）" } else { "有系统性偏差" });
}
