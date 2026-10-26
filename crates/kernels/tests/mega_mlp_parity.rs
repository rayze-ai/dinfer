//! dense MLP 支路 persistent megakernel 对拍：mega（一次 launch）vs 现有逐 op 路径。
//!
//! 判据：两条路径给同输入，输出 bf16 逐位或仅 fp8 级差（megakernel 的 fp8 量化/累加顺序与
//! `gemm_fp8_static_bf16`(cuBLASLt) 不同，允许 ~1e-3 相对差，但不能有系统性偏差）。
//!
//! 跑：`CUDA_ARCHS=120 cargo test --release -p dinfer-kernels --features sm120 --test mega_mlp_parity -- --ignored --nocapture`
use dinfer_kernels::{bf16, ffi, Device, Gemm};
use std::ffi::c_void;

const H: usize = 2816;
const I: usize = 2112;
fn m_val() -> usize { std::env::var("MEGA_M").ok().and_then(|v| v.parse().ok()).unwrap_or(4) }
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
  let gemm = Gemm::new(&st).unwrap();

  // ---- 输入与权重 ----
  let M = m_val();
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

  eprintln!("STEP1 before rmsnorm");
  let mut norm1 = st.alloc_zeros::<bf16>(M * H).unwrap();
  dinfer_kernels::ops::rmsnorm_bf16_n(&st, &dx, &dpre, &mut norm1, M, H, EPS).unwrap();
  st.synchronize().unwrap(); eprintln!("STEP2 rmsnorm ok");

  let mut gate = st.alloc_zeros::<bf16>(M * I).unwrap();
  let mut up = st.alloc_zeros::<bf16>(M * I).unwrap();
  let mut dn_out = st.alloc_zeros::<bf16>(M * H).unwrap();
  let mut act_bf = st.alloc_zeros::<bf16>(M * I).unwrap();
  let mut ref_out = st.alloc_zeros::<bf16>(M * H).unwrap();
  gemm.forward_fp8(&norm1, gate_as, &dgw, gate_ws, &mut gate, M, I, H).unwrap();
  st.synchronize().unwrap(); eprintln!("STEP3 gate ok");
  gemm.forward_fp8(&norm1, up_as, &duw, up_ws, &mut up, M, I, H).unwrap();
  st.synchronize().unwrap(); eprintln!("STEP4 up ok");
  dinfer_kernels::ops::gelu_tanh_mul_round2_bf16(&st, &gate, &up, &mut act_bf, M, I).unwrap();
  st.synchronize().unwrap(); eprintln!("STEP5 gelu ok");
  gemm.forward_fp8(&act_bf, dn_as, &ddw, dn_ws, &mut dn_out, M, H, I).unwrap();
  st.synchronize().unwrap(); eprintln!("STEP6 down ok");
  dinfer_kernels::ops::rmsnorm_bf16_n(&st, &dn_out, &dpost, &mut ref_out, M, H, EPS).unwrap();
  st.synchronize().unwrap(); eprintln!("STEP7 post ok (reference path done)");
  st.synchronize().unwrap();
  let refv = st.dtoh_sync_copy(&ref_out).unwrap();

  eprintln!("STEP8 megakernel start");
  // ---- megakernel 路径 ----
  let mut norm1_m = st.alloc_zeros::<bf16>(M * H).unwrap();
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
  let mut dep2 = st.alloc_zeros::<i32>(8).unwrap();
  let mut dep2b = st.alloc_zeros::<i32>(8).unwrap();
  let mut wc = st.alloc_zeros::<u64>(8).unwrap();
  let mut ec = st.alloc_zeros::<u64>(8).unwrap();

  let n_gu = I.div_ceil(128) as i32;
  let n_dn = H.div_ceil(128) as i32;
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
      st.ptr_mut(&mut dep2) as *mut i32,
      2,
      1,
      st.ptr_mut(&mut wc) as *mut u64,
      st.ptr_mut(&mut ec) as *mut u64,
      st.raw() as _,
    )
  };
  let se = st.synchronize();
  eprintln!("STEP9 after mega launch: sync={:?} rc={rc}", se.as_ref().err().map(|e| format!("{e}")));
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
  // 分阶段耗时（clock64 周期，所有 task 累加）：定位瓶颈
  let wcv = st.dtoh_sync_copy(&wc).unwrap();
  let ecv = st.dtoh_sync_copy(&ec).unwrap();
  let names = ["NORM","GATE","UP","ACT","DOWN","POST"];
  let tot: u64 = ecv[..6].iter().sum();
  println!("[分阶段 exec 周期]");
  for i in 0..6 {
    let pct = if tot>0 {ecv[i] as f64/tot as f64*100.0} else {0.0};
    println!("  {:5} exec={:>12} ({:5.1}%)  wait={:>12}", names[i], ecv[i], pct, wcv[i]);
  }

  // ---- 计时：mega（一次 launch）vs 逐 op 路径（6 次 launch） ----
  const ITERS: usize = 200;
  // 逐 op 路径
  st.synchronize().unwrap();
  let t0 = std::time::Instant::now();
  for _ in 0..ITERS {
    dinfer_kernels::ops::rmsnorm_bf16_n(&st, &dx, &dpre, &mut norm1, M, H, EPS).unwrap();
    gemm.forward_fp8(&norm1, gate_as, &dgw, gate_ws, &mut gate, M, I, H).unwrap();
    gemm.forward_fp8(&norm1, up_as, &duw, up_ws, &mut up, M, I, H).unwrap();
    dinfer_kernels::ops::gelu_tanh_mul_round2_bf16(&st, &gate, &up, &mut act_bf, M, I).unwrap();
    gemm.forward_fp8(&act_bf, dn_as, &ddw, dn_ws, &mut dn_out, M, H, I).unwrap();
    dinfer_kernels::ops::rmsnorm_bf16_n(&st, &dn_out, &dpost, &mut ref_out, M, H, EPS).unwrap();
  }
  st.synchronize().unwrap();
  let eager_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
  // mega 路径
  let mut rdy2 = st.alloc_zeros::<i32>(8).unwrap();
  st.synchronize().unwrap();
  let t1 = std::time::Instant::now();
  for _ in 0..ITERS {
    let rc = unsafe {
      ffi::mega_mlp_dense_branch(
        st.ptr(&dx) as *const c_void, st.ptr_mut(&mut out_m) as *mut c_void,
        st.ptr(&dpre) as *const c_void, st.ptr(&dpost) as *const c_void,
        st.ptr(&dgw) as *const c_void, gate_ws, 1.0 / gate_as,
        st.ptr(&duw) as *const c_void, up_ws, 1.0 / up_as,
        st.ptr(&ddw) as *const c_void, dn_ws, 1.0 / dn_as,
        st.ptr_mut(&mut norm1_m) as *mut c_void, st.ptr_mut(&mut norm1_q) as *mut c_void,
        st.ptr_mut(&mut gate_m) as *mut c_void, st.ptr_mut(&mut up_m) as *mut c_void,
        st.ptr_mut(&mut act_bf_m) as *mut c_void, st.ptr_mut(&mut act_q) as *mut c_void,
        st.ptr_mut(&mut dn_m) as *mut c_void,
        M as i32, H as i32, I as i32, EPS, n_gu, n_dn,
        st.ptr_mut(&mut dep) as *mut i32, st.ptr_mut(&mut cnt) as *mut i32,
        st.ptr_mut(&mut sta) as *mut i32, st.ptr_mut(&mut rdy2) as *mut i32,
        st.ptr_mut(&mut dep2b) as *mut i32, 2, 0, std::ptr::null_mut(), std::ptr::null_mut(), st.raw() as _,
      )
    };
    assert_eq!(rc, 0);
    // ready 计数器必须清零才能再次 launch
    rdy2 = st.alloc_zeros::<i32>(8).unwrap();
  }
  st.synchronize().unwrap();
  let mega_us = t1.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
  println!("[计时] B={M} 逐op路径={eager_us:.1}us  mega一次launch={mega_us:.1}us  speedup={:.2}x",
           eager_us / mega_us);
}
