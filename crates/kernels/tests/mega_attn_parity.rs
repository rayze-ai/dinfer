//! B1（kaon3-mega R36）：persistent 版 paged GQA decode attention 对拍 + 计时。
//!
//! 参考路径 = 现有 `gqa_decode_v3_bf16`（grid launch）+ merge。
//! mega 路径  = `mega_attn_decode`（persistent 一次 launch）+ 同一 merge。
//! 判据：**逐位一致 0/N**（走同一个 core device 函数、同样的 split 边界与 merge → 应逐位）。
//! 先退化检查（两边范数非零且同量级）。再计时 persistent 一次 launch vs v3 grid launch 的 speedup。
//!
//! 跑：`CUDA_ARCHS=120 cargo test --release -p dinfer-kernels --test mega_attn_parity -- --ignored --nocapture`
use dinfer_kernels::{bf16, ffi, Device};
use std::ffi::c_void;
use std::time::Instant;

fn rand_bf16(n: usize, seed: u64) -> Vec<bf16> {
  let mut s = seed | 1;
  (0..n)
    .map(|_| {
      s ^= s << 13;
      s ^= s >> 7;
      s ^= s << 17;
      bf16::from_f32(((s % 2000) as f32 / 1000.0) - 1.0)
    })
    .collect()
}

/// 一个 (b, h, hk, d, n) 配置跑一轮对拍 + 计时。
fn run_case(b: usize, h: usize, hk: usize, d: usize, n: usize) {
  let group = h / hk;
  let dev = Device::new(0).unwrap();
  dev.disable_event_tracking();
  let st = dev.new_stream().unwrap();
  let bs = 16usize; // block_size（D=256 要求偶数；16 满足）
  let blocks_per = n.div_ceil(bs);
  let max_blocks = blocks_per;

  // paged KV cache：[slot][Hkv][2][D]，slot = blk*bs + off
  let cache = rand_bf16(b * blocks_per * bs * hk * 2 * d, 0x1234 + (d as u64) + (n as u64) * 7);
  let dcache = st.htod_copy(cache).unwrap();
  // 乱序 block table（模拟真实分配）
  let mut ids: Vec<i32> = (0..(b * blocks_per) as i32).collect();
  let mut x = 0x9e3779b9u32;
  for i in (1..ids.len()).rev() {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    ids.swap(i, (x as usize) % (i + 1));
  }
  let dbt = st.htod_copy(ids).unwrap();
  let seq = st.htod_copy(vec![n as i32; b]).unwrap();
  let dcu = st.htod_copy((0..=b as i32).collect::<Vec<_>>()).unwrap();
  let q = rand_bf16(b * h * d, 0x5678 + (d as u64) + (n as u64) * 13);
  let dq = st.htod_copy(q).unwrap();
  let scale = 1.0 / (d as f32).sqrt();

  // split 规划（照 gqa_decode_bf16 的口径，固定值便于两路一致；不写死进 kernel，这里是 host 选的）
  let keys_per_split = 32i64;
  let target_blocks = 1024i64;
  let mut splits = (target_blocks + (b * hk) as i64 - 1) / (b * hk) as i64;
  splits = splits.clamp(1, 32);
  let scratch_len = dinfer_kernels::gqa_decode_scratch_len(b, h, d);

  // ---- 参考：v3 grid launch + merge ----
  let mut scratch_ref = st.alloc_zeros::<f32>(scratch_len).unwrap();
  let mut out_ref = st.alloc_zeros::<bf16>(b * h * d).unwrap();
  let direct_ref = std::ptr::null_mut::<c_void>(); // splits>1 → 走 scratch + merge
  let rc = unsafe {
    ffi::gqa_decode_v3_bf16(
      st.ptr(&dq) as *const c_void,
      st.ptr(&dcache) as *const c_void,
      st.ptr(&dbt) as *const i32,
      st.ptr(&seq) as *const i32,
      st.ptr(&dcu) as *const i32,
      st.ptr_mut(&mut scratch_ref) as *mut f32,
      direct_ref,
      b as i64,
      h as i64,
      hk as i64,
      d as i64,
      bs as i64,
      max_blocks as i64,
      scale,
      0,
      splits,
      keys_per_split,
      st.raw() as _,
    )
  };
  assert_eq!(rc, 0, "v3 返回 {rc}");
  let rc = unsafe {
    ffi::gqa_decode_merge_v2_launch(
      st.ptr(&scratch_ref) as *const f32,
      st.ptr(&dcu) as *const i32,
      st.ptr(&seq) as *const i32,
      st.ptr_mut(&mut out_ref) as *mut c_void,
      b as i32,
      h as i32,
      d as i32,
      splits as i32,
      keys_per_split as i32,
      st.raw() as _,
    )
  };
  assert_eq!(rc, 0, "ref merge 返回 {rc}");
  st.synchronize().unwrap();
  let refv = st.dtoh_sync_copy(&out_ref).unwrap();
  let scratch_refv = st.dtoh_sync_copy(&scratch_ref).unwrap();

  // ---- mega：persistent 一次 launch + 同一 merge ----
  let mut scratch_m = st.alloc_zeros::<f32>(scratch_len).unwrap();
  let mut out_m = st.alloc_zeros::<bf16>(b * h * d).unwrap();
  let mut dep = st.alloc_zeros::<i32>(1).unwrap();
  let mut cnt = st.alloc_zeros::<i32>(1).unwrap();
  let mut sta = st.alloc_zeros::<i32>(1).unwrap();
  let mut rdy = st.alloc_zeros::<i32>(1).unwrap();
  let mut wc = st.alloc_zeros::<u64>(1).unwrap();
  let mut ec = st.alloc_zeros::<u64>(1).unwrap();
  let rc = unsafe {
    ffi::mega_attn_decode(
      st.ptr(&dq) as *const c_void,
      st.ptr(&dcache) as *const c_void,
      st.ptr(&dbt) as *const i32,
      st.ptr(&seq) as *const i32,
      st.ptr(&dcu) as *const i32,
      st.ptr_mut(&mut scratch_m) as *mut f32,
      std::ptr::null_mut::<c_void>(),
      b as i32,
      h as i32,
      hk as i32,
      d as i32,
      bs as i32,
      max_blocks as i32,
      scale,
      0,
      splits as i32,
      keys_per_split as i32,
      st.ptr_mut(&mut dep) as *mut i32,
      st.ptr_mut(&mut cnt) as *mut i32,
      st.ptr_mut(&mut sta) as *mut i32,
      st.ptr_mut(&mut rdy) as *mut i32,
      st.ptr_mut(&mut wc) as *mut u64,
      st.ptr_mut(&mut ec) as *mut u64,
      1,
      st.raw() as _,
    )
  };
  assert_eq!(rc, 0, "mega_attn 返回 {rc}");
  let rc = unsafe {
    ffi::gqa_decode_merge_v2_launch(
      st.ptr(&scratch_m) as *const f32,
      st.ptr(&dcu) as *const i32,
      st.ptr(&seq) as *const i32,
      st.ptr_mut(&mut out_m) as *mut c_void,
      b as i32,
      h as i32,
      d as i32,
      splits as i32,
      keys_per_split as i32,
      st.raw() as _,
    )
  };
  assert_eq!(rc, 0, "mega merge 返回 {rc}");
  st.synchronize().unwrap();
  let megav = st.dtoh_sync_copy(&out_m).unwrap();
  let scratch_mv = st.dtoh_sync_copy(&scratch_m).unwrap();

  // ---- 退化检查 + 对比 ----
  // 退化检查：两边范数都非零（非全零比全零），且 ref≈mega 同量级（长序列 softmax 近均匀 → 输出范数本就小）。
  let nr: f64 = refv.iter().map(|v| (v.to_f32() as f64).powi(2)).sum::<f64>().sqrt();
  let nm: f64 = megav.iter().map(|v| (v.to_f32() as f64).powi(2)).sum::<f64>().sqrt();
  assert!(nr > 1e-3, "[{b}x{h}x{hk} d={d} n={n}] 参考输出范数异常 {nr}（可能全零）");
  assert!(nm > 1e-3, "[{b}x{h}x{hk} d={d} n={n}] mega 输出范数异常 {nm}（可能全零）");
  assert!((nr - nm).abs() / nr.max(1e-30) < 0.5, "[{b}x{h}x{hk} d={d} n={n}] ref/mega 范数不同量级 {nr} vs {nm}");

  // scratch（attention 直出 partial）逐位
  let mut sdiff = 0usize;
  for i in 0..scratch_len {
    if scratch_refv[i].to_bits() != scratch_mv[i].to_bits() {
      sdiff += 1;
    }
  }
  // 最终输出逐位
  let mut odiff = 0usize;
  let mut mx = 0f32;
  for i in 0..b * h * d {
    if refv[i].to_bits() != megav[i].to_bits() {
      odiff += 1;
    }
    mx = mx.max((refv[i].to_f32() - megav[i].to_f32()).abs());
  }
  println!(
    "[{b}x{h}/{hk} d={d} n={n} g={group} splits={splits}] ‖ref‖={nr:.3} ‖mega‖={nm:.3}  \
     scratch_diff={sdiff}/{scratch_len}  out_diff={odiff}/{}  max|d|={mx:.3e}  → {}",
    b * h * d,
    if sdiff == 0 && odiff == 0 { "逐位一致" } else { "不一致!" }
  );
  assert_eq!(sdiff, 0, "scratch 非逐位（attention stage 抽函数/索引错）");
  assert_eq!(odiff, 0, "输出非逐位");

  // ---- 计时：persistent 一次 launch vs v3 grid launch ----
  let mut f_v3 = || unsafe {
    ffi::gqa_decode_v3_bf16(
      st.ptr(&dq) as *const c_void, st.ptr(&dcache) as *const c_void, st.ptr(&dbt) as *const i32,
      st.ptr(&seq) as *const i32, st.ptr(&dcu) as *const i32, st.ptr_mut(&mut scratch_ref) as *mut f32,
      std::ptr::null_mut::<c_void>(), b as i64, h as i64, hk as i64, d as i64, bs as i64, max_blocks as i64,
      scale, 0, splits, keys_per_split, st.raw() as _,
    );
  };
  let mut f_mega = || unsafe {
    ffi::mega_attn_decode(
      st.ptr(&dq) as *const c_void, st.ptr(&dcache) as *const c_void, st.ptr(&dbt) as *const i32,
      st.ptr(&seq) as *const i32, st.ptr(&dcu) as *const i32, st.ptr_mut(&mut scratch_m) as *mut f32,
      std::ptr::null_mut::<c_void>(), b as i32, h as i32, hk as i32, d as i32, bs as i32, max_blocks as i32,
      scale, 0, splits as i32, keys_per_split as i32, st.ptr_mut(&mut dep) as *mut i32,
      st.ptr_mut(&mut cnt) as *mut i32, st.ptr_mut(&mut sta) as *mut i32, st.ptr_mut(&mut rdy) as *mut i32,
      st.ptr_mut(&mut wc) as *mut u64, st.ptr_mut(&mut ec) as *mut u64, 0, st.raw() as _,
    );
  };
  let time = |f: &mut dyn FnMut()| {
    for _ in 0..10 {
      f();
    }
    st.synchronize().unwrap();
    let t = Instant::now();
    let iters = 300;
    for _ in 0..iters {
      f();
    }
    st.synchronize().unwrap();
    t.elapsed().as_secs_f64() * 1e6 / iters as f64
  };
  let t_v3 = time(&mut f_v3);
  let t_mega = time(&mut f_mega);
  println!(
    "[{b}x{h}/{hk} d={d} n={n}] v3 grid={t_v3:.2}us  persistent={t_mega:.2}us  speedup={:.2}x",
    t_v3 / t_mega
  );
}

#[test]
#[ignore]
fn mega_attn_matches_v3() {
  // Qwen3.5 两种层：滑窗（D=256, 16q/8kv → g=2）与全注意力（D=512 不支持→用 D=256 2q/… 近似）。
  // v3 支持 D∈{128,256}；这里覆盖 D=256 的 g=1/2/4/6 与 D=128 的 g=1/2/4，B=1 decode。
  let cases = [
    (1usize, 16usize, 8usize, 256usize, 1024usize), // 滑窗层形状
    (1, 16, 8, 256, 4096),
    (1, 16, 4, 256, 2048), // g=4
    (1, 12, 2, 256, 2048), // g=6
    (1, 16, 16, 256, 1024), // g=1
    (1, 16, 8, 128, 1024),  // D=128 g=2
    (1, 16, 4, 128, 2048),  // D=128 g=4
    (1, 16, 16, 128, 1024), // D=128 g=1
  ];
  for &(b, h, hk, d, n) in &cases {
    run_case(b, h, hk, d, n);
  }
}
