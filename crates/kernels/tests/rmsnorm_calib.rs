//! RMSNorm 的位级标定：找出与 vLLM（`VLLM_BATCH_INVARIANT=1` → Triton `_rms_norm_kernel`）逐位一致的
//! **平方和归约顺序 × inv 算法**组合。
//!
//! 背景（`test/kaon3/2026-10-23/D-16-REBASE.md` §52）：Kaon-V3 L2 隔离里 `dense` 逐位相同，`ff1 = RMSNorm₁(dense)`
//! 却差 1 元素；`post_ff_norm` 在输入相同的 token 8 上也差 1 元素。差在 norm kernel 自身。
//!
//! 变体（NVRTC 现编，sm 取本卡）：
//! - SS 0 = 我们 `rmsnorm_vec_kernel<1024,1>`（每线程 8 连续元素 `ss += a·a + b·b` → 32 warp 两级蝶形）；
//!   SS 1/2 = Triton（BLOCK_SIZE=1024、4 warp × 32 lane × 8 元素；线程内顺序累加，2 = fma 收缩；
//!   warp 蝶形 → 4 warp 蝶形 → 块间 `sum_sq += chunk` 顺序累加）。
//! - INV 0 = `rsqrtf`；1 = `1/sqrt` 全 IEEE；2 = `sqrt.approx` + IEEE 除；3 = `sqrt.approx` + `div.full`。
//!
//! 数据目录（`RMS_CASES` 逗号分隔）每个含 `x_f32.bin`（[rows,h]，bf16 精确值）、`w_bf16.bin`（[h]）、`ref_f32.bin`。
//! 跑：`RMS_CASES=/w/d17b/rms_ff1,/w/d17b/rms_pff RMS_EPS=1e-6 cargo test --release -p dinfer-kernels \
//!        --test rmsnorm_calib -- --ignored --nocapture`

use dinfer_kernels::cudarc::{self, driver::{LaunchConfig, PushKernelArg}};
use dinfer_kernels::{bf16, ops, Device};

const SRC: &str = r#"
__device__ __forceinline__ float b2f(unsigned short b) { return __uint_as_float(((unsigned)b) << 16); }
__device__ __forceinline__ unsigned short f2b(float f) {
  unsigned u = __float_as_uint(f); return (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float sqrt_approx(float x) { float r; asm("sqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(x)); return r; }
__device__ __forceinline__ float bfly(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ float inv_of(float ss, int n, float eps, int inv_mode) {
  float mean = __fdiv_rn(ss, (float)n);
  float t = mean + eps;
  if (inv_mode == 0) return rsqrtf(t);
  if (inv_mode == 1) return __fdiv_rn(1.0f, __fsqrt_rn(t));
  if (inv_mode == 2) return __fdiv_rn(1.0f, sqrt_approx(t));
  return __fdividef(1.0f, sqrt_approx(t));
}
// ours：1024 线程，线程 t 持 8t..8t+7
extern "C" __global__ void rms_ours(const unsigned short* x, const unsigned short* w, unsigned short* y,
                                    int h, float eps, int inv_mode) {
  __shared__ float sm[32];
  __shared__ float s_inv;
  const unsigned short* xr = x + (long)blockIdx.x * h;
  float ss = 0.f;
  int t = threadIdx.x;
  if (8 * t < h) {
    for (int c = 0; c < 4; ++c) { float a = b2f(xr[8*t+2*c]), b = b2f(xr[8*t+2*c+1]); ss += a * a + b * b; }
  }
  ss = bfly(ss);
  if ((t & 31) == 0) sm[t >> 5] = ss;
  __syncthreads();
  if (t < 32) { float r = sm[t]; r = bfly(r); if (t == 0) s_inv = inv_of(r, h, eps, inv_mode); }
  __syncthreads();
  for (int i = t; i < h; i += blockDim.x) y[(long)blockIdx.x * h + i] = f2b(b2f(xr[i]) * s_inv * b2f(w[i]));
}
// triton：128 线程（4 warp），每 1024 列一块
extern "C" __global__ void rms_triton(const unsigned short* x, const unsigned short* w, unsigned short* y,
                                      int h, float eps, int inv_mode, int use_fma) {
  __shared__ float sm[4];
  __shared__ float s_inv;
  const unsigned short* xr = x + (long)blockIdx.x * h;
  int t = threadIdx.x;
  float sum_sq = 0.f;
  for (int off = 0; off < h; off += 1024) {
    float acc = 0.f;
    for (int e = 0; e < 8; ++e) {
      int col = off + 8 * t + e;
      float v = col < h ? b2f(xr[col]) : 0.f;
      float sq = __fmul_rn(v, v);
      if (e == 0) acc = sq;
      else acc = use_fma ? __fmaf_rn(v, v, acc) : __fadd_rn(acc, sq);
    }
    acc = bfly(acc);
    if ((t & 31) == 0) sm[t >> 5] = acc;
    __syncthreads();
    float r = (t < 4) ? sm[t] : 0.f;
    if (t < 32) {
      r += __shfl_xor_sync(0xffffffffu, r, 2);
      r += __shfl_xor_sync(0xffffffffu, r, 1);
    }
    if (t == 0) sum_sq = __fadd_rn(sum_sq, r);
    __syncthreads();
  }
  if (t == 0) s_inv = inv_of(sum_sq, h, eps, inv_mode);
  __syncthreads();
  for (int i = t; i < h; i += blockDim.x) y[(long)blockIdx.x * h + i] = f2b(__fmul_rn(__fmul_rn(b2f(xr[i]), s_inv), b2f(w[i])));
}
"#;

fn read_f32(p: &str) -> Vec<f32> {
    std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}")).chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[test]
#[ignore = "需要 GPU + 标定数据"]
fn calibrate_rmsnorm() {
    let cases = std::env::var("RMS_CASES").expect("RMS_CASES");
    let eps: f32 = std::env::var("RMS_EPS").ok().and_then(|v| v.parse().ok()).unwrap_or(1e-6);
    let dev = Device::new(0).expect("cuda");
    let st = dev.default_stream();
    let cs = st.inner_arc();
    let (maj, min) = dev.compute_capability().expect("cc");
    let opts = cudarc::nvrtc::CompileOptions { arch: Some(Box::leak(format!("sm_{maj}{min}").into_boxed_str())), ..Default::default() };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(SRC, opts).expect("nvrtc");
    let module = cs.context().load_module(ptx).expect("module");
    let f_ours = module.load_function("rms_ours").expect("rms_ours");
    let f_tri = module.load_function("rms_triton").expect("rms_triton");

    let mut prod_bad: Vec<(String, usize)> = Vec::new();
    for dir in cases.split(',') {
        let x = read_f32(&format!("{dir}/x_f32.bin"));
        let rf = read_f32(&format!("{dir}/ref_f32.bin"));
        let wb: Vec<u16> = std::fs::read(format!("{dir}/w_bf16.bin")).expect("w").chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let h = wb.len();
        let rows = x.len() / h;
        assert_eq!(rows * h, x.len());
        let xb: Vec<u16> = x.iter().map(|v| bf16::from_f32(*v).to_bits()).collect();
        let dx = st.htod_copy(xb.clone()).expect("x");
        let dw = st.htod_copy(wb.clone()).expect("w");
        let diff = |y: &[u16]| -> (usize, Vec<(usize, usize)>) {
            let d: Vec<(usize, usize)> = (0..rows * h).filter(|&i| bf16::from_bits(y[i]).to_f32() != rf[i]).map(|i| (i / h, i % h)).collect();
            (d.len(), d.into_iter().take(4).collect())
        };
        println!("== {dir}  rows={rows} h={h}");
        // 生产 kernel（ops::rmsnorm_bf16_n）
        {
            let xs: Vec<bf16> = xb.iter().map(|b| bf16::from_bits(*b)).collect();
            let ws: Vec<bf16> = wb.iter().map(|b| bf16::from_bits(*b)).collect();
            let (ax, aw) = (st.htod_copy(xs).expect("x"), st.htod_copy(ws).expect("w"));
            let mut o = st.alloc_zeros::<bf16>(rows * h).expect("o");
            ops::rmsnorm_bf16_n(&st, &ax, &aw, &mut o, rows, h, eps).expect("rmsnorm");
            let y: Vec<u16> = st.dtoh_sync_copy(&o).expect("dl").iter().map(|v| v.to_bits()).collect();
            let (n, ex) = diff(&y);
            println!("  production          diff {n:<5} {ex:?}");
            prod_bad.push((dir.to_string(), n));
        }
        for inv in 0..4i32 {
            let mut o = st.alloc_zeros::<u16>(rows * h).expect("o");
            let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (1024, 1, 1), shared_mem_bytes: 0 };
            let hi = h as i32;
            unsafe { cs.launch_builder(&f_ours).arg(&dx).arg(&dw).arg(&mut o).arg(&hi).arg(&eps).arg(&inv).launch(cfg) }.expect("launch");
            let (n, ex) = diff(&st.dtoh_sync_copy(&o).expect("dl"));
            println!("  ours   ss  inv={inv}    diff {n:<5} {ex:?}");
            for fma in 0..2i32 {
                let mut o = st.alloc_zeros::<u16>(rows * h).expect("o");
                let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
                unsafe { cs.launch_builder(&f_tri).arg(&dx).arg(&dw).arg(&mut o).arg(&hi).arg(&eps).arg(&inv).arg(&fma).launch(cfg) }.expect("launch");
                let (n, ex) = diff(&st.dtoh_sync_copy(&o).expect("dl"));
                println!("  triton fma={fma} inv={inv}  diff {n:<5} {ex:?}");
            }
        }
    }
    println!("DONE");
    // 守护放最后：先让所有变体打印完，便于定位。
    for (d, n) in &prod_bad {
        assert_eq!(*n, 0, "生产 rmsnorm 与 vLLM 不逐位一致：{d}");
    }
}
