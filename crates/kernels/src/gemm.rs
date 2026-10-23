//! bf16 GEMM：`C[M,N] = A[M,K] · W[N,K]^T (+ bias[N])`，fp32 累加。
//!
//! 两条实现路径，按 (M, N, K, 有无 bias) 查自调优表选：
//! 1. cuBLASLt：默认每次调用让启发式选 algo（与 cudarc `CudaBlasLT::matmul` 等价）；自调优后用表里实测最快的 algo。
//! 2. 自写小 M kernel（`cuda/gemm_small_m.cu`，M ≤ 32，纯访存型，split-K 份数由表给）。
//!
//! 表由 `Gemm::autotune(entries)` 在模型加载后、CUDA graph 捕获前填一次（每个形状 × M 桶，各候选实测几次取最快）；
//! 没调优过的形状走 cuBLASLt 默认。`GEMM_TUNE=off` 关掉自调优（全部走默认，回到旧行为）。
//! 数值参数（M 桶、候选数、split-K 候选）来自 `ArchProfile`，kernel 内不写死（CLAUDE.md 硬性规则）。
//!
//! 权重按 PyTorch `Linear` 的 `[out_features, in_features]`（即 `[N,K]`）行主序存，激活 `[M,K]` 行主序。
//! cuBLAS 是列主序，等价于算 `C^T[N,M] = W^T_colmajor[N,K] · A_colmajor[K,M]`：A 行主序 `[M,K]` 就是列主序 `[K,M]`（ldb=K），
//! W 行主序 `[N,K]` 就是列主序 `[K,N]`（lda=K），所以 transa=T（对 W）、transb=N（对 A），输出列主序 `[N,M]`（ldc=N）即行主序 `[M,N]`。

use crate::arch::ArchProfile;
use crate::device::{Stream, TimingEvent};
use crate::ops;
use anyhow::{anyhow, bail, ensure, Result};
use cudarc::cublaslt::{result as lt, sys};
use cudarc::driver::CudaSlice;
use half::bf16;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Key {
    m: usize,
    n: usize,
    k: usize,
    bias: bool,
    /// 数据类型：bf16 与 fp8 两条路径的 algo 选择规则不同（fp8 要按实测配置表选，
    /// 见 `forward_fp8_cublaslt`），缓存键必须区分，否则 bf16 先跑过同形状会让 fp8 复用错的 algo。
    fp8: bool,
}

impl Key {
    const fn bf16(m: usize, n: usize, k: usize, bias: bool) -> Self {
        Self { m, n, k, bias, fp8: false }
    }
    const fn fp8(m: usize, n: usize, k: usize) -> Self {
        Self { m, n, k, bias: false, fp8: true }
    }
}

/// 某个 (M, N, K) 形状选定的执行方式。
#[derive(Clone, Copy, Debug)]
pub enum GemmPlan {
    /// cuBLASLt，指定 algo（自调优实测最快）。
    LtAlgo(sys::cublasLtMatmulAlgo_t),
    /// 自写小 M kernel，split-K 份数。
    SmallM { splits: usize },
    /// 自写 tensor-core 小 M kernel（r11b）：split-K 份数 + 每 block warp 数（归约在 kernel 内）。
    SmallMma { splits: usize, warps: usize, stripes: usize },
}

/// 自调优输入：一个形状（N, K）和它的真实权重（计时用真权重，避免再分配一份）。
pub struct GemmTuneEntry<'a> {
    pub name: String,
    pub n: usize,
    pub k: usize,
    pub w: &'a CudaSlice<bf16>,
    pub bias: Option<&'a CudaSlice<bf16>>,
}

/// 一个 (形状, M 桶) 的实测结果（µs）。
#[derive(Clone, Debug)]
pub struct GemmTuneRow {
    pub name: String,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    /// cuBLASLt 启发式第一候选（= 不调优时的默认）
    pub lt_default_us: f32,
    /// cuBLASLt 候选中最快者
    pub lt_best_us: f32,
    pub lt_best_desc: String,
    /// 小 M kernel 最快者（M 超过上限或 K 不对齐则 None）
    pub small_m_us: Option<f32>,
    pub small_m_splits: usize,
    /// tensor-core 小 M kernel 最快者（M 超上限或 K % 32 != 0 则 None）
    pub mma_us: Option<f32>,
    pub mma_splits: usize,
    pub mma_warps: usize,
    pub mma_stripes: usize,
    pub chosen: GemmPlan,
}

impl GemmTuneRow {
    /// 只读一遍权重的有效带宽（GB/s）。
    pub fn gbps(&self, us: f32) -> f32 {
        (self.n * self.k * 2) as f32 / us / 1e3
    }
    pub fn chosen_us(&self) -> f32 {
        match self.chosen {
            GemmPlan::LtAlgo(_) => self.lt_best_us,
            GemmPlan::SmallM { .. } => self.small_m_us.unwrap_or(f32::INFINITY),
            GemmPlan::SmallMma { .. } => self.mma_us.unwrap_or(f32::INFINITY),
        }
    }
}

pub struct GemmTuneReport {
    pub rows: Vec<GemmTuneRow>,
    pub elapsed_ms: f64,
    /// 为 split-K 分配的 partial 工作区字节数
    pub partial_bytes: usize,
    /// 非桶 M 借桶结论的条数 / 因 AlgoCheck 不合法而放弃的条数
    pub borrowed: usize,
    pub rejected: usize,
}

impl GemmTuneReport {
    /// 一行一个形状：`name: m1=small_m(s4) 41us | m2=lt#2 ...`
    pub fn summary_lines(&self) -> Vec<String> {
        let mut names: Vec<&str> = Vec::new();
        for r in &self.rows {
            if !names.contains(&r.name.as_str()) {
                names.push(&r.name);
            }
        }
        names
            .iter()
            .map(|nm| {
                let parts: Vec<String> = self
                    .rows
                    .iter()
                    .filter(|r| r.name == *nm)
                    .map(|r| {
                        let what = match r.chosen {
                            GemmPlan::LtAlgo(_) => format!("lt[{}]", r.lt_best_desc),
                            GemmPlan::SmallM { splits } => format!("small_m(s{splits})"),
                            GemmPlan::SmallMma { splits, warps, stripes } => format!("mma(s{splits}w{warps}r{stripes})"),
                        };
                        format!("m{}={} {:.0}us({:+.0}%)", r.m, what, r.chosen_us(), (r.chosen_us() / r.lt_default_us - 1.0) * 100.0)
                    })
                    .collect();
                format!("{nm}: {}", parts.join(" | "))
            })
            .collect()
    }
}

/// cuBLASLt 的一组描述符（op + A/B/C layout），用完即毁。
struct LtDescs {
    op: sys::cublasLtMatmulDesc_t,
    a: sys::cublasLtMatrixLayout_t,
    b: sys::cublasLtMatrixLayout_t,
    c: sys::cublasLtMatrixLayout_t,
}

impl Drop for LtDescs {
    fn drop(&mut self) {
        // SAFETY: 句柄由 create_* 创建且只在这里销毁一次。
        unsafe {
            let _ = lt::destroy_matrix_layout(self.a);
            let _ = lt::destroy_matrix_layout(self.b);
            let _ = lt::destroy_matrix_layout(self.c);
            let _ = lt::destroy_matmul_desc(self.op);
        }
    }
}

fn lt_err<T>(r: Result<T, cudarc::cublaslt::result::CublasError>, what: &str) -> Result<T> {
    r.map_err(|e| anyhow!("{what}: cublasLt {:?}", e.0))
}

fn status_ok(st: sys::cublasStatus_t, what: &str) -> Result<()> {
    ensure!(st == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS, "{what}: cublasLt {st:?}");
    Ok(())
}

pub struct Gemm {
    handle: sys::cublasLtHandle_t,
    stream: Stream,
    workspace: CudaSlice<u8>,
    profile: ArchProfile,
    enabled: bool,
    plans: Mutex<HashMap<Key, GemmPlan>>,
    /// 未调优形状的 cuBLASLt 默认 algo（启发式第 1 个）缓存：形状 + workspace 固定时启发式结果确定，
    /// 不缓存则 decode 每步几百次 GEMM 各查一次启发式（host 侧 µs 级 × 几百 = 每步毫秒级 CPU）。
    default_algos: Mutex<HashMap<Key, sys::cublasLtMatmulAlgo_t>>,
    /// split-K 的 partial 工作区（自调优后按所选 plan 的最大需要分配；forward 内不分配，便于 graph 捕获）
    partial: Mutex<Option<CudaSlice<f32>>>,
    /// tensor-core kernel split-K 的每 N-tile 到达计数器（自调优后按最大 ceil(N/16) 分配；kernel 自复位，常为 0）
    counters: Mutex<Option<CudaSlice<i32>>>,
    /// 形状分布统计（`GEMM_SHAPE_STATS=1` 时开；只累加计数，不做时间测量，开销可忽略）
    shape_stats: Option<Mutex<HashMap<(usize, usize, usize), (u64, f64)>>>,
}

// SAFETY: cublasLt 句柄按 NVIDIA 文档可跨线程使用；其余字段自身 Send/Sync（Mutex 包着可变状态）。
unsafe impl Send for Gemm {}
unsafe impl Sync for Gemm {}

impl Drop for Gemm {
    fn drop(&mut self) {
        // `GEMM_SHAPE_STATS=1`：退出时按调用次数打印形状分布（诊断用，找 GEMM 形状热点）。
        if let Some(st) = &self.shape_stats {
            if let Ok(g) = st.lock() {
                let mut v: Vec<_> = g.iter().map(|(k, (c, t))| (*k, *c, *t)).collect();
                v.sort_by_key(|(_, _, t)| std::cmp::Reverse((t * 1000.0) as u64));
                let tot: f64 = v.iter().map(|(_, _, t)| *t).sum();
                eprintln!("[gemm-shape] {} shapes, {:.1} ms host-side total", v.len(), tot);
                for ((m, n, k), c, t) in v.iter().take(30) {
                    eprintln!("[gemm-shape] m={m:<6} n={n:<6} k={k:<6} calls={c:<6} {t:8.2} ms {:5.1}%", 100.0*t/tot);
                }
            }
        }
        // SAFETY: 句柄由 create_handle 创建，只销毁一次。
        unsafe {
            let _ = lt::destroy_handle(self.handle);
        }
    }
}

impl Gemm {
    pub fn new(stream: &Stream) -> Result<Self> {
        let mut profile = ArchProfile::detect(&stream.device())?;
        // 环境变量只做覆盖（默认值在 ArchProfile）：实验更多启发式候选对小 M 桶的收益
        if let Some(n) = std::env::var("GEMM_LT_CANDIDATES").ok().and_then(|v| v.parse::<usize>().ok()) {
            profile.gemm_tune.lt_candidates = n.clamp(1, 64);
        }
        let handle = lt_err(lt::create_handle(), "cublasLt create")?;
        let workspace = stream.alloc_zeros::<u8>(profile.cublaslt_workspace_bytes)?;
        let enabled = std::env::var("GEMM_TUNE").map_or(true, |v| v != "off" && v != "0");
        let shape_stats = (std::env::var("GEMM_SHAPE_STATS").ok().as_deref() == Some("1"))
            .then(|| Mutex::new(HashMap::new()));
        Ok(Self {
            handle,
            stream: stream.clone(),
            workspace,
            profile,
            enabled,
            plans: Mutex::new(HashMap::new()),
            default_algos: Mutex::new(HashMap::new()),
            partial: Mutex::new(None),
            counters: Mutex::new(None),
            shape_stats,
        })
    }

    pub fn profile(&self) -> &ArchProfile {
        &self.profile
    }
    /// 自调优是否开启（`GEMM_TUNE=off` 关）。
    pub fn tune_enabled(&self) -> bool {
        self.enabled
    }

    /// FP8 per-tensor GEMM（sm_120 专用，`cuda/gemm_fp8_sm120.cu`）：
    /// `out[m,n] = (e4m3(a[m,k] / a_scale) · w[n,k]^T) * (a_scale * w_scale)`：`a` 是 bf16 激活，`a_scale` 是静态
    /// per-tensor 激活 scale（checkpoint 的 `input_scale`，没有就传 1.0），`w` 是 [n,k] 的 e4m3 字节。
    ///
    /// 编译时未编入 sm_120 实现（CUDA_ARCHS 无 120）时返回错误，调用方自行回退。
    #[allow(clippy::too_many_arguments)]
    pub fn forward_fp8(
        &self,
        a: &CudaSlice<bf16>,
        a_scale: f32,
        w: &CudaSlice<u8>,
        w_scale: f32,
        out: &mut CudaSlice<bf16>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        ensure!(a.len() >= m * k, "gemm fp8: a.len {} < m*k {}", a.len(), m * k);
        ensure!(w.len() == n * k, "gemm fp8: w.len {} != n*k {}", w.len(), n * k);
        ensure!(out.len() >= m * n, "gemm fp8: out.len {} < m*n {}", out.len(), m * n);
        if m == 0 || n == 0 {
            return Ok(());
        }
        ensure!(k > 0, "gemm fp8: k == 0");
        ensure!(m <= i32::MAX as usize && n <= i32::MAX as usize && k <= i32::MAX as usize, "gemm fp8: 维度 > i32");
        let rc = unsafe {
            crate::ffi::gemm_fp8_static_bf16(
                self.stream.ptr(a) as *const c_void,
                a_scale,
                self.stream.ptr(w) as *const c_void,
                w_scale,
                self.stream.ptr_mut(out) as *mut c_void,
                m as i64,
                n as i64,
                k as i64,
                self.stream.raw() as _,
            )
        };
        crate::ops::check(rc, "gemm_fp8_bf16")
    }

    /// **预量化** FP8 GEMM（sm_120）：`a` 已是 e4m3 字节 `[m, k]`，kernel 不再做量化。
    /// `out[m,n] = (Σ_k a[m,k]·w[n,k]) * (a_scale * w_scale)`，`w` 仍是 `[n, k]` 的 e4m3。
    ///
    /// 与 [`forward_fp8`] 共用同一套 smem 布局 / mma / 累加顺序，唯一差别是 A 不做 `x/a_scale` + RNE。
    /// 用途：与外部实现（vLLM 的 FlashInfer）**逐位对拍**——把对方真实喂进算子的字节原样灌进来比输出。
    pub fn forward_fp8_prequant(
        &self,
        a: &CudaSlice<u8>,
        a_scale: f32,
        w: &CudaSlice<u8>,
        w_scale: f32,
        out: &mut CudaSlice<bf16>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        ensure!(a.len() >= m * k, "gemm fp8 prequant: a.len {} < m*k {}", a.len(), m * k);
        ensure!(w.len() == n * k, "gemm fp8 prequant: w.len {} != n*k {}", w.len(), n * k);
        ensure!(out.len() >= m * n, "gemm fp8 prequant: out.len {} < m*n {}", out.len(), m * n);
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }
        ensure!(m <= i32::MAX as usize && n <= i32::MAX as usize && k <= i32::MAX as usize, "维度 > i32");
        let rc = unsafe {
            crate::ffi::gemm_fp8_prequant_bf16(
                self.stream.ptr(a) as *const c_void,
                a_scale,
                self.stream.ptr(w) as *const c_void,
                w_scale,
                self.stream.ptr_mut(out) as *mut c_void,
                m as i64,
                n as i64,
                k as i64,
                self.stream.raw() as _,
            )
        };
        crate::ops::check(rc, "gemm_fp8_prequant_bf16")
    }

    /// NVFP4 块缩放 GEMM（sm_120 专用，`cuda/gemm_nvfp4_sm120.cu`）：
    /// `out[m,n] = global * Σ_k a[m,k] · packed[n,k]`。`a` 是 bf16 激活，**在 kernel 内按 16 元素一块量化成 fp4**
    /// （块缩放 = amax/6 的 e4m3 编码，与权重侧同口径）；`packed` 是 [n, k/2]（每字节两个 fp4，低半字节是偶数列），
    /// `scale` 是 [n, k/16] 的 e4m3 块缩放，`global` 是最终输出乘子（w_global · input_scale）。要求 `k % 64 == 0`。
    /// `a_gscale` = 激活全局 scale（vLLM a1_gscale = 1/input_scale；不用时 1.0）：块缩放 = e4m3(gs·amax/6)。
    #[allow(clippy::too_many_arguments)]
    pub fn forward_nvfp4(
        &self,
        a: &CudaSlice<bf16>,
        a_gscale: f32,
        packed: &CudaSlice<u8>,
        scale: &CudaSlice<u8>,
        global: f32,
        out: &mut CudaSlice<bf16>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        ensure!(a.len() >= m * k, "gemm nvfp4: a.len {} < m*k {}", a.len(), m * k);
        ensure!(packed.len() == n * (k / 2), "gemm nvfp4: packed.len {} != n*k/2 {}", packed.len(), n * (k / 2));
        ensure!(scale.len() == n * (k / 16), "gemm nvfp4: scale.len {} != n*k/16 {}", scale.len(), n * (k / 16));
        ensure!(out.len() >= m * n, "gemm nvfp4: out.len {} < m*n {}", out.len(), m * n);
        if m == 0 || n == 0 {
            return Ok(());
        }
        ensure!(k > 0 && k % 64 == 0, "gemm nvfp4: k {k} 不是 64 的倍数（块缩放 mma 是 k64）");
        ensure!(m <= i32::MAX as usize && n <= i32::MAX as usize && k <= i32::MAX as usize, "gemm nvfp4: 维度 > i32");
        let rc = unsafe {
            crate::ffi::gemm_nvfp4_gs_bf16(
                self.stream.ptr(a) as *const c_void,
                a_gscale,
                self.stream.ptr(packed) as *const c_void,
                self.stream.ptr(scale) as *const c_void,
                global,
                self.stream.ptr_mut(out) as *mut c_void,
                m as i64,
                n as i64,
                k as i64,
                self.stream.raw() as _,
            )
        };
        crate::ops::check(rc, "gemm_nvfp4_bf16")
    }

    /// out[M,N] = a[M,K] · w[N,K]^T，可选 bias[N]。
    #[allow(clippy::too_many_arguments)]
    /// 同 [`forward_bf16`]，但**指定启发式候选的下标**（对拍/标定用，不进热路径）。
    #[allow(clippy::too_many_arguments)]
    pub fn forward_bf16_algo(
        &self,
        a: &CudaSlice<bf16>,
        w: &CudaSlice<bf16>,
        bias: Option<&CudaSlice<bf16>>,
        out: &mut CudaSlice<bf16>,
        m: usize,
        n: usize,
        k: usize,
        algo_idx: usize,
    ) -> Result<()> {
        ensure!(a.len() >= m * k && w.len() == n * k && out.len() >= m * n, "gemm bf16 algo: 尺寸不符");
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }
        let d = self.make_descs(m, n, k, bias.map(|b| self.stream.ptr(b)))?;
        let hs = self.heuristics(&d, 32)?;
        ensure!(algo_idx < hs.len(), "只有 {} 个候选", hs.len());
        self.run_lt(&d, &hs[algo_idx].algo, a, w, out)
    }

    /// 当前形状的 bf16 启发式候选数。
    pub fn bf16_algo_count(&self, m: usize, n: usize, k: usize, bias: bool) -> Result<usize> {
        let d = self.make_descs(m, n, k, None)?;
        let _ = bias;
        Ok(self.heuristics(&d, 32)?.len())
    }

    /// 第 `algo_idx` 个候选的可读摘要（`id=/tile=/splitk=`）。
    pub fn bf16_algo_desc(&self, m: usize, n: usize, k: usize, algo_idx: usize) -> Result<String> {
        let d = self.make_descs(m, n, k, None)?;
        let hs = self.heuristics(&d, 32)?;
        ensure!(algo_idx < hs.len(), "只有 {} 个候选", hs.len());
        Ok(Self::algo_desc_full(&hs[algo_idx].algo))
    }

    pub fn forward_bf16(
        &self,
        a: &CudaSlice<bf16>,
        w: &CudaSlice<bf16>,
        bias: Option<&CudaSlice<bf16>>,
        out: &mut CudaSlice<bf16>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        // a / out 允许是按 max_tokens 预分配的更大缓冲（只用前 m 行）；权重必须精确。
        ensure!(a.len() >= m * k, "gemm: a.len {} < m*k {}", a.len(), m * k);
        ensure!(w.len() == n * k, "gemm: w.len {} != n*k {}", w.len(), n * k);
        ensure!(out.len() >= m * n, "gemm: out.len {} < m*n {}", out.len(), m * n);
        if let Some(b) = bias {
            ensure!(b.len() == n, "gemm: bias.len {} != n {n}", b.len());
        }
        if m == 0 {
            return Ok(());
        }
        let _t0 = if self.shape_stats.is_some() { Some(std::time::Instant::now()) } else { None };
        let r = self.forward_inner(a, w, bias, out, m, n, k);
        if let (Some(st), Some(t0)) = (&self.shape_stats, _t0) {
            if let Ok(mut g) = st.lock() {
                let e = g.entry((m, n, k)).or_insert((0, 0.0));
                e.0 += 1;
                e.1 += t0.elapsed().as_secs_f64() * 1e3;
            }
        }
        return r;
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_inner(
        &self,
        a: &CudaSlice<bf16>,
        w: &CudaSlice<bf16>,
        bias: Option<&CudaSlice<bf16>>,
        out: &mut CudaSlice<bf16>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        match self.plan_for(m, n, k, bias.is_some()) {
            Some(GemmPlan::SmallM { splits }) => {
                let mut guard = self.partial.lock().unwrap();
                let fits = splits <= 1 || guard.as_ref().is_some_and(|p| p.len() >= splits * m * n);
                if fits {
                    return ops::gemm_small_m_bf16(&self.stream, a, w, bias, out, guard.as_mut(), m, n, k, splits);
                }
                drop(guard);
                self.run_lt_default(a, w, bias, out, m, n, k)
            }
            Some(GemmPlan::SmallMma { splits, warps, stripes }) => {
                let mut pg = self.partial.lock().unwrap();
                let mut cg = self.counters.lock().unwrap();
                let fits = splits <= 1
                    || (pg.as_ref().is_some_and(|p| p.len() >= splits * m * n) && cg.as_ref().is_some_and(|c| c.len() >= n.div_ceil(8)));
                if fits {
                    return ops::gemm_small_m_mma_bf16(&self.stream, a, w, bias, out, pg.as_mut(), cg.as_mut(), m, n, k, splits, warps, stripes);
                }
                drop(pg);
                drop(cg);
                self.run_lt_default(a, w, bias, out, m, n, k)
            }
            Some(GemmPlan::LtAlgo(algo)) => {
                let d = self.make_descs(m, n, k, bias.map(|b| self.stream.ptr(b)))?;
                self.run_lt(&d, &algo, a, w, out)
            }
            None => self.run_lt_default(a, w, bias, out, m, n, k),
        }
    }

    fn plan_for(&self, m: usize, n: usize, k: usize, bias: bool) -> Option<GemmPlan> {
        if !self.enabled {
            return None;
        }
        let plans = self.plans.lock().unwrap();
        if plans.is_empty() {
            return None;
        }
        if let Some(p) = plans.get(&Key::bf16(m, n, k, bias)) {
            return Some(*p);
        }
        // 非桶 M 在 autotune 末尾已按桶展开（1..=最大桶 每个 m 一条，Lt algo 过 AlgoCheck 才借）；
        // 这里只兜底 small_m（对任意 m ≤ 桶都正确）。
        if m <= self.profile.gemm_tune.small_m_max {
            let bucket = m.next_power_of_two();
            if let Some(p @ (GemmPlan::SmallM { .. } | GemmPlan::SmallMma { .. })) = plans.get(&Key::bf16(bucket, n, k, bias)) {
                return Some(*p);
            }
        }
        None
    }

    /// 桶 M 选出的 Lt algo 能否用于同 (N,K) 的另一个 M（layout 的列数变了）：`cublasLtMatmulAlgoCheck`。
    fn algo_valid_for(&self, d: &LtDescs, algo: &sys::cublasLtMatmulAlgo_t) -> bool {
        // SAFETY: 描述符与 algo 均有效；result 只是输出缓冲。
        unsafe {
            let mut res: sys::cublasLtMatmulHeuristicResult_t = std::mem::zeroed();
            let st = sys::cublasLtMatmulAlgoCheck(self.handle, d.op, d.a, d.b, d.c, d.c, algo as *const _, &mut res);
            st == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS
                && res.state == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS
                && res.workspaceSize <= self.workspace.len()
        }
    }

    /// 调优时的实测：某 plan 在实际 (m,n,k) 上的 µs（微基准用；LtAlgo 先过 AlgoCheck，不合法返回 None）。
    #[allow(clippy::too_many_arguments)]
    pub fn time_plan(
        &self,
        plan: GemmPlan,
        m: usize,
        n: usize,
        k: usize,
        w: &CudaSlice<bf16>,
        bias: Option<&CudaSlice<bf16>>,
        a_scratch: &CudaSlice<bf16>,
        out_scratch: &mut CudaSlice<bf16>,
        partial_scratch: &mut CudaSlice<f32>,
    ) -> Result<Option<f32>> {
        let iters = self.profile.gemm_tune.timing_iters.max(1);
        match plan {
            GemmPlan::LtAlgo(algo) => {
                let d = self.make_descs(m, n, k, bias.map(|b| self.stream.ptr(b)))?;
                if !self.algo_valid_for(&d, &algo) {
                    return Ok(None);
                }
                self.time_us(iters, || self.run_lt(&d, &algo, a_scratch, w, out_scratch)).map(Some)
            }
            GemmPlan::SmallM { splits } => {
                if m > self.profile.gemm_tune.small_m_max || k % 8 != 0 || splits * m * n > partial_scratch.len() {
                    return Ok(None);
                }
                self.time_us(iters, || ops::gemm_small_m_bf16(&self.stream, a_scratch, w, bias, out_scratch, Some(&mut *partial_scratch), m, n, k, splits)).map(Some)
            }
            GemmPlan::SmallMma { splits, warps, stripes } => {
                if m > self.profile.gemm_tune.small_m_max || k % 128 != 0 || splits * m * n > partial_scratch.len() {
                    return Ok(None);
                }
                let mut ctr = self.stream.alloc_zeros::<i32>(n.div_ceil(8))?;
                self.time_us(iters, || {
                    ops::gemm_small_m_mma_bf16(&self.stream, a_scratch, w, bias, out_scratch, Some(&mut *partial_scratch), Some(&mut ctr), m, n, k, splits, warps, stripes)
                })
                .map(Some)
            }
        }
    }

    fn run_lt_default(
        &self,
        a: &CudaSlice<bf16>,
        w: &CudaSlice<bf16>,
        bias: Option<&CudaSlice<bf16>>,
        out: &mut CudaSlice<bf16>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let d = self.make_descs(m, n, k, bias.map(|b| self.stream.ptr(b)))?;
        let key = Key::bf16(m, n, k, bias.is_some());
        let cached = self.default_algos.lock().unwrap().get(&key).copied();
        let algo = match cached {
            Some(a) => a,
            None => {
                // **与 fp8 同样的问题**：默认取启发式第 0 个，与 vLLM（PyTorch `F.linear` → cuBLAS）
                // 的累加序不同。实测 router 形状 (m=24,n=128,k=2816) 上第 0 个差 **768/3072**。
                // 区分点是 **reduction scheme**，不是 id/tile/splitk：8 个候选里 `red=4`（OUTPUT_TYPE，
                // split-K 部分和先舍入到 bf16）的全部差 768–803，`red=2`（COMPUTE_TYPE，f32 归约）的 3 个全为 0
                // （短描述同为 `id=21 tile=11 splitk=8` 的 [2]/[3] 一个 801 一个 0）。
                // 但 L0 那份参照是事后单独 `F.linear` 算的；**vLLM 真实前向（oracle 在 `VLLM_BATCH_INVARIANT=1`
                // 下生成）**在 sm_120 上走的是：`preferred_blas_library("cublaslt")` + 禁降精度归约 +
                // `CUBLASLT_WORKSPACE_SIZE=1`（1 KiB，禁掉要 workspace 的 split-K）。L2 真实输入上，32 MiB
                // workspace 下没有任何候选为 0（最好 3/3072），按这套约束则 L0/L2 均为 0（§51）。
                // 所以：归约掩码一律排除 OUTPUT_TYPE；workspace 上限读与 torch 同名同单位（KiB）的
                // `CUBLASLT_WORKSPACE_SIZE`（parity 对 BI oracle 时设 1），未设 = 整块 workspace。
                let ws = std::env::var("CUBLASLT_WORKSPACE_SIZE").ok().and_then(|v| v.trim().parse::<usize>().ok()).map_or(usize::MAX, |kib| kib * 1024);
                use sys::cublasLtReductionScheme_t as R;
                let mask = R::CUBLASLT_REDUCTION_SCHEME_NONE as u32 | R::CUBLASLT_REDUCTION_SCHEME_INPLACE as u32 | R::CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE as u32;
                let hs = self.heuristics_with(&d, 32, ws, Some(mask))?;
                ensure!(!hs.is_empty(), "cublasLt: no algo for m={m} n={n} k={k}");
                let pick = hs[0].algo;
                if std::env::var("GEMM_ALGO_LOG").is_ok_and(|v| v == "1") {
                    let all: Vec<String> = hs.iter().map(|h| Self::algo_desc_full(&h.algo)).collect();
                    eprintln!("[gemm] bf16 m={m} n={n} k={k} pick={} cands={all:?}", Self::algo_desc_full(&pick));
                }
                self.default_algos.lock().unwrap().insert(key, pick);
                pick
            }
        };
        self.run_lt(&d, &algo, a, w, out)
    }

    fn make_descs(&self, m: usize, n: usize, k: usize, bias_ptr: Option<u64>) -> Result<LtDescs> {
        use sys::cublasLtMatmulDescAttributes_t as D;
        let ty = sys::cudaDataType_t::CUDA_R_16BF;
        // cuBLAS 视角：A 操作数 = W（transa=T，存储 [K,N] 列主序，lda=K）；B = 激活（[K,M] 列主序，ldb=K）；C = [N,M]，ldc=N
        let a = lt_err(lt::create_matrix_layout(ty, k as u64, n as u64, k as i64), "layout W")?;
        let b = match lt::create_matrix_layout(ty, k as u64, m as u64, k as i64) {
            Ok(b) => b,
            Err(e) => {
                unsafe { let _ = lt::destroy_matrix_layout(a); }
                bail!("layout A: cublasLt {:?}", e.0);
            }
        };
        let c = match lt::create_matrix_layout(ty, n as u64, m as u64, n as i64) {
            Ok(c) => c,
            Err(e) => {
                unsafe { let _ = lt::destroy_matrix_layout(a); let _ = lt::destroy_matrix_layout(b); }
                bail!("layout C: cublasLt {:?}", e.0);
            }
        };
        let op = match lt::create_matmul_desc(sys::cublasComputeType_t::CUBLAS_COMPUTE_32F, sys::cudaDataType_t::CUDA_R_32F) {
            Ok(op) => op,
            Err(e) => {
                unsafe { let _ = lt::destroy_matrix_layout(a); let _ = lt::destroy_matrix_layout(b); let _ = lt::destroy_matrix_layout(c); }
                bail!("matmul desc: cublasLt {:?}", e.0);
            }
        };
        let d = LtDescs { op, a, b, c };
        // cublasOperation_t 以 int 传：1 = T，0 = N（与 cudarc set_transpose 一致）
        let t: i32 = 1;
        let nt: i32 = 0;
        // SAFETY: 属性值类型与 cublasLt 文档一致，指针指向本栈变量。
        unsafe {
            lt_err(lt::set_matmul_desc_attribute(d.op, D::CUBLASLT_MATMUL_DESC_TRANSA, &t as *const _ as *const c_void, std::mem::size_of_val(&t)), "transa")?;
            lt_err(lt::set_matmul_desc_attribute(d.op, D::CUBLASLT_MATMUL_DESC_TRANSB, &nt as *const _ as *const c_void, std::mem::size_of_val(&nt)), "transb")?;
            if let Some(bp) = bias_ptr {
                let epi = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS;
                lt_err(lt::set_matmul_desc_attribute(d.op, D::CUBLASLT_MATMUL_DESC_EPILOGUE, &epi as *const _ as *const c_void, std::mem::size_of_val(&epi)), "epilogue")?;
                lt_err(lt::set_matmul_desc_attribute(d.op, D::CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bp as *const _ as *const c_void, std::mem::size_of_val(&bp)), "bias ptr")?;
            }
        }
        Ok(d)
    }

    /// 启发式候选（最多 `count` 个，按 cuBLASLt 的排序；过滤掉 state != SUCCESS 的）。
    fn heuristics(&self, d: &LtDescs, count: usize) -> Result<Vec<sys::cublasLtMatmulHeuristicResult_t>> {
        self.heuristics_with(d, count, self.workspace.len(), None)
    }

    /// 带约束的启发式：`ws` = workspace 上限（字节），`red_mask` = 允许的 reduction scheme 掩码。
    fn heuristics_with(&self, d: &LtDescs, count: usize, ws: usize, red_mask: Option<u32>) -> Result<Vec<sys::cublasLtMatmulHeuristicResult_t>> {
        let pref = lt_err(lt::create_matmul_pref(), "matmul pref")?;
        let ws = ws.min(self.workspace.len());
        // SAFETY: 属性/缓冲区按 cublasLt 文档；pref 创建成功后在此销毁。
        let r = unsafe {
            lt::set_matmul_pref_attribute(
                pref,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &ws as *const usize as *const c_void,
                std::mem::size_of::<usize>(),
            )
            .and_then(|_| match red_mask {
                Some(m) => lt::set_matmul_pref_attribute(
                    pref,
                    sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                    &m as *const u32 as *const c_void,
                    std::mem::size_of::<u32>(),
                ),
                None => Ok(()),
            })
            .map_err(|e| anyhow!("pref workspace/reduction: cublasLt {:?}", e.0))
            .and_then(|_| {
                let mut res: Vec<sys::cublasLtMatmulHeuristicResult_t> = vec![std::mem::zeroed(); count.max(1)];
                let mut cnt: i32 = 0;
                let st = sys::cublasLtMatmulAlgoGetHeuristic(self.handle, d.op, d.a, d.b, d.c, d.c, pref, count.max(1) as i32, res.as_mut_ptr(), &mut cnt);
                status_ok(st, "algo heuristic")?;
                res.truncate(cnt.max(0) as usize);
                res.retain(|h| h.state == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS);
                Ok(res)
            })
        };
        unsafe {
            let _ = lt::destroy_matmul_pref(pref);
        }
        r
    }

    /// bf16 → e4m3 量化（静态 per-tensor 激活 scale）：`out[i] = e4m3(round(x[i] / a_scale))`。
    /// 与 `forward_fp8` 内联的量化**逐位一致**，供 [`forward_fp8_cublaslt`] 前使用。
    pub fn quant_fp8(&self, x: &CudaSlice<bf16>, a_scale: f32, out: &mut CudaSlice<u8>, n: usize) -> Result<()> {
        ensure!(a_scale > 0.0, "quant fp8: a_scale 必须 > 0");
        ensure!(x.len() >= n, "quant fp8: x.len {} < {}", x.len(), n);
        ensure!(out.len() >= n, "quant fp8: out.len {} < {}", out.len(), n);
        if n == 0 {
            return Ok(());
        }
        let rc = unsafe {
            crate::ffi::quant_fp8_bf16(
                self.stream.ptr(x) as *const c_void,
                a_scale,
                self.stream.ptr_mut(out) as *mut c_void,
                n as i64,
                self.stream.raw() as _,
            )
        };
        crate::ops::check(rc, "quant_fp8_bf16")
    }

    /// **cuBLASLt 的 fp8 per-tensor GEMM**（sm_120）：`out[m,n] = (A·Wᵀ) * (a_scale * w_scale)`，
    /// `A`/`W` 都是 e4m3 字节，`W` 是 `[n, k]`。
    ///
    /// 为什么走这条路：实测 vLLM 的 fp8 GEMM 真实命中 **cuBLAS**（`CublasFp8GemmRunner`），
    /// 而我们自写 kernel 与它差 8/196608 个 1-ulp（累加序不同），却怎么都追不齐。
    /// 直接调同一个库 → 实测 **0/196608**（`torch._scaled_mm` 交叉验证，见
    /// `test/kaon3/2026-10-23/D-16-REBASE.md` §24）。
    ///
    /// cuBLAS 视角与 bf16 路径相同：A 操作数 = W（存储 `[K,N]` 列主序，即物理 `[n,k]`），
    /// B 操作数 = 激活（`[K,M]` 列主序，即物理 `[m,k]`），C = `[N,M]`。
    /// 两个 scale 是**设备上的 f32 标量指针**（`A_SCALE_POINTER` / `B_SCALE_POINTER`）。
    #[allow(clippy::too_many_arguments)]
    pub fn forward_fp8_cublaslt(
        &self,
        a: &CudaSlice<u8>,
        a_scale: &CudaSlice<f32>,
        w: &CudaSlice<u8>,
        w_scale: &CudaSlice<f32>,
        out: &mut CudaSlice<bf16>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        ensure!(a.len() >= m * k, "gemm fp8 lt: a.len {} < m*k {}", a.len(), m * k);
        ensure!(w.len() == n * k, "gemm fp8 lt: w.len {} != n*k {}", w.len(), n * k);
        ensure!(out.len() >= m * n, "gemm fp8 lt: out.len {} < m*n {}", out.len(), m * n);
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }
        let d = self.make_descs_fp8(m, n, k, self.stream.ptr(a_scale), self.stream.ptr(w_scale))?;
        let key = Key::fp8(m, n, k);
        let cached = self.default_algos.lock().unwrap().get(&key).copied();
        let algo = match cached {
            Some(al) => al,
            None => {
                // **按 algo 的配置属性选，不按下标**。实测（Kaon-V3 的 6 种 (n,k)，见
                // test/kaon3/2026-10-23/D-16-REBASE.md §40–41）：与 vLLM 逐位一致的配置，
                // 除 K=8192 外都是 `id=67 tile=13 splitk=1`；而该配置在启发式列表里的**位置随形状漂移**
                // （某形状排第 0、另一个排第 2），所以下标不是稳定标识。
                // `2816×8192` 是例外：需要 split-K 的 `id=35 tile=13 splitk=3`。
                let hs = self.heuristics(&d, 32)?;
                ensure!(!hs.is_empty(), "cublasLt fp8: no algo for m={m} n={n} k={k}");
                // 实测表：按 (n, k) 给首选配置，未命中则用通配。
                const TABLE: &[(usize, usize, &str)] = &[(2816, 8192, "id=35 tile=13 splitk=3")];
                const FALLBACK: &str = "id=67 tile=13 splitk=1";
                let want = TABLE
                    .iter()
                    .find(|(tn, tk, _)| *tn == n && *tk == k)
                    .map(|(_, _, s)| *s)
                    .unwrap_or(FALLBACK);
                let pick = hs
                    .iter()
                    .find(|h| Self::algo_desc(&h.algo) == want)
                    .map(|h| h.algo)
                    .unwrap_or(hs[0].algo);
                self.default_algos.lock().unwrap().insert(key, pick);
                pick
            }
        };
        self.run_lt_fp8(&d, &algo, a, w, out)
    }

    /// 同 [`forward_fp8_cublaslt`]，但**指定启发式候选的下标**（`None` = 第 0 个，即默认）。
    ///
    /// 存在意义：实测 vLLM（PyTorch `_scaled_mm` 路径）能逐位复现，而默认启发式第 0 个不同
    /// —— 说明两者选的 algo 不同。用它把候选逐个扫一遍，找出与参照一致的那个。
    /// 仅供对拍 / 定标用，不要进热路径。
    #[allow(clippy::too_many_arguments)]
    pub fn forward_fp8_cublaslt_algo(
        &self,
        a: &CudaSlice<u8>,
        a_scale: &CudaSlice<f32>,
        w: &CudaSlice<u8>,
        w_scale: &CudaSlice<f32>,
        out: &mut CudaSlice<bf16>,
        m: usize,
        n: usize,
        k: usize,
        algo_idx: usize,
    ) -> Result<()> {
        ensure!(a.len() >= m * k && w.len() == n * k && out.len() >= m * n, "gemm fp8 lt algo: 尺寸不符");
        if m == 0 || n == 0 || k == 0 {
            return Ok(());
        }
        let d = self.make_descs_fp8(m, n, k, self.stream.ptr(a_scale), self.stream.ptr(w_scale))?;
        let hs = self.heuristics(&d, 32)?;
        ensure!(algo_idx < hs.len(), "只有 {} 个候选 algo，要第 {}", hs.len(), algo_idx);
        self.run_lt_fp8(&d, &hs[algo_idx].algo, a, w, out)
    }

    /// 当前形状的启发式候选数量（配合 `forward_fp8_cublaslt_algo` 用来枚举）。
    pub fn fp8_cublaslt_algo_count(
        &self,
        a_scale: &CudaSlice<f32>,
        w_scale: &CudaSlice<f32>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<usize> {
        let d = self.make_descs_fp8(m, n, k, self.stream.ptr(a_scale), self.stream.ptr(w_scale))?;
        Ok(self.heuristics(&d, 32)?.len())
    }

    /// 当前形状第 `algo_idx` 个候选 algo 的可读摘要（`id=/tile=/splitk=`），用于标定与日志。
    pub fn fp8_cublaslt_algo_desc(
        &self,
        a_scale: &CudaSlice<f32>,
        w_scale: &CudaSlice<f32>,
        m: usize,
        n: usize,
        k: usize,
        algo_idx: usize,
    ) -> Result<String> {
        let d = self.make_descs_fp8(m, n, k, self.stream.ptr(a_scale), self.stream.ptr(w_scale))?;
        let hs = self.heuristics(&d, 32)?;
        ensure!(algo_idx < hs.len(), "只有 {} 个候选", hs.len());
        Ok(Self::algo_desc(&hs[algo_idx].algo))
    }

    fn run_lt_fp8(
        &self,
        d: &LtDescs,
        algo: &sys::cublasLtMatmulAlgo_t,
        a: &CudaSlice<u8>,
        w: &CudaSlice<u8>,
        out: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        let alpha = 1.0f32;
        let beta = 0.0f32;
        // 描述符里 A 操作数 = 权重、B 操作数 = 激活（见 make_descs_fp8），所以传参顺序是 (w, a)。
        let pw = self.stream.ptr(w) as *const c_void;
        let pa = self.stream.ptr(a) as *const c_void;
        let po = self.stream.ptr_mut(out) as *mut c_void;
        let pws = self.stream.ptr(&self.workspace) as *mut c_void;
        // SAFETY: 描述符由 make_descs_fp8 构造，指针指向调用方持有的设备缓冲。
        let r = unsafe {
            lt::matmul(
                self.handle,
                d.op,
                &alpha as *const f32 as *const c_void,
                &beta as *const f32 as *const c_void,
                pw,
                d.a,
                pa,
                d.b,
                po as *const c_void,
                d.c,
                po,
                d.c,
                algo as *const _,
                pws,
                self.workspace.len(),
                self.stream.raw() as *mut _,
            )
        };
        lt_err(r, "cublasLt matmul fp8")
    }

    /// fp8 版本的描述符。
    ///
    /// **不是**照抄 flashinfer 的 `Fp8GemmDescriptors`：它用 `d_layout(m,n,ld=m)`（列主序输出），
    /// 而我们输出是行主序 `bf16 out[m,n]`。实测把方向照搬过来后，cuBLASLt 在 fp8 的 heuristic
    /// 查询上直接返回 `CUBLAS_STATUS_INVALID_VALUE`（试过带/不带 `FAST_ACCUM` 都一样）。
    /// 因此保留原来的操作数方向（A 操作数 = 权重物理 `[n,k]`），它可用且能收敛到 0（靠 algo 选择）。
    fn make_descs_fp8(&self, m: usize, n: usize, k: usize, a_scale_ptr: u64, w_scale_ptr: u64) -> Result<LtDescs> {
        use sys::cublasLtMatmulDescAttributes_t as D;
        let f8 = sys::cudaDataType_t::CUDA_R_8F_E4M3;
        let out_ty = sys::cudaDataType_t::CUDA_R_16BF;
        // cuBLAS 视角（沿用 bf16 路径）：A 操作数 = W（物理 [n,k] 当 [K,N] 列主序，lda=K），
        // B 操作数 = 激活（[K,M] 列主序，ldb=K），C = [N,M]，ldc=N
        //
        // 关于 flashinfer 的 `a_layout(type, m, k, k, /*t=*/true)`：它内部 `LayoutCreate`
        // 传的是 `(t ? cols : rows, t ? rows : cols)` = **(k, m, ld=k)**，与这里的写法等价。
        // （我一度按字面 `(m,k)` 改过，cuBLASLt 直接返回 INVALID_VALUE——注意那个 `t` 参数。）
        let a = lt_err(lt::create_matrix_layout(f8, k as u64, n as u64, k as i64), "layout W")?;
        let b = match lt::create_matrix_layout(f8, k as u64, m as u64, k as i64) {
            Ok(b) => b,
            Err(e) => {
                unsafe { let _ = lt::destroy_matrix_layout(a); }
                bail!("layout A: cublasLt {:?}", e.0);
            }
        };
        let c = match lt::create_matrix_layout(out_ty, n as u64, m as u64, n as i64) {
            Ok(c) => c,
            Err(e) => {
                unsafe { let _ = lt::destroy_matrix_layout(a); let _ = lt::destroy_matrix_layout(b); }
                bail!("layout C: cublasLt {:?}", e.0);
            }
        };
        let op = match lt::create_matmul_desc(sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                                             sys::cudaDataType_t::CUDA_R_32F) {
            Ok(op) => op,
            Err(e) => {
                unsafe {
                    let _ = lt::destroy_matrix_layout(a);
                    let _ = lt::destroy_matrix_layout(b);
                    let _ = lt::destroy_matrix_layout(c);
                }
                bail!("matmul desc: cublasLt {:?}", e.0);
            }
        };
        let d = LtDescs { op, a, b, c };
        let t: i32 = 1;
        let nt: i32 = 0;
        // SAFETY: 属性值类型与 cublasLt 文档一致：transpose 是 int，scale 是设备指针。
        unsafe {
            lt_err(lt::set_matmul_desc_attribute(d.op, D::CUBLASLT_MATMUL_DESC_TRANSA,
                                                 &t as *const _ as *const c_void, std::mem::size_of_val(&t)), "transa")?;
            lt_err(lt::set_matmul_desc_attribute(d.op, D::CUBLASLT_MATMUL_DESC_TRANSB,
                                                 &nt as *const _ as *const c_void, std::mem::size_of_val(&nt)), "transb")?;
            lt_err(lt::set_matmul_desc_attribute(d.op, D::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                 &a_scale_ptr as *const _ as *const c_void,
                                                 std::mem::size_of_val(&a_scale_ptr)), "a_scale")?;
            lt_err(lt::set_matmul_desc_attribute(d.op, D::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                 &w_scale_ptr as *const _ as *const c_void,
                                                 std::mem::size_of_val(&w_scale_ptr)), "b_scale")?;
        }
        Ok(d)
    }

    fn run_lt(&self, d: &LtDescs, algo: &sys::cublasLtMatmulAlgo_t, a: &CudaSlice<bf16>, w: &CudaSlice<bf16>, out: &mut CudaSlice<bf16>) -> Result<()> {
        let alpha = 1.0f32;
        let beta = 0.0f32;
        let pw = self.stream.ptr(w) as *const c_void;
        let pa = self.stream.ptr(a) as *const c_void;
        let po = self.stream.ptr_mut(out) as *mut c_void;
        let pws = self.stream.ptr(&self.workspace) as *mut c_void;
        // SAFETY: 描述符与指针按上面布局构造，workspace 为本对象持有的设备缓冲。
        let r = unsafe {
            lt::matmul(
                self.handle,
                d.op,
                &alpha as *const f32 as *const c_void,
                &beta as *const f32 as *const c_void,
                pw,
                d.a,
                pa,
                d.b,
                po as *const c_void,
                d.c,
                po,
                d.c,
                algo as *const _,
                pws,
                self.workspace.len(),
                self.stream.raw() as *mut _,
            )
        };
        lt_err(r, "cublasLt matmul bf16")
    }

    /// algo 的可读摘要：`id=<algo id> tile=<tile id> splitk=<n>`。
    fn algo_desc(algo: &sys::cublasLtMatmulAlgo_t) -> String {
        use sys::cublasLtMatmulAlgoConfigAttributes_t as C;
        let get = |attr: C| -> i32 {
            let mut v: i32 = -1;
            let mut written: usize = 0;
            // SAFETY: 属性为 int 类型，缓冲 4 字节。
            unsafe {
                let _ = sys::cublasLtMatmulAlgoConfigGetAttribute(algo as *const _, attr, &mut v as *mut i32 as *mut c_void, 4, &mut written);
            }
            v
        };
        format!("id={} tile={} splitk={}", get(C::CUBLASLT_ALGO_CONFIG_ID), get(C::CUBLASLT_ALGO_CONFIG_TILE_ID), get(C::CUBLASLT_ALGO_CONFIG_SPLITK_NUM))
    }

    /// `CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME`（split-K 部分和按什么精度归约；见 `run_lt_default`）。
    fn algo_reduction(algo: &sys::cublasLtMatmulAlgo_t) -> u32 {
        let mut v: u32 = 0;
        let mut written: usize = 0;
        // SAFETY: 该属性为 u32，缓冲 4 字节。
        unsafe {
            let _ = sys::cublasLtMatmulAlgoConfigGetAttribute(
                algo as *const _,
                sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                &mut v as *mut u32 as *mut c_void,
                4,
                &mut written,
            );
        }
        v
    }

    /// 完整配置串：`id/tile/splitk` 相同的候选仍可能累加序不同（reduction / stages / swizzle 等），
    /// 2026-10-23 router 实测 [2] 与 [3] 短描述同为 `id=21 tile=11 splitk=8`，一个差 801/3072、一个 0。
    fn algo_desc_full(algo: &sys::cublasLtMatmulAlgo_t) -> String {
        use sys::cublasLtMatmulAlgoConfigAttributes_t as C;
        let get = |attr: C| -> u32 {
            let mut v: u32 = 0;
            let mut written: usize = 0;
            // SAFETY: 属性为 ≤4 字节整型（u16/u32/i32），v 预置 0，缓冲 4 字节。
            unsafe {
                let _ = sys::cublasLtMatmulAlgoConfigGetAttribute(algo as *const _, attr, &mut v as *mut u32 as *mut c_void, 4, &mut written);
            }
            v
        };
        format!(
            "{} red={} stages={} swz={} custom={} inner={} cluster={}",
            Self::algo_desc(algo),
            Self::algo_reduction(algo),
            get(C::CUBLASLT_ALGO_CONFIG_STAGES_ID),
            get(C::CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING),
            get(C::CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION),
            get(C::CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID),
            get(C::CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID),
        )
    }

    /// 流上计时：预热 1 次，再跑 `iters` 次取平均（µs）。
    fn time_us(&self, iters: usize, mut f: impl FnMut() -> Result<()>) -> Result<f32> {
        f()?;
        let t0 = self.stream.new_timing_event()?;
        let t1 = self.stream.new_timing_event()?;
        self.stream.record_timing(&t0)?;
        for _ in 0..iters {
            f()?;
        }
        self.stream.record_timing(&t1)?;
        self.stream.synchronize()?;
        Ok(TimingEvent::elapsed_ms(&t0, &t1)? * 1e3 / iters as f32)
    }

    /// 对一个形状的一个 M 实测全部候选。`a_scratch` ≥ m·k，`out_scratch` ≥ m·n，`partial_scratch` 供 split-K 候选用（不够的候选跳过）。
    #[allow(clippy::too_many_arguments)]
    pub fn tune_shape(
        &self,
        name: &str,
        m: usize,
        n: usize,
        k: usize,
        w: &CudaSlice<bf16>,
        bias: Option<&CudaSlice<bf16>>,
        a_scratch: &CudaSlice<bf16>,
        out_scratch: &mut CudaSlice<bf16>,
        partial_scratch: &mut CudaSlice<f32>,
    ) -> Result<GemmTuneRow> {
        ensure!(w.len() == n * k && a_scratch.len() >= m * k && out_scratch.len() >= m * n, "tune_shape: 形状");
        let tp = &self.profile.gemm_tune;
        let iters = tp.timing_iters.max(1);
        let d = self.make_descs(m, n, k, bias.map(|b| self.stream.ptr(b)))?;
        let cands = self.heuristics(&d, tp.lt_candidates)?;
        ensure!(!cands.is_empty(), "tune_shape: cublasLt 无候选 m={m} n={n} k={k}");
        let mut lt_default_us = f32::INFINITY;
        let mut lt_best = (f32::INFINITY, 0usize);
        for (i, h) in cands.iter().enumerate() {
            // 个别候选可能在 matmul 时失败（workspace 不足等）：跳过
            let Ok(us) = self.time_us(iters, || self.run_lt(&d, &h.algo, a_scratch, w, out_scratch)) else { continue };
            if i == 0 {
                lt_default_us = us;
            }
            if us < lt_best.0 {
                lt_best = (us, i);
            }
        }
        ensure!(lt_best.0.is_finite(), "tune_shape: cublasLt 候选全部失败 m={m} n={n} k={k}");
        if !lt_default_us.is_finite() {
            lt_default_us = lt_best.0;
        }
        let mut small_m_us = None;
        let mut small_m_splits = 0;
        if m <= tp.small_m_max && k % 8 == 0 {
            // kernel 结构常量：每 block 4 个 warp，每 warp 1 个 N → 不切时 ceil(n/4) 个 block
            let blocks = n.div_ceil(4);
            for s in self.profile.small_m_split_candidates(blocks, k) {
                if s > 1 && (s * m * n > partial_scratch.len() || s * m * n * 4 > tp.split_workspace_max_bytes) {
                    continue;
                }
                let us = self.time_us(iters, || ops::gemm_small_m_bf16(&self.stream, a_scratch, w, bias, out_scratch, Some(&mut *partial_scratch), m, n, k, s))?;
                if small_m_us.is_none_or(|best| us < best) {
                    small_m_us = Some(us);
                    small_m_splits = s;
                }
            }
        }
        // tensor-core 小 M kernel：候选 = warps（ArchProfile）× split-K（同 small_m 的波数目标，按 16·warps 行/block 算 block 数）
        let mut mma_us = None;
        let mut mma_splits = 0;
        let mut mma_warps = 0;
        let mut mma_stripes = 0;
        if m <= tp.small_m_max && k % 128 == 0 {
            let mut ctr = self.stream.alloc_zeros::<i32>(n.div_ceil(8))?;
            for &wp in &tp.small_m_mma_warps {
                for &rs in &tp.small_m_mma_stripes {
                    let blocks = n.div_ceil(8 * rs * wp);
                    for s in self.profile.small_m_split_candidates(blocks, k) {
                        let s = s.min((k / 128).max(1));
                        if s > 1 && (s * m * n > partial_scratch.len() || s * m * n * 4 > tp.split_workspace_max_bytes) {
                            continue;
                        }
                        // 个别配置 launch 失败（资源）就跳过，不让整个自调优失败
                        let Ok(us) = self.time_us(iters, || {
                            ops::gemm_small_m_mma_bf16(&self.stream, a_scratch, w, bias, out_scratch, Some(&mut *partial_scratch), Some(&mut ctr), m, n, k, s, wp, rs)
                        }) else {
                            continue;
                        };
                        if mma_us.is_none_or(|best| us < best) {
                            mma_us = Some(us);
                            mma_splits = s;
                            mma_warps = wp;
                            mma_stripes = rs;
                        }
                    }
                }
            }
        }
        let mut chosen = match small_m_us {
            Some(su) if su < lt_best.0 => GemmPlan::SmallM { splits: small_m_splits },
            _ => GemmPlan::LtAlgo(cands[lt_best.1].algo),
        };
        let best_so_far = match chosen {
            GemmPlan::SmallM { .. } => small_m_us.unwrap_or(f32::INFINITY),
            _ => lt_best.0,
        };
        if let Some(mu) = mma_us {
            if mu < best_so_far {
                chosen = GemmPlan::SmallMma { splits: mma_splits, warps: mma_warps, stripes: mma_stripes };
            }
        }
        Ok(GemmTuneRow {
            name: name.to_string(),
            m,
            n,
            k,
            lt_default_us,
            lt_best_us: lt_best.0,
            lt_best_desc: format!("#{} {}", lt_best.1, Self::algo_desc(&cands[lt_best.1].algo)),
            small_m_us,
            small_m_splits,
            mma_us,
            mma_splits,
            mma_warps,
            mma_stripes,
            chosen,
        })
    }

    /// 启动自调优：对每个 entry × `ArchProfile::gemm_tune.m_buckets` 实测并填表；为 split-K 分配 partial 工作区。
    /// 模型加载后、CUDA graph 捕获前调一次。`GEMM_TUNE=off` 时直接返回空报告。
    pub fn autotune(&self, entries: &[GemmTuneEntry<'_>]) -> Result<GemmTuneReport> {
        let t0 = Instant::now();
        let mut report = GemmTuneReport { rows: Vec::new(), elapsed_ms: 0.0, partial_bytes: 0, borrowed: 0, rejected: 0 };
        if !self.enabled || entries.is_empty() {
            return Ok(report);
        }
        let tp = &self.profile.gemm_tune;
        // `GEMM_TUNE_M=<逗号分隔>`：额外把指定的 M 纳入调优（诊断 prefill 的大 M ——
        // `m_buckets` 只有 decode 的 1..64，prefill 的 M=6019 从来没被调过，
        // 只能吃 cuBLAS 的默认启发式。设了之后 tune_shape 会给出该 M 上
        // lt_default / lt_best / small_m / mma 四路的实测对比）。
        let mut tp_buckets = tp.m_buckets.clone();
        if let Ok(v) = std::env::var("GEMM_TUNE_M") {
            for t in v.split(',') {
                if let Ok(m) = t.trim().parse::<usize>() {
                    if m > 0 && !tp_buckets.contains(&m) {
                        tp_buckets.push(m);
                    }
                }
            }
            tp_buckets.sort_unstable();
        }
        let m_max = tp_buckets.iter().copied().max().unwrap_or(1);
        let k_max = entries.iter().map(|e| e.k).max().unwrap_or(0);
        let n_max = entries.iter().map(|e| e.n).max().unwrap_or(0);
        let a_scratch = self.stream.alloc_zeros::<bf16>(m_max * k_max)?;
        let mut out_scratch = self.stream.alloc_zeros::<bf16>(m_max * n_max)?;
        let mut partial_scratch = self.stream.alloc_zeros::<f32>(tp.split_workspace_max_bytes / 4)?;
        self.stream.synchronize()?;
        let mut need_partial = 0usize;
        let mut need_counters = 0usize;
        let mut plans = HashMap::new();
        for e in entries {
            for &m in &tp_buckets {
                let row = self.tune_shape(&e.name, m, e.n, e.k, e.w, e.bias, &a_scratch, &mut out_scratch, &mut partial_scratch)?;
                match row.chosen {
                    GemmPlan::SmallM { splits } => need_partial = need_partial.max(splits * m * e.n),
                    GemmPlan::SmallMma { splits, .. } if splits > 1 => {
                        need_partial = need_partial.max(splits * m * e.n);
                        need_counters = need_counters.max(e.n.div_ceil(8));
                    }
                    _ => {}
                }
                plans.insert(Key::bf16(m, e.n, e.k, e.bias.is_some()), row.chosen);
                report.rows.push(row);
            }
        }
        // 非桶 M 展开（eager 混批 / 请求交错完成时 decode 批常是 7、9、31 这类非 2 的幂）：借上一个 2 的幂桶的结论。
        // small_m 对任意 m ≤ 桶正确；Lt algo 对别的 M 是否合法要过 AlgoCheck（不合法就留给默认启发式）。
        // 行数按桶算 partial，覆盖 m < 桶。
        let mut borrowed = 0usize;
        let mut rejected = 0usize;
        for e in entries {
            let key_bias = e.bias.is_some();
            // 诊断模式（GEMM_TUNE_M 引入大 M）下不做非桶展开：1..=6019 会让 AlgoCheck 跑上万次。
            let expand_max = if tp_buckets.len() == tp.m_buckets.len() { m_max } else { 0 };
            for m in 1..=expand_max {
                if tp_buckets.contains(&m) {
                    continue;
                }
                let bucket = m.next_power_of_two();
                let Some(&bp) = plans.get(&Key::bf16(bucket, e.n, e.k, key_bias)) else { continue };
                let plan = match bp {
                    GemmPlan::SmallM { .. } | GemmPlan::SmallMma { .. } if m <= tp.small_m_max => Some(bp),
                    GemmPlan::SmallM { .. } | GemmPlan::SmallMma { .. } => None,
                    GemmPlan::LtAlgo(algo) => {
                        let d = self.make_descs(m, e.n, e.k, e.bias.map(|b| self.stream.ptr(b)))?;
                        self.algo_valid_for(&d, &algo).then_some(bp)
                    }
                };
                match plan {
                    Some(p) => {
                        plans.insert(Key::bf16(m, e.n, e.k, key_bias), p);
                        borrowed += 1;
                    }
                    None => rejected += 1,
                }
            }
        }
        report.borrowed = borrowed;
        report.rejected = rejected;
        drop(partial_scratch);
        if need_counters > 0 {
            *self.counters.lock().unwrap() = Some(self.stream.alloc_zeros::<i32>(need_counters)?);
        }
        if need_partial > 0 {
            *self.partial.lock().unwrap() = Some(self.stream.alloc_zeros::<f32>(need_partial)?);
            report.partial_bytes = need_partial * 4;
        }
        self.stream.synchronize()?;
        *self.plans.lock().unwrap() = plans;
        report.elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
        tracing::info!(
            shapes = entries.len(),
            buckets = tp.m_buckets.len(),
            elapsed_ms = format!("{:.0}", report.elapsed_ms),
            partial_mb = format!("{:.1}", report.partial_bytes as f64 / 1048576.0),
            borrowed = report.borrowed,
            rejected = report.rejected,
            "gemm autotune done"
        );
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    #[test]
    #[ignore = "needs a CUDA device"]
    fn gemm_matches_cpu_reference() {
        let dev = Device::new(0).unwrap();
        let s = dev.default_stream();
        let g = Gemm::new(&s).unwrap();
        let (m, n, k) = (5usize, 7usize, 64usize);
        let a: Vec<f32> = (0..m * k).map(|i| ((i * 37) % 23) as f32 / 23.0 - 0.5).collect();
        let w: Vec<f32> = (0..n * k).map(|i| ((i * 53) % 29) as f32 / 29.0 - 0.5).collect();
        let bias: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let ab: Vec<bf16> = a.iter().map(|&v| bf16::from_f32(v)).collect();
        let wb: Vec<bf16> = w.iter().map(|&v| bf16::from_f32(v)).collect();
        let bb: Vec<bf16> = bias.iter().map(|&v| bf16::from_f32(v)).collect();
        let da = s.htod_copy(ab.clone()).unwrap();
        let dw = s.htod_copy(wb.clone()).unwrap();
        let db = s.htod_copy(bb.clone()).unwrap();
        let mut dout = s.alloc_zeros::<bf16>(m * n).unwrap();
        g.forward_bf16(&da, &dw, Some(&db), &mut dout, m, n, k).unwrap();
        let out = s.dtoh_sync_copy(&dout).unwrap();
        for i in 0..m {
            for j in 0..n {
                let mut acc = bb[j].to_f32();
                for t in 0..k {
                    acc += ab[i * k + t].to_f32() * wb[j * k + t].to_f32();
                }
                let got = out[i * n + j].to_f32();
                assert!((got - acc).abs() <= 2e-2 + 2e-2 * acc.abs(), "({i},{j}) got={got} want={acc}");
            }
        }
    }
}
