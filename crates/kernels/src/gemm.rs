//! bf16 GEMM：`C[M,N] = A[M,K] · W[N,K]^T (+ bias[N])`，fp32 累加。
//!
//! 两条实现路径，按 (M, N, K, 有无 bias) 查自调优表选：
//! 1. cuBLASLt：默认每次调用让启发式选 algo（与 cudarc `CudaBlasLT::matmul` 等价）；自调优后用表里实测最快的 algo。
//! 2. 自写小 M kernel（`cuda/gemm_small_m.cu`，M ≤ 8，纯访存型，split-K 份数由表给）。
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
}

/// 某个 (M, N, K) 形状选定的执行方式。
#[derive(Clone, Copy, Debug)]
pub enum GemmPlan {
    /// cuBLASLt，指定 algo（自调优实测最快）。
    LtAlgo(sys::cublasLtMatmulAlgo_t),
    /// 自写小 M kernel，split-K 份数。
    SmallM { splits: usize },
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
}

// SAFETY: cublasLt 句柄按 NVIDIA 文档可跨线程使用；其余字段自身 Send/Sync（Mutex 包着可变状态）。
unsafe impl Send for Gemm {}
unsafe impl Sync for Gemm {}

impl Drop for Gemm {
    fn drop(&mut self) {
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
        Ok(Self {
            handle,
            stream: stream.clone(),
            workspace,
            profile,
            enabled,
            plans: Mutex::new(HashMap::new()),
            default_algos: Mutex::new(HashMap::new()),
            partial: Mutex::new(None),
        })
    }

    pub fn profile(&self) -> &ArchProfile {
        &self.profile
    }
    /// 自调优是否开启（`GEMM_TUNE=off` 关）。
    pub fn tune_enabled(&self) -> bool {
        self.enabled
    }

    /// out[M,N] = a[M,K] · w[N,K]^T，可选 bias[N]。
    #[allow(clippy::too_many_arguments)]
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
        if let Some(p) = plans.get(&Key { m, n, k, bias }) {
            return Some(*p);
        }
        // 非桶 M 在 autotune 末尾已按桶展开（1..=最大桶 每个 m 一条，Lt algo 过 AlgoCheck 才借）；
        // 这里只兜底 small_m（对任意 m ≤ 桶都正确）。
        if m <= self.profile.gemm_tune.small_m_max {
            let bucket = m.next_power_of_two();
            if let Some(p @ GemmPlan::SmallM { .. }) = plans.get(&Key { m: bucket, n, k, bias }) {
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
        let key = Key { m, n, k, bias: bias.is_some() };
        let cached = self.default_algos.lock().unwrap().get(&key).copied();
        let algo = match cached {
            Some(a) => a,
            None => {
                let h = self.heuristics(&d, 1)?;
                ensure!(!h.is_empty(), "cublasLt: no algo for m={m} n={n} k={k}");
                self.default_algos.lock().unwrap().insert(key, h[0].algo);
                h[0].algo
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
        let pref = lt_err(lt::create_matmul_pref(), "matmul pref")?;
        let ws = self.workspace.len();
        // SAFETY: 属性/缓冲区按 cublasLt 文档；pref 创建成功后在此销毁。
        let r = unsafe {
            lt::set_matmul_pref_attribute(
                pref,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &ws as *const usize as *const c_void,
                std::mem::size_of::<usize>(),
            )
            .map_err(|e| anyhow!("pref workspace: cublasLt {:?}", e.0))
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
        let chosen = match small_m_us {
            Some(su) if su < lt_best.0 => GemmPlan::SmallM { splits: small_m_splits },
            _ => GemmPlan::LtAlgo(cands[lt_best.1].algo),
        };
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
        let m_max = tp.m_buckets.iter().copied().max().unwrap_or(1);
        let k_max = entries.iter().map(|e| e.k).max().unwrap_or(0);
        let n_max = entries.iter().map(|e| e.n).max().unwrap_or(0);
        let a_scratch = self.stream.alloc_zeros::<bf16>(m_max * k_max)?;
        let mut out_scratch = self.stream.alloc_zeros::<bf16>(m_max * n_max)?;
        let mut partial_scratch = self.stream.alloc_zeros::<f32>(tp.split_workspace_max_bytes / 4)?;
        self.stream.synchronize()?;
        let mut need_partial = 0usize;
        let mut plans = HashMap::new();
        for e in entries {
            for &m in &tp.m_buckets {
                let row = self.tune_shape(&e.name, m, e.n, e.k, e.w, e.bias, &a_scratch, &mut out_scratch, &mut partial_scratch)?;
                if let GemmPlan::SmallM { splits } = row.chosen {
                    need_partial = need_partial.max(splits * m * e.n);
                }
                plans.insert(Key { m, n: e.n, k: e.k, bias: e.bias.is_some() }, row.chosen);
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
            for m in 1..=m_max {
                if tp.m_buckets.contains(&m) {
                    continue;
                }
                let bucket = m.next_power_of_two();
                let Some(&bp) = plans.get(&Key { m: bucket, n: e.n, k: e.k, bias: key_bias }) else { continue };
                let plan = match bp {
                    GemmPlan::SmallM { .. } if m <= tp.small_m_max => Some(bp),
                    GemmPlan::SmallM { .. } => None,
                    GemmPlan::LtAlgo(algo) => {
                        let d = self.make_descs(m, e.n, e.k, e.bias.map(|b| self.stream.ptr(b)))?;
                        self.algo_valid_for(&d, &algo).then_some(bp)
                    }
                };
                match plan {
                    Some(p) => {
                        plans.insert(Key { m, n: e.n, k: e.k, bias: key_bias }, p);
                        borrowed += 1;
                    }
                    None => rejected += 1,
                }
            }
        }
        report.borrowed = borrowed;
        report.rejected = rejected;
        drop(partial_scratch);
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
