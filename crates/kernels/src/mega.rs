//! Qwen3.5 GDN 层 decode 的单 persistent kernel（`cuda/mega_sm100.cu`）Rust 包装。**实验件**：不接 executor，
//! 只给 `tests/mega_parity.rs` 做对拍 / 计时（docs/loops/mega-sm100.md）。形状 / 调优数值全部在这里（host）算好传下去。
use crate::arch::MegaParams;
use crate::device::Stream;
use crate::ffi;
use anyhow::{bail, ensure, Result};
use cudarc::driver::CudaSlice;
use half::bf16;
use std::ffi::c_void;

/// 一层 GDN（含 MLP）的形状（模型无关：只有维度）。
#[derive(Clone, Copy, Debug)]
pub struct MegaShape {
    pub hidden: usize,
    pub inter: usize,
    pub heads_k: usize,
    pub heads_v: usize,
    pub k_dim: usize,
    pub v_dim: usize,
    pub width: usize,
    pub eps: f32,
    pub scale: f32,
}

impl MegaShape {
    pub fn conv_dim(&self) -> usize {
        2 * self.heads_k * self.k_dim + self.heads_v * self.v_dim
    }
    /// 融合 in_proj 行宽 `[qkv | z | a | b]`。
    pub fn in_row(&self) -> usize {
        self.conv_dim() + self.heads_v * self.v_dim + 2 * self.heads_v
    }
    pub fn state_stride_tok(&self) -> usize {
        self.heads_v * self.v_dim * self.k_dim
    }
}

pub struct MegaWeights<'a> {
    pub norm_w: &'a CudaSlice<bf16>,
    pub post_norm_w: &'a CudaSlice<bf16>,
    pub in_w: &'a CudaSlice<bf16>,
    pub conv_w: &'a CudaSlice<bf16>,
    pub a_log: &'a CudaSlice<f32>,
    pub dt_bias: &'a CudaSlice<f32>,
    pub gnorm_w: &'a CudaSlice<bf16>,
    pub out_w: &'a CudaSlice<bf16>,
    pub gate_up_w: &'a CudaSlice<bf16>,
    pub down_w: &'a CudaSlice<bf16>,
}

/// 中间量 + split-K partial + 计数器。计数器首次清零，之后 kernel 自复位。
pub struct MegaScratch {
    pub max_batch: usize,
    pub params: MegaParams,
    pub sm_count: usize,
    pub splits: [usize; 4],
    xn: CudaSlice<bf16>,
    qkvzab: CudaSlice<bf16>,
    conv_out: CudaSlice<bf16>,
    o: CudaSlice<bf16>,
    n: CudaSlice<bf16>,
    h2: CudaSlice<bf16>,
    xn2: CudaSlice<bf16>,
    gu: CudaSlice<bf16>,
    act: CudaSlice<bf16>,
    d: CudaSlice<bf16>,
    partial: CudaSlice<f32>,
    counters: CudaSlice<i32>,
    /// kernel host 侧回填 [grid, occupancy, smem_bytes]（launch 时同步写，host 内存）。
    pub info: Box<[i32; 3]>,
    /// 阶段计时（globaltimer ns）：`profile = true` 时 kernel 写 `prof[stage][4] = {~min_start, max_end, Σwait, Σbusy}`。
    pub profile: bool,
    prof: CudaSlice<u64>,
}

/// 一个阶段的 prof 读数（µs）。
#[derive(Clone, Copy, Debug, Default)]
pub struct StageProf {
    pub start_us: f64,
    pub end_us: f64,
    pub wait_us: f64,
    pub busy_us: f64,
}

pub const MEGA_STAGES: usize = 11;
pub const MEGA_STAGE_NAMES: [&str; MEGA_STAGES] =
    ["rmsnorm", "in_proj", "conv", "gdn", "gated_norm", "out_proj+add", "rmsnorm2", "gate_up+silu", "(silu)", "down+add", "(add)"];

impl MegaScratch {
    pub fn new(stream: &Stream, sh: &MegaShape, max_batch: usize, params: MegaParams, sm_count: usize) -> Result<Self> {
        ensure!(max_batch >= 1 && max_batch <= 32, "mega: batch ≤ 32");
        let (hid, inter, in_row) = (sh.hidden, sh.inter, sh.in_row());
        let hvv = sh.heads_v * sh.v_dim;
        // gate_up 的 tile 同时覆盖 gate 行与 up 行（kernel mode 2）：tile 数按 inter / (tile_n/2) = 2·inter / tile_n
        let gemvs = [(in_row, hid), (hid, hvv), (2 * inter, hid), (hid, inter)];
        let mut splits = [1usize; 4];
        let mut partial = 0usize;
        let mut tiles = 0usize;
        for (i, &(n, k)) in gemvs.iter().enumerate() {
            splits[i] = params.splits_for(sm_count, n, k);
            partial = partial.max(splits[i] * max_batch * n);
            tiles += n.div_ceil(params.tile_n());
        }
        let b = max_batch;
        Ok(Self {
            max_batch,
            params,
            sm_count,
            splits,
            xn: stream.alloc_zeros(b * hid)?,
            qkvzab: stream.alloc_zeros(b * in_row)?,
            conv_out: stream.alloc_zeros(b * sh.conv_dim())?,
            o: stream.alloc_zeros(b * hvv)?,
            n: stream.alloc_zeros(b * hvv)?,
            h2: stream.alloc_zeros(b * hid)?,
            xn2: stream.alloc_zeros(b * hid)?,
            gu: stream.alloc_zeros(b * 2 * inter)?,
            act: stream.alloc_zeros(b * inter)?,
            d: stream.alloc_zeros(b * hid)?,
            partial: stream.alloc_zeros(partial.max(1))?,
            counters: stream.alloc_zeros(12 + tiles)?,
            info: Box::new([0; 3]),
            profile: false,
            prof: stream.alloc_zeros(MEGA_STAGES * 4)?,
        })
    }

    pub fn reset_prof(&mut self, stream: &Stream) -> Result<()> {
        stream.memset_zeros_n(&mut self.prof, MEGA_STAGES * 4)
    }

    /// 读阶段计时（相对最早阶段起点，µs）。只在 eager 单次 launch 后读有意义。
    pub fn read_prof(&self, stream: &Stream) -> Result<Vec<StageProf>> {
        let raw = stream.dtoh_sync_copy(&self.prof)?;
        let base = (0..MEGA_STAGES).map(|s| !raw[s * 4]).min().unwrap_or(0);
        Ok((0..MEGA_STAGES)
            .map(|s| {
                let st = !raw[s * 4];
                StageProf {
                    start_us: st.saturating_sub(base) as f64 * 1e-3,
                    end_us: raw[s * 4 + 1].saturating_sub(base) as f64 * 1e-3,
                    wait_us: raw[s * 4 + 2] as f64 * 1e-3,
                    busy_us: raw[s * 4 + 3] as f64 * 1e-3,
                }
            })
            .collect())
    }
}

/// 一层 GDN decode（B 个请求各 1 token）：`res` 原位更新为层输出，conv_state / rec_state 原位推进。
/// 返回 `Err` 含 "not supported" 表示本构建未编入 mega kernel（CUDA_ARCHS 无 100/103）。
#[allow(clippy::too_many_arguments)]
pub fn mega_gdn_layer_decode(
    stream: &Stream,
    sh: &MegaShape,
    w: &MegaWeights<'_>,
    sc: &mut MegaScratch,
    res: &mut CudaSlice<bf16>,
    conv_state: &mut CudaSlice<bf16>,
    rec_state: &mut CudaSlice<f32>,
    state_indices: Option<&CudaSlice<i32>>,
    batch: usize,
) -> Result<()> {
    ensure!(batch >= 1 && batch <= sc.max_batch, "mega: batch {batch} > max {}", sc.max_batch);
    let (hid, inter, in_row) = (sh.hidden, sh.inter, sh.in_row());
    ensure!(res.len() >= batch * hid, "mega: res");
    ensure!(w.in_w.len() == in_row * hid && w.out_w.len() == hid * sh.heads_v * sh.v_dim, "mega: in/out 权重形状");
    ensure!(w.gate_up_w.len() == 2 * inter * hid && w.down_w.len() == hid * inter, "mega: mlp 权重形状");
    ensure!(w.conv_w.len() == sh.conv_dim() * sh.width && w.a_log.len() == sh.heads_v && w.dt_bias.len() == sh.heads_v);
    ensure!(w.norm_w.len() == hid && w.post_norm_w.len() == hid && w.gnorm_w.len() == sh.v_dim, "mega: norm 权重");
    if let Some(si) = state_indices {
        ensure!(si.len() >= batch, "mega: state_indices");
    }
    let p = sc.params;
    let bf = |s: &CudaSlice<bf16>| stream.ptr(s) as *const c_void;
    let args = ffi::MegaLayerArgs {
        res: stream.ptr_mut(res) as *const c_void,
        norm_w: bf(w.norm_w),
        post_norm_w: bf(w.post_norm_w),
        in_w: bf(w.in_w),
        conv_w: bf(w.conv_w),
        a_log: stream.ptr(w.a_log) as *const f32,
        dt_bias: stream.ptr(w.dt_bias) as *const f32,
        gnorm_w: bf(w.gnorm_w),
        out_w: bf(w.out_w),
        gate_up_w: bf(w.gate_up_w),
        down_w: bf(w.down_w),
        conv_state: stream.ptr_mut(conv_state) as *mut c_void,
        rec_state: stream.ptr_mut(rec_state) as *mut c_void,
        state_indices: state_indices.map(|s| stream.ptr(s) as *const i32).unwrap_or(std::ptr::null()),
        xn: stream.ptr_mut(&mut sc.xn) as *mut c_void,
        qkvzab: stream.ptr_mut(&mut sc.qkvzab) as *mut c_void,
        conv_out: stream.ptr_mut(&mut sc.conv_out) as *mut c_void,
        o: stream.ptr_mut(&mut sc.o) as *mut c_void,
        n: stream.ptr_mut(&mut sc.n) as *mut c_void,
        h2: stream.ptr_mut(&mut sc.h2) as *mut c_void,
        xn2: stream.ptr_mut(&mut sc.xn2) as *mut c_void,
        gu: stream.ptr_mut(&mut sc.gu) as *mut c_void,
        act: stream.ptr_mut(&mut sc.act) as *mut c_void,
        d: stream.ptr_mut(&mut sc.d) as *mut c_void,
        partial: stream.ptr_mut(&mut sc.partial) as *mut f32,
        counters: stream.ptr_mut(&mut sc.counters) as *mut i32,
        info: sc.info.as_mut_ptr(),
        prof: if sc.profile { stream.ptr_mut(&mut sc.prof) as *mut u64 } else { std::ptr::null_mut() },
        batch: batch as i64,
        hidden: hid as i64,
        inter: inter as i64,
        heads_k: sh.heads_k as i64,
        heads_v: sh.heads_v as i64,
        k_dim: sh.k_dim as i64,
        v_dim: sh.v_dim as i64,
        width: sh.width as i64,
        state_stride_tok: sh.state_stride_tok() as i64,
        eps: sh.eps,
        scale: sh.scale,
        groups: p.groups as i64,
        unroll: p.unroll as i64,
        stages: p.stages as i64,
        ctas_per_sm: p.ctas_per_sm as i64,
        splits_in: sc.splits[0] as i64,
        splits_out: sc.splits[1] as i64,
        splits_gu: sc.splits[2] as i64,
        splits_down: sc.splits[3] as i64,
        gdn_hv: p.gdn_hv_for(sc.sm_count, batch, sh.heads_v) as i64,
        gdn_bufs: p.gdn_bufs as i64,
    };
    // SAFETY: 所有指针来自存活的 CudaSlice，长度已校验；info 指向 sc 内 Box，kernel host 侧同步写。
    let rc = unsafe { ffi::mega_gdn_layer_decode_sm100(&args, stream.raw() as _) };
    if rc == ffi::CUDA_ERROR_NOT_SUPPORTED {
        bail!("mega_gdn_layer_decode_sm100: not supported in this build (CUDA_ARCHS 无 100/103)");
    }
    if rc != 0 {
        bail!("mega_gdn_layer_decode_sm100 failed: cudaError {rc}");
    }
    Ok(())
}
