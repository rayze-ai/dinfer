//! Gated DeltaNet 层（Qwen3.5 的 48 个线性注意力层）。数学规格见 `docs/loops/qwen35-gdn.md`（kernel 已三卡对拍）：
//!
//! `x → in_proj_qkv [q|k|v] → conv1d(silu, w=4) → in_proj_z / a / b → gate(g, beta) → recurrent(delta rule)
//!   → gated_rmsnorm(z, silu) → out_proj`
//!
//! 状态（每层每 slot）：conv `[slots, conv_dim, width-1]` bf16 + 递推 `[slots, HV, V, K]` f32，全部由本 crate 自管
//! （`SlotTable` 按 `RequestId` 分 slot，新请求首次出现时清零）；core 的 `KvCacheSpec::Mamba` 接法留 §3 待办。
//!
//! prefill（`t > num_reqs`）走 chunk 三段（`gdn_chunk_{prep,wy,ho}`，`docs/loops/qwen35-perf.md` 第 1 轮），
//! decode 单 token 走 `gdn_recurrent`；`GDN_PREFILL=recurrent` 回退到老路径。chunk 中间缓冲全模型共用一份
//! （`GdnChunkWorkspace`，executor 每步 `prepare` 一次，缓冲按需增长）。
use crate::config::Qwen35Config;
use crate::scratch::ActScratch;
use crate::weights::Table;
use anyhow::{ensure, Result};
use dinfer_core::runner::BatchMode;
use dinfer_kernels as k;
use std::cell::RefCell;
use dinfer_layers::{bf16, Gemm, Linear, Stream, Tensor, WeightSource};

/// 一格状态（第 `l` 层的状态张量）。
pub struct GdnState {
    /// `[slots, conv_dim, width-1]` bf16（列 0 最旧）。
    pub conv: Tensor<bf16>,
    /// `[slots, HV, V, K]` f32。
    pub rec: Tensor<f32>,
    /// 快照区（design §15.7，prefix cache 对齐位置的状态；`alloc_snapshots` 之后才有）：`[snapshots, conv_dim, width-1]`。
    pub snap_conv: Option<Tensor<bf16>>,
    /// `[snapshots, HV, V, K]` f32。
    pub snap_rec: Option<Tensor<f32>>,
}

impl GdnState {
    fn per_slot(&self) -> (usize, usize) {
        (self.rec.numel() / self.rec.shape[0], self.conv.numel() / self.conv.shape[0])
    }
    /// 活槽 `slot` → 快照槽 `snap`（同流 dtod；prefill 步 forward 之后调用）。
    pub fn save_snapshot(&mut self, stream: &Stream, slot: usize, snap: usize) -> Result<()> {
        let (n, cn) = self.per_slot();
        let (sr, sc) = (self.snap_rec.as_mut(), self.snap_conv.as_mut());
        let (Some(sr), Some(sc)) = (sr, sc) else { anyhow::bail!("gdn: 快照区未分配") };
        ensure!(snap < sr.shape[0], "gdn: snapshot slot {snap} >= {}", sr.shape[0]);
        let src = self.rec.data.slice(slot * n..(slot + 1) * n);
        let mut dst = sr.data.slice_mut(snap * n..(snap + 1) * n);
        stream.inner_arc().memcpy_dtod(&src, &mut dst)?;
        let src = self.conv.data.slice(slot * cn..(slot + 1) * cn);
        let mut dst = sc.data.slice_mut(snap * cn..(snap + 1) * cn);
        stream.inner_arc().memcpy_dtod(&src, &mut dst)?;
        Ok(())
    }
    /// 快照槽 `snap` → 活槽 `slot`（forward 之前调用；之后该请求 `has_initial = 1`）。
    pub fn restore_snapshot(&mut self, stream: &Stream, snap: usize, slot: usize) -> Result<()> {
        let (n, cn) = self.per_slot();
        let (Some(sr), Some(sc)) = (self.snap_rec.as_ref(), self.snap_conv.as_ref()) else {
            anyhow::bail!("gdn: 快照区未分配")
        };
        ensure!(snap < sr.shape[0], "gdn: snapshot slot {snap} >= {}", sr.shape[0]);
        let src = sr.data.slice(snap * n..(snap + 1) * n);
        let mut dst = self.rec.data.slice_mut(slot * n..(slot + 1) * n);
        stream.inner_arc().memcpy_dtod(&src, &mut dst)?;
        let src = sc.data.slice(snap * cn..(snap + 1) * cn);
        let mut dst = self.conv.data.slice_mut(slot * cn..(slot + 1) * cn);
        stream.inner_arc().memcpy_dtod(&src, &mut dst)?;
        Ok(())
    }
    pub fn num_snapshots(&self) -> usize {
        self.snap_rec.as_ref().map_or(0, |t| t.shape[0])
    }
}

/// 步内元数据（executor 上传后的设备张量）。
pub struct GdnMeta<'a> {
    pub cu_seqlens: &'a Tensor<i32>,
    pub state_indices: &'a Tensor<i32>,
    pub has_initial: &'a Tensor<i32>,
    pub num_reqs: usize,
    pub mode: BatchMode,
    /// chunk prefill 的共享工作区；`None` = 本步走顺序递归（decode / 回退）。
    pub chunk: Option<&'a RefCell<GdnChunkWorkspace>>,
    /// 递归状态边界快照表（design §15.7 r8b；`None` = 本步无快照）：序列 i 的条目在 `pos / slot[off[i]..off[i+1])`，
    /// `pos` = 相对该序列段首的 token 偏移。chunk prefill kernel 在 chunk 边界写 rec 状态；conv 状态由
    /// `gdn_conv_snapshot_strided` 在短卷积之前写。只在 chunk 路径有效（顺序递归路径遇到它报错）。
    pub snapshot: Option<GdnSnapshotMeta<'a>>,
}

/// 见 `GdnMeta::snapshot`。
#[derive(Clone, Copy)]
pub struct GdnSnapshotMeta<'a> {
    pub off: &'a Tensor<i32>,
    pub pos: &'a Tensor<i32>,
    pub slot: &'a Tensor<i32>,
}

/// chunk prefill 的切块表 + 中间缓冲，全模型 48 层共用一份（每层串行，无重入）。
/// 块大小：`BT`（kernel 当前只支持 64；由 host 选，不是 kernel 内常量）。
pub struct GdnChunkWorkspace {
    pub plan: Option<k::ops::KdaChunkPlan>,
    pub scratch: Option<k::ops::GdnChunkScratch>,
    pub tokens: usize,
    bt: usize,
}

impl GdnChunkWorkspace {
    pub const BT: usize = 64;

    pub fn new() -> Self {
        Self { plan: None, scratch: None, tokens: 0, bt: Self::BT }
    }

    /// 启动一次按上限分配（engine-gap 第 7 轮）：`max_tokens` 个 token、`max_seqs` 条序列的最坏块数
    /// （每条序列最多多出一个不满的块）。之后 `prepare` 只重填切块表，不再 free + alloc。
    /// 字节数：`scratch_bytes(cfg, max_tokens, max_seqs)`。
    pub fn with_capacity(stream: &Stream, cfg: &Qwen35Config, max_tokens: usize, max_seqs: usize) -> Result<Self> {
        let bt = Self::BT;
        let h = cfg.linear_num_key_heads as usize;
        let hv = cfg.linear_num_value_heads as usize;
        let (kd, vd) = (cfg.linear_key_head_dim as usize, cfg.linear_value_head_dim as usize);
        let max_chunks = Self::max_chunks(max_tokens, max_seqs);
        let scratch = k::ops::GdnChunkScratch::new(stream, max_tokens.max(1), h, hv, kd, vd, max_chunks.max(1), bt)?;
        let plan = k::ops::KdaChunkPlan::with_capacity(stream, max_chunks, max_seqs.max(1), bt)?;
        Ok(Self { plan: Some(plan), scratch: Some(scratch), tokens: 0, bt })
    }

    fn max_chunks(max_tokens: usize, max_seqs: usize) -> usize {
        max_tokens.div_ceil(Self::BT) + max_seqs
    }

    /// `with_capacity` 会占用的设备字节数（供日志 / 预算核对）。
    pub fn scratch_bytes(cfg: &Qwen35Config, max_tokens: usize, max_seqs: usize) -> usize {
        let h = cfg.linear_num_key_heads as usize;
        let hv = cfg.linear_num_value_heads as usize;
        let (kd, vd) = (cfg.linear_key_head_dim as usize, cfg.linear_value_head_dim as usize);
        let t = max_tokens.max(1);
        let c = Self::max_chunks(max_tokens, max_seqs).max(1);
        // qn/kn bf16 [T,H,K] ×2；gc f32 [T,HV]；aq f32 [C,HV,BT,BT]；u f32 [T,HV,V]；w/kg f32 [T,HV,K] ×2
        2 * (t * h * kd * 2) + t * hv * 4 + c * hv * Self::BT * Self::BT * 4 + t * hv * vd * 4 + 2 * (t * hv * kd * 4)
    }

    /// 每步一次：按 host 侧 `cu_seqlens` 重填切块表；缓冲不够才按本步形状重分（只增不减；
    /// `with_capacity` 分配过的正常不会触发）。
    pub fn prepare(&mut self, stream: &Stream, cu_seqlens: &[i32], cfg: &Qwen35Config) -> Result<()> {
        match self.plan.as_mut() {
            Some(plan) => plan.fill(stream, cu_seqlens)?,
            None => self.plan = Some(k::ops::KdaChunkPlan::new(stream, cu_seqlens, self.bt)?),
        }
        let num_chunks = self.plan.as_ref().map_or(0, |p| p.num_chunks);
        let tokens = (*cu_seqlens.last().unwrap_or(&0)).max(0) as usize;
        let need_realloc = match &self.scratch {
            Some(sc) => !sc.fits(tokens, num_chunks),
            None => true,
        };
        if need_realloc {
            let h = cfg.linear_num_key_heads as usize;
            let hv = cfg.linear_num_value_heads as usize;
            let (kd, vd) = (cfg.linear_key_head_dim as usize, cfg.linear_value_head_dim as usize);
            let old = self.scratch.as_ref().map_or((0, 0), |s| (s.tokens, s.num_chunks));
            let cap_t = tokens.max(old.0);
            let cap_c = num_chunks.max(old.1);
            if old != (0, 0) {
                tracing::debug!(tokens, num_chunks, old_tokens = old.0, old_chunks = old.1, "gdn chunk workspace: 重分配（预分配容量不够）");
            }
            self.scratch = None; // 先释放旧的再分新的
            self.scratch = Some(k::ops::GdnChunkScratch::new(stream, cap_t, h, hv, kd, vd, cap_c, self.bt)?);
        }
        self.tokens = tokens;
        Ok(())
    }
}

impl Default for GdnChunkWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

pub struct GdnLayer {
    /// 融合投影 `[qkv(conv_dim) | z(HV·V) | a(HV) | b(HV)]`，行宽 `in_row`（第 9 轮，vLLM `in_proj_qkvz` 同做法）：
    /// 一次 GEMM 读 A 一遍，下游 conv1d / gate / gated_rmsnorm 按列段 + 行距读，不拷贝。
    in_proj: Linear,
    in_row: usize,
    conv_w: Tensor<bf16>, // [conv_dim, width]
    a_log: Tensor<f32>,   // [HV]
    dt_bias: Tensor<f32>, // [HV]
    norm_w: Tensor<bf16>, // [V]
    out_proj: Linear,
    pub state: GdnState,
    rec_zeros: Option<Tensor<f32>>,
    conv_zeros: Option<Tensor<bf16>>,
}

impl GdnLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        stream: &Stream,
        src: &WeightSource,
        t: &Table,
        layer: usize,
        cfg: &Qwen35Config,
        max_tokens: usize,
        num_slots: usize,
    ) -> Result<Self> {
        let p = format!("layers.{layer}.linear_attn");
        let hv = cfg.linear_num_value_heads as usize;
        let h = cfg.linear_num_key_heads as usize;
        let (kd, vd) = (cfg.linear_key_head_dim as usize, cfg.linear_value_head_dim as usize);
        let width = cfg.linear_conv_kernel_dim as usize;
        let conv_dim = 2 * h * kd + hv * vd;
        let hidden = cfg.hidden_size as usize;

        // 四个投影按行拼成一个权重（TP 时每段各自按头切后再拼；本 crate 目前 world == 1）
        let in_qkv = t.load_bf16(stream, src, &format!("{p}.in_proj_qkv.weight"))?;
        ensure!(in_qkv.shape == [conv_dim, hidden], "in_proj_qkv shape {:?}", in_qkv.shape);
        let in_z = t.load_bf16(stream, src, &format!("{p}.in_proj_z.weight"))?;
        ensure!(in_z.shape == [hv * vd, hidden], "in_proj_z shape {:?}", in_z.shape);
        let in_a = t.load_bf16(stream, src, &format!("{p}.in_proj_a.weight"))?;
        let in_b = t.load_bf16(stream, src, &format!("{p}.in_proj_b.weight"))?;
        ensure!(in_a.shape == [hv, hidden] && in_b.shape == [hv, hidden], "in_proj_a/b shape {:?} {:?}", in_a.shape, in_b.shape);
        let in_row = conv_dim + hv * vd + 2 * hv;
        let mut w = Tensor::<bf16>::zeros(stream, &[in_row, hidden])?;
        {
            let m = stream.inner_arc();
            let mut off = 0usize;
            for part in [&in_qkv, &in_z, &in_a, &in_b] {
                let n = part.numel();
                let mut dst = w.data.slice_mut(off..off + n);
                m.memcpy_dtod(&part.data, &mut dst)?;
                off += n;
            }
            stream.synchronize()?;
        }
        drop((in_qkv, in_z, in_a, in_b));
        let in_proj = Linear::new(w, None)?;
        let conv_w = crate::weights::load_conv1d(stream, src, t, layer, conv_dim, width)?;
        let a_log = t.load_f32(stream, src, &format!("{p}.A_log"))?;
        let dt_bias = t.load_f32(stream, src, &format!("{p}.dt_bias"))?;
        ensure!(a_log.numel() == hv && dt_bias.numel() == hv, "A_log/dt_bias {}/{} != HV {hv}", a_log.numel(), dt_bias.numel());
        let norm_w = t.load_bf16(stream, src, &format!("{p}.norm.weight"))?;
        ensure!(norm_w.numel() == vd, "norm.weight {} != V {vd}", norm_w.numel());
        let out_proj = t.linear(stream, src, &format!("{p}.out_proj.weight"), None)?;
        ensure!(out_proj.w.shape == [hidden, hv * vd], "out_proj shape {:?}", out_proj.w.shape);

        let st = cfg.linear_conv_kernel_dim as usize - 1;
        let state = GdnState {
            // 多留 1 个 slot（下标 num_slots）作 CUDA graph 捕获 / padding 行的废纸篓：decode kernel 总是读写
            // 状态，捕获与 padding 行都指到它，才不会碰活请求的 slot（见 executor `scratch_slot`）。
            conv: Tensor::zeros(stream, &[num_slots + 1, conv_dim, st])?,
            rec: Tensor::zeros(stream, &[num_slots + 1, hv, vd, kd])?,
            snap_conv: None,
            snap_rec: None,
        };
        let _ = max_tokens; // 激活缓冲改为全模型共享（`crate::scratch::ActScratch`），不再按层预分配
        Ok(Self {
            in_proj,
            in_row,
            conv_w,
            a_log,
            dt_bias,
            norm_w,
            out_proj,
            state,
            rec_zeros: Some(Tensor::zeros(stream, &[hv, vd, kd])?),
            conv_zeros: Some(Tensor::zeros(stream, &[conv_dim, st])?),
        })
    }

    /// 分配 `n` 个快照槽（design §15.7；不清零：写快照前不会被读）。
    pub fn alloc_snapshots(&mut self, stream: &Stream, n: usize) -> Result<()> {
        let hv = self.state.rec.shape[1];
        let (vd, kd) = (self.state.rec.shape[2], self.state.rec.shape[3]);
        let (conv_dim, st) = (self.state.conv.shape[1], self.state.conv.shape[2]);
        self.state.snap_rec = Some(Tensor::uninit(stream, &[n, hv, vd, kd])?);
        self.state.snap_conv = Some(Tensor::uninit(stream, &[n, conv_dim, st])?);
        Ok(())
    }

    /// 清零 slot `s` 的两份状态（新请求首次出现时调用）。
    pub fn reset_slot(&mut self, stream: &Stream, s: usize, cfg: &Qwen35Config) -> Result<()> {
        let hv = cfg.linear_num_value_heads as usize;
        let h = cfg.linear_num_key_heads as usize;
        let (kd, vd) = (cfg.linear_key_head_dim as usize, cfg.linear_value_head_dim as usize);
        let conv_dim = 2 * h * kd + hv * vd;
        let st = cfg.linear_conv_kernel_dim as usize - 1;
        // 只清本 slot（整块 memset 会抹掉其他请求的状态）：从常驻零张量 dtod 一份。
        // 每层每 slot ≈ 3.1 MB f32 + 60 KB bf16（真 27B），仅新请求首次出现时走一次。
        let _ = (conv_dim, st);
        let n = hv * vd * kd;
        {
            let zeros = self.rec_zeros.as_ref().expect("reset_slot: zero buffer");
            let mut dst = self.state.rec.data.slice_mut(s * n..(s + 1) * n);
            stream.inner_arc().memcpy_dtod(&zeros.data, &mut dst)?;
        }
        let cn = self.state.conv.numel() / self.state.conv.shape[0];
        let zeros_c = self.conv_zeros.as_ref().expect("reset_slot: zero buffer");
        let mut cd = self.state.conv.data.slice_mut(s * cn..(s + 1) * cn);
        stream.inner_arc().memcpy_dtod(&zeros_c.data, &mut cd)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    /// 本层所有 Linear（GEMM 自调优用）。
    pub fn linears(&self) -> Vec<(&'static str, &Linear)> {
        vec![("gdn.in_proj", &self.in_proj), ("gdn.out_proj", &self.out_proj)]
    }

    pub fn forward(
        &mut self,
        stream: &Stream,
        gemm: &Gemm,
        act: &impl dinfer_layers::Act,
        cfg: &Qwen35Config,
        t: usize,
        x: &Tensor<bf16>,
        meta: &GdnMeta<'_>,
        abuf: &mut ActScratch,
        out: &mut Tensor<bf16>,
    ) -> Result<()> {
        let hv = cfg.linear_num_value_heads as usize;
        let h = cfg.linear_num_key_heads as usize;
        let (kd, vd) = (cfg.linear_key_head_dim as usize, cfg.linear_value_head_dim as usize);
        let width = cfg.linear_conv_kernel_dim as usize;
        let conv_dim = 2 * h * kd + hv * vd;
        let group = hv / h;
        ensure!(t <= abuf.max_tokens, "gdn: {t} tokens > max {}", abuf.max_tokens);
        ensure!(meta.state_indices.numel() >= meta.num_reqs, "gdn: state_indices");

        // 1. 融合投影：abuf.qkv[t, in_row] = [qkv | z | a | b]；下游按列段 + 行距 in_row 读
        let row = self.in_row;
        ensure!(row == conv_dim + hv * vd + 2 * hv && abuf.qkv.numel() >= t * row, "gdn: in_proj 行宽 / 缓冲");
        self.in_proj.forward_rows(gemm, x, &mut abuf.qkv, t)?;
        let (z_off, a_off, b_off) = (conv_dim, conv_dim + hv * vd, conv_dim + hv * vd + hv);
        let x_qkv = abuf.qkv.data.slice(..);
        let z_view = abuf.qkv.data.slice(z_off..);
        let a_view = abuf.qkv.data.slice(a_off..);
        let b_view = abuf.qkv.data.slice(b_off..);

        // `GDN_PROFILE=1`：GDN 内部五个阶段的 GPU 时间（CUDA 事件）。
        // 只在 chunk prefill 且 t 大时开；每层 6 个事件，用完即同步读。
        let gprof = std::env::var("GDN_PROFILE").ok().as_deref() == Some("1")
            && meta.mode != BatchMode::Decode
            && t >= 256;
        let gev = if gprof { Some(stream.new_timing_events(6)?) } else { None };
        if let Some(e) = &gev {
            stream.record_timing(&e[0])?;
        }

        // 2. 短卷积（decode 单 token 走 update，其余走 varlen prefill）；输出紧凑 [t, conv_dim]
        let single = meta.mode == BatchMode::Decode && meta.num_reqs == t;
        // r8b：边界快照先写 conv 状态（要在 conv1d_prefill 覆盖活槽之前；从原始输入行 + 入口状态取）
        if let Some(sn) = meta.snapshot.as_ref() {
            let sc_snap = self.state.snap_conv.as_mut().ok_or_else(|| anyhow::anyhow!("gdn: 快照区未分配"))?;
            k::gdn_conv_snapshot_strided(
                stream,
                &x_qkv,
                &self.state.conv.data,
                &mut sc_snap.data,
                &meta.cu_seqlens.data,
                Some(&meta.state_indices.data),
                Some(&meta.has_initial.data),
                &sn.off.data,
                &sn.pos.data,
                &sn.slot.data,
                meta.num_reqs,
                conv_dim,
                row,
                width - 1,
            )?;
        }
        if single {
            k::gdn_conv1d_update_strided(
                stream,
                &x_qkv,
                &self.conv_w.data,
                &mut abuf.conv.data,
                &mut self.state.conv.data,
                Some(&meta.state_indices.data),
                meta.num_reqs,
                conv_dim,
                row,
                width,
                width - 1,
                k::GdnAct::Silu,
            )?;
        } else {
            let rows = k::gdn_conv_rows(stream, t, conv_dim, width, width - 1);
            k::gdn_conv1d_prefill_strided(
                stream,
                &x_qkv,
                &self.conv_w.data,
                &mut abuf.conv.data,
                &mut self.state.conv.data,
                &meta.cu_seqlens.data,
                Some(&meta.state_indices.data),
                Some(&meta.has_initial.data),
                meta.num_reqs,
                t,
                conv_dim,
                row,
                width,
                width - 1,
                k::GdnAct::Silu,
                rows,
            )?;
        }

        if let Some(e) = &gev {
            stream.record_timing(&e[1])?;
        }
        // 3. gate / beta（decode 走 recurrent kernel 内 fused-gate 现算，省一个 launch；prefill 预算好给 chunk 用）
        //    chunk 路径把 gate/beta 与块内 cumsum 融进一次 launch（`gdn_gate_cumsum`，
        //    对齐 vLLM FLA 的 fused_kda_gate_chunk_cumsum），所以这里只处理非 chunk 的 prefill。
        let use_chunk = !single && t > meta.num_reqs && meta.chunk.is_some();
        if !single && !use_chunk {
            k::gdn_gate_strided(
                stream,
                &a_view,
                &b_view,
                &self.a_log.data,
                &self.dt_bias.data,
                &mut abuf.g.data,
                &mut abuf.beta.data,
                t,
                hv,
                row,
            )?;
        }

        if let Some(e) = &gev {
            stream.record_timing(&e[2])?;
        }
        // 4. delta rule 核心（q/k 的 l2norm 与 q 的 scale 在 kernel 内）：
        //    prefill 走 chunk 三段（prep → wy → ho），decode / 回退走逐 token 递归。两者数值等价（gdn_parity chunk case）。
        let scale = (kd as f32).powf(-0.5);
        if use_chunk {
            let mut ws = meta.chunk.unwrap().borrow_mut();
            let ws = &mut *ws;
            let plan = ws.plan.as_ref().ok_or_else(|| anyhow::anyhow!("gdn: chunk workspace 未 prepare"))?;
            let sc = ws.scratch.as_mut().ok_or_else(|| anyhow::anyhow!("gdn: chunk scratch 未分配"))?;
            ensure!(plan.num_seqs == meta.num_reqs && ws.tokens == t, "gdn: chunk plan 与本步不符");
            tracing::trace!(t, num_reqs = meta.num_reqs, chunks = plan.num_chunks, "gdn prefill: chunk 路径");
            // gate + beta + 块内 cumsum 一次算完（gc 直接写进 scratch），prep 只做 q/k 的 norm。
            k::gdn_gate_cumsum_strided(
                stream,
                &a_view,
                &b_view,
                &self.a_log.data,
                &self.dt_bias.data,
                &mut abuf.g.data,
                &mut abuf.beta.data,
                &mut sc.gc,
                &meta.cu_seqlens.data,
                &plan.chunk_tab,
                plan.num_chunks,
                hv,
                plan.bt,
                row,
            )?;
            k::gdn_chunk_prep(stream, &abuf.conv.data, &abuf.g.data, &meta.cu_seqlens.data, plan, sc, t, h, hv, kd, vd, scale, /*do_cumsum=*/false)?;
            if let Some(e) = &gev {
                stream.record_timing(&e[3])?;
            }
            k::gdn_chunk_wy(stream, &abuf.conv.data, &abuf.beta.data, &meta.cu_seqlens.data, plan, sc, t, h, hv, kd, vd)?;
            if let Some(e) = &gev {
                stream.record_timing(&e[4])?;
            }
            let snap = match meta.snapshot.as_ref() {
                Some(sn) => Some(k::ops::GdnSnapshot {
                    off: &sn.off.data,
                    pos: &sn.pos.data,
                    slot: &sn.slot.data,
                    state: &mut self.state.snap_rec.as_mut().ok_or_else(|| anyhow::anyhow!("gdn: 快照区未分配"))?.data,
                }),
                None => None,
            };
            // r11d：ho 段可走 tcgen05（UMMA）版 —— BV = 全 V、h 主本在 state、A 补零到 M=128。
            // 选路见 `ArchProfile::gdn_ho_umma`（Sm100 且 K=V=128 才开，`GDN_HO_UMMA` 可覆盖）。
            // tcgen05 版暂不支持步内快照（`snap`），有快照时回退 mma.sync 路径。
            // 架构按**实际编译目标**判：`gdn_ho_sm100.cu` 只在 CUDA_ARCHS 含 100/103 时编入，
            // 其它目标下 `ffi::gdn_ho_umma` 是返回 CUDA_ERROR_NOT_SUPPORTED 的桩。
            // 架构按**编译目标**判：没编入 sm100 文件时 `ArchProfile::has_sm100()` 为 false，
        // `ffi::gdn_ho_umma` 也只是返回 NotSupported 的桩，这里直接走 mma.sync。
        // 卡型分支在 `ArchProfile::gdn_ho_umma` 里（Sm100 开、Sm90/Sm8x 关）。
        let use_umma = snap.is_none()
            && k::arch::ArchProfile::gdn_ho_umma(k::arch::Arch::Sm100, kd, vd);
            if use_umma {
                let stride = (self.state.rec.numel() / self.state.rec.shape[0] / hv) as i64;
                k::ops::gdn_ho_umma(
                    stream,
                    sc,
                    &mut abuf.o.data,
                    &mut self.state.rec.data,
                    &meta.cu_seqlens.data,
                    &plan.cu_nt,
                    Some(&meta.state_indices.data),
                    stride,
                    meta.num_reqs,
                    h,
                    hv,
                    kd,
                    vd,
                )?;
            } else {
                k::gdn_chunk_ho_snap(
                    stream,
                    &meta.cu_seqlens.data,
                    plan,
                    sc,
                    &mut abuf.o.data,
                    &mut self.state.rec.data,
                    Some(&meta.state_indices.data),
                    t,
                    h,
                    hv,
                    kd,
                    vd,
                    snap,
                )?;
            }
            if let Some(e) = &gev {
                stream.record_timing(&e[5])?;
                stream.synchronize()?;
                let ms = |a: usize, b: usize| k::TimingEvent::elapsed_ms(&e[a], &e[b]).unwrap_or(0.0);
                let (conv, gate, prep, wy, ho) =
                    (ms(0, 1), ms(1, 2), ms(2, 3), ms(3, 4), ms(4, 5));
                eprintln!(
                    "[gdn-prof] t={t} | conv={conv:.3} gate={gate:.3} prep={prep:.3} wy={wy:.3} ho={ho:.3} total={:.3} ms",
                    ms(0, 5)
                );
            }
        } else {
            ensure!(meta.snapshot.is_none(), "gdn: 边界快照只支持 chunk prefill 路径（GDN_PREFILL=recurrent 不支持步内快照）");
            let (g, beta, a_raw, b_raw) = if single {
                (None, None, Some(&a_view), Some(&b_view))
            } else {
                (Some(&abuf.g.data), Some(&abuf.beta.data), None, None)
            };
            k::gdn_recurrent_strided(
                stream,
                &abuf.conv.data,
                g,
                beta,
                a_raw,
                b_raw,
                row,
                &self.a_log.data,
                &self.dt_bias.data,
                &mut abuf.o.data,
                &mut self.state.rec.data,
                &meta.cu_seqlens.data,
                Some(&meta.state_indices.data),
                meta.num_reqs,
                t,
                h,
                hv,
                kd,
                vd,
                scale,
                None,
                None,
                None,
            )?;
        }

        // 5. 输出门 + RMSNorm（z 门，silu）；z 是融合输出的列段，行距 in_row
        k::gdn_gated_rmsnorm_strided(
            stream,
            &abuf.o.data,
            &self.norm_w.data,
            &z_view,
            &mut abuf.n.data,
            t * hv,
            vd,
            cfg.rms_norm_eps,
            k::GdnAct::Silu,
            hv,
            row,
        )?;

        // 6. out_proj
        let _ = act;
        let _ = group;
        self.out_proj.forward_rows(gemm, &abuf.n, out, t)
    }
}

/// 请求 → slot 分配表：公共件 `dinfer_layers::exec::StateSlotTable`（qwen35 GDN 与 kimik3 KDA 同一套）。
pub use dinfer_layers::exec::StateSlotTable as SlotTable;
