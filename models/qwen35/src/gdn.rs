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

    /// 每步一次：按 host 侧 `cu_seqlens` 重建切块表，缓冲不够就按本步形状重分（只增不减）。
    pub fn prepare(&mut self, stream: &Stream, cu_seqlens: &[i32], cfg: &Qwen35Config) -> Result<()> {
        let plan = k::ops::KdaChunkPlan::new(stream, cu_seqlens, self.bt)?;
        let tokens = (*cu_seqlens.last().unwrap_or(&0)).max(0) as usize;
        let need_realloc = match &self.scratch {
            Some(sc) => !sc.fits(tokens, plan.num_chunks),
            None => true,
        };
        if need_realloc {
            let h = cfg.linear_num_key_heads as usize;
            let hv = cfg.linear_num_value_heads as usize;
            let (kd, vd) = (cfg.linear_key_head_dim as usize, cfg.linear_value_head_dim as usize);
            let old = self.scratch.as_ref().map_or((0, 0), |s| (s.tokens, s.num_chunks));
            let cap_t = tokens.max(old.0);
            let cap_c = plan.num_chunks.max(old.1);
            self.scratch = None; // 先释放旧的再分新的
            self.scratch = Some(k::ops::GdnChunkScratch::new(stream, cap_t, h, hv, kd, vd, cap_c, self.bt)?);
        }
        self.tokens = tokens;
        self.plan = Some(plan);
        Ok(())
    }
}

impl Default for GdnChunkWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

pub struct GdnLayer {
    in_qkv: Linear,
    in_z: Linear,
    in_a: Linear,
    in_b: Linear,
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

        let in_qkv = t.linear(stream, src, &format!("{p}.in_proj_qkv.weight"), None)?;
        ensure!(in_qkv.w.shape == [conv_dim, hidden], "in_proj_qkv shape {:?}", in_qkv.w.shape);
        let in_z = t.linear(stream, src, &format!("{p}.in_proj_z.weight"), None)?;
        ensure!(in_z.w.shape == [hv * vd, hidden], "in_proj_z shape {:?}", in_z.w.shape);
        let a = t.linear(stream, src, &format!("{p}.in_proj_a.weight"), None)?;
        let b = t.linear(stream, src, &format!("{p}.in_proj_b.weight"), None)?;
        ensure!(a.w.shape == [hv, hidden] && b.w.shape == [hv, hidden], "in_proj_a/b shape {:?} {:?}", a.w.shape, b.w.shape);
        let (in_a, in_b) = (a, b);
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
        };
        let _ = max_tokens; // 激活缓冲改为全模型共享（`crate::scratch::ActScratch`），不再按层预分配
        Ok(Self {
            in_qkv,
            in_z,
            in_a,
            in_b,
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
        vec![("gdn.in_proj_qkv", &self.in_qkv), ("gdn.in_proj_z", &self.in_z), ("gdn.in_proj_a", &self.in_a), ("gdn.in_proj_b", &self.in_b), ("gdn.out_proj", &self.out_proj)]
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

        // 1. 三个投影
        self.in_qkv.forward_rows(gemm, x, &mut abuf.qkv, t)?;
        self.in_z.forward_rows(gemm, x, &mut abuf.z, t)?;
        self.in_a.forward_rows(gemm, x, &mut abuf.a, t)?;
        self.in_b.forward_rows(gemm, x, &mut abuf.b, t)?;

        // 2. 短卷积（decode 单 token 走 update，其余走 varlen prefill）
        let single = meta.mode == BatchMode::Decode && meta.num_reqs == t;
        if single {
            k::gdn_conv1d_update(
                stream,
                &abuf.qkv.data,
                &self.conv_w.data,
                &mut abuf.conv.data,
                &mut self.state.conv.data,
                Some(&meta.state_indices.data),
                meta.num_reqs,
                conv_dim,
                width,
                width - 1,
                k::GdnAct::Silu,
            )?;
        } else {
            k::gdn_conv1d_prefill(
                stream,
                &abuf.qkv.data,
                &self.conv_w.data,
                &mut abuf.conv.data,
                &mut self.state.conv.data,
                &meta.cu_seqlens.data,
                Some(&meta.state_indices.data),
                Some(&meta.has_initial.data),
                meta.num_reqs,
                t,
                conv_dim,
                width,
                width - 1,
                k::GdnAct::Silu,
            )?;
        }

        // 3. gate / beta（decode 走 recurrent kernel 内 fused-gate 现算，省一个 launch；prefill 预算好给 chunk 用）
        if !single {
            k::gdn_gate(
                stream,
                &abuf.a.data,
                &abuf.b.data,
                &self.a_log.data,
                &self.dt_bias.data,
                &mut abuf.g.data,
                &mut abuf.beta.data,
                t,
                hv,
            )?;
        }

        // 4. delta rule 核心（q/k 的 l2norm 与 q 的 scale 在 kernel 内）：
        //    prefill 走 chunk 三段（prep → wy → ho），decode / 回退走逐 token 递归。两者数值等价（gdn_parity chunk case）。
        let scale = (kd as f32).powf(-0.5);
        let use_chunk = !single && t > meta.num_reqs && meta.chunk.is_some();
        if use_chunk {
            let mut ws = meta.chunk.unwrap().borrow_mut();
            let ws = &mut *ws;
            let plan = ws.plan.as_ref().ok_or_else(|| anyhow::anyhow!("gdn: chunk workspace 未 prepare"))?;
            let sc = ws.scratch.as_mut().ok_or_else(|| anyhow::anyhow!("gdn: chunk scratch 未分配"))?;
            ensure!(plan.num_seqs == meta.num_reqs && ws.tokens == t, "gdn: chunk plan 与本步不符");
            tracing::trace!(t, num_reqs = meta.num_reqs, chunks = plan.num_chunks, "gdn prefill: chunk 路径");
            k::gdn_chunk_prep(stream, &abuf.conv.data, &abuf.g.data, &meta.cu_seqlens.data, plan, sc, t, h, hv, kd, vd, scale)?;
            k::gdn_chunk_wy(stream, &abuf.conv.data, &abuf.beta.data, &meta.cu_seqlens.data, plan, sc, t, h, hv, kd, vd)?;
            k::gdn_chunk_ho(
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
            )?;
        } else {
            let (g, beta, a_raw, b_raw) = if single {
                (None, None, Some(&abuf.a.data), Some(&abuf.b.data))
            } else {
                (Some(&abuf.g.data), Some(&abuf.beta.data), None, None)
            };
            k::gdn_recurrent(
                stream,
                &abuf.conv.data,
                g,
                beta,
                a_raw,
                b_raw,
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
            )?;
        }

        // 5. 输出门 + RMSNorm（z 门，silu）
        k::gdn_gated_rmsnorm(
            stream,
            &abuf.o.data,
            &self.norm_w.data,
            &abuf.z.data,
            &mut abuf.n.data,
            t * hv,
            vd,
            cfg.rms_norm_eps,
            k::GdnAct::Silu,
        )?;

        // 6. out_proj
        let _ = act;
        let _ = group;
        self.out_proj.forward_rows(gemm, &abuf.n, out, t)
    }
}

/// 请求 → slot 分配表：公共件 `dinfer_layers::exec::StateSlotTable`（qwen35 GDN 与 kimik3 KDA 同一套）。
pub use dinfer_layers::exec::StateSlotTable as SlotTable;
