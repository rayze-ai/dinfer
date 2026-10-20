//! KDA 层（Kimi Delta Attention，K3 的 69 个线性注意力层）。数学规格与 kernel 口径见 `docs/loops/k3-kda.md`，
//! 结构事实见 `docs/design-kimik3.md` §1。
//!
//! ```text
//! x → q/k/v_proj → 三个 depthwise causal conv1d(silu, w=4)
//!   → f_b(f_a(x)) = raw_g [T, H·K]、b_proj(x) = raw_beta [T, H]
//!   → KDA 核心（kernel 内：qk l2norm、gate = lb·sigmoid(exp(A_log[h])·(raw_g + dt_bias))、beta = sigmoid(raw_beta)）
//!       prefill：chunk 三段 `kda_chunk_{prep,wy,ho}`；decode / 回退：逐 token `kda_recurrent`
//!   → o_norm：RMSNorm(o, head_dim) · w · sigmoid(g_proj(x))
//!   → o_proj
//! ```
//!
//! 状态（每层每 slot）：conv `[slots, H·K, w-1]` bf16 ×3（q/k/v 各一份）+ 递推 `[slots, H, V, K]` f32（6 MB），本 crate
//! 自管（`SlotTable` 按请求分 slot，新请求首次出现时 `reset_slot`）。
//!
//! 与 HF `modeling_kimi_linear.py::KimiDeltaAttention` 的对应：HF 的 `A_log` / `dt_bias` 是 f32 参数、`o_norm` 是
//! `FusedRMSNormGated(activation='sigmoid')`（`norm(x)·w·sigmoid(g)`，非 zero-centered）；真分片的 `A_log` 是 `[128]`
//! 零填充，装载只取前 96（`weights::load_a_log`）。
//!
//! 待办（里程碑 C-2+）：六条投影 GEMM 融合成 vLLM 的 `in_proj_qkvgfab`（一次 GEMM 再按列切）；
//! 激活缓冲改全模型共享（现在每层一套，`max_tokens` 大时占内存）；TP 按头切。
use crate::config::KimiLinearConfig;
use crate::weights::{self, KdaShapes, Table};
use anyhow::{ensure, Result};
use dinfer_core::runner::BatchMode;
use dinfer_kernels::ops as k;
use dinfer_layers::{bf16, Gemm, Linear, Stream, Tensor, WeightSource};
use std::cell::RefCell;

/// 一层的状态张量。
pub struct KdaState {
    /// q/k/v 三份 `[slots, H·K, width-1]` bf16（列 0 最旧）。
    pub conv: [Tensor<bf16>; 3],
    /// `[slots, H, V, K]` f32。
    pub rec: Tensor<f32>,
}

/// 步内元数据（executor 上传后的设备张量）。
pub struct KdaMeta<'a> {
    pub cu_seqlens: &'a Tensor<i32>,
    pub state_indices: &'a Tensor<i32>,
    /// 每请求：该 slot 是否已有历史（新请求 = 0）。只有 conv prefill 用；递推状态靠 `reset_slot` 清零。
    pub has_initial: &'a Tensor<i32>,
    pub num_reqs: usize,
    pub mode: BatchMode,
    /// chunk prefill 的共享工作区；`None` = 本步走顺序递归。
    pub chunk: Option<&'a RefCell<KdaChunkWorkspace>>,
}

/// chunk prefill 的切块表 + 中间缓冲，全模型 KDA 层共用一份（每层串行，无重入）。
/// 块大小 `BT` 由 host 选（kernel 当前只特化了 64），不是 kernel 内常量。
pub struct KdaChunkWorkspace {
    pub plan: Option<k::KdaChunkPlan>,
    pub scratch: Option<k::KdaChunkScratch>,
    pub tokens: usize,
    cap_tokens: usize,
    cap_chunks: usize,
    bt: usize,
}

impl KdaChunkWorkspace {
    pub const BT: usize = 64;

    pub fn new() -> Self {
        Self { plan: None, scratch: None, tokens: 0, cap_tokens: 0, cap_chunks: 0, bt: Self::BT }
    }

    /// 每步一次：按 host 侧 `cu_seqlens` 重建切块表；缓冲不够就按本步形状重分（只增不减；三段 kernel 都按本步
    /// `tokens` 判容量，缓冲偏大无妨）。
    pub fn prepare(&mut self, stream: &Stream, cu_seqlens: &[i32], sh: &KdaShapes) -> Result<()> {
        let plan = k::KdaChunkPlan::new(stream, cu_seqlens, self.bt)?;
        let tokens = (*cu_seqlens.last().unwrap_or(&0)).max(0) as usize;
        let fits = self.scratch.is_some() && tokens <= self.cap_tokens && plan.num_chunks <= self.cap_chunks;
        if !fits {
            let cap_t = tokens.max(self.cap_tokens);
            let cap_c = plan.num_chunks.max(self.cap_chunks);
            self.scratch = None;
            self.scratch = Some(k::KdaChunkScratch::new(stream, cap_t, sh.heads, sh.head_dim, cap_c, self.bt)?);
            self.cap_tokens = cap_t;
            self.cap_chunks = cap_c;
        }
        self.tokens = tokens;
        self.plan = Some(plan);
        Ok(())
    }
}

impl Default for KdaChunkWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

pub struct KdaLayer {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    g_proj: Linear,
    f_a: Linear,
    f_b: Linear,
    b_proj: Linear,
    conv_w: [Tensor<bf16>; 3], // 各 [H·K, width]
    a_log: Tensor<f32>,        // [H]
    dt_bias: Tensor<f32>,      // [H·K]
    norm_w: Tensor<bf16>,      // [K]
    o_proj: Linear,
    pub state: KdaState,
    sh: KdaShapes,
    lower_bound: Option<f32>,
    eps: f32,
    // 激活缓冲（[T, ·]）
    q_buf: Tensor<bf16>,
    k_buf: Tensor<bf16>,
    v_buf: Tensor<bf16>,
    qc_buf: Tensor<bf16>,
    kc_buf: Tensor<bf16>,
    vc_buf: Tensor<bf16>,
    g2_buf: Tensor<bf16>,    // g_proj(x)，输出门
    fa_buf: Tensor<bf16>,    // [T, K]
    g1_buf: Tensor<bf16>,    // f_b(f_a(x))，raw gate
    b_buf: Tensor<bf16>,     // [T, H]
    beta_buf: Tensor<f32>,   // [T, H]
    o_buf: Tensor<bf16>,
    n_buf: Tensor<bf16>,
}

impl KdaLayer {
    /// 装载第 `layer` 层（必须是 KDA 层）。`max_tokens` 决定激活缓冲，`num_slots` 决定状态表大小。
    pub fn load(
        stream: &Stream,
        src: &WeightSource,
        t: &Table,
        layer: usize,
        cfg: &KimiLinearConfig,
        max_tokens: usize,
        num_slots: usize,
    ) -> Result<Self> {
        ensure!(cfg.is_kda_layer(layer as u32), "第 {layer} 层不是 KDA 层");
        let sh = KdaShapes::new(cfg);
        let (hidden, h, kd, width) = (sh.hidden, sh.heads, sh.head_dim, sh.width);
        let proj = sh.proj();
        let p = format!("layers.{layer}.self_attn");
        let lin = |name: &str, out: usize, inp: usize| -> Result<Linear> {
            let l = t.linear(stream, src, &format!("{p}.{name}.weight"))?;
            ensure!(l.w.shape == [out, inp], "{p}.{name}: shape {:?}（期望 [{out}, {inp}]）", l.w.shape);
            Ok(l)
        };
        let q_proj = lin("q_proj", proj, hidden)?;
        let k_proj = lin("k_proj", proj, hidden)?;
        let v_proj = lin("v_proj", proj, hidden)?;
        let g_proj = lin("g_proj", proj, hidden)?;
        let f_a = lin("f_a_proj", kd, hidden)?;
        let f_b = lin("f_b_proj", proj, kd)?;
        let b_proj = lin("b_proj", h, hidden)?;
        let o_proj = lin("o_proj", hidden, proj)?;
        let conv_w = [
            weights::load_conv1d(stream, src, t, &format!("{p}.q_conv1d.weight"), proj, width)?,
            weights::load_conv1d(stream, src, t, &format!("{p}.k_conv1d.weight"), proj, width)?,
            weights::load_conv1d(stream, src, t, &format!("{p}.v_conv1d.weight"), proj, width)?,
        ];
        let a_log = weights::load_a_log(stream, src, t, &format!("{p}.A_log"), h)?;
        let dt_bias = t.load_f32(stream, src, &format!("{p}.dt_bias"))?;
        ensure!(dt_bias.numel() == proj, "dt_bias {} != H·K {proj}", dt_bias.numel());
        let norm_w = t.load_bf16(stream, src, &format!("{p}.o_norm.weight"))?;
        ensure!(norm_w.numel() == kd, "o_norm.weight {} != K {kd}", norm_w.numel());

        let st = width - 1;
        let state = KdaState {
            conv: [
                Tensor::zeros(stream, &[num_slots, proj, st])?,
                Tensor::zeros(stream, &[num_slots, proj, st])?,
                Tensor::zeros(stream, &[num_slots, proj, st])?,
            ],
            rec: Tensor::zeros(stream, &[num_slots, h, kd, kd])?,
        };
        let tt = max_tokens;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            g_proj,
            f_a,
            f_b,
            b_proj,
            conv_w,
            a_log,
            dt_bias,
            norm_w,
            o_proj,
            state,
            lower_bound: cfg.linear_attn_config.gate_lower_bound,
            eps: cfg.rms_norm_eps,
            q_buf: Tensor::zeros(stream, &[tt, proj])?,
            k_buf: Tensor::zeros(stream, &[tt, proj])?,
            v_buf: Tensor::zeros(stream, &[tt, proj])?,
            qc_buf: Tensor::zeros(stream, &[tt, proj])?,
            kc_buf: Tensor::zeros(stream, &[tt, proj])?,
            vc_buf: Tensor::zeros(stream, &[tt, proj])?,
            g2_buf: Tensor::zeros(stream, &[tt, proj])?,
            fa_buf: Tensor::zeros(stream, &[tt, kd])?,
            g1_buf: Tensor::zeros(stream, &[tt, proj])?,
            b_buf: Tensor::zeros(stream, &[tt, h])?,
            beta_buf: Tensor::zeros(stream, &[tt, h])?,
            o_buf: Tensor::zeros(stream, &[tt, proj])?,
            n_buf: Tensor::zeros(stream, &[tt, proj])?,
            sh,
        })
    }

    pub fn shapes(&self) -> &KdaShapes {
        &self.sh
    }

    /// 清零 slot `s` 的全部状态（新请求首次出现时调用）。只清本 slot，不碰别的请求。
    pub fn reset_slot(&mut self, stream: &Stream, s: usize) -> Result<()> {
        let n = self.sh.heads * self.sh.head_dim * self.sh.head_dim;
        ensure!(s < self.state.rec.shape[0], "reset_slot: slot {s} 越界");
        {
            let mut v = self.state.rec.data.slice_mut(s * n..(s + 1) * n);
            stream.inner_arc().memset_zeros(&mut v)?;
        }
        for c in &mut self.state.conv {
            let cn = c.shape[1] * c.shape[2];
            let mut v = c.data.slice_mut(s * cn..(s + 1) * cn);
            stream.inner_arc().memset_zeros(&mut v)?;
        }
        Ok(())
    }

    /// `x: [t, hidden]`（已过 `input_layernorm`）→ `out: [t, hidden]`。
    pub fn forward(
        &mut self,
        stream: &Stream,
        gemm: &Gemm,
        t: usize,
        x: &Tensor<bf16>,
        meta: &KdaMeta<'_>,
        out: &mut Tensor<bf16>,
    ) -> Result<()> {
        let (h, kd, width) = (self.sh.heads, self.sh.head_dim, self.sh.width);
        let proj = self.sh.proj();
        ensure!(t <= self.q_buf.shape[0], "kda: {t} tokens > max {}", self.q_buf.shape[0]);
        ensure!(meta.state_indices.numel() >= meta.num_reqs, "kda: state_indices");
        ensure!(x.numel() >= t * self.sh.hidden && out.numel() >= t * self.sh.hidden, "kda: x/out 不足 {t} 行");

        // 1. 投影（q/k/v/g 各 7168→12288；f_a 7168→128 → f_b 128→12288；b 7168→96）
        self.q_proj.forward_rows(gemm, x, &mut self.q_buf, t)?;
        self.k_proj.forward_rows(gemm, x, &mut self.k_buf, t)?;
        self.v_proj.forward_rows(gemm, x, &mut self.v_buf, t)?;
        self.g_proj.forward_rows(gemm, x, &mut self.g2_buf, t)?;
        self.f_a.forward_rows(gemm, x, &mut self.fa_buf, t)?;
        self.f_b.forward_rows(gemm, &self.fa_buf, &mut self.g1_buf, t)?;
        self.b_proj.forward_rows(gemm, x, &mut self.b_buf, t)?;
        k::bf16_to_f32_n(stream, &self.b_buf.data, &mut self.beta_buf.data, t * h)?;

        // 2. 三个短卷积（decode 单 token 走 update，其余走 varlen prefill）
        let single = meta.mode == BatchMode::Decode && meta.num_reqs == t;
        let srcs = [&self.q_buf, &self.k_buf, &self.v_buf];
        let dsts = [&mut self.qc_buf, &mut self.kc_buf, &mut self.vc_buf];
        for (i, (src, dst)) in srcs.into_iter().zip(dsts).enumerate() {
            if single {
                k::kda_conv1d_update(
                    stream,
                    &src.data,
                    &self.conv_w[i].data,
                    &mut dst.data,
                    &mut self.state.conv[i].data,
                    Some(&meta.state_indices.data),
                    meta.num_reqs,
                    proj,
                    width,
                    width - 1,
                )?;
            } else {
                k::kda_conv1d_prefill(
                    stream,
                    &src.data,
                    &self.conv_w[i].data,
                    &mut dst.data,
                    &mut self.state.conv[i].data,
                    &meta.cu_seqlens.data,
                    Some(&meta.state_indices.data),
                    Some(&meta.has_initial.data),
                    meta.num_reqs,
                    proj,
                    width,
                    width - 1,
                )?;
            }
        }

        // 3. KDA 核心（gate / beta / l2norm / scale 都在 kernel 内）
        let scale = (kd as f32).powf(-0.5);
        let use_chunk = !single && t > meta.num_reqs && meta.chunk.is_some();
        if use_chunk {
            let mut ws = meta.chunk.unwrap().borrow_mut();
            let ws = &mut *ws;
            let plan = ws.plan.as_ref().ok_or_else(|| anyhow::anyhow!("kda: chunk workspace 未 prepare"))?;
            let sc = ws.scratch.as_mut().ok_or_else(|| anyhow::anyhow!("kda: chunk scratch 未分配"))?;
            ensure!(plan.num_seqs == meta.num_reqs && ws.tokens == t, "kda: chunk plan 与本步不符");
            k::kda_chunk_prep(
                stream,
                &self.qc_buf.data,
                &self.kc_buf.data,
                &self.g1_buf.data,
                &self.beta_buf.data,
                &self.a_log.data,
                &self.dt_bias.data,
                &meta.cu_seqlens.data,
                plan,
                sc,
                t,
                h,
                kd,
                scale,
                self.lower_bound,
            )?;
            k::kda_chunk_wy(stream, &self.vc_buf.data, &meta.cu_seqlens.data, plan, sc, t, h, kd)?;
            k::kda_chunk_ho(
                stream,
                &meta.cu_seqlens.data,
                plan,
                sc,
                &mut self.o_buf.data,
                &mut self.state.rec.data,
                Some(&meta.state_indices.data),
                t,
                h,
                kd,
            )?;
        } else {
            k::kda_recurrent(
                stream,
                &self.qc_buf.data,
                &self.kc_buf.data,
                &self.vc_buf.data,
                &self.g1_buf.data,
                &self.beta_buf.data,
                &self.a_log.data,
                &self.dt_bias.data,
                &mut self.o_buf.data,
                &mut self.state.rec.data,
                &meta.cu_seqlens.data,
                Some(&meta.state_indices.data),
                meta.num_reqs,
                h,
                kd,
                scale,
                self.lower_bound,
            )?;
        }

        // 4. o_norm：RMSNorm(o)·w·sigmoid(g2)
        k::kda_gated_rmsnorm(stream, &self.o_buf.data, &self.norm_w.data, &self.g2_buf.data, &mut self.n_buf.data, t * h, kd, self.eps)?;

        // 5. o_proj
        self.o_proj.forward_rows(gemm, &self.n_buf, out, t)
    }

    /// 对拍 / 调试用：上一步 `o_norm` 的输出 `[t, H·K]`（`o_proj` 之前）。
    pub fn last_normed(&self) -> &Tensor<bf16> {
        &self.n_buf
    }
}

/// 请求 → slot 分配表：公共件 `dinfer_layers::exec::StateSlotTable`（qwen35 GDN 与 kimik3 KDA 同一套）。
pub use dinfer_layers::exec::StateSlotTable as SlotTable;
