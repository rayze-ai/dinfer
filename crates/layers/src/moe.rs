//! MoE 参考实现（DeepSeek-V3 / GLM-5.3 口径，HF `DeepseekV3MoE`）：
//! ```text
//! logits = f32(x) · f32(W_gate)^T                       （fp32）
//! scores = sigmoid(logits); choice = scores + e_score_correction_bias
//! ids    = topk(choice, K)；w = scores[ids]；w /= sum(w)+1e-20（norm_topk_prob）；w *= routed_scaling_factor
//! y      = bf16(Σ_k w_k · Expert_{ids_k}(x)) + SharedExperts(x)
//! ```
//! 专家执行方式（P4 参考版）：GPU 上按专家分桶 → D2H 读 offsets（一次同步）→ 逐专家 gather / MLP / scatter-add。
//! 真机性能版（grouped GEMM、无同步）放 P10。

use crate::nn::{Linear, Mlp};
use crate::tensor::Tensor;
use crate::weights::WeightSource;
use anyhow::{ensure, Context, Result};
use dinfer_kernels::{self as k, Gemm, MoeBuckets, Stream};
use half::bf16;

#[derive(Debug, Clone)]
pub struct MoeConfig {
    pub hidden: usize,
    pub num_experts: usize,
    pub topk: usize,
    pub num_shared: usize,
    pub renormalize: bool,
    pub scale: f32,
}

pub struct Moe {
    pub cfg: MoeConfig,
    pub gate: Linear, // [E, hidden] bf16（HF gate.weight 存 bf16；路由时在 kernel 内升到 fp32）
    pub bias: Option<Tensor<f32>>,
    pub experts: Vec<Mlp>,
    pub shared: Option<Mlp>,
    buckets: MoeBuckets,
    // scratch
    logits: Tensor<f32>,
    topk_ids: Tensor<i32>,
    topk_w: Tensor<f32>,
    xg: Tensor<bf16>,
    yg: Tensor<bf16>,
    acc: Tensor<f32>,
    offsets_host: Vec<i32>,
}

impl Moe {
    /// `prefix` 形如 `model.layers.1.mlp`。
    pub fn load(
        stream: &Stream,
        src: &WeightSource,
        prefix: &str,
        cfg: MoeConfig,
        max_tokens: usize,
    ) -> Result<Self> {
        let gate = Linear::load(stream, src, &format!("{prefix}.gate"))?;
        ensure!(gate.out_features == cfg.num_experts && gate.in_features == cfg.hidden, "gate shape");
        let bias_name = format!("{prefix}.gate.e_score_correction_bias");
        let bias = if src.contains(&bias_name) {
            let (_, v) = src.load_f32_host(&bias_name)?;
            Some(Tensor::from_vec(stream, v, &[cfg.num_experts])?)
        } else {
            None
        };
        // 单个专家一步最多处理 max_tokens 行（每 token 对一个专家只选一次）
        let mut experts = Vec::with_capacity(cfg.num_experts);
        for e in 0..cfg.num_experts {
            experts.push(
                Mlp::load(stream, src, &format!("{prefix}.experts.{e}"), max_tokens)
                    .with_context(|| format!("expert {e}"))?,
            );
        }
        let shared = if cfg.num_shared > 0 {
            Some(Mlp::load(stream, src, &format!("{prefix}.shared_experts"), max_tokens)?)
        } else {
            None
        };
        Ok(Self {
            buckets: MoeBuckets::new(stream, cfg.num_experts, max_tokens, cfg.topk)?,
            logits: Tensor::zeros(stream, &[max_tokens, cfg.num_experts])?,
            topk_ids: Tensor::zeros(stream, &[max_tokens, cfg.topk])?,
            topk_w: Tensor::zeros(stream, &[max_tokens, cfg.topk])?,
            xg: Tensor::zeros(stream, &[max_tokens, cfg.hidden])?,
            yg: Tensor::zeros(stream, &[max_tokens, cfg.hidden])?,
            acc: Tensor::zeros(stream, &[max_tokens, cfg.hidden])?,
            offsets_host: vec![0; cfg.num_experts + 1],
            gate,
            bias,
            experts,
            shared,
            cfg,
        })
    }

    /// 调试 / 对拍：上一次 forward 的路由结果（[t, K] 专家 id 与归一化后权重）。
    pub fn last_routing(&self, stream: &Stream, t: usize) -> Result<(Vec<i32>, Vec<f32>)> {
        let k = self.cfg.topk;
        let ids = self.topk_ids.to_vec(stream)?;
        let w = self.topk_w.to_vec(stream)?;
        Ok((ids[..t * k].to_vec(), w[..t * k].to_vec()))
    }

    /// out[:t] = MoE(x[:t])（覆盖写）。
    pub fn forward(
        &mut self,
        stream: &Stream,
        gemm: &Gemm,
        t: usize,
        x: &Tensor<bf16>,
        out: &mut Tensor<bf16>,
    ) -> Result<()> {
        let c = &self.cfg;
        let (h, e, kk) = (c.hidden, c.num_experts, c.topk);
        ensure!(x.numel() >= t * h && out.numel() >= t * h, "Moe: buffer");
        ensure!(self.acc.numel() >= t * h, "Moe: tokens > max_tokens");

        // 1. 路由（fp32 logits）
        k::moe_gate_f32(stream, &x.data, &self.gate.w.data, &mut self.logits.data, t, e, h)?;
        k::moe_route_f32(
            stream,
            &self.logits.data,
            self.bias.as_ref().map(|b| &b.data),
            &mut self.topk_ids.data,
            &mut self.topk_w.data,
            t,
            e,
            kk,
            c.renormalize,
            c.scale,
        )?;
        // 2. 分桶 + 读 offsets（一次 D2H 同步）
        k::moe_bucket(stream, &self.topk_ids.data, &mut self.buckets, t, kk)?;
        self.offsets_host = stream.dtoh_sync_copy(&self.buckets.offsets)?;
        // 3. 逐专家：gather 该专家的 token 行 → MLP → 加权 scatter-add 到 fp32 累加器
        stream.memset_zeros_n(&mut self.acc.data, t * h)?;
        for (eid, start, len) in k::moe_expert_ranges(&self.offsets_host) {
            k::moe_gather_bf16(stream, &x.data, &mut self.xg.data, &self.buckets.sorted_tok, start, len, h)?;
            self.experts[eid].forward(stream, gemm, &crate::DefaultOps, len, &self.xg, &mut self.yg)?;
            k::moe_scatter_add_f32(
                stream,
                &self.yg.data,
                &mut self.acc.data,
                &self.buckets.sorted_tok,
                &self.buckets.sorted_k,
                &self.topk_w.data,
                start,
                len,
                kk,
                h,
            )?;
        }
        // 4. shared experts → out；out = bf16(acc) + out
        if let Some(sh) = &mut self.shared {
            sh.forward(stream, gemm, &crate::DefaultOps, t, x, out)?;
        } else {
            stream.memset_zeros_n(&mut out.data, t * h)?;
        }
        k::f32_add_to_bf16_n(stream, &self.acc.data, &mut out.data, t * h)
    }
}
