//! Kimi-K3 单层 `KimiDecoderLayer`（HF `modeling_kimi_linear.py::KimiDecoderLayer._forward_attn_residual`）。
//!
//! 有 AttnRes（`attn_res_block_size=12`）时一层的数据流（`x` 是上一层输出的 prefix_sum，`B` 是块残差列表）：
//!
//! ```text
//! h   = B 非空 ? AttnRes(B ∪ {x}; self_attention_res_proj/norm) : x
//! 写块层（layer % 12 == 0）：B ← B ∪ {x}，prefix = None
//! a   = attn(input_layernorm(h))
//! p   = prefix is None ? a : x + a                      ← 注意残差加的是 x，不是 h
//! m   = AttnRes(B ∪ {p}; mlp_res_proj/norm)
//! y   = mlp(post_attention_layernorm(m))
//! out = p + y
//! ```
//!
//! `AttnRes(V; proj, norm)`：对每个候选行 `v_r`（R = |B|+1 个），`s_r = Σ_d norm(v_r)_d · (norm.w · proj.w)_d`（f32），
//! `softmax_r(s)`，输出 `Σ_r p_r · v_r`。本轮按 HF 公式用普通算子拼（`rmsnorm` + `moe_gate_f32`(E=1) 取分数、host softmax、
//! `moe_scatter_add_f32` 做加权和），不做融合 kernel（vLLM `models/kimi_k3/nvidia/ops/attn_res.py` 是融合版，后补）。
//! 第 0 层是唯一的 `B = ∅` 层；第 1 层对拍时 `B = {x₀}`（x₀ 由测试给，HF 里是第 0 层输入即 embedding 输出）。
use crate::config::KimiLinearConfig;
use crate::kda::{KdaLayer, KdaMeta};
use crate::latent_moe::{LatentMoe, Situ};
use crate::mla::{MlaLayer, MlaMeta};
use crate::weights::{Table, LM_PREFIX};
use anyhow::{ensure, Result};
use dinfer_kernels::ops as k;
use dinfer_layers::{bf16, Gemm, Linear, Mlp, RmsNorm, Stream, Tensor, WeightSource};

#[allow(clippy::large_enum_variant)]
pub enum Attn {
    Kda(KdaLayer),
    Mla(MlaLayer),
}

/// 一步前向里两种注意力各自需要的 batch 描述（KDA 用 slot 状态表，MLA 用 paged KV）。
pub struct StepMeta<'a> {
    pub kda: KdaMeta<'a>,
    pub mla: MlaMeta<'a>,
    /// MLA 层 latent 写 cache 的槽位（每 token 一个，-1 跳过）。
    pub slot_mapping: &'a Tensor<i64>,
}

/// 装载时的缓冲 / cache 容量。
#[derive(Clone, Copy, Debug)]
pub struct LayerBudget {
    pub max_tokens: usize,
    /// KDA 状态槽数（并发请求数上限）。
    pub num_slots: usize,
    /// 每个 MLA 层的 paged KV 块数与块大小。
    pub kv_blocks: usize,
    pub block_size: usize,
}

#[allow(clippy::large_enum_variant)]
pub enum Ffn {
    Dense(Mlp),
    LatentMoe(LatentMoe),
}

/// 块残差门（`self_attention_res_*` / `mlp_res_*`）。
pub struct AttnRes {
    attn_proj: Linear, // [1, hidden]
    attn_norm: RmsNorm,
    mlp_proj: Linear,
    mlp_norm: RmsNorm,
    pub block_size: usize,
    max_cands: usize,
    nrm: Tensor<bf16>,  // [T, hidden]
    score: Tensor<f32>, // [T]
    probs: Tensor<f32>, // [T, R_max]
    acc: Tensor<f32>,   // [T, hidden]
    ident: Tensor<i32>, // [R_max·T_max]：块 r 内第 i 项 = i
    kidx: Tensor<i32>,  // [R_max·T_max]：块 r 内全为 r
}

impl AttnRes {
    /// `max_blocks`：模型里块残差最多几个（93 层 / 12 = 8）；候选行数 R ≤ max_blocks + 1。
    pub fn load(
        stream: &Stream,
        src: &WeightSource,
        t: &Table,
        layer: usize,
        cfg: &KimiLinearConfig,
        max_tokens: usize,
        max_blocks: usize,
    ) -> Result<Self> {
        let p = format!("layers.{layer}.");
        Self::load_named(
            stream,
            src,
            t,
            cfg,
            max_tokens,
            max_blocks,
            &format!("{p}self_attention_res"),
            &format!("{p}mlp_res"),
        )
    }

    /// 模型级 `output_attn_res_{proj,norm}`（只有一个门，两个槽位装同一组权重，`apply(which=0)`）。
    pub fn load_output(
        stream: &Stream,
        src: &WeightSource,
        t: &Table,
        cfg: &KimiLinearConfig,
        max_tokens: usize,
        max_blocks: usize,
    ) -> Result<Self> {
        Self::load_named(stream, src, t, cfg, max_tokens, max_blocks, "output_attn_res", "output_attn_res")
    }

    #[allow(clippy::too_many_arguments)]
    fn load_named(
        stream: &Stream,
        src: &WeightSource,
        t: &Table,
        cfg: &KimiLinearConfig,
        max_tokens: usize,
        max_blocks: usize,
        attn_key: &str,
        mlp_key: &str,
    ) -> Result<Self> {
        let hidden = cfg.hidden_size as usize;
        let block_size =
            cfg.attn_res_block_size.ok_or_else(|| anyhow::anyhow!("attn_res_block_size 缺失"))? as usize;
        let proj = |key: &str| -> Result<Linear> {
            let l = t.linear(stream, src, &format!("{key}_proj.weight"))?;
            ensure!(l.w.shape == [1, hidden], "{key}_proj: shape {:?}", l.w.shape);
            Ok(l)
        };
        let max_cands = max_blocks + 1;
        let tm = max_tokens;
        let mut ident = Vec::with_capacity(max_cands * tm);
        let mut kidx = Vec::with_capacity(max_cands * tm);
        for r in 0..max_cands {
            ident.extend(0..tm as i32);
            kidx.extend(std::iter::repeat_n(r as i32, tm));
        }
        Ok(Self {
            attn_proj: proj(attn_key)?,
            attn_norm: t.rmsnorm(stream, src, &format!("{attn_key}_norm.weight"), cfg.rms_norm_eps)?,
            mlp_proj: proj(mlp_key)?,
            mlp_norm: t.rmsnorm(stream, src, &format!("{mlp_key}_norm.weight"), cfg.rms_norm_eps)?,
            block_size,
            max_cands,
            nrm: Tensor::zeros(stream, &[tm, hidden])?,
            score: Tensor::zeros(stream, &[tm])?,
            probs: Tensor::zeros(stream, &[tm, max_cands])?,
            acc: Tensor::zeros(stream, &[tm, hidden])?,
            ident: Tensor::from_vec(stream, ident, &[max_cands * tm])?,
            kidx: Tensor::from_vec(stream, kidx, &[max_cands * tm])?,
        })
    }

    /// `out[:t] = Σ_r softmax_r(s)_r · cands[r]`，`cands` 的最后一项是 prefix_sum。`which`：0 = attention 门，1 = mlp 门。
    pub fn apply(
        &mut self,
        stream: &Stream,
        gemm: &Gemm,
        t: usize,
        cands: &[&Tensor<bf16>],
        which: usize,
        out: &mut Tensor<bf16>,
    ) -> Result<()> {
        let _ = gemm;
        let (proj, norm) =
            if which == 0 { (&self.attn_proj, &self.attn_norm) } else { (&self.mlp_proj, &self.mlp_norm) };
        let r_n = cands.len();
        let hidden = norm.hidden;
        let tm = self.nrm.shape[0];
        ensure!(r_n >= 1 && r_n <= self.max_cands, "AttnRes: 候选数 {r_n} 超过 {}", self.max_cands);
        ensure!(t <= tm && out.numel() >= t * hidden, "AttnRes: t / out");
        // 1. 每个候选：s_r[tok] = Σ_d bf16(w·norm(v_r))_d · proj_d（f32 累加；HF 全 f32）
        let mut scores = vec![0f32; t * r_n];
        for (r, v) in cands.iter().enumerate() {
            ensure!(v.numel() >= t * hidden, "AttnRes: 候选 {r} 行数不足");
            k::rmsnorm_bf16_n(stream, &v.data, &norm.w.data, &mut self.nrm.data, t, hidden, norm.eps)?;
            k::moe_gate_f32(stream, &self.nrm.data, &proj.w.data, &mut self.score.data, t, 1, hidden)?;
            let s = stream.dtoh_sync_copy(&self.score.data)?;
            for i in 0..t {
                scores[i * r_n + r] = s[i];
            }
        }
        // 2. host softmax（R ≤ 9，T 行）
        let mut probs = vec![0f32; t * self.max_cands];
        for i in 0..t {
            let row = &scores[i * r_n..(i + 1) * r_n];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let ex: Vec<f32> = row.iter().map(|s| (s - m).exp()).collect();
            let sum: f32 = ex.iter().sum();
            for r in 0..r_n {
                probs[i * r_n + r] = ex[r] / sum;
            }
        }
        stream.htod_into(&probs[..t * r_n], &mut self.probs.data)?;
        // 3. acc = Σ_r p_r · v_r（复用 scatter_add：tok=i，k=r，权重表 probs[i·R + r]）
        stream.memset_zeros_n(&mut self.acc.data, t * hidden)?;
        for (r, v) in cands.iter().enumerate() {
            k::moe_scatter_add_f32(
                stream,
                &v.data,
                &mut self.acc.data,
                &self.ident.data,
                &self.kidx.data,
                &self.probs.data,
                r * tm,
                t,
                r_n,
                hidden,
            )?;
        }
        stream.memset_zeros_n(&mut out.data, t * hidden)?;
        k::f32_add_to_bf16_n(stream, &self.acc.data, &mut out.data, t * hidden)
    }
}

pub struct KimiDecoderLayer {
    pub layer: usize,
    pub attn: Attn,
    pub ffn: Ffn,
    pub attn_res: Option<AttnRes>,
    input_norm: RmsNorm,
    post_norm: RmsNorm,
    act: Situ,
    hidden: usize,
    // 激活缓冲 [T, hidden]
    h_in: Tensor<bf16>, // AttnRes(attention 门) 输出
    hn: Tensor<bf16>,   // input_layernorm 输出
    a: Tensor<bf16>,    // attention 输出
    p: Tensor<bf16>,    // prefix_sum（残差）
    m: Tensor<bf16>,    // AttnRes(mlp 门) 输出
    mn: Tensor<bf16>,   // post_attention_layernorm 输出
    y: Tensor<bf16>,    // mlp 输出
}

impl KimiDecoderLayer {
    /// （KDA | MLA）× （Dense | latent MoE）+ 可选 AttnRes。
    pub fn load(
        stream: &Stream,
        src: &WeightSource,
        t: &Table,
        layer: usize,
        cfg: &KimiLinearConfig,
        b: LayerBudget,
    ) -> Result<Self> {
        let hidden = cfg.hidden_size as usize;
        let max_tokens = b.max_tokens;
        let attn = if cfg.is_kda_layer(layer as u32) {
            Attn::Kda(KdaLayer::load(stream, src, t, layer, cfg, max_tokens, b.num_slots)?)
        } else {
            Attn::Mla(MlaLayer::load(stream, src, t, layer, cfg, max_tokens, b.kv_blocks, b.block_size)?)
        };
        let act = Situ::from_config(cfg)?;
        let ffn = if cfg.is_moe_layer(layer as u32) {
            Ffn::LatentMoe(LatentMoe::load(stream, src, t, layer, cfg, max_tokens)?)
        } else {
            let m = Mlp::load(stream, src, &format!("{LM_PREFIX}layers.{layer}.mlp"), max_tokens)?;
            ensure!(m.inter == cfg.intermediate_size as usize, "dense mlp inter {}", m.inter);
            Ffn::Dense(m)
        };
        let attn_res = if cfg.attn_res_block_size.is_some() {
            let max_blocks =
                (cfg.num_hidden_layers as usize).div_ceil(cfg.attn_res_block_size.unwrap() as usize);
            Some(AttnRes::load(stream, src, t, layer, cfg, max_tokens, max_blocks)?)
        } else {
            None
        };
        let p = format!("layers.{layer}");
        let z = |s: &Stream| Tensor::<bf16>::zeros(s, &[max_tokens, hidden]);
        Ok(Self {
            layer,
            attn,
            ffn,
            attn_res,
            input_norm: t.rmsnorm(stream, src, &format!("{p}.input_layernorm.weight"), cfg.rms_norm_eps)?,
            post_norm: t.rmsnorm(
                stream,
                src,
                &format!("{p}.post_attention_layernorm.weight"),
                cfg.rms_norm_eps,
            )?,
            act,
            hidden,
            h_in: z(stream)?,
            hn: z(stream)?,
            a: z(stream)?,
            p: z(stream)?,
            m: z(stream)?,
            mn: z(stream)?,
            y: z(stream)?,
        })
    }

    /// 本层是否是块残差写入层（`layer % block_size == 0`）：是则调用方在本层 forward **之后**把本层输入 `x` 追加进 `blocks`。
    pub fn writes_block(&self) -> bool {
        self.attn_res.as_ref().is_some_and(|r| self.layer % r.block_size == 0)
    }

    /// `x[:t]`（上一层 prefix_sum）、`blocks`（已写入的块残差，每个 `[≥t, hidden]`）→ `out[:t]`。
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        stream: &Stream,
        gemm: &Gemm,
        t: usize,
        x: &Tensor<bf16>,
        blocks: &[&Tensor<bf16>],
        meta: &StepMeta<'_>,
        out: &mut Tensor<bf16>,
    ) -> Result<()> {
        let h = self.hidden;
        ensure!(t <= self.hn.shape[0], "layer: {t} > max_tokens");
        ensure!(x.numel() >= t * h && out.numel() >= t * h, "layer: x/out");
        let writer = self.writes_block();
        let n = t * h;

        // 1. attention 门：B 非空时 h = AttnRes(B ∪ {x})
        let use_res_in = self.attn_res.is_some() && !blocks.is_empty();
        if use_res_in {
            let mut cands: Vec<&Tensor<bf16>> = blocks.to_vec();
            cands.push(x);
            self.attn_res.as_mut().unwrap().apply(stream, gemm, t, &cands, 0, &mut self.h_in)?;
        } else {
            ensure!(self.attn_res.is_some() || blocks.is_empty(), "无 AttnRes 的模型不该有块残差");
        }
        let h_src = if use_res_in { &self.h_in } else { x };
        // 2. attention
        k::rmsnorm_bf16_n(
            stream,
            &h_src.data,
            &self.input_norm.w.data,
            &mut self.hn.data,
            t,
            h,
            self.input_norm.eps,
        )?;
        match &mut self.attn {
            Attn::Kda(kda) => kda.forward(stream, gemm, t, &self.hn, &meta.kda, &mut self.a)?,
            Attn::Mla(mla) => {
                mla.forward(stream, gemm, t, &self.hn, meta.slot_mapping, &meta.mla, &mut self.a)?
            }
        }
        // 3. prefix_sum：写块层 p = a；否则 p = x + a
        if writer {
            stream.dtod_copy_n(&self.a.data, &mut self.p.data, n)?;
        } else {
            stream.dtod_copy_n(&x.data, &mut self.p.data, n)?;
            k::add_inplace_bf16_n(stream, &self.a.data, &mut self.p.data, n)?;
        }
        // 4. mlp 门：m = AttnRes(B' ∪ {p})，B' = writer ? B ∪ {x} : B
        let m_src = if let Some(res) = self.attn_res.as_mut() {
            let mut cands: Vec<&Tensor<bf16>> = blocks.to_vec();
            if writer {
                cands.push(x);
            }
            cands.push(&self.p);
            res.apply(stream, gemm, t, &cands, 1, &mut self.m)?;
            &self.m
        } else {
            &self.p
        };
        // 5. mlp
        k::rmsnorm_bf16_n(
            stream,
            &m_src.data,
            &self.post_norm.w.data,
            &mut self.mn.data,
            t,
            h,
            self.post_norm.eps,
        )?;
        match &mut self.ffn {
            Ffn::LatentMoe(moe) => moe.forward(stream, gemm, t, &self.mn, &mut self.y)?,
            Ffn::Dense(mlp) => mlp.forward(stream, gemm, &self.act, t, &self.mn, &mut self.y)?,
        }
        // 6. out = p + y
        stream.dtod_copy_n(&self.p.data, &mut out.data, n)?;
        k::add_inplace_bf16_n(stream, &self.y.data, &mut out.data, n)
    }

    // ---- 对拍 / 调试用中间量（上一步，前 t 行有效）----
    pub fn last_attn_in(&self) -> &Tensor<bf16> {
        &self.h_in
    }
    pub fn last_attn_out(&self) -> &Tensor<bf16> {
        &self.a
    }
    pub fn last_prefix_sum(&self) -> &Tensor<bf16> {
        &self.p
    }
    pub fn last_mlp_gate_out(&self) -> &Tensor<bf16> {
        &self.m
    }
    pub fn last_mlp_in(&self) -> &Tensor<bf16> {
        &self.mn
    }
    pub fn last_mlp_out(&self) -> &Tensor<bf16> {
        &self.y
    }
}
