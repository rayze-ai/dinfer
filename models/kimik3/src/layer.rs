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
pub use crate::scratch::{ActScratch, ResScratch};
use crate::kda::{KdaLayer, KdaMeta};
use crate::latent_moe::{LatentMoe, Situ};
use crate::mla::{MlaLayer, MlaMeta};
use crate::weights::{Shard, Table, LM_PREFIX};
use dinfer_kernels::comm::Comm;
use anyhow::{ensure, Context, Result};
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
    /// TP/EP 组（`None` = 单卡，无任何通信调用，路径逐位不变）。通信点见 design-kimik3 §2.4。
    pub comm: Option<&'a Comm>,
}

/// bf16 部分和的 all-reduce。默认直接 bf16 all-reduce（NCCL 内部 f32 累加、落回 bf16 一次舍入；与 qwen3 TP 同）。
/// `TP_REDUCE_F32=1`（实验，docs/loops/k3-tp.md 第 3 轮）：先升 f32 再 all-reduce 再降 bf16，通信字节 ×2，用来判定 bf16 双舍入
/// 对 tp 漂移的贡献。`scratch` 需 ≥ n 个 f32。
pub fn all_reduce_partial_bf16(
    stream: &Stream,
    comm: &Comm,
    buf: &mut Tensor<bf16>,
    n: usize,
    scratch: &mut Tensor<f32>,
) -> Result<()> {
    static F32: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let f32_mode = *F32.get_or_init(|| std::env::var("TP_REDUCE_F32").map(|v| v == "1").unwrap_or(false));
    if !f32_mode {
        return dinfer_kernels::comm::all_reduce_bf16(comm, &mut buf.data, n);
    }
    ensure!(scratch.numel() >= n, "all_reduce_partial_bf16: scratch {} < {n}", scratch.numel());
    k::bf16_to_f32_n(stream, &buf.data, &mut scratch.data, n)?;
    {
        let mut v = scratch.data.slice_mut(..n);
        comm.all_reduce_sum(&mut v, n)?;
    }
    stream.memset_zeros_n(&mut buf.data, n)?;
    k::f32_add_to_bf16_n(stream, &scratch.data, &mut buf.data, n)
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
}

impl AttnRes {
    /// `max_blocks`：模型里块残差最多几个（93 层 / 12 = 8）；候选行数 R ≤ max_blocks + 1。
    pub fn load(
        stream: &Stream,
        src: &WeightSource,
        t: &Table,
        layer: usize,
        cfg: &KimiLinearConfig,
    ) -> Result<Self> {
        let p = format!("layers.{layer}.");
        Self::load_named(
            stream,
            src,
            t,
            cfg,
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
    ) -> Result<Self> {
        Self::load_named(stream, src, t, cfg, "output_attn_res", "output_attn_res")
    }

    #[allow(clippy::too_many_arguments)]
    fn load_named(
        stream: &Stream,
        src: &WeightSource,
        t: &Table,
        cfg: &KimiLinearConfig,
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
        Ok(Self {
            attn_proj: proj(attn_key)?,
            attn_norm: t.rmsnorm(stream, src, &format!("{attn_key}_norm.weight"), cfg.rms_norm_eps)?,
            mlp_proj: proj(mlp_key)?,
            mlp_norm: t.rmsnorm(stream, src, &format!("{mlp_key}_norm.weight"), cfg.rms_norm_eps)?,
            block_size,
        })
    }

    /// `out[:t] = Σ_r softmax_r(s)_r · cands[r]`，`cands` 的最后一项是 prefix_sum。`which`：0 = attention 门，1 = mlp 门。
    #[allow(clippy::too_many_arguments)]
    pub fn apply(
        &self,
        stream: &Stream,
        gemm: &Gemm,
        t: usize,
        cands: &[&Tensor<bf16>],
        which: usize,
        sc: &mut ResScratch,
        acc: &mut Tensor<f32>,
        out: &mut Tensor<bf16>,
    ) -> Result<()> {
        let _ = gemm;
        let (proj, norm) =
            if which == 0 { (&self.attn_proj, &self.attn_norm) } else { (&self.mlp_proj, &self.mlp_norm) };
        let r_n = cands.len();
        let hidden = norm.hidden;
        let tm = sc.nrm.shape[0];
        ensure!(acc.numel() >= t * hidden, "AttnRes: acc scratch 不足");
        ensure!(r_n >= 1 && r_n <= sc.max_cands, "AttnRes: 候选数 {r_n} 超过 {}", sc.max_cands);
        ensure!(t <= tm && out.numel() >= t * hidden, "AttnRes: t / out");
        // 1. 每个候选：s_r[tok] = Σ_d bf16(w·norm(v_r))_d · proj_d（f32 累加；HF 全 f32）
        let mut scores = vec![0f32; t * r_n];
        for (r, v) in cands.iter().enumerate() {
            ensure!(v.numel() >= t * hidden, "AttnRes: 候选 {r} 行数不足");
            k::rmsnorm_bf16_n(stream, &v.data, &norm.w.data, &mut sc.nrm.data, t, hidden, norm.eps)?;
            k::moe_gate_f32(stream, &sc.nrm.data, &proj.w.data, &mut sc.score.data, t, 1, hidden)?;
            let s = stream.dtoh_sync_copy(&sc.score.data)?;
            for i in 0..t {
                scores[i * r_n + r] = s[i];
            }
        }
        // 2. host softmax（R ≤ 9，T 行）
        let mut probs = vec![0f32; t * sc.max_cands];
        for i in 0..t {
            let row = &scores[i * r_n..(i + 1) * r_n];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let ex: Vec<f32> = row.iter().map(|s| (s - m).exp()).collect();
            let sum: f32 = ex.iter().sum();
            for r in 0..r_n {
                probs[i * r_n + r] = ex[r] / sum;
            }
        }
        stream.htod_into(&probs[..t * r_n], &mut sc.probs.data)?;
        // 3. acc = Σ_r p_r · v_r（复用 scatter_add：tok=i，k=r，权重表 probs[i·R + r]）
        stream.memset_zeros_n(&mut acc.data, t * hidden)?;
        for (r, v) in cands.iter().enumerate() {
            k::moe_scatter_add_f32(
                stream,
                &v.data,
                &mut acc.data,
                &sc.ident.data,
                &sc.kidx.data,
                &sc.probs.data,
                r * tm,
                t,
                r_n,
                hidden,
            )?;
        }
        stream.memset_zeros_n(&mut out.data, t * hidden)?;
        k::f32_add_to_bf16_n(stream, &acc.data, &mut out.data, t * hidden)
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
        Self::load_tp(stream, src, t, layer, cfg, b, Shard::SINGLE)
    }

    /// TP/EP 分片装载（design-kimik3 §2.4）：attention 按头、MoE 按专家 id、dense / shared MLP 按 inter；AttnRes / norm 复制。
    pub fn load_tp(
        stream: &Stream,
        src: &WeightSource,
        t: &Table,
        layer: usize,
        cfg: &KimiLinearConfig,
        b: LayerBudget,
        shard: Shard,
    ) -> Result<Self> {
        let hidden = cfg.hidden_size as usize;
        let attn = if cfg.is_kda_layer(layer as u32) {
            Attn::Kda(KdaLayer::load_tp(stream, src, t, layer, cfg, b.num_slots, shard)?)
        } else {
            Attn::Mla(MlaLayer::load_tp(stream, src, t, layer, cfg, b.kv_blocks, b.block_size, shard)?)
        };
        let act = Situ::from_config(cfg)?;
        let ffn = if cfg.is_moe_layer(layer as u32) {
            Ffn::LatentMoe(LatentMoe::load_tp(stream, src, t, layer, cfg, shard)?)
        } else {
            let m = Mlp::load_tp(
                stream,
                src,
                &format!("{LM_PREFIX}layers.{layer}.mlp"),
                0,
                shard.rank,
                shard.world,
            )?;
            ensure!(m.inter * shard.world == cfg.intermediate_size as usize, "dense mlp inter {}×{}", m.inter, shard.world);
            Ffn::Dense(m)
        };
        let attn_res = if cfg.attn_res_block_size.is_some() {
            Some(AttnRes::load(stream, src, t, layer, cfg)?)
        } else {
            None
        };
        let p = format!("layers.{layer}");
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
        sc: &mut ActScratch,
        out: &mut Tensor<bf16>,
    ) -> Result<()> {
        let h = self.hidden;
        let ActScratch { h_in, hn, a, p, m, mn, y, red_f32, acc, res: res_sc, kda: kda_sc, mla: mla_sc, ffn: ffn_sc, .. } = sc;
        ensure!(t <= hn.shape[0] && hn.shape[1] == h, "layer: {t} > max_tokens / scratch hidden");
        ensure!(x.numel() >= t * h && out.numel() >= t * h, "layer: x/out");
        let writer = self.writes_block();
        let n = t * h;

        // 1. attention 门：B 非空时 h = AttnRes(B ∪ {x})
        let use_res_in = self.attn_res.is_some() && !blocks.is_empty();
        if use_res_in {
            let mut cands: Vec<&Tensor<bf16>> = blocks.to_vec();
            cands.push(x);
            let rs = res_sc.as_mut().ok_or_else(|| anyhow::anyhow!("layer: AttnRes 但 scratch 无 res"))?;
            self.attn_res.as_ref().unwrap().apply(stream, gemm, t, &cands, 0, rs, acc, h_in)?;
        } else {
            ensure!(self.attn_res.is_some() || blocks.is_empty(), "无 AttnRes 的模型不该有块残差");
        }
        let h_src = if use_res_in { &*h_in } else { x };
        // 2. attention
        k::rmsnorm_bf16_n(
            stream,
            &h_src.data,
            &self.input_norm.w.data,
            &mut hn.data,
            t,
            h,
            self.input_norm.eps,
        )?;
        match &mut self.attn {
            Attn::Kda(kda) => kda.forward(stream, gemm, t, hn, &meta.kda, kda_sc, a)?,
            Attn::Mla(mla) => {
                mla.forward(stream, gemm, t, hn, meta.slot_mapping, &meta.mla, mla_sc, a)?
            }
        }
        // all-reduce ①：o_proj 列切的部分和（TP）
        if let Some(c) = meta.comm {
            let red = red_f32.as_mut().ok_or_else(|| anyhow::anyhow!("layer: comm 但 scratch 按 world=1 分配"))?;
            all_reduce_partial_bf16(stream, c, a, n, red).context("layer attn all-reduce")?;
        }
        // 3. prefix_sum：写块层 p = a；否则 p = x + a
        if writer {
            stream.dtod_copy_n(&a.data, &mut p.data, n)?;
        } else {
            stream.dtod_copy_n(&x.data, &mut p.data, n)?;
            k::add_inplace_bf16_n(stream, &a.data, &mut p.data, n)?;
        }
        // 4. mlp 门：m = AttnRes(B' ∪ {p})，B' = writer ? B ∪ {x} : B
        let m_src = if let Some(res) = self.attn_res.as_ref() {
            let mut cands: Vec<&Tensor<bf16>> = blocks.to_vec();
            if writer {
                cands.push(x);
            }
            cands.push(&*p);
            let rs = res_sc.as_mut().ok_or_else(|| anyhow::anyhow!("layer: AttnRes 但 scratch 无 res"))?;
            res.apply(stream, gemm, t, &cands, 1, rs, acc, m)?;
            &*m
        } else {
            &*p
        };
        // 5. mlp
        k::rmsnorm_bf16_n(
            stream,
            &m_src.data,
            &self.post_norm.w.data,
            &mut mn.data,
            t,
            h,
            self.post_norm.eps,
        )?;
        match &mut self.ffn {
            Ffn::LatentMoe(moe) => moe.forward_tp(stream, gemm, t, mn, ffn_sc, acc, red_f32.as_mut(), y, meta.comm)?,
            Ffn::Dense(mlp) => {
                mlp.forward_with_scratch(stream, gemm, &self.act, t, mn, y, &mut ffn_sc.gate_up, &mut ffn_sc.act)?;
                // 稠密层 down 行切的部分和（TP）
                if let Some(c) = meta.comm {
                    let red = red_f32.as_mut().ok_or_else(|| anyhow::anyhow!("layer: comm 但 scratch 按 world=1 分配"))?;
                    all_reduce_partial_bf16(stream, c, y, n, red).context("layer dense all-reduce")?;
                }
            }
        }
        // 6. out = p + y
        stream.dtod_copy_n(&p.data, &mut out.data, n)?;
        k::add_inplace_bf16_n(stream, &y.data, &mut out.data, n)
    }
}
