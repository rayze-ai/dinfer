//! Kaon-V3 模型主体：embedding → 30 层 `Gemma4Layer` → final norm → tied lm_head → softcap。
//!
//! 结构事实（`docs/loops/kaon-v3.md`「Gemma4 前向语义」）：
//! - embedding 后乘 `sqrt(hidden_size)`（vLLM 的 `normalizer`）；
//! - `tie_word_embeddings = true`：lm_head 与 embedding 共用同一张表；
//! - `final_logit_softcapping = 30`：`logits = tanh(logits/30)·30`；
//! - 每层 KV cache 的 `head_dim` 不同（滑窗 256 / 全注意力 512），必须**逐层分配**。

use crate::config::KaonConfig;
use crate::layer::{ActBuf, Gemma4Layer, LayerScratch, Ropes, StepCtx};
use crate::weights::{self, QWeight};
use anyhow::{anyhow, ensure, Result};
use dinfer_kernels::{self as k, bf16, Gemm, Stream};
use dinfer_layers::{Embedding, Norm as _, OpSet, Tensor, WeightSource};

/// 一层的 paged KV cache：[num_blocks, block_size, nkv, 2, head_dim]。
pub struct LayerKv {
    /// B15：`is_fp8` 时 cache 以 e4m3 字节存，`cache` 的 bf16 长度 = 元素数 / 2（kernel 按 u8 解释）。
    pub cache: Tensor<bf16>,
    pub is_fp8: bool,
    pub nkv: usize,
    pub head_dim: usize,
}

impl LayerKv {
    pub fn new(stream: &Stream, cfg: &KaonConfig, l: usize, num_blocks: usize, block_size: usize) -> Result<Self> {
        let (head_dim, nkv, _) = cfg.attn_shape(l);
        // B15/B15b：kv_fp8 层默认真存 e4m3（vLLM fp8 KV 口径；D70：c8 decode -2%、TTFT +4%、cache 容量翻倍）。`KV_FP8_STORE=0` 回退 bf16 存。
        let is_fp8 = cfg.kv_fp8 && std::env::var("KV_FP8_STORE").ok().as_deref() != Some("0");
        let elems = num_blocks * block_size * nkv * 2 * head_dim;
        let cache = if is_fp8 {
            Tensor::zeros(stream, &[elems / 2])?
        } else {
            Tensor::zeros(stream, &[num_blocks, block_size, nkv, 2, head_dim])?
        };
        Ok(Self { cache, is_fp8, nkv, head_dim })
    }
}

pub struct KaonModel {
    pub cfg: KaonConfig,
    pub embed: Embedding,
    pub layers: Vec<Gemma4Layer>,
    pub final_norm: Tensor<bf16>,
    pub kv: Vec<LayerKv>,
    pub ropes: Ropes,
    pub block_size: usize,
    pub num_blocks: usize,
    /// embedding 后的缩放系数（`sqrt(hidden_size)`）。
    pub embed_scale: f32,
}

impl KaonModel {
    /// 只装权重结构（不含 KV cache，KV 由 `alloc_kv` 单独按显存预算分配）。
    pub fn load(stream: &Stream, src: &WeightSource, cfg: KaonConfig) -> Result<Self> {
        let embed_name = weights::EMBED;
        let embed = Embedding::load(stream, src, &format!("{}{}", weights::TEXT_PREFIX, "embed_tokens"))?;
        ensure!(embed.vocab == cfg.vocab_size, "vocab {} != {}", embed.vocab, cfg.vocab_size);

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for l in 0..cfg.num_layers {
            layers.push(Gemma4Layer::load(stream, src, &cfg, l)?);
        }
        let final_norm = src.load_bf16(stream, weights::NORM)?;
        ensure!(final_norm.numel() == cfg.hidden, "final_norm 长度 {} != hidden {}", final_norm.numel(), cfg.hidden);
        let _ = embed_name;
        let ropes = Ropes::new(stream, &cfg)?;
        Ok(Self {
            // vLLM：normalizer 以模型 dtype 存（"avoid mixed-precision drift"），sqrt(2816)=53.07 → bf16 53.0
            embed_scale: half::bf16::from_f32((cfg.hidden as f32).sqrt()).to_f32(),
            cfg,
            embed,
            layers,
            final_norm,
            kv: Vec::new(),
            ropes,
            block_size: 0,
            num_blocks: 0,
        })
    }

    /// cuBLASLt autotune 条目：kaon3 decode 唯一走 `forward_bf16`(cuBLASLt) 的是 tied lm_head
    /// （[tokens, vocab] = final_normed × embed.table^T）。fp8 投影 / MoE grouped 走自写 kernel 无 algo 选择，不入表。
    /// executor 新 Gemm 后、CUDA graph 捕获前调一次，使 graph 与 eager fallback 共享同一 algo（否则各自启发式选到不同 algo → logits 末位差 → 近并列 token 翻转）。
    pub fn gemm_tune_entries(&self) -> Vec<dinfer_kernels::GemmTuneEntry<'_>> {
        vec![dinfer_kernels::GemmTuneEntry {
            name: "lm_head".to_string(),
            n: self.embed.vocab,
            k: self.embed.hidden,
            w: &self.embed.table.data,
            bias: None,
        }]
    }

    /// 按 `num_blocks` 分配每层 KV cache（逐层 head_dim 不同）。
    pub fn alloc_kv(&mut self, stream: &Stream, num_blocks: usize, block_size: usize) -> Result<()> {
        ensure!(block_size > 0 && num_blocks > 0, "kv: 参数非法");
        let mut kv = Vec::with_capacity(self.cfg.num_layers);
        for l in 0..self.cfg.num_layers {
            kv.push(LayerKv::new(stream, &self.cfg, l, num_blocks, block_size)?);
        }
        self.kv = kv;
        self.num_blocks = num_blocks;
        self.block_size = block_size;
        Ok(())
    }

    /// 全模型前向：返回 `[tokens, vocab]` logits（bf16，softcap 已施加）。
    ///
    /// `tokens` 为本步 token 数；`ids` 为对应 token id（长度 ≥ tokens）。
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        stream: &Stream,
        gemm: &Gemm,
        ops: &impl OpSet,
        ids: &Tensor<i32>,
        meta: &StepMeta<'_>,
        sc: &mut LayerScratch,
        acts: &mut ActBuf,
        logits: &mut Tensor<bf16>,
        hidden: &mut Tensor<bf16>,
        tokens: usize,
        // `Some(l)` 时把第 l 层之后的残差流拷进 `hidden`（逐层对拍用）。
        dump_layer: Option<usize>,
        // `Some(dir)` 时把**每一层**之后的残差流写 `<dir>/L{l}.bin`（f32 LE，[T,H]），
        // 与 `scripts/dev/kaon3_vllm_hidden.py` 抓的 vLLM 逐层残差流同名同序逐层比。
        hidden_dir: Option<&str>,
        // `Some((data, layer, skip))`：**单层隔离**测试。把 `data`（[T,H] f32）灌进残差流当作
        // 第 `layer` 层的输入，只跑 `layer` 这一层（`skip` 个层先跳过），输出写 `hidden_dir/L{layer}.bin`。
        // 用来把逐层累积误差剥掉，单独判断某一层的实现对不对（输入取自参照，输出与参照比）。
        inject: Option<(&[f32], usize, usize)>,
    ) -> Result<()> {
        let cfg = &self.cfg;
        let h = cfg.hidden;

        // ---- embedding × sqrt(hidden) ----
        self.embed.forward_rows(stream, ids, &mut sc.x, tokens)?;
        if self.embed_scale != 1.0 {
            k::ops::scale_bf16_n(stream, &mut sc.x.data, tokens * h, self.embed_scale)?;
        }
        // L-1 = embed 之后（layer 0 之前的残差流），用于把"embed 错"与"layer 0 错"分开。
        if let Some(dir) = hidden_dir {
            stream.dtod_copy_n(&sc.x.data, &mut hidden.data, tokens * h)?;
            let v = hidden.to_vec_f32(stream)?;
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(format!("{dir}/Lm1.bin"), bytes)?;
        }

        // 单层隔离：把参照的残差流灌进来，只跑指定那一层。
        if let Some((data, layer, skip)) = inject {
            ensure!(data.len() >= tokens * h, "inject: 数据长度 {} < {}", data.len(), tokens * h);
            // 参照的残差流是 f32，模型 buffer 是 bf16：逐元素转换后上传（与 kernel 的舍入口径一致）。
            let up: Vec<bf16> = data[..tokens * h].iter().map(|x| bf16::from_f32(*x)).collect();
            let up = Tensor::from_vec(stream, up, &[tokens, h])?;
            stream.dtod_copy_n(&up.data, &mut sc.x.data, tokens * h)?;
            let mut kv_cache = std::mem::take(&mut self.kv);
            let l = layer;
            // 前置各层照跑，为第 l 层的注意力填好 KV（它们的输出不进入本层输入，输入是被灌进去的）。
            for pre in 0..skip {
                if pre == l { continue; }  // 第 l 层自己只跑一次（用灌入的输入）
                let kv_fp8_pre = kv_cache[pre].is_fp8;
                let mut c = StepCtx {
                    mode: meta.mode,
                    cu_seqlens: &meta.cu_seqlens,
                    seq_lens: &meta.seq_lens,
                    block_table: &meta.block_table,
                    positions: &meta.positions,
                    slot_mapping: &meta.slot_mapping,
                    num_reqs: meta.num_reqs,
                    max_blocks: meta.max_blocks,
                    max_seqlen: meta.max_seqlen,
                    block_size: self.block_size,
                    kv_mut: &mut kv_cache[pre].cache.data,
                    kv_fp8_store: kv_fp8_pre,
                };
                self.layers[pre].forward(stream, gemm, ops, cfg, &self.ropes, sc, &mut c, tokens, None)?;
            }
            let kv_fp8_l = kv_cache[l].is_fp8;
            let mut ctx = StepCtx {
                mode: meta.mode,
                cu_seqlens: &meta.cu_seqlens,
                seq_lens: &meta.seq_lens,
                block_table: &meta.block_table,
                positions: &meta.positions,
                slot_mapping: &meta.slot_mapping,
                num_reqs: meta.num_reqs,
                max_blocks: meta.max_blocks,
                max_seqlen: meta.max_seqlen,
                block_size: self.block_size,
                kv_mut: &mut kv_cache[l].cache.data,
                kv_fp8_store: kv_fp8_l,
            };
            let so_l: usize = std::env::var("SUB_LAYER").ok().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
            let so = hidden_dir.filter(|_| l == so_l).map(|d| (std::path::Path::new(d), l));
            self.layers[l].forward(stream, gemm, ops, cfg, &self.ropes, sc, &mut ctx, tokens, so)?;
            self.kv = kv_cache;
            stream.dtod_copy_n(&sc.x.data, &mut hidden.data, tokens * h)?;
            return Ok(());
        }

        // ---- 30 层 ----
        // KV cache 逐层不同（head_dim 256/512），写与读都要按层取。
        let mut kv_cache = std::mem::take(&mut self.kv);
        for (l, layer) in self.layers.iter().enumerate() {
            let kv_fp8_l = kv_cache[l].is_fp8;
            let mut ctx = StepCtx {
                mode: meta.mode,
                cu_seqlens: &meta.cu_seqlens,
                seq_lens: &meta.seq_lens,
                block_table: &meta.block_table,
                positions: &meta.positions,
                slot_mapping: &meta.slot_mapping,
                num_reqs: meta.num_reqs,
                max_blocks: meta.max_blocks,
                max_seqlen: meta.max_seqlen,
                block_size: self.block_size,
                kv_mut: &mut kv_cache[l].cache.data,
                kv_fp8_store: kv_fp8_l,
            };
            // 子模块 dump 的层选择：`SUB_LAYER=<n>` 覆盖，默认第 0 层；
            // `SUB_LAYER=all` 每层都 dump，写到 `<dir>/sub_L{l}/`（逐层 router 真值扫描用）。
            let sub_env = std::env::var("SUB_LAYER").ok();
            let sub_l: usize = sub_env.as_deref().and_then(|v| v.parse().ok()).unwrap_or(0);
            let sub_dir_l = hidden_dir
                .filter(|_| sub_env.as_deref() == Some("all"))
                .map(|d| std::path::Path::new(d).join(format!("sub_L{l}")));
            if let Some(p) = &sub_dir_l {
                std::fs::create_dir_all(p)?;
            }
            let so = match &sub_dir_l {
                Some(p) => Some((p.as_path(), l)),
                None => hidden_dir.filter(|_| l == sub_l).map(|d| (std::path::Path::new(d), l)),
            };
            layer.forward(stream, gemm, ops, cfg, &self.ropes, sc, &mut ctx, tokens, so)?;
            if dump_layer == Some(l) {
                stream.dtod_copy_n(&sc.x.data, &mut hidden.data, tokens * h)?;
            }
            if let Some(dir) = hidden_dir {
                stream.dtod_copy_n(&sc.x.data, &mut hidden.data, tokens * h)?;
                let v = hidden.to_vec_f32(stream)?;
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(format!("{dir}/L{l}.bin"), bytes)
                    .map_err(|e| {
                        anyhow::anyhow!("写 {dir}/L{l}.bin 失败: {e}")
                    })?;
            }
        }

        self.kv = kv_cache;

        // ---- final norm ----
        // 固定缓冲（CUDA graph 捕获期不能每步 `Tensor::uninit`）。
        ops.norm().rmsnorm(stream, &sc.x.data, &self.final_norm.data, &mut sc.final_normed.data, tokens, h, cfg.eps)?;
        if let Some(dir) = hidden_dir {
            // final norm 之后、lm_head 之前的输入（验证 lm_head GEMM 本身）
            let v = sc.final_normed.to_vec_f32(stream)?;
            let bytes: Vec<u8> = v.iter().take(tokens * h).flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(format!("{dir}/final_norm.bin"), bytes)?;
        }

        // ---- tied lm_head ----
        // vocab 262144、hidden 2816：一次全词表 GEMM。decode（tokens 少）时应换 split/分块策略，
        // 阶段 5 调优时再处理（`docs/loops/kaon-v3.md` 的 roofline 一节）。
        gemm.forward_bf16(
            &sc.final_normed.data,
            &self.embed.table.data,
            None,
            &mut logits.data,
            tokens,
            cfg.vocab_size,
            h,
        )?;

        // ---- softcap: tanh(x/30)·30 ----
        // 舍入次数：**单次**（`to_bf16(c*tanhf(v/c))`）。曾按 vLLM `LogitsProcessor` 的
        // `/c → tanh → *c` 三步各回 bf16 做过 A/B，实测对 parity 指标无影响
        // （翻转率 15.32% 不变、Δlp 1.2220→1.2228），已回滚，不留分支。
        k::ops::softcap_bf16_n(stream, &mut logits.data, tokens * cfg.vocab_size, cfg.softcap)?;
        let _ = hidden;
        Ok(())
    }
}

/// 一步的 batch 元数据（由 executor 上传）。
pub struct StepMeta<'a> {
    pub mode: dinfer_core::runner::BatchMode,
    pub cu_seqlens: &'a Tensor<i32>,
    pub seq_lens: &'a Tensor<i32>,
    pub block_table: &'a Tensor<i32>,
    pub positions: &'a Tensor<i32>,
    pub slot_mapping: &'a Tensor<i64>,
    pub num_reqs: usize,
    pub max_blocks: usize,
    pub max_seqlen: usize,
}

/// 权重清单核对：跑之前确认三件事（避免"编错 feature 空跑"这类假数据）。
pub fn check_weights(src: &WeightSource, cfg: &KaonConfig) -> Result<()> {
    ensure!(src.contains(weights::EMBED), "缺 embedding：{}", weights::EMBED);
    ensure!(src.contains(weights::NORM), "缺 final norm：{}", weights::NORM);
    let l0 = weights::layer_prefix(0);
    for n in [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "pre_feedforward_layernorm.weight",
        "pre_feedforward_layernorm_2.weight",
        "post_feedforward_layernorm.weight",
        "post_feedforward_layernorm_1.weight",
        "post_feedforward_layernorm_2.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "router.proj.weight",
        "router.scale",
        "router.per_expert_scale",
        "layer_scalar",
    ] {
        ensure!(src.contains(&format!("{l0}.{n}")), "缺第 0 层的 {n}");
    }
    ensure!(src.contains(&format!("{l0}.experts.0.gate_proj.weight")), "缺第 0 层专家 0");
    ensure!(cfg.layer_types.len() == cfg.num_layers, "layer_types 长度不符");
    Ok(())
}

/// 权重格式统计（启动日志用）。
pub fn format_summary(src: &WeightSource, name: &str) -> &'static str {
    match name {
        _ if src.info(&format!("{name}_scale")).map(|i| i.shape.len() == 2).unwrap_or(false) => "NVFP4",
        _ => match src.info(name).map(|i| i.dtype.as_str()) {
            Some("F8_E4M3") => "FP8",
            Some("BF16") => "BF16",
            _ => "?",
        },
    }
}

/// 未使用但保留：确认 `QWeight` 的三态与加载分派一致（编译期契约）。
const _: fn(QWeight) = |_q| ();
