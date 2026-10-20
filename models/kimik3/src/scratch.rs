//! 全模型共享的一套 per-token 激活缓冲（按 `max_tokens` 预分配一次，93 层轮流借用）。
//!
//! 为什么是一套而不是每层一套：层在同一条流上串行执行，层 i 的中间量在层 i+1 开始前已经消费完，不存在同时存活；
//! 每层各持一套时（k3-serve 3b 根因），预算 16384 下 KDA 13 块 `[T, 12288]` ≈ 5.2 GB、LatentMoe 四块 `[T·16, …]` ≈ 4.3 GB、
//! shared experts 自带两块 `[T, 2I / I]`，一层 ≈ 13 GB，93 层 tp8 必 OOM。共享后 tp8 每 token ≈ 0.8 MB（`bytes` 可算）。
//!
//! 合并规则（照 qwen35 `scratch.rs`）：同一角色的缓冲按各层形状取最大列数开一块；kernel 只按 `t` 行 × 本层列数读写前缀，
//! 多出的列 / 行是尾部死区。本文件里只做**生命周期显然不重叠**的合并：
//! - `ffn.gate_up` / `ffn.act`：dense Mlp（`[T, 2I/w]`）与 MoE shared experts（`[T, 2·Is·Ns/w]`）取大，二者不同层；
//! - `ffn.zg`：MoE gather 输入 `[T·K, latent]` 在 w13 GEMM 后即死，w2 GEMM 的输出 `[T·K, latent]` 复用同一块；
//! - `acc`：AttnRes 的 `[T, hidden]` f32 与 MoE 的 `[T, latent]` f32 取大（AttnRes 在 attention 前 / MoE 前各算完即消费）；
//! - `red_f32`：layer（attn / dense all-reduce）与 LatentMoe（shared all-reduce）共用（`TP_REDUCE_F32`），world=1 不开。
//!
//! KDA / MLA 两组不互相别名（命名会失去可读性，且合计只占 tp8 总量 ~15%）。
//! `pairs = max_tokens × topk` 四块按"全部 pair 都落在本 rank"开（EP 不能缩，每步各 rank 实际段长不可预知）。
use crate::config::KimiLinearConfig;
use crate::weights::{KdaShapes, Shard};
use anyhow::Result;
use dinfer_kernels::ops as k;
use dinfer_layers::{bf16, Stream, Tensor};

/// 块残差门 `AttnRes::apply` 的工作区（`attn_res_block_size` 为 None 的模型不开）。
pub struct ResScratch {
    /// 候选的 RMSNorm 输出 `[T, hidden]`。
    pub nrm: Tensor<bf16>,
    /// 每候选的门分数 `[T]` f32。
    pub score: Tensor<f32>,
    /// host softmax 后的权重 `[T, R_max]` f32。
    pub probs: Tensor<f32>,
    /// `[R_max·T]`：块 r 内第 i 项 = i / 全为 r（scatter_add 的索引表，常量）。
    pub ident: Tensor<i32>,
    pub kidx: Tensor<i32>,
    pub max_cands: usize,
}

/// KDA 层（`KdaLayer::forward`）：`proj = H·K / world`。
pub struct KdaScratch {
    pub q: Tensor<bf16>,
    pub k: Tensor<bf16>,
    pub v: Tensor<bf16>,
    pub qc: Tensor<bf16>,
    pub kc: Tensor<bf16>,
    pub vc: Tensor<bf16>,
    /// g_proj(x)，输出门 `[T, proj]`。
    pub g2: Tensor<bf16>,
    /// f_a(x) `[T, K]`。
    pub fa: Tensor<bf16>,
    /// f_b(f_a(x))，raw gate `[T, proj]`。
    pub g1: Tensor<bf16>,
    /// b_proj(x) `[T, H]` 与其 f32 副本。
    pub b: Tensor<bf16>,
    pub beta: Tensor<f32>,
    /// KDA 核心输出 / o_norm 输出 `[T, proj]`。
    pub o: Tensor<bf16>,
    pub n: Tensor<bf16>,
}

/// MLA 层（`MlaLayer::forward`）：`heads = H / world`。
pub struct MlaScratch {
    pub fused_out: Tensor<bf16>,
    pub q_c_raw: Tensor<bf16>,
    pub q_c: Tensor<bf16>,
    pub q: Tensor<bf16>,
    pub kv_c_raw: Tensor<bf16>,
    pub kv_c: Tensor<bf16>,
    pub k_pe: Tensor<bf16>,
    pub gate: Tensor<bf16>,
    pub latent: Tensor<bf16>,
    pub kv_full: Tensor<bf16>,
    pub k_full: Tensor<bf16>,
    pub v_full: Tensor<bf16>,
    pub attn_out: Tensor<bf16>,
    pub q_lat: Tensor<bf16>,
    pub o_lat: Tensor<bf16>,
    pub head_in: Tensor<bf16>,
    pub head_out: Tensor<bf16>,
}

/// FFN（LatentMoe + dense Mlp 共用）。
pub struct FfnScratch {
    /// 路由：fp32 logits `[T, E]`、top-k `[T, K]`。
    pub logits: Tensor<f32>,
    pub topk_ids: Tensor<i32>,
    pub topk_w: Tensor<f32>,
    /// 分桶（sorted_tok / sorted_k `[T·K]`、counts / offsets `[E(+1)]`）。
    pub buckets: k::MoeBuckets,
    /// down_proj 输出 `[T, latent]`。
    pub z: Tensor<bf16>,
    /// gather 的 `[T·K, latent]`；w13 GEMM 消费后即死，w2 GEMM 的输出 `[T·K, latent]` 复用这一块。
    pub zg: Tensor<bf16>,
    /// MoE：w13 GEMM 输出 `[T·K, 2·Im]` / situ 输出 `[T·K, Im]`；dense Mlp / shared experts：`[T, 2I/w]` / `[T, I/w]`。按 numel 取大。
    pub gate_up: Tensor<bf16>,
    pub act: Tensor<bf16>,
    /// 加权和（norm 前）/ norm 后 `[T, latent]`；routed up_proj 输出 `[T, hidden]`。
    pub ysum: Tensor<bf16>,
    pub yn: Tensor<bf16>,
    pub yup: Tensor<bf16>,
}

pub struct ActScratch {
    /// layer 级 `[T, hidden]`：attn 门输出 / input_norm 输出 / attn 输出 / prefix_sum / mlp 门输出 / post_norm 输出 / ffn 输出。
    pub h_in: Tensor<bf16>,
    pub hn: Tensor<bf16>,
    pub a: Tensor<bf16>,
    pub p: Tensor<bf16>,
    pub m: Tensor<bf16>,
    pub mn: Tensor<bf16>,
    pub y: Tensor<bf16>,
    /// `TP_REDUCE_F32` 的 all-reduce 工作区 `[T, hidden]` f32（world=1 不开）。
    pub red_f32: Option<Tensor<f32>>,
    /// f32 累加区：AttnRes `[T, hidden]` 与 MoE scatter_add `[T, latent]` 取大。
    pub acc: Tensor<f32>,
    pub res: Option<ResScratch>,
    pub kda: KdaScratch,
    pub mla: MlaScratch,
    pub ffn: FfnScratch,
    pub max_tokens: usize,
}

/// 各缓冲的列数（分配前就能算）。
struct Cols {
    hidden: usize,
    max_cands: usize,
    kda_proj: usize,
    kda_h: usize,
    kda_k: usize,
    mla_heads: usize,
    q_lora: usize,
    kv_lora: usize,
    nope: usize,
    rope: usize,
    v_dim: usize,
    experts: usize,
    topk: usize,
    latent: usize,
    moe_inter: usize,
    /// dense Mlp 与 shared experts 的本 rank inter 取大（gate_up = 2×，act = 1×）。
    ffn_inter: usize,
}

impl Cols {
    fn new(cfg: &KimiLinearConfig, shard: Shard) -> Self {
        let kda = KdaShapes::with_shard(cfg, shard);
        let hidden = cfg.hidden_size as usize;
        let max_blocks = match cfg.attn_res_block_size {
            Some(bs) => (cfg.num_hidden_layers as usize).div_ceil(bs as usize),
            None => 0,
        };
        let moe_inter = cfg.moe_intermediate_size as usize;
        let dense_inter = shard.part(cfg.intermediate_size as usize);
        let shared_inter = shard.part(moe_inter * cfg.num_shared_experts as usize);
        Self {
            hidden,
            max_cands: max_blocks + 1,
            kda_proj: kda.proj(),
            kda_h: kda.heads,
            kda_k: kda.head_dim,
            mla_heads: shard.part(cfg.num_attention_heads as usize),
            q_lora: cfg.q_lora_rank.unwrap_or(0) as usize,
            kv_lora: cfg.kv_lora_rank as usize,
            nope: cfg.qk_nope_head_dim as usize,
            rope: cfg.qk_rope_head_dim as usize,
            v_dim: cfg.v_head_dim as usize,
            experts: cfg.num_experts as usize,
            topk: cfg.num_experts_per_token as usize,
            latent: cfg.routed_expert_hidden_size.unwrap_or(0) as usize,
            moe_inter,
            ffn_inter: dense_inter.max(shared_inter),
        }
    }
    fn entry(&self) -> usize {
        self.kv_lora + self.rope
    }
    fn wide(&self) -> usize {
        self.q_lora + self.entry() + self.mla_heads * self.v_dim
    }
    /// `gate_up` / `act` 的 numel 除以 T：MoE 的 `[T·K, 2·Im]` 与 dense/shared 的 `[T, 2·I]` 取大。
    fn gate_up_per_tok(&self) -> usize {
        (self.topk * 2 * self.moe_inter).max(2 * self.ffn_inter)
    }
    fn act_per_tok(&self) -> usize {
        (self.topk * self.moe_inter).max(self.ffn_inter)
    }
    fn acc_cols(&self) -> usize {
        self.hidden.max(self.latent)
    }
}

impl ActScratch {
    /// 一套的总字节数（分配前就能算，日志 / 预算 / 单测用）。与 `new` 的分配逐项对应。
    pub fn bytes(cfg: &KimiLinearConfig, max_tokens: usize, shard: Shard) -> usize {
        let c = Cols::new(cfg, shard);
        let t = max_tokens;
        let (h, cands) = (c.hidden, c.max_cands);
        // layer 级 7 块 bf16 + acc f32（+ red_f32）
        let mut b = 7 * t * h * 2 + t * c.acc_cols() * 4;
        if !shard.is_single() {
            b += t * h * 4;
        }
        if cfg.attn_res_block_size.is_some() {
            b += t * h * 2 + t * 4 + t * cands * 4 + 2 * cands * t * 4;
        }
        // KDA：10 块 [T, proj] + fa [T, K] + b [T, H] bf16 + beta [T, H] f32
        b += (10 * c.kda_proj + c.kda_k + c.kda_h) * t * 2 + c.kda_h * t * 4;
        // MLA
        let (hd, qk, en) = (c.mla_heads, c.nope + c.rope, c.entry());
        let mla_cols = c.wide()
            + 2 * c.q_lora
            + hd * qk
            + 2 * c.kv_lora
            + c.rope
            + hd * c.v_dim
            + en
            + hd * (c.nope + c.v_dim)
            + hd * qk
            + hd * c.v_dim
            + hd * c.v_dim
            + hd * en
            + hd * c.kv_lora
            + c.kv_lora.max(c.nope)
            + c.kv_lora.max(c.v_dim);
        b += mla_cols * t * 2;
        // FFN
        let pairs = t * c.topk;
        b += t * c.experts * 4 + pairs * 8; // logits + topk_ids/topk_w
        b += (2 * c.experts + 1 + 2 * c.experts) * 4 + 2 * pairs * 4; // buckets：counts/offsets/cursor + sorted_tok/sorted_k
        b += (c.latent + c.topk * c.latent + c.gate_up_per_tok() + c.act_per_tok() + 2 * c.latent + h) * t * 2;
        b
    }

    pub fn new(stream: &Stream, cfg: &KimiLinearConfig, max_tokens: usize, shard: Shard) -> Result<Self> {
        let c = Cols::new(cfg, shard);
        let t = max_tokens;
        let h = c.hidden;
        let z = |s: &Stream| Tensor::<bf16>::zeros(s, &[t, h]);
        let res = if cfg.attn_res_block_size.is_some() {
            let cands = c.max_cands;
            let mut ident = Vec::with_capacity(cands * t);
            let mut kidx = Vec::with_capacity(cands * t);
            for r in 0..cands {
                for i in 0..t {
                    ident.push(i as i32);
                    kidx.push(r as i32);
                }
            }
            Some(ResScratch {
                nrm: z(stream)?,
                score: Tensor::zeros(stream, &[t])?,
                probs: Tensor::zeros(stream, &[t, cands])?,
                ident: Tensor::from_vec(stream, ident, &[cands * t])?,
                kidx: Tensor::from_vec(stream, kidx, &[cands * t])?,
                max_cands: cands,
            })
        } else {
            None
        };
        let (proj, kh, kd) = (c.kda_proj, c.kda_h, c.kda_k);
        let kda = KdaScratch {
            q: Tensor::zeros(stream, &[t, proj])?,
            k: Tensor::zeros(stream, &[t, proj])?,
            v: Tensor::zeros(stream, &[t, proj])?,
            qc: Tensor::zeros(stream, &[t, proj])?,
            kc: Tensor::zeros(stream, &[t, proj])?,
            vc: Tensor::zeros(stream, &[t, proj])?,
            g2: Tensor::zeros(stream, &[t, proj])?,
            fa: Tensor::zeros(stream, &[t, kd])?,
            g1: Tensor::zeros(stream, &[t, proj])?,
            b: Tensor::zeros(stream, &[t, kh])?,
            beta: Tensor::zeros(stream, &[t, kh])?,
            o: Tensor::zeros(stream, &[t, proj])?,
            n: Tensor::zeros(stream, &[t, proj])?,
        };
        let (hd, nope, rope, v_dim, q_lora, kv_lora, en) =
            (c.mla_heads, c.nope, c.rope, c.v_dim, c.q_lora, c.kv_lora, c.entry());
        let mla = MlaScratch {
            fused_out: Tensor::zeros(stream, &[t, c.wide()])?,
            q_c_raw: Tensor::zeros(stream, &[t, q_lora])?,
            q_c: Tensor::zeros(stream, &[t, q_lora])?,
            q: Tensor::zeros(stream, &[t, hd, nope + rope])?,
            kv_c_raw: Tensor::zeros(stream, &[t, kv_lora])?,
            kv_c: Tensor::zeros(stream, &[t, kv_lora])?,
            k_pe: Tensor::zeros(stream, &[t, rope])?,
            gate: Tensor::zeros(stream, &[t, hd * v_dim])?,
            latent: Tensor::zeros(stream, &[t, en])?,
            kv_full: Tensor::zeros(stream, &[t, hd, nope + v_dim])?,
            k_full: Tensor::zeros(stream, &[t, hd, nope + rope])?,
            v_full: Tensor::zeros(stream, &[t, hd, v_dim])?,
            attn_out: Tensor::zeros(stream, &[t, hd, v_dim])?,
            q_lat: Tensor::zeros(stream, &[t, hd, en])?,
            o_lat: Tensor::zeros(stream, &[t, hd, kv_lora])?,
            head_in: Tensor::zeros(stream, &[t, kv_lora.max(nope)])?,
            head_out: Tensor::zeros(stream, &[t, kv_lora.max(v_dim)])?,
        };
        let (e, kk, latent) = (c.experts, c.topk, c.latent);
        let ffn = FfnScratch {
            logits: Tensor::zeros(stream, &[t, e])?,
            topk_ids: Tensor::zeros(stream, &[t, kk])?,
            topk_w: Tensor::zeros(stream, &[t, kk])?,
            buckets: k::MoeBuckets::new(stream, e, t, kk)?,
            z: Tensor::zeros(stream, &[t, latent])?,
            zg: Tensor::zeros(stream, &[t * kk, latent])?,
            gate_up: Tensor::zeros(stream, &[t, c.gate_up_per_tok()])?,
            act: Tensor::zeros(stream, &[t, c.act_per_tok()])?,
            ysum: Tensor::zeros(stream, &[t, latent])?,
            yn: Tensor::zeros(stream, &[t, latent])?,
            yup: Tensor::zeros(stream, &[t, h])?,
        };
        Ok(Self {
            h_in: z(stream)?,
            hn: z(stream)?,
            a: z(stream)?,
            p: z(stream)?,
            m: z(stream)?,
            mn: z(stream)?,
            y: z(stream)?,
            red_f32: if shard.is_single() { None } else { Some(Tensor::zeros(stream, &[t, h])?) },
            acc: Tensor::zeros(stream, &[t, c.acc_cols()])?,
            res,
            kda,
            mla,
            ffn,
            max_tokens: t,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::KimiK3Config;

    fn real_cfg() -> Option<KimiLinearConfig> {
        let p = "/data/ref/models/kimik3/config.json";
        let Ok(s) = std::fs::read_to_string(p) else {
            eprintln!("skip: {p} not present");
            return None;
        };
        Some(KimiK3Config::from_json(&s).unwrap().text_config)
    }

    /// 全模型一套（不再乘层数）：真 K3 config 下的字节数要在 KV 预算之外留得住。
    /// 旧口径（k3-serve 3b）每层 ≈ 13 GB，93 层 tp8 预算 16384 ≈ 1.2 TB；共享后 tp8 16384 < 14 GiB、2048 < 2 GiB。
    #[test]
    fn bytes_real_config_bounded() {
        let Some(cfg) = real_cfg() else { return };
        let gib = 1usize << 30;
        let tp8 = Shard::new(0, 8).unwrap();
        let b_2048 = ActScratch::bytes(&cfg, 2048, tp8);
        let b_16384 = ActScratch::bytes(&cfg, 16384, tp8);
        let b_1 = ActScratch::bytes(&cfg, 16384, Shard::SINGLE);
        eprintln!(
            "tp8 2048 = {:.2} GiB, tp8 16384 = {:.2} GiB, world=1 16384 = {:.2} GiB",
            b_2048 as f64 / gib as f64,
            b_16384 as f64 / gib as f64,
            b_1 as f64 / gib as f64
        );
        assert!(b_2048 < 2 * gib, "tp8 2048: {b_2048}");
        assert!(b_16384 < 14 * gib, "tp8 16384: {b_16384}");
        // 线性于 max_tokens（没有按层数乘的项；只有 buckets 的 counts/offsets/cursor `[E(+1)]` 是常数项）
        let fixed = (2 * cfg.num_experts as usize + 1 + 2 * cfg.num_experts as usize) * 4;
        assert_eq!(b_16384 - fixed, 8 * (b_2048 - fixed));
        // 每层旧口径下界：仅 LatentMoe 四块 pairs + KDA 13 块就 ≈ 9.5 GB/层；共享一套必须远小于 93 层中任意 8 层的和
        let old_per_layer = 16384 * 16 * (3584 + 6144 + 3072 + 3584) * 2 + 13 * 16384 * (12288 / 8) * 2;
        assert!(b_16384 < 2 * old_per_layer, "{b_16384} vs old/layer {old_per_layer}");
    }

    /// `bytes` 与 `new` 的分项逐一对应：改一处必须改另一处。这里用列数公式再算一遍（独立于 `bytes` 的累加写法）。
    #[test]
    fn bytes_matches_itemized_formula() {
        let Some(cfg) = real_cfg() else { return };
        let t = 2048usize;
        let sh = Shard::new(0, 8).unwrap();
        let c = Cols::new(&cfg, sh);
        assert_eq!((c.kda_proj, c.kda_h, c.kda_k), (12288 / 8, 96 / 8, 128));
        assert_eq!((c.mla_heads, c.q_lora, c.kv_lora, c.nope, c.rope, c.v_dim), (12, 1536, 512, 128, 64, 128));
        assert_eq!((c.experts, c.topk, c.latent, c.moe_inter), (896, 16, 3584, 3072));
        // dense inter 33792/8 = 4224；shared 2×3072/8 = 768 → ffn_inter = 4224；gate_up 按 MoE pairs 16×2×3072 = 98304 > 2×4224
        assert_eq!(c.ffn_inter, 4224);
        assert_eq!(c.gate_up_per_tok(), 98304);
        assert_eq!(c.act_per_tok(), 49152);
        assert_eq!(c.acc_cols(), 7168);
        assert_eq!(c.max_cands, 9);
        let b = ActScratch::bytes(&cfg, t, sh);
        // 每 token 字节（tp8）：layer 7×7168×2 + acc 7168×4 + red 7168×4 + res (7168×2 + 4 + 9×4 + 2×9×4)
        let layer = 7 * 7168 * 2 + 7168 * 4 + 7168 * 4 + (7168 * 2 + 4 + 9 * 4 + 2 * 9 * 4);
        let kda = (10 * 1536 + 128 + 12) * 2 + 12 * 4;
        let mla = (1536 + 512 + 64 + 12 * 128 // fused_out
            + 2 * 1536 + 12 * 192 + 2 * 512 + 64 + 12 * 128 + 576 + 12 * 256 + 12 * 192 + 12 * 128 + 12 * 128
            + 12 * 576 + 12 * 512 + 512 + 512)
            * 2;
        let ffn = 896 * 4 + 16 * 8 + 2 * 16 * 4 + (3584 + 16 * 3584 + 98304 + 49152 + 2 * 3584 + 7168) * 2;
        let per_tok = layer + kda + mla + ffn;
        let fixed = (2 * 896 + 1 + 2 * 896) * 4;
        assert_eq!(b, per_tok * t + fixed, "per_tok={per_tok}");
    }
}
