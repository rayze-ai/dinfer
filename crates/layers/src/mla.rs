//! Multi-head Latent Attention（DeepSeek-V2/V3 / GLM-5.3 / Kimi-K3 共用结构）。
//!
//! 对照 HF `DeepseekV3Attention` 与 vLLM `MLAAttention`：
//! ```text
//! qkv_lora = fused_qkv_a(x)                  // [T, q_lora + kv_lora + rope]
//! q_c  = rmsnorm(qkv_lora[:, :q_lora])        // q_a_layernorm
//! q    = q_b(q_c).view(T, H, nope + rope)
//! kv_c = rmsnorm(qkv_lora[:, q_lora : q_lora+kv_lora])   // kv_a_layernorm
//! k_pe = qkv_lora[:, q_lora+kv_lora :]        // [T, rope]，所有 head 共享
//! rope(q[..., nope:], k_pe)
//! latent = [kv_c | k_pe]  → paged KV cache（每 token entry = kv_lora + rope）
//! prefill : k = [kv_b(kv_c).nope | k_pe 广播], v = kv_b(kv_c).v ；attention(q, k, v)
//! decode  : q_lat = [q_nope · W_UK | q_pe]  (W_UK: 每 head [nope, kv_lora])；attention(q_lat, latent cache) → o_lat[H, kv_lora]
//!           o = o_lat · W_UV^T  (W_UV: 每 head [v, kv_lora])
//! out = o_proj(o.view(T, H*v))
//! ```
//! softmax scale = (nope + rope)^-0.5（无 yarn 时）。

use crate::indexer::{Indexer, IndexerKv, IndexerMeta};
use crate::nn::{Linear, RmsNorm, Rope};
use crate::tensor::Tensor;
use crate::weights::WeightSource;
use anyhow::{ensure, Result};
use dinfer_kernels::{self as k, Gemm, RopeStyle, Stream};
use half::bf16;

#[derive(Debug, Clone)]
pub struct MlaConfig {
    pub hidden: usize,
    pub heads: usize,
    pub q_lora: usize,
    pub kv_lora: usize,
    pub nope: usize,
    pub rope: usize,
    pub v_dim: usize,
    pub rope_style: RopeStyle,
    pub rope_theta: f64,
    pub max_pos: usize,
    pub rms_eps: f32,
}

impl MlaConfig {
    #[inline]
    pub fn qk_dim(&self) -> usize {
        self.nope + self.rope
    }
    #[inline]
    pub fn entry(&self) -> usize {
        self.kv_lora + self.rope
    }
    #[inline]
    pub fn scale(&self) -> f32 {
        (self.qk_dim() as f32).powf(-0.5)
    }
}

/// 一层的 paged latent KV cache：[num_blocks, block_size, entry] bf16。
pub struct PagedKv {
    pub cache: Tensor<bf16>,
    pub num_blocks: usize,
    pub block_size: usize,
    pub entry: usize,
}

impl PagedKv {
    pub fn new(stream: &Stream, num_blocks: usize, block_size: usize, entry: usize) -> Result<Self> {
        Ok(Self {
            cache: Tensor::zeros(stream, &[num_blocks, block_size, entry])?,
            num_blocks,
            block_size,
            entry,
        })
    }
}

/// 一步前向的 attention 元数据（由 runner 构造）。
pub enum AttnMeta<'a> {
    /// 所有请求都是从 0 开始的 prefill（本 step 内无历史 KV）：展开 K/V 的 causal attention。
    /// `block_table / seq_lens / max_blocks` 只有 DSA 索引器（读 paged 索引 cache）用；稠密模型可给空表 + max_blocks 0。
    Prefill {
        cu_seqlens: &'a Tensor<i32>,
        num_reqs: usize,
        block_table: &'a Tensor<i32>,
        seq_lens: &'a Tensor<i32>,
        max_blocks: usize,
    },
    /// 所有请求都是单 token decode：吸收路径读 paged cache。
    Decode { block_table: &'a Tensor<i32>, seq_lens: &'a Tensor<i32>, max_blocks: usize },
    /// 通用：每请求任意 query 长 + 历史在 cache 中（chunked prefill / prefill+decode 混合 batch）。
    /// 吸收路径（q 投影到 latent 空间，对 cache 做 causal attention）。`seq_lens[b]` = 历史 + 本步。
    Paged {
        cu_seqlens_q: &'a Tensor<i32>,
        block_table: &'a Tensor<i32>,
        seq_lens: &'a Tensor<i32>,
        num_reqs: usize,
        max_blocks: usize,
    },
}

impl<'a> AttnMeta<'a> {
    /// 索引器视角的 batch 描述（三种 meta 统一成 cu_seqlens_q + seq_lens + block_table）。
    /// Decode 没有 cu_seqlens_q（每请求 1 行）→ 调用方传 `decode_cu`（[0,1,…,B]）。
    pub fn indexer_meta(&self, decode_cu: &'a Tensor<i32>) -> IndexerMeta<'a> {
        match self {
            AttnMeta::Prefill { cu_seqlens, num_reqs, block_table, seq_lens, max_blocks } => IndexerMeta {
                cu_seqlens_q: cu_seqlens,
                seq_lens,
                block_table,
                num_reqs: *num_reqs,
                max_blocks: *max_blocks,
            },
            AttnMeta::Decode { block_table, seq_lens, max_blocks } => IndexerMeta {
                cu_seqlens_q: decode_cu,
                seq_lens,
                block_table,
                num_reqs: seq_lens.numel(),
                max_blocks: *max_blocks,
            },
            AttnMeta::Paged { cu_seqlens_q, block_table, seq_lens, num_reqs, max_blocks } => IndexerMeta {
                cu_seqlens_q,
                seq_lens,
                block_table,
                num_reqs: *num_reqs,
                max_blocks: *max_blocks,
            },
        }
    }
}

/// DSA 稀疏上下文：本层若带索引器则在 forward 内算 top-k 写入 `topk`；否则复用 `topk` 里上一个 full 层的结果。
pub struct SparseCtx<'a> {
    pub indexer: Option<(&'a mut Indexer, &'a mut IndexerKv)>,
    /// 共享 top-k 缓冲 [max_tokens, topk] i32（跨层传递）。
    pub topk: &'a mut Tensor<i32>,
    pub topk_k: usize,
    /// Decode 用的 cu_seqlens_q = [0, 1, …, B]。
    pub decode_cu: &'a Tensor<i32>,
    /// 对拍隔离：本层索引器算完（写了索引 K cache）后，用这些 host 侧索引（[t*topk] i32）覆盖共享缓冲。
    pub topk_override: Option<&'a [i32]>,
}

pub struct MlaAttention {
    pub cfg: MlaConfig,
    pub fused_qkv_a: Linear, // [q_lora + kv_lora + rope, hidden]
    pub q_a_norm: RmsNorm,
    pub q_b: Linear, // [H*(nope+rope), q_lora]
    pub kv_a_norm: RmsNorm,
    pub kv_b: Linear,   // [H*(nope+v), kv_lora]  —— prefill 用
    pub o_proj: Linear, // [hidden, H*v]
    /// 每 head 的 W_UK：GEMM 权重布局 [kv_lora, nope]（输入 nope，输出 kv_lora）
    w_uk_t: Vec<Tensor<bf16>>,
    /// 每 head 的 W_UV：GEMM 权重布局 [v, kv_lora]（输入 kv_lora，输出 v）
    w_uv: Vec<Tensor<bf16>>,
    pub rope: Rope,
    // scratch（按 max_tokens）
    qkv_lora: Tensor<bf16>,
    q_c_raw: Tensor<bf16>,
    q_c: Tensor<bf16>,
    q: Tensor<bf16>,
    kv_c_raw: Tensor<bf16>,
    kv_c: Tensor<bf16>,
    k_pe: Tensor<bf16>,
    latent: Tensor<bf16>,
    kv_full: Tensor<bf16>,
    k_full: Tensor<bf16>,
    v_full: Tensor<bf16>,
    attn_out: Tensor<bf16>,
    q_lat: Tensor<bf16>,
    o_lat: Tensor<bf16>,
    head_in: Tensor<bf16>,
    head_out: Tensor<bf16>,
    max_tokens: usize,
}

impl MlaAttention {
    /// `prefix` 形如 `model.layers.0.self_attn`。
    pub fn load(
        stream: &Stream,
        src: &WeightSource,
        prefix: &str,
        cfg: MlaConfig,
        max_tokens: usize,
    ) -> Result<Self> {
        let MlaConfig { hidden, heads, q_lora, kv_lora, nope, rope, v_dim, .. } = cfg;
        // fused_qkv_a = [q_a_proj ; kv_a_proj_with_mqa]
        let (qa_s, qa) = src.load_f32_host(&format!("{prefix}.q_a_proj.weight"))?;
        let (kva_s, kva) = src.load_f32_host(&format!("{prefix}.kv_a_proj_with_mqa.weight"))?;
        ensure!(qa_s == [q_lora, hidden], "q_a_proj shape {qa_s:?}");
        ensure!(kva_s == [kv_lora + rope, hidden], "kv_a_proj_with_mqa shape {kva_s:?}");
        let mut cat = qa;
        cat.extend_from_slice(&kva);
        let fused_qkv_a =
            Linear::new(Tensor::from_f32(stream, &cat, &[q_lora + kv_lora + rope, hidden])?, None)?;
        let q_a_norm = RmsNorm::load(stream, src, &format!("{prefix}.q_a_layernorm"), cfg.rms_eps)?;
        let q_b = Linear::load(stream, src, &format!("{prefix}.q_b_proj"))?;
        ensure!(q_b.out_features == heads * (nope + rope) && q_b.in_features == q_lora, "q_b_proj shape");
        let kv_a_norm = RmsNorm::load(stream, src, &format!("{prefix}.kv_a_layernorm"), cfg.rms_eps)?;
        let kv_b = Linear::load(stream, src, &format!("{prefix}.kv_b_proj"))?;
        ensure!(
            kv_b.out_features == heads * (nope + v_dim) && kv_b.in_features == kv_lora,
            "kv_b_proj shape"
        );
        let o_proj = Linear::load(stream, src, &format!("{prefix}.o_proj"))?;
        ensure!(o_proj.in_features == heads * v_dim && o_proj.out_features == hidden, "o_proj shape");

        // 拆 kv_b [H*(nope+v), kv_lora]（行主序，行 = 输出维）。
        // W_UK_h（[nope, kv_lora]，输出 nope ← 输入 kv_lora）；decode 需要反向：q_nope[nope] → q_lat[kv_lora]，
        // 即 GEMM 权重 [kv_lora, nope] = W_UK_h^T。
        // W_UV_h（[v, kv_lora]，输出 v ← 输入 kv_lora）；decode 用它正向：o_lat[kv_lora] → o[v]。
        let (_, kvb) = src.load_f32_host(&format!("{prefix}.kv_b_proj.weight"))?;
        let mut w_uk_t = Vec::with_capacity(heads);
        let mut w_uv = Vec::with_capacity(heads);
        for h in 0..heads {
            let base = h * (nope + v_dim);
            let uk = &kvb[base * kv_lora..(base + nope) * kv_lora]; // [nope, kv_lora]
            let mut uk_t = vec![0f32; kv_lora * nope];
            for i in 0..nope {
                for j in 0..kv_lora {
                    uk_t[j * nope + i] = uk[i * kv_lora + j];
                }
            }
            w_uk_t.push(Tensor::from_f32(stream, &uk_t, &[kv_lora, nope])?);
            let uv = &kvb[(base + nope) * kv_lora..(base + nope + v_dim) * kv_lora];
            w_uv.push(Tensor::from_f32(stream, uv, &[v_dim, kv_lora])?);
        }
        let rope_tab = Rope::new(stream, rope, cfg.max_pos, cfg.rope_theta, cfg.rope_style)?;
        let t = max_tokens;
        Ok(Self {
            fused_qkv_a,
            q_a_norm,
            q_b,
            kv_a_norm,
            kv_b,
            o_proj,
            w_uk_t,
            w_uv,
            rope: rope_tab,
            qkv_lora: Tensor::zeros(stream, &[t, q_lora + kv_lora + rope])?,
            q_c_raw: Tensor::zeros(stream, &[t, q_lora])?,
            q_c: Tensor::zeros(stream, &[t, q_lora])?,
            q: Tensor::zeros(stream, &[t, heads, nope + rope])?,
            kv_c_raw: Tensor::zeros(stream, &[t, kv_lora])?,
            kv_c: Tensor::zeros(stream, &[t, kv_lora])?,
            k_pe: Tensor::zeros(stream, &[t, rope])?,
            latent: Tensor::zeros(stream, &[t, kv_lora + rope])?,
            kv_full: Tensor::zeros(stream, &[t, heads, nope + v_dim])?,
            k_full: Tensor::zeros(stream, &[t, heads, nope + rope])?,
            v_full: Tensor::zeros(stream, &[t, heads, v_dim])?,
            attn_out: Tensor::zeros(stream, &[t, heads, v_dim])?,
            q_lat: Tensor::zeros(stream, &[t, heads, kv_lora + rope])?,
            o_lat: Tensor::zeros(stream, &[t, heads, kv_lora])?,
            head_in: Tensor::zeros(stream, &[t, kv_lora.max(nope)])?,
            head_out: Tensor::zeros(stream, &[t, kv_lora.max(v_dim)])?,
            cfg,
            max_tokens,
        })
    }

    /// 吸收路径 step 6：q_lat[t, h, entry] = [q_nope[h] · W_UK_h | q_pe[h]]（每 head 一次小 GEMM，P10 换 batched）。
    fn absorbed_q(&mut self, stream: &Stream, gemm: &Gemm, t: usize) -> Result<()> {
        let c = &self.cfg;
        let (h, nope, rope, kv_lora) = (c.heads, c.nope, c.rope, c.kv_lora);
        let (qk, entry) = (nope + rope, kv_lora + rope);
        for hh in 0..h {
            k::gather_head_bf16(stream, &self.q.data, &mut self.head_in.data, t, h, qk, hh, 0, nope)?;
            gemm.forward_bf16(
                &self.head_in.data,
                &self.w_uk_t[hh].data,
                None,
                &mut self.head_out.data,
                t,
                kv_lora,
                nope,
            )?;
            k::scatter_head_bf16(
                stream,
                &self.head_out.data,
                &mut self.q_lat.data,
                t,
                h,
                entry,
                hh,
                0,
                kv_lora,
            )?;
            k::gather_head_bf16(stream, &self.q.data, &mut self.head_in.data, t, h, qk, hh, nope, rope)?;
            k::scatter_head_bf16(
                stream,
                &self.head_in.data,
                &mut self.q_lat.data,
                t,
                h,
                entry,
                hh,
                kv_lora,
                rope,
            )?;
        }
        Ok(())
    }

    /// 吸收路径 step 7：attn_out[t, h, v] = o_lat[h] · W_UV_h^T。
    fn absorbed_o(&mut self, stream: &Stream, gemm: &Gemm, t: usize) -> Result<()> {
        let c = &self.cfg;
        let (h, kv_lora, v_dim) = (c.heads, c.kv_lora, c.v_dim);
        for hh in 0..h {
            k::gather_head_bf16(
                stream,
                &self.o_lat.data,
                &mut self.head_in.data,
                t,
                h,
                kv_lora,
                hh,
                0,
                kv_lora,
            )?;
            gemm.forward_bf16(
                &self.head_in.data,
                &self.w_uv[hh].data,
                None,
                &mut self.head_out.data,
                t,
                v_dim,
                kv_lora,
            )?;
            k::scatter_head_bf16(
                stream,
                &self.head_out.data,
                &mut self.attn_out.data,
                t,
                h,
                v_dim,
                hh,
                0,
                v_dim,
            )?;
        }
        Ok(())
    }

    /// x: [tokens, hidden] → out: [tokens, hidden]。`slot_mapping`: 每 token 写入 cache 的槽位（-1 跳过）。
    /// `t`：本步实际 token 数（x / out 缓冲可按 max_tokens 预分配，只用前 t 行）。
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        stream: &Stream,
        gemm: &Gemm,
        t: usize,
        x: &Tensor<bf16>,
        positions: &Tensor<i32>,
        slot_mapping: &Tensor<i64>,
        kv: &mut PagedKv,
        meta: &AttnMeta<'_>,
        mut sparse: Option<SparseCtx<'_>>,
        out: &mut Tensor<bf16>,
    ) -> Result<()> {
        let c = self.cfg.clone();
        ensure!(t <= self.max_tokens, "MLA: {t} tokens > max {}", self.max_tokens);
        ensure!(x.numel() >= t * c.hidden && out.numel() >= t * c.hidden, "MLA: x/out buffer too small");
        let (h, nope, rope, kv_lora, v_dim, q_lora) = (c.heads, c.nope, c.rope, c.kv_lora, c.v_dim, c.q_lora);
        let qk = nope + rope;
        let entry = kv_lora + rope;
        let wide = q_lora + entry;

        // 1. fused qkv_a: [t, q_lora + kv_lora + rope]
        gemm.forward_bf16(
            &x.data,
            &self.fused_qkv_a.w.data,
            None,
            &mut self.qkv_lora.data,
            t,
            wide,
            c.hidden,
        )?;
        // 2. 拆列 + norm
        k::copy_cols_bf16(
            stream,
            &self.qkv_lora.data,
            &mut self.q_c_raw.data,
            t,
            wide,
            0,
            q_lora,
            0,
            q_lora,
        )?;
        k::rmsnorm_bf16_n(
            stream,
            &self.q_c_raw.data,
            &self.q_a_norm.w.data,
            &mut self.q_c.data,
            t,
            q_lora,
            c.rms_eps,
        )?;
        k::copy_cols_bf16(
            stream,
            &self.qkv_lora.data,
            &mut self.kv_c_raw.data,
            t,
            wide,
            q_lora,
            kv_lora,
            0,
            kv_lora,
        )?;
        k::rmsnorm_bf16_n(
            stream,
            &self.kv_c_raw.data,
            &self.kv_a_norm.w.data,
            &mut self.kv_c.data,
            t,
            kv_lora,
            c.rms_eps,
        )?;
        k::copy_cols_bf16(
            stream,
            &self.qkv_lora.data,
            &mut self.k_pe.data,
            t,
            wide,
            q_lora + kv_lora,
            rope,
            0,
            rope,
        )?;
        // 2b. DSA 索引器：用 q_c 与本层输入 x 算 top-k（写索引 K cache）；shared 层跳过，直接用缓冲里的
        if let Some(sp) = sparse.as_mut() {
            if let Some((idx, ikv)) = sp.indexer.as_mut() {
                let im = meta.indexer_meta(sp.decode_cu);
                idx.forward(stream, gemm, t, x, &self.q_c, positions, slot_mapping, ikv, &im, sp.topk)?;
                if let Some(ov) = sp.topk_override {
                    ensure!(ov.len() == t * sp.topk_k, "topk_override: {} != t*k", ov.len());
                    let tmp = Tensor::from_vec(stream, ov.to_vec(), &[t, sp.topk_k])?;
                    k::copy_i32_n(stream, &tmp.data, &mut sp.topk.data, t * sp.topk_k)?;
                }
            }
        }
        let topk = sparse.as_ref().map(|sp| (&sp.topk.data, sp.topk_k));
        // 3. q = q_b(q_c): [t, H, qk]
        gemm.forward_bf16(&self.q_c.data, &self.q_b.w.data, None, &mut self.q.data, t, h * qk, q_lora)?;
        // 4. rope：q 的每 head 的 [nope:qk]；k_pe 作为 1 个 head、head_dim=rope 的张量整体旋转
        //    rope kernel 只旋转前 rope_dim 维，所以对 q 要先把 rope 段挪到独立缓冲 —— 用 k_full 作临时：
        //    更简单：kernel 支持 head_dim > rope_dim 时只旋前 rope_dim 维；这里 q 的 rope 段在尾部。
        //    解决：把 q 的 [nope:qk] 列搬到 q_lat 的尾部临时区做旋转再搬回（见 rope_q_tail）。
        self.rope_q_tail(stream, positions, t)?;
        k::rope_bf16_n(
            stream,
            &mut self.k_pe.data,
            None,
            &positions.data,
            &self.rope.cos_sin.data,
            t,
            1,
            0,
            rope,
            rope,
            c.rope_style,
        )?;
        // 5. latent = [kv_c | k_pe] → cache
        k::copy_cols_bf16(stream, &self.kv_c.data, &mut self.latent.data, t, kv_lora, 0, entry, 0, kv_lora)?;
        k::copy_cols_bf16(stream, &self.k_pe.data, &mut self.latent.data, t, rope, 0, entry, kv_lora, rope)?;
        k::mla_cache_write_bf16(
            stream,
            &self.latent.data,
            &mut kv.cache.data,
            &slot_mapping.data,
            t,
            entry,
            kv.block_size,
        )?;

        match meta {
            AttnMeta::Prefill { cu_seqlens, num_reqs, .. } => {
                // 6a. kv_full = kv_b(kv_c): [t, H, nope+v] → k_full = [nope | k_pe 广播], v_full
                gemm.forward_bf16(
                    &self.kv_c.data,
                    &self.kv_b.w.data,
                    None,
                    &mut self.kv_full.data,
                    t,
                    h * (nope + v_dim),
                    kv_lora,
                )?;
                k::copy_head_cols_bf16(
                    stream,
                    &self.kv_full.data,
                    &mut self.k_full.data,
                    t,
                    h,
                    nope + v_dim,
                    0,
                    qk,
                    0,
                    nope,
                )?;
                k::bcast_kpe_bf16(stream, &self.k_pe.data, &mut self.k_full.data, t, h, qk, nope, rope)?;
                k::copy_head_cols_bf16(
                    stream,
                    &self.kv_full.data,
                    &mut self.v_full.data,
                    t,
                    h,
                    nope + v_dim,
                    nope,
                    v_dim,
                    0,
                    v_dim,
                )?;
                k::mla_prefill_bf16(
                    stream,
                    &self.q.data,
                    &self.k_full.data,
                    &self.v_full.data,
                    &mut self.attn_out.data,
                    &cu_seqlens.data,
                    topk,
                    *num_reqs,
                    t,
                    h,
                    qk,
                    v_dim,
                    c.scale(),
                )?;
            }
            AttnMeta::Decode { block_table, seq_lens, max_blocks } => {
                self.absorbed_q(stream, gemm, t)?;
                k::mla_decode_bf16(
                    stream,
                    &self.q_lat.data,
                    &kv.cache.data,
                    &block_table.data,
                    &seq_lens.data,
                    topk,
                    &mut self.o_lat.data,
                    t,
                    h,
                    entry,
                    kv_lora,
                    kv.block_size,
                    *max_blocks,
                    c.scale(),
                )?;
                self.absorbed_o(stream, gemm, t)?;
            }
            AttnMeta::Paged { cu_seqlens_q, block_table, seq_lens, num_reqs, max_blocks } => {
                self.absorbed_q(stream, gemm, t)?;
                k::mla_prefill_paged_bf16(
                    stream,
                    &self.q_lat.data,
                    &kv.cache.data,
                    &block_table.data,
                    &seq_lens.data,
                    &cu_seqlens_q.data,
                    topk,
                    &mut self.o_lat.data,
                    *num_reqs,
                    t,
                    h,
                    entry,
                    kv_lora,
                    kv.block_size,
                    *max_blocks,
                    c.scale(),
                )?;
                self.absorbed_o(stream, gemm, t)?;
            }
        }
        // 8. o_proj
        gemm.forward_bf16(
            &self.attn_out.data,
            &self.o_proj.w.data,
            None,
            &mut out.data,
            t,
            c.hidden,
            h * v_dim,
        )
    }

    /// 对 q[t, H, qk] 每 head 的尾部 rope 段做旋转：搬到连续缓冲 → rope → 搬回。
    fn rope_q_tail(&mut self, stream: &Stream, positions: &Tensor<i32>, t: usize) -> Result<()> {
        let (h, nope, rope, qk) = (self.cfg.heads, self.cfg.nope, self.cfg.rope, self.cfg.qk_dim());
        // 复用 k_full 的前 t*h*rope 元素作为 [t, h, rope] 临时区
        k::copy_head_cols_bf16(stream, &self.q.data, &mut self.k_full.data, t, h, qk, nope, rope, 0, rope)?;
        k::rope_bf16_n(
            stream,
            &mut self.k_full.data,
            None,
            &positions.data,
            &self.rope.cos_sin.data,
            t,
            h,
            0,
            rope,
            rope,
            self.cfg.rope_style,
        )?;
        k::copy_head_cols_bf16(stream, &self.k_full.data, &mut self.q.data, t, h, rope, 0, qk, nope, rope)
    }
}
