//! Kimi-K3 MLA 全注意力层（24 / 93 层；HF `modeling_kimi_linear.py::KimiMLAAttention`）。
//!
//! 与 GLM-5.3 / DeepSeek 的 MLA（`crates/layers/src/mla.rs`）的差别（以 HF 代码为唯一口径）：
//! - **无 RoPE**：`mla_use_nope=True`，`rotary_emb = None`。`qk_rope_head_dim = 64` 的列**存在**（q_b 的每 head 尾 64 维、
//!   `kv_a_proj_with_mqa` 的尾 64 维 = `k_rot`，所有 head 共享），但**不旋转**，直接拼进 q / k 参与点积。
//!   所以 KV cache 每 token 仍是 `kv_lora + rope = 576` 维 latent，吸收路径与 GLM 完全一样，只是少了旋转。
//! - **输出门**：`mla_use_output_gate=True` → `attn_out *= sigmoid(g_proj(x))`，`g_proj: [H·v, hidden]`，在 o_proj 之前。
//! - 投影融合：vLLM 把 `q_a_proj / kv_a_proj_with_mqa / g_proj` 合成 `fused_qkv_a_g_proj`；HF 分片里三者分开存，
//!   我们装载时按行拼成一条 `[q_lora + kv_lora + rope + H·v, hidden]` GEMM（顺序同 vLLM），输出按列拆。
//! - scale = `(nope + rope)^-0.5 = 192^-0.5`（HF `self.scaling = q_head_dim ** -0.5`）。
//! - flash_attention_2 路径里 v 从 128 pad 到 192 再截回，数值上等价于 `dv = 128` 直接算。
//!
//! 数据流：
//! ```text
//! [q_c_raw | kv_c_raw | k_pe | g] = fused(x)
//! q   = q_b(rmsnorm(q_c_raw)).view(T, H, nope + rope)              // 不旋转
//! kv_c = rmsnorm(kv_c_raw);  latent = [kv_c | k_pe] → paged cache（entry = 576）
//! prefill : k = [kv_b(kv_c).nope | k_pe 广播], v = kv_b(kv_c).v ; attention(q, k, v)
//! decode  : q_lat = [q_nope · W_UK | q_pe] ; attention(q_lat, cache) → o_lat · W_UV^T
//! out = o_proj(attn * sigmoid(g))
//! ```
//! 性能项（后补）：吸收路径每 head 一次小 GEMM（96 × 2 次），GLM 线同样待 batched GEMM。
use crate::config::KimiLinearConfig;
use crate::weights::Table;
use anyhow::{ensure, Result};
use dinfer_core::runner::BatchMode;
use dinfer_kernels::ops as k;
use dinfer_layers::{bf16, Gemm, Linear, RmsNorm, Stream, Tensor, WeightSource};

/// HF `KimiRMSNorm` 的默认 eps：`q_a_layernorm` / `kv_a_layernorm` 构造时没传 `rms_norm_eps`。
const A_NORM_EPS: f32 = 1e-6;

/// 一步 attention 的 batch 描述（设备张量由调用方上传）。
/// - `Prefill`：所有请求从 0 开始，用 `cu_seqlens`（展开 K/V 的 causal attention）；
/// - `Decode`：每请求 1 token，用 `block_table / seq_lens / max_blocks`（`seq_lens` 含本步）；
/// - `Paged`：每请求任意 query 长 + 历史在 cache，用 `cu_seqlens`（query 侧）+ `block_table / seq_lens`。
pub struct MlaMeta<'a> {
    pub mode: BatchMode,
    pub cu_seqlens: &'a Tensor<i32>,
    pub block_table: &'a Tensor<i32>,
    pub seq_lens: &'a Tensor<i32>,
    pub num_reqs: usize,
    pub max_blocks: usize,
}

/// 一层 paged latent KV cache：`[num_blocks, block_size, entry]` bf16（entry = kv_lora + rope = 576）。
pub struct PagedKv {
    pub cache: Tensor<bf16>,
    pub num_blocks: usize,
    pub block_size: usize,
    pub entry: usize,
}

pub struct MlaLayer {
    fused: Linear, // [q_lora + kv_lora + rope + H·v, hidden]：[q_a | kv_a_with_mqa | g]
    q_a_norm: RmsNorm,
    q_b: Linear, // [H·(nope+rope), q_lora]
    kv_a_norm: RmsNorm,
    kv_b: Linear, // [H·(nope+v), kv_lora]（prefill 用）
    o_proj: Linear, // [hidden, H·v]
    /// 每 head W_UK^T：GEMM 权重 [kv_lora, nope]（decode：q_nope → q_lat）
    w_uk_t: Vec<Tensor<bf16>>,
    /// 每 head W_UV：GEMM 权重 [v, kv_lora]（decode：o_lat → o）
    w_uv: Vec<Tensor<bf16>>,
    pub kv: PagedKv,
    heads: usize,
    hidden: usize,
    q_lora: usize,
    kv_lora: usize,
    nope: usize,
    rope: usize,
    v_dim: usize,
    scale: f32,
    max_tokens: usize,
    // scratch（按 max_tokens）
    fused_out: Tensor<bf16>,
    q_c_raw: Tensor<bf16>,
    q_c: Tensor<bf16>,
    q: Tensor<bf16>,
    kv_c_raw: Tensor<bf16>,
    kv_c: Tensor<bf16>,
    k_pe: Tensor<bf16>,
    gate: Tensor<bf16>,
    latent: Tensor<bf16>,
    kv_full: Tensor<bf16>,
    k_full: Tensor<bf16>,
    v_full: Tensor<bf16>,
    attn_out: Tensor<bf16>,
    q_lat: Tensor<bf16>,
    o_lat: Tensor<bf16>,
    head_in: Tensor<bf16>,
    head_out: Tensor<bf16>,
}

impl MlaLayer {
    /// 装载第 `layer` 层（必须是 MLA 层）。`num_blocks × block_size` 决定本层 KV cache 容量。
    #[allow(clippy::too_many_arguments)]
    pub fn load(stream: &Stream, src: &WeightSource, t: &Table, layer: usize, cfg: &KimiLinearConfig, max_tokens: usize, num_blocks: usize, block_size: usize) -> Result<Self> {
        ensure!(!cfg.is_kda_layer(layer as u32), "第 {layer} 层是 KDA 层，不是 MLA");
        ensure!(cfg.mla_use_nope, "只支持 mla_use_nope=True（K3）");
        ensure!(cfg.mla_use_output_gate, "只支持 mla_use_output_gate=True（K3）");
        let hidden = cfg.hidden_size as usize;
        let heads = cfg.num_attention_heads as usize;
        let q_lora = cfg.q_lora_rank.ok_or_else(|| anyhow::anyhow!("q_lora_rank 缺失"))? as usize;
        let kv_lora = cfg.kv_lora_rank as usize;
        let (nope, rope, v_dim) = (cfg.qk_nope_head_dim as usize, cfg.qk_rope_head_dim as usize, cfg.v_head_dim as usize);
        let p = format!("layers.{layer}.self_attn");

        // fused = [q_a_proj ; kv_a_proj_with_mqa ; g_proj]（行拼接；文件是 bf16，经 f32 往返无损）
        let (qa_s, qa) = t.load_f32_host(src, &format!("{p}.q_a_proj.weight"))?;
        let (kva_s, kva) = t.load_f32_host(src, &format!("{p}.kv_a_proj_with_mqa.weight"))?;
        let (g_s, g) = t.load_f32_host(src, &format!("{p}.g_proj.weight"))?;
        ensure!(qa_s == [q_lora, hidden], "q_a_proj shape {qa_s:?}");
        ensure!(kva_s == [kv_lora + rope, hidden], "kv_a_proj_with_mqa shape {kva_s:?}");
        ensure!(g_s == [heads * v_dim, hidden], "g_proj shape {g_s:?}");
        let mut cat = qa;
        cat.extend_from_slice(&kva);
        cat.extend_from_slice(&g);
        let wide = q_lora + kv_lora + rope + heads * v_dim;
        let fused = Linear::new(Tensor::from_f32(stream, &cat, &[wide, hidden])?, None)?;
        drop(cat);

        // HF `KimiMLAAttention` 里 `q_a_layernorm = KimiRMSNorm(q_lora_rank)` / `kv_a_layernorm = KimiRMSNorm(kv_lora_rank)` **没传 eps**，
        // 用 KimiRMSNorm 默认 1e-6（vLLM 用 rms_norm_eps=1e-5；以 HF 为口径，差异 ~1e-6 相对，远低于 bf16）。
        let q_a_norm = t.rmsnorm(stream, src, &format!("{p}.q_a_layernorm.weight"), A_NORM_EPS)?;
        ensure!(q_a_norm.hidden == q_lora, "q_a_layernorm {}", q_a_norm.hidden);
        let kv_a_norm = t.rmsnorm(stream, src, &format!("{p}.kv_a_layernorm.weight"), A_NORM_EPS)?;
        ensure!(kv_a_norm.hidden == kv_lora, "kv_a_layernorm {}", kv_a_norm.hidden);
        let q_b = t.linear(stream, src, &format!("{p}.q_b_proj.weight"))?;
        ensure!(q_b.w.shape == [heads * (nope + rope), q_lora], "q_b_proj shape {:?}", q_b.w.shape);
        let kv_b = t.linear(stream, src, &format!("{p}.kv_b_proj.weight"))?;
        ensure!(kv_b.w.shape == [heads * (nope + v_dim), kv_lora], "kv_b_proj shape {:?}", kv_b.w.shape);
        let o_proj = t.linear(stream, src, &format!("{p}.o_proj.weight"))?;
        ensure!(o_proj.w.shape == [hidden, heads * v_dim], "o_proj shape {:?}", o_proj.w.shape);

        // 拆 kv_b [H·(nope+v), kv_lora]（行 = 输出维）：W_UK_h [nope, kv_lora] 需转置成 GEMM 权重 [kv_lora, nope]；W_UV_h [v, kv_lora] 直接用。
        let (_, kvb) = t.load_f32_host(src, &format!("{p}.kv_b_proj.weight"))?;
        let mut w_uk_t = Vec::with_capacity(heads);
        let mut w_uv = Vec::with_capacity(heads);
        for h in 0..heads {
            let base = h * (nope + v_dim);
            let uk = &kvb[base * kv_lora..(base + nope) * kv_lora];
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
        let entry = kv_lora + rope;
        let tm = max_tokens;
        Ok(Self {
            fused,
            q_a_norm,
            q_b,
            kv_a_norm,
            kv_b,
            o_proj,
            w_uk_t,
            w_uv,
            kv: PagedKv { cache: Tensor::zeros(stream, &[num_blocks, block_size, entry])?, num_blocks, block_size, entry },
            heads,
            hidden,
            q_lora,
            kv_lora,
            nope,
            rope,
            v_dim,
            scale: ((nope + rope) as f32).powf(-0.5),
            max_tokens,
            fused_out: Tensor::zeros(stream, &[tm, wide])?,
            q_c_raw: Tensor::zeros(stream, &[tm, q_lora])?,
            q_c: Tensor::zeros(stream, &[tm, q_lora])?,
            q: Tensor::zeros(stream, &[tm, heads, nope + rope])?,
            kv_c_raw: Tensor::zeros(stream, &[tm, kv_lora])?,
            kv_c: Tensor::zeros(stream, &[tm, kv_lora])?,
            k_pe: Tensor::zeros(stream, &[tm, rope])?,
            gate: Tensor::zeros(stream, &[tm, heads * v_dim])?,
            latent: Tensor::zeros(stream, &[tm, entry])?,
            kv_full: Tensor::zeros(stream, &[tm, heads, nope + v_dim])?,
            k_full: Tensor::zeros(stream, &[tm, heads, nope + rope])?,
            v_full: Tensor::zeros(stream, &[tm, heads, v_dim])?,
            attn_out: Tensor::zeros(stream, &[tm, heads, v_dim])?,
            q_lat: Tensor::zeros(stream, &[tm, heads, entry])?,
            o_lat: Tensor::zeros(stream, &[tm, heads, kv_lora])?,
            head_in: Tensor::zeros(stream, &[tm, kv_lora.max(nope)])?,
            head_out: Tensor::zeros(stream, &[tm, kv_lora.max(v_dim)])?,
        })
    }

    pub fn entry(&self) -> usize {
        self.kv.entry
    }

    /// `x[:t]`（input_layernorm 之后）→ `out[:t]`。`slot_mapping[i]`：第 i 个 token 的 latent 写入 cache 的槽位（-1 跳过）。
    pub fn forward(&mut self, stream: &Stream, gemm: &Gemm, t: usize, x: &Tensor<bf16>, slot_mapping: &Tensor<i64>, meta: &MlaMeta<'_>, out: &mut Tensor<bf16>) -> Result<()> {
        let (h, nope, rope, kv_lora, v_dim, q_lora, hidden) = (self.heads, self.nope, self.rope, self.kv_lora, self.v_dim, self.q_lora, self.hidden);
        let qk = nope + rope;
        let entry = kv_lora + rope;
        let wide = q_lora + entry + h * v_dim;
        ensure!(t <= self.max_tokens, "mla: {t} tokens > max {}", self.max_tokens);
        ensure!(x.numel() >= t * hidden && out.numel() >= t * hidden, "mla: x/out 不足 {t} 行");
        ensure!(slot_mapping.numel() >= t, "mla: slot_mapping");

        // 1. fused：[q_c_raw | kv_c_raw | k_pe | g]
        self.fused.forward_rows(gemm, x, &mut self.fused_out, t)?;
        k::copy_cols_bf16(stream, &self.fused_out.data, &mut self.q_c_raw.data, t, wide, 0, q_lora, 0, q_lora)?;
        k::copy_cols_bf16(stream, &self.fused_out.data, &mut self.kv_c_raw.data, t, wide, q_lora, kv_lora, 0, kv_lora)?;
        k::copy_cols_bf16(stream, &self.fused_out.data, &mut self.k_pe.data, t, wide, q_lora + kv_lora, rope, 0, rope)?;
        k::copy_cols_bf16(stream, &self.fused_out.data, &mut self.gate.data, t, wide, q_lora + entry, h * v_dim, 0, h * v_dim)?;
        // 2. 两个 a_layernorm
        k::rmsnorm_bf16_n(stream, &self.q_c_raw.data, &self.q_a_norm.w.data, &mut self.q_c.data, t, q_lora, self.q_a_norm.eps)?;
        k::rmsnorm_bf16_n(stream, &self.kv_c_raw.data, &self.kv_a_norm.w.data, &mut self.kv_c.data, t, kv_lora, self.kv_a_norm.eps)?;
        // 3. q = q_b(q_c)：[t, H, nope + rope]，不旋转
        self.q_b.forward_rows(gemm, &self.q_c, &mut self.q, t)?;
        // 4. latent = [kv_c | k_pe] → paged cache
        k::copy_cols_bf16(stream, &self.kv_c.data, &mut self.latent.data, t, kv_lora, 0, entry, 0, kv_lora)?;
        k::copy_cols_bf16(stream, &self.k_pe.data, &mut self.latent.data, t, rope, 0, entry, kv_lora, rope)?;
        k::mla_cache_write_bf16(stream, &self.latent.data, &mut self.kv.cache.data, &slot_mapping.data, t, entry, self.kv.block_size)?;
        // 5. attention
        match meta.mode {
            BatchMode::Prefill => {
                self.kv_b.forward_rows(gemm, &self.kv_c, &mut self.kv_full, t)?;
                k::copy_head_cols_bf16(stream, &self.kv_full.data, &mut self.k_full.data, t, h, nope + v_dim, 0, qk, 0, nope)?;
                k::bcast_kpe_bf16(stream, &self.k_pe.data, &mut self.k_full.data, t, h, qk, nope, rope)?;
                k::copy_head_cols_bf16(stream, &self.kv_full.data, &mut self.v_full.data, t, h, nope + v_dim, nope, v_dim, 0, v_dim)?;
                k::mla_prefill_bf16(stream, &self.q.data, &self.k_full.data, &self.v_full.data, &mut self.attn_out.data, &meta.cu_seqlens.data, None, meta.num_reqs, t, h, qk, v_dim, self.scale)?;
            }
            BatchMode::Decode => {
                self.absorbed_q(stream, gemm, t)?;
                k::mla_decode_bf16(
                    stream,
                    &self.q_lat.data,
                    &self.kv.cache.data,
                    &meta.block_table.data,
                    &meta.seq_lens.data,
                    None,
                    &mut self.o_lat.data,
                    t,
                    h,
                    entry,
                    kv_lora,
                    self.kv.block_size,
                    meta.max_blocks,
                    self.scale,
                )?;
                self.absorbed_o(stream, gemm, t)?;
            }
            BatchMode::Paged => {
                self.absorbed_q(stream, gemm, t)?;
                k::mla_prefill_paged_bf16(
                    stream,
                    &self.q_lat.data,
                    &self.kv.cache.data,
                    &meta.block_table.data,
                    &meta.seq_lens.data,
                    &meta.cu_seqlens.data,
                    None,
                    &mut self.o_lat.data,
                    meta.num_reqs,
                    t,
                    h,
                    entry,
                    kv_lora,
                    self.kv.block_size,
                    meta.max_blocks,
                    self.scale,
                )?;
                self.absorbed_o(stream, gemm, t)?;
            }
        }
        // 6. 输出门 + o_proj
        k::mul_sigmoid_bf16(stream, &mut self.attn_out.data, &self.gate.data, t * h * v_dim)?;
        self.o_proj.forward_rows(gemm, &self.attn_out, out, t)
    }

    /// 吸收路径：q_lat[t, H, entry] = [q_nope[h] · W_UK_h | q_pe[h]]（每 head 一次小 GEMM）。
    fn absorbed_q(&mut self, stream: &Stream, gemm: &Gemm, t: usize) -> Result<()> {
        let (h, nope, rope, kv_lora) = (self.heads, self.nope, self.rope, self.kv_lora);
        let (qk, entry) = (nope + rope, kv_lora + rope);
        for hh in 0..h {
            k::gather_head_bf16(stream, &self.q.data, &mut self.head_in.data, t, h, qk, hh, 0, nope)?;
            gemm.forward_bf16(&self.head_in.data, &self.w_uk_t[hh].data, None, &mut self.head_out.data, t, kv_lora, nope)?;
            k::scatter_head_bf16(stream, &self.head_out.data, &mut self.q_lat.data, t, h, entry, hh, 0, kv_lora)?;
            k::gather_head_bf16(stream, &self.q.data, &mut self.head_in.data, t, h, qk, hh, nope, rope)?;
            k::scatter_head_bf16(stream, &self.head_in.data, &mut self.q_lat.data, t, h, entry, hh, kv_lora, rope)?;
        }
        Ok(())
    }

    /// 吸收路径：attn_out[t, H, v] = o_lat[h] · W_UV_h^T。
    fn absorbed_o(&mut self, stream: &Stream, gemm: &Gemm, t: usize) -> Result<()> {
        let (h, kv_lora, v_dim) = (self.heads, self.kv_lora, self.v_dim);
        for hh in 0..h {
            k::gather_head_bf16(stream, &self.o_lat.data, &mut self.head_in.data, t, h, kv_lora, hh, 0, kv_lora)?;
            gemm.forward_bf16(&self.head_in.data, &self.w_uv[hh].data, None, &mut self.head_out.data, t, v_dim, kv_lora)?;
            k::scatter_head_bf16(stream, &self.head_out.data, &mut self.attn_out.data, t, h, v_dim, hh, 0, v_dim)?;
        }
        Ok(())
    }

    // ---- 对拍 / 调试用中间量（上一步，前 t 行有效）----
    /// o_proj 之前、门之后的 [t, H·v]。
    pub fn last_attn_pre(&self) -> &Tensor<bf16> {
        &self.attn_out
    }
    pub fn last_q(&self) -> &Tensor<bf16> {
        &self.q
    }
    pub fn last_latent(&self) -> &Tensor<bf16> {
        &self.latent
    }
}
