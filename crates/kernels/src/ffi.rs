//! `cuda/kernels.h` 的 Rust 声明。手写而非 bindgen，保持签名可审计。
//! 每个入口返回 cudaError_t 数值（0 = success）。符号名不带项目前缀（项目可能改名）。

#![allow(non_camel_case_types)]

use std::ffi::c_void;

pub type cudaStream_t = *mut c_void;

#[cfg(not(no_nvcc_stub))]
extern "C" {
    pub fn add_f32(a: *const f32, b: *const f32, out: *mut f32, n: i64, stream: cudaStream_t) -> i32;

    pub fn rmsnorm_bf16(
        x: *const c_void,
        w: *const c_void,
        out: *mut c_void,
        tokens: i64,
        hidden: i64,
        eps: f32,
        layout: i32,
        mode: i32,
        stream: cudaStream_t,
    ) -> i32;
    pub fn gather_rmsnorm_bf16(
        x: *const c_void,
        rows: *const i32,
        w: *const c_void,
        out: *mut c_void,
        num_rows: i64,
        hidden: i64,
        eps: f32,
        layout: i32,
        mode: i32,
        stream: cudaStream_t,
    ) -> i32;
    pub fn fused_add_rmsnorm_bf16(
        x: *const c_void,
        res: *mut c_void,
        w: *const c_void,
        out: *mut c_void,
        tokens: i64,
        hidden: i64,
        eps: f32,
        layout: i32,
        mode: i32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn rmsnorm_scale2_bf16(
        x: *const c_void,
        scale: *const c_void,
        out: *mut c_void,
        tokens: i64,
        hidden: i64,
        eps: f32,
        rs: f32,
        mode: i32,
        stream: cudaStream_t,
    ) -> i32;
    pub fn rmsnorm_quant_fp8_bf16(
        x: *const c_void,
        w: *const c_void,
        out: *mut c_void,
        tokens: i64,
        hidden: i64,
        eps: f32,
        a_scale: f32,
        stream: cudaStream_t,
    ) -> i32;
    pub fn rope_bf16(
        q: *mut c_void,
        k: *mut c_void,
        positions: *const i32,
        cos_sin: *const f32,
        tokens: i64,
        q_heads: i64,
        k_heads: i64,
        head_dim: i64,
        rope_dim: i64,
        neox: i32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn rope_span_bf16(
        q: *mut c_void,
        k: *mut c_void,
        positions: *const i32,
        cos_sin: *const f32,
        tokens: i64,
        q_heads: i64,
        k_heads: i64,
        head_dim: i64,
        rope_dim: i64,
        span: i64,
        neox: i32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn silu_and_mul_bf16(
        x: *const c_void,
        out: *mut c_void,
        tokens: i64,
        inter: i64,
        stream: cudaStream_t,
    ) -> i32;

    pub fn gelu_tanh_mul_bf16(
        gate: *const c_void,
        up: *const c_void,
        out: *mut c_void,
        tokens: i64,
        inter: i64,
        mode: i32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn mul_row_bf16(x: *mut c_void, w: *const c_void, rows: i64, cols: i64, stream: cudaStream_t) -> i32;

    pub fn softcap_bf16(x: *mut c_void, n: i64, c: f32, stream: cudaStream_t) -> i32;

    pub fn embedding_bf16(
        ids: *const i32,
        table: *const c_void,
        out: *mut c_void,
        tokens: i64,
        hidden: i64,
        vocab: i64,
        stream: cudaStream_t,
    ) -> i32;

    pub fn penalties_f32(
        logits: *mut f32,
        ids: *const i32,
        offsets: *const i32,
        prompt_len: *const i32,
        rep: *const f32,
        freq: *const f32,
        pres: *const f32,
        batch: i64,
        vocab: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn min_p_mask_f32(
        logits: *mut f32,
        min_p: *const f32,
        batch: i64,
        vocab: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn temperature_f32(
        logits: *mut f32,
        temperature: *const f32,
        batch: i64,
        vocab: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn logprobs_f32(
        logits: *const f32,
        chosen: *const i32,
        k: i64,
        out_ids: *mut i32,
        out_lp: *mut f32,
        rank: *mut i32,
        batch: i64,
        vocab: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn argmax_f32(logits: *const f32, out: *mut i32, batch: i64, vocab: i64, stream: cudaStream_t)
        -> i32;
    pub fn row_gather_i32(
        prev: *const i32,
        gather_idx: *const i32,
        ids: *mut i32,
        n: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn topk_topp_mask_f32(
        logits: *mut f32,
        top_k: *const i32,
        top_p: *const f32,
        batch: i64,
        vocab: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn sample_f32(
        logits: *const f32,
        temperature: *const f32,
        seeds: *const u64,
        offsets: *const u64,
        out: *mut i32,
        batch: i64,
        vocab: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn logits_mask_f32(
        logits: *mut f32,
        mask_row: *const i32,
        mask_bits: *const u32,
        mask_words: i64,
        bias_offsets: *const i32,
        bias_ids: *const i32,
        bias_vals: *const f32,
        batch: i64,
        vocab: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn sample_fused_f32(
        work: *mut f32,
        temperature: *const f32,
        min_p: *const f32,
        top_k: *const i32,
        top_p: *const f32,
        temp01: *const f32,
        seeds: *const u64,
        offsets: *const u64,
        out: *mut i32,
        batch: i64,
        vocab: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn mla_cache_write_bf16(
        latent: *const c_void,
        cache: *mut c_void,
        slot_mapping: *const i64,
        tokens: i64,
        entry: i64,
        block_size: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn mla_prefill_bf16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        out: *mut c_void,
        cu_seqlens: *const i32,
        topk_indices: *const i32,
        topk: i64,
        num_reqs: i64,
        heads: i64,
        dq: i64,
        dv: i64,
        scale: f32,
        stream: cudaStream_t,
    ) -> i32;
    pub fn mla_prefill_paged_bf16(
        q: *const c_void,
        cache: *const c_void,
        block_table: *const i32,
        seq_lens: *const i32,
        cu_seqlens_q: *const i32,
        topk_indices: *const i32,
        topk: i64,
        out: *mut c_void,
        num_reqs: i64,
        heads: i64,
        entry: i64,
        kv_lora: i64,
        block_size: i64,
        max_blocks: i64,
        scale: f32,
        stream: cudaStream_t,
    ) -> i32;
    pub fn gqa_cache_write_bf16(
        k: *const c_void,
        v: *const c_void,
        cache: *mut c_void,
        slot_mapping: *const i64,
        tokens: i64,
        kv_heads: i64,
        head_dim: i64,
        block_size: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn gqa_prefill_bf16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        out: *mut c_void,
        cu_seqlens: *const i32,
        num_reqs: i64,
        max_seqlen: i64,
        heads: i64,
        kv_heads: i64,
        head_dim: i64,
        scale: f32,
        q_stride: i64,
        kv_stride: i64,
        cfg: i32,
        stream: cudaStream_t,
    ) -> i32;
    /// 滑窗变体（Kaon-V3 / Gemma4 滑窗层）：`window <= 0` = 全注意力。
    pub fn gqa_prefill_win_bf16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        out: *mut c_void,
        cu_seqlens: *const i32,
        num_reqs: i64,
        max_seqlen: i64,
        heads: i64,
        kv_heads: i64,
        head_dim: i64,
        scale: f32,
        q_stride: i64,
        kv_stride: i64,
        cfg: i32,
        window: i32,
        p_fp8: i32,
        stream: cudaStream_t,
    ) -> i32;
    pub fn gqa_paged_bf16(
        q: *const c_void,
        cache: *const c_void,
        block_table: *const i32,
        seq_lens: *const i32,
        cu_seqlens_q: *const i32,
        out: *mut c_void,
        num_reqs: i64,
        max_q_len: i64,
        heads: i64,
        kv_heads: i64,
        head_dim: i64,
        block_size: i64,
        max_blocks: i64,
        scale: f32,
        q_stride: i64,
        skip_single: i32,
        cfg: i32,
        stream: cudaStream_t,
    ) -> i32;
    pub fn gqa_paged_win_bf16(
        q: *const c_void,
        cache: *const c_void,
        block_table: *const i32,
        seq_lens: *const i32,
        cu_seqlens_q: *const i32,
        out: *mut c_void,
        num_reqs: i64,
        max_q_len: i64,
        heads: i64,
        kv_heads: i64,
        head_dim: i64,
        block_size: i64,
        max_blocks: i64,
        scale: f32,
        q_stride: i64,
        skip_single: i32,
        cfg: i32,
        window: i32,
        p_fp8: i32,
        stream: cudaStream_t,
    ) -> i32;
    pub fn gqa_decode_bf16(
        q: *const c_void,
        cache: *const c_void,
        block_table: *const i32,
        seq_lens: *const i32,
        out: *mut c_void,
        scratch: *mut f32,
        num_reqs: i64,
        heads: i64,
        kv_heads: i64,
        head_dim: i64,
        block_size: i64,
        max_blocks: i64,
        scale: f32,
        q_stride: i64,
        cu_seqlens_q: *const i32,
        stream: cudaStream_t,
    ) -> i32;
    // B1（kaon3-mega R36）：v3 decode attention 直入口（splits/keys_per_split 显式传，对拍用）。
    #[allow(clippy::too_many_arguments)]
    pub fn gqa_decode_v3_bf16(
        q: *const c_void,
        cache: *const c_void,
        block_table: *const i32,
        seq_lens: *const i32,
        cu_seqlens_q: *const i32,
        scratch: *mut f32,
        out_direct: *mut c_void,
        num_reqs: i64,
        heads: i64,
        kv_heads: i64,
        head_dim: i64,
        block_size: i64,
        max_blocks: i64,
        scale: f32,
        q_stride: i64,
        splits: i64,
        keys_per_split: i64,
        stream: cudaStream_t,
    ) -> i32;
    // B1：split partial → 输出 merge（persistent attn 对拍把 scratch 合并成最终输出）。
    #[allow(clippy::too_many_arguments)]
    pub fn gqa_decode_merge_v2_launch(
        scratch: *const f32,
        cu_seqlens_q: *const i32,
        seq_lens: *const i32,
        out: *mut c_void,
        num_reqs: i32,
        heads: i32,
        head_dim: i32,
        splits: i32,
        keys_per_split: i32,
        stream: cudaStream_t,
    ) -> i32;
    pub fn qkv_post_bf16(
        qkv: *mut c_void,
        wq: *const c_void,
        wk: *const c_void,
        positions: *const i32,
        cos_sin: *const f32,
        cache: *mut c_void,
        slot_mapping: *const i64,
        tokens: i64,
        heads: i64,
        kv_heads: i64,
        head_dim: i64,
        eps: f32,
        block_size: i64,
        stream: cudaStream_t,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub fn qkv_post_ext_bf16(
        qkv: *mut c_void,
        wq: *const c_void,
        wk: *const c_void,
        positions: *const i32,
        cos_sin: *const f32,
        cache: *mut c_void,
        slot_mapping: *const i64,
        tokens: i64,
        heads: i64,
        kv_heads: i64,
        head_dim: i64,
        rope_dim: i64,
        eps: f32,
        norm_beta: f32,
        qmul: i64,
        block_size: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn mul_sigmoid_bf16(x: *mut c_void, gate: *const c_void, n: i64, stream: cudaStream_t) -> i32;
    pub fn mul_sigmoid_rows_bf16(x: *mut c_void, gate: *const c_void, rows: i64, cols: i64, gate_stride: i64, stream: cudaStream_t) -> i32;
    pub fn situ_and_mul_bf16(
        x: *const c_void,
        out: *mut c_void,
        tokens: i64,
        inter: i64,
        beta: f32,
        linear_beta: f32,
        stream: cudaStream_t,
    ) -> i32;
    pub fn mla_decode_bf16(
        q: *const c_void,
        cache: *const c_void,
        block_table: *const i32,
        seq_lens: *const i32,
        topk_indices: *const i32,
        topk: i64,
        out: *mut c_void,
        batch: i64,
        heads: i64,
        entry: i64,
        kv_lora: i64,
        block_size: i64,
        max_blocks: i64,
        scale: f32,
        stream: cudaStream_t,
    ) -> i32;
    pub fn copy_cols_bf16(
        src: *const c_void,
        dst: *mut c_void,
        tokens: i64,
        src_cols: i64,
        col0: i64,
        dst_cols: i64,
        dcol0: i64,
        width: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn bcast_kpe_bf16(
        kpe: *const c_void,
        k: *mut c_void,
        tokens: i64,
        heads: i64,
        qk: i64,
        nope: i64,
        rope: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn copy_head_cols_bf16(
        src: *const c_void,
        dst: *mut c_void,
        tokens: i64,
        heads: i64,
        in_w: i64,
        in_col0: i64,
        out_w: i64,
        out_col0: i64,
        width: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn gather_head_bf16(
        src: *const c_void,
        dst: *mut c_void,
        tokens: i64,
        heads: i64,
        w: i64,
        head: i64,
        col0: i64,
        width: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn scatter_head_bf16(
        src: *const c_void,
        dst: *mut c_void,
        tokens: i64,
        heads: i64,
        w: i64,
        head: i64,
        col0: i64,
        width: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn zero_rows_where_pos0_bf16(
        x: *mut c_void,
        pos: *const i32,
        tokens: i64,
        cols: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn add_inplace_bf16(x: *const c_void, y: *mut c_void, n: i64, stream: cudaStream_t) -> i32;
    pub fn kv_pack(
        layer_ptrs: *const *const c_void,
        blocks: *const i32,
        staging: *mut c_void,
        n_blocks: i64,
        num_layers: i64,
        per_layer_bytes: i64,
        block_size: i64,
        kv_heads: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn kv_unpack(
        layer_ptrs: *const *mut c_void,
        blocks: *const i32,
        staging: *const c_void,
        n_blocks: i64,
        num_layers: i64,
        per_layer_bytes: i64,
        block_size: i64,
        kv_heads: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn gather_rows_bf16(
        src: *const c_void,
        dst: *mut c_void,
        rows: *const i32,
        num_rows: i64,
        width: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn bf16_to_f32(x: *const c_void, y: *mut f32, n: i64, stream: cudaStream_t) -> i32;
    pub fn gather_blocks_bf16_to_f32(
        x: *const c_void,
        y: *mut f32,
        nb: i64,
        rows: i64,
        cols: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn scale_f32(x: *mut f32, n: i64, a: f32, stream: cudaStream_t) -> i32;

    // ---- DSA ----
    pub fn layernorm_bf16(
        x: *const c_void,
        w: *const c_void,
        b: *const c_void,
        out: *mut c_void,
        tokens: i64,
        hidden: i64,
        eps: f32,
        stream: cudaStream_t,
    ) -> i32;

    pub fn scale_bf16(x: *mut c_void, n: i64, a: f32, stream: cudaStream_t) -> i32;
    pub fn fp8_e4m3_roundtrip_bf16(x: *mut c_void, n: i64, scale: f32, stream: cudaStream_t) -> i32;
    pub fn dsa_index_score_f32(
        q: *const c_void,
        w: *const f32,
        k_cache: *const c_void,
        block_table: *const i32,
        seq_lens: *const i32,
        cu_seqlens_q: *const i32,
        score: *mut f32,
        num_reqs: i64,
        heads: i64,
        head_dim: i64,
        block_size: i64,
        max_blocks: i64,
        max_seq: i64,
        scale: f32,
        stream: cudaStream_t,
    ) -> i32;
    pub fn dsa_topk_i32(
        score: *const f32,
        vis: *const i32,
        out: *mut i32,
        rows: i64,
        max_seq: i64,
        topk: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn dsa_row_vis_i32(
        seq_lens: *const i32,
        cu_seqlens_q: *const i32,
        vis: *mut i32,
        num_reqs: i64,
        stream: cudaStream_t,
    ) -> i32;

    // ---- MoE ----
    pub fn router_gemm_bi_f32(
        x: *const c_void,
        w: *const c_void,
        out: *mut f32,
        tokens: i64,
        num_experts: i64,
        hidden: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn moe_gate_f32(
        x: *const c_void,
        w: *const c_void,
        logits: *mut f32,
        tokens: i64,
        num_experts: i64,
        hidden: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn moe_route_f32(
        logits: *const f32,
        bias: *const f32,
        topk_ids: *mut i32,
        topk_w: *mut f32,
        tokens: i64,
        num_experts: i64,
        topk: i64,
        renormalize: i32,
        scale: f32,
        stream: cudaStream_t,
    ) -> i32;
    /// Gemma4 路由：全专家 softmax → 按 logits 取 top-k → top-k 上归一化 → 乘 per_expert_scale。
    /// （与 `moe_route_f32` 的 sigmoid 语义不同，见 `cuda/moe.cu` 注释与 vLLM `gemma4.py`。）
    pub fn moe_route_softmax_f32(
        logits: *const f32,
        topk_ids: *mut i32,
        per_expert_scale: *const c_void,
        topk_w: *mut f32,
        tokens: i64,
        num_experts: i64,
        topk: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn moe_bucket(
        topk_ids: *const i32,
        counts: *mut i32,
        offsets: *mut i32,
        cursor: *mut i32,
        sorted_tok: *mut i32,
        sorted_k: *mut i32,
        tokens: i64,
        num_experts: i64,
        topk: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn moe_gather_bf16(
        x: *const c_void,
        xg: *mut c_void,
        sorted_tok: *const i32,
        n: i64,
        hidden: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn moe_scatter_add_f32(
        y: *const c_void,
        acc: *mut f32,
        sorted_tok: *const i32,
        sorted_k: *const i32,
        topk_w: *const f32,
        per_expert: *const c_void,
        w_scale: f32,
        n: i64,
        topk: i64,
        hidden: i64,
        mode: i32,
        stream: cudaStream_t,
    ) -> i32;
    pub fn f32_add_to_bf16(acc: *const f32, out: *mut c_void, n: i64, stream: cudaStream_t) -> i32;

    // one-shot all-reduce（allreduce.cu）
    pub fn ar_signal_bytes() -> i64;
    pub fn ar_rank_data_bytes() -> i64;
    pub fn ar_oneshot_bf16(
        rank_data: *mut c_void,
        signals: *const *const c_void,
        out: *mut c_void,
        rank: i32,
        world: i32,
        n: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn ar_oneshot_add_rmsnorm_bf16(
        rank_data: *mut c_void,
        signals: *const *const c_void,
        res: *mut c_void,
        w: *const c_void,
        normed: *mut c_void,
        rank: i32,
        world: i32,
        tokens: i64,
        hidden: i64,
        eps: f32,
        stream: cudaStream_t,
    ) -> i32;
    // ---- KDA（cuda/kda.cu）----
    pub fn kda_conv1d_prefill(
        x: *const c_void,
        w: *const c_void,
        out: *mut c_void,
        conv_state: *mut c_void,
        cu_seqlens: *const i32,
        state_indices: *const i32,
        has_initial: *const i32,
        num_seqs: i64,
        dim: i64,
        width: i64,
        state_len: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn kda_conv1d_update(
        x: *const c_void,
        w: *const c_void,
        out: *mut c_void,
        conv_state: *mut c_void,
        state_indices: *const i32,
        batch: i64,
        dim: i64,
        width: i64,
        state_len: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn kda_gate(
        g: *const c_void,
        a_log: *const f32,
        dt_bias: *const f32,
        out: *mut f32,
        tokens: i64,
        heads: i64,
        head_dim: i64,
        lower_bound: f32,
        use_lower_bound: i32,
        stream: cudaStream_t,
    ) -> i32;
    pub fn kda_recurrent(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        raw_g: *const c_void,
        raw_beta: *const f32,
        a_log: *const f32,
        dt_bias: *const f32,
        out: *mut c_void,
        state: *mut c_void,
        cu_seqlens: *const i32,
        state_indices: *const i32,
        num_seqs: i64,
        heads: i64,
        head_dim: i64,
        state_stride_tok: i64,
        scale: f32,
        lower_bound: f32,
        use_lower_bound: i32,
        stream: cudaStream_t,
    ) -> i32;
    pub fn kda_gated_rmsnorm(
        x: *const c_void,
        w: *const c_void,
        gate: *const c_void,
        out: *mut c_void,
        rows: i64,
        head_dim: i64,
        eps: f32,
        stream: cudaStream_t,
    ) -> i32;
    // ---- KDA chunk 形式 prefill（kda_chunk.cu）----
    pub fn kda_chunk_prep(
        q: *const c_void,
        k: *const c_void,
        raw_g: *const c_void,
        raw_beta: *const f32,
        a_log: *const f32,
        dt_bias: *const f32,
        qn: *mut c_void,
        kn: *mut c_void,
        beta: *mut f32,
        gc: *mut f32,
        cu_seqlens: *const i32,
        num_seqs: i64,
        heads: i64,
        head_dim: i64,
        bt: i64,
        scale: f32,
        lower_bound: f32,
        use_lower_bound: i32,
        stream: cudaStream_t,
    ) -> i32;
    pub fn kda_chunk_wy(
        qn: *const c_void,
        kn: *const c_void,
        v: *const c_void,
        gc: *const f32,
        beta: *const f32,
        aq: *mut f32,
        u: *mut f32,
        w: *mut f32,
        kg: *mut f32,
        cu_seqlens: *const i32,
        chunk_tab: *const i32,
        num_chunks: i64,
        heads: i64,
        head_dim: i64,
        bt: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn kda_chunk_ho(
        qn: *const c_void,
        gc: *const f32,
        aq: *const f32,
        u: *const f32,
        w: *const f32,
        kg: *const f32,
        out: *mut c_void,
        state: *mut c_void,
        cu_seqlens: *const i32,
        cu_nt: *const i32,
        chunk_off: *const i32,
        state_indices: *const i32,
        state_stride_tok: i64,
        num_seqs: i64,
        heads: i64,
        head_dim: i64,
        bt: i64,
        stream: cudaStream_t,
    ) -> i32;

    // ---- GDN chunk 形式 prefill（gdn_chunk.cu）----
    pub fn gdn_chunk_prep(
        mixed_qkv: *const c_void,
        g: *const f32,
        qn: *mut c_void,
        kn: *mut c_void,
        gc: *mut f32,
        cu_seqlens: *const i32,
        chunk_tab: *const i32,
        total: i64,
        num_chunks: i64,
        h: i64,
        hv: i64,
        k: i64,
        v: i64,
        bt: i64,
        scale: f32,
        do_cumsum: i32,
        stream: cudaStream_t,
    ) -> i32;
    pub fn gdn_chunk_wy(
        qn: *const c_void,
        kn: *const c_void,
        mixed_qkv: *const c_void,
        gc: *const f32,
        beta: *const f32,
        aq: *mut c_void,
        u: *mut c_void,
        w: *mut c_void,
        kg: *mut c_void,
        cu_seqlens: *const i32,
        chunk_tab: *const i32,
        num_chunks: i64,
        h: i64,
        hv: i64,
        k: i64,
        v: i64,
        bt: i64,
        wy_mode: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn gdn_chunk_ho(
        qn: *const c_void,
        gc: *const f32,
        aq: *const c_void,
        u: *const c_void,
        w: *const c_void,
        kg: *const c_void,
        out: *mut c_void,
        state: *mut c_void,
        cu_seqlens: *const i32,
        cu_nt: *const i32,
        chunk_off: *const i32,
        state_indices: *const i32,
        state_stride_tok: i64,
        num_seqs: i64,
        h: i64,
        hv: i64,
        k: i64,
        v: i64,
        bt: i64,
        ho_bv: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn gdn_chunk_ho_v2(
        qn: *const c_void,
        gc: *const f32,
        aq: *const c_void,
        u: *const c_void,
        w: *const c_void,
        kg: *const c_void,
        h16: *mut c_void,
        v16: *mut c_void,
        out: *mut c_void,
        state: *mut c_void,
        cu_seqlens: *const i32,
        cu_nt: *const i32,
        chunk_off: *const i32,
        chunk_tab: *const i32,
        state_indices: *const i32,
        state_stride_tok: i64,
        num_seqs: i64,
        num_chunks: i64,
        h: i64,
        hv: i64,
        k: i64,
        v: i64,
        bt: i64,
        ho_bv: i64,
        prefetch: i64,
        split: i64,
        stream: cudaStream_t,
    ) -> i32;

    // ---- GDN（gdn.cu）----
    /// r8b：v2 + 快照表（`snap_off == null` = 无快照）；`snap_state` 与 `state` 同布局（槽 stride = state_stride_tok）。
    pub fn gdn_chunk_ho_v3(
        qn: *const c_void,
        gc: *const f32,
        aq: *const c_void,
        u: *const c_void,
        w: *const c_void,
        kg: *const c_void,
        h16: *mut c_void,
        v16: *mut c_void,
        out: *mut c_void,
        state: *mut c_void,
        cu_seqlens: *const i32,
        cu_nt: *const i32,
        chunk_off: *const i32,
        chunk_tab: *const i32,
        state_indices: *const i32,
        state_stride_tok: i64,
        num_seqs: i64,
        num_chunks: i64,
        h: i64,
        hv: i64,
        k: i64,
        v: i64,
        bt: i64,
        ho_bv: i64,
        prefetch: i64,
        split: i64,
        snap_off: *const i32,
        snap_pos: *const i32,
        snap_slot: *const i32,
        snap_state: *mut c_void,
        stream: cudaStream_t,
    ) -> i32;
    /// r8b：短卷积状态的边界快照（在 `gdn_conv1d_prefill` 覆盖活槽之前调用）。
    pub fn gdn_conv_snapshot(
        x: *const c_void,
        conv_state: *const c_void,
        snap_state: *mut c_void,
        cu_seqlens: *const i32,
        state_indices: *const i32,
        has_initial: *const i32,
        snap_off: *const i32,
        snap_pos: *const i32,
        snap_slot: *const i32,
        num_seqs: i64,
        dim: i64,
        state_len: i64,
        x_stride: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn gdn_conv1d_prefill(
        x: *const c_void,
        w: *const c_void,
        out: *mut c_void,
        conv_state: *mut c_void,
        cu_seqlens: *const i32,
        state_indices: *const i32,
        has_initial: *const i32,
        num_seqs: i64,
        dim: i64,
        width: i64,
        state_len: i64,
        act: i32,
        total: i64,
        rows_per_block: i64,
        x_stride: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn gdn_conv1d_update(
        x: *const c_void,
        w: *const c_void,
        out: *mut c_void,
        conv_state: *mut c_void,
        state_indices: *const i32,
        batch: i64,
        dim: i64,
        width: i64,
        state_len: i64,
        act: i32,
        x_stride: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn gdn_gate(
        a: *const c_void,
        b: *const c_void,
        a_log: *const f32,
        dt_bias: *const f32,
        g_out: *mut f32,
        beta_out: *mut f32,
        total: i64,
        heads: i64,
        do_beta: i32,
        ab_stride: i64,
        stream: cudaStream_t,
    ) -> i32;
    /// gate + beta + 块内 cumsum 一次算完（prefill；替代 gdn_gate + cumsum 两次 launch）。
    pub fn gdn_gate_cumsum(
        a: *const c_void,
        b: *const c_void,
        a_log: *const f32,
        dt_bias: *const f32,
        g_out: *mut f32,
        beta_out: *mut f32,
        gc_out: *mut f32,
        cu_seqlens: *const i32,
        chunk_tab: *const i32,
        num_chunks: i64,
        hv: i64,
        bt: i64,
        do_beta: i32,
        ab_stride: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn gdn_recurrent(
        mixed_qkv: *const c_void,
        g: *const f32,
        beta: *const f32,
        a_raw: *const c_void,
        b_raw: *const c_void,
        a_log: *const f32,
        dt_bias: *const f32,
        out: *mut c_void,
        state: *mut c_void,
        cu_seqlens: *const i32,
        state_indices: *const i32,
        num_seqs: i64,
        heads_k: i64,
        heads_v: i64,
        k_dim: i64,
        v_dim: i64,
        group: i64,
        scale: f32,
        state_stride_tok: i64,
        fused_gate: i32,
        vsplit: i64,
        decode_bv: i64,
        decode_variant: i32,
        decode_grid: i64,
        ab_stride: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn gdn_gated_rmsnorm(
        x: *const c_void,
        w: *const c_void,
        z: *const c_void,
        out: *mut c_void,
        rows: i64,
        head_dim: i64,
        eps: f32,
        act: i32,
        heads: i64,
        z_stride: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn mxfp4_dequant_bf16(
        packed: *const c_void,
        scale: *const c_void,
        out: *mut c_void,
        rows: i64,
        cols: i64,
        group: i64,
        stream: cudaStream_t,
    ) -> i32;
    // ---- MXFP4 grouped GEMM（mxfp4_gemm_sm8x.cu）----
    pub fn mxfp4_gemm_grouped(
        xg: *const c_void,
        wp: *const c_void,
        ws: *const c_void,
        expert_off: *const i32,
        tile_e: *const i32,
        tile_m: *const i32,
        tile_n: *const i32,
        out: *mut c_void,
        num_tiles: i64,
        k: i64,
        n: i64,
        group: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn mxfp4_gemm_grouped_sm100(
        xg: *const c_void,
        wp: *const c_void,
        ws: *const c_void,
        expert_off: *const i32,
        tile_e: *const i32,
        tile_m: *const i32,
        tile_n: *const i32,
        out: *mut c_void,
        num_tiles: i64,
        k: i64,
        n: i64,
        group: i64,
        stream: cudaStream_t,
    ) -> i32;

    pub fn gemm_small_m_bf16(
        a: *const c_void,
        w: *const c_void,
        bias: *const c_void,
        out: *mut c_void,
        partial: *mut f32,
        m: i64,
        n: i64,
        k: i64,
        splits: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn gemm_small_m_mma_bf16(
        a: *const c_void,
        w: *const c_void,
        bias: *const c_void,
        out: *mut c_void,
        partial: *mut f32,
        counters: *mut i32,
        m: i64,
        n: i64,
        k: i64,
        splits: i64,
        warps: i64,
        stripes: i64,
        stream: cudaStream_t,
    ) -> i32;
}

/// `kernels.h` 的 `MegaLayerArgs`（mega_sm100.cu，Qwen3.5 GDN 层 decode 单 kernel 实验）。字段顺序 / 类型必须与 C 一致。
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MegaLayerArgs {
    pub res: *const c_void,
    pub norm_w: *const c_void,
    pub post_norm_w: *const c_void,
    pub in_w: *const c_void,
    pub conv_w: *const c_void,
    pub a_log: *const f32,
    pub dt_bias: *const f32,
    pub gnorm_w: *const c_void,
    pub out_w: *const c_void,
    pub gate_up_w: *const c_void,
    pub down_w: *const c_void,
    pub conv_state: *mut c_void,
    pub rec_state: *mut c_void,
    pub state_indices: *const i32,
    pub xn: *mut c_void,
    pub qkvzab: *mut c_void,
    pub conv_out: *mut c_void,
    pub o: *mut c_void,
    pub n: *mut c_void,
    pub h2: *mut c_void,
    pub xn2: *mut c_void,
    pub gu: *mut c_void,
    pub act: *mut c_void,
    pub d: *mut c_void,
    pub partial: *mut f32,
    pub counters: *mut i32,
    pub info: *mut i32,
    pub prof: *mut u64,
    pub batch: i64,
    pub hidden: i64,
    pub inter: i64,
    pub heads_k: i64,
    pub heads_v: i64,
    pub k_dim: i64,
    pub v_dim: i64,
    pub width: i64,
    pub state_stride_tok: i64,
    pub eps: f32,
    pub scale: f32,
    pub groups: i64,
    pub unroll: i64,
    pub stages: i64,
    pub ctas_per_sm: i64,
    pub splits_in: i64,
    pub splits_out: i64,
    pub splits_gu: i64,
    pub splits_down: i64,
    pub gdn_hv: i64,
    pub gdn_bufs: i64,
}

/// cudaErrorNotSupported：mega kernel 未编入（CUDA_ARCHS 无 100/103 且无 MEGA_SM100_DEV）或 no-nvcc。
pub const CUDA_ERROR_NOT_SUPPORTED: i32 = 801;

#[cfg(all(not(no_nvcc_stub), mega_sm100))]
extern "C" {
    pub fn mega_gdn_layer_decode_sm100(a: *const MegaLayerArgs, stream: cudaStream_t) -> i32;
}
#[cfg(not(all(not(no_nvcc_stub), mega_sm100)))]
#[allow(clippy::missing_safety_doc)]
pub unsafe fn mega_gdn_layer_decode_sm100(_a: *const MegaLayerArgs, _s: cudaStream_t) -> i32 {
    CUDA_ERROR_NOT_SUPPORTED
}

/// dense MLP 支路 persistent megakernel（`mega_mlp_sm120.cu`，sm120 单卡）：
/// pre_norm → gate ‖ up → gelu·mul(round2) → down → post_norm，一次 launch。
/// 未编入（CUDA_ARCHS 无 120 或 no-nvcc）时返回 CUDA_ERROR_NOT_SUPPORTED。
#[cfg(all(not(no_nvcc_stub), sm120))]
extern "C" {
    #[allow(clippy::too_many_arguments)]
    pub fn mega_mlp_dense_branch(
        x: *const c_void,
        out: *mut c_void,
        pre_w: *const c_void,
        post_w: *const c_void,
        gate_w: *const c_void,
        gate_wscale: f32,
        gate_a_inv: f32,
        up_w: *const c_void,
        up_wscale: f32,
        up_a_inv: f32,
        dn_w: *const c_void,
        dn_wscale: f32,
        dn_a_inv: f32,
        norm1: *mut c_void,
        norm1_q: *mut c_void,
        gate_buf: *mut c_void,
        up_buf: *mut c_void,
        act_bf: *mut c_void,
        act: *mut c_void,
        down_out: *mut c_void,
        m: i32,
        h: i32,
        inter: i32,
        eps: f32,
        n_gu: i32,
        n_dn: i32,
        dep: *mut i32,
        count: *mut i32,
        start: *mut i32,
        ready: *mut i32,
        dep2: *mut i32,
        pipe: i32,
        prof: i32,
        wait_cyc: *mut u64,
        exec_cyc: *mut u64,
        sp_gu: i32,
        sp_dn: i32,
        part_gate: *mut c_void,
        part_up: *mut c_void,
        part_dn: *mut c_void,
        sem_gate: *mut i32,
        sem_up: *mut i32,
        sem_dn: *mut i32,
        ctas: i32,
        stream: cudaStream_t,
    ) -> i32;
}
#[cfg(not(all(not(no_nvcc_stub), sm120)))]
#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
pub unsafe fn mega_mlp_dense_branch(
    _x: *const c_void, _out: *mut c_void, _pre_w: *const c_void, _post_w: *const c_void, _gate_w: *const c_void,
    _gw: f32, _ga: f32, _up_w: *const c_void, _uw: f32, _ua: f32, _dn_w: *const c_void, _dw: f32, _da: f32,
    _n1: *mut c_void, _n1q: *mut c_void, _gb: *mut c_void, _ub: *mut c_void, _ab: *mut c_void, _a: *mut c_void,
    _do: *mut c_void, _m: i32, _h: i32, _i: i32, _eps: f32, _ngu: i32, _ndn: i32, _dep: *mut i32, _cnt: *mut i32,
    _st: *mut i32, _rdy: *mut i32, _dep2: *mut i32, _pipe: i32, _prof: i32, _wc: *mut u64, _ec: *mut u64,
    _spg: i32, _spd: i32, _pg: *mut c_void, _pu: *mut c_void, _pd: *mut c_void, _sg: *mut i32, _su: *mut i32,
    _sd: *mut i32, _ctas: i32, _s: cudaStream_t,
) -> i32 {
    CUDA_ERROR_NOT_SUPPORTED
}

/// B1（kaon3-mega R36）：persistent 版 paged GQA decode attention（`mega_attn_sm120.cu`，sm120 单卡）。
/// 一次 launch 持久 CTA 调度，数值走与 v3 同一 core（`gqa_decode_v3_core`），应与 gqa_decode_v3_bf16 逐位一致。
#[cfg(all(not(no_nvcc_stub), sm120))]
extern "C" {
    #[allow(clippy::too_many_arguments)]
    pub fn mega_attn_decode(
        q: *const c_void,
        cache: *const c_void,
        block_table: *const i32,
        seq_lens: *const i32,
        cu_seqlens_q: *const i32,
        scratch: *mut f32,
        out_direct: *mut c_void,
        num_reqs: i32,
        heads: i32,
        kv_heads: i32,
        head_dim: i32,
        block_size: i32,
        max_blocks: i32,
        scale: f32,
        q_stride: i32,
        splits: i32,
        keys_per_split: i32,
        dep: *mut i32,
        count: *mut i32,
        start: *mut i32,
        ready: *mut i32,
        wait_cyc: *mut u64,
        exec_cyc: *mut u64,
        prof: i32,
        stream: cudaStream_t,
    ) -> i32;
}
#[cfg(not(all(not(no_nvcc_stub), sm120)))]
#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
pub unsafe fn mega_attn_decode(
    _q: *const c_void, _cache: *const c_void, _bt: *const i32, _sl: *const i32, _cu: *const i32, _scr: *mut f32,
    _od: *mut c_void, _nr: i32, _h: i32, _kv: i32, _hd: i32, _bs: i32, _mb: i32, _scale: f32, _qs: i32, _sp: i32,
    _kps: i32, _dep: *mut i32, _cnt: *mut i32, _st: *mut i32, _rdy: *mut i32, _wc: *mut u64, _ec: *mut u64, _prof: i32,
    _s: cudaStream_t,
) -> i32 {
    CUDA_ERROR_NOT_SUPPORTED
}

/// tcgen05 工具链探针（`umma_peak_sm100.cu`，r12a）：M=128 N=256、K = 64·k_iters 的最小 UMMA GEMM。
#[cfg(all(not(no_nvcc_stub), mega_sm100))]
extern "C" {
    pub fn umma_peak_launch(a: *const c_void, b: *const c_void, c: *mut c_void, k_iters: i64, blocks: i64,
                            check: i64, stream: cudaStream_t) -> i32;
}
#[cfg(all(not(no_nvcc_stub), mega_sm100))]
extern "C" {
    pub fn gqa_fa_umma_launch(paged: i32, max_q: i32, heads: i32, num_reqs: i32, q: *const c_void, k: *const c_void,
                              v: *const c_void, cache: *const c_void, bt: *const i32, seq_lens: *const i32,
                              cu: *const i32, out: *mut c_void, h: i32, hkv: i32, scale_log2: f32, q_stride: i32,
                              kv_stride: i32, block_size: i32, max_blocks: i32, skip_single: i32, prof: *mut c_void,
                              stream: cudaStream_t) -> i32;
}
/// GDN ho 段的 tcgen05（UMMA）版（`gdn_ho_sm100.cu`，r11d）：BV = 全 V、h 主本在 state、A 补零到 M=128。
/// 未编入（CUDA_ARCHS 无 100/103 或 no-nvcc）时返回 CUDA_ERROR_NOT_SUPPORTED，调用方回退 `gdn_chunk_ho`。
#[cfg(all(not(no_nvcc_stub), mega_sm100))]
extern "C" {
    pub fn gdn_ho_umma(qn: *const c_void, gc: *const f32, aq: *const c_void, u: *const c_void, w: *const c_void,
                       kg: *const c_void, o: *mut c_void, st: *mut c_void, cu: *const i32, nt: *const i32,
                       si: *const i32, sst: i64, ns: i64, h: i64, hv: i64, k: i64, v: i64,
                       s: cudaStream_t) -> i32;
}
#[cfg(not(all(not(no_nvcc_stub), mega_sm100)))]
#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
pub unsafe fn gdn_ho_umma(_qn: *const c_void, _gc: *const f32, _aq: *const c_void, _u: *const c_void,
                          _w: *const c_void, _kg: *const c_void, _o: *mut c_void, _st: *mut c_void,
                          _cu: *const i32, _nt: *const i32, _si: *const i32, _sst: i64, _ns: i64, _h: i64,
                          _hv: i64, _k: i64, _v: i64, _s: cudaStream_t) -> i32 {
    CUDA_ERROR_NOT_SUPPORTED
}

#[cfg(not(all(not(no_nvcc_stub), mega_sm100)))]
#[allow(clippy::missing_safety_doc)]
pub unsafe fn umma_peak_launch(_a: *const c_void, _b: *const c_void, _c: *mut c_void, _k: i64, _b2: i64,
                               _ck: i64, _s: cudaStream_t) -> i32 {
    CUDA_ERROR_NOT_SUPPORTED
}

/// kaon3-mega B2：整层 persistent megakernel（`mega_layer_sm120.cu`）的 Params 镜像。
/// **字段顺序/类型必须与 C++ `dinfer::mega_layer::Params` 逐一对应**（repr(C)）。
/// 所有设备指针用 `*mut c_void`（指针大小无关具体类型）。
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MegaLayerParams {
    pub x: *const c_void,
    pub out: *mut c_void,
    pub input_w: *const c_void,
    pub q_norm_w: *const c_void,
    pub k_norm_w: *const c_void,
    pub post_attn_w: *const c_void,
    pub pre_ff_w: *const c_void,
    pub post_ff1_w: *const c_void,
    pub qkv_w: *const c_void,
    pub qkv_wscale: f32,
    pub qkv_a_inv: f32,
    pub o_w: *const c_void,
    pub o_wscale: f32,
    pub o_a_inv: f32,
    pub gate_w: *const c_void,
    pub gate_wscale: f32,
    pub gate_a_inv: f32,
    pub up_w: *const c_void,
    pub up_wscale: f32,
    pub up_a_inv: f32,
    pub dn_w: *const c_void,
    pub dn_wscale: f32,
    pub dn_a_inv: f32,
    pub norm_bf: *mut c_void,
    pub norm_q: *mut c_void,
    pub qkv_buf: *mut c_void,
    pub attn_buf: *mut c_void,
    pub attn_q: *mut c_void,
    pub o_buf: *mut c_void,
    pub x2: *mut c_void,
    pub norm1_bf: *mut c_void,
    pub norm1_q: *mut c_void,
    pub gate_buf: *mut c_void,
    pub up_buf: *mut c_void,
    pub act_q: *mut c_void,
    pub down_out: *mut c_void,
    pub act_scratch: *mut c_void,
    pub kv_cache: *mut c_void,
    pub block_table: *const i32,
    pub seq_lens: *const i32,
    pub slot_mapping: *const i32,
    pub attn_scratch: *mut f32,
    pub cos_sin: *const f32,
    pub positions: *const i32,
    pub rope_dim: i32,
    pub rope_span: i32,
    pub rope_neox: i32,
    pub m: i32,
    pub h: i32,
    pub i: i32,
    pub heads: i32,
    pub nkv: i32,
    pub d: i32,
    pub block_size: i32,
    pub max_blocks: i32,
    pub eps: f32,
    pub attn_scale: f32,
    pub kv_fp8: i32,
    pub splits: i32,
    pub keys_per_split: i32,
    pub sp_qkv: i32,
    pub sp_o: i32,
    pub sp_gu: i32,
    pub sp_dn: i32,
    pub part_qkv: *mut f32,
    pub part_o: *mut f32,
    pub part_gate: *mut f32,
    pub part_up: *mut f32,
    pub part_dn: *mut f32,
    pub sem_qkv: *mut i32,
    pub sem_o: *mut i32,
    pub sem_gate: *mut i32,
    pub sem_up: *mut i32,
    pub sem_dn: *mut i32,
    /// 片上流水：1 → S_QKNORM 吸收 rope+kv 写（不落 global 中间量）；0 → 原分离三 stage（基线）。
    pub fuse_glue: i32,
}

/// 整层 megakernel 一次/分段 launch。融合：stage_lo=0,stage_hi=14,setup=1 一次跑全链。
/// 分离基线：对每个 s 调一次 (s,s)，首个 setup=1，其余 0（流序列化 + ready 累加 → 等价分别 launch）。
#[cfg(all(not(no_nvcc_stub), sm120))]
extern "C" {
    #[allow(clippy::too_many_arguments)]
    pub fn mega_layer_forward(
        p: *const MegaLayerParams,
        stage_lo: i32,
        stage_hi: i32,
        setup: i32,
        dep: *mut i32,
        count: *mut i32,
        start: *mut i32,
        ready: *mut i32,
        dep2: *mut i32,
        wait_cyc: *mut u64,
        exec_cyc: *mut u64,
        prof: i32,
        stream: cudaStream_t,
    ) -> i32;
}
#[cfg(not(all(not(no_nvcc_stub), sm120)))]
#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
pub unsafe fn mega_layer_forward(
    _p: *const MegaLayerParams, _lo: i32, _hi: i32, _setup: i32, _dep: *mut i32, _cnt: *mut i32, _st: *mut i32,
    _rdy: *mut i32, _dep2: *mut i32, _wc: *mut u64, _ec: *mut u64, _prof: i32, _s: cudaStream_t,
) -> i32 {
    CUDA_ERROR_NOT_SUPPORTED
}

#[cfg(not(all(not(no_nvcc_stub), mega_sm100)))]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gqa_fa_umma_launch(_paged: i32, _max_q: i32, _heads: i32, _num_reqs: i32, _q: *const c_void,
                                 _k: *const c_void, _v: *const c_void, _cache: *const c_void, _bt: *const i32,
                                 _seq_lens: *const i32, _cu: *const i32, _out: *mut c_void, _h: i32, _hkv: i32,
                                 _scale_log2: f32, _q_stride: i32, _kv_stride: i32, _block_size: i32,
                                 _max_blocks: i32, _skip_single: i32, _prof: *mut c_void, _s: cudaStream_t) -> i32 {
    CUDA_ERROR_NOT_SUPPORTED
}

/// sm_120 专用低精度 GEMM（`gemm_fp8_sm120.cu` / `gemm_nvfp4_sm120.cu`）。
/// 未编入（CUDA_ARCHS 无 120 或 no-nvcc）时返回 `CUDA_ERROR_NOT_SUPPORTED`，调用方回退。
/// 判据与 build.rs 的 `cargo:rustc-cfg=sm120` 同源：CUDA_ARCHS 含 120 / 120a。
#[cfg(all(not(no_nvcc_stub), sm120))]
extern "C" {
    pub fn gemm_fp8_bf16(a: *const c_void, w: *const c_void, w_scale: f32, out: *mut c_void, m: i64, n: i64,
                         k: i64, stream: cudaStream_t) -> i32;
    pub fn gemm_fp8_static_bf16(a: *const c_void, a_scale: f32, w: *const c_void, w_scale: f32, out: *mut c_void,
                                m: i64, n: i64, k: i64, stream: cudaStream_t) -> i32;
    pub fn gemm_fp8_prequant_bf16(a: *const c_void, a_scale: f32, w: *const c_void, w_scale: f32, out: *mut c_void,
                                  m: i64, n: i64, k: i64, stream: cudaStream_t) -> i32;
    /// bf16 → e4m3 量化（按静态 per-tensor 激活 scale），与 `gemm_fp8_static_bf16` 内联的量化逐位一致。
    pub fn quant_fp8_bf16(x: *const c_void, a_scale: f32, out: *mut c_void, n: i64, stream: cudaStream_t) -> i32;
    pub fn gemm_nvfp4_bf16(a: *const c_void, packed: *const c_void, scale: *const c_void, global: f32,
                           out: *mut c_void, m: i64, n: i64, k: i64, stream: cudaStream_t) -> i32;
    pub fn gemm_nvfp4_prequant_bf16(
        aq: *const c_void,
        a_s: *const c_void,
        packed: *const c_void,
        scale: *const c_void,
        global: f32,
        out: *mut c_void,
        m: i64,
        n: i64,
        k: i64,
        swz: i32,
        stream: cudaStream_t,
    ) -> i32;
    pub fn gemm_nvfp4_gs_bf16(a: *const c_void, a_gscale: f32, packed: *const c_void, scale: *const c_void, global: f32,
                              out: *mut c_void, m: i64, n: i64, k: i64, stream: cudaStream_t) -> i32;
}
#[cfg(not(all(not(no_nvcc_stub), sm120)))]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_fp8_static_bf16(_a: *const c_void, _as: f32, _w: *const c_void, _ws: f32, _o: *mut c_void, _m: i64,
                                   _n: i64, _k: i64, _s: cudaStream_t) -> i32 {
    CUDA_ERROR_NOT_SUPPORTED
}
#[cfg(not(all(not(no_nvcc_stub), sm120)))]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_fp8_prequant_bf16(_a: *const c_void, _as: f32, _w: *const c_void, _ws: f32, _o: *mut c_void,
                                     _m: i64, _n: i64, _k: i64, _s: cudaStream_t) -> i32 {
    CUDA_ERROR_NOT_SUPPORTED
}
#[cfg(not(all(not(no_nvcc_stub), sm120)))]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn quant_fp8_bf16(_x: *const c_void, _as: f32, _o: *mut c_void, _n: i64, _s: cudaStream_t) -> i32 {
    CUDA_ERROR_NOT_SUPPORTED
}
#[cfg(not(all(not(no_nvcc_stub), sm120)))]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_fp8_bf16(_a: *const c_void, _w: *const c_void, _ws: f32, _o: *mut c_void, _m: i64, _n: i64,
                            _k: i64, _s: cudaStream_t) -> i32 {
    CUDA_ERROR_NOT_SUPPORTED
}
#[cfg(not(all(not(no_nvcc_stub), sm120)))]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_nvfp4_prequant_bf16(_aq: *const c_void, _as: *const c_void, _p: *const c_void,
                                       _sc: *const c_void, _g: f32, _o: *mut c_void, _m: i64, _n: i64,
                                       _k: i64, _swz: i32, _s: cudaStream_t) -> i32 {
    CUDA_ERROR_NOT_SUPPORTED
}
#[cfg(not(all(not(no_nvcc_stub), sm120)))]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_nvfp4_gs_bf16(_a: *const c_void, _gs: f32, _p: *const c_void, _sc: *const c_void, _g: f32,
                                 _o: *mut c_void, _m: i64, _n: i64, _k: i64, _s: cudaStream_t) -> i32 {
    CUDA_ERROR_NOT_SUPPORTED
}
#[cfg(not(all(not(no_nvcc_stub), sm120)))]
#[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
pub unsafe fn gemm_nvfp4_bf16(_a: *const c_void, _p: *const c_void, _sc: *const c_void, _g: f32, _o: *mut c_void,
                              _m: i64, _n: i64, _k: i64, _s: cudaStream_t) -> i32 {
    CUDA_ERROR_NOT_SUPPORTED
}

#[cfg(no_nvcc_stub)]
mod stubs {
    use super::*;
    macro_rules! stub {
        ($name:ident ( $($arg:ident : $ty:ty),* )) => {
            #[allow(clippy::too_many_arguments, clippy::missing_safety_doc)]
            pub unsafe fn $name($(_: $ty),*) -> i32 {
                panic!(concat!("dinfer-kernels built with `no-nvcc`: ", stringify!($name), " unavailable"))
            }
        };
    }
    stub!(add_f32(a: *const f32, b: *const f32, out: *mut f32, n: i64, s: cudaStream_t));
    stub!(rmsnorm_bf16(x: *const c_void, w: *const c_void, o: *mut c_void, t: i64, h: i64, e: f32, l: i32, m: i32, s: cudaStream_t));
    stub!(rmsnorm_scale2_bf16(x: *const c_void, sc: *const c_void, o: *mut c_void, t: i64, h: i64, e: f32, r: f32, m: i32, s: cudaStream_t));
    stub!(rmsnorm_quant_fp8_bf16(x: *const c_void, w: *const c_void, o: *mut c_void, t: i64, h: i64, e: f32, a: f32, s: cudaStream_t));
    stub!(fused_add_rmsnorm_bf16(x: *const c_void, r: *mut c_void, w: *const c_void, o: *mut c_void, t: i64, h: i64, e: f32, l: i32, m: i32, s: cudaStream_t));
    stub!(rope_bf16(q: *mut c_void, k: *mut c_void, p: *const i32, cs: *const f32, t: i64, qh: i64, kh: i64, hd: i64, rd: i64, n: i32, s: cudaStream_t));
    stub!(rope_span_bf16(q: *mut c_void, k: *mut c_void, p: *const i32, cs: *const f32, t: i64, qh: i64, kh: i64, hd: i64, rd: i64, sp: i64, n: i32, s: cudaStream_t));
    stub!(silu_and_mul_bf16(x: *const c_void, o: *mut c_void, t: i64, i: i64, s: cudaStream_t));
    stub!(mul_sigmoid_bf16(x: *mut c_void, g: *const c_void, n: i64, s: cudaStream_t));
    stub!(mul_sigmoid_rows_bf16(x: *mut c_void, g: *const c_void, r: i64, c: i64, gs: i64, s: cudaStream_t));
    stub!(situ_and_mul_bf16(x: *const c_void, o: *mut c_void, t: i64, i: i64, b: f32, lb: f32, s: cudaStream_t));
    stub!(qkv_post_ext_bf16(q: *mut c_void, wq: *const c_void, wk: *const c_void, p: *const i32, cs: *const f32, c: *mut c_void, sm: *const i64, t: i64, h: i64, hk: i64, hd: i64, rd: i64, e: f32, nb: f32, qm: i64, bs: i64, s: cudaStream_t));
    stub!(embedding_bf16(ids: *const i32, tb: *const c_void, o: *mut c_void, t: i64, h: i64, v: i64, s: cudaStream_t));
    stub!(argmax_f32(l: *const f32, o: *mut i32, b: i64, v: i64, s: cudaStream_t));
    stub!(topk_topp_mask_f32(l: *mut f32, k: *const i32, p: *const f32, b: i64, v: i64, s: cudaStream_t));
    stub!(sample_f32(l: *const f32, t: *const f32, sd: *const u64, of: *const u64, o: *mut i32, b: i64, v: i64, s: cudaStream_t));
    stub!(mla_cache_write_bf16(l: *const c_void, c: *mut c_void, sm: *const i64, t: i64, e: i64, bs: i64, s: cudaStream_t));
    stub!(mla_prefill_bf16(q: *const c_void, k: *const c_void, v: *const c_void, o: *mut c_void, cu: *const i32, ti: *const i32, tk: i64, n: i64, h: i64, dq: i64, dv: i64, sc: f32, s: cudaStream_t));
    stub!(mla_prefill_paged_bf16(q: *const c_void, c: *const c_void, bt: *const i32, sl: *const i32, cu: *const i32, ti: *const i32, tk: i64, o: *mut c_void, n: i64, h: i64, e: i64, l: i64, bs: i64, mb: i64, sc: f32, s: cudaStream_t));
    stub!(mla_decode_bf16(q: *const c_void, c: *const c_void, bt: *const i32, sl: *const i32, ti: *const i32, tk: i64, o: *mut c_void, b: i64, h: i64, e: i64, l: i64, bs: i64, mb: i64, sc: f32, s: cudaStream_t));
    stub!(layernorm_bf16(x: *const c_void, w: *const c_void, b: *const c_void, o: *mut c_void, t: i64, h: i64, e: f32, s: cudaStream_t));
    stub!(dsa_index_score_f32(q: *const c_void, w: *const f32, kc: *const c_void, bt: *const i32, sl: *const i32, cu: *const i32, sc: *mut f32, n: i64, h: i64, d: i64, bs: i64, mb: i64, ms: i64, scale: f32, s: cudaStream_t));
    stub!(dsa_topk_i32(sc: *const f32, vis: *const i32, o: *mut i32, r: i64, ms: i64, k: i64, s: cudaStream_t));
    stub!(dsa_row_vis_i32(sl: *const i32, cu: *const i32, v: *mut i32, n: i64, s: cudaStream_t));
    stub!(copy_cols_bf16(a: *const c_void, b: *mut c_void, t: i64, sc: i64, c0: i64, dc: i64, d0: i64, w: i64, s: cudaStream_t));
    stub!(bcast_kpe_bf16(a: *const c_void, b: *mut c_void, t: i64, h: i64, qk: i64, n: i64, r: i64, s: cudaStream_t));
    stub!(copy_head_cols_bf16(a: *const c_void, b: *mut c_void, t: i64, h: i64, iw: i64, ic: i64, ow: i64, oc: i64, w: i64, s: cudaStream_t));
    stub!(gather_head_bf16(a: *const c_void, b: *mut c_void, t: i64, h: i64, w: i64, hd: i64, c0: i64, wd: i64, s: cudaStream_t));
    stub!(scatter_head_bf16(a: *const c_void, b: *mut c_void, t: i64, h: i64, w: i64, hd: i64, c0: i64, wd: i64, s: cudaStream_t));
    stub!(moe_gate_f32(x: *const c_void, w: *const c_void, l: *mut f32, t: i64, e: i64, h: i64, s: cudaStream_t));
    stub!(router_gemm_bi_f32(x: *const c_void, w: *const c_void, o: *mut f32, t: i64, e: i64, h: i64, s: cudaStream_t));
    stub!(moe_route_softmax_f32(l: *const f32, ti: *mut i32, pes: *const c_void, tw: *mut f32, t: i64, e: i64, k: i64, s: cudaStream_t)
        => moe_route_softmax_f32, "moe_route_softmax_f32"
        [l, ti, pes, tw, t, e, k, s]);
    stub!(moe_route_f32(l: *const f32, b: *const f32, ti: *mut i32, tw: *mut f32, t: i64, e: i64, k: i64, r: i32, sc: f32, s: cudaStream_t));
    stub!(moe_bucket(ti: *const i32, c: *mut i32, o: *mut i32, cu: *mut i32, st: *mut i32, sk: *mut i32, t: i64, e: i64, k: i64, s: cudaStream_t));
    stub!(moe_gather_bf16(x: *const c_void, xg: *mut c_void, st: *const i32, n: i64, h: i64, s: cudaStream_t));
    stub!(moe_scatter_add_f32(y: *const c_void, a: *mut f32, st: *const i32, sk: *const i32, tw: *const f32, n: i64, k: i64, h: i64, s: cudaStream_t));
    stub!(mxfp4_dequant_bf16(p: *const c_void, sc: *const c_void, o: *mut c_void, r: i64, c: i64, g: i64, s: cudaStream_t));
    stub!(mxfp4_gemm_grouped(xg: *const c_void, wp: *const c_void, ws: *const c_void, eo: *const i32, te: *const i32, tm: *const i32, tn: *const i32, o: *mut c_void, nt: i64, k: i64, n: i64, g: i64, s: cudaStream_t));
    stub!(mxfp4_gemm_grouped_sm100(xg: *const c_void, wp: *const c_void, ws: *const c_void, eo: *const i32, te: *const i32, tm: *const i32, tn: *const i32, o: *mut c_void, nt: i64, k: i64, n: i64, g: i64, s: cudaStream_t));
    stub!(kda_conv1d_prefill(x: *const c_void, w: *const c_void, o: *mut c_void, cs: *mut c_void, cu: *const i32, si: *const i32, hi: *const i32, ns: i64, d: i64, wid: i64, sl: i64, s: cudaStream_t));
    stub!(kda_conv1d_update(x: *const c_void, w: *const c_void, o: *mut c_void, cs: *mut c_void, si: *const i32, b: i64, d: i64, wid: i64, sl: i64, s: cudaStream_t));
    stub!(kda_gate(g: *const c_void, al: *const f32, db: *const f32, o: *mut f32, t: i64, h: i64, hd: i64, lb: f32, ulb: i32, s: cudaStream_t));
    stub!(kda_recurrent(q: *const c_void, k: *const c_void, v: *const c_void, g: *const c_void, rb: *const f32, al: *const f32, db: *const f32, o: *mut c_void, st: *mut c_void, cu: *const i32, si: *const i32, ns: i64, h: i64, hd: i64, sst: i64, sc: f32, lb: f32, ulb: i32, s: cudaStream_t));
    stub!(kda_gated_rmsnorm(x: *const c_void, w: *const c_void, g: *const c_void, o: *mut c_void, r: i64, hd: i64, e: f32, s: cudaStream_t));
    stub!(kda_chunk_prep(q: *const c_void, k: *const c_void, g: *const c_void, rb: *const f32, al: *const f32, db: *const f32, qn: *mut c_void, kn: *mut c_void, be: *mut f32, gc: *mut f32, cu: *const i32, ns: i64, h: i64, hd: i64, bt: i64, sc: f32, lb: f32, ulb: i32, s: cudaStream_t));
    stub!(kda_chunk_wy(qn: *const c_void, kn: *const c_void, v: *const c_void, gc: *const f32, be: *const f32, aq: *mut f32, u: *mut f32, w: *mut f32, kg: *mut f32, cu: *const i32, ct: *const i32, nc: i64, h: i64, hd: i64, bt: i64, s: cudaStream_t));
    stub!(kda_chunk_ho(qn: *const c_void, gc: *const f32, aq: *const f32, u: *const f32, w: *const f32, kg: *const f32, o: *mut c_void, st: *mut c_void, cu: *const i32, nt: *const i32, co: *const i32, si: *const i32, sst: i64, ns: i64, h: i64, hd: i64, bt: i64, s: cudaStream_t));
    stub!(gdn_chunk_prep(mq: *const c_void, g: *const f32, qn: *mut c_void, kn: *mut c_void, gc: *mut f32, cu: *const i32, ct: *const i32, t: i64, nc: i64, h: i64, hv: i64, k: i64, v: i64, bt: i64, sc: f32, s: cudaStream_t));
    stub!(gdn_chunk_wy(qn: *const c_void, kn: *const c_void, mq: *const c_void, gc: *const f32, be: *const f32, aq: *mut f32, u: *mut f32, w: *mut f32, kg: *mut f32, cu: *const i32, ct: *const i32, nc: i64, h: i64, hv: i64, k: i64, v: i64, bt: i64, wy_mode: i64, s: cudaStream_t));
    stub!(gdn_chunk_ho(qn: *const c_void, gc: *const f32, aq: *const f32, u: *const f32, w: *const f32, kg: *const f32, o: *mut c_void, st: *mut c_void, cu: *const i32, nt: *const i32, co: *const i32, si: *const i32, sst: i64, ns: i64, h: i64, hv: i64, k: i64, v: i64, bt: i64, ho_bv: i64, s: cudaStream_t));
    stub!(gdn_chunk_ho_v2(qn: *const c_void, gc: *const f32, aq: *const f32, u: *const f32, w: *const f32, kg: *const f32, h16: *mut c_void, v16: *mut c_void, o: *mut c_void, st: *mut c_void, cu: *const i32, nt: *const i32, co: *const i32, ct: *const i32, si: *const i32, sst: i64, ns: i64, nc: i64, h: i64, hv: i64, k: i64, v: i64, bt: i64, ho_bv: i64, pf: i64, split: i64, s: cudaStream_t));
    stub!(gdn_chunk_ho_v3(qn: *const c_void, gc: *const f32, aq: *const f32, u: *const f32, w: *const f32, kg: *const f32, h16: *mut c_void, v16: *mut c_void, o: *mut c_void, st: *mut c_void, cu: *const i32, nt: *const i32, co: *const i32, ct: *const i32, si: *const i32, sst: i64, ns: i64, nc: i64, h: i64, hv: i64, k: i64, v: i64, bt: i64, ho_bv: i64, pf: i64, split: i64, so: *const i32, sp: *const i32, ss: *const i32, sstate: *mut c_void, s: cudaStream_t));
    stub!(gdn_conv_snapshot(x: *const c_void, cs: *const c_void, ss: *mut c_void, cu: *const i32, si: *const i32, hi: *const i32, so: *const i32, sp: *const i32, sl: *const i32, ns: i64, d: i64, slen: i64, xs: i64, s: cudaStream_t));
    stub!(gdn_conv1d_prefill(x: *const c_void, w: *const c_void, o: *mut c_void, cs: *mut c_void, cu: *const i32, si: *const i32, hi: *const i32, ns: i64, d: i64, wd: i64, sl: i64, a: i32, total: i64, rpb: i64, xs: i64, s: cudaStream_t));
    stub!(gdn_conv1d_update(x: *const c_void, w: *const c_void, o: *mut c_void, cs: *mut c_void, si: *const i32, b: i64, d: i64, wd: i64, sl: i64, a: i32, xs: i64, s: cudaStream_t));
    stub!(gdn_gate(a: *const c_void, b: *const c_void, al: *const f32, db: *const f32, g: *mut f32, be: *mut f32, t: i64, h: i64, d: i32, abs: i64, s: cudaStream_t));
    stub!(gdn_gate_cumsum(a: *const c_void, b: *const c_void, al: *const f32, db: *const f32, g: *mut f32, be: *mut f32, gc: *mut f32, cu: *const i32, ct: *const i32, nc: i64, hv: i64, bt: i64, d: i32, abs: i64, s: cudaStream_t) -> i32);
    stub!(gdn_recurrent(mq: *const c_void, g: *const f32, be: *const f32, ar: *const c_void, br: *const c_void, al: *const f32, db: *const f32, o: *mut c_void, st: *mut c_void, cu: *const i32, si: *const i32, ns: i64, hk: i64, hv: i64, k: i64, v: i64, gp: i64, sc: f32, sst: i64, fg: i32, vs: i64, dbv: i64, dv: i32, dg: i64, abs: i64, s: cudaStream_t));
    stub!(gemm_small_m_bf16(a: *const c_void, w: *const c_void, b: *const c_void, o: *mut c_void, p: *mut f32, m: i64, n: i64, k: i64, sp: i64, s: cudaStream_t));
    stub!(gemm_small_m_mma_bf16(a: *const c_void, w: *const c_void, b: *const c_void, o: *mut c_void, p: *mut f32, c: *mut i32, m: i64, n: i64, k: i64, sp: i64, wp: i64, st: i64, s: cudaStream_t));
    stub!(gdn_gated_rmsnorm(x: *const c_void, w: *const c_void, z: *const c_void, o: *mut c_void, r: i64, hd: i64, e: f32, a: i32, h: i64, zs: i64, s: cudaStream_t));
    stub!(f32_add_to_bf16(a: *const f32, o: *mut c_void, n: i64, s: cudaStream_t));
    stub!(gather_blocks_bf16_to_f32(x: *const c_void, y: *mut f32, nb: i64, r: i64, c: i64, s: cudaStream_t));
    stub!(add_inplace_bf16(x: *const c_void, y: *mut c_void, n: i64, s: cudaStream_t));
    stub!(gather_rows_bf16(a: *const c_void, b: *mut c_void, r: *const i32, n: i64, w: i64, s: cudaStream_t));
    stub!(bf16_to_f32(x: *const c_void, y: *mut f32, n: i64, s: cudaStream_t));
    stub!(scale_f32(x: *mut f32, n: i64, a: f32, s: cudaStream_t));
    stub!(ar_oneshot_bf16(rd: *mut c_void, sg: *const *const c_void, o: *mut c_void, r: i32, w: i32, n: i64, s: cudaStream_t));
    stub!(ar_oneshot_add_rmsnorm_bf16(rd: *mut c_void, sg: *const *const c_void, res: *mut c_void, w: *const c_void, o: *mut c_void, r: i32, wd: i32, t: i64, h: i64, e: f32, s: cudaStream_t));
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn ar_signal_bytes() -> i64 {
        0
    }
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn ar_rank_data_bytes() -> i64 {
        0
    }
}
#[cfg(no_nvcc_stub)]
pub use stubs::*;
