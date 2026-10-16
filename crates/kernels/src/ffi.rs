//! `cuda/dinfer_kernels.h` 的 Rust 声明。手写而非 bindgen，保持签名可审计。
//! 每个入口返回 cudaError_t 数值（0 = success）。符号名不带项目前缀（项目可能改名）。

#![allow(non_camel_case_types)]

use std::ffi::c_void;

pub type cudaStream_t = *mut c_void;

#[cfg(not(dinfer_no_nvcc))]
extern "C" {
    pub fn add_f32(a: *const f32, b: *const f32, out: *mut f32, n: i64, stream: cudaStream_t) -> i32;

    pub fn rmsnorm_bf16(
        x: *const c_void,
        w: *const c_void,
        out: *mut c_void,
        tokens: i64,
        hidden: i64,
        eps: f32,
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

    pub fn silu_and_mul_bf16(
        x: *const c_void,
        out: *mut c_void,
        tokens: i64,
        inter: i64,
        stream: cudaStream_t,
    ) -> i32;

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
    pub fn min_p_mask_f32(logits: *mut f32, min_p: *const f32, batch: i64, vocab: i64, stream: cudaStream_t) -> i32;
    pub fn temperature_f32(logits: *mut f32, temperature: *const f32, batch: i64, vocab: i64, stream: cudaStream_t) -> i32;
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
        heads: i64,
        kv_heads: i64,
        head_dim: i64,
        scale: f32,
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
        heads: i64,
        kv_heads: i64,
        head_dim: i64,
        block_size: i64,
        max_blocks: i64,
        scale: f32,
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
    pub fn gather_rows_bf16(
        src: *const c_void,
        dst: *mut c_void,
        rows: *const i32,
        num_rows: i64,
        width: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn bf16_to_f32(x: *const c_void, y: *mut f32, n: i64, stream: cudaStream_t) -> i32;
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
        n: i64,
        topk: i64,
        hidden: i64,
        stream: cudaStream_t,
    ) -> i32;
    pub fn f32_add_to_bf16(acc: *const f32, out: *mut c_void, n: i64, stream: cudaStream_t) -> i32;
}

#[cfg(dinfer_no_nvcc)]
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
    stub!(rmsnorm_bf16(x: *const c_void, w: *const c_void, o: *mut c_void, t: i64, h: i64, e: f32, s: cudaStream_t));
    stub!(fused_add_rmsnorm_bf16(x: *const c_void, r: *mut c_void, w: *const c_void, o: *mut c_void, t: i64, h: i64, e: f32, s: cudaStream_t));
    stub!(rope_bf16(q: *mut c_void, k: *mut c_void, p: *const i32, cs: *const f32, t: i64, qh: i64, kh: i64, hd: i64, rd: i64, n: i32, s: cudaStream_t));
    stub!(silu_and_mul_bf16(x: *const c_void, o: *mut c_void, t: i64, i: i64, s: cudaStream_t));
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
    stub!(moe_route_f32(l: *const f32, b: *const f32, ti: *mut i32, tw: *mut f32, t: i64, e: i64, k: i64, r: i32, sc: f32, s: cudaStream_t));
    stub!(moe_bucket(ti: *const i32, c: *mut i32, o: *mut i32, cu: *mut i32, st: *mut i32, sk: *mut i32, t: i64, e: i64, k: i64, s: cudaStream_t));
    stub!(moe_gather_bf16(x: *const c_void, xg: *mut c_void, st: *const i32, n: i64, h: i64, s: cudaStream_t));
    stub!(moe_scatter_add_f32(y: *const c_void, a: *mut f32, st: *const i32, sk: *const i32, tw: *const f32, n: i64, k: i64, h: i64, s: cudaStream_t));
    stub!(f32_add_to_bf16(a: *const f32, o: *mut c_void, n: i64, s: cudaStream_t));
    stub!(add_inplace_bf16(x: *const c_void, y: *mut c_void, n: i64, s: cudaStream_t));
    stub!(gather_rows_bf16(a: *const c_void, b: *mut c_void, r: *const i32, n: i64, w: i64, s: cudaStream_t));
    stub!(bf16_to_f32(x: *const c_void, y: *mut f32, n: i64, s: cudaStream_t));
    stub!(scale_f32(x: *mut f32, n: i64, a: f32, s: cudaStream_t));
}
#[cfg(dinfer_no_nvcc)]
pub use stubs::*;
