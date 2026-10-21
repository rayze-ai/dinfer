# Kaon-V3 阶段 3：config 解析 + 滑窗 kernel（2026-10-21）

## 结论

1. **Gemma4 的 proportional RoPE 分母是 head_dim，不是 rope_dim**：full 层旋转 128 维
   （`head_dim 512 × partial_rotary_factor 0.25`），但 `inv_freq[i] = theta^(-2i/512)`。
   依据 vLLM `vllm/model_executor/layers/rotary_embedding/gemma4_rope.py` 与 HF
   `_compute_proportional_rope_parameters`。**单测通过**。
2. **滑窗 mask 只需上下界 + 起始 tile 跳读**：三个 tensor-core kernel（v5/v6/v7）与两个参考 kernel
   全部改完，`CUDA_ARCHS=120 cargo build --release -p dinfer-kernels` **通过（18.34 s）**。
3. `window=0` 路径与改动前语义等价（下界 `INT_MIN`、`tile0=0`、双缓冲索引一致），
   **但逐位回归尚未跑**（下一步第一件事）。

## 验证命令与输出

```
# 单测（config::tests::proportional_rope_full_layer_uses_head_dim_denom）
docker run --rm --entrypoint bash --gpus device=0 -v /hf_data/yaochen:/w ... \
  -e CUDA_ARCHS=120 image.paigpu.com/.../vllm-openai:v0.28.0-pegaflow-...-cu130 \
  -lc 'export LD_LIBRARY_PATH=<nccl dir>:<nvrtc dir>:$LD_LIBRARY_PATH; cargo test -p dinfer-kaon3 config::'
→ test result: ok. 1 passed; 0 failed

# kernel 编译
cargo build --release -p dinfer-kernels
→ Finished `release` profile [optimized + debuginfo] target(s) in 18.34s
```

## 编译期踩的三个坑（已修）

1. `tile0` / `lo_a` 插在 `warp_min_vis` 声明之前 → `identifier is undefined`；
2. v7 没有 `warp_min_vis`，只有 `pair_min_vis`，`tile0` 必须用后者；
3. `ffi.rs` 批量替换时把 `window` 误加到旧 `gqa_paged_bf16` 签名上（已回退）。

详见 `docs/pitfalls.md` 同日条目。
