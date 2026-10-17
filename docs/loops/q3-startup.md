# Q3-k 启动时间（loop 规格）

> 目标：Qwen3-0.6B 启动 total 2.1–3.0 s → <0.6 s（design-qwen3.md §6.3）。先定位 model_load 里 ~0.8 s 未归段。

## 环境
本机 3090：`RUST_LOG=info ./target/release/dinfer serve --model /data/ref/models/qwen3-0.6b --port 8099 --max-model-len 4096 --max-num-seqs 16`，看 `load timing: *` 与 `startup timing` 行。页缓存热（连跑两次取第二次）。

## 观察方法
`load timing` 分段：open / first tensor / rope / per-layer zeros / weights(mmap_touch, host_convert, h2d) / scratch / mem_info / kv / `wall after <阶段>` 累计墙钟。各段之和应 ≈ `model_load_ms`，差额就是未归段。

## 停止条件
1. 分段之和与 model_load_ms 差 < 50 ms（定位完成）。
2. 优化后 3090 total < 0.8 s、4090 < 0.6 s；`real_parity` / e2e 不变。
迭代上限 6 轮。

## 记录

| 轮 | 块 | 现象 | 根因 | 处理 |
|---|---|---|---|---|
| 1 | 分段 | 3090 热缓存：total 2163 = tokenizer 381 + cuda 99 + **model_load 1529** + engine 153。model_load 内已归：first_tensor 168 + rope 126 + zeros 10 + weights(touch 66 + convert 237 + h2d 105) + kv 32 ≈ **744**；未归 ≈ **785 ms** | 未归段不在 Tensor::zeros / load_bf16 / rope / kv 里 → 在 load_with_budget 的其他语句（Linear::new 校验？Rope 以外的 host 计算？），或 `tracing::info!` 自身 | 加 `wall after <阶段>` 累计墙钟（embed / layers / lm_head / rope / kv）二分 |
| 2 | 定位 | `wall after layers loop` = 1750 ms，而循环内已计时的 weights（touch 106 + convert 315 + h2d 141）+ zeros 15 + first_tensor 227 ≈ 800 → 差 ~950 ms 在循环里**未经 load_bf16 的路径**：`Attention::load` 把 q/k/v 三个权重用 `load_f32_host`（bf16 → f32 Vec）读出、host 上 concat、再 `Tensor::from_f32`（f32 → bf16 → H2D）。每层 3.1M×4 B 的 f32 往返 ≈ 30 ms × 28 层 ≈ 0.85 s，且不进 LOAD_STATS | 早期为"拼 qkv"图省事走 f32；统计只盯着 load_bf16 所以看不见 | 改为 bf16 直接上传 + 设备上 `memcpy_dtod` 拼接（零 host 计算）|
| 3 | 定位 | qkv 改设备拼接后 4090 total 2.93 s 几乎不变；per-layer 分段（28 层求和）：qkv_load 70 / qkv_concat 3 / o_proj 31 / qk_norm 0 / attn_zeros 7 / **mlp 1234** / layer_total 1350 ms | 大头在 `Mlp::load`（layers/nn.rs），不在 Attention；它也不进 load_bf16 的统计 → 同样是走了 f32 host 路径（gate/up 拼接）| 看 Mlp::load 源码，改 bf16 设备拼接 |
| 4 | 修复 | `Mlp::load` gate/up 同样走 f32 host 往返 → 改 bf16 上传 + 设备 `memcpy_dtod` 拼接。4090：**total 2.93 → 1.88 s**（model_load 2.04 → 1.00 s；mlp 1234 → 171 ms）。baseline PASS（refs 22 / parity 8 / 单测 8 / e2e 6）。现分段：tokenizer 416 / cuda_init 239 / model_load 999（first_tensor 247 + rope 120 + layers 289 + weights touch 185 + convert 413 + h2d 187，weights 与 layers 有重叠）/ graphs 213 | 两处 f32 host 拼接合计 ~1 s，是启动的最大项，之前统计只盯 load_bf16 所以不可见 | 下一轮：① tokenizer 416 ms 与 CUDA init / 权重加载并行（独立线程）；② 权重去 Vec 中转：bf16 张量直接 `cudaMemcpy` mmap 指针（省 convert 413 + touch 一部分）；③ rope 表 GPU 算（120 → ~1）；④ graph 捕获与权重加载并行。目标 <0.8 s |
