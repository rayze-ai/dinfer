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
