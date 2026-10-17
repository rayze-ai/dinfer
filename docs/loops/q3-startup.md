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
| 5 | 去 Vec 中转 + tokenizer 并行 | ① `load_bf16` BF16 快路径：mmap 切片直接 `htod_into`（不再 collect 成 Vec）→ host_convert 413 → **0 ms**，H2D 7–11 GiB/s；② tokenizer + 模板在独立线程加载，与 CUDA init / 权重并行，join 等待 39–63 ms。4090：**total 1.88 → 1.05–1.11 s**（cuda_init 260 / model_load 490–590 / engine 260–290）。baseline PASS | — | 剩余大项：cuda_init 260（Device::new + Gemm::new 的 cuBLASLt 句柄）、first_tensor 55–76、rope 122（host 算 40960×128 cos/sin）、mmap_touch 164–207（页缓存热时仍有，是 mmap 缺页 + 复制到 staging）、graphs 210 + xstream 副套。下一轮：rope 在 GPU 算；graph 捕获与权重并行；cuBLASLt 延后到首次 GEMM。目标 <0.8 s |
| 6 | 四项并行小改 | ① service 删掉没人用的 `Gemm::new`（executor 自建两个句柄）；② rope 表按位置分片多线程算；③ decode 图改按桶懒捕获（首次用到才捕获，`GRAPH_CAPTURE=eager` 恢复）；④ mmap `populate()` 一次填页表 | 对应分段：cuda_init ~130、rope 122、graphs 210、mmap_touch 160–200 | 4090 计时 ×2 + 首请求 TTFT（懒捕获代价）+ baseline |
| 6 结果 | 验收 | 4090：rope 122 → **21**；mmap_touch 164–207 → **3**（populate）；graphs 210 → 懒捕获（engine 258 → 45；首次用到桶 1 时 142 ms 含首次 GEMM 启发式，之后每桶 5–22 ms）；删 service 多余 Gemm 后 cuda_init 260 → 250（没省多少：大头是 Device::new）。**total lazy 719–730 ms、eager 933 ms**。首 3 请求 TTFT p50 4.2 / max 160（懒捕获代价一次性）。baseline PASS | 剩余：cuda_init ~250（Device::new = CUDA context + 模块加载）、model_load ~420（first_tensor ~60 + weights H2D 125 @ 11 GiB/s + layers 其他 ~230？）| 下一轮：① 看 model_load 内 420 的构成（weights 只 128，差 ~290）；② cuda_init 与 tokenizer 已并行，再与 safetensors open/populate 并行 |
