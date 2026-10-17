# Q3-m 采样 kernel 融合（loop 规格）

> 目标：decode 步的采样从 `copy → temperature → min_p → topk/topp → sample` 5 次 launch 变 `copy → fused` 2 次；语义逐位不变。
> 预期：图内少 3 个节点；每步 ~10–20 µs（launch + 尾部）；c1 TPOT 2.0 → ~1.98。小改动，主要是为后面 "逐层 megakernel" 练手。

## 设计
sampling.cu 的四个 kernel 体抽成 `row_temperature / row_min_p / row_topk_topp / row_sample` 四个 `__device__` 原语（独立 kernel 变成一行包装），融合 kernel 顺序调用四个原语（中间 `__syncthreads`）。原语共用保证语义相同。

## 停止条件
1. 新 ref `sample_fused_matches_chain`（24 行 × 4096 vocab，混合 greedy/随机/top-k/top-p/min-p）：选中 token 与 mask 后 logits **逐位**相同。
2. 原 4 个采样 ref + sampling_parity 5/5 + real_parity 3/3 不变。
3. 4090 baseline PASS；bench25 c1/c8 不退。
迭代上限 3 轮。

## 记录
| 轮 | 块 | 现象 | 根因 | 处理 |
|---|---|---|---|---|
| 1 | 实现 | 2026-10-17 原语抽取 + `sample_fused_f32` + executor 图路径接入 + ref | — | 3090 kernels.sh → parity → 4090 |
