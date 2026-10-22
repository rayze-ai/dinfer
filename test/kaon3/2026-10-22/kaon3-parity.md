# kaon3-parity 迭代记录（2026-10-22）

规格：`docs/loops/kaon3-parity.md`；脚本：`bash scripts/verify/kaon3-parity.sh`。指标 = 可判定行 top1 不一致数 / top5 |Δlp| 均值 / top1 |Δlp| 最大。

| 轮 | 改动（一句话） | bos1 | raw9 | chat_en | chat_zh | 结论 / 下一步 |
|---|---|---|---|---|---|---|
| 0 | 基线（RoPE span 已修，MAX_D=512） | 1 / 13.2 / 42.4 | 6/6 / 19.7 / 42.4 | 20/23 / 14.8 / 42.4 | 17/20 / 13.8 / 42.4 | row 0（T=1）就错 → 逐 token 路径 |
| 1 | MoE 支路：专家输入补 `pre_ff_norm_2`；相加用 `post_ff_norm_2` 之后的结果（原来加的是未归一化的 `sc.y`） | 0 / 1.23 / 0.009 | 2/6 / 2.42 / 11.4 | 10/23 / 4.75 / 34.2 | 6/20 / 2.55 / 6.06 | row 0 对上；多 token 行仍错 → attention / RoPE / KV |
| A/B | `GQA_PREFILL_CFG` 默认 / 0（v5）/ −1（新加：强制通用慢路径） | — | 2/6 · 2/6 · 2/6 | 10 · 10 · 11 | — | 三者几乎一致 → attention kernel 不是主因 |
| 自洽 | `dup4`（4 个相同 token：V 无 RoPE，各行应完全相同） | — | — | — | — | 我们 4 行完全一致；oracle 行间差 ≤0.02 → 多行路径无整体错位 |
| 2 | FP8 KV 模拟：K/V 写 cache / 做 attention 前按 e4m3（饱和、scale 1.0）往返；vLLM 实测 `kv_cache_dtype=fp8_e4m3`、无 k/v scale | 0 / 1.23 / 0.009 | 4/6 / 2.39 / 9.1 | 10/23 / 4.81 / 33.6 | 6/20 / 3.56 / 20.6 | 无改善（保留：是 vLLM 真实口径）；long_zh 222/1024 不一致 → 下一步截层二分 |
| 3 | FP8 静态激活 scale（`gemm_fp8_static_bf16`：x/input_scale 再转 e4m3，输出乘 a·w scale；实测 input_scale 0.03–3.9，>1 时原实现截断 448）+ 合并组（qkv / gate_up）按 vLLM `requantize_with_max_scale` 重量化 | 0 / 0.95 / 0.013 | 2/6 / 3.06 / 9.7 | 10/23 / 5.15 / 32.2 | — | 单 token 改善；多 token 不动 |
| 诊断 | `--dump-oracle`：我们 v5 对我们慢路径 | — | 1/9 / 1.12 / 1.88 | 6/21 / 2.46 / 20.1 | 4/19 / 3.27 / 23.0 | 同配置两次完全一致（确定）；两个正确 kernel 互差 20 nat → 模型对 attention 舍入级差异极敏感 |
| 4 | kernel 测试 `gqa_window_parity` 抓到滑窗 bug：tile0 按 warp 算（K/V 是 block 协作装的）、下界 mask 只在 need_mask 时做、v6 首 tile 固定装 0 → 三处修复，v5/v6/v6B/v6F/v7 全过 | | | | | 窗口生效时原误差 5.9 → 0.0087 |
