# kaon3-parity 迭代记录（2026-10-22）

规格：`docs/loops/kaon3-parity.md`；脚本：`bash scripts/verify/kaon3-parity.sh`。指标 = 可判定行 top1 不一致数 / top5 |Δlp| 均值 / top1 |Δlp| 最大。

| 轮 | 改动（一句话） | bos1 | raw9 | chat_en | chat_zh | 结论 / 下一步 |
|---|---|---|---|---|---|---|
| 0 | 基线（RoPE span 已修，MAX_D=512） | 1 / 13.2 / 42.4 | 6/6 / 19.7 / 42.4 | 20/23 / 14.8 / 42.4 | 17/20 / 13.8 / 42.4 | row 0（T=1）就错 → 逐 token 路径 |
| 1 | MoE 支路：专家输入补 `pre_ff_norm_2`；相加用 `post_ff_norm_2` 之后的结果（原来加的是未归一化的 `sc.y`） | 0 / 1.23 / 0.009 | 2/6 / 2.42 / 11.4 | 10/23 / 4.75 / 34.2 | 6/20 / 2.55 / 6.06 | row 0 对上；多 token 行仍错 → attention / RoPE / KV |
