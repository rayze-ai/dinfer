# 2026-10-29 Kaon-V3 第四轮（loop kaon3-decode-opt D199–D234）归档
| 目录 | 链 | 内容 |
|---|---|---|
| stepA4/ | Step A v4 | 5 轮 c1–64 两端对标（基线，二进制 9046f6ef） |
| lpt/ | D206 | long_prefill_token_threshold 512/1024 零效果 |
| b49/ | B46 + B47 | lm_head 采样行 ✓ / 整步 prefill 配额 ✗ |
| b50/ b51/ b53/ | B48 / B48b / B48c | MoE prefill p64 BM64 / down BK64 / BN128 ✓ |
| b52/ | B49 | fp8 Lt 自调优 m ≥ 64 ✓ |
| stepA5/ | Step A v5 | 五刀后 3 轮对标 |
| b56/ | B50 | XSTREAM 单流混批 ≥ pair ✓ |
| b57/ | nsys | c16 纯 decode 步 kernel 表（M=16 profile） |
| b58/ | B51 | M=9..16 qkv/o 走 Lt 自调优 ✓ |
| b59/ | B52 | route 拆两段 ✓（c16 收益待 b61 复核） |
| b60/ | B53 | dense MLP M=9..16 走 Lt ✓ |
每轮判定与数字在 `docs/loops/kaon3-decode-opt.md`，CSV 在 `test/kaon3/e2e-rounds.csv`（round 列 = 目录名）。
| b61/ | B52 复测 | 顺序反转 split/fused：c32/c64 收益确认，c16 两轮 −3.5% / 0% → 判噪声级 |
| stepA6/ | Step A v6 | 九刀后 3 轮对标 + 各阶段耗时（D239） |
| b62/ | E1 prefill 异步收割 | 机制正确但无收益（decode 图被 prefill 副流抢 SM），默认关；下一步流优先级 |
| b63/ | E4 流优先级（+E1） | c16 −4.8% / c32 −4.9% / c64 −5.7%，TTFT c32 +50%（228 ≤ vLLM 231）；E1+E4 合入默认开 |
