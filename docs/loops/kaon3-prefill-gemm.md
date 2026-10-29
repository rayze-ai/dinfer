# loop: kaon3-prefill-gemm —— prefill 单步吞吐 37 → 50 tok/ms（MoE p64 再 1.5–2×）

> 来源：`kaon3-prefill-async` E6 收口（2026-10-29）。十一刀后 c16 1.25× / c32 1.29× / c64 1.19×；剩余差额 = 混批步插入
> （c16 1.2 / c32 2.7 / c64 7.3 ms/步）——现在 prefill 在低优先级副流上跑、decode 图能插进去，但 prefill 每 CTA 波仍挡 decode；
> 缩短 prefill 本身（37 → 50 tok/ms，vLLM 口径）是剩下唯一的大项。D219 nsys：MoE p64 占混批步 45%、57–248 TFLOPS（峰值 1/4）。

## 环境
5090 容器 `kaon3-full` GPU0；十一刀默认二进制（含 E1/E4）；`envab_c.sh` 口径 c8–c64 两轮；微基准 `moe_gemm_timing`（P424 / P2148 形状）两进程 CHECKSUM 门禁。

## Step A（本 loop 的测）
1. 副流 prefill 真实时长：`HOST_TIME=1` 下 `[late-begin] prefill_tokens=… ` → `[late-harvest] tokens=… wall_ms=…`（已在 runner，待编入），按 tokens 桶给 tok/ms。
2. nsys 副流 kernel 表（`kaon3_nsys_prefill16.sh` 口径，按 streamId 区分主/副流），p64 gate‖up / down 每层 us、占比。
3. 微基准 P2148：gate‖up 538 / down 410 us（b53 BN128）= 248 / 164 TFLOPS 基线。

## Step B 候选（按 ROI）
- **G1 ldmatrix**：p64 的 A/B 片段现在逐 32-bit 从 smem 读（每 mma 2+2 次 LDS），换 `ldmatrix.x4`（A 一次 4 reg、B x2）→ smem 指令数 −60%。k 序不变 → 逐位同。
- **G2 BM=128**：每 warp 2 个 m 片段（32 行）× NJ，A 片段再翻倍复用 B；smem 3 stage ≈ 66 KB → 1 CTA/SM，要配 STAGES 2 或 BK 64。可能与 G1 叠加。
- **G3 fp8 Lt 自调优候选数 32 → 64**（`lt_candidates`）：D219 Lt 占混批步 24%，128x128 tile 已选，看更大 tile / split-K 组合。
- **G4 attention prefill fa_v5 98 us × 25**：D227 估离下限 2.4×；中刀，排后。

## 停止条件
① P2148 微基准 gate‖up ≤ 400 us 且 down ≤ 300 us（各 −25%），CHECKSUM 两进程同；② 端到端 c16 TPOT ≤ 8.5（1.19×）、c32 ≤ 11.0（1.19×）、c64 不升；TTFT 不升 > 3%；③ 每刀进 CSV（含负结果）。连续 3 刀无进展回 Step A。

## 记录
G 编号（接 D/E）。
