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

## G0 2026-10-29 Step A：副流 prefill 真实墙钟（`[late-begin]`→`[late-harvest]`，c16/32/64 混跑，二进制 4fb03abb）
| prefill tok | n | 墙钟 ms（中位） | tok/ms |
|---|---|---|---|
| ~300 | 8 | 71 | 4 |
| 1254 | 4 | 66 | 19 |
| 1900 | 2 | 72 | 26 |
| 2325 | 3 | 77 | 30 |
| 2569 | 15 | 92 | 28 |
| 3629 | 19 | **203** | **18** |
| 4040 | 14 | 206 | 20 |
- 口径注：墙钟 = 下发到收割的 host 时间，含 (a) 等下一个 decode 步 finish 的离散（≤ 1 步 8–11 ms）、(b) **低优先级下被 decode 图抢占的时间**。所以 18–30 tok/ms 是"被让位后"的有效吞吐，低于 D236 单流混批测的 37 tok/ms——这正是 E4 换 TPOT 付出的 prefill 时长（prefill 3600 tok 从 ~100 ms 拖到 200 ms，与 E5 的 120 → 177 一致）。
- 结论：**prefill GEMM 本体越快，让位期间的拖长也越短**（CTA 波少）；G1/G2 的价值比单流时更大。微基准仍以 P2148 为准（gate‖up 538 / down 410 us）。
- nsys 副流 kernel 表：nsysp16c 已抓（见下）。
- **nsysp16c 副流（stream 15，prefill）kernel 表**（最长步 kernel 和 164 ms / span 170，这是 nsys 态 + 被 decode 图让位后的一个大 prefill 步）：
  | 段 | ms | 占比 | 次数 × us |
  |---|---|---|---|
  | **fp8 cuBLASLt**（nvjet 128x128 37.1 + 64x64 19.7 + 64x32 17.0 + 128x64 16.3） | **90.1** | **55%** | 215×172 + 60×328 + 36×473 + 60×272 |
  | MoE p64 gate‖up + down | 44.5 | 27% | 60 × 742 |
  | norm 7.2 / attention 6.3 / quant 4.5 / moe_misc 4.4 / route 1.5 | ~24 | 15% | |
  - **大头变了**：D204/D219 时 MoE 45% / Lt 24%；现在 **Lt 55% / MoE 27%**。原因：Lt 自调优（B49）在 m=4160 桶选的 128x128 tile 在**让位（低优先级）** 下吃亏更重（大 tile 单 CTA 长、CTA 少，被 decode 图抢占后恢复慢），且 `64x32x64_4`（473 us × 36）/ `64x64x64_3`（328 × 60）这两个形状仍是小 tile——是哪两个 GEMM？按次数 36 = 可能 dense down（N=2816 K=2112，36 = 30 层 + 6?）、60 = qkv / o_proj。
  - **Step B 顺序改为**：**G3 先做**（fp8 Lt：对 m ≥ 64 桶把候选数 32 → 64 并把 timing 放在"有并发 decode 图"的真实环境测——现在 autotune 在空 GPU 上量，选出的 tile 在让位场景不一定最优；或直接对 4160 桶强制 128x128 + 看 splitk），**G1 ldmatrix 次之**（已写、g1 跑中），G2 BM128 再后。
- **nvjet 按 grid 归类（副流全程，m=4160 桶 autotune 选的 `128x128x64` 几乎全覆盖）**：grid (176,4) 145 us × 384 = 55.7 ms（N=2816 K=4096?）、(512,4) 292 × 162、(176,3) 113 × 300、(136,4) 89 × 372、(512,3) 216 × 125、(136,3) 78 × 300、(576,4) 327 × 42。**grid.y = 3/4 = m 块数**（M≈ 384–512 行一个 Lt 调用？不对：grid.y 3–4 个 128 行块 = m 384–512）——这些是**单请求 ~430 tok 的 prefill 子批**（late 路径一次只下发一个 prefill 子批、c16 下常是 1–3 个请求），m 不是 2148 而是 430–1290。m=430 时 128x128 tile 只有 4 行块 × 176 列块 = 704 CTA ≈ 2 波，被 decode 图抢占后每个 CTA 都要等；**对 m≈400–1300 这一档，128x128 不是好 tile**。
- → **G3 具体化**：fp8 Lt 自调优的 m 桶（2 的幂：512 / 1024 / 2048 / 4096）要**各自实测**（现在 m ≥ 64 的首个形状落在哪个桶就调哪个桶——late 子批 m 多为 430–1300 → 桶 512 / 1024 / 2048，日志里 D219 只见 4160 桶的调优是因为那是单流大混批时）；加 `GEMM_FP8_DEBUG=1` 看现在各桶选了什么；候选 32 → 64；并考虑在让位场景下重新计时（timing 时主流跑一个 decode 图作背景）。
