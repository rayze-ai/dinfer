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

## G1 2026-10-29 第一版编译失败：`(cond ? kernel<…,true> : kernel<…,false>)<<<…>>>(args)` 三目取到的是函数指针，kernel 的默认实参（Bu/Su/alphaU/Du）随之丢失 → "too few arguments"。改成变参模板 `p64_launch<BK,ST,BN>(grid, stream, args...)` 内部两分支各自 `<<<>>>`。本机 nvcc（sm86）过，重传。
- 校正：nsysp16b 日志里 m=428 与 m=432 都调了一遍——那枚二进制（85b85d5a，14:5x）在 2 的幂桶 key（09d077a7，14:58）之前；现在的二进制每桶只调一次。m=428/432 桶的选择：n=2112/2816 → `tile=15/18`（64x32 / 64x64），n=8192/9216 → `tile=20`（128x128）；m=4096/4160 全 tile=20。即 **512 桶的 qkv（n=8192/9216）已选 128x128**，副流 nsys 看到的 (176,4) / (512,4) 大 tile 就是它们——在空 GPU 上 40 us 最快，但在被 decode 图抢占的副流上 145 us（3.6×）。
- **G3a**：自调优的计时环境要像生产：在 `tune` 时让主流同时跑一个 decode 图（或用 `cudaStreamCreateWithPriority` 低优先级流计时 + 主流 busy kernel）。更简单的替代：对 m ≥ 64 桶把"最快"换成"最快且 CTA 数 ≥ 2 × SM 数"（小 tile 多 CTA 在抢占下恢复快）——先用 env `FP8_LT_PREFER_SMALL_TILE=1` 过滤掉 tile=20 看端到端。

## G3a 2026-10-29 设计（b60 B 侧 debug 日志证据：桶 64 → tile 15、183 → 18、**426 / 637 / 1284 / 3854 / 4160 → tile 20（128x128）**，n=8192）
- 空 GPU 计时下 m ≥ 426 都选 128x128；副流抢占环境下这个 tile CTA 少（m=430：4 行块 × 64 列块 = 256 CTA，1.5 波）、每 CTA 长（145 us 实测 vs 空 GPU 40 us），被 decode 图抢占后尾部等待大。
- **实现**：`FP8_LT_PREFER_SMALL_TILE=1` → tune 候选过滤：m 桶 ∈ [64, 2048] 时排除 `CUBLASLT_ALGO_CONFIG_TILE_ID == 20` 的候选（128x128），剩下的照常实测取最快（通常落 tile 18 = 64x64 或 15 = 64x32，CTA 数 ×4–8）。≥ 4096 桶不动（单流大混批仍以吞吐为先）。默认关，g2 链 A/B；赢了再考虑把"抢占环境计时"做进 autotune。

## G1 2026-10-29 微基准（LDSM 0/1，CHECKSUM 全同 GATE=0，二进制 599a54fa）：P2148 gate‖up 500.5 → **466.4（−7%）**，down 371.8 → 372.2（0%，BK=64 路径每 BK 块只 1 次 mma 组、smem 读本来就少）；P424 持平。smem 指令减半只换来 7% → p64 的瓶颈不是 LDS 发射，是 **mma 发射密度 / 每 CTA 流水深度**（STAGES 3、每 warp 8 个 n 片段串行等 A/B 到位）。ABAB 跑中；预期端到端 < 1%，G1 作为"顺手合入"看两轮方向。
- **G1 定稿（g1 两轮 ABAB，A=LDSM 0 / B=1）**：c8 7.28 vs 7.37、c16 8.98 vs 9.08、c32 12.03 vs 11.85、c64 18.52 vs 18.40 —— 全在 ±1.5% 噪声内，无同向。微基准 −7%（gate‖up）端到端摊到 ≈ 0.1 ms/步。**默认留关**，代码保留（`MOE_P64_LDSM=1`）；CSV g1 记中性。
- 教训（进 knowledge）：p64 的 smem 读不是瓶颈，mma 发射密度 / 流水深度才是；再往下要么 BM128（每 warp 2 个 m 片段把 B 复用翻倍）要么 TMA + 更深 stage——但 MoE 只占副流 27%，**G3a（Lt 55%）先**。
- g2 启动器按 g1 c64 B<A（18.36 vs 18.42，噪声内）自动给两侧加了 `MOE_P64_LDSM=1`；A/B 两侧同配置，G3a 判定不受影响，只是 g2 的绝对值含 LDSM。
- 日志里同桶出现两条（m=427 与 433 都调）：不是桶 key 失效，是**主流与副流各有一个 `Gemm` 实例**（各自 `default_algos` 缓存、各自调一遍、各自 32 × 5 次计时）。候选 **G3b**：把 fp8 Lt 的 algo 缓存做成跨实例共享（`Arc<Mutex<HashMap>>` 由 `Gemm::new` 克隆入），省一半首批请求的调优抖动；或在 autotune 阶段预调 m 桶 {64..4096} × 6 形状。

## G3a 2026-10-29 g2 第 1 轮（A = 默认 / B = `FP8_LT_PREFER_SMALL_TILE=1`，两侧 LDSM=1）
- 过滤生效：B 侧 426/429/850 桶 → `tile=18`（64x64，空 GPU 计时 52.0 us vs A 的 tile=20 39.7 us）。
- 端到端 **无差**：c8 7.29 / 7.29、c16 8.97 / 9.11、c32 11.92 / 11.84、c64 18.29 / 18.38；TTFT 同。
- 解释待定：① 128x128 在抢占下 145 us 的读数来自 nsys（膨胀 + 只抓了一次），小 tile 可能同样被拖；② qkv/o 的 Lt 只占副流 55% 里的一部分，n=2816 K=2112/4096 的形状本来就选 tile 15/18。等 r2；若仍无差，判负默认留关，并改用 nsys A/B 直接比两侧副流上同形状 nvjet 的实测时长（g3 链：nsysp16 口径跑 A/B 各一次，按 grid 归类对比）。

## G0 补：抢占把副流 prefill 的每个 kernel 拉长多少（已有数据对比：b54 = 单流混批无抢占；nsysp16c = E6 副流被 decode 图抢占）
| kernel（同形状） | b54 单流 us | p16c 副流 us | 拉伸 |
|---|---|---|---|
| p64 gate‖up（grid 6×421×2） | 474 | 742 | **1.57×** |
| nvjet 128x128 (176,3) | 100 | 113 | 1.13× |
| nvjet 128x128 (512,3) | 177 | 216 | 1.22× |
| nvjet 128x128 (136,3) | 46 | 78 | 1.70× |
| gqa_fa_v5 | 106 | 105 | 1.0× |
| 副流 kernel 和（一个 ~2148 tok prefill） | ~58（HOST_TIME 混批步） | 164 | **2.8×** |
- 单 kernel 拉伸 1.1–1.7×，但**整步 2.8×**：差额是 kernel 之间的空隙——副流 kernel 排队等 SM 时主流 decode 图在跑，nsys 看到的是副流 kernel 的 start→end（含被抢占等待）加上 kernel 间 gap。这就是 E6 的代价结构：prefill 让出的不是"每个 kernel 慢一点"，是"整条流被插队"。
- **推论**：改 prefill kernel 的 tile（G1/G3a）只动 1.1–1.7× 里的那部分，对 2.8× 的整体影响小，与 g1/g2 端到端无差一致。要缩短副流 prefill 墙钟，杠杆是 **(a) 减少 prefill 的 kernel 数 / launch 数**（gap 多来自 launch 粒度：每层 ~30 个小 kernel 各排一次队），**(b) 减少 FLOPs**（没有），**(c) 让 decode 图别那么密地插**（反向：会还回 TPOT）。(a) 可做：prefill 路径的 norm / quant / route / gelu 小 kernel 融合（D219 表里 norm 7.2 + quant 4.5 + moe_misc 4.4 + route 1.5 = 18 ms 的"小 kernel"在副流上每个都要排队）。
- 停止条件 ① 的"微基准 −25%"对端到端无意义（g1 微基准 −7% → 端到端 0），**改判据**：以 `[late-harvest] wall_ms` 的 tok/ms（G0：2.5K tok 档 28 → 目标 ≥ 36）与 c16 TPOT 为准。

## G4（新候选，取代原 attention 项）：prefill 路径减 launch 数
- 依据：副流上每个 kernel 都要在 decode 图的 CTA 间排队，整步 2.8× 拉伸主要是 gap；一个 2148 tok prefill 步 ~1175 个 kernel（nsys 片段）≈ 39 个/层。目标：把每层 prefill 的小 kernel（norm → quant 已融；attn_pre_fused 已融；route_topk + softmax、combine_norm、rmsnorm_vec ×2、quant_fp8 ×2、gelu_grouped_quant、rmsnorm_grouped_quant）再合，每层 ≤ 25 个 launch。
- 先列清单（stream 15 kernel 名 × 次数），再定融合对象。
- **stream 15 清单（2148 tok prefill 步，1175 kernel ≈ 39/层）**：GEMM 类 82%（nvjet 215+60+36+60 = 371 次 90 ms；p64 60 次 44.5 ms）；小 kernel 18%（combine_norm 30×144 us = 4.3、rmsnorm_vec 124×26 = 3.2、quant_fp8 124×24 = 3.0、rmsnorm_grouped_quant 30×95 = 2.8、attn_pre_fused 2.1、Kernel2（bf16 router GEMM）31×50 = 1.6、gelu_grouped_quant 1.5、gelu_tanh_mul 1.0、route_topk 0.9、rmsnorm_scale2 0.7、route_softmax 0.6、rmsnorm_quant 0.5 …）。
- **修正 G0 补的结论**：这个 nsys 片段 kernel 和 163.8 / span 170 → **gap 只 6 ms（4%）**，不是 2.8× 拉伸的来源；2.8× 的差是**每个 kernel 自身变慢**（p64 474 → 742、nvjet 46 → 78 等，被抢占时 CTA 在 SM 上与 decode 图共驻 → 慢）+ 这个片段本身 tokens 不同（b54 的 58 ms 是 HOST_TIME 混批步 2148 tok；p16c 的 164 ms 片段含 371 次 nvjet ≈ 多个请求的 prefill 合并）。两个口径不可直接比，收回 "gap 为主" 的判断。**G4（减 launch）收益上限 = 小 kernel 18% × 融合比例 ≈ 5–8%**，排在 GEMM 之后。
- **GEMM 82% 里**：nvjet 90 ms（Lt，各 tile 已自调优）+ p64 44.5 ms。Lt 这头 G3a 换 tile 端到端无差（r1）；真正能动的是 **FLOPs 本身**——qkv n=8192/9216 K=2816 与 o_proj K=4096/8192（full 层 D=512 ×16 头 = 8192）是 Kaon-V3 的模型形状，vLLM 同样要算。所以 prefill 单步 37 vs 50 tok/ms 的差距可能主要来自 **vLLM 的 fp8 GEMM 走 cutlass/FlashInfer 更快的 kernel**，而不是我们的调度。下一步：**用 nsys 抓 vLLM 的 prefill kernel 表**（GPU2 基线容器 :28201 不能动；在 GPU3 起一个 nsys 包着的 vLLM 实例跑 c16 同 bench），比同形状 fp8 GEMM 的 us。这是 Step A 的缺口。
- **G3a 定稿（g2 两轮）**：无差（c16 8.97/9.10 vs 9.11/9.01、c32 11.92/11.78 vs 11.84/11.89、c64 18.29/18.52 vs 18.38/18.38；TTFT 同）。默认留关，旋钮保留。CSV g2 中性。**Lt tile 不是杠杆**。
- 连续两刀（G1、G3a）中性 → 按停止条件回 Step A：缺的是 **vLLM 同形状 prefill kernel 表**（它 50 tok/ms 我们 37 的差在哪个 kernel）。g3（我们自己的 A/B nsys）先跑完给数据，再在 GPU3 起 nsys 包的 vLLM。

## Step A 补（2026-10-29）：vLLM 同口径 kernel 表
- 容器 kaon3-full 自带 vllm 0.26.1rc1（与基线镜像同版）+ nsys；在 **GPU3**（空）起 `vllm serve` 同基线参数、c16 × 128 请求同 bench，`--cuda-graph-trace=node`。产出：decode 图 kernel 表（`nsys_kern.py --all`）+ eager prefill 步 kernel 表（`nsys_prefill_steps.py`）。与我们的 b57（decode）/ nsysp16c（prefill 副流）对照：同形状 fp8 GEMM 各走什么 kernel、多少 us；MoE NVFP4 走 cutlass/flashinfer 哪个、每层 us。
- 这是本 loop 真正的 Step A 缺口：之前所有判断都只看我们自己的表。
- g3 第一跑：链脚本 `$((… [ ] …))` 语法错，两侧没跑（pitfalls 已记），修后重排。
- vLLM-in-kaon3-full 第一跑：`Triton is installed but 0 active driver(s) found` → 模型注册 `Gemma4ForConditionalGeneration` 子进程失败。kaon3-full 容器的 vllm 0.26 环境缺 Triton 可用的驱动接口（容器是 pegaflow 镜像衍生，自带 vllm 但没按 vLLM 方式起过）。**改法**：用基线镜像 `image.paigpu.com/library/vllm-openai:v0.28.0-pegaflow-pr455-51261c6-cu130` 在 GPU3 起一个新容器（同 vllm-noneager 的参数 + `-v /hf_data/yaochen:/w`），把 kaon3-full 里的 nsys 目录 bind 进去（`/opt/nvidia/nsight-systems/2026.3.2`），容器内 `nsys profile vllm serve …`。需要宿主 docker 权限（有）。
- Step A 补 v2（`kaon3_vllm_nsys2.sh`，宿主）：基线镜像临时容器 `vllm-nsys` @ GPU3，nsys 目录从 kaon3-full `docker cp` 到 `/hf_data/yaochen/nsys` 再 bind；nsys 自带 libcupti，目标镜像不需装。与 g3（GPU0）并行。
- g3 脚本的 "alt stream = nvjet 最多的流" 选错（选到了 decode 图所在的主流 20：图内 nvjet 小 tile 每步都有）；正确的副流是 **stream 15**（eager、16302 kernel）。用 graphId==0 过滤重算 A/B。
- vllm_nsys2 第一跑：`--gpus device=3` 撞上 vllm-eager 基线（GPU3 占 28 GB），`Free memory 2.89/31.36 GiB` 起不来。**低级错误**：launch 前没看 `nvidia-smi` 的 GPU3 占用（之前 D199 环境事实里写的是 GPU2 基线，GPU3 是 eager 基线也写了）。改 GPU4 重跑。

## G3a 机理核（g3，nsys A/B 副流 eager nvjet 按 grid；二进制 42dfba32，LDSM=1 两侧）
| grid（m 块, n 块） | A 128x128 us | B（过滤后应换 tile） |
|---|---|---|
| (176,4) qkv n=8192 m≈430 | 143.5 × 384 | **143.8 × 324（仍 128x128）** |
| (512,4) o_proj k=8192 | 290.6 × 162 | 291.5 × 137（仍） |
| (136,4) / (176,3) / (512,3) / (136,3) | 89 / 110 / 218 / 78 | 89 / 110 / 229 / 80（仍） |
- **B 侧副流上的大 GEMM 没换 tile**。原因：这些 m≈430–512 的调用的 m 桶是 512（在 [64, 2048] 内）——但过滤只在 **tune 时**生效，而**副流的 `Gemm` 实例**是在 `AltSet::new` 里独立建的，它的 autotune / 首次 tune 发生在……两边各调一次（D"同桶两条"），应该都走了过滤。再看：B 的 (176,4) 仍是 `128x128x64_6`——说明 **tile=20 不是 128x128**；`algo_desc_full` 的 `tile=20` 与 nvjet 名字里的 `128x128` 的对应是我从 D215 推的，没验证。B 侧确实多了一类 `128x64x64_8`（n=211，45.9 us，grid (512,1)）——那才是被过滤后换上的（原 A 对应项 `64x64x64_12` 20.7 us → 变慢了 2×）。即：**过滤排除的 tile=20 实际是 64x64 那一档，把 m 桶 512 的 n=2816 小 GEMM 推去了更差的 128x64**；而真正的 128x128（nvjet `_6_`）对应别的 tile id。G3a 的实现靶子就错了。
- 另外 eager 总时长 A 2544 vs B 2436 ms（nvjet 只 16%，p64 21%，**other 62%**）——副流上大头是 "other"，下条查。
- 结论：G3a 判负不是"tile 不是杠杆"，是实现错；但要重做得先把 TILE_ID ↔ 实际 tile 名的映射用 `GEMM_FP8_DEBUG` 的 cands 列表对上。鉴于 nvjet 在副流只占 16%，**不重做**，记入 pitfalls（没验证的映射别当事实用）。

## ★★★★★ G5 2026-10-29 发现：副流（prefill）上 lm_head **仍按全部 tokens 行算**（nsys g3 A，stream 15 稳态：`cutlass Kernel2 grid (8,2048)` 912 us × 131 次，从 0.1 s 到 5.9 s 持续）
- grid.y = 2048 = 2048 行 / 1 → M ≈ 2148 tok（prefill 子批整段），N=262144 词表 bf16 GEMM 912 us；B46 的"只算采样行"在**late 路径没生效**——每个 2148 tok prefill 步白算 ~0.9 ms + softcap；按 c16 26 个 prefill 步/窗口 = 24 ms，占副流 eager 稳态时长的 ~20%（p64 两项 ~140 ms / 26 步 = 5.4 ms/步 vs lm_head 0.9 ms/步 —— 占 prefill 步 kernel 和的 ~12%）。
- 待查：B46 在 executor `forward_device_mode` 里按 `nrows < t && lm_head_rows_only()` 传 `sample_rows`；late 路径走 `enqueue(plan_p, false)` → 同一函数 → 条件应成立。可能 (a) `nrows`（`plan.logits_rows.len()`）在 prefill 子 plan 里等于 t？不会；(b) `hidden_dir.is_some()`？否；(c) **副流的 `LayerScratch.lm_in` 未分配 / `swap_alt` 后 `self.scratch` 是 alt 的**——`ensure!(nrows <= sc.lm_in.numel() / h)` 失败会报错而非回退…；(d) 最可能：`gather_rows_bf16` 走了但 GEMM 的 `lm_rows` 仍是 tokens——看 model.rs 分支。直接加 `eprintln!` 到副流路径核。
- 这是"改了执行结构后探针口径变了"的又一例：D205 的 B46 用单流 nsys 核过，E6 换副流后没人再看副流上的 lm_head。
- **G5 撤回**（自我纠错）：`Kernel2 grid (8,2048)` 131 次里 90 次前一个 kernel 也是它自己、gap 0.9 ms 连发——这是 **alt `Gemm` 的 autotune 候选扫描**（m=2048 桶 × 候选，启动期 0.1–5.9 s 整段都是 warmup / 自调优时间窗，g3 的 nsys 只跑 ~6 s）。B46 在 late 路径**没问题**。
- 由此 g3 的 "eager 口径" 几乎全是启动期（autotune + warmup prefill 4096 tok），**稳态的副流 prefill 样本太少**——nsysp16 口径 c16 × 128 请求只 ~6 s，自调优就占一半。要看稳态副流表，nsys 抓取要么跳过前 10 s（`--capture-range` / `--delay`），要么把 bench 拉到 512 请求。之前 D219/G0 的副流表同样含启动期，**p64 27% / nvjet 55% 的比例不可信**。
- **口径修正进 Step A**：nsysp16 加 `--delay 15`（跳过 autotune / warmup），请求数 ×4。

## 中途小结（2026-10-29，两份稳态 nsys 抓取中）
- 已定：G1 ldmatrix 中性（默认关）、G3a 实现错 + nvjet 在副流非大头（默认关）、G5 撤回（autotune 探针）。**本 loop 至今 0 刀合入**，连续 3 刀中性 → 按停止条件回 Step A。
- Step A 缺口两项并行补：① 我们副流稳态 kernel 表（`nsysp16s`，delay 20 s，32×16 请求）；② vLLM 同口径 kernel 表（`vllm-nsys` @GPU4）。对比脚本 `nsys_family_compare.py`。
- 判据不变：`[late-harvest] wall_ms` tok/ms（G0：2.5K 档 28 → ≥ 36）与 c16 TPOT ≤ 8.5。
- vLLM 在 nsys 下的 c16 bench（GPU4，同 bench 参数）：TPOT p50 **8.08 / 8.13**（无 nsys 基线 :28201 是 7.1–7.2）、TTFT 261、1320 tok/s；我们在 nsys 下（nsysp16s）TPOT ~8.9、1419 tok/s。**nsys 态两边都膨胀**（vLLM +13%，我们 +0 ~ −1%——我们 decode 走图、nsys 节点级开销小；vLLM 的 FULL cudagraph 同理但 prefill eager 多）。kernel 表只比相对占比与同形状 us，不比绝对 TPOT。

## ★★★★ G6 2026-10-29 我们的**稳态**副流表（nsysp16s：delay 20 s、c16 × 512 请求、26 s 窗口、eager kernel 7944 ms = 窗口 30%；host-time 混批步 98 × 2254 tok 47 ms）
| 家族 | ms | 占比 | 主 kernel（次 × us） |
|---|---|---|---|
| **MoE p64** | 3053 | **38%** | 5902 × 516 |
| **fp8 Lt** | 2398 | **30%** | nvjet 128x128 14925 × 96；Kernel2 3550 × 155（bf16 router GEMM）；small_m_mma 505 × 1039（**alt Gemm 的 lm_head m=16..64 桶自调优残留？delay 20 s 后仍有 505 次——待查，可能是 late 收割步 lm_head 走 small_m 路径）|
| other | 869 | 11% | |
| norm | 713 | 9% | rmsnorm_grouped_quant 3161 × 66、rmsnorm_vec 12631 × 15、combine_norm 3161 × 76 |
| attention | 627 | 8% | fa_v5 2460 × 111、fa_v6 492 × 414 |
| route / quant / elem | 284 | 4% | |
- 每个 prefill 步（98 步）：eager 和 81 ms / 步 ≈ 2254 tok → **28 tok/ms（nsys 态）**，host-time 47 ms（主流视角，不含副流）；G0 的 late 墙钟 28–30 tok/ms 一致。
- 与启动期污染的 D219/G0 表比：p64 从 27% 回到 38%、Lt 55% → 30%。**MoE p64 是稳态第一项**（3053 ms / 98 步 = 31 ms/步，占 prefill 步 38%）。
- 等 vLLM 表（export 中）对比同家族。
- G6 校正：`small_m_mma 1039 us × 505` 全在 0.1–1.6 s（autotune 残留，delay 20 s 没盖住——nsys `--delay` 从进程起算，模型加载 35 s 后 autotune 才跑；下次 delay 60）；`Kernel2 (8,2048)` 分三路：s20 主流 2006 次 gap 8.4 ms = decode 步 lm_head（图内）；**s15 副流 152 次 897 us gap 0.9 ms = late prefill 子批的 lm_head**（B46 后只算采样行 → grid.y=2048 不是行数而是词表 262144/128 = 2048 列块；m 小 → 这是 cuBLAS 对 m=1..16 的 wmma 选择，897 us 与 decode 同）；s14 启动期。即每个 late prefill 步付一次 ~0.9 ms lm_head（词表 bf16 GEMM 的带宽下限 0.83 ms），**无可省**。
- vLLM sqlite 落盘（53 MB rep / 143 MB sqlite，275 s 全程）。整程 eager 表被启动期污染（`delayStreamKernel` 54% = vLLM cudagraph 捕获 / warmup 的占位 kernel；`device_kernel` 是 CUTLASS 统一入口名，要看 demangled）。按 kernel 密度切出 bench 窗口再比（`vprobe.py`）。

## ★★★★ G7 2026-10-29 vLLM nsys（GPU4 基线镜像，c16 × 128 请求，bench 窗口 264 s 含 warmup / 捕获）
- eager（prefill）侧 18.9 s（去掉 `delayStreamKernel` 10.2 s = cudagraph 捕获占位）≈ 8.7 s：**CUTLASS `device_kernel<GemmUniversal<cutlass…>>` 3.11 s（19800 × 157 us）= MoE NVFP4 grouped GEMM**（vLLM 走 CUTLASS grouped，不是 FlashInfer）；`vectorized_elementwise` 1.17 s（38241 × 30 us）；nvjet 128x128 / 64x64 / 128x64 / 64x32 共 1.6 s（fp8 dense：qkv/o/dense MLP 与我们同 cuBLASLt 家族）；`Kernel2` bf16 wmma 0.5 + 0.13 s（router / lm_head）；`triton_` 0.42 s；`finalizeMoeRouting` 0.19；`doActivation` 0.18；`kernel_unified_attention` 0.11 s（**prefill attention 只 2340 × 48 us**——我们 fa_v5 2310 × 110 + fa_v6 462 × 390 = 0.43 s，**我们 attention 慢 4×**）；另一个 CUTLASS `GemmUniversal<cute::…>` 0.26 s（2280 × 115）。
- graph（decode）侧 4.08 s / ~20100 步 = **0.2 ms/步 kernel 和**？与 TPOT 8 ms 不符 → vLLM 用 PIECEWISE：图里只有 GEMM/MoE 片段，attention 等走 eager（上面的 `kernel_unified_attention` 2340 次在 eager 里但 decode 步应有 ~20000 次 → 不对）。**graph kernel 数 503530 / 25 kernel 每步 = 20100 步，而 eager attention 只 2340**——说明 decode 的 attention 也在图里但名字不同（`kernel_unified_attention` 19680 × 24.6 us 在 graph 表第三行 ✓）。那 decode 步 kernel 和 0.2 ms 还是太小：**节点级 nsys 对 vLLM 的 FULL 图可能只记录了部分节点**（CUTLASS device_kernel 40200 × 45.6 us = 2 次/步 × 30 层？不：2 次/步 → 它的 MoE 是**每步 2 个 kernel 覆盖全部 30 层？** 不可能）。→ vLLM 的 decode 图统计不可信（CUPTI 对其 PIECEWISE+FULL 混合图的节点计数口径不同），**只比 prefill eager 侧**。
- **prefill 侧可比项（同 bench、同 tok 数量级）**：
  | 家族 | vLLM（bench 窗口总） | 我们（nsysp16s 26 s 窗口总，98 步 × 2254 tok） | 注 |
  |---|---|---|---|
  | MoE grouped GEMM | 3.11 s / 19800 次 = 157 us | 2.93 s / 5542 次 = 529 us（p64 gate‖up+down） | vLLM 每层 2–3 个 CUTLASS kernel 各 157 us；我们每层 2 个 529 us → **我们每层 MoE GEMM ≈ 1.06 ms vs vLLM ≈ 0.4 ms（2.7×）** |
  | fp8 dense（nvjet） | 1.6 s | 1.36 s | 同家族同 tile（128x128 为主） |
  | attention | 0.11 s（48 us/次） | 0.43 s（fa_v5 110 + fa_v6 390） | **我们 4×** |
  | elementwise / norm / quant 等 | 1.17 + 0.42 + 0.37 ≈ 2.0 s | norm 0.68 + quant 0.23 + combine 0.23 + other 0.47 ≈ 1.6 s | 持平 |
  - 两边窗口不同（vLLM 264 s 含大量空闲；我们 26 s 稳态），**总量不能直接比**，只比 **单次 kernel us 与每层次数**。
- **结论（Step A 定靶）**：① **MoE prefill GEMM 每层 1.06 ms vs 0.4 ms**——CUTLASS grouped NVFP4 在 M≈2148×8 行上是我们 p64 的 2.7×；这是本 loop 真正的刀：要么把 p64 做到 BM128 + TMA + 多 stage（CUTLASS 水平），要么**直接接 CUTLASS grouped GEMM**（vLLM 用的就是它，`采用`）；② prefill attention 4×（fa_v5/v6 vs `kernel_unified_attention`）：第二刀。
- **G7 读数修正**：按"≥600 kernel 的最长步"挑出的两个 vLLM 步分别是 **视觉塔 warmup**（`fmha_cutlassF_bf16`、`relu_bf16_256x128` CUTLASS、grid (632,3)——不是 Kaon 文本层）和 **启动期 elementwise/triton 风暴**；都不是文本 prefill。G7 表里 "CUTLASS device_kernel 19800 × 157 us = MoE" 的归因也要重核——改用 **gridY 代理（GEMM 行块数 14..26 ≈ 1800–3300 行）筛文本 prefill 步**再聚合（`vstep2.py`）。前面写的 "MoE 2.7× / attention 4×" 暂作疑点，等 vstep2。
- **vLLM 基线实际后端（启动日志）**：MoE = `FLASHINFER_CUTLASS` NvFp4（kernel 名应为 `cutlass::device_kernel<…GemmUniversal…>`，即 G7 里的 CUTLASS 行）；dense fp8 = `FlashInferFP8ScaledMMLinearKernel`（底层仍是 cuBLASLt nvjet / `sm89_xmma_gemm_e4m3`）；attention = **`TRITON_ATTN`**（`kernel_unified_attention` = Triton）。所以可比的"同功能 kernel"：MoE ↔ CUTLASS device_kernel；attention ↔ kernel_unified_attention；fp8 dense ↔ nvjet。
- vstep2 的 gridY 14..26 筛出 0 步：vLLM chunked prefill 每步 ≤ 4096 tok 但**一波 16 个请求的 prefill 被切成多步、每步 1–4 个请求（iteration_tokens_mean 97）**，所以 vLLM 的 prefill 步 gridY 代理多在 4–8（430–860 行），不是 2148。要按它自己的分布比。
- vstep3：vLLM 的 MoE kernel `cutlass::device_kernel<GemmUniversal<GroupProblemShape…>>` grid **(1,170,1) = 170 SM 一块一个 → persistent grouped GEMM**，与行数无关，gridY 代理失效；按 nvjet（dense fp8）的 gridY × 128 估行数重分桶（`vstep4.py` / `ostep4.py`），两边同桶比每步 MoE / fp8 / attention 的 ms。vLLM 每步 MoE ≈ 14.8 次 × 157 us（每层 ~0.5 次？= 30 层每步只有一半层在这个 bucket…待按桶看）。

## ★★★★★★ G8 2026-10-29 Step A 定靶（两边稳态、按 nvjet 行数分桶、按**每层每次调用** 比；vLLM 的"步"是被 decode 图切碎的片段 ≈ 3 层，我们的是整个 prefill 子批）
| 项 | 我们（p64 / fa_v5 / nvjet） | vLLM（CUTLASS persistent grouped / Triton unified_attention / nvjet） | 比 |
|---|---|---|---|
| **MoE gate‖up + down，每层** | 2 × 464–562 us = **0.93–1.12 ms** | 2 × 157–207 us = **0.31–0.41 ms** | **≈ 3×** |
| attention prefill，每次 | 115–128 us | 92–114 us | ≈ 1.1× |
| fp8 dense（qkv / o / dense MLP），每次 | nvjet 128x128 等，同 tile | 同 nvjet 家族 + 少量 `sm89_xmma_e4m3` | ≈ 1× |
| 小 kernel（norm / quant / combine / route / gelu） | ~0.4 ms/层 | ~0.3 ms/层（`finalizeMoeRouting` / `doActivation` / triton fused） | ≈ 1.3× |
- vLLM 的 MoE：`cutlass::device_kernel<GemmUniversal<GroupProblemShape…>>`，grid **(1,170,1)**（每 SM 一个 persistent CTA，TMA + warp-specialized，FlashInfer CUTLASS 后端），157 us 处理 ~2 K tok × 8 专家 → 权重 256 MB / 157 us = **1.6 TB/s ≈ 带宽上限**（所以 vLLM 的 MoE prefill 已经是 memory-bound 下限）。我们 p64 464–562 us = 0.5 TB/s。
- **结论**：prefill 37 vs 50 tok/ms 的差，**MoE grouped GEMM 一项就解释了全部**（每层多 0.6–0.7 ms × 30 层 = 18–21 ms/步 vs 混批步差 ~20 ms）。attention / fp8 / 小 kernel 不是靶子（G1/G3a 中性与此一致）。
- **两条路**（用户决策）：
  (a) **采用**：接 CUTLASS grouped GEMM（vLLM 同款，`GroupProblemShape` + TMA + persistent）——要把 CUTLASS 头文件进 `crates/kernels/cuda`（nvcc 编 sm_120a），NVFP4 block-scaled 的 CUTLASS 实例 + scale 布局要和我们的 `tile_off/offsets` 对接；逐位不同（累加序），走"批不变性"口径验收。工时 1–2 天。
  (b) **自写到 CUTLASS 水平**：p64 → persistent CTA（grid = SM 数、循环取 tile）+ TMA 多 stage（smem 不走寄存器）+ BM128。G1 已证 LDS 不是瓶颈、G2 BM128 预期 ≤ 1.3×；要到 3× 必须 TMA + persistent，等于重写一个 kernel，工时 2–3 天，且 5090 的 TMA/UMMA 路径（blackwell-mma.md）我们只做过 dense。
  按"目标是超越不是重写、vLLM 做得对的直接采用"，**建议 (a)**。

## G9 2026-10-29 路线 (a) 落地方案（已核对接口，待用户点头开工）
- **源**：容器自带 FlashInfer 0.6.18 `data/include/flashinfer/gemm/group_gemm_nvfp4_groupwise_sm120.cuh`（317 行）+ `data/cutlass/include`（CUTLASS 4.x，28 MB，含 cute / Sm120 BlockScaled 集合）。vLLM 基线跑的就是它（`FLASHINFER_CUTLASS` 后端，nsys 名 `GemmUniversal<GroupProblemShape…>` grid (1,170,1) persistent）。
- **接口**（`CutlassNVFP4GroupwiseScaledGroupGEMMSM120<TileM,TileN,TileK,SwapAB,…>(int_ws, float_ws, A, B, SFA, SFB, D, alpha, m_indptr, n, k, num_groups, stream)`）：**A 是全部 group 行拼接的一块、`m_indptr[num_groups+1]` 是 device 端行前缀和**——与我们 `sorted` 行 + `offsets[E+1]` 完全同形（不需要 D2H、不需要 host 循环）；B 是 `[num_groups, n, k/2]` 连续（我们是每专家独立分配 → 要在加载时把 128 专家的 gate/up/down 各拼成一块连续缓冲，一次性）；SFA/SFB 是 **cutlass Sm1xx 块缩放布局**（`tile_atom_to_shape_SFA`，即 128×4 swizzle）——我们 `gemm_nvfp4_sm120.cu` 已有 `swz==1` 读法（吃参照 a_s 时用），**写法**（量化时直接写 swizzled）要补；per-group `alpha`（我们有 `alpha[e]`）；问题形状由设备端 kernel 从 `m_indptr` 算（`problem_sizes[i]`），零 host 参与。
- **要做**：① build.rs：sm_120 一轮加 `-I<cutlass>/include -I<flashinfer>/data/include --expt-relaxed-constexpr -std=c++17`，新文件 `moe_grouped_cutlass_sm120.cu` 实例化 `TileM=128, TileN=128|64, TileK=128|256, SwapAB=false`（jinja 里的组合）并 `extern "C"` 包一层；CUTLASS 头文件进仓（`third_party/cutlass`，只 include 子目录 ~28 MB，或 git submodule）。② 权重加载：每专家 gate/up/down 拼连续 + 权重 scale 转 swizzled（一次性，`weights.rs`）。③ 激活量化 kernel 加 `swz` 输出（`grouped_quant_kernel` / `rmsnorm_grouped_quant` / `gelu_grouped_quant`：写 SFA 到 128×4 swizzle 位置）。④ `moe_grouped_gemm_cutlass` 的 ops.rs 包装 + layer.rs 在 `R > MOE_DEC_RMAX` 时走它（decode 小 R 仍走我们的 pipe4 BM16，它在 M ≤ 64 比 CUTLASS 快——vLLM decode 侧用的是另一套）。⑤ 门禁：`moe_gemm_timing` P2148 两进程 CHECKSUM **不会**同（累加序不同）→ 改用 `moe_splitk_parity` 式阈值（fp4 码差 ≤ 2% 且 worst ≤ 1、a_s 差 ≤ 2%）+ 端到端输出批不变性口径；ABAB c8–c64。
- **预期**：MoE prefill 每层 1.0 → 0.4 ms（vLLM 同 kernel），混批步 58 → ~40 ms，prefill 37 → ~50 tok/ms；c16 TPOT −0.5…−0.8、c64 −3（混批插入砍一半）。工时 1–2 天（主要在 ② ③ 的布局与 ① 的编译链）。
- **风险**：CUTLASS 在 nvcc 13.0 + sm_120a 的编译时间（每个实例 1–3 min）；FlashInfer 头依赖 `tvm_ffi`（只在 .cu 包装层，.cuh 本体不依赖，已核：.cuh 只 include cutlass）；5090 上 persistent kernel 与我们 decode 图高优先级流的共存（CUTLASS kernel 一个 CTA 跑完整 tile 队列，抢占粒度变粗——E4 的收益可能回吐一部分，要 ABAB 看 TPOT 与 TTFT 一起）。
