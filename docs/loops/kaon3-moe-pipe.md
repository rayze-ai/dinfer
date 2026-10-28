# loop: kaon3-moe-pipe —— MoE grouped NVFP4 GEMM 主循环 cp.async 软件流水

> 第一刀（R71/R72 定位）：`moe_grouped_sm120.cu` 的 `grouped_gemm_kernel` K 循环是
> `load sA/sfa/sB/sfb → __syncthreads → mma → __syncthreads` 纯串行、无 cp.async、无多 stage。
> 实测大 M（每专家 M≈132）下仍只 68 TFLOP/s = NVFP4 峰值 502 的 **13%**，根因 = load 停等 mma（无重叠）。
> vLLM 走 TRT-LLM fused MoE（flashinfer，日志 `trtllm::fused_moe::gemm2`+autotune），主循环 TMA/cp.async 多 stage 流水。

## 目标
`gemm3` 有效带宽从 ~300GB/s 往峰值（1.7TB/s）走，gemm3 从 13% 峰值往上；**M 无关**（decode+prefill 通吃）。
不动数值：cp.async 只改 global→smem 拷贝的时机，不改落到 smem 的字节与 mma 归约序 → **逐位一致是硬门槛**。

## 环境
- 5090（sm_120）容器编译（本机 nvcc 12.4 编不了 sm_120a）。本地写 → `put` 单文件 → 容器 `nvcc`/`cargo`。
- kernel：`crates/kernels/cuda/moe_grouped_sm120.cu`；FFI `crates/kernels/src/ffi.rs`；调用 `crates/kernels/src/ops.rs` + `models/kaon3/src/layer.rs` moe_branch。

## 做法（分步，每步过门禁再下一步）
1. **保留旧 kernel 当 oracle**：`grouped_gemm_kernel<BK>` 不删，作为逐位基准。
2. **加流水版** `grouped_gemm_kernel_pipe<BK,STAGES>`：双/三缓冲 smem，`cp.async` 预取下一 K 块 +
   `cp.async.commit_group`/`wait_group` 错位，当前块跑 mma。smem 布局 / mma 读法 / 归约序**与旧版逐字相同**。
   - 越界行（m_blk+r≥M / n_blk+r≥N）默认填值（sA=0x11、sB=0、sfa=0x38383838、sfb=0）：
     边界块少，这些行**不走 cp.async，直接 smem 写默认**（cp.async 后、wait_group 前）。
   - alpha/epilogue/量化 kernel 全不动。
3. **env 开关选版本**：`MOE_GEMM_PIPE=1` 走流水版，默认旧版（serve/对拍可 A/B）。

## 停止条件（机器可判定）
- ① **逐位门禁**：新测 `moe_pipe_parity`（self-contained，随机 A/B/scale/offsets，跑旧版+流水版，断言
  **逐字节相等 bad=0**）。`scripts/verify/kaon3-moe-pipe.sh` exit 0 = 过。**不过不接 serve**。
- ② gemm3 墙钟降：隔离微基准（或 serve KAON3_MOE_TIME）gemm3 从 ~0.096ms（decode）/~2.96ms（大M prefill）下降。
- ③ 端到端：serve bench（绑核 60-91，seed42/全共享1024/128out/rate8）吞吐升、进 CSV 趋势（CHANGE）。

## 风险
- cp.async 双缓冲同 buffer 读写竞态（compute(kb) 读 buf 与 load(kb+1) 写同 buf）：需 wait_group + __syncthreads 顺序正确。
- NVFP4 scale（sfa/sfb）也要流水预取，布局别错位。
- cp.async 要求 16B 对齐 src（旧版已 uint4=16B 拷贝，天然对齐）；越界行默认值不能用 cp.async 的 zfill（0x11/0x38 非 0）。
- 逐位是硬门槛：流水只改时机不改值，若 bad≠0 说明顺序/布局引入了差异，回退重查。

## 迭代上限 / 记录
每轮 put→编→对拍→（过了）bench 进 CSV，写进 loop 末尾 R 编号续 `kaon3-fp8gemm.md`（同一性能线）。

## R73 2026-10-28 第一刀首版(双缓冲)对拍失败 bad=11491/15488(74%),in-range也错→退单缓冲隔离
流水版首版(cp.async双缓冲B/sfb预取+A/sfa寄存器暂存STS)编译过,对拍 bad=11491/15488 first diff@16 row0(in-range)。
**in-range也错=不是纯时序竞态,是双缓冲引入数据错位**(prologue/store_a时机或buf索引)。
**稳妥法(先对再快)**:退成**单缓冲cp.async版**——旧版A/sfa STS原样,只把B/sfb的STS换cp.async+wait_group 0+__syncthreads。这样隔离"cp.async B索引错"vs"双缓冲时序错":单缓冲过了=cp.async索引对,再加双缓冲。
附带已知小bug(不影响in-range):越界fill z写成make_uint4(0x11111111)应是make_uint4(0x11)(每分量0x00000011);只影响越界行A,不写D故不影响对拍。

## ★ R73c 2026-10-28 单缓冲cp.async版对拍 bad=0/15488 通过→cp.async索引/zfill/布局全对
退成单缓冲(B/sfb换cp.async+wait_group 0,A/sfa原样STS)后对拍**bad=0**。坐实:cp.async的B/sfb索引、zfill(越界0)、落smem字节与旧版逐字一致。之前R73双缓冲74%错=**双缓冲时序/buf索引**引入,非cp.async本身。
但单缓冲wait_group 0无load/mma重叠=不提速(只换了拷贝指令)。下一步:在此基础加双缓冲(B/sfb预取[2]buf+错位wait_group 1),重点防R73的坑——compute(cur)时load(nxt)不能覆盖cur的buf,A/sfa也要[2]buf且store时机对。每加一步先对拍bad=0再谈性能。

## ★★ R74 2026-10-28 双缓冲cp.async流水版对拍 bad=0/15488 通过(逐位门禁过)
照搬已验证fp8范式(prologue存buf0+循环内store nxt在wait_group 1+__syncthreads之后,B/sfb cp.async预取到[2]buf,错位wait_group 1,trailing sync保证下轮写前本轮读完)。修了R73坑(store时机)。对拍**bad=0**。
逐位硬门槛过。下一步验第二关:gemm3墙钟降没降(大M prefill态每专家M≈132,旧版vs流水版kernel墙钟),确认cp.async重叠真提效率(目标13%峰值往上)。过了再接serve bench进CSV。

## ★★★ R75 2026-10-28 gemm3墙钟微基准:cp.async流水decode(小M)2.15×,prefill(大M)无效——与R69预期相反
逐位bad=0后测墙钟(8专家,N=704,K=2816,200迭代):
| 态 | 旧版 | 流水版 | 加速 |
|---|---|---|---|
| **小M=1(decode)** | 22.3us(356GB/s) | **10.35us(766GB/s)** | **2.15×** |
| 大M=132(prefill) | 35.2us(225GB/s) | 35.96us(221GB/s) | **0.98×无效** |
**与R69预期相反**:R69说"大M仍13%峰值→低效与M无关→否定M=1特化"。微基准显示**cp.async流水只救小M(decode),大M没用**。
→ 大M瓶颈不是流水(cp.async帮不上),是别的(tile/mma利用率/occupancy);**被流水救的恰是小M(decode)**——R64/R65最初方向(M=1效率)其实对,我R69反而带偏。
**但要对账(别又误读)**:微基准"大M 35us/225GB/s"与R69 serve实测"大M gemm3 2.96ms"量级差很多——微基准只1个grouped GEMM(grid.z=8激活专家),serve gemm3=3 GEMM×grid.z=128全专家。**不能直接套**。小M decode同理(微基准grid.z=8 vs serve 128)。
**但方向明确**:decode(serve主战场,TPOT大头)小M流水2.15×是真收益。decode TPOT=51.8ms,MoE 63%,gemm3=MoE 54%→gemm3减半可观。**接serve验**:MOE_GEMM_PIPE=1跑serve,KAON3_MOE_TIME看gemm3降没降,bench看TPOT/吞吐。逐位已过(bad=0)可直接上。

## R76 2026-10-28 接serve验PIPE:prefill gemm3持平(3.08 vs旧2.96,印证大M无效);decode态HTTP不通没测到
PIPE serve(MOE_GEMM_PIPE=1)编译起成功,warmup prefill态 gemm3=3.08-3.13ms vs R69旧版2.96ms=**大M持平**(印证R75微基准:流水对大M无效)。
**但decode态(小M,真正2.15×受益方)没测到**:临时ctr容器(NO_GRAPH+默认bridge网络)上 /v1/completions 请求不进forward(listening后无tok日志,curl空响应)——反复挡路(第N次),prefill moe-time是warmup产的非请求产的。
**换法**:别再跟临时容器HTTP死磕。用微基准模拟serve真实gemm3调用(grid.z=128全专家+decode小M,8激活专家各1行+120空),对齐R60 serve场景(gemm3 0.096ms),不经HTTP。改moe_pipe_timing支持可配E与激活数。
**或**:直接在生产容器kaon3-full(host网络,:8142,HTTP通)上验——但它是旧二进制(无PIPE)。要么重编traffic到生产容器,要么docker cp新二进制。下一步二选一。

## ★★★★ R77 2026-10-28 serve真实场景(grid.z=128)gemm3墙钟:decode 1.83×坐实,第一刀成立有效
对齐serve真实配置(E=128 grid.z,与serve一致):
| 场景 | 旧版 | 流水版 | 加速 |
|---|---|---|---|
| **decode(ACT=8,M=1)** | 22.6us(352GB/s) | **12.3us(644GB/s)** | **1.83×** |
| prefill(ACT=128,M=16) | 137us(926GB/s) | 128us(987GB/s) | 1.07× |
**decode单GEMM 22.6→12.3us=1.83×**(带宽352→644GB/s)。与微基准(2.15×)一致,grid.z=128/激活8的真实场景下仍1.83×。prefill大M仍无效(印证R75)。
**端到端账(估)**:decode gemm3 3个GEMM~1.8×→serve gemm3 0.096→~0.053ms省0.043;MoE 0.178→0.135;整层decode~0.43→0.39ms。TPOT/吞吐升几个百分点(decode侧)。
**第一刀结论:逐位bad=0(R74)+decode 1.83×(R77)=成立且有效**。收益集中在decode小M(serve主战场,TPOT大头),prefill无效(但prefill本就带宽高926GB/s没空间)。
**纠正R69/R75叙事**:R69"大M 13%峰值→低效与M无关"误导;真相=小M(decode)受流水救(访存流水浅),大M本就带宽够(926GB/s)。R64/R65最初盯M=1效率的方向对。
**下一步**:真实serve端到端bench(吞吐/TPOT进CSV)。临时ctr容器HTTP不通→上生产容器kaon3-full(host网络HTTP通),但它旧二进制,需docker cp新release二进制进去或重起带PIPE。逐位已过可直接上生产。
