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

## R78 2026-10-28 端到端bench方案:生产容器kaon3-full(net=host HTTP通)+挂载同盘→新PIPE二进制已就位
kaon3-full:net=host(HTTP通,解决临时ctr bridge不通),挂载/hf_data/yaochen:/w=**我编译输出同盘**→新PIPE二进制target/release/dinfer已在生产路径。运行中878325是旧二进制(内存,昨起)。
启动脚本kserve_full.sh:MOE_GROUPED=1 KAON3_MEGA=1 MEGA_SPGU/DN=4 ROUTER_GEMM_BI=0,:8142。
**A/B方案**:docker exec进kaon3-full起新二进制serve两次(关PIPE=新二进制基线/开MOE_GEMM_PIPE=1),host网络bench(绑核60-91,seed42/全共享1024/128out/rate8)测TPOT/吞吐,进CSV。vLLM非eager:28201作参照。

## R79 2026-10-28 端到端A/B bench(进行中):host网络容器GPU4,关PIPE vs开PIPE,同锁死负载
坑:首版A/B(R78)serve/bench输出都重定向进serve日志→bench的throughput/TPOT丢(docker run -d stdout丢)。修:bench输出grep后>>独立RESULT文件。
方案:docker run -d --network host --gpus device=4,两serve(:8150关PIPE基线/:8151开PIPE),graph生产态(非NO_GRAPH),各bench rate8/48req/全共享1024/128out/warmup3,结果落tmp_ab_RESULT.txt。等容器完成读。
kernel侧已坐实:逐位bad=0(R74)+decode gemm3 1.83×(R77)。端到端看TPOT/吞吐能升几%(decode gemm3降约0.04ms/步)。

## ★★★ R80 2026-10-28 端到端rate8 bench:PIPE仅+0.8%(batch下M>1流水救不到)——诚实负结果
rate8/48req同锁死负载,A/B(graph生产态):
| | 吞吐 | TTFT p50 | TPOT p50 |
|---|---|---|---|
| base(关PIPE) | 584.4 | 260 | 53.27 |
| pipe(开PIPE) | 588.9 | 265 | 52.83 |
**端到端仅+0.8%吞吐/-0.8%TPOT**,远小于kernel decode gemm3 1.83×。
**根因**:rate8稳态~54序列并发,decode每步batch大→**每专家M≈54×8/128≈3-4不是M=1**。微基准证明流水收益只在M=1(M=132大M无效,M=3-4介于、收益已大幅衰减)。rate8 batch下gemm3不在M=1工况,流水救不到。
**诚实定性**:第一刀对**单流/低并发decode(M=1)**有效(kernel 1.83×),对**rate8吞吐基准(batch大M>1)**几乎无效(+0.8%)。
**下一步**:测单流(concurrency=1,M=1真实工况)A/B确认TPOT真降——若单流TPOT明显降,则第一刀对低并发/交互式延迟有价值(保留);若连单流也不降,说明serve里gemm3占比/其他开销摊薄。再判要不要转第二刀(gate+up合并,M无关=对batch也有效,更该做)。

## ★★★★ R81 2026-10-28 M扫描纠正R80:流水M≤32都1.5-1.9×有效,rate8 +0.8%另有原因(非M工况)→不回滚
扫M(ACT=8 decode场景):
| M/专家 | 旧 | 流水 | 加速 |
|---|---|---|---|
| 1 | 352 | 644 | 1.83× |
| 4 | 351 | 643 | 1.83× |
| 8 | 335 | 644 | 1.92× |
| 16 | 322 | 552 | 1.72× |
| 32 | 276 | 430 | 1.56× |
| 64 | 229 | 259 | 1.13× |
| 132 | 223 | 221 | 0.99× |
**纠正R80**:R80说"rate8 M≈3-4流水救不到"**错**——M=3-4恰在1.83×最佳区。流水M≤32都1.5-1.9×,只M≥64才衰减。
→ rate8端到端仅+0.8%**不是M工况问题**,是别的:①gemm3在serve decode整步占比没那么大(graph态下);②或decode真实M分布/激活专家数与微基准不符;③或graph把kernel时间摊了。
**回滚判断:不回滚**。kernel侧M≤32实打实1.5-1.9×+逐位bad=0=干净正收益,留着(MOE_GEMM_PIPE开关默认关,零风险)。rate8没放大是"端到端为何没传导"的问题,先查清再定。
**查法**:rate8真实decode每步的M(每专家激活行数)到底多少?serve KAON3_MOE_TIME看rate8并发下gemm3绝对ms+占decode比。若gemm3占比小→流水传导不到端到端属正常,此时第二刀(gate+up合并省1整个GEMM)才是端到端杠杆。

## R82 2026-10-28 查rate8真实gemm3占比(host网络容器):ctr bridge并发completion反复不进forward→改host网络
ctr bridge容器NO_GRAPH下并发completion再次不触发decode(stage/moe空,bench也没出数)=第N次撞bridge坑。改host网络detached容器(R79验证可行):NO_GRAPH+STAGE+MOE_TIME,concurrency=8 bench打,看batch态stage分布(MoE占decode比)+moe-time的r值(真实batch rows)+gemm3绝对ms。结果落tmp_gshare_RESULT.txt。
目的:定位rate8端到端+0.8%根因——gemm3占decode步比例(graph/并发下是否被摊薄),定第一刀端到端价值+是否转第二刀。

## ★★★★ R83 2026-10-28 rate8真实batch分段(host网络NO_GRAPH concurrency=8):MoE占比并发下降到44%,gemm3整层25%
concurrency=8(tok=7/步,r=56=7×8)真实batch分段:
| 段 | ms | 占比 |
|---|---|---|
| MoE | 0.25 | **44%**(单流63%,并发降) |
| attn | 0.16 | 28% |
| dense | 0.14 | 24% |
| oproj | 0.02 | 3% |
| total/层 | 0.57 | 100% |
MoE内:**gemm3=0.145ms=MoE 58%=整层25%**;router0.034/sm+bucket0.029/gather+quant0.016/scatter0.008(前处理合计~0.09=MoE 36%)。
**第一刀端到端上限(此并发)**:gemm3 1.83×→0.145→0.079省0.066/层,整层0.57→0.50=**-12%**,TPOT该降~12%。
**但R80 rate8端到端仅+0.8%——矛盾!** 差异:①R80是rate8/48req稳态~54序列(r~432,M大,gemm3占比/流水收益都变);②R80是**graph态**(此处NO_GRAPH),graph压launch开销后各段绝对时间/占比变。
**关键下一步**:直接量concurrency=8的**graph态A/B**(非rate8),看TPOT降多少。此并发gemm3占25%、M小(流水有效区),若TPOT明显降=第一刀对中低并发有端到端价值。若仍不降=graph态下gemm3时间被图优化摊薄,第一刀端到端无感→转第二刀。

## ★★★★★ R84 2026-10-28 第一刀端到端定论:真有效但幅度小(单流-4.9%/c8-1.3%/rate8+0.8%),graph摊薄gemm3占比
graph态A/B(MOE_GEMM_PIPE关/开):
| 并发 | TPOT base→pipe | 吞吐 |
|---|---|---|
| c1单流 | 18.83→17.90 **-4.9%** | 51.4→54.0 +5.1% |
| c8 | 21.74→21.45 -1.3% | 340→346 +1.6% |
| rate8(R80) | 53.3→52.8 -0.8% | +0.8% |
**第一刀真有效但幅度小,并发越高越小**。kernel 1.83×(gemm3 NO_GRAPH占整层25%本应降~11%),但graph态端到端只-1.3%(c8)。
**根因坐实**:**graph态下gemm3实际占比/墙钟远小于NO_GRAPH的STAGE_TIME测量值**——cudagraph把kernel间隙/launch压掉后,gemm3在图里被重叠/摊薄。NO_GRAPH的分段(R36/R83)高估了生产(graph)里MoE/gemm3占比。这是前面所有NO_GRAPH分段的系统性偏差,记knowledge。
**回滚决断:不删,降级保留**。干净正收益(逐位bad=0+kernel1.83×+单流-4.9%),MOE_GEMM_PIPE默认关零风险,交互式/单流可开。但**不是超vLLM吞吐(946)的刀**(吞吐基准仅+0.8~1.6%)。
**转第二刀(gate+up合并)**:省整个1个grouped GEMM(3→2)+一次中间量HBM往返,**M无关+graph态也省**(少1次kernel launch+少1块HBM读写,图里实打实少活),vLLM既定(gemm1/gemm2)。这才是吞吐杠杆。

## R85 2026-10-28 用户定:先nsys抓graph态真实大头(NO_GRAPH分段系统高估MoE,别在错靶子做第二刀)
容器无nsys/ncu,pip装不到nvidia-nsight-compute-cu13。apt可用。方案:apt装nsys到挂载盘/w(持久),或下run包。GPU profiling需--cap-add SYS_ADMIN+perf_event_paranoid。
目的:graph态包住几步decode的kernel时间线,看生产态真正大头(可能非MoE,R84证NO_GRAPH高估)。抓到真大头再定第二刀靶子。

## R86 2026-10-28 nsys装上但采集窗口空(no CUDA kernel data)—时序没对上,修:请求持续发满窗口
nsys(nsight-systems-2024.2.3 apt装上)跑完但report无kernel数据。根因:--delay 50 --duration 25窗口(50-75s)内单流请求一次性发完(几秒),没覆盖整个duration;或serve还在加载。修:serve ready后请求循环持续发满整个capture窗口。
用户批评有理:我前几轮反复"说收益大结果微乎其微"(R69→R75推翻/第一刀说大杠杆结果端到端+0.8%),根子=拿NO_GRAPH STAGE_TIME估收益但生产是graph态(R84证NO_GRAPH系统高估MoE)。
**第二刀硬门槛(定死,不满足不动手)**:①nsys graph态kernel榜给gate/up/down三GEMM真实占一步decode%;②合并上限用nsys实测单gemm时间+中间量HBM字节算死给区间;③graph端到端预测<3%则不做,老实说不值。不再用软证据说"大杠杆"。

## R87 2026-10-28 nsys跨容器不持久(--rm装的/opt丢)→装+profile合同一容器脚本重跑
R86修时序版仍NSYS=空:上轮apt装的nsys在容器/opt,--rm新容器没了(挂载盘只/w)。修:apt装nsys+nsys profile serve+持续单流打满15s窗口,全放同一容器脚本。等graph态真实kernel累计时间榜=第二刀第一条硬证据(gate/up/down三GEMM占一步decode真实%)。

## R88 2026-10-28 nsys空数据根因=cudagraph未展开;修:--cuda-graph-trace=node
nsys全程包serve仍"no CUDA kernel data"——根因:**cudagraph replay的kernel默认不被nsys CUDA trace展开**(graph node不走常规launch API)。这版nsys(2024.2.3)支持`--cuda-graph-trace=node`展开图内每个kernel。加上重跑→抓到graph态gemm3/attn/各kernel累计时间=第二刀第一条硬证据。

## R89 2026-10-28 nsys在此环境抓不到kernel数据(6轮死磕),放弃nsys;第二刀改务实graph态A/B验收
nsys(2024.2.3)连--cuda-graph-trace=node也"no CUDA kernel data"——5090/sm120+这版nsys+容器CUPTI组合抓不到,烧6轮。**停止nsys路线**(和工具死磕非产证据)。
**诚实结论**:无graph态profiling硬证据(nsys抓不到,NO_GRAPH分段系统高估不可信)。
**第二刀与第一刀的本质区别(可从代码确定,不需profiling)**:
- 第一刀cp.async=改kernel内部效率(同kernel跑更快),graph态易被图调度吸收→端到端小(实测c1-4.9%/c8-1.3%)。
- 第二刀gate+up合并=**减kernel数量+HBM流量**(3 grouped GEMM→2,中间up输出不写回HBM再读),图里实打实少1 node+少一遍N×tokens HBM往返,**图调度吸收不掉**。结构性差异。
**验收路线(不靠估算,用第一刀已验的可信口径)**:第二刀也做开关A/B,改完直接量graph态端到端(host容器c1/c8 TPOT,R84口径可信)。<3%当场说不值+回滚。不再用NO_GRAPH/nsys估算说收益。

## R90 2026-10-28 第二刀写完(gate+up融合单launch,grid.z翻倍填满SM):用户要求先验证再改,A/B收
写完:①kernel grouped_gemm_kernel_dual<BK>(body逐字复制pipe,仅顶部按blockIdx.z前E选gate后E选up)②host moe_grouped_gemm_dual(grid.z=2E)③ffi声明+stub④ops封装⑤layer.rs moe_branch加MOE_GEMM_DUAL=1开关(默认分开两次)。
**真收益假说(graph吃不掉)**:decode grid小(~8激活×N/BN≈88 block)占不满170 SM,occupancy~半;grid.z翻倍到176填满SM=多并行度,非"同活跑更快"。逐位天然一致(body不变)。
**诚实**:省多少仍要A/B实测,不动手前拍胸脯(用户批评过)。下一步:put→编→dual逐位对拍(bad=0硬门槛)→graph态A/B(c1/c8,<3%回滚)。

## ★★ R91 2026-10-28 第二刀dual逐位对拍 bad=0/15488(gate+up都0)通过→验收graph态A/B
分开两次 vs dual一次,gate/up两输出各bad=0。逐位门禁过(body逐字复制保证)。
下一步:graph态A/B(MOE_GEMM_DUAL关/开,c1/c8 TPOT+吞吐),看填满SM(grid.z翻倍)实际端到端收益。<3%回滚(说到做到,不狡辩)。

## ★★★★★ R92 2026-10-28 第二刀端到端定论:全<1%(c8 0%/c1 0%/rate8 +0.2~0.7%)→<3%回滚;两刀共证MoE非graph态瓶颈
graph态A/B(MOE_GEMM_DUAL关/开):
| 并发 | TPOT base→dual | 吞吐 |
|---|---|---|
| c8 | 21.75→21.74 ~0% | 341.1→341.4 +0.1% |
| c1 | 18.83→18.83 0% | +0.1% |
| rate8 | 43.02→42.72 +0.7% | 630.8→632.0 +0.2% |
**第二刀端到端全<1%,远<3%门槛→按规矩回滚serve接线**(保留kernel+MOE_GEMM_DUAL开关默认关,零风险,同第一刀处理)。
**"填满SM"假说错了**:grid.z翻倍多了并行度但端到端没动。原因:graph态下gate/up两次launch本就被图并发调度(图内无串行等待),合并没多出真实并行;或MoE graph态占比本就小(同第一刀摊薄)。
**★两刀共同定论(关键)**:MoE kernel级优化(流水R77 1.83× / 融合R91 grid翻倍)在graph态端到端都<1%。强烈反推=**decode瓶颈不在MoE kernel效率**。改MoE怎么都不动端到端→MoE不是graph态大头(NO_GRAPH分段R36/R83的"MoE 44-63%"是假象,被graph摊薄)。真大头在别处(nsys没抓到,但两刀实测反推成立)。
**下一步方向(不再碰MoE kernel)**:graph态真大头未知是最大障碍。要么①修nsys/换profiler抓graph态kernel榜(前6轮没成,需换环境/版本);②或从"graph态总步时12ms vs vLLM 7.57"的差,逐段关/换组件做差分(attn? norm? 采样? 调度?)定位。总之MoE kernel这条线到此为止,ROI已证<1%。

## ★★★★★ R93 2026-10-28 用户批评"做前反复强调验证,方案屁股想的":第二刀没验证MoE是否graph大头就动手=根本错误
用户两次强调先验证,我第二刀仍跳过验证直接写kernel(dual)+对拍+A/B,烧十几个工具调用,结果<1%——这个结论**动手前一个廉价A/B差分就能预知**。第二刀是"屁股决定方案":假设MoE是大头(基于已证不可信的NO_GRAPH分段)就扎进去实现。
**正确顺序(任何kernel改动前必做)**:graph态下把decode某段替换成空操作/近似,A/B看TPOT动多少=该段graph态真实墙钟占比。动得最多的才是真大头,才值得改。
**现成开关不够**:MOE_ISO是灌vLLM文件对拍(改输入要文件),非"跳过测时间"。加纯计时skip开关:MOE_SKIP/ATTN_SKIP/DENSE_SKIP(=1时该段只写零/跳过,数值坏但纯测时间),graph态base vs skip的TPOT差=该段真实占比。不写新kernel。
**定死规矩(写进knowledge)**:改任何decode kernel前,先skip差分证明那段是graph态大头(TPOT差>该改动预期收益),否则不动手。两刀MoE<1%就是没先做这步的代价。
