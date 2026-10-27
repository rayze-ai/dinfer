# loop: kaon3 fp8 GEMM 优化 —— 逐环节攻破

> 2026-10-27 开。整层 mega 中 fp8 GEMM 占 ~92%(QKV 40.6/OPROJ 19.6/GATE·UP·DOWN 各~11)。
> 三段实测:**载入 68.6% / mma 30% / 写出 1.4%** → 访存量瓶颈(M≤16 decode 瘦长,算访比极低)。
> 双缓冲已试无效(mma太短盖不住预取,smem翻倍降occ)——根因是访存量不是调度。

## 环境/验证
- 远端 5090 ctr。逐位:`mega_layer_parity`(fused vs separated 0/N)。
- 耗时:同测试打印 QKV/OPROJ/GATE/UP/DOWN stage exec cycle + 整层融合 us。基线:融合~251us, QKV exec~2917555。
- 停止条件(机器可判):某环节改动后 ①逐位仍0/N ②QKV等GEMM stage exec 明显降(>5%)且整层融合us降(>3%,过噪声)。不满足=该环节无效,回退,下一环节。

## 环节清单(逐个打,不行看vLLM)
1. [ ] **M 攒批(算访比)**:decode 把并发请求攒进同一步提高 M(4→16+),同权重喂更多 mma。改引擎调度非kernel。最对症(前两次失败共因都是M小)。
2. [ ] **权重常驻/复用**:高频权重留 smem/L2 跨 tile 复用,减重复搬。查当前 sB 每 tile 是否重复搬同块权重。
3. [ ] **K 维 tile / BK 调**:BK=128 固定,调大/调 tile 让权重重用率升。
4. [ ] **cp.async 发射方式**:查 cp_async16_zfill 是否逐 16B 发射(发射开销),能否合并大块/用 cp.async.bulk(TMA)。
5. [ ] **向量化/对齐**:ldmatrix/mma 的 smem bank conflict、RS=144 对齐。
6. [ ] **参照 vLLM**:主力 cutlass(TMA+warp-spec pipeline,/data/ref/vllm-img.../kernels/linear/scaled_mm/cutlass.py);deep_gemm.py 也看。不照搬重依赖,学它小M decode 怎么解访存(TMA 批量搬 / persistent / M 对齐)。

## vLLM 参照点
- fp8 scaled_mm: `/data/ref/vllm-img-v0.28.0-pr455/vllm/model_executor/kernels/linear/scaled_mm/{cutlass,deep_gemm,triton}.py`
- 底层 cutlass fp8 kernel(c++)待定位。

## 记录
- R0 2026-10-27:loop开。双缓冲(环节未列,属调度)已验无效回退。从环节1(M攒批)开始——但先确认生产decode实际M(MEGA_ROWS=8上限),若已≈8则攒批空间看调度。

## R1 2026-10-27 环节2查明:权重每tile只搬一次(无重复),根因=算访比=M极低(带宽下界)
读 fp8_gemm_tile 访存账:每tile sB=BN×BK=128×128 fp8=16KB,B_PER_T=4(每线程4×16B cp.async,coalesced 128B行)。**每块权重只搬一次**(M≤16一次装下,无M-tiling重复)→环节2"权重复用"**不适用,本就没重复搬**。
**真根因(比双缓冲更本质)**:M=4 GEMM 算访比 = M·N·K/(N·K) = **M = 4 MAC/权重字节**,极低。读MB级权重只做4次乘加 → **HBM带宽下界**,kernel绕不开。且decode**每步重搬全部权重**(装不进smem),vLLM同样逃不掉。
**kernel级天花板**:只能把30%的mma藏进68.6%的load,≤30%,双缓冲已证occupancy吃不到。
**→ 环节2/3(权重复用/BK调)对load-once模式无效,跳过。** 真正杠杆:①M攒批(算访比↑,引擎层)②确认load是否已达峰值带宽(达峰=kernel级到顶只剩M攒批;未达=有coalescing/效率bug可修)。
**下一步**:验load实际带宽 vs 5090峰值(~1.7TB/s)。先算QKV权重字节数(ΣN=8192 × H)。

## R2 2026-10-27 带宽账(推翻R1"带宽下界"早结论):实测251us是理论31us的8×,远未达峰
真实尺寸:H=2816,I=2112,heads=16,kv=8,head_dim=256 → ΣN=8192。
权重字节(fp8=1B)/层:QKV ΣN×H=8192×2816=**23MB**(对上qkv requantize 23068672)、OPROJ H×heads·D=2816×4096=11.5MB、GATE/UP各I×H=2112×2816=5.9MB、DOWN H×I=5.9MB → **整层~52MB**。
5090带宽~1.7TB/s → 纯搬权重理论 **52e6/1.7e12=~31us/层**。实测整层融合 **251us = 8× 理论**。
**→ 远未达峰值带宽(~13%),不是带宽下界,是kernel低效可挖(R1"带宽下界kernel绕不开"下早了,收回)。** 8×gap来源待查:occupancy不足(mega持久CTA数少)/ cp.async效率 / 非GEMM开销(glue/sync/attention)/ exec cycle%不等于wall%。
**下一步**:实测达到带宽——QKV单stage wall时间 vs 其23MB理论13.5us。若QKV也8×→确认kernel低效,逐项查occupancy/发射;若接近→251us大头在别处(sync/glue/attention串行等待)。先看wait cycle:之前QKV wait=3.75M远大于exec,DOWN wait=10M——**wait(stage间等待)可能才是真大头,非GEMM本身**。

## R3 2026-10-27 wait语义+定位方法:exec/wait是全CTA累加cycle≠墙钟,需隔离单GEMM实测带宽
wait_cyc(mega_common.cuh:56-68)=CTA等依赖stage的自旋(while ld_acquire<need nanosleep40),**跨所有持久CTA累加**。wait≫exec→大量CTA空等少数干活(并行度/负载不均),但**累加cycle≠wall time**,不能直接换算。
**定位方法纠正**:251us是墙钟,exec/wait%是cycle占比,两者口径不同。要找8×gap真因必须隔离单GEMM看wall级带宽,排除stage依赖/glue/attention干扰。
**隔离方案**:mega_mlp_parity(dense branch)全零坏了不能用;mega_attn_parity不含GEMM。→ 需要一个**直接调fp8_gemm_tile单发**的微基准:给定M=4/8、QKV形状(ΣN=8192,K=H=2816),跑N次测wall us,对比理论13.5us(23MB/1.7TB/s)。
**下一步**:写最小微基准(或复活mega_mlp_parity全零——先查它为何全零,可能是同一类低效/配置问题的线索)。先查mega_mlp dense branch全零根因(R15曾过)——它既是测试也可能暴露GEMM本身bug。

## R4 2026-10-27 诊断偏离纠正:device/host printf调试绕路,改直接occupancy查询
插 MEGA_DIAG_GRID printf 查 grid/occ,但输出没出来(31处MEGA_DIAG全是nvcc把printf源码行当warning回显,运行时host printf被cargo test吞/或mega_layer_forward走的launch分支不同)。**又钻进"printf为何不出"的工具兔子洞,偏离攻破GEMM**。清理此插桩。
**纠正方法**:不靠printf。grid/occ 是确定值可直接算:occ=cudaOccupancyMaxActiveBlocksPerMultiprocessor(kern,256,smem),5090 sms≈170。smem单缓冲20.7KB、256线程/8warp。occ受smem+寄存器+线程约束。**可写一个独立小bin直接查询occ打印**(host函数,printf必出),或读kernel寄存器用量(cargo build -v看ptxas -v)。
**更该做的**:8×gap先别纠结occ单因素。QKV理论13.5us,整层理论31us实测251us。**最该先隔离:单个GEMM(非整层)的wall时间** → 复活/新建单GEMM微基准。整层251含15stage+glue+attention+stage依赖自旋,混太多。

## ✅ R5 2026-10-27 8×gap第一主因:mega_kernel 158寄存器卡occupancy=1 block/SM
ptxas -v(NVCC_PREPEND_FLAGS=--ptxas-options=-v):mega_kernel **Used 158 registers, 0 spill, 16592B smem(静态)**。
occupancy算:sm120每SM 64K寄存器。256线程×158=40448 reg/block → 64K/40448=**1.6→寄存器限1 block/SM**。(smem 16.6静态+20.7动态≈37KB,每SM~100KB非瓶颈。)
**→ 寄存器158把occupancy卡死1 CTA/SM**。整层mega把15 stage代码全塞一个kernel→寄存器爆→每SM仅1 CTA→GEMM阶段只~170 CTA(sms×1),**无法用更多warp藏HBM延迟→带宽打不满→8×gap**。这是访存瓶颈的真机制:不是带宽物理下界,是occupancy不足导致in-flight访存不够多。
**修法候选**:①降寄存器(拆kernel/减每线程活/__launch_bounds__(256,2)强制2block但会spill)②persistent本就1CTA/SM设计,但GEMM stage需要更多并行——矛盾:megakernel融合省launch vs GEMM要高occupancy。③GEMM stage单独用高occ kernel(退出megakernel融合,QKV/GATE等走独立launch)。
**停止条件验证**:降occ到2+后QKV stage wall降、整层us降>3%、逐位0/N。
**下一步**:试__launch_bounds__(256,2)看occ能否到2且不严重spill;或测独立GEMM launch(高occ)vs megakernel内GEMM的带宽差。参照vLLM:它GEMM不融进megakernel,独立cutlass kernel(高occ)——这正是它小M也快的原因之一。

## R6 2026-10-27 OCC=2试:寄存器158→128(0spill)但整层反慢(277 vs 251)→回退,须隔离单GEMM
改mega_kernel OCC 1→2:ptxas 158→128寄存器0 spill,occupancy应升2。逐位0/N。**但整层融合 277.7us vs OCC=1 251us,反慢10%**。
分析:①m=4 task少,2block/SM填不满(task不够);②128寄存器下代码变低效(虽0spill);③持久megakernel 2block自旋竞争增。**整层测试混15stage+glue+attn,GEMM带宽改善被淹没/被别的stage拖累看不清**。
→ 回退OCC=2。**R4/R6都指向同一结论:必须隔离单GEMM微基准**,整层parity判不了GEMM级优化(task少+混杂)。
**下一步(不再在整层上试)**:写最小micro-bench直接launch一个高occ的fp8 GEMM kernel(QKV形状M=4/ΣN=8192/K=2816),测wall us vs理论13.5us,且扫occupancy(1/2/4 block)看带宽曲线。这才能干净回答"GEMM独立高occ能否接近带宽"=整层融合该不该让位。

## ✅ R7 2026-10-27 决定性参照:独立GEMM M=4达峰值带宽,坐实megakernel融合是错权衡
gemm_small_m(现成微基准,qwen35形状,M=4)三者带宽:
| M=4 GEMM | lt_default | small_m | mma |
|---|---|---|---|
| attn.qkv 8192×5120 | 3103 GB/s(27us) | 2885 | 4096 |
| mlp.gate_up 34816×5120 | 1600 | 1655 | 1583 |
| mlp.down 5120×17408 | 1522 | 1472 | 1560 |
**独立launch GEMM在M=4就达~1500-3100 GB/s=接近/达5090峰值(~1.7TB/s)**。对比mega融合fp8_gemm_tile=理论1/8(~13%)。
**坐实R5**:①独立高occ GEMM M=4就打满带宽(硬件+好kernel做得到,非M小的物理限);②mega融合GEMM慢8×是occupancy=1卡的(158寄存器塞15stage)。
**核心结论(loop攻破)**:**整层megakernel对GEMM占92%的decode层是错误权衡**——融合省~1.2×launch,远抵不过GEMM带宽损失8×。megakernel适合小算子密集(省launch主导),不适合GEMM-bound(occ主导)。
**注**:参照是bf16 cuBLASLt/qwen35形状,非kaon3 fp8;原理成立(M=4可达峰)但kaon3 fp8独立路径带宽待直接测确认。
**下一步方向(重大,待定)**:decode层GEMM退出megakernel融合,走独立高occ kernel(如现成gemm_fp8_static_bf16或cuBLASLt),只把小算子(norm/rope/gelu/attention)留融合或也独立。预期GEMM段8×→接近1×,整层可能从251us→大幅降。需重新设计mega边界——这推翻"整层mega"路线,回到"关键算子融合+GEMM独立"。

## ⚠️ R8 2026-10-27 重大纠正:"融合亏GEMM带宽8×"无数据支撑,R5/R7结论作废
用户要8×的数据支撑。如实交代来源:
**8×唯一出处=R2**:整层权重52MB/1.7TB/s=理论31us,实测整层融合251us,251/31≈8×。**就这一个比值。**
**为何不成立(三点)**:
1. 分子分母跨口径:251us是**整层墙钟**(含attention+glue+15stage+stage间自旋),31us是**纯搬权重**理论。整层÷纯权重的商,不该叫"GEMM带宽倍数"。
2. **从没单独测过融合里GEMM的墙钟**。只有exec cycle占比(R3自己记了≠墙钟)。融合GEMM实际花多少us=未知。
3. R7"独立3103GB/s"是**bf16 cuBLASLt/qwen35形状**,非kaon3 fp8 fused。与整层13%并排=苹果比橘子。
**自相矛盾铁证**:R6把OCC 1→2(寄存器158→128)整层反而**251→277us变慢**。若occupancy=1真卡GEMM带宽,提occ该变快。变慢→occupancy至少非简单主因。R6记了却R7照写"决定性"。
**作废**:R5"8×gap第一主因=occupancy"、R7"决定性参照/整层mega错权衡"**均无干净数据支撑,撤销结论**。合并本身有实测真收益(parity 1.55× + 逐位正确 + 根治乱码),**不推翻**。
**唯一能定论的实验(此前从未做)**:同形状同dtype(kaon3 fp8)、同权重,**GEMM融在megakernel里 vs 单独launch**,各测墙钟带宽。在此之前不谈改架构。
**8×真因仍未知**:可能在attention/glue/stage自旋(wait≫exec的线索),不一定在GEMM。需先把整层251us按墙钟拆到stage级(非cycle占比)。

## ⚠️ R9 2026-10-27 用户提醒接上R26:瓶颈是wait(等依赖)非GEMM,本轮GEMM叙事整个推翻
用户："QKV/OPROJ/GATE/UP/DOWN每环节都验证过,时间没那么大"。翻历史坐实:
- **kaon3-mega R26(2026-10-26)已定论**:"分阶段计时:瓶颈不是GEMM本身,是**等依赖(wait)占大头**";三GEMM占96% exec但**wait≫exec**。
- dense MLP(GATE+UP+DOWN+norm+gelu)mega一次launch=**53.9us**(R3最佳split-K),**根本无"每GEMM慢8×"**。
**本轮错误链彻底复盘**:R2拿整层251us÷纯权重理论31us得8×→R3自己发现wait≫exec却没翻R26→R5硬拐到"GEMM occupancy=1"→R7拿bf16参照凑"独立快8×"→R8已自我纠正无数据→**R9接上R26坐实:exec(GEMM真算)不慢,wait(stage间CTA等依赖自旋)才是251us大头**。
**结论**:①"融合亏GEMM带宽8×"彻底不成立,撤销;②真瓶颈=stage间等依赖自旋(R26早定位),**这恰是融合调度可优化的**(依赖排布/并行度/split-K平衡wait),不是"融合错了要拆GEMM";③我把方向搞反——该优化的是wait,不是GEMM kernel,更不是拆融合。
**正确下一步(回到R26方向)**:降wait——看哪些stage串行等待可并行/重叠(dep链)、split-K平衡各stage wait(R2/R3已在做:sp_gu/sp_dn调到4/4使DOWN wait-23%、mega 66.6→53.9us)。继续沿R26/R3的split-K+依赖优化,不是推翻融合。
**作废本loop的R5/R7方向**(GEMM独立/拆融合)。保留R2带宽账(但它是整层非GEMM)、R5 ptxas寄存器事实(158/occ=1,真的,但非8×主因)。

## R10 2026-10-27 方法纠正(用户):别绕,就"跑10轮抓每stage墙钟,由大到小排"
用户:"排查没那么复杂,跑10轮每轮把每个环节耗时抓出来,由大到小就定位了"。对。我前9轮绕occupancy/带宽/dtype/拆融合=过度分析。
**标准动作(以后照做)**:mega_layer_parity 已加逐stage墙钟(每stage单launch+sync计时);跑N轮取均值+看稳定,15 stage由大到小排,大头即瓶颈。当前后台b4h13yyj5在跑(编译完在计时)。
**判据**:由大到小排完,top1-3 stage就是优化目标;若top是GEMM→优化GEMM,若是wait/attention/glue→优化那个。不预设结论(前9轮预设了"GEMM"才绕歪)。

## ★ R11 2026-10-27 整体复盘(用户要求认真过一遍):只认硬数据,问题到底在哪
**硬测数据(全可追溯)**:
- R26: dense MLP B=1 **wait 574万cycle = exec 225万的2.5×**;DOWN wait是exec 4.5×;三GEMM占exec 96%。
- R3: dense mega一次launch **53.9us**(split-K 4/4最佳)。
- 整层exec: QKV37.9%+OPROJ18.6%+GATE/UP/DOWN各~10%=GEMM~87%。
- **融合vs分离(同测试,可比!): m=4=247vs289us 融合更快**;m=1 179/228;m=8 354/395。
- R5: mega_kernel 158寄存器→occupancy=1 block/SM(ptxas实锤)。
- R7: 独立GEMM(bf16/qwen35形状)M=4达1500-3100GB/s(但异dtype/形状,不能直接比mega fp8)。
**拼起来问题在哪**:
1. **"融合错了"不成立**:同测试融合247<分离289,融合净赚+逐位+根治乱码。无任何数据说融合更慢。
2. **"融合亏GEMM 8×"是我编的**:=整层251÷纯权重理论31(跨口径+混bf16),撤销。GEMM exec占96%但非瓶颈。
3. **真低效=wait(CTA等依赖自旋),R26早定位,硬数据**:wait=exec 2.5×。两因叠加:①stage串行(UP依赖GATE但本可并行);②occupancy=1(158寄存器)CTA少自旋空等。
4. occupancy=1与wait同一硬币:R5发现occ=1对(ptxas实锤),但R6单提OCC到2反慢→光提occ不解串行,两者须一起治。
**还不知道(诚实标注)**:251us绝对意义算不算慢,无干净判据。理论31us是纯权重带宽下界(忽略计算+小M现实)非合理目标。要判"整层该多快"需每层预算(N层→端到端tok/s vs vLLM),没算过。
**结论(不跳步)**:问题从来不是融合伤GEMM,是我反复拿跨口径数据凑8×带歪方向。真优化点=降wait(gate/up等独立stage解串并行,融合内可做;occ要和依赖一起治)。GEMM kernel本身不该动(exec非瓶颈),拆融合是错的(融合净赚)。
**反省**:前10轮反复推翻的根因=每轮预设结论("GEMM慢")+拿跨口径数据外推,不先摊硬数据。R26一年前已答,我没先翻历史。

## R12 2026-10-27 坑:逐stage单独launch计时会死自旋(卡63分钟)
我加的"每stage单独launch+sync计时"段卡死63分钟(etimes 3799s)。根因:**单个stage脱离依赖链单独launch时,dep/ready表的need永远等不到前序signal→持久CTA在while ld_acquire<need死自旋**(这正是wait机制,单stage跑必卡)。方法本身错。
已杀(pkill -x cargo,cargo=0;pgrep -f匹配到自身非残留)。
**改对的计时法**:用已验证能跑的**分离模式**(15 stage按依赖顺序launch,mega_layer_parity原有),在每个stage的launch前后插 sync+Instant计时——它走完整依赖链不卡,逐stage墙钟=相邻sync时间差。不单独跑孤立stage。
**下一步**:删掉死自旋的逐stage段,改在分离模式循环里逐stage sync计时。

## ★★ R13 2026-10-27 逐stage墙钟(用户方法,顺序launch+sync,50轮,由大到小)——分布平,无单一瓶颈
| stage | 墙钟us | % | 类型 |  | stage | 墙钟us | % |
|---|---|---|---|---|---|---|---|
| KVWR | 47.3 | 11.6 | glue(写KV几乎无算) |  | QKV | 25.8 | 6.3 (GEMM,exec占40%那个!) |
| IN | 37.5 | 9.2 | norm |  | UP | 25.7 | 6.3 |
| RES | 37.3 | 9.1 | glue |  | POST | 24.3 | 6.0 |
| OPROJ | 32.3 | 7.9 | GEMM |  | QKNORM | 22.7 | 5.6 |
| PREFF | 27.2 | 6.7 | glue |  | DOWN | 22.4 | 5.5 |
| ROPE | 26.5 | 6.5 | glue |  | OQNT | 19.5 | 4.8 |
| GATE | 25.9 | 6.4 | GEMM |  | ACT | 17.7 | 4.3 |
|  |  |  |  |  | ATTN | 16.0 | 3.9 |
合计408us(单launch+sync,含每stage固定开销);对照:分离15launch=296us,融合1launch=252us。
**决定性读法**:
1. **分布平,无单一瓶颈**(11.6%→3.9%无独大)。exec占40%的QKV墙钟仅25.8us/6.3%,与其他同量级。
2. **最耗的KVWR(47us)是trivial glue(只写KV)**→数字被**每stage launch+sync固定开销主导**,非真实计算量。
3. **这证明融合为何对**:分离15launch 296us→融合1launch 252us,省的=堆叠launch开销。**GEMM非瓶颈**(每GEMM stage仅~6%)。
**彻底结论(数据说话)**:**无单一瓶颈stage可打,252us平摊15 stage,大头是众多小stage的固定开销非GEMM**。→单kernel优化(GEMM/attn/双缓冲/方向三)都不会有大收益(没有独大stage),完美解释方向三(3.6%)/双缓冲(GEMM非瓶颈)为何都无效=都在打小占比目标。
**前10轮错因彻底明确**:盯着"GEMM exec占87%"以为GEMM是瓶颈,但exec占比≠墙钟;墙钟平摊,融合已经是对的方向(省launch),真实gap不在任何单个kernel。
**真要再提速的方向(基于平摊分布)**:①减stage数(更多片上融合glue,已在做:fuse_glue/fuse_ff把KVWR/ROPE/RES/PREFF吸收→252降到209/190);②降每stage固定开销(占大头);不是优化任何单个GEMM/attn。

## ★★★ R14 2026-10-27 10轮逐stage墙钟(用户要求跑10次+列明细)——极稳,GEMM非瓶颈坐实
10轮×50iter,大多数stage抖动<2%(r0-r9几乎一模一样),偶发尖刺(r9 KVWR53/r6 ATTN25)是单轮噪声,中位不受影响。
**汇总(10轮中位,由大到小)**:
KVWR47.1(11.6%glue写KV) IN37.4(9.2%norm) RES37.1(9.1%glue) OPROJ32.0(7.9%GEMM) PREFF27.1(6.7%glue) ROPE26.4(6.5%glue) QKV25.7(6.3%GEMM) UP25.6(6.3%) GATE25.6(6.3%) POST24.2(6.0%) QKNORM22.6(5.6%) DOWN22.2(5.5%GEMM) OQNT19.4(4.8%) ACT17.6(4.3%) ATTN15.9(3.9%)
合计中位406us(含每stage launch+sync固定开销);对照分离15launch294.8us,融合1launch251us,glue融合208us,+ff融合192us。
**10轮坐实**:
1. **分布极平极稳(抖<2%),无任何单一瓶颈**(11.6%→3.9%)。exec占40%的QKV墙钟仅6.3%,与norm/写KV glue同量级。
2. 5个GEMM墙钟合计~131us占32%;单个QKV才6.3%。
3. **最贵前三=KVWR(写KV)/IN/RES=glue+norm,几乎不算数→launch+sync固定开销主导**。
**彻底定论(10轮硬数据,抖<2%)**:GEMM非瓶颈,无单一瓶颈,251us平摊15stage大头是固定开销。→减stage数(片上融合)才有效(已证251→192),优化任何单GEMM/attn/双缓冲/方向三都无效(都打小占比)。"GEMM慢8×"彻底证伪。
**注**:此表含每stage launch+sync开销(合计406>真实融合251),绝对值偏大,但相对分布+抖动结论硬。真实融合内各stage占比需看融合态per-stage(exec cycle,R26有);但"无单一瓶颈""GEMM非独大"两个口径一致。

## ★★★★ R16 2026-10-27 真问题坐实(用户方向:等待):融合态 wait=exec的5.12×,喂不饱是大头
融合态(真实1 launch不打断)抓每stage exec vs wait(CTA等依赖自旋),按wait排序:
**exec合计747万 wait合计3828万 = wait/exec 5.12×**。CTA等前一步的时间是真算的5倍!
| stage | exec% | wait% | wait/exec | 等谁 |
|---|---|---|---|---|
| DOWN | 10.1 | 24.1 | **12.2×** | ACT |
| GATE | 10.7 | 17.1 | **8.2×** | PREFF |
| UP | 10.4 | 17.1 | **8.4×** | PREFF |
| ATTN | 7.1 | 15.5 | **11.2×** | KVWR→QKNORM |
| OPROJ | 18.8 | 13.4 | 3.7× | OQNT |
| QKV | 38.9 | 9.0 | **1.2×** | IN |
**读法(前一步喂不饱后一步)**:
- QKV wait/exec仅1.2×=唯一喂得饱(依赖IN快,一来就有活)。其余全在空等。
- DOWN等ACT 12×、ATTN等KVWR→QKNORM 11×、GATE/UP各等PREFF 8×=这些stage绝大部分时间空等前序,真算只占零头。
**根因=持久megakernel的CTA调度让各stage疯狂空等**:
1. GATE·UP都等PREFF各8×,但GATE/UP本可并行(都只依赖norm)→调度挤一起等。
2. DOWN等ACT 12×:ACT(gelu)快但DOWN干等,ACT→DOWN衔接大空隙。
3. ATTN等KVWR→QKNORM 11×:等KV写完+qknorm链太长。
**与vLLM差距真因对上**:不是GEMM慢(QKV exec38.9%但wait仅1.2×健康),是**CTA调度让各stage空等,wait是exec 5×**→消除等待理论可快到~1/5。这是实打实代码/调度问题,不是"没瓶颈"(我之前R13/R14说"无瓶颈"是错的:墙钟测法打断流水掩盖了wait,融合态wait才暴露)。
**纠正R13/R14**:逐stage单独sync计时打断流水→测不到wait→误判"无瓶颈分布平"。融合态exec/wait才是真相:**瓶颈=等待(5.12×),不是任何单算子**。
**优化方向(数据指向,真方向)**:降wait——①GATE/UP解依赖并行(都只依赖norm,别串PREFF);②缩依赖链(ATTN等的KVWR→QKNORM);③ACT→DOWN衔接;④根上:持久CTA调度/occupancy=1让CTA无法重叠不同stage。先啃最大的DOWN(24%wait)和GATE/UP(各17%)。

## R17 2026-10-27 GATE/UP依赖已并行,wait非依赖串是CTA不够(纠正R16打法)
读dep表(mega_layer_sm120.cu:541-542):`h_dep[S_GATE]=S_RES, h_dep[S_UP]=S_RES`,**GATE/UP都只依赖RES,彼此无依赖=已并行**。R26说的"dep[UP]=GATE串"现已不成立(改过了)。
**那wait 8×哪来的**:不是依赖串,是**调度**——GATE(n_gu×sp_gu个task)+UP(n_gu×sp_gu个)都在RES完成后ready,但**occupancy=1→CTA少(~170)→两批task排队串行执行**。wait=task等CTA空闲,非等依赖。
→ 纠正R16"GATE/UP解依赖并行"打法:依赖已并行,真问题是**CTA数不够让task并行**。根子还是occupancy=1(158寄存器)。
**但R6已试OCC=2反而慢**——矛盾待解:occ=1说CTA少,提occ=2又慢。可能OCC=2的128寄存器spill或task调度变差。需重新审视:wait大是因为①CTA总数少(occ low)还是②单stage内task粒度/分配问题。
**下一步**:先量GATE stage的task数 vs grid CTA数(n_gu·sp_gu vs sms·occ),看是不是task≫CTA导致排队;再看wait到底是"等CTA"还是"等RES这一个前序stage"(若RES慢,GATE/UP等的是RES不是CTA)。

## R18 2026-10-27 算账:GATE task数(17)≪CTA数(170),wait非排队是空转CTA自旋
算账:n_gu=I/128=2112/128=**17 task**(sp_gu=1)。grid CTA=sms×occ=170×1=**170**。**17 task≪170 CTA**→不存在task排队等CTA。
→ GATE/UP wait 8×既非依赖串(已并行)也非task排队。只剩:**wait_stage自旋**——GATE阶段170 CTA只17个有task干,**其余153 CTA领不到task却在自旋**(wait_stage等dep signal或空转),wait被灌爆。exec/wait是per-task thread0累加,但空转CTA也可能计入wait。
**真根因浮现**:持久megakernel每stage只有少量task(n_gu=17/n_dn=22/sp=1),却起了170 CTA→**绝大多数CTA全程空转自旋**,wait虚高且真实浪费SM。这不是"喂不饱后一步",是**CTA数远超task数,大量CTA空占SM自旋**。
**这解释R6 OCC=2为何更慢**:task本就少,CTA已过剩,再翻倍occ只增空转竞争。也解释wait=5×exec:5×里大半是过剩CTA的空转,非真实数据依赖等待。
**下一步验证**:看wait_stage(mega_common.cuh:56-74)——领不到task的CTA在干嘛;grid CTA数是否该按task数砍(grid=min(sms·occ, 各stage max task)而非sms·occ)。若砍到~22 CTA,空转消失,wait应暴降。先读mega_kernel的task循环+grid设定。

## ★ R19 2026-10-27 真机制坐实:依赖链串行,后段stage的CTA自旋等前链(非CTA不够)
读task循环(mega_common.cuh:140)+grid(mega_layer:595):grid=min(sms·occ, **n_tasks=整层所有stage task总和**)。CTA轮转领全局task,`stage_of`映射到stage,**同CTA可跨stage领不同task**。
**wait真相**:CTA领到靠后stage的task(如DOWN)→`wait_stage`里`while ld_acquire<need __nanosleep(40)`自旋等该stage的dep(ACT)signal完。DOWN数据要等ACT→GATE/UP→RES→OPROJ→...→QKV整条前链算完。**wait 12×=DOWN的CTA等这条长链**。
**精确机制(前一步喂不饱后一步)**:不是CTA不够(R18错),是**15 stage串成一条依赖链,后段stage的CTA只能干等前段**。持久megakernel用全局task+stage依赖,后段CTA领到task却要自旋等整条前链→wait=exec 5×。
**R18"CTA空转"部分对**(后段CTA确在自旋)**但归因错**(不是"领不到task空转",是"领到了但等依赖")。
**这是megakernel范式的固有代价**:单kernel内stage串行,无法像多kernel那样让GPU在stage间流水重叠(多kernel时GPU调度器可让kernel N+1的早期块趁kernel N尾部空闲就上)。megakernel反而把这重叠能力关了——**这可能就是比vLLM慢的结构性原因**:vLLM多kernel独立launch,GPU硬件调度器天然重叠相邻kernel;megakernel串成一条链,后段纯等。
**下一步(验证这个结构假设)**:对比同形状 megakernel(1launch串链) vs 多kernel(各stage独立launch,让GPU调度器重叠)的墙钟。若多kernel更快=坐实megakernel串链是病根,融合方向要重新权衡(省launch vs 丢硬件流水重叠)。这次别只看cycle,直接比两种launch的wall us(测试里已有:分离294.7 vs 融合251——但分离是顺序launch+我可能没让它重叠,需确认分离是否真并发)。

## ★ R20 2026-10-27 用户两问驳回R19拆融合:合并是解法非病根,wait多半是m=4并行度小非可回收
用户:"合并了就不能解决吗?分开launch不是又多了吗?" 两问都对,R19"拆回多kernel"错,撤销。
**1. wait是数据依赖(物理),合并/拆开都存在**:DOWN必等ACT出输入,拆成多kernel同样等。拆开不消除依赖等待。wait非合并造成。
**2. 真地板=依赖链关键路径延迟**(QKV→...→DOWN串行,各stage exec之和≈251us)。要快只有:①每stage更快 ②缩链(减stage)。**减stage正是融合在做**(fuse_glue/ff 251→192实测有效)→**合并是对的方向**。
**3. 拆开两头亏**:launch开销加回+依赖等待没少。拆错。
**R19错因**:说"vLLM多kernel有硬件流水重叠、mega关掉了"——**单stream上kernel默认串行也等依赖,不成立**。又一次拿没验证的结构假设往"融合错"靠。
**wait=5×exec真相(重估)**:m=4每stage仅17-22 task,但170 SM→任何时刻只17-22 task能跑,**150 SM本就没活**(合并时自旋计入wait,拆开时闲置不计数,物理浪费一样)。**wait大半是m=4并行度小GPU天然吃不满,非"修调度可快5×"的可回收时间**。
**重新解释与vLLM差距**:decode m=4延迟受限(并行度小),拼关键路径短+每stage快。vLLM优势可能是**关键路径上每个kernel本身更优(cutlass)**,非流水重叠。
**诚实**:我这十几轮反复拿未验证假设往"融合错/GEMM慢/拆开"上靠,被用户逐次驳回。真相朴素:①融合方向对(缩关键路径);②decode m=4 GPU吃不满是本质;③要快=缩链+每stage更快,不是推翻融合。
**下一步(不再推翻融合)**:沿缩关键路径——①继续融合吸收剩余glue stage(KVWR/ROPE等);②看关键路径上哪个stage的exec(真算)最大可优化(QKV exec 38.9%最大,但前测它带宽健康);③对比vLLM同形状单层wall确认差距绝对值(还没做)。

## ★ R21 2026-10-27 回答"为什么等"(用户):前段stage并行度低(sp=1)没吃满GPU→算得慢→下游等
用户:"找为什么等,解决为什么等,不是融合/拆"。对。直追:
- DOWN等12×=领到DOWN task的CTA从t=0自旋等整条前链(QKV→..→ACT)算完。
- **前链为何慢**:sp=1时每stage task少(QKV=ΣN/128=64,GATE/UP各17,DOWN=22),GPU 170 SM→**150 SM全程闲**,每stage用不满GPU算得慢→下游等久。
- **"为什么等"的答案=前段stage并行度太低没吃满GPU,拖慢关键路径**。非融合/拆问题,是每stage并行度。
**直接手段=split-K**:K维切sp份→task×sp→用上闲置SM→stage更快→wait降。**R16测wait时sp全=1(split-K关着)!**(parity:244-247 sp_qkv/o/gu/dn=1)。历史R3已验sp_gu/dn=4/4使dense 66.6→53.9us、DOWN wait-23%,方向对,只是测wait用了sp=1。
改parity:sp_qkv/o/gu/dn 改环境变量可调(envsp,默认1)。**下一步:同测试跑 sp全1 vs sp全4,对比各stage wait——验证split-K降wait(解决为什么等)**。

## R22 2026-10-27 sp=1基线拿到;警觉:wait累加cycle指标会误导,验split-K要看墙钟
sp=1基线(网关断,sp=4未跑完):wait/exec=5.14×,DOWN wait 24.1%(12.2×)、GATE/UP各17%(8×)、ATTN15.5%、QKV仅9%(1.2×)。与R16一致(稳定)。
**警觉(别被累加指标带偏)**:wait=所有task thread0累加cycle。DOWN 22 task(sp=1),每个CTA从t=0自旋等前链→22份"等前链"cycle累加→自然巨大。**wait大=依赖深度×并行宽度的累加,不等于"有这么多时间可省"**。
**sp=4的两种相反可能**:①task×4每stage更快吃满SM→下游少等→wait降;②task×4更多CTA从t=0自旋等前链→wait累加反涨。**wait累加指标本身可能误导**。
**→ 验split-K必须看墙钟(融合us),不是wait累加**。R3历史已答:sp_gu/dn=4/4使dense 66.6→53.9us(墙钟降)。所以split-K有用,但指标要对。
**下一步(网关恢复)**:跑 sp=1 vs sp=4 的**融合墙钟us**(测试里"融合launch无glue"那行,251us),看split-K降多少墙钟;扫sp∈{1,2,4,8}找最优(如R3 dense的4/4)。不看wait累加。
**同时诚实**:wait=5.14×这个"5倍"我之前当"可省5倍"宣传过(R16),现在明确——它是累加指标,真能省多少只能看墙钟。又一个被单指标带偏的教训。

## ★★★ R23 2026-10-27 split-K扫描(看墙钟)：sp=4最优251→215us降14%,坐实"前段并行度低是等的根因"
sp扫描(融合墙钟,逐位全✓):
| sp | 融合us | wait/exec |
|---|---|---|
| 1 | 250.9 | 5.12× |
| 2 | 228.4 | 5.75× |
| **4** | **215.3** | 5.61× |
| 8 | 229.8(过切反弹) | 5.45× |
**sp=4最优:250.9→215.3us降14%,逐位保持**。sp=8过切(task太多超SM又排队)反弹。
**证两件事**:
1. **split-K解决"为什么等"(用户方向对)**:切K维→每stage task×4→用上闲置SM→前段算更快→关键路径缩→墙钟降14%。**前段并行度低确是等的根因,split-K是解法**。
2. **wait累加指标会骗人(R22警觉对)**:墙钟降了(251→215)但wait/exec反从5.12涨5.61!因sp=4后task×4,更多CTA从t0自旋等前链,wait累加涨。**只看wait会误判"split-K更糟",墙钟才是真相**。必须看墙钟。
**对上R3历史**:R3 dense-only sp_gu/dn=4/4 使66.6→53.9us(-19%);这里整层sp全4 251→215(-14%),一致方向。
**下一步(继续解决为什么等)**:①sp=4非均匀细调(各stage task数不同:QKV n=64已多可能sp小,GATE/UP n=17少需sp大;分别扫sp_qkv/o/gu/dn找各自最优,非全4);②这是整层默认该开的(service.rs 当前sp默认?需确认kaon3 serve是否已用sp=4);③215us里还有啥等待→继续追关键路径上最慢stage。

## ★★★ R24 2026-10-27 端到端 dinfer vs vLLM非eager(锁死负载,进dashboard)
可信bench(seed42/全共享1024前缀/128输出/ignore_eos/REQS48,两端同负载),test/kaon3/2026-10-27/e2e_vs_vllm:
| rate=8 | 完成 | 吞吐tok/s | TTFT p50 | TPOT p50 | cache |
|---|---|---|---|---|---|
| **dinfer(splitK4+mega+prefixcache)** | 48/48 | **357** | 3357ms | 40.2ms | 0.983 |
| **vLLM非eager** | 48/48 | **936** | 81ms | 8.8ms | — |
**诚实差距:dinfer 357 = vLLM 936 的38%(慢2.6×)**。TPOT 40 vs8.8(慢4.6×)、TTFT 3357 vs81(**慢41×**)。
**生产已含所有优化**(kserve_full.sh已设MEGA_SPGU=4/SPDN=4/KAON3_MEGA=1,split-K 14%已落地),这357就是优化后真实值,仍慢2.6×。
**差距拆解(下一步攻哪)**:
- **TTFT慢41×(3357ms最夸张)**:prefill排队/首token路径——rate8下48请求TTFT堆到3.3秒,远超vLLM 81ms。调度/prefill并发是最大短板,比decode更该先攻。
- TPOT慢4.6×:decode每步(关键路径+每stage效率,split-K已降14%仍差)。
**dashboard**:engine-trend/perf-trend.json已更新,bench_index生成test/index.html(8090 HTTP 401=需鉴权文件在,路径对)。数据已进。
**结论(给用户的真话)**:优化有效(split-K落地)但不够,端到端仍慢2.6×。最大短板是TTFT(prefill/调度)41×,不是decode kernel。下一步该攻prefill/调度而非继续抠GEMM/mega。

## ⚠️ R25 2026-10-27 路径澄清(选下一步前必读):那张profiling表是整层mega测试,不代表serve
查layer.rs:636-639/877-911:**KAON3_MEGA=1只让dense MLP支路(GATE/UP/DOWN)走megakernel**。serve真实decode一步=独立QKV GEMM → q/k/v_norm+rope → attention → 独立OPROJ GEMM → dense MLP megakernel(GATE/UP/DOWN融合)。
**→ 那张"QKV38.9%/OPROJ18.8%"表来自mega_layer_parity(整层15 stage全融的测试),serve不走整层mega**。QKV/OPROJ在serve里是**独立kernel**,不在megakernel内,那张表的exec/wait不代表serve真实时间分布。
**纠正选优化对象的依据**:
- GATE/UP/DOWN的split-K(MEGA_SPGU/SPDN=4)serve已用(走dense branch megakernel)。
- QKV/OPROJ是独立kernel,另一套代码(gemm路径),split-K/效率那张表管不着。
- 要选serve下一步优化,必须**在serve真实路径profile**(独立QKV/attn/OPROJ/dense mega各占多少),不是拿整层mega测试表猜。又差点优化错对象(整层mega非serve路径)。
**下一步(待定方向)**:在serve decode路径加分段计时(或用nsys),看独立QKV/attention/OPROJ/dense-mega真实占比,再定攻谁。端到端TTFT慢41×(prefill)可能比decode更该先攻(R24)。

## R26 2026-10-27 下一步=攻TTFT(主);wait优化项挂账(用户:别忘wait)
**主攻:TTFT慢41×(3357ms vs vLLM 81ms)**。端到端最大洞,rate8下48请求首token堆3.3秒,量级异常→大概率prefill调度/并发bug(非kernel微优化)。先查TTFT为何3.3秒:prefill请求是否串行/排队、chunked prefill是否生效、首token路径开销。
**wait优化项(挂账,别忘,用户强调)**——注意:下列来自**整层mega测试**,serve走dense branch非整层,serve真实wait待在serve路径核实后再优化:
1. DOWN 等 ACT(12.2×):ACT→DOWN衔接空隙
2. GATE/UP 各等 PREFF(8×):依赖已并行(都dep RES),wait是前段并行度不足→split-K已部分解(sp=4降14%);可各stage非均匀细调sp
3. ATTN 等 KVWR→QKNORM(11×):依赖链长,可缩
4. QKV 1.2×健康(不动)
5. split-K非均匀细调:各stage task数不同(QKV64/GATE17/DOWN22),分别扫sp_qkv/o/gu/dn找各自最优(非全4),可能再降
6. 根上:occupancy/CTA调度(但R6单提occ反慢,需和依赖一起治)
**顺序**:先TTFT(端到端大头)→回来按serve真实路径profile定decode侧攻谁(wait项)。CSV/dashboard子代理处理中。

## ★★ R27 2026-10-27 TTFT根因:并发槽位(max_num_seqs=16)太小+decode慢→排队堆积,非prefill慢
每请求TTFT:前5个=63-93ms(和vLLM一样快!)→线性涨→后5个=7500-8000ms。客户端=服务端自报(srv TTFT 3403 n=51),真堆积非压测端。
**机制(数据支持)**:
1. max_num_seqs=16(main.rs:45默认),vLLM默认256。dinfer最多16请求同时跑。
2. 每请求生成128token,TPOT40ms→占槽**5.1秒**才释放。
3. 48请求rate8涌入,16槽很快满,第17起排队等前面跑完(5秒级)→TTFT堆到8秒。
**vLLM TTFT平81ms**:槽多(256)+TPOT快(8.8ms占槽1.1s)→排空快,prefill不饿死。
**结论**:非prefill本身慢(首5个80ms不输vLLM)。是**并发槽位16太小 + decode慢(TPOT4.6×)致占槽久**双重叠加堆积。MEGA_ROWS=8不是硬限(超了回退逐op不卡batch)。
**验证(下一步)**:抬max_num_seqs(16→64/128)重测TTFT——若TTFT降=坐实槽位限流。但注意:decode>8行走非mega逐op路径(mega_dense只≤8),batch大可能decode变慢;需同时看TTFT降vs TPOT变化的净效果。真要追平vLLM还需降TPOT(decode侧,回到wait/split-K挂账项)。
**双因耦合**:TTFT和TPOT其实是一个问题——decode慢→占槽久→排队→TTFT高。降TPOT同时改善两者。所以decode侧wait优化(挂账项)其实也是TTFT的解,不是两件事。

## ★★★★ R28 2026-10-27 TTFT验证成功:max_num_seqs 16→64,TTFT降13.7×+吞吐涨51%
同锁死负载(seed42/全共享1024/128out/REQS48),仅改max_num_seqs:
| max_num_seqs | 吞吐 | TTFT p50 | TPOT p50 | cache |
|---|---|---|---|---|
| 16(原默认) | 357 | 3357ms | 40.2 | 0.983 |
| **64** | **540** | **245ms** | 59.4 | 0.976 |
| (vLLM非eager) | 936 | 81ms | 8.8 | - |
**TTFT 3357→245ms降13.7×**(坐实R27槽位限流堆积),**吞吐357→540涨51%**,TPOT略升(batch大decode每步多行)。
**对vLLM**:吞吐从vLLM的38%→**58%**,TTFT从慢41×→**慢3×**。**改一个配置参数(非代码)**,纯配置洞:serve默认max_num_seqs=16太小(vLLM默认256)致堆积。
**坑**:max_num_seqs=64需配套降max-model-len(16384→2048,否则KV池OOM)或加num-blocks;生产要按真实并发+序列长设KV池。num-blocks 8192+len16384直接OOM(32GB GPU)。
**dashboard/CSV**:子代理修好了——engine轮要存`engine_*`命名目录才被engine_trend.py自动扫(e2e_vs_vllm没匹配,加symlink补救);8090是缓存鉴权网关HTTP码不可靠(401↔404乱跳),磁盘文件在才是准的。
**下一步**:①max_num_seqs/KV池调到最优(64是否够?试128,配KV池)；②剩余差距(吞吐58%、TTFT3×、TPOT慢6.7×)——TPOT是decode侧,回到wait/split-K挂账项（R26）；③这轮进engine-trend入dashboard。

## R29 2026-10-27 攻TPOT(decode侧)前置:那张wait表不适用serve,需serve真实路径profile
用户再贴wait表(DOWN等ACT12×等),但**R25已澄清:表来自mega_layer_parity整层15stage全融测试,serve只融dense MLP(GATE/UP/DOWN),QKV/OPROJ/attention是独立kernel**。表里"QKV等IN/ATTN等KVWR"在serve里不是同kernel内stage,不能直接拿来优化serve TPOT。
serve真实decode一步(layer.rs:9-14注释)=qkv独立GEMM→q/k/v_norm+rope→paged attention→o_proj独立GEMM→**dense MLP与MoE并联**(h2=post_norm(moe(pre_norm)),dense一支+MoE一支,kaon3混合模型每层两支都算)。
**profile手段**:远端无nsys/ncu(只nvcc.profile)。只能代码内分段计时(CUDA event)——大改(serve层加event+重编重起)。
**TPOT现状**:seqs64下59.4ms(vLLM 8.8ms,慢6.7×)。但注意seqs64是batch大TPOT反升(seqs16时40ms),所以TPOT要在**固定batch**下比才公平。
**攻法(定清再动)**:①先在serve decode加分段CUDA event计时(qkv/attn/oproj/dense/moe各段),实测serve真实TPOT大头(不是拿整层mega表猜);②定位后针对最大段优化。先确认MoE那支——kaon3每层dense+MoE并联,MoE(128专家top8)可能是decode大头(exec重),这是整层mega测试没覆盖的(测试只测sliding dense层)。
**挂账(R26的wait项)**:仅对"整层mega"有效,serve不走整层,**降级为"若将来serve改走整层mega才用"**;serve当前TPOT优化以实测分段为准。

## R30 2026-10-27 最新exec/wait表(sp=4,用户要求重跑)——split-K重排了wait分布
当前sp=4代码重跑(wait/exec=5.70×,融合215us):
| stage | exec% | wait% | wait/exec | 等谁 | vs旧(sp=1) |
|---|---|---|---|---|---|
| GATE | 10.8 | **27.4** | 14.5× | PREFF | 17→27↑(新最大) |
| OPROJ | 18.3 | **24.2** | 7.6× | OQNT | 13→24↑(新第二) |
| QKV | 40.1 | 17.7 | 2.5× | IN | 9→18↑ |
| UP | 10.8 | 12.6 | 6.7× | PREFF | 17→13↓ |
| DOWN | 11.4 | 10.9 | 5.4× | ACT | 24→11↓↓(split-K最有效) |
| ATTN | 5.6 | 6.4 | 6.6× | KVWR | 15→6↓ |
**split-K效果**:墙钟251→215(-14%已验),wait分布重排——**DOWN 24→11%(sp_dn切得好)**,但GATE升27%、OPROJ升24%成新大头。
**exec绝对值(真算最重)**:QKV 40%、OPROJ 18%、GATE/UP/DOWN各~11%。QKV/OPROJ是算力大头。
**读法**:GATE wait14.5×空等PREFF(但PREFF exec=0?PREFF被fuse_ff吸进RES了,GATE实际等RES);OPROJ wait7.6×空等OQNT(量化)。
**注意**:仍是整层mega测试,serve只融dense MLP(R25/R29)。此表指导"整层mega"优化;serve TPOT另需serve路径profile。但GATE/OPROJ的wait升暴露split-K非均匀问题:全sp=4后GATE(n_gu=17 task)可能过切或与其他stage抢CTA。
**下一步候选**:①GATE/OPROJ的wait为何升(sp=4非均匀细调:各stage task数不同,统一sp=4未必对每个最优,R23挂账的非均匀细调);②serve路径实测(R29)定TPOT真大头。

## R31 2026-10-27 非均匀split-K扫描:全4已最优(215us),细调无额外收益,split-K到头
| 组合(qkv/o/gu/dn) | 融合us |
|---|---|
| **4/4/4/4(基线)** | **215.4** ← 最优 |
| 4/8/8/4(加大GATE/OPROJ) | 220.5↑ |
| 2/8/8/4(QKV减) | 220.6↑ |
| 8/8/8/8(全8) | 229.5↑ |
**全4已最优,所有非均匀方向都更慢**。加大GATE/OPROJ的sp反升。
**关键教训**:**wait%高≠加split-K能降**。GATE wait27%是"等前序RES"型,给它加sp只是更多task抢CTA更乱,无用。split-K只对"自己task少吃不满SM"的stage有效(DOWN当初如此,sp_dn=4治好),对"等上游"的stage(GATE等RES)无效。R30我以为GATE/OPROJ是split-K新目标,错——它们wait高是依赖等待不是并行度。
**split-K到头**:全4的215us(相对sp=1的251降14%)是split-K最优,非均匀无额外空间。**这条优化路结束**。
**剩余真问题(回到本质)**:GATE/OPROJ的wait是"等前序"——要降得缩依赖链或让前序更快,不是split-K。但⚠️仍是整层mega测试,serve不走整层。
**下一步(转serve真实路径)**:split-K已到头,整层mega测试的优化空间榨干(251→215)。serve TPOT的真大头必须在serve路径实测(R29:qkv独立/attn/oproj独立/dense与MoE并联)。不再在整层mega测试上调参。先给serve decode加分段CUDA event计时,看真大头(疑MoE 128专家)。

## R32 2026-10-27 GATE wait机制:等整个attention半层(真依赖),降GATE wait=优化它前面的QKV/OPROJ
用户要攻最大的GATE(wait27.4%/14.5×)。读结构:
- h_dep[S_GATE]=S_RES, GATE的fp8输入norm1_q由RES算(post_attn_norm+残差→x2→pre_ff_norm→量化)。
- RES等OPROJ等OQNT等ATTN...**GATE必须等整个attention半层(QKV→ATTN→OPROJ→RES)跑完**,因dense MLP输入x2是注意力残差输出。**真物理依赖,改不掉**。
- GATE有68 task(n_gu17×sp4),task最多且在链后段,68 CTA从启动自旋等前链→68×前链时间累加→wait最大。
**关键**:GATE wait**不是给GATE加优化能降的**(加split-K已证反而慢,R31)。它慢因为等前面attention半层。**降GATE空等=让它等的链更快=优化链上exec最重的QKV(40%)/OPROJ(18%)**。即"优化GATE的wait"="优化它前一步"(用户最初"前一步喂不饱后一步"正是此)。
**→ 攻GATE wait 实际 = 攻 QKV/OPROJ(attention半层的计算大头)**。QKV exec 40%是单个最大真算,它快→RES早出→GATE/UP早开工→它们wait降。
**下一步**:优化QKV(exec 40%,attention半层+GATE前链的最重计算)。但⚠️整层mega测试,serve QKV是独立kernel——serve的QKV优化和整层mega的QKV是**同一个fp8_gemm_tile代码**(mega_layer case S_QKV调fp8_gemm_tile,serve的QLinear也走fp8 GEMM),优化GEMM本身两边都受益。先看QKV GEMM(exec 40%)能否更快:它是ΣN=8192×H的大fp8 GEMM,之前R13测过它带宽健康(exec占比高是因为它真算量大=合并qkv)。

## ★★ R33 2026-10-27 决定性:serve的QKV/OPROJ走cuBLASLt(非fp8_gemm_tile),与vLLM同款库
读QLinear::forward(layer.rs:180/223):serve的fp8 GEMM**默认走forward_fp8_cublaslt(cuBLASLt)**,**不是**mega的手写fp8_gemm_tile。第37行注释明说"默认走cuBLASLt:实测vLLM的fp8 GEMM真实命中cuBLAS"。
**→ 我读半天的fp8_gemm_tile的m16浪费(mma m16n8k32但m=4只25%利用)只影响**mega dense branch(GATE/UP/DOWN)+整层mega测试**,不影响serve的QKV/OPROJ**(它们走cuBLASLt)。
**serve真实decode GEMM路径(终于分清)**:
- QKV/OPROJ = cuBLASLt(forward_fp8_cublaslt),与vLLM同库→大概率非serve瓶颈(vLLM也这速度)。
- GATE/UP/DOWN = mega dense branch的fp8_gemm_tile(手写,有m16浪费)——**这才是serve里可能慢于vLLM的手写部分**。
- MoE = 128专家top8(forward_nvfp4?)——整层mega测试完全没覆盖。
- attention = paged gqa。
**那张wait表(整层mega)和serve是两套GEMM路径**,一路困惑的根源。整层mega测试的QKV用fp8_gemm_tile,serve的QKV用cuBLASLt,不是一回事。
**重新定位serve TPOT真大头(必须serve路径实测)**:疑点转向①mega dense branch(fp8_gemm_tile m16浪费,serve走这条)②MoE(128专家,整层mega没测)。cuBLASLt的QKV/OPROJ大概率不是问题(同vLLM)。
**下一步**:serve decode加分段CUDA event(qkv-cublaslt/attn/oproj-cublaslt/dense-mega/moe),实测serve真大头。别再碰整层mega测试(serve不走)。

## ★★ R34 2026-10-27 坐实各算子量化+GEMM路径(用户质疑QKV是否真走cuBLASLt)
读hf_quant_config(MIXED_PRECISION)+QLinear::forward全+serve调用链(layer.rs:1021-1188):
| 算子 | quant | serve走的GEMM |
|---|---|---|
| QKV(q/k/v_proj) | **FP8** | cuBLASLt(默认;FP8_SELF_KERNEL=1才手写fp8_gemm_tile)=**同vLLM** |
| o_proj | FP8 | cuBLASLt |
| dense gate/up/down | FP8 | mega_dense_branch手写(mega_done真)或fallback cuBLASLt |
| **MoE experts** | **NVFP4** | **forward_nvfp4手写**(整层mega测试从没覆盖!) |
**用户质疑对**:QKV走cuBLASLt是默认forward的行为(203行self_kernel默认false),但要确认serve调forward非别的——1021/1023确实调forward/forward_preq(都cuBLASLt for FP8)。所以QKV确走cuBLASLt(R33对)。
**但MoE是NVFP4走手写forward_nvfp4(226行)**,不是cuBLASLt——serve每层都跑、整层mega测试完全没覆盖、手写kernel。
**疑点聚焦(排除同vLLM的cuBLASLt QKV/OPROJ后,serve慢在手写路径)**:
1. **MoE(NVFP4手写128专家top8)——最大嫌疑,从没profile过**
2. dense mega_dense_branch(FP8手写,m16浪费)
**下一步**:serve decode分段计时,重点量MoE vs dense vs attention vs cuBLASLt-QKV各占TPOT多少。MoE最可疑(手写+128专家+没测过)。别再碰整层mega测试(QKV用fp8_gemm_tile但serve用cuBLASLt,两回事)。

## R35 2026-10-27 serve decode完整分段(读全forward)+计时方案
serve decode forward(layer.rs:1000-1206)完整序列:
1. QKV(FP8 cuBLASLt,1021/1023)→ 2. q/k/v_norm+rope → 3. paged attention(gqa) → 4. o_proj(FP8 cuBLASLt,1157) → 5. post_attn_norm+残差 → 6. **dense MLP支路**(1167 mega_dense_branch[FP8手写]或fallback) → 7. **MoE支路**(1195 moe_branch,与dense并联) → 8. 两支相加+post_ff_norm+残差。
MoE内部(1368):MOE_GROUPED=1(serve已设)走**device-grouped**(非逐专家循环1429);关时逐专家(top8→24个小GEMM串行)。MoE专家=NVFP4手写。
**计时方案(下一步,完整子任务)**:serve forward各段插CUDA event(qkv/attn/oproj/dense/moe),env开关`KAON3_STAGE_TIME=1`门控(默认关零开销),decode步累加各段ms,serve退出或/metrics打印。重编重起,锁死负载测,看MoE vs dense vs attn vs cuBLASLt-QKV各占TPOT。
**已确立事实(别再推翻)**:
- QKV/o_proj走cuBLASLt同vLLM(非瓶颈嫌疑低)
- dense gate/up/down走mega手写(m16浪费,FP8)
- MoE走NVFP4手写grouped(128专家,整层mega测试0覆盖)=最大嫌疑
- 整层mega测试/那张wait表≠serve路径(serve QKV用cuBLASLt非fp8_gemm_tile)
**已落地优化**:①max_num_seqs16→64(TTFT 3357→245降13.7×,吞吐357→540)②split-K sp=4(dense mega 251→215,但那是整层mega测试,serve dense branch是否受益待确认)。
**下一步**:加KAON3_STAGE_TIME分段event,测serve真实TPOT大头。

## ★★★★★ R36 2026-10-27 serve真实decode分段实测:MoE占63%=TPOT真大头(整轮最关键)
serve真实decode一层(tok=1,KAON3_STAGE_TIME=1,8次稳定):
| 段 | ms | 占比 |
|---|---|---|
| **MoE** | **0.397** | **63%** |
| attn(qkv/norm/rope/paged) | 0.122 | 19% |
| dense MLP(gate/up/down) | 0.092 | 15% |
| oproj | 0.020 | 3% |
| **total/层** | **0.63** | 100% |
**MoE一个人占decode每层63%!** 它=NVFP4手写kernel+128专家top8+**整层mega测试0覆盖+前十几轮完全没碰的那块**。我全程抠attn/GEMM(占19%+15%)完全打偏。
**这是整轮排查最有价值结论**:decode TPOT大头是**MoE(63%)**,不是QKV/attention/dense/整层mega那些我反复折腾的东西。attn+oproj+dense才37%。
**与vLLM差距的真源头基本锁定**:vLLM的MoE(fused grouped GEMM/cutlass)很可能远快于我们的NVFP4手写grouped。TPOT慢6.7×,MoE占63%——优化MoE是把TPOT拉近vLLM的唯一大杠杆。
**下一步(终于对了)**:profile MoE内部(moe_branch,MOE_GROUPED路径)——128专家NVFP4 grouped GEMM慢在哪:①是否每专家小GEMM串行②NVFP4量化开销③grouped kernel效率④对比vLLM MoE实现。这才是该花力气的地方。
**教训总结**:十几轮绕(GEMM 8×/occupancy/split-K/整层mega/那张wait表)全因没先在serve真实路径分段实测。一个KAON3_STAGE_TIME分段,立刻看出MoE 63%。用户"谁慢解决谁"+"读全代码"+"serve不走整层mega"全是对的,我该一开始就分段实测。

## ★★ R37 2026-10-27 MoE瓶颈机制:grouped_gemm grid.z=128全专家,decode只激活8个→94%block空载
读moe_grouped_sm120.cu:112-123:grouped_gemm grid=(N/BN, ceil(R/BM), **E=128专家**),blockIdx.z=e,空专家(row_lo==row_hi)第123行直接return。
**decode R=8行(top_k=8)只激活≤8专家→120/128=94% block启动后立即空载return**。grouped_gemm跑3次(gate/up/down),grid含N/BN×128,N=moe_inter大→海量空block启动开销。
**与vLLM差距对上**:vLLM MoE只给激活专家起block,我们给全128个→decode 94%浪费。
**但需确认(别假设)**:stage-time只测MoE整体0.397ms,没细分grouped_gemm vs router/bucket/gather/quant×2/gelu/scatter(MoE有~10 kernel)。grid.z=128浪费是确凿机制,但是否MoE大头待细分。
**下一步**:moe_branch内部加细分计时(router/gather/quant/3×gemm/scatter各段),确认grouped_gemm(grid.z=128)是否MoE大头;若是→改grid.z=激活专家数(或compact offsets只给非空专家起block),对齐vLLM。这是把TPOT拉近vLLM的核心杠杆(MoE占63%,其中grid浪费94%)。

## R38 2026-10-27 坑:event计时的sync与cudagraph capture冲突,serve崩;MoE细分要换法
加KAON3_MOE_TIME后serve崩:`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`。根因:**serve decode走cudagraph,capture期间任何stream.synchronize()非法**。我的event计时(record+sync读elapsed)与graph冲突。
**反思R36的0.397ms MoE数据**:stage-time能出数是因serve首次decode(graph未捕获)跑了几次,捕获后崩——那0.397ms是**非graph路径**的,可能偏高(graph省launch开销)。但MoE占63%的**相对结论仍成立**(attn/dense同样非graph,相对比例可比)。
已关MOE_TIME/STAGE_TIME,serve恢复UP。
**MoE细分正确测法(graph兼容)**:①用NO_GRAPH=1跑serve(禁graph,event sync合法),测MoE细分——但那不是生产路径;或②在mega_mlp_parity式的独立微基准里测MoE grouped_gemm(不经serve/graph);或③grid.z=128浪费是**读代码确凿的机制**(R37),不一定非要精确计时——可直接改grid试(只给激活专家起block),重测stage-time MoE整体降没降,降了就坐实。
**下一步**:走③最直接——改grouped_gemm grid.z从128(全专家)到激活专家数,重测MoE整体ms(用NO_GRAPH或首次decode窗口),降了=坐实grid浪费是大头。先读grouped_gemm host launch怎么设grid.z。

## ★★ R39 2026-10-27 MoE grid.z=128根因+优化思路:改按行块分配(graph友好且不空载)
读moe_grouped_sm120.cu:259-265:grid=(N/BN, ceil(R/BM), num_experts=128),grid.z=128固定是**为cudagraph友好**(graph要求grid维度固定,不能动态=激活专家数)。decode R=8只激活≤8专家→120专家block空载(N/BN×1×128,94%白起)。
**两难**:decode要cudagraph(固定grid)→grid.z固定128→94%空载。vLLM不用persistent-expert graph故无此问题。
**优化思路(graph友好且不空载)**:把grid从(N/BN, ceil(R/BM), 128)改**(N/BN, ceil(R/BM))二维,每block用blockIdx.y的行块查row2e定位专家**——**按行块分配而非按专家分配**。R=tokens×top_k固定(=8)→grid固定→graph友好,且**只起R行需要的block(8行→1-2个BM块),不起128专家**。block数从 N/BN×128 降到 N/BN×ceil(R/BM),decode降~64×(128→ceil(8/BM))。
前提:row2e[r]已有(moe_row2expert算,1400行),每行知道它的专家id→block按行段定位专家的B/S/alpha。需改grouped_gemm_kernel:blockIdx.z去掉,改用row2e[m_blk起始行]查e。
**风险**:一个BM行块可能跨两个专家(边界),需按sorted保证同块同专家(buckets.sorted已按专家排序,BM对齐专家边界或块内判专家)。
**这是对齐vLLM的核心改动**,MoE占63%,grid浪费94%→理论可大幅降MoE。下一步:改kernel按行块分配,重测MoE整体(NO_GRAPH或首decode窗口)。先读row2e/sorted布局确认行块与专家对齐。

## R40 2026-10-27 按行块分配受阻(BM跨专家),改思路:grid.z压到激活专家上界
读BM=32,decode R=8<BM→每专家段≤1个BM块。但**offsets非BM对齐(专家段紧挨)**→按全局行块切会跨多专家(decode 8行分属8专家挤一个BM=32块),而kernel要加载单专家权重B[e]→**不能简单按行块分配**。R39思路受阻。
**更可行思路**:grid.z不必=128。decode R=tokens×top_k,**激活专家≤min(128,R)**。decode tokens≤max_rows(serve 8或64),R≤max_rows×8,**激活专家数有小上界**。cudagraph在decode固定batch捕获→**grid.z可固定=该batch的激活专家上界(远<128)**,而非恒128。
但激活哪些专家是动态的(路由决定),grid.z=激活数还需**compact offsets**(把非空专家排前面,grid.z=非空数)。这要在bucket阶段算compact映射。
**权衡**:①compact方案(grid.z=激活数,需bucket改)最省但复杂;②简化:grid.z=固定上界(如min(128, max_rows*top_k)),decode max_rows=8→上界64,仍比128省一半,且无需compact,graph友好。先试②(改动小)看MoE降多少,不够再上①。
**但⚠️**:grid.z压到64,哪64个?仍要映射。最简:**grid.z=128不变,但让空专家block更早退出/不参与**——其实第123行已经return了,空block开销只是launch+return。真问题是**launch 120个空block的开销**有多大?需实测:grid.z=128 vs 人为减小 的MoE时间差,才知道空block launch是不是真瓶颈(可能return很快、开销小,MoE慢另有原因如量化/scatter)。
**下一步(先验证再改)**:别急着改kernel。先确认"94%空block"到底占多少时间——用NO_GRAPH跑serve+KAON3_MOE_TIME看gemm3 vs scatter+rest占比,或直接grid.z临时改小测MoE变化。先读MoE是不是真卡在grouped_gemm(R38没测出,sync崩了)。

## ★★★ R41 2026-10-27 反转:MoE大头是前处理(router/bucket/gather/quant 74%),非grouped_gemm(24%)
NO_GRAPH下MoE细分(r=8,event sync合法):
| MoE内部 | ms | 占MoE |
|---|---|---|
| MoE整体(stage-time) | 0.408 | 100% |
| 3×grouped_gemm(grid.z=128那个) | 0.098 | **24%** |
| scatter+rest | 0.008 | 2% |
| **router/bucket/gather/quant(gemm前)** | **0.302** | **74%** |
moe-time只量"quant后→scatter"=0.106,MoE整体0.408→**前面一段0.302ms(74%)是真大头**:router(moe_gate GEMM+softmax+top-k)+moe_bucket+pre_ff2_norm+moe_gather+moe_grouped_quant。
**又差点改错地方**:R37-R40花4轮分析grid.z=128空载,但grouped_gemm才占MoE 24%!改grid最多省24%的一部分。真大头74%在**前处理**。幸好R40刹住先验证。
**教训(再次)**:grid.z=128空载是读代码看到的"显眼机制",但实测它只占24%。**显眼≠大头,必须实测**。我又被一个显眼机制带偏(像之前GEMM exec 40%、occupancy)。
**真大头74%前处理,下一步细分它**:router(含一个betgate GEMM到128专家logits+softmax+top8)/bucket/gather/quant各占多少。疑点:①router的moe_gate GEMM(N=128专家,小)②moe_bucket/sort(host?D2H?)③gather/quant kernel。细分前处理再定攻谁。先读moe_branch前处理那几步(1340-1405)看有无D2H/host往返(decode最怕D2H同步)。

## ★★★ R42 2026-10-27 坐实:MoE前处理0.29ms=MoE 74%=整层decode ~46%,头号矛盾
MoE 0.40ms分三段(NO_GRAPH实测稳定):pre(router/bucket/gather/quant)0.29ms(74%)、gemm3 0.096(24%)、scatter 0.008(2%)。
MoE占decode 63% × 前处理占MoE 74% = **前处理占整层decode ~46%,头号矛盾**。幸好R40/R41没冲动改grid(只动24%的gemm)。
前处理步骤(layer.rs):router GEMM(1335 forward_bf16到128专家)或moe_gate_f32(1339)→moe_route_softmax_f32(1359,softmax+top8)→moe_bucket(1377,排序分桶)→pre_ff_norm_2(1381)→moe_row2expert(1407)→moe_gather(1402)→moe_grouped_quant(1404)。
**下一步:一次性插4段event拆前处理**(router完/softmax+bucket完/gather+quant完),一轮测清谁大头。疑点:①router GEMM(R×128,小但到128专家)②moe_bucket(排序,可能host/原子)③moe_gather+quant(R=8行小应快)。
**注意**:decode最怕host往返/D2H同步。前处理主路径(非sub_out分支)读过没明显D2H,但moe_bucket内部实现待查(排序若有device-host往返就是元凶)。

## ★★★★★ R43 2026-10-27 最终定位:router=0.253ms=MoE 62%=整层decode~40%,TPOT头号元凶
MoE前处理4段拆解(NO_GRAPH实测稳定):
| 段 | ms | 占MoE |
|---|---|---|
| **router** | **0.253** | **62%** |
| sm+bucket | 0.025 | 6% |
| norm+gather+quant | 0.016 | 4% |
| gemm3(grid.z=128) | 0.096 | 24% |
| scatter | 0.008 | 2% |
**router占MoE 62%、整层decode~40%=TPOT头号元凶**。我前面全程(整层mega/GEMM/occupancy/split-K/grid.z=128)连边都没碰到。
**router做什么**:算token对128专家logits,layer.rs:1335 `gemm.forward_bf16(routed, router.bf16_weight)` 或1339 moe_gate_f32。形状[R=8,H=2816]×[2816,128]——**一个这么小的GEMM要0.25ms极不正常**(8×2816×128 MAC,纳秒级计算量)。必有问题:低效路径/隐藏同步/host往返/反复量化/权重每次load。
**下一步深挖router**:读1300-1340 router完整实现,查为何0.25ms:①forward_bf16走哪kernel(bf16 GEMM对128这种小N可能用了通用kernel极低效)②是否每次重新量化/load权重③per_expert bias/norm开销④有无to_vec/dtoh。这是把TPOT拉近vLLM的最大单点(省router可砍decode~40%)。
**教训汇总(整轮)**:从GEMM 8×绕到occupancy到split-K到grid.z,全是"显眼但非大头";唯有serve真实路径分段实测(R36→R43)一步步逼出router。**该一开始就分段实测,不是读代码猜显眼机制**。

## ★★★★★ R44 2026-10-27 router根因:BI persistent matmul单block单SM串行K=2816,为逐位牺牲全并行
读router_gemm_bi.cu:BI_BM=128/BI_BN=128,grid=(ceil(M8/128),ceil(N128/128))=**(1,1)=单block**!这1个block用8warp但M=8只需1warp,在**1个SM**上**串行走完K=2816(44 ktile)顺序累加**(为逐位复刻vLLM batch-invariant顺序累加口径),**其余169 SM全闲**。
→ 0.25ms=单block单SM串行44 K-tile。为逐位对齐vLLM BI牺牲全部并行度。
**BI默认开(ROUTER_GEMM_BI=1,layer.rs:1557)**,但kaon3-compile loop记过"生产不开BI"(用户2026-10-22 dinfer-kaon3-prod-config:生产vLLM不开BI,parity目标oracle_nobi)。**生产既然不要BI逐位,关掉走cuBLASLt(1336)就能并行化**——可能最快大收益(router占decode~40%)。
**验证**:ROUTER_GEMM_BI=0重测router ms(走cuBLASLt forward_bf16并行)。若router从0.25降到几us→坐实,且符合生产口径(不开BI)。

## ★★★★★ R45 2026-10-27 关BI验证成功:router 0.25→0.03ms(8.4×),整层decode 0.63→0.43(降32%)
ROUTER_GEMM_BI=0(走cuBLASLt并行)重测:
| | 关BI前 | 关BI后 | 降 |
|---|---|---|---|
| router | 0.253ms | **0.030ms** | 8.4× |
| MoE整体 | 0.408 | **0.19** | 2.1× |
| 整层decode total | 0.63 | **0.43** | **32%** |
生成通顺(中国的项目的活动...非乱码),质量OK。
**符合生产口径**:memory dinfer-kaon3-prod-config "生产vLLM不开BI,parity目标oracle_nobi"。BI(batch-invariant顺序累加)只为逐位对拍,生产不需要,**却默认开着(ROUTER_GEMM_BI=1)拖慢40% decode**。关掉=生产本该用的配置+大收益。
**根因总结**:BI router用persistent matmul grid=(1,1)单block单SM串行K=2816(为逐位),169 SM闲=0.25ms。cuBLASLt并行化=0.03ms。
**落地**:kserve_full.sh已加ROUTER_GEMM_BI=0。整层decode 0.63→0.43ms,TPOT应从~59ms(seqs64)相应降。需端到端bench确认TPOT/吞吐提升+质量(prompt_logprobs对拍oracle_nobi)。
**整轮排查闭环**:TTFT(max_num_seqs 16→64,13.7×)已拿;TPOT真凶=router BI默认开(关掉整层decode降32%)。两个都是**配置洞**(非代码重写),都是serve真实路径实测逼出来的。

## R46 2026-10-27 端到端关BI结果:吞吐540→573(+6%)/TPOT59.4→54.5(-8%),但远小于单层32%
锁死负载(seed42/全共享1024/128out/REQS48)dinfer全优化 vs vLLM:
| | 吞吐 | TTFT p50 | TPOT p50 |
|---|---|---|---|
| 起点 | 357 | 3357 | - |
| seqs64 | 540 | 245 | 59.4 |
| seqs64+关BI | **573** | 254 | **54.5** |
| vLLM非eager | 944 | 25 | 6.6 |
**端到端TPOT只降8%(59.4→54.5),远小于单层decode的32%(0.63→0.43)**。诚实落差。
**关键新发现**:单层decode 0.43ms×层数≈十几ms,但**TPOT 54ms远大于此**→TPOT大头不是单层GPU计算,是**每步host/调度/async/采样开销**。router优化只动了GPU算那块的一小部分。
→ **TPOT 54 vs vLLM 6.6(8×)的大头在每步host/调度开销,不在单层kernel**。这又是一层:我一直测单层GPU时间,但TPOT主要花在host侧(类似之前TTFT是调度非prefill)。
**下一步**:测TPOT的host vs GPU构成——每decode步的host时间(调度/采样/launch)vs GPU时间。疑点:async调度是否真重叠(之前R16-20绕过)、每步launch开销、采样/D2H。vLLM TPOT 6.6ms说明每步host开销极小,我们54ms必有大host开销。
**已确认的两配置洞收益(落地)**:max_num_seqs=64(TTFT 3357→254,13×)+ROUTER_GEMM_BI=0(单层decode降32%,端到端TPOT-8%)。吞吐357→573(vLLM的61%)。

## ★★★ R47 2026-10-27 实测TPOT构成(非推断):GPU实算仅24%,76%是host+排队
硬数据(engine_nobi summary,30层):
| 层级 | 时间 | 占TPOT |
|---|---|---|
| GPU实算(30层×0.43ms单层) | ~12.9ms | 24% |
| host decode步时(srv自报engine_step_execute_decode均值) | 27.1ms | 50% |
| 客户端TPOT | 54.5ms | 100% |
**逐级放大**:①GPU 12.9→host步时27.1=host多14ms(每步调度/launch/采样/finish等待)②host步时27.1→客户端TPOT54.5=又27ms(rate8/48请求/async流水的并发排队延迟)。
**GPU实算仅占TPOT 24%,76%是host+排队开销**。vLLM TPOT 6.6ms≈其GPU实算,说明vLLM几乎无这些host开销。
**确认=代码/架构问题非kernel慢**(用户要求测试确认):大头在host侧——每decode步调度/launch/finish等待(14ms)+并发流水效率(27ms)。
**→ 优化方向彻底转向host/调度**(不再是kernel):①每步host开销14ms(30层×0.47ms/层host?每层launch+同步开销大)②async调度流水是否真重叠(R16-20绕过,现在是大头必须解)。
**下一步**:测host步时27.1ms里——是30层各自的host launch开销累加,还是finish等待(async没重叠)。若per-layer host开销大→减per-layer launch(cudagraph应该已捕获?查decode是否真走graph);若finish等待大→async没重叠。先查decode是否真走cudagraph(走了host launch应极小,27ms host就反常)。

## R48 2026-10-27 bench序列(每轮CHANGE,用户要求至少到bench5)+清调试开关
清掉调试开关(NO_GRAPH/STAGE_TIME/MOE_TIME,它们让bench测不准:NO_GRAPH禁cudagraph decode更慢,event有开销),serve恢复生产态(graph开+MEGA_SPGU/DN=4+ROUTER_GEMM_BI=0+MOE_GROUPED=1+KAON3_MEGA=1)。
**bench序列(锁死负载seed42/全共享1024/128out/REQS48 vs vLLM,每轮改动说明)**:
| bench | 改动(CHANGE) | dinfer吞吐 | TTFT p50 | TPOT p50 | vs vLLM |
|---|---|---|---|---|---|
| bench1 起点 | max_num_seqs=16默认,未关BI | 357 | 3357ms | - | 38% |
| bench2 seqs64 | max_num_seqs 16→64 | 540 | 245ms | 59.4 | 57% |
| bench3 关BI | +ROUTER_GEMM_BI=0(router cuBLASLt) | 573 | 254ms | 54.5 | 61% |
| bench4/5 生产态 | 子代理跑:graph开+全优化干净态(之前带NO_GRAPH测不准) | 待测 | | | |
| 基准 vLLM非eager | - | 944 | 25ms | 6.6 | 100% |
**每轮改动已说明**(用户要求页面也显示CHANGE,子代理改engine_trend.py让html显示)。
**整轮排查结论(供收口)**:两个配置洞落地(seqs64→TTFT13×、关BI→decode32%),吞吐357→573(vLLM61%)。但**TPOT 54.5 vs vLLM6.6(8×)真源头=host/调度开销(R47实测:GPU实算仅24%,76%是host步时+并发排队)**,非kernel。下一步该攻host/调度(查decode cudagraph是否生效、async是否重叠),不是继续抠kernel。
**dashboard坑(已知)**:8090被平台HF代理占(非我们的);真dashboard在8192(trend_server.sh);engine轮要engine_*命名才被engine_trend.py扫。子代理修到能访问。

## R49 2026-10-27 dashboard入口=本机perf-trend.html(非engine-trend),数据没进去
用户纠正:dashboard入口是 `http://192.168.72.130:8090/kaon3/perf-trend.html#kaon3/5090/single`(折线图,本机8090根=/data/code/dinfer/test,**本机就是192.168.72.130**)。
**之前所有"8090是5090的HF代理"判断对本机不成立**:本机8090是正常的test/ dashboard,一直在跑。看不到=①我给的URL是engine-trend.html,用户要的是perf-trend.html ②perf-trend.csv是旧bench1-3(kernel层splitK数据,bench_trend.py生成),没有我们这轮端到端r1-r5。
perf-trend.csv格式:round,date,change,group(c1/c8/c32),metric(out_tok_per_s等),dinfer,vllm,ratio,...;tab=single(URL #model/gpu/tab)。是4组闭环bench口径,与engine端到端(单rate8)不同。
**engine-trend已base64取回本机**(r1-r5在engine-trend.csv),但那是另一个页。
**下一步**:派子代理把r1-r5端到端数据按perf-trend格式进去(或确认该用哪个页),让用户那个折线图URL能看到5轮+CHANGE。

## R50 2026-10-27 重跑整层mega wait(用户要求)+诚实判断:对serve TPOT无用
全优化态(sp=4)整层mega exec/wait(与R30一致稳定):GATE wait27.7%(14.7×等RES)、OPROJ24.1%(等OQNT)、QKV17%(2.4×健康)、UP12.8%、DOWN11.2%。wait/exec=5.65×。
**wait可攻性**:GATE/OPROJ/UP的wait全是"等上游算完"真数据依赖(链后段等QKV→ATTN→OPROJ→RES),给它们本身加split-K已证无效/更慢(R31)。真能降wait=①让上游QKV更快(但serve QKV走cuBLASLt已不慢,整层mega的QKV用fp8_gemm_tile慢但serve不走)②缩依赖链(片上融合,175us<215已证有效)。
**最关键诚实判断**:**这张wait表是整层mega测试,serve不走整层mega**(serve只融dense MLP GATE/UP/DOWN,QKV/OPROJ cuBLASLt,MoE grouped)。在整层mega测试上抠wait对serve TPOT无用。
**serve真实TPOT大头(R47实测)=host/调度开销(占76%),GPU实算仅24%**,不是这张表的stage间wait。
→ **继续优化应攻serve真实路径的host/调度**(R47),不是整层mega测试的wait。wait这个方向在整层mega测试里已到头(片上融合175us是最优,单stage加split-K无效)。

## R51 2026-10-27 GATE wait根因算清:等QKV/OPROJ占满CTA,非等RES(RES快)
算账:GATE单task wait=216795cyc,前置链(IN→QKV→...→RES)总exec=6281026cyc。GATE 68 task(17×sp4)。
**矛盾解开**:GATE CTA不是从t=0等整条链(那样wait会~前置链)。持久kernel**轮转领task**:前面stage的task先被CTA处理,轮到GATE时前链大部分已算完,每GATE task只等"尾部+CTA空闲"。
**GATE wait 14.7×真相**:前链exec大头=QKV(40%)+OPROJ(18%)=58% exec,**它们占满170 CTA**,GATE task排后面领不到CTA→wait_stage自旋等CTA空闲+等dep signal。**不是等RES(RES exec仅0.9%很快),是等QKV/OPROJ这些大exec把CTA占着**。
**→ 降GATE wait = 让QKV/OPROJ(占满CTA的大exec)更快**,或GATE与之并行(有依赖不能)。回到:QKV/OPROJ是真算大头,它们快→GATE少等。
**但⚠️(反复提醒)**:整层mega测试里QKV用fp8_gemm_tile,**serve的QKV走cuBLASLt(已不慢)**。所以整层mega里"QKV占40% exec拖慢GATE"**在serve不成立**(serve QKV快)。整层mega的GATE wait问题serve没有。
**结论**:GATE wait在整层mega里的解=加速QKV/OPROJ(fp8_gemm_tile),但serve不走fp8_gemm_tile的QKV→**对serve无用**。整层mega若要独立跑(非serve),可优化其fp8_gemm_tile QKV。serve的TPOT仍回host/调度(R47)。

## R52 2026-10-27 深入读纠正(用户:别轻易下结论):serve dense branch确有独立wait,需实测
读mega_dense_branch(layer.rs:884-960):serve走`mega_mlp_dense_branch`=**6-stage dense megakernel**(NORM→GATE→UP→ACT→DOWN→POST),非整层15 stage。**它内部也有stage依赖wait**(GATE/UP等NORM,DOWN等ACT),与整层mega的GATE wait同构但独立(6 stage版)。
**纠正R51"serve GATE wait没有"**:错。serve dense branch里GATE/UP/DOWN一样有stage间wait。只是dense branch占decode仅15%(R36 stage-time dense 0.09/total 0.63),GATE wait在这15%内,量级远小于MoE 63%/router 40%。
**已有测法**:mega_mlp_parity测的正是dense branch 6-stage(=serve真实走的),有NORM/GATE/UP/ACT/DOWN exec/wait打印(R26见过)。**这才是serve真实dense wait,不是整层mega**。但R38记过dense branch测试曾全零(需确认当前能跑)。
**用户要求的正路**:先调研测试serve真实路径的wait(dense branch用mega_mlp_parity;MoE/router已测=router 0.25ms最大但已靠关BI降到0.03),把serve各部分wait摸清,再决定转host/调度。
**下一步**:跑mega_mlp_parity看dense branch(serve真实)的GATE/UP/DOWN wait实际多少;若dense branch wait可观且能降(片上融合/依赖),就在serve有收益(不同于整层mega对serve无用)。先确认mega_mlp_parity当前能跑(非全零)。
