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
