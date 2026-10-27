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
