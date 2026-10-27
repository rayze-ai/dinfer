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
