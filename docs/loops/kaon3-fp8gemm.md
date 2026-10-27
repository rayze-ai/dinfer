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
