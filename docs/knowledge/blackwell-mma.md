# Blackwell（sm_100/103/120）低精度 MMA：编译、布局、实测数字

> 从 `kernel-rules.md` 拆出（2026-10-21，该页 151 行超限）。这里只放 sm_100/103/120 专属事实；
> 通用 kernel 规则见 [kernel-rules.md](kernel-rules.md)。

**架构专用指令必须显式 gencode**：`-arch=sm_120a` 生成的 PTX 仍是 `compute_120`，
`tcgen05` / `mxf4nvf4.block_scale` 会报 `not supported on .target`。
写 `-gencode=arch=compute_120a,code=sm_120a`。sm_103 上块缩放 FP4 **不可用**，sm_120a 可用。

**NVFP4 块缩放 MMA**（`m16n8k64`，`scale_vec::4X`）：

```ptx
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
  {d0,d1,d2,d3},{a0,a1,a2,a3},{b0,b1},{c0,c1,c2,c3},{sa},{bidA,tidA},{sb},{bidB,tidB};
```

- `.kind::` 与 `.block_scale` 在 `.m16n8k64` **之前**；scale 后要跟 `{bid,tid}` 两个 u16
- A/B 每线程 4/2 个 u32，按列主序 `lin = m + 16*k` 取值，每组 8 个 nibble 沿 k 连续 → 可单条 u32 读
- 一个 32-bit scale 装 **4 个 16-元素块的 e4m3 字节**，k 小到大从低位排
- 输出同 SM80_16x8_Row

**FP8 e4m3**（`m16n8k32`）在 sm_120 / sm_120a 都支持，标准 `ldmatrix.x4`(A) / `.x2`(B) 布局。

**吞吐参考（5090 实测，非理论峰值）**：NVFP4 502 TF/s、FP8 292 TF/s（均含 ldmatrix + padding）。
smem padding 16 B 对 NVFP4 有 **+43%**；同样的 cp.async 双缓冲在 NVFP4 上**反而 −38%**（occupancy），
在 FP8 上 +173%。**双缓冲不是无条件的优化，要按 kernel 实测**。

**硬件常量（5090，实测）**：空 kernel launch 2.2 µs；持续 HBM 读 **1458 GB/s**（标称 1790）；
L2 命中可达 5159 GB/s。小矩阵 decode 往往是 launch 开销主导，应把多个专家/矩阵合并成一次 launch。

# 附：从 perf-method 移来的两条（2026-10-21 r12a）

## 13. "等访存"占大头时，先查访存是否合并，再想 TMA / 加深流水（2026-10-21 r12a）

**判断**：kernel 里"等装载"占比很高、但算出来的总流量又远低于 HBM 带宽时，**不是带宽问题、也不是 outstanding 不够**，
八成是**相邻线程访存不合并**——每个 16 B 请求独占一条 128 B cache line，L2 流量放大 8×，带宽全浪费在无用字节上。

**证据**（attn tcgen05，B300 GPU 5）：

| | 等 K/V 占比 | QK 发射 | softmax | T=6019 |
|---|---|---|---|---|
| 合并前（相邻线程读相邻 key，地址相隔 `kv_stride=16 KB`） | **91%** | — | 22% | 5158 µs |
| 合并后（相邻线程读同一 key 的相邻 16 B 片） | **3%** | 67% | 34% | 3311 µs |

**怎么改**：让相邻线程覆盖**同一行内的连续字节**，而不是跨行。
例：一个 key 的 256 维 = 32 个 16 B 片；64 key × 32 片 = 2048 片 / 128 线程 = 每线程 16 片连续，
这样一条 128 B cache line 被 8 个线程一次吃满。

**为什么比 TMA 优先**：TMA 解决的是"outstanding 请求数不够"，那是**第二个**嫌疑；
合并问题用普通 `cp.async` 就够了，改一行索引逻辑，不需要 mbarrier 改造。先量再选，别直奔复杂方案。

**顺带的一条**：判断"带宽受限 vs 延迟受限"要用**有效吞吐**比：算 kernel 实际搬的字节 / 实际耗时，
得到一个 GB/s，和实测流式上限（B300 约 6 TB/s）比。远低于它 → 延迟受限。
r12a 一开始算成 3.5 GB / 3.5 TB/s 就 1 ms，是把 L2 流量当成了 HBM 流量（全部 K/V 其实只有 24.7 MB）。

## 14. 1 CTA/SM 的 UMMA 结构对"短 K、频繁 commit"的 GEMM 不利（2026-10-21 r12a）

`umma_peak` 能到 2236 TF/s（标称 99.4%），是因为它 K 方向长、连续 4096 轮把发射开销摊掉。
attn 的 QK 是"16 条 K=16 全部累加到同一个 TMEM 累加器 → 必须串行 → commit + 等"，每 tile 都付一次发射 + 同步，
**1 个 CTA 只有一条 MMA 发射流，没有第二条填发射间隙** → 每 tile 8.9 µs vs 理想 0.56（16×）。

**用法**：给 GEMM 选结构时先看"发射/依赖比"——K 越长、commit 越少，越适合 1 CTA/SM 的 UMMA。
GDN 的 chunk 内 GEMM（K=128 连续、state 现成在 smem）属于好的那一类；attention 的 QK/PV 属于坏的那一类。

# 附：从 kernel-rules 移来的指针段
## Blackwell 低精度 MMA（sm_100/103/120）
架构专用指令的 gencode、NVFP4/FP8 指令形式与寄存器布局、5090 实测吞吐与硬件常量，全部在 **[blackwell-mma.md](blackwell-mma.md)**。动 `crates/kernels` 里任何 sm_100+ 的 MMA 前先读它。

