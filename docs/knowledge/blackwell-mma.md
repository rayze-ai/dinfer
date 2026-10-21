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

