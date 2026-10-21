# Kaon-V3 kernel 实测（2026-10-21，RTX 5090 / sm_120）

结论放最前面；原始命令在文末。

## 结论

| 粒子 | 吞吐 | 数值 |
|---|---|---|
| NVFP4 `m16n8k64` block-scale（4096³） | **502.6 TF/s** | bad=0/4096 |
| NVFP4 decode（合并 8 专家，M=1） | **1930 GB/s**，8.2 µs | — |
| FP8 e4m3 `m16n8k32`（4096³） | **292.4 TF/s** | bad=0/16384 |

## 各版本演进（用来定位下一步优化点）

NVFP4：21（无 smem）→ 80（smem + 逐字节解包）→ 351（u32 直读 + 修 row stride）→ **502（smem padding 16 B）**；
cp.async 双缓冲**退化到 313**，已弃用。
FP8：107（标量 smem）→ **292**（ldmatrix.x4/.x2 + padding + cp.async 双缓冲）。

## 硬件常量（`lo.cu`）

```
空 kernel launch   2.2 µs
2 MB 读            2.45 µs  → 1631 GB/s
32 MB 读           12.4 µs  → 5159 GB/s（L2）
256 MB / 1 GB 读   1458 GB/s（持续 HBM）
```

## 复现命令

```bash
# 全部需要显式 gencode
nvcc -gencode=arch=compute_120a,code=sm_120a -O3 nvfp4_gemm5.cu -o ng5
./ng5 4096 4096 4096 30        # NVFP4
nvcc -gencode=arch=compute_120a,code=sm_120a -O3 fp8_v2.cu -o fv
./fv 4096 4096 4096 50         # FP8
```
源码在远端 `/hf_data/yaochen/work/`（`nvfp4_gemm5.cu`、`nvfp4_merged.cu`、`fp8_v2.cu`、`lo.cu`）。
