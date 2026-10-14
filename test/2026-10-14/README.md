# 2026-10-14 测试记录

| 文件 | 内容 | 结论 |
|---|---|---|
| `k8s-gpu-probe.log` | `host-10-121-192-44` 上 `nvidia-smi` 输出 | RTX 4090，sm_89，24564 MiB，驱动 580.178.04 |
| `k8s-gpu-probe-98.log` | `host-10-121-192-98` 同上 | 同款 RTX 4090，同驱动 |
| `build-glm53.log` | `cargo build --workspace --features glm53` | 第 1 次：cc 把 gcc 参数塞给 nvcc；第 2 次：bin 缺 axum 依赖（改为 `dinfer_server::serve`）；第 3 次见下方 p0 日志 |
| `p0.log` | `scripts/verify/p0.sh` 全量输出 | **P0 PASS**：fmt / build ×2 / clippy / 9 CPU 单测 / GPU add kernel / device-info 全过 |

环境：本机 RTX 3090 / CUDA 12.4 / rustc 1.89.0；k8s `admin@polaris`。
