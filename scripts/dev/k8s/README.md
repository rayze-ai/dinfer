# k8s 4090 环境使用

集群 `admin@polaris`（kubectl 客户端 v1.31，服务端 v1.36）。

| 节点 | GPU | 备注 |
|---|---|---|
| `host-10-121-192-44` | 8 × RTX 4090（sm_89，24564 MiB），驱动 580.178.04 | 已探测确认 |
| `host-10-121-192-98` | 8 × RTX 4090（同款，同驱动） | 已探测确认 |
| `host-10-121-192-182` | 无 | control-plane |

- 容器运行时 containerd 2.3.5 + **RuntimeClass `nvidia`**：Pod 必须写 `runtimeClassName: nvidia`，否则容器里没有 `/dev/nvidia*` 和 `nvidia-smi`（已踩坑）。
- device plugin v0.17.4，带 MPS control daemon；`NVIDIA_DRIVER_CAPABILITIES=compute,utility`。
- **无 StorageClass**：没有持久卷，Pod 内只能用 emptyDir。
- 节点 allocatable：cpu 127 核、内存 ~503 GiB、ephemeral ~846 GB。
- 镜像仓库：集群内用 `ppio-harbor-registry.tencentcloudcr.com`、`k8s.nju.edu.cn` 等镜像站；`nvidia/cuda:12.4.1-*` 直接从 Docker Hub 拉成功。

| 文件 | 用途 |
|---|---|
| `gpu-probe.yaml` | 一次性探测 GPU 型号 / 算力 / 驱动 |
| `dev-pod.yaml` | CUDA 12.4 devel + Rust 1.89 的开发 Pod，1 卡，用于跑 `scripts/verify/*.sh` 的 GPU 部分 |

规则：
- 验证用完即删（`kubectl delete pod dinfer-dev`），不占卡。
- 源码用 git 同步进 Pod，不把 Pod 当开发盘。
- 本机 3090（sm86）做日常开发与 CPU 测试；4090（sm89）用于 FP8 相关 kernel 的真算验证。
