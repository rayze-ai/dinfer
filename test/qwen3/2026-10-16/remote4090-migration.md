# 4090 测试机迁移（2026-10-16）

## 结论（截至 12:50）

- 登录：JumpServer 公钥（本机 `~/.ssh/id_ed25519_gpu4090`）+ token 用户名，常驻 ControlMaster。token **密码**登录在第 2–4 个 token 上全部 `Permission denied`，第 1 个成功过；原因未查明，不再依赖密码。
- 目录约束：远程只用 `/data/yaochen/infer`。12.4 安装器自建的 `/usr/local/cuda` 软链已删（指向我装的 `cuda-12.4/`，不是机器原有的）。
- **机器原本没有 CUDA toolkit**（审计：dpkg 无 cuda/toolkit 包；`/usr/local` 下只有 2024 年的基础目录，`lib`/`bin` mtime 10-12/10-13 为驱动相关；宿主无 nvcc/libcudart/libcublasLt；`find` 到的 cuda-13.0 / 12.9 / cu13 全在 containerd 镜像层 `/var/lib/containerd/.../snapshots/*/fs`，是容器内的）。`nvidia-smi` 的 `CUDA Version: 13.0` 是驱动能力。没有覆盖任何已有安装。
- 首次编译（CUDA 12.4，`CUDA_ARCHS=89`，`--features qwen3`）：**通过**，164 个 crate，含 nvcc 编译的 kernels。crates 走 rsproxy.cn（static.rust-lang.org 只有 ~100 KB/s）。
- 用户要求改用 CUDA 13.0 → `cuda_13.0.2_580.95.05_linux.run`（4.3 GB，下载 ~0.8 MB/s 直连，约 10 min）安装到 `cuda-13.0/`，进行中。cudarc 0.18.2 支持 `cuda-13000`；Cargo.toml 当前固定 `cuda-12040`，远端需按 13.0 编译（方案见 progress）。
- 权重 rsync：经跳板机约 0.4 MB/s，1.5 GB 预计 1 h+；`--partial --inplace` 可续传。golden / kernel_refs（76 MB）排在权重之后。
- **CUDA 13.0.2 编译通过**（13:0x，`BUILD13_EXIT=0`）。两处改动：① cudarc feature 改 `cuda-version-from-build-system`（固定 `cuda-12040` + `CUDARC_CUDA_VERSION=13000` 会 E0428 重定义）；② `kernels/build.rs` 加链 `stdc++`（nvcc 13.0 host stub 引用 `__cxa_guard_*`）。本机 3090 / 12.4 两处改动后仍过。12.4 误装已删。
- **kernels ref 测试 16/16 PASS**（4090 / sm_89 / CUDA 13.0，2.0 s；含 Q2-a 的 4 个采样 kernel）。踩坑：toolkit 不在 /usr/local 时运行要 `LD_LIBRARY_PATH=$CUDA_HOME/lib64`（已进 env.sh）。
- 权重（1 503 300 328 B，与本机一致）+ golden（11 文件逐个同大小）rsync 完成 12:36。CPU 单测：core 31 / qwen3 8 / server 9 全过。
- `remote4090_baseline.sh` 首跑 4 步全 rc=127：脚本没 `source env.sh`（cargo 不在 PATH）→ 已修（开头 source）。重跑结果见下。
- **`remote4090_baseline.sh` PASS（exit 0，13:1x，CUDA 13.0 / sm_89）**：kernels 16/16（2.0 s）、real_parity 3/3（67 s；3090 上 65 s）、sampling_parity 5/5（193 s；3090 上 173 s）、core 31 / qwen3 8 单测。与 3090 / CUDA 12.4 的结果逐项一致 → 迁移基线成立。server e2e 单独补跑，结果在本文件末尾。

## 原始记录

- 探测：8 × RTX 4090 24 GB，驱动 580.178.04，Ubuntu 22.04.5，128 核 / 503 GB，`/data` 7 TB。
- 网络 GET（直连）：crates.io 206 / github 206 / rustup 206 / nvidia 301→200 / huggingface 000；代理 `127.0.0.1:1081`：hf 307、github 0.5 MB/s、nvidia 0.8 MB/s。
- build.log：`BUILD_EXIT=0`，`Compiling` 164 行，无 error。

## 补跑：server e2e（13:2x）

- `serve_e2e_qwen3::e2e_qwen3_real_model_chat` PASS（20.8 s；3090 上 20.3 s）。至此 q1.sh 对应的全部 GPU 项 + Q2-a sampling_parity 在 4090 / CUDA 13.0 上全过。
