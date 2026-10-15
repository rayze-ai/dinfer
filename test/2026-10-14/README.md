# 2026-10-14 测试记录

| 文件 | 内容 | 结论 |
|---|---|---|
| `k8s-gpu-probe.log` | `host-10-121-192-44` 上 `nvidia-smi` 输出 | RTX 4090，sm_89，24564 MiB，驱动 580.178.04 |
| `k8s-gpu-probe-98.log` | `host-10-121-192-98` 同上 | 同款 RTX 4090，同驱动 |
| `build-glm53.log` | `cargo build --workspace --features glm53` | 第 1 次：cc 把 gcc 参数塞给 nvcc；第 2 次：bin 缺 axum 依赖（改为 `dinfer_server::serve`）；第 3 次见下方 p0 日志 |
| `p0.log` | `scripts/verify/p0.sh` 全量输出 | **P0 PASS**：fmt / build ×2 / clippy / 9 CPU 单测 / GPU add kernel / device-info 全过 |
| `p1.log` | `scripts/verify/p1.sh` 全量输出 | **P1 PASS**：p0 + 27 个 core 单测 + criterion 门禁（1024 decode step：pc=off 95 µs，pc=on 146 µs，< 200 µs） |
| `kernels.log` | `scripts/verify/kernels.sh` 中间几轮 | 路径 bug → 8/10 → top-p 口径修正 |
| `p2.log` | `scripts/verify/p2.sh` 全量输出 | **P2 PASS**：p1 + 10 个算子 PyTorch 对拍（rmsnorm / fused_add_rmsnorm / rope×2 / silu_and_mul / embedding / gemm / argmax / topk_topp_mask / sample 分布）全过 |
| `p3.log` | `scripts/verify/p3.sh` 全量输出 | **P3 PASS**：p2 + tiny GLM（2 层 MLA + dense MLP，HF DeepseekV3 同构）prefill logits max abs 0.004 / 32 步 decode logits max abs 0.007 / greedy 仅在 bf16 平局步（gap 0.0039）分叉 |
| `p4.log` | `scripts/verify/p4.sh` 全量输出（最终版，含 fp8 步） | **P4 PASS**：p3 + tiny MoE GLM 3 对拍 + runner 2 单测 + runner 双请求对拍 plain/chunked × dense/MoE + tiny fp8 GLM 3+2 对拍 |
| `p4_runner_parity_{dense,moe}.log` | Scheduler+Runner+Glm53Executor 2 并发请求逐步对拍（plain：1 prefill + 31 decode；chunked/mixed：预算 8 → Prefill 1 + Paged 3 + Decode 31 = 35 步） | plain：dense worst 0.0063 / MoE 0.0071；chunked：dense 0.0059 / MoE 0.0078；greedy 仅在 HF 平局步（gap ≤ 0.0059）分叉 |
| `p4_fp8_parity.log` | tiny fp8 GLM（128×128 块量化 e4m3 权重，加载期解量化）同套 3 + 2 对拍 | **P4-d PASS**：prefill 0.0076 / 32 步 worst 0.0073 / runner plain 0.0077 / chunked 0.0078；greedy 仅平局步分叉 |
| `../tiny_glm_fp8/` | `gen_tiny_glm.py --first-k-dense 1 --fp8` 产出（不进 git） | 投影权重 F8_E4M3 + `weight_scale_inv` F32，与 GLM-5.3 真权重同格式；期望值用解量化权重算 |
| `p4_moe_parity.log` | tiny MoE GLM 对拍 `--nocapture` 输出 | prefill logits max abs 0.0069（3 行 bf16 平局行 argmax 不同、HF gap ≤ 0.002）；逐步 decode / greedy 见文件 |
| `../tiny_glm_moe/` | `gen_tiny_glm.py --first-k-dense 1` 产出（不进 git） | 第 0 层 dense、第 1 层 MoE（4 专家 top-2 + 1 shared，e_score_correction_bias 随机） |
| `../tiny_glm/` | `gen_tiny_glm.py` 产出（不进 git） | config.json、model.safetensors（bf16 2.9 MB）、prompt、expect_prefill_logits、expect_step_logits、expect_greedy |
| `../kernel_refs/` | `gen_kernel_refs.py` 产出的 `.npy`（不进 git） | 输入与期望，29 个文件 |

环境：本机 RTX 3090 / CUDA 12.4 / rustc 1.89.0；k8s `admin@polaris`。
