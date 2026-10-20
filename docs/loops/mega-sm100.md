# 质变实验第一阶段：Qwen3.5 单 GDN 层 decode persistent kernel（design.md §14.3，用户 2026-10-20 定"实验和 K3 同步进行"）

> 目的：**只拿数字**，验证"消掉 kernel 边界"在 B300 上值多少；≥ 25% 才进第二阶段（任务图架构，§14.5）。不改 executor / 不接服务。

## 环境
- B300-50 GPU 4–7（`B300_TAG=50 scripts/dev/remote_b300.sh`，远端 `source /data/yaochen/infer/env.sh`，`CUDA_DEVICE_ORDER=PCI_BUS_ID`）；GPU 0/1 Qwen3.5 常驻、2/3 K3 子代理用。本机 3090 只做编译 / 对拍（sm86 无 TMA，不测性能）。
- 代码：`crates/kernels/cuda/mega_sm100.cu`（Blackwell 专用；按分文件规则不写 sm8x 版），C ABI `mega_gdn_layer_decode_sm100(...)`，Rust `ops::mega_gdn_layer_decode`；测试 `crates/kernels/tests/mega_parity.rs`（对拍 + 计时）。
- 真权重形状：hidden 5120、in_proj N=16480（qkv 10240 | z 6144 | a 48 | b 48）、conv W=4、HV=48 / H=16 / K=V=128、out_proj 6144→5120、gate_up N=34816、down K=17408；B ∈ {1, 8, 32}。权重用随机 bf16（形状对就行，不加载真权重）。

## 观察方法
- 基线：现有路径该层 decode 的 **11 个 kernel**（rmsnorm → in_proj GEMM → conv1d_update → recurrent_decode(fused gate) → gated_rmsnorm → out_proj → fused_add_rmsnorm → gate_up → silu_and_mul → down → add）在 B300 的 nsys node 级总和（含 kernel 间隙）：现在 B=1 ≈ 0.14 ms/层、B=32 ≈ 0.21 ms/层（取自 c1 / c32 decode 步 ÷ 64 的 GDN 层份额，实验开头先精确测一次写进记录）。
- 实验：同输入、同随机权重，persistent kernel 一次 launch 的耗时（CUDA event，1000 次均值）；输出与基线 bf16 对拍 cos ≥ 0.9999 / max rel ≤ 1e-2。
- 中间量：每阶段的 SM 利用率（nsys / ncu）、依赖等待占比（kernel 内计时器写全局，可选）。

## 停止条件（机器可判定）
`cargo test -p dinfer-kernels --features ref-tests --test mega_parity -- --ignored` exit 0 且打印 `speedup_B1 / B8 / B32`；**任一 B 档 ≥ 1.25× 进第二阶段**；三档都 < 1.10× 判失败，记原因收工。

## 迭代上限
7 个工作日 / 12 轮。每轮一行记录。

## 设计约束（不是调优数值，是结构）
- grid = 常驻 CTA（数量 = occupancy × SM，由 host 查）；任务 = (阶段 id, 分片 id)，静态编号；依赖 = 全局 `ready[stage]` 计数器，消费者 `ld.acquire.gpu` 自旋，生产者 `red.release.gpu` 加一。
- GEMV 阶段：权重按 N 切片给 CTA（每 CTA 若干 N 行，K 维整段），A 行（B×K bf16）从 L2 广播读；TMA（`cp.async.bulk`）把权重切片流进 smem 双缓冲；累加 f32，M=B 行。
- GDN 递归阶段：复用 `recurrent_decode_kernel` 的寄存器方案（每 warp BV/4 行状态），任务 = (seq, hv, V 块)。
- 阶段间激活走全局（一层 B=32 × 16480 × 2 B ≈ 1 MB，L2 常驻）。
- 所有分片大小 / CTA 数 / 双缓冲深度由 host（`ArchProfile`）给，kernel 内无常量。
- 先做**不含 attention 的 GDN 层**（48/64 层是它）；全注意力层和 lm_head 留第二阶段。

## 记录

| 轮 | 改动 | B=1 / B=8 / B=32（基线 → 实验，ms/层，倍率） | 结论 |
|---|---|---|---|
