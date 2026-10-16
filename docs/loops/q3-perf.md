# loop 规格：Q3 单卡性能（Qwen3-0.6B，4090，对照 vLLM v0.31.0）

- 状态：进行中（2026-10-16 开）
- 验证脚本：`scripts/verify/q3.sh`（待写；先用 `scripts/verify/bench_compare.sh` + kernel ref 测试）
- 迭代记录：`test/2026-10-16/q3-perf.md`

## 目标

在同卡、同模型、同配置下，dinfer 的 TTFT / TPOT / 吞吐**不低于** vLLM；先做单卡 mix 实例（无 PD）。基线见 `test/2026-10-16/bench/summary.md`：
c1 out 221 vs 447 tok/s、c32 2707 vs 8065、c8 p1024 56 vs 1916；TTFT 84 vs 18 ms（p171）、TPOT 3.9 vs 2.1 ms。

## 范围（按收益排序，一块一轮验收）

| 块 | 内容 | 现状 → 目标 |
|---|---|---|
| a prefill attention | 重写 `gqa_prefill`：grid (ceil(T/BQ), heads, reqs)，每 block 处理 BQ 个 query；K/V 按 tile 进 shared memory，online softmax 单遍；warp 内按 d 分工，fp32 累加，bf16 I/O。causal 只扫 ≤ 对角 tile。 | 1514 token 6.36 s → ≤ 50 ms（vLLM 0.04 s）；c8 p1024 组 TTFT 9.4 s → ≤ 150 ms |
| b decode attention | `gqa_paged`：每 (req, head) 一个 block 串行扫 key → split-K：grid (reqs, heads, splits)，每 split 扫一段 key 出 (m, l, acc)，第二个小 kernel 合并；batch 大时并行度 ×splits | c32 TPOT 10.0 → ≤ 5 ms |
| c CUDA graph | decode 步（全部请求单 token）整步捕获为一张图：H2D 输入先拷到固定地址缓冲，图内 forward + 采样；按 batch 大小分桶（1,2,4,8,16,32,64）各捕获一次 | c1 TPOT 3.9 → ≤ 2.5 ms |
| d 小开销 | 每 step 7 次 `Tensor::from_vec` H2D → 一块 pinned 固定缓冲一次拷；采样管线 kernel 融合（temperature+min_p+topk/topp 一个 kernel；logprobs 不再复制 raw）；`copy_cols` ×4 → GEMM 直接写分块或一个 split kernel | 每 step 省 ~0.5 ms |
| e 激活复用 | per-layer 的 qkv/q/k/v/attn/mlp 中间（28 层各一份 × max_tokens）→ 全模型共享一组 | 省 ~2.5 GiB 显存 → KV 更多 |

不在 Q3：PD 分离（Q4）、fp8、多卡、Triton。

## 环境

4090（`scripts/dev/remote4090.sh`），dinfer `--release` 常驻 :8012（tmux `dinfer`），vLLM Docker :8013。改 kernel 后：`push` → 远端 `cargo build --release -p dinfer --features qwen3` → 重启 tmux `dinfer` → `bench_compare.sh`。

## 观察方法

| 看什么 | 怎么看 |
|---|---|
| kernel 正确性 | `KERNEL_REFS_DIR=… cargo test -p dinfer-kernels --features ref-tests --test ref_tests gqa -- --ignored`（与 torch 对拍，atol 见测试） |
| 端到端正确性 | `real_parity`（greedy 32 步 = HF）、`sampling_parity`、`serve_e2e_qwen3`（q2.sh 全套） |
| 单 kernel 耗时 | `cargo test --release -p dinfer-kernels --test cold_shape -- --ignored --nocapture`（GEMM / attention 按形状） |
| 服务指标 | `bench_compare.sh` 四组 + `/metrics` 的 `engine_step_seconds` 直方图 |
| GPU 侧 | `nsys profile`（4090 有 nsys/ncu）看 kernel 占比 |

## 停止条件（机器判定）

1. 每块改完：kernel ref 测试全过（对拍 atol 不放宽）；`real_parity` 3/3、`sampling_parity` 5/5、`serve_e2e_qwen3` 6/6 仍过。
2. a：`cold_shape` 中 gqa_prefill T=1514 单 head-set ≤ 30 ms；`bench_compare` c8 p1024 TTFT p50 ≤ 150 ms。
3. b：c32 p128 TPOT p50 ≤ 5 ms。
4. c：c1 p128 TPOT p50 ≤ 2.5 ms。
5. Q3 总验收：四组 out tok/s 均 ≥ vLLM 的 1.0×（`scripts/verify/q3.sh` 读两边 JSON 比较，exit 0）。

## 上限

每块 6 轮；总 3 个工作日。达不到 1.0× 的组写清差在哪个 kernel，交给用户决定是否继续。

## 已知可能出错点

1. online softmax 的数值：fp32 累加 + bf16 输出，与 torch SDPA 的差异在 1e-2 内（现有 ref 测试口径）；causal 边界 tile 的 mask 要按绝对位置。
2. GQA：heads/kv_heads = 2，一个 kv head 服务两个 q head；tile K/V 到 shared 时可以让同一 block 处理同 kv head 的两个 q head 以省一半带宽（第二版再做）。
3. split-K decode 合并时 m/l 的数值稳定性（log-sum-exp 合并）。
4. CUDA graph 要求所有地址固定：`Tensor::from_vec` 每步新分配 → 必须先做 d。
5. 对拍中 bf16 near-tie 翻 token 是已知噪声（pitfalls），不能用"逐 token 相同"当 kernel 正确性标准，用 ref 测试的 atol。

## 记录

| 轮 | 块 | 现象 | 根因 | 处理 |
|---|---|---|---|---|
| 0 | 基线 | `bench/summary.md`；1514 token prefill 6.36 s（prefix 不命中）vs vLLM 0.04 s | `gqa_prefill_kernel` 16 个 block 串行遍历 T 个 query、每 query 三遍扫 key | 块 a 开工 |
