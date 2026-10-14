# loop: P4 MoE + 最小 runner

## 目标

1. MoE 参考路径与 HF `DeepseekV3MoE` 对拍：sigmoid 路由 + e_score_correction_bias + noaux_tc（n_group=1）+ top-k + renormalize + routed_scaling_factor；
   专家 MLP（silu）；shared expert；加权归约。tiny MoE GLM（第 0 层 dense、第 1 层 MoE 4 专家 top-2 + 1 shared）greedy 对拍。
2. `core::runner` 最小版：`SchedulerOutput` → flat batch → `Model::forward` → greedy → `ModelRunnerOutput`，与 `Scheduler` 串起来跑多请求。

## 环境

- 本机 3090。参考：`scripts/verify/gen_tiny_glm.py test/tiny_glm_moe --first-k-dense 1`（e_score_correction_bias 随机化，避免全 0 掩盖 bias 逻辑）。
- 对拍口径沿用 P3：全词表 logits max abs < 0.05；argmax 在 HF gap > 0.02 时一致；greedy 平局步可分叉。

## 观察方法

`scripts/verify/p4.sh` = p3 + `cargo test -p dinfer-glm53 --features ref-tests --test tiny_parity -- --ignored moe_` + runner 多请求测试。

## 停止条件

`scripts/verify/p4.sh` exit 0。

## 迭代上限

8 轮。

## MoE 参考实现方案（P4，不追求性能）

- `moe_route`：每 token 一个 block；fp32 logits = x·W_gate^T（cuBLASLt bf16→先转 fp32 GEMM？简化：gate 权重 bf16，输出 bf16 再转 fp32，误差可接受）；
  sigmoid → +bias → top-k（选择）；权重取 **原始 sigmoid 分数**（不含 bias）→ renorm → ×scale。输出 `topk_ids[T,K] i32`、`topk_w[T,K] f32`。
- 专家执行：CPU 侧不参与；GPU 上 `moe_count_sort`：按专家 bucket 排 (token, k) 对，产 `sorted_pairs`、`expert_offsets[E+1]`；
  对每个专家 e：gather 其 token 行 → gate_up GEMM → silu_and_mul → down GEMM → 乘 topk_w 加到 out[token]（`moe_scatter_add`）。
  专家数 ≤ 几百时 H2D 读一次 `expert_offsets`（一次同步）即可；真机再换 grouped GEMM。
- shared expert：普通 `Mlp`，结果加到 out。

## 记录

| 轮 | 现象 | 根因 | 处理 |
|---|---|---|---|
| 1 | MoE tiny 模型 prefill 对拍：logits max abs 0.0069 但 3 行 argmax 不一致（row 7/9/10） | 三行 HF 自身 top1-top2 gap = 0.0000 / 0.0020 / 0.0000，bf16 平局；与 P3 踩坑同类 | prefill 测试改为与 step 测试一致的"gap ≤ TIE_GAP 视为平局不计"；3 测试全过 |
| 1 | clippy `large_enum_variant`（`FeedForward::Moe` 1640 B vs Dense 520 B） | — | `Moe(Box<Moe>)`，Dense 不再包一层，enum 上 `#[allow]` |
| 2 | runner 对拍（dense）：第 3 步 req2 "sampled token must be argmax"，GPU 选 230、host 选 336，两者 logits 完全相等 | 测试里 host argmax 用 `max_by` → 平局取**大**下标；GPU kernel / torch 取**小**下标 | host argmax 改为首个最大值；该行其实是精确平局（fp32 相等） |
| 2 | runner 对拍：结束后 `runner.num_states() != 0` | 完成请求的 id 在**下一次** `schedule()` 的 `finished_req_ids` 里才传给 worker（vLLM 同） | 测试末尾多做一次 schedule+execute 再断言 |

## 结果（runner 部分，2026-10-14）

- `core/src/runner.rs`：`ReqState` / `BatchPlan` / `ModelExecutor` trait / `Runner`（update_states → build_plan → forward_sample → 回填）；2 个 mock 单测。
- `glm53/src/executor.rs`：`Glm53Executor`（H2D 全部输入 → `Glm53Model::forward` → `argmax_f32_n` → D2H）。
- `glm53/tests/runner_parity.rs`：Scheduler + Runner + Executor 跑 2 个并发请求（prompt 14 / 11，block 16 不对齐），1 步双 prefill + 31 步双 decode；dense 与 MoE tiny 模型各跑一遍，每步每请求 logits 对 HF 全序列前向：dense worst 0.0063、MoE worst 0.0071；greedy 均只在 HF 平局步分叉（dense req1 第 8 步 gap 0.0039、MoE req2 第 8 步 gap 0.0059，均 ≤ 2 ULP）。
- 限制（P4-b 放开）：一个 step 内不能混 prefill 与 decode；chunked prefill 带历史 KV 不支持（runner 直接报错）。

## 结果（MoE 部分，2026-10-14）

- 路由改成 fp32 gate kernel（`moe_gate_f32`）而非 bf16 GEMM 再转 fp32：HF 把 x 与 gate.weight 都 cast 到 fp32 再 linear，bf16 输出会在 top-k 边界翻转专家选择。
- 归约顺序与 HF 对齐：`out = bf16(bf16(Σ w·expert) + shared)`。
- 指标：prefill 14 行 logits max abs 0.0069；32 步 decode logits worst max abs 0.0078；greedy 仅在第 31 步分叉（HF gap 0.0078 = 2 ULP，平局）。日志 `test/2026-10-14/p4_moe_parity.log`。

