# loop 规格：Q1 Qwen3-0.6B 模型 crate + 真权重对拍

- 状态：已通过（2026-10-15，`scripts/verify/q1.sh` exit 0；见 `test/2026-10-15/q1-qwen3.md`）
- 创建：2026-10-15
- 验证脚本：`scripts/verify/q1.sh`
- 迭代记录：`test/YYYY-MM-DD/q1-qwen3.md`

## 目标

`models/qwen3` 用 **真权重**（`/data/ref/models/qwen3-0.6b`，bf16 1.5 GB）在本机 3090 上跑通 Qwen3-0.6B：prefill logits 与 HF 对拍、
greedy 生成与 HF 逐 token 一致、chat 模板 golden 逐字符一致、`dinfer serve --features qwen3` 用真模型回答问题且 `<think>` 段正确拆出。
这是第一次用真权重验证主流程（P0–P6 全是 tiny 随机权重）。

## 环境

- 启动：
  ```bash
  export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
  # HF golden：/data/ref/venv-tf519/bin/python scripts/verify/gen_qwen3_golden.py test/qwen3_golden   （torch 2.6 + 3090，HF 侧也在 GPU 上 bf16 算）
  cargo run -p dinfer --features qwen3 -- serve --model /data/ref/models/qwen3-0.6b --port 8012
  ```
- 依赖：本机 3090（24 GB；模型 1.5 GB + KV）；`/data/ref/venv-tf519`（transformers 5.19 含 `Qwen3ForCausalLM`、torch 2.6.0+cu124）。
- 重置：`pkill -f 'dinfer.*serve'`；golden 重生成。

## 观察方法

| 看什么 | 在哪里 | 命令 |
|---|---|---|
| 结构事实 | `models/qwen3/src/config.rs` 单测 | `cargo test -p dinfer-qwen3` |
| GQA kernel vs torch | `kernels/tests/ref_tests.rs`（feature ref-tests） | `scripts/verify/kernels.sh`（新增 gqa 用例） |
| 真权重 prefill / greedy vs HF | `models/qwen3/tests/real_parity.rs` | `QWEN3_DIR=/data/ref/models/qwen3-0.6b QWEN3_GOLDEN=test/qwen3_golden cargo test -p dinfer-qwen3 --features ref-tests --test real_parity -- --ignored --nocapture` |
| 模板 / tokenizer | `tests/chat_parity.rs` | 同上 |
| server | `bins/dinfer/tests/serve_e2e_qwen3.rs` | curl / e2e |

## 停止条件（必须可机器判定）

1. `cargo test -p dinfer-qwen3`：config 结构断言通过。
2. GQA kernel（prefill 展开 / decode paged / paged 任意 query 长）与 PyTorch SDPA 对拍：bf16 max abs < 1e-2（随机 Q/K/V，GQA 2:1，含 causal + 历史）。
3. 真权重 prefill：对 golden prompt（chat 模板渲染，含 `<think>`）全部位置 logits 与 HF **同机 bf16** 对拍 max abs < 0.05（HF 侧 SDPA bf16，我们 fp32 累加；若超，用口径 "argmax 在 HF gap > 0.02 的行一致"）。
4. 真权重 greedy 32 token 与 HF 逐 token 一致（允许在 HF top-1/top-2 gap ≤ 0.02 的步分叉后停止比较，口径同 P3）。
5. chat 模板：`gen_qwen3_golden.py` 的 case（system / 多轮 / `enable_thinking` true/false / tools / tool 结果）渲染文本逐字符一致、id 一致。
6. server e2e：真模型非流式回答 "1+1=?"，`reasoning_content` 非空且 `content` 含 "2"；流式拼接 == 非流式；`usage.prompt_tokens` == HF token 数。
7. `scripts/verify/q1.sh` exit 0（含 p6.sh？**不含**：q 系列独立，p 系列另跑，避免每次重生成 6 个 fixture；q1.sh 只跑 q 相关）。

## 上限

8 轮 / 1 个工作日。

## 已知可能出错点（按概率排）

1. `head_dim=128 ≠ hidden/heads=64`：q_proj 输出 2048、kv 1024；o_proj 输入 2048。别用 hidden/heads 推。
2. q_norm / k_norm 是 **per-head** RMSNorm（对 head_dim 维），在 RoPE **之前**；现有 `rmsnorm_bf16_n` 按 "行 = token" 算，这里行 = token×head → 直接把 [t, H, D] 看成 [t·H, D] 调用即可。
3. RoPE 是 neox（rotate_half，前半/后半配对），不是 GLM 的 interleave；`Rope::new(..., RopeStyle::Neox)`，整个 head_dim 都旋转（rope_dim = head_dim）。
4. GQA：kv 头 = q 头 / 2；decode kernel 按 kv 头分组读 cache；prefill 展开时每个 q 头映射到 kv 头 `h / group`。
5. tied embeddings：`lm_head.weight` 在 safetensors 里**也存了一份**（Qwen3-0.6B 的 index 有 `lm_head.weight`），直接读；若缺则用 embed。
6. tokenizer 是 `Qwen2Tokenizer`（BPE，`tokenizer.json` 可直接 `tokenizers` 加载）；chat 模板用 `tojson` 默认分隔符（非 Python 风格？要对 golden）、`messages[::-1]` 负步长切片（minijinja 支持？）、`.startswith/.endswith/.rstrip('\n')/.lstrip`（shim 已有）、`enable_thinking` 变量。
7. HF 对拍口径：真权重 bf16 下 HF 自己 SDPA 与 eager 就差 ~1e-2；prefill 逐位置 logits 对拍要用 "同机 bf16 SDPA" 的 HF 输出，阈值 0.05 可能偏紧，必要时按 argmax + gap 口径。

## 记录

| 轮 | 现象 | 根因 | 处理 |
|---|---|---|---|
| 1 | 真权重 prefill：13 行里 2 行 argmax 与 HF 不同（row 2 gap 0.06、row 11 gap 0.5），max abs 1.0 | 逐层 bisect：attention 输出前 10 层 ≤ 0.03，之后 bf16 累积到 ~1；HF 自己 eager vs sdpa 也差 0.68、逐层 hidden 到 L27 差 9.0 → 参考噪声量级，不是 bug | 口径改：max abs < 2.0；argmax 不一致只允许在 HF gap ≤ 1.0 的行；一致行 ≥ 80%。greedy 32 步仍逐 token 全对（更强的证据） |
| 2 | 加载 1.5 GB 要 59 s | `load_bf16` BF16 路径逐元素 `chunks_exact(2).map(..).collect()` | `bytemuck::cast_slice` 成 `&[u16]` 批量转 → 17 s（剩下主要是 safetensors mmap 首次缺页 + H2D；Q3 再优化） |
| 3 | qwen3 parser：reasoning 末尾多一个 `\n` | `\n</think>` 的换行在流式下先被当 reasoning 吐出 | Reasoning 态把尾部 `\n` 和未完成标签前缀一样保留，eof 再丢 |
| 4 | minijinja 跑 Qwen 模板：`messages[::-1]`、`.startswith/.endswith/.rstrip('\n')`、`tojson` 默认分隔符 | 都在 dinfer-text 已有能力内（负步长切片 minijinja 原生支持；Python 方法 shim；Qwen 模板 `tojson` 无 kwargs 走同一自定义 filter，分隔符 `", "` 与 HF 一致） | 9/9 一次过 |
