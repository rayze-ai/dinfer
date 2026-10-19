# Qwen35 里程碑 4：`models/qwen35` crate 到 tiny golden 对拍

> 用户 2026-10-19："赶紧把 Qwen35 完成，目录名字叫 Qwen35"。前置：GDN kernel（`qwen35-gdn.md`）与 attention 扩模板（`qwen35-attn.md`）均两卡对拍通过。
> 第一位里程碑 4 子代理 40 min 只读文档无产出，主会话 20:1x 接手搭骨架。

## 环境
本机 3090（编译 / 单测 / tiny golden GPU 对拍）、4090 GPU 7（sm89 对拍，`CUDA_VISIBLE_DEVICES=7 GPU_WAIT_S=5`）。
远端跑脚本要显式给 `RUSTUP_TOOLCHAIN=1.89.0-x86_64-unknown-linux-gnu`：4090 的 rustup 没有默认工具链，
且 2026-10-19 晚其 1.89.0 工具链只剩 58 MB（缺 rustc，`cargo test` 静默卡在 rustup 下载，表现为脚本超时而日志只有一行），
已从本机 rsync 完整工具链（`~/.rustup/toolchains/1.89.0-...`，633 MB）修好。B300 不跑（用户规则：等确认）。

## 观察方法
- `cargo test -p dinfer-qwen35`（CPU：config / weights 映射单测）。
- `scripts/verify/qwen35_golden.sh`：造 3 层 tiny golden（[linear, linear, full]）→ torch 参考 logits → Rust prefill logits / greedy 16 / 双请求混批对拍；exit 0 = 通过。
- `cargo build --workspace --features qwen3` 与 `--features qwen35` 都过；`ops_parity.sh` 不退化。

## 停止条件
`qwen35_golden.sh` exit 0 + 双 feature 构建过 + `chat_parity` 过。

**状态（2026-10-19 子代理收工）**：前两项已达成（`qwen35_golden.sh` exit 0 4/4；`--features qwen3` 与 `--features qwen35`
两个 workspace 构建都过；`ops_parity.sh` 65/65 不退化）。`chat_parity` 未做——本机没有 27B 的 tokenizer 之外的聊天 golden，
且需要真权重才能生成；真权重要 ≥ 60 GB 显存（B300，用户 2026-10-19 "B300 先不用"）。

## 迭代上限
10 轮。

## 记录

| 轮 | 改动 | 结果 |
|---|---|---|
| 1 | 骨架：Cargo.toml / lib.rs / config.rs（壳或裸 text_config 均可解析，`check()` 钉 head_dim 128|256、GROUP 1|2|4|6、rope_dim、GDN 整除、f32 状态、MTP ≤ 1；派生 gqa_group / rope_dim / gdn_group / eos_ids）+ workspace / bins 接 feature `qwen35` | config 单测 2/2 |
| 2 | weights.rs（前缀探测 VL/LLM/Plain、q/gate per-head 交织拆段、conv1d 形变、TP 分片、单位置映射单测） | `cargo test -p dinfer-qwen35` 5/5；commit d8e8523 |
| 3 | gdn.rs（conv→gate→recurrent→gated RMSNorm→out_proj、per-request `SlotTable`、slot 清零用常驻零张量 dtod） | 6/6 单测；commit 0836b06 |
| 4 | full_attn.rs + weights 拆双 GEMM（qkv `qmul=1` + 独立 gate）：`qkv_post_ext(rope_dim=64, norm_beta=1)`、GQA D=256 | 编译过；commit d0a65c5 |
| 5 | layer.rs + model.rs（异构层循环、独立 lm_head、gather_rmsnorm + logits） | 编译过；commit 35127d4 |
| 6 | executor.rs（eager 单流、plan 上传、GDN slot 分配 + "位置0 清零"、采样流水） | 编译过；commit 869b1c5 |
| 7 | service.rs + parser.rs + bins/dinfer `--features qwen35` | `cargo build -p dinfer --features qwen35` 过；commit 5766cdc / 8c3e01a |
| 8 | tiny golden（`scripts/dev/make_qwen35_tiny.py`）+ `golden_parity.rs` + `qwen35_golden.sh`。**踩坑**：① 参考侧 `rmsnorm` 写成"f32 归一化 × f32 权重再落 bf16"，与 kernel 的"乘权重**后**落 bf16"差一个舍入点 → prefill 差 0.2 且 greedy 第 1 步偏；改成 `qb(norm*w)` 后 prefill max abs 0.20 → 一致。② 参考 `step_logits[i]` 最初是"喂了第 i 个 token 之后"的 logits，对拍口径要"预测第 i 个 token"的那一行 → 索引整体错一位（表现为教师强制步 0 差 4.0）。③ tiny 随机权重下 greedy 有大量平局（ref top-2 gap 最低 0.03），我们的 logits 与参考差 ~0.2 → 平局步 argmax 断言无意义，改成"只有 ref gap > 0.15 的行断言 argmax"，并把 chunked 用例改成教师强制（喂 golden token）。 | **4/4 通过**（prefill 28 行 max abs 0.2031 / argmax 全对；greedy 16 全对；chunked 4/4/1 + 15 步教师强制；双请求混批各 16 全对）；commit b8162b1 |
| 9 | 收口：KV 块数改为按剩余显存自动估算（去掉硬编码 54 GiB）、TP（world > 1）明确报错而不是静默少算、GDN slot 复用给带历史请求时清零并告警、GDN 状态显存进启动日志 | golden 仍 4/4；commit 29276b1 / 74e81d3 / 58dccd5 / 3083226 |

## 里程碑 5：B300-50 真权重 e2e + vLLM 对比（2026-10-19 晚，用户："qwen35 在 B300-50 上跑测试包括 vllm 基准测试，跑完再跑 k3"）

环境：B300-50 GPU 0 dinfer `:8012`（`/data/models/Qwen/Qwen3.8-27B` bf16，tmux `q35`，日志 `$R/bootstrap/q35_serve.log`）、
GPU 1 vLLM nightly 容器 `vllm-qwen35-b300 :8013`（`VLLM_MODEL=... VLLM_NAME=... scripts/dev/vllm_nerdctl.sh run 1 8013 1`）。
停止条件：3 条 greedy prompt 两边前缀一致 + `bench_compare.sh test/qwen35/2026-10-19/bench_b300_1` 四组跑通。

| 轮 | 改动 / 观察 | 结果 |
|---|---|---|
| 10 | push + `cargo build --release --features qwen35 -p dinfer -p dinfer-bench`（sm_103，38 s）；`qwen35_golden.sh` 在 sm_103 | golden 4/4 + 单测 13/13 过 |
| 11 | 真权重 serve：加载 8.2 s（64 层 6.1 s），GDN 状态 9.2 GiB / 64 slot，KV 自动预算 161245 块 157 GiB。首个请求 → `engine step failed: qwen35 enqueue: DriverError(CUDA_ERROR_INVALID_CONTEXT)`，之后所有请求 0 token `finish_reason=stop`（引擎已死但 HTTP 仍应答）。tiny golden 单线程不触发 → 引擎线程上某条 CUDA 路径没绑上下文 | 排查中 |
| 12 | 根因：引擎线程 ≠ 构建线程，cudarc 不自动把 ctx 绑到新线程；qwen3 的 `mark_gpu_start → record_timing` 顺带 `bind_ctx`，qwen35 没有。`forward_sample` 开头加 `self.stream.bind_ctx()` | INVALID_CONTEXT 消失，请求返回 48 token |
| 13 | 真权重输出乱码：`/v1/completions` "The capital of France is" → `'�� +:+'`，top-3 logprob 全 −9.1~−9.3（近均匀）；vLLM 同 prompt `' Paris'` −0.45。tiny golden 4/4 过 → 真权重特有：大形状（hidden 5120 / vocab 248320 / 64 层）、权重映射（layout=Vl tensors=851）、或 layer_types 表 | 排查中 |
| 14 | HF 参考（vLLM nightly 镜像内 transformers 5.15.1，GPU 2 一次性容器 `hfref`，`$R/bootstrap/hf_ref_q35.py` → `hf_ref_q35.pt`）："The capital of France is" ids `[760,6511,314,9338,369]`（与 vLLM /tokenize 一致）top1 ' Paris' 17.5。逐层末 token 残差范数 HF：embed 1.01 → L0 11.93 → L1 14.53 → L3 21.76 … L63 380 → final norm 141.9。dinfer（`QWEN35_TRACE=1` 新加逐层打点 `trace_norm`）：embed 1.008（一致）→ **L0 24.49**（attn_out 9.19）→ L1 29.45 → L3 60.7 → L6 119.8 → L15 219 → 之后 ~230 平台。**第 0 层（GDN）起即偏**，全注意力层不是首因 | GDN 真权重路径有误；tiny golden 盲区 |
| 15 | **根因**：HF `Qwen3_5RMSNorm` 是 zero-centered（`y = norm(x)·(1+w)`，`weight` 初值 0），用于 `input_layernorm / post_attention_layernorm / norm`；真权重 L0 input_ln 均值 −0.033、post_ln 均值 −0.217、final norm 均值 0.944。dinfer 与 tiny 参考都按普通 `norm(x)·w` 算；tiny 把这三类权重设成全 1 → `1·x ≡ (1+0)·x`，golden 分不出来（**golden 盲区**）。q/k_norm（`norm_beta=1`）与 `linear_attn.norm`（gated，初值 1，非 zero-centered）之前已对。L0 HF 内部：ln.out 75.5、qkv 397.8、z 226、a 25.2、b 16.8、core→norm.out 0.025、out_proj 11.24、post_ln 6.11、mlp 0.958。修：`weights.rs::rmsnorm` 加载时 `1+w`（kernel 不动）；tiny 参考改 `1+w` 且三类权重改随机（σ 0.2）后重生成 golden | 待验证 |
| 16 | 修后重启：3 条 greedy prompt（中文解释 / Python fib / 欧洲首都）与 vLLM **逐字一致**（傅里叶那条 36 vs 34 token，只是末尾多 2 个 token，前 34 一致）。**真权重 e2e 通过**。tiny golden 重生成后 prefill / chunked 2/2 过，greedy a/b 在第 2/0 步撞近平局（gap 0.0078 / 0.094）→ 可比步数不足判 FAIL（随机权重性质，需换 seed，待办） | e2e ✅；golden 待换 seed |
| 17 | **bench_b300_1**（`bench_compare.sh`，dinfer GPU0 :8012 vs vLLM GPU1 :8013，同 seed）：c1 TPOT **24.1 vs 10.3** ms（0.40×）、c8 28.4 vs 11.2（0.36×，TTFT p99 1250）、c32 54.8 vs 12.9（0.21×，TTFT p50 2200）、c8-long 119 vs 11.2（**0.09×**，TTFT 7147）。eager 单流、无 CUDA graph、GDN 递归 kernel 顺序扫描、chunk prefill 未接（prefill 走逐 token 递归）。**c8-long 期间 87 条 `GDN 状态 slot 被复用给带历史的请求` 警告**（req 85–89 pos 160/176 slot 20–26）：max_num_seqs 64 = slot 数 64，但在飞 ≤ 8 → 不该满；疑 SlotTable 没在请求完成时释放，靠 LRU 淘汰"不在本步"的请求 → 被抢占 / 分块 prefill 中途不在本步的活请求被当受害者，**结果会错** | 性能基线落地；slot 复用是正确性 bug，下一轮修 |
| 18 | ① slot 复用根因：`SlotTable` 从不在请求结束时归还 slot，只靠 LRU 淘汰"本步不在场"的请求；c8-long 累计 > 64 个请求后，分块 prefill 中途不在场的活请求被当受害者。修：`BatchPlan.finished_req_ids`（runner `build_plan` 从 `SchedulerOutput.finished_req_ids` 填）+ `SlotTable::release` + executor 每步先 release。② tiny golden：`make_qwen35_tiny.py --seed`，扫 12 个 seed 选 **seed 8**（a/b/c 决定性步 11/11/11）；`compare_greedy` 改为"平局步跳过、决定性步必须一致"，不再因首步平局判 FAIL | 本机 golden + B300 复测 |
