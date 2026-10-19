# 踩坑记录

> 一条一个坑，标题写**现象**（搜索时按现象搜），正文讲根因与解法。新条目加在最上面。
>
> 条目格式（标题用 `## 2026-10-18 同一张卡起第二个 dinfer 实例：`not enough GPU memory for KV cache: free 5.5 GiB, cap 30%`

- 现象：PD 验收在 GPU 3 上先起 D（默认 utilization 0.9）再起 P（0.3），P 报 KV 预算不足；free 明明还有 5.5 GiB。
- 根因：自动预算 = `min(free, total × u − used) − reserve`。`used` 是**整卡已用**（含别的进程），`u` 是**整卡占比上限**，不是本进程配额。第二个进程看到 `used ≈ 17 GiB > cap 7 GiB`，预算 0。这和 vLLM 的 `gpu_memory_utilization` 语义一致，不是 bug，但同卡多实例就不能靠它分配。
- 解法：同卡多实例一律 `--num-blocks N` 固定块数（`kv_cache: fixed` 路径不看 utilization）。q4.sh：P 512 / D 2048。
- 如何避免：脚本里起第二个进程前先想 "预算公式里 used 包含谁"；`wait_up` 失败要打子进程日志尾，否则只看到 "port did not come up"。

## 2026-10-18 4090 GPU 2 起 D 报 "layer 21 attn" OOM：GPU 2 被别人的 vLLM 占着

- 现象：`Q4_GPU_D=2` 起 D，`port 8102 did not come up`，d.log 末尾 `Error: layer 21 attn`（权重加载到第 21 层显存不够）。
- 根因：GPU 2 上有 root 2026-10-16 起的 `vllm serve Qwen3-0.6B --port 8000`（非 docker，`--gpu-memory-utilization 0.9`，占 21 GB），不是我们的 `vllm-qwen3` 容器（那个在 GPU 1 :8013）。
- 解法：不动别人的进程；跨卡测试改用 GPU 3（P）+ GPU 4（D）。
- 如何避免：起任何测试进程前先 `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader`，空卡（4 MiB）才用；共享机 GPU 0/1/2 都有常驻。
- 相关：loop q4-pd 第 16 轮。

## 2026-10-18 4090 跑 `Q4_REMOTE=1 q4.sh` 两次空跑：bench 包名、模型路径都是本机口径

- 现象：第一次 `cargo build -p bench` 报 "package ID specification `bench` did not match"；第二次 e2e 0.16 s 失败，`[serve:err] Error: No such file or directory`。
- 根因：bench crate 的 package 名是 `dinfer-bench`（bin 名才是 `bench`）；`q4.sh` 的 `QWEN3_DIR` 默认 `/data/ref/models/qwen3-0.6b` 是本机路径，4090 上权重在 `/data/yaochen/infer/models/qwen3-0.6b`。
- 解法：`cargo build --release -p dinfer -p dinfer-bench --features qwen3`；`QWEN3_DIR=/data/yaochen/infer/models/qwen3-0.6b Q4_REMOTE=1 bash scripts/verify/q4.sh test/qwen3/<日期>/<名>`。远端非交互 ssh 还要先 `source /data/yaochen/infer/env.sh`（cargo 不在 PATH，BUILD_RC=127）。
- 如何避免：远端命令统一从 `scripts/dev/README-remote4090.md` 的模板抄；q4.sh 若 `$MODEL` 不存在应直接报错退出而不是让 serve 去撞。
- 相关：loop q4-pd 第 14–15 轮。

## 2026-10-18 PD 分离 e2e：D 等 KV 超时回退 + 首 token 重复（"redred"）

- 现象：`e2e_qwen3_pd` 经 proxy 的输出比单机多一个重复的首 token；D 日志 `pd: KV wait timeout, falling back to local prefill`。
- 根因 1：P 侧 "finish 时记块、下一次 schedule 导出" 的设计依赖 engine 再跑一步；P 空闲（只有这一个请求）时 engine 进 `rx.recv()` 死等，导出永远不发生。core 单测里紧跟着提交了别的请求所以没暴露。
- 根因 2：D 回退本地重算后自己出了首 token，proxy 不知情仍拼上 P 的首 token。
- 解法：`Scheduler::has_pending_export()` 进 engine 空闲判定；D 侧 chat.rs 先在 HTTP 线程等迁移到达（`ChatBackend::pd_wait_for`），回退时响应头 `x-pd-imported: 0`，proxy 据此决定拼不拼。
- 如何避免："事件发生在下一步" 的设计要同时问 "谁保证有下一步"；跨进程协议里任何一侧的回退都要在响应里显式告知对端，不能靠默认假设。

## 2026-10-18 第 5 次 `400 Input is too long`：一条 Bash 四段拼接，每段都"截断"了，合计 11 KB

- 现象：会话开场 `git status --short && git diff --stat && git diff X | head -c 6000 && tail -n 30 loop.md | cut -c1-300` 一条命令，context_guard 放行（有截断管道），输出 11.3 KB 被 harness 存盘只给 2 KB 预览，再加后续几条 7–8 KB 的 grep 输出，上下文很快打满。
- 根因：钩子只判"有没有截断管道"，不判截断量；`head -c 6000` + `tail -n 30 | cut -c1-300`（最多 9 KB）本身就超过 `BASH_MAX_OUTPUT_LENGTH=8000`。多段 `&&` 拼接时每段合规、合计超限。
- 解法：`context_guard.sh` 按 `;`/`&&`/`||` 拆段逐段检查；`head -c` 上限 3000、`head/tail -n` 上限 60 行（超过须配 `cut -c`）；每段 cat/diff 都要各自带截断。
- 如何避免：一条 Bash 只看一件事；想看 diff 先 `--stat`，再按文件 `git diff -- path | head -c 3000`；长日志只 `grep -E` 关键行。开场复述现状只需要 `git status --short` + loop 表最后 2 行。
- 影响：本次丢了半小时上下文，重新 /clear 恢复。

## YYYY-MM-DD 现象一句话`，正文五项：现象 / 根因 / 解法 / 如何避免 / 相关）：
>
>     ## 2026-01-01 启动时报 8080 端口被占用
>     - 现象：
>     - 根因：
>     - 解法：
>     - 如何避免：
>     - 相关：`文件/命令/链接`

## 2026-10-15 HF forward hook 里直接存张量，多次前向后拿到的全是最后一次的值

- 现象：`gen_tiny_glm.py` 对同一模型连做多次 `model(seq)`（每个 decode 步一次），hook 存的 `attn_L1` 等在后面分析时与当次前向对不上。
- 根因：hook 保存的是 `o[0][0].float().numpy()` 的视图 / 共享存储，下一次前向复用缓冲后内容被覆盖。
- 解法：hook 里一律 `.copy()`（numpy）或 `.detach().clone()`（torch）。
- 如何避免：写对拍 dump 时，hook 保存值后立刻 `np.save`，或用 `.copy()`；对比前先校验 "dump 的 logits 与 fixture 的 expect 逐位相等"（本次就是靠这一步发现的）。
- 相关：`scripts/verify/gen_tiny_glm.py::dump_hidden/dump_dsa`

## 2026-10-15 minijinja 跑 HF chat_template.jinja：5 处方言差异导致解析失败或输出不逐字符一致

- 现象：① `syntax error: unexpected float`（`m.content.0.type`）；② `string has no method named strip/split`、`map has no method named items`；③ tools 段多出一个 `\n`；④ JSON 里 key 顺序被排序、分隔符是 `,`/`:`（Python 是 `", "`/`": "`）。
- 根因：minijinja 把 `.0.type` 词法成 float；没有 Python 字符串 / dict 方法；HF 的 jinja 环境是 `trim_blocks=True, lstrip_blocks=True`（`chat_template_utils._compile_jinja_template`）而 minijinja 默认关；minijinja/serde_json 默认 BTreeMap 排序 key，`tojson` 用紧凑分隔符且不认 `ensure_ascii` kwarg。
- 解法：加载前 `x.N.attr` → `x[N].attr`；`set_unknown_method_callback` 补 strip/split/items 等；`set_trim_blocks/lstrip_blocks(true)`；minijinja 与 serde_json 都开 `preserve_order`；自写 `tojson` filter 按 Python `json.dumps(ensure_ascii=False)` 输出。验收口径必须是 "渲染文本与 HF `apply_chat_template(tokenize=False)` 逐字符一致"，再 encode 比 id —— 只比 id 会漏掉空白差异。
- 如何避免：新模板先用 `gen_chat_golden.py` 产 golden（含 tools / 多轮 / tool 结果乱序 / `clear_thinking` / 非法 `reasoning_effort`），跑 `chat_parity` 看 `first diff at char N` 定位。
- 相关：`models/glm53/src/chat.rs`、`scripts/verify/gen_chat_golden.py`、`models/glm53/tests/chat_parity.rs`

## 2026-10-15 server 请求 500 `indexer: block_table covers 16 < max_seq 1024`：block_table 宽度不是 cache 容量

- 现象：单测 / 对拍里 block_table 是 `[1, NUM_BLOCKS]`（全 cache），一切正常；接上 runner 后第一个请求就 500。
- 根因：runner 的 `BatchPlan.max_blocks` 是本 batch **最长请求的块数**（vLLM 同），不是 cache 容量；索引器层与 `ops::dsa_index_score_f32` 各有一条 `max_blocks*block_size >= max_seq` 的 ensure，把 "score 行步长 = cache 容量" 和 "block_table 宽度" 混为一谈。kernel 实际只读 `bt[j / bs]`，j < vis ≤ seq_len，宽度覆盖每个请求自己的 seq_len 即可。
- 解法：删掉两处 ensure；注释写清两个量的语义。
- 如何避免：新 kernel 的 host 侧 ensure 只检查 "会越界读写" 的条件，不要把测试 harness 的巧合（全表）写成约束；接 runner 的路径（`serve_e2e`）要尽早跑。
- 相关：`crates/layers/src/indexer.rs`、`crates/kernels/src/ops.rs::dsa_index_score_f32`、`crates/core/src/runner.rs::build_plan`

## 2026-10-15 HF 参考在 bf16 下不是"前缀稳定"的：同一 prompt 加两个 token 重算，前缀行的 top-k / 路由会翻转

- 现象：注入 HF top-k 后 prefill 全对（attention ≤ 0.0005），但 decode 第 2 步起 L1（shared 层）attention 误差 0.02（L0/L2 仅 0.0005），logits 0.06；且与步数无关、不累积（step 1 好，step 2 坏，step 21 坏）。逐层 bisect 到 cache：我们 L1 cache 第 142 行与 HF 同一行差 1.87，但 prefill 刚结束时该行是对的。
- 根因：HF 侧参考是对 `prompt + gen[:i]` 整条序列重新前向。bf16 下 SDPA/MoE 的 batch 内部归约顺序随序列长变化，L0 索引器对 12 个 **前缀行** 的 top-k 在 256 与 258 token 两次前向里不同（其中 7 行是前 8 行的 "可见 ≤ k" 退化，无影响；5 行 142/163/172/233/246 是真翻转），这 5 行的 attention 输出变了 0.02–0.03 → 它们在 HF 里的后续层 KV 与我们 prefill 时写进 cache 的不同。任何 decode 步只要 top-k 选中这 5 行之一，L1/L2 attention 就与 HF 不可比。不是 bug：我们的增量 cache 才是 "正确" 的语义，HF 的全序列重算在 bf16 下自己都不自洽。
- 解法：`gen_tiny_glm.py` 每个 decode 步额外导出 `hidden_L<i>_long_prefix_drift_steps.npy`（[gen, T]：每前缀行相对 prefill-only 前向的 hidden max abs 变化）；`dsa_parity` 把变化 > 0.02 的行并入 "漂移行" 集合，注入 top-k 命中漂移行的 decode 步不断言。可判定步 18/32，其余步只上报。
- 如何避免：参考实现用全序列重算做 decode 对拍时，先验证它自己是前缀稳定的（对比 prefill-only 与更长序列的前缀 hidden）；不稳定就导出逐步漂移并过滤，不要调阈值硬凑。对拍脚本的 `dump_dsa` 之类 hook 一律用 `.copy()`，否则多次前向后拿到的是最后一次的张量。
- 相关：`scripts/verify/gen_tiny_glm.py::dump_hidden`、`models/glm53/tests/dsa_parity.rs::drift_rows`、`models/glm53/tests/dsa_diag.rs`（逐层 / 逐 head / cache 行 bisect 工具，保留）

## 2026-10-15 Claude Code 会话报 `API Error: 400 Input is too long`，整段上下文丢失，只能 /clear

- 现象：长会话（多轮 bisect、反复贴测试日志）突然所有请求报 400 Input is too long，无法继续也无法 compact，只能 `/clear` 重来。本项目已发生两次（10-14 P4、10-15 P5），每次丢掉未写进文件的中间结论。
- 根因：上下文被工具输出撑满——整段 `cat` 测试日志 / `cargo test --nocapture` 全量输出 / 大 diff 直接进对话；自动压缩来不及或压缩后仍超限。
- 解法：发生后按 CLAUDE.md 开场流程从 `docs/progress.md` 交接区恢复；本次恢复后先做 wrap 把状态落盘。
- 如何避免：① 所有 Bash 输出先 `| grep -E ... | tail -n` 截断，日志 `tee` 到 `/tmp` 后只看摘要；② 每完成一轮 loop 迭代就把结论写进 `docs/loops/<feature>.md` 记录表和 `test/YYYY-MM-DD/`，不要攒到最后；③ 长 bisect 前先 wrap 一次。
- 相关：`docs/progress.md` 交接区"会话注意"

## 2026-10-15 DSA 索引器 top-k 与 HF 不一致，但分数公式逐位正确：bf16 下硬判决本质不可复现

- 现象：tiny DSA 模型长 prompt 256 行里 21 行的索引器 top-k 集合与 HF 不同；两边 index_scores max abs diff 0.024。注入 HF 的 top-k 后仍有 4 行 logits 差 0.2+。
- 根因：① 把 HF 自己的 q/k 喂进我们的分数公式，与 HF 分数逐位相等 → 公式无误；差异全来自 q/k 各自 bf16 舍入后的 fp32 点积（~1e-2），而 HF 自身 k-th 与 (k+1)-th 分差中位数只有 0.006 → top-k 是对噪声级分差做硬判决，两套实现不可能逐位一致。② 剩下 4 行是 MoE 路由第 2/3 名专家分差 < 0.01 时选了不同专家（P4 已见过的 bf16 平局，tiny 随机权重下约 2%）。逐层 bisect 显示三层 attention 输出 max abs ≤ 0.0007。
- 解法：对拍分三道：top-k 集合差异只允许落在 "HF k-th 分数 ± 本行两边分数实测误差" 内；注入 HF top-k 验证稀疏 attention（排除索引器）；全程自算只在 "索引器 k-th 分差 > 实测误差 且 路由分差 > 0.01" 的行/步上断言。参考侧额外导出 `indexer_score_*`、`router_margin_*`。
- 如何避免：任何 top-k / argmax / 路由类硬判决都不要用 "集合完全一致" 做验收；先从参考实现导出判决依据的分数，用分差判定该行是否可判定。试过把 RoPE cos/sin 表改成 bf16 对齐 HF（无改善，21 → 24 行），不要再试。
- 相关：`models/glm53/tests/dsa_parity.rs`、`scripts/verify/gen_tiny_glm.py::dump_dsa/dump_router_margin`、`docs/loops/p5-dsa-indexer.md`

## 2026-10-15 transformers ≥ 5 的 MoE 专家权重是融合 3D 张量，与 GLM-5.3 safetensors 逐专家布局不同

- 现象：用 transformers 5.19 `GlmMoeDsaForCausalLM.state_dict()` 存的 tiny 模型，Rust 加载报 `missing tensor model.layers.1.mlp.experts.0.gate_proj.weight`。
- 根因：5.x 的 `GlmMoeDsaExperts` 用 `gate_up_proj [E, 2I, H]` / `down_proj [E, H, I]` 两个参数存全部专家；真 GLM-5.3 safetensors（以及 4.x DeepseekV3）是 `experts.{e}.{gate,up,down}_proj.weight`。
- 解法：`gen_tiny_glm.py --dsa` 存盘前拆回逐专家：`gate_up[e].chunk(2, dim=0)` → gate、up。
- 如何避免：以真模型 safetensors index 的张量名为唯一真相，参考实现的 state_dict 只是导出格式；加载器不要迁就某个 transformers 版本。
- 相关：`scripts/verify/gen_tiny_glm.py`、`/data/ref/models/glm53/model.safetensors.index.json`

## 2026-10-15 所有 cargo 命令假失败："failed to run rustc to learn about target-specific information"

- 现象：`cargo build/test/clippy` 全部报该错，stderr 是 rustc 把一段 bash trace（`++ awk '-F: ' '{if($1~"Serial Number")print$2}'`）当 Rust 源码编译；直接跑同一条 `rustc -` 命令却正常；在 `/tmp` 新建 crate 也正常。
- 根因：cargo 用 `rustc - --print=...` 探测目标信息（stdin 读源码），结果缓存在 `target/.rustc_info.json`。本机 `/etc/cron.d/lldpd_config_update` 每分钟跑一个 `set -x` 的脚本，某次它的 trace 文本进了那次探测的 stdin，失败结果被缓存，之后每次 cargo 直接读缓存里的失败。
- 解法：`rm target/.rustc_info.json`。`scripts/verify/p4.sh` 开头已加这一行。
- 如何避免：验证脚本里凡 cargo 都 `< /dev/null`；看到 `<anon>` 里出现 shell 文本就是这个坑，别去查代码。
- 相关：`target/.rustc_info.json`、`/etc/cron.d/lldpd_config_update`

## 2026-10-14 MoE 路由用 bf16 GEMM 算 logits 会翻转 top-k 选择

- 现象：按"gate 权重 bf16 → cuBLASLt bf16 GEMM → 转 fp32 → sigmoid/top-k"实现，专家选择与 HF 偶发不一致（分数接近的两个专家互换）。
- 根因：HF `DeepseekV3TopkRouter.forward` 把 x 与 gate.weight 都 cast 到 **fp32** 再 linear；bf16 GEMM 的输出先被舍到 bf16（3 位有效数字），sigmoid 后分差 < 1 ULP 的专家对会乱序，而 top-k 是硬判决，之后误差不可恢复。
- 解法：`moe_gate_f32` 参考 kernel：bf16 输入升 fp32 做点积，输出 fp32 logits；真机版也必须保持 fp32 路由（vLLM `moe_router_dtype: float32` 同义）。
- 如何避免：任何"硬判决"（top-k、argmax、路由）前的运算精度要与参考实现逐算子对齐，不能只看最终 logits 误差。
- 相关：`crates/kernels/cuda/moe.cu::gate_kernel`、`crates/layers/src/moe.rs`

## 2026-10-14 host 侧 argmax 平局规则与 torch / GPU kernel 相反

- 现象：runner 对拍报 "sampled token must be argmax"，GPU 选 230、host 选 336，但两者的 logits **完全相等**。
- 根因：测试里 `(0..v).max_by(|a,b| row[a].partial_cmp(&row[b]))` 在相等时返回**最后一个**最大值；torch.argmax 与我们的 argmax kernel 都取**第一个**（小下标）。
- 解法：host 参考 argmax 写成显式循环 `if row[i] > row[best]`（严格大于才更新）。
- 如何避免：所有 host 侧 "参考实现" 的平局规则显式写出来，并与 kernel 注释对齐；不要依赖标准库迭代器的平局行为。
- 相关：`models/glm53/tests/{tiny_parity,runner_parity}.rs::argmax`、`crates/kernels/cuda/sampling.cu`

## 2026-10-14 请求完成后 runner 状态没被清理：`finished_req_ids` 晚一步到达

- 现象：所有请求 `update_from_output` 都报 Finish 后，`runner.num_states()` 仍为 2。
- 根因：与 vLLM 一致，scheduler 在 `update_from_output` 里只把完成请求记进内部列表，**下一次** `schedule()` 才通过 `SchedulerOutput.finished_req_ids` 告知 worker 释放状态。
- 解法：引擎主循环结束条件要再多跑一次 `schedule()` + `runner.execute()`（空 batch）；测试同样处理。另：两个请求可能不同步完成，`finished_req_ids` 要跨 step 累计。
- 如何避免：runner 状态的生命周期以 `SchedulerOutput` 的 `new_reqs / finished_req_ids` 为唯一来源，不要从 `StepResults.finish` 推。
- 相关：`crates/core/src/runner.rs::update_states`、`models/glm53/tests/runner_parity.rs`

## 2026-10-14 用 scratch 缓冲的 `shape[0]` 当 token 数，把 max_tokens 当成了本步 token 数

- 现象：MLA forward 内 `rope_n` ensure 失败：`q.len 12288 < 64*4*16`，positions 只有 14。
- 根因：所有层的 scratch 按 `max_tokens` 预分配；`forward` 里 `let t = x.shape[0]` 取到的是预分配的 64，不是本步的 14。
- 解法：`forward(stream, gemm, t, x, ...)` 显式传 `t`；所有 `_n` 后缀的 kernel 包装按"容量 ≥ 所需"校验；GEMM 的激活/输出缓冲也改 `>=`，权重仍精确相等。
- 如何避免：张量形状只用于校验容量，不用于推导"本步有多少 token"；token 数永远从调度器/输入显式传。
- 相关：`crates/layers/src/{mla,nn}.rs`、`crates/kernels/src/{ops,gemm}.rs`

## 2026-10-14 Rust 包装层参数顺序与 C ABI 不一致（copy_cols）

- 现象：`copy_cols: range` ensure 失败，但调用处参数看起来合理。
- 根因：`ops.rs` 里 `copy_cols_bf16` 的参数顺序写成 `(src_cols, col0, width, dcol0, dst_cols)`，头文件是 `(src_cols, col0, dst_cols, dcol0, width)`；调用方按头文件传。
- 解法：包装层参数顺序改成与 `dinfer_kernels.h` 完全一致。
- 如何避免：写包装时直接从头文件复制参数列表；考虑在 ops.rs 里给多 usize 参数的函数用结构体参数（P4 起）。
- 相关：`crates/kernels/src/ops.rs::copy_cols_bf16`

## 2026-10-14 greedy 对拍在 bf16 平局步必然分叉，不是 bug

- 现象：prefill logits 全对、32 步 decode logits 最大偏差 0.007，但 greedy 第 7 步 token 不同。
- 根因：HF 该步 top1=0.508、top2=0.504，差 0.0039 = bf16 在 0.5 附近的 1 ULP；两边各自的累加顺序决定谁略大，无法也不应要求一致。
- 解法：对拍口径改为三层：① 全词表 logits max abs < 0.05；② argmax 只在 HF top1-top2 gap > 0.02 时要求一致；③ greedy 允许在平局步分叉，且分叉 token 必须是 HF top-2 之一。
- 如何避免：所有 greedy 对拍都带 `expect_step_logits`（HF 全序列前向逐步取最后一行），用 gap 判定平局，不用"token 序列完全一致"做唯一标准。
- 相关：`models/glm53/tests/tiny_parity.rs`、`scripts/verify/gen_tiny_glm.py`、`docs/loops/p3-mla-dense.md`

## 2026-10-14 top-p 掩码与 vLLM 口径不一致：top-p 必须在 top-k 之后的分布上算

- 现象：`topk_topp_mask` 对拍，`k=5, p=0.5` 这一行 kernel 保留 5 个、参考保留 2 个。
- 根因：第一版 kernel 把 top-k 阈值和 top-p 阈值各自在 **原始分布** 上算再取 max；vLLM `apply_top_k_top_p_pytorch` 是先 top-k 置 -inf，再对剩余元素 softmax 后做 top-p（`probs_sum <= 1 - p` 置 -inf，至少留 1 个）。两者在 k 小、p 小时结果不同。
- 解法：kernel 里 top-p 的 softmax 分母与质量累计只统计 `v >= top_k_threshold` 的元素；Python 参考改成升序 + `cumsum <= 1-p` 与 vLLM 完全同形。
- 如何避免：采样链的任何算子先抄 vLLM 的 PyTorch 参考实现到 `gen_kernel_refs.py`，再写 kernel；容差按"边界 token 可能差 ≤ max(2, 0.1%)"设。
- 相关：`crates/kernels/cuda/sampling.cu::topk_topp_mask_kernel`、`scripts/verify/gen_kernel_refs.py::mask_ref`、`vllm/v1/sample/ops/topk_topp_sampler.py:473`

## 2026-10-14 cargo 集成测试的 cwd 是 crate 目录，不是工作区根

- 现象：`tests/ref_tests.rs` 用相对路径 `test/kernel_refs` 读 `.npy`，10 个测试全部 "No such file"。
- 根因：`cargo test` 跑集成测试时 cwd = `crates/kernels/`。
- 解法：相对路径按 `env!("CARGO_MANIFEST_DIR")/../..` 解析；脚本里用 `realpath -m` 传绝对路径。
- 如何避免：测试读外部文件一律走环境变量 + `CARGO_MANIFEST_DIR` 兜底。
- 相关：`crates/kernels/tests/ref_tests.rs::refs()`

## 2026-10-14 clippy 偶发 "failed to run rustc to learn about target-specific information"

- 现象：`cargo clippy` 报 rustc 解析 `<anon>` 失败，错误内容是一段 shell 脚本（`awk '-F: '`、`SN='Not Specified'`）。
- 根因：cargo 用 `rustc -` 从 stdin 探测目标信息，本机某个后台进程（疑似监控 agent）往同一终端 stdin 写了东西。与代码无关，重跑即过。
- 解法：重跑。若频繁出现，给 verify 脚本的 cargo 命令加 `< /dev/null`。
- 相关：`test/2026-10-14/p2.log`（第一次运行）

## 2026-10-14 criterion `iter_batched` 把 setup 成本算进了被测函数

- 现象：调度器 bench 报 197 / 310 µs，但手工 `Instant` 分段计时只有 62 / 92 µs；两条 bench 之间 110 µs 的差距与代码路径差异不符。
- 根因：`iter_batched(setup, routine, SmallInput)` 每次迭代都跑一遍 setup（重建 1024 个请求、prefill 一步）。criterion 文档说 setup 不计时，但 `SmallInput` 模式下 setup 产生的内存分配 / cache 污染会让 routine 本身变慢，且 setup 里 `Scheduler::new` + 1024 次 `add_request` 的 drop 成本落在 routine 之后的同一测量窗口内。
- 解法：对"有持久状态、每步修改状态"的被测对象，用 `b.iter(|| ...)` 持续跑同一个实例；把 `max_tokens` / `max_model_len` / 块数设到测量期间不会触发完成或抢占。
- 如何避免：写 bench 前先用 `examples/probe_sched.rs` 这类手工计时对一遍量级；两者差 > 30% 先怀疑 bench 写法。
- 相关：`crates/core/benches/scheduler.rs`、`crates/core/examples/probe_sched.rs`、`docs/loops/p1-scheduler-perf.md`

## 2026-10-14 Claude Code 的 Bash 工具被平台安全分类器间歇拦截

- 现象：`Bash` 调用返回 "claude-opus-5[1m] is temporarily unavailable, so auto mode cannot determine the safety of Bash"，连续数十次；`Read/Write/Edit` 多数时候可用。
- 根因：Claude Code auto mode 依赖一个独立分类器判定命令安全性，该服务不可用时所有非只读工具被拒。与本项目无关。
- 解法：拦截期间只做写文件；把要跑的命令列给用户在终端执行并贴回输出；或用户关闭 auto mode（本次用户说"拦截已经放开"后恢复一段时间，随后又复发）。
- 如何避免：长命令写进 `scripts/`，拦截时用户可直接跑；每次会话开头先用一条无害命令试探工具是否可用。
- 相关：`docs/progress.md` 2026-10-13/14 日志

## 2026-10-14 cc crate 的 `cuda(true)` 给 nvcc 塞 gcc 参数导致编译失败

- 现象：`cargo build` 报 `nvcc fatal : Unknown option '-ffunction-sections'`，另有 `incompatible redefinition for option 'optimize'`。
- 根因：`cc::Build::cuda(true)` 仍把 debug profile 的 gcc 参数（`-ffunction-sections -fdata-sections -O0 -gdwarf-4`）原样传给 nvcc；nvcc 不认。
- 解法：`build.rs` 直接 `Command::new(nvcc)` 编 `.cu` 为 `.o`，再用 `cc::Build::object()` 只做归档。
- 如何避免：所有 CUDA 编译不经 `cc` 的 compiler 路径；需要 host 侧参数用 `-Xcompiler` 显式传。
- 相关：`crates/kernels/build.rs`、`test/2026-10-14/build-glm53.log`

## 2026-10-14 k8s Pod 申请了 `nvidia.com/gpu` 但容器里没有 nvidia-smi

- 现象：`nvidia-smi: not found`，`/dev/nvidia*` 不存在，但 `/proc/driver/nvidia/gpus/*/information` 能看到 4 张 RTX 4090。
- 根因：集群用 containerd + nvidia RuntimeClass，默认 runtime 不注入 GPU 设备与驱动库；device plugin 只负责调度计数。
- 解法：Pod spec 加 `runtimeClassName: nvidia`。
- 如何避免：所有 GPU Pod 模板都带该字段；`scripts/dev/k8s/*.yaml` 已加。
- 相关：`kubectl get runtimeclass` → `nvidia`

## 2026-10-13 本机 3090 跑不了两模型的任何快路径 kernel

- 现象：vLLM 中 Kimi-K3 / GLM-5.3 的 MLA decode（FlashMLA）、FP8 块 GEMM（DeepGEMM）、MXFP4 MoE、KDA fused decode 全部要求 sm90 / sm100。
- 根因：RTX 3090 是 sm86，无 FP8 / FP4 张量核。
- 解法：每类算子先写 sm80+ 可跑的参考版（bf16 / dequant 回退）做正确性验证；快路径在目标机器做（design.md 决策表）。
- 如何避免：所有 kernel 的 FFI 保持同一签名，参考版与快路径可互换；验证脚本按 `nvidia-smi` 的 compute capability 自动选路径。
- 相关：`docs/design.md` §6 §7；`vllm/models/kimi_k3/nvidia/kda.py:161-357`（各 backend 的架构门限）

## 2026-10-13 HF 上 Kimi-K3 / GLM-5.3 只有主仓库可匿名访问

- 现象：`moonshotai/Kimi-K3-Instruct|Thinking|Flash`、`zai-org/GLM-5.3-Air`、`THUDM/GLM-5.3` 返回 401；`moonshotai/Kimi-K3`、`zai-org/GLM-5.3`、`zai-org/GLM-5.3-Flash` 返回 200。
- 根因：前者不存在或为 gated 仓库。
- 解法：目标固定为 `moonshotai/Kimi-K3` 与 `zai-org/GLM-5.3`；`GLM-5.3-Flash`（`Glm5NextForCausalLM`，结构不同）不在范围。
- 如何避免：拉文件前先 `curl -s -o /dev/null -w '%{http_code}' https://huggingface.co/api/models/<repo>`。
- 相关：`/data/ref/models/*/meta.json`

## 2026-10-16 采样分布测试在 chat prompt 首 token 上"全部样本同一 token、L1 = 0"

- 现象：100–200 个不同 seed 的请求在同一 prompt 上采样（temperature 0.7 / top_k 20 / top_p 0.9），全部得到 `151667`（`<think>`），host 侧 mask 后候选集大小 1，测试当成"采样坍缩"失败。
- 根因：Qwen3 chat 模板后的首 token 几乎确定（logprob ≈ −2e-4），任何 top-p < 1 都只剩一个候选；不是 kernel 问题。
- 解法：先 greedy 跑 24 步，拿每步原始 logits 在 host 侧套同样 mask，选候选最多的一步（实测 step 23，候选 3），用 `prompt + greedy[..step]` 作采样 prompt。
- 如何避免：分布 / 随机性测试先用 host 参考确认该位置的候选集 ≥ 2；不要默认首 token 有随机性。顺带：N 个 36-token prompt 要一次 prefill 放下，`max_num_batched_tokens` 与 KV 块数要按 `N × ceil(len/16)` 算（100 × 3 块 → 预算 8192、512 块）。
- 相关：`models/qwen3/tests/sampling_parity.rs::mask_and_distribution_match_host_reference`

## 2026-10-16 远端编译 cudarc 报 `E0428: the name CUDA_VERSION is defined multiple times`

- 现象：4090 机器上装了 CUDA 13.0 后，`CUDARC_CUDA_VERSION=13000 cargo build` 在 `Compiling cudarc v0.18.2` 处报 `CUDA_VERSION` 重定义。
- 根因：Cargo.toml 固定了 feature `cuda-12040`（本机 3090 / CUDA 12.4），cudarc 的 build.rs 发现环境变量时**额外**再 `rustc-cfg=feature="cuda-13000"`，于是两个版本 cfg 同时为真。
- 解法：feature 改为 `cuda-version-from-build-system`，cudarc 在 build 时跑 `nvcc --version` 自动探测；本机得 12.4、远端得 13.0。不再设 `CUDARC_CUDA_VERSION`。
- 如何避免：多机器多 CUDA 版本时不要在 Cargo.toml 固定 `cuda-NNNN0`；确保 build 时 `PATH` 里的 `nvcc` 就是 `CUDA_HOME` 那个（`env.sh` 已保证）。
- 相关：`scripts/dev/README-remote4090.md`、`Cargo.toml` `[workspace.dependencies] cudarc`

## 2026-10-16 CUDA 13.0 下链接 `dinfer` 报 `undefined reference to __cxa_guard_acquire / __gxx_personality_v0`

- 现象：4090 机器（nvcc 13.0.88）`cargo build` 编译全过，最后链接 bin 时 133 个 undefined reference，全部来自 `libdinfer_kernels.rlib(*.o)` 的 `__device_stub_*`，符号是 `__cxa_guard_{acquire,release,abort}` 与 `__gxx_personality_v0`。本机 CUDA 12.4 不出现。
- 根因：nvcc 13.0 生成的 host 侧 kernel stub 用了函数内 static 的线程安全初始化（C++ ABI `__cxa_guard_*`）并带异常 personality，需要 libstdc++；`build.rs` 只链了 `cudart`，12.4 的 stub 恰好不引用这些符号。
- 解法：`kernels/build.rs` 加 `cargo:rustc-link-lib=dylib=stdc++`（12.4 下多链无害，已验证本机仍过）。
- 如何避免：nvcc 产物当 C++ 对象看待，始终链 `stdc++`；换 CUDA 大版本后先在最小 bin 上做一次完整链接再跑测试。
- 相关：`crates/kernels/build.rs`、`test/qwen3/2026-10-16/remote4090-migration.md`

## 2026-10-16 4090 上测试二进制 `error while loading shared libraries: libcudart.so.13`

- 现象：`cargo build` 通过，`cargo test -- --ignored` 一跑就 exit 127，`ref_tests-*: error while loading shared libraries: libcudart.so.13: cannot open shared object file`。
- 根因：CUDA 13.0 装在 `/data/yaochen/infer/cuda-13.0`（目录约束），不在 `/usr/local/cuda`，也没进 `ld.so.conf`；链接期靠 `rustc-link-search` 找到了 .so，运行期动态加载找不到。
- 解法：`env.sh` 加 `export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}`；`remote4090_baseline.sh` 开头 `source env.sh`。
- 如何避免：toolkit 不在系统路径时，PATH（nvcc）、CUDA_HOME（build.rs）、LD_LIBRARY_PATH（运行）三者一起设；`ldd target/debug/dinfer | grep "not found"` 作为 bootstrap 的最后一步检查。
- 相关：`scripts/dev/bootstrap_remote4090.sh`、`scripts/dev/README-remote4090.md`

## 2026-10-16 prefix cache 默认开后，e2e "流式 == 非流式" 逐字比较开始随机失败

- 现象：Q2-c 把 `enable_prefix_caching` 默认打开后，`serve_e2e_qwen3` 里"同一 prompt 发两次（非流式 / 流式）逐字相同"的断言在第 100+ 字符处翻一个词（"to the stars" → "of the deep"），greedy 和固定 seed 都会。
- 根因：第二次请求命中前缀块，调度只重算最后一块，attention 走 paged（带历史）路径；与第一次从 0 开始的 prefill 路径 bf16 累加顺序不同，logits 有 ~1e-2 级漂移，在 near-tie 位置翻 token。与 Q1 对 HF 对拍时的噪声底同源（pitfalls 2026-10-15 "HF 全序列重算 bf16 不前缀稳定"），不是功能 bug。
- 解法：需要逐字比较的 e2e 用 `--no-prefix-caching` 起服务；prefix cache 的正确性单独用 TTFT 下降 + 回答合理来验（`e2e_qwen3_prefix_cache`）。
- 如何避免：任何"两次请求输出逐字相同"的断言都要先问 KV 路径是否相同；要么关 prefix cache，要么只比到第一个 near-tie 之前。
- 相关：`bins/dinfer/tests/serve_e2e_qwen3.rs::spawn_server`

## 2026-10-16 `n=3` 同 seed 的流式与非流式输出不逐字相同（prefix cache 已关）

- 现象：`e2e_qwen3_request_features` 里 `n=3, seed=42, temperature=1.0` 非流式与流式各发一次，三个 choice 的文本在十几个 token 后各自翻词；而 `sampling_parity::seed_deterministic_and_batch_independent`（同 seed 两次、与另一请求混批）逐 token 相同。
- 根因：Philox (seed, offset) 本身可复现，但 logits 不是：3 个请求两次调用时 batch 组成 / 调度顺序（哪些 token 一起 prefill、decode 时 batch 大小）不保证相同，bf16 GEMM / attention 的累加顺序随 batch 形状变，logits 漂 1e-2 量级，在 near-tie 处翻 token。vLLM 同样声明"同 seed 仅在相同 batch 下可复现"。
- 解法：e2e 对 n>1 只断言形状保证（index 0..n、各有 finish、≥2 不同）；逐字可复现只在单请求同 seed（batch 形状确定）上断言；固定 batch 形状下的逐 token 复现由 `sampling_parity` 覆盖。
- 如何避免：任何"两次运行逐字相同"的断言必须保证两次的 batch 形状相同（单请求或受控的测试 harness）；服务层只承诺 OpenAI 语义，不承诺跨 batch 位比特一致。
- 相关：pitfalls 同日 "prefix cache 默认开后…"（同源）

## 2026-10-16 同为 `--gpu-memory-utilization 0.9`，dinfer 自动算出的 KV 只有 6.0 GiB，vLLM 19.4 GiB

- 现象：用户问"0.9 在 dinfer 里为什么没有"。参数有（默认 0.9），但 4090 上 dinfer `kv_cache: auto` 得 3534 块 / 6.0 GiB，`available_gib=16.3`（权重只 1.5 GB，却已用 7.2 GiB），vLLM 同卡同配置 19.4 GiB。
- 根因：logits 缓冲按 `max_tokens`（一步最多 token 数，4096）× vocab 151936 分配：model 的 bf16 logits 1.2 GiB、final_normed；executor 的 fp32 logits 2.3 GiB；再加预算公式里按 3 份 fp32 logits 预留 7 GiB。但采样只取每个请求最后一行，行数上限是 `max_num_seqs`（64），不是 4096。合计白占约 10 GiB。
- 解法：`Qwen3Model::load_with_budget` 加 `max_logits_rows`（服务 = max_num_seqs；对拍 = max_tokens），logits / final_normed / executor fp32 logits / 预算预留全部按它算；预留余量 1 GiB → 512 MiB。改后 4090 自动 8491 块 / 14.5 GiB（available 17.5）。剩余差距主要是 per-layer 激活 scratch（qkv/q/k/v/attn/mlp 中间 × 28 层 × max_tokens）约 2.5 GiB，Q3 做激活复用再收。
- 如何避免：任何按 `vocab` 宽的缓冲，行数只能是 logits 行数上限；自动预算要用真实分配后的 free，不要再手工"预留"一份估算值。
- 相关：`models/qwen3/src/model.rs::load_with_budget`、`models/qwen3/src/executor.rs`

## 2026-10-16 4090 上测试在"空闲" GPU 2 OOM：`layer 24 mlp` 分配失败

- 现象：`CUDA_VISIBLE_DEVICES=2` 跑 real_parity，模型加载到第 24 层 OOM；`nvidia-smi` 显示 GPU 2 已用 22 GiB。
- 根因：首次 `vllm_docker.sh run` 时（容器参数后来改过）有一个 `VLLM::EngineCore` 子进程逃出容器生命周期留在宿主上，占着 GPU 2（父进程是 `vllm serve /data/models/Qwen/Qwen3-0.6B`，不属于当前 `vllm-qwen3` 容器）。
- 解法：`docker top vllm-qwen3` 确认不属于它后 kill；GPU 2 回到 4 MiB，测试通过。
- 如何避免：跑 GPU 测试前先 `nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv`，确认目标卡无人占；`docker rm -f` 后也查一次有没有孤儿 EngineCore。
- 相关：`scripts/dev/README-remote4090.md` GPU 分配一节

## 2026-10-16 宿主 `nsys profile docker run …` 抓 vLLM 容器得到 0 个 kernel

- 现象：想给容器里的 vLLM 做 kernel 分解，宿主上 `nsys profile ... docker run ...` 跑完报告里没有任何 CUDA kernel；`nsys --attach-pid` 也抓不到。
- 根因：`docker run` 只是客户端，容器进程由 dockerd / containerd 启动，不是 nsys 的子进程，nsys 的 CUDA 注入层进不去；vLLM 官方镜像里也没有 nsys。
- 解法：把宿主的 CUDA 13 toolkit 目录（含 `bin/nsys`）只读挂进容器（`-v $R/cuda-13.0:/opt/cuda13:ro`），`--entrypoint /opt/cuda13/bin/nsys profile … python3 -m vllm.entrypoints.openai.api_server …`，加 `--cap-add=SYS_ADMIN`；结束用 `docker kill -s INT` 让 nsys 正常落盘。实测能抓到 kernel。
- 如何避免：对容器内进程做 profiling，profiler 必须在容器内启动被测进程；宿主 nsys 版本要与容器内 CUDA 驱动兼容（这里都是 CUDA 13）。
- 相关：`scripts/verify/stage_profile.sh`

## 2026-10-16 decode attention v2 kernel 在 ref 测试里 GPU 100% 卡死不返回

- 现象：`gqa_decode_d128_splitk_ref` 调新 v2 kernel 后进程不结束，`nvidia-smi` 显示 100% 利用率，要 `kill -9`。
- 根因：`const float sc = valid ? sum8(part) : -INFINITY;` —— `sum8` 内是 `__shfl_xor_sync(0xffffffff, …)`（全 warp mask），而 `valid` 在 lane 间不一致（尾部 key 不满 4 个时部分 lane 为 false），无效 lane 不进入 shuffle，有效 lane 永远等不到它们 → 死锁。
- 解法：shuffle 无条件执行（所有 lane 都调 `sum8`），再按 `valid` 选结果；无效 lane 的输入已清零不影响和。
- 如何避免：任何 `*_sync(0xffffffff, …)` 都不能放在 lane 间可能分叉的分支里；分叉只能在 shuffle 之后做选择。写 warp 级归约时先想"这个条件在 32 个 lane 上是否一致"。
- 相关：`crates/kernels/cuda/gqa.cu::gqa_decode_v2_kernel`

## 2026-10-16 bench7 的 8 个结果 JSON 丢失：`remote4090.sh push` 的 `--delete` 把远端 bench 目录覆盖成了本机的半成品

- 现象：4090 上 `bench7/` 只剩 `CHANGE`，趋势表没有 bench7；本机 `bench7/` 也只有 `CHANGE`（之前 pull 时 rsync 断管只拿到部分）。
- 根因：`push` 用 `rsync -az --delete` 把本机仓库同步到远端，而 bench 结果是**远端产出、本机只 pull 回来**的；本机那次 pull 不完整 → 再 push 时把远端完整的 bench7 删掉了。方向搞反的目录不能被 `--delete` 覆盖。
- 解法：`push` 排除 `test/qwen3/*/bench*/`、`stage/`、`perf-trend.*`；bench7 的数值从 `bench_compare7.log` 的 summary 表恢复（JSON 不可恢复，这一轮在趋势里按 summary 手工补）。
- 如何避免：产出在远端的目录只 pull 不 push；pull 后 `ls` 核对文件数再 commit；双向同步的目录绝不加 `--delete`。
- 相关：`scripts/dev/remote4090.sh`

## 2026-10-16 prefill v3（WMMA）要点：>48 KB 共享内存必须动态分配并 opt-in；P 以 bf16 进 tensor core

- 现象 / 约束：v3 一个 block 用 76 KB smem（Q 16K + K/V 16K + S 8K + P 4K + O 32K），静态 `__shared__` 上限 48 KB 编译期就报错。
- 解法：`extern __shared__` 动态 smem + 启动前 `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`（只需一次，用 `static bool` 守卫），launch 第三参数传字节数。4090 每 SM 100 KB → 1 block/SM，8 warp，够用。
- 数值：P = exp(s − m) 存成 bf16 再进 mma（FA2 同样做法），O 累加 fp32；对 HF 的 prefill logits worst-abs 从 0.81 → 0.97（bf16 噪声底 2.0 以内），greedy 32 步仍逐 token一致（一次 near-tie 翻转在允许口径内）。
- 如何避免：改 attention 精度路径后一定跑 `real_parity`（不是只跑 kernel ref）；ref 的 1e-2 容差对 32×32 小 fixture 不敏感，长序列累积误差要靠真模型对拍看。
- 相关：`crates/kernels/cuda/gqa.cu::gqa_prefill_tc_kernel / gqa_paged_tc_kernel`

## 2026-10-16 bench 固定 seed → 常驻 vLLM 的 prefix cache 全命中，c8-long TTFT 对比失真

- 现象：bench11 c8-long（8×1024 prompt）dinfer TTFT 85 ms vs vLLM 41 ms；但紧接着用同一 seed 再跑一次，dinfer 17.7 ms、vLLM 37.8 ms。vLLM `/metrics` 显示 `prefix_cache_hits/queries = 90%`，dinfer 48%。
- 根因：`bench` 的 prompt 由 `--seed`（默认固定）决定，每轮 bench 发的 prompt 完全一样。vLLM 从 Q3 开始一直常驻、KV 181k token，历史轮次的 prompt 全在缓存里，1024 token 的 prefill 实际只算十几个 token；dinfer 每轮改 kernel 后重启，缓存是冷的。因此历史各轮 vLLM 侧 TTFT（尤其 c8-long）被低估，dinfer/vLLM 比值被高估的不是 dinfer 而是对手的"冷"数据不存在。
- 解法：`bench_compare.sh` 每次运行取新 seed（`SEED=${SEED:-$(date +%s)}`，写进 summary），prompt 不与历史重复；两边都只能命中聊天模板前缀。bench12 起为新口径，与 bench≤11 的 TTFT / c8-long 不可直接比。
- 如何避免：对比压测时先看两边 `/metrics` 的 prefix cache 命中数；凡"常驻 vs 重启"的对比都要排除缓存状态差异。

## 2026-10-16 WMMA 版 prefill（v3）比 FA2 结构（v4）慢 4.8×：S/P/O 走 smem + 每 tile 3 次 block sync 是主因

- 现象：v3（WMMA 16×16×16，S/P/O 在 smem，76 KB → 1 block/SM）4090 T=1514 0.66 ms/层；v4（裸 `mma.sync.m16n8k16` + `ldmatrix`，每 warp 16 行 S/P/O 全寄存器，64-key tile，34 KB smem → 2 block/SM）0.14 ms。单发 1024 prompt TTFT 25 → 15 ms（vLLM 21）。
- 根因：WMMA 的 fragment 布局不公开，行级 softmax 只能把 S 存 smem 再按行读、P 再写回、O 乘 alpha 也要走 smem；每 tile 3 次 `__syncthreads`；smem 大到每 SM 只驻 1 block，装载和计算不能重叠。
- 解法：用 PTX `mma.sync.m16n8k16.row.col.f32.bf16.bf16.f32` 的公开 C 片布局（lane g=lane/4 持第 g 行和第 g+8 行、列 2(lane%4)+{0,1}）：4 lane 一行 shuffle 做 max/sum；P 的 C 片按 FA2 技巧直接 pack 成下一步的 A 片；V 用 `ldmatrix.trans` 取 B 片。K/V smem 行跨 136 个 bf16（272 B）避免 ldmatrix 8 行同 bank。
- 如何避免：attention 类 kernel 直接按 FA2 结构写，不要先用 WMMA 过渡；可用 `cold_shape` 的 q1024 单形状先对比，再接服务。
- 相关：`crates/kernels/cuda/gqa.cu::gqa_fa_v4_kernel<PAGED>`，round 37–39。

## 2026-10-16 "kernel 微基准快了、端到端不动"两次：decode splits、混批分流

- 现象：decode_v2 splits 256 → 1024 微基准 +20%，bench13 持平；混批分流（decode 行走 decode_v2）微基准 C 形状 -12%，bench14 持平。
- 根因：两处优化的绝对量（每层几十 µs × 28 层 ≈ 1–3 ms）相对 c8-long 当时 85–100 ms 的 TTFT 不可见；真正的大头是 prefill attention（每 1024-token prefill step ~11 ms attention）。
- 如何避免：改 kernel 前先用 nsys 总时间 × 占比估算"最多能省多少 ms / 请求"，低于端到端指标的 5% 不做；优先攻占比最大的 kernel。

## 2026-10-17 "微基准 / 结构性优化在闭环 bench 下看不到"：跨流 prefill/decode 并行

- 现象：executor 双流 + 调度免预算都实现正确，bench c8-long TPOT 3.9 ms 不变；step 计数显示 4000 步里只有 18 步同时含 decode 与 prefill 行。
- 根因：bench 是**闭环固定并发**（8 个 worker 串行发），1024 prompt 在预算 2048 下一步算完，一次压测只有 32 个 prefill step，其余 ~4600 步纯 decode；而且新请求到达时常常正好别的请求也在等——两类行天然错开。能藏的上限 = 32 × 20 ms / (32 × 128 token) ≈ 0.15 ms/TPOT。
- 如何避免：做结构性优化前先用 `runner step mix` 日志（mixed / pair / prefill_only 计数）确认目标形态在负载里真的出现；评估这类优化要用开环到达（泊松）+ 长短混合 prompt 的 bench。
- 相关：`docs/loops/q3-xstream.md` 第 7–9 轮；`crates/core/src/runner.rs::try_execute_pair`；`XSTREAM=0` / `--no-decode-exempt` 回退。

## 2026-10-17 serve_e2e 失败两种假阳性

- ① 3090 本机：`QWEN3_GOLDEN` 相对路径（`test/qwen3_golden`）在 `cargo test -p dinfer` 的 cwd（`bins/dinfer`）下找不到 → 永远传绝对路径（`$PWD/test/qwen3_golden`）。
- ② 4090 baseline：`remote4090_baseline.sh` 的 `run()` 只 grep `^test |test result|panicked`，e2e 子进程起服务失败（显存被上一步 parity 进程短暂占着）时 panic 行不含这些词，日志里只剩 `FAILED (rc=101)`。单独重跑 6/6 过。改进：baseline 失败时把该步全量日志 tail 出来；e2e 前 `sleep 3` 等显存释放。

## 2026-10-17 启动慢 1 s 的真凶：qkv / gate_up 拼接走了 f32 host 往返，而且不在任何计时里

- 现象：`model_load_ms` 2.0 s，但 `load_bf16` 统计（mmap touch + 转 Vec + H2D）只有 0.7 s；逐段加 `wall after` 累计墙钟才看出 layers 循环 1.6 s；再按 Attention/Mlp::load 内部分段，`Mlp::load` 1234 ms / 28 层。
- 根因：`Attention::load`（q|k|v）和 `Mlp::load`（gate|up）为了拼接，用 `load_f32_host` 把 bf16 读成 f32 Vec、host 上 concat、再 `Tensor::from_f32` 转回 bf16 上传：每层 ~3–4 千万元素 × 2 次 dtype 转换 + 3 次 memcpy。这条路不经 `load_bf16`，所以 LOAD_STATS 看不见。
- 解法：bf16 直接 `load_bf16` 上传，设备上 `memcpy_dtod` 拼进一个大 Tensor（零 host 计算）。4090 total 2.93 → 1.88 s。
- 如何避免：定位启动耗时先放"累计墙钟"（`wall after <阶段>`）二分，再在最大段内按子阶段累加；任何 `load_f32_host` / `from_f32` 在加载路径上出现都要问一句"为什么不是 bf16 直传"。
- 相关：`models/qwen3/src/model.rs::Attention::load`、`crates/layers/src/nn.rs::Mlp::load`、`docs/loops/q3-startup.md`。

## 2026-10-17 启动优化里两个"理论上应该快、实测更慢"的坑

- ① mmap populate / 页表填充与 `CudaContext::new` 并行：两者都重度占内核态内存操作（页表、驱动映射），并行后 cuda_init 250 → 430 ms，总时长不变。串行即可；真正并行有效的是 tokenizer（纯用户态 CPU）。
- ② 自建 pinned 双缓冲 H2D（64 MiB × 2 + 异步 DMA + 事件轮转）比驱动的 pageable `cuMemcpyHtoD` 慢（6.0 vs 7.8 GiB/s）：pageable 路径驱动内部已经是 staging 流水且用多个 copy engine；自建版的瓶颈变成单线程 `memcpy` mmap → pinned。要超过它必须多线程 memcpy + 多流，收益 <60 ms，不值。
- 如何避免：启动类优化每轮只改一项、4090 跑 3 次取中位；"并行"先问两件事是否争同一硬件资源。
- 相关：`docs/loops/q3-startup.md` 第 7、10 轮；`crates/kernels/src/device.rs::htod_into_staged`（保留未用）。

## 2026-10-17 `stage_profile.sh` 的 "ms/step" 不是实测每步时长

- 现象：c32 对比表给出 dinfer 0.97 vs vLLM 1.39 ms/step，看起来 GPU 已快 0.4 ms，但端到端 TPOT 3.6 vs 3.3 反过来。
- 根因：`nsys_stage_compare.py --steps` 的步数是 "完成 token 数 / 并发" 估的，两边 prefill 混批、抢占、warmup 都没算；而且 vLLM 侧 nsys 没开 `--cuda-graph-trace=node`，图内 kernel 不展开，"copy/misc 30%" 很可能是图节点被归错类。
- 如何避免：比 decode 每步时长用各自引擎的 step 直方图（dinfer `/metrics` `engine_step_execute_seconds`）或固定 batch 的离线微基准；nsys 只用来看 kernel 构成比例，不用来算 ms/step。
- 相关：`docs/loops/q3-perf.md` 第 46–47 轮。

## 2026-10-17 裸 `cudarc::driver::result::*` 调用在 engine 线程报 CUDA_ERROR_INVALID_CONTEXT / INVALID_HANDLE

- 现象：`PinnedUploader::flush`（裸 `memcpy_htod_async`）在 parity 测试里全过，一进服务就 `engine step failed: INVALID_CONTEXT`；`TimingEvent::elapsed_ms` 同样报 INVALID_HANDLE。
- 根因：cudarc 的 safe 封装（`CudaStream::memcpy_*`、`CudaEvent::*`）每次调用前都 `ctx.bind_to_thread()`；直接用 `result::*` 不会。测试在创建 context 的线程上跑所以没事，服务的 engine 是另一个线程。
- 解法：`Stream::bind_ctx()`（= `context().bind_to_thread()`），所有裸驱动调用前先调；代价一次 `cuCtxSetCurrent` 几百 ns。
- 如何避免：任何新增 `result::` 调用都放进 `Stream` 的方法里并以 `bind_ctx()` 开头；e2e（多线程）是唯一能抓到的测试，不能只跑 parity。
- 相关：`crates/kernels/src/device.rs::{PinnedUploader, TimingEvent, StagedUploader}`。

## 2026-10-17 pinned 打包上传：同一 step 内两次 flush 回卷 host 偏移 → 随机 ILLEGAL_ADDRESS

- 现象：`PinnedUploader` 第一版 parity / baseline 全过，c32 压测几百步后 `CUDA_ERROR_ILLEGAL_ADDRESS`。
- 根因：`cuMemcpyHtoDAsync` 从 pinned host 读是异步的；第二轮 `begin()` 把偏移回卷到 0 后 CPU 立刻 memcpy 新数据进同一区域，第一轮 DMA 还没读完 → 设备收到半新半旧的 ids / block_table / slot_mapping → attention 越界。单请求 / 小 batch 时 DMA 太快碰不到，B=32 时碰到。
- 解法：host 区按 step 分配（step 开头回卷一次），同一 step 内多轮 flush 各用不同区间；下一 step 开始前 step 末尾的 D2H 同步已保证所有 DMA 完成。
- 如何避免：任何"host 侧写 + 异步 DMA 读"的缓冲都要么双缓冲 + 事件，要么保证同一同步窗口内不重写；压测必须包含 B≥32 的场景。
- 相关：`crates/kernels/src/device.rs::PinnedUploader`，`docs/loops/q3-perf.md` 第 50–51 轮。

## 2026-10-17 `tmux kill-session` 杀不死 `cmd | tee log` 里的 serve 进程：常驻 :8012 跑的是 20 小时前的旧二进制

- 现象：修了 block_table 跨度后 GPU 3 单测 c8 TPOT 2.53，GPU 0 "常驻服务" 2.78 且怎么改都不动；`ps` 看到 serve 进程 uptime 72381 s、参数还是早上的 `--max-num-batched-tokens 4096`，机器上有 3 个 `tee serve.log`。
- 根因：tmux 会话里跑的是 `bash -c "source env; dinfer serve ... | tee log"`，kill-session 只杀 bash，管道两端的 serve / tee 成孤儿继续跑并占着端口；新起的实例要么绑端口失败退出，要么起来了但 bench 连的还是旧的（同端口竞争）。tee 覆盖同一个日志文件，看日志像是新进程在跑。
- 影响：bench16–22 里对 GPU 0 dinfer 的数字不保证是当轮代码；c1/c32/c8-long 那几轮的结论要用 bench23 重新对照。
- 解法：`scripts/dev/restart_dinfer.sh`：按 `--port` 模式 `pkill -INT`，等进程退、等显存回落，再起；结束打印 serve 进程数（必须是 1）。bench 脚本一律调它。
- 如何避免：任何"重启服务后测"的脚本都要在起新之前确认旧 pid 没了、端口空了、显存回落了；性能数字异常先 `ps -o etimes` 看进程年龄。

## 2026-10-17 用 Python 按 "起点 index … 下一个锚点 index" 切片替换函数体时把相邻函数一起删了

- 现象：重写 `Runner::execute` 为 begin/finish 时，切片终点选了 `/// 一行是否是 "纯 decode"` 的注释行，中间夹着的 `update_states` 整个被删，编译才发现。
- 如何避免：替换函数体时终点锚点用**该函数自己的结束**（下一个 `\n    pub fn ` / `\n    fn ` 行），或先 `grep -n '^    fn \|^    pub fn '` 列出函数边界再切；切完 `grep -c 'fn <name>'` 核对一次。
- 相关：`docs/loops/q3-async.md` 第 3 轮。

## 2026-10-17 验证异步调度等价性：必须用 batch=1 串行请求，不能用并发

- 现象：ASYNC_SCHED=0 / 1 两个服务，3 个并发请求 greedy 对比，1 条中途分叉；以为异步有 bug。
- 根因：并发时两边的 batch 组成不同（异步版新请求晚一步进 batch），bf16 下 near-tie token 翻转（pitfalls 2026-10-16 已记）。串行单请求 5/5 逐字相同。
- 如何避免：所有 "改调度 / 改 batch 组成" 的等价性测试用 batch=1 串行；并发只断言形状与完成。

## 2026-10-17 "异步下 running 行都被 placeholder 挡住" 不成立：刚完成 prefill 的请求会被再次调度

- 现象：插队 prefill 用普通 `schedule()`，断言 "step 只含新请求" 在 4090 压测里触发 `cut-in step contains an in-flight decode row` 引擎退出（3090 e2e 没碰到：请求到达节奏不同）。
- 根因：`schedule()` 跳过 running 行的条件是 "placeholder 已覆盖到 max_tokens"，不是 "有 token 在飞"。一个请求的 prefill 步（同步）刚完成、首个 decode 还没被调度，它在 running 里且没有 placeholder → 会被排进来，而它的首 token 恰好是在飞的（来自上一步的采样）。
- 解法：`Scheduler::schedule_waiting_only()`——跳过整个 running 循环；插队步永远只有新请求。
- 如何避免：任何 "我以为 scheduler 不会排 X" 的假设都写成 scheduler 的显式 API，不靠隐含条件；对调度路径的改动要在 4090 压测（请求到达密集）下跑过，e2e 的节奏太稀。

## 2026-10-17 插队步结果排队晚发 → scheduler 在不知道首 token 的情况下调度下一步，首 token 丢失

- 现象：e2e `request_features` 的 stop-string 用例输出 `", 2, 3, …"`，首 token "1" 不见了；只在 CUT_IN=1 出现。
- 根因：engine 把插队步的 MRO 放进 `sync_done` 队列等 n 结束后统一发布，但在那之前已经 `schedule()` 了 n+1。scheduler 的 `update_from_output(so_c)` 没跑，请求状态还是 "prefill 完、0 输出"，n+1 把它再排一次；之后两次 update 的 append 顺序与实际采样顺序不一致。
- 解法：`schedule()` 之前必须把所有已完成步的 `update_from_output` 做掉；插队步 finish 后立刻 update + 推送，不排队。
- 如何避免：engine 里 "调度" 和 "回填" 之间的不变式写成注释并断言：`schedule()` 调用点上 sync_done 必须为空。

## 2026-10-17 A/B 脚本在同一端口连起两个实例时，bench 打在了残留的旧实例上

- 现象：host tier A/B（:8096，gib=0 / 16 两组）第一组吞吐 0.11 req/s、第二组日志里的 `num_blocks` 与本组参数不符。
- 根因：`( serve & ); … pkill -INT -f "port 8096"; sleep 3` ——没等旧进程真正退出、端口释放，新实例起不来或起来后 bench 连到旧的；与同日 "tmux 孤儿进程" 是同一类问题。
- 解法：`scripts/dev/ab_serve.sh start <port> <gpu> [args]`：起前 pkill + 等 pgrep 为空 + 等端口不响应，起后等 ready 并打印 **pid / 真实进程数 / num_blocks**；`stop <port>` 同样等退干净。所有 A/B 都用它。
- 如何避免：任何 "起服务 → 测 → 换参数再起" 的脚本，每轮打印被测进程的 pid 与关键启动参数并与预期核对；吞吐离谱（<1 req/s）先怀疑连错进程。

## 2026-10-17 显存压力下最后一个请求 "抢占自己" → 引擎 2M 空步/秒 livelock

- 现象：host tier A/B（`--num-blocks 512`，c16 × 1024 prompt）跑到剩 1 个请求时 bench 卡住；`engine_steps_total` 每秒涨 200 万、`generation_tokens` 不动、GPU 0%、CPU 一核满。
- 根因：running 循环分不到块时抢占末尾 running；末尾就是自己 → `preempt(self)` 释放块、回 waiting、本步空；下一步从 waiting 重排需要全部块，还是不够（prefix cache 里 ref_cnt=0 的块占位 / 池本来就小）→ waiting 循环 break → 空步 → 下一步再来。engine 对 "有请求但空步" 没有退避，满转。
- 解法（三层）：① scheduler 不自抢占：victim == self 时以 `Length` 结束该请求（`force_finished` 队列，engine 推 Finished）；② engine 空步 200 µs 退避 + 每 1000 次 warn（running/waiting/free_blocks）；③ 启动期校验 `max_model_len/block_size ≤ 0.9 × num_blocks`。
- vLLM 对照：靠启动期校验（`max_model_len` 必须能装进 KV）+ 单请求不抢占；没有空步退避（它的 step 本身有 GPU 等待）。
- 如何避免：所有 "分不到资源 → 重排" 的路径都要问 "重排后一定能前进吗"；压测必须含 "池远小于活跃集合" 的配置。单测 `self_preempt_under_pressure_makes_progress`。

## 2026-10-17 同一张卡上刚停掉的实例显存还没回落，紧接着的 auto num_blocks 测试算出偏小的池 → 误判 FAIL

- 现象：A/B 实例 `ab_serve.sh stop` 后立刻跑 baseline，`e2e_qwen3_auto_num_blocks` 的 80% 估计断言失败；单独重跑 PASS（10698 vs 估计 7848）。
- 根因：驱动回收 ~20 GiB 需要几秒；`stop` 只等进程退出，没等 `nvidia-smi memory.used` 回落。
- 如何避免：所有 "在同卡起服务" 的脚本起前等显存 < 2 GB（`restart_dinfer.sh` 已有，`ab_serve.sh`/`remote4090_baseline.sh` 补上）。

## 2026-10-18 把 "KV 命中单元" 的模型相关规则写进了公共 crate（用户指正）

- 现象：`core::kv::KvCacheManager::host_tier_lookup` 只支持单组全注意力、block_size == hash_block_size、group 0；GLM / K3 下直接返回空（不报错、不工作）。
- 根因：按 Qwen3 的形状写了 "一个命中 = 一个 16-token 块 × 全部层"，没想 K3 的 KDA 状态块（边界检查点，不是 token 区间）和 GLM 的两种缓存。字节布局放对了（模型 crate），策略放错了（core）。
- 原则：core 的 KV 层只放 **与模型无关的容器与钩子**（块池、哈希链、索引、order 类型、调度钩子）；凡是 "什么算一个命中单元、多组如何合成前缀命中、一个单元多少 token、一个单元多少字节" 都是模型 crate 的事，经 trait 注入。对照：GPU 侧的 `get_computed_blocks` 其实已经按组分别处理（attention 组取最短、Mamba 组边界查），host 侧应复用同一规则而不是另写一套。
- 如何避免：core 里出现 `groups.len() != 1` / `is_mamba[0]` / 具体 block_size 比较这类条件时，就是信号——该抽 trait 了。


## 4090 之间没有 PCIe P2P：CUDA IPC 跨卡 open 直接失败（2026-10-18）
- 现象：`cudaIpcOpenMemHandle` 在另一张 4090 上报 `peer access is not supported between these two devices`；`cudaDeviceCanAccessPeer` 全 0；`nvidia-smi topo -p2p r` 全 CNS。同卡 open 正常，64 MB D2D 0.15 ms（446 GB/s）。
- 根因：消费卡（GeForce）驱动禁 PCIe P2P；且 GPU0–3 / GPU4–7 分属两个 NUMA（`topo -m` SYS）。
- 解法：PD 跨卡 KV 搬运走 host：shm pinned 共享区（design §10.3 后端 b），P D2H 写、D H2D 读；同卡才用 cuda-ipc D2D。NIXL 在这台机器上也只能走 UCX host 路径，不引入。探针：`scripts/dev/probe_ipc.cu`（`probe_ipc A B`）。
- 避免：换机器先跑探针；有 NVLink / 数据中心卡时 cuda-ipc 跨卡才有意义。

## cudarc nccl：集合通信返回 `NcclStatus`，不是 `()`

- 现象：`Comm::all_reduce_in_place(..)` 返回 `Result<NcclStatus, NcclError>`，直接 `map_err` 后类型不匹配。
- 解法：`.map(|_| ())`（`NcclStatus::InProgress` 只在非阻塞 comm 下出现，我们用阻塞 comm 可忽略）。
- 另：开 `nccl` feature 后 cudarc build.rs 链接 `-lnccl`（dynamic-linking 下运行期 dlopen `libnccl.so.2`）；无 NCCL 的机器需 `--features no-nvcc` 以外再想办法（当前 `no-nvcc` 不去掉 nccl）。

## TP 多卡：各 rank 自动 KV 预算不一致 / 单卡进程未退干净

- 现象：`--tp 2` 首跑一 rank 算出 10392 块、另一 rank 报 "not enough GPU memory"：前一个单卡 serve 刚 `pkill -INT`，显存还没释放，rank 0 所在卡 free 只剩 8 GiB。
- 解法：测试脚本等 `pgrep` 消失再起下一进程；TP 下传固定 `--num-blocks`（service 已校验各 rank 块数一致，不一致直接报错并提示传 `--num-blocks`）。
- 相关：4090 Pod GPU 0–2 长期被 vLLM 基线 / 旧 dinfer 占着（23 GB），多卡测试用 3–7。Pod 镜像无 NCCL，见 loops/q5-tp-dp.md 环境节。

## 远端 `until ! pgrep -f "<pattern>"` 等待永不结束（2026-10-18 晚）

- 现象：4090 上 `until ! pgrep -f "cargo build --release -p dinfer"; do sleep 5; done` 等了 25 分钟，build 其实 40 s 就完了。
- 根因：`ssh host 'bash -c "...pgrep -f cargo build..."'` 的 bash 自身命令行就含该模式，pgrep 匹配到自己（和同时挂着的另一条 ssh 等待命令）。另外 `nohup ... &` 在 ssh 非交互会话退出时被一起收掉，脚本根本没起来。
- 解法：等编译改用 `cargo build` 的退出码 / 日志里的 `Finished`，不用 pgrep；远端长任务用 `setsid nohup ... < /dev/null & disown`（或 tmux），起完 `ps` 确认进程在。
- 避免：pgrep 模式里排除自己：`pgrep -f "[c]argo build"`（方括号技巧）。

## 2026-10-18 远端一条命令里 `pkill -f "<pattern> --role"` 把承载它的 `bash -c` 自己杀了
- 现象：`remote4090.sh run '... pkill -9 -f "target/release/dinfer serve.*--role"; ...; q4.sh ...'` 返回 exit 0 但 0 行输出、日志文件不存在、q4 没跑。
- 根因：`ssh host "bash -c '<整条命令>'"` 的 bash 进程命令行里含同样的文本，`pkill -f` 按整行匹配 → 先杀了自己（与此前 `pgrep -f` 匹配到自己同源）。
- 解法 / 避免：远端不要在同一条命令里用 `pkill -f` 清理；用 `kill $(cat *.pid)`、`pgrep -x` 或把 pattern 写成 `[d]infer` 这类不自匹配的形式；q4.sh 自己会清理上一轮进程，不需要预先 pkill。

## 2026-10-19 子代理把模型特定内容写进公用 crate
- 现象：受限解码子代理在 `crates/server/src/chat.rs` 硬编码 `enable_thinking=false`（Qwen3 模板变量名），在 `crates/guided/src/lib.rs` 的单测里写死 Qwen3 tokenizer 路径 / eos id / vocab。用户检查时发现。
- 根因：brief 里只说"core / kernels 不得依赖 llguidance"，没把"crates/ 下不得出现任何模型特定常量 / 变量名 / 路径"写成硬约束；子代理为了让 e2e 过就近取巧。
- 解法：模型相关的量一律走 `ChatBackend` 钩子（`grammar_factory / tool_call_tags / grammar_render_extra`）由 `models/<m>/service.rs` 提供；模型相关测试放 `models/<m>/tests/`。
- 避免：给子代理的 brief 里固定加一句"`crates/` 只放模型无关代码，模型名 / token id / 模板变量名 / 模型路径出现在 `crates/` 即为错误"；收尾 review 用 `grep -rniE "qwen|glm|kimi|<tool_call>|<think>|15164[35]|151936" crates/` 自查（注释里引用为例子可留）。


## 2026-10-19 JumpServer 连接 token 一次性
- 现象：B300 机器 token 第一次 ssh 成功，几分钟后同 token 再 ssh `Permission denied (password,publickey)`；第二个新 token 也直接被拒（可能是密码与 token 不配对，或 token 在生成后被别处用过）。
- 根因：JumpServer 的"连接 token"是一次性凭据（4090 那台靠公钥 + 常驻 ControlMaster 才稳定，掩盖了这一点）。
- 解法 / 避免：拿到 token **第一件事**就是 `remote_b300.sh up` 建常驻连接（ControlPersist），之后全部 `run` 复用；不要先单独 ssh 一下"试试"。长期方案：让用户在 JumpServer 上给账号配公钥，或给 `yaochen` 账号固定密码走资产形式登录。

## 2026-10-19 bench_compare.sh 在 8 卡机器上 exit 141（SIGPIPE）
- 现象：B300 上 `bench_compare.sh` 起了 serve、写了 CHANGE / GPU 就退出，exit 141，没有任何 json；`bash -x` 停在 `gpu_tag` 第二次调用的 `nvidia-smi --query-gpu=name | head -1`。
- 根因：8 卡可见时 `nvidia-smi` 输出 8 行，`head -1` 读完一行关管道，`nvidia-smi` 收 SIGPIPE 退出码 141，`set -o pipefail` 下整条命令失败 → `set -e` 退出。4090 Pod 只可见 1 张卡从未触发。第一次猜是 `docker inspect`（没先 `bash -x`）——猜错一轮。
- 解法：`nvidia-smi -i 0 --query-gpu=name`（只查一张卡，不再 `| head`）；`docker inspect` 那行顺带加了 `/version` 回退。
- 避免：`pipefail` 脚本里 `<多行命令> | head -N` 一律改成让生产者只产 N 行，或 `|| true`；新机器首次跑 verify 脚本**先 `bash -x`**，不要猜。

## 2026-10-19 同进程内 `cuIpcOpenMemHandle` 自己导出的 handle 失败

- 现象：B300 上自定义 one-shot all-reduce 建立时每个 rank `cuIpcOpenMemHandle(对端 handle)` 都报错（`custom_ar_finish -> false`），而 PD 的 cuda-ipc 槽区（P / D 两个进程）同一段代码正常。
- 根因：CUDA 限制——IPC handle 不能在导出它的**同一进程**里打开（文档：`cuIpcOpenMemHandle` "cannot be used to open a handle created in the same process"）。TP 是一进程多线程，每个 rank 一个主 ctx，不是多进程。vLLM `custom_all_reduce` 是多进程（一 rank 一进程）所以能用 IPC。
- 解法：一进程多线程下直接交换 `cuMemAlloc` 的裸指针（UVA 下进程内全局唯一）+ `cuCtxEnablePeerAccess(cuDevicePrimaryCtxRetain(对端设备))`（`CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED` 当成功），kernel 直接读对端地址。`comm.rs::enable_peer_access`；handle 仍导出随 `ArPeerInfo` 传递，留给将来跨进程。
- 避免：凡"同进程多 ctx 共享设备内存"都走 peer access，不走 IPC；IPC 只给跨进程（PD）。
