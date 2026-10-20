# executor 公共件（`dinfer_layers::exec`，第 8 轮抽出；用户 2026-10-20："调度不该每个模型实现一遍"）

| 公共件 | 作用 | 接口 | 模型侧只剩 |
|---|---|---|---|
| `AsyncResultRing` | 两槽 pinned D2H + 票据 FIFO，让 runner 的 `begin/finish` 两段式真正重叠 | `new(stream, depth, rows)`、`enqueue(stream, chosen, n) -> Ticket`、`wait(Ticket)`；`async_sched_enabled()` 读 `ASYNC_SCHED` | `supports_async` 返回 true；`begin` = enqueue + ring.enqueue；`finish` = ring.wait + 组装 |
| `DecodeGraphs` | 按 batch 桶懒捕获 / 全捕获 CUDA graph | `ensure(n, body)` / `get(n)` / `capture_all(body)`；`body(nb, CapturePhase::{Prepare, Body})`——**Prepare 在捕获外做 H2D**，Body 只 launch | 捕获体闭包（上传假 plan → forward_device → sample_device_fixed） |
| `StateSlotTable` | 递归状态槽分配（请求 → slot），`release(finished)` + LRU 兜底 | `begin_step(req_ids) -> fresh` / `indices` / `release` | 清零 fresh slot 的状态 |
| `kv::StateSnapshotPool`（core，scheduler 持有） | 递归状态**快照**槽（design §15.7）：键 = 对齐位置的块 hash → 快照槽，LRU + pin；scheduler 据此发 `BatchPlan.recurrent_resume / recurrent_snapshot (req, 槽)` | executor 只做两件事：forward 前 `restore(槽 → 活槽)` + `has_init=1`；本步 kernel 之后 `save(活槽 → 槽)`（同流 dtod，图外） | 快照区显存（每槽 = 全部递归层状态）、`save/restore` 实现、`load` 从预算扣快照区 |
| `load::LoadProgress` / `StartupTimer`（2026-10-20，r9 遗留） | **装载骨架也是公共件**：逐层计时 / `mem_info` 显存增量 / GB/s / `layer=l/n` 进度 / `model loaded` 汇总；服务启动三段 `startup done model_load_ms / graph_capture_ms / warmup_ms / engine_ms / total_ms`（graph / warmup 由 `DecodeGraphs::capture_secs` + executor 自报并从模型段扣掉） | `LoadProgress::new(tp)` → `timed(what, ‖ …)`（顶层张量）→ `load_layers(stream, n, ‖l‖ Ok((layer, kind)))` → `finish()`；`StartupTimer::start()` / `model_loaded()` / `done(graph_secs, warmup_secs)`；`DecodeGraphs::report_startup(reason)` 打 `decode graphs captured/disabled` 一行 | 装层闭包（开分片 / 建表 / `XxxLayer::load_tp`）+ 一个 `&'static str` 标签（`kda/moe`、`gdn`、`mla+idx/moe`…）；模型名不进 `crates/` |
| `PlanBuffers` | plan 上传到固定地址缓冲 + pad 到桶（graph 重放只换内容不换地址） | `upload(stream, uploader, plan, pad_to)`、`gather_inflight`、`save_chosen` | 模型特有列（GDN 的 state_idx / has_init）之后追加 |

## 规则
- 新模型 executor **直接用这四件**，不复制 qwen3 / qwen35 的内联实现（kimik3 已接 `StateSlotTable`，`DecodeGraphs` / `AsyncResultRing` 等 C-4b 做 executor 时接）。
- 发现 qwen3 里有、别的模型没有的性能机制 → 先抽公共件再接。
- 启动日志（装载每层一行 + 汇总、`decode graphs captured` 一行、`startup done` 一行）只由 `load.rs` / `exec.rs` 打，模型 crate 不再各写 `Instant` + `mem_info` 差（2026-10-20 四模型各有一份、字段还不一致，kimik3 `fb8886e` 加的 GB/s、done=l/n 已收进公共件）。想看某模型独有阶段（qwen3 的 tokenizer / cuda_init）可另打一行，但不要重复三段。
- 图捕获体内不得有 H2D / 分配 / 同步 / 跨流等待：graph 会把 host 指针记进去，重放读已释放内存 → `row_gather` 越界（`compute-sanitizer` 直接指到行；`CUDA_LAUNCH_BLOCKING` 只指下一个同步点）。
- executor 自持非默认流 + `disable_event_tracking`，cuBLASLt 句柄在该流上建（默认流不能 capture）。

## 接入经验（K3 C-4b，2026-10-20）
- 接 `DecodeGraphs` 前先 `grep -nE 'dtoh|synchronize' ` 整条 forward 路径：捕获体内任何同步 D2H 在 capture 下读到 0，表现是 **host 侧形状断言**（如 `LatentMoe: pairs 0 ≠ t·K`），错误信息完全不提 graph。K3 的 MoE tile 计划在 host 展平、AttnRes host softmax 都是这种 → graph 默认关直到 GPU 侧 tile 计划 + AttnRes 融合。
- `StateSlotTable` 容量留 1 个"废纸篓"槽给图捕获 / padding 行（qwen35 第 2 轮同做法），活请求的槽不被假 plan 碰。
- 需要装载前定大小的缓存（K3 MLA latent cache 在 `MlaLayer::load` 分配）不能按剩余显存算块数，用 `max_num_seqs × max_model_len / block_size` 上限。
- 三个模型的采样流水（温度 / top-k / top-p / 惩罚 / logprobs 上传 + fused sample）仍是三份同文 → 下一个公共件候选 `exec::Sampler`。

## PlanBuffers 上传规则与四段计时（engine-gap 第 7 轮，2026-10-20）
- **每步小输入的 H2D 不是瓶颈**：B300 engine 负载稳态每 decode 步 ~17 次 pageable `cuMemcpyHtoDAsync`（4 B–128 KB，最大项 block_table = nb × max_blocks × 4）合计 GPU 30 µs + host 50 µs，< 0.5%。`PinnedUploader` 打包版（1 次 H2D + N 次 D2D 散开）host 调用数不减反增，默认关是对的；想真正少到 1 次要把所有 plan 列放进**一块**设备 arena（图地址固定不变），收益 < 0.3%，暂不做。
- 上传规则：列按实际 `nb`（桶）行写，不按 `max_rows`；block_table 必须保持 `max_blocks` 行距（kernel 以它为 stride、图里固定），只能少传行不能少传列；padding 行 slot=-1 不写 KV、GDN slot 指废纸篓。
- **四段计时**：`ModelExecutor::take_step_phases()`（默认 None）→ `ModelRunnerOutput.phases` → `/metrics` 的 `engine_step_phase_{upload,launch,wait}_{prefill,decode,mixed}_seconds`。upload = 状态 slot 处理 + plan / 采样 H2D；launch = forward + 采样入队（eager 时含 kernel launch、**工作区分配、kernel 懒加载**）；wait = 等 GPU（同步 D2H 或 async ring `finish`）。qwen35 已接；qwen3 的 `account()` 仍只打日志，接法同 qwen35 `set_phases`。
- **prefill 步 host 空转的三个来源与解法**（qwen35）：按需增长的 chunk 工作区 free+alloc（`GdnChunkWorkspace::with_capacity` 启动一次按 `max_tokens`，`KdaChunkPlan::fill` 复用设备缓冲）；prefill kernel 首次使用的 `cuLibraryLoadData`（启动 `warmup_prefill` 两条 65 token 假请求，slot=-1 / 临时 GDN slot / 不写 prev_chosen）；首次图捕获（默认全桶预捕获，`GRAPH_CAPTURE=lazy` 回退）。
- nsys 按步分类用 `scripts/dev/nsys_steps_sqlite.py`（稳态窗口 = 最后一次 ≥ 8 MB H2D 之后），不要用按 csv 切步的旧脚本。

## per-token 激活缓冲：全模型一套（ActScratch，四例已落地三例）
- **判断**：任何在层 `load` 里 `Tensor::zeros(&[max_tokens, 列数])` 的东西，先问"层间会同时存活吗"。层在同一条流上串行，层 i 的中间量在层 i+1 开始前已消费完 → **全模型只需一套**；按层数 × 预算爆炸（qwen3 28 层 × 16400 token × 30 KB ≈ 13.7 GiB，16384 预算直接 OOM；qwen35 64 层 ≈ 180 GB）。
- **怎么用**：模型 crate 里 `scratch.rs::ActScratch { … , max_tokens }` + `cols(cfg)`（列宽从**本 rank 的** cfg 推，TP 后已除）+ `bytes(cfg, t)`（分配前算，进启动日志）+ `new(stream, cfg, t)`；`ModelScratch.act` 给副流第二套，`swap_scratch` 就是一次 `mem::swap`。`Mlp` 走 `load_tp(…, 0, …)` + `forward_with_scratch`（`max_tokens == 0` 不分配，已有路径）。
- **陷阱**：共享块往往比单层逻辑行宽**宽**。GEMM 的 `ld = n`、`qkv_post` 逐行原地写、attention 的 `q_stride` 三者都必须是本层行宽，**不能**从 `numel / 行数` 反推；加 `numel ≥ t × 行宽` 断言。缓冲更宽是安全的（kernel 只读前缀）。
- **证据**：qwen35 / kimik3 / qwen3 三例（`models/*/scratch.rs`），qwen3 改前 16384 预算 3/3 OOM、改后 3/3 起来且 `num_blocks` 3050 → 10808（`test/qwen3/2026-10-21/actscratch.md`）。glm53 同构未改。
- **验证**：副套正确性用 `real_parity::real_scratch_swap_is_bitwise_identical`（主套 / 副套 / 换回，logits 逐位比）；**别用 HTTP greedy**，同二进制连跑两次就有 6/12 输出不同。

## 下一步（design.md §14.5）
任务图抽象：模型 crate 声明层结构 → 公共层生成图 → 同一张图可喂 megakernel 或退化成逐 kernel launch。`DecodeGraphs` 的闭包捕获体是它的前身。
