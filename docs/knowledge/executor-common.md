# executor 公共件（`dinfer_layers::exec`，第 8 轮抽出；用户 2026-10-20："调度不该每个模型实现一遍"）

| 公共件 | 作用 | 接口 | 模型侧只剩 |
|---|---|---|---|
| `AsyncResultRing` | 两槽 pinned D2H + 票据 FIFO，让 runner 的 `begin/finish` 两段式真正重叠 | `new(stream, depth, rows)`、`enqueue(stream, chosen, n) -> Ticket`、`wait(Ticket)`；`async_sched_enabled()` 读 `ASYNC_SCHED` | `supports_async` 返回 true；`begin` = enqueue + ring.enqueue；`finish` = ring.wait + 组装 |
| `DecodeGraphs` | 按 batch 桶懒捕获 / 全捕获 CUDA graph | `ensure(n, body)` / `get(n)` / `capture_all(body)`；`body(nb, CapturePhase::{Prepare, Body})`——**Prepare 在捕获外做 H2D**，Body 只 launch | 捕获体闭包（上传假 plan → forward_device → sample_device_fixed） |
| `StateSlotTable` | 递归状态槽分配（请求 → slot），`release(finished)` + LRU 兜底 | `begin_step(req_ids) -> fresh` / `indices` / `release` | 清零 fresh slot 的状态 |
| `PlanBuffers` | plan 上传到固定地址缓冲 + pad 到桶（graph 重放只换内容不换地址） | `upload(stream, uploader, plan, pad_to)`、`gather_inflight`、`save_chosen` | 模型特有列（GDN 的 state_idx / has_init）之后追加 |

## 规则
- 新模型 executor **直接用这四件**，不复制 qwen3 / qwen35 的内联实现（kimik3 已接 `StateSlotTable`，`DecodeGraphs` / `AsyncResultRing` 等 C-4b 做 executor 时接）。
- 发现 qwen3 里有、别的模型没有的性能机制 → 先抽公共件再接。
- 图捕获体内不得有 H2D / 分配 / 同步 / 跨流等待：graph 会把 host 指针记进去，重放读已释放内存 → `row_gather` 越界（`compute-sanitizer` 直接指到行；`CUDA_LAUNCH_BLOCKING` 只指下一个同步点）。
- executor 自持非默认流 + `disable_event_tracking`，cuBLASLt 句柄在该流上建（默认流不能 capture）。

## 接入经验（K3 C-4b，2026-10-20）
- 接 `DecodeGraphs` 前先 `grep -nE 'dtoh|synchronize' ` 整条 forward 路径：捕获体内任何同步 D2H 在 capture 下读到 0，表现是 **host 侧形状断言**（如 `LatentMoe: pairs 0 ≠ t·K`），错误信息完全不提 graph。K3 的 MoE tile 计划在 host 展平、AttnRes host softmax 都是这种 → graph 默认关直到 GPU 侧 tile 计划 + AttnRes 融合。
- `StateSlotTable` 容量留 1 个"废纸篓"槽给图捕获 / padding 行（qwen35 第 2 轮同做法），活请求的槽不被假 plan 碰。
- 需要装载前定大小的缓存（K3 MLA latent cache 在 `MlaLayer::load` 分配）不能按剩余显存算块数，用 `max_num_seqs × max_model_len / block_size` 上限。
- 三个模型的采样流水（温度 / top-k / top-p / 惩罚 / logprobs 上传 + fused sample）仍是三份同文 → 下一个公共件候选 `exec::Sampler`。

## 下一步（design.md §14.5）
任务图抽象：模型 crate 声明层结构 → 公共层生成图 → 同一张图可喂 megakernel 或退化成逐 kernel launch。`DecodeGraphs` 的闭包捕获体是它的前身。
