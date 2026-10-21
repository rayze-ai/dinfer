# Kaon-V3 适配（5090 ×8）——先把基线摸清，再谈超越

> 状态：**开工（2026-10-21）**。目标：在 RTX 5090 上把 Kaon-V3 跑起来，性能 ≥ 基线容器 `kaon-vllm-pegaflow` 的 **1.2×**，
> 达不到就继续优化直到满足。每轮干净编译 → 实测 → 有提升才 commit。

## 机器（2026-10-21 首次连通确认）

| 项 | 值 |
|---|---|
| 主机 | `host-172-16-20-59`（JumpServer 资产 172.16.20.59，用户标识"瑞航"） |
| 跳板 | `jump.aicloud.paigod.work:2222`，**token 一次性**，用完即失效；过期时间见用户给的表格 |
| 账号 | root |
| GPU | **8 × NVIDIA GeForce RTX 5090，32607 MiB each**，driver 显示 814 MiB 常驻 |
| 占用 | **GPU 7 已占 30.3 GB**（疑似基线容器），**0–6 空闲** |
| 连接 | `scripts/dev/remote_5090.sh {up,status,ssh,run,push,pull,put}`，凭据 `~/.ssh/jms_5090_{user,pass}` |

**token 纪律（重要）**：token 是一次性凭据，**第一条命令必须直接 `up` 建常驻 ControlMaster**，
不能先"探活"再建——探活就把 token 用掉了。之后所有操作走 mux。

## 模型

- 名称：**Kaon-V3**，路径 `/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed`
- 目录名说明是 HF cache 布局（`models--<org>--<name>`），**nvpf4 = NVFP4 量化**，`mixed` 表示部分层未量化
- **架构未知**（待读 `config.json`）：需确认 hidden / layers / 注意力类型（是否有线性注意力 / MoE）、
  量化 scheme（NVFP4 block scale 格式）、tokenizer、chat template
- 基线容器：`kaon-vllm-pegaflow`（pegaflow = B300 上见过的那个 KV/专家缓存服务）

## 待摸清（第一步，不改任何代码）

1. `config.json` 全文 → 架构参数、是否 MoE、`quantization_config`
2. 目录结构与权重文件清单、总大小、dtype
3. 基线容器怎么起的（镜像、命令、端口、`--max-num-batched-tokens` 等），**只读不改**
4. 基线实测：TTFT / TPOT / 吞吐 / 并发曲线，用统一的 `ttft_probe.py` + bench 口径
5. dinfer 现有模型 crate 里哪个最接近 → 决定是新建 `models/kaon3` 还是扩展现有

## 停止条件（机器可判定）

1. `scripts/verify/kaon_v3.sh` exit 0：服务起来、`/v1/models` id 正确、greedy 输出与基线逐 token 一致（或 perplexity 同档）
2. 性能：同并发 / 同 prompt / 同 out_len，**out tok/s 与 TTFT 双双 ≥ 基线 1.2×**
3. 不退化：c1 与 c32 两端都要达标（只赢高并发不算）
4. 迭代上限：30 轮；每 5 轮无进展回退审视路线

## 轮次

| 轮 | 改动 | 结果 | 结论 |
|---|---|---|---|
| 0 | 连通 + 环境勘察 | 8×5090 确认，GPU 7 被占 | token 一次性；待新 token 建 mux |

## 卡点

- **token 一次性**：每次拿到的 token 只能建一次连接。mux 断了就要新 token，所以要把"一次建成、长期复用"做对。
