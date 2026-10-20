# 接新模型：方法与盲区清单（Qwen3 / GLM / Qwen3.5 / K3 四次沉淀）

流程见 `docs/model-onboarding.md`；这里只放会翻车的地方。

## tiny golden 的盲区
- **所有权重随机**（norm / bias / 门 / A_log / dt_bias 都不能全 1 全 0）。Qwen3.5 的 zero-centered RMSNorm `(1+w)` 被全 1 的 norm 权重完全掩盖，tiny golden 4/4 过而真权重乱码（logprob 全 −9）。接模型先 grep HF 源码 `1.0 + self.weight`。
- golden 的决定性步数要够：随机权重 greedy 常近平局，换 seed 扫到每条 prompt ≥ 8 个决定性步（`make_qwen35_tiny.py --seed`）；对拍"平局步跳过、决定性步必须一致"。
- tiny 跑单线程，暴露不了"引擎线程未绑 CUDA ctx"（首请求 `INVALID_CONTEXT`）：`forward_sample` 开头 `bind_ctx`。

## 真权重才暴露的事实（每个模型都要核分片 header）
- 真 dtype 与注释不符（K3 `A_log` F32 [128] 前 96 有效、conv 权重 F32；Qwen3.5 `A_log` bf16）。
- 投影是分开还是融合（K3 q/k/v/g 四个独立；vLLM 名字是融合后的）。
- norm 的 eps 可能用类默认而非 config（K3 MLA a_layernorm 1e-6 ≠ rms_norm_eps 1e-5）。
- 聊天渲染参照物：**基线 vLLM 的实际路径**（K3 在 `de8291e` 直接调 HF python，保留 `name`；Moonshot 线上 Rust renderer 则丢 `name`、浮点 serde 化）→ 做成方言开关，默认与基线一致。

## 递归状态模型（GDN / KDA / Mamba）的额外项
- prefix cache 要配合**递归状态按块缓存**（design §15.7，2026-10-20 已做，qwen35 已接）：命中长度对齐到 `SchedulerParams.recurrent_snapshot_tokens`（qwen35 默认 1024；K3 建议 1536），对齐位置的状态从 `kv::StateSnapshotPool` 的快照槽恢复（executor 读 `BatchPlan.recurrent_resume`，forward 前 dtod 进活槽、`has_initial = 1`），步末落在对齐点的请求按 `BatchPlan.recurrent_snapshot` 把活槽存进快照槽。新递归模型接入清单：① 状态结构体加快照区 + `save/restore_snapshot`（同流 dtod）；② `load` 从显存预算里扣快照区（qwen35：`snapshot_bytes`，默认 KV 预算 10%，每快照 = 全部递归层状态字节）；③ executor `enqueue` 两处各 ≤ 10 行（见 `models/qwen35/src/executor.rs` "递归状态按块缓存"）；④ service 把粒度 / 槽数填进 `SchedulerParams`，粒度 0 = 关 prefix cache；⑤ golden 加"同 prompt 两次 / 多轮 / 混批+抢占"三个 case（模板 `golden_parity.rs::run_engine_cfg`）。粒度 0 时一切回到旧行为，非递归模型不受影响。
- 状态槽表要在请求结束时归还（`BatchPlan.finished_req_ids` → `StateSlotTable::release`），LRU 兜底会误淘汰分块 prefill 中途不在场的活请求。
- 激活缓冲全模型一套（`ActScratch`），不要每层一套：预算 16384 时每层一套 = 170 GB。

## 对拍口径（与 vLLM 参考）
- 逐层末 token 范数 + cos（`QWEN35_TRACE` 式打点）定位首个偏离层，再对该层内部子块；HF 参考在 vLLM 镜像容器里跑（无 fla 时用 vLLM 内置 kernel，注意 `use_beta_sigmoid_in_kernel` 这类被 `**kwargs` 吞掉的开关）。
- 链式多层 cos 每层衰减 2–3e-4（bf16 + MoE 平局行），全模型只能看 logits top-k / 生成文本；隔离口径（每层喂 HF 输入）才断言 ≥ 0.999。

## MoE 模型的对拍口径（K3 C-5，2026-10-20）
- **单卡自身不逐位**：`moe_scatter_add_f32` 原子加顺序 + 路由平局 → 同一输入两次运行链式 5 层 final cos 0.99898。所以"tp2 vs 单卡链式 ≥ 0.9998"这类门槛在 MoE 上不可达，不是 bug。
- TP / EP 对拍的可达口径：隔离各层（两边 `topk_ids` 完全相同的行）cos ≥ 0.9999；路由一致行的 `expert_sum` ≤ bf16 1 ulp；对 HF 的 logits **top-1 一致**；链式 cos 与 top-5 只报数（tp2 实测 top-5 第 5 位因上游平局翻转不同，margin 一 ulp）。
- 中间量定位顺序：attn_out → mlp_in（TP attention + all-reduce 是否对）→ expert_sum → routed_out（EP 路径）；`TP_REDUCE_F32=1` 开关可排除 bf16 双舍入。
- 确定性 TODO：scatter_add 改按行归约（两边都受益）；之后链式门槛才有意义。
