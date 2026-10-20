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
- prefix cache 默认**关**：命中后 scheduler 从 position > 0 喂 prompt，全注意力能复用 KV，递归状态没有按块快照 → 从零起、结果错（Qwen3.5 c8-long 86 条告警）。真支持要像 vLLM `mamba_cache_mode=align`：状态快照粒度 = 注意力 block 的整数倍，由 KV 页配平决定（K3 实测 1536 = 12×128）。
- 状态槽表要在请求结束时归还（`BatchPlan.finished_req_ids` → `StateSlotTable::release`），LRU 兜底会误淘汰分块 prefill 中途不在场的活请求。
- 激活缓冲全模型一套（`ActScratch`），不要每层一套：预算 16384 时每层一套 = 170 GB。

## 对拍口径（与 vLLM 参考）
- 逐层末 token 范数 + cos（`QWEN35_TRACE` 式打点）定位首个偏离层，再对该层内部子块；HF 参考在 vLLM 镜像容器里跑（无 fla 时用 vLLM 内置 kernel，注意 `use_beta_sigmoid_in_kernel` 这类被 `**kwargs` 吞掉的开关）。
- 链式多层 cos 每层衰减 2–3e-4（bf16 + MoE 平局行），全模型只能看 logits top-k / 生成文本；隔离口径（每层喂 HF 输入）才断言 ≥ 0.999。
