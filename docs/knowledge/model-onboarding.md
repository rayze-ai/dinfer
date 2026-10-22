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
- prefix cache 要配合**递归状态按块缓存**（design §15.7，2026-10-20 已做，qwen35 已接）：命中长度对齐到 `SchedulerParams.recurrent_snapshot_tokens`（qwen35 默认 1024；K3 建议 1536），对齐位置的状态从 `kv::StateSnapshotPool` 的快照槽恢复（executor 读 `BatchPlan.recurrent_resume`，forward 前 dtod 进活槽、`has_initial = 1`），步末落在对齐点的请求按 `BatchPlan.recurrent_snapshot` 把活槽存进快照槽。新递归模型接入清单：① 状态结构体加快照区 + `save/restore_snapshot`（同流 dtod）；② `load` 从显存预算里扣快照区（qwen35：`snapshot_bytes`，默认 KV 预算 10%，每快照 = 全部递归层状态字节）；③ executor `enqueue` 两处各 ≤ 10 行（见 `models/qwen35/src/executor.rs` "递归状态按块缓存"）；④ service 把粒度 / 槽数填进 `SchedulerParams`，粒度 0 = 关 prefix cache；⑤ golden 加"同 prompt 两次 / 多轮 / 混批+抢占"三个 case（模板 `golden_parity.rs::run_engine_cfg`）。粒度 0 时一切回到旧行为，非递归模型不受影响。**收益（B300 Qwen3.8-27B，2026-10-20）**：prefix_repeat 0.5 负载下 rate 8 goodput 1.32 → 2.99（超 vLLM 2.87）、TTFT p99 402 → 117、cache hit 0.81；代价是快照区显存（默认 KV 预算 10%，每 1k token 快照 = 全部递归层状态 147 MB）。
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

## 四类"看着像标准实现其实不是"的配置项（Gemma4 实测，2026-10-21）
读新模型 `config.json` 时，下面四项**必须去 vLLM 源码确认语义**，凭名字猜一定错：
1. **`rope_parameters` 可以是「每种 layer_type 一套」**：Gemma4 里 full / sliding 分别有 theta 1e6 / 1e4，
   而且 `rope_type: "proportional"` 意味着 **inv_freq 的分母是 head_dim，不是 rotary_dim**
   （分母 512），**而且配对也变了**：vLLM `gemma4_rope.py` 把 inv_freq 零频补齐到 head_dim/2 对、
   整头 neox 旋转 → 非零频配对是 `(i, i+head_dim/2)`，不是"只转前 rotary_dim 维"的 `(i, i+rotary_dim/2)`。
   `Rope` 要有 `denom` 和 `span` 两个独立参数（`Rope::with_span`）。**partial rotary 一律去读 rope 类的
   `forward` / `_compute_inv_freq` 确认配对**，只看 config 字段会错两次（本项 2026-10-21 / 10-22 各错一次）。
2. **`attention_k_eq_v` 可能只对部分层成立**：Gemma4 的 full 层没有 `v_proj`（K 权重兼作 V），
   但 sliding 层**有**独立 `v_proj`。按层类型分别处理，不能全局一个开关。
3. **dense MLP 与 MoE 可以并联**：Gemma4 每层**无条件**执行 dense MLP，MoE 是另一条并行支路，
   两条输出相加后再过一次 `post_feedforward_layernorm`；再乘 `layer_scalar`。不是"二选一"。
4. **RMSNorm 不一定有权重**：Gemma4 的 `v_norm` 与 router 预处理是无权重 RMSNorm；
   我们 `rmsnorm_bf16` 必带 `w`，要加 `nullptr` 分支（乘 1）。

**同类陷阱**：Gemma 系的 norm 是「attn 前一个 + attn 输出上一个」，残差加在 norm **之后**——
层结构照抄 vLLM 的 `DecoderLayer.forward`，不要套 Qwen 的 `input_norm → attn → +res` 模板。

## 对拍参照必须自己起，且不能带任何 KV 缓存外挂

- **判断**：vLLM 对拍的参照实例必须**自己起**，参数与探针侧假设逐项对齐，且**不能带
  `--kv-transfer-config`**（PegaKV / LMCache / 任何外部 KV 连接器）。共享基线容器往往带业务配置
  （外部 KV、不同的 block size、别的 `max-num-seqs`），它的输出**不是逐位置 logprob 的合法参照**。
- **为什么**：外部 KV 连接器会让 K/V 经过搬运与可能的布局重排，数值路径与"纯 GPU 内 paged cache"不同；
  block size / max-num-batched-tokens 会改变 attention kernel 的 tile 划分与 batch 分派。这些差异会
  混进"我们 vs vLLM"的比较里，且**无法事后剥离**。
- **代价**（2026-10-22 实例）：整条 Kaon-V3 parity 线用了带 PegaKV 的基线做参照，
  由此得出的下限（1.51%/0.100/0.001）、判据线（5.27%/0.150/0.01）、以及"修复前 28.87%→修复后 15.26%"
  全部需要重测。同期还因为参照集是多进程拼接、以及 `--network host` 端口抢占抓到别人数据，
  得出过两条相反的错误结论。
- **怎么用**：`scripts/dev/kaon3_oracle_server.sh` 一律不带 `--kv-transfer-config`；
  参照集与下限集各用**一个进程、一次抓完**；跨进程下限单独测（同参数两个进程之差）。
- **证据**：`docs/pitfalls.md` 2026-10-22 三条（端口抢占 / 假就绪 / 误停基线），
  `test/kaon3/2026-10-22/kaon3-parity.md` 的作废记录。

## 逐位对拍的三大坑（Kaon-V3 2026-10-22 一次实测沉淀）

接一个**有权重目录 + vLLM 参照**的模型做逐位置 logprob 对拍，先按这个顺序排除，
能省掉几小时"以为是自己模型语义写错"的排查。三条都不是模型语义问题。

### 1. 常数张量的精度由它在 vLLM 里**注册成什么 dtype** 决定，不是由数学决定

`x * hidden_size**-0.5` 看着是个常数，但 vLLM 常写成
`register_buffer("root_size", torch.tensor(h**-0.5), persistent=False)` —— buffer 的 dtype
跟随模型（bf16），于是 `root_size.to(bf16)` **先把常数舍入到 bf16** 再乘。我们直接用 f64/f32 常数，
就差出 1.5e-3（bf16 尾数 8 位 → 相对分辨率 ~2e-3）。
**做法**：grep vLLM 里每个常数乘子，看它是 `register_buffer` / `nn.Parameter` / Python 标量。
buffer/Parameter → 按模型 dtype 舍入；Python 标量 → f32/f64。

### 2. RoPE 的 cos/sin 表：vLLM 存的是**模型 dtype（bf16）**，不是 f32

`vllm/model_executor/layers/rotary_embedding/base.py`：
`cache = self._compute_cos_sin_cache(); cache = cache.to(dtype)` —— `dtype` 是模型 dtype。
CUDA kernel 再把 bf16 表 `static_cast<float>` 做 fp32 乘加。所以正确口径是
**"bf16 表 + fp32 旋转"**，不是 f32 表。我们用 f32 表 → `q_rope/k_rope` 相对误差 1.5–1.8e-3。
**做法**：`Rope::with_bf16_cache()`（opt-in，把表按 bf16 就近偶数舍入后仍存 f32，kernel 不变）。

### 3. 参照实现自己就不精确，不能假设"参照即真值"（已用 fp64 实测三次）

用 fp64 真值（bf16 原始权重 + fp32 输入，float64 矩阵乘）校验 router GEMM：
**我们 max 7.4e-07 / top-8 错 0/24；vLLM max 1.5e-02 / top-8 错 3/24。**
vLLM 的 `GateLinear` 在 sm_120 走 `torch.mm(out_dtype=f32)`，累加精度不如逐元素 f32 归约。
后果：router logits 在 top-8 边界附近时参照会**选不同专家**，下游 MoE 输出差到 1e-1 —— 不是 bug。
**做法**：① 判据里单列"top-8 集合不一致的 token 数"，这些行不作判据；
② 常驻 fp64 校验（`scripts/verify/kaon3_router_truth.py`），改动后一条命令回答谁更准。

**同一结论在最终 logprob 上也成立（Kaon-V3 实测）**：拿 `final_norm` 输出 + bf16 原始 lm_head 权重
在 fp64 里算精确 logits/logprob，**我们差 0.02 nat、vLLM 差 0.28–0.39 nat**；
完整 logits 的 top-5 **id 与真值完全一致**，而 vLLM 的 top-5 顺序与真值不同。
**推论：接混精度模型时，"与 vLLM 逐位一致"是个错误的验收目标**——参照自身有 0.3 nat 级误差。
正确判据是"与 fp64 真值的距离不超过参照的"。实现方式：
`--hidden-all` 已顺手写 `final_norm.bin`、`--logits-out` 写完整 logits，
配 safetensors 原始权重即可在 fp64 里重算任一站点的真值。

### 4. 源码里读到的"vLLM 行为"必须 A/B 证伪（本轮 3 次里 2 次推断错）

| 假设 | 源码依据 | A/B 实测 |
|---|---|---|
| query 不量化 | `triton_attn.py:659` 的 `query.dtype == fp8` | **错**：关掉后 ΔNLL −0.0018→+0.0438 |
| P 要乘 p_scale=256 | `triton_unified_attention.py:592` | **错**：加上后 ΔNLL −0.0018→+0.0244 |
| softcap 三步各舍入 | `logits_processor.py:113` | 无变化（已回滚）|

原因：**不知道模型实际命中哪个后端 / 哪条 dispatch 分支**。先 `docker logs | grep "Attention backend"`
确认（Kaon-V3 是 `TRITON_ATTN`），再动手；动手就带 `ENV=1` 的 A/B 开关，用端到端指标判。

### 配套仪器：一次前向 dump 全层 + 全子模块

- `kaon3_probe --hidden-all <dir>`：每层残差流 `L{l}.bin` + `Lm1`（embed）；
- 同一目录还写 layer 0 的 `input_norm/q_norm/k_norm/v_norm/q_rope/k_rope/attn_out/o_proj/
  post_attn_norm/pre_ff_norm/mlp_gate_up/mlp_gelu/dense/router_in/router_logits/moe_y/ff2/post_ff_norm`；
- 参照侧 `scripts/dev/kaon3_vllm_sublayer.py` 用 forward hook 抓同名张量；
- `--sub-cmp <refdir>` 纯 Rust 逐项比。
**先用 4-token 短 prompt 冒烟**（几秒）：hook 脚本崩了要几秒发现，别等 10 分钟完整抓取。
