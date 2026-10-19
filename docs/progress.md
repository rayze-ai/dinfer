# 进度

<!-- ===== 当前状态（交接区）：每次收功整体重写，保持 ≤ 20 行 ===== -->
## 当前状态

- 更新日期：2026-10-20 03:xx（Qwen3.5 bench_b300_13 吞吐 0.97/0.97/0.92/0.91×、第 7 轮进行中；K3 C-3 对拍收尾中）
- 总览：Qwen3-0.6B 线 Q1–Q5 阶段性完成（Q3 四组 ≥ vLLM；Q4 PD / Q5 TP 功能通过，细节见 `docs/design.md` §10/§13 与 10-18 日志）；GLM 线 P0–P6 通过、真权重挂起；B300 单卡 B4 调优进行中（`docs/loops/b4-sm100.md`，c1/c8/c32 已追平或超越，c8-long 落后 38%）。
- **Qwen3.5-27B（`models/qwen35`）里程碑 5 完成（2026-10-19，B300-50 GPU 0，`docs/loops/qwen35-model.md` 第 10–20 轮）**：真权重 bf16 52 GB 加载 7–8 s；3 条 greedy prompt 与 vLLM 逐字一致。修了 3 个真权重才暴露的 bug：① 引擎线程未绑 CUDA ctx（`forward_sample` 加 `bind_ctx`）；② **HF `Qwen3_5RMSNorm` 是 zero-centered `(1+w)`**（input/post/final norm 加载时 `1+w`；tiny golden 之前全 1 权重是盲区，已随机化 + seed 8）；③ prefix cache 命中让 GDN 递归状态从零起（强制关；真支持要按块缓存状态）+ `SlotTable::release`（`BatchPlan.finished_req_ids` 新字段）。
- Qwen3.5 性能基线 `test/qwen35/2026-10-19/bench_b300_{1,2,3}` + `test/qwen35/perf-trend.*`：**bench_b300_3** TPOT c1 24.1 / c8 28.3 / c32 59.7 / c8-long 157.5 ms vs vLLM 10.3 / 11.2 / 12.9 / 11.2（0.07–0.43×）；TTFT c8-long 8.1 s vs 0.41 s。差距来源：GDN prefill 逐 token 顺序递归（`gdn.cu` 无 chunk 版）、decode 无 CUDA graph、eager 单流。vLLM 基线数字在 `docs/design-qwen35.md` §0.1/§0.2。
- **下一步（Qwen3.5 里程碑 6 性能，用户要求超越 vLLM）**：先开 `docs/loops/qwen35-perf.md`；① GDN chunk prefill kernel（规格 `docs/loops/qwen35-gdn.md` 第 7 条，可参考 `kda_chunk.cu` 的做法）→ c8-long TTFT；② decode CUDA graph（复用 qwen3 executor 的 bucket 捕获）→ c1 TPOT 24 → 目标 < 10；③ GDN 状态按块缓存重开 prefix cache。每轮 `bench_compare.sh test/qwen35/2026-10-19/bench_b300_N "改动"`。
- **K3 线**：kernel 级（KDA 顺序/chunk、MXFP4 解包/grouped GEMM）在 3090/4090/B300 全过（`test/kimik3/2026-10-19/b300_50/`）；`models/kimik3` 已有 config / weights / kda / latent_moe / layer（C-1、C-2 见下）。**vLLM Kimi-K3 基线已跑完归档**（tp8 独占 8 卡，tp6 因视觉塔 4096/6 失败；init 624 s；TPOT c1 8.70 / c8 13.09 / c32 19.15 / c8-long 13.11 ms，`test/kimik3/2026-10-19/bench_b300_vllm1/`，`docs/design-kimik3.md` §0.1；容器已停）。B300-50 GPU 2 复测 KDA 11/11、MXFP4 3/3、GDN 14/14（含 chunk），`kda_chunk_bench` 7.41×、`gdn_chunk_bench` 27.2×。**里程碑 C-1 完成（2026-10-20 00:xx，子代理）**：`models/kimik3/{weights.rs,kda.rs}`（真分片 header 核对：q/k/v/g_proj 四个独立 `[12288,7168]`、`A_log` F32 `[128]` 前 96 有效、conv/o_norm 权重 F32、`KimiRMSNorm` 非 zero-centered），B300 真权重第 1 层 KDA 单层对拍 `scripts/verify/k3_kda_layer.sh` exit 0（prefill chunk / decode 与 vLLM fla 及纯 torch f32 参考 cos ≥ 0.99997，rel 6–7e-3 = bf16 一 ulp；`test/kimik3/2026-10-19/kda_layer/`）。顺带修 `kda_chunk_*` 显式 `tokens`、`Linear::forward_rows`。**里程碑 C-2 完成（2026-10-20，子代理）**：`models/kimik3/src/{latent_moe.rs,layer.rs}`（MXFP4 专家 packed 常驻 15.72 GB/层 + `mxfp4_gemm_grouped`、路由/分桶复用 `moe.cu`、新 kernel `situ_and_mul_bf16`、AttnRes 按 HF `_forward_attn_residual` 用普通算子拼），B300 真权重第 1 层**整层**（KDA + latent MoE + AttnRes）对拍 `scripts/verify/k3_layer.sh` exit 0：整层 cos 0.999999 / rel 5.1e-3，各子块 rel 4–10e-3（bf16 ulp 量级），top-16 pair 级 99.71%、可判定行全中、3 个翻转全是平局行；单层 T=64 B300 39.5 ms（`test/kimik3/2026-10-20/layer/`，`docs/loops/k3-moe.md`「C-2」）。**里程碑 C-3 代码完成、对拍进行中（2026-10-20，子代理）**：`models/kimik3/src/{mla.rs,model.rs}`（MLA：fused `[q_a|kv_a_with_mqa|g]` 一条 GEMM、**无 RoPE 但 64 维原样参与点积**、sigmoid 输出门、latent 576 维 paged KV、prefill 展开 / decode 吸收复用 `mla.cu`；`KimiK3Model::load(dir, n_layers, LayerBudget)` 按层打开分片、块残差列表按 HF 语义、`output_attn_res` → norm → lm_head），`layer.rs` 改 `LayerBudget` / `StepMeta{kda,mla,slot_mapping}`；测试 `tests/model_parity.rs` + `scripts/verify/{gen_k3_model_refs.py,k3_model.sh}`（前 5 层：0 稠密、1/2/4 KDA+MoE、3 MLA+MoE，真 token T=31，逐层 cos + 末 token logits top-1 + decode）。结果与 commit 见 `docs/loops/k3-model.md`「结果」（跑完后填）。**下一步 C-4**：tokenizer（tiktoken + XTML）、service / executor（KDA slot 表 + MLA paged KV 共用 `StepMeta`）、MTP。C-2 TODO：AttnRes 融合 kernel（去 4 次 D2H）、grouped GEMM tile 计划 GPU 侧、专家 GEMM tensor core。踩坑（已进 pitfalls）：vLLM 镜像无 fla、`chunk_kda_with_fused_gate` 吞掉 `use_beta_sigmoid_in_kernel` 要预 sigmoid；`remote_b300.sh push --delete` 会删远端生成的 refs，先 pull 再 push。**Qwen3.5 里程碑 6 已开**（`docs/loops/qwen35-perf.md`），**第 1 轮 GDN chunk prefill 完成并进 CSV（`bench_b300_4`）**：TTFT c1 45 / c8 169 / c32 357 / c8-long 470 ms（vLLM 46 / 105 / 261 / 403），c8-long 吞吐 +445%；3090 顺序 41.5 → chunk 4.2 ms；两卡对拍过。**第 2 轮 decode CUDA graph（`bench_b300_5`，−1%）、第 3 轮 GEMM 自调优 + small_m kernel + GDN vsplit（`bench_b300_6`，−4%）都没动 TPOT；nsys node 级定位真因：`Linear::forward` 对预分配缓冲推断行数，decode 每步按 max_tokens=2112 行算 GEMM**（pitfalls 已记）。改 7 处 `forward_rows` 后 **`bench_b300_7`：TPOT c1 11.8（vLLM 10.3）/ c8 16.4（11.2）/ c32 34.1（12.9）/ c8-long 22.4（11.2）；TTFT c1 35 < 46；吞吐 0.88× / 0.68× / 0.41× / 0.55×**。**第 4 轮 GDN decode 寄存器 kernel 完成进 CSV（`bench_b300_9`；中间 `bench_b300_8` 持平是 `gdn_recurrent` 的 tokens 由缓冲推断导致新 kernel 未命中，已改显式参数）：TPOT c1 10.7（vLLM 10.3）/ c8 11.6（11.2）/ c32 15.9（12.9）/ c8-long 18.0（11.2）；吞吐 0.97× / 0.93× / 0.80× / 0.66×；TTFT c1 35 < 48，c8 / c32 落后 50%。** **第 5 轮两线完成进 CSV**：5B（`bench_b300_10`）自调优表展开非幂 M（借桶 + AlgoCheck，borrowed=456）+ 默认 algo 缓存（之前每次 forward 查启发式）→ **TPOT c1 10.7 / c8 11.7 / c32 15.0 / c8-long 14.3；TTFT c1 22 / c8 86 / c32 202 / c8-long 214，四组 TTFT 全部低于 vLLM（47 / 104 / 264 / 411）；吞吐 0.98× / 0.97× / 0.90× / 0.90×**。5A（`bench_b300_11`）GDN prefill 三段 mma 化（B300 每层 ho 1346 → 204 µs、conv 874 → 76、wy 652 → 221，nsys 证实命中）但四组持平 → prefill 步里现在是别的主导（排查中，见 loop 5D）。`bench_b300_12`（调度预算 2048 → 16384 对齐 vLLM）：TPOT c32 14.2 / c8-long 12.1（vLLM 12.9 / 11.2），吞吐 **0.98 / 0.97 / 0.92 / 0.91×**；预算是 TTFT↔TPOT tradeoff，bench 两边要一致（serve 默认仍 2048，bench 时加 `--max-num-batched-tokens 16384`）。第 6 轮（`bench_b300_13`）激活缓冲全模型共享（`scratch.rs::ActScratch`，170 GB → 2.8 GB，KV 回到 174 GiB），速度持平 **0.97 / 0.97 / 0.92 / 0.91×**。decode GEMM 已到 HBM 72–76%（`gate_up` 58 µs 即自调优最优）不再是差距来源；**剩余 c32 1.2 ms / c8-long 0.9 ms 在 B 维扩展的小算子（rmsnorm × 161、gated_rmsnorm、qkv_post、gqa_decode v3 B=32 splits）与混批**，nsys B=32 vs B=1 decode 步分解已做（loop 6d/6e）：**+4.3 ms = GDN decode kernel +1.7（43.6 µs/层 vs 下限 25）、cuBLASLt splitK reduce +0.9、GEMM tile +0.9、rmsnorm +0.3**。**第 7 轮本机部分完成（commit f64a466 / 93f7227）**：GDN decode 四变体 `GDN_DECODE_VARIANT=base|tr|pipe|pipe_base`（默认 tr；3090 已到带宽下限，四者差 < 5%，B300 才见分晓）+ `GDN_DECODE_GRID=`；rmsnorm 三布局 `RMSNORM_LAYOUT=block256|warp|block1024`（`ArchProfile::rmsnorm_layout` 默认 h∈(1024,8192] 选 block1024）；gated_rmsnorm warp 版。gdn 17/17、ops_parity 109/109、golden 4/4、四 feature 构建过。**待 B300**：① `gdn_decode_bench` 四变体对比定 Sm100 分叉；② `RMSNORM_LAYOUT` 三种对比；③ `bench_b300_14`。**第 8 轮（用户 2026-10-20 定，TODO 已进 design.md）：executor 公共件抽取**（AsyncResultRing / DecodeGraphs / StateSlotTable / PlanUpload 从 qwen3 内联代码抽到 core，qwen3 接回不退化，qwen35 接 async → 每步 0.6–0.9 ms GPU 空闲消失，是 c1 / c8 剩余差距的全部）。差距分析见 loop 6d/6e 与本会话 03:xx 回复：GEMM 已到 HBM 72–76% 不是来源；剩余 = 同步调度空闲 + GDN decode 1.7× 下限 + M=32 splitK reduce + 小算子延迟。**push 冻结到 K3 C-3 子代理 commit 为止**（见 pitfalls push --delete）。
- **恢复清单（/clear 后）**：① `B300_TAG=50 scripts/dev/remote_b300.sh status`（token 约 2 h 过期，过期向用户要）；② Qwen3.5 bench 口径 = GPU 0 dinfer `:8012`（tmux `q35`，`GRAPH_CAPTURE=eager … --max-num-batched-tokens 16384`）+ GPU 1 vLLM `:8013`（容器 `vllm-qwen35-b300`），`bench_compare.sh test/qwen35/2026-10-19/bench_b300_N "改动"` → pull 目录 + `test/qwen35/perf-trend.*` → commit；③ nsys 用 GPU 4 端口 8412（`--gpu-memory-utilization 0.6`，之前失败是旧进程没清干净）；④ 子代理 brief 模板见本会话各 loop 的"轮次计划"；三方并行时按文件集分工（gemm* / gdn* / kimik3）。
- B300-50 现状：GPU 0 dinfer qwen35 `:8012`（tmux `q35`）、GPU 1 vLLM qwen35 `:8013`（容器 `vllm-qwen35-b300`）、GPU 2–3 K3 C-3 子代理用（参考生成容器）、GPU 4 nsys、5–7 空。**用完要 `vllm_nerdctl.sh stop` / `nerdctl rm -f`**（用户 2026-10-19）。token 约 2 h 过期，过期向用户要。
- 新硬性规则（本次踩坑总结，已进 pitfalls）：tiny golden 的**所有**权重必须随机（norm / bias / 门都不能全 1 全 0）；带递归状态的模型（GDN / KDA / Mamba）prefix cache 默认关。

## 2026-10-18

- PD v2 (d) 跨卡：P GPU 3 / D GPU 4，TPOT 4.72 → 3.75（单机 3.53）、吞吐 1464 → 1786、TTFT 67 → 58 ms；Q4 PD 功能线收口。GPU 2 被别人进程占用，首次起 D OOM（pitfalls）。
- PD v2 (c)：proxy 先发 P 首 chunk、D 在飞；`first_token` 传 D 使超时回退不重复首 token（修了潜在 "redred" 类隐患的另一入口）。3090 e2e PASS；4090 e2e + 三故障 PASS，TTFT p50 97 → 67 ms、p99 423 → 140 ms，TPOT / 吞吐持平。loop 第 14–15 轮。两次远端跑空：`cargo build -p bench` 包名其实是 `dinfer-bench`；q4.sh 默认模型路径是本机的，4090 要 `QWEN3_DIR=/data/yaochen/infer/models/qwen3-0.6b`。

## 2026-10-17

- 00:00–00:30 `q3.sh` 重跑 exit 0 → **Q3 正式通过**。prefill v5（cp.async + NBUF 环形缓冲）：3090 双缓冲更慢，默认单缓冲。启动日志分段（weights host/h2d、scratch、kv）。`design-qwen3.md` 新增 §5（prefill_tc vs paged_tc：同工作量只差 3–30%，1.8× 是形状差异；服务选 paged 因历史 KV 零拷贝 + 调度统一）与 §6（优化空间：megakernel、跨流并行、启动、KV fp8、优先级）。commit 见 git log。
- 00:30–01:30 Q3-j 跨流并行 9 轮：Event API、ModelScratch 整套换入换出、executor AltSet + `forward_sample_pair`、runner 拆分散回、scheduler decode 免预算；bench17/19 持平，step 计数证明闭环 bench 下混批步极少（18/4000）；收益上限分析进 pitfalls。可读性整理一轮（helper 拆分、删 v3 302 行）。design.md §10 新增算子插拔 / 第三方 KV 方案（用户新需求）。
- 01:30–02:10 启动定位（loop `q3-startup.md` 4 轮）：`wall after` 累计墙钟二分 → layers 循环 1.6 s → `Attention::load` / `Mlp::load` 的 f32 host 拼接共 ~1 s；改 bf16 直传 + 设备拼接。4090 total 2.93 → 1.88 s，baseline PASS。pitfalls +1。
- 02:10–02:40 启动第 5 轮：BF16 权重 mmap 直接 H2D（host 拷贝 0）、tokenizer 独立线程并行；4090 total 1.88 → 1.05 s，baseline PASS。design-qwen3 §6.3 改为实测分段与剩余项。
- 02:40–03:40 启动第 6–10 轮：rope 多线程 122→21、mmap 并行填页 190→110、懒捕获图 engine 258→45、KV uninit、去逐张量同步；两次无效尝试（open∥Device::new 争资源；pinned 双缓冲 H2D 更慢）已回退并记录。4090 total **0.61–0.68 s**，baseline 全过。
- 03:40–04:10 bench 开环泊松到达 + 长短混合 prompt；`bench_poisson.sh` 首轮：TTFT 领先、TPOT 8/16 持平、24 req/s 落后 7%；跨流开关无差别（待 nsys 验证是否真并行）。
- 04:10–04:40 nsys 证实跨流真并行（39% 重叠）但混批步仅 5%；c32 stage compare 显示 GPU 构成 dinfer 不差，但 ms/step 口径不可靠（pitfalls）；c32 差距定位转下一会话。
- 04:40–06:00 decode B=32 差距定位：metrics 直方图口径 3.46 vs vLLM 3.40（离线）；launch 610 µs = block_table 全跨度上传（1.3 MB/步）；PinnedUploader 实现 + 两次翻车（INVALID_CONTEXT：裸调用没 bind ctx；ILLEGAL_ADDRESS：pinned 区回卷竞态）+ A/B 零收益 → 默认关；真修复 = block_table 跨度改 max_model_len/block_size。
- 06:00–07:00 发现 :8012 孤儿进程（20 h）；`restart_dinfer.sh`；干净重启 bench23：四组 1.12/1.06/1.05/1.01× 全 ≥ vLLM；block_table 跨度修复 launch 610 → 250 µs。pitfalls +3（裸驱动 bind_ctx、pinned 复用竞态、tmux 孤儿）。
- 07:00–08:30 Q3-l 异步调度 6 轮：runner begin/finish + Pending、engine pending/sync_done 双槽、executor row_gather 图内 + PinnedSlot 双槽 D2H；两处坑（切片误删 update_states；in-flight 容量 1 vs 设计 2）。bench24 四组 1.13/1.08/1.08/1.01×。CLAUDE.md 加 "超越不是重写" 规则 + design §11 采用/改进台账。
- 08:30–09:10 Q3-m 采样四 kernel 融合（row_* 原语共用，逐位同语义 ref），bench25 1.14/1.08/1.08/1.02×。
- 09:10–09:40 Q3-n gather+rmsnorm 融合（逐位 ref），bench26 持平；剩余两项并入 megakernel 阶段。
- 09:40–11:00 Q3-o 插队 prefill 6 轮：scheduler `schedule_waiting_only`、engine `publish_step` 提前回填、`CUT_IN_MAX_ROWS` 门槛；两次翻车（隐含假设、首 token 丢失）进 pitfalls；bench28 TTFT c8 14 → 8、c8-long 54 → 35。
- 11:00–11:30 design §10 细化完成（算子插拔 OpSet 三方案；KV 分层 KvTier + 4 个调度接入点 + PinnedHostTier → Remote → PD）。盘点：26 个 k:: 入口 + 1 Gemm（26 个 k:: 入口 + 1 个 Gemm）与 KV 管理器钩子（get_computed_blocks / allocate_slots / cache_blocks / free），再写成可落地的 trait 与接入点。
- 11:30–12:30 Q3-q OpSet 方案 A：opset.rs 四组 trait + DefaultOps，模型泛型化，moe/glm53 调用点同步；bench29 零差异。
- 12:30–14:30 Q3-r PinnedHostTier：索引 / 调度接入 / executor 竞技场 + copy stream / CLI / 指标 / e2e；第 7 轮 A/B 发现每块 28 次拷贝的 host 开销（TPOT +0.5）→ pack/unpack kernel + 每块一次拷贝；unsafe 全部下沉到 kernels::HostArena。
- 14:30–16:00 A/B 两次翻车（残留实例 → `ab_serve.sh`；引擎 livelock → 不自抢占 + 退避 + 启动校验 + 回归单测）；pitfalls +2。
- 16:00–17:30 第三次 A/B 发现 forced-Length 误判（末尾≠唯一）→ 修 + 单测；baseline 7/7；A/B 定论 0 失败、tier 命中但吞吐 -5%；泊松 / A/B 结果追加进 CSV（用户问）。

## 2026-10-16

- 03:10 会话第三次因 `Input is too long` /clear，按交接区恢复。恢复时工作区：Q2-a 采样 kernel 未提交改动 8 文件（+483 行）：`sampling.cu` 新增 `penalties_f32 / temperature_f32 / min_p_mask_f32 / logprobs_f32`，`ops.rs/ffi.rs` 封装，`ref_tests.rs` 4 个新用例，`gen_kernel_refs.py` 对应参考；`core::sample` 加 `has_penalties / validate`；`docs/loops/q2-serving.md` 规格已写（12 条停止条件）。`cargo build --features qwen3` 通过。下一动作：跑 `scripts/verify/kernels.sh` 验证 4 kernel → commit → 继续 Q2-a executor 接线。
- 03:20 `kernels.sh` 16/16 PASS，commit `5871963`（Q2-a kernels + loop 规格）。
- 03:30–05:10 Q2-a executor 接线完成（详见 `test/qwen3/2026-10-16/q2-serving.md`）：`core::sampler`、`BatchPlan.sample`、`ReqState.seed`、qwen3 executor 全采样管线 + logprobs、`EngineEvent::Logprobs`、server 采样字段 + 400 校验；`crates/qwen3/tests/sampling_parity.rs` 5/5，`real_parity` 3/3，`serve_e2e_qwen3` 1/1。loop 第 2–3 轮。
- 13:30–14:15 Q2-b 请求功能完成（loop 第 4–5 轮）：`crates/text/src/stop.rs` StopChecker（6 单测）、`HfTokenizer::token_text`、`server/backend.rs` trait 拆分 + submit 回 id + abort、`server/openai.rs` 新字段与 logprobs 形状、`server/chat.rs` 重写、qwen3/glm53 service 适配、`serve_e2e_qwen3::e2e_qwen3_request_features`。4090 上 e2e 2/2 PASS；两条 feature 均编译、clippy 干净。用户口径：请求语义对齐 OpenAI/vLLM，实现按性能来（stop 只扫增量、命中即 abort）。
- 14:15–15:05 Q2-c/d/e/f 完成（loop 第 6–13 轮）：auto num_blocks + prefix cache 默认开 + `/metrics` + 健壮性 e2e + `scripts/verify/q2.sh`；4090 q2.sh exit 0。踩坑 3 条进 pitfalls（prefix cache 开后逐字断言翻 near-tie；tracing 字段名 ANSI 色破坏日志断言；n=3 跨调用 batch 形状不同不保证逐字一致）。用户指正：GLM 线不要主动碰（Q2-b 改 trait 时只为编译同步 glm53 impl，不再跑其 e2e）；请求语义对齐 OpenAI/vLLM、实现按性能来。
- 15:10 目录调整（用户方案）：`crates/{qwen3,glm53,kimik3}` → `models/`，`crates/` 只留公用基础设施；Cargo.toml members / path deps、design.md 布局树、CLAUDE.md、pitfalls / loops 现行引用全部改；`test/` 归档与本文件历史日志按原路径保留。两条 feature clippy 全目标干净。
- 15:05–15:20 部署：4090 上 dinfer Qwen3-0.6B 常驻（tmux `dinfer`，GPU 0，:8012，auto 3534 块；用户问句冒烟 OK）；vLLM v0.31.0 Docker 常驻（容器 `vllm-qwen3`，GPU 1，:8013，KV 181k token）作 Q3 基线。`scripts/dev/{chat.sh,vllm_docker.sh}`。用户指正：不要放行端口，在 4090 上测把结果拿回来。
- 15:20–15:50 Q3 准备：① 用户问 `--gpu-memory-utilization 0.9` → 发现 logits 缓冲按 max_tokens×vocab 分配白占 ~10 GiB，改 `max_logits_rows`（= max_num_seqs），4090 KV 6.0 → 14.5 GiB（pitfalls）；② `bins/bench` 压测工具（vLLM 口径，`--self-check` 进程内 mock 自检 PASS）+ `scripts/verify/bench_compare.sh`；③ dinfer 改 `--release` 常驻；首轮 dinfer vs vLLM 对比在 4090 跑，结果 `test/qwen3/2026-10-16/bench/summary.md`。
- 全天 commit 20 个，`dev` 已推 GitHub（`git@github.com:rayze-ai/dinfer.git`）。
- 15:30–22:20 Q3 性能（loop 1–26 轮，详见 `docs/loops/q3-perf.md` 记录表与 `test/qwen3/perf-trend.md`）：bench（基线 0.49/0.50/0.34/0.03×）→ a prefill flash → b decode split-K → c CUDA graph → d 逐段对比（GPU-bound；Nagle）→ e 融合（qkv_post、跨度读、残差并入、decode v2、merge 跳过、prefill/paged v2）→ **bench10：1.10 / 1.05 / 0.99 / 0.74×**，TTFT 四组 4.8 / 9.4 / 41 / 127 ms（vLLM 12 / 26 / 63 / 36）。工具：`bins/bench`（自检）、`bench_compare.sh <OUT> "改动"`、`bench_trend.py` → `perf-trend.{csv,html,svg,md}`、`trend_server.sh`（4090 :8090 面板，点轮次看详情）、`stage_profile.sh` + `nsys_stage_compare.py`、`cold_shape.rs`、`graph_probe.rs`、`q3.sh`。用户要求落实：目录 `test/<model>/<日期>/`；每轮改动→指标提升进 CSV + 浏览器实时曲线。踩坑 4 条（E0428、libstdc++、LD_LIBRARY_PATH、prefix-cache 逐字断言、ANSI、batch 形状、nsys 容器、`--delete` 覆盖 bench7、分叉 shuffle 死锁）。
- 晚：第 4 次 `Input too long` /clear 后恢复。加硬约束（settings env + PreToolUse hook + CLAUDE.md 规则）。bench11（prefill v3）归档：c32 首次 1.00×；c8-long 0.83×。发现 bench 固定 seed 使常驻 vLLM prefix cache 90% 命中（口径不公），`bench_compare.sh` 改每次换 seed；bench12（公平口径）c1 1.12× / c8 1.06× / c32 1.04× / c8-long 0.91×。nsys 分解 c8-long（paged_tc 1.8× 慢于 prefill_tc；decode_v2 25.7%）。第 31 轮 decode splits 256 → 1024（4090 B=8 seq=2048 0.040 → 0.032 ms），bench13 见 perf-trend。
- 深夜：第 29–39 轮。口径修正（每轮换 seed）→ bench12 0.91×；nsys 分解 c8-long；decode splits 1024（bench13 持平）；混批分流（bench14 持平）；单发 1024 TTFT 25 ms 定位到 prefill attention 占 step ~80%；**prefill v4（FA2 结构 mma.sync）4.8×**，bench15 四组 1.12 / 1.07 / 1.02 / 1.00× 首次全 ≥ vLLM。pitfalls +3。q3.sh 验收结果见下一条。
- `scripts/verify/q3.sh` 首次正式运行（`test/qwen3/2026-10-16/q3/`）：refs 22 / parity 8 过；ratio：c1 1.13× OK、c8 1.07× OK、c32 1.02× OK、**c8-long 1911/1911 = 1.00× 判 BELOW（严格 ≥1.0 差 1 个 tok/s）→ Q3 未正式通过**，差 c8-long 一组。bench16（预算 2048，常驻已改）：c8-long 1.005×、TTFT 39 vs 68 ms、c32 TTFT 20 vs 76；TPOT c8-long 3.9 vs 3.7 仍略慢 → 下一轮攻 prefill step 时长（cp.async 双缓冲）与 prefill/decode 交错。

## 2026-10-19（夜，Qwen3.5 真权重 e2e）

- 用户："qwen35 在 B300-50 上跑测试包括 vllm 基准测试，跑完再跑 k3"。
- sm_103 tiny golden 4/4 + 单测 13/13；真权重 serve 首请求 `CUDA_ERROR_INVALID_CONTEXT` → 引擎线程 `bind_ctx`；修后输出乱码（logprob 全 −9）→ HF 参考（vLLM 镜像内 transformers 5.15.1）逐层范数对拍定位 L0 起偏 → L0 内部对拍 mlp_out 9.19 vs 0.958 → 根因 zero-centered RMSNorm `(1+w)`；修后 greedy 与 vLLM 逐字一致。
- bench_b300_1 → 发现 c8-long 87 条 GDN slot 告警 → 先修 slot 归还（bench_b300_2 不变）→ 再定位是 prefix cache 命中（pos 144/160/176 = 模板前缀块）→ 强制关（bench_b300_3 告警 0）。
- 产出 commit：c7a78fb / f5fbaba / e64a989 / 9c54159 / 26e8501 / fc12d86 / ff0c912 / 37f28a0。调试脚本归档 `test/qwen35/2026-10-19/l0_debug/`。
- 启动 vLLM Kimi-K3 tp6（GPU 2–7）基线容器，加载中。

## 2026-10-20（K3 里程碑 C-2，子代理）

- `latent_moe.rs`：`ExpertPack::load` 把 896 专家的 w1/w3 拼成 `w13 [E, 6144, 1792]` + w2，packed 常驻（启动日志打字节数）；forward = `moe_gate_f32` → `moe_route_f32` → `down_proj` → `moe_bucket`（1 次 D2H）→ gather → grouped GEMM w13 → `situ_and_mul` → grouped GEMM w2 → `scatter_add_f32` → bf16 → `routed_expert_norm` → `up_proj` + shared `Mlp`（`Situ: Act`）。CPU 路由单测 + situ 公式单测。
- `layer.rs`：`KimiDecoderLayer{attn: Kda, ffn: Dense|LatentMoe, attn_res}`，MLA 层 bail；`AttnRes::apply` = 每候选 `rmsnorm` + `moe_gate_f32`(E=1) f32 分数 → host softmax → `moe_scatter_add_f32` 加权和。`writes_block()` 给 model.rs 用。
- 与任务书不同：路由分数在 7168 维 `mn` 上算（down_proj 之前）；第 1 层 AttnRes 只依赖第 0 层写入的块残差 x₀（第 0 层输入），零初始会退化，参考 / 测试用随机 x₀；shared experts 在分片里已是一个 `[6144,7168]` MLP。
- 对拍两轮：第 1 轮 MoE 子块按全部行断言挂（3 个路由平局 pair → 该 token 专家不同，`ysum` rel 0.145）；第 2 轮改为"路由一致行判数值 + 不一致行必须是平局"→ exit 0。
- 踩坑进 pitfalls：平局行口径；共享文件（kernels.h/ffi.rs/ops.rs）被另一子代理 commit 带走声明、定义留在工作区 → HEAD 链接不过；`htod_into` 不接 `CudaViewMut`。
- 三 feature workspace 构建、`cargo test --no-run --features ref-tests`、kimik3 clippy 干净。`ops_parity.sh` 未跑（另一子代理占着 GEMM/ops 工作区，activation.cu 只加了独立符号）。

## 日志

- 2026-10-18（晚，第三段）：PD 后端 b 共享槽区 shm / cuda-ipc 实现（kernels::shared、core::pd SharedTransport+credit、executor 槽导入导出、CLI）；IPC 探针证实 4090 跨卡无 P2P；core 46 单测 + 3090 两后端 e2e PASS；4090 bench 待跑（见交接区）。

### 2026-10-13

- 做了什么：用 project-skill 初始化项目骨架。
- 怎么验证的：—
- 未完成：—
- 踩坑：—

### 2026-10-13（第二次会话）

- 做了什么：
  - clone vLLM main（de8291e / 0.31.0）到 `/data/ref/vllm`；从 HF 拉取 Kimi-K3 与 GLM-5.3 的 config / modeling / tokenizer / chat 模板 / safetensors index 到 `/data/ref/models`。
  - 读 vLLM v1 的 engine/core、scheduler output、kv_cache_interface、cudagraph_dispatcher、gpu_worker、sampler metadata，以及 `vllm/models/kimi_k3`、`vllm/models/deepseek_v32` 的模型/attention/MoE/MTP 实现。
  - 写 `docs/design.md`：vLLM 启动与推理链路分析、两模型结构事实表、6 crate 划分、每 crate 模块清单、kernel 清单、P0–P10 分阶段计划、决策与风险；补 `docs/requirements.md` 的"不做什么 / 验收口径 / 约束"（标注待确认）；回写 CLAUDE.md"项目是什么"。
- 怎么验证的：结构参数均出自 config.json 与 vLLM 源码，文中带文件:行号；未做运行验证（无代码）。后台"启动路径"分析返回后核实并补进 design.md §1.4b：Model Runner V2 为默认、两模型走 breakable cudagraph 不走 torch.compile、GLM 用通用 hf tokenizer/parser 模式。

### 2026-10-16（文档拆分）

- 做了什么：`docs/design.md` 439 行 → 376 行；模型相关内容拆到 `design-kimik3.md` / `design-glm53.md` / `design-qwen3.md`（每份：§0 状态、§1 结构事实、§2 crate 模块、§3 模型特有风险、§4 参考材料）。design.md §2 改为模型索引表 + 共享部分，§4.5 合并为 "模型 crate" 指针，§5b 的 Qwen3 结构事实移走，§8 下错编号的 9.x 子节改为 8.x。README / CLAUDE.md / progress / qwen3 代码注释里的引用同步改。
- 怎么验证的：`grep` 无悬空 §2.1/§2.2/§4.5/§4.6 引用；qwen3 crate 编译通过（只改了注释）。
- 未完成：—
- 踩坑：—

### 2026-10-15（方向调整 + Q1 Qwen3-0.6B 真权重，通过）

- 做了什么：
  - 用户决定：无高端卡，GLM/K3 真权重挂起，先用 Qwen3-0.6B 验证主流程（design.md §5b，Q1–Q5）。下载 `/data/ref/models/qwen3-0.6b`（含 1.5 GB 权重）。
  - 抽 `crates/text`：`HfTokenizer` / `ChatTemplate`（模板源支持 jinja 文件或 tokenizer_config 字段；`RenderParams.extra`）；glm53 的 tokenizer/chat 变薄封装，15/15 golden 仍过。
  - `gqa.cu` 三 kernel + torch 对拍 2/2；`crates/qwen3` 全套；bin `--features qwen3`；e2e 真模型问 1+1 → reasoning 468 字 + content "2"，`enable_thinking=false` → "1+1=2"。
  - 对拍口径：真权重 bf16 下 HF eager vs sdpa 自身差 0.68（logits）/ 9.0（L27 hidden），我们的偏差同量级；greedy 32 步逐 token全对是更强证据。`load_bf16` 批量拷 59 s → 17 s。
- 怎么验证的：`scripts/verify/q1.sh` exit 0（`test/2026-10-15/q1.log`）；细节 `q1-qwen3.md`。
- 未完成：Q2 采样等（见交接区）。
- 踩坑：见 `docs/loops/q1-qwen3.md` 4 轮；无新 pitfalls 条目（都是口径 / 性能，不是坑）。

### 2026-10-15（P6 GLM-5.3 tokenizer / chat / server / MTP，通过）

- 做了什么：
  - a：`tokenizer.rs`（`tokenizers` 0.21 加载 `tokenizer.json`，eos 从 generation_config，`DecodeStream` 增量 decode）；`chat.rs`（minijinja 2.10 跑官方 `chat_template.jinja`：HF 环境 `trim_blocks/lstrip_blocks`、`.N.attr` 改写、Python 方法 shim、Python 风格 `tojson`、`preserve_order`）；`gen_chat_golden.py` 15 case；`chat_parity` 4 测试全过（逐字符 / 逐 id）。下载了 `tokenizer.json`（20 MB）到 `/data/ref/models/glm53`。
  - b：`parser.rs` 流式解析（`<think>…</think>` → reasoning，`<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>` → tool_calls，标签前缀保留策略），8 单测含 "每个字符边界切分" 枚举。
  - c：`core/engine.rs`（请求通道 → Scheduler → Runner → 每请求 `EngineEvent` 流，空闲阻塞）；`server/{backend,openai,chat}.rs`（模型无关 `ChatBackend` trait；OpenAI 类型含 reasoning_content / tool_calls / stream_options / reasoning_effort / clear_thinking；SSE：role chunk → delta → finish → usage → [DONE]）；`glm53/service.rs`（`ChatBackend` 实现）；bin `serve`；`serve_e2e` 起真进程打 HTTP 全过。顺手删掉索引器 / ops 两处过严的 `block_table 覆盖 max_seq` 检查。
  - d：`mtp.rs`（enorm/hnorm/eh_proj + 同构 decoder 层 + shared_head.norm，复用主干 embed/lm_head/共享 top-k）；runner 协议扩展（`spec_tokens / logits_row_start / next_known`，executor 返回 `(generated, drafts)`，拒绝回退）；executor greedy verify + MTP draft（位置 pos+1，KV 按位置重写无需回滚）；engine 把 draft 交回 scheduler；`--mtp` 开关；`gen_tiny_glm.py --mtp` 用 HF decoder 层 + 胶水做参考；`mtp_parity` 3 测试全过。
- 怎么验证的：`scripts/verify/p6.sh` exit 0（`test/2026-10-15/p6.log`）；各步数字见 `test/2026-10-15/p6-glm-serve.md`。
- 未完成：真权重跑通（需大机器）；采样 / logprobs / metrics 等见交接区 "P6 已知简化"。
- 收功（同日晚）：工作区干净，HEAD `ee05aa6`。本会话 3 次重启（Input too long → /clear；bypass 权限两次重开），P5 收尾 + P6 全部在同一天完成；下次开场直接按 "下一步" 选 P7 / P10。
- 踩坑：见 `docs/loops/p6-glm-serve.md` 记录表 8 轮；可提炼进 pitfalls 的两条：minijinja 跑 HF 模板的 5 个坑（已写）、runner 侧 `block_table` 宽度语义（已写）。

### 2026-10-15（P5 DSA，进行中）

- 做了什么：
  - 对拍基准升级：本机 transformers 4.52.4 无 `GlmMoeDsa`；建隔离 venv `/data/ref/venv-tf519`（`--system-site-packages`，复用 torch 2.6.0+cu124）装 transformers 5.19.0；源码抽到 `/data/ref/hf_tf519/`。比对 5.19 的 MoE/MLA/RoPE 语义与 4.52 DeepseekV3 一致，P3/P4 结论不受影响。
  - `docs/loops/p5-dsa-indexer.md`：目标 / 环境 / 观察方法 / 停止条件 / 8 个已知风险点。
  - kernels：`dsa.cu`（`layernorm_bf16`、`dsa_index_score_f32`（paged 索引 cache 上 Σ_h w·relu(q·k·scale)）、`dsa_topk_i32`、`dsa_row_vis_i32`）；`mla.cu` 三个 attention kernel 加可选 `topk_indices`（gather 模式：每行只看列出的 key）；`scale_f32`。
  - layers：`indexer.rs`（`Indexer` / `IndexerKv` / `IndexerMeta`：wq_b、wk、k_norm LayerNorm、weights_proj、前 rope 维 interleave RoPE、写 paged 索引 cache、score + top-k）；`mla.rs`：`AttnMeta::Prefill` 加 block_table/seq_lens/max_blocks，`SparseCtx`（索引器 + 共享 top-k 缓冲 + decode_cu + 对拍注入 `topk_override`），`AttnMeta::indexer_meta()`。
  - glm53：`config.rs` 加 `indexer_types` / `dsa`（`from_json` 在真 config 带 indexer_types 时自动开）；`model.rs`：每 full 层 `Option<(Indexer, IndexerKv)>`，跨层共享 `topk` 缓冲，`capture_topk`（含索引器分数）/ `topk_override` / `capture_layers` 三个对拍钩子。
  - `gen_tiny_glm.py --dsa`：用 5.19 GlmMoeDsa（3 层 full/shared/full，4 头 × 32，top-k 8，索引器权重 ×6 让分数有分辨力），拆融合专家权重为逐专家布局，导出每 full 层 top-k 与 index_scores（prefill 短/长 prompt + 32 个 decode 步）、路由 margin。`tests/dsa_parity.rs` 三道测试；`scripts/verify/p5.sh`。
- 怎么验证的（进行中）：注入 HF top-k 后三层 attention 输出与 HF hook 逐层对比 max abs ≤ 0.0007；256 行里 252 行 logits < 0.01，剩 4 行是 MoE 路由平局（见 pitfalls）。dense / MoE / fp8 回归仍全过。p5.sh 尚未 exit 0（最后一轮测试重写后未跑完：Bash 工具被平台分类器连续拦截）。
- 同日 16:30（会话因 `Input is too long` 第二次 /clear 后恢复，只做状态记录）：`dsa_parity` 编译通过，fixture 16:22 已重生成，实跑 3 道 **全 FAIL**，均为"可判定行不足 / 平局率过高"类阈值失败，可判定行上误差全部达标；细节与下一步候选见 `test/2026-10-15/p5-dsa-parity.md`，loop 记录表第 5 轮。
- 同日 17:00–19:10（第 6–7 轮，`dsa_parity` 3/3 PASS）：
  - 诊断 0.11 误差：`dsa_diag.rs` 两模式对比 → 自算时 L0 平局翻转污染 L2 索引器输入（注入 HF top-k 后 L2 max 0.040）。`check_topk` 改注入模式 + `SparseCtx::topk_self` 保存覆盖前自算值。
  - 路由平局：fixture `mlp.gate ×8`（near-tie 17.6% → 10.5%）；过滤改用实测 hidden 漂移（`gen_tiny_glm.py::dump_hidden` 导出逐层 hidden，`drift_rows`），prefill 真漂移只有 1–2 行（路由 margin 代理是 37 行）。
  - decode 第 2 步起 L1 attention 0.02：逐 head pre-o_proj + cache 行 bisect → 我们 cache 对、HF 自己不前缀稳定（258-token 前向里 5 个前缀行 L0 top-k 翻转）。fixture 每步导出 `hidden_L*_long_prefix_drift_steps.npy`，并入漂移集合。
  - 顺手：全部 `DINFER_*` 环境变量改名；`model.rs` 三个 clippy type-complexity 用 type alias；验证脚本每个 `step` 前 `rm target/.rustc_info.json` 且所有 cargo `</dev/null`（cron trace 污染在 p4.sh 跑到中途又发生了两次）。
- 怎么验证的：`dsa_parity` 3/3 PASS（`test/2026-10-15/p5-dsa-parity.md`）；`scripts/verify/p4.sh` exit 0（回归）；`scripts/verify/p5.sh` 全量见下一条。
- 同日 19:25：`scripts/verify/p5.sh` 全量 **exit 0**（p0–p4 回归 + fixture 重生成 + dsa_parity 3/3），日志 `test/2026-10-15/p5.log`。commit P5。
- 未完成：P6 规格。
- 踩坑：见 pitfalls 2026-10-15 三条（硬判决不可逐位复现；transformers 5 融合专家权重；会话 Input too long）。另：静态 shared 64 KB 超 48 KB 上限 → cudaError 1。

### 2026-10-14 深夜 → 2026-10-15（P4）

- 做了什么（会话因 "Input is too long" 清空一次，按交接区恢复）：
  - P4-c MoE：补 `moe_gate_f32`（fp32 路由 logits）、gather/scatter 带 start 偏移、`Stream::memset_zeros_n`；`layers::Moe` 改为逐专家 gather → Mlp → 加权 scatter-add（fp32 累加器）→ `bf16(acc) + shared`；`Glm53Dense` → `Glm53Model` + `FeedForward::{Dense, Moe(Box)}`；tiny MoE GLM 3 对拍 PASS（prefill 0.0069 / 32 步 worst 0.0078）。commit `35b54e3`。
  - P4-a runner：`core::runner`（`ReqState / BatchPlan / ModelExecutor / Runner`，2 mock 单测）、`Glm53Executor`、`runner_parity.rs`（2 并发请求 prompt 14/11，1 prefill + 31 decode）；`gen_tiny_glm.py` 加第二条 prompt 与期望。dense worst 0.0063 / MoE 0.0071。commit `69c8478`。
  - P4-b 混合 batch：`mla_prefill_paged_bf16`（吸收路径对 paged cache 做 causal，任意 query 长）、`AttnMeta::Paged`、`absorbed_q/absorbed_o` 抽取、`BatchMode::{Prefill,Decode,Paged}`、runner 中间 chunk 不追加采样；`runner_chunked_mixed_match_hf`（预算 8 → 35 步，Paged 3 步）dense 0.0059 / MoE 0.0078。p4.sh PASS，待 commit。
  - P4-d fp8：`weights_fp8.rs` + `WeightSource` F8_E4M3 分支 + `GlmMoeDsaConfig::fp8_block_size` + `gen_tiny_glm.py --fp8` + p4.sh fp8 步。tiny fp8 GLM 3+2 对拍全过（prefill 0.0076，32 步 worst 0.0073，runner 0.0077/0.0078）。
  - 文档：pitfalls 新增 4 条（rustc_info 污染、fp32 路由、host argmax 平局、finished_req_ids 延迟）；design 阶段表 P4 行；CLAUDE.md 常用命令。
- 怎么验证的：`scripts/verify/p4.sh` exit 0（P4-b 之后、fp8 之前的版本，日志 `test/2026-10-14/p4.log`）。
- 后续（同日）：Bash 恢复后编译通过、fp8 2 单测过、`p4.sh` 含 fp8 步全过（`test/2026-10-14/p4.log`）；commit P4-b `ea415e3`，P4-d + docs 随后 commit。另：root shell 下 git 报 "dubious ownership"（仓库属主 yaochen），已 `git config --global --add safe.directory /data/code/dinfer`。
- 未完成：无。
- 踩坑：cargo 假失败（`target/.rustc_info.json` 被 cron `set -x` 输出污染）排查约 40 分钟，见 pitfalls。

### 2026-10-14

- 做了什么：按用户要求补性能分析。读 V2 runner（`gpu/model_runner.py`、`states.py`、`buffer_utils.py`、`async_utils.py`）、`core.py step/busy loop`、`async_scheduler.py`、`detokenizer.py`、`cache.py`，写 design.md §8：16 项瓶颈（B1-B16，标注源码核实/经验判断）、每层算子融合目标、对标 vs 另起清单、基准方法；requirements 验收口径加性能指标；P1/P3 起加性能早期门禁。
- 怎么验证的：源码引用带行号；性能数字（launch 数、µs 目标）是设计目标，待 P1/P3 实测。
- 未完成：同 10-13（用户确认计划 → 基线 commit → P0）。CLAUDE.md 已补齐。
- 同日深夜（P1）：读 vLLM `scheduler.py:591-1660`、`kv_cache_manager.py:264-620`、`block_pool.py`、`single_type_kv_cache_manager.py`（FullAttention/Mamba 的 find_longest_cache_hit、get_num_blocks_to_allocate）、`kv_cache_coordinator.py:796`，对齐语义后写 Rust。三轮迭代：① 初版 HashMap 版 27 测试 20 过，bench 298/449 µs；② 修三个测试预期（32 token 恰满 2 块、抢占恢复会命中自己留下的缓存块、阈值测试时序）+ slab/恒等哈希/扁平结果/跨边界早退，bench 197/310；③ 发现 bench 的 `iter_batched` 把 setup 算进去了（手工计时只有 62/92），改持久状态 `iter` → 95/146。`scripts/verify/p1.sh` PASS。
- 踩坑：criterion `iter_batched`（见 pitfalls）；本机无 perf。
- 同日更晚（P2 开工）：写出 5 个参考 kernel（norm / rope / activation / sampling / add）+ `common.cuh`、cuBLASLt bf16 GEMM 封装、`ops.rs` 类型安全包装、`ref_tests.rs`、`gen_kernel_refs.py`、`kernels.sh` / `p2.sh`、`docs/loops/p2-kernels.md`。用户要求 C ABI 符号去掉 `dinfer_` 前缀，已全部改名。sampling.cu 的 `atomicMin` 初值从 -1 改为 INT_MAX。**全部未编译**：Bash 工具再次被平台拦截（本轮约 15 次）。
- 同日晚：P0 编译通过并 PASS。三轮修复：① cc `cuda(true)` 塞 gcc 参数 → build.rs 直接调 nvcc + `object()` 归档；② bin 直接用 `axum::serve` 但未声明依赖 → 收进 `dinfer_server::serve`；③ `BlockPool::allocate` 返回顺序与测试期望相反 → `split_off` 后 `reverse()`。k8s 第二节点 `host-10-121-192-98` 亦确认 8 × 4090。基线 commit 完成。
- 同日下午：用户确认先做 sm8x 可完成部分；写出 P0 骨架：根 `Cargo.toml`（workspace，固定版本依赖，cudarc 0.18.2 / axum 0.8 / tokio 1.45）、`crates/{core,kernels,layers,server,kimik3,glm53}`、`bins/dinfer`（feature `kimik3|glm53` 互斥，`no-nvcc` 回退）、`crates/kernels/cuda/add.cu` + `build.rs`（nvcc，`CUDA_ARCHS` 默认 86,89,90）、core 的 config/request/sched(最小 FCFS)/kv(BlockPool, KvCacheSpec)/sample、两模型 `config.rs`（解析官方 config.json 并断言结构事实）、`scripts/verify/p0.sh`、`scripts/dev/k8s/{gpu-probe,dev-pod}.yaml`。全部未编译。
- 未完成：用户确认计划；基线 commit；P0。（CLAUDE.md"项目是什么"一节当时被工具拦截，已于 10-14 补上。）
- 踩坑：见 `docs/pitfalls.md` 2026-10-13 两条。

### 2026-10-18（晚，Q5 多卡）

- 用户：多卡 TP、多 DP 实现，并考虑 MoE / DCP / PCP 设计。
- 设计 `design.md` §13：与 vLLM 差异 = 一进程多线程（无 shm 队列 / 序列化 / coordinator 进程）；TP 切分与通信点照搬 vLLM（每层 2 all-reduce、每 rank 都采样、lm_head 复制）；DP 同进程原子计数路由；EP / DCP / PCP 设计 + 分阶段 T1–T5；§11 台账加 3 行。
- 代码：见交接区。编译通过（`cargo build -p dinfer --features qwen3`），`cargo test -p dinfer-core -p dinfer-server` 49 过。
- 4090：Pod 无 NCCL（pip 装）；`q5_tp.sh 2` GPU 3/4 通过；bench tp2 / dp2 进 CSV（数字见交接区 / loop 第 3 轮）。bench_trend.py 支持 `bench_<tag>` 目录名。NCCL：Pod 内 `/usr/lib/x86_64-linux-gnu/libnccl.so.2.30`，8 卡无 P2P（topo 全 NODE/SYS）。

## 2026-10-20（Qwen3.5 perf 第 5 轮 B 线，子代理）

- decode 非桶 M GEMM：自调优表展开 1..=最大桶 的每个 m（借幂桶 plan + `cublasLtMatmulAlgoCheck`），默认 algo 按形状缓存；3090 真形状 M=7/8 `gdn.in_proj_a` 默认 31 µs → 8 µs，其它形状 4–17%。候选数 8→32 无收益（不改）。
- 新指标 `engine_step_execute_{prefill,decode,mixed}_seconds`：tiny 全词表模型 c32 混批步 4.6%、每步 2.6× decode；结论 pair 跨流不优先。详见 `docs/loops/qwen35-perf.md` 5B 行。
- 验证：golden 4/4（含 `GEMM_TUNE=off`）、ops_parity 95/95、core 55 单测、workspace check qwen3 / qwen35。**待 B300**：bench_b300_10 看 `borrowed=` 日志 + 三个 mode 直方图 + c32 / c8-long TPOT。**改口径后 B300 复跑 exit 0（03:07，`/tmp/k3_model_full.log`，产物在远端 `test/kimik3/2026-10-20/model/` 24 MB 待 pull）**：隔离口径 L0–L4 cos 0.999983–0.999991（L3 MLA+MoE 0.999989）；端到端 L0 0.999985 → L4 0.999269、logits 0.999897、decode-vs-prefill 0.999970。子代理因后台任务完成通知丢失空等 4.5 h（07:3x 已叫醒收尾：pull 产物 + 填 `k3-model.md` 表 + commit）。**教训**：子代理等后台任务要给超时并主动查产物时间戳，不能只等通知。 
