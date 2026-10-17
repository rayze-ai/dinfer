# 进度

<!-- ===== 当前状态（交接区）：每次收功整体重写，保持 ≤ 20 行 ===== -->
## 当前状态

- 更新日期：2026-10-17
- 阶段：Q1 / Q2 通过；**Q3 通过**（2026-10-17 `scripts/verify/q3.sh` exit 0：c1 1.12× / c8 1.07× / c32 1.02× / c8-long 1.00× vLLM，`test/qwen3/2026-10-17/q3/`）。规格 `docs/loops/q3-perf.md`（44 轮），趋势 `test/qwen3/perf-trend.{csv,md,svg,html}`（bench1–16 每轮带改动说明）。GLM 线 P0–P6 通过，真权重挂起。
- Q3 关键产出：prefill/paged attention v4/v5（`gqa.cu::gqa_fa_v5_kernel<PAGED,NBUF>`，FA2 结构 mma.sync，4090 T=1514 0.14 ms/层；双缓冲在 sm86/89 更慢，默认 NBUF=1，`GQA_PREFILL_NBUF=2` 开关）；decode v2 split-K + splits 1024；混批分流 `gqa_mixed_bf16`；CUDA graph；qkv_post 融合；bench 口径（每轮换 seed）；服务预算 2048（与 vLLM 对齐）。设计说明 `docs/design-qwen3.md` §5（prefill_tc vs paged_tc 是什么、为何选 paged）。
- **Q3-j 跨流 prefill/decode 并行：实现完成，默认开，闭环 bench 下无可见收益**（`docs/loops/q3-xstream.md` 9 轮）。executor 双套资源 + `forward_sample_pair`（事件同步）、runner 混批拆分/散回（`try_execute_pair`）、scheduler `decode_exempt_from_budget`（默认开）。parity / e2e / core 34 单测全过；bench17/19 四组与 bench16 持平（c8-long 1.00×）。原因：闭环固定并发下 prefill 与 decode 天然错开（4000 步只 18 步混批），收益上限 ~0.15 ms TPOT（pitfalls）。回退 `XSTREAM=0` / `--no-decode-exempt`。**启动 loop 结束**（`docs/loops/q3-startup.md` 10 轮）：4090 **2.93 → 0.61–0.68 s**（cuda_init 250 不可压 / open 110 / layers 115 / h2d 180 / kv 60 / engine 45；懒捕获图默认，`GRAPH_CAPTURE=eager` 恢复）。**开环泊松 bench 已加**（`bench --request-rate R --prompt-len-mix ...`、`scripts/verify/bench_poisson.sh`，结果 `test/qwen3/2026-10-17/poisson1/`）：TTFT 全面领先 vLLM（15/9/10 vs 23/17/19 ms），TPOT 在 8/16 req/s 持平、**24 req/s 落后 7%（4.9 vs 4.6）**；跨流开/关 TPOT 完全一致 → 要么两条流被事件串行化没真重叠，要么收益确实为零。**跨流已用 nsys 证实真并行**（副流 39% 时间与主流重叠），但开环 24 req/s 下混批步仅 5%，TPOT 无感；默认保留。**decode B=32 差距定位完成（q3-perf 第 48–57 轮）**：① block_table 缓冲用 KV 池总块数做行跨度 → 每步 1.3 MB 零填充 + H2D（已改 max_model_len/block_size，launch 610 → 250 µs）；② **:8012 常驻是 20 h 前的孤儿进程**（tmux kill-session 杀不到 tee 管道里的 serve），bench16–22 的 GPU 0 数字部分打在旧二进制上，已写 `scripts/dev/restart_dinfer.sh` 并重测。**bench23（新基线，干净重启）：c1 1.12× / c8 1.06× / c32 1.05× / c8-long 1.01×，四组全部 ≥ vLLM，TTFT 全部领先。** 每步构成：GPU 忙 2.6 ms（vLLM 离线整步 3.40）+ host ~0.3 ms（upload 50 + launch 250）。PinnedUploader（`PINNED_UPLOAD=1`）与 timing 事件（`GPU_TIMING=1`）A/B 均零影响，默认关。**异步调度完成（`docs/loops/q3-async.md` 6 轮，默认开，`ASYNC_SCHED=0` 关）**：runner `begin/finish`、engine 双步流水、executor 图内 `row_gather_i32` + pinned 双槽异步 D2H。**bench24：c1 1.13× / c8 1.08× / c32 1.08× / c8-long 1.01×**（c32 TPOT 3.56 → 3.46；代价 TTFT 多等一步，c8-long 39 → 54 ms，与 vLLM 持平）。**采样融合完成（`docs/loops/q3-sample-fuse.md`）**：bench25 **c1 1.14× / c8 1.08× / c32 1.08× / c8-long 1.02×**，四组全部 ≥ vLLM。**小 kernel 融合（Q3-n）收口**：gather_rows + 末层 rmsnorm → `gather_rmsnorm_bf16`（bench26 持平，预期内）；剩余两项（lm_head f32 直出、silu·mul GEMM epilogue）都要自写 GEMM，并入 megakernel 阶段。**Q3 单卡性能线至此：bench26 c1 1.14× / c8 1.07× / c32 1.07× / c8-long 1.01×，TTFT 全面领先，启动 0.6 s。****Q3-o 插队 prefill 完成（`docs/loops/q3-cutin.md` 6 轮，默认开；`CUT_IN=0` 关、`CUT_IN_MAX_ROWS` 门槛默认 8）**：bench28 **c1 1.14× / c8 1.09× / c32 1.07× / c8-long 1.00×**，TTFT c8 8 ms（vLLM 35）、c8-long 35 ms（vLLM 68）。两处坑进 pitfalls（scheduler 隐含假设；结果晚发丢首 token）。**Q3 单卡性能线收口成绩：四组吞吐全 ≥ vLLM，TTFT 领先 2–4×，启动 0.6 s。****下一步**：① `design.md` §10 算子插拔 / KV 分层细化（用户要求的扩展点）；② KV 28 层合一次 cuMemAlloc（-45 ms）；③ 逐层 megakernel（design-qwen3 §6.1，含 lm_head f32 直出 + silu epilogue，预期 decode 1.4×）；④ Q4 PD 分离；⑤ 可生产补齐（鉴权 / 限流 / 优雅停机 / >8k 验证）。用户要求：测试尽可能在 4090；代码可读性；每轮进 CSV；采用/改进记 design §11。
- **下一步按 `design-qwen3.md` §6.5 优先级**：① 跨流 prefill/decode 并行（c8-long TPOT 3.9 → ~3.0，Q4 单卡前置）；② 启动 3.0 s → <0.8 s（瓶颈是权重 mmap→Vec 复制 1.9 s，不是 PCIe；tokenizer 545 ms）；③ 采样 4 kernel 融合 + 异步调度；④ 逐层 megakernel（decode 1.4×，独立阶段）。之后 Q4 PD 分离 + "可生产"补齐（鉴权 / 限流 / 优雅停机 / >8k 上下文 / 多卡）。
- 启动日志已分段（用户要求）：`load timing: weights|activation scratch|kv cache alloc` + `cuda graphs captured` + `startup timing`；运行期 `executor step breakdown`（upload/launch/wait µs）每 1000 步、`/metrics` 的 engine_step_{schedule,execute,update} 直方图。
- **上下文硬约束**：`.claude/settings.json` `BASH_MAX_OUTPUT_LENGTH=8000` + `.claude/hooks/context_guard.sh`。**每轮改动必须 `bench_compare.sh <OUT> "改动一句话"` 进 CSV 趋势**。本会话 130+ 次调用后收功。
- 4090 常驻：dinfer tmux `dinfer` GPU 0 :8012（release，`--max-num-batched-tokens 2048`，decode 免预算 + 跨流默认开，commit 148b758+）；vLLM Docker `vllm-qwen3` GPU 1 :8013；测试 GPU 3；趋势面板 :8090。远端无 torch → refs/golden 本机生成后 `push`；`push` 带 `--delete`，远端 bench 先 `pull`。
- 验证入口：`scripts/verify/remote4090_baseline.sh`（refs 22 + parity 8 + 单测 8 + e2e 6）；`scripts/verify/q3.sh`；本机 `scripts/verify/kernels.sh`；`cold_shape`（prefill / paged 四形状 / decode 微基准）。
- 环境：本机 3090 nvcc 12.4；**cargo 假失败 `rm target/.rustc_info.json`**；远程 `git@github.com:rayze-ai/dinfer.git` 分支 `dev`。
- 历史阶段 / 已知简化 / 对拍口径：git log、`docs/design.md` §4、`test/<model>/<日期>/*.md`、pitfalls。
- 阻塞 / 待决策：`docs/design.md` §9。

<!-- ===== 日志：按日期追加，不改旧条目 ===== -->

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
- 11:00– design §10 细化（算子插拔 / KV 分层）：先盘现有算子调用面（26 个 k:: 入口 + 1 个 Gemm）与 KV 管理器钩子（get_computed_blocks / allocate_slots / cache_blocks / free），再写成可落地的 trait 与接入点。

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

## 日志

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
