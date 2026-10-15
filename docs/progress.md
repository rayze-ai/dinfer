# 进度

<!-- ===== 当前状态（交接区）：每次收功整体重写，保持 ≤ 20 行 ===== -->
## 当前状态

- 更新日期：2026-10-15
- 阶段：**P4 通过（a/b/c/d）**，`scripts/verify/p4.sh` exit 0（= p0–p3 + MoE 对拍 + runner 单测 + Scheduler+Runner+Executor 双请求对拍 plain/chunked × dense/MoE + tiny fp8 GLM 同套 3+2 对拍）。commits：P0 `cda9aea`，P1 `65a67f1`，P2 `d44e49f`，P3 `17930a5`，P4-c `35b54e3`，P4-a `69c8478`，P4-b `ea415e3`，P4-d + docs 见 git log
- P4-d fp8（已验证）：`layers/src/weights_fp8.rs`（e4m3 解码 + 128×128 块 dequant，2 单测）、`WeightSource::{with_quant_block, dequant_fp8}`（`load_bf16 / load_f32_host` 对 `F8_E4M3` 自动解量化，需同名 `weight_scale_inv`，块数 `⌈N/128⌉×⌈K/128⌉` 末块可不满）、`GlmMoeDsaConfig::fp8_block_size()`、`gen_tiny_glm.py --fp8`（28 个投影权重量化，期望值用解量化权重算）。tiny fp8 GLM：prefill 0.0076 / 32 步 worst 0.0073 / runner plain 0.0077 / chunked 0.0078
- 正在做：**P5 DSA 索引器 + 稀疏 MLA**。代码齐、能编译、**24 个文件未提交**；`scripts/verify/p5.sh` **未通过：3 道 `dsa_parity` 全 FAIL**（2026-10-15 16:30 实跑，详见 `test/2026-10-15/p5-dsa-parity.md`）。失败都不是算错而是"可判定行太少"：self 自算可判定 28/256（需 ≥ 64，可判定行 worst 0.0078 ✅）；注入 HF top-k 路由 near-tie 45/256（需 ≤ 25，211 可判定行 worst 0.0098 ✅）；topk 集合 L2_long row 124 分差 0.1166 刚超 band 0.1130（索引器分数误差涨到 0.11，之前 0.024，原因未查）。**下一步（可直接执行）**：① 改 `gen_tiny_glm.py --dsa` 的 fixture 让平局率降下来（路由 near-tie < 10%、索引器可判定 ≥ 64：调索引器权重放大系数 ×6 / 换 seed / 减专家数），重跑 `DINFER_TINY_GLM=$(realpath test/tiny_glm_dsa) cargo test -p dinfer-glm53 --features ref-tests --test dsa_parity -- --ignored --test-threads=1 --nocapture`；② 查 row 124 的 0.11 分数误差；③ PASS 后 `scripts/verify/p5.sh` → commit "P5: DSA indexer + sparse MLA"，design.md P5 行打勾 → P6。loop 规格 `docs/loops/p5-dsa-indexer.md` 已记 5 轮，上限 10
- P4 产出：`kernels/cuda/moe.cu`（gate_f32 / route / bucket / gather / scatter_add / f32_add_to_bf16）、`kernels/cuda/mla.cu::mla_prefill_paged`、`layers/src/moe.rs`、`layers/src/mla.rs::AttnMeta::Paged + absorbed_q/absorbed_o`、`core/src/runner.rs`（ReqState / BatchPlan / BatchMode / ModelExecutor / Runner）、`glm53/src/{model.rs::Glm53Model+FeedForward, executor.rs::Glm53Executor}`、`glm53/tests/{tiny_parity,runner_parity}.rs`、`scripts/verify/p4.sh`、`docs/loops/p4-moe-runner.md`
- P4 已知简化（P10 / 后续补）：runner 每 step 全量 H2D 输入、argmax 后同步 D2H，无 pinned / 双缓冲 / CUDA graph；MoE 逐专家循环 + 一次 D2H 读 offsets（真机换 grouped GEMM）；Decode/Paged 路径每 head 一次小 GEMM（W_UK/W_UV）；fp8 只做加载期 dequant（sm86 无 fp8 张量核，1×128 激活量化 + 真 FP8 GEMM 留 4090/sm90）；采样只有 greedy；无投机解码；只支持单 KV group
- 下一步（P4 收尾后）：P5 DSA 索引器（`index_n_heads 32 / index_head_dim 128 / index_topk 2048 / index_topk_freq 4`，`indexer_types` full/shared 交替；参考 vLLM `deepseek_v32/indexer.py`），tiny GLM seq 4096 对拍
- 环境：本机 3090（sm86）日常开发；k8s `admin@polaris` 两节点各 8 × RTX 4090（sm_89，驱动 580.178.04），Pod 需 `runtimeClassName: nvidia`，见 `scripts/dev/k8s/README.md`。用户决定：先做 sm8x 能完成的部分，sm90 以后再给
- P1 产出：`core/src/sched/mod.rs`（running-first / waiting、token 预算、chunked prefill、`long_prefill_token_threshold`、末尾抢占、async scheduling 占位 + spec 回退、slab 索引）、`core/src/kv/{block_pool,hash,manager}.rs`（LRU 双链表 + (group,hash) 索引、xxh3-128 链式块哈希、多 group 管理器含 Mamba 单状态块、decode 零分配快路径）、`sched/output.rs`（`StepResults` 扁平缓冲）、`examples/probe_sched.rs`、`docs/loops/p1-scheduler-perf.md`
- P1 已知简化（P7 补）：Mamba 组只在 block 对齐边界登记/命中，不做逐级回退与 checkpoint 多块；所有 group 要求同 page 大小；无 priority 调度、无 encoder 预算
- P2 产出：`kernels/cuda/{common.cuh,add,norm,rope,activation,sampling}.cu`（9 个参考算子，sm80+，bf16 用 `__nv_bfloat16`），`kernels/src/{ffi,ops,gemm}.rs`（类型安全包装 + cuBLASLt bf16 GEMM，权重 `[N,K]` 行主序），`kernels/tests/ref_tests.rs`（feature `ref-tests`，读 `.npy`，路径按 `CARGO_MANIFEST_DIR/../..` 解析），`scripts/verify/{gen_kernel_refs.py,kernels.sh,p2.sh}`，`docs/loops/p2-kernels.md`。**C ABI 符号不带 `dinfer_` 前缀（用户要求）**。本机 torch 2.6.0+cu124 / numpy 1.26 / safetensors 0.5 可用
- P3 产出：`kernels/cuda/{mla,layout,elementwise}.cu`（mla_cache_write / mla_prefill / mla_decode；copy_cols / bcast_kpe / copy_head_cols / gather_head / scatter_head；add_inplace / gather_rows / bf16_to_f32）+ 对应 ffi/ops；`layers/src/{tensor,weights,nn,mla}.rs`（WeightSource 自解析 safetensors mmap；MlaAttention 含 W_UK^T/W_UV 按 head 拆分、prefill 展开路径与 decode 吸收路径）；`glm53/src/model.rs::Glm53Dense`；`glm53/tests/tiny_parity.rs`（3 个测试）；`scripts/verify/{gen_tiny_glm.py,p3.sh}`；`docs/loops/p3-mla-dense.md`
- P3 已知简化（后续补）：decode 路径每 head 一次小 GEMM（W_UK/W_UV），H 个 launch → P10 换 batched GEMM；prefill 不支持 chunked context（有历史 KV 的 prefill）→ P4 runner 接 chunked prefill 时补 "prefill + 读 cache" 路径；无 runner / Sampler 层，测试直接喂 Forward 结构体
- 阻塞 / 待决策：`docs/design.md` §9（KDA kernel 手写 CUDA 还是 Triton AOT；structured output；性能对比基线配置）
- 本机工具链：rustc/cargo 1.89.0，nvcc 12.4 在 `/usr/local/cuda/bin`（不在默认 PATH；p0.sh 已自动加）；crates.io 直连可用。**cargo 假失败时 `rm target/.rustc_info.json`**（cron trace 污染，见 pitfalls 2026-10-15）
- 进行中的 loop 规格：`docs/loops/p5-dsa-indexer.md`（第 5 轮，未通过）
- 会话注意：本项目会话两次因 `API Error: 400 Input is too long` 被迫 /clear（10-14 P4、10-15 P5）。长 bisect / 大日志输出前先 wrap；Bash 输出一律 `grep`/`tail` 截断，不要整段 cat 日志
- 参考材料：`/data/ref/vllm`（main de8291e）、`/data/ref/models/{kimik3,glm53}`（config.json、modeling_*.py、tokenizer、chat 模板、safetensors index；无权重）

<!-- ===== 日志：按日期追加，不改旧条目 ===== -->
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
- 未完成：让 fixture 平局率降下来（或阈值自适应）→ `p5.sh` PASS → commit P5 → design.md 阶段表 P5 行。
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
- 同日下午：用户确认先做 sm8x 可完成部分；写出 P0 骨架：根 `Cargo.toml`（workspace，固定版本依赖，cudarc 0.18.2 / axum 0.8 / tokio 1.45）、`crates/{core,kernels,layers,server,kimik3,glm53}`、`bins/dinfer`（feature `kimik3|glm53` 互斥，`no-nvcc` 回退）、`crates/kernels/cuda/add.cu` + `build.rs`（nvcc，`DINFER_CUDA_ARCHS` 默认 86,89,90）、core 的 config/request/sched(最小 FCFS)/kv(BlockPool, KvCacheSpec)/sample、两模型 `config.rs`（解析官方 config.json 并断言结构事实）、`scripts/verify/p0.sh`、`scripts/dev/k8s/{gpu-probe,dev-pod}.yaml`。全部未编译。
- 未完成：用户确认计划；基线 commit；P0。（CLAUDE.md"项目是什么"一节当时被工具拦截，已于 10-14 补上。）
- 踩坑：见 `docs/pitfalls.md` 2026-10-13 两条。
