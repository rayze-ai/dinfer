# 进度

<!-- ===== 当前状态（交接区）：每次收功整体重写，保持 ≤ 20 行 ===== -->
## 当前状态

- 更新日期：2026-10-14
- 阶段：**P1 通过**（`scripts/verify/p1.sh` exit 0：p0 全部 + 27 个 core 单测 + criterion 门禁 1024 decode step pc=off 86 µs / pc=on 138 µs < 200 µs）。P0 基线 `cda9aea`，文档 `e7943b8`，P1 待 commit
- 正在做：P1 收尾 → 开 P2
- 环境：本机 3090（sm86）日常开发；k8s `admin@polaris` 两节点各 8 × RTX 4090（sm_89，驱动 580.178.04），Pod 需 `runtimeClassName: nvidia`，见 `scripts/dev/k8s/README.md`。用户决定：先做 sm8x 能完成的部分，sm90 以后再给
- P1 产出：`core/src/sched/mod.rs`（running-first / waiting、token 预算、chunked prefill、`long_prefill_token_threshold`、末尾抢占、async scheduling 占位 + spec 回退、slab 索引）、`core/src/kv/{block_pool,hash,manager}.rs`（LRU 双链表 + (group,hash) 索引、xxh3-128 链式块哈希、多 group 管理器含 Mamba 单状态块、decode 零分配快路径）、`sched/output.rs`（`StepResults` 扁平缓冲）、`examples/probe_sched.rs`、`docs/loops/p1-scheduler-perf.md`
- P1 已知简化（P7 补）：Mamba 组只在 block 对齐边界登记/命中，不做逐级回退与 checkpoint 多块；所有 group 要求同 page 大小；无 priority 调度、无 encoder 预算
- 下一步（具体到可直接执行）：
  1. commit P1
  2. P2 基础算子 + layers：`kernels/cuda/{rmsnorm,rope,silu_and_mul,embedding,sampling}.cu` 参考版（sm80+，bf16 用 `__nv_bfloat16`）；cuBLASLt bf16 GEMM 封装；`layers` 里 `Linear / RmsNorm / Rope / Embedding / Mlp / Sampler`；每个算子与 PyTorch 对拍（`scripts/verify/kernels.sh`：Python 生成随机输入与期望输出到 `.npy`，Rust 读入比对 atol 1e-2）
  3. P2 门禁：`scripts/verify/p2.sh` = p1 + kernels 对拍全过
- 阻塞 / 待决策：`docs/design.md` §9（KDA kernel 手写 CUDA 还是 Triton AOT；structured output；性能对比基线配置）
- 本机工具链：rustc/cargo 1.89.0，nvcc 12.4 在 `/usr/local/cuda/bin`（不在默认 PATH；p0.sh 已自动加）；crates.io 直连可用
- 进行中的 loop 规格：无
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

### 2026-10-14

- 做了什么：按用户要求补性能分析。读 V2 runner（`gpu/model_runner.py`、`states.py`、`buffer_utils.py`、`async_utils.py`）、`core.py step/busy loop`、`async_scheduler.py`、`detokenizer.py`、`cache.py`，写 design.md §8：16 项瓶颈（B1-B16，标注源码核实/经验判断）、每层算子融合目标、对标 vs 另起清单、基准方法；requirements 验收口径加性能指标；P1/P3 起加性能早期门禁。
- 怎么验证的：源码引用带行号；性能数字（launch 数、µs 目标）是设计目标，待 P1/P3 实测。
- 未完成：同 10-13（用户确认计划 → 基线 commit → P0）。CLAUDE.md 已补齐。
- 同日深夜（P1）：读 vLLM `scheduler.py:591-1660`、`kv_cache_manager.py:264-620`、`block_pool.py`、`single_type_kv_cache_manager.py`（FullAttention/Mamba 的 find_longest_cache_hit、get_num_blocks_to_allocate）、`kv_cache_coordinator.py:796`，对齐语义后写 Rust。三轮迭代：① 初版 HashMap 版 27 测试 20 过，bench 298/449 µs；② 修三个测试预期（32 token 恰满 2 块、抢占恢复会命中自己留下的缓存块、阈值测试时序）+ slab/恒等哈希/扁平结果/跨边界早退，bench 197/310；③ 发现 bench 的 `iter_batched` 把 setup 算进去了（手工计时只有 62/92），改持久状态 `iter` → 95/146。`scripts/verify/p1.sh` PASS。
- 踩坑：criterion `iter_batched`（见 pitfalls）；本机无 perf。
- 同日晚：P0 编译通过并 PASS。三轮修复：① cc `cuda(true)` 塞 gcc 参数 → build.rs 直接调 nvcc + `object()` 归档；② bin 直接用 `axum::serve` 但未声明依赖 → 收进 `dinfer_server::serve`；③ `BlockPool::allocate` 返回顺序与测试期望相反 → `split_off` 后 `reverse()`。k8s 第二节点 `host-10-121-192-98` 亦确认 8 × 4090。基线 commit 完成。
- 同日下午：用户确认先做 sm8x 可完成部分；写出 P0 骨架：根 `Cargo.toml`（workspace，固定版本依赖，cudarc 0.18.2 / axum 0.8 / tokio 1.45）、`crates/{core,kernels,layers,server,kimik3,glm53}`、`bins/dinfer`（feature `kimik3|glm53` 互斥，`no-nvcc` 回退）、`crates/kernels/cuda/add.cu` + `build.rs`（nvcc，`DINFER_CUDA_ARCHS` 默认 86,89,90）、core 的 config/request/sched(最小 FCFS)/kv(BlockPool, KvCacheSpec)/sample、两模型 `config.rs`（解析官方 config.json 并断言结构事实）、`scripts/verify/p0.sh`、`scripts/dev/k8s/{gpu-probe,dev-pod}.yaml`。全部未编译。
- 未完成：用户确认计划；基线 commit；P0。（CLAUDE.md"项目是什么"一节当时被工具拦截，已于 10-14 补上。）
- 踩坑：见 `docs/pitfalls.md` 2026-10-13 两条。
