# 进度

<!-- ===== 当前状态（交接区）：每次收功整体重写，保持 ≤ 20 行 ===== -->
## 当前状态

- 更新日期：2026-10-14
- 阶段：**P0 通过**（`scripts/verify/p0.sh` exit 0：fmt / build glm53+kimik3 / clippy -D warnings / 9 个 CPU 单测 / 3090 上 add kernel GPU 测试 / `dinfer device-info`）。基线已 commit
- 正在做：准备 P1
- 环境：本机 3090（sm86）日常开发；k8s `admin@polaris` 两节点各 8 × RTX 4090（sm_89，驱动 580.178.04），Pod 需 `runtimeClassName: nvidia`，见 `scripts/dev/k8s/README.md`。用户决定：先做 sm8x 能完成的部分，sm90 以后再给
- 下一步（具体到可直接执行）：
  1. P1 调度器：按 vLLM `v1/core/sched/scheduler.py` 实现 running-first / waiting、token 预算、chunked prefill、抢占（释放末尾 running）、spec token 占位（async scheduling 的 `num_output_placeholders`）；表驱动单测覆盖 chunked / 抢占 / 命中三类 case
  2. P1 KvCacheManager：多 group（Mla / Mamba / Indexer 各一组）、`get_computed_blocks`（xxh3 链式 hash，`hash_block_size`）、`allocate_slots`、LRU 淘汰、Mamba 组按 `tokens_per_state` 折算
  3. criterion 门禁：`cargo bench -p dinfer-core` 中 `schedule+update 1024 decode` < 200 µs（现在只有骨架）
  4. 每阶段完成跑 `scripts/verify/p1.sh`（待写：p0 全部 + bench 阈值断言）
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
- 同日下午：用户确认先做 sm8x 可完成部分；写出 P0 骨架：根 `Cargo.toml`（workspace，固定版本依赖，cudarc 0.18.2 / axum 0.8 / tokio 1.45）、`crates/{core,kernels,layers,server,kimik3,glm53}`、`bins/dinfer`（feature `kimik3|glm53` 互斥，`no-nvcc` 回退）、`crates/kernels/cuda/add.cu` + `build.rs`（nvcc，`DINFER_CUDA_ARCHS` 默认 86,89,90）、core 的 config/request/sched(最小 FCFS)/kv(BlockPool, KvCacheSpec)/sample、两模型 `config.rs`（解析官方 config.json 并断言结构事实）、`scripts/verify/p0.sh`、`scripts/dev/k8s/{gpu-probe,dev-pod}.yaml`。全部未编译。
- 未完成：用户确认计划；基线 commit；P0。（CLAUDE.md"项目是什么"一节当时被工具拦截，已于 10-14 补上。）
- 踩坑：见 `docs/pitfalls.md` 2026-10-13 两条。
