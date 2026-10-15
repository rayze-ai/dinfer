# 需求

## 要做什么
现在vllm 为了兼容各种模型，写了大量的插件，支持丰富模型的同时也带来了项目的臃肿，
现在用rust来重构下下vllm,针对每个模型形成自己独特的推理，规划如下
1. 先分析vllm 主要的核心部分，
比如启动阶段：参数解析，权重加载，WARMUP，cudagraph；推理阶段：tokenizer,block分配，调度等，需要有整体的模块划分
2. 目录规划
crate: core 核心抽象
crate: server httpserver 推理的相关接口
kimik3: kimik3的具体模型实现
glm53: glm53的具体模型实现
构建是选 core+ server+具体模型实现就行（比如kimik3）

> 分析结果与实现计划见 `docs/design.md`。目标模型版本：`moonshotai/Kimi-K3`、`zai-org/GLM-5.3`（HF 2026-10 主分支）；参考 vLLM main `de8291e`。

## 不做什么

<!-- 以下为根据分析提出的建议范围，待用户确认 -->

- 不做通用 HF 模型加载与模型注册表；只支持上述两个模型（及其同结构 tiny 测试模型）。
- 第一版不做：LoRA、Pooling/Embedding 任务、Encoder-Decoder、Pipeline Parallel、EPLB、KV connector / P-D 分离、sleep mode、torch.compile 等价物、CPU/ROCm/其他平台。
- 不做鉴权；服务默认只监听本机，对外需反代。
- 不复现 vLLM 的全部 CLI 参数，只保留与两模型相关的。

## 验收口径

<!-- 建议口径，待用户确认 -->

1. 正确性：同结构 tiny 随机权重模型下，greedy 生成前 N token 与 HF transformers 一致；每个 kernel 与 PyTorch 逐算子对拍（bf16 atol 1e-2）。
2. 功能：OpenAI SDK 能走通 `/v1/chat/completions` 流式与非流式，含 `reasoning_content`、tool_calls、`reasoning_effort`；K3 支持图像输入。
3. 真权重：在目标机器上加载真权重并输出合理文本；MTP 投机解码可开关。
4. 性能（核心目标之一，详见 `docs/design.md` §8）：同机同配置与 vLLM `de8291e` 对比，decode TPOT P50 优于 vLLM ≥ 20%，TTFT P50/P99 不劣，P99 TPOT 抖动更小；每 step 引擎 CPU 开销 < 200 µs @ 1024 并发；decode 阶段每层 kernel launch 数 GLM ≤ 5、K3 ≤ 6（nsys 统计）。
5. 所有阶段有 `scripts/verify/*.sh`，exit 0 为通过。

## 约束

- 开发机：RTX 3090 24 GB（sm86）、CUDA 12.4、cargo 1.89。本机只能跑 tiny 模型与算子对拍；真权重需多卡 H100/H200/B200 级别机器。
- 语言：Rust；CUDA/C++ 仅限 kernel 与 FFI 层。
- 依赖固定版本；第三方 kernel（FlashMLA、DeepGEMM 等）以 git submodule 或 vendored 源码引入，许可证需为 MIT/Apache/BSD。
- 文档中文，代码与命令原样。

## 范围调整（2026-10-15）

- 暂无高端卡：GLM-5.3 / Kimi-K3 真权重验收挂起，已完成部分（GLM-5.3 全链路在 tiny fixture 上通过 P0–P6）记为阶段性成果。
- 新增 **Qwen/Qwen3-0.6B** 作为主流程验证模型（本机 3090 可跑真权重）：真权重端到端、采样、性能、PD 分离都在它上做。
- kimik3 / glm53 / qwen3 三个模型 crate 地位相同：`core + server + 一个模型 crate`。详见 design.md §5b。
