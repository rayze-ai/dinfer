# 新模型接入方法论（2026-10-19，从 Qwen3 / GLM-5.3 / Kimi-K3 三条线提炼）

> 用户 2026-10-19："写了那么久，接新模型应该有一套方法论了"。本文是那套方法论；每接一个模型按此走，走完回来修订本文。

## 0. 一句话

**先把"事实"钉死，再把"差距"分三类，kernel 先于模型、对拍先于性能、单请求先于批、本机先于大卡。**

## 1. 钉事实（动代码前，半天）

1. **config 不可信，权重 header 才可信**。`config.json` 看 `model_type` / `architectures` / 是否多模态壳（`text_config`）；再用 safetensors header（前 8 B 长度 + JSON，不读数据）抽 2–3 个分片，拿到**真实张量名、dtype、形状**。GLM 的 DSA 索引器、K3 的 MXFP4 `weight_packed`、Qwen3.5 的 `in_proj_a/b` 和 q_proj 翻倍（输出门）都只在 header 里看得出来。
2. **写进 `docs/design-<model>.md` §1 "结构事实"**：层数、hidden、inter、每类层的头数 / head_dim、RoPE 类型与比例、norm 位置、激活、vocab、tie、eos、量化格式、MTP 层数、chat 模板与特殊 token。一行一个事实，带来源。
3. **记"不确定"**：mrope 对纯文本怎么退化、output gate 的拼接顺序、A_log 的符号约定——这些写进 §3 "风险 / 待定"，由 vLLM 源码或对拍解决，不猜。

## 2. 分差距（半天）

对照 `crates/{core,kernels,layers,text,server}` 已有能力，把模型需求分三类，写进 §1 末尾"与现有代码的差异"：

| 类 | 判据 | 做法 | 例 |
|---|---|---|---|
| **直接复用** | 形状参数化即可 | 不改 | RMSNorm、SwiGLU、GEMM、scheduler / runner / engine、server、tokenizer、minijinja 模板 |
| **扩模板 / 加参数** | 同算法、新形状或新开关 | 加模板实例、运行时参数，**不复制 kernel** | gqa D=128 → 256、GROUP 2 → 6、partial RoPE、KV spec 新变体 |
| **全新** | 新算法 | 新 `.cu`（按 arch 分文件规则）+ Rust 封装 + 对拍 | KDA / GDN delta-rule、MXFP4 GEMM、DSA 索引器 |

全新类每个都要一份 `docs/loops/<feature>.md`（环境 / 观察 / 停止条件 / 上限）。

## 3. 实现顺序（固定，不要跳）

1. **kernel 先于模型**：全新类 kernel 先独立写、独立对拍（`ops_parity` 模式：torch / Triton 参考 → `.npy` → Rust 断言；硬判决按 `docs/pitfalls.md` 口径只在可判定行断言）。两卡（4090 / B300）都过。
2. **config.rs + weights.rs**：解析 + `check()` 断言支持范围（不支持就报错，不静默）；名字映射、融合规则（qkv 合并、gate_up 合并、输出门拆分）写成表放设计文档 §2.1，代码照表。先用 `scripts/dev` 造 **tiny golden**（2–3 层、小 vocab、真结构、随机权重 + HF 侧 dump），本机 3090 就能跑。
3. **单请求 prefill logits 对拍 HF**（tiny golden），再 greedy 32 步，再双请求混批。只在这一步过了才碰 scheduler / graph / 采样。
4. **ChatBackend**：tokenizer / 模板 / 思考段 / tool_call 解析对拍 HF golden（`chat_parity.rs` 模式）。
5. **真权重**：目标卡加载、e2e 与 vLLM 容器输出对拍（前 N token 一致，temperature 0），再 `bench_compare.sh` 进 CSV（新模型新目录 `test/<model>/<日期>/`，CSV 加模型维度）。
6. **性能**：到这一步才开 loop 调性能，口径同 B4：nsys 定位 → 改 → 两卡门禁。

## 4. 硬性规则（CLAUDE.md 已有，这里只列接模型时最容易犯的）

- `crates/` 零模型信息：模型名 / token id / 模板变量 / 路径出现即错，走 `ChatBackend` 钩子。
- kernel 不写死调优数值；按 arch 分文件；kernel 体内禁 `#if __CUDA_ARCH__`。
- 与 vLLM 的对照：直接采用或改进都记 `design.md` §11 台账一行。
- 每个可反复调试的功能先 loop 文档再动手；每轮 bench 进 CSV。
- 文档中文；commit 带 Co-Authored-By。

## 5. 每模型一张"接入清单"（复制到 design-<model>.md §0 跟踪）

```
[ ] §1 结构事实（config + header）   [ ] §1 差距三分类   [ ] 全新 kernel 的 loop 文档
[ ] kernel 对拍（两卡）              [ ] config/weights + tiny golden     [ ] prefill logits 对拍
[ ] greedy 32 / 混批                 [ ] chat_parity                     [ ] 真权重 e2e vs vLLM
[ ] bench_compare 进 CSV             [ ] design.md §11 台账              [ ] 本文修订
```

## 6. 已接模型的经验（供下一个模型参考）

- **Qwen3-0.6B**：全部复用 + 一个全新 kernel（GQA paged attention）。坑：HF eager vs sdpa 自身差 0.68，对拍口径要以 "argmax 在 HF top-2 gap 内" 为准，不是逐位。
- **GLM-5.3**：DSA 索引器 + FP8 MoE + MTP；坑：硬判决（top-k 路由）不可逐位复现，参考侧要导出判决分数。真权重挂起。
- **Kimi-K3**：KDA + MLA 混合 + MXFP4 + 多模态壳；K3 线独立子代理实现中（`docs/loops/k3-kda.md`）。
- **Qwen3.5-27B**（本轮）：GDN + 全注意力混合、head_dim 256、输出门、partial mrope、lm_head 不 tie、FP8 版可选。见 `docs/design-qwen35.md`。
