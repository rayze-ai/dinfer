# loop: kaon3-mega —— Kaon-V3(Gemma4) decode persistent megakernel

> 2026-10-26 开。用户定：为性能直接上 megakernel（非 eager 的天花板，原生 Rust 的结构性优势），
> **公共骨架抽到 common**（crates/kernels，模型无关），kaon3 只放自己的阶段表 + 算子体（models/kaon3 侧或 kaon3 专有 cu）。

## 目标
Gemma4 decode 一步（attn + dense MLP ∥ MoE）编成单 persistent kernel，一次 launch，逼近 HBM 带宽上限。
对标：当前 eager ~25 tok/s；vLLM 非 eager ~200 tok/s。目标 ≥ 逼近后者并争取超越。

## 架构约定（承 design.md §14.3 / mega-sm100.md）
- **common（crates/kernels，模型无关，本 loop 新抽）**：持久 CTA 调度（grid=occupancy×SM、task=(stage,shard) 静态编号、
  blockIdx 轮转领取）、阶段依赖（全局 ready[stage] 计数器 + threadfence/red.release/ld.acquire 自旋）、
  GEMV persistent 模板（权重按 N 切 tile、split-K、cp.async 流水、mma.sync）。
- **kaon3 专有**：Gemma4 的阶段表、GQA paged attention 的 persistent 化、128 专家 NVFP4 MoE 的 persistent 化
  （动态 top-8 路由如何进静态任务表 = 头号拦路虎）。
- sm120(5090) 目标 arch；按分文件规则 `*_sm120.cu`，kernel 内禁 #if __CUDA_ARCH__ / 禁调优常量（走 ArchProfile）。
- 现有 mega_sm100.cu(Qwen3.5 GDN, sm100) 是参考：抽其模型无关骨架进 common，GDN 递归/conv1d/in_proj 布局是 Qwen 专有不要。

## 步骤
0. 调研(子代理 a3c3a048)：mega_sm100.cu 骨架可复用比例 + Gemma4 阶段表 + MoE 动态路由可行性 + sm120 差异。按结论定抽 common 的边界。
1. 抽 common：把持久调度 + 依赖 + GEMV persistent 做成模型无关模板（Rust 侧 + cuh）。
2. kaon3 阶段表 + 算子体：先 attn+dense persistent（MoE 若动态路由难，第一版可仍单独 launch，拿部分收益），再 MoE。
3. 对拍(同输入 vs 现 eager 路径逐位/近似) + 计时(speedup) + 端到端 serve 接入 + qa 质量不回退。

## 停止条件
decode 一步 megakernel vs 现 eager 11+ kernel 路径：输出一致(或亚决策噪声内) + speedup 打印；端到端 tok/s 明显跳变且质量不回退。

## 记录
- R0 2026-10-26：方向定(方案B,直接 kaon3 megakernel,公共抽 common)；调研中。
