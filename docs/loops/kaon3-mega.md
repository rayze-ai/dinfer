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

## R1 2026-10-26 调研结论 + 按 sm 分 case 设计（用户：需要按照 sm 做不同的 case）
**风险如实记录**：唯一可对照实验（Qwen3.5 单 GDN 层 mega，B300）**失败 0.59–0.91×**，根因
  ① persistent 1 CTA/SM 占用打不过各自调优的独立 GEMM kernel；② 小算子铺上百 CTA 变延迟链。
  sm120(5090) SM 更少、带宽更低 → 两根因更重。**第一版必须把"是否正收益"当真实问题验，不假定会赢。**

### 文件划分（arch 分文件，符合 CLAUDE.md 多卡型规则）
```
crates/kernels/cuda/mega_common.cuh       # 模型无关+arch无关：持久外壳(任务领取/stage二分/末CTA自复位)/GEMV persistent/同步(cp.async,ld.acquire,red.release)/rmsnorm_row
crates/kernels/cuda/arch.cuh              # 唯一分叉点，只用有名能力常量（kSmemBytes/kHasTma/kMmaFp4Kind）
crates/kernels/cuda/nvfp4_common.cuh      # 从 gemm_nvfp4_sm120.cu 抽 fp4 mma/quant16/e4m3_f（MoE 第二阶段用）
models/kaon3/cuda/mega_kaon3_sm120.cu     # 5090 case：Gemma4 stage 表 + attn/rope/norm 阶段体
models/kaon3/cuda/mega_kaon3_sm100.cu     # B300 case：同接口另一套装载/参数
crates/kernels/src/arch.rs                # 加 Arch::Sm120 + 5090 MegaParams
```

### 两卡差异（实测，非猜）
| | sm100/sm103 (B300) | sm120 (5090) |
|---|---|---|
| smem/CTA | 228 KB | ~100 KB |
| 指令 | cp.async+mma.sync（无 TMA/tcgen05/cluster/mbarrier） | 同左；fp4 `mxf4nvf4.block_scale` 需 compute_120a |
| 编译链 | sm100_only + cfg!(mega_sm100) | build.rs 已有 sm120 nvcc 链(compute_120a) |
| MegaParams | groups=4/stages=3/228KB | 按 ~100KB 重算；**加 ctas_per_sm 1|2 两档**（正对失败根因①） |

### 必修坑
arch.rs 现**无 Sm120 变体**，`for_cc` 把 `cc>=10` 全归 Sm100 → 5090 错用 B300 的 228KB 参数 → launch 失败。
必须加 `Arch::Sm120` + 5090 MegaParams，Rust 侧按 ArchProfile 选。

### 第一版范围（快出数字）
只吞 attn+dense MLP（阶段 1–11：input_norm→qkv→qk/v-norm→rope→kv写→paged attn→o_proj+残差→pre_ff_norm→dense gate_up→dense down+norm1），
**MoE 保持现有逐 kernel launch**（top-8/128 数据依赖，硬塞静态表→128 专家占坑>90%空转，撞根因②）。
收益预期：乐观 5–12% 单层，**可能 0 或负**；第一版价值=拿到 5090 上 attn+dense 真实数字（1–2 周），再定是否碰 MoE。

### 验收
- 对拍：MoE 分界点两边均为现有缓冲，layer.rs sub_dump 逐子模块比。
- 计时：打印 speedup(B1/B2/B4)；**两卡门禁**（bench_gate.sh，任一卡退化>3% 不合入）。
- 端到端：serve 接入后 qa_zh2 质量不回退。

### 动手顺序
1. arch.cuh 能力常量 + arch.rs Sm120/MegaParams（两 case 共同地基）
2. mega_common.cuh 抽骨架（从 mega_sm100.cu 迁，token 级验证 = 原 Qwen mega_parity 仍过）
3. mega_kaon3_sm120.cu 第一版阶段表 + 对拍 + 计时
