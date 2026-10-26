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

## R2 2026-10-26 地基：arch.rs 加 Sm120 + arch.cuh 能力常量
- arch.rs：`enum Arch` 加 `Sm120`；`for_cc` 改 `10=>Sm100, _=>Sm120`（原 `_=>Sm100` 会让 5090 cc=12 错用 B300 228KB 参数→launch 失败，即调研说的必修坑）；
  `mega_params` 给 Sm120 初值 groups=3/stages=2/ctas_per_sm=1（~100KB smem，待 5090 实测调）；其余 Arch 匹配位补 Sm120 保持一致。
  校验：`cargo check -p dinfer-kernels --features no-nvcc` 里 arch.rs 相关错误 0（余 34 个是既有 no-nvcc stub 生成失败，改动前后同数，与本改无关）。
- 新建 `crates/kernels/cuda/arch.cuh`：唯一 arch 分叉点，只放有名能力常量（kArchSm8x/90/100/120、kSmemBytes、kHasTma、kHasMmaFp4BlockScale、kHasTcgen05）；
  规则写明 kernel 体内禁 #if __CUDA_ARCH__、调优数值不落此处（走 ArchProfile）。**待接进 build.rs 编译链并加 smoke 编译验证。**

## R3 2026-10-26 约束（用户）：5090 只考虑单卡
- **单卡**（用户 2026-10-26 明确：不是单机多卡，是**只考虑单卡**）：无 TP / DP / PD / 分布式 EP，无任何卡间通信。
  megakernel 面向单卡全层（hidden 2816 不分片、128 专家全在卡内、KV 全在本卡）。
- 因此 persistent 任务表**不含跨卡 allreduce/all-gather 阶段**；MoE 专家分桶沿用单卡版本。
- 性能验收：**5090 单卡**门禁；不为 B300/多卡做适配而阻塞第一版（sm100 case 保留但非当前目标，不为其改设计）。

## R4 2026-10-26 抽出 mega_common.cuh（模型无关骨架）+ arch.cuh 冒烟通过
新建 `crates/kernels/cuda/mega_common.cuh`（namespace dinfer::mega），只放模型无关件：
  同步原语（ld_acquire_gpu / red_release_gpu_add / gtimer）、`StagePlan{dep,count,start,ready,prof}` +
  `wait_stage/signal_stage`（依赖表由 host 填，不绑任何模型 Stage 枚举）、cp.async(zfill/wait_pipe)、
  `mma16816`、`swz`、`stage_of`（task→(stage,shard) 二分）、`mega_kernel<Dispatch,OCC>` 持久外壳
  （blockIdx 轮转领 task → 二分 stage → 等依赖 → dispatch.run → signal）。模型侧只需给 `Dispatch{Params, run()}`。
arch.cuh 修正：常量原只在 `#if defined(__CUDA_ARCH__)` 内定义 → host pass 取不到；改为 `DINFER_ARCH_CC` 宏
  （device=__CUDA_ARCH__，host=0），两 pass 都可见；host 侧不用这些常量（走 Rust ArchProfile）。
**冒烟**：本机 nvcc12.4 不支持 sm_120a/sm_100，用 sm_86 做语法冒烟 → `mega_common.cuh`+`arch.cuh` 编译零错误
  （kernel 只用 sm80+ 指令，符合设计）。真实 sm120a/sm100 编译验证到远端 5090/B300 做。
下一步：kaon3 侧 Dispatch（Gemma4 阶段表 + attn/dense 任务体），先 attn+dense、MoE 留单独 launch。

## R5 2026-10-26 委派实现第一版（attn+dense 11 阶段）
子代理 a3c2dc5e 写 `models/kaon3/cuda/mega_kaon3_sm120.cu`（Gemma4 Stage 枚举 + Params + Dispatch 任务体 + host C ABI）
+ Rust 包装 + `KAON3_MEGA=1` env 开关走 megakernel、否则现有路径（便于 A/B）。
硬要求：编过(远端5090) → 逐子模块对拍(sub_dump) → 打印 speedup(B=1) → **如实报负收益**。
用户指示：**编过就先对拍看数字**（先不管性能调优）。
主会话纪律：不抢子代理文件/远端连接，等它报结果再推动。

## R6 2026-10-26 主会话自己写第一版：dense MLP 支路 persistent（收窄范围）
**停掉子代理（读代码太久零产出），主会话自己写。并进一步收窄第一版范围**：
  不做 attn、不做 MoE，**只做 dense MLP 支路**：pre_ff_norm → gate ‖ up(fp8 GEMM) → gelu·mul(round2) → down(fp8 GEMM) → post_ff_norm_1。
  理由：最快拿到"消 kernel 边界值多少"的可编可对拍最小闭环，避开两个硬骨头。
**写的过程中修掉的真实错误（草稿→正确）**：
  1. dense gate/up 是**两个独立 QLinear**（各自权重+各自 per-tensor scale），不是合并 [2I,H]（读 layer.rs:353-354 确认）；
  2. gelu 必须走 `gelu_tanh_mul_round2_bf16` 口径（gelu 先舍 bf16 再乘 up，vLLM act_and_mul）；
  3. fp8 片段寻址必须照 `gemm_fp8_sm120.cu` 已验证的版本（A 用 ldm_x4 + `r=lane&15, c=(lane>=16)?16:0`；B 用 ldm_x2 + `rb=lane&7, cb=(lane&8)?16:0`），我草稿里的寻址是错的。
  4. 依赖表无法表达"两前驱"（gate 与 up 都要完成后才 act）→ 第一版**串行 gate→up**（dep[UP]=GATE），牺牲并行保正确。
**文件位置**：`crates/kernels/cuda/mega_mlp_sm120.cu`（不是 models/kaon3/cuda —— 那目录没有 build.rs 不参与编译；
  且此 kernel 只吃维度参数 H/I/m + gelu_tanh，**模型无关**，按 CLAUDE.md 规则应放 crates/，已验证零模型名）。
编译中（5090，CUDA_ARCHS=120）。

## R7 2026-10-26 mega_mlp_sm120 **编译通过**（5090/sm120a）+ 对拍测试已写
编译错误逐个修完：①缺 `#include <cuda_fp8.h>`；②A 必须先量化成 fp8 才能喂 fp8 GEMM（原直接传 bf16 = 硬 bug）→
  S_NORM 里同时产出 norm1(bf16) + norm1_q(fp8)；③`StagePlan` 要写 `mega::StagePlan` 限定。
FFI：`crates/kernels/src/ffi.rs` 加 `mega_mlp_dense_branch`（`cfg(all(not(no_nvcc_stub), sm120))` + 非 sm120 返回 801 stub）。
对拍：`crates/kernels/tests/mega_mlp_parity.rs`——参考路径用现有逐 op（rmsnorm→gemm_fp8_static(cuBLASLt)×2→
  gelu_round2→gemm→rmsnorm），mega 一次 launch；先退化检查（两边范数非零同量级）再比 ndiff/max|d|/rel。
  fp8 量化/累加顺序两路径不同，预期非逐位，应在 fp8 级差(rel<5e-3)；系统性偏差=我的 kernel 逻辑错。
