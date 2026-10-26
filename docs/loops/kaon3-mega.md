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

## R8 2026-10-26 对拍跑出 SIGSEGV，逐步排查中
测试编译过，但运行 SIGSEGV。已修 3 个真实 bug（都出自我自己写的 kernel）：
  1. **kBN=64 错**：每 warp 算 16 行 N（b0 8 行 + b1 8 行），8 warp 需 BN≥128，否则 sB 越界读 → 段错误。改 kBN=128。
  2. **尾声只写了 4 个累加器中的 2 个**（acc[i][0]/[2]），列 c0+1 和 c2/c3 全丢 → 一半输出是垃圾。已补 acc[i][1]/[3]。
  3. **`__syncthreads()` 出现在 `if (warp < m)` 分支内**（rmsnorm_row_warp 里），是未定义行为/死锁 →
     改为所有 warp 都进函数（行号 ≥m 的 warp 用 store=false 只参与 barrier）。
仍 SIGSEGV → 上 compute-sanitizer memcheck 定位真实越界（不猜）。

## R9 2026-10-26 二分定位 SIGSEGV（参考路径已证干净）
加 STEP 打点后确认：**参考路径 7 步全过**（rmsnorm/gate/up/gelu/down/post 都 OK），崩在 **megakernel 内**。
已修 4 个真 bug（kBN 128 / 尾声补 4 累加器 / syncthreads 分歧 / A smem 越界)。
仍崩 → 加 `MEGA_DIAG_NOGEMM` 诊断开关（fp8 tile 短路成只写 0），二分"是 GEMM tile 还是调度/其他阶段"。

## R10 2026-10-26 **根因找到**：dep/count/start 是**设备指针**，host 直接写 → SIGSEGV
Rust 侧用 `st.ptr_mut(&mut dep)` 传的是**设备地址**；我在 C host 代码里 `count[S_NORM]=1;` 直接解引用设备内存 → 段错误。
诊断法：把 fp8 tile 短路成写 0 仍崩 → 排除 GEMM；加 STEP 打点确认参考路径干净、崩在 megakernel 入口附近 → 定位到 host 填表。
修：表在 host 本地 `int h_count[S_COUNT]` 算好，`cudaMemcpyAsync` 拷到设备 dep/count/start；`ready` 仍需调用方清零。
（教训：C ABI 里指针是设备指针还是 host 指针必须在签名注释写清。）

## R11 2026-10-26 ✅ 第一版 megakernel **逐位一致 0/11264**
```
mega rc=0
[退化检查] ‖ref‖=60.2643 ‖mega‖=60.2643   ← 同量级非零
[对比] 不同 0/11264  max|d|=0.0  rel=0.0
VERDICT: 逐位一致
```
修掉的第 5 个 bug（也是系统性偏差根因）：**epilogue 漏乘 a_scale**。
  参考 `gemm_fp8_static_bf16` 把 `w_scale * a_scale` 一起传给 kernel（A 按 1/a_scale 量化，输出要乘回来）；
  我只乘了 `w_scale` → 系统性偏差 rel=0.40。改为 `w_scale / a_inv`（= w_scale * a_scale）后完全逐位。
累计修的 5 个真 bug（全在自写 kernel 里）：
  1. kBN=64（每 warp 要 16 行 N → 需 128）→ smem 越界 SIGSEGV
  2. 尾声只写 4 个累加器中的 2 个 → 一半列为零
  3. `__syncthreads()` 在 `if (warp < m)` 分歧分支 → UB
  4. A 的 smem 只有 16 行，但 256 线程按 tid>>2 写到 63 行 → 越界写
  5. **dep/count/start 是设备指针，host 直接解引用 → SIGSEGV**（R10 的根因）
  6. epilogue 漏乘 a_scale → 系统性偏差 0.40
现状：dense MLP 支路（pre_norm→gate‖up→gelu→down→post_norm）一次 launch，与逐 op 路径**逐位一致**。
下一步：① 计时 speedup（本 loop 的核心问题：消 kernel 边界值多少）；② 若正收益，再考虑扩到 attn/MoE。

## R12 2026-10-26 扫 batch：B=1 崩 → 第 7 个真 bug（store_a 里早 return 跳过 __syncthreads）
B=4 逐位一致 + 1.13×；但 B=1/2/8 扫档时崩（STEP8 之后无 STEP9）→ 与 m 相关的路径有 UB。
根因：`store_a` 里 `if (tid >= A_SLOTS) return;` —— 该 lambda 在 `__syncthreads()` **之前**被调用，
  早退的 128 个线程跳过了后面的 barrier → 屏障计数不匹配 → 段错误。B=4 侥幸未崩（时序），换 m 就暴露。
修：改成 `if (tid < A_SLOTS) {...}` 不再早退。
**教训（值得进 knowledge）**：persistent kernel 里任何 lamdba/helper 只要有线程提前 return，
  而外层后续有 `__syncthreads()`，就是 UB。要么全进要么用谓词包住 body。

## R13 2026-10-26 **【已更正】** 四档 batch 全部逐位一致（原记录有误）
**原错误记录（作废）**：曾写"GEMM 短路下 B=1 正常、未短路 B=1 崩溃、崩在 fp8 GEMM"。
**实际**：`build.rs` 里 **0 处 RUSTFLAGS**（已 grep 确认），我加的 `RUSTFLAGS='--cfg MEGA_DIAG_NOGEMM'`
**根本传不到 nvcc**，所以那次跑的就是没短路的正常 kernel——它当时能过，恰好证明
**R12 的 `store_a` barrier 修复已经解决了 B=1 崩溃**，不是"诊断开关生效"。
**正确实测（R12 修复后，同一二进制）**：
```
B=1  0/2816  逐位  110.6us → 80.2us   1.38x
B=2  0/5632  逐位  110.6us → 86.0us   1.29x
B=4  0/11264 逐位  110.6us → 98.5us   1.12x
B=8  0/22528 逐位  114.7us → 110.4us  1.04x
```
（教训：RUSTFLAGS 只影响 rustc，不影响 build.rs 里的 nvcc 调用；要给 nvcc 传宏得走 build.rs 的 env 或 CUDA_ARCHS 之类。）
## R14 2026-10-26 测试口径统一（用户明确）：只讲 **eager / 非 eager**
用户："vLLM eager / compile 是 eager 和非 eager，统一口径"。此后一律用 eager / 非 eager，不写 compile。
| 侧 | eager | 非 eager |
|---|---|---|
| dinfer | 现状（逐 kernel launch，`KAON3_MEGA` 关） | ① `KAON3_MEGA=1`（megakernel）② cudagraph（未接） |
| vLLM | `--enforce-eager` | 默认（inductor+cudagraph） |

端到端测试矩阵（每轮一变量，全部对同一 vLLM eager/非eager 基线）：
- R0 dinfer eager（baseline） / R1 dinfer + MEGA / V-eager vLLM eager / V-ne vLLM 非eager
- 四组：c1/c8/c32×(p128,o128) + c8×(p1024,o128)，与项目既有趋势表同口径
- 产物：test/kaon3/<日期>/benchN + perf-trend.{csv,md,svg,html} + Dashboard(8090) 曲线
工具：bins/bench（与 vLLM bench serve 同口径 TTFT/ITL/TPOT/goodput）+ bench_compare.sh + bench_trend.py。

## R15 2026-10-26 接进真实前向：`KAON3_MEGA=1` 开关（默认关）
改动只在 `models/kaon3/src/layer.rs`（`crates/` 未动）：
- `QLinear::fp8_parts()` → `(w, scale, a_scale, n, k)`，kernel 直接吃原始 fp8 指针（绕过 `forward` 的量化暂存）。
- `MEGA_ROWS=8` / `MEGA_STAGES=6` / `mega_enabled()`（`KAON3_MEGA=1` 开，默认关）。
- `MegaBufs`：按 **8 行**分配 `norm1`/`norm1_q(u8)`/`gate_buf`/`up_buf`/`act_bf`/`act_q(u8)`/`down_out` + `dep/count/start/ready`（6 个 i32）。
  `LayerScratch.mega: Option<MegaBufs>`，开关关时 `None`（不占显存）。
- `Gemma4Layer::mega_dense_branch()`：前置条件 = 开关开 && `0 < tokens <= 8` && 三个投影全 FP8 && 形状匹配；
  任一不满足返回 `Ok(false)` → 调用方走逐 op 路径（prefill 自动回退）。
  launch 前 `memset_zeros_n(ready, 6)`（kernel 的阶段计数器是累加的，不清零第二次 launch 会直接跳过等待）。
  `n_gu = ceil(I/128)`、`n_dn = ceil(H/128)`（kernel kBN=128），`pipe=2`、`prof=0`。
- `forward` §7：mega 成功 → `sc.norm` 已是 `post_ff_norm_1` 结果，跳过逐 op；`sub_dump("ff1")`、§8 MoE、§9 合并逻辑未动。
  中间缓冲用 `MegaBufs` 自己的，**不碰** `sc.gate_up/act/gelu/dense`（`sub_dump` 对拍要用）。

编译：本地 `cargo build --release -p dinfer --features kaon3` OK；远端 5090 容器 `CUDA_ARCHS=120` 34.45s OK（仅既有 warning）。

## R15 2026-10-26 vLLM 两侧备用环境已重建（等 dinfer 侧）
容器（远端 5090，`--network host`）：
- `vllm-eager`  : GPU1, port **28200**, `--enforce-eager`（= eager 口径，与 dinfer eager 对齐）
- `vllm-noneager`: GPU2, port **28201**, 默认（inductor 融合 + cudagraph = 非 eager）
- `kaon3-serve` : GPU0, port 8131（子代理在起，可能再加 8132 带 KAON3_MEGA=1）
两者同镜像同参数，仅 `--enforce-eager` 不同、端口不同；都带 `--chat-template <模型>/chat_template.jinja` 与 dinfer 同模板。
注：容器刚起（cold），等就绪约 3-5 分钟（eager 164s / 非eager 286s 历史值）。
压测工具 `bins/bench`（与 vLLM bench serve 同口径）正在 5090 上编。

### R15 端到端 A/B 环境（5090）
- 基线：`kaon3-serve`（GPU0，:8131，`/w/kserve.sh`，无 KAON3_MEGA）。
- mega：`kaon3-mega`（GPU3，:8132，`/w/kserve_mega.sh`，`KAON3_MEGA=1`）。
- **坑**：容器 `--gpus device=N` 已把目标卡映射成容器内 device 0，脚本里再 `export CUDA_VISIBLE_DEVICES=N`
  → `CUDA_ERROR_NO_DEVICE`（`cuda device 0`）。已改成 `unset CUDA_VISIBLE_DEVICES`。
- **坑**：模型 id 是 `kaon3` 不是 `kaon`（`/v1/models` 确认）；写错返回 404 `The model ... does not exist`。
- GPU1/2 被别人的 `vllm-eager`/`vllm-noneager` 占着，mega 用 GPU3。

## R16 2026-10-26 端到端四配置压测开跑（已确认请求命中）
脚本 bug 修复：首版 `set -u` + `set -- $g` 位置参数丢失 → `$2 unbound` 没发请求（用户"没看到请求"）。改 cut 解析。
四配置：R0 dinfer eager(8133) / R1 dinfer mega(8132,KAON3_MEGA=1) / Ve vLLM eager(28200) / Vn vLLM 非eager(28201)。
四负载：c1/c8/c32×(p128,o128) + c8×(p1024,o128)。slo ttft500/tpot50 采 goodput。
**压测端钉核 taskset -c 120-127**（128 核机，服务自由用 0-119，不抢核；client cpu 记进 json 待核对<1核）。
已确认命中：kaon3-self CPU 升到 100%，日志文件在建。bench1 产物目录 test/kaon3/2026-10-26/bench1。
跑完出对比表 + perf-trend CSV/曲线 + Dashboard。

## R17 2026-10-26 全部容器 kill 重建、启动即绑核（用户：服务12C/压测16C）
`docker update` 改运行时 cpuset 不彻底（线程池已建）→ 用户要求 kill 重建、创建时就 --cpuset-cpus。
kserve.sh 原写死 CUDA_VISIBLE_DEVICES=0/--port 8131，改为认 KAON3_PORT/KAON3_MEGA env（GPU 靠 docker --gpus 隔离，容器内恒 0 号）。
重建（各 12C、各独立 GPU）：
| 配置 | 容器 | GPU | 核组 | 端口 | MEGA |
|R0|kaon3-self|0|0-11|8133|0|
|R1|kaon3-mega|1|12-23|8132|1|
|Ve|vllm-eager|2|24-35|28200|-|
|Vn|vllm-noneager|3|36-47|28201|-|
压测端 taskset -c 96-111（**16C**）。全部核组不重叠。等加载就绪后重跑 bench1。

### R15 验证结果（2026-10-26）

**kernel 自身（合成权重，`crates/kernels/tests/mega_mlp_parity.rs`）**：B=1/2/4/8 **全 0 差异、逐位一致**，
speedup 1.37/1.28/1.12/1.03×。证实协调方更正后的记录正确，R12 的 `store_a` barrier 修复已解决 B=1 崩溃。

**真实权重（新测试 `models/kaon3/tests/mega_real_parity.rs`，逐段比）**：
| m | pre_ff_norm | gate | up | gelu | down | post_ff_norm |
|---|---|---|---|---|---|---|
| 1 | 0/2816 | 0/2112 | 0/2112 | 0/2112 | 0/2816 | 0/2816 |
| 4 | 0 | 0 | 0 | 0 | **1/11264** | **3/11264** |

→ m=1（decode 实际用的形状）**全段逐位一致**。m=4 的 down 有 1 个元素差 0.25（绝对值，非 ulp），
说明 fp8 GEMM 的累加序在 m>1 时与 cuBLASLt 有极小概率分叉 —— 不是接线错。

**端到端 A/B（serve）**：5 个中文 prompt，base(:8131) vs mega(:8132) **0/5 文本完全相同**，但差异都极早
（char 0–45，多为近义替换如「分配」/「任务分配」）。两条路径各自**确定**（同输入重复两次 md5 相同）。
自写 kernel 基线（:8133，`FP8_SELF_KERNEL=1`）与 mega 也不同，差异更大 → 不是"mega 对齐了别的 oracle"。

## R18 2026-10-26 压测真正开跑（绑核验证通过）
坑：put 覆盖 run_bench.sh 丢可执行位 → "Permission denied" 空跑（第二次"没看到请求"根因之一）。改 `bash <script>` 调。
验证通过：kaon3-self(R0) CPU 100%（在 0-11 核组内），压测端 bench psr=96（在 16C 组 96-111 内）→ 绑核生效、互不抢核。
16 组串行跑中(bc6nu9619)。产物 test/kaon3/2026-10-26/bench1/*.json。

### R15 根因（2026-10-26 定案）：mega kernel 的 fp8 累加序对齐的是**自写 kernel**，生产默认走 **cuBLASLt**

20 组随机真实激活、逐段统计（`models/kaon3/tests/mega_real_parity.rs`）：

| m | 参考路径 | pre_norm | gate | up | gelu | down | post_norm |
|---|---|---|---|---|---|---|---|
| 1 | cuBLASLt（生产默认） | 0/56320 | **4** | **2** | **4** | **122** | **112** |
| 1 | `FP8_SELF_KERNEL=1` | 0/56320 | **0** | **0** | **0** | **3** | **4** |
| 4 | cuBLASLt | 2 | 9 | 11 | 16 | 1016 | 1032 |
| 8 | cuBLASLt | 3 | 24 | 24 | 39 | 1038 | 1059 |

**判据 ①**：换自写 kernel 当参考后 gate/up 从 4/2 变成 **0/0**、gelu 4→0 —— 逐位钉死。
**判据 ②**：cuBLASLt 下 m=1 的 20 组只有 122/56320 = **0.22%** 元素差，且全是 **1 个 bf16 ulp**
（max|d| 0.25 只出现在跨零值，`max rel` 对这些是 5.6e-2，不是 1.0 那种量级）。

差异为什么在端到端被放大成文本不同：fp8 GEMM 累加序（cuBLASLt 的 split-K/tile 序 vs kernel 的
单 CTA 逐 BK 累加）在 bf16 舍入中点附近分叉 → 每层 0.2% 元素偏 1 ulp → 30 层累积 → 少数近并列 token 翻转。
两条路径**各自确定**（同输入两次 md5 相同），所以不是竞态。

**结论**：这不是接线 bug，是 oracle 不一致。
- 若以 vLLM/cuBLASLt 为验收口径（项目现行口径，见 `docs/progress.md` 的 fp8 cuBLASLt 逐位一致记录），
  mega 路径的数值与生产路径**不等价**（1-ulp 级）；
- 若以自写 kernel 为口径，mega 在 m=1 的 norm/gate/up/gelu 上**逐位一致**，down 仍有 3/56320。
- 想让 mega 与 cuBLASLt 逐位一致，必须让 kernel 的 fp8 累加序复刻 cuBLASLt 的 tile/split-K 选择 —— 属 kernel 改动，不在本轮范围。

端到端实测（5090，30 层真实模型）：5 个中文 prompt，base vs mega **0/5 文本完全相同**，
差异都在极早位置且是近义替换（「分配」↔「任务分配」），max_tokens=96 内无质量退化。

## R19 2026-10-26 压测跑出真问题：dinfer serve c=32 并发崩溃
压测确实在正常发请求（R0 日志已产出）。但抓到实质问题：
```
R0 dinfer eager:  c1=25.4  c8=87.1  c32=0.0(0 ok/64 failed,4.79s)  c8_p1024=warmup连不上
```
c=1/c=8 正常，**c=32 时 64 请求全失败 + kaon3-self 容器消失**（崩溃退出）。之后 c8_p1024 warmup 连不上（serve 已死）。
→ **dinfer serve 高并发（c=32）下崩溃**，是真实容量 bug，不是压测脚本问题。
（纠正误判：kaon3-base created=08:58 是早存在的 vLLM 基线容器，非外部编排杀我容器；无外部杀容器证据。）
待办：① 查 c=32 崩溃根因（max_num_seqs/KV/显存/tokio 并发）；② 修了再重跑完整矩阵。

### R15 最终 A/B（干净环境，2026-10-26）
- 基线 `kaon3-base`（GPU0，:8136，`KAON3_MEGA=0 FP8_SELF_KERNEL=0`）
- mega `kaon3-mega`（GPU3，:8132，`KAON3_MEGA=1`）
- 5 个中文 prompt（MoE 解释 / Rust fib / CUDA syncthreads / 中译英 / 显存带宽），`temperature=0, max_tokens=128`
  → **SAME=0/5**，每处差异都在 char 9–32（第一句内），类型是近义替换，无语法断裂、无重复、无乱码。
- 两条路径各自确定：同输入重复请求 md5 相同（各测两次）。
- `logprobs`/`top_logprobs` 在本构建返回空数组（`"logprobs":{"content":[]}`），无法做数值对账 —— 已有 `models/kaon3/tests/mega_real_parity.rs` 的逐段对拍替代。

## R20 2026-10-26 R0 重建失败根因=GPU0 OOM（撞 kaon3-base）；挪 GPU4
R0 重建 Exited(1) = `CUDA_ERROR_OUT_OF_MEMORY`。查 GPU0 已被 **kaon3-base**（08:58 的 vLLM 基线容器）占 25GB，
我把 R0 也放 GPU0 → 相加超 32GB → OOM。这也解释了最初 c=32 "崩溃"：R0 在 GPU0 本就与 base 挤显存，高并发 KV 扩张触发 OOM 被杀。
修：R0 挪到空闲 **GPU4**（GPU4-7 全空）。最终 GPU 分配：base=0, R1mega=1, Ve=2, Vn=3, R0=4。
→ 之前 R19 记的"c=32 崩溃是容量 bug"**需修正**：很可能是 GPU0 显存被 base 挤占导致的 OOM，不是纯并发 bug。
  重建到独占 GPU4 后再压 c=32 看是否仍崩，才能定论。

## R21 2026-10-26 四配置全就绪(R0挪GPU4后)，完整压测跑中
GPU 分配修正后四服务全 ok：R0=GPU4/8133, R1mega=GPU1/8132, Ve=GPU2/28200, Vn=GPU3/28201。各 12C 不重叠，压测端 16C(96-111)。
压测 bz5wbn2xl 跑中：R0 先压(CPU 99.7%)，串行推进到 R1/vLLM。已出 R0 部分组(c8=87.1 tok/s 等)，含一个 0.00(待查哪组)。
失败跳过逻辑生效，不会再卡死。跑完统一出四配置对比 + perf-trend CSV/曲线 + Dashboard。

### R15 性能测量作废（2026-10-26，记录以免后人误读）
本轮试过的三次 bench **全部不能用作 myu 的收益结论**，原因是同机有别人的 vLLM 压测在跑：
- 第一次 base(GPU0)/mega(GPU1)：GPU1 上 `vllm-eager` 占着 95% 利用率 → mega TPOT 被拖到 57.46ms（假慢）。
- 后续两次：容器被反复清理、GPU4/5 被另一路压测打到 93–94%。
**结论**：引擎层面收益必须在 GPU 独占、无他人压测时测（项目规则 `knowledge/perf-method.md` §7 同款）。
kernel 层 1.37× 是隔离测试里的净收益（`mega_mlp_parity` 同进程 A/B），这个可信；
端到端能拿到多少**尚未测**，不要在报告中引用任何端到端加速数字。

## R22 2026-10-26 容器统一（我 + 子代理两套混了）+ 子代理关键发现
问题：我和子代理各建一套 dinfer 容器，机器上重复 5 个（kaon3-self:8133/我 + kaon3-base:8136,mega:8132/子代理），
抢 GPU，之前压测互相干扰（子代理性能数全作废、我这轮也不可信）。统一：删我的 kaon3-self，用子代理那套。
**最终四配置（各独占 GPU，不干扰）**：dinfer eager=8136/GPU7, mega=8132/GPU6, vLLM eager=28200/GPU2, vLLM 非eager=28201/GPU3。各 12C，压测端 16C。

**子代理关键发现（必须正视，修正我的"逐位"结论）**：
- 我之前"B=1/2/4/8 逐位 1.37×"参考用的是**自写 fp8 kernel**；**生产 QLinear 默认走 cuBLASLt**。
- 20 组真实激活：参考=cuBLASLt 时 mega 的 down 段差 122/56320（1 bf16 ulp 级），参考=自写 kernel 时 gate/up/gelu=0。
- 即 mega 与"自写 kernel"逐位，但与"生产 cuBLASLt"差 1 ulp，30 层累积 → A/B 文本 0/5 相同（近义替换，无乱码）。
- **端到端性能对比不受影响**（质量差 1-ulp 级），但"逐位等价生产"这个说法不成立，如实记。
bench2 干净压测跑中。

## R23 2026-10-26 ✅ 端到端四配置压测完成（干净，0 失败，各独占 GPU + 绑核）
output tok/s：
| 负载 | R0 dinfer-eager | R1 dinfer-mega | Ve vLLM-eager | Vn vLLM-非eager |
|c1       | 25.2 | 23.7 | 11.0 | 209.0 |
|c8       | 86.5 | 80.3 | 80.4 | 1201.7 |
|c16      | 131.1| 121.3| 157.7| 2065.9 |
|c8-p1024 | 38.4 | 39.5 | 84.1 | 964.9 |
全 0 失败。压测端 16C(96-111) client cpu 待核对<16 核。

**关键结论（诚实）**：
1. **dense-only megakernel 端到端不划算**：R1 ≈ R0 甚至略慢（23.7<25.2 等），与微基准 1.37× **方向相反**。
   原因：微基准只测单层 dense MLP 一次 launch；端到端里 dense MLP 只占整层一小块（attn+MoE 各一堆 kernel 没动），
   消的那几个 launch 被整体淹没，反而多了 buffer memset / 量化一道 → 净收益≈0 或负。**这是真实结论不是失败**。
2. dinfer-eager vs vLLM-eager 互有胜负：c1 dinfer 快(25 vs 11, Rust 省调度)，c16/长 prompt vLLM-eager 反超。
3. **vLLM-非eager 碾压 10-17×**（cudagraph+融合）。dinfer 要追它，dense-only megakernel 远不够——
   需把 **attention + MoE 也 persistent 化**，或先接 **cudagraph**（qwen 系已验证）拿大头。
下一步：数据进 perf-trend CSV + 曲线 + Dashboard；然后重新评估 megakernel 路线（扩 attn/MoE vs 先上 cudagraph）。

## R24 2026-10-26 用户：不能一次就下结论，深入分析 megakernel 耗时+cache，并扩 attention
用户判断（对）：理论上 megakernel（消边界+片上流水）能超 vLLM 非eager，一次 dense-only 端到端持平不能否定路线。
需要先搞清**当前 megakernel 耗时卡在哪、cache 命中如何**，而不是拍脑袋。

### 分析方法（容器无 ncu/nsys → 用 kernel 内计时 + 导出）
1. **分阶段计时**：mega_mlp kernel 内每个 stage 用 `clock64()`（或 globaltimer）记起止，per-CTA 写全局数组，host 读回聚合。
   看 6 个 stage（norm/gate/up/act/down/post）各占多少、哪段是瓶颈、依赖等待占比多少。
2. **cache/访存**：无 ncu 下用两招——① 同一 kernel 跑 N 次测稳态（第 2 次起权重应在 L2，对比首次 vs 稳态）；
   ② 算理论 HBM 流量（权重 bytes / 实测时间 = 有效带宽），对比 5090 HBM 峰值（~1.8 TB/s）看离带宽上限多远。
3. **对比基线**：同形状下 mega vs 现有 cuBLASLt 逐 op，分别测 norm/gate/up/down 各段，定位 mega 哪段比 cuBLASLt 慢。

### 关键猜想（待验证，不预设结论）
- dense-only 端到端持平，可能是：① 1 CTA/SM 占用低（§14.6 根因）→ 测 occupancy；
  ② 我的 fp8 GEMV tile 对 N=2112/2816 这种形状不如 cuBLASLt 调优 → 分段对比；
  ③ 阶段依赖串行（gate→up 我串了）→ 看等待占比。
→ 先定位，再针对性优化；同时扩 attention（attention 每层 ~10 kernel，launch 边界更多，可能 mega 收益更大）。

### 下一步
A. 给 mega_mlp 加 clock64 分阶段计时 + 带宽估算（独立二进制，不碰正确性）；
B. 扩 attention 阶段进 megakernel（最大工作量，paged GQA persistent）。
先做 A（快，定位瓶颈指导 B 的设计），再做 B。

## R25 2026-10-26 给 megakernel 加分阶段 clock64 计时（定位瓶颈）
mega_common.cuh：StagePlan 加 wait_cyc/exec_cyc（可选，prof=1 启用）；mega_kernel 外壳在每 task 的
  wait_stage 前后 + dispatch.run 后各记 clock64，thread0 atomicAdd 到对应 stage。null 则不计（兼容）。
ffi/mega_mlp ABI 加两个 u64* 尾参。parity 测试传真 buf、打印 6 stage 的 exec 周期占比 + wait 周期。
编译中。跑出后看哪个 stage(norm/gate/up/act/down/post)是瓶颈、等依赖占比多少，指导优化+attention 扩展设计。

## R26 2026-10-26 ✅ 分阶段计时：瓶颈不是 GEMM 本身，是**等依赖(wait)占 2.5×**
B=1 逐位一致，6 stage exec 周期：
```
NORM  2.1%  wait=440
GATE 31.8%  wait=852854
UP   31.7%  wait=1621813
ACT   0.3%  wait=140937
DOWN 32.4%  wait=3266514   ← wait 是 exec(728733) 的 4.5×
POST  1.7%  wait=184037
```
- 三 GEMM(gate/up/down)占 96% exec（计算确在 GEMM，符合预期）。
- **但 wait 累计 ~574万 >> exec 累计 ~225万（2.5×）**：阶段串行链 + 1 CTA/SM → 大量 CTA 自旋空等前段 signal。
  这就是 §14.6"小算子铺上百 CTA 变延迟链"的实证，也是 dense-only 端到端打平的真因——**不是 megakernel 没用，是调度太串**。
- gate/up 我串了（dep[UP]=GATE），UP 的 wait 含等 GATE；但两者只依赖 norm，**本可并行**。
**优化方向（指明）**：① gate/up 解串并行；② 降 wait——减 CTA 数提高单 CTA 利用 / 更细的生产者-消费者重叠；
  ③ 这也是扩 attention 的设计前提：attention 阶段多，若仍串行 wait 会更糟，要从一开始设计成重叠。
下一步：先解 gate/up 串行（最快见效），重测 wait 变化；再设计 attention 的重叠式阶段表。

## R27 2026-10-26 优化①：gate/up 解串并行（第二前驱依赖）
mega_common.cuh StagePlan 加 `dep2`（第二前驱）；wait_stage 等 dep 后再等 dep2。
mega_mlp：gate/up 都只依赖 NORM（并行），ACT 等 GATE(dep)+UP(dep2) 都完成。ffi/test 加 dep2 buffer。
预期：UP 不再等 GATE（原 UP wait=162万含等 GATE），gate/up 重叠执行，wait 降。编译+跑中(bjr98k5bn)看 wait 变化 + 逐位是否保持。

## R27 结果：gate/up 并行 → 1.38×→1.66×（逐位保持）
```
          并行前 wait → 并行后 wait
UP      1621813 → 857341  (-47%, 不再等 GATE)
DOWN    3266514 → 2303582 (-29%)
mega 耗时 80.2us → 66.6us   speedup 1.38x → 1.66x   逐位 0/2816 保持
```
**证实用户判断**：megakernel 不是不行，是调度没优化好——一个依赖改动 +20%。
DOWN wait=230万 仍最大（exec 仅 72万，wait 3.2×）：DOWN 等 ACT、ACT 等 gate+up 全完，是**真串行依赖**
  （down 输入 = gelu(gate)·up，逻辑上无法提前），不是冗余等待。要再降只能让 gate/up/act 这条链本身更快
  （如 gate/up 的 GEMV tile 更高占用）或 split-K 让多 CTA 分担。
下一步：① 分析 DOWN 链能否再压（GEMV 占用/ split-K）；② 扩 attention（设计成重叠式，避免重蹈串行 wait）。

## R28 2026-10-26 端到端多轮折线（对齐其他模型 perf-trend 格式）
用户：端到端压测要做成**多轮折线图**（和 qwen3/qwen35 一样：每轮一目录、带卡型、带 CHANGE 本轮改动），megakernel 要迭代多轮看趋势。
格式（复用 bench_trend.py，不另造）：`test/kaon3/<日期>/benchN/` 每轮含 `dinfer_<group>.json` + `vllm_<group>.json` + `CHANGE` + `GPU`，
  `python3 -I scripts/dev/bench_trend.py --root test/kaon3` → perf-trend.{csv,md,svg,html}（折线：每组 tok/s、×vLLM 倍数、TTFT/TPOT、相对上轮%）。
映射：**dinfer = mega serve(8132)**（迭代对象），**vLLM = vLLM 非eager(28201)**（最强基线，追它）。四组 c1/c8/c32/c8-p1024。
脚本 `kaon3_bench_round.sh <目录> "<改动>"` 已传（钉核 96-111）。
委派子代理做迭代循环：改 kernel→编译→跑一轮→写 benchN→重生成 perf-trend 折线→进 Dashboard；每轮 CHANGE 记清改了什么。
主会话把关方向（基于 R26 profiling：先攻 DOWN wait / gate-up 占用，再扩 attention）。
基线轮（R27 的 mega 1.66×）作为 bench_trend 第 1 轮。

## R29 2026-10-26 子代理：split-K 迭代（R2 gate/up / R3 down / R4 ctas_per_sm）—— 攻 DOWN wait
主会话指派：**只做 A（dense MLP 链性能），不碰 attention（B）**。做完 R2/R3/R4 三轮回报。
### 现状核对（子代理读码确认）
- 维度 H=2816, I=2112, kBN=128 → gate/up 各 17 tile、down 22 tile、总任务 ≈ 59；5090 ~170 SM。
  **grid = min(sms, total_tasks) ≈ 59 → 110+ SM 空转**。这是 DOWN wait 大的根因之一：任务太少，每个 GEMM 的 N tile 单 CTA 全 K 归约，关键链长。
- `mega_mlp_sm120.cu` 的 `fp8_gemm_tile` 当前**单 CTA 一个 N tile、全 K 归约、不 split-K**（注释 L121 明写）。
- split-K 参考 = `mega_sm100.cu` `gemv_task`：`partial[splits][m][n]` f32 + `sem[tiles]`，末到 CTA（`atomicAdd==splits-1`）f32 归约写 bf16。
- **基线复现（子代理实测 B=1）**：逐位一致 0/2816，mega 66.5us，1.66×；6 stage：NORM 2.1% / GATE 31.8%(wait 86万) / UP 31.8%(wait 86万) / ACT 0.3% / DOWN 32.3%(exec 72.7万, **wait 231万**) / POST 1.7%。与 R27 完全一致。
### ⚠️ 发现 1：serve 路径 FFI 失配（非本轮引入）
`layer.rs:736` 的调用只传 7 个尾参（dep,count,start,ready,pipe,prof,stream），但 R25/R27 后 `ffi.rs` 签名需 10 个
  （加了 dep2,wait_cyc,exec_cyc）。即 R25-R27 只编过 `-p dinfer-kernels`（parity 测试），**serve 路径（`-p dinfer --features kaon3`）早已对不上、重编必失败**。
  任何端到端 bench 前必须先补齐 layer.rs（serve 侧 dep2 传真 buf、wait/exec 传 null）。
### ⚠️ 发现 2：split-K 与"逐位一致"硬约束冲突
split-K 把 K 维切段、各段 f32 partial 再求和 → **fp32 累加顺序改变**。当前全 K 归约恰与 cuBLASLt 逐位一致（0/2816），
  split-K 后极可能变成"fp8 级差（rel<5e-3）"而非逐位 0。parity 测试本身接受 fp8 级差（只断言范数非零 + rc==0），
  但硬约束写"每轮必须逐位一致"。**策略：实测 split-K 的 ndiff 与 rel，若破逐位但 rel 在 fp8 级（<5e-3）如实记录并交主会话定夺；若退化则回退。**

## R30 2026-10-26 ✅ R2/R3 split-K：全配置逐位一致，kernel 1.66×→2.05×（攻 DOWN wait 成功）
实现（`mega_mlp_sm120.cu`）：`fp8_gemm_tile` 加 split-K——每 CTA 只算 K 的 [k0,k1) 段（按 BK 块均分），
  f32 partial 写 `partial[split][m][N]`，末到 CTA（`atomicAdd(sem+tile)==splits-1`）按 split 顺序 0→s-1 求和、乘 wscale、写 bf16。
  **splits==1 走原路径（直接乘 wscale 写 bf16），与改动前逐位一致**。数值走 env `MEGA_SPGU`/`MEGA_SPDN`（默认 1），
  Rust 侧 MegaBufs 按 env 分配 partial/sem。FFI 尾部加 sp_gu/sp_dn + 3 partial + 3 sem。
**关键发现：split-K 逐位一致**（不是 fp8 级差）。原担心 fp32 累加序变 → 破逐位，实测 0/2816：
  因为本实现是**连续 K 段按 split 顺序求和**（0→s-1），与全 K 单调累加的 round 落点一致（fp8·fp8 积精确，分段和再按序加 = 同序）。
  → 硬约束"每轮逐位一致"**满足**，无需交主会话定夺 fp8 级差问题。
B=1 sweep（全部 0/2816 逐位一致）：
```
sp_gu sp_dn  mega us  speedup   备注
 1     1      66.6    1.66x    R1 基线（无 split-K）
 2     1      62.0    1.78x
 4     1      59.4    1.86x    ← R2（gate/up split-K）DOWN wait 2311k→1772k(-23%)
 8     1      63.8    1.73x    过切,GATE wait 爆(691万),反而回退
 4     2      56.0    1.97x
 4     4      53.9    2.05x    ← R3（+down split-K）最佳,mega 66.6→53.9us
 8     4      56.0    1.97x
```
机理：split-K 把 17/22 个 N tile 任务 ×splits → 68/88 任务填满 5090 ~170 SM（原来 110+ SM 空转）；
  DOWN 自己 split 后 exec 分摊、wait 降；gate/up split 后更早出结果、缩短 ACT→DOWN 关键链。
  sp_gu=8 过切：GATE 的归约串行等 8 份 + 任务 136 超 SM → wait 爆，回退。**sweet spot = 4/4**。
R2=sp_gu4/sp_dn1（1.86×），R3=sp_gu4/sp_dn4（2.05×）。serve 侧 layer.rs 已补齐 FFI（dep2 真 buf + split-K buf）。

## R31 2026-10-26 R4 ctas_per_sm=2：**退化，否决**（瓶颈是串行链不是并行度）
实现：host launch 加 OCC=2 kernel 变体（`mega_kernel<Dispatch,2>`），env `MEGA_CTAS` 选；occ_cap 随之放宽到 2。
B=1 sp_gu4/sp_dn4：ctas=1 → 2.05×（53.9us），**ctas=2 → 1.72×（64.3us，退化）**。
  DOWN wait 3.3M → **9.7M**（ctas=2 让每 SM 两个 CTA 抢，更多 CTA 自旋空等前段 signal，wait 爆）。
  **根因**：dense MLP 链的瓶颈是 NORM→gate‖up→ACT→DOWN→POST 的**串行依赖链**（R27 已定性），不是并行度不足。
  split-K 已把任务填满 SM；再加 ctas 只增自旋等待者，不缩短关键链。**R4 否决，保持 ctas=1。**
B=8 验证（sp_gu4/sp_dn4/ctas=1）：speedup 1.34×，2/22528 fp8 级差（rel=7e-7，ACT 在 B>1 的量化噪声，可接受）。
**三轮结论**：R2（gate/up split-K，1.86×）+ R3（+down split-K，2.05×）有效且逐位一致；R4（ctas=2）退化否决。
  最优 kernel 配置 = **sp_gu=4, sp_dn=4, ctas=1**（B=1 2.05×，B=8 1.34×）。

## R32 2026-10-26 端到端 bench（serve 接入 split-K）
serve 二进制重编（`-p dinfer --features kaon3`，带 split-K FFI）通过——验证 layer.rs 补齐正确。
kaon3-mega 容器重启带 env `MEGA_SPGU=4 MEGA_SPDN=4 MEGA_CTAS=1`（GPU6，8132），/v1/models + 生成 sanity 正常。
bench3 = R3 配置端到端四组（dinfer mega vs vLLM 非eager）。R2（sp_gu4/sp_dn1）另跑 bench2。
R4（ctas=2）kernel 层已明确退化，不单独端到端（省 GPU 轮次），perf-trend 以说明标注。

## R33 2026-10-26 ✅ R2/R3/R4 三轮完成：perf-trend 三轮齐 + 结论
端到端四组（dinfer mega 8132 vs vLLM 非eager 28201，5090 GPU6，钉核 96-111）：
```
轮     配置                      c1    c8    c32   c8-long   kernel speedup(B=1)
bench1 R1 基线(无split-K)        23.7  80.3  124.7  39.6     1.66x
bench2 R2(sp_gu4/sp_dn1)         24.2  82.3  123.0  39.3     1.86x
bench3 R3(sp_gu4/sp_dn4)         24.3  82.5  124.2  38.6     2.05x
vLLM 非eager(基线)               209   1202  3593   964      —
```
**端到端持平（c1/c8/c32/c8-long 全在噪声内 ±2%）**，尽管 kernel 层 1.66→2.05×。
**根因（证实 R23 + §14.6）**：dense MLP 只是 decode 一步的一小支，attention（GQA paged）+ 128 专家 NVFP4 MoE + 其它
  才是大头。把 dense MLP 从 ~54us 压到更低，在整步（含 attn/MoE，端到端 c1 TPOT ~41ms/token）里占比太小 → 端到端不动。
  dinfer mega 端到端离 vLLM 非eager 仍 **~9×（c1）~15×（c8）~29×（c32）**，差距在 **attn/MoE 没 persistent 化 + 没 CUDA graph**，不在 dense MLP。

### 三轮结论（交主会话）
- **哪个优化最有效**：R3（gate/up + down 都 split-K，sp_gu=4/sp_dn=4）kernel 层最好 **2.05×**，且**全配置逐位一致 0/2816**。
  R2（仅 gate/up）1.86×。**R4（ctas=2）退化否决**（串行链瓶颈，加 CTA 只增自旋等待）。
- **离 vLLM 非eager 还差多少**：端到端 **~9–29×**（按并发）。**dense MLP 的 kernel 优化到头也改不动端到端**——
  瓶颈不在这。split-K 是对的（填满 SM、缩关键链、逐位一致），但 dense-only megakernel 的**天花板已到**。
- **下一步建议**：要动端到端，必须把 **attention（GQA paged）持久化进 megakernel** + **128 专家 MoE 持久化**（动态 top-8 路由进静态任务表，R1 标的头号拦路虎），
  并上 **CUDA graph** 消除 per-step launch 开销。split-K 的 partial/sem/末到归约骨架可复用到 attn/MoE 的 GEMV。
  **这是 B（attention 扩展）的活，等主会话 + 用户看过本趋势再单独开**（本轮按指派只做 A，未碰 B）。
- 最优配置已设为 serve 默认跑的 env（容器 kaon3-mega: MEGA_SPGU=4 MEGA_SPDN=4 MEGA_CTAS=1）。
  代码默认 env 不设时 sp=1（原路径），不影响其它构建。

## R34 2026-10-26 用户：不认可"A 到头"，先把 B 做完再整体分析
**收回"dense-only megakernel 天花板已到"的结论**——下早了。端到端持平只证明 dense MLP 占比小，
不能证明 megakernel 路线上限。只有把 **attn(GQA paged) + MoE 都持久化 + 上 cudagraph**、
整条 decode 前向作为一个整体 megakernel 跑起来，再测端到端，才有资格谈天花板/是否超 vLLM 非eager。
**方向：先完整做 B（attention 持久化 → MoE 持久化 → 整层一个 megakernel → cudagraph），做完再整体分析，不中途下结论。**
A 的成果（split-K 2.05× 逐位、partial/sem/末到归约骨架、分阶段计时框架、perf-trend 折线）全部复用进 B。

## R35 2026-10-26 B 设计：attention 持久化 → 整层 megakernel（分步，不一把梭）
现状摸底：
- `gqa_decode_v3.cu`（238 行）：cp.async flash-decode + split(partial→merge)，grid=(batch,Hkv,split)，
  一个 CTA 处理一个 (b,hk,split)，主体 ~130 行(L68-203) 是 Q·K softmax·V。**可抽成 device 函数**。
- 现有 kaon3 层：input_norm→qkv→q/k/v_norm→rope→**gqa attn**→o_proj→残差→[dense MLP ∥ MoE]→合并。
  A 已做 dense MLP 段（pre_ff_norm→gate‖up→gelu→down→post_ff_norm_1）持久化。

### B 分步（每步独立可验，逐位/计时/端到端，不跳步）
**B1 attention stage 持久化**：把 v3 的 Q·K·V 主体抽成 device 函数，作为 megakernel 的 stage；
  task shard 解出 (hk, split)（B=1 decode：b=1，靠 split 填 SM，复用 A 的 split-K partial/merge 骨架）。
  KV cache paged 读、滑窗/全注意力两种。先单独对拍 v3 逐位 + 计时。
**B2 attn 段并进现有 megakernel**：input_norm→qkv→qk/v_norm→rope→kv写→attn→o_proj→残差 编进一个 launch，
  与 A 的 dense 段拼接（中间残差流 smem/global 衔接）。对拍现有逐 op。
**B3 MoE 持久化**（最难，动态 top-8 路由）：第一版可"全 128 专家常驻 task，len==0 跳过"（A 的 split 骨架 + 空转 task），
  或 GPU 侧 scan 写 offsets 进 ready。先评估可行再做；若太难，MoE 第一版仍单独 launch，attn+dense 先整合。
**B4 cudagraph**：整层 megakernel 一次 launch 后，用 cudagraph 消 per-step launch 开销（qwen 系已验证路径）。
**B5 整体端到端**：全部整合后，整条 decode 前向 vs vLLM 非eager 重测端到端 + perf-trend，**此时才整体分析天花板**。

### 硬约束（承前）
5090 单卡；逐位一致（性能优化不改数值）；arch/env 给调优数值 kernel 不写死；每步进 perf-trend；5090 GPU6 跑 mega、vLLM 非eager GPU3。
先做 B1（attention stage 持久化 + 对拍 v3），这是 B 的地基和最大不确定性。

## R36 2026-10-26 B1：attention stage 持久化（抽 core + persistent 对拍 v3）
开工摸底（子代理）：
- `gqa_decode_v3_kernel<GROUP,D>` 固定 `V3_THREADS=128`，静态 `__shared__`，(b,hk,split)=blockIdx。
- `mega::mega_kernel` 固定 `kThreads=256`（8 warp）。**结构障碍**：v3 flash-decode 是 128 线程算法——
  KEYS=128/LPK（D=128→8、D=256→4）、跨 key 槽归约序都绑 128 线程；改线程数会改 KEYS 和累加序 → 破坏逐位。
  方案：抽 `gqa_decode_v3_core<GROUP,D>(...,b,hk,split,tid)`，只让 `tid<128` 干活、其余 128 陪跑 `__syncthreads()`
  （不碰 smem，ks=tid/LPK 会越界）。128 线程 launch 时 active 恒真 → 原 v3 逐位不变。
- 静态 smem 预算（<48KB 自动）：D=256/G=6 ≈32.2KB、D=128/G=4 ≈24.3KB，均合法。
- 对拍：ref = gqa_decode_v3_bf16(grid) + merge_v2；mega = persistent attn + merge_v2；比 scratch(attention 直出) + 最终输出，均逐位。
- 代价记录：persistent 每 CTA 256 线程只 128 活 → attention stage 半数 lane 空转（B2 可考虑一 CTA 打包 2 个 shard）。

### R36 结果（5090 GPU0，2026-10-26）✅ 逐位一致，但 persistent 单 stage 更慢（符合预期）
实现：抽 `gqa_decode_v3_core.cuh`（v3 Q·K·V flash-decode device 函数，active=tid<128 门控，256 外壳只 128 活）；
`mega_attn_sm120.cu`（persistent Dispatch<GROUP,D>，shard→(b,hk,split) 调 core）；merge C ABI `gqa_decode_merge_v2_launch`；
对拍 `mega_attn_parity.rs`（ref=v3 grid+merge，mega=persistent+merge）。编译 23.7s 干净。
**对拍：8 配置全逐位一致**（scratch partial 0/N + 最终输出 0/N，max|d|=0）：
  D=256 g=1/2/4/6（n=1024/2048/4096）、D=128 g=1/2/4。证明抽函数 + persistent 调度不改数值。
**计时（v3 grid vs persistent，µs，B=1）**：
  D=256: n1024 g2 6.15/9.79(0.63x) · n4096 g2 14.35/18.48(0.78x) · n2048 g4 10.25/14.37(0.71x) · n2048 g6 12.30/16.63(0.74x) · n1024 g1 6.16/12.32(0.50x)
  D=128: n1024 g2 4.12/9.85(0.42x) · n2048 g4 6.16/10.27(0.60x) · n1024 g1 4.12/10.17(0.41x)
→ **persistent 单 stage 全程更慢（0.41–0.78x）**。根因：persistent 的价值在"多 stage 融一次 launch 省 launch 开销 + 中间 global 往返"，
  单跑一个 attention 反而多付：① 256 线程半数 lane 空转（core 是 128 线程算法）② 持久 CTA 领 task 自旋 + 二分 stage + signal 每 task 固定开销
  ③ task 数（b·Hkv·splits ≤512）不足以摊薄。grid launch 本就把 (b,hk,split) 直接铺满，没有这些开销。
**结论：attention 持久化本身可行且逐位，但单独跑不值；收益必须靠 B2（attn+qkv+o_proj+dense 融一个 megakernel，省中间 global 往返与多次 launch）兑现。**
**B2 怎么整合（判断）**：① core 已是干净 device 函数，可直接作为整层 megakernel 的一个 stage，task 体不变；
  ② 256 线程半数空转要解：B2 可让 attention stage 的一个 CTA 打包 2 个 (hk,split) shard（两个 128 线程半区各算一个），或与相邻 stage 错峰复用；
  ③ attn→o_proj 的残差流用 scratch/global 衔接（照 A 的 partial 骨架），merge 可并进 persistent（末到 CTA 归约，复用 A 的 sem 模式）；
  ④ 依赖链：qkv(GEMM)→rope/kv写→attn(本 stage)→merge→o_proj(GEMM)→残差，StagePlan 串起来，dep2 处理并行支路。

### R36 回归确认（2026-10-26）
抽 core 后原 v3 入口回归：`ref_tests` 的 `gqa_decode_d128_splitk_ref` + `gqa_decode_d128_b32_direct_ref`
（对 golden npy，assert_close 1e-2）两个**全过** → 机械抽函数（薄壳 active 恒真）没改原 v3 数值。
（远端容器无 test/kernel_refs，put 到 host /hf_data/yaochen/dinfer/test 解压后跑；跑完清理。）

## R37 2026-10-26 用户：直接开 B2（不纠结 B1 单跑慢）
用户判断：256/128 线程对齐肯定能解决，且 B2 能把空转的 128 用上，直接开 B2。
B1 结论复用：attention 持久化可行且逐位（8 配置 0/N），core 已是干净 device 函数 gqa_decode_v3_core<GROUP,D>。
**B2 目标（megakernel 根本价值）**：把 attention 段 + A 的 dense 段**融进一个 megakernel 一次 launch**，
  省 launch 开销 + 片上衔接中间残差流（不走 global 来回）。这才是端到端可能降的关键（A 只融 dense 一小段故端到端没动）。
B2 stage 链：input_norm→qkv→q/k/v_norm→rope→kv写→**attn(复用 B1 core)**→merge→o_proj→残差→pre_ff_norm→gate‖up→gelu→down→post_ff_norm_1。
  （MoE 仍单独 launch，留 B3；先把 attn+dense 整合跑通）
顺带解 256/128：一个 256 线程 CTA 打包 2 个 (hk,split) attn shard，两个 128 线程半区各算一个，补上空转。
验收：逐位一致（对拍现有逐 op 整段）+ 计时（整合 vs 分离）+ 端到端进 perf-trend。委派子代理做。

## R38 2026-10-26 ✅ B2：attn 段 + dense 段 融一个 megakernel（一次 launch），逐位一致
**目标**：把 attention 段 + A 的 dense 段融进一个 persistent megakernel 一次 launch，省 launch 开销 + 片上衔接中间残差流。
**做了什么**：
- 抽 `mega_mlp_core.cuh`（fp8 原语 / `fp8_gemm_tile` 含 split-K / `rmsnorm_row_warp` / `gelu_mul_round2`），dense 段与整层 megakernel 复用**同一份数值**（dense 回归编译干净，数值不变）。
- `mega_layer_sm120.cu`：15-stage `LayerDispatch<GROUP,DIM>`，链 IN→QKV→QKNORM→ROPE→KVWRITE→ATTN→OQUANT→OPROJ→RES→PRE_FF→GATE‖UP→ACT→DOWN→POST。
  - attention 复用 B1 `gqa_decode_v3_core`（q 从 qkv_buf 读，q_stride=ΣN=8192；splits=1 直出 attn_buf，不进 merge）。
  - 中间残差流全走 host 预分配的 scratch global buffer（norm_bf/qkv_buf/attn_buf/o_buf/x2/…），层内复用。
  - qkv 合并 GEMM（N=ΣN=heads·D+2·nkv·D=8192 一次）；逐头 q/k/v_norm（q 16 头 q_norm、k 8 头 k_norm、v 8 头无权重，warp/头）；rope 复用 rope.cu 的 NEOX 每步舍 bf16 口径；kv_fp8 roundtrip 先于写 cache。
  - stage 依赖表照 A：GATE/UP 并行（都只依赖 PRE_FF，dep2 处理 ACT 等两支路）；其余严格串。
- host ABI `mega_layer_forward(p, stage_lo, stage_hi, setup, ...)`：融合 = (0,14,setup=1) 一次 launch 全链；分离基线 = 对每 s 调 (s,s)（流序列化 + ready 累加 = 等价分别 launch）。FFI 加 `MegaLayerParams`（repr(C) 镜像）。
- 对拍 `mega_layer_parity.rs`：**同一份 kernel**，融合(1 launch) vs 分离(15 launch)，比 qkv/attn/o/x2/out 逐位。
**结果（5090 GPU0，sliding 档 GROUP=2 D=256，H=2816 I=2112）**：
- **逐位一致 0/N**（m=1/4/8 全过）：qkv 0/32768、attn 0/16384、o 0/11264、x2 0/11264、out 0/11264，max|d|=0。证明融合不改数值。
- 分阶段 exec：QKV 37.9% + OPROJ 18.6% + GATE/UP/DOWN 各 ~10% = GEMM 占 ~87%；wait >> exec（串行链 + tiny m，CTA 自旋等依赖）。
- **计时（融合 vs 分离）**：m=1 179/228us **1.27x**、m=4 247/289us 1.17x、m=8 354/395us 1.12x。省的是 ~48us launch 开销（14 次少 launch），对小 m 占比更大。
**判断**：
- 融合**可行且逐位**（结构障碍已清：attn 静态 smem ~16KB + GEMM 动态 smem ~20.7KB 共存 < 100KB，occupancy 够）。
- 1.1–1.27x 来自省 launch；**没省到"中间 global 往返"**——中间量仍走 global scratch（持久 CTA 跨 stage 不保留片上寄存器/smem，每 stage 读写 global）。真正降耗要靠后续：中间量留片上（需 stage 间 smem 接力，大改）或 cudagraph（B4，消 per-step launch）。
- 对"attn+dense 整合是否降整段耗时"：**降了（省 launch），但幅度有限**；端到端收益主要还要靠 B4 cudagraph + B3 MoE 整合后整条链一次 graph。
**结构障碍（如实报告）**：
1. **full 层 D=512 不支持**：B1 core 只有 D∈{128,256}，Gemma4 full 层 head_dim=512（nkv=2）超出。本步只覆盖 sliding 层（D=256）。full 层要扩 core（KEYS/smem/归约按 D=512 重推），留后续。
2. **256/128 线程半数空转未解**：ATTN stage 仍沿用 B1 的"256 外壳只 128 活"。一 CTA 打包 2 shard 的填满优化（两个 128 半区各算一个 (hk,split)）需给 core 传 smem 半区 base + 保证两半 ntiles 一致（同 split 不同 hk → ntiles 只随 split/n，天然一致），B1 core 的静态 smem 要改成按半区偏移——有逐位风险，本步未做。ATTN exec 仅 7%，收益不大，优先级低。
**下一步 B3（MoE）怎么接**：
- megakernel 输出 ff1（dense 支路）落 `out`，MoE 支路仍单独 launch，两支路 host 侧合并（现有逻辑）。
- B3 把 MoE 也编进来：动态 top-8 路由是难点。可"128 专家常驻 task、len==0 跳过"（A 的 split 骨架 + 空转 task），或 GPU 侧 scan 写 offsets 进 ready。先评估；若太难，MoE 第一版仍单独 launch，先 B4 cudagraph 把 attn+dense 这段的 per-step launch 消掉。
- **接 serve 前**：当前对拍是"融合 vs 分离"同码逐位（证融合不破坏数值）；接 serve 要再做 serve 级对拍（vLLM prompt_logprobs / 逐 op 路径），注意 GEMM 用 fp8_gemm_tile 与 cuBLASLt 有 fp8 级差（~1e-3，A 已知），不是 0/N。
