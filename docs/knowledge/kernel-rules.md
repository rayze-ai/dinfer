# kernel 规则与已验证的结构选择

## 硬规则（CLAUDE.md，用户 2026-10-19）
- kernel 只收运行时参数；splits / BV / tile / warp 数等由 `ArchProfile`（`crates/kernels/src/arch.rs`，按 cc + SM 数）与启动自调优给，环境变量只做覆盖。
- 多卡型实现按 arch **分文件**（`*_sm8x.cu` / `*_sm100.cu`），kernel 体内禁 `#if __CUDA_ARCH__`；有卡型差异才分，没差异一份。
- 预分配缓冲上的算子**显式传 token 数**（`forward_rows(…, t)` / `tokens` 参数），禁止从缓冲长度推（两次踩坑：GEMM M=2112、GDN decode 分发失效）。
- 新加 C ABI 符号：声明（`kernels.h` / `ffi.rs` 含 no-nvcc stub）与定义**同一个 commit**（并行子代理的 commit 曾带走别人的声明，HEAD 链接失败）。

## 已验证的结构（B300 sm_103 实测，3090 多为带宽下限看不出差别）
| 场景 | 选择 | 数字 | 证据 |
|---|---|---|---|
| 递归状态 decode（GDN/KDA，状态 [V,K] f32 每 (seq, 头)） | 寄存器驻留 warp kernel：grid (B, HV, V/BV)，K 维 warp shuffle；多值蝶式归约（`tr`）+ 一遍 FMA；BV 由 SM 数定（Sm100 ≤ 32） | B300 B=32 431 → 32 µs/层（下限 25） | loop 4 / 7a / 7c |
| cp.async 流水预取状态 | **不赢**（占用 2 → 1 block/SM 抵消） | pipe 35.4 vs tr 34.1 | loop 7c |
| chunk prefill（delta rule） | prep / wy / ho 三段，ho 按 (seq, head, V 块) 并行 + mma.sync bf16；conv1d 按 T 分块（含 bos 的块独占 conv_state） | ho 1346 → 204 µs、conv 874 → 76、wy 652 → 221 | loop 5A / 5E |
| 小 M GEMM（M ≤ 8） | cuBLASLt 前 N 候选逐个计时 + 自写 split-K 取优，表键 (M 桶, N, K, bias)，非桶 M 借幂桶（`AlgoCheck`）；默认 algo 按键缓存（否则每次 forward 查启发式） | M=1 大 N：lt 默认 81 → 16 µs（splitK 5） | loop 3 / 5B |
| M ≥ 16 小 M kernel | fp32 FMA 成算力型，不如 lt；留作候选由自调优定 | m32 N=6144 355 vs lt 97 | loop 9 疑点 2 |
| 同输入多投影 | 权重按行拼成一条 GEMM，下游按行距读不拷贝 | 每步少 160 launch，c32 TPOT 13.2 → 12.5 | loop 9 |
| rmsnorm h ∈ (1024, 8192] | 1024 线程 × 1 向量（单次访存等待） | −4–8% | loop 7b / 7c |
| decode GEMM 效率上限 | HBM 72–76%（`gate_up` 58 µs = 356 MB），与 vLLM 同档，不再是差距来源 | — | loop 5G |

## 数值口径
- 硬判决（top-k / 路由 / argmax）只在可判定行断言（ref top-2 gap > 阈值）；bf16 输出链 rel ~5e-3 = 一 ulp，用 cos ≥ 0.999x 而不是 abs。
- 参考侧 rmsnorm 的 bf16 落点要与 kernel 一致（乘权重**后**落 bf16）。
- 运行时下标的寄存器数组会落 local memory（stack frame ≠ spill）：模板递归展开。

## megakernel 第一阶段实测（`docs/loops/mega-sm100.md`，2026-10-20）
- **第 1 轮：persistent kernel 慢于 11 kernel + graph 基线**（B300 单 GDN 层 decode：146/158/190 µs → 188/215/339 µs，0.78/0.74/0.56×）。基线口径公平（现路径也走 graph + 自调优 GEMM）。
- 阶段 prof 显示的真因：① GEMV 阶段只到 4.5 TB/s（cuBLASLt 自调优的 splitK 算法 6+ TB/s）——persistent CTA 内手写 GEMV 的访存并行度不如专用 kernel；② GDN 递归阶段 B=32 80 µs（独立 kernel 32 µs）——任务粒度 / 占用被统一 grid 绑住；③ 依赖等待与小阶段（conv 16、add_norm 10 µs）叠加。
- **结论先记为假设**：消掉 kernel 边界省的 ~2–3 µs/边界 × 11，抵不过每个阶段失去"各自最优 launch 配置"的损失；megakernel 要赢必须每阶段都达到独立 kernel 的效率（OCC / 每阶段任务粒度分开调），第 2 轮验证。vLLM 做不到这条路不等于这条路一定赢。

## megakernel（单层 persistent kernel）实验学到的结构事实（2026-10-20，docs/loops/mega-sm100.md）
- **常驻 grid 的占用是第一约束**：GEMV 分支（G=4 组 × MB=2 × 4 累加器 + 片段）把寄存器推到 170–255 → 1 CTA/SM = 8 warp/SM，
  cp.async 流水本身够深（2–3 级 × 72 KB 在飞）但 B300 上只到 4.5 TB/s（基线 cuBLASLt 6.5）；要 2 CTA/SM 必须 `launch_bounds(…, 2)`
  把寄存器压到 128（会 spill ~270 B），且 GDN 分支的寄存器也算在同一上限里。证据：loop 第 1 轮 prof（in_proj busy 57 GB/s/SM）。
- **小任务在常驻 grid 上串行化**：独立 kernel 的 6144 个 GDN 小 block 靠高占用并行藏延迟；常驻 grid 只有 148 个 CTA，
  同样的任务粒度变成 41 波 × 单次状态读延迟 = 80 µs（独立 kernel 32）。解法：任务变粗（每 warp 16 行、任务 = (seq, 若干 hv)）+
  跨 tile cp.async 预取（smem 双缓冲，每 lane 只取自己要读的 16 B，warp 内无同步）。同理 conv 每线程多维必须先全读再算（15 → 0.5 µs）。
- **先发权重流、再等依赖**真能把阶段屏障藏掉一部分（prof Σwait 占 busy 的 10–30%），但 B=1 的"小阶段链"
  （conv → gdn → gated_norm → rmsnorm）每级 3–10 µs 串行，是 megakernel 相对 graph 基线的净损失来源；能融的融进 GEMV 尾声
  （add / silu，省两级），不能融的（gdn、gated_norm）只能靠粗任务 + 预取。
- **同一 kernel 内"读 → 被别的 SM 改 → 再读"的缓冲必须 `__ldcg`**（L1 不相干，kernel 边界才失效）：res、split-K partial、o/z。
  权重 / 一次写一次读的中间量用普通读即可。
- **split-K 的 tile 信号量与 ready 计数器由 kernel 自复位**（最后到的任务 / 最后退出的 CTA 清零），graph 重放不需要 host memset。
- cp.async `wait_group N`：第 i 轮顶部已提交 pipe-1+i 个 group，要 group i 完成须允许挂起 pipe-2 个（写成 pipe-1 时 pipe=2 等于不等，
  第一版全是垃圾但不报错——对拍是唯一的护栏）。
- **结论（2026-10-20，2 轮 / B300）：单层 megakernel 不成立**——最好 0.91 / 0.82 / 0.59×（B=1/8/32），三档 < 1.10×。
  原因按证据排序：(1) 常驻 grid 1 CTA/SM 的占用输给独立 GEMM kernel 的 4–8 block/SM（GEMV 5.1 vs 6.5 TB/s，B=32 mma 环 3.2）；
  (2) 小算子在 148 个 CTA 上变延迟链（gdn B=1 span 15 µs、busy 2）；(3) graph 基线的边界效应只有 ~17%（147 µs vs 下限 ~120），
  不够抵消 (1)(2)。依赖等待不是瓶颈（Σwait 10–30%）。**判断**：在"权重流下限 + graph"这个基线上，减 kernel 数的收益上限 ≈ 边界效应
  占比，27B 级模型 < 20%，不值一次架构重写；质变要靶向字节（NVFP4）或算子融合本身（K3 MoE 路径）。证据：`docs/loops/mega-sm100.md`。

## smem 瘦身优先于算法改动；寄存器预取在 255 上限附近反伤（r9，2026-10-20）

- **判断**：block 级 kernel 慢，先看 `--resource-usage` 的 smem / 寄存器，把 smem 压到让 occupancy 从 1 → 2 block/SM，往往比改算法收益大；寄存器已接近 255 时再加"预取下一块到寄存器"会 spill，反而慢 20–70%。
- **为什么**：r9 wy kernel 把 sM 复用 sK∪sQ（72 → 56 KB）占了 2.2× 提速的大半，分块三角逆只再省 17–20%；ho 的 PF=1 预取在 3090 / 占满寄存器时全形状退化。两段版 ho（h-pass + o-pass）在 3090 比融合慢 10–30%，因为中间 h16 一写一读 201 MB @ 8K 吃带宽——这类"拆 kernel 换并行度"的改动收益随卡的带宽变，门限必须进 `ArchProfile` 按卡实测，不能按一张卡的结论定死。
- **怎么用**：改 kernel 前 `nvcc --resource-usage` 记 smem / reg / occupancy；先做瘦身，再做算法；"预取 / 拆段"这类结构开关一律 `ArchProfile` 默认 + 环境变量覆盖，两卡各测后再定默认。
- **证据**：`docs/loops/qwen35-gdn-r9.md` 轮次 1–5；`test/qwen35/2026-10-20/probe_gdn_r9_*`；commits ba23769 / 515ef0c / e78ccfe。
