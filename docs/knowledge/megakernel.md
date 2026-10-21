# megakernel（单层 persistent kernel）的结构事实与实测

> 从 `kernel-rules.md` 拆出（2026-10-21）。通用规则见 [kernel-rules.md](kernel-rules.md)。

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

