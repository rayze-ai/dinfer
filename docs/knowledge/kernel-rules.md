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

- **GDN chunk prefill（r10，B300 实测）**：wy 单 kernel（点积 → 分块三角逆 → u/w 两 GEMM）**8 warp + bf16 中间量 + 16 B 向量装载**，8×1024 532 → 360 µs（fla 三 kernel 合计 334）；ho（串行 chunk 链）**BV 按 `base_blocks·(V/BV)` 能否铺满 `sm × 2` 选 64 / 32**（block 多 64 赢 19%，单长序列 64 输 18%），498 → 308（fla 332）。ho 的 B 操作数 ldmatrix 化、svT bank 冲突都**不动时钟**——ho 是 ILP / 并行度型，不是 smem 型。单长序列（192 block < 148 SM × 2）仍差 fla 15%，要 chunk 级并行的 o kernel（r11）。
- **形状分叉不翻默认，进 ArchProfile 按形状选档**（r10 第 6–7 轮）：BV=64 在 8×1024 赢 19%、在 1×8192 输 18%，若按"平均"翻默认两头都亏；正确做法是找到分叉的**物理量**（这里是 `base_blocks·(V/BV)` 能否铺满 `sm × 常驻 block 数`）写成 `ArchProfile::xxx_for(arch, 形状…)` 的规则 + 单测两边各一例，kernel 保留两档特化、零数值。分叉量找不到物理解释时才走启动自调优。
- **寄存器预取（PF=1）的收益不稳**：bf16 中间量后 B300 1×8192 从退 → 赢 25%，wy 改 16 B 写后又被抹平（L2 命中态变了）。这类"随邻居 kernel 变"的选项不要写死进 ArchProfile 分档，走启动自调优。

## 在哪张卡上迭代（用户 2026-10-20，r9 教训）
- **性能改动在 B300 上出数定去留**，3090 只跑 parity / 编译。r9 六轮 3090 迭代 + 两轮 B300 复测，B300 没超 vLLM：3090 撞带宽墙（conv 84% 带宽、两段版多搬 200 MB 输），B300 撞延迟 / 并行度墙（h-pass 单独 721 > 融合 659 µs，串行链每 chunk ~5 µs 掩不住），3090 上筛出的候选集不含 B300 要的改法。
- **先看 B300 的 nsys / ncu 再写改法**，不从 3090 profile 推；"验过" = ≥ 2 卡型 × 全部 bench 形状进 probe.csv；`ArchProfile` 每个分档要有该卡的实测行。
- 3090 上看不出的 B300 特性：smem 227 KB（双缓冲装得下）、HBM 延迟型（occupancy / 软件流水比减字节数更要紧）、148 SM（单序列 192 block 铺不满）。
- **B300 上 ncu 要 `sudo -n /usr/local/cuda-13.1/bin/ncu ...`**（普通用户 ERR_NVGPUCTRPERM；nerdctl 已走 sudo -n 免密）。看三个数就够定方向：`Eligible Warps Per Scheduler`（< 1 = 延迟型，加 warp / 减依赖链）、`Block Limit Registers/Shared Mem`（哪个限占用）、`DRAM Throughput`（< 20% 就别再减字节）。 `--kernel-name regex:` 只匹配裸函数名（`ho_mma_kernel`），带模板参数的 `regex:"ho_mma_kernel<64, 32, 128, 1>"` 匹配不到任何 kernel（"No kernels were profiled"）；要选特化用 `--launch-skip/--launch-count` 定位，或 `--kernel-id ::regex:ho_mma_kernel:3`。r10 第 1 轮 bf16 中间量只换来 wy −5%，ncu 一看 DRAM 6%——"减写量"本来就不是 wy 的瓶颈（r10 loop 页第 3 轮前置）。
- **"Mem Busy 77%" 不等于 smem 是瓶颈**（r10 第 5 轮）：把 ho 的 B 操作数 4×lds_u32 改 ldmatrix 后 shared_ld 指令 −79%、mio_throttle −33%，**时钟一个 µs 没动**——那 77% 是 mma 链空转时 smem 管线相对占比高的症状。改 kernel 前先看三个"产出"指标：`smsp__issue_active`（发射槽占用，ho 23%）、`sm__pipe_tensor_cycles_active`（tensor 管线，19%）、`sm__warps_active`（8.5%）；三个都低 = 并行度 / ILP 不够，减指令数没用，要加每 warp 的独立 mma 链（BV / tile 数）或每 SM 的 block 数。`--launch-skip` 要挑**真正慢的那个 launch**（1×8192 的 192 block 与 8×1024 的 3072 block 是两种病）。
- **fla 对照要按 kernel 对段**（`scripts/dev/fla_prof.py`，torch.profiler 在 vLLM 容器里跑）：kkt+tril+wu ↔ wy、fwd_h+fwd_o ↔ ho、l2norm+cumsum ↔ prep，fla 不含 conv（vLLM 另走 causal_conv1d）。r9 的 "chain vs fla" 数把 conv 219 µs 算进了自己一侧。

## 数值口径
- 硬判决（top-k / 路由 / argmax）只在可判定行断言（ref top-2 gap > 阈值）；bf16 输出链 rel ~5e-3 = 一 ulp，用 cos ≥ 0.999x 而不是 abs。
- 参考侧 rmsnorm 的 bf16 落点要与 kernel 一致（乘权重**后**落 bf16）。
- 运行时下标的寄存器数组会落 local memory（stack frame ≠ spill）：模板递归展开。

## megakernel（单层 persistent kernel）
结构事实（组数 / 流水级数 / CTAs per SM 的实测选择）与第一阶段数字见 **[megakernel.md](megakernel.md)**。

## smem 瘦身优先于算法改动；寄存器预取在 255 上限附近反伤（r9，2026-10-20）

- **判断**：block 级 kernel 慢，先看 `--resource-usage` 的 smem / 寄存器，把 smem 压到让 occupancy 从 1 → 2 block/SM，往往比改算法收益大；寄存器已接近 255 时再加"预取下一块到寄存器"会 spill，反而慢 20–70%。
- **为什么**：r9 wy kernel 把 sM 复用 sK∪sQ（72 → 56 KB）占了 2.2× 提速的大半，分块三角逆只再省 17–20%；ho 的 PF=1 预取在 3090 / 占满寄存器时全形状退化。两段版 ho（h-pass + o-pass）在 3090 比融合慢 10–30%，因为中间 h16 一写一读 201 MB @ 8K 吃带宽——这类"拆 kernel 换并行度"的改动收益随卡的带宽变，门限必须进 `ArchProfile` 按卡实测，不能按一张卡的结论定死。
- **怎么用**：改 kernel 前 `nvcc --resource-usage` 记 smem / reg / occupancy；先做瘦身，再做算法；"预取 / 拆段"这类结构开关一律 `ArchProfile` 默认 + 环境变量覆盖，两卡各测后再定默认。
- **证据**：`docs/loops/qwen35-gdn-r9.md` 轮次 1–5；`test/qwen35/2026-10-20/probe_gdn_r9_*`；commits ba23769 / 515ef0c / e78ccfe。

## tcgen05（UMMA）在这套描述符构造下只支持 M=128（2026-10-21 r11d 实测）

**判断**：用 `k_major_128b_desc`（LBO=0、SBO=1024B、128B swizzle）+ `make_instr_desc_bf16(m, ...)` 时，
**M=64 会静默算错**：每 16 行里只有前 16 行有结果，其余为 0，且 `cudaGetLastError()` 返回 no error。

**证据**（B300 GPU 4，身份矩阵 A × 全 1 的 B，期望输出全 1）：

| M | 结果 |
|---|---|
| 64 | `good=4096 / 8192`，`out[0..15]` 对、`out[16..31]` 零、`out[32..47]` 对、`out[48..63]` 零 |
| 128 | **`good=16384 / 16384`，零错** |

试过调 SBO（128 / 256 / 512 / 1024 / 2048 B）**完全不影响结果** —— 说明 M 方向的寻址不走 SBO，
M=64 的累加器在 TMEM 里不是「64 lane」而是别的组织，按 32-lane 取回只能拿到一半。

**怎么用**：GDN / 任何 BT=64 的 chunk GEMM，**把 A 补零到 128 行、指令描述符写 M=128**，
多算那一半零行。代价 2× FLOP，但 tcgen05 有 2236 TF/s，这点浪费远小于调描述符的时间。
如果将来必须 M=64，照 CUTLASS `make_smem_layout_a` 的完整 canonical 布局生成（而不是手写 LBO/SBO）。

**教训**：`maxabs` 大但 `out[0][0]` 对，不要当成「bf16 累积误差」放过——
r11d 里程碑 1 就是这样把 `maxabs=12.94` 解释掉了，实际是 3/4 的行全错。

## mbarrier 的 parity phase 必须逐次翻转（2026-10-21 r11d）

**判断**：kernel 每次运行结果**都不相同**、且没有报错时，先查 mbarrier 的 phase，
不要先怀疑布局 / 数值。

**证据**：r11d 的 `ho_umma_kernel` 三处 `mbar_wait(bar, 0)` 都写死 phase=0。
chunk 0 正确（`tmp0 = 0.95790` 与参考一致），但 chunk 1 起等的是**尚未发生的到达**，
于是读到上一轮或半成品数据 —— 连续几次运行的 `state[0][0]` 分别是
`-58.24 / -54.71 / -36.29 / -25.44`。改成 `ph ^= 1` 后立刻稳定在 `-60.94007`（参考 `-60.94093`）。

**怎么用**：每个 mbarrier 维护自己的 `uint32_t phase`，**每次 wait 后 `phase ^= 1`**。
`umma_peak` 里是对的（`phase ^= 1`），照抄那段就行。

**为什么容易漏**：只有跨多次到达（chunk 循环、stage 流水）时才会暴露；单次 kernel 永远看不出问题。
而且它不报错、不崩溃，只是数值随机 —— 很容易被误判成"布局还不对"去改描述符。

## 给已有的 attention kernel 加滑窗（mask）时
- **三件事一起改，少一件就退化**：① 可见性加**下界**（`key >= vis - window + 1`，用 `INT_MIN` 表示无窗口）；
  ② **起始 tile 跳读** `tile0 = max(0,(min_vis-window+1)/TILE)`，否则滑窗层照样扫全部历史，
  复杂度一点没降；③ 双缓冲索引改成 `(tile - tile0) & 1`，否则 `load_tile(tile0)` 预取的 tile 与循环首轮的
  buf 不一致。
- **参考 kernel 也要改**：`*_ref_kernel` 一般是三遍扫（max / sum / PV），起扫点从 0 换成
  `j0 = max(0, n - window)`，三个循环都要换（少换一个就只错一部分）。
- **新增入口而不是改旧签名**：加 `gqa_*_win_bf16(..., int window, ...)`，旧符号转发 `window=0`。
  这样 Qwen3 / Qwen3.5 / GLM / K3 全部零改动、零回归风险，`opset.rs` 的 trait 再加带窗口的方法即可。
- **证据**：2026-10-21 Kaon-V3 阶段 3（`docs/loops/kaon-v3.md`，commit 4af6379）。

## 改模板化 kernel（v5/v6/v7 这种多版本共存）时的定位方法
- **不要凭"看起来在一起"插代码**：先用 `grep -n` 拿到目标变量**声明行号**，再按行号区间插入。
  同一个概念在不同版本里变量名不同（v5/v6 用 `warp_min_vis`，v7 用 `pair_min_vis`），
  用统一锚点批量替换一定会踩空。
- 批量替换后先 `grep -c` 数命中数，再编译；nvcc 的 `identifier is undefined` 一次只报一处，来回很慢。

## 改 extern "C" 签名 / 批量脚本编辑的硬规矩（2026-10-28，一天内各犯 2–3 次，每次赔 10 分钟远端编译）
- 改任何 C ABI 函数签名：`kernels.h` / `.cu` 定义 / `ffi.rs` 声明（+ `stub!`）**三处同改**；提交前 `grep -n '<fn>(' ` 三处核参数个数。漏 kernels.h 的症状是 nvcc "more than one instance of overloaded function has C linkage"。
- Python 批量编辑脚本：任一 assert 失败 = **该文件全部改动未写盘**（前面文件已写）。每文件独立写盘，脚本末尾对每个目标串 `count` 打印；编译前再 `grep -c`。
- 不同实现路径的对拍（如 bf16 cache vs e4m3 cache）要**固定同一 kernel**（env 强制 cfg），否则差的是 kernel 累加序不是被测项（D67）。

## 同一算子的两条快路径（v5 / v6）累加序不同：按 cache dtype 分路会让"同 prompt 不同步型"结果不同（2026-10-29，kaon3 D121）
- kaon3 e4m3 cache 时 D=256 滑窗层 Paged 步退 v5（v6/v7 不读 e4m3），bf16 cache 时走 v6；纯 Prefill 两种都 v6。于是 prefix cache 命中后的 chunk（Paged）与首次 prefill 的 top-1 可差几 ulp，词沙拉 prompt 上贪心翻转 3–4/7，自然语言几乎不翻。不是 bug，但是**非确定性来源**。
- **规则**：新加 dtype / 布局变体时，优先给**同一个 kernel** 加模板参数（B26 给 v6 加 KV_FP8），不要让某种 dtype 整体退到另一个结构不同的 kernel；对拍测试要比"同一形状 × 两条路径"，不只比"同一路径 × 两种 dtype"（histpar 的位差 0 是 v5 vs v5）。

## 小 M grouped GEMM（MoE decode）：先量 L2 热微基准，再看"链长"（2026-10-29 kaon3 D180–D183）
- **判断**：decode（每专家 1–8 行）的 mma tile grouped GEMM 不是带宽瓶颈而是**每 CTA 的串行 K 链**：CTA 数 ≈ 激活专家 × N 块（~176–616），每 CTA 串行 K/BK 次 {cp.async wait + 2 sync + mma}，每次 ~0.2 us（L2 热）/更多（DRAM 冷）。
- **证据**：M=1 权重 18 MB 全在 L2 热时仍只 ~1 TB/s；BK 128 → 256（迭代 22 → 11）微基准 -15%，端到端 c1 -3.8% / c8 -10.6%（冷数据下链长代价更大）。
- **怎么用**：① 先用 `moe_gemm_timing` 一类 nsys 外微基准（真实 decode 形状、下限按 DRAM 算，但注意 L2 热会高估）；② 刀的顺序：加大 BK（smem 允许时，K 尾块零填充保持 `K % 64 == 0` 即可）→ 减 BN/加 CTA（在飞字节）→ split-K；每一刀都用"四进程 dump cmp"做位同门禁（启动器 env 是 static once，同进程扫不动）。
- **反例**：B7 的带宽型 GEMV（warp 一行、lane 流式）在这种形状慢 4–8×：工作太碎，每 block 装载 + barrier 占大头。
- **融合改线程→元素映射时先核合并率**（2026-10-29 kaon3 D194）：把"按 d 跨线程连续"改成"线程持 8 元素组"后，标量 bf16 读的 warp 合并率 1/8，parity 过但端到端退化 1–2%；改成每线程一次 16 B 向量读即可，累加序不变。

## 换 ldmatrix 前先量 LDS 占比：p64（BM64×BN128 NVFP4，每 mma 2+2 次 LDS）换 ldmatrix 只 −7%（2026-10-29 kaon3 G1）
- 片段读指令减半不等于时间减半：mma.sync m16n8k64 本身 + 等 cp.async 到位占了大头。算力型 grouped GEMM 的下一档是更大 warp tile（BM128：A 片段复用 B ×2）或更深流水 / TMA，不是 LDS。端到端 c16–c64 噪声内，默认关。
