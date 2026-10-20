# 质变实验第一阶段：Qwen3.5 单 GDN 层 decode persistent kernel（design.md §14.3，用户 2026-10-20 定"实验和 K3 同步进行"）

> 目的：**只拿数字**，验证"消掉 kernel 边界"在 B300 上值多少；≥ 25% 才进第二阶段（任务图架构，§14.5）。不改 executor / 不接服务。

## 环境
- B300-50 GPU 4–7（`B300_TAG=50 scripts/dev/remote_b300.sh`，远端 `source /data/yaochen/infer/env.sh`，`CUDA_DEVICE_ORDER=PCI_BUS_ID`）；GPU 0/1 Qwen3.5 常驻、2/3 K3 子代理用。本机 3090 只做编译 / 对拍（sm86 无 TMA，不测性能）。
- 代码：`crates/kernels/cuda/mega_sm100.cu`（Blackwell 专用；按分文件规则不写 sm8x 版），C ABI `mega_gdn_layer_decode_sm100(...)`，Rust `ops::mega_gdn_layer_decode`；测试 `crates/kernels/tests/mega_parity.rs`（对拍 + 计时）。
- 真权重形状：hidden 5120、in_proj N=16480（qkv 10240 | z 6144 | a 48 | b 48）、conv W=4、HV=48 / H=16 / K=V=128、out_proj 6144→5120、gate_up N=34816、down K=17408；B ∈ {1, 8, 32}。权重用随机 bf16（形状对就行，不加载真权重）。

## 观察方法
- 基线：现有路径该层 decode 的 **11 个 kernel**（rmsnorm → in_proj GEMM → conv1d_update → recurrent_decode(fused gate) → gated_rmsnorm → out_proj → fused_add_rmsnorm → gate_up → silu_and_mul → down → add）在 B300 的 nsys node 级总和（含 kernel 间隙）：现在 B=1 ≈ 0.14 ms/层、B=32 ≈ 0.21 ms/层（取自 c1 / c32 decode 步 ÷ 64 的 GDN 层份额，实验开头先精确测一次写进记录）。
- 实验：同输入、同随机权重，persistent kernel 一次 launch 的耗时（CUDA event，1000 次均值）；输出与基线 bf16 对拍 cos ≥ 0.9999 / max rel ≤ 1e-2。
- 中间量：每阶段的 SM 利用率（nsys / ncu）、依赖等待占比（kernel 内计时器写全局，可选）。

## 停止条件（机器可判定）
`cargo test -p dinfer-kernels --features ref-tests --test mega_parity -- --ignored` exit 0 且打印 `speedup_B1 / B8 / B32`；**任一 B 档 ≥ 1.25× 进第二阶段**；三档都 < 1.10× 判失败，记原因收工。

## 迭代上限
7 个工作日 / 12 轮。每轮一行记录。

## 设计约束（不是调优数值，是结构）
- grid = 常驻 CTA（数量 = occupancy × SM，由 host 查）；任务 = (阶段 id, 分片 id)，静态编号；依赖 = 全局 `ready[stage]` 计数器，消费者 `ld.acquire.gpu` 自旋，生产者 `red.release.gpu` 加一。
- GEMV 阶段：权重按 N 切片给 CTA（每 CTA 若干 N 行，K 维整段），A 行（B×K bf16）从 L2 广播读；TMA（`cp.async.bulk`）把权重切片流进 smem 双缓冲；累加 f32，M=B 行。
- GDN 递归阶段：复用 `recurrent_decode_kernel` 的寄存器方案（每 warp BV/4 行状态），任务 = (seq, hv, V 块)。
- 阶段间激活走全局（一层 B=32 × 16480 × 2 B ≈ 1 MB，L2 常驻）。
- 所有分片大小 / CTA 数 / 双缓冲深度由 host（`ArchProfile`）给，kernel 内无常量。
- 先做**不含 attention 的 GDN 层**（48/64 层是它）；全注意力层和 lm_head 留第二阶段。

## 记录

| 轮 | 改动 | B=1 / B=8 / B=32（基线 → 实验，ms/层，倍率） | 结论 |
|---|---|---|---|
| 0 | 结构落地（commit 38f103e）：11 阶段静态任务表 + ready 计数器；GEMV = cp.async.cg 流水 + mma.sync m16n8k16（A/B 同置换装载）+ split-K 自复位信号量；GDN 复用 gdn_decode.cuh | 3090 dev 模式（sm86，`MEGA_SM100_DEV=1`，groups=2/unroll=4/stages=2）：0.901→0.936 / 0.987→1.005 / 1.254→1.308，0.96 / 0.98 / 0.96×；对拍 res cos 0.999999 max_rel 5e-3（1 ulp），rec/conv 状态 cos 1.0 | 3090 是带宽下限（766 MB/层 ÷ 936 GB/s ≈ 0.82 ms），两边都贴着下限，看不出边界效应；数字只证明正确性。踩坑：cp.async `wait_group` 深度写成 pipe-1（应 pipe-2），pipe=2 时等于不等，第一版全是垃圾 |
| 1 | B300 首测（默认 groups=4/unroll=4/stages=3/ctas=1，splits in/out/gu/down = 2/7/1/7，grid 148 occ 1，smem 209 KB）；加阶段 prof（globaltimer：每阶段 span / Σwait / Σbusy）；GDN 任务从 BV=32 改为每 warp 16 行、任务 = (seq, hv) | 0.146→0.188 / 0.158→0.215 / 0.190→0.339，**0.78 / 0.74 / 0.56×**（GDN 改前 B=32 0.409）；对拍同第 0 行 | prof（B=1，µs）：in_proj span 37（169 MB → 4.5 TB/s；每 CTA busy 22.7 = 57 GB/s/SM，1 CTA×8 warp 藏不住延迟）、conv **16**（每线程 8 维串行 + break 阻止提前发读）、gdn 8、gated_norm 3、out_proj 16（63 MB，理想 8）、add_norm **10**（单 CTA 一行 rmsnorm，4 B 读）、gate_up 55（356 MB → 6.5 TB/s ≈ 基线 58）、silu 3、down 30（理想 22）、add 3。B=32：gdn 80（1536 任务 × 4.9 µs，10 波无重叠）、conv 75、gate_up 70（MB=2 的 mma 环 43 GB/s/SM）。结论：输在 (a) 占用（寄存器 169–255 → 1 CTA/SM）(b) 小阶段串行延迟链 (c) GDN 无跨任务预取。参数扫描（ctas=2 因寄存器仍 occ 1；stages=4 G=2 最好 0.73×）不解决结构问题 |
| 2 | 结构改（commit cad272b）：add / silu 融进 out_proj / down / gate_up 尾声（11 → 8 个非空阶段）；GDN 任务 = (seq, gdn_hv 个 v 头) + 下一 tile 状态 cp.async 双缓冲预取；conv 每线程 2 维全读后算；rmsnorm 16 B 向量；OCC=2 模板（G=2，128 寄存器 spill 272 B） | **B300 GPU 4**（默认 groups=4/unroll=4/stages=3/ctas=1，splits 2/7/1/7）：0.1475→0.1621 / 0.1582→0.1942 / 0.1904→0.3203，**0.91 / 0.82 / 0.59×**；对拍 res cos 0.999999 max_rel ≤ 8e-3，rec/conv 1.0。9 组扫描无一更好：stages=2 0.77×、U=2/stages=4 0.88×、G=2/stages=4 0.82×（B=32 最好 0.65×）、**OCC=2 全部更差**（0.73 / 0.60 / 0.58×：spill + 每 CTA 流水减半）、gdn_waves 1/4 与 2 持平 | prof（B=1，µs）：rmsnorm 3、in_proj 33（5.1 TB/s；基线 cuBLASLt ≈ 26）、conv 4、gdn 15（48 任务 × 2 µs busy，但 span 15 = 3 波延迟）、gated_norm 4、out_proj 17（63 MB 理想 8）、rmsnorm2 5、gate_up 58（6.1 TB/s = 基线）、down 33（理想 22）。B=32：gdn 80 → 55（预取有效但仍 2× 独立 kernel 的 32）、gate_up 112（MB=2 mma 环 3.2 TB/s，基线 Lt ~70）、down 140（含等 gate_up）。**判定：三档 < 1.10×，实验失败，收工**（见下"结论"） |

## 结论（2026-10-20，第 2 轮后判定失败）

**数字**：B300 单 GDN 层 decode，基线（11 kernel，Gemm 自调优，CUDA graph）147 / 158 / 190 µs；persistent kernel 最好 162 / 194 / 320 µs = **0.91 / 0.82 / 0.59×**，三档都 < 1.10×。对拍一直是过的（res 1 ulp，状态逐位），输的是性能不是正确性。

**为什么输（prof 证据，不是猜）**：
1. **占用**。常驻 grid 每 SM 只有 1 个 CTA（8 warp）：GEMV 分支的 mma 累加器 + 片段把寄存器推到 170–255。独立 GEMM kernel（cuBLASLt / small_m）每 SM 跑 4–8 个 block、32–64 warp。结果 in_proj 5.1 TB/s vs Lt 6.5；B=32 的 mma 环只 3.2 TB/s。压到 128 寄存器（OCC=2）spill，更差。这是 B=8/32 输的主因。
2. **小算子在常驻 grid 上变成延迟链**。基线的 conv / gdn / gated_norm / rmsnorm 各是一个高占用 kernel，2–6 µs；在 148 个 CTA 上同样的工作被切成"几波 × 单次全局访存延迟"，gdn B=1 span 15 µs（busy 2）。融合（add / silu）和粗任务 + 预取把 B=1 从 188 压到 162，再往下每个阶段都要重写成"一个 CTA 内部流水"的形式——那就是在重写每个算子。
3. **kernel 边界效应比预估小**。design §14.1 把 3.8 ms/步 归到 ~900 个 kernel 的边界；但 graph 重放下 11 个 kernel 的单层基线 B=1 只有 147 µs，而其中 GEMV 权重流本身的下限（766 MB / 8 TB/s）就是 96 µs，再加 GDN 状态 / 小算子 ≈ 115–120 µs。基线离下限只有 ~25 µs（17%），megakernel 能赢的空间最多就是这 25 µs，而它自己为了常驻付出的占用代价（GEMV 慢 6–10 µs/个 × 4）已经吃掉了。
4. 依赖等待本身不是瓶颈：Σwait/grid 占 busy 的 10–30%，"先发权重流再等依赖"确实把屏障藏掉了大半。megakernel 的机制是对的，输在它改变不了的东西（寄存器 / 占用）。

**没试、但判断不会翻盘的**：TMA（cp.async.bulk）只减少发指令的线程开销，不改变占用；tcgen05（Blackwell 原生 mma）能把 mma 环加速但 B=1 不是算力瓶颈；warp-specialized（生产者 / 消费者 warp）能提高单 CTA 的在飞字节，但 1 CTA/SM 的 smem 带宽 / 延迟隐藏仍不如 4–8 个独立 block。Hazy Research 的 1.5× 是在 Llama-1B（权重小、每层 < 20 µs、边界效应占比高）上对非 graph 的 vLLM/SGLang 测的；27B + graph 基线下边界效应只剩 17%。

**对 §14.5 的影响**：阶段二"任务图架构"不以 megakernel 为目标继续；任务图抽象本身仍可做（给逐 kernel 路径用，减少 host 侧 launch 序列的重复代码），但不是性能项。性能质变的路留 B（NVFP4 权重，下限 6.5 → 1.6 ms）与 C（K3 MoE 融合）。本实验的可复用件：`mega_sm100.cu` 的 GEMV mma 置换装载 + split-K 自复位、`gdn_decode.cuh` 抽出的共享递推体、阶段 prof 方法。
