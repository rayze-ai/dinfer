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
| 2 | 结构改（commit 见 git log "mega 第 2 轮"）：add / silu 融进 out_proj / down / gate_up 尾声（11 → 8 个非空阶段）；GDN 任务 = (seq, gdn_hv 个 v 头) + 下一 tile 状态 cp.async 双缓冲预取；conv 每线程 2 维全读后算；rmsnorm 16 B 向量；OCC=2 模板（G=2） | 3090（dev 模式）：0.893→0.914 / — / 1.244→1.278，0.98 / — / 0.97×（第 0 行 0.96 / 0.96）；对拍同前。**B300 待 token**（10:44 过期，coordinator 在要） | 3090 prof：conv busy 15 → 0.5 µs、rmsnorm 7 → 2–4 µs、add/silu 阶段消失；GEMV 阶段仍 ≈ 带宽下限（3090 看不出流水差别）。B300 要看的：GDN B=32 80 → ? µs、in_proj 4.5 TB/s → ?、OCC=2（spill 272 B）vs OCC=1 |
