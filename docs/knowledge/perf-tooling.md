# 性能工具链：怎样拿到可信的"谁大"（perf-method 的工具页）

> 与 [perf-method.md](perf-method.md) 分工：那页讲方法论与口径，这页讲**把测量做对**的具体手法与坑（编译验证、nsys、A/B 脚本）。来源：loop `kaon3-decode-opt` 2026-10-28。

## 1. 判瓶颈先做"整段 skip"，但 skip 结果只在**确认改动已编进二进制**后才算数（2026-10-28 kaon3-decode-opt）
- **判断**：改 kernel / 加 skip 开关后，必须 `strings 二进制 | grep <新字面量>` 验证进了，再看结果；否则"skip 全 0 影响"可能只是老二进制（D5–D16 整天的"瓶颈在 host"全是假象，D45 新二进制 skip 立刻看到 MoE 5.3 / paged 3.2 / dense 2.8 ms）。
- **为什么**：三条编译链路都会静默跳过：`cargo -p dinfer` 不重编依赖 crate；rsync 保留旧 mtime 让 cargo 认为源没变；改 `.cu` 后 nvcc 增量不重跑。`remote_*.sh put` 已加 `--no-times`。
- **怎么用**：Step B 验证脚本（`scripts/dev/kaon3_stepB_verify.sh`）固定做"备份旧二进制 → 编 → strings 验 marker → A/B/A2/B2 同轮 + 固定 prompt 输出"。
- **证据**：loop kaon3-decode-opt D42 / D45 / D49。

## 2. 图内 kernel 清单要用 nsys `--cuda-graph-trace=node`，且必须 `--cuda-flush-interval` + 不 INT nsys 本体（2026-10-28）
- **判断**：定 decode 图内谁大，nsys 节点级清单（kernel 名 / 次数 / 时长 / grid）比任何 host 计时或 NO_GRAPH 分段都准；一步 decode 1298 kernel 的分布半小时就把"按 max_tokens 行空跑的 norm/rope"（3 ms，18%）揪出来。
- **为什么**：CUPTI 缓冲默认只在 `cudaProfilerStop()` 落盘，被 INT 杀的服务不调它 → 报告里只有加载期 memcpy、无 kernel 表（两个版本 nsys 各失败一次）；INT 发给 nsys 本体会打断报告生成（无 .nsys-rep）。
- **怎么用**：`scripts/dev/kaon3_nsys_decode.sh`（flush 1 s，只 INT dinfer，等 nsys 自然退出）+ `scripts/dev/nsys_kern.py`（按 graphExecId 聚类取 decode 图实例，逐 kernel 聚合）。看 grid 是否等于 max_tokens / max_rows 这类上限值——那是"按缓冲满长跑"的标志。
- **证据**：loop kaon3-decode-opt D52 / D54；B1 一刀 TPOT −19.4%。

## 3. 卡型旋钮（norm layout 等）先用现成 env 三态 A/B，再落进 ArchProfile（2026-10-28）
- **判断**：能用环境变量切的实现选择，先同轮三态 A/B（含输出一致），赢家写进 `ArchProfile` 按 `Arch` 分支，不留 env 依赖。B2：Sm120 h=2816 单行 norm Block256 2.4 us < Block1024 6.8 us < warp，−3.4%。
- **怎么用**：A/B 脚本每个变体都要抓固定 prompt 的输出；输出坏（"is is is…"）的变体数字作废（D57 `KAON3_MEGA=0` 11.33 ms 就是坏的）。

## 18. 小 M（decode）GEMM 先做 roofline 表，再选结构（2026-10-28 kaon3 B5–B7）
- 每段：字节/层 ÷ 卡带宽 = 下限，与 nsys 实测比。离下限 >5× 的段才是刀；dense mega 8.7× → 换带宽型 GEMV 一刀 -22%。
- "warp 一行 GEMV" 只对**单个大矩阵**（每 warp 流 ≥10 KB）有效；MoE 单专家 N=704 每 warp 只 2.8 KB，比 mma tile 慢 1.8 ms（B7 翻车）。结构迁移前先算每 warp 工作量。
- A 驻 smem 的 GEMV：smem 占用决定 block/SM；m=8,K=2816 的 45 KB 让 c8 反慢（B6），门槛 ≤24 KB。warp 数不是越多越好——block 数（并行度）比 A 复用重要。

## 19. graph 内 launch 间隙 ≈ 0.5–0.7 us，不是 1.5（2026-10-28 D65/D66）
7 launch → 1 实省 0.16 ms；5 launch → 1 省 0.07。用 NO_GRAPH/eager 的间隙估 graph 态收益会高估 2–3×。减 kernel 数是小刀，>0.2 ms 的刀在 kernel 本身（KV fp8 存、MoE GEMM 流水/tile）。

## 20. ABAB 多变量叠加必拆（2026-10-28 D64/D65/D66）
一个链里 ≥2 个改动时，用 `envab.sh`（同二进制、env 开关）逐个拆；A 侧二进制是谁（bN 备份的是上一次编的）要写进记录；两个 RESULT 拼在一条输出里按文件名分段读，别按行序归档（D64 错读过）。

## 21. nsys 全表的 grid 列按 max_tokens 过滤；融合 kernel 要比"被融合的 kernel 之和"（2026-10-28 D72–D74）
- 预分配缓冲（max_tokens 行）配 "行数 = len()/hidden" 的包装 = 按满长 launch。每次 nsys 复抓先 `grep 4160`（或你的 max_tokens）——B1 和 B18 两处同病相隔 10 刀才发现，各值 0.5 ms/步（TPOT 的 8%）。
- 融合的判据是 kernel 时间 ≤ 原各 kernel 之和 + 省下的间隙；B10 单线程 top-k 16 us > 原 7.3+5.6，靠 launch 数省的 0.6 us 盖不住。单 block 融合 kernel 里的串行循环（K×E）要按 warp 并行写。
- 两刀合计 c1 -12.7%，是全天单链最大收益——**复抓 nsys 的 ROI 高于再写一个 kernel**。

## 22. 验证链的三方钉核要**写进 RESULT 头部**，不然会漏（2026-10-28 B22a）
kaon3 5090 的 bN/envab 只钉了 bench（60-91），serve 浮动 128 核跑了一整天 20 多轮，没人发现——因为 RESULT 里没有钉核字段。规矩：链脚本第一行写 `serve_taskset= bench_taskset= vllm_taskset=`；serve 绑 GPU 所在 NUMA 的核（`nvidia-smi topo -m` 的 CPU Affinity），与 bench 不重叠。

## 23. 一刀一链的固定模板（2026-10-29 kaon3 b49–b53）
- `scripts/dev/kaon3_envab_c.sh`：同二进制 env/args ABAB（c8–c64 两轮交错、每窗口换 seed、固定 prompt 输出、HOST_TIME 混批步统计、启动时间），`TAG/AENV/BENV/AARGS/BARGS/CS` 参数化，结束标记 `<TAG>_DONE`。
- kernel 改动前置门禁：微基准（`moe_gemm_timing` 的 P424/P2148 形状）两进程 env 开/关各跑一次，**输出 CHECKSUM 必须逐位同**（`GATE=0`）才编主二进制起 ABAB；门禁失败链直接退出不占 GPU。
- 结果入库：`scripts/dev/e2e_round.sh kaon3 <round> "<一句话>" <group> d_tps v_tps d_ttft v_ttft d_tpot v_tpot`（vLLM 列用 stepA4 中位数、注明），每刀 4 行（c8/16/32/64），负结果也进（round 名加后缀如 `b49c`）。
- 一天 5 刀的节奏：写 → 本地 `cargo check`（Rust 侧）+ `CUDA_ARCHS=86 cargo build -p dinfer-kernels`（nvcc 语法）→ `sync` → 远端链 → 等的时候写下一刀 / 记录 / 知识。
