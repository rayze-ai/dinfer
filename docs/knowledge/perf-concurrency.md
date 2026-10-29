# 并发 / prefill / 混批步的性能方法（perf-method 的续页，来源 kaon3 loop 2026-10-28/29）

> 与 [perf-method.md](perf-method.md) 分工：那页是单请求 decode 的 profile → 下限 → 门禁方法；这页是 **c ≥ 8 时** 并发差距怎么定位（环节对标表）、prefill 步 GEMM 为什么慢、调度旋钮的真假、pair 步的结构性代价。

## 15. 性能 bench 查不出"输出错"：每条链必须带 c≥2 的逐请求输出比对（2026-10-28，kaon3 D87/D103）
- 几十轮 c8 bench 的吞吐 / TPOT 全是在**两处正确性 bug** 下测的（混批步在飞 decode 行 attention 没人算；同步 runner 的步 n 采样结果被 cut-in / 提前 begin(n+1) 覆盖）：token 数、步数、时延一个不变，输出乱码。c1 的两条 curl 永远对（c1 不混批、无 pending）。
- **怎么用**：bN 链前置 `scripts/dev/kaon3_pair2.sh`（2 路/8 路同时发 vs 串行，env 变体）+ `kaon3_mixparity.sh`（同 prompt 串行 vs c2/c8）+ `kaon3_cacheprobe.sh`（c1 同 prompt 三次 = prefix cache 命中后的带历史 chunk）；三者全 SAME 才允许比性能。差异从 **char 1** 起 = 首 token 就错 = 采样行/结果映射问题，不是数值精度。
- 同服务同 seed 重跑 bench 命中 prefix cache，TTFT 掉 7× 是假象；多次 bench 换 seed（见 pitfalls 2026-10-28）。

## 16. "快路径缺一个分支"类慢：先看分派表再看 kernel（2026-10-28，kaon3 B26）
- c8 TTFT 545 ms（vLLM 146）根因是 `gqa_paged_win` 对 D=512 没有快分支，落到 grid=(req,head) 的串行标量 kernel；一个 Paged 步 476 ms vs 同 token 纯 Prefill 127 ms。补 v6 PAGED 分支后 TTFT -71%、吞吐 +45%。
- **判法**：某类步（混批 / chunked / 某 head_dim）比同 token 数的"正常"步慢 ≥ 3× 且有固定开销，先 grep 分派函数里每个 `head_dim ==` / `cfg >=` 分支，列"形状 × 路径"表，找落到 `_ref` / `generic` 的格子；比 nsys 快。
- **§15 补（2026-10-29）**：并发 / cache 命中 / chunked prefill 三种触发下的输出差异，先用 `--no-prefix-caching` + 同批构成重复（c8 vs c8）判**确定性**；确定 + 语义等价 ⇒ 批不变性（m 依赖数值），不是功能 bug。真正的功能 bug 形态：同批构成不确定、或乱码 / 语义崩坏（如 B27 修前的混批 decode 行）。

## 17. 追并发差距先做"环节 × 并发"对标表，TPOT − 纯 decode 步 = prefill 插入量（2026-10-29 kaon3 D202）
- **判断**：bench TPOT 与 `HOST_TIME` 的纯 decode 步 exec 之差就是 prefill / 混批步摊进来的量；把它和 vLLM 的 `/metrics`（prefill_steps、request_prefill_time、iteration_tokens）并排，一眼看出差在"prefill 步本身慢"还是"调度把它堆成一大步"。c16/32/64 的差额 64% / 100% / 130% 都在这里，纯 decode 步 M≥32 其实已持平。
- **怎么用**：`scripts/dev/kaon3_stepA4.sh`（两端交错、同窗口同 seed、两端 /metrics 增量）；表格模板见 loop D202。领先的环节列出来，后续改动不许退化它们。
- **证据**：`test/kaon3/2026-10-29/stepA4/`。

## 18. 调度旋钮 A/B 前先核它的语义对不对得上负载（2026-10-29 kaon3 D206）
- `long_prefill_token_threshold` 是**单请求**上限，prompt 428 tok < 512 时形同虚设，两轮白跑；一步 2148 tok 是 10 个整请求，要的是**整步 prefill 配额**（B47 `--prefill-tokens-per-step`）。零代码 A/B 便宜，但先花 1 分钟读 scheduler 那几行，别靠参数名猜。

## 19. prefill / 混批步的 lm_head 要只算采样行（2026-10-29 kaon3 D205）
- nsys 混批步里 `cutlass Kernel2 grid (56,256)` = 2148 行 × 262144 词表全算再 gather，占步时 14%（c64 1/6）。按 grid 复核"行数是不是 tokens"这一条（perf-tooling §21）对 lm_head 同样适用。vLLM 用 `logits_indices` 只算最后行；接新模型时 lm_head 一律按采样行算（B46 模式：gather hidden → GEMM nrows 行）。

## 20. "把大步切小"只在每步固定开销可忽略时成立；eager 混批步固定开销 5–8 ms，切 N 段就付 N 次（2026-10-29 kaon3 D209）
- B47 整步 prefill 配额 1024：TTFT p50 −60–75%，但 TPOT 全档 +10–26%、吞吐降。切出来的每个混批步都是 eager 副流 + 主流等待，launch 2900 个 kernel 的 host 开销 + 两流同步没被摊。vLLM 的 chunked prefill 能便宜是因为它的 prefill 单步吞吐高（50 tok/ms），不是因为切得细。
- **判断**：改调度前先量"一个混批步的固定开销"（HOST_TIME 的 exec − nsys kernel 和），> 2 ms 就别靠切步。根治在 prefill 步 GEMM 效率。

## 21. decode 调出来的 GEMM tile 不要直接给 prefill 用：M > 1000 行时 BM=32 tile 只有算力峰值的 1/4（2026-10-29 kaon3 D211/D212）
- MoE grouped NVFP4 GEMM 为 decode 优化到 BM=16/32（CTA 数多、单 CTA 工作少），prefill 17184 行时同一个 kernel 724 us = 190 TFLOPS；换 BM=64×BN=64、每 warp 2×2 片段（A 片段 n 向复用 4 次）→ 549 us（−24%）、端到端 c64 TPOT −4% / TTFT −8%，逐位一致（k 累加序不变）。
- **判断**：grouped GEMM 至少两套 tile（decode 小 M 占用型 / prefill 大 M 算力型），按 R 阈值分派；两套的 k 累加序保持一致才能逐位互换。微基准（`moe_gemm_timing` P424/P2148）+ 两进程 CHECKSUM 是门禁。再往上是 BN=128 / ldmatrix，smem 片段读是下一瓶颈。

## 22. cuBLASLt 启发式第 0 候选对 prefill 大 M 不是最快：m ≥ 64 实测选 algo、按 m 2 的幂桶缓存（2026-10-29 kaon3 D215）
- fp8 Lt 为逐位对齐 vLLM 锁死的 `tile=13` 配置在 M=2148 走 nvjet 64x32 小 tile（153 us × 125 次/步）。m ≥ 64 对 32 候选各计时 5 次取最快 → c64 TPOT −7%、TTFT 全档 −15%、吞吐 +9%；m < 64 保持 parity 配置不动，输出逐字同。
- **判断**：对拍口径的 algo 只锁 decode（m < 64）；prefill 形状必须实测。缓存 key 用 m 的 2 的幂桶（混批步 m 每步不同，逐 m 调会每步付 32×5 次计时）。调优应挪到启动 warmup（按桶 × 形状预跑），别在首批请求里付。

## 23. "prefill 副流 ‖ decode 主流图" 在 prefill 占满 SM 时没有并行收益，单流混批反而略好（2026-10-29 kaon3 b56 D224）
- 同二进制 ABAB：`XSTREAM=0`（单流 Paged 混批，decode 行并进 prefill 的 eager 前向）比 `XSTREAM=1`（pair 步）全档 TPOT 低 0.4–1.3%、吞吐高 1–2%。prefill 2148 tok 的 GEMM 把 170 SM 填满，decode 图只能排队；两流还多付事件同步 + 一套 AltSet 显存。
- **判断**：跨流并行只在"小 kernel 填不满 SM"时有收益（Q3 qwen3 小模型是这种）；大 M prefill 场景下要并行就得让 prefill **让 SM**（低优先级流 + 限 grid / 分块 launch），否则不如单流。先量 prefill 步的 SM 占用再决定两流。

## 24. 自写 gemv 的优势区间要按 M 实测画边界，别从 m=8 外推到 m=16（2026-10-29 kaon3 D227/D228）
- b31b 时只量了 m=8（gemv 比 Lt 快 1.8–5×）就把 gemv 门槛设成 `tokens ≤ 16`；nsys c16 发现 gemv<16> 31.7 us，而 cuBLASLt tile=13 在 m=16 只要 14.5 us（qkv）/ 8.6 us（o_proj split-K 3）。改门槛 8 + Lt 自调优 m ≥ 9 → c16 TPOT −7.6%。
- **判断**：每条"小 M 快路径"的 M 上界必须在每个 2 的幂桶（1/2/4/8/16/32）上与 Lt 自调优实测比过；只量过一个点的门槛写进 loop 时标"仅 m=8 实测"。M=16 的 decode 步里 MoE（2.8 ms）已到带宽下限，剩下可拿的全在小 M dense GEMM：qkv/o（B51，−7.6%）+ dense MLP（B53，−3.5%）两刀把 M=9..16 全部交给 Lt 自调优，gemv 只留 m ≤ 8。

## 25. 混批步耗时 ≈ prefill token 数 ÷ 单步 prefill 吞吐，与并进的 decode 行数无关（2026-10-29 kaon3 D236）
- HOST_TIME 按 (tokens, nreq) 分组：3800 tok 的混批步 11 / 23 / 55 个请求都是 95–108 ms；decode 行从 1 到 55 只加 ~5 ms。c16 混批步 2148 tok 59 ms = 37 tok/ms。
- **怎么用**：TPOT − 纯 decode 步 = 混批步摊入量，拆成两半：(a) prefill 单步吞吐差（我们 37 vs vLLM 50 tok/ms）、(b) decode 行在混批步里停转。(a) 靠 GEMM；(b) 靠异步收割。两半各占一半，别只做一边。

## 26. "decode 不等 prefill" ≠ "decode 不被 prefill 拖"：大 M prefill GEMM 把 SM 驻留槽占满时，另一条流上的 decode 图照样排队（2026-10-29 kaon3 E3）
- E1 异步收割机制完全正确（late_begin == late_harvest、输出同），TPOT 却持平 / 略升：p64 一次 launch 4752 CTA、每 SM 只驻 2 个（smem 49.5 KB）、14 波，decode 图的 800 个小 kernel 在 pending 队列里等 CTA 槽。两流并行的前提是**有空 SM**，不是"有另一条流"。
- **怎么用**：做跨流并行前先算对方 kernel 的 CTA 数 / 驻留数 / 波数；波数 ≫ 1 时要么流优先级（只影响 pending 顺序，CTA 边界抢占，最多省一个 CTA 时长的等待），要么限对方每次 launch 的 CTA 数留口子，要么别做。E4 验证流优先级够不够。
