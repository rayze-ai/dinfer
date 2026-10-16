# loop 规格：Q3 单卡性能（Qwen3-0.6B，4090，对照 vLLM v0.31.0）

- 状态：进行中（2026-10-16 开）
- 验证脚本：`scripts/verify/q3.sh`（待写；先用 `scripts/verify/bench_compare.sh` + kernel ref 测试）
- 迭代记录：`test/qwen3/2026-10-16/q3-perf.md`

## 目标

在同卡、同模型、同配置下，dinfer 的 TTFT / TPOT / 吞吐**不低于** vLLM；先做单卡 mix 实例（无 PD）。基线见 `test/qwen3/2026-10-16/bench/summary.md`：
c1 out 221 vs 447 tok/s、c32 2707 vs 8065、c8 p1024 56 vs 1916；TTFT 84 vs 18 ms（p171）、TPOT 3.9 vs 2.1 ms。

## 范围（顺序按用户 2026-10-16 建议：先补齐 CUDA graph 等结构性功能并与 vLLM 对齐，逐段比耗时；kernel 合并最后做）

| 块 | 内容 | 现状 → 目标 | 状态 |
|---|---|---|---|
| a prefill attention | flash 风格 tile kernel（prefill + paged 两条路径，D=128） | 1514 token 6.36 s → 0.27 s | ✅ 2026-10-16 |
| b decode attention | split-K kernel（reqs × Hkv × 8 split × 4 warp + merge） | nsys 85% → 待 bench4 | 实现完，验收中 |
| c CUDA graph | ① 固定地址输入缓冲：runner 每 step 的 ids / positions / slot_mapping / block_table / seq_lens / 采样参数一次 H2D 到预分配 pinned→device 缓冲（替代每 step 7 次 `Tensor::from_vec`）；② decode 步（纯 Decode 模式）整步 forward + 采样捕获为 CUDA graph，按 batch 大小分桶 {1,2,4,8,16,32,64}，启动时捕获；③ 图内采样：penalties/temperature/min_p/topk-topp/sample/logprobs 全部按固定地址运行；非 decode 步照常 eager。与 vLLM 对齐：`cudagraph_mode=FULL`、`cudagraph_capture_sizes` | c1 TPOT 3.8 → ≤ 2.5 ms；c32 TPOT ≤ 5 ms | ✅ c1 2.0 ms（2026-10-16）；c32 5.7 未达 → d 分析 |
| d 逐段对齐与耗时比对 | nsys 按阶段并排 + executor 内 upload/launch/wait 分段 + engine 分段直方图 | `test/qwen3/2026-10-16/stage/` | ✅ 结论：GPU-bound（CPU 每步 70 µs）；TTFT 差距是 Nagle（已修）；余量 = 480 kernel/step 的融合 + decode attention 并行度 |
| e kernel 合并（最后） | rmsnorm+残差、rope+cache write、qkv 拆分并入 GEMM epilogue、采样管线一个 kernel、logprobs 不复制 raw | 每 step 省 ~0.5 ms | 待开 |
| f 激活复用 | per-layer scratch → 全模型共享一组 | 省 ~2.5 GiB → KV 更多 | 待开 |
| g 权重加载 | mmap 页 pinned + 异步 H2D / 多线程按张量并行 | 2.0 s → < 1 s | 待开 |

不在 Q3：PD 分离（Q4）、fp8、多卡、Triton。

## e 续：decode attention v2 设计（下一轮，写在前面）

现状（`cold_shape`，4090）：split-K decode B=32 seq=2048 1.24 ms、B=64 seq=1024 1.09 ms；c32 TPOT 5.7 ms 里 attention ≈ 28 层 × 40 µs ≈ 1.1 ms。问题：
1. 每个 key 的 K 行 128 元素由一个 lane 串行读（`for d<128: s_q[d]*k[d]`），bf16 逐元素 load，无向量化；每 key 还查一次 `block_table`。
2. V 累加时每个 key 一次 `__shfl` 广播 + 4 次 bf16 标量读。
3. B 小时 grid = B×8×8 个 block，B=1 只 64 个 block（128 SM 半空）。
改法：① K/V 用 `bf16x2`/`uint4`（8 个 bf16）向量读，一 warp 一次读 4 个 key 的 K（lane 分工：8 lane 一 key，每 lane 16 维，warp 内 shuffle 归约）；② block_table 每 16 key 查一次（block_size 对齐）；③ splits 随 B 自适应：`splits = clamp(256 / (B×Hkv), 1, 16)`。
验收：`gqa_decode_d128_splitk_ref` 仍过；`cold_shape` B=32 seq=2048 ≤ 0.6 ms；bench c32 TPOT ≤ 4.5 ms。

## e 续 2：剩余融合候选（decode 每 step，28 层 × …）

| 候选 | 现状 | 做法 | 预期 |
|---|---|---|---|
| silu_and_mul → down GEMM | gate_up GEMM → silu_and_mul kernel → down GEMM（3 launch/层） | cuBLASLt epilogue 只有 RELU/GELU，无 SwiGLU；可做：silu_and_mul 写成 down GEMM 的 A 侧预处理不可行（GEMM 要读两遍）。保留；或用 cuBLASLt 的 `GELU_AUX`？不是 SiLU → 放弃。 | 0 |
| decode merge kernel | split kernel + merge kernel（2 launch/层） | splits==1 时跳过 merge 直接写 out（B×Hkv ≥ 256 时就是 1）；否则并入 o_proj 前的 gather | B≥32 时 -28 launch |
| 采样 4 kernel + 1 dtod | temperature / min_p / topk_topp / sample 各一 launch + logits 复制 | 一个 `sample_fused` kernel：每行一 block，读原始 logits，依次做温度、min_p、top-k/p、Philox 采样，不写回 work（logprobs 时才复制） | -4 launch/step + 省 1 次 [B,V] fp32 复制 |
| fused_add_rmsnorm 之前的 o_proj 输出 | o_proj 写 attn_out 再 fused_add 读 | cuBLASLt beta=1 直接累加进 hidden（C = A·B + hidden），再普通 rmsnorm | -0（kernel 数同），省一次读 |

## e 续 3：prefill attention v2 设计（c8-long TTFT 488 ms 的来源）

分解（3090 `cold_shape`，T=1514）：flash prefill **9.6 ms/层**，三个 GEMM 合计 0.8 ms/层 → 28 层 ≈ 270 ms attention + 22 ms GEMM；T=1139 时 5.6 ms/层。即 c8-long 一个 1.2k prompt 的 prefill ≈ 160–270 ms 几乎全是 attention；8 个同时到达、4096 token/step 预算 → 3 个 prompt 一步、其余排队，TTFT p50 488。vLLM（FlashAttention-2）同 T 约 0.3 ms/层。
现 kernel 的问题：① 每 block 只有 4 个 query 行（PF_WARPS=4）→ K/V tile 从全局内存读进 smem 后只服务 4 行，T=1514 时每 block 读 48 个 tile × 32 key，整体 K/V 重读 T/4 ≈ 380 次；② 装载用 bf16 标量读 + 转置写 smem；③ 分数计算每 lane 串行 128 维 FMA（无 tensor core）。
v2：① BQ = 64 行/block（16 warp，每 warp 4 行，或 8 warp × 8 行），K/V 重读次数降 16×；② uint4 向量装载；③ 同一 kv head 的两个 q head 共用 K/V tile（grid 按 kv head 而非 q head，再省一半读）；④ 分数/PV 用 `mma.sync` bf16 m16n8k16（tensor core）—— 这是 FA2 的本质，第二步做；先做 ①②③（预期 9.6 → ~1.5 ms/层），再做 ④。
验收：`gqa_prefill_d128_flash_ref` + `gqa_paged_d128_flash_ref` 过（paged 同样改）；cold_shape T=1514 ≤ 2 ms；bench c8-long TTFT ≤ 150 ms。

## e 续 4：prefill v3（tensor core）设计

现状（4090，v2）：T=1514 1.3 ms/层 × 28 = 36 ms/请求；c8-long 8 个请求排队 → TTFT 127 ms（vLLM 36）。v2 的分数 / PV 仍是标量 FMA（每 lane 128 次），只用了 CUDA core。
v3：block 仍 = (req, kv head, 32 行 query)，但 32×32 的 S = Q·Kᵀ 与 O += P·V 用 `nvcuda::wmma` bf16 16×16×16 片：
- S：Q [32×128] bf16（smem，预乘 scale）× Kᵀ（smem K [32×128] 行主 = col_major Kᵀ）→ 2×2 个 16×16 fp32 累加片，k 维 8 步；8 warp 分 4 个 S 片 × 2 个 head（每 warp 一片）；
- S 写回 smem [2 head][32][32] fp32 → 每 warp 负责 8 行做 online softmax（max/sum、alpha、P=exp）→ P 写 bf16 smem [2][32][32]；O 累加器 [2][32][128] fp32 在 smem，整体 × alpha（按行）；
- PV：P [32×32] × V [32×128] → 2×8 个 16×16 片，k 维 2 步；8 warp 分 16 片 × 2 head；累加到 smem O。
smem：K 8K + V 8K + Q 2×8K + S 2×4K + P 2×2K + O 2×16K = 72 KB → 动态 smem + `cudaFuncSetAttribute` 到 96 KB（4090 每 SM 100 KB，1 block/SM；或 O 放寄存器以后再压）。
验收：ref 同（1e-2）；4090 T=1514 ≤ 0.4 ms/层；bench c8-long TTFT ≤ 60、吞吐 ≥ 1.0×。

## 环境

4090（`scripts/dev/remote4090.sh`），dinfer `--release` 常驻 :8012（tmux `dinfer`），vLLM Docker :8013。改 kernel 后：`push` → 远端 `cargo build --release -p dinfer --features qwen3` → 重启 tmux `dinfer` → `bench_compare.sh`。

## 每轮记录要求（用户 2026-10-16）

每轮 bench 都用 `bench_compare.sh <OUT> "改动一句话"` 跑，`CHANGE` 进目录；`scripts/dev/bench_trend.py` 汇总所有轮次到 `test/qwen3/perf-trend.{md,json,svg}`：每组 tok/s（×vLLM）、TTFT、TPOT 的绝对值与相对上一轮的百分比，加三张曲线。改动 → 指标提升一一对应，不许只留最新一轮。

## 观察方法

| 看什么 | 怎么看 |
|---|---|
| 逐段耗时 vs vLLM（块 d） | 两边各起一个独立实例在 GPU 3 上跑同一 bench 负载，`nsys profile --trace=cuda --cuda-graph-trace=node`（图内 kernel 逐个计时）；vLLM 容器内无 nsys → 宿主 `nsys profile docker run ...`（nsys 注入 CUDA 层到子进程）或装 `nsys` 的 wheel；`scripts/dev/nsys_stage_compare.py a.csv b.csv --steps Na Nb` 出并排表 |
| kernel 正确性 | `KERNEL_REFS_DIR=… cargo test -p dinfer-kernels --features ref-tests --test ref_tests gqa -- --ignored`（与 torch 对拍，atol 见测试） |
| 端到端正确性 | `real_parity`（greedy 32 步 = HF）、`sampling_parity`、`serve_e2e_qwen3`（q2.sh 全套） |
| 单 kernel 耗时 | `cargo test --release -p dinfer-kernels --test cold_shape -- --ignored --nocapture`（GEMM / attention 按形状） |
| 服务指标 | `bench_compare.sh` 四组 + `/metrics` 的 `engine_step_seconds` 直方图 |
| GPU 侧 | `nsys profile`（4090 有 nsys/ncu）看 kernel 占比 |

## 停止条件（机器判定）

1. 每块改完：kernel ref 测试全过（对拍 atol 不放宽）；`real_parity` 3/3、`sampling_parity` 5/5、`serve_e2e_qwen3` 6/6 仍过。
2. a：`cold_shape` 中 gqa_prefill T=1514 单 head-set ≤ 30 ms；`bench_compare` c8 p1024 TTFT p50 ≤ 150 ms。
3. b：c32 p128 TPOT p50 ≤ 5 ms。
4. c：c1 p128 TPOT p50 ≤ 2.5 ms。
5. Q3 总验收：四组 out tok/s 均 ≥ vLLM 的 1.0×（`scripts/verify/q3.sh` 读两边 JSON 比较，exit 0）。

## 上限

每块 6 轮；总 3 个工作日。达不到 1.0× 的组写清差在哪个 kernel，交给用户决定是否继续。

## 已知可能出错点

1. online softmax 的数值：fp32 累加 + bf16 输出，与 torch SDPA 的差异在 1e-2 内（现有 ref 测试口径）；causal 边界 tile 的 mask 要按绝对位置。
2. GQA：heads/kv_heads = 2，一个 kv head 服务两个 q head；tile K/V 到 shared 时可以让同一 block 处理同 kv head 的两个 q head 以省一半带宽（第二版再做）。
3. split-K decode 合并时 m/l 的数值稳定性（log-sum-exp 合并）。
4. CUDA graph 要求所有地址固定：`Tensor::from_vec` 每步新分配 → 必须先做 d。
5. 对拍中 bf16 near-tie 翻 token 是已知噪声（pitfalls），不能用"逐 token 相同"当 kernel 正确性标准，用 ref 测试的 atol。

## 记录

| 轮 | 块 | 现象 | 根因 | 处理 |
|---|---|---|---|---|
| 0 | 基线 | `bench/summary.md`；1514 token prefill 6.36 s（prefix 不命中）vs vLLM 0.04 s | `gqa_prefill_kernel` 16 个 block 串行遍历 T 个 query、每 query 三遍扫 key | 块 a 开工 |
| 1 | a prefill attention | 重写为 flash 风格 tile kernel（grid ceil(T/4)×H×reqs，block 4 warp 各一行 query，K/V 32-key tile 进 smem、K 转置，online softmax，causal 只扫到对角 tile；D=128 专用，其他 D 仍走参考 kernel）；新增 D=128 torch 对拍 `gqa_prefill_d128_flash_ref`（3 请求 70/33/5）。3090：gqa 3 个 ref 测试全过 | 旧 kernel：16 block 串行 T 行、每行三遍扫 key；`cold_shape` 实测 T=1514 单次 224 ms × 28 层 = 6.3 s | 4090 验收中：cold_shape / ref / real_parity / bench |
| 2 | a 验收（进行中） | 4090 release：`cold_shape` gqa_prefill T=1514 **224 ms → 4.5 ms**（50×），T=1900 353 → 6.9 ms；gqa ref 3/3。real_parity 3 个全 FAIL | FAIL 是 `CUDA_ERROR_OUT_OF_MEMORY`：测试默认 device 0，与常驻 dinfer（0.9 利用率、14.5 GiB KV）同卡 | 测试改默认 GPU 2（脚本 `CUDA_VISIBLE_DEVICES`），重跑；服务重启到新 kernel 后再 bench |
| 3 | a 验收 | 服务重启到新 prefill kernel：单请求 1515 token 冷 prefill **6.36 s → 0.27 s**；启动 3.1 s（tokenizer 0.4 / CUDA 0.66 / 权重+KV 2.0）。但 bench c8 p1024 组仍 TTFT 9.4 s、c1 TTFT 84 → 43 ms | 并发下 scheduler 把"新 prompt 的 prefill"与"别人的 decode"组成混批 → `BatchMode::Paged` → 走的是旧 `gqa_paged_kernel`（同样每 (req,head) 一 block、逐行三遍扫），prefill kernel 根本没被调到 | 把 paged 路径也换成 flash tile kernel（读 paged cache），加 D=128 paged torch 对拍 |
| 4 | a 验收（进行中） | paged 路径换 flash tile kernel；3090 gqa ref 4/4；4090 跑 ref / parity / 重启服务 / bench3 | — | 等结果 |
| 5 | a 验收 | bench3（prefill + paged 均为 flash kernel）：c8 p1024 **TTFT 9373 → 495 ms，out tok/s 56 → 348（6.2×）**；c32 TTFT 366 → 92 ms、tok/s 2707 → 3481；c1 TTFT 84 → 44 ms。对 vLLM：c1 0.54×、c8 0.50×、c32 0.41×、c8 p1024 0.17×。gqa ref 4/4；real_parity 3/3（3090）；4090 GPU 2 又被一个 VLLM::EngineCore 占满 → 4090 parity OOM（追查中） | 停止条件 2 的 ≤150 ms 未达（495）：剩余差距要看 nsys 分解 | nsys 分析 → 下一轮 |
| 6 | b decode attention | nsys（GPU 3 独立实例，c8 p1024 负载）：**gqa_paged_flash_kernel 85.4% GPU 时间**（6407 次、每次 825 µs，即 decode 步每层一次），GEMM 合计 ~7%，prefill flash 2.5%，其余小 kernel < 1% 各。memcpy 总共 1.6 ms 可忽略。决定：decode 单独走 split-K kernel（grid reqs × Hkv × 8 split × 4 warp，K 每 key 读一次服务 2 个 q head，merge kernel 做 LSE 合并）；加 D=128 decode torch 对拍（300/70/1） | decode 时 flash paged kernel 每 (req, head) 只有 1 个活跃 warp 串行扫全部 key，GPU 占用极低 | 实现中 |
| 7 | b 验收 | 4090：gqa ref 5/5、real_parity 3/3、sampling_parity 5/5（GPU 3）。bench4：c1_p128_o128: tok/s 327/456 = 0.72×, TTFT 43/11 ms, TPOT 2.7/2.1 ms；c8_p128_o128: tok/s 2283/2944 = 0.78×, TTFT 44/25 ms, TPOT 3.2/2.5 ms；c32_p128_o128: tok/s 5048/8536 = 0.59×, TTFT 99/65 ms, TPOT 5.8/3.3 ms；c8_p1024_o128: tok/s 615/2103 = 0.29×, TTFT 497/36 ms, TPOT 9.2/3.5 ms。相对 bench3：c1 TPOT 3.8→2.7、c32 吞吐 3481→5048、c8 p1024 吞吐 348→615 | — | b 完成（c32 TPOT 5.8 未到 ≤5，交给 c CUDA graph 继续）；按用户顺序下一块 c |
| 8 | c CUDA graph 探针 | `graph_probe.rs`：直接 capture 报 `CUDA_ERROR_STREAM_CAPTURE_ISOLATION`；根因 cudarc 0.18 每次取设备指针都 record/wait 事件（跨流同步管理），capture 期间非法。`Device::disable_event_tracking()` 后 rmsnorm + cuBLASLt GEMM 捕获成图、重放 3 次与 eager 逐元素相同（diff 0）。结论：我们单流顺序执行，关掉事件跟踪即可捕获；cuBLASLt 启发式查询是 host 侧，capture 兼容 | — | 进入实现：固定地址输入 + 按 batch 桶捕获 decode 整步 |
| 9 | c 实现 | executor 重写：`StepBuffers` 固定地址（ids/pos/slots/rows/cu/seq/bt + 全部采样列，按 max_tokens / max_logits_rows 上限一次分配，每 step 各一次 `memcpy_htod_async`，去掉每 step 7 次 `Tensor::from_vec`）；executor 自建非默认流 + 该流上的 cuBLASLt；`Forward{t, rows}` 显式计数；纯 decode（无惩罚 / 无 logprobs）按桶 {1,2,4,8,16,32,64,…,max} 捕获 forward+temperature+min_p+topk/topp+sample 为图，不满桶 padding（slot=-1、seq=1、温度 0）；`--no-cuda-graph` 可关；`Graph: Send` 封装放 kernels crate（qwen3 crate forbid unsafe） | — | 本机 parity → 4090 bench5 |
| 10 | c 本机验收 | 3090：`graph_probe` PASS；`real_parity` 3/3（greedy 32 步全走捕获图，32 步 0.10 s ≈ 3.1 ms/步，比图前 3090 的 ≈7.5 ms/步快一倍；两请求并发 34 步 0.11 s）。logits 与 HF 一致性不变 | — | 4090 parity + bench5 进行中 |
| 11 | c 验收 | 4090：real_parity 3/3、sampling_parity 5/5（GPU 3）；服务启动捕获 7 个桶 {1..64} 共 222 ms（startup 3.4 s）。bench5：c1_p128_o128: tok/s 423/455 = 0.93×, TTFT 43/12 ms, TPOT 2.0/2.1 ms；c8_p128_o128: tok/s 2548/2943 = 0.87×, TTFT 43/26 ms, TPOT 2.9/2.5 ms；c32_p128_o128: tok/s 5166/8577 = 0.60×, TTFT 92/61 ms, TPOT 5.7/3.3 ms；c8_p1024_o128: tok/s 630/2104 = 0.30×, TTFT 497/36 ms, TPOT 8.8/3.5 ms。相对 bench4：c1 TPOT 2.7→2.0（**已低于 vLLM 2.1**）、c1 吞吐 327→423；c8 TPOT 3.2→2.9；c32 / c8p1024 基本不变（混批走 eager、prefill 占主导） | 剩余差距：① TTFT 43 vs 12 ms（prefill 本身 + 调度）；② c32 TPOT 5.7 vs 3.3（batch 大时 decode 图内 kernel 慢，attention/GEMM 规模效应）；③ c8 p1024 TTFT 497 vs 36（混批 eager + prefill kernel） | c 完成；d 逐段 nsys 对比进行中 |
| 12 | d 逐段对比 | `stage_profile.sh` 跑通（vLLM 用容器内 nsys）：GPU 时间 dinfer 2.9 vs vLLM ≈3.7 ms/step（剔启动期 Fill），attention 持平、GEMM 里 gemv 占 65%、norm 多一倍、copy_cols 0.15 ms；但端到端 TPOT 3.9 vs 2.7 → **差距在 CPU/GPU 不重叠（同步 execute）**。HTTP 侧 CPU ≈ 0.3 ms 可忽略 | 同步 step 流水；小 M 走 gemv | 下一步：异步 step 流水（双缓冲、D2H 不阻塞下一步调度）；GEMM 小 M 强制 tensorop；再做 e 融合 |
| 13 | d TTFT 拆解 | 174-token prompt：engine 内 TTFT 5.0 ms（首次 14.4 ms 含 cuBLASLt 启发式），非流式 wall 6 ms；HTTP 侧模板 0.02 + 编码 0.3 + 解码 µs。但 bench（流式、连接复用）TTFT 稳定 43 ms（p50=p99），vLLM 12 ms | 38 ms 差额与 Nagle + 延迟 ACK（40 ms）吻合：SSE 首个 role chunk 和首个内容 chunk 都是小写入，复用连接上 quickack 关闭后第二个小段被攒 40 ms；uvicorn 默认 TCP_NODELAY，axum 默认不设 | 服务端 `tcp_nodelay(true)`，重测 |
| 14 | d GEMM 小 M 微基准 | 3090 `cold_shape`：qkv/gate_up/down GEMM 在 M=1,2,4,8（gemv 路径）与 M=16（tensorop）耗时相同（0.016 / 0.020 / 0.014 ms）→ 小 M 是权重带宽下限，gemv 不是问题，stage README 已改口 | — | GEMM 无算法层余量，只剩减 kernel 数（e 阶段） |
| 15 | d Nagle | axum 0.8.4 `Serve` 无 `tcp_nodelay()`；用 `ListenerExt::tap_io` 对每个 accept 的 socket `set_nodelay(true)`；engine 加 step 分段直方图（schedule / execute / update） | — | 推 4090 → bench6 看 TTFT |
| 16 | d Nagle 验证 | bench6（TCP_NODELAY）：**TTFT c1 43 → 6.5 ms（vLLM 12.4）、c8 43 → 12.5 ms（vLLM 24.8）**，吞吐不变。c1_p128_o128: tok/s 423/455 = 0.93×, TTFT 6.5/12.4 ms, TPOT 2.3/2.1 ms；c8_p128_o128: tok/s 2542/2948 = 0.86×, TTFT 12.5/24.8 ms, TPOT 3.0/2.5 ms；c32_p128_o128: tok/s 5165/8498 = 0.61×, TTFT 92.1/65.4 ms, TPOT 5.7/3.2 ms；c8_p1024_o128: tok/s 630/2100 = 0.30×, TTFT 492.5/37.2 ms, TPOT 8.9/3.5 ms。/metrics 分段：step 4.10 ms = schedule 0.001 + execute 4.100 + update 0.001 → **全部时间在 execute**（含 H2D、GPU、D2H 同步） | Nagle 确认是 TTFT 差距的全部；现在 TTFT 已优于 vLLM。c32 / c8p1024 的差距全在 execute 内部 | 下一步拆 execute：H2D（7 次 memcpy）/ GPU（图 ≈2.9 ms）/ D2H 同步；做异步流水 |
| 17 | d 分段实测 | executor 分段（4090，图路径）：upload 41 µs、launch 29 µs、**wait 2290 µs**/step（c1）；engine schedule/update 各 1 µs。c1 TPOT 2.33、c8 2.99、c32 5.50 ms（图内 GPU 时间随 batch 线性涨：attention split kernel + GEMM M 增大） | **GPU-bound**，不是 CPU/GPU 串行（首版判断更正，README 已改）。异步流水最多省 3%，放弃 | 转 e：删 copy_cols（attention 按 stride 读 qkv_buf）、q/k norm 合一、rope+cache_write 合一、残差加并入 norm；decode split kernel 改 2 warp 一 key 减少 block_table 重复读 |
| 18 | e kernel 融合（实现） | `qkv_post_bf16`：一个 kernel 做 q/k per-head RMSNorm + neox RoPE（原地写回 qkv GEMM 输出）+ K/V 写 cache（替代每层 copy_cols×4 + rmsnorm×2 + rope + cache_write = 8 个 kernel）；GQA 三 kernel 加 q_stride/kv_stride 直接按跨度读 qkv_buf（省 q/k/v 三个 scratch）；MLP 残差加并入下一层 input_norm 的 fused_add（省 add_inplace）。每层 kernel 数 17 → 8；torch 对拍 `qkv_post_fused_ref`（T=6，含 slot=-1 行） | — | 3090 ref/parity → 4090 bench7 |
| 19 | e 本机验收 | 3090：20 个 kernel ref 全过（含新 `qkv_post_fused_ref`）；real_parity 3/3（prefill logits worst 0.81、greedy 32 = HF、并发混批 = HF）、sampling_parity 5/5（L1 0.085）。融合路径数值与 HF 一致 | — | 等 4090 bench7 |
| 20 | e 验收 | 4090：refs 20/20、parity 8/8；bench7（e 融合）：**c1 471 tok/s = 1.04× vLLM（首次反超）、TPOT 2.1 = vLLM**；c8 2755 (0.94×)；c32 5788 (0.68×, TPOT 5.0)；c8-long 645 (0.31×)。相对 bench6：c1 +11%、c8 +8%、c32 +12%、c8-long +2%；TTFT 6.0 / 11.5 / 88 / 489 | — | 剩余：c32 TPOT 5.0 vs 3.3（decode attention 随 batch 线性）、c8-long TTFT 489 vs 38（混批 eager + prefill 占主导）。下一轮 decode v2（已写完，本机测试中） |
| 21 | e 续 decode v2 | 8 lane/key + uint4 向量读 K/V + 4 key/warp + 两 q head 共享 K 读 + splits 自适应（≥256 block）。首版死锁（分叉里的全 mask shuffle，pitfalls）；修后 ref 过；3090 `cold_shape`：B=1 0.027→0.019、B=8 0.150→0.095、B=32 1.237→0.331（3.7×）、B=64 1.093→0.335 ms | — | 4090 parity + bench8 进行中 |
| 22 | e 续 merge 跳过 | splits==1（B×Hkv ≥ 256，即 B ≥ 32）时 v2 kernel 直接归一化写输出，不启 merge；新增 B=32 随机长度 torch 对拍 `gqa_decode_d128_b32_direct_ref`，3090 两条 decode ref 过 | — | 随 bench8 之后一并推 4090（bench9） |
| 23 | e 续 验收 | 4090 bench8（decode v2）：**c1 499 tok/s = 1.09× vLLM、c8 3022 = 1.04×、c32 7659 = 0.90×（TPOT 5.0→3.7）、c8-long 795 = 0.38×（TPOT 8.6→6.3）**；相对 bench7：c1 +6%、c8 +10%、c32 +32%、c8-long +23%。refs 20/20、parity 8/8。4090 cold_shape B=32 attention 1.24→0.30 ms | — | 剩余：c32 TPOT 3.7 vs 3.3；c8-long TTFT 488 vs 35（prefill 占主导，混批 eager）。merge 跳过已写（bench9） |
| 24 | e 续 验收 | bench9（merge 跳过）：与 bench8 持平（c1 497 / c8 3013 / c32 7691 / c8-long 795；差异 <1% 在抖动内）—— merge kernel 本来只 1.2 µs × 28，省不出可见量；保留（少 28 launch 无害）。c8-long TTFT 593（p50 抖动，p99 同 835） | — | 转 prefill v2（本机已过 ref：T=1514 9.6 → 2.6 ms） |
| 25 | e 续 3 prefill v2 | prefill + paged flash kernel 重写：block = (req, kv head, 32 行)，8 warp × 4 (row, head) 组合，两 q head 共享每个 K/V tile（全局读 /16），uint4 装载、smem 存 bf16 不转置。3090 `cold_shape` T=389/889/1514/1900：0.26/1.04/2.6/4.1 ms（v1 0.38/1.6/4.5/6.9 → 3090 实测 v1 为 17/78/224/353 ms 的是最老版本；v1 flash 在 3090 未单测）；gqa ref 6/6 | — | 4090 chain → bench10 |
| 26 | e 续 3 验收 | 4090 bench10（prefill/paged v2）：refs 21/21、parity 8/8；cold_shape prefill T=1514 **1.3 ms**（4090；v1 4.5）。**c8-long TTFT 488 → 127 ms、吞吐 795 → 1543（0.74× vLLM）；c32 TTFT 88 → 41、吞吐 7691 → 8484（0.99×，TPOT 3.5 vs 3.3）；c8 3097（1.05×，TTFT 9.4 vs 25.6）；c1 500（1.10×，TTFT 4.8 vs 12.1）** | — | 停止条件 2（c8-long TTFT ≤150）达成；c8-long 吞吐 0.74× 是最后一组 <1.0；其 TTFT 127 vs 36 仍是 prefill attention（4090 1.3 ms/层 × 28 = 36 ms/请求，8 个排队） → prefill v3 tensor core |
| 27 | e 续 4 prefill v3 | WMMA bf16 16×16×16 做 S=QKᵀ 与 O+=PV（动态 smem 76 KB，`cudaFuncSetAttribute` 一次）；online softmax 每 warp 8 行在 smem 上做。3090 `cold_shape` T=389/889/1514/1900：0.16/0.54/1.34/1.99 ms（v2 0.26/1.04/2.6/4.1，再 2×）；prefill ref 2/2 过。paged 同样改 v3 | — | 本机 parity → 4090 bench11 |
| 28 | e 续 4 验收 | 4090 bench11（prefill/paged v3 WMMA）：**c1 500 = 1.10×、c8 3101 = 1.05×、c32 8521 = 1.00×（首次追平）、c8-long 1728 = 0.83×（+12%）**；TTFT c1 4.7 / c8 8.4 / c32 38.6 / c8-long 84.6 ms（vLLM 11.8 / 25.0 / 66.6 / 41.4）；TPOT c8-long 4.0 vs 3.5。bench 本身完整，上个会话断在回传前（第 4 次 Input too long） | c8-long 差距已不在 prefill kernel（TTFT 127 → 85 只少 1/3，kernel 2.6 → 1.34 ms/层 应省 ~35 ms/1024 token）：剩余在调度（8 路 1024 prompt 一起到，`max_num_batched_tokens 4096` 下排队 2 轮；且 prefill 混批打断 decode → TPOT 4.0） | ① 4090 nsys/分段看 c8-long 一步 prefill 批的墙钟；② 调度：prefill 批与 decode 分离 / 预算调整 |
| 29 | 口径修正 | bench11 同 seed 复跑：dinfer c8-long TTFT 85 → 17.7 ms、vLLM 41 → 38；`/metrics` vLLM prefix cache 命中 90%、dinfer 48% | bench 固定 seed=42，常驻 vLLM 把历史 prompt 全缓存了，dinfer 每轮重启是冷的 → 历史 vLLM TTFT 偏低（pitfalls） | `bench_compare.sh` 每次换 seed（写进 summary）。**bench12（新口径，代码同 bench11）：c1 499 = 1.12×、c8 3095 = 1.06×、c32 8524 = 1.04×、c8-long 1727 = 0.91×**；TTFT 5.0/9.5/38.6/84.6 vs vLLM 17.2/28.4/78.7/68.0；TPOT c8-long 4.0 vs 3.7。bench≤11 的 TTFT 列不可与 bench12+ 直接比 |
| 30 | c8-long 分解 | 4090 GPU 3 独立实例 nsys（c8 p1024 o128 ×32，`stage/c8long_kern.csv`）：GPU 总 2857 ms；cutlass wmma GEMM 27.6%（52565 次 15 µs）、**gqa_decode_v2 25.7%（20133 次 36.5 µs）**、gemv 13.2%、**gqa_paged_tc 11.7%（420 次 794 µs）**、gqa_prefill_tc 1.3%（84 次 431 µs）、silu 3.6%、qkv_post 3.3%、rmsnorm 3.1% | ① 32 个 1024-prompt 中 420/28=15 个走了 paged 路径（混批：与别人 decode 同步）、只有 3 个走纯 prefill；paged_tc 每 token 比 prefill_tc 慢 ~1.8×（794 vs 431 µs，token 数相近）→ TTFT 双峰 85/141；② decode_v2 在 seq≈1100、B=8 时 36.5 µs/层（B=8 × Hkv=8 = 64 block，splits 自适应后仍不满 128 SM）→ TPOT 4.0 vs 3.5 | 下一步：a) paged_tc 内核对齐 prefill_tc（block table 读取的 K/V tile 装载应与连续路径同速，查 tile 装载与 causal 跳过）；b) decode_v2 长序列 split 数 |
| 31 | f decode splits | decode_v2 splits 目标 256 → 1024 block（只依赖 batch，图安全）：B=8 时 splits 2 → 8（64 → 512 block），B=32 时 1 → 4 | c8-long nsys：decode_v2 36.5 µs/层 占 25.7%，B=8 × Hkv=8 × splits 2 = 128 block ≈ 1 block/SM，长序列每 block 串行扫 550 key | 本机 cold_shape / refs → 4090 bench13 |
| 32 | f 验收 | 4090 baseline PASS（refs 21 / parity 3+5 / 单测 8 / e2e 6）。bench13（decode splits 1024）：c1 500 = 1.12×、c8 3082 = 1.06×、c32 8331 = 1.00×（-2%，抖动内）、**c8-long 1730 = 0.91×（持平）**，TTFT 102 vs 60（p50 抖动，p99 同 140） | 微基准 B=8 seq=2048 快 20%，但端到端 c8-long TPOT 3.9 vs 4.0 不变：decode_v2 不是 c8-long 的主矛盾；真正差距在 TTFT（paged_tc 混批 + 排队）。保留改动（无害） | 下一轮：① cold_shape 加 paged 形状（hist=0, q=1024 与 prefill 同 token 量）对比 paged_tc / prefill_tc；② 查 vLLM 4090 的 max_num_batched_tokens 默认（arg_utils 2960+）对齐预算；③ 调度上让新 prompt 的 prefill 优先独占 step |
| 33 | g paged_tc 分析 | `cold_shape` 新增 paged vs prefill 四形状（3090 warm ms）：A q1024 h0 **0.698** ≈ prefill_tc q1024 **0.672**（间接寻址无代价）；B q1024 h1024 1.825（历史 tile 翻倍，合理）；**C q1024 + 7×decode(h1100) 0.867（+30%）**；D 7×decode only 0.323 | 混批里每个 decode 行独占一个 block、串行扫 ~35 个 tile（76 KB smem → 每 SM 只能驻 1 block），7 行就是 7 × 35 次 WMMA 空转占着 SM；nsys 1.8× 的另一半是调度：预算 4096 下 8 个 1024 prompt 分两步，第二批被切块后与 decode 混批 | ① 混批时 decode 行（q_len==1）走 `gqa_decode_v2`（按 cu_seqlens_q 定位行，paged_tc 跳过 q_len==1 的请求）；② 核对 vLLM 4090 实际 `max_num_batched_tokens`（arg_utils else 分支 OPENAI_API_SERVER=2048，但容器可能按显存走别的分支），对齐后再 bench |
| 34 | g 混批分流（实现） | `gqa_decode_v2`/`merge_v2` 加可选 `cu_seqlens_q`（非空时只处理 q_len==1 请求、行号取 cu_seqlens_q[b]）；`gqa_paged_tc` 加 `skip_single`；`ops::gqa_mixed_bf16` = decode_v2(mixed) + paged_tc(skip)；qwen3 `BatchMode::Paged && d==128` 走 mixed。3090 refs 21/21、real_parity 3/3 | vLLM 4090 实际 `max_num_batched_tokens=2048`（容器日志 + EngineArgs 复算），比 dinfer 的 4096 还小 → 调度预算不是 vLLM 的优势，差距在混批 kernel | 混批 ref 测试 + cold_shape C 形状对比 → 4090 bench14 |
| 35 | g 验收 | 4090 baseline PASS（refs 22 含新 `gqa_mixed_d128_ref`）。bench14（混批分流）：c1 499 = 1.11×、c8 3075 = 1.05×、c32 8323 = 1.00×、**c8-long 1729 = 0.90×（不变）**，TTFT 101 vs 54 | 3090 微基准 C 形状只省 12%（0.86 → 0.76 ms），28 层 ≈ 3 ms，淹没在 100 ms TTFT 里 → **TTFT 的大头不在 attention kernel，在调度 / 排队**。vLLM 预算 2048 却 TTFT 54：说明 1024-prompt 的 prefill step 本身应 ≤ 15 ms | 看 `engine_step_execute_seconds` 直方图（c8-long 下 step 耗时分布）+ 单发 1024 prompt 的 TTFT；对照 vLLM 调度（新请求 prefill 在 running decode 之前还是之后拿预算） |
| 36 | h prefill 单发拆解 | 4090 重启后：**单发 1024 prompt TTFT 25 ms**（c1）；c8-long 80/140 双峰；`engine_step_execute` 直方图：774 步中 755 步 <5 ms，17 步 50–100 ms（= 8 个 prefill step，每步 4 个 1024 prompt 填满 4096 预算，一步 ~50–60 ms）。按 nsys 均值折算 1024-token prefill step：paged_tc 420 次×794 µs → **~11 ms/步 attention**，GEMM ~2.4 ms → attention 占 prefill step ~80% | prefill_tc / paged_tc 是 latency-bound：T=1024 时 grid = 32 行块 × 8 kv head = 256 block，76 KB smem → 每 SM 1 block、128 SM 只有 2 波；对角块串行 32 个 tile，每 tile 3 次 __syncthreads + S/P 经 smem 往返；tensor core 利用率极低（1024² × 128 × 2 × 2 FLOP ≈ 0.5 GFLOP/层 应 <10 µs） | prefill v4：FA2 结构——每 warp 持有自己 16 行的 S/O 片在寄存器（mma.sync m16n8k16 或 wmma 累加器直接 softmax，不经 smem）、K/V tile 双缓冲 cp.async、64 key tile、去掉 S/P 的 smem 往返；目标 4090 T=1024 ≤ 150 µs/层（单发 TTFT 25 → ~8 ms） |
| 37 | h prefill v4（实现中） | `gqa_fa_v4_kernel<PAGED>`：FA2 结构，裸 `mma.sync.m16n8k16 bf16` + `ldmatrix(.trans)`；block = 4 warp = (req, q head, 64 行)，每 warp 16 行：Q 片 / S(16×64) / O(16×128) 全寄存器，softmax 4-lane shuffle，P 的 C 片直接复用为 PV 的 A 片；K/V 64-key tile smem 行跨 136（无 bank 冲突）34 KB 静态 → 2 block/SM；exp2 域；causal 只对越界 tile 算 mask。单发 1024 TTFT 25 vs vLLM 21、2048 64 vs 36（二次项）；vLLM 用 FA2 | v3 瓶颈：S/P/O 经 smem 往返 + 每 tile 3 次 block sync + 76 KB smem 1 block/SM | refs（prefill/paged/mixed 4 个）→ cold_shape 对比 v3 → real_parity → 4090 bench15 |
| 38 | h prefill v4 本机验收 | 3090：refs 22/22、real_parity 3/3（logits worst 0.875，与 v3 同级）。`cold_shape` prefill T=389/889/1514/1900：0.05/0.12/**0.28**/0.37 ms（v3 0.16/0.54/1.34/1.99，**4.8×**）；q1024 0.677 → 0.144；paged A 0.695 → 0.189、B 1.81 → 0.45、C 0.86 → 0.31（mixed 0.25） | 寄存器驻留 + 64-key tile + 2 block/SM 消掉 v3 的 smem 往返与 sync | 4090 baseline + bench15 |
| 39 | h 验收 | 4090 baseline PASS（refs 22 / parity 3+5 / 单测 8 / e2e 6）；cold_shape T=1514 **0.14 ms**（目标 ≤0.4）、T=1900 0.20；单发 1024 TTFT **15 ms**（vLLM 21）。**bench15：c1 501 = 1.12×、c8 3093 = 1.07×、c32 8444 = 1.02×、c8-long 1915 = 1.00×**（TTFT 68 vs 68、TPOT 3.8 vs 3.7）。**首次四组全部 ≥ vLLM** | — | 跑 `scripts/verify/q3.sh` 验收；c8-long 还只是持平，下一轮：prefill 优先独占 step（减少 decode 被 50 ms prefill step 打断）、chunk 预算与 TTFT/TPOT 权衡；采样 kernel 融合 |
