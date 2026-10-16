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
| d 逐段对齐与耗时比对 | 对每个阶段（H2D 输入、embed、每层 attention/MLP、lm_head、采样、D2H、HTTP）与 vLLM 的同阶段并排：nsys 的 kernel 分组 + `/metrics` 的 step 直方图；写成一张表，找出每段差距 | 产出 `test/qwen3/<日期>/stage/stage-compare.md` | 进行中（`scripts/verify/stage_profile.sh`） |
| e kernel 合并（最后） | rmsnorm+残差、rope+cache write、qkv 拆分并入 GEMM epilogue、采样管线一个 kernel、logprobs 不复制 raw | 每 step 省 ~0.5 ms | 待开 |
| f 激活复用 | per-layer scratch → 全模型共享一组 | 省 ~2.5 GiB → KV 更多 | 待开 |
| g 权重加载 | mmap 页 pinned + 异步 H2D / 多线程按张量并行 | 2.0 s → < 1 s | 待开 |

不在 Q3：PD 分离（Q4）、fp8、多卡、Triton。

## 环境

4090（`scripts/dev/remote4090.sh`），dinfer `--release` 常驻 :8012（tmux `dinfer`），vLLM Docker :8013。改 kernel 后：`push` → 远端 `cargo build --release -p dinfer --features qwen3` → 重启 tmux `dinfer` → `bench_compare.sh`。

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
