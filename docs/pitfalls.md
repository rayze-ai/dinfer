## 2026-10-24 参照自己两次抓取相差 0.0077，我拿它当判据做了 5 轮逐层分析

- **现象**：用 `kaon3_vllm_hidden.py` 抓 vLLM 的逐层残差流 `L0..L29`，逐层比，
  得出"L0 差 5.8e-3、L1 放大 5 倍、之后缓慢累积"的结论，据此推了 5 轮。
  后来为了对齐一段中间量，把**同一份参照**在两个目录里的 `L0.bin` 互相比：**relL2 = 0.0077**。
  而且 `(residual + post_ff_norm)` 与 `L0.bin` 的比值中位数是 0.0371（不是 1，也不是 `layer_scalar`）。
- **根因**：那一路 dump 把 `DecoderLayer.forward` 的返回值当层输出，但它与层内子模块 hook
  抓到的量**不是同一口径**（中间还有处理/warmup 调用混淆）。
  参照数据自身不自洽，任何"我们 vs 参照"的比较都只是噪声。
- **解法**：参照数据到手后，**第一步先比"参照 vs 参照"** —— 重抓一次，或做内部自洽性检查
  （`a + b == c` 这类恒等式、两次抓取的逐位比较）。
- **避免**：① 子模块级 hook 抓的量（口径一致、可自洽验证）**可信**；
  层返回值的整体 dump **必须先自证**。② 看到恒定的比值（如 0.0703），先找是不是某个
  scale 没对齐（本例是 `layer_scalar = 1/14.222`），不要当成"神秘常数"去反推。
  ③ 用 `--layer-iso` 时**输入必须是上一层的输出**（L0 用 embedding `Lm1`，不是 `L0`），
  传错层会得到 relL2 ≈ 1.0 的"整层全错"的假象。

## 2026-10-28 同一服务上重复跑 bench，第 2 次起 TTFT 掉 7×——是 prefix cache 命中，不是优化
- 现象：ttft8 探针里 c8×64 TTFT p50 75 ms，而 bN 链同参数 545 ms；差了 7×，一度以为是 HOST_TIME 或"c1 之后"触发了什么。
- 根因：`bench` 的 prompt 由 `(seed, idx)` 决定性生成，同一服务上同 seed 再跑一遍 → 每条 prompt 都命中 prefix cache，Paged 步只剩 ~13 + ~90 个新 token；bN 每条链新起服务才是"新 prompt 态"。
- 解法：同服务多次 bench 要换 `--seed`；或看服务日志里 Paged/Prefill 步的 t（命中时 t 远小于 prompt 长度）。
- 避免：任何"第 N 次比第 1 次快很多"先查 cache 命中（`/metrics` prefix_cache_hits），再谈优化。[[dinfer-prefix-cache-bench-nondeterministic]] 同类。

## 2026-10-24 `remote_5090.sh ctr` 跑长任务被静默杀掉（`docker run --rm` 随 SSH 会话结束）

- **现象**：`GPU=4 scripts/dev/remote_5090.sh ctr '... python3 capture.py'` 抓参照中间量，
  反复出现"跑了 90 s 没有任何输出、`/w/tmp/moe4` 目录都没建、`pgrep python3` 返回 0"。
  改用 `setsid nohup ... &` 放进容器内也没用。
- **根因**：`ctr` 是 `docker run --rm ...`（`remote_5090.sh:59`）。SSH 命令返回 / 会话断，
  容器被回收，里面的进程跟着死。容器内再 `nohup` 也救不了 —— 父容器没了。
- **解法**：长任务在**宿主机**上拉起容器并脱离会话：
  `setsid nohup docker run --rm --entrypoint bash --gpus device=N ... image /w/script.sh > log 2>&1 < /dev/null &`
  脚本先写到 `/hf_data/yaochen/cap.sh`（容器里是 `/w/cap.sh`）。
- **避免**：`ctr` 只适合**短命令**（编译、跑测试，几分钟内返回）。
  抓数据 / 跑服务 / 任何超过几分钟的任务，一律走宿主机 `setsid nohup docker run`。

## 2026-10-24 抓到的参照中间量全是 0，据此报出"四段全部逐位 0"（整两轮作废）

- **现象**：为定位 Kaon MoE 的 1 ULP 差异，我 patch vLLM 的模块函数抓 `c1/c2/c3/y`，
  Rust 侧逐段比对报 `gate 0/2883584`、`up 0/2883584`、`act 0/2883584`、`down 0/11534336`，
  看起来"MoE 内部全部逐位一致"。后来 `od -t x1` 一看：**抓到的四个文件每个字节都是 0**。
  `ndiff=0` 是**全零比全零**。
- **根因**：vLLM 启动时先跑一次**显存 profile**，用**全零 dummy 输入**走一遍前向。
  我的 patch 条件是"第一次调用就 dump"（`if cap["n"] == 0`），抓到的正是那次 dummy。
  更糟的是上游还有一层：`moe_capture.py`（eager）那次是真实数据，所以"抓取脚本能用"的表象成立，
  换成 compile 口径后失败却没人发现 —— **数据是 0 不会报错，只会让判据变得极容易通过**。
- **解法**：dump 前加非零守卫 `if nz(t) > 0`，并打印每次调用的 nz 比例，抓完先看 nz 再比对。
- **避免**：**任何"抓参照中间量"的脚本，必须在 dump 前校验数据非常量/非全零**，
  并把校验结果打印出来。`ndiff=0` 这类"太好了"的结果，第一反应应该是
  "两边是不是同一个退化的东西"，先去 `od`/`numpy` 看一眼原始字节，再下结论。
  同理：**eager 抓的数据不能当 compile 口径的判据**（本项目的规格明写），
  换口径就要重抓并重新自证。

## 2026-10-24 用 `torch.matmul` 当对照，测出"参照与我们差 43%"，据此白写了一个 kernel

- **现象**：为判断"router GEMM 要不要换成参照的 BI persistent matmul"，我用随机正态输入比
  `torch.matmul` 与参照的 `matmul_persistent`，得到 **43% 元素不同（max 7.8e-3）**。
  据此写了逐位复刻的 `router_gemm_bi_f32`（kernel 级 0/8192 ✓），接进端到端一跑，
  **A/B 输出逐字节相同**。回头用**真实 routed 输入**再比：**0/3072，逐位相同**。
- **根因**：两处都错在"看起来严谨"的地方。
  ① **对照选错**：`torch.matmul` 在 2D 形状上可能用 split-k / 别的 tile，
     而我们代码走的是显式 cuBLASLt + `CUBLASLT_WORKSPACE_SIZE=1` 钉死的 algo —— 两个不同的 cuBLAS 入口。
  ② **数据选错**：随机正态的动态范围比真实激活大一个量级，两者恰好都舍到同一 bf16 不奇怪。
- **解法**：在同一份**真实** `sc.routed` 上把两条路都跑一遍（`ROUTER_GEMM_DIAG=1`），得到 0/3072。
- **避免**：判"要不要改"必须满足三条 ——
  ① 用**真实前向里的那份张量**（dump/hook 下来），不自己造；
  ② 对照必须是**被测代码真正走的那个封装**，不能用 `torch.*` 的等价 API 代替；
  ③ 先做真实输入的 A/B，再决定要不要写新实现。**顺序反了会白写一个 kernel。**
  已提炼进 `docs/knowledge/parity-localization.md` §8。

## 2026-10-24 定标只在 k=64 做过一次，就当成通用公式，导致布局 bug 多绕 3 轮

- **现象**：NVFP4 的块缩放 swizzle 公式，我先用 `k=64`（`Kp=4`）定标通过，就以为通用；
  真实数据 `k=2816`（`Kp=176`）时怎么都对不上，先后猜错 permute 方向 4 次。
- **根因**：`Kp=4` 时"K 方向外层"（`kO`）这个维度大小是 **1**，
  **任何与 kO 相关的索引错误都不可见**。`Mp=128` 两次都一样，所以也掩盖不掉。
- **解法**：把定标改成真实形状（`k=2816`），并**逐块验证**：只让 block0/1/2/3 非零、
  缩放取 1/2/4/8，读出 `1.0/2.0/4.0/8.0` 全部命中，才算公式对。
- **避免**：**定标必须覆盖真实形状的每个维度**，尤其是"大小恰好为 1"的退化维度。
  单点定标通过 ≠ 公式通用。

## 2026-10-23 漏设 `CUBLASLT_WORKSPACE_SIZE=1`，逐层对拍测出一整套"假发散"（L2 1.1e-4、L3 ×133）

- **现象**：`kaon3_probe --hidden-all --hidden-cmp <eager 参照>` 报 L0/L1 逐位 0、L2 = 1.1e-4、L3 放大 **133 倍**、
  L29 到 4.4e-1。跨多个 commit 复现、参照重抓两次也一致，看起来像真的实现 bug。
  据此推了一整天"差异在 L2/L3""§57 的全 0 声明不可复现""交接区记录有错"等结论——**全部是假的**。
- **根因**：vLLM 的 BI（batch invariant）在 sm_120 上不改 Triton，只改 **cuBLAS 配置**，其中一条是
  `CUBLASLT_WORKSPACE_SIZE=1`（workspace 限 1 KiB → 改变 split-K 归约序）。我们的 `run_lt_default` 归一化掩码
  与 workspace 上限都**读这个环境变量**来对齐。不设它 → router 的 bf16 GEMM 选到别的 algo → L2 起出现 1-ulp 差，
  再被 L3 的注意力放大 133 倍。**数据看起来完全像真的发散**（连"多次运行确定性"都成立）。
- **解法**：设上即归零。`CUBLASLT_WORKSPACE_SIZE=1 RMSNORM_MODE=1 ./target/release/kaon3_probe … --hidden-cmp`
  → **L0–L29 全部 30 层 `0.000e0`**（2026-10-23 亲自复验）。
- **避免**：
  1. 任何对 **BI oracle** 的逐层/逐子模块对拍，**必须先带 `CUBLASLT_WORKSPACE_SIZE=1`**；它已写在
     `docs/loops/kaon3-parity.md` 的"运行口径"里，跑之前照抄，不要凭记忆敲命令。
  2. **看到"发散"先自证环境**：同一命令换一份参照、换一个 commit 都复现，**不等于**结论正确——两者可能共享同一个
     错误环境。正确的自证是：拿一个**已知应当为 0** 的点（L0）去跑；L0 不为 0 就是环境问题，不是实现问题。
  3. `D-16-REBASE.md` 的 §44（L3 ×133）是 **GEMM 对齐之前**的中间状态，§52/§53 之后已归零；
     引用该文件时必须看**最后**一节，不要拿中间节的表当最终结论。

## 2026-10-23 逐层全部逐位一致，端到端却一点没动——参照是 eager，判据是 compile

- **现象**：p02 对 vLLM 30 层隐状态逐位 0，30 用例端到端 flip 16.56% / |Δlp| 1.20，和修复前几乎一样。
- **根因**：逐层参照脚本都用 `LLM(enforce_eager=True)`（forward hook 依赖 eager）；端到端 oracle 是服务实例，
  没加 `--enforce-eager` → torch.compile + CUDA graph。vLLM 自己 eager 对 compile 就差 4/23 翻转、|Δlp| 1.57。
- **解法**：离线用与服务**完全相同参数**跑 `prompt_logprobs` 对 oracle：compile 版 0/23、0.0000（复现），eager 版 1.57（不是）。
- **避免**：开始逐层对拍前，先证明"逐层参照的抓取进程"和"端到端判据"是同一口径——
  拿抓取进程直接出 `prompt_logprobs` 对 oracle，必须 0 差异，否则逐层参照没有意义。

## 2026-10-23 在单层数据上标定的 algo / 公式"全对"，换一层就错

- **现象**：bf16 router 在 L0 数据上选出的 algo diff 0，L2 上 diff 3；fp8 `2816×8192` 在 L5 上 diff 0，L11 上 diff 2；
  RMSNorm"乘倒数 + IEEE"在 L2 两组数据上 0，L3/L4 新数据上错。
- **根因**：多个候选在某一份数据上碰巧都为 0；一份数据区分力不够。
- **解法**：每种标定至少用两层以上真实数据（`SHAPE_NTH`、`RLAYER`、各层 `ref_sub_L*`），选**所有数据上都为 0** 的那个；
  最终以参照的 SASS 为准（见 `knowledge/bitexact-replication.md`）。
- **避免**：标定测试的数据目录列表要包含不同层；只有一份数据时结论写"未验证泛化"。

## 2026-10-23 没确认目标调用走哪条路径就下诊断（plan_for 优先级），整轮白改

- **现象**：上一轮交接写"SMALL_M_LT 没生效是因为 plan_for 先命中 autotune 的 SmallM 计划"，并据此改代码。
- **根因**：kaon3 从不调 `autotune`（plans 为空），router 也根本不走 `Gemm`，走自写 `moe_gate_f32`；两层前提都没核实。
- **解法**：grep 调用点（`layer.rs` 里 router 实际调用的函数）+ 运行期日志（`GEMM_ALGO_LOG=1` 打印实际选中的 algo）。
- **避免**：任何"X 没生效因为走了 Y"的结论，先用日志证明目标调用确实到达 Y；交接区的诊断标"已验证 / 未验证"。

## 2026-10-23 改脚本没 sync 就跑，两次"不同配置"的 A/B 实际跑同一份代码

- **现象**：`FORCE_KERNEL=flashinfer` 与 `=cutlass` 两轮跑出**逐位相同**的结果（31 层全 0，`L29.bin` md5 一致），
  差点据此得出"vLLM 的 kernel 变体下限是 0"。
- **根因**：改的是本地 `scripts/dev/kaon3_vllm_hidden.py`，容器里跑的是远端旧副本。
  `remote_5090.sh sync` 不会自动发生。且当轮日志里没有新加的 `[force_kernel]` 行，我没看。
- **解法**：改完先 `scripts/dev/remote_5090.sh sync`，并在容器里 `grep` 确认新代码存在；
  两个"不同配置"的运行，**必须证明输出不同**（md5 / 日志标志行），否则不算 A/B。
- **避免**：任何"两次运行结论相反/相同"的结果，先问"这两次真的跑的是不同东西吗"。
  A/B 的前置条件要写进记录：同步了没有、命中了几个 kernel、输出 md5 是否不同。

## 2026-10-23 自己的残留进程让 vLLM 启动失败，却记成"机器被占"

- **现象**：`ValueError: Free memory on device cuda:0 (3.07/31.36 GiB) ... less than desired`，
  我写成"GPU 被别的进程占着"。整机其实只有我在用。
- **根因**：上一轮 vLLM 进程在 `llm.generate` 之后抛异常（我的 dump hook 写错），
  NCCL 残留进程不退（不占 GPU、`ps` 可见），把显存/句柄留着。
- **解法**：跑前跑后都执行闸门脚本 `bash /w/hid/cleanrun.sh`（任何卡 >500 MiB 就 kill 并重试，仍脏则 exit 1）。
  失败判据：容器日志 < 30 行且没有 `写 N 个张量` → 先查残留进程。
- **避免**：不要把自己的泄漏描述成外部占用；先查 `ps -eo pid,etime,comm` 与
  `nvidia-smi --query-compute-apps=pid,used_memory`，再下判断。

## 2026-10-22 长 prompt 带 `prompt_logprobs` 请求把共享基线 vLLM 打崩重启（两次）

- **现象**：给基线 `:28067` 发 1252 token 的 `/v1/completions`（`prompt_logprobs:5` + `return_tokens_as_token_ids`），
  返回空 / `http=000`；`docker ps` 显示 `kaon-vllm-pegaflow` "Up 37 seconds"，`RestartCount=2`。
- **根因**：APIServer 输出处理在反分词时抛 `OverflowError: out of range integral type conversion attempted`
  （`async_llm.py:746` output_handler），EngineCore 随之退出，容器按 `restart: always` 重启（重载约 1 分钟）。
  短 prompt（≤ 25 token）同样参数正常。
- **解法**：实验请求改打私有副本 `kaon-oracle`（GPU 1，`:28100`，同镜像同参数，`--restart no`）；
  共享基线只做短 prompt 交叉核对。
- **避免**：**共享服务只发已验证过的请求形状**；新请求形状先在私有副本上试。跳板 `run` 失败时 rc=0，
  脚本要按响应内容判失败（`remote_5090.sh` 已改成自动重试）。

## 2026-10-22 Gemma4 proportional RoPE：分母修了，配对还是错的

- **现象**：Kaon-V3 30 层端到端跑通，但最后一行 top-5 与基线服务完全无交集。
- **根因（之一）**：上一轮只把 inv_freq 分母改成 head_dim，仍按"只转前 128 维、配对 (i, i+64)"旋转；
  vLLM `Gemma4RotaryEmbedding` 实际是 `rotary_dim = head_size`、inv_freq 零频补齐 → 整头 neox，配对 (i, i+256)。
  torch 参考脚本 `kaon_forward.py` 同样写错（还把 `k_eq_v` 的 V 取成 norm+rope 之后的 K），所以拿它对拍发现不了。
- **解法**：`rope_span_bf16` + `Rope::with_span(head_dim/2)`；自写 Python 参考已弃用（用户：Rust 实现，oracle 用 vLLM 服务 `prompt_logprobs`）。
- **避免**：参考实现与被测实现**不能出自同一次理解**；partial rotary 必须读 rope 类源码的配对方式；
  端到端先拿基线服务的 `logprobs` + `return_tokens_as_token_ids` 做一次 top-k 对照，再信任自写参考。

## 2026-10-22 e4m3 次正规值公式多乘了 1/8，导致"缩放不生效"的假象

- **现象**：NVFP4 对拍里 A 侧块缩放看起来没生效；同时手算锚点 `0x23` 解出 0.005859（正确值 0.171875）。
- **根因**：e4m3 次正规（e=0）应由 `v = m/8 · 2^-6` 给出，探针与参考侧都写成了多乘一个 1/8。
  参考侧错 + 随机数据 → A 侧缩放偏差与 fp4 量化误差互相补偿，相对误差 1e-2 判据照样 `bad=0`。
- **解法**：修正参考实现；保留手算期望值用例（全 1 输入、单字节常量）。
- **避免**：**随机数据 100% 通过不可信**（本轮第二次栽在这上面，见 2026-10-21「参考侧没初始化」）。

## 2026-10-22 `common.cuh` 的符号在 `namespace dinfer` 里

- **现象**：新 kernel 文件调 `to_f32` / `to_bf16` 报未定义；本地定义的 `last_err` 与库内同名函数冲突。
- **根因**：`common.cuh` 全部内容包在 `namespace dinfer` 中。
- **解法**：新 `.cu` 开头加 `using namespace dinfer;`；本地辅助函数避开 `last_err` 这类既有名字。
- **避免**：往 `cuda/` 加新文件时先抄一个现有文件的头 5 行。

## 2026-10-22 `scale_vec::4X` 的 u32 四字节是 4 个连续 k-block、lane 共享

- **现象**：按"每 lane 一个 block"理解 scale 打包时，NVFP4 结果只在部分行正确。
- **根因**：一个 32-bit scale 寄存器装的是 **4 个连续的 16-元素 k-block** 的 e4m3 字节，
  同一 warp 内 lane 之间共享同一组 k-block，不是按 lane 切分。
- **解法/避免**：构造 scale 缓冲区时按 k 方向每 64 列一个 u32、低字节对应最小 k。
## 2026-10-21 派生量插在用到它的声明之前，nvcc 报 "identifier is undefined"

- **现象**：给 gqa.cu 加滑窗时，`tile0` / `lo_a` 插到了 `warp_min_vis` 声明**之前**，
  报 `cuda/gqa.cu(161): error: identifier "warp_min_vis" is undefined`（v6、v7 各一处）。
- **根因**：`warp_min_vis` 原本在 `n_tiles` 之后才声明；我用"可见性块"做锚点插入，
  新代码落在声明点之前。v7 更隐蔽：它**没有** `warp_min_vis`，只有 `pair_min_vis`。
- **解法**：派生量统一插在 `warp_min_vis`（v7 用 `pair_min_vis`）声明之后；`tile0` 还要在 `load_tile(tile0)` 之前。
- **避免**：改 kernel 时不要凭"看起来在一起"插代码——**先 grep 出目标变量的声明行号**，
  按行号区间插入；v5/v6/v7 同名变量不同（`warp_min_vis` vs `pair_min_vis`），逐个确认。

## 2026-10-21 容器里只给 LIBRARY_PATH 不够，运行期还要 LD_LIBRARY_PATH

- **现象**：`cargo test` 链接通过，运行时
  `error while loading shared libraries: libnccl.so.2: cannot open shared object file`（exit 127）。
- **根因**：软链接目录只加进 `LIBRARY_PATH`（影响 ld），动态加载器不查它。
- **解法**：`export LD_LIBRARY_PATH=$(dirname $(find / -name libnccl.so.2|head -1)):…:$LD_LIBRARY_PATH`。
- **避免**：5090 基线容器里跑任何 dinfer 二进制 / 测试，两套路径都给（命令块见 `docs/loops/kaon-v3.md`）。

## 2026-10-21 `--features no-nvcc` 在本分支已坏，别拿它做本地验证

- **现象**：本地 `cargo build -p dinfer-kaon3 --features no-nvcc` 报 24 个错误，
  `expected *const f32, found *const c_void`，指向 `ffi.rs` 的 `stub!` 宏与 `ops.rs` 调用不符。
- **根因**：`stub!` 宏声明的参数类型与真实 FFI 签名早就漂移（`gdn_chunk_ho_v2` 等几处），没人走这条路所以没暴露。
- **解法/避免**：本地 CPU 验证这条路当前不可用，**一律远端 nvcc 编译**；
  修 stub 时按 `kernels.h` 逐条对齐，且要连带检查 `ops.rs` 的调用实参。

## 2026-10-21 对拍脚本的参考侧没初始化，得出"差 1/4"的假结论

- **现象**：NVFP4 MMA 对拍连续两轮得到"D 恰好是期望值的 1/4"，据此判断是 `scale_vec::4X` 语义问题，白花两轮。
- **根因**：探针脚本里 B 的初始化循环写成了 `Af[i] = ...`，`Bf` 从未赋值，所谓"参考值"本身就是垃圾；
  另一轮里 `acc2` 在 k 循环内声明，每块都被清零。
- **解法**：先用**全 1 探针**验证（期望值可手算），再跑随机值。
- **避免**：对拍脚本的**参考侧也要先自证**；任何"刚好差整数倍"的误差，先怀疑两边数组本身就是错的。

## 2026-10-21 `-arch=sm_120a` 不会生成 `a` 变体 PTX

- **现象**：`nvcc -arch=sm_120a` 编译 NVFP4 块缩放 MMA 报
  `Instruction 'mma with block scale' not supported on .target 'sm_120'`。
- **根因**：`-arch=sm_120a` 生成的 PTX 目标仍是 `compute_120`。`a` 变体（架构专用指令）必须显式 gencode。
- **解法**：`-gencode=arch=compute_120a,code=sm_120a`。
- **避免**：任何 Blackwell 架构专用指令（tcgen05、mxf4nvf4 块缩放）一律用显式 gencode 写。

## 2026-10-21 NVFP4 的 K 是落盘列数的两倍

- **现象**：加载器报 `scale [704,176] 与 weight 704x1408 不匹配（块大小应为 16）`。
- **根因**：NVFP4 的 `weight` 是 `[N, K/2] U8`（每字节两个 fp4），真实 K = `shape[1] * 2`。
- **解法**：`k = cols * 2`，再校验 `scale.shape == [n, k/16]`。
- **避免**：读量化权重先确认**落盘列数是元素数还是字节数**；用 scale 形状反推是很好的交叉校验。

## 2026-10-21 容器里缺 libnccl.so / libnvrtc.so 的未版本化软链接

- **现象**：`cargo build` 链接失败 `cannot find -lnccl`，但 `find` 能看到 `libnccl.so.2`。
- **根因**：包只提供 `.so.2` / `.so.13`，链接器要 `-lnccl` 对应的 `.so`。
- **解法**：建 `/tmp/libstub` 软链接并设 `LIBRARY_PATH`。
- **避免**：新容器里先 `find / -name "lib*.so*"` 核对软链接是否齐全。

# 踩坑记录

> 一条一个坑，标题写**现象**（搜索时按现象搜），正文讲根因与解法。新条目加在最上面。
>
> 条目格式（标题用 `## 2026-10-19 4090 上 `cargo test` 静默卡死：rustup 工具链只剩 58 MB（缺 rustc）

- 现象：`remote4090.sh run '... bash scripts/verify/qwen35_golden.sh'` 只有第一行 `== qwen35 golden parity ==`，25 min 后超时；
  远端 `ps` 里 cargo 是 `S (sleeping)`，`/proc/<pid>/fd` 打开着 `~/.rustup/downloads/*.partial`，没有 rustc/nvcc 子进程。
- 根因：4090 的 `~/.rustup/toolchains/1.89.0-...` 只有 58 MB（正常 633 MB），缺 `bin/rustc`；`rust-toolchain.toml` 指 1.89.0，
  cargo 于是去 rustup 下载（网络极慢），且 `rustup default` 未配置 → 非交互 ssh 里连 `rustc -V` 都报 "no default toolchain"。
- 解法：本机 `rsync -az ~/.rustup/toolchains/1.89.0-x86_64-unknown-linux-gnu/ 4090:同名目录/`，之后远端命令带
  `RUSTUP_TOOLCHAIN=1.89.0-x86_64-unknown-linux-gnu`；旧的卡住的 cargo 要先 `kill`（它持有 target 锁）。
- 如何避免：远端跑任何 cargo 前先 `du -sh ~/.rustup/toolchains/*`（< 500 MB 就是坏的）与 `rustc -V`；脚本超时但日志只有一行时，
  第一反应查工具链而不是查代码。
- 附带坑（同一现象的第二层）：工具链修好后 cargo 仍卡在 `Blocking waiting for file lock on package cache` + crates.io
  `Timeout was reached (less than 10 bytes/sec)`——4090 的 `~/.cargo/registry` 只有 6 MB（没缓存过依赖），远端下载极慢。
  解法：本机 `rsync -az ~/.cargo/registry/ 4090:同名目录/`（887 MB），之后离线可解析；同时先 `pkill -f "cargo test"` 清掉
  上次卡住的进程（它持有 package cache 锁）。

## 2026-10-19 Qwen35 tiny golden 对拍差 0.2 / 差一位 / 平局步 argmax 乱跳（三个独立坑）

- 现象一：prefill 前几行 logits 对得很好，`prompt_b` 起 max abs 2.85、argmax 错 2 行。根因不是模型，是**测试助手没清 GDN slot**：
  `prefill()` 复用 slot 0，上一个 prompt 的 conv / 递推状态还在。executor 里靠"首个 token 位置 == 0 → 清零"做这件事，裸 `Forward` 调用要自己调 `reset_slots`。
- 现象二：整段 prefill 的 logits 差到 0.2（本应 ≤ 0.01）。根因是 torch 参考的 `rmsnorm` 写成 `qb(f32 归一化) * f32 权重`，
  而 kernel 是 `qb(归一化 * 权重)`——**乘权重之后才落 bf16**。改成 `qb(norm * w)` 后 0.2 → 一致级别。写 bf16 参考时每个 kernel 的
  "bf16 边界在哪一步" 都要按 kernel 源码抄，不能按 "f32 内部算到底再转" 的直觉写。
- 现象三：教师强制 decode 第一步就差 4.0。根因是参考 `expect_step_logits[i]` 存的是"喂了第 i 个 token **之后**"的 logits，
  而对拍要的是"**预测**第 i 个 token"的那一行（= prefill 末行 / 喂第 i-1 个 token 之后）。差一位。
- 现象四：tiny 随机权重下 greedy 经常平局（参考 top-2 gap 最低 0.03），我们的 logits 与参考差 ~0.2 时 argmax 自然翻。
  这是随机权重的性质，不是 bug；对拍口径同 `硬判决不可逐位复现`——只有参考 gap > 阈值的行才断言 argmax，其余步改成教师强制。
- 如何避免：写 golden 前先把参考实现的每个算子与 kernel 源码对齐"bf16 边界"；先跑单 prompt 全量 prefill 把 max abs 压到
  0.01 级，再加 greedy / chunked / 混批；`reset_slots` 这类状态清零在裸调用路径上显式做。

## 2026-10-18 同一张卡起第二个 dinfer 实例：`not enough GPU memory for KV cache: free 5.5 GiB, cap 30%`

- 现象：PD 验收在 GPU 3 上先起 D（默认 utilization 0.9）再起 P（0.3），P 报 KV 预算不足；free 明明还有 5.5 GiB。
- 根因：自动预算 = `min(free, total × u − used) − reserve`。`used` 是**整卡已用**（含别的进程），`u` 是**整卡占比上限**，不是本进程配额。第二个进程看到 `used ≈ 17 GiB > cap 7 GiB`，预算 0。这和 vLLM 的 `gpu_memory_utilization` 语义一致，不是 bug，但同卡多实例就不能靠它分配。
- 解法：同卡多实例一律 `--num-blocks N` 固定块数（`kv_cache: fixed` 路径不看 utilization）。q4.sh：P 512 / D 2048。
- 如何避免：脚本里起第二个进程前先想 "预算公式里 used 包含谁"；`wait_up` 失败要打子进程日志尾，否则只看到 "port did not come up"。

## 2026-10-18 4090 GPU 2 起 D 报 "layer 21 attn" OOM：GPU 2 被别人的 vLLM 占着

- 现象：`Q4_GPU_D=2` 起 D，`port 8102 did not come up`，d.log 末尾 `Error: layer 21 attn`（权重加载到第 21 层显存不够）。
- 根因：GPU 2 上有 root 2026-10-16 起的 `vllm serve Qwen3-0.6B --port 8000`（非 docker，`--gpu-memory-utilization 0.9`，占 21 GB），不是我们的 `vllm-qwen3` 容器（那个在 GPU 1 :8013）。
- 解法：不动别人的进程；跨卡测试改用 GPU 3（P）+ GPU 4（D）。
- 如何避免：起任何测试进程前先 `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader`，空卡（4 MiB）才用；共享机 GPU 0/1/2 都有常驻。
- 相关：loop q4-pd 第 16 轮。

## 2026-10-18 4090 跑 `Q4_REMOTE=1 q4.sh` 两次空跑：bench 包名、模型路径都是本机口径

- 现象：第一次 `cargo build -p bench` 报 "package ID specification `bench` did not match"；第二次 e2e 0.16 s 失败，`[serve:err] Error: No such file or directory`。
- 根因：bench crate 的 package 名是 `dinfer-bench`（bin 名才是 `bench`）；`q4.sh` 的 `QWEN3_DIR` 默认 `/data/ref/models/qwen3-0.6b` 是本机路径，4090 上权重在 `/data/yaochen/infer/models/qwen3-0.6b`。
- 解法：`cargo build --release -p dinfer -p dinfer-bench --features qwen3`；`QWEN3_DIR=/data/yaochen/infer/models/qwen3-0.6b Q4_REMOTE=1 bash scripts/verify/q4.sh test/qwen3/<日期>/<名>`。远端非交互 ssh 还要先 `source /data/yaochen/infer/env.sh`（cargo 不在 PATH，BUILD_RC=127）。
- 如何避免：远端命令统一从 `scripts/dev/README-remote4090.md` 的模板抄；q4.sh 若 `$MODEL` 不存在应直接报错退出而不是让 serve 去撞。
- 相关：loop q4-pd 第 14–15 轮。

## 2026-10-18 PD 分离 e2e：D 等 KV 超时回退 + 首 token 重复（"redred"）

- 现象：`e2e_qwen3_pd` 经 proxy 的输出比单机多一个重复的首 token；D 日志 `pd: KV wait timeout, falling back to local prefill`。
- 根因 1：P 侧 "finish 时记块、下一次 schedule 导出" 的设计依赖 engine 再跑一步；P 空闲（只有这一个请求）时 engine 进 `rx.recv()` 死等，导出永远不发生。core 单测里紧跟着提交了别的请求所以没暴露。
- 根因 2：D 回退本地重算后自己出了首 token，proxy 不知情仍拼上 P 的首 token。
- 解法：`Scheduler::has_pending_export()` 进 engine 空闲判定；D 侧 chat.rs 先在 HTTP 线程等迁移到达（`ChatBackend::pd_wait_for`），回退时响应头 `x-pd-imported: 0`，proxy 据此决定拼不拼。
- 如何避免："事件发生在下一步" 的设计要同时问 "谁保证有下一步"；跨进程协议里任何一侧的回退都要在响应里显式告知对端，不能靠默认假设。

## 2026-10-18 第 5 次 `400 Input is too long`：一条 Bash 四段拼接，每段都"截断"了，合计 11 KB

- 现象：会话开场 `git status --short && git diff --stat && git diff X | head -c 6000 && tail -n 30 loop.md | cut -c1-300` 一条命令，context_guard 放行（有截断管道），输出 11.3 KB 被 harness 存盘只给 2 KB 预览，再加后续几条 7–8 KB 的 grep 输出，上下文很快打满。
- 根因：钩子只判"有没有截断管道"，不判截断量；`head -c 6000` + `tail -n 30 | cut -c1-300`（最多 9 KB）本身就超过 `BASH_MAX_OUTPUT_LENGTH=8000`。多段 `&&` 拼接时每段合规、合计超限。
- 解法：`context_guard.sh` 按 `;`/`&&`/`||` 拆段逐段检查；`head -c` 上限 3000、`head/tail -n` 上限 60 行（超过须配 `cut -c`）；每段 cat/diff 都要各自带截断。
- 如何避免：一条 Bash 只看一件事；想看 diff 先 `--stat`，再按文件 `git diff -- path | head -c 3000`；长日志只 `grep -E` 关键行。开场复述现状只需要 `git status --short` + loop 表最后 2 行。
- 影响：本次丢了半小时上下文，重新 /clear 恢复。

## YYYY-MM-DD 现象一句话`，正文五项：现象 / 根因 / 解法 / 如何避免 / 相关）：
>
>     ## 2026-01-01 启动时报 8080 端口被占用
>     - 现象：
>     - 根因：
>     - 解法：
>     - 如何避免：
>     - 相关：`文件/命令/链接`

## 2026-10-15 HF forward hook 里直接存张量，多次前向后拿到的全是最后一次的值

- 现象：`gen_tiny_glm.py` 对同一模型连做多次 `model(seq)`（每个 decode 步一次），hook 存的 `attn_L1` 等在后面分析时与当次前向对不上。
- 根因：hook 保存的是 `o[0][0].float().numpy()` 的视图 / 共享存储，下一次前向复用缓冲后内容被覆盖。
- 解法：hook 里一律 `.copy()`（numpy）或 `.detach().clone()`（torch）。
- 如何避免：写对拍 dump 时，hook 保存值后立刻 `np.save`，或用 `.copy()`；对比前先校验 "dump 的 logits 与 fixture 的 expect 逐位相等"（本次就是靠这一步发现的）。
- 相关：`scripts/verify/gen_tiny_glm.py::dump_hidden/dump_dsa`

## 2026-10-15 minijinja 跑 HF chat_template.jinja：5 处方言差异导致解析失败或输出不逐字符一致

- 现象：① `syntax error: unexpected float`（`m.content.0.type`）；② `string has no method named strip/split`、`map has no method named items`；③ tools 段多出一个 `\n`；④ JSON 里 key 顺序被排序、分隔符是 `,`/`:`（Python 是 `", "`/`": "`）。
- 根因：minijinja 把 `.0.type` 词法成 float；没有 Python 字符串 / dict 方法；HF 的 jinja 环境是 `trim_blocks=True, lstrip_blocks=True`（`chat_template_utils._compile_jinja_template`）而 minijinja 默认关；minijinja/serde_json 默认 BTreeMap 排序 key，`tojson` 用紧凑分隔符且不认 `ensure_ascii` kwarg。
- 解法：加载前 `x.N.attr` → `x[N].attr`；`set_unknown_method_callback` 补 strip/split/items 等；`set_trim_blocks/lstrip_blocks(true)`；minijinja 与 serde_json 都开 `preserve_order`；自写 `tojson` filter 按 Python `json.dumps(ensure_ascii=False)` 输出。验收口径必须是 "渲染文本与 HF `apply_chat_template(tokenize=False)` 逐字符一致"，再 encode 比 id —— 只比 id 会漏掉空白差异。
- 如何避免：新模板先用 `gen_chat_golden.py` 产 golden（含 tools / 多轮 / tool 结果乱序 / `clear_thinking` / 非法 `reasoning_effort`），跑 `chat_parity` 看 `first diff at char N` 定位。
- 相关：`models/glm53/src/chat.rs`、`scripts/verify/gen_chat_golden.py`、`models/glm53/tests/chat_parity.rs`

## 2026-10-15 server 请求 500 `indexer: block_table covers 16 < max_seq 1024`：block_table 宽度不是 cache 容量

- 现象：单测 / 对拍里 block_table 是 `[1, NUM_BLOCKS]`（全 cache），一切正常；接上 runner 后第一个请求就 500。
- 根因：runner 的 `BatchPlan.max_blocks` 是本 batch **最长请求的块数**（vLLM 同），不是 cache 容量；索引器层与 `ops::dsa_index_score_f32` 各有一条 `max_blocks*block_size >= max_seq` 的 ensure，把 "score 行步长 = cache 容量" 和 "block_table 宽度" 混为一谈。kernel 实际只读 `bt[j / bs]`，j < vis ≤ seq_len，宽度覆盖每个请求自己的 seq_len 即可。
- 解法：删掉两处 ensure；注释写清两个量的语义。
- 如何避免：新 kernel 的 host 侧 ensure 只检查 "会越界读写" 的条件，不要把测试 harness 的巧合（全表）写成约束；接 runner 的路径（`serve_e2e`）要尽早跑。
- 相关：`crates/layers/src/indexer.rs`、`crates/kernels/src/ops.rs::dsa_index_score_f32`、`crates/core/src/runner.rs::build_plan`

## 2026-10-15 HF 参考在 bf16 下不是"前缀稳定"的：同一 prompt 加两个 token 重算，前缀行的 top-k / 路由会翻转

- 现象：注入 HF top-k 后 prefill 全对（attention ≤ 0.0005），但 decode 第 2 步起 L1（shared 层）attention 误差 0.02（L0/L2 仅 0.0005），logits 0.06；且与步数无关、不累积（step 1 好，step 2 坏，step 21 坏）。逐层 bisect 到 cache：我们 L1 cache 第 142 行与 HF 同一行差 1.87，但 prefill 刚结束时该行是对的。
- 根因：HF 侧参考是对 `prompt + gen[:i]` 整条序列重新前向。bf16 下 SDPA/MoE 的 batch 内部归约顺序随序列长变化，L0 索引器对 12 个 **前缀行** 的 top-k 在 256 与 258 token 两次前向里不同（其中 7 行是前 8 行的 "可见 ≤ k" 退化，无影响；5 行 142/163/172/233/246 是真翻转），这 5 行的 attention 输出变了 0.02–0.03 → 它们在 HF 里的后续层 KV 与我们 prefill 时写进 cache 的不同。任何 decode 步只要 top-k 选中这 5 行之一，L1/L2 attention 就与 HF 不可比。不是 bug：我们的增量 cache 才是 "正确" 的语义，HF 的全序列重算在 bf16 下自己都不自洽。
- 解法：`gen_tiny_glm.py` 每个 decode 步额外导出 `hidden_L<i>_long_prefix_drift_steps.npy`（[gen, T]：每前缀行相对 prefill-only 前向的 hidden max abs 变化）；`dsa_parity` 把变化 > 0.02 的行并入 "漂移行" 集合，注入 top-k 命中漂移行的 decode 步不断言。可判定步 18/32，其余步只上报。
- 如何避免：参考实现用全序列重算做 decode 对拍时，先验证它自己是前缀稳定的（对比 prefill-only 与更长序列的前缀 hidden）；不稳定就导出逐步漂移并过滤，不要调阈值硬凑。对拍脚本的 `dump_dsa` 之类 hook 一律用 `.copy()`，否则多次前向后拿到的是最后一次的张量。
- 相关：`scripts/verify/gen_tiny_glm.py::dump_hidden`、`models/glm53/tests/dsa_parity.rs::drift_rows`、`models/glm53/tests/dsa_diag.rs`（逐层 / 逐 head / cache 行 bisect 工具，保留）

## 2026-10-15 Claude Code 会话报 `API Error: 400 Input is too long`，整段上下文丢失，只能 /clear

- 现象：长会话（多轮 bisect、反复贴测试日志）突然所有请求报 400 Input is too long，无法继续也无法 compact，只能 `/clear` 重来。本项目已发生两次（10-14 P4、10-15 P5），每次丢掉未写进文件的中间结论。
- 根因：上下文被工具输出撑满——整段 `cat` 测试日志 / `cargo test --nocapture` 全量输出 / 大 diff 直接进对话；自动压缩来不及或压缩后仍超限。
- 解法：发生后按 CLAUDE.md 开场流程从 `docs/progress.md` 交接区恢复；本次恢复后先做 wrap 把状态落盘。
- 如何避免：① 所有 Bash 输出先 `| grep -E ... | tail -n` 截断，日志 `tee` 到 `/tmp` 后只看摘要；② 每完成一轮 loop 迭代就把结论写进 `docs/loops/<feature>.md` 记录表和 `test/YYYY-MM-DD/`，不要攒到最后；③ 长 bisect 前先 wrap 一次。
- 相关：`docs/progress.md` 交接区"会话注意"

## 2026-10-15 DSA 索引器 top-k 与 HF 不一致，但分数公式逐位正确：bf16 下硬判决本质不可复现

- 现象：tiny DSA 模型长 prompt 256 行里 21 行的索引器 top-k 集合与 HF 不同；两边 index_scores max abs diff 0.024。注入 HF 的 top-k 后仍有 4 行 logits 差 0.2+。
- 根因：① 把 HF 自己的 q/k 喂进我们的分数公式，与 HF 分数逐位相等 → 公式无误；差异全来自 q/k 各自 bf16 舍入后的 fp32 点积（~1e-2），而 HF 自身 k-th 与 (k+1)-th 分差中位数只有 0.006 → top-k 是对噪声级分差做硬判决，两套实现不可能逐位一致。② 剩下 4 行是 MoE 路由第 2/3 名专家分差 < 0.01 时选了不同专家（P4 已见过的 bf16 平局，tiny 随机权重下约 2%）。逐层 bisect 显示三层 attention 输出 max abs ≤ 0.0007。
- 解法：对拍分三道：top-k 集合差异只允许落在 "HF k-th 分数 ± 本行两边分数实测误差" 内；注入 HF top-k 验证稀疏 attention（排除索引器）；全程自算只在 "索引器 k-th 分差 > 实测误差 且 路由分差 > 0.01" 的行/步上断言。参考侧额外导出 `indexer_score_*`、`router_margin_*`。
- 如何避免：任何 top-k / argmax / 路由类硬判决都不要用 "集合完全一致" 做验收；先从参考实现导出判决依据的分数，用分差判定该行是否可判定。试过把 RoPE cos/sin 表改成 bf16 对齐 HF（无改善，21 → 24 行），不要再试。
- 相关：`models/glm53/tests/dsa_parity.rs`、`scripts/verify/gen_tiny_glm.py::dump_dsa/dump_router_margin`、`docs/loops/p5-dsa-indexer.md`

## 2026-10-15 transformers ≥ 5 的 MoE 专家权重是融合 3D 张量，与 GLM-5.3 safetensors 逐专家布局不同

- 现象：用 transformers 5.19 `GlmMoeDsaForCausalLM.state_dict()` 存的 tiny 模型，Rust 加载报 `missing tensor model.layers.1.mlp.experts.0.gate_proj.weight`。
- 根因：5.x 的 `GlmMoeDsaExperts` 用 `gate_up_proj [E, 2I, H]` / `down_proj [E, H, I]` 两个参数存全部专家；真 GLM-5.3 safetensors（以及 4.x DeepseekV3）是 `experts.{e}.{gate,up,down}_proj.weight`。
- 解法：`gen_tiny_glm.py --dsa` 存盘前拆回逐专家：`gate_up[e].chunk(2, dim=0)` → gate、up。
- 如何避免：以真模型 safetensors index 的张量名为唯一真相，参考实现的 state_dict 只是导出格式；加载器不要迁就某个 transformers 版本。
- 相关：`scripts/verify/gen_tiny_glm.py`、`/data/ref/models/glm53/model.safetensors.index.json`

## 2026-10-15 所有 cargo 命令假失败："failed to run rustc to learn about target-specific information"

- 现象：`cargo build/test/clippy` 全部报该错，stderr 是 rustc 把一段 bash trace（`++ awk '-F: ' '{if($1~"Serial Number")print$2}'`）当 Rust 源码编译；直接跑同一条 `rustc -` 命令却正常；在 `/tmp` 新建 crate 也正常。
- 根因：cargo 用 `rustc - --print=...` 探测目标信息（stdin 读源码），结果缓存在 `target/.rustc_info.json`。本机 `/etc/cron.d/lldpd_config_update` 每分钟跑一个 `set -x` 的脚本，某次它的 trace 文本进了那次探测的 stdin，失败结果被缓存，之后每次 cargo 直接读缓存里的失败。
- 解法：`rm target/.rustc_info.json`。`scripts/verify/p4.sh` 开头已加这一行。
- 如何避免：验证脚本里凡 cargo 都 `< /dev/null`；看到 `<anon>` 里出现 shell 文本就是这个坑，别去查代码。
- 相关：`target/.rustc_info.json`、`/etc/cron.d/lldpd_config_update`

## 2026-10-14 MoE 路由用 bf16 GEMM 算 logits 会翻转 top-k 选择

- 现象：按"gate 权重 bf16 → cuBLASLt bf16 GEMM → 转 fp32 → sigmoid/top-k"实现，专家选择与 HF 偶发不一致（分数接近的两个专家互换）。
- 根因：HF `DeepseekV3TopkRouter.forward` 把 x 与 gate.weight 都 cast 到 **fp32** 再 linear；bf16 GEMM 的输出先被舍到 bf16（3 位有效数字），sigmoid 后分差 < 1 ULP 的专家对会乱序，而 top-k 是硬判决，之后误差不可恢复。
- 解法：`moe_gate_f32` 参考 kernel：bf16 输入升 fp32 做点积，输出 fp32 logits；真机版也必须保持 fp32 路由（vLLM `moe_router_dtype: float32` 同义）。
- 如何避免：任何"硬判决"（top-k、argmax、路由）前的运算精度要与参考实现逐算子对齐，不能只看最终 logits 误差。
- 相关：`crates/kernels/cuda/moe.cu::gate_kernel`、`crates/layers/src/moe.rs`

## 2026-10-14 host 侧 argmax 平局规则与 torch / GPU kernel 相反

- 现象：runner 对拍报 "sampled token must be argmax"，GPU 选 230、host 选 336，但两者的 logits **完全相等**。
- 根因：测试里 `(0..v).max_by(|a,b| row[a].partial_cmp(&row[b]))` 在相等时返回**最后一个**最大值；torch.argmax 与我们的 argmax kernel 都取**第一个**（小下标）。
- 解法：host 参考 argmax 写成显式循环 `if row[i] > row[best]`（严格大于才更新）。
- 如何避免：所有 host 侧 "参考实现" 的平局规则显式写出来，并与 kernel 注释对齐；不要依赖标准库迭代器的平局行为。
- 相关：`models/glm53/tests/{tiny_parity,runner_parity}.rs::argmax`、`crates/kernels/cuda/sampling.cu`

## 2026-10-14 请求完成后 runner 状态没被清理：`finished_req_ids` 晚一步到达

- 现象：所有请求 `update_from_output` 都报 Finish 后，`runner.num_states()` 仍为 2。
- 根因：与 vLLM 一致，scheduler 在 `update_from_output` 里只把完成请求记进内部列表，**下一次** `schedule()` 才通过 `SchedulerOutput.finished_req_ids` 告知 worker 释放状态。
- 解法：引擎主循环结束条件要再多跑一次 `schedule()` + `runner.execute()`（空 batch）；测试同样处理。另：两个请求可能不同步完成，`finished_req_ids` 要跨 step 累计。
- 如何避免：runner 状态的生命周期以 `SchedulerOutput` 的 `new_reqs / finished_req_ids` 为唯一来源，不要从 `StepResults.finish` 推。
- 相关：`crates/core/src/runner.rs::update_states`、`models/glm53/tests/runner_parity.rs`

## 2026-10-14 用 scratch 缓冲的 `shape[0]` 当 token 数，把 max_tokens 当成了本步 token 数

- 现象：MLA forward 内 `rope_n` ensure 失败：`q.len 12288 < 64*4*16`，positions 只有 14。
- 根因：所有层的 scratch 按 `max_tokens` 预分配；`forward` 里 `let t = x.shape[0]` 取到的是预分配的 64，不是本步的 14。
- 解法：`forward(stream, gemm, t, x, ...)` 显式传 `t`；所有 `_n` 后缀的 kernel 包装按"容量 ≥ 所需"校验；GEMM 的激活/输出缓冲也改 `>=`，权重仍精确相等。
- 如何避免：张量形状只用于校验容量，不用于推导"本步有多少 token"；token 数永远从调度器/输入显式传。
- 相关：`crates/layers/src/{mla,nn}.rs`、`crates/kernels/src/{ops,gemm}.rs`

## 2026-10-14 Rust 包装层参数顺序与 C ABI 不一致（copy_cols）

- 现象：`copy_cols: range` ensure 失败，但调用处参数看起来合理。
- 根因：`ops.rs` 里 `copy_cols_bf16` 的参数顺序写成 `(src_cols, col0, width, dcol0, dst_cols)`，头文件是 `(src_cols, col0, dst_cols, dcol0, width)`；调用方按头文件传。
- 解法：包装层参数顺序改成与 `kernels.h` 完全一致。
- 如何避免：写包装时直接从头文件复制参数列表；考虑在 ops.rs 里给多 usize 参数的函数用结构体参数（P4 起）。
- 相关：`crates/kernels/src/ops.rs::copy_cols_bf16`

## 2026-10-14 greedy 对拍在 bf16 平局步必然分叉，不是 bug

- 现象：prefill logits 全对、32 步 decode logits 最大偏差 0.007，但 greedy 第 7 步 token 不同。
- 根因：HF 该步 top1=0.508、top2=0.504，差 0.0039 = bf16 在 0.5 附近的 1 ULP；两边各自的累加顺序决定谁略大，无法也不应要求一致。
- 解法：对拍口径改为三层：① 全词表 logits max abs < 0.05；② argmax 只在 HF top1-top2 gap > 0.02 时要求一致；③ greedy 允许在平局步分叉，且分叉 token 必须是 HF top-2 之一。
- 如何避免：所有 greedy 对拍都带 `expect_step_logits`（HF 全序列前向逐步取最后一行），用 gap 判定平局，不用"token 序列完全一致"做唯一标准。
- 相关：`models/glm53/tests/tiny_parity.rs`、`scripts/verify/gen_tiny_glm.py`、`docs/loops/p3-mla-dense.md`

## 2026-10-14 top-p 掩码与 vLLM 口径不一致：top-p 必须在 top-k 之后的分布上算

- 现象：`topk_topp_mask` 对拍，`k=5, p=0.5` 这一行 kernel 保留 5 个、参考保留 2 个。
- 根因：第一版 kernel 把 top-k 阈值和 top-p 阈值各自在 **原始分布** 上算再取 max；vLLM `apply_top_k_top_p_pytorch` 是先 top-k 置 -inf，再对剩余元素 softmax 后做 top-p（`probs_sum <= 1 - p` 置 -inf，至少留 1 个）。两者在 k 小、p 小时结果不同。
- 解法：kernel 里 top-p 的 softmax 分母与质量累计只统计 `v >= top_k_threshold` 的元素；Python 参考改成升序 + `cumsum <= 1-p` 与 vLLM 完全同形。
- 如何避免：采样链的任何算子先抄 vLLM 的 PyTorch 参考实现到 `gen_kernel_refs.py`，再写 kernel；容差按"边界 token 可能差 ≤ max(2, 0.1%)"设。
- 相关：`crates/kernels/cuda/sampling.cu::topk_topp_mask_kernel`、`scripts/verify/gen_kernel_refs.py::mask_ref`、`vllm/v1/sample/ops/topk_topp_sampler.py:473`

## 2026-10-14 cargo 集成测试的 cwd 是 crate 目录，不是工作区根

- 现象：`tests/ref_tests.rs` 用相对路径 `test/kernel_refs` 读 `.npy`，10 个测试全部 "No such file"。
- 根因：`cargo test` 跑集成测试时 cwd = `crates/kernels/`。
- 解法：相对路径按 `env!("CARGO_MANIFEST_DIR")/../..` 解析；脚本里用 `realpath -m` 传绝对路径。
- 如何避免：测试读外部文件一律走环境变量 + `CARGO_MANIFEST_DIR` 兜底。
- 相关：`crates/kernels/tests/ref_tests.rs::refs()`

## 2026-10-14 clippy 偶发 "failed to run rustc to learn about target-specific information"

- 现象：`cargo clippy` 报 rustc 解析 `<anon>` 失败，错误内容是一段 shell 脚本（`awk '-F: '`、`SN='Not Specified'`）。
- 根因：cargo 用 `rustc -` 从 stdin 探测目标信息，本机某个后台进程（疑似监控 agent）往同一终端 stdin 写了东西。与代码无关，重跑即过。
- 解法：重跑。若频繁出现，给 verify 脚本的 cargo 命令加 `< /dev/null`。
- 相关：`test/2026-10-14/p2.log`（第一次运行）

## 2026-10-14 criterion `iter_batched` 把 setup 成本算进了被测函数

- 现象：调度器 bench 报 197 / 310 µs，但手工 `Instant` 分段计时只有 62 / 92 µs；两条 bench 之间 110 µs 的差距与代码路径差异不符。
- 根因：`iter_batched(setup, routine, SmallInput)` 每次迭代都跑一遍 setup（重建 1024 个请求、prefill 一步）。criterion 文档说 setup 不计时，但 `SmallInput` 模式下 setup 产生的内存分配 / cache 污染会让 routine 本身变慢，且 setup 里 `Scheduler::new` + 1024 次 `add_request` 的 drop 成本落在 routine 之后的同一测量窗口内。
- 解法：对"有持久状态、每步修改状态"的被测对象，用 `b.iter(|| ...)` 持续跑同一个实例；把 `max_tokens` / `max_model_len` / 块数设到测量期间不会触发完成或抢占。
- 如何避免：写 bench 前先用 `examples/probe_sched.rs` 这类手工计时对一遍量级；两者差 > 30% 先怀疑 bench 写法。
- 相关：`crates/core/benches/scheduler.rs`、`crates/core/examples/probe_sched.rs`、`docs/loops/p1-scheduler-perf.md`

## 2026-10-14 Claude Code 的 Bash 工具被平台安全分类器间歇拦截

- 现象：`Bash` 调用返回 "claude-opus-5[1m] is temporarily unavailable, so auto mode cannot determine the safety of Bash"，连续数十次；`Read/Write/Edit` 多数时候可用。
- 根因：Claude Code auto mode 依赖一个独立分类器判定命令安全性，该服务不可用时所有非只读工具被拒。与本项目无关。
- 解法：拦截期间只做写文件；把要跑的命令列给用户在终端执行并贴回输出；或用户关闭 auto mode（本次用户说"拦截已经放开"后恢复一段时间，随后又复发）。
- 如何避免：长命令写进 `scripts/`，拦截时用户可直接跑；每次会话开头先用一条无害命令试探工具是否可用。
- 相关：`docs/progress.md` 2026-10-13/14 日志

## 2026-10-14 cc crate 的 `cuda(true)` 给 nvcc 塞 gcc 参数导致编译失败

- 现象：`cargo build` 报 `nvcc fatal : Unknown option '-ffunction-sections'`，另有 `incompatible redefinition for option 'optimize'`。
- 根因：`cc::Build::cuda(true)` 仍把 debug profile 的 gcc 参数（`-ffunction-sections -fdata-sections -O0 -gdwarf-4`）原样传给 nvcc；nvcc 不认。
- 解法：`build.rs` 直接 `Command::new(nvcc)` 编 `.cu` 为 `.o`，再用 `cc::Build::object()` 只做归档。
- 如何避免：所有 CUDA 编译不经 `cc` 的 compiler 路径；需要 host 侧参数用 `-Xcompiler` 显式传。
- 相关：`crates/kernels/build.rs`、`test/2026-10-14/build-glm53.log`

## 2026-10-14 k8s Pod 申请了 `nvidia.com/gpu` 但容器里没有 nvidia-smi

- 现象：`nvidia-smi: not found`，`/dev/nvidia*` 不存在，但 `/proc/driver/nvidia/gpus/*/information` 能看到 4 张 RTX 4090。
- 根因：集群用 containerd + nvidia RuntimeClass，默认 runtime 不注入 GPU 设备与驱动库；device plugin 只负责调度计数。
- 解法：Pod spec 加 `runtimeClassName: nvidia`。
- 如何避免：所有 GPU Pod 模板都带该字段；`scripts/dev/k8s/*.yaml` 已加。
- 相关：`kubectl get runtimeclass` → `nvidia`

## 2026-10-13 本机 3090 跑不了两模型的任何快路径 kernel

- 现象：vLLM 中 Kimi-K3 / GLM-5.3 的 MLA decode（FlashMLA）、FP8 块 GEMM（DeepGEMM）、MXFP4 MoE、KDA fused decode 全部要求 sm90 / sm100。
- 根因：RTX 3090 是 sm86，无 FP8 / FP4 张量核。
- 解法：每类算子先写 sm80+ 可跑的参考版（bf16 / dequant 回退）做正确性验证；快路径在目标机器做（design.md 决策表）。
- 如何避免：所有 kernel 的 FFI 保持同一签名，参考版与快路径可互换；验证脚本按 `nvidia-smi` 的 compute capability 自动选路径。
- 相关：`docs/design.md` §6 §7；`vllm/models/kimi_k3/nvidia/kda.py:161-357`（各 backend 的架构门限）

## 2026-10-13 HF 上 Kimi-K3 / GLM-5.3 只有主仓库可匿名访问

- 现象：`moonshotai/Kimi-K3-Instruct|Thinking|Flash`、`zai-org/GLM-5.3-Air`、`THUDM/GLM-5.3` 返回 401；`moonshotai/Kimi-K3`、`zai-org/GLM-5.3`、`zai-org/GLM-5.3-Flash` 返回 200。
- 根因：前者不存在或为 gated 仓库。
- 解法：目标固定为 `moonshotai/Kimi-K3` 与 `zai-org/GLM-5.3`；`GLM-5.3-Flash`（`Glm5NextForCausalLM`，结构不同）不在范围。
- 如何避免：拉文件前先 `curl -s -o /dev/null -w '%{http_code}' https://huggingface.co/api/models/<repo>`。
- 相关：`/data/ref/models/*/meta.json`

## 2026-10-16 采样分布测试在 chat prompt 首 token 上"全部样本同一 token、L1 = 0"

- 现象：100–200 个不同 seed 的请求在同一 prompt 上采样（temperature 0.7 / top_k 20 / top_p 0.9），全部得到 `151667`（`<think>`），host 侧 mask 后候选集大小 1，测试当成"采样坍缩"失败。
- 根因：Qwen3 chat 模板后的首 token 几乎确定（logprob ≈ −2e-4），任何 top-p < 1 都只剩一个候选；不是 kernel 问题。
- 解法：先 greedy 跑 24 步，拿每步原始 logits 在 host 侧套同样 mask，选候选最多的一步（实测 step 23，候选 3），用 `prompt + greedy[..step]` 作采样 prompt。
- 如何避免：分布 / 随机性测试先用 host 参考确认该位置的候选集 ≥ 2；不要默认首 token 有随机性。顺带：N 个 36-token prompt 要一次 prefill 放下，`max_num_batched_tokens` 与 KV 块数要按 `N × ceil(len/16)` 算（100 × 3 块 → 预算 8192、512 块）。
- 相关：`models/qwen3/tests/sampling_parity.rs::mask_and_distribution_match_host_reference`

## 2026-10-16 远端编译 cudarc 报 `E0428: the name CUDA_VERSION is defined multiple times`

- 现象：4090 机器上装了 CUDA 13.0 后，`CUDARC_CUDA_VERSION=13000 cargo build` 在 `Compiling cudarc v0.18.2` 处报 `CUDA_VERSION` 重定义。
- 根因：Cargo.toml 固定了 feature `cuda-12040`（本机 3090 / CUDA 12.4），cudarc 的 build.rs 发现环境变量时**额外**再 `rustc-cfg=feature="cuda-13000"`，于是两个版本 cfg 同时为真。
- 解法：feature 改为 `cuda-version-from-build-system`，cudarc 在 build 时跑 `nvcc --version` 自动探测；本机得 12.4、远端得 13.0。不再设 `CUDARC_CUDA_VERSION`。
- 如何避免：多机器多 CUDA 版本时不要在 Cargo.toml 固定 `cuda-NNNN0`；确保 build 时 `PATH` 里的 `nvcc` 就是 `CUDA_HOME` 那个（`env.sh` 已保证）。
- 相关：`scripts/dev/README-remote4090.md`、`Cargo.toml` `[workspace.dependencies] cudarc`

## 2026-10-16 CUDA 13.0 下链接 `dinfer` 报 `undefined reference to __cxa_guard_acquire / __gxx_personality_v0`

- 现象：4090 机器（nvcc 13.0.88）`cargo build` 编译全过，最后链接 bin 时 133 个 undefined reference，全部来自 `libdinfer_kernels.rlib(*.o)` 的 `__device_stub_*`，符号是 `__cxa_guard_{acquire,release,abort}` 与 `__gxx_personality_v0`。本机 CUDA 12.4 不出现。
- 根因：nvcc 13.0 生成的 host 侧 kernel stub 用了函数内 static 的线程安全初始化（C++ ABI `__cxa_guard_*`）并带异常 personality，需要 libstdc++；`build.rs` 只链了 `cudart`，12.4 的 stub 恰好不引用这些符号。
- 解法：`kernels/build.rs` 加 `cargo:rustc-link-lib=dylib=stdc++`（12.4 下多链无害，已验证本机仍过）。
- 如何避免：nvcc 产物当 C++ 对象看待，始终链 `stdc++`；换 CUDA 大版本后先在最小 bin 上做一次完整链接再跑测试。
- 相关：`crates/kernels/build.rs`、`test/qwen3/2026-10-16/remote4090-migration.md`

## 2026-10-16 4090 上测试二进制 `error while loading shared libraries: libcudart.so.13`

- 现象：`cargo build` 通过，`cargo test -- --ignored` 一跑就 exit 127，`ref_tests-*: error while loading shared libraries: libcudart.so.13: cannot open shared object file`。
- 根因：CUDA 13.0 装在 `/data/yaochen/infer/cuda-13.0`（目录约束），不在 `/usr/local/cuda`，也没进 `ld.so.conf`；链接期靠 `rustc-link-search` 找到了 .so，运行期动态加载找不到。
- 解法：`env.sh` 加 `export LD_LIBRARY_PATH=$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}`；`remote4090_baseline.sh` 开头 `source env.sh`。
- 如何避免：toolkit 不在系统路径时，PATH（nvcc）、CUDA_HOME（build.rs）、LD_LIBRARY_PATH（运行）三者一起设；`ldd target/debug/dinfer | grep "not found"` 作为 bootstrap 的最后一步检查。
- 相关：`scripts/dev/bootstrap_remote4090.sh`、`scripts/dev/README-remote4090.md`

## 2026-10-16 prefix cache 默认开后，e2e "流式 == 非流式" 逐字比较开始随机失败

- 现象：Q2-c 把 `enable_prefix_caching` 默认打开后，`serve_e2e_qwen3` 里"同一 prompt 发两次（非流式 / 流式）逐字相同"的断言在第 100+ 字符处翻一个词（"to the stars" → "of the deep"），greedy 和固定 seed 都会。
- 根因：第二次请求命中前缀块，调度只重算最后一块，attention 走 paged（带历史）路径；与第一次从 0 开始的 prefill 路径 bf16 累加顺序不同，logits 有 ~1e-2 级漂移，在 near-tie 位置翻 token。与 Q1 对 HF 对拍时的噪声底同源（pitfalls 2026-10-15 "HF 全序列重算 bf16 不前缀稳定"），不是功能 bug。
- 解法：需要逐字比较的 e2e 用 `--no-prefix-caching` 起服务；prefix cache 的正确性单独用 TTFT 下降 + 回答合理来验（`e2e_qwen3_prefix_cache`）。
- 如何避免：任何"两次请求输出逐字相同"的断言都要先问 KV 路径是否相同；要么关 prefix cache，要么只比到第一个 near-tie 之前。
- 相关：`bins/dinfer/tests/serve_e2e_qwen3.rs::spawn_server`

## 2026-10-16 `n=3` 同 seed 的流式与非流式输出不逐字相同（prefix cache 已关）

- 现象：`e2e_qwen3_request_features` 里 `n=3, seed=42, temperature=1.0` 非流式与流式各发一次，三个 choice 的文本在十几个 token 后各自翻词；而 `sampling_parity::seed_deterministic_and_batch_independent`（同 seed 两次、与另一请求混批）逐 token 相同。
- 根因：Philox (seed, offset) 本身可复现，但 logits 不是：3 个请求两次调用时 batch 组成 / 调度顺序（哪些 token 一起 prefill、decode 时 batch 大小）不保证相同，bf16 GEMM / attention 的累加顺序随 batch 形状变，logits 漂 1e-2 量级，在 near-tie 处翻 token。vLLM 同样声明"同 seed 仅在相同 batch 下可复现"。
- 解法：e2e 对 n>1 只断言形状保证（index 0..n、各有 finish、≥2 不同）；逐字可复现只在单请求同 seed（batch 形状确定）上断言；固定 batch 形状下的逐 token 复现由 `sampling_parity` 覆盖。
- 如何避免：任何"两次运行逐字相同"的断言必须保证两次的 batch 形状相同（单请求或受控的测试 harness）；服务层只承诺 OpenAI 语义，不承诺跨 batch 位比特一致。
- 相关：pitfalls 同日 "prefix cache 默认开后…"（同源）

## 2026-10-16 同为 `--gpu-memory-utilization 0.9`，dinfer 自动算出的 KV 只有 6.0 GiB，vLLM 19.4 GiB

- 现象：用户问"0.9 在 dinfer 里为什么没有"。参数有（默认 0.9），但 4090 上 dinfer `kv_cache: auto` 得 3534 块 / 6.0 GiB，`available_gib=16.3`（权重只 1.5 GB，却已用 7.2 GiB），vLLM 同卡同配置 19.4 GiB。
- 根因：logits 缓冲按 `max_tokens`（一步最多 token 数，4096）× vocab 151936 分配：model 的 bf16 logits 1.2 GiB、final_normed；executor 的 fp32 logits 2.3 GiB；再加预算公式里按 3 份 fp32 logits 预留 7 GiB。但采样只取每个请求最后一行，行数上限是 `max_num_seqs`（64），不是 4096。合计白占约 10 GiB。
- 解法：`Qwen3Model::load_with_budget` 加 `max_logits_rows`（服务 = max_num_seqs；对拍 = max_tokens），logits / final_normed / executor fp32 logits / 预算预留全部按它算；预留余量 1 GiB → 512 MiB。改后 4090 自动 8491 块 / 14.5 GiB（available 17.5）。剩余差距主要是 per-layer 激活 scratch（qkv/q/k/v/attn/mlp 中间 × 28 层 × max_tokens）约 2.5 GiB，Q3 做激活复用再收。
- 如何避免：任何按 `vocab` 宽的缓冲，行数只能是 logits 行数上限；自动预算要用真实分配后的 free，不要再手工"预留"一份估算值。
- 相关：`models/qwen3/src/model.rs::load_with_budget`、`models/qwen3/src/executor.rs`

## 2026-10-16 4090 上测试在"空闲" GPU 2 OOM：`layer 24 mlp` 分配失败

- 现象：`CUDA_VISIBLE_DEVICES=2` 跑 real_parity，模型加载到第 24 层 OOM；`nvidia-smi` 显示 GPU 2 已用 22 GiB。
- 根因：首次 `vllm_docker.sh run` 时（容器参数后来改过）有一个 `VLLM::EngineCore` 子进程逃出容器生命周期留在宿主上，占着 GPU 2（父进程是 `vllm serve /data/models/Qwen/Qwen3-0.6B`，不属于当前 `vllm-qwen3` 容器）。
- 解法：`docker top vllm-qwen3` 确认不属于它后 kill；GPU 2 回到 4 MiB，测试通过。
- 如何避免：跑 GPU 测试前先 `nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv`，确认目标卡无人占；`docker rm -f` 后也查一次有没有孤儿 EngineCore。
- 相关：`scripts/dev/README-remote4090.md` GPU 分配一节

## 2026-10-16 宿主 `nsys profile docker run …` 抓 vLLM 容器得到 0 个 kernel

- 现象：想给容器里的 vLLM 做 kernel 分解，宿主上 `nsys profile ... docker run ...` 跑完报告里没有任何 CUDA kernel；`nsys --attach-pid` 也抓不到。
- 根因：`docker run` 只是客户端，容器进程由 dockerd / containerd 启动，不是 nsys 的子进程，nsys 的 CUDA 注入层进不去；vLLM 官方镜像里也没有 nsys。
- 解法：把宿主的 CUDA 13 toolkit 目录（含 `bin/nsys`）只读挂进容器（`-v $R/cuda-13.0:/opt/cuda13:ro`），`--entrypoint /opt/cuda13/bin/nsys profile … python3 -m vllm.entrypoints.openai.api_server …`，加 `--cap-add=SYS_ADMIN`；结束用 `docker kill -s INT` 让 nsys 正常落盘。实测能抓到 kernel。
- 如何避免：对容器内进程做 profiling，profiler 必须在容器内启动被测进程；宿主 nsys 版本要与容器内 CUDA 驱动兼容（这里都是 CUDA 13）。
- 相关：`scripts/verify/stage_profile.sh`

## 2026-10-16 decode attention v2 kernel 在 ref 测试里 GPU 100% 卡死不返回

- 现象：`gqa_decode_d128_splitk_ref` 调新 v2 kernel 后进程不结束，`nvidia-smi` 显示 100% 利用率，要 `kill -9`。
- 根因：`const float sc = valid ? sum8(part) : -INFINITY;` —— `sum8` 内是 `__shfl_xor_sync(0xffffffff, …)`（全 warp mask），而 `valid` 在 lane 间不一致（尾部 key 不满 4 个时部分 lane 为 false），无效 lane 不进入 shuffle，有效 lane 永远等不到它们 → 死锁。
- 解法：shuffle 无条件执行（所有 lane 都调 `sum8`），再按 `valid` 选结果；无效 lane 的输入已清零不影响和。
- 如何避免：任何 `*_sync(0xffffffff, …)` 都不能放在 lane 间可能分叉的分支里；分叉只能在 shuffle 之后做选择。写 warp 级归约时先想"这个条件在 32 个 lane 上是否一致"。
- 相关：`crates/kernels/cuda/gqa.cu::gqa_decode_v2_kernel`

## 2026-10-16 bench7 的 8 个结果 JSON 丢失：`remote4090.sh push` 的 `--delete` 把远端 bench 目录覆盖成了本机的半成品

- 现象：4090 上 `bench7/` 只剩 `CHANGE`，趋势表没有 bench7；本机 `bench7/` 也只有 `CHANGE`（之前 pull 时 rsync 断管只拿到部分）。
- 根因：`push` 用 `rsync -az --delete` 把本机仓库同步到远端，而 bench 结果是**远端产出、本机只 pull 回来**的；本机那次 pull 不完整 → 再 push 时把远端完整的 bench7 删掉了。方向搞反的目录不能被 `--delete` 覆盖。
- 解法：`push` 排除 `test/qwen3/*/bench*/`、`stage/`、`perf-trend.*`；bench7 的数值从 `bench_compare7.log` 的 summary 表恢复（JSON 不可恢复，这一轮在趋势里按 summary 手工补）。
- 如何避免：产出在远端的目录只 pull 不 push；pull 后 `ls` 核对文件数再 commit；双向同步的目录绝不加 `--delete`。
- 相关：`scripts/dev/remote4090.sh`

## 2026-10-16 prefill v3（WMMA）要点：>48 KB 共享内存必须动态分配并 opt-in；P 以 bf16 进 tensor core

- 现象 / 约束：v3 一个 block 用 76 KB smem（Q 16K + K/V 16K + S 8K + P 4K + O 32K），静态 `__shared__` 上限 48 KB 编译期就报错。
- 解法：`extern __shared__` 动态 smem + 启动前 `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`（只需一次，用 `static bool` 守卫），launch 第三参数传字节数。4090 每 SM 100 KB → 1 block/SM，8 warp，够用。
- 数值：P = exp(s − m) 存成 bf16 再进 mma（FA2 同样做法），O 累加 fp32；对 HF 的 prefill logits worst-abs 从 0.81 → 0.97（bf16 噪声底 2.0 以内），greedy 32 步仍逐 token一致（一次 near-tie 翻转在允许口径内）。
- 如何避免：改 attention 精度路径后一定跑 `real_parity`（不是只跑 kernel ref）；ref 的 1e-2 容差对 32×32 小 fixture 不敏感，长序列累积误差要靠真模型对拍看。
- 相关：`crates/kernels/cuda/gqa.cu::gqa_prefill_tc_kernel / gqa_paged_tc_kernel`

## 2026-10-16 bench 固定 seed → 常驻 vLLM 的 prefix cache 全命中，c8-long TTFT 对比失真

- 现象：bench11 c8-long（8×1024 prompt）dinfer TTFT 85 ms vs vLLM 41 ms；但紧接着用同一 seed 再跑一次，dinfer 17.7 ms、vLLM 37.8 ms。vLLM `/metrics` 显示 `prefix_cache_hits/queries = 90%`，dinfer 48%。
- 根因：`bench` 的 prompt 由 `--seed`（默认固定）决定，每轮 bench 发的 prompt 完全一样。vLLM 从 Q3 开始一直常驻、KV 181k token，历史轮次的 prompt 全在缓存里，1024 token 的 prefill 实际只算十几个 token；dinfer 每轮改 kernel 后重启，缓存是冷的。因此历史各轮 vLLM 侧 TTFT（尤其 c8-long）被低估，dinfer/vLLM 比值被高估的不是 dinfer 而是对手的"冷"数据不存在。
- 解法：`bench_compare.sh` 每次运行取新 seed（`SEED=${SEED:-$(date +%s)}`，写进 summary），prompt 不与历史重复；两边都只能命中聊天模板前缀。bench12 起为新口径，与 bench≤11 的 TTFT / c8-long 不可直接比。
- 如何避免：对比压测时先看两边 `/metrics` 的 prefix cache 命中数；凡"常驻 vs 重启"的对比都要排除缓存状态差异。

## 2026-10-16 WMMA 版 prefill（v3）比 FA2 结构（v4）慢 4.8×：S/P/O 走 smem + 每 tile 3 次 block sync 是主因

- 现象：v3（WMMA 16×16×16，S/P/O 在 smem，76 KB → 1 block/SM）4090 T=1514 0.66 ms/层；v4（裸 `mma.sync.m16n8k16` + `ldmatrix`，每 warp 16 行 S/P/O 全寄存器，64-key tile，34 KB smem → 2 block/SM）0.14 ms。单发 1024 prompt TTFT 25 → 15 ms（vLLM 21）。
- 根因：WMMA 的 fragment 布局不公开，行级 softmax 只能把 S 存 smem 再按行读、P 再写回、O 乘 alpha 也要走 smem；每 tile 3 次 `__syncthreads`；smem 大到每 SM 只驻 1 block，装载和计算不能重叠。
- 解法：用 PTX `mma.sync.m16n8k16.row.col.f32.bf16.bf16.f32` 的公开 C 片布局（lane g=lane/4 持第 g 行和第 g+8 行、列 2(lane%4)+{0,1}）：4 lane 一行 shuffle 做 max/sum；P 的 C 片按 FA2 技巧直接 pack 成下一步的 A 片；V 用 `ldmatrix.trans` 取 B 片。K/V smem 行跨 136 个 bf16（272 B）避免 ldmatrix 8 行同 bank。
- 如何避免：attention 类 kernel 直接按 FA2 结构写，不要先用 WMMA 过渡；可用 `cold_shape` 的 q1024 单形状先对比，再接服务。
- 相关：`crates/kernels/cuda/gqa.cu::gqa_fa_v4_kernel<PAGED>`，round 37–39。

## 2026-10-16 "kernel 微基准快了、端到端不动"两次：decode splits、混批分流

- 现象：decode_v2 splits 256 → 1024 微基准 +20%，bench13 持平；混批分流（decode 行走 decode_v2）微基准 C 形状 -12%，bench14 持平。
- 根因：两处优化的绝对量（每层几十 µs × 28 层 ≈ 1–3 ms）相对 c8-long 当时 85–100 ms 的 TTFT 不可见；真正的大头是 prefill attention（每 1024-token prefill step ~11 ms attention）。
- 如何避免：改 kernel 前先用 nsys 总时间 × 占比估算"最多能省多少 ms / 请求"，低于端到端指标的 5% 不做；优先攻占比最大的 kernel。

## 2026-10-17 "微基准 / 结构性优化在闭环 bench 下看不到"：跨流 prefill/decode 并行

- 现象：executor 双流 + 调度免预算都实现正确，bench c8-long TPOT 3.9 ms 不变；step 计数显示 4000 步里只有 18 步同时含 decode 与 prefill 行。
- 根因：bench 是**闭环固定并发**（8 个 worker 串行发），1024 prompt 在预算 2048 下一步算完，一次压测只有 32 个 prefill step，其余 ~4600 步纯 decode；而且新请求到达时常常正好别的请求也在等——两类行天然错开。能藏的上限 = 32 × 20 ms / (32 × 128 token) ≈ 0.15 ms/TPOT。
- 如何避免：做结构性优化前先用 `runner step mix` 日志（mixed / pair / prefill_only 计数）确认目标形态在负载里真的出现；评估这类优化要用开环到达（泊松）+ 长短混合 prompt 的 bench。
- 相关：`docs/loops/q3-xstream.md` 第 7–9 轮；`crates/core/src/runner.rs::try_execute_pair`；`XSTREAM=0` / `--no-decode-exempt` 回退。

## 2026-10-17 serve_e2e 失败两种假阳性

- ① 3090 本机：`QWEN3_GOLDEN` 相对路径（`test/qwen3_golden`）在 `cargo test -p dinfer` 的 cwd（`bins/dinfer`）下找不到 → 永远传绝对路径（`$PWD/test/qwen3_golden`）。
- ② 4090 baseline：`remote4090_baseline.sh` 的 `run()` 只 grep `^test |test result|panicked`，e2e 子进程起服务失败（显存被上一步 parity 进程短暂占着）时 panic 行不含这些词，日志里只剩 `FAILED (rc=101)`。单独重跑 6/6 过。改进：baseline 失败时把该步全量日志 tail 出来；e2e 前 `sleep 3` 等显存释放。

## 2026-10-17 启动慢 1 s 的真凶：qkv / gate_up 拼接走了 f32 host 往返，而且不在任何计时里

- 现象：`model_load_ms` 2.0 s，但 `load_bf16` 统计（mmap touch + 转 Vec + H2D）只有 0.7 s；逐段加 `wall after` 累计墙钟才看出 layers 循环 1.6 s；再按 Attention/Mlp::load 内部分段，`Mlp::load` 1234 ms / 28 层。
- 根因：`Attention::load`（q|k|v）和 `Mlp::load`（gate|up）为了拼接，用 `load_f32_host` 把 bf16 读成 f32 Vec、host 上 concat、再 `Tensor::from_f32` 转回 bf16 上传：每层 ~3–4 千万元素 × 2 次 dtype 转换 + 3 次 memcpy。这条路不经 `load_bf16`，所以 LOAD_STATS 看不见。
- 解法：bf16 直接 `load_bf16` 上传，设备上 `memcpy_dtod` 拼进一个大 Tensor（零 host 计算）。4090 total 2.93 → 1.88 s。
- 如何避免：定位启动耗时先放"累计墙钟"（`wall after <阶段>`）二分，再在最大段内按子阶段累加；任何 `load_f32_host` / `from_f32` 在加载路径上出现都要问一句"为什么不是 bf16 直传"。
- 相关：`models/qwen3/src/model.rs::Attention::load`、`crates/layers/src/nn.rs::Mlp::load`、`docs/loops/q3-startup.md`。

## 2026-10-17 启动优化里两个"理论上应该快、实测更慢"的坑

- ① mmap populate / 页表填充与 `CudaContext::new` 并行：两者都重度占内核态内存操作（页表、驱动映射），并行后 cuda_init 250 → 430 ms，总时长不变。串行即可；真正并行有效的是 tokenizer（纯用户态 CPU）。
- ② 自建 pinned 双缓冲 H2D（64 MiB × 2 + 异步 DMA + 事件轮转）比驱动的 pageable `cuMemcpyHtoD` 慢（6.0 vs 7.8 GiB/s）：pageable 路径驱动内部已经是 staging 流水且用多个 copy engine；自建版的瓶颈变成单线程 `memcpy` mmap → pinned。要超过它必须多线程 memcpy + 多流，收益 <60 ms，不值。
- 如何避免：启动类优化每轮只改一项、4090 跑 3 次取中位；"并行"先问两件事是否争同一硬件资源。
- 相关：`docs/loops/q3-startup.md` 第 7、10 轮；`crates/kernels/src/device.rs::htod_into_staged`（保留未用）。

## 2026-10-17 `stage_profile.sh` 的 "ms/step" 不是实测每步时长

- 现象：c32 对比表给出 dinfer 0.97 vs vLLM 1.39 ms/step，看起来 GPU 已快 0.4 ms，但端到端 TPOT 3.6 vs 3.3 反过来。
- 根因：`nsys_stage_compare.py --steps` 的步数是 "完成 token 数 / 并发" 估的，两边 prefill 混批、抢占、warmup 都没算；而且 vLLM 侧 nsys 没开 `--cuda-graph-trace=node`，图内 kernel 不展开，"copy/misc 30%" 很可能是图节点被归错类。
- 如何避免：比 decode 每步时长用各自引擎的 step 直方图（dinfer `/metrics` `engine_step_execute_seconds`）或固定 batch 的离线微基准；nsys 只用来看 kernel 构成比例，不用来算 ms/step。
- 相关：`docs/loops/q3-perf.md` 第 46–47 轮。

## 2026-10-17 裸 `cudarc::driver::result::*` 调用在 engine 线程报 CUDA_ERROR_INVALID_CONTEXT / INVALID_HANDLE

- 现象：`PinnedUploader::flush`（裸 `memcpy_htod_async`）在 parity 测试里全过，一进服务就 `engine step failed: INVALID_CONTEXT`；`TimingEvent::elapsed_ms` 同样报 INVALID_HANDLE。
- 根因：cudarc 的 safe 封装（`CudaStream::memcpy_*`、`CudaEvent::*`）每次调用前都 `ctx.bind_to_thread()`；直接用 `result::*` 不会。测试在创建 context 的线程上跑所以没事，服务的 engine 是另一个线程。
- 解法：`Stream::bind_ctx()`（= `context().bind_to_thread()`），所有裸驱动调用前先调；代价一次 `cuCtxSetCurrent` 几百 ns。
- 如何避免：任何新增 `result::` 调用都放进 `Stream` 的方法里并以 `bind_ctx()` 开头；e2e（多线程）是唯一能抓到的测试，不能只跑 parity。
- 相关：`crates/kernels/src/device.rs::{PinnedUploader, TimingEvent, StagedUploader}`。

## 2026-10-17 pinned 打包上传：同一 step 内两次 flush 回卷 host 偏移 → 随机 ILLEGAL_ADDRESS

- 现象：`PinnedUploader` 第一版 parity / baseline 全过，c32 压测几百步后 `CUDA_ERROR_ILLEGAL_ADDRESS`。
- 根因：`cuMemcpyHtoDAsync` 从 pinned host 读是异步的；第二轮 `begin()` 把偏移回卷到 0 后 CPU 立刻 memcpy 新数据进同一区域，第一轮 DMA 还没读完 → 设备收到半新半旧的 ids / block_table / slot_mapping → attention 越界。单请求 / 小 batch 时 DMA 太快碰不到，B=32 时碰到。
- 解法：host 区按 step 分配（step 开头回卷一次），同一 step 内多轮 flush 各用不同区间；下一 step 开始前 step 末尾的 D2H 同步已保证所有 DMA 完成。
- 如何避免：任何"host 侧写 + 异步 DMA 读"的缓冲都要么双缓冲 + 事件，要么保证同一同步窗口内不重写；压测必须包含 B≥32 的场景。
- 相关：`crates/kernels/src/device.rs::PinnedUploader`，`docs/loops/q3-perf.md` 第 50–51 轮。

## 2026-10-17 `tmux kill-session` 杀不死 `cmd | tee log` 里的 serve 进程：常驻 :8012 跑的是 20 小时前的旧二进制

- 现象：修了 block_table 跨度后 GPU 3 单测 c8 TPOT 2.53，GPU 0 "常驻服务" 2.78 且怎么改都不动；`ps` 看到 serve 进程 uptime 72381 s、参数还是早上的 `--max-num-batched-tokens 4096`，机器上有 3 个 `tee serve.log`。
- 根因：tmux 会话里跑的是 `bash -c "source env; dinfer serve ... | tee log"`，kill-session 只杀 bash，管道两端的 serve / tee 成孤儿继续跑并占着端口；新起的实例要么绑端口失败退出，要么起来了但 bench 连的还是旧的（同端口竞争）。tee 覆盖同一个日志文件，看日志像是新进程在跑。
- 影响：bench16–22 里对 GPU 0 dinfer 的数字不保证是当轮代码；c1/c32/c8-long 那几轮的结论要用 bench23 重新对照。
- 解法：`scripts/dev/restart_dinfer.sh`：按 `--port` 模式 `pkill -INT`，等进程退、等显存回落，再起；结束打印 serve 进程数（必须是 1）。bench 脚本一律调它。
- 如何避免：任何"重启服务后测"的脚本都要在起新之前确认旧 pid 没了、端口空了、显存回落了；性能数字异常先 `ps -o etimes` 看进程年龄。

## 2026-10-17 用 Python 按 "起点 index … 下一个锚点 index" 切片替换函数体时把相邻函数一起删了

- 现象：重写 `Runner::execute` 为 begin/finish 时，切片终点选了 `/// 一行是否是 "纯 decode"` 的注释行，中间夹着的 `update_states` 整个被删，编译才发现。
- 如何避免：替换函数体时终点锚点用**该函数自己的结束**（下一个 `\n    pub fn ` / `\n    fn ` 行），或先 `grep -n '^    fn \|^    pub fn '` 列出函数边界再切；切完 `grep -c 'fn <name>'` 核对一次。
- 相关：`docs/loops/q3-async.md` 第 3 轮。

## 2026-10-17 验证异步调度等价性：必须用 batch=1 串行请求，不能用并发

- 现象：ASYNC_SCHED=0 / 1 两个服务，3 个并发请求 greedy 对比，1 条中途分叉；以为异步有 bug。
- 根因：并发时两边的 batch 组成不同（异步版新请求晚一步进 batch），bf16 下 near-tie token 翻转（pitfalls 2026-10-16 已记）。串行单请求 5/5 逐字相同。
- 如何避免：所有 "改调度 / 改 batch 组成" 的等价性测试用 batch=1 串行；并发只断言形状与完成。

## 2026-10-17 "异步下 running 行都被 placeholder 挡住" 不成立：刚完成 prefill 的请求会被再次调度

- 现象：插队 prefill 用普通 `schedule()`，断言 "step 只含新请求" 在 4090 压测里触发 `cut-in step contains an in-flight decode row` 引擎退出（3090 e2e 没碰到：请求到达节奏不同）。
- 根因：`schedule()` 跳过 running 行的条件是 "placeholder 已覆盖到 max_tokens"，不是 "有 token 在飞"。一个请求的 prefill 步（同步）刚完成、首个 decode 还没被调度，它在 running 里且没有 placeholder → 会被排进来，而它的首 token 恰好是在飞的（来自上一步的采样）。
- 解法：`Scheduler::schedule_waiting_only()`——跳过整个 running 循环；插队步永远只有新请求。
- 如何避免：任何 "我以为 scheduler 不会排 X" 的假设都写成 scheduler 的显式 API，不靠隐含条件；对调度路径的改动要在 4090 压测（请求到达密集）下跑过，e2e 的节奏太稀。

## 2026-10-17 插队步结果排队晚发 → scheduler 在不知道首 token 的情况下调度下一步，首 token 丢失

- 现象：e2e `request_features` 的 stop-string 用例输出 `", 2, 3, …"`，首 token "1" 不见了；只在 CUT_IN=1 出现。
- 根因：engine 把插队步的 MRO 放进 `sync_done` 队列等 n 结束后统一发布，但在那之前已经 `schedule()` 了 n+1。scheduler 的 `update_from_output(so_c)` 没跑，请求状态还是 "prefill 完、0 输出"，n+1 把它再排一次；之后两次 update 的 append 顺序与实际采样顺序不一致。
- 解法：`schedule()` 之前必须把所有已完成步的 `update_from_output` 做掉；插队步 finish 后立刻 update + 推送，不排队。
- 如何避免：engine 里 "调度" 和 "回填" 之间的不变式写成注释并断言：`schedule()` 调用点上 sync_done 必须为空。

## 2026-10-17 A/B 脚本在同一端口连起两个实例时，bench 打在了残留的旧实例上

- 现象：host tier A/B（:8096，gib=0 / 16 两组）第一组吞吐 0.11 req/s、第二组日志里的 `num_blocks` 与本组参数不符。
- 根因：`( serve & ); … pkill -INT -f "port 8096"; sleep 3` ——没等旧进程真正退出、端口释放，新实例起不来或起来后 bench 连到旧的；与同日 "tmux 孤儿进程" 是同一类问题。
- 解法：`scripts/dev/ab_serve.sh start <port> <gpu> [args]`：起前 pkill + 等 pgrep 为空 + 等端口不响应，起后等 ready 并打印 **pid / 真实进程数 / num_blocks**；`stop <port>` 同样等退干净。所有 A/B 都用它。
- 如何避免：任何 "起服务 → 测 → 换参数再起" 的脚本，每轮打印被测进程的 pid 与关键启动参数并与预期核对；吞吐离谱（<1 req/s）先怀疑连错进程。

## 2026-10-17 显存压力下最后一个请求 "抢占自己" → 引擎 2M 空步/秒 livelock

- 现象：host tier A/B（`--num-blocks 512`，c16 × 1024 prompt）跑到剩 1 个请求时 bench 卡住；`engine_steps_total` 每秒涨 200 万、`generation_tokens` 不动、GPU 0%、CPU 一核满。
- 根因：running 循环分不到块时抢占末尾 running；末尾就是自己 → `preempt(self)` 释放块、回 waiting、本步空；下一步从 waiting 重排需要全部块，还是不够（prefix cache 里 ref_cnt=0 的块占位 / 池本来就小）→ waiting 循环 break → 空步 → 下一步再来。engine 对 "有请求但空步" 没有退避，满转。
- 解法（三层）：① scheduler 不自抢占：victim == self 时以 `Length` 结束该请求（`force_finished` 队列，engine 推 Finished）；② engine 空步 200 µs 退避 + 每 1000 次 warn（running/waiting/free_blocks）；③ 启动期校验 `max_model_len/block_size ≤ 0.9 × num_blocks`。
- vLLM 对照：靠启动期校验（`max_model_len` 必须能装进 KV）+ 单请求不抢占；没有空步退避（它的 step 本身有 GPU 等待）。
- 如何避免：所有 "分不到资源 → 重排" 的路径都要问 "重排后一定能前进吗"；压测必须含 "池远小于活跃集合" 的配置。单测 `self_preempt_under_pressure_makes_progress`。

## 2026-10-17 同一张卡上刚停掉的实例显存还没回落，紧接着的 auto num_blocks 测试算出偏小的池 → 误判 FAIL

- 现象：A/B 实例 `ab_serve.sh stop` 后立刻跑 baseline，`e2e_qwen3_auto_num_blocks` 的 80% 估计断言失败；单独重跑 PASS（10698 vs 估计 7848）。
- 根因：驱动回收 ~20 GiB 需要几秒；`stop` 只等进程退出，没等 `nvidia-smi memory.used` 回落。
- 如何避免：所有 "在同卡起服务" 的脚本起前等显存 < 2 GB（`restart_dinfer.sh` 已有，`ab_serve.sh`/`remote4090_baseline.sh` 补上）。

## 2026-10-18 把 "KV 命中单元" 的模型相关规则写进了公共 crate（用户指正）

- 现象：`core::kv::KvCacheManager::host_tier_lookup` 只支持单组全注意力、block_size == hash_block_size、group 0；GLM / K3 下直接返回空（不报错、不工作）。
- 根因：按 Qwen3 的形状写了 "一个命中 = 一个 16-token 块 × 全部层"，没想 K3 的 KDA 状态块（边界检查点，不是 token 区间）和 GLM 的两种缓存。字节布局放对了（模型 crate），策略放错了（core）。
- 原则：core 的 KV 层只放 **与模型无关的容器与钩子**（块池、哈希链、索引、order 类型、调度钩子）；凡是 "什么算一个命中单元、多组如何合成前缀命中、一个单元多少 token、一个单元多少字节" 都是模型 crate 的事，经 trait 注入。对照：GPU 侧的 `get_computed_blocks` 其实已经按组分别处理（attention 组取最短、Mamba 组边界查），host 侧应复用同一规则而不是另写一套。
- 如何避免：core 里出现 `groups.len() != 1` / `is_mamba[0]` / 具体 block_size 比较这类条件时，就是信号——该抽 trait 了。


## 4090 之间没有 PCIe P2P：CUDA IPC 跨卡 open 直接失败（2026-10-18）
- 现象：`cudaIpcOpenMemHandle` 在另一张 4090 上报 `peer access is not supported between these two devices`；`cudaDeviceCanAccessPeer` 全 0；`nvidia-smi topo -p2p r` 全 CNS。同卡 open 正常，64 MB D2D 0.15 ms（446 GB/s）。
- 根因：消费卡（GeForce）驱动禁 PCIe P2P；且 GPU0–3 / GPU4–7 分属两个 NUMA（`topo -m` SYS）。
- 解法：PD 跨卡 KV 搬运走 host：shm pinned 共享区（design §10.3 后端 b），P D2H 写、D H2D 读；同卡才用 cuda-ipc D2D。NIXL 在这台机器上也只能走 UCX host 路径，不引入。探针：`scripts/dev/probe_ipc.cu`（`probe_ipc A B`）。
- 避免：换机器先跑探针；有 NVLink / 数据中心卡时 cuda-ipc 跨卡才有意义。

## cudarc nccl：集合通信返回 `NcclStatus`，不是 `()`

- 现象：`Comm::all_reduce_in_place(..)` 返回 `Result<NcclStatus, NcclError>`，直接 `map_err` 后类型不匹配。
- 解法：`.map(|_| ())`（`NcclStatus::InProgress` 只在非阻塞 comm 下出现，我们用阻塞 comm 可忽略）。
- 另：开 `nccl` feature 后 cudarc build.rs 链接 `-lnccl`（dynamic-linking 下运行期 dlopen `libnccl.so.2`）；无 NCCL 的机器需 `--features no-nvcc` 以外再想办法（当前 `no-nvcc` 不去掉 nccl）。

## TP 多卡：各 rank 自动 KV 预算不一致 / 单卡进程未退干净

- 现象：`--tp 2` 首跑一 rank 算出 10392 块、另一 rank 报 "not enough GPU memory"：前一个单卡 serve 刚 `pkill -INT`，显存还没释放，rank 0 所在卡 free 只剩 8 GiB。
- 解法：测试脚本等 `pgrep` 消失再起下一进程；TP 下传固定 `--num-blocks`（service 已校验各 rank 块数一致，不一致直接报错并提示传 `--num-blocks`）。
- 相关：4090 Pod GPU 0–2 长期被 vLLM 基线 / 旧 dinfer 占着（23 GB），多卡测试用 3–7。Pod 镜像无 NCCL，见 loops/q5-tp-dp.md 环境节。

## 远端 `until ! pgrep -f "<pattern>"` 等待永不结束（2026-10-18 晚）

- 现象：4090 上 `until ! pgrep -f "cargo build --release -p dinfer"; do sleep 5; done` 等了 25 分钟，build 其实 40 s 就完了。
- 根因：`ssh host 'bash -c "...pgrep -f cargo build..."'` 的 bash 自身命令行就含该模式，pgrep 匹配到自己（和同时挂着的另一条 ssh 等待命令）。另外 `nohup ... &` 在 ssh 非交互会话退出时被一起收掉，脚本根本没起来。
- 解法：等编译改用 `cargo build` 的退出码 / 日志里的 `Finished`，不用 pgrep；远端长任务用 `setsid nohup ... < /dev/null & disown`（或 tmux），起完 `ps` 确认进程在。
- 避免：pgrep 模式里排除自己：`pgrep -f "[c]argo build"`（方括号技巧）。

## 2026-10-18 远端一条命令里 `pkill -f "<pattern> --role"` 把承载它的 `bash -c` 自己杀了
- 现象：`remote4090.sh run '... pkill -9 -f "target/release/dinfer serve.*--role"; ...; q4.sh ...'` 返回 exit 0 但 0 行输出、日志文件不存在、q4 没跑。
- 根因：`ssh host "bash -c '<整条命令>'"` 的 bash 进程命令行里含同样的文本，`pkill -f` 按整行匹配 → 先杀了自己（与此前 `pgrep -f` 匹配到自己同源）。
- 解法 / 避免：远端不要在同一条命令里用 `pkill -f` 清理；用 `kill $(cat *.pid)`、`pgrep -x` 或把 pattern 写成 `[d]infer` 这类不自匹配的形式；q4.sh 自己会清理上一轮进程，不需要预先 pkill。

## 2026-10-19 子代理把模型特定内容写进公用 crate
- 现象：受限解码子代理在 `crates/server/src/chat.rs` 硬编码 `enable_thinking=false`（Qwen3 模板变量名），在 `crates/guided/src/lib.rs` 的单测里写死 Qwen3 tokenizer 路径 / eos id / vocab。用户检查时发现。
- 根因：brief 里只说"core / kernels 不得依赖 llguidance"，没把"crates/ 下不得出现任何模型特定常量 / 变量名 / 路径"写成硬约束；子代理为了让 e2e 过就近取巧。
- 解法：模型相关的量一律走 `ChatBackend` 钩子（`grammar_factory / tool_call_tags / grammar_render_extra`）由 `models/<m>/service.rs` 提供；模型相关测试放 `models/<m>/tests/`。
- 避免：给子代理的 brief 里固定加一句"`crates/` 只放模型无关代码，模型名 / token id / 模板变量名 / 模型路径出现在 `crates/` 即为错误"；收尾 review 用 `grep -rniE "qwen|glm|kimi|<tool_call>|<think>|15164[35]|151936" crates/` 自查（注释里引用为例子可留）。


## 2026-10-19 JumpServer 连接 token 一次性
- 现象：B300 机器 token 第一次 ssh 成功，几分钟后同 token 再 ssh `Permission denied (password,publickey)`；第二个新 token 也直接被拒（可能是密码与 token 不配对，或 token 在生成后被别处用过）。
- 根因：JumpServer 的"连接 token"是一次性凭据（4090 那台靠公钥 + 常驻 ControlMaster 才稳定，掩盖了这一点）。
- 解法 / 避免：拿到 token **第一件事**就是 `remote_b300.sh up` 建常驻连接（ControlPersist），之后全部 `run` 复用；不要先单独 ssh 一下"试试"。长期方案：让用户在 JumpServer 上给账号配公钥，或给 `yaochen` 账号固定密码走资产形式登录。

## 2026-10-19 bench_compare.sh 在 8 卡机器上 exit 141（SIGPIPE）
- 现象：B300 上 `bench_compare.sh` 起了 serve、写了 CHANGE / GPU 就退出，exit 141，没有任何 json；`bash -x` 停在 `gpu_tag` 第二次调用的 `nvidia-smi --query-gpu=name | head -1`。
- 根因：8 卡可见时 `nvidia-smi` 输出 8 行，`head -1` 读完一行关管道，`nvidia-smi` 收 SIGPIPE 退出码 141，`set -o pipefail` 下整条命令失败 → `set -e` 退出。4090 Pod 只可见 1 张卡从未触发。第一次猜是 `docker inspect`（没先 `bash -x`）——猜错一轮。
- 解法：`nvidia-smi -i 0 --query-gpu=name`（只查一张卡，不再 `| head`）；`docker inspect` 那行顺带加了 `/version` 回退。
- 避免：`pipefail` 脚本里 `<多行命令> | head -N` 一律改成让生产者只产 N 行，或 `|| true`；新机器首次跑 verify 脚本**先 `bash -x`**，不要猜。

## 2026-10-19 同进程内 `cuIpcOpenMemHandle` 自己导出的 handle 失败

- 现象：B300 上自定义 one-shot all-reduce 建立时每个 rank `cuIpcOpenMemHandle(对端 handle)` 都报错（`custom_ar_finish -> false`），而 PD 的 cuda-ipc 槽区（P / D 两个进程）同一段代码正常。
- 根因：CUDA 限制——IPC handle 不能在导出它的**同一进程**里打开（文档：`cuIpcOpenMemHandle` "cannot be used to open a handle created in the same process"）。TP 是一进程多线程，每个 rank 一个主 ctx，不是多进程。vLLM `custom_all_reduce` 是多进程（一 rank 一进程）所以能用 IPC。
- 解法：一进程多线程下直接交换 `cuMemAlloc` 的裸指针（UVA 下进程内全局唯一）+ `cuCtxEnablePeerAccess(cuDevicePrimaryCtxRetain(对端设备))`（`CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED` 当成功），kernel 直接读对端地址。`comm.rs::enable_peer_access`；handle 仍导出随 `ArPeerInfo` 传递，留给将来跨进程。
- 避免：凡"同进程多 ctx 共享设备内存"都走 peer access，不走 IPC；IPC 只给跨进程（PD）。

## vLLM 容器里软链接模型目录悬空（2026-10-19，Qwen3.5 基线）

- 现象：`vllm_nerdctl.sh run` 起容器后 `wait` 超时，日志 `HFValidationError: Repo id must be in the form 'repo_name'...: '/models/qwen3.5-27b'`，bench 全部 connection refused。
- 根因：`$ROOT/models/qwen3.5-27b` 是指向 `/data/models/Qwen/Qwen3.8-27B` 的软链接；容器只挂了 `$ROOT/models:/models`，链接目标在容器内不存在，vLLM 把路径当 HF repo id 去校验。
- 解法：**不用软链接**（用户 2026-10-19："软链接是你加的"），`VLLM_MODEL` 直接给 `/data/models/...` 物理路径，容器只挂 `/data/models:ro`。
- 避免：权重路径一律写物理路径，不在 `$ROOT/models` 下造别名；`wait` 失败先看 `logs` 的最后一个 Traceback。

## bench 对本机 vLLM 四组全部 "send: error sending request"，curl 同 body 却 200（2026-10-19，B300-50）

- 现象：`bench_vllm_only.sh` 四组预热请求全失败；`curl -N` 用完全相同的 body（stream + ignore_eos + chat_template_kwargs）打同一端口 200 正常；容器日志无任何 4xx/5xx。
- 根因（部分确认）：失败与成功的唯一差别是那条 ssh 命令有没有 `source env.sh`，env.sh 导出代理 B 的 4 个 `*_proxy` 变量 + **只有小写 `no_proxy`**。本机用死代理复现：reqwest 0.12.15 对大小写 `no_proxy` / CIDR 都能识别（self-check 四种组合全过），**没能在本机复现"小写 no_proxy 不生效"**——所以精确触发条件未钉死（可能与 B300-50 上代理 B 对 CONNECT 127.0.0.1 的行为或别的 env.sh 变量有关）。
- 解法（两层）：① env.sh 同时导出 `no_proxy` 与 `NO_PROXY`（`localhost,127.0.0.1,::1,172.16.0.0/16,10.0.0.0/8`，用户要求"注意设置 NO_PROXY"）；② bench 脚本开头仍 `unset` 全部代理变量（本地压测本来就不该走代理）。
- 避免：新机器 env.sh 的代理行大小写两套都给；排查这类错先对比 "有没有 source env.sh"。

## Qwen3.5 真权重乱码：zero-centered RMSNorm 被 tiny golden 的全 1 权重掩盖（2026-10-19）

- 现象：B300 真权重 27B 首次 e2e，输出 `'�� +:+'`，top logprob 全 −9（近均匀）；tiny golden 4/4 过。逐层末 token 残差范数：embed 一致，**L0 起偏**（24.5 vs HF 11.9），L0 mlp_out 9.19 vs 0.958。
- 根因：HF `Qwen3_5RMSNorm.forward` 是 `norm(x) * (1.0 + weight)`（weight 初值 0），dinfer / 参考脚本按 `norm(x) * weight`。tiny 生成器把 `input_layernorm / post_attention_layernorm / norm` 设成 `torch.ones` → 两种约定数值相同，golden 不可能发现。
- 解法：`models/qwen35/src/weights.rs::rmsnorm` 加载时把 `1 + w` 算进权重；`make_qwen35_tiny.py` 参考侧改 `1 + w` 且三类权重改随机。
- 避免：**造 tiny golden 时所有权重必须随机（含 norm / bias / 门）**，任何"全 1 / 全 0"的权重都是对拍盲区；接新模型先 grep HF 源码里的 `1.0 + self.weight` / `zero-centered`。

## Qwen3.5：prefix cache 命中让 GDN 递归状态从零开始（2026-10-19）

- 现象：c8-long bench 86 条 `GDN 状态 slot 被复用给带历史的请求` 告警，pos 全是 144/160/176；c1/c8/c32 没有。
- 根因：prefix cache 命中后 scheduler 从 position>0 喂 prompt。全注意力层能复用 KV 块，GDN 的递归 / conv 状态是"每请求一份、随 token 顺序递推"的，没有按块快照 → 新请求拿到新 slot 但历史已跳过。
- 解法：qwen35 service 强制关 prefix caching（warn）。真正支持要像 vLLM `mamba_block_size` / `mamba_cache_mode` 那样按块缓存 GDN 状态（设计待办）。
- 避免：接任何带递归状态的模型（Mamba / GDN / KDA），prefix cache 默认关，直到状态缓存实现。K3（KDA）同样适用。

## Qwen3.5：给 `StepMeta` 加字段后 golden 测试编译失败（2026-10-19，GDN chunk 第 1 轮）

- 现象：`cargo build --workspace --features qwen35` 过，`scripts/verify/qwen35_golden.sh` rc=101：`missing field gdn_chunk in initializer of StepMeta`（`models/qwen35/tests/golden_parity.rs` 3 处）。
- 根因：golden 测试不走 `Qwen35Executor::enqueue`，自己手搓 `StepMeta` / `Forward` 调 `model.forward`；executor 里加的"每步 prepare 工作区"在测试里没有对应物。
- 解法：测试里加 `chunk_ws()` 小助手（prefill 步建 `GdnChunkWorkspace`，`GDN_PREFILL=recurrent` → None），3 处补字段。
- 避免：改 `StepMeta` / `Forward` 这类"executor 与测试都在构造"的结构体时，先 `grep -rn "StepMeta {" models/qwen35` 把所有构造点一起改；`cargo build` 不编译 `tests/`，要用 `cargo test --no-run`。

## Qwen3.5 CUDA graph 捕获首个 kernel 就报 CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED（2026-10-19，性能第 2 轮）

- 现象：`capture_bucket` 里 `begin_capture` 后第一个 `embedding_bf16` 就 `cudaError 901`（operation not permitted when stream is capturing）；eager 跑完全正常。
- 根因（三件叠在一起）：① executor 用的是 `dev.default_stream()`（legacy 默认流不能 capture）；② cudarc 的自动跨流事件跟踪在 capture 期间会被驱动拒绝（Qwen3 早就 `disable_event_tracking`，qwen35 没抄这段）；③ `gdn.cu` 五个 launcher 都没收 stream 参数（发到 legacy 默认流），`gdn_recurrent` 还每次 launch 都 `cudaFuncSetAttribute`——capture 期间都不允许。前两条只在真正进 capture 时才暴露，eager 下全部"碰巧能跑"。
- 解法：`Qwen35Executor::with_graphs` 照 Qwen3：`stream.synchronize()` → `disable_event_tracking()` → `new_stream()` → 在新流上重建 `Gemm`；`gdn.cu` 五个 launcher 加 `cudaStream_t` 末参数、`ops.rs` 传 `stream.raw()`，`set_smem` 按模板实例只设一次（static 记已设的 smem）。
- 避免：新 kernel 的 host launcher **必须**收 `cudaStream_t` 并用 4 参 `<<<grid, block, smem, stream>>>`（`grep -n "<<<" crates/kernels/cuda/*.cu | grep -v "stream>>>"` 自查；`kda.cu` / `mxfp4.cu` 还在用 `nullptr`，K3 接 graph 前要改）；`cudaFuncSetAttribute` / `cudaOccupancy*` 这类"配置型"调用放 once-guard 里；任何新模型 executor 接 graph 时先抄 Qwen3 的流 / 事件跟踪设置，别用 default_stream。

## Qwen3.5 CUDA graph：捕获 / padding 行会写活请求的 GDN 状态（2026-10-19，设计时规避）

- 现象（推演，未发生）：Qwen3 的捕获假 plan 用 slot=-1 不写 KV 就够了；但 GDN 的 `conv1d_update` / `recurrent` kernel 对**每一行**都读写 `state[state_indices[i]]`，捕获时的 eager 预热 + 图里的 padding 行若指到 0..nb-1，会把活请求的递归状态改掉（且 capture 期间的 kernel 不执行，只有预热那次写，排查会很诡异）。
- 解法：`GdnState` 多分 1 个 slot（下标 `num_slots`）做废纸篓，捕获假 plan 与 padding 行的 `state_indices` 全指它；`has_initial` 填 1（不走清零分支）。
- 避免：带递归状态的模型（GDN / KDA / Mamba）接 graph，先问"padding 行写到哪"；KV 有 slot=-1 约定，状态 kernel 没有，要显式留废纸篓。

## K3 单层对拍：vLLM `chunk_kda_with_fused_gate` 的 beta 要先 sigmoid（2026-10-20）

- 现象：torch 参考与 Rust kernel 互相 cos 0.99998，但两者对 vLLM fla 参考都只有 cos 0.23。
- 根因：HF 调 `fla.ops.kda.chunk_kda(use_beta_sigmoid_in_kernel=True)`；vLLM 自带的 `chunk_kda_with_fused_gate` 签名里**没有**这个开关，
  `**kwargs` 静默吞掉，beta 被当成已 sigmoid 的值。vLLM 模型侧是在投影融合 kernel 里先做了 sigmoid。
- 解法：参考脚本 `beta=torch.sigmoid(raw_beta)`。
- 避免：对拍参考至少两份且互相独立（fla + 纯 torch），哪一份离群一眼可见；vLLM 的 fla 变体与上游 fla 的 kwargs 不要混用。

## 层内激活缓冲按 max_tokens 预分配时，kernel 封装不能按缓冲长度判容量（2026-10-20，K3 KDA）

- 现象：`Linear::forward` 报 `out shape [64, 12288]`（decode t=1），`kda_chunk_prep` 报 `scratch 容量`（prefill 63）。
- 根因：封装用 `x.numel()` 推 tokens / 用 `sc.qn.len() >= q.len()` 判容量，缓冲比本步大就误判。
- 解法：`Linear::forward_rows(…, tokens)`；`kda_chunk_{prep,wy,ho}` 加显式 `tokens` 参数，只按本步行数判。
- 避免：新 kernel 封装一律收显式 `tokens`，`ensure!(buf.len() >= tokens·row)`，不要用 `==` 或互相比缓冲长度。

## cudarc 0.18 cuBLASLt：多候选 algo 要走 `cublaslt::result` / `sys` 裸层（2026-10-20）
- 现象：`CudaBlasLT::matmul` 每次只取启发式第 1 个 algo（`requestedAlgoCount=1`）且不暴露 algo；想按形状选最快候选没有安全 API。
- 根因 / 解法：`Gemm` 自持 `cublasLtHandle_t` + 工作区，用 `result::create_matrix_layout / create_matmul_desc / set_*_attribute` 建描述符，
  直接调 `sys::cublasLtMatmulAlgoGetHeuristic(…, N, …)` 取前 N 个，`result::matmul` 传 `&h.algo`。注意：`sys` 里没有 `cublasOperation_t`
  枚举（cudarc 以 `i32` 传 TRANSA/TRANSB：1 = T，0 = N）；`cublasLtMatmulAlgo_t` 是 `[u64; 8]` 不透明值，可 Copy 进 HashMap 缓存。
- 避免：algo 只对同 (M,N,K,layout) 有效，不要跨 M 借用；bf16 GEMM 的工作区 32 MiB（sm90+）/ 4 MiB 与 cudarc 同策略放 `ArchProfile`。

## 新 .cu 里 `bf16` 未定义：common.cuh 的工具都在 `namespace dinfer`（2026-10-20）
- 现象：`identifier "bf16" is undefined`、`to_f32` 找不到。
- 解法：实现文件照 `gdn.cu`：`namespace dinfer { namespace { …kernels… } extern "C" { … } }`，`extern "C"` 放在 `namespace dinfer` 内（C 符号不受命名空间影响）。

## MoE 整层对拍：路由平局行的专家输出按定义不同，数值容差只能在路由一致的行判（2026-10-20，K3 C-2）
- 现象：第 1 层整层 `out` cos 0.999999，但 MoE 子块 `ysum` rel 0.145 / cos 0.9985 → 断言挂。top-16 pair 级 99.71%（1021/1024），
  3 个 pair 不同，全落在 `choice` 第 16/17 大相差 ≤ 2e-3 的行（bf16 输入 + f32 logits 的舍入已经能翻转）。
- 根因：硬判决（top-k）不可逐位复现（记忆 / pitfalls 已有口径）；路由一次翻转 = 该 token 换了一个专家，输出整行不同，
  但权重 ~1/16、再过 up_proj 与残差后对整层只有 1e-3 量级影响。
- 解法：`layer_parity.rs` 把 token 分成"路由整行一致"与"平局（margin ≤ 2e-3）"，断言 (a) 不一致行必须都是平局，(b) 可判定行全中，
  (c) MoE 子块（ysum / routed up / MoE out）的 rel / cos **只取路由一致的行**；全部行的数字照常打印供参考。
- 避免：以后多层 / 整模对拍同样先比路由、再按一致行比数值；整层输出用 cos 判（残差主导）。

## 其它子代理并行改同一批共享文件（kernels.h / ffi.rs / ops.rs）时的提交策略（2026-10-20）
- 现象：我给 `situ_and_mul_bf16` 加的 kernels.h / ffi.rs / ops.rs 声明被另一子代理的 commit（`ffea2b3`）一并带进仓库，
  而 `activation.cu` 的定义还在我的工作区 → HEAD 一度链接不过（声明有、定义无）。
- 解法：发现后立刻把定义文件 commit；共享文件只加"自己的符号"且放在文件末尾 / 独立 hunk，便于对方 `git add -p` 或整文件 add 时不冲突。
- 避免：改共享文件前先 `git status --short` 看是否有别人的未提交改动；声明与定义**同一次 commit**。

## `Stream::htod_into` 只接 `&mut CudaSlice`，往大张量的一段拷要走 `inner_arc().memcpy_htod(src, &mut slice_mut(..))`（2026-10-20）
- 现象：`ExpertPack::load` 把 896 个专家拼进一个 `[E, N, K/2]` 大张量时 `htod_into(.., &mut big.slice_mut(a..b))` 类型不匹配（`CudaViewMut` ≠ `CudaSlice`）。
- 解法：`latent_moe.rs::h2d`：`stream.inner_arc().memcpy_htod(src, &mut dst.slice_mut(off..off+len))`。

## `Linear::forward` 对预分配缓冲推断行数 → decode 每步按 max_tokens 行算 GEMM（2026-10-20）

- 现象：Qwen3.5 c1 decode TPOT 24 ms，CUDA graph / GEMM 自调优都没收益；nsys 里 decode 步出现 `nvjet 128x192 2cta` 81 µs 的大 tile GEMM，每步 112 次。
- 根因：`Linear::forward(gemm, x, out)` 用 `x.numel() / in_features` 推断 M，而 qwen35 的激活缓冲按 `max_tokens`（2112）预分配 → 每步都算 2112 行，GEMM 时间 ≈ 2112/1 倍，且自调优表（M=1 键）不命中。
- 解法：所有预分配缓冲上的 GEMM 一律 `forward_rows(gemm, x, out, tokens)` 显式传本步 token 数（qwen35 7 处；K3 在 C-1 已如此）。
- 避免：新模型 crate 禁用 `Linear::forward`（考虑把它改成只接受精确形状或删掉）；nsys 中 decode 出现大 M tile 名字（`128x192` / `2cta`）即是信号；自调优日志的 µs 与 nsys 实测对不上时先查 M。

## `remote_b300.sh push --delete` 会删掉远端**空目录**，容器里的 `-v $ABS_REFS:/out` 随之写进"不存在的目录"（2026-10-20，K3 C-3）

- 现象：`k3_model.sh` 远端 `mkdir -p $REFS` 后起容器，脚本第一个 `np.save('/out/ids.npy')` 看似成功，几分钟后 `layer0.npy` 报
  `FileNotFoundError: /out/layer0.npy`，远端 `test/kimik3/2026-10-20/model` 不存在。
- 根因：另一个会话 / 子代理在这期间 `push --delete`，本地没有这个目录 → rsync 把远端刚建的空目录删了；nerdctl 的 bind mount 指向
  被删的 inode，写入全丢（容器里看不到错误直到下一次 open）。
- 解法：**本地先建目录 + `.gitkeep` 并 commit**，再 push / 起容器；参考生成完 `pull` 回本地后再做下一次 push（与 k3-kda.md 踩坑 0 同根）。
- 避免：所有会写远端生成物的脚本，REFS 目录都先在仓库里占位。多子代理共用一台远端机时，push 前 `pull` 一次对方可能生成的目录。

## 自调优表只键精确 M → 非 2 的幂 batch 全部掉回 cuBLASLt 默认，且默认路径每次 forward 都查启发式（2026-10-20）

- 现象：B300 nsys c8-long decode 步里 M=8 的 GEMM（`nvjet 64x8_64x16` 60 µs）比 M=1 自调优后的 16 µs 慢 4×；3090 真形状复现：默认启发式在 M=7/8 对 `gdn.in_proj_a`（N=48）选 31 µs 的 algo，最优 8 µs。
- 根因：① `Gemm::plan_for` 只精确命中 {1,2,4,8,16,32,64} 桶，Lt algo 不借；eager 混批 / 请求交错完成时 decode 批常是 7 / 9 / 31 → `run_lt_default`；② `run_lt_default` 每次调用都 `cublasLtMatmulAlgoGetHeuristic`，host 侧 µs 级 × 每步 ~400 次 GEMM，CPU 开销与 GPU 时间同量级。
- 解法：autotune 末尾把 1..=最大桶 的非桶 m 展开进表（借上一个 2 的幂桶的 plan，Lt algo 过 `cublasLtMatmulAlgoCheck`，日志 `borrowed=… rejected=…`）；默认 algo 按 Key 缓存。
- 避免：任何"按形状查表"的加速路径都要想清楚**运行时形状是否真的落在表键上**（CUDA graph 捕获桶是幂，eager 不是）；启动日志只打桶的 µs 对不上 nsys 时，先核对 nsys 里该 kernel 的 grid / M。`/metrics` 的 `engine_step_execute_{prefill,decode,mixed}_seconds` 可直接看混批步比例。

## tiny 模型 `serve` 要全词表：`model vocab 512 < tokenizer 248077` 直接拒绝启动（2026-10-20）

- 现象：用 `test/qwen35/golden/tiny`（vocab 512）跑 `dinfer serve --tokenizer test/qwen35/tokenizer` 启动即报 `model vocab 512 < tokenizer 248077`。
- 解法：`scripts/dev/make_qwen35_tiny.py <out> --vocab 248320 --gen 2` 另生成一份全词表 tiny（lm_head 2.5 GB bf16，放 /tmp 不进仓库），只用来量调度 / 混批 / metrics，不做数值对拍（golden 仍用 vocab 512 那份）。

## tensor core kernel 里混入"每列一线程"的串行段时，线程数 ≠ 列数 → 越界写邻接 smem（2026-10-20，第 5A 轮）

- 现象：`wy_mma_kernel` 新版 aq 与老版全等，u / w 全 0、kg 没乘衰减；parity out max_err 0.04、cos 0.08。
- 根因：block 从 256 线程改成 128（4 warp 各一个 16 行 tile），M 前代那段仍按"每线程 2 列 = BT"写，线程 64–127 写到 `sM` 之后的 `sgc` / `sb`（β、gc 被清 0）。
- 解法：串行段显式 `if (tid < BT)`，列数由 `static_assert(BT <= 线程数)` 守；定位靠 `gdn_chunk_wy_mma_vs_legacy` 直接比 scratch（aq / u / w / kg 分开看，一眼看出哪段坏）。
- 避免：mma 改写时把 block 线程数写成常量（`kWyWarps * 32`），所有 `idx += 256` 一律换成 `+= NW * 32`；每个有"状态"的中间缓冲都做一次老 vs 新的逐缓冲对拍，不只比最终输出。

## 按 token 分块的 conv1d：旧 state 的读与新 state 的写落在不同 block 时顺序无保证（2026-10-20）

- 现象（设计期发现，未上线）：序列跨多个 token 块时，首块读 conv_state 做 halo、末块写 conv_state；末块先跑完则首块读到新状态。
- 解法：规定旧状态只被"含 bos 的 block"读、新状态也只由它写（新状态 = 序列最后 W-1 个输入，从只读的 x 取，len < W-1 时用旧状态补），
  并让它额外接管 [bos, bos+W-1) 几行，后续 block 从 max(r0, bos+W-1) 起、halo 全在 x 内。前提 rows_per_block ≥ W-1（host 保证）。
- 避免：任何"按 T 分块 + 块间状态传递"的 kernel，先列出每个状态槽的读者 / 写者 block，读写不在同一 block 就要重新切分职责，不要靠启动顺序。

## B300-50：nsys 起服务报 "not enough GPU memory … free 34.6 GiB"，而卡是空的（2026-10-20）

- 现象：`--max-num-batched-tokens 16384 --gpu-memory-utilization 0.6` 下 dinfer qwen35 在空卡上报 KV 预算不足；先误判为 CUDA 设备编号与 nvidia-smi 不一致（换卡 / `CUDA_DEVICE_ORDER=PCI_BUS_ID` 后依旧）。
- 根因：qwen35 的激活缓冲按 `max_tokens`（≈ 预算 + max_num_seqs）预分配，每层 GDN / 全注意力各有多块 `[max_tokens, 10240/6144/8192…]` bf16，预算从 2048 到 16384 时这部分从 ~几 GB 涨到 ~180 GB（free 从 ~200 GiB 掉到 34.6）；再叠加 `gpu_memory_utilization 0.6` 的预算公式 `min(free, cap − used)`，cap 160 GiB < used → 0。
- 解法：① bench 用 0.9（与 vLLM 一致），nsys 也用 0.9；② 激活缓冲应全模型共享一套（qwen3 是这样），不是每层一份——性能 loop 待办；③ 启动日志打印激活缓冲总字节数。
- 避免：改预算 / max_tokens 后看一眼启动日志的 free / 激活字节；"free 很小但卡空"先怀疑自己分配了什么。`CUDA_DEVICE_ORDER=PCI_BUS_ID` 仍建议加（无害，编号稳定）。

## 每层各持一套按 `max_tokens` 预分配的激活缓冲：预算一上去显存按"层数 × 预算"爆炸（2026-10-20，Qwen35 perf 第 6 轮）

- 现象：`--max-num-batched-tokens` 从 2048 提到 16384（对齐 vLLM）后，27B 在 B300 上 utilization 0.9 只剩 ~20 GB 给 KV（块数大缩水，长上下文抢占），0.6 直接 `not enough GPU memory`，而卡是空的（见上条 "B300-50：nsys 起服务报 not enough GPU memory"）。
- 根因：`GdnLayer` / `FullAttention` / `dinfer_layers::Mlp` 各自在 `load` 里 `Tensor::zeros(&[max_tokens, 列数])`，每层一套、互不复用；按 token 数线性、按层数线性，预算 ×8 就是 ×8：48 GDN 层 ×（2×10240 + 3×6144 + …）+ 64 层 Mlp ×（34816 + 17408）bf16 × 16448 token ≈ **170 GB**。小预算 / 小模型（qwen3-0.6B、tiny）看不出来，所以一直没发现。
- 解法：层在同一条流上串行，层 i 的中间量在层 i+1 开始前已消费完，**全模型只需一套**：`models/qwen35/src/scratch.rs::ActScratch`（列数取各层类型最大值），forward 以 `&mut ActScratch` 借用；`Mlp` 自带缓冲改 `Option`（`max_tokens == 0` 不分配）+ `forward_with_scratch`。改后 ≈ 2.8 GB；启动日志 `qwen35: activation scratch bytes=… (max_tokens=…)` 直接看。
- 避免：① 任何 `[max_tokens, …]` 预分配都要问"是每层一份还是全模型一份"，层内临时量一律借全模型的一套，只有跨步存活的状态（KV、GDN slot 状态）才每层持有；② 新模型 crate 起步时就在启动日志打激活字节，并把"预算 × 8 后显存去哪了"写进验收；③ 真 27B 预算改动后看一眼 `kv_gib` 是否随预算反向缩水。qwen3 crate 目前仍是每层一套（0.6B 列数小暂无害），上大模型前要同样改。

## `remote_b300.sh push --delete` 会删掉远端由容器 / 脚本生成、尚未 pull 回来的文件（2026-10-20，两次）

- 现象：K3 子代理在 B300 生成的参考 `.npy`（`test/kimik3/2026-10-20/model/`）被主会话推 Qwen3.5 改动时清空；`.gitkeep` 只保目录不保内容。
- 根因：push 用 rsync `--delete`，远端工作目录被当成本地的镜像；多人 / 多子代理共用一个远端目录时，任何一方 push 都会删别人没 pull 的生成物。
- 解法：① 生成物落在 `$R/bootstrap/` 或 `/tmp`（push 不覆盖的目录），用完 pull；② 要 push 前先问同在跑 B300 的子代理 / 先 pull；③ 考虑给 push 加 `--exclude 'test/**/refs*/'` 之类的保护（已有 bench*/ab_*/stage* 排除，参考目录命名要进这个列表）。

## 用运行时循环变量做下标的"寄存器数组"会被 ptxas 压进 local memory：无 spill 报告，但慢 40%（2026-10-20，Qwen35 perf 第 7 轮）

- 现象：`recurrent_decode_kernel` 的 tr 变体（多值 warp 归约 `warp_sum_multi<32>`）3090 B=32 351 µs，比 base 的 248 慢 40%；`-Xptxas -v` 显示 `Used 128 registers, 0 bytes spill`，但多了 **`128 bytes stack frame`**。
- 根因：归约写成 `for (n = N, half = 16; n > 1; n >>= 1, half >>= 1) { for (i < n/2) v[i] = ... v[i + n/2] ... }`，外层循环的 `n` 虽然每轮可推出，但 `#pragma unroll` 没把它完全展开，`v[i + hn]` 变成运行时下标 → 数组整体落 local memory（stack frame ≠ spill，spill 统计为 0）。
- 解法：改成模板递归 `WarpSumMulti<N, HALF>::run` 把每一层折半展开成独立函数，所有下标编译期常量；stack frame 0，B=32 351 → 246.5 µs。
- 避免：寄存器数组的 kernel 看 `-Xptxas -v` 时**同时看 stack frame**，不只看 spill；任何"跳变步长"的循环（n >>= 1）对数组下标都要用模板递归或 `if constexpr` 手写展开。

## 测试里对非默认流做 CUDA graph 捕获报 901，哪怕缓冲都在该流上分配（2026-10-20，`rmsnorm_bench`）

- 现象：`d.new_stream()` + 在该流 `htod_copy` / `alloc_zeros` → `begin_capture` → 第一个 kernel launch 返回 `cudaError 901`。
- 根因：cudarc `device_ptr()` 默认在每次取指针时记录事件（event tracking），`cudaEventRecord` 在 thread-local 捕获模式下不允许；`graph_probe.rs` 之所以能过是因为它先调了 `dev.disable_event_tracking()`（executor 也是，见上文 Qwen35 capture 条）。
- 解法 / 避免：任何要 capture 的测试 / bench 开头 `dev.disable_event_tracking()`；非默认流仍是必要条件。

## CUDA graph 捕获体里含 H2D（`htod` / `PinnedUploader::push`）：重放时读已释放的 host 内存（2026-10-20，executor 公共件抽取）

- 现象：`DecodeGraphs` 第一版把"上传假 plan + 采样参数 + forward + 采样"整个当捕获体（预热一次、capture 内再跑一次），qwen35 golden 第一步 decode 就 `CUDA_ERROR_ILLEGAL_ADDRESS`；`compute-sanitizer` 指到 `row_gather_i32_kernel` 读 `prev[g]` 越界；NO_GRAPH 下全过；ASYNC_SCHED 开关无关。
- 根因：capture 内的 `cuMemcpyHtoDAsync`（pageable 源）被记成 memcpy 节点，**源地址是那个临时 `Vec` 的 host 指针**；重放时 Vec 早已释放，`gather_idx` / `ids` 被填成垃圾 → gather 越界。原来两个 executor 的 `capture_bucket` 都是上传在 `begin_capture` 之前、只把 forward + sample 进图，抽公共件时把顺序合掉了。
- 解法 / 避免：捕获闭包分 `CapturePhase::Prepare`（capture 外，一次）/ `CapturePhase::Body`（预热 + capture 内）；**任何进图的代码路径里不能有 host → device 拷贝**（固定地址缓冲的内容在图外换）。定位手段：`compute-sanitizer --tool memcheck <test binary> <name> --ignored`，比 `CUDA_LAUNCH_BLOCKING` 直接（异步错误只会在下一次同步点冒出来，backtrace 指向 `dtoh_sync` / `wait_read` 没有信息量）。

## nsys kernel 汇总"次数 ÷ 步数"要先确认 trace 覆盖的步：把 M=1 warmup 的 small_m 当成 M=32 的 bug（2026-10-20，Qwen35 perf 第 8d/9 轮）

- 现象：c32 decode 的 nsys 汇总里 `small_m_kernel<1>` 57192 次 × 16.2 µs，按 ~600 步除得"96 次/步 = 48 层 × in_proj_a/b"，于是认为 M=32 的 in_proj_a/b 错选了 M_TILE=1 的 small_m（预期省 1 ms/步）。
- 根因：`b300_nsys_decode.sh` 抓的是**整个 serve 进程**，含 bench 的 warmup 2 条串行请求（M=1，256 步）、128 条请求收尾时的小批段、自调优与 graph 捕获预热；M=1 步里 in_proj_qkv/z/a/b、attn.qkv/gate、out_proj、gate_up、down 的 m1 行自调优全选 small_m，这些 launch 全是 `small_m_kernel<1>`，16.2 µs 是混合均值。代码上 `small_m_kernel<1>` 只在 `m == 1` 实例化，`m > small_m_max` 在 Rust / C 两侧都直接拒绝，M=32 不可能走到。
- 解法：不改代码；第 9 轮改 GEMM 合并（in_proj_a/b 不再单独存在）。
- 避免：nsys 汇总先找一个**每步固定次数、与 M 无关**的 kernel（如 `recurrent_decode`：48 次/步）反推步数，再除；或只用 `--capture-range` 抓纯 decode 段。按模板实参（`<1>`）认 M 时先查 host 分发代码。

## 2026-10-20 CUDA graph capture 体内有同步 D2H：表现为 host 侧 shape 断言，不是 CUDA 错（K3 C-4b）

- 现象：`dinfer --features kimik3 serve` 首个 decode 步 `graph capture: body: LatentMoe: pairs 0 ≠ t·K`，引擎线程退出，之后所有请求无 chunk、`/metrics` 还能访问。
- 根因：`LatentMoe::forward` 每步 `dtoh_sync_copy(offsets)` 在 host 展平 tile 计划、`AttnRes::apply` host softmax；capture 期间 kernel 不真正执行，D2H 读回全 0，host 断言失败。错误信息完全不提 graph / stream capture。
- 解法：K3 executor decode 图默认关（`K3_GRAPH=1` 强制开）；根治 = GPU 侧 tile 计划 + AttnRes 融合 kernel（`docs/loops/k3-serve.md` TODO）。
- 避免：新模型接 `DecodeGraphs` 前先 `grep -n 'dtoh\|synchronize' models/<m>/src/*.rs`，forward 路径上一处都不能有；有则先不开图，日志明示原因。

## 2026-10-20 多线程对拍测试里的"静默"不能用进程级静态（K3 C-5）

- 现象：`KIMIK3_TP=2` 跑 model_parity，rank 0 一行数字都不打、dump 目录为空，只剩 panic 行。
- 根因：`static QUIET: AtomicBool` 让 rank 1 线程置 true 后整个进程都静默。
- 解法：`thread_local!`（每 rank 一线程）。
- 避免：多 rank 共进程的测试，所有"按 rank 区分"的状态都走线程局部或显式传参。

## 2026-10-20 MoE 模型 tp vs 单卡不能用链式 cos 做停止条件（K3 C-5）

- 现象：tp2 vs 单卡链式 final cos 0.9984、logits 0.9982，看起来像 bug；但单卡自己跑两次也只有 chain4 0.99918 / final 0.99898。
- 根因：`moe_scatter_add_f32` 原子加顺序不确定 + bf16 一次 all-reduce 舍入 → top-16 末位 margin ≤ 4e-3 的平局行翻转，链式逐层放大；隔离口径下 tp 的 attn_out / mlp_in 与单卡只差 4e-6。
- 解法：对拍口径改成"隔离各层只在**两边 topk_ids 相同的行**比（≥ 0.9999），expert_sum 一致行 ≤ 1 ulp，logits top-1 一致，链式只报数"（`scripts/verify/k3_tp_compare.py --matched-only` + `k3_tp_rows.py`）。
- 避免：带硬判决（top-k / argmax）的模型，先跑单卡 ×2 看噪声底再定门槛；要逐位就得把原子加改成按行归约（TODO）。

## 2026-10-20 远端 `pkill -f <串>` 会杀掉发出它的 ssh shell 自身

- 现象：`remote_b300.sh run 'pkill -f "while pgrep"; ...后续命令'` 后续全没执行、无输出。
- 根因：ssh 远端 shell 的命令行里含同一字串，`pkill -f` 把自己杀了。
- 解法 / 避免：启动器写成文件（`scripts/dev/k3_b300_tp_launch.sh`）再 `bash` 跑；要 pkill 用 pid 文件或 `pgrep -f 'pattern' | grep -v $$`。

## 远端 `cargo build --release -p dinfer` 忘带 `--features qwen35` → 编出 kimik3 分支二进制，Qwen3.5 配置解析报 `missing field kv_lora_rank`（2026-10-20）

- 现象：nsys 起 serve 立即 `Error: missing field kv_lora_rank at line 118`；同一二进制已被用来重启 :8012（服务没起来，bench 空跑 goodput 0.00）。
- 根因：主会话推 server 修复后在 B300 重编时漏了 `--features qwen35`（之前每次都带）；`bins/dinfer` 的 feature 决定模型分支，默认 feature 是别的模型。
- 解法：重编带 feature；**启动日志第一行打印编译进的模型 feature**，bench 脚本启动前核对 `/v1/models` 的 id 与预期一致再压（`bench_compare.sh` 已有 self-check，`bench_engine.sh` 也要）。
- 避免：远端构建命令写进 `scripts/dev/`（`q35_b300_build.sh`），不手敲。

## 压测工具自身开销把被测引擎拖慢 2×（2026-10-20，用户："测试程序的准确性、测试程序占用资源的影响"）

- 现象：同负载（rate 8、1024/256）dinfer decode 步均值 GPU 0 上 29.4 ms、TPOT 24.5；GPU 5 单独跑 14.9 ms、TPOT 18.3。
- 根因：`vllm bench serve`（Python）在 vLLM 容器里跑，rate 8 时 CPU 229–410%、192–322 线程；dinfer / vLLM / 压测端都没钉核（128 核 2 NUMA 全开放），dinfer 的异步调度线程被抢。两个工具对同一 vLLM 服务测 TTFT p99 差 15%。
- 解法：三方 `taskset` 隔离（dinfer 0-31、bench 96-127），Rust `bench`（CPU 0.3 s / 64 req）作主口径；隔离后 dinfer TPOT 23.3 / vLLM 16.9（差 1.4×，这才是引擎差距）。
- 避免：CLAUDE.md 硬性规则"测试程序本身要可信"五条；`bench_engine.sh` 自动钉核 + 写 CPU 集合进 summary。

## nsys 按 sample kernel 切步会把加载期的 memcpy 摊进"decode 步"（2026-10-20，engine-gap 第 6→7 轮）
- 现象：按 `cuda_gpu_trace` csv 以 sample kernel 为界切步后，"decode 步 H2D 17.9 次 × 336 µs = 31%"。
- 根因：权重装载（178 MB × 192 等 852 次 H2D、5.8 s）、自调优、图捕获期间的 memcpy 被归到第一批"步"里，再求平均——分母是步数、分子含加载期。稳态真值：每步 17 次 × 1.7 µs。
- 解法：`scripts/dev/nsys_steps_sqlite.py` 直接读 sqlite，稳态窗口 = 最后一次 ≥ 8 MB H2D 之后；memcpy 按 `correlationId` 关联 RUNTIME 行看 host 侧耗时；GPU 空档叠 host API 归因。
- 避免：任何"次数 ÷ 步数"的指标先确认窗口不含加载期；跨步平均前看字节直方图（336 µs 一次而 4 B–128 KB 的拷贝本就不可能）。

## `pkill -f "dinfer serve"` 会连 `nsys profile ... dinfer serve` 一起杀，没有 .nsys-rep（2026-10-20）
- 现象：bench 跑完、`nsys export` 报找不到文件，`/tmp/x.sqlite` 0 字节。
- 根因：nsys 的命令行包含被跟踪程序的全部参数，`pkill -f` 模式同时匹配 nsys 自身；nsys 被 SIGINT 后不落盘。
- 解法：`pgrep -x dinfer` + 读 `/proc/<pid>/cmdline` 匹配端口，只给 dinfer 发 SIGINT；等 `nsys profile` 进程自然退出。见 `scripts/dev/b300_engine2_ab.sh`。


## 异步调度下被抢占请求的在飞 token 被 scheduler 丢掉（2026-10-20，r8 golden 混批+抢占 case 抓到）
- 现象：Qwen3.5 tiny golden `golden_mixed_hit_and_preempt`：被抢占再恢复的请求 Q 从第 8 个输出 token 起与教师强制参考分叉（非平局，gap 0.34）；trace 看恢复步 `num_new` 比预期少 1（51 vs 52）。
- 根因：异步调度时步 n 在飞，`schedule(n+1)` 抢占了请求（`reset_for_preemption` 清 placeholders / computed、free 块）；步 n 的 `ModelRunnerOutput` 到达时 `update_from_output` 对 `status != Running` 直接 `continue`，采样 token 没进 scheduler 的序列；而 runner `finish` 已把它追加进 `token_ids`。两边差 1 → 恢复后 scheduler 按短 1 的长度喂 token，位置错位、输出错。prefix cache 关着也会发生（任何异步 + 抢占的负载），只是以前没有 case 覆盖。
- 解法：`update_from_output` 对 `Preempted` 请求照常 `append_output` + `check_stop`（vLLM `_update_from_output` 同样不区分状态），单测 `async_preempted_request_keeps_in_flight_token`。
- 避免：涉及"请求状态机 × 异步在飞结果"的分支，先想"结果到达时请求可能已不是 Running"；golden 要有一个**逼出抢占**的 case（小块池 + 两条长请求），而不是只测顺利路径。

## ASYNC_SCHED=0 下 `*_overlap` golden case 必失败是预期（2026-10-20）
- 现象：`ASYNC_SCHED=0 scripts/verify/qwen35_golden.sh` 里 `golden_greedy_overlap` / `golden_two_requests_overlap` 报 "overlap requested but no step went through the async path"。
- 根因：这两个 case 强制 `overlap = true` 并断言至少一步走了异步路径，但 executor 的 `async_sched_enabled()` 读同一个环境变量 → 关掉后不可能有异步步。
- 解法 / 避免：新 case（prefix hit / multi-turn / mixed）都按 `ASYNC_SCHED` 决定 overlap，两种模式都过；`*_overlap` 两个只在默认（异步）模式有意义，`ASYNC_SCHED=0` 时跳过它们看另外 8 个。

## 2026-10-20 多线程 TP 下 `cudaFuncSetAttribute` 的 static 缓存只对一张卡生效（K3 C-6 卡死）

- **现象**：K3 全 93 层 tp8 首请求 prefill 过、decode 第一步永不返回；5 张卡 100%、3 张 0%；日志空白。加 worker 出错日志后：4 个 rank `kda_recurrent failed: cudaError 1`（InvalidValue），其余 rank 在该层 all-reduce 等死。单卡 / tp2 × 5 层全过。
- **根因**：`kda.cu` 等 6 处用函数内 `static` 变量缓存"动态 smem 上限已设"（`cudaFuncSetAttribute(MaxDynamicSharedMemorySize)`），而该属性**按设备（当前 ctx）生效**。一进程多线程 TP、每线程一张卡并发首次调用：先到者设完就把 static 标已设，其余卡没设，动态 smem > 48 KB 的 launch 直接 InvalidValue。static 还不是线程安全的。
- **解法**：缓存改 `thread_local`（每 rank 线程 = 一张卡，一份缓存），`common.cuh` 提供统一 `set_smem_once(fn, bytes)`；或缓存键加 `cudaGetDevice`。TP worker / rank0 forward 出错当场 `tracing::error`（`models/kimik3/src/tp.rs`），否则 rank 0 卡在集合通信永远读不到 Err。
- **避免**：kernel host 侧凡 `static` 缓存与设备 / ctx 相关的状态（func attribute、occupancy 查询、cuBLAS handle）一律 `thread_local` 或按 device 键；新多卡路径先用 tp≥4 × 小层数冒烟（tp2 碰不到的并发撞车 tp8 会碰到）；`grep -n 'static int' crates/kernels/cuda/*.cu` 进 PR 自查。

## 2026-10-20（再犯）子代理 `remote_b300.sh push` 删掉另一子代理的远端产物

- **现象**：r8 在 B300 跑完 engine bench，compare 跑到第 2 组时远端 `test/qwen35/2026-10-20/engine_r8/` 整目录消失；同时出现 r9 的 probe 目录。
- **根因**：`push` = rsync `--delete`，远端有、推送方本机没有的目录一律删。两个子代理各自本机工作区不含对方的新目录。10-20 上午 K3 refs 目录同款事故已记一次。
- **解法 / 避免**：**子代理禁止 push**，代码同步发消息让主会话推（主会话工作区含全部 commit）；子代理只能 `put` 单文件；远端产物一律先落 `/tmp` 或 `bootstrap/`，由主会话 `pull` 回本机 `test/`，不要直接写远端 `test/`。写进 `knowledge/subagent-ops.md`。

## 2026-10-20 `k3_model.sh` 的 cargo test 在 DUMP 为空时根本没跑（exit 仍 0）

- **现象**：乱文取证跑 `K3_SKIP_GEN=1 k3_model.sh`，`/tmp/k3_model.log` 只有 70 字节：`line 34: RUST_LOG=info: command not found`；脚本 `rc=${PIPESTATUS[0]}` 取的是这条失败命令的 127，但外层 grep 管道吞掉后看起来像"没输出"，上游调用方按 exit 判就会误判。
- **根因**：`${DUMP:+KIMIK3_DUMP="$DUMP"} RUST_LOG=info cargo …` 在 DUMP 为空时第一个词展开为空串，bash 把下一个词 `RUST_LOG=info` 当**命令名**而不是赋值前缀。C-3 / C-5 时都设了 DUMP 或没走这条分支，所以当时的 `test result: ok` 是真的。
- **解法**：`env VAR=… ${DUMP:+…} RUST_LOG=info cargo …`（`env` 后的词全按变量赋值解析，空词无害）；脚本末尾加 `grep -q "^test result: ok" $LOG || rc=1`，让"没跑"显式失败。用户 2026-10-20：修复即可，不单独重跑（装载太久）。
- **避免**：shell 里条件展开的环境变量前缀一律用 `env`；验证脚本的 exit 必须由"产物存在 / 结果行存在"兜底，不能只信管道 rc。

## 2026-10-20 门禁 bench 六次才跑成：全是环境 / 口径，不是代码

- **现象**：4090 `bench_r9gate` 前五次没出数：① `ps -o args` 抓到 bash 包装进程；② grep 日志抓到脚本模板串（`$R/models … "$@"`）；③④ GPU 0 被别人临时起的 vLLM 占 16–20 GB，KV 块 9900 → 708，首请求 OOM；⑤ 换空卡 GPU 3 仍只 3135 块——我起 serve 用了 `--max-num-batched-tokens 16384`，门禁口径是 2048，qwen3 每层各持激活缓冲随预算线性，14 GB 挤掉 KV。
- **解法 / 避免**：门禁链写死启动参数（与上一轮门禁完全一致，不从进程 / 日志抓）；serve 起来后读 `num_blocks`，低于基线 10% 直接 abort；4090 是共享机，起测前 `nvidia-smi --query-compute-apps` 看外来进程，GPU 0 被占换 3–7。qwen3 的 per-layer 激活缓冲改全模型一套（qwen35 / kimik3 已修，第三个同款）记 TODO。

## 2026-10-20 远端链里 `pkill -f "dinfer serve"` 会杀掉自己（pattern 匹配到包装 bash 的 cmdline）

- **现象**：B300 合体测试链一发就 exit 143、输出只有 `HEAD=`；ff 失败没察觉 + 链自杀两件事叠在一起，白等一轮。
- **根因**：`remote_b300.sh run '<一长串含 "dinfer serve" 的脚本>'` 远端是 `bash -c "<整段>"`，它自己的 cmdline 就含 `dinfer serve`，`pkill -f` 把包装进程杀了。
- **解法 / 避免**：杀 serve 一律 `pkill -x dinfer`（按进程名精确匹配）；链里任何 `pkill -f` 的 pattern 不能出现在链自身文本里。另：golden（`--test-threads=1`）不要与 `cargo build` 并发跑——同 GPU 抢资源时 1/11 偶发失败，单跑 / 复跑都过。

## 2026-10-20 nsys 抓 dinfer serve 的三个坑（四次空报告）

- **现象**：`nsys profile … dinfer serve` 四次报告 0 kernel 或 0 字节，同环境抓 `gdn_parity` 微基准正常。
- **根因**：① `kill -INT <serve>` 只杀子进程，nsys 没收到信号不写报告（SIGINT 要发给 **nsys 进程**）；② CUPTI 活动缓冲没填满不 flush，只发 3–5 条请求时缓冲里的 kernel 在进程被杀时丢失（`--cuda-flush-interval=1000` 不保证），要发够请求（三长度各 10 次）并在 SIGINT 前 `sleep 20`；③ nsys 收 SIGINT 后先建 0 字节 `.nsys-rep` 再异步写入几十秒，"等文件出现"就 `tmux kill-session` 会把它杀在半路——要等 `pgrep -x nsys` 退出。
- **解法**：`scripts/dev/q35_nsys_prefill.sh` 已含三条；分析用 `nsys export -t sqlite` + `scripts/dev/nsys_prefill_by_len.py <sqlite> 0`（逐步）/ `3 16`（聚簇）。
- **避免**：nsys 脚本收尾固定三步：发够请求 → sleep 20 → `kill -INT $(pgrep -x nsys)` → 等 nsys 退出 → 再清 serve。另：实验里**不要用全局 `pkill -x dinfer`**，会杀掉 GPU 0 常驻 `:8012`（今天误杀两次），按端口 `pkill -f "dinfer serve .*--port N\b"`。

## 2026-10-20 ncu 包 27B serve 进程：装载第 24 层 CUDA OOM

- **现象**：`sudo ncu -k regex:gqa_fa --launch-skip 16 --launch-count 1 … dinfer serve` 装到 layer 24 报 `CUDA_ERROR_OUT_OF_MEMORY`，GPU 本来空（288 GB）。
- **根因**：ncu 为 kernel replay 维护内存快照 / 影子分配，`-k` 过滤只决定哪些 kernel 被**采**，被 ncu 接管的进程所有 cudaMalloc 都带额外开销；52 GB 权重 + 17 GB 快照池 + ncu 影子超过显存。
- **解法**：ncu 只包**微基准**（`gdn_parity`/`gqa_fa_bench` 这类几百 MB 的 `cargo test --ignored`），不包 serve；serve 级用 nsys（`q35_nsys_prefill.sh` / `q35_nsys_decode.sh`）看 kernel 时间分布，再用微基准复现同形状给 ncu。另：`sudo -n ncu` 无 PATH，用绝对路径 `/usr/local/cuda-13.1/bin/ncu`。

## `bench_compare.sh` 用 `./target/release/bench` + 脚本内 `./target/release/dinfer`：远端 `CARGO_TARGET_DIR` 一换，测的就是旧二进制（2026-10-21，B300 合并门禁）

- **现象**：r11a 合并后跑四组门禁与 6 K TTFT，数字与合并前**一字不差**（103 / 743 / 2257 / 578、284.3 vs 284.9），而 r11a 微基准明明 6 K 单层 2040 → 1558 µs、16 层应省 ~7.7 ms。查 `ls --time-style` 才发现实际跑的 `dinfer` 是 18:34 的旧构建，新编译的躺在 `$CARGO_TARGET_DIR/release/dinfer`（22:57）。
- **根因**：B300 上各线用 `CARGO_TARGET_DIR=/data/yaochen/infer/target_*` 分目录构建，但 `scripts/verify/bench_compare.sh` 里 `BENCH=./target/release/bench` 是相对仓目录写死的；服务端二进制也常被写成 `./target/release/dinfer`。仓目录下的 `target/` 是别人早先的构建，`ls` 看不出来，数字看起来"完全合理"。
- **解法 / 避免**：① 起服务与压测端一律用**绝对** `$CARGO_TARGET_DIR/release/{dinfer,bench}`；② 启动脚本里加前置断言 `[ "$BIN" -nt crates/kernels/cuda/<改过的文件> ] || exit 1`，并把 `bin=… <时间> src=… <时间>` 打进日志；③ 判据：本轮改动在微基准上有可量化收益，端到端却 0 变化 → **先怀疑跑错了二进制**，再怀疑归因。

## nsys 默认 CUPTI 缓冲只装得下 ~1.2 s：目标请求落在窗口外，分解出来的是启动段（2026-10-21，r11 合并后 6 K prefill）

- **现象**：用 `q35_nsys_prefill.sh` 式流程抓 6 K 单发 prefill，`nsys export` 出的报告里 `gqa_fa_v6_kernel` 只有 16 次、每次 7.7 µs（那是 1 K 预热请求的 16 层），GDN 只有 2.8 ms，整份报告 span **1.16 s**——真实 6 K prefill 一步就该 277 ms、16 层 attn 各 1.5 ms 以上，全都不在里面。同时报告里 `small_m_mma_kernel` 4968 次共 460 ms，看着像"推理路径退化"，其实是**启动自调优**试跑所有候选。
- **根因**：CUPTI 的 `--cuda-flush-interval` 只决定"多久写一次"，不决定"能装多久"；默认容量下 B300 上个把秒就满了，早段被覆盖/丢弃。启动 + 1 K 预热已经吃掉 1.1 s，6 K 那批自然没进去。`--cuda-buffer-size` **不是合法选项**（`unrecognised option`）。
- **解法 / 避免**：① 抓报告前先 `nsys stats` 或 sqlite 查 `min(start)/max(end)` 的 span，确认覆盖到目标请求的绝对时间；② 用「按 kernel 名的 count」做自检——16 层模型跑一步 6 K，`gqa_fa_*` 就该是 16 次且每次 ~1.5 ms，count 或单次时长对不上就是抓错了窗口；③ 自调优的候选试跑会污染分解，按时间分桶（`t+0.0s … t+1.1s`）把启动段与推理段切开再统计，或 `RUST_LOG` 里先读 `gemm autotune done` 的耗时。④ 报告里出现"某 kernel 几千次、总量惊人"时，先查是不是自调优。

## Qwen3 每层一套激活缓冲：预算 16384 下**直接起不来**（2026-10-21，ActScratch 第四例）

- 现象：`serve --max-num-batched-tokens 16384`，`kv_cache: auto num_blocks=3050` 之后紧跟 `Error: DriverError(CUDA_ERROR_OUT_OF_MEMORY)`，日志里 `per-layer activation buffers zeros_mib=14013`。改小预算（2048）就正常，容易误判成"卡不够"。
- 根因：`Attention::load` 与 `Mlp::load_tp` 每层各 `Tensor::zeros(&[max_tokens, …])`（qkv 4096 + attn 2048 + gate_up 6144 + act 3072 列 × 2 B = 30 KB/token/层），28 层 × 16400 token ≈ **13.7 GiB**；KV 池先分完（3050 块）之后才轮到 `xstream` 的**第二套**每层缓冲，那一步就炸——所以不是"浪费"而是起不来。
- 解法：`models/qwen3/src/scratch.rs::ActScratch`（qkv / attn / gate_up / act 四块，列宽从本 rank cfg 推），`Attention::forward` / `Mlp::forward_with_scratch` 借用；`swap_scratch` 退化成一次 `mem::swap`。改后 480 MiB，`num_blocks` 3050 → 10808。
- **共享缓冲后行跨度必须显式传本层行宽**：GEMM 的 `ld = n`、`qkv_post` 逐行原地写、attention 的 `q_stride` 三者都必须是 `(H+2Hkv)·D`；共享块比它宽是安全的（只读前缀），**不能**用 `numel / t` 反推。qwen3 里加了 `numel ≥ t × 行宽` 断言兜住。
- **别用 HTTP 端到端 greedy 判正确性**：同一二进制同一配置连跑两次，12 条里 6 条输出不同（批组成随机 + greedy 放大逐位分歧）。要确定性证据用 `real_parity::real_scratch_swap_is_bitwise_identical`（主套 / 副套 / 换回，logits 逐位比）。

## `--network host` 多实例：端口被占时 vLLM 退出，但健康检查看到的是"别人的 200"（2026-10-22，Kaon-V3 P1）

- **现象**：起第二个 oracle 容器（同镜像、同参数，只想换个进程），脚本打印"就绪"，
  抓取也正常返回，但数据与预期差了 15–18%。事后 `docker ps -a` 才发现该容器 `Exited (1)`，
  日志 `OSError: [Errno 98] Address already in use`。
- **根因**：`kaon3_oracle_server.sh` 的就绪判定是 `curl :PORT/v1/models == 200`。端口被别人占着时，
  **占用者的服务**照样返回 200，脚本于是宣布"就绪"；后续所有请求都打在了另一个配置的实例上
  （本例：以为是 BI=1 的第二个进程，实际是 BI=0 的 nobi）。
- **代价**：据此得出过一条错误结论（"跨进程噪声 15%"，并改了判据），三份抓取全部作废。
- **解法**：就绪前额外校验 `docker inspect -f '{{.State.Running}}' $ON == true`，否则 exit 3 并提示换端口。
  多实例一律**每实例独占端口**（本机 28100/28101/28102…），并在抓取前用
  `docker ps` 确认目标容器是 `Up` 而不是"端口能答"。
- **推广**：任何"服务能答 = 就绪"的健康检查，在多实例 / 端口可能冲突的环境里都不成立；
  要校验"答话的是我起的那个进程"。同类风险：`bench_compare.sh` 用旧二进制（见 2026-10-21 条目）。

## 5090 上 6 张卡全 OOM：截层 oracle 容器还占着 GPU 2–6（2026-10-22，Kaon-V3 P1）

- **现象**：`kaon3-parity.sh` 的 kernel 对拍通过，随后 5 个分片同时报
  `DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")`，一行别的信息都没有。
- **根因**：`kaon-oracle-L{3,6,12,18,24}` 五个截层 oracle 容器(每个 ~28 GB)还开在 GPU 2–6，
  每卡只剩 ~4 GB，装不下 16 GB 权重。
- **解法**：跑批量前先 `docker ps` 对照 `nvidia-smi`，清掉不用的 oracle 容器。
  脚本侧暂时靠人；更稳的做法是 verify 开头检查各卡 free memory 并给出明确提示。
- **注意**：GPU 7 是共享基线（`kaon-vllm-pegaflow`），**永远不要清**；GPU 1 是本机 oracle。

## 清理容器时把共享基线一起停了：`docker rm` 的过滤器必须点名，不能按状态扫（2026-10-22）

- **现象**：执行 "清理 oracle 与退出的旧容器" 后，GPU 7 的共享基线 `kaon-vllm-pegaflow`（:28067，跑了 6 小时）
  和 `pegaflow-server-test` 一起 Exited；GPU 7 内存归零，基线服务不可用。
- **根因**：清理命令用了 `docker rm -f` 遍历 `status=exited` 与 `name=kaon-oracle` 两个范围；
  基线容器虽然当时是 Up，但清理脚本里同时有 `docker rm -f <显式名单>` 的遗留写法，
  且 `RESTART=always` 在 `docker rm` 之后无法自愈（容器对象已删/已停）。
- **后果**：基线需要重新 torch.compile（2–3 分钟）才能服务；期间任何依赖基线的对拍都不可用。
- **解法**：清理只允许**按名字精确删除自己起的容器**（`kaon-oracle`、`kaon-oracle-b`、`kaon-oracle-L*`），
  永远不要用 `--filter status=exited` 这类范围过滤器；执行前先 `docker ps` + `docker inspect` 确认目标。
  共享基线 `kaon-vllm-pegaflow` 与 `pegaflow-server-test` 一律**只读不碰**。
- **恢复**：`docker start kaon-vllm-pegaflow pegaflow-server-test`，并确认 `docker inspect -f '{{.HostConfig.RestartPolicy.Name}}'` 仍是 `always`。

## 2026-10-22 权重字节对拍时按"逻辑分片"切转置布局，得到假的"0 差异"

**现象**：报告"我们与 vLLM 的 L11 q/k 权重字节 100% 一致（0 差异）"，据此判定修复完成并跑端到端。

**根因**：vLLM 的 FP8 线性层把 `qkv_proj.weight` 存成 **`[hidden, rows]` 的转置布局**
（L11 是 `(2816, 10240)`）。我拿 `.npy` 后直接 `W.T`，再按 `[q|k|v]` 的**逻辑行数**切片，
切出来的行与真实分片不对应 —— 于是拿"我们的 A 段"比"vLLM 的 B 段"，
两边**恰好都来自同一份 checkpoint** 时就报 0 差异，那是巧合，不是验证。

**怎么发现的**：改用"vLLM 权重 vs **checkpoint 原始字节**"直接比，发现 q 槽与原始 q 完全相同
（说明 vLLM 对 q 没做重量化），而 k 槽与原始 k 几乎全不同 —— 与我此前的"一致性"结论冲突，
才回头检查切槽方式。

**解法/避免**：
1. 字节级对拍**先确定布局**，并且至少要有一个"**与源数据直接比**"的对照分支：
   若 vLLM 某槽 == checkpoint 原始槽，它就没有被重量化；否则不该是 0 差异。
2. 报"100% 一致"之前，先证明**不一致**能被检测到（拿相邻两个分片互比，必须非 0）。
3. 转置布局的常见形态：`nn.Linear` 存成 `[out, in]`，但框架做 TP 融合后可能变成 `[in, out]`。
   看到"行数 = 你期望的 in 维"就要怀疑转置了。

## 2026-10-22 "字节与参照一致"不等于"端到端更准"

**现象**：把 FP8 重量化的中间精度从 f32 改成 fp16 后，L0/L11 的 q/k 权重字节与 vLLM
**完全一致**（这是用正确布局复核过的），但端到端 ΔNLL 反而从 **−0.0018 恶化到 +0.1055**。

**根因（未验证，两种可能）**：① 原来的 −0.0018 是"错得抵消"的假象；
② 字节修复正确但触发了第二处偏差。

**教训**：**每一处"更贴近参照实现"的修改，都必须用端到端指标复核，不能只靠字节/张量级一致就收工。**
字节一致只是必要条件。反过来说，这也意味着"与参照字节一致"不能当验收目标——
真正的目标是"与 fp64 真值更近"（见 `docs/knowledge/model-onboarding.md` §3）。

## 2026-10-23 把"换算子变体"的差异误读成"参照自复现失败"，白绕多轮

- **现象**：记录里写"连 vLLM 调它自己的 flashinfer 算子都复现不出它自己的前向（7/196608）"，
  并据此得出"逐位复刻不可达、目标只能落在 kernel 变体量级内"。
- **根因**：那 7 处其实是 **`flashinfer` vs `cutlass`**（两个不同 kernel）的差，被当成了
  "同一 kernel 重跑 vs 前向"。两件事混在一次比较里，结论完全反了。
  实测：同入参重复 6 次 **0 差异**；重调 flashinfer vs 前向真实返回值 **0/196608**。
  → **vLLM 的前向是逐位可复现的**。
- **解法**：判定必须**分成两个实验**——① 同入参重复调同一算子（确定性 + 自复现）；
  ② 同入参换 kernel 变体（变体下限）。混在一起比，就会把②读成①。
- **避免**：任何"参照自己都做不到"的结论，先问一句"我是拿同一份入参、同一个 kernel 比的吗"。
  详见 `docs/knowledge/parity-localization.md` §3。

## 2026-10-23 改 kernel/脚本后不看结构直接跑（同一类失误第 3 次，废弃两轮编译）

- **现象**：给 `gemm_fp8_sm120.cu` 加入口时，Edit 把原 kernel 声明整段替换、又留下 4 行残块；
  两次都是 sync + 远端编译后才发现，每次浪费约 6 分钟。
- **根因**：多行 Edit 之后没有回读确认；把"编辑成功返回"当成"文件正确"。
- **解法/避免**：改 `.cu` / 大段 Rust 后先自查结构再 sync：
  `grep -n '<关键符号>\|^}' 文件 | head -20` + `wc -l`，确认函数个数与大括号配对；
  改脚本同理先 `python3 -I -c 'import ast,sys;ast.parse(open(p).read())'` 或 `bash -n`。

## 2026-10-23 改了却"没生效"，且差点被当成"影响小"（第 4 次同类）

- **现象**：给 bf16 GEMM 加了按配置选 algo 的逻辑，端到端读数与上一轮**逐位相同**（13.04% / 1.2847，
  连 rows/decisive 都一样）。我第一反应是"bf16 影响不大"，实际是**改动根本没执行**。
- **根因**：`forward_bf16` → `forward_inner` → `plan_for`，其中
  `plans.get(&Key::bf16(m,n,k,bias))` **先**命中 autotune 为 m=24 生成的 `SmallM` 计划，
  走的是自写 split-K kernel，根本到不了我改的 `run_lt_default`。
  第二次加 `SMALL_M_LT` 开关也一样没生效（开关加在了后面那个兜底分支上，同样走不到）。
- **解法**：改这条路径前先读 `plan_for` 的**分派优先级**；开关必须加在**最先命中的分支**上，
  或在 autotune 阶段就不生成 small_m 计划。
- **识别指纹**：**两次独立测量的读数逐位相同**（不是"接近"而是每一位都一样）→ 几乎一定是
  "改的东西没被执行"，而不是"效果小"。此时应先做一次单层隔离（`router_logits` 是否变 0），
  用几十秒证明改动生效与否，**不要直接重跑端到端**。

## 2026-10-27 NO_GRAPH serve 起 stage 实验,就绪判定连栽6次都是"serve没问题是我grep误杀"
现象:起 NO_GRAPH+KAON3_STAGE/DENSE/MOE_TIME 的临时 serve 测分段,连续 6 次报 NOTREADY 并杀掉 serve,但日志显示 serve 每次都在 ~42s 正常 `listening addr=8144`(model_load_ms≈42000)。白加载 6 次(每次 42s)。
根因(逐个排掉):① `grep "\"data\""` 经 `ctr` 的 SSH→docker→heredoc 多层,引号转义被打乱→永不匹配;② 换 `grep "listening addr"` 仍栽——**tracing 日志带 ANSI 色码**,原始字节是 `listening \x1b[3maddr`,"listening addr"(带空格字面量)被色码打断不匹配;③ GPU4 上一轮 --rm 容器残留占 25GB,新 serve OOM 也会 NOTREADY(复用 GPU 前先 nvidia-smi 查空)。
解法:就绪判定要么 `sed 's/\x1b\[[0-9;]*m//g'` 去色码再 grep,要么干脆 `sleep 55`(serve 稳定 42s bind)不 grep。读 serve 日志一律先去 ANSI。
避免:① ctr heredoc 里别用带引号/空格的 grep 字面量(转义+色码双坑);② 判定失败先看 serve 日志证明它到底起没起,别默认"没起"就杀重来——我默认错了方向栽了6次=白等4分钟。

## 2026-10-28 nsys 两个版本都"无 kernel 表"：CUPTI 缓冲只在 cudaProfilerStop 落盘 + INT 了 nsys 本体
- 现象：`nsys profile -t cuda --cuda-graph-trace=node` 抓 dinfer serve，sqlite 只有加载期 memcpy，无 KERNEL 表；第二次连 .nsys-rep 都没有。
- 根因：① 诊断表写明 "Buffers holding CUDA trace data will be flushed on CudaProfilerStop()"，dinfer 被 INT 退出不调它；② `kill -INT nsys` 打断报告生成。
- 解法：`--cuda-flush-interval=1000`；只 INT 被测进程，`wait` nsys 自然退出。脚本 `scripts/dev/kaon3_nsys_decode.sh`。
- 避免：任何"工具抓不到"先读它自己的诊断表（DIAGNOSTIC_EVENT），别换版本死磕。

## 2026-10-28 排队等待用 `pgrep -f "bash /w/x.sh"` 自匹配 → 两条链各死锁 30 分钟；`pkill -f` 又把自己的壳杀了
- 现象：等待进程永远不结束；重启链里 `cp` 备份没执行，A 基线用了错的二进制。
- 根因：`pgrep/pkill -f` 匹配整条命令行，包含发起它的 `bash -c "... pgrep -f ..."` 自身。
- 解法：串行一条 `bash a.sh; bash b.sh`，或 `until grep -q ^DONE 结果文件`；杀进程按 PID 或 `-x`。CLAUDE.md 早有此规则，第三次踩。

## 2026-10-28 decode split-K attention scratch 按 max_tokens 分配 → 4.4 GB → warmup OOM → kv_cache 空 panic
- 现象：新二进制起服务 `warmup prefill failed OOM`，随后 `model.rs:207 index out of bounds len 0`。
- 根因：scratch = B×heads×32 split×(D+2)×4B，B 用了 max_tokens=4160（应是 max_num_seqs=64）。
- 解法：`LayerScratch::new_with_rows(max_tokens, max_rows)`，68 MB。
- 避免：按"行数"分配的缓冲都用 max_rows，按"token 数"的才用 max_tokens；新缓冲算一下字节数再上。

## 2026-10-28 `KAON3_MEGA=0`（非 mega dense 路径）与 `MEGA_SPGU=8` 输出 "is is is…"
- 现象：A/B 里这两档 TPOT 更低（11.33）但输出坏。
- 根因：未查（不在本 loop 关键路径）；非 mega dense 路径可能自 mega 落地后未再验证。
- 避免：所有 A/B 变体必抓固定 prompt 输出，输出坏的数字作废；交接区已记 dense 可省 ~1.7 ms 待专刀。

## 2026-10-28 批量编辑脚本一处 assert 失败 = 该文件全部改动丢失（3 次，各赔 10 分钟远端编译）
现象：Python 脚本连改多文件，某文件中途 assert 失败→异常退出，**前面文件已写盘、该文件一字未改**，我只看到前面的 "ok" 就 put 去编。根因：改动都在内存，写盘在末尾。解法：每文件独立写盘；脚本末尾对每个目标串打印 count；编译前 `grep -c`。

## 2026-10-28 改 extern "C" 签名漏 kernels.h（2 次）
现象：nvcc "more than one instance of overloaded function has C linkage"。根因：只改了 .cu 定义和 ffi.rs 声明。解法：kernels.h / .cu / ffi.rs（+stub!）三处同改，`grep -n '<fn>('` 核参数个数。

## 2026-10-28 对拍把两条不同 kernel 当成"同一算法两种输入"比
现象：fp8 cache vs bf16 cache 的 paged 路径 D=256 位差 135/20480，D=512 位差 0。根因：bf16 侧 D=256 默认走 fa_v5，fp8 侧被强制慢路径——差的是两 kernel 的累加序，不是 fp8。解法：对拍时 env 强制同一 cfg/kernel；D=512 两侧都慢路径所以恰好 0 是线索。

## 2026-10-28 验证链编译失败仍跑 A/B（跑了旧二进制出"结果"）+ 同一链被起两次
现象：bN.sh 编译 EXIT=101 后照常跑 B 轮，B 侧是上一版二进制 + 新 env，数字像真的；另一次 b8.sh 失败后我又起 b8b.sh，前者的 bN 还没停，RESULT 里 B/A2 重复出现。解法：bN 加 "EXIT≠0 → NOBUILD DONE 退出"；起新链前 `pgrep -ax bash | grep /w/` 看旧链；监控脚本只在 bN 段判 DONE。

## 2026-10-28 验证链 serve 进程没绑核，跑了一天 20 多轮才发现
现象：CLAUDE.md 要求三方（dinfer / vLLM / 压测端）taskset 不重叠并写进报告；bN.sh/envab.sh 只给 bench 绑了 60-91，serve 在 128 核上浮动（2 NUMA，GPU0 亲和 0-31,64-95）。根因：RESULT 文件没有钉核字段，没人核对。解法：bN/stepB_verify 加 `taskset -c ${SERVE_TASKSET:-0-31}`，RESULT 头部打印钉核口径；c8 图外时间（D75 ~4 ms/步）的归因必须先过绑/不绑 A/B。
