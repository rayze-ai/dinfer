# 逐位复刻参照算子（bit-exact replication）

> 与 `parity-localization.md` 的分工：那页讲"差异在哪"，这页讲"定位到某个算子之后，怎样写到与参照逐位相同"。
> 证据全部来自 Kaon-V3 对 vLLM（`VLLM_BATCH_INVARIANT=1`，sm_120），`test/kaon3/2026-10-23/D-16-REBASE.md` §51–§55。

## 0. 先证明逐层参照与端到端判据是同一口径（2026-10-23，最贵的一课）

**判断**：抓逐层参照的进程，必须先用它直接出 `prompt_logprobs` 对端到端 oracle，**0 差异**才往下做。
**证据**：Kaon-V3 对 eager vLLM 30 层逐位 0，端到端纹丝不动——oracle 服务是 compile 模式，vLLM 自身 eager↔compile 差 |Δlp| 1.57（§58）。
**怎么用**：抓取脚本的 LLM 参数逐项抄服务（含**是否 enforce_eager**、max_num_seqs、batched tokens、block、BI）；hook 依赖 eager 时改用截层 + 二分。

## 1. 真值是参照实际执行的机器码：源码 → PTX → SASS，逐级下钻

**判断**：源码只能用来猜结构，PTX 能看出顺序，**只有 SASS 是真值**。
**为什么**：Kaon-V3 的 RMSNorm 连写四版：照源码 `sum_sq / n_cols` + `1/sqrt` 写（§52）、照 PTX `div.full` + `sqrt.approx` 写（§55 第一版），
都在新数据上被推翻。SASS 才显示 `div.full(ss,n)` 被拆成 `RCP(n)` 并和 `+eps` 融合成一条 `FFMA`，`div.full(1,rms)` 是 `RCP`。
Triton 用的是**自带 ptxas**（`triton/backends/nvidia/bin/ptxas{,-blackwell}`），和我们 nvcc 的 ptxas 不同，**同一段 PTX 会编出不同 SASS**。
**怎么用**：
- `TRITON_CACHE_DIR=<dir>` 跑一次参照算子 → 取 `<dir>/*/<kernel>.ptx` 和 `.cubin`；
- 用 Triton 自带 `nvdisasm`（容器里没有系统版）反汇编 cubin，看 `MUFU` / `FFMA` / `FMUL` 链；
- 我们这边**每一步都写显式 PTX 指令**（`rcp.approx.ftz` / `sqrt.approx.ftz` / `__fmaf_rn` / `__fadd_rn`），不赌编译器怎么合并。

## 2. 编译器调度也会改累加顺序

**判断**：线程内归约的顺序要看 PTX，不能按"元素 0..7"假设。
**证据**：Triton 的 8 元素平方和实际顺序是 `mul(e1)` → `fma(e0)` → `fma(e2..e7)`；按 0..7 写，3 行的 inv 偏 1 ulp。
warp / block 的结构（每 warp 蝶形 → 4 warp 蝶形 → 块间顺序加）也必须照搬，否则只是碰巧对上。

## 3. 观测参照时，探针本身会改变结果

**判断**：在参照 kernel 里多存一个中间量，编译器的合并行为就会变，结果跟着变。
**证据**：复制 Triton RMSNorm kernel，把 `rms` 存出来，`inv` 就从 0.8536648 变成 0.8536649（不存的版本里 RCP 和 SQRT 的合并方式不同）。
**怎么用**：探针只能存**最终用到的量**（ss、inv）；需要看中间量时，以 SASS 为准，不以探针输出为准。

## 4. 用"允许区间"反推，先排除输入问题

**判断**：某行输出不一致时，枚举标量（inv、ss）的 ulp 偏移，看是否存在一个值能让**整行**全部一致。
- 存在 → 输入没问题，差在标量算法；区间端点能告诉你偏差有多大；
- 不存在 → 输入就不一样，先去查数据来源。
**补充**：拼接来的输入（别的脚本抓的层输出、上游 hook 的输出）要和**本算子 hook 的 `inp[0]`** 逐位比一次，再下结论。

## 5. 先确认参照走的是哪条路径，并用参照自己的函数复现

**判断**：把真实输入喂给参照自己的函数（`rms_norm_batch_invariant`、`F.linear`），看能否复现前向的真值。
**证据**：
- 独立进程里 `F.linear` 和前向的 router 差 748，因为没开 BI；按 BI 的设置（`preferred_blas_library("cublaslt")`
  + 禁降精度归约 + `CUBLASLT_WORKSPACE_SIZE=1`）就为 0（§51）；
- 前向的 RMSNorm 是 BI Triton kernel，不是 `forward_native`（后者先舍入 x·inv，会差 719）；vLLM C 版 `rms_norm` 会错在和我们一样的位置。
**怎么用**：不要拿"离线单独调一次 F.linear"当参照，那个进程的全局开关和服务不同。

## 6. 第三方库 algo：短描述不够，要看完整属性

**判断**：cuBLASLt 的 `id/tile/splitk` 相同，累加序仍可能不同；关键是 `REDUCTION_SCHEME`（OUTPUT_TYPE = 部分和先舍入到输出类型）。
**证据**：`id=21 tile=11 splitk=8` 有两个候选，`red=4` 差 801/3072，`red=2` 为 0（§51）。`Gemm::algo_desc_full` 会打印全部属性。

## 7. attention：同一个数学式，指数参数写法不同就会错

**判断**：`ex2(s·log2e − m₂)` 和 `ex2((s − m)·log2e)` 舍入不同；Triton `tl.exp(S − m)` 是后者。
**证据**：v6 改成自然域后，Kaon-V3 L4 隔离从 4457 降到 1（§54）。默认已开（`ATTN_MODE` bit1）。
