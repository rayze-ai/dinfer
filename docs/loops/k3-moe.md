# Loop: MXFP4 专家权重（Kimi-K3 线）

> 状态：**进行中**。3a（解包参考路径）已完成；**3b（MXFP4 grouped GEMM）v1 完成**：
> 两卡对拍通过（3090 / 4090 GPU 7），**B300（sm_103）对拍 + 计时 + 真权重逐位核对也已完成（2026-10-19）**，
> sm100 原生 FP4 路径仍留空壳 + TODO。

## 环境

- 规格来源：K3 `config.json` 的 `quantization_config`（`mxfp4-pack-quantized`，group 32，symmetric，
  `scale_dtype: torch.uint8`，`num_bits: 4`）+ vLLM `quantization/mxfp4.py:895` 的转换路径
  （`mxfp4_to_f32` → 乘 `e8m0_to_f32` → 按 group 32 reshape）。
- 对拍：`scripts/verify/gen_mxfp4_refs.py`（按位构造，不依赖 torch 的 fp4 dtype——本机 torch 2.6 没有
  `float4_e2m1fn_x2`）+ `crates/kernels/tests/mxfp4_parity.rs`。
- 真权重：B300 `/data/models/moonshotai/Kimi-K3`（96 分片）。

## 格式（已用真分片头核对，2026-10-19）

`model-00002-of-000096.safetensors` 里 layer 1 的 expert 0：

| tensor | dtype | shape | 解释 |
|---|---|---|---|
| `experts.0.w1.weight_packed` | U8 | [3072, 1792] | out=3072（moe_intermediate），in=3584/2 |
| `experts.0.w1.weight_scale` | U8 | [3072, 112] | in/32 = 112 |
| `experts.0.w2.weight_packed` | U8 | [3584, 1536] | out=3584（latent），in=3072/2 |
| `experts.0.w2.weight_scale` | U8 | [3584, 96] | in/32 = 96 |
| `experts.0.w3.weight_packed` | U8 | [3072, 1792] | 同 w1 |

→ 与 `design-kimik3.md §1` 的 latent MoE 一致：专家在 3584 维上工作，w1/w3: 3584→3072，w2: 3072→3584。
索引里 expert 张量 494868 个（= 92 层 × 896 专家 × 6），总 497220 个张量。

**打包约定**：一个字节两个 E2M1，**低半字节是偶数下标**；E2M1 幅值表 `[0, .5, 1, 1.5, 2, 3, 4, 6]`，
bit3 是符号；scale 是 e8m0，值 = `2^(u8 - 127)`（`__uint_as_float(s << 23)`）。

## 观察方法

1. `python3 scripts/verify/gen_mxfp4_refs.py <dir>`：随机全 16 种半字节 + e8m0 指数 [117,131] 的三个形状。
2. `cargo test -p dinfer-kernels --features ref-tests --test mxfp4_parity -- --ignored`：
   GPU 解包 → 与参考比，相对误差 1e-3 或绝对 1e-6（解包应逐位一致）。
3. 真权重核对：B300 上取一个真实分片的 w1/w2/w3，解包后看统计量（absmax / 是否有 NaN / 与 vLLM
   同一参数的 bf16 权重比 cos）。

## 停止条件（机器可判定）

- `scripts/verify/k3_moe.sh` exit 0：解包三 case 逐位一致 + grouped GEMM 与 torch 参考在 1e-2/0.999 内。

## 迭代上限

同一 case 最多 3 轮；总 8 轮。

## 轮次记录

| # | 日期 | 改动 | 结果 |
|---|---|---|---|
| 1 | 2026-10-19 | `crates/kernels/cuda/mxfp4.cu` 解包 kernel（u8 packed + e8m0 scale + group 32），Rust 封装，参考与对拍测试 | **3/3 case 逐位一致（max_err 0）** |
| 2 | 2026-10-19 | 用 B300 真分片头核对 K3 专家形状与 dtype | w1/w3 [3072,1792]，w2 [3584,1536]，scale 列数 = in/32 |

## 3b 规格：MXFP4 grouped GEMM（里程碑 B，2026-10-19）

### 环境

- 实现文件按 arch 分（CLAUDE.md 多卡型规则）：
  - `crates/kernels/cuda/mxfp4_gemm_sm8x.cu`：**本阶段写的**（sm80+，bf16 mma.sync + kernel 内解包）
  - `crates/kernels/cuda/mxfp4_gemm_sm100.cu`：**留空壳 + TODO**（原生 `tcgen05` / `mma.sync.m16n8k64.e2m1`）
  - 选择逻辑与 arch 无关的部分（分组、host 分派）放 `mxfp4_gemm.cu`，按 `ArchProfile` 选实现
- 复用 `moe.cu` 现有的分组设施：`moe_bucket`（count/scan/fill + `sorted_tok`/`sorted_k`）、
  `moe_gather_bf16`、`moe_scatter_add_f32`、`f32_add_to_bf16`。**不新造 token 排序**。
- 参考：torch 侧 `dequant → bf16 matmul`（`gen_mxfp4_refs.py` 扩展）。
- 规格来源：vLLM `quantization/mxfp4.py` 的 `mxfp4_to_f32` / `e8m0_to_f32`；格式细节见本文件「格式」节。

### 第一版口径（v1，求对优先）

```
out[t, n] = bf16( Σ_k bf16(dequant(w[n, k])) · bf16(x[t, k]) )      fp32 累加
```

- 按专家分组：专家 e 处理 `sorted_tok[offsets[e] .. offsets[e+1])` 这些 token 行。
- **权重在 kernel 内解包**：`weight_packed` u8 `[N, K/2]` + `weight_scale` u8 `[N, K/32]`，每 32 列一个
  e8m0 scale；解包到 bf16 后走 `mma.sync.m16n8k16` 累加，输出 bf16。
- 激活是 bf16（v1 不做激活量化），因此**不是**原生 FP4 路径；`w13`/`w2` 直接用 bf16 激活乘。
- 输出累加口径与 GLM 线一致：`moe_scatter_add_f32` 按 `(sorted_tok, sorted_k, topk_w)` 加权散回。
- BT/BM/BN/BK（tile 尺寸）是**模板参数**，由 host 按 `ArchProfile` 选，kernel 体内不写死。

### 对拍

- torch 参考：`gen_mxfp4_refs.py` 新 case `gemm_<name>`，存 `packed`/`scale`/`x`/`out`；
  参考值 = 解包成 f32 → 与 bf16 激活做 fp32 matmul → 转 bf16。
- 形状：K3 真实 `[3072, 1792]`（w1/w3：3584→3072，K=3584 打包成 1792）与 `[3584, 1536]`
  （w2：3072→3584，K=3072 打包成 1536），各配一个 M；另加小形状（M 不是 tile 整数倍、单 token、E=1）。
- 容差：bf16 输出 atol 1e-2 + cos ≥ 0.999（与 KDA 的软判决口径一致；GEMM 的 k 维较长，
  bf16 操作数会有约 1e-3 相对误差，1e-2 绝对量按输出量级定）。

### 观察方法

1. `python3 scripts/verify/gen_mxfp4_refs.py <dir>`（本机 torch 即可，参考不依赖 fp4 dtype）。
2. `MXFP4_REFS_DIR=<dir> cargo test -p dinfer-kernels --features ref-tests --test mxfp4_parity -- --ignored --nocapture`
3. `scripts/verify/k3_moe.sh` 汇总，exit 0 = 通过。

### 停止条件（机器可判定）

- `k3_moe.sh` exit 0：解包 3 case 逐位一致（保持原状）+ grouped GEMM 全部 case 在 1e-2 / cos 0.999 内。
- `cargo build --workspace --features qwen3` 仍 exit 0。

### 迭代上限

同一 case 3 轮；总 8 轮。

### sm_103（B300）版的走法（只记录，本阶段不实现）

- 按 arch 分文件：`mxfp4_gemm_sm100.cu`，同样由 host 按 `ArchProfile` 选。
- 原生路径要点：
  1. 权重保持 packed（`[N, K/2]` u8）直接进场，**不解包成 bf16**——这是省显存的关键
     （896 专家 × 3 张量 × ~5.5 MB bf16 每层 ≈ 14 GB，92 层 ≈ 1.3 TB）；
  2. 用 sm100 的 block-scaled MMA（`tcgen05.mma` 的 MXFP4 变体，scale 由 `tcgen05` 的 scale 因子
     通道喂入；或用 `mma.sync.aligned.m16n8k64.row.col.kind::mxf4` 一类的指令，视 CUDA 13 头文件而定）；
  3. 激活也要量化成 MXFP4（group 32 + e8m0），量化 kernel 与 GEMM 融合（A 侧量化在 tile 载入时做）；
  4. 不能用 `#if __CUDA_ARCH__` 在 kernel 体内分叉——两条路径是两个文件、两套入口，
     host 侧按 `ArchProfile` 选（与 `gqa` / `gqa_decode_v3` 的既有做法一致）。
- 门禁：两张卡（sm89 / sm103）都要过 `bench_gate.sh`，任一退化 > 3% 不合入。

### 轮次记录

| # | 日期 | 改动 | 结果 |
|---|---|---|---|
| 1 | 2026-10-19 | 规格定稿（本节） | 规格定稿 |
| 2 | 2026-10-19 | `mxfp4_gemm_sm8x.cu` tile kernel（权重分块解包到 shared，256 线程 × 每线程 4 输出列的 4 路 FMA）+ `mxfp4_gemm_sm100.cu` 空壳；`kernels.h`/`ffi.rs`/`ops.rs` 加 `Mxfp4GemmTiles` 与入口；参考侧加 3 个 GEMM case | **3/3 通过**，见下表 |
| 3 | 2026-10-19 | B300-50（sm_103）`CUDA_ARCHS=103` 重编 + `k3_moe.sh`（`MXFP4_SKIP_GEN=1`） | **2/2 通过，exit 0**，数值与两卡一致；sm100 空壳照常编，无 arch 编译问题 |
| 4 | 2026-10-19 | 新增 `mxfp4_gemm_bench`（E=8 / 256 pairs / 真形状，release，5 次均值）+ 解包 bf16→cuBLASLt 参照 | **B300 w1/w3 1.67 ms、w2 1.61 ms**；3090 5.50 / 5.20 ms；参照 0.06 ms（v2 差距 28×） |
| 5 | 2026-10-19 | 真权重解包核对：`k3_mxfp4_realcheck.py`（纯 numpy）+ `mxfp4_realcheck.rs`（`#[ignore]`，`MXFP4_REALCHECK_DIR`）+ `k3_realcheck.sh` | **L1 e0 w1/w2/w3 各 11,010,048 元素逐位一致（max_err 0）**；scale u8 ∈ [112,122]，无 0/255 |

### 实测（2026-10-19，`k3_moe.sh` exit 0，3090 与 4090 同结果）

| case | E | N | K | pairs | tiles | \|out\|max | max_err | rel | cos |
|---|---|---|---|---|---|---|---|---|---|
| `gemm_small`（含空专家、单 token、M 非 tile 整数倍） | 4 | 128 | 256 | 82 | 14 | 12.56 | 6.3e-2 | 5.0e-3 | 0.9999966 |
| `gemm_w1_like`（K3 真实 w1/w3 形状） | 3 | 3072 | 3584 | 109 | 384 | 42.5 | 2.5e-1 | 5.9e-3 | 0.9999967 |
| `gemm_w2_like`（K3 真实 w2 形状） | 2 | 3584 | 3072 | 52 | 280 | 41.3 | 2.5e-1 | 6.1e-3 | 0.9999967 |

解包 3 case 仍**逐位一致**（max_err 0，无回归）。

### B300（sm_103）实测（2026-10-19，B300-50 GPU 0，CUDA 13.1，release）

对拍：2 个 test 全过（exit 0），数值与两卡一致（解包 max_err 0；GEMM rel 4.98e-3 / 5.88e-3 / 6.06e-3，
cos 0.9999966–0.9999967）。sm_103 上没有编译问题——sm8x 路径本身 arch 无关，`mxfp4_gemm_sm100.cu` 空壳照常编。

计时（新加 `mxfp4_gemm_bench`：E=8，每专家 32 token = 256 pairs，w1/w3 与 w2 真形状，5 次均值）：

| 卡 | w1/w3（N=3072 K=3584，768 tiles） | w2（N=3584 K=3072，896 tiles） | 解包 bf16×8（GPU 侧，一次） | bf16 cuBLASLt ×8（参照） |
|---|---|---|---|---|
| 3090（sm86） | 5.50 ms | 5.20 ms | 0.7 ms | 0.30 ms |
| **B300（sm_103）** | **1.67 ms** | **1.61 ms** | **0.4 ms** | **0.06 ms** |

**关键对比**：解包成 bf16 后走 cuBLASLt 的单层 GEMM 在 B300 上只要 0.06 ms，比 v1 标量路径快 **28×**。
这是 v2（`mma.sync` 或 sm100 原生 FP4）要补的差距——目前 v1 完全没有性能门槛，这个数字只作基线。
注意参照里的 0.4 ms 是**这 8 个专家的权重解包一次**的 GPU 时间；K3 每层 896 专家（≈ 45 ms/层，92 层），
而且 bf16 解包后放不下（见上文「显存」），
所以真路径必须让权重保持 packed，这正是 sm100 原生 FP4 的价值。

### 容差口径（重要，与 KDA 一致）

逐元素 **atol 1e-2 不适用**：输出量级随 K 与 e8m0 scale 变化（本组 case \|out\| ~ 1e1），
bf16 输出本身的粒度就是**相对 2^-8 ≈ 4e-3**。因此断言用 **max_err / \|out\|max ≤ 1e-2 + cos ≥ 0.999**；
实测 5.0–6.1e-3 正好是 bf16 输出的舍入量级（不是 kernel 的数值误差）。
第一版参考把 e8m0 指数取在 [120,130)，输出到 \|out\| ~ 7e2，逐元素 atol 判据失真——已改为真权重量级 [118,126)。

### 实现要点 / 已知限制

1. **v1 是"解包 + bf16 标量乘加"**，不是 tensor core：`gemm_tile_kernel` 里每线程 4 个输出列、
   内层扫 BK 做 FMA。求对优先，速度未做门槛（本阶段无性能指标）。
2. tile 尺寸 `BM=16 / BN=64 / BK=64` 是 host 侧常量（模板参数）；`BM*BN` 必须整除 256 线程
   （`static_assert`），v2 换 mma 时一并重定。
3. **tile 计划在 host 侧展平**（`Mxfp4GemmTiles::new` 读 `expert_off`，每专家 `ceil(ntok/BM) × N/BN` 块）。
   这是每步一次 host 计算 + H2D；深批（896 专家）下需要改成 GPU 侧生成，见 TODO。
4. 权重布局按 `[E, N, K/2]` 连续（专家维最外层），与 K3 的 per-expert 张量排布一致；
   真权重加载时要按这个顺序拼成一个大张量（`weights.rs` 的事，里程碑 C）。

### TODO（v2 / 后续）

1. **tensor core**：换成 `mma.sync.m16n8k16`（复用 `gqa.cu` 的 `ldsm_x4` / `mma16816`），
   A=激活 tile、B=解包后的权重 tile；预期 tile 内吞吐提升数倍。
2. **GPU 侧 tile 计划**：避免每步 host 读 `expert_off` + H2D。
3. **原生 FP4**：sm100 路径（见上节）。

### 真权重解包核对（2026-10-19，B300-50）

链路：`scripts/verify/k3_mxfp4_realcheck.py`（纯 numpy + safetensors header，`python3 -I`，无需 torch）
读 L1 / expert 0 的 w1/w2/w3 `weight_packed` + `weight_scale`，按本文件的解包口径解到 f32 作参考；
`crates/kernels/tests/mxfp4_realcheck.rs`（`#[ignore]`，`MXFP4_REALCHECK_DIR` 未设则 skip）
读同一批张量走 `mxfp4.cu` 解包，**逐位**比。外壳 `scripts/verify/k3_realcheck.sh`。
`crates/` 里只有「读目录 + manifest + 任意 `*.weight_packed`」的逻辑，模型名 / 层号全在脚本参数里。

**结果：3/3 逐位一致**（w1 3072×3584、w2 3584×3072、w3 3072×3584，各 11,010,048 个元素，
`|got-ref|max 0`，`B300 0.65–0.69 s/张量`）。

scale 统计（`scripts/verify/k3_mxfp4_scale_stats.py`，36 张量 = L{1,12,45,91} × e{0,447,895} × w{1,2,3}，
共 12,386,304 个 e8m0）——**sm100 原生 FP4 路径要用的事实**：

- u8 范围 **[112, 122]**，只用 11 个指数值；**u8=0 与 u8=255 一次都没出现**（0/12.4M）。
  → 原生路径不需要处理 e8m0 的两个退化值（0 在 `s<<23` 口径下是 0.0，255 是 inf/NaN）。
- 众数 121（`2^-6`，84.2%）、次之 120（`2^-7`，14%）、122（`2^-5`，0.7%）；值域 `2^-15 .. 2^-5`
  = 3.05e-5 .. 3.125e-2。
- 分布高度集中：**121±1 覆盖 99.0%** → descale 后的权重可近似落在 fp32 的窄指数带里，
  但 MXFP4 GEMM 的 mma 仍须按 per-group scale 走，不能硬编码指数。
- 解包后 |w|max = 0.125，无 NaN/Inf，零值（E2M1 全零半字节）占比 11.5–12.0%。

## 待办（3b）

1. **grouped GEMM**：先 dequant 到 bf16 再走 cuBLASLt 的 grouped/batched 路径（每专家一个 GEMM，
   按 token 分组），求对；原生 MXFP4 GEMM（sm100 的 `tcgen05.mma` / mxfp4 指令）留 v2。
2. **显存**：896 专家 × 3 × ~5.5 MB（bf16 解包后）每层 ≈ 14 GB，92 层解包后 ≈ 1.3 TB —— 不能全解包。
   bf16 回退只用于**单测 / 小批专家**；真跑要保留 packed 形式（原生 MXFP4 或按需分块解包）。
3. **激活量化**：专家输入需要量化成 MXFP4（group 32）才能走原生路径；参考路径直接用 bf16 激活。

## 阻塞记录

- ~~2026-10-19 收尾时 B300 8 卡被别的作业占满~~ → **当晚 21:00 卡全空，真权重解包核对已完成**（逐位一致，见「真权重解包核对」）。
- 尚未做的：与 vLLM 的 bf16 参数逐元素比 cos（需要 vLLM 侧同参数解包或原始 bf16 权重，K3 只发布 MXFP4，
  这条可能没有参考可对，后续并入里程碑 C 的真权重 forward 对拍一起做）。
