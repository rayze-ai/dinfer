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

## 里程碑 C-2：第 1 层整层真权重对拍（KDA + latent MoE + AttnRes，2026-10-20，B300-50）

> 状态：**完成（2026-10-20，`k3_layer.sh` exit 0，B300-50）**。crate 侧 `models/kimik3/src/{latent_moe.rs,layer.rs}`，
> 测试 `models/kimik3/tests/layer_parity.rs`，外壳 `scripts/verify/k3_layer.sh`（exit 0 = 过），参考 `scripts/verify/gen_k3_layer_refs.py`。

### 与 HF 代码核对（`modeling_kimi_linear.py`，唯一口径）

| 项 | HF | 我们 |
|---|---|---|
| 路由 | `KimiMoEGate`：f32 `linear(x, gate.w)` → sigmoid → `+e_score_correction_bias` 选 top-16（`sorted=False`）→ 权重 = **不含 bias 的 sigmoid 分数** / Σ（`moe_renormalize`）× `routed_scaling_factor`(1.0)；`num_expert_group=1` → 无分组 | `moe_gate_f32` + `moe_route_f32`（GLM 线同 kernel，同式；相等取小下标） |
| latent | `routed_expert_down_proj`(7168→3584) → 专家 → `routed_expert_norm`(KimiRMSNorm, 3584) → `routed_expert_up_proj`(3584→7168) | `Linear` / `rmsnorm_bf16_n` / `Linear` |
| 专家 | `KimiBlockSparseMLP`：`w2(situ(cat[w1 z, w3 z]))`，**w1 = gate 在前、w3 = up 在后** | 装载时 w1/w3 拼成 `w13 [E, 2I, L/2]`，一条 `mxfp4_gemm_grouped` 出 `gate\|up` → `situ_and_mul` → 第二条 grouped GEMM（w2） |
| 加权和 | `moe_infer`：bf16 专家输出 `.type(topk_weight.dtype)`(f32) `.mul_(w).sum(dim=1)` → bf16 | `moe_scatter_add_f32`（bf16 y × f32 w，f32 累加）→ `f32_add_to_bf16` |
| shared | `KimiMLP`（intermediate 2×3072，`cat[gate, up]` → situ → down），加到 routed 结果上 | `dinfer_layers::Mlp` + `Situ: Act` |
| situ | `beta·tanh(g/beta)·sigmoid(g) · linear_beta·tanh(u/linear_beta)`，全 f32，beta 4.0 / linear_beta 25.0 | 新 kernel `activation.cu::situ_and_mul_bf16`，beta / linear_beta 走 host 参数（config），kernel 内无常量 |
| AttnRes | `_forward_attn_residual`：`h = res(B∪{x})`（B 非空）→ `a = attn(norm(h))` → **`p = x + a`**（残差加 x 不是 h）→ `m = res(B∪{p})` → `y = mlp(norm(m))` → `out = p + y`；写块层（`layer%12==0`）把 x 追加进 B 且 p = a | `layer.rs` 同序；`AttnRes::apply` = 每候选 `rmsnorm` + `moe_gate_f32`(E=1) 取 f32 分数 → host softmax → `moe_scatter_add_f32` 加权和。**第 1 层的 B = {x₀}**：HF 里 x₀ 是第 0 层输入（embedding 输出），本轮对拍用随机 x₀（seed+1）——不是零初始，避免 norm 退化 |
| 第 0 层 | `B = ∅`：`h = x`，不走 attention 门；写块 | `blocks.is_empty()` 分支 |

与任务书不同之处：
1. **专家权重形状**：任务书写"w1/w3: 3584→3072"正确，但 HF 里专家输入是 `routed_expert_down_proj` 的输出 `z`，**路由分数却在 7168 维的 `mn` 上算**（gate 在 down_proj 之前）。
2. **AttnRes 不依赖第 0 层的"状态"**，只依赖第 0 层**写入的块残差 x₀**（即第 0 层输入）。零初始会让 `norm(0)` 全 0、分数 0，门退化成均匀混合；所以参考与测试都用随机 x₀。
3. shared experts 的 `gate_proj/up_proj` 在真分片里是 `[6144, 7168]`（2×3072 已合并成一个 MLP），不是两个独立专家。

### 环境

- 权重 `/data/models/moonshotai/Kimi-K3`（物理路径），第 1 层全在 `model-00002-of-000096.safetensors`（16 GB：含 896 专家）；
  Rust 侧 `shards_for_layer` 只 mmap 这一片，`ExpertPack::load` 按专家 896×6 次 H2D 拼成 4 个大张量。
- 参考：vLLM 容器（`vllm/vllm-openai:nightly`，GPU 3），fla kda 复用 C-1 脚本；MoE 纯 torch（只解包被路由到的专家，
  `dequant_mxfp4` 与 `k3_mxfp4_realcheck.py` 逐位同口径），Rust 测试 GPU 2（`--release`）。
- 输入：`x ~ N(0, 0.5²)` bf16 `[64, 7168]`（seed 0）、`x₀`（seed 1）。

### 结果

`scripts/verify/k3_layer.sh` **exit 0**（B300-50，2026-10-20；日志 + .npy 归档 `test/kimik3/2026-10-20/layer/`，18 MB）。
T=64、第 1 层、单 slot、`--release`。rel = max_err / max|参考|。

| 比较 | rel | mean_abs | cos | 门槛 |
|---|---|---|---|---|
| attn 门 `h = AttnRes({x₀,x})` vs HF | 4.59e-3 | 2.6e-5 | 0.9999998 | cos ≥ 0.999 |
| attn out vs fla | 9.87e-3 | 2.5e-5 | 0.9999514 | cos ≥ 0.999 |
| prefix `p = x + a` vs HF | 3.36e-3 | 2.5e-5 | 0.9999999 | — |
| mlp 门 `m = AttnRes({x₀,p})` vs HF | 4.59e-3 | 3.6e-5 | 0.9999998 | cos ≥ 0.999 |
| post_norm `mn` vs HF | 6.17e-3 | 4.4e-5 | 0.9999958 | — |
| router logits（f32）vs HF | 4.21e-3 | 5.5e-4 | 0.9999956 | cos ≥ 0.999 |
| **top-16 命中** | pair 级 **1021/1024（99.71%）**；可判定行（margin > 2e-3）23/64 **全中**；路由整行一致 61/64，不一致 3 行全是平局 | | | 可判定行全中 |
| 专家加权和 `ysum`（路由一致 61 行） | 3.65e-3 | 2.0e-5 | 0.9999860 | cos ≥ 0.999 |
| routed `up_proj`（61 行） | 3.76e-3 | 4.4e-5 | 0.9999802 | cos ≥ 0.999 |
| MoE out（+shared，61 行） | 7.09e-3 | 4.8e-5 | 0.9999782 | cos ≥ 0.999 |
| （参考）`ysum` / routed / MoE out **全部 64 行** | 1.45e-1 / 7.98e-2 / 7.54e-2 | — | 0.99851 / 0.99840 / 0.99854 | 不判（平局行专家不同） |
| **整层 `out` vs HF** | **5.07e-3** | 1.6e-4 | **0.9999990** | cos ≥ 0.999 |

读法：所有子块 rel 在 4–10e-3 = bf16 一 ulp 量级，无系统性偏差；MXFP4 专家 GEMM（kernel 内解包 + bf16 乘加）对 torch
"解包→bf16 linear" 参考 3.7e-3。3 个路由翻转 pair 全在 `choice` 第 16/17 大相差 ≤ 2e-3 的行（bf16 输入已足以翻转），
这是硬判决不可逐位复现的既定口径（pitfalls 新条目）；对整层只有 1e-3 量级影响（整层 cos 0.999999）。
参考侧：64×16 pairs 命中 400 个专家，torch 解包+算 9.5 s；attn / mlp 门对 prefix 的平均概率 0.494 / 0.498（两候选接近均匀——随机 x₀ 的预期）。

| 事实 | 数值 |
|---|---|
| 装载（mmap 16 GB 分片 → 896×6 次 H2D + 其余 bf16） | 2.0 s，显存 +17.11 GB（专家 packed 15.72 GB + bf16 投影/shared ~1.4 GB） |
| 单层 T=64 耗时（B300，release） | 首轮 194–204 ms（含 cuBLASLt 首次 / tile 计划），**之后 5 次均值 39.5 ms**（含 MoE 1 次 D2H + AttnRes 2 门 × 2 候选 = 4 次 D2H + host softmax） |
| 每 token 专家 GEMM | 16 × 2 = 32 次；grouped 实现每步 2 条 kernel |

### 性能事实（单层，T=64，B300）

- 每 token 专家 GEMM：top-16 × 2（w13、w2）= **32 次**；grouped 实现每步只有 **2 条 kernel**（tile 数 = Σ_e ceil(n_e/16) × N/64）。
- 专家权重 packed 常驻 **15.72 GB/层**（计算见 design-kimik3 §2.2）；不整层解包。
- 每步 host 同步：MoE 1 次（读 offsets 展平 tile 计划）+ AttnRes 2 门 × R 次（取分数做 host softmax）。两者都是 C-2 的"先对"口径，
  融合（GPU 侧 tile 计划、attn_res 融合 kernel）记入 TODO。

### TODO（C-2 之后）

1. AttnRes 融合 kernel（vLLM `models/kimi_k3/nvidia/ops/attn_res.py`）：去掉 R 次 D2H + host softmax。
2. grouped GEMM tile 计划 GPU 侧生成；专家 GEMM 换 tensor core（见「TODO（v2 / 后续）」）。
3. `ExpertPack::load` 896×6 次小 H2D → 先在 host 拼好再一次 H2D 或 pinned 流水（装载时间见结果表）。

## 阻塞记录

- ~~2026-10-19 收尾时 B300 8 卡被别的作业占满~~ → **当晚 21:00 卡全空，真权重解包核对已完成**（逐位一致，见「真权重解包核对」）。
- 尚未做的：与 vLLM 的 bf16 参数逐元素比 cos（需要 vLLM 侧同参数解包或原始 bf16 权重，K3 只发布 MXFP4，
  这条可能没有参考可对，后续并入里程碑 C 的真权重 forward 对拍一起做）。

## vLLM 基线（B300-50，2026-10-19 夜）

- `vllm/vllm-openai:nightly`（0.26.1rc1.dev1177）支持 `KimiK3ForConditionalGeneration`（`tokenizer_mode=kimi_k3` 自动，`--trust-remote-code`）；prefix caching 开时默认 `Mamba cache mode = align`（KDA 递归状态按块对齐缓存——我们 Qwen35 / K3 都还没有，见 design §11 台账）。
- **tp=6（GPU 2–7）失败**：`AssertionError: 4096 is not divisible by 6`，栈在 `kimi_k25_vit.py:331 → linear.py:467`，是**视觉塔**的 4096 维列切，不是文本侧。MXFP4 权重 1561 GB，4 卡放不下 → 只能 **tp=8 独占 8 卡**（已停掉 GPU 0/1 的 Qwen35 两端后启动）。
- 基线口径：`bench_vllm_only.sh test/kimik3/2026-10-19/bench_b300_vllm1 Kimi-K3`（`VLLM_URL=:8014`），四组同 Qwen35。
- **基线已跑（2026-10-19 23:2x）**：tp8，TPOT c1 8.70 / c8 13.09 / c32 19.15 / c8-long 13.11 ms，TTFT 193 / 368 / 421 / 508 ms，详见 `docs/design-kimik3.md` §0.1。容器已 stop，8 卡释放。
- 2026-10-19 23:3x B300-50 GPU 2 复测（release，CUDA_ARCHS=103）：KDA parity 11/11、MXFP4 parity 3/3；`kda_chunk_bench` 顺序 36.3 → chunk 4.90 ms（7.41×），与白天数字一致。
