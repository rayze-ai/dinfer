# Loop: MXFP4 专家权重（Kimi-K3 线）

> 状态：**进行中**。里程碑 3a（解包参考路径）已完成；3b（grouped GEMM）待做。

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

## 待办（3b）

1. **grouped GEMM**：先 dequant 到 bf16 再走 cuBLASLt 的 grouped/batched 路径（每专家一个 GEMM，
   按 token 分组），求对；原生 MXFP4 GEMM（sm100 的 `tcgen05.mma` / mxfp4 指令）留 v2。
2. **显存**：896 专家 × 3 × ~5.5 MB（bf16 解包后）每层 ≈ 14 GB，92 层解包后 ≈ 1.3 TB —— 不能全解包。
   bf16 回退只用于**单测 / 小批专家**；真跑要保留 packed 形式（原生 MXFP4 或按需分块解包）。
3. **激活量化**：专家输入需要量化成 MXFP4（group 32）才能走原生路径；参考路径直接用 bf16 激活。

## 阻塞记录

- **2026-10-19 收尾时 B300 8 卡被别的作业占满**（llm-d PD 推理，每卡约 248 GB），真权重上的
  「解包后统计量 / 与 vLLM bf16 参数比 cos」这一步没做。分片头核对（形状、dtype）已完成。
- 真权重解包对比需要 ~14 GB/层的 bf16 缓冲，等卡空出来再跑单层。
