# loop 规格：kaon3-compile

- 状态：**进行中**（2026-10-23 创建）
- 创建：2026-10-23
- 验证脚本：`scripts/verify/kaon3-compile.sh`
- 迭代记录：`test/kaon3/2026-10-23/kaon3-compile.md`
- 上游：`docs/loops/kaon3-parity.md`（子 loop D）、`docs/loops/kaon3-gelu.md`

## 口径（用户 2026-10-23 明确）

**全程 compile 模式，环境必须干净，不许拿 eager 当靶子、也不许用 eager 抓的中间量做判据。**

- 参照 = vLLM `torch.compile`（服务实例不设 `enforce_eager`；vLLM 默认即 compile，
  `vllm/config/vllm.py:1379`）。生产也是它。
- 我们侧 = `RMSNORM_MODE=0`（compile 口径的 norm），**探针与整跑一律带这个默认**。
- eager（`RMSNORM_MODE=1` / `enforce_eager=True` / `kaon3_vllm_hidden.py` 抓的 `/w/hid2/*`）
  只能用于"诊断哪种实现更接近"，**不得写进任何判据、不得作为 FLOOR 分母**。
- 已知不变量：compile 逐位确定（同配置两进程 → 0.0000），所以**目标是逐位 0，噪声底为 0**。

## 目标

`models/kaon3` 的 Rust 前向在 **p02 等 30 个用例**上与 **compile 参照**（`test/kaon3/oracle_clean`）
达到 FLOOR 线：可判定行 top1 翻转率 ≤ 1.5×FLOOR + 3pp、共有 token `|Δlp|` ≤ 1.5×FLOOR，
自然长文本 `|ΔNLL|` ≤ max(3×|FLOOR|, 0.01)。现状 **flip 16.56% / |Δlp| 1.1994**（FLOOR 0.83% / 0.0276）。

## 环境

- 机器：5090，`scripts/dev/remote_5090.sh`（ControlMaster；断了向用户要 token）。cwd=`/w/dinfer`。
- GPU 分工：**0 / 2–6 = 我们**，**1 = 私有 oracle**，**7 = 共享基线（禁用）**。
- 编译（每轮都要，sm_120 本机编不了）：
  ```bash
  GPU=0 scripts/dev/remote_5090.sh ctr 'cd /w/dinfer && CUDA_ARCHS=120 cargo build --release -p dinfer-kaon3 --bins'
  ```
- 参照服务（**干净启动 = 唯一允许的起点**）：
  ```bash
  scripts/dev/remote_5090.sh run 'docker rm -f kaon-oracle'      # 先拆，防上次遗留的截层/改参实例
  scripts/dev/kaon3_oracle_server.sh                              # 无参数 = 全 30 层、无 --enforce-eager、BI=1
  ```
  启动后**必须核三项**，任一不符就重来：① 容器在跑且端口是它自己的（脚本已查）；
  ② `num_hidden_layers` 没出现在 cmd 里（= 全 30 层）；③ 环境里没有 `VLLM_*EAGER*`。
- 重置到干净状态（每轮开始前）：
  ```bash
  scripts/dev/remote_5090.sh run 'docker rm -f kaon-oracle' && scripts/dev/kaon3_oracle_server.sh
  scripts/dev/remote_5090.sh run 'nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader'  # 无我们自己的残留
  ```
  （2026-10-23 实测：上一轮会话残留了一个 `--hf-overrides num_hidden_layers:1` 的截层实例，
  任何读数都会是错的——**先拆再起**是硬步骤，不是建议。）

## 观察方法

| 看什么 | 在哪里 | 命令 |
|---|---|---|
| 端到端判定 | `kaon3-parity.sh` 汇总行 | `PROBE_ENV="CUBLASLT_WORKSPACE_SIZE=1 RMSNORM_MODE=0" bash scripts/verify/kaon3-parity.sh` |
| 单点 A/B | 同一份二进制，只换环境变量 | 见下"开关表" |
| 截层（仅当不饱和） | 截层 oracle + `--layers N` | `scripts/dev/kaon3_oracle_server.sh <N>`；**先验 top-5 不并列**，并列即作废 |
| inductor 生成代码 | 远端正向导出目录 | 需重建：`/w/tmp/ind` 在 2026-10-23 已丢失，按 `kaon3-gelu.md` 第 2 轮方法重导 |

**开关表**（一次只改一个，默认值 = compile 口径）：

| 变量 | 默认 | compile 期望 | 作用点 |
|---|---|---|---|
| `RMSNORM_MODE` | 1（待改回 0） | **0** | norm 数值口径 |
| `NORM_FUSE_QUANT` | 0 | 未定（inductor 融合链） | norm→fp8 量化是否融合 |
| `ROUTER_F32` | 1（on = 精确） | 未定 | router bf16 舍入点 |
| `MOE_COMBINE_F32` | 0 | 未定 | 专家 combine 精度 |
| `GELU_F32` | 1 | 已确认对 compile 逐位 0 | gelu·mul |
| `CUBLASLT_WORKSPACE_SIZE=1` | — | **测量前置条件** | 固定 cuBLASLt algo，否则 router GEMM 选错 algo 制造假发散 |

**每轮读数必须自证**：① `sync` 后容器内 `grep` 到新代码；② A/B 两次输出 md5 不同、日志命中标志；
③ 跑前跑后 `nvidia-smi` 无自己的残留进程。

## 停止条件（可机器判定，`scripts/verify/kaon3-compile.sh`）

1. **口径干净**：oracle 容器在跑、cmd 里无 `num_hidden_layers` 截断、env 无 `enforce_eager`；
   探针以 `RMSNORM_MODE=0` 运行（脚本显式注入）。
2. **端到端**：`flip_pct ≤ 1.5×FLOOR_flip + 3` **且** `mean_common_d ≤ 1.5×FLOOR_d`，FLOOR 由
   `kaon3_oracle_diff oracle_clean_b oracle_clean` 现算（不写死 0.83/0.0276，它们会随参照重抓而变）。
3. **长文本**：`|ΔNLL|(long_zh) ≤ max(3×|FLOOR|, 0.01)`。
4. **单 token**：`bos1` top1 翻转数 = 0。

四条全满足 exit 0。

## 上限

- 最多迭代：12 轮
- 到上限仍未通过：停下，把已排除的假设写进"卡点"，交给用户。中途不停下、不中途问。

## 假设清单（按怀疑度排序，只列 compile 口径下**未验证**的点）

1. **`RMSNORM_MODE=0` 整跑**——交接区点名的"唯一还没验过的已知差异点"（现状读数 16.56% 是在 MODE=1 下得的）。
   先跑它，再看要不要往下走。
2. **层外：`final_norm` + `lm_head` fp8 量化**——vocab=262144 的大形状，inductor 可能把 norm 与量化融成链
   （和 `input_norm→qkv` 同一模式）。分层对拍覆盖不到这里。
3. **`logits` softcap**——`tanh(x/30)·30` 的位置与精度（是否 f32 全链、舍入次数）。
4. **MoE combine**——`MOE_COMBINE_F32` 两种口径哪一个是 compile 的真实行为。
5. **attention `o_proj` 前的量化点**——`unified_attention_with_output` 是 custom op，出口是否又量化一次。
6. **router 预处理 `rmsnorm_scale2`**——fp32 融合链目前"kernel 级差半 ulp"，未逐位。

## 卡点（到上限或放弃时填）

（未开始）

## 最新一次结果

- 日期：2026-10-23
- 第 0 轮（基线，MODE=1 得的历史数）：flip 16.56% / `|Δlp|` 1.1994 / FLOOR 0.83% / 0.0276
- 第 N 轮：

## 进度快照（2026-10-24 凌晨，loop 进行中）

- **第 1 轮**（`RMSNORM_MODE=0`）：flip **16.98%** / |Δlp| **1.1816**（基线 MODE=1 是 16.56% / 1.1994）
  → **compile 口径的 norm 不是主因**，交接区点名的"唯一未验差异点"已排除。
- **第 2 轮**（读 FX 图）：从 oracle 容器 `docker cp` 出 `computation_graph.py`（12451 行，**不用重导 inductor**）。
  读出：norm 返回 bf16 / 量化是 `f32×reciprocal(scale)` → clamp ±448 → fp8 / router 预处理三次 bf16 舍入 /
  router GEMM 走 **BI Triton persistent matmul**（不是 cuBLAS）/ `v_norm` weight=None / q 进 attention 前是 fp8。
- **第 3 轮**（`RMSNORM_MODE=2`：`x·inv` 后先舍 bf16 再乘 w）：照 FX 图 + `ir/ops/layernorm.py:17` 源码。
  第一次因 `norm.cu` 签名漏 `mode` 编译失败**且未 sync**，脚本只看 `^error` 没拦住 → **作废**；
  已加**编译门禁**（匹配 `cargo:warning=.*error:` 与 `nvfb failed`）后重跑。结果见迭代记录。

### 已排除的假设（别重复）

| 假设 | 结论 |
|---|---|
| lm_head 量化 | ❌ `quantized_layers` 424 项里没有 lm_head / embed |
| `fused_add_rms_norm` 残差语义 | ❌ Gemma4 用显式 bf16 加法（`gemma4.py:702-731`） |
| q 未做 fp8 量化 | ❌ 我们已做（`layer.rs:753`），A/B 证实必需 |
| `oracle_clean` 口径不对 | ❌ 与当前 compile 服务重抓逐项 0 差异 |
| compile 口径 norm（mode 0） | ❌ 第 1 轮：与 eager 口径无实质差别 |

### 尚未核对（按嫌疑度）

1. `bmm_fp8` ×120（fp8 GEMM 的 scale 应用顺序与 out_dtype）
2. `moe_forward` ×30（combine 口径 / per_expert_scale 折进位置）
3. router 的 BI persistent matmul（我们用 cuBLAS → 需 `CUBLASLT_WORKSPACE_SIZE=1` 才稳）
4. `unified_kv_cache_update` 的 kv fp8 量化点
