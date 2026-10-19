# Loop: Qwen3.5 全注意力算子（D=256 / GROUP 6 / partial RoPE / 输出门）

> 状态：进行中（2026-10-19，里程碑 3）。方法论 `docs/model-onboarding.md` §3。前置：`docs/loops/qwen35-gdn.md`（GDN 已过三卡）。

## 环境

- 目标卡：**只用本地**——本机 3090（sm86）与 4090 GPU 7（sm89，`scripts/dev/remote4090.sh`，`CUDA_VISIBLE_DEVICES=7`）。
  **不执行任何 `remote_b300.sh`**（用户 2026-10-19 指示）；B300 留给里程碑 5 真权重。
- 参考源码（口径唯一来源）：
  - `vllm/model_executor/models/qwen3_next.py::Qwen3NextAttention`（q_proj = H·D·2，per-head `[q|gate]`；
    `attn_output * sigmoid(gate)` 在 attention **之后**、`o_proj` **之前**）。
  - `vllm/model_executor/layers/fused_qk_norm_rope.py::fused_qk_rmsnorm_rope_gate`（partial RoPE：只旋前 `rotary_dim`，
    其余维只做 RMSNorm；pair 为 `(i, i + rotary_dim/2)` 的 neox；norm 权重 `w + norm_beta`，Qwen3.5 的 `q_norm/k_norm`
    是 **zero-centered**，`norm_beta = 1.0`）。
  - HF `transformers/models/qwen3_5/modeling_qwen3_5.py`（同上语义的朴素实现）+ `Qwen3_5RMSNorm`（`(1 + w)`）。
- 对拍夹具：`crates/kernels/tests/ops_parity.rs` + `scripts/verify/ops_parity_ref.py`（纯 torch），
  Qwen3.5 真实形状 `H=24 / Hkv=4 / D=256 / rope_dim=64`；旧 Qwen3 行（`H=16/Hkv=8/D=128`）必须保持不变。

## 拓扑改动（都按"扩模板 / 加参数"，不复制 kernel）

| 算子 | 改动 |
|---|---|
| `gqa.cu::gqa_fa_v5_kernel` | `D`（`V4_D`）模板化：实例 128 / 256；tile 128→64 key、256→32 key（`V4_LDS = D+8`，smem = 2·NBUF·TILE·LDS·2 B：D=256 单级 33 KB，双级 68 KB 走 `cudaFuncSetAttribute`）；host 在 `D ∈ {128,256}` 时走 v5，其余仍走通用慢路径 |
| `gqa.cu` partial merge | `gqa_decode_merge_v2_kernel` 的 `PF_D` 改运行时 `head_dim`（blockDim = head_dim），D=256 复用 |
| `gqa_decode_v3.cu` | `D` 模板化 + GROUP 加 **6**；`V3_LPK = D/8`（D=256 时 32 lane/key，即每 warp 一个 key、每 tile 4 key）；跨 lane 归约按 LPK 选 `sum16` / `sum32`；收尾合并 `for (d = tid; d < D; d += V3_THREADS)`（原 `tid < V3_D` 在 D=256 会只写一半） |
| `gqa_decode` host | D=256 **只走 v3**（v2 的 `D2_MAXG=2` / `PF_D` 均不适用）；D=128 保持 `GQA_DECODE_V3` 开关，`=0` 仍走 v2 |
| `qkv_post.cu` | `D` 模板化（128/256）+ 新参数 `rope_dim`（≤ D，必须 64 的倍数）与 `norm_beta`（0 = `w`，1 = `1+w`）+ `gated`（行布局 `[q | gate | k | v]`，gate 段原样跳过）。**旧入口 `qkv_post_bf16` 转发到新实现（rope_dim=D, beta=0, gated=false）**，语义不变 |
| `activation.cu` | 新增 `mul_sigmoid_bf16(out, gate, n)`：`out *= sigmoid(gate)`（vLLM 的 attention 后置输出门） |

**行布局约定**（写进 `kernels.h` 注释）：`q_proj` 输出在权重侧是 per-head 交错的 `[H][q(D), gate(D)]`；
我们要求**加载期拆成两段**（q 取每头前 D 行、gate 取后 D 行），kernel 侧看到的是 `[q(H·D) | gate(H·D) | k | v]`。
拆行在 `models/qwen35/weights.rs` 做（里程碑 4），不进 kernel。

## 观察方法（机器可判定）

1. `scripts/verify/ops_parity.sh` → 打印逐算子表，**exit 0 = 通过**；这是唯一判定入口。
2. 基线：改前 **75/75**（`ref_tests.rs` 24 + `ops_parity.rs` 51）。改后要求：
   - `ops_parity.sh` 全过且旧行误差不变（行数 = 75 + 新增）；
   - `GQA_DECODE_V3=0 scripts/verify/ops_parity.sh` 同样全过（D=128 回退 v2 的路径仍在跑）。
3. 两卡：本机 3090 先过，再 4090 GPU 7（`scripts/dev/remote4090.sh` 推脚本过去跑同一个 `ops_parity.sh`）。
4. 新 case 清单（每个都进表）：
   - `qkv_post`：`(H=24,Hkv=4,D=256,rope=64,beta=1,gated)`；`(H=16,Hkv=8,D=128,rope=64,beta=1)`（partial rope 独立验证）；
   - `gqa_prefill` / `gqa_paged` / `gqa_decode`：`H=24,Hkv=4,D=256`（GROUP 6），长度含单请求 1/37/1514 与双请求；
   - `mul_sigmoid`：bf16 逐元素。
5. 容差沿用现有口径：attention 每行 cos > 0.999 且 max abs < 2e-2；qkv_post 2e-2；逐元素 1e-2。

## 停止条件

- `ops_parity.sh` exit 0（含新 case）**且** `GQA_DECODE_V3=0 ops_parity.sh` exit 0，**且** `cargo build --workspace --features qwen3` exit 0；
- 3090 与 4090 GPU 7 两卡都 exit 0。

## 迭代上限

- 同一 case 同一方向偏差最多 3 轮；总轮数上限 8；到上限收功并记录现象与已排除假设。

## 轮次记录

| # | 日期 | 改动 | 结果 |
|---|---|---|---|
| 1 | 2026-10-19 | 建规格：钉死 gate 语义（attention 后置）、zero-centered norm、partial rope 的 pair 口径，列拓扑改动表 | 规格完成 |
| 2 | 2026-10-19 | v5 模板化 D（128/256；D=256 tile 32 key 保 smem ~34 KB）、merge_v2 运行时 head_dim、v3 模板化 D + GROUP 6、qkv_post 扩展入口（partial RoPE / norm_beta / gate 行布局）、`mul_sigmoid_bf16`、新对拍 case（prefill/paged/decode/mixed D=256 × GROUP 6、qkv_post_ext × 3、mul_sigmoid × 3） | 首次跑：18/21 新行过；v5 的 smem 行跨度误写成 `TILE+8`（应为 `D+8`）→ gqa 全红，改后全过 |
| 3 | 2026-10-19 | 定位间歇 NaN：同进程先跑 D=256 后，D=128 混批 `h4_hk2` 用例出随机 NaN（2–16 个元素，位置每次不同） | 根因：v3 对越界 key **不发** cp.async，smem 残留 inf/NaN 与 `p=0` 相乘得 NaN。改为 `cp16z` 零填充后 5 轮连跑全 0 NaN。**这是既有隐患**，D=256 实例化后 smem 布局变化才稳定暴露 |
| 4 | 2026-10-19 | 3090 全量 `ops_parity.sh`（v3 开 / 关各一次） | 见下（待填） |

## 环境备注（踩过的坑）

- **同一次 `ops_parity.sh` 里 4090 的 `wait_gpu` 会等 GPU 0 上别人的 `dinfer serve`**（`test 固定 `Device::new(0)`）。
  远端跑必须自己包一层 `CUDA_VISIBLE_DEVICES=7`，并设 `GPU_WAIT_S=5` 避免长时间等待（已写进 `/tmp/rp.sh` 的做法）。
- **`/data/yaochen/infer/env.sh` 里有 `$RUSTFLAGS` 未定义**：远端脚本不能 `set -u`，否则 source 就退出。
- 4090 没 torch：远端只做 dump，`.npy` 拉回本机（387 MB，~5 分钟）用 `ops_parity_ref.py` 比对。
- 同一 workspace 的另一个子代理在改 `kda_chunk.cu`（编译不过时挡住全局构建）：本里程碑的验收在
  `rsync --exclude target` 出来的隔离副本里跑，避免相互影响；commit 只 `git add` 自己的文件。
