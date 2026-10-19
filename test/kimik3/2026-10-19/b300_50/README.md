# B300-50（8 × B300 SXM6，sm_103，CUDA 13.1）K3 kernel 级验证 — 2026-10-19

只做 kernel 级（不做模型层）。环境：`B300_TAG=50 scripts/dev/remote_b300.sh`，GPU 0，
`CUDA_ARCHS=103`，`cargo build --release -p dinfer-kernels --features ref-tests --tests`（1m08s）。
参考 .npy 随仓库带过去（远端无 torch），`KDA_SKIP_GEN=1 MXFP4_SKIP_GEN=1`。

## ① 对拍 + 计时（`scripts/dev/k3_b300_run.sh`，exit 0）

KDA：11 test 全过（`kda_rc=0`），误差与 3090/4090 同量级（chunk out max_err 2.4e-4 / cos 0.9999951，
state max_err 2.05e-3 / 相对 2.68e-3 / cos 0.9999985）。MXFP4：2 test 全过（`moe_rc=0`），
解包 3 case 仍逐位一致（max_err 0），grouped GEMM rel 4.98e-3 / 5.88e-3 / 6.06e-3，cos 0.9999966–0.9999967。

### KDA chunk prefill（T=1024，H=96，K=128，BT=64；`kda_parity --test kda_chunk_bench`）

| 卡 | 顺序递归 | chunk 全链 | 加速 |
|---|---|---|---|
| 3090（sm86） | 76.94 ms（同日 release 复测） | 10.30 ms（同日复测） | 7.47× |
| 4090（sm89） | 29.2 ms | 5.66 ms | 5.15× |
| **B300（sm_103）** | **36.33 ms** | **4.87–4.90 ms** | **7.41–7.46×** |

B300 的顺序递归比 4090 慢 24%（该 kernel 是逐 (seq,head) 块的标量 FMA + shared 复用，
不吃 sm_103 的额外算力），chunk 全链比 4090 快 14%。

### MXFP4 grouped GEMM（真形状，`mxfp4_parity --test mxfp4_gemm_all_cases`）

`gemm_w1_like`（E=3 N=3072 K=3584，pairs=109，384 tiles）/ `gemm_w2_like`（E=2 N=3584 K=3072，pairs=52，280 tiles）
两卡数值一致。

**计时（2026-10-19 新增 `mxfp4_gemm_bench`：E=8，每专家 32 token = 256 pairs，release，5 次均值）**：

| 卡 | w1/w3（768 tiles） | w2（896 tiles） | 解包 bf16（8 个 w1/w3 专家，一次） | bf16 cuBLASLt ×8（参照） |
|---|---|---|---|---|
| 3090（sm86） | 5.50 ms | 5.20 ms | 0.7 ms | 0.30 ms |
| **B300（sm_103）** | **1.67 ms** | **1.61 ms** | **0.4 ms** | **0.06 ms** |

v1 是解包 + 标量 FMA，没有性能门槛。这里的数字是给 sm100 原生 FP4 路径（`tcgen05` + 权重保持 packed）
的对照基线：解包成 bf16 再走 tensor core 的单层 GEMM 只要 0.06 ms，比 v1 标量路径快 **28×**——
这正是 v2 要补的差距（不算每步解包的 0.4 ms；真权重不能每步解包，见 `docs/loops/k3-moe.md` TODO）。

## 日志

- `parity.log`：B300 上 `k3_b300_run.sh`（11 个 KDA + 2 个 MXFP4 test，含 `kda_chunk_bench`）的完整输出。
- `bench.log`：B300 上 `k3_b300_bench.sh`（`kda_chunk_bench` + `mxfp4_gemm_bench`，release 二进制）。

## ② 真权重 MXFP4 解包核对（`scripts/verify/k3_realcheck.sh`，exit 0）

L1 / expert 0 的 w1 / w2 / w3，各 11,010,048 个元素，numpy（`python3 -I`，纯 numpy 解 safetensors header）
参考 vs `mxfp4.cu` 解包：**3/3 逐位一致，`|got-ref|max 0`**（0.65–0.69 s/张量，含 D2H）。
B300 容器没有 numpy / torch，用 `/data/yaochen/infer/pyenv-numpy`（python3 -m venv + pip 装 numpy 2.2.6）。

scale 统计（36 张量 = L{1,12,45,91} × e{0,447,895} × w{1,2,3}，12,386,304 个 e8m0）：u8 ∈ [112, 122]，
**u8=0 / 255 均未出现**；众数 121（`2^-6`，84.2%），121±1 覆盖 99.0%；值域 `2^-15 .. 2^-5`。
解包后 |w|max 0.125，无 NaN/Inf，零值占比 11.5–12.0%。详见 `docs/loops/k3-moe.md`「真权重解包核对」。

- `realcheck.log`：Rust 侧逐位对拍输出；`scale_stats.txt`：scale 直方图。
