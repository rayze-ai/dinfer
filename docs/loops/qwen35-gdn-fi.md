# Qwen3.5 r11c：GDN prefill 接入 FlashInfer（替代自写 mma.sync 三件套）

> 状态：**开工（2026-10-21）**。触发：用户问"vLLM 的实现有没有参考价值"→ 查 vLLM B300 实际路径，发现它**不是**走 fla/Triton
> （我们 r9/r10 一直对照的那个），而是 `Using FlashInfer GDN prefill kernel`。实测差距 **3.8–4.9×**（下），远超任何 kernel 微调能拿到的量。
> 用户 2026-10-21 决定："先量 FlashInfer GDN 的实际耗时，快很多就直接搞"。

## 实测（2026-10-21，B300-50 GPU 4，`probe_flashinfer_gdn`）

容器 `docker.io/vllm/vllm-openai:nightly`（torch 2.13 cu130）+ `--net host`（**必须**：本机没装 CNI bridge 插件，默认网络起不来），
`scripts/dev/fi_prof.py` 走 vLLM `fi_chunk_gated_delta_rule` 的真实调用口径（q/k/v bf16、`g=exp(g)` f32、beta f32、
initial_state f32、cu_seqlens int64），torch.profiler 20 次均值，形状 HV=48 / K=V=128：

| 形状 | FlashInfer total | 主 kernel | dinfer r10 全链 | 倍数 |
|---|---|---|---|---|
| 1×1024 | **49 µs** | 42（84%） | 208 | **4.2×** |
| 8×1024 | **156 µs** | 142（91%） | 770 | **4.9×** |
| 1×8192 | **253 µs** | 242（96%） | 969 | **3.8×** |

**整个 GDN 每层只有 1 个 kernel**（`kernel_cutlass_..._GatedDeltaNet_Chunked...`），其余 4 个是 2–5 µs 的
copy/fill/exp（都是 `torch.exp(gg)` / contiguous / `FillFunctor` 这类**外部准备**，可以省掉）。
我们是 wy + ho + conv + prep 四段、几十次 launch。

**收益折算（engine-gap 9e：48 层 GDN = 52 ms / 6 K 的 277 ms）**：按 253 µs × 48 = 12 ms → **6 K TTFT 可省 ~40 ms**
（281.7 → ~242，vLLM 260.6），这一项就能把 6 K 从落后 8% 变成领先 7%。

## 它的算法结构（读 `flashinfer/gdn_kernels/blackwell/gated_delta_net_chunked.py`，4754 行，BSD-3）

每个 chunk（C 个 token）固定 **7 个 tcgen05 GEMM**，state 全程留在 **TMEM**、不进显存：
1. `cumsumlog` / `cumprod` / `T_pairwise[i,j]=cumprod[i]/cumprod[j]` 全在寄存器（128 regs/thread），**不落显存**；
2. `W_kk = K·Kᵀ`（下三角）、`W_qk = Q·Kᵀ`；
3. `KS = K·S_prev`、`QS = Q·S_prev`（跨 chunk 输出）；
4. `A_inv = (I+M_kk)^{-1}`（层次分块逆）→ `NV = A_inv·V`；
5. `O_intra = W_qkv·NV`；`dS = Kᵀ·delta` 更新 state。

对照 r9/r10 的教训（`qwen35-gdn-r9.md` 收口分析）：我们 ho 的瓶颈正是"每 block 串行走 128 个 chunk 的延迟链"，
而 FlashInfer 把 state 放在 TMEM、在一个 kernel 里迭代完——这不是"调 tile 能追"的差距，是**数据放哪**的差距。

## 落地路径（必须先验证，不许猜）

FlashInfer 这套是 **CuTeDSL**（Python DSL → 运行时编译）。dinfer 是 Rust，三条路：

| 路径 | 做法 | 代价 | 风险 |
|---|---|---|---|
| **A. AOT cubin + Rust 加载**（首选） | `cute.compile_to(ArtifactType.PreCompiledMlir)` / `KeepCUBIN dump_path` 出 cubin → Rust 用 cuda driver `cuModuleLoadData` + `cuLaunchKernel` | 需确认产物是独立 cubin、参数 ABI 稳定 | 若产物依赖 TVM FFI 运行时（`aot_config` 里有 `-ltvm_ffi` 迹象）则不成立 |
| B. 链接 FlashInfer 的 C++ AOT 库 | 走 `flashinfer` 的 C++ 头 + `.so` | 依赖上游 ABI，升级会破 | 慢 |
| C. 照抄算法自己写 tcgen05（`*_sm100.cu`） | 按上面 7 个 GEMM 的结构重写 | 2–3 天 + 调试 TMEM 描述符 | 大，但可控、无外部依赖 |

**先做 A 的决定性实验**：在容器里把 GDN kernel AOT 成 cubin 落盘，检查 (a) 是不是单个 `.cubin`，(b) `cuobjdump -res-usage` 看寄存器/smem，(c) 能否用最小 CUDA driver 程序 launch 起来、结果与 PyTorch 版逐位一致。A 不成立就转 C。

## 环境

- B300-50 GPU **4**（`CUDA_VISIBLE_DEVICES=4`），容器一律 `sudo -n nerdctl -n k8s.io run --rm --net host --gpus device=4 -v /data/yaochen/infer/bootstrap:/b`。
- **容器默认网络起不来**（无 CNI bridge），`--net host` 必带。
- 权重 `/data/models/Qwen/Qwen3.8-27B`；`text_config` 里 `linear_key_head_dim=linear_value_head_dim=128`、`linear_num_key_heads=16`、`linear_num_value_heads=48`。
- Rust 侧开发/parity 在 3090 本机（只做编译与对拍，不判性能）。

## 文件归属

新增 `crates/kernels/cuda/gdn_fi_sm100.cu`（或 `crates/kernels/src/gdn_fi.rs` + cubin 资源）、`crates/kernels/src/arch.rs` 的选路、`models/qwen35/src/gdn.rs` 的调用点。
**旧路径全部保留**，由 `ArchProfile` + 环境变量选（Sm100 且 head_dim=128 且 cubin 可用 → FlashInfer；否则 r10 自写）。

## 停止条件（机器可判定）

1. `gdn_parity` 14/14（3090 + B300），与 r10 默认路径误差同量级（out max_err ≈2.4e-4 / state rel ≈6.7e-3，或更好）；
2. B300 `gdn_chunk_bench` 同形状：**8×1024 ≤ 200 µs、1×8192 ≤ 300 µs**（FlashInfer 156 / 253 的 1.2× 内）；
3. 单发 6 K TTFT **≤ 245 ms**（现 281.7；GDN 省 ~40 ms 的兑现率 ≥80%，即至少 −32 ms）；
4. 4090 / B300 四组闭环不退化 > 3%；
5. 迭代上限：A 路线 3 轮（含 AOT 可行性），转 C 后 5 轮；卡住 2 轮回报主会话。

## 轮次

| 轮 | 改动 | 结果 | 结论 |
|---|---|---|---|
| 0 | 量 FlashInfer 实际耗时（`fi_prof.py`，GPU 4） | 49 / 156 / 253 µs vs dinfer 208 / 770 / 969 | 3.8–4.9×，值得接 |
