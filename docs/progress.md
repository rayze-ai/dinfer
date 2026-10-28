# 进度

<!-- ===== 当前状态（交接区）：每次收功整体重写，保持 ≤ 20 行 ===== -->
## 当前状态

> 更新 2026-10-28 +0800（Kaon-V3 perf 节）。**本区按模型分节，每节只写该模型自己的交接内容；不要覆盖别人的分节。**


### Kaon-V3 parity（2026-10-24 第三次收工，loop `kaon3-layer1`）

**一句话**：逐段定位做完，**差异收敛到 L0 的 fp8 qkv GEMM 形状**；norm/FFN 已证明逐位正确。

**已逐位证明正确的部分（可直接复用，别重查）**

| 段 | 判据 | 结果 |
|---|---|---|
| `input_norm` / `pre_ff_norm` / `pre_ff2_norm` / `ff1` / `post_ff_norm` | `seg_l0.py` 按 FX 图逐步复现 | **0/67584 全同** |
| norm 实现（`RMSNORM_MODE=2`） | 与运行时 `vllm.ir.ops.rms_norm` 比 | **0/67584** |
| dense NVFP4 GEMM | 自洽合成判据，三条路径互证 | **ndiff=0** |
| qkv requantize（`load_group`） | 与 vLLM 运行时合并权重逐字节比 | **23068672 字节全同** |

**根因（第 7 轮，证据确凿）**
- vLLM 的 `qkv_proj` 是**单个** `[2816, 8192]` GEMM（FX 图 `unsqueeze_1: f8e4m3fn[1,2816,8192]`）；
- 我们跑的是 q/k/v 三次独立 GEMM；
- 合并一次 N=8192 实测 `ndiff=13/196608`，分开三次差 **13 万个元素** → 形状决定 fp8 的 tile 与 split-K，累加序不同。

**卡点（下次接手第一件事）**
**第 0 层 `k_eq_v=True` 时，参照的 `qkv_proj` 是 N=8192 还是 N=6144？**
两处源码给出相反暗示（`gemma4.py:413` 说结构不变、K 填 K/V 两槽；`gemma4.py:1513` 注释说
k_eq_v 层用分开的 q/k 不打包）。已抓到 vLLM 运行时真实 `[2816,8192]` 权重（`scripts/dev/capqkv.py`），
用第 8 轮已验证的 requantize 公式分别按「q,k,v」与「q,k,k」复现，看哪个逐字节相等 —— **一步定死**。

**代码状态**：`layer.rs` 里未验证的合并改动**已回退**，工作区干净。
端到端现状：分开三次 `flip 6.12% / |Δlp| 1.0062`（2 个中文用例）。

**环境**：5090 `/hf_data/yaochen`，`kaon-oracle-1`（GPU1 :28101，compile，1 层）在跑；
GPU2 可抓取。`put` 有 bug（目标路径不生效），改用 `rsync -az` 直传。

### Kaon-V3（5090，阶段 1–2 完成，阶段 3 接近打通）

#### 性能优化（2026-10-27，loop `kaon3-fp8gemm`）——两配置洞已落地 + 三优化点待做
**已落地（配置，非代码，都进 kserve_full.sh）**：吞吐 357→594（vLLM 的 63%）、TTFT 3357→250ms（13×）。
- ① `max_num_seqs` 16→64：默认 16 太小（vLLM 256），48 请求涌入堆积 TTFT 到 8s。改 64 后 TTFT 降 13×、吞吐 +51%。**生产 KV 池要配套**（max-model-len 降或 num-blocks 够，16384+8192 会 OOM）。
- ② `ROUTER_GEMM_BI=0`：router 的 batch-invariant persistent matmul 单 block 单 SM 串行 K=2816（为逐位对拍，生产不需要），0.25→0.03ms。单层 decode 降 32%。符合生产口径（[[dinfer-kaon3-prod-config]] 生产不开 BI）。

#### ★ 2026-10-28 晚（loop `kaon3-decode-opt` 第二轮，用户 step A/B/C 流程）——Step A 定论 + B1 一刀 -19%
**旧结论作废**：上午"瓶颈 100% 在 host / MoE 不在关键路径"全是旧二进制假象（改动没编进，D42/D45）。**真相：decode 100% 是图内 GPU 时间**（host 链 tokenize 1.1ms + sched/publish/htod 全 <50us；decode 步 = 0.36ms graph 提交 + 16.6ms 等 GPU，sm 96%）。
**Step A 对齐基线（D51，5 轮 × c1–16 中位数，两端非 eager / 16384 / 4096 / seqs64 / 1560 tok）**：c1 TPOT 17.07 vs vLLM 4.69（3.6×）、c8 19.71 vs 6.24、吞吐 29–34%；TTFT c≥4 已反超 vLLM，c1 慢 18ms（prefill host launch 16ms）。mem util 36–47% → 不是带宽墙，是小 kernel / 并行度。
**nsys 图内清单（D54，`scripts/dev/kaon3_nsys_decode.sh` + `nsys_kern.py`）**：每步 **1298 kernel**；逐头 norm 2.29ms + rope 0.70ms **按 max_tokens=4160 行空跑**；dense mega 2.61；D=512 全注意力慢路径 1.85（5 层 × 369us，16 block）；h=2816 单行 norm 1.23（181 次 × 6.8us）；3 个 grouped GEMM 仅 1.29（D53 skip 实证）；launch 间隙 1.89。
**B1（commit B1）**：逐头 norm / rope 行数改实际 tokens → **c1 TPOT 16.16→13.02（-19.4%）**，输出一致。**B2**：Sm120 rmsnorm layout Block256 → 13.03→12.58（-3.4%）。**B3（D58/D59）**：decode 专用 split-K attention（v3 加 window + 新 D=512 kernel）→ **12.58→9.79（-22%）**、c8 16.01→13.63；GPU 对拍 `gqa_decode_win_parity` 6 场景超差 0 → 保留。三刀累计 c1 **17.07→9.79（-43%）**、吞吐 57→97.7；vs vLLM 4.69 还差 2.1×。CSV round b1/b2/b3 已入 `test/kaon3/e2e-rounds.csv`。
**B4（定稿）**：gather_rows 多 block 16B 拷贝 → c1 9.77→9.64（-1.4%），逐位一致，CSV round b4。**B5（定稿，c1 TPOT 9.64→7.47 -22.5%，c8 13.45→10.94，CSV round b5）**：decode dense MLP 不走 persistent mega（nsys 86.9 us/层 vs 带宽下限 10 us，D60 roofline），改 3 个带宽型 fp8 GEMV kernel（`crates/kernels/cuda/mlp_fp8_decode.cu`，`DENSE_DECODE_OLD=1` 回退）；GPU 对拍 `mlp_fp8_decode_parity` 5 场景 fp8 码差 0。五刀累计 c1 17.07→7.47（-56%），vs vLLM 4.69 差 1.59×。**B6（D62/D64，回滚为默认关 `FP8_GEMV=1` 才开）**：qkv/o_proj fp8 走 gemv，c1 -7% 但 m=8 时 c8 +5%（smem 45 KB 占用低）、warp 调多后 c1 也慢 1.8 ms → 教训：小 M 下 block 数（并行度）比 A 复用重要；B6c 门槛改 smem ≤ 24 KB（m ≤ 4）+ pick_warps 上限 4，待验。**B7（D63/D64，放弃接入，`MOE_DECODE_GEMV` 默认关）**：MoE 单专家 N=704 的"warp 一行 GEMV"每 warp 只流 2.8 KB，比 mma grouped GEMM 慢 1.8 ms；GEMV 结构只对单个大矩阵有效。**B8（commit 6f14bcb，待验）**：grouped GEMM 一维 tile 表（`tile_off` 由 `moe_bucket` scan 产出，grid.y = R/BM + E 代替 R/BM × E 满格），prefill down GEMM 5.8M 空 block → 0，预期 TTFT -30%+，decode 也少 44×1040×127 空 block；逐位不变（同 tile 同序），门禁 `moe_pipe_parity`。**B10（commit 2aac32e，待验，默认开）**：MoE 路由前处理 7 launch → 1（`route_bucket_small_kernel`，tokens ≤ 64；`MOE_ROUTE_FUSED=0` 回退），预期 -0.6 ms/步。**b8c 定稿（CSV round b8）= B6c（fp8 gemv m≤4）+ B8（grouped GEMM tile 表）+ B10（路由 7 launch→1）+ B11（gather 融 quant / combine 替 scatter）**：c1 TPOT 7.49 → **6.64（-11%）**，**TTFT c1 38→30、c8 580→493**；c8 TPOT 11.1→11.6（+0.5，split 链 `/w/b8csplit.sh` 跑中：FP8_GEMV / MOE_ROUTE_FUSED=0 各一组）。六刀累计 c1 **17.07 → 6.64（-61%）**，vs vLLM 4.69 差 1.42×。**B12（commit f7f9cee，已 put 未编）**：kv_fp8 q/k/v 往返 + cache_write 融 1 launch（`KV_FP8_FUSED=0` 回退），链 `/w/b8d.sh`（等 split 结束后起）。**B13（commit dec4683，已 put）**：norm+残差加融合（`rmsnorm_add_bf16`，-2 launch/层，`NORM_ADD_FUSED=0` 回退）。**D65 split**：B6c 与 B10 c1/c8 都赚；c8 +0.5 归因剩 B8/B11（b8d 链后再拆 `MOE_COMBINE_OLD=1`）。**launch 间隙模型修正**：graph 内 ~0.7 us/launch（B10 7→1 实省 0.16 ms），剩 ~900 launch ≈ 0.6 ms，不再是最大头。**下一大刀 B15**：KV cache 真存 e4m3（现只往返、cache 仍 bf16；vLLM 存 fp8）→ decode KV 读带宽减半，attention 1.2 → ~0.95 ms，cache 容量翻倍；设计见 loop B15 节。 **B15 定稿**：默认关（`KV_FP8_STORE=1` 开，CSV round b15）—c1 持平、c8 TPOT -3.6% 但 c8 TTFT 2.8×（混批 prefill 被强制走慢路径）；**B15b**（fa_v5 PAGED 读 e4m3）代码落地随 b16 链验，过了再默认开。**B16（链 `/w/b16.sh` 跑中，门禁 `moe_pipe4_matches_baseline` bad=0 已过）**：MoE grouped GEMM 4 级 cp.async 环形流水（`MOE_GEMM_STAGES` 默认 4，=2 回退；BK=128 最多 4 级），预期三 GEMM 50 → ~20 us/层 ≈ -0.9 ms/步。
**工具坑（都已修进脚本）**：nsys 需 `--cuda-flush-interval` 且不能 INT nsys 本体；`remote_5090.sh put` 现 `--no-times`（否则 cargo 不重编）；排队等待别用 `pgrep -f`（自匹配死锁）；`remote run` 退出码不透传，判 DONE 要拉文件本地 grep；容器 `kaon3-full` 已重建为常驻 `sleep infinity`。
**CSV/dashboard**：e2e 行改走 `test/kaon3/e2e-rounds.csv`（`scripts/dev/e2e_round.sh` 一行入库，bench_trend 重生成时合并，不再丢）。

- 目标（用户 2026-10-21）：5090 **单卡**适配，性能 ≥ 基线容器 1.2×；**不反量化权重**。
- **机器**：`host-172-16-20-59`（8×RTX 5090 32GB，sm_120），跳板 `jump.aicloud.paigod.work:2222`。
  **token 一次性**（用户现给 → `~/.ssh/jms_5090_{user,pass}`，第一条命令必须直接
  `scripts/dev/remote_5090.sh up`，不能先探活）。**GPU 7 跑基线（`:28067`），绝不占用**；0–6 可用。
  工作目录 **`/hf_data/yaochen`**，远端仓库 `/hf_data/yaochen/dinfer`（**rsync 树、无 `.git`**，
  一律本地写 → `put` 单文件 → 远端编）。代理 `172.17.0.1:1081`。
- **容器命令（每个新容器都要重做，`/tmp/stub` 不持久）**：`--entrypoint bash` 必带；
  先
  `mkdir -p /tmp/stub && ln -sf /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2 /tmp/stub/libnccl.so && ln -sf /usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib/libnvrtc.so.13 /tmp/stub/libnvrtc.so`
  再 `export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/{nccl,cu13}/lib:$LD_LIBRARY_PATH`。
  **不加这些 = 链接失败或运行期 127，而且会静默跑到旧二进制**（本轮踩过两次）。
- **模型**：`/hf_data/hub/models--kaon-release--Kaon-V3-nvpf4-mixed/snapshots/e268bef1…/`
  （**HF cache 布局**，config 在 `snapshots/*/` 下）。Gemma4 MoE，30 层 / 128 专家 top-8 /
  滑窗 1024 + 5 层全注意力（下标 5/11/17/23/29）。容器要 `-v /hf_data/hub:/hf_data/hub:ro` 才能读到。
- **基线**：c1 TPOT 8.34 ms / 108 tok/s；c8 758；c32 2261；6K TTFT 244.6 ms（1.2× 线 203.8）。
- **已完成**：阶段 1 NVFP4 探针 502 TF/s；阶段 2 FP8 292 TF/s + 加载器。
  **阶段 3**：`models/kaon3` 完整可编译（config / weights / layer / model / probe）；
  gqa 五 kernel 滑窗；无权重 RMSNorm；gelu_tanh；`scale/mul_row/softcap` kernel；
  FP8/NVFP4 GEMM 移植（`Gemm::forward_fp8` / `forward_nvfp4`，对拍 bad=0）；
  `per_expert_scale` 折进专家 down 的全局 scale。**端到端探针前 5 层（滑窗层）已跑过**。
- **2026-10-22 白天**：30 层端到端已跑通（`MAX_D` 提到 512 走慢路径）；修了 proportional RoPE 配对
  （整头 neox，配对 (i, i+256)，`rope_span_bf16`）；**但 logits 与基线不符**（top-5 无交集，见 loop 文档末节）。
  `scripts/dev/remote_5090.sh ctr '<cmd>'` 一条命令进容器编 / 跑。
- **下一步（按序）**：① loop `kaon3-parity`：oracle = 基线服务 `prompt_logprobs`（curl 存文件），Rust 探针比对，不用 Python 参考
  → 定位发散（T=1 隔离 attention / 逐项语义核对 / Rust kernel 测试）；② NVFP4 对拍改写成 Rust GPU 测试并重跑；
  ③ 高效 D=512 attention（慢路径当参照）；④ decode 侧滑窗（`gqa_decode_v3.cu`）；⑤ 阶段 4 服务。
- **性能缺口**：NVFP4 移植 4096³ 260 TF/s（探针 509）；**512×2816×2816 只有 141（探针 330）**，
  该形状是 Kaon 真实输入，根因是并行度（704 block / 148 SM），需改 tile 形状或 split-K。
  根因已定位：block 内现场量化重复 N/BN 次（已改预量化），剩余是两路动态 smem scale 读的寄存器压力（48 reg）。
- **已知阻塞**：`--features no-nvcc` 在本分支已坏（stub 签名 24 处不符），本地 CPU 编不了。

### Qwen3.5-27B（B300，暂停）

- 目标（用户 2026-10-20）：全面超 vLLM 1.2×。四组闭环 1.10/1.10/1.05/1.04×。
- 用户 2026-10-21 决定：不推倒重来，GEMM 封顶不动，attn 保 r11a v6，只换 GDN 一段。
- 两条线（r11d GDN tcgen05 / r12a attn tcgen05）状态与下一步见 **`docs/loops/qwen35-todo.md`**（14 条汇总）。
- r11d 基线：1×1024 **208** / 8×1024 **769** / 32×128 421 / 1×8192 **967** µs。
- **B300 卡位**：0 常驻 `:8012` / 1 vLLM `:8013` / 2–3 主会话 / 4 r11d / 5 r12a / 6–7 留。
  **硬约束：`B300_TAG=50` 必带，`.225` 禁占**；token 过期后向用户要新的。
- 子代理远端：`/data/yaochen/infer/dinfer_r11d`（GPU 4）、`dinfer_tc05`（GPU 5），都不 push。

### 其他

- K3 挂起（TODO 五条在 `docs/loops/k3-serve.md` 末行）；Qwen3-0.6B Q1–Q5 完成；GLM P0–P6 通过、真权重挂起。
- 页面 `test/index.html`（perf / engine / probe 三页签）。
- **知识库欠账**：`docs/knowledge/perf-method.md` 126 行（上限 120），且章节编号有两组重复的
  7/8/9（2026-10-20 与 2026-10-21 各写了一遍）。CLAUDE.md 引用了「§7」，重排编号会打断指针，
  下次整理时先确认所有 `§N` 引用再合并。

## 2026-10-21（Kaon-V3 / 5090）

- **环境**：新机器 8×RTX 5090（sm_120）。JumpServer token **一次性**，用过即废；第一条命令必须直接建
  ControlMaster（`scripts/dev/remote_5090.sh up`），不能先探活。代理是 **`172.17.0.1:1081`**。
  宿主机无 nvcc/rust，全部在基线容器内编译（`--entrypoint bash` 必带）。工作目录 `/hf_data/yaochen`。
- **模型勘察**：Gemma4 MoE 多模态，30 层 / 128 专家 top-8 / 滑窗 1024 + 5 层全注意力 /
  vocab 262144 tied；`MIXED_PRECISION` 量化（专家 NVFP4、其余 FP8、KV FP8），16 GB。
  读 vLLM `gemma4.py` 1708 行核实语义，写进 `docs/loops/kaon-v3.md`。
- **基线**：vLLM 0.28.0-pegaflow TP=1 `:28067`。c1 TPOT 8.34 ms / 108 tok/s；c8 758；c32 2261；
  6K TTFT **244.6 ms**。发现裸 prompt 输出退化，必须走 chat 模板。
- **阶段 1（NVFP4 GEMM）**：`kind::mxf4nvf4.block_scale` **只在 `compute_120a` 且必须显式 gencode**。
  寄存器映射打通，随机值全矩阵 **bad=0**。吞吐 21 → 80 → 351 → **502 TF/s**（padding 后）。
  decode 合并 8 专家一次 launch 达 **1930 GB/s**，30 层约 0.74 ms。
- **阶段 2（FP8 GEMM + 加载器）**：FP8 e4m3 m16n8k32 在 sm_120/sm_120a 都可用，
  ldmatrix + padding + cp.async 双缓冲把 **107 → 292 TF/s**。权重加载器跑通真实 checkpoint，
  三种格式分派正确，专家总量 12.85 GB。
- **投影**：FP8=280 + NVFP4=502 → 6K prefill ≈ 155 ms = 基线 **1.58×**，1.2× 纸面可达。
- **未完成**：阶段 3（Gemma4 层结构 + 滑窗注意力）、阶段 4（MoE 路由 + 服务）、阶段 5（对拍 + 调优）。
- **踩坑 4 条**（详见 `docs/pitfalls.md`）：NVFP4 的 K 是 `shape[1]*2`；对拍脚本参考侧未初始化导致
  假结论；`-arch=sm_120a` 不生成 `a` PTX；容器缺 libnccl/libnvrtc 未版本化软链接。
- **阶段 3 开工（同日晚，第二个会话）**：
  - 先把阶段 1/2 只存在于远端 `/hf_data/yaochen/work/*.cu` 的探针源码拉回 `models/kaon3/cuda/`（用户要求"代码要及时同步回来"）。
  - `models/kaon3/src/config.rs`：Gemma4 配置解析。**proportional RoPE 的分母是 head_dim（512）而不是 rope_dim（128）**，
    单测 `proportional_rope_full_layer_uses_head_dim_denom` 断言三件事：full 层 128/512、sliding 层 256/256、K==V 只在 full 层。
  - `crates/layers/src/nn.rs`：`Rope::with_denom(...)`，`new` 等价于 `denom = rope_dim`，其它模型零影响。
  - `crates/kernels/cuda/gqa.cu`：v5/v6/v7 三个 tensor-core kernel + 2 个参考 kernel 加 `window`；
    上下界 mask + 起始 tile `tile0` 跳读 + 双缓冲索引 `(tile-tile0)&1`。`CUDA_ARCHS=120 cargo build -p dinfer-kernels` **通过**（18 s）。
  - `kernels.h` / `ffi.rs`：新增 `gqa_prefill_win_bf16` / `gqa_paged_win_bf16`，旧符号转发 `window=0`。
  - **未完成**：`ops.rs` 包装、无权重 RMSNorm、`layer.rs`/`model.rs`、decode 侧滑窗、D=512 注意力。
- **踩坑 4 条**（详见 `docs/pitfalls.md`）：派生量必须声明在 `warp_min_vis`/`pair_min_vis` 之后；
  v7 的 `tile0` 要用 `pair_min_vis`；容器运行期要 `LD_LIBRARY_PATH`（只给 `LIBRARY_PATH` 会 127）；
  `--features no-nvcc` 分支已坏。

## 2026-10-18

- PD v2 (d) 跨卡：P GPU 3 / D GPU 4，TPOT 4.72 → 3.75（单机 3.53）、吞吐 1464 → 1786、TTFT 67 → 58 ms；Q4 PD 功能线收口。GPU 2 被别人进程占用，首次起 D OOM（pitfalls）。
- PD v2 (c)：proxy 先发 P 首 chunk、D 在飞；`first_token` 传 D 使超时回退不重复首 token（修了潜在 "redred" 类隐患的另一入口）。3090 e2e PASS；4090 e2e + 三故障 PASS，TTFT p50 97 → 67 ms、p99 423 → 140 ms，TPOT / 吞吐持平。loop 第 14–15 轮。两次远端跑空：`cargo build -p bench` 包名其实是 `dinfer-bench`；q4.sh 默认模型路径是本机的，4090 要 `QWEN3_DIR=/data/yaochen/infer/models/qwen3-0.6b`。

## 2026-10-17

- 00:00–00:30 `q3.sh` 重跑 exit 0 → **Q3 正式通过**。prefill v5（cp.async + NBUF 环形缓冲）：3090 双缓冲更慢，默认单缓冲。启动日志分段（weights host/h2d、scratch、kv）。`design-qwen3.md` 新增 §5（prefill_tc vs paged_tc：同工作量只差 3–30%，1.8× 是形状差异；服务选 paged 因历史 KV 零拷贝 + 调度统一）与 §6（优化空间：megakernel、跨流并行、启动、KV fp8、优先级）。commit 见 git log。
- 00:30–01:30 Q3-j 跨流并行 9 轮：Event API、ModelScratch 整套换入换出、executor AltSet + `forward_sample_pair`、runner 拆分散回、scheduler decode 免预算；bench17/19 持平，step 计数证明闭环 bench 下混批步极少（18/4000）；收益上限分析进 pitfalls。可读性整理一轮（helper 拆分、删 v3 302 行）。design.md §10 新增算子插拔 / 第三方 KV 方案（用户新需求）。
- 01:30–02:10 启动定位（loop `q3-startup.md` 4 轮）：`wall after` 累计墙钟二分 → layers 循环 1.6 s → `Attention::load` / `Mlp::load` 的 f32 host 拼接共 ~1 s；改 bf16 直传 + 设备拼接。4090 total 2.93 → 1.88 s，baseline PASS。pitfalls +1。
- 02:10–02:40 启动第 5 轮：BF16 权重 mmap 直接 H2D（host 拷贝 0）、tokenizer 独立线程并行；4090 total 1.88 → 1.05 s，baseline PASS。design-qwen3 §6.3 改为实测分段与剩余项。
- 02:40–03:40 启动第 6–10 轮：rope 多线程 122→21、mmap 并行填页 190→110、懒捕获图 engine 258→45、KV uninit、去逐张量同步；两次无效尝试（open∥Device::new 争资源；pinned 双缓冲 H2D 更慢）已回退并记录。4090 total **0.61–0.68 s**，baseline 全过。
- 03:40–04:10 bench 开环泊松到达 + 长短混合 prompt；`bench_poisson.sh` 首轮：TTFT 领先、TPOT 8/16 持平、24 req/s 落后 7%；跨流开关无差别（待 nsys 验证是否真并行）。
- 04:10–04:40 nsys 证实跨流真并行（39% 重叠）但混批步仅 5%；c32 stage compare 显示 GPU 构成 dinfer 不差，但 ms/step 口径不可靠（pitfalls）；c32 差距定位转下一会话。
- 04:40–06:00 decode B=32 差距定位：metrics 直方图口径 3.46 vs vLLM 3.40（离线）；launch 610 µs = block_table 全跨度上传（1.3 MB/步）；PinnedUploader 实现 + 两次翻车（INVALID_CONTEXT：裸调用没 bind ctx；ILLEGAL_ADDRESS：pinned 区回卷竞态）+ A/B 零收益 → 默认关；真修复 = block_table 跨度改 max_model_len/block_size。
- 06:00–07:00 发现 :8012 孤儿进程（20 h）；`restart_dinfer.sh`；干净重启 bench23：四组 1.12/1.06/1.05/1.01× 全 ≥ vLLM；block_table 跨度修复 launch 610 → 250 µs。pitfalls +3（裸驱动 bind_ctx、pinned 复用竞态、tmux 孤儿）。
- 07:00–08:30 Q3-l 异步调度 6 轮：runner begin/finish + Pending、engine pending/sync_done 双槽、executor row_gather 图内 + PinnedSlot 双槽 D2H；两处坑（切片误删 update_states；in-flight 容量 1 vs 设计 2）。bench24 四组 1.13/1.08/1.08/1.01×。CLAUDE.md 加 "超越不是重写" 规则 + design §11 采用/改进台账。
- 08:30–09:10 Q3-m 采样四 kernel 融合（row_* 原语共用，逐位同语义 ref），bench25 1.14/1.08/1.08/1.02×。
- 09:10–09:40 Q3-n gather+rmsnorm 融合（逐位 ref），bench26 持平；剩余两项并入 megakernel 阶段。
- 09:40–11:00 Q3-o 插队 prefill 6 轮：scheduler `schedule_waiting_only`、engine `publish_step` 提前回填、`CUT_IN_MAX_ROWS` 门槛；两次翻车（隐含假设、首 token 丢失）进 pitfalls；bench28 TTFT c8 14 → 8、c8-long 54 → 35。
- 11:00–11:30 design §10 细化完成（算子插拔 OpSet 三方案；KV 分层 KvTier + 4 个调度接入点 + PinnedHostTier → Remote → PD）。盘点：26 个 k:: 入口 + 1 Gemm（26 个 k:: 入口 + 1 个 Gemm）与 KV 管理器钩子（get_computed_blocks / allocate_slots / cache_blocks / free），再写成可落地的 trait 与接入点。
- 11:30–12:30 Q3-q OpSet 方案 A：opset.rs 四组 trait + DefaultOps，模型泛型化，moe/glm53 调用点同步；bench29 零差异。
- 12:30–14:30 Q3-r PinnedHostTier：索引 / 调度接入 / executor 竞技场 + copy stream / CLI / 指标 / e2e；第 7 轮 A/B 发现每块 28 次拷贝的 host 开销（TPOT +0.5）→ pack/unpack kernel + 每块一次拷贝；unsafe 全部下沉到 kernels::HostArena。
- 14:30–16:00 A/B 两次翻车（残留实例 → `ab_serve.sh`；引擎 livelock → 不自抢占 + 退避 + 启动校验 + 回归单测）；pitfalls +2。
- 16:00–17:30 第三次 A/B 发现 forced-Length 误判（末尾≠唯一）→ 修 + 单测；baseline 7/7；A/B 定论 0 失败、tier 命中但吞吐 -5%；泊松 / A/B 结果追加进 CSV（用户问）。

## 2026-10-16

- 03:10 会话第三次因 `Input is too long` /clear，按交接区恢复。恢复时工作区：Q2-a 采样 kernel 未提交改动 8 文件（+483 行）：`sampling.cu` 新增 `penalties_f32 / temperature_f32 / min_p_mask_f32 / logprobs_f32`，`ops.rs/ffi.rs` 封装，`ref_tests.rs` 4 个新用例，`gen_kernel_refs.py` 对应参考；`core::sample` 加 `has_penalties / validate`；`docs/loops/q2-serving.md` 规格已写（12 条停止条件）。`cargo build --features qwen3` 通过。下一动作：跑 `scripts/verify/kernels.sh` 验证 4 kernel → commit → 继续 Q2-a executor 接线。
- 03:20 `kernels.sh` 16/16 PASS，commit `5871963`（Q2-a kernels + loop 规格）。
- 03:30–05:10 Q2-a executor 接线完成（详见 `test/qwen3/2026-10-16/q2-serving.md`）：`core::sampler`、`BatchPlan.sample`、`ReqState.seed`、qwen3 executor 全采样管线 + logprobs、`EngineEvent::Logprobs`、server 采样字段 + 400 校验；`crates/qwen3/tests/sampling_parity.rs` 5/5，`real_parity` 3/3，`serve_e2e_qwen3` 1/1。loop 第 2–3 轮。
- 13:30–14:15 Q2-b 请求功能完成（loop 第 4–5 轮）：`crates/text/src/stop.rs` StopChecker（6 单测）、`HfTokenizer::token_text`、`server/backend.rs` trait 拆分 + submit 回 id + abort、`server/openai.rs` 新字段与 logprobs 形状、`server/chat.rs` 重写、qwen3/glm53 service 适配、`serve_e2e_qwen3::e2e_qwen3_request_features`。4090 上 e2e 2/2 PASS；两条 feature 均编译、clippy 干净。用户口径：请求语义对齐 OpenAI/vLLM，实现按性能来（stop 只扫增量、命中即 abort）。
- 14:15–15:05 Q2-c/d/e/f 完成（loop 第 6–13 轮）：auto num_blocks + prefix cache 默认开 + `/metrics` + 健壮性 e2e + `scripts/verify/q2.sh`；4090 q2.sh exit 0。踩坑 3 条进 pitfalls（prefix cache 开后逐字断言翻 near-tie；tracing 字段名 ANSI 色破坏日志断言；n=3 跨调用 batch 形状不同不保证逐字一致）。用户指正：GLM 线不要主动碰（Q2-b 改 trait 时只为编译同步 glm53 impl，不再跑其 e2e）；请求语义对齐 OpenAI/vLLM、实现按性能来。
- 15:10 目录调整（用户方案）：`crates/{qwen3,glm53,kimik3}` → `models/`，`crates/` 只留公用基础设施；Cargo.toml members / path deps、design.md 布局树、CLAUDE.md、pitfalls / loops 现行引用全部改；`test/` 归档与本文件历史日志按原路径保留。两条 feature clippy 全目标干净。
- 15:05–15:20 部署：4090 上 dinfer Qwen3-0.6B 常驻（tmux `dinfer`，GPU 0，:8012，auto 3534 块；用户问句冒烟 OK）；vLLM v0.31.0 Docker 常驻（容器 `vllm-qwen3`，GPU 1，:8013，KV 181k token）作 Q3 基线。`scripts/dev/{chat.sh,vllm_docker.sh}`。用户指正：不要放行端口，在 4090 上测把结果拿回来。
- 15:20–15:50 Q3 准备：① 用户问 `--gpu-memory-utilization 0.9` → 发现 logits 缓冲按 max_tokens×vocab 分配白占 ~10 GiB，改 `max_logits_rows`（= max_num_seqs），4090 KV 6.0 → 14.5 GiB（pitfalls）；② `bins/bench` 压测工具（vLLM 口径，`--self-check` 进程内 mock 自检 PASS）+ `scripts/verify/bench_compare.sh`；③ dinfer 改 `--release` 常驻；首轮 dinfer vs vLLM 对比在 4090 跑，结果 `test/qwen3/2026-10-16/bench/summary.md`。
- 全天 commit 20 个，`dev` 已推 GitHub（`git@github.com:rayze-ai/dinfer.git`）。
- 15:30–22:20 Q3 性能（loop 1–26 轮，详见 `docs/loops/q3-perf.md` 记录表与 `test/qwen3/perf-trend.md`）：bench（基线 0.49/0.50/0.34/0.03×）→ a prefill flash → b decode split-K → c CUDA graph → d 逐段对比（GPU-bound；Nagle）→ e 融合（qkv_post、跨度读、残差并入、decode v2、merge 跳过、prefill/paged v2）→ **bench10：1.10 / 1.05 / 0.99 / 0.74×**，TTFT 四组 4.8 / 9.4 / 41 / 127 ms（vLLM 12 / 26 / 63 / 36）。工具：`bins/bench`（自检）、`bench_compare.sh <OUT> "改动"`、`bench_trend.py` → `perf-trend.{csv,html,svg,md}`、`trend_server.sh`（4090 :8090 面板，点轮次看详情）、`stage_profile.sh` + `nsys_stage_compare.py`、`cold_shape.rs`、`graph_probe.rs`、`q3.sh`。用户要求落实：目录 `test/<model>/<日期>/`；每轮改动→指标提升进 CSV + 浏览器实时曲线。踩坑 4 条（E0428、libstdc++、LD_LIBRARY_PATH、prefix-cache 逐字断言、ANSI、batch 形状、nsys 容器、`--delete` 覆盖 bench7、分叉 shuffle 死锁）。
- 晚：第 4 次 `Input too long` /clear 后恢复。加硬约束（settings env + PreToolUse hook + CLAUDE.md 规则）。bench11（prefill v3）归档：c32 首次 1.00×；c8-long 0.83×。发现 bench 固定 seed 使常驻 vLLM prefix cache 90% 命中（口径不公），`bench_compare.sh` 改每次换 seed；bench12（公平口径）c1 1.12× / c8 1.06× / c32 1.04× / c8-long 0.91×。nsys 分解 c8-long（paged_tc 1.8× 慢于 prefill_tc；decode_v2 25.7%）。第 31 轮 decode splits 256 → 1024（4090 B=8 seq=2048 0.040 → 0.032 ms），bench13 见 perf-trend。
- 深夜：第 29–39 轮。口径修正（每轮换 seed）→ bench12 0.91×；nsys 分解 c8-long；decode splits 1024（bench13 持平）；混批分流（bench14 持平）；单发 1024 TTFT 25 ms 定位到 prefill attention 占 step ~80%；**prefill v4（FA2 结构 mma.sync）4.8×**，bench15 四组 1.12 / 1.07 / 1.02 / 1.00× 首次全 ≥ vLLM。pitfalls +3。q3.sh 验收结果见下一条。
- `scripts/verify/q3.sh` 首次正式运行（`test/qwen3/2026-10-16/q3/`）：refs 22 / parity 8 过；ratio：c1 1.13× OK、c8 1.07× OK、c32 1.02× OK、**c8-long 1911/1911 = 1.00× 判 BELOW（严格 ≥1.0 差 1 个 tok/s）→ Q3 未正式通过**，差 c8-long 一组。bench16（预算 2048，常驻已改）：c8-long 1.005×、TTFT 39 vs 68 ms、c32 TTFT 20 vs 76；TPOT c8-long 3.9 vs 3.7 仍略慢 → 下一轮攻 prefill step 时长（cp.async 双缓冲）与 prefill/decode 交错。

## 2026-10-19（夜，Qwen3.5 真权重 e2e）

- 用户："qwen35 在 B300-50 上跑测试包括 vllm 基准测试，跑完再跑 k3"。
- sm_103 tiny golden 4/4 + 单测 13/13；真权重 serve 首请求 `CUDA_ERROR_INVALID_CONTEXT` → 引擎线程 `bind_ctx`；修后输出乱码（logprob 全 −9）→ HF 参考（vLLM 镜像内 transformers 5.15.1）逐层范数对拍定位 L0 起偏 → L0 内部对拍 mlp_out 9.19 vs 0.958 → 根因 zero-centered RMSNorm `(1+w)`；修后 greedy 与 vLLM 逐字一致。
- bench_b300_1 → 发现 c8-long 87 条 GDN slot 告警 → 先修 slot 归还（bench_b300_2 不变）→ 再定位是 prefix cache 命中（pos 144/160/176 = 模板前缀块）→ 强制关（bench_b300_3 告警 0）。
- 产出 commit：c7a78fb / f5fbaba / e64a989 / 9c54159 / 26e8501 / fc12d86 / ff0c912 / 37f28a0。调试脚本归档 `test/qwen35/2026-10-19/l0_debug/`。
- 启动 vLLM Kimi-K3 tp6（GPU 2–7）基线容器，加载中。

## 2026-10-20（wrap：qwen35 多轮 / 长 prompt 双线收口）

**做了什么**
- r8（子代理）：GDN 状态按块快照 + scheduler 对齐 → prefix cache 重开。engine rate 8 goodput 1.32 → 3.04（vLLM 2.87），TPOT p99 31.4 → 18.8。顺带修了公共 scheduler 的既有 bug（异步调度下被抢占请求丢一个采样 token）。
- r8b（子代理）：kernel 沿 chunk 边界写快照、调度不再切步 → 6 K 单发 346.8 → 291.8 ms / 1 步。
- r10（子代理，8 轮）：GDN chunk kernel 三段，8×1024 全链 1109 → **770 µs（−31%）**，不含 conv 反超 vLLM fla 26%；1×8192 1271 → 969。
- 主会话：engine 口径 8c–8y（纯 prefill 分解、fla 对照、TTFT 长度扫描、4 次合体/门禁）、装载骨架抽公共件（`crates/layers/src/load.rs`，四模型统一三段启动日志）、probe 页签分类、`ttft_probe.py` 入库、`build.rs` nvcc 并行、`k3_model.sh` 空跑修复、engine_trend 排序修复。
- K3（子代理）：全 93 层 tp8 首轮跑通（TPOT 慢 19–43×），修 `cudaFuncSetAttribute` static 缓存按设备生效的坑（7 处改 `thread_local`，Qwen3.5 TP 同受益）；greedy 低熵坍缩未解，按用户决定挂起。

**怎么验证的**：B300 四个口径（四组闭环 / 6 K 单发 / engine rate 4,8）+ 4090 公共件门禁（`bench_r8bgate`，最差 +1.6%）；两卡 `gdn_parity` 15/15、golden 11/11、core 68 单测。归档：`test/qwen35/2026-10-20/`（bench_b300_17/18/19、engine_merge、engine_final、probe_gdn_r9_0b300…r10_8、probe_prefill_1…7）、`test/kimik3/2026-10-20/`（c6_tp8_1/2、model24、probe_c6_tp8_1/2）、`test/qwen3/2026-10-20/`（bench_r8gate / r9gate / r8bgate）。

**未完成**：见交接区"下一步"6 条。

**踩的坑**（详见 `docs/pitfalls.md`）：`cudaFuncSetAttribute` static 缓存按设备生效（多线程 TP 卡死）；`k3_model.sh` DUMP 为空时 `RUST_LOG=info` 被当命令名、cargo test 空跑；`pkill -f "dinfer serve"` 匹配包装 bash 自杀；`pgrep -f` 等待自匹配自锁；4090 是共享机（外来 vLLM 占卡 → KV 9900 → 708 → 首请求 OOM）；门禁预算口径 16384 ≠ 2048；远端容器 root 产物让 push --delete 报错退出；探针目录混放三个对比项。

## 2026-10-20（K3 里程碑 C-5 TP + EP，子代理）
- 设计：`docs/design-kimik3.md` §2.4 切分表（每张量切法 / 每 rank 字节 / 通信点）；EP v1 = 本地全量路由 + 本地专家段 + acc f32 all-reduce（不 all-to-all、不 all-gather，design.md §11 一行）。每层 3 次通信；world=8 全模型 ≈ 209 GB/卡（B300 288 GB 能放 93 层）。
- 代码（commit 77f8b91 / e02ffc7 / 2ca2d29）：`weights.rs` `Shard{rank,world}` + `Table::linear_shard / load_f32_shard1d`、`load_conv1d_shard / load_a_log_shard`、`KdaShapes::with_shard`；`kda.rs::load_tp`（q/k/v/g/f_b/b/conv/A_log/dt_bias 行切、o_proj 列切、状态只存本 rank 头）；`mla.rs::load_tp`（g/q_b/kv_b/W_UK/W_UV 按头、o_proj 列切、q_a/kv_a/cache 复制）；`latent_moe.rs::load_tp` + `ExpertPack::load_range`（专家 id 段）+ `forward_tp`（本段 pair 区间 `[p_lo,p_hi)`、local offsets、acc all-reduce f32、shared `Mlp::load_tp` + all-reduce）；`layer.rs` `StepMeta.comm` + attn / dense all-reduce；`model.rs::load_tp`；`executor.rs::new_with_tp`（Comm 在 executor 流上、图捕获前）；`tp.rs` `TpExecutor`（照 qwen3 去 PD）；`service.rs` 每 rank 一线程装载 + `ServeOptions.tp`；`bins` `--tp` 接 kimik3。
- 验证（本机 3090）：四 feature `cargo build --workspace` 全过；`cargo test -p dinfer-kimik3` 24 过；`k3_chat.sh` 过；clippy 本 crate 新代码无警告。
- B300 GPU 2–3（11:00–11:18，4 轮，`docs/loops/k3-tp.md`）：单卡 `k3_model.sh` exit 0 且隔离数字与 C-3 记录逐位同（不退化）；`K3_TP=2` 第 1 次因进程级 QUIET 看不到数字、第 2 次链式 logits 0.99825 < 0.999 → 单卡 ×2 发现自身链式只有 0.9992（MoE 原子加 + 平局）→ 中间量 dump 定位 attn/mlp_in 4e-6 一致、差在 MoE → 按"两边 topk_ids 相同的行"拆比 expert_sum ≤ 1 ulp → **机制正确，口径改为隔离一致行 ≥ 0.9999 + top-1**；`TP_REDUCE_F32=1` 实验无影响（排除 bf16 双舍入）。tp2 smoke exit 0（c1 TPOT 9.65 / c8 25.2 ms；TTFT 217 / 1234 ms）。
- 坑（已进 pitfalls）：线程静默用 thread_local；MoE 模型链式 cos 不能做 tp 停止条件；远端 `pkill -f` 杀自己；B300 token 两次到手即过期（主会话处理）。
## 2026-10-20（K3 里程碑 C-4b，子代理）

- `executor.rs`：`KimiK3Executor: ModelExecutor`，**直接用公共件** `PlanBuffers`（公共列 + KDA `state_idx/has_init` 追加）/ `StateSlotTable`（容量 = `num_slots - 1`，最后一槽废纸篓，`finished_req_ids` 归还）/ `DecodeGraphs`（Prepare 图外 H2D、Body 只 launch；假 plan KDA 指废纸篓、MLA slot=-1）/ `AsyncResultRing`（begin/finish）。prefill：KDA chunk（`KdaChunkWorkspace` 全模型共用，按步 prepare）+ MLA 展开；decode：KDA 递归 + MLA 吸收读 paged latent。
- `model.rs`：`forward_rows`（只对采样行 `gather_rmsnorm` + lm_head；`forward` 保持每行 logits 给对拍）、`load_with_rows`、`reset_slots`、truncated warn。
- `service.rs`：`ServeOptions{layers, chat_dialect, ..}` + `KimiK3Service::start` + `impl ChatBackend`：`KvCacheSpec::Mla{head_size 576}` 块数 = `max_num_seqs × max_model_len / bs`（MLA 每 token 1.1 KB，不按剩余显存压满；装载前就得定）、KDA 槽 = `max_num_seqs + 1`、prefix cache 强制关；render 走 `xtml`（`reasoning_effort` → thinking/effort；`--chat-dialect hf|serving`）；输出流 bytes 增量 detokenize **不跳特殊 token**（parser 靠 XTML 标记切段，只滤 eos 163586）。
- `bins/dinfer`：`kimik3` 分支接 serve；新参数 `--layers N`（= `KIMIK3_LAYERS`）、`--chat-dialect`。
- B300 冒烟两轮（`scripts/dev/k3_b300_smoke.sh`，GPU 2，`:8014`）：第 1 轮首请求 `graph capture: body: LatentMoe: pairs 0 ≠ t·K`（LatentMoe 体内 `dtoh_sync_copy(offsets)`、AttnRes host softmax → capture 内 D2H 读到 0，引擎线程退出）；第 2 轮图默认关后 **exit 0**，数字见交接区 / `docs/loops/k3-serve.md`。
- 验证：四 feature workspace 构建、`cargo test -p dinfer-kimik3` 24/24、ref-tests `--no-run`、kimik3 clippy 干净；`k3_model.sh`（B300 复跑）见 loop 记录。
- 未做 / TODO：decode 图（两处 host 同步要搬 GPU）、guided / 强制 tool_choice（`tool_call_tags` None）、`OutputStream` 拿不到 render 的 thinking 开关、采样流水三份同文待抽 `exec::Sampler`。
- 踩坑：① capture 体内任何 `dtoh_sync_copy` 都会让后续 host 逻辑读到 0（表现为 shape 断言而非 CUDA 错），进 pitfalls；② `cargo build -p bench` 不是包名（`dinfer-bench`）；③ 远端 `dinfer` 目录不是 git 仓库，`git log` 看不到版本，靠 `put` 同步文件。

## 2026-10-20（K3 里程碑 C-4a，子代理）

- tokenizer / XTML / parser 三模块 + golden 对拍一轮全过（细节、与 vLLM 的差异、采纳 `llm-cache-prediction` 哪些口径见 `docs/loops/k3-chat.md`）。
- 事实：基线 vLLM `de8291e` 对 K3 走 HF Python `apply_chat_template`（无 Rust renderer），所以默认方言 Hf；Moonshot 线上 Rust renderer 口径做成 `Dialect::Serving`（浮点塌整数、参数重序列化、丢 `name`），golden 两份都有。
- 踩坑：Python `regex` 模块不认 `&&` 类交集，不能当 tiktoken 预分词参照（tiktoken 内核是 `fancy_regex`，我们同款）。
- 新依赖：`fancy-regex =0.14.0`、`base64 =0.22.1`（workspace 固定版本）。

## 2026-10-20（K3 里程碑 C-3，子代理）

- `mla.rs`（MLA 层，与 HF 核对：无 RoPE 但 rope 64 维存在并参与点积、v pad 等价、输出门、fused 投影、a_layernorm eps 1e-6）、`model.rs`（多层 + 块残差 + output_attn_res + lm_head，只加载前 N 层）、`layer.rs` `Attn::Mla` / `LayerBudget` / `StepMeta`。
- 对拍三轮：第 1 轮远端 refs 目录被别的会话 `push --delete` 删掉（pitfalls）；第 2 轮 layer 4 链式 cos 0.99877 差门槛；第 3 轮改两口径（隔离断言 / 链式报数字 + top-1）+ eps 修正后 exit 0。数字见 `docs/loops/k3-model.md`。
- 三 feature workspace 构建、`cargo test --no-run --features ref-tests`、kimik3 clippy 干净。

## 2026-10-20（K3 里程碑 C-2，子代理）

- `latent_moe.rs`：`ExpertPack::load` 把 896 专家的 w1/w3 拼成 `w13 [E, 6144, 1792]` + w2，packed 常驻（启动日志打字节数）；forward = `moe_gate_f32` → `moe_route_f32` → `down_proj` → `moe_bucket`（1 次 D2H）→ gather → grouped GEMM w13 → `situ_and_mul` → grouped GEMM w2 → `scatter_add_f32` → bf16 → `routed_expert_norm` → `up_proj` + shared `Mlp`（`Situ: Act`）。CPU 路由单测 + situ 公式单测。
- `layer.rs`：`KimiDecoderLayer{attn: Kda, ffn: Dense|LatentMoe, attn_res}`，MLA 层 bail；`AttnRes::apply` = 每候选 `rmsnorm` + `moe_gate_f32`(E=1) f32 分数 → host softmax → `moe_scatter_add_f32` 加权和。`writes_block()` 给 model.rs 用。
- 与任务书不同：路由分数在 7168 维 `mn` 上算（down_proj 之前）；第 1 层 AttnRes 只依赖第 0 层写入的块残差 x₀（第 0 层输入），零初始会退化，参考 / 测试用随机 x₀；shared experts 在分片里已是一个 `[6144,7168]` MLP。
- 对拍两轮：第 1 轮 MoE 子块按全部行断言挂（3 个路由平局 pair → 该 token 专家不同，`ysum` rel 0.145）；第 2 轮改为"路由一致行判数值 + 不一致行必须是平局"→ exit 0。
- 踩坑进 pitfalls：平局行口径；共享文件（kernels.h/ffi.rs/ops.rs）被另一子代理 commit 带走声明、定义留在工作区 → HEAD 链接不过；`htod_into` 不接 `CudaViewMut`。
- 三 feature workspace 构建、`cargo test --no-run --features ref-tests`、kimik3 clippy 干净。`ops_parity.sh` 未跑（另一子代理占着 GEMM/ops 工作区，activation.cu 只加了独立符号）。

## 日志

- 2026-10-18（晚，第三段）：PD 后端 b 共享槽区 shm / cuda-ipc 实现（kernels::shared、core::pd SharedTransport+credit、executor 槽导入导出、CLI）；IPC 探针证实 4090 跨卡无 P2P；core 46 单测 + 3090 两后端 e2e PASS；4090 bench 待跑（见交接区）。

### 2026-10-13

- 做了什么：用 project-skill 初始化项目骨架。
- 怎么验证的：—
- 未完成：—
- 踩坑：—

### 2026-10-13（第二次会话）

- 做了什么：
  - clone vLLM main（de8291e / 0.31.0）到 `/data/ref/vllm`；从 HF 拉取 Kimi-K3 与 GLM-5.3 的 config / modeling / tokenizer / chat 模板 / safetensors index 到 `/data/ref/models`。
  - 读 vLLM v1 的 engine/core、scheduler output、kv_cache_interface、cudagraph_dispatcher、gpu_worker、sampler metadata，以及 `vllm/models/kimi_k3`、`vllm/models/deepseek_v32` 的模型/attention/MoE/MTP 实现。
  - 写 `docs/design.md`：vLLM 启动与推理链路分析、两模型结构事实表、6 crate 划分、每 crate 模块清单、kernel 清单、P0–P10 分阶段计划、决策与风险；补 `docs/requirements.md` 的"不做什么 / 验收口径 / 约束"（标注待确认）；回写 CLAUDE.md"项目是什么"。
- 怎么验证的：结构参数均出自 config.json 与 vLLM 源码，文中带文件:行号；未做运行验证（无代码）。后台"启动路径"分析返回后核实并补进 design.md §1.4b：Model Runner V2 为默认、两模型走 breakable cudagraph 不走 torch.compile、GLM 用通用 hf tokenizer/parser 模式。

### 2026-10-16（文档拆分）

- 做了什么：`docs/design.md` 439 行 → 376 行；模型相关内容拆到 `design-kimik3.md` / `design-glm53.md` / `design-qwen3.md`（每份：§0 状态、§1 结构事实、§2 crate 模块、§3 模型特有风险、§4 参考材料）。design.md §2 改为模型索引表 + 共享部分，§4.5 合并为 "模型 crate" 指针，§5b 的 Qwen3 结构事实移走，§8 下错编号的 9.x 子节改为 8.x。README / CLAUDE.md / progress / qwen3 代码注释里的引用同步改。
- 怎么验证的：`grep` 无悬空 §2.1/§2.2/§4.5/§4.6 引用；qwen3 crate 编译通过（只改了注释）。
- 未完成：—
- 踩坑：—

### 2026-10-15（方向调整 + Q1 Qwen3-0.6B 真权重，通过）

- 做了什么：
  - 用户决定：无高端卡，GLM/K3 真权重挂起，先用 Qwen3-0.6B 验证主流程（design.md §5b，Q1–Q5）。下载 `/data/ref/models/qwen3-0.6b`（含 1.5 GB 权重）。
  - 抽 `crates/text`：`HfTokenizer` / `ChatTemplate`（模板源支持 jinja 文件或 tokenizer_config 字段；`RenderParams.extra`）；glm53 的 tokenizer/chat 变薄封装，15/15 golden 仍过。
  - `gqa.cu` 三 kernel + torch 对拍 2/2；`crates/qwen3` 全套；bin `--features qwen3`；e2e 真模型问 1+1 → reasoning 468 字 + content "2"，`enable_thinking=false` → "1+1=2"。
  - 对拍口径：真权重 bf16 下 HF eager vs sdpa 自身差 0.68（logits）/ 9.0（L27 hidden），我们的偏差同量级；greedy 32 步逐 token全对是更强证据。`load_bf16` 批量拷 59 s → 17 s。
- 怎么验证的：`scripts/verify/q1.sh` exit 0（`test/2026-10-15/q1.log`）；细节 `q1-qwen3.md`。
- 未完成：Q2 采样等（见交接区）。
- 踩坑：见 `docs/loops/q1-qwen3.md` 4 轮；无新 pitfalls 条目（都是口径 / 性能，不是坑）。

### 2026-10-15（P6 GLM-5.3 tokenizer / chat / server / MTP，通过）

- 做了什么：
  - a：`tokenizer.rs`（`tokenizers` 0.21 加载 `tokenizer.json`，eos 从 generation_config，`DecodeStream` 增量 decode）；`chat.rs`（minijinja 2.10 跑官方 `chat_template.jinja`：HF 环境 `trim_blocks/lstrip_blocks`、`.N.attr` 改写、Python 方法 shim、Python 风格 `tojson`、`preserve_order`）；`gen_chat_golden.py` 15 case；`chat_parity` 4 测试全过（逐字符 / 逐 id）。下载了 `tokenizer.json`（20 MB）到 `/data/ref/models/glm53`。
  - b：`parser.rs` 流式解析（`<think>…</think>` → reasoning，`<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>` → tool_calls，标签前缀保留策略），8 单测含 "每个字符边界切分" 枚举。
  - c：`core/engine.rs`（请求通道 → Scheduler → Runner → 每请求 `EngineEvent` 流，空闲阻塞）；`server/{backend,openai,chat}.rs`（模型无关 `ChatBackend` trait；OpenAI 类型含 reasoning_content / tool_calls / stream_options / reasoning_effort / clear_thinking；SSE：role chunk → delta → finish → usage → [DONE]）；`glm53/service.rs`（`ChatBackend` 实现）；bin `serve`；`serve_e2e` 起真进程打 HTTP 全过。顺手删掉索引器 / ops 两处过严的 `block_table 覆盖 max_seq` 检查。
  - d：`mtp.rs`（enorm/hnorm/eh_proj + 同构 decoder 层 + shared_head.norm，复用主干 embed/lm_head/共享 top-k）；runner 协议扩展（`spec_tokens / logits_row_start / next_known`，executor 返回 `(generated, drafts)`，拒绝回退）；executor greedy verify + MTP draft（位置 pos+1，KV 按位置重写无需回滚）；engine 把 draft 交回 scheduler；`--mtp` 开关；`gen_tiny_glm.py --mtp` 用 HF decoder 层 + 胶水做参考；`mtp_parity` 3 测试全过。
- 怎么验证的：`scripts/verify/p6.sh` exit 0（`test/2026-10-15/p6.log`）；各步数字见 `test/2026-10-15/p6-glm-serve.md`。
- 未完成：真权重跑通（需大机器）；采样 / logprobs / metrics 等见交接区 "P6 已知简化"。
- 收功（同日晚）：工作区干净，HEAD `ee05aa6`。本会话 3 次重启（Input too long → /clear；bypass 权限两次重开），P5 收尾 + P6 全部在同一天完成；下次开场直接按 "下一步" 选 P7 / P10。
- 踩坑：见 `docs/loops/p6-glm-serve.md` 记录表 8 轮；可提炼进 pitfalls 的两条：minijinja 跑 HF 模板的 5 个坑（已写）、runner 侧 `block_table` 宽度语义（已写）。

### 2026-10-15（P5 DSA，进行中）

- 做了什么：
  - 对拍基准升级：本机 transformers 4.52.4 无 `GlmMoeDsa`；建隔离 venv `/data/ref/venv-tf519`（`--system-site-packages`，复用 torch 2.6.0+cu124）装 transformers 5.19.0；源码抽到 `/data/ref/hf_tf519/`。比对 5.19 的 MoE/MLA/RoPE 语义与 4.52 DeepseekV3 一致，P3/P4 结论不受影响。
  - `docs/loops/p5-dsa-indexer.md`：目标 / 环境 / 观察方法 / 停止条件 / 8 个已知风险点。
  - kernels：`dsa.cu`（`layernorm_bf16`、`dsa_index_score_f32`（paged 索引 cache 上 Σ_h w·relu(q·k·scale)）、`dsa_topk_i32`、`dsa_row_vis_i32`）；`mla.cu` 三个 attention kernel 加可选 `topk_indices`（gather 模式：每行只看列出的 key）；`scale_f32`。
  - layers：`indexer.rs`（`Indexer` / `IndexerKv` / `IndexerMeta`：wq_b、wk、k_norm LayerNorm、weights_proj、前 rope 维 interleave RoPE、写 paged 索引 cache、score + top-k）；`mla.rs`：`AttnMeta::Prefill` 加 block_table/seq_lens/max_blocks，`SparseCtx`（索引器 + 共享 top-k 缓冲 + decode_cu + 对拍注入 `topk_override`），`AttnMeta::indexer_meta()`。
  - glm53：`config.rs` 加 `indexer_types` / `dsa`（`from_json` 在真 config 带 indexer_types 时自动开）；`model.rs`：每 full 层 `Option<(Indexer, IndexerKv)>`，跨层共享 `topk` 缓冲，`capture_topk`（含索引器分数）/ `topk_override` / `capture_layers` 三个对拍钩子。
  - `gen_tiny_glm.py --dsa`：用 5.19 GlmMoeDsa（3 层 full/shared/full，4 头 × 32，top-k 8，索引器权重 ×6 让分数有分辨力），拆融合专家权重为逐专家布局，导出每 full 层 top-k 与 index_scores（prefill 短/长 prompt + 32 个 decode 步）、路由 margin。`tests/dsa_parity.rs` 三道测试；`scripts/verify/p5.sh`。
- 怎么验证的（进行中）：注入 HF top-k 后三层 attention 输出与 HF hook 逐层对比 max abs ≤ 0.0007；256 行里 252 行 logits < 0.01，剩 4 行是 MoE 路由平局（见 pitfalls）。dense / MoE / fp8 回归仍全过。p5.sh 尚未 exit 0（最后一轮测试重写后未跑完：Bash 工具被平台分类器连续拦截）。
- 同日 16:30（会话因 `Input is too long` 第二次 /clear 后恢复，只做状态记录）：`dsa_parity` 编译通过，fixture 16:22 已重生成，实跑 3 道 **全 FAIL**，均为"可判定行不足 / 平局率过高"类阈值失败，可判定行上误差全部达标；细节与下一步候选见 `test/2026-10-15/p5-dsa-parity.md`，loop 记录表第 5 轮。
- 同日 17:00–19:10（第 6–7 轮，`dsa_parity` 3/3 PASS）：
  - 诊断 0.11 误差：`dsa_diag.rs` 两模式对比 → 自算时 L0 平局翻转污染 L2 索引器输入（注入 HF top-k 后 L2 max 0.040）。`check_topk` 改注入模式 + `SparseCtx::topk_self` 保存覆盖前自算值。
  - 路由平局：fixture `mlp.gate ×8`（near-tie 17.6% → 10.5%）；过滤改用实测 hidden 漂移（`gen_tiny_glm.py::dump_hidden` 导出逐层 hidden，`drift_rows`），prefill 真漂移只有 1–2 行（路由 margin 代理是 37 行）。
  - decode 第 2 步起 L1 attention 0.02：逐 head pre-o_proj + cache 行 bisect → 我们 cache 对、HF 自己不前缀稳定（258-token 前向里 5 个前缀行 L0 top-k 翻转）。fixture 每步导出 `hidden_L*_long_prefix_drift_steps.npy`，并入漂移集合。
  - 顺手：全部 `DINFER_*` 环境变量改名；`model.rs` 三个 clippy type-complexity 用 type alias；验证脚本每个 `step` 前 `rm target/.rustc_info.json` 且所有 cargo `</dev/null`（cron trace 污染在 p4.sh 跑到中途又发生了两次）。
- 怎么验证的：`dsa_parity` 3/3 PASS（`test/2026-10-15/p5-dsa-parity.md`）；`scripts/verify/p4.sh` exit 0（回归）；`scripts/verify/p5.sh` 全量见下一条。
- 同日 19:25：`scripts/verify/p5.sh` 全量 **exit 0**（p0–p4 回归 + fixture 重生成 + dsa_parity 3/3），日志 `test/2026-10-15/p5.log`。commit P5。
- 未完成：P6 规格。
- 踩坑：见 pitfalls 2026-10-15 三条（硬判决不可逐位复现；transformers 5 融合专家权重；会话 Input too long）。另：静态 shared 64 KB 超 48 KB 上限 → cudaError 1。

### 2026-10-14 深夜 → 2026-10-15（P4）

- 做了什么（会话因 "Input is too long" 清空一次，按交接区恢复）：
  - P4-c MoE：补 `moe_gate_f32`（fp32 路由 logits）、gather/scatter 带 start 偏移、`Stream::memset_zeros_n`；`layers::Moe` 改为逐专家 gather → Mlp → 加权 scatter-add（fp32 累加器）→ `bf16(acc) + shared`；`Glm53Dense` → `Glm53Model` + `FeedForward::{Dense, Moe(Box)}`；tiny MoE GLM 3 对拍 PASS（prefill 0.0069 / 32 步 worst 0.0078）。commit `35b54e3`。
  - P4-a runner：`core::runner`（`ReqState / BatchPlan / ModelExecutor / Runner`，2 mock 单测）、`Glm53Executor`、`runner_parity.rs`（2 并发请求 prompt 14/11，1 prefill + 31 decode）；`gen_tiny_glm.py` 加第二条 prompt 与期望。dense worst 0.0063 / MoE 0.0071。commit `69c8478`。
  - P4-b 混合 batch：`mla_prefill_paged_bf16`（吸收路径对 paged cache 做 causal，任意 query 长）、`AttnMeta::Paged`、`absorbed_q/absorbed_o` 抽取、`BatchMode::{Prefill,Decode,Paged}`、runner 中间 chunk 不追加采样；`runner_chunked_mixed_match_hf`（预算 8 → 35 步，Paged 3 步）dense 0.0059 / MoE 0.0078。p4.sh PASS，待 commit。
  - P4-d fp8：`weights_fp8.rs` + `WeightSource` F8_E4M3 分支 + `GlmMoeDsaConfig::fp8_block_size` + `gen_tiny_glm.py --fp8` + p4.sh fp8 步。tiny fp8 GLM 3+2 对拍全过（prefill 0.0076，32 步 worst 0.0073，runner 0.0077/0.0078）。
  - 文档：pitfalls 新增 4 条（rustc_info 污染、fp32 路由、host argmax 平局、finished_req_ids 延迟）；design 阶段表 P4 行；CLAUDE.md 常用命令。
- 怎么验证的：`scripts/verify/p4.sh` exit 0（P4-b 之后、fp8 之前的版本，日志 `test/2026-10-14/p4.log`）。
- 后续（同日）：Bash 恢复后编译通过、fp8 2 单测过、`p4.sh` 含 fp8 步全过（`test/2026-10-14/p4.log`）；commit P4-b `ea415e3`，P4-d + docs 随后 commit。另：root shell 下 git 报 "dubious ownership"（仓库属主 yaochen），已 `git config --global --add safe.directory /data/code/dinfer`。
- 未完成：无。
- 踩坑：cargo 假失败（`target/.rustc_info.json` 被 cron `set -x` 输出污染）排查约 40 分钟，见 pitfalls。

### 2026-10-14

- 做了什么：按用户要求补性能分析。读 V2 runner（`gpu/model_runner.py`、`states.py`、`buffer_utils.py`、`async_utils.py`）、`core.py step/busy loop`、`async_scheduler.py`、`detokenizer.py`、`cache.py`，写 design.md §8：16 项瓶颈（B1-B16，标注源码核实/经验判断）、每层算子融合目标、对标 vs 另起清单、基准方法；requirements 验收口径加性能指标；P1/P3 起加性能早期门禁。
- 怎么验证的：源码引用带行号；性能数字（launch 数、µs 目标）是设计目标，待 P1/P3 实测。
- 未完成：同 10-13（用户确认计划 → 基线 commit → P0）。CLAUDE.md 已补齐。
- 同日深夜（P1）：读 vLLM `scheduler.py:591-1660`、`kv_cache_manager.py:264-620`、`block_pool.py`、`single_type_kv_cache_manager.py`（FullAttention/Mamba 的 find_longest_cache_hit、get_num_blocks_to_allocate）、`kv_cache_coordinator.py:796`，对齐语义后写 Rust。三轮迭代：① 初版 HashMap 版 27 测试 20 过，bench 298/449 µs；② 修三个测试预期（32 token 恰满 2 块、抢占恢复会命中自己留下的缓存块、阈值测试时序）+ slab/恒等哈希/扁平结果/跨边界早退，bench 197/310；③ 发现 bench 的 `iter_batched` 把 setup 算进去了（手工计时只有 62/92），改持久状态 `iter` → 95/146。`scripts/verify/p1.sh` PASS。
- 踩坑：criterion `iter_batched`（见 pitfalls）；本机无 perf。
- 同日更晚（P2 开工）：写出 5 个参考 kernel（norm / rope / activation / sampling / add）+ `common.cuh`、cuBLASLt bf16 GEMM 封装、`ops.rs` 类型安全包装、`ref_tests.rs`、`gen_kernel_refs.py`、`kernels.sh` / `p2.sh`、`docs/loops/p2-kernels.md`。用户要求 C ABI 符号去掉 `dinfer_` 前缀，已全部改名。sampling.cu 的 `atomicMin` 初值从 -1 改为 INT_MAX。**全部未编译**：Bash 工具再次被平台拦截（本轮约 15 次）。
- 同日晚：P0 编译通过并 PASS。三轮修复：① cc `cuda(true)` 塞 gcc 参数 → build.rs 直接调 nvcc + `object()` 归档；② bin 直接用 `axum::serve` 但未声明依赖 → 收进 `dinfer_server::serve`；③ `BlockPool::allocate` 返回顺序与测试期望相反 → `split_off` 后 `reverse()`。k8s 第二节点 `host-10-121-192-98` 亦确认 8 × 4090。基线 commit 完成。
- 同日下午：用户确认先做 sm8x 可完成部分；写出 P0 骨架：根 `Cargo.toml`（workspace，固定版本依赖，cudarc 0.18.2 / axum 0.8 / tokio 1.45）、`crates/{core,kernels,layers,server,kimik3,glm53}`、`bins/dinfer`（feature `kimik3|glm53` 互斥，`no-nvcc` 回退）、`crates/kernels/cuda/add.cu` + `build.rs`（nvcc，`CUDA_ARCHS` 默认 86,89,90）、core 的 config/request/sched(最小 FCFS)/kv(BlockPool, KvCacheSpec)/sample、两模型 `config.rs`（解析官方 config.json 并断言结构事实）、`scripts/verify/p0.sh`、`scripts/dev/k8s/{gpu-probe,dev-pod}.yaml`。全部未编译。
- 未完成：用户确认计划；基线 commit；P0。（CLAUDE.md"项目是什么"一节当时被工具拦截，已于 10-14 补上。）
- 踩坑：见 `docs/pitfalls.md` 2026-10-13 两条。

### 2026-10-18（晚，Q5 多卡）

- 用户：多卡 TP、多 DP 实现，并考虑 MoE / DCP / PCP 设计。
- 设计 `design.md` §13：与 vLLM 差异 = 一进程多线程（无 shm 队列 / 序列化 / coordinator 进程）；TP 切分与通信点照搬 vLLM（每层 2 all-reduce、每 rank 都采样、lm_head 复制）；DP 同进程原子计数路由；EP / DCP / PCP 设计 + 分阶段 T1–T5；§11 台账加 3 行。
- 代码：见交接区。编译通过（`cargo build -p dinfer --features qwen3`），`cargo test -p dinfer-core -p dinfer-server` 49 过。
- 4090：Pod 无 NCCL（pip 装）；`q5_tp.sh 2` GPU 3/4 通过；bench tp2 / dp2 进 CSV（数字见交接区 / loop 第 3 轮）。bench_trend.py 支持 `bench_<tag>` 目录名。NCCL：Pod 内 `/usr/lib/x86_64-linux-gnu/libnccl.so.2.30`，8 卡无 P2P（topo 全 NODE/SYS）。

## 2026-10-20（Qwen3.5 perf 第 5 轮 B 线，子代理）

- decode 非桶 M GEMM：自调优表展开 1..=最大桶 的每个 m（借幂桶 plan + `cublasLtMatmulAlgoCheck`），默认 algo 按形状缓存；3090 真形状 M=7/8 `gdn.in_proj_a` 默认 31 µs → 8 µs，其它形状 4–17%。候选数 8→32 无收益（不改）。
- 新指标 `engine_step_execute_{prefill,decode,mixed}_seconds`：tiny 全词表模型 c32 混批步 4.6%、每步 2.6× decode；结论 pair 跨流不优先。详见 `docs/loops/qwen35-perf.md` 5B 行。
- 验证：golden 4/4（含 `GEMM_TUNE=off`）、ops_parity 95/95、core 55 单测、workspace check qwen3 / qwen35。**待 B300**：bench_b300_10 看 `borrowed=` 日志 + 三个 mode 直方图 + c32 / c8-long TPOT。**改口径后 B300 复跑 exit 0（03:07，`/tmp/k3_model_full.log`，产物在远端 `test/kimik3/2026-10-20/model/` 24 MB 待 pull）**：隔离口径 L0–L4 cos 0.999983–0.999991（L3 MLA+MoE 0.999989）；端到端 L0 0.999985 → L4 0.999269、logits 0.999897、decode-vs-prefill 0.999970。子代理因后台任务完成通知丢失空等 4.5 h（07:3x 已叫醒收尾：pull 产物 + 填 `k3-model.md` 表 + commit）。**教训**：子代理等后台任务要给超时并主动查产物时间戳，不能只等通知。 

## 2026-10-23（Kaon-V3 parity，子 loop D 第 18 轮）

- 上轮"plan_for 优先级"诊断错误（kaon3 不调 autotune，router 走自写 kernel）→ 撤回 SMALL_M_LT；router 对齐 vLLM BI cuBLASLt 口径。
- 逐个子模块归零：L2（router / post_ff_norm / v_norm / attn_out）→ L3/L4（RMSNorm 按 SASS、attention 自然域指数）→
  L7（路由权重融合版口径）→ L11（D=512 慢路径、fp8 o_proj algo）。p02 整跑 **L0–L29 逐位 0**（对 eager vLLM）。
- 30 用例端到端无改善（16.56% / 1.1994）：定位为参照口径——服务实例是 compile 模式，与 eager 自身差 1.57（§58）。
- 新知识页 `docs/knowledge/bitexact-replication.md`；踩坑 +3（pitfalls 2026-10-23 三条）；记录 `test/kaon3/2026-10-23/D-16-REBASE.md` §51–§58。


## 2026-10-24（Kaon-V3 compile 口径 parity，17 轮 loop 到上限）

**做了什么**：按用户定的「全程 compile 口径、eager 不作判据」建立了 `docs/loops/kaon3-compile.md`
与 `scripts/verify/kaon3-compile.sh`（含编译门禁），跑了 17 轮。

**结果**：flip **16.56% → 14.08%**，|Δlp| 1.1994 → 1.2847（两个指标不同向）。
唯一产生质变的改动是 `MOE_COMBINE_F32` 默认值 `0 → 1`（p02 单例 top-1 翻转 4/22 → 0/22）。
离 FLOOR（0.83% / 0.0276）仍差 17 倍。

**逐位达成的部分**：router 全段 0/3072（含 top-8 集合）、gelu 0/50688、
`router_gemm_bi_f32` 0/8192、q 的 fp8 量化已做。

**卡点**：专家 NVFP4 dense GEMM 差 7.8%（`ndiff=11210/11264`），用参照自己的 `a_q/a_s` 也一样。
参照走 `VLLM_CUTLASS` 的 FP4 融合 MoE kernel，我们逐专家 GEMM；`cutlass_fp4_moe_mm` 是 torch op，
不能从 Rust 直接链接。详见交接区与 `docs/loops/kaon3-compile.md` 的卡点节。

**新增工具**（`scripts/dev/`）：`moe_capture.py`（抓参照真实 MoE 输入输出）、
`fp4_dense_probe.py` / `fp4_calib.py` / `fp4_quant_calib.py`（FP4 定标）、
`which_moe.py`（查参照 MoE 后端）、`cutlass_moe_probe.py`、`read_scale.py`。

**踩坑 2 条**（`docs/pitfalls.md` 顶部）：
用 `torch.matmul` 当对照测出假的 43% 差异（白写一个 kernel）；
定标只在 k=64 做过一次就当通用公式（布局 bug 多绕 3 轮）。

**下一次接手第一件事**：核对 `fp4_dense_probe.py` 里 `a` 与导出的 `a_q/a_s` 是否同一次调用产物。
