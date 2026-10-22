# kaon3-parity 迭代记录（2026-10-22）

规格：`docs/loops/kaon3-parity.md`；脚本：`bash scripts/verify/kaon3-parity.sh`。指标 = 可判定行 top1 不一致数 / top5 |Δlp| 均值 / top1 |Δlp| 最大。

| 轮 | 改动（一句话） | bos1 | raw9 | chat_en | chat_zh | 结论 / 下一步 |
|---|---|---|---|---|---|---|
| 0 | 基线（RoPE span 已修，MAX_D=512） | 1 / 13.2 / 42.4 | 6/6 / 19.7 / 42.4 | 20/23 / 14.8 / 42.4 | 17/20 / 13.8 / 42.4 | row 0（T=1）就错 → 逐 token 路径 |
| 1 | MoE 支路：专家输入补 `pre_ff_norm_2`；相加用 `post_ff_norm_2` 之后的结果（原来加的是未归一化的 `sc.y`） | 0 / 1.23 / 0.009 | 2/6 / 2.42 / 11.4 | 10/23 / 4.75 / 34.2 | 6/20 / 2.55 / 6.06 | row 0 对上；多 token 行仍错 → attention / RoPE / KV |
| A/B | `GQA_PREFILL_CFG` 默认 / 0（v5）/ −1（新加：强制通用慢路径） | — | 2/6 · 2/6 · 2/6 | 10 · 10 · 11 | — | 三者几乎一致 → attention kernel 不是主因 |
| 自洽 | `dup4`（4 个相同 token：V 无 RoPE，各行应完全相同） | — | — | — | — | 我们 4 行完全一致；oracle 行间差 ≤0.02 → 多行路径无整体错位 |
| 2 | FP8 KV 模拟：K/V 写 cache / 做 attention 前按 e4m3（饱和、scale 1.0）往返；vLLM 实测 `kv_cache_dtype=fp8_e4m3`、无 k/v scale | 0 / 1.23 / 0.009 | 4/6 / 2.39 / 9.1 | 10/23 / 4.81 / 33.6 | 6/20 / 3.56 / 20.6 | 无改善（保留：是 vLLM 真实口径）；long_zh 222/1024 不一致 → 下一步截层二分 |
| 3 | FP8 静态激活 scale（`gemm_fp8_static_bf16`：x/input_scale 再转 e4m3，输出乘 a·w scale；实测 input_scale 0.03–3.9，>1 时原实现截断 448）+ 合并组（qkv / gate_up）按 vLLM `requantize_with_max_scale` 重量化 | 0 / 0.95 / 0.013 | 2/6 / 3.06 / 9.7 | 10/23 / 5.15 / 32.2 | — | 单 token 改善；多 token 不动 |
| 诊断 | `--dump-oracle`：我们 v5 对我们慢路径 | — | 1/9 / 1.12 / 1.88 | 6/21 / 2.46 / 20.1 | 4/19 / 3.27 / 23.0 | 同配置两次完全一致（确定）；两个正确 kernel 互差 20 nat → 模型对 attention 舍入级差异极敏感 |
| 4 | kernel 测试 `gqa_window_parity` 抓到滑窗 bug：tile0 按 warp 算（K/V 是 block 协作装的）、下界 mask 只在 need_mask 时做、v6 首 tile 固定装 0 → 三处修复，v5/v6/v6B/v6F/v7 全过 | | | | | 窗口生效时原误差 5.9 → 0.0087 |
| 标定 | vLLM 关 BI vs 开 BI（同镜像私有副本）：raw9 3/6、chat_en 5/23、chat_zh 2/20、long_zh 116/1024（11.3%），共有 token Δ 0.59–1.79 | | | | | **原判据连 vLLM 自己都过不了** → 判据改为相对 vLLM 自身噪声的统计量 |
| 5 | NVFP4 激活全局 scale：照抄 vLLM `nvfp4_utils.cuh`（gs=1/input_scale、块 scale e4m3(gs·amax·rcp6)、rcp.approx）；专家 gate/up 共用 max(input_scale) | | 2/6 · 0.88 | 9/23 · 2.06 | 6/20 · 1.81 | long_zh 205→277/1024（单条长序列行间强相关，统计不可靠） |
| 测试 | `gemm_lowp_parity`：FP8 静态 scale 3 档 + NVFP4 gs∈{1,25,900}、K∈{2816(BK128),704(BK64)} 全过（相对误差 ≤3.9e-3）；首跑失败是用例没把 1/gs 折进 global | | | | | 交接区第 ② 步（NVFP4 对拍）完成 |

### 新判据（相对 vLLM 自身噪声）下的轮次

| 轮 | 改动 | 短 prompt 汇总（翻转率 / 共有 Δ / ΔNLL） | long_zh ΔNLL（下限 −0.08） | 结论 |
|---|---|---|---|---|
| 5' | 第 5 轮状态，新判据首测（下限：13.95% / 1.19 / −0.25） | 32.72% / 1.82 / −0.41 | **+0.72**（分段：开头 +1.08 → 1024 后 +0.32） | 不是滑窗；全位置系统性差异，开头最大 |
| 6 | **Q 也做 e4m3 往返**（vLLM Triton `supports_quant_query_input` + FP8 KV → `query_quant`，q_scale = k_scale = 1.0，饱和 448） | （快测 5 例）| **+0.097** ✅ | 大幅改善；下一步 P→e4m3（Triton `tl.dot(P.to(V.dtype), V)`，V 为 fp8） |
| 7 | **P 以 e4m3 参与 PV**（Triton `tl.dot(P.to(V.dtype), V)`，V 为 fp8；行和仍 fp32），进程级开关 `gqa_set_p_fp8` | 25.25% / 1.80 / −0.86 | **+0.048 ✅** | 长序列 / 单 token / kernel 三项过；短 prompt 翻转与共有 Δ 仍超；短 prompt ΔNLL 由模板 / 用户消息里 lp≈−20 的离群 token 主导，见下一步 |
| 8 | RoPE 表舍入到 bf16（vLLM `cos_sin_cache.to(dtype)`）+ embed normalizer 用 bf16（53.0）；同时 verify 改为批量 + 6 卡分片 | 28.24% / 1.77 ✅ / −0.79 | **+0.475 ❌**（第 7 轮 +0.048） | 变差；A/B：两项都退回 / 只退回 RoPE 表，用同一快照在不相交卡上并行跑（用户：并行时编译不要互相覆盖） |

### P1 参数对齐：疑似"参照集本身不干净"（2026-10-22 深夜，进行中）

用显式参数（block 32 / max-num-batched 4096 / max_model_len 16384 / BI=1）重启 `kaon-oracle`，
重抓 29 个用例到 `test/kaon3/oracle_recheck/`，对旧 `test/kaon3/oracle/` 做 diff：

```
POOLED cases=29 rows=655 decisive=607 flips=61 flip_pct=10.05 mean_common_d=0.7611 nll_a=13.3821 nll_b=13.4256 dnll=+0.0435
```

**逐用例看，差异不是均匀的，而是按抓取时间分成两组**：

| 组 | 用例 | 翻转率 | 抓取时刻（`oracle/*.json` mtime） |
|---|---|---|---|
| A | `bos1` `raw9` `chat_en` `chat_zh` `dup4` | **全部 0.00** | 07:44–08:05 |
| B | `p01`–`p24` | 4.2%–21.7%（多数 10–16%）| 09:40–09:41 |

**已排除的假设**：BI=1 不是确定性问题——同一进程连抓两次（`oracle_recheck` vs `oracle_recheck2`，p01/p02/p10/chat_en）：
`flips=0 mean_common_d=0.0000 dnll=+0.0000`，逐位完全一致。

**推断**：参照集是两个不同进程的产物。A 组来自 07:4x 的 oracle，B 组（p01–p24）来自 09:40 的另一个 oracle
（`86ec5e9` 前后在调 oracle 多实例参数化）。若如此，B 组的旧文件与现在这个 BI=1 进程本来就不是同一配置，
**10.05% 里混着"参照集内部不一致"，不能直接当作我们的误差**。

**决定性实验（进行中）**：用**旧参数**（不带 `--max-num-batched-tokens`，其余同）另起 `kaon-oracle-old`，
抓 p01 后分别与 ① 旧 `oracle/p01.json` ② 新 `oracle_recheck/p01.json` 比：
- 若 ≈旧文件、≠新文件 → 差异来自 `max-num-batched-tokens` 这一项（隐式 2496 vs 显式 4096 改变了 batch 分派）；
- 若 ≈新文件、≠旧文件 → B 组旧文件确实来自第三个配置，参照集必须整体重抓。

**这条不查清，P2–P4 的所有数字都没有意义**（判据的下限和我们的读数会来自不同世界的样本）。

**三方对比结果（2026-10-22 深夜）**，p01，全部 BI=1、block 32：

| A | B | flips | flip_pct | mean_common_d | dnll |
|---|---|---|---|---|---|
| old-params 新进程 | 旧 `oracle/p01` | 2/19 | 10.53% | 1.2236 | −0.1354 |
| old-params 新进程 | 新 `oracle_recheck/p01`（4096）| 3/19 | 15.79% | 1.1366 | −0.2480 |
| `oracle_recheck` | `oracle_recheck2`（**同进程**）| **0**/89 | **0.00%** | **0.0000** | **0.0000** |

**关键推论**：旧参数的新进程与旧文件也差 10.53% → **`max-num-batched-tokens` 不是主因**；
同进程逐位一致、跨进程必然不同 → 差异是**进程级别的**，不是参数级别的。

怀疑方向（按先验）：① CUDA graph 捕获时的 buffer 地址 / 分桶不同；② torch.compile 缓存命中与否
（日志里 `torch_compile_cache/95a7cdd52e/...`，若某些进程重新编译会走不同 kernel）；③ 其它按进程随机的
自动调优（cuBLAS workspace、cutlass heuristics、FlashInfer autotune 的 benchmark 选择）。

**这意味着什么（重要）**：`VLLM_BATCH_INVARIANT=1` 只保证**同一进程内** batch 组成不影响结果，
**不保证跨进程一致**。所以判据里"BI=1 是确定性参照"这个说法要修正为"**同一进程内确定**"。
而 `oracle/` 目录是多个进程拼成的，内部就不自洽 —— 必须用**单个进程一次抓完**的参照集。

**决定性实验 2（进行中）**：同参数再起一个进程（`kaon-oracle-dup`，GPU 0 :28101）抓 p01，
若与 `oracle_recheck` 不同 → 坐实"跨进程不确定"，参照集与下限集都必须单进程一次抓完。

### P1 结论（2026-10-22 深夜）：**跨进程不确定是判据的根本障碍，不是参数问题**

决定性实验 2 结果：`kaon-oracle-dup`（GPU 0 :28101）与 `kaon-oracle`（GPU 1 :28100）**参数完全相同**
（同镜像、同 checkpoint、BI=1、block 32、max-num-batched 4096、max_model_len 16384），只是两个进程：

```
POOLED cases=4 rows=100 decisive=89 flips=16 flip_pct=17.98 mean_common_d=1.2362 dnll=+0.1303
  p01 15.79% / p02 21.74% / p10 16.67% / chat_en 17.39%
```

对照：**同一进程**连抓两次 flips=0 / Δ=0.0000。

**修正此前所有判断**：

1. 「BI=1 是确定性参照」**错**。正确表述：`VLLM_BATCH_INVARIANT=1` 保证*同一进程内* batch 组成不影响结果，
   **不保证跨进程一致**。跨进程翻转率 ~18%，比 BI 开/关的 11–22% 是同一个量级。
2. 我上一轮把它归因为「CUDA graph 桶 / torch.compile 缓存」是**猜测，未验证**，不作为结论。
3. `oracle/` 目录由 07:4x 与 09:40 两个进程拼成 → 参照集内部就不自洽（这解释了 A 组 0%、B 组 10–16% 的分布）。
4. 这不影响 pre2 的判断：~18% 的进程级噪声下 vLLM 仍**强判** 1852 @0.986（top1−top2 ≈ 1.2 的间隔），
   而我们判 5976 → pre2 仍是真 bug，且比原先认为的更显著。

**判据必须改（P1 的真正产出）**：

| 旧 | 新 |
|---|---|
| FLOOR = vLLM(BI=1) vs vLLM(BI=0) | **FLOOR = 同一参数的两个独立进程**（如 `kaon-oracle` vs `kaon-oracle-dup`），这才是真实可达下限 |
| OURS = 单进程 BI=1 vs 我们 | OURS 保持不变；但参照集必须**单进程一次抓完** |
| 参照集 = `oracle/`（多进程拼接）| 重抓一个 `oracle_p1/`：一个进程、一次抓完、29 个短用例 + long_zh |

**下一步（P1 收尾，待执行）**：单进程一次抓完整套 → `test/kaon3/oracle_p1/`（参照）与 `oracle_p1_b/`（同参数第二进程，定下限），
再改 verify 的 FLOOR/OURS 行，然后跑第 7 轮状态拿新基线。**在此之前 P2–P4 的任何数字都不可比。**

### P1 完成（2026-10-22 深夜）：真实下限 = 跨进程 15.04%，旧判据的"BI 噪声"大部分是它

单进程一次抓完 30 个用例（含 long_zh），两个**参数完全相同**的进程各抓一套：

| 集 | 容器 | 用途 |
|---|---|---|
| `test/kaon3/oracle_p1/` | `kaon-oracle`（GPU 1 :28100，BI=1，block 32，4096）| 判定参照 REF |
| `test/kaon3/oracle_p1b/` | `kaon-oracle-dup`（GPU 0 :28101，**同参数**）| 真实下限 FLOOR |

**下限（FLOOR = p1b vs p1，两个进程同一配置）**：

```
POOLED cases=30 rows=655 decisive=605 flips=91 flip_pct=15.04 mean_common_d=1.1276 dnll=+0.0728
LONG long_zh  rows=1252 decisive=1044 flips=182 flip_pct=17.43 mean_common_d=0.7068 dnll=-0.0974
```

**与旧下限（BI 开 vs 关）对比**：13.95% / 1.19 / −0.25 → 15.04% / 1.128 / −0.097。
**结论：旧"BI 噪声"下限里绝大部分就是跨进程噪声**，BI kernel 变体本身贡献很小。
这解释了第 5–8 轮为什么一直在 ±0.1–0.7 nat 之间反复：判据一直在拿我们跟一个自身 15% 噪声的参照比。

**判据修正（写进规格）**：
- FLOOR 用本表数字（15.04% / 1.128 / |ΔNLL| 0.097），不再用 BI 开/关之差；
- OURS 必须显著优于 FLOOR 才算通过——**我们 vs REF 的差值 ≤ FLOOR 自身差值**时，说明我们与 vLLM 的差距
  不超过 vLLM 与自己另一个进程的差距，这是能做到的最强判据；
- 原公式 `1.5×FLOOR + 3pp` 保留但用新 FLOOR 重算：翻转率线 = 1.5×15.04+3 = **25.6%**，
  共有 Δ 线 = 1.5×1.128 = **1.69**，长文本 |ΔNLL| 线 = max(3×0.097, 0.01) = **0.29**。

**注意**：判据变松了（第 7 轮 25.25% / 1.80 / +0.048 在新线下翻转与共有 Δ 都刚好压在线上，ΔNLL 大幅通过）。
这不是"放宽标准凑通过"，而是**原来那条线本身就不合理**——下限被低估了一半。真正要盯的是 pre2 那类
远超噪声的定性错误（vLLM 强判 1852 间隔 1.2，我们判 5976）。

### ⚠️ 上一节的结论作废（2026-10-22 深夜，端口抢占）

**`kaon-oracle-old` 与 `kaon-oracle-dup` 都 `Exited (1)`**，日志是 `OSError: [Errno 98] Address already in use`。
两者都用了 `--network host` + 端口 28101，而 `kaon-oracle-nobi`（BI=0）已经占着 28101。

因此：

| 我以为在比 | 实际比的是 | 真实含义 |
|---|---|---|
| `oracle_recheck` vs `oracle_dup`（"同参数两进程"） | BI=1(28100) vs **BI=0(28101)** | 跨进程噪声 + BI 变体，**不是纯跨进程** |
| `oracle_oldp`（"旧参数进程"） | 实际打到 **nobi（BI=0）** | 同样作废 |
| `oracle_p1b`（"FLOOR 第二进程"） | 实际打到 **nobi（BI=0）** | FLOOR = BI 开 vs 关，**不是跨进程** |

**仍然成立的**：同一进程连抓两次 `flips=0 / Δ=0.0000`（`oracle_recheck` vs `oracle_recheck2`，都是 28100 的 BI=1 进程）——
BI=1 进程内确定这一条没被推翻。

**作废的**：① 「跨进程不确定 ~18%」的定量结论；② 「旧 BI 下限里大部分是跨进程噪声」的推断；
③ 由此改的判据线（25.6% / 1.69 / 0.29）**暂时回退**，等真正的跨进程实验。

**根因**：`kaon3_oracle_server.sh` 的健康检查只看 `curl :PORT/v1/models == 200`。
端口被别人占着时，被占的那一方返回 200，脚本就打印"就绪" —— 假就绪。
**修法**（同时在做）：就绪前校验本容器 `State.Running == true`，否则 exit 3 并提示端口被占。

**教训（写进 pitfalls）**：`--network host` 的多实例必须**每实例独占端口**，且"服务能答"≠"我的容器在跑"。

### P1 真正完成（2026-10-22 深夜，端口独占后重做）

`kaon-oracle-b`（GPU 0 :28102，BI=1，block 32，4096，**独立端口**）单进程一次抓完 30 例 → `test/kaon3/oracle_p1c/`。

**真实跨进程下限（`oracle_p1c` vs `oracle_p1`，同参数两进程）**：

```
POOLED cases=30 rows=655 decisive=598 flips=9 flip_pct=1.51 mean_common_d=0.1000 dnll=+0.0155
LONG long_zh  rows=1252 decisive=1027 flips=71 flip_pct=6.91 mean_common_d=0.4574 dnll=-0.0010
```

**与那次作废的数字对比**：

| | 翻转率 | 共有 Δ | long_zh ΔNLL | 说明 |
|---|---|---|---|---|
| 真正的跨进程（本次） | **1.51%** | **0.100** | **−0.001** | 同参数、两进程、端口独占 |
| 作废那次（"15.04%"） | 15.04% | 1.128 | −0.097 | **实际是 BI=1 vs BI=0** |
| 旧判据下限（BI 开 vs 关）| 13.95% | 1.19 | −0.25 | 同一件事 |

**关键结论：BI=1 的跨进程复现性很好（1.51%），真正的大噪声来自 BI 开/关这个 kernel 变体。**
所以之前那句"旧 BI 下限里大部分是跨进程噪声"是**错的**——方向正好相反：跨进程只贡献 1.5%，
剩下 12+ 个百分点是 BI 关掉后 kernel 换成变体造成的。

**逐用例看，跨进程差异集中在少数用例**（其余 21 个用例逐位完全一致）：
`long_zh` 1228/1252 行浮点不同但翻转仅 6.91%；短用例里 `p02` 12/24、`p04` 12/24、`p08` 27/36、`p11` 15/24
的**行内容变了**（说明这些用例的 batch 分派或 kernel 路径在两进程间不同），`p01`/`p10`/`chat_en` 等逐位一致。

**判据线（用真实下限重算）**：
- 翻转率 ≤ 1.5 × 1.51 + 3 = **5.27%**（比旧线 25.6% 紧 5 倍）
- 共有 token |Δlp| ≤ 1.5 × 0.100 = **0.150**
- 长文本 |ΔNLL| ≤ max(3 × 0.001, 0.01) = **0.01**

**这意味着第 7 轮状态（25.25% / 1.80 / +0.048）在新线下是明显超标**，不是"刚好压线"。
之前那次因为下限被高估 10 倍，判据被放松了，掩盖了真实差距。**P4（短 prompt 超标）必须做。**

### P1 收官：新判据下第 7 轮状态的实测基线（run `1022-134436-502964`）

环境：GPU 0/2–6 各 812 MiB 全空（先清掉 5 个截层 oracle + 本机 oracle；GPU 7 基线不动）。
REF=`oracle_p1`，FLOOR=`oracle_p1c`（两个同参数独立进程，各单进程一次抓完 30 例）。

| 项 | FLOOR（vLLM vs vLLM） | OURS（我们 vs vLLM） | 判定 |
|---|---|---|---|
| 可判定行 top1 翻转率 | 1.87% | **28.87%** | ❌ 线 = 1.5×1.87+3 = **5.81%** |
| 共有 token \|Δlp\| | 0.1227 | **1.8250** | ❌ 线 = 1.5×0.1227 = **0.184** |
| 短 prompt ΔNLL（参考） | +0.0191 | −0.4983 | （不判定）|
| long_zh ΔNLL | −0.0010 | **+0.0646** | ❌ 线 = max(3×0.001, 0.01) = **0.01** |
| long_zh 翻转率 | 6.91% | 19.79% | （参考）|
| 单 token（bos1，无 attention）| — | top1 一致 | ✅ |
| kernel 对拍 | — | gqa_window / gemm_lowp 全过 | ✅ |

**通过 4 项 / 失败 3 项。**

**读法（对 P2–P4 的含义）**：

1. 翻转率 28.87% 是下限的 **15 倍**，共有 Δ 是下限的 **15 倍** —— 这不是"压线"，是明确的超限。
   旧判据（下限被高估到 13.95%/1.19）把这条差距盖了 7 轮。
2. long_zh ΔNLL +0.0646 相对旧口径的 +0.048 基本一致（参照换成 `oracle_p1` 后的正常偏移），
   说明**长序列的真实质量差距一直在 +0.05~0.07**，没变好过；新线 0.01 下还差 6 倍。
3. 单 token 通过 → embed / norm / proj / MLP / MoE / lm_head / softcap 这一串在 T=1 下是对的，
   **问题集中在多 token 路径**（attention 或逐位置累积）。
4. 短 prompt ΔNLL 我们反而更低（−0.4983），与之前"由 chat 模板远尾 token 主导"的记录一致，
   继续只作参考。

**下一步直接进 P2'（合并 P2/P3/P4 的优先级）**：
① `pre2` 截层二分（已知 vLLM 强判 1852 @0.986，是最干净的定性错误，且 oracle_layers 已抓好）；
② 二分定位后再看短 prompt 的 28.87% 是否同源；③ embed normalizer 的 bf16/f32 A/B 留到最后（影响面最小）。

### P3 pre2 截层二分（2026-10-22 深夜，进行中）

用例 `pre2 = [2, 105]`（2 token，只有 row1 可判）。截层 oracle 在 `test/kaon3/oracle_layers/L{3,6,12,18,24}/pre2.json`，
探针 `--layers N --oracle <同层 oracle>`。

**vLLM 侧逐层 top-3（row1）**：

| 层 | top-3 (logprob) | 说明 |
|---|---|---|
| L3 | 105 / 17 / 1（全 −11.12）| 完全平坦 |
| L6 | 220 / 51 / 223（全 −9.77）| 完全平坦 |
| L12 | **237323 (−2.12) / 1852 (−2.37)** / 238215 (−2.62) | 开始出现强候选，1852 rank2 |
| L18 | **25407 (−0.11)** / 17218 (−3.61) / 1436 (−5.61) | 25407 主导，1852 掉出 top-3 |
| L24 | **25407 (−1.19)** / 537 (−3.94) / 23621 (−4.07) | 25407 仍主导 |

→ **决定性变化在 12–18 层之间**（1852 在 L12 还是 −2.37 的有力候选，L18 已被 25407 压到 −3.61 以外）。
18 层与 24 层的 top-1 一致（25407），说明 **19–29 层不再改变判决**。

**我们侧**：

| 层 | 结果 |
|---|---|
| L3 | ✅ PASS（可判定 0 行；top5 \|Δlp\| 0.0969，top1 最大 0.1535）—— L3 前向与 vLLM 一致 |
| L6 / L12 / L18 / L24 | 进行中（每层权重加载 ~33 s）|

**注意**：截层前 3 层就 PASS，说明**前 3 层（含 0–2 层滑窗 attention）没问题**，
问题在 3 层之后的某个组件。二分继续。

**二分结果（前 12 层全部 PASS）**：

| 层 | 我们 vs vLLM | top5 \|Δlp\| | top1 最大 | 判定 |
|---|---|---|---|---|
| L3 | 一致（0 可判定行）| 0.0969 | 0.1535 | ✅ PASS |
| L6 | 一致（0 可判定行）| 0.0784 | 0.1121 | ✅ PASS |
| L12 | 一致（0 可判定行，top1 都是 237323）| 0.0536 | 0.0680 | ✅ PASS |

**这覆盖了**：L0–L2（滑窗 attention）、L3（dense MLP / MoE）、L4（滑窗）、**L5（第一个 full attention，D=512）**、
L6–L11（滑窗 + MoE）。**结论：12 层之前的所有组件（含 D=512 全注意力与 MoE）都是对的。**

分歧必定在 **L12–L17** 之间（vLLM 侧 1852 在 L12 是 −2.37 的 rank2，L18 已被 25407 压掉）。
下一步：L13/L14/L15/L16/L17 逐层抓 oracle + 探针，把区间从 6 层收到 1 层。

**分界锁定（L12 PASS → L18 FAIL）**：

| 层 | top5 \|Δlp\| | top1 最大 | 可判定行 top1 | 判定 |
|---|---|---|---|---|
| L12 | 0.0536 | 0.0680 | 0 不一致 | ✅ PASS |
| **L18** | **0.3319** | 0.5520 | 0 不一致 | **❌ FAIL（6.2× 于 L12）** |

**关键细节**：L18 上 top1 仍然一致（都是 25407），错的是**整条 top-5 分布**（均值 0.33，单行最大 0.55）。
这说明不是"选错 token"这类硬错误，而是**数值偏移**——到了 L18 累积到 0.33 nat，再往后 6 层
（L18→L24→L29）继续放大，最终在完整 30 层上把 1852 压成了 5976。
**这与 verify 的结论一致**：翻转率 28.87% 但单 token 通过 —— 每步都在偏一点，多步后跨过判决边界。

**待收窄区间：L13–L18（6 层）**。缩到 1 层就能定位到具体组件。
权重加载开销是主要成本（18 层 ~50 s + 前向），下一轮跑 L14 / L16 先做二分，再回到 1 层。

**L24 结果与一个必须正视的问题**：

| 层 | top5 \|Δlp\| | 判定 |
|---|---|---|
| L12 | 0.0536 | ✅ PASS |
| L18 | 0.3319 | ❌ FAIL |
| L24 | 0.1051 | ✅ PASS |

**非单调**（12 过 → 18 挂 → 24 又过）。两种解释：

1. **L18 那个截层 oracle 自身有偏差**（每个截层 oracle 都是**独立进程**，跨进程下限是 0.100；
   L24 的 0.1051 正好压在下限上，L18 的 0.3319 是下限的 3.3 倍）。
2. pre2 只有 2 行、最多 2 个可判定行，**统计量太小**，`mean_common_d` 这种均值极不稳定。

**这动摇了"用 pre2 做单层二分"的方法本身**。pre2 是当初用来抓硬判决错误的（vLLM 强判 1852 间隔 1.2），
它能证明"全 30 层确实错了"，但**不足以定位到某一层**——2 行样本 + 每层一个独立 oracle 进程的噪声
（0.1 量级）会把真实信号淹掉。

**方法论修正（下一步照这个做）**：
- 二分改用**大样本用例**（`p01`/`p02`/`chat_en` 等 22–25 行的短 prompt），不用 pre2；
- 每个截层层级都要**同时**抓两个独立 oracle 进程，用它自己的跨进程差作该层的判定线，
  而不是拿全 30 层算出的全局 1.51%/0.100 去套所有层；
- 或者更省：**不做逐层**，改为"组件级"定位 —— 直接对 L13–L17 的**子组件**做单元级比对
  （滑窗 attention 已有 kernel 对拍过；剩下 MoE 路由 / dense MLP / norm / RoPE 位置编码）。

### 排除"验证脚本批量模式有 bug"（2026-10-22 深夜）

怀疑：`--batch` 只 `setup()` 一次、在同一个 ctx 上连跑 30 次 forward，可能与单用例模式（每次全新 ctx）结果不同，
从而让 verify 的 28.87% / 1.825 成为脚本产物。

**实验**：同一用例 `p02`（24 token）分别用 `--batch`（单用例目录）与 `--forward --dump-oracle` 跑，
逐行比对 `prompt_logprobs`：

```
rows 24 24
identical rows: 24 / 24
```

**结论：批量模式与单用例模式逐位一致，脚本无此 bug。** verify 的数字不是批量模式的产物。
（`forward()` 里每次 `alloc_kv` 重建 KV、`LayerScratch`/`ActBuf` 每次新建，状态确实是干净的。）

**但这只排除了"模式差异"**，还没排除"两者的输入假设都不对"，例如：
`BLOCK_SIZE=32` 与 oracle 的 block 32 是否真的一致、`slot_mapping` 的排布、`block_table` 的构造。
这些由 `p02`（24 token，1 个 block）覆盖不足 —— 需要跨 block 的用例才能暴露。

### 【真 bug】MoE 路由用错语义：sigmoid vs Gemma4 的 softmax（2026-10-22 深夜）

**逐行读 vLLM `model_executor/models/gemma4.py` 得到的事实**（不是推测）：

`Gemma4Router.forward`（271–277 行）：
```python
x = self.norm(x)                      # RMSNorm，has_weight=False
x = x * self.root_size.to(x.dtype)    # hidden_size ** -0.5
x = x * self.scale.to(x.dtype)        # 每维学习 scale
router_logits, _ = self.proj(x)       # GateLinear → fp32
```

`gemma4_routing_function_torch`（186–212 行）：
```python
_, topk_ids = torch.topk(gating_output, k=topk, dim=-1)          # 按**原始 logits**
router_probabilities = torch.nn.functional.softmax(gating_output, dim=-1)   # **全专家 softmax**
gate_weights = one_hot(topk_ids, E).sum(-2) * router_probabilities
renorm_factor = gate_weights.sum(-1, keepdim=True)
dispatch_weights = gate_weights / renorm_factor                  # 只在 top-k 上归一化
topk_weights = dispatch_weights.gather(1, topk_ids)
expert_scales = per_expert_scale[topk_ids]
topk_weights = topk_weights * expert_scales                      # 乘 per_expert_scale
```

**我们的实现（`crates/kernels/cuda/moe.cu` 的 `route_kernel`）**：
```c
const float sc = 1.f / (1.f + __expf(-logits[t * E + e]));   // ← sigmoid！
```
该 kernel 头注释写明它是 `DeepSeekV3TopkRouter / vLLM grouped_topk` 的口径 —— 那是 GLM/K3 用的，
**不是 Gemma4 的口径**。两者完全不同：sigmoid 逐元素独立，softmax 归一化竞争。

**修法**：新增 `moe_route_softmax_f32`（`cuda/moe.cu`），严格照上面五个步骤；`layer.rs` 的 MoE 支路改用它。
原 `moe_route_f32` 保留给 GLM/K3。

**`per_expert_scale` 折进 down 的 global scale 是等价的**（down 是线性），不是 bug，保持原样。

编译通过。下一步跑 verify 看指标变化。

### 修复效果（run `1022-142936-540047`）：翻转 28.87% → 15.26%

| 指标 | 修复前 | 修复后 | 变化 |
|---|---|---|---|
| 短 prompt 翻转率 | 28.87% | **15.26%** | −47% |
| 短 prompt 共有 Δ | 1.8250 | **1.2303** | −33% |
| long_zh 翻转率 | 19.79% | **12.36%** | −38% |
| long_zh ΔNLL | +0.0646 | **−0.0454** | 绝对值 −30% |
| 单 token / kernel 对拍 | 通过 | 通过 | 不变 |

**仍超限**（线：5.81% / 0.184 / 0.01）。所以 sigmoid→softmax 是**主因之一，不是全部**。
下一步继续逐行读 vLLM 找剩余差异。

## 第 8 轮：判据被"阈值"放水，0–7 层要重判（2026-10-22 深夜，用户指出）

用户 2026-10-22：「我觉的 0–7 层可能也是不对的，只是我们的标准定的让他过」。**成立**。

干净参照下**跨进程下限是 0.0000（逐位一致）**，而 L8 的三个用例分别是
`0.0613 / 0.0449 / 0.0574`。这不是噪声，是确凿偏差；只是被"阈值 0.15"和
"1.5×FLOOR"这类**以噪声为基准的相对判据**放过了。下限为 0 时，相对判据退化成"任意非零都过"。

### 判据修正（用户要求）

- 分层二分一律用**绝对判据**：跨进程下限 == 0（已验证）→ 探针与 oracle 的
  top5 |Δlp| 均值、ΔNLL 都按**"要求 0"**判，不再用 0.15 这类阈值。
- 端到端（30 层）保留相对判据，因为那里的下限不为 0；但**只能在"每层绝对为 0"之后**再谈。

### 本轮摄入 vLLM 源码的确认（`/data/ref/vllm/vllm/model_executor/models/gemma4.py`）

逐项对照 `models/kaon3/src/layer.rs::forward`，已确认**一致**的：

| vLLM | 我们 | 结论 |
|---|---|---|
| `input_layernorm` → attn → `post_attention_layernorm` → +residual | 同序，`rmsnorm_into` + `add_inplace` | ✅ |
| dense MLP **无条件**执行（MoE 并联而非替换） | `dense_gate/up/down` 无条件 | ✅ |
| `post_feedforward_layernorm_1(dense)` + `post_feedforward_layernorm_2(moe)` 相加 | `post_ff_norm_1` / `post_ff_norm_2` | ✅ |
| MoE 输入 = `pre_feedforward_layernorm_2(residual)`，router 输入 = **无权重** norm | `sc.h2` / `rmsnorm_noweight` | ✅ |
| router 预处理 `norm(x) * hidden^-0.5 * scale` | `scale_bf16_n(h^-0.5)` → `mul(scale)` | ✅ |
| 路由：原始 logits topk → **全专家 softmax** → top-k 重归一 → ×per_expert_scale | `moe_route_softmax_f32`（softmax 限定 top-k 等价） | ✅ |
| q_norm/k_norm（带权）/ v_norm（**无**权），RoPE 在 q/k 上、v 不动 | `head_rmsnorm(q,Some) (k,Some) (v,None)`，再 rope(q,k) | ✅ |
| k_eq_v：K 权重填 K/V 两槽 | `v_proj=None` + dtod 拷贝 | ✅ |
| `layer_scalar` 乘在**所有**文本层末尾 | `scale_bf16_n` at end | ✅ |
| **PLE（per-layer embedding）** | 未实现 | ✅ **不适用**：config `hidden_size_per_layer_input = 0`，vLLM 侧 `> 0` 守卫不进入 |

所以"读一遍 vLLM"没有新发现语义错误；剩余差异必须靠**逐层残差流对拍**定位，不能再靠端到端 logprob 猜。

### 下一个仪器：逐层残差流（工具早就有，从没用过）

- vLLM 侧 `scripts/dev/kaon3_vllm_hidden.py`：monkey-patch `Gemma4DecoderLayer.forward`，
  存每层返回的 hidden（已 ×layer_scalar，与 `sc.x` 同定义）→ `L{i}.bin` (f32 LE, [T,H])。
- 我们侧 `kaon3_probe <M> --forward --layers N --dump-layer N-1 --hidden-out f.bin`
  （`--hidden-ref` 可逐行比相对误差）。
- `test/kaon3/**` 里目前**一个 `.bin` 都没有** → 这条链路从未跑通过。这是本轮的起点。

### 仪器改造（2026-10-22 深夜）

之前 `test/kaon3/**` 里**一个 `.bin` 都没有**：逐层残差流这条链路写了工具但从未跑通。本轮补齐：

1. `models/kaon3/src/model.rs::forward` 新增 `hidden_dir: Option<&str>`：每层 forward 后把 `sc.x`
   写 `<dir>/L{l}.bin`（f32 LE，[T,H]）。与 vLLM 侧同名同序，**定义一致**（都含 `layer_scalar`）。
2. `kaon3_probe --hidden-all <dir>`：一次前向 dump 全部层（省掉 30 次权重加载）。仍保留
   `--layers N`（前后都按 N 截断）。
3. `kaon3_probe --hidden-all <ours> --hidden-cmp <ref>`：**纯 Rust 逐层比对**，输出每层
   `mean_rel / max_rel / 相对上一层放大倍数`，用来定位误差从哪层起跳。不再需要 Python 参考。
4. 修 `scripts/dev/kaon3_vllm_hidden.py` 的取批 bug：原来 `cap[-30:]` 在 `max_tokens=1` 时会
   取到**最后一个 decode 步**（单行）而不是 T 行 prefill；改为先按 `h.shape[0] == t` 过滤再取前 30，
   并 `assert len == 30`。**这条 bug 会让逐层对拍从一开始就对错层**，是此前无法推进的直接原因。

编译：`cargo build --release -p dinfer-kaon3 --bin kaon3_probe` 通过（5090 容器内）。

### 【关键结果】逐层残差流首次跑通：**L0 就错 1.9e-2，L1 翻倍**（2026-10-22 深夜）

用例 `p02`（T=24），参照 = `kaon-hid` 容器（GPU 5，干净镜像，BI=1，`enforce_eager`，无 PegaKV）。
参照侧同时修了取批 bug（见上），确认取到的是 T=24 的 prefill 批。

| 层 | mean_rel | max_rel | 相对上一层 |
|---|---|---|---|
| **L0** | **1.913e-2** | 4.607e-2 | — |
| **L1** | **3.754e-2** | 9.311e-2 | **1.96×** |
| L2 | 4.315e-2 | 8.101e-2 | 1.15× |
| L3 | 4.510e-2 | 8.747e-2 | 1.05× |
| L4 | 4.694e-2 | 1.024e-1 | 1.04× |
| L5 (full) | 4.805e-2 | 8.904e-2 | 1.02× |
| L6–L11 | 5.08→6.62e-2 | | 1.01–1.10× |
| L12 | 6.611e-2 | 1.479e-1 | 1.00× |
| L17 (full) | 9.481e-2 | 2.093e-1 | 1.25× |
| L18 | 1.154e-1 | 2.725e-1 | 1.22× |
| L29 | 1.712e-1 | 4.797e-1 | 0.82× |

**结论（判据修正后）**：

1. 用户 2026-10-22 的判断**成立**：0–7 层确实不对。此前 L8 "全 PASS" 只是
   `top5 |Δlp| ≤ 0.15` 这个阈值把 0.06 放过了；残差流相对误差显示 **L0 就有 1.9%**，
   而且在干净参照下这是**确凿的、非噪声的**偏差。
2. **误差不是"到中段才出现"，是从第 0 层开始的**，L1 一次放大到 1.96×，
   之后每层 1.0–1.3× 平稳累积。所以此前"分歧锁定 L8–L11"的结论作废：
   那个结论测的是**logprob 的可见差异**，不是**残差流的第一处偏离**。
3. L0 之后仍每层缓慢放大，说明除 L0/L1 外还有持续的小偏差（每层 1.0–1.3×），
   但**首要目标是把 L0 打平**——否则后面每一层都在错误的输入上比较。

**下一步**：加 `Lm1.bin`（embed 输出，layer 0 之前）两侧都抓，二分
"embed 错" vs "layer 0 错"。

### 读 vLLM 的 RMSNorm 路径（BI=1 时不是默认 kernel）

`VLLM_BATCH_INVARIANT=1` 下 `RMSNorm.forward_cuda` 走
`vllm/model_executor/determinism/batch_invariant.py::rms_norm_batch_invariant`，即
`_rms_norm_kernel`（Triton，**一行一个 program，BLOCK_SIZE=1024 分块累加**）：

1. `sum_sq += tl.sum(where(mask, vals_f32*vals_f32, 0))` 逐块累加到 fp32 标量；
2. `mean_sq = sum_sq / n_cols`；`rms = sqrt(mean_sq + eps)`；`inv_rms = 1/rms`；
3. `y = (x_f32 * inv_rms) * w_f32` → **转回输入 dtype（bf16）**。

关键点：**每个乘法都是"先转 fp32 算，再转回 bf16"**，不是 bf16 链式乘。
我们的 `rmsnorm_into` 走 `ops.norm().rmsnorm(...)`，需要确认是否同样
`(x*inv_rms)*w` 且在 fp32 中算中间值——若我们做的是 `x*(inv_rms*w)` 或先乘 w 再归一，
bf16 舍入顺序不同，**在 L0 就会产生 1e-2 量级的相对差**（bf16 有 8 位尾数，1e-2 正好是
一两次舍入的量级）。这条要作为 L0 的第一嫌疑直接查 kernel 实现。

同时注意 `n_cols = 2816`，BLOCK_SIZE=1024 → **3 个块**累加，第 3 块只有 768 有效。
若我们的 kernel 用不同分块（例如 2048 或 512），fp32 累加顺序不同，也会有 ~1e-7 级差，
但那不足以解释 1e-2。**差三个数量级，说明不是累加顺序，而是 bf16 舍入位置**。

### 【二分第一刀】embed 逐位一致，误差 100% 在 layer 0 内部

加了 `Lm1.bin`（embed 之后、layer 0 之前）两侧抓取后：

```
Lm1    24  0.000e0  0.000e0  -        ← embedding × sqrt(H) 逐位一致
L0     24  1.913e-2  4.607e-2  infx
L1     24  3.754e-2  9.311e-2  1.96x
```

**结论**：`embed_tokens(ids) * sqrt(2816)` 与 vLLM 完全一致（bf16 逐位）。
**所有偏差都产生在 layer 0 内部**。此前"embed 或 0–7 层"的二选一，答案是后者。

### layer 0 的量化配置（决定嫌疑范围）

`config.json::quantization_config`（modelopt，MIXED_PRECISION）：

| 层 0 的模块 | 算法 |
|---|---|
| `self_attn.{q,k,v,o}_proj` | **FP8**（per-tensor 静态激活 scale，`dynamic:false`）|
| `mlp.{gate,up,down}_proj` | **FP8** |
| `experts` | **NVFP4** |

所以 layer 0 里只有 MoE 走 NVFP4，attention 与 dense MLP 都是 FP8 静态。
我们 FP8 路径已核对为 `x*(1/input_scale)` → `__nv_cvt_float_to_fp8(E4M3, SATFINITE)`，
输出乘 `a_scale*w_scale`，与 vLLM `scaled_mm` 口径一致；**1.9e-2 需要继续往 layer 0 内部切**。

### 仪器二：layer 0 内部子模块对拍

`scripts/dev/kaon3_vllm_sublayer.py`：patch `Gemma4DecoderLayer.forward`，**只在
`layer_idx == 0` 且 `hidden_states.shape[0] == T` 时**给 layer 0 的子模块挂 forward hook，
抓 `input_layernorm / self_attn（含 o_proj）/ post_attention_layernorm / mlp /
router / experts / post_feedforward_layernorm_2 / post_feedforward_layernorm`，写 `<name>.bin`。
`enforce_eager=True` 保证不走 torch.compile 图、hook 必触发。
输出文件名与我们 `layer.rs` 待加的 `--sub-out` 同名，便于逐项对比。

（vLLM 侧 `self_attn` 的输出已经过 `o_proj`，见 `gemma4.py:553` `output, _ = self.o_proj(attn_output)`。
所以 `o_proj.bin` 对应我们的 `sc.o`。）

### 【二分第二刀】第一处偏差在 **layer 0 的 attention 路径**，不在 MLP/MoE

`--sub-cmp` 逐项对拍（p02，T=24，参照 vLLM layer0 hook）：

| 子模块 | mean_rel | max_rel | 读法 |
|---|---|---|---|
| **input_norm** | **0.000e0** | **0.000e0** | ✅ 逐位一致 → RMSNorm 与 embed 都不是问题 |
| **o_proj** | **7.653e-3** | 1.547e-2 | ❌ **第一处偏差**（vLLM 的 self_attn 输出已含 o_proj）|
| post_attn_norm | 9.014e-3 | 1.778e-2 | 继承 |
| dense（MLP）| 1.680e-2 | 3.942e-2 | 继承 + 自身 |
| router_logits | 6.276e-3 | 1.303e-2 | 继承 |
| post_ff_norm | 2.136e-2 | 5.249e-2 | 累积 |

**结论**：
1. `input_layernorm` 逐位一致 → **此前怀疑的 RMSNorm bf16 舍入位置不成立**（我们与 vLLM 同序）。
2. 偏差首次出现在 `self_attn`（输出含 o_proj）这一步，**7.65e-3**。
   layer 0 的 attention 路径是：`q/k/v FP8 投影 → 逐头 RMSNorm → RoPE → GQA(窗口 1024) → o_proj(FP8)`。
3. 由此把区间收窄到这 5 个环节。下一刀：抓 `q_norm / k_norm / v_norm`（RoPE 之前）
   与 `q_rope / k_rope`（RoPE 之后）以及 `attn_out`（o_proj 之前），一次定位到具体环节。

`experts` 的 hook 抓不到（MoERunner 里不直挂为子模块），要另抓；本轮先解决 attention。

### 下一刀：layer 0 attention 内部 5 环

layer 0 是**滑窗层**（`layer_types[0] = sliding_attention`），所以 `head_dim = 256`、`nkv = 8`、
`num_heads = 16`，RoPE 是 `default`（theta 10000，全 256 维旋转，`partial_rotary_factor` 只对
full 层生效）。路径：`q/k/v FP8 投影 → 逐头 RMSNorm(256) → RoPE → GQA(窗 1024) → o_proj(FP8)`。

两侧新增同名 dump（RoPE 之前 / 之后 / attention 输出）：

| 我们 | vLLM | 位置 |
|---|---|---|
| `q_norm.bin` / `k_norm.bin` / `v_norm.bin` | hook `self_attn.{q,k,v}_norm` | RoPE **前**（head-norm 输出）|
| `q_rope.bin` / `k_rope.bin` | （需另加 hook 或从 rope 输出取）| RoPE **后** |
| `attn_out.bin` | （需 patch `self.attn`）| o_proj **前** |

缓冲区布局已核对：`q_s = [max_tokens, num_heads*head_dim_sliding]`，第 0 层用 `q_s`（滑窗），
所以 dump 时取前 `tokens*heads*256` 个元素就是紧凑的 `[T, 4096]`，与 vLLM hook 的
`(T, num_heads*head_dim)` 同序（vLLM 在 `q_norm` 前后都做 `unflatten/flatten`，展平后
仍是 head-major 连续）。

### 【二分第三刀】q/k/v 的 head-norm 基本一致，偏差在 **RoPE → attention → o_proj** 之间

p02 / layer 0 逐项对拍（T=24）：

| 子模块 | mean_rel | max_rel | 判定 |
|---|---|---|---|
| input_norm | **0.000e0** | 0.000e0 | ✅ 逐位一致 |
| **k_norm**（RoPE 前）| **0.000e0** | 0.000e0 | ✅ 逐位一致 |
| **v_norm**（无权重）| 4.528e-10 | 1.054e-8 | ✅ 可视为一致（远小于 bf16 ULP≈4e-3）|
| **q_norm**（RoPE 前）| **8.942e-6** | 1.193e-4 | ✅ 远小于 1 个 bf16 ULP，可忽略 |
| o_proj（含 attention）| **7.653e-3** | 1.547e-2 | ❌ **第一处实质性偏差** |

**推论**：
1. FP8 的 q/k/v 投影 + 逐头 RMSNorm **都是对的**（k_norm 逐位、q_norm 1e-4 以内）。
   所以"三个独立 GEMM vs vLLM 一个融合 GEMM、input_scale 是否取对"这个疑虑**基本排除**
   （若 q 的 input_scale 取错，误差会是 1e-2 量级而不是 1e-4）。
2. 偏差产生在 **RoPE 之后到 o_proj 输出** 这一段：`RoPE → GQA(滑窗 1024, FP8 KV) → o_proj`。
3. 继续按同名 dump 切这一刀：`q_rope/k_rope`（RoPE 后）、`attn_out`（o_proj 前）。

侧注：`dense`（1.68e-2）虽比 `o_proj`（7.65e-3）大，但它是**继承** attention 的偏差后
再走 FP8 MLP，不构成独立的第一处偏差。

### 读码补充：BI=1 下 query **不**量化（我们多做了一个 fp8 roundtrip）

`vllm/v1/attention/backends/triton_attn.py:659-667`：

```python
q_descale = (
    layer._q_scale
    if (self._kv_quant_mode == KVQuantMode.FP8_PER_TENSOR
        and query.dtype == self.fp8_dtype)   # ← 只有 query 已是 fp8 才用它
    else None
)
```

即：**query 是否 fp8 由上游决定**，`unified_attention` 只在 `query.dtype == fp8` 时才反量化。
Gemma4 的 `self.attn(q, k, v)` 传进去的 `q` 是 bf16（`q_norm` 是 bf16 RMSNorm，rope 也是 bf16）。
**所以 vLLM 在这里用 bf16 query 算 QK。**

我们 `layer.rs`（`cfg.kv_fp8` 分支）额外做了：

```rust
fp8_e4m3_roundtrip_bf16_n(&mut q.data, tokens*heads*d, 1.0)   // ← 把 q 转成 e4m3 再转回 bf16
```

这会把 q 的每个元素压到 e4m3（**相对精度 ~2^-3 ≈ 6%，即 e4m3 只有 3 位尾数**）。
QK^T 的分数因此带 ~6% 的量化噪声，softmax 之后传到 `attn_out`——
**量级正好落在我们观测到的 `o_proj` 7.65e-3 上**，而且能解释为什么
`q_norm`（RoPE 前，1e-4 以内、可忽略）与 `o_proj` 之间有跳变。

这条**不依赖任何新数据就能判定为可疑**：vLLM 源码里没有对 bf16 query 做量化这一步。
下一轮 A/B：去掉 `q` 的 fp8 roundtrip（K/V 仍按 FP8 KV cache 量化），看 `o_proj` 是否归零。

## 【根因 1，已修】RoPE 的 cos/sin 表：vLLM 用 **bf16**，我们用了 f32

### 证据链（每一刀都是同名张量直接对比，无推断）

| 步骤 | 张量 | 修前 mean_rel | 修后 mean_rel |
|---|---|---|---|
| 投影 + head-norm 输入 | `input_norm` | 0.000e0 | 0.000e0 |
| head-norm 输出（RoPE 前）| `k_norm` / `v_norm` | 0.000e0 / 4.5e-10 | 同 |
| head-norm 输出（RoPE 前）| `q_norm` | 8.94e-6 | 8.94e-6（同，< bf16 ULP）|
| **RoPE 之后** | **`k_rope`** | **1.550e-3** | **0.000e0** |
| **RoPE 之后** | **`q_rope`** | **1.834e-3** | **1.017e-5** |
| **attention 输出（o_proj 前）** | **`attn_out`** | **8.234e-3** | **0.000e0** |
| o_proj 输出 | `o_proj` | 7.653e-3 | **1.149e-7** |
| dense（FP8 MLP）| `dense` | 1.680e-2 | **1.951e-3** |

### 源码依据

`vllm/model_executor/layers/rotary_embedding/base.py:59-63`：

```python
cache = self._compute_cos_sin_cache()      # fp32 计算
if not self.use_flashinfer:
    cache = cache.to(dtype)                # ← dtype = 模型 dtype = bf16
self.register_buffer("cos_sin_cache", cache, ...)
```

`forward_cuda` 里再 `_match_cos_sin_cache_dtype(query)` 对齐一次。
CUDA kernel（`csrc/pos_encoding_kernels.cu`）把 bf16 表 `static_cast<float>` 后做
**fp32 乘加**、结果再存 bf16。所以正确口径是 **"bf16 表 + fp32 旋转"**。

我们原来存 f32 表（注释还写着"vLLM 的 rope 实际按 fp32 算"——**那句是错的**）。
bf16 尾数 8 位 → 相对分辨率 2^-9 ≈ 2e-3，与实测 1.83e-3 完全吻合。

### 修复

`crates/layers/src/nn.rs` 新增 `Rope::with_bf16_cache(&stream)`：把表按 `half::bf16`
就近偶数舍入后仍存 f32（kernel 不变），只复现精度。**opt-in**，其它模型默认仍用 f32 表。
Kaon-V3 的 `Ropes::new` 两套表都调用它。

### 连带修正一个历史误判

此前的记录说"bf16 表让 long_zh ΔNLL 从 +0.03 恶化到 +0.48"——那是在**外部 KV 污染参照**
（PegaKV）下测的，且当时还叠加着 MoE 路由语义错误。**该结论作废**，需在干净参照下重测。

## 【根因 2，已修】router 的 `root_size` 是 bf16 buffer，不是 f64 常数

`gemma4.py::Gemma4Router`：

```python
self.register_buffer("root_size", torch.tensor(self.hidden_size**-0.5), persistent=False)
...
def forward(self, x):
    x = self.norm(x)
    x = x * self.root_size.to(x.dtype)     # ← root_size 先舍入到 bf16
    x = x * self.scale.to(x.dtype)
    router_logits, _ = self.proj(x)
```

我们原来用 f32 `(h as f32).powf(-0.5)`（`scale_bf16_n` 内部 f32 乘、舍入一次），
少了 `root_size → bf16` 那一步。`hidden=2816` 时 `2816^-0.5 = 0.018843…`，
bf16 舍入后相对变化约 **1.5e-3**，正好是 `router_logits` 残差的量级。

修后 `router_logits` **3.284e-3 → 1.716e-3**（改善一半，尚未归零 → 还有第二处，见下）。

### 顺带撤回一次错误尝试（记下来避免重犯）

我曾试图把 `x*root_size` 与 `x*scale` **融合成一个 kernel**（只舍入一次）。
**那是错的**：vLLM 是**两次独立的 bf16 舍入**，融合反而偏离。
已 `git checkout` 撤回 `elementwise.cu` / `ffi.rs` / `ops.rs` 的改动，没有进入主线。

### 当前状态（第 8 轮中途）

| 子模块 | 修 RoPE 前 | 修 root_size 后 | 状态 |
|---|---|---|---|
| input_norm / k_norm / v_norm / attn_out / k_rope | — | **0.000e0** | ✅ 逐位一致 |
| q_norm / q_rope | 8.9e-6 / 1.83e-3 | 8.9e-6 / 1.02e-5 | ✅ < bf16 ULP |
| o_proj | 7.65e-3 | **1.149e-7** | ✅ bf16 噪声量级 |
| post_attn_norm | 9.01e-3 | 3.105e-10 | ✅ |
| dense（FP8 MLP）| 1.68e-2 | **1.951e-3** | 待查 |
| router_logits | 6.28e-3 | **1.716e-3** | 待查（已加 `router_in` 对拍）|
| ff2（MoE 输出）| 3.74e-2 | **1.156e-2** | 待查 |
| post_ff_norm | 2.14e-2 | **6.300e-3** | 继承 |

整层残差流：L0 **1.913e-2 → 6.184e-3**；L1 **3.754e-2 → 2.952e-2**；L29 1.712e-1 → 1.779e-1。

### 读码核对（`dense` 与 router 的 FP8 路径）

- `requantize_with_max_scale`（`w8a8_utils.py:75`）：`max_w_scale = weight_scale.max()` 取**原始 fp32**，
  再 `per_tensor_dequantize(w, s_i)` → `scaled_fp8_quant(w_dq, max_w_scale)`。
  `per_tensor_dequantize` 是 `tensor.to(float16) * inv_scale`，**f16 相乘、不会溢出**
  （f16 上限 65504；权重 dq 值在 e4m3 量级 × e-2 级 scale）。与我们 `load_fp8_group` 的口径一致。
- 激活侧 `layer.input_scale = layer.input_scale.max()`（`modelopt.py:2136`，原始 fp32 取 max），
  与我们 `amax` 一致。
- `scaled_fp8_quant(x, scale)` 静态路径：`x / scale` → 饱和 ±448 → RNE → e4m3；输出乘 `scale`。
  与我们 `gemm_fp8_static_bf16` 的 `x*(1/a_scale)` / `a_scale*w_scale` 一致。

**所以 `dense` 的 1.95e-3 不在 scale 口径上**，下一刀要抓 `dense` 的内部：
`pre_ff_norm`（dense 输入，已知逐位级）、`gelu(gate)*up` 中间量、`down` 输入/输出。

### router：输入与权重都逐位一致，残差只能来自累加顺序（待处理）

- `router_in`（norm+root_size+scale 之后、proj 之前）：**0.000e0**，逐位一致。
- 权重字节校验：`--dump-weights` 导出我们加载后的 bf16 字节，与 safetensors 原始字节
  `cmp` 对比 —— `router.proj.weight` / `router.scale` / `router.per_expert_scale`
  **三者全部 IDENTICAL**。加载路径没有改数值。
- 所以 `router_logits` 的 1.716e-3 **只可能来自 GEMM 累加顺序**。
  vLLM 的 `GateLinear` 在 sm_120（Blackwell family 100）上走 Tier 5
  `torch.mm(x, weight.T, out_dtype=torch.float32)`（cuBLAS），我们用自写 `gate_kernel`
  （128 线程、`block_reduce_sum` 树形归约）。
- 一个**可检验的推论**：对**同一条输入**，两个正确实现在**不同行**上的精度不应相关。
  若 `router_logits` 的相对误差在**同一 token 的 128 个专家间**近似常数 → 是归一化/缩放问题；
  若是随机的逐元素抖动 → 是累加顺序。下一轮直接看逐元素分布即可判定。

### 【已修】重量化的 f16 舍入（`load_fp8_group`）

vLLM `per_tensor_dequantize`：`fake_qweight = tensor.to(float16)` 然后
`dq = fake_qweight * inv_scale`，`inv_scale` 是 **fp32 张量** → torch 类型提升，乘法在 **f32** 里做。
我们原来写 `f16(f16(值) * f16(scale))`，把 scale 先舍入成 f16（10 位尾数，相对误差 ~5e-4），**多一次舍入**。
已改成 f32 相乘、只把最终结果交给 `f32_to_e4m3`。

**实测：`dense` 完全没变（1.951e-3 → 1.951e-3）。**
解释：e4m3 只有 3 位尾数，量化网格步长约 6%，5e-4 的乘子差异几乎不可能让量化结果翻到相邻格子。
所以这处修改**是正确的但非本因**，不解决 `dense` 的残差。继续切 MLP 内部（已加
`pre_ff_norm / mlp_gate_up / mlp_down` 两侧 dump）。

### 我们侧 dump 清单（layer 0）

`--hidden-all <dir>` 现在一次写：`Lm1`（embed）、`L0..L29`（各层残差流）、以及 layer 0 的
`input_norm / q_norm / k_norm / v_norm / q_rope / k_rope / attn_out / o_proj / post_attn_norm /
pre_ff_norm / mlp_gate_up / dense / router_in / router_logits / moe_y / ff2 / post_ff_norm`。
与 `kaon3_vllm_sublayer.py` 的同名 `.bin` 一一对应（`moe_y` 例外：vLLM 的 `experts` 不直挂子模块）。

一层 ≈ 30 s（含 33 s 权重加载），所以**一次前向拿全部中间量**，不再逐项重跑。

### `dense` 残差定位：FP8 GEMM 激活量化在阈值附近翻转（不可归因于我们的 bug）

又切了两刀：

| 张量 | mean_rel | 结论 |
|---|---|---|
| pre_ff_norm（MLP 输入）| **0.000e0** | ✅ 逐位一致 |
| **mlp_gate_up**（gate∥up，gelu 之前）| **1.047e-5** | ✅ 几乎一致（只有 FP8 GEMM 的 fp32 累加重排）|
| dense（down 之后）| **1.951e-3** | ❌ 跳了两个数量级 |

偏差正好卡在 **`gelu(gate)*up` → FP8 激活量化 → down GEMM** 这一段。
`gemm_fp8_static_bf16` 是 `e4m3(x / input_scale)` 后乘回：**e4m3 只有 3 位尾数**
（网格步长约 6%，局部 ~2^-3），输入只要有 1e-5 级的差异，就会有一小部分元素跨过量化网格边界，
输出随之跳动 ~1e-3。这与 `kernel` 正确性无关，是 **W8A8 静态量化的固有敏感性**。

**这条与本轮最初的判据问题直接相关**：它正是"下限不为 0 时，任何实现都不可能逐位一致"的
微观机制。但在 layer 0 这一层，vLLM 自己对自己是 0.0000——原因是**同一实现、同一累加顺序**；
换成任何不同的 GEMM 实现（哪怕完全正确）都会出现这 1e-3。**结论：`dense` 的 1.95e-3
不应作为"我们错了"的证据**；判据要看是否**系统性**偏离，而非是否非零。

### router 误差分布：随机逐元素，不是尺度错

```
row0: mean_d=-3.67e-04 rms_d=2.24e-03 mean_rel=1.36e-03 min_rel=3.2e-05 max_rel=3.60e-03
row1: mean_d=+8.95e-05 rms_d=2.57e-03 mean_rel=1.58e-03 min_rel=4.4e-06 max_rel=3.65e-03
```

`mean_d` 在 0 附近来回（有正有负），`mean_rel` 与 `max_rel` 差 **3 个数量级**，
是典型的**逐元素随机抖动**，不是"整体少乘一个系数"。与 fp32 累加顺序不同的预期一致。

### 抓取脚本两次静默失败（记下来避免再踩）

1. `mlp_` 在使用之后才赋值 → 每次 layer 0 都抛 `UnboundLocalError`，`vllm.generate` 直接崩，
   **结果目录为空**，但外层只 grep 了 `写 ` 这行，看起来像"还在跑"。
2. `MoERunner.forward` 的 patch 写在**每层执行路径里**，且引用模块级 `t`；已改为**模块级只 patch 一次**。

**教训**：抓取脚本必须**先用一个 4-token 短 prompt 冒烟**（几秒完成），确认 `写 N 个张量`
且张量数符合预期（现在 19 个，含 `moe_y`），再对真实用例跑几分钟的完整抓取。
本轮两次都是先跑了完整抓取才发现崩，白等了 ~10 分钟。

## 【重要】router GEMM：我们对，vLLM 错 20 倍；top-8 分歧不是我们的问题

用 **fp64 精确真值**（bf16 原始权重 + `router_in` 直接做 `X @ W.T`，numpy float64）三方对比：

| 实现 | 与 fp64 真值的 max 偏差 | mean 偏差 | top-8 与真值不同的 token |
|---|---|---|---|
| **我们**（`gate_kernel`，bf16×bf16→f32 逐元素累加）| **7.409e-07** | **8.846e-08** | **0/24** |
| **vLLM**（`GateLinear` → cuBLAS `torch.mm(out_dtype=f32)`）| **1.545e-02** | 2.023e-03 | **3/24** |

我们逐元素 100% 更接近真值。**vLLM 的 router logits 有 1.5e-2 量级的误差** —— 这不是我们实现错了。

### 直接后果：`moe_y` 的 1.43e-2 不应归因于我们

`router_logits` 在 top-8 边界附近的分歧（24 个 token 里 2 个、重叠 7/8）导致**部分 token 选了不同专家**，
下游 `moe_y` 因此有 **max 1.16e-1** 的差。**而真值显示分歧是我们对、vLLM 错**（我们 0/24 错、vLLM 3/24 错）。

### 为什么会这样（需要确认，但机制上合理）

`GateLinear` 在 sm_120 上走 Tier 5：`torch.mm(x, weight.T, out_dtype=torch.float32)`。
cuBLAS 在这个形状（M=24、N=128、K=2816）走 split-K + bf16 张量核时，**累加未必在 f32 里逐元素做**，
最后再 cast 到 f32，于是出现 1e-2 级偏差。我们的 `gate_kernel` 是每线程 f32 累加 + 树形归约，
精度接近 fp64 真值。

**结论：这条不是 bug，是我们的实现更准。** 但它是 parity 判据的**根本障碍**——
只要参照实现自身在路由边界上不稳定，逐位置 logprob 就可能因为**换了一个专家**而显著偏移，
这种偏移**不是我们该修的**。判据必须能区分"我们错"与"参照在路由边界抖动"。

### 立即行动

1. 把这个 fp64 三方校验做成**常驻脚本**（`scripts/verify/` 下），以后每次改动都能回答
   "谁的 router 更准"，而不是靠猜。
2. parity 判据加一条：**统计 top-8 集合不一致的 token 数**，并在报告里单列。
   这些 token 上的 logprob 差异不作判据。

### 常驻校验：`scripts/verify/kaon3_router_truth.py`

把上面的 fp64 判定固化成脚本（不进 Python 参考实现，只做真值校验）：
读 bf16 原始权重 + 我们的 `router_in.bin`，用 float64 算精确 logits，比两个实现的偏差与 top-8。

```
OURS vs fp64: max=7.409e-07 mean=8.846e-08  top8 错 0/24
REF  vs fp64: max=1.545e-02 mean=2.023e-03  top8 错 3/24
我们更接近真值的元素比例: 100.0%
ROUTER TRUTH PASS（我们不劣于参照）
```

exit 0 = 我们不劣于参照（逐元素更准比例 ≥ 50% 且 top-8 错得更少）。
以后每次动 router / GEMM 都能一条命令回答"谁的 router 更准"，不用靠推理。

### 撤回一次无效的真值校验（`dense` / gate 投影）

试图用 numpy 复现 FP8 重量化（`per_tensor_dequantize` + `scaled_fp8_quant`）来给 `dense` 算 fp64 真值，
结果**两边与"真值"的偏差完全相同**（max 3.522e-01、mean 4.241e-02 一模一样），
`更接近真值比例 0.0%`。这是**我的仿真错了**（e4m3 的 RNE 舍入与饱和边界我没复现对），
不是两边都错。**该项结论作废，不再引用。**

可用的对照只有相对量：`mlp_gate_up` 相对误差 1.047e-5、`mlp_gelu` 7.525e-4、`dense` 1.951e-3，
逐级放大但都在 FP8（e4m3，3 位尾数）的量化噪声量级内。

### FAQ：`mlp_gelu` 的 7.5e-4 是否可归因于我们的 gelu kernel

查 vLLM CUDA 路径：`GeluAndMul.forward_cuda` → `torch.ops._C.gelu_tanh_and_mul`
→ `act_and_mul_kernel`（`csrc/activation_kernels.cu:110`）：

```cpp
const scalar_t* x_ptr = input + token_idx * 2 * d;   // gate 在前半
const scalar_t* y_ptr = x_ptr + d;                    // up   在后半
x.elts[j] = packed_compute<PACKED_ACT_FN, act_first, HAS_CLAMP>(x.elts[j], y.elts[j]);
```

`gelu_tanh_kernel` 的式子（`activation_kernels.cu:204`）与我们的
`gelu_tanh_f` **逐项相同**：`BETA = sqrt(2/pi) = 0.7978845608028654`、`KAPPA = 0.044715`、
`0.5*f*(1+tanhf(BETA*(f + KAPPA*f*f*f)))`，且都在 **f32** 里算、最后 cast 回 bf16。
布局也相同（gate = 前半、up = 后半）。

所以 `mlp_gelu` 的 7.5e-4 **不可能来自公式或布局**。输入 `mlp_gate_up` 已有 1.047e-5 的差异
（FP8 GEMM 累加重排），经过 `tanh` 后在少量元素上放大到 1.9e-3 max —— 属于
**W8A8 静态量化 + tanh 敏感区的固有放大**，与本轮两个真 bug（RoPE 表精度、router root_size）不同性质。

## 端到端实测：两处真 bug 修完后 **ΔNLL 质变，但翻转率没变**

`scripts/verify/kaon3-parity.sh` run `1022-170737-660897`（30/30 完整，6 卡并行，GPU 全空）：

| 指标 | 修前（P1 干净参照）| **修后** | 下限 FLOOR | 线 |
|---|---|---|---|---|
| 短 prompt 可判定行 top1 翻转率 | 15.53% | **15.32%** | 0.83% | 4.25% ❌ |
| 共有 token \|Δlp\| 均值 | 1.2578 | **1.2220** | 0.0276 | 0.041 ❌ |
| **短 prompt ΔNLL** | −0.0252 | **−0.0018** | −0.0025 | — |
| long_zh ΔNLL | −0.0464 | **−0.0482** | −0.0817 | 0.245 ✅ |
| 单 token top1 | 一致 | 一致 | — | ✅ |
| kernel 对拍 | 全过 | 全过 | — | ✅ |

**通过 5 项，失败 2 项**（此前 4 过 2 败，现在多过了"分片完整性"一项）。

### 这组数字说明什么（重要，决定下一步方向）

1. **ΔNLL 从 −0.0252 收敛到 −0.0018，已经与跨进程下限 −0.0025 齐平**——
   即"我们算出的**实际下一个 token** 的 logprob"与 vLLM 已无实质差别。
   两处修复（RoPE bf16 表、router root_size）**确实是对的、确实有收益**。
2. **但翻转率 15.32%、Δlp 1.2220 几乎没动。** 说明：
   - 这两处 1e-3 级的问题**不是** 15% 翻转的成因；
   - ΔNLL ≈ 0 而 Δlp ≈ 1.2 是**关键矛盾**：若两边 logits 有整体平移，
     两者应同量级变化。现在 NLL 齐平、top-5 候选的 lp 差 1.2 →
     **差异集中在"次要候选的排序/取值"上，且是零均值抖动**（正负相消，故 NLL 不受影响）。
3. 因此剩下的主因**不是数值精度**，而是候选 top-5 里的**系统性排序差异**或
   oracle 比对口径（`kaon3_oracle_diff` 如何取 top-5 / 如何算 lp）。

### 下一步（按优先级）

1. **先查比对口径**：读 `kaon3_oracle_diff` 的 `mean_common_d` 定义，确认它比的是"同 id 的
   logprob"还是"同排名的 logprob"。若后者，排序抖动会直接放大成 1.2。
2. 拿一个具体 case 打印两侧 top-5（id + lp）并排看，判断是"id 不同"还是"同 id 值不同"。
3. 若确实是同 id 值差 1.2，再回去查 lm_head / final norm —— 但那与 ΔNLL≈0 矛盾，故可能性低。

## 【决定性】差异的真实形态：**分布不够尖锐**，不是随机抖动

直接并排打印 `p01` 前几行的 top-5（id, logprob）：

| row | 侧 | rank1 | rank2 | 实际 token |
|---|---|---|---|---|
| 1 | 我们 | **−0.017** | −7.392 | −22.111 |
| 1 | 参照 | **−0.014** | −5.389 | **−23.264** |
| 3 | 我们 | **−0.521** | −3.708 | — |
| 3 | 参照 | **−0.111** | −3.861 | — |
| 4 | 我们 | **−1.099** | −1.724 | −26.161 |
| 4 | 参照 | **−0.803** | −2.178 | **−25.349** |

**规律（三个层次，与"某个 token 的 lp 偏 1 nat"完全不同的诊断）**：

1. **rank1 的 logprob 系统性偏小**（−0.017 vs −0.014、−0.521 vs −0.111、−1.099 vs −0.803）。
   即我们对最高概率 token 的置信度**不够**。
2. **实际出现的 token 反而更接近 0**（−22.111 vs −23.264、−26.161 vs −25.349）。
   即长尾在**尾部**又比参照高。
3. **top-5 的 id 集合只重叠 2–5 个 /5**，且第二名位置常出现完全不同的 id
   （我们 `237323` vs 参照 `5596`/`236866`）。

三条合起来只有一个自洽解释：**我们的输出分布在概率质量分配上系统性地偏离**——
头部不够尖、尾部不够低——不是某个算子的常数偏移。

### 这解释了 ΔNLL≈0 与 Δlp≈1.2 并存的矛盾

- 实际 token 的 logprob 差有正有负（头部偏小 = 负贡献，尾部偏高 = 正贡献），**相互抵消** → ΔNLL≈0。
- 但每个位置 top-5 候选的值与排序都变了 → `mean_common_d` 停在高位。
- **与 router top-8 分歧叠加**：top-8 里换一个专家，第二三名候选就会整批换 id（row1/row2 正是如此）。

### 下一步（按此形态收窄）

优先查**最后一次 logits 变换**这一段：`final_norm → tied lm_head → softmax(logits·final_logit_softcapping)`
- `final_logit_softcapping` 若取值或作用位置不同，**恰好**表现为"头部被压平"（softcap 会把大 logit 压小）；
- `lm_head` 的 tie 与 final norm 的 eps/精度次之。
这是与观测形态最吻合的嫌疑，优先于继续追 router / 量化噪声。

## 【根因 3 候选】softcap 的**舍入次数**与参照不同

`vllm/model_executor/layers/logits_processor.py:113-116`：

```python
logits = logits / self.soft_cap     # logits 是 bf16 张量，Python 标量参与运算，结果回 bf16
logits = torch.tanh(logits)         # 再一次 bf16 舍入
logits = logits * self.soft_cap     # 再一次
```

**三步各舍入一次 bf16**。我们的 `softcap_bf16_kernel` 是
`to_bf16(c * tanhf(to_f32(x[i]) / c))` —— **只在最后舍入一次**。

这与"头部不够尖、尾部偏高"的观测形态吻合：单次舍入的结果更接近真值，
但**偏差方向在头/尾相反**，所以相对参照表现为系统性形态差异，而不是常数平移。
另外我们的 logits 全程存 bf16（lm_head 输出即 bf16），vLLM 在
`_get_logits` 末尾才 `to(head_dtype)`，中间也是 bf16 —— 这部分口径相同。

**已实现 A/B**：`SOFTCAP_ONE_ROUND=1` 走单次舍入（旧行为），默认走新的
`softcap_bf16_round3`（`/c`、`tanh`、`*c` 各回 bf16 一次）。p02 末行 top-5 已有可见变化
（25.75 → 25.625），正在跑完整 verify 测全局指标。
