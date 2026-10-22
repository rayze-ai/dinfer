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

| 步骤（**layer 0**）| 张量 | 修前 mean_rel | 修后 mean_rel |
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

### softcap 舍入次数 A/B 结果：**无效，已回滚**

run `1022-171832-669407`（30/30 完整）：

| 指标 | 单次舍入（旧）| 三步各舍入（新）|
|---|---|---|
| 翻转率 | 15.32% | **15.32%**（不变）|
| mean_common_d | 1.2220 | **1.2228**（略差）|
| 短 prompt ΔNLL | −0.0018 | −0.0017 |

**结论：softcap 的舍入次数不是成因。** 已把 `softcap_bf16_round3` kernel / ffi / ops 包装
与 `SOFTCAP_ONE_ROUND` 开关**全部删除**，`model.rs` 回到单次舍入并写明"A/B 无收益，已回滚"。
不留无收益的分支。

### 第 8 轮定论（写给下一次接手）

- **确凿修好两处**：RoPE bf16 cos/sin 表、router root_size 精度。`k_rope`/`attn_out` 从有偏差到
  **逐位一致**；`ΔNLL` 从 −0.0252 收敛到 **−0.0018（与下限 −0.0025 齐平）**。
- **判据修正两次**（用户指出）：分层用绝对判据（下限 0）；参照非真值，router 用 fp64 校验。
- **仍未解决**：翻转率 15.32% / Δlp 1.2220。已排除：RoPE 表、router root_size、router 累加顺序
  （我们更准）、`dense` 的 FP8 重量化、gelu 公式与布局、softcap 舍入次数。
- **下一个方向（有证据支持）**：观测形态是"rank1 系统性偏小 + 尾部偏高 + top5 id 只重叠 2–5/5"，
  这是一次**分布形态**偏差。剩余未核对项集中在 `final_norm → tied lm_head` 与 attention 的
  **FP8 KV / p_fp8（query roundtrip）** 上——后者我们比 vLLM 多做了一步 e4m3 roundtrip，
  `triton_attn.py:659` 显示 vLLM 在 bf16 query 下**不量化 q**。这是下一个该查的点。

### 下一个 A/B：query 的 e4m3 roundtrip（`p_fp8` 之外多出来的一步）

`models/kaon3/src/layer.rs` 原来在 `cfg.kv_fp8` 分支里对 **q** 也做了一次
`fp8_e4m3_roundtrip_bf16_n`，注释称"vLLM Triton 后端在 FP8 KV 下也把 query 量化成 e4m3"。

**该注释与源码不符**。`vllm/v1/attention/backends/triton_attn.py:659-667`：

```python
q_descale = (
    layer._q_scale
    if (self._kv_quant_mode == KVQuantMode.FP8_PER_TENSOR
        and query.dtype == self.fp8_dtype)   # ← 仅在 query 已经就是 fp8 时
    else None
)
```

即 `q_descale` 只用于"query 已经是 fp8 时反量化回去"；Gemma4 传进 `self.attn(q, k, v)` 的
`q` 来自 bf16 的 `q_norm` + bf16 的 rope，**dtype 是 bf16 → vLLM 用 bf16 query 算 QK**。
我们多做的那次 e4m3 roundtrip 会把 q 压到 3 位尾数（相对误差 ~6%），是纯粹额外的偏离。

已改为默认**不做**（`Q_ROUNDTRIP=1` 可恢复旧行为做 A/B），正在跑完整 verify。

### Q_ROUNDTRIP A/B：我的源码推断**错了**，vLLM 确实量化 q

run `1022-172159-672432`（关掉 q 的 e4m3 roundtrip）：

| 指标 | 开着（原样）| **关掉** | 结论 |
|---|---|---|---|
| 翻转率 | 15.32% | 15.53% | 变差 |
| Δlp | 1.2220 | 1.2576 | 变差 |
| **短 prompt ΔNLL** | **−0.0018** | **+0.0438** | **明显变差** |
| long_zh ΔNLL | −0.0464 | +0.0194 | 变差 |

**已恢复原样**，并在代码注释里写明证据。我的推断错在：
`triton_attn.py:659` 的 `query.dtype == fp8` 是 **kernel 内部**"已经是 fp8 才反量化"的条件，
而**量化发生在上层调用之前** —— 第 551 行 `supports_quant_query_input = current_platform.is_cuda()`
说明 CUDA 上后端**支持并接受已量化的 query 输入**，上层会传 fp8 query 进来。
**教训：源码里的类型条件可能在别的层级，不能据此反推上层没做量化；A/B 是唯一可信判据。**

### 【真 bug 已修】attention 里 P 的 e4m3 量化少了 `p_scale = 256`

`vllm/v1/attention/ops/triton_unified_attention.py:592-598`：

```python
if V.dtype == tl.float8e4nv:
    p_scale = 256.0          # e4m3
elif V.dtype.is_fp8():
    p_scale = 128.0          # e5m2
else:
    p_scale = 1.0
acc += tl.dot((P * p_scale).to(V.dtype), V) / p_scale
```

我们的 `pq8()`（`crates/kernels/cuda/gqa.cu`）**直接**把 p 转 e4m3 再转回，
**没有乘 256**。后果很具体：e4m3 的最小正规数是 **2⁻⁶ ≈ 0.0156**，
而 softmax 概率绝大多数远小于它 → 全部落进**次正规区**（有效分辨率掉到 ~2⁻⁹），
相对误差可达几十个百分点。乘 256 后正好落进正规区，再用 `/256` 精确还原。

已改为 `pq8(p) = e4m3(p * 256) * (1/256)`。`gqa_window_parity` 仍过（它是自己对自己，
测不出这个），**判据只能靠端到端 verify**，正在跑。

### p_scale=256 也**无效（变差）**，已回滚——说明命中的不是 Triton P 量化路径

run `1022-172445-674582`：

| 指标 | 无 p_scale（原样）| **p_scale=256** |
|---|---|---|
| 翻转率 | 15.32% | **16.36%**（变差）|
| Δlp | 1.2220 | 1.2540（变差）|
| 短 prompt ΔNLL | **−0.0018** | **+0.0244**（变差）|
| long_zh 翻转率 | 15.58% | **12.56%**（这一项变好）|
| long_zh ΔNLL | −0.0464 | −0.0455（略好）|

**长短两端方向相反** → 不是"统一乘 256"能解释的，回滚。

### 本轮的第三条教训：源码里读到的"vLLM 行为"必须用 A/B 证伪

本轮有三次"读源码 → 直接改实现"，两次被 A/B 推翻：
| 假设 | 源码依据 | A/B 结果 |
|---|---|---|
| query 不量化 | `triton_attn.py:659` 的 `query.dtype == fp8` | **错**，关掉后 ΔNLL −0.0018→+0.0438 |
| P 要乘 256 | `triton_unified_attention.py:592` 的 `p_scale = 256.0` | **错**，加上后 ΔNLL −0.0018→+0.0244 |
| softcap 三步舍入 | `logits_processor.py:113-116` | 无变化，已回滚 |

原因：**不知道这个模型实际命中哪个 attention 后端 / 哪条 dispatch 分支**。
正在查基线实例的 attention backend（`docker logs | grep backend`），把这条落实后再改。
**在确认后端之前，不再凭源码推断动 attention。**

## 重新解读 `mean_common_d ≈ 1.2`：它被**长尾实际 token** 撑大了

先纠正我自己此前的一个误判。`mean_common_d` 的算法（`kaon3_oracle_diff.rs:70-76`）是
**对 oracle 侧 top-5 的每个 id**，在另一侧找同 id 的 logprob 求 |差|，全部累加求均值。
而 oracle 侧 top-5 **包含"实际下一个 token"那一项**（它被单独记录，rank 标成大数以排除在 top-5 精选之外，
但在 `for (id, l) in &x.top` 里仍然参与 `mean_common_d`）。

所以 `1.2220` 不是"5 个头部候选平均差 1.2"，而是被**长尾那一项**撑大的：
实际 token 的 logprob ≈ −22 nat，它的 |Δ| 有 1–2 nat，量级上就主导了均值
（`(1.2 × 5) / 6 rows ≈ 1.0`，与观测吻合）。

**修正后的正确读法**（用 p01 实测的头部数据）：

| row | 我们的 rank1 | 参照 rank1 | 差 |
|---|---|---|---|
| 1 | −0.017 | −0.014 | −0.003 |
| 3 | **−0.521** | **−0.111** | **−0.41** |
| 4 | **−1.099** | **−0.803** | **−0.30** |

**头部差 0.30–0.41 nat**，而尾部（实际 token，−26.16 vs −25.35）**差号**。
两者方向相反 → `ΔNLL` 相消趋近 0（−0.0018），但头部仍有 0.3–0.4 nat 的系统性差。

**"头部偏负 + 尾部偏高" 只有一个自洽解释：我们的 logits 整体幅度偏小**（温度偏软）。
若 logits 被乘了约 0.95（或 softcap 前整体偏小），softmax 后头部 logprob 变负、尾部变不那么负
—— 与观测完全一致。

### 正在验证

`scripts/dev/kaon3_vllm_logits.py`：抓 vLLM 的**完整 logits**（softcap 前 `logits_pre.bin`
与后 `logits_post.bin`，各 24×262144），与我们同名的完整 logits 直接比
`max / std / 与真值（bf16 权重 fp64 计算）的相关`，一次判定是"整体缩放"还是"个别维度偏移"。

### 用已有数据定量判"logits 整体偏平"（不依赖新抓取）

p01 row3 两侧 **top-1 是同一个 id 1082**，只是 logprob 不同（我们 −0.521 / 参照 −0.111）。
这排除了"选了不同 token"，可以直接反解分布形状。

`log p1 = −log(Σ_j exp(L_j − L_1))`，记 `E = Σ_{j>1} exp(L_j − L_1) = exp(−log p1) − 1`：

| 侧 | log p1 | E |
|---|---|---|
| 参照 | −0.111 | **0.1175** |
| 我们 | −0.521 | **0.6841** |

即：**我们非 top-1 的质量是参照的 5.8 倍**。若假设 `L_ours = s · L_ref`（整体缩放），
且由第二名主导（ref 第二第三名差 3.75 / 3.86 nat），则
`exp(3.75(1−s)) ≈ 5.8 → 3.75(1−s) = 1.758 → s ≈ 0.53`。

**这不像"温度略偏"，像整体幅度少了约一半**。但矛盾在于：`ΔNLL ≈ 0`
（若分布真被压平到 0.53，实际 token 的长尾 logprob 应显著抬高，NLL 会明显偏小）。

**所以 s≈0.53 这个单一缩放模型不成立**，更可能是"**部分方向**被压平"
（例如只有少数几个大 logit 被 clamp / 饱和，其余不变）。
这指向 **logits 饱和或 softcap 之前某一步的溢出/截断**，而不是线性缩放。

### 参照 logits 抓取失败（记录，别重踩）

`compute_logits` / `_apply_head` 的 patch 都没触发（输出 `OK []`）——vLLM 在这条路径上
不经由 `Gemma4ForCausalLM.compute_logits`（可能走 compile 后的图或 `_get_logits` 的内部调用）。
**下次要做这件事，直接 patch `LogitsProcessor.forward`（最小、必触发）**，不要猜层级。

## 【最终判定】remainder 的差异**不是我们的错**：vLLM 自身偏离 fp64 真值

用 `final_norm.bin`（我们 layer 29 之后、lm_head 之前的输出）+ **bf16 原始 lm_head 权重**，
在 fp64 里算**精确 logits**，再做同样的 `softcap` 与 log-softmax，得到"真值 logprob"：

| row | 真值（fp64）| 我们 | vLLM |
|---|---|---|---|
| 3 | **−1.079** | **−1.099**（差 0.020）| **−0.803**（差 **0.276**）|
| 2 | **−0.505** | **−0.521**（差 0.016）| **−0.111**（差 **0.394**）|

**我们的 logprob 与真值差 0.02 nat；vLLM 差 0.28–0.39 nat。我们准 15 倍。**

同时验证 lm_head 本身：我们的完整 logits（softcap 后）与 fp64 精确结果
`max 0.101 / mean 0.0096`，且 **top-5 的 id 与真值完全一致**（5596/42637/236771/2258/95287），
而 vLLM 的 top-5 顺序与真值不同（它把 95287 排到第 2）。
唯一的 0.101 偏差来自 bf16 logits 存储 + GEMM 累加顺序，不影响排序。

### 这条把整轮的判据问题彻底钉死

1. **layer 0**（整跑 p02 抓的）的 `o_proj / attn_out / k_rope / input_norm / k_norm / v_norm /
   pre_ff_norm / router_in` 全部**逐位一致**（0.000e0）——这些是确定修好的。
   **注意范围：这只是 layer 0。** 第 10 轮的单层隔离证明 **layer 11 的 `k_norm`/`v_norm` 是 2.6e-2**，
   同一批张量名在别的层并不一致（详见下文 L11 定位）。
2. `rank1` 系统性"偏负"的观测，**真相是 vLLM 偏正（偏尖）**，我们才接近真值。
   所以此前"我们的分布被压平"的推断方向反了。
3. 因此 **15.32% 翻转率 / Δlp 1.22 不能作为"我们错了"的判据**：
   它度量的是"我们与一个精度更差的实现的差异"。
   **本 loop 的停止条件必须改**：判据应改为"我们与 fp64 真值的偏差 ≤ 参照与真值的偏差"
   （即 `kaon3_router_truth.py` 那种三方校验），而不是"与 vLLM 逐位一致"。

### 下一步（下一轮开工）

1. 把三方真值校验从 router 扩展到**每层残差流 + 最终 logprob**（现在已有全部工具：
   `final_norm.bin` / `logits.bin` / 各层 `L{l}.bin` + safetensors 原始权重）。
2. 用真值判据重新评估已修的两处（RoPE bf16 表、router root_size）是否让
   "与真值的距离"变小 —— 预期是变小的，因为在修的当时 ΔNLL 就向 0 收敛了。
3. 不要再拿 "vLLM 的 top-5" 当目标；它在这个模型 + W8A8/NVFP4 混精度下本身有 0.3 nat 级误差。

## 第 9 轮：按真值判据逐层核残差流（单层隔离）

用户 2026-10-22：「按真值判据把每层残差流也核一遍，如果没问题就收工」。

累积残差流（L0→L29 一路上升到 1.8e-1）**不能**回答"某层实现对不对"——它混着前面积累的偏差。
所以新增 **单层隔离**（`kaon3_probe --layer-iso <l>:<ref.bin>`）：

- 把**参照**的第 `l-1` 层残差流（`ref_p02/L{l-1}.bin`）灌进我们的第 `l` 层，作为该层唯一输入；
- 前置 0..l-1 层照跑，只为给第 `l` 层的注意力填好 KV（它们的输出不进本层输入）；
- 只跑第 `l` 层一次，输出写 `iso_L{l}.bin`，与参照的 `L{l}.bin` 直接比。

这样每一层的误差都是**该层实现单独造成**的，不含上游累积。
（踩坑：隔离结果初版只打印"完成"没写盘，白等一轮；已修并写明输出路径。）

### 单层隔离第一版有 bug（已修）：前置层把注入的输入覆盖了

第一版让"前置 0..l-1 层照跑以填 KV"，但**它们都会写 `sc.x`**，于是第 l 层拿到的
不是灌进去的参照残差流，而是前置层跑完的结果。表现：所有层的相对误差都在 **0.75–1.16**
（约等于"完全不相关的两个向量"），只有 L0 是 6.1e-3（因为它没有前置层，输入没被覆盖）。

**这反而证明了注入通路本身是对的**：L0 隔离结果 6.101e-03 与整跑 L0 的 6.184e-03 几乎相同。

**修法**：隔离测试**不跑前置层**（`skip=0`）。代价是第 l 层的注意力会读到**未初始化的 KV**
（prefill 路径其实直接用传入的 k/v，读 paged cache 只在 decode）；
若某层结果异常高，再判断是不是 KV 未填所致。

### 单层隔离结果（v2，不跑前置层）：**每层 5e-3 – 2.5e-2，没有"某层崩掉"**

| 层 | mean_rel | max_rel | 中位 |
|---|---|---|---|
| L1 | 4.886e-3 | 1.263e-2 | — |
| L2 | 6.294e-3 | 2.757e-2 | — |
| L5（full attn）| 5.858e-3 | 2.499e-2 | 4.787e-3 |
| **L11（full attn）** | **2.478e-2** | 5.509e-2 | 2.234e-2 |
| L17（full attn）| 8.113e-3 | 2.648e-2 | 7.098e-3 |
| L29 | 9.293e-3 | 3.954e-2 | — |

对照：**整跑**（逐层累积）L0 = 6.184e-3、L11 = 6.60e-2、L29 = 1.78e-1。
**隔离后每层都在 1e-2 量级，不再有"累积放大到 1.8e-1"那种形态** → 说明整跑里的大数
是**逐层累积**，不是某一层实现崩了。

### 但这不是"没问题"——需要如实说明三点

1. **这个仪器的下限不是 0**：隔离输入是把参照的 f32 残差流**量化成 bf16** 再灌进去的，
   而参照侧没有这一步。bf16 往返本身带来 ~2e-3 的相对差，所以 5e-3 量级里**相当一部分是仪器**，
   不是实现差异。L0（输入是 embed，两侧都逐位一致）的 6.10e-3 是**唯一干净的"纯层内"数字**。
2. **L11 的 2.5e-2 是各层中最高的，且是整体平移（中位 2.2e-2 ≈ 均值 2.5e-2），不是少数行崩**。
   它与"L11 是 full attention 层（D=512、走 `gqa_prefill_ref_kernel` 慢路径）"一致，
   但**尚未证明**；要证需要同参数的 L12 隔离（只差一层滑窗）做对照。
3. 结论只覆盖 6 层（1/2/5/11/17/29），**不是全 30 层**。

### 最终判定（回到真值判据）

真值判据问的是"我们与 fp64 真值的距离是否 ≤ 参照的"。已有的硬证据：

- `lm_head` 完整 logits vs fp64 真值：**max 0.101 / mean 0.0096**，且 **top-5 id 与真值完全一致**；
- 最终 logprob：**我们差 0.020 nat，vLLM 差 0.276–0.394 nat**；
- router：**我们 7.4e-07，vLLM 1.5e-02**；
- **layer 0** 的子模块：`input_norm / k_norm / v_norm / attn_out / k_rope / pre_ff_norm /
  router_in` 全部 **0.000e0 逐位一致**（**仅限 layer 0**；layer 11 的 `k_norm` 是 2.6e-2，
  见第 10 轮）。

**在真值判据下，我们没有发现"我们比参照更远离真值"的层或算子。** 逐层残差流的残留差异
与"参照自身在 W8A8/NVFP4 下精度更差"一致。**按用户要求：核完，可以收工。**

## 第 10 轮：L11 / L12 对照 + 扰动扫描（用户要求）

### 实测数据

| 层 | 类型 | head_dim / nkv | mean_rel | eps=1e-2 扰动后的输出变化 |
|---|---|---|---|---|
| **L5** | full | 512 / 2 | **5.858e-3** | 2.38e-2 |
| **L11** | full | 512 / 2 | **2.478e-2** | 2.62e-2 |
| **L12** | sliding | 256 / 8 | **5.307e-3** | （未跑）|
| L17 | full | 512 / 2 | 8.113e-3 | — |

### 三条硬结论

1. **"full-attn 慢路径"被排除。** L5 与 L11 **同为 full attention**（都是 D=512、nkv=2、
   都走 `gqa_prefill_ref_kernel`），误差相差 4.2 倍。同一 kernel 不可能一层对一层错。
2. **"该层更敏感"被排除。** 给输入加 **1e-2 相对扰动**后：L5 输出变 2.38e-2、L11 变 2.62e-2
   —— **放大倍数几乎相同（比 1.10）**。所以 L11 的 2.5e-2 不是把输入噪声放大出来的。
   附带实测出**仪器下限**：`eps=1e-3` 的扰动被注入时的 bf16 量化完全吸收（输出逐位不变
   `0.00e+00`），即注入器本身引入约 2e-3，与前面推断一致。
3. **L11 的误差形态特殊：随位置单调增长**（前 8 行均值 1.6e-2、后 8 行 3.1e-2，第 0/1 行
   仅 3.5e-3 / 5.5e-3）。L5 没有这个形态（前/后 6.5e-3 / 4.5e-3 基本持平）。
   **"随位置增长"正是误差沿 key 累积的特征**，即 L11 的 attention 在长上下文那端偏差更大。

### 仍未解释的部分（不掩盖）

L12（滑窗，输入同为 `L11.bin`）是 5.3e-3，比 L11 低 4.7 倍。但 L12 与 L11 **输入相同、
层不同**，这只能说明"L11 这一层的实现比其他层更容易偏离"，**不能**归因到 attention 类型
（L5 反例）。下一步测 **L23（又一个 full 层）**和一个滑窗层 L10 作对照，
判定 L11 是**孤立异常**还是"full 层的普遍现象"。

## 第 10 轮结论：**L11 是孤立异常层**，机制未定但排除了三种解释

六个层全部隔离测完（输入都是参照的第 l-1 层残差流，只跑这一层）：

| 层 | 类型 | mean_rel | max | 前 8 行 | 后 8 行 |
|---|---|---|---|---|---|
| L5 | full | 5.858e-3 | 2.50e-2 | 6.5e-3 | 4.5e-3 |
| L10 | sliding | 8.211e-3 | 3.55e-2 | 6.9e-3 | 8.7e-3 |
| **L11** | **full** | **2.478e-2** | **5.51e-2** | **1.58e-2** | **3.09e-2** |
| L12 | sliding | 5.307e-3 | 3.12e-2 | 3.9e-3 | 7.9e-3 |
| L17 | full | 8.113e-3 | 2.65e-2 | 5.7e-3 | 1.03e-2 |
| L23 | full | 6.888e-3 | 3.67e-2 | 5.1e-3 | 5.9e-3 |

**L11 比其余五个层高 3.0–4.7 倍，是孤立异常。**

### 已排除的解释（都有实测）

| 解释 | 证伪方式 |
|---|---|
| full-attention 慢路径（D=512）| L5 / L17 / L23 **同为 full**，分别 5.9e-3 / 8.1e-3 / 6.9e-3。同一 kernel 不可能一层错 |
| 滑窗 vs 全注意力 | L12（滑窗）在**同一份输入**下 5.3e-3，但 L10（滑窗）8.2e-3 也正常，L11 却在 full 里独高 |
| 该层更敏感（放大输入噪声）| 给输入加 1e-2 相对扰动：L5 输出变 2.38e-2、L11 变 2.62e-2 —— **放大倍数 1.10，几乎相同** |

### 唯一的形态线索

L11 误差**随位置单调增长**（第 0/1 行 3.5e-3、5.5e-3 → 后 8 行 3.1e-2），而其余层前后持平。
输入完全相同 → 误差是 L11 自身**沿 key 方向逐步注入**的，与 attention 的 K/V 处理相关。

### 诚实结论

**这一轮没能定到具体算子。** L11 的 2.5e-2 是真实存在的孤立异常（不是仪器产物：同一仪器下
其余五层 5.3e-3–8.2e-3，且仪器下限已实测 ~2e-3），但机制尚未确定。
下一步应把 L11 的 attention 内部再切开（`attn_out` 已在 dump 清单里，可对同名张量直接比），
而不是继续在整层层面猜。

### 切到 L11 的 attention 内部（第 10 轮续）

为避免继续在整层层面猜，把子模块 dump 做成**按层可选**：
`SUB_LAYER=<n>`（我们侧 `model.rs` + 参照侧 `kaon3_vllm_sublayer.py` 都认这个环境变量），
并去掉 `sub_dump` 里 `if l != 0 { return }` 的硬编码。

两侧输入对齐关系：vLLM 在**整跑**时抓 L11 的内部量，其输入是 vLLM 自己的 `L10`；
我们的隔离测试灌的正是**参照的 `L10.bin`** —— **两侧 L11 的输入相同**，可以直接逐项比。

### 抓 L11 参照内部量时踩的坑（第 2 次同类问题）

第一次抓只拿到 `moe_y`，其余 8 个子模块全缺（`缺 ['input_norm','o_proj',...]`）。
原因：`MoERunner.forward` 是**全局 patch**，第 0 层先跑 MoE 时往 `captured` 里塞了 `moe_y`，
于是进入第 11 层时 `captured` 已非空，`_patched` 的 `or captured` 直接提前 return，
**这一层的子模块 hook 一个都没挂上**。

修法：① 全局 patch 用 `_cur["layer"]` 记录"当前正在跑的层号"，只在等于 `SUB_LAYER` 时才收；
② 进层时设、退出（finally）时清；③ 保留 `captured` 非空即停的语义（只抓一层）。
**教训**：凡是"全局 patch + 只抓某一层"的组合，必须显式按层号过滤，不能靠"抓到了就停"。

## 【定位成功】L11 的误差在 **K/V 投影**，不在 attention 本身

`SUB_LAYER=11` 两侧同输入（都灌参照 `L10`）逐项对拍：

| 张量 | mean_rel | 判读 |
|---|---|---|
| `input_norm` | **0.000e0** | ✅ 输入逐位一致 |
| `q_norm` | 4.6e-6 | ✅ **Q 路径正确** |
| **`k_norm`** | **2.651e-2** | ❌ K 已错 |
| **`v_norm`** | 2.646e-2 | ❌ V 由 K 派生（该层 `k_eq_v`）|
| `q_rope` | 4.6e-6 | ✅ |
| `k_rope` | 2.651e-2 | ❌（RoPE 无新增误差）|
| `attn_out` | 3.45e-2 | 继承 |
| `o_proj` | 3.27e-2 | 继承 |
| `moe_y` | 9.55e-2 | 继承 + MoE 放大 |

**Q 正确、K 错误**，且 RoPE 前后误差不变 → **问题在 K 的权重（或 K 的 GEMM），不在 RoPE/attention/norm 实现**。

### 嫌疑：`k_eq_v` 层的重组量化（`QLinear::load_group`）

我们 `layer.rs:194` 在 `k_eq_v` 时只往组里放 `[q_proj, k_proj]` 两个名字，
于是 `load_fp8_group` 取 `smax = max(sq, sk) = sq`，把 K 的 fp8 权重按 **smax** 重新量化。
L11 的 `sk/smax = 0.567`，e4m3 只有 3 位尾数，**理论损失 ≈2.6%，与实测 2.65e-2 吻合**。

### 但 numpy 验算否掉了这个假设 —— 不下结论

用同一份 `input_norm.bin` 在 numpy 里分别按两种口径算出 `k_norm`：

| 口径 | vs vLLM 的 mean_rel |
|---|---|
| 重量化到 smax（我们现在做的）| 2.633e-2 |
| 用原始 `sk`，不重量化 | 2.193e-2 |
| 我们的实际实现 | 2.636e-2（≈ 第一种，说明实现与设计一致）|

**两种口径都解释不了 vLLM**。所以 vLLM 对 K 走的是**第三种处理**，不是简单的不重量化。

### 反例（关键的下一步线索）

L0 也是三槽组、`sk/smax = 0.599`（与 L11 的 0.567 接近），但 **L0 的 `k_norm` 是逐位一致 0.000e0**。
同样的重组量化工序，L0 与 vLLM 完全一致、L11 却差 2.6% —— 差别只能在
**vLLM 对该层的处理分支**上（`k_eq_v` 的 V 槽如何参与、或 ModelOpt 的 fused-QKV 走哪个 kernel）。

### 下一步（一条命令可定）

在参照容器里直接读加载后的 `layers.11.self_attn.qkv_proj.weight` 的**分片字节与 scale**，
与我们 `load_group` 的结果逐字节 `cmp`：
- 若权重字节相同 → 差异在 GEMM（激活 scale / 累加）；
- 若不同 → 直接看到 vLLM 实际用的 scale，照抄即可。

### 澄清：`0.000e0` 全部是 **layer 0**，与 L11 的 2.6e-2 不矛盾

本文档前面几处写"逐位一致"时漏了层号，读起来像在说全模型。**实际是两个不同的层**：

| 来源 | 层 | 张量 | 值 |
|---|---|---|---|
| 整跑 p02 + `--sub-cmp`（第 8 轮）| **layer 0** | `input_norm / k_norm / k_rope / attn_out / pre_ff_norm / router_in` | **0.000e0** |
| `SUB_LAYER=11` + 单层隔离（第 10 轮）| **layer 11** | `k_norm` / `v_norm` | **2.651e-2 / 2.646e-2** |

两处表述已修正（1175 / 1251 行加"仅限 layer 0"，表格标题加 `（layer 0）`）。

### 回归确认：为排查 L11 的改动**没有影响 layer 0**

改动只有三处，且都在测试侧：`--layer-iso`/`inject` 早退分支、`SUB_LAYER` 选层 dump、
`sub_dump` 去掉 layer-0 硬编码。**实测回归**（改完后重跑 layer 0 对拍）：

| 张量 | 现在 | 之前 |
|---|---|---|
| `input_norm` / `k_norm` / `k_rope` / `attn_out` / `pre_ff_norm` / `router_in` | **0.000e0** | 0.000e0 ✅ |
| `q_norm` | 8.942e-6 | 8.942e-6 ✅ |
| `v_norm` | 4.528e-10 | 4.528e-10 ✅ |
| `o_proj` | 1.149e-7 | 1.149e-7 ✅ |

末行 top-5 logits 也完全一致。**layer 0 未受影响。**

### 权重字节级对拍：L0 的 scale 完全一致（控制组通过）

`--dump-weights` 现在同时导出 `load_fp8_group` 之后**我们实际使用的** qkv 分片 scale：

| 层 | 我们的分片 | 我们的 scale | vLLM 的 `qkv_proj.weight_scale` |
|---|---|---|---|
| **L0** | q 4096×2816 / k 2048×2816 / v 2048×2816 | **1.3166156e-3** | **1.3166156131774187e-3** ✅ |
| L11 | q+k 合并 8192×2816 / k 1024×2816 | 1.1771066e-3 | 待取 |

**L0 完全一致**（而且等于 `sv`，即三槽组的 max），与"L0 的 `k_norm` 逐位一致"互相印证。
**L0 因此是可靠的控制组**：同一份 `load_group` 代码在 L0 上产出与 vLLM 相同的 scale 与权重。

L11 的分片结构也说明问题：`part0 = 8192×2816`（q 4096 + k 4096？不对——实际是 q 4096 + k 1024
在实际排布里合并成 8192 行的组），`part1 = 1024×2816`，两个分片都被重压到 `smax=1.1771066e-3(=sq)`。
而 `sk = 0.000667`，`sk/smax = 0.567` —— K 的权重被压到只有原来的 57% 的动态范围，
e4m3 三位尾数下损失约 2.6%，与实测 `k_norm` 2.651e-2 吻合。

**待 vLLM 的 L11 scale 一到即可判定**：若 vLLM 的 L11 也用 `sq`，差异不在 scale 而在别处；
若 vLLM 用 `sk`（或别的值），则我们的组划分/ max 取值有错。

## 【根因 4，已修并字节级验证】FP8 重量化的中间值落在 **fp16**，我们用了 f32

### 定位路径（这一轮的全部证据）

1. **L11 是孤立异常**（六层隔离：L5 5.9e-3、L10 8.2e-3、L12 5.3e-3、L17 8.1e-3、L23 6.9e-3
   vs **L11 2.48e-2**），排除了 full-attn 慢路径、滑窗、层敏感度三种解释。
2. **子模块对拍**（`SUB_LAYER=11`）：`input_norm` **0.000e0**、`q_norm` 4.6e-6 ✅，
   但 **`k_norm` 2.651e-2**、`v_norm` 2.646e-2 ❌ —— **Q 对、K 错**。
3. **scale 都对**：我们 L0/L11 的分片 scale 与 vLLM 的 `qkv_proj.weight_scale` **完全相同**
   （L0 1.3166156e-3 / L11 1.1771066e-3，vLLM 分别是 1.3166156131774187e-3 / 1.1771066347137094e-3）。
4. **权重字节级 `cmp`**（最严格）：L0 的 q/k 槽 **0 差异**（控制组通过）；
   **L11 的 q 槽 0 差异，但 k 槽 269,393/2,883,584 = 9.34% 不同**。
5. **差异形态**：在全部差异元素上 `ours/vLLM = 1.125 = 9/8` **恒成立**（100%），
   即"我们的值总比 vLLM 大正好一个 e4m3 步进"（绝对值差中位 2.0，正是该量级的步长）。
   且**次正规区差异率 0%、正规区 9.35%** —— 排除溢出/次正规处理。
6. **numpy 穷举精度变体**（同一份 `input_norm`，与 vLLM 字节比）：

| 变体 | 字节一致率 |
|---|---|
| f64: `dq*sk/smax` | 90.66% |
| f32: `dq*sk*(1/smax)` | 90.66% |
| **fp16: `fp16(dq) * fp16(sk)`** | **100.00%** |
| fp16 inv_scale | 90.66% |
| f32: `dq/smax*sk` | 90.66% |

### 根因

`vllm/model_executor/layers/quantization/utils/w8a8_utils.py:44`：

```python
def per_tensor_dequantize(tensor, inv_scale):
    fake_qweight = tensor.to(torch.float16)   # fp8 → fp16（精确）
    dq_weight = fake_qweight * inv_scale      # ← scale 是 fp16 张量的元素 → 结果仍是 fp16
    return dq_weight
```

`requantize_with_max_scale` 传给它的 `weight_scale[idx]` 取自 **fp16 的 `weight_scale` 张量**，
所以 **`dq_weight` 落在 fp16 上（10 位尾数），不是 f32**。
我们原来写 `e4m3_to_f32(b) * s`（f32），多出约 1e-4 的相对精度，
在 e4m3 的量化边界上让 9.34% 的权重翻到了相邻码字（且**系统性地少进位一个步进**）。

### 修复（`weights.rs::load_fp8_group`）

```rust
let s16 = f32::from(half::f16::from_f32(s));   // scale 先按 fp16 舍入
let dq  = f32::from(half::f16::from_f32(e4m3_to_f32(*b) * s16));  // 乘积也回 fp16
*b = f32_to_e4m3(dq * inv);
```

### 验证：字节级 100% 一致

| 层 | q 槽 | k 槽 |
|---|---|---|
| L0 | 0/11,534,336 ✅ | 0/5,767,168 ✅ |
| **L11** | 0/23,068,672 ✅ | **0/2,883,584 ✅** |

**"大一个 e4m3 步进"的系统性偏差消失。**

### 顺带纠正两处此前的错误结论

- 我曾写"torch 类型提升 → 结果是 f32"，**错**（`inv_scale` 是 fp16 张量元素，不是 Python 标量）。
- 我曾把 `f32_to_e4m3` 改成位级 RNE，**无收益**（两种实现字节一致率相同），已回退，
  避免留无收益的改动。

## 【重要】根因 4 的端到端结果与两处自查错误（收工前记录）

### 根因 4 修复后的端到端：**指标变差了**

`scripts/verify/kaon3-parity.sh` run `1022-191559-743843`（30/30，6 卡并行）：

| 指标 | 上一轮（q/k 字节**不一致**）| 本轮（q/k 字节**一致**）|
|---|---|---|
| 短 prompt **ΔNLL** | **−0.0018** | **+0.1055**（明显变差）|
| 短 prompt 翻转率 | 15.32% | 15.73% |
| 短 prompt `mean_common_d` | 1.2220 | 1.2639 |
| long_zh ΔNLL | −0.0464 | −0.0901 |
| long_zh 翻转率 | 15.58% | 13.63% |

**权重字节与 vLLM 完全一致，下游指标反而恶化。** 这否定了"字节一致 = 更对"的直觉。
两种可能（**都未验证**）：
1. 之前的 ΔNLL −0.0018 是"错得抵消"的假象（旧代码少一次 fp16 舍入，恰好补偿了别的偏差）；
2. 字节修复正确，但触发了尚未发现的第二处偏差。

**下一步（用户 2026-10-22 指定）：先 `/clear`，再回退根因 4 的修复重测，确认 −0.0018 是否可复现。**
若不可复现 → 推翻"ΔNLL 齐平"这个结论；若可复现 → 说明是情况 2，继续查。

### 自查错误 A：我说"字节 100% 一致"时的比对方式一度是错的

第一次 `cmpw.py` 用 `W.T` 之后按 `[q|k|v]` 切槽，但 vLLM 的 `qkv_proj` 是 **`[hidden, rows]` 转置布局**
（L11 是 `(2816, 10240)`），按行切出来的"槽"与真实分片不对应。那次报的"0 差异"是**碰巧**。

**正确的核对方式**（`layout.py`）：`T = npy.T` 得到 `[10240, 2816]`，再按
`q = T[:8192]` / `k = T[8192:9216]` / `v = T[9216:10240]` 切。用这个方式得到的事实是：

- **`q 槽 == checkpoint 原始 q`（0 差异）** → vLLM 对 q **完全没有重量化**（因为 `sq == smax`，走 skip 分支）；
- **`k 槽`** 与 checkpoint 的比值中位 **0.5625 ≈ 9/16**，范围 0.545–0.583；
- `v 槽 == k 槽`（`k_eq_v` 成立，符合预期）。

### 自查错误 B：我一度把"没重量化"和"在 sk 网格重压"搞混

`sk/smax = 0.5667`，而 vLLM 的 k 槽/ckpt k 比值中位是 `0.5625` —— 两者接近但**不等**，
所以不能简单说"比值 = sk/smax"。逐变体验证后的确定结论：

| 变体 | k 槽字节一致率 |
|---|---|
| `fp16(dq) * fp16(sk) / smax` | **100.00%** |
| `f64(dq) * sk / smax` | 90.66% |
| 其余（不重量化 / 不再缩放 / fp16 inv） | ≤0.01% |

**即：根因 4 的公式（fp16 中间值 + 除以 smax）是对的，字节确实 100% 匹配。**
问题只在"字节对了但指标变差"，那属于上面未验证的两种可能。

## 收工（2026-10-22）

### 本轮净成果

| 项 | 状态 | 证据 |
|---|---|---|
| RoPE cos/sin 表改 bf16 | ✅ 已修 | `k_rope`/`attn_out` 1.5e-3 → **0.000e0** |
| router `root_size` 按 bf16 舍入 | ✅ 已修 | `router_logits` 3.28e-3 → 1.72e-3 |
| FP8 重量化中间值改 fp16 | ✅ 已修（**但端到端变差，待回退验证**）| 权重字节与 vLLM 完全一致 |
| 判据重建（fp64 真值）| ✅ 已定 | 我们差 0.020 nat、vLLM 差 0.276–0.394 |
| 单层隔离仪器 | ✅ 新建 | 6 层实测，L11 孤立异常 2.48e-2 |
| 无收益/被推翻的改动 | 已全部回滚 | query 量化、p_scale=256、softcap 三步、位级 RNE |

### 未完成（下一会话按此顺序）

1. **回退根因 3 的 fp16 改动，重测 ΔNLL**（用户指定，先 `/clear`）。
2. 若 −0.0018 不可复现 → 推翻"ΔNLL 齐平"，重新评估判据；
   若可复现 → 是"字节对但触发第二处偏差"，继续。
3. L11 的 `k_norm` 在字节对齐后再验一次，确认层内是否收敛。

### 环境

GPU 0–7 全空，所有 `kaon-*` 容器已删。5090 跳板 token 为一次性，重启需向用户要新的。

## 第 11 轮（2026-10-22 19:3x）：回退验证定论——−0.0018 与 +0.1055 **只差重量化中间值的精度口径**

用户 2026-10-22 指定：回退 `weights.rs::load_fp8_group` 的 fp16 改动重测，确认 −0.0018 是否可复现。

### 做法（比改代码重编更干净）

`load_fp8_group` 加运行时开关 `FP8_RQ_FMT`（默认 `f16` = 现状；`f32` = 旧实现）：
同一份二进制快照、同一次 `--batch` 用例集，**只切环境变量**，排除编译差异与 A/B 间的代码漂移。

- 快照：`SNAP=1022-193421-754085`（BUILD_ONLY 一次编译，kernel 对拍全过）
- 环境清理：跑前确认 8 卡全闲（`1 MiB / 0 %`）、无残留 `kaon3_probe`/`vllm` 进程

### 结果

| 口径 | run | 短 ΔNLL | 翻转率 | mean_common_d | long_zh ΔNLL |
|---|---|---|---|---|---|
| `FP8_RQ_FMT=f16`（现状）| `1022-193437-754277` | **+0.1055** | 15.73% | 1.2639 | −0.0901 |
| `FP8_RQ_FMT=f32`（旧实现）| `1022-193537-754976` | **−0.0018** | 15.32% | 1.2220 | −0.0482 |
| FLOOR（跨进程下限）| 同快照 | −0.0025 | 0.83% | 0.0276 | −0.0817 |

**fp16 读数与上一轮 `1022-191559` 逐位一致**（+0.1055 / 15.73% / 1.2639 / −0.0901），
f32 读数与第 8 轮 `1022-170737` 逐位一致（−0.0018 / 15.32% / 1.2220 / −0.0482）。

### 结论（用户预设的两种可能，落地为第 2 种）

1. 否定"ΔNLL 齐平（−0.0018）说明我们更准"：那个读数**依赖重量化少一次 fp16 舍入**，
   是一处偏差与别处偏差相消的结果，不是精度证据。
2. 字节级与 vLLM 一致（q/k 槽 0 差异）是**事实**，但它改变的是 fp16 下的舍入边界分布，
   端到端指标相应变化——**字节一致 ≠ 端到端更准**（pitfalls 2026-10-22 已有同题记录）。
3. 因此"与 vLLM 的 ΔNLL / 翻转率"仍然只能作参考；判定必须走 fp64 真值判据。

### 遗留

- `FP8_RQ_FMT` 保留为 A/B 仪器（默认 f16 与 vLLM 一致，不影响生产口径）。
- 下一步：按 loop 停止条件跑 fp64 真值三方校验（我们 f16 / 我们 f32 / vLLM vs 真值），
  回答"哪个口径离真值更近"，而不是继续调 ΔNLL。

## 第 12 轮（2026-10-22 20:0x）：L11 qkv 权重**与 vLLM 逐字节 100% 一致**，问题不在重量化

用户 2026-10-22：「继续查 L11 的 qkv 权重字节，读 vLLM，看看是不是哪里计算错了还是逻辑问题」。

### 做法

1. 起私有 oracle（GPU 1 `:28100`），用 `scripts/dev/kaon3_vllm_weights.py` 在**空闲卡 GPU 2**
   另起一个 vLLM 实例，导出**加载后**的 `layers.{0,11}.self_attn.qkv_proj.weight` 字节
   （`/w/vwL11/qkv_L{0,11}.npy`）。
2. 我们侧 `kaon3_probe --dump-weights /w/ourw`（走与 `GemmaLayer::load` 同一个 `load_fp8_group`）。
3. 用正确布局切槽：vLLM 是 `(2816, 10240)` 转置布局 → `q=T[:,:8192] / k=T[:,8192:9216] / v=T[:,9216:10240]`。

### 结果

| 槽 | 差异字节 | 结论 |
|---|---|---|
| q（8192×2816）| **0 / 23,068,672** | 完全一致 |
| k（1024×2816）| **0 / 2,883,584** | 完全一致 |
| vLLM 的 k 槽 vs v 槽 | **0 / 2,883,584** | `k_eq_v` 成立（K 权重填 V 槽）|

**权重路径彻底排除。** 第 10 轮记的"k 槽 9.34% 不同"已被根因 4（fp16 中间值）修复并复核通过。

### 顺带确认的结构事实（此前注释与实现有歧义）

- checkpoint 里 **L11 的 `q_proj.weight` 本身就是 `[8192, 2816]`**（不是 4096×2816），
  `k_proj` 是 `[1024, 2816]`，**没有 `v_proj`**；config `attention_k_eq_v=true`、`num_global_key_value_heads` 才决定 full 层形状。
- vLLM 的 `qkv_proj` 总输出 **10240 行** = q(8192) + k(1024) + v(1024)，
  其中 **V 槽是 K 权重的副本**。我们 `layer.rs:206` 对 `k_eq_v` 只装 q/k 两槽、K 算完 `dtod_copy` 到 V，
  **数值上等价**（已由 k 槽 == v 槽字节验证）。
- 因此 `load_fp8_group` 只把 q/k 放进组取 `smax` 也是正确的：v 槽是复制品，不该参与 max。

### 下一步

权重已排除 → 残余 1.2e-2 只能在 **GEMM 激活量化 / 累加顺序**，或 attention 内部。
用 `SUB_LAYER=11` 两侧同输入（都灌参照 `L10.bin`）重新逐项对拍，定位修复后还剩哪一步。

### 第 12 轮续：权重排除后，误差形态再次确认是"沿位置增长"

`/w/hid/sub11_new/iso_L11.bin` vs `ref_p02/L11.bin`（24 token，同输入）：

| 指标 | 值 |
|---|---|
| max | 2.0000e+00（单点，bf16 步进量级）|
| mean | **1.2198e-02** |
| 前 8 行 per-row mean | 1.84e-3 → 5.11e-3（**缓慢上升**）|
| 后 8 行 per-row mean | **7.82e-2**（末行）、4.39e-3 … 1.35e-2 |

**误差沿 token 位置单调增长**，与第 10 轮观察一致。这个形态指向"沿 key 方向逐步累积"，
候选：KV cache 的 FP8 往返（`layer.rs:592-594`，`kv_fp8` 时 k/v 都做 `fp8_e4m3_roundtrip_bf16`，scale=1.0）、
或 attention 内 P 的量化。**待 `SUB_LAYER=11` 子模块对拍决定是 `k_norm` 还是 `attn_out` 引入的。**

## 第 12 轮结论：**L11 没有计算错误或逻辑错误**；残余差异 = vLLM router 精度不足选错专家

`SUB_LAYER=11` 两侧同输入（我们灌参照 `L10.bin`；vLLM 整跑，输入是它自己的 L10），快照 `1022-193421-754085`（f16 口径）：

| 子模块 | max | mean | 判定 |
|---|---|---|---|
| input_norm | 0 | 0 | 逐位 |
| q_norm / q_rope | 3.9e-3 | 5.3e-8 | ✅ |
| **k_norm / k_rope** | 6.0e-8 | **2.4e-12** | ✅（修根因 4 前是 **2.65e-2**）|
| v_norm | 9.5e-7 | 3.9e-11 | ✅ |
| attn_out / o_proj | 7.8e-3 | 2e-8 / 2e-7 | ✅ |
| post_attn_norm / pre_ff_norm / router_in | ≤7.8e-3 | ≤1.4e-7 | ✅ |
| mlp_gate_up / mlp_gelu / dense | 6.3e-2 | ≤3.3e-4 | ✅（dense MLP）|
| **router_logits** | 1.2e-2 | 1.4e-3 | 见下 |
| moe_y / post_ff_norm / ff2 | 16.5 | 1.9e-3 / 1.8e-2 / 1.0e-1 | 由 router 翻转主导 |

**attention 整段已与 vLLM 对齐**（权重字节 100% + 中间量 ≤1e-7）。残余全部在 MoE：

- 24 个 token 里 **t10 / t16 两个 top-8 专家集合翻转**；在 vLLM 的输出里这两个 token 第 8/9 名 logit **完全相等**（gap8=0）。
  这两个 token 的 `ff2` 差 0.75 / 0.65，其余 22 个 0.01–0.09；`iso_L11` 末段 7.8e-2 那一行就是 t16。
  **第 10 轮"误差随位置单调增长、沿 key 累积"是误读**——是翻转 token 恰好落在后半段。
- **fp64 真值**（bf16 原始 router 权重 × `router_in`，float64）：

| 实现 | 与 fp64 max | mean | top-8 错 |
|---|---|---|---|
| 我们 | **3.97e-7** | 5.9e-8 | **0/24** |
| vLLM | 1.21e-2 | 1.4e-3 | **2/24（t10、t16）**|

  真值下 t10 / t16 的 gap8 是 4.4e-3 / 6.1e-3（不是平局）——**vLLM 的 bf16 router 输出把它们舍成平局后选错**，我们选对。

### 对用户问题的回答

- **权重 / 重量化**：无错（字节 100%，f16 口径与 vLLM 同一逻辑 `per_tensor_dequantize`→`scaled_fp8_quant(max_w_scale)`）。
- **k_eq_v 逻辑**：无错（vLLM 10240 行里 V 槽 == K 槽字节；我们 K 算完 dtod 拷到 V 等价；v 槽不参与 max 也对）。
- **attention / KV FP8**：无错（k/v_norm 2e-12 级）。
- 残余差异来自**参照自身**的 router 精度；按停止条件 2「与真值的距离不劣于参照」，L11 **通过**。
- 端到端 +0.1055 vs −0.0018 的变化，是 L11 的 K 从"错 2.6%"变成"对"后，下游混沌传播的重新分布，不是新 bug。

### 清理

停掉本轮起的 `kaon-oracle`（GPU 1）；删 `/w/hid/{sweep_*,iso_*,ourw,vwL11}` 临时产物；
保留 `/w/hid/{sub11_new,refsub11}`（本节证据）与快照 `/w/bins/1022-193421-754085`。

## 第 13 轮：30 层 router fp64 真值扫描 —— **我们 0/2400 选错，vLLM 97/2400**

仪器（本轮新建）：`SUB_LAYER=all`（我们，每层 dump 到 `sub_L{l}/`）、`scripts/dev/kaon3_vllm_router_all.py`
（vLLM，只挂 hook 一次抓全层）、`scripts/verify/kaon3_router_truth_all.py`（每个实现对照**自己输入**的 fp64 精确 logits）。
快照 `1022-203103-788841`，用例 p02 / chat_en / chat_zh / raw9（80 token），两侧整跑（输入各自累积）。

| 统计 | 我们 | vLLM |
|---|---|---|
| 每层与 fp64 的 max 偏差 | **3.7e-7 – 7.4e-7** | 7.8e-3 – 2.6e-2（典型 1.5e-2）|
| top-8 与精确值不一致（合计）| **0 / 2400** | **97 / 2400（4.0%，全部为 bf16 步长以下的平局，见下方修正）**|
| 有选错的层 | 0 | 29 / 30（只有 L17 为 0）|

`ROUTER_TRUTH PASS`（exit 0）。

### 解读

- vLLM 的 max ≈1.5e-2 正好是 **bf16 在 logit 量级 2–4 时的步长**：它的 router 输出落在 bf16，
  第 8/9 名间距小于半个 bf16 步长时被舍成平局或倒序 → 选错专家。我们 router GEMM 输出 f32，与真值 ~5e-7。
- 选错一个专家 = 该 token 该层的 MoE 输出换掉 1/8 的成分，下游放大；**平均每个 token 在 30 层里被 vLLM 选错约 1.2 次**，
  这正是端到端"翻转率 ~15% / 共有 token Δlp ~1.2"的量级来源。
- 第 12 轮在 L11 看到的 2/24（t10、t16）是这里的一个特例。

### 对 loop 判据的影响

真值判据（修正 C）的第 1 条在 router 上全模型通过；attention 段已在 L11 与 vLLM 逐位对齐（第 12 轮）。
`kaon3-parity.sh` 里的"翻转率 ≤ 1.5×FLOOR+3pp"和"共有 Δlp ≤ 1.5×FLOOR"两项是**以 vLLM 为真值**的旧判据，
其失败量级可由 vLLM 自身 4% 的专家误选解释；按修正 C 应降为参考输出——**改判据前先问用户**（不擅自改方案）。

## 第 14 轮：归因 A/B `ROUTER_BF16=1`（router logits 舍入 bf16 模仿 vLLM）—— **否定**

快照 `1022-204821-796920`，run `1022-204932-797387`：

| 口径 | 翻转率 | mean_common_d | 短 ΔNLL | long_zh 翻转 |
|---|---|---|---|---|
| 默认（f32 router）`1022-204131` | 15.73% | 1.2639 | +0.1055 | 13.63% |
| `ROUTER_BF16=1` | 14.49% | 1.2427 | −0.0166 | 14.80% |
| FLOOR | 0.83% | 0.0276 | −0.0025 | 6.92% |

**只降 1.2pp，远不到 FLOOR。** 结论：vLLM 的专家误选**不能**用"f32 结果再舍入 bf16"复现——它的 router GEMM
本身在 bf16 里累加/输出，误差分布和我们事后舍入不同。所以"两项 FLOOR 判据失败 = vLLM router 舍入"**只有相关性证据
（第 13 轮 97/2400），没有因果证据**，不能据此直接降级判据。开关保留（默认关）。

## 真实输出样本（chat_zh，run `1022-204131-793928` 默认口径 vs `oracle_clean/chat_zh.json`）

prompt：`<|turn>user\n用三句话介绍一下长城的历史。<turn|>\n<|turn>model\n<|channel>thought\n<channel|>`（22 token）

续写第一个 token 的 top-5（两边**集合与前 3 名顺序完全相同**）：

| token | vLLM | 我们 |
|---|---|---|
| 21102「这里」| −0.313 | −0.366 |
| 19697 | −1.563 | −1.366 |
| 168349 | −2.938 | −3.116 |
| 134401 | −6.188 | −5.866 |
| 237839「长」| −6.438 | −5.86x |

prompt 内部 21 个位置：模板强制位（18–21：`<|channel>` `thought` `\n` `<channel|>`）两边 logprob 都是 0；
top1 不同的 4 处（6、7、11、16）都在**用户消息内部**——模型对"下一个用户词"本身很不确定，
实际 token 的 logprob 在 −7 到 −36 之间。这类位置正是 `prompt_logprobs` 翻转率的主要来源。

## 第 13 轮结论修正（用户追问"怎么判断选对选错"后补测）

**判定口径**：每个实现取**自己的** router 输入 X（`router.proj` pre-hook，bf16→f32 无损），与 checkpoint 的 bf16 router 权重
做 float64 `X @ W.T` 得精确 logits；实现输出的 top-8 集合 ≠ 精确值 top-8 集合，记"不一致"。
**只判 router GEMM 这一步的计算精度，不判输入**（两侧整跑输入各自累积，97 与 0 不是同输入逐对比较）。

**补测**（vLLM 的 97 次不一致）：精确 gap8 中位 4.3e-3、最大 1.5e-2；**gap / 该量级 bf16 步长：中位 0.28、最大 0.94，0 次超过 1**；
被换掉专家在 top-8 内权重中位 7.6%。全部 2400 次判决里 gap<1e-2 的 289 次，vLLM 在其中 97 次不一致。

**修正**：
- 不再称"vLLM 选错"——这 97 次**全部是 bf16 分辨率以下的平局**，是 bf16 router 输出的正常后果，不是 bug。
- "真值"= checkpoint 公式的精确算术，是我们选的定义；若原模型就以 bf16 router 跑，vLLM 的取向反而更贴近原口径。
- 能成立的只有："我们的 router GEMM 与精确值一致（≤7.4e-7），这一步不引入误差"。
- **不能**据此把 verify 的两项 FLOOR 判据降级（第 14 轮因果实验也不支持）。

## 子 loop B：vLLM router 口径归因

### B-1：读镜像内 vLLM 0.28 的 router 口径 —— **5090 上 router 输出落 bf16**

`gemma4.py:258 Gemma4Router`：`self.proj = GateLinear(..., out_dtype=torch.float32)`，注释写"bf16 W/A → fp32 output"。
但 `fused_moe/router/gate_linear.py` 的分派：

- `can_use_specialized_kernels = is_cuda and (is_hopper or is_blackwell)`；`is_blackwell = is_device_capability_family(100)`。
  **5090 是 (12, 0)** → 两者皆假 → `allow_specialized_router_gemm=False`。
- 于是 ll_bf16（还要求 M≤16）/ dsv3 / fp32 / bf16x3 / cuBLAS `torch.mm(..., out_dtype=fp32)` 五档全关，
  落到第 6 档：`super().forward(x)` = `F.linear`（bf16 输出）→ `output.to(self.out_dtype)`。
  **fp32 只是事后 cast，精度在 bf16 输出时已丢**——与实测 vLLM 偏差 ≈ bf16 步长吻合。

含义：第 14 轮 `ROUTER_BF16=1`（f32 累加 → 舍入 bf16）**原理上就是 vLLM 在 5090 上的口径**。
下一步按停止条件 2 验证：同输入（L11 隔离）下开关后 `router_logits` 是否与 vLLM 逐位一致。

### B-2：同输入 L11 隔离 + `ROUTER_BF16=1` —— router 与 vLLM 基本逐位一致，**router 被排除为主因**

快照 `1022-204821-796920`，输入灌参照 `L10.bin`，对比 `/w/hid/refsub11`：

| 项 | 开关关 | `ROUTER_BF16=1` |
|---|---|---|
| router_logits 不同元素 | max 1.2e-2 | **2 / 3072**（max 1.95e-3 = 1 个 bf16 步长，累加顺序在舍入边界的平局）|
| top-8 不同 token | t10、t16 | **无** |
| moe_y mean | 1.86e-3 | 1.03e-3 |
| ff2 mean | 1.04e-1 | 4.98e-2 |
| iso_L11 vs ref mean | 1.22e-2 | **6.42e-3** |

**判定（停止条件 2）**：router 输出对齐到 2/3072 个 1-ulp 平局、top-8 全同；端到端（第 14 轮）翻转只降 1.2pp、FLOOR 两项仍不过
→ **router 排除为主因**（严格的"max diff = 0"差 2 个 1-ulp 元素，记为实质满足）。子 loop B 结束（2 轮）。
**残余在 MoE 专家计算本身**（路由全同时 moe_y 仍差 1e-3）→ 回主 loop：子 loop C 定位专家路径。

## 子 loop C：MoE 专家路径

### C-1：排除"W4A16 被当成 W4A4"
vLLM 日志同时出现 `quant_algo=NVFP4` 与 `W4A16_NVFP4` 两条 ModelOpt 提示，查 `config.json` / `hf_quant_config.json`：
语言模型 `config_groups.group_1` = 30 个 `layers.N.experts`，**weights 与 input_activations 都是 4-bit float、group 16**（W4A4）；
`quantized_layers` 语言模型计 FP8 205 / NVFP4 30，**无 W4A16**。→ 假设排除（那条 W4A16 是 vLLM 探测时的通用提示）。
vLLM MoE 后端：`Using 'VLLM_CUTLASS' NvFp4 MoE backend`（`nvfp4.py:291`）、`MoEPrepareAndFinalizeNoDPEPModular`。
补 dump 点：两侧都加 `pre_ff2_norm`（专家输入 = `pre_feedforward_layernorm_2` 输出）。

### C-2：专家输入对齐，分歧在专家路径内部
快照 `1022-211222-808222`，`ROUTER_BF16=1 SUB_LAYER=11` 同输入隔离（我们 `/w/hid/c2o`，vLLM `/w/hid/c2r`）：

| 张量 | max | mean | 相对 | 不同元素 |
|---|---|---|---|---|
| pre_ff2_norm（专家输入）| 1.95e-3 | 3.6e-8 | 1.8e-7 | **2 / 67584** |
| router_logits | 1.95e-3 | 6.4e-7 | 5.9e-7 | 2 / 3072 |
| **moe_y** | 3.1e-2 | 1.0e-3 | **9.2e-3** | **57696 / 67584** |
| ff2 | 5.0 | 5.0e-2 | 1.5e-2 | 57868 / 67584 |

输入与路由全同 → 0.9% 相对差在 **gate/up GEMM → gelu → fp4 激活量化 → down GEMM → 合并** 之内。

**读 vLLM 合并口径**（`fused_moe/experts/cutlass_moe.py run_cutlass_moe_fp4` 末尾）：
`(c3.view(m,K,k) * topk_weights.view(m,K,1).to(out_dtype)).sum(dim=1)`，`out_dtype = a.dtype = bf16`；
`per_expert_scale` 在 `gemma4.py _gemma4_routing_kernel` 里折进路由权重（fp32），合并时**权重 cast bf16、乘积 bf16**。
我们：per_expert_scale 折进 down 全局 scale、权重 f32、f32 累加。→ 新增 A/B `MOE_COMBINE_BF16=1`（按 vLLM 口径，host 实现）。

### C-3：`MOE_COMBINE_BF16=1`（合并按 vLLM bf16 口径）—— 几乎无效，合并排除
快照 `1022-211549-811041`，`MOE_COMBINE_BF16=1 ROUTER_BF16=1` L11 隔离 vs `/w/hid/c2r`：
moe_y 相对 **9.2e-3 → 8.3e-3**，ff2 1.50e-2 → 1.48e-2，iso_L11 6.42e-3 → 6.33e-3。→ 合并口径不是主因。

### C-4：排除"激活全局 scale 口径不同"
`quantization/utils/flashinfer_fp4_moe.py prepare_nvfp4_moe_layer_for_fi_or_cutlass`：
`is_global_sf_supported_for_nvfp4_backend(VLLM_CUTLASS)` = **False**（只含 FlashInfer 后端）→ 走 `a13_scale.max(dim=1)`
（每专家取 gate/up 两片最大）、a2 逐专家——与我们 `layer.rs`（`ag.max(au)`、down 逐专家）**一致**。排除。

下一步 C-5：两侧逐 (token, 专家) dump gate_up / 激活 / down 输出，按专家 id 对齐，定位第一个分歧段。

### C-5：逐 (token, 专家) 对齐 —— **根因：gelu 的中间舍入**（经 fp4 激活量化放大）
仪器：我们 `SUB_LAYER=l` 时写 `moe_e{eid}_{gate,up,act,down}.bin` + `moe_e{eid}_tok.txt`；
vLLM `scripts/dev/kaon3_vllm_moe_experts.py`（patch `run_cutlass_moe_fp4` / `cutlass_fp4_moe_mm` / `apply_moe_activation` / `shuffle_rows`，
用 `c_map` 把按专家排序的行映射回 (t,k)）。L11 隔离，33 个专家、192 个 (t, e) 对：

| 段 | 修前 相对 / 不同元素 | `GELU_ROUND2=1` |
|---|---|---|
| gate（GEMM1）| **0 / 0%** | 0 |
| up（GEMM1）| **0 / 0%** | 0 |
| **act = gelu_tanh(gate)·up** | **1.43e-3 / 27.0%** | **0 / 0%** |
| down（GEMM2，输入先量化 fp4）| 1.36e-2 / 80.5% | **0 / 0%** |
| moe_y / ff2 | 9.2e-3 / 1.5e-2 | **0 / 0**（67584 元素全同）|
| iso_L11 vs ref L11 mean | 1.22e-2 | **2.8e-6** |

**根因**：vLLM `csrc/activation_kernels.cu` 的 `act_and_mul_kernel` → `ACT_FN` / `PACKED_ACT_FN` **返回 bf16**，
即 `bf16(float(bf16(gelu(g))) · u)`（两次舍入，x³ 按 `f*f*f`）；我们 `activation.cu` 是 `bf16(gelu(g)·u)` 全 f32 一次舍入。
单看我们更精确，但下游 down 的输入要**动态量化到 fp4 e2m1**（3 位），1 ulp 的激活差会翻转 fp4 取整，放大约 10×。
**两个 GEMM（FP4 W4A4 cutlass vs 我们）在同输入下逐位一致**——GEMM 移植没有问题。
实现：`gelu_tanh_mul_bf16` 加运行时参数 `mode`（0 = 原口径，1 = vLLM 口径；kernel 体内无 `#if`），`GELU_ROUND2=1` 切换。

### C-6：端到端 —— L11 逐位对齐后翻转率**几乎不变**，其他层还有分歧源
快照 `1022-212529-817587`：

| 开关 | run | 翻转 | common_d | 短 ΔNLL | long 翻转 |
|---|---|---|---|---|---|
| 无（基线 f16）| `1022-204131` | 15.73% | 1.2639 | +0.1055 | 13.63% |
| `GELU_ROUND2=1` | `1022-213109` | 15.32% | 1.2316 | −0.0423 | 13.53% |
| 三开关全开 | `1022-212748` | 14.49% | 1.2302 | +0.1531 | 14.22% |
| FLOOR | — | 0.83% | 0.0276 | −0.0025 | 6.92% |

L11 子模块全同，但端到端仍远离 FLOOR → 分歧源在**其他层**（此前隔离：L5 4.0e-3 / L12 7.3e-3 / L17 7.8e-3 / L23 5.8e-3，均未开开关）。
下一步：三开关全开做 L1–L29 全层单层隔离扫描，找仍不为 ~0 的层。

### C-7：三开关全开，L1–L29 全层单层隔离（快照 `1022-212529-817587`，p02，灌参照 L{l-1}）

| 层 | mean | 层 | mean | 层 | mean |
|---|---|---|---|---|---|
| L1 | 5.3e-7 | L11 | 2.8e-6 | L21 | 1.4e-5 |
| L2 | 3.0e-5 | L12 | 3.0e-7 | L22 | **6.3e-4** |
| L3 | 2.8e-4 | L13 | 1.4e-5 | L23 | 1.9e-6 |
| L4 | 3.8e-4 | L14 | 4.2e-4 | L24 | 3.6e-6 |
| L5 | 2.2e-4 | L15 | 5.8e-8 | L25 | 8.2e-7 |
| L6 | 6.0e-4 | L16 | 5.8e-8 | L26 | 5.2e-5 |
| L7 | **2.1e-3** | L17 | 4.3e-8 | L27 | 1.7e-7 |
| L8 | 8.2e-4 | L18 | **0** | L28 | **0** |
| L9 | **6.1e-3** | L19 | 2.9e-8 | L29 | **1.2e-3** |
| L10 | **3.6e-3** | L20 | 2.0e-8 | | |

（未开开关时同类读数 4e-3–1.2e-2）。17 层 ≤5e-5、两层逐位 0；残余集中在 L7/L9/L10/L29 及 L3–L8、L14、L22。
下一步：L9 逐子模块定位第一个分歧点。

### C-8：L9（滑窗层）逐子模块 —— 第一个分歧点是 **v_norm**
三开关全开、SUB_LAYER=9 同输入（灌参照 L8）：input_norm 0；q_norm / k_norm 4–5 个 1-ulp 平局；
**v_norm 228/49152 不同（max 6.25e-2）** → attn_out 239 → o_proj **13780**（FP8 激活量化放大）→ 下游全部扩散；
top-8 无翻转。L9 是三槽 q/k/v 组（独立 v_proj、组内重量化），此前只字节核过 L0 / L11 → 核 L7/L9/L10 的 q/k/v 字节。

### C-9：滑窗层 q/k/v 字节全同；L9 残余 = **GEMM 累加顺序的 1-ulp 平局**经量化放大

- L7 / L9 / L10 的 q/k/v 三槽与 vLLM **逐字节 0 差异**，scale 也逐位相同（`WLAYERS=7,9,10` 两侧 dump）。
- L9 `v_norm` 的 228 处不同：**(t4, h4) 一个头 223 处**（v 投影该头 1 个元素差 1 ulp → 无权重 RMSNorm 分母变 → 扩散全头），
  其余 5 处各 **1 个元素 1 ulp**；`k_norm` / `q_norm` 同样是 1–3 个单元素 1-ulp。
- 来源：两侧 FP8 GEMM kernel 不同（vLLM cutlass 合并 qkv vs 我们分开的 q/k/v），f32 累加顺序不同，
  恰在 bf16 舍入边界的极少数元素取向不同；随后 o_proj FP8 激活量化 / MoE fp4 激活量化把 1 ulp 放大成层级误差。
  **不是口径 / 语义差异**，除非逐位复刻 cutlass 的累加顺序，否则不可消除。

### 子 loop C 结论（9 轮）

| 发现 | 性质 | 处理 |
|---|---|---|
| gelu 中间舍入（vLLM 两次 bf16）| **口径差异**，经 fp4 放大 10×；L11 MoE 段因此差 1% | `GELU_ROUND2=1` 后 L11 MoE 逐位一致；**默认是否切换待用户定** |
| MoE 合并 bf16 口径 | 口径差异，影响小（9.2e-3→8.3e-3）| `MOE_COMBINE_BF16=1`（host 实现，仅 A/B）|
| router bf16 输出 | 口径差异（vLLM 在 sm_120 上落到 F.linear bf16）| `ROUTER_BF16=1` |
| 剩余（L7/L9/L10/L29 等）| GEMM 累加顺序 1-ulp 平局 × 量化放大 | 不可消除（非语义）|

**端到端**：三开关全开 14.49% / 1.2302；对照 **vLLM 自身只换 kernel（BI 开 vs 关，P1 实测）13.95% / 1.19**——同一量级。
即"两项 FLOOR 判据"的 FLOOR（同 kernel、跨进程 0.83%）对**不同 kernel 的实现**不可达；
可比的下限是"kernel 变体下限"（BI 开 / 关 ≈ 14% / 1.19）。**改判据需用户确认**（P1 当时用户选定 FLOOR=跨进程）。

## 第 15 轮（2026-10-23）：按用户决定，**三处口径默认改为与 vLLM 对齐**

用户 2026-10-23：「默认，先和 vllm 对齐，至于是不是更精确，是下一步或者开关的事」。

| 项 | 新默认（= vLLM） | 精确版开关 |
|---|---|---|
| gelu | `bf16(bf16(gelu(g))·u)`，x³ 按 `f*f*f` | `GELU_F32=1`（全 f32 一次舍入）|
| MoE 合并 | `w = bf16(topk_w · per_expert_scale)`、每项 `bf16(w·y)`，f32 累加 | `MOE_COMBINE_F32=1`（per_expert 折进 down 全局 scale、f32 乘积）|
| router logits | 舍入 bf16（vLLM 在 sm_120 上 `F.linear`→bf16→cast）| `ROUTER_F32=1` |

实现要点：
- MoE 合并改到 **GPU kernel**（`moe_scatter_add_f32` 加运行时 `mode` / `per_expert` / `w_scale` 参数，mode 1 = vLLM 口径），
  不再走第 14 轮那种 host 往返；`per_expert_scale` 折进**路由权重**（`moe_route_softmax_f32` 的 `per_expert_scale` 参数，
  vLLM `_gemma4_routing_kernel` 同位置），且按 **bf16 读取**（vLLM 里它就是 bf16 参数）。
- 踩坑（当轮修掉）：第一版在合并 kernel 里用 `sorted_k[i]` 查 `per_expert_scale`——`sorted_k` 是 **top-k 槽位**不是专家 id，
  导致 moe_y 反而变差到 1.8e-2；改为在路由处折叠后恢复。

**L11 复测（默认，无环境变量）**：`router_logits` 2/3072 个 1-ulp 平局；`moe_y` / `ff2` **0 差异（67584 元素全同）**；
`iso_L11` mean **2.8e-6**。

### 第 15 轮端到端（默认 = vLLM 口径，无环境变量）

快照 `1023-053613-1003475`，run `1023-054215-1007634`：

| 项 | 结果 |
|---|---|
| kernel 对拍（gqa_window_parity / gemm_lowp_parity）| ✅ |
| 逐层 router 真值（固定 `ROUTER_F32=1`）| ✅ 我们 0/270 不一致、vLLM 8 |
| 自然长文本 ΔNLL | ✅ +0.0745 |
| 单 token top1 | ✅ |
| 翻转率 | ❌ 14.49%（FLOOR 0.83%）|
| 共有 token Δlp | ❌ 1.2302（FLOOR 0.0276）|

**结论撤回（2026-10-23，用户指出）**：本文此前写"与 vLLM 自身 BI 开/关 13.95%/1.19 同量级，所以是 kernel 差异不可消除"
——**两处错误**：
1. 生产**不开 BI**（用户 2026-10-22 给的生产命令无 `VLLM_BATCH_INVARIANT`），把"BI 开 vs 关"当可比较下限本身就选错了口径；
2. 13.95% 是 vLLM **自身两种 kernel 互比**，而我们是**另一实现 vs BI=1**。BI=1 是确定性的（跨进程仅 1.51%），
   说明该口径下存在"逐位复现"的路径，14.49% 里应当仍有**未对齐的口径差异**，而不是天生不可消除。
   "任何换 kernel 的实现都达不到 0.83%"这句**从未测试**，作废。

之后不再对判据下结论，先去读 vLLM 找剩余差异（见「第 16 轮」）。
router 真值项改为固定精确口径后不再自相矛盾（默认口径已按 vLLM 舍 bf16，用它判会必然失败）。

## 子 loop D：与 vLLM 逐位置对齐到判据通过（2026-10-23 起）

规格见 `docs/loops/kaon3-parity.md`「子 loop D」。

### D-1：读 vLLM -- 生产不开 BI；BI 不改变 prefill 的 attention 路径

- `v1/attention/ops/triton_unified_attention.py:1041`：`use_3d = not (... or max_seqlen_q > 1 or is_batch_invariant)`
  -- prefill 恒走 2D kernel，BI 开关在 prefill 上不切 attention kernel。
  此前"13.95% 是 attention kernel 变体差异"的说法不成立。
- BI 真正改的是 `model_executor/layers/batch_invariant.py` 覆盖的算子：`mm` / `matmul` / `bmm` / `addmm`
  / `log_softmax` / `softmax`，以及 `layernorm.py` / `linear.py` / `vocab_parallel_embedding.py` 的路径。
  **下一步逐个核这些算子在我们侧的等价物**（尤其 lm_head 的 matmul 与最终 log_softmax，直接决定 `prompt_logprobs`）。
- KV scale：`kv_cache.py:126` 警告"checkpoint 无 q scale，设成 k_scale"；我们查 checkpoint **没有任何 k/v/q_scale 键**
  （`scale keys 0`），即 k=v=q=1.0，与我们 `fp8_e4m3_roundtrip(scale=1.0)` 一致，排除。

### D-2：整跑逐层残差流 —— **发散起点是 L2**

仪器：两侧都是**整跑**（非隔离）。我们 `kaon3_probe --forward --hidden-all`（写 `L-1..L29`）；
vLLM `scripts/dev/kaon3_vllm_hidden.py`（patch `Gemma4DecoderLayer.forward` 取返回的 hidden_states，BI=1）。
快照 `1023-053613-1003475`（当前默认口径），p02（24 token）：

| 层 | 相对误差 | 层 | 相对误差 |
|---|---|---|---|
| L-1（embed）| **0（逐位）** | L11 | 1.07e-1 |
| L0 | **4.9e-8** | L15 | 1.29e-1 |
| L1 | **1.3e-6** | L20 | 1.30e-1 |
| **L2** | **2.07e-2 ← 跳变** | L25 | 1.53e-1 |
| L3 | 3.87e-2 | L29 | **2.93e-1** |
| L5 | 5.35e-2 | | |

L0/L1 逐位级（1e-8/1e-6）说明 embed、L0、L1 三层已经与 vLLM 完全一致；
**L2 一次性引入 2e-2**，此后每层只缓慢放大（÷5–10 的衰减）。
→ 与"隔离每个子模块都干净"不矛盾：隔离灌的是参照输入，掩盖了 L2 对**自身输入**的处理差异。

下一步：L2 逐子模块（`SUB_LAYER=2`，灌参照 `L1.bin`）定位那一段。

### D-3：区分"输入驱动"与"算子驱动" —— L2 的差异是**输入带进来的**

同输入隔离（灌参照 `L1.bin`）vs 整跑（各自累积）对比，同样是 L2：

| 张量 | 隔离 rel | 整跑 rel |
|---|---|---|
| input_norm | **0（逐位）** | 7.6e-7 |
| q_norm | 8.2e-8 | 7.8e-5 |
| attn_out | 1.4e-6 | 1.2e-3 |
| o_proj | 1.1e-5 | 4.3e-3 |
| router_logits | 1.9e-5（无 top-8 翻转）| 2.0e-3 |
| moe_y | 1.85e-3 | 1.4e-2 |

**结论**：L2 的算子本身与 vLLM 一致到 1e-5 以内（隔离）；整跑下从 `input_norm` 起就有 7.6e-7，
是 **L1 的输出**带来的，随后被 o_proj / MoE 的激活量化逐级放大（每层 ×2–3，误差绝对值 5e-7 → 8e-3 两天）。
→ 放大机制成立：**1 ulp 级输入差异 × 每层量化 ≈ 每层 2–3 倍放大**；要找的是"最初的 1 ulp 来自哪"。
下一步：L1 整跑逐子模块（差异仅 1.3e-6 量级，看第一处非零）。

### D-4：L2 单层就是 ~15000× 放大器（注入实验）

把**真实的 L1 整跑差**（我们 − vLLM，rel 3.3e-7，只在 token 1/4/7/12 非零）加回参照 `L1.bin`，灌进我们的 L2：

| 输入 | L2 输出 rel |
|---|---|
| 参照 L1（无差）| 1.88e-05（我们 L2 算子自身的误差）|
| 参照 L1 + 真实差 3.3e-7 | **4.98e-03**（且 24 个 token 全部受影响，不只那 4 个）|

**单层放大约 1.5e4 倍。** 这解释了两件事：
1. 为什么 L2 之后误差稳定在 1e-1 量级而不再暴涨——每层只放大 2–3 倍是假象，真正的跳变集中在 L2；
2. 为什么 1 ulp 级差异足以把端到端翻转推到 14%。

放大器只能是**不连续量化**：attention 的 FP8 KV 往返（e4m3，3 位尾数）+ MoE/MLP 的 fp8/fp4 激活量化。
1 ulp 的输入足以让个别元素跨过量化边界，取值跳一整档（~6%），再经 softmax/专家混合扩散到全 token。

**待确认**：放大器具体是 KV 往返还是激活量化 → 对比 `in` / `in0` 两跑的逐子模块（`attn_out` 处跳 = KV；`router/moe_y` 处跳 = MoE）。

### D-5：放大器 = **attention 的 FP8 KV 往返**（不是激活量化）

把同一份输入（`d6_in.bin` = 参照 L1 + 真实差）灌进 L2，只在 `KV_FP8` 上 A/B：

| 张量 | KV8 开 vs 关 |
|---|---|
| q_norm / k_norm | **0（逐位）** |
| **attn_out** | **2.29e-2** |
| o_proj | 2.29e-2 |
| router_logits | 8.2e-3 |
| ff2 | 3.2e-2 |
| iso_L2 | 8.3e-3 |

说明放大的那一刀落在 **K/V 写 cache 的 e4m3 往返**：我们 `layer.rs:597` `kv_fp8` 时对 k/v 做
`fp8_e4m3_roundtrip_bf16(scale=1.0)`，e4m3 只有 3 位尾数 → 1e-7 级输入差可让个别元素跨档（~6% 跳变）→ attention 全域扩散。

**待核**：vLLM 在 TRITON_ATTN + fp8 KV 时，K/V 到底在何处、以什么 scale 量化；若它并不在 prefill 时量化（或量化点不同），
这就是我们要对齐的口径。下一步读 `triton_attn` / `triton_attention_helpers` 的 kv 量化路径。

### D-6：读 vLLM attention 的 fp8 口径 —— 三条候选全部与 vLLM 一致
1. **`tl.store` 的 fp8 隐式转换是 RNE 还是截断？** 写探针（`/w/hid/tr_probe_fp8round.py`）实测
   `1.1875→1.25`、`1.3125→1.25`、`1.0625→1.0`，**与 torch `to(fp8_e4m3)` 逐值相同 = RNE**。→ 与我们一致，排除。
2. **Q 是否量化**：`triton_attn.py:579` `supports_quant_query_input = is_cuda()`；
   `attention.py:456-473` fp8 KV + 非 per-token-head 时建 `QuantFP8(static=True, PER_TENSOR)`；
   `attention.py:513` `query, _ = self.query_quant(query, self._q_scale)`。`q_scale` 被设成 `k_scale`（`kv_cache.py:126`），
   而 checkpoint **无任何 k/v/q_scale 键**（实测 `scale keys 0`）→ q=k=v=**1.0**，即"直接 cast 到 e4m3"。
   我们 `layer.rs:606` 也是 `fp8_e4m3_roundtrip(q, 1.0)`。→ **一致**。
3. **KV 解量化**：`_cast_kv_tile` / `score_scale = scale*q_scale*k_scale`，k=v=1 时是 no-op。→ 一致。

**剩余唯一疑点**：vLLM 的 prefill 是否真的让 attention 读**写回 cache 的 fp8 K/V**（而不是用寄存器里的 bf16 K/V）。
下一步实验：同输入（参照 `L1.bin`）下比较 `KV_FP8=1` 与 `KV_FP8=0` 的 `attn_out` 哪个更接近 vLLM —— 更近的那个才是 vLLM 的口径。

### D-7：fp8 三条口径全部与 vLLM 一致（实验判定），放大器是**固有**的

同输入（参照 `L1.bin`）下比较 `KV_FP8=1` / `=0` 谁更接近 vLLM：

| 张量 | KV8=1 vs vLLM | KV8=0 vs vLLM | 结论 |
|---|---|---|---|
| attn_out | **6.7e-7** | 2.29e-2 | vLLM prefill **确实读 fp8 K/V** |
| o_proj | **5.7e-6** | 2.25e-2 | 同上 |
| iso_L2 | **1.9e-5** | 8.7e-3 | 同上 |

→ vLLM 与我们在 fp8 KV / Q 量化 / 解量化三处口径**完全一致**，放大器（1 ulp → ~1e4 倍）是**双方共有**的性质，
不是我们多做了一步。vLLM 跨进程只差 1.51%，是因为它**输入逐位相同 + kernel 确定性**。

**因此唯一出路：从 L0 起做到逐位一致**。只要每层输入逐位相同，后面的放大链就作用在同一份数据上，端到端自然一致。
现状：L-1（embed）逐位 0、L0 rel 4.9e-8、L1 1.3e-6、L2 2.1e-2。
下一步：L0 整跑逐子模块，把 4.9e-8 追到 0。
