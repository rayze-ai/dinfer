# Qwen35 里程碑 6：性能（目标超越 vLLM，B300 单卡）

> 起点 `test/qwen35/2026-10-19/bench_b300_3`：TPOT c1 24.1 / c8 28.3 / c32 59.7 / c8-long 157.5 ms，vLLM 10.3 / 11.2 / 12.9 / 11.2；
> TTFT c8-long 8138 vs 408 ms。正确性已过（`qwen35-model.md` 第 10–20 轮）。用户 2026-10-19："目标是超越，不是重写"。

## 环境
- kernel 开发 / 对拍：本机 3090（sm86）、4090 GPU 7（sm89，`scripts/dev/remote4090.sh`）。
- 真权重 bench：B300-50 GPU 0 dinfer `:8012` + GPU 1 vLLM `:8013`（`VLLM_MODEL=/data/models/Qwen/Qwen3.8-27B VLLM_NAME=vllm-qwen35-b300 scripts/dev/vllm_nerdctl.sh run 1 8013 1`）。
  **K3 vLLM 基线（tp8）跑时 B300 无卡**，先做本机部分，等它 stop 再 bench。
- 每轮：`bench_compare.sh test/qwen35/2026-10-19/bench_b300_N "改动一句话"` → `pull` 回本机 commit（硬性规则）。

## 观察方法
- 算子级：`scripts/verify/qwen35_gdn.sh`（gdn_parity，加 chunk case 后要覆盖 varlen 3 序列 + 有/无初始 state + 非 BT 整数倍长度），`ops_parity.sh` 不退化。
- 模型级：`scripts/verify/qwen35_golden.sh`（tiny golden 4/4，含 chunked prefill 4/4/1）。
- 真权重：3 条 greedy prompt 与 vLLM 逐字一致（`docs/loops/qwen35-model.md` 第 16 轮的三条）；bench 四组。

## 停止条件
四组 TPOT 与 out tok/s 均 ≥ vLLM（`bench_compare.sh` 倍率 ≥ 1.00×），TTFT 不劣于 vLLM 的 1.2×；golden / parity 全过。

## 迭代上限
30 轮；每 5 轮无进展回 design 重审。

## 轮次计划（按收益排序）
1. **GDN chunk prefill kernel**（`crates/kernels/cuda/gdn_chunk.cu`，模型无关；BT=64 模板参数由 host 选）：
   复用 `kda_chunk.cu` 的 prep / wy / ho 三段结构，差异：① gate 是逐 v 头标量 `g[t,hv]`（KDA 是逐 k 维），
   `gc` 变 `[T, HV]`，exp2 因子不再按维；② V 与 K 分开（27B 都是 128，但接口不假设相等）；③ q/k 头是 `h = hv / group`；
   ④ 输出门 / RMSNorm 仍用 `gdn_gated_rmsnorm`。代数（fla `chunk_gated_delta_rule`，与 `gdn.cu` 顺序递归等价）：
   `gc = cumsum_chunk(g)/ln2`；`A[i,j] = β_i (k_i·k_j) exp2(gc_i−gc_j), j<i`；`M=(I+A)⁻¹`；`u=M(β⊙v)`；`w=M(β⊙k⊙exp2(gc))`；
   `kg = k·exp2(gc_last−gc)`；`v' = u − w·h_prev`；`o = (q⊙exp2(gc))·h_prev + Aq·v'`；`h_next = h_prev·exp2(gc_last) + v'ᵀ·kg`。
   状态布局与 `gdn.cu` 一致 `[slot, HV, V, K]` f32，conv 仍走 `gdn_conv1d_prefill`。
   Rust：`ops::gdn_chunk_{prep,wy,ho}` + `gdn.rs` prefill（`t > num_reqs`）走 chunk，decode 仍 `gdn_recurrent`；
   `GDN_PREFILL=recurrent` 环境变量回退。对拍 case：`gen_gdn_refs.py` 加 `recurrent_chunk{,_varlen,_tail}`（复用 KDA 命名）。
   预期：c8-long TTFT 8.1 s → < 1 s（KDA 同形状 B300 顺序 36 ms → chunk 4.9 ms / 层 T=1024）。
2. **decode CUDA graph**：qwen35 executor 接 qwen3 的 bucket 捕获（`capture_bucket` / `ensure_graph`），GDN decode 走 `gdn_conv1d_update` + fused-gate `gdn_recurrent`（都无 host 同步）。预期 c1 24 → ~10 ms。
3. 跨流 prefill/decode（`forward_sample_pair`）与混批分流 → c32 TPOT p99。
4. GDN 状态按块缓存（vLLM `mamba_cache_mode=align`）重开 prefix cache → c8 / c8-long TTFT 回命中档。
5. B4 的 sm100 小算子（rmsnorm / silu 向量化）已共享；再看 GDN decode kernel 的带宽利用。

## 记录

| 轮 | 改动 | 结果 |
|---|---|---|
| 1 | **GDN chunk prefill kernel**：`crates/kernels/cuda/gdn_chunk.cu`（prep / wy / ho 三段，全 fp32，BT=64 host 选）+ `ops::gdn_chunk_{prep,wy,ho}`（切块表复用 `KdaChunkPlan`，`GdnChunkScratch` 按 `[T,H,K]`/`[T,HV,V]`）；`gdn.rs` prefill（`t > num_reqs`）走 chunk，decode 仍 `gdn_recurrent`；`GDN_PREFILL=recurrent` 回退；chunk 中间缓冲全模型一份（`GdnChunkWorkspace`，executor 每步 `prepare`，按需增长） | 对拍（3090，参考 = 顺序递归 `gdn_ref`）：`chunk`(T=200) out max_err 2.4e-4 cos 0.999995 / state rel 2.5e-3；`chunk_varlen`([130,64,7]，首条带初始 state) out 1.2e-4 / state rel 2.0e-3；`chunk_qwen`(27B 形状 T=256) out 1.2e-4 / state rel 2.3e-3（容差 out ≤ 5e-4、state rel ≤ 5e-3）。**3090 T=1024 27B 形状（含 gate）：顺序 41.46 ms → chunk 4.16 ms（9.96×）**。golden 见下行。 |
| 1b | 接入验证（同上改动） | `qwen35_golden.sh` 4/4（chunk 路径实测命中：单序列 prefill t=4/7/8/9/12、chunked prefill 4/4/1、双请求混批 t=8 n=2 c=2）；`GDN_PREFILL=recurrent` 回退也 4/4；`ops_parity.sh` PASS；`cargo build --workspace --features qwen3` / `qwen35` 都过。**B300 真权重 bench 未跑**（K3 基线占卡）→ 下一步：`bench_compare.sh test/qwen35/2026-10-19/bench_b300_4 "GDN chunk prefill"`，看 c8-long TTFT 8.1 s 降到哪。 |
| 1c | **bench_b300_4**（B300-50，chunk prefill 真权重）：greedy 3/3 与 vLLM 逐字一致，告警 0（只 1 条 prefix-cache 强制关 warn）。TTFT p50：c1 **44.8**（vLLM 46.1，之前 319）/ c8 **169**（105，之前 1218）/ c32 **357**（261，之前 2547）/ c8-long **470**（403，之前 8138）。TPOT：c1 24.1（不变）/ c8 28.3 / c32 **44.5**（之前 59.7）/ c8-long **36.3**（之前 157.5）。吞吐倍率 0.44× / 0.41× / 0.32× / 0.36×（上轮 0.40 / 0.32 / 0.19 / 0.07）。4090 sm89 `qwen35_gdn.sh` 14/14 过（两卡门禁）。剩余差距全在 decode（TPOT 24 vs 10）→ 第 2 轮 CUDA graph | ✅ 进 CSV |
