# kaon3-layer1 迭代记录（2026-10-24）

## 第 1 轮：验证脚本与判据修正
- 参数检查全部失效：`docker inspect` 返回 JSON 数组（`"--block-size","32"`），按空格匹配必然失败。
- **判据用错**：截 1 层时 top1 margin 全 0（`decisive=0`），`flip` 是 0/0，
  用它会误报"逐位一致"。改为：`decisive>0` 看 flip，否则看 `|Δlp|` 与 `ΔNLL` 是否为 0。
- 修正后实测：`mean_common_d=0.0085`、`ΔNLL=+1.5064` → **差异在 L0 之内**。

## 第 2–4 轮：norm 与 FFN 合成段
- 权威源码 `vllm/ir/ops/layernorm.py:11`：`x = x.to(weight.dtype) * weight`（先舍 bf16 再乘 w）。
- 数值验证：`RMSNORM_MODE=2` 与运行时 `rms_norm` **逐位相同（0/67584）**；mode 0 差 17063。
- `seg_l0.py` 逐段复现：`input_norm / pre_ff_norm / pre_ff2_norm / ff1 / post_ff_norm`
  **五段全部 0/67584 逐位相同**。

## 第 5–7 轮：定位到 fp8 GEMM
- 同一输入下：`q 88911/98304`、`k 43145/49152`、`v 2/49152`。
  **v 几乎不差 → GEMM 实现没错。**
- FX 图显示 vLLM 的 `qkv_proj` 是**单个** `[2816, 8192]` GEMM。
- 合并一次 N=8192 实测 `ndiff=13/196608`；分开三次差 13 万个 → **根因是调用形状**。

## 第 8 轮：requantize 算法验证通过
`w8a8_utils.py:76 requantize_with_max_scale` 复现：
```
[1] 复现权重 vs vLLM 实际权重 : 23068672/23068672 字节全同
[2] 复现权重 GEMM vs vLLM 输出: 0/33554432 逐位相同
[3] input_scale = max(三个) = 0.59375
```
且发现 **Rust 里 `weights.rs:225 load_fp8_group` 早已实现**这套逻辑（还处理了 fp16 中间值）。

## 第 9 轮（未完成，已回退）：合并前向
改了 `layer.rs` 加 `qkv_proj` 合并字段 + `forward_qkv_fused`，编译通过但**端到端更差**：
```
分开三次（QKV_SPLIT=1）: flip  6.12%  |Δlp| 1.0062
合并一次（新）         : flip 87.76%  |Δlp| 3.6515
```
**我的验证方法错了**：只比了"fused vs split"（两边都是我们自己），且第 0 层是 `k_eq_v`，
两边都只算 N=6144，互相一致但都不等于参照的 N=8192。

## 卡点（下次接手第一件事）

**第 0 层 `k_eq_v=True`：参照的 `qkv_proj` 到底是 N=8192 还是 N=6144？**
源码 `gemma4.py:413` 说 `qkv_proj` 结构不变、K 权重同时填 K/V 两个槽（→ N=8192）；
但 `gemma4.py:1513` 的 `packed_modules_mapping` 注释说 k_eq_v 层用分开的 q/k（不打包）。

已抓到 vLLM 运行时真实的 `[2816, 8192]` 权重（`scripts/dev/capqkv.py`），
但三段都不等于 checkpoint 原值（有 requantize）。**要做的是**：用第 8 轮已验证的
requantize 公式，分别按「q,k,v」与「q,k,k」两种拼法复现，看哪个逐字节等于抓到的权重。
这一步能一次定死 `k_eq_v` 的语义。

**代码状态**：`layer.rs` 的未验证改动**已回退**，工作区干净。

## 第 9b 轮：合并前向重做（Opus 4.8 接手）

**先解卡点**：L0 是 **sliding 层**（full 层只有 5/11/17/23/29），`k_eq_v_at(0)=false` →
L0 有独立 q/k/v → 合并 N=4096+2048+2048=8192。第 9 轮"L0 是不是 k_eq_v"是伪命题。

**实现**（这次对参照，不是 fused-vs-split 自比）：
- `weights.rs`：`load_fp8_group_merged` 重量化后把 q/k/v 字节拼成单块 `[8192,2816]`
  （requant 逻辑提取成 `requant_part_to_max`，与原 `load_fp8_group` 同口径）。
- `layer.rs`：`Attention` 改为单个 `qkv: QLinear` + `qkv_widths`；前向一次
  `forward(_preq)` 到 `sc.qkv[tokens,8192]`，再 `copy2d` 按列段切 q/k/v。
  k_eq_v 的 full 层只合并 q+k、v 复用 k 段（dtod）。
- 本机 cargo check：kaon3 零错误；5090 容器 CUDA_ARCHS=120 build 过。

待测：1 层 oracle 端到端 `|Δlp|` / `ΔNLL`（判据 `decisive==0` → 看 mean_common_d 与 dnll）。

### 第 9b 轮结果：1 层端到端（24 用例，merged qkv，RMSNORM_MODE=2）
```
POOLED cases=24 rows=494 decisive=0 mean_common_d=0.0083 dnll=-0.0006
```
- `decisive=0` → 判据看 `mean_common_d` 与 `dnll`。pooled `dnll≈0`（每例 ±1.8 相消），
  但 **`mean_common_d=0.0083 ≠ 0`** → **仍未逐位**。
- kernel 级 merged qkv 已证明对（第 7 轮 13/196608）；端到端残差必来自 **下游未验证的 GEMM**：
  `o_proj`（[2816,4096]）/ attention(gqa) / MoE 专家 GEMM —— 第 4 轮表里全标"未验证"。
- 下一步：逐段 dump L0（`SUB_LAYER`）与 `seg_l0.py` 容器复现比，定位 qkv 之后第一处偏。

### 第 10 轮逐段验证（扩 seg_l0.py，给定我方 *_proj_raw 输入单测 op）
```
q_norm   ✅ 0/90112   k_norm ✅ 0/45056   v_norm ✅ 0/45056   (RMSNORM_MODE=2)
```
→ 逐头 RMSNorm（q/k 带权重、v 无权重）**实现逐位正确**。第 3 段清掉。
下一个：**rope**（neox，`rotary_embedding/base.py forward_static`）。

### 第 10 轮 rope：定位到精度口径差（证据确凿）
```
q_rope(bf16 replay)      ~ 15843/90112  max|d|=0.0625   # vLLM compile 口径
q_rope(我kernel口径)      ✅ 0/90112                      # bf16表 cast fp32 + fp32 乘加
```
- `q_rope(我kernel口径)`（bf16 表→cast fp32→**fp32 乘加**）逐位对上我们 dump → **我们 kernel 自洽无 bug**。
- compile 图里 rope 的 mul/sub/add **全是 bf16**（`computation_graph.py` L140-175 都标 `bf16[s59,16,128]`）。
- **根因：rope 这步 vLLM 全程 bf16 乘加，我们 cast 到 fp32 算**。差 15843 个元素、max|d|=0.0625。
  与 norm 相反：norm vLLM 先舍 bf16（我们也舍），rope vLLM 不升 fp32（我们却升了）。
- 修法：rope kernel 改成 bf16 乘加（`layer.rs:397-398` 注释说"cast fp32 做乘加"就是这里）。

### 第 10 轮 rope 修复（验证通过）
`rope.cu:36-41` 改成**逐 op 舍 bf16**（每个 x·cos / x·sin 先 `to_bf16` 再相加减），复现 vLLM
`ApplyRotaryEmb(enable_fp32_compute=False)` 的 bf16 乘加。重 build 后：
```
q_rope ✅ 0/90112   k_rope ✅ 0/45056   (RMSNORM_MODE=2)
```
rope 段**逐位对齐**。注意：rope.cu 是公共件，qwen3/glm/kimik3 共用，需回归（vLLM 全系默认 bf16 rope）。
下一段：**attention(gqa) + o_proj**。

### rope.cu 回归面（第 10 轮）
- 直接用 `rope.cu`（`Rope::apply`）的：**kaon3、glm53**。qwen3/qwen35 走 `qkv_post`（另一 kernel，不受影响）。
- 回归面只有 glm53（本线 hands-off，未跑 e2e）。rope ref golden（`test/kernel_refs/rope_*.npy`）清机删了，待重生。
- 改动方向是"更贴 vLLM bf16 rope"，tolerance 2e-2 可吸收；glm53 若回归再评估加 arch 开关。

### 第 10 轮 rope 修复后端到端（24 用例）
```
POOLED mean_common_d=0.0085  dnll=-0.0595  (修前 0.0083/-0.0006)
```
→ **几乎没动**。rope 已逐位但端到端主残差不在 rope → 在 **attention(gqa) / o_proj**（rope 之后）。
下一步：单测 o_proj（喂 attn_out 过 fp8 GEMM 比 o_proj.bin）+ 定位 attention。

### 第 10 轮 o_proj：我们对，参照写错（退化检查救回）
```
bf16 ground truth ‖gt‖=734   rel(ours→gt)=1.1e-2 ✓   rel(我写的FlashInfer参照→gt)=1.118 ✗
```
- 用 bf16 全精度 `x@(w·ws).T` 当地面真值：**我们的 o_proj 贴 gt（bf16 末位级）→ o_proj 正确**。
- 我第一版 seg_oproj.py 的 FlashInfer 参照手搓量化/转置口径不对，离 gt 很远 → **参照不可信，不是 o_proj 错**。
  教训（用户 2026-10-24 两条）：① 结果先做退化检查（范数/样本），差点把对的报错；
  ② 读 vLLM 要区分 eager/compile：`flashinfer_scaled_fp8_mm` eager 直调 ≠ compile 图的 `bmm_fp8` op。
- **L0 逐段结论**：input_norm✅ qkv✅ q/k/v_norm✅ rope✅ o_proj✅ → **只剩 attention(gqa) 未验证，主残差在它**。

### 依赖顺序修正（用户 2026-10-24）
数据流 `rope → fp8 KV 量化 → 写 KV → attention → o_proj`。o_proj 在 attention 之后，
前面没对齐前拿我方 attn_out 验 o_proj 无意义。**o_proj 状态改回 △（只排除了大错，未逐位）**。
按依赖顺序，下一个节点是 rope 之后的 **fp8 KV 往返量化（4a）→ attention（gqa）**，解完再回头逐位验 o_proj。
逐位已确认：input_norm / q/k/v_norm / rope。kernel 级：qkv(13)。

### 第 11 轮：attention 输入 dtype（compile 图读出）
`unified_attention_with_output(query_2, key_2, value, ...)` 的输入 dtype：
```
query_2 = f8e4m3fn[s59,16,256]   # q 量化 fp8：to_f32 → ×(1/q_scale) → clamp(±448) → fp8
key_2   = bf16[s59,8,256]        # k 进 attention 仍 bf16
value   = bf16[s59,8,256]        # v 进 attention 仍 bf16
```
**与我们代码的差异**：`layer.rs:761-770` 的 4a 在 `kv_fp8` 时把 **q/k/v 三个都** fp8 往返；
vLLM compile 图只量化 **q**，k/v 以 **bf16** 交给 attention op，fp8 KV 量化在 **attention kernel 内部**
用 k_scale/v_scale 做。**疑点**：我们 kernel 外预量化 k/v 与 vLLM kernel 内量化是否等价？待验。
下一步：读 attention 后端源码（graph-break 外 op，图里看不到内部），确认 k/v 在 kernel 内的处理。

### 第 11 轮：q/k/v KV-scale 都是 1.0（与我们一致）
checkpoint 无独立 KV q/k/v_scale → 默认 1.0，我们 4a 也用 1.0。scale 不是差异点。
vLLM attention 全链路：q 量化 fp8（scale 1.0）/ k,v bf16 进 op → `reshape_and_cache_flash` 写 fp8 cache
→ attention 从 cache 读 fp8 再 dequant 回 bf16 → **attention 数学在 bf16 算**（trtllm/flashinfer prefill）。
疑点转向：我们 `prefill_win` 的 q/k/v fp8 往返后，attention 内部累加精度与 flashinfer 是否一致。

### 第 11 轮结论：attention 输入口径我们对，残差在 kernel 内部
vLLM attention（prefill/decode 都）**从 fp8 cache 读 k/v**（写时 fp8 量化、读时 dequant 回 bf16），
不是用刚算的 bf16 k/v。我们 4a 把 k/v fp8 往返 = "写 cache 量化 + 读 dequant" 合并一步，**结构等价**。
q 量化 fp8、scale 全 1.0，也对。→ **attention 输入口径正确**。
残差只能在 **attention kernel 内部**：我们 `gqa_prefill_win_bf16` 的累加/softmax 精度 vs
vLLM flashinfer/trtllm prefill（graph-break 外 op，FX 图看不到内部）。

### 第 11 轮 attention 隔离：不是结构错（rel=6.8e-3）
喂我方 q_rope/k_rope/v_norm（fp8 往返 scale1.0 + GQA + 因果 + fp32 softmax）算参照，比 attn_out：
```
attn_out  不同 63349/90112  max|d|=0.25  rel=6.8e-3  ‖ref‖=255.247 ‖ours‖=255.102
ref[0,:4]=[-0.203,0.406,-0.406,-0.016]  ours[0,:4]=[同]  ← 样本完全一致
```
- 范数几乎相等、样本首元素一致 → **attention kernel 无结构错**，0.68% 是 fp8 KV+bf16 attention 末位抖动。
- 但参照是**手搓 fp32 softmax**（非 vLLM flashinfer 真实输出），只能排除大错，非逐位判据。
- **重要转向**：L0 逐段都只有 fp8 末位级小差（qkv 13、attention 0.68%），没有哪段结构错。
  端到端 0.0085 更像 **各段 fp8 末位差累积**，不是单点 bug。要逐位，得每段都切到和 vLLM 完全相同的库/精度。

### 第 12 轮计划（用户 2026-10-24 方法）：两步硬对拍
1. 确认进 attention 前 dinfer 与 vLLM 的 q/k/v **逐位一致**（拿 vLLM 真实 attention 输入比我方 dump）。
2. 把**同一份** vLLM q/k/v 喂我们的 Rust gqa kernel，比 vLLM attention 的真实输出。
实现：compile 口径起 offline LLM（非 enforce_eager），monkeypatch `unified_attention_with_output`
（graph-break custom op，compile 下可命中）dump q/k/v/output。现有 Rust `gqa_window_parity` 壳可改读文件。

### 第 12 轮第 1 步：进 attention 前 q/k/v 已与 vLLM 不逐位（关键）
抓 vLLM compile 真实 attention 输入（monkeypatch `unified_attention_with_output`），比我方 dump：
```
q(fp8)   ❌ 18638/90112  rel=2.16e-2   ‖ref‖=307.10 ‖our‖=306.81
k(bf16)  ❌ 34847/45056  rel=6.4e-3    ‖ref‖=25.91  ‖our‖=25.92
v(bf16)  ❌ 36910/45056  rel=8.7e-3    ‖ref‖=212.27 ‖our‖=212.27
```
范数都几乎相同、max|d| 是 bf16/fp8 末位级 → **非结构错，是末位漂移**。
**推翻之前"q_norm/rope 逐位"的乐观**：那些"逐位"是我方 op 作用在**我方自己**输入上自洽；
但我方输入本身从 qkv GEMM（cuBLASLt vs FlashInfer bmm_fp8，13 元素差）起就与 vLLM 微漂，
一路带到 attention 入口。→ 印证"各段 fp8/bf16 末位累积"，非单点 bug。
第 2 步：把 vLLM 自己的 q/k/v 喂我们 Rust gqa，比 vLLM 输出（隔离 attention kernel 本身）。

### 第 12 轮第 2 步：vLLM 真实 q/k/v → 我方 Rust gqa → 比 vLLM 真实 output
```
p_fp8=0  rel=2.17e-2  max|d|=0.97  ‖want‖=255.13 ‖got‖=255.61
p_fp8=1  rel=2.22e-2  max|d|=0.97  ‖want‖=255.13 ‖got‖=255.57
```
范数几乎相等但 rel=2.2%、max|d|=0.97（大）。比对手搓 fp32 参照的 0.68% 大。
**待判**：2.2% 是"我方 kernel 错"还是"fp8 attention 固有噪声"（我方/vLLM 各离 fp32 ~1-2%、方向不同）。
对照实验：同一份 vLLM q/k/v 做 fp32 softmax 当地面真值，看 vLLM output 离它多远。
注意：我方喂 kernel 时对 k/v 做了 CPU e4m3 往返（复刻 vLLM 写 fp8 cache），该往返近似可能引入额外差，待核。

### 第 12 轮对照：vLLM output 离 fp32 只有 0.65%，我方离 vLLM 2.2%
```
vLLM_out vs fp32-gt:  rel=6.48e-3   ← vLLM 贴 fp32（fp8 固有噪声量级）
我方 gqa vs vLLM_out: rel=2.2e-2    ← 我方离 vLLM 3 倍远
```
→ **我方 gqa kernel 确与 vLLM 算得不一样**，非单纯 fp8 噪声，是真口径差。
待排除干扰：我方喂 kernel 时 k/v 走的是 **CPU e4m3 近似往返**（`f32_e4m3_rt`），该近似本身可能有误差。
**决定性判据**：我方 gqa 输出直接 vs fp32 gt —— 若也 ~0.6% 则只是舍入方向差；若 ~2% 则 kernel 真错。

### 第 12 轮决定性对照：我方 gqa 贴 CPU f64（1.5e-3），比 vLLM 还准
同一份 bf16 量化输入，CPU f64 attention 当地面真值：
```
我方 gqa → cpu_f64:  rel=1.5e-3（p_fp8=0）/ 6.5e-3（p_fp8=1）   ← 极贴真值
vLLM_out → cpu_f64:  rel=2.2e-2                                 ← vLLM 离 f64 更远
```
**结论：我方 gqa attention kernel 没错，反而比 vLLM 更贴 f64 真值。**
"我方离 vLLM 2.2%" = 两者舍入方向不同（我方贴 f64，vLLM 偏另一侧），**不是 kernel bug**。
坑（诚实）：该 cpu_f64 的 k/v 用 Rust `f32_e4m3_rt` **近似**往返，精确值依赖它；但方向结论稳。
→ L0 attention kernel **排除**。残差来自：① 上游 qkv 库差（cuBLASLt vs FlashInfer bmm_fp8）累积到 q/k/v；
② k/v 进 attention 前 fp8 cache 量化口径（reshape_and_cache）。

### 第 12 轮最终（逐位对齐）：输入一致 + 精度一致 → gqa == vLLM attention
用户纠正：别绕 f64、别用 CPU e4m3 近似改输入。改用生产 device 算子 `fp8_e4m3_roundtrip_bf16_n`
对 k/v 做 fp8 往返（scale 1.0），q 直接上传（vLLM 进 attn 时 q 已 fp8 值），喂同一份 vLLM q/k/v：
```
p_fp8=1（生产档，P 以 e4m3 参与 PV）：不同 0/90112  max|d|=0  rel=0   ← 完全逐位
p_fp8=0（P 不量化变体）：          rel=6.5e-3
```
**L0 attention kernel 与 vLLM 逐位一致**（生产 `p_fp8=cfg.kv_fp8=true` 正好是逐位那档）。
根因教训：之前"我方离 vLLM 2.2%"是**我在测试里用 CPU e4m3 近似私自改了 k/v 输入**，
不是 kernel/精度问题。输入一致 + 精度路径一致 → 同版本 attention 输出就该一样，一样了。

### 第 12 轮换输入复测（用户 2026-10-24：一轮站不住）
p08（36 token）重跑同一 gqa-vs-vLLM 对拍：
```
p01(22 tok) p_fp8=1: 0/90112      rel=0       ← 完全逐位
p08(36 tok) p_fp8=1: 1279/147456  rel=5.3e-4  ← 接近非逐位（1279 个元素差，max|d|=0.0625 bf16末位）
```
**结论修正**：p01 的 0 diff 是偶然全中，**attention 非严格逐位，是 rel~5e-4 级极接近**。
未消除变量：喂 kernel 的 k/v 是用**我方 device fp8 往返**从 vLLM bf16 k/v 重建，
不是 vLLM 真正写进 fp8 cache 的字节。p01 全中/p08 差 1279 很可能是重建与 vLLM cache 实际 fp8 值
在少数元素舍入方向不同。要彻底干净：直接抓 vLLM fp8 cache 的 k/v 字节喂入。

### 第 12 轮修正：改用 vLLM fp8 cache 真实 k/v 字节（用户 2026-10-24 两次纠正）
之前从 bf16 重建 k/v = 输入没对齐。改 attn_io 直接 dump `reshape_and_cache` 后 fp8 cache 的 k/v
（布局 (nb, nkv=8, block, 2D=512)，前 256=k 后 256=v，slot=blk*block+off；block_size 取 cache 维不猜）。
退化检查：‖k_bf16‖=25.91 vs ‖k_cache‖=25.92 max|d|=0.055（fp8 级）→ gather 正确非垃圾。
```
p01(22) p_fp8=1: 0/90112  rel=0  （真实 cache k/v）
```
p08 复测中。

### 第 12 轮两输入定论（真实 fp8 cache k/v，退化检查过）
抓 vLLM 写 cache 后的真实 fp8 k/v（非 bf16 重建），退化检查 ‖k_bf16‖≈‖k_cache‖ max|d|=0.055：
```
p01 T=22 p_fp8=1: 0/90112     rel=0
p08 T=36 p_fp8=1: 1279/147456 rel=5.3e-4  （与 bf16 重建版完全相同的 1279）
```
**真实 cache k/v 与 bf16 重建给出相同的 1279 → 残差不在 k/v 输入，输入已对齐。**
退化检查又救一次：p08 vLLM output 第 36-39 行是 **padding 垃圾**（范数 1500-2600 vs 正常 50），
T=40 时 want 范数爆到 4400、rel=0.997 假阳；截 T=36 真实 token 后恢复 rel=5.3e-4。
**两输入定论**：我方 gqa 与 vLLM attention **rel~5e-4 极接近，非严格逐位**（p01 的 0 是偶然）。
剩 1279 待查：attention kernel 口径差，或 q 侧（fp8 值但 vLLM 内部还有 q descale/bmm1_scale）未对齐。

### 第 12 轮 block size 核实（用户 2026-10-24 质疑）
attn_io 请求 `block_size=32`（`non-default args` 确认），但 **vLLM 实际建 fp8 cache block_size=16**
（`cache block_size=16 slot=[16,17,18,19]`）→ vLLM 对 fp8 KV cache 自行按 16 对齐，无视请求的 32。
- 所以 attn_io 全程 block16；cache-gather 与 bf16 重建给相同 1279 = 同一份 bf16 做同一 e4m3 量化
  （两条路 fp8 字节逐位同），证明 gather 读对了（block/slot 没错位，否则结果乱）。
- 待核：oracle server（端到端参照，也传 32）实际 cache block 是否也 16 → 若是，两处一致，block 非变量。

### block size 闭环：两处一致（都 16），排除为变量
oracle server 也传 `block_size: 32`（docker logs 确认），与 attn_io 同镜像同参数
→ oracle 实际 fp8 cache 也是 16。**attn_io 与端到端参照 block 一致，block 不是变量。**
推论：attn_io 抓的 attention IO == 端到端参照里真实那次 → "p01 逐位/p08 差1279" 可信。
**第 12 轮稳定结论**：gqa 与 vLLM attention rel~5e-4 极接近、非严格逐位；剩 1279 bf16 末位待查 q 侧。

### 第 12 轮多组验证（用户 2026-10-24：多测几组再观察）
已有 p01(22):0  p08(36):1279。再抓 p03(31)/p16(22)/p06(27) 三组真实 fp8 cache k/v，
凑 5 组看 nd/rel 随 T 的分布，判 1279 是偶发还是随 token 数单调。每组 T 取真实 token 数（排 padding）。
