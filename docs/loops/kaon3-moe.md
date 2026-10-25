
## 卡点（第 1 步，用户 2026-10-25）：MoE hook 抓不到 vLLM 输出
monkeypatch 三种都 grabbed=[]：① MoERunner.forward ② MoERunner._forward_impl ③ 重注册 vllm::moe_forward CUDA impl。
根因：compile 缓存 6c6aae890c 已把 moe_forward 实现引用烘进编译图，运行时 patch/重注册对已缓存图不生效。
（attention 能抓是其 custom op 边界每次 eager 调 _orig；MoE 这条被 inductor 吞得更深。）
可行路（下次选一）：
  A. 清 compile 缓存或换 cache_dir 让它带 patch 重编（慢，但能命中）；
  B. enforce_eager=True 抓 MoE 输入/输出——MoE 是 NVFP4 专家 GEMM+路由，eager/compile 的 MoE 数值应接近
     （不像 attention 有 FEXP 差异），可先用 eager 定位"差在路由还是专家 GEMM"，再回 compile 精修；
  C. 从我方 moe_e{eid}_* 逐专家 dump + router_logits，直接对 vLLM 的专家权重/路由做隔离复现（不抓 vLLM 运行时）。
推荐 B 先定位（快），再按 qkv/attention 同法深入。

## 第 18 轮定位链（用户 2026-10-25 loop）
```
router_logits 差 1.1% → 3/36 行 topk8 选错专家 → 选错行 MoE 输出大偏 → MoE 整体 rel=1.9%
```
- MoE 是 L0 第一个**系统性非末位**差异（qkv/attention 是末位，这个是 1.1% 结构差）→ 待修 bug。
- router = norm(add)·root_size·scale → proj(bf16 GEMM)→f32。注释(layer.rs:905)已对齐 root_size bf16 舍入、换过 GEMM。
- 关键机制：topk 的**离散翻转**放大——router_logits 末位差在 topk 边界把专家选翻转（非连续传染）。
- 下一步：查 router_logits 1.1% 来自 ① router 自身口径（GEMM 累加/norm/scale）还是 ② 上游 add 残差流带入。
  方法：抓 vLLM 的 router 输入（add 或 router_in），比我方；或比 router GEMM 隔离（同输入同权重）。
- 数据：/hf_data/yaochen/moe_e08（vllm_moe_hidden/logits/out），我方 /w/sd_now（router_logits/router_in/moe_y/moe_e*）。

## 第 18 轮续：根因往上收一层——MoE 输入(pre_ff2_norm)就偏 2.3%
vLLM moe_hidden vs 我方 pre_ff2_norm：rel=2.3%（范数都 92.5，分布偏）→ **MoE 输入已偏，非 router 自身**。
pre_ff2_norm = rms_norm(add, pre_ff_norm_2)；add = post_attn_norm(o)+embed（o=attention 输出）。
→ add 残差流带着偏差进来。**印证 attention 偏差向下游传**（但传到 pre_ff2/router，经 topk 离散放大）。
更正：seg_l0 测 pre_ff2_norm "逐位" 是用**我方自己 add** 当输入（自洽）；对 vLLM 的 add 则偏 2.3%。
根因链：add 偏 → pre_ff2_norm 2.3% → router_logits 1.1% → topk 翻转 3 行 → MoE 1.9%。
下一步：抓 vLLM 的 add（= attention 后残差，o_proj+post_attn_norm+embed）比我方，定位 add 偏的来源。

## 第 18 轮关键（用户 2026-10-25）：先验 MoE 的 eager vs compile 差异
用户："Eager 我 moe 已经对齐了" —— 但我 eager 抓的 vLLM MoE vs 我方 moe_y 差 1.9%，矛盾待解。
方法：VLLM_DISABLE_COMPILE_CACHE=1 强制重编译（patch 能被吞进图），抓 compile 口径 vLLM MoE，
与 eager 的 vLLM MoE 比：
  - 若 eager==compile → MoE 两模式无差异，eager 数据可直接用，1.9% 是我方真 bug；
  - 若有差异 → 看差异结构，eager 不能当 compile 判据，1.9% 里混了模式差。
产物：/moe_cc08（compile-nocache）vs /moe_e08（eager）vs /sd_now（我方）。

## 第 18 轮关键结构事实：MoE 在 compile 下被内联进图，抓不到
`splitting_ops` 只含 attention 类（unified_attention_with_output 等），**不含 moe_forward**。
→ compile 下 MoE 被 inductor 完全追踪/融合进图，**不走任何可 patch 的 Python 函数**（类方法/模块函数/重注册 op impl 全失效，含 DISABLE_COMPILE_CACHE 重编译也 grabbed=[]）。
→ eager 下 MoE 走 Python 专家实现（patch 命中，抓到 vLLM MoE，vs 我方差 1.9%）。
**回答"MoE eager vs compile 有无差异"**：执行路径不同（eager=Python 专家循环，compile=inductor 融合图），
不能默认相等；但 compile 口径 MoE 输出**无法 hook 抓取**——"抓 vLLM compile MoE 直接比"这条路不通。
待解矛盾：用户"eager moe 已对齐"，但我 eager 抓的 vLLM MoE vs 我方 moe_y 差 1.9% → 需确认比对口径
（是否 moe_y 定义 / 加权和 / post_ff_norm_2 前后 对错了量）。

## 第 19 轮（用户 2026-10-25 纠正）：MoE 隔离正确做法=同输入比输出
之前错在：拿"我方 moe_y(我方残差流)"比"vLLM moe_out(vLLM 残差流)"——输入不同，差 1.9% 必然，不能判 MoE kernel。
正确：喂 vLLM 的 MoE 输入 → 我方完整 MoE → 比 vLLM moe_out（同输入比输出）。
回看 vLLM `_forward_impl`(moe_runner.py:830)：收 hidden_states + router_logits，但 **gate 不为 None 时用
hidden 重算 router_logits**（self.gate(hidden) 或 fused gate，忽略传入 logits）→ MoE 真正输入只是 hidden_states。
已抓：/moe_e08 的 vllm_moe_hidden[36,2816]（MoE 输入）、vllm_moe_out[36,2816]（输出）。
下一步：我方加"喂外部 hidden 跑完整 MoE"入口（或 kaon3_probe 支持），喂 vllm_moe_hidden，比 vllm_moe_out。
待确认：我方路由是否也从 hidden 重算（与 vLLM gate 口径一致）。

## 第 19 轮：回看 vLLM 确认 MoE 干净隔离方案
- gemma4.py:371 `Gemma4MoE.forward(x, router_logits)` → `self.experts(x, router_logits)`。
  两输入：**x=专家 hidden（moe_hidden）+ router_logits（外部算好传入）**。
- `_forward_impl` 的 `self.gate` 对 Gemma4=None（gate 在 Gemma4Router 外算好传入），**不重算** → 用传入 logits。
- router 预处理(norm·root_size·scale) **只给 router 不给专家**（gemma4.py:265 注释），与我方一致，结构没搞错。
- **干净隔离**：喂我抓的 x(moe_hidden)+router_logits(moe_logits) 进我方专家，比 moe_out。
  纯测专家 GEMM(NVFP4)+topk+加权和，路由用 vLLM 的（排除路由差）。
- 待做：我方加入口接收外部 hidden+logits 跑专家（kaon3_probe --moe-iso 或测试）；已有数据 /moe_e08。
- 注：routing 是 gemma4_routing_function（softmax over ALL → topk → renormalize），top_k=config.top_k_experts。

## 第 20 轮：loop 固化——核心约束 compile 口径 + 当前卡点
**目标**：L0 MoE 在 **compile 口径**（oracle 是 compile）下与 vLLM 对齐；端到端 mean_common_d 的 MoE 残差清掉。
**铁律**：判据必须 compile 口径。eager 抓的 MoE io 不能直接当 compile 判据（犯过多次"eager 当 compile"坑）。

**当前卡点**：compile 下 MoE 内联进图（不在 splitting_ops），输入/输出 hook 抓不到
（3 种 patch + VLLM_DISABLE_COMPILE_CACHE=1 重编 均 grabbed=[]）。eager 能抓（/moe_e08）但口径存疑。

**迭代步骤（机器判定，按序）**：
1. 查 vLLM FusedMoE 专家 kernel 在 eager vs compile 是否同一实现（dispatch 有无换 kernel）。
   - 若同一 kernel（MoE 无 FEXP 类 compile 专属近似，很可能）→ eager io 可当 compile 判据，用 /moe_e08。
   - 若不同 → 必须拿 compile io，走步骤 2b。
2a. 我方加 `--moe-iso` 入口：喂外部 x(moe_hidden)+router_logits → 只跑专家(NVFP4 GEMM+topk+加权和) → 比 vllm_moe_out。
2b. 若需 compile io：从相邻可抓点夹（pre_ff2_norm 输入 / post_ff_norm_2 后输出），或用 oracle 端到端 prompt_logprobs 粗判。
3. 定位差在 ① NVFP4 解包/scale ② topk/renormalize ③ 专家 GEMM 累加/per_expert_scale 折法。按 qkv/attention 同法修。
**停止条件**：compile 口径下 MoE 输出 vs vLLM 逐位或仅 bf16 末位（fp64 判决），且端到端 MoE 残差消失。
**上限**：每轮 ≤10 迭代；单点超 3 轮无结论写卡点换方向。
**数据**：/moe_e08(eager vllm_moe_hidden/logits/out)、/w/sd_now(我方 moe_y/moe_e*/router_logits)。
**下一步（接手第一件事）**：执行步骤 1——查 FusedMoE eager vs compile kernel 是否同一。

## 第 20 轮（卡点解开）：moe_forward 是 opaque custom op → eager==compile
moe_runner.py:190 注释明示 **"moe_forward being opaque custom ops is a load-bearing assumption"**。
不透明 custom op：inductor **不追踪其内部**，compile 下按原样 eager 执行 `_moe_forward`。
→ **compile 与 eager 跑同一个 `_moe_forward` 实现 → eager 抓的 MoE io 就是 compile 口径**（之前"内联进图"判错）。
→ 用户"eager moe 已对齐"可直接作 compile 判据；"抓不到 compile io"是伪问题（eager io 即 compile io）。
patch 没抓到 compile 版：因 direct_register_custom_op 注册时固化 _moe_forward 引用，运行时 patch 不改已注册 op；
但 eager 跑也走同一 op，故 eager patch 命中（已抓 /moe_e08 成功）。
**结论**：MoE 隔离可直接用 eager 抓的 /moe_e08（= compile 口径）。回到步骤 2a：加 --moe-iso 喂 x+logits 比 moe_out。

## 第 20 轮结果：MOE_ISO 同输入隔离 → rel 1.9%→0.43%
喂 vLLM moe_hidden+logits 进我方专家，比 vLLM moe_out：
```
不同输入(旧):  rel=1.9e-2
同输入(iso):   rel=4.3e-3  max|d|=0.25  ‖vllm‖=365.93 ‖ours‖=366.69  36/36 行差
```
→ 证实 1.9% 大部分是**输入差（残差流偏）带入**，非 MoE kernel。剩 0.43% 是 MoE kernel 自身系统性差。
下一步（拆 MoE 内部，定位 0.43% 在哪步）：
  ① topk 选择：我方 topk_ids vs vLLM（用同 logits，应完全一致，否则 topk/renorm 口径差）；
  ② 专家 GEMM(NVFP4 gate/up/down)：我方 moe_e{id}_* vs vLLM 专家中间量；
  ③ 加权和 / per_expert_scale 折法 / renormalize。
数据：/w/sd_iso（iso 的 moe_y + moe_e*）、/moe_e08（vLLM）。

## 第 21 轮：MoE 0.43% 定位到 topk_w（路由权重），非专家选择/GEMM
同 vLLM logits 下：
```
topk 专家选择：36/36 全一致（选对）
topk_w 路由权重：max|d|=8.2e-3  ‖our‖=2.504 vs ‖vllm‖=2.492（偏大）
```
→ MoE 0.43% 落在 **topk_w 口径**（权重差按比例传到加权和），非专家 GEMM/选择。
vLLM 路由(gemma4_routing_function_torch)：softmax(全体)→one_hot累加→renorm→×per_expert_scale，全程 fp32。
下一步：查我方 moe_route_softmax_f32 kernel 的 renorm/per_expert_scale 口径 vs vLLM（哪步舍 bf16 或折法不同）。

## 第 21 轮根因：route kernel 用 sigmoid，vLLM gemma4 用 softmax
moe.cu:30 route_softmax_kernel 里 `sc = 1/(1+exp(-logit))`（**sigmoid**），再 topk/renorm/scale。
vLLM gemma4_routing_function：`softmax(全体 logits)` → one_hot 累加 → renorm → ×per_expert_scale。
**sigmoid vs softmax 不同**→ topk 专家选择因单调性相同（36/36），但**权重值不同**→ topk_w 差 8e-3 → MoE 0.43%。
注意：layer.rs:977 注释早标"不是 sigmoid（DeepSeek/GLM/K3 口径）"，但 kernel 名叫 moe_route_softmax 实际却是 sigmoid——名实不符。
修法：kaon3 路由权重走 softmax 口径（全体 softmax 的概率，不是 sigmoid）。moe.cu 公共件，需按口径分支（K3/GLM=sigmoid, gemma4=softmax）。
待确认：kaon3 调的 moe_route_softmax_f32 是否真走这个 sigmoid kernel，还是有别的 softmax 变体没接上。

## 第 21 轮续：route_softmax kernel 用 ex2.approx + rcp.approx（近似指数/倒数，同 FEXP 性质）
moe.cu route_ex2=`ex2.approx.ftz`、route_rcp=`rcp.approx.ftz`。vLLM gemma4 是 torch 精确 softmax+精确除。
→ topk_w 差 8e-3 = 近似指数/倒数（与 attention FEXP 同性质）。已有 ROUTER_RCP_EXACT=1 开关（倒数精确），ex2 无开关需加。

## 第 21 轮：排除 exp/rcp 近似——强制精确 exp 后 topk_w 仍差 8.2201e-3（一字不差）
ROUTER_RCP_EXACT=1 + 强制 exp2f 精确，topk_w max|d| 仍 8.2201e-3 不变 → **8e-3 不是 ex2.approx/rcp.approx 造成**。
两边同 logits（iso 灌），差只能在 softmax/renorm **数学口径**本身。
重新定位方向：① 我方只对 topk 的 exp 做分母 vs vLLM 全体 softmax 概率再 renorm（数学应等价，需核）；
② per_expert_scale 折入顺序/dtype；③ renorm 的 1e-20 vs vLLM where(>0)。
下一步：Python 精确复现我方 kernel 口径 vs vLLM 口径，在 f64 下比，定位哪步口径差。

## 第 21 轮：topk_w 8.2e-3 = f32 实现差（非口径、非 exp 近似）
- 退化检查救：tw3 脚本 id 对齐写错报 rel=3.8%，作废；按 oid 顺序直接比得 rel=1.2% max|d|=8.2e-3（与直接比 vLLM 一致）。
- f64 口径验证：我方 kernel 口径 == vLLM 口径（1e-16）→ 数学没错。
- row0 样本：我方 0.4021 vs f64 0.40838（差 0.006，~1.5%）。我方与 vLLM 都离 f64 ~1%，方向不同。
- 根因推断：f32 下 vLLM 全体 128 softmax 概率 gather+renorm vs 我方只 topk 8 个 exp 求和分母，
  数学等价但 f32 归约舍入不同 → topk_w 差 8.2e-3（2% 相对）。这是真实实现差，非 bug/口径。
- 影响：传到 MoE out rel=4.3e-3（0.43%）。是否修需权衡（改我方 kernel 走全体 128 softmax 复刻 vLLM 归约）。

## 第 22 轮：推远程 + 继续对齐（用户 2026-10-25）
已 git push origin dev（217 commits，至 754d1d0）。
**L0 对齐总表**（compile 口径 vs vLLM）：
| 段 | 状态 |
|---|---|
| input_norm / q/k/v_norm / rope | ✅ 逐位 |
| qkv GEMM | ✅ 逐位（TABLE splitk=1） |
| o_proj / gate_up / down | ✅ 逐位（fallback splitk=1） |
| attention | ✅ 差2（去 FEXP，cfg 464）；残2个 bf16 格点 TODO |
| MoE 专家选择 | ✅ 36/36 |
| MoE topk_w | ⚠️ 差8.2e-3（f32 归约：我方 topk 分母 vs vLLM 全体128 softmax）→ MoE out 0.43% |
| MoE 专家 GEMM(NVFP4) | ❓ 未单独隔离（topk_w 修齐后再看残差是否全消） |
**下一步（继续对齐）**：
1. 改 route kernel：topk softmax 分母改为**全体 128 专家 exp 和**（复刻 vLLM torch softmax 归约），对齐 topk_w。
   moe.cu 公共件——K3/GLM 走 sigmoid(moe_route_f32) 不受影响；只动 moe_route_softmax_f32(gemma4)。
2. 修后重测 MOE_ISO：topk_w 是否降到末位、MoE out rel 是否 <1e-3。
3. 若 MoE out 仍有残差 → 隔离专家 NVFP4 GEMM（喂同 hidden+topk 比专家输出）。
4. 全 L0 对齐后：端到端 24 用例 mean_common_d 应显著下降，记入。

## 第 22 轮判决：我方 route kernel 偏，vLLM 贴真值（fp64 定）
用 vLLM **自己的** gemma4_routing_function_torch 算 topk_w：
```
vLLM 真函数 vs f64真值: max|d|=4.6e-8  ← vLLM 就是真值
我方 vs vLLM:           max|d|=8.2e-3  ← 我方偏
row0 vLLM: [0.40838,0.30948,...]  我方: [0.4021,0.31315,...]（第1偏小/第2偏大）
```
→ **是我方 route kernel 算偏 8e-3，不是末位/vLLM**。改全体Z归约、精确exp 均无效（topk_w 两版 max|d|=3e-8 没动）。
疑点转向：我方 softmax 用 `ex2(x·log2e_f32)`（2为底，log2e 乘法 f32 舍入），vLLM torch softmax 用 `expf`（e为底）。
f32 下 `exp2f(x·log2e)` ≠ `expf(x)`。下一步：route_ex2 精确分支改直接 `expf(x_原始)`（不乘 log2e 走 2 底），对齐 vLLM。

## 第 22 轮卡点（诚实记录）：连续 4 改 topk_w 零变化
改了 ① exp 近似→精确 ② 全体 Z 归约 ③ expf(e底) ④ ROUTER_RCP_EXACT=1，我方 topk_w 一字不变（sd_ex/sd_z/sd_e2 完全同，row0=0.4021）→ **这些都不是 8e-3 的因**。
排除：sc.logits 是 f32、iso 灌 f32 无降精度；调用链 layer→moe_route_softmax_f32→moe.cu 确认无误。
我方 row0=[0.4021,0.31315,..] vs vLLM=[0.40838,0.30948,..]：第1偏小/第2偏大 → softmax 相对大小差，非末位。
**下一步（不猜，插桩）**：kernel dump raw[k]（softmax 概率，renorm 前）比 vLLM router_probabilities，
定位 exp 出来就差 还是 renorm/pes 后才差。疑点剩：max 用全体 vs vLLM、分母 Z 的实际值、per_expert_scale 乘入点。
注意：ROUTER_RCP_EXACT 是否真生效待核（static getenv 缓存；probe 新进程应读到，但改 exp 无效可能它压根没进精确分支）。

## 第 22 轮重大转折：口径对，kernel 实现有 bug
Python f32 复刻我方口径(m=max全体, Z=Σexp, prob=exp/Z, renorm, ×pes) vs vLLM：**max|d|=0 完全一致**（row0=0.40838）。
但我方 **kernel 实际跑出 0.4021（偏 8e-3）** → **CUDA kernel 实现与应算口径不符，是实现 bug，不是口径/精度**。
- 连续 4 改无效的原因找到：我加的"全体 Z"里 raw=exp·invZ 后面又 renorm=Σraw 再除，invZ 被消掉 → 等于没改。
- 真正 kernel 算 exp_k/Σ_topk exp（topk 分母），数学该 =0.40838，但实际 0.4021 → lane 蝶形归约(line 236-240)或 raw 计算有 bug。
**下一步**：kernel 插桩 dump raw[k]（renorm 前）与 Σ，对比 Python；查 lane 蝶形求和（ept 分配、K=8 的归约是否漏项/重复）。

## 第 22 轮：kernel 插桩 dump raw/Z/renorm（ROUTE_DBG）
moe.cu route_softmax_kernel 加 printf dump t=0 的 exact/m/Z/renorm/raw0/tw0/id0/sc0。
对比 Python f32 复刻（row0 应 m/Z/raw0 一致、tw0=0.40838）。若 kernel 的 Z 或 raw0 就偏 → 定位到 exp/归约；
若 raw0 对但 tw0=0.4021 → renorm/pes 环节 bug。需 #include <cstdio>（已加）。

## 第 22 轮真相：topk_w 的 8e-3 是参照错（我方 pes 折进 down，topk_w 本就不含 pes）
kernel 插桩：m/Z 对，但 **sc0=1.0** → route kernel 没乘 per_expert_scale。
原因：`moe_combine_f32()` 默认 true → `pes_route=None` → **per_expert_scale 折进专家 down 的 global scale**（设计如此，layer.rs:324），不在 route 乘。
→ 我方 topk_w **本就不含 pes**，而我的 cmp_topk/Python 复刻都 `*pes[tid]` → **参照比较错**（又一个 tw3 式参照 bug）。
pes[79]=1.0156，0.40838/0.40209=1.0156 吻合 → topk_w 的 8e-3 = 少乘的 pes，是假差。
**真实残差**：MoE out（moe_y，含 down 的 pes）rel=0.43%。topk_w 对齐（去掉参照的 pes 后应逐位或末位）。
教训（用户 2026-10-24 反复强调）：结果先做退化检查/对口径；连续 4 改无效早该怀疑参照而非实现。
下一步：重验 topk_w（参照不乘 pes）；再定位 MoE out 0.43% 真正来源（专家 NVFP4 GEMM / down 的 pes 折法）。

## 第 22 轮结果：路由对齐（参照去 pes 后 topk_w max|d|=4.5e-8）
```
topk_ids: 36/36 一致   topk_w: max|d|=4.47e-8（浮点末位）
```
→ **MoE 路由完全正确**。之前 8e-3 全是参照多乘 per_expert_scale（我方折进 down）的假差。
剩 MoE out rel=0.43% 必在**专家 NVFP4 GEMM（gate/up/down）或 pes 折进 down**。路由排除。
下一步：隔离专家——喂同 hidden + 我方 topk（已对）跑专家，比 vLLM moe_out；或逐专家 moe_e{id}_* 比。
注意：route kernel 现有"全体Z+expf"改动数值等价（不影响），保留；性能未测。

## 第 22 轮收口：MoE 残差 0.43% 锁定在 NVFP4(W4A4) 专家 GEMM
- 我方 experts = NVFP4（packed 4-bit + block scale + global），W4A4 最低精度。
- 路由已逐位对齐（topk_w 4.5e-8，专家 36/36）→ moe_y 0.43% 只能是专家 GEMM（gate/up/down NVFP4）+gelu+加权和。
- 判断：NVFP4 是 4-bit 权重+激活，走自写 kernel，累加/解包末位差天然大于 FP8（dense 能逐位是 FP8+cuBLASLt algo 可选）。
  0.43% 对 W4A4 属合理噪声量级，性质同 attention bf16 末位（精度更低故更大），大概率非 bug。
- 待坐实（可选，较重）：解包 NVFP4 做 f64 专家复现，三方判决（我方/vLLM/f64），看是否 fp4 噪声底。
**MoE loop 阶段成果**：1.9%→路由逐位 + 残差 0.43% 定位到 NVFP4 专家 GEMM；三个"参照 bug"（tw3/pes/布局）均被退化检查/插桩抓出。

## 第 23 轮：MoE 0.43% 均匀分布（指向 NVFP4 噪声，非 bug）+ 自检教训
moe_y vs vLLM 真 out（同输入同路由）每行 rel：最大 0.0045 最小 0.0025，**36 行全在 0.3-0.45% 窄带**，不集中 → 均匀铺开 = NVFP4 量化末位噪声（bug 会集中在特定行/专家）。性质同 attention 1-ULP。
端到端 24 用例 mean_common_d 仍 0.0086（L0 全修复后）——说明端到端残差主要由 NVFP4 专家噪声 + 各段 fp 末位累积，无单点 bug 可再降。
**自检（用户 2026-10-25 批评"参照反复出错怎么修正对"）**：
1. 根因=先写参照再对，没先用 f64/已知值自检参照 → tw3/pes/o_proj/布局 4 次假差。改：参照必须先和已知真值对齐确认可信，再判被测。
2. 连续改无效（route 4 改 topk_w 不变）第 1 次就该停下查参照，不该改到第 4 次才插桩。
3. 专家选择 36/36 一致时范围已很窄（只剩权重值/量化），我绕远了。
坐实下一步：解包 NVFP4 做 f64 专家复现，三方判决（我方/vLLM/f64）确认 0.43% 是 fp4 噪声底。
