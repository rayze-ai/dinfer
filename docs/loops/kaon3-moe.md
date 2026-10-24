
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
