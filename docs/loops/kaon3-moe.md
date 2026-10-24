
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
