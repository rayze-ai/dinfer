
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
