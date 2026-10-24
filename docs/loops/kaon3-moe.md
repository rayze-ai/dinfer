
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
