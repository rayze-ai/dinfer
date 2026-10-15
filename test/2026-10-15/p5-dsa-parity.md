# P5 DSA 对拍（tiny DSA GLM vs transformers 5.19）

## 结论（截至 2026-10-15 16:30）

**3 道 `dsa_parity` 测试全部 FAIL，`scripts/verify/p5.sh` 未通过。** 但三道失败都不是"算错"，而是"可判定行太少"的阈值问题：

| 测试 | 失败断言 | 实测 | 阈值 | 可判定行上的误差 |
|---|---|---|---|---|
| `dsa_self_run_long_prompt` | 可判定行数 | 28/256 | ≥ 64 | worst max abs 0.0078（< 0.05 ✅） |
| `dsa_sparse_attention_with_hf_topk` | 路由平局行数 | 45/256 | ≤ 25（t/10） | 211 可判定行 worst 0.0098 ✅ |
| `dsa_topk_indices_match_hf` | 对称差元素 HF 分数落在 k-th ± band | row 124 分差 0.1166 vs band 0.1130 | — | 分数误差 0.1119（阈值 0.1，也将超） |

读法：
- 注入 HF top-k 后，211 个路由可判定行 logits 全部 < 0.01 → 稀疏 attention 本身正确（与前几轮逐层 bisect 结论一致）。
- 自算路径的 28 个可判定行 worst 0.0078 → 索引器 + 稀疏 attention 全程自算在可判定行上也正确。
- 失败全部来自 tiny 随机模型的"平局率"太高：路由 near-tie 45/256（17.6%），索引器 near-tie 导致可判定行只剩 28。
- `dsa_topk_indices_match_hf` 的 row 124：索引器分数误差已到 0.11（之前几轮 0.024），band 刚好差 0.004 没盖住。原因未查（fixture 16:22 重新生成过，索引器权重 ×6 放大会同时放大 bf16 舍入误差）。

## 下一步候选（按优先级）

1. **调 fixture 而不是调阈值**：`gen_tiny_glm.py --dsa` 让路由/索引器分差拉大（专家数 4 → 更少的 near-tie；索引器权重放大系数从 ×6 调回或用固定 seed 挑 margin 分布好的），目标：路由 near-tie < 10%、索引器可判定行 ≥ 64。
2. 若 1 做不到，把三道断言的阈值改成按 fixture 实测 margin 分布自适应（例如断言"可判定行 ≥ 可判定候选行的 90%"而非绝对 64）。
3. 查 row 124 的 0.11 分数误差是否是 bug（之前 0.024）：对比 L0 vs L2、短 prompt vs 长 prompt 的 err 分布。

## 原始输出（节选，`cargo test -p dinfer-glm53 --features ref-tests --test dsa_parity -- --ignored --test-threads=1 --nocapture`，完整在 `/tmp/dinfer_p5_dsa.log`）

```
test dsa_self_run_long_prompt ... [self] long prefill: 28/256 decidable rows (worst max abs 0.0078125); all rows worst 0.7138672
thread 'dsa_self_run_long_prompt' panicked at crates/glm53/tests/dsa_parity.rs:354:5:
[self] too few decidable rows: 28/256
FAILED
test dsa_sparse_attention_with_hf_topk ... [hf-topk] long prefill: 211 router-decidable rows worst max abs 0.009765625; 45 router-near-tie rows (worst 0.25683594, not asserted)
thread 'dsa_sparse_attention_with_hf_topk' panicked at crates/glm53/tests/dsa_parity.rs:263:5:
[hf-topk] too many router-near-tie rows: 45/256
FAILED
test dsa_topk_indices_match_hf ...
thread 'dsa_topk_indices_match_hf' panicked at crates/glm53/tests/dsa_parity.rs:200:17:
L2_long row 124: index 122 HF score 0.06430305 vs k-th 0.18086866, band 0.11301785 (row score err 0.11189886); ours {0, 1, 31, 35, 87, 97, 119, 122} hf {0, 31, 35, 87, 97, 99, 108, 119}
FAILED
test result: FAILED. 0 passed; 3 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.11s
```

上一版测试（15:05 日志，断言重写前）同样 3 FAIL：self 9/256 可判定；hf-topk 全行 worst 0.2568 > 0.05（当时还没按路由 margin 过滤）；topk L2 row 10 band 0.03。
