# 2026-10-15 测试记录

| 文件 | 内容 | 结论 |
|---|---|---|
| `p5-dsa-parity.md` | tiny DSA GLM `dsa_parity` 三道测试（断言重写后、fixture 16:22 重生成）实跑结果 + 下一步候选 | **3/3 FAIL（阈值类）**：self 可判定 28/256（需 ≥ 64）、hf-topk 路由 near-tie 45/256（需 ≤ 25）、topk row 124 分差刚超 band；可判定行上误差全达标（0.0078 / 0.0098） |
| `../tiny_glm_dsa/` | `/data/ref/venv-tf519/bin/python scripts/verify/gen_tiny_glm.py test/tiny_glm_dsa --layers 3 --first-k-dense 1 --dsa` 产出（不进 git） | 见 `../2026-10-14/README.md` 同名行 |
