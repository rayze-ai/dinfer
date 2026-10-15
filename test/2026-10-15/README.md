# 2026-10-15 测试记录

| 文件 | 内容 | 结论 |
|---|---|---|
| `p5-dsa-parity.md` | tiny DSA GLM `dsa_parity` 三道测试：从 3/3 FAIL 到 3/3 PASS 的三个根因（L0 平局翻转污染 L2 索引器输入；路由平局 → 用实测 hidden 漂移过滤；HF 参考不前缀稳定）+ 最终数字 | **3/3 PASS**（19:10）；`p5.sh` 全量日志 `p5.log` |
| `p5.log` | `scripts/verify/p5.sh` 全量输出（p0–p4 回归 + DSA fixture 重生成 + `dsa_parity`） | **P5 PASS**，exit 0（19:25） |
| `../tiny_glm_dsa/` | `/data/ref/venv-tf519/bin/python scripts/verify/gen_tiny_glm.py test/tiny_glm_dsa --layers 3 --first-k-dense 1 --dsa` 产出（不进 git） | 见 `../2026-10-14/README.md` 同名行 |
