# 2026-10-15 测试记录

| 文件 | 内容 | 结论 |
|---|---|---|
| `p5-dsa-parity.md` | tiny DSA GLM `dsa_parity` 三道测试：从 3/3 FAIL 到 3/3 PASS 的三个根因（L0 平局翻转污染 L2 索引器输入；路由平局 → 用实测 hidden 漂移过滤；HF 参考不前缀稳定）+ 最终数字 | **3/3 PASS**（19:10）；`p5.sh` 全量日志 `p5.log` |
| `p5.log` | `scripts/verify/p5.sh` 全量输出（p0–p4 回归 + DSA fixture 重生成 + `dsa_parity`） | **P5 PASS**，exit 0（19:25） |
| `../tiny_glm_dsa/` | `/data/ref/venv-tf519/bin/python scripts/verify/gen_tiny_glm.py test/tiny_glm_dsa --layers 3 --first-k-dense 1 --dsa` 产出（不进 git） | 见 `../2026-10-14/README.md` 同名行 |
| `p6-glm-serve.md` | P6 四步（tokenizer/chat 模板 HF 对拍、解析器单测、server e2e、MTP 对拍）结论 + 关键数字 + 日志节选 | **P6 PASS** |
| `p6.log` | `scripts/verify/p6.sh` 全量输出（p0–p5 回归 + P6 a/b/c/d） | exit 0 |
| `../tiny_glm_dsa_fullvocab/`、`../tiny_glm_dsa_mtp/` | `gen_tiny_glm.py --dsa --vocab 154880`（e2e 用）/ `--dsa --mtp`（MTP 用）产出（不进 git） | — |
| `q1-qwen3.md` | Q1：Qwen3-0.6B **真权重**（3090）— GQA kernel 对拍、tokenizer/模板 9/9、prefill logits、greedy 32 全对、双请求混批、真模型 e2e | **Q1 PASS** |
| `q1.log` | `scripts/verify/q1.sh` 全量输出 | exit 0 |
| `../qwen3_golden/` | `gen_qwen3_golden.py` 产出（HF 同机 bf16 sdpa；不进 git） | — |
