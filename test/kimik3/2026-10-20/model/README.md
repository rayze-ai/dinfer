# Kimi-K3 里程碑 C-3：前 5 层主干真权重对拍（B300-50，2026-10-20）

`scripts/verify/k3_model.sh` **exit 0**（第 3 轮；记录与口径见 `docs/loops/k3-model.md`）。

- 参考：`scripts/verify/gen_k3_model_refs.py`（vLLM nightly 容器，GPU 3，纯 torch 按 HF `modeling_kimi_linear.py` 逐层；KDA 用 vLLM 自带 fla）。
  真 token（HF tokenizer）T=31：`ids.npy`；`embed.npy`；每层 `in{l}.npy` / `layer{l}.npy`（输入 / 输出 prefix_sum，f32）；
  `attn3.npy`（第 3 层 MLA 子块输出）；`moe{l}/`（路由 logits / topk_ids / topk_margin / 子块）；`res_out / final / logits_last`；`meta.json`。
- Rust：`models/kimik3/tests/model_parity.rs`（GPU 2，release）。`run.txt` = 参考生成 + 对拍完整输出。
- 结论：隔离口径 5 层 cos 0.99998–0.99999；链式 5 层 cos 0.99999 → 0.99927（逐层累积）；末 token logits cos 0.99990、top-5 id 完全一致；
  decode(30+1) vs prefill 末行 cos 0.99997。装载 5 层 + 顶层 11.6 s / 74.9 GB；prefill T=31 98.5 ms、decode 25.7 ms。
