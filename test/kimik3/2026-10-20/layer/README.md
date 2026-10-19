# K3 里程碑 C-2：第 1 层整层真权重对拍（B300-50，2026-10-20）

- 参考：`scripts/verify/gen_k3_layer_refs.py`（vLLM 容器 GPU 3，fla kda + 纯 torch MoE，MXFP4 按 realcheck 口径解包）→ `*.npy` + `meta.json`。
- 对拍：`scripts/verify/k3_layer.sh`（Rust 测试 GPU 2，`--release`）。
- `k3_layer_run1_refgen.log`：第 1 轮（含参考生成；MoE 子块按全部行断言 → 平局行挂，见 docs/pitfalls.md）。
- `k3_layer_run2.log`：第 2 轮（`K3_SKIP_GEN=1`，MoE 子块只在路由一致行判）**exit 0**。
- 结果汇总：`docs/loops/k3-moe.md`「里程碑 C-2」。
