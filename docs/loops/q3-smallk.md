# Q3-n 小 kernel 融合（loop 规格）

> decode 步每层现在 8 个 kernel：qkv GEMM、qkv_post、attention(+merge)、o_proj GEMM、fused_add_rmsnorm、gate_up GEMM、silu_mul、down GEMM；末尾 gather_rows、rmsnorm、lm_head GEMM、bf16→f32、copy、fused_sample。
> 可融：① gather_rows + 末层 rmsnorm → `gather_rmsnorm`（本轮）；② bf16→f32 + copy → 让 lm_head 直接出 f32（cudarc `Matmul<bf16>` 不支持混合输出 dtype，需走 cublasLt raw 或自写，后置）；③ silu·mul 进 gate_up GEMM epilogue（需自写 GEMM，与 megakernel 一起做）。
> 预期：-1 launch/步（~5 µs），主要是清理。

## 停止条件
1. `gather_rmsnorm_matches_two_step` 逐位相同；refs/parity 不变；4090 baseline PASS。
2. bench26 不退。
## 记录
| 轮 | 块 | 现象 | 根因 | 处理 |
|---|---|---|---|---|
| 1 | gather+rmsnorm | 2026-10-17 norm.cu 加 rows 间接（nullptr = 原语义），`gather_rmsnorm_bf16`；model.rs lm_head 前用它 | — | 3090 → 4090 |
| 2 | 验收 | 3090 refs 24/24（`gather_rmsnorm_matches_two_step` 逐位）、parity 3+5；4090 baseline PASS；bench26：c1 509 = 1.14× / c8 3100 = 1.07× / c32 8831 = 1.07× / c8-long 1942 = 1.01× —— 与 bench25 持平（±1%，抖动内）| -1 launch ≈ 5 µs，在 1.9–3.4 ms 的步里不可见，预期内 | **Q3-n 结论**：剩两项都要自写 GEMM：② lm_head bf16→f32 输出（cudarc `Matmul<T>` 要求 A/B/C 同 dtype；走 `cublaslt::result::matmul` 裸 API 可以 D=f32，-1 launch -1 次 4 MB 读写）；③ silu·mul 进 gate_up epilogue（cuBLASLt epilogue 只有 bias/relu/gelu）。两项并入 "逐层 megakernel" 阶段一起做，不单独开轮 |
