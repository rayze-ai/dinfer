# 逐段 GPU 耗时：dinfer vs vLLM（2026-10-16 17:00:23，GPU 3，负载 --concurrency 8 --num-requests 24 --prompt-len 256 --max-tokens 64 --warmup 2）

注：nsys 抓的是整个进程生命周期（含启动 / 预热 / 图捕获）；按 step 归一化用 bench 完成 token 数 / 并发 ≈ decode step 数（dinfer 192，vLLM 192）。vLLM 的 kernel 若为空，说明宿主 nsys 没注入到容器（见脚本头注释）。

| 阶段 | dinfer ms (%) | vLLM ms (%) | dinfer kernel 次数 | vLLM 次数 | dinfer ms/step | vLLM ms/step |
|---|---|---|---|---|---|---|
| attention | 170.5 (31%) | 147.9 (10%) | 14771 | 20588 | 0.888 | 0.770 |
| gemm | 290.0 (52%) | 389.8 (25%) | 21426 | 31079 | 1.510 | 2.030 |
| norm | 31.3 (6%) | 21.1 (1%) | 20022 | 13608 | 0.163 | 0.110 |
| act/mlp | 20.9 (4%) | 18.5 (1%) | 4961 | 6899 | 0.109 | 0.097 |
| rope/embed | 11.0 (2%) | 0.0 (0%) | 5316 | 0 | 0.057 | 0.000 |
| sampling | 2.4 (0%) | 9.8 (1%) | 696 | 963 | 0.013 | 0.051 |
| copy/misc | 28.0 (5%) | 823.2 (53%) | 19844 | 20573 | 0.146 | 4.288 |
| other | 4.8 (1%) | 131.7 (9%) | 5138 | 12590 | 0.025 | 0.686 |
| **total GPU** | 558.9 | 1542.0 | 92174 | 106300 | 2.911 | 8.031 |

<details><summary>dinfer top kernels</summary>

-    187.1 ms  [gemm]  `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv`
-     85.1 ms  [attention]  `<unnamed>::gqa_decode_split_kernel(const __nv_bfloat16 *, const __nv_b`
-     79.7 ms  [gemm]  `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_1`
-     51.9 ms  [attention]  `<unnamed>::gqa_paged_flash_kernel(const __nv_bfloat16 *, const __nv_bf`
-     28.0 ms  [copy/misc]  `<unnamed>::copy_cols_kernel(const __nv_bfloat16 *, __nv_bfloat16 *, in`
-     22.4 ms  [norm]  `void <unnamed>::rmsnorm_kernel<(int)256, (bool)0>(const __nv_bfloat16 `
-     20.9 ms  [act/mlp]  `<unnamed>::silu_and_mul_kernel(const __nv_bfloat16 *, __nv_bfloat16 *,`
-     20.7 ms  [attention]  `<unnamed>::gqa_prefill_flash_kernel(const __nv_bfloat16 *, const __nv_`
-     10.4 ms  [rope/embed]  `void <unnamed>::rope_kernel<(bool)1>(__nv_bfloat16 *, __nv_bfloat16 *,`
-      9.7 ms  [gemm]  `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn`
-      8.9 ms  [norm]  `void <unnamed>::rmsnorm_kernel<(int)256, (bool)1>(const __nv_bfloat16 `
-      7.1 ms  [attention]  `<unnamed>::gqa_cache_write_kernel(const __nv_bfloat16 *, const __nv_bf`
</details>

<details><summary>vLLM top kernels</summary>

-    807.8 ms  [copy/misc]  `void at::native::vectorized_elementwise_kernel<(int)4, at::native::Fil`
-    193.2 ms  [gemm]  `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, __nv`
-    107.8 ms  [gemm]  `void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_1`
-    107.5 ms  [attention]  `void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<(int)128,`
-     57.2 ms  [other]  `triton_`
-     45.8 ms  [other]  `triton_red_fused_3`
-     29.0 ms  [attention]  `void flash::flash_fwd_splitkv_combine_kernel<Flash_fwd_kernel_traits<(`
-     26.5 ms  [other]  `triton_red_fused_1`
-     24.8 ms  [gemm]  `ampere_bf16_s1688gemm_bf16_128x128_ldg8_f2f_stages_32x1_tn`
-     24.5 ms  [gemm]  `void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64`
-     19.6 ms  [gemm]  `ampere_bf16_s16816gemm_bf16_64x64_ldg8_f2f_stages_64x5_tn`
-     18.5 ms  [act/mlp]  `triton_poi_fused_mul_silu_slice_1`
</details>
