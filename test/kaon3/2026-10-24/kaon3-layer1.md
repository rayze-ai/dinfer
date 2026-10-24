# kaon3-layer1 迭代记录（2026-10-24）

## 第 1 轮：验证脚本与判据修正
- 参数检查全部失效：`docker inspect` 返回 JSON 数组（`"--block-size","32"`），按空格匹配必然失败。
- **判据用错**：截 1 层时 top1 margin 全 0（`decisive=0`），`flip` 是 0/0，
  用它会误报"逐位一致"。改为：`decisive>0` 看 flip，否则看 `|Δlp|` 与 `ΔNLL` 是否为 0。
- 修正后实测：`mean_common_d=0.0085`、`ΔNLL=+1.5064` → **差异在 L0 之内**。

## 第 2–4 轮：norm 与 FFN 合成段
- 权威源码 `vllm/ir/ops/layernorm.py:11`：`x = x.to(weight.dtype) * weight`（先舍 bf16 再乘 w）。
- 数值验证：`RMSNORM_MODE=2` 与运行时 `rms_norm` **逐位相同（0/67584）**；mode 0 差 17063。
- `seg_l0.py` 逐段复现：`input_norm / pre_ff_norm / pre_ff2_norm / ff1 / post_ff_norm`
  **五段全部 0/67584 逐位相同**。

## 第 5–7 轮：定位到 fp8 GEMM
- 同一输入下：`q 88911/98304`、`k 43145/49152`、`v 2/49152`。
  **v 几乎不差 → GEMM 实现没错。**
- FX 图显示 vLLM 的 `qkv_proj` 是**单个** `[2816, 8192]` GEMM。
- 合并一次 N=8192 实测 `ndiff=13/196608`；分开三次差 13 万个 → **根因是调用形状**。

## 第 8 轮：requantize 算法验证通过
`w8a8_utils.py:76 requantize_with_max_scale` 复现：
```
[1] 复现权重 vs vLLM 实际权重 : 23068672/23068672 字节全同
[2] 复现权重 GEMM vs vLLM 输出: 0/33554432 逐位相同
[3] input_scale = max(三个) = 0.59375
```
且发现 **Rust 里 `weights.rs:225 load_fp8_group` 早已实现**这套逻辑（还处理了 fp16 中间值）。

## 第 9 轮（未完成，已回退）：合并前向
改了 `layer.rs` 加 `qkv_proj` 合并字段 + `forward_qkv_fused`，编译通过但**端到端更差**：
```
分开三次（QKV_SPLIT=1）: flip  6.12%  |Δlp| 1.0062
合并一次（新）         : flip 87.76%  |Δlp| 3.6515
```
**我的验证方法错了**：只比了"fused vs split"（两边都是我们自己），且第 0 层是 `k_eq_v`，
两边都只算 N=6144，互相一致但都不等于参照的 N=8192。

## 卡点（下次接手第一件事）

**第 0 层 `k_eq_v=True`：参照的 `qkv_proj` 到底是 N=8192 还是 N=6144？**
源码 `gemma4.py:413` 说 `qkv_proj` 结构不变、K 权重同时填 K/V 两个槽（→ N=8192）；
但 `gemma4.py:1513` 的 `packed_modules_mapping` 注释说 k_eq_v 层用分开的 q/k（不打包）。

已抓到 vLLM 运行时真实的 `[2816, 8192]` 权重（`scripts/dev/capqkv.py`），
但三段都不等于 checkpoint 原值（有 requantize）。**要做的是**：用第 8 轮已验证的
requantize 公式，分别按「q,k,v」与「q,k,k」两种拼法复现，看哪个逐字节等于抓到的权重。
这一步能一次定死 `k_eq_v` 的语义。

**代码状态**：`layer.rs` 的未验证改动**已回退**，工作区干净。
