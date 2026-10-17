# Q3-q OpSet（算子插拔方案 A，loop 规格）

> design §10.1 方案 A：四组热路径算子（Attention / Norm / QkvPost / Act）进 trait，`DefaultOps` 零成本包装现有 ops.rs；
> `Qwen3Model<O: OpSet = DefaultOps>`；启动期 `OPS=default`（唯一实现，为后续 .so / 自研 GEMM 预留槽位）。
> Gemm（独立类型）与 Sampler（在 executor，不在模型）下一轮。

## 停止条件
1. 编译期：`Qwen3Model` 的 `k::xxx` 直调在四组内清零（grep）；泛型默认参数让所有现有调用方不改。
2. 3090 refs / parity / e2e 不变；4090 baseline PASS；bench29 与 bench28 四组 ±2%（零成本验证）。
3. `cargo asm`/符号层面：不要求；靠 bench。
## 记录
| 轮 | 块 | 现象 | 根因 | 处理 |
|---|---|---|---|---|
| 1 | 实现 | 2026-10-17 | — | opset.rs + model/nn 泛型化 + 启动期选择 |
| 2 | 本机验收 | `crates/kernels/src/opset.rs`（4 trait + OpSet + DefaultOps，全 `#[inline]` 转发）；`Qwen3Model<O: OpSet = DefaultOps>`、`Attention::forward<O>`、`Mlp::forward(.., act: &impl Act, ..)`；model.rs 四组 k:: 直调清零（剩 `gqa_decode_scratch_len` 两处分配期调用，非热路径）；glm53 / moe 调用点传 `&DefaultOps`。core 34/34、real_parity 3/3、sampling 5/5；`OPS=bogus` 启动即报错 | — | 4090 baseline + bench29 |
| 3 | 4090 验收 | baseline PASS；bench29：c1 509 = 1.14× / c8 3106 = 1.07× / c32 8815 = 1.07× / c8-long 1911 = 1.00× —— 与 bench28 四组差 <1%（零成本验证通过） | — | **Q3-q 完成**。后续：Gemm / Sampler 并入 OpSet；ref 测试泛型化；第二个实现（.so 或自研 GEMM）出现时再做 `--ops` 多分支 |
