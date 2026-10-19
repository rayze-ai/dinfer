# Qwen35 里程碑 4：`models/qwen35` crate 到 tiny golden 对拍

> 用户 2026-10-19："赶紧把 Qwen35 完成，目录名字叫 Qwen35"。前置：GDN kernel（`qwen35-gdn.md`）与 attention 扩模板（`qwen35-attn.md`）均两卡对拍通过。
> 第一位里程碑 4 子代理 40 min 只读文档无产出，主会话 20:1x 接手搭骨架。

## 环境
本机 3090（编译 / 单测 / tiny golden GPU 对拍）、4090 GPU 7（sm89 对拍，`CUDA_VISIBLE_DEVICES=7 GPU_WAIT_S=5`）。B300 不跑（用户规则：等确认）。

## 观察方法
- `cargo test -p dinfer-qwen35`（CPU：config / weights 映射单测）。
- `scripts/verify/qwen35_golden.sh`：造 3 层 tiny golden（[linear, linear, full]）→ torch 参考 logits → Rust prefill logits / greedy 16 / 双请求混批对拍；exit 0 = 通过。
- `cargo build --workspace --features qwen3` 与 `--features qwen35` 都过；`ops_parity.sh` 不退化。

## 停止条件
`qwen35_golden.sh` exit 0 + 双 feature 构建过 + `chat_parity` 过。

## 迭代上限
10 轮。

## 记录
| 轮 | 改动 | 结果 |
|---|---|---|
| 1 | 骨架：Cargo.toml / lib.rs / config.rs（壳或裸 text_config 均可解析，`check()` 钉 head_dim 128|256、GROUP 1|2|4|6、rope_dim、GDN 整除、f32 状态、MTP ≤ 1；派生 gqa_group / rope_dim / gdn_group / eos_ids）+ workspace / bins 接 feature `qwen35` | config 单测 2/2 |
