# Kimi-K3 前 24 层主干参考（B300-50 容器纯 torch，2026-10-20 17:2x，3 min）

`scripts/verify/gen_k3_model_refs.py --layers 24`，真 token T=31，同 `model/` 口径（每层 in/layer/moe 子块、attn 子块、res_out / final / logits_last）。
用途：K3 乱文（全 93 层低熵坍缩，13 层仍高熵）找链式 cos 掉点：
`K3_LAYERS=24 K3_TP=2 K3_GPU=2,3 K3_SKIP_GEN=1 REFS=test/kimik3/2026-10-20/model24 bash scripts/verify/k3_model.sh`
（脚本 0bccba5 已修空跑；Rust 侧 24 层 ≈ 410 GB 要 tp2）。K3 挂起中（用户 2026-10-20 17:3x），未跑。
