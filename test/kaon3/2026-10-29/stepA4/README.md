# Step A v4（D199–D202）：5 轮 × c ∈ {1,2,4,8,16,32,64}，逐环节对标 vLLM
口径：两端非 eager；max-model-len 16384 / max-num-batched-tokens 4096；dinfer max-num-seqs 64（vLLM 默认 256）；400 tok prompt / 64 out / 8c 请求 / warmup 2；
同窗口同 seed、跨窗口换 seed（1000r+c）。钉核 serve 0-31 / bench 60-91 / vLLM 容器 GPU2。dinfer 二进制 md5 9046f6ef（HEAD，MOE_DEC_RMAX 默认 512）。
`RESULT.txt`：每窗口 bench 三指标 + GPU util + host 分段 + 两端 /metrics 增量（dinfer 侧首版前缀没剥，全 0，`snaps.txt` 里有原始快照可重算）。
中位数表见 `docs/loops/kaon3-decode-opt.md` D202。
