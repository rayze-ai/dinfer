# 远端机器口径（B300-50 / 4090 / 3090）

## B300-50（`B300_TAG=50 scripts/dev/remote_b300.sh up|run|push|pull|put`）
- token 一次性约 2 h；`run` 无输出 / rsync code 12 = 过期 → 向用户要，不要猜。
- 远端 `source /data/yaochen/infer/env.sh`（CUDA 13.1、CUDA_ARCHS=103、代理变量）；**本地 HTTP 前 `unset http_proxy https_proxy`**（reqwest 会把 127.0.0.1 发给代理）。
- 权重用物理路径 `/data/models/...`，不拷贝不软链。
- 多卡编号：`CUDA_DEVICE_ORDER=PCI_BUS_ID`（无害且稳定）；但"卡空却 OOM"先怀疑自己的分配（见 perf-method 误读表）。
- 容器：`nerdctl -n k8s.io`，vLLM `vllm/vllm-openai:nightly`（0.26.1rc1），`--net host`，用完 `rm -f`；K3 vLLM 只能 tp8 独占 8 卡（tp6 视觉塔 4096/6 不整除），init 624 s。
- Qwen3.5 bench 口径：GPU 0 dinfer `:8012`（tmux `q35`，`GRAPH_CAPTURE=eager --max-num-batched-tokens 16384`）、GPU 1 vLLM `:8013`（容器 `vllm-qwen35-b300`，`vllm_nerdctl.sh run 1 8013 1`）；`bench_compare.sh test/qwen35/<日期>/bench_b300_N "改动"` → pull 目录 + `perf-trend.*` → commit。
- nsys：GPU 4–7，端口 8412，`--gpu-memory-utilization 0.9`，`--cuda-graph-trace=node`；分析脚本 `$R/bootstrap/nsys_steps.py`；旧 `.sqlite` 存在时加 `--force-export=true`。
- 容器写进 bind mount 的文件属 root，`push --delete` 会 Permission denied → `sudo chown`。

## 4090（`scripts/dev/remote4090.sh`）
- 无 nvcc 在 PATH：非交互 ssh 必须 `source /data/yaochen/infer/env.sh`（CUDA 13.0，`CUDA_ARCHS=89`），否则 cudarc build.rs panic `nvcc --version failed`。
- 两卡门禁的 sm89 侧：`qwen35_gdn.sh` / `ops_parity.sh` / `bench_compare.sh`（qwen3 线常驻 :8012/:8013，`restart_dinfer.sh`）。

## 3090（本机）
- kernel 开发 / 对拍 / golden；B300 特有的收益（HBM 延迟型）在 3090 上看不出，别据此下结论。

## 趋势面板（本机 :8090，`scripts/dev/trend_server.sh`）
- 服务根是 `test/`：`index.html`（所有模型 × 卡型总览，`bench_index.py` 从各 `test/<model>/perf-trend.json` 生成）→ `<model>/perf-trend.html#<model>/<gpu>/single`。
- `bench_compare.sh` 跑完 `bench_trend.py` 后顺带重生成 `index.html`；但 bench 在远端（B300 / 4090）跑时生成的是远端的 index，**pull 回本机后要在本机再跑一次 `python3 -I scripts/dev/bench_index.py`**，否则本机 index 停在旧日期（2026-10-20 发现 index 只有 qwen3 / 10-19）。
- 重启服务按监听 PID 杀（`ss -ltnp | grep :8090`），不要 `pkill -f` 模式——模式会匹配到执行命令的 shell 自己。
