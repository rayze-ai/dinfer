# 远端机器口径（B300-50 / 4090 / 3090）

## B300-50（`B300_TAG=50 scripts/dev/remote_b300.sh up|run|push|pull|put`）
- token 一次性约 2 h；`run` 无输出 / rsync code 12 = 过期 → 向用户要，不要猜。**拿到 token 先看"过期时间"字段再动手**（2026-10-20 一枚 10:44 过期的 token 10:40 才到手，`up` 成功但 3 分钟后全部失效；`status` 的 alive 是陈旧 socket，不可信）。有效窗口 < 30 min 的 token 直接要新的。
- 远端 `source /data/yaochen/infer/env.sh`（CUDA 13.1、CUDA_ARCHS=103、代理变量）；**本地 HTTP 前 `unset http_proxy https_proxy`**（reqwest 会把 127.0.0.1 发给代理）。
- 权重用物理路径 `/data/models/...`，不拷贝不软链。
- 多卡编号：`CUDA_DEVICE_ORDER=PCI_BUS_ID`（无害且稳定）；但"卡空却 OOM"先怀疑自己的分配（见 perf-method 误读表）。
- 容器：`nerdctl -n k8s.io`，vLLM `vllm/vllm-openai:nightly`（0.26.1rc1），`--net host`，用完 `rm -f`；K3 vLLM 只能 tp8 独占 8 卡（tp6 视觉塔 4096/6 不整除），init 624 s。
- Qwen3.5 bench 口径：GPU 0 dinfer `:8012`（tmux `q35`，`GRAPH_CAPTURE=eager --max-num-batched-tokens 16384`）、GPU 1 vLLM `:8013`（容器 `vllm-qwen35-b300`，`vllm_nerdctl.sh run 1 8013 1`）；`bench_compare.sh test/qwen35/<日期>/bench_b300_N "改动"` → pull 目录 + `perf-trend.*` → commit。
- nsys：GPU 4–7，端口 8412，`--gpu-memory-utilization 0.9`，`--cuda-graph-trace=node`；分析脚本 `$R/bootstrap/nsys_steps.py` / `nsys_mixed.py`（**先切稳态窗口**）；旧 `.sqlite` 存在时加 `--force-export=true`。**结束方式**：对 **nsys 进程本身** `kill -INT`（它会转发给 serve 并在退出时写报告）；只 `kill -INT` serve 子进程时 nsys 可能不落 .nsys-rep（2026-10-20 第二次空跑）；`--kill=sigint` 在这版 nsys 不是合法选项（打印 usage、serve 没起、报告 5 KB 空壳——2026-10-20 浪费一轮）；`pkill -f` 模式会误杀 nsys 自身丢 .nsys-rep。
- 容器写进 bind mount 的文件属 root，`push --delete` 会 Permission denied → `sudo chown`。

## 4090（`scripts/dev/remote4090.sh`）
- 无 nvcc 在 PATH：非交互 ssh 必须 `source /data/yaochen/infer/env.sh`（CUDA 13.0，`CUDA_ARCHS=89`），否则 cudarc build.rs panic `nvcc --version failed`。
- 两卡门禁的 sm89 侧：`qwen35_gdn.sh` / `ops_parity.sh` / `bench_compare.sh`（qwen3 线常驻 :8012/:8013，`restart_dinfer.sh`）。

## 3090（本机）
- kernel 开发 / 对拍 / golden；B300 特有的收益（HBM 延迟型）在 3090 上看不出，别据此下结论。

## 趋势面板（本机 :8090，`scripts/dev/trend_server.sh`）
- 服务根是 `test/`：`index.html`（所有模型 × 卡型总览，`bench_index.py` 从各 `test/<model>/perf-trend.json` 生成）→ `<model>/perf-trend.html#<model>/<gpu>/single`。
- 远端跑的 `bench_trend.py` 只看到那台卡的 bench 目录（4090 上生成的 qwen3 趋势没有 B300 4 轮；B300 上生成的没有 4090）；**权威的 `perf-trend.*` 与 `index.html` 以本机全树生成为准**：pull 回 bench 目录后跑 `scripts/dev/trend_rebuild.sh`（所有模型 + index），再 commit。2026-10-20 两次踩：index 只有 qwen3/4090、qwen3 缺 B300 行。
- 重启服务按监听 PID 杀（`ss -ltnp | grep :8090`），不要 `pkill -f` 模式——模式会匹配到执行命令的 shell 自己。

## 4090 测试机是共享的：bench 前核 KV 块数 / 外来进程（2026-10-20）

- **判断**：4090 ×8 不是独占机，root 会临时在任意卡起 vLLM（`--port 8000/8200`，20 GB+）。我们的 dinfer 按 `utilization 0.9` 自动算 KV 块数，被占时块数骤降（9900 → 902）、首请求 OOM，bench 看起来是"我们挂了"。
- **怎么用**：bench 链 serve 起来后读启动日志 `num_blocks`，低于基线 10% 直接 abort（`bench_compare` 前置）；`nvidia-smi --query-compute-apps` 看非 dinfer / 非 vllm-qwen3 容器的进程；GPU 0 被占就换 3–7（都空），把 `CUDA_VISIBLE_DEVICES` 写进链。
- **证据**：`docs/progress.md` 2026-10-20 4090 门禁第三次 OOM。

## 两台 B300 只用 `B300_TAG=50`；无 TAG 的旧连接会"alive 但 run 空"（2026-10-20）

- **判断**：`remote_b300.sh status` 的 alive 只看 ControlMaster 进程在，JumpServer 侧会话失效后 run 仍 rc=0、stdout 0 字节。2026-10-19 起所有工作都在 TAG=50（host-172-16-3-50，8 × B300，有 nerdctl 容器、口径与 vLLM 对齐）。
- **怎么用**：子代理 brief 里写死 `export B300_TAG=50`；`run 'hostname'` 返回空就是会话失效，不要以为是远端卡；旧 memory "TAG=50 无容器、口径不同"已过时。
- **证据**：`docs/progress.md` 2026-10-20 r8b 条目。
