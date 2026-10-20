# 引擎层面 bench：泊松到达 + 长度分布 + 多轮前缀 + SLO goodput（2026-10-20 起）

> 口径定义：`docs/requirements.md` "验收口径 v2" / `docs/design.md` §15。用户："我要的是引擎层面测超越"——四组闭环固定并发是
> 模型层面（kernel 竞速），只作回归门禁；"超越 vLLM" 的判定换成本文口径。工具：`bins/bench`、`scripts/verify/bench_engine.sh`、
> `scripts/dev/engine_trend.py`、第三方交叉验证 `scripts/verify/bench_engine_vllm_tool.sh`。

## 环境
- 首轮：Qwen3.5-27B，B300-50 GPU 0 dinfer `:8012`（`GRAPH_CAPTURE=eager --max-num-batched-tokens 16384`，prefix cache 对 GDN 强制关），
  GPU 1 vLLM `:8013`（容器 `vllm-qwen35-b300`，开 prefix cache = 默认；`mamba_cache_mode=align`）。两边**各自最佳配置**写进 `summary.md` 头。
- 之后 K3 tp8（C-6）同口径。
- 本机 3090 只做工具冒烟：tiny 全词表 qwen35（`scripts/dev/make_qwen35_tiny.py /tmp/q35tiny/model --vocab 248320 --gen 2` → 权重在
  `<out>/tiny/`），`dinfer serve --model /tmp/q35tiny/model/tiny --tokenizer test/qwen35/tokenizer --port 8012`，
  `VLLM_URL= RATES="4 8" REQS=24 bash scripts/verify/bench_engine.sh test/qwen35/<日期>/engine_smoke "..."`。

## 负载（bench_engine.sh 默认，环境变量可改）
| 项 | 默认 | bench 参数 |
|---|---|---|
| 到达 | 泊松开环，`RATES="2 4 8 12 16 24"` req/s，每率 `REQS=96` + 2 预热 | `--request-rate R --num-requests 96 --warmup 2` |
| prompt 长度 | `MIX="128:0.5,1024:0.35,4096:0.15"` | `--prompt-len-mix` |
| 输出长度 | `OUT_MIX="128:0.5,512:0.35,1024:0.15"`，**不 ignore_eos**（实际 = min(eos, 抽样)） | `--output-len-mix --allow-eos` |
| 前缀复用 | `PREFIX_REPEAT=0.5`：一半请求共享 `PROMPT_LEN=1024` 词前缀 + 16 随机词 | `--prefix-repeat 0.5 --prompt-len 1024` |
| 多轮 | `MULTI_TURN=1`；>1 时到达单位是会话，每会话顺序发 K 轮（messages 带全部历史），会话率 = R/K | `--multi-turn K` |
| SLO | `SLO_TTFT=500` `SLO_TPOT=20` ms | `--slo-ttft-ms --slo-tpot-ms` |
| seed | 每次运行取当前秒（两边同 seed → 同到达序列、同 prompt） | `--seed` |

## 观察方法（每率两端各一行进 `summary.md`，并进 `test/<model>/engine-trend.{csv,md,html}`）
- **goodput**（首要）：TTFT ≤ SLO **且** TPOT ≤ SLO 的完成数 / 墙钟（vLLM `--goodput ttft:500 tpot:20` 同义，DistServe 定义；输出 ≤ 1 token 的 TPOT 记 0）。
  **SLO 下最大 req/s** = good / completed ≥ 0.95 的最大到达率（engine_trend.py 算；不用 goodput ≥ 0.95×rate，短跑收尾长尾会压低实际 req/s）。
- TTFT / TPOT 的 **p50 / p90 / p99 / max**（尾部）。TTFT = 首个带内容 chunk；另记 `ttft_first_chunk_ms`（任意首 SSE 事件 = vLLM bench serve 口径，
  dinfer 首 chunk 是空 role delta → 它测出 ~5 ms 伪像，只用于与第三方工具对账）。
- prefix cache 命中率 = Σ`usage.prompt_tokens_details.cached_tokens` / Σprompt_tokens（两边同字段；dinfer 由 `Request.num_cached_tokens` 首次调度时记）。
- `len-stop`：finish_reason == length 的条数（不 ignore_eos 下看多少条跑满）。
- dinfer `/metrics` `engine_step_execute_{prefill,decode,mixed}_seconds_{count,sum}` 在本率 bench 前后的增量 → 每率一列 "P/D/M count (mean ms)"：混批步占比与均值。
- 饱和行为：failed 列（超时 / 拒绝）、good/completed 掉到 < 0.95 之后 TTFT p99 的增长斜率。
- **测试程序自身的准确性与开销**（用户 2026-10-20："测试数据的准确性，还得考虑测试程序的准确性、测试程序占用资源的影响"）：
  - 计时口径：时间戳在收到字节时打（单调时钟），SSE 切分 / JSON 解析在后、耗时另计（`client.parse_ms_per_request`）；prompt / 到达时刻 / max_tokens
    发压前一次预生成（`Plan`），发送期只做 HTTP + 解析；`--self-check` 对内置固定延迟 mock（TTFT 50 / ITL 10 ms）报测量误差 µs
    （3090：TTFT p50 +1.5 ms——mock 侧 tokio 定时器 + 回环；TPOT +13 µs；ITL p50 +0.19 ms；解析 0.11 ms/req），写进 summary 头。
  - 资源：bench JSON `client` 段 = getrusage CPU（user+sys，平均占用核数）、峰值 RSS、线程数、`sched_getaffinity` CPU 集合；`bench_engine.sh` 用
    `taskset -c $BENCH_CPUS`（默认 nproc 最后 8 核）钉压测进程，summary 头写两个服务进程（按端口找 pid 的 Cpus_allowed_list）与压测进程的核集合，重叠标 ⚠。
  - 服务端口径对账：每率 bench 前后抓两边 `/metrics` 整份快照，`scripts/dev/metrics_delta.py` 算差值——dinfer `time_to_first_token_seconds` /
    `time_per_output_token_seconds` 直方图增量的 mean/p50/p99（桶内插值，只对量级）、`prefix_cache_{queries,hits}_total`、`engine_step_execute_{prefill,decode,mixed}`
    count/均值；vLLM `vllm:time_to_first_token_seconds` / `vllm:inter_token_latency_seconds`（旧版 `time_per_output_token_seconds`）、`vllm:prefix_cache_{queries,hits}`。
    同一行里 客户端 TTFT − 服务端 TTFT = 网络 + 工具开销。**注意**：dinfer `engine_step_execute_*` 在 async 调度下量的是含 finish 等待的 host 侧步时，不等于 GPU 步时间。
- 交叉验证：`bench_engine_vllm_tool.sh`（B300 容器内 `vllm bench serve --dataset-name random --random-prefix-len 512 --goodput ...`）打两端，
  然后（`OURS=1` 默认）用我们的 bench 在同端 / 同 rate / 同 seed / 同长度各跑一组，`scripts/dev/engine_xcheck_row.py` 把 `ttft_first_chunk_ms`、`tpot_ms` 的 p50 / p99
  与工具的 `p50_ttft_ms` 等并排写进 summary 的"对账"节，差 > 10%（`THRESH`）标 ⚠；差太大先核两边 prompt 实际 token 数与 len-stop 比例，再怀疑计时口径。

## 停止条件
两边各自最佳配置下扫 rate：dinfer **SLO 下最大 req/s ≥ vLLM** 且在该 rate 及以下 TTFT p99 / TPOT p99 均不劣于 vLLM；多轮负载（`MULTI_TURN=3`）下同样成立；
长跑（固定 rate 1 h）TPOT p99 / 显存不漂移。达成 → 引擎层面超越，写 `docs/design.md` §15 结论。

## 迭代上限
10 轮 `engine_N`。每轮一个改动（预期第 1 优先级是 GDN 状态按块缓存，见 §15.3），改动写 `CHANGE`，不进 CSV 的轮不算。

## 日志
| 轮 | 结果 / 结论 |
|---|---|
| 0 | 2026-10-20 工具落地：bench `--output-len-mix / --slo-* / --multi-turn / cached_tokens / ttft_first_chunk_ms / client 段 / 预生成 Plan`；server `usage.prompt_tokens_details.cached_tokens`；`bench_engine.sh`（taskset、/metrics 差值对账）、`engine_trend.py`、`metrics_delta.py`、`bench_engine_vllm_tool.sh`（含与我们 bench 的对账节）。3090 tiny qwen35 冒烟 `test/qwen35/2026-10-20/engine_smoke`（仅 dinfer，rate 4 / 8，24 条）：JSON 字段齐、goodput 手算一致、cache hit 0（GDN 强制关 prefix cache，预期）。 |
