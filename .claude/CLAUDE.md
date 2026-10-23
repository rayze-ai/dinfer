# dinfer

> 本文件是项目的长期记忆，每次会话自动加载。保持 ≤ 80 行：只放稳定、短、每次都要用的内容；长内容放 `docs/`，这里只留指针。

## 开场必做

1. 读 `docs/knowledge/README.md`（提炼后的方法与规则索引，按主题；开工前读对应页），再读 `docs/progress.md` 顶部"当前状态"区，弄清上次做到哪、下一步是什么。
2. `git status --short` 核对工作区是否与交接区一致。
3. 用几行向用户复述现状，再动手。

## 项目是什么

用 Rust 重写 vLLM v1 引擎主干（启动 / 调度 / KV / 执行 / 采样 / HTTP），去掉通用插件层，目标之一是性能超越 vLLM（瓶颈分析见 design.md §8）；
每个模型一个 crate：Kimi-K3、GLM-5.3、**Qwen3-0.6B**、Qwen3.5-27B（`models/qwen35`，2026-10-19 起：48 GDN + 16 全注意力混合模型，tiny golden 4/4 已过，真权重待 B300）（2026-10-15 起用它做主流程验证：真权重 / 采样 / 性能 / PD 分离，因为暂无高端卡；K3/GLM 真权重挂起，见 design.md §5b）。构建 = `core + server + 一个模型 crate`（cargo feature）。目录：公用基础设施在 `crates/{core,kernels,layers,text,server}`，模型 crate 在 `models/{qwen3,glm53,kimik3,qwen35}`（2026-10-19 起）。
远程：`git@github.com:rayze-ai/dinfer.git`，工作分支 **`dev`**（2026-10-16 起；远端暂无 main）。编译 / GPU 测试在 4090 测试机（`scripts/dev/remote4090.sh`，见 `scripts/dev/README-remote4090.md`）；B300 **只用 B300-50**（`B300_TAG=50 scripts/dev/remote_b300.sh …`，脚本已强制），**第一台 .225 用户禁止占用（2026-10-20）**。
分析基线：vLLM main `de8291e`（0.31.0），源码在 `/data/ref/vllm`；**Kaon-V3 例外：读远端镜像里的 vLLM（0.26.1rc1，副本 `/data/ref/vllm-img-v0.28.0-pr455/`，用户 2026-10-23；镜像暂不升级）**；模型 HF 文件在 `/data/ref/models/{kimik3,glm53,qwen3-0.6b}`（qwen3 含真权重；均不进仓库）。
详细模块划分与分阶段计划见 `docs/design.md`；各模型的结构事实 / 模块 / 状态见 `docs/design-{kimik3,glm53,qwen3}.md`。

## 目录约定

| 路径 | 用途 |
|---|---|
| `.claude/CLAUDE.md` | 本文件，项目记忆（短、稳定） |
| `docs/requirements.md` | 需求与验收口径 |
| `docs/design.md` | 整体设计、关键决策；`design-<model>.md` 每模型一份 |
| `docs/progress.md` | 进度：顶部交接区 + 按日期日志 |
| `docs/pitfalls.md` | 踩坑：现象 / 根因 / 解法 / 避免（按时间堆；可复用的判断提炼进 knowledge） |
| `docs/knowledge/` | **知识库**：按主题提炼的判断 + 为什么 + 怎么用 + 证据（perf-method / kernel-rules / model-onboarding / executor-common / subagent-ops / infra-b300）；每页 ≤ 120 行，过时的删 |
| `docs/loops/<feature>.md` | 需反复调试功能的环境-观察-停止条件 |
| `test/<model>/YYYY-MM-DD/` | 按模型、按日期归档的测试结果与每轮改进记录（用户 2026-10-16 要求：每一轮验证与改进都要记进去）；GLM 线历史在 `test/2026-10-1x/` |
| `scripts/verify/` | 验证脚本，exit 0 = 通过 |
| `scripts/dev/` | 启动环境、造数据、清理等 |

## 常用命令

```bash
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH   # nvcc 不在默认 PATH
cargo build --workspace --features glm53          # 或 --features kimik3 / qwen3 / qwen35；无 nvcc 加 --features no-nvcc
cargo test --workspace --features glm53           # CPU 单测；GPU 测试：cargo test -p dinfer-kernels -- --ignored
scripts/verify/p6.sh                              # GLM 线验收（含 p0–p5）；Qwen3 线：scripts/verify/q1.sh；OpenAI 服务面：scripts/verify/{api_surface,guided,sdk_smoke}.sh；投机：ngram.sh；算子对拍：ops_parity.sh
cargo run -p dinfer --features glm53 -- serve --model test/tiny_glm_dsa_fullvocab --tokenizer /data/ref/models/glm53 --port 8011 [--mtp]
kubectl apply -f scripts/dev/k8s/dev-pod.yaml     # 4090 开发 Pod（runtimeClassName: nvidia 必带）
rm -f target/.rustc_info.json                     # cargo 报 "failed to run rustc to learn about target" 时（见 pitfalls）
CUDA_ARCHS=86 cargo build --release -p dinfer-kernels   # 本机开发只编本卡 arch：23 s vs 默认 86,89,90 的 2 min（2026-10-20）
scripts/dev/ttft_probe.py <port> 5 860            # 单发 TTFT（words 430≈1K / 860≈6K），钉核 100-103；探针结果进 probe_*/probe.csv
python3 scripts/dev/probe_trend.py --root test/qwen35 && python3 scripts/dev/bench_index.py   # 探针 / 索引页重生成（页面 8090）
```

## 硬性规则

<!-- 不能动什么、必须兼容什么、安全/合规约束 -->
- 文档用中文，命令与代码原样。
- **多卡型实现（用户 2026-10-19）**：一套接口多套实现，按 arch **分文件**（`*_sm8x.cu` / `*_sm100.cu`），Rust 侧按 `ArchProfile` 选；**kernel 体内禁止 `#if __CUDA_ARCH__`**，共享头的分叉只在 `arch.cuh` 用有名字的能力常量（`kHasTma`），不写裸数字；性能改动必须过两卡门禁（`scripts/verify/bench_gate.sh`，任一卡退化 > 3% 不合入）。规格 `docs/loops/arch-profile.md`。
- **kernel 内不写卡型 / 模型相关的调优数值**（splits、目标 block 数、向量档位、warp 数等，用户 2026-10-19）：kernel 只收运行时参数，数值由 `ArchProfile`（按卡默认）+ 启动自调优（按模型形状实测）给，环境变量只做覆盖。规格 `docs/loops/arch-profile.md`。
- `crates/` 只放模型无关代码：模型名 / token id / 模板变量名 / 模型路径出现在 `crates/` 即为错误，走 `ChatBackend` 钩子由 `models/<m>` 提供；模型相关测试放 `models/<m>/tests/`（用户 2026-10-19）。
- 代码 / 脚本 / 环境变量 / C ABI 符号不带项目名前缀（`DINFER_*`、`dinfer_*`），项目可能改名；新代码前自查：`grep -rniE '\bdinfer_[a-z]|DINFER_' crates scripts | grep -v 'dinfer_[a-z0-9]*::'`（crate 名 `dinfer-xxx` 与 `use dinfer_xxx::` 是例外，大小写都查）。
- 所有跨会话要记住的事写进文件并进 git，不要只留在对话里。
- **`docs/progress.md` 交接区按模型分节（用户 2026-10-21）**：每节只写该模型自己的内容，收功时**只改自己那一节**，不要整体重写覆盖别的模型。
- **子代理（2026-10-20）**：**本地写 → `put` 单文件 → 远端编 / 跑**（用户 2026-10-21：本地有编辑器和 git diff，比 ssh heredoc 顺手；但本机 nvcc 12.4 编不了 sm_103a，**每一轮都必须传过去在 B300 上编译**，不许只写不编）。收口时列改动文件清单，由主会话 `pull` 回本机 commit。禁止 `remote_*.sh push`（rsync `--delete` 两次删掉别人的远端产物），要同步代码发消息给主会话；产物落 `/tmp` 或 `bootstrap/` 由主会话 pull；任何 GPU 动作前先问主会话；可失败步骤（merge / rebase / 编译）单独发一条看结果再推远端起长链；链内杀进程用 `pkill -x` / `pgrep -x`，不用 `-f`。
- **Rust 实现、Rust 验证（用户 2026-10-22）**：不写 / 不入库 Python 参考实现，对拍以 vLLM 服务输出（`prompt_logprobs`）+ Rust GPU 测试为准；交接区定好的方案不擅自改。
- **目标是超越，不是重写**（用户 2026-10-17）：vLLM 做得对的直接采用，能改进的改进，两种情况都要在 `docs/design.md` §11 "采用 / 改进台账" 记一行（采用了什么、为什么、我们改了哪里、收益）。不为"不一样"而不一样。
- **测试程序本身要可信（用户 2026-10-20）**：引擎层面 bench 必须 ① 三方（dinfer / vLLM / 压测端）`taskset` 到不重叠核组并写进报告；② 压测端上报自身 CPU 时间 / RSS / 线程数（Rust `bench` 作主口径，CPU < 1%；Python `vllm bench serve` rate 8 时吃 2–4 核、与被测共核会把 dinfer decode 步拖 2×，只做交叉）；③ 同负载与第三方工具 p50/p99 差 ≤ 10%；④ 客户端测量与服务端自报（`/metrics`）对账；⑤ 启动前核 `/v1/models` id（编错 feature 的二进制会让 bench 空跑出 0）。不满足任一条的数字作废。规格 `requirements.md` 验收 v2 补充 / `design.md` §15.5 / `knowledge/perf-method.md` §7。
- **每轮性能改动必须进 CSV 趋势**（用户 2026-10-16 两次强调）：改动 → 4090 `bench_compare.sh test/qwen3/<日期>/benchN "改动一句话"`（写 `CHANGE`）→ 自动重生成 `test/qwen3/perf-trend.{csv,md,svg,html}`（每组 tok/s ×vLLM、TTFT、TPOT、相对上一轮 %）→ `pull` 回本机 commit。没有 bench 记录的改动视为未完成。
- 上下文纪律（已 **4 次** 撞 `Input is too long`，2026-10-16 起机器强制）：`.claude/settings.json` 设 `BASH_MAX_OUTPUT_LENGTH=8000`；`.claude/hooks/context_guard.sh`（PreToolUse）拒绝 Read >200 行不带 limit、拒绝不截断的 `cat`/`git diff`/`git show`，并在第 40 次工具调用起每 10 次提醒写记录 + commit。自律部分：Bash 输出一律 `grep`/`tail`/`cut -c1-200`，长任务日志 `tee /tmp` 只看摘要；每完成一个子块就写 loop 记录并 commit；**loop 未达停止条件不停下、不中途问（用户 2026-10-22 覆盖旧的"60 次收功"）**：每完成一个子块写 loop 记录并 commit 后继续，上下文由系统自动压缩。

## 收功（每次结束前）

用户说"收功 / 收尾 / 今天到这"，或长会话即将压缩、准备关机迁移时，执行 `/project-skill wrap`：
更新 `docs/progress.md`（重写交接区 + 追加日志）、`docs/pitfalls.md`、**把本次 loop 里可复用的教训提炼进 `docs/knowledge/` 对应页（用户 2026-10-20：知识要落地，不要重启就丢）**、归档 `test/<model>/YYYY-MM-DD/`、必要时回写本文件。

## 反复调试类功能

先 `/project-skill loop <feature>` 写清环境、观察方法、机器可判定的停止条件（`scripts/verify/<feature>.sh`）和迭代上限，再开始改代码。
