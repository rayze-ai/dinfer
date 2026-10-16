# dinfer

> 本文件是项目的长期记忆，每次会话自动加载。保持 ≤ 80 行：只放稳定、短、每次都要用的内容；长内容放 `docs/`，这里只留指针。

## 开场必做

1. 读 `docs/progress.md` 顶部"当前状态"区，弄清上次做到哪、下一步是什么。
2. `git status --short` 核对工作区是否与交接区一致。
3. 用几行向用户复述现状，再动手。

## 项目是什么

用 Rust 重写 vLLM v1 引擎主干（启动 / 调度 / KV / 执行 / 采样 / HTTP），去掉通用插件层，目标之一是性能超越 vLLM（瓶颈分析见 design.md §8）；
每个模型一个 crate：Kimi-K3、GLM-5.3、**Qwen3-0.6B**（2026-10-15 起用它做主流程验证：真权重 / 采样 / 性能 / PD 分离，因为暂无高端卡；K3/GLM 真权重挂起，见 design.md §5b）。构建 = `core + server + 一个模型 crate`（cargo feature）。
分析基线：vLLM main `de8291e`（0.31.0），源码在 `/data/ref/vllm`；模型 HF 文件在 `/data/ref/models/{kimik3,glm53,qwen3-0.6b}`（qwen3 含真权重；均不进仓库）。
详细模块划分与分阶段计划见 `docs/design.md`；各模型的结构事实 / 模块 / 状态见 `docs/design-{kimik3,glm53,qwen3}.md`。

## 目录约定

| 路径 | 用途 |
|---|---|
| `.claude/CLAUDE.md` | 本文件，项目记忆（短、稳定） |
| `docs/requirements.md` | 需求与验收口径 |
| `docs/design.md` | 整体设计、关键决策；`design-<model>.md` 每模型一份 |
| `docs/progress.md` | 进度：顶部交接区 + 按日期日志 |
| `docs/pitfalls.md` | 踩坑：现象 / 根因 / 解法 / 避免 |
| `docs/loops/<feature>.md` | 需反复调试功能的环境-观察-停止条件 |
| `test/YYYY-MM-DD/` | 按日期归档的测试结果 |
| `scripts/verify/` | 验证脚本，exit 0 = 通过 |
| `scripts/dev/` | 启动环境、造数据、清理等 |

## 常用命令

```bash
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH   # nvcc 不在默认 PATH
cargo build --workspace --features glm53          # 或 --features kimik3 / qwen3；无 nvcc 加 --features no-nvcc
cargo test --workspace --features glm53           # CPU 单测；GPU 测试：cargo test -p dinfer-kernels -- --ignored
scripts/verify/p6.sh                              # GLM 线验收（含 p0–p5）；Qwen3 线：scripts/verify/q1.sh
cargo run -p dinfer --features glm53 -- serve --model test/tiny_glm_dsa_fullvocab --tokenizer /data/ref/models/glm53 --port 8011 [--mtp]
kubectl apply -f scripts/dev/k8s/dev-pod.yaml     # 4090 开发 Pod（runtimeClassName: nvidia 必带）
rm -f target/.rustc_info.json                     # cargo 报 "failed to run rustc to learn about target" 时（见 pitfalls）
```

## 硬性规则

<!-- 不能动什么、必须兼容什么、安全/合规约束 -->
- 文档用中文，命令与代码原样。
- 代码 / 脚本 / 环境变量 / C ABI 符号不带项目名前缀（`DINFER_*`、`dinfer_*`），项目可能改名；新代码前 `grep -rn DINFER crates scripts` 自查。
- 所有跨会话要记住的事写进文件并进 git，不要只留在对话里。

## 收功（每次结束前）

用户说"收功 / 收尾 / 今天到这"，或长会话即将压缩、准备关机迁移时，执行 `/project-skill wrap`：
更新 `docs/progress.md`（重写交接区 + 追加日志）、`docs/pitfalls.md`、归档 `test/YYYY-MM-DD/`、必要时回写本文件。

## 反复调试类功能

先 `/project-skill loop <feature>` 写清环境、观察方法、机器可判定的停止条件（`scripts/verify/<feature>.sh`）和迭代上限，再开始改代码。
