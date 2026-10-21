#!/usr/bin/env bash
# 远端 r11d 树与本机 git 工作区的**文件齐备性**校验（不比对内容，只查缺失）。
#
# 为什么需要：2026-10-21 远端树缺 `crates/kernels/cuda/gqa_fa_sm100.cu`，
# 所有 `CUDA_ARCHS=100,103` 的构建都在链接期才报 `undefined reference`，
# 而默认 `CUDA_ARCHS=86,89,90` 不编该文件、一切正常 —— 于是「带 sm100 的构建
# 其实从来没成功过」这件事被隐藏了很久。构建前跑这个脚本，1 秒出结果。
#
# 用法：B300_TAG=50 bash scripts/verify/remote_sync.sh [远端树路径]
# 退出码：0 = 齐备；1 = 有缺失（打印清单）
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
REMOTE="${1:-/data/yaochen/infer/dinfer_r11d}"
: "${B300_TAG:?必须设 B300_TAG（项目规则：只用 B300-50）}"

R="scripts/dev/remote_b300.sh"
[ -x "$R" ] || { echo "找不到 $R"; exit 2; }

tmp=$(mktemp); trap 'rm -f "$tmp"' EXIT
# 需要校验的文件：git 跟踪的源码/构建输入（不含 test 产物与文档）
git ls-files 'crates/**/*.cu' 'crates/**/*.cuh' 'crates/**/*.rs' 'crates/**/*.h' \
              'models/**/*.rs' 'bins/**/*.rs' 'Cargo.toml' 'Cargo.lock' 2>/dev/null > "$tmp"
n=$(wc -l < "$tmp")
[ "$n" -gt 0 ] || { echo "git ls-files 没匹配到文件"; exit 2; }

echo "校验 $n 个文件 → $REMOTE"
# 一次 ssh 批量 stat（逐文件 ssh 太慢）
# 把清单喂给远端，远端逐行检查存在性，只回不存在的行
missing=$(B300_TAG="$B300_TAG" "$R" run "cd '$REMOTE' && while IFS= read -r f; do [ -e \"\$f\" ] || echo \"\$f\"; done" < "$tmp" 2>/dev/null)

if [ -n "$missing" ]; then
  echo "缺失文件："
  printf '  %s\n' $missing
  exit 1
fi
echo "齐备"
exit 0
