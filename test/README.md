# test 目录约定

- 按日期建目录：`test/YYYY-MM-DD/`
- 每个主题一个文件：`test/YYYY-MM-DD/<主题>.md`，同一天多轮就在同一文件内追加"第 N 轮"
- 每个文件**结论先行**，原始输出放后面；超过 200 行截取关键段并注明来源命令
- loop 规格的迭代记录也放这里：`test/YYYY-MM-DD/<feature>.md`

示例：

```
test/
  2026-10-13/
    login-flow.md
    perf-baseline.md
```
