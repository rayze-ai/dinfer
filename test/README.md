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


## 2026-10-16 起的约定

按 **模型 / 日期** 归档：`test/<model>/YYYY-MM-DD/`（如 `test/qwen3/2026-10-16/`）。每一轮验证与改进（loop 的每一轮、每次 bench）都要在当天目录下留记录：现象 / 改了什么 / 数字 / 结论。GLM 线 2026-10-14、10-15 的历史目录保持原路径。
