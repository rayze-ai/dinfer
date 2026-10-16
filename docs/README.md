# docs 目录索引

| 文件 | 内容 | 谁来更新 |
|---|---|---|
| `requirements.md` | 需求、不做什么、验收口径 | 需求变化时 |
| `design.md` | 整体架构、共性模块、阶段计划、性能分析、关键决策 | 设计变化时，决策只追加不删 |
| `design-kimik3.md` / `design-glm53.md` / `design-qwen3.md` | 各模型：结构事实、crate 模块、实现状态、模型特有风险 | 该模型有进展时更新 §0 状态 |
| `progress.md` | 顶部交接区（每次收功重写）+ 按日期日志（只追加） | 每次 `/project-skill wrap` |
| `pitfalls.md` | 踩坑：现象 / 根因 / 解法 / 避免 | 踩到就记 |
| `loops/<feature>.md` | 需反复调试功能的环境、观察方法、停止条件、上限 | `/project-skill loop <feature>` |
