# rl-algorithm-capabilities 进展

对齐结论、依赖矩阵、工作包与待批准事项见 [`../rl-infra-spec/alignment.md`](../rl-infra-spec/alignment.md)。

## 2026-09-29（Agent ALIGN，阶段 0）

- 规划文档从 `/home/michael/work/rl-algos`（分支 `rl-algorithms` HEAD 18695ae，含未跟踪文件）原样复制到分支 `rl-infra-spec`（提交 `5753e30`），此后以本分支副本为准；rl-algos worktree 未改。
- P0 框架；注册入口先于四个子 change。与 infra 的接口：A1（execution 能力由执行模式给出，算法哈希即 ExecutionProfile 契约身份）；`driver.py` 4.2 按 alignment §7 顺序。
- 任务状态：全部未完成（0 勾选），没有代码改动，没有使用 GPU 或云资源。
- `openspec validate rl-algorithm-capabilities --strict`：见 rl-infra-spec/progress.md 的 ALIGN 条目。

## 待批准

见 alignment.md §8（与本 change 相关的是：GPU 预算、`miles_adapter/config.py` 与 `entry.py` 的归属、“commit/push 需用户确认”一句的澄清）。

## 下一步

按 alignment.md §7 的工作包派发。
