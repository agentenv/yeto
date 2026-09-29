# 2.6 运行结果（Megatron upstream parse_args）

- **第 1 次**：sandbox sb-19VMbFwFBWDkDbQKMCyrsA，只用 CPU，失败。原因是 `transformer_engine` 找不到 `libcuda.so.1`，一例都没有解析，不计入结果。日志：`attempt1/run.log`。
- **第 2 次**（按 `attempt2-plan.md`）：sandbox sb-Yj4cmBPAv7aacpvyoirEHa，Tesla T4 ×1；镜像 digest 5da40a07…；`/root/miles` HEAD 为 0394715083c9…，等于 MILES_NEXT_COMMIT。
  - **20/20 通过**：默认 GRPO 加 19 个非默认映射。每例都对 ports `translate_run_config` 生成的完整 argv 跑了 `mc.parse_miles_args`，也就是 megatron 后端的 upstream `parse_args`，其中包含 Megatron `validate_args`、`miles_validate_args`、`sglang_validate_args` 和 yeto 的 `validate_parsed_args`；expected 中的字段全部相等。
  - 默认 GRPO 例的 argv 与本地用同一 config 做的翻译逐字节一致。
  - 日志与每例 argv/结果：`attempt2/run.log`、`attempt2/results.json`。
- **资源与回收**：Modal app `algocap-parse`（ap-PjhAyHI1wEyCDN4MAl7LPx），两个 sandbox 都在脚本的 finally 中 terminate；之后执行 `modal app stop -y`，`modal app list` 显示 stopped、0 tasks；`Sandbox.list` 已查不到该 app。看门狗进程已结束。
- **费用**：第 2 次包括镜像拉取在内约 3 分钟，T4 加 CPU 估计 < $0.1；第 1 次只用 CPU，约 2 分钟，< $0.05。均未经账单核实。

## 复审更正（2026-09-29，E1/E2/E3）

- **比较口径偏离（E2）**：`../2.6-plan.md` 的成功条件写的是“默认 GRPO 例的 argv 与 `tests/test_rl_argv_snapshot` 使用的翻译逐字节一致”，但 `parse_all.py` 没有做这项比较。我事后用“与本地同一 config 翻译一致”代替了它，这属于事后更改比较口径。另外，原计划的口径本身也有误：`test_rl_argv_snapshot` 测的是 legacy argv，不是 ports argv。
- **结论**：第 2 次运行不满足原计划的全部成功条件，2.6 改回未勾选。20/20 解析通过这一事实仍然保留，但不能作为 2.6 的验收依据。新计划见 `attempt3-plan.md`。
- **yeto 版本（E3）**：第 1、2 次上传的是 algo-cap 工作树。运行时没有记录 SHA；按提交时间和当时工作区干净推断，yeto/ 和 tests/ 与 `b76a91f` 相同（`7fd5bdd` 只改了证据文件）。这是推断，未在运行时核实。
- **日志入库（E1）**：`attempt1/run.log` 和 `attempt2/run2.log` 已用 `git add -f` 入库；入库前扫描过 GitHub/HF/Modal/W&B 凭据模式，没有命中。
