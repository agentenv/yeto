# A4 / A4b 8 卡 H100 批 结果（2026-10-01，代码 b19b781，计划 gpu-plan-v2 §9.23）——进行中的中间版本

平台 Nebius eu-north1 8×H100 on-demand（$30.8/h）。每项只给四态：通过 / 不通过 / 测试无效 / 未运行。tasks.md 勾选状态未改动。

## 当前状态（随运行更新）
| 项 | 状态 | 依据 |
|---|---|---|
| 冒烟（测量） | 完成（无验收结论） | up SUCCEEDED 144.3 s、start_cells 142.4 s（≤160 s 停止线）、down 5.3 s；cleanup 闭环查出缺陷（已修，base 验证） |
| E1-A 基线 base | 完成（作为 e1a 的比较基线） | 12 轮 rc=0 |
| E1-A e1a | **通过**（(c) 首次判读 FAIL 为工具缺陷，已勘误重判） | 见下 |
| E1-B | 未运行（被线程防护阻塞，见"阻塞"） | |
| watchdog / A4b / E1-D | 未运行 | |

## 冒烟 infra-v2-b1-a8sm-20261001-1（runs/smoke/）
- 平台断言 8×H100 通过，指纹 `sha256:1c4ceeb1…` 与 rl_driver_start 一致，router 采样器与 fork 探针自检通过。
- **阶段耗时（launch 起算，s）**：sky 实例创建（Launching→Instance is up）≈ 302；拉镜像/起容器（→Docker container is up）≈ 521（至 02:34:38）；Cluster launched 863；job submitted 870；岛首条日志 897；首个 generate 1187；up 事务 144.3（其中 start_cells 142.4，fork_op start issued→VERIFYING）；down 5.3；Job finished 1395；launcher 拆除 1412→1534（122 s）。整次 1534 s = 25.6 min，≤$13.1。→ 每次运行底价 ≈ $12–13.5，与轮数关系不大。
- **cleanup 闭环缺陷（冒烟查出，已修）**：自动 cleanup 的退出码 141（cleanup.out 只有 phase 1 一行）。根因：`nstop.sh` 用 `cleanup_run.sh | tee <run dir>/cleanup.out`，tee 的命令行含 `/<prefix>/`，被 cleanup 阶段 1 的前缀进程扫描 SIGTERM，cleanup_run 随后写管道 SIGPIPE。修复：改为 shell 重定向；`tests/test_nstop.sh` 回归测试（旧版复现 rc=141，新版 rc=0）。真机验证：base、e1a 的 cleanup_rc=0，cleanup.out 四阶段齐全，两次复查（间隔 60 s）干净。冒烟本身的 cleanup 用手动 `cleanup_run.sh` 补做：退出码 0（runs/smoke/cleanup_manual.out 在运行目录，已复制）。
- **采集缺口（已修）**：岛随 job 结束被 launcher 拆除，router/gpu 采样、inwatch 日志没拉到；n2run puller 加了周期性小文件包（`samplers.tgz.b64`），base 起生效。

## base infra-v2-b1-a8base-20261001-1（runs/base/）
12 轮，rc=0，无请求。1405 s，≤$12.0。作为 e1a 的 sample-id 基线。cleanup_rc=0。

## E1-A e1a infra-v2-b1-a8e1a-20261001-1（runs/e1a/）——通过
注入/请求真实发生：触发器在 train rid1 提交 up1、rid4 重复提交同一 id、rid6 提交 dn1（`samplers/inwatch.out`）；两事务 SUCCEEDED（up 总 146.3 s，start_cells 143.9 s；down 5.3 s）。1578 s，≤$13.5。
| 判据 | 结果 | 依据 |
|---|---|---|
| (a) 两事务 SUCCEEDED，config_epoch 0→1→2 | 通过 | analysis.json a |
| (b) 12 轮 sample-id 哈希与基线相等，每轮 1 个 optimizer 步 | 通过 | analysis.json b，differ=[] |
| (c) 成员数序列（rl_membership）+ 缩容后首次发布成员数 | **通过**（见勘误） | judgment.json：`[2,2,4,4,4,4,4,2,2,2,2,2]` 精确相等；缩容后首次 rl_publication（pv8）2 成员 |
| (d) trainer PID / G0–G3 UUID 全程不变 | 通过 | 16 次采样唯一 |
| (e) 池外 GPU 无新进程 | 通过 | 36 次采样，池外 0 |
| (f) down：QUIESCING→TRANSFERRING→stop_cells | 通过 | |
| (g) 备用卡 GPU-hours | 记录 | 8 × 26.3 min |
| E1-E | 通过 | 重复提交同一 request_id 同一 tx（status 文件 tx-0-…-up1，journal 仅 1 条 up1 request）；WAIT_SAFE 边界 up1=rollout 2、dn1=rollout 7；ledger 12 轮每类恰 1 条 |
**(c) 原始 FAIL 与勘误**：第一次判读（`judgment.v1-FAIL.json`，`judge.v1.out` 原样保留）输出 FAIL，`members_per_round=[null,4,4,4,4,4,2,2,2,2,2,2]`，且 `publication_after_down_is_1_member=false`。原因是工具缺陷而非产品：(i) 磁带的 `rl_membership.round` 是 0-based rollout_id（up=2、down=7），judge 当成 1-based 轮次（off-by-one）且第一个事件之前没有初始成员；(ii) "缩容后发布成员数"写死成 4 卡的 1。§9.22/§9.23 里"`round ≤ r`"是笔误，应与 §9.18 的 rollout_id = r−1 一致；勘误在判读前写入 §9.23 并连同修复提交（a4960bb），回归测试用真机 e1a 磁带片段，断言新读法 PASS、旧读法 FAIL。预期序列与判据文字未改。
