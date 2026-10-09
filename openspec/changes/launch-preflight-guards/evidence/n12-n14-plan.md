# 任务 4.1–4.3：N12/N14 负例上卡复核与预登记

写于 2026-10-09（LPG 子 agent）。按 memory gpu-review-before-launch：先捋清代码路径、写复核结论，再预登记、再开机。

## 批准（4.2）

- 主 agent 按用户的代拍板授权批准，预算上限 $20（10-09 派单）。
- 三个运行（10-09 主 agent 代用户拍板，多条消息合并后的最终版）：
  1. legacy 调度（strict-avg，HELLO 带会话契约哈希）＋ 岛 1 `rl_lr_schedule=constant`（岛 0 为 auto＝linear）。
  2. elastic ＋ 岛 1 `identity_test_salt=n12neg`。
  3. elastic ＋ 岛 1 `rl_max_policy_age=1`（岛 0 为 0）。
- legacy 无 HMAC、syncer 端口对公网开放：主 agent 允许本次使用，条件是运行尽量短、结束立即拆除（风险见下文）。
- 负例岛晚连 180 s：主 agent 拍板加入（见 design 决定 4）。

## 形状

- 脚本 /home/michael/work/s1-runs/s18-lpg-neg.sh（由 s17-g1-island.sh 改）：Nebius eu-north1 无卡 head（cpu-d3 8vCPU，约 $0.20/h）＋ 两个 Modal 岛各 1×H100!（`--modal-gpu-exact`），Qwen3-0.6B LoRA r16 all-linear，gsm8k，4×8/轮，上下文 1024、回复 384，lr 1e-5，seed 17。
- 与 g1 的差别：去掉 killer/stopper/pause；去掉 `--modal-launcher-relaunch` 和 `--recover-timeout`（被拒的岛立即拆掉，不重开）；加 `--rl-negative-test-run --rl-island-override $OV --preflight-threads wait`。
- 代码：worktree /home/michael/work/s18-lpg（分支 s18-lpg，开机时的提交写进各运行目录 yeto_sha.txt），经 git archive 打包到 head，岛从 head 的工作目录拿到同一份代码。镜像用 main 钉住的 MILES_NEXT_IMAGE，不重建。
- 轮数与时限：legacy STEPS=4、HARD=1800 s（看门狗 2400 s）；elastic STEPS=10、HARD=2400 s（看门狗 3000 s）、`--rl-soft-deadline-s 60`。

## 代码路径复核（逐项）

1. 线程预检（首次实地使用）：本机 `cmd_launch_head` 在 `prepare_launch_args` 之后、起 head VM 之前调 `pre_cloud_checks`（线程＋显存＋换参数校验＋续训标记）。脚本不再自己卡线程数，只把 `ps -L` 读数记进 threads-ps.txt 作对照。head 上 `launcher.run` 再读一次 head VM 的线程数。
2. 显存预检：dry-run 已跑，两岛估算 36.6 GiB ≤ 71.3 GiB（0.9×79.18），通过；Qwen3-0.6B 未校准，超限也只告警。
3. 换参数：dry-run（PLAN_ONLY）三种配置都过：岛 1 命令行分别带 `--rl-lr-schedule constant`、无变化（盐只走环境变量）、`--rl-max-policy-age 1`；岛 0 命令行不变。真实 sky.Task → `build_modal_island_config` 后，岛 1 的 Modal envs 有 `YETO_ISLAND_OVERRIDE`（含 join_delay_s 180），两岛都有 `YETO_NEGATIVE_TEST_RUN=1`（gpu-head venv 实测）。
4. 岛侧：`run_miles` 装好 tape 回显后调 `island_startup`：写 `rl_island_override`（经 append_record，Modal 岛回显到日志流），查 checkpoint store 标记（本次没有 store，跳过），负例岛等 180 s。Ray 作业级 env_vars 透传两个变量（单测覆盖）。
5. 身份：legacy 走 `StrictRlBridge`（bridge.py:282），契约＝layout＋`island_contract_sha256(身份, LR 哈希)`；岛 1 constant 与岛 0 linear 的 LR 哈希不同 → 契约不同。elastic 走 `ElasticAvgSync`（engine/bridges.py:439），JOIN 带 `island_contract_sha256(..., test_salt)`；盐或 `bind_policy_age(…,1)` 都会改变身份。
6. syncer：legacy 第一个 HELLO 定契约，之后不同的 HELLO 只拒这条连接（server.rs:1140，#143 已在 main）。elastic 第一个被接纳的 JOIN 钉身份，之后不同的 JOIN 回 "backend identity mismatch, JOIN refused"（elastic_server.rs:398，#140 已在 main）。岛 1 初次 JOIN 被拒直接抛错退出（elastic_client.py:757），不会长时间重试。
7. 先到先定：负例岛晚 180 s 启动。两岛在 head 的同一个循环里先后提交，容器开机差一般在几十秒内，所以岛 0 先定契约。若岛 1 仍先连上，被拒的会是岛 0，判据 P3/P4 记"失败"，不改判。
8. launcher 对被拒岛的处理：legacy 是固定名单，岛 1 退出 → 恢复超时为 0 → FixedRosterIslandAbandoned → 整场停（按设计，N14 早报）。elastic 修复后：岛 1 日志含拒绝标记 → 只拆岛 1、记 island_lost、不重开，岛 0 继续。
9. 证据采集窗口（memory gpu-evidence-window-lesson）：
   - 拒绝发生在岛 1 开机＋180 s 之后。elastic 下岛 0 每轮约 1 min（N17 elastic：6 轮 15 min 含开机约 8 min），10 轮约 10 min，比 180 s 加开机差长得多，所以拒绝发生时岛 0 还在训练。
   - syncer 日志在 head 上（~/yeto-syncer.log），脚本在拆 head 之前打包拉回（head UP 时才拉）。岛日志：Modal app logs（app stop 后按 app id 拉）与 tape volume `yeto-event-tapes`（拆后拉）。拒绝信息同时出现在岛 1 的 stderr（Modal 日志）和 syncer 日志里，两处都在拆机后仍可取。
   - legacy 整场停得快：拒绝之后 launcher 拆岛、head 作业 FAILED，脚本随即拉 head 日志再拆 head，日志不会丢。
10. legacy 风险：syncer 端口对公网开放且无 HMAC，任何人都能连。运行时长约 15 min（开机约 8 min ＋ 180 s ＋ 拆除），结束后 teardown.sh 立即 `modal app stop` 与 `sky down -y <本次 head>`（只拆本次，不用 `sky down -a`），看门狗 2400 s 兜底。
11. 线程：派单要求开机时 <2800。开机前本机读数约 2790（ARU-3 正在上卡）。launcher 预检按 wait 模式等到 <2800 才继续，最多 1800 s，超时报错不花钱。

## 预登记判据（每个运行，judge：s1-runs/s18-lpg-judge.py）

- P1 预检真机执行：本机提交日志有 `[preflight] threads OK: N < 2800`，两岛都有 `[preflight] island i H100: estimated peak`，并打印 NEGATIVE-TEST RUN 警告。
- P2 只有岛 1 的 tape 有 `rl_island_override` 事件。
- P3 岛 1 被拒，原因与参数对应：legacy 为 "session mismatch (HELLO refused, session keeps running)"，elastic 为 "backend identity mismatch, JOIN refused: island 1 declares"，且拒绝信息含两边哈希（两个 64 位十六进制）。
- P4 只拒这条连接：syncer 没有致命退出（无 panic、无 layout_hash_mismatch 退出），岛 0 没有被拒。
- P5（只 elastic）岛 0 完成全部轮次（local_round_id 1..N）并写 rl_learner_finalized。legacy 不适用：固定名单按设计整场停下。
- P6 看板：用运行时的 reducer 读全部事件，岛 1 卡片有"负例岛：…"，岛 0 没有。
- 结论只填"通过"或"失败"。证据缺失记"失败（证据不全）"，不补数。

## 费用（预登记写进台账，只写本笔）

- 单价：Modal H100! 约 $3.95/h·卡，head 约 $0.20/h。
- 期望：每个运行约 15–25 min ×2 卡 ≈ $2–3.3，三个合计约 $8–10。
- 最坏（看门狗时限）：legacy 2400 s ×2 卡 ≈ $5.3；elastic 3000 s ×2 卡 ≈ $6.6 ×2；head 合计约 $0.5。总计约 $19，不超过 $20。
- 结束后按 app id 核对 Modal app 已停，按 head 集群名核对 Nebius VM 已删。

## 复核中发现并已处理的问题

- 先到先定契约 → 负例岛晚连 180 s（主 agent 拍板，design 决定 4）。
- RL 运行一律按固定名单（launcher.py:7281），elastic 被拒岛退出会停整场 → 主 agent 拍板修根因：`fixed_roster` 只对非 elastic 为真，elastic 下被拒或严格失败的岛只拆该岛、记 island_lost、不重开（单独提交，记入 rl-inter-island-scheduling tasks 10.1）。
- 岛 1 不会写 rl_learner_finalized，运行结束码可能是 3（tape 不完整），按实际记录，不算判据失败。
