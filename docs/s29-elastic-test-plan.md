# S29 弹性 RL / 纯 spot rollout 岛健康测试设计

目标：在真实的跨云拓扑上，对 yeto 的"纯 spot rollout 岛 + 同步 trainer 组"做完备健康测试，尽量暴露缺陷。

## 1. 被测拓扑

| 角色 | 位置 | 说明 |
|---|---|---|
| syncer | npu1 CPU，`/data/work/s29-spot/yeto-syncer`（aarch64 musl） | `--learners 2 --max-subscribers 2 --sample-relay`，full-parameter async，8 fragment |
| trainer 岛 0 / 1 | n4 卡 0 / 卡 1（避开链路抖动的卡 4、5），verl FSDP2 全参数 relay trainer | Qwen3-0.6B，每轮 1 组 × 4 条，`--rl-max-policy-age 4`，TIS |
| spot 岛 a（channel 0） | Nebius eu-north1，1×H100 preemptible，`s29-spot-a` | SGLang TP1 + records proxy + Harbor/Codex/boat |
| spot 岛 b（channel 1） | Nebius us-central1，1×RTX6000 preemptible，`s29-spot-b` | 同上 |
| 链路 | npu1 → VM 反向 SSH 隧道 `-R 127.0.0.1:29400` | VM 只经隧道见 syncer；boat 沙箱经 VM 公网 IP:31600 访问 proxy |

数据：TB2.1 89 题（`agentic-tb21-89.jsonl`），岛 i 取 `data_slot i/2`。boat 沙箱开 VM 复用。

控制脚本：npu1 `/data/work/s29-spot/ctl.sh`，n4 `/data/work/s29-spot/n4ctl.sh`。

## 2. 证据来源

- syncer tape：`run/yeto-tape.jsonl`，syncer 日志 `run/syncer.log`
- trainer tape：n4 `run/i{0,1}/tape.jsonl`，日志 `run/island{0,1}.log`
- spot 岛 tape：VM `/root/spot/events.<incarnation>.jsonl`，`sampler.log`，`vm_run.log`
- boat：`boat limits`（activeSandboxes、starts.day），`boat` 列表
- Nebius：`nebius compute instance list`（只看 `s29-spot-` 前缀）

## 3. 测试项与通过判据

### T1 基线链路（5 step）
- 两个 spot 岛都出 `rl_spot_sampler_channels`、`rl_sampler_load_checked`，并产出组（`rl_sampler_engine_round` groups ≥ 1）。
- syncer relay 统计 `stored` / `acked` 两个 channel 都 > 0。
- 两个 trainer 岛各完成 5 个 `rl_round_trained`，`trained_groups` ≥ 1。
- 判据：全部满足；trainer 无异常退出（ISLAND-RC=0 或按设计结束）。

### T2 tape 完整性与一致性
- 每个 tape 行是合法 JSON，`time_unix` 单调不减（同一进程内）。
- 每个 spot 岛 incarnation：恰有一个 `rl_sampler_join`，以 `rl_sampler_leave` 结束（被 kill -9 的除外，记录为缺失）。
- trainer 训练的样本 id 集合 ⊆ spot 岛写出的样本 id 集合；无重复训练同一 sample_id。
- 每个训练组的 behavior version 与 trainer 当时版本之差 ≤ `--rl-max-policy-age`。
- 判据：全部满足；违反即缺陷。

### T3 dashboard / 汇总数据正确性
- `yeto status` / 事件汇总（reward_mean、groups、tokens、step_seconds）与 tape 原始行重算一致（误差 0）。
- relay 统计（syncer 日志）与 spot 岛 tape 的组数、trainer tape 的组数三方对账。
- 判据：三方数一致，或差额可由在途/被丢弃项完全解释。

### T4 syncer 同步 / 版本推进 / staleness
- 每轮 merge 后两个岛 `policy_version` 前进；spot 岛 `PolicyFollower` 加载到新版本（`rl_sampler_load_checked` 版本号递增）。
- spot 岛加载后的 post-load probe 通过（`max_probe_jump` 0.5 内）。
- staleness 超限的组被跳过并记录（`rl_buffer_stale_skipped`），不进入训练。
- 判据：版本单调前进；spot 岛加载版本 ≤ syncer 最新版本且滞后 ≤ 2；超龄组从不进训练。

### T5 relay 样本流转
- 每个组从 spot 岛写出到 trainer 消费有完整链：producer seq → syncer stored → consumer acked → trainer buffer。
- channel 隔离：channel 0 的组只进岛 0，channel 1 只进岛 1。
- 判据：无丢失（除被 fence 的 producer 的未 ack 项且有记录），无串道，无重复。

### T6 抢占与故障注入（每项单独记录恢复时间与恢复后状态）
| 编号 | 注入 | 期望 |
|---|---|---|
| T6.1 | kill -9 spot 岛 sampler 进程（VM 不动），再重启 | 新 incarnation supersede 旧的；relay 未 ack 项按设计处理；trainer 不崩，继续训练 |
| T6.2 | SIGTERM（模拟云抢占通知） | 写 reclaim marker，以 PREEMPTED 离开（`rl_sampler_leave` reason PREEMPTED） |
| T6.3 | sky down 一台 VM 再重开 | trainer 等待（不超时退出）；新 VM 订阅后从当前版本继续出样本 |
| T6.4 | 断隧道 60 s 后恢复 | subscriber/relay 重连；`syncer_lost_s` 内不退出；无重复样本 |
| T6.5 | syncer 重启（带 `--resume`） | trainer 与 spot 岛重连；版本与 cursor 不回退 |
| T6.6 | trainer 岛 kill -9 后重启 | 岛从 checkpoint 恢复 cursor/版本；relay channel 上积压组被正确消费 |
| T6.7 | 两个 spot 岛同时断开 | trainer 等待；`--rl-sample-wait-s` 到期行为符合设计 |

判据：每项期望全部满足；恢复后版本号不回退、样本不重复、cursor 正确。

### T7 恢复后版本与 cursor 正确性
- 每次注入后核对：trainer `train_step` 连续，`policy_version` 不回退，spot 岛 `data_cursor` 不重复取同一 prompt 行（除设计允许的回收）。

### T8 多岛 quorum / grace
- quorum=1：一个 trainer 岛停住时另一个仍推进；停住的岛回来后追上（`max-base-lag` 内）。
- 判据：syncer tape 的 responders 与 grace 记录符合配置；无死锁。

### T9 长跑资源泄漏（30+ step）
- trainer：NPU HBM、host RSS 随 step 不单调增长（首尾差 < 10%）。
- spot 岛：GPU 显存、进程 RSS、`/tmp/harbor-trials` 目录、weights 目录（旧版本应清理）不无界增长。
- syncer：RSS、relay depth 不无界增长。
- 判据：首尾差 < 10% 或有界。

### T10 boat 沙箱泄漏
- 运行中 activeSandboxes ≤ 各岛 `max_in_flight` 之和（8）。
- 每次注入（kill/down）后 ≤ 20 分钟内（idle VM TTL）活跃数回落；全部结束后为 0。
- 判据：结束后 `boat limits` activeSandboxes = 0。

### T11 错误注入
- 损坏的 relay 项 / 错误 HMAC 的连接被拒（syncer 记录 refused）。
- 错误 layout 的 trainer 被拒（已在基线中观察到 layout hash 校验）。
- spot 岛加载失败（如 probe 超阈值）进入 `max_consecutive_load_failures` 处理。
- 判据：错误被显式拒绝并记录，不静默。

## 4. boat 额度策略

当日剩约 22 次创建（UTC 0 点重置为 200）。真实 TB2.1 trial 优先用于 T1/T6；额度耗尽时，链路类测试（T4/T5/T6/T9）改用不依赖新建沙箱的固定 reward 路径继续，重置后补跑真实 TB2.1。

## 5. 成本护栏

VM 总花费 ≤ $60，并发 ≤ 3 台，空闲等待时关机；只管理 `s29-spot-` 前缀资源。

## 6. 执行结果（10-31 03:45）

| 项 | 结果 | 证据 / 缺陷 |
|---|---|---|
| T1 基线 TB2.1 5 step | 通过（两个岛各训 5 个本地 step，每步 4.4–9.6 s，reward 全 0） | D1 D3 D5 D6 已修；docs/s29-evidence/t1-tb21 |
| T2 tape 一致性 | 发现问题后已修：重复训练（D6 跨 run，D12 重启） | D6 D12 |
| T3 三方对账 | relay stored/acked、本地 buffer、trainer 训练数一致（修复后） | final/*.tgz |
| T4 版本推进/staleness | 合并推进正常；spot 岛跟不上版本（WAN 0.18 MB/s） | D9 D10 D13 |
| T5 relay 流转 | 无丢失、无串道；崩溃时已 ack 的组没有丢（D2 关闭） | D2 |
| T6.1 kill -9 sampler | 修复后通过（孤儿 SGLang 进程组被回收） | D8 |
| T6.2 SIGTERM | 通过（PREEMPTED 离开，boat 活跃数归 0） | — |
| T6.3 sky down VM | 已 down；重开放到 08:00 后的 TB2.1 回归（boat 额度） | — |
| T6.4 断隧道 60 s | 通过（重拨 7 次后自动恢复，未退出） | drill.log |
| T6.5 syncer --resume 重启 | 通过（恢复到 step 42，全部成员重连，无 fence） | drill.log |
| T6.6 trainer 岛 kill -9 | 默认配置下失败（D11）；开 island pool 并修 D12 后通过 | D11 D12 |
| T6.7 两个 spot 岛同时离线 | trainer 等待不退出（D10 修复后）；未单独计时 | D10 |
| T8 quorum/grace | quorum=1 时单岛也能推进合并（syncer tape responders 只有 1 个） | syncer.log |
| T9 长跑泄漏 | 合计约 150 轮；trainer 共享内存 +35% 待确认 | D14 |
| T10 boat 泄漏 | 通过（全部结束后 activeSandboxes=0） | — |
| T11 错误注入 | layout hash 不符、policy age 超出 verl 上限都在启动时被显式拒绝 | 基线日志 |
