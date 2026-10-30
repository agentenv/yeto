# S29 弹性 RL / spot rollout 岛缺陷记录

格式：现象 / 复现 / 根因 / 修复 / 复测。

## D1 FSDP buffer trainer 的 prompt 行宽固定 512，agentic 组一进来就崩

- 现象：trainer 岛 0 收到 spot 岛 a 的第一组即退出：`BufferFeedError: group buf0 sample 0: prompt 7940 / response 1010 tokens exceed the rows (512 / 20480)`，`island_leave reason INFRA_FAILED`。
- 复现：verl FSDP 全参数 trainer，`--rl-sample-role trainer --rl-sample-relay`，`--seq-len 32768 --rollout-max-response-len 20480`，喂 agentic 组（Codex 系统提示约 8k token）。
- 根因：`island_entry.main` 只在 `--rl-agentic-harbor` 时按序列预算给 prompt 行宽（`seq_len - response_len`），否则写死 512。buffer trainer 本身不跑 Harbor，没有这个 flag，但它训练的正是 agentic sampler 写出的组。Megatron 路径（`megatron_island_plan`）一直按序列预算，所以 S28/S29 的 120B 运行没暴露。
- 修复：`sample_role == trainer` 时同样按序列预算（`yeto/rl/adapters/verl/island_entry.py`）。
- 复测：见下文 T1 结果。

## D2 trainer 崩溃时，已从 relay ack 的组随之丢失（观察，待确认）

- 现象：D1 崩溃前 syncer relay 统计 channel 0 `stored=1 acked=1`，trainer 随即退出，该组没有被训练，也没有回到 relay。
- 待确认：relay pump 是先 ack 再写本地 buffer，还是写入本地 buffer 后 ack；本地 buffer（`--rl-sample-store`）在重启后是否会重新读到这组（T6.6 复测）。

## D3 不足 n 条的组让 FSDP update_actor 断言失败

- 现象：trainer 岛 0 训完第 0 轮（4 条），第 1 轮收到一个只有 3 条的组（boat 里一个 trial 丢了，sampler 按 `min_samples 2` 保留了这组），`AssertionError: 3 % 4 != 0`（verl `tensordict_utils.make_iterator`），岛退出。
- 复现：`--groups-per-round 1 --samples-per-group 4`，sampler `min_samples < n`，出现残缺组即复现。
- 根因：岛把 `actor.ppo_mini_batch_size` 设为 `groups_per_round`（每轮一个 mini-batch），verl 换算成行数时乘 `rollout.n`，默认每组恰好 n 条；buffer 里的组可以少于 n 条。
- 修复：`fully_async_runner.whole_round_mini_batch`：update_actor 期间把 mini-batch 设为本轮实际行数（`rollout.n` 临时为 1），调用后恢复（`yeto/rl/adapters/verl/fully_async_runner.py`）。
- 复测：通过。trainer 岛 0 第 1 轮训练了 3 条的组（`trained_samples 3`，step 4.45 s），随后第 2 轮 4 条正常。限制：dp>1 时还要求行数能被 dp 整除，本次 dp=1 未覆盖。

## D4 syncer 不带 --resume 重启后，spot 岛把旧 run 的组重发给新 run（观察）

- 现象：syncer 三次 fresh 重启（无 `--resume`）期间，spot 岛 a 的 subscriber 和 relay producer 只是"connection reset; redialing"，随后照常连上新 syncer，没有被 fence。新 run 的 trainer 岛 0 在 11 s 内训了 3 个组（16:37:32 / 38 / 43），这些组大多是 spot 岛 a 在前几个 syncer run 里生成的。
- 影响：这次权重都是 v0，所以结果无害。但如果旧 run 已经推进过版本，旧组带着旧 run 的版本号进入新 run，`max_policy_age` 的判断就没有意义了。
- 根因（待查）：PROTOCOL 规定"from before a syncer restart"要 fence，但 syncer 不带 checkpoint 重启时没有 epoch 可比，客户端的 incarnation 在新 syncer 上算新成员，直接被接受。
- 处理：先记录；T6.5 再用 `--resume` 复测"真重启"的行为。

## D5 spot 岛在首个合并 cut 之前 backlog 只增不减，与 fragment 逐个合并互相等待（死锁）

- 现象：两个 spot 岛各出 1 组后一直停在 `rl_sampler_backlog_full`（`backlog_groups 1`，`max_backlog_groups 1`），而 syncer relay 显示两条 channel 都已 `acked=1`、`depth_items=0`，trainer 在等样本。整条链路停住约 15 分钟，VM 空转。
- 复现：spot 岛 + full-parameter 同步组，`max_backlog_groups` 小于产出首个完整 cut 所需的 trainer 轮数（这里 8 个 fragment，每个 outer step 只合并 1 个 fragment）。
- 根因：`SamplerLoop.backlog` = 写出组数 − `follower.last_seen.buffer_selected`。spot 的 `SubscriberReleaseStore.latest()` 在第一个合并 cut 之前返回 None，`last_seen` 为空，读位置按 0 算，于是 backlog = 累计写出组数，只增不减。第一个 cut 需要 8 次 outer step，即 trainer 要训 8 轮，而 trainer 要等 sampler 的组，sampler 要等 cut：互相等待。
- 修复：relay 上的 backlog 直接取 channel 里尚未被 trainer 取走的组数（`RelaySampleBuffer.pending_groups`，`SamplerLoop.backlog` 优先使用），不再依赖 release 的读位置（`yeto/rl/spot_rollout/relay_buffer.py`，`yeto/rl/engine/sampler_loop.py`）。
- 复测：见 T1 续跑结果。

## D2 更新：没有丢，但暴露了 D6

D1 崩溃前被 ack 的那组并没有丢：relay pump 先把组写进 trainer 的本地 buffer 再 ack，本地 buffer 在下次启动时被重新读到（见 D6）。所以"trainer 崩溃 → 已 ack 的组丢失"不成立，D2 关闭。

## D6 relay trainer 的本地 buffer 放在主机临时目录、只按 learner id 区分，跨 run 串数据、重复训练

- 现象：n4 `/data/tmp/yeto-relay-buffer/island0/sample-buffer/index.jsonl` 里，位置 0、1 是 16:28、16:32 由上一次（已崩溃的）trainer run 写进来的组；16:35 起的新 run（新 syncer、新 run 目录）从位置 0 开始训练，把这两组又训了一遍（位置 0 在崩溃的 run 里已经训过）。这也解释了"relay channel 0 只 stored=3，岛 0 却训了 5 组"。
- 复现：同一主机上先后启动两个 relay trainer run，learner id 相同（几乎总是 0/1），第二个 run 一启动就先训第一个 run 留下的组。两个 run 并发时还会写同一个目录。
- 根因：`relay_trainer.relay_buffer` 用 `tempfile.gettempdir()/yeto-relay-buffer/island<learner_id>`，和 run 无关；`buffer_trainer_runner` 调用时没有传 `tmp_root`。
- 修复：本地 buffer 放进该岛自己的 `--rl-sample-store`（本地路径时），同一个 run 重启仍能读回自己的 buffer（`yeto/rl/adapters/verl/buffer_trainer_runner.py`）。
- 复测：见 T6.6。

## D7 `--global-rounds` 在 full-parameter 异步同步下表示"同步轮"，本地训练步数是它的 fragment 倍（口径问题）

- 现象：`--global-rounds 40 --fragments 8` 的 trainer 岛训到第 53、55 步还在继续；T1 时的"5 step"实际是在 5 个本地 step 时手动停的，按配置应训 40 个本地 step。
- 根因：`--total-fragment-steps = global_rounds × fragments`，syncer 每个 outer step 合并 1 个 fragment，岛每个本地 step 推 1 个 fragment，到 syncer 走完 `total_fragment_steps` 才结束，所以本地 step 数 = `global_rounds × fragments`。launcher 和岛的参数名、日志都没有说明这一点。
- 处理：不改代码；在测试计划和汇报里按"本地 step"计数。建议 launcher 打印"本地训练步数 = R × P"。

## D8 kill -9 spot sampler 后 SGLang scheduler 成为孤儿，继续占显存，同机新进程起不来

- 现象：T6.1 `kill -9` spot 岛 a 的 sampler 进程后，`sglang::scheduler`（pid 10712）继续占用 58.7 GiB 显存。新进程的 sky job 一直排队（旧 job 的 pipe 被孤儿占着），即使排上，也会因显存不足起不来 SGLang。
- 复现：spot 岛运行中对 sampler 进程 `kill -9`，然后在同一台 VM 上重启 sampler。
- 根因：server 适配器（`sampler_engines/server.py`）启动的 SGLang 与岛同进程组，只有 `stop()`（SIGTERM）会结束它；SIGKILL 无法被捕获，scheduler 子进程就留下了。真实抢占时 VM 会随之消失，影响不大；但"同机进程重启"这条恢复路径（OOM 被杀、看门狗重启）会被卡住。
- 修复：server 改为在独立会话/进程组里启动，启动时把进程组号写入 `<tmp>/yeto-<engine>-server-<port>.pgid`；下次启动前若该文件存在，就对旧进程组发 SIGKILL（`reap_stale_server`）；正常 `stop()` 后删除该文件（`yeto/rl/adapters/sampler_engines/server.py`）。
- 复测：见 T6.1b（修复后再做一次 kill -9 → 重启）。

## D9 重启后的 spot 岛先在 base 权重（v0）上出样本，不等当前合并版本

- 现象：T6.1 重启 spot 岛 a（新 incarnation `aa489912`，17:38:51 join，`latest_release null`）。它立刻在 base 权重 v0 上生成组，而这时 run 的合并版本已经到 6–7。trainer 岛 0 拉到的 8 个样本全部超出 `max_policy_age 4`。
- 根因：`SubscriberReleaseStore` 在拿到第一个完整 cut 之前，把 base_dir 当作 v0 提供；`SamplerLoop` 的 `require_release` 对 spot 岛为假，所以重启后马上开轮。新 subscriber 要等 syncer 把 8 个 fragment 全部发完才能拼出第一个 cut，在 npu1→Nebius 的链路上要几十分钟（见 D10 的带宽数据）。
- 建议修复：run 已经过了第一次合并（syncer 告诉 subscriber 当前版本 > 0）时，spot 岛应等到当前 cut 再开轮（`require_release`），或者至少按当前合并版本判断自己的版本差，超过 `max_policy_age` 就不开轮。

## D10 relay trainer 一轮样本全部超龄就以 INFRA_FAILED 退出；跨洲 WAN 下 spot 岛跟不上版本

- 现象：GSM8K 长跑中，两个 trainer 岛都退出：`RuntimeError: rollout 57: 8 queued samples pulled, only 0/1 within the policy-age limit`（岛 0）、`rollout 91: …`（岛 1），`island_leave reason INFRA_FAILED`。岛 1 的最后几轮 `rl_group_policies` 显示 `ages {"4": 1}`：组的行为版本 v2，trainer 合并版本 6，正好卡在上限，下一次合并就全部超龄。
- 根因 1（设计）：buffer trainer 的轮收集（`fully_async_ports` 的 `RoundCollector`）把"拉满一轮但全部超龄"当作致命错误。对 spot 岛来说这是正常的瞬时状态（抢占后重启、链路慢），应当丢弃超龄组、继续等新组，而不是让整个 trainer 退出。
- 根因 2（环境，放大了根因 1）：spot 岛拿新权重太慢。npu1→Nebius 的 SSH 隧道实测不到 250 KB/s（100 MB 400 s 未完成）；Qwen3-0.6B 一个 bf16 cut 约 1.2 GB。spot 岛 b 的加载记录是 v1 17:25:58、v2 17:27:47、v3 17:46:22，而同期 trainer 每 1–2 分钟推进一个合并版本。
- 处理：先记录；长跑和后续演练把 `--rl-max-policy-age` 放宽，以便继续测其他机制。trainer 侧"全部超龄即退出"需要改成"丢弃并等待（有上限）"，见修复记录。
