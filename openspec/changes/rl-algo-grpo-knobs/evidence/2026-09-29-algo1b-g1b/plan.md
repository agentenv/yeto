# algo1b-g1b 计划：证明 clip_higher / over_sampling 在 GPU 上生效（实验前提交，事后不改）

背景：第一次 G1 中 clip_higher 与 dual_clip 的 clipfrac 为 0（每轮 1 步、on-policy，ratio≡1），over_sampling 的补采次数为 0。主 agent 决定：只有 GPU 上确实生效的机制才声明。

## 入口
- `yeto launch --rl-single-island-no-sync --controller local --rl-optimizer-steps 2`，Modal 岛 modal:1xh100。事件磁带回传使用 algo-cap 2bce8ed 的修复，判据只读取回的事件磁带和岛日志；若事件磁带取不回，该 run 判为"证据不足"，不改用其他来源补判。
- 模型：Qwen/Qwen3-0.6B@c1899de2；数据：zhuzilin/gsm8k@0cbd9f31；reward：`yeto.rl.math_reward:score`；LoRA r16 all-linear；seed 17。

## 两个 run
| run | spec | 训练配置 | 生效判据 |
|---|---|---|---|
| A clip_higher | eps_clip 0.001、eps_clip_high 0.002（把 clip 窗口收到最小，使第 2 步的 ratio 落到窗口外） | rollout-batch-size 4，n-samples 8，--rl-optimizer-steps 2，inner-lr 1e-4，total-steps 3 | 至少一个训练 step 的 pg_clipfrac > 0，且全部有限 |
| C over_sampling | bounded filter，max_replacements 2，over_sampling_batch_size 16 | rollout-batch-size 8，n-samples 8，--over-sampling-batch-size 16，--dynamic-sampling-filter-path bounded，--dynamic-sampling-max-replacements 2，optimizer-steps 1，total-steps 5 | 事件中 dynamic_filter_dropped_groups 或 dynamic_filter_replacement_attempts 在至少一轮 > 0 |

- 两个 run 还都要满足第一次 G1 的通过条件：rc=0，所有轮完成，没有 invariant 错误，指标有限。SGLang `post-warmup freeze_gc failed` 这条良性链式异常按审查决定不算失败。
- 若不触发：报告"未能证明生效"，不声明。同一 run 只有在查明原因并修复后才重跑，不换 seed 重试。
- dual_clip 本轮不跑：dual 分支只在 A<0 且 ratio > c 时起作用，Miles 没有单独记录它的指标（没有 dual clipfrac），日志里无法把它与普通 clip 区分开，本轮无法证明它生效。结论记为"未能证明生效"，不声明。需要另写一个记录 dual 分支触发次数的探针，另立计划。

## 资源、时长、回收
- 每个 run 是 1 张 H100（Modal app `yeto-algo1b-g1b-a` / `yeto-algo1b-g1b-c`）。预计每个 20–30 分钟（含镜像），两个合计 ≤ $6。
- 硬超时：
  - 本地 `timeout 3600` 包裹每次 `yeto launch`；
  - 独立 watchdog（setsid nohup）在 3900 秒时执行 `modal app stop -y yeto-algo1b-g1b-<x>`；
  - 结束后 `modal app list` 核实 stopped，写入 teardown_proof.txt。
- 串行执行：A 完成并核实回收后，再跑 C。

## run A 第 1 次尝试结论
- launcher 拉取私有 ports 镜像 `ghcr.io/michaellchung/yeto-miles-ports@sha256:5da40a07…` 时，skopeo copy 失败，Modal 镜像构建失败。原因是没有导出 registry 凭据。本次没有启动 GPU，Modal 上也不存在对应的 app。
- 修复：run.sh 在进程内从 `~/.docker/config.json` 的 ghcr.io auth 解码出 `SKYPILOT_DOCKER_USERNAME/PASSWORD/SERVER` 并导出，凭据不打印、不写日志。其余不变，重跑 A。日志在 `attempt1-out-a/`。

## 补充说明（run A 第 2 次尝试开始后加入，不改变任何判据）
- 本计划的判据只依赖 learner 进程写出的事件（rl_local_round 的 clip_fraction/dynamic_filter_*）和训练日志里的 train/pg_clipfrac，不依赖 reward_pipeline 在 rollout 子进程中写的事件，因此不受 P0 50fe818 回传范围的限制。
- 从 run C 起，yeto 代码包含 `emit_event` 的 stdout 回显（`YETO_RL_EVENT `）。
- no-sync 运行如果缺少 rl_learner_finalized，launcher 会以退出码 3 结束并标记 .incomplete。这种情况视为 rc≠0，按失败处理。

## run A 第 2 次尝试结论（18:27:42–19:27:44Z，Modal app yeto-algo1b-g1b-a，已 stopped）
- 本地 `timeout 3600` 到期后终止（rc=124），一轮训练都没完成。
- 原因是 harness 配置错误，与被测机制无关：`--reward-function yeto.rl.math_reward:score` 被当作 Miles 的 `--custom-rm-path` 使用，而 Miles 的调用签名是 `async (args, sample)`，`score(response, label)` 于是收到 args，每个 rollout 任务都抛 `TypeError: 'NoneType' object is not callable`。rollout 一直补不满批次，直到超时。
- 修复：改用 `yeto.rl.math_reward:reward_func`（Miles 签名）。其余不变，重跑 A。
- 回收：app 已 stopped。两个残留的 watchdog sleep 进程已按 pid 结束（run.sh 的 pkill 匹配式没有匹配到它们，需要修）。另外发现 `sbx.py list` 通过 App.lookup(create_if_missing) 建出了一个空的 `algo1b-g1` app，处于 deployed 状态，已 `modal app stop`。
- 费用：约 1 小时 H100，估算约 $4，未按账单核实。日志：`attempt2-out-a/`。

## run A 第 3 次尝试结论（19:29:02–19:29:06Z）
- launcher 启动时就拒绝了："run 'algo1b-g1b-a' already has a live worker (pid 1915592)"。第 2 次尝试的本地 `timeout` 只结束了前台的 `yeto launch`，launcher 分离出的 `yeto _worker` 进程仍在运行。这次没有建 app，也没有用 GPU。
- 处理：先 `yeto down algo1b-g1b-a` 并结束 pid 1915592（已核实进程不存在，Modal app 为 stopped），再修改 run.sh：结束时先 `yeto down <prefix>`，然后再 `modal app stop`。

## 三次失败汇总与重跑依据（按主 agent 要求）
| 尝试 | 失败阶段 | 原因 | 修复与提交 | 是否用 GPU |
|---|---|---|---|---|
| 1 | Modal 镜像构建 | 私有 ghcr 镜像缺拉取凭据 | 在进程内导出凭据，提交 `algo1b-g1b: run A attempt 1 … + fix`（先于第 2 次尝试） | 否 |
| 2 | rollout 阶段，一轮也没完成 | reward 入口写错：`score` 被当作 Miles 的 `custom_rm`，签名不符，每个任务都抛 TypeError | 改为 `reward_func`，提交 `… attempt 2 (wrong reward entry, timeout) + fix`（先于第 3 次尝试） | 是，约 1 小时 H100 |
| 3 | launcher 启动 | 第 2 次尝试残留的 detached worker 还在 | 结束残留 worker；run.sh 增加 `yeto down`，本条提交先于第 4 次尝试 | 否 |

- 三次都是环境或 harness 问题，都发生在第一个训练 step 之前，没有产生任何判据数据。没有因为结果不理想而重跑；判据、spec、配置、seed 都没有改动（`clip_higher.json`、`over_sampling.json` 与预登记时一致）。
- 第 4 次尝试如果能完成训练，无论 pg_clipfrac 是否大于 0，都按预登记判据如实交付，不再加跑。

## 无进展超时（主 agent 要求；在 run A 第 4 次尝试运行期间加入，不改变任何判据）
- `noprogress.sh`（setsid 后台运行，与终端无关）满足任一条件即执行 `yeto down` 加 `modal app stop` 并回收：
  - `t_start` 后 **25 分钟**内日志中没有出现任何 `'train/pg_loss'` 训练 step 行（第一次 G1 的单个 run 全程 5–7 分钟，镜像拉取也包含在这 25 分钟里）；
  - 出现第一个训练 step 之前，`Task raised exception` 与 `Traceback` 累计达到 **20 次**。SGLang 那条良性的 freeze_gc 链每次运行只有 4 行，不会触发。
- 第 4 次尝试是在它启动约数分钟后才挂上这个监控的；run C 从启动起就用它（run.sh 会自动拉起）。
- 如果第 4 次尝试仍然在第一个训练 step 之前失败：保存证据，报告阻塞，转做其他项，不再无限重试。

## run A 第 4 次尝试结论（19:30:07–19:30:21Z）
- 还是在 launcher 启动阶段就失败："event tapes already exist for run 'algo1b-g1b-a'"。第 2 次尝试的事件磁带仍留在 `~/.yeto/runs/algo1b-g1b-a/`。这次没有建 app，也没有用 GPU。
- 修复：把第 2 次尝试的 run 目录（events、meta.json）存进 `attempt2-out-a/yeto-run/` 作为证据，原目录不删；之后每次尝试用新的 cluster-prefix，`ATTEMPT=5` 时为 `algo1b-g1b-a-5`（app 为 `yeto-algo1b-g1b-a-5`）。判据与配置都不变。
- 按主 agent 的规定：如果第 5 次尝试仍然在第一个训练 step 之前失败，就保存证据、报告阻塞，不再重试。

## run A 第 5 次尝试结论（prefix algo1b-g1b-a-5，19:31:15–19:40:35Z，app 已 stopped）
- 训练：3 轮全部完成，每轮 2 个 optimizer step，共 6 个 step。事件磁带回传 33 条（`out-a/algo1b-g1b-a-5-l0-modal.jsonl`），其中有 `rl_round_trained`×3、`rl_local_round`×3 和 `rl_learner_finalized`。island job 为 SUCCEEDED。
- **生效判据（pg_clipfrac > 0）：满足。** 每轮第 1 步（on-policy）为 0.0；第 2 步分别为 0.1046、0.1046、0.1107，都有限。
- 没有 zero_grad、nonfinite、StrictRlInvariant、RoundFailed。4 行 Traceback 都属于那条良性的 SGLang freeze_gc 链。grad_norm 为 0.90/0.44/0.46。
- **退出码的偏差（如实记录）**：launcher 最后以 exit code 2 结束，原因是 launcher.py:3994 "--rl-single-island-no-sync island ran on Modal … its ~/yeto-output is not fetchable over ssh and there is no syncer checkpoint"。这是 P0 no-sync Modal 路径的固定行为：learner 已经 finalized，launcher 仍然返回 2。预登记写的是"rc=0"，照字面这一条不满足；按"learner job SUCCEEDED 且有 finalized 事件"理解则满足。这里不自行放宽，交主 agent 判定。
- 峰值显存在这个入口下没有采集（计划里也没有要求）。

### run A 第 5 次尝试的完成记录（按主 agent 判定）
run A 第 5 次：生效判据满足（第 2 步 pg_clipfrac 0.105/0.105/0.111）；判据"rc=0"字面上不满足（launcher 的 no-sync Modal 路径固定返回 2），因此本次不作为声明依据。

## 修订计划 R1（单独提交，在任何修订重跑之前）
- 只修改"运行成功"这一条的措辞；其余判据、spec、配置、seed、资源、回收、无进展超时全部不变：
  - 以下三条**同时**满足时视为运行成功：launcher 退出码为 0，或者退出码为 2 且 launcher 日志含 launcher.py 的那行提示 "--rl-single-island-no-sync island ran on Modal (...); its ~/yeto-output is not fetchable over ssh and there is no syncer checkpoint"；回传的事件磁带中有 `rl_learner_finalized`；launcher 日志中有 "job finished: SUCCEEDED"。
  - 退出码 3（缺 finalized，run 标记为 .incomplete）或其他退出码均为失败。
- 这次重跑是因为判据措辞，不是因为结果：run A 按 R1 **只重跑一次**（prefix `algo1b-g1b-a-r1`），run C 若同样出现退出码 2，也只重跑一次（prefix `algo1b-g1b-c-r1`）。结果按预登记判据如实交付，不再加跑。

## run C 结论（prefix algo1b-g1b-c，19:42:12–19:54:34Z，app 已 stopped）
- 5 轮全部完成，每轮 8 组 64 条样本；有 finalized，job 为 SUCCEEDED；没有 invariant 错误，4 行 Traceback 都是 freeze_gc 良性链。launcher 退出码为 2，原因与 run A 相同（no-sync Modal 路径无法回传产物）。
- **生效判据（事件中 dynamic_filter_dropped_groups 或 dynamic_filter_replacement_attempts 至少一轮 > 0）：不满足**，5 轮都是 0/0，generated 也是 0。另外，判据"rc=0"字面上也不满足。结论：**未能证明 over_sampling 生效，不声明。**
- 补充观察（不作为判定依据，未预登记）：Miles 的 rollout 指标中，`raw_reward_unfiltered`（0.6125/0.7125/0.875/0.675/0.8125）与训练所用的 `raw_reward`（0.64/0.77/0.84/0.72/0.77）每轮都不相同，说明生成的样本比训练用的多，超采样或过滤在 Miles 侧可能起了作用。与此同时 yeto 的 `rl_local_round.dynamic_filter_*` 在 ports 路径上可能根本没有接线（generated 恒为 0）。这属于接口缺口（INFRA/P0），不能据此补判。
- 按 R1 规定，run C 本可以再跑一次；但预登记的生效判据依赖的字段在 ports 路径上始终为 0，重跑不会改变结论，所以**不重跑**。需要先由 INFRA/P0 在 ports 路径接好 `dynamic_filter_*`（或另立判据的新计划）。
