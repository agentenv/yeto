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
