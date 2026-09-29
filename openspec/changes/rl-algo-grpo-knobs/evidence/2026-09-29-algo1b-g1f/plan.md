# algo1b-g1f 计划：修复后的 token 聚合，与 baseline 的配对对照（实验前提交，事后不改）

## 背景
- g1c 中 token 与 baseline 的梯度逐位相同。根因（独立审查）：Miles 的 LoRA bridge 没有设置 `calculate_per_token_loss`，而 g1c 的 micro_batch_size=1，在这个条件下 args 与 config 不一致的混合状态恰好与样本均值逐位相同。
- 修复在 michaellchung/miles yeto/ports 0af62f4d，镜像为 `ghcr.io/michaellchung/yeto-miles-ports@sha256:c6f5455c8a88d131c780cf99d07c3fdd913d2beda25b6d94739db06916d602ed`，已进入集成分支 integ-decl 501d71d。

## 入口与版本
- 代码：分支 `algo-1b-token`，基于 integ-decl 501d71d（MILES_NEXT_COMMIT=0af62f4d，ports 默认镜像为上述 digest）。每个 run 的 YETO_SHA 记在 `out-<x>/YETO_SHA`，镜像 digest 以 launch.log 中的实际镜像行为准。
- `yeto launch --rl-single-island-no-sync --controller local --gpu modal:1xh100`，模型、数据、超参与 algo1b-g1 相同（Qwen3-0.6B@c1899de2，每轮 4 组 × 8 条，response 384，lr 1e-5，seed 17，3 轮，reward `yeto.rl.math_reward:reward_func`）。
- 两个 run：baseline（默认 GRPO）；token（aggregation=token，外加 `--rl-allow-unverified-mechanism loss_aggregations:token`）。串行执行。
- 无进展超时沿用 g1b 的 `noprogress.sh`：25 分钟内没有训练 step，或第一个 step 之前累计 20 次异常，即中止。

## 事先固定的判据
- 运行成功需要同时满足：
  - launcher 退出码为 0，或者为 2 且日志中有"not fetchable over ssh"那行提示；
  - 回传的事件磁带中有 `rl_learner_finalized`；
  - 日志中有 "job finished: SUCCEEDED"；
  - 3 轮完成，没有 invariant 错误（freeze_gc 良性链不算）。
  退出码 3、4、5 或其他值都算失败。
- 配对有效：两个 run 第 1 步的 `rollout/raw_reward` 相同。若不同，判"对照无效"，不声明。
- **token 生效**：配对有效，并且 token 第 1 步的 `train/grad_norm` 与 baseline 第 1 步**不相等**。若相等，判"未能证明生效"，不声明。
- 结果按预登记如实交付；只有 harness 或环境问题可以在修复后重跑，不因结果重跑。

## 资源与回收
- 每个 run 1 张 H100（Modal app `yeto-algo1b-g1f-<x>`）。硬超时：本地 `timeout 3600`，独立 watchdog 3900 秒时 `modal app stop`，无进展超时 25 分钟；结束时执行 `yeto down` 加 `modal app stop`，并用 `modal app list` 核实。
- 预计每个 run 10–15 分钟，两个合计 ≤ $3。

## 第 1 次尝试结论（21:1x Z，没有使用 GPU）
- 两个 run 都在 Modal 镜像构建阶段失败（launcher rc=1）：`launch.log was modified during build process`。原因是日志写在 yeto 工作树里，而 launcher 构建镜像时会同步这个工作树，日志在构建过程中被改写。app 为 stopped，没有产生训练。
- 修复：运行输出改写到 `/tmp/algo1b-g1f/out-<x>`（工作树之外），结束后再拷回证据目录。判据与配置不变。日志在 `attempt1/`。
- 修复（第 1 次修复后的立即补正，不涉及 GPU）：run.sh 中 out-$X 用的是相对路径，需要先 cd 到 /tmp/algo1b-g1f。补正后的第一次执行在 GPU 启动前就失败了（找不到目录），没有建 app。

## 结论（第 2 次尝试；两个 app 均已 stopped）
- 运行成功（按预登记判据）：两个 run 都是 launcher 退出码 2，日志中有"not fetchable over ssh"，事件磁带中有 `rl_learner_finalized`，job 为 SUCCEEDED，3 轮共 6 个训练日志行，没有 invariant 错误。
- 版本：YETO_SHA 为 8d30ad2（分支 algo-1b-token，基于 501d71d），token run 中 `calculate_per_token_loss = True`。launch.log 里没有打印镜像 digest，所以镜像是按这个 SHA 的 ports 默认值 `MILES_NEXT_IMAGE`（`…@sha256:c6f5455c…`）推定的，并非从日志中直接核实。
- 配对有效：两者第 1 步的 rollout/raw_reward 都是 0.90625。
- **token 生效**：第 1 步 grad_norm 为 0.4310283064842224，baseline 为 0.48665371537208557，两者不相等，判据满足。**loss_aggregations:token 可以声明**（仅限修复后的镜像，即 Miles 0af62f4d 及以后）。
