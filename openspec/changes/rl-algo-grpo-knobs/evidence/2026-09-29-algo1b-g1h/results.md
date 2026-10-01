# algo1b-g1h 结果（审查后改写；plan.md 中的判据段未改）

## 1. 预登记判据：字面满足，但区分力弱
- (a) over_sampling 臂第 2–6 轮的 submitted_groups 都是 8；(b) os_off 臂依次为 8/4/8/8/8，其中第 3 轮恰为 4。两条都字面满足。
- 但区分力弱：over_sampling 臂每轮都是 8，也可以解释为"每一轮都恰好触发了一次过滤补采"（每批 4 组，补采一批也会得到 8）。从 os_off 臂的补采频率（5 轮中有 4 轮）估算，这种解释的概率约为 0.33。所以单靠 (a)(b) 不足以排除它。

## 2. 决定性证据（事后推断，只有一轮）
- over_sampling 臂 rollout 1：submitted_groups=8，aborted_in_flight_groups=4。按 `rollout_meta_hook.py:385-390` 的公式 `aborted = submitted − (completed + filtered)`，可推出 completed + filtered = 4；而本轮训练了 4 组（completed=4），所以 filtered = 0。
- 如果提交批大小是 4（不开超采样），第二批提交只会在有组被过滤掉后才发生；filtered=0 却提交了 8 组，在批大小 4 下不可能出现。因此这一轮必然是一次性提交了 8 组，也就是 over_sampling_batch_size=8 生效了。
- 限定条件：
  - 这是**事后推断**，不是预登记判据；
  - **只有 rollout 1 这一轮**；
  - 依赖 hook 的公式正确；
  - 依赖"被中止的组不进入 all_samples"这一前提（INFRA 已核实 Miles 行为）。
- 另外：两臂参数表中只有 over_sampling_batch_size 不同（4 对 8）。
- 更正：先前"over_sampling 每一轮都一次提交 8 组"的说法只对 rollout 1 成立，已删除。

## 3. 版本与证据来源更正
- 代码包含的是 `submitted_groups` 这项改动在集成分支上的挑拣提交 **0f69c09**，不是 infra-a 上的 cbb5d98，两者提交说明相同。
- 证据目录里没有 run_manifest.json：本次用的是 sandbox harness，没有经过 launcher。代码与镜像版本以 `harness/YETO_SHA`、`sbx.py`（miles 0af62f4d、sglang 9f29303、基础镜像 digest）和 `out/setup.log`（打印了检出的提交与 GPU 型号）为准。

## 4. 待办
- 如果需要强证据：让 Miles 在每次提交时记录批大小（或 yeto 在 generate 调用处记录），然后另立计划补跑一次。
