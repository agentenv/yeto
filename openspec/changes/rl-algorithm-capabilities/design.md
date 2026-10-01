# Design

## Context

动机见 proposal.md 的 Why，调研依据见 `docs/research/rl-algorithms/research.md`。本 design 使用的现状均来自源码，基线为 yeto `18695ae`、Miles `9e4260d`：

- `AlgorithmSpec` v1 共 5 个字段，schema 为 `yeto-rl-algorithm-spec-v1`（`yeto/rl/engine/algorithm.py`）。ports 的构建入口是 `learner.py:1686` 的 `AlgorithmSpec.from_legacy_args(args)`。
- 翻译在 `miles_adapter/config.py::translate_run_config` 中完成。额外参数由 `check_extra_argv` 检查，但只拦截 `ADAPTER_OWNED_FLAGS`（`config.py:71-96, 405-409`）。
- 能力声明写在 `miles_adapter/entry.py:49-50`（`advantage_estimators={"grpo"}`），比对在 `EngineCapabilities.check()`（`capabilities.py`）。
- 零梯度不变量在 `driver.py:355-370`，判定写死为 `any(g.reward_std > 0)`。
- 导出来源记录在 `export.py:386-391`，ports 分支只写入 `rl_engine`。
- 在 Miles 中，grpo/gspo 的 advantage 会丢弃 reward 中的 KL（`loss_hub/advantages.py:53-57`）。只要 `kl_coef != 0`，就会加载参考模型（`miles/ray/specs/train.py:58`）。
- 约束：legacy 的 `build_miles_argv` 必须逐字节不变；默认 GRPO 行为不变；不修改 fork。

## Goals / Non-Goals

**Goals:**
- 建立一个框架，让后续算法 change 只需做三件事：加映射行、在 adapter 声明支持、补验证。
- 默认 GRPO 的哈希和 argv 逐字节不变。

**Non-Goals:**
- 本 change 不开放任何新机制。adapter 的能力声明除执行能力字段外，与 R0 相同。
- 不改 syncer 的 Rust 代码，不改 bridge 的元数据格式，因为 legacy 也在使用。
- 不实现 critic、异步执行和全参数训练。

## Decisions

### D1. 分组的冻结数据类，v1 字段保持平铺兼容

`AlgorithmSpec` 改成多个嵌套的冻结数据类：`AdvantageSpec`、`LossSpec`、`KlSpec`、`CorrectionSpec`、`SamplingSpec`、`ExecutionSpec`，再加上 `entropy_coef` 和 `plugins: tuple[PluginRef, ...]`。v1 的 5 个字段映射到新位置：
- `advantage_estimator` 映射到 `advantage.estimator`；
- `loss` 映射到 `loss.variant`（`policy_loss`）；
- `kl_coef` 映射到 `kl`，解释为 `placement=reward`，见 D5；
- `dynamic_sampling_filter` 和 `dynamic_sampling_max_replacements` 映射到 `sampling`。

`from_dict` 同时接受 v1 和 v2 两种 schema。

备选方案：一个平铺的大数据类。字段会超过 25 个，无法按机制分组校验，也不利于后续 change 各自扩展，因此不采用。

### D2. 规范化：只有"v1 可表达"时才输出 v1 形态

`canonical_json()` 先判断当前描述能否用 v1 表达，条件是所有新字段都等于默认值，且 KL 只有 `placement∈{none,reward}` 加系数这种形式。能表达时，输出与 R0 逐字节相同的 v1 JSON，哈希因此不变；否则输出 `schema: yeto-rl-algorithm-spec-v2` 的完整形态。数值统一为 float，枚举统一为小写字符串，`plugins` 按路径排序。

用 golden 测试锁定 R0 已有的哈希值（`tests/test_rl_engine_algorithm.py` 中已有的期望值）。

备选方案：一律升级为 v2。代价是历史事件中的哈希全部改变，等价性实验的记录无法再对照，因此不采用。

### D3. 映射表：Miles 参数与 spec 字段一一对应，由它驱动吸收和翻译

在 `yeto/rl/engine/miles_adapter/algorithm_flags.py` 中声明一张表。每行包含：Miles 参数、spec 字段路径、类型或解析器、默认值（不输出）、翻译函数。它有三个用途：
1. `translate_run_config` 按表生成 argv，与 v1 默认对应的参数不输出，以保持 argv 快照不变；
2. extra argv 中出现表内参数时吸收进 spec，冲突时报错；
3. 这些参数全部加入 `ADAPTER_OWNED_FLAGS`，之后只能经由吸收路径进入。

另有一份"影响训练目标的 Miles 参数"清单，是映射表的超集，按 `arguments.py` 中 loss、advantage、rollout 相关的参数组整理。清单里有、映射表里没有的参数，在 extra argv 中出现时直接拒绝。清单用测试锁定：在 miles-next-venv 中读取 upstream parser，逐一核对清单里的参数确实存在。

本 change 映射的参数：`--advantage-estimator`、`--eps-clip`、`--eps-clip-high`、`--eps-clip-c`、`--calculate-per-token-loss`、`--disable-grpo-std-normalization`、`--disable-rewards-normalization`、`--normalize-advantages`、`--kl-coef`、`--use-kl-loss`、`--kl-loss-coef`、`--kl-loss-type`、`--use-unbiased-kl`、`--entropy-coef`、`--use-tis`、`--tis-clip`、`--tis-clip-low`、`--custom-tis-function-path`、`--use-rollout-logprobs`、`--get-mismatch-metrics`、`--use-opsm`、`--opsm-delta`、`--custom-pg-loss-reducer-function-path`、`--custom-reward-post-process-path`、`--loss-type`、`--custom-loss-function-path`、`--dynamic-sampling-filter-path`、`--over-sampling-batch-size`。

能被映射，不代表 adapter 声明支持。是否支持由 D4 决定。

### D4. 能力按机制声明，执行能力单独成组

`EngineCapabilities` 新增以下字段：`losses`、`loss_aggregations`、`kl_placements`、`corrections`、`reward_postprocessors`，以及 `execution: {critic: bool, max_policy_staleness: int, rollout_logprobs: bool}`。它们作为顶层新键写入 attestation，#66 的 `attestation_from_dict` 会忽略未知键。`check()` 按 spec 计算出"所需机制集合"，与声明逐项比对，把所有问题汇总后一次报出。

本 change 中 Miles adapter 的声明：
- 机制维度与 R0 相同，即 grpo、`policy_loss`、默认聚合、KL 为 `none`/`reward`、无修正、bounded 过滤；
- `execution` 为 `critic=False`、`max_policy_staleness=0`、`rollout_logprobs=True`（依据 `sglang_rollout.py:187` 的 `return_logprob=True`）。

与 `rl-infra-spec` 的对齐（alignment.md A1）：`execution.max_policy_staleness` 表示该运行所选执行模式可能产生的最大策略年龄，由 `rl-infra-spec` 的 `ExecutionProfile.max_policy_age` 提供；本 change 与 `rl-infra-spec` 的所有执行模式（含 partitioned-overlap）均为 0；大于 0 只能来自另立 change 认证的算法契约。`ExecutionProfile` 以本 change 的 `AlgorithmSpec` 规范化哈希作为算法契约身份，不另设算法 schema。

fake engine 同步声明，供 CPU 测试使用。

### D5. KL 放置与估计方式的规则

规则由 yeto 自己校验，不依赖 Miles 报错：
- `placement=reward` 且 `coef>0`，配 grpo/gspo：拒绝（依据 Miles 源码）。
- `placement=loss`：翻译为 `--use-kl-loss --kl-loss-coef c --kl-loss-type t`。
- `reward` 与 `loss` 不能同时出现，因为结构上只有一个 `placement`。
- v1 描述 `kl_coef=None`：对应 `placement=none`。
- v1 描述 `kl_coef=0.0`：保持 R0 行为，照样输出 `--kl-coef 0.0`，哈希不变。这种情况下 Miles 不会加载参考模型。
- v1 描述 `kl_coef>0`：解释为 `placement=reward`，按第一条规则在 grpo 下被拒。

ports 用户能观察到这一变化，proposal 的 Impact 已经写明。

### D6. 零梯度判定由 spec 推导

在 spec 上加一个派生方法 `expects_gradient(batch_summary, step_metrics) -> bool`：
- 默认 GRPO：`any(reward_std > 0)`，与 R0 相同；
- 声明了屏蔽行为的机制：结合 `step_metrics` 中的屏蔽比例判定。本 change 先把接口和默认实现写好，各机制的判定由对应 change 补充。

（追加，2026-09-29）各机制的判定规则也可以**收紧**，即在 R0 判定式认为“不期望梯度”时要求有梯度（例如 GDPO、REINFORCE++）。规则只在 spec 需要该机制时生效，且收紧优先于放宽，默认 GRPO 的判定不变；不允许借此放宽 R0 判定。

`TrainStepMetrics` 增加一个可选字段 `masked_fraction`，adapter 从 Miles 训练日志中的对应键读取；读不到时记为 None，判定按"期望有梯度"处理，即保守地沿用 R0 的行为。

### D7. 插件身份

`PluginRef(path, sha256)` 中，sha256 取自插件所在模块文件的字节。启动时 import 该模块并重新计算，不一致就拒绝。插件只允许放在 `yeto.` 命名空间或 Miles 内置命名空间下；Miles 内置函数的 sha256 取自 Miles 源文件，这样 Miles commit 的变化会反映到算法哈希上。

### D8. 用户输入：新增 `--rl-algorithm-spec`

learner 和 launcher 新增 `--rl-algorithm-spec PATH`，指向一个 JSON 文件，内容是 v1 或 v2 的 dict，只在 ports 路径生效；与 legacy 组合时启动前拒绝，不改变 legacy。优先级如下：
1. `--rl-algorithm-spec` 给出基础描述；
2. 如果没有给，就用 `from_legacy_args(args)` 从现有 CLI 构建，与 R0 相同；
3. 最后吸收 extra argv 中的参数（D3）。

### D9. 岛间一致：由 launcher 下发预期哈希

launcher 在 ports 模式下先构建一次 spec，算出 `expected_algorithm_sha256`，再通过 `--rl-expected-algorithm-sha256` 传给每个 learner。learner 构建出本岛 spec 后，在连接 bridge 之前核对；不一致时写入 `rl_algorithm_mismatch` 事件并退出。直接启动 learner、没有这个参数时，会写一条警告事件，不拒绝，以兼容手工启动单岛的调试场景。

`export.py` 的 ports 分支写入 `algorithm_spec`（规范化 JSON）和 `algorithm_spec_sha256`。

备选方案：
- 放进 bridge 元数据，或让 syncer 校验。bridge 元数据的 schema 被 legacy 共用，syncer 又要改 Rust。可以作为后续的加固项。
- 只记录在事件中。不能阻止错配的平均，因此不采用。

### D10. R0 归档后的合并清单

R0 归档后，要把本 change 中以下要求改写为 R0 capability 的 delta：
- `rl-engine-ports` 的"能力声明与握手"：加入机制维度和执行能力；
- `rl-engine-ports` 的"算法描述复用引擎算法"：从只支持 GRPO 改为支持声明的算法，并加入来源记录；
- `rl-engine-ports` 的"串行共置执行循环"：零梯度判定改为按算法判定；
- `rl-engine-selection` 的"不支持的组合明确拒绝"：GRPO 改为按能力声明判定。

### D11. 未验证机制放行

后续算法 change 都要求"GPU 冒烟（G1）通过后才在 adapter 里声明支持"，但未声明的机制在启动时就会被拒绝，冒烟无法运行。本 change 统一提供 `--rl-allow-unverified-mechanism NAME`（learner 与 launcher 都支持，可重复），各算法 change 复用这一个开关，不再各自发明：
- `check()` 接收放行集合，只豁免集合内机制的"未声明"问题；
- 放行集合非空、且处于多岛或外层同步模式时，启动前拒绝；
- 事件记录 `rl/unverified_mechanisms`，来源记录写同名字段；放行不进入算法哈希，因为它描述的是验证状态，不是算法本身；
- G1 通过后，在 adapter 中正式声明支持；G3（两岛）只使用正式声明，不使用放行开关。

备选方案：每个算法 change 在代码里加内部开关。这样会分散，而且外部看不到，不采用。

## Risks / Trade-offs

- [吸收 extra argv 会改变现有 ports 配置的哈希] → ports 目前不是默认路径；在 `docs/MILES_RL.md` 中写明，并在事件里记录哪些参数是吸收来的。
- ["影响训练目标的参数"清单不完整，有参数漏掉] → 清单用测试与 upstream parser 核对；upstream 升级时要求重新审查清单（写入 `docs/MILES_RL.md` 的升级步骤）。
- [`masked_fraction` 在 Miles 日志中的键名不稳定] → 读不到时退回到 R0 的判定，也就是更严格的一侧，不会漏报。
- [插件哈希取模块文件字节，模块内无关的改动也会让哈希变化] → 这是有意为之：宁可误报身份变化，也不漏报。
- [launcher 下发哈希挡不住手工拉起的错配岛] → 已写警告事件；syncer 侧校验作为后续加固项。

- [放行开关被误用于正式实验] → 只允许单岛运行；来源记录和导出产物都标记"含未验证机制"。

## Migration Plan

1. 本 change 只影响 ports 路径，legacy 不变。合入后，现有的 ports 用法（不带新参数）行为和哈希都不变。
2. 回滚：revert 本 change 即可，不涉及数据格式迁移。来源记录中新增的键只是附加内容。
3. R0 合入 main 后 rebase，并按 D10 做 spec 合并。

## Open Questions

- syncer 是否应该在协议层校验算法哈希。可以作为后续加固，不影响本 change 的 spec 和任务拆分。
