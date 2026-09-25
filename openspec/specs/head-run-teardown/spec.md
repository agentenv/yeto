# head-run-teardown Specification

## Purpose

保证一个由 head 控制器托管的 run 在 `yeto down` 之后不会在云上留下任何计费中的机器：每个 learner 从唯一能看见它的地方拆除并逐个确认，head 只有在 learner 全部确认后才能删除，回收结束前向云核对，任何不确定都以显式失败而不是假成功结束。

## Requirements

### Requirement: learner 必须从能看见它的 sky 拆除并逐个确认

对 head 模式的 run，`yeto down` SHALL 先在 head 上、用 head 的 sky 拆除每个 learner cluster，并为每个 learner 取得一条明确的"已拆除"或"本就不存在"的确认。本机 sky 对 learner 返回"不存在" MUST NOT 被视为该 learner 已拆除。

#### Scenario: 岛由 head 开出，本机 sky 不认识它
- **WHEN** run 的元数据记录 `controller == "head"`，且某个 learner cluster 只存在于 head 的 sky 记录中
- **THEN** `yeto down` 通过 head 执行该 learner 的拆除，并在输出中记录该 learner 的确认结果
- **AND** 本机 sky 返回的 "does not exist" 不产生任何"已拆除"的判定

#### Scenario: 控制器 job 先于拆除被取消
- **WHEN** `yeto down` 开始拆除 head 模式 run 的 learner
- **THEN** 它先取消 head 上仍在运行的控制器 job
- **AND** 拆除过程中不会有新的 learner 被重新拉起

### Requirement: head 只有在所有 learner 确认后才可删除

`yeto down` MUST NOT 删除 head，除非该 run 的每个 learner 都已取得拆除确认。存在未确认 learner 时，它 SHALL 保留 head、以非零退出码结束，并列出未确认的 learner 名字与下一步（重跑 `yeto down` 或到云控制台删除）。

#### Scenario: 某个 learner 无法确认拆除
- **WHEN** 经 head 拆除后，重试用尽仍有 learner 没有确认行
- **THEN** head 不被删除
- **AND** 命令以非零退出码结束，错误信息列出未确认的 learner

#### Scenario: head 暂时无法响应
- **WHEN** 经 head 的拆除请求因 head 自身正在收尾（如其 sky 返回服务端错误）而失败
- **THEN** `yeto down` 在有限次数内重试
- **AND** 重试仍失败时按上一场景处理，而不是转而删除 head

### Requirement: 回收结束前必须向云核对无残留

在宣布 run 已回收之前，`yeto down` SHALL 对每个能被云端查询的 cluster 向云（而不是 sky 的状态库）核对没有非终止状态的实例；发现残留 SHALL 重试拆除，重试后仍有残留 SHALL 以非零退出码结束并打印残留实例的标识。Modal learner SHALL 按其 app 已停止且无运行中任务来核对。

#### Scenario: 云上仍有实例存活
- **WHEN** 拆除后云端查询仍返回该 cluster 的存活实例
- **THEN** `yeto down` 重试拆除，重试后仍存活则非零退出并打印实例标识
- **AND** 不打印 run 已回收的成功信息

#### Scenario: 云端无法核验
- **WHEN** 某个 cluster 所在的云不支持云端实例查询，或查询本身出错
- **THEN** `yeto down` 明确打印该 cluster 未经云端核验、按 sky 的结果信任
- **AND** 这一情况不导致命令失败

### Requirement: 成功信息只在完全回收后出现

只有当所有 learner 已确认、head 已删除且云端核对无残留时，`yeto down` SHALL 打印 run 已回收并以零退出。任何一步未确认都 MUST NOT 产生零退出码。

#### Scenario: 全部回收成功
- **WHEN** 所有 learner 确认拆除、head 删除成功、云端核对为空
- **THEN** 命令打印 run 已回收并以零退出

#### Scenario: 本地模式的 run 不受影响
- **WHEN** run 的元数据记录 `controller == "local"`
- **THEN** learner 由本机 sky 直接拆除，仍执行云端核对与成功信息的判定
