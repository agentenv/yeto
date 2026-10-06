# Spec Delta

## Purpose

为多岛 DiLoCo RL 运行提供以 syncer 为中心、对比各 learner 的只读 dashboard：同一份磁带归约结果既可经 SSH 隧道实时查看，也可导出为离线静态页面，覆盖告警、训练曲线、round 表、岛健康、成本效率与岛明细下钻。

## ADDED Requirements

### Requirement: 单一归约口径
实时服务与静态导出 SHALL 使用同一归约逻辑从 learner 磁带、syncer 磁带与 `fleet.jsonl` 生成视图；对同一组磁带，两者给出的数据 MUST 一致。未知事件类型与字段 MUST NOT 导致归约失败；缺失数据 MUST 显示为“无数据”，不得显示为 0。

#### Scenario: 实时与离线一致
- **WHEN** 对已结束运行的同一组磁带分别调用 `/api/overview` 与静态导出
- **THEN** 两者内嵌的 overview 数据相同

#### Scenario: 旧运行磁带
- **WHEN** 归约一份不含新增字段的旧磁带
- **THEN** 页面正常渲染，缺失指标标为无数据

### Requirement: 只读本机服务
实时服务 SHALL 只监听 127.0.0.1，只接受 GET，提供 `/api/overview`、`/api/islands/<id>`、`/api/rounds`、`/api/fleet`、`/api/events` 与页面本身。要求监听非回环地址时 MUST 拒绝启动。页面 MUST NOT 提供任何改变运行状态的操作，只可提供“复制 CLI 命令”。

#### Scenario: 非回环地址
- **WHEN** 以 `--host 0.0.0.0` 启动服务
- **THEN** 启动失败并说明只允许回环地址，建议使用 SSH 隧道

#### Scenario: 写请求
- **WHEN** 客户端向任一端点发送 POST
- **THEN** 返回 405，运行状态不变

### Requirement: 离线静态导出
静态导出 SHALL 生成单个自包含 HTML 文件，打开时不发起任何网络请求，并标明为离线导出及生成时间。

#### Scenario: 断网打开
- **WHEN** 在无网络环境打开导出文件
- **THEN** 告警、曲线、round 表、岛健康与成本面板均完整显示

### Requirement: syncer 总览与岛明细下钻
主视图 SHALL 以 syncer 视角对比所有 learner：按岛叠加的训练曲线（可切换 reward mean/p10/p90、pg_loss、KL、entropy、grad_norm、clip 总/低侧/高侧、response len、tok/s），曲线上标注外层同步 round 边界、有 missed 的 round 与告警点。岛明细 MUST 为同页下钻面板（进度/policy、cell 表、E1 事务、最近事件、Ray 嵌入），不是独立页面。

#### Scenario: 点击岛健康卡
- **WHEN** 用户点击某岛健康卡
- **THEN** 该岛曲线高亮、其余变淡，并在同页展开该岛明细

### Requirement: syncer 派生 round 表
round 表 SHALL 由 syncer merge 记录与各岛 fragment_push、policy_apply、pull_resend、member_publication 事件关联得出，每行含 round、fragment、responded/expected、missed 岛、quorum/sync/merge 时延与重发数，并可筛选仅异常 round。该表 MUST 标注为派生数据。

#### Scenario: 某岛未推送
- **WHEN** 某 round 的 merge 记录之前只有 3 个岛推送了 fragment，而 roster 为 4
- **THEN** 该行 responded/expected 为 3/4，missed 列出未推送的岛，并被“仅异常”筛选保留

### Requirement: 告警与联动定位
页面 SHALL 按严重度（严重/警告/提示）列出告警，至少覆盖：心跳超时、连续 missed、quorum 时延异常、grad_norm 尖峰或 NaN、clipfrac/KL 超阈值、预算逼近。每条告警 MUST 可点击定位到对应岛、round 与指标；阈值 MUST 可配置。

#### Scenario: 心跳超时
- **WHEN** 某岛最后一条事件距今超过 300 秒
- **THEN** 出现该岛“严重”级心跳告警，岛健康卡标为掉线疑似

#### Scenario: 点击 grad_norm 告警
- **WHEN** 用户点击某岛 grad_norm 尖峰告警
- **THEN** 曲线切换到 grad_norm、定位到对应 round，round 表高亮该行，并展开该岛明细

### Requirement: 成本估算与预算告警
成本面板 SHALL 以配置的价目表 × GPU 数 × 墙钟估算各岛与合计金额、当前每小时速率、预计触达预算上限时间，以及 `$/1M tok` 与 `reward/$`。所有金额 MUST 标注为估算非账单；价目表缺失的岛 MUST 显示“未定价”而非 0。累计达预算 60%/80% 时 MUST 产生警告/严重告警。

#### Scenario: 逼近预算
- **WHEN** 累计估算达到预算上限的 82%
- **THEN** 头部成本条与告警均为严重级，并显示预计触达时间

### Requirement: Ray dashboard 嵌入与降级
岛明细 SHALL 按岛类型提供 Ray dashboard：可经 SSH 隧道或本机直连访问的岛显示隧道命令并在用户点击后以 iframe 嵌入；无稳定入站端口的岛（Modal）MUST 显示“不可嵌入”并退化为资源采样与事件面板。

#### Scenario: Modal 岛
- **WHEN** 用户展开 Modal 岛明细
- **THEN** 不出现 iframe，显示 GPU 利用、显存与 tok/s 的最新值

### Requirement: ssh_harness 运行的实时来源
ssh_harness 启动的运行 SHALL 可被实时服务跟随：head 侧持续获取远端 learner 磁带增量；来源断开时 MUST 自动重连，并在事件视图中记录来源丢失。

#### Scenario: SSH 断线
- **WHEN** 与某远端 learner 的 SSH 连接中断 20 秒后恢复
- **THEN** 断线期间该岛显示来源丢失，恢复后补齐中断期间的记录且不重复
