## ADDED Requirements

### Requirement: Modal 岛必须有持久的检查点存储，不得静默丢弃
launcher SHALL 接受 `--rl-checkpoint-store modal-volume://<卷名>[/前缀]`，在 Modal 岛上挂载该卷并传给 learner。Modal 岛收到不能挂载的写法（如 `s3://`）时 MUST 报错退出，MUST NOT 静默写到容器本地。

#### Scenario: Modal 卷写法
- **WHEN** 以 `--gpu modal:1xh100 --rl-checkpoint-store modal-volume://yeto-ckpt/run1` 启动
- **THEN** Modal 岛配置包含该卷挂载，learner 收到的 store 路径位于挂载点下

#### Scenario: 不支持的写法
- **WHEN** Modal 岛给 `--rl-checkpoint-store s3://b/p`
- **THEN** launcher 在起机器前报错，提示改用 `modal-volume://`

### Requirement: 续训不依赖 --rl-elastic
单岛不同步运行在给了 store 时 SHALL 按 `--rl-cut-every` 保存轮切点，并在存在 `LATEST` 时自动续训；`--rl-elastic` 下的行为不变。

#### Scenario: 普通单岛续训
- **WHEN** 不开 `--rl-elastic`、给 store，跑 3 轮后停，再用同一运行名启动
- **THEN** 第二次启动从第 3 轮开始，发出 `rl_resume` 事件

### Requirement: 恢复校验全部通过才续训
恢复 SHALL 依次校验：store MANIFEST 逐文件 sha256、运行配置指纹、切点片与策略哈希、版本号接上、学习率接上（逐位）、数据题号不重复。任一失败 MUST 报错退出，MUST NOT 从头训练。

#### Scenario: 切点文件损坏
- **WHEN** store 中某片文件被改动
- **THEN** 恢复报哈希不符并退出，不发起训练

#### Scenario: 配置被改
- **WHEN** 续训时 LoRA rank 与切点不同且未给 `--rl-resume-allow-config-change`
- **THEN** 恢复拒绝并列出差异项

### Requirement: 切点原子可见与保留
新切点 SHALL 在所有数据与 MANIFEST 写完并提交后才更新 `LATEST`；旧切点 SHALL 在新 `LATEST` 生效后才按 `--rl-cut-keep` 删除。

#### Scenario: 写一半被杀
- **WHEN** 写切点过程中容器被强杀
- **THEN** 下次启动从上一个 `LATEST` 指向的完整切点恢复

### Requirement: 多岛续训
多岛运行 SHALL 把 syncer 检查点与各岛切点都存入持久存储；岛切点外层版本与 syncer 不符时 SHALL 丢弃岛内切点并走重入追平。

#### Scenario: head 重启
- **WHEN** syncer head 机器换新
- **THEN** 新 head 从 store 取 syncer 检查点 `--resume`，外层版本接上

### Requirement: tape 与 dashboard 认续训
所有事件 SHALL 带启动序号；mismatch tape 与 Modal tape 镜像 MUST NOT 覆盖之前启动的记录；dashboard SHALL 画出续训分界与每段启动开销。

#### Scenario: 续训后的 tape
- **WHEN** 第二次启动重跑了第一次未进切点的第 4 轮
- **THEN** 第一次的第 4 轮记录仍在，`rl_resume.discarded_rounds` 含 4
