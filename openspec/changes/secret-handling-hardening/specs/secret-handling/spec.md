# Spec Delta

## Purpose

岛与 syncer 之间必须有认证；会话契约由 head 的配置给定；学习岛不持有云凭据，并在启动时自查。

## ADDED Requirements

### Requirement: legacy syncer 必须认证岛
legacy 模式的 syncer SHALL 在没有岛 HMAC 密钥时拒绝启动。只有显式传 `--allow-unauthenticated-islands` 或设 `YETO_SYNCER_ALLOW_UNAUTHENTICATED=1` 时，syncer 才 SHALL 无认证运行，并 SHALL 打印告警。

#### Scenario: 无密钥且未声明无认证
- **WHEN** 运维以 legacy 模式启动 syncer，没有给密钥，也没有声明无认证
- **THEN** syncer 报错退出，错误信息说明需要 `YETO_ISLAND_HMAC_KEY`

### Requirement: legacy HELLO 带 HMAC
有密钥时，岛 SHALL 在 HELLO 末尾附加 HMAC-SHA256(密钥, `yeto-hello-mac-v1\0` ‖ HELLO 正文)。syncer SHALL 校验该值；不符或缺失时 syncer SHALL 只拒绝这一条连接，已接纳的岛继续运行。

#### Scenario: 密钥不符
- **WHEN** 一个岛用错误的密钥发 HELLO
- **THEN** syncer 回 MSG_ERROR（含 "HELLO authentication failed"）并关闭该连接，会话不受影响

#### Scenario: 密钥相符
- **WHEN** 岛与 syncer 使用同一把密钥
- **THEN** HELLO 被接纳，后续流程与原来相同

### Requirement: RL launcher 总是发密钥
RL 运行中 launcher SHALL 把同一把岛 HMAC 密钥作为 secret 发给 syncer 与每个岛。启动环境没有密钥且模式为 legacy 时，launcher SHALL 生成一把随机密钥。elastic 模式仍 SHALL 要求启动环境提供密钥。密钥 SHALL NOT 出现在命令行或普通环境变量里。

#### Scenario: legacy 且启动环境无密钥
- **WHEN** 用户以 legacy 模式启动 RL，没有设 `YETO_ISLAND_HMAC_KEY`
- **THEN** launcher 生成 64 位十六进制密钥，syncer 任务与岛任务的 secrets 里是同一个值

### Requirement: 会话契约由 head 配置给定
用户给出 `--rl-island-contract-sha256` 时，launcher SHALL 把它作为 `--expected-island-contract` 传给 syncer。legacy syncer SHALL 只接纳 session_contract_hash 等于 sha256(`yeto-rl-session-contract-v2\0` ‖ 该 HELLO 的布局指纹 ‖ 钉定契约) 的 HELLO。elastic syncer SHALL 在第一个 JOIN 之前就钉定后端身份。未给出时 SHALL 保持原来的"第一个岛决定"。

#### Scenario: 负例岛先连上
- **WHEN** head 钉定了契约 C，负例岛（契约不同）先于好岛连上
- **THEN** 负例岛被拒，好岛随后连上并被接纳

#### Scenario: elastic 续跑时检查点钉的身份与配置不同
- **WHEN** elastic syncer 以 `--resume` 启动，检查点里的后端身份与 head 配置不同
- **THEN** syncer 报错退出

### Requirement: 学习岛不持有 Modal 主令牌
launcher SHALL NOT 把 `MODAL_TOKEN_ID` 或 `MODAL_TOKEN_SECRET` 发给学习岛。需要 Modal Sandbox 的岛 SHALL 使用 `YETO_SANDBOX_MODAL_TOKEN_ID/SECRET`；启动环境缺少它们时 launcher SHALL 起机前报错。

#### Scenario: Modal Sandbox 奖励环境缺沙箱令牌
- **WHEN** 用户选择 Modal Sandbox provider，但启动环境只有主令牌
- **THEN** launcher 起机前报错，说明需要独立的沙箱令牌

### Requirement: 令牌走 secrets
launcher SHALL 把 HF_TOKEN、WANDB_API_KEY、CYBERGYM_API_KEY、TBENCH_REWARD_HMAC_KEY、沙箱令牌、岛 HMAC 密钥放进 sky `secrets=`（Modal 岛放进 `modal.Secret`），SHALL NOT 放进 `envs=`。

#### Scenario: 学习岛任务
- **WHEN** 启动环境有 HF_TOKEN 与 WANDB_API_KEY
- **THEN** 学习岛任务的 `envs` 里没有这两个名字，`secrets` 里有

### Requirement: 岛启动时检查云凭据
每个学习岛入口模块 SHALL 在进入训练前检查本进程环境。发现云凭据环境变量（如 `MODAL_TOKEN_ID`、`AWS_SECRET_ACCESS_KEY`）时 SHALL 报错退出；设 `YETO_ALLOW_ISLAND_CLOUD_CREDENTIALS=1` 时改为告警。发现家目录里的云凭据文件时 SHALL 告警。报错与告警 SHALL 只写变量名或文件名，不写值。

#### Scenario: 岛环境里有 Modal 主令牌
- **WHEN** 岛进程环境里有 `MODAL_TOKEN_SECRET`
- **THEN** 岛进程报错退出，信息里只有变量名
