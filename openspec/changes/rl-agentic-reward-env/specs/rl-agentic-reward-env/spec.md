# Spec Delta

## Purpose
agentic RL 的奖励环境：benchmark 中立的适配器接口、预装依赖的沙箱镜像、判分接口与留出评测集。不改变现有 TB2 provider 与签名结果的行为。

## ADDED Requirements

### Requirement: benchmark 适配器与注册表
系统 SHALL 提供与具体 benchmark 无关的适配器协议（任务列表、任务规格、预装计划、判分命令、判分解析）与按名字取适配器的注册表；内置适配器 MUST 在首次按名字获取时自动注册，未知名字 MUST 报错并列出已注册名字，同名重复注册 MUST 报错。

#### Scenario: 取 TB2 适配器
- **WHEN** `get_adapter("tb2", tasks_dir=…)`
- **THEN** 返回 TB2 适配器，`registered()` 含 `tb2`

#### Scenario: 未知 benchmark
- **WHEN** `get_adapter("no_such_bench")`
- **THEN** 抛出错误，信息含已注册名字

### Requirement: 预装依赖不改变判分
预装计划 MUST 只包含判分脚本在测试命令之前的依赖安装步骤与工具环境预热；识别不了的步骤 MUST 不预装并在计划中列出；判分时 MUST 执行未修改的官方判分脚本；预装镜像身份 MUST 为（schema、官方镜像、命令）的内容哈希。

#### Scenario: 题目专属步骤留到判分时
- **WHEN** `test.sh` 在测试前拷贝 `/tests` 下文件或下载数据
- **THEN** 这些行不进预装计划，计划 `complete=false` 并列出它们

#### Scenario: 关闭预装
- **WHEN** `YETO_REWARD_ENV_PREBAKE=0`
- **THEN** 沙箱镜像为官方镜像，标签 `yeto-prebake=none`

### Requirement: 不影响生产判分应用
构建工具 MUST 拒绝以 `yeto-tbench2` 为目标；新 provider 的运行时沙箱默认 Modal app MUST 为 `yeto-reward-env`。

#### Scenario: 构建目标为生产应用
- **WHEN** `--build --app yeto-tbench2`
- **THEN** 拒绝构建

### Requirement: 留出评测集
系统 SHALL 按 benchmark 官方难度分桶、按 seed 可复现地生成留出名单（`yeto-eval-holdout/1` 格式，与输入顺序无关），某桶题数不足 MUST 报错，名单内重复题号 MUST 报错，并 MUST 在训练数据含留出题时报错。每题元数据 MUST 含 `task_id`、`benchmark`、`benchmark_version`、`difficulty`（官方原文）、`difficulty_source`，有桶时含 `eval_bucket`。

#### Scenario: 留出题泄漏进训练数据
- **WHEN** 训练数据题号与留出集有交集
- **THEN** `assert_disjoint` 报错并列出交集

### Requirement: SWE-bench Verified 判分与官方一致
SWE-bench Verified 判分 MUST 在该题官方镜像的全新沙箱中，按官方补丁应用顺序打上答案补丁、运行数据集 `eval_script`，并由官方 `get_eval_report` 评分：FAIL_TO_PASS 与 PASS_TO_PASS 全部通过才记为解决。补丁打不上或测试超时 MUST 记为未解决；判分脚本未运行到测试 MUST 记为基础设施错误而不是未解决。

#### Scenario: 阳性对照
- **WHEN** 以数据集 `patch` 作为答案
- **THEN** 期望记为解决（实测不符的题进排除表）

#### Scenario: 阴性对照
- **WHEN** 不打补丁（`submission=None`）
- **THEN** 期望记为未解决

#### Scenario: PASS_TO_PASS 被破坏
- **WHEN** FAIL_TO_PASS 全过但任一 PASS_TO_PASS 失败
- **THEN** 记为未解决

### Requirement: 沙箱网络出口默认关闭
奖励环境的任务沙箱 SHALL 默认没有网络出口；只有网络规则里列出的任务才 SHALL 按其规则放行（全部放行、域名白名单或 CIDR 白名单）。规则文件格式错误时 SHALL 报错，不得退回放行。

#### Scenario: 未列出的任务
- **WHEN** 为网络规则里没有的任务建 Modal 沙箱
- **THEN** 沙箱以 `block_network=True` 创建

#### Scenario: 域名白名单
- **WHEN** 任务规则为 `{"domains": ["pypi.org"]}`
- **THEN** 沙箱只放行 `pypi.org`（`outbound_domain_allowlist`）

### Requirement: 本地沙箱只传最少环境变量
`LocalProcessSandbox` SHALL 不继承父进程环境；命令只拿到 `PATH`、`LANG`、`LC_ALL`、`TZ`（父进程有时）以及沙箱自己的 `HOME`、`TB2_TESTS_DIR`、`TB2_VERIFIER_LOGS_DIR`。

#### Scenario: 父进程带令牌
- **WHEN** 父进程环境里有 `MODAL_TOKEN_SECRET`
- **THEN** 沙箱命令的环境里没有该变量
