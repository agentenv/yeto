# Spec Delta

## Purpose

规定本机安全测试集的约定（CI 不在范围内：PM 决定不管 CI，10-09 用户）：会拉起 Ray 的测试默认不跑，缺环境的测试带原因跳过，一条命令跑全安全集且必须全绿，作为"完成"判断的依据。

## ADDED Requirements

### Requirement: 会拉起 Ray 的测试默认不收集
凡是会在本机启动 Ray（gcs_server、raylet 或 `ray.init`）的测试 SHALL 带 `ray_local` 标记。不显式开启时，pytest SHALL 不收集这些测试。显式开启方式 SHALL 只有一种命令行开关，并写进文档。

#### Scenario: 默认运行
- **WHEN** 开发者不带开关运行 pytest
- **THEN** 带 `ray_local` 标记的测试被取消收集，运行结束后本机没有新的 Ray 进程

#### Scenario: 显式开启
- **WHEN** 开发者带上开启开关运行 pytest
- **THEN** `ray_local` 测试被收集并运行

### Requirement: 未标记的 Ray 启动被拦截
默认运行中，若某个未带 `ray_local` 标记的测试尝试 `ray.init`，测试框架 SHALL 让该测试失败，并在失败信息里提示补标记。

#### Scenario: 漏标记
- **WHEN** 一个未标记的测试调用 `ray.init`
- **THEN** 该测试失败，信息写明"该测试会拉起 Ray，请加 ray_local 标记"

### Requirement: 环境依赖的跳过必须写明原因
因缺少可选依赖或外部工具而无法运行的测试 SHALL 用带原因的跳过，原因写明缺什么、在哪种环境下会运行。代码与测试不一致的失败 SHALL 修复，不得用跳过掩盖。

#### Scenario: 缺 cargo
- **WHEN** 本机 PATH 上没有 cargo
- **THEN** 需要现编 syncer 的测试被跳过，原因写明"需要 cargo 构建 syncer"

### Requirement: 本机安全测试集一条命令且全绿
文档 SHALL 给出一条命令运行本机安全测试集。该命令在约定环境中 SHALL 全部通过或带原因跳过，没有失败与错误。以后"完成"的判断 SHALL 以这条命令的结果为准。

#### Scenario: 合入前检查
- **WHEN** 开发者在约定环境中运行文档里的命令
- **THEN** 结果没有失败和错误，跳过项都有原因
