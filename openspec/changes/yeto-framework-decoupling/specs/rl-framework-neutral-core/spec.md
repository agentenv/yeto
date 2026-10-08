## Purpose

规定 yeto RL 引擎核心与训练框架、云、硬件之间的边界：核心只认中立端口与中立数据，去耦合过程不改变 Miles 现有可观测行为与契约哈希。

## ADDED Requirements

### Requirement: 标准样本回归
仓库 SHALL 保存一组标准样本：若干典型配置下的 Miles 命令行输出、`AlgorithmSpec.sha256()`、`ExecutionProfile.contract_hash`、strict（schema 3）与 decoupled（schema 4）进度文件、tape 事件片段，全部可在 CPU 上生成。去耦合的每个阶段合并前，重新生成的结果 MUST 与标准样本逐字节一致。

#### Scenario: 阶段合并前比对
- **WHEN** 任一去耦合阶段的改动在 CPU 上重新生成标准样本
- **THEN** 命令行、两种契约哈希、进度文件、tape 片段与已存样本逐字节一致，否则测试失败并指出差异项

#### Scenario: 不需要 GPU 与 Ray
- **WHEN** 运行标准样本生成与比对测试
- **THEN** 不访问 GPU、不连云、不拉起本机 Ray

### Requirement: 静态边界检查
仓库 SHALL 提供只做语法树分析（不 import 被检模块）的边界检查测试：引擎核心（`yeto/rl/engine/` 除适配层外）、`yeto/rl/rewards/`、`yeto/rl/harness/`（除适配层胶水）、`yeto/rl/algos/` MUST NOT import 训练框架或推理框架（miles、megatron、sglang、verl、vllm、torch_npu）、Miles 适配层与旧版引擎模块，MUST NOT import 云库（sky、modal、`yeto.modal_runner`、`yeto.shape.providers`）；各适配层之间 MUST NOT 互相 import；云层 MUST NOT import 后端适配层。现有违规 SHALL 列入白名单，白名单 MUST 只减不增。

#### Scenario: 新增违规
- **WHEN** 有人在引擎核心新增一条 `import miles...`，且该文件:模块对不在白名单中
- **THEN** 边界检查失败并打印文件、行号、被禁模块

#### Scenario: 白名单只减不增
- **WHEN** 白名单条目数比仓库记录的上限多
- **THEN** 边界检查失败

#### Scenario: 修掉违规后白名单未删
- **WHEN** 白名单中某条在代码里已不存在
- **THEN** 边界检查失败，提示删除该条

### Requirement: 核心不依赖具体后端实现
引擎核心 SHALL 只通过端口与中立类型使用训练后端：需要恢复的异常、切点上下文、重分片计划 MUST 定义在核心；重分片可行性 MUST 经训练组的可选方法或能力声明获得；事件写入 MUST 经核心的事件写入器（后端注入实现）；运行时清单的能力探针、版本模块表、补丁记录 MUST 由适配层提供。事件字段名、进度文件格式 MUST 保持不变。

#### Scenario: 无 Miles 也能导入核心
- **WHEN** 在没有安装 Miles 的环境中导入引擎核心模块并用假引擎跑驱动器
- **THEN** 导入成功，驱动器测试通过

#### Scenario: 事件字段不变
- **WHEN** Miles 路径经新的事件写入器写 tape
- **THEN** 事件名与字段与标准样本一致

### Requirement: 权重传输方式用中立名
放置描述中的权重传输方式 SHALL 使用中立名（同设备进程间共享、集合通信广播、落盘），具体实现名（如 cuda-ipc、nccl-broadcast）由后端与设备族翻译；凡已写入 tape 或哈希的旧名 MUST 在 Miles 下保持原值。

#### Scenario: Miles 下 tape 不变
- **WHEN** Miles colocated 放置写出传输方式字段
- **THEN** tape 中的值与标准样本一致

### Requirement: 后端身份进契约
系统 SHALL 计算后端身份哈希（后端名、后端 commit、设备族、参数名映射表哈希），与 `AlgorithmSpec.sha256()`、`ExecutionProfile.contract_hash` 并列，MUST NOT 改变这两个旧哈希的输入。不同后端身份的岛请求合并时 MUST 拒绝。

#### Scenario: 旧哈希不变
- **WHEN** 加入后端身份后对标准样本配置重算两种旧哈希
- **THEN** 与标准样本逐字节一致

#### Scenario: 混合后端被拒
- **WHEN** 一个 Miles 岛与一个 verl 岛（或后端相同但设备族不同的岛）尝试合并
- **THEN** 握手阶段拒绝并报出双方后端身份
