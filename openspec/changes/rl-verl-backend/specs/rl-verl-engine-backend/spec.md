## ADDED Requirements

### Requirement: verl 作为可选引擎
yeto RL learner SHALL 支持以显式配置选择引擎 `verl`；未选择时 MUST 保持 Miles 路径行为不变。verl 引擎 SHALL 实现推理池、训练组、策略状态、发布器、放置五个端口，MUST NOT 在能力声明中声明弹性、成员发布、可重配放置、切点保存恢复等未实现端口。

#### Scenario: 能力声明只含五端口
- **WHEN** 以引擎 `verl` 启动并读取能力声明
- **THEN** 声明的端口恰为五个必选端口，引擎名为 `verl`

#### Scenario: 默认引擎不变
- **WHEN** 未配置引擎
- **THEN** 使用 Miles 路径，行为与本 change 之前一致

### Requirement: verl 版本钉死
verl 引擎 SHALL 使用 https://github.com/michaellchung/verl 的 commit `acad9875a8bdfc81afbcb0a50d146b2630f44093`；运行时清单 MUST 记录该 commit，与之不符时 MUST 拒绝启动。

#### Scenario: commit 不符
- **WHEN** 镜像内 verl commit 不是钉定值
- **THEN** 启动失败并报出实际 commit

### Requirement: 训练后端可替换
verl 引擎 SHALL 通过单一配置项选择训练后端（第一版支持 FSDP2，预留 Megatron/MindSpeed），切换训练后端 MUST NOT 改变五端口的外部语义；每个训练后端 MUST 提供自己的"原生参数名→yeto 规范名"映射表。

#### Scenario: 未实现的训练后端
- **WHEN** 选择尚未实现的训练后端
- **THEN** 配置期报错，不进入训练

### Requirement: LoRA 配置遵守 yeto 规则
verl 引擎下 LoRA SHALL 使用 `lora_alpha = rank`、dropout 0、bias none；配置与之不符时 MUST 拒绝启动，MUST NOT 沿用 verl 默认 alpha。

#### Scenario: alpha 与 rank 不等
- **WHEN** 配置 rank 32、alpha 16
- **THEN** 启动失败

### Requirement: 采样默认值
verl 引擎 SHALL 默认采样温度 1、top_p=1、top_k=-1。

#### Scenario: 未指定采样参数
- **WHEN** 用户未给出采样参数
- **THEN** 实际下发给推理端的为温度 1、top_p 1、top_k -1

### Requirement: 推理批次元数据
推理池 SHALL 为每组返回 policy_token、reward 均值与方差、token 数；未能提供的可选字段 MUST 报"未报告"而非填 0。

#### Scenario: 元数据非空
- **WHEN** driver 跑完一轮
- **THEN** tape 中该轮各组 reward/token 字段非空

### Requirement: 策略状态导出与应用
策略状态 SHALL 导出按 yeto 规范名排序的可训练张量（LoRA 时仅 adapter），apply 后再 export MUST 与原导出逐位相等。

#### Scenario: 往返逐位一致
- **WHEN** export → apply → export
- **THEN** 两次导出逐位相等

### Requirement: 发布与读回校验
发布器 SHALL 支持两种 LoRA 传送方式并由配置切换，对外为同一发布接口：内存热更新（默认）；落盘到按版本号命名的本地目录再加载，加载成功后删除旧版本目录（退路）。发布器 SHALL 在推送后按规范名计算推理端 adapter 校验和并与发送端比较，结果中写明校验层级；第一版校验层级为"推理端已收到这一版"。无法校验时 MUST 返回显式降级结果 `LORA_UNVERIFIABLE`，MUST NOT 报告成功。

#### Scenario: 落盘方式版本目录
- **WHEN** 配置为落盘方式，连续发布版本 n 与 n+1
- **THEN** 版本 n+1 加载成功后版本 n 的目录被删除，读回校验通过

#### Scenario: 篡改可被发现
- **WHEN** 发送后人为改动推理端一个 adapter 张量
- **THEN** 读回校验失败

### Requirement: 契约哈希包含后端身份
训练契约哈希 SHALL 纳入引擎名、引擎 commit、设备种类、参数名映射表哈希、LoRA 配置哈希、线上 dtype。第一版合并兼容模式只有 `strict`：Miles 岛与 verl 岛（或不同设备的岛）MUST 握手失败。哈希构造 SHALL 集中一处并保留兼容模式字段，供以后混合模式使用。

#### Scenario: 混合岛被拒
- **WHEN** 一个 Miles 岛与一个 verl 岛连接同一 syncer，张量形状相同
- **THEN** 握手因契约哈希不同而失败

### Requirement: 运行时清单
运行时清单 SHALL 记录 verl commit、vllm、torch、transformers、peft 版本；在 NPU 上 SHALL 另记 torch_npu、CANN、vllm_ascend、mindspeed，且以 CANN/HCCL 取代 cuda/nccl 字段。

#### Scenario: GPU 清单
- **WHEN** 在 Modal GPU 上启动 verl 引擎
- **THEN** 清单含上述 GPU 侧全部版本字段
