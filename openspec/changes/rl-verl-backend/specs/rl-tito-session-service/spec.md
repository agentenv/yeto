## ADDED Requirements

### Requirement: yeto 会话服务与 Miles 会话服务同接口
yeto SHALL 提供自有会话服务，接口与 Miles 会话服务一致，对外提供 codex 网关现用的聊天接口；接入它时 codex 网关 MUST NOT 需要修改。会话服务对内 SHALL 以词元 ID 调推理端（verl 后端为 vLLM）生成，并保存累积词元、logprob、损失掩码，拼成训练样本交给训练端。

#### Scenario: 网关不改即可接 verl
- **WHEN** codex 网关以现有配置指向 yeto 会话服务，后端为 verl
- **THEN** 多轮轨迹正常完成并产出训练样本

#### Scenario: 两种后端结果一致
- **WHEN** 同一段 codex 轨迹（相同消息与相同推理输出词元）分别经 Miles 后端与 verl 后端的会话服务组装
- **THEN** 两者得到的词元序列与损失掩码逐个相同

### Requirement: 词元原样进入训练样本
推理端生成的词元 ID 与 logprob SHALL 原样进入训练样本；首轮按固定模板渲染，后续轮只对新追加消息增量分词，MUST NOT 对整段文本重新套模板再分词。

#### Scenario: 多轮样本
- **WHEN** 一条多轮轨迹被组装为训练样本
- **THEN** 其中生成段词元与推理端返回的词元逐个相同，掩码只覆盖生成词元

### Requirement: TITO 检查在 yeto 层且两后端共用
Qwen3.8 模板构建器、追加角色白名单、标准模板重渲染比对 SHALL 实现在 yeto 会话服务层，Miles 与 verl 后端共用同一实现。Qwen3.8 家族 MUST NOT 回退到默认构建器。

#### Scenario: 选择构建器
- **WHEN** 模型为 Qwen3.8 家族
- **THEN** 使用 Qwen3.8 构建器，日志可见所选家族

### Requirement: 追加角色白名单
后续轮追加消息的角色 SHALL 限于 tool 与 user；出现其他角色 MUST 拒绝该追加并记录。

#### Scenario: 追加 system 消息
- **WHEN** agent 在后续轮追加 system 消息
- **THEN** 追加被拒并记一条失配记录

### Requirement: 模板重渲染比对
会话服务 SHALL 在轨迹收集后用标准模板重渲染整段消息并与累积词元逐段比较，按类别（special_token_count、special_token_type、non_assistant_text、assistant_text）产出 `tito_session_mismatch` 元数据，经 `rl_harness_mismatch` 事件上报；该检查只观察，不改样本。

#### Scenario: 故意制造失配
- **WHEN** 人为在累积词元中多插一个特殊词元
- **THEN** `rl_harness_mismatch` 事件中 special_token_count 类计数 ≥1
