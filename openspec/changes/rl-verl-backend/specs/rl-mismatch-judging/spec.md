## ADDED Requirements

### Requirement: 判据指标口径与后端无关
训推不一致判据使用的指标（逐词元 |Δlogprob| 先样本内再跨样本平均、k3 KL、tis_clipfrac、带正负号的平均差）SHALL 按同一口径计算，不随引擎改变含义；这些指标 SHALL 统一在 yeto 侧按 Miles 口径重算；后端原生指标 SHALL 只存档作参考，MUST NOT 参与判据。

#### Scenario: 同口径落 tape
- **WHEN** verl 引擎完成一轮训练
- **THEN** tape 的 mismatch 字段含统一口径的 abs_diff、k3、tis_clipfrac、带符号平均差，且另存 verl 原生 `rollout_corr/*` 作参考

### Requirement: 带正负号的平均 logprob 差
判据 SHALL 计算并上报"训练 logprob − 推理 logprob"的带符号平均值，并可对其绝对值设阈值。

#### Scenario: 识别 top_p 偏移
- **WHEN** 以 top_p=0.9、lr 0 运行（S16 实测带符号均值约 −0.032，k3 与 tis_clipfrac 未见异常）
- **THEN** 带符号平均差判据报警

### Requirement: 旁路模式与判据互斥
系统 MUST 在配置期拒绝同时开启"旁路模式（old logprob 直接取推理 logprob）"与任何训推不一致判据。

#### Scenario: 同时开启
- **WHEN** 配置旁路模式且启用判据
- **THEN** 启动失败并说明原因

### Requirement: 确定性选项只用于诊断
`full_determinism` SHALL 只在诊断/复现模式下允许，且必须同时开启 eager 执行；在训练模式下 MUST 拒绝。

#### Scenario: 训练模式开确定性
- **WHEN** 正常训练配置中开启 full_determinism
- **THEN** 启动失败（理由：组内采样完全相同，优势为 0）

#### Scenario: 诊断复现
- **WHEN** 诊断模式下开启 full_determinism 与 eager，同配置跑两次
- **THEN** 两次回答集合与两侧 logprob 逐位一致

### Requirement: 阈值按后端+硬件+版本标定
判据阈值 SHALL 以（引擎、训练后端、推理引擎及版本、硬件型号）为键记录，MUST NOT 跨键复用；缺少对应键的阈值时 MUST 明确报"未标定"而非套用其他后端的数值。具体数值：待定。

#### Scenario: 无标定值
- **WHEN** 在尚无标定的 NPU 组合上运行判据
- **THEN** 判据输出"未标定"，不给通过结论

### Requirement: 修正只开放等价子集
verl 后端下训推修正 SHALL 只开放与 Miles 公式等价的 TIS（下界 0）与 IcePop；选择其他修正时 MUST 启动即报错"verl 后端不支持此修正"。

#### Scenario: 请求不等价修正
- **WHEN** verl 引擎下配置 TIS 下界 0.5
- **THEN** 启动即报错"verl 后端不支持此修正"
