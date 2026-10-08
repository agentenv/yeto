## Purpose

规定每个训练框架一块适配层的位置、选择方式，以及中立配置与算法字段如何翻译到各后端、不支持时如何拒绝。

## ADDED Requirements

### Requirement: 适配层目录约定
每个训练后端的全部框架专用代码 SHALL 位于 `yeto/rl/adapters/<后端名>/`：Miles 为 `yeto/rl/adapters/miles/`（含原 `engine/miles_adapter/`、旧版引擎、Miles 专用模型文件、harness 中 Miles 胶水、Miles 版本常量与补丁），verl 为 `yeto/rl/adapters/verl/`。原 `yeto.rl.engine.miles_adapter.*` 等旧 import 路径 MUST 在过渡期继续可用。

#### Scenario: 旧路径可用
- **WHEN** 现有代码或测试以旧路径 import Miles 适配层模块
- **THEN** 导入成功且得到同一对象

### Requirement: 后端注册表
系统 SHALL 提供按名字取后端的注册表，返回能力声明、放置请求、岛入口、命令行或配置生成、运行时探针；启动器与命令行 MUST 只经注册表使用后端。命令行 SHALL 提供中立选项 `--rl-backend {miles,verl}`，默认 `miles`；`--rl-engine {legacy,ports}` 仅作为 Miles 内部选择。

#### Scenario: 默认后端不变
- **WHEN** 未指定 `--rl-backend`
- **THEN** 选 Miles，生成的命令行与标准样本一致

#### Scenario: 未知后端
- **WHEN** 指定未注册的后端名
- **THEN** 启动前报错并列出已注册后端

### Requirement: 中立配置与算法字段
运行配置与算法规格 SHALL 使用中立字段名（如张量并行度、非零奖励方差过滤）；算法扩展只注册中立字段、校验与默认值；"中立字段→后端参数"映射表 MUST 位于各后端适配层。某后端映射表中缺少的字段被启用时，MUST 在启动前报"该后端不支持此字段"。

#### Scenario: 后端不支持的字段
- **WHEN** 在某后端上启用该后端映射表中没有的算法字段
- **THEN** 启动前拒绝并指出字段名与后端名

### Requirement: 中立名与新哈希对照
过滤器与插件改用中立名后，系统 SHALL 把旧名（含 `miles.` 路径）读入时规范化为新名，并按新名与新源码计算 `AlgorithmSpec.sha256()`；系统 MUST 维护"旧哈希 → 新哈希"对照表，每条附 CPU 逐位一致对照证据。

#### Scenario: 旧名读入得新哈希
- **WHEN** 同一算法分别用旧 Miles 路径名与新中立名配置
- **THEN** 两者得到同一个新哈希，且该新哈希与标准样本中对应的旧哈希一起列在对照表里

#### Scenario: 新旧版本不混跑
- **WHEN** 新版本代码的岛与旧版本代码的岛（算法哈希不同）尝试合并，或用旧哈希检查点续训
- **THEN** 拒绝并提示查看对照表与发布说明

### Requirement: 改名后训练端绑定核对
对每个启用的中立过滤器/插件名，系统 SHALL 在启动前核对映射后的训练端绑定与改名前一致：传给后端的参数逐字相同；加载的插件模块路径、函数限定名、源码哈希相同；以固定输入经映射路径实际调用一次插件，结果与改名前路径逐位相同。任一项不一致 MUST 启动失败并指出不一致项；核对结果 MUST 写入运行时清单。

#### Scenario: 映射到错误函数
- **WHEN** 某中立名被映射到与改名前不同的函数
- **THEN** 启动失败，报出插件身份不一致的中立名、期望与实际函数

#### Scenario: 参数拼写漂移
- **WHEN** 映射后传给 Miles 的某个参数与改名前不逐字相同
- **THEN** 启动失败并列出差异参数

#### Scenario: 正常配置
- **WHEN** 标准样本中的任一配置启动
- **THEN** 三项核对全部通过并记入运行时清单

### Requirement: 逐词元损失参考函数只做对照
yeto SHALL 提供中立的逐词元策略损失参考函数（PPO clip/dual-clip、CISPO、SAPO、KL 估计器、TIS/IcePop 权重与掩码），仅用于 CPU 对照测试；本 change MUST NOT 改动 Miles fork 的损失分派。

#### Scenario: 与 Miles fork 对照
- **WHEN** 在 CPU 上以相同 dtype 与输入比较参考函数与 Miles fork 对应函数
- **THEN** 结果逐位一致，或测试记录差异并标明该组合未对齐
