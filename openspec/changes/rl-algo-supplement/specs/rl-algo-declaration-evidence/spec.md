## Purpose

规定一个 RL 算法机制从“可表达未开放”转为“已在引擎上声明”时必须具备的 GPU 生效证据、运行纪律，以及永久不可声明的机制类别。

## ADDED Requirements

### Requirement: 声明必须有事先固定判据的生效证据
机制只有在具备以下 GPU 证据后，才可以在 Miles 引擎上声明开放：判据、最小证据、硬超时和费用上限都在运行前写入计划并提交；证据由单次运行得到，不挑 seed；运行后放宽判据无效。声明条目 SHALL 附证据路径，并按取得证据的 Miles 提交限定适用范围；只有在提交之间的代码差异经论证不触及该机制时，才允许把声明迁移到其他提交。

#### Scenario: 无证据的机制保持未开放
- **WHEN** spec 使用了一个没有 GPU 生效证据的机制，并且运行带有外层同步
- **THEN** 能力检查在任何 GPU 进程创建之前拒绝，报错为 expressible but not enabled on this engine，并列出该机制名

#### Scenario: 判据失败不得声明
- **WHEN** 某项 G1 运行有任一预登记判据不满足
- **THEN** 该机制保持未开放；只有在记录原因并修复之后才允许重跑，而且不得修改判据

### Requirement: 生效证据必须避开退化情形
依赖 ratio 偏离 1 的机制（loss 变体、dual-clip、clip-higher、OPSM）的证据运行 SHALL 每轮至少执行 2 个 optimizer step，并以第 2 步及以后的数值作为比较对象。生效 SHALL 通过该机制的专有指标证明；没有专有指标时，SHALL 通过与同 seed、同 pin、同卡型的默认 GRPO 对照组相比，指标不相等来证明。

#### Scenario: 第 1 步数值相同不算证据
- **WHEN** 某变体只在第 1 个 optimizer step 与 GRPO 对照组比较，而且数值相同或不同
- **THEN** 该比较不能作为生效证据

#### Scenario: dual-clip 以专有指标证明
- **WHEN** dual-clip 运行报告的 dual 分支生效比例大于 0，且其他通用判据都满足
- **THEN** dual-clip 可以声明；如果这一指标恒为 0，就不能声明

### Requirement: 疑似有缺陷的指标不得作为证据
当某一指标被发现与 loss 不一致，并且尚未定性时，这一指标 SHALL NOT 用作任何机制的生效证据。如果修复该指标，修复前后 loss 必须逐位相同。

#### Scenario: clipfrac 定性之前
- **WHEN** pg_clipfrac 的疑似缺陷尚未经离线复现定性
- **THEN** clip-higher 等机制的新证据不以 pg_clipfrac 作为判据

### Requirement: 用户自带代码的机制永不声明
由用户提供代码的机制（自定义 loss、插件、自定义修正函数、除内置 Dr.GRPO reducer 以外的任意 pg loss reducer）SHALL 永久保持未声明。这类机制 SHALL 只能通过放行开关使用，并且只在单岛无外层同步的运行中生效。

#### Scenario: 多岛运行使用自定义 loss
- **WHEN** 一个两岛 strict-avg 运行的 spec 使用了自定义 loss，并带有对应的放行参数
- **THEN** 启动前拒绝，报错说明放行只适用于单岛无同步运行

#### Scenario: 声明清单与不可声明清单不相交
- **WHEN** 检查引擎的声明清单
- **THEN** 其中不包含任何用户代码类机制

### Requirement: GPU 验证的成本与回收纪律
每次 GPU 验证 SHALL 使用满足需求的最便宜卡型；只有写明理由的逐位比较才可以使用高端卡。每次运行 SHALL 有云端超时和独立回收机制，有单项费用上限，结束后有无残留证明。阶段累计费用 SHALL 不超过设计中给出的阶段上限。

#### Scenario: 单项超出上限
- **WHEN** 某项运行的实际费用达到该项上限
- **THEN** 运行被中止并记为未完成，不得在同一项上追加预算继续运行

### Requirement: 变 DP 边认证范围不因声明而扩大
在固定 DP 下声明的机制 SHALL NOT 自动获得变 DP 重配置边的认证。

#### Scenario: grpo 加白化请求变 DP 边
- **WHEN** grpo 加 advantage 白化已经声明，而运行请求变 DP 重配置
- **THEN** 该重配置请求仍被拒绝
