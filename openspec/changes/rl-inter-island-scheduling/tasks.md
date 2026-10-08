# Implementation tasks

标记：`[CPU]` 本机可完成；`[GPU]` 需上卡（开卡前需用户批准并预登记）；`[RS]` 需改 syncer Rust。不改 fork Miles/SGLang。

## 0. 阶段 0：CPU（$0）

- [x] 0.1 [CPU] design.md 逐题回答 §4.1 的 10 问，列"待用户裁定"。
- [x] 0.2 [CPU] `yeto/rl/engine/island_ledger.py`：条目、陈旧度（max_outer_lag=2）、三态判定、合并权重、成员 join/leave、P4 算力加权步进 + carried_over γ^lag、P9 syncer_epoch fencing（用户裁定 2026-10-07）。验收：`tests/test_rl_inter_island_ledger.py`。
- [x] 0.3 [CPU] `IslandStatus` 调度字段 + `inspect()` 填充。验收：`tests/test_rl_inter_island_status.py`。
- [x] 0.4 [CPU] journal pool_* 写入与重放（单调校验）。验收：同上测试文件。
- [x] 0.5 [CPU] `yeto/rl/engine/pause_advice.py`：PauseAdvice 与合并规则（纯函数；接入 controller pause 调用点留待 0.9）。验收：`tests/test_rl_inter_island_status.py`。
- [x] 0.6 [CPU] `yeto/rl/engine/fake_islands.py` 多进程假岛 harness（quorum 超时步进、租约过期退岛、catch-up 零权重），输出 tape JSON。验收：`tests/test_rl_inter_island_harness.py`。
- [x] 0.7 [CPU] 跑新测试与受影响既有测试（controller），结果写 progress.md 与 EXPLORE-S15 §B。
- [x] 0.9 [CPU] PauseAdvice 接入 controller 的 pause 判定调用点（`IslandController(pause_advice_source=...)`，来源 = 协调器租约）；只在 elastic 生效，读不到建议时拒绝暂停；legacy 下与原 pause_decision 结果逐项相等（单测）。
- [ ] 0.8 [RS][CPU] syncer 协议扩展 D-S1..S6（含 P4 θ/T_soft/γ 编入契约哈希、syncer_epoch）+ Rust 单测；Python 假岛 tape 作为黄金用例比对。依赖：Q2 已裁定（P8a）。 **进展（分支 s15-interisland-rs，未勾）**：模式开关与契约哈希、五种新消息帧与 HMAC、syncer_epoch 防旧实例、算力比例步进 / 软截止 / 迟到折扣 / 退岛丢弃 / 新岛首轮权重 0 的状态机已在 `syncer/src/elastic.rs` 实现，`cargo test` 129 项全过；尚缺：状态机接入服务器连接循环（目前 elastic 启动即拒绝）、D-S5 状态导出、D-S6 检查点字段、与 Python 假岛 tape 的黄金比对。
- [x] 0.8b [CPU] Python 侧模式开关：`IslandSchedulingMode`（默认 legacy），账本 / journal pool 记录 / IslandStatus 调度字段 / 假岛按模式分支；单测证明 legacy 等价现有 syncer 严格同步（到齐数等于岛数、额外等待 0、超时失败、拒收迟到增量）。
- [ ] 0.8a [RS][CPU] syncer Rust 侧按契约中的模式字段分支：legacy 不加载新消息类型、合并步进不变（legacy 路径逐行不动），elastic 走新语义；Rust 单测覆盖 legacy 等价与 elastic 的算力比例步进 / epoch 防旧实例。
- [ ] 0.13 [CPU] launcher / driver 增加 `--rl-island-scheduling legacy|elastic`（默认 legacy），传入 IslandController 与 syncer 契约。
- [ ] 0.10 [CPU] 用已存 rollout 数据离线比较 M2PO vs 现有 TIS/IcePop 在 lag 1–4 外层步的 IS 比分布（截断比例、方差、有效样本数），据此定 `StalenessPolicy.correction` 默认值（用户裁定 2026-10-07）。
  - 2026-10-07 夜：本机已存数据只有陈旧度 0 的每轮汇总标量，缺逐 token 的跨版本对数概率，**数据不足，未比较**；需采集字段见 progress.md。不勾，等真机顺带采集。
- [x] 0.11 [CPU] 慢岛降级建议：`pause_advice.slow_island_advice`（某岛每轮耗时超过其它岛中位数的 2 倍即出建议，`target_resource_intent={action:"rollout_only", requires_human_confirmation: true}`，不是否决、不给暂停预算、不执行）；假岛在 elastic 下把建议写进 tape（`timeline` 的 `pause_advice` 事件），legacy 不出建议。云价再分配只保留字段接口。
- [x] 0.12 [CPU] 对象存储样本池：索引格式 `SampleIndexEntry`（schema `yeto.rl.sample-index/v1`）、`SamplePoolIndex`（组不跨岛、按 syncer 记录判定筛选、清理旧索引）与 driver 接口草案 `CrossIslandSampleSource.fetch`（未实现下载）；design 新增一节；单测 `tests/test_rl_inter_island_sample_pool.py`。

## 1. 阶段 1：两岛小模型真机（≈$30，待批，不预登记）

- [ ] 1.1 [CPU] prelaunch review + 脚本（两岛小模型，中途 kill/拉起一岛，跨岛样本 ACCEPT_IS、P4 carried_over、P6 降级 advice）；θ/γ 在此校准。依赖：0.8、rl-infra-spec 3.8/X6。
- [ ] 1.0 [GPU] legacy 回归（待批）：两岛小模型真机用 legacy 模式跑一次，与历史两岛结果对比（外层合并次数、reward、权重哈希），确认旧模式未变。
- [ ] 1.2 [GPU] 执行；判据：成员变化不触发退出码 4/6、catch-up 首轮零权重在 syncer tape 可见、reward 不劣于无跨岛样本基线、带宽实测入档。

## 2. 阶段 2：FN 2×8（待批）

- [ ] 2.1 [GPU] FN 两岛异构等待时间与跨岛样本收益；依赖阶段 1 与 FN 单岛结论。
