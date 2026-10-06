# Proposal

## Why

yeto 在 `docs/CLOUDS.md` 里把 Verda 列为可用云，但到目前为止**还没有一次通过 yeto 自己的 launcher 在 Verda 上跑成功**。2026-09-29 的 RL 真实运行（`rl-engine-ports` 任务 7.1 与跨云测试）暴露了一串问题，其中最严重的会**删掉 yeto 自己正在运行的 Verda 实例**：

1. **SkyPilot 0.13 的 Verda 适配器有参数错位（R2）。** `provision/__init__.py:193` 以 `(provider, cluster_name, cluster_name_on_cloud, provider_config, …)` 调用 `query_instances`，而 `provision/verda/instance.py:389` 仍按旧签名 `(cluster_name_on_cloud, provider_config, …)` 接收，于是拿显示名当云上名字。yeto 的集群名含大写 region（`FIN-01`），与小写主机名做子串匹配必然失败，状态刷新对正在运行的实例返回空，sky 删掉集群记录。Nebius 等云的集群名全小写，碰巧能匹配，所以只在 Verda 上出现。
2. **恢复时误删运行中的实例（R1）。** 记录丢失后，yeto 的恢复逻辑用同一个集群名重拉（`yeto/launcher.py:3162-3260`）。sky 查找已有实例时匹配状态 `'ACTIVE'`（`instance.py:102-103`），Verda 实际返回 `running`，于是另建新实例；新建因容量失败后，sky 清理时按主机名子串、且不看状态地 terminate（`instance.py:42`、`:252-295`），把原来那台正在运行的实例删掉。跨云测试的 attempt1、attempt2 都这样丢了岛。
3. **抢不到卡（R3、R4、R7）。** sky 的 Verda catalog 是每 7 小时拉一次的库存快照（`catalog/verda_catalog.py:18-20`），只含抓取那一刻有货的组合：没有 L40S，H100 只有 spot；provision 时也不查实时库存。yeto 的候选是"全部型号 × 全部 location"的笛卡尔积（`yeto/shape/providers.py:937-970`），下发给 sky 时不带 `instance_type`（`launcher.py:1929-1936`）。结果是 yeto 去申请没货的型号，真正有货又便宜的型号却申请不了。
4. **RL 岛在 Verda 上根本起不来（R5）。** sky 在 Verda 上不支持 docker 镜像和对象存储挂载（`clouds/verda.py:38-49`），而 yeto 无条件设置 `image_id`（`launcher.py:1938`），spot 模式还会挂存储；planner 也把 Verda 排除在 RL 白名单之外（`yeto/shape/plan.py:343`、`catalog.py:79-80`）。
5. **Verda 当不了 head（open_ports）。** head 模式的 syncer 任务向 sky 声明 `ports`（`launcher.py:394`、`:421`，`cli.py:1335`），sky 的 Verda 适配器不支持 `OPEN_PORTS`，直接报 `NotSupportedError`。但 Verda 平台层没有防火墙，实例的端口本身是可达的，这只是 sky 的能力声明问题。
6. **yeto provider 自身的 bug（R6）。** 凭据只读 JSON 格式的 `~/.verda/config.json`，不认 Verda 官方 CLI 写的 INI 格式 `~/.verda/credentials`；非 JSON 响应（创建返回纯文本 id、删除返回空）直接 `json.load` 报错；备用 location 列表里还有已下线的 ICL-01；型号表缺 RTX PRO 6000 与 A100 40GB（`providers.py:858-912`）。

这些问题在 yeto 侧与 SkyPilot 侧各有一部分。完整根因报告见 `/home/michael/work/verda-rca/report.md`，修复思路的脑暴见本 change 的讨论记录。

## What Changes

按优先级分块（P0 必须先做）：

- **A. provider、凭据与候选（P0）**
  - 同时支持 JSON 与 INI 两种凭据文件；非 JSON 响应按文本处理，HTTP 错误带上响应体；去掉 ICL-01；型号表补 RTX PRO 6000、A100 40GB。
  - 候选只保留"sky 可识别"且"实时 `/instance-availability` 有货"的组合，按实时库存排序；下发 sky 时带显式 `instance_type`，并以 `any_of` 给出多个候选，容量失败后按退避重试。
  - 生成一份全量的本地 sky Verda catalog（型号 × location × 价格），本机与 head 使用同一份，让 sky 能识别所有真实存在的型号。
- **B. 误删防护（P0）**
  - yeto 的集群名统一为小写，从源头避开 R2 的大小写匹配失败。
  - yeto 自带 sky Verda 适配器的运行时补丁，仅在已验证的 sky 版本上生效：修正 `query_instances` 签名；查找已有实例用 `running`；主机名精确匹配并排除已删除实例；失败清理只删本次新建的实例；补全状态映射；查询为空时按实例 id 逐台复查。补丁对本机与 head（包括 sky API server 进程）都生效。
  - 恢复前先按 instance id 向 Verda 确认旧实例状态：仍在运行则不重拉；确认已消失才以**新的集群名**重拉。
  - 补丁未生效（sky 版本不在允许列表）时，Verda 岛禁止自动恢复，并给出告警。
- **C. RL 岛运行时（P0）**
  - Verda 岛不设 sky `image_id`，改为在 Verda 默认 VM 内以 `docker run` 运行 digest 固定的镜像；环境变量安全传递，不写进日志。
  - 本 change 只支持 on-demand，不挂对象存储；spot 另行设计。
- **E. planner 集成（P1）**
  - 新增"VM 内 docker"类云的集合，Verda 据此通过容器镜像要求；不进入多节点与 spot 存储白名单。
  - 价格按 on-demand 计分，实时库存只做加权，不直接删除候选；不支持的组合在 plan 输出中写明拒绝原因。
- **G. 拆除与可观测（P1）**
  - 拆除以 Verda API 按 instance id 核实（deleted / discontinued / 404 才算已删），不再依赖主机名子串。
  - 拆除包含实例的系统卷：卷在 Verda 回收站中也必须永久删除，才算拆除完成。
  - 拆除前 best-effort 回传岛的事件磁带、岛的任务日志（setup/run 输出）与 head 的 sky 日志。2026-09-29 跨云测试中，Verda 岛失败后日志随实例一起消失，无法定位原因；`yeto down` 还曾报告 Verda 岛"已拆除"而实例仍在运行。
- **D. Verda 作 head / syncer（P2）**
  - 在 Verda 上起 head 时不向 sky 声明 `ports`，由 yeto 在启动后从外部探测 syncer 端口可达，并在实例内用 ufw 只放行 SSH 与 syncer 端口。文档仍默认建议 head 放在 Nebius 或 AWS。

## Capabilities

### New Capabilities
- `verda-provisioning`：在 Verda 上创建、恢复与拆除 yeto 实例的安全性。yeto 不得删除任何非本次创建的实例；恢复前必须按 instance id 确认旧实例状态并以新名字重拉；依赖的 SkyPilot 行为只在已验证版本上启用，否则禁止自动恢复；拆除按 instance id 核实。
- `verda-capacity-selection`：Verda 候选的生成与下发。候选必须可被 SkyPilot 识别、带显式实例型号，并按实时库存排序与重试；凭据与 API 响应的解析必须兼容 Verda 官方格式。
- `verda-island-runtime`：Verda 上 RL learner 岛的运行方式。以 VM 内容器运行固定 digest 的镜像，只使用 on-demand，不依赖对象存储；planner 据此接纳 Verda。
- `verda-control-plane`：在 Verda 上运行 head / syncer 时不依赖 SkyPilot 的端口声明，启动后必须验证 syncer 端口从外部可达。

### Modified Capabilities
<!-- 无。openspec/specs/ 目前只有 head-run-teardown；本 change 的拆除核实是 Verda 专属要求，不改动其现有需求。 -->

## Impact

- **yeto 代码**：`yeto/shape/providers.py`（Verda 部分）、`yeto/shape/catalog.py`、`yeto/shape/plan.py`、`yeto/launcher.py`（集群命名、`build_learner_task`、恢复监督器、`terminate_and_verify`、syncer 任务）、`yeto/cli.py`（head bootstrap 与 syncer 端口）；新增 SkyPilot Verda 运行时补丁模块。
- **文档**：`docs/CLOUDS.md` 的 Verda 条目（凭据格式、location 列表、状态由 pending 改为实际验证结果、head 放置建议）。
- **依赖**：SkyPilot 保持 0.13（补丁带版本守卫）；不 fork SkyPilot。
- **与其他 change 的关系**：
  - `rl-engine-ports` 的 7.1 在 Verda 上的真实运行已暂停，等本 change 完成后重新发起。
  - C 块与 `rl-engine-ports` 同改 `launcher.py`，建议在 `rl-engine-ports` 合入 main 之后再落地 C 块，避免两次处理冲突。
- **费用**：真机验收预计约 $10–15（单卡 A100 80GB 或 L40S on-demand）。

## Non-goals

- 不支持 Verda spot 与多节点。
- 不做镜像预热或快照（冷启动优化列为后续调研）。
- 不 fork SkyPilot；是否向 SkyPilot 上游提交 R1/R2 修复 PR 另行征得用户同意。
- 不修改 Modal 相关问题（全 Modal fleet 的 head 放置、Modal 岛事件磁带随容器退出丢失），这些另立 change。

## 待确认的决策

以下按推荐方案写入本 proposal，用户确认或修改后再写 spec 与 design：

1. 误删补丁的载体：yeto 内置 monkeypatch 加版本守卫，不 fork SkyPilot。
2. 是否向 SkyPilot 上游提 R1/R2 修复 PR：先用内置补丁验证，之后再征得同意。
3. 补丁上线前，Verda 岛强制禁止自动恢复。
4. 本 change 只支持 on-demand，spot 推后。
5. catalog 策略：本地全量 CSV + 实时 availability 排序 + `any_of` 多候选。
6. Verda 作 head：不声明 ports、外部探测、ufw；文档默认仍建议 head 放 Nebius 或 AWS。
7. C 块在 `rl-engine-ports` 合入 main 之后落地。
8. 真机验收预算上限约 $15。
9. 镜像预热快照不纳入本 change。
