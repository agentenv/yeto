# Tasks

## 1. provider、凭据与候选（A 块）

- [ ] 1.1 `yeto/shape/providers.py`：凭据同时支持 JSON、INI（Verda CLI 格式）与环境变量；非 JSON 响应按文本处理，HTTP 错误保留响应内容且不输出凭据；地区列表去掉 ICL-01；型号表补 RTX PRO 6000、A100 40GB（参考 `/home/michael/work/gpu-verda/r0fix/verda.diff`）。验证：用 `/home/michael/work/verda-rca/*.json` 与新增的纯文本、空响应、503 夹具做单测，覆盖三种凭据来源。
- [ ] 1.2 生成本地 SkyPilot Verda 目录（全部型号 × 全部地区 × 价格），本机在启动前生成，head 在 bootstrap 时生成同一份。验证：单测用临时 `SKY_HOME` 让 SkyPilot 的 Verda 目录模块读取生成的 CSV，L40S、RTX PRO 6000 等型号可被识别。
- [ ] 1.3 候选按实时 `/instance-availability` 过滤与排序；下发 SkyPilot 时带 `instance_type` 与 `any_of` 多候选；容量失败后刷新可用性并有上限退避，全部失败时报告各候选的失败原因。验证：单测覆盖排序、`any_of` 生成、无货降权、退避上限与失败报告。

## 2. 误删防护（B 块）

- [ ] 2.1 集群名统一小写（`yeto/launcher.py` 的 `learner_cluster_names` 及其他生成集群名的位置）。验证：单测断言含大写 region 的输入生成全小写集群名；已有运行的 `yeto down` 仍按运行记录中的名字拆除。
- [ ] 2.2 新增 `yeto/sky_patches/verda.py`：修正 `query_instances` 签名、查找状态、主机名精确匹配、失败清理只删本次新建 id、状态映射补全、等待条件 `>=`、空结果按 id 复查；带版本守卫。验证：mock Verda 客户端的单测覆盖同名前缀实例不被删、创建失败保留运行中实例、未知状态映射为 INIT、空结果按 id 复查、版本不匹配时不打补丁。
- [ ] 2.3 补丁在本机与 head 都生效：本机在 yeto 导入 sky 后应用；head bootstrap 安装固定版本的 SkyPilot（`yeto/cli.py:1244`），并通过 `.pth` 入口让 SkyPilot API 服务进程也加载补丁。验证：单测检查 bootstrap 命令固定了版本并写入 `.pth`；真机验收项见 5.x。
- [ ] 2.4 运行记录保存 Verda 实例 id；恢复前按 id 核实旧实例状态：仍在运行则不重拉并报告，已消失则以新集群名重拉。验证：恢复监督器单测覆盖 running、deleted、不存在三种情况与新名字生成。
- [ ] 2.5 补丁未生效（版本不在已验证列表）时，Verda 岛强制 `recover_timeout=0` 并在启动时告警。验证：单测断言版本守卫失败时 Verda 岛的恢复被禁用、日志含告警，其他云不受影响。

## 3. 拆除与可观测（G 块）

- [ ] 3.1 拆除前 best-effort 回传岛的事件磁带与作业日志、head 的 SkyPilot 日志与集群事件，失败只告警。验证：单测用 mock 的回传命令覆盖成功、超时、失败三种情况，拆除都会继续。
- [ ] 3.2 Verda 拆除核实改为按实例 id 查询 Verda，并确认系统卷已永久删除（含回收站）；未确认时报告拆除未完成并列出剩余资源。验证：单测覆盖"SkyPilot 报告已删但实例仍在运行"与"卷在回收站"两种情况。

## 4. 虚拟机内容器与规划器（C、E 块，在 `rl-engine-ports` 合入 main 之后）

- [ ] 4.1 新增"虚拟机内容器"云集合；对 Verda 不设 `image_id`，改为 setup 中拉取固定 digest 镜像、run 中 `docker run --gpus all --network host`；环境变量经 0600 的 `--env-file` 传入；处理容器内外文件属主差异。验证：launcher 单测断言 Verda 任务无 `image_id`、命令被正确包装、日志中不出现环境变量值、属主处理步骤存在。
- [ ] 4.2 规划器用"已验证容器镜像云 ∪ 虚拟机内容器云"判断容器需求；Verda 按按需价格计分，不进多节点与 spot 存储白名单；Verda spot 或需要对象存储的配置在启动前拒绝并说明原因。验证：plan 单测覆盖 Verda 容器岛可行、Verda spot 被拒、拒绝原因写入输出。
- [ ] 4.3 更新 `docs/CLOUDS.md` 的 Verda 条目：凭据格式、地区列表、实际验证结果、head 放置建议。验证：文档中的命令在 `--dry-run` 下可执行。

## 5. Verda head / syncer（D 块）

- [ ] 5.1 对 Verda 的 syncer 任务不传 `ports`；启动后本机对 syncer 端口做 TCP 探测，成功才启动岛，失败则拆除 head 并报告；实例内 ufw 只放行 SSH 与 syncer 端口。验证：单测覆盖探测成功与失败两条路径、ufw 规则生成。

## 6. 真机验收（单卡按需 A100 80GB 或 L40S，预算约 $15）

- [ ] 6.1 按默认参数在 Verda 上起单卡 RL 岛（head 在 Nebius 或本机），岛在容器内完成 fork checkout 与校验并完成至少 1 轮同步；拆除后按 id 确认实例与卷都已删除。验证：证据目录含岛作业日志、事件磁带、拆除证明。
- [ ] 6.2 误删回归：先起一台 Verda 岛，再以同名触发一次容量不足的重拉，确认旧实例仍在运行；确认 head 上 SkyPilot API 服务进程已加载补丁。验证：Verda API 查询记录与 head 日志。
- [ ] 6.3 恢复：手动删除岛实例，yeto 按 id 确认已消失后以新集群名重拉并继续训练。验证：launch 日志与 Verda API 记录。
- [ ] 6.4 Verda head：在 Verda CPU 实例上起 head，外部探测 syncer 端口成功，ufw 拒绝其他端口；岛（Verda 或其他云）连上 syncer 完成至少 1 轮同步。验证：探测与同步日志、拆除证明。
- [ ] 6.5 恢复 `rl-engine-ports` 7.1 的 Verda 默认参数真实运行（2 岛 strict-avg 3 轮），作为本 change 的收尾验证。验证：两岛每轮 hash 一致、grad_norm 与 delta 非零、拆除证明。
