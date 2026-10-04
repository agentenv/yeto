# Design

## Context

- 根因与证据见 `proposal.md` 与 `/home/michael/work/verda-rca/report.md`；跨云证据在 `/home/michael/work/gpu-xcloud/evidence/2026-09-29-xcloud/`（含 attempt3 的 head 诊断，确认了 `query_instances` 参数错位），默认参数 Verda 失败证据在 `/home/michael/work/gpu-default-verda/evidence/`。
- SkyPilot 版本：本机 0.13.0。head 在 bootstrap 时执行 `pip install -q "skypilot[aws,gcp,runpod,nebius,verda]>=0.12"`（`yeto/cli.py:1244`），**没有固定版本**，每次起 head 都可能装到新版本。
- 涉及的 sky 代码：`sky/provision/__init__.py:193`（`query_instances` 调用方）、`sky/provision/verda/instance.py`（`:42` 子串匹配、`:102-103` `'ACTIVE'`、`:197-222` 等待、`:252-295` terminate、`:389` 旧签名、`:400-411` 状态映射）、`sky/clouds/verda.py:38-49`（不支持 docker 镜像、存储挂载、开端口）、`sky/catalog/verda_catalog.py:18-20`（每 7 小时拉取的目录）、`sky/catalog/common.py:222`（本地目录被修改后不再覆盖）。
- 涉及的 yeto 代码（main `e21a7ff`）：`yeto/shape/providers.py:846-1050`（`VerdaSignals`）、`yeto/shape/catalog.py:72-80`（白名单）、`yeto/shape/plan.py:343`、`:354`（拒绝逻辑）、`yeto/launcher.py:1854`（`image_id`）、`:2539`（`learner_cluster_names`）、`:2940-3290`（恢复监督器，`recover_timeout <= 0` 已可禁用恢复，`:3148`、`:3173`）、`terminate_and_verify`、`:394`、`:421` 与 `yeto/cli.py:1335`（syncer 的 `ports`）。
- `rl-engine-ports` 分支也修改了 `launcher.py`（行号不同），最终会合入 main。

## Goals / Non-Goals

**Goals:**
- Verda 上的实例不会被 yeto 或 SkyPilot 误删；拆除结果以 Verda 实际状态为准。
- yeto 能用 Verda 实际有货的型号起单卡按需 RL 岛。
- Verda 可以作为 head / syncer 所在的云。
- 修复在本机与 head 上一致生效，SkyPilot 版本变化时安全降级。

**Non-Goals:**
- Verda spot、多节点、镜像预热与快照。
- fork SkyPilot，或本 change 内向 SkyPilot 上游提交 PR。
- Modal 的 head 放置与事件磁带问题（另立 change，并与本 change 的 D 块在"syncer 可达性"上统一设计）。

## Decisions

**D1. 集群名统一小写（B 块第一步，改动最小、收益最大）。**
`learner_cluster_names` 等生成集群名的函数把 region 等部分转成小写。这样即使 SkyPilot 的 `query_instances` 参数错位、拿显示名当云上名字，也与小写主机名一致，状态查询不再误返回空，整条"记录丢失 → 同名重拉 → 清理误删"的链条在第一环就断开。
- 跨云测试 attempt4 已在本地验证：同样的容量失败下实例没有被误删。
- 已有运行的集群名会变化：只影响新运行；`yeto down` 按运行记录中的名字拆除，不受影响。

**D2. yeto 自带 SkyPilot Verda 适配器的运行时补丁，并固定 head 上的 SkyPilot 版本。**
新增 `yeto/sky_patches/verda.py`，在 yeto 导入 sky 之后应用：
- `query_instances` 按新签名接收参数；
- 查找已有实例使用 `running` 等真实状态；
- 主机名精确匹配（`{cluster}-head` / `{cluster}-worker-N`），排除已删除、已停止的实例；
- 创建失败时只终止本次新建的实例 id；
- 状态映射补全 `ordered`、`starting_hibernation`、`hibernating`、`restoring` 等，未知状态按 INIT 处理；
- 等待条件改为运行数量 `>=` 目标数量；
- 批量查询为空但有已知实例 id 时按 id 复查（R2 的防御）。

补丁带版本守卫：只在已验证的 SkyPilot 版本上生效；否则不打补丁，并禁用 Verda 岛的自动恢复（见 D4）。head bootstrap 改为安装固定版本（`skypilot[...]==<已验证版本>`），使本机与 head 的版本一致、补丁可预期。
- 补丁必须在 head 的 SkyPilot API 服务进程里也生效。实现方式：在 head 的 Python 环境中放置一个 `.pth` 入口，使任何进程导入 sky 时都加载 yeto 的补丁；真机验收时核实 API 服务进程确实加载了补丁。
- 备选方案：fork SkyPilot（分发与维护成本高）；完全绕开 SkyPilot 直接调用 Verda API（需重做 SSH、任务执行、日志、状态机，工作量大）；等待上游修复（上游 master 仍未修）。

**D3. 恢复前按实例 id 核实，重拉换新名字。**
yeto 在运行记录中保存每个 Verda 集群的实例 id。恢复监督器在决定重拉前，用 yeto 自己的 Verda HTTP 客户端按 id 查询：
- running / provisioning / ordered：不重拉，报告"作业失败、实例仍在"；
- deleted / discontinued / 不存在：用新的集群名（在原名后加递增后缀）重拉。

**D4. 补丁未生效时禁用 Verda 自动恢复。**
沿用现有 `recover_timeout <= 0` 的语义（`launcher.py:3148`、`:3173`），对 Verda 岛强制设为 0，并在启动时给出告警。宁可让岛失败，也不冒误删的风险。

**D5. 候选：本地全量目录 + 实时可用性 + 显式型号与多候选。**
- 从 Verda `/instance-types` 与地区列表生成"全部型号 × 全部地区 × 价格"的本地 SkyPilot 目录（`~/.sky/catalogs/v8/verda/vms.csv`）。本地文件被修改后 SkyPilot 不会覆盖它（`catalog/common.py:222`）。本机与 head 在 bootstrap 时各自生成同一份。
- `VerdaSignals` 用实时 `/instance-availability` 过滤和排序候选；下发给 SkyPilot 时带 `instance_type`，并以 `any_of` 给出 2–4 个候选，由 SkyPilot 依次尝试。
- 容量失败后刷新可用性（缓存失效），有上限的指数退避。
- 型号表补 RTX PRO 6000、A100 40GB；地区列表去掉 ICL-01。
- 备选：只覆盖目录（会反复申请没货的型号）；只用实时可用性（型号不在目录中时 SkyPilot 直接拒绝）。

**D6. 虚拟机内容器运行镜像。**
新增"虚拟机内容器"云集合（初始为 `{"verda"}`）。对这类云，launcher 不设 SkyPilot `image_id`，而是在 setup 中 `docker pull` 固定 digest 的镜像，在 run 中用 `docker run --gpus all --network host` 执行原本的 setup 与 run：
- 挂载工作目录时处理容器内 root 与宿主 ubuntu 用户的属主差异（跨云 attempt4 的岛失败疑似与此有关，例如 git 的 dubious ownership）；
- 环境变量通过 `--env-file` 传入，文件权限 0600，不写入日志；
- 规划器用"已验证容器镜像云 ∪ 虚拟机内容器云"判断能否满足 `needs_container_image`。
- 这块与 `rl-engine-ports` 同改 `launcher.py`，在 `rl-engine-ports` 合入 main 之后实施。

**D7. Verda 上的 head / syncer 不声明端口，启动后外部探测。**
对 Verda，syncer 任务不向 SkyPilot 传 `ports`（`launcher.py:394`、`:421`、`cli.py:1335`）；启动后由本机对 syncer 端口做 TCP 连接探测，成功才启动岛；在实例内用 ufw 只放行 SSH 与 syncer 端口。依据：Verda 平台层没有防火墙（https://docs.verda.com/cpu-and-gpu-instances/securing-your-instance/），端口默认可达，需真机确认。这一方案与之后统一的"syncer 可达性"设计兼容：它只是"开放端口"这种可达方式在 Verda 上的实现。

**D8. 拆除以 Verda 为准，并先回传诊断。**
- 拆除前 best-effort 回传：岛的事件磁带与作业日志（`sky logs` 输出）、head 的 `~/.sky/api_server/server.log` 与集群事件；有超时，失败只告警。
- 拆除后按实例 id 查询 Verda，确认实例已删除；再确认其系统卷已永久删除（回收站中的也删除）。
- `terminate_and_verify` 对 Verda 改用 yeto 自己的 HTTP 客户端核实，不经过 SkyPilot 的 `query_instances`。

## Risks / Trade-offs

- [运行时补丁依赖 SkyPilot 私有函数签名，升级后可能失效] → 版本守卫 + 固定 head 版本 + 单测覆盖每个被替换函数的签名；守卫不通过时自动禁用 Verda 恢复。
- [补丁在 head 的 SkyPilot API 服务进程中未生效] → 用 `.pth` 入口加载，并把"API 服务进程已加载补丁"列为真机验收项。
- [镜像约 70GB，Verda 上拉取约 6 分钟] → 本 change 只保证正确性；镜像预热列为后续调研。
- [Verda 库存薄、变化快] → 多候选 + 退避；全部失败时明确报告，不无限重试。
- [容器内外文件属主不一致] → 容器以宿主 ubuntu 用户的 uid/gid 运行，或在准备步骤前显式设置 `safe.directory`；真机验收覆盖源码 checkout 与校验。
- [端口默认可达的假设不成立] → 外部探测失败即拆除 head 并报告，不会让岛连接不可达的 syncer。

## Migration Plan

1. A、B 块（provider、候选、集群名小写、补丁、恢复与拆除核实）可以直接基于 main 实施。
2. C 块（虚拟机内容器）与 E 块（规划器）在 `rl-engine-ports` 合入 main 之后实施。
3. D 块（Verda head）最后实施。
4. 回滚：各块独立；撤销补丁后 yeto 回到"Verda 禁用自动恢复"的保守状态，不影响其他云。

## Open Questions

以下为按推荐方案写入的假设，用户确认前不影响 A、B 块的实施：
- 是否之后向 SkyPilot 上游提交 R1/R2 修复 PR（需用户另行同意）。
- 真机验收预算上限（按约 $15 规划）。
