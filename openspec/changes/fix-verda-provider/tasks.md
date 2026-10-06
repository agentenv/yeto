# Tasks

## 1. provider、凭据与候选（A 块）

- [x] 1.1 `yeto/shape/providers.py`：凭据同时支持 JSON、INI（Verda CLI 格式）与环境变量；非 JSON 响应按文本处理，HTTP 错误保留响应内容且不输出凭据；地区列表去掉 ICL-01；型号表补 RTX PRO 6000、A100 40GB（参考 `/home/michael/work/gpu-verda/r0fix/verda.diff`）。验证：用 `/home/michael/work/verda-rca/*.json` 与新增的纯文本、空响应、503 夹具做单测，覆盖三种凭据来源。
- [x] 1.2 生成本地 SkyPilot Verda 目录（全部型号 × 全部地区 × 价格），本机在启动前生成，head 在 bootstrap 时生成同一份。验证：单测用临时 `SKY_HOME` 让 SkyPilot 的 Verda 目录模块读取生成的 CSV，L40S、RTX PRO 6000 等型号可被识别。
- [x] 1.3 候选按实时 `/instance-availability` 过滤与排序；下发 SkyPilot 时带 `instance_type` 与 `any_of` 多候选；容量失败后刷新可用性并有上限退避，全部失败时报告各候选的失败原因。验证：单测覆盖排序、`any_of` 生成、无货降权、退避上限与失败报告。

## 2. 误删防护（B 块）

- [x] 2.1 集群名统一小写（`yeto/launcher.py` 的 `learner_cluster_names` 及其他生成集群名的位置）。验证：单测断言含大写 region 的输入生成全小写集群名；已有运行的 `yeto down` 仍按运行记录中的名字拆除。
- [x] 2.2 新增 `yeto/sky_patches/verda.py`：修正 `query_instances` 签名、查找状态、主机名精确匹配、失败清理只删本次新建 id、状态映射补全、等待条件 `>=`、空结果按 id 复查；带版本守卫。验证：mock Verda 客户端的单测覆盖同名前缀实例不被删、创建失败保留运行中实例、未知状态映射为 INIT、空结果按 id 复查、版本不匹配时不打补丁。
- [x] 2.3 补丁在本机与 head 都生效：本机在 yeto 导入 sky 后应用；head bootstrap 安装固定版本的 SkyPilot（`yeto/cli.py:1244`），并通过 `.pth` 入口让 SkyPilot API 服务进程也加载补丁。验证：单测检查 bootstrap 命令固定了版本并写入 `.pth`；真机验收项见 5.x。
- [x] 2.4 运行记录保存 Verda 实例 id；恢复前按 id 核实旧实例状态：仍在运行则不重拉并报告，已消失则以新集群名重拉。验证：恢复监督器单测覆盖 running、deleted、不存在三种情况与新名字生成。
- [x] 2.5 补丁未生效（版本不在已验证列表）时，Verda 岛强制 `recover_timeout=0` 并在启动时告警。验证：单测断言版本守卫失败时 Verda 岛的恢复被禁用、日志含告警，其他云不受影响。

## 3. 拆除与可观测（G 块）

- [x] 3.1 拆除前 best-effort 回传岛的事件磁带与作业日志、head 的 SkyPilot 日志与集群事件，失败只告警。验证：单测用 mock 的回传命令覆盖成功、超时、失败三种情况，拆除都会继续。
- [x] 3.2 Verda 拆除核实改为按实例 id 查询 Verda，并确认系统卷已永久删除（含回收站）；未确认时报告拆除未完成并列出剩余资源。验证：单测覆盖"SkyPilot 报告已删但实例仍在运行"与"卷在回收站"两种情况。

## 4. 虚拟机内容器与规划器（C、E 块，在 `rl-engine-ports` 合入 main 之后）

- [x] 4.1 新增"虚拟机内容器"云集合；对 Verda 不设 `image_id`，改为 setup 中拉取固定 digest 镜像、run 中 `docker run --gpus all --network host`；环境变量经 0600 的 `--env-file` 传入；处理容器内外文件属主差异。验证：launcher 单测断言 Verda 任务无 `image_id`、命令被正确包装、日志中不出现环境变量值、属主处理步骤存在。
  - 完成记录（2026-10-04，S8 verda-fix C 块）：`launcher.in_vm_docker_image/in_vm_docker_setup`（Verda 不设 image_id，VM 内拉固定 digest 并 docker run）、`_verda_copy_override`；修复 .pth exec 作用域、stale yeto 回退、503 后 `KeyError region`。真机：Verda G0（1×A100 FIN-01，`verda-g0-20261004c`，≈$0.5）功能检查全过（judge 仅 `gpu_is_L40S` 因卡型 FAIL）。
- [ ] 4.2 规划器用"已验证容器镜像云 ∪ 虚拟机内容器云"判断容器需求；Verda 按按需价格计分，不进多节点与 spot 存储白名单；Verda spot 或需要对象存储的配置在启动前拒绝并说明原因。验证：plan 单测覆盖 Verda 容器岛可行、Verda spot 被拒、拒绝原因写入输出。
- [ ] 4.3 更新 `docs/CLOUDS.md` 的 Verda 条目：凭据格式、地区列表、实际验证结果、head 放置建议。验证：文档中的命令在 `--dry-run` 下可执行。

## 5. Verda head / syncer（D 块）

- [x] 5.1 对 Verda 的 syncer 任务不传 `ports`；启动后本机对 syncer 端口做 TCP 探测，成功才启动岛，失败则拆除 head 并报告；实例内 ufw 只放行 SSH 与 syncer 端口。验证：单测覆盖探测成功与失败两条路径、ufw 规则生成。

## 6. 真机验收（单卡按需 A100 80GB 或 L40S，预算约 $15）

- [ ] 6.1 按默认参数在 Verda 上起单卡 RL 岛（head 在 Nebius 或本机），岛在容器内完成 fork checkout 与校验并完成至少 1 轮同步；拆除后按 id 确认实例与卷都已删除。验证：证据目录含岛作业日志、事件磁带、拆除证明。
  - 进度（2026-10-04）：单卡 G0 已在 Verda 真机跑通（controller 本机，1 次 train + 1 次 generate，容器内 fork checkout 校验通过）；本条要求的"≥1 轮同步"默认参数整轮运行尚未执行，保持未勾。另：sky 0.13 Verda 后端 `num_nodes>1` 直接 ValueError（`sky/clouds/verda.py`），多节点岛不可用。
- [ ] 6.2 误删回归：先起一台 Verda 岛，再以同名触发一次容量不足的重拉，确认旧实例仍在运行；确认 head 上 SkyPilot API 服务进程已加载补丁。验证：Verda API 查询记录与 head 日志。
- [ ] 6.3 恢复：手动删除岛实例，yeto 按 id 确认已消失后以新集群名重拉并继续训练。验证：launch 日志与 Verda API 记录。
- [ ] 6.4 Verda head：在 Verda CPU 实例上起 head，外部探测 syncer 端口成功，ufw 拒绝其他端口；岛（Verda 或其他云）连上 syncer 完成至少 1 轮同步。验证：探测与同步日志、拆除证明。
- [ ] 6.5 恢复 `rl-engine-ports` 7.1 的 Verda 默认参数真实运行（2 岛 strict-avg 3 轮），作为本 change 的收尾验证。验证：两岛每轮 hash 一致、grad_norm 与 delta 非零、拆除证明。

## 完成记录

分支 `fix-verda-provider-r0`（基于 `origin/rl-engine-ports` d355e06）。证据目录 `openspec/changes/fix-verda-provider/evidence/`：
`test_verda_provider.txt`（53 项单测逐项结果）、`real-sky-scenarios.json`（在 SkyPilot 0.13.0 真实 Verda 适配器上、以内存假 Verda 客户端驱动的补丁场景）、`pytest-summary.txt` 与 `pytest-failures-{before,after}.txt`（全量回归失败集合对比，按测试 id 去重后完全相同）。
以下均为 CPU 单测 / 本地假客户端验证，**不含真实云实验**；真机验收在 6.x。

- 1.1 `yeto/shape/providers.py`：`verda_credentials` 依次读环境变量、`~/.verda/config.json`、`~/.verda/credentials`（INI，任意 section）；`_verda_request` 对 JSON / 纯文本 id / 空响应分别返回对象 / 字符串 / None，HTTP 错误带响应体且脱敏凭据与 bearer；地区列表去掉 ICL-01；型号表加 `RTX PRO 6000 → RTX-PRO-6000`、`A100 40GB → A100`（`launcher.GPU_MEM_GB`、`catalog.PEAK_TFLOPS_BF16` 同步补齐）。测试：`test_credentials_*`、`test_request_handles_json_text_and_empty[*]`（含 verda-rca 抓取的 `instance_types.json`）、`test_http_error_keeps_body_and_hides_credentials`（503）、`test_locations_and_models`。
- 1.2 `write_verda_sky_catalog` 写"全部型号 × 全部地区 × 价格"到 `$SKY_RUNTIME_DIR/.sky/catalogs/v8/verda/vms.csv`（SkyPilot 0.13 实际使用的变量是 `SKY_RUNTIME_DIR`，即任务中的"临时 SKY_HOME"）；本机在 `launcher.prepare_verda_islands` 生成，head 在 bootstrap 的 `VERDA_HEAD_CATALOG_STEP` 生成。测试 `test_generated_catalog_is_what_sky_reads` 在 SkyPilot 0.13.0 Python 子进程中用临时目录让 `sky.catalog.verda_catalog` 读取生成的 CSV：`1L40S.20V`、`1RTXPRO6000.30V` 存在，加速器含 L40S / RTX-PRO-6000 / A100 / A100-80GB / H100，价格与地区正确；`test_head_bootstrap_writes_the_catalog_for_verda_fleets`。
- 1.3 `verda_candidates`（按实时可用性过滤，按失败次数降权→价格→地区排序，上限 4）、`verda_any_of`（带 `instance_type` 的有序多候选）、`launch_with_verda_candidates`（每次重试重新拉取可用性、指数退避有上限、非容量错误立即抛出、全部失败抛 `VerdaCapacityExhausted` 并逐候选列原因）；`launcher.launch_verda_island` 把候选作为有序 `Resources` 列表下发。测试：`test_candidates_*`、`test_launch_*`、`test_launch_verda_island_sends_ordered_any_of`。
- 2.1 `launcher.sky_cluster_name` 把学习岛、syncer、head 集群名统一小写；`yeto down` 使用运行记录中的原名。测试：`test_cluster_names_are_lower_case`、`test_down_uses_recorded_names_verbatim`。
- 2.2 `yeto/sky_patches/verda.py`：新签名 `query_instances`、`ACTIVE→running`、主机名精确匹配（`-head` / `-worker[-N]`）、失败清理只删本次新建 id（随后 provisioner 的 teardown 不再删旧实例）、状态映射补全且未知→INIT、等待条件 `>=`、空结果按已记录 id 复查、版本守卫（仅 0.13.0）。测试：假模块 8 项 `test_patch_*`，以及 `test_patch_against_real_sky_provisioner`（真实 sky 0.13.0 适配器，结果见 `real-sky-scenarios.json`：同名前缀实例未删、503 时运行中实例保留、未知状态 INIT、空列表按 id 复查为 UP、经 dispatcher 的参数顺序正确、版本不匹配不打补丁）。
- 2.3 本机：`launcher.run` 导入 sky 后 `sky_patches.install()`（导入钩子），并为本机 SkyPilot API 服务进程写 `.pth`（`ensure_local_pth`；已在运行的 API 服务需 `sky api stop` 重启，会提示）。head：bootstrap 固定 `skypilot[...]==0.13.0`（`cli.HEAD_SKYPILOT_VERSION`）并写 `.pth`。测试：`test_head_bootstrap_pins_sky_and_writes_pth`、`test_pth_line_installs_the_hook_in_a_fresh_interpreter`、`test_install_patches_on_import_of_the_target`、`test_local_pth_for_the_sky_api_server`。"API 服务进程确实加载补丁"的真机核验在 6.2。
- 2.4（审查后重做）`yeto/verda_ops.py` `VerdaInstanceGuard`：岛启动后用 launch handle 的 `cluster_name_on_cloud`（含 user hash；sky 真实主机名为 `<name_on_cloud>-head`）精确匹配取实例 id，经 `on_instance_ids` 写入运行记录 `verda_instance_ids`；重拉成功后用 sky 记录中的新 `cluster_name_on_cloud` 重新记录。重拉前按 id 查询：running/provisioning 等→不重拉并报告；deleted / 404→以 `<name>-rN` 新名重拉；无 id 记录或查询失败→"unknown"，本轮不重拉（不再按显示名猜）。测试：`test_guard_by_instance_id[running|provisioning|deleted|None]`（含"显示名匹配不到、无 id 不重拉"）、`test_controller_*`、`test_worker_saves_verda_ids_in_the_run_record`；真实命名：`test_patch_against_real_sky_provisioner` 中 `real_naming`，用 sky 0.13.0 的 `make_cluster_name_on_cloud` 生成名字（`vfix-l0-fin-01-1057ec71`），经补丁后的 `run_instances` 创建 `…-head`，显示名匹配为空、真实名匹配到 id。
- 2.5 `prepare_verda_islands`：版本不在已验证列表（或 `.pth` 写不进）时 Verda 岛进入 `no_recover`（等价 `recover_timeout=0`）并告警，其他云照常恢复。测试：`test_unverified_sky_disables_verda_recovery_only`、`test_local_pth_for_the_sky_api_server`。
- 3.1 `launcher.collect_teardown_diagnostics`（`sky logs`、岛 `~/yeto-output/*.jsonl` 事件磁带；head 上另取 `~/.sky/api_server/server.log` 与集群事件），每条有超时、失败只告警；`teardown_island` 先回传再拆除。测试：`test_diagnostics_never_block_teardown[ok|timeout|fail]`。
- 3.2（审查后重做）`verda_ops.verify_teardown(api, ids, name_on_cloud)` 只处理运行记录中的实例 id：残留则按 id 再删；只永久删除这些实例的 **OS 卷**（`os_volume_id`，含回收站；yeto 不创建其他卷，附加卷不动）；按主机名新匹配到、但不在记录中的实例只报告为"疑似残留"、不删除，且结果判为未完成；id 集为空时判为失败。`launcher._verda_teardown_check` 在无 id 时返回 None，回到 sky 的云端探测，不会把空集当成"已删除"。测试：`test_teardown_reports_a_node_sky_called_deleted`、`test_teardown_purges_volumes_from_the_trash`、`test_teardown_never_succeeds_without_ids_and_never_deletes_strangers`、`test_verda_teardown_check_falls_back_without_ids`、`test_terminate_and_verify_uses_the_verda_answer`；真实命名下 `real_naming.verify_after_down == [true, []]`、`verify_without_ids` 为 false。
- 5.1 `launcher.syncer_ports` 对 Verda 返回 None（syncer 与 head 任务都不传 `ports`），`ufw_setup` 只放行 22 与 syncer 端口；本地 syncer 集群起来后 `tcp_probe`；head 模式先在 head 上起一次性监听（`probe_listener_command`）由本机探测，失败则拆除 head、不提交控制作业。测试：`test_verda_syncer_gets_no_ports_and_a_ufw_setup`、`test_tcp_probe_success_and_failure`、`test_head_launch_on_verda_probes_before_islands[True|False]`。

审查修复（2026-09-29 第二轮）：
- 2.5：本机 `.pth` 本次才写入（已运行的 API 服务可能未加载），或 sky API 服务在远端（`remote_sky_api_server`），都按"补丁未生效"处理、Verda 岛不自动恢复（`test_unverified_sky_disables_verda_recovery_only`、`test_remote_api_server_detection`）。持久安装入口 `python -m yeto.sky_patches install|uninstall|status`。
- `.pth` 改为惰性钩子：只有导入 `sky.provision.verda.instance` 时才导入 yeto（优先已安装的 yeto，否则回退到记录的 worktree 路径，路径不存在时静默跳过）；首行记录 worktree；覆盖其他 worktree 的钩子会告警；launcher 只在运行结束时删除本次自己写入且内容未变的 `.pth`（`test_pth_line_installs_the_hook_in_a_fresh_interpreter`、`test_local_pth_for_the_sky_api_server`）。
- `VerdaApi` token 按 `expires_in` 提前 60 s 刷新，遇 401 强制刷新并重试一次（`test_verda_api_refreshes_expired_token_and_retries_401`）；脱敏正则加入 `refresh_token`、`client_secret`。
- 补丁 `_FAILED_LAUNCH` 带时间戳、TTL 600 s，第一次 teardown 即清除，失败清理时重试未删掉的新建实例，之后的 `sky down` 是完整拆除（`test_failed_launch_marker_expires_and_retries_failed_deletes`）。
- 拆除前诊断改为按岛并行、总时长上限 300 s（`test_parallel_diagnostics_are_bounded`）。
- 5.1 `_probe_head_port`：监听作业在后台线程提交，探测同时进行，不依赖 `sky.exec` 的 `stream_and_get` 是否阻塞（`test_probe_runs_while_the_listener_job_is_still_submitting`）；`stream_and_get` 在真实 head 上的实际行为列入 6.4 核实项。

未完成：4.1–4.3 依赖 `rl-engine-ports`（PR #69）合入 main；6.x 真机验收尚未运行（计划见 `progress.md`）。
