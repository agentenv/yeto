# Tasks

## 1. Ray 测试隔离（先做，后面的本机运行依赖它）
- [ ] 1.1 静态审计会拉起 Ray 的测试：grep `ray`、`ray start`、`miles.`、`upstream_parse_args`，列出候选名单与依据写进 design 附录（验证：附录名单含用户给出的四组并标明各自判断）
- [ ] 1.2 在没有上卡链时逐个确认候选：单独运行后检查本机是否出现 raylet/gcs_server 进程，结束后清理（验证：附录逐项记录"拉起 / 未拉起"）
- [ ] 1.3 tests/conftest.py 注册 `ray_local` 标记、`--run-ray-local` 开关、默认取消收集，并加 `ray.init` 拦截夹具（验证：新增 tests/test_conftest_ray_guard.py，用 pytester 断言默认取消收集、开关后收集、未标记 `ray.init` 失败且信息含"ray_local"）
- [ ] 1.4 给确认会拉起 Ray 的测试加 `ray_local` 标记（验证：默认运行这些文件时收集数为 0）

## 2. 本机老失败测试
- [ ] 2.1 test_rl_codex_schema：构造环境补 YETO_CODEX_OPENENV_MODEL_REVISION，使用例走到 "requires backend profile" 分支（验证：单文件运行通过）
- [ ] 2.2 test_rl_ir_harness：报错正则改为匹配现行措辞（验证：单文件运行通过）
- [ ] 2.3 test_rl_dense_full_parameter_sweep 两例：找不到 cargo 时带原因跳过；PATH 含 cargo 时运行通过（验证：两种 PATH 各运行一次）
- [ ] 2.4 test_rl_m1_dense_full_direct_launch：确认用例本意后补齐 v2 证据必填字段或改断言（验证：单文件运行通过）
- [ ] 2.5 test_rl_benchmark：改用临时假文件作为实现输入，或缺 release syncer 时带原因跳过（验证：单文件运行通过）
- [ ] 2.6 test_verda_provider：替身改为接收 num_nodes（验证：单文件运行通过）
- [ ] 2.7 test_export_records_algorithm_like_the_event 与 test_ports_megatron_pythonpath：importorskip 并写原因（验证：缺依赖时显示带原因的跳过）

## 3. CI 转绿
- [ ] 3.1 ci.yml 安装步骤补 pylatexenc、pytest-asyncio（验证：CI python 作业不再报这两个导入错误）
- [ ] 3.2 checkout 加 `fetch-depth: 0`；test_rl_inter_island_launcher 找不到 fd37129e 时带原因跳过（验证：CI 通过；本机浅克隆显示跳过）
- [ ] 3.3 读 CI 日志定位 test_delta_protocol 超时原因并修正（复用 rust 作业构建产物或延长夹具等待），若审计确认它会拉 Ray 则按 1.4 处理（验证：CI 中该文件无 ERROR）
- [ ] 3.4 test_rl_fn_provider_view、test_rl_fn_boot_only：先确认不拉 Ray，本机单跑记录现象，查根因并修（验证：两文件单跑通过，根因写进 design 表）
- [ ] 3.5 tool_wait_workload 缺 generate 及其他"代码与测试不一致"项：按现行接口修正（验证：相关测试通过，改动说明写进 design 表）
- [ ] 3.6 gpu 作业改为手动触发或 `continue-on-error: true`（验证：自托管机器离线时 CI 总体结论不受影响）
- [ ] 3.7 复核 test_rl_integration 的 `--resume` 问题是否已修，未修则修（验证：CI 中该文件通过）
- [ ] 3.8 推分支后看 CI：python 与 rust 作业通过（验证：CI run 链接写进本文件）

## 4. 命令与文档
- [ ] 4.1 新建 docs/TESTING.md，写本机安全测试集一条命令、约定环境、`--run-ray-local` 用法与"上卡链在跑时不要运行"提醒，README 加链接（验证：文档存在且命令可复制运行）
- [ ] 4.2 在没有上卡链时运行一次本机安全测试集，记录通过/跳过数与耗时写进文档（验证：结果无失败与错误，运行前后本机无新 Ray 进程）
- [ ] 4.3 `openspec validate fix-known-red-tests` 通过（验证：命令输出）
