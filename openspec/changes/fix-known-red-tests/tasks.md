# Tasks

## 1. Ray 测试隔离（先做，后面的本机运行依赖它）
- [x] 1.1 静态审计会拉起 Ray 的测试：grep `ray`、`ray start`、`miles.`、`upstream_parse_args`，列出候选名单与依据写进 design 附录（验证：附录名单含用户给出的四组并标明各自判断）
  - 证据：名单与依据见 design.md 附录 A。
- [x] 1.2 在没有上卡链时逐个确认候选：单独运行后检查本机是否出现 raylet/gcs_server 进程，结束后清理（验证：附录逐项记录"拉起 / 未拉起"）
  - 改法说明：规则禁止本机运行会拉 Ray 的测试，所以没有逐个单独运行候选。改用守卫确认：全量默认运行中守卫拦截 ray.init，后台每 3 秒检查 raylet/gcs_server，两次全量运行都没有出现。结果见附录 A。
- [x] 1.3 tests/conftest.py 注册 `ray_local` 标记、`--run-ray-local` 开关、默认取消收集，并加 `ray.init` 拦截夹具（验证：新增 tests/test_conftest_ray_guard.py，用 pytester 断言默认取消收集、开关后收集、未标记 `ray.init` 失败且信息含"ray_local"）
  - 证据：tests/test_conftest_ray_guard.py 4 passed（默认取消收集、开关后收集、未标记 ray.init 失败含 ray_local、守卫自身不导入 ray 且能拦测试内的晚导入）。守卫不导入 ray，是因为 test_decoupling_golden 断言 ray 不在 sys.modules。
- [x] 1.4 给确认会拉起 Ray 的测试加 `ray_local` 标记（验证：默认运行这些文件时收集数为 0）
  - 证据：三个原名单文件默认运行 48 passed / 1 skipped / 19 deselected，无 raylet。全量运行时守卫又拦下 3 例（test_rl_grpo_knobs 两例、test_rl_eval_batch_buckets 一例，build_metadata→current_policy_token 自动启动 Ray），已补标记。

## 2. 本机老失败测试
- [x] 2.1 test_rl_codex_schema：构造环境补 YETO_CODEX_OPENENV_MODEL_REVISION，使用例走到 "requires backend profile" 分支（验证：单文件运行通过）
  - 根因：S17 M1 把 qwen35 加进 OPENENV_BACKEND_PROFILES，测试仍用 qwen35 当"不在名单的 profile"。处理：改测试，换成 not_an_openenv_profile。代码行为正确。证据：单文件 7 passed。
- [x] 2.2 test_rl_ir_harness：报错正则改为匹配现行措辞（验证：单文件运行通过）
  - 根因：报错改为多个 profile 的复数措辞（"fix the append roles"），测试正则没跟上。处理：改测试正则。证据：单文件 26 passed。
- [x] 2.3 test_rl_dense_full_parameter_sweep 两例：找不到 cargo 时带原因跳过；PATH 含 cargo 时运行通过（验证：两种 PATH 各运行一次）
  - 根因有两个。① PATH 无 cargo：加 skipif，原因写明。② PATH 有 cargo 后仍失败：syncer 08-25（1c1098d3）规定 sweep 必须带 --resume，08-27（d4c72cd7）规定 --resume 时检查点必须已存在，两条叠加使全新 sweep 无法启动。处理：主 agent 代用户拍板选 A，改代码 syncer/src/server.rs，sweep 校验不再要求 --resume，--resume 文件检查抽成 check_resume_checkpoint；测试首次启动不带 --resume，重启带。新增 Rust 单测 policy_sweep_first_start_needs_no_resume_and_resume_needs_a_file。证据：无 cargo 时 4 passed / 2 skipped；有 cargo 时 6 passed；cargo test policy_sweep 11 passed。
- [x] 2.4 test_rl_m1_dense_full_direct_launch：确认用例本意后补齐 v2 证据必填字段或改断言（验证：单文件运行通过）
  - 根因：v2 证据新增 active_token_count、loss_mask_hash、active_token_ids_hash 必填，夹具按旧格式构造。用例本意是合法夹具，所以补字段并用 v2 批哈希。证据：单文件 5 passed。
- [x] 2.5 test_rl_benchmark：改用临时假文件作为实现输入，或缺 release syncer 时带原因跳过（验证：单文件运行通过）
  - 根因：身份指纹要读 release syncer 二进制。用例只测 JSON 往返，改为替换 implementation_fingerprint（同文件下一个用例已用此法）。证据：单文件 52 passed。
- [x] 2.6 test_verda_provider：替身改为接收 num_nodes（验证：单文件运行通过）
  - 根因：_down_and_verify 增加 num_nodes 参数，替身只收一个参数。处理：改替身。证据：单文件 63 passed / 1 skipped。
- [ ] 2.7 test_export_records_algorithm_like_the_event 与 test_ports_megatron_pythonpath：importorskip 并写原因（验证：缺依赖时显示带原因的跳过）
  - accelerate：test_export_records_algorithm_like_the_event 加 importorskip 并写原因，显示为带原因的跳过。megatron：根因不是缺 megatron.post_training，是 miles-next-venv 装了一个普通包 megatron 0.5.1，遮住了测试自建的假镜像目录。处理：测试子进程加 -S，不读 venv 的 site-packages，测试照常运行。证据：test_ports_megatron_pythonpath 3 passed。

- [x] 2.8 全量运行中新发现的失败与 SLIM 报告的 10 个失败：逐个查根因并处理（验证：附录 B 每项有根因与处理，本机安全测试集无失败）
  - 证据：design.md 附录 B。SLIM 的 10 个里 1 个是 2.4，另 9 个是 async 测试缺 pytest-asyncio，选择装依赖不改测试。
- [x] 2.9 主 agent 代用户拍板选 B：新建 /home/michael/work/yeto-test-venv（uv，CPU torch 2.8.0），pyproject 加 test 组（skypilot 0.13.0、boto3 1.43.104、peft 0.21.0、accelerate 1.15.0、pylatexenc 2.10、pillow、pytest-asyncio）。不碰共享 venv（验证：venv 内可收集全部测试且无收集错误）
  - 证据：--collect-only 5086/5109 collected，23 deselected，0 error。

## 3. CI 转绿（不做：PM 决定不管 CI（10-09 用户））
- [ ] 不做：PM 决定不管 CI（10-09 用户）。原内容：3.1 ci.yml 安装步骤补 pylatexenc、pytest-asyncio（验证：CI python 作业不再报这两个导入错误）
- [ ] 不做：PM 决定不管 CI（10-09 用户）。原内容：3.2 checkout 加 `fetch-depth: 0`；test_rl_inter_island_launcher 找不到 fd37129e 时带原因跳过（验证：CI 通过；本机浅克隆显示跳过）
- [ ] 不做：PM 决定不管 CI（10-09 用户）。原内容：3.3 读 CI 日志定位 test_delta_protocol 超时原因并修正（复用 rust 作业构建产物或延长夹具等待），若审计确认它会拉 Ray 则按 1.4 处理（验证：CI 中该文件无 ERROR）
- [ ] 不做：PM 决定不管 CI（10-09 用户）。原内容：3.4 test_rl_fn_provider_view、test_rl_fn_boot_only：先确认不拉 Ray，本机单跑记录现象，查根因并修（验证：两文件单跑通过，根因写进 design 表）
- [ ] 不做：PM 决定不管 CI（10-09 用户）。原内容：3.5 tool_wait_workload 缺 generate 及其他"代码与测试不一致"项：按现行接口修正（验证：相关测试通过，改动说明写进 design 表）
- [ ] 不做：PM 决定不管 CI（10-09 用户）。原内容：3.6 gpu 作业改为手动触发或 `continue-on-error: true`（验证：自托管机器离线时 CI 总体结论不受影响）
- [ ] 不做：PM 决定不管 CI（10-09 用户）。原内容：3.7 复核 test_rl_integration 的 `--resume` 问题是否已修，未修则修（验证：CI 中该文件通过）
- [ ] 不做：PM 决定不管 CI（10-09 用户）。原内容：3.8 推分支后看 CI：python 与 rust 作业通过（验证：CI run 链接写进本文件）

## 4. 命令与文档
- [x] 4.1 新建 docs/TESTING.md，写本机安全测试集一条命令、约定环境、`--run-ray-local` 用法与"上卡链在跑时不要运行"提醒，README 加链接（验证：文档存在且命令可复制运行）
  - 证据：docs/TESTING.md 已写创建 venv 与运行命令。README "Testing and CI" 节加了指向它的一行。4.2 的运行就用文档里这条命令。
- [x] 4.2 在没有上卡链时运行一次本机安全测试集，记录通过/跳过数与耗时写进文档（验证：结果无失败与错误，运行前后本机无新 Ray 进程）
  - 证据：10-09 在 yeto-test-venv 运行：5055 passed、45 skipped、23 deselected、0 failed，413 s。运行后无 raylet/gcs_server。另 cargo test（syncer）149 passed。日志 /tmp/fkrt-full5.log（未提交）。
- [x] 4.3 `openspec validate fix-known-red-tests` 通过（验证：命令输出）
  - 证据：Change 'fix-known-red-tests' is valid。
