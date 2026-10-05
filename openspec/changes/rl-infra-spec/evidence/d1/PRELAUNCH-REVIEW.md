# D1-2/D1-3 上卡前逻辑复核（2026-10-05，代码 = d1-gpu，已 merge integ-decl 6324cd4b）

链脚本：`tests/multinode_gpu/s11d1chain.sh <prefix>`。顺序：T2R1S1 s17（冷启）→ T2R2S0 s17 → T2R1S1 s29 → T2R2S0 s29 → d1e1（KEEP_M3=1）→ T1R3S0 s17 → T1R3S0 s29（风险最高的放链尾）→ trap：杀所有 watchdog、`sky down <本集群>`、cleanup_run.sh 跑两遍、sky status 检查。每次运行之间：先确认集群 UP，再 s1reset.sh（两节点都要求无计算进程且显存 <1.5G，不干净就停止）；每次运行后 kill 它自己的 watchdog（否则会把共享集群 down 掉）；预算守卫为 3.5 h 后不再开新运行。

## (a) 用例参数
- d1sweep：`--keep`；seed 用 `SEED`（`--seed ${SEED:-17}`）；`--total-steps 6`；T2R1S1 = ELASTIC22（rollout 1 + standby 1，PP2）；T2R2S0 = rollout 2，无 standby，PP2；T1R3S0 = rollout 3，PP1，用 resources-2x2-t1r3.json；三个配置都开 `--rl-observe-timeline`。不需要 attestation（不发请求）。
- d1e1：沿用 m3 参数，加 6 个触发器（up@train rid1/5/9，dn@3/7/11；expected_config_epoch 0..5 递增）。
- **发现并已修复的问题 1**：fp_local22 算 attestation 指纹时写死 `--total-steps 6`。实测指纹随 total-steps 变化（6 → 0d17e24b…，14 → ae19cdd6…），因此 d1e1 改了步数后，所有请求都会被 controller 以"no capability attestation"拒绝（与 s8-m1m3a M3 FAIL 同一根因）。已修复：s1run 把 `--total-steps $STEPS --seed $SEED` 传给 fp_local22。核对：在 HEAD 上用 6/17 重算，仍得到 0d17e24b，说明 merge 没有改变指纹。
- **发现并已修复的问题 2**：fp_local22 的 stdout 现在第一行是 launcher 的提示行（`[launcher] nebius/eu-north1: VM image … pre-pulled`），原先的 `| python3 json.load` 会解析失败（exit 67）。已改为 `| tail -1`。
- **发现并已修复的问题 3**：12 步时 dn3 在 rid 11 触发，安全点在 rid 12，此时训练已结束，dn3 拿不到 SUCCEEDED。改为 14 步（dn3 之后还有 rid 12、13 两轮）。
- 复用 m3 的方式：attestation 写入 `$R/attestation-m3.json`；分支条件扩为 `m3 || d1e1`。

## (b) 布局
- 在 launcher 层做了 CPU 干跑：T2R2S0 的 trainer shape 为 (2,1)，T1R3S0 为 (1,1)，都通过校验，bundle map 由 cfg placement 推导。controller 要求 initial_config 必须存在于 resources 中，三个配置都满足。
- T1R3 的运行时风险：rollout 引擎落在 n1:0、n1:1，与 trainer 不在同一节点，需要跨节点发布权重。这条路径已被 T2R2S0（n1:1 引擎）和 M3 的 up 边覆盖过；新增的风险点是 n1 上同时有 2 个引擎，以及 n1 上没有 trainer。回退方案：T1R3 放在链尾，失败就记为无效，不重跑，2.4 只用 T2R1S1 和 T2R2S0 两个配置下结论（与计划一致）。

## (c) 判据与采集
- 2.4 有效性与轮时由 `s1cost.py sweep` 判定：rc0、rl_round_trained 数 = steps、gpu-*.txt 为 L40S 且 4 个 UUID 与 gpu_pool 一致、无 RECOVERY_REQUIRED。每轮时长 = 该 rid 的 generate+train+outer_sync 加上随后一次 publish，取 rid≥1 的中位数。用 M3 旧数据验证可跑通：中位数 13.5 s。
- 5.1：`s1cost.py <run>` 按请求输出 wait_safe/drain/init/verify/resume/阻塞合计/首轮额外。所需字段都在 journal 与 rl-island tape 中。
- 采集窗口 ≥ 采集延迟：puller 每 10 s 拉取一次，launcher 退出后还有一次终拉（rc.txt 延迟 20 s 写出）。M3 旧证据里的 journal 含 finalization 记录，说明终拉可以拿全。触发器是容器内 0.5 s 轮询，相对约 13 s 的轮时足够。终态探针（s1probe）在容器内运行。
- 1.7 load sample：M3 在 `--rl-observe-timeline` 下产生了 rl_load_sample；本链所有用例都开了该开关。

## (d) 代码逻辑（HEAD 6324cd4b）
- 8036c769 到 6324cd4b 的变更：auto.py、recommend.py、controller（recommend_mode 默认 disabled，只新增 inbox 的 `mode` 动词）、timeline 小改。E1 的 request/plan/fork 路径没有行为变化，指纹也不变（见 a）。
- s1reset 会清掉 ~/yeto-output 与 elastic-state，上一次运行的 tape 和 inbox 不会误触发 s1inwatch。

## (e) 时长与费用
- 历史参照：M1 冷启 18 min；暖启的 M3 用 8 min。本链共 7 次运行：18 + 5×8 + 10（d1e1）+ 清理 ≈ 75 min，约 $12。
- 硬顶：每次 HARD 3000 s；预算守卫 3.5 h（$32），低于预登记上限 $37（含重跑 ≤$57）。
