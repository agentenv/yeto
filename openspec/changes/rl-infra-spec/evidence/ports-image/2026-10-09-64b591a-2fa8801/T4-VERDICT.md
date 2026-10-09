# S19 #1 T4 parse_args 检查判读
- 镜像 ghcr.io/michaellchung/yeto-miles-ports:64b591a-2fa8801@sha256:fa2413be4c4fcf066437f946365f01392d6f884002f8fa19c770c12c08f2acf0
- try6（10-09）：卡名 Tesla T4；manifest miles.commit 与 /root/miles HEAD 都是 64b591a4bec1ffa37fb089d3e3b99c84773b1ffc。
- pytest 三组 25 passed（translation [True]/[False]、critic 家族 sao/vapo/compactionrl、test_rl_algorithm_flags_upstream.py 20 项）。
- 结论：通过。
- try5 查因：sao/compactionrl 失败是测试缺陷。`--critic-updates-per-step` 是 `--num-critic-epochs` 的别名，dest 为 num_critic_epochs；测试按旗标名查 hasattr。已改为别名映射并断言 num_critic_epochs == 2。
