# F-E2 冒烟（4.4 rebuild-trainer，不计入 task）结果，2026-09-30

计划 §9.8。代码 ec249ef（含 3f88c1d），镜像 db815884…；Modal 3×L40S，`--modal-retries 0 --modal-timeout-s 2700`，进度看门狗 20 min。
本地 dry-run：`test_fe2_dryrun.py` 1 passed；launcher `--dry-run` rc=0。

结果：**未得到观察项（环境：岛容器始终未启动）**。07:06:54 app 部署、launcher "launching learner 0" 之后，`modal container list` 20 分钟内没有任何属于该 app 的容器，岛无任何日志；进度看门狗于 07:26:52 按规则 `modal app stop`。launcher 随后记录 job FAILED，并尝试 relaunch（app 已停，查找失败）——注意：launcher 自身的 recovery 逻辑在 `--modal-retries 0` 下仍会尝试重新拉起一次（本次因 app 已停而失败，无费用）。
原因（随后 F-E1 指纹运行 `modal app logs` 证实同一时段）：Modal "waiting to be scheduled on a GPU_L40S worker … acquiring more capacity"，即 L40S 无容量（同一 digest 在 06:45 的 fe1r 中已在 L40S 上拉起过，镜像缓存不是原因）；未能从 Modal 侧取得排队原因。rebuild 事务、`rl_trainer_rebuilt`、cut 落盘均未观察到。
费用：未见运行容器，按 Modal 计费应为 ≈$0；上界按墙钟 3×1.95×20/60 = $1.95 记入台账。app ap-u2WJPfFinX9pUs6UhkT9Md stopped/0。
