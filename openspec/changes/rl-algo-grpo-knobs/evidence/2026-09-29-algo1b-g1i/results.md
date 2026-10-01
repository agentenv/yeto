# algo1b-g1i 结果（按预登记判据；sandbox 已终止，app 已 stopped）

- 版本：YETO_SHA 见 `harness/YETO_SHA`，Miles 0af62f4d（sbx.py）。本次走 sandbox harness，没有 run_manifest。
- 两个臂均 rc=0、3 轮完成，无 invariant 错误，零梯度不变量没有误报。第 3 轮 of_on 有 31/32 条样本被过滤，没有失败。
- 配对有效：第 1 步 raw_reward 都是 0.65625，truncated_ratio 都是 0.5（存在截断）。
- (a) of_on 各轮的 `rl_round_trained.filtered_samples` 为 16/16/31，都大于 0 且不超过 32，满足。
- (b) of_off 各轮为 None，满足。
- (c) 第 1 步 grad_norm：of_on 为 0.5646753311157227，of_off 为 0.6329281330108643，不相等，满足。
- **结论：overlong_filter 在 GPU 上证明生效，可以声明 `features:overlong_filter`。**要求代码包含 1b-hook（integ-decl 21912fe 及以后）。
