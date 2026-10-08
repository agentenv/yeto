# Tasks：rl-eval-difficulty-buckets

（实现排在 WP7 去耦合阶段 3 之后；开工前用户需先拍板 design.md 的 Open Questions。）

## 1. 评测集构建（离线，不上卡）

- [ ] 1.1 `tools/build_eval_buckets.py`：下载并钉 `zhuzilin/dapo-math-17k@2e656129` 与 `qgallouedec/DAPO-Math-17k-Processed-Scored@b9e6dd45`；规范化题干（去提示前缀、空白）后匹配，输出匹配率
- [ ] 1.2 按 design D2 分 5 桶，统计各桶真实题数；固定种子每桶抽 40 题，写评测集 jsonl（含 `eval_item_id`、`bucket`、`difficulty_source`、`difficulty_value`、`prompt`、`label`）与分桶定义文件，记 sha256
- [ ] 1.3 生成剔除评测题后的训练数据文件，记新 sha256 与剔除题数
- [ ] 1.4 CPU 单测：可复现（同种子同哈希）、无交集、桶边界（r=0、r=1、0.4、0.8）

## 2. 运行配置与启动检查

- [ ] 2.1 `EvalConfig`（`yeto/rl/engine/run_config.py`）加 `set_sha256`、`bucket_def_sha256`、首轮回答数（8）；启动时校验评测集哈希
- [ ] 2.2 启动检查训练数据与评测集交集为空
- [ ] 2.3 第 0 轮评测：驱动在初始发布后已调用 `_maybe_eval(start.rollout_id, force=start.rollout_id == 0)`（driver.py 约 1556 行）；补上第 0 轮每题 8 个回答（其余 4 个）的参数切换，单测覆盖
- [ ] 2.4 采样参数固定检查；单测覆盖"参数被改即停"

## 3. 指标回传与事件

- [ ] 3.1 在 yeto rollout 包装（`yeto/rl/miles.py` 的 `generate_rollout(..., evaluation=True)`）取评测样本，按 `bucket` 算 D5 指标、按题自助重采样算标准误、写逐题 jsonl
- [ ] 3.2 ports 路径 `evaluate`（`miles_adapter/entry.py`）把 3.1 的结果返回驱动；`rl_eval` 事件加 D5 字段（现有字段不变）
- [ ] 3.3 相对第 0 轮的配对差值与标准误
- [ ] 3.4 后备核对：每桶一个 Miles 数据集名时，用 `loss_curve.py` 解析的 `eval/<名>`、`-truncated_ratio` 与 3.1 结果对照，一致性写进测试
- [ ] 3.5 CPU 单测：假样本 → 指标、事件字段、逐题文件哈希

## 4. 训练脚本与复核

- [ ] 4.1 FN 训练脚本加 `--eval-interval 10`、`--n-samples-per-eval-prompt 4`、评测集路径与哈希、训推分离时 `--yeto-rl-overlap-eval`
- [ ] 4.2 复核文档写入评测成本预估（D6，按 WP2 定的布局重算）并台账预登记
- [ ] 4.3 首次真机运行后用 `eval/wall_s` 校正 D6 估计；人工抽查 b0 可疑题（D2）

## 5. 对接与以后

- [ ] 5.1 把 D8 的接口需求交给 WP4（yeto-fleet-dashboard）
- [ ] 5.2 训练批次按难度分桶统计（D4.4，用户已定要做）：构建工具在训练数据行写 `bucket`；`rollout_meta_hook.build_metadata` 里按桶调用 `batch_summary`，写入 `rl_rollout` 事件 `batch_summary_by_bucket`；CPU 单测 + 真机核对开销
- [ ] 5.3 codex/TB2 评测集方案另开 change（D7）
