# 自定义奖励示例

`reward.py` 是一个最小的中立奖励：输入中立轨迹（`yeto.rl.rewards.types.Trajectory`），返回奖励结果（`RewardResult`），不 import 任何训练框架。

- 按名称：`import examples.custom_reward.reward` 后名称 `ends_with_label` 已注册。
- 按引用：`examples.custom_reward.reward:ends_with_label`。
- Miles 下取 `--custom-rm-path` 的值：`yeto.rl.engine.miles_adapter.rewards.miles_custom_rm_path(<名称或引用>)`；名称未注册时报错并列出已注册名称。
- 加载时记录插件身份（模块、函数名、源码文件 sha256，见 `yeto.rl.rewards.registry.load_reward`），改源码身份即变。

测试：`python -m pytest examples/custom_reward/test_reward.py`。
