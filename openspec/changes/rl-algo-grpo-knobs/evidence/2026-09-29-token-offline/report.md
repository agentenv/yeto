# token 聚合疑似缺陷：离线排查报告（ALGO-1b，未使用 GPU）

## 现象（g1c）
第 1 步输入相同（raw_reward 都是 0.8125，response 长度 74–384 不等），而 token 与 baseline 的第 1 步 grad_norm 逐位相同（0.6349465847015381），此后各步也逐位相同。第一次 G1 中第 2、3 轮的 rollout 也完全一样，说明权重更新完全一样。

## 链路追踪（miles-next 0394715）
1. 参数：`--calculate-per-token-loss` 由 yeto 翻译产生，miles.log 的参数表中 `calculate_per_token_loss = True`，说明参数确实到达了 Miles。
2. Miles 的损失（`miles/backends/training_utils/loss.py`）：
   - `:167-175` `get_sum_of_sample_mean(..., args.calculate_per_token_loss, ...)` 在 per-token 模式下改用 `sum_of_token`（`cp_utils.py:122-128`）；
   - `:203-220` 在 per-token 模式下不再乘 `num_microbatches/global_batch`，返回 `(loss, num_tokens=Σclamp(mask.sum,1), ...)`；
   - 日志中的 train/pg_loss 由 1e-8 变为 0.035，说明这一段在训练 actor 中确实执行了。
3. Megatron 的模型配置：
   - 非 LoRA 的 bridge 路径中，`model_provider.py:49` `_apply_bridge_runtime_config` 会设置 `provider.calculate_per_token_loss = args.calculate_per_token_loss`；
   - **LoRA actor 走的是另一条路**：`model.py:150-155` 在 `is_lora_enabled and role == "actor" and megatron_to_hf_mode == "bridge"` 时调用 `lora/bridge.py:123 _setup_lora_model_via_bridge`。该函数（`:145-190`）逐项拷贝了 TP/PP/EP/SP/CP、recompute、fusion、moe 等 provider 字段，**但没有拷贝 `calculate_per_token_loss`**，并且不调用 `_apply_bridge_runtime_config`。因此 LoRA 路径下 Megatron 的 `TransformerConfig.calculate_per_token_loss` 保持默认值 False。**这是一个确认存在的不一致，位于 Miles（`lora/bridge.py`），计入 fork 待办。**
4. Megatron 的 schedule：config 为 False 且收到 3 元组时，会先除以 num_tokens 再除以 num_microbatches。这一段根据 Megatron 源码的行为转写，本地环境没有 megatron.core 的源码，**未在本机核实**。

## CPU 最小验证（`repro_token_grad.py`，输出见 `repro_output.txt`）
- 直接调用 Miles 的 `get_sum_of_sample_mean`，并把 loss.py 与 Megatron 的两处缩放转写进来，比较两种聚合对同一批 token 的梯度。
- 响应长度不等（3/7/5/9）时，grad_norm 为 0.1297 对 0.0962，`bitwise equal: False`；长度全相等时两者一致，这是对照组。
- 所以按 Miles 加 Megatron 的数学，**在 g1c 的配置下（长度 74–384 不等），两种聚合不应得到相同梯度**。"此配置下不可区分"这个解释不成立。

## 结论
- **缺陷可以确认的部分**：Miles 的 LoRA bridge 路径（`lora/bridge.py:_setup_lora_model_via_bridge`）没有把 `calculate_per_token_loss` 设到 Megatron 的 config 上，属于 Miles 问题（fork 待办）；非 LoRA 的 bridge 路径设置正确。
- **尚未解释的部分**：只有 config 缺失这一条，按上面的推导梯度仍然会变化（Megatron 会改为除以 num_tokens），**不足以解释 GPU 上的逐位相同**。说明训练 actor 中实际参与反向的 loss 很可能没有走 per-token 分支，或者在别处被归一化成了同一个值。原因未确认。
- yeto 这一侧：翻译和参数传递都正确（参数表中为 True），目前没有发现 yeto 的缺陷。
- 处理：token 聚合不声明。定位还需要一个 GPU 探针：在训练 actor 中记录 `loss_function` 返回的 (loss, num_tokens)、`model.config.calculate_per_token_loss`，以及 backward 之前的标量 loss；token 与 baseline 各跑 1 步。该实验需另立计划并单独提交。
