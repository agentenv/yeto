# Proposal

## Why

走 Yeto 的 Miles policy-sync 集成层的 LoRA RL 训练**全程是空操作**：梯度正常算出来了，但从来没进入 Megatron DDP 的梯度缓冲区，于是每一次优化器步都没有改动适配器。一次完整的 8 卡 H100 MATH-500 运行（`yeto-rl84f`，任务 8.4）"成功"结束——3 轮同步、真实 rollout、平均原始奖励 0.609——却什么都没学到。

这个 bug 的危险之处在于它**完全静默**：运行退出码为 0，syncer 还会提交一个非零的 delta（实际是 fp32 往返噪声），所以下游没有任何环节会报警。唯一的征兆 `train/grad_norm: 0.0` 被连续两个 session 误判过——先是被当成显示假象，后是被当成 GRPO 优势退化。按 8 卡 H100 约 35 美元/小时计，这类静默空转很贵。

## What Changes

- **去掉跨 dtype 的 `.data` 交换**。`_optimizer_masters_as_model_parameters` 把 fp32 张量赋给 bf16 `Parameter.data`；PyTorch 在 dtype 不一致时会重置参数的梯度累加器，销毁 Megatron DDP 注册在上面的 hook，导致 `main_grad` 永远为零。改为把 `main_param` 作为普通张量传给 Bridge 转换函数，不再别名到 `Parameter` 上。
- **新增训练进展不变量**。当一轮的优势不全为零、而报告的 `grad_norm` 恰好为 `0.0` 时，该轮**失败**，而不是记录一条日志后照常提交。
- **推进 Miles 的 pin**。岛的 setup 从 GitHub 按固定 commit clone Miles，因此只改本地检出对任何真实运行都无效；修复必须通过 vendored bundle（或 base commit）投放。
- **补不依赖 GPU 的回归测试**：一个 CPU 级测试断言跨 dtype 的 `.data` 交换会丢掉已注册的累加器 hook；另一个断言优势非零时不变量会拒绝零 `grad_norm`。

## Capabilities

### New Capabilities
- `rl-lora-gradient-flow`：Miles RL 岛本地训练产生的梯度必须真正到达优化器并改动适配器；policy-sync 集成层不得扰动 autograd 状态；一轮在非退化优势上训练却没有产生梯度时，必须显式失败而不是提交噪声。

### Modified Capabilities
<!-- 无。`openspec/specs/` 当前为空：在途的 add-nebius-verda-modal-clouds
     尚未归档或同步，因此不存在被本 change 改动需求的既有能力。 -->

## Impact

- **`miles/backends/megatron_utils/trainable_state.py`**（第 493-507 行的 `_optimizer_masters_as_model_parameters`，以及它的两个调用点 `_collective_adapter_tensors` 和 `apply_trainable_state`）。该文件物理上住在 Miles 仓库，但整个是 **Yeto 的 external policy-sync 集成层**——满篇由 `yeto_rl_*` 参数驱动，纯 Miles 的 LoRA 路径根本不经过它，这正是 Miles 自己的 e2e 测试测不出来的原因。
- **Miles 的 pin**：`yeto/rl/__init__.py` 的 `MILES_BASE_COMMIT`、`MILES_BUNDLE_PATH`、`MILES_BUNDLE_SHA256`，以及 `yeto/launcher.py` 的 `make_miles_island_task`。改了源码不动 pin，运行时毫无变化。
- **`yeto/rl/miles.py` / `yeto/rl/bridge.py`**：不变量及其读取的 round stats。
- **SFT 岛不受影响**，它不走这条路径。

## Non-goals

- 不改动 GRPO、奖励函数或采样配置。rollout 和优势已验证健康（224/224 个参数带梯度，`lora_grad_l2 = 2.79e-02`），坏的只是梯度到优化器这一段传递。
- 不在本 change 内重跑任务 8.4。修复后是否重新验收 8.4 是独立决定；8.4 的基础设施要求（2 轮同步、WAN 耗时、`--pipeline` 结论）已经达标并记录在 `docs/CLOUDS.md`。
