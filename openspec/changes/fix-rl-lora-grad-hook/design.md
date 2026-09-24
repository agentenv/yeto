# Design

## Context

动机见 `proposal.md` 的 Why。决定方案的机制细节：

`_optimizer_masters_as_model_parameters`
（`miles/backends/megatron_utils/trainable_state.py:493-507`）会把每个适配器
`Parameter` 的存储临时别名到它的 fp32 优化器 master 上，好让读取
`parameter.data` 的 Megatron-Bridge 转换器看到 fp32 数值：

```python
originals.append((parameter, parameter.data))
parameter.data = parameter.main_param.view(parameter.shape)
```

适配器参数是 bf16，`main_param` 是 fp32。PyTorch 的 `VariableHooks::set_data`
在 `!new_data.options().type_equal(self.options())` 时会调用
`autograd_meta->grad_accumulator_.reset()`，销毁 `AccumulateGrad` 节点，
连带销毁 Megatron DDP 在 `__init__` 时注册在该节点上的 hook。退出时的还原
是同样的跨 dtype 赋值，不会重新注册任何东西；下一次反向会新建一个没有 hook
的累加器。于是梯度落在 `param.grad` 上，永远进不了 `main_grad`，而
`_copy_model_grads_to_main_grads` 读的正是 `main_grad`——fp32 master 梯度为零，
`grad_norm` 精确为 `0.0`，优化器步是彻底的空操作。

该 context manager 包裹了两个调用点——`_collective_adapter_tensors`（导出）和
`apply_trainable_state`——两者都在第一个训练步之前的
`MilesPolicySync._initialize` 中跑过。所以第 0 步时 hook 就已经没了，且每轮
重新销毁一次。

单卡 H100 复现（`gnz1`）确认：224/224 个适配器参数 `requires_grad=True` 且
`param.grad` 非零（`lora_grad_l2 = 2.79e-02`），224 个的 `main_grad` 全为零，
`grad_added_to_main_grad=False`，`step_returned=(True, 0.0, None)`，步后所有
`delta` 恰好为 `+0.0`。本机 CPU 上的最小复现把触发条件锁定在 dtype 变化：
bf16↔fp32 的 `.data` 交换会丢 hook，bf16↔bf16 的不会。

投放上有两条约束：

- 岛的 setup 从 `MILES_REPOSITORY` 按 `MILES_BASE_COMMIT` clone Miles，再应用
  vendored 的 `yeto/rl/vendor/miles-qwen38.bundle`（带 SHA256 校验）。**改本地
  Miles 检出对运行时毫无影响。**
- 集成层的**写入侧**（`side.param_weight.main_param.view(...).copy_(target)`）
  本来就直接操作 `main_param`，从不需要这个别名。只有读取/转换侧需要。

## Goals / Non-Goals

**Goals:**

- 每一轮（包括第一轮）梯度都能到达 `main_grad`。
- 把失败模式变响：一轮不可能在真实优势上训练却静默地什么都不产出。
- 回归在没有 GPU 的情况下也能被抓住，从而能一直被抓住。

**Non-Goals:**

- 不审计 Megatron-Bridge 转换器中本次修复之外的其他 `parameter.data` 假设。
- 不改动 SFT 路径，它不走这条集成层。
- 不在本 change 内端到端重新验收任务 8.4。

## Decisions

### D1：移除别名，而不是重新注册 hook

把 fp32 张量显式传给转换器，不再赋给 `Parameter.data`。转换器需要的是一个
**张量**而不是 `Parameter`；这个别名的存在只是为了迎合读取 `parameter.data`
的调用签名。

*考虑过的替代方案——还原后重新注册 DDP hook*：通过
`param.expand_as(param).grad_fn.next_functions[0][0]` 拿到新的累加器，再调
`ddp._make_param_hook(...)`。不作为首选：依赖 Megatron 私有 API，存在一个
hook 确实缺失的时间窗，而且保留了一个其 docstring 已经（错误地）宣称安全的
别名——"Expose f32 optimizer masters to Bridge conversion without replacing
Parameters"。仅在某个转换器确实需要真正的 `Parameter` 时作为记录在案的兜底。

*考虑过的替代方案——把 `main_param` 改成 bf16*：改变优化器数值行为。否决。

### D2：不变量放在 Yeto 侧，不放在 Miles 里

Yeto 在组装 `LocalRoundStats` 时已经拿到了 `grad_norm` 和该轮的优势统计，并且
Yeto 拥有严格不变量机制（`StrictRlInvariantError`）和事件 tape。放在这里可以
让检查留在测试会 gate 本 change 的那个仓库里，也使它**不依赖岛跑的是哪个 Miles
版本**——从而顺带防住将来某个岛被误 pin 到修复前的 commit。

判据必须是"优势不全为零 **且** `grad_norm == 0.0`"，而不能只看
`grad_norm == 0.0`：一个完全退化的 GRPO 轮（每组样本奖励都相同）本就合法地
不产生梯度，而当前配置并不过滤这类组
（`grpo_filter_groups_with_same_reward=False`）。用与零的精确相等而不是小阈值
是刻意的——观测到的故障恰好是 `0.0`，而真实的一小步在 `1e-2` 量级；阈值需要
调参，而且有可能掩盖这个 bug 的下一个变种。

### D3：通过 vendored bundle 投放，修复被上游接受后再推进 base commit

bundle 本来就是承载 Yeto 专属 Miles 补丁的机制，已在 `yeto/rl/__init__.py` 里
按 SHA256 固定，而且**不需要等上游接受**就能让岛跑对。因此投放路径是重新生成
bundle 并更新 `MILES_BUNDLE_SHA256`。如果修复也被上游接受，再推进
`MILES_BASE_COMMIT` 并相应缩小 bundle。

*考虑过的替代方案——等上游*：在此期间每个 RL 岛都在静默损坏。否决。

### D4：直接测 PyTorch 的行为，而不只测集成层

根因是一个 Yeto 和 Miles 都没有断言过的 PyTorch 语义（`set_data` 在 dtype
变化时重置梯度累加器）。一个 CPU 级测试——构造带已注册累加器 hook 的 bf16
参数，做跨 dtype `.data` 交换，断言 hook 不再触发——把这个陷阱记录下来，并在
将来某次重构重新引入它时失败。它在普通测试套件里跑，不需要 GPU，也不需要
import Miles。

## Risks / Trade-offs

- **转换器可能要求真正的 `Parameter` 而非张量** → 移除别名前先对照
  `bridge.export_adapter_weights` 和 `megatron_bridge_utils.patch_megatron_model`
  验证。D1 的兜底方案覆盖这种情况。
- **不变量可能在合法的退化轮上误报** → 已用"优势非零"作为前置条件，且退化
  情形在 spec 里有专门场景覆盖。
- **过期的 bundle 会无声地把 bug 带回来** → bundle 在 setup 阶段有 SHA256
  校验；不变量是运行时兜底，且不依赖 pin 是否正确。
- **没有 GPU 无法完全验证修复** → CPU 测试覆盖机制和不变量；端到端确认需要
  一次开岛。单卡 H100 复现约 1.5 美元且能精确重现该签名，所以完整验证不需要
  8 张卡。
- **在修复进入上游之前与上游 Miles 分叉** → 接受；bundle 本来就带着 Yeto 补丁。

## Migration Plan

1. 修好集成层并重新生成 vendored bundle；更新 `MILES_BUNDLE_SHA256`（若修复
   已进入上游，同时推进 `MILES_BASE_COMMIT`）。
2. 落地不变量和 CPU 测试；这两项自身即可 gate 本 change。
3. 在单卡 H100 岛上确认 `train/grad_norm` 非零、syncer 的 `gnorm` 在 `1e-2`
   量级而不是 `1e-6`。

回滚：撤回 bundle SHA 和 pin。**不变量可以保留**——在回滚后的岛上它会把静默
空转变成立刻的显式失败，无论如何都是更安全的状态。

## Open Questions

- 修复后是否要在 8 卡 H100 上重跑任务 8.4，还是接受单卡确认加上已记录的基础
  设施结果。这不影响 specs、方案和任务拆分。
