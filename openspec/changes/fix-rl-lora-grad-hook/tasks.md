# Tasks

## 1. 用不依赖 GPU 的测试把 bug 钉住

- [ ] 1.1 新增 CPU 级测试：构造一个 bf16 `Parameter`，在它的 `AccumulateGrad` 节点上注册 hook，做一次 bf16←fp32 的 `.data` 交换再还原，断言 hook 不再触发且 `main_grad` 保持为零；同一测试中断言 bf16←bf16 的交换 hook 仍触发。验证：该测试在修复前**失败于"期望 hook 触发"**（即先确认它确实复现了 bug），不需要 GPU、不需要 import Miles
- [ ] 1.2 新增测试断言 `_optimizer_masters_as_model_parameters` 进入和退出后，受影响参数的梯度累加器节点身份不变。验证：修复前该测试失败，修复后通过

## 2. 修复集成层

- [ ] 2.1 查明 `bridge.export_adapter_weights` 和 `megatron_bridge_utils.patch_megatron_model` 是否要求真正的 `Parameter`，还是接受普通张量。验证：在 `design.md` 的 D1 下记录结论；若要求 `Parameter`，改走 D1 记录的兜底方案并在此说明
- [ ] 2.2 改掉 `_optimizer_masters_as_model_parameters`（`trainable_state.py:493-507`），不再把 fp32 张量赋给 bf16 `Parameter.data`，改为把 `main_param` 作为张量传给转换路径。验证：任务 1.1 和 1.2 的测试通过
- [ ] 2.3 核对 `_collective_adapter_tensors`（导出）与 `apply_trainable_state`（应用）两个调用点在新写法下行为不变。验证：现有 `tests/` 中覆盖导出/应用的测试全部通过，导出张量的数值与修复前逐元素相同
- [ ] 2.4 评估是否可以直接删掉这个 context manager（写入侧本就直接操作 `main_param`，见 `design.md` Context）。验证：若可删则删除并说明；若不可删，记录仍需它的调用点

## 3. 运行时不变量

- [ ] 3.1 在 round stats 中记录该轮优势是否全为零（或其非零计数），使不变量可以区分退化轮与真实轮。验证：新增单元测试覆盖"全零优势"和"含非零优势"两种输入
- [ ] 3.2 在 `MilesPolicySync` 提交本地状态前加入不变量：优势不全为零且 `grad_norm` 恰好为 `0.0` 时，以 `StrictRlInvariantError` 失败，错误信息含轮次与梯度范数，且不提交本地状态。验证：新增测试覆盖 spec 的三个场景（零梯度+非零优势→失败且未提交、零梯度+全零优势→放行、非零梯度→放行）
- [ ] 3.3 把该失败写入事件 tape，使其在 syncer 侧可见。验证：测试断言 tape 中出现对应事件

## 4. 投放到真实的岛

- [ ] 4.1 重新生成 `yeto/rl/vendor/miles-qwen38.bundle` 使其包含修复，并更新 `yeto/rl/__init__.py` 的 `MILES_BUNDLE_SHA256`。验证：`yeto/launcher.py` 中 setup 的 bundle 校验步骤在本地模拟下通过，且新 bundle 应用后 Miles 工作区确实含修复
- [ ] 4.2 若修复已被上游接受，推进 `MILES_BASE_COMMIT` 并相应缩小 bundle。验证：记录上游 commit；若未被接受则明确标注本任务跳过及原因
- [ ] 4.3 全量 `pytest tests/` 通过，且不需要任何云凭据。验证：与本 change 无关的既有失败需对照改动前基线确认不是新引入的

## 5. 真机确认

- [ ] 5.1 在 Modal 单卡 H100 上跑一次 RL 岛（head/syncer 在 Nebius），确认 `train/grad_norm` 非零、每步都变化，且 syncer tape 的 `gnorm` 在 `1e-2` 量级而不是 `1e-6`。验证：结果记入 `docs/CLOUDS.md`，并在 `live-run-failures.md` 第 23 条下补记修复已验证。注意读 tape 的原始值而不是日志行（日志用 `{gnorm:.4}` 会把 `4e-06` 显示成 `0.0000`）
- [ ] 5.2 确认适配器真的在学：连续几步的原始奖励或 loss 有可观察的变化趋势，而不只是梯度非零。验证：记录逐步数值
- [ ] 5.3 收尾核验——`yeto down <prefix>`、`modal app list` 中自己的 app 为 `stopped` 且 tasks 为 0、Nebius 上无自己前缀的实例残留。验证：三项各自确认，**不要动不属于本次运行的实例**

## 6. 决策点

- [ ] 6.1 与用户确认 `design.md` 的 Open Question：修复后是否要在 8 卡 H100 上重跑任务 8.4，还是接受单卡确认加已记录的基础设施结果。验证：结论记入 `docs/CLOUDS.md`，并据此决定是否勾选 8.4
