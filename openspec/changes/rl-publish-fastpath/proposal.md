# Proposal

## Why

S16 的 FN 2×8 训推分离早门 `s16-rawlora-fn2x8-long-20261008a`（Modal 2×8 H200，单岛两节点，训练节点 TP2 PP4 EP2、推理节点 SGLang TP8）5 轮跑满，但稳态每轮约 660 s 里**约 400 s（60%）花在 yeto 驱动进程自己身上**，不是训练、不是生成，也不是跨节点发送（Miles 真正的发布每轮 25–30 s，其中 NCCL 发送约 8 s）。读码 + 实测时间线（infra-drafts/FN2X8-MODAL-PRELAUNCH-REVIEW.md §9.5）定位到两处：

1. **训练后同步（`outer_sync`，每轮 175–216 s）**：单岛没有岛间同步，`LocalOnlySync.boundary` 仍调用 `driver.export_local()` → `MilesPolicyState.export`，经 Ray 把整份 LoRA 状态（float32，11.2 GB）从训练进程搬到驱动进程，再在驱动上规范化、复制一份（`_at_version` 用 `canonical_state` 拷贝）。
2. **发布前（`publish` 里 Miles 更新权重之前约 213 s）**：驱动先对 11.2 GB 算一次 `policy_tensor_hash`；`MilesPublisher.publish` 又**再导出一次整份状态**算哈希，确认"训练进程手里的权重就是要发布的那份"；再算 `payload_digest`（第三遍）；manifest 哈希本身很小。

而发布本身走 Miles `update_weights`，权重直接从训练进程发给推理引擎，**驱动从头到尾不需要这 11.2 GB 张量**，只需要两个哈希值。

## What Changes

- **哈希在训练进程里算**：新增训练进程插件 `export_digest`，在训练主 rank 上照旧导出（同一个 `export_state`，含 NaN/Inf 检查和梯度流检查），然后在**同一次调用里**用两个线程并行算出 `policy_tensor_hash` 和 `payload_digest`（两个哈希的定义一个字节都不改），只把哈希值、字节数和（名字, 形状）清单经 Ray 带回驱动。
- **驱动侧用"留在训练进程里的策略"句柄**：新增 `TrainerResidentState`，带版本号、各身份字段和上面的哈希，不带张量。单岛无同步时 `LocalOnlySync.start/boundary` 返回它；版本号改写不再复制张量。
- **发布复用这次导出的结果**：`MilesPublisher.publish/publish_members` 遇到这种句柄时，`payload_hash/payload_bytes` 直接取训练进程算好的值；"训练进程手里的权重就是要发布的那份"这项检查**保留**，改为让训练进程就地重新导出+哈希、只回传哈希（不再搬 11.2 GB）。
- **兜底**：凡是确实要张量的地方（`.tensors`、`to_lora()`、`policy_hash()`）自动触发一次完整导出，并核对哈希一致；后端没有 `export_digest`（测试用假对象、以后的 verl 等）或设了 `YETO_RL_PUBLISH_FASTPATH=0` 时走原路径。
- **不改**：tape 事件与字段（`rl/policy_token`、`sync/publication_payload_hash`、`sync/publication_payload_bytes`、manifest 各字段）的值和语义；Miles 侧 WeightChecker 校验和与 `[LORA-CHECK]`；多岛同步（strict-avg 等仍需要张量，走原路径）。
- 只加日志（不加 tape 字段）：每次导出打印总耗时、训练进程导出耗时、哈希耗时，供下次真机分解每轮时间。

## Capabilities

### New Capabilities
- `rl-publish-fastpath`：单岛无岛间同步时，策略身份（张量哈希、发布载荷哈希）在训练进程内计算，驱动只接收哈希；发布与"训练端仍持有该策略"的检查语义不变。

### Modified Capabilities
<!-- 无：不改既有规格的需求，只换计算位置。 -->

## Impact

- `yeto/rl/engine/policy_digest.py`（新）：`digest_canonical_tensors`、`PolicyDigest`、`TrainerResidentState`。
- `yeto/rl/engine/miles_adapter/state_plugin.py`：新插件 `export_digest`（`EXPORT_DIGEST`）。
- `yeto/rl/engine/miles_adapter/state.py`：`MilesPolicyState.export_digest / _digest_result / _current_digest`。
- `yeto/rl/engine/miles_adapter/publish.py`：`MilesPublisher._check_trainer_holds`、`_payload_of`，`publish` 与 `publish_members` 改用它们。
- `yeto/rl/engine/bridges.py`：`_local_state`，`LocalOnlySync.start/boundary` 改用它。
- `yeto/rl/engine/driver.py`：`IslandDriver.export_local_resident`（含开关 `YETO_RL_PUBLISH_FASTPATH`）。
- 测试：新 `tests/test_rl_publish_fastpath.py`；`tests/test_rl_engine_selection.py` 的假训练组加 `export_digest` 分支、最终策略在关闭循环前取张量；`tests/test_rl_fn_layout.py` 源码断言改为认 `_local_state`。
- 与去耦合（yeto-framework-decoupling）的关系：本分支叠在 `s16-decouple-p2` 上；阶段 4 把 `miles_adapter/` 搬到 `adapters/miles/` 时由去耦合分支 rebase 过来。`policy_digest.py` 不依赖 Miles，放在 `engine/`（中立层）。
- 镜像：改动都在 yeto 包里，训练进程插件也是 yeto 代码，随 yeto 源码进容器；**不需要重建 Miles 镜像**（待真机确认 yeto 源码确实按现有方式进入训练进程）。
