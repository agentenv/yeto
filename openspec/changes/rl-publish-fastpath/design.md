# Design

## 背景：一轮时间花在哪（实测）

数据来源：`s1-runs/s16-rawlora-fn2x8-long-20261008a/metrics.json` 的阶段耗时（训练节点资源采样里的 span），以及 FN2X8-MODAL-PRELAUNCH-REVIEW.md §9.5 的时间线。

| 轮 | 生成 s | 训练 s | 训练后同步 outer_sync s | 发布 publish s | 其中 Miles update_weights s |
|---|---|---|---|---|---|
| 0 | 171.4 | 615.5（含首轮编译） | 215.7 | 242.6 | 29.7 |
| 1 | 151.0 | 143.0 | 182.1 | 235.7 | 27.7 |
| 2 | 151.3 | 85.4 | 178.7 | 241.8 | 25.2 |
| 3 | 158.3 | 91.7 | 175.3 | 237.5 | 24.7 |
| 4 | 158.4 | 91.9 | 177.7 | （v5 未观测） | — |

（"发布"一行是本轮训练后的那次发布；Miles update_weights 来自 §9.5 的表，对应发布 v1–v4。）

稳态（第 2–4 轮）≈ 生成 155 + 训练 90 + 同步 177 + 发布 240 ≈ **660 s**。同步与发布里 GPU 利用率 0–6%。

读码（agentenv/main 59f8fb7a 起，本分支 s16-decouple-p2 上相同）：

- 同步 177 s = `LocalOnlySync.boundary`（bridges.py）→ `driver.export_local()` → `MilesPolicyState.export`（miles_adapter/state.py）→ Ray `run_plugin(export_state)`：训练进程里 Flash-Next 原生导出（各 PP 段导出后 `dist.gather_object` 汇到主 rank、转 CPU float32、NaN/Inf 检查）→ 整份 11.2 GB 经 Ray 回到驱动 → `canonical_state_from_owned_tensors`（规范化）→ `_at_version` 用 `canonical_state` **再复制一份**并再查 NaN/Inf。
- 发布中 Miles 之前的约 213 s = 驱动 `policy_tensor_hash()`（11.2 GB sha256）→ `MilesPublisher.publish` 里 `self._export()` **再整份导出一次**并再算 `policy_tensor_hash` → `payload_digest`（第三次 sha256）→ manifest（小）。
- **各子项没有单独计时**：训练进程内导出（含 `gather_object`）、Ray 传输、驱动复制、哈希各占多少，目前只能推断。本 change 加了日志，下次真机就能分开。

## 决定

### D1 驱动不再拿张量，只拿哈希

单岛无岛间同步时，驱动拿到策略后只做三件事：比哈希、拼 token、写 tape；真正的发布是 Miles `update_weights` 从训练进程直接发给推理引擎。所以把两个哈希挪进训练进程：

- 新插件 `state_plugin.export_digest(actor, policy_version, base_model_revision, lora_config_hash)`：调用**原来的** `export_state`（导出、NaN/Inf、梯度流检查都不变），在主 rank 上调用 `policy_digest.digest_canonical_tensors`，返回 `{policy_version, digest:{policy_tensor_hash, payload_hash, payload_bytes, specs, hash_seconds}, export_seconds}`；非主 rank 返回 None（与 `export_state` 一致）。
- 驱动侧 `MilesPolicyState.export_digest` 校验"恰好一个主 rank 结果、版本一致"，用 (名字, 形状) 重建布局哈希并与已钉的布局哈希比对（首次导出时学习布局哈希，与 `export` 同口径），返回 `TrainerResidentState`。

### D2 两个哈希的定义一字不改

`digest_canonical_tensors` 里两个哈希流逐字节复刻：

- `policy_tensor_hash`：前缀 `yeto-rl-policy-tensors-v1\0` + 基座版本 + LoRA 配置哈希 + 布局哈希 + 按名字排序的（名字 + float32 原始字节）——同 `yeto.rl.core.policy_tensor_hash`；
- `payload_digest`：前缀 `yeto-rl-publication-payload-v1\0` + 按名字排序的（名字长度 4 字节小端 + 名字 + 形状 JSON + 原始字节），字节数累加——同 `publish.payload_digest`。

所以 `rl/policy_token`、`sync/publication_payload_hash`、`sync/publication_payload_bytes`、manifest 的 `target_policy_hash/payload_hash/payload_bytes` 与旧路径**数值相同**；CPU 单测逐项比对旧函数。哈希不按 PP 段拆开（那会改变哈希定义），而是**两个哈希各一个线程同时算**（`hashlib` 处理大块数据时释放 GIL），墙钟约等于一遍。

### D3 "训练端仍持有要发布的那份"检查改为比权重版本号（用户 S17 裁定）

原 `publish` 在发送前再整份导出一次、比内容哈希。用户裁定：快路径下**只比"训练进程里的权重版本号"**，不再重新导出、算哈希。

- **由谁维护**：每个训练进程自己维护（`state_plugin` 模块里的两个值）：`process_id`（进程启动时生成的随机 id）和 `version`（整数，从 0 开始）。驱动不维护、不推算，只读取主 rank 的值。
- **什么时候加一**：训练进程里所有会写可训练权重的入口都加一，宁可多加不漏加：
  1. 每个优化器步：`install_grad_norm_recorder` 包装的 Miles `train_one_step` 返回后（抛异常也加，`finally`）；
  2. `apply_state`（写入全局策略/扰动注入的应用），在写之前加一；
  3. 切点恢复 `cut_plugin.restore_cut_shard`、`restore_resharded_shard`，在写之前加一。
  导出、挪显存（offload/onload）、读梯度范数等不改权重，不加。所有插件入口都会先装好 `train_one_step` 包装，驱动在第一个训练步之前必先导出一次，所以不会漏掉第一步。
- **怎么比**：`export_digest` 导出时把当时的 `{process_id, version}` 一并返回，记在句柄上（`TrainerResidentState.weights_mark`；改版本号的 `with_version` 保留它）。发布前（`publish` 与 `publish_members`）调用新插件 `weights_version` 取主 rank 当前值：
  - 同一进程、版本号相同 → 通过，**不导出、不算哈希**，只一次很小的 Ray 调用；
  - 同一进程、版本号不同 → 失败；
  - 进程 id 不同（训练进程被重建/扩缩容替换，计数从 0 重来，版本号无法比较）→ 退回比内容：训练进程就地导出+哈希、只回传哈希，与句柄的 `policy_tensor_hash` 比。这条只在重建后发生（`driver` 训练进程重建、`trainer_transition` 扩缩容后的补发），与旧语义相同；
  - 句柄上没有版本号（旧插件返回）→ 失败。
- **对不上时报什么错**：训练进程侧抛 `WeightsChanged`（`policy_digest.py`，`PolicyDigestError` 的子类），消息分别为 `trainer weights version X != Y recorded for policy vN (weights written since the export)`、`rebuilt trainer holds <哈希>, not the policy <哈希>`、`the resident state carries no trainer weights version`。发布器把它转成 `PublicationError("trainer weights differ from the state requested for publication: <上述消息>")`——前半句与旧路径完全相同，走同样的发布失败处理；其他异常（如 Ray 调用失败）原样抛出，不冒充"权重不同"。
- **语义变化（用户已接受）**：从"比内容"变为"比是否被写过"。比内容更严的地方：权重被写回原值（如 E1 的 LoRA 扰动注入：先扰动、再恢复原值）也算变化——**带扰动注入的测试运行若在扰动之后还要补发同一版本，需设 `YETO_RL_PUBLISH_FASTPATH=0`**。比内容更松的地方：不经过上述入口的改写（例如直接改 GPU 显存）查不出来；现有代码里没有这种路径。
- 兜底取张量（D4）仍比内容哈希，不受影响。

### D4 兜底：要张量时自动取

`TrainerResidentState` 的 `.tensors`、`to_lora()`、`policy_hash()` 会调用 `materialize`：走原 `export` 整份导出，并核对 `policy_tensor_hash` 与句柄一致（不一致报 `PolicyDigestError`），结果缓存。这样本 change 没预见到的消费者仍然正确，只是慢。已核对的单岛路径消费者只用哈希与版本：`driver.publish`、`driver` 的训练进程重建（只比哈希、再调 publisher）、`controller` 的成员核对与成员补发（`publish_members`）、`rebuild_wiring`/`round_cut` 的进度记录（`policy_tensor_hash()`）、`trainer_transition`（`policy_tensor_hash()` 与 `publish_members`）。`driver.run()` 返回最终策略，生产调用方（learner.py `run_ports_island`）不读它。

### D5 只在单岛无同步时启用，可关

- 只有 `LocalOnlySync` 用 `_local_state`；strict-avg 等多岛同步需要张量做平均，不变。
- `IslandDriver.export_local_resident` 在 `policy_state` 没有 `export_digest`（假对象、以后的其他后端）或环境变量 `YETO_RL_PUBLISH_FASTPATH=0` 时返回 None，调用方回到 `export_local()` 老路。默认开启（用户 S17 裁定：统一默认开启）。
- 没有改 `config.py`/`entry.py`/`cli.py`（去耦合阶段 3 在改），所以开关用环境变量而非命令行参数。

### D6 不在本 change 里做、但可能需要的

- 若训练进程内 `gather_object`（pickle 后经 NCCL 收集 PP 各段）是大头：改为按张量 `send/recv` 或各段先算再汇总——后者会改变哈希定义，需另走"新哈希 + 对照表"流程，本 change 不做。
- （已定，见 D3）发布前检查改为比权重版本号。

## 每轮耗时预估（全部**未验证**）

记 E = 训练进程内一次导出（含 PP 汇总 `gather_object`、转 CPU、NaN/Inf 检查），H = 两线程并行哈希 11.2 GB。均无实测：

- H：sha256 单线程常见 0.5–1.5 GB/s（Modal gVisor 下未测）→ 约 8–22 s。
- E：下限参照 Miles 自己一次导出+汇总+NCCL 广播约 8 s（同一份数据但走 GPU）；上限为整个同步阶段 177 s 减去 Ray 搬运和驱动处理，没有分项数据。按 10–100 s 给区间。用户裁定"汇到主进程"那一步要多久等真机再看。
- 发布前版本号检查：一次小 Ray 调用，按 ≤1 s 计。

| 项 | 现在（实测稳态） | 本 change 后（估） |
|---|---|---|
| 生成 | 155 | 155 |
| 训练 | 90 | 90 |
| 训练后同步 | 177 | E + H ≈ 18–122 |
| 发布（Miles 之前） | ≈213 | 版本号检查 ≈1 |
| Miles update_weights | ≈25 | ≈25 |
| **合计** | **≈660 s（11 min）** | **≈289–393 s（4.8–6.6 min）** |

S17 交接里写的"约 4.5 min"对应 E 很小的情况，需真机确认。

**真机实测（S17 G4 `s17-fn2x8-fastpath-20261008a`，回答 12288，见 tasks §5.x）**：E = 38.7–40.1 s，H = 7.8 s（单线程 sha256 实测约 1.0 GiB/s）；训练后同步 46.7–48.2 s；发布 span 25.9–31.4 s（版本号检查 + Miles update_weights）；稳态每轮约 433 s（生成 222–243 s 因回答变长而比表里多约 75 s）。剩下的大头是 E（训练进程内 PP 汇总 + 转 CPU），见任务 5.5。

## 风险

- 训练进程主 rank 多出两个哈希线程，与训练不重叠（同步/发布阶段训练已结束），内存占用不变（张量本来就在 CPU 上被导出）。
- 训练进程内导出时的 CPU 内存峰值与原路径相同（原路径同样先在训练进程里拼好整份再交给 Ray）；驱动进程峰值内存大幅下降（不再持有 11.2 GB×2）。
- 布局哈希在首个导出学习：`export_digest` 与 `export` 同口径学习（单测 `test_rl_fn_layout` 已改为同时认两条路径）。
