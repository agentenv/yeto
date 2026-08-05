# Miles RL 的 Decoupled DiLoCo 设计

> 状态：源码实现、自动化 protocol/oracle 验证、Miles stop-signal 修改和 clean
> commit pin 已完成。真实 causal-LM GPU、跨机与故障验证矩阵尚未完成，因此
> 当前不标记为 release-usable。
>
> 本文只定义 causal-LM LoRA 的 Miles RL 分布式同步。它不修改 SFT、
> Diffusion、local PPO、CyberGym 环境或其他 Yeto 子系统。

## 1. 目标与定位

当前 `strict-avg` Miles RL 是固定 roster、单 fragment、同步等权 FedAvg。
每个 island 完成一次 rollout/train 后等待完整全局 LoRA，所有 island
应用同一版本后才能继续。该模式用于证明 Miles island 与 Yeto syncer
之间的正确性，不是 Decoupled DiLoCo。

本文增加一个显式选择的模式：

~~~text
--rl-sync-preset decoupled
~~~

该模式的目标是：

1. Miles 持续执行完整 rollout、reward、GRPO 和 local train；
2. LoRA 被拆成多个确定性 fragments；
3. 不同 fragments 的 outer rounds 可以同时在途；
4. island 不等待远端 quorum 或 merge 就继续下一次 local RL round；
5. 已返回的 fragment 只在 rollout/train 安全边界应用；
6. 每次 rollout 始终使用一个原子的、可证明的完整 policy snapshot；
7. syncer 继续提供 outer optimizer、checkpoint、恢复和最终权威 LoRA。

`strict-avg` 保持默认且行为不变。选择 `decoupled` 是有意进入新的
RL v1 合同，不会由 launcher 自动推断。

## 2. 与 INIT v0 的关系

`INIT_Decoupled_DiLoCo_RL_Support.md` 描述的是首个严格同步 v0。
本文不是对 INIT 的重新解释，也不修改 INIT。以下偏差是方案 C 的必要定义：

| 项目 | INIT v0 / strict-avg | decoupled |
| --- | --- | --- |
| fragments | 1 | P >= 2 |
| pipeline | 1 | 1 <= tau <= P，推荐 2 |
| global boundary | 每个 local round 后全局 barrier | local work 与 fragment rounds 重叠 |
| outer optimizer | lr=1, momentum=0 | f32 Nesterov，lr=0.7, momentum=0.9 |
| local SGLang publication | 只发布 committed global LoRA | 每个安全边界发布完整 island-local snapshot |
| optimizer state | 每次完整 global apply 后清空 | in-process fragment apply 保留；进程恢复时清空 |
| policy version | 单一 global version | rollout id + 完整 policy hash + fragment version vector |

除上表列出的变化外，INIT 的这些边界继续成立：

- causal-LM；
- Miles + Megatron；
- LoRA only；
- TP=1、PP=1；
- colocated SGLang rollout；
- 完整 trajectory 和完整 GRPO group；
- 一个 group 只使用一个实际 policy snapshot；
- 固定 logical island roster；
- 不接受 same-fragment stale base；
- model、dataset、reward、Miles 和 Yeto source 均可验证；
- 最终只导出 syncer 权威 checkpoint 中的标准 PEFT LoRA。

## 3. 明确范围

### 3.1 必须实现

- 多 fragment、pipelined Miles RL outer synchronization；
- 每个 fragment 的 exact raw anchor 和 base-relative delta；
- f32、全 roster、等权 AVG outer gradient；
- f32 Nesterov outer optimizer；
- local rollout/train 与远端 fragment merge 的时间重叠；
- rollout policy 的原子发布和精确身份校验；
- in-process partial global apply 时保留 inner optimizer moments；
- island 与 syncer 的现有故障恢复；
- 多 fragment authoritative finalization 和标准 PEFT export；
- 与 native Miles、strict-avg 等工作量的真实 RL benchmark。

### 3.2 明确不做

- full-parameter RL；
- TP>1 或 PP>1 的 LoRA gather/scatter；
- Diffusion RL；
- local PPO 或 CyberGym 环境的新功能；
- actor/critic RL 或非 GRPO advantage estimator；
- 一条 trajectory 中途更新 policy；
- partial trajectory 跨 policy snapshot 恢复；
- same-fragment stale update；
- dynamic quorum、成员缩减或异步 learner weighting；
- RDA、IsoLoCo、HeLoCo、delta correction；
- broadcast blending；
- bf16/q4 wire format；
- adaptive grace；
- optimizer moment federation；
- 多版本 SGLang adapter 常驻；
- dashboard、调度平台或新的通用训练框架；
- 对 SFT `sync_diloco_boundary()` 的重构。

这些项目只有在独立设计被审查后才可进入后续实现，不得作为本方案的
“顺手完善”加入 diff。

## 4. 符号和预算语义

| 符号 | 含义 |
| --- | --- |
| M | 固定 logical Miles islands 数 |
| P | 实际 LoRA fragment 数 |
| tau | 同时在途、且 fragment id 不同的 outer rounds 数 |
| H | 同一 fragment 两次有效 PUSH 之间至少完成的 local optimizer steps |
| N | public `--total-steps`，完整 fragment sweeps 数 |
| T | syncer outer fragment steps，固定为 N * P |
| r_i | island i 的单调 local rollout id |
| s_i | island i 已完成的累计 optimizer steps |
| u_i | island i 已完成的累计 action tokens |
| V_i | island i 当前完整 policy 的 P 维 fragment version vector |

在首版中，一次 Miles rollout/train cycle 固定执行一个 optimizer step。
因此 H 也等于同一 fragment 两次提交之间至少完成的完整 local RL rounds。
这与“在同一批 rollout 上执行多个 optimizer steps”不同，后者不在本文范围。

RL 模式下 `--total-steps N` 始终表示完整 LoRA sweeps：

~~~text
strict-avg:  P=1, syncer total_steps=N
decoupled:   P>=2, syncer total_steps=N*P
~~~

这保证最终 cut 中每个 fragment 正好参与 N 次 round-robin outer rounds，
不会产生由 `T % P != 0` 导致的残缺 sweep。

## 5. Public CLI 和固定配置

示例：

~~~bash
yeto launch \
  --training-mode rl \
  --rl-sync-preset decoupled \
  --fragments 8 \
  --pipeline 2 \
  --local-rl-rounds-per-sync 4 \
  --total-steps 8 \
  ...现有 Miles RL 参数...
~~~

选择 `decoupled` 后：

- `--fragments` 必须显式满足 `P >= 2`；
- canonical LoRA tensor 数必须不少于 P，实际 layout 必须恰好生成 P 个 fragments；
- `--pipeline` 必须满足 `1 <= tau <= P`；
- `--local-rl-rounds-per-sync` 必须满足 `H >= 2`；
- `--fragment-pattern` 必须为 `binpack`；
- `--experimental-rl-sync` 被拒绝，避免两套覆盖机制叠加。

launcher 固定或归一化：

~~~text
quorum                 = M
grace_ms                = 0
max_base_lag            = 0
learner_weight          = equal
sync_interval_steps     = H
outer_lr                = 0.7
outer_momentum          = 0.9
merge_alpha             = 0
delta_correction        = none
wire_dtype              = f32
checkpoint_every        = 1
optimizer_steps/rollout = 1
syncer total_steps      = N * P
~~~

首个真实验证配置采用 `P=8, tau=2, H=4`。它是验证起点，不在文档中
宣称为跨模型最优配置。

## 6. 组件职责

### 6.1 Miles

Miles 继续独占：

- SGLang rollout；
- 多轮或工具调用环境；
- reward 和 GRPO group；
- Megatron local train；
- trainer 到 SGLang 的完整 LoRA publication；
- local optimizer 和 scheduler。

Miles 不理解 Yeto fragment merge 数学，也不直接连接 syncer。

### 6.2 Yeto RL adapter

Yeto RL adapter 负责：

- canonical PEFT LoRA export/apply；
- deterministic RL fragment layout；
- PULL/BCAST/PUSH 状态；
- raw anchors 和 fragment version vector；
- rollout policy snapshot；
- local durable progress；
- final fragment assembly 和 ACK。

### 6.3 Rust syncer

现有 syncer 原样负责：

- P 个 fragment 的 round-robin PULL；
- tau 个不同 fragments 同时在途；
- exact-base 和 full-roster validation；
- AVG outer gradient；
- f32 Nesterov outer optimizer；
- checkpoint-before-broadcast；
- checkpoint resume；
- f32 FINAL_FRAGMENT / FINAL_MANIFEST / FINAL_ACK。

本方案不增加 wire message，也不修改 Rust。

## 7. Canonical LoRA 与 fragment layout

### 7.1 两个不同的 hash

现有单 fragment v0 将两个概念合并成一个 layout hash。多 fragment 后必须
明确区分：

1. `canonical_layout_hash`：Miles 和 Yeto 之间完整 tensor schema 的身份，
   只包含 canonical name、shape、dtype 和 order，不因 P 改变；
2. `sync_layout_fingerprint`：protocol HELLO 和 syncer checkpoint 使用的
   P-fragment layout 身份，包含 fragment id、merge mode、tensor membership、
   numel 和 shape。

`CanonicalLoraState.layout_hash` 继续表示第一种身份，避免破坏 strict-avg
与 Miles 薄分支合同。decoupled bridge 单独持有第二种 fingerprint。

### 7.2 RL all-AVG binpack

RL 使用一个小型、确定性的 all-AVG builder：

1. 按 `(-numel, name)` 排序 canonical tensors；
2. 建立 P 个空 bins；
3. 每次放入当前总 numel 最小的 bin，tie 按 fragment id；
4. 每个 fragment 的 merge mode 固定为 AVG；
5. 每个 fragment 保存完整 tensor shapes 用于 semantic fingerprint。

不复用 SFT 的 embedding 特判、RDA 或 matrix merge。复用的只有
`Fragment`、`FragmentLayout`、protocol fingerprint 和 tensor I/O。

所有 island 和 exporter 必须从相同 specs 重建 bit-identical layout。
任一 HELLO fingerprint 不一致都在训练开始前失败。

## 8. Policy snapshot

decoupled 模式下，一个可用于 rollout 的 policy 定义为：

~~~text
PolicySnapshot {
    rollout_id: island-local monotonic integer
    fragment_versions: tuple[u64; P]
    policy_hash: sha256(full canonical f32 LoRA)
}
~~~

SGLang weight version token 为：

~~~text
yeto:<rollout_id>:<policy_hash>
~~~

event tape 同时记录 token、完整 `fragment_versions`、canonical layout hash
和 sync layout fingerprint。token 本身不压入 version vector，避免长度随 P
增长；event tape 和 island checkpoint 提供可审计映射。

以下条件是 hard failure：

- trajectory 缺少 weight version；
- trajectory 记录多个 token；
- 同一 GRPO group 内 token 不一致；
- rollout 返回的 token 与预期 snapshot 不同；
- trainer apply 后的完整 policy hash 与待发布 snapshot 不同。

## 9. 安全边界和完整状态机

Miles 当前 hook 位于 local train 完成之后、SGLang update 之前。该位置保持不变。
syncer 的接收线程可在 rollout/train 期间缓存消息，但不得修改 trainer 或 SGLang。

### 9.1 初始化

1. 每个 island 从 Miles 导出完整 canonical LoRA specs。
2. 每个 island 重建相同 P-fragment all-AVG layout。
3. learner 0 分别发送 P 个 INIT_PARAMS。
4. 所有 islands 等待 P 个初始 BCAST，暂存完整 raw global LoRA，但不提交
   anchor 或 version。
5. 以 `reset_optimizer=True` 应用完整初始 policy，并重新导出校验 hash。
6. 只有 apply/hash 成功后才提交每个 fragment 的 raw anchor、version 和 counters。
7. 计算首个 PolicySnapshot，设置 SGLang token。
8. Miles 完成完整 LoRA update 后才开始 rollout 0。

### 9.2 普通 local rollout/train

Miles 顺序执行：

~~~text
generate complete groups using snapshot Q_r
-> validate every trajectory token == Q_r
-> GRPO train once
-> call Yeto after_local_train hook
-> publish the hook's resulting complete LoRA to SGLang
-> start rollout r+1
~~~

rollout 与 train 期间到达的 BCAST/PULL 只进入队列。

### 9.3 after_local_train 的严格顺序

每次 hook 必须：

1. 验证刚完成的 rollout snapshot；
2. 更新累计 optimizer step、action token 和 RL metrics；
3. 从 Miles 导出 train 后的完整 canonical LoRA；
4. drain BCAST，并按 fragment id 排序，但只暂存 update；
5. 将所有单调的新 BCAST 写入暂存的完整 LoRA；
6. 如有 BCAST，仅调用一次完整 Miles apply，且 `reset_optimizer=False`；
7. 重新导出并校验 trainer full-policy hash；
8. 只有 apply/hash 成功后才提交收到 BCAST 的 raw anchors、versions 和
   reset counters；
9. drain PULL，并保存每个 fragment 的唯一 pending permit；
10. 对达到 H 的 permits，从同一份 post-apply snapshot 计算 fragment delta；
11. 发送 PUSH，不等待 quorum、merge 或 BCAST；
12. 过滤不再匹配完整 policy hash 的 completed groups；
13. 原子写 island checkpoint；
14. 生成下一 PolicySnapshot，并更新 SGLang version token；
15. 返回 Miles，让 Miles 发布完整 LoRA。

必须先 apply BCAST，再回答 PULL。这样下一轮同 fragment 的 PULL 即使在控制流上
越过前一 BCAST，也只能基于新的 raw anchor 发送。

若 trainer apply 或 hash 校验失败，步骤 4-5 已 drain 的 BCAST 不得改变 bridge
中的 committed anchor/version；进程失败后由现有权威 cut 恢复。

### 9.4 PULL eligibility

fragment p 只有在以下条件全部满足时才能 PUSH：

~~~text
raw_anchor[p] exists
pending_pull[p] exists
current_optimizer_steps - steps_at_anchor[p] >= H
no response has been sent for the same (global_step, round_attempt)
~~~

PUSH 内容：

~~~text
fragment_id = p
global_step = permit.global_step
round_attempt = permit.round_attempt
base_version = fragment_version[p]
local_step = cumulative optimizer steps
c_steps = current steps - steps_at_anchor[p]
c_tokens = current action tokens - tokens_at_anchor[p]
payload = local_fragment[p] - raw_anchor[p]
~~~

`learner_weight=equal` 保证 `c_steps/c_tokens` 只用于审计、节奏估计和 ledger，
不改变 island 的全局贡献。

PUSH 后不移动 anchor。只有收到对应的 committed BCAST 后才替换 anchor 和
重置 fragment counters。

### 9.5 本地计时和流量口径

wire 不增加字段。Python receiver 在接受完整 `PULL_REQ` 或 `BCAST_FRAGMENT` 时，
只在本地对象记录 monotonic `received_at`：

- PULL-to-PUSH：本地接收 PULL 到 `push_fragment` enqueue；
- BCAST queue：本地接收完整 BCAST 到安全边界 drain；
- fragment payload bytes：仅 tensor payload，不含 message header、frame、chunk、
  HELLO、PULL、manifest 或 ACK 等 framing/control bytes。

普通 BCAST、PUSH 和普通 finalization 的 FINAL_FRAGMENT payload 都必须纳入统计。

## 10. Outer update

由于 `max_base_lag=0`，进入同一 fragment round 的所有 islands 都以 syncer
当前 fragment 为 raw anchor。对 fragment p：

~~~text
d_i,p = theta_i,p - anchor_i,p
g_i,p = -d_i,p = anchor_i,p - theta_i,p
g_p   = mean_i(g_i,p)

momentum_p = mu * momentum_p + g_p
Theta_p    = Theta_p - lr * (g_p + mu * momentum_p)
~~~

固定 `lr=0.7, mu=0.9`。merge、momentum、version update 和 checkpoint
全部由 syncer 现有 f32 路径完成。

并行 rounds 始终针对不同 fragment，因此：

- 一个 fragment 内不存在两个并发 owners；
- 每个 fragment version 单调；
- rounds 可乱序完成；
- syncer global step 单调取已完成 round 的最大值；
- 完整 policy 可以有不同的 per-fragment versions，这是 Decoupled DiLoCo
  的预期状态，不是 mixed-version trajectory。

## 11. Trainer 与 SGLang 的原子性

Yeto 不在 rollout 或 train 中途调用 apply。Miles 的训练循环仍是顺序的，
下一次 generate 只会发生在：

1. Yeto hook 已返回；
2. Miles 已获得 rollout engine update lock；
3. 完整 LoRA 已写入所有 SGLang engines；
4. 对应 PolicySnapshot token 已设置。

token 在 weights publication 前设置是安全的，因为这段区间内 Miles 不启动
generation。若 weights publication 失败，进程失败并进入现有恢复流程，不允许
继续生成。

不允许按 fragment 分多次更新 SGLang。无论本次 hook 收到几个 BCAST，
对 SGLang 都只发布一个完整 snapshot。

## 12. Optimizer 和 scheduler

### 12.1 In-process fragment apply

收到 BCAST 后，Yeto 构造一份完整 TrainableState：

- BCAST fragments 使用 raw global 值；
- 其他 tensors 使用当前 local 值；
- policy progress 使用当前 island-local rollout progress。

调用现有 Miles：

~~~python
apply_trainable_state(full_state, reset_optimizer=False)
~~~

因此：

- LoRA f32 master params 和 model params 被同步更新；
- actor backup 被刷新；
- Adam moments 不清空；
- scheduler 不回退、不重新 warmup；
- 下一次 local train 延续同一个 inner optimizer。

保留 moments 是方案 C 的固定算法语义。按 fragment 重置 moments 或联邦合并
moments 都不在本设计内。

### 12.2 初始化与进程恢复

初始化或新进程恢复时使用 `reset_optimizer=True`。新进程没有可继续使用的
local Adam history，但 scheduler progress 必须从 island checkpoint 恢复。

Miles TrainableState 的 `policy_version` 在 decoupled apply 中只承载
island-local rollout progress，以复用现有 scheduler alignment；它不代表
syncer global step。syncer 身份只由 fragment version vector 表示。

## 13. Completed groups

completed group checkpoint 必须保存其完整 policy token。一个 group 仅在以下
条件满足时可进入训练：

~~~text
all trajectories terminal
group size == n_samples_per_prompt
all trajectory tokens identical
token == current PolicySnapshot token
~~~

每次 local train 或 BCAST apply 后 full policy hash 都可能改变。hook 在发布
下一 snapshot 前删除所有旧 token groups。

这意味着 oversampling 的额外 groups 只能在同一 generate/selection 周期内使用，
不能跨 local policy update 复用。首版接受这部分长尾浪费，以保持严格 on-policy
GRPO 语义。

## 14. Island checkpoint

每个 island 的现有 checkpoint 扩展为一个原子文件，至少包含：

~~~text
schema version
immutable run config
next_rollout_id
cumulative optimizer steps
cumulative action tokens
last PolicySnapshot token/hash
last observed fragment version vector
rollout metrics
completed groups with exact token
~~~

它不保存 local LoRA 或 optimizer moments。local LoRA 不是权威状态；syncer
checkpoint 才是唯一全局权威。

checkpoint 必须在下一次 rollout 开始前完成写入。Spot 模式继续使用现有
per-island reconstruction storage mount，不新增存储系统。

## 15. 故障恢复

### 15.1 Island 进程或节点失败

1. client connection failure 继续使 island 进程退出；
2. launcher 以相同 logical learner id 重启整个 Miles island；
3. syncer 向新 generation 广播当前 P 个 committed fragments；
4. island 从 checkpoint 恢复 local rollout/scheduler progress；
5. 组装并应用完整权威 global cut，重置 optimizer moments；
6. 重新计算 policy hash；
7. 仅当重建 snapshot 的 token、full-policy hash 和 fragment-version vector 与
   checkpoint 完全一致时保留 completed groups；否则丢弃；
8. 继续响应当前或重试的 exact-base permits。

未提交的 local LoRA 和正在运行的 trajectory 被丢弃。

若 syncer 已是非零 cut，而 island checkpoint 缺失、损坏或配置不匹配，
island 必须失败，不能从 syncer global step 猜测 local scheduler progress。

### 15.2 Syncer 失败

syncer 从现有 checkpoint 恢复：

- params、outer momentum、per-fragment versions 和 ledger 恢复；
- 所有 in-flight、未 checkpointed rounds 被丢弃；
- islands 因连接失败退出并由 launcher 重启；
- 新 syncer 广播 committed cut；
- 训练从该 cut 继续。

由于 `checkpoint_every=1` 且 exact-base 模式在 BCAST 前提交 checkpoint，
不会发布一个没有持久化的 fragment version。

### 15.3 网络中断

首版不让 island 在失联期间无限本地训练。现有 `max_reconnects=0` 行为保留：
连接失败触发 island 重启和权威 cut 重应用。这避免增加 same-fragment stale
updates 或新的故障状态机。

## 16. Finalization

syncer 完成 `T=N*P` 个 outer fragment steps 后：

1. 停止发出普通 PULL；
2. 写入最终 quiescent checkpoint；
3. 向 frozen final roster 发送 P 个 f32 FINAL_FRAGMENT；
4. 发送包含 `global_step=T` 和 P 个 expected versions 的 FINAL_MANIFEST。

island 在下一个安全 hook：

1. 不再提交普通 local delta；
2. 等待 manifest 要求的全部 exact FINAL_FRAGMENT；
3. 禁用 blending，组装完整 canonical LoRA；
4. apply 到 trainer，`reset_optimizer=False`；
5. export 并校验完整 policy hash；
6. 原子写最终 island checkpoint；
7. 发送 FINAL_ACK；
8. 设置最终 SGLang token；
9. 通知 Miles 在本次完整 weights publication 后退出。

finalization 到达前已经开始的 rollout/train 可以完成，但其未进入已完成 outer
round 的 local work 会被最终 cut 覆盖，不进入 artifact。

## 17. Miles 薄分支的唯一新增行为

现有 export/apply 和 post-train hook 已足够。Miles 薄分支只需让 hook 可以请求
训练循环停止：

~~~python
should_stop = False
if external_policy_sync is not None:
    should_stop = bool(
        await external_policy_sync.after_local_train(...)
    )

await actor_model.update_weights(rollout_id=rollout_id)

if should_stop:
    break
~~~

要求：

- 返回 `None` 的现有 hook 等价于 `False`；
- native Miles 和 strict-avg 行为不变；
- stop 只能发生在完整 SGLang weights publication 之后；
- loop 退出后仍调用现有 `external_policy_sync.finalize()`；
- 不修改 Megatron trainer、rollout manager、reward、Sample 或 actor APIs。

decoupled 模式由外部 sync finalization 决定训练停止，不使用伪造的巨大
`--num-rollout`。Miles 训练循环仅在 external sync 模式下采用
run-until-stop；native Miles 继续使用固定 `range(num_rollout)`。

## 18. Yeto 实现边界

### 18.1 允许修改

| 文件/区域 | 必要职责 |
| --- | --- |
| `yeto/cli.py` | 增加 `decoupled` preset 选择和必要参数校验 |
| `yeto/launcher.py` | 固定合同、计算 `T=N*P`、传入 P/tau/H |
| `yeto/protocol.py` | 仅给 PULL/BCAST 本地对象记录接收时间，不修改 wire |
| `yeto/rl/core.py` | all-AVG binpack layout、PolicySnapshot 与 token/hash helpers |
| `yeto/rl/decoupled.py` | 独立 async fragment bridge 和状态机 |
| `yeto/rl/miles.py` | snapshot 校验、hook 驱动、checkpoint、strict/decoupled 选择 |
| `yeto/rl/learner.py` | Miles 参数映射和 decoupled runtime config |
| `yeto/rl/export.py` | 多 fragment checkpoint 重建和标准 PEFT export |
| `scripts/benchmark_rl.py` | decoupled benchmark arm 和等工作量 budget |
| RL tests/docs | 本文列出的合同验证和用户说明 |

`StrictRlBridge` 保留给 `strict-avg`。新的 async bridge 放在独立 RL 文件，
不把两种状态机揉成大量条件分支。

### 18.2 不允许修改

- `syncer/src/**`；
- `yeto/diloco_sync.py`；
- SFT learner、SFT benchmark 和 SFT adapter lifecycle；
- Diffusion modules 和 benchmark；
- local PPO、CyberGym env/reward；
- cloud provider、dashboard、controller 或通用恢复框架；
- 与 RL canonical LoRA 无关的 Miles 文件。

若实现发现必须修改上述区域，应停止并重新审查设计，不能扩大 diff 后再解释。

## 19. Export

`yeto-rl-export` 增加训练时使用的 P，并固定使用 RL all-AVG binpack：

1. 从 base model revision 和 LoRA config 重建 canonical specs；
2. 重建恰好 P 个 RL fragments；
3. 比较 checkpoint `sync_layout_fingerprint`；
4. 检查 fragment 数、numel 和每个 terminal version；
5. 将 P 个 f32 fragment params 写回 canonical tensor mapping；
6. 验证完整 canonical policy hash 和 finite values；
7. 写标准 `adapter_model.safetensors` 与 `adapter_config.json`。

checkpoint `global_step` 是 outer fragment step，不假装是单一 rollout policy
version。artifact provenance 同时记录：

~~~text
sync preset = decoupled
P, tau, H, N, T
outer lr/momentum
final fragment versions
canonical layout hash
sync layout fingerprint
checkpoint SHA256
~~~

## 20. Metrics

### 20.1 Rollout policy

- `rl/rollout_id`
- `rl/policy_hash`
- `rl/fragment_versions`
- `rl/mixed_version_group_count`
- `rl/completed_groups`
- `rl/completed_trajectories`
- reward、KL、ESS、clip fraction 和 action tokens

### 20.2 Fragment synchronization

- fragment id、permit global step 和 round attempt；
- raw anchor version 和 returned version；
- `c_steps`、`c_tokens`、realized H；
- local fragment delta norm；
- PULL-to-PUSH 时间；
- BCAST queue 时间和 apply 时间；
- hook 总时间；
- fragment tensor payload bytes sent/received，不含 framing/control；
- policy snapshot 发布次数；
- in-process apply count；
- recovery optimizer reset count。

### 20.3 系统效果

benchmark 记录：

- rollout、train、hook、finalization wall time；
- 等待远端 quorum 的时间，普通 hook 中应为 0；
- 每 GPU active time 和 active fraction；
- 平均/最小 GPU utilization；
- trajectories/s 和 action tokens/s；
- artifact-ready wall time 和 GPU-hours。

## 21. 测试和真实验证

### 21.1 Unit tests

- all-AVG binpack 在不同输入顺序下生成相同 P fragments；
- P 大于 tensor 数、空 layout、fingerprint mismatch 明确失败；
- PolicySnapshot token round-trip 和 policy hash；
- completed groups 只接受 exact token；
- BCAST 单调版本、重复版本和非法 fragment；
- BCAST 在 trainer apply/hash 成功前不提交 anchor/version；
- BCAST-before-PULL ordering；
- PULL/BCAST 本地接收计时和 fragment payload/final-cut 统计口径；
- `c_steps < H` 时保留 permit；
- 多个不同 fragment permits 在同一 hook 正确 PUSH；
- fragment delta 精确等于 local minus raw anchor；
- partial apply 使用 `reset_optimizer=False`；
- final fragment assembly 和 exporter layout 校验。

### 21.2 Mock integration

用两个 logical islands、`P=2, tau=2, H=2`：

- 手工 f32 oracle 验证每个 fragment 的 AVG + Nesterov；
- 证明一个 fragment 等待 quorum 时另一个 round 和 local RL 继续；
- 证明 rollout/train 期间到达 BCAST 不改变当前 token；
- 证明 final policy、两 islands trainer、SGLang 和 exporter hash 一致；
- 证明每个 completed round 都包含完整 roster。

### 21.3 Recovery

- island 在 local train 中途退出；
- island 在 PUSH 后、BCAST 前退出；
- syncer 在一个 fragment committed、另一个仍在途时退出；
- checkpoint 中 fragment versions 不相等时恢复；
- exact token/hash/version snapshot 的 completed groups 在恢复后保留；
- stale completed group 在恢复后被删除；
- 非零 syncer cut 缺少 island checkpoint 时明确失败；
- finalization 期间断开并以新 generation 完成 ACK。

### 21.4 Miles tests

- hook 返回 `None/False` 不停止；
- hook 返回 `True` 时先 update_weights 再退出；
- native loop 和 strict-avg loop 不变；
- trainer apply/export 在多次 `reset_optimizer=False` 后仍全 rank 一致；
- scheduler local progress 不因 fragment global step 改变。

### 21.5 GPU E2E

必须使用真实 causal LM、真实 SGLang generation、真实 reward 和真实 GRPO：

1. 单 island `P=2`，验证 fragment pipeline、final export 和推理；
2. 双 island `P=2, tau=2, H=2`，验证 f32 oracle；
3. 双 island推荐配置 `P=8, tau=2, H=4`；
4. 至少一次真实跨机运行；
5. island kill/restart 和 syncer kill/restart；
6. 标准 PEFT load 和生成。

不允许用 injected rollout、synthetic optimizer step、fake reward 或跳过
trainer/SGLang publication 来替代上述验证。

## 22. Benchmark 公平性

普通 production run 由 outer budget `N*P` 停止，网络和 straggler 会使
不同 arm 实际 local work 略有差异，因此不能直接作为质量 benchmark。

正式 benchmark 复用现有 protocol `BUDGET_DONE` 和 consolidation restart：

1. native、strict、decoupled 使用相同 local optimizer-step budget R；
2. 每个 island 达到 R 后冻结 trainer 和 rollout，不再产生 local work；
3. 向 syncer 报告 BUDGET_DONE；
4. syncer 取消未完成 rounds并写未标记 checkpoint；
5. benchmark harness 从该 checkpoint 重启 syncer，pipeline=1；
6. frozen islands 依次响应 P 个普通 fragment pulls；
7. 完成一轮全 fragment consolidation 和正常 finalization；
8. 最终才导出并执行 held-out evaluation。

该流程复用现有 Rust 协议和 scheduler，不增加 RL 特殊 wire message。

正式对比至少包含：

| arm | 目的 |
| --- | --- |
| native Miles | 同 runtime、同总 GPU、无 Yeto sync |
| strict-avg | 当前同步 FedAvg 控制组 |
| decoupled | 本文方案 C |

三者必须匹配：

- model/revision、LoRA config、prompt stream、reward；
- total GPUs 和每个 seed；
- optimizer steps、prompt groups、trajectories；
- 最大 action-token budget；
- held-out prompts 和 generation seeds。

评估 reward、pass@k、KL/ESS/clip、wall time、GPU active fraction、
artifact-ready time、GPU-hours、realized H、fragment traffic 和 recovery evidence。

## 23. Definition of Done

只有同时满足以下条件，才能把 `decoupled` 从“已实现”描述为“可用”：

1. default `strict-avg` 行为和 tests 不变；
2. Miles 薄分支只增加本文定义的 stop signal；
3. Rust 和 SFT 没有功能 diff；
4. 多 fragment layout 和 exporter 由同一 deterministic helper 构建；
5. 每个 ordinary outer round 使用完整 logical roster 和 exact fragment base；
6. 普通 local hook 从不等待远端 quorum/merge；
7. rollout/group policy token 始终单一且可映射到 hash/version vector；
8. in-process fragment apply 保留 optimizer moments；
9. island/syncer failure 能从权威 cut 恢复；
10. final artifact 只来自 exact FINAL_FRAGMENT cut；
11. 双 island f32 outer-update oracle 通过；
12. 真实跨机 RL run、failure run、PEFT export/load/generation 通过；
13. equal-work native/strict/decoupled benchmark 完成；
14. MILES_RL 与 MILES_RL_ZH 只描述实际完成并验证的行为。

## 24. 审查边界

实现时，每一处 diff 必须能映射到以下四类之一：

1. fragment synchronization；
2. rollout policy snapshot correctness；
3. decoupled recovery/finalization/export；
4. 必要的真实 RL validation。

无法映射到这四类的修改不属于本方案。遇到需要修改 Rust、SFT、Diffusion、
local PPO、CyberGym 或通用基础设施的情况时，应停止实现并重新审查，而不是扩大
本文边界。
