# Yeto RL v0：固定成员同步 LoRA FedAvg 方案

> 状态（2026-07-27）：经第二轮方案对齐复核，计划内 Yeto 侧代码已收口，固定
> Miles 源码已核实，并已通过 CPU、FakeMiles 和真实 Rust syncer 自动化验证；
> 真实 Miles/Megatron/SGLang GPU 验收尚未完成，因此第 13.4、13.5 节和第 14 节
> 仍未完成，不能据此宣称 RL v0 已完成或可用于生产。
>
> 固定依赖：`https://github.com/radixark/miles` @
> `dfc66ff38752bfa2c5d325e0037ebc4b537c06de`。
>
> 本文只定义 Yeto 当前 RL v0 所需的最小闭环，不包含异步 RL、Decoupled
> DiLoCo、模型扩容路线、人员排期或后续算法研究计划。

## 0. 结论

Yeto RL v0 采用以下分层：

```text
Yeto
├── 固定 island roster 的启动与恢复
├── 全局 LoRA 的同步等权聚合
├── policy version、commit 和 authoritative checkpoint
└── 最终标准 PEFT LoRA 导出

每个 island
└── Miles
    ├── SGLang rollout 与工具环境
    ├── 完整 GRPO group 组装
    ├── reward / advantage
    ├── Megatron 本地优化
    └── trainer 与 SGLang 的 LoRA 应用
```

不把 `yeto/learner.py` 或 `yeto/megatron/learner.py` 改造成 RL trainer。RL
走独立 runtime 入口，只复用 Yeto 的 canonical tensor、`SyncerClient`、Rust
syncer、launcher 和 provenance 能力。

算法上，v0 是：

> 多个固定成员的 Miles island，从同一全局 LoRA 出发，各自执行固定工作量的
> GRPO，再由 Yeto 同步、等权平均本地 LoRA。

它是对本地 GRPO/Adam 结果做同步 FedAvg，不是 Decoupled DiLoCo，也不等价于
把所有 trajectories 放到一个集中式 GRPO optimizer 中训练。文件名暂时保留，
但实现和日志必须使用 `rl-strict-avg` / `synchronous LoRA FedAvg`，避免误报
算法语义。

---

## 1. 已核实事实与外部假设

### 1.1 Yeto 当前代码事实

| 事实 | 对 v0 的影响 |
| --- | --- |
| `build_layout(..., num_fragments=1)` 默认仍把普通矩阵放入 RDA fragment | RL 不能直接调用默认 `build_layout`；必须显式构造单个 `MERGE_AVG` fragment |
| Rust syncer 使用 `min(configured quorum, live members)` | `quorum=M` 目前不等于固定 M 个成员，必须增加 strict roster 语义 |
| `FleetController` 会 abandon 失败 learner，并让剩余 fleet 继续 | RL strict 模式必须禁止 fleet shrink；无法恢复任一逻辑 learner 时整次 run 失败 |
| 默认 quorum timeout 是 900 秒，超时会清空已收 push、增加 attempt 并重新拉取 | 长任务 RL 不能沿用该超时语义 |
| `HEARTBEAT` 当前只校验消息，没有续租 round | 不能把现有 heartbeat 当成长 rollout 的正确性保证 |
| `complete_round()` 先广播，再按默认每 8 轮 checkpoint | 存在 learner 已见 v+1、syncer 却只能从 v 恢复的回滚窗口 |
| checkpoint 只保存 version、numel、参数、momentum 和 ledger | resume/export 还必须绑定 run manifest 与 semantic layout fingerprint |
| protocol v4 已有 base version、round attempt、layout fingerprint、final manifest/fragment/ACK | 核心数据面可以复用，不需要为 v0 增加 wire message 类型 |
| 普通 exporter 主要按 fragment 数量和 numel 校验 | 不能用于 Megatron 名称到标准 PEFT 名称尚未证明一致的 RL artifact |
| 当前 Yeto Megatron backend 仍标为未完成真实多节点验证 | RL v0 不能把它视为 Miles/Megatron 集成已成立的证据 |

### 1.2 固定 Miles 边界与验收前提

Yeto 只支持 `radixark/miles` 的上述固定 commit。launcher 以 detached HEAD
检出该 revision；learner 启动时再次校验 HEAD、origin、整个 checkout 无 tracked
改动或非 ignored untracked 文件，并确认实际 import 的 `miles` 来自该 checkout。
run manifest 同时绑定 Miles repository/commit 和 immutable learner image。

已对固定源码核实 `weight_versions` 的 trajectory→train batch 传递、SGLang
pause/continue 与 version API、Megatron-Bridge adapter conversion task、rollout
offload/onload 和 trainer→SGLang update 路径。Yeto 不修改上游 checkout，而是在
固定版本外实现窄适配层，以提供以下语义：

1. 暂停新 rollout admission，并等待或取消当前未完成任务；
2. 禁止本地训练完成后自动把 island-local LoRA 发布到 SGLang；
3. 导出和应用完整 trainable LoRA，并在 TP=PP=EP=1 下给出稳定映射；
4. 在每个 trajectory/group 上保留实际 rollout policy version；
5. 正确重置 Megatron `DistributedOptimizer` 的 master parameters、moments 和
   step counters，而不只是修改表层 `optimizer.state`；
6. 等待 SGLang 完成整套 LoRA 更新，并能返回已应用的 policy version/hash；
7. 在 global policy 应用完成前保持 rollout admission 关闭。

这些是 v0 本身的必要合同，不是后续扩展项。源码存在相应原语只证明适配路径
可实现；optimizer master、trainer/SGLang 数值一致性等设备行为仍必须通过第
13.4 节的真实 GPU 验收。

该 Miles commit 在 tokenizer/processor、Megatron-Bridge 和 SGLang 路径中会
无条件使用 `trust_remote_code=True`。因此 Yeto 不会静默继承该信任决定：RL
launch 必须显式提供 `--trust-remote-code`，并把该值写入 manifest，否则在创建
云资源前失败。

---

## 2. 功能范围

### 2.1 必须支持

- 长任务、多轮和工具调用的完整 trajectory；
- 每个 GRPO group 固定 `K` 条完整 trajectories；
- 一个 group 内所有 trajectories 由同一 global policy version 生成；
- 每个 island 独立完成 rollout、reward、GRPO 和固定次数本地 optimizer step；
- 固定 `M` 个 island 的完整 LoRA 等权平均；
- 所有 island 应用并验证同一个 global LoRA 后才进入下一轮；
- learner、syncer 或网络中断后的同版本重试与恢复；
- authoritative checkpoint 的单调版本恢复；
- 从 authoritative checkpoint 导出可由标准 PEFT 加载的 LoRA artifact。

### 2.2 v0 明确不做

- full-parameter RL；
- partial trajectory 跨 policy version 继续；
- stale update、partial quorum 或 fleet shrink；
- RDA、Iso、HeLoCo、server momentum、local/global blending；
- 多 fragment、pipelined global round 或异步 global policy；
- TP>1、PP>1、EP>1 的 LoRA 全局同步；
- 不同 island 使用不同 group 数、optimizer step 数或训练超参数；
- optimizer moments 的跨 island 合并；
- turn-level credit assignment 或其他 GRPO 之外的 advantage estimator；
- NVIDIA trainer 与远程 Ascend rollout 的混合拓扑；
- MoE、FP8、INT4/Q4 与 RL global sync 同时首发。

---

## 3. 算法合同

### 3.1 符号

- `M`：启动时固定的逻辑 island 数；逻辑 ID 为 `0..M-1`；
- `v`：已持久化、已发布的 global policy version；
- `θ_v`：version `v` 的 canonical global LoRA；
- `G`：每个 island、每轮用于训练的完整 GRPO group 数；
- `K`：每个 group 的 trajectory 数；
- `U`：每个 island、每轮固定的本地 optimizer step 数；
- `θ_{i,v}`：island `i` 从 `θ_v` 开始完成本地训练后的 LoRA。

Base model 在整个 run 中不可变；`policy version` 只表示 LoRA 版本。
`θ` 的数值以 Megatron FP32 master LoRA state 为准；低精度 compute copy 不作为
canonical hash 的输入。

### 3.2 一轮的唯一合法语义

对每个 island `i`：

1. 接收并验证 authoritative `θ_v`；
2. 暂停 rollout admission，写入 trainer，重建干净 optimizer，再写入 SGLang；
3. 验证 trainer 与 rollout runtime 都声明已安装 `(v, policy_hash_v)`；
4. 等待并接受以 `v` 为 base 的唯一 strict PULL permit；
5. 恢复 admission，收集恰好 `G` 个完整 group，每组恰好 `K` 条 trajectory；
6. 校验这些 group 的 rollout version 全部为 `v`；
7. 使用相同超参数执行恰好 `U` 个本地 optimizer step；
8. 暂停 admission，导出 `θ_{i,v}`；SGLang 此时仍保持 `θ_v`，不得安装
   `θ_{i,v}`；
9. 计算并缓存 `d_{i,v} = θ_{i,v} - θ_v`，提交给 Yeto；
10. 等待 Yeto 收到固定 roster 中全部 `M` 个结果并提交 `θ_{v+1}`；
11. 应用、验证 `θ_{v+1}`，再开始下一轮。

Yeto 不根据 trajectory 数、action token 数或 reward 改变 merge 权重。为了让
现有权重函数得到严格等权，每个 push 固定发送：

```text
global_step = target step t = v + 1
local_step = t
c_steps = 1
c_tokens = 1
```

strict syncer 同时校验这些字段，避免调用方配置错误后静默改变权重。

### 3.3 聚合公式

learner wire payload 是：

\[
d_{i,v} = \theta_{i,v} - \theta_v
\]

现有 Rust server 解码后转成 outer gradient
`g_{i,v} = θ_v - θ_{i,v}`。在单个 `MERGE_AVG` fragment、等权、
`outer_lr=1`、`outer_momentum=0` 时：

\[
\theta_{v+1}
= \theta_v - \frac{1}{M}\sum_{i=1}^{M}
  (\theta_v - \theta_{i,v})
= \frac{1}{M}\sum_{i=1}^{M}\theta_{i,v}
\]

任何一个条件变化都不再是这里定义的等权参数平均，因此 v0 不提供覆盖开关。

### 3.4 等工作量约束

所有 island 必须具有相同的：

- `G`、`K`、`U`；
- prompt 分配规则和每轮 batch 结构；
- GRPO、KL、clip、reward normalization 配置；
- LoRA 配置；
- optimizer 类型、常量 learning rate、weight decay 和 gradient clipping；
- trainer 并行拓扑。

trajectory 长度和 action token 数可以自然不同，但 merge 仍是 island 等权。
若某个 island 无法产出规定的 `G/K/U`，该轮不能提交一个“较小工作量”的更新，
只能继续恢复或让 run 失败。

### 3.5 Optimizer 合同

v0 在每次安装 global policy 后使用全新的本地 optimizer 状态：

- first/second moments 为零；
- optimizer step counter 为零；
- FP32 master parameters 与刚安装的 global LoRA 一致；
- learning rate 为固定常量，不运行 warmup 或依赖进程生命周期的 scheduler。

这使 learner 重启与正常轮次边界具有相同语义。不能用
`optimizer.state.pop(param, None)` 作为 Megatron `DistributedOptimizer` 的
实现，因为它可能遗漏分片 state 和 master parameters。实现必须调用经验证的
Miles/Megatron reset/rebuild 路径，并在 reset 后立刻重新导出 LoRA 与
`θ_v` 做逐 tensor 校验。

---

## 4. v0 拓扑与职责边界

### 4.1 强制拓扑

```text
model kind: causal LM
supported dense families: Llama / Qwen2 / Qwen3
tuning: LoRA only
Miles trainer: Megatron
TP: 1
PP: 1
EP: 1
DP: 1
rollout: island 内 SGLang
global fragments: 1
wire dtype: f32
```

模型边界采用 fail-closed：

- 只接受 Transformers 精确的 `LlamaConfig/LlamaForCausalLM`、
  `Qwen2Config/Qwen2ForCausalLM`、`Qwen3Config/Qwen3ForCausalLM` 配对；
  `architectures` 必须精确声明对应 causal-LM class，remote-code 自定义 subclass
  即使伪装相同 `model_type` 也拒绝；
- Llama、Qwen2、Qwen3 都固定使用 SwiGLU、RMSNorm 和 dense decoder；Qwen2
  显式映射 QKV bias，Qwen3 显式映射 QK layernorm；构造 meta model 后还要核验
  实际 model、RMSNorm 和 gated dense MLP class；
- `rope_theta` 必须是正整数；同时兼容 Transformers 5 的 `rope_parameters` 和
  旧版 `rope_scaling` 字段；
- 除默认 RoPE 外，只接受标准 Llama 3 语义：正 `factor`、
  `low_freq_factor=1`、`high_freq_factor=4`、
  `original_max_position_embeddings=8192`；
- 拒绝 MoE、多模态、量化、MLP bias、非 Qwen2 attention bias、sliding-window
  attention、非 SiLU/SwiGLU、未知架构和其他 RoPE 语义。

TP/PP/EP/DP 中任何一个大于 1 都需要额外的 canonical gather/scatter 和真实设备
验证，不属于本规格。

### 4.2 所有权

| 状态/动作 | 唯一 owner |
| --- | --- |
| rollout admission、trajectory、工具状态、reward、GRPO batch | Miles |
| Megatron optimizer 与 island-local `θ_{i,v}` | Miles |
| logical island ID 与恢复 | Yeto launcher |
| committed `θ_v`、version、policy hash、checkpoint | Yeto syncer |
| canonical LoRA 命名、layout 和 Megatron↔PEFT 映射合同 | 固定 Miles 上的 Yeto RL 窄适配层 |
| 最终标准 PEFT artifact | Yeto RL exporter |

island-local LoRA 和 optimizer checkpoint 都不是全局权威状态。只有 syncer 已
持久化的 committed policy 可以推进 policy version。

---

## 5. Canonical LoRA 与 run identity

### 5.1 Canonical tensor 规范

每个 trainable tensor 必须有：

```python
@dataclass(frozen=True)
class CanonicalTensorSpec:
    name: str
    shape: tuple[int, ...]
    numel: int

@dataclass(frozen=True)
class CanonicalLoraState:
    policy_version: int
    layout_fingerprint: str
    policy_hash: str
    tensors: Mapping[str, torch.Tensor]
```

规范如下：

- `name` 使用可逆到标准 PEFT state dict 的稳定名称；
- tensor 按 canonical name 升序排列；
- tensor 在 hash/传输边界统一为 CPU、contiguous、little-endian f32；
- 不包含 base weights、optimizer state 或 runtime rank 前缀；
- 名称、shape、numel 和顺序全部进入 semantic layout fingerprint；
- NaN/Inf 在生成 delta 前直接失败。

policy hash 定义为：

```text
SHA256(
  "yeto-rl-policy-v1\0" ||
  layout_fingerprint ||
  canonical ordered f32 tensor bytes
)
```

version 与 hash 成对记录；hash 本身不包含 version，因此两个数值完全相同的
policy 可以具有相同 hash。`layout_fingerprint` 使用 protocol 中的原始 32 bytes，
tensor bytes 使用 canonical 顺序的 IEEE-754 little-endian f32 表示。

### 5.2 显式 AVG layout

RL 不能调用默认 `build_layout(..., 1)`。它必须直接构造：

```python
ordered = sorted(tensor_specs, key=lambda item: item.name)
layout = FragmentLayout([
    Fragment(
        merge_mode=MERGE_AVG,
        tensors=[(item.name, item.numel) for item in ordered],
        identity_shapes={item.name: item.shape for item in ordered},
    )
])
```

所有 island 的 HELLO 必须具有相同 `layout_fingerprint`。现有 protocol v4 已将
该 fingerprint 纳入 session identity，可继续复用。

### 5.3 静态 run manifest

launcher 在花费云资源前解析并持久化 canonical JSON manifest，至少包含：

- schema version 与 run ID；
- base model resolved identifier/revision；
- tokenizer/chat template identity；
- LoRA rank、alpha、targets、bias 和 canonical mapping version；
- `M/G/K/U` 与完整 GRPO/optimizer 配置；
- dataset/prompt assignment revision；
- reward function source hash 与配置；
- 可选完整 trajectory generator 的 callable 与 source hash；
- Yeto source hash、Miles commit 和容器 digest；
- trainer/rollout 拓扑。

syncer 接收 `run_manifest_sha256`，并把它与 HELLO 中的
`layout_fingerprint` 一起写入每个 authoritative checkpoint。resume 和 export
都必须同时匹配这两个 identity；仅比较 fragment numel 不足以证明兼容。
manifest 编码固定为 UTF-8、key 排序、无非语义空白的 canonical JSON，禁止直接
对不同实现默认输出的 JSON 文本求 hash。

### 5.4 Megatron↔PEFT 映射验收

映射不是字符串替换猜测。固定 Miles/Megatron 版本必须通过以下 round trip：

```text
标准 PEFT LoRA
→ canonical state
→ Miles/Megatron apply
→ Miles/Megatron export
→ canonical state
→ 标准 PEFT LoRA
```

要求 tensor 名集合和 shape 完全一致，f32 tensor 值逐元素一致，并在干净进程中
成功加载标准 PEFT adapter。当前 PEFT layout 构造、checkpoint 导出、独立进程
hash 复核和 tiny Llama 的 `PeftModel.from_pretrained()` 已有 CPU 自动化覆盖；
Megatron FP32 master 的真实 apply/export round trip 仍属于第 13.4 节，完成前不能
宣称真实 Miles 产出的 artifact 已通过验收。

---

## 6. Rust syncer 的 `rl-strict-avg` 模式

这是 v0 必需的原子协调模式，不是普通 sync 参数的松散 preset。普通 SFT/
diffusion 路径保持现有行为。

### 6.1 强制配置

| 配置 | strict 值 |
| --- | --- |
| fragment layout | 1 个显式 `MERGE_AVG` fragment |
| pipeline | `1` |
| roster | 固定逻辑 ID `0..M-1` |
| quorum | 恰好 `M`，不得取 `min(M, live)` |
| grace | 不参与完成条件 |
| sync interval | `0` |
| learner counters | `local_step=step, c_steps=1, c_tokens=1`，server 强校验 |
| outer LR | `1.0` |
| outer momentum | `0.0` |
| delta correction | `none` |
| learner merge alpha | `0.0` |
| wire dtype | `f32` |
| checkpoint | 初始化 v0 和每次 commit 都写 |

top-level `--training-mode rl` 生成这组内部配置。用户若同时传入冲突的通用
sync 参数，CLI 必须报错；v0 不提供绕过 strict 语义的 experimental flag。
strict 模式始终满足 `global_step == fragment_version == v`，下一次 PULL 的
`step` 必须是 `v+1`。
strict syncer 在 HELLO 时也必须主动验证 fragment 数、merge mode 和 wire dtype，
不能只依赖 bridge 正确构造配置。

### 6.2 固定逻辑成员

当前 scheduler 按连接 generation 冻结成员，retry 时又会从 live groups 重建
membership。strict 模式改为：

- roster 在 run 开始时固定为 learner ID `0..M-1`；
- generation 只表示传输连接，不改变逻辑 membership；
- round 的 accepted results 按 `learner_id` 保存；
- learner 断线不删除已接受结果；
- 同一 learner 新 generation 重连后，syncer 先发送 committed global state，
  再重发当前未完成 round 的同一 PULL；
- 只有 `M` 个不同逻辑 ID 都提交合法结果时才 merge；
- launcher 无法恢复任一逻辑 ID 时，整次 run 失败，不能继续使用 survivors。

终态 finalization 同样等待固定 roster，而不是只等待 final cut 时在线的 groups。

### 6.3 严格 push 接受规则

对当前 round `(step=t, fragment=0, attempt=a, base_version=v)`，push 必须满足：

- learner ID 属于固定 roster，且该连接 generation 确实收到过本 round 的 PULL；
- `step/fragment/attempt` 与当前 permit 完全一致；
- `base_version == v`，不是小于等于；
- `local_step == step`；
- `c_steps == 1 && c_tokens == 1`；
- payload 长度、dtype 和全部数值合法；
- 每个 logical learner 只有一个结果。

同一 learner 的重复 push：

- push 字段和 payload digest 完全相同：幂等忽略；transport generation 不进入
  result digest；
- 任一字段或 payload 不同：protocol violation，strict run 失败。

已完成旧 round 的迟到重复包可以丢弃并记录，不得影响新 round；当前 round 的
错误 base/version 不能像普通模式一样作为 stale delta 接受。

重连时，旧 generation 已在传输中的合法 push 和新 generation 的 cache 重发可能
竞态。syncer 应记录每个 learner 在本 round 获得过 permit 的 generation 集合，
接受其中第一个合法结果；后续结果按上述 digest 规则判定幂等或冲突。不能因新
generation 已成为 current 就丢掉旧 generation 在合法 permit 下完成的结果。

### 6.4 长 round 与重试

strict 模式不能沿用“900 秒后清空 pushes 并换 membership/attempt”的行为：

- 等待期间不清空已接受的 push；
- 不因 wall-clock timeout 缩小 roster；
- 可以周期性向缺失 learner 重发同一 PULL；
- `HEARTBEAT` 不改变 round 或 version；如记录它，只用于 liveness/诊断；
- 可选 `rl_round_timeout_s` 是整轮失败上限，默认 `0` 表示不设置算法层上限；
  达到上限时让 run 失败，而不是用 partial quorum merge。

协议违例或显式 RL round/finalization deadline 到期属于 run 级永久失败。syncer
先原子写入与 `run_manifest_sha256` 绑定的相邻 `.fatal` marker，再发送
`MSG_ERROR` 并退出；launcher 在重启 syncer 前检查该 marker 并终止固定 roster。
进程被杀、节点丢失或 checkpoint 写入失败不会生成 fatal marker，仍按最后一个
authoritative checkpoint 恢复，因而不会把正常 crash recovery 错判为协议失败。

同样的规则适用于终态 ACK：不能使用普通模式的 900 秒 quorum timeout 缩小或
放弃 final roster；只能等待恢复，或由 RL run 的整体失败上限终止。

bridge 必须缓存本地结果，因此 PULL 重发或 syncer 重启不会强制重复昂贵的
rollout/train。

### 6.5 确定性聚合

syncer 按 `learner_id` 升序把 `M` 个 f32 delta 送入 AVG，不能依赖 Rust
`HashMap` 迭代顺序。离线 oracle 使用相同顺序和 f32 累加规则。

### 6.6 原子 commit 顺序

初始化 `θ_0` 时，只有在固定 `M` 个 learner 以同一 session identity 完成 HELLO
且 learner 0 的 INIT 已到齐后，才能持久化 checkpoint v0 并广播；HELLO 与 INIT
的实际到达顺序不影响该条件。每个后续 policy 遵循：

```text
验证固定 roster 的 M 个合法 push
→ 按 learner_id 确定性计算 candidate θ_{v+1}
→ 计算 policy_hash_{v+1}
→ 原子写 authoritative checkpoint v+1
→ checkpoint 成功后发布 BCAST v+1
→ 记录 commit event
```

checkpoint 失败必须终止 syncer，不能继续广播内存中的 candidate。这样：

- crash 在 checkpoint 前：恢复到 v，learner 重发/重算本轮；
- crash 在 checkpoint 后、广播前：恢复到 v+1，并重新广播 v+1；
- crash 在部分广播后：恢复后仍广播相同 version/hash，已收到者幂等忽略，
  未收到者补齐；
- 永远不会出现 learner 已接受 v+1、syncer 却只恢复出 v 的合法路径。

最后一轮的 checkpoint 只表示 final policy 已 committed，不表示整次 run 已完成。
strict syncer 必须在固定 roster 全部完成 final apply/ACK 之后才原子写 final marker。
若在 ACK 阶段或写 marker 前 crash，resume 重新发送 final cut 并重新收齐 ACK，
不能提前把 checkpoint 标记为可发布 artifact。
终态顺序固定为 `send final cut → collect M ACKs → write final marker → send
SHUTDOWN`。

resume 时若 checkpoint 已到配置的最终 version、但 final marker 不存在，只能
重做 final cut/ACK，不能再启动训练 round；若匹配的 marker 已存在，则 run 已
完成。syncer 不再推进 version，也不重写 checkpoint/marker；它只可向重新连接的
bridge 幂等重放同一 final cut，重新收取该残留连接的 ACK 并发送 SHUTDOWN，不能
重新等待历史固定 roster 全部同时在线；launcher 直接以绑定本次 manifest 的 final
marker 为完成事实，停止可能仍未退出的 island 进程。

### 6.7 RL checkpoint envelope

现有 checkpoint 格式不能只靠 fragment numel 恢复 strict run。RL checkpoint
至少额外持久化并校验：

```text
schema / mode = rl-strict-avg
run_manifest_sha256
layout_fingerprint
fixed roster size M
global version
committed policy hash
fragment version / flat f32 params
ledger
```

使用临时文件、file fsync 和原子 rename；恢复时任何 mode、manifest、layout、
roster 或 policy hash 不一致都 fail closed。普通 checkpoint 的兼容读取行为不因
此模式改变。

### 6.8 Wire protocol 决策

v0 不增加 message type：

- `PULL_REQ` 提供 `step/attempt` permit；
- `PUSH_FRAGMENT` 已携带 base version 和本地计数；
- `BCAST_FRAGMENT` 发布 committed policy；
- `FINAL_FRAGMENT`、`FINAL_MANIFEST`、`FINAL_ACK` 完成终态覆盖与确认；
- 现有 `MSG_ERROR` 可报告 strict protocol violation。

local RL stats、trainer/SGLang apply hash 写 island event log；它们不参与 merge，
因此不塞入 tensor wire protocol。

---

## 7. Yeto RL bridge 状态机

### 7.1 状态机

```text
BOOTSTRAP
  连接 syncer；learner 0 提交初始 LoRA；等待 committed θ_0
        │
        ▼
APPLY_GLOBAL(v)
  pause admission → drain/cancel → apply trainer → reset optimizer
  → verify trainer hash → apply SGLang → verify rollout version/hash
        │
        ▼
WAIT_PERMIT(v)
  等待 PULL(step=v+1)；重复 permit 幂等；提前到达的下一 permit 可缓存
        │
        ▼
COLLECT(v)
  收集 G 个完整、每组 K 条、全部标记为 v 的 GRPO groups
        │
        ▼
LOCAL_TRAIN(v)
  固定配置执行 U steps；SGLang 仍保持 θ_v
        │
        ▼
EXPORT_AND_CACHE(v)
  pause admission → export θ_{i,v} → validate → cache d_{i,v}
        │
        ▼
PUSH_AND_WAIT(v)
  按最新 permit 发送或重发 cache；等待 committed θ_{v+1}
        │
        └──────────────────────────────→ APPLY_GLOBAL(v+1)

任意等待点收到 FINAL_MANIFEST
  → FINAL_APPLY → VERIFY → FINAL_ACK → EXIT
```

PULL 与大 tensor BCAST 可能跨 socket 重排。现有 PULL payload 不携带
`base_version`；strict 单 fragment 模式规定 step `t` 的 base 必须恰好是 `t-1`。
bridge 可以先缓存 permit，但只有在本地已安装并验证 version `t-1` 及其派生
policy hash 后才能开始本轮。对同一 `(step, attempt)` 的重复 PULL 只能复用正在
执行的 round 或已缓存结果，不能再启动一次 rollout/train。已完成 step 的迟到
permit 幂等丢弃；未来 step 只能等待其前一版 BCAST，不能跨版本执行。

### 7.2 Miles runtime 的语义接口

不要先绑定未经验证的方法名。Yeto 只依赖以下粗粒度语义：

```text
initialize()
apply_global_policy(canonical_state)
run_local_round(policy_identity=(version, hash), groups=G, samples=K, optimizer_steps=U)
export_local_policy()
cancel_or_drain_rollouts()
read_trainer_policy_identity()
read_rollout_policy_identity()
shutdown()
```

`run_local_round` 返回时必须满足：

- 被训练数据全部属于请求的 policy version；
- 本地 optimizer 已完成恰好 `U` steps；
- rollout admission 已暂停；
- trainer 是 local policy，但 SGLang 仍是输入的 global policy；
- 没有后台 `update_weights()` 会随后把 local policy 覆盖到 SGLang。

因此，单纯在 Miles `train()` 之后加一个 hook 不足以满足合同；外部 policy
boundary 还必须控制 rollout admission、任务 drain/cancel 和原有 trainer→SGLang
更新路径。

当前实现为 `yeto.rl.miles.MilesIslandRuntime`。它在 Miles actor 被
`ray.remote` 包装前安装固定的 export/apply/identity/optimizer-counter 方法，
直接使用 Megatron-Bridge conversion task 和 FP32 optimizer master；每次全局
apply 都重建 optimizer/scheduler，再同步 compute copy 和 SGLang。该适配层只对
固定 commit 生效，不在 Miles checkout 中打补丁，也不对未知上游版本做兼容猜测。
固定的 `yeto.rl.miles.generate_rollout` 包装 Miles 默认 rollout function，在 Miles
递归展平 train data 之前强校验恰好 `G` 个 group、每组恰好 `K` 个
`COMPLETED/TRUNCATED` 终态 trajectory；custom generator 返回嵌套的多 sample
leaf 会直接失败，不能靠展平后的总数通过合同。

### 7.3 Policy version 规则

每条 trajectory 在生成请求进入 SGLang 时记录 `(policy_version,
policy_hash)`，该 identity 随样本进入 group。group close 时必须满足：

```python
identities == {(expected_policy_version, expected_policy_hash)}
```

否则该 group 不得进入训练，strict run 失败。不能根据“group 开始时的当前全局
变量”倒推版本；必须记录实际 dispatch 使用的版本。

达到本轮 `G` 个 group 后：

- 已选 group 完成训练；
- 未完成 group 取消；
- 完整但未选中的 group 在 version 变化时丢弃；
- completed/in-progress queue 都不能跨到 v+1。

island 进程重启时可以直接丢弃未完成或未使用 group，从当前 committed version
重新收集；v0 不要求恢复环境内部的 partial trajectory。

### 7.4 非权威本地结果缓存

在首次 push 前，bridge 原子保存：

```text
run_manifest_sha256
learner_id
layout_fingerprint
base_version / base_policy_hash
target step
canonical f32 delta
delta_sha256
local round stats
```

缓存用于幂等重发，不代表 global commit：

- 收到相同 base 的重发 PULL：使用缓存并带上当前 permit 的 attempt 发送；
- syncer 从 v 恢复：使用缓存重发；
- 收到并成功应用 v+1：删除 v 的缓存；
- authoritative version 已大于缓存 base：丢弃缓存；
- identity/hash 不匹配：不得发送，删除后重算；
- 没有缓存的 learner 重启：应用当前 global policy 并重做该轮。

由于 `PUSH_FRAGMENT` 没有独立 ACK，缓存只能在更高 committed version 已应用后
清理。

---

## 8. 故障与恢复语义

| 故障点 | 恢复行为 |
| --- | --- |
| rollout/local train 中网络断开 | Miles 继续当前 base version 的本地工作；bridge 重连后等待/接收同一 permit |
| learner 在本地 cache 前退出 | launcher 用同一 learner ID 恢复；覆盖为 syncer 当前 global policy，重做该轮 |
| learner 在 cache 后、push 前退出 | 恢复后校验 cache，并对同一 base 重发 |
| learner push 后退出 | syncer 已收结果则保留该 logical ID 的结果；若 syncer 也重启，learner 从 cache 重发 |
| syncer 在 commit checkpoint 前退出 | 从 v 恢复，向固定 roster 重发本轮 PULL；不保留内存中的 partial merge |
| syncer 在 checkpoint 后、广播前退出 | 从 v+1 恢复并广播 v+1 |
| syncer 广播给部分 learner 后退出 | 恢复后重发相同 version/hash；客户端按相同 version/digest 幂等处理 |
| learner 收到 BCAST、apply 中退出 | 恢复后重新接收 committed policy，完整覆盖 trainer/optimizer/SGLang |
| 某 logical learner 超出恢复预算 | 整次 strict run 失败；不缩 roster、不提交 partial average |
| manifest/layout/base revision 不匹配 | 启动或 resume 立即失败 |

任何恢复路径都先以 syncer checkpoint 为准覆盖本地 trainer，不从未 committed 的
local model 继续训练。

---

## 9. CLI 与 launcher

### 9.1 最小 public 配置

新增独立训练模式和 RL 必需输入：

```text
--training-mode {sft,rl}          # 默认 sft
--rl-runtime miles
--rl-global-rounds N
--rl-groups-per-island-round G
--rl-samples-per-group K
--rl-local-optimizer-steps U
--reward-function package.module:function
--rl-generate-function package.module:function  # 可选
--learner-image <immutable digest>
--rl-round-timeout-s 0
--trust-remote-code                    # 固定 Miles 必须显式确认
```

模型、LoRA、数据、GPU fleet 和 provenance 继续复用已有参数。不要再增加一个与
`--learner-image` 重复的 RL image 参数。
`--rl-global-rounds N` 一一映射为 strict syncer 的 `total_steps=N`。

### 9.2 validation

`training-mode=rl` 启动前必须验证：

- causal LM、LoRA、Miles runtime；
- `M/G/K/U > 0`；
- TP=PP=EP=1，所有 island 拓扑一致；
- model、dataset、Miles、image、reward source 和可选 generate source 已固定
  revision/hash；
- reward callable 在 controller preflight 环境中可导入且为 callable；learner 在
  固定 image 内、校验 source hash 后再次导入，任一阶段失败都在训练前终止；
- strict sync 派生配置与第 6.1 节完全一致；
- authoritative checkpoint 路径可用；
- 不存在与 strict 值冲突的通用 sync flag。

CLI 打印并持久化最终 resolved manifest 后才能创建云资源。

### 9.3 launcher 分支

launcher 只增加清晰分支：

```python
if args.training_mode == "rl":
    return make_miles_island_task(...)
return make_existing_learner_task(...)
```

RL task 启动固定 Miles runtime，并把 learner ID、syncer address、resolved
manifest 和 source hash 传入。不要把 Miles 特例继续塞进 causal-LM torch/
Megatron learner 的参数拼接逻辑。

`--rl-generate-function` 使用 `package.module:function`，launcher 对其 Python
源文件计算 SHA256 并写入 manifest；learner 在导入前复核 source hash，再把固定
callable 转为 Miles 所需的 dotted path。未提供时使用固定 Miles 的默认完整
trajectory generator。

RL strict 模式下，`FleetController` 对 learner 的恢复状态只能是
`running ↔ recovering → failed-run`，不能进入“abandoned 后继续”。syncer 仍从
authoritative checkpoint 恢复。

---

## 10. Checkpoint、终态与导出

### 10.1 权威层级

```text
syncer committed checkpoint
  = 唯一 authoritative global policy

island local-result cache
  = 可丢弃、可重算的幂等传输缓存

Miles rollout / optimizer checkpoint
  = island 内部临时状态，不得推进 global version
```

### 10.2 终态协议

bridge 必须完整实现现有终态流程：

1. 看到 `client.finalizing` 后停止新 rollout；
2. `wait_for_final_fragments()` 收齐 `FINAL_MANIFEST` 指定的唯一 fragment；
3. 校验 manifest version；使用本次 session 的 layout fingerprint 与收到的 raw
   f32 fragment 计算 canonical policy hash；
4. 完整覆盖 trainer，重置 optimizer，完整更新 SGLang；
5. 验证 trainer/rollout policy identity 均等于 final identity；
6. 调用 `acknowledge_finalization()`；
7. 等待正常 shutdown 并退出。

任何 apply/hash 校验失败都不能发送 `FINAL_ACK`。strict syncer 等待固定 roster
的 ACK；缺失成员意味着 run 未正常完成。
bridge 的 final wait 使用 RL run 的整体 deadline；不能让普通
`SyncerClient.finalization_timeout` 先把一个仍可恢复的 strict run 结束。
`--rl-round-timeout-s=0` 映射为无穷 deadline，而不是任意大的有限秒数；普通 SFT
仍保留原有 900 秒默认值。

### 10.3 专用 RL exporter

不能直接复用当前只按 fragment numel 校验的通用 exporter。当前已实现专用
`yeto.rl.export`：

```text
读取 strict checkpoint + resolved run manifest
→ 验证 manifest SHA、layout fingerprint、policy hash 和 ACK 后写入的 final marker
→ 从固定 base revision 与 LoRA config 构造标准 PEFT adapter
→ 重建 canonical tensor spec，并要求名称/shape/fingerprint 完全相同
→ 应用 checkpoint 的 ordered f32 values
→ 写 adapter_model.safetensors + adapter_config.json + provenance
→ 在干净进程重新加载并复核 canonical policy hash
```

最终 artifact 从 authoritative checkpoint 生成，不依赖某个 surviving learner 的
本地 save。导出失败不改变训练 checkpoint 的权威性，但不能把 run 标记为 artifact
完成。

---

## 11. 最小可观测性与失败条件

### 11.1 每个 island 每轮记录

```text
run_id / learner_id
base_version / base_policy_hash
groups / trajectories / optimizer_steps
rollout identity set
local policy hash / delta hash / delta norm
trainer applied identity / rollout applied identity
rollout_seconds / train_seconds / sync_wait_seconds
cache resend count
```

reward mean/std、action tokens 和 KL 可作为训练指标记录，但不参与 merge。

bridge 将上述字段以 canonical JSONL 写入
`~/yeto-rl-cache/events.jsonl`。push attempt 和首次 push 时间与本地 result cache
一起原子持久化，因此进程恢复后的 resend count 和 sync wait 仍可追踪；最后一轮
即使直接通过 final cut 交付，也会在清理 cache 前记录 committed round。

### 11.2 syncer 每次 commit 记录

```text
run_manifest_sha256 / layout_fingerprint
base_version / committed_version
fixed roster / responder learner IDs
ordered delta digests
committed policy hash
checkpoint completion / broadcast enqueue result
```

strict syncer 将这些字段写入既有 `--event-tape`；`broadcast_queued_to` 明确表示
成功进入各连接发送队列的 learner ID，而不是虚构网络已送达。后续 PULL 和最终
ACK 承担实际 apply 的协议证明。

### 11.3 必须 fail closed

- mixed rollout policy identity；
- 少于 `G/K/U` 却提交结果；
- current round 的 stale/future base；
- 同一 logical learner 的冲突重复 push；
- 非 finite tensor/delta；
- manifest/layout/policy hash mismatch；
- trainer 或 SGLang apply identity mismatch；
- fixed roster 中任一 learner 被 abandon；
- checkpoint 未成功就尝试 broadcast；
- final cut 未完整应用就 ACK。

---

## 12. 必需代码改动

只实现下列 v0 边界：

| 组件 | 最小改动 |
| --- | --- |
| Rust syncer | 增加 `rl-strict-avg` 模式；固定 logical roster；严格 base/weight/duplicate 校验；长 round 重发；按 learner ID 聚合；commit-before-broadcast；RL checkpoint/final/fatal markers |
| Yeto RL core | canonical LoRA/PEFT mapping、显式 AVG layout、policy hash、result cache、`SyncerClient` bridge、终态处理 |
| CLI/launcher | `training-mode=rl`、resolved manifest、strict 参数派生、Miles task、禁止 fleet shrink、按 final/fatal marker 收口 run |
| 固定 Miles 窄适配层 | admission/drain、禁止 local publish、canonical export/apply、完整 optimizer reset、rollout identity、SGLang apply confirmation |
| RL exporter | strict checkpoint 到标准 PEFT LoRA 的名称/shape/hash 强校验导出 |

不修改现有 SFT loss 或把 RL 数据结构加入 `yeto/learner.py`、
`yeto/megatron/learner.py`、`yeto/losses.py`。

---

## 13. 实施与验证顺序

以下顺序只覆盖 v0 的依赖关系。

本分支当前自动化快照：Python `744 passed, 4 skipped`；Rust `61 passed`；
`python -m compileall -q yeto`、`cargo fmt --check` 和 `git diff --check` 通过。
这些结果不包含真实 GPU 验收。

### 13.1 Canonical 与算法合同

状态：**已实现并通过自动化验证**。

实现并验证：

- 显式单 AVG fragment，证明不经过 RDA；
- canonical name/order/shape/fingerprint/policy hash 稳定；
- PEFT↔canonical round trip；
- 两个 tiny local states 的手工等权 oracle；
- strict CLI 对每个冲突参数（包括 `merge_alpha`）fail closed；
- 精确 Transformers config/model/architecture 白名单拒绝 remote-code subclass
  伪装，并核验 RMSNorm/gated dense MLP 结构。

退出条件：给定相同 canonical tensors，所有进程生成完全相同 layout/policy
identity；给定两个本地 state，计算结果符合第 3.3 节。

### 13.2 Strict syncer 与恢复

状态：**代码已实现，关键路径已有自动化覆盖；完整 crash-injection 矩阵尚未全部
执行**。现有覆盖包括 strict roster/permit/counter/duplicate 规则、确定性聚合、
checkpoint/marker identity、fatal `MSG_ERROR`、两个 FakeMiles island 的真实 Rust
进程闭环、run-bound fatal marker 阻止恢复继续推进，以及 final marker 后只有一个
残留 learner 重连时的终态重放与 launcher 清理。

使用真实 Rust syncer 和 raw/Fake learner 验证：

- 只有 `M-1` 个 push 时永不 merge；
- learner 断线/新 generation 重连不改变 logical roster；
- 把普通 retry interval 配成很短的测试值，证明 strict round 不清空 accepted result；
- stale/current-conflict push 失败；
- identical duplicate 幂等，conflicting duplicate 失败；
- 确定性 learner-ID 顺序 AVG；
- manifest/layout 不匹配 resume 失败；
- 在 merge 前、checkpoint 前、checkpoint 后、部分 broadcast 后注入 crash，
  恢复后的 version/hash 符合第 6.6 节；
- fixed roster 全部完成 final apply/ACK 后才结束。

退出条件：任何故障注入都只得到旧 committed policy 或新 committed policy，绝不
出现已发布但不可恢复的 version。

### 13.3 Bridge 与 FakeMilesRuntime

状态：**核心闭环已实现并通过自动化验证**。两个 FakeMiles island 已经通过真实
Rust syncer 连续执行两轮，验证手工 f32 平均、final apply/ACK、authoritative
checkpoint、最终 marker、重启重放和 island JSONL 观测字段。其余逐故障点注入与
第 13.5 节一起保留为端到端验收，不据此宣称真实 Miles 可用。

固定 Miles rollout wrapper 已在上游递归展平前验证 `G×K` 的 group 边界、终态和
单 trajectory leaf，避免 custom generator 通过相同展平总数掩盖错误 group；
`U` 继续由 Megatron optimizer step counter 在训练后强校验。

测试必须使用真实 Rust syncer，不能只用 FakeSyncer，否则会掩盖 membership、
timeout 和 checkpoint ordering 问题。验证：

- 两个 FakeMiles island 的 manual average；
- PULL/BCAST 重排；
- cache 后断线、syncer 重启和重复 PULL；
- apply 前 admission 已暂停；
- group mixed version 失败；
- local policy 从未发布给 rollout runtime；
- terminal manifest → apply → verify → ACK 闭环。

退出条件：两轮连续 global update 后，两个 island 的 trainer/rollout identity 都
等于 syncer committed identity。

### 13.4 固定 Miles commit 的单 island 验证

状态：**未执行；需要固定 learner image 和真实 GPU**。这里的 `M=1` 是内部
runtime 验收 harness，不放宽 public RL launch 对 `M≥2` 的约束。

在进入多 island 前逐项证明第 1.2 节能力：

- 完整 group 与 policy identity 记录；
- trainer local update 不泄漏到 SGLang；
- distributed optimizer reset 后 moments/step/master params 正确；
- canonical apply/export 无损；
- trainer 与 SGLang 对固定输入的 logprob 在定义的数值容差内一致；
- process restart 后能从 committed global policy 重做或重发当前 round。

退出条件：`M=1` 时 Yeto 结果等于 Miles local LoRA，且 final artifact 可由标准
PEFT 加载。

### 13.5 双 island 端到端

状态：**未执行；需要两个真实 GPU island**。

固定随机种子和小模型，保存两个 local states，离线按相同 f32 顺序平均，并与
syncer checkpoint 比较。随后覆盖以下故障点：

```text
rollout 中
local train 后、cache 前
cache 后、push 前
push 后、commit 前
checkpoint 后、broadcast 前
global apply 中
final apply/ACK 中
```

每个 case 必须满足：不缩 roster、不重复计入 learner、不跳/回退 version、所有
成功恢复的 island 最终 hash 相同。

退出条件：第 14 节全部满足，且现有 Python/Rust test suites 无 regression。

---

## 14. Definition of Done

当前状态：**未达到**。以下条目仍是发布验收条件，不能用源码审计、FakeMiles
或 CPU 测试替代真实 Miles GPU 结果。

Yeto RL v0 只有同时满足以下条件才完成：

1. 一条命令启动固定 `M≥2` 个 Miles island 和 strict syncer；
2. 每个 island 使用同一 committed policy 完成规定的 `G/K/U`；
3. 任一训练 group 都不存在 mixed policy identity；
4. syncer 只在固定 `M` 个合法 push 到齐后 commit；
5. global LoRA 与离线、确定性 f32 等权平均 oracle 一致；
6. 每次 broadcast 前对应 checkpoint 已持久化；
7. learner/syncer/network 故障恢复不导致 roster shrink、duplicate merge、version
   rollback 或 silent divergence；
8. 每次 global apply 后，所有 island 的 trainer policy hash 与 committed hash
   一致，SGLang 报告相同 version/hash，并通过 trainer/rollout logprob parity；
9. terminal final manifest 被固定 roster 全部完整应用和确认；
10. authoritative checkpoint 能导出并重新加载标准 PEFT LoRA，canonical hash
    不变；
11. run manifest、layout 和 artifact provenance 完整且 resume/export fail closed；
12. 现有 SFT、diffusion 和普通 syncer 行为及测试无 regression。

在固定 Miles commit 未通过第 13.4、13.5 节前，本方案只能声明 Yeto 侧实现、
固定源码审计和 FakeMiles/CPU 验证完成，不能宣称 RL v0 已完成或已具备生产可用性。
