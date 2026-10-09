# Design: secret-handling-hardening

## Context

见 proposal.md。代码现状（agentenv/main cbf10edd）：

- syncer 只有 elastic 模式校验 HMAC（`elastic.rs` 的 `seal/open`，只覆盖 elastic 消息类型）。legacy 的 HELLO 没有认证。
- `admit_session` 让第一个被接纳的 HELLO 确定会话契约；elastic 的 `backend_identity` 同样由第一个 JOIN 钉定。
- 岛的会话契约 = sha256(`yeto-rl-session-contract-v2\0` ‖ 布局指纹 ‖ 岛契约)。岛契约 = `island_contract_sha256(后端身份, 学习率调度)`，elastic JOIN 里发的也是它。
- `HARNESS_PASSTHROUGH_ENV` 含 `MODAL_TOKEN_ID/SECRET`，经普通 `envs=` 进学习岛。

## Decisions

### D1 legacy 也要求 HMAC（不改默认模式）

子 agent 代拍板 + 理由。比较了两种做法：

| 做法 | 改动量 | 兼容性 |
|---|---|---|
| A. legacy 也要求 HMAC | Rust 约 80 行（HELLO 末尾 32 字节 MAC + 启动检查）；Python `protocol.py` 约 25 行；launcher 改 `island_hmac_secret` 一个函数 | 调度语义不变（仍是严格平均、每岛必到）。旧岛二进制连新 syncer 会被拒，但岛与 syncer 总是同一次发布，一起换。 |
| B. 默认改成 elastic | CLI 默认值一行，但 elastic 改变调度语义（容量加权、迟到折扣、加入/离开） | 所有默认运行的数值行为改变；elastic 的张量合并路径与 legacy 不同，已有证据和基线不再可比；`--rl-island-scheduling legacy` 仍无认证，问题没消除。 |

选 A：最少改动、不改训练数值，并且真正堵住 legacy 的口子。B 不能单独解决问题。

实现要点：

- MAC = HMAC-SHA256(密钥, `yeto-hello-mac-v1\0` ‖ HELLO 正文)，附在 HELLO 末尾。复用 `elastic::hmac_sha256`；比较用常数时间。
- syncer 有密钥就要求 MAC。没有密钥时，legacy syncer 拒绝启动，除非 `--allow-unauthenticated-islands` 或 `YETO_SYNCER_ALLOW_UNAUTHENTICATED=1`（只给本机测试与基准脚本用）。
- 密钥与 elastic 共用 `YETO_ISLAND_HMAC_KEY`。岛侧 `SyncerClient` 默认从该环境变量读取。
- RL launcher：legacy 且启动环境没有密钥时，生成 `secrets.token_hex(32)` 并写回本进程 `os.environ`。这样 syncer 任务、head 任务、各岛（sky secrets 或 Modal Secret）拿到同一个值。elastic 仍要求用户提供（行为不变）。
- DATA_HELLO 不带 MAC。它只能挂到一个已认证 HELLO 的 connection_generation 上；该值是 64 位随机数。

已知限制（不在本 change 内）：

- 没有 TLS。能监听链路的人可以重放 HELLO 或篡改后续帧。MAC 只挡住"随便连上来"的攻击。
- SFT 的 syncer 命令显式带 `--allow-unauthenticated-islands`：SFT 岛任务没有密钥通道，补齐要改 SFT 的三处任务构造，留作后续。
- `yeto/rl/ssh_harness.py`（直连 SSH 编排）同样显式带该开关，原因同上。
- `scripts/` 下自己起 syncer 的本地基准脚本没改，需要设 `YETO_SYNCER_ALLOW_UNAUTHENTICATED=1`。

### D2 会话契约由 head 配置给定

子 agent 代拍板 + 理由：head 在不跑模型的情况下算不出布局指纹，也很难可靠地重算学习率调度哈希（要复现 Miles 参数命名空间）。所以 head 钉的是"岛契约"（32 字节），由用户用 `--rl-island-contract-sha256` 给出；syncer 用 HELLO 自带的布局指纹现场算出期望的会话契约再比较。

- legacy：`check_pinned_contract` 在 `admit_session` 之前执行。不符只拒该连接，会话继续。
- elastic：`ElasticServerConfig.expected_backend_identity` 作为 `backend_identity` 的初值，JOIN 原有的比较逻辑直接生效。续跑时检查点里的身份与配置不同则报错。
- 未给出时保持"第一个岛决定"，与现在兼容；LPG 的"负例岛晚连 180 s"仍可用。后续可以加一个工具，在本机按运行参数算出岛契约。
- 未验证：critic 通道（`critic_syncer_command`）的岛是否用同一个岛契约。只有用户给出钉定值时才受影响。

### D3 学习岛不持有 Modal 主令牌；令牌走 secrets

- `HARNESS_PASSTHROUGH_ENV` 去掉 `MODAL_TOKEN_*`，换成 `YETO_SANDBOX_MODAL_TOKEN_ID/SECRET`。用户需要另建一个只用于沙箱的 Modal 令牌（建议放在单独的 Modal workspace，限制它能动的资源）。这一步需要用户操作。
- `ModalSandboxBackend` 有沙箱令牌时用 `modal.Client.from_credentials` 建客户端，传给 `App.lookup` 与 `Sandbox.create`。
- launcher 选择 Modal Sandbox provider 且缺沙箱令牌时，起机前报错。
- `split_secret_envs`：名单内的环境变量从 `envs` 移到 `secrets`。用于学习岛任务（RL、SFT）、扩散采样任务和 head 任务。Modal 岛把 task.secrets 里名单内的值并入 `cfg.envs`，后者本来就整体作为 `modal.Secret` 发送。
- HF 令牌文件挂载（`~/.cache/huggingface/token`）不变。

### D4 岛启动检查：环境变量报错，文件告警

子 agent 代拍板 + 理由：

- 环境变量里的云凭据（名单与 `launcher.CLOUD_CREDENTIAL_ENV` 相同）→ 报错退出。launcher 已经保证不发，出现就说明有路径漏了，越早失败越省钱（检查在加载模型和占卡之前）。
- 家目录里的凭据文件 → 只告警。部分镜像可能自带这类文件但岛不读，直接报错风险大于收益。
- `YETO_ALLOW_ISLAND_CLOUD_CREDENTIALS=1` 把报错降为告警，给手工调试用。
- 检查放在五个岛入口模块的 `if __name__ == "__main__":` 里（`yeto.learner`、`yeto.megatron.learner`、`yeto.diffusion.learner`、Miles 与 verl 的 `island_entry`），不放进 `main()`，这样进程内调用 `main()` 的单测不受本机环境影响。
- 未验证：Modal 函数容器是否会自动注入 `MODAL_TOKEN_ID/SECRET`。若会，Modal 岛会在启动时报错（不花卡时），需要下次上卡时确认。

## Risks

- 旧镜像里的岛代码不发 MAC，连新 syncer 会被拒。岛与 syncer 来自同一次发布，正常不会混用。
- 默认路径（未给 `--rl-island-contract-sha256`）仍是"第一个岛决定"。
