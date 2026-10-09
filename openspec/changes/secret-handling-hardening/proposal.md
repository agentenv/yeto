# Proposal: secret-handling-hardening

## Why

10-09 安全调查（SESSION18-HANDOFF.md 第 105、120 行）发现四个问题：

1. `--rl-island-scheduling legacy`（默认值）下 syncer 没有任何认证，监听 0.0.0.0，sky 还为它开公网端口。任何能连到该端口的人都能以岛的身份发 HELLO 并推送增量。
2. legacy 会话契约由第一个被接纳的 HELLO 决定（server.rs `admit_session`）。负例岛先连上时，好岛会被拒。
3. 学习岛持有 Modal 主令牌（`MODAL_TOKEN_ID/SECRET` 经 `HARNESS_PASSTHROUGH_ENV` 进岛的普通环境变量）。这与"云凭据不进学习岛"的规定冲突。HF、WANDB 令牌也走普通 `envs=`，会以明文进入 sky 的任务记录。
4. "云凭据不进学习岛"只在 launcher 单测里检查，岛进程运行时没有检查。

用户 10-09 批准立项。

## What Changes

- **BREAKING** legacy syncer 没有岛 HMAC 密钥时拒绝启动。只有显式传 `--allow-unauthenticated-islands`（或环境变量 `YETO_SYNCER_ALLOW_UNAUTHENTICATED=1`）才允许无认证运行。
- legacy HELLO 末尾带 HMAC-SHA256。syncer 有密钥时校验，不符就拒绝该连接。
- RL launcher 在 legacy 下也给 syncer 与每个岛发同一把密钥（走 secrets）。启动环境没有密钥时，launcher 为本次运行生成随机密钥。
- 新增 launcher 参数 `--rl-island-contract-sha256` 与 syncer 参数 `--expected-island-contract`。给出后，legacy 与 elastic 都按 head 配置的契约接纳岛，不再由第一个岛决定。
- **BREAKING** Modal Sandbox 奖励环境改用独立的沙箱令牌 `YETO_SANDBOX_MODAL_TOKEN_ID/SECRET`。主令牌 `MODAL_TOKEN_*` 不再发给学习岛。
- HF、WANDB、CyberGym、TBench 奖励 HMAC、沙箱令牌、岛 HMAC 密钥一律走 sky `secrets=`（Modal 岛走 `modal.Secret`），不进 `envs=`。
- 新增岛进程启动检查：环境变量里有云凭据就报错退出；家目录里有云凭据文件就告警。

## Capabilities

### New Capabilities

- `secret-handling`：岛认证、会话契约来源、学习岛凭据边界、岛启动时的凭据检查。

### Modified Capabilities

无。现有 `openspec/specs/` 下没有覆盖这些行为的 spec。

## Impact

- Rust：`syncer/src/main.rs`、`server.rs`、`elastic_server.rs`。
- Python：`yeto/protocol.py`、`yeto/launcher.py`（小改）、`yeto/cli.py`、`yeto/rl/ssh_harness.py`、`yeto/rl/harness/codex/tb2_provider.py`、`yeto/cloud/modal_reward_env.py`、新文件 `yeto/island_credential_guard.py`、五个岛入口模块的 `__main__` 块。
- 运维：用 Modal Sandbox 奖励环境的运行，需要用户另建一个只用于沙箱的 Modal 令牌并放进启动环境。
- 本地脚本：自己起 legacy syncer 的本地基准脚本需要设 `YETO_SYNCER_ALLOW_UNAUTHENTICATED=1` 或给密钥。
