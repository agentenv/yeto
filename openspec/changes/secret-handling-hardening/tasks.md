# Tasks

## 1. syncer 认证（D1）
- [x] 1.1 server.rs：HELLO 末尾 HMAC 校验（`open_hello`），不符只拒该连接
- [x] 1.2 main.rs：legacy 无密钥拒绝启动；`--allow-unauthenticated-islands` / `YETO_SYNCER_ALLOW_UNAUTHENTICATED=1`
- [x] 1.3 protocol.py：`seal_hello`，`SyncerClient` 默认从 `YETO_ISLAND_HMAC_KEY` 读密钥
- [x] 1.4 launcher：RL legacy 也发密钥，缺失时生成；SFT 与 ssh_harness 显式无认证（已知缺口）
- [x] 1.5 单测：Rust（MAC 正确/错误/缺失）、Python（真 syncer 端到端：对的密钥接纳，错的被拒；无密钥启动被拒）

## 2. 会话契约由 head 配置给定（D2）
- [x] 2.1 syncer `--expected-island-contract`；legacy `check_pinned_contract`；elastic 初值钉定 + 续跑检查
- [x] 2.2 launcher/cli `--rl-island-contract-sha256` → syncer 命令
- [x] 2.3 单测：负例岛先连被拒、好岛随后被接纳（真 syncer）

## 3. 令牌边界（D3）
- [x] 3.1 `MODAL_TOKEN_*` 移出 `HARNESS_PASSTHROUGH_ENV`，换成沙箱令牌；缺沙箱令牌起机前报错
- [x] 3.2 tb2_provider / modal_reward_env 用沙箱令牌建 Modal 客户端
- [x] 3.3 `split_secret_envs` 用于学习岛、SFT、扩散采样、head 任务；Modal 岛并入 `modal.Secret`
- [x] 3.4 单测

## 4. 岛启动检查（D4）
- [x] 4.1 `yeto/island_credential_guard.py`
- [x] 4.2 五个岛入口 `__main__` 调用
- [x] 4.3 单测（含与 launcher 名单一致）

## 5. 收尾
- [x] 5.1 本机安全测试集、cargo test
- [ ] 5.2 上卡确认 Modal 容器内无 `MODAL_TOKEN_*`（未验证，下次上卡顺带）
- [ ] 5.3 后续：SFT 与 ssh_harness 的密钥通道；本机计算岛契约的工具
