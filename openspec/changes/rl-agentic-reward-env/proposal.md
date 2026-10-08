# Proposal: agentic RL 专用的奖励环境沙箱（多 benchmark，先 TB2）

## Why
- 用户 S17 裁定（经主 agent 转达，2026-10-08）：agentic 评测用 TB2 与 SWE-bench Verified；在 Modal 上搭一个专门给 agentic RL 用的奖励环境沙箱，在现有 yeto-tbench2 判分沙箱基础上做，支持多个 benchmark（先 TB2；benchmark 选择 WP3 待用户定，接口要留扩展）；TB2 判分沙箱要预装依赖；TB2 留出 30 个评测任务。
- 现状 [实测]：TB2 每题判分 2.5–4 min，大部分是 `tests/test.sh` 每次重新 `apt-get install curl`、下载 uv、`uvx` 建 pytest 环境（FNCODEX-STAGE2-ANALYSIS §7，regex-log 约 260 s）。这段时间计入每轮 rollout 墙钟。
- 现有 `yeto/rl/harness/codex/tb2_provider.py` 只认 TB2、只用任务官方镜像、且位于 codex harness 目录（去耦合阶段 3/4 会改动、并在签名范围附近）。

## What Changes
- 新包 `yeto/rl/harness/reward_env/`（不在 `codex/` 下，codex 与将来 verl/其他 harness 共用）：
  - `benchmark.py`：中立的 benchmark 适配器协议（任务列表、任务规格、预装计划、判分命令、判分输出解析）、注册表、判分执行器 `run_judge`、`JudgeResult`。
  - `swebench_verified.py`：SWE-bench Verified 适配器（用户 S17 裁定的第二个 benchmark；判分按官方 harness 5.0.2）。
  - `tb2.py`：TB2 适配器（复用 `tb2_provider` 的任务解析与判分命令，只 import 不改）；从 `test.sh` 抽取依赖安装步骤生成预装计划；`PrebakedModalSandboxBackend` 与 `modal_provider`（同一个 `Tb2EnvironmentProvider`，只把沙箱镜像换成预装镜像，Modal app 默认 `yeto-reward-env`）；留出评测集工具。
- `tools/reward_env/build_images.py`：默认只出计划报告；`--build` 在 Modal 上构建（仅 CPU），拒绝目标 `yeto-tbench2`。
- 启动器把新 provider 视同 Modal provider（岛上装 Modal 客户端）。
- 不改：`test.sh`、判分命令、签名结果路径、`yeto-tbench2` 应用。

## 为什么新开 change 而不并入 rl-codex-harness-rollout
1. 范围不同：rl-codex-harness-rollout 是"codex 这个 agent 怎么接进 RL"（网关、TITO、签名、会话）；本 change 是"奖励从哪个环境、怎么判"，与 agent 无关，verl 接入与别的 harness 也要用。
2. 多 benchmark 的选择在 WP3（rl 评测分桶 spec）等用户定，本 change 只给接口；放进 codex change 会把两个独立决策绑在一起。
3. 文件不冲突：codex change 的代码在 `harness/codex/`（去耦合阶段 3/4 要动），本 change 全在新目录。
4. 关系：rl-codex-harness-rollout §8（SandboxBroker 协议、环境注册表 digest 锁定）与本 change 有交集；本 change 的预装镜像以内容哈希作为身份，可作为 8.2"digest 锁定"的一部分输入，届时在那边引用，不在这里重复实现 broker。

## Impact
- 新代码：`yeto/rl/harness/reward_env/`、`tools/reward_env/`、`tests/test_reward_env.py`；`yeto/launcher.py` 两行。
- 默认行为不变：不设置新 provider 时一切照旧。
