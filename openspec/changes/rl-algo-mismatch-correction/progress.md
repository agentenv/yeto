# rl-algo-mismatch-correction 进展

对齐结论、依赖矩阵、工作包与待批准事项见 [`../rl-infra-spec/alignment.md`](../rl-infra-spec/alignment.md)。

## 2026-09-29（Agent ALIGN，阶段 0）

- 规划文档从 `/home/michael/work/rl-algos`（分支 `rl-algorithms`，HEAD 18695ae，含未跟踪文件）原样复制到分支 `rl-infra-spec`（提交 `5753e30`），此后以该分支上的副本为准。
- 本 change 编号 1a，依赖 P0。与 infra 的接口：A5（G2 报告注明执行模式与权重传输方式，不外推到分区模式）、A6（staleness>0 另立 change）。

## 2026-09-29（Agent ALGO-1a，实现轮）

### 分支与状态

- 分支 `algo-1a`，worktree `/home/michael/work/algo-1a`；基底 algo-cap `3d1b466`，之后 merge 了 `origin/algo-cap` ebd436b（P0 扩展点）和 `origin/rl-integ` f6194da（私有 ports 镜像 pins）。已普通推送到 `origin/algo-1a`。HEAD 与未提交改动见文末“状态快照”。
- 共享文件只改了 `yeto/rl/algos/__init__.py` 中的一行（EXTENSION_MODULES 注册）。其余共享接口的需求写成补丁 `/home/michael/work/infra-drafts/1a-shared.patch`，本分支没有应用它。

### 新增文件

- `yeto/rl/algos/mismatch_correction.py`：注册模块。命名机制包括 mismatch_observe、icepop、opsm_trainer、opsm_rollout、mis、mis_mask；扩展字段包括 `opsm_old_logprob_source`、`mis_*`；另有拒绝规则、MIS 的 runtime attrs，以及 `masked_fraction_from_metrics`。
- `yeto/rl/algos/mismatch_observe.py`：只观测插件。
- `yeto/rl/algos/vendor/miles_mis.py`：Miles `mis.py` 的逐字副本，文件头记录来源、commit 与 Apache-2.0 许可证。
- `tests/test_rl_mismatch_correction.py`：yeto venv 测试。
- `tests/test_rl_mismatch_observe.py`：在 miles-next-venv 中针对 Miles 源码做数值测试。
- `docs/MILES_RL.md` 新增一节 “Train/inference mismatch corrections”。

（任务逐项状态、证据与 GPU 记录见下文；本文件在 GPU 轮结束后更新。）
