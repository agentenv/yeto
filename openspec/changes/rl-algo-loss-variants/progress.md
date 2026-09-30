# rl-algo-loss-variants 进展

对齐结论、依赖矩阵、工作包与待批准事项见 [`../rl-infra-spec/alignment.md`](../rl-infra-spec/alignment.md)。

## 2026-09-29（Agent ALIGN，阶段 0）

- 规划文档从 `/home/michael/work/rl-algos`（分支 `rl-algorithms` HEAD 18695ae，含未跟踪文件）原样复制到分支 `rl-infra-spec`（提交 `5753e30`），此后以本分支副本为准；rl-algos worktree 未改。
- 2b；依赖 P0 与用户路线决定（待批准）。与 infra 的接口：A7（路线 B 与 M1–M6 串行合回 `yeto/ports`，由 Agent IMG 单独更新 pin）、A4（GMPO 在 CP 下）。
- 任务状态：全部未完成（0 勾选），没有代码改动，没有使用 GPU 或云资源。
- `openspec validate rl-algo-loss-variants --strict`：见 rl-infra-spec/progress.md 的 ALIGN 条目。

## 待批准

见 alignment.md §8（与本 change 相关的是：GPU 预算、`miles_adapter/config.py` 与 `entry.py` 的归属、“commit/push 需用户确认”一句的澄清、2b 路线决定）。

## 下一步

按 alignment.md §7 的工作包派发。

## 2026-09-30（Agent ALGO-2b，yeto 侧，路线 B）

分支 `algo-2b`（michaellchung/yeto），worktree `/home/michael/work/algo-2b`，基于 integ-decl `ef2d6b0`。未改 `driver.py`、`trainer.py`、`yeto/rl/__init__.py` pins、miles 仓库。没有使用 GPU 或云资源（费用 0，无残留）。

### 1.1 事实核对（Miles pin `MILES_NEXT_COMMIT` = 0af62f4d）
在 `/home/michael/work/miles-elastic` 上 `git show 0af62f4d:<file>`：
- `miles/backends/training_utils/loss_hub/losses.py:14-20` 以 `from ...math_utils import (..., compute_gspo_kl, ..., compute_policy_loss, ...)` 按名导入；`policy_loss_function` 定义在 `:63`，调用 `compute_policy_loss` 在 `:210`；GSPO/OPSM 的全收集在 `:144`（`need_full_log_probs`）与 `:177`（`compute_gspo_kl`）；`pg_clipfrac` 聚合在 `:303`，写入 loss dict 在 `:373`。
- `math_utils.py:253-254`：`@torch.compile(dynamic=True)` + `def compute_policy_loss`；`compute_gspo_kl` 在 `:224`。
- `losses.py` 中没有 `policy_objective` 分支（grep 为空）。
- batch 键：`miles/backends/training_utils/data.py:170` 的 `get_batch(keys)` 取固定键列表（`:207` 只追加 `adapter_slots`），不含 `sample.metadata`。
- 结论：design Context 的事实在 0af62f4d 上仍成立；路线 B 的分支点就是 `:210`。

### 1.2 用户决定
`/home/michael/work/infra-drafts/SESSION2-HANDOFF.md` 第 0 节第 1 条（2026-09-30，主 agent 记录的用户决定）："2b（rl-algo-loss-variants）选路线 B：在 michaellchung/miles 的 `yeto/ports` 分支上给 policy loss 加 variant 分支（CISPO、SAPO-Qwen、GMPO）与参数……用户已同意向该 fork 分支提交。"alignment.md §7b 追加（2026-09-30）同。第 3 组（路线 A）不执行。

### 实现（yeto 侧）
- 新 `yeto/rl/algos/loss_variants.py`（注册模块，已加入 `EXTENSION_MODULES`）：
  - 字段 `loss.policy_loss_variant ∈ {policy_loss, cispo, sapo, gmpo}`（默认不输出）；`loss.sapo_tau_pos`(1.0)、`loss.sapo_tau_neg`(1.05)、`loss.gmpo_log_clip_low/high`(0.4)，正有限数校验，报错指明字段；参数只在所属变体被选中时进入规范化（此时即使等于默认值也输出）。**与 design D2 的命名偏离**：P0 的 `loss.variant` 已是 `--loss-type`（policy_loss/custom_loss），变体在 `--loss-type policy_loss` 内部只替换 pg_loss，故另设字段，名称与 fork flag 一一对应。
  - 机制 `losses:cispo/sapo/gmpo`；flag 行 `--policy-loss-variant`、`--sapo-tau-pos`、`--sapo-tau-neg`、`--gmpo-log-clip-low`、`--gmpo-log-clip-high`（吸收 + 冲突检测；翻译时显式输出所选变体的参数）。flag 名已与 FORK-2b 工作区 `/home/michael/work/miles-2b`（未提交 diff，基于 0af62f4d）核对：用 `/tmp/review-miles-venv` + `PYTHONPATH=miles-2b` 构造 upstream parser，5 个 flag 都存在，choices = `['policy_loss','cispo','sapo','gmpo']`。fork 提交后需再核对一次。
  - 拒绝（D4）：变体+GSPO、变体+dual-clip、变体+custom_loss、SAPO/GMPO 设置无效的 eps_clip/eps_clip_high、其他变体的参数；启动检查：pin 不在 `FORK_COMMITS`（当前为空）时拒绝，GMPO+CP>1 拒绝（仅 CPU 验证）。
  - 梯度（D5）：CISPO/SAPO 不注册规则（沿用 GRPO 判定，全部越界仍期望梯度）；GMPO 在 `masked_fraction`=1 时放宽，未知退回 GRPO 判定；非有限 grad_norm 由 driver 在任何变体下判失败。
- `yeto/rl/engine/fake.py`：fake 声明 `losses={policy_loss,cispo,sapo,gmpo}`。
- `miles_adapter/entry.py`：**未改**，Miles adapter 不声明三个变体（可表达未开放；原因：GPU 验证被用户暂停，且 pin 尚无 fork 提交）。
- `miles_adapter/algorithm_flags.py`：5 个 fork flag 加入 `_UNMAPPED` 初值，使其在 import 时即属 `ADAPTER_OWNED_FLAGS`（由 loss_variants 的 `register_flag` 移出未映射集合）。`config.py` 未改（翻译经注册表自动进入 `algorithm_argv`/`check_extra_argv`）。
- `tests/test_rl_algorithm_flags_upstream.py`：pin 不含 fork 提交时，upstream 存在性检查豁免这 5 个 flag。
- 文档：`docs/MILES_RL.md` 新小节 "Policy-loss variants"。
- 交给 INFRA 的补丁（未应用）：`/home/michael/work/infra-drafts/patches/algo-2b-trainer.patch` —— `MilesTrainerGroup.train_step` 在 GMPO 下收集 step losses 并把 `pg_clipfrac` 作为 `masked_fraction`（与 GSPO 相同路径）。未应用前 GMPO 的 `masked_fraction` 为 None，只会更严格（可能误报，不会漏报）。`driver.py` 不需要改（`gradient_expectation` 已支持注册规则）。

### 测试
- 基线（ef2d6b0 临时 worktree `/tmp/algo2b-base`）：`OMP_NUM_THREADS=1 /tmp/yeto-venv/bin/python -m pytest -q --continue-on-collection-errors -p no:cacheprovider -rfE` → 68 failed, 2668 passed, 49 skipped, 26 errors；失败 id 94 条存于 `baseline-failures.txt`（命令与 1.3 原文相比未显式加 `tests/`，与 BRIEF 规定的全量口径一致）。
- 本分支同命令 → 68 failed, 2742 passed, 49 skipped, 26 errors；失败 id 集合与基线 diff 为空。
- 新测试 `tests/test_rl_loss_variants_spec.py`（74 项）+ 参考实现 `tests/rl_loss_variant_reference.py`。
- `openspec validate rl-algo-loss-variants --strict` → valid。

### 任务状态
| task | 状态 |
|---|---|
| 1.1 1.2 1.3 | 已完成（勾选） |
| 2.1 | 已实现 / CPU 通过；未勾：离线无法核对论文公式编号 |
| 2.2 2.3 2.4 2.5 | CPU 通过（勾选） |
| 3.x | 路线 A 未选，不执行 |
| 4.1 4.2 | FORK-2b 负责（miles-2b，本 agent 未改） |
| 4.3 | 未完成：待 fork 独立审查 → 快进 yeto/ports → IMG 重建镜像与 pin |
| 4.4 | 已实现（yeto 映射、吸收、冲突单测）；未勾：需新 pin 下 upstream `parse_args` 与来源记录提交号（届时把提交号加入 `FORK_COMMITS`） |
| 5.1 | CPU 通过（勾选）：fake 声明、adapter 未放行拒绝、单岛放行通过能力检查、argv 快照不变。真实 Miles 启动另受 pin 检查约束 |
| 5.2 | 已实现（文档 + dry-run 核对一致）；未勾：pin/提交号未落地 |
| 6.1–6.4 | 未完成：GPU 暂停（见下方计划） |
| 7.1 7.2 7.3 | 完成（勾选；fork 合入与 pin 更新后需重跑 7.1） |

### 待本地 GPU 验证计划（判据事先固定，执行时不得修改）
前置：fork 提交已审查并快进 `yeto/ports`；IMG 已更新 `MILES_NEXT_COMMIT`/镜像；该提交已写入 `loss_variants.FORK_COMMITS`；trainer 补丁已合入。模型与规模与 1b/2a 的 G1 相同（Qwen2.5-0.5B LoRA，单岛 colocated-serial，每轮 `num_steps_per_rollout=2` 以使 ratio 偏离 1），固定 seed 17，只跑一次，不挑 seed。
- 6.2 G1（每个变体 1 卡，3 轮，`--rl-single-island-no-sync --rl-allow-unverified-mechanism losses:<v>`，CP=1）：运行前记录 `nvidia-smi --query-gpu=name,driver_version`。通过判据全部满足：(a) 每轮 loss、grad_norm 有限；(b) 无零梯度误报（未因 zero-gradient invariant 失败）；(c) 事件磁带/来源中 `algorithm_spec_sha256` 对应的规范化 JSON 含变体名与参数，runtime fingerprint 的 miles_commit 等于 fork 提交；(d) 生效证据：与同 seed 默认 GRPO 对照，第 1 轮第 2 个 optimizer step 的 pg_loss 或 grad_norm 不相等（第 1 步 ratio=1 时 CISPO/SAPO 可能与 GRPO 数值相同，故比较第 2 步）；GMPO 另需 `pg_clipfrac` 有限。任一不满足即失败，只有找出并修复原因后才重跑。通过后才在 `entry.MILES_DECLARED` 声明该变体（附证据路径）。
- 6.3（CISPO 两岛 strict-avg 1+1 卡，3 轮，不带放行参数，须先在 6.2 后正式声明）：两岛 `algorithm_spec_sha256` 相同，每轮外层平均后的 canonical LoRA 哈希两岛相同。
- 6.4：本地卡无云资源；记录进程已退出。GMPO CP>1 不在本计划内，保持启动前拒绝。

### 待批准
- 无新增。trainer 补丁需主 agent 在 INFRA-E2 空档合入（`trainer.py` 属 INFRA-E2）。
- `algorithm_flags.py`、`fake.py`、`yeto/rl/algos/__init__.py`（一行）属 ALGO-CAP 共享注册文件，本 agent 按注册扩展点做了最小改动，请主 agent 确认。

### 下一步
1. FORK-2b 提交后：`git -C /home/michael/work/miles-2b diff 0af62f4d` 再核对 flag 名、默认值与 `pg_clipfrac` 语义（GMPO：序列内被 clip 的 token 比例）。
2. yeto/ports 快进 + IMG 更新 pin 后：把新提交写入 `FORK_COMMITS`，在钉住镜像中用 upstream `parse_args` 解析 `algorithm_argv` 输出（4.4），重跑 7.1。
3. 本地 GPU 可用后按上面的计划执行第 6 组。

### 2026-09-30 审查修复（ALGO-2b 第二轮）
- 高1：参考实现 GMPO 改为论文 arXiv:2507.20673v3 式 (4) 的单侧截断，在 log 空间 ℓ = sign·min(sign·log r, sign·clamp(log r, −δl, δh))。推导依据是论文式 (4) 与论文伪代码（HTML v3 全文，通过 WebFetch 读取）；官方仓库 github.com/callsys/GMPO 未逐行读取。新增用例：A>0 且 log r≪−δ 时不截断、仍有梯度；δl≠δh 非对称；A<0；A=0。与 FORK-2b 工作区（未提交）`compute_gmpo_loss` 核对：公式与 clipfrac 口径一致（fork 在无 A≠0 token 时返回 0，参考实现返回 None；两者都不会触发放宽）。
- 中2：GMPO clip 比例定义为"有效且 A≠0 的 token 中截断生效的比例"，写入 design D5、docs 并加测试。注意：Miles 用 sum_of_sample_mean 聚合 pg_clipfrac，全零优势序列计为 0，所以一轮里只要有这样的序列，整轮比例就小于 1。这是保守方向：不放宽、可能误报。
- 中3：GMPO 规则改读 `TrainStepMetrics.clip_fraction`（driver 已有字段，不改 driver），不读 `masked_fraction`；trainer 补丁改为只在 GMPO 下收集 step losses（`clip_fraction` 由现有 `_mean_clipfrac` 填充），不再写 `masked_fraction`。补丁已更新并在临时副本上 apply，相关测试通过（除基线环境性错误）。
- 中4：CISPO 必须显式给出 `loss.eps_clip` 与 `loss.eps_clip_high`（新拒绝 `loss_variant_cispo_clip`），两者进入哈希与 argv。
- 低5：GMPO CP 检查加注释（调用方目前硬编码 CP=1，属防护）。低6：dry-run 输出 `launch_warnings`（跑 launch_problems，CP=1），会显示 pin 检查。
- 5.1 完成记录已注明：仅能力检查层面放行，真实启动仍被 pin 检查阻止。
