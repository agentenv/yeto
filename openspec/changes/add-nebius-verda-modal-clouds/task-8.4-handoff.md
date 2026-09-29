# 任务 8.4 重跑交接（2026-09-23）

给接手重跑 8.4 的新 session。先读完这份，再动手。

## 先读这条：不要碰 git

**这个任务不包含任何版本控制操作。** 禁止 `git commit`、`git merge`、
`git rebase`、`git reset`、`git push`、`git checkout -b`、`git stash`。
仓库当前的状态是刻意如此的，工作区里未提交的改动是有效工作，不是待清理的脏数据。

你的任务只有一件事：**在 Modal 上把 8 卡 H100 的 RL 岛跑起来，拿到 2 轮同步和
计时数据。** 需要改代码就直接改工作区的文件，不要提交。

如果你认为必须动 git 才能继续，**停下来问用户**，不要自己决定。

## 任务定义

tasks.md 里 8.4 的原文要求用 CyberGym 环境。**用户已改为用 MATH-500**，其余不变：

- 在 Modal 上起 **一个 8 卡 H100 单容器 RL 岛**，`--training-mode rl`。
- syncer 事件 tape 里记录 **至少 2 整轮同步**。
- 记录跨公网（WAN）同步耗时。
- 给出 `--pipeline` 是否需要调大的结论。
- 结果写入 `docs/CLOUDS.md`。

## 上一轮的结论：部分达标，未勾选

2026-09-23 的 `yeto-rl84e` 跑完了 **3 整轮同步**，训练正常结束，rollout 真实（平均原始奖励约 0.6，16% 到 33% 的回答在 1024 token 处被截断）。WAN 同步和 pipeline 结论都已拿到。

**没有勾选 8.4，两个原因：**

1. **梯度范数三步都是 `gnorm="0.0000"`。** 而同期的 SFT 运行每步都是非零值。这说明岛提交的 LoRA 参数可能和上一轮的全局参数完全一致，也就是训练没有真正改动适配器。上一轮没有深究。**这是重跑时最该先查清楚的事。**
2. **用的不是默认镜像。** 属于对 spec 的偏离，见下一节。

## 代码在哪，以及在哪个目录跑

8.4 需要的代码**不在主目录**，在一个已经存在的 git worktree 里。不要为此做任何
合并操作，直接进那个目录跑就行。

**推荐做法：就在 worktree 里工作。**

```
cd /home/michael/yeto/.claude/worktrees/agent-a56a34e562486d7d7
```

那里已经包含 8.4 需要的全部代码：MATH-500 数据适配、数学奖励函数、
Modal RL 岛的容器内 setup、SGLang router 外部启动、Miles clone 的 origin 修正、
Ray dashboard 开启等。用主目录的解释器执行，`REPO_ROOT` 会自动指向 worktree：

```
/home/michael/yeto/.venv/bin/python -m yeto.cli launch ...
```

**两个目录各自的状态：**

| 目录 | 状态 |
|---|---|
| 主目录 `/home/michael/yeto` | HEAD 在 `7f5acbe`，与远端一致。工作区有 12 个文件的未提交改动，是跨云主线的修复。**没有 8.4 的代码，直接在这里跑会失败。** |
| worktree `.claude/worktrees/agent-a56a34e562486d7d7` | 分支 `worktree-agent-a56a34e562486d7d7`，HEAD 在 `1155682`，工作区干净。内容等于主目录 09:06 时的状态加 8 个 8.4 专属提交。 |

**worktree 唯一缺的东西：** 主目录在 09:06 之后又加了 `yeto down` 在 head 模式下的
回收加固（重试、逐个确认、未确认就保留 head）。worktree 里没有这部分，所以它的
`yeto down` 可能谎报成功并留下孤儿 learner。**应对办法是每次回收后手动到 Nebius
上核对实例列表**，命令见下一节。不要为了这个去合并代码。

两边改了同一批文件且已分叉，合并需要逐处核对，属于独立工作，不在本任务范围内。

## 环境前提

- **用仓库的 venv**：`/home/michael/yeto/.venv/bin/python`，已装 torch、skypilot 0.13（含 nebius/verda 扩展）、modal 1.5.5。
- **凭据已配好**：Modal token（workspace `yetalabs`）、Nebius（`~/.nebius/*.txt` 和 `~/.sky/config.yaml` 里 eu-north1 的 project id）。Verda 按用户要求不使用。
- **head 必须放 Nebius**：`--syncer-region nebius/eu-north1`。
- **Nebius 公网 IP 配额只有 3 个**，每台机器占 1 个。启动前先列实例确认有空位：

```bash
PID=$(.venv/bin/python -c "import yaml,os;print(yaml.safe_load(open(os.path.expanduser('~/.sky/config.yaml')))['nebius']['region_configs']['eu-north1']['project_id'])")
~/.nebius/bin/nebius compute instance list --parent-id "$PID" --format json
```

- **镜像必须覆盖**。默认的 `ghcr.io/agentenv/miles` 是私有的，匿名拉取返回 403，本机没有 GitHub token 也没有 `gh`。用公开的上游镜像，按摘要固定：

```
--rl-image docker:radixark/miles@sha256:cd40db923225c4146e90fdf4aa04bc000b71c1e980cc42df6f368de7545eaa09
```

该摘要是 tag `v0.1.0`，2026-08-18 构建，基础镜像 `lmsysorg/sglang:v0.5.16`，正好是 Miles 官方 Dockerfile 里 `SGLANG_IMAGE_TAG` 的默认值。**不要用可变标签**，启动前校验会拒绝，而且该仓库每天都有 `dev-*` 构建。

Miles 和 SGLang 的**源码仓库都是公开的**，setup 会在容器里拉到固定 commit 再覆盖安装，所以换镜像不改变实际运行的 Miles 代码。

## 上一轮跑通的确切命令

```bash
yeto launch --training-mode rl --gpu modal:8xh100 --syncer-region nebius/eu-north1 \
  --cluster-prefix yeto-rl84e \
  --rl-image docker:radixark/miles@sha256:cd40db923225c4146e90fdf4aa04bc000b71c1e980cc42df6f368de7545eaa09 \
  --model Qwen/Qwen3-1.7B --model-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --data HuggingFaceH4/MATH-500 --data-revision 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be \
  --reward-function yeto.rl.math_reward:reward_func \
  --apply-chat-template-kwargs '{"enable_thinking": false}' \
  --tuning lora --lora-r 8 --lora-targets attention --total-steps 3 \
  --rollout-batch-size 16 --n-samples-per-prompt 4 --rollout-max-response-len 1024 \
  --seq-len 2048 --inner-lr 1e-5 --seed 17 --trust-remote-code --on-demand
```

配置说明：LoRA 共卡模式，rollout 和训练共用同一组 8 张卡（Miles 的 `--colocate`）；TP 和 PP 都是 1，所以 DP=8，每张卡一份完整的 1.7B 基座加一份 LoRA；算法是 GRPO；同步用默认的 `strict-avg`，1 个片段，每个外层步就是一整轮。

## 耗时与花费预期

| 项 | 值 |
|---|---|
| 首次拉镜像（24 GB 压缩，44 GB 解压） | 6.5 分钟，仅第一次 |
| 冷启动到连上 syncer | 约 14 分钟 |
| 其中 8 个 SGLang 引擎启动与 import | 3.2 分钟 |
| 其中 HF 模型下载（**未认证**） | 3.4 分钟 |
| 其中 8 个 Megatron rank 启动 | 2.5 分钟 |
| 其中 CUDA graph 捕获 | 1.6 分钟 |
| 其中容器内 setup（clone + pip） | 1.8 分钟 |
| 实际加载权重 | 1.5 秒 |
| 稳定后每轮 | 约 26 秒 |
| WAN 同步每轮 | 5.2 到 5.4 秒，其中约 0.9 秒是本来就有的权重发布 |
| 上一轮总花费 | 约 25 美元（5 次尝试，合计约 40 GPU 分钟） |

8 卡 H100 按 Modal 价格表约 35 美元/小时。启动慢的主因是进程启动和 Python import 重复 16 次，不是 GPU 工作。可优化项：配 HF token 或把模型预置进 Modal Volume（省 3.4 分钟）；把 Miles 和 SGLang 装进镜像（省 1.8 分钟）；冒烟时关掉 CUDA graph 捕获（省 1.6 分钟）。

## 已知的坑

上一轮从第一次尝试到跑通试了 5 次，踩过的坑记录在同目录的 `live-run-failures.md` 第 13 到 20 条。
**注意那份文件在主目录和 worktree 里是两个不同版本**，编号含义不同：主目录那份的 13 到 21 条是跨云主线的问题，
worktree 那份的 13 到 20 条才是 8.4 的。读 worktree 里的那份。

这些坑的修复都已经在 worktree 的代码里，所以**只要在 worktree 目录下跑就不会复现**，不需要做任何合并。

## 重跑时的重点

1. **先查梯度范数为 0。** 建议用 `modal shell` 在同一个固定镜像上开一个 **1 张卡** 的交互容器（约 4 美元/小时），跑一轮最小的 GRPO，断言 LoRA 的 B 矩阵在优化器步之后确实变了。比直接重跑 8 卡便宜得多。Modal 的日志即使应用停止后仍可取回：`modal app logs <app> --since <ISO时间> --tail 20000 --timestamps`，上限 20000 行。
2. 确认 2 整轮同步，且每步的 responders 里都有该岛。
3. 结果写入 `docs/CLOUDS.md` 的 Run log，只更新真正验证过的那一格。
4. 新踩的坑补进 **worktree 里那份** `live-run-failures.md`，接着它的编号往下写。
   两份文件的编号冲突是已知的，等这个任务结束后由用户决定怎么统一，不要自己去合。

## 收尾义务

每次运行结束后必须确认三件事：

1. `yeto down <prefix>`。
2. `modal app list --json` 里自己的应用全部是 `stopped` 且 tasks 为 0。
3. Nebius 上没有自己前缀的实例残留。**不要动不属于自己的实例。**

worktree 里的 `yeto down` **没有**主目录后加的回收加固，所以这一步必须手动核对。
`yeto down` 在 head 模式下曾多次漏删 learner，本次会话手动删过 4 台孤儿 H100。修复已在主目录（重试加确认，未确认就保留 head 并返回非零），但**最终版本还没有在真机上经历一次真正需要它的回收**，所以每次仍要手动核对。
