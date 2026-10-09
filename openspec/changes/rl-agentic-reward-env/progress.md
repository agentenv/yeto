# Progress: rl-agentic-reward-env

## S17 夜间 G3（N2，2026-10-08，Modal CPU，app yeto-reward-env）
原始数据：`s1-runs/s17-g3-reward-env/`（probe.py、*.jsonl、*.log）。代码：agentenv/yeto 分支 `s17-codex-events`。费用见 `infra-drafts/gpu-spend.md` G3 回填（≈$1.2 [估算]，复跑 8 题另约 $0.05）。

### 修掉的两个缺陷
1. 预装镜像构建：多行脚本放进一条 Dockerfile RUN，Modal 拒绝（could not parse Dockerfile）。改为 base64 单行 RUN（`yeto/cloud/modal_reward_env.prebake_run_command`），单测 `tests/test_reward_env.py`。[已实现，已在 Modal 实测]
2. 判分命令超参数上限：`tb2_provider.verifier_command` 把 tests 目录 tar+base64 直接拼进命令行，7 题超过 Modal exec 64 KiB 上限，永远判不了分。改为：小的照旧内联（命令不变），大的先分块（每块 ≤48 KB）追加到沙箱里的暂存文件再解码（`verifier_stage_commands` / `stage_tests`，reward_env 适配器经 `judge_setup_commands`）。单测 `tests/test_harness_tb2_provider.py` 两项（模拟 64 KiB 上限）。[已实现，已在 Modal 实测：7 题复跑全部得 1]

### 89 题官方解阳性对照（预装镜像，任务默认 CPU/内存）
首轮 68/89 得 1；修缺陷 2 后复跑 7 题 + 探针自身超限的 make-doom-for-mips 共 8 题，**全部得 1**（`pos-argmax-rerun.jsonl`）→ **76/89**。其余 13 题逐题分类（`pos-all.jsonl` 的 solve_tail / judge_log_tail）：

| 类别 | 题目 | 依据 |
|---|---|---|
| 环境：上游软件源失效 | qemu-alpine-ssh、qemu-startup | 预装阶段 test.sh 的 apt 安装在 Debian bullseye-security 上 404，镜像都建不出来 |
| 环境：上游软件源失效 | build-pmars | solve.sh 钉的 dpkg-dev 1.22.21 已从源里消失（apt rc 100） |
| 环境：上游依赖漂移 | rstan-to-pystan | pystan 导入 pkg_resources 失败（setuptools 新版已移除） |
| 环境：依赖安装失败 | mcmc-sampling-stan | solve.sh 安装 rstan 失败（there is no package called 'rstan'） |
| 环境：沙箱权限 | custom-memory-heap-crash | solve.sh 的 `ulimit -c` 在 Modal 沙箱里不允许，release 构建那项测试失败（其余 5 项过） |
| 官方解本身不稳定/不过 | protein-assembly | 官方求解器报 "Wrong translation"、提示重跑 |
| 官方解本身不过 | build-cython-ext | 11 项中 pyknotid 仓库自带测试 1 项失败（可能上游漂移，未细查） |
| 官方解本身不过 | schemelike-metacircular-eval | 63 项中 2 项失败，用时 397 s（1 核下可能偏慢，未细查） |
| 超时（官方解跑满任务时限） | adaptive-rejection-sampler、sqlite-with-gcov、caffe-cifar-10、crack-7z-hash | solve 在 agent 时限（900–1800 s）到点，按任务默认 CPU |

结论：判分链路本身（沙箱、预装、暂存、reward 解析）在 89 题上没有剩余缺陷；13 题不可用的原因都在任务环境或官方解上。训练/评测选题时把这 13 题列入排除表（**未做**，留给 3.1/3.4）。

### 其他实测（详见 tasks 2.1–2.3）
- 冒烟 6 题 A/B 两遍 48 次：官方解全 1、空解全 0；判分中位 预装 ~5 s、不预装 ~7.5 s。S15 记录的每题 2.5–4 min 今天复现不出（原因未查）。
- 并发 24/64/256 个空沙箱全部创建成功；256 时创建中位 1.6 s、首次 exec 中位 7.6 s。账户上限未触到。
- 镜像大小 Modal 不提供，未测。

### 未验证
- 训练中（GPU）判分段时间：见 G6（rl-fn-codex-rollout 4.5 / 本 change 4.1）。
- SWE 部分（2.5/2.6）：无 Docker Hub 账号，跳过。
