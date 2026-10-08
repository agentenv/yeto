# Design: rl-agentic-reward-env

标注：[实测] 有原始数据；[读码]；[估算] 推算未验证；[未验证]。

## D1. 分层
```
BenchmarkAdapter（每个 benchmark 一个；注册表按名字取）
  task_ids / task_spec(镜像, cpu, 内存, 工作目录, agent 时限, 判分时限, 题面)
  prebake_plan(要预装的命令)  judge_command  parse_judge(输出 -> reward/passed/附加字段)
SandboxBackend（Modal 实现：PrebakedModalSandboxBackend；本地实现沿用 LocalProcessBackend）
EnvironmentProvider（沿用 tb2_provider.Tb2EnvironmentProvider：租约、中继、签名结果）
```
新 benchmark = 写一个适配器 + `register`；沙箱后端、判分执行器、启动器不动。适配器协议不含 Modal/Miles 依赖，verl 侧可直接复用。

## D2. 判分接口
- 训练路径：仍由 `Tb2Verifier.evaluate` 判分并写签名结果（不改信任分层）；本 change 只改沙箱镜像。
- 独立判分（评测、阳性对照、预装 A/B）：`run_judge(adapter, task_id, handle, prebaked=…)` → `JudgeResult{benchmark, task_id, reward, passed, exit_code, timed_out, seconds, prebaked, log(末 2000 字符), extra}`。TB2 的 `extra` 含 `testsh_rc`、`reward_txt`。判定只看 `reward.txt`（S15 已证实 testsh_rc 无信息量）。
- 以后多 benchmark：reward 统一为 float，`passed` 为布尔；部分得分的 benchmark 由适配器在 `parse_judge` 给出。

## D3. 预装依赖
- 来源：每题 `tests/test.sh` 在测试命令之前的安装行（apt-get / 安装器 `curl … | sh` / `source` / `pip install` / `uv …`），原样照抄；测试命令若是 `uvx …`，追加一条"同参数 + `pytest --version`"的预热命令，把 Python 解释器和 `-w` 依赖装进 uv 缓存。测试命令行含 shell 变量时不预热（避免改变引号语义）。识别不了的行（拷测试文件、下载数据、编译）**不预装**，留在判分时照常执行，计划标 `complete=false` 并列出。
- 判分时 `test.sh` 不改，照常执行全部步骤，但都命中缓存（apt 已装、uv 已在、uv 缓存已有）。`apt-get update` 仍联网 [估算 10–30 s]。
- 构建：`Image.from_registry(官方镜像).run_commands("bash -lc <set -e 脚本>")`；镜像身份 = sha256(schema, 官方镜像, 命令)。Modal 按内容缓存层，沙箱用同一定义即复用。
- 本地实测（读 tb2-data 2fd12b8，89 题，`/tmp/wp6-tb2-prebake-plan.json`）[实测]：89 题都有可预装命令；81 题带 uv 预热；84 题无未识别行；5 题有题目专属步骤留到判分时（cancel-async-tasks、fix-ocaml-gc、large-scale-text-editing、sam-cell-seg、train-fasttext）；89 个官方镜像各不相同 → 89 个预装镜像。
- 风险：(1) 模型在沙箱里能看到预装的 uv/pytest，可能改变行为（任务未禁止，判分仍以官方测试为准）；(2) 依赖装在镜像里后，若模型删改了它们，判分时 `test.sh` 会重新安装，结果不受影响只是变慢；(3) 预装后需逐题重做阳性对照（官方解=1、空解=0）才能用于训练。

## D4. 并发、冷启动、成本
| 项 | 现状 | 预装后 | 标注 |
|---|---|---|---|
| 判分耗时/题 | 150–260 s | 目标 <60 s | 现状 [实测 S15 §7]；预装后 [估算]，需实测 |
| 沙箱冷启动（拉官方镜像到起 sleep） | 未单独计时 | 镜像变大一点（uv+Python+pytest，约数百 MB [估算]） | [未验证]，验证时记录 Sandbox.create→首个 exec 的时间 |
| 镜像首次构建 | — | 每题一次，估 2–5 min/题，89 题可并行 | [估算] |
| 并发 | 一轨迹一沙箱；S15 FN 跑 24 条并发无基础设施错误 [实测] | 同 | Modal 账户沙箱并发上限未查证 [未验证]；全量训练 256 样本/轮需 256 并发沙箱，必须先实测 |
| 每题资源 | 84 题 1 CPU、3 题 2、2 题 4；69 题 2 GiB、17 题 4、3 题 8 [实测 task.toml] | 同 | |
| 成本 | 沙箱按 CPU/内存秒计费；S15 阳性对照 12 沙箱 ≈$0.03 [估算]；单价按 s14 台账口径 ≈$0.047/CPU·h + $0.008/GiB·h [估算] | 判分段节省约 2–3 min×(1 CPU+2 GiB) ≈ $0.002/条 [估算]；主要收益是墙钟：每轮 rollout 末尾少等 2–3 min | GPU 在等判分，墙钟才是大头：8×H200 ≈$46/h，每轮省 2–3 min ≈ $1.5–2.3/轮 [估算] |
| 构建成本 | — | 89 题 × 约 3 min × 1–2 CPU ≈ $0.3–1 [估算] | 需批准后在 Modal 构建（不占 GPU） |

## D5. 留出评测集（与 WP3 rl-eval-difficulty-buckets D2/D6 对齐）
- 每题元数据（WP3 D6a）：`TaskSpec.metadata()` 给 `task_id / benchmark / benchmark_version / difficulty（官方原文）/ difficulty_source / eval_bucket`。TB2 难度取 `task.toml [metadata].difficulty`（easy 4 / medium 55 / hard 30 [实测 tb2-data 2fd12b8]），桶 `tb2-easy|medium|hard`；SWE-bench Verified 取数据集 `difficulty` 列（官方 4 档），桶 `swev-lt15m`（<15 min fix）、`swev-15m-1h`、`swev-ge1h`（1-4 hours 与 >4 hours 合并）。
- 名单文件（WP3 D6b，`schema: yeto-eval-holdout/1`）：`benchmark.build_holdout(适配器, 每桶题数, seed, rule)`，桶内按 sha256(f"{seed}/{task_id}") 排序取前 N，与输入顺序无关；某桶不够即报错。TB2：easy 2 / medium 18 / hard 10，seed 20261008；SWE：<15 min 30 / 15 min–1 h 30 / ≥1 h 全部 45，共 105 题，"全部只评测"。生成工具 `tools/reward_env/holdout.py`；本地生成结果 [实测]：tb2-holdout.json 30 题 sha256 1a6a7a0c…，swebench-verified-eval.json 105 题 sha256 547dbd24…（文件放 `data/eval/` 由 WP3 定稿后提交）。
- **注意**：按此规则冒烟 6 题中的 `regex-log` 落入 TB2 留出集，而它在 S15 两次上卡里用作训练题；是否把冒烟 6 题排除出评测池（`exclude=`）需定。
- 训练数据过滤：`split_rows` 按 `metadata.task_id` 拆分，缺题号即报错；`assert_disjoint` 在启动前检查交集（WP3 D6c 第 1 条）；SWE 另给 `contamination_keys`（`repo@base_commit` 与规范化题面 sha256，D6c 第 2 条）。启动器接入在 tasks 3.x。
- 判分结果（WP3 D6d）：`JudgeResult` 给 `reward / passed / timed_out / infra_error / seconds`；`infra_error`（沙箱 exec 抛错、SWE 判分脚本没跑到测试、官方分类为环境故障）单列，不计入通过率。`turns / tokens / end_reason / policy_version` 由 harness 轨迹记录给，不在判分层。

## D6. 命名与隔离
- Modal app：运行时沙箱默认 `yeto-reward-env`（可用 `YETO_REWARD_ENV_MODAL_APP` 覆盖），构建脚本拒绝 `yeto-tbench2`；现有 `yeto-tbench2` 与旧 provider 不动。
- 开关：`YETO_REWARD_ENV_PREBAKE=0` 关闭预装（同一 provider 做 A/B 计时）。
- 启动：`YETO_HARNESS_ENVIRONMENT_PROVIDER=yeto.cloud.modal_reward_env:modal_provider`；启动器把它和旧 Modal provider 一样处理（装 Modal 客户端）。

## D7. 第二个 benchmark：SWE-bench Verified
### D7.1 数据与许可证 [实测除注明外]
- 数据：`SWE-bench/SWE-bench_Verified` 修订 78f471bf655a…（500 题，parquet sha256 030cfd7f…，本地 `~/work/swebench-data/`）。用这一版而不是 `princeton-nlp/SWE-bench_Verified`：官方 harness `swebench==5.0.2` 的 `make_test_spec` 要求 `image / eval_script / log_parser / eval_type` 字段，只有 SWE-bench 组织版有。两版 500 个 instance_id 相同、difficulty/补丁/base_commit 相同，但 **有 2 题的 FAIL_TO_PASS/PASS_TO_PASS 列表不同**（WP3 引用的是 princeton-nlp 版，需统一到这一版）。
- 难度：`difficulty` 列 <15 min fix 194 / 15 min - 1 hour 261 / 1-4 hours 42 / >4 hours 3。
- 许可证：官方 harness MIT（PyPI 元数据与 GitHub LICENSE）；两版数据卡都**未声明许可证**（HF API cardData.license 为空）；镜像 Docker Hub `swebench/sweb.eval.x86_64.*` 仓库页无说明、无许可证声明（抽查 1 个）。镜像内容包括 12 个上游仓库（django、sympy、sphinx、matplotlib、scikit-learn、astropy、xarray、pytest、pylint、requests、seaborn、flask，各自许可证**待逐项核对**）与 Miniconda（Anaconda 服务条款对商业使用有限制，**待核对**是否适用）。只拉取使用、不再分发；若要转存到我们自己的镜像仓库，先核完许可证。
### D7.2 镜像来源、构建/拉取、存储 [实测除注明外]
- 来源：每题数据行的 `image` 字段，Docker Hub 官方预构建镜像 `swebench/sweb.eval.x86_64.<id，__ 换成 _1776_>:latest`（官方 harness 的命名规则，`image_builder/image_spec.py`）。`latest` 会变（最近更新 2026-08-13/16），所以用 `tools/reward_env/swev_image_manifest.py` 查 Docker Hub API（只查元数据、不拉镜像）钉成 `@sha256:` 摘要：500/500 有摘要（`swev-images-78f471bf.json`，sha256 5d3e18f7…）。
- 大小（amd64 压缩后）：每个 0.96–3.17 GB，中位 1.16 GB，P90 1.93 GB，合计 664 GB（未扣除共享层，实际唯一存储更小，未测）。
- Modal 上：`Image.from_registry(钉摘要的引用)` 直接拉取，不在 Modal 上重建（官方 Modal 路径是从 ubuntu+conda 现建，每题数分钟且结果可能漂移，不采用）。Docker Hub 匿名拉取有频率限制（数值**待核**），批量预拉需配 Docker Hub 账号的 Modal secret（`SKYPILOT_DOCKER_*` 同类做法）。Modal 镜像缓存是否计费、上限多少**待核**。
- 冷启动：首次拉取 1–3 GB 镜像的时间**未测**；之后同一镜像走 Modal 缓存，**未测**。
### D7.3 判分接口（与官方一致）
- 每次判分开一个**全新沙箱**（官方做法，避免 agent 改动环境影响测试）；agent 的答案 = agent 沙箱里 `git add -A && git diff --cached <base_commit>`（`submission_command`）。
- 判分脚本（`judge_script`）按官方 `run_instance` 顺序：写补丁 → 依次 `git apply --verbose` / `--3way` / `--reject` / `patch --batch --forward --fuzz=5 -p1 -i`（每次失败后 `git checkout -- . ; git clean -fd`）→ `git apply --check --reverse` → 都不行即"补丁打不上 = 未解决"；然后跑数据集 `eval_script`（经官方 `make_test_spec` 加上测试退出码记录；脚本内部会重置测试文件、打官方 test_patch、只跑本题测试）。
- 评分：可信侧用官方 `swebench.harness.grading.get_eval_report`：FAIL_TO_PASS 全过且 PASS_TO_PASS 全过才算解决（reward 1），否则 0。测试超时按官方算未解决。需要在评分进程里装 `swebench==5.0.2`（本地单测用独立 venv；岛上安装放进接入任务）。
- 阴性对照 = 官方"不打补丁"模式（`submission=None`），阳性对照 = 数据集 `patch`（`gold_patch`）。
### D7.4 资源、并发、时间、费用 [估算，全部待实测]
| 项 | 值 | 依据 |
|---|---|---|
| 每个沙箱 | 4 CPU（官方 Modal 路径用 cpu=4）、内存先给 8 GiB | 内存未测 |
| 每题判分 | 官方超时上限 1800 s；典型值未测，估 2–10 min | 无实测 |
| 每次判分费用 | 4 CPU×5 min + 8 GiB×5 min ≈ $0.02 | 单价按 s14 台账口径 $0.047/CPU·h、$0.008/GiB·h |
| 每条 RL 轨迹 | agent 沙箱（同镜像，跑完整对话）+ 判分沙箱各一个 | |
| 全量 500 题阳性+阴性对照 | 1000 次判分 ≈ $20，外加首次拉取时间 | |
| 评测（WP3：每次 105 条） | 判分 ≈ $2/次 | GPU 推理费另计 |
| 并发 | 与 TB2 共用 Modal 账户沙箱上限（未查证） | 2.3 并发探针一起测 |

## D8. 未决（需用户/主 agent 定）
1. 首次镜像构建（Modal CPU，估 $0.3–1）是否批准；先构建冒烟 6 题还是全部 89 题。
2. 冒烟 6 题是否排除出 TB2 评测池（按现规则 regex-log 会进评测集）。
4. SWE-bench Verified 统一用 SWE-bench 组织版（与 WP3 引用的 princeton-nlp 版有 2 题测试列表不同）。
5. SWE 镜像只从 Docker Hub 拉（需 Docker Hub 账号避开频率限制）还是转存到自有仓库（需先核许可证）。
3. 预装后的阳性对照范围（建议先冒烟 6 题，再全部）。
