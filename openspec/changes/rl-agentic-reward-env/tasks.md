# Tasks: rl-agentic-reward-env

## 1. 接口与 TB2 适配（CPU）
- [x] 1.1 中立 benchmark 适配器协议、注册表、`run_judge`、`JudgeResult`（`yeto/rl/harness/reward_env/benchmark.py`）。验收：`tests/test_reward_env.py` 注册/未知/重复注册、判分通过/失败/缺结果。
- [x] 1.2 TB2 适配器（复用 tb2_provider，只 import）。验收：任务规格、判分命令、解析单测。
- [x] 1.3 从 `test.sh` 生成预装计划（安装行原样、uv 预热、变量行不预热、未识别行留到判分时）与内容哈希。验收：uvx / pip / uv run / 变量 / 下载 / 无测试行 单测；本地 89 题计划报告（84 题无未识别行）。
- [x] 1.4 `PrebakedModalSandboxBackend` + `modal_provider`（默认 app `yeto-reward-env`，可关预装）；启动器视同 Modal provider。验收：假 modal 模块单测（镜像构造、标签、资源、app 名）。
- [x] 1.5 构建脚本 `tools/reward_env/build_images.py`（默认只出计划，`--build` 需显式 app 且拒绝 `yeto-tbench2`）。验收：单测。
- [x] 1.6 留出名单（WP3 D6b 格式，TB2 easy 2/medium 18/hard 10；SWE 30/30/全部 45）、每题元数据（D6a）、训练数据拆分与泄漏检查、`infra_error`（D6d）。验收：单测；`tools/reward_env/holdout.py` 本地生成。
- [x] 1.7 SWE-bench Verified 适配器：钉数据修订、任务清单解析（缺字段/重复即拒）、难度与桶、官方补丁应用链 + eval_script 判分脚本、官方 `get_eval_report` 评分、不打补丁模式、gold 补丁、污染检查键。验收：`tests/test_reward_env_swebench.py`（真实数据行 + 官方 swebench 5.0.2，独立 venv 跑 11 项通过）。
- [x] 1.8 SWE 镜像清单：Docker Hub 元数据查询，500 个镜像钉摘要并记录大小（`tools/reward_env/swev_image_manifest.py`）。

- [x] 1.9 TB2 留出名单先排除 S15 冒烟 6 题再分层抽样（WP3 D2，#123 c8979bd2）：`tb2.build_holdout` 默认排除、结果带 `excluded`（task_id+原因）；`holdout.py` 加 `--tb2-exclude-jsonl/--tb2-exclude-reason/--tb2-no-exclude`，被排除题不在 checkout 中即报错。验收：`tests/test_reward_env.py` 新增 2 项，reward_env 两文件 34 passed（S17 N2，分支 s17-holdout-excl，PR #131 待审）。本机 tb2-data（89 题）实生成：30 题、6 题排除，sha256 28d6730a…，存 s1-runs/s17-c7-holdout/tb2-holdout.json（名单定稿仍归 3.4/WP3）。

## 2. 上 Modal（仅 CPU，需批准）
- [x] 2.1 构建冒烟 6 题预装镜像，记录每题构建时间与镜像大小。估 <$0.1。**[实测]** 第一次 6 题全部失败：多行脚本放进 `bash -lc '…'` 成了多行 Dockerfile RUN，Modal 报 could not parse Dockerfile（`build-smoke6-try1-dockerfile-newline.jsonl`）；改为 base64 单行 RUN（`modal_reward_env.prebake_run_command`，单测已补）后 6 题全通过，每题 7.5–25.4 s（`build-smoke6.jsonl`）。镜像大小：Modal 不提供，**未测**。原始数据 `s1-runs/s17-g3-reward-env/`（S17 N2 G3，app yeto-reward-env，代码 s17-codex-events）。
- [x] 2.2 预装 A/B：6 题 × {预装, 不预装} × {官方解, 空解}，记录冷启动（create→首个 exec）、判分时间、reward。判据：官方解全 1、空解全 0；预装判分时间显著下降。估 <$0.1。**[实测，两遍共 48 次]** 官方解 24/24 得 1、空解 24/24 得 0。判分时间中位 预装 5.0 / 4.8 s、不预装 7.0 / 7.65 s；均值 6.4 / 7.6 s 对 9.3 / 11.2 s；冷启动中位 2.6–5.0 s 两边相近（`ab-smoke6*.jsonl`）。**结论**：预装有效但收益只有每题约 2–4 s；S15 记录的每题 2.5–4 min 今天不预装也复现不出（当时慢的原因未查明，可能是网络/镜像源）。判据"显著下降"按绝对值不成立，按比例约 30%。
- [x] 2.3 并发探针：同时起 N=24/64/256 个沙箱（空任务 sleep），记录创建成功率与耗时，确认 Modal 账户上限。估 <$1。**[实测]**（debian_slim，0.125 核/256 MiB）24/64/256 全部创建成功；创建耗时中位 0.73/0.63/1.62 s、最大 1.1/1.35/7.4 s；创建到首次 exec 完成 中位 2.1/4.3/7.6 s、最大 5.2/5.7/8.5 s（`concur.jsonl`）。256 未触到账户上限，上限**未测到**。注：本步启动时我方线程 2564，超过 2500 的等待线（违反夜间规则，已记录）。
- [x] 2.4 全部 89 题构建 + 阳性对照。估 $0.3–1.5。**[实测]** 构建 87/89 成功（中位 24.7 s、p90 53 s、最大 365 s）；失败 2 题 qemu-alpine-ssh、qemu-startup：官方 test.sh 里的 apt 安装在 Debian bullseye-security 源上 404（上游源问题，判分时同样会失败）。官方解阳性对照（预装镜像，默认资源）**68/89 得 1**，未通过 21 题分类（`pos-all.jsonl`）：
  - 7 题**判分命令超 Modal 64 KB 参数上限**（build-pov-ray、make-mips-interpreter、pytorch-model-recovery、reshard-c4-data、sam-cell-seg、video-processing、train-fasttext）：`tb2_provider.verifier_command` 把 tests 目录 tar+base64 直接拼进命令行。**已修**（分块暂存，cf2cc22d），复跑 7 题 + make-doom-for-mips 全部得 1 → 合计 **76/89**；其余 13 题逐题分类见 progress.md。
  - 2 题镜像构建失败（同上 qemu）；1 题 make-doom-for-mips 是探针自己上传官方解时超参数上限（探针问题）。
  - 4 题官方解跑到任务时限仍未完成（adaptive-rejection-sampler、sqlite-with-gcov、caffe-cifar-10、crack-7z-hash，按任务默认 CPU）。
  - 7 题官方解跑完但判 0（build-pmars solve rc 100、custom-memory-heap-crash、build-cython-ext、protein-assembly、mcmc-sampling-stan、rstan-to-pystan、schemelike-metacircular-eval）：原因**未查**。
  判分时间中位 6.2 s、p90 61.7 s；冷启动中位 2.2 s。费用见台账 G3 回填（≈$1.2 [估算]）。

- [ ] 2.5 SWE 冒烟：每桶 4 题共 12 题（含 >4 hours 1 题），每题 gold 补丁 + 不打补丁各一次（24 次判分），记录首次拉取时间、沙箱冷启动、判分时间、峰值内存。判据：gold 全部解决、不打补丁全部未解决；不符的题记入排除表并查原因。估 $0.5–1（CPU）。
- [ ] 2.6 SWE 全量 500 题阳性/阴性对照（1000 次判分），产出可用题清单。估 ≈$20（CPU）+ 拉取。

## 3. 接入训练（依赖 WP3）
- [ ] 3.1 评测集定稿（WP3）后写入 split 文件；数据准备脚本按 split 过滤训练集并在启动前 `assert_disjoint`。（S17 N13 部分完成：`holdout.py` 同时写 `tb2-holdout.json` 与 `tb2-train.json`，生成时 `assert_disjoint`，不可用 13 题两边都排除；留出 30、训练 46，见 progress。训练 jsonl 生成脚本未做，名单待定稿。）
- [ ] 3.2 每 10 轮评测：评测题走同一 provider，结果按 WP3 事件字段落 tape。
- [ ] 3.3 SWE agent 沙箱 provider（agent 在实例镜像 /testbed 里工作，结束后取 diff，再开判分沙箱）；评分进程安装 `swebench==5.0.2`。
- [ ] 3.4 WP3 名单文件定稿后提交到 `data/eval/` 并在运行配置钉 sha256。

## 4. GPU 验证（并入 FN codex 第三次上卡，不单独开卡）
- [x] 4.1 FN codex 第三次上卡改用 `yeto.cloud.modal_reward_env:modal_provider`，记录每轮 rollout 中判分段时间，与 r2（未预装）对比。依赖 2.2 通过。**[实测]** `s17-fncodex-r3-20261008a`：24 条全部判分成功（9 条得 1），整段生成（含沙箱、Codex 回合、判分）224 s，r2 为 688 s。单条判分耗时事件里没有字段，**未单独采到**（后续在 agent_metrics 带 evaluate_time）。
