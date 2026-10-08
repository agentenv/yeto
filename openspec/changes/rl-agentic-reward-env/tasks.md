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

## 2. 上 Modal（仅 CPU，需批准）
- [ ] 2.1 构建冒烟 6 题预装镜像，记录每题构建时间与镜像大小。估 <$0.1。
- [ ] 2.2 预装 A/B：6 题 × {预装, 不预装} × {官方解, 空解}，记录冷启动（create→首个 exec）、判分时间、reward。判据：官方解全 1、空解全 0；预装判分时间显著下降。估 <$0.1。
- [ ] 2.3 并发探针：同时起 N=24/64/256 个沙箱（空任务 sleep），记录创建成功率与耗时，确认 Modal 账户上限。估 <$1。
- [ ] 2.4 全部 89 题构建 + 阳性对照。估 $0.3–1.5。

- [ ] 2.5 SWE 冒烟：每桶 4 题共 12 题（含 >4 hours 1 题），每题 gold 补丁 + 不打补丁各一次（24 次判分），记录首次拉取时间、沙箱冷启动、判分时间、峰值内存。判据：gold 全部解决、不打补丁全部未解决；不符的题记入排除表并查原因。估 $0.5–1（CPU）。
- [ ] 2.6 SWE 全量 500 题阳性/阴性对照（1000 次判分），产出可用题清单。估 ≈$20（CPU）+ 拉取。

## 3. 接入训练（依赖 WP3）
- [ ] 3.1 评测集定稿（WP3）后写入 split 文件；数据准备脚本按 split 过滤训练集并在启动前 `assert_disjoint`。
- [ ] 3.2 每 10 轮评测：评测题走同一 provider，结果按 WP3 事件字段落 tape。
- [ ] 3.3 SWE agent 沙箱 provider（agent 在实例镜像 /testbed 里工作，结束后取 diff，再开判分沙箱）；评分进程安装 `swebench==5.0.2`。
- [ ] 3.4 WP3 名单文件定稿后提交到 `data/eval/` 并在运行配置钉 sha256。

## 4. GPU 验证（并入 FN codex 第三次上卡，不单独开卡）
- [ ] 4.1 FN codex 第三次上卡改用 `reward_env.tb2:modal_provider`，记录每轮 rollout 中判分段时间，与 r2（未预装）对比。依赖 2.2 通过。
