# agentic RL 评测基准调研（S17 WP3，2026-10-08）

目的：我们要做 agentic RL 并和 codex harness 结合，需要一个固定评测集（与 openspec change `rl-eval-difficulty-buckets` 同一套接口：评测样本行带 `bucket`、`difficulty_source`，`rl_eval` 事件分桶出指标）。本文比较候选基准并推荐 1–2 个。数据来自各仓库 GitHub API、数据文件与公开文章（链接见文末），2026-10-08 查询；标"未验证"的是没亲自核实的。

## 1. 比较

| 基准 | 任务形式 | 能否离线自动判分 | 沙箱依赖 | 规模 | 许可证 | 难度分层 | 接 codex harness 的工作量 |
|---|---|---|---|---|---|---|---|
| **Terminal-Bench 2.0**（harbor-framework/terminal-bench-2） | 在容器终端里完成真实任务（软件工程、系统管理、安全、数据科学等） | 能：每任务自带测试脚本 | 每任务一个 Docker 环境；我们已有 Modal 常驻判分沙箱 yeto-tbench2，判分每任务约 2.5–4 分钟 | 89 个任务（2.1 版也是 89） | Apache-2.0 | 有：本地 tb2-data（commit 2fd12b8）`task.toml` 的 `difficulty`：easy 4 / medium 55 / hard 30，另有专家/新手耗时估计 | **已接好**（rl-fn-codex-rollout 在用）；只需按留出划分 |
| **SWE-bench Verified**（SWE-bench/SWE-bench，500 题） | 给真实 Python 仓库的 issue，改代码让隐藏测试通过 | 能：跑仓库测试判分 | 每题一个 Docker 镜像（镜像体积大，具体大小未验证） | 500 题 | 代码仓库 MIT；HF 数据卡未写许可证 | 有：OpenAI 人工标注修复耗时 4 档，<15 分钟 194、15 分钟–1 小时 261、1–4 小时 42、>4 小时 3（vals.ai 统计；OpenAI 自报 easy 196、hard 45，略有差异） | 中：Harbor 有 `swebench` 适配器（已确认目录存在），可转成与 TB2 同格式的任务；需要补镜像拉取与判分时间评估。另有 `swegym`、`swesmith` 适配器可作**训练**任务来源（与评测不重叠需核对） |
| **τ²-bench**（sierra-research/tau2-bench） | 客服对话：智能体按政策调用工具，用户由另一个大模型模拟 | 判分按数据库最终状态与期望动作，规则判分；但**每次运行都要一个"用户模拟"大模型**在线参与 | 不需要容器，Python 环境即可 | airline 50（训练 30/测试 20）、retail 114（74/40）、telecom 基础 114（74/40，另有 2,285 题的完整生成池）——取自仓库 `split_tasks.json` | MIT | 官方无难度标签（telecom 有按问题类型分组；`audio_difficulty.json` 只针对语音） | 大：任务是多轮对话 + 函数调用，不是终端/写代码，codex CLI 的交互方式不匹配；需要接一个用户模拟模型（多一份推理成本与不确定性）。Harbor 有 `tau3-bench` 适配器（是否适合 codex 未验证） |
| **BFCL v4**（ShishirPatil/gorilla） | 函数调用：多数是单轮按格式输出调用（语法树比对），另有多轮、网页搜索、记忆类 | 单轮/多轮能离线判；网页搜索类要 SerpAPI key | 不需要容器 | 数千条，分十几类（具体条数未核） | Apache-2.0 | 无难度标签，只有类别 | 中到大：主要考函数调用格式，与 codex 在终端里干活不是一回事。Epoch AI 抽查 50 条有 24 条（48%）存在可能影响判分的缺陷 |
| **AppWorld**（StonyBrookNLP/appworld） | 在 9 个模拟 App（457 个 API）里写代码完成日常任务 | 能：自带评测程序，按状态判分 | Python 环境 + 其 App 引擎，不需要每题容器 | 750 题：训练 105 / 开发 60 / 测试-普通 168 / 测试-挑战 417 | Apache-2.0，但任务/API/判分部分以加密包发布，**公开再分发必须加密**（训练与对外提供模型输出不算再分发） | 有：作者人工标 1–3 级；挑战集更难 | 中：任务形式是"写代码调 API"，和 codex 比较接近；需要把 AppWorld 环境包成 codex 可用的工具/沙箱；测试集看不到初始状态与答案，只能拿总分 |

## 2. 推荐

1. **Terminal-Bench 2 留出集（首选，马上能用）**：已接入 codex harness 与判分沙箱，任务形式就是我们要训练的能力，有官方难度。缺点是只有 89 个任务，若训练也用 TB2，必须留出一部分只做评测（见 §3）。
2. **SWE-bench Verified（第二个，规模与难度分层最好）**：500 题、人工修复耗时 4 档难度（可直接当桶：<15 分钟 / 15 分钟–1 小时 / ≥1 小时 三桶，后两档太少需合并）、MIT，Harbor 已有适配器，接入工作量中等。建议从中抽 3 桶 × 30–40 题做固定评测集；判分要跑测试，单次评测成本需先小规模测量（**未测**）。

τ²-bench 和 BFCL 考的是对话式函数调用，与 codex 终端型 harness 不匹配，暂不推荐作为 codex 的主评测；AppWorld 形式接近但许可证要求加密再分发、测试集不透明，作为以后的候选。

## 3. "TB2 另外留出一部分任务"是什么意思

与数学评测集从训练集删除是同一个意思：从 TB2 的 89 个任务里固定挑出一部分，**只做评测、永远不进训练数据**，其余才给训练。否则同一批任务既训又考，分数上升分不清是学会了还是记住了。

建议留出 30 个（约 1/3），训练用其余 59 个，按官方难度分层抽：easy 2 / medium 18 / hard 10。30 个 × 每任务 4 次 = 120 次试验，通过率标准误最多约 0.046（只算采样噪声）。如果训练任务能改用别的来源（如 Harbor 的 swegym/swesmith 适配器），也可以让 TB2 89 个全部只做评测，最干净。

## 4. 需要用户拍板

1. agentic 评测选 TB2 留出集 + SWE-bench Verified 吗？
2. TB2 留出 30 / 训练 59，还是 TB2 全部只做评测、训练另找来源？
3. SWE-bench Verified 接入前是否同意先做一次小规模判分耗时测量（要上卡，另行预登记）？

## 来源
- https://github.com/harbor-framework/terminal-bench-2 ；https://huggingface.co/datasets/harborframework/terminal-bench-2.0 ；https://arxiv.org/abs/2601.11868
- https://openai.com/index/introducing-swe-bench-verified/ ；https://www.vals.ai/benchmarks/swebench ；https://github.com/SWE-bench/SWE-bench
- https://github.com/sierra-research/tau2-bench （data/tau2/domains/*/split_tasks.json）
- https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/README.md ；https://epoch.ai/benchmarks/berkeley-function-calling-leaderboard/review
- https://github.com/StonyBrookNLP/appworld ；https://arxiv.org/abs/2407.18901
- https://github.com/harbor-framework/harbor （adapters/ 目录）
