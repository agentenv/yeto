# rl-resume-from-checkpoint：跨启动续训

## 为什么
- FN 全量训练要分段跑（FN-FULLTRAIN-CONFIG-REVIEW-S17.md §10.2–§10.3：20 / +30 / +150 轮），而 Modal 单个函数最长 24 小时（https://modal.com/docs/guide/timeouts），200 轮估 28–63 小时，一次启动跑不完；Modal GPU 函数默认可被抢占且不能关（https://modal.com/docs/guide/preemption）。所以"停了能接着训"是全量训练的前提。
- 现状（读码，agentenv/main dcde202e，详见 design.md §1）：
  - 单岛"轮切点"（round cut）已经能存并恢复 LoRA adapter、优化器（含 FP32 主副本）、学习率调度、四种随机数、Megatron iteration、数据游标、策略版本号，恢复时逐片 sha256 + manifest 哈希 + 恢复后重算策略哈希，任一不符报错退出。但它**只在"开 `--rl-elastic` + 单岛不同步 + 设了 `--rl-checkpoint-store`"时生效**。
  - **Modal 上 `--rl-checkpoint-store` 被静默丢弃**：Modal 岛配置不读存储挂载，并显式过滤 `s3://` 等前缀，learner 写进容器本地目录，换容器即丢，没有任何报错（launcher.py:4801-4804、4846-4850）。s16 的 FN 运行每次都从底座开始。
  - 多岛只能续上 syncer 的外层底座和轮号；岛内 adapter/优化器/随机数不恢复；syncer 检查点写在 head 本地盘（launcher.py:859-862），head 机器没了就没了。
  - store 复制本身没有哈希校验，只看 STORE-MANIFEST 是否存在（controller.py:714-740）。
  - mismatch tape 按轮号以覆盖方式写，续训重跑同一轮会覆盖旧记录；Modal 的 tape 镜像是整文件替换，换容器后可能把卷上的旧 tape 覆盖（推测，未实测）。dashboard 没有"续训"概念。

## 改什么
1. **存储**：`--rl-checkpoint-store` 新增 Modal 卷写法（`modal-volume://<卷名>[/前缀]`），Modal 岛把卷挂上并传给 learner；Modal 上给出不支持的写法时**报错而不是静默丢弃**。sky 岛继续用 bucket（S3）或共享盘绝对路径。
2. **保存**：沿用现有轮切点格式；新增"每 N 轮存一次"（默认 1）与"保留最近 K 个"（默认 2）；store 同步改为"先写数据、逐文件 sha256 进 STORE-MANIFEST、最后原子写 MANIFEST"。
3. **恢复校验**（全部在 yeto 核心做）：store 哈希一致 → 切点片哈希一致 → 策略哈希重算一致 → 策略版本号接上（下一轮 = 切点轮 + 1）→ 学习率接上（恢复后第一轮学习率等于不中断时该轮学习率）→ 数据不重复（数据游标接上、已用题号与新题号无交集）→ 运行配置指纹一致（算法、并行布局、模型、LoRA 形状、学习率配置；改了就拒绝，除非显式 `--rl-resume-allow-config-change` 并写入 tape）。
4. **把续训从 `--rl-elastic` 解绑**：单岛不同步的普通运行也能用轮切点（新开关 `--rl-resume`，存储给了就默认开）。
5. **多岛**：syncer 检查点同步到同一 store；每个岛的岛内状态各自存轮切点；恢复时岛内切点必须与 syncer 外层版本对得上，否则该岛按"重入"从外层底座追平（沿用岛间调度已验证的重入路径）。
6. **tape 与 dashboard**：续训写入同一运行目录，按"启动序号"（incarnation）分段；事件流追加；mismatch tape 文件名带启动序号，不覆盖；Modal tape 镜像改为追加/合并；dashboard 显示续训分界线与每段启动开销。
7. **抢占**：Modal 回收通知只有 30 秒（见 design.md §2），FN 全量切点写不完；所以 FN 在 Modal 上只靠"定期切点"，回收时不尝试写全量切点，只写"中断标记"；I1 级别的"回收时保存"只对通知时间够长的云开启（实测保存时间后再定）。

## 不改什么
- 不改轮切点的文件格式与已有校验（只加）；不改 tape 已有字段语义；不改训练数学。
- 不做跨卡型/跨并行布局的续训（`restore_resharded_shard` 只用于运行中改数据并行，跨启动改布局留到以后）。
- 不做全参训练的续训（只做 LoRA；全参切点体积另评）。

## 影响
- 代码：yeto 核心（存储抽象、校验、续训流程、launcher 的 Modal 挂载、tape/dashboard）；Miles 适配层（切点里框架相关的保存/恢复，已有，基本只接线）。去耦合阶段 4 会把 `yeto/rl/engine/miles_adapter/` 搬到 `yeto/rl/adapters/miles/`，本 change 的实现应在阶段 4 合入后开工，避免撞文件。
- 费用：Modal 卷存储 $0.09/GiB/月、每月 1 TiB 免费（https://modal.com/pricing），FN 一个切点约 45–80 GB（估算），保留 2 个在免费额度内。
- GPU 验证：见 tasks.md 第 6 组，交统一 GPU 表由用户批。
