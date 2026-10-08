# 任务

前提：去耦合阶段 4（`miles_adapter/` → `adapters/miles/`）合入后再开工 1–5 组；本 PR 只含文档。

## 1. 存储与 launcher（yeto 核心）
- [x] 1.1 存储抽象：本地路径 / Modal 卷 / bucket 三种实现（写目录、显式提交、读、列出），单测
- [x] 1.2 `--rl-checkpoint-store modal-volume://`：Modal 岛挂卷（v1 卷）；Modal 上不支持的写法报错（修 launcher.py:4801-4804、4846-4850 的静默丢弃），单测
- [x] 1.3 store 同步改为 MANIFEST 逐文件 sha256 + `LATEST` 最后原子更新 + 卷显式提交；`--rl-cut-keep`
- [ ] 1.4 syncer 检查点同步到 store，head 重启从 store 取回 `--resume`

## 2. 保存（核心流程 + Miles 适配层接线）
- [x] 2.1 续训从 `--rl-elastic` 解绑：普通单岛挂轮切点；`--rl-elastic` 行为不变（金样对照）
- [x] 2.2 `--rl-cut-every N`；最后一轮与我方停机时必存
- [x] 2.3 pointer 新增 `run_fingerprint`、`lr_at_next_round`、`consumed_prompt_ids_digest`、`outer_version`
- [x] 2.4 Miles 适配层报告下一轮学习率；确认切点含 iteration/consumed_samples（已有）
- [x] 2.5 回收通知处理：Modal 中断信号只写中断标记并退出；"回收时保存"按云开关，默认关
- [ ] 2.6 （第二步）显存→内存后写卷放后台，与下一轮推理重叠

## 3. 恢复与校验（yeto 核心）
- [x] 3.1 按 design §4 顺序实现校验，任一失败报错退出；CPU 单测覆盖每种失败
- [x] 3.2 `--rl-resume-allow-config-change`，差异写入 `rl_resume`
- [ ] 3.3 多岛：外层版本对不上走重入追平

## 4. tape 与 dashboard
- [x] 4.1 事件带 `incarnation`；新增 `rl_cut_saved`、`rl_resume`、`rl_preempt_notice`
- [ ] 4.2 mismatch tape 文件名带启动序号；Modal tape 镜像改分文件追加
- [ ] 4.3 dashboard：续训分界线、每段启动开销、丢弃轮灰显

## 5. 本机验证（不上卡，不跑 Ray 测试）
- [x] 5.1 CPU 单测：存储三实现、MANIFEST、LATEST 原子性、校验失败路径、tape 不覆盖
- [x] 5.2 用假训练进程（CPU）跑"存 → 杀 → 续"，比对状态哈希

## 6. GPU 验证（交统一 GPU 表，用户批后再跑；上卡前复核 + 台账预登记）
- [ ] 6.1 Qwen3-0.6B LoRA，Modal 1×H100：
  - A：不中断 6 轮；A'：同配置再跑一次不中断 6 轮（量噪声基线）；
  - B：3 轮 → 我方停 → 续 3 轮；C：3 轮后在第 4 轮中途用 `modal container stop` 模拟抢占 → 自动续训；
  - 判据：恢复后状态哈希 = 中断前（必须）；续训第 4–6 轮学习率、题号与 A 逐位一致（必须）；loss/reward/梯度范数与 A 的差 ≤ A 与 A' 的差（或开确定性推理后逐位一致）；记录保存/恢复耗时与字节数、卷写速度。
  - 估计：4 次启动、每次约 10–20 分钟 ≈ 1–1.3 卡时 × $3.95/h ≈ **$5–8**（含 CPU 内存另计约 $1–2）[估算]
- [ ] 6.2 FN 全尺寸（LoRA）切点保存/恢复，Modal 2×8 H200（s16 已验证布局，最少卡）：2 轮（每轮存）→ 我方停 → 续 2 轮；不跑不中断对照，判据为恢复哈希一致、版本号/学习率/题号接上、恢复后发布成功、指标与前 2 轮同量级；实测切点体积、各阶段保存耗时、2 节点并行写速度、恢复耗时。
  - 估计：2 次启动（每次启动开销约 25–35 分钟）+ 4 轮（WP1 后约 4.5 分钟/轮，前约 11 分钟/轮）+ 保存恢复 ≈ 1.5–2.3 小时 × 16 × $4.54/h ≈ **$110–170** [估算]
  - 省钱备选：并入全量训练第 1 段（4×8）末尾：第 1 段结束存切点、第 2 段启动即为恢复验证，不额外花钱，但若恢复失败第 2 段的启动开销（约 $120–150）白花。
- [ ] 6.3 （可选）跨云：把 6.1 的切点经 CPU 任务拷到 S3，在 AWS spot 单卡恢复一次；约 $1–3 [估算]

## 进度（S17 C4，N8，分支 s17-resume-impl）
- 已实现 + CPU 单测：1.1、1.2、1.3（非 elastic 路径快照 + LATEST + commit + `--rl-cut-keep`；elastic 路径平铺拷贝加每文件 sha256 并在恢复前核对）、2.1–2.5、3.1、3.2、4.1（`rl_resume`、`rl_cut_saved`、`rl_resume_check`、`rl_stop_requested`；`incarnation` 在 `rl_resume` 与切点指针里，未加到每条事件）、5.1、5.2。测试 `tests/test_rl_resume_from_checkpoint.py`（22 例）。
- 2.4 说明：下一轮学习率由核心按调度公式算（Miles 参数名在适配层 `entry.next_lr_for`），未让 Miles 进程上报；iteration/consumed_samples 已在切点里。
- 2.5 说明：信号处理器只写 `preempt-notices/*.json`；Modal 的停止信号能否传到 learner 进程未验证（G2 C 顺带看）。"回收时保存"未实现（默认关，等实测保存时间）。
- 部分：4.2 只做了 Modal tape 镜像按容器分文件（`.inc<k>`，不覆盖旧容器）；mismatch 目录写法未改（见 design §9）。4.3 只做了 dashboard 数据层。
- 未做：1.4、2.6、3.3（多岛岛内状态：切点仍只在 LocalOnlySync 上挂；多岛需要每岛切点 + syncer 检查点进 store + 外层版本比对，见 design §4.7、§6）。
- 6.1：见 `infra-drafts/S17-G2-PRELAUNCH-REVIEW.md` §7（结果）。
