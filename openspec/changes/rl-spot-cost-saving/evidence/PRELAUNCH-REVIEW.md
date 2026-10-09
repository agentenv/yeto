# S19 #9 rl-spot-cost-saving 3.2 评测岛抢占演练：上卡前复核（子 agent 写，开卡前完成）

## 跑什么
- 真的评测岛 Modal 函数（`yeto.cloud.modal_eval_island.build_eval_app` 原样，main 05918cfd + 本分支 s19-b3x）。存储是新 Modal Volume `yeto-eval-store-s19spot`，单元日志、#180 的回收处理（`install_reclaim_handler`）、launcher 重开都走真代码。
- 推理与判分换成演练绑定 `yeto.rl.eval.drill`（本分支新增，子 agent 代拍板）：不加载模型、不开沙箱，每个单元等 15 s 后返回固定结果。理由：3.2 要验的是回收处理、提交、续跑，与模型无关；用真模型要 codex 契约、SGLang、TB2 沙箱，一次约 $2.5 且失败点多。
- 题目：TB2 留出集 4 题 × 第 0 版 2 次 = 8 个单元。
- 卡：Modal `T4` 1 张（最便宜的卡；评测岛函数开头调 nvidia-smi，所以要有卡），2 核 / 8 GiB，函数超时 1500 s，launcher 最多开 3 次。
- 模拟抢占：本机驱动等 Volume 上已提交 3 个结果后，用 `modal container exec` 找到容器里 `_container_entrypoint` 进程，发 SIGINT（Modal 抢占时发的信号）。

## 代码复核
1. `install_reclaim_handler` 只在主线程装信号处理（`ModalExitHandler.install` 非主线程返回空）。Modal 同步函数是否在主线程跑，未验证；这正是本次要看的事实之一。若没装上，D1 失败，照实记录。
2. 信号到达时主线程在 `time.sleep` 里；处理函数设停止标志 → 按上次提交耗时判断够不够 → 在副线程提交 → 写 `spot_reclaim`。当前单元跑完后，循环检查停止标志并抛 `Preempted`，`island_main` 的 finally 再提交一次（事件随之落盘），函数报错 → `call.get()` 抛异常 → launcher 记 `preempted` 并开第 2 个容器。
3. 演练是模拟：容器不会被平台真杀，所以"30 s 后被强杀"这一段不在本次覆盖内。
4. 本机 CPU 预演（/tmp/drilltest.py，真 SIGINT 打本进程）：抛 Preempted、写 1 条 spot_reclaim、续跑后 eval/units=4、重复 0。
5. 无 Miles 初始化（不起训练），显存无占用。

## 预登记判据（全部满足才算通过）
- D1 只有 1 条 `spot_reclaim`：cloud=modal、role=eval、source=modal_signal、remaining_s=25、saved=true、outcome=saved。
- D2 处理耗时 handler_s ≤ 25 s；发信号到 launcher 记 `preempted` ≤ 60 s。
- D3 launcher 开了第 2 个容器并 `finished`。
- D4 续跑不重不漏：最终 `rl_eval` 的 eval/units = 8、eval/duplicate_results = 0；结果文件每个单元恰好 1 行。
- D5 eval/results_sha256 等于本机一次跑完同一计划的值。
缺证据记"失败（证据不全）"。

## 花费
T4 $0.59/h + 2 核 + 8 GiB ≈ $0.75/h；两个容器加拉镜像约 15 min ≈ $0.2；最坏 1500 s × 3 次 ≈ $0.95。上限 $5。

## 试跑 6 上卡前复核（10-09 22:3xZ，#9 子 agent）
- 试跑 5 根因调查：
  1. Modal app 日志（`modal app logs ap-OWFji6…`）显示：第 1 个容器收到 SIGINT 后，回收处理写了 spot_reclaim（saved），抛 Preempted，launcher 记 preempted。这一段产品代码按设计工作。
  2. launcher 第 2 次起函数时，Modal 把输入送回了同一个已收信号的容器（演练只发信号、不杀容器，容器还热着）。3 s 后该输入报空的 RemoteError，日志随后出现 "Stopping app - user stopped from dashboard"，第 3 次起函数报 ConflictError。
  3. 谁停了 app：未查明。本机记录里没有任何 agent 在 15:02–15:04Z 对这个 app 执行 `modal app stop`。
  4. CPU 复现（yeto-s19-sigrepro，同样向函数进程发 SIGINT，再起 2 次输入）：没有复现，app 一直运行，后续输入都完成。
- 判断：真实 Modal 抢占会结束被回收的容器，下一次输入一定落到新容器，app 不会停。产品 launcher 不需要改。问题出在演练与真实抢占不一致（容器没被结束）。
- 演练 harness 修改：launcher 每次重开前，先 `modal container stop --yes <已收信号的容器>`，等它从容器列表消失，再起函数。CPU 复现已检查：容器停止不会停 app，下一次输入落到新容器并完成。
- 另加：保存 `modal volume get` 的输出（试跑 5 卷导出为空，原因未记录）和结束时的 `modal app list`。
- app 名改为每跑唯一（yeto-s19-spot-drill-f），避免与旧跑的看门狗按名字误停。
- 判据 D1–D5 不变。

## 试跑 6 结果：通过（10-09 22:31:46–22:36:11Z，#9 子 agent 判读）
- 运行目录 /home/michael/work/s1-runs/s19-spot-drill-20261009f；app ap-cZkqrtJHdKSMlxNgBe2JXI；卷 yeto-eval-store-s19spot-f。原始数据已传 yeto-evidence-archive:/openspec/changes/rl-spot-cost-saving/evidence/gpu/s19-spot-drill-20261009f。
- 时间线：22:33:30Z 第 3 个结果落盘后向容器 ta-01M4HCPZ… 的函数进程（pid 2）发 SIGINT；16.1 s 后 launcher 记 preempted；harness 停掉该容器（33.3 s 后从列表消失）；第 2 个容器 ta-01M4HCV4… 跑完剩余单元，launcher 记 finished。
- 判据：
  - D1 通过：只有 1 条 spot_reclaim，cloud=modal、role=eval、source=modal_signal、remaining_s=25.0、saved=true、outcome=saved。
  - D2 通过：handler_s=0.004 s；发信号到 preempted 16.11 s（≤60 s）。
  - D3 通过：launcher 第 2 次起函数并 finished（新容器）。
  - D4 通过：rl_eval 的 eval/units=8、eval/duplicate_results=0；结果文件 8 行 result，每个单元恰好 1 行（另有 8 行 start）。
  - D5 通过：eval/results_sha256 = 本机一次跑完的值，两者都是 5ae51639731cc329a58963bad512e3a12662b8cb363170f578824d449dbf2641。
- 判读说明（如实记录）：跑内的 `modal volume get` 失败（"[Errno 21] Is a directory"，见 volume-get.out），跑内判读因此全记失败（judgment.in-run.json）。跑结束后，子 agent 从同一个卷（app 已停，无人再写）重新下载到 volume2/，用同一判读函数重判，得到上面的结果（judgment.json）。harness 已改为先建目录、在 /tmp 下执行下载。
- 未覆盖：平台 30 s 后强杀容器这一段（演练用 `modal container stop` 代替）；真实 Modal 抢占未遇到。
- 试跑 5 的 app 被停（"user stopped from dashboard"）原因仍未查明；试跑 6 改为重开前先停已收信号的容器后，没有再出现。
