# S29 弹性测试进度（每个大阶段一行）

- 10-31 00:26 阶段0 准备：测试设计文档完成；trainer 迁到 n4 卡0/1（避开链路抖动卡4/5），syncer 在 npu1 CPU；spot-a(eu-north1 H100) 已出样本，spot-b 换为 us-central1 RTX6000 启动中；B200 已 down。
- 10-31 00:35 阶段1 基线 T1 进行中：发现并修复 D1（FSDP buffer trainer prompt 行宽 512）、D3（不足 n 条的组让 update_actor 断言失败）；n4 Ray 节点 IP 问题已修；spot-b 换 RTX6000 已 UP；boat 已用 4 次（VM 复用生效）。
- 10-31 00:59 阶段1 基线 T1：两 trainer 在 n4 持续运行（岛0 训 3 轮、岛1 训 1 轮）；00:42 起停滞的原因是 D5（spot 岛 backlog 在首个合并 cut 前从不下降，与 fragment 逐个合并互相等待，死锁），已修复并重启两个 spot 岛；T6.2（SIGTERM→PREEMPTED 离开、boat 沙箱自动清零）通过。
- 10-31 01:02 核实（回应巡检）：trainer 在 n4 = 192.168.17.6（hostname host-9tird90004），island_entry PID 1191336 1191341 ；ASCEND_RT_VISIBLE_DEVICES 卡 0/1。tape 证据：island 0 train_step 3 samples 4 17:02:13Z；island 1 train_step 2 samples 4 17:02:21Z（tape：n4 /data/work/s29-spot/run/i{0,1}/tape.jsonl）。D5 修复后已恢复：relay channel1 stored=2 acked=2，岛0 训到第 4 轮、岛1 训到第 2 轮。0.6B 单卡显存占用很小，所以 npu-smi 看着接近基线。
- 10-31 01:09 阶段1 完成：T1 基线（TB2.1×Codex×boat）两 trainer 岛各训满 5 step（每步 4.4–9.6 s，reward 全 0），共发现 6 个缺陷（D1/D3/D5/D6 已修，D2 关闭，D4 待测）。boat 已用 194/200（剩 6）。阶段2 开始：GSM8K 长跑 40 轮/岛 + 故障注入。
- 10-31 01:54 阶段2 长跑中断：两 trainer 岛在本地第 57/91 步因'整轮样本超龄'退出（D10）；根因是 npu1→Nebius 隧道 <250KB/s，spot 岛拉一个 1.2GB 的 cut 要几十分钟，跟不上版本；另发现 D7（rounds 口径）、D8（kill -9 后 SGLang 孤儿占显存，已修）、D9（重启的 spot 岛先在 v0 上出样本）。下一步：修 D10 的致命退出，放宽 policy age 继续长跑与演练。
- 10-31 02:02 阶段2 继续：D9/D10 修复后重启长跑（policy age 4，verl 上限）；关键观察：npu1→Nebius WAN 约 0.18 MB/s，1.2 GB 的 cut 很难跟上合并节奏，长跑主要检验'超龄丢弃+等待'是否稳定、是否泄漏。修复已提交 6ee7d5c9。
