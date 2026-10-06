# fix-verda-provider 进度

## 2026-09-29（Agent V）

- 规划文件从 `fix-verda-provider`（main e21a7ff，未提交）搬到新分支 `fix-verda-provider-r0`（基于 `origin/rl-engine-ports` d355e06），旧分支保留。
- tasks.md 共 19 个任务（1.x 3、2.x 5、3.x 2、4.x 3、5.x 1、6.x 5）。
- 已完成（CPU 单测 + 真实 sky 0.13.0 适配器上的假客户端场景）：1.1–1.3、2.1–2.5、3.1–3.2、5.1。证据与细节见 tasks.md 的"完成记录"与 `evidence/`。
- 全量 pytest：前 68 failed / 1984 passed / 26 errors；后 68 failed / 2037 passed / 26 errors。失败集合（94 个 id，均为已知环境问题：缺 syncer 二进制、miles、boto3 等）前后完全一致，没有新增失败。
- 本轮没有创建任何云资源，费用 $0。

## 2026-09-29 第二轮（独立审查修复）

- 审查指出 2.4、3.2 用显示名匹配主机名（真实为 `cluster_name_on_cloud`，含 user hash），先取消勾选，修复并以 sky 真实命名测试后重新勾选。其余 7 条（远端/新写 `.pth` 视为未生效、token 刷新、teardown 只处理记录 id、`_FAILED_LAUNCH` TTL、`.pth` 惰性化与清理、并行诊断、探测不依赖 exec 阻塞语义）均已修复，见 tasks.md。
- 本机负载高时，导入 torch 的 pytest 进程会无输出退出（在基线 d355e06 上同样复现，OMP_NUM_THREADS=1 可避免）；本轮前后对比均在 `OMP_NUM_THREADS=1` 下跑：基线 68 failed / 1984 passed / 26 errors，修复后 68 failed / 2044 passed / 26 errors，失败 id 集合（94）完全相同。
- 未在 `yeto down` 中移除 `.pth`（运行被 SIGKILL 时残留）：可用 `python -m yeto.sky_patches uninstall` 清理。

## 阻塞

- 4.1（虚拟机内容器）、4.2（规划器放行 Verda）、4.3（CLOUDS.md）等 PR #69（`rl-engine-ports`）合入 main 后再做。在 4.1 完成之前，Verda 学习岛仍然会带 `image_id`，而 sky 的 Verda 不支持 docker 镜像（R5），所以 6.1、6.2 的训练部分、6.3、6.5 也都依赖 4.1。

## 6.x 真机验收计划（尚未运行，等用户批准）

公共约束：集群名 `vfix-` 前缀、小写；Verda 岛 `recover_timeout=0`（6.3 除外，该项专门验证恢复）；head/syncer 放 Nebius（公网 IPv4 配额只有 3 个）或本机；拆除前拉回日志和事件磁带；拆除以 Verda API 为准，按实例 id 核实并清空回收站卷；不做全局清理。

| 项 | 内容 | 资源 | 预计时长 | 预计费用 |
|---|---|---|---|---|
| 6.2a | 补丁/误删回归：起 1 台 `1L40S.20V`（或 `1A100.22V`），再以同名触发一次必然 503 的重拉（例如请求 count=2 或在无货地区），确认旧实例仍在运行；查 head 的 API 服务进程 `/proc/<pid>/maps` 或打印 `yeto.sky_patches.status()` | L40S $1.543/h 或 A100 80GB $1.797/h；Nebius CPU head 约 $0.1/h | 约 0.5 h | 约 $1 |
| 6.3 | 恢复：手动删除岛实例，yeto 按 id 确认已消失后以 `-r1` 重拉 | 同上 | 约 0.7 h | 约 $1.5 |
| 6.4 | Verda head（另核实 `sky.exec`+`stream_and_get` 是否阻塞到作业结束）：`CPU.4V.16G`（$0.048/h）起 head，外部探测 syncer 端口，nmap 验证其他端口被 ufw 拒绝；岛用 Nebius 或 Verda L40S 完成 1 轮同步 | CPU 节点 + 1 张 L40S | 约 1 h | 约 $2 |
| 6.1 | 默认参数单卡 RL 岛（需 4.1 完成）：容器内 fork checkout、校验、至少 1 轮同步；拆除证明 | 1 张 A100 80GB / L40S + Nebius head | 约 1.5 h（镜像拉取约 6 分钟） | 约 $3 |
| 6.5 | `rl-engine-ports` 7.1 的 Verda 默认参数真实运行：2 岛 strict-avg 3 轮（需 4.x 完成） | 2 张 A100 80GB + Nebius head | 约 1.5–2 h | 约 $6–7 |

合计约 $13.5–14.5，在 $15 上限内。按"6.2a → 6.3 → 6.4 → 6.1 → 6.5"的顺序执行，每一项结束都先拆除并核实，再开始下一项。运行中每 10 分钟用 Verda API 按 id 核对一次在跑实例，并累计费用；预计总额一旦超过 $12，停下汇报。6.2a、6.3、6.4 不依赖 4.x，批准后即可先做。
