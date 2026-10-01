
## 2026-10-01 A4/A4b 8 卡 H100 批（会话 4，用户改用 Nebius 8×H100；代码 b19b781；计划 gpu-plan-v2 §9.23）——中间状态
- 分支 gpu-b1（普通 push，最新含本节）；已合入 origin/integ-decl 到 d1cc74e（tasks.md 勾选，非 yeto 变化）；**尚未合入 integ**（按主 agent 指示批末一起合）。证据与 RESULT：`evidence/infra-v2-b1/a4-8card-s4/`（`RESULT.md`、`runs/{smoke,base,e1a,e1b,e1b-attempt2-provision-failed,chain-aborted}/`、`scripts/ cfg/ tests/`）。
- 完成：冒烟（up 144.3 s，start_cells 142.4 s，≤160 s）；E1-A 基线+切换 **通过**（(c) 首次判读 FAIL 为 judge 单位缺陷，勘误+回归测试先提交再重判）；E1-B **不通过**（注入已应用，终态 SUCCEEDED；疑读回校验不覆盖 LoRA adapter）。
- 未运行：watchdog、A4b、E1-D d123/d4/d5/d6/d7——主 agent 通知 integ-decl 已到 06a754bf（LoRA 准入 fail-closed，需重建镜像 sglang a1240c530，用户已批准、进行中）；之后在新 SHA+新镜像上重取 attestation 指纹、重核各开关 argv，按 wd → A4b → E1-D → E1-B（验证修复）运行，用复用链，本批剩余预算 $40。
- 工具/缺陷（均有 CPU 回归测试）：cleanup 闭环（nstop 的 tee 被前缀扫描杀死 → rc 141，已修）、周期性拉取采样小文件包、judge_inject 单位勘误、`scan_run.py`/`diag_pull.sh`/`save8.sh`、集群复用链 `chain8.sh`/`reset_island.sh`/`nstop_item.sh`（真机未验证）。
- 费用 ≤$60.0（含主 agent 误叫停的链启动 $5.7）；云资源：无（nebius 仅他人实例，sky 无集群；各 run cleanup 退出码 0）。
- 下一步：等主 agent 给新 SHA；`mkatt8.sh` 在新 SHA 上重取 3/4/5/12 轮指纹并核各开关；`chain8.sh <CP> 60.0 wd:1920 a4b:1200 d123:1800 d4:1200 e1b:1200 ...`（门控逐项）。
