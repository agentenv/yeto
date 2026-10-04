# 4.4 完整 parse_args（钉住镜像，Modal T4）

- 计划与三次运行记录：plan.md（运行前提交 d083165；run1/run2 失败原因与修复分别在 e5f5a83、a215d82 中先提交再重跑）。
- 有效运行：run3，app ap-CwXDZTPSL1AoyMskEPagWU，sandbox sb-QcM2Zv7kRqxskIgHrYdON4，Tesla T4，约 1.5 分钟，sandbox returncode 0。
- 结果：result.json（`overall: PASS`，含三个变体的完整 argv 原文、解析值、拒绝组合与异常信息）；run3.log 为原始输出（run3.log 中 VALID cispo/sapo 两行被 `tail -80` 截掉，以 result.json 为准）。
- 驱动：modal_t4_parse.py（Sandbox，timeout=840）、full_parse.py（容器内脚本）、watchdog.sh（独立回收）。
- 回收：teardown_proof.txt（三个 algo2b-t4-parse app 均 stopped，无 watchdog 残留）。
- 费用：billing_2026-09-30T0415Z.json 为 Modal 账单（run1 $0.0315、run2 $0.0386；run3 当时尚未入账，按 ~1.5 分钟 T4 估约 $0.03）。合计约 $0.10，上限 $2，单独记账。
- 凭据：拉取凭据仅作 from_registry pull secret；证据目录对 token 与 base64 auth 做了扫描，无命中。
