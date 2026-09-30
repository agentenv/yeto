# 镜像内 learner preflight（Modal T4，B2，主 agent 批准；2026-09-30）

- 代码：infra-e2 = integ-decl e052dc8（`MILES_NEXT_IMAGE` 已修正为 db815884）。镜像 `sha256:db81588406e157baa6a579f6378484b890371065abcc51eacd5a9650b5820cbf`（2f23a0f-9f29303）。
- 资源：Modal Sandbox，T4 ×1（只用来让 libcuda 可以 dlopen），2 核 / 8 GiB，sandbox timeout 1080 s，本地 timeout 1200 s，独立 watchdog 1320 s。app `ap-stLfBcIQrfvS6YVKTKqmNr`，06:13:55–06:23:57Z，已 stopped/0（`teardown_proof.txt`）。
- 费用：≤ $0.13（上界估算，未取账单）；上限 $0.30。
- 内容：对 c1、c2、c3-rb 运行真实 `yeto.rl.learner.main` 加 `--rl-print-attestation-fingerprint`（Bridge provider → translate → Miles parse_args/validate → verify_ports_algorithm，之后退出，不起 Ray、不训练），并在解析后的 Miles 参数上做 E2 静态检查；另跑 runtime manifest。
- 结果：**三项全部通过**，manifest rc=0。
  - 解析结果：requested_load=None，deterministic_mode=True，bf16，DistOpt，TP/PP/CP/EP=1，c1 DP1、c2/c3 DP2，GBS 16，bridge 模式。
  - manifest：miles 2f23a0fc，sglang 9f29303，megatron.core 0.19.0+a84b10547，torch 2.13.0+cu130，与 pin 一致。
- 偏差：c3-rb 这次的参数是 `lora_dropout 0.0`。plan-v2 §0 规定 C3 除模型外与 C2 相同，应为 0.05。工具已修正（C3 加 `--rl-lora-dropout 0.05`），本地 dry-run 已按修正版重跑，22/22 rc=0（`dry-run-v4b-20260930/`），但镜像内 preflight 没有按修正版重跑 c3。c1/c2 已在镜像内验证过 dropout=0.05 这条路径。
