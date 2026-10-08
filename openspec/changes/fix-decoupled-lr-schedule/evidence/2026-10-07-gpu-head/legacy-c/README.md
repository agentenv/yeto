# legacy-c：s14-dlr-legacy-20261007c（2026-10-07T13:51–14:01Z）— INCOMPLETE，3.2 仍未勾
- 代码 s14-legacypp@（见 yeto_sha.txt）：legacy 引擎在 ports 镜像内也把 /root/Megatron-LM 加进 island PYTHONPATH（修 b 的假设根因）。
- 结果：两岛 H100 确认（failure-key-lines.txt），Miles 启动在同一处再次失败：
  `/root/miles/miles/backends/megatron_utils/arguments.py:5 from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding` → `No module named 'megatron.training.tokenizer'`。
- 真正根因：agentenv/miles（legacy，ae475060）依赖旧 Megatron-LM 的 `megatron.training.tokenizer`；ports 镜像的 Megatron-LM（core 0.19）已移除该模块（ports fork 改用 `megatron.core.tokenizers.utils.build_tokenizer.vocab_size_with_padding`，见 miles-next 的同名文件第 4 行）。PYTHONPATH 修复必要但不充分；legacy 不能在 ports 镜像里跑。
- 岛 0 失败后被 launcher 自动 relaunch 一次（同样失败），随后手动 teardown（early-*）；0 轮、无 tape。legacy 对照需用 legacy 自己的 MILES_IMAGE（ghcr.io/agentenv/miles 私有 → `--rl-image-private` + read:packages token）。
- 费用（估算）：Modal app 13:56:00–≈14:00:19，2×H100 ≈ $0.57 上界；Nebius VM 13:52–14:01 ≈ $0.03；合计 ≈ $0.6。
