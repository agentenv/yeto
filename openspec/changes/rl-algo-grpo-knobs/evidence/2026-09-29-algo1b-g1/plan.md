# algo1b-g1 计划（task 8.1/8.2；实验前提交，事后不改）

## 放置与环境
- 单岛、无 outer sync：P0 的 `--rl-single-island-no-sync` 加 `--rl-allow-unverified-mechanism dimension:name`（algo-cap 2f9f02c/09607d6）。harness 沿用 rl-engine-ports R0 冒烟（`2026-09-29-rerun-harness`，经 benchmark_rl worker 调用 `run_miles`），只把 spec、放行开关和 no-sync 写进 worker 参数。
- 镜像：`radixark/miles@sha256:90940828…`，其中检出 miles-next 0394715 与 sglang-next 9f29303（与 R0 冒烟相同，见 `sbx.py`）。yeto 代码为 `git archive HEAD`，SHA 写入 `harness/YETO_SHA` 并随证据保存。
- GPU：Modal Sandbox，`H100!`×1（禁止升级为 H200），cpu 16、内存 128GiB；`setup.sh` 用 nvidia-smi 断言 GPU 为 H100、不是 H200，断言失败就退出。
- 模型与数据：Qwen/Qwen3-0.6B@c1899de2，gsm8k 训练集前 36 题；LoRA r16 all-linear；每轮 4 组 × 8 条，rollout_max_response_len 384，seq_len 1024，lr 1e-5，seed 17；每个机制 3 轮。
- 前缀与资源：app `algo1b-g1`（前缀 algo1b-），sandbox id 记在 `out/sandbox_id`。

## 机制（每项单独运行一次）
| 名称 | spec |
|---|---|
| baseline | 默认 GRPO（KL 显存与耗时对比用） |
| clip_higher | eps_clip 0.2 / eps_clip_high 0.28 |
| dual_clip | eps_clip_c 3.0 |
| token | aggregation=token |
| drgrpo | std_normalization=false + constant D=384 + vendor reducer |
| kl_k3 | kl loss 0.001 k3，ref_model=Qwen/Qwen3-0.6B@c1899de2 |
| entropy | entropy_coef 0.001 |
| over_sampling | bounded filter，max_replacements 2，over_sampling_batch_size 8 |
| overlong_penalty | 分派器，overlong_penalty max 384 / cache 128 |

**不在本轮范围**：overlong_filter。它的 hook 接线（1b-hook.patch）尚未合入，只能在合入后另立计划运行。

## 事先声明的判定（每个机制独立判定）
- 通过需要同时满足：
  - worker rc=0，事件带中完成 3 轮训练；
  - 事件带中没有 `zero_grad_norm_with_nonzero_advantages`、`nonfinite_grad_norm`、policy token 不匹配、receipt 失败，也没有别的 invariant 错误；
  - 相关指标键存在且全部有限：
    - clip_higher、dual_clip、token、drgrpo：训练指标中的 pg_clipfrac（或 clipfrac）与 pg_loss；
    - kl_k3：kl_loss；
    - entropy：entropy_loss；
    - over_sampling：每轮 rollout 元数据中的 completed/filtered 组数（写进报告；7.2 的事件字段另算）；
    - overlong_penalty：日志或事件中出现 `rl_reward_shaping`，且原始与塑形后奖励都有限；
    - baseline：pg_loss 与 grad_norm。
- KL loss 另外记录峰值显存（nvidia-smi 每 2 秒采样取最大值）和每轮耗时，与 baseline 对比。不设门槛，OOM 记为"该卡型不支持"。
- 失败：不满足上述任一条。该机制不声明，记录原因。同一失败只有在提出原因并修复后才重跑，不挑 seed。
- 不做数值对比或收益判断（G4 不在范围内）。

## 时长、费用与回收
- 预计：setup 约 15 分钟，每个机制约 6–10 分钟，合计约 1.5–2 小时。H100 约 $4/h，加 CPU 和内存，预计 ≤ $12。
- 硬超时：
  - sandbox `timeout=10800` 秒（云端强制）；
  - 独立 watchdog（setsid nohup，与终端和 agent 都无关）在 11100 秒时按 sandbox id 执行 kill；
  - 每个机制 exec 超时 1800 秒，worker 内部 subprocess 超时 1700 秒；
  - 本地 `timeout 11400` 包裹 run_all.sh；
  - EXIT trap 按 id 终止 sandbox。
- 结束后先拉取日志和事件（out/<mech>），再终止 sandbox；用 `modal app list` 和 sandbox 列表核实无残留，结果写入 `out/teardown_proof.txt`。不创建卷。

## 第 1 次尝试（sandbox sb-fwUsN4xXObdicf3CYM1dIm）结论
- setup 通过（SETUP_OK，GPU 为 NVIDIA H100 80GB HBM3，GPU 断言通过）。
- 所有机制都在启动前失败：exec 命令写 `/work/out/<m>.log` 时 `/work/out` 目录不存在。这是 harness 的 bug，不涉及被测代码，也没有训练过任何一轮。
- 修复：exec 前先 `mkdir -p /work/out`。其余计划不变，按第 2 次尝试执行。
- sandbox 已由 EXIT trap 终止（输出 terminated），watchdog 已停止，`sbx.py list` 为空。日志在 `attempt1/`。
