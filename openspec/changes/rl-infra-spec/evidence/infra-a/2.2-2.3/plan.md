# 2.2 + 2.3（X9 guard）GPU 实验计划（INFRA，事前计划；本文件单独提交，得到 SHA 后才启动）

## 验收原文
- 2.2："partitioned-serial 完成固定算法步数，不因分卡改变 sample IDs/optimizer 时序；保留 R0 每组 policy token 校验"。
- 2.3（本轮只覆盖其中的 X9 guard 部分）："延迟发布不能触发旧版本生成，队列有界"。本轮**不交付** 2.3 的整体结论：age 0 下合法的重叠对（train‖eval、outer_sync‖eval、generate‖checkpoint 等，见 tasks 2.3 进展）尚未实现，2.3 保持未完成。

## 代码与镜像
- yeto：本计划提交后的 HEAD（冻结快照上传；reward 文件与 1.2 相同，sha256 743d433c…）。镜像 `MILES_NEXT_IMAGE` 5da40a07（不含 M-fork；T1R1 不需要 standby，不需要 M1）。
- profile：与 1.2 相同（Qwen3-0.6B LoRA r16 all-linear、GRPO、strict-avg、bf16、rollout-batch-size 4 × 8、response 384、seq 1024、lr 1e-5、seed 17、total-steps 3），**单 island**（1 个 learner，本机 head）。

## 三个 arm（各一次 launch，前缀 `infra-a-e0a` / `infra-a-e0b` / `infra-a-e0c`）
| arm | 放置 | GPU | 目的 |
|---|---|---|---|
| A | colocated-serial（T1 共置） | modal:1xh100（`H100!`） | 对照：同 trainer DP=1 |
| B | partitioned-serial T1R1（`--rl-placement fixed-partition --rollout-num-gpus 1`） | modal:2xh100（`H100!`） | 2.2 |
| C | 同 B，另在快照根目录放 `yeto-rl-fault-injection.json` = `{"publish_delay_s": 30}` | modal:2xh100（`H100!`） | 2.3 X9 guard |

## 事先登记的判定（不设数值容差；数值只记录）
2.2 通过需同时满足：
1. B 的 learner 为 SUCCEEDED，`rl_driver_start.execution_mode == "partitioned-serial"`，没有 `rl_strict_failure`。
2. B 完成的轮数等于 A 完成的轮数，且每一轮 `rl_round_trained.trained_sample_ids_sha256`、`trained_groups`、`trained_samples` 与 A 同轮逐一相等（sample IDs 不因分卡改变）。
3. optimizer 时序：A 与 B 每轮恰好一次 train 阶段，且每轮 `applied_lrs` 长度相同（=1）；每轮 phase 顺序在 B 中为 generate→train→sync→publish，与 A 去掉 offload/onload 后的顺序相同。
4. 每轮 `rl_publication` 成员非空，publication token 与 apply 哈希一致（每组 token 校验未触发 `PolicyIdentityError`）。
X9 guard（C）通过需同时满足：
5. 每次发布前都有 `rl_fault_injected`（publish_delay 30 s）事件，且 C 为 SUCCEEDED。
6. 对每一轮 r：`generate(r)` 的 phase 时间晚于 `rl_publication(policy_version=r)` 的时间；从 publish 阶段开始到 `rl_publication` 之间没有任何 generate 事件（队列有界：在途 batch ≤ 1）。
7. 没有 `PolicyIdentityError` 或 `rl_strict_failure`。
任一条件不满足即判该项未通过，并记录原因；只有找到原因并修复后才重跑，不修改本条件。

## 资源、费用、回收
- 预计每个 arm 约 20 分钟。费用：(1+2+2) H100 × 0.35 h × $3.95 ≈ $7。每个 arm 硬超时 60 分钟（`timeout 3600`），上限 5 GPU·h ≈ $20。
- 回收：launcher 结束时自行拆除；`timeout 3600` 包裹；独立 watchdog（setsid），3900 s 后执行 `modal app stop -y yeto-<prefix>`；结束后用 `modal app list --json` 核实 stopped/0 tasks 并存档；停止 puller 与 watchdog。
- 三个 arm 串行执行（避免本机线程压力），同一 provider、同一型号断言。

## 追加（事后，判定条件未改）：测量手段修正
A-attempt2（a37403a，1×H100!，SUCCEEDED）的 island 磁带靠轮询拉取，只拉到 2/3 个 `rl_round_trained`：最后一轮写出后容器很快被拆除。这会让条件 2（A/B 轮数与逐轮 sample-id 比较）因为采集缺失而误判。修正：driver 在快照根目录存在 `yeto-rl-echo-events` 时，把每条事件同时打印到 stdout（前缀 `YETO_RL_EVENT`），由 head 的 launch.log 完整收集；该功能默认关闭。修正提交之后，A、B、C 都用同一个新 SHA 重跑，A-attempt2 只作为记录保留，不参与判定。

## 追加（事后，判定条件未改）：第二轮（d2018b5）结果与第三轮
- round2-e0a（d2018b5，1×H100!）SUCCEEDED，事件 echo 完整（见 `round2-e0a/tape.jsonl`）。
- round2-e0b / round2-e0c 在 deploy 之前失败，没有占用 GPU。原因：`launch` 没有 `--rollout-num-gpus` 选项，argparse 把它当作 `--rollout-num-gpus-per-engine` 的缩写接受了，结果 rollout GPU 数为空，被 selection 拒绝。修复：launch 新增 `--rl-rollout-gpus`（8cf1dec）。
- 修复提交中还包含与本实验无关的代码变更（receipt 标签、1a/2a 通道）。为保证 A/B/C 使用同一份代码，第三轮三个 arm 全部使用同一个新 SHA 重跑；A 的前两次尝试（a37403a、d2018b5）作为记录保留，如与第三轮 A 的可比数据（逐轮 sample-id 哈希、轮数、applied_lrs 长度）有差异，在完成记录中如实报告。
