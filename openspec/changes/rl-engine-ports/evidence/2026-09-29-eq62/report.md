# RL engine equivalence report (legacy vs ports, layered)

- mode: real
- fake: False
- preset: strict-avg
- config: --model Qwen/Qwen3-0.6B --model-revision c1899de289a04d12100db370d81485cdf75e47ca --data /work/data/gsm8k.jsonl --reward-function gsm8k_reward:score --islands 2 --gpus-per-island 1 --groups-per-island 4 --samples-per-group 8 --rollout-max-response-len 384 --seq-len 1024 --lora-r 16 --lora-targets all-linear --eval-prompts 8 --eval-samples-per-prompt 1 --pass-k 1 --eval-device cuda --miles-root /opt/miles-next --apply-chat-template-kwargs '{"enable_thinking": false}' --trust-remote-code --arms federated --global-rounds 3 --arm-timeout-min 45 (legacy: --miles-root /opt/miles-legacy)
- primary_seed: 17
- legacy_seeds: [17, 18, 19, 20, 21]
- ports_seeds: [17, 18, 19, 20, 21]
- created_unix: 1790666694.748265

## Thresholds (fixed in scripts/rl_engine_equivalence.py before the experiment)

- round1_exact: ['completed_groups', 'action_tokens', 'reward_mean']
- round1_grad_norm_rel: 0.03
- tf_loss: |d| <= max(1e-06, 0.001*|loss_legacy|)
- tf_grad_norm_rel: 0.03
- tf_update_rel_l2: 0.05
- tf_initial_lora_rel_l2: 1e-06
- distribution: per seed: mean over (island, round >= 2) of ['reward_mean', 'loss', 'grad_norm', 'action_tokens']; exact two-sided permutation test legacy vs ports (mean diff), FAIL if p < 0.05/4
- hash: within path: all islands equal per policy version; never cross-path

## Result: FAIL

- tier 1_round1: PASS
- tier 2_teacher_forcing: FAIL
- tier 3_distribution: PASS
- tier 4_hash: PASS

## Tier 1: round 1 (primary seed)

| island | metric | legacy | ports | rel diff | rule | pass |
|---|---|---|---|---|---|---|
| 0 | completed_groups | 4 | 4 | - | exact | ok |
| 0 | action_tokens | 5227 | 5227 | - | exact | ok |
| 0 | reward_mean | 0.8125 | 0.8125 | - | exact | ok |
| 0 | grad_norm | 0.638508 | 0.634947 | 0.0055772 | rel <= 0.03 | ok |
| 1 | completed_groups | 4 | 4 | - | exact | ok |
| 1 | action_tokens | 6145 | 6145 | - | exact | ok |
| 1 | reward_mean | 0.4375 | 0.4375 | - | exact | ok |
| 1 | grad_norm | 0.397639 | 0.40184 | 0.0105653 | rel <= 0.03 | ok |

## Tier 2: teacher forcing (legacy round-1 rollouts replayed on both engines)

| island | check | legacy | ports | value | rule | pass |
|---|---|---|---|---|---|---|
| 0 | fidelity legacy-TF completed_groups == reference | 4 | 4 | None | exact | ok |
| 0 | fidelity ports-TF completed_groups == reference | 4 | 4 | None | exact | ok |
| 0 | fidelity legacy-TF action_tokens == reference | 5227 | 5227 | None | exact | ok |
| 0 | fidelity ports-TF action_tokens == reference | 5227 | 5227 | None | exact | ok |
| 0 | fidelity legacy-TF reward_mean == reference | 0.8125 | 0.8125 | None | exact | ok |
| 0 | fidelity ports-TF reward_mean == reference | 0.8125 | 0.8125 | None | exact | ok |
| 0 | legacy-TF grad_norm vs reference (info) | 0.638508 | 0.638508 | 0 | reported | info |
| 0 | loss | 1.21072e-08 | 1.21072e-08 | 0 | |d| <= max(1e-06, 0.001*|loss_l|) | ok |
| 0 | grad_norm | 0.638508 | 0.634947 | 0.0055772 | rel <= 0.03 | ok |
| 0 | layout_hash equal | f1c29481efeffd57ff1451a1e510ab088b2fdb082e01341d6313140ccf03b8b3 | f1c29481efeffd57ff1451a1e510ab088b2fdb082e01341d6313140ccf03b8b3 | None | exact | ok |
| 0 | initial LoRA rel L2 | None | None | 0 | <= 1e-06 | ok |
| 0 | LoRA update rel L2 | None | None | 0.333494 (cos 0.944391) | <= 0.05 | FAIL |
| 1 | fidelity legacy-TF completed_groups == reference | 4 | 4 | None | exact | ok |
| 1 | fidelity ports-TF completed_groups == reference | 4 | 4 | None | exact | ok |
| 1 | fidelity legacy-TF action_tokens == reference | 6145 | 6145 | None | exact | ok |
| 1 | fidelity ports-TF action_tokens == reference | 6145 | 6145 | None | exact | ok |
| 1 | fidelity legacy-TF reward_mean == reference | 0.4375 | 0.4375 | None | exact | ok |
| 1 | fidelity ports-TF reward_mean == reference | 0.4375 | 0.4375 | None | exact | ok |
| 1 | legacy-TF grad_norm vs reference (info) | 0.397639 | 0.397639 | 0 | reported | info |
| 1 | loss | -1.86265e-09 | 5.58794e-09 | 7.45058e-09 | |d| <= max(1e-06, 0.001*|loss_l|) | ok |
| 1 | grad_norm | 0.397639 | 0.40184 | 0.0105653 | rel <= 0.03 | ok |
| 1 | layout_hash equal | f1c29481efeffd57ff1451a1e510ab088b2fdb082e01341d6313140ccf03b8b3 | f1c29481efeffd57ff1451a1e510ab088b2fdb082e01341d6313140ccf03b8b3 | None | exact | ok |
| 1 | initial LoRA rel L2 | None | None | 0 | <= 1e-06 | ok |
| 1 | LoRA update rel L2 | None | None | 0.424396 (cos 0.909945) | <= 0.05 | FAIL |

## Tier 3: distribution over seeds (rounds >= 2, per-seed means, permutation test)

- seeds: legacy 5, ports 5; exact enumeration of 252 splits; alpha 0.05 Bonferroni -> 0.0125 per metric
- power limitation: smallest attainable p = 0.007937; with 5 vs 5 the test rejects only when the two groups are completely separated; a Gaussian mean shift of about 3.2 pooled SD is needed for 80% power, so smaller real differences usually PASS

| metric | legacy per seed | ports per seed | legacy mean | ports mean | effect size | p | pass |
|---|---|---|---|---|---|---|---|
| reward_mean | 17:0.382812, 18:0.382812, 19:0.421875, 20:0.382812, 21:0.382812 | 17:0.390625, 18:0.460938, 19:0.375, 20:0.40625, 21:0.40625 | 0.390625 | 0.407813 | 0.66033 | 0.420635 | ok |
| loss | 17:1.62981e-09, 18:-3.25963e-09, 19:4.65661e-09, 20:-2.32831e-09, 21:5.3551e-09 | 17:-4.65661e-10, 18:-4.65661e-10, 19:5.12227e-09, 20:-6.98492e-09, 21:-1.07102e-08 | 1.21072e-09 | -2.70084e-09 | -0.753773 | 0.277778 | ok |
| grad_norm | 17:0.216965, 18:0.269034, 19:0.316535, 20:0.355121, 21:0.310138 | 17:0.269324, 18:0.346344, 19:0.306897, 20:0.276288, 21:0.415322 | 0.293559 | 0.322835 | 0.519049 | 0.452381 | ok |
| action_tokens | 17:8872.75, 18:8912.25, 19:8710, 20:8531.75, 21:8697.25 | 17:8944.75, 18:8767, 19:8806.75, 20:8654.5, 21:8524 | 8744.8 | 8739.4 | -0.034646 | 0.960317 | ok |

## Tier 4: within-path hash consistency

| path | seed | policy version | islands agree |
|---|---|---|---|
| legacy | 17 | 0 | ok |
| legacy | 17 | 1 | ok |
| legacy | 17 | 2 | ok |
| legacy | 17 | 3 | ok |
| legacy | 18 | 0 | ok |
| legacy | 18 | 1 | ok |
| legacy | 18 | 2 | ok |
| legacy | 18 | 3 | ok |
| legacy | 19 | 0 | ok |
| legacy | 19 | 1 | ok |
| legacy | 19 | 2 | ok |
| legacy | 19 | 3 | ok |
| legacy | 20 | 0 | ok |
| legacy | 20 | 1 | ok |
| legacy | 20 | 2 | ok |
| legacy | 20 | 3 | ok |
| legacy | 21 | 0 | ok |
| legacy | 21 | 1 | ok |
| legacy | 21 | 2 | ok |
| legacy | 21 | 3 | ok |
| ports | 17 | 0 | ok |
| ports | 17 | 1 | ok |
| ports | 17 | 2 | ok |
| ports | 17 | 3 | ok |
| ports | 18 | 0 | ok |
| ports | 18 | 1 | ok |
| ports | 18 | 2 | ok |
| ports | 18 | 3 | ok |
| ports | 19 | 0 | ok |
| ports | 19 | 1 | ok |
| ports | 19 | 2 | ok |
| ports | 19 | 3 | ok |
| ports | 20 | 0 | ok |
| ports | 20 | 1 | ok |
| ports | 20 | 2 | ok |
| ports | 20 | 3 | ok |
| ports | 21 | 0 | ok |
| ports | 21 | 1 | ok |
| ports | 21 | 2 | ok |
| ports | 21 | 3 | ok |
