# RL engine equivalence report (legacy vs ports, layered)

- mode: real
- fake: False
- preset: decoupled
- config: --model Qwen/Qwen3-0.6B --model-revision c1899de289a04d12100db370d81485cdf75e47ca --data /work/data/gsm8k.jsonl --reward-function gsm8k_reward:score --islands 2 --gpus-per-island 1 --groups-per-island 4 --samples-per-group 8 --rollout-max-response-len 384 --seq-len 1024 --lora-r 16 --lora-targets all-linear --eval-prompts 8 --eval-samples-per-prompt 1 --pass-k 1 --eval-device cuda --miles-root /opt/miles-next --apply-chat-template-kwargs '{"enable_thinking": false}' --trust-remote-code --arms decoupled --global-rounds 8 --fragments 8 --pipeline 2 --local-horizon 4 --arm-timeout-min 60 (legacy: --miles-root /opt/miles-legacy)
- primary_seed: 17
- legacy_seeds: [17, 18, 19, 20, 21]
- ports_seeds: [17, 18, 19, 20, 21]
- created_unix: 1790666717.0347345

## Thresholds (fixed in scripts/rl_engine_equivalence.py before the experiment)

- round1_exact: ['completed_groups', 'action_tokens', 'reward_mean']
- round1_grad_norm_rel: 0.03
- tf_loss: |d| <= max(1e-06, 0.001*|loss_legacy|)
- tf_grad_norm_rel: 0.03
- tf_update_rel_l2: 0.05
- tf_initial_lora_rel_l2: 1e-06
- distribution: per seed: mean over (island, round >= 2) of ['reward_mean', 'loss', 'grad_norm', 'action_tokens']; exact two-sided permutation test legacy vs ports (mean diff), FAIL if p < 0.05/4
- hash: within path: all islands equal per policy version; never cross-path

## Result: PASS

- tier 1_round1: PASS (reported, not gated)
- tier 2_teacher_forcing: PASS
- tier 3_distribution: PASS
- tier 4_hash: PASS
- tier 5_peft: PASS

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
| 0 | LoRA update (audit f32) | False | False | None | not available (see strict-avg TF) | info |
| 1 | fidelity legacy-TF completed_groups == reference | 4 | 4 | None | exact | ok |
| 1 | fidelity ports-TF completed_groups == reference | 4 | 4 | None | exact | ok |
| 1 | fidelity legacy-TF action_tokens == reference | 6145 | 6145 | None | exact | ok |
| 1 | fidelity ports-TF action_tokens == reference | 6145 | 6145 | None | exact | ok |
| 1 | fidelity legacy-TF reward_mean == reference | 0.4375 | 0.4375 | None | exact | ok |
| 1 | fidelity ports-TF reward_mean == reference | 0.4375 | 0.4375 | None | exact | ok |
| 1 | legacy-TF grad_norm vs reference (info) | 0.397639 | 0.397639 | 0 | reported | info |
| 1 | loss | -1.86265e-09 | -1.86265e-09 | 0 | |d| <= max(1e-06, 0.001*|loss_l|) | ok |
| 1 | grad_norm | 0.397639 | 0.40184 | 0.0105653 | rel <= 0.03 | ok |
| 1 | LoRA update (audit f32) | False | False | None | not available (see strict-avg TF) | info |

## Tier 3: distribution over seeds (rounds >= 2, per-seed means, permutation test)

- seeds: legacy 5, ports 5; exact enumeration of 252 splits; alpha 0.05 Bonferroni -> 0.0125 per metric
- power limitation: smallest attainable p = 0.007937; with 5 vs 5 the test rejects only when the two groups are completely separated; a Gaussian mean shift of about 3.2 pooled SD is needed for 80% power, so smaller real differences usually PASS

| metric | legacy per seed | ports per seed | legacy mean | ports mean | effect size | p | pass |
|---|---|---|---|---|---|---|---|
| reward_mean | 17:0.553571, 18:0.604911, 19:0.569196, 20:0.580357, 21:0.578125 | 17:0.600446, 18:0.604911, 19:0.544643, 20:0.546875, 21:0.600446 | 0.577232 | 0.579464 | 0.0875209 | 0.97619 | ok |
| loss | 17:7.98276e-10, 18:-2.32831e-09, 19:-2.66092e-10, 20:2.66092e-09, 21:-2.19526e-09 | 17:1.86265e-09, 18:5.32184e-10, 19:-1.79612e-09, 20:5.92055e-09, 21:-3.32615e-10 | -2.66092e-10 | 1.23733e-09 | 0.588773 | 0.412698 | ok |
| grad_norm | 17:0.290349, 18:0.300768, 19:0.332879, 20:0.320776, 21:0.452384 | 17:0.298094, 18:0.274956, 19:0.290145, 20:0.344078, 21:0.394433 | 0.339431 | 0.320341 | -0.331231 | 0.611111 | ok |
| action_tokens | 17:7882.36, 18:7670.79, 19:7232.64, 20:7536.29, 21:6483.21 | 17:7720.36, 18:7735.93, 19:7220.57, 20:6875.93, 21:6845.14 | 7361.06 | 7279.59 | -0.165308 | 0.809524 | ok |

## Tier 4: within-path hash consistency

| path | seed | policy version | islands agree |
|---|---|---|---|
| legacy | 17 | 0 | ok |
| legacy | 17 | 8 | ok |
| legacy | 18 | 0 | ok |
| legacy | 18 | 8 | ok |
| legacy | 19 | 0 | ok |
| legacy | 19 | 8 | ok |
| legacy | 20 | 0 | ok |
| legacy | 20 | 8 | ok |
| legacy | 21 | 0 | ok |
| legacy | 21 | 8 | ok |
| ports | 17 | 0 | ok |
| ports | 17 | 8 | ok |
| ports | 18 | 0 | ok |
| ports | 18 | 8 | ok |
| ports | 19 | 0 | ok |
| ports | 19 | 8 | ok |
| ports | 20 | 0 | ok |
| ports | 20 | 8 | ok |
| ports | 21 | 0 | ok |
| ports | 21 | 8 | ok |

## Decoupled: PEFT load and final LoRA distance

- legacy-s17/work/seed-17/yeto-decoupled-m2/adapter: ok
- legacy-s18/work/seed-18/yeto-decoupled-m2/adapter: ok
- legacy-s19/work/seed-19/yeto-decoupled-m2/adapter: ok
- legacy-s20/work/seed-20/yeto-decoupled-m2/adapter: ok
- legacy-s21/work/seed-21/yeto-decoupled-m2/adapter: ok
- ports-s17/work/seed-17/yeto-decoupled-m2/adapter: ok
- ports-s18/work/seed-18/yeto-decoupled-m2/adapter: ok
- ports-s19/work/seed-19/yeto-decoupled-m2/adapter: ok
- ports-s20/work/seed-20/yeto-decoupled-m2/adapter: ok
- ports-s21/work/seed-21/yeto-decoupled-m2/adapter: ok
- final_lora_rel_l2_legacy_vs_ports: 0.00828608998540591
- final_lora_rel_l2_legacy_seed_to_seed: {'17->18': 1.4149817493402717, '17->19': 1.4144959885815358, '17->20': 1.4146158882749975, '17->21': 1.4140258766986409}
- final LoRA distance is reported without a hard threshold: sampling diverges from round 2, so it measures trajectory divergence, not trainer error; compare it with the legacy seed-to-seed distance.
