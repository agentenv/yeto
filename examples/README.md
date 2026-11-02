# Examples

## `math_reward.py` — an RL smoke test on MATH-500

The built-in RL environments (SecRLEnv / CyberGym) need an external package
and an API key. This example needs neither: it grades the `\boxed{...}`
answer of each rollout against the MATH-500 `answer` column with Miles' own
math graders, so it is the quickest way to check that an RL island trains at
all (rollouts happen, rewards are non-zero, `train/grad_norm` is non-zero,
the syncer completes its rounds).

It is a smoke test, not a benchmark: MATH-500 is a *test* set and the reward
is exact-match.

```bash
yeto launch --training-mode rl \
  --gpu modal:8xh100 --syncer-region nebius/eu-north1 \
  --cluster-prefix math-smoke \
  --model Qwen/Qwen3-1.7B --model-revision 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --data HuggingFaceH4/MATH-500 --data-revision 6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be \
  --rl-prompt-column problem --rl-label-column answer \
  --reward-function examples/math_reward.py:reward_func \
  --apply-chat-template-kwargs '{"enable_thinking": false}' \
  --tuning lora --lora-r 8 --lora-targets attention --total-steps 3 \
  --rollout-batch-size 16 --n-samples-per-prompt 4 --rollout-max-response-len 1024 \
  --seq-len 2048 --inner-lr 1e-5 --seed 17 --trust-remote-code --on-demand
```

- Without `--rl-image` the island runs the pinned default ports image
  (`MILES_NEXT_IMAGE` in `yeto/rl/adapters/miles/pins.py`).
- `--rl-prompt-column` / `--rl-label-column` name the dataset columns; the RL
  data path otherwise expects `messages` / `prompt` / `input` and `label`,
  and never guesses. A named column that is missing is an error.
- `--reward-function` takes any `file.py:func` inside the repo; the file is
  synced to the island and its sha256 is attested like any other reward.
- `--gpu` can be any cloud entry; the syncer can run in any supported region.
- Any Qwen3 checkpoint works; `enable_thinking: false` keeps rollouts short
  enough for `--rollout-max-response-len 1024`.

## Other examples

- [`custom_reward/`](custom_reward/README.md): a minimal framework-neutral
  reward plugin.
- `rl_algorithms/`: `AlgorithmSpec` JSON files for `--rl-algorithm-spec`.
