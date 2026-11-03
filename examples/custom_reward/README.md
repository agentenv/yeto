# Custom reward example

`reward.py` is a minimal neutral reward: it takes a neutral trajectory
(`yeto.rl.rewards.types.Trajectory`), returns a reward result (`RewardResult`)
and imports no training framework.

- By name: after `import examples.custom_reward.reward`, the name
  `ends_with_label` is registered.
- By reference: `examples.custom_reward.reward:ends_with_label`.
- Value for Miles' `--custom-rm-path`:
  `yeto.rl.adapters.miles.rewards.miles_custom_rm_path(<name or reference>)`; an
  unregistered name is an error that lists the registered names.
- Loading records the plugin identity (module, function name, sha256 of the
  source file; see `yeto.rl.rewards.registry.load_reward`), so editing the source
  changes the identity.

Test: `python -m pytest examples/custom_reward/test_reward.py`.
