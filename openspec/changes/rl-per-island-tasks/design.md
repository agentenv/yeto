# Design

## Flag format

`--rl-island-tasks` takes a string or a file path:

- String: `l0=web1-10,pwn1-10;l1=web11-20,pwn11-20`. The island key is `l{id}` or `{id}`, the same key rule as `launcher.py:2963 learner_image_for`.
- A range `web1-10` expands to `web1` through `web10`. The expansion rule accepts only "prefix + start and end integers".
- File: a JSON object `{"0": ["web1", ...], "1": [...]}`.

The parse function lives on the launcher side, not in `island_overrides.py`. Reason: `island_overrides.py` requires `--rl-negative-test-run` and is used only for negative-test runs (module docstring). It is not suitable for a production flag.

## Data flow (Q1 recommendation A)

```
cli --rl-island-tasks
  -> launcher parses, expands, validates
  -> per island: write_prompt_data(catalog, out, task_ids)  [local machine]
  -> per island --data <mount path>; the run manifest records task_ids and sha256
  -> inside the island, island_entry.prepare_prompt_data reads the local JSONL as usual
```

Without `--rl-island-tasks`, behavior does not change.

## Validation

1. Island numbers must be in `0..N-1`, and every island must have a list (a partly missing island is not allowed).
2. Every task id must be in the catalog.
3. strict-avg: the task counts of the islands must be equal (Q2 recommendation).
4. dense-full or SAO: reject (Q5 recommendation B).

## Contract hash

- LoRA ports: `session_contract_hash` has no data, so no change is needed (verified by reading code).
- dense-full: `_dense_full_training_contract_sha256` contains `prompt_data_sha256` and `data`, and enters the payload at `local_learner.py:276`. Whether the syncer compares it across islands: unverified. Q5 recommends rejecting first.

## Risks

- When task difficulty is unequal, elastic weighting leans toward the easier islands. This change only records it and does not correct it.
- Resume: a restarted island must get the same prompt file. The run manifest records the sha256, and the island compares it at start.
