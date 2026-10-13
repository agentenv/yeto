# Proposal

## Why

PM item 5 asks islands to run different tasks: island 1 does web1-10 plus pwn1-10, and island 2 does web11-20 plus pwn11-20. Today all islands receive the same `--data` (`make_miles_island_task` in `launcher.py` uses `learner_data_arg(args.data)`). The only difference between islands is the rollout seed (`run_config.py:941`, `args.seed + args.learner_id`). A seed changes only the sampling order and does not split tasks. This change is not implemented this week. This proposal is for the user to decide.

## What Changes

- `yeto/cli.py` gets a new `--rl-island-tasks`, which gives a task id list per island (for the format, see design.md).
- When the launcher builds the command of each island, it takes the task list of that island by learner_id, generates the prompt file of that island through `rlf/catalog.py:94 write_prompt_data(task_ids=...)`, and uses it as the `--data` of that island.
- Validation at launch: island number out of range, a task id not in the catalog; unequal task counts give a warning and are not rejected (Q2 decided).
- The `rl_driver_start` event of each island and the run manifest record the task list of that island and the sha256 of the prompt file.
- The training contract hash of the dense-full and SAO paths contains the prompt data hash (see "Obstacles" below). This change defines the "shared part" and the "island-local part", so that per-island tasks do not trigger a cross-island rejection.

## Current state and obstacles from reading code (2026-10-13)

1. **Half exists already**: `rlf/catalog.py:69 prompt_rows` and `:94 write_prompt_data` already support filtering by `task_ids`.
2. **Overturned**: the P3 draft says `eval/rlf_rl_gate.py:132-136 prepare_prompt_data` "does not pass task_ids". The actual file is `yeto/rl/eval/rlf_rl_gate.py`. It builds the task package directory from `config.task_ids` and then calls `import_catalog`. So it already filters by task. It just does not use the `task_ids` parameter of `write_prompt_data`.
3. **Overturned**: the P3 draft says "connect `prepare_prompt_data` where each island command is built, at `launcher.py:3985-4005`". At that place (`make_miles_island_task`, starting around line 3949), the code only forwards `--data`. The file that generates the prompt file is `island_entry.py:1818 prepare_prompt_data(source, revision, ...)` inside the island. It reads an HF or local dataset and does not know the RLF task directory. So there are two candidate integration points (see pending Q1).
4. **New obstacle (not in the P3 draft)**: the dense-full training contract hash `_dense_full_training_contract_sha256` (from `island_entry.py:827`) contains `prompt_data_sha256` and `data`. That hash enters the profile payload at `local_learner.py:276`. By reading code, this payload is judged to be compared across islands, so per-island tasks would be rejected under dense-full. The cross-island comparison point was not traced line by line into the syncer, so it is marked "unverified". The SAO path `sao_streaming_runtime.py:203` also has a `training_contract_hash`. Whether it contains the data hash is unverified.
5. The island contract hash `session_contract_hash` (`backend_identity.py:82-87`) contains only the layout fingerprint and the backend identity and no data. The LoRA ports path is not affected by item 4.

## Dependencies (P0 `rl-code-quality-cleanup`)

Must be done before this change starts:
- **1.7b.4 compatibility policy for backend_identity and the dense session contract hash**: per section 3 of `infra-drafts/SESSION23/COORDINATION.md`, the island contract hash, `backend_identity`, and the dense session contract hash are decided together by P0 1.7b.4. This change does not change these hashes by itself. If Q5 chooses A, 1.7b.4 must handle it together.
- **1.10 remove legacy**: SAO, dense-full, and DSV4 now go through the legacy `create_policy_sync` (tasks 1.7b). The position of the contract hash changes before and after removing legacy. Remove first and change after, to avoid changing twice.
- **4.7 RL flag schema**: the new flag must pass through three layers: cli, launcher, island_entry. After 4.7 is done, only one place in `run_config.py` needs an addition. If the user asks to do this change first, the flag is written by hand in the existing three layers and migrated together at 4.7 (see Q4).
- **4.8 embed the algorithm hash in the island contract hash**: this change must define "which fields may differ between islands". That is the same group of hashes that 4.8 changes, so this change must come after 4.8.

Not dependent: phase 3 (resume abstraction). But on resume, the same prompt file of that island must be rebuilt. Writing the task list into the run manifest satisfies this.

## Pending decisions

**Decided (user, 2026-10-13): Q1 chooses A (the launcher generates the task list file of each island on the local machine); Q2 chooses B (unequal task counts are allowed, a warning is given at launch, and strict and elastic are handled the same way); Q3 eval uses the full set and runs only on a separate eval island (`yeto/cloud/modal_eval_island.py`), and training islands do not load the eval set; Q4 chooses A (wait for P0 4.7); Q5 opens up the data comparison.**

**Q1 On which side the task list becomes a prompt file**
- Option A: the launcher generates the prompt file of each island on the local machine and passes it to the island as a mounted file.
- Option B: the launcher only forwards the task id list, and the island generates the file with `write_prompt_data`.
- Recommend A. Reason: errors show up before cards are opened; the sha256 of the prompt file can be written into the run manifest on the local machine; the island does not need the RLF task directory. The cost is one more file mount per island.

**Q2 What to do when task counts differ between islands under strict-avg**
- Option A: refuse to start.
- Option B: allow it and only give a warning.
- Recommend A. Reason: strict-avg has a barrier for all members each round, so `rollout_batch_size` and total steps of each island must be equal. When task counts differ, the smaller island samples repeatedly, which changes the weights in effect. Under elastic, unequal counts are allowed, but a warning explains that task difficulty affects the delta weights.

**Q3 Whether eval uses the full set or the subset of the island**
- Option A: full set. Option B: subset of the island.
- Recommend A. Reason: the eval numbers of the islands can be compared directly; the merged model must be judged on the full set.

**Q4 Whether to wait for the P0 4.7 flag schema**
- Option A: wait for 4.7. Option B: write the three layers by hand first and migrate at 4.7.
- Recommend A. Reason: this change is not implemented this week, so waiting has no extra cost; writing by hand first adds one more flag for 4.7 to migrate.

**Q5 How the dense-full contract hash handles the data fields**

**Decided (user, 2026-10-13): open it up directly. Different data and tasks per island is the necessary result of per-island training. The data fields move out of the contract compared between islands (no more rejection), and per-island tasks are also allowed under dense-full and SAO. The concrete changes are merged into P0 1.7b.4 and done together.**
- Option A: move `prompt_data_sha256` and `data` to the island-local part, and record "the sha256 of the full task set" in the shared part.
- Option B: per-island tasks are supported only on the LoRA ports path and rejected under dense-full.
- Recommend B first. Reason: changing the contract hash changes the cross-island rejection semantics and needs a review on GPU; B delivers the PM scenario without running on GPU. A involves the dense contract hash, which belongs to P0 1.7b.4 per section 3 of COORDINATION.md. This change does not do it.

## Capabilities

### New Capabilities
- `rl-per-island-tasks`: Specify the training task subset per island, including validation at launch, recording, and contract hash rules.

### Modified Capabilities
(None.)

## Impact

- Code: `yeto/cli.py`, `yeto/launcher.py` (near `make_miles_island_task`), `yeto/rl/rlf/catalog.py` (called only), `yeto/rl/engine/run_config.py` (after 4.7), `yeto/rl/adapters/miles/island_entry.py` (when Q5 chooses A).
- Tests: new local tests; parsing, validation, and prompt file generation can be covered without running on GPU.
- Run on GPU: acceptance needs one run with the minimum card count, see section 4 of tasks.md. This proposal does not open cards.
