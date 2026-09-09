# Corrected Qwen3.8 training quality checks

This is a fixed **16-task synthetic action monitor**, separate from Terminal Bench and excluded from training. It tests whether the checkpoint executes real file operations and satisfies exact verifiers. It cannot establish broad benchmark quality from 16 simple tasks. No model evaluation has been run by this package yet.

Evaluate both corrected no-CoT and masked-CoT runs after **25, 50, and 100 completed optimizer updates**, then every **100 updates**. Those are zero-based checkpoint steps **24, 49, 99, 199, …**. These are measurement points, not a prediction that improvement will appear by any particular step. `protocol.checkpoints_through` and `next_checkpoint` expose the schedule for the checkpoint dispatcher.

Use the same frozen tasks and harness for both arms: Codex 0.142.5, native Qwen3.8 rendering (`qwen3_6` TiTo/token builder), xhigh, context 262144, auto-compaction 196608, output cap 32768, temperature 1.0 and top-p 1.0. Preserve the qwen3_5 tool/mask parser family required by the existing runtime. Use **one attempt per task**, no best-of selection and no automatic retry of model failures. Preserve all outcomes. A separate infrastructure repair must preserve the failed original and must never silently choose a better model response.

Report cumulative supervised training tokens with every checkpoint, plus input tokens and GPU hours when available. Equal optimizer updates do not imply equal training-token budget. Compare matching checkpoints descriptively and use token budgets when interpreting the two arms. The baseline and masked datasets also differ, so this does not isolate CoT as the sole cause of any difference. Heldout training loss is a complementary metric and must carry its actual example count and identity; loss alone is not tool-use quality.

## Artifacts and stages

- `suite.py` freezes the original tasks, instructions, input files and exact verifiers. Expected answers are present only in verifier files; the task image copies only input files. It pins the already validated task-base image by digest and clears its inherited entrypoint. Full source and byte hashes are recorded.
- `prepare.py` validates the frozen suite and prepares a fresh checkpoint-specific Harbor job, model catalog, authenticated gateway integration and four-TP2 native serving configuration. It preserves the already validated runtime configuration without touching old evaluation plans or training runs.
- `checkpoint.py` preserves the scheduled model shards with hard links before trainer retention removes their original paths. `export.py` uses the pinned CPU converter and binds its receipt to the exact checkpoint and cumulative training-token budget.
- `supervisor.py` watches one exact trainer container/configuration and preserves the finite first three checkpoints (25, 50 and 100 updates). It queues one isolated CPU export at a time, with no GPU or network access, and publishes immutable `ready-updates-N.json` handoffs. It never launches evaluation or follows a restarted trainer automatically. Later 100-update monitoring requires another explicit finite plan; the first watcher does not silently extend itself.
- `collect.py` automatically finds every original task result in the finite run, binds its saved bundle/trajectory, compares actual initial input IDs with native rendering, and preserves the shared checkpoint-level supplied-tool-history proof. It writes the outcome inventory and invokes aggregation after Harbor exits.
- `aggregate.py` checks the exact run and artifact hashes, accepts a finite inventory of first-attempt trial results, and reads original Harbor results, ATIF trajectories and saved proxy bundles. Its output contains only identities, hashes and metrics, not transcript text.
- `compare.py` reports paired gains/losses and budget differences from two complete summaries. It does not select a best checkpoint or claim statistical significance.

**Checkpoint dispatch is explicit.** The new `run.py` accepts the complete checkpoint-specific export receipt SHA, verifies every weight/asset file plus arm/update/token-budget identity, parses the installed Harbor job, and checks the node is idle. It starts only its own services, runs actual native initial and incremental supplied-tool-history parity, then starts the 16-task Harbor job. Weak model behavior does not determine admission: the qualification supplies a fixed synthetic tool history instead of requiring the model to act correctly. The historical step299 runners are not used. Prepared artifacts remain `dispatch_ready:false` until the actual export/parity gates pass. No jobs have been launched during implementation. The parent dispatcher owns checkpoint retention/export and invocation; no background schedule is silently created.

## Prepare

Freeze once; subsequent runs reuse those exact bytes:

```bash
python -m evaluation.qwen38_training_quality.suite --output evaluation/qwen38_training_quality/frozen-v1
```

Create `checkpoint.json` using `protocol.checkpoint_identity`, with the actual completed checkpoint artifact-manifest SHA256, arm (`no-cot-turn-v2` or `masked-cot-native-gap-v3`), completed updates, and cumulative supervised tokens. Do not put an old checkpoint identity into a new run. Then:

```bash
python -m evaluation.qwen38_training_quality.prepare \
  --suite-dir evaluation/qwen38_training_quality/frozen-v1 \
  --output /private/tmp/quality-no-cot-u25 \
  --remote-root /data/evals/quality-no-cot-u25 \
  --checkpoint-json /private/tmp/checkpoint.json \
  --model-catalog /path/to/preserved/qwen38-codex-models.json
```

The output directory must not exist. The catalog is copied and only its model slug/display name change. Task instances and all job/artifact paths are fresh. Authentication remains in the established process-local environment, with no key stored in the artifacts.

After the checkpoint exporter and node reservation have passed, the parent dispatcher can invoke the new runner inside the established evaluation container (with this repository on PYTHONPATH):

```bash
python -m evaluation.qwen38_training_quality.run \
  --root /data/evals/quality-no-cot-u25 \
  --export-receipt-sha256 ACTUAL_COMPLETE_EXPORT_RECEIPT_SHA256
```

This is a real GPU/model operation and is not part of the CPU tests. The runner generates a temporary process-local proxy key, never writes it to receipts, and cleans up only service processes it created. Existing training/inference processes cause a refusal to start. Checkpoint-specific synthetic qualification token records stay in the run's `launch/` directory and never contribute to task scores.

## Aggregate actual outcomes

The dispatcher produces an inventory with schema `yeta.qwen38-training-quality-outcomes/v1`, the exact `run_sha256`, `first_attempts_only:true`, `best_of_selection:false`, and an `outcomes` list. Each entry has a frozen `task_id` and a `result` reference `{ "path": "…", "sha256": "…" }`; optional `trajectory`, `bundle`, and `native_parity` use the same reference format. Results must be finished, unique and match the actual model/agent/task settings. No result is invented for a missing task.

A task's native-parity receipt uses schema `yeta.qwen38-quality-native-parity/v1`, exact checkpoint-manifest/result/bundle/template hashes, and actual parity booleans. These receipts must come from real saved-input comparisons and the checkpoint-specific serving qualification, not be inferred from a config or fabricated by the aggregator.

```bash
python -m evaluation.qwen38_training_quality.aggregate \
  --run-dir /path/to/prepared-run \
  --inventory /path/to/outcomes.json \
  --output /path/to/quality-summary.json
```

Reported measures:

- Exact verifier passes divided by the fixed 16-task denominator, with missing/infrastructure counts visible. A secondary rate over executed model outcomes is labeled separately.
- Tool calls, observations, and observations linked to actual call IDs.
- No-tool responses and the narrower premature-stop symptom: exactly one recorded LLM call, no tools, normal stop, and an EOT in actual saved generated IDs.
- Output-limit terminations, recorded parser-error signatures and explicit infrastructure exceptions, kept separate.
- Coverage of original trajectories, saved bundles, and genuine native-parity receipts. Missing evidence stays unavailable rather than becoming a measured zero.
- Checkpoint identity and cumulative supervised/input tokens and GPU hours.

A low no-tool rate does not by itself prove successful actions. Pass rate remains the primary action-quality measurement. A complete set with infrastructure or missing parity evidence is reported but not presented as a fully qualified model comparison.

## CPU validation and limits

Run `python -m pytest -q evaluation/qwen38_training_quality`. Tests execute all 16 actual verifiers against correct outputs and missing outputs, check immutable task/config identities and checkpoint scheduling, and exercise the aggregator with synthetic saved-artifact fixtures including early EOT, tool use, output caps, parser warnings, infrastructure failure and missing outcomes. These tests do not substitute for the actual installed Harbor/Docker/image preflight or a real checkpoint evaluation.
