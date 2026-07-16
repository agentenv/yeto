# Best-paper v2 E1/E4 multi-seed executor changes

## Outcome and scope

The local executor now materializes and validates paired-seed E1/E4 campaigns with seed and learner count as first-class cell dimensions. A complete `(H, seed, M)` seed block is the atomic wave and contains the three explicit mixed-eta arms required by the experiment program. The parallel controller deterministically randomizes block order, arm-to-slot assignment, and launch order; enforces one machine/GPU-packing profile within each block; retries only whole blocks after a registered infrastructure failure; and retains scientific divergence as the frozen capped-loss outcome.

This work performed no cloud mutation, VM launch, scientific launch, or commit. The local bound contract and parallel binder both remain non-authorizing: they emit `launch_authorized: false`, and execution requires a separate, exact, independently reviewed runtime-authorization object.

The tuned values were read from `P1-ADAPTIVE-FINAL.md` lines 119–127. The statistical requirements were read from `EXPERIMENT-PROGRAM.md` lines 45–51. The source bytes incorporated into the design contract currently hash to:

- `P1-ADAPTIVE-FINAL.md`: `bda2890669e135f27565c628991c012c1ee297863f93a2f0971c81e37e81d394`
- `EXPERIMENT-PROGRAM.md`: `7ece64ac0c828b183266c451b61c196dbab89762d95703bcf820ca01f7f8e958`

## Frozen E1/E4 design

The exact arm tuples are:

| H | Live memoryless control | Independently tuned momentum | Same-eta momentum contrast |
|---:|---|---|---|
| 16 | `(mu=0, eta=0.021875)` | `(mu=0.9, eta=0.002734375)` | `(mu=0.9, eta=0.021875)` |
| 256 | `(mu=0, eta=0.04375)` | `(mu=0.9, eta=0.0109375)` | `(mu=0.9, eta=0.04375)` |

The eight frozen fresh shuffle/training-seed pairs are:

| Shuffle seed | Training seed |
|---:|---:|
| 383 | 383383 |
| 397 | 397397 |
| 409 | 409409 |
| 421 | 421421 |
| 433 | 433433 |
| 443 | 443443 |
| 457 | 457457 |
| 461 | 461461 |

Materialized sizes are therefore:

- E1: `8 seeds × 2 H × 1 M × 3 arms = 48 cells`, grouped into 16 atomic seed blocks. E1 remains at `M=4`.
- E4: `8 seeds × 2 H × 2 M × 3 arms = 96 cells`, grouped into 32 atomic seed blocks, with `M in {1,16}`.

Explicit `--block-arm H:MU:ETA` declarations are accepted in any declaration order, checked as the exact frozen tuple set for each H, and canonicalized before the seeded within-block permutation. A legacy same-eta three-mu wave cannot substitute for a mixed-eta E1/E4 block.

## File:line implementation summary

| File:line | Change |
|---|---|
| `scripts/compare_diloco.py:173` | Added the `m16` preset alongside the existing `m1`/`m4` presets, so result/work names and the syncer learner count reflect the selected M. |
| `scripts/compare_diloco.py:346` | Added `learner_gpu_packing()`, the pure round-robin packing calculation used to verify 1–4 GPU behavior. The production launcher uses the same `learner_id % gpu_slots` rule at lines 357–374. |
| `scripts/compare_diloco.py:671` | Existing syncer construction is now exercised for M=1 and M=16: it emits `--learners M`, `--quorum M`, and `--strict-quorum`. |
| `scripts/run_phase_map.py:62` | Added supported-M, H, eight-seed, tuned-eta, and E1/E4 study-phase registries. |
| `scripts/run_phase_map.py:324` | Made paired shuffle/training seeds a first-class ordered stage input, including duplicate and mixed-CLI-form rejection. |
| `scripts/run_phase_map.py:376` | Added the exact mixed-eta arm registry, explicit-arm canonicalization, and frozen E1/E4 design validation. |
| `scripts/run_phase_map.py:1279` | Computes exact per-learner work from `token_budget / (M × micro_batch_size × seq_len)` and fails closed on a remainder. |
| `scripts/run_phase_map.py:1310` | Wires M into `--settings m<M>`, the M-length delay vector, per-learner step ceiling, strict barrier/quorum execution, and exact syncer outer-step count. |
| `scripts/run_phase_map.py:1447` | Adds the pairing-command hash, redacting only `--outer-momentum` and `--outer-lr`; every other argv token must match within a block. |
| `scripts/run_phase_map.py:1475` | Builds the block identity binding the frozen model, shuffle/training seeds, learner process-seed formula, seed-specific train file, row-sharding rule, learner IDs, GPU mapping rule, H, quorum, fixed-window schedule, and exact work. |
| `scripts/run_phase_map.py:1553` | Enforces the three-arm block invariant: common `(H, seed, training_seed, M)`, common identity/command hashes, three distinct treatments, and exactly one shared live `mu=0` control. |
| `scripts/run_phase_map.py:1578` | Builds seed and M as cell dimensions; uses `(H, seed, M)` block IDs; independently permutes blocks and arms; emits 48 E1 or 96 E4 cells. |
| `scripts/run_phase_map.py:1849` | Builds the complete local E1/E4 bound runtime contract, requires both governing source documents, and binds their hashes, per-seed train hashes, command registry, design hash, pairing registry, exact work, divergence policy, and `launch_authorized: false`. |
| `scripts/run_phase_map.py:3165` | Generalizes tape, barrier-trace, layout, eval, result, and acquisition-path validation from fixed `m4`/four learners to the cell’s M. |
| `scripts/run_phase_map.py:3875` | Records `m`, exact work, and capped divergence outcomes in terminal result rows. |
| `scripts/run_phase_map.py:4415` | Allows DIVERGED only under the explicit frozen policy, with null raw loss, exact positive cap in `analysis_loss`, and `scientific_divergence` provenance; legacy manifests remain strict. |
| `scripts/run_phase_map.py:4943` | Prevents the serial runner from executing E1/E4 and bypassing the authorized parallel launch controller. Materialization remains allowed. |
| `scripts/run_phase_map.py:5379` | Adds `--seed-pair`, `--m`/`--learner-counts`/`--learners`, `--block-arm`, and `--multiseed-design`. |
| `scripts/run_parallel_phase_map.py:520` | Extends the deterministic roster to carry M, training seed, live-control ID, pairing hashes, and exact learner work for E1/E4. |
| `scripts/run_parallel_phase_map.py:680` | Validates exact 48/96-cell and 16/32-block E1/E4 registries, the eight seeds, allowed M values, mixed-eta tuples, and one live control per block. |
| `scripts/run_parallel_phase_map.py:884` | Uses amendment-style keyed ranking for deterministic slot binding and launch order, with the entire three-arm seed block retained as one loss-blind atomic group. |
| `scripts/run_parallel_phase_map.py:1087` | Extends capacity metadata with the campaign’s learner counts and the maximum learners/GPU for each authorized 4g and 1g shape. |
| `scripts/run_parallel_phase_map.py:1156` | Validates the exact divergence policy; materializes capped analysis fields; validates the exact external authorization object; and materializes the learner-to-GPU map. |
| `scripts/run_parallel_phase_map.py:1917` | Generalizes completed-work evidence to arbitrary M, including exact learner IDs, full M quorum, M-dependent steps, result arm name, and zero exit codes for every learner. |
| `scripts/run_parallel_phase_map.py:2737` | Reconstructs each actual wave, rejects manual slot/launch changes, and enforces one machine type, GPU-slot count, pairing identity, and learner/GPU map throughout an E1/E4 block. |
| `scripts/run_parallel_phase_map.py:2978` | Makes the campaign aggregator require the same external authorization, retain its hash in VM/campaign evidence, and carry the frozen analysis policy into the final campaign manifest/seal. |
| `scripts/run_parallel_phase_map.py:3541` | Adds M, quorum, landed GPU slots, exact learner/GPU map, and pairing identity to every dispatch request. |
| `scripts/run_parallel_phase_map.py:3597` | Makes the wave executor fail before provisioning when E1/E4 authorization is missing/non-exact; chooses a deterministic shape-homogeneous slot subset; binds packing from the landed shape; and materializes capped divergence analysis fields. |
| `scripts/run_parallel_phase_map.py:4149` | The binder writes roster/plan hashes and the exact externally supplied authorization fields required later, while explicitly leaving launch unauthorized. |
| `scripts/validate_phase_map.py:51` | Keeps legacy P0a→P0b source-rebind validation pinned to the four exact committed v1.0–v1.3 amendment revisions rather than one stale hash. The original replay attestation remains pinned to the adopted v1.0 hash. |
| `tests/test_multiseed_executor.py:120` | Adds 16 conformance tests covering the exact E1/E4 grids, explicit mixed-eta arms, pairing drift rejection, M/quorum/work wiring, 1–4 GPU packing, nonintegral-work refusal, deterministic plans, legacy-wave substitution rejection, exact runtime authorization, serial bypass refusal, capped divergence, required source documents, and non-authorizing bound contracts. |

## Pairing invariants and seed plumbing

The same shuffle/training pair is used by all three arms in a block. The relevant runtime path is:

1. `scripts/run_phase_map.py:1310` passes the block shuffle seed as `--shuffle-rows-seed` and the block training seed as `--training-seed`.
2. `scripts/compare_diloco.py:545–557` passes that training seed to every learner as `--seed`, while also passing the same `--num-learners M` and deterministic learner IDs `0..M-1`.
3. `yeto/learner.py:900` derives each process RNG seed as `training_seed + 1009 * learner_id + distributed_rank` before model/optimizer/data-loader construction.
4. `yeto/data.py:164–165` assigns pre-shuffled row positions by `range(learner_id, num_rows, num_learners)`, i.e. row-position modulo M.
5. `yeto/learner.py:1162–1167` shuffles each learner’s packed dataset under the already seeded process RNG.

Consequently, equal shuffle seed, training seed, M, learner IDs, rank layout, seed-specific materialized train file, and fixed H/quorum/work settings establish the required same initialization, data order, worker allocation, and work schedule. The block identity records all of those inputs. The independent pairing-command hash additionally proves that the only command-line differences among the three arms are the registered treatment fields `(mu, eta)`.

At runtime, all three arms must also share:

- the same landed machine type;
- the same GPU-slot count;
- the same exact `learner_id -> local GPU slot` map;
- the same block/time-block identity and terminal-prefix seal;
- the same whole-block retry authorization and retry round, if a retry is needed.

Scientific outcomes never enter block order, slot assignment, launch order, shape selection, or retry assignment. DIVERGED and FAILED outcomes are not retry triggers. Only a registered infrastructure failure can cause the immediately following whole-block retry, with fresh attempts and no checkpoint/optimizer/tape/result reuse.

## M, quorum, and exact work

For every cell:

```text
learner_steps_per_learner = token_budget / (M × micro_batch_size × seq_len)
outer_fragment_commits    = (learner_steps_per_learner / H) × 4 fragments
quorum                    = M, strict
```

Both divisions must be exact. The executor does not pad scientific work, drop a tail, silently round, or export a partially synchronized endpoint.

The historical `655,360`-token budget gives only 320 steps/learner at `M=16`, batch 1, sequence length 128. Since 320 is not divisible by `H=256`, that E4 design is rejected. For the full `{M=1,16} × {H=16,256}` E4 grid, the token budget must be divisible by the largest work quantum:

```text
M × H × micro_batch_size × seq_len
= 16 × 256 × 1 × 128
= 524,288 tokens
```

The conformance tests use `1,048,576` tokens, producing:

| M | H | Steps/learner | Outer fragment commits |
|---:|---:|---:|---:|
| 1 | 16 | 8192 | 2048 |
| 1 | 256 | 8192 | 128 |
| 16 | 16 | 512 | 128 |
| 16 | 256 | 512 | 8 |

## M=16 GPU packing and feasibility

The exact runtime packing is round-robin: `local_gpu_slot = learner_id mod gpu_slots`.

| GPU slots | Busiest GPU | Learners/GPU |
|---:|---|---:|
| 1 | GPU 0 | 16 |
| 2 | GPUs 0–1 | 8 |
| 3 | GPU 0 | 6 (the other two receive 5 each) |
| 4 | GPUs 0–3 | 4 |

The authorized runtime shapes currently project only to 1 GPU (`a2-highgpu-1g`) or 4 GPUs (`a2-highgpu-4g`); 2- and 3-GPU packing were nevertheless verified as pure mapping cases because the launcher formula supports all 1–4 slot counts.

Local exact accounting against cached SmolLM2-135M revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, using PyTorch 2.8.0, found:

- Parameters: `134,515,008`.
- bf16 weights: `269,030,016` bytes = `0.250554 GiB`.
- PyTorch AdamW `exp_avg` and `exp_avg_sq` were both confirmed as `torch.bfloat16` for bf16 parameters.
- Persistent weights + gradients + two Adam moments are therefore at least `1,076,120,064` bytes = `1.002215 GiB` per learner, before activations, temporary workspaces, allocator fragmentation, and CUDA context overhead.

One isolated CPU full-tune step at batch 1 × sequence length 128 measured process RSS as follows:

| Point | RSS bytes | RSS GiB |
|---|---:|---:|
| Process start | 304,152,576 | 0.283264 |
| Model loaded | 318,160,896 | 0.296310 |
| After forward | 785,793,024 | 0.731827 |
| After backward | 1,244,282,880 | 1.158829 |
| After Adam step | 1,785,528,320 | 1.662903 |
| Increment from process start after Adam step | 1,481,375,744 | 1.379639 |

Applying that measured per-process increment only as an accounting analogue gives approximately 5.519 GiB for 4 learners/GPU, 8.278 GiB for 6, 11.037 GiB for 8, and 22.074 GiB for 16. It is not a substitute for CUDA measurement.

No CUDA device or `nvidia-smi` inventory was available locally, so no real A100 smoke test was performed. The operational conclusion is:

- Four learners per A100 (`M=16` on a 4g VM) has ample model-state/activation headroom for this 135M full tune and is the preferred binding.
- Sixteen learners in sixteen separate CUDA processes on one A100 has a roughly 16.0 GiB persistent-state floor and a roughly 22.1 GiB CPU-RSS analogue before CUDA contexts, CUDA workspaces, allocator fragmentation, and other resident processes. It is likely viable on an otherwise empty 80 GiB A100 and may fit a 40 GiB A100, but this has not been empirically established.
- A 1g M=16 fallback must therefore remain conditional on an exact-image preflight/smoke that records total/free GPU memory, confirms zero competing processes, runs all 16 learners through model load + forward + backward + Adam step, and preserves a safety margin. It must not be described as validated merely from this local CPU accounting.

## Exact launch-controller runtime bindings

The launch controller must bind the following before any provider mutation.

### 1. Exact external authorization object

For E1/E4, the controller and final aggregator require an object with exactly these fields and no extras:

```json
{
  "schema": "yeto_multiseed_runtime_authorization_v1",
  "launch_authorized": true,
  "stage_code": "e1-or-e4",
  "best_paper_v2_design_contract_hash": "<64-hex>",
  "roster_hash": "<64-hex>",
  "parallel_plan_hash": "<64-hex>",
  "bound_manifest_canonical_sha256": "<64-hex>",
  "scientific_randomization_plan_hash": "<64-hex>"
}
```

The controller hashes this exact object and carries `multiseed_runtime_authorization_hash` into every VM partial manifest and the final campaign manifest/seal. The local binder only prints the values that an independent reviewer must authorize; it cannot authorize itself.

### 2. Campaign/design identities

- Stage code (`e1` or `e4`), study ID, git commit, image numeric ID/digest, model ID/revision/hash, and data hash.
- Both source-document SHA-256 values and the resulting `best_paper_v2_design_contract_hash`.
- Canonical bound-manifest hash, scientific randomization-plan hash, roster hash, and parallel-plan hash.
- The complete cell-command registry, aggregate command hash, and every cell’s frozen and shape-projected executed command hash.
- Per-seed train-row and train-source-index hashes; shared frozen development/audit eval identities and evaluation-registry path/hash bindings.
- Frozen divergence cap and the requirements that divergence is an outcome and silent exclusion is forbidden.

### 3. Atomic block identities and randomization

- One block ID for exactly one `(H, shuffle seed, training seed, M)` tuple.
- Exactly the three registered `(mu, eta)` arms for that H, with one common live `mu=0` control ID.
- One `pairing_identity_hash` and one `pairing_command_hash` shared by all three arms.
- The deterministic block rank/order, available-slot subset, arm rank, logical-slot assignment, dispatch-batch index, launch rank/order, and time-block index derived from the frozen rank domains.
- A shape-homogeneous runtime subset: one machine type and one GPU-slot count for the entire block.
- A single terminal-prefix seal time covering all three arms before any next or retry block begins.

### 4. Per-arm execution binding

- Cell ID, H, M, mu, eta, shuffle seed, training seed, paired-control ID, command argv/hash, and normalized workload hash.
- `learner_count=M`, `quorum=M`, strict quorum, learner IDs `0..M-1`, and process seeds `training_seed + 1009 × learner_id + distributed_rank`.
- Exact learner steps, H-window size, fixed-window tokens, four-fragment outer-commit count, zero-delay vector, barrier/version-matched settings, and zero WAN streams.
- Landed machine type, `gpu_slots`, exact `learner_gpu_slot_map`, and `maximum_learners_per_gpu`.
- Provider generation/run ID, ownership nonce, instance and boot-disk numeric IDs, region/zone, create-only attempt prefix, provider-evidence hash, and exact teardown lineage.

### 5. Retry and outcome binding

- Only a registered direct infrastructure failure may trigger a retry.
- Retry authorization binds the immediately prior whole-block manifest hash, group ID, parallel-plan hash, retry round, and authorization time.
- Every arm restarts fresh from the frozen initial model; checkpoint, optimizer state, tape, and result reuse are forbidden.
- A completed arm carries its exact finite endpoint in `analysis_loss`.
- A diverged arm carries `status=DIVERGED`, null raw loss, `analysis_loss=<frozen cap>`, `analysis_loss_kind=capped_divergence_endpoint_nll`, and `divergence_retained=true`. It remains in the paired analysis set and is not retried or silently excluded.

## Verification results

All commands used the canonical interpreter `/tmp/yeto-optimizer-state-capture/.venv/bin/python3` with `PYTHONPATH=.`.

- New multi-seed conformance file: `16 passed`.
- Focused phase-map + parallel + multi-seed regression set: `171 passed`.
- Full repository suite: `878 passed, 8 skipped, 5 warnings` in 97.22 seconds. The warnings are unrelated SQLAlchemy 2.0 deprecation warnings from SkyPilot imports.
- `py_compile` passed for `scripts/run_phase_map.py`, `scripts/run_parallel_phase_map.py`, `scripts/compare_diloco.py`, `scripts/validate_phase_map.py`, and `tests/test_multiseed_executor.py`.
- `git diff --check` passed.

## Working-tree handoff

No commit was created. The intended edited/new files are:

- `scripts/compare_diloco.py`
- `scripts/run_phase_map.py`
- `scripts/run_parallel_phase_map.py`
- `scripts/validate_phase_map.py`
- `tests/test_multiseed_executor.py`
- `MULTISEED-CHANGES.md`

The pre-existing untracked context documents `EXPERIMENT-PROGRAM.md`, `P1-ADAPTIVE-FINAL.md`, `P1R0-FINAL.md`, and `PAPER-9B.md` were read but not modified.
