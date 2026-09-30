# Upstream mechanisms for E0–E3 (research note)

Pins: Miles `/home/michael/work/miles-next` = `yeto/ports` (radixark/miles `9e4260de` + `e8b3b657` run_plugin + `03947150` port isolation).
SGLang `/home/michael/work/sglang-next` HEAD `9f29303be` (port of agentenv `e1b57eb`, disk-backed weight memory saver; yeto pin `SGLANG_COMMIT` in `yeto/rl/__init__.py:19`).
Megatron reference: `/tmp/mcore` (temporary checkout; re-verify against the image Megatron before citing it in a PR).
Unless a line says otherwise, paths are relative to the Miles fork. Line numbers are for the commits above.

Task mapping: E0 = tasks group 2, E1 = group 3, E2 = 4.1–4.5, E3 = 4.6–4.8. Group 5 (optimization) reuses the E2/E3 mechanisms.

---

## E0: fixed partition (tasks 2.1–2.4)

| Mechanism | Citation | Consequence |
|---|---|---|
| One Ray PG, `PACK`, one `{GPU:1,CPU:1}` bundle per GPU, sorted by (node, gpu) | `miles/ray/placement_group.py:81-119` | Trainer and rollout share one PG. The split is a single offset, not an explicit map. |
| Layout: non-colocate = `trainer + rollout(+eval)`, with rollout starting at offset `trainer_num_gpus`. Colocate = `max(trainer, rollout)`, offset 0 | `placement_group.py:122-138`, `:146-162` | Physical GPUs cannot be chosen. A "standby" GPU is expressible only as extra rollout cells that are never started (see E1). |
| `trainer_num_gpus = actor_num_nodes*actor_num_gpus_per_node*num_policies` | `placement_group.py:141-143` | |
| Colocate normalization: forces `offload_train/offload_rollout=True` and overrides `rollout_num_gpus` to the trainer size | `miles/utils/arguments.py:3609-3630` | This is why `yeto/rl/engine/miles_adapter/placement.py` detects rewrites. |
| `--debug-rollout-only` turns colocate off and rewrites the actor size | `arguments.py:3548-3557` | Must stay rejected for E0 runs. |
| `--offload` expands to both offloads | `arguments.py:3543-3546` | |
| Weight transport chosen by placement: colocate → CUDA IPC (`UpdateWeightFromTensor`), otherwise `--update-weight-transfer-mode` broadcast / disk-delta / p2p | `miles/backends/training_utils/weight_update/protocol.py:73-89` | |
| LoRA: broadcast and CUDA IPC set `supports_lora=True` (`protocols/broadcast.py:27`, `protocols/cuda_ipc.py:43`). p2p and disk-delta assert no LoRA | `arguments.py:3581`, `:3595` | A LoRA fixed partition **must** use `broadcast` (NCCL group trainer-rank0 + all engine GPUs). |
| LoRA `--lora-base-cpu-backup` applies only with colocate | `miles/utils/lora/utils.py:39-41`, `utils/lora/arguments.py:80-82` | Drop it in partitioned mode (it has no effect, and the flag set must stay consistent for the fingerprint). |
| Multi-LoRA forbids colocate and offload_train | `utils/lora/arguments.py:250-256` | Not used, but it is proof that the partitioned path is supported upstream. |
| `--fully-async` forbids colocate | `arguments.py:85-89` | Upstream's only train/rollout overlap mode. It is not an island-internal policy for us (task 2.3 must not silently enable it). |

**What the LoRA path forces.** Partitioned + LoRA requires broadcast transport. Every publish connects over NCCL to every engine, so the trainer must *not* offload between steps (no `--offload-train`). Also, `load_state` hot reload is unusable with LoRA (E2 below).

**Missing for 2.1.** Miles has no explicit GPU→role map. The `Placement` port can express only "first T GPUs trainer, next R rollout". Real standby/NUMA placement needs a fork commit to `create_placement_groups` that accepts an explicit bundle list (fork-M1).

---

## E1: rollout engine add/remove (tasks 3.1–3.8)

**Membership chain:** worker manager → provider watch → `InferenceController._reconcile` → `RolloutServer.add_cell/remove_cell` → `ServerCell` state machine → router.

| Mechanism | Citation | Notes |
|---|---|---|
| `RayWorkerManager.start_cells / stop_cells` under `_membership_lock` | `miles/utils/workers/ray_worker_manager.py:88-102` | Works only on **pre-declared** cell ids: `_PoolManager.initial` makes `range(spec.scheduling.num_cells)` cells bound to PG bundles (`:178-191`). Cells beyond the startup count cannot be created. |
| `RayWorkerProvider.stop_cells` is marked TEMPORARY ("not meant to serve a suspend...") | `miles/utils/workers/worker_provider/ray.py:28-31` | The provider exposes no `start_cells`. |
| `InferenceController._reconcile` (watch callback) adds/removes cells | `miles/ray/rollout/inference_controller.py:359-382` | Driven by provider observation, not by an imperative API. |
| `InferenceController.stop_cell_between_weight_updates` (TEMPORARY, takes the lock) | `inference_controller.py:91-95` | The only imperative removal. It is labelled as a stopgap for weight-update FT. |
| `RolloutServer.add_cell` (init unless colocate+offload) / `remove_cell` (dispose) | `miles/ray/rollout/rollout_server.py:111-135` | |
| `ServerCell`: Uninitialized→Initializing→PendingWeights→Serving. It is registered with the router only in `mark_weights_ready` | `miles/ray/rollout/server_cell.py:206-217` | A new engine takes no traffic until it has weights. This matches 3.5 "isolated load then atomic commit". |
| `ServerCell.dispose`: health checker stop → router `remove_worker` (timeout, warn and continue) | `server_cell.py:219-246` | **No drain.** |
| Miles router `remove_worker` only deletes counters | `miles/router/router.py:62-63`, `:195-214`; selection `_use_url` `:229-240` | In-flight HTTP requests are not tracked for drain, and there is no "cordon". |
| Health monitor pause/resume via `health_checker_activeness.bump_active` | `inference_controller.py:385-393` | Paused in `start_update_weights` and resumed in `prepare_rollout` (`:150-153`). |
| Weight-update membership: `start_update_weights` → `UpdatableEngines{rollout_engines, gpu_counts, gpu_offsets, snapshot_cell_id_to_hashes}` | `inference_controller.py:219-239`, dataclass `:396-401` | `end_update_weights` marks ready only the cells whose hash matches the snapshot (`:245-256`). This gives membership-generation semantics we can reuse for epochs. |
| Trainer reconnects when the membership hash changes | `miles/backends/megatron_utils/actor.py:906-916` | Broadcast `connect` tears down and rebuilds the NCCL group `miles-pp_{shard}` over all engines (`protocols/broadcast.py:42-68`, `:85-128`). Any add/remove means a full group rebuild on the next publish, which blocks until every engine joins. |
| `abort_all` for the whole controller | `inference_controller.py:123-126`, `rollout_server.py:154` | Not per engine. |
| SGLang endpoints: `/pause_generation`, `/continue_generation`, `/abort_request`, `/init_weights_update_group`, `/destroy_weights_update_group`, `/release_memory_occupation`, `/resume_memory_occupation` | sglang `python/sglang/srt/entrypoints/http_server.py:1787`, `:1800`, `:1709`, `:1411`, `:1427`, `:1582`, `:1594` | Per-engine drain primitives exist on the engine side. |

**Viable E1 design with minimal fork.** Declare `num_cells = max rollout engines` at startup, with standby cells stopped (their GPUs count as reserved). Then:

- **Add:** `start_cells` → reconcile → PendingWeights → yeto `Publisher.publish(policy, members)` → `mark_weights_ready` registers with the router.
- **Remove:** yeto admission fence (3.3) → per-engine `/pause_generation` → wait until active is 0 (tool-wait is tracked by yeto, not Miles) → `stop_cells`.

**Missing (fork commits).**
- **fork-M2:** a non-TEMPORARY public `InferenceController.start_cells/stop_cells(cell_ids, expected_epoch)` plus `start_cells` on `RayWorkerProvider`.
- **fork-M3:** per-cell cordon (router remove without dispose) and a drain wait (per-worker in-flight count exposed by the router).
- **fork-M4:** `UpdatableEngines` restricted to an explicit member set (for publish to new members only, or a full re-publish with an epoch).

SGLang: no fork needed for E1.

**Weight-group cost.** Because of the broadcast rebuild, every membership change costs one NCCL init across all engines (task 5.3 targets this).

---

## E2: same-shape recovery with ReconfigurationCut (tasks 4.1–4.5)

| State | Upstream save/load | Citation | Gap |
|---|---|---|---|
| Full-param model + distributed optimizer + scheduler | `megatron_utils/model.py:866-905` `save()` → Megatron `save_checkpoint`. Load via `_load_state_core`, honoring `no_load_optim/no_load_rng` | `megatron_utils/checkpoint.py:147`; fresh-run defaults set `no_load_optim/no_load_rng/finetune` (`megatron_config.py:365-372`); `--debug-disable-optimizer` sets no save/load optim (`arguments.py:3317-3319`) | The default launch path loads no optimizer/RNG. A cut must override these explicitly (as the heal path does with `heal_load_overrides`, `actor.py:206-208`). |
| LoRA adapter + optimizer + scheduler | `save_lora_checkpoint` writes `adapter_megatron_rank{r}.pt` + `training_state_rank{r}.pt` = {iteration, optimizer.state_dict(), scheduler} | `megatron_utils/lora/utils.py:170-214`; load `:290-310` | **No RNG, no data-iterator state.** The state is keyed by *global rank*, so it is not reshardable (this blocks E3 for LoRA). |
| Hot in-process reload `TrainerGroup.load_state` → `actor.load_state` | `megatron_utils/actor.py:334-380`, `ray/train/group.py:364` | Asserts **not LoRA, not colocate, not offload_train, not local non-persistent ckpt, not keep_old_actor**. Unusable for our LoRA profile, so E2 must use a **process rebuild** (4.3), not `load_state`. |
| Post-init RNG capture `RandomState.capture()` | `actor.py:214` | A cut needs an explicit RNG save/restore (CPU, CUDA, and Megatron tracker) through `run_plugin`. |
| Rollout data source (sample_offset, epoch_id, group/sample index, metadata) | `miles/rollout/data_source.py:128-160`; executor save/load `miles/ray/rollout/rollout_executor.py:295-310` | Saved only if `rollout_global_dataset`. It does **not** include in-flight/ready-unconsumed groups, so yeto's ledger (3.6) must own those. |
| `run_plugin` hook on the trainer | `miles/ray/train/group.py:467` (fork commit `e8b3b657`) | The natural carrier for `save_cut/restore_cut` without forking Megatron code paths. It is already used by `yeto/rl/engine/miles_adapter/trainer.py:112`. |

**What a quiescent cut needs.** It must be taken between optimizer steps, with no pending async save (`_finalize_pending_async_save`, `actor.py:357`).

1. The rollout admission fence and drain have finished (E1).
2. No publish is in flight: the `start_update_weights` lock is released.
3. It records: model and adapter, optimizer moments and master params, scheduler step, RNG, data-source cursor, yeto ledger (ready groups, policy token, epoch), and the bridge/outer-sync state.

**Authority rule.** The yeto syncer checkpoint stays the sole durable authority. A `ReconfigurationCut` is an *ephemeral, epoch-scoped* snapshot: manifest + fsync (4.2), deleted on commit, and never loadable as a resume point by `--load`. Miles `save_model` must not be driven by the cut (it would advance Miles' tracker and collide with syncer semantics). Write through `run_plugin` to a cut dir instead.

**Forbidden alternatives.** Miles' own recovery is `TrainerController._refresh_cells` / indep-DP `recv_ckpt` healing (`ray/train/group.py:529-690`, `actor.py:197-208`, `miles/utils/ft_utils/indep_dp.py`) and `mini_ft_controller.py`. It is exactly the FT stack yeto rejects (`yeto/rl/engine/miles_adapter/config.py:11-14`, `reject_fault_tolerance_flags` `:396-409`). Use a fresh trainer group + `restore_cut` instead.

---

## E3: trainer DP change and GPU role transfer (tasks 4.6–4.8)

- **In-process DP change: no.** Megatron `parallel_state` and the DDP/distributed-optimizer buffers are built once in `build_model_and_optimizer` (`actor.py:209`). The only upstream "DP change" is indep-DP cell add/remove (`ray/train/group.py:529-690` `_refresh_cells`, `_compute_indep_dp_info` `:676`), and that is forbidden FT. A DP change therefore means **destroying the trainer group and re-creating it with the new world size** (process restart), followed by a restore.
- **Distributed optimizer resharding.** Megatron dist-ckpt can load the optimizer at a different DP through `sharding_type` `dp_reshardable` / `fully_reshardable` (`/tmp/mcore/megatron/core/optimizer/distrib_optimizer.py:127-128`, `:1075-1088`, `:1460-1500`). Miles notes that new Megatron replaced `fully_shard` with `dp_reshardable` (`megatron_utils/arguments.py:34-36`). So resharding is viable **for full-parameter torch_dist ckpts only**. The LoRA path saves `optimizer.state_dict()` per global rank (`lora/utils.py:192-214`, `:300`), so it **cannot** reshard. A fork commit is needed (fork-M5: save LoRA optimizer state as a DP-invariant sharded/gathered dict, and add RNG). The 4.6 spike decides go/no-go.
- **GPU handoff between trainer and rollout.**
  - Offload/onload exists for colocate only: `TrainerController.offload/onload` (`ray/train/group.py:404-420`), `InferenceController.offload/onload/offload_weights/onload_kv` (`inference_controller.py:175-214`), and SGLang release/resume memory occupation.
  - torch_memory_saver: train disk-offload dir (`megatron_utils/actor.py:86-99`, spec preload `miles/ray/specs/train.py:259-267`), `weights_backuper`/`get_cpu_backup` (`actor.py:238`, `megatron_utils/named_weights.py:38-40`).
  - SGLang `enable_weights_disk_backup` (sglang `arg_groups/fields/exec_.py:69`, used at `model_runner_components/load_model_utils.py:314`; port notes in `openspec/changes/rl-engine-ports/sglang-patch-port.md:30`) lets an engine drop weights to disk instead of pinned CPU. That helps a *memory-time-share*, but it is not a role transfer: the GPU stays in the same PG bundle with both processes resident.
  - A real transfer (e.g. P62↔P44) needs: trainer group recreated on a different bundle subset, rollout cells created on freed bundles, and the PG split changed. Upstream has a fixed offset split (`placement_group.py:146-162`) and fixed per-pool cells, so it **requires fork-M1 (explicit bundle map) and fork-M6** (`RayWorkerManager` able to (re)bind a cell to a different bundle index, and trainer spec re-creation with new `actor_num_gpus`).

---

## Forbidden by our rules → alternatives

| Forbidden | Where | Alternative |
|---|---|---|
| indep-DP / `_refresh_cells` / heal `recv_ckpt` | `ft_utils/indep_dp.py`, `ray/train/group.py:529-690`, `actor.py:197-208` | yeto `TrainerGroup.rebuild(plan)` + `restore_cut` (process rebuild) |
| `mini_ft_controller` | `miles/utils/ft_utils/mini_ft_controller.py:47-170` | yeto single-transaction controller with a journal (3.2) |
| FT api_server (`maybe_start_api_server`, `ft_utils/api_server/handles.py:91 resume`) | `placement_group.py:347` | yeto plan/status/cancel entry (3.2) |
| `stop_cell_between_weight_updates` / `inject_fault_*` (TEMPORARY, FT-coupled) | `inference_controller.py:91-113` | fork-M2 public cell start/stop with an epoch; fault injection only in the test harness |
| Miles FT health-checker-driven cell replacement | `ft_utils/health_checker.py`, `server_cell.py:292-310` | Keep the monitor for *detection* only. yeto decides and commits the epoch. |
| `--fully-async` as "overlap" | `arguments.py:85-89` | partitioned-serial, plus only algorithm-legal overlap (2.3) |

---

## Gaps: port verbs vs `yeto/rl/engine/ports.py`

| Needed verb (tasks) | ports.py reserved comment | Upstream support | Fork needed |
|---|---|---|---|
| `RolloutPool.add_engines(count, *, epoch)` (3.4) | `:96` | start of pre-declared cells + reconcile + PendingWeights | Miles fork-M2 |
| `RolloutPool.remove_engines(members, *, epoch)` (3.4) | `:97` | stop_cells (TEMPORARY) | Miles fork-M2 |
| `RolloutPool.drain(members, deadline)` (3.3/3.4) | **not reserved** | SGLang `/pause_generation`, `/abort_request`; router lacks cordon and in-flight counts | Miles fork-M3 (router cordon + inflight); SGLang none |
| `Publisher.publish(policy, members)` (3.5) | not reserved (current `publish(state)` `:124`) | `UpdatableEngines` hash snapshot, `end_update_weights` | Miles fork-M4 (member-filtered UpdatableEngines) |
| `Placement.reconfigure(plan, epoch)` (3.4, 4.7) | `:130` (E3 only) | none (fixed offset PG) | Miles fork-M1 (+M6 for E3) |
| `TrainerGroup.save_cut / restore_cut` (4.2) | `:105-106` | `run_plugin` + Megatron save; LoRA branch lacks RNG | Mostly yeto plugin code. Miles fork-M5 for a LoRA RNG/reshardable optimizer. |
| `TrainerGroup.rebuild(plan)` (4.3) | `:107` (E3 generic) | `create_training_models` (`placement_group.py:216`) at startup only | Miles fork-M6 (re-create trainer handles on a new bundle set, dispose old) |
| Pool/membership epoch query (`members()` + epoch) | `:95` without epoch | cell statuses `get_cell_statuses` (`inference_controller.py:319`) | none |

The reserved comment for `reconfigure` is marked E3. The spec needs it at E1 (task 3.4), so update the comment. `drain` and member-scoped `publish` are not reserved at all.

---

## GPU experiments

The estimates below assume a small model (e.g. Qwen3-0.6B/1.7B LoRA). Hours are wall-clock including image pull and warmup. Real rental follows task 1.3.

| Stage | Experiments | GPUs | Type | Est. duration |
|---|---|---|---|---|
| 1.2 baseline | serial colocated compat on ports | 4 (1 node) | H100/H200 80GB (A100 80GB acceptable) | 2–3 h |
| E0 2.1–2.2 | fixed partition start + partitioned-serial N steps (LoRA broadcast) | 8 (T4R4, T4R2S2) | same | 4–6 h |
| E0 2.4 | fixed-config sweep ≈4–6 configs × 2 seeds × ~1 h | 8 | same | 10–14 h |
| E1 3.4–3.8 | T4R2S2↔T4R4S0 both directions, failure injection (start/publish/NCCL timeout), two-small-island strict pause | 8 per island; 16 for the 2-island test | same | 12–20 h (8 GPU) + 4–6 h (16 GPU) |
| E2 4.3–4.5 | same-shape rebuild + frozen-batch numeric compare; fault matrix (~8 cases) | 8 | same | 8–12 h |
| E3 4.6 spike | DP1↔2 reshard of a full-param dist-ckpt + LoRA after fork-M5, numeric next-step compare | 4–8 | same | 6–8 h |
| E3 4.7–4.8 | P62↔P44 (or smaller equivalent) role transfer both directions; multi-seed learning validation, fixed vs same-shape vs DP-change | 8 (P62/P44 needs 8, or 12 with spare) | same, single NVLink node | 30–50 h (≈3 seeds × 3 arms × 3–5 h) |

**Total:** about 80–120 GPU-node-hours, mostly on 8×H100 single nodes, plus one 16-GPU (2-node) window. E1 and E2 can share rentals.
