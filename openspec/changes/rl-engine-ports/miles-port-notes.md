# Miles port notes (tasks 1.4 / 1.4b)

Base: `michaellchung/miles` @ `9e4260de`. Local working copy: `/home/michael/work/miles-next`, branch `yeto/ports`, **uncommitted**.
CPU test venv: `/home/michael/work/miles-next-venv` (torch-cpu, ray, no megatron).

## 1.4 `run_plugin` (patch: `/home/michael/work/miles-next-run_plugin.patch`)

- `TrainRayActor.run_plugin(fn_path: str, kwargs: dict[str, Any] | None = None) -> Any` in `miles/ray/train_actor.py`
  (base class, so Megatron and FSDP actors both get it). Loads with `load_function(fn_path, sync_required=True)`, calls `fn(self, **kwargs)`.
- `TrainerController.run_plugin(fn_path, kwargs=None) -> list[Any]` in `miles/ray/train/group.py` (upstream's `TrainGroup`):
  asserts all cells alive, `gather_and_raise_first` over `cell.execute("run_plugin", ...)`, results flattened cell-then-rank order.
- `TrainerCell`: no change needed; its generic `execute(fn_name, **kwargs)` already forwards.
- Deviation from D4 wording: kwargs is an explicit `kwargs` dict, not `**kwargs`/`*args`. Upstream's rpc surface
  (`miles/utils/workers/rpc/common/metadata.py:137-145`) rejects `*args/**kwargs` on any public worker method, and the
  `Pickled` escape hatch is whitelisted (`tests/fast/utils/workers/test_pickled_hatch_boundary.py`). Under
  `--worker-comm-backend rpc`, kwargs and return value must be JSON-serializable; under `ray` they are pickled by Ray.
- Test: `tests/fast/ray/train/test_run_plugin.py` (6 tests; actor call, async plugin rejected, rpc spec round-trip,
  controller ordering, dead-cell refusal, controller rpc surface). Passed together with existing rpc-surface,
  load_state and pickled-hatch tests: 45 passed, 4 skipped.

### Draft upstream PR (not submitted)

Title: `Add run_plugin: run a user function inside every train worker`

Body:
> Out-of-tree integrations sometimes need per-rank code inside the train actor (e.g. exporting/importing trainable
> state, inspecting optimizer state) without subclassing the actor, which is not configurable today.
>
> - `TrainRayActor.run_plugin(fn_path, kwargs=None)`: loads `fn_path` with `load_function` (same convention as the
>   existing `--*-path` hooks), rejects async functions, returns `fn(actor, **kwargs)`.
> - `TrainerController.run_plugin(fn_path, kwargs=None)`: runs it on every worker of every cell (all cells must be
>   alive) and returns results in cell, then rank order.
> - Parameters are wire-typed (no `*args/**kwargs`, no `Pickled`), so the method is callable under
>   `--worker-comm-backend rpc` as well.
>
> Tests: `tests/fast/ray/train/test_run_plugin.py` (CPU only).

## 1.4b agentenv/miles fork commits (source: `/tmp/mf`)

### 5494a6ce "isolate concurrent training ports" -> PARTIALLY equivalent; ported (patch `/home/michael/work/miles-next-5494a6ce.patch`)
Fork parts vs upstream @9e4260d:
- `provider.attention_backend = args.attention_backend` in bridge LoRA provider: **already upstream**
  (`miles/backends/megatron_utils/lora/bridge.py:160`, also `model_provider.py:83`).
- `--rollout-engine-base-port` / `PortCursors`: upstream deleted `miles/ray/rollout/addr_allocator.py`; all dynamic
  worker ports (engines, router prometheus, trainer MASTER_PORT, rpc) now come from one `PortAllocator`
  (`miles/utils/workers/addr_allocator.py`, hard-coded `_DYNAMIC_PORT_START = 20000`), owned by `RayWorkerManager`
  (`miles/utils/workers/ray_worker_manager.py:58`, used at `:339-342`). Specs: `miles/ray/specs/train.py:214`,
  `miles/ray/specs/inference.py:200,360-375` (all `allow_dynamic=True`).
- `--sglang-router-prometheus-port`: superseded (router prometheus is a dynamic port via the same allocator).
- `--train-master-base-port`: superseded (trainer master port also from `PortAllocator`; the remaining
  `random.randint(20000, 21000)` in `TrainRayActor.propose_master_addr_and_port` (`miles/ray/train_actor.py:85`) has no
  caller in-tree and was left untouched).
- Not equivalent: two co-resident drivers each start their allocator at 20000 and probe-then-bind, so they can race to
  the same port. Port: new `--worker-dynamic-port-start` (default 20000) -> `PortAllocator(dynamic_port_start=...)`
  set in `RayWorkerManager.init`; wrap-around returns to the configured start. 2 new tests in
  `tests/fast/utils/workers/test_addr_allocator.py` (15 passed). Wider run of `tests/fast/utils/workers`
  (excl. real_ray) + `test_platform_contract.py` + `test_arguments.py`: 2288 passed, 19 failed, all 19 in
  `test_argv_utils.py` (python argv parsing, untouched by the patch; env-related).
- Router primary port stays static when `--sglang-router-port` is set (`inference.py:215`); co-resident runs must pass
  distinct values or leave it unset.

### 5a9cf0d0 + e2ad83d8 (TITO Qwen3.8) -> EQUIVALENT upstream, not ported
- Upstream has `Qwen38SmallTITOTokenizer` (`miles/utils/chat_template_utils/tito_tokenizer.py:455-465`), type
  `qwen38small` (`:1057`, `:1083`), bundled `templates/qwen3.8_small_and_flash_next_fixed.jinja`, auto-detected for
  `Qwen/Qwen3.8-27B` (`tests/fast/utils/chat_template_utils/test_template.py:218`).
- Fork pinned `enable_thinking=True, preserve_thinking=True, reasoning_effort=xhigh` on the native template. Upstream pins
  `preserve_thinking=True`, keeps `enable_thinking`/`reasoning_effort` consistent across turns (`consistant_kwargs`),
  and the fixed template defaults `reasoning_effort` to `xhigh` (jinja line 47). To get fork behaviour, launch with
  `chat_template_kwargs={"enable_thinking": true}` (effort default already xhigh).
- e2ad83d8 (dummy user turn) worked around the *native* template rejecting assistant-without-user; the upstream fixed
  template has no such `raise_exception` (only lines 10,21,33,39,43,49,103,157), so the fix is unnecessary.
- Fork type name `qwen38` does not exist upstream; yeto config must use `qwen38small`.
- Tests: `tests/fast/utils/chat_template_utils` full: 1889 passed, 10 skipped (Qwen3.8 subset: 30 passed).

## LoRA export: canonical PEFT names? -> NO (SGLang-style HF names)
- Adapter export path: `LoraActor.export_slot` -> `SnapshotPublisher.write_adapter`
  (`miles/backends/training_utils/weight_update/snapshot_publisher.py:41-59`) writes raw
  `AutoBridge.export_adapter_weights` names (`update_weight/hf_weight_iterator_bridge.py:67-76`, only `.base_layer.`
  removed). Megatron-Bridge's own `save_hf_adapter` runs `convert_adapter_weights_to_peft_state` first; Miles does not.
- Resulting keys look like `model.layers.0.self_attn.q_proj.lora_A.weight` (see fixtures in
  `tests/fast/backends/training_utils/weight_update/test_hf_weight_iterator.py:25`): no `base_model.model.` prefix,
  shared MoE factor kept as a single `[1, ...]` tensor (SGLang layout), fused modules may appear under fused HF names
  (e.g. `gate_up_proj` in inkling). `get_adapter_target_modules` tolerates either prefix.
- Implication for yeto `TrainableState`/policy_hash: normalise names (add `base_model.model.` prefix) in the
  state plugin, or hash on Miles names and convert only at PEFT save. Not verified on GPU (megatron-bridge not
  installed in CPU venv).
