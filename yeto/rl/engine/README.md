# yeto/rl/engine: core and adapter conventions

This directory is the yeto RL engine. Everything here is **core**: it knows only
the five role ports defined in `ports.py` (`RolloutPool`, `TrainerGroup`,
`PolicyState`, `Publisher`, `Placement`) and neutral data (opaque batch handles,
`TrainableState`, `AlgorithmSpec`, `ExecutionProfile`). It never touches a
concrete training framework. Framework code lives in the backend adapters under
`yeto/rl/adapters/<backend>/`.

## Boundary rules (checked by `tests/test_import_boundaries.py`)

1. The neutral core — `yeto/rl/engine/`, `yeto/rl/rewards/`, `yeto/rl/harness/`,
   `yeto/rl/algos/`, plus the reward / filter / algorithm-extension / harness files
   that still live directly under `yeto/rl/` (listed in `CORE_FILES` in the test) —
   must not import:
   - training or inference frameworks: `miles`, `miles_plugins`, `megatron`,
     `sglang`, `verl`, `vllm`, `torch_npu`;
   - backend adapters: `yeto.rl.adapters.miles`, `yeto.rl.adapters.*`;
   - Miles patch and entry modules: `yeto.rl.adapters.miles.overlay`,
     `yeto.rl.adapters.miles.overlays`, `yeto.rl.adapters.miles.island_entry`,
     `yeto.rl.adapters.miles.models.full_parameter*`, and so on;
   - cloud libraries: `sky`, `modal`, `yeto.modal_runner`, `yeto.shape.providers`.
2. Backend adapters must not import each other.
3. The cloud / launch layer (`launcher.py`, `modal_runner.py`, `cli.py`,
   `stage_w_entry.py`, `yeto/shape/`) must not import a backend adapter.

The check is a syntax-tree scan: imports inside functions, relative imports and
literal `importlib.import_module(...)` calls all count.

## Existing violations and the allowlist

Existing violations are listed in `tests/import_boundary_allowlist.txt` (file,
forbidden module, note). The list **only shrinks**:

- a new violation fails the test, which prints the file, line and forbidden module;
- a fixed violation whose allowlist entry was not removed fails the test with a
  request to delete the entry;
- the number of entries must equal `ALLOWLIST_CAP` in the test; lower the cap
  when you delete an entry.

When the core needs a backend capability, add a neutral optional method or field
to a port or to the capability declaration (`capabilities.py`) and implement it in
the adapter. Do not import the adapter from the core.

## Golden samples

`tests/test_decoupling_golden.py` must pass: for typical configurations the Miles
command line, `AlgorithmSpec.sha256()`, `ExecutionProfile.contract_hash`, plugin
source hashes, strict/decoupled progress content and fake-engine tape excerpts
must match `tests/golden/decoupling/` byte for byte. An intentional hash change
regenerates the golden files in the same change.

## Backends

- The core, the launch layer and the CLI reach backend modules only through the
  registry in `yeto/rl/engine/backends.py`, by role (`backends.module("entry")`
  and so on); they never import an adapter statically. The backend is
  `--rl-backend` (launcher) or the `YETO_RL_BACKEND` environment variable
  (inference processes and subprocesses); with neither set it is `miles`.
- The old Miles engine is removed; no file may import or name it
  (`test_removed_legacy_engine_is_never_imported`,
  `test_removed_legacy_engine_is_never_named`).
- Policy tokens and round counters inside inference processes go through
  `yeto/rl/engine/rollout_meta.py`. Dynamic sampling filters use neutral names; a
  backend translates them in its `binding` role and runs the binding checks before
  launch.

## Miles adapter (`yeto/rl/adapters/miles/`)

Connects the five core ports to the pinned Miles commit (`MILES_NEXT_COMMIT` in
`yeto/rl/adapters/miles/pins.py`). This directory may import
Miles / Megatron / SGLang / Ray; the core may not import it.

What it translates:

- Run config to Miles argv: `config.py` (`translate_run_config`) turns
  `RLRunConfig` + `AlgorithmSpec` into upstream Miles argv and returns
  `MilesLaunchArgs` (argv, placement request, algorithm hash, runtime
  attributes). Every config leaf is either translated or refused.
- Algorithm fields to Miles flags: `algorithm_flags.py` (`algorithm_argv`,
  `absorb_extra_argv`, including the `--dry-run` check entry).
- Port implementations: `rollout.py` (RolloutPool), `trainer.py` (TrainerGroup),
  `state.py` + `state_plugin.py` (PolicyState, exported and applied inside each
  Megatron rank), `publish.py` (Publisher, over `update_weights`), `placement.py`
  (placement description and rewrite detection).
- Composition root: `entry.py` (capability declaration, execution profile, island
  assembly and the Ray connection).
- Elasticity and cut points: `elastic_*`, `cut_plugin.py`, `round_cut.py`,
  `trainer_rebuild.py`, `trainer_resize.py`, `rebuild_wiring.py`, `reshard.py`,
  `bundles.py`.
- Metadata hook inside the Miles rollout process: `rollout_meta_hook.py`.
- Remote-code prewarm before engine start: `remote_code_prewarm.py` (after the
  island's Ray is up and before Miles starts any engine, one subinterpreter per
  machine fills the transformers module cache from the model directory's `*.py`
  under a file lock and runs AutoConfig / AutoTokenizer / AutoProcessor /
  AutoImageProcessor once; a failure only logs a warning).
- Miles patches: `overlay.py`, served to the launch layer through the registry's
  `overlay` role; the patch files are in `overlays/`.

What it declares:

- Capabilities: `entry.miles_capabilities()` and `MILES_DECLARED`; some
  mechanisms depend on the Miles commit (`MILES_DECLARED_PINS`). An undeclared
  mechanism is refused before launch unless `--rl-allow-unverified-mechanism` is
  given.
- Execution profile: `entry.execution_profile_for()` (colocated-serial,
  partitioned-serial, partitioned-overlap), bound to the algorithm hash the
  launcher computed.
- Runtime fingerprint: `entry.ports_runtime_fingerprint()`.

What it does not support:

- A config item with no translation raises `UnmappedConfigError`.
- Miles fault-tolerance semantics, more than one cell per trainer group, and a
  placement rewritten by Miles are refused (`FaultToleranceArgsError`,
  `SingleCellError`, placement rewrite detection).
- NVIDIA CUDA only (the determinism environment `cut_plugin.DETERMINISM_ENV` is
  CUDA-specific).
