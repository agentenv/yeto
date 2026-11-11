"""Kimi-K3.1 synced trainer islands behind the Rust syncer (CPU rehearsal; no GPU, no Ray).

The K3.1 recipe trains every parameter of the selected experts in place
(``--tuning full --rl-model-recipe kimi_k31``). Behind a syncer, K >= 1 trainer
islands of ``--rl-sample-role trainer`` run the full-parameter preset with
asynchronous rounds: each island reads its buffer lane, pushes its
full-parameter pseudo-gradient (the delta of the selected experts' parameters),
the syncer merges the K pseudo-gradients and broadcasts the merged policy, the
publisher island (learner 0) writes the merged release with the selection's
sha256 in its manifest, and the samplers refuse every release of another
selection. One learner (learners=1 / quorum=1) is the degenerate form: the
merge is the identity.

Covered here:

1. the launcher rules (``rl_args``, ``sample_roles.check_synced_trainers``,
   ``selection.ports_rejections``): two learners and one learner pass, the
   island commands parse on the Miles island entry and pass its own checks, the
   no-sync GPU-A form still passes, LoRA synced trainers keep their rules;
2. the Miles k31/b2 pin and the per-rank state helpers yeto reads through
   (``full_parameter.rank_state``), with fakes of the Miles module;
3. the real Rust syncer merging two fake trainer islands' pseudo-gradients
   (and one island's, identity), the merged release written to a fake object
   store with the K3.1 manifest, and the sampler-side selection gate.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from free_ports import free_port
from process_groups import group_popen, kill_group
from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from test_rl_k31_e2e_cpu import EP, MOE_LAYERS, NUM_EXPERTS, NUM_TRAINABLE, Bucket, _hf_config
from yeto.rl.adapters.miles import island_entry, pins
from yeto.rl.adapters.miles.models.kimi_k31 import release as k31_release
from yeto.rl.adapters.miles.models.partial_experts import selection as sel
from yeto.rl.engine import sample_roles
from yeto.rl.engine.merged_release import BufferLane
from yeto.rl.engine.selection import FULL_PARAMETER_SYNC_PRESET, ports_rejections

LAYOUT = "e" * 64
STORE = "modal-volume://yeto-k31/run"
TWO = "modal:1x8xb300,modal:1x8xb300"
ONE = "modal:1x8xb300"


@pytest.fixture(autouse=True)
def _island_hmac_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k31-synced-test-key")


def _selection() -> sel.ExpertSelection:
    return sel.ExpertSelection.synthetic("uniform", model="kimi_k31", num_experts=NUM_EXPERTS,
                                         num_trainable=NUM_TRAINABLE, ep_size=EP, layers=MOE_LAYERS)


def _tis_spec(tmp_path: Path) -> Path:
    """An age-2 spec with a declared importance-sampling correction (a run with
    outer sync takes declared mechanisms only, no --rl-allow-unverified-mechanism)."""
    from yeto.rl.engine.algorithm import AlgorithmSpec

    spec = AlgorithmSpec().structured_dict()
    spec["execution"] = {**(spec.get("execution") or {}), "max_policy_staleness": 2}
    spec["correction"] = {**(spec.get("correction") or {}), "method": "tis", "upper": 2.0, "lower": 0.5}
    path = tmp_path / "algorithm_spec.json"
    path.write_text(AlgorithmSpec.from_dict(spec).canonical_json())
    return path


def _k31(tmp_path: Path, *extra: str, synced: bool = True) -> tuple[str, ...]:
    """The K3.1 trainer-island launch flags: full tuning, the recipe and its
    selection, EP8 on one B300 node, samples from the store; synced (full-parameter
    preset, async rounds) or the GPU-A no-sync form."""
    selection_path = tmp_path / "expert_selection.json"
    _selection().write(selection_path)
    form = (("--rl-sync-preset", FULL_PARAMETER_SYNC_PRESET, "--rl-sync-full-parameter-rounds", "async",
             "--rl-parameter-layout-sha256", LAYOUT) if synced else ("--rl-single-island-no-sync",))
    return ("--tuning", "full", "--rl-model-recipe", "kimi_k31", "--rl-expert-selection", str(selection_path),
            "--expert-parallel", "8", "--tensor-parallel", "1", "--rl-sample-role", "trainer",
            "--rl-checkpoint-store", STORE, "--rl-max-policy-age", "2", "--rl-rollout-top-logprobs-num", "128",
            "--rl-algorithm-spec", str(_tis_spec(tmp_path)), "--cluster-prefix", "k31-synced", *form, *extra)


def _plan(tmp_path: Path, gpu: str, *extra: str, synced: bool = True):
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    args = _launcher_args("ports", _k31(tmp_path, *extra, synced=synced), gpu=gpu)
    args.controller = "local"
    _prepare_rl_args(args)
    return args, dry_run_plan(args)


def _island_args(plan: dict, island: dict, learner: int):
    tail = island["learner_command"].split("-m yeto.rl.adapters.miles.island_entry", 1)[1]
    tokens = [{"$LEARNER_ID": str(learner), "$SYNCER_ADDR": "10.0.0.1:29400"}.get(t, t) for t in shlex.split(tail)]
    return island_entry.parse_args(tokens)


def _syncer_flags(plan: dict) -> dict[str, str]:
    (actor,) = plan["role_syncers"]
    tokens = shlex.split(actor["command"].split("$(", 1)[0])
    return {tokens[i]: tokens[i + 1] for i in range(1, len(tokens) - 1) if tokens[i].startswith("--")}


# ------------------------------------------------------------------ 1. launcher rules

def test_two_k31_trainer_islands_behind_the_syncer_pass_every_launch_rule(tmp_path):
    args, plan = _plan(tmp_path, TWO)
    assert plan["outer_sync"] and plan["total_gpus"] == 16 and len(plan["island_requests"]) == 2
    assert sample_roles.synced_full_parameter_trainer(args) and sample_roles.is_synced_trainer(args)
    assert not sample_roles.buffer_full_parameter(args)  # not the no-sync form
    flags = _syncer_flags(plan)
    assert flags["--learners"] == "2" and flags["--quorum"] == "1"
    assert flags["--outer-lr"] == "1.0" and flags["--outer-momentum"] == "0.0"  # H=1: the plain mean
    for k, island in enumerate(plan["island_requests"]):
        assert island["placement"]["placement"] == "colocated"  # a trainer island: no engine partition
        parsed = _island_args(plan, island, k)
        assert parsed.rl_sample_role == "trainer" and not parsed.rl_single_island_no_sync
        assert parsed.parameter_mode == "full" and parsed.rl_model_recipe == "kimi_k31"
        assert parsed.sync_preset == FULL_PARAMETER_SYNC_PRESET and parsed.full_parameter_rounds == "async"
        assert parsed.rl_sample_lanes == 2 and parsed.num_learners == 2 and parsed.learner_id == k
        assert sample_roles.buffer_lane(parsed) == BufferLane(k, 2)
        sample_roles.check_sample_store(parsed)  # the island-side store check
        island_entry._require_ports_supported(parsed)  # the island-side ports matrix


def test_one_k31_trainer_island_behind_the_syncer_is_learners_one_quorum_one(tmp_path):
    args, plan = _plan(tmp_path, ONE)
    assert plan["outer_sync"] and len(plan["island_requests"]) == 1
    flags = _syncer_flags(plan)
    assert flags["--learners"] == "1" and flags["--quorum"] == "1"
    # the merge of one pseudo-gradient is the identity: no outer optimizer
    assert flags["--outer-lr"] == "1.0" and flags["--outer-momentum"] == "0.0"
    assert args.rl_sync_mode_effective == "async"
    parsed = _island_args(plan, plan["island_requests"][0], 0)
    assert parsed.num_learners == 1 and parsed.rl_sample_lanes == 1 and parsed.learner_id == 0
    assert sample_roles.buffer_lane(parsed) is sample_roles.OWN_LANE if hasattr(sample_roles, "OWN_LANE") else True
    sample_roles.check_sample_store(parsed)
    island_entry._require_ports_supported(parsed)


def test_one_learner_with_a_nesterov_horizon_still_merges_as_the_identity(tmp_path):
    _, plan = _plan(tmp_path, ONE, "--local-rl-rounds-per-sync", "8")
    flags = _syncer_flags(plan)
    assert flags["--learners"] == "1" and flags["--outer-lr"] == "1.0" and flags["--outer-momentum"] == "0.0"
    _, plan = _plan(tmp_path, TWO, "--local-rl-rounds-per-sync", "8")
    flags = _syncer_flags(plan)
    assert flags["--learners"] == "2" and flags["--outer-lr"] == "0.7" and flags["--outer-momentum"] == "0.9"


def test_the_no_sync_gpu_a_form_still_passes(tmp_path):
    args, plan = _plan(tmp_path, ONE, synced=False)
    assert not plan["outer_sync"] and sample_roles.buffer_full_parameter(args)
    assert not sample_roles.synced_full_parameter_trainer(args)


@pytest.mark.parametrize("extra, gpu, match", [
    # --tuning full behind a syncer without the full-parameter preset is the K3.1 whole-checkpoint
    # form (CONV) for --rl-model-recipe kimi_k31 only; the generic recipe is refused
    (("--rl-model-recipe", "generic"), TWO, "RL v0 requires --tuning lora"),
    (("--rl-sync-full-parameter-rounds", "sweep"), TWO, "need --rl-sync-full-parameter-rounds async"),
    # a trainer island has no engine to partition
    (("--rl-placement", "fixed-partition", "--rl-rollout-gpus", "8"), "modal:2x8xb300,modal:2x8xb300",
     "trainers start no engine"),
])
def test_synced_full_tuning_refusals(tmp_path, extra, gpu, match):
    selection_path = tmp_path / "expert_selection.json"
    _selection().write(selection_path)
    base = ("--tuning", "full", "--rl-model-recipe", "kimi_k31", "--rl-expert-selection", str(selection_path),
            "--expert-parallel", "8", "--tensor-parallel", "1", "--rl-sample-role", "trainer",
            "--rl-checkpoint-store", STORE, "--rl-max-policy-age", "2", "--rl-rollout-top-logprobs-num", "128",
            "--rl-algorithm-spec", str(_tis_spec(tmp_path)), "--cluster-prefix", "k31-synced")
    if "--rl-sync-full-parameter-rounds" in extra or "--rl-placement" in extra:
        base += ("--rl-sync-preset", FULL_PARAMETER_SYNC_PRESET, "--rl-parameter-layout-sha256", LAYOUT)
        if "--rl-placement" in extra:
            base += ("--rl-sync-full-parameter-rounds", "async")
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", (*base, *extra), gpu=gpu)
    args.controller = "local"
    with pytest.raises((ValueError, SystemExit), match=match):
        _prepare_rl_args(args)


def test_lora_synced_trainers_keep_their_rules():
    lora = SimpleNamespace(rl_sample_role="trainer", rl_single_island_no_sync=False, tuning="lora",
                           rl_sync_preset="strict-avg", rl_island_sample_role=None, rl_node_class=None)
    with pytest.raises(ValueError, match="at least two islands"):
        sample_roles.check_synced_trainers(lora, 1)
    sample_roles.check_synced_trainers(lora, 2)
    full_lora_preset = SimpleNamespace(**{**vars(lora), "tuning": "full"})
    with pytest.raises(ValueError, match="full-parameter pseudo-gradients"):
        sample_roles.check_synced_trainers(full_lora_preset, 2)
    assert not sample_roles.synced_full_parameter_trainer(full_lora_preset)
    full = SimpleNamespace(**{**vars(lora), "tuning": "full", "rl_sync_preset": FULL_PARAMETER_SYNC_PRESET,
                              "rl_sync_full_parameter_rounds": "async"})
    assert sample_roles.synced_full_parameter_trainer(full)
    sample_roles.check_synced_trainers(full, 1)
    sample_roles.check_synced_trainers(full, 2)


def test_ports_matrix_spares_a_trainer_island_the_rollout_gpu_rule():
    # a trainer-role island has no engine: a fixed partition needs no rollout GPUs on it
    assert ports_rejections(sync_preset=FULL_PARAMETER_SYNC_PRESET, tuning="full", parameter_mode="full",
                            placement="fixed-partition", rollout_num_gpus=None, sample_role="trainer") == []
    # every other island keeps the rule
    reasons = ports_rejections(sync_preset=FULL_PARAMETER_SYNC_PRESET, tuning="full", parameter_mode="full",
                               placement="fixed-partition", rollout_num_gpus=None, sample_role="sampler")
    assert reasons and "fixed partition needs --rollout-num-gpus >= 1" in reasons[0]
    assert ports_rejections(placement="fixed-partition", rollout_num_gpus=None)  # default role: both
    assert ports_rejections(placement="fixed-partition", rollout_num_gpus=2, sample_role="both") == []


# ------------------------------------------------------------------ 2. the Miles k31/b2 pin and helpers

def test_k31_lineage_registers_the_b2_full_parameter_transport():
    # the lineage head moved on to k31/b3-b7; B2's features carry forward
    assert pins.MILES_NEXT_COMMIT == pins.MILES_K31_FULL_D_COMMIT  # the k31 lineage head
    assert pins.pin_has("full_parameter_transport", pins.MILES_NEXT_COMMIT)
    assert pins.pin_has("full_parameter_transport", pins.MILES_K31_B2_COMMIT)
    assert pins.pin_has("full_parameter_rank_state", pins.MILES_K31_B2_COMMIT)
    assert not pins.pin_has("full_parameter_transport", pins.MILES_K31_F1_COMMIT)
    assert pins.pin_has("run_plugin", pins.MILES_K31_B2_COMMIT)  # F1 features carry forward
    assert pins.lineage_of(pins.MILES_K31_B2_COMMIT).name == "k31"
    from yeto.rl.adapters.miles import runtime_manifest as rm

    for interface in rm.CAPABILITY_REQUIREMENTS["Cut.full_parameter_transport"]:
        assert interface in rm.INTERFACES
    module, attribute = rm.INTERFACES["fork_b2_full_parameter_transport"]
    assert module == "miles.ray.full_parameter_transport" and attribute == "apply_chunked_cut"
    module, attribute = rm.INTERFACES["fork_b2_set_model_weight_version"]
    assert module == "miles.backends.megatron_utils.full_parameter_state"
    assert attribute == "set_model_weight_version"


class _Companion:
    def __init__(self, version: int) -> None:
        import torch

        self.weight_version = torch.tensor(version, dtype=torch.int64)

    def named_parameters(self):
        return [("model_companion.weight_version", self.weight_version)]


def _fake_miles_state(monkeypatch, *, companions: dict[int, list[_Companion]]):
    """A fake ``miles.backends.megatron_utils.full_parameter_state`` with the
    k31/b2 helper signatures yeto depends on."""
    import types

    from dataclasses import dataclass, field

    @dataclass
    class FullParameterRankState:
        policy_version: int | None = None
        apply_stats: dict | None = None
        committed_cut: object | None = None
        extras: dict = field(default_factory=dict)

    calls: list = []

    def full_parameter_rank_state(actor):
        state = getattr(actor, "full_parameter", None)
        if not isinstance(state, FullParameterRankState):
            state = FullParameterRankState()
            actor.full_parameter = state
        return state

    def model_weight_version(actor):
        mine = companions.get(id(actor), [])
        if not mine:
            return None
        versions = {int(c.weight_version.item()) for c in mine}
        assert len(versions) == 1
        return versions.pop()

    def set_model_weight_version(actor, version: int) -> int:
        calls.append(("set", version))
        current = model_weight_version(actor)
        if current is None:
            raise RuntimeError("the model has no companion; cannot set its weight version")
        if version < current:
            raise RuntimeError(f"weight version {version} is below the installed {current}")
        for c in companions[id(actor)]:
            c.weight_version.fill_(version)
        return version

    module = types.ModuleType("miles.backends.megatron_utils.full_parameter_state")
    module.FullParameterRankState = FullParameterRankState
    module.full_parameter_rank_state = full_parameter_rank_state
    module.model_weight_version = model_weight_version
    module.set_model_weight_version = set_model_weight_version
    for name in ("miles", "miles.backends", "miles.backends.megatron_utils"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "miles.backends.megatron_utils.full_parameter_state", module)
    # state_plugin.model_companions reads upstream's helper
    companion_module = types.ModuleType("miles.backends.training_utils.model_companion")
    companion_module._get_companions_of_model = lambda model: list(model.companions)
    monkeypatch.setitem(sys.modules, "miles.backends.training_utils", types.ModuleType("miles.backends.training_utils"))
    monkeypatch.setitem(sys.modules, "miles.backends.training_utils.model_companion", companion_module)
    return module, calls


def test_rank_state_helpers_read_the_b2_dataclass_and_fall_back_to_the_legacy_attributes(monkeypatch):
    from yeto.rl.adapters.miles import cut_plugin
    from yeto.rl.adapters.miles.full_parameter import rank_state, rank_stats

    actor = SimpleNamespace(model=SimpleNamespace(companions=[]))
    companions = {id(actor): [_Companion(3), _Companion(3)]}
    actor.model.companions = companions[id(actor)]
    module, calls = _fake_miles_state(monkeypatch, companions=companions)
    # per-rank state: the dataclass on actor.full_parameter, through the helper
    assert rank_state.rank_policy_version(actor) is None
    assert cut_plugin.rank_coords is not None  # module imported
    module.full_parameter_rank_state(actor).policy_version = 7
    module.full_parameter_rank_state(actor).apply_stats = {"commit": {"seconds": 1.5, "copy_seconds": 0.5}}
    assert rank_state.rank_policy_version(actor) == 7
    assert rank_stats.rank_apply_stats(actor) == {"commit": {"seconds": 1.5, "copy_seconds": 0.5}}
    # the published version: the companion, through model_weight_version / set_model_weight_version
    assert cut_plugin.miles_counters(actor) == {"weight_version": 3}
    cut_plugin.set_miles_counters(actor, {"weight_version": 5})
    assert calls == [("set", 5)] and cut_plugin.miles_counters(actor) == {"weight_version": 5}
    with pytest.raises(RuntimeError, match="below the installed"):
        cut_plugin.set_miles_counters(actor, {"weight_version": 4})
    # align_weight_version (the publisher's "stamp the policy version") routes to the same helper
    from yeto.rl.adapters.miles import state_plugin

    state_plugin.align_weight_version(actor, next_version=9)
    assert calls[-1] == ("set", 9) and cut_plugin.miles_counters(actor) == {"weight_version": 9}
    # a rank without a companion and without the legacy counter has no carrier
    bare = SimpleNamespace(model=SimpleNamespace(companions=[]))
    assert cut_plugin.miles_counters(bare) == {}
    with pytest.raises(cut_plugin.CutPluginError, match="no model companion"):
        cut_plugin.set_miles_counters(bare, {"weight_version": 1})


def test_rank_state_helpers_on_the_ports_v1_pin_read_the_private_attributes(monkeypatch):
    monkeypatch.setitem(sys.modules, "miles", None)  # no Miles importable: the legacy spellings
    from yeto.rl.adapters.miles import cut_plugin
    from yeto.rl.adapters.miles.full_parameter import rank_state, rank_stats

    actor = SimpleNamespace(_external_full_policy_version=4,
                            _external_full_parameter_apply_stats={"export": {"seconds": 2.0}},
                            weight_updater=SimpleNamespace(weight_version=6))
    assert rank_state.rank_policy_version(actor) == 4
    assert rank_stats.rank_apply_stats(actor) == {"export": {"seconds": 2.0}}
    assert cut_plugin.miles_counters(actor) == {"weight_version": 6}
    cut_plugin.set_miles_counters(actor, {"weight_version": 8})
    assert actor.weight_updater.weight_version == 8
    assert rank_state.rank_policy_version(SimpleNamespace()) is None


def test_megatron_fields_are_read_from_the_typed_backend_first():
    from yeto.rl.adapters.miles.cut_plugin import config_problems, megatron_field

    typed = SimpleNamespace(backend=SimpleNamespace(fp16=True, expert_model_parallel_size=8, use_distributed_optimizer=True,
                                                    tensor_model_parallel_size=2, pipeline_model_parallel_size=1,
                                                    context_parallel_size=1, num_experts=896,
                                                    use_precision_aware_optimizer=False,
                                                    num_distributed_optimizer_instances=1))
    assert megatron_field(typed, "fp16", False) is True and megatron_field(typed, "num_experts") == 896
    assert megatron_field(typed, "optimizer_cpu_offload", False) is False  # absent on both: the default
    flat = SimpleNamespace(fp16=True, expert_model_parallel_size=8)
    assert megatron_field(flat, "fp16", False) is True and megatron_field(flat, "expert_model_parallel_size", 1) == 8
    problems = config_problems(typed)
    assert any("--fp16" in p for p in problems)
    assert any("DP resharding with TP/PP>1" in p for p in config_problems(typed, reshard=True))
    assert any("EP>1" in p for p in config_problems(typed, reshard=True))
    assert config_problems(SimpleNamespace(backend=SimpleNamespace(fp16=False))) == []


def test_chunked_apply_passes_the_weight_version_only_to_a_miles_that_takes_it():
    from yeto.rl.adapters.miles.models.chunked_full_parameter import weight_version_apply_supported

    class NewMiles:
        async def apply_full_parameter_chunked_cut(self, cut, *, commit_token=None, max_commit_attempts=3,
                                                   skip_resident_fragments=False, verified_fragments=None,
                                                   weight_version=None):
            return 1

    class OldMiles:
        async def apply_full_parameter_chunked_cut(self, cut, *, commit_token=None, max_commit_attempts=3,
                                                   skip_resident_fragments=False, verified_fragments=None):
            return 1

    assert weight_version_apply_supported(NewMiles()) is True
    assert weight_version_apply_supported(OldMiles()) is False
    assert weight_version_apply_supported(SimpleNamespace()) is False


def test_fork_only_fields_cross_as_yeto_ext_on_a_config_that_does_not_declare_them():
    import pydantic

    from yeto.rl.adapters.miles.ext_args import WIRE_KEYS, ext_attr
    from yeto.rl.adapters.miles.island_args import IslandArgs, declares_field, override_fork_field

    class Config(pydantic.BaseModel):
        model_config = pydantic.ConfigDict(extra="forbid", frozen=True)
        yeto_ext: dict = {}
        wandb_project: str | None = None

    facade = IslandArgs(Config())
    assert declares_field(facade, "wandb_project") and not declares_field(facade, "external_policy_sync_run_until_stop")
    carried = override_fork_field(facade, external_policy_sync_run_until_stop=False, wandb_project="p")
    assert carried == {"external_policy_sync_run_until_stop": "ext", "wandb_project": "native"}
    assert facade.miles.wandb_project == "p"
    config = facade.freeze_for_slicing()
    assert "rl_external_policy_sync_run_until_stop" in WIRE_KEYS
    assert ext_attr(config, "rl_external_policy_sync_run_until_stop") is False
    # a mutable namespace (the ports-v1 pin) takes the field natively
    legacy = SimpleNamespace()
    assert override_fork_field(legacy, external_policy_identity_setter_path="x.y") == {
        "external_policy_identity_setter_path": "native"}
    assert legacy.external_policy_identity_setter_path == "x.y"


# ------------------------------------------------------------------ 3. the real Rust syncer

ROOT = Path(__file__).resolve().parent.parent
CARGO = shutil.which("cargo") or str(Path.home() / ".cargo/bin/cargo")
needs_cargo = pytest.mark.skipif(not Path(CARGO).exists(), reason="needs cargo to build the syncer")
P = 2  # fragments: the selected experts of the two MoE layers under test
EXPERTS_PER_FRAGMENT = 4
NUMEL = 3


@pytest.fixture(scope="module")
def syncer_binary() -> Path:
    subprocess.run([CARGO, "build", "-q"], cwd=ROOT / "syncer", check=True)
    return ROOT / "syncer/target/debug/yeto-syncer"


def _expert_names(fragment: int) -> list[str]:
    """The selected experts' tensors one fragment carries (partial experts: only
    the trainable experts of layer ``MOE_LAYERS[fragment]`` are in the pseudo-gradient)."""
    layer = MOE_LAYERS[fragment]
    return [f"actor.model.layers.{layer}.block_sparse_moe.experts.{e}.w1" for e in range(EXPERTS_PER_FRAGMENT)]


def _layout():
    from yeto.fragments import MERGE_AVG, Fragment, FragmentLayout

    return FragmentLayout([Fragment(MERGE_AVG, [(name, NUMEL) for name in _expert_names(f)]) for f in range(P)])


def _client(learner_id: int, port: int, profile_hash: str):
    from yeto.protocol import DTYPE_F32, SyncerClient

    return SyncerClient(("127.0.0.1", port), learner_id, _layout(), dtype=DTYPE_F32, num_streams=1,
                        connect_timeout=10, max_reconnects=0, finalization_timeout=30,
                        syncer_profile_hash=bytes.fromhex(profile_hash))


def _f32(values) -> bytes:
    return np.asarray(values, dtype=np.float32).tobytes()


def _initial(fid: int) -> np.ndarray:
    return np.asarray([float(fid) + 0.25 * i for i in range(NUMEL * EXPERTS_PER_FRAGMENT)], dtype=np.float32)


def _pseudo_gradient(learner: int, step: int) -> np.ndarray:
    """Learner ``learner``'s full-parameter pseudo-gradient of the round (its
    selected experts' trained minus base values)."""
    return np.asarray([0.5 + 0.125 * learner + 0.03125 * step + 0.001 * i
                       for i in range(NUMEL * EXPERTS_PER_FRAGMENT)], dtype=np.float32)


def _wait(predicate, what: str, timeout: float = 60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.005)
    raise TimeoutError(what)


def _start_syncer(binary: Path, tmp_path: Path, learners: int):
    """The launcher's own syncer command of the K3.1 synced plan (two or one
    learner), on a free port with a local checkpoint and tape."""
    from yeto import launcher

    gpu = TWO if learners == 2 else ONE
    args, plan = _plan(tmp_path, gpu, "--fragments", str(P), "--total-steps", "1",
                       "--rl-sync-final-cut", "download")
    command = launcher.role_syncer_commands(args, learners, binary=str(binary))["actor"]
    port = free_port()
    command = (command.replace(f"--port {launcher.SYNCER_PORT}", f"--port {port}")
               .replace(launcher.RL_SYNCER_CHECKPOINT, str(tmp_path / "state.ckpt"))
               .replace(launcher.RL_SYNCER_EVENT_TAPE, str(tmp_path / "tape.jsonl")))
    assert f"--learners {learners} --quorum 1" in command and "--outer-lr 1.0 --outer-momentum 0.0" in command
    log_file = tmp_path / "syncer.log"
    with log_file.open("w") as out:
        process = group_popen(["bash", "-c", command], stdout=out, stderr=subprocess.STDOUT, text=True)
    return process, port, launcher.rl_syncer_profile(args, learners).sha256, log_file


def _merge_through_the_syncer(binary: Path, tmp_path: Path, learners: int):
    """K fake trainer islands push their pseudo-gradients through the real syncer;
    returns (finals of learner 0, the rounds {step: {learner: delta}}, the merged
    broadcasts learner 0 saw)."""
    process, port, profile_hash, log_file = _start_syncer(binary, tmp_path, learners)
    clients = [_client(m, port, profile_hash) for m in range(learners)]
    seen: list[set] = [set() for _ in range(learners)]
    rounds: dict[int, dict[int, np.ndarray]] = {}
    broadcasts: dict[int, dict[int, np.ndarray]] = {}  # fragment -> version -> bytes learner 0 saw

    def poll():
        for learner, client in enumerate(clients):
            client.check_health()
            for update in client.drain_updates():
                if learner == 0:
                    broadcasts.setdefault(update.fragment_id, {})[update.version] = (
                        np.frombuffer(bytes(update.data), dtype=np.float32).copy())
            for permit in client.drain_pulls():
                key = (permit.global_step, permit.round_attempt)
                if key in seen[learner]:
                    continue
                seen[learner].add(key)
                delta = _pseudo_gradient(learner, permit.global_step)
                rounds.setdefault(permit.global_step, {})[learner] = delta
                client.push_fragment(permit.fragment_id, permit.global_step, permit.round_attempt,
                                     max(0, permit.global_step - P), permit.global_step, 1, 10, _f32(delta))
        return all(client.finalizing.is_set() for client in clients)

    try:
        for client in clients:
            client.start()
        for fid in range(P):
            clients[0].send_init(fid, _f32(_initial(fid)))
        _wait(poll, "syncer never finalized", timeout=90)
        finals = []
        for client in clients:
            manifest, fragments = client.wait_for_final_fragments(timeout=30)
            finals.append({f.fragment_id: np.frombuffer(bytes(f.data), dtype=np.float32)[-NUMEL * EXPERTS_PER_FRAGMENT:]
                           for f in fragments})
            client.acknowledge_finalization(manifest)
        assert process.wait(timeout=15) == 0, log_file.read_text()[-2000:]
    finally:
        for client in clients:
            client.close()
        kill_group(process)
        process.wait(timeout=10)
    for final in finals[1:]:
        for fid in range(P):
            assert np.array_equal(finals[0][fid], final[fid])  # one global cut on every island
    return finals[0], rounds, broadcasts


def _expected_merge(rounds: dict[int, dict[int, np.ndarray]]) -> dict[int, np.ndarray]:
    """theta <- theta + mean(pseudo-gradients) (outer lr 1, momentum 0; equal
    (c_steps, c_tokens) on every learner -> equal weights)."""
    params = {fid: _initial(fid) for fid in range(P)}
    for step in sorted(rounds):
        fid = (step - 1) % P
        deltas = [rounds[step][m] for m in sorted(rounds[step])]
        params[fid] = params[fid] + np.mean(np.stack(deltas), axis=0).astype(np.float32)
    return params


def _write_merged_release(directory: Path, merged: dict[int, np.ndarray]) -> None:
    """The publisher island's merged release: the selected experts' merged
    values (one file per fragment / MoE layer, as the K3.1 release plan names
    them) plus the config and index the sampler's loader reads."""
    directory.mkdir(parents=True, exist_ok=True)
    index = {}
    for fid in range(P):
        layer = MOE_LAYERS[fid]
        file = f"model-l{layer}-experts.safetensors"
        (directory / file).write_bytes(merged[fid].tobytes())
        for name in _expert_names(fid):
            index[name.removeprefix("actor.")] = file
    (directory / "config.json").write_text(json.dumps(_hf_config()))
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))


@needs_cargo
@pytest.mark.parametrize("learners", [2, 1])
def test_trainer_islands_merge_through_the_syncer_and_release_under_the_selection(syncer_binary, tmp_path, learners):
    finals, rounds, broadcasts = _merge_through_the_syncer(syncer_binary, tmp_path, learners)
    # every round closed with every learner's pseudo-gradient; one round per fragment
    assert sorted(rounds) == list(range(1, P + 1))
    assert all(sorted(rounds[step]) == list(range(learners)) for step in rounds)
    expected = _expected_merge(rounds)
    for fid in range(P):
        # the final cut every island downloads is the merged policy
        np.testing.assert_allclose(finals[fid], expected[fid], rtol=0, atol=1e-6)
        # every broadcast learner 0 saw of that fragment is one of its versions (the
        # initial cut or a merged round), never a foreign island's local values
        for version, data in broadcasts.get(fid, {}).items():
            values = data[-NUMEL * EXPERTS_PER_FRAGMENT:]
            assert np.allclose(values, _initial(fid)) or np.allclose(values, expected[fid], atol=1e-6), (fid, version)
    if learners == 1:
        # one learner: the merge is the identity (base + its own pseudo-gradient)
        for fid in range(P):
            np.testing.assert_allclose(finals[fid], _initial(fid) + rounds[fid + 1][0], rtol=0, atol=1e-6)
    else:
        # two learners: neither island's own result, the mean of both
        for fid in range(P):
            assert not np.array_equal(finals[fid], _initial(fid) + rounds[fid + 1][0])
            assert not np.array_equal(finals[fid], _initial(fid) + rounds[fid + 1][1])
    # the publisher (learner 0) releases the merged policy into the object store with the selection's sha
    bucket = Bucket(tmp_path)
    releases, _ = bucket.stores("trainer-0")
    selection = _selection()
    manifest = k31_release.release_manifest_fields(selection, layers=MOE_LAYERS)
    policy_hash = hashlib.sha256(b"".join(finals[fid].tobytes() for fid in range(P))).hexdigest()
    record = releases.release(1, policy_hash, lambda d: _write_merged_release(d, finals), manifest=manifest)
    assert record is not None and record.manifest["expert_selection_sha256"] == selection.sha256
    # the sampler node: pull LATEST, gate on the running selection, read the merged experts back
    sampler_releases, _ = bucket.stores("sampler")
    latest = sampler_releases.latest()
    assert latest is not None and latest.version == 1 and latest.manifest == manifest
    k31_release.check_release_manifest(latest.manifest, expected_selection_sha256=selection.sha256)
    other = selection.with_ep_size(4)
    with pytest.raises(ValueError, match="!= running selection"):
        k31_release.check_release_manifest(latest.manifest, expected_selection_sha256=other.sha256)
    with pytest.raises(ValueError, match="lacks expert_selection_sha256"):
        k31_release.check_release_manifest({}, expected_selection_sha256=selection.sha256)
    directory = sampler_releases.resolve(latest)
    for fid in range(P):
        stored = np.frombuffer((directory / f"model-l{MOE_LAYERS[fid]}-experts.safetensors").read_bytes(),
                               dtype=np.float32)
        np.testing.assert_array_equal(stored, finals[fid])
