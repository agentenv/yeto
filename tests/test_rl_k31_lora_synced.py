"""Kimi-K3.1 LoRA trainer islands behind the syncer (decoupled DiLoCo), on the CPU.

1. rules: attention targets on strict-avg only (``kimi_k31_lora.check_sync``);
2. launcher: two K3.1 LoRA trainer islands behind one syncer pass every launch rule;
   all-linear and decoupled are refused before a GPU starts;
3. island: ``_run_ports`` runs a kimi_k31 native export behind the syncer with a
   learned layout; Flash-Next stays no-sync only;
4. the native adapter-slots state (``native_slots_state``): pieces of TP shards and DP
   replicas assemble into each HF tensor exactly once, and apply cuts them back with
   the current moments; ``state_plugin`` dispatches a kimi_k31 actor there;
5. strict-avg sync sessions learn the layout from the first export
   (``bridges.learned_layout``), against the fake strict syncer and the real Rust syncer;
6. a merged LoRA release of the publisher (learner 0) reaches an A-card loader that
   merges it into the MXFP4 base.
"""

from __future__ import annotations

import dataclasses
import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_rl_k31_lora import _attention_adapter, _base, _engine as _loader_engine, _lora_launch
from test_rl_k31_synced_cpu import _launcher_args, _no_modal_listing, _fake_sky  # noqa: F401
from yeto.rl.adapters.miles import island_entry
from yeto.rl.adapters.miles import native_slots_state as nss
from yeto.rl.adapters.miles import state_plugin as sp
from yeto.rl.engine import kimi_k31_lora as kl


@pytest.fixture(autouse=True)
def _island_hmac_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k31-lora-synced-test-key")


# ------------------------------------------------------------------ 1. rules
def test_sync_rules():
    assert kl.check_tuning("lora", lora_targets="attention", policy_sync=True, sync_preset="strict-avg") == "lora"
    assert kl.check_tuning("lora", lora_targets="attention", policy_sync=True) == "lora"  # preset checked later
    with pytest.raises(kl.KimiK31LoraError, match="--lora-targets attention"):
        kl.check_tuning("lora", lora_targets="all-linear", policy_sync=True)
    with pytest.raises(kl.KimiK31LoraError, match="decoupled fragment plan"):
        kl.check_tuning("lora", lora_targets="attention", policy_sync=True, sync_preset="decoupled")
    # the full form keeps its own rules
    assert kl.check_tuning("full", policy_sync=True, sync_preset="full-parameter") == "full"


# ------------------------------------------------------------------ 2. launcher
GPU2 = "modal:1x8xb300,modal:1x8xb300"


def _synced_launch(tmp_path: Path, targets: str = "attention", *extra: str) -> tuple[str, ...]:
    launch = list(_lora_launch(tmp_path, targets, trainer_role=True))
    launch.remove("--rl-single-island-no-sync")
    i = launch.index("--lora-r")
    launch[i + 1] = "16"
    return (*launch, *extra)


def _plan(tmp_path: Path, launch: tuple[str, ...], gpu: str = GPU2):
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    args = _launcher_args("ports", launch, gpu=gpu)
    args.controller = "local"
    _prepare_rl_args(args)
    return args, dry_run_plan(args)


def _island_args(island: dict, learner: int, tmp_path: Path):
    tail = island["learner_command"].split("-m yeto.rl.adapters.miles.island_entry", 1)[1]
    tokens = [{"$LEARNER_ID": str(learner), "$SYNCER_ADDR": "10.0.0.1:29400"}.get(t, t) for t in shlex.split(tail)]
    tokens = [str(tmp_path / "algorithm_spec.json") if t.endswith("/algorithm_spec.json") else t for t in tokens]
    return island_entry.parse_args(tokens)


def test_two_k31_lora_trainer_islands_behind_one_syncer(tmp_path):
    args, plan = _plan(tmp_path, _synced_launch(tmp_path))
    assert plan["outer_sync"] is True and plan["total_gpus"] == 16
    (actor,) = plan["role_syncers"]
    flags = shlex.split(actor["command"])
    assert flags[flags.index("--learners") + 1] == "2"
    islands = plan["island_requests"]
    assert [r["learner_id"] for r in islands] == [0, 1]
    for learner, island in enumerate(islands):
        command = island["learner_command"]
        assert "--rl-single-island-no-sync" not in command and "--syncer $SYNCER_ADDR" in command
        assert "--rl-sample-lanes 2" in command and "--lora-targets attention" in command
        iargs = _island_args(island, learner, tmp_path)
        assert iargs.sync_preset == "strict-avg" and iargs.lora_r == 16
        # the island's own rule (island_entry.run_miles) passes for this island
        assert kl.check_tuning(iargs.parameter_mode, lora_targets=iargs.lora_targets,
                               policy_sync=not iargs.rl_single_island_no_sync,
                               sync_preset=iargs.sync_preset) == "lora"


def test_launcher_refuses_routed_expert_lora_and_decoupled_behind_the_syncer(tmp_path):
    with pytest.raises(ValueError, match="--lora-targets attention"):
        _plan(tmp_path, _synced_launch(tmp_path, "all-linear"))
    with pytest.raises(ValueError, match="decoupled fragment plan"):
        _plan(tmp_path, _synced_launch(tmp_path, "attention", "--rl-sync-preset", "decoupled",
                                       "--fragments", "2", "--local-rl-rounds-per-sync", "2"))
    # one island, no syncer: all-linear is allowed when its samplers can serve routed-expert LoRA
    launch = (*_lora_launch(tmp_path, "all-linear", trainer_role=True), "--rl-sampler-engine", "sglang")
    _plan(tmp_path, launch, gpu="modal:1x8xb300")
    with pytest.raises(ValueError, match="rl-sampler-engine"):
        _plan(tmp_path, _lora_launch(tmp_path, "all-linear", trainer_role=True), gpu="modal:1x8xb300")


# ------------------------------------------------------------------ 3. island
def test_run_ports_runs_the_k31_native_export_behind_the_syncer(monkeypatch):
    from yeto.rl.adapters.miles import entry

    seen = {}
    monkeypatch.setattr(entry, "run_ports_island", lambda m, l, a, **kw: seen.update(kw))
    monkeypatch.setattr(island_entry, "apply_ports_infra_switches", lambda *a: None)
    args = SimpleNamespace(lora_r=16, event_tape=None, learner_id=1, model_revision="r")
    miles_args = SimpleNamespace()
    island_entry._run_ports(args, miles_args, None, None, specs=(), canonical_targets=["attn"],
                            yeto_policy_sync=True, native_lora_export=kl.NATIVE_EXPORT)
    assert seen["layout_hash"] is None and seen["yeto_policy_sync"] is True
    assert miles_args.yeto_rl_native_lora_export == "kimi_k31"
    with pytest.raises(ValueError, match="without Yeto policy sync"):
        island_entry._run_ports(args, SimpleNamespace(), None, None, specs=(), canonical_targets=["q_proj"],
                                yeto_policy_sync=True, native_lora_export="qwen3_8_next")


# ------------------------------------------------------------------ 4. native adapter slots
def _spec(key, global_shape, offsets, local_shape, primary=True):
    return SimpleNamespace(key=key, global_shape=tuple(global_shape), offsets=tuple(offsets),
                           local_shape=tuple(local_shape), is_primary=primary)


QA = "language_model.model.layers.3.self_attn.q_a_proj.lora_A"
QB = "language_model.model.layers.3.self_attn.q_a_proj.lora_B"
OA = "language_model.model.layers.1.self_attn.o_proj.lora_A"


def _whole():
    torch.manual_seed(3)
    return {QA: torch.randn(4, 6), QB: torch.randn(5, 4), OA: torch.randn(4, 8)}


def _rank_states(whole):
    """TP2 x DP2: o_proj's A is split over TP along its input; q_a_proj replicated on TP;
    the DP replica (ranks 2, 3) holds the same pieces, not primary."""
    states = []
    for rank in range(4):
        tp, primary = rank % 2, rank < 2
        specs = [_spec(OA, (4, 8), (0, 4 * tp), (4, 4), primary)]
        if tp == 0:
            specs += [_spec(QA, (4, 6), (0, 0), (4, 6), primary), _spec(QB, (5, 4), (0, 0), (5, 4), primary)]
        weights = [whole[s.key][tuple(slice(o, o + n) for o, n in zip(s.offsets, s.local_shape))].clone()
                   for s in specs]
        states.append(SimpleNamespace(shard_specs=specs, weights=weights,
                                      exp_avgs=[torch.full_like(w, 0.5) for w in weights],
                                      exp_avg_sqs=[torch.full_like(w, 0.25) for w in weights], step=7))
    return states


def test_pieces_assemble_each_hf_tensor_exactly_once():
    whole = _whole()
    tensors = nss.assemble([nss.primary_pieces(s) for s in _rank_states(whole)])
    assert sorted(tensors) == sorted(nss.canonical_name(k) for k in whole)
    for key, value in whole.items():
        assert torch.equal(tensors[f"base_model.model.{key}.weight"], value)
    assert nss.hf_key(nss.canonical_name(QA)) == QA
    # a missing TP shard leaves a hole; a second primary copy overlaps
    states = _rank_states(whole)
    with pytest.raises(nss.NativeSlotsError, match="unset"):
        nss.assemble([nss.primary_pieces(states[0])])
    states[2].shard_specs[0].is_primary = True
    with pytest.raises(nss.NativeSlotsError, match="more than once"):
        nss.assemble([nss.primary_pieces(s) for s in states])
    with pytest.raises(nss.NativeSlotsError, match="no LoRA pieces"):
        nss.assemble([[], []])


class _Members:
    """Miles' DistributedOptimizerMembers of one rank: the state it exports and what it imports."""

    def __init__(self, state):
        self.state, self.imported, self.finished = state, None, False

    def export_adapter_state(self, slot):
        assert slot == 0
        return self.state

    def import_adapter_state(self, slot, state):
        self.imported = state

    def finish_import_adapter_state(self, slot):
        self.finished = True


def test_export_and_apply_through_the_members():
    whole = _whole()
    states = _rank_states(whole)
    members = [_Members(s) for s in states]
    pieces = [nss.primary_pieces(s) for s in states]
    exported = nss.export_tensors(None, is_main=True, members=lambda a, save_moments: members[0],
                                  gather=lambda local: pieces)
    assert torch.equal(exported[nss.canonical_name(OA)], whole[OA])
    assert nss.export_tensors(None, is_main=False, members=lambda a, save_moments: members[1],
                              gather=lambda local: pieces) is None
    new = {name: value + 1.0 for name, value in exported.items()}
    for rank, member in enumerate(members):
        count = nss.apply_tensors(None, new, members=lambda a, save_moments, m=member: m,
                                  adapter_state=lambda specs, weights, cur: SimpleNamespace(
                                      specs=specs, weights=weights, exp_avgs=cur.exp_avgs, step=cur.step))
        assert member.finished and count == len(states[rank].shard_specs)
        for spec, weight, old in zip(member.imported.specs, member.imported.weights, states[rank].weights):
            assert torch.equal(weight, old + 1.0)  # each rank gets exactly its own rectangle
        assert member.imported.step == 7 and float(member.imported.exp_avgs[0][0, 0]) == 0.5
    with pytest.raises(nss.NativeSlotsError, match="missing"):
        nss.apply_tensors(None, {nss.canonical_name(QA): new[nss.canonical_name(QA)]},
                          members=lambda a, save_moments: _Members(states[0]))
    with pytest.raises(nss.NativeSlotsError, match="not a canonical"):
        nss.apply_tensors(None, {**new, "model.x": torch.zeros(1)},
                          members=lambda a, save_moments: _Members(states[0]))


class _Adapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(4, 6))


def test_state_plugin_dispatches_a_kimi_k31_actor_to_the_adapter_slots(monkeypatch):
    whole = {QA: torch.randn(4, 6)}
    module = _Adapter()
    member = _Members(SimpleNamespace(shard_specs=[_spec(QA, (4, 6), (0, 0), (4, 6))], weights=[whole[QA]],
                                      exp_avgs=[torch.zeros(4, 6)], exp_avg_sqs=[torch.zeros(4, 6)], step=0))
    monkeypatch.setattr(nss, "miles_members", lambda actor, save_moments: member)
    monkeypatch.setattr(nss, "_adapter_state", lambda specs, weights, cur: SimpleNamespace(weights=weights))
    reset = []
    monkeypatch.setattr(sp, "reset_optimizer_moments", lambda opt: reset.append(opt))
    actor = SimpleNamespace(
        args=SimpleNamespace(yeto_rl_native_lora_export="kimi_k31", global_batch_size=4),
        model=[module], optimizer=object(), _is_first_replica_megatron_main_rank=True,
        opt_param_scheduler=SimpleNamespace(num_steps=0, step=lambda increment: None),
        adapter_slots=SimpleNamespace(slot_parameters=lambda slot: [module.a]))
    assert sp.is_native_slots(actor) and not sp.is_native_flash_next(actor)
    out = sp.export_state(actor, policy_version=3)
    assert out["policy_version"] == 3 and torch.equal(out["tensors"][nss.canonical_name(QA)], whole[QA])
    summary = sp.apply_state(actor, tensors={nss.canonical_name(QA): whole[QA] * 2}, policy_version=4,
                             local_step=0, optimizer="reset")
    assert summary["tensors"] == 1 and reset == [actor.optimizer] and member.finished
    assert torch.equal(member.imported.weights[0], whole[QA] * 2)
    with pytest.raises(sp.StatePluginError, match="finite FP32"):
        sp.apply_state(actor, tensors={nss.canonical_name(QA): whole[QA].half()}, policy_version=5,
                       local_step=0, optimizer="preserve")


# ------------------------------------------------------------------ 5. learned sync layout
def test_strict_avg_islands_learn_the_layout_from_the_first_export(tmp_path):
    from test_rl_engine_driver import NAME, _driver, _engine, _run_threads, _strict_config, _strict_syncer
    from yeto.rl.engine.bridges import StrictAvgSync, learned_layout

    engines = (_engine(torch.tensor([1.0, 3.0])), _engine(torch.tensor([3.0, 5.0])))
    syncer = _strict_syncer(engines[0], learners=2, rounds=2)
    drivers = []
    for i, engine in enumerate(engines):
        predicted = _strict_config(tmp_path, engine, learner_id=i, rounds=2, tape=f"island-{i}.jsonl")
        config = dataclasses.replace(predicted, expected_specs=(), layout_hash=None)
        sync = StrictAvgSync(config, client_factory=lambda _bridge, i=i: syncer.client(i))
        drivers.append(_driver(engine, sync, tmp_path, name=f"island-{i}.jsonl", learner_id=i))
    results, errors = _run_threads(drivers)
    assert errors == {}
    assert results[0].policy_tensor_hash() == results[1].policy_tensor_hash()
    for driver in drivers:
        assert driver.sync.config.layout_hash == engines[0].canonical(0).layout_hash
        assert driver.sync.config.expected_specs == engines[0].canonical(0).specs
    # a config that predicts its layout is kept as it is
    predicted = _strict_config(tmp_path, engines[0], learner_id=0, rounds=2)
    assert learned_layout(predicted, engines[1].canonical(0)) is predicted
    assert NAME in results[0].tensors


def test_two_learned_layout_islands_merge_through_the_real_syncer(tmp_path):
    from test_rl_engine_driver import _driver, _engine, _run_threads, _strict_config
    from test_rl_integration import _port, _start
    import subprocess

    from yeto.rl.engine.bridges import StrictAvgSync

    root = Path(__file__).resolve().parents[1]
    try:
        subprocess.run(["cargo", "build", "-q"], cwd=root / "syncer", check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("needs cargo to build the syncer")
    binary = root / "syncer/target/debug/yeto-syncer"
    port = _port()
    process = _start(binary, port, tmp_path / "state.ckpt", rounds=2)
    engines = (_engine(torch.tensor([1.0, 3.0])), _engine(torch.tensor([3.0, 5.0])))
    try:
        drivers = []
        for i, engine in enumerate(engines):
            config = dataclasses.replace(
                _strict_config(tmp_path, engine, learner_id=i, rounds=2, tape=f"island-{i}.jsonl"),
                syncer_addr=("127.0.0.1", port), expected_specs=(), layout_hash=None)
            drivers.append(_driver(engine, StrictAvgSync(config), tmp_path, name=f"island-{i}.jsonl",
                                   learner_id=i))
        results, errors = _run_threads(drivers)
        assert errors == {}
        assert results[0].policy_tensor_hash() == results[1].policy_tensor_hash()
        assert results[0].policy_version == 2
        assert process.wait(timeout=10) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


# ------------------------------------------------------------------ 6. merged release -> A card
def test_the_publishers_merged_lora_release_merges_on_the_a_card(tmp_path):
    import shutil

    from yeto.rl.adapters.miles.sampler_island import ReleasingPublisher
    from yeto.rl.engine.merged_release import MergedReleaseGuard
    from yeto.rl.engine.policy_release import LORA, PolicyFollower, PolicyReleaseStore
    from yeto.rl.engine.resume import CheckpointStore
    from yeto.rl.engine.sampler_checks import PortReleaseLoader, PostLoadChecks

    base = _base(tmp_path)
    adapter, _ = _attention_adapter(tmp_path / "trained")
    releases = PolicyReleaseStore(CheckpointStore(tmp_path / "store"))
    events = []
    manifest = kl.manifest_fields(rank=2, lora_targets="attention")
    publisher = ReleasingPublisher(
        SimpleNamespace(publish=lambda state, **kw: None), releases,
        lambda version, directory: shutil.copytree(adapter, directory, dirs_exist_ok=True),
        read_position=lambda: 0, emit=lambda event, **f: events.append((event, f)), weights=LORA,
        version_of=lambda state: state.merged, guard=MergedReleaseGuard(lambda p, v: None),
        origin=lambda state: {"kind": "merged", "learner_id": 0, "num_learners": 2,
                              "local_version": int(state.policy_version)},
        manifest=manifest)
    state = SimpleNamespace(policy_version=5, merged=2, policy_tensor_hash=lambda: "h2")
    publisher.publish(state)
    publisher.publish(SimpleNamespace(policy_version=6, merged=2, policy_tensor_hash=lambda: "h2"))  # not newer
    (record,) = releases.records()
    assert record.version == 2 and record.weights == LORA and record.origin["kind"] == "merged"
    assert kl.lora_manifest(record.manifest) == manifest["lora"]
    assert [e for e, _ in events].count("rl_policy_release_skipped") == 1
    served = []
    loader = PortReleaseLoader(_loader_engine(base), PostLoadChecks(), lambda *a, **k: None)
    loader._serve = lambda release, directory, version, first: served.append((version, Path(directory)))
    follower = PolicyFollower(releases, loader)
    follower.refresh()
    assert follower.version == 2
    ((version, directory),) = served
    assert version == 2 and directory.name.endswith(".merged")
    index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    assert any(name.endswith("q_a_proj.weight") and file == "release-lora-merged.safetensors"
               for name, file in index.items())
