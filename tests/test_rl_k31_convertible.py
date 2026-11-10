"""Kimi-K3.1 island role switch on CPU: two engine sampler islands (the N-card node
and an A-card node) write their shards of one object-store buffer; the N-card node
converts to a full-parameter trainer (the trainer reads every island's shard from
the watermark, releases with the selection manifest), the A-card sampler pulls the
release, and the N-card node comes back as a sampler on the released weights.
Fake engine, fake S3; nothing here starts Miles, SGLang or a GPU.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto.rl.adapters.miles import role_switch as miles_rs
from yeto.rl.adapters.miles.buffer_rollout import BatchReader
from yeto.rl.adapters.miles.models.kimi_k31 import release as k31_release
from yeto.rl.adapters.miles.models.partial_experts import selection as sel
from yeto.rl.engine import buffer_payload as bp
from yeto.rl.engine import engine_sampler_entry as entry
from yeto.rl.engine import role_switch as rs
from yeto.rl.engine.sample_buffer import FileSampleBuffer, merge_shards
from yeto.rl.engine.trainer_remote import LocalStoreFiles

from test_rl_k31_e2e_cpu import (
    EXAMPLES,
    MOE_LAYERS,
    Bucket,
    _filled,
    _release_dir,
    _spec,
    reward_fn,
)

N_NODE, A_NODE = "n-island-0", "a-mi355x-0"
BUCKET_URI = "s3://fake-bucket/_e2e/k31-conv"


# ------------------------------------------------------------------ fixtures
class MemFiles:
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}

    def read(self, rel):
        return self.data.get(rel)

    def write(self, rel, data):
        self.data[rel] = bytes(data)


class ObjectFiles:
    """``StoreFiles`` over the bucket's object API (what an s3:// island mounts)."""

    def __init__(self, store) -> None:
        self.objects = store.objects

    def read(self, rel):
        return self.objects.read(rel) if self.objects.exists(rel) else None

    def write(self, rel, data):
        self.objects.write(rel, bytes(data))


def _bucket(tmp_path: Path) -> Bucket:
    return Bucket(tmp_path, uri=BUCKET_URI)


def _stores(bucket: Bucket, name: str, node: str):
    """``(release store, buffer shard of node)`` as ``engine_sampler_entry.build_stores`` builds them."""
    from yeto.rl.engine.policy_release import PolicyReleaseStore
    from yeto.rl.engine.sample_buffer import shard_index

    store = bucket.store(name)
    return (PolicyReleaseStore(store, cache_dir=bucket.tmp / f"release-cache-{name}"),
            FileSampleBuffer(store, batch_size=1, index=shard_index(node)))


def _node_spec(tmp_path: Path, node: str, *, num_rollouts: int, **more) -> dict:
    spec = _spec(tmp_path, num_rollouts=num_rollouts, sampler_id=node, buffer_shards=[N_NODE, A_NODE],
                 event_tape=str(tmp_path / "tape" / node / "events.jsonl"), **more)
    spec["reward"] = f"{__name__}:reward_fn"
    return spec


def _run_sampler(spec: dict, bucket: Bucket, name: str, *, rounds: int | None = None, **kw) -> tuple[int, list]:
    """One sampler incarnation; ``rounds``: stop after this many rounds of this
    incarnation (an A-card node samples for the whole run: its spec's num_rollouts is
    never reached, the operator or the data end stops it)."""
    events: list = []
    done = lambda: rounds is not None and sum(1 for e, _ in events if e == "rl_sampler_engine_round") >= rounds  # noqa: E731
    ran = entry.run(spec, environ={}, resource_sampler=lambda emit: None, stop=done,
                    stores=lambda incarnation, emit: _stores(bucket, name, spec["sampler_id"]),
                    on_event=lambda event, **fields: events.append((event, fields)), **kw)
    return ran, events


def _trainer_buffer(bucket: Bucket, name: str) -> FileSampleBuffer:
    """The trainer's buffer: every island's shard merged into the index."""
    return FileSampleBuffer(bucket.store(name), batch_size=1, merge_shards=True, expected_shards=[N_NODE, A_NODE])


def _view(bucket: Bucket, name: str):
    buffer = _trainer_buffer(bucket, name)

    def view() -> rs.BufferView:
        buffer.refresh()
        end = len(buffer)
        return rs.BufferView(end=end, shards=buffer.shard_positions(end), shards_below=buffer.shard_positions)

    return view


TRAINER_LOG: list[dict] = []


def fake_trainer_phase(state: rs.RoleState, plan: rs.RoleSwitchPlan, context: rs.TrainerContext,
                       ) -> rs.TrainerPhaseReport:
    """The trainer island of the CPU rehearsal: read every group of every shard from
    the role record's watermark (``BatchReader``, policy age 2), select the experts
    from the routed-expert statistics of what it read, release the next version with
    the selection sha in the manifest and the consumed position as ``buffer_selected``."""
    bucket: Bucket = plan.trainer["bucket"]
    buffer = _trainer_buffer(bucket, f"trainer-{state.incarnation}")
    buffer.refresh()
    buffer.reset_to(int(state.consumed_watermark))
    pending = len(buffer) - int(state.consumed_watermark)
    reader = BatchReader(buffer, max_policy_age=2, wait_s=0.5, poll_s=0.01, sleep=lambda s: None, prefetch=False,
                         read_workers=1)
    taken, report = reader.take(state.incarnation, pending, current_version=int(state.policy_version))
    samples = [s for _, group in taken for s in group]
    counts: dict = {}
    for sample in samples:
        sel.counts_from_routed_experts(bp.rollout_routed_experts(sample), num_experts=896, moe_layer_offset=1,
                                       into=counts)
    selection = sel.ExpertSelection.from_counts(counts, model="kimi_k31", num_trainable=128, ep_size=8,
                                                source={"prompts_sha256": "0" * 64, "tokens": len(samples)})
    cursor = int(buffer.read_cursor)
    buffer.commit_consumed(cursor)
    version = int(state.policy_version) + 1
    release = context.releases.release(version, f"policy-hash-{version}",
                                       lambda d: _release_dir(d, changed=["model-l1-e0.safetensors"]),
                                       buffer_selected=cursor,
                                       manifest=k31_release.release_manifest_fields(selection, layers=MOE_LAYERS))
    TRAINER_LOG.append({"incarnation": state.incarnation, "from": state.consumed_watermark, "to": cursor,
                        "groups": report.groups, "stale": report.stale_groups,
                        "shards": buffer.shard_positions(cursor), "versions": sorted(set(report.behavior_versions)),
                        "selection": selection.sha256, "release": release.seq})
    return rs.TrainerPhaseReport(policy_version=version, released_seq=release.seq, release_manifest=release.manifest,
                                 consumed_watermark=cursor)


# ------------------------------------------------------------------ the flow
def test_two_sampler_islands_then_the_n_node_converts_to_trainer_and_back(tmp_path, monkeypatch):
    """A-card sampler: 3 rounds x 2 groups. N-card node: sampler until 4 groups are
    written, trainer over all 10 groups (both shards) from watermark 0, release v1,
    back to sampler on v1. Then the A-card sampler's next round pulls v1."""
    TRAINER_LOG.clear()
    bucket = _bucket(tmp_path)
    engines: list = []
    original = entry.build_engine
    monkeypatch.setattr(entry, "build_engine", lambda spec: engines.append(original(spec)) or engines[-1])

    # (1) both islands start as samplers on the same buffer, each its own shard
    a_rounds, a_events = _run_sampler(_node_spec(tmp_path, A_NODE, num_rollouts=64), bucket, "a", rounds=3)
    assert a_rounds == 3 and engines[-1].stopped
    n_spec = _node_spec(tmp_path, N_NODE, num_rollouts=64)
    n_spec["role_switch"] = {"node": N_NODE, "after_groups": 4, "then_sampler": True,
                             "trainer": {"run": f"{__name__}:fake_trainer_phase", "bucket": bucket}}
    plan = rs.RoleSwitchPlan.from_spec(n_spec)
    assert plan.after_groups == 4 and plan.then_sampler and plan.marker and plan.store_marker
    files = ObjectFiles(bucket.store("files"))
    releases_n, _ = bucket.stores("n-releases")
    tape: list = []
    island = rs.ConvertibleIsland(
        n_spec, plan, files=files, releases=releases_n, buffer_view=_view(bucket, "view"),
        run_sampler=lambda spec, *, stop, on_event: entry.run(
            spec, environ={}, resource_sampler=lambda emit: None, stop=stop, on_event=on_event,
            stores=lambda incarnation, emit: _stores(bucket, f"n-{incarnation}", N_NODE)),
        trainer=fake_trainer_phase, environ={rs.SWITCH_MARKER_ENV: str(tmp_path / "switch-marker")},
        device_idle=rs.DeviceIdle(probe=lambda: [512, 0]), emit=lambda event, **f: tape.append((event, f)))
    states = island.run()

    # (2) sampler phase: 2 rounds (4 groups >= after_groups), the K3.1 fields on every group,
    #     the engine stopped by the entry, the device checked free
    sampler_start, to_trainer, trained, back = states[0], states[1], states[2], states[3]
    assert [s.role for s in states] == ["sampler", "trainer", "trainer", "sampler", "sampler"]
    assert [s.incarnation for s in states] == [1, 2, 2, 3, 3]
    assert to_trainer.reason == "after_groups:4" and to_trainer.sampler_rounds == 2 and to_trainer.sampler_groups == 4
    assert engines[1].stopped  # the N node's SGLang stand-in was stopped before the trainer phase
    phase = [f for e, f in tape if e == rs.PHASE_EVENT and f["role"] == "sampler"][0]
    assert phase["device"] == {"checked": True, "used_mib": [512, 0]}
    # the merged buffer: 6 groups of A, 4 of N; nothing consumed yet (no cut, first trainer)
    assert to_trainer.buffer_end == 10 and to_trainer.shard_ends == {A_NODE: 6, N_NODE: 4}
    assert to_trainer.consumed_watermark == 0 and to_trainer.shard_watermarks == {}
    # the N node's sampler-phase groups (v0; the phase after the trainer writes v1 ones)
    n_groups = [bp.decode_group(buffer.read_payload(e.payload_ref)) for buffer in [_trainer_buffer(bucket, "check")]
                for e in buffer.entries() if e.sample_id.startswith(N_NODE + "-") and e.behavior_version == 0]
    assert len(n_groups) == 4
    for group in n_groups:
        for sample in group:
            assert bp.rollout_routed_experts(sample).shape[1:] == (3, 16)
            assert sample[bp.ROLLOUT_TOP_LOGPROBS_IDS].shape[1] == 128
            assert sample["metadata"]["engine"]["version_fields"]["fp32_lm_head"] is True

    # (3) trainer phase: every group of both shards from watermark 0 to 10, release v1 with the
    #     selection sha; the role record carries the watermark per shard
    assert TRAINER_LOG == [{**TRAINER_LOG[0], "from": 0, "to": 10, "groups": 10, "stale": 0,
                            "shards": {A_NODE: 6, N_NODE: 4}, "versions": [0], "release": 1}]
    assert trained.reason == "released" and trained.policy_version == 1 and trained.released_seq == 1
    assert trained.consumed_watermark == 10 and trained.shard_watermarks == {A_NODE: 6, N_NODE: 4}
    assert trained.release_manifest["expert_selection_sha256"] == TRAINER_LOG[0]["selection"]
    assert json.loads(files.read(rs.role_path(N_NODE)))["policy_version"] == 1  # the record on the store
    switches = [f for e, f in tape if e == rs.SWITCH_EVENT]
    assert [s["direction"] for s in switches] == ["to_trainer", "to_sampler"]
    assert switches[1]["policy_version"] == 1 and switches[1]["base_dir"]

    # (4) back to sampler: the engine starts on the release directory at v1 and writes v1 groups
    assert engines[2].loads[0][1] == 1 and engines[2].loads[0][0] == switches[1]["base_dir"]
    assert back.role == "sampler" and back.policy_version == 1 and back.reason == "start"
    final = states[4]
    assert final.reason == "finished" and final.policy_version == 1 and final.sampler_groups > 0
    buffer = _trainer_buffer(bucket, "after")
    buffer.refresh()
    versions: dict = {N_NODE: set(), A_NODE: set()}
    for e in buffer.entries():
        versions[N_NODE if e.sample_id.startswith(N_NODE + "-") else A_NODE].add(e.behavior_version)
    assert versions[N_NODE] == {0, 1} and versions[A_NODE] == {0}

    # (5) the A-card sampler keeps generating and pulls the N node's release (selection gate passes)
    a_spec = _node_spec(tmp_path, A_NODE, num_rollouts=64, selection_sha256=TRAINER_LOG[0]["selection"])
    a_rounds, a_events = _run_sampler(a_spec, bucket, "a-2", rounds=1)
    loaded = [f for e, f in a_events if e == "rl_sampler_policy_loaded"]
    assert a_rounds == 1 and [(f["seq"], f["policy_version"]) for f in loaded] == [(1, 1)]
    assert [f for e, f in a_events if e == "rl_sampler_engine_round"][0]["policy_version"] == 1


def test_a_second_trainer_phase_continues_from_the_cut_watermark(tmp_path):
    """The watermark the trainer starts from is the newest cut's, never ahead of the
    buffer end; the role record's per-shard form follows the merged index."""
    bucket = _bucket(tmp_path)
    _run_sampler(_node_spec(tmp_path, A_NODE, num_rollouts=64), bucket, "a", rounds=2)
    _run_sampler(_node_spec(tmp_path, N_NODE, num_rollouts=64), bucket, "n", rounds=1)
    files = MemFiles()
    # a cut from an earlier trainer phase consumed 3 of the merged positions
    files.write("LATEST", json.dumps({"seq": 1}).encode())
    from yeto.rl.engine import recovery

    pointer = {"next_rollout_id": 1, recovery.CUT_KEY: recovery.ConsistentCut(
        checkpoint_id="cut-1", trigger="scheduled", buffer_watermark=3).to_record()}
    files.write("state/00000001/round-cut.json", json.dumps(pointer).encode())
    spec = {**_node_spec(tmp_path, N_NODE, num_rollouts=1),
            "role_switch": {"node": N_NODE, "after_groups": 1, "trainer": {"run": "x:y"}}}
    island = rs.ConvertibleIsland(spec, rs.RoleSwitchPlan.from_spec(spec), files=files, releases=None,
                                  buffer_view=_view(bucket, "view"), run_sampler=lambda *a, **k: 0,
                                  trainer=lambda *a: rs.TrainerPhaseReport(policy_version=1))
    assert island.incarnation == 0
    report = rs.SamplerPhaseReport(rounds=1, groups=2, buffer_end=6, shard_ends={A_NODE: 4, N_NODE: 2},
                                   policy_version=0, reason="after_groups:1", seconds=0.1)
    state = island.switch_to_trainer(report)
    assert state.consumed_watermark == 3 and sum(state.shard_watermarks.values()) == 3
    assert state.incarnation == 1 and rs.read_role_state(files, N_NODE) == state
    # a trainer phase that reports no watermark keeps the record's
    done = island.trainer_phase(state)
    assert done.policy_version == 1 and island.states[-1].consumed_watermark == 3
    # the restarted container continues the incarnation count from the record
    again = rs.ConvertibleIsland(spec, rs.RoleSwitchPlan.from_spec(spec), files=files, releases=None,
                                 buffer_view=_view(bucket, "view-2"), run_sampler=lambda *a, **k: 0,
                                 trainer=lambda *a: rs.TrainerPhaseReport(policy_version=1))
    assert again.incarnation == 1


# ------------------------------------------------------------------ triggers
def test_switch_trigger_fires_on_time_groups_local_marker_or_store_marker(tmp_path):
    clock = [100.0]
    groups = [0]
    files = MemFiles()
    marker = tmp_path / "marker"
    trigger = rs.SwitchTrigger(after_seconds=60, after_groups=3, marker=marker, files=files,
                               store_marker=rs.switch_marker_path(N_NODE), groups=lambda: groups[0],
                               clock=lambda: clock[0]).start()
    assert not trigger() and trigger.reason is None
    groups[0] = 3
    assert trigger() and trigger.reason == "after_groups:3"
    assert trigger()  # latched
    for kw, act, reason in (
            ({"after_seconds": 60}, lambda: clock.__setitem__(0, 161.0), "after_seconds:60"),
            ({"marker": marker}, lambda: marker.write_text("{}"), f"marker:{marker}"),
            ({"files": files, "store_marker": rs.switch_marker_path(N_NODE)},
             lambda: rs.write_switch_marker(files, N_NODE, clock=lambda: 1.0), f"store_marker:{rs.switch_marker_path(N_NODE)}")):
        one = rs.SwitchTrigger(clock=lambda: clock[0], **kw).start()
        assert not one()
        act()
        assert one() and one.reason == reason
    assert json.loads(files.read(rs.switch_marker_path(N_NODE))) == {"to": "trainer", "by": "operator", "time": 1.0}
    with pytest.raises(rs.RoleSwitchError, match="at least one trigger"):
        rs.SwitchTrigger()
    with pytest.raises(rs.RoleSwitchError, match="after_groups"):
        rs.SwitchTrigger(after_groups=0)
    with pytest.raises(rs.RoleSwitchError, match="store files"):
        rs.SwitchTrigger(store_marker="x")


def test_role_switch_plan_rejects_incomplete_sections():
    base = {"sampler_id": N_NODE}
    with pytest.raises(rs.RoleSwitchError, match="no role_switch section"):
        rs.RoleSwitchPlan.from_spec(base)
    with pytest.raises(rs.RoleSwitchError, match="trainer needs"):
        rs.RoleSwitchPlan.from_spec({**base, "role_switch": {"after_groups": 1}})
    with pytest.raises(rs.RoleSwitchError, match="unknown keys"):
        rs.RoleSwitchPlan.from_spec({**base, "role_switch": {"trainer": {"run": "a:b"}, "bogus": 1}})
    with pytest.raises(rs.RoleSwitchError, match="needs a trigger"):
        rs.RoleSwitchPlan.from_spec({**base, "role_switch": {"trainer": {"run": "a:b"}, "marker": False,
                                                              "store_marker": False}})
    plan = rs.RoleSwitchPlan.from_spec({**base, "role_switch": {"trainer": {"command": ["python", "-m", "x"]},
                                                                "after_seconds": 1800, "device_idle": False}})
    assert plan.node == N_NODE and plan.device_idle is None and plan.after_seconds == 1800.0 and plan.max_cycles == 1


def test_device_idle_waits_for_the_memory_and_passes_without_a_probe():
    clock = [0.0]
    assert rs.DeviceIdle(probe=lambda: None).wait() == {"checked": False}
    assert rs.DeviceIdle(probe=lambda: [900, 10], used_mib=1024).wait() == {"checked": True, "used_mib": [900, 10]}
    readings = iter([[50_000, 10], [30_000, 10], [100, 10]])
    idle = rs.DeviceIdle(probe=lambda: next(readings), timeout_s=60, poll_s=5, clock=lambda: clock[0],
                         sleep=lambda s: clock.__setitem__(0, clock[0] + s))
    assert idle.wait()["used_mib"] == [100, 10] and clock[0] == 10.0
    stuck = rs.DeviceIdle(probe=lambda: [50_000], timeout_s=60, poll_s=5, clock=lambda: clock[0],
                          sleep=lambda s: clock.__setitem__(0, clock[0] + s))
    with pytest.raises(rs.RoleSwitchError, match="still held"):
        stuck.wait()
    assert rs.nvidia_smi_used_mib(run=lambda *a, **k: SimpleNamespace(returncode=0, stdout="1024\n 0 \n")) == [1024, 0]
    assert rs.nvidia_smi_used_mib(run=lambda *a, **k: (_ for _ in ()).throw(OSError())) is None


def test_sampler_spec_on_release_takes_the_newest_release_and_gates_the_selection(tmp_path):
    bucket = _bucket(tmp_path)
    releases, _ = bucket.stores("t")
    spec = {"sampler_id": N_NODE, "expert_selection_sha256": "a" * 64}
    assert rs.sampler_spec_on_release(spec, releases) == spec  # no release yet
    releases.release(1, "h1", lambda d: _release_dir(d), manifest={"expert_selection_sha256": "a" * 64})
    on_v1 = rs.sampler_spec_on_release(spec, releases)
    assert on_v1["base_version"] == 1 and Path(on_v1["base_dir"]).is_dir()
    releases.release(2, "h2", lambda d: _release_dir(d), manifest={"expert_selection_sha256": "b" * 64})
    with pytest.raises(rs.RoleSwitchError, match="carries selection"):
        rs.sampler_spec_on_release(spec, releases)
    assert rs.sampler_spec_on_release({"sampler_id": N_NODE}, releases)["base_version"] == 2  # no gate


def _index(directory: Path, weight_map: dict) -> None:
    import json as _json

    for name in set(weight_map.values()):
        (directory / name).write_bytes(b"x")
    (directory / "model.safetensors.index.json").write_text(_json.dumps({"metadata": {}, "weight_map": weight_map}))


def test_a_partial_experts_release_is_served_over_the_base_checkpoint(tmp_path):
    import json as _json

    base = tmp_path / "base"
    base.mkdir()
    expert = "language_model.model.layers.1.block_sparse_moe.experts.7.w1.weight_packed"
    _index(base, {expert: "model-1.safetensors", "language_model.model.embed_tokens.weight": "model-0.safetensors"})
    (base / "config.json").write_text("{}")
    bucket = _bucket(tmp_path)
    releases, _ = bucket.stores("t")
    spec = {"sampler_id": N_NODE, "engine": {"server": {"base_dir": str(base)}}}

    def partial(directory):
        _index(Path(directory), {expert: "model-00001-of-00001.safetensors"})

    releases.release(1, "h1", partial, manifest={"partial_experts": {"num_trainable": 128}})
    on_v1 = rs.sampler_spec_on_release(spec, releases)
    served = Path(on_v1["base_dir"])
    weight_map = _json.loads((served / "model.safetensors.index.json").read_text())["weight_map"]
    assert weight_map[expert] == "release-model-00001-of-00001.safetensors"
    assert weight_map["language_model.model.embed_tokens.weight"] == "model-0.safetensors"
    assert (served / "config.json").resolve() == (base / "config.json").resolve()
    assert (served / "release-model-00001-of-00001.safetensors").is_file()
    bare = rs.sampler_spec_on_release({"sampler_id": N_NODE}, releases)  # no server checkpoint: as is
    assert not bare["base_dir"].endswith(".overlay")


# ------------------------------------------------------------------ adapter trainer
def test_subprocess_trainer_runs_the_island_command_and_reads_the_release(tmp_path):
    bucket = _bucket(tmp_path)
    releases, _ = bucket.stores("t")
    state = rs.RoleState(node=N_NODE, role="trainer", incarnation=2, at=0.0, buffer_end=10,
                         shard_ends={A_NODE: 6, N_NODE: 4}, consumed_watermark=0, policy_version=0)
    command = ["python", "-m", "yeto.rl.adapters.miles.island_entry", "--rl-sample-role", "trainer",
               "--rl-single-island-no-sync", "--tuning", "full"]
    plan = rs.RoleSwitchPlan.from_spec({"sampler_id": N_NODE, "role_switch": {
        "after_seconds": 10, "trainer": {"command": command, "env": {"EXTRA": "1"}}}})
    context = rs.TrainerContext(files=MemFiles(), releases=releases, buffer=lambda: rs.BufferView(end=10),
                                environ={"PATH": "/bin"})
    calls: list = []

    def run(argv, *, env, check):
        calls.append((list(argv), dict(env)))
        releases.release(1, "h1", lambda d: _release_dir(d), buffer_selected=10,
                         manifest={"expert_selection_sha256": "c" * 64})
        return SimpleNamespace(returncode=0)

    report = miles_rs.subprocess_trainer(state, plan, context, run=run)
    assert report.policy_version == 1 and report.released_seq == 1 and report.consumed_watermark == 10
    assert report.release_manifest == {"expert_selection_sha256": "c" * 64}
    argv, env = calls[0]
    assert argv == command and env["EXTRA"] == "1" and env["PATH"] == "/bin"
    assert (env[miles_rs.NODE_ENV], env[miles_rs.INCARNATION_ENV], env[miles_rs.WATERMARK_ENV]) == (N_NODE, "2", "0")
    with pytest.raises(rs.RoleSwitchError, match="without a new release"):
        miles_rs.subprocess_trainer(state, plan, context, run=lambda *a, **k: SimpleNamespace(returncode=0))
    with pytest.raises(rs.RoleSwitchError, match="exited with code 3"):
        miles_rs.subprocess_trainer(state, plan, context, run=lambda *a, **k: SimpleNamespace(returncode=3))
    with pytest.raises(rs.RoleSwitchError, match="--rl-sample-role trainer"):
        miles_rs.check_trainer_command(["python", "--rl-sample-role", "sampler"])
    with pytest.raises(rs.RoleSwitchError, match="argv list"):
        miles_rs.check_trainer_command("python -m x")


# ------------------------------------------------------------------ relaxed rules
def test_full_parameter_is_accepted_where_only_lora_was(tmp_path):
    """config.py (convertible on a colocated island), sample_roles (convertible pool,
    synced trainers) take --tuning full; the generic synced trainer still needs LoRA."""
    from yeto.rl.adapters.miles import config as mc
    from yeto.rl.engine import sample_roles
    from yeto.rl.engine.algorithm import AlgorithmSpec, CorrectionSpec, ExecutionSpec

    from test_rl_miles_adapter_config import make_config, sub

    spec = AlgorithmSpec(execution=ExecutionSpec(max_policy_staleness=1),
                         correction=CorrectionSpec(method="tis", tis_clip=2.0, tis_clip_low=0.0))
    cfg = dataclasses.replace(make_config(), sample_buffer=sample_roles.SampleBufferConfig(
        role="convertible", store="/mnt/store"))
    full = sub(cfg, "trainable", parameter_mode="full", target_modules=())
    argv = list(mc.translate_run_config(full, spec, max_policy_age=1, full_parameter_sync=False).argv)
    assert "--colocate" in argv and "--lora-rank" not in argv
    with pytest.raises(mc.MilesConfigError, match="on a colocated island"):
        mc.translate_run_config(dataclasses.replace(make_config(colocated=False), sample_buffer=full.sample_buffer),
                                spec, max_policy_age=1, full_parameter_sync=False)
    synced = SimpleNamespace(rl_sample_role="trainer", rl_single_island_no_sync=False, tuning="full",
                             rl_model_recipe="kimi_k31", rl_node_class=None, rl_sync_preset=None,
                             rl_island_sample_role=None)
    sample_roles.check_synced_trainers(synced, 2)
    with pytest.raises(ValueError, match="need --rl-model-recipe kimi_k31"):
        sample_roles.check_synced_trainers(dataclasses.replace(synced, rl_model_recipe="generic")
                                           if dataclasses.is_dataclass(synced) else
                                           SimpleNamespace(**{**vars(synced), "rl_model_recipe": "generic"}), 2)
    with pytest.raises(ValueError, match="--tuning lora or --tuning full"):
        sample_roles.check_synced_trainers(SimpleNamespace(**{**vars(synced), "tuning": "qlora"}), 2)


def test_default_grpo_golden_is_unchanged():
    out = subprocess.run([__import__("sys").executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
                          str(Path(__file__).with_name("test_decoupling_golden.py"))],
                         capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]),
                         env={**__import__("os").environ, "PYTHONPATH": ".:tests"})
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]


# ------------------------------------------------------------------ examples
def test_example_convertible_spec_runs_and_the_launch_draft_parses(tmp_path):
    TRAINER_LOG.clear()
    subprocess.run(["bash", "-n", str(EXAMPLES / "launch-convertible-modal.sh")], check=True)
    text = (EXAMPLES / "launch-convertible-modal.sh").read_text()
    assert "r2.cloudflarestorage.com/" not in text
    bucket = os.environ.get("YETO_E2E_R2_BUCKET")  # the real bucket name: never in an example
    assert not bucket or bucket not in text
    assert "yeto.rl.engine.role_switch --spec" in text and "--rl-sample-role trainer" in text
    spec = _filled(json.loads((EXAMPLES / "convertible-spec.cpu-fake.json").read_text()), tmp_path)
    Path(spec["prompts"]["path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(spec["prompts"]["path"]).write_text("".join(json.dumps({"prompt": f"{i}+{i}", "answer": str(2 * i)}) + "\n"
                                                     for i in range(4)))
    spec.update(reward=f"{__name__}:reward_fn", reward_kind="neutral")
    plan = rs.RoleSwitchPlan.from_spec(spec)
    assert plan.after_groups == 4 and plan.after_seconds == 1800 and plan.then_sampler
    assert rs._import(plan.trainer["run"]) is fake_trainer_phase
    bucket = _bucket(tmp_path)
    spec["role_switch"]["trainer"]["bucket"] = bucket
    island = rs.ConvertibleIsland(
        spec, rs.RoleSwitchPlan.from_spec(spec), files=MemFiles(), releases=bucket.stores("r")[0],
        buffer_view=_view(bucket, "view"),
        run_sampler=lambda s, *, stop, on_event: entry.run(s, environ={}, resource_sampler=lambda e: None, stop=stop,
                                                          on_event=on_event,
                                                          stores=lambda inc, emit: _stores(bucket, f"n-{inc}", N_NODE)),
        trainer=fake_trainer_phase)
    states = island.run()
    assert [s.role for s in states] == ["sampler", "trainer", "trainer", "sampler", "sampler"]
    assert TRAINER_LOG[0]["groups"] == 4 and states[-1].policy_version == 1


def test_the_follower_serves_a_partial_release_over_the_base_and_keeps_the_one_it_started_on(tmp_path):
    """GPU-A it11b: after a trainer phase the sampler started on the v2 overlay, then its
    follower reloaded v2 from the bare partial release (no attention / dense / frozen
    weights: SGLang refused it three times, SamplerLoadFatal)."""
    import json as _json

    from yeto.rl.engine import engine_sampler as es
    from yeto.rl.engine.sampler_checks import PortReleaseLoader, PostLoadChecks

    base = tmp_path / "base"
    base.mkdir()
    expert = "language_model.model.layers.1.block_sparse_moe.experts.7.w1.weight_packed"
    _index(base, {expert: "model-1.safetensors", "language_model.model.embed_tokens.weight": "model-0.safetensors"})
    bucket = _bucket(tmp_path)
    releases, _ = bucket.stores("t")
    releases.release(2, "h2", lambda d: _index(Path(d), {expert: "model-00001-of-00001.safetensors"}),
                     manifest={"partial_experts": {"num_trainable": 128}})
    latest = releases.latest()
    engine = SimpleNamespace(config=SimpleNamespace(base_dir=base), identity=SimpleNamespace(quant_scheme="bf16"))
    loader = PortReleaseLoader(engine, PostLoadChecks(), lambda *a, **k: None)
    served = []
    loader._serve = lambda release, directory, version, first: served.append(Path(directory))
    loader(latest, Path(releases.resolve(latest)))
    weight_map = _json.loads((served[0] / "model.safetensors.index.json").read_text())["weight_map"]
    assert served[0].name.endswith(".overlay")
    assert weight_map["language_model.model.embed_tokens.weight"] == "model-0.safetensors"
    assert weight_map[expert] == "release-model-00001-of-00001.safetensors"

    # the follower of a sampler that started on v2 does not load v2 again
    assert es.started_on_release(releases, 2).seq == latest.seq
    assert es.started_on_release(releases, 0) is None and es.started_on_release(releases, 1) is None
    follower = es.PolicyFollower(releases, lambda r, d: served.append(d), version=2)
    follower.release = es.started_on_release(releases, 2)
    assert follower.refresh() is None and len(served) == 1


def test_an_mxfp4_sampler_serves_a_partial_release_in_its_base_scheme(tmp_path):
    """A K3.1 trainer's partial-experts release has no ``derived_from`` (release_scheme says
    bf16), but its experts are the base checkpoint's MXFP4 tensors: an MXFP4_EXPERTS_RTN
    engine serves it over the base. A partial release with another quantization config, or a
    plain trained (bf16) release, is still refused (scheme_mismatch)."""
    import json as _json

    from yeto.rl.engine.policy_release import ReleaseNotLoaded
    from yeto.rl.engine.sampler_checks import PortReleaseLoader, PostLoadChecks

    # nested as in Kimi-K3.1-Pro's config.json (text_config.quantization_config)
    mxfp4 = {"text_config": {"quantization_config": {"quant_method": "compressed-tensors",
                                                     "format": "mxfp4-pack-quantized"}}}
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(_json.dumps(mxfp4))
    expert = "language_model.model.layers.1.block_sparse_moe.experts.7.w1.weight_packed"
    _index(base, {expert: "model-1.safetensors", "language_model.model.embed_tokens.weight": "model-0.safetensors"})
    bucket = _bucket(tmp_path)
    releases, _ = bucket.stores("t")

    def write(config):
        def files(d):
            _index(Path(d), {expert: "model-00001-of-00001.safetensors"})
            (Path(d) / "config.json").write_text(_json.dumps(config))
        return files

    engine = SimpleNamespace(config=SimpleNamespace(base_dir=base),
                             identity=SimpleNamespace(quant_scheme="MXFP4_EXPERTS_RTN"),
                             served_version=lambda: 1)
    loader = PortReleaseLoader(engine, PostLoadChecks(), lambda *a, **k: None)
    served = []
    loader._serve = lambda release, directory, version, first: served.append(Path(directory))

    releases.release(1, "h1", write(mxfp4), manifest={"partial_experts": {"num_trainable": 32}})
    first = releases.latest()
    loader(first, Path(releases.resolve(first)))
    assert len(served) == 1 and served[0].name.endswith(".overlay")

    releases.release(2, "h2", write({}), manifest={"partial_experts": {"num_trainable": 32}})
    with pytest.raises(ReleaseNotLoaded) as info:
        loader(releases.latest(), Path(releases.resolve(releases.latest())))
    assert info.value.reason == "scheme_mismatch"

    releases.release(3, "h3", write(mxfp4))  # trained weights, not a partial release
    with pytest.raises(ReleaseNotLoaded) as info:
        loader(releases.latest(), Path(releases.resolve(releases.latest())))
    assert info.value.reason == "scheme_mismatch" and len(served) == 1
