"""Local-init fallback end to end: real Rust syncer, two CPU verl islands.

Island 1 loaded *different weights* (not just different reported digests).
With ``--rl-sync-local-init-fallback`` the syncer broadcasts learner 0's INIT
to it; the island must write that version 0 into its trainer, re-verify its
digests, and train from it, so its first push carries only its own local
step (no initial difference) and the merged result equals the run where
both islands loaded the same weights. Without the flag the run stops as on
main (``init_digest_mismatch``).
"""
from __future__ import annotations

import hashlib
import json
import threading
import time

import numpy as np
import pytest
import torch

from process_groups import kill_group
from test_rl_algorithm_provenance import _fake_sky, _no_modal_listing  # noqa: F401
from test_verl_full_parameter_sync import (
    SHAPES,
    TOKENS,
    _config,
    _Driver,
    _expected_average,
    _init,
    _Island,
    _layout,
    _Model,
    _plan,
    _start_syncer,
    _step,
    needs_cargo,
    syncer_binary,  # noqa: F401 - fixture
)
from yeto.rl.adapters.verl import full_parameter_stream as vfs
from yeto.rl.contracts import LocalStepReceipt

ROUNDS = 3
OTHER_SEED = 7  # island 1's checkpoint: another base entirely
SWEEP_LOCAL = ("--rl-sync-full-parameter-rounds", "sweep", "--rl-sync-init-mode", "local",
               "--rl-sync-final-cut", "download")


def _record_first_deltas(monkeypatch) -> dict:
    """``{(adapter id, fragment): first pushed delta}`` as the push streams it."""
    first: dict = {}
    original = vfs.VerlChunkedFullParameterAdapter.delta_parts_from_authoritative

    def recording(self, stored, local, fragment_id, **kwargs):
        parts = original(self, stored, local, fragment_id, **kwargs)

        def replay():
            chunks = [bytes(part) for part in parts()]
            first.setdefault((id(self), fragment_id), np.frombuffer(b"".join(chunks), dtype=np.float32))
            yield from chunks
        return replay

    monkeypatch.setattr(vfs.VerlChunkedFullParameterAdapter, "delta_parts_from_authoritative", recording)
    return first


def _run_island(learner_id: int, seed: int, plan: dict, results: dict) -> None:
    model = _Model(seed=seed)
    island = _Island(model)
    island.full_group = vfs.VerlFullParameterGroup(island)
    sync, policy_state, stream = vfs.build_full_parameter_sync(island, plan, backend_identity_sha256="c" * 64)
    driver = _Driver()
    record: dict = {}
    try:
        state = sync.start(driver).state
        record["started"] = model.values()
        record["adapter"] = id(stream.adapter)
        for round_id in range(ROUNDS):
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.add_(_step(learner_id, round_id)[name])
            ids = (f"l{learner_id}-r{round_id}",)
            batch_hash = hashlib.sha256(ids[0].encode()).hexdigest()
            policy_hash = state.policy_tensor_hash()
            island.full_group.record_round(
                LocalStepReceipt(algorithm="grpo", learner_id=learner_id, learner_generation=0,
                                 base_policy_version=round_id, base_policy_hash=policy_hash,
                                 input_batch_hash=batch_hash, trajectory_ids=ids, trained_tokens=TOKENS,
                                 optimizer_steps=1, optimizer_step_succeeded=True,
                                 parameter_layout_hash=plan[vfs.PLAN_KEY]["parameter_layout_sha256"]),
                vfs.VerlRoundEvidence(round_id, policy_hash, batch_hash, TOKENS, ids))
            boundary = sync.boundary(driver, rollout_id=round_id, stats=None)
            state = boundary.state
            if boundary.stop:
                sync.published(driver, rollout_id=round_id, policy_hash=state.policy_tensor_hash())
                break
        sync.finish(driver)
        record["values"] = model.values()
        results[learner_id] = record
    except BaseException as error:  # noqa: BLE001 - reported by the test
        results[learner_id] = error
    finally:
        sync.close()
        del policy_state


def _run(binary, tmp_path, monkeypatch, *, fallback: bool):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "verl-full-parameter-test-key")
    monkeypatch.setenv("TOKIO_WORKER_THREADS", "4")
    fragments = len(_layout().fragments.fragments)
    extra = SWEEP_LOCAL + (("--rl-sync-local-init-fallback",) if fallback else ())
    process, port, profile = _start_syncer(binary, tmp_path, fragments=fragments, rounds=ROUNDS, extra=extra)
    results: dict = {}
    try:
        threads = []
        for learner_id, seed in ((0, 0), (1, OTHER_SEED)):
            plan = _plan(learner_id, port, profile, rounds=ROUNDS, fragments=fragments, async_rounds=False,
                         matrix_merge="avg", pipeline=1, final_cut="download")
            plan[vfs.PLAN_KEY]["local_init"] = True
            if fallback:
                plan[vfs.PLAN_KEY]["local_init_fallback"] = True
            threads.append(threading.Thread(target=_run_island, args=(learner_id, seed, plan, results)))
        # Island 1 first, so its digests reach the syncer before learner 0's
        # INIT is applied (the in-time report: the run stops without the
        # flag, the island gets the initial broadcast with it). The first
        # dial waits ~2 s for the syncer to listen.
        threads[1].start()
        time.sleep(4.0)
        threads[0].start()
        for thread in threads:
            thread.join(timeout=120)
        code = process.wait(timeout=30)
    finally:
        kill_group(process)
        log = process.communicate(timeout=10)[0]
    tape = [json.loads(line) for line in (tmp_path / "tape.jsonl").read_text().splitlines() if line]
    return code, results, log, tape, fragments


def _fragment_values(model_values: dict[str, torch.Tensor], fragments: int) -> list[np.ndarray]:
    """The f32 bytes of each fragment of a model holding ``model_values``."""
    model = _Model()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(model_values[name])
    cut = _init(_Island(model), _config(minimum_fragments=fragments))[2]
    return [cut.fragments[i].numpy().copy() for i in range(fragments)]


@needs_cargo
def test_a_fallback_island_with_other_weights_trains_from_version_zero(
        syncer_binary, tmp_path, monkeypatch, capfd):  # noqa: F811
    first = _record_first_deltas(monkeypatch)
    code, results, log, tape, fragments = _run(syncer_binary, tmp_path, monkeypatch, fallback=True)
    for learner_id in (0, 1):
        assert not isinstance(results.get(learner_id), BaseException), (results.get(learner_id), log[-3000:])
    assert code == 0, log[-3000:]
    base, other = _Model(seed=0).values(), _Model(seed=OTHER_SEED).values()
    initial_gap = max(float((other[name] - base[name]).abs().max()) for name in SHAPES)
    assert initial_gap > 1.0  # the islands really loaded different weights
    # Island 1 was pulled back to learner 0's version 0 before it trained.
    for name in SHAPES:
        assert torch.equal(results[1]["started"][name], base[name]), name
    # Its digests now equal learner 0's INIT, fragment by fragment.
    v0 = _fragment_values(base, fragments)
    started = _fragment_values(results[1]["started"], fragments)
    assert [hashlib.sha256(a.tobytes()).digest() for a in started] == [
        hashlib.sha256(a.tobytes()).digest() for a in v0]
    # Its first push of every fragment is its own round-0 step, nothing more.
    stepped = {name: base[name] + _step(1, 0)[name] for name in SHAPES}
    want = [after - before for after, before in zip(_fragment_values(stepped, fragments), v0, strict=True)]
    adapter = results[1]["adapter"]
    for fragment_id in range(fragments):
        delta = first[(adapter, fragment_id)]
        np.testing.assert_allclose(delta, want[fragment_id], rtol=0, atol=1e-6)
        assert float(np.abs(delta).max()) < 0.1 < initial_gap
    # The merge equals the run where both islands loaded the same base.
    expected = _expected_average(ROUNDS)
    for learner_id in (0, 1):
        for name, value in expected.items():
            assert torch.allclose(results[learner_id]["values"][name], value, atol=1e-6), (learner_id, name)
    # One tape record and one island log line, for island 1 only.
    events = [r for r in tape if r.get("event") == "init_digest_fallback_upload"]
    assert [(r["learner_id"], r["mismatched_fragments"]) for r in events] == [(1, fragments)]
    assert not any(r.get("event") == "init_digest_mismatch" for r in tape)
    err = capfd.readouterr().err
    lines = [line for line in err.splitlines() if "local-init fallback" in line]
    assert len(lines) == 1 and "learner 1 local-init fallback" in lines[0], lines
    assert f"{fragments}/{fragments} fragment(s) differed" in lines[0]


@needs_cargo
def test_without_the_flag_other_weights_stop_the_run_as_on_main(
        syncer_binary, tmp_path, monkeypatch, capfd):  # noqa: F811
    code, results, log, tape, _fragments = _run(syncer_binary, tmp_path, monkeypatch, fallback=False)
    assert code != 0
    assert "RL strict failure init_digest_mismatch" in log
    assert "every island must load the same base checkpoint (or start with --rl-init-mode upload)" in log
    # main's strict-failure record, and nothing else on the tape but the syncer's
    # own connection records (keyed by "kind")
    assert {r["kind"] for r in tape if "kind" in r} <= {"member_connected", "member_disconnected"}, tape
    tape = [r for r in tape if "kind" not in r]
    assert [(r["event"], r["metric"], sorted(r)) for r in tape] == [
        ("rl_strict_failure", "init_digest_mismatch", ["error", "event", "metric", "value"])], tape
    assert tape[0]["error"].startswith("init digest mismatch: fragment")
    for learner_id in (0, 1):
        assert isinstance(results.get(learner_id), BaseException), results.get(learner_id)
    assert "local-init fallback" not in capfd.readouterr().err


# ----------------------------------------------------------------------------- Miles
MILES_ROUNDS = 6  # outer versions per fragment (total steps)
MILES_OFFSET = 1.0  # island 1's checkpoint: learner 0's base plus this
MILES_STEP = 0.05  # island 1's local step per round, every element


def _miles_run(binary, tmp_path, monkeypatch):
    """Learner 0: a raw client that loaded the base (digests + INIT, answers
    every PULL). Learner 1: the real Miles role stream (CPU fakes for the
    Megatron group and Ray) whose trainer loaded base + MILES_OFFSET."""
    pytest.importorskip("miles")
    import asyncio
    import time
    from dataclasses import replace

    from test_rl_full_parameter_async import _start_async_syncer
    from test_rl_full_parameter_sync import _role_stream_config
    from test_rl_miles_chunked_full_parameter import FakeRay, _evidence, component
    from test_rl_miles_restart_rejoin import _FreshTrainer, _raw_client, _RawIsland
    from test_sync_overlap_miles_role_stream import SIZES as MILES_SIZES
    from test_sync_overlap_miles_role_stream import _adapter
    from test_sync_overlap_miles_role_stream import _Driver as MilesDriver
    from yeto.rl.adapters.miles.full_parameter.role_stream import (
        MilesFullParameterRoleStream,
        role_stream_session_contract_hash,
    )
    from yeto.rl.adapters.miles.models.chunked_full_parameter import MilesChunkedFullParameterAdapter
    from yeto.rl.engine.full_parameter_stream import STAGGER_FIRST_PUSH_ENV
    from yeto.rl.engine.full_parameter_sync import FullParameterSync, FullParameterSyncProfile, RolePairing

    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "miles-local-init-fallback-test-key")
    monkeypatch.setenv(STAGGER_FIRST_PUSH_ENV, "0")
    fragments = len(MILES_SIZES)
    process, port, profile_hash = _start_async_syncer(
        binary, tmp_path, "--rl-sync-local-init-fallback", total_steps=MILES_ROUNDS)
    adapter = _adapter()
    config = replace(
        _role_stream_config("actor", port), component=component(), learner_id=1, learner_generation=0,
        learner_generations=(0, 0), total_fragment_steps=MILES_ROUNDS * fragments,
        expected_fragments=fragments, expected_layout_hash=adapter.layout.layout_hash,
        syncer_profile_hash=profile_hash, pipeline_depth=2, wait_timeout=60.0, poll_seconds=0.005,
        max_fragment_bytes=64, max_chunk_bytes=8, partial_quorum=True,
        local_init=True, local_init_fallback=True)
    contract = role_stream_session_contract_hash(adapter.layout.layout_hash, config)
    base = {f: torch.arange(n, dtype=torch.float32) + float(f) for f, n in enumerate(MILES_SIZES)}
    first: dict = {}
    original = MilesChunkedFullParameterAdapter.delta_parts_from_authoritative

    def recording(self, stored, local, fragment_id, **kwargs):
        parts = original(self, stored, local, fragment_id, **kwargs)

        def replay():
            chunks = [bytes(part) for part in parts()]
            first.setdefault(fragment_id, np.frombuffer(b"".join(chunks), dtype=np.float32))
            yield from chunks
        return replay

    monkeypatch.setattr(MilesChunkedFullParameterAdapter, "delta_parts_from_authoritative", recording)
    ray = FakeRay()
    trainer = _FreshTrainer(adapter, ray, {f: v + MILES_OFFSET for f, v in base.items()},
                            lambda _k: {f: torch.full((n,), MILES_STEP) for f, n in enumerate(MILES_SIZES)})

    async def initialize(cls, group, *, policy_version, **_kwargs):
        cut = await group.export_full_parameter_chunked_cut(policy_version, 0,
                                                            max_chunk_bytes=adapter.max_chunk_bytes)
        return adapter, adapter.validate_cut(cut, policy_version, 0)

    monkeypatch.setattr(MilesChunkedFullParameterAdapter, "initialize", classmethod(initialize))
    seed = _RawIsland(_raw_client(0, port, profile_hash, adapter, contract), 0)
    stream = None
    out: dict = {}
    try:
        seed.client.set_init_digests(hashlib.sha256(adapter.layout.layout_hash.encode()).digest(),
                                     [hashlib.sha256(base[f].numpy().tobytes()).digest()
                                      for f in range(fragments)])
        seed.client.start()
        for fid in range(fragments):
            seed.client.send_init(fid, base[fid].numpy().tobytes())
        seed.start()
        stream = MilesFullParameterRoleStream(config, ray_module=ray)
        session = FullParameterSync(
            RolePairing({"actor": stream}, FullParameterSyncProfile.from_configs({"actor": config})),
            groups={"actor": trainer}, run=asyncio.run,
            trajectories=lambda rollout_id: _evidence(rollout_id=rollout_id,
                                                      behavior_policy_hash=stream.current.policy_hash))
        driver = MilesDriver()
        session.start(driver)
        out["started"] = {f: v.clone() for f, v in trainer.values.items()}
        out["anchors"] = [stream._anchors[f].version for f in range(fragments)]
        stop, deadline, rollout_id = False, time.monotonic() + 60, -1
        while time.monotonic() < deadline:
            rollout_id += 1
            boundary = session.boundary(driver, rollout_id=rollout_id, stats=None)
            if boundary.stop:
                stop = True
                session.published(driver, rollout_id=rollout_id, policy_hash=boundary.state.policy_tensor_hash())
                break
        assert stop, "the fallback island never reached the final cut"
        session.finish(driver)
        seed.join(timeout=30)
        assert seed.error is None, seed.error
        manifest, _finals = seed.client.wait_for_final_fragments(timeout=30)
        seed.client.acknowledge_finalization(manifest)
        assert process.wait(timeout=20) == 0
    finally:
        seed.halt.set()
        seed.client.close()
        if stream is not None:
            stream.close()
        if process.poll() is None:
            process.terminate()
        log, _ = process.communicate(timeout=10)
    tape = [json.loads(line) for line in (tmp_path / "tape.jsonl").read_text().splitlines() if line]
    return base, out, first, tape, log


@needs_cargo
def test_a_fallback_miles_island_with_other_weights_trains_from_version_zero(
        syncer_binary, tmp_path, monkeypatch, capfd):  # noqa: F811
    base, out, first, tape, log = _miles_run(syncer_binary, tmp_path, monkeypatch)
    fragments = len(base)
    for f in range(fragments):  # pulled back: the trainer holds learner 0's version 0
        assert torch.equal(out["started"][f], base[f]), (f, log[-2000:])
        assert hashlib.sha256(out["started"][f].numpy().tobytes()).digest() == hashlib.sha256(
            base[f].numpy().tobytes()).digest()
    assert out["anchors"] == [0] * fragments
    for f in range(fragments):
        # The first push is whole local steps of MILES_STEP from version 0;
        # carrying the initial difference would add MILES_OFFSET to every element.
        delta = first[f]
        rounds = round(float(delta[0]) / MILES_STEP)
        assert rounds >= 1
        np.testing.assert_allclose(delta, np.full_like(delta, rounds * MILES_STEP), rtol=0, atol=1e-6)
        assert float(np.abs(delta).max()) < MILES_OFFSET
    events = [r for r in tape if r.get("event") == "init_digest_fallback_upload"]
    assert [(r["learner_id"], r["mismatched_fragments"]) for r in events] == [(1, fragments)]
    lines = [line for line in capfd.readouterr().err.splitlines() if "local-init fallback" in line]
    assert len(lines) == 1 and "learner 1 local-init fallback" in lines[0], lines
