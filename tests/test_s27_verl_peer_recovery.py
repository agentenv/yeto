"""S27 G-ED: peer recovery of a verl full-parameter island (VERL-PARITY #17).

A verl island whose process died rejoins with a fresh trainer and takes the
inner optimizer state (Adam moments and step, LR scheduler, RNG) from a live
peer, through the snapshot store, as a Miles island does. The exchange, the
rank list, the digest check and the stream island are the engine-neutral code
(``yeto.rl.peer_recovery``, ``yeto.rl.engine.stream_recovery``); the verl part
is the FSDP2 rank files of ``adapters/verl/cut.py`` (``adapters/verl/peer_recovery.py``).

The trainer is an in-process torch model with a real Adam optimizer and LR
scheduler behind ``VerlIsland.call_workers``; the syncer is the real Rust
syncer, started with the launcher's peer-recovery flags. CPU only; the
real-syncer test needs cargo.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from test_s27_verl_full_parameter_launch import MILES_ONLY, _flags, _island_argv, _prepared, _verl_args
from test_s27_verl_full_parameter_sync import (  # noqa: F401 - syncer_binary is a fixture
    SHAPES, TOKENS, _Island, _layout, _Model, _plan, _run_dict, _start_syncer, needs_cargo, syncer_binary,
)
from yeto.rl.adapters.verl import cut as vcut
from yeto.rl.adapters.verl import full_parameter_stream as vfs
from yeto.rl.adapters.verl import island_entry
from yeto.rl.adapters.verl.peer_recovery import VerlRankShards
from yeto.rl.contracts import LocalStepReceipt as Receipt
from yeto.rl.engine import stream_recovery
from yeto.rl.peer_recovery import SnapshotIntegrityError


class _AdamIsland(_Island):
    """``VerlIsland.call_workers`` over one in-process FSDP-like worker with a
    real Adam optimizer and LR scheduler (``cut.train_handles`` finds them)."""

    def __init__(self, model: _Model, *, lr: float = 0.05):
        super().__init__(model)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: 1.0 / (1.0 + 0.1 * step))
        self.worker = SimpleNamespace(actor=SimpleNamespace(engine=SimpleNamespace(
            module=model, optimizer=self.optimizer, lr_scheduler=self.scheduler)))

    def train_round(self, learner_id: int, round_id: int) -> None:
        """One verl round: a deterministic loss, one Adam step, one scheduler step."""
        generator = torch.Generator().manual_seed(1000 * learner_id + round_id)
        self.optimizer.zero_grad()
        loss = sum(((parameter - torch.randn(parameter.shape, generator=generator)) ** 2).sum()
                   for _, parameter in sorted(self.model.named_parameters()))
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def adam_steps(self) -> set[float]:
        return {float(state["step"]) for state in self.optimizer.state.values()}


def _optimizer_snapshot(island: _AdamIsland) -> dict:
    return {"optim": copy.deepcopy(island.optimizer.state_dict()),
            "scheduler": copy.deepcopy(island.scheduler.state_dict()),
            "rng": torch.get_rng_state().clone()}


def _assert_same_optimizer(a: dict, b: dict) -> None:
    sa, sb = a["optim"]["state"], b["optim"]["state"]
    assert sorted(sa) == sorted(sb)
    for index in sa:
        assert sorted(sa[index]) == sorted(sb[index])
        for key in sa[index]:
            assert torch.equal(sa[index][key], sb[index][key]), (index, key)
    assert a["optim"]["param_groups"] == b["optim"]["param_groups"]
    assert a["scheduler"] == b["scheduler"]
    assert torch.equal(a["rng"], b["rng"])


# ----------------------------------------------------------------------------- the rank-file port
def _shards(island: _AdamIsland, lines: list[str]):
    adapter = vfs.VerlChunkedFullParameterAdapter(_layout(), max_chunk_bytes=8)
    return VerlRankShards(vfs.VerlFullParameterGroup(island), adapter, log=lines.append), adapter


def test_a_restore_reexports_the_donors_adam_scheduler_and_rng_state(tmp_path):
    donor = _AdamIsland(_Model(seed=0))
    for round_id in range(3):
        donor.train_round(0, round_id)
    lines: list[str] = []
    shards, _ = _shards(donor, lines)
    saved = _optimizer_snapshot(donor)
    vfs.asyncio.run(shards.save(tmp_path))
    ranks = json.loads((tmp_path / stream_recovery.RANKS).read_text())
    assert ranks["schema"] == "yeto.verl_peer_snapshot/v1" and ranks["layout_tag"].startswith("verl-")
    [row] = ranks["files"]
    assert row["path"] == "rank00000.pt" and row["coord"] == {"rank": 0, "world": 1}
    assert all(key in row for key in stream_recovery.DIGEST_KEYS)

    fresh = _AdamIsland(_Model(seed=7))  # a relaunched process: other weights, no Adam state
    torch.manual_seed(123)  # and another RNG position
    restored_shards, adapter = _shards(fresh, lines)
    adapter.resident = object()
    fingerprint = vfs.asyncio.run(restored_shards.restore(tmp_path))
    assert adapter.resident is None  # the next apply writes and proves every fragment
    _assert_same_optimizer(_optimizer_snapshot(fresh), saved)
    assert all(torch.equal(a, b) for a, b in zip(fresh.model.parameters(), donor.model.parameters()))
    assert fresh.adam_steps() == {3.0}
    donor_line, restored_line = lines
    assert re.search(r"donor snapshot: 1 rank shard\(s\), \d+ bytes, digests ([0-9a-f]{16})", donor_line)
    assert restored_line.endswith(f"digests equal the donor's, digests {fingerprint}")
    assert donor_line.endswith(f"digests {fingerprint}")


def test_a_restore_that_does_not_land_the_donors_state_fails(tmp_path, monkeypatch):
    donor = _AdamIsland(_Model(seed=0))
    donor.train_round(0, 0)
    shards, _ = _shards(donor, [])
    vfs.asyncio.run(shards.save(tmp_path))
    real = vcut.remote_restore_rank

    def lossy(worker, *args):
        result = real(worker, *args)
        with torch.no_grad():
            next(worker.actor.engine.module.parameters()).add_(1.0)
        result.update(vcut._rank_summary(vcut.train_handles(worker), rank=0, world=1,
                                         digest=vcut._digest(vcut.train_handles(worker))))
        return result

    monkeypatch.setattr(vcut, "remote_restore_rank", lossy)
    fresh, _ = _shards(_AdamIsland(_Model(seed=0)), [])
    with pytest.raises(RuntimeError, match="not the donor's state.*state_digest"):
        vfs.asyncio.run(fresh.restore(tmp_path))


def test_a_snapshot_of_another_layout_is_refused(tmp_path):
    donor = _AdamIsland(_Model(seed=0))
    donor.train_round(0, 0)
    shards, _ = _shards(donor, [])
    vfs.asyncio.run(shards.save(tmp_path))
    ranks = json.loads((tmp_path / stream_recovery.RANKS).read_text())
    ranks["layout_tag"] = "verl-" + "0" * 64
    (tmp_path / stream_recovery.RANKS).write_text(json.dumps(ranks))
    fresh, _ = _shards(_AdamIsland(_Model(seed=0)), [])
    with pytest.raises(SnapshotIntegrityError, match="another trainer layout"):
        vfs.asyncio.run(fresh.restore(tmp_path))


def test_a_trainer_with_optimizer_state_refuses_a_restore(tmp_path):
    donor = _AdamIsland(_Model(seed=0))
    donor.train_round(0, 0)
    shards, _ = _shards(donor, [])
    vfs.asyncio.run(shards.save(tmp_path))
    used = _AdamIsland(_Model(seed=0))
    used.train_round(1, 0)
    other, _ = _shards(used, [])
    with pytest.raises(vcut.CutError, match="freshly built trainer"):
        vfs.asyncio.run(other.restore(tmp_path))


def test_miles_and_verl_layout_tags_differ_for_one_fragment_layout():
    from yeto.rl.adapters.miles.full_parameter.peer_recovery import layout_tag as miles_tag

    layout = _layout()
    island = _AdamIsland(_Model(seed=0))
    shards, _ = _shards(island, [])
    verl = vfs.asyncio.run(shards.layout_tag())
    miles = miles_tag(layout.layout_hash, {"world": 1, "tp": 1, "pp": 1, "cp": 1, "ep": 1, "dp": 1})
    assert verl != miles and verl.startswith("verl-") and miles.startswith("miles-")


# ----------------------------------------------------------------------------- the island plan and launcher
def test_the_launcher_sends_peer_recovery_to_verl_islands_and_the_plan_carries_it(model_dir, monkeypatch):
    store = "modal-volume://v/run"
    args = _prepared(_verl_args("--rl-checkpoint-store", store, "--rl-peer-recovery-ttl-s", "900"))
    assert args.rl_peer_recovery is True and args.rl_island_state_source == "peer"
    monkeypatch.delenv(stream_recovery.PEER_RECOVER_ENV, raising=False)
    for learner_id in range(2):
        parsed, ignored = island_entry.parse_args(_island_argv(args, learner_id))
        assert _flags(ignored) <= MILES_ONLY, _flags(ignored) - MILES_ONLY
        assert parsed.rl_peer_recovery_store and parsed.rl_peer_recovery_ttl_s == 900.0
        section = vfs.plan_section(parsed, _run_dict(), model_path=str(model_dir), sync="strict",
                                   algorithm_spec_sha256="a" * 64)["full_parameter"]
        assert section["peer_recovery"] == {"store": parsed.rl_peer_recovery_store, "recover": False,
                                            "ttl_s": 900.0, "timeout_s": 900.0, "staging": None}
        assert section["syncer_epoch"] == parsed.rl_syncer_epoch
        config = vfs.role_stream_config(section, syncer="h:1", learner_id=learner_id, model_revision="a" * 40)
        assert config.peer_recovery.recover is False and config.syncer_epoch == parsed.rl_syncer_epoch
        vfs.refuse_unsupported(config)  # offered now
    # a relaunched island (launch.peer_recovery.relaunch_envs) recovers from a peer
    monkeypatch.setenv(stream_recovery.PEER_RECOVER_ENV, "1")
    parsed, _ = island_entry.parse_args(_island_argv(args, 1))
    section = vfs.plan_section(parsed, _run_dict(), model_path=str(model_dir), sync="strict",
                               algorithm_spec_sha256="a" * 64)["full_parameter"]
    assert section["peer_recovery"]["recover"] is True


@pytest.fixture
def model_dir(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text('{"model_type": "qwen3", "torch_dtype": "bfloat16"}')
    return root


def test_without_peer_recovery_the_plan_and_the_syncer_are_unchanged(model_dir):
    args = _prepared(_verl_args("--rl-checkpoint-store", "modal-volume://v/run", "--no-rl-peer-recovery"))
    parsed, _ = island_entry.parse_args(_island_argv(args, 0))
    section = vfs.plan_section(parsed, _run_dict(), model_path=str(model_dir), sync="strict",
                               algorithm_spec_sha256="a" * 64)["full_parameter"]
    assert "peer_recovery" not in section and "syncer_epoch" not in section


@pytest.mark.parametrize("extra, sync, match", [
    (dict(rl_peer_recovery_store="/s"), "none", "belongs to a synced"),
    (dict(rl_peer_recovery_store="/s", full_parameter_rounds="sweep"), "strict", "needs --full-parameter-rounds async"),
    (dict(rl_peer_recovery_store="/s", bcast_delta=True), "strict", "without --bcast-delta"),
    (dict(rl_peer_recover=True), "strict", "need --rl-peer-recovery-store"),
])
def test_the_island_refuses_peer_recovery_where_it_does_not_apply(model_dir, extra, sync, match):
    from test_s27_verl_full_parameter_sync import _island_args

    args = _island_args(**{"full_parameter_rounds": "async", **extra})
    with pytest.raises(ValueError, match=match):
        vfs.plan_section(args, _run_dict(), model_path=str(model_dir), sync=sync, algorithm_spec_sha256="a" * 64)


# ----------------------------------------------------------------------------- the real syncer
def _recovery_plan(learner_id, port, profile, store, *, rounds, fragments, recover):
    plan = _plan(learner_id, port, profile, rounds=rounds, fragments=fragments, async_rounds=True,
                 matrix_merge="avg", pipeline=2, final_cut="download")
    plan[vfs.PLAN_KEY]["peer_recovery"] = {"store": str(store), "recover": recover, "ttl_s": 600.0,
                                           "timeout_s": 120.0, "staging": None}
    plan[vfs.PLAN_KEY]["syncer_epoch"] = 0
    return plan


def _train(learner_id, plan, results, *, stop_after=None, pace=lambda: 0.05, max_rounds=4000, hooks=None):
    """One verl island process (Adam rounds): train and sync until the final
    cut, or die (close its links, no LEAVE) after ``stop_after`` rounds."""
    model = _Model(seed=0)  # every process loads the same checkpoint
    island = _AdamIsland(model)
    island.full_group = vfs.VerlFullParameterGroup(island)
    sync, _policy_state, stream = vfs.build_full_parameter_sync(island, plan, backend_identity_sha256="c" * 64)
    driver = SimpleNamespace(events=[])
    driver.emit = lambda event, **fields: driver.events.append((event, fields))
    record = {"island": island, "events": driver.events, "stream": stream, "rounds": 0}
    if hooks is not None:
        hooks.append(record)
    try:
        state = sync.start(driver).state
        record["outcome"] = stream.recovery_outcome
        record["adam_steps_at_start"] = island.adam_steps()
        record["optimizer_at_start"] = _optimizer_snapshot(island)
        record["values_at_start"] = model.values()
        record["anchors_at_start"] = [stored.values().clone() for stored in stream._anchors]
        for round_id in range(max_rounds):
            if stop_after is not None and round_id == stop_after:
                record["died"] = True
                return
            island.train_round(learner_id, round_id)
            ids = (f"l{learner_id}-p{id(record)}-r{round_id}",)
            batch_hash = hashlib.sha256(ids[0].encode()).hexdigest()
            policy_hash = state.policy_tensor_hash()
            island.full_group.record_round(
                Receipt(algorithm="grpo", learner_id=learner_id, learner_generation=0,
                        base_policy_version=round_id, base_policy_hash=policy_hash, input_batch_hash=batch_hash,
                        trajectory_ids=ids, trained_tokens=TOKENS, optimizer_steps=1,
                        optimizer_step_succeeded=True,
                        parameter_layout_hash=plan[vfs.PLAN_KEY]["parameter_layout_sha256"]),
                vfs.VerlRoundEvidence(round_id, policy_hash, batch_hash, TOKENS, ids))
            boundary = sync.boundary(driver, rollout_id=round_id, stats=None)
            record["rounds"] = round_id + 1
            if round_id == 0:
                record["adam_steps_after_first_round"] = island.adam_steps()
            state = boundary.state
            if boundary.stop:
                sync.published(driver, rollout_id=round_id, policy_hash=state.policy_tensor_hash())
                break
            time.sleep(pace())
        sync.finish(driver)
        record["values"] = model.values()
        record["final_ack"] = [f for e, f in driver.events if e == "rl_full_parameter_final_ack"]
    except BaseException as error:  # noqa: BLE001 - reported by the test
        record["error"] = error
    finally:
        sync.close()
        results.setdefault(learner_id, []).append(record)


@needs_cargo
@pytest.mark.timeout(300)
def test_a_relaunched_verl_island_recovers_adam_state_from_a_peer_through_the_real_syncer(
        syncer_binary, tmp_path, monkeypatch, capfd):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "verl-peer-recovery-test-key")
    monkeypatch.setenv("TOKIO_WORKER_THREADS", "4")
    store = tmp_path / "peer-store"
    rounds, fragments = 12, len(_layout().fragments.fragments)
    process, port, profile = _start_syncer(syncer_binary, tmp_path, fragments=fragments, rounds=rounds, extra=(
        "--rl-sync-full-parameter-rounds", "async", "--rl-sync-final-cut", "download",
        "--rl-sync-quorum", "1", "--rl-peer-recovery", "--rl-peer-recovery-store", str(store)))
    # What the donor held when it wrote its snapshot (read on the donor's worker).
    saved: list[dict] = []
    real_save = vcut.remote_save_rank

    def recording_save(worker, *args):
        result = real_save(worker, *args)
        handles = vcut.train_handles(worker)
        saved.append({"optim": copy.deepcopy(handles.optimizer.state_dict()),
                      "scheduler": copy.deepcopy(handles.scheduler.state_dict()),
                      "rng": torch.get_rng_state().clone(),
                      "params": [p.detach().clone() for p in handles.module.parameters()]})
        return result

    monkeypatch.setattr(vcut, "remote_save_rank", recording_save)
    back = threading.Event()
    results: dict = {}
    try:
        plans = [_recovery_plan(i, port, profile, store, rounds=rounds, fragments=fragments, recover=False)
                 for i in range(2)]
        # the donor slows down until the relaunched island is back (else the run may end first)
        steady = threading.Thread(target=_train, args=(0, plans[0], results),
                                  kwargs={"pace": lambda: 0.05 if back.is_set() else 0.2})
        steady.start()
        first = threading.Thread(target=_train, args=(1, plans[1], results), kwargs={"stop_after": 3})
        first.start()
        first.join(timeout=60)
        assert results[1][0].get("died") and "error" not in results[1][0], results[1][0].get("error")
        hooks: list[dict] = []
        relaunch = _recovery_plan(1, port, profile, store, rounds=rounds, fragments=fragments, recover=True)
        again = threading.Thread(target=_train, args=(1, relaunch, results), kwargs={"hooks": hooks})
        again.start()
        deadline = time.monotonic() + 120
        while not (hooks and ("outcome" in hooks[0] or "error" in hooks[0])) and time.monotonic() < deadline:
            time.sleep(0.05)
        back.set()
        for thread in (steady, again):
            thread.join(timeout=180)
        assert process.wait(timeout=30) == 0
    finally:
        if process.poll() is None:
            process.terminate()
        log = process.communicate(timeout=10)[0]
    err = capfd.readouterr().err
    donor, restarted = results[0][0], results[1][1]
    assert "error" not in restarted, (restarted.get("error"), log[-3000:])
    assert "error" not in donor, (donor.get("error"), log[-3000:])

    # P1: the syncer recorded a peer recovery of island 1 from donor 0.
    tape = [json.loads(line) for line in (tmp_path / "tape.jsonl").read_text().splitlines() if line]
    [recovery] = [r for r in tape if r.get("kind") == "recovery"]
    assert recovery["path"] == "peer" and recovery["donor"] == 0, recovery
    outcome = restarted["outcome"]
    assert outcome is not None and outcome.path == "peer" and outcome.donor == 0, outcome

    # P2: bit for bit, the relaunched trainer took the donor's Adam moments and
    # step, LR scheduler and RNG as the donor held them at its snapshot.
    [at_snapshot] = saved
    _assert_same_optimizer(restarted["optimizer_at_start"], at_snapshot)
    assert restarted["adam_steps_at_start"] == {float(s["step"]) for s in at_snapshot["optim"]["state"].values()}
    assert min(restarted["adam_steps_at_start"]) >= 1  # not fresh: the donor had trained
    # The donor and the recovering island log the same rank digests.
    donor_digest = re.findall(r"donor snapshot: 1 rank shard\(s\), \d+ bytes, digests ([0-9a-f]{16})", err)
    restored_digest = re.findall(r"restored 1/1 rank shard\(s\): state, RNG and LR scheduler digests equal the "
                                 r"donor's, digests ([0-9a-f]{16})", err)
    assert donor_digest == restored_digest and len(donor_digest) == 1, err[-3000:]
    assert "not the donor's state" not in err

    # The parameters it starts from are the syncer's resent fragments (its
    # anchors), not the donor's local progress and not the checkpoint.
    layout = _layout()
    adapter = vfs.VerlChunkedFullParameterAdapter(layout, max_chunk_bytes=8)
    held = [adapter.flatten(p, restarted["values_at_start"]) for p in range(layout.fragments.num_fragments)]
    assert all(torch.equal(h, a) for h, a in zip(held, restarted["anchors_at_start"]))
    checkpoint = _Model(seed=0).values()
    assert any(not torch.equal(restarted["values_at_start"][n], checkpoint[n]) for n in SHAPES)
    assert max(outcome.versions) >= 1

    # The first round after the restore continues the donor's Adam step count.
    assert restarted["adam_steps_after_first_round"] == {s + 1 for s in restarted["adam_steps_at_start"]}

    # P5: it is back on the roster: its pushes are merged and it acknowledges the final cut.
    after = tape[tape.index(recovery) + 1:]
    merged = [r["step"] for r in after for m in r.get("responders", ())
              if m["id"] == 1 and m["generation"] == recovery["generation"]]
    assert len(merged) >= 3, ("the recovered island's pushes were not merged", merged, restarted["rounds"])
    assert restarted["final_ack"] == [{"acknowledged": True}] == donor["final_ack"]
    for name in SHAPES:  # both downloaded the same final cut
        assert torch.equal(restarted["values"][name], donor["values"][name]), name
    assert list(store.rglob("MANIFEST.json")) == [], "a restored snapshot is removed"
