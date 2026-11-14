"""Heterogeneous-hardware simulation on CPU (gap O2 / T3 / S2): fake islands of
different device families and speeds against the real Rust syncer, and fake
sampler sources of different cards of which one has drifting log-probs, judged
by the per-source xhw guard. No GPU, no Ray, no model.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from free_ports import free_port
from process_groups import group_popen, kill_group
from test_rl_algorithm_provenance import _fake_sky, _no_modal_listing  # noqa: F401
from test_rl_algorithm_provenance import _launcher_args as _provenance_launcher_args
from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.adapters.sampler_engines.fake import (FAKE_IDENTITY, FakeEngine, _unit, fake_identity,
                                                   fake_sampler_fleet)
from yeto.rl.engine import xhw_guard as xg
from yeto.rl.engine.fake_islands import FakeIsland
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer
from yeto.rl.engine.sampler_engine import SamplingParams
from yeto.rl.engine.trainer_lifecycle import SampleIndexEntry

ROOT = Path(__file__).resolve().parent.parent
CARGO = shutil.which("cargo") or str(Path.home() / ".cargo/bin/cargo")
needs_cargo = pytest.mark.skipif(not Path(CARGO).exists(), reason="needs cargo to build the syncer")
NUMEL = 4
FRAGMENTS = 2


# ---------------------------------------------------------------- the fakes: defaults unchanged
def test_fake_island_defaults_are_the_old_behaviour():
    spec = FakeIsland("a", work_s=0.2)
    assert spec.device_family == "cpu" and spec.backend == "fake" and spec.speed == 1.0
    assert spec.round_wall_s == 0.2 and spec.behavior_logprobs(3) == [-0.5] * 4
    fast = FakeIsland("b", work_s=0.2, speed=4.0, noise_std=0.1, device_family="ascend", backend="mindspeed")
    assert fast.round_wall_s == pytest.approx(0.05)
    assert fast.noise(1, 3) == fast.noise(1, 3) != fast.noise(2, 3)  # seeded by island and round
    assert all(v <= 0 for v in fast.behavior_logprobs(1, 64))
    for bad in ({"speed": 0.0}, {"noise_std": -1.0}):
        with pytest.raises(ValueError):
            FakeIsland("c", **bad)


def test_fake_engine_defaults_are_the_old_log_probs():
    sampling = SamplingParams(n=2, max_tokens=8)
    plain, zero = FakeEngine(), FakeEngine(logprob_bias=0.0, logprob_noise=0.0)
    for engine in (plain, zero):
        engine.start(None, 3)
    a, b = plain.generate(["p"], sampling), zero.generate(["p"], sampling)
    assert a == b
    assert a[0][0].logprobs == tuple(-_unit("p", 0, k, 3) for k in range(len(a[0][0].token_ids)))
    assert fake_identity() == FAKE_IDENTITY
    assert plain.score("p", "abc") == [-_unit("p", "abc", k) for k in range(3)]


def test_fake_engine_bias_noise_and_fleet_identities():
    fleet = fake_sampler_fleet([
        {"sampler_id": "b300", "device_family": "nvidia", "engine": "sglang"},
        {"sampler_id": "mi355x", "device_family": "amd", "engine": "vllm", "logprob_noise": 0.02},
        {"sampler_id": "910b", "device_family": "ascend", "engine": "vllm-ascend", "logprob_bias": -0.3},
    ])
    assert {e.identity.card_model for e in fleet.values()} == {"B300", "MI355X", "910B"}
    assert fleet["910b"].provenance()["card_model"] == "910B"
    assert xg.source_key(fleet["910b"].provenance()) == "910b|vllm-ascend|0|910B"
    sampling = SamplingParams(n=1, max_tokens=16)
    out = {}
    for sid, engine in fleet.items():
        engine.start(None, 0)
        out[sid] = np.asarray(engine.generate(["q"], sampling)[0][0].logprobs)
    ref = out["b300"]  # unshifted: the reference
    diff_bias = ref - out["910b"]
    assert np.all(out["910b"] <= 0) and np.mean(diff_bias) > 0.2
    assert 0 < np.max(np.abs(ref - out["mi355x"])) <= 0.02 + 1e-12
    with pytest.raises(ValueError):
        fake_sampler_fleet([{"sampler_id": "x"}, {"sampler_id": "x"}])


def test_fake_island_harness_runs_two_hardware_families_at_a_speed_gap(tmp_path):
    """The multiprocess ledger harness with labelled islands 8x apart: the slow
    island's round wall follows ``speed`` and it is advised as the slow one."""
    from yeto.rl.engine.fake_islands import run_fake_islands

    islands = [FakeIsland("b300", work_s=0.05, device_family="nvidia", backend="megatron", noise_std=0.01),
               FakeIsland("910b", work_s=0.05, speed=0.125, device_family="ascend", backend="mindspeed",
                          noise_std=0.01)]
    res = run_fake_islands(islands, tmp_path / "run", rounds=6, quorum_timeout_s=0.2, lease_s=2.0,
                           hb_s=0.05, deadline_s=60, mode="elastic")
    assert res["failure"] is None and res["final"]["outer_version"] == 6
    cfg = {i["island_id"]: i for i in res["config"]["islands"]}
    assert cfg["910b"]["device_family"] == "ascend" and cfg["910b"]["speed"] == 0.125
    steps = [e for e in res["ledger_events"] if e["kind"] == "outer_step"]
    assert any("910b" in e["absent"] for e in steps)  # the slow family misses windows
    advice = [e for e in res["timeline"] if e["kind"] == "pause_advice"]
    assert all(a["target_resource_intent"]["island_id"] == "910b" for a in advice)


# ---------------------------------------------------------------- 1: speed gap vs the real syncer
def _launcher_args(engine, extra=(), gpu="aws:1xa100@us-east-1"):
    return _provenance_launcher_args(engine, ("--rl-sync-mode", "strict", *extra), gpu=gpu)


@pytest.fixture(scope="module")
def syncer_binary() -> Path:
    subprocess.run([CARGO, "build", "-q"], cwd=ROOT / "syncer", check=True)
    return ROOT / "syncer/target/debug/yeto-syncer"


def _start_decoupled_syncer(binary, workdir, islands, total_steps, *options):
    from yeto import launcher
    from yeto.fragments import MERGE_AVG, Fragment, FragmentLayout
    from yeto.launcher import _prepare_rl_args

    layout = FragmentLayout([Fragment(MERGE_AVG, [(f"actor.w{i}", NUMEL)]) for i in range(FRAGMENTS)])
    args = _launcher_args("ports", (
        "--rl-sync-preset", "decoupled", "--fragments", str(FRAGMENTS), "--pipeline", "1",
        "--total-steps", str(total_steps), "--local-rl-rounds-per-sync", "3", *options),
        gpu=",".join(["modal:2xh100"] * islands))
    args.controller = "local"
    _prepare_rl_args(args)
    command = launcher.role_syncer_commands(args, islands, binary=str(binary))["actor"]
    port = free_port()
    command = (command.replace(f"--port {launcher.SYNCER_PORT}", f"--port {port}")
               .replace(launcher.RL_SYNCER_CHECKPOINT, str(workdir / "state.ckpt"))
               .replace(launcher.RL_SYNCER_EVENT_TAPE, str(workdir / "tape.jsonl")))
    log = (workdir / "syncer.log").open("w")
    process = group_popen(["bash", "-c", command], stdout=log, stderr=subprocess.STDOUT, text=True)
    return process, port, layout, command


class _IslandThread(threading.Thread):
    """One fake island on the real wire: answers the newest PULL it has, after
    ``spec.round_wall_s`` of "training" from the fragment version it held when it
    started; the delta carries the island's numeric noise."""

    def __init__(self, spec, learner_id, client, stop):
        super().__init__(daemon=True, name=spec.island_id)
        self.spec, self.learner_id, self.client, self.stop = spec, learner_id, client, stop
        self.versions = [0] * FRAGMENTS
        self.pushes: list[tuple[int, int, int]] = []  # (step, base, version gap at push)
        self.skipped_pulls = 0
        self.error: BaseException | None = None

    def _drain(self):
        for update in self.client.drain_updates():
            self.versions[update.fragment_id] = max(self.versions[update.fragment_id], update.version)

    def run(self):
        try:
            done = 0
            while not self.stop.is_set() and not self.client.finalizing.is_set():
                self.client.check_health()
                self._drain()
                pulls = sorted(self.client.drain_pulls(), key=lambda p: p.global_step)
                if not pulls:
                    time.sleep(0.005)
                    continue
                self.skipped_pulls += len(pulls) - 1  # a busy island answers only the newest
                pull = pulls[-1]
                base = self.versions[pull.fragment_id]  # what this island trains from
                deadline = time.monotonic() + self.spec.round_wall_s
                while time.monotonic() < deadline and not self.stop.is_set():
                    time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
                self._drain()
                delta = np.full(NUMEL, 0.01, dtype=np.float32) + np.asarray(
                    self.spec.noise(done, NUMEL), dtype=np.float32)
                self.client.push_fragment(pull.fragment_id, pull.global_step, pull.round_attempt, base,
                                          pull.global_step, self.spec.steps_per_round,
                                          self.spec.tokens_per_round, delta.tobytes())
                self.pushes.append((pull.global_step, base, self.versions[pull.fragment_id] - base))
                done += 1
        except BaseException as exc:  # surfaced by the test
            if not self.stop.is_set():
                self.error = exc


def _run_speed_gap(binary, workdir, slow_factor, *, total_steps, max_base_lag):
    from yeto.protocol import DTYPE_F32, SyncerClient

    specs = [FakeIsland("b300", work_s=0.04, device_family="nvidia", backend="megatron", noise_std=1e-3),
             FakeIsland("910b", work_s=0.04, speed=1.0 / slow_factor, device_family="ascend",
                        backend="mindspeed", noise_std=1e-3)]
    process, port, layout, command = _start_decoupled_syncer(
        binary, workdir, len(specs), total_steps, "--rl-sync-quorum", "1",
        "--rl-sync-max-base-lag", str(max_base_lag))
    clients = [SyncerClient(("127.0.0.1", port), i, layout, dtype=DTYPE_F32, num_streams=1,
                            connect_timeout=10, max_reconnects=0, finalization_timeout=60)
               for i in range(len(specs))]
    stop = threading.Event()
    threads = [_IslandThread(s, i, c, stop) for i, (s, c) in enumerate(zip(specs, clients))]
    started = time.monotonic()
    try:
        for client in clients:
            client.start()
        for fid in range(FRAGMENTS):
            clients[0].send_init(fid, np.arange(NUMEL, dtype=np.float32).tobytes())
        for t in threads:
            t.start()
        deadline = time.monotonic() + 120
        while not all(c.finalizing.is_set() for c in clients) and time.monotonic() < deadline:
            assert process.poll() is None, (workdir / "syncer.log").read_text()[-3000:]
            for t in threads:
                assert t.error is None, t.error
            time.sleep(0.02)
        assert all(c.finalizing.is_set() for c in clients), "syncer never finalized"
        stop.set()
        for t in threads:
            t.join(timeout=10)
        finals = []
        for client in clients:
            manifest, fragments = client.wait_for_final_fragments(timeout=30)
            finals.append({f.fragment_id: bytes(f.data)[-NUMEL * 4:] for f in fragments})
            client.acknowledge_finalization(manifest)
        assert process.wait(timeout=15) == 0
    finally:
        stop.set()
        for client in clients:
            client.close()
        kill_group(process)
        process.wait(timeout=10)
    wall = time.monotonic() - started
    assert finals[0] == finals[1]  # one global cut on both hardware families
    rounds = []
    for line in (workdir / "tape.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if "responders" in rec:
            rounds.append(rec)
    log = (workdir / "syncer.log").read_text()
    merged = Counter(r["id"] for rec in rounds for r in rec["responders"])
    carried = Counter(c["id"] for rec in rounds for c in rec.get("carried", []))
    return {
        "command": command, "wall_s": wall, "rounds": len(rounds),
        "contribution": {s.island_id: (merged[i] + carried[i]) / max(len(rounds), 1)
                         for i, s in enumerate(specs)},
        "merged": {s.island_id: merged[i] for i, s in enumerate(specs)},
        "carried": {s.island_id: carried[i] for i, s in enumerate(specs)},
        "pushes": {s.island_id: len(t.pushes) for s, t in zip(specs, threads)},
        "skipped_pulls": {s.island_id: t.skipped_pulls for s, t in zip(specs, threads)},
        "stale_base": log.count("StaleBase"),
        "out_of_round": log.count("OutOfRound"),
        "max_push_lag": {s.island_id: max((g for _, _, g in t.pushes), default=0)
                         for s, t in zip(specs, threads)},
    }


@needs_cargo
def test_speed_gap_islands_against_the_real_syncer(syncer_binary, tmp_path, monkeypatch):
    """A B300-like island and a 910B-like island 3x and 10x slower, quorum 1, base
    lag of one fragment round. The syncer keeps stepping with the fast island in
    every round; the slow island's deltas arrive late (carried over with a lag
    discount) or from a base the lag no longer admits (StaleBase, refused, not
    fatal); its contribution rate falls as the speed gap grows."""
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "hetero-sim-test-key")
    stats = {}
    for factor in (3, 10):
        workdir = tmp_path / f"x{factor}"
        workdir.mkdir()
        stats[factor] = _run_speed_gap(syncer_binary, workdir, factor, total_steps=12, max_base_lag=1)
        print(f"\nslow_factor={factor}: "
              f"{json.dumps({k: v for k, v in stats[factor].items() if k != 'command'})}")
    for factor, st in stats.items():
        assert st["rounds"] >= 12
        assert st["contribution"]["b300"] >= 0.9, st  # the fast island is in (almost) every round
        assert st["contribution"]["910b"] < st["contribution"]["b300"], st
        assert st["merged"]["910b"] + st["carried"]["910b"] > 0, st  # late, but not lost
    assert stats[10]["contribution"]["910b"] < stats[3]["contribution"]["910b"], stats
    assert stats[10]["skipped_pulls"]["910b"] > stats[3]["skipped_pulls"]["910b"], stats
    assert stats[10]["pushes"]["910b"] < stats[10]["pushes"]["b300"], stats


# ---------------------------------------------------------------- 2: one drifting sampler source
def _write_round(buf, fleet, rollout_id, sampling, counts):
    """Each source writes ``counts[sid]`` groups; returns {position: (sid, prompt)}."""
    for sid, engine in fleet.items():
        for g in range(counts[sid]):
            prompt = f"r{rollout_id}-{sid}-{g}"
            group = engine.generate([prompt], sampling)[0]
            samples = [{"tokens": list(c.token_ids), "reward": float(i % 2),
                        "rollout_log_probs": list(c.logprobs),
                        "metadata": {"prompt": prompt, "index": i}} for i, c in enumerate(group)]
            name = f"{rollout_id}-{sid}-{g}"
            ref = buf.write_payload(name, br.encode_group(samples))
            buf.append(SampleIndexEntry(name, 0, ref, tokens=sum(len(s["tokens"]) for s in samples),
                                        provenance=engine.provenance()))


def _train_step(groups, reference):
    """A fake trainer: recomputes every sampled token's log-prob with the reference
    (unshifted) numerics and returns the Miles step metrics of the batch."""
    diffs = []
    for _, samples in groups:
        for s in samples:
            meta = s["metadata"]
            train = [-_unit(meta["prompt"], meta["index"], k, reference) for k in range(len(s["tokens"]))]
            diffs.extend(t - r for t, r in zip(train, s["rollout_log_probs"]))
    d = np.asarray(diffs, dtype=np.float64)
    ratio = np.exp(d)
    return {"train_rollout_logprob_abs_diff": float(np.mean(np.abs(d))),
            "train/train_rollout_kl": float(np.mean(ratio - 1 - d)),  # k3 estimator
            "sc_is_clipped": float(np.mean(np.abs(ratio - 1) > 0.2))}


def _run_three_sources(tmp_path, bias, *, min_rounds, rounds=8):
    fleet = fake_sampler_fleet([
        {"sampler_id": "b300", "device_family": "nvidia", "engine": "sglang"},
        {"sampler_id": "mi355x", "device_family": "amd", "engine": "vllm", "logprob_noise": 0.04},
        {"sampler_id": "910b", "device_family": "ascend", "engine": "vllm-ascend", "logprob_bias": bias},
    ])
    keys = {sid: xg.source_key(e.provenance()) for sid, e in fleet.items()}
    bad = keys["910b"]
    for engine in fleet.values():
        engine.start(None, 0)
    buf = FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1)
    reader = br.BatchReader(buf, max_policy_age=4, prefetch=False, read_workers=1, wait_s=5)
    guard = xg.XhwGuard(min_rounds=min_rounds, window=max(3, min_rounds), quarantine_rounds=8)
    sampling = SamplingParams(n=4, max_tokens=12)
    events, seen = [], []
    for rollout_id in range(rounds):
        # token shares change every round, so a mixed window is separable
        counts = {sid: 1 + (rollout_id + j) % 3 for j, sid in enumerate(fleet)}
        _write_round(buf, fleet, rollout_id, sampling, counts)
        expected = sum(counts.values()) - sum(
            counts[sid] for sid, key in keys.items() if guard.actions().get(key, {}).get("action") == "quarantine")
        if expected == 0:
            events.append((None, None))
            break
        groups, rep = reader.take(rollout_id, expected, current_version=0)
        seen.append((dict(rep.source_groups), rep.quarantined_groups))
        metrics = _train_step(groups, 0)
        ev = guard.observe(rollout_id, rep.source_tokens, metrics)
        reader.set_xhw_actions(ev["actions"])
        events.append((ev, metrics))
    return keys, events, seen


@pytest.mark.parametrize("bias", [-0.3, -0.5])
def test_xhw_guard_attributes_and_quarantines_the_drifting_source(tmp_path, bias):
    """Three sources (B300 SGLang, MI355X vLLM with small noise, 910B vllm-ascend
    with a log-prob bias), token shares varying per round. With a window of three
    rounds the mix is separable: the guard names the 910B source alone, quarantines
    it, and the following batches are clean."""
    keys, events, seen = _run_three_sources(tmp_path, bias, min_rounds=3)
    bad = keys["910b"]
    tripped = next(i for i, (ev, _) in enumerate(events) if bad in ev["actions"])
    ev = events[tripped][0]
    assert ev["actions"] == {bad: {"action": "quarantine", "until_round": tripped + 1 + 8}}
    row = ev["sources"][bad]
    assert row["level"] == "act" and "abs_diff" in row["over"] and row["card_model"] == "910B"
    assert row["abs_diff"] == pytest.approx(abs(bias), rel=0.25)
    for sid in ("b300", "mi355x"):  # the healthy sources are not blamed
        other = ev["sources"][keys[sid]]
        assert other["level"] == "ok" and other["action"] is None, other
    assert tripped <= 3
    # after the trip the bad source is passed over and the batch is clean again
    after = seen[tripped + 1:]
    assert after and all(bad not in groups for groups, _ in after)
    assert sum(q for _, q in after) > 0  # its groups were passed over, not trained on
    assert all(m["train_rollout_logprob_abs_diff"] < 0.05 for _, m in events[tripped + 1:])
    assert all(math.isfinite(m["train/train_rollout_kl"]) for _, m in events)


@pytest.mark.xfail(strict=True, reason=(
    "xhw_guard acts on an unidentifiable estimate: with fewer window rows than sources "
    "_solve falls back to the per-source mean of its rows and every mixed source crosses "
    "the act limit (min_rounds=2, three sources) -- healthy sources are quarantined too"))
def test_xhw_guard_does_not_blame_healthy_sources_before_the_mix_is_separable(tmp_path):
    keys, events, _ = _run_three_sources(tmp_path, -0.3, min_rounds=2, rounds=4)
    actions = next(ev["actions"] for ev, _ in events if ev and ev["actions"])
    assert set(actions) == {keys["910b"]}


@pytest.mark.xfail(strict=True, reason=(
    "xhw_guard least squares leaks blame through a non-stationary metric: a source whose "
    "log-probs read too high saturates at 0, its per-round clip fraction moves with the "
    "tokens it sampled, and with window == sources the exact solve hands the residual to "
    "a healthy source (B300 quarantined on clip alone)"))
def test_xhw_guard_positive_saturating_bias_blames_only_its_source(tmp_path):
    keys, events, _ = _run_three_sources(tmp_path, 0.25, min_rounds=3)
    actions = next(ev["actions"] for ev, _ in events if ev and ev["actions"])
    assert set(actions) == {keys["910b"]}
