"""rl-infra-spec 2.3: IslandDriver partitioned-overlap (eval||train/outer_sync) on the fake engine."""

from __future__ import annotations

import json

import pytest

from yeto.rl.engine.algorithm import AlgorithmSpec
from yeto.rl.engine.driver import DriverError
from yeto.rl.engine.execution_profile import ExecutionProfile
from yeto.rl.engine.fake import fake_capabilities
from yeto.rl.engine.overlap import IMPLEMENTED_OVERLAP
from yeto.rl.engine.timeline import Span, summarize

from test_rl_driver_profiles import _driver, _engine, _events

MODES = {"colocated-serial", "partitioned-serial", "partitioned-overlap"}


def _profile(mode, pairs=frozenset()):
    return ExecutionProfile(name=f"t-{mode}", execution_mode=mode, outer_protocol="none",
                            allowed_overlap=pairs).bind_algorithm(AlgorithmSpec())


def _run(tmp_path, mode, name, *, pairs=frozenset(), engine=None):
    import time

    engine = engine or _engine(placement_kind="fixed-partition")
    evals, windows, cancelled = [], [], []

    def evaluate(rid):
        evals.append(("serial", rid))
        return {"score": float(rid)}

    class Handle:
        def __init__(self, rid):
            self.rid, self.at = rid, len(engine.calls)
            self.begin = time.monotonic()
            self.end = self.begin + 1e-6

        def cancel(self):
            cancelled.append(self.rid)

        def result(self):
            windows.append((self.rid, list(engine.calls[self.at:])))
            evals.append(("overlap", self.rid))
            return {"score": float(self.rid)}

    driver, trained = _driver(
        engine, tmp_path, name, profile=_profile(mode, pairs), observe=True,
        capabilities=fake_capabilities(execution_modes=MODES),
        evaluate=evaluate, eval_interval=1, evaluate_start=Handle,
    )
    _run.cancelled = cancelled
    final = driver.run()
    return driver, final, trained, evals, windows, _events(tmp_path / name)


def test_overlap_matches_serial_training_and_evals_the_same_policies(tmp_path):
    _, s_final, s_trained, s_evals, _, s_events = _run(tmp_path, "partitioned-serial", "s.jsonl")
    _, o_final, o_trained, o_evals, windows, o_events = _run(
        tmp_path, "partitioned-overlap", "o.jsonl", pairs=IMPLEMENTED_OVERLAP)
    # Same sample IDs, same optimizer order, same final weights.
    assert o_trained == s_trained and len(o_trained) == 3
    assert o_final.policy_tensor_hash() == s_final.policy_tensor_hash()
    # Same eval points and policy tokens; all but the final one overlapped.
    tokens = lambda ev: [(e["policy_version"], e["rl/policy_token"]) for e in ev  # noqa: E731
                         if e["event"] == "rl_eval"]
    assert sorted(tokens(o_events)) == sorted(tokens(s_events))
    assert [k for k, _ in o_evals] == ["overlap", "overlap", "overlap", "serial"]
    # While each eval was in flight the trainer trained, and nothing was published
    # or generated (rollout role exclusive, eval joined before publication).
    for rid, calls in windows:
        names = [c[0] for c in calls]
        assert "train" in names, (rid, names)
        assert "publish" not in names and "generate" not in names, (rid, names)
    order = [e["event"] for e in o_events]
    # every overlapped eval is emitted before the next publication
    for e in (e for e in o_events if e["event"] == "rl_eval" and e.get("overlapped")):
        i = o_events.index(e)
        nxt = next(x for x in o_events[i:] if x["event"] == "rl_publication")
        assert nxt["policy_version"] == e["policy_version"] + 1
    assert order.count("rl_eval_overlap_start") == 3
    # M2: the readiness snapshot of each train step sees the eval in flight
    train_ready = [e for e in o_events if e["event"] == "rl_readiness"
                   and e.get("inflight_batches") == 1]
    assert train_ready and all(e["eval_in_flight"] == 1 for e in train_ready)
    spans = [Span(**{k: e[k] for k in ("task", "role", "kind", "start", "end", "rollout_id")},
                  profile_hash=e["profile_hash"], epoch=e["epoch"])
             for e in o_events if e["event"] == "rl_timeline_span"]
    summary = summarize(spans)
    assert "eval" in summary["by_task"]
    assert summary["wall_s"] <= sum(s.end - s.start for s in spans) + 1e-9


def test_overlap_profiles_other_than_the_implemented_pairs_are_refused(tmp_path):
    for pairs in (frozenset(), frozenset({("eval", "train")}),
                  frozenset({("reward", "checkpoint")})):
        engine = _engine(placement_kind="fixed-partition")
        driver, _ = _driver(engine, tmp_path, profile=_profile("partitioned-overlap", pairs),
                            capabilities=fake_capabilities(execution_modes=MODES),
                            evaluate=lambda r: {}, eval_interval=1,
                            evaluate_start=lambda r: None)
        with pytest.raises(DriverError, match="2.3"):
            driver.run()
        assert engine.calls == []
    engine = _engine(placement_kind="fixed-partition")
    driver, _ = _driver(engine, tmp_path, profile=_profile("partitioned-overlap",
                                                           IMPLEMENTED_OVERLAP),
                        capabilities=fake_capabilities(execution_modes=MODES))
    with pytest.raises(DriverError, match="evaluate_start"):
        driver.run()


def test_delayed_publication_waits_for_the_eval_and_never_starts_generation(
        tmp_path, monkeypatch):
    cfg = tmp_path / "fault.json"
    cfg.write_text(json.dumps({"publish_delay_s": 0.01}))
    monkeypatch.setenv("YETO_RL_FAULT_INJECTION", str(cfg))
    _, _, trained, _, windows, events = _run(
        tmp_path, "partitioned-overlap", "d.jsonl", pairs=IMPLEMENTED_OVERLAP)
    assert len(trained) == 3
    assert sum(e["event"] == "rl_fault_injected" for e in events) == 4
    published = -1
    for e in events:
        if e["event"] == "rl_publication":
            published = e["policy_version"]
        if e["event"] == "rl_driver_phase" and e["phase"] == "generate":
            assert e["policy_version"] == published  # never ahead of / behind publication
        if e["event"] == "rl_eval_overlap_start":
            assert e["policy_version"] == published
    assert all("publish" not in [c[0] for c in calls] for _, calls in windows)


def test_ports_entry_opts_into_eval_overlap_only_when_asked():
    from types import SimpleNamespace

    from yeto.rl.engine.execution_profile import ProfileError
    from yeto.rl.engine.miles_adapter import entry
    from yeto.rl.engine.miles_adapter.placement import PlacementRequest

    spec = AlgorithmSpec()
    args = SimpleNamespace(rollout_batch_size=4, n_samples_per_prompt=8, num_steps_per_rollout=1)
    part = SimpleNamespace(placement=PlacementRequest("fixed-partition", 2, 2, 1), argv=("x",))
    colo = SimpleNamespace(placement=PlacementRequest("colocated", 2, 2, 1), argv=("x",))
    build = lambda launch: entry.execution_profile_for(  # noqa: E731
        args, launch, spec, yeto_policy_sync=False, expected_sha256=spec.sha256())
    assert build(part).execution_mode == "partitioned-serial"  # default unchanged
    args.yeto_rl_overlap_eval = True
    args.eval_interval = 1
    p = build(part)
    assert p.execution_mode == "partitioned-overlap" and p.allowed_overlap == IMPLEMENTED_OVERLAP
    caps = entry.with_partitioned_serial(entry.miles_capabilities("sha256:" + "1" * 64))
    entry.preflight(p, spec, caps)
    with pytest.raises(ProfileError, match="fixed-partition"):
        build(colo)
    args.eval_interval = None
    with pytest.raises(ProfileError, match="eval-interval"):
        build(part)
    args.eval_interval = 1
    args.eval_uses_snapshots = True
    with pytest.raises(ProfileError, match="snapshots"):
        build(part)


def test_train_failure_cancels_the_in_flight_eval(tmp_path):
    engine = _engine(placement_kind="fixed-partition")
    original = engine.trainer.train_step

    def boom(batch):
        if batch.rollout_id == 1:
            raise RuntimeError("trainer died")
        return original(batch)

    engine.trainer.train_step = boom
    with pytest.raises(RuntimeError, match="trainer died"):
        _run(tmp_path, "partitioned-overlap", "f.jsonl", pairs=IMPLEMENTED_OVERLAP,
             engine=engine)
    assert _run.cancelled == [1]  # eval of v1 was in flight during train(1)
    events = _events(tmp_path / "f.jsonl")
    assert [e["policy_version"] for e in events
            if e["event"] == "rl_eval_overlap_aborted"] == [1]
