"""rl-infra-spec 1.4/1.7/2.2: IslandDriver execution profiles and opt-in observation (CPU)."""

from __future__ import annotations

import json

import pytest
import torch

from yeto.rl.engine.algorithm import AlgorithmSpec
from yeto.rl.engine.bridges import LocalOnlySync
from yeto.rl.engine.driver import DriverError, EventTape, IslandDriver, PolicyIdentityError
from yeto.rl.engine.execution_profile import ExecutionProfile, ReadinessError
from yeto.rl.engine.fake import FakeEngine, fake_capabilities
from yeto.rl.engine.timeline import Span, summarize

NAME = "base_model.model.layer.lora_A.weight"
MODES = {"colocated-serial", "partitioned-serial"}


def _engine(**kw):
    return FakeEngine(tensors={NAME: torch.zeros(1, 2)}, step_delta=1.0, **kw)


def _profile(mode, algorithm=None, **kw):
    p = ExecutionProfile(name=f"t-{mode}", execution_mode=mode, outer_protocol="none", **kw)
    return p.bind_algorithm(algorithm or AlgorithmSpec()) if algorithm is not False else p


def _driver(engine, tmp_path, name="events.jsonl", rounds=3, **kw):
    kw.setdefault("capabilities", fake_capabilities(execution_modes=MODES))
    trained = []
    original = engine.trainer.train_step

    def train_step(batch):
        trained.append(tuple(s for g in batch.groups for s in g.sample_ids))
        return original(batch)

    engine.trainer.train_step = train_step
    driver = IslandDriver(
        learner_id=0,
        rollout=engine.rollout,
        trainer=engine.trainer,
        policy_state=engine.policy_state,
        publisher=engine.publisher,
        placement=engine.placement,
        algorithm=kw.pop("algorithm", AlgorithmSpec()),
        sync=LocalOnlySync(rounds),
        events=EventTape(tmp_path / name, 0),
        **kw,
    )
    return driver, trained


def _events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _strip(events):
    return [{k: v for k, v in e.items() if k not in ("time_unix",) and "seconds" not in k}
            for e in events]


NEW_EVENTS = {"rl_timeline_span", "rl_readiness", "rl_round_labels"}


def test_observation_off_keeps_the_r0_event_tape(tmp_path):
    legacy, _ = _driver(_engine(), tmp_path, "legacy.jsonl")
    legacy.run()
    off, _ = _driver(_engine(), tmp_path, "off.jsonl", observe=False)
    off.run()
    on, _ = _driver(_engine(), tmp_path, "on.jsonl", observe=True)
    on.run()
    base = _strip(_events(tmp_path / "legacy.jsonl"))
    assert _strip(_events(tmp_path / "off.jsonl")) == base
    observed = _events(tmp_path / "on.jsonl")
    assert _strip([e for e in observed if e["event"] not in NEW_EVENTS]) == base
    assert {e["event"] for e in observed} >= {"rl_timeline_span", "rl_round_labels"}


def test_partitioned_serial_keeps_sample_ids_and_optimizer_order(tmp_path):
    colo_engine = _engine()
    colo, colo_trained = _driver(colo_engine, tmp_path, "c.jsonl",
                                 profile=_profile("colocated-serial"))
    colo_final = colo.run()
    part_engine = _engine(placement_kind="fixed-partition")
    part, part_trained = _driver(part_engine, tmp_path, "p.jsonl",
                                 profile=_profile("partitioned-serial"), observe=True)
    part_final = part.run()
    assert part_trained == colo_trained and len(part_trained) == 3
    strip = lambda calls: [c for c in calls if c[0] not in ("export", "offload", "onload")]  # noqa: E731
    assert strip(part_engine.calls) == strip(colo_engine.calls)
    assert not any(c[0] in ("offload", "onload") for c in part_engine.calls)
    assert torch.equal(part_final.tensors[NAME], colo_final.tensors[NAME])
    events = _events(tmp_path / "p.jsonl")
    start = next(e for e in events if e["event"] == "rl_driver_start")
    assert start["execution_mode"] == "partitioned-serial"
    assert start["profile_hash"] == part.profile.contract_hash
    labels = [e for e in events if e["event"] == "rl_round_labels"]
    assert {e["weight_transport"] for e in labels} == {"nccl-broadcast"}
    assert {e["profile_hash"] for e in labels} == {part.profile.contract_hash}
    spans = [
        Span(e["task"], e["role"], e["kind"], e["start"], e["end"], e["profile_hash"],
             e["epoch"], e["rollout_id"])
        for e in events if e["event"] == "rl_timeline_span"
    ]
    summary = summarize(spans)
    assert summary["overlap_s"] == pytest.approx(0.0, abs=1e-9)  # serial: nothing double-billed
    assert {"generate", "train", "outer_sync", "publish"} <= set(summary["by_task"])


def test_profile_bound_to_another_algorithm_is_refused_before_engine_calls(tmp_path):
    engine = _engine()
    other = _profile("colocated-serial", AlgorithmSpec(kl_coef=0.1))
    driver, _ = _driver(engine, tmp_path, profile=other)
    with pytest.raises(DriverError, match="bound to algorithm"):
        driver.run()
    assert engine.calls == []
    unbound, _ = _driver(_engine(), tmp_path, "u.jsonl", profile=_profile("colocated-serial", False))
    with pytest.raises(DriverError, match="not bound"):
        unbound.run()


def test_overlap_mode_and_placement_mismatch_are_refused(tmp_path):
    overlap = _profile("partitioned-overlap")
    engine = _engine(placement_kind="fixed-partition")
    driver, _ = _driver(engine, tmp_path, profile=overlap)
    with pytest.raises(DriverError, match="2.3"):
        driver.run()
    engine = _engine()  # colocated placement
    driver, _ = _driver(engine, tmp_path, "m.jsonl", profile=_profile("partitioned-serial"))
    with pytest.raises(DriverError, match="needs placement 'fixed-partition'"):
        driver.run()


def test_partitioned_serial_keeps_the_per_group_policy_token_check(tmp_path):
    engine = _engine(placement_kind="fixed-partition", stale_token_rounds={1})
    driver, trained = _driver(engine, tmp_path, profile=_profile("partitioned-serial"))
    with pytest.raises(PolicyIdentityError):
        driver.run()
    assert len(trained) == 1  # round 1 never trained


def test_strict_profile_refuses_generation_on_an_older_published_policy(tmp_path):
    engine = _engine(placement_kind="fixed-partition")
    driver, _ = _driver(engine, tmp_path, profile=_profile("partitioned-serial"))
    driver.handshake()
    driver.publish(engine.policy_state.export(), rollout_id=0)
    driver.trained_version = 1  # trainer applied an update that was not published
    with pytest.raises(ReadinessError, match="behind trained"):
        driver._generate(0)
    assert ("generate", 0) not in engine.calls
