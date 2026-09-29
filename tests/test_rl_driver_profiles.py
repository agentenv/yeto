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


R0_TAPE = json.loads(
    (__import__("pathlib").Path(__file__).parent / "data" / "r0_driver_tape_a50e9d2.json").read_text()
)["tapes"]


@pytest.mark.parametrize("observe", [False, True])
def test_bound_profile_matches_the_recorded_r0_tape(tmp_path, observe):
    """1.7: the production path always builds a profile; with observe=False the
    tape equals the one recorded from the R0 driver at a50e9d2; with observe=True
    it equals it after removing the added events and start-event labels."""
    driver, _ = _driver(_engine(), tmp_path, profile=_profile("colocated-serial"),
                        observe=observe)
    driver.run()
    events = _events(tmp_path / "events.jsonl")
    if observe:
        assert {e["event"] for e in events} >= {"rl_timeline_span", "rl_round_labels"}
        events = [e for e in events if e["event"] not in NEW_EVENTS]
        for e in events:
            if e["event"] == "rl_driver_start":
                e.pop("profile_hash"), e.pop("config_epoch")
    assert _strip(events) == R0_TAPE["colocated"]


def test_profile_none_matches_the_recorded_r0_tape(tmp_path):
    for kind in ("colocated", "fixed-partition"):
        driver, _ = _driver(_engine(placement_kind=kind), tmp_path, f"{kind}.jsonl",
                            capabilities=fake_capabilities())
        driver.run()
        assert _strip(_events(tmp_path / f"{kind}.jsonl")) == R0_TAPE[kind]


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
    # F5: the fake engine reports no filtered/carried_over counts -> None, not aborted
    assert {(e["rl/filtered_groups"], e["rl/carried_over_groups"]) for e in labels} == {(None, None)}
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


def test_ports_entry_builds_a_bound_profile_and_preflights_before_gpu():
    from types import SimpleNamespace

    from yeto.rl.engine.capabilities import CapabilityMismatch
    from yeto.rl.engine.execution_profile import ProfileError
    from yeto.rl.engine.miles_adapter import entry
    from yeto.rl.engine.miles_adapter.placement import PlacementRequest

    args = SimpleNamespace(rollout_batch_size=4, n_samples_per_prompt=8, num_steps_per_rollout=1,
                           yeto_rl_sync_preset="strict-avg")
    part = SimpleNamespace(placement=PlacementRequest("fixed-partition", 2, 2, 1), argv=("x",))
    spec = AlgorithmSpec()
    with pytest.raises(ProfileError, match="expected AlgorithmSpec hash"):
        entry.execution_profile_for(args, part, spec, yeto_policy_sync=True)
    profile = entry.execution_profile_for(args, part, spec, yeto_policy_sync=True,
                                          expected_sha256=spec.sha256())
    assert profile.execution_mode == "partitioned-serial"
    assert profile.outer_protocol == "strict-avg"
    assert profile.algorithm_spec_sha256 == spec.sha256()
    r0 = entry.miles_capabilities("sha256:" + "1" * 64)
    with pytest.raises(CapabilityMismatch):
        entry.preflight(profile, spec, r0)  # R0 declaration: colocated only
    caps = entry.with_partitioned_serial(r0)
    entry.preflight(profile, spec, caps)
    assert caps.partitioned_driver and r0.advantage_estimators == caps.advantage_estimators
    # F2: the launcher intended another algorithm than the runtime built
    launcher_hash = AlgorithmSpec(kl_coef=0.1).sha256()
    mismatched = entry.execution_profile_for(args, part, spec, yeto_policy_sync=True,
                                             expected_sha256=launcher_hash)
    with pytest.raises(ProfileError, match="bound to algorithm"):
        entry.preflight(mismatched, spec, caps)
    args.yeto_rl_expected_algorithm_sha256 = launcher_hash
    assert entry.expected_algorithm_sha256(args, environ={}) == launcher_hash
    del args.yeto_rl_expected_algorithm_sha256
    assert entry.expected_algorithm_sha256(args, environ={entry.EXPECTED_ALGORITHM_ENV: "ab"}) == "ab"
    assert entry.expected_algorithm_sha256(args, environ={}) is None
    colo = SimpleNamespace(placement=PlacementRequest("colocated", 2, 2, 1), argv=("x",))
    p = entry.execution_profile_for(args, colo, spec, yeto_policy_sync=False)
    assert (p.execution_mode, p.outer_protocol) == ("colocated-serial", "none")
    entry.preflight(p, spec, r0)


@pytest.mark.parametrize("mode,kind", [("colocated-serial", "colocated"),
                                       ("partitioned-serial", "fixed-partition")])
def test_fewer_groups_than_rollout_batch_size_still_trains(tmp_path, mode, kind):
    """Review F3: partial rollouts / filtering may return fewer groups than
    rollout_batch_size; like R0, the round trains instead of ReadinessError."""
    engine = _engine(placement_kind=kind)
    driver, trained = _driver(engine, tmp_path, profile=_profile(mode, groups_per_batch=100))
    driver.run()
    assert len(trained) == 3
