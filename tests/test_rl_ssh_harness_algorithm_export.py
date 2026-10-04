"""SSH-harness export carries the ports run's algorithm (P0 D9 interface)."""

import json

import pytest

from tests.test_rl_ssh_harness import _decoupled_oracle_fixture, _write_plan
from yeto.rl import ssh_harness
from yeto.rl.engine.algorithm import AlgorithmSpec, LossSpec
from yeto.rl.ssh_harness import HarnessError

SPEC = AlgorithmSpec(loss=LossSpec(eps_clip_high=0.28))


def _event(spec=SPEC, mechanisms=None, **extra):
    event = {
        "event": "rl_engine_selected",
        "rl_engine": "ports",
        "rl/algorithm_spec": spec.canonical_json(),
        "rl/algorithm_spec_sha256": spec.sha256(),
    }
    if mechanisms:
        event["rl/unverified_mechanisms"] = mechanisms
    event.update(extra)
    return event


def _tapes(artifacts, per_island):
    for learner_id, events in enumerate(per_island):
        path = artifacts / f"island-{learner_id}" / "output" / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(e) + "\n" for e in [{"event": "other"}, *events]))


def _plan(engine="ports", islands=2):
    return {"rl_engine": engine, "islands": [{"hosts": ["h"]}] * islands}


def test_legacy_runs_export_no_algorithm(tmp_path):
    assert ssh_harness._run_algorithm_provenance(_plan("legacy"), tmp_path) == (None, ())


def test_ports_run_algorithm_comes_from_the_island_events(tmp_path):
    _tapes(tmp_path, [[_event(mechanisms=["m"])], [_event(mechanisms=["m"]), _event(mechanisms=["m"])]])
    spec, mechanisms = ssh_harness._run_algorithm_provenance(_plan(), tmp_path)
    assert spec == SPEC.canonical_json() and mechanisms == ("m",)


@pytest.mark.parametrize(
    "per_island, match",
    [
        ([[_event()], []], "no rl_engine_selected"),
        ([[_event()], [_event(AlgorithmSpec())]], "disagree"),
        ([[_event()], [_event(mechanisms=["m"])]], "disagree"),
        ([[_event()], [_event(**{"rl/algorithm_spec_sha256": "0" * 64})]], "does not match"),
        ([[_event()], [{"event": "rl_engine_selected"}]], "incomplete"),
    ],
)
def test_ports_export_fails_closed(tmp_path, per_island, match):
    _tapes(tmp_path, per_island)
    with pytest.raises(HarnessError, match=match):
        ssh_harness._run_algorithm_provenance(_plan(), tmp_path)


@pytest.mark.parametrize("engine", ["ports", "legacy"])
def test_verify_passes_the_algorithm_to_export(tmp_path, monkeypatch, engine):
    plan, checkpoint, final_hash = _decoupled_oracle_fixture(tmp_path / "fixture")
    if engine == "ports":
        plan["rl_engine"] = "ports"  # legacy plans have no rl_engine key
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, plan)
    artifacts = tmp_path / "artifacts"
    for learner_id, island in enumerate(plan["islands"]):
        for node_id in range(len(island["hosts"])):
            path = artifacts / f"island-{learner_id}" / f"node-{node_id}.inspect.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"Status": "exited", "ExitCode": 0, "OOMKilled": False, "RestartCount": 0}))
    if engine == "ports":
        _tapes(artifacts, [[_event()] for _ in plan["islands"]])

    import yeto.export as checkpoint_export
    from yeto.rl import export as rl_export

    monkeypatch.setattr(checkpoint_export, "parse_checkpoint", lambda _path: checkpoint)
    monkeypatch.setattr(ssh_harness, "_verify_decoupled", lambda *_a: final_hash, raising=False)
    exported = {}
    monkeypatch.setattr(rl_export, "export_rl_checkpoint", lambda *_a, **kw: exported.update(kw))
    ssh_harness.verify(plan_path, str(tmp_path / "adapter"))
    assert exported["rl_engine"] == engine
    if engine == "ports":
        assert exported["algorithm_spec"] == SPEC.canonical_json()
    else:
        assert exported["algorithm_spec"] is None
    assert exported["unverified_mechanisms"] == ()
