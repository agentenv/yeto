"""CPU tests of the capability evaluation suite (tools/eval): matrix, verdict
status, evidence loading, a judge on synthetic data, Miles argument building,
and the report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.eval import __main__ as cli
from tools.eval import cells, miles, report
from tools.eval.verdict import Verdict


def test_matrix_covers_every_capability_and_backend():
    assert {(c.capability, c.backend) for c in cells.CELLS.values()} == {
        (cap, b) for cap in cells.CAPABILITIES for b in cells.BACKENDS}
    assert [c.key for c in cells.CELLS.values() if not c.supported] == ["E2-verl"]
    with pytest.raises(SystemExit):
        cells.get("E9", "verl")


@pytest.mark.parametrize("supported,passed,status", [
    (True, True, "pass"), (True, False, "fail"), (False, False, "expected_fail"), (False, True, "unexpected_pass")])
def test_verdict_status(supported, passed, status):
    v = Verdict("E1", "verl", "r", supported)
    v.check("a", "a", True)
    v.check("b", "b", passed)
    assert v.status == status
    assert Verdict("E1", "verl", "r", True).status == "error"  # no criteria


def _island_run(tmp: Path, rounds: int, loss: float = 0.1) -> Path:
    run = tmp / "run"
    comp = run / "raw" / "run" / "island-c"
    (comp / "yeto-output").mkdir(parents=True)
    cell = cells.get("E1", "verl")
    (run / "cell.json").write_text(json.dumps({"capability": "E1", "backend": "verl", "run_id": "run",
                                               "config": cell.config, "code_sha": "x"}))
    (comp / "meta.json").write_text(json.dumps({"rc": 0, "gpu": ["NVIDIA A10"], "seconds": 100.0}))
    rows = []
    for r in range(rounds):
        rows.append({"event": "rl_round_trained", "rollout_id": r, "time_unix": 10.0 + r})
        rows.append({"event": "rl_local_round", "loss": loss, "grad_norm": 0.2, "reward_mean": 0.5,
                     "base_policy_version": r, "time_unix": 10.5 + r})
    rows.append({"event": "rl_verl_island_done", "rounds_completed": rounds, "time_unix": 99.0})
    (comp / "yeto-output" / "rl-island-0.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    return run


def test_e1_judge_passes_and_reports(tmp_path):
    run = _island_run(tmp_path, rounds=4)
    verdict = cli.judge(run)
    assert verdict.status == "pass", [c for c in verdict.criteria if not c.passed]
    assert verdict.run["card_seconds"] == 100.0
    verdict.write(run)
    md = report.markdown([run], title="t")
    assert "| E1 island rollout + trainer | not run | PASS" in md
    assert "<html>" in report.html([run], title="t")


def test_e1_judge_fails_on_missing_round_and_nan(tmp_path):
    run = _island_run(tmp_path, rounds=3, loss=float("nan"))
    verdict = cli.judge(run)
    failed = {c.id for c in verdict.criteria if not c.passed}
    assert verdict.status == "fail" and {"rounds", "finite"} <= failed


def test_miles_launcher_args():
    run = "eval-x"
    e2 = miles.launcher_args(cells.get("E2", "miles"), run)
    assert "--rl-island-sample-role 0:sampler --rl-island-sample-role 1:sampler" in e2["ARGS"]
    assert "--rl-island-sample-role 2:trainer" in e2["ARGS"] and "modal:1xa10g,modal:1xa10g,modal:1xa10g" in e2["ARGS"]
    assert f"modal-volume://{miles.STORE_VOLUME}/{run}" in e2["ARGS"]
    assert (e2["RECLAIM_LABEL"], e2["RECLAIM_ROUND"]) == ("l0", "1")
    e4 = miles.launcher_args(cells.get("E4", "miles"), run)
    assert "--rl-sample-role trainer" in e4["ARGS"] and "--rl-sample-role sampler" in e4["SARGS"]
    assert f"--cluster-prefix {run}-s" in e4["SARGS"] and f"/{run} " in e4["SARGS"]  # same store
    assert "--rl-trainer-scheduler s1" in miles.launcher_args(cells.get("E5", "miles"), run)["ARGS"]
    with pytest.raises(ValueError):
        miles.launcher_args(cells.get("E3", "miles"), run)


def test_run_refuses_unregistered_run_id(tmp_path, monkeypatch):
    ledger = tmp_path / "gpu-spend.md"
    ledger.write_text("| other |\n")
    monkeypatch.setattr(cli, "LEDGER", ledger)
    with pytest.raises(SystemExit, match="not pre-registered"):
        cli._preflight("eval-unregistered")


class _E4Evidence:
    """Just the event reads judge_e4 makes (one Miles trainer, one sampler)."""
    backend = "miles"

    def __init__(self, trainer_events, sampler_events):
        self.events = {"trainer": trainer_events, "sampler-0": sampler_events}

    def trainers(self):
        return ["trainer"]

    def samplers(self):
        return ["sampler-0"]

    def of(self, component, name):
        return [e for e in self.events.get(component, []) if e["event"] == name]


@pytest.mark.parametrize("stale,consumed,checks", [
    ((0, 16, 0, 0), 80, {"stale_event": True, "stale_cross_check": True, "over_age_drop": True}),
    ((0, 15, 0, 0), 80, {"stale_event": True, "stale_cross_check": False, "over_age_drop": True}),
    ((), 80, {"stale_event": False, "stale_cross_check": False, "over_age_drop": False}),
])
def test_e4_miles_reads_the_stale_event_and_cross_checks_consumed_minus_trained(monkeypatch, stale, consumed,
                                                                                checks):
    from tools.eval import judges

    for name in ("_rc_ok", "_gpu"):
        monkeypatch.setattr(judges, name, lambda *a, **kw: None)
    monkeypatch.setattr(judges, "_round_metrics", lambda ev, t: {})
    rounds = 4
    trainer = [{"event": "rl_round_trained", "rollout_id": r, "mismatch": {"tis": 1.0, "tis_abs": 0.01}}
               for r in range(rounds)]
    trainer += [{"event": "rl_local_round", "base_policy_version": r} for r in range(rounds)]
    trainer += [{"event": "rl_round_labels", "rollout_id": r, "rl/groups": 16} for r in range(rounds)]
    trainer += [{"event": "rl_buffer_stale_skipped", "rollout_id": r, "stale_groups": n, "max_policy_age": 1}
                for r, n in enumerate(stale)]
    trainer += [{"event": "rl_policy_released", "policy_version": rounds, "buffer_selected": consumed}]
    monkeypatch.setattr(judges, "_trained_rounds", lambda ev, t: list(range(rounds)))
    sampler = [{"event": "rl_sampler_policy_loaded", "policy_version": v} for v in (0, 1)]
    v = Verdict("E4", "miles", "r", True)
    judges.judge_e4(_E4Evidence(trainer, sampler), v, {"rounds": rounds, "max_policy_age": 1})
    assert {c.id: c.passed for c in v.criteria if c.id in checks} == checks
