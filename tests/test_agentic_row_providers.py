"""Each agentic data row's own sandbox provider (its task mapping overrides the spec's
``agentic.task_defaults``) is checked against the pool's before anything is ordered: the
sampler image carries the pool provider's SDK only, so a row that asks for another
provider fails at its first import on every node (``pool --dry-run`` and ``yeto run
up``'s fleet preflight; nothing is printed when the rows agree or are not readable here)."""

from __future__ import annotations

import json
from pathlib import Path

from tests.test_fleet_agentic_sandbox import _fleet_cfg, _write_cfg, fleet_files  # noqa: F401 (fixture)
from yeto.launch import agentic_rows
from yeto.rl import up

SPEC = {"prompts": {"path": "/mnt/yeto-models/data/tb2/train.jsonl", "input_key": "prompt", "task_key": "metadata",
                    "format": "agentic", "groups_per_round": 1},
        "agentic": {"task_defaults": {"provider": "agentenv", "agent": "pi"}}}


def _rows(path: Path, providers) -> Path:
    path.write_text("".join(json.dumps({"prompt": f"t{i}", "metadata": {"instance_id": f"t{i}", "agent_name": "pi",
                                                                         **({} if p is None else {"provider": p})}})
                            + "\n" for i, p in enumerate(providers)))
    return path


def test_rows_that_override_the_pools_provider_are_a_problem_naming_rows_and_module(tmp_path):
    prompts = _rows(tmp_path / "train.jsonl", ["boat", "agentenv", None, "boat"])
    (problem,) = agentic_rows.pool_rows_problems({"prompts_file": str(prompts)}, SPEC)
    assert "2 of 4 row(s)" in problem and "provider 'boat'" in problem and "row(s) 1, 4" in problem
    assert "'agentenv'" in problem and "module boat_sdk" in problem and "metadata.provider" in problem


def test_rows_that_agree_or_cannot_be_read_here_are_silent(tmp_path):
    prompts = _rows(tmp_path / "train.jsonl", ["agentenv", None, "agentenv"])
    assert agentic_rows.pool_rows_problems({"prompts_file": str(prompts)}, SPEC) == []
    assert agentic_rows.pool_rows_problems({}, SPEC) == []  # the volume path: not on this machine
    assert agentic_rows.pool_rows_problems({"prompts_file": str(prompts)}, {"prompts": SPEC["prompts"]}) == []  # not agentic


def test_the_pools_provider_falls_back_to_setup_env_then_remote_docker():
    assert agentic_rows.pool_provider({"agentic": {"setup": {"provider": "boat"}}}) == "boat"
    assert agentic_rows.pool_provider({"agentic": {}}, {"HARBOR_ENV_TYPE": "agentenv"}) == "agentenv"
    assert agentic_rows.pool_provider({"agentic": {}}) == "remote-docker"
    assert agentic_rows.pool_provider({}) is None


def test_the_run_preflight_reads_the_trainers_local_data_of_the_same_name(tmp_path):
    """P5b g2: the pool reads its prompts from a Modal volume; the trainer's local --data is the
    same file. Its rows asked for boat while the pool ran AgentENV: refused before ordering."""
    data = _rows(tmp_path / "train.jsonl", ["boat"] * 3)
    pools = [("i0-vllm", {"provider": {"kind": "modal"}}, SPEC)]
    (problem,) = agentic_rows.run_rows_problems(pools, data_files=[str(data)])
    assert problem.startswith("i0-vllm: agentic rows: 3 of 3 row(s)") and "pool i0-vllm's provider" in problem
    other = _rows(tmp_path / "other.jsonl", ["boat"])
    assert agentic_rows.run_rows_problems(pools, data_files=[str(other)]) == []  # another file
    (tmp_path / "fixed").mkdir()
    fixed = _rows(tmp_path / "fixed" / "train.jsonl", ["agentenv"])
    assert agentic_rows.run_rows_problems(pools, data_files=[str(fixed)]) == []


def test_pool_dry_run_refuses_rows_on_another_provider(tmp_path, monkeypatch, capsys):
    """``nc pool --dry-run`` of a Modal pool whose prompts_file rows ask for boat under an
    AgentENV spec: a problem, nothing ordered; the same pool with agreeing rows prints no line of it."""
    from test_rollout_modal_provider import _cfg
    from yeto.launch.nc import pool_run

    monkeypatch.setattr(pool_run, "check_sampler_spec", lambda cfg, pc: json.loads(json.dumps(SPEC)))
    prompts = _rows(tmp_path / "train.jsonl", ["boat", "agentenv"])
    assert pool_run.dry_run(_cfg(tmp_path, prompts_file=str(prompts)), "pool.json") == 1
    out = capsys.readouterr().out
    assert "problem: agentic rows: 1 of 2 row(s)" in out and "module boat_sdk" in out
    _rows(prompts, ["agentenv", None])
    pool_run.dry_run(_cfg(tmp_path, prompts_file=str(prompts)), "pool.json")
    assert "agentic rows" not in capsys.readouterr().out


def test_the_fleet_preflight_refuses_rows_on_another_provider(fleet_files, monkeypatch):  # noqa: F811
    """``yeto run up``: the fleet preflight reads each agentic pool's rows (its prompts.path
    here) and refuses one that asks for another provider than the pool's."""
    spec = json.loads((fleet_files / "agentic.json").read_text())
    rows = _rows(fleet_files / "p.jsonl", ["boat", None])
    spec["prompts"]["path"] = str(rows)
    (fleet_files / "agentic.json").write_text(json.dumps(spec))
    run = up.Run(up.load_config(_write_cfg(fleet_files, _fleet_cfg(fleet_files))))
    problems = run.agentic_row_problems([])
    assert len(problems) == 2 and all("provider 'boat'" in p and "'remote-docker'" in p for p in problems)
    assert {p.split(":")[0] for p in problems} == {"i0-agent", "i1-agent"}
    verdict = run.fleet_preflight() or ""
    assert "provider 'boat'" in verdict
    _rows(fleet_files / "p.jsonl", ["remote-docker", None])
    assert run.agentic_row_problems([]) == []
