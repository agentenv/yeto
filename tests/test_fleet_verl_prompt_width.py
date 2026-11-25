"""The prompt row width of a verl fleet's trainers (rl-fleet-verl-islands 4.1).

A verl buffer trainer sizes its prompt rows (``data.max_prompt_length``) to the widest
first prompt its pools write (their sampler spec's ``agentic.max_prompt_len``), and
refuses at its start rows that do not fit its ``--seq-len`` next to a full response
(``buffer_feed.buffer_prompt_width``). The fleet compile takes each island's width from
its pools and passes it as ``--rl-island-buffer-max-prompt-len K:N``; the preflight
calls the same function and refuses, naming the island and its pools, before anything
is ordered; the launch forwards the width to the island as ``--rl-buffer-max-prompt-len``.
A Miles fleet's plan, argv and plan.json do not change. No cloud, no GPU.
"""

from __future__ import annotations

import dataclasses
import json
import shlex

import pytest

from test_fleet_verl_admission import VERL_ARGV, _run
from test_fleet_verl_island_launches import _island_env  # noqa: F401
from test_fleet_verl_island_launches import _run as _example_run
from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from tests.test_run_up_fleet import TRAINER_ARGV, files, no_subprocess  # noqa: F401
from yeto.rl import fleet
from yeto.rl import island_launches as il
from yeto.rl.engine.sample_roles import check_island_prompt_width

FLAG = "--rl-island-buffer-max-prompt-len"


def _bound(files, vllm: int | None, sglang: int | None) -> None:
    """The fixture's two sampler specs with their prompt bounds (None: no bound)."""
    for name, width in (("vllm", vllm), ("sglang", sglang)):
        path = files / f"{name}.json"
        spec = json.loads(path.read_text())
        spec.pop("agentic", None)
        if width is not None:
            spec["agentic"] = {"max_prompt_len": width}
        path.write_text(json.dumps(spec))


def _values(argv: list[str], flag: str) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv) if a == flag]


def test_each_verl_island_gets_its_pools_widest_prompt(files):
    _bound(files, vllm=1536, sglang=1024)
    run = _run(files, VERL_ARGV)
    assert run.fleet_problem() is None
    plan = run.fleet_plan()
    # island 0 (east): vllm 1536 and sglang 1024 -> 1536; island 1 (west): sglang only -> 1024
    assert [i.buffer_max_prompt_len for i in plan.islands] == [1536, 1024]
    argv = [str(a) for a in run.cfg["trainer"]["argv"]]
    assert _values(argv, FLAG) == ["0:1536", "1:1024"]
    # the plan keeps it: plan.json and back, and each per-island launch carries its own as 0:N
    again = fleet.FleetPlan.from_json(json.loads(json.dumps(plan.to_json())))
    assert [i.buffer_max_prompt_len for i in again.islands] == [1536, 1024]
    per_island = dataclasses.replace(plan, max_islands=4)
    for island in plan.islands:
        launch = il.island_argv(VERL_ARGV, island, per_island, learners=2)
        assert _values(launch, FLAG) == [f"0:{island.buffer_max_prompt_len}"]


def test_an_island_with_an_unbounded_pool_takes_its_sequence_budget(files):
    _bound(files, vllm=1536, sglang=None)
    run = _run(files, VERL_ARGV)
    assert run.fleet_problem() is None
    plan = run.fleet_plan()
    assert [i.buffer_max_prompt_len for i in plan.islands] == [None, None]
    assert FLAG not in [str(a) for a in run.cfg["trainer"]["argv"]]
    assert all("buffer_max_prompt_len" not in i for i in plan.to_json()["islands"])


def test_a_prompt_bound_over_the_sequence_budget_is_refused_naming_the_island_and_pools(files):
    _bound(files, vllm=3000, sglang=1024)
    problem = _run(files, VERL_ARGV).fleet_problem()
    assert problem == ("island 0 (east), pools i0-vllm, i0-sglang: pool max_prompt_len 3000 + "
                       "--rollout-max-response-len 2048 = 5048 tokens per row > the trainer's --seq-len 4096 "
                       "(its memory plan): raise --seq-len to 5048, or lower the pools' max_prompt_len")
    # no bound anywhere: the rows take the sequence budget, refused when there is none
    _bound(files, vllm=None, sglang=None)
    problem = _run(files, [*VERL_ARGV, "--seq-len", "2048"]).fleet_problem()
    assert problem.startswith("island 0 (east), pools i0-vllm, i0-sglang: --seq-len 2048 leaves no prompt tokens")


def test_a_miles_fleet_plan_is_unchanged_by_its_pools_prompt_bound(files):
    _bound(files, vllm=1536, sglang=1024)
    bounded = _run(files, TRAINER_ARGV)
    assert bounded.fleet_problem() is None
    plan = bounded.fleet_plan()
    assert [i.buffer_max_prompt_len for i in plan.islands] == [None, None]
    argv = [str(a) for a in bounded.cfg["trainer"]["argv"]]
    assert FLAG not in argv
    _bound(files, vllm=None, sglang=None)
    plain = _run(files, TRAINER_ARGV)
    assert ([str(a) for a in plain.cfg["trainer"]["argv"]] == argv
            and json.dumps(plain.fleet_plan().to_json(), sort_keys=True) == json.dumps(plan.to_json(), sort_keys=True))
    assert plan.lines() == plain.fleet_plan().lines()


def _launch_args(*extra: str, backend: str = "verl"):
    import argparse

    return argparse.Namespace(rl_backend=backend, seq_len=4096, rollout_max_response_len=2048,
                              rl_island_buffer_max_prompt_len=list(extra) or None)


def test_the_launch_checks_each_trainers_width_and_refuses_it_on_miles():
    check_island_prompt_width(_launch_args("0:1536", "1:2048"), 2)
    check_island_prompt_width(_launch_args(), 2)  # nothing given: nothing checked
    with pytest.raises(ValueError, match=r"--rl-island-buffer-max-prompt-len 1:3000 \(trainer 1\): pool "
                                         r"max_prompt_len 3000 .* raise --seq-len to 5048"):
        check_island_prompt_width(_launch_args("0:1536", "1:3000"), 2)
    with pytest.raises(ValueError, match=r"must be an integer >= 1"):
        check_island_prompt_width(_launch_args("0:wide"), 2)
    with pytest.raises(ValueError, match=r"--rl-backend verl\); --rl-backend miles does not"):
        check_island_prompt_width(_launch_args("0:1536", backend="miles"), 2)


def test_the_launch_forwards_each_trainers_width_to_its_verl_island(tmp_path, monkeypatch):
    from yeto.cli import parse_args
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    run, _ = _example_run(tmp_path, monkeypatch, per_island=False)
    argv = [str(a) for a in run.cfg["trainer"]["argv"]]
    args = parse_args([*argv, FLAG, "1:1536"])
    args.model_revision, args.data_revision = "a" * 40, "b" * 40
    args.source_sha256, args.reward_sha256 = "c" * 64, "d" * 64
    args.controller = "local"
    _prepare_rl_args(args)
    trainers = [i for i in dry_run_plan(args)["island_requests"] if i.get("sample_role") == "trainer"]
    widths = []
    for island in trainers:
        tokens = shlex.split(island["learner_command"].split("-m yeto.rl.adapters.verl.island_entry", 1)[1])
        widths.append(_values(tokens, "--rl-buffer-max-prompt-len"))
    assert widths == [[], ["1536"]]
    # over the example's budget (--seq-len 4096, responses 2048): refused before anything is ordered
    args = parse_args([*argv, FLAG, "0:3000"])
    args.model_revision, args.data_revision = "a" * 40, "b" * 40
    args.source_sha256, args.reward_sha256 = "c" * 64, "d" * 64
    args.controller = "local"
    with pytest.raises(ValueError, match=r"--rl-island-buffer-max-prompt-len 0:3000 \(trainer 0\)"):
        _prepare_rl_args(args)
        dry_run_plan(args)


def test_an_island_added_at_run_time_is_sized_and_checked_like_a_compiled_one(tmp_path, monkeypatch):
    """``yeto fleet island add --dry-run``: the new island takes its pools' widest prompt
    (``fleet_islands.compile_island``), and its launch's dry run refuses one over the
    trainer's budget (the example: --seq-len 4096, responses 2048) before it is written."""
    from yeto.rl import fleet_islands

    run, _ = _example_run(tmp_path, monkeypatch)
    plan = run.fleet_plan()

    def added(width: int):
        spec = tmp_path / f"spec.{width}.json"
        spec.write_text(json.dumps({"sampler_id": "x", "sampling": {"n": 8}, "agentic": {"max_prompt_len": width}}))
        entry = dict(run.cfg["islands"][1]["samplers"][0], name="sglang-north", spec=str(spec))
        member, _pools = fleet_islands.compile_island(run.cfg, plan, k=2, name="north", gpu="modal:1x2xh100",
                                                      usd_per_h=8.0, samplers=[entry])
        assert _values(member["argv"], FLAG) == [f"0:{width}"]
        return member

    def prepare(args):  # the launch's own preparation, its revisions given (no Hub here)
        from yeto.launcher import _prepare_rl_args

        args.model_revision, args.data_revision = "a" * 40, "b" * 40
        args.source_sha256, args.reward_sha256 = "c" * 64, "d" * 64
        _prepare_rl_args(args)

    fleet_islands.launch_dry_run(added(1536), prepare=prepare)
    with pytest.raises(ValueError, match=r"--rl-island-buffer-max-prompt-len 0:3000 \(trainer 0\): pool "
                                         r"max_prompt_len 3000 \+ --rollout-max-response-len 2048"):
        fleet_islands.launch_dry_run(added(3000), prepare=prepare)
