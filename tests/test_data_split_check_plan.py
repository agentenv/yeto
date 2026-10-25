"""tools/eval/data_split_check --plan: trainer islands fed by sampler islands (S27 E2,
s27-cc-e2-a). Their rl_driver_start names no data slot; the plan gives each trainer the
--rl-sampler-data-slot of the samplers whose feeds_trainer is that trainer."""

from __future__ import annotations

import json

import pytest

from tools.eval.data_split_check import check, main, read_records, trainer_slots_from_plan


def _request(learner, role, *, feeds=None, slot=None):
    command = "python -m yeto.rl.adapters.miles.island_entry --learner-id 0"
    if slot is not None:
        command += f" --rl-sample-role sampler --rl-sampler-data-slot {slot}"
    request = {"learner_id": learner, "sample_role": role, "learner_command": command}
    if role == "sampler":
        request["feeds_trainer"] = feeds
    return request


def _plan(*samplers):
    """Two trainers (learner ids 0, 1) and the given (learner id, feeds, slot) samplers."""
    return {"island_requests": [_request(0, "trainer"), _request(1, "trainer"),
                                *(_request(n, "sampler", feeds=f, slot=s) for n, f, s in samplers)]}


E2 = _plan((2, 0, "0/2"), (3, 1, "1/2"))


def _tape(tmp_path, name, island, records):
    path = tmp_path / name
    path.write_text("".join(json.dumps({"island_id": island, "time_unix": t, **r}) + "\n" for t, r in records))
    return path


def _trainer(tmp_path, name, island, rows, *, slot=None):
    start = {"event": "rl_driver_start", "data_slot": slot}
    records = [(1, start)]
    records += [(2, {"event": "rl_trajectory_reward", "rollout_id": 0, "data_row": row}) for row in rows]
    records += [(3, {"event": "rl_full_parameter_round", "policy_version": 1})]
    return _tape(tmp_path, name, island, records)


def _sampler(tmp_path, name):
    # a sampler island is learner 0 of 1 and runs no driver
    return _tape(tmp_path, name, 0, [(1, {"event": "rl_sampler_start", "store": "/s"})])


def test_plan_names_the_sampler_slots_of_each_trainer():
    assert trainer_slots_from_plan(E2) == ({0: ["0/2"], 1: ["1/2"]}, [])
    lanes = _plan((2, 0, "0/4"), (3, 1, "1/4"), (4, 0, "2/4"), (5, 1, "3/4"))
    assert trainer_slots_from_plan(lanes) == ({0: ["0/4", "2/4"], 1: ["1/4", "3/4"]}, [])
    broken = _plan((2, None, "0/2"), (3, 1, None))
    assert trainer_slots_from_plan(broken) == ({}, ["plan: sampler island 2 feeds no single trainer (feeds_trainer)",
                                                    "plan: sampler island 3 has no --rl-sampler-data-slot"])


def test_fed_trainers_pass_with_the_plan_and_fail_without_it(tmp_path):
    tapes = [_trainer(tmp_path, "t0.jsonl", 0, [0, 2, 4]), _trainer(tmp_path, "t1.jsonl", 1, [1, 3]),
             _sampler(tmp_path, "s2.jsonl"), _sampler(tmp_path, "s3.jsonl")]
    report = check(read_records(tapes), plan=E2)
    assert report["ok"], report["problems"]
    assert report["slots"] == {"0": "0/2", "1": "1/2"}
    assert report["sampler_slots_of_trainer"] == {"0": ["0/2"], "1": ["1/2"]}
    assert report["rows_per_island"] == {"0": 3, "1": 2}
    plain = check(read_records(tapes))
    assert not plain["ok"] and "sampler_slots_of_trainer" not in plain
    assert plain["problems"] == ["slots: islands do not hold distinct slots of one count: "
                                 "{0: {'None'}, 1: {'None'}} (a trainer fed by sampler islands "
                                 "names no slot: pass --plan PLAN.json)"]


def test_a_fed_trainer_with_a_foreign_row_fails_own_slot_and_disjoint(tmp_path):
    tapes = [_trainer(tmp_path, "t0.jsonl", 0, [0, 2, 3]), _trainer(tmp_path, "t1.jsonl", 1, [1, 3])]
    problems = check(read_records(tapes), plan=E2)["problems"]
    assert "own_slot: island 0 trained rows outside slot 0/2: [3]" in problems
    assert any(p.startswith("disjoint: 1 rows") for p in problems)


def test_a_trainer_fed_by_two_samplers_owns_both_slots(tmp_path):
    lanes = _plan((2, 0, "0/4"), (3, 1, "1/4"), (4, 0, "2/4"), (5, 1, "3/4"))
    tapes = [_trainer(tmp_path, "t0.jsonl", 0, [0, 2, 4, 6]), _trainer(tmp_path, "t1.jsonl", 1, [1, 3, 5])]
    report = check(read_records(tapes), plan=lanes)
    assert report["ok"], report["problems"]
    assert report["slots"] == {"0": "0/4,2/4", "1": "1/4,3/4"}
    bad = [_trainer(tmp_path, "b0.jsonl", 0, [0, 1]), tapes[1]]
    assert "own_slot: island 0 trained rows outside slot 0/4,2/4: [1]" in check(read_records(bad), plan=lanes)["problems"]


def test_a_slot_named_by_a_fed_trainer_must_be_its_samplers(tmp_path):
    tapes = [_trainer(tmp_path, "t0.jsonl", 0, [0], slot="1/2"), _trainer(tmp_path, "t1.jsonl", 1, [1])]
    problems = check(read_records(tapes), plan=E2)["problems"]
    assert "slots: island 0 names data slot 1/2, but the plan feeds it from sampler slots ['0/2']" in problems


def test_islands_that_read_their_own_data_are_unchanged_by_a_plan(tmp_path):
    plan = {"island_requests": [_request(0, "both"), _request(1, "both")]}
    tapes = [_trainer(tmp_path, "i0.jsonl", 0, [0, 2], slot="0/2"), _trainer(tmp_path, "i1.jsonl", 1, [1], slot="1/2")]
    assert check(read_records(tapes), plan=plan)["ok"]
    assert check(read_records(tapes))["ok"]


def test_cli_reads_the_plan(tmp_path, capsys):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(E2))
    tapes = [str(_trainer(tmp_path, "t0.jsonl", 0, [0])), str(_trainer(tmp_path, "t1.jsonl", 1, [1]))]
    assert main(["--plan", str(plan), *tapes]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert main(tapes) == 1
    with pytest.raises(SystemExit):
        main(["--plan", str(plan)])
