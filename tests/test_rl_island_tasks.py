"""rl-per-island-tasks: per-island task lists (parse, check, launch wiring). No Ray."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from pathlib import Path

import pytest

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from yeto.rl.island_tasks import (
    format_island_tasks,
    parse_island_tasks,
    plan_island_tasks,
    task_count_warning,
    verify_island_data,
)

TWO_ISLANDS = "modal:1xh100,modal:1xh100"
F = "--rl-island-tasks"


def _write_rows(path, task_ids, *, nested=True):
    rows = []
    for task_id in task_ids:
        row = {"prompt": f"solve {task_id}"}
        if nested:
            row["metadata"] = {"task_id": task_id}
        else:
            row["task_id"] = task_id
        rows.append(json.dumps(row) + "\n")
    path.write_text("".join(rows), encoding="utf-8")
    return path


def _ids(prefix, start, end):
    return [f"{prefix}{n}" for n in range(start, end + 1)]


@pytest.fixture
def catalog(tmp_path):
    return _write_rows(tmp_path / "tasks.jsonl", _ids("web", 1, 20) + _ids("pwn", 1, 20))


# --- spec parsing -------------------------------------------------------------------------


def test_inline_spec_accepts_both_island_key_forms():
    spec = parse_island_tasks("l0=web1-10,pwn1-10; 1=web11-20 , pwn11-20", flag=F)
    assert spec == {0: ("web1-10", "pwn1-10"), 1: ("web11-20", "pwn11-20")}
    assert parse_island_tasks(format_island_tasks(spec), flag=F) == spec


def test_json_file_spec(tmp_path):
    path = tmp_path / "islands.json"
    path.write_text(json.dumps({"0": ["web1", "web2"], "l1": ["pwn1"]}))
    assert parse_island_tasks(str(path), flag=F) == {0: ("web1", "web2"), 1: ("pwn1",)}


@pytest.mark.parametrize("value, message", [
    ("", "empty"),
    ("/no/such/file.json", "neither"),
    ("x0=web1", "bad island key"),
    ("l0=web1;0=web2", "given twice"),
    ("l0=", "empty task list"),
])
def test_bad_inline_spec_is_refused(value, message):
    with pytest.raises(ValueError, match=message):
        parse_island_tasks(value, flag=F)


def test_json_spec_rejects_non_lists_and_separator_ids(tmp_path):
    path = tmp_path / "islands.json"
    path.write_text(json.dumps({"0": "web1"}))
    with pytest.raises(ValueError, match="list of strings"):
        parse_island_tasks(str(path), flag=F)
    path.write_text(json.dumps({"0": ["a,b"]}))
    with pytest.raises(ValueError, match="contains one of"):
        parse_island_tasks(str(path), flag=F)


# --- plan against the data ----------------------------------------------------------------


def test_plan_gives_each_island_only_its_tasks(catalog):
    spec = parse_island_tasks("l0=web1-10,pwn1-10;l1=web11-20,pwn11-20", flag=F)
    plan = plan_island_tasks(catalog, spec, [0, 1], flag=F)
    assert set(plan[0].task_ids) == set(_ids("web", 1, 10) + _ids("pwn", 1, 10))
    assert set(plan[1].task_ids) == set(_ids("web", 11, 20) + _ids("pwn", 11, 20))
    rows = [json.loads(line) for line in plan[0].content.decode().splitlines()]
    assert [row["metadata"]["task_id"] for row in rows] == list(plan[0].task_ids)
    # source order and bytes: the same inputs always give the same SHA256
    assert plan[0].sha256 == plan_island_tasks(catalog, spec, [0, 1], flag=F)[0].sha256
    assert plan[0].sha256 != plan[1].sha256
    assert task_count_warning(plan, flag=F) is None


def test_literal_task_id_wins_over_range_and_zero_padding(tmp_path):
    data = _write_rows(tmp_path / "t.jsonl", ["cve-2021-4", "x01", "x02", "x03"], nested=False)
    plan = plan_island_tasks(data, {0: ("cve-2021-4",), 1: ("x01-03",)}, [0, 1], flag=F)
    assert plan[0].task_ids == ("cve-2021-4",)
    assert plan[1].task_ids == ("x01", "x02", "x03")


@pytest.mark.parametrize("spec, islands, message", [
    ({0: ("web1",), 2: ("web2",)}, [0, 1], r"names islands \[2\]"),
    ({0: ("web1",)}, [0, 1], r"no task list for islands \[1\]"),
    ({0: ("web1",), 1: ("web99",)}, [0, 1], r"not in .*\['web99'\]"),
    ({0: ("web1-3", "web2"), 1: ("pwn1",)}, [0, 1], r"\['web2'\] twice"),
    ({0: ("web3-1",), 1: ("pwn1",)}, [0, 1], "runs backwards"),
])
def test_plan_refuses_bad_configurations(catalog, spec, islands, message):
    with pytest.raises(ValueError, match=message):
        plan_island_tasks(catalog, spec, islands, flag=F)


def test_plan_refuses_rows_without_task_id(tmp_path):
    data = tmp_path / "t.jsonl"
    data.write_text('{"metadata": {"task_id": "a"}}\n{"prompt": "no id"}\n')
    with pytest.raises(ValueError, match=":2: row has no metadata.task_id"):
        plan_island_tasks(data, {0: ("a",)}, [0], flag=F)


def test_unequal_task_counts_only_warn(catalog):
    plan = plan_island_tasks(catalog, {0: ("web1-20",), 1: ("pwn1-18",)}, [0, 1], flag=F)
    warning = task_count_warning(plan, flag=F)
    assert "island 0: 20" in warning and "island 1: 18" in warning


# --- island side --------------------------------------------------------------------------


def test_island_refuses_changed_data_and_reports_both_hashes(catalog, tmp_path):
    plan = plan_island_tasks(catalog, {0: ("web1-2",)}, [0], flag=F)
    local = tmp_path / "island.jsonl"
    local.write_bytes(plan[0].content)
    assert verify_island_data(local, plan[0].sha256.upper()) == plan[0].record()
    assert plan[0].record() == {"task_ids": ["web1", "web2"], "data_sha256": plan[0].sha256}
    local.write_bytes(plan[0].content + b'{"task_id": "web3"}\n')
    with pytest.raises(ValueError, match=f"launcher sent {plan[0].sha256}, file has "):
        verify_island_data(local, plan[0].sha256)


def test_island_entry_checks_only_with_a_data_sha256(catalog):
    from yeto.rl.adapters.miles.island_entry import _verify_island_task_data

    assert _verify_island_task_data(argparse.Namespace(data=str(catalog), data_sha256=None)) is None
    digest = hashlib.sha256(catalog.read_bytes()).hexdigest()
    record = _verify_island_task_data(argparse.Namespace(data=str(catalog), data_sha256=digest))
    assert record["data_sha256"] == digest and len(record["task_ids"]) == 40


# --- launcher -----------------------------------------------------------------------------


def _island_args(catalog, tmp_path, monkeypatch, *extra):
    from yeto import runs

    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    args = _launcher_args("ports", ("--rl-island-tasks", *extra), gpu=TWO_ISLANDS)
    args.data, args.data_revision, args.controller = str(catalog), None, "local"
    return args


def _flag(command, flag):
    tokens = shlex.split(command)
    return tokens[tokens.index(flag) + 1]


def test_dry_run_gives_each_island_its_own_data(catalog, tmp_path, monkeypatch):
    from yeto import launcher

    args = _island_args(catalog, tmp_path, monkeypatch, "l0=web1-10,pwn1-10;l1=web11-20,pwn11-20")
    launcher._prepare_rl_args(args)
    plan = launcher.dry_run_plan(args)
    islands = plan["island_requests"]
    hashes = [_flag(i["learner_command"], "--data-sha256") for i in islands]
    assert hashes[0] != hashes[1]
    for island, digest in zip(islands, hashes):
        assert island["island_tasks"]["data_sha256"] == digest
        assert len(island["island_tasks"]["task_ids"]) == 20
        written = tmp_path / "runs" / args.cluster_prefix / "island-tasks" / \
            f"island-{island['learner_id']}.jsonl"
        assert hashlib.sha256(written.read_bytes()).hexdigest() == digest
    manifest = launcher._island_tasks_manifest(args)["island_tasks"]
    assert [manifest[k]["data_sha256"] for k in ("0", "1")] == hashes


def test_island_task_mount_replaces_the_shared_data(catalog, tmp_path, monkeypatch):
    from yeto import launcher
    from yeto.datasource import learner_data_arg

    args = _island_args(catalog, tmp_path, monkeypatch, "l0=web1;l1=web2")
    launcher._prepare_rl_args(args)
    data, sha_flag, mounts = launcher.island_task_data(args, 1)
    assert sha_flag.startswith(" --data-sha256 ")
    (source,) = mounts.values()
    assert source.endswith("island-1.jsonl") and data == learner_data_arg(source)
    assert str(catalog) not in mounts.values()


def test_without_the_flag_every_island_shares_data(tmp_path, monkeypatch):
    from yeto import launcher

    args = _launcher_args("ports", gpu=TWO_ISLANDS)
    args.controller = "local"
    launcher._prepare_rl_args(args)
    commands = [i["learner_command"] for i in launcher.dry_run_plan(args)["island_requests"]]
    assert {_flag(c, "--data") for c in commands} == {"org/data"}
    assert all("--data-sha256" not in c for c in commands)
    assert launcher._island_tasks_manifest(args) == {}


def test_json_spec_is_rewritten_inline_for_the_head(catalog, tmp_path, monkeypatch):
    from yeto import launcher

    spec = tmp_path / "islands.json"
    spec.write_text(json.dumps({"0": ["web1"], "1": ["web2", "web3"]}))
    args = _island_args(catalog, tmp_path, monkeypatch, str(spec))
    launcher._prepare_rl_args(args)
    assert args.rl_island_tasks == "l0=web1;l1=web2,web3"


def test_unequal_counts_warn_and_continue(catalog, tmp_path, monkeypatch, capsys):
    from yeto import launcher

    args = _island_args(catalog, tmp_path, monkeypatch, "l0=web1-3;l1=web4")
    launcher._prepare_rl_args(args)
    assert "island 0: 3, island 1: 1" in capsys.readouterr().err


def test_launcher_refusals(catalog, tmp_path, monkeypatch):
    from yeto import launcher

    args = _island_args(catalog, tmp_path, monkeypatch, "l0=web1;l1=web2")
    args.data = "org/data"
    with pytest.raises(ValueError, match="one local JSONL file"):
        launcher._prepare_rl_args(args)
    args = _island_args(catalog, tmp_path, monkeypatch, "l0=web1")
    with pytest.raises(ValueError, match=r"no task list for islands \[1\]"):
        launcher._prepare_rl_args(args)
    args = _island_args(catalog, tmp_path, monkeypatch, "l0=web1;l1=web2")
    args.external_learners = 1
    with pytest.raises(ValueError, match="external-learners"):
        launcher._prepare_rl_args(args)


def test_training_islands_do_not_run_eval(catalog, tmp_path, monkeypatch):
    from yeto import launcher

    heldout = _write_rows(tmp_path / "heldout.jsonl", ["web1"])
    args = _island_args(catalog, tmp_path, monkeypatch, "l0=web1;l1=web2",
                        "--rl-eval-interval", "1", "--rl-eval-data", str(heldout),
                        "--rl-eval-dataset-name", "heldout", "--rl-eval-samples-per-prompt", "1")
    with pytest.raises(ValueError, match="training islands do not run eval"):
        launcher._prepare_rl_args(args)


def test_dense_contract_drops_island_local_flags_only():
    from yeto.rl.adapters.miles.island_entry import _dense_shared_miles_argv

    argv = ["--lr", "1e-6", "--prompt-data", "/p.jsonl", "--session-server-port", "1", "2",
            "--eval-prompt-data", "held", "/h.jsonl", "--skip-eval-before-train",
            "--eval-interval", "3", "--log-passrate", "--num-rollout", "5"]
    assert _dense_shared_miles_argv(argv) == ["--lr", "1e-6", "--num-rollout", "5"]
    with pytest.raises(ValueError, match="no value: --eval-interval"):
        _dense_shared_miles_argv(["--eval-interval", "--lr", "1"])


def test_plan_is_computed_once_and_cached(catalog, tmp_path, monkeypatch):
    from yeto import launcher
    from yeto.rl import island_tasks

    args = _island_args(catalog, tmp_path, monkeypatch, "l0=web1;l1=web2")
    launcher._prepare_rl_args(args)
    monkeypatch.setattr(island_tasks, "plan_island_tasks",
                        lambda *a, **k: pytest.fail("plan recomputed"))
    launcher.dry_run_plan(args)
    assert set(launcher._island_tasks_manifest(args)["island_tasks"]) == {"0", "1"}
    with pytest.raises(ValueError, match=r"no task list for learner 2 \(planned islands: \[0, 1\]\)"):
        launcher.island_task_data(args, 2)


def test_head_second_hop_rebuilds_the_same_files(catalog, tmp_path, monkeypatch):
    """First hop stages --data for the head (head_stage); the head prepares again."""

    import shutil

    from yeto import launcher
    from yeto.cli import _serializable_args
    from yeto.datasource import head_stage

    spec = tmp_path / "islands.json"
    spec.write_text(json.dumps({"0": ["web1-3"], "1": ["pwn1-3"]}))
    first = _island_args(catalog, tmp_path, monkeypatch, str(spec))
    launcher._prepare_rl_args(first)
    first_hashes = {k: v.sha256 for k, v in launcher.island_task_plan(first).items()}
    staged_path, mounts = head_stage(first.data)
    (head_copy_source,) = mounts.values()
    head_copy = tmp_path / "head" / Path(staged_path).name
    head_copy.parent.mkdir()
    shutil.copyfile(head_copy_source, head_copy)
    replay = _serializable_args(first)
    assert "_island_task_plan" not in replay
    second = argparse.Namespace(**{**replay, "data": str(head_copy)})
    launcher._prepare_rl_args(second)
    assert {k: v.sha256 for k, v in launcher.island_task_plan(second).items()} == first_hashes


@pytest.mark.parametrize("factory", ["make_learner_task", "make_diffusion_sample_task"])
def test_non_ports_paths_refuse_island_tasks(factory):
    from yeto import launcher
    from yeto.gpu_spec import parse_gpu_spec

    args = argparse.Namespace(rl_island_tasks="l0=web1")
    spec = parse_gpu_spec(TWO_ISLANDS)[0]
    call = (lambda: launcher.make_learner_task(args, spec, 0, 1, "x")) \
        if factory == "make_learner_task" else (lambda: launcher.make_diffusion_sample_task(args, spec))
    with pytest.raises(ValueError, match="ports RL islands only"):
        call()
