"""rl-mixed-island-placement: per-island placement under inter-island elastic
scheduling (island 0 colocated, island 1 fixed partition). Pure launcher checks
and dry-run commands; no cloud, no Ray."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from test_rl_engine_selection import _cli, _island_task  # noqa: F401  (sky stub helper)
from yeto import launcher
from yeto.gpu_spec import parse_gpu_spec
from yeto.rl.engine import island_placement as ip

GPUS = ("--gpu", "aws:1xa100@us-east-1,aws:2xa100@us-west-2")
MIXED = (*GPUS, "--rl-island-placement", "1:placement=fixed-partition", "--rl-rollout-gpus", "1")
ELASTIC = ("--rl-island-scheduling", "elastic")


@pytest.fixture(autouse=True)
def _hmac_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k3y")


def _island_tasks(args, monkeypatch):
    _island_task(args, monkeypatch)  # installs the sky stub and the provenance fields
    specs = parse_gpu_spec(args.gpu)
    return [launcher._island_task(launcher.make_miles_island_task, args, spec, m, len(specs),
                                  "127.0.0.1:29400", {}, None)
            for m, spec in enumerate(specs)]


def _learner_flags(task) -> list[str]:
    line = next(line for line in task.run.splitlines() if "--learner-id" in line)
    return line.split()


def _value(flags: list[str], name: str) -> str | None:
    return flags[flags.index(name) + 1] if name in flags else None


# --- parsing -------------------------------------------------------------------

def test_parse_overrides_maps_keys_to_run_wide_dests():
    assert ip.parse_overrides(["0:placement=colocated", "1:standby-gpus=1",
                               "1:use-gpus-per-node=4"], 2) == {
        0: {"rl_placement": "colocated"},
        1: {"rl_standby_gpus": 1, "rl_island_use_gpus_per_node": 4},
    }
    assert ip.parse_overrides(None, 3) == {}


@pytest.mark.parametrize("item, message", [
    ("placement=colocated", "ISLAND:KEY=VALUE"),
    ("x:placement=colocated", "ISLAND must be an integer"),
    ("2:placement=colocated", "island 2 is not in 0..1"),
    ("0:rollout-gpus=1", "unknown key"),
    ("0:placement=mixed", "placement must be one of"),
    ("0:standby-gpus=-1", "must be >= 0"),
    ("0:use-gpus-per-node=0", "must be >= 1"),
])
def test_parse_overrides_refuses_bad_items(item, message):
    with pytest.raises(ValueError, match=message):
        ip.parse_overrides([item], 2)


def test_parse_overrides_refuses_a_key_given_twice():
    with pytest.raises(ValueError, match="gives 'placement' twice"):
        ip.parse_overrides(["0:placement=colocated", "0:placement=fixed-partition"], 2)


def test_cli_flag_is_defined_once_in_the_neutral_schema():
    from yeto.rl.engine import run_flags

    assert run_flags.rl_option(ip.OVERRIDES_DEST, run_flags.CLI) == "--rl-island-placement"
    with pytest.raises(KeyError):
        run_flags.rl_option(ip.OVERRIDES_DEST, run_flags.ISLAND)  # launcher-only


# --- resolution ------------------------------------------------------------------

def test_without_overrides_every_island_uses_the_run_wide_flags():
    args = _cli((*GPUS, "--rl-placement", "fixed-partition", "--rl-rollout-gpus", "1"))
    assert launcher.rl_island_args(args, 0) is args and launcher.rl_island_args(args, 1) is args
    assert [p.placement for p in ip.island_placements(args, 2)] == ["fixed-partition"] * 2


def test_a_colocated_island_drops_the_fixed_partition_only_values():
    args = _cli((*MIXED, "--rl-standby-gpus", "0", *ELASTIC))
    colocated, partitioned = launcher.rl_island_args(args, 0), launcher.rl_island_args(args, 1)
    assert (colocated.rl_placement, colocated.rollout_num_gpus) == ("colocated", None)
    assert (partitioned.rl_placement, partitioned.rollout_num_gpus) == ("fixed-partition", 1)
    assert args.rl_placement == "colocated" and args.rollout_num_gpus == 1  # run-wide unchanged


# --- launch checks (before any cloud work) ------------------------------------------

def test_mixed_placement_under_legacy_scheduling_is_refused():
    args = _cli(MIXED)
    with pytest.raises(ValueError, match="needs --rl-island-scheduling elastic") as error:
        launcher._prepare_rl_args(args)
    assert "island 0 colocated, island 1 fixed-partition" in str(error.value)


def test_mixed_placement_under_elastic_scheduling_passes():
    args = _cli((*MIXED, *ELASTIC))
    launcher._prepare_rl_args(args)  # includes the per-island shape checks
    assert args.rl_expected_algorithm_sha256


@pytest.mark.parametrize("flag, dest", [("--rl-elastic", "rl_elastic"),
                                        ("--rl-overlap-eval", "rl_overlap_eval")])
def test_fixed_partition_only_switches_name_the_colocated_island(flag, dest):
    args = _cli((*MIXED, *ELASTIC))
    setattr(args, dest, True)
    with pytest.raises(ValueError, match=rf"{flag} needs .* colocated island\(s\): \[0\]"):
        launcher._check_ports_infra_switches(args)


def test_a_colocated_island_cannot_reserve_standby_gpus():
    args = _cli((*MIXED, *ELASTIC, "--rl-island-placement", "0:standby-gpus=1"))
    with pytest.raises(ValueError, match="island 0 is colocated and cannot reserve standby"):
        launcher._check_ports_infra_switches(args)


def test_mixed_launch_refusals_create_no_cloud_resources(monkeypatch):
    """The refusals happen in _prepare_rl_args, which the launch runs before sky."""
    import sys

    monkeypatch.setitem(sys.modules, "sky", None)  # any sky import would fail loudly
    with pytest.raises(ValueError, match="needs --rl-island-scheduling elastic"):
        launcher._prepare_rl_args(_cli(MIXED))


# --- per-island shapes and commands ------------------------------------------------

def test_each_island_gets_its_own_shape_and_command(monkeypatch):
    args = _cli((*MIXED, *ELASTIC))
    launcher._prepare_rl_args(args)
    colocated, partitioned = (_learner_flags(t) for t in _island_tasks(args, monkeypatch))
    assert "--rl-placement" not in colocated and "--rollout-num-gpus" not in colocated
    assert _value(colocated, "--actor-num-gpus-per-node") == "1"
    assert _value(partitioned, "--rl-placement") == "fixed-partition"
    assert _value(partitioned, "--rollout-num-gpus") == "1"
    assert _value(partitioned, "--actor-num-gpus-per-node") == "1"
    # same algorithm on both islands (P4-Q4: placement capabilities stay out of the hash)
    assert (_value(colocated, "--rl-expected-algorithm-sha256")
            == _value(partitioned, "--rl-expected-algorithm-sha256")
            == args.rl_expected_algorithm_sha256)
    for flags in (colocated, partitioned):
        assert _value(flags, "--rl-island-scheduling") == "elastic"
        assert "--rl-per-island-placement" in flags


def test_uniform_fixed_partition_command_is_unchanged_by_this_change(monkeypatch):
    plain = _cli((*GPUS, "--rl-placement", "fixed-partition", "--rl-rollout-gpus", "1"))
    plain.gpu = "aws:2xa100@us-east-1,aws:2xa100@us-west-2"
    explicit = _cli((*GPUS, "--rl-placement", "fixed-partition", "--rl-rollout-gpus", "1",
                     "--rl-island-placement", "0:placement=fixed-partition",
                     "--rl-island-placement", "1:placement=fixed-partition"))
    explicit.gpu = plain.gpu
    for args in (plain, explicit):
        launcher._prepare_rl_args(args)
    plain_runs = [t.run for t in _island_tasks(plain, monkeypatch)]
    explicit_runs = [t.run for t in _island_tasks(explicit, monkeypatch)]
    assert all("--rl-per-island-placement" not in run for run in plain_runs)
    # per-island values present: only the contract marker is added
    assert [run.replace(" --rl-per-island-placement", "") for run in explicit_runs] == plain_runs


def test_policy_age_overlap_differs_by_island_placement():
    """P0-S: with --rl-max-policy-age N>0 a fixed-partition island overlaps rollout
    and training (Miles fully_async), a colocated island carries over at the cut-off
    (--partial-rollout). In a mixed run the two islands differ, by design."""
    from tests.test_rl_miles_adapter_config import make_config, sub
    from tests.test_rl_policy_versions import _tolerant
    from yeto.rl.adapters.miles import config as mc

    partitioned = sub(make_config(colocated=False), "serving", offload_train=False)
    island_1 = mc.translate_run_config(partitioned, _tolerant(), max_policy_age=1)
    island_0 = mc.translate_run_config(make_config(), _tolerant(), max_policy_age=1)
    assert island_1.rollout_overlap and "--fully-async" in island_1.argv
    assert not island_0.rollout_overlap and "--partial-rollout" in island_0.argv
    assert "--fully-async" not in island_0.argv and "--partial-rollout" not in island_1.argv


# --- records -----------------------------------------------------------------------

def test_run_manifest_rows_record_every_island_placement():
    args = _cli((*MIXED, *ELASTIC))
    launcher._prepare_rl_args(args)
    rows = launcher.rl_island_placement_records(args)
    assert [(r["island"], r["placement"], r["standby_gpus"]) for r in rows] == [
        (0, "colocated", 0), (1, "fixed-partition", 0)]
    assert {r["algorithm_sha256"] for r in rows} == {args.rl_expected_algorithm_sha256}


# --- static check (tasks 1.3) --------------------------------------------------------

PLACEMENT_DESTS = {"rl_placement", "rl_standby_gpus", "rl_island_use_gpus_per_node",
                   "rollout_num_gpus"}
# Functions that read the placement values from the args they are given. Each
# caller passes the island's args (rl_island_args); the run-wide readers are the
# presets that need one placement on every island (both require the legacy
# schedule, so a mixed run never reaches them with different placements).
ISLAND_ARGS_READERS = {
    "rl_island_spec", "rl_actor_gpus_per_node", "rl_trainer_shape", "rl_island_layout",
    "rl_min_nodes", "_ports_algorithm_flags",
}
RUN_WIDE_READERS = {"_prepare_sao_streaming", "_prepare_dense_full_sweep"}


def _launcher_layer_tree() -> ast.Module:
    """yeto/launcher.py and the yeto/launch package it composes, as one module."""
    launcher_py = Path(inspect.getsourcefile(launcher))
    files = [launcher_py, *sorted((launcher_py.parent / "launch").glob("*.py"))]
    body = [node for f in files for node in ast.parse(f.read_text(encoding="utf-8")).body]
    return ast.Module(body=body, type_ignores=[])


def _placement_readers(tree: ast.AST) -> set[str]:
    """Functions that read a placement value from a variable named ``args``
    (``args.rl_placement`` or ``getattr(args, "rl_placement", ...)``)."""

    def is_args(node: ast.AST) -> bool:
        return isinstance(node, ast.Name) and node.id == "args"

    readers = set()
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(func):
            attribute = (isinstance(node, ast.Attribute) and node.attr in PLACEMENT_DESTS
                         and is_args(node.value) and isinstance(node.ctx, ast.Load))
            getter = (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "getattr"
                      and len(node.args) >= 2 and is_args(node.args[0])
                      and isinstance(node.args[1], ast.Constant)
                      and node.args[1].value in PLACEMENT_DESTS)
            if attribute or getter:
                readers.add(func.name)
    return readers


def test_launcher_reads_placement_values_only_from_island_args():
    tree = _launcher_layer_tree()
    assert _placement_readers(tree) == ISLAND_ARGS_READERS | RUN_WIDE_READERS


def test_island_tasks_are_built_only_through_island_task():
    """Every island task factory call passes the island's args (_island_task)."""
    tree = _launcher_layer_tree()
    callers = set()
    for func in ast.walk(tree):
        if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(func):
                if isinstance(node, ast.Call) and getattr(node.func, "id", None) in (
                        "make_miles_island_task", "task_factory"):
                    callers.add(func.name)
    assert callers == {"_island_task"}
