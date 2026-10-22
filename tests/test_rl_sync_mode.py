"""--rl-sync-mode: the async defaults of a multi-island RL run and the strict
control. No Ray, no syncer process.

The strict mode must produce the commands the launcher produced before the
modes existed; the comparison runs the launcher of ``BASE`` (checked out with
``git archive``) and of this tree on the same command lines.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tarfile
from io import BytesIO
from pathlib import Path

import pytest

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from test_rl_full_parameter_async import FULL, TWO_ISLANDS

ROOT = Path(__file__).resolve().parents[1]
BASE = "a7589467"  # the last commit before --rl-sync-mode

DECOUPLED = ("--rl-sync-preset", "decoupled", "--local-rl-rounds-per-sync", "2",
             "--fragments", "2", "--fragment-pattern", "binpack")
POOL = ("--rl-sync-quorum-theta", "0.75", "--rl-sync-max-base-lag", "2")
FULL_ASYNC = FULL + ("--rl-sync-full-parameter-rounds", "async")
# Command lines whose strict-mode plan must equal the plan before the modes.
CONTROL_CASES = {
    "strict-avg": (),
    "strict-avg-pool": POOL,
    "strict-avg-k1": ("--rl-sync-quorum", "1", "--rl-sync-max-base-lag", "1"),
    "decoupled": DECOUPLED,
    "full-parameter-sweep": FULL,
    "full-parameter-async": FULL_ASYNC,
    "full-parameter-int8": FULL + ("--rl-sync-push-dtype", "int8"),
}

_PLAN_SCRIPT = r"""
import json, sys, types
sys.path[:0] = [sys.argv[1] + "/tests", sys.argv[1]]
from test_rl_launcher import _Resources, _Storage, _StorageMode, _Task
sys.modules["sky"] = types.SimpleNamespace(
    Task=_Task, Resources=_Resources, Storage=_Storage, StorageMode=_StorageMode)
import yeto.launch.modal_islands as modal_islands
modal_islands._list_modal_apps = lambda: []
from test_rl_algorithm_provenance import _launcher_args
from yeto.launcher import _prepare_rl_args, dry_run_plan
plans = {}
for name, extra in json.loads(sys.argv[2]).items():
    args = _launcher_args("ports", tuple(extra), gpu=sys.argv[3])
    args.controller = "local"
    _prepare_rl_args(args)
    plans[name] = dry_run_plan(args)
print(json.dumps(plans, sort_keys=True, default=str).replace(sys.argv[1], "<ROOT>"))
"""


def _plans(root: Path, cases: dict) -> dict:
    env = {**os.environ, "YETO_ISLAND_HMAC_KEY": "k3y", "PYTHONDONTWRITEBYTECODE": "1"}
    result = subprocess.run(
        [sys.executable, "-c", _PLAN_SCRIPT, str(root), json.dumps(cases), TWO_ISLANDS],
        cwd=root, env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-4000:]
    return json.loads(result.stdout.splitlines()[-1])


@pytest.fixture(scope="module")
def base_tree(tmp_path_factory):
    archive = subprocess.run(["git", "-C", str(ROOT), "archive", BASE],
                             capture_output=True)
    if archive.returncode != 0:
        pytest.skip(f"git archive {BASE} is unavailable here")
    root = tmp_path_factory.mktemp("base")
    with tarfile.open(fileobj=BytesIO(archive.stdout)) as tar:
        tar.extractall(root, filter="data")
    return root


def test_strict_mode_commands_equal_the_commands_before_the_modes(base_tree):
    before = _plans(base_tree, CONTROL_CASES)
    strict = _plans(ROOT, {name: (*extra, "--rl-sync-mode", "strict")
                           for name, extra in CONTROL_CASES.items()})
    for name in CONTROL_CASES:
        assert strict[name] == before[name], name


# -- the async defaults -------------------------------------------------------


@pytest.fixture(autouse=True)
def _hmac_key(monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k3y")


def _plan(extra=(), gpu=TWO_ISLANDS):
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    args = _launcher_args("ports", tuple(extra), gpu=gpu)
    args.controller = "local"
    _prepare_rl_args(args)
    return args, dry_run_plan(args)


def _flags(command: str, *names: str) -> dict[str, str | None]:
    tokens = shlex.split(command.replace("$(", "").replace(")", ""))
    return {name: tokens[tokens.index(name) + 1] if name in tokens else None for name in names}


SYNCER_FLAGS = ("--quorum", "--grace-ms", "--grace-gamma", "--grace-tau", "--pipeline",
                "--outer-lr", "--outer-momentum", "--max-base-lag", "--learner-weight")


def test_full_parameter_defaults_to_asynchronous_rounds():
    args, plan = _plan(FULL)
    assert args.rl_sync_mode_effective == "async"
    command = plan["role_syncers"][0]["command"]
    assert "--policy-sweep-fragments" not in command
    assert "--require-profile-record" in command
    assert _flags(command, *SYNCER_FLAGS) == {
        "--quorum": "1", "--grace-ms": "300000", "--grace-gamma": "0.8", "--grace-tau": "2.0",
        "--pipeline": "2", "--outer-lr": "1.0", "--outer-momentum": "0.0",
        "--max-base-lag": str(args.fragments), "--learner-weight": "tokens2-over-steps"}
    for island in plan["island_requests"]:
        assert _flags(island["learner_command"], "--full-parameter-rounds", "--sweep-push-dtype",
                      "--rl-sync-matrix-merge") == {
            "--full-parameter-rounds": "async", "--sweep-push-dtype": "int4",
            "--rl-sync-matrix-merge": "rda"}


@pytest.mark.parametrize(("rounds", "lr", "momentum"), [("3", "1.0", "0.0"), ("4", "0.7", "0.9")])
def test_async_outer_step_depends_on_the_local_rounds_per_sync(rounds, lr, momentum):
    _, plan = _plan(FULL + ("--local-rl-rounds-per-sync", rounds))
    command = plan["role_syncers"][0]["command"]
    assert _flags(command, "--outer-lr", "--outer-momentum") == {
        "--outer-lr": lr, "--outer-momentum": momentum}


def test_explicit_options_override_the_async_defaults():
    args, plan = _plan(FULL + (
        "--rl-sync-quorum", "2", "--rl-sync-grace-ms", "0", "--rl-sync-learner-weight", "equal",
        "--rl-sync-max-base-lag", "0", "--rl-sync-outer-lr", "0.5", "--rl-sync-outer-momentum",
        "0.5", "--rl-sync-matrix-merge", "avg", "--rl-sync-push-dtype", "bf16",
        "--rl-sync-fragment-pipeline", "1", "--local-rl-rounds-per-sync", "8"))
    command = plan["role_syncers"][0]["command"]
    assert _flags(command, "--quorum", "--grace-ms", "--pipeline", "--outer-lr",
                  "--outer-momentum", "--max-base-lag", "--learner-weight") == {
        "--quorum": "2", "--grace-ms": "0", "--pipeline": "1", "--outer-lr": "0.5",
        "--outer-momentum": "0.5", "--max-base-lag": "0", "--learner-weight": "equal"}
    island = plan["island_requests"][0]["learner_command"]
    assert _flags(island, "--sweep-push-dtype")["--sweep-push-dtype"] == "bf16"
    assert "--rl-sync-matrix-merge" not in island


def test_explicit_sweep_keeps_the_strict_control():
    args, default = _plan(FULL + ("--rl-sync-full-parameter-rounds", "sweep"))
    _, strict = _plan(FULL + ("--rl-sync-mode", "strict"))
    assert args.rl_sync_mode_effective == "strict"
    assert default == strict


def test_sweep_only_option_names_the_strict_mode():
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", FULL + ("--rl-sync-async-checkpoint",), gpu=TWO_ISLANDS)
    with pytest.raises(ValueError, match="--rl-sync-mode strict"):
        _prepare_rl_args(args)


def test_lora_defaults_to_quorum_one_with_grace_and_token_weights():
    args, plan = _plan()
    command = plan["role_syncers"][0]["command"]
    assert _flags(command, *SYNCER_FLAGS) == {
        "--quorum": "1", "--grace-ms": "300000", "--grace-gamma": "0.8", "--grace-tau": "2.0",
        "--pipeline": "1", "--outer-lr": "1.0", "--outer-momentum": "0.0",
        "--max-base-lag": "1", "--learner-weight": "tokens2-over-steps"}
    # strict-avg merges one whole-adapter fragment; RDA is for fragmented presets.
    assert args.rl_sync_matrix_merge == "avg"


def test_decoupled_defaults_take_rda_and_nesterov_from_four_rounds():
    args, plan = _plan(DECOUPLED[:2] + ("--local-rl-rounds-per-sync", "4") + DECOUPLED[4:])
    command = plan["role_syncers"][0]["command"]
    assert _flags(command, "--quorum", "--outer-lr", "--outer-momentum", "--learner-weight") == {
        "--quorum": "1", "--outer-lr": "0.7", "--outer-momentum": "0.9",
        "--learner-weight": "tokens2-over-steps"}
    assert args.rl_sync_matrix_merge == "rda"


def test_capacity_quorum_replaces_the_count_quorum_default():
    args, _ = _plan(POOL)
    assert args.rl_sync_quorum is None and args.quorum == 1


@pytest.mark.parametrize("extra", [(), FULL])
def test_one_island_keeps_the_strict_values(extra):
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", extra, gpu="modal:2xh100")
    args.controller = "local"
    try:
        _prepare_rl_args(args)
    except ValueError:
        pass  # a one-island preset may be refused for other reasons
    assert args.rl_sync_mode_effective == "strict"
    assert args.rl_sync_full_parameter_rounds == "sweep"
    assert args.rl_sync_push_dtype == "f32"


def test_peer_recovery_defaults_on_where_it_applies():
    from yeto.launcher import _prepare_rl_args

    store = ("--rl-checkpoint-store", "/shared/run-store")
    args, _ = _plan(FULL + store)
    assert args.rl_peer_recovery is True
    args, _ = _plan(FULL)  # no snapshot store: off
    assert args.rl_peer_recovery is False
    # Off: the store alone is then refused, which shows recovery stayed off.
    for extra in (("--no-rl-peer-recovery",), ("--rl-sync-mode", "strict"), POOL):
        args = _launcher_args("ports", FULL + store + extra, gpu=TWO_ISLANDS)
        with pytest.raises(ValueError, match="or --rl-peer-recovery"):
            _prepare_rl_args(args)
        assert args.rl_peer_recovery is False
