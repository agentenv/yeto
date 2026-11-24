"""A deterministic configuration error fails before any GPU, and is never retried.

A Qwen3-4B launch with ``--rl-max-tokens-per-gpu 16384`` and ``--seq-len 32768``
passed ``launch --dry-run`` (rc 0); the island refused it ("16384 x CP 1 is below
--seq-len 32768", ``run_config._train_memory``) and Modal re-ran the failed
container for ten minutes. Now:

* the launcher runs the island's trainer memory rules before any cloud work, so
  the dry run fails with the fix in the message;
* an island that still refuses its options (a :class:`ConfigError`) exits
  ``CONFIG_ERROR_EXIT`` with a ``[yeto-config-error]`` line; the Modal function
  returns instead of raising (no Modal retry), and the fleet controller ends the
  run with ``CONFIG_ERROR_EXIT`` without a relaunch.
"""

from __future__ import annotations

import io
import json
import types

import pytest

import yeto.launcher as launcher
import yeto.modal_runner as mr
from test_modal_runner import _cfg, fake_modal
from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from test_rl_launcher_island_failure import Clock, _setup
from yeto.config_errors import CONFIG_ERROR_EXIT, CONFIG_ERROR_MARKER, ConfigError, config_error_in, config_error_line
from yeto.launch import fleet
from yeto.rl.adapters.miles.config import MilesConfigError

TOO_SMALL = ("--seq-len", "32768", "--rl-max-tokens-per-gpu", "16384")


def test_the_exit_code_is_registered_and_distinct():
    registered = {fleet.ISLAND_FAILED_EXIT, fleet.TEARDOWN_UNVERIFIED_EXIT, fleet.RUN_STALLED_EXIT,
                  fleet.CONTAINER_CHANGED_EXIT, fleet.TRAINER_GAVE_UP_EXIT, fleet.RUN_STOPPED_EXIT,
                  launcher.NO_SYNC_INCOMPLETE_EXIT, 1, 2}
    assert CONFIG_ERROR_EXIT == fleet.CONFIG_ERROR_EXIT == 10
    assert CONFIG_ERROR_EXIT not in registered
    from yeto.wan_probe import EXIT_SLOW_LINK

    assert CONFIG_ERROR_EXIT != EXIT_SLOW_LINK


def test_config_errors_are_value_errors_and_found_in_a_chain():
    assert issubclass(ConfigError, ValueError) and issubclass(MilesConfigError, ConfigError)
    try:
        try:
            raise ConfigError("bad option")
        except ConfigError as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as outer:
        assert str(config_error_in(outer)) == "bad option"
    assert config_error_in(ValueError("other")) is None
    assert CONFIG_ERROR_MARKER in config_error_line(ConfigError("x"))


# ------------------------------------------------------------------ launcher preflight / dry run


def test_prepare_refuses_a_micro_batch_below_seq_len_before_any_cloud_work():
    args = _launcher_args("ports", (*TOO_SMALL, "--rl-single-island-no-sync"))
    args.controller = "local"
    with pytest.raises(ConfigError, match=r"16384 x CP 1 is below --seq-len 32768.*at least 32768"):
        launcher._prepare_rl_args(args)


@pytest.mark.parametrize("extra, match", [
    (("--rl-optimizer-offload-fraction", "1.5"), "offload-fraction"),
    (("--rl-precision-aware-optimizer",), "needs --rl-single-island-no-sync"),
])
def test_prepare_runs_the_other_trainer_memory_rules(extra, match):
    args = _launcher_args("ports", extra)
    args.controller = "local"
    with pytest.raises(ConfigError, match=match):
        launcher._prepare_rl_args(args)


def test_prepare_accepts_a_budget_that_holds_one_row():
    args = _launcher_args("ports", ("--seq-len", "32768", "--rl-max-tokens-per-gpu", "32768",
                                    "--rl-single-island-no-sync"))
    args.controller = "local"
    launcher._prepare_rl_args(args)


def test_cli_dry_run_fails_with_the_fix(monkeypatch, capsys):
    import yeto.cli as cli

    calls = []
    monkeypatch.setattr(launcher, "prepare_launch_args", launcher._prepare_rl_args)
    monkeypatch.setattr(launcher, "run", lambda *a, **k: calls.append("run"))
    monkeypatch.setattr(cli, "_spawn_worker", lambda *a, **k: calls.append("spawn"))
    monkeypatch.setattr(cli.runs, "create_run", lambda *a, **k: calls.append("create"))
    args = _launcher_args("ports", (*TOO_SMALL, "--rl-single-island-no-sync", "--dry-run",
                                    "--controller", "local"))
    assert cli.cmd_launch(args) != 0
    captured = capsys.readouterr()
    assert "below --seq-len 32768" in captured.err + captured.out
    assert "--rl-max-tokens-per-gpu to at least 32768" in captured.err + captured.out
    assert calls == []
    # the same options with a budget that fits: a dry-run plan, rc 0
    # an sm90+ card: the Miles image is refused below it (trainer_image_check)
    ok = _launcher_args("ports", ("--seq-len", "32768", "--rl-max-tokens-per-gpu", "32768",
                                  "--rl-single-island-no-sync", "--dry-run", "--controller", "local"),
                        gpu="aws:1xh100@us-east-1")
    assert cli.cmd_launch(ok) == 0
    assert json.loads(capsys.readouterr().out)["dry_run"]


# ------------------------------------------------------------------ the island


def test_the_island_entry_exits_config_error_for_a_refused_option():
    from yeto.rl.adapters.miles.island_entry import config_error_exit

    stream = io.StringIO()
    try:
        raise MilesConfigError("the ports path cannot express this")
    except MilesConfigError as error:
        assert config_error_exit(error, stream) == CONFIG_ERROR_EXIT
    text = stream.getvalue()
    assert "Traceback" in text and f"{CONFIG_ERROR_MARKER} MilesConfigError: the ports path" in text
    assert config_error_exit(RuntimeError("CUDA out of memory"), io.StringIO()) is None


def test_the_modal_function_returns_a_config_error_instead_of_raising(monkeypatch):
    """A raise would be retried by Modal (up to 10 times); the return value ends the call."""
    monkeypatch.setattr(mr.subprocess, "call", lambda cmd, env=None: CONFIG_ERROR_EXIT)
    assert mr.island_main(_cfg(num_nodes=1, run_script="exit 10").to_json()) == CONFIG_ERROR_EXIT
    monkeypatch.setattr(mr.subprocess, "call", lambda cmd, env=None: 1)
    with pytest.raises(RuntimeError, match="exited with 1"):
        mr.island_main(_cfg(num_nodes=1, run_script="exit 1").to_json())


def test_modal_status_reads_a_returned_config_error_as_failed(monkeypatch, capsys):
    state = fake_modal(monkeypatch)
    ops = mr.ModalOps("yeto-run")
    cfg = _cfg()
    ops.define(cfg)
    cid = ops.spawn(cfg)
    state["calls"][cid].get = lambda timeout=None, index=0: CONFIG_ERROR_EXIT
    assert ops.status(cid) == "FAILED"
    assert "refused its configuration" in capsys.readouterr().err
    state["calls"][cid].get = lambda timeout=None, index=0: 0
    assert ops.status(cid) == "SUCCEEDED"


def test_the_log_probes_report_the_marker():
    from yeto.launch.ssh_resident import STRICT_MARKERS

    assert CONFIG_ERROR_MARKER in STRICT_MARKERS
    ops = mr.ModalIslandOps(types.SimpleNamespace(
        tail_logs=lambda job_id, entries=400: ["noise", f"{CONFIG_ERROR_MARKER} ConfigError: x"]))
    assert ops.rl_strict_failure("island", "fc-1").startswith(CONFIG_ERROR_MARKER)


# ------------------------------------------------------------------ the fleet controller


@pytest.mark.parametrize("provider", ["modal", "sky"])
@pytest.mark.parametrize("syncer", [True, False])
def test_a_refused_island_ends_the_run_with_the_config_error_code_and_no_relaunch(
        monkeypatch, tmp_path, provider, syncer):
    record, clock = [], Clock()
    line = f"{CONFIG_ERROR_MARKER} MilesConfigError: refused (configuration error: exit 10, not retried)"
    _setup(monkeypatch, tmp_path, failing={0}, clock=clock, record=record, log_lines=("noise", line))
    one = "modal:1xa100" if provider == "modal" else "aws:1xa100@us-east-1"
    extra = ("--controller", "local", "--rl-image", "docker:ghcr.io/x/y@sha256:" + "a" * 64)
    if not syncer:
        extra += ("--rl-single-island-no-sync",)
    args = _launcher_args("ports", extra, gpu=f"{one},{one}" if syncer else one)
    args.keep = False
    args.recover_timeout = 600.0  # recovery on: still no relaunch
    args.controller_poll = 30.0
    assert launcher.run(args) == CONFIG_ERROR_EXIT
    assert not [r for r in record if r[0] == "relaunch"]
    assert record.count(("spawn", 0)) <= 1
    if provider == "modal":
        assert record.count(("stop_app",)) == 1
