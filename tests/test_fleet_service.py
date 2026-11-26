"""The on-site control plane of a fleet (``yeto fleet service``, ``fleet.syncer_service``):
the systemd units of ``yeto run up`` and of the fleet's syncer, their install over SSH (a
fake ``ssh``), the syncer launch that watches an on-site syncer service instead of starting
one, and the syncer service itself (fake syncer; the real one in
``test_fleet_service_syncer_e2e.py``). No cloud, no systemd."""

from __future__ import annotations

import json
import pathlib
import types

import pytest

from test_fleet_island_launches import _prepared, example, example_doc
from yeto.config_errors import CONFIG_ERROR_EXIT, ISLAND_STOP_EXIT
from yeto.rl import fleet_service as fs
from yeto.launch import fleet_syncer
from yeto.launch.pool import fleet_cli
from yeto.rl import fleet, island_launches, up
from yeto.rl.engine.object_store import DirectoryObjectStore

KEY = "onsite-test-island-key"
SECRETS = {"YETO_ISLAND_HMAC_KEY": KEY, "YETO_RL_STORE_S3_ACCESS_KEY_ID": "onsite-test-access-key-id",
           "YETO_RL_STORE_S3_SECRET_ACCESS_KEY": "onsite-secret-access-value", "HF_TOKEN": "onsite-hf-token-value"}


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    from yeto import runs

    for name, value in SECRETS.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")  # the run's STOP flag, if any
    import os

    before = dict(os.environ)
    yield
    os.environ.clear()  # the syncer service adds the run config's env to its own: not to the next test's
    os.environ.update(before)


def onsite_doc(tmp_path, host: str = "10.0.0.5") -> dict:
    doc = example_doc(tmp_path)
    doc["fleet"]["syncer_service"] = host
    doc.setdefault("env", {})["HF_TOKEN"] = "run-config-hf-token-value"  # a credential in the run config's env
    return doc


def onsite_config(tmp_path, **kw):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(onsite_doc(tmp_path, **kw)))
    return path


# -- the compile -------------------------------------------------------------------------------

def test_a_syncer_service_hands_the_syncer_launch_to_it_and_nothing_else_changes(tmp_path):
    base = example(tmp_path)
    cfg = up.load_config(str(onsite_config(tmp_path)))
    plan = fleet.plan_of(cfg)
    assert plan.syncer_service == "10.0.0.5"
    argv = cfg["trainer"]["argv"]
    assert argv[-4:] == ["--rl-fleet-syncer-service", "10.0.0.5", "--controller", "local"]
    assert argv[:-4] == base["trainer"]["argv"]  # the syncer launch is the same launch, watching the service
    assert plan.to_json()["syncer"]["service"] == "10.0.0.5"
    assert plan.to_json()["syncer"]["placement"] == {"where": "onsite", "provider": None}
    assert "service" not in fleet.plan_of(base).to_json()["syncer"]  # no key without the service
    assert fleet.FleetPlan.from_json(plan.to_json()).syncer_service == "10.0.0.5"
    for before, after in zip(island_launches.trainer_members(base, fleet.plan_of(base)),
                             island_launches.trainer_members(cfg, plan)):
        assert before == after  # the island launches never see it
    from yeto.breakers.fleet import syncer_price

    assert syncer_price(cfg) == 0.0  # the operator's own machine: no cloud bill


@pytest.mark.parametrize("change, match", [
    ({"launch": "synced_group", "max_islands": None}, "applies to fleet.launch per_island"),
    ({"syncer_service": "10.0.0.5:29400"}, "without a port"),
    ({"syncer_service": "host name"}, "without a port"),
    ({"syncer_service": 5}, "without a port"),
])
def test_a_bad_syncer_service_is_refused(tmp_path, change, match):
    doc = onsite_doc(tmp_path)
    for key, value in change.items():
        if value is None:
            doc["fleet"].pop(key, None)
        else:
            doc["fleet"][key] = value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(Exception, match=match):
        up.load_config(str(path))


def test_the_syncer_launch_dry_run_names_the_service(tmp_path):
    from yeto.launcher import dry_run_plan

    cfg = up.load_config(str(onsite_config(tmp_path)))
    args = _prepared(cfg["trainer"]["argv"])
    assert dry_run_plan(args)["syncer"] == "on-site service on 10.0.0.5"


def test_the_service_flag_belongs_to_the_syncer_launch_only(tmp_path):
    cfg = up.load_config(str(onsite_config(tmp_path)))
    member = island_launches.trainer_members(cfg, fleet.plan_of(cfg))[0]
    argv = island_launches.launch_argv(member, syncer={"addr": "10.0.0.5:29400", "syncer_epoch": 0})
    with pytest.raises(ValueError, match="belongs to the syncer launch"):
        _prepared([*argv, "--rl-fleet-syncer-service", "10.0.0.5"])
    with pytest.raises(ValueError, match="--controller local"):
        _prepared([*cfg["trainer"]["argv"], "--controller", "head"])


def test_run_up_never_generates_a_key_for_an_on_site_syncer(tmp_path, monkeypatch):
    monkeypatch.delenv("YETO_ISLAND_HMAC_KEY")
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    run = up.Run(up.load_config(str(onsite_config(tmp_path))), state_root=tmp_path / "state")
    run.env.pop("YETO_ISLAND_HMAC_KEY", None)
    assert "YETO_ISLAND_HMAC_KEY is not set" in run.fleet_session_problem(dry_run=True)
    with pytest.raises(up.RunUpRefused, match="none is generated"):
        run.island_hmac_env()
    assert not (run.dir / "island_hmac.key").exists()
    plain = up.Run(example(tmp_path), state_root=tmp_path / "plain")
    plain.env.pop("YETO_ISLAND_HMAC_KEY", None)
    plain.island_hmac_env()  # without the service: generated and kept, as before
    assert (plain.dir / "island_hmac.key").exists()


# -- the units -----------------------------------------------------------------------------------

def spec(role: str = "control", **kw) -> fs.ServiceSpec:
    fields = dict(run="r", role=role, config="/srv/yeto/run.json", env_file="/etc/yeto/r.env",
                  workdir=f"/var/tmp/yeto/r-{role}", user="yeto", python="/opt/yeto-venv/bin/python",
                  yeto_dir="/opt/yeto")
    fields.update(kw)
    return fs.ServiceSpec(**fields)


CONTROL_UNIT = """\
# The fleet control plane (yeto run up) of run r, written by `yeto fleet service --role control`.
# Credentials come from EnvironmentFile only (no access for group and others); this file holds none.
[Unit]
Description=yeto fleet control plane (yeto run up) of run r
Wants=network-online.target
After=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
User=yeto
EnvironmentFile=/etc/yeto/r.env
Environment=PYTHONPATH=/opt/yeto
Environment=PYTHONUNBUFFERED=1
Environment=YETO_STATE_DIR=/var/tmp/yeto/r-control/rl-up
Environment=YETO_RUNS_DIR=/var/tmp/yeto/r-control/runs
ExecStartPre=/bin/sh -c 'case "$$(stat -c %%a /etc/yeto/r.env)" in *00) ;; *) echo "/etc/yeto/r.env must give group and others no access (chmod 600)" >&2; exit 1;; esac'
ExecStartPre=/bin/mkdir -p /var/tmp/yeto/r-control
WorkingDirectory=-/var/tmp/yeto/r-control
ExecStart=/opt/yeto-venv/bin/python -m yeto.cli fleet service-run --role control --config /srv/yeto/run.json
Restart=on-failure
RestartSec=30
RestartPreventExitStatus=10 75 76
SuccessExitStatus=143
KillMode=process
TimeoutStopSec=300

[Install]
WantedBy=multi-user.target
"""


def test_the_control_unit_snapshot():
    assert fs.unit_text(spec("control")) == CONTROL_UNIT
    # a configuration error and run up's exit 1, mapped; a refusal before ordering (2) restarts, without a limit
    assert fs.CONTROL_PREVENT == (fs.CONFIG_EXIT, fs.CONTROL_DECIDED_EXIT, fs.CONTROL_FENCED_EXIT) == (10, 75, 76)
    assert "StartLimitIntervalSec=0\n" in CONTROL_UNIT and "StartLimitBurst" not in CONTROL_UNIT


def test_the_syncer_unit_restarts_on_a_crash_only_and_stops_its_whole_group():
    text = fs.unit_text(spec("syncer"))
    assert ("ExecStart=/opt/yeto-venv/bin/python -m yeto.cli fleet service-run --role syncer --config "
            "/srv/yeto/run.json --stop-grace-s 170\n") in text
    assert "Restart=on-failure\n" in text
    assert "RestartPreventExitStatus=2 9 10 11\n" in text  # a refusal, the fleet's STOP, a config error, strict
    assert "SuccessExitStatus=143\n" in text  # a stop of the unit: not a failure
    assert "KillMode=control-group\n" in text  # the syncer gets SIGTERM and checkpoints
    assert "TimeoutStopSec=180\n" in text and fs.SYNCER_STOP_GRACE_S == 170  # SIGKILL inside the stop timeout
    assert fs.unit_name("r", "syncer") == "yeto-r-syncer.service"
    assert 1 not in fs.SYNCER_PREVENT and fs.CRASH_EXIT not in fs.SYNCER_PREVENT + fs.CONTROL_PREVENT


@pytest.mark.parametrize("role", fs.ROLES)
def test_a_unit_holds_no_credential_value(tmp_path, monkeypatch, capsys, role):
    path = onsite_config(tmp_path)
    args = fleet_cli_args(["fleet", "service", "--config", str(path), "--role", role,
                           "--env-file", "/etc/yeto/r.env", "--print"])
    assert fleet_cli.main(args) == 0
    text = capsys.readouterr().out
    assert text.startswith("# The fleet ") and "EnvironmentFile=/etc/yeto/r.env\n" in text
    for value in [*SECRETS.values(), "run-config-hf-token-value", "R2_BUCKET"]:
        assert value not in text
    assert "Environment=" not in "".join(line for line in text.splitlines(True)
                                         if line.startswith("Environment=")
                                         and line.split("=", 2)[1] not in ("PYTHONPATH", "PYTHONUNBUFFERED",
                                                                           "YETO_STATE_DIR", "YETO_RUNS_DIR"))


def fleet_cli_args(argv: list[str]):
    from yeto import cli

    return cli.build_parser().parse_args(argv)


@pytest.mark.parametrize("field, value", [
    ("env_file", "/etc/yeto/r env"), ("config", "run.json"), ("workdir", "/var/tmp/$HOME"),
    ("python", "/opt/%h/python"), ("user", "Root User"), ("role", "head"),
    # injection through the run name (the Description and comment lines) or a path
    ("run", "r\nExecStartPre=/bin/sh -c id"), ("run", "r%h"), ("run", "r$USER"), ("run", "r x"), ("run", ".."),
    ("run", "r;id"), ("config", "/srv/run.json\nUser=root"), ("env_file", "/etc/yeto/%i.env"),
    ("yeto_dir", "/opt/$(id)"), ("workdir", "/var/tmp/yeto/../../etc"),
])
def test_a_unit_line_that_would_not_hold_the_value_is_refused(field, value):
    with pytest.raises(fs.ServiceError):
        fs.unit_text(types.SimpleNamespace(**{**spec().__dict__, field: value}))
    with pytest.raises(fs.ServiceError):
        fs.as_unit_script(types.SimpleNamespace(**{**spec().__dict__, field: value}), ["stop-run", "r"])


def test_a_run_name_from_the_config_that_a_unit_cannot_hold_is_refused(tmp_path, capsys):
    doc = onsite_doc(tmp_path)
    doc["run"] = "r%n"
    path = tmp_path / "bad-run.json"
    path.write_text(json.dumps(doc))
    args = fleet_cli_args(["fleet", "service", "--config", str(path), "--role", "control", "--env-file",
                           "/etc/yeto/r.env", "--print"])
    code = fleet_cli.main(args)
    out = capsys.readouterr()
    assert code == 2 and "[Service]" not in out.out


def test_the_unit_is_for_an_on_site_fleet_only(tmp_path, capsys):
    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps(example_doc(tmp_path)))
    args = fleet_cli_args(["fleet", "service", "--config", str(plain), "--role", "syncer",
                           "--env-file", "/etc/yeto/r.env", "--print"])
    assert fleet_cli.main(args) == 2
    assert "fleet.syncer_service" in capsys.readouterr().err
    group = tmp_path / "group.json"
    group.write_text(json.dumps(example_doc(tmp_path, per_island=False)))
    args.config, args.role = str(group), "control"
    assert fleet_cli.main(args) == 2
    assert "per_island" in capsys.readouterr().err


class FakeSsh:
    def __init__(self, rc: int = 0) -> None:
        self.calls: list[tuple[list[str], str]] = []
        self.rc = rc

    def __call__(self, argv, *, input, text, capture_output, timeout):  # noqa: A002 - subprocess.run's name
        self.calls.append((list(argv), input))
        return types.SimpleNamespace(returncode=self.rc, stdout="", stderr="mode 644\n" if self.rc else "")


def test_install_writes_the_unit_and_enables_it_and_does_nothing_else():
    ssh, logs = FakeSsh(), []
    assert fs.install(spec("syncer"), "ops@site-a", runner=ssh, log=logs.append) == 0
    ((argv, script),) = ssh.calls
    assert argv == ["ssh", "-o", "BatchMode=yes", "ops@site-a", "bash", "-s"]
    body = script.split("<<'YETO_UNIT_EOF'\n", 1)[1].split("\nYETO_UNIT_EOF\n", 1)[0]
    assert body + "\n" == fs.unit_text(spec("syncer"))  # the unit, verbatim
    commands = [line for line in script.replace(body, "").splitlines()
                if line and not line.startswith(("mode=", "case ", "echo", "YETO_UNIT_EOF"))]
    assert commands[1].startswith("sudo -n true 2>/dev/null || ") and "exit 4" in commands[1]  # sudo itself, first
    del commands[1]
    assert commands == ["set -eu", "sudo -n tee /etc/systemd/system/yeto-r-syncer.service >/dev/null <<'YETO_UNIT_EOF'",
                        "sudo -n systemctl daemon-reload", "sudo -n systemctl enable --now yeto-r-syncer.service"]
    # the env file: its mode is checked (as root: the file is the service user's), its content never read
    assert "sudo -n stat -c %a /etc/yeto/r.env" in script
    for verb in ("cat ", "source ", ". /etc", "grep ", "head ", "less "):
        assert verb not in script.replace(body, "")


_REAL_RUN = __import__("subprocess").run  # tests that fake ssh patch subprocess.run

FAKE_TOOLS = {
    "sudo": '[ "$1" = -n ] && shift\necho "sudo $*" >> "$LOG"\nexec "$@"',
    "stat": 'echo "stat $*" >> "$LOG"\necho "$FAKE_MODE"',
    "tee": 'echo "tee $*" >> "$LOG"\nwhile IFS= read -r _line; do :; done',
    "systemctl": 'echo "systemctl $*" >> "$LOG"',
    "systemd-run": 'echo "systemd-run $*" >> "$LOG"',
    "true": 'exit "${FAKE_SUDO_RC:-0}"',  # `sudo -n true`: whether sudo runs without a password
}


def run_script(tmp_path, script: str, mode: str, sudo_rc: int = 0) -> tuple[int, list[str], str]:
    """``script`` run by bash with only fake sudo / stat / tee / systemctl / systemd-run on PATH."""
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in FAKE_TOOLS.items():
        tool = bin_dir / name
        tool.write_text(f"#!/bin/bash\n{body}\n")
        tool.chmod(0o755)
    log = tmp_path / f"calls-{mode}.log"
    done = _REAL_RUN(["/bin/bash", "-s"], input=script, text=True, capture_output=True, timeout=30,
                          env={"PATH": str(bin_dir), "LOG": str(log), "FAKE_MODE": mode, "FAKE_SUDO_RC": str(sudo_rc)})
    calls = log.read_text().splitlines() if log.exists() else []
    return done.returncode, calls, done.stderr


@pytest.mark.parametrize("mode, ok", [("600", True), ("400", True), ("644", False), ("640", False),
                                      ("604", False), ("660", False)])
def test_the_install_script_refuses_an_env_file_others_can_read_before_it_writes_anything(tmp_path, mode, ok):
    code, calls, err = run_script(tmp_path, fs.install_script(spec("syncer"), fs.unit_text(spec("syncer"))), mode)
    # sudo itself first, then stat under sudo
    assert calls[:3] == ["sudo true", "sudo stat -c %a /etc/yeto/r.env", "stat -c %a /etc/yeto/r.env"]
    if ok:
        assert code == 0, err
        assert [c.split()[0] for c in calls[3:]] == ["sudo", "tee", "sudo", "systemctl", "sudo", "systemctl"]
        assert calls[-1] == "systemctl enable --now yeto-r-syncer.service"
    else:
        assert code == 3 and f"has mode {mode}" in err
        assert not any(c.startswith(("tee", "systemctl", "sudo tee")) for c in calls)  # nothing written


def test_an_install_that_fails_on_the_host_says_so():
    logs: list[str] = []
    assert fs.install(spec(), "site-a", runner=FakeSsh(rc=3), log=logs.append) == 1
    assert "failed (exit 3): mode 644" in logs[-1]
    with pytest.raises(fs.ServiceError):
        fs.install(spec(), "-oProxyCommand=x", runner=FakeSsh())


def test_the_cli_install_goes_through_ssh(tmp_path, monkeypatch, capsys):
    ssh = FakeSsh()
    monkeypatch.setattr(fs.subprocess, "run", ssh)
    path = onsite_config(tmp_path)
    args = fleet_cli_args(["fleet", "service", "--config", str(path), "--role", "control", "--env-file",
                           "/etc/yeto/r.env", "--install", "site-a", "--user", "yeto", "--python",
                           "/opt/yeto-venv/bin/python", "--yeto-dir", "/opt/yeto", "--remote-config",
                           "/srv/yeto/run.json", "--workdir", "/var/tmp/yeto/r-control"])
    assert fleet_cli.main(args) == 0
    ((argv, script),) = ssh.calls
    assert argv[-3:] == ["site-a", "bash", "-s"] and CONTROL_UNIT.rstrip("\n") in script


UNIT_FLAGS = ["--user", "yeto", "--python", "/opt/yeto-venv/bin/python", "--yeto-dir", "/opt/yeto",
              "--remote-config", "/srv/yeto/run.json"]


def test_stop_runs_stop_run_on_the_control_host_as_the_unit_runs(tmp_path, monkeypatch):
    ssh = FakeSsh()
    monkeypatch.setattr(fs.subprocess, "run", ssh)
    path = onsite_config(tmp_path)
    args = fleet_cli_args(["fleet", "service", "stop", "site-c", "--config", str(path), "--role", "control",
                           "--env-file", "/etc/yeto/r.env", *UNIT_FLAGS])
    assert fleet_cli.main(args) == 0
    ((argv, script),) = ssh.calls
    assert argv == ["ssh", "-o", "BatchMode=yes", "site-c", "bash", "-s"]
    code, calls, err = run_script(tmp_path, script, "600")
    assert code == 0, err
    (run_line,) = [c for c in calls if c.startswith("systemd-run")]
    assert run_line == ("systemd-run --quiet --wait --pipe --collect --uid=yeto "
                        "--working-directory=/var/tmp/yeto/r-control --property=EnvironmentFile=/etc/yeto/r.env "
                        "--setenv=PYTHONPATH=/opt/yeto --setenv=PYTHONUNBUFFERED=1 "
                        "--setenv=YETO_STATE_DIR=/var/tmp/yeto/r-control/rl-up "
                        "--setenv=YETO_RUNS_DIR=/var/tmp/yeto/r-control/runs "
                        "/opt/yeto-venv/bin/python -m yeto.cli stop-run r")
    assert run_script(tmp_path, script, "644")[0] == 3  # the same env file check as the install
    args.role = "syncer"
    assert fleet_cli.main(args) == 2  # the syncer service ends on the fleet's STOP; no stop-run there
    args.role, args.host = "control", None
    assert fleet_cli.main(args) == 2 and len(ssh.calls) == 1


def test_clear_ended_runs_on_the_service_host_as_the_unit_runs(tmp_path, monkeypatch):
    ssh = FakeSsh()
    monkeypatch.setattr(fs.subprocess, "run", ssh)
    path = onsite_config(tmp_path)
    args = fleet_cli_args(["fleet", "service", "clear-ended", "site-s", "--config", str(path), "--role", "syncer",
                           "--env-file", "/etc/yeto/r.env", *UNIT_FLAGS])
    assert fleet_cli.main(args) == 0
    ((_argv, script),) = ssh.calls
    assert script.splitlines()[-1].endswith("--setenv=YETO_RUNS_DIR=/var/tmp/yeto/r-syncer/runs "
                                            "/opt/yeto-venv/bin/python -m yeto.cli fleet service-run --role syncer "
                                            "--config /srv/yeto/run.json --clear-ended")
    args.print = True
    assert fleet_cli.main(args) == 2  # an action takes no --print


def test_the_control_unit_refuses_a_config_that_drains_on_exit(tmp_path, capsys):
    doc = onsite_doc(tmp_path)
    doc["rollout"]["drain_on_exit"] = True
    path = tmp_path / "drain.json"
    path.write_text(json.dumps(doc))
    args = fleet_cli_args(["fleet", "service", "--config", str(path), "--role", "control", "--env-file",
                           "/etc/yeto/r.env", "--print"])
    assert fleet_cli.main(args) == 2
    assert "rollout.drain_on_exit is true" in capsys.readouterr().err
    args.role = "syncer"
    assert fleet_cli.main(args) == 0  # the syncer service drives no pool


# -- the syncer launch watches the service ---------------------------------------------------------

class Clock:
    """This host's monotonic clock; ``sleep`` advances it (a cap on the calls: a loop that
    never ends fails the test instead of hanging it)."""

    def __init__(self, t: float = 1_000_000.0, max_sleeps: int = 2000) -> None:
        self.t, self.sleeps, self.max_sleeps = t, 0, max_sleeps

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps += 1
        assert self.sleeps <= self.max_sleeps, "the loop under test never ended"
        self.t += s


SKEW = -3 * 3600.0  # the service host's clock: three hours behind this host's (only its moves are read)


def watch_args(tmp_path):
    return types.SimpleNamespace(rl_fleet_syncer_service="10.0.0.5", cluster_prefix="r",
                                 rl_checkpoint_store=str(tmp_path / "store"), rl_fleet_syncer_grace_s=60.0,
                                 rl_syncer_epoch=None)


def beat(objects, clock, *, key=KEY, state=fleet_syncer.SERVICE_RUNNING, generation=7, host="10.0.0.5", **kw):
    fleet_syncer.write_service_status(objects, host=host, state=state, key=fleet_syncer.key_id(key),
                                      generation=generation, node="site-s", clock=lambda: clock() + SKEW, **kw)


def service_up(objects, clock, *, key=KEY, epoch=3, generation=7, record_generation=None, **kw):
    island_launches.write_syncer_record(
        objects, addr="10.0.0.5:29400", syncer_epoch=epoch, learners=2, max_learners=4, run="r",
        clock=lambda: clock() + SKEW,
        extra={"service": "10.0.0.5", "service_node": "site-s", "key_id": fleet_syncer.key_id(key),
               "service_generation": generation if record_generation is None else record_generation})
    beat(objects, clock, key=key, generation=generation, **kw)


def test_the_syncer_launch_waits_for_the_service_records_it_again_and_ends_with_it(tmp_path):
    import time

    objects, clock, logs = DirectoryObjectStore(tmp_path / "store"), Clock(), []
    args = watch_args(tmp_path)

    def sleep(s):
        clock.sleep(s)
        n = clock.sleeps
        if n == 3:
            service_up(objects, clock)  # the service comes up while the launch waits
        elif n > 3:
            beat(objects, clock, state=fleet_syncer.SERVICE_RUNNING if n < 8 else fleet_syncer.SERVICE_FINISHED)
            if n == 4:
                record = island_launches.read_syncer_record(objects)
                assert record["at"] > time.time() - 60  # announced again for this run up (this host's wall clock)
                assert record["syncer_epoch"] == 3 and record["service"] == "10.0.0.5"
                assert record["service_generation"] == 7 and record["service_node"] == "site-s"

    assert fleet_syncer.watch_service(args, objects=objects, sleep=sleep, clock=clock, log=logs.append) == 0
    assert args.rl_syncer_epoch == 3 and clock.sleeps == 8
    assert any("starts no syncer" in m for m in logs)


def test_a_heartbeat_that_never_moves_is_not_a_running_service(tmp_path):
    objects, clock, logs = DirectoryObjectStore(tmp_path / "store"), Clock(), []
    service_up(objects, clock)  # left by a service that is gone: "running", but never written again
    assert fleet_syncer.watch_service(watch_args(tmp_path), objects=objects, sleep=clock.sleep, clock=clock,
                                      wait_s=300, log=logs.append) == 1
    assert "no running syncer service" in logs[-1]
    assert island_launches.read_syncer_record(objects)["at"] < 1e7  # not announced again


@pytest.mark.parametrize("state", [fleet_syncer.SERVICE_STARTING, "running-generation-mismatch"])
def test_only_a_running_service_with_its_own_record_is_taken(tmp_path, state):
    objects, clock, logs = DirectoryObjectStore(tmp_path / "store"), Clock(), []

    def sleep(s):
        clock.sleep(s)
        if state == fleet_syncer.SERVICE_STARTING:
            service_up(objects, clock, state=fleet_syncer.SERVICE_STARTING)
        else:
            service_up(objects, clock, generation=8, record_generation=7)  # the record of the service before

    assert fleet_syncer.watch_service(watch_args(tmp_path), objects=objects, sleep=sleep, clock=clock,
                                      wait_s=120, log=logs.append) == 1
    assert "no running syncer service" in logs[-1]


@pytest.mark.parametrize("theirs, mine", [("another-key", KEY), (KEY, None)])
def test_a_key_that_is_not_the_service_s_is_refused_not_replaced(tmp_path, monkeypatch, theirs, mine):
    if mine is None:
        monkeypatch.delenv("YETO_ISLAND_HMAC_KEY")
    objects, clock, logs = DirectoryObjectStore(tmp_path / "store"), Clock(), []
    service_up(objects, clock, key=theirs)
    code = fleet_syncer.watch_service(watch_args(tmp_path), objects=objects, sleep=clock.sleep, clock=clock,
                                      log=logs.append)
    assert code == CONFIG_ERROR_EXIT
    assert ("not the syncer service's" if mine else "is not set on this host") in logs[-1]
    assert KEY not in "\n".join(logs) and "another-key" not in "\n".join(logs)  # ids only, never the key


def test_a_service_silent_past_the_grace_fails_the_launch_a_failed_one_at_once(tmp_path):
    objects, clock, logs = DirectoryObjectStore(tmp_path / "store"), Clock(), []
    service_up(objects, clock)

    def sleep(s):
        clock.sleep(s)
        if clock.sleeps == 1:
            beat(objects, clock)  # it beat once while the launch looked, then went silent

    assert fleet_syncer.watch_service(watch_args(tmp_path), objects=objects, sleep=sleep, clock=clock,
                                      log=logs.append) == 1
    assert any("no new heartbeat from the syncer service" in m for m in logs)
    clock2, logs2 = Clock(), []
    objects2 = DirectoryObjectStore(tmp_path / "store2")
    service_up(objects2, clock2)

    def sleep2(s):
        clock2.sleep(s)
        beat(objects2, clock2, state=fleet_syncer.SERVICE_FAILED if clock2.sleeps > 2 else fleet_syncer.SERVICE_RUNNING,
             reason="strict: merge mismatch")

    assert fleet_syncer.watch_service(watch_args(tmp_path), objects=objects2, sleep=sleep2, clock=clock2,
                                      log=logs2.append) == 1
    assert "strict: merge mismatch" in logs2[-1]
    assert clock2.sleeps < 10  # final at once, no grace


def test_a_heartbeat_of_another_host_is_down_not_up(tmp_path):
    objects, clock, logs = DirectoryObjectStore(tmp_path / "store"), Clock(), []
    service_up(objects, clock)

    def sleep(s):
        clock.sleep(s)
        beat(objects, clock, host="10.0.0.5" if clock.sleeps < 3 else "10.0.0.9")  # moving, but not this fleet's

    assert fleet_syncer.watch_service(watch_args(tmp_path), objects=objects, sleep=sleep, clock=clock,
                                      log=logs.append) == 1
    assert any("names host '10.0.0.9', not 10.0.0.5" in m for m in logs)


def test_the_run_s_stop_flag_is_the_fleet_s_stop_in_the_store_and_a_stopped_service_ends_the_launch(tmp_path):
    from yeto import runs
    from yeto.launch.fleet import RUN_STOPPED_EXIT

    objects, clock, logs = DirectoryObjectStore(tmp_path / "store"), Clock(), []
    service_up(objects, clock)

    def sleep(s):
        clock.sleep(s)
        beat(objects, clock)
        if clock.sleeps == 4:
            runs.request_stop("r")  # `yeto stop-run r` on the control host

    assert fleet_syncer.watch_service(watch_args(tmp_path), objects=objects, sleep=sleep, clock=clock,
                                      log=logs.append) == RUN_STOPPED_EXIT
    stop = fleet_syncer.read_onsite_stop(objects)
    assert stop is not None and "STOP flag of run r" in stop["why"]
    objects2, clock2 = DirectoryObjectStore(tmp_path / "store2"), Clock()
    service_up(objects2, clock2)

    def sleep2(s):
        clock2.sleep(s)
        beat(objects2, clock2, state=fleet_syncer.SERVICE_STOPPED if clock2.sleeps > 3 else fleet_syncer.SERVICE_RUNNING,
             reason="the fleet's STOP")

    (tmp_path / "runs" / "r" / "STOP").unlink()
    assert fleet_syncer.watch_service(watch_args(tmp_path), objects=objects2, sleep=sleep2, clock=clock2,
                                      log=logs.append) == RUN_STOPPED_EXIT


def test_no_service_within_the_wait_starts_no_island(tmp_path):
    objects, clock, logs = DirectoryObjectStore(tmp_path / "store"), Clock(), []
    assert fleet_syncer.watch_service(watch_args(tmp_path), objects=objects, sleep=clock.sleep, clock=clock,
                                      wait_s=60, log=logs.append) == 1
    assert "no island was started" in logs[-1] and island_launches.read_syncer_record(objects) is None


# -- the syncer service ---------------------------------------------------------------------------------

class FakeSyncer:
    def __init__(self, strict: bool = False) -> None:
        self.events: list[str] = []
        self.strict = strict

    def start(self):
        self.events.append("start")

    def stop(self):
        self.events.append("stop")

    def probe(self):
        from yeto.launch.fleet import _RlStrictFailure

        return _RlStrictFailure("merge_invariant: broken") if self.strict else None


def fake_serve(objects, seen, code, *, until=None):
    """``fleet_syncer.serve`` as far as the service sees it: the record (with the service's
    fields), ``on_record``, then ``code`` (or 9 once ``stopped()`` says so, with ``until``)."""
    import time

    def serve(args, addr, n, *, clusters, head_mode, local_syncer, record_extra, on_record, stopped, early_finish,
              **kw):
        seen.update(addr=addr, n=n, head_mode=head_mode, extra=record_extra, epoch=args.rl_syncer_epoch,
                    service=getattr(args, "rl_fleet_syncer_service", None),
                    before=fleet_syncer.read_service_status(objects))
        record = island_launches.write_syncer_record(objects, addr=addr, syncer_epoch=args.rl_syncer_epoch, learners=n,
                                                     max_learners=4, run="r", extra=record_extra)
        on_record(record)
        seen["after"] = fleet_syncer.read_service_status(objects)
        if until is not None:
            deadline = time.monotonic() + 20
            while not stopped():
                until()
                assert time.monotonic() < deadline
                time.sleep(0.02)
            seen["stopped"] = stopped()
            return 9
        return code

    return serve


def run_service(tmp_path, monkeypatch, code: int, *, strict: bool = False, objects=None, node="site-s",
                until=None, beat_s=None, workdir=None):
    path = onsite_config(tmp_path)
    objects = objects or DirectoryObjectStore(tmp_path / "store")
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    seen: dict = {}
    syncer = FakeSyncer(strict)
    logs: list[str] = []
    workdir = str(workdir or tmp_path / "work")

    def make_syncer(args, n):  # the syncer it starts writes its checkpoint (under the service's home)
        for c in fs._checkpoints(args, str(pathlib.Path(workdir) / "home")):
            pathlib.Path(c).parent.mkdir(parents=True, exist_ok=True)
            pathlib.Path(c).write_text("{}")
        return syncer

    rc = fs.run_syncer(str(path), objects=objects, make_syncer=make_syncer,
                       serve=fake_serve(objects, seen, code, until=until), parse=_prepared, node=node,
                       beat_s=beat_s, workdir=workdir, log=logs.append)
    return rc, seen, syncer, objects, logs


def test_the_service_runs_the_syncer_launch_s_syncer_here_and_says_when_it_finished(tmp_path, monkeypatch):
    from yeto.launch.pool.store_lock import StoreLock

    rc, seen, syncer, objects, logs = run_service(tmp_path, monkeypatch, 0)
    assert rc == 0 and syncer.events == ["start", "stop"]
    assert seen["addr"] == "10.0.0.5:29400" and seen["n"] == 2 and seen["head_mode"] is True
    assert seen["service"] is None  # the syncer runs here: the launch's own args, not a watcher's
    assert seen["extra"] == {"service": "10.0.0.5", "service_node": "site-s", "service_generation": 1,
                             "key_id": fleet_syncer.key_id(KEY)}
    # starting until its record is written, then running (the generation is the lock's token)
    assert seen["before"]["state"] == "starting" and seen["after"]["state"] == "running"
    assert seen["after"]["generation"] == 1 and seen["after"]["key_id"] == fleet_syncer.key_id(KEY)
    assert fleet_syncer.read_service_status(objects)["state"] == "finished"
    assert KEY not in (tmp_path / "store" / fleet_syncer.SERVICE_RECORD).read_text()
    assert StoreLock.holder_of(objects, fleet_syncer.SERVICE_LOCK) is None  # released at the end


def test_a_strict_failure_of_the_service_is_final_anything_else_restarts(tmp_path, monkeypatch):
    rc, _seen, _syncer, objects, logs = run_service(tmp_path, monkeypatch, 1, strict=True)
    assert rc == ISLAND_STOP_EXIT and rc in fs.SYNCER_PREVENT
    status = fleet_syncer.read_service_status(objects)
    assert status["state"] == "failed" and "merge_invariant" in status["reason"]
    (tmp_path / "again").mkdir()
    rc, _seen, _syncer, objects, _logs = run_service(tmp_path / "again", monkeypatch, 1)
    assert rc == 1 and rc not in fs.SYNCER_PREVENT  # the unit restarts it; it resumes from its checkpoint
    assert fleet_syncer.read_service_status(objects)["state"] == "running"


def test_one_syncer_service_at_a_time(tmp_path, monkeypatch):
    from yeto.launch.pool.store_lock import StoreLock

    objects = DirectoryObjectStore(tmp_path / "store")
    other = StoreLock.acquire(objects, fleet_syncer.SERVICE_LOCK, holder="site-t/yeto-r-syncer.service", ttl_s=120)
    rc, seen, syncer, _objects, logs = run_service(tmp_path, monkeypatch, 0, objects=objects)
    assert rc == 2 and rc in fs.SYNCER_PREVENT and syncer.events == [] and seen == {}
    assert "another syncer service of run r runs (site-t/yeto-r-syncer.service" in logs[-1]
    other.release()
    # this host's own lease (a service that crashed: the unit restarts it at once) is taken back
    StoreLock.acquire(objects, fleet_syncer.SERVICE_LOCK, holder="site-s/yeto-r-syncer.service", ttl_s=120)
    rc, seen, syncer, _objects, logs = run_service(tmp_path, monkeypatch, 0, objects=objects)
    assert rc == 0 and seen["extra"]["service_generation"] == 3  # a new token: the crashed one is fenced


def test_a_service_whose_lock_another_took_ends_without_a_final_state(tmp_path, monkeypatch):
    from yeto.launch.pool.store_lock import StoreLock

    objects = DirectoryObjectStore(tmp_path / "store")
    taken = []

    def take_over():
        if not taken:  # its host was cut off past its lease: another service took the lock
            taken.append(StoreLock.acquire(objects, fleet_syncer.SERVICE_LOCK, holder="site-s/yeto-r-syncer.service",
                                           ttl_s=120, reclaim=True))

    rc, seen, syncer, objects, logs = run_service(tmp_path, monkeypatch, 0, objects=objects, until=take_over,
                                                  beat_s=0.05)
    assert rc == 2 and "no longer this service's" in seen["stopped"]
    assert fleet_syncer.read_service_status(objects)["state"] == "running"  # never final: not this service's now
    assert syncer.events[:2] == ["start", "stop"] and set(syncer.events[1:]) == {"stop"}


def test_the_term_goes_on_here_and_moves_up_on_another_host(tmp_path, monkeypatch):
    monkeypatch.setattr(fs, "_checkpoints", lambda args, home: [home + "/ckpt.json"])
    (tmp_path / "work" / "home").mkdir(parents=True)
    (tmp_path / "work" / "home" / "ckpt.json").write_text("{}")  # this host's syncer ran it before
    objects = DirectoryObjectStore(tmp_path / "store")
    island_launches.write_syncer_record(objects, addr="10.0.0.5:29400", syncer_epoch=4, learners=2, max_learners=4,
                                        run="r", extra={"service": "10.0.0.5", "service_node": "site-s"})
    rc, seen, *_ = run_service(tmp_path, monkeypatch, 0, objects=objects)
    assert rc == 0 and seen["epoch"] == 4  # this host's service, restarted: the same term
    (tmp_path / "b").mkdir()
    rc, seen, _syncer, _objects, logs = run_service(tmp_path / "b", monkeypatch, 0, objects=objects, node="site-u")
    assert rc == 0 and seen["epoch"] == 5  # another host: one more, fencing the old syncer's islands
    assert any("not this host's service's" in m for m in logs)


def test_the_fleet_s_stop_in_the_store_ends_the_service(tmp_path, monkeypatch):
    objects = DirectoryObjectStore(tmp_path / "store")

    def stop():
        if fleet_syncer.read_onsite_stop(objects) is None:
            fleet_syncer.write_onsite_stop(objects, why="STOP flag of run r", by="site-c")

    rc, seen, syncer, objects, logs = run_service(tmp_path, monkeypatch, 0, objects=objects, until=stop)
    assert rc == 9 and rc in fs.SYNCER_PREVENT and "STOP flag of run r" in seen["stopped"]
    status = fleet_syncer.read_service_status(objects)
    assert status["state"] == "stopped" and "by site-c" in status["reason"]
    (tmp_path / "again").mkdir()  # started again (a reboot) while the STOP stands: the syncer never starts
    rc, seen, syncer, objects, logs = run_service(tmp_path / "again", monkeypatch, 0, objects=objects,
                                                  workdir=tmp_path / "work")
    assert rc == 9 and syncer.events == [] and seen == {}


def test_the_service_refuses_without_the_key_and_never_generates_one(tmp_path, monkeypatch):
    monkeypatch.delenv("YETO_ISLAND_HMAC_KEY")
    path = onsite_config(tmp_path)
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    logs: list[str] = []
    assert fs.run_syncer(str(path), objects=DirectoryObjectStore(tmp_path / "store"),
                         make_syncer=lambda *a: pytest.fail("no syncer without the key"),
                         log=logs.append) == CONFIG_ERROR_EXIT
    assert "none is generated" in logs[-1]
    import os

    assert "YETO_ISLAND_HMAC_KEY" not in os.environ


def test_the_service_s_environment_is_run_up_s_and_never_written_to_this_process(tmp_path, monkeypatch):
    import os

    from yeto.launch import local_syncer

    doc = onsite_doc(tmp_path)
    doc["env"].update({"YETO_ONSITE_PROBE": "from-the-run-config", "YETO_ONSITE_CONFIG_ONLY": "config-only"})
    path = tmp_path / "env.json"
    path.write_text(json.dumps(doc))
    monkeypatch.setenv("YETO_ONSITE_PROBE", "from-the-env-file")
    monkeypatch.setenv("YETO_ONSITE_FILE_ONLY", "file-only")
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    made = {}

    class Captured(FakeSyncer):
        def __init__(self, args, n, *, binary, log_file, env, stop_grace_s, home):
            super().__init__()
            made.update(env=env, stop_grace_s=stop_grace_s, home=home)

    monkeypatch.setattr(local_syncer, "LocalSyncer", Captured)
    objects = DirectoryObjectStore(tmp_path / "store")
    before = dict(os.environ)
    assert fs.run_syncer(str(path), binary="/opt/yeto-syncer", objects=objects, parse=_prepared, node="site-s",
                         serve=fake_serve(objects, {}, 0), stop_grace_s=170, log=lambda m: None) == 0
    env = made["env"]
    # the run config's env over the env file's (run up's order), both in the syncer's environment
    assert env["YETO_ONSITE_PROBE"] == "from-the-run-config" and env["YETO_ONSITE_FILE_ONLY"] == "file-only"
    assert env["YETO_ONSITE_CONFIG_ONLY"] == "config-only" and env["YETO_ISLAND_HMAC_KEY"] == KEY
    assert made["stop_grace_s"] == 170
    assert dict(os.environ) == before  # nothing written to this process's environment
    run = up.Run(up.load_config(str(path)), state_root=tmp_path / "run-up")  # the control unit: the same order
    assert run.env["YETO_ONSITE_PROBE"] == env["YETO_ONSITE_PROBE"]
    assert fs.Onsite.open(str(path), "control").env["YETO_ONSITE_PROBE"] == "from-the-run-config"


def test_the_syncer_s_stop_waits_its_grace_then_kills(tmp_path):
    import subprocess
    import time

    from yeto.launch.local_syncer import LocalSyncer

    syncer = LocalSyncer.__new__(LocalSyncer)
    syncer.command = "trap '' TERM; sleep 30"  # a syncer that does not end on SIGTERM
    syncer.binary, syncer.log_file = str(tmp_path / "none"), str(tmp_path / "syncer.log")
    syncer.proc, syncer._reaped, syncer.stop_grace_s, syncer.env = None, False, 1.5, None
    syncer.start()
    time.sleep(0.3)
    started = time.monotonic()
    syncer._stop_tree()
    took = time.monotonic() - started
    assert syncer.proc.poll() is not None and 1.4 <= took < 8
    assert isinstance(syncer.proc, subprocess.Popen)


# -- what the units run: the ended marker ------------------------------------------------------------

def service_args(path, role, *, clear_ended=False, fresh=False):
    return types.SimpleNamespace(role=role, config=str(path), syncer_binary=None, stop_grace_s=170,
                                 clear_ended=clear_ended, fresh=fresh)


@pytest.fixture
def no_sigterm_handler(monkeypatch):
    monkeypatch.setattr(fs.signal, "signal", lambda *a: None)  # not this test process's SIGTERM


@pytest.mark.usefixtures("no_sigterm_handler")
def test_a_control_service_whose_run_up_ended_the_run_never_orders_again(tmp_path, monkeypatch):
    path = onsite_config(tmp_path)
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    objects, ups, logs = DirectoryObjectStore(tmp_path / "store"), [], []

    def run_up(args):  # run up: a budget breaker tripped, it tore the run down and exits 1
        ups.append(args.config)
        return 1

    monkeypatch.setattr(up, "main", run_up)
    args = service_args(path, "control")
    assert fs.service_run(args, objects=objects, log=logs.append) == fs.CONTROL_DECIDED_EXIT
    marker = json.loads((tmp_path / "state" / "r" / "onsite-control.ended").read_text())
    assert marker["exit_code"] == 75 and "breaker" in marker["reason"] and marker["at"] > 0
    assert json.loads(objects.read("fleet/onsite/control.ended"))["exit_code"] == 75
    # the host reboots: systemd starts the unit again, and nothing is ordered
    assert fs.service_run(args, objects=objects, log=logs.append) == 0 and len(ups) == 1
    assert "already ended" in logs[-1] and "clear-ended" in logs[-1]
    # the unit started on another host (no local marker): the run store's marker holds
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "other-host")
    assert fs.service_run(args, objects=objects, log=logs.append) == 0 and len(ups) == 1
    assert fs.service_run(service_args(path, "control", clear_ended=True), objects=objects, log=logs.append) == 0
    assert objects.read("fleet/onsite/control.ended") is None
    assert fs.service_run(args, objects=objects, log=logs.append) == 75 and len(ups) == 2  # starts again


@pytest.mark.usefixtures("no_sigterm_handler")
def test_a_syncer_service_after_a_strict_failure_never_resumes(tmp_path, monkeypatch):
    path = onsite_config(tmp_path)
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    objects, logs = DirectoryObjectStore(tmp_path / "store"), []
    real = fs.run_syncer

    def strict(config, **kw):
        return real(config, objects=objects, make_syncer=lambda args, n: FakeSyncer(strict=True),
                    serve=fake_serve(objects, {}, 1), parse=_prepared, node="site-s", log=logs.append)

    monkeypatch.setattr(fs, "run_syncer", strict)
    args = service_args(path, "syncer")
    assert fs.service_run(args, objects=objects, log=logs.append) == ISLAND_STOP_EXIT
    assert json.loads((tmp_path / "state" / "r" / "onsite-syncer.ended").read_text())["exit_code"] == 11
    monkeypatch.setattr(fs, "run_syncer", lambda *a, **kw: pytest.fail("a syncer that failed strict is not resumed"))
    assert fs.service_run(args, objects=objects, log=logs.append) == 0
    fleet_syncer.write_onsite_stop(objects, why="x", by="site-c")
    assert fs.service_run(service_args(path, "syncer", clear_ended=True), objects=objects, log=logs.append) == 0
    assert fleet_syncer.read_onsite_stop(objects) is None  # the syncer's clear also lifts the fleet's STOP


@pytest.mark.usefixtures("no_sigterm_handler")
@pytest.mark.parametrize("code", [2, 10, 1, fs.CRASH_EXIT])
def test_a_refusal_or_a_crash_leaves_no_ended_marker(tmp_path, monkeypatch, code):
    path = onsite_config(tmp_path)
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    objects = DirectoryObjectStore(tmp_path / "store")
    monkeypatch.setattr(fs, "run_syncer", (lambda *a, **kw: 1 / 0) if code == fs.CRASH_EXIT else
                        (lambda *a, **kw: code))
    assert fs.service_run(service_args(path, "syncer"), objects=objects, log=lambda m: None) == code
    assert not (tmp_path / "state" / "r" / "onsite-syncer.ended").exists()
    assert objects.read("fleet/onsite/syncer.ended") is None


@pytest.mark.usefixtures("no_sigterm_handler")
def test_a_marker_that_does_not_read_is_waited_for_never_a_crash(tmp_path, monkeypatch):
    """A run store that does not answer for two hours: the service keeps asking in the process
    (5 s, backing off to 60 s, a line per try), never exits for it (a unit restarted for it would
    spend its start limit), and starts once it answers."""
    clock = Clock(0.0, max_sleeps=500)

    class Down(DirectoryObjectStore):
        def read(self, key):
            if clock.t < 7200:
                raise OSError("store down")
            return super().read(key)

    path = onsite_config(tmp_path)
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(fs, "_sleep", clock.sleep)
    started, logs = [], []
    monkeypatch.setattr(fs, "run_control", lambda *a, **kw: started.append(clock.t) or 0)
    assert fs.service_run(service_args(path, "control"), objects=Down(tmp_path / "store"), log=logs.append) == 0
    assert started and 7200 <= started[0] < 7300
    tries = [m for m in logs if "does not answer" in m]
    assert len(tries) == clock.sleeps > 100 and "try 1)" in tries[0]
    assert "in 5s" in tries[0] and "in 10s" in tries[1] and all("in 60s" in m for m in tries[5:])


@pytest.mark.usefixtures("no_sigterm_handler")
def test_a_store_that_answers_three_minutes_late_is_waited_for_in_the_process(tmp_path, monkeypatch):
    clock = Clock(0.0)

    class Late(DirectoryObjectStore):
        def read(self, key):
            if clock.t < 180:
                raise OSError("store not up yet")
            return super().read(key)

    path = onsite_config(tmp_path)
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(fs, "_sleep", clock.sleep)
    started, logs = [], []
    monkeypatch.setattr(fs, "run_control", lambda *a, **kw: started.append(clock.t) or 2)
    assert fs.service_run(service_args(path, "control"), objects=Late(tmp_path / "store"), log=logs.append) == 2
    assert started and 180 <= started[0] < 300  # started once the store answered: never a failed unit
    assert sum("does not answer" in m for m in logs) >= 3


@pytest.mark.usefixtures("no_sigterm_handler")
def test_the_control_service_maps_run_up_s_exits(tmp_path, monkeypatch):
    path = onsite_config(tmp_path)
    seen = []

    def run_up(args):
        cfg = up.load_config(args.config)
        args.config_adjust(cfg)
        seen.append(cfg["rollout"]["drain_on_exit"])
        code = codes.pop(0)
        if code == "config":  # run up refused its configuration (RunConfigError)
            args.config_refused = "bad config"
            return 2
        return code

    codes = [1, 2, 0, 143, "config"]
    monkeypatch.setattr(up, "main", run_up)
    logs: list[str] = []
    assert [fs.run_control(str(path), log=logs.append) for _ in range(5)] == [75, 2, 0, 143, fs.CONFIG_EXIT]
    assert 2 not in fs.CONTROL_PREVENT  # a refusal before ordering: restarted (it costs nothing)
    assert seen == [False] * 5  # never drains the pools on exit
    assert any("rollout.drain_on_exit is not set" in m for m in logs)


def test_the_control_unit_runs_run_up_without_the_exit_drain(tmp_path, monkeypatch):
    path = onsite_config(tmp_path)
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    logs: list[str] = []
    run = up.open_run(str(path), adjust=lambda cfg: fs.no_exit_drain(cfg, logs.append))
    assert up.rollout_section(run.cfg)["drain_on_exit"] is False and len(logs) == 1
    doc = onsite_doc(tmp_path)
    doc["rollout"]["drain_on_exit"] = True
    explicit = tmp_path / "drain.json"
    explicit.write_text(json.dumps(doc))
    with pytest.raises(up.RunConfigError, match="drain_on_exit is true"):
        up.open_run(str(explicit), adjust=lambda cfg: fs.no_exit_drain(cfg, logs.append))
    monkeypatch.setattr(fs.signal, "signal", lambda *a: None)
    assert fs.service_run(service_args(explicit, "control"), objects=DirectoryObjectStore(tmp_path / "s"),
                          log=logs.append) == fs.CONFIG_EXIT  # a configuration error: never restarted
    assert "rollout.drain_on_exit is true" in logs[-1] and fs.CONFIG_EXIT in fs.CONTROL_PREVENT
    # the real run up refuses it too (its open_run): a configuration error, never restarted
    assert fs.run_control(str(explicit), log=logs.append) == fs.CONFIG_EXIT
    assert "refused its configuration" in logs[-1]


def test_the_documented_commands_run(tmp_path, monkeypatch, capsys):
    """docs/OWN_CLUSTERS.md, "On-site control plane": every `yeto fleet service` command parses,
    the --print ones write a unit, the others go through (a fake) ssh only."""
    import re
    from pathlib import Path

    doc = (Path(__file__).resolve().parents[1] / "docs" / "OWN_CLUSTERS.md").read_text()
    section = doc.split("## On-site control plane", 1)[1].split("\n## ", 1)[0]
    commands = re.findall(r"^\s*yeto (fleet service .+)$", section, flags=re.M)
    assert len(commands) == 6
    config = onsite_config(tmp_path)
    ssh = FakeSsh()
    monkeypatch.setattr(fs.subprocess, "run", ssh)
    for line in commands:
        argv = (line.split("#", 1)[0].replace("run.json", str(config)).replace("<run>", "r")
                .replace("<syncer-host>", "site-s").replace("<control-host>", "site-c").split())
        assert fleet_cli.main(fleet_cli_args(argv)) == 0, line
    out = capsys.readouterr().out
    assert out.count("[Service]") == 2
    assert [c[0][3] for c in ssh.calls] == ["site-s", "site-c", "site-s", "site-c"]


def recording_controller(drain_on_exit):
    """A real RolloutController (its drain decisions are drain_active's and drain_controller's
    own) with one pool on the store control plane; its stop and detach are only recorded."""
    from yeto.launch.pool import controller as ctl

    class Recording(ctl.RolloutController):
        def __init__(self):
            super().__init__({}, drain_on_exit=drain_on_exit, log=lambda m: None)
            self.slots = {"i0-sglang": types.SimpleNamespace(pool=types.SimpleNamespace(control=object()))}
            self.calls = []

        def stop(self, why, **kw):
            self.calls.append(("stop", why))
            self.stopped = why
            return {}

        def detach(self, why, **kw):
            self.calls.append(("detach", why))
            self.detached = why
            return {}

    controller = Recording()
    with ctl._ACTIVE_LOCK:
        ctl._ACTIVE.append(controller)  # as RolloutController.start registers it
    return controller


@pytest.fixture
def active_controllers():
    from yeto.launch.pool import controller as ctl

    before = list(ctl._ACTIVE)
    yield
    with ctl._ACTIVE_LOCK:
        ctl._ACTIVE[:] = before


@pytest.mark.usefixtures("no_sigterm_handler", "active_controllers")
@pytest.mark.parametrize("end, code", [("stop-run", 0), ("fleet stop", 0), ("trainer exit", 0),
                                       ("budget breaker", 1), ("other breaker", 1), ("syncer exit 9", 0),
                                       ("syncer exit 11", 1), ("sigterm", None), ("crash", "crash")])
def test_the_control_service_drains_when_run_up_ends_the_run_and_keeps_the_pools_on_a_sigterm(
        tmp_path, monkeypatch, end, code):
    """The on-site control unit runs run up with rollout.drain_on_exit false, but only a SIGTERM
    (a stop or restart of the unit) or a crash leaves the pools running: an end run up decides
    itself (a stop-run, the fleet's STOP, the trainer's exit, a breaker, the syncer's end) stops
    the controller, every pool drained. drain_active is the real one (it only detaches a
    controller with drain_on_exit false), and so is the interpreter-exit drain after it."""
    from yeto.launch.pool import controller as ctl

    path = onsite_config(tmp_path)
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    objects, made = DirectoryObjectStore(tmp_path / "store"), []
    monkeypatch.setattr(up.Run, "_objects", lambda self, *a, **kw: objects)  # the run store (its state mirror)

    def pool(self):
        self.controller = recording_controller(bool(up.rollout_section(self.cfg).get("drain_on_exit", True)))
        made.append(self.controller)

    for name in ("preflight", "cluster_preflight", "write_fleet_plan", "sandbox_prepare"):
        monkeypatch.setattr(up.Run, name, lambda self, *a, **kw: True)
    monkeypatch.setattr(up.Run, "log_fleet_plan", lambda self: None)
    monkeypatch.setattr(up.Run, "pool", pool)

    def drive(run, args):
        if code is None:
            raise SystemExit(143)  # SIGTERM: the unit is stopped or restarted
        if code == "crash":
            raise RuntimeError("a bug")
        return code

    monkeypatch.setattr(up, "_drive", drive)
    args = service_args(path, "control")
    marker = tmp_path / "state" / "r" / "onsite-control.ended"
    if code is None:
        with pytest.raises(SystemExit):
            fs.service_run(args, objects=objects, log=lambda m: None)
    else:
        rc = fs.service_run(args, objects=objects, log=lambda m: None)
        assert rc == {0: 0, 1: fs.CONTROL_DECIDED_EXIT, "crash": fs.CRASH_EXIT}[code]
    (controller,) = made
    assert controller.drain_on_exit is False  # the unit's run up never drains on a SIGTERM or a crash
    ctl.drain_active("process exit")  # the interpreter exit after it
    if code in (0, 1):
        assert controller.calls == [("stop", "run up ended the run")] and marker.exists()  # drained, run over
        assert controller not in ctl._ACTIVE
    else:
        assert not marker.exists()
        assert [c for c in controller.calls if c[0] == "stop"] == []  # the pools stay, detached
        assert controller.calls[0] == ("detach", "process exit")


@pytest.mark.usefixtures("active_controllers")
def test_a_plain_run_up_drains_as_before(tmp_path):
    """Without the service mode drain_controller is what it was: drain_on_exit decides (the real
    drain_active)."""
    run = up.Run(up.load_config(str(onsite_config(tmp_path))), state_root=tmp_path / "state")
    run.controller = kept = recording_controller(False)
    assert run.drain_controller(decided=True) is True and kept.calls == [("detach", "process exit")]  # not the service
    run.controller = kept = recording_controller(False)
    run.onsite_service = True
    assert run.drain_controller(decided=False) is True and kept.calls == [("detach", "process exit")]  # a SIGTERM
    run.controller = ended = recording_controller(False)
    assert run.drain_controller(decided=True) is True and ended.calls == [("stop", "run up ended the run")]
    run.onsite_service = False
    run.controller = drained = recording_controller(True)
    assert run.drain_controller(decided=False) is True and drained.calls == [("stop", "process exit")]


@pytest.mark.usefixtures("no_sigterm_handler")
def test_a_stopped_syncer_service_is_ended_too(tmp_path, monkeypatch):
    path = onsite_config(tmp_path)
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    objects = DirectoryObjectStore(tmp_path / "store")
    monkeypatch.setattr(fs, "run_syncer", lambda *a, **kw: 9)
    assert fs.service_run(service_args(path, "syncer"), objects=objects, log=lambda m: None) == 9
    marker = json.loads((tmp_path / "state" / "r" / "onsite-syncer.ended").read_text())
    assert marker["exit_code"] == 9 and "STOP" in marker["reason"]
    assert json.loads(objects.read("fleet/onsite/syncer.ended"))["exit_code"] == 9


def test_sudo_that_asks_for_a_password_is_said_as_such(tmp_path):
    for n, script in enumerate((fs.install_script(spec("syncer"), fs.unit_text(spec("syncer"))),
                                fs.as_unit_script(spec("control"), ["stop-run", "r"]))):
        (tmp_path / str(n)).mkdir()
        code, calls, err = run_script(tmp_path / str(n), script, "600", sudo_rc=1)
        assert code == fs.SUDO_EXIT and "sudo -n does not run on this host" in err
        assert calls == ["sudo true"]  # nothing else: no stat, no tee, no systemd-run


def test_stop_refuses_a_run_name_a_command_cannot_hold(tmp_path, monkeypatch, capsys):
    ssh = FakeSsh()
    monkeypatch.setattr(fs.subprocess, "run", ssh)
    monkeypatch.setattr(fs, "trainer_launch_name", lambda cfg: "r;id")
    args = fleet_cli_args(["fleet", "service", "stop", "site-c", "--config", str(onsite_config(tmp_path)), "--role",
                           "control", "--env-file", "/etc/yeto/r.env", *UNIT_FLAGS])
    assert fleet_cli.main(args) == 2 and ssh.calls == []
    assert "letters, digits" in capsys.readouterr().err


def test_the_watcher_never_overwrites_a_record_the_service_rewrote_meanwhile(tmp_path):
    objects, clock, logs = DirectoryObjectStore(tmp_path / "store"), Clock(), []
    service_up(objects, clock)
    writes = []
    real_write = objects.write

    def write(key, data):
        if key == "fleet/island_syncer.json":
            writes.append(json.loads(data).get("service_generation"))
        real_write(key, data)

    objects.write = write
    real_read, reads = objects.read, {"n": 0}

    def read(key):
        data = real_read(key)
        if key == "fleet/island_syncer.json":
            reads["n"] += 1
            if reads["n"] == 2:  # right after the read that found it running: the service restarts (generation 8)
                service_up(objects, clock, generation=8)
        return data

    objects.read = read

    def sleep(s):
        clock.sleep(s)
        beat(objects, clock, generation=8 if reads["n"] >= 2 else 7,
             state=fleet_syncer.SERVICE_FINISHED if clock.sleeps > 6 else fleet_syncer.SERVICE_RUNNING)

    assert fleet_syncer.watch_service(watch_args(tmp_path), objects=objects, sleep=sleep, clock=clock,
                                      log=logs.append) == 0
    assert 7 not in writes[1:]  # after the service's own generation-8 record, never generation 7 again
    assert writes[-1] == 8


def test_a_lease_that_could_not_be_renewed_fences_the_service_in_time_and_restartably(tmp_path):
    """A renewal that fails fences the service once the lease left before it no longer covers the
    beat thread's worst case (three cut-off calls and a beat), a poll, the syncer's stop and the
    lock's margin (exit 1: restarted); another holder's token is exit 2 (not restarted). The
    lease leaves two worst-case heartbeat rounds before the fence."""
    from yeto.launch.pool.store_lock import LockLost

    class Lock:
        margin_s = 12.0

        def __init__(self, left, error):
            self.left, self.error = left, error

        def renew(self):
            self.left -= 5.0  # the call took its time: the lease left before it is what counts
            raise self.error

        def lease_left(self):
            return self.left

    objects = DirectoryObjectStore(tmp_path / "store")
    grace, call = fs.SYNCER_STOP_GRACE_S, fleet_syncer.HEARTBEAT_CALL_S
    assert call == 2 * (fleet_syncer.HEARTBEAT_MAX_ATTEMPTS * (fleet_syncer.HEARTBEAT_CONNECT_TIMEOUT_S
                                                                + fleet_syncer.HEARTBEAT_READ_TIMEOUT_S) + 1.0)
    fence = 12.0 + fleet_syncer.SERVICE_BEAT_S + fleet_syncer.POLL_S + grace + 3 * call
    for left, error, code in [(fence + 1, OSError("store blip"), None), (fence, OSError("store down"), fs.SELF_FENCED_EXIT),
                              (300.0, LockLost("fleet/syncer_service/lock", 1, 2), fs.LOCK_LOST_EXIT)]:
        beat_ = fs.Heartbeat(objects, "10.0.0.5", "kid", lock=Lock(left, error), generation=1, log=lambda m: None,
                             stop_grace_s=grace)
        assert beat_.fence_s == fence
        beat_.renew()
        assert beat_.lost_code == code and (beat_.lost is not None) is (code is not None)
    assert fs.SELF_FENCED_EXIT not in fs.SYNCER_PREVENT and fs.LOCK_LOST_EXIT in fs.SYNCER_PREVENT
    ttl = fs.service_lock_ttl_s(grace)
    margin = min(60.0, ttl * 0.1)  # StoreLock.margin_s
    fence_s = margin + fleet_syncer.SERVICE_BEAT_S + fleet_syncer.POLL_S + grace + 3 * call
    assert ttl - fence_s >= 2 * (fleet_syncer.SERVICE_BEAT_S + 2 * call) - 1e-9  # two slow rounds before the fence
    assert ttl >= fleet_syncer.SERVICE_LOCK_TTL_S >= 4 * fleet_syncer.SERVICE_BEAT_S + 7 * call


class HangingStore:
    """A store whose calls, once ``down``, each block ``block_s`` and then fail (a store that
    hangs: the heartbeat's cut-off ends the wait first when it is shorter)."""

    def __init__(self, inner, block_s):
        self.inner, self.block_s, self.down = inner, float(block_s), False

    def _call(self, name, *a):
        import time

        if self.down:
            time.sleep(self.block_s)
            raise OSError(f"the store did not answer the {name}")
        return getattr(self.inner, name)(*a)

    def read(self, key):
        return self._call("read", key)

    def write(self, key, data):
        return self._call("write", key, data)

    def create(self, key, data):
        return self._call("create", key, data)

    def list(self, prefix):
        return self._call("list", prefix)

    def delete(self, key):
        return self._call("delete", key)


# the production budget scaled down (seconds x SCALE): a store that blocks N seconds per call
SCALE = 0.01


@pytest.mark.parametrize("block_s, main_hangs", [(0, False), (45, False), (91, False), (181, False),
                                                 (15, True), (60, True), (181, True), (585, True)])
def test_a_store_that_hangs_never_keeps_the_syncer_running_past_the_lease(tmp_path, monkeypatch, block_s, main_hangs):
    """H1: each heartbeat store call (and with ``main_hangs`` each call of the service's main
    client too) blocks ``block_s`` (production seconds, scaled) and then fails. The beat thread
    stops the syncer at the fence itself, its stop taking the whole grace, more than the lock's
    margin before the lease ends, every time; the main thread's store calls (the STOP read, a
    record) are cut off, so it ends soon after."""
    import time

    from yeto.launch.pool.store_lock import StoreLock

    monkeypatch.setattr(fleet_syncer, "SERVICE_LOCK_TTL_S", 0.0)  # the scaled lease, not the floor
    beat_s, poll_s = fleet_syncer.SERVICE_BEAT_S * SCALE, fleet_syncer.POLL_S * SCALE
    call_s, grace = fleet_syncer.HEARTBEAT_CALL_S * SCALE, fs.SYNCER_STOP_GRACE_S * SCALE
    directory = DirectoryObjectStore(tmp_path / "store")
    hanging = HangingStore(directory, block_s * SCALE)
    main = HangingStore(directory, block_s * SCALE) if main_hangs else directory
    times: dict = {}

    class SlowStop(FakeSyncer):
        def stop(self):
            time.sleep(grace)  # the syncer's checkpoint takes the whole grace
            times.setdefault("stopped", time.time())  # the first stop: the fence's
            super().stop()

    syncer = SlowStop()

    def serve(args, addr, n, *, clusters, head_mode, local_syncer, record_extra, on_record, stopped, early_finish,
              **kw):
        record = island_launches.write_syncer_record(kw["objects"], addr=addr, syncer_epoch=args.rl_syncer_epoch,
                                                     learners=n, max_learners=4, run="r", extra=record_extra)
        on_record(record)
        lock = StoreLock(directory, fleet_syncer.SERVICE_LOCK, record_extra["service_generation"], "x")
        times["lease"] = lambda: float(json.loads(directory.read(lock.lease_key(lock.prefix, lock.token)))["until"])
        hanging.down = main.down = True  # from now on the store hangs
        deadline = time.monotonic() + 60
        while True:
            polled = time.monotonic()
            why = stopped()
            try:  # a record of a restart on the main path
                island_launches.read_syncer_record(kw["objects"])
            except Exception:  # noqa: BLE001 - cut off
                pass
            assert time.monotonic() - polled <= 2 * call_s + 0.2  # never held by the store
            if why:
                times["seen"] = time.time()
                return 9
            assert time.monotonic() < deadline
            time.sleep(poll_s)  # supervise's poll

    path = onsite_config(tmp_path)
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    logs: list[str] = []
    rc = fs.run_syncer(str(path), objects=main, heartbeat_store=hanging, make_syncer=lambda args, n: syncer,
                       serve=serve, parse=_prepared, node="site-s", beat_s=beat_s, poll_s=poll_s, call_s=call_s,
                       stop_grace_s=grace, workdir=str(tmp_path / "work"), log=logs.append)
    times["returned"] = time.time()
    assert rc == fs.SELF_FENCED_EXIT and syncer.events[:2] == ["start", "stop"], logs
    ttl = fs.service_lock_ttl_s(grace, beat_s, poll_s=poll_s, call_s=call_s)
    margin = min(60.0, ttl * 0.1)
    assert times["stopped"] < times["lease"]() - margin / 2, (times, logs)  # stopped before the lease ends
    assert times["returned"] - times["seen"] < 2 * grace + 2 * call_s + 1.0, times  # the main thread ends too
    assert any("fenced by itself" in m for m in logs)
    assert any("the syncer is stopped (SIGTERM" in m for m in logs)


class GateStore:
    """A store whose writes of keys containing ``match`` block until ``gate`` is set once
    ``armed``; ``writes``: (key, data) in the order they landed."""

    def __init__(self, inner, match):
        import threading

        self.inner, self.match, self.armed = inner, match, False
        self.gate, self.entered, self.writes = threading.Event(), threading.Event(), []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def write(self, key, data):
        if self.armed and self.match in key and not self.gate.is_set():
            self.entered.set()
            self.gate.wait(10)
        self.inner.write(key, data)
        self.writes.append((key, json.loads(data)))


@pytest.mark.parametrize("slow", ["heartbeat", "lease"])
@pytest.mark.parametrize("late", [False, True])
def test_the_end_waits_for_a_call_in_flight_before_its_final_heartbeat_and_release(tmp_path, slow, late):
    """M-a: a heartbeat (E) or a lease renewal (D) still in flight at the end lands before the
    final heartbeat, never over it; a renewal in flight is never followed by a release it would
    undo. Event-driven: the call is held at a gate. Not late: the gate opens while the end
    waits (call_s 5 s), then final, then release. Late: the end's waits run out first (call_s
    0.05 s), no release, and the final heartbeat comes after the call (the gate opened later)."""
    import threading

    from yeto.launch.pool.store_lock import StoreLock

    directory = DirectoryObjectStore(tmp_path / "store")
    gated = GateStore(directory, "island_syncer_service" if slow == "heartbeat" else ".lease")
    lock = StoreLock.acquire(gated, fleet_syncer.SERVICE_LOCK, holder="h", ttl_s=60)
    logs: list[str] = []
    beat_ = fs.Heartbeat(gated, "h", "kid", lock=lock, generation=lock.token, every_s=0.01,
                         call_s=0.05 if late else 5.0, log=logs.append)
    beat_.final_wait_s = 10.0  # the late final heartbeat waits for the gate, however slow the host
    beat_.set(fleet_syncer.SERVICE_RUNNING)
    gated.armed = True
    beat_._thread.start()
    assert gated.entered.wait(10)  # the beat thread's call is in flight now, held at the gate
    beat_.stop()
    if late:
        beat_.finish(fleet_syncer.SERVICE_FINISHED, None, release=True)  # its waits run out: the gate is shut
        finals = [t for t in threading.enumerate() if t.name == "syncer-service-final"]
        assert finals and all(not t.daemon for t in finals)  # the interpreter waits for the final heartbeat
        gated.gate.set()
        for thread in finals:
            thread.join(15)
    else:
        ending = threading.Thread(target=beat_.finish, args=(fleet_syncer.SERVICE_FINISHED, None),
                                  kwargs={"release": True})
        ending.start()
        gated.gate.set()  # the call lands while the end waits for it
        ending.join(15)
    status = [d for k, d in gated.writes if "island_syncer_service" in k]
    leases = [d for k, d in gated.writes if k.endswith(".lease")]
    assert status[-1]["state"] == "finished"  # never a late "running" over it
    if late:  # no release a late renewal could undo; the lease runs out
        assert all(not d["released"] for d in leases) and any("not released" in m for m in logs)
    else:
        assert leases[-1]["released"] is True and StoreLock.holder_of(directory, fleet_syncer.SERVICE_LOCK) is None


def test_the_end_waits_for_a_beat_thread_that_is_about_to_renew(tmp_path):
    """M-a: the beat thread past its wait but not yet in its store call (the lease read before
    the renewal) is joined first: its renewal lands before the release, never after it."""
    import threading
    import time

    from yeto.launch.pool.store_lock import StoreLock

    directory = DirectoryObjectStore(tmp_path / "store")
    lock = StoreLock.acquire(directory, fleet_syncer.SERVICE_LOCK, holder="h", ttl_s=60)
    leases: list = []
    real_write = directory.write

    def write(key, data):
        real_write(key, data)
        if key.endswith(".lease"):
            leases.append(json.loads(data))

    directory.write = write
    beat_ = fs.Heartbeat(directory, "h", "kid", lock=lock, generation=lock.token, every_s=0.01, call_s=5.0,
                         log=lambda m: None)
    entered, go = threading.Event(), threading.Event()
    real_left = lock.lease_left

    def lease_left():
        if threading.current_thread().name == "syncer-service-beat" and not go.is_set():
            entered.set()
            go.wait(10)
        return real_left()

    lock.lease_left = lease_left
    calls: list[str] = []
    for name in ("renew", "release"):
        def called(real=getattr(lock, name), name=name):
            calls.append(name)
            return real()
        setattr(lock, name, called)
    beat_._thread.start()
    assert entered.wait(10)
    beat_.stop()
    ending = threading.Thread(target=beat_.finish, args=(None, None), kwargs={"release": True})
    ending.start()
    ending.join(0.3)  # the end waits for the beat thread (a scheduling wait, not a measured bound)
    go.set()
    ending.join(15)
    assert calls == ["renew", "release"]  # the renewal came first, the release last
    assert leases[-1]["released"] is True


def test_a_self_fenced_service_exits_1_and_stops_its_syncer_before_its_lease_ends(tmp_path, monkeypatch):
    from yeto.launch.pool import store_lock

    objects = DirectoryObjectStore(tmp_path / "store")
    real = store_lock.StoreLock.renew
    state = {"down": False}

    def renew(self):
        if state["down"]:
            raise OSError("store down")
        return real(self)

    monkeypatch.setattr(store_lock.StoreLock, "renew", renew)
    def cut():
        state["down"] = True  # the store stops answering the renewals

    monkeypatch.setattr(fs, "service_lock_ttl_s", lambda grace, beat=None, **kw: 0.6)  # a short lease for the test
    rc, seen, syncer, objects, logs = run_service(tmp_path, monkeypatch, 0, objects=objects, until=cut, beat_s=0.05)
    assert rc == fs.SELF_FENCED_EXIT and "fenced by itself" in seen["stopped"]
    assert syncer.events[:2] == ["start", "stop"] and set(syncer.events[1:]) == {"stop"}


def test_two_runs_on_one_host_never_share_a_syncer_state(tmp_path, monkeypatch):
    from yeto.launch import local_syncer

    homes = []

    class Captured(FakeSyncer):
        def __init__(self, args, n, *, binary, log_file, env, stop_grace_s, home):
            super().__init__()
            homes.append((env["HOME"], home))

    monkeypatch.setattr(local_syncer, "LocalSyncer", Captured)
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    for run in ("r", "r2"):
        doc = onsite_doc(tmp_path)
        doc["run"] = run
        doc["fleet"]["trainer_argv"] = [*doc["fleet"]["trainer_argv"]]
        path = tmp_path / f"{run}.json"
        path.write_text(json.dumps(doc))
        workdir = tmp_path / f"var/{run}-syncer"  # each run's unit has its own working directory
        workdir.mkdir(parents=True)
        objects = DirectoryObjectStore(tmp_path / f"store-{run}")
        assert fs.run_syncer(str(path), binary="/opt/yeto-syncer", objects=objects, parse=_prepared, node="site-s",
                             serve=fake_serve(objects, {}, 0), workdir=str(workdir), log=lambda m: None) == 0
    (home_a, _), (home_b, _) = homes
    assert home_a != home_b and home_a.endswith("r-syncer/home") and home_b.endswith("r2-syncer/home")
    assert all(env_home == home for env_home, home in homes)  # the syncer's ~ and LocalSyncer's agree


def test_a_checkpoint_of_another_run_is_never_resumed(tmp_path, monkeypatch):
    ckpt = tmp_path / "work" / "home" / "yeto-output" / "yeto-state.ckpt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"an earlier run's state")
    objects = DirectoryObjectStore(tmp_path / "store")
    rc, seen, syncer, _objects, logs = run_service(tmp_path, monkeypatch, 0, objects=objects)  # no record: new store
    assert rc == 2 and syncer.events == [] and "another run's state" in logs[-1]
    island_launches.write_syncer_record(objects, addr="10.0.0.5:29400", syncer_epoch=2, learners=2, max_learners=4,
                                        run="other-run", extra={"service": "10.0.0.5", "service_node": "site-s"})
    rc, *_ , logs = run_service(tmp_path, monkeypatch, 0, objects=objects)
    assert rc == 2 and "of run 'other-run'" in logs[-1]
    island_launches.write_syncer_record(objects, addr="10.0.0.5:29400", syncer_epoch=2, learners=2, max_learners=4,
                                        run="r", extra={"service": "10.0.0.5", "service_node": "site-s"})
    rc, seen, *_ = run_service(tmp_path, monkeypatch, 0, objects=objects)
    assert rc == 0 and seen["epoch"] == 2  # this run's own checkpoint: resumed, at its term


def test_an_old_finished_status_never_ends_the_head_wait_early(tmp_path, monkeypatch):
    """serve(early_finish=True): only the syncer process's own exit 0 ends the wait for it to
    listen; a status.json that says finished (a run before, a resume not up yet) does not."""
    objects = DirectoryObjectStore(tmp_path / "store")
    monkeypatch.setattr(fleet_syncer, "_store_objects", lambda args: objects)
    polls = {"n": 0}

    def listening(args, **kw):
        polls["n"] += 1
        return polls["n"] >= 3

    monkeypatch.setattr(fleet_syncer, "syncer_listening", listening)

    class Syncer:
        exited = False

        def probe(self):
            return None

        def finished(self):  # an old status.json: finished
            return True

        def exited_ok(self):
            return self.exited

    args = types.SimpleNamespace(cluster_prefix="r", rl_syncer_epoch=4, rl_sync_max_learners=4,
                                 rl_checkpoint_store=str(tmp_path / "store"))
    clock = Clock()
    first = Syncer()
    first.exited_ok = lambda: clock.sleeps >= 8  # the syncer process itself ends later
    assert fleet_syncer.serve(args, "10.0.0.5:29400", 1, clusters=[], head_mode=True, local_syncer=first,
                              sleep=clock.sleep, early_finish=True) == 0
    assert polls["n"] == 3 and island_launches.read_syncer_record(objects) is not None  # waited, then recorded
    assert clock.sleeps == 8  # supervised on: the old finished status.json never ended it either
    done, polls["n"] = Syncer(), -100
    done.exited = True  # the resumed syncer itself ran to its end before it was seen listening
    objects2 = DirectoryObjectStore(tmp_path / "store2")
    monkeypatch.setattr(fleet_syncer, "_store_objects", lambda args: objects2)
    assert fleet_syncer.serve(args, "10.0.0.5:29400", 1, clusters=[], head_mode=True, local_syncer=done,
                              sleep=clock.sleep, early_finish=True) == 0
    assert island_launches.read_syncer_record(objects2) is None


@pytest.mark.usefixtures("no_sigterm_handler")
def test_a_checkpoint_gone_while_the_record_says_it_is_here_is_refused_until_fresh(tmp_path, monkeypatch):
    """The run store's record says this host's service ran this run's syncer, but its checkpoint is
    gone (the workdir was cleared): refused (exit 2), naming the recovery. `fresh` lets the next
    start begin the syncer at the next term; the marker goes once that term is recorded."""
    objects = DirectoryObjectStore(tmp_path / "store")
    island_launches.write_syncer_record(objects, addr="10.0.0.5:29400", syncer_epoch=4, learners=2, max_learners=4,
                                        run="r", extra={"service": "10.0.0.5", "service_node": "site-s"})
    rc, seen, syncer, _objects, logs = run_service(tmp_path, monkeypatch, 0, objects=objects)
    assert rc == 2 and rc in fs.SYNCER_PREVENT and syncer.events == [] and seen == {}
    assert "its checkpoint is not here" in logs[-1] and "yeto fleet service fresh" in logs[-1]
    path = onsite_config(tmp_path)
    monkeypatch.setattr(fs.socket, "gethostname", lambda: "site-s")
    assert fs.service_run(service_args(path, "syncer", fresh=True), objects=objects, log=logs.append) == 0
    fresh = tmp_path / "state" / "r" / "onsite-syncer.fresh"
    assert fresh.exists() and "begins the syncer at the next term (after epoch 4" in logs[-1]
    real = fs.run_syncer
    monkeypatch.setattr(fs, "run_syncer", lambda config, **kw: real(
        config, objects=objects, make_syncer=lambda args, n: FakeSyncer(False), serve=fake_serve(objects, seen, 0),
        parse=_prepared, node="site-s", workdir=str(tmp_path / "work"), log=logs.append,
        fresh_path=kw["fresh_path"]))
    assert fs.service_run(service_args(path, "syncer"), objects=objects, log=logs.append) == 0
    assert seen["epoch"] == 5 and not fresh.exists()  # a new term, the islands rejoin it; the marker used up
    assert island_launches.read_syncer_record(objects)["syncer_epoch"] == 5
    assert any("started fresh (--fresh)" in m for m in logs)


@pytest.mark.usefixtures("no_sigterm_handler")
def test_a_fresh_start_that_fails_before_its_record_is_fresh_again(tmp_path, monkeypatch):
    """M2: the --fresh marker stays until this service's record at the new term is in the run
    store: a syncer that fails to start (or dies before its record) is fresh again on the restart,
    at the same next term, and a checkpoint it wrote meanwhile does not undo that."""
    objects = DirectoryObjectStore(tmp_path / "store")
    island_launches.write_syncer_record(objects, addr="10.0.0.5:29400", syncer_epoch=4, learners=2, max_learners=4,
                                        run="r", extra={"service": "10.0.0.5", "service_node": "site-s"})
    path = onsite_config(tmp_path)
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    monkeypatch.setattr(fs.socket, "gethostname", lambda: "site-s")
    logs: list[str] = []
    assert fs.service_run(service_args(path, "syncer", fresh=True), objects=objects, log=logs.append) == 0
    fresh = tmp_path / "state" / "r" / "onsite-syncer.fresh"
    workdir = tmp_path / "work"

    class Fails(FakeSyncer):
        def start(self):
            for c in fs._checkpoints(self.args, str(workdir / "home")):  # it got as far as a checkpoint
                pathlib.Path(c).parent.mkdir(parents=True, exist_ok=True)
                pathlib.Path(c).write_text("{}")
            raise OSError("the syncer binary did not start")

    def failing(args, n):
        syncer = Fails()
        syncer.args = args
        return syncer

    def run(make):
        seen: dict = {}
        rc = fs.run_syncer(str(path), objects=objects, make_syncer=make, serve=fake_serve(objects, seen, 0),
                           parse=_prepared, node="site-s", workdir=str(workdir), log=logs.append,
                           fresh_path=fresh)
        return rc, seen

    with pytest.raises(OSError, match="did not start"):
        run(failing)
    assert fresh.exists() and island_launches.read_syncer_record(objects)["syncer_epoch"] == 4
    rc, seen = run(lambda args, n: FakeSyncer(False))  # the unit's restart
    assert rc == 0 and seen["epoch"] == 5 and not fresh.exists()
    assert island_launches.read_syncer_record(objects)["syncer_epoch"] == 5


@pytest.mark.usefixtures("no_sigterm_handler")
def test_fresh_on_a_host_that_is_not_the_record_s_node_is_refused(tmp_path, monkeypatch):
    """L2: `fresh` only where the run store's record says this run's syncer ran (its
    service_node); elsewhere, or with no record of this run, exit 2 naming the node, no marker."""
    objects = DirectoryObjectStore(tmp_path / "store")
    path = onsite_config(tmp_path)
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    fresh = tmp_path / "state" / "r" / "onsite-syncer.fresh"
    logs: list[str] = []
    monkeypatch.setattr(fs.socket, "gethostname", lambda: "site-t")
    assert fs.service_run(service_args(path, "syncer", fresh=True), objects=objects, log=logs.append) == 2
    assert "no syncer record of run 'r'" in logs[-1] and not fresh.exists()
    island_launches.write_syncer_record(objects, addr="10.0.0.5:29400", syncer_epoch=4, learners=2, max_learners=4,
                                        run="other", extra={"service": "10.0.0.5", "service_node": "site-t"})
    assert fs.service_run(service_args(path, "syncer", fresh=True), objects=objects, log=logs.append) == 2
    assert "run 'other'" in logs[-1] and not fresh.exists()
    island_launches.write_syncer_record(objects, addr="10.0.0.5:29400", syncer_epoch=4, learners=2, max_learners=4,
                                        run="r", extra={"service": "10.0.0.5", "service_node": "site-s"})
    assert fs.service_run(service_args(path, "syncer", fresh=True), objects=objects, log=logs.append) == 2
    assert "on node 'site-s', not this one ('site-t')" in logs[-1] and not fresh.exists()
    assert not fresh.parent.exists() or list(fresh.parent.iterdir()) == []
    # a marker left on a host the record does not name is never used later (removed at the start)
    fresh.parent.mkdir(parents=True, exist_ok=True)
    fresh.write_text("{}")
    seen: dict = {}
    rc = fs.run_syncer(str(path), objects=objects, make_syncer=lambda args, n: FakeSyncer(False),
                       serve=fake_serve(objects, seen, 0), parse=_prepared, node="site-t",
                       workdir=str(tmp_path / "work"), log=logs.append, fresh_path=fresh)
    assert rc == 0 and seen["epoch"] == 5 and not fresh.exists()  # another host's service: the next term anyway
    assert any("is not for this record" in m for m in logs)


def test_fresh_runs_on_the_syncer_host_as_the_unit_runs(tmp_path, monkeypatch, capsys):
    ssh = FakeSsh()
    monkeypatch.setattr(fs.subprocess, "run", ssh)
    config = onsite_config(tmp_path)
    argv = ["fresh", "site-s", "--config", str(config), "--role", "syncer", "--env-file", "/etc/yeto/r.env"]
    assert fleet_cli.main(fleet_cli_args(["fleet", "service", *argv])) == 0
    script = ssh.calls[0][1]
    assert "fleet service-run --role syncer --config" in script and script.rstrip().endswith("--fresh")
    argv[5] = "control"
    assert fleet_cli.main(fleet_cli_args(["fleet", "service", *argv])) == 2
    assert "fresh: --role syncer" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["-r", ".", ".."])
def test_a_run_name_that_is_an_option_or_a_dot_path_is_refused(tmp_path, monkeypatch, name):
    doc = onsite_doc(tmp_path)
    doc["run"] = name
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(doc))
    with pytest.raises(fs.ServiceError, match="letters, digits"):
        fs.Onsite.open(str(path), "syncer", objects=DirectoryObjectStore(tmp_path / "s"))
    assert not fs._SAFE_RUN.fullmatch("-r") and fs._SAFE_RUN.fullmatch("r-1.a_b")


@pytest.mark.parametrize("conditional", [True, False])
def test_the_watcher_rewrites_the_record_only_over_the_service_s_own(tmp_path, conditional):
    """The watcher's rewrite is the store's conditional write (replace_if): a record the service
    rewrote between the watcher's read and its write is never overwritten. A store without it
    reads again right before a plain write (the window left is that one write)."""
    objects, clock = DirectoryObjectStore(tmp_path / "store"), Clock()
    service_up(objects, clock)
    status = fleet_syncer.read_service_status(objects)
    args = watch_args(tmp_path)
    if not conditional:
        class Plain:
            def __init__(self, inner):
                self.inner = inner

            def read(self, key):
                return self.inner.read(key)

            def write(self, key, data):
                self.inner.write(key, data)

        store = Plain(objects)
    else:
        store = objects
        plain = []
        real_write = objects.write
        objects.write = lambda key, data: (plain.append(key), real_write(key, data))
    written = fleet_syncer._rerecord(store, status, args)
    if conditional:
        assert "fleet/island_syncer.json" not in plain  # the conditional write, never a plain one
    assert written is not None and island_launches.read_syncer_record(objects)["at"] == written["at"]
    # the service restarts between the watcher's read and its write: its generation-8 record stands
    real_read = store.read

    def read(key):
        data = real_read(key)
        if key == "fleet/island_syncer.json" and not reads:
            reads.append(1)
            service_up(objects, clock, generation=8)
        return data

    reads: list = []
    store.read = read
    assert fleet_syncer._rerecord(store, status, args) is None
    assert island_launches.read_syncer_record(objects)["service_generation"] == 8


def test_replace_if_writes_only_over_the_bytes_read(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "s")
    objects.write("k", b"a")
    assert objects.replace_if("k", b"b", b"a") is True and objects.read("k") == b"b"
    assert objects.replace_if("k", b"c", b"a") is False and objects.read("k") == b"b"
    assert objects.replace_if("gone", b"c", b"a") is False and objects.read("gone") is None
    assert objects.list("") == ["k"]  # the lock file is never listed


def test_the_service_s_tape_upload_uses_its_store_client_and_run_up_s_environment(tmp_path, monkeypatch):
    from yeto.launch import local_syncer

    seen = {}

    class Uploads(FakeSyncer):
        def start_store_upload(self, args, objects=None):
            seen["objects"] = objects

    objects = DirectoryObjectStore(tmp_path / "store")
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    syncer = Uploads(False)
    path = onsite_config(tmp_path)
    fs.run_syncer(str(path), objects=objects, make_syncer=lambda args, n: syncer, serve=fake_serve(objects, {}, 0),
                  parse=_prepared, node="site-s", workdir=str(tmp_path / "work"), log=lambda m: None)
    assert seen["objects"] is objects  # never a client built from os.environ
    ls = local_syncer.LocalSyncer.__new__(local_syncer.LocalSyncer)
    ls.env = {"YETO_RL_TAPE_UPLOAD_S": "0"}
    calls = []
    monkeypatch.setattr("yeto.rl.engine.tape_upload.upload_interval_s",
                        lambda spec_value=None, environ=None: calls.append(environ))
    ls.start_store_upload(types.SimpleNamespace(rl_checkpoint_store="s3://b/p"), objects)
    assert calls == [ls.env]


def etag_s3(mode: str = "applies"):
    """A FaultS3 with ETags and PutObject ``If-Match``: ``applies`` (412 on a wrong ETag),
    ``ignores`` (writes over any ETag, as some S3-compatible endpoints do) or an error code the
    endpoint answers every ``If-Match`` with."""
    import hashlib

    from store_fault_fakes import FaultS3, client_error

    def tag(data):
        return '"' + hashlib.md5(data).hexdigest() + '"'

    class ETagS3(FaultS3):
        refuse = False

        def get_object(self, Bucket, Key, Range=None, IfMatch=None):  # noqa: N803
            out = super().get_object(Bucket, Key, Range, IfMatch)
            out["ETag"] = tag(self.objects[Key][0])
            return out

        def put_object(self, Bucket, Key, Body, IfNoneMatch=None, IfMatch=None, **kw):  # noqa: N803
            if IfMatch is None:
                return super().put_object(Bucket, Key, Body, IfNoneMatch, **kw)
            self._enter("put-if", Key)
            if self.refuse:
                raise client_error("NotImplemented", "PutObject")
            if mode not in ("applies", "ignores"):
                raise client_error(mode, "PutObject")
            current = self.objects.get(Key)
            if mode == "applies" and (current is None or tag(current[0]) != IfMatch):
                raise client_error("PreconditionFailed", "PutObject")
            self.objects[Key] = (bytes(Body), self.now)

    return ETagS3()


@pytest.fixture
def replace_probe(monkeypatch):
    from yeto.cloud import s3_store

    monkeypatch.setattr(s3_store, "_REPLACE_IF_USABLE", {})  # probed afresh in each test


@pytest.mark.usefixtures("replace_probe")
def test_the_s3_store_s_replace_if_is_an_if_match_put(tmp_path):
    fake = etag_s3()
    store = fake.store()
    store.write("k", b"a")
    assert store.replace_if("k", b"b", b"a") is True and store.read("k") == b"b"
    assert store.replace_if("k", b"c", b"a") is False and store.read("k") == b"b"  # not what was read
    fake.before("put-if", "k", lambda key: store.write("k", b"other"))  # rewritten between its GET and its PUT
    assert store.replace_if("k", b"c", b"b") is False and store.read("k") == b"other"
    assert store.replace_if("gone", b"c", b"a") is False
    fake.refuse = True
    assert store.replace_if("k", b"c", b"other") is None and store.read("k") == b"other"  # the caller falls back
    assert store.replace_if_usable() is False  # an unsupported answer: the fallback from now on


@pytest.mark.usefixtures("replace_probe")
@pytest.mark.parametrize("mode", ["applies", "ignores", "501", "NotImplemented", "InvalidArgument", "InvalidRequest"])
def test_an_endpoint_that_ignores_if_match_is_found_by_a_probe_and_takes_the_fallback(tmp_path, mode):
    """M3: the first replace_if probes the endpoint once (a probe object, a PutObject with a wrong
    ETag): only a 412 makes the conditional write usable; a write that goes through or an
    unsupported answer is the fallback (None). Cached per endpoint and bucket; the probe object is
    deleted."""
    from yeto.cloud import s3_store

    fake = etag_s3(mode)
    store = fake.store()
    store.write("k", b"a")
    usable = mode == "applies"
    assert store.replace_if("k", b"b", b"a") is (True if usable else None)
    assert store.read("k") == (b"b" if usable else b"a")  # an endpoint that ignores If-Match is never written
    probes = [k for k in fake.objects if k.startswith(s3_store.REPLACE_IF_PROBE_PREFIX)]
    assert probes == []  # deleted
    assert fake.calls["put-if"] == 1 + (1 if usable else 0)  # the probe, then (usable) the write itself
    other = fake.store()  # the same endpoint and bucket: never probed again in this process
    assert other.replace_if("k", b"c", b"b" if usable else b"a") is (True if usable else None)
    assert fake.calls["put-if"] == 1 + (2 if usable else 0)
    assert s3_store._REPLACE_IF_USABLE == {(None, "bkt"): usable}
    # the watcher's rewrite then reads again right before a plain write (its documented window)
    status = {"generation": 7}
    island_launches.write_syncer_record(store, addr="10.0.0.5:29400", syncer_epoch=3, learners=2, max_learners=4,
                                        run="r", extra={"service": "10.0.0.5", "service_generation": 7})
    args = types.SimpleNamespace(cluster_prefix="r")
    assert fleet_syncer._rerecord(store, status, args) is not None
    assert fake.calls["put-if"] == 1 + (3 if usable else 0)


def test_a_store_error_during_the_probe_is_raised_and_not_cached(monkeypatch):
    from store_fault_fakes import client_error

    from yeto.cloud import s3_store

    monkeypatch.setattr(s3_store, "_REPLACE_IF_USABLE", {})
    fake = etag_s3()
    store = fake.store()
    fake.fail("put-if", s3_store.REPLACE_IF_PROBE_PREFIX, client_error("InternalError", "PutObject"))
    with pytest.raises(Exception, match="InternalError"):
        store.replace_if_usable()
    assert s3_store._REPLACE_IF_USABLE == {} and not any(k.startswith(".yeto") for k in fake.objects)
    assert store.replace_if_usable() is True


def test_the_service_s_record_write_waits_for_a_conditional_rewrite_in_a_directory_store(tmp_path):
    """M4: a directory store's replace_if holds a lock the service's record write takes too: a
    record the service writes while the watcher's rewrite is between its read and its write waits
    for it, and stands after it (never overwritten)."""
    import threading

    objects, clock = DirectoryObjectStore(tmp_path / "store"), Clock()
    service_up(objects, clock, generation=7)
    raw = objects.read(island_launches.SYNCER_RECORD)
    real_staged = objects._staged
    seen: dict = {}

    def service_restarts():
        island_launches.write_syncer_record(objects, addr="10.0.0.5:29400", syncer_epoch=3, learners=2,
                                            max_learners=4, run="r",
                                            extra={"service": "10.0.0.5", "service_generation": 8})

    def staged(path, data):  # inside replace_if, after its read: the service records generation 8 now
        if "thread" not in seen and path.name == "island_syncer.json":
            seen["thread"] = threading.Thread(target=service_restarts)
            seen["thread"].start()
            seen["thread"].join(0.3)
            seen["blocked"] = seen["thread"].is_alive()
        return real_staged(path, data)

    objects._staged = staged
    rewritten = json.loads(raw)
    rewritten["at"] += 1
    assert objects.replace_if(island_launches.SYNCER_RECORD, json.dumps(rewritten).encode(), raw) is True
    seen["thread"].join(5)
    assert seen["blocked"] is True  # it waited for the conditional write
    assert island_launches.read_syncer_record(objects)["service_generation"] == 8  # and stands after it


@pytest.mark.parametrize("state", sorted(fleet_syncer.SERVICE_FINAL))
def test_a_final_heartbeat_an_earlier_service_left_never_ends_the_wait(tmp_path, state):
    """M1: a final heartbeat (finished / failed / stopped) already in the run store when the syncer
    launch begins to wait is an earlier service's: it waits on for a newer generation, and then
    ends with that service as ever. clear-ended leaves the heartbeat alone."""
    objects, clock, logs = DirectoryObjectStore(tmp_path / "store"), Clock(), []
    service_up(objects, clock, generation=7, state=state, reason="the earlier run")
    onsite = fs.Onsite(up.load_config(str(onsite_config(tmp_path))), "syncer", objects=objects)
    assert onsite.clear_ended(log=logs.append) == 0
    assert fleet_syncer.read_service_status(objects)["state"] == state  # not removed by clear-ended

    def sleep(s):
        clock.sleep(s)
        n = clock.sleeps
        if n == 4:
            service_up(objects, clock, generation=8, state=fleet_syncer.SERVICE_STARTING)
        elif 5 <= n < 9:
            beat(objects, clock, generation=8)
        elif n >= 9:
            beat(objects, clock, generation=8, state=state, reason="this run")

    expected = {"finished": 0, "stopped": 9, "failed": 1}[state]
    assert fleet_syncer.watch_service(watch_args(tmp_path), objects=objects, sleep=sleep, clock=clock,
                                      log=logs.append) == expected
    assert clock.sleeps >= 9  # never at the earlier service's final heartbeat
    assert island_launches.read_syncer_record(objects)["service_generation"] == 8
    assert any("an earlier service's final one" in m for m in logs)


def test_a_final_heartbeat_of_a_newer_generation_ends_the_wait(tmp_path):
    objects, clock, logs = DirectoryObjectStore(tmp_path / "store"), Clock(), []
    service_up(objects, clock, generation=7, state=fleet_syncer.SERVICE_FINISHED)

    def sleep(s):
        clock.sleep(s)
        if clock.sleeps == 2:
            beat(objects, clock, generation=8, state=fleet_syncer.SERVICE_FAILED, reason="no checkpoint")

    assert fleet_syncer.watch_service(watch_args(tmp_path), objects=objects, sleep=sleep, clock=clock,
                                      log=logs.append) == 1
    assert clock.sleeps == 2 and "no checkpoint" in logs[-1]


@pytest.mark.parametrize("problem", ["credentials", "no object API"])
def test_a_syncer_service_without_a_usable_store_is_a_configuration_error(tmp_path, monkeypatch, problem):
    """M5: the run store's credentials missing (a ValueError of the store client) or a store with
    no object API here: exit 10, never restarted."""
    from yeto.launch import island_lifecycle

    def store_objects(uri, environ=None):
        if problem == "credentials":
            raise ValueError("s3 store: YETO_RL_STORE_S3_ACCESS_KEY_ID is not set")
        return None

    monkeypatch.setattr(island_lifecycle, "store_objects", store_objects)
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    logs: list[str] = []
    syncer = FakeSyncer()
    rc = fs.run_syncer(str(onsite_config(tmp_path)), make_syncer=lambda args, n: syncer, parse=_prepared,
                       node="site-s", workdir=str(tmp_path / "work"), log=logs.append)
    assert rc == fs.CONFIG_EXIT and rc in fs.SYNCER_PREVENT and syncer.events == []
    assert ("is not set" if problem == "credentials" else "no object API") in logs[-1]


@pytest.mark.usefixtures("no_sigterm_handler")
def test_a_control_unit_without_the_island_hmac_key_is_a_configuration_error(tmp_path, monkeypatch):
    """M5: run up of the control unit without YETO_ISLAND_HMAC_KEY (its preflight, or its start:
    RunUpConfigRefused): exit 10, never restarted; another refusal before ordering stays 2."""
    monkeypatch.delenv("YETO_ISLAND_HMAC_KEY")
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    path = onsite_config(tmp_path)
    logs: list[str] = []
    assert fs.run_control(str(path), log=logs.append) == fs.CONFIG_EXIT
    assert "refused its configuration" in logs[-1] and fs.CONFIG_EXIT in fs.CONTROL_PREVENT
    run = up.open_run(str(path))
    with pytest.raises(up.RunUpConfigRefused, match="YETO_ISLAND_HMAC_KEY is not set"):
        run.island_hmac_env()
    assert not (run.dir / "island_hmac.key").exists()  # never generated
    # the start's refusal (a preflight skipped): the same exit
    monkeypatch.setattr(up.Run, "preflight", lambda self, dry_run=False: True)
    monkeypatch.setattr(up.Run, "cluster_preflight", lambda self, **kw: True)
    monkeypatch.setattr(up.Run, "write_fleet_plan", lambda self: True)
    monkeypatch.setattr(up.Run, "sandbox_prepare", lambda self: True)
    monkeypatch.setattr(up.Run, "pool", lambda self: self.island_hmac_env())
    assert fs.run_control(str(path), log=logs.append) == fs.CONFIG_EXIT
    monkeypatch.setattr(up.Run, "pool", lambda self: (_ for _ in ()).throw(up.RunUpRefused("a pool refused")))
    assert fs.run_control(str(path), log=logs.append) == 2  # another refusal: restarted


@pytest.mark.usefixtures("no_sigterm_handler")
@pytest.mark.parametrize("fails", ["here", "store"])
def test_an_ended_marker_that_does_not_write_keeps_the_service_s_exit(tmp_path, monkeypatch, fails):
    """L1: the ended marker goes to the run store first, then here; a write that fails is logged
    and the service exits with its own code (never 70)."""
    objects = DirectoryObjectStore(tmp_path / "store")
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    path = onsite_config(tmp_path)
    if fails == "here":
        (tmp_path / "state").write_text("not a directory")  # this host's marker cannot be written
    else:
        def write(key, data):
            raise OSError("the store does not answer")

        monkeypatch.setattr(objects, "write", write)
    monkeypatch.setattr(fs, "run_syncer", lambda *a, **kw: 9)
    logs: list[str] = []
    assert fs.service_run(service_args(path, "syncer"), objects=objects, log=logs.append) == 9
    stored = objects.read("fleet/onsite/syncer.ended") if fails == "here" else None
    assert (stored is not None) is (fails == "here")
    assert (tmp_path / "state" / "r" / "onsite-syncer.ended").exists() is (fails == "store")
    assert any("WARNING" in m and "ended marker" in m for m in logs)


def test_the_lease_numbers_in_the_guide_are_the_code_s():
    """L3: docs/OWN_CLUSTERS.md states the syncer service's lease, fence threshold, heartbeat
    client and exits as the code computes them."""
    import re

    from yeto.cloud import s3_store

    text = (pathlib.Path(__file__).resolve().parents[1] / "docs" / "OWN_CLUSTERS.md").read_text()
    grace = fs.SYNCER_STOP_GRACE_S
    ttl = fs.service_lock_ttl_s(grace)
    margin = min(60.0, ttl * 0.1)
    fence = margin + fleet_syncer.SERVICE_BEAT_S + fleet_syncer.POLL_S + grace + 3 * fleet_syncer.HEARTBEAT_CALL_S
    assert fs.Heartbeat(None, "h", None, lock=types.SimpleNamespace(margin_s=margin), stop_grace_s=grace).fence_s == fence

    def numbers(pattern):
        found = re.findall(pattern, text)
        assert found, pattern
        return {float(n) for n in found}

    assert numbers(r"`fleet/syncer_service/lock`, (\d+) s") == {round(ttl)}
    assert numbers(r"at most (\d+) s left before") == {round(fence)}
    assert numbers(r"the lock's margin \((\d+) s\)") == {round(margin)}
    assert numbers(r"a poll of the service \((\d+) s\)") == {fleet_syncer.POLL_S}
    assert numbers(r"renewed with every heartbeat\s+\((\d+) s\)") == {fleet_syncer.SERVICE_BEAT_S}
    assert numbers(r"up to (\d+) s\) and three heartbeat calls") == {grace}
    assert numbers(r"each of its calls is cut off after (\d+) s") == {fleet_syncer.HEARTBEAT_CALL_S}
    assert numbers(r"connect (\d+) s") == {fleet_syncer.HEARTBEAT_CONNECT_TIMEOUT_S}
    assert numbers(r"read (\d+) s, \d+ requests") == {fleet_syncer.HEARTBEAT_READ_TIMEOUT_S}
    assert numbers(r"(\d+) requests per\s+call") == {fleet_syncer.HEARTBEAT_MAX_ATTEMPTS}
    assert numbers(r"stops its syncer and exits (\d+)") == {fs.SELF_FENCED_EXIT}
    assert numbers(r"fences itself \(exit (\d+)\)") == {fs.SELF_FENCED_EXIT}
    # the heartbeat client is the S3 client with those timeouts
    env = {}
    store = fs.heartbeat_objects("s3://bkt/run", {**env, s3_store.KEY_ID_ENV: "k", s3_store.SECRET_ENV: "s"})
    cfg = store.client.meta.config
    assert (cfg.connect_timeout, cfg.read_timeout) == (fleet_syncer.HEARTBEAT_CONNECT_TIMEOUT_S,
                                                       fleet_syncer.HEARTBEAT_READ_TIMEOUT_S)
    assert cfg.retries["total_max_attempts"] == fleet_syncer.HEARTBEAT_MAX_ATTEMPTS
    assert fs.heartbeat_objects("store-dir", {}) is None


@pytest.mark.parametrize("first", [None, fleet_syncer.SERVICE_STARTING])
def test_a_final_heartbeat_written_while_the_launch_waits_ends_the_wait_at_once(tmp_path, first):
    """M1: no heartbeat (or a non-final one) when the wait began: a final one of the same
    generation written meanwhile is this service's and ends the wait at once."""
    objects, clock, logs = DirectoryObjectStore(tmp_path / "store"), Clock(), []
    if first is not None:
        beat(objects, clock, generation=7, state=first)

    def sleep(s):
        clock.sleep(s)
        if clock.sleeps == 2:
            beat(objects, clock, generation=7, state=fleet_syncer.SERVICE_FAILED, reason="its checkpoint is bad")

    assert fleet_syncer.watch_service(watch_args(tmp_path), objects=objects, sleep=sleep, clock=clock,
                                      log=logs.append) == 1
    assert clock.sleeps == 2 and "its checkpoint is bad" in logs[-1]


def test_a_heartbeat_call_still_running_fails_the_next_one_at_once():
    """H1: the beat thread never has two store calls at a time: a call cut off after call_s
    that still runs makes the next one fail at once, without calling the store."""
    import threading
    import time

    beat_ = fs.Heartbeat(None, "h", None, call_s=0.05, log=lambda m: None)
    release, calls = threading.Event(), []

    def hangs():
        calls.append(1)
        release.wait(5)

    with pytest.raises(TimeoutError, match="did not answer the renewal within"):
        beat_._bounded("renewal", hangs)
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="still waits for the store"):
        beat_._bounded("heartbeat", lambda: calls.append(2))
    assert calls == [1] and time.monotonic() - started < 0.04  # not called, not waited for
    release.set()
    beat_._pending.join(5)
    assert beat_._bounded("heartbeat", lambda: calls.append(3) or "ok") == "ok" and calls == [1, 3]


@pytest.fixture
def no_store_credentials(monkeypatch, tmp_path):
    """No credential of any kind for a real S3 client: none in the environment, empty AWS
    files, no instance metadata (botocore never sends a request without a credential)."""
    import os

    for name in list(os.environ):
        if name.startswith(("AWS_", "YETO_RL_STORE_S3_")):
            monkeypatch.delenv(name)
    (tmp_path / "aws-config").write_text("")
    (tmp_path / "aws-credentials").write_text("")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "aws-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "aws-credentials"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")


@pytest.mark.usefixtures("no_sigterm_handler", "no_store_credentials")
@pytest.mark.parametrize("role", ["syncer", "control"])
@pytest.mark.parametrize("profile", [None, "not-configured-here"])
def test_a_unit_without_store_credentials_is_a_configuration_error(tmp_path, monkeypatch, role, profile):
    """M-b: the real S3 client of the run store (s3://), no credential (or a profile this host
    does not have): exit 10 at once, before the ended marker read, never an endless retry."""
    from yeto.cloud import s3_store

    s3_store._profile_config_cached.cache_clear()
    if profile:
        monkeypatch.setenv("YETO_RL_STORE_S3_PROFILE", profile)
    path = onsite_config(tmp_path)
    assert up.load_config(str(path))["store"].startswith("s3://")
    monkeypatch.setattr(fs, "run_syncer", lambda *a, **kw: pytest.fail("started without credentials"))
    monkeypatch.setattr(fs, "run_control", lambda *a, **kw: pytest.fail("started without credentials"))
    logs: list[str] = []
    assert fs.service_run(service_args(path, role), log=logs.append) == fs.CONFIG_EXIT
    assert fs.CONFIG_EXIT in (fs.SYNCER_PREVENT if role == "syncer" else fs.CONTROL_PREVENT)
    assert ("not configured on this host" if profile else "no store credentials") in logs[-1]
    assert not any("does not answer" in m for m in logs)  # not taken for a store that is down


@pytest.mark.usefixtures("no_store_credentials")
def test_a_credential_error_inside_the_marker_read_is_a_configuration_error(tmp_path):
    """M-b: the default credential chain finds nothing (botocore's NoCredentialsError at the
    first read): the marker read ends with a configuration error, not a retry."""
    onsite = fs.Onsite(up.load_config(str(onsite_config(tmp_path))), "syncer")
    def sleep(s):
        pytest.fail("a credential error was retried as a store that does not answer")

    with pytest.raises(fs.ServiceError, match="NoCredentialsError"):
        onsite.read_ended(sleep=sleep, log=lambda m: None)


@pytest.mark.usefixtures("no_store_credentials")
def test_the_syncer_service_checks_its_launch_s_store_credentials_too(tmp_path, monkeypatch):
    logs: list[str] = []
    rc = fs.run_syncer(str(onsite_config(tmp_path)), make_syncer=lambda args, n: pytest.fail("started"),
                       parse=_prepared, node="site-s", workdir=str(tmp_path / "work"), log=logs.append)
    assert rc == fs.CONFIG_EXIT and "store credentials" in logs[-1]

    def parse(argv):  # a launch that parses (its islands' credential checked elsewhere)
        with monkeypatch.context() as m:
            m.setenv("YETO_RL_STORE_S3_ACCESS_KEY_ID", "parse-only")
            m.setenv("YETO_RL_STORE_S3_SECRET_ACCESS_KEY", "parse-only")
            return _prepared(argv)

    rc = fs.run_syncer(str(onsite_config(tmp_path)), make_syncer=lambda args, n: pytest.fail("started"),
                       parse=parse, node="site-s", workdir=str(tmp_path / "work"), log=logs.append)
    assert rc == fs.CONFIG_EXIT and "no store credentials" in logs[-1]  # checked before any store call


@pytest.mark.usefixtures("no_store_credentials")
def test_the_credential_check_needs_no_request(tmp_path, monkeypatch):
    """M-b: the check reads the environment and the local AWS files only: a profile this host
    does not have is named before any client is built."""
    from yeto.cloud import s3_store

    s3_store._profile_config_cached.cache_clear()
    monkeypatch.setattr(s3_store, "_client", lambda *a, **kw: pytest.fail("a client was built"))
    none = {"HOME": str(tmp_path), "AWS_CONFIG_FILE": str(tmp_path / "aws-config"),
            "AWS_SHARED_CREDENTIALS_FILE": str(tmp_path / "aws-credentials"), "AWS_EC2_METADATA_DISABLED": "true"}
    env = {**none, "YETO_RL_STORE_S3_PROFILE": "not-here"}
    assert "AWS profile 'not-here' (YETO_RL_STORE_S3_PROFILE) is not configured" in fs.store_credential_problem(
        "s3://b/r", env)
    assert "no store credentials" in fs.store_credential_problem("r2://b/r", none)
    assert fs.store_credential_problem("s3://b/r", {"YETO_RL_STORE_S3_ACCESS_KEY_ID": "k",
                                                    "YETO_RL_STORE_S3_SECRET_ACCESS_KEY": "s"}) is None
    assert fs.store_credential_problem(str(tmp_path / "dir-store"), {}) is None
    assert "needs" in fs.store_credential_problem("oss://b/r", {})
    path = onsite_config(tmp_path)
    monkeypatch.setenv("YETO_RL_STORE_S3_PROFILE", "not-here")
    logs: list[str] = []
    monkeypatch.setattr(fs.signal, "signal", lambda *a: None)
    assert fs.service_run(service_args(path, "syncer"), log=logs.append) == fs.CONFIG_EXIT
    assert "not configured on this host" in logs[-1] and "refuses this host's credentials" not in logs[-1]


def test_launch_args_reads_the_store_environment_it_is_given(tmp_path, monkeypatch):
    from yeto import launcher
    from yeto.launch.store_env import store_client_environ

    monkeypatch.setattr(launcher, "prepare_launch_args", lambda args: None)
    monkeypatch.setattr("yeto.launch.data_mix.materialize_data_mix", lambda args: None)
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    argv, _host = fs.syncer_launch_argv(up.open_run(str(onsite_config(tmp_path))))
    args = fs.launch_args(argv, {"YETO_RL_STORE_S3_PROFILE": "p"})
    assert store_client_environ(args)["YETO_RL_STORE_S3_PROFILE"] == "p"
    assert "YETO_RL_STORE_S3_PROFILE" not in __import__("os").environ


def test_a_control_unit_restarted_after_the_syncer_ended_the_run_ends_with_it(tmp_path):
    """L1: the final heartbeat left when the wait began is this run's end when the syncer
    service's ended marker is in the run store and the syncer record is of its generation."""
    objects, clock, logs = DirectoryObjectStore(tmp_path / "store"), Clock(), []
    service_up(objects, clock, generation=7, state=fleet_syncer.SERVICE_FINISHED)
    onsite = fs.Onsite(up.load_config(str(onsite_config(tmp_path))), "syncer", objects=objects)
    assert onsite.ended_key == fleet_syncer.ONSITE_SYNCER_ENDED
    onsite.write_ended(0, "the run is finished", log=logs.append)
    assert fleet_syncer.watch_service(watch_args(tmp_path), objects=objects, sleep=clock.sleep, clock=clock,
                                      log=logs.append) == 0
    assert clock.sleeps == 0 and any("ended this run (its ended marker" in m for m in logs)
    # a record of another generation (a newer service started since): not taken
    service_up(objects, clock, generation=7, record_generation=8, state=fleet_syncer.SERVICE_FINISHED)
    assert fleet_syncer.watch_service(watch_args(tmp_path), objects=objects, sleep=clock.sleep, clock=clock,
                                      wait_s=60, log=logs.append) == 1
    assert clock.sleeps > 0


def test_the_ended_marker_is_on_this_host_before_a_store_that_hangs_is_cut_off(tmp_path, monkeypatch):
    """The ended marker is written on this host first, then in the run store with the
    heartbeat's cutoff: a store that hangs neither holds the end up nor loses the marker."""
    import threading
    import time as _time

    monkeypatch.setattr(fleet_syncer, "HEARTBEAT_CALL_S", 0.3)
    release, seen = threading.Event(), []

    class Hangs(DirectoryObjectStore):
        def write(self, key, data):
            seen.append(onsite.ended_path.exists())
            release.wait(5)
            return super().write(key, data)

    objects, logs = Hangs(tmp_path / "store"), []
    onsite = fs.Onsite(up.load_config(str(onsite_config(tmp_path))), "syncer", objects=objects)
    begun = _time.monotonic()
    try:
        onsite.write_ended(2, "the lock is lost", log=logs.append)
        took = _time.monotonic() - begun
    finally:
        release.set()
    assert seen == [True] and took < 2
    assert json.loads(onsite.ended_path.read_text())["exit_code"] == 2
    assert any("the ended marker is not in the run store" in m for m in logs)


@pytest.mark.usefixtures("replace_probe")
def test_the_probe_object_is_under_the_store_s_prefix(tmp_path):
    """M2: the probe object is ``<prefix>/.yeto/replace-if-probe/<uuid>``, where a key limited
    to the store's prefix may write."""
    from yeto.cloud import s3_store

    fake = etag_s3()
    store = s3_store.S3ObjectStore("s3://bkt/runs/r", client=fake)
    keys: list[str] = []
    fake.before("put-if", lambda key: True, keys.append, times=5)
    store.write("k", b"a")
    assert store.replace_if("k", b"b", b"a") is True
    assert keys[0].startswith("runs/r/" + s3_store.REPLACE_IF_PROBE_PREFIX + "/")
    assert keys[1] == "runs/r/k" and not any(".yeto" in k for k in fake.objects)


@pytest.mark.usefixtures("replace_probe")
@pytest.mark.parametrize("code, at", [("AccessDenied", "put"), ("Forbidden", "put"), ("AccessDenied", "put-if")])
def test_a_probe_the_key_may_not_write_takes_the_fallback(tmp_path, capsys, code, at):
    """M2: a 403 on the probe (a key limited to another prefix) is "no conditional write": the
    fallback, with a warning, never an error that keeps the syncer launch waiting."""
    from store_fault_fakes import client_error

    from yeto.cloud import s3_store

    fake = etag_s3()
    store = s3_store.S3ObjectStore("s3://bkt/runs/r", client=fake)
    store.write("k", b"a")
    fake.fail(at, s3_store.REPLACE_IF_PROBE_PREFIX, client_error(code, "PutObject"))
    assert store.replace_if("k", b"b", b"a") is None and store.read("k") == b"a"
    assert store.replace_if_usable() is False and not any(".yeto" in k for k in fake.objects)
    if at == "put":
        assert "may not write" in capsys.readouterr().err
    status = {"generation": 7}  # the watcher's rewrite: re-read, then a plain write
    island_launches.write_syncer_record(store, addr="10.0.0.5:29400", syncer_epoch=3, learners=2, max_learners=4,
                                        run="r", extra={"service": "10.0.0.5", "service_generation": 7})
    assert fleet_syncer._rerecord(store, status, types.SimpleNamespace(cluster_prefix="r")) is not None


def test_the_heartbeat_client_s_retry_mode_is_always_standard(monkeypatch):
    """L3: AWS_RETRY_MODE / YETO_RL_STORE_RETRY_MODE never change the heartbeat client's backoff."""
    from yeto.cloud import s3_store

    env = {s3_store.KEY_ID_ENV: "k", s3_store.SECRET_ENV: "s", "AWS_RETRY_MODE": "adaptive",
           s3_store.RETRY_MODE_ENV: "legacy"}
    cfg = fs.heartbeat_objects("s3://bkt/run", env).client.meta.config
    assert cfg.retries == {"mode": "standard", "total_max_attempts": fleet_syncer.HEARTBEAT_MAX_ATTEMPTS}
    assert s3_store.head_objects("s3://bkt/run", environ=env).client.meta.config.retries["mode"] == "legacy"


def test_the_service_s_records_go_to_its_store_client_from_run_up_s_environment(tmp_path, monkeypatch):
    """The syncer record of the service (the real serve) is written with the run store client
    built from run up's environment: credentials only in the run config's env, none in this
    process's, and no client from os.environ."""
    import os

    from store_fault_fakes import FaultS3

    from yeto.cloud import s3_store

    for name in list(os.environ):
        if name.startswith(("AWS_", "YETO_RL_STORE_S3_")):
            monkeypatch.delenv(name)
    doc = onsite_doc(tmp_path)
    doc["env"].update({"YETO_RL_STORE_S3_ACCESS_KEY_ID": "run-config-key-id",
                       "YETO_RL_STORE_S3_SECRET_ACCESS_KEY": "run-config-secret"})
    path = tmp_path / "creds.json"
    path.write_text(json.dumps(doc))
    fake, environs = FaultS3(), []

    def client(environ, *, require_credentials):
        environs.append(dict(environ))
        return fake

    monkeypatch.setattr(s3_store, "_client", client)
    monkeypatch.setattr(fleet_syncer, "_store_objects", lambda args: pytest.fail("a client from os.environ"))
    monkeypatch.setattr(fleet_syncer, "syncer_listening", lambda args, **kw: True)
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")

    class Done(FakeSyncer):
        def exited_ok(self):
            return "start" in self.events

    def parse(argv):  # _prepared, for the environment run_syncer's own launch_args gives
        from yeto.cli import parse_args
        from yeto.launcher import _prepare_rl_args

        args = parse_args(argv)
        from yeto.launch.store_env import bind_environ

        bind_environ(args, up.open_run(str(path)).env)
        args.model_revision, args.data_revision = "a" * 40, "b" * 40
        args.source_sha256, args.reward_sha256 = "c" * 64, "d" * 64
        _prepare_rl_args(args)
        return args

    seen = {}

    def launch_args(argv, environ=None):
        seen["environ"] = environ
        return parse(argv)

    monkeypatch.setattr(fs, "launch_args", launch_args)
    logs: list[str] = []
    rc = fs.run_syncer(str(path), make_syncer=lambda args, n: Done(), node="site-s",
                       workdir=str(tmp_path / "work"), log=logs.append)
    assert seen["environ"]["YETO_RL_STORE_S3_ACCESS_KEY_ID"] == "run-config-key-id"  # run up's environment
    assert rc == 0, logs
    store = s3_store.S3ObjectStore("s3://b/runs/r", client=fake)
    record = island_launches.read_syncer_record(store)
    assert record is not None and record["service_node"] == "site-s"
    assert environs and all(e.get("YETO_RL_STORE_S3_ACCESS_KEY_ID") == "run-config-key-id" for e in environs)
    assert "YETO_RL_STORE_S3_ACCESS_KEY_ID" not in os.environ


def test_a_state_change_waits_for_the_beat_thread_s_call_in_flight(tmp_path):
    """`running` (on_record) is never dropped because the beat thread's call is in flight: it is
    written once that call ends (within call_s)."""
    import threading

    directory = DirectoryObjectStore(tmp_path / "store")
    gated = GateStore(directory, "island_syncer_service")
    beat_ = fs.Heartbeat(gated, "h", "kid", generation=1, call_s=1.0, log=lambda m: None)
    gated.armed = True
    in_flight = threading.Thread(target=beat_.beat)  # the beat thread's heartbeat, held at the gate
    in_flight.start()
    threading.Timer(0.2, gated.gate.set).start()
    import time

    time.sleep(0.05)
    beat_.set(fleet_syncer.SERVICE_RUNNING)
    in_flight.join(5)
    assert fleet_syncer.read_service_status(directory)["state"] == "running"
    assert [d["state"] for k, d in gated.writes] == ["starting", "running"]


@pytest.mark.parametrize("cause", ["taken", "fenced"])
def test_the_beat_thread_stops_the_syncer_itself_when_the_lock_is_lost(tmp_path, monkeypatch, cause):
    """H1(a): another service took the lock (LockLost) or the service fenced itself: the syncer
    is stopped in the beat thread, before the main thread ever looks, and never started again."""
    import threading
    import time

    from yeto.launch.pool import store_lock
    from yeto.launch.pool.store_lock import StoreLock

    objects = DirectoryObjectStore(tmp_path / "store")
    stops, restarts = [], []

    class Watched(FakeSyncer):
        def stop(self):
            stops.append(threading.current_thread().name)
            super().stop()

        def restart(self):
            restarts.append(1)

    syncer = Watched()
    if cause == "fenced":
        real = store_lock.StoreLock.renew
        down = {"on": False}

        def renew(self):
            if down["on"]:
                raise OSError("store down")
            return real(self)

        monkeypatch.setattr(store_lock.StoreLock, "renew", renew)
        monkeypatch.setattr(fs, "service_lock_ttl_s", lambda grace, beat=None, **kw: 0.6)

    def serve(args, addr, n, *, clusters, head_mode, local_syncer, record_extra, on_record, stopped, early_finish,
              **kw):
        on_record(island_launches.write_syncer_record(objects, addr=addr, syncer_epoch=args.rl_syncer_epoch,
                                                      learners=n, max_learners=4, run="r", extra=record_extra))
        if cause == "taken":
            StoreLock.acquire(objects, fleet_syncer.SERVICE_LOCK, holder="site-s/yeto-r-syncer.service", ttl_s=120,
                              reclaim=True)
        else:
            down["on"] = True
        deadline = time.monotonic() + 20
        while not stops:  # the main thread never asks stopped(): the beat thread acts alone
            assert time.monotonic() < deadline
            time.sleep(0.02)
        with pytest.raises(RuntimeError, match="fenced"):
            local_syncer.restart()  # serve's restart after the fence: refused
        return 1

    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    rc = fs.run_syncer(str(onsite_config(tmp_path)), objects=objects, make_syncer=lambda args, n: syncer,
                       serve=serve, parse=_prepared, node="site-s", beat_s=0.05, workdir=str(tmp_path / "work"),
                       log=lambda m: None)
    assert rc == (fs.LOCK_LOST_EXIT if cause == "taken" else fs.SELF_FENCED_EXIT)
    assert stops[0] == "syncer-service-beat" and restarts == []


@pytest.mark.parametrize("after_read", [1, 2])
def test_a_newer_record_landing_after_any_read_is_never_overwritten(tmp_path, after_read):
    """H1: another service's newer record (generation 2) lands right after the conditional
    write's first or second read of the record. It is never overwritten: the replace holds
    only while the record is still the one read."""
    objects = DirectoryObjectStore(tmp_path / "store")
    key = fs._syncer_record_key()
    island_launches.write_syncer_record(objects, addr="a:1", syncer_epoch=1, learners=1, max_learners=4, run="r",
                                        extra={"service_generation": 1})
    newer = json.dumps({"addr": "b:2", "service_generation": 2}).encode()
    reads = {"n": 0, "landed": False}
    real_read = objects.read

    def read(name):
        raw = real_read(name)
        if name == key:
            reads["n"] += 1
            if reads["n"] == after_read:
                objects.write(key, newer)
                reads["landed"] = True
        return raw

    objects.read = read
    store = fs.BoundedStore(objects, fs.CallChannel(5.0), token=1, lost=lambda: None)
    mine = json.dumps({"addr": "a:1", "service_generation": 1, "again": True}).encode()
    try:
        store.write(key, mine)
    except fs.RecordRefused:
        pass
    if reads["landed"]:
        assert real_read(key) == newer
    else:
        assert real_read(key) == mine


class NoExclusiveStore:
    """A store with no ``write_exclusive`` (as S3): the syncer record comes by a plain ``write``."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def __getattr__(self, name):
        if name == "write_exclusive":
            raise AttributeError(name)
        return getattr(self._inner, name)


@pytest.mark.parametrize("plain", [False, True], ids=["exclusive", "plain-write"])
@pytest.mark.parametrize("seen_lost", [False, True])
def test_a_record_after_another_service_took_the_lock_never_overwrites_its_record(tmp_path, monkeypatch, seen_lost,
                                                                                    plain):
    """H1: the main thread restarts the syncer while another host's service takes the lock
    (token 2) and records itself. The record the restart then writes is refused: by the fence
    once the beat thread saw the lock lost, else by the conditional write (the store's record is
    a newer generation). The other service's record stands."""
    import threading
    import time

    from yeto.launch.pool.store_lock import StoreLock

    objects = DirectoryObjectStore(tmp_path / "store")
    seen: dict = {}

    class Restarts(FakeSyncer):
        def terminate(self):
            self.events.append("terminate")

        def start(self):
            super().start()
            if self.events.count("start") == 2:  # the restart: another host takes over meanwhile
                # its lease ran out, as the other host sees it: another claim (a reclaim of the same
                # holder name here) takes token 2
                assert StoreLock.acquire(objects, fleet_syncer.SERVICE_LOCK, holder="site-s/yeto-r-syncer.service",
                                         ttl_s=120, reclaim=True).token == 2
                island_launches.write_syncer_record(objects, addr="10.0.0.9:29400", syncer_epoch=5, learners=2,
                                                    max_learners=4, run="r",
                                                    extra={"service": "10.0.0.9", "service_generation": 2})
                if seen_lost:
                    deadline = time.monotonic() + 10
                    while not seen.get("fenced") and time.monotonic() < deadline:
                        time.sleep(0.01)

    syncer = Restarts()
    real_terminate = syncer.terminate

    def terminate():
        if threading.current_thread().name == "syncer-service-beat":
            seen["fenced"] = True
        real_terminate()

    syncer.terminate = terminate

    def serve(args, addr, n, *, clusters, head_mode, local_syncer, record_extra, on_record, stopped, early_finish,
              **kw):
        on_record(fleet_syncer.record(args, addr, n, objects=kw["objects"], extra=record_extra))
        try:
            local_syncer.restart()
        except RuntimeError as error:
            seen["restart"] = str(error)
        with pytest.raises(fs.RecordRefused) as refused:
            fleet_syncer.record(args, addr, n, objects=kw["objects"], extra=record_extra)
        seen["refused"] = str(refused.value)
        return 9

    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    fs.run_syncer(str(onsite_config(tmp_path)), objects=NoExclusiveStore(objects) if plain else objects,
                  make_syncer=lambda args, n: syncer, serve=serve, parse=_prepared, node="site-s",
                  beat_s=0.05 if seen_lost else 30.0,
                  workdir=str(tmp_path / "work"), log=lambda m: None)
    record = island_launches.read_syncer_record(objects)
    assert record["service_generation"] == 2 and record["addr"] == "10.0.0.9:29400"  # never written back
    assert ("is not written: the syncer service lock is no longer this service's" if seen_lost
            else "newer than this service's") in seen["refused"]
    assert seen_lost is ("fenced" in seen.get("restart", ""))  # a fence seen during the restart stops its syncer


@pytest.mark.parametrize("during", ["start", "terminate"])
def test_a_fence_during_a_restart_never_waits_for_it_and_no_syncer_outlives_it(tmp_path, monkeypatch, during):
    """H2: the fence comes while the main thread is inside the restart (its start, or its
    terminate, blocks). The beat thread stops the syncer at once, without waiting for the
    restart; a process the restart then starts is stopped as soon as the start returns, and
    after a terminate the fence came in, none is started. In
    the scaled budget the last syncer is down more than half the lock's margin before the lease
    ends."""
    import threading
    import time

    from yeto.launch.pool.store_lock import StoreLock

    monkeypatch.setattr(fleet_syncer, "SERVICE_LOCK_TTL_S", 0.0)
    beat_s, poll_s = fleet_syncer.SERVICE_BEAT_S * SCALE, fleet_syncer.POLL_S * SCALE
    call_s, grace = fleet_syncer.HEARTBEAT_CALL_S * SCALE, fs.SYNCER_STOP_GRACE_S * SCALE
    directory = DirectoryObjectStore(tmp_path / "store")
    hanging = HangingStore(directory, 181 * SCALE)
    times: dict = {}
    fenced = threading.Event()

    class Proc(FakeSyncer):
        running = False

        def start(self):
            super().start()
            if during == "start" and self.events.count("start") == 2:  # the restart's start: held until the fence
                fenced.wait(30)
                times["started_after_fence"] = time.time()
            self.running = True

        def terminate(self):
            if threading.current_thread().name == "syncer-service-beat":
                times.setdefault("fence_terminate", time.time())
                fenced.set()
            elif during == "terminate" and "fence_terminate" not in times:  # the restart's: the fence comes meanwhile
                fenced.wait(30)
                times["terminated_after_fence"] = time.time()
            if self.running:
                time.sleep(grace)  # the stop takes the whole grace
                self.running = False
                times["down"] = time.time()
            self.events.append("terminate")

        def stop(self):
            self.terminate()

    syncer = Proc()

    def serve(args, addr, n, *, clusters, head_mode, local_syncer, record_extra, on_record, stopped, early_finish,
              **kw):
        on_record(island_launches.write_syncer_record(kw["objects"], addr=addr, syncer_epoch=args.rl_syncer_epoch,
                                                      learners=n, max_learners=4, run="r", extra=record_extra))
        lock = StoreLock(directory, fleet_syncer.SERVICE_LOCK, record_extra["service_generation"], "x")
        times["lease"] = lambda: float(json.loads(directory.read(lock.lease_key(lock.prefix, lock.token)))["until"])
        hanging.down = True
        # the syncer died (the restart's start is held), or it hangs (the restart's terminate is held)
        syncer.running = during == "terminate"
        with pytest.raises(RuntimeError, match="fenced"):
            local_syncer.restart()
        times["restart_returned"] = time.time()
        while not stopped():
            time.sleep(poll_s)
        return 9

    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    rc = fs.run_syncer(str(onsite_config(tmp_path)), objects=directory, heartbeat_store=hanging,
                       make_syncer=lambda args, n: syncer, serve=serve, parse=_prepared, node="site-s", beat_s=beat_s,
                       poll_s=poll_s, call_s=call_s, stop_grace_s=grace, workdir=str(tmp_path / "work"),
                       log=lambda m: None)
    ttl = fs.service_lock_ttl_s(grace, beat_s, poll_s=poll_s, call_s=call_s)
    margin = min(60.0, ttl * 0.1)
    assert rc == fs.SELF_FENCED_EXIT
    if during == "start":
        assert times["fence_terminate"] <= times["started_after_fence"]  # the fence never waited for the restart
        assert syncer.events.count("start") == 2  # none started after that
    else:
        assert times["fence_terminate"] <= times["terminated_after_fence"]
        assert syncer.events.count("start") == 1  # the restart's start never came after the fence
    assert syncer.running is False
    assert times["down"] < times["lease"]() - margin / 2, times


@pytest.mark.parametrize("failure", ["raises", "log breaks", "silent"])
def test_a_heartbeat_thread_that_dies_still_ends_the_syncer(tmp_path, monkeypatch, failure):
    """M1: the beat thread raises (or its log line breaks, a closed pipe), or ends without a
    word: the syncer is stopped all the same, by the beat thread's own guard, or by the main
    thread once the lease left is within the lock's margin."""
    import time

    monkeypatch.setattr(fs, "service_lock_ttl_s", lambda grace, beat=None, **kw: 0.8)
    objects = DirectoryObjectStore(tmp_path / "store")
    if failure == "raises":
        monkeypatch.setattr(fs.Heartbeat, "renew", lambda self: (_ for _ in ()).throw(BrokenPipeError("pipe")))
    elif failure == "log breaks":
        monkeypatch.setattr(fs.Heartbeat, "renew",
                            lambda self: self.log("[syncer service] WARNING: lock not renewed (store down)"))
    else:
        monkeypatch.setattr(fs.Heartbeat, "_loop", lambda self: None)
    logs: list[str] = []

    def log(message):
        if failure == "log breaks" and "WARNING" in message:
            raise BrokenPipeError("the journal's pipe is closed")
        logs.append(message)

    syncer = FakeSyncer()
    stopped_at: dict = {}

    def serve(args, addr, n, *, clusters, head_mode, local_syncer, record_extra, on_record, stopped, early_finish,
              **kw):
        on_record(island_launches.write_syncer_record(objects, addr=addr, syncer_epoch=args.rl_syncer_epoch,
                                                      learners=n, max_learners=4, run="r", extra=record_extra))
        deadline = time.monotonic() + 10
        while not stopped():
            assert time.monotonic() < deadline
            time.sleep(0.02)
        stopped_at["main"] = "stop" in syncer.events  # stopped before the main thread returned
        return 9

    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    rc = fs.run_syncer(str(onsite_config(tmp_path)), objects=objects, make_syncer=lambda args, n: syncer, serve=serve,
                       parse=_prepared, node="site-s", beat_s=0.05, workdir=str(tmp_path / "work"), log=log)
    assert rc == fs.SELF_FENCED_EXIT and stopped_at["main"]
    assert any(("heartbeat thread failed" if failure != "silent" else "fenced by the main thread") in m
               for m in logs + [str(rc)]) or failure == "log breaks"


def test_a_lost_lock_stops_the_syncer_before_any_log_line(tmp_path, monkeypatch):
    """M1: once another service holds the lock, the beat thread stops the syncer before it
    writes a word: a log that hangs (a full journal pipe) never holds the fence up."""
    import threading
    import time

    from yeto.launch.pool.store_lock import StoreLock

    objects = DirectoryObjectStore(tmp_path / "store")
    release = threading.Event()

    def log(message):
        if threading.current_thread().name == "syncer-service-beat":
            release.wait(5)

    syncer = FakeSyncer()
    stopped_at: dict = {}

    def serve(args, addr, n, *, clusters, head_mode, local_syncer, record_extra, on_record, stopped, early_finish,
              **kw):
        on_record(island_launches.write_syncer_record(objects, addr=addr, syncer_epoch=args.rl_syncer_epoch,
                                                      learners=n, max_learners=4, run="r", extra=record_extra))
        assert StoreLock.acquire(objects, fleet_syncer.SERVICE_LOCK, holder="site-s/yeto-r-syncer.service",
                                 ttl_s=120, reclaim=True).token == 2
        deadline = time.monotonic() + 10
        while not stopped():
            assert time.monotonic() < deadline
            time.sleep(0.02)
        stopped_at["main"] = "stop" in syncer.events
        return 9

    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    try:
        rc = fs.run_syncer(str(onsite_config(tmp_path)), objects=objects, make_syncer=lambda args, n: syncer,
                           serve=serve, parse=_prepared, node="site-s", beat_s=0.05, workdir=str(tmp_path / "work"),
                           log=log)
    finally:
        release.set()
    assert rc == fs.LOCK_LOST_EXIT and stopped_at["main"]


def test_the_fence_stops_the_syncer_s_process_group_not_the_whole_syncer(tmp_path, monkeypatch):
    """The fence calls the syncer's terminate (its process group; the main thread's stop ends
    the uploads afterwards), and LocalSyncer.terminate really ends the process."""
    import subprocess
    import threading

    from yeto.launch.local_syncer import LocalSyncer

    calls: list = []

    class Both(FakeSyncer):
        def terminate(self):
            calls.append(("terminate", threading.current_thread().name))

        def stop(self):
            calls.append(("stop", threading.current_thread().name))

    monkeypatch.setattr(fs, "service_lock_ttl_s", lambda grace, beat=None, **kw: 0.6)
    objects = DirectoryObjectStore(tmp_path / "store")
    from yeto.launch.pool import store_lock

    monkeypatch.setattr(store_lock.StoreLock, "renew", lambda self: (_ for _ in ()).throw(OSError("store down"))
                        if getattr(self, "_down", False) else None)

    def serve(args, addr, n, *, clusters, head_mode, local_syncer, record_extra, on_record, stopped, early_finish,
              **kw):
        import time

        deadline = time.monotonic() + 10
        while not stopped():
            assert time.monotonic() < deadline
            time.sleep(0.02)
        return 9

    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    real_acquire = store_lock.StoreLock.acquire

    def acquire(*a, **kw):
        lock = real_acquire(*a, **kw)
        lock.renewed_at = lock.clock()
        lock._down = True
        return lock

    monkeypatch.setattr(store_lock.StoreLock, "acquire", acquire)
    rc = fs.run_syncer(str(onsite_config(tmp_path)), objects=objects, make_syncer=lambda args, n: Both(),
                       serve=serve, parse=_prepared, node="site-s", beat_s=0.05, workdir=str(tmp_path / "work"),
                       log=lambda m: None)
    assert rc == fs.SELF_FENCED_EXIT
    assert calls[0] == ("terminate", "syncer-service-beat") and ("stop", "syncer-service-beat") not in calls
    proc = subprocess.Popen(["sleep", "30"], start_new_session=True)
    real = LocalSyncer.__new__(LocalSyncer)
    real.proc, real._reaped, real.stop_grace_s = proc, False, 1.0
    real.terminate()
    assert proc.poll() is not None


class FakeAws:
    """A local stand-in, on one port in 8880-8899, for every endpoint an S3 client's default
    credential chain and its calls may reach: S3 (no object: 404; a PUT: 200), the container
    credential endpoint, the instance metadata service (IMDSv2), STS (web identity). The access
    key id each S3 request was signed with is kept (``signed``)."""

    def __init__(self, *, imds_role: bool = True):
        import http.server
        import socket
        import threading

        fake = self
        self.signed: list[str] = []
        self.imds_role = imds_role

        def creds(key):
            return {"AccessKeyId": key, "SecretAccessKey": "test-secret", "Token": "test-token",
                    "Expiration": "2099-01-01T00:00:00Z", "Code": "Success", "Type": "AWS-HMAC"}

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                return None

            def _send(self, code, body=b"", kind="application/json", headers=None):
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def do_PUT(self):  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                if self.path == "/latest/api/token":
                    return self._send(200, b"imds-token", "text/plain")
                fake.signed.append(self.headers.get("Authorization", ""))
                return self._send(200, b"", headers={"ETag": '"0"'})

            def do_GET(self):  # noqa: N802
                if self.path == "/creds":
                    return self._send(200, json.dumps(creds("test-container-key")).encode())
                if self.path.startswith("/latest/meta-data/iam/security-credentials/"):
                    if not fake.imds_role:
                        return self._send(404, b"", "text/plain")
                    if self.path.endswith("/"):
                        return self._send(200, b"test-role", "text/plain")
                    return self._send(200, json.dumps(creds("test-imds-key")).encode())
                fake.signed.append(self.headers.get("Authorization", ""))
                return self._send(404, b"<Error><Code>NoSuchKey</Code><Message>none</Message></Error>",
                                  "application/xml")

            def do_POST(self):  # noqa: N802 - STS AssumeRoleWithWebIdentity
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                body = ("<AssumeRoleWithWebIdentityResponse><AssumeRoleWithWebIdentityResult><Credentials>"
                        "<AccessKeyId>test-webid-key</AccessKeyId><SecretAccessKey>test-secret</SecretAccessKey>"
                        "<SessionToken>test-token</SessionToken><Expiration>2099-01-01T00:00:00Z</Expiration>"
                        "</Credentials></AssumeRoleWithWebIdentityResult></AssumeRoleWithWebIdentityResponse>")
                return self._send(200, body.encode(), "text/xml")

        for port in range(8880, 8900):
            try:
                self.server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
                break
            except OSError:
                continue
        else:
            raise RuntimeError("no free port in 8880-8899")
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()
        del socket

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def keys(self) -> set[str]:
        import re

        return {m.group(1) for a in self.signed for m in [re.search(r"Credential=([^/]+)/", a)] if m}


@pytest.fixture
def fake_aws(monkeypatch, tmp_path):
    """No credential source of this host reaches the test: the environment's AWS and store
    variables gone, empty AWS files, every endpoint (S3, the instance metadata) the local fake."""
    import os

    from yeto.cloud import s3_store

    for name in list(os.environ):
        if name.startswith(("AWS_", "YETO_RL_STORE_S3_")):
            monkeypatch.delenv(name)
    (tmp_path / "aws-config").write_text("")
    (tmp_path / "aws-credentials").write_text("")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "aws-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "aws-credentials"))
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    fake = FakeAws()
    monkeypatch.setenv("YETO_RL_STORE_S3_ENDPOINT_URL", fake.url)
    monkeypatch.setenv("AWS_EC2_METADATA_SERVICE_ENDPOINT", fake.url + "/")
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    s3_store._profile_config_cached.cache_clear()
    yield fake
    fake.close()


@pytest.mark.usefixtures("no_sigterm_handler")
@pytest.mark.parametrize("source", ["aws env", "container", "credential_process", "web identity", "imds",
                                    "r2 with aws env"])
def test_a_default_chain_credential_is_never_refused(tmp_path, monkeypatch, fake_aws, source):
    """H3: credentials the S3 client's default chain finds (no store key pair) pass the check
    and sign the unit's reads: AWS_* variables, a container endpoint, credential_process, a web
    identity (STS), the instance metadata, an R2 endpoint with AWS_* keys. Each against the
    local fake, never a cloud."""
    import os
    import stat

    expected = {"aws env": "test-env-key", "container": "test-container-key",
                "credential_process": "test-process-key", "web identity": "test-webid-key",
                "imds": "test-imds-key", "r2 with aws env": "test-env-key"}[source]
    if source in ("aws env", "r2 with aws env"):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-env-key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret")
    elif source == "container":
        monkeypatch.setenv("AWS_CONTAINER_CREDENTIALS_FULL_URI", fake_aws.url + "/creds")
    elif source == "credential_process":
        script = tmp_path / "creds.sh"
        script.write_text("#!/bin/sh\nprintf '%s' '{\"Version\": 1, \"AccessKeyId\": \"test-process-key\", "
                          "\"SecretAccessKey\": \"test-secret\"}'\n")
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
        (tmp_path / "aws-config").write_text(f"[default]\ncredential_process = {script}\n")
    elif source == "web identity":
        (tmp_path / "token").write_text("test-web-identity-token")
        monkeypatch.setenv("AWS_WEB_IDENTITY_TOKEN_FILE", str(tmp_path / "token"))
        monkeypatch.setenv("AWS_ROLE_ARN", "arn:aws:iam::000000000000:role/test")
        monkeypatch.setenv("AWS_ENDPOINT_URL_STS", fake_aws.url)
    if source != "imds":  # the source alone: no instance metadata to fall back on
        monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    if source == "r2 with aws env":
        r2 = "https://0123456789abcdef.r2.cloudflarestorage.com"
        assert fs.store_credential_problem("s3://b/runs/r", {**os.environ, "YETO_RL_STORE_S3_ENDPOINT_URL": r2}) is None
    assert fs.store_credential_problem("s3://b/runs/r", dict(os.environ)) is None
    monkeypatch.setattr(fs, "run_syncer", lambda *a, **kw: 0)
    logs: list[str] = []
    assert fs.service_run(service_args(onsite_config(tmp_path), "syncer"), log=logs.append) == 0, logs
    assert expected in fake_aws.keys(), (fake_aws.signed, logs)  # the read (and the marker) signed with it


@pytest.mark.usefixtures("no_sigterm_handler")
@pytest.mark.parametrize("case", ["named profile missing", "no source at all"])
def test_only_a_missing_named_profile_or_no_source_at_all_is_refused(tmp_path, monkeypatch, fake_aws, case):
    """H3: the two refusals left (exit 10): the store's named profile is not on this host, or
    no credential source exists at all (the instance metadata off too)."""
    if case == "named profile missing":
        monkeypatch.setenv("YETO_RL_STORE_S3_PROFILE", "not-on-this-host")
    else:
        monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    logs: list[str] = []
    assert fs.service_run(service_args(onsite_config(tmp_path), "syncer"), log=logs.append) == fs.CONFIG_EXIT
    assert ("'not-on-this-host'" if case == "named profile missing" else "no AWS credential source") in logs[-1]
    assert fake_aws.signed == []  # decided here, no request


def test_no_credential_yet_from_a_dynamic_source_is_asked_again(tmp_path, monkeypatch, fake_aws):
    """H3: the instance metadata answers without a role yet (NoCredentialsError at the read):
    a source that may still come up, so the read is asked again as for a store that does not
    answer, never exit 10."""
    fake_aws.imds_role = False
    onsite = fs.Onsite(up.load_config(str(onsite_config(tmp_path))), "syncer")
    tries: list = []

    class Again(Exception):
        pass

    def sleep(s):
        tries.append(s)
        if len(tries) == 2:
            fake_aws.imds_role = True  # the role arrives: the next read is signed with it
        if len(tries) > 3:
            raise Again()

    logs: list[str] = []
    assert onsite.read_ended(sleep=sleep, log=logs.append) is None
    assert len(tries) >= 1 and "does not answer" in logs[0] and "test-imds-key" in fake_aws.keys()


def test_the_store_environment_given_to_a_launch_is_never_in_its_args(tmp_path, monkeypatch):
    """L: run up's environment (credentials) bound to the service's launch args reaches the
    store clients only: never an attribute of the args, so not in the run record a launch
    writes (cli: runs.create_run), the head's replayed args (_serializable_args) or the W&B
    config (vars(args))."""
    import gc
    import os

    from yeto import cli, launcher, runs
    from yeto.launch import store_env
    from yeto.launch.store_env import store_client_environ

    secret = "run-up-secret-value-for-the-test"
    monkeypatch.setattr(launcher, "prepare_launch_args", lambda args: None)
    monkeypatch.setattr("yeto.launch.data_mix.materialize_data_mix", lambda args: None)
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    argv, _host = fs.syncer_launch_argv(up.open_run(str(onsite_config(tmp_path))))
    args = fs.launch_args(argv, {"YETO_RL_STORE_S3_SECRET_ACCESS_KEY": secret, "YETO_RL_STORE_S3_ACCESS_KEY_ID": "k"})
    assert store_client_environ(args)["YETO_RL_STORE_S3_SECRET_ACCESS_KEY"] == secret
    assert secret not in repr(vars(args)) and secret not in json.dumps(cli._serializable_args(args), default=str)
    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    args_dict = {k: v for k, v in vars(args).items() if k != "command"}
    runs.create_run("r-test", json.loads(json.dumps(args_dict, default=str)))
    for root, _dirs, files in os.walk(tmp_path / "runs"):
        for name in files:
            assert secret not in (pathlib.Path(root) / name).read_text(errors="replace")
    assert secret not in os.environ.get("YETO_RL_STORE_S3_SECRET_ACCESS_KEY", "")
    key = id(args)
    del args
    gc.collect()
    assert key not in store_env._BOUND  # dropped with its args
