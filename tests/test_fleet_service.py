"""The on-site control plane of a fleet (``yeto fleet service``, ``fleet.syncer_service``):
the systemd units of ``yeto run up`` and of the fleet's syncer, their install over SSH (a
fake ``ssh``), the syncer launch that watches an on-site syncer service instead of starting
one, and the syncer service itself (fake syncer; the real one in
``test_fleet_service_syncer_e2e.py``). No cloud, no systemd."""

from __future__ import annotations

import json
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
StartLimitIntervalSec=3600
StartLimitBurst=5

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
RestartPreventExitStatus=75
SuccessExitStatus=143
KillMode=process
TimeoutStopSec=300

[Install]
WantedBy=multi-user.target
"""


def test_the_control_unit_snapshot():
    assert fs.unit_text(spec("control")) == CONTROL_UNIT
    assert fs.CONTROL_PREVENT == (fs.CONTROL_DECIDED_EXIT,) == (75,)  # run up's exit 1, mapped; nothing else


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

    def serve(args, addr, n, *, clusters, head_mode, local_syncer, record_extra, on_record, stopped, early_finish):
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
    rc = fs.run_syncer(str(path), objects=objects, make_syncer=lambda args, n: syncer,
                       serve=fake_serve(objects, seen, code, until=until), parse=_prepared, node=node,
                       beat_s=beat_s, workdir=str(workdir or tmp_path / "work"), log=logs.append)
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
    assert syncer.events == ["start", "stop"]


def test_the_term_goes_on_here_and_moves_up_on_another_host(tmp_path, monkeypatch):
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
    rc, seen, syncer, objects, logs = run_service(tmp_path / "again", monkeypatch, 0, objects=objects)
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

def service_args(path, role, *, clear_ended=False):
    return types.SimpleNamespace(role=role, config=str(path), syncer_binary=None, stop_grace_s=170,
                                 clear_ended=clear_ended)


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
def test_a_marker_that_does_not_read_is_a_crash_not_a_start(tmp_path, monkeypatch):
    class Broken(DirectoryObjectStore):
        def read(self, key):
            raise OSError("store down")

    path = onsite_config(tmp_path)
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    clock = Clock(0.0)
    monkeypatch.setattr(fs, "_sleep", clock.sleep)
    monkeypatch.setattr(fs, "_monotonic", clock)
    monkeypatch.setattr(fs, "run_control", lambda *a, **kw: pytest.fail("not started without knowing"))
    assert fs.service_run(service_args(path, "control"), objects=Broken(tmp_path / "store"),
                          log=lambda m: None) == fs.CRASH_EXIT
    assert clock.t >= fs.ENDED_READ_WAIT_S  # asked again in the process for ten minutes first


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
    monkeypatch.setattr(fs, "_monotonic", clock)
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
        return codes.pop(0)

    codes = [1, 2, 0, 143]
    monkeypatch.setattr(up, "main", run_up)
    logs: list[str] = []
    assert [fs.run_control(str(path), log=logs.append) for _ in range(4)] == [75, 2, 0, 143]
    assert seen == [False] * 4  # never drains the pools on exit
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
                          log=logs.append) == 2
    assert "rollout.drain_on_exit is true" in logs[-1]


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


@pytest.mark.usefixtures("no_sigterm_handler")
@pytest.mark.parametrize("end, code", [("stop-run", 0), ("fleet stop", 0), ("trainer exit", 0), ("breaker", 1),
                                       ("sigterm", None), ("crash", "crash")])
def test_the_control_service_drains_when_run_up_ends_the_run_and_keeps_the_pools_on_a_sigterm(
        tmp_path, monkeypatch, end, code):
    """The on-site control unit runs run up with rollout.drain_on_exit false, but only a SIGTERM
    (a stop or restart of the unit) or a crash leaves the pools running: an end run up decides
    itself (a stop-run, the fleet's STOP, the trainer's exit, a breaker) drains them."""
    from yeto.launch.pool import controller as ctl

    path = onsite_config(tmp_path)
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    objects, drains, made = DirectoryObjectStore(tmp_path / "store"), [], []
    monkeypatch.setattr(ctl, "drain_active", lambda why: drains.append(why))

    class Controller:
        stopped, detached = False, None
        pools = {"i0-sglang": types.SimpleNamespace(control=object())}  # a pool on the store control plane

        def __init__(self, drain_on_exit):
            self.drain_on_exit = drain_on_exit

        def detach(self, why):
            self.detached = why

    def pool(self):
        self.controller = Controller(bool(up.rollout_section(self.cfg).get("drain_on_exit", True)))
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
    assert controller.drain_on_exit is False  # the unit's run up never drains on its own
    if code in (0, 1):
        assert drains == ["process exit"] and marker.exists()  # drained, and the run is over
    else:
        assert drains == [] and not marker.exists() and controller.detached == "process exit"  # the pools stay


def test_a_plain_run_up_drains_as_before(tmp_path):
    """Without the service mode drain_controller is what it was: drain_on_exit decides."""
    from yeto.launch.pool import controller as ctl

    run = up.Run(up.load_config(str(onsite_config(tmp_path))), state_root=tmp_path / "state")

    class Controller:
        stopped, detached, pools = False, None, {}
        drain_on_exit = False

        def detach(self, why):
            self.detached = why

    run.controller = Controller()
    calls = []
    real, ctl.drain_active = ctl.drain_active, lambda why: calls.append(why)
    try:
        assert run.drain_controller(decided=True) is True and calls == []  # not the service: no drain
        run.onsite_service = True
        assert run.drain_controller(decided=False) is True and calls == []  # a SIGTERM of the unit: kept
        assert run.drain_controller(decided=True) is True and calls == ["process exit"]
    finally:
        ctl.drain_active = real


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


def test_a_lease_that_could_not_be_renewed_near_its_end_fences_the_service(tmp_path):
    from yeto.launch.pool.store_lock import LockUnconfirmed

    class Lock:
        margin_s = 12.0

        def __init__(self, left, error):
            self.left, self.error = left, error

        def renew(self):
            raise self.error

        def lease_left(self):
            return self.left

    objects = DirectoryObjectStore(tmp_path / "store")
    for left, error, lost in [(100.0, OSError("store blip"), False), (12.0, OSError("store down"), True),
                              (90.0, LockUnconfirmed("not confirmed"), True)]:
        beat_ = fs.Heartbeat(objects, "10.0.0.5", "kid", lock=Lock(left, error), generation=1, log=lambda m: None)
        beat_.renew()
        assert (beat_.lost is not None) is lost


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
    assert fleet_syncer.serve(args, "10.0.0.5:29400", 1, clusters=[], head_mode=True, local_syncer=Syncer(),
                              sleep=clock.sleep, early_finish=True) == 0
    assert polls["n"] == 3 and island_launches.read_syncer_record(objects) is not None  # waited, then recorded
    done, polls["n"] = Syncer(), -100
    done.exited = True  # the resumed syncer itself ran to its end before it was seen listening
    objects2 = DirectoryObjectStore(tmp_path / "store2")
    monkeypatch.setattr(fleet_syncer, "_store_objects", lambda args: objects2)
    assert fleet_syncer.serve(args, "10.0.0.5:29400", 1, clusters=[], head_mode=True, local_syncer=done,
                              sleep=clock.sleep, early_finish=True) == 0
    assert island_launches.read_syncer_record(objects2) is None
