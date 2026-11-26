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

def spec(role: str = "control") -> fs.ServiceSpec:
    return fs.ServiceSpec(run="r", role=role, config="/srv/yeto/run.json", env_file="/etc/yeto/r.env",
                          workdir=f"/var/tmp/yeto/r-{role}", user="yeto", python="/opt/yeto-venv/bin/python",
                          yeto_dir="/opt/yeto")


CONTROL_UNIT = """\
# The fleet control plane (yeto run up) of run r, written by `yeto fleet service --role control`.
# Credentials come from EnvironmentFile only (mode 0600); this file holds none.
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
ExecStartPre=/bin/sh -c 'test "$$(stat -c %%a /etc/yeto/r.env)" = 600 || { echo "/etc/yeto/r.env must have mode 0600" >&2; exit 1; }'
ExecStartPre=/bin/mkdir -p /var/tmp/yeto/r-control
WorkingDirectory=-/var/tmp/yeto/r-control
ExecStart=/opt/yeto-venv/bin/python -m yeto.cli fleet service-run --role control --config /srv/yeto/run.json
Restart=on-failure
RestartSec=30
RestartPreventExitStatus=1 2
KillMode=process
TimeoutStopSec=300

[Install]
WantedBy=multi-user.target
"""


def test_the_control_unit_snapshot():
    assert fs.unit_text(spec("control")) == CONTROL_UNIT


def test_the_syncer_unit_restarts_on_a_crash_only_and_stops_its_whole_group():
    text = fs.unit_text(spec("syncer"))
    assert "ExecStart=/opt/yeto-venv/bin/python -m yeto.cli fleet service-run --role syncer --config " \
           "/srv/yeto/run.json" in text
    assert "Restart=on-failure\n" in text
    assert "RestartPreventExitStatus=2 10 11\n" in text  # a refusal, a config error, a strict failure
    assert "KillMode=control-group\n" in text  # the syncer gets SIGTERM and checkpoints
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


@pytest.mark.parametrize("field, value", [("env_file", "/etc/yeto/r env"), ("config", "run.json"),
                                          ("workdir", "/var/tmp/$HOME"), ("python", "/opt/%h/python"),
                                          ("user", "Root User"), ("role", "head")])
def test_a_unit_line_that_would_not_hold_the_value_is_refused(field, value):
    with pytest.raises(fs.ServiceError):
        fs.unit_text(types.SimpleNamespace(**{**spec().__dict__, field: value}))


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
    commands = [line for line in script.splitlines() if line and not line.startswith(("[", "mode=", "echo"))]
    body = script.split("<<'YETO_UNIT_EOF'\n", 1)[1].split("\nYETO_UNIT_EOF\n", 1)[0]
    assert body + "\n" == fs.unit_text(spec("syncer"))  # the unit, verbatim
    outside = [c for c in commands if c not in body.splitlines() and c != "YETO_UNIT_EOF"]
    assert outside == ["set -eu", "sudo -n tee /etc/systemd/system/yeto-r-syncer.service >/dev/null <<'YETO_UNIT_EOF'",
                       "sudo -n systemctl daemon-reload", "sudo -n systemctl enable --now yeto-r-syncer.service"]
    # the env file: its mode is checked, its content never read
    assert "stat -c %a /etc/yeto/r.env" in script
    for verb in ("cat ", "source ", ". /etc", "grep ", "head ", "less "):
        assert verb not in script.replace(body, "")


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


# -- the syncer launch watches the service ---------------------------------------------------------

class Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


def watch_args(tmp_path):
    return types.SimpleNamespace(rl_fleet_syncer_service="10.0.0.5", cluster_prefix="r",
                                 rl_checkpoint_store=str(tmp_path / "store"), rl_fleet_syncer_grace_s=60.0,
                                 rl_syncer_epoch=None)


def service_up(objects, clock, *, key=KEY, epoch=3, state=fleet_syncer.SERVICE_RUNNING):
    island_launches.write_syncer_record(objects, addr="10.0.0.5:29400", syncer_epoch=epoch, learners=2,
                                        max_learners=4, run="r", clock=lambda: clock() - 7200,
                                        extra={"service": "10.0.0.5", "key_id": fleet_syncer.key_id(key)})
    fleet_syncer.write_service_status(objects, host="10.0.0.5", state=state, key=fleet_syncer.key_id(key),
                                      clock=clock)


def test_the_syncer_launch_waits_for_the_service_records_it_again_and_ends_with_it(tmp_path, monkeypatch):
    objects, clock, logs = DirectoryObjectStore(tmp_path / "store"), Clock(), []
    args = watch_args(tmp_path)
    ticks = {"n": 0}

    def sleep(s):
        clock.sleep(s)
        ticks["n"] += 1
        if ticks["n"] == 3:
            service_up(objects, clock)  # the service comes up while the launch waits
        elif ticks["n"] > 3:
            fleet_syncer.write_service_status(objects, host="10.0.0.5", key=fleet_syncer.key_id(KEY), clock=clock,
                                              state=fleet_syncer.SERVICE_RUNNING if ticks["n"] < 8
                                              else fleet_syncer.SERVICE_FINISHED)
            if ticks["n"] == 4:
                record = island_launches.read_syncer_record(objects)
                assert record["at"] >= clock() - 60  # announced again for this run up's trainer worker
                assert record["syncer_epoch"] == 3 and record["service"] == "10.0.0.5"

    assert fleet_syncer.watch_service(args, objects=objects, sleep=sleep, clock=clock, log=logs.append) == 0
    assert args.rl_syncer_epoch == 3 and ticks["n"] == 8
    assert any("starts no syncer" in m for m in logs)


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
    assert fleet_syncer.watch_service(watch_args(tmp_path), objects=objects, sleep=clock.sleep, clock=clock,
                                      log=logs.append) == 1  # no heartbeat after the first: down, then the grace
    assert any("no heartbeat from the syncer service" in m and "for " in m for m in logs)
    clock2, logs2 = Clock(), []
    objects2 = DirectoryObjectStore(tmp_path / "store2")
    service_up(objects2, clock2)

    def sleep(s):
        clock2.sleep(s)
        fleet_syncer.write_service_status(objects2, host="10.0.0.5", state=fleet_syncer.SERVICE_FAILED,
                                          key=fleet_syncer.key_id(KEY), reason="strict: merge mismatch", clock=clock2)

    assert fleet_syncer.watch_service(watch_args(tmp_path), objects=objects2, sleep=sleep, clock=clock2,
                                      log=logs2.append) == 1
    assert "strict: merge mismatch" in logs2[-1]


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


def run_service(tmp_path, monkeypatch, code: int, *, strict: bool = False):
    path = onsite_config(tmp_path)
    objects = DirectoryObjectStore(tmp_path / "store")
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "state")
    seen = {}

    def serve(args, addr, n, *, clusters, head_mode, local_syncer, record_extra):
        seen.update(addr=addr, n=n, head_mode=head_mode, extra=record_extra, epoch=args.rl_syncer_epoch,
                    service=getattr(args, "rl_fleet_syncer_service", None))
        status = fleet_syncer.read_service_status(objects)
        seen["beat"] = status
        return code

    syncer = FakeSyncer(strict)
    logs: list[str] = []
    rc = fs.run_syncer(str(path), objects=objects, make_syncer=lambda args, n: syncer, serve=serve,
                       parse=_prepared, log=logs.append)
    return rc, seen, syncer, objects, logs


def test_the_service_runs_the_syncer_launch_s_syncer_here_and_says_when_it_finished(tmp_path, monkeypatch):
    rc, seen, syncer, objects, logs = run_service(tmp_path, monkeypatch, 0)
    assert rc == 0 and syncer.events == ["start", "stop"]
    assert seen["addr"] == "10.0.0.5:29400" and seen["n"] == 2 and seen["head_mode"] is True
    assert seen["service"] is None  # the syncer runs here: the launch's own args, not a watcher's
    assert seen["extra"] == {"service": "10.0.0.5", "key_id": fleet_syncer.key_id(KEY)}
    assert seen["beat"]["state"] == "running" and seen["beat"]["key_id"] == fleet_syncer.key_id(KEY)
    assert fleet_syncer.read_service_status(objects)["state"] == "finished"
    raw = (tmp_path / "store" / fleet_syncer.SERVICE_RECORD).read_text()
    assert KEY not in raw


def test_a_strict_failure_of_the_service_is_final_anything_else_restarts(tmp_path, monkeypatch):
    rc, _seen, _syncer, objects, logs = run_service(tmp_path, monkeypatch, 1, strict=True)
    assert rc == ISLAND_STOP_EXIT and rc in fs.SYNCER_PREVENT
    status = fleet_syncer.read_service_status(objects)
    assert status["state"] == "failed" and "merge_invariant" in status["reason"]
    (tmp_path / "again").mkdir()
    rc, _seen, _syncer, objects, _logs = run_service(tmp_path / "again", monkeypatch, 1)
    assert rc == 1 and rc not in fs.SYNCER_PREVENT  # the unit restarts it; it resumes from its checkpoint
    assert fleet_syncer.read_service_status(objects)["state"] == "running"


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


def test_a_crash_of_the_service_process_is_the_restartable_exit(monkeypatch):
    monkeypatch.setattr(fs.signal, "signal", lambda *a: None)  # not this test process's SIGTERM
    monkeypatch.setattr(fs, "run_control", lambda config: 1 / 0)
    args = types.SimpleNamespace(role="control", config="/srv/run.json", syncer_binary=None)
    assert fs.service_run(args) == fs.CRASH_EXIT
    monkeypatch.setattr(fs, "run_control", lambda config: 1)  # a breaker trip: run up's own exit
    assert fs.service_run(args) == 1 and 1 in fs.CONTROL_PREVENT


def test_the_documented_commands_run(tmp_path, monkeypatch, capsys):
    """docs/OWN_CLUSTERS.md, "On-site control plane": every `yeto fleet service` command parses,
    the --print ones write a unit, the --install ones go through (a fake) ssh only."""
    import re
    from pathlib import Path

    doc = (Path(__file__).resolve().parents[1] / "docs" / "OWN_CLUSTERS.md").read_text()
    section = doc.split("## On-site control plane", 1)[1].split("\n## ", 1)[0]
    commands = re.findall(r"^yeto (fleet service .+)$", section, flags=re.M)
    assert len(commands) == 4
    config = onsite_config(tmp_path)
    ssh = FakeSsh()
    monkeypatch.setattr(fs.subprocess, "run", ssh)
    for line in commands:
        argv = (line.replace("run.json", str(config)).replace("<run>", "r")
                .replace("<syncer-host>", "site-s").replace("<control-host>", "site-c").split())
        assert fleet_cli.main(fleet_cli_args(argv)) == 0, line
    out = capsys.readouterr().out
    assert out.count("[Service]") == 2 and [c[0][3] for c in ssh.calls] == ["site-s", "site-c"]
