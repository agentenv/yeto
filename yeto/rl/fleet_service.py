"""``yeto fleet service``: a fleet's control plane and syncer as systemd services on site.

A fleet launched island by island (``fleet.launch: per_island``, store control plane) can
keep its control plane off the operator's laptop: ``yeto run up`` (the fleet controller,
its pool workers and its trainer worker) and the fleet's syncer each run as a systemd
service on a machine of the cluster's site (``fleet.syncer_service`` names the syncer's
host). ``--print`` writes the unit to stdout; ``--install <host>`` writes it on that host
over SSH (``/etc/systemd/system/<unit>``, ``sudo``) and runs ``systemctl enable --now``,
nothing else.

The units (:func:`unit_text`):

* ``Restart=on-failure``, but never after an exit yeto chose itself
  (``RestartPreventExitStatus``): ``run up`` exits 1 after a breaker trip and 2 when it
  refuses; the syncer service exits 2 / 10 on a configuration refusal and 11 on a strict
  failure. A crash (an uncaught exception: :data:`CRASH_EXIT`, a signal) is restarted;
  the restarts are capped (``StartLimitBurst`` within ``StartLimitIntervalSec``).
* Credentials only through ``EnvironmentFile=`` (mode 0600; ``ExecStartPre`` refuses the
  start otherwise). The unit names the file, never a value; ``--install`` checks the
  file's mode on the host (``stat``) and never reads it. The island HMAC key
  (``YETO_ISLAND_HMAC_KEY``) belongs in that file: both units must read the same key.
* The working directory, run up's state (``YETO_STATE_DIR``) and the launch registry
  (``YETO_RUNS_DIR``) are under the host's scratch directory (``--workdir``).
* The control unit kills only ``run up`` itself on a stop or restart (``KillMode=process``):
  the island launches it started keep supervising their islands, and the next ``run up``
  adopts them. The syncer unit stops the whole process group: the syncer gets SIGTERM and
  writes its checkpoint, and the restarted service resumes from it.

``yeto fleet service-run --role control|syncer --config <run.json>`` is what the units run:
``control`` is ``yeto run up --config`` (an uncaught exception exits :data:`CRASH_EXIT`,
restarted); ``syncer`` runs the fleet's syncer on this host (:func:`run_syncer`).
"""

from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROLES = ("control", "syncer")
UNIT_DIR = "/etc/systemd/system"
DEFAULT_WORKDIR = "/var/tmp/yeto"  # <workdir>/<run>/<role>
CRASH_EXIT = 70  # an uncaught exception in the service process (EX_SOFTWARE): restarted
# exits a unit is never restarted after: yeto decided them
CONTROL_PREVENT = (1, 2)  # run up: a breaker trip / failure it handled, a refusal
SYNCER_PREVENT = (2, 10, 11)  # the syncer service: a refusal, a configuration error, a strict failure
RESTART_SEC = 30
START_LIMIT_INTERVAL_S = 3600
START_LIMIT_BURST = 5
STOP_TIMEOUT_S = {"control": 300, "syncer": 180}  # run up's exit detach / the syncer's checkpoint
# what a unit file line may hold of a path or name: no spaces, quotes, specifiers or variables
_SAFE_PATH = re.compile(r"/[A-Za-z0-9._/+-]*")
_SAFE_USER = re.compile(r"[a-z_][a-z0-9_-]{0,31}")
_SAFE_HOST = re.compile(r"[A-Za-z0-9_.@-]+|\[[0-9A-Fa-f:]+\]")


class ServiceError(ValueError):
    """A unit that cannot be written (a path, user or role the unit file cannot hold)."""


@dataclass(frozen=True)
class ServiceSpec:
    run: str
    role: str
    config: str  # the run config's absolute path on the service host
    env_file: str  # the 0600 environment file on the service host
    workdir: str  # the service's scratch directory on its host
    user: str
    python: str  # the Python that runs yeto on the service host
    yeto_dir: str  # the yeto checkout on the service host (PYTHONPATH)

    @property
    def unit(self) -> str:
        return unit_name(self.run, self.role)


def unit_name(run: str, role: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "-", str(run)).strip("-.") or "run"
    return f"yeto-{safe}-{role}.service"


def check_spec(spec: ServiceSpec) -> None:
    if spec.role not in ROLES:
        raise ServiceError(f"--role must be one of {list(ROLES)} (got {spec.role!r})")
    for what, value in (("--config", spec.config), ("--env-file", spec.env_file), ("--workdir", spec.workdir),
                        ("--python", spec.python), ("--yeto-dir", spec.yeto_dir)):
        if not _SAFE_PATH.fullmatch(str(value)) or "/../" in f"{value}/":
            raise ServiceError(f"{what} {value!r}: an absolute path on the service host of letters, digits and "
                               "._/+- only (a unit file line holds it as is)")
    if not _SAFE_USER.fullmatch(spec.user):
        raise ServiceError(f"--user {spec.user!r} is not a user name")


def unit_text(spec: ServiceSpec) -> str:
    """The systemd unit of ``spec`` (module doc). No value of the environment is in it."""
    check_spec(spec)
    state = f"{spec.workdir}/rl-up"
    registry = f"{spec.workdir}/runs"
    prevent = CONTROL_PREVENT if spec.role == "control" else SYNCER_PREVENT
    what = "control plane (yeto run up)" if spec.role == "control" else "syncer"
    lines = [
        f"# The fleet {what} of run {spec.run}, written by `yeto fleet service --role {spec.role}`.",
        "# Credentials come from EnvironmentFile only (mode 0600); this file holds none.",
        "[Unit]",
        f"Description=yeto fleet {what} of run {spec.run}",
        "Wants=network-online.target",
        "After=network-online.target",
        f"StartLimitIntervalSec={START_LIMIT_INTERVAL_S}",
        f"StartLimitBurst={START_LIMIT_BURST}",
        "",
        "[Service]",
        "Type=simple",
        f"User={spec.user}",
        f"EnvironmentFile={spec.env_file}",
        f"Environment=PYTHONPATH={spec.yeto_dir}",
        "Environment=PYTHONUNBUFFERED=1",
        f"Environment=YETO_STATE_DIR={state}",
        f"Environment=YETO_RUNS_DIR={registry}",
        # the env file must not be readable by others: refused before anything starts
        f"ExecStartPre=/bin/sh -c 'test \"$$(stat -c %%a {spec.env_file})\" = 600 || "
        f"{{ echo \"{spec.env_file} must have mode 0600\" >&2; exit 1; }}'",
        f"ExecStartPre=/bin/mkdir -p {spec.workdir}",
        f"WorkingDirectory=-{spec.workdir}",
        f"ExecStart={spec.python} -m yeto.cli fleet service-run --role {spec.role} --config {spec.config}",
        "Restart=on-failure",
        f"RestartSec={RESTART_SEC}",
        f"RestartPreventExitStatus={' '.join(str(c) for c in prevent)}",
        # control: only run up itself; its island launches keep their islands and are adopted again
        f"KillMode={'process' if spec.role == 'control' else 'control-group'}",
        f"TimeoutStopSec={STOP_TIMEOUT_S[spec.role]}",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
        "",
    ]
    return "\n".join(lines)


def install_script(spec: ServiceSpec, text: str) -> str:
    """The shell script ``--install`` runs on the service host: check the env file's mode
    (``stat``, its content is never read), write the unit, ``systemctl enable --now``."""
    unit = f"{UNIT_DIR}/{spec.unit}"
    marker = "YETO_UNIT_EOF"
    if marker in text:
        raise ServiceError("the unit text holds the heredoc marker")
    env = shlex.quote(spec.env_file)
    return "\n".join([
        "set -eu",
        f"mode=$(stat -c %a {env} 2>/dev/null) || {{ echo 'yeto fleet service: no env file {spec.env_file} "
        "on this host' >&2; exit 3; }",
        f"[ \"$mode\" = 600 ] || {{ echo \"yeto fleet service: {spec.env_file} has mode $mode; chmod 600 it\" "
        ">&2; exit 3; }",
        f"sudo -n tee {shlex.quote(unit)} >/dev/null <<'{marker}'",
        text.rstrip("\n"),
        marker,
        "sudo -n systemctl daemon-reload",
        f"sudo -n systemctl enable --now {shlex.quote(spec.unit)}",
        f"echo 'yeto fleet service: {spec.unit} enabled and started'",
        "",
    ])


def install(spec: ServiceSpec, host: str, *, runner: Callable[..., Any] | None = None,
            log: Callable[[str], None] = print) -> int:
    """Write the unit on ``host`` and enable it (``ssh <host> bash -s``)."""
    runner = runner or subprocess.run
    if not _SAFE_HOST.fullmatch(str(host)) or str(host).startswith("-"):
        raise ServiceError(f"--install {host!r} is not an SSH host")
    script = install_script(spec, unit_text(spec))
    done = runner(["ssh", "-o", "BatchMode=yes", str(host), "bash", "-s"], input=script, text=True,
                  capture_output=True, timeout=300)
    for line in str(getattr(done, "stdout", "") or "").splitlines():
        log(line)
    if done.returncode != 0:
        err = str(getattr(done, "stderr", "") or "").strip().splitlines()
        log(f"yeto fleet service: install of {spec.unit} on {host} failed (exit {done.returncode})"
            + (f": {err[-1]}" if err else ""))
        return 1
    log(f"yeto fleet service: {spec.unit} on {host}: `ssh {host} systemctl status {spec.unit}`, "
        f"`ssh {host} journalctl -u {spec.unit}`")
    return 0


# -- the CLI ------------------------------------------------------------------------------------

def add_parser(fsub: Any) -> None:
    p = fsub.add_parser("service", help="the fleet's control plane or syncer as a systemd unit on site")
    p.add_argument("--config", required=True, help="the run config (fleet.launch per_island, store control plane)")
    p.add_argument("--role", required=True, choices=ROLES,
                   help="control: yeto run up of the fleet; syncer: the fleet's syncer (fleet.syncer_service)")
    p.add_argument("--env-file", required=True,
                   help="the environment file (mode 0600) on the service host: the run's credentials and "
                   "YETO_ISLAND_HMAC_KEY. The unit names it; no value goes into the unit")
    where = p.add_mutually_exclusive_group(required=True)
    where.add_argument("--print", action="store_true", help="write the unit to stdout")
    where.add_argument("--install", metavar="HOST", default=None,
                       help="write the unit on HOST over SSH and run systemctl enable --now there")
    p.add_argument("--workdir", default=None,
                   help=f"the scratch directory on the service host (default {DEFAULT_WORKDIR}/<run>/<role>)")
    p.add_argument("--user", default=None, help="the user the service runs as (default: this user)")
    p.add_argument("--python", default=None, help="the Python on the service host (default: this one)")
    p.add_argument("--yeto-dir", default=None, help="the yeto checkout on the service host (default: this one)")
    p.add_argument("--remote-config", default=None,
                   help="the run config's path on the service host (default: --config, made absolute)")
    r = fsub.add_parser("service-run", help="what a `yeto fleet service` unit runs (in the foreground)")
    r.add_argument("--role", required=True, choices=ROLES)
    r.add_argument("--config", required=True, help="the run config")
    r.add_argument("--syncer-binary", default=None,
                   help="syncer: a built yeto-syncer (default: cargo build --release of this checkout)")


def spec_from_args(args: Any) -> ServiceSpec:
    from yeto.rl import up  # noqa: PLC0415

    cfg = up.load_config(args.config)  # the config only: no state directory is made here
    problem = onsite_problem(cfg, args.role)
    if problem:
        raise ServiceError(problem)
    name = str(cfg["run"])
    import getpass  # noqa: PLC0415

    from yeto.launch import REPO_ROOT  # noqa: PLC0415

    return ServiceSpec(run=name, role=args.role,
                       config=args.remote_config or str(Path(args.config).expanduser().resolve()),
                       env_file=str(args.env_file),
                       workdir=args.workdir or f"{DEFAULT_WORKDIR}/{unit_name(name, args.role)[5:-8]}",
                       user=args.user or getpass.getuser(), python=args.python or sys.executable,
                       yeto_dir=args.yeto_dir or str(REPO_ROOT))


def onsite_problem(cfg: dict[str, Any], role: str) -> str | None:
    """Why the run config cannot run on site as ``role`` (None: it can)."""
    from yeto.rl import fleet  # noqa: PLC0415
    from yeto.rl.up import rollout_section  # noqa: PLC0415

    plan = fleet.plan_of(cfg)
    if plan is None or not plan.per_island:
        return "an on-site control plane is a fleet launched island by island (fleet.launch per_island)"
    if rollout_section(cfg).get("control_plane") != "store":
        return "an on-site control plane needs rollout.control_plane \"store\" (its state lives in the run store)"
    if role == "syncer" and not plan.syncer_service:
        return "--role syncer: the run config names no syncer host (fleet.syncer_service)"
    return None


def main(args: Any) -> int:
    if args.fleet_command == "service-run":
        return service_run(args)
    try:
        spec = spec_from_args(args)
        if args.print:
            sys.stdout.write(unit_text(spec))
            return 0
        return install(spec, args.install)
    except (ServiceError, ValueError, OSError) as error:
        print(f"yeto fleet service: refused: {error}", file=sys.stderr)
        return 2


def service_run(args: Any) -> int:
    signal.signal(signal.SIGTERM, _exit_on_sigterm)  # the exit path runs (the syncer's stop)
    try:
        if args.role == "control":
            return run_control(args.config)
        return run_syncer(args.config, binary=args.syncer_binary)
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 - a crash: the unit restarts it
        import traceback  # noqa: PLC0415

        traceback.print_exc()
        return CRASH_EXIT


def _exit_on_sigterm(signum: int, frame: Any) -> None:
    raise SystemExit(128 + signum)


def run_control(config: str) -> int:
    """``yeto run up --config <config>`` in this process."""
    from yeto import cli  # noqa: PLC0415

    return int(cli.main(["run", "up", "--config", str(config)]) or 0)


# -- the syncer service -----------------------------------------------------------------------

def launch_args(argv: list[str]) -> Any:
    """``yeto launch <argv>`` parsed and prepared as the launch does (the syncer's profile and
    command are built from these)."""
    from yeto import cli  # noqa: PLC0415
    from yeto.launch.data_mix import materialize_data_mix  # noqa: PLC0415
    from yeto.launcher import prepare_launch_args  # noqa: PLC0415

    parser = cli.build_parser()
    args = parser.parse_args(["launch", *argv])
    if getattr(args, "breakers", None):
        from yeto.launch import breakers_file  # noqa: PLC0415

        launch_argv, resolved = breakers_file.apply(args.breakers, argv)
        args = parser.parse_args(["launch", *launch_argv])
        args._breakers_resolved = resolved
    materialize_data_mix(args)
    prepare_launch_args(args)
    return args


def syncer_launch_argv(run: Any) -> tuple[list[str], str]:
    """The fleet's syncer launch argv as ``run up`` starts it, without the service flag
    (this process runs that syncer), and the service host."""
    from yeto.rl.island_launches import SYNCER_SERVICE_FLAG, _without  # noqa: PLC0415

    argv = [str(a) for a in run.trainer_argv()]
    host = run.fleet_plan().syncer_service
    return _without(argv, (SYNCER_SERVICE_FLAG,)), str(host)


class Heartbeat:
    """The service's heartbeat in the run store every :data:`SERVICE_BEAT_S` (a thread)."""

    def __init__(self, objects: Any, host: str, key: str | None, *, state: Callable[[], str],
                 every_s: float | None = None, log: Callable[[str], None] = print) -> None:
        from yeto.launch.fleet_syncer import SERVICE_BEAT_S  # noqa: PLC0415

        self.objects, self.host, self.key, self.state, self.log = objects, host, key, state, log
        self.every_s = SERVICE_BEAT_S if every_s is None else float(every_s)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="syncer-service-beat", daemon=True)

    def beat(self, state: str | None = None, reason: str | None = None) -> None:
        from yeto.launch.fleet_syncer import write_service_status  # noqa: PLC0415

        try:
            write_service_status(self.objects, host=self.host, state=state or self.state(), key=self.key,
                                 reason=reason)
        except Exception as error:  # noqa: BLE001 - tried again at the next beat
            self.log(f"[syncer service] WARNING: heartbeat not written ({type(error).__name__}: {error})")

    def start(self) -> None:
        self.beat()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self.every_s):
            self.beat()


def run_syncer(config: str, *, binary: str | None = None, objects: Any = None,
               make_syncer: Callable[..., Any] | None = None, serve: Callable[..., int] | None = None,
               parse: Callable[[list[str]], Any] | None = None, log: Callable[[str], None] = print) -> int:
    """The fleet's syncer on this host (``fleet.syncer_service``): the syncer launch's syncer,
    run as the head runs it (``LocalSyncer``: a process that dies is restarted from its
    checkpoint, and its record written again), plus the service's heartbeat. 0: the run is
    finished; 11: a strict failure (never restarted); 2 / 10: a refusal; 1: anything else
    (the unit restarts the service, which resumes from the checkpoint)."""
    from yeto.config_errors import CONFIG_ERROR_EXIT, ISLAND_STOP_EXIT  # noqa: PLC0415
    from yeto.gpu_spec import parse_gpu_spec  # noqa: PLC0415
    from yeto.rl import up  # noqa: PLC0415
    from yeto.rl.engine import sample_roles  # noqa: PLC0415

    from yeto.launch import fleet_syncer  # noqa: PLC0415
    from yeto.launch.fleet import _RlStrictFailure  # noqa: PLC0415
    from yeto.launch.syncer import ISLAND_HMAC_KEY_ENV, syncer_shard_addrs  # noqa: PLC0415

    try:
        run = up.open_run(config)
    except up.RunConfigError as error:
        log(f"[syncer service] refused: {error}")
        return 2
    problem = onsite_problem(run.cfg, "syncer")
    if problem:
        log(f"[syncer service] refused: {problem}")
        return 2
    for name, value in (run.cfg.get("env") or {}).items():  # the run config's own env (run up gives its children)
        os.environ.setdefault(str(name), os.path.expanduser(str(value)))
    key = os.environ.get(ISLAND_HMAC_KEY_ENV)
    refused = fleet_syncer.key_mismatch(fleet_syncer.key_id(key), None)
    if refused:
        log(f"[syncer service] refused: {refused}")
        return CONFIG_ERROR_EXIT
    argv, host = syncer_launch_argv(run)
    args = (parse or launch_args)(argv)
    objects = objects if objects is not None else fleet_syncer._store_objects(args)
    if objects is None:
        log(f"[syncer service] refused: the run store {args.rl_checkpoint_store!r} has no object API here")
        return 2
    num_learners = (sample_roles.syncer_learner_count(args, len(parse_gpu_spec(args.gpu)))
                    + (getattr(args, "external_learners", 0) or 0))
    # its term: never lower than the fleet's last record (a resumed syncer keeps its own)
    fleet_syncer.resolve_epoch(args, sample_roles.FLEET_ROLE_SYNCER, adopted=False, reader=lambda: None,
                               objects=objects, log=log)
    addr = syncer_shard_addrs(host, args)
    if make_syncer is None:
        from yeto.launch.local_syncer import LocalSyncer  # noqa: PLC0415
        from yeto.launch.syncer import build_syncer_binary  # noqa: PLC0415

        path = binary or str(build_syncer_binary())
        syncer = LocalSyncer(args, num_learners, binary=path, log_file=str(Path.cwd() / "yeto-syncer.log"))
    else:
        syncer = make_syncer(args, num_learners)
    beat = Heartbeat(objects, host, fleet_syncer.key_id(key),
                     state=lambda: fleet_syncer.SERVICE_RUNNING, log=log)
    syncer.start()
    for hook in ("start_log_forwarder", "start_store_upload"):
        method = getattr(syncer, hook, None)
        if method is not None:
            method(args) if hook == "start_store_upload" else method()
    log(f"[syncer service] the fleet syncer of run {run.run} runs on this host at {addr} "
        f"(epoch {args.rl_syncer_epoch}, {num_learners} launched learner(s))")
    beat.start()
    code, reason = 1, None
    try:
        code = (serve or fleet_syncer.serve)(
            args, addr, num_learners, clusters=[], head_mode=True, local_syncer=syncer,
            record_extra={"service": host, "key_id": fleet_syncer.key_id(key)})
        if code != 0:
            probe = syncer.probe()
            if isinstance(probe, _RlStrictFailure):
                reason = str(getattr(probe, "reason", probe))
                code = ISLAND_STOP_EXIT
    finally:
        beat.stop()
        state = (fleet_syncer.SERVICE_FINISHED if code == 0 else
                 fleet_syncer.SERVICE_FAILED if code == ISLAND_STOP_EXIT else None)
        if state is not None:
            beat.beat(state, reason)
        syncer.stop()
    log(f"[syncer service] ended: exit {code}" + (f" ({reason})" if reason else ""))
    return code

