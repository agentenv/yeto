"""``yeto fleet service``: a fleet's control plane and syncer as systemd services on site.

A fleet launched island by island (``fleet.launch: per_island``, store control plane) can
keep its control plane off the operator's laptop: ``yeto run up`` (the fleet controller,
its pool workers and its trainer worker) and the fleet's syncer each run as a systemd
service on a machine of the cluster's site (``fleet.syncer_service`` names the syncer's
host). ``--print`` writes the unit to stdout; ``--install <host>`` writes it on that host
over SSH (``/etc/systemd/system/<unit>``, ``sudo``) and runs ``systemctl enable --now``,
nothing else. ``stop <host>`` (control) and ``clear-ended <host>`` run one command on the
host as the unit runs it (its user, env file and directories, ``systemd-run``).

The units (:func:`unit_text`):

* ``Restart=on-failure``, but never after an exit yeto chose itself
  (``RestartPreventExitStatus``): both services exit :data:`CONFIG_EXIT` on a configuration
  error; the control service exits :data:`CONTROL_DECIDED_EXIT` when ``run up`` ended the run
  after a breaker trip or a failure it handled (its exit 1); the syncer service exits 2 on a
  refusal (another service holds the lock, a checkpoint that is not this run's, or none where
  the record says there is one), 9 on the fleet's STOP and 11 on a strict failure. A crash
  (an uncaught exception: :data:`CRASH_EXIT`, a signal) is restarted. The control unit has
  no start limit (``StartLimitIntervalSec=0``): its other refusals (exit 2) come before
  anything is ordered and are retried every :data:`RESTART_SEC`. The syncer unit's restarts
  are capped (``StartLimitBurst`` within ``StartLimitIntervalSec``). A stop of the unit
  (SIGTERM, exit 143) is a success (``SuccessExitStatus``).
* An **ended marker**: a service that ended its run (:data:`ENDED`) writes one to its local
  state directory and to the run store (``fleet/onsite/<role>.ended``: exit code, reason,
  time). A later start (a reboot of the host, the unit started on another host) reads it
  and exits 0 at once: nothing is ordered again. ``clear-ended`` removes it.
* Credentials only through ``EnvironmentFile=`` (group and others without any access;
  ``ExecStartPre`` refuses the start otherwise). The unit names the file, never a value;
  ``--install`` checks the file's mode on the host (``sudo stat``) and never reads it. The
  island HMAC key (``YETO_ISLAND_HMAC_KEY``) belongs in that file: both units must read
  the same key.
* The working directory, run up's state (``YETO_STATE_DIR``) and the launch registry
  (``YETO_RUNS_DIR``) are under the host's scratch directory (``--workdir``).
* The control unit kills only ``run up`` itself on a stop or restart (``KillMode=process``),
  and runs it with ``rollout.drain_on_exit`` false: the island launches and the sampler
  pools it started keep running, and the next ``run up`` adopts them. The syncer unit stops
  the whole process group: the syncer gets SIGTERM and writes its checkpoint (SIGKILL only
  after :data:`SYNCER_STOP_GRACE_S`, inside ``TimeoutStopSec``), and the restarted service
  resumes from it.

``yeto fleet service-run --role control|syncer --config <run.json>`` is what the units run:
``control`` is ``yeto run up --config`` (:func:`run_control`); ``syncer`` runs the fleet's
syncer on this host (:func:`run_syncer`).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROLES = ("control", "syncer")
UNIT_DIR = "/etc/systemd/system"
DEFAULT_WORKDIR = "/var/tmp/yeto"  # <workdir>/<run>-<role>
CRASH_EXIT = 70  # an uncaught exception in the service process (EX_SOFTWARE): restarted
CONTROL_DECIDED_EXIT = 75  # run up's exit 1: it ended the run (a breaker trip, a failure it handled)
# a configuration error (yeto.config_errors.CONFIG_ERROR_EXIT): never restarted, a start cannot fix it.
# The control unit's other refusals (exit 2: a preflight, another run up of the run) come before
# anything is ordered and cost nothing: restarted every RESTART_SEC without a start limit
SIGTERM_EXIT = 128 + int(signal.SIGTERM)  # a stop of the unit: a success, not restarted
# exits a unit is never restarted after: yeto decided them
CONFIG_EXIT = 10
CONTROL_PREVENT = (CONFIG_EXIT, CONTROL_DECIDED_EXIT)
SYNCER_PREVENT = (2, 9, 10, 11)  # a refusal, the fleet's STOP, a configuration error, a strict failure
RESTART_SEC = 30
START_LIMIT_INTERVAL_S = 3600
START_LIMIT_BURST = 5
STOP_TIMEOUT_S = {"control": 300, "syncer": 180}  # run up's exit / the syncer's checkpoint
SYNCER_STOP_GRACE_S = STOP_TIMEOUT_S["syncer"] - 10  # SIGTERM to SIGKILL of the syncer, inside TimeoutStopSec
# the exits after which a service's run is over (its ended marker): role -> exit -> reason
ENDED = {
    "control": {0: "run up ended the run",
                CONTROL_DECIDED_EXIT: "run up ended the run after a breaker trip or a failure it handled (exit 1)"},
    "syncer": {0: "the run is finished", 9: "stopped by the fleet's STOP", 11: "a strict failure of the syncer"},
}
ENDED_SCHEMA = "yeto.fleet.onsite-ended/v1"
ENDED_READ_FIRST_S, ENDED_READ_MAX_S = 5.0, 60.0  # a store that does not answer at the start: asked again
_sleep = time.sleep  # the ended marker's read retries (tests use a fake)
# what a unit file line may hold: no spaces, quotes, newlines, specifiers (%) or variables ($)
_SAFE_RUN = re.compile(r"[A-Za-z0-9_.][A-Za-z0-9_.-]*")  # never an option ("-..."), see also "." / ".."
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
    env_file: str  # the environment file on the service host (no access for group and others)
    workdir: str  # the service's scratch directory on its host
    user: str
    python: str  # the Python that runs yeto on the service host
    yeto_dir: str  # the yeto checkout on the service host (PYTHONPATH)

    @property
    def unit(self) -> str:
        return unit_name(self.run, self.role)

    @property
    def state_dir(self) -> str:
        return f"{self.workdir}/rl-up"

    @property
    def runs_dir(self) -> str:
        return f"{self.workdir}/runs"


def unit_name(run: str, role: str) -> str:
    return f"yeto-{run}-{role}.service"


def check_spec(spec: ServiceSpec) -> None:
    """Every value a unit line (or a remote command) holds comes from the run config or the
    command line: each is checked against what a unit file holds as is, never escaped."""
    if spec.role not in ROLES:
        raise ServiceError(f"--role must be one of {list(ROLES)} (got {spec.role!r})")
    if not isinstance(spec.run, str) or not _SAFE_RUN.fullmatch(spec.run) or spec.run in (".", ".."):
        raise ServiceError(f"run {spec.run!r}: a run name of a unit is letters, digits and _.- only")
    for what, value in (("--config", spec.config), ("--env-file", spec.env_file), ("--workdir", spec.workdir),
                        ("--python", spec.python), ("--yeto-dir", spec.yeto_dir)):
        if not isinstance(value, str) or not _SAFE_PATH.fullmatch(value) or "/../" in f"{value}/":
            raise ServiceError(f"{what} {value!r}: an absolute path on the service host of letters, digits and "
                               "._/+- only (a unit file line holds it as is)")
    if not isinstance(spec.user, str) or not _SAFE_USER.fullmatch(spec.user):
        raise ServiceError(f"--user {spec.user!r} is not a user name")


def unit_text(spec: ServiceSpec) -> str:
    """The systemd unit of ``spec`` (module doc). No value of the environment is in it."""
    check_spec(spec)
    prevent = CONTROL_PREVENT if spec.role == "control" else SYNCER_PREVENT
    what = "control plane (yeto run up)" if spec.role == "control" else "syncer"
    run_args = f"--role {spec.role} --config {spec.config}"
    if spec.role == "syncer":
        run_args += f" --stop-grace-s {SYNCER_STOP_GRACE_S}"
    lines = [
        f"# The fleet {what} of run {spec.run}, written by `yeto fleet service --role {spec.role}`.",
        "# Credentials come from EnvironmentFile only (no access for group and others); this file holds none.",
        "[Unit]",
        f"Description=yeto fleet {what} of run {spec.run}",
        "Wants=network-online.target",
        "After=network-online.target",
        # control: no start limit (its refusals order nothing; a unit the limit failed for good
        # would leave the pools undriven); syncer: a crash loop stops after START_LIMIT_BURST
        *(["StartLimitIntervalSec=0"] if spec.role == "control" else
          [f"StartLimitIntervalSec={START_LIMIT_INTERVAL_S}", f"StartLimitBurst={START_LIMIT_BURST}"]),
        "",
        "[Service]",
        "Type=simple",
        f"User={spec.user}",
        f"EnvironmentFile={spec.env_file}",
        f"Environment=PYTHONPATH={spec.yeto_dir}",
        "Environment=PYTHONUNBUFFERED=1",
        f"Environment=YETO_STATE_DIR={spec.state_dir}",
        f"Environment=YETO_RUNS_DIR={spec.runs_dir}",
        # the env file must give group and others no access: refused before anything starts
        f"ExecStartPre=/bin/sh -c 'case \"$$(stat -c %%a {spec.env_file})\" in *00) ;; *) echo \"{spec.env_file} "
        "must give group and others no access (chmod 600)\" >&2; exit 1;; esac'",
        f"ExecStartPre=/bin/mkdir -p {spec.workdir}",
        f"WorkingDirectory=-{spec.workdir}",
        f"ExecStart={spec.python} -m yeto.cli fleet service-run {run_args}",
        "Restart=on-failure",
        f"RestartSec={RESTART_SEC}",
        f"RestartPreventExitStatus={' '.join(str(c) for c in prevent)}",
        f"SuccessExitStatus={SIGTERM_EXIT}",
        # control: only run up itself; its island launches keep their islands and are adopted again
        f"KillMode={'process' if spec.role == 'control' else 'control-group'}",
        f"TimeoutStopSec={STOP_TIMEOUT_S[spec.role]}",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
        "",
    ]
    return "\n".join(lines)


SUDO_EXIT = 4  # the host's sudo does not run without a password (the scripts need sudo -n)


def _env_file_check(spec: ServiceSpec) -> list[str]:
    """Shell lines that first check that ``sudo -n`` runs at all (exit :data:`SUDO_EXIT`,
    said as such), then the env file's mode on the host (``sudo stat``: the file is the
    service user's; its content is never read): exit 3 unless group and others have no
    access (0600, 0400, ...)."""
    env = shlex.quote(spec.env_file)
    return [
        "sudo -n true 2>/dev/null || { echo 'yeto fleet service: sudo -n does not run on this host (a password "
        f"is asked, or this user may not sudo): the unit needs it to write and start; nothing done' >&2; "
        f"exit {SUDO_EXIT}; }}",
        f"mode=$(sudo -n stat -c %a {env} 2>/dev/null) || {{ echo 'yeto fleet service: no env file "
        f"{spec.env_file} on this host' >&2; exit 3; }}",
        f"case \"$mode\" in *00) ;; *) echo \"yeto fleet service: {spec.env_file} has mode $mode; group and "
        "others must have no access (chmod 600)\" >&2; exit 3;; esac",
    ]


def install_script(spec: ServiceSpec, text: str) -> str:
    """The shell script ``--install`` runs on the service host: check the env file's mode,
    write the unit, ``systemctl enable --now``."""
    unit = f"{UNIT_DIR}/{spec.unit}"
    marker = "YETO_UNIT_EOF"
    if marker in text:
        raise ServiceError("the unit text holds the heredoc marker")
    return "\n".join([
        "set -eu",
        *_env_file_check(spec),
        f"sudo -n tee {shlex.quote(unit)} >/dev/null <<'{marker}'",
        text.rstrip("\n"),
        marker,
        "sudo -n systemctl daemon-reload",
        f"sudo -n systemctl enable --now {shlex.quote(spec.unit)}",
        f"echo 'yeto fleet service: {spec.unit} enabled and started'",
        "",
    ])


def as_unit_script(spec: ServiceSpec, argv: list[str]) -> str:
    """The shell script that runs ``yeto <argv>`` on the service host as its unit runs: its
    user, env file, ``PYTHONPATH``, ``YETO_STATE_DIR``, ``YETO_RUNS_DIR`` and working
    directory (``systemd-run --wait --pipe``)."""
    check_spec(spec)
    command = ["sudo", "-n", "systemd-run", "--quiet", "--wait", "--pipe", "--collect", f"--uid={spec.user}",
               f"--working-directory={spec.workdir}", f"--property=EnvironmentFile={spec.env_file}",
               f"--setenv=PYTHONPATH={spec.yeto_dir}", "--setenv=PYTHONUNBUFFERED=1",
               f"--setenv=YETO_STATE_DIR={spec.state_dir}", f"--setenv=YETO_RUNS_DIR={spec.runs_dir}",
               spec.python, "-m", "yeto.cli", *argv]
    return "\n".join(["set -eu", *_env_file_check(spec), shlex.join(command), ""])


def _ssh(host: str, script: str, *, runner: Callable[..., Any] | None, what: str,
         log: Callable[[str], None]) -> int:
    runner = runner or subprocess.run
    if not _SAFE_HOST.fullmatch(str(host)) or str(host).startswith("-"):
        raise ServiceError(f"{host!r} is not an SSH host")
    done = runner(["ssh", "-o", "BatchMode=yes", str(host), "bash", "-s"], input=script, text=True,
                  capture_output=True, timeout=300)
    for line in str(getattr(done, "stdout", "") or "").splitlines():
        log(line)
    if done.returncode != 0:
        err = str(getattr(done, "stderr", "") or "").strip().splitlines()
        log(f"yeto fleet service: {what} on {host} failed (exit {done.returncode})" + (f": {err[-1]}" if err else ""))
        return 1
    return 0


def install(spec: ServiceSpec, host: str, *, runner: Callable[..., Any] | None = None,
            log: Callable[[str], None] = print) -> int:
    """Write the unit on ``host`` and enable it (``ssh <host> bash -s``)."""
    if not _SAFE_HOST.fullmatch(str(host)) or str(host).startswith("-"):
        raise ServiceError(f"--install {host!r} is not an SSH host")
    script = install_script(spec, unit_text(spec))
    if _ssh(host, script, runner=runner, what=f"install of {spec.unit}", log=log):
        return 1
    log(f"yeto fleet service: {spec.unit} on {host}: `ssh {host} systemctl status {spec.unit}`, "
        f"`ssh {host} journalctl -u {spec.unit}`")
    return 0


def remote(spec: ServiceSpec, host: str, argv: list[str], *, what: str, runner: Callable[..., Any] | None = None,
           log: Callable[[str], None] = print) -> int:
    """``yeto <argv>`` on ``host`` as the unit of ``spec`` runs it (:func:`as_unit_script`)."""
    return _ssh(host, as_unit_script(spec, argv), runner=runner, what=what, log=log)


# -- the CLI ------------------------------------------------------------------------------------

ACTIONS = ("stop", "clear-ended", "fresh")


def add_parser(fsub: Any) -> None:
    p = fsub.add_parser("service", help="the fleet's control plane or syncer as a systemd unit on site",
                        description="--print / --install HOST: the unit. stop HOST (--role control): `yeto "
                        "stop-run` of the fleet on HOST, as the unit runs. clear-ended HOST: remove the ended "
                        "marker of the role's service, so that its unit starts the run again. fresh HOST (--role "
                        "syncer): let the next start of the syncer service begin a fresh syncer at the next term "
                        "when its checkpoint is gone.")
    p.add_argument("action", nargs="?", choices=ACTIONS, default=None,
                   help="stop: the fleet's STOP flag on the control host; clear-ended: the service's ended marker; "
                   "fresh: a fresh syncer at the next term on the syncer service's next start")
    p.add_argument("host", nargs="?", default=None, help="stop / clear-ended / fresh: the service's host (SSH)")
    p.add_argument("--config", required=True, help="the run config (fleet.launch per_island, store control plane)")
    p.add_argument("--role", required=True, choices=ROLES,
                   help="control: yeto run up of the fleet; syncer: the fleet's syncer (fleet.syncer_service)")
    p.add_argument("--env-file", required=True,
                   help="the environment file on the service host (no access for group and others): the run's "
                   "credentials and YETO_ISLAND_HMAC_KEY. The unit names it; no value goes into the unit")
    where = p.add_mutually_exclusive_group()
    where.add_argument("--print", action="store_true", help="write the unit to stdout")
    where.add_argument("--install", metavar="HOST", default=None,
                       help="write the unit on HOST over SSH and run systemctl enable --now there")
    p.add_argument("--workdir", default=None,
                   help=f"the scratch directory on the service host (default {DEFAULT_WORKDIR}/<run>-<role>)")
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
    r.add_argument("--stop-grace-s", type=float, default=SYNCER_STOP_GRACE_S,
                   help="syncer: SIGTERM to SIGKILL of the syncer on a stop (keep it inside the unit's "
                   f"TimeoutStopSec; default {SYNCER_STOP_GRACE_S})")
    r.add_argument("--clear-ended", action="store_true",
                   help="remove this role's ended marker (local and in the run store) and exit")
    r.add_argument("--fresh", action="store_true",
                   help="syncer: let the next start begin a fresh syncer at the next term although the run store's "
                   "record says this host's checkpoint should be here (it is gone), and exit")


def load_onsite(args: Any) -> tuple[dict[str, Any], ServiceSpec]:
    from yeto.rl import up  # noqa: PLC0415

    cfg = up.load_config(args.config)  # the config only: no state directory is made here
    problem = onsite_problem(cfg, args.role)
    if problem:
        raise ServiceError(problem)
    name = str(cfg["run"])
    import getpass  # noqa: PLC0415

    from yeto.launch import REPO_ROOT  # noqa: PLC0415

    spec = ServiceSpec(run=name, role=args.role,
                       config=args.remote_config or str(Path(args.config).expanduser().resolve()),
                       env_file=str(args.env_file), workdir=args.workdir or f"{DEFAULT_WORKDIR}/{name}-{args.role}",
                       user=args.user or getpass.getuser(), python=args.python or sys.executable,
                       yeto_dir=args.yeto_dir or str(REPO_ROOT))
    check_spec(spec)
    return cfg, spec


def spec_from_args(args: Any) -> ServiceSpec:
    return load_onsite(args)[1]


def onsite_problem(cfg: dict[str, Any], role: str) -> str | None:
    """Why the run config cannot run on site as ``role`` (None: it can)."""
    from yeto.rl import fleet  # noqa: PLC0415
    from yeto.rl.up import rollout_section  # noqa: PLC0415

    plan = fleet.plan_of(cfg)
    if plan is None or not plan.per_island:
        return "an on-site control plane is a fleet launched island by island (fleet.launch per_island)"
    rollout = rollout_section(cfg)
    if rollout.get("control_plane") != "store":
        return "an on-site control plane needs rollout.control_plane \"store\" (its state lives in the run store)"
    if role == "syncer" and not plan.syncer_service:
        return "--role syncer: the run config names no syncer host (fleet.syncer_service)"
    if role == "control" and rollout.get("drain_on_exit") is True:
        return ("--role control: rollout.drain_on_exit is true, but the control unit runs run up with it false: a "
                "stop or restart of the unit (KillMode=process) must leave the sampler pools running for the next run "
                "up to adopt, not drain and release them. Remove the key (the unit sets it false) or set it false")
    return None


def main(args: Any) -> int:
    if args.fleet_command == "service-run":
        return service_run(args)
    try:
        cfg, spec = load_onsite(args)
        if args.action is None:
            if not args.print and not args.install:
                raise ServiceError("give --print or --install HOST (or the action stop / clear-ended / fresh HOST)")
            if args.host is not None:
                raise ServiceError(f"{args.host!r}: a host goes with stop / clear-ended / fresh, or with --install")
            if args.print:
                sys.stdout.write(unit_text(spec))
                return 0
            return install(spec, args.install)
        if args.print or args.install:
            raise ServiceError(f"{args.action} HOST takes neither --print nor --install")
        if not args.host:
            raise ServiceError(f"{args.action}: name the service's host (yeto fleet service {args.action} HOST ...)")
        if args.action == "stop":
            if spec.role != "control":
                raise ServiceError("stop: --role control (the fleet's STOP flag is on the control host; the "
                                   "syncer service ends on the fleet's STOP in the run store)")
            name = trainer_launch_name(cfg)
            if not _SAFE_RUN.fullmatch(name):
                raise ServiceError(f"stop: the fleet's run name {name!r} (its --cluster-prefix) is not letters, digits "
                                   "and _.- only")
            return remote(spec, args.host, ["stop-run", name], what=f"stop-run {name}")
        if args.action == "fresh":
            if spec.role != "syncer":
                raise ServiceError("fresh: --role syncer (a fresh syncer at the next term)")
            return remote(spec, args.host, ["fleet", "service-run", "--role", "syncer", "--config", spec.config,
                                            "--fresh"], what=f"fresh syncer of {spec.unit}")
        return remote(spec, args.host, ["fleet", "service-run", "--role", spec.role, "--config", spec.config,
                                        "--clear-ended"], what=f"clear-ended of {spec.unit}")
    except (ServiceError, ValueError, OSError) as error:
        print(f"yeto fleet service: refused: {error}", file=sys.stderr)
        return 2


def trainer_launch_name(cfg: dict[str, Any]) -> str:
    """The fleet's run name for ``yeto stop-run`` (the syncer launch's ``--cluster-prefix``)."""
    import types  # noqa: PLC0415

    from yeto.rl import up  # noqa: PLC0415

    return str(up.Run.trainer_launch_name(types.SimpleNamespace(cfg=cfg)))


# -- what a unit runs -----------------------------------------------------------------------------

class Onsite:
    """One role's service of a run: its config, environment (the process's, with the run
    config's ``env`` over it, as ``run up`` gives it) and ended marker."""

    def __init__(self, cfg: dict[str, Any], role: str, *, objects: Any = None) -> None:
        self.cfg, self.role, self.run = cfg, role, str(cfg["run"])
        self.env = dict(os.environ)
        self.env.update({str(k): os.path.expanduser(str(v)) for k, v in (cfg.get("env") or {}).items()})
        self._objects = objects

    @classmethod
    def open(cls, config: str, role: str, *, objects: Any = None) -> Onsite:
        from yeto.rl import up  # noqa: PLC0415

        cfg = up.load_config(str(config))
        problem = onsite_problem(cfg, role)
        if problem:
            raise ServiceError(problem)
        if not _SAFE_RUN.fullmatch(str(cfg["run"])) or str(cfg["run"]) in (".", ".."):
            raise ServiceError(f"run {cfg['run']!r}: letters, digits and _.- only, not starting with '-', "
                               "and not . or ..")
        return cls(cfg, role, objects=objects)

    @property
    def objects(self) -> Any:
        if self._objects is None:
            from yeto.cloud.s3_store import head_objects  # noqa: PLC0415

            self._objects = head_objects(self.cfg["store"], environ=self.env)
        return self._objects

    @property
    def ended_path(self) -> Path:
        from yeto.rl import up  # noqa: PLC0415

        return Path(up.STATE_ROOT) / self.run / f"onsite-{self.role}.ended"

    @property
    def ended_key(self) -> str:
        return f"fleet/onsite/{self.role}.ended"

    @property
    def fresh_path(self) -> Path:
        """``service-run --fresh``: the next start of the syncer begins a fresh syncer at the next
        term although this host's checkpoint is gone (:func:`run_syncer`; consumed by it)."""
        return self.ended_path.with_name("onsite-syncer.fresh")

    def read_ended(self, *, sleep: Callable[[float], None] | None = None, log: Callable[[str], None] = print
                   ) -> dict[str, Any] | None:
        """The ended marker: this host's, else the run store's. A store that does not answer
        is asked again in this process for as long as it does not (the store may come up
        after the host, at a boot), backing off from :data:`ENDED_READ_FIRST_S` to
        :data:`ENDED_READ_MAX_S`, one log line per try: nothing is started meanwhile, and the
        service never exits for it (a unit restarted for it would spend its start limit)."""
        if self.ended_path.exists():
            try:
                doc = json.loads(self.ended_path.read_text())
            except (OSError, ValueError):
                doc = None
            return doc if isinstance(doc, dict) else {"reason": f"marker {self.ended_path} (unreadable)"}
        sleep = sleep or _sleep
        delay, tries = ENDED_READ_FIRST_S, 0
        while True:
            try:
                raw = self.objects.read(self.ended_key)
                break
            except Exception as error:  # noqa: BLE001 - asked again until it answers
                tries += 1
                log(f"[{self.role} service] the run store does not answer ({type(error).__name__}: {error}); its ended "
                    f"marker is read again in {delay:.0f}s (try {tries}), nothing is started meanwhile")
                sleep(delay)
                delay = min(delay * 2, ENDED_READ_MAX_S)
        if raw is None:
            return None
        try:
            doc = json.loads(raw)
        except ValueError:
            doc = None
        return doc if isinstance(doc, dict) else {"reason": f"marker {self.ended_key} in the run store (unreadable)"}

    def write_ended(self, code: int, reason: str, log: Callable[[str], None] = print) -> dict[str, Any]:
        doc = {"schema": ENDED_SCHEMA, "run": self.run, "role": self.role, "exit_code": int(code),
               "reason": reason, "at": round(time.time(), 3), "node": socket.gethostname()}
        data = json.dumps(doc, sort_keys=True)
        path = self.ended_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(data + "\n")
        os.replace(tmp, path)
        try:
            self.objects.write(self.ended_key, data.encode())
        except Exception as error:  # noqa: BLE001 - this host's marker still holds; said
            log(f"[{self.role} service] WARNING: the ended marker is not in the run store ({type(error).__name__}: "
                f"{error}); only this host's ({path}) keeps the run from starting again")
        log(f"[{self.role} service] run {self.run} ended (exit {code}: {reason}): ended marker {path} and "
            f"{self.ended_key} in the run store; the unit does not start it again until `clear-ended`")
        return doc

    def clear_ended(self, log: Callable[[str], None] = print) -> int:
        from yeto.launch.fleet_syncer import ONSITE_STOP  # noqa: PLC0415

        removed = []
        if self.ended_path.exists():
            self.ended_path.unlink()
            removed.append(str(self.ended_path))
        keys = [self.ended_key] + ([ONSITE_STOP] if self.role == "syncer" else [])
        for key in keys:
            if self.objects.read(key) is not None:
                self.objects.delete(key)
                removed.append(f"{key} (run store)")
        log(f"[{self.role} service] run {self.run}: " + (f"removed {', '.join(removed)}" if removed else
                                                          "no ended marker") + "; the unit starts the run again")
        return 0


def service_run(args: Any, *, objects: Any = None, log: Callable[[str], None] = print) -> int:
    signal.signal(signal.SIGTERM, _exit_on_sigterm)  # the exit path runs (the syncer's stop)
    try:
        try:
            onsite = Onsite.open(args.config, args.role, objects=objects)
        except (ServiceError, ValueError) as error:
            log(f"[{args.role} service] refused: {error} (a configuration error: exit {CONFIG_EXIT}, not restarted)")
            return CONFIG_EXIT
        if getattr(args, "clear_ended", False):
            return onsite.clear_ended(log)
        if getattr(args, "fresh", False):
            if args.role != "syncer":
                log(f"[{args.role} service] --fresh: the syncer role only")
                return CONFIG_EXIT
            onsite.fresh_path.parent.mkdir(parents=True, exist_ok=True)
            onsite.fresh_path.write_text(json.dumps({"at": round(time.time(), 3)}) + "\n")
            log(f"[syncer service] run {onsite.run}: its next start begins a fresh syncer at the next term (the "
                f"islands rejoin it) if its checkpoint is gone ({onsite.fresh_path}); start the unit")
            return 0
        ended = onsite.read_ended(log=log)
        if ended is not None:
            at = ended.get("at")
            when = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(at))) if isinstance(at, (int, float)) else "?"
            log(f"[{args.role} service] run {onsite.run} already ended (exit {ended.get('exit_code')}: "
                f"{ended.get('reason')}, at {when} on {ended.get('node')}): not started again. To start it again, "
                f"remove the marker: `yeto fleet service clear-ended <this host> --role {args.role} --config "
                f"<run.json> --env-file <env file>` (or here, as the unit runs: `yeto fleet service-run --role "
                f"{args.role} --config {args.config} --clear-ended`)")
            return 0
        if args.role == "control":
            code = run_control(args.config, log=log)
        else:
            code = run_syncer(args.config, binary=args.syncer_binary, log=log,
                              stop_grace_s=getattr(args, "stop_grace_s", SYNCER_STOP_GRACE_S),
                              fresh_path=onsite.fresh_path)
        reason = ENDED[args.role].get(code)
        if reason is not None:
            onsite.write_ended(code, reason, log)
        return code
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 - a crash: the unit restarts it
        import traceback  # noqa: PLC0415

        traceback.print_exc()
        return CRASH_EXIT


def _exit_on_sigterm(signum: int, frame: Any) -> None:  # noqa: ARG001
    raise SystemExit(128 + signum)


def no_exit_drain(cfg: dict[str, Any], log: Callable[[str], None] = print) -> None:
    """The control unit's ``run up`` never drains the pools when it exits
    (``rollout.drain_on_exit`` false): set when the config leaves it out, refused when true."""
    section = cfg.get("rollout")
    if section is None:
        section = cfg["rollout"] = {}
    if section.get("drain_on_exit") is True:
        raise ValueError(onsite_problem(cfg, "control") or "rollout.drain_on_exit true")
    if "drain_on_exit" not in section:
        section["drain_on_exit"] = False
        log("[control service] rollout.drain_on_exit is not set: run up runs with it false here (a stop or restart "
            "of the unit leaves the sampler pools running; the next run up adopts them)")


def run_control(config: str, *, log: Callable[[str], None] = print) -> int:
    """``yeto run up --config <config>`` in this process, with ``rollout.drain_on_exit``
    false (:func:`no_exit_drain`) in service mode: a SIGTERM or a crash of the unit leaves
    the pools running for the next run up, and an end run up decides itself (the run
    finished, a stop, a breaker) drains them as ever. Its exit 1 is :data:`CONTROL_DECIDED_EXIT`."""
    from yeto import cli  # noqa: PLC0415
    from yeto.rl import up  # noqa: PLC0415

    args = cli.build_parser().parse_args(["run", "up", "--config", str(config)])
    args.config_adjust = lambda cfg: no_exit_drain(cfg, log)
    args.onsite_service = True  # run up drains when it ends the run itself; not on a SIGTERM or a crash
    code = int(up.main(args) or 0)
    if code == 2 and getattr(args, "config_refused", None):
        log(f"[control service] run up refused its configuration: exit {CONFIG_EXIT}, not restarted")
        return CONFIG_EXIT
    if code == 1:
        log(f"[control service] run up exited 1 (it ended the run): exit {CONTROL_DECIDED_EXIT}, not restarted")
        return CONTROL_DECIDED_EXIT
    return code


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


def _checkpoints(args: Any, home: str) -> list[str]:
    """The syncer's checkpoint files (every trainable role's), ``~`` under ``home``."""
    from yeto.launch.syncer import rl_role_syncers  # noqa: PLC0415

    try:
        paths = [s.checkpoint for s in rl_role_syncers(args)]
    except Exception:  # noqa: BLE001 - a launch without RL roles: the actor's
        from yeto.launch.syncer import RL_SYNCER_CHECKPOINT  # noqa: PLC0415

        paths = [RL_SYNCER_CHECKPOINT]
    return [home + p[1:] if p.startswith("~/") else p for p in paths]


LOCK_LOST_EXIT = 2  # another service took the lock: not restarted (a restart would be refused anyway)
SELF_FENCED_EXIT = 1  # this service could not renew its own lease: restarted (acquire refuses a lock held by another)


def service_lock_ttl_s(stop_grace_s: float | None, beat_s: float | None = None) -> float:
    """The service lock's lease: long enough that a service whose renewals fail fences itself
    (:meth:`Heartbeat.renew`) and still has its syncer stopped (its checkpoint written within
    ``stop_grace_s``) before the lease ends, after at least two renewals that failed."""
    from yeto.launch import fleet_syncer  # noqa: PLC0415

    beat = fleet_syncer.SERVICE_BEAT_S if beat_s is None else float(beat_s)
    return max(fleet_syncer.SERVICE_LOCK_TTL_S, float(stop_grace_s or 0.0) + fleet_syncer.POLL_S + 3 * beat + 60.0)


class Heartbeat:
    """The service's heartbeat in the run store every :data:`SERVICE_BEAT_S` (a thread),
    which also renews the service lock. ``starting`` until :meth:`set` says otherwise;
    ``lost``: why the lock is no longer this service's (the beats stop), ``lost_code`` the
    service's exit for it."""

    def __init__(self, objects: Any, host: str, key: str | None, *, lock: Any = None, generation: int | None = None,
                 node: str | None = None, every_s: float | None = None, log: Callable[[str], None] = print,
                 stop_grace_s: float | None = None) -> None:
        from yeto.launch.fleet_syncer import POLL_S, SERVICE_BEAT_S, SERVICE_STARTING  # noqa: PLC0415

        self.objects, self.host, self.key, self.lock, self.log = objects, host, key, lock, log
        self.generation, self.node = generation, node
        self.state = SERVICE_STARTING
        self.lost: str | None = None
        self.lost_code: int | None = None
        # the lease this service must still hold when it fences itself: a beat until the next
        # renewal, a poll until serve sees it, the syncer's stop, and the lock's margin
        self.poll_s, self.stop_grace_s = POLL_S, float(stop_grace_s or 0.0)
        self.every_s = SERVICE_BEAT_S if every_s is None else float(every_s)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="syncer-service-beat", daemon=True)

    def beat(self, state: str | None = None, reason: str | None = None) -> None:
        from yeto.launch.fleet_syncer import write_service_status  # noqa: PLC0415

        if self.lost is not None:
            return
        try:
            write_service_status(self.objects, host=self.host, state=state or self.state, key=self.key,
                                 reason=reason, generation=self.generation, node=self.node)
        except Exception as error:  # noqa: BLE001 - tried again at the next beat
            self.log(f"[syncer service] WARNING: heartbeat not written ({type(error).__name__}: {error})")

    def set(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.beat()

    @property
    def fence_s(self) -> float:
        """Lease left at or below which a service whose renewal failed fences itself."""
        return float(self.lock.margin_s) + self.every_s + self.poll_s + self.stop_grace_s

    def renew(self) -> None:
        """Renew the service lock. Another holder's token (``LockLost``): lost, exit
        :data:`LOCK_LOST_EXIT`. A renewal that fails (the store does not answer): tried again
        at the next beat, unless the lease left is within :attr:`fence_s`; then the service
        fences itself (exit :data:`SELF_FENCED_EXIT`, restarted) while its syncer can still be
        stopped before the lease ends."""
        from yeto.launch.pool.store_lock import LockLost  # noqa: PLC0415

        if self.lock is None or self.lost is not None:
            return
        try:
            self.lock.renew()
            return
        except LockLost as error:
            self.lost, self.lost_code = f"the syncer service lock is no longer this service's ({error})", LOCK_LOST_EXIT
            self.log(f"[syncer service] ERROR: {self.lost}: ending this syncer")
            return
        except Exception as error:  # noqa: BLE001 - a store blip: renewed at the next beat
            self.log(f"[syncer service] WARNING: lock not renewed ({type(error).__name__}: {error})")
        left = self.lock.lease_left()
        if left <= self.fence_s:  # stopped now, the syncer is down before another service may take the lock
            self.lost = (f"the syncer service lock was not renewed and its lease ends in {left:.0f}s (the syncer's "
                         f"stop needs up to {self.fence_s:.0f}s of it): fenced by itself")
            self.lost_code = SELF_FENCED_EXIT
            self.log(f"[syncer service] ERROR: {self.lost}: ending this syncer (exit {SELF_FENCED_EXIT}: restarted)")

    def start(self) -> None:
        self.beat()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self.every_s):
            self.renew()
            self.beat()


def run_syncer(config: str, *, binary: str | None = None, objects: Any = None,
               make_syncer: Callable[..., Any] | None = None, serve: Callable[..., int] | None = None,
               parse: Callable[[list[str]], Any] | None = None, node: str | None = None,
               stop_grace_s: float | None = SYNCER_STOP_GRACE_S, beat_s: float | None = None,
               workdir: str | None = None, log: Callable[[str], None] = print,
               fresh_path: Path | None = None) -> int:
    """The fleet's syncer on this host (``fleet.syncer_service``): the syncer launch's syncer,
    run as the head runs it (``LocalSyncer``: a process that dies is restarted from its
    checkpoint, and its record written again), under the service lock, plus the service's
    heartbeat. 0: the run is finished; 9: the fleet's STOP; 11: a strict failure (never
    restarted); 2 / 10: a refusal (another service holds the lock, a configuration error);
    1: anything else (the unit restarts the service, which resumes from the checkpoint).

    The environment is ``run up``'s: this process's (the unit's env file) with the run
    config's ``env`` over it, passed to the store client and the syncer (never written to
    this process's environment). The term (``syncer_epoch``): the fleet's last record's
    when that record is this host's service's (a restart resumes it), one more when it is
    another host's (a syncer that starts over elsewhere fences the old one's islands).

    The checkpoint must match the record: one here while the record is missing or another
    run's is refused (exit 2), and so is none here while the record says this host's service
    ran this run's syncer (its workdir was cleared: a syncer from step 0 under the islands'
    term), unless ``fresh_path`` exists (``service-run --fresh``): then a fresh syncer starts
    at the next term, which the islands rejoin."""
    from yeto.config_errors import CONFIG_ERROR_EXIT, ISLAND_STOP_EXIT  # noqa: PLC0415
    from yeto.gpu_spec import parse_gpu_spec  # noqa: PLC0415
    from yeto.launch import fleet_syncer  # noqa: PLC0415
    from yeto.launch.fleet import RUN_STOPPED_EXIT, _RlStrictFailure  # noqa: PLC0415
    from yeto.launch.island_lifecycle import store_objects  # noqa: PLC0415
    from yeto.launch.pool.store_lock import StoreLock, StoreLockError  # noqa: PLC0415
    from yeto.launch.storage import store_client_environ  # noqa: PLC0415
    from yeto.launch.syncer import ISLAND_HMAC_KEY_ENV, syncer_shard_addrs  # noqa: PLC0415
    from yeto.rl import up  # noqa: PLC0415
    from yeto.rl.engine import sample_roles  # noqa: PLC0415
    from yeto.rl.island_launches import read_syncer_record  # noqa: PLC0415

    try:
        run = up.open_run(config)
    except up.RunConfigError as error:
        log(f"[syncer service] refused: {error}")
        return CONFIG_EXIT
    problem = onsite_problem(run.cfg, "syncer")
    if problem:
        log(f"[syncer service] refused: {problem}")
        return CONFIG_EXIT
    env = dict(run.env)  # run up's: this process's environment, the run config's env over it
    key = env.get(ISLAND_HMAC_KEY_ENV)
    kid = fleet_syncer.key_id(key)
    refused = fleet_syncer.key_mismatch(kid, None)
    if refused:
        log(f"[syncer service] refused: {refused}")
        return CONFIG_ERROR_EXIT
    argv, host = syncer_launch_argv(run)
    args = (parse or launch_args)(argv)
    if objects is None:
        objects = store_objects(getattr(args, "rl_checkpoint_store", None), store_client_environ(args, env))
    if objects is None:
        log(f"[syncer service] refused: the run store {args.rl_checkpoint_store!r} has no object API here")
        return 2
    node = node or socket.gethostname()
    # the syncer's checkpoint and tape (~/yeto-output/...) under this service's working directory: a
    # run of its own (each run has its own workdir), never another run's on this host
    workdir = str(workdir or Path.cwd())  # the unit's WorkingDirectory
    home = str(Path(workdir) / "home")
    env["HOME"] = home
    checkpoints = [c for c in _checkpoints(args, home) if Path(c).exists()]
    last = read_syncer_record(objects)
    if checkpoints and (last is None or last.get("run") != args.cluster_prefix or last.get("service") != host):
        log(f"[syncer service] refused: a syncer checkpoint is here ({checkpoints[0]}), but the run store's syncer "
            f"record is {'missing' if last is None else 'of run ' + repr(last.get('run')) + ' on ' + repr(last.get('service'))}"
            f", not run {args.cluster_prefix!r} on {host}: it is another run's state (or this store is new). Move it "
            "away, or give this run its own --workdir")
        return 2
    mine = (last is not None and last.get("run") == args.cluster_prefix and last.get("service") == host
            and last.get("service_node") == node)
    fresh = fresh_path is not None and Path(fresh_path).exists()
    if mine and not checkpoints and not fresh:
        paths = ", ".join(_checkpoints(args, home))
        log(f"[syncer service] refused: the run store's syncer record says this host's service ran run "
            f"{args.cluster_prefix!r}'s syncer (epoch {last.get('syncer_epoch')}), but its checkpoint is not here "
            f"({paths}): the workdir was cleared? A syncer started now would begin from step 0 under the islands' "
            "term. To start a fresh syncer at the next term (the islands rejoin it): `yeto fleet service fresh "
            f"<this host> --role syncer --config <run.json> --env-file <env file>` (or here, as the unit runs: `yeto "
            "fleet service-run --role syncer --config <run.json> --fresh`), then start the unit. Or put the "
            "checkpoint back")
        return 2
    holder = f"{node}/{unit_name(run.run, 'syncer')}"
    try:
        lock = StoreLock.acquire(objects, fleet_syncer.SERVICE_LOCK, holder=holder,
                                 ttl_s=service_lock_ttl_s(stop_grace_s, beat_s), reclaim=True)
    except StoreLockError as error:
        log(f"[syncer service] refused: the run store cannot hold the service lock ({error})")
        return 2
    if lock is None:
        other = StoreLock.holder_of(objects, fleet_syncer.SERVICE_LOCK) or {}
        log(f"[syncer service] refused: another syncer service of run {run.run} runs ({other.get('holder')}, its "
            f"lease {fleet_syncer.SERVICE_LOCK} runs out at {float(other.get('until') or 0):.0f}); one syncer at a "
            "time. Stop that unit first (or wait for its lease to run out if its host is gone)")
        return 2
    beat = Heartbeat(objects, host, kid, lock=lock, generation=lock.token, node=node, every_s=beat_s, log=log,
                     stop_grace_s=stop_grace_s)
    syncer, code, reason, final = None, 1, None, None
    try:
        stop = fleet_syncer.read_onsite_stop(objects)
        if stop is not None:
            reason = f"the fleet's STOP in the run store ({stop.get('why')}, by {stop.get('by')})"
            log(f"[syncer service] {reason}: the syncer is not started; ended: exit {RUN_STOPPED_EXIT}")
            code, final = RUN_STOPPED_EXIT, fleet_syncer.SERVICE_STOPPED
            return code
        num_learners = (sample_roles.syncer_learner_count(args, len(parse_gpu_spec(args.gpu)))
                        + (getattr(args, "external_learners", 0) or 0))
        here = last is not None and last.get("service") == host and last.get("service_node") == node
        if fresh and mine and not checkpoints:  # service-run --fresh: a new term, the islands rejoin it
            here = False
            Path(fresh_path).unlink(missing_ok=True)
        if last is not None and not here and last.get("syncer_epoch") is not None:
            args.rl_syncer_epoch = int(last["syncer_epoch"]) + 1
            log(f"[syncer service] the fleet's last syncer record (epoch {last['syncer_epoch']}) is "
                + ("this host's, but its checkpoint is gone (--fresh)" if mine else "not this host's service's")
                + f": this syncer starts at epoch {args.rl_syncer_epoch}")
        fleet_syncer.resolve_epoch(args, sample_roles.FLEET_ROLE_SYNCER, adopted=False, reader=lambda: None,
                                   objects=objects, log=log)
        addr = syncer_shard_addrs(host, args)
        if make_syncer is None:
            from yeto.launch.local_syncer import LocalSyncer  # noqa: PLC0415
            from yeto.launch.syncer import build_syncer_binary  # noqa: PLC0415

            path = binary or str(build_syncer_binary())
            syncer = LocalSyncer(args, num_learners, binary=path, log_file=str(Path(workdir) / "yeto-syncer.log"),
                                 env=env, stop_grace_s=stop_grace_s, home=home)
        else:
            syncer = make_syncer(args, num_learners)
        beat.start()  # starting: no island dials it until its record is written
        syncer.start()
        for hook in ("start_log_forwarder", "start_store_upload"):
            method = getattr(syncer, hook, None)
            if method is not None:  # the tape upload: this service's store client, run up's environment
                method(args, objects) if hook == "start_store_upload" else method()
        log(f"[syncer service] the fleet syncer of run {run.run} runs on this host at {addr} (epoch "
            f"{args.rl_syncer_epoch}, generation {lock.token}, {num_learners} launched learner(s))")

        def recorded(record: dict[str, Any]) -> None:
            if record.get("service_generation") == lock.token:
                beat.set(fleet_syncer.SERVICE_RUNNING)

        def stopped() -> str | None:
            if beat.lost is not None:
                return beat.lost
            try:
                doc = fleet_syncer.read_onsite_stop(objects)
            except Exception:  # noqa: BLE001 - read again at the next poll
                return None
            return None if doc is None else f"the fleet's STOP in the run store ({doc.get('why')}, by {doc.get('by')})"

        code = (serve or fleet_syncer.serve)(
            args, addr, num_learners, clusters=[], head_mode=True, local_syncer=syncer,
            record_extra={"service": host, "service_node": node, "service_generation": lock.token, "key_id": kid},
            on_record=recorded, stopped=stopped, early_finish=True)
        if beat.lost is not None:
            code, reason = beat.lost_code or LOCK_LOST_EXIT, beat.lost
        elif code == RUN_STOPPED_EXIT:
            final, reason = fleet_syncer.SERVICE_STOPPED, stopped() or "STOP flag"
        elif code == 0:
            final = fleet_syncer.SERVICE_FINISHED
        else:
            probe = syncer.probe()
            if isinstance(probe, _RlStrictFailure):
                reason = str(getattr(probe, "reason", probe))
                code, final = ISLAND_STOP_EXIT, fleet_syncer.SERVICE_FAILED
    finally:
        beat.stop()
        if final is not None:
            beat.beat(final, reason)
        if syncer is not None:
            syncer.stop()
        if beat.lost_code != LOCK_LOST_EXIT:  # another holder's lease is not ours to end
            lock.release()  # the next service (a restart, another host) need not wait for the lease
    log(f"[syncer service] ended: exit {code}" + (f" ({reason})" if reason else ""))
    return code
