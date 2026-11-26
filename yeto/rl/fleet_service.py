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
import queue
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
        self._built = False  # the store client is this object's own (built from the environment)

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
            self._built = True
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

    def store_problem(self) -> str | None:
        """:func:`store_credential_problem` of this run's store (none with a store given)."""
        if self._objects is not None:
            return None
        return store_credential_problem(self.cfg.get("store"), self.env)

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
                if _credential_error(error, self.env):  # no store will answer this host: not one that is down
                    raise ServiceError(f"the run store refuses this host's credentials ({type(error).__name__}: "
                                       f"{error})") from error
                no_credentials = type(error).__name__ == "NoCredentialsError"
                if no_credentials and self._built:
                    self._objects = None  # a dynamic source may give one later: the next try's client asks again
                tries += 1
                if no_credentials:  # a dynamic source (the instance metadata, a container endpoint) gave none
                    sources = ", ".join(credential_sources(self.env)["dynamic"]) or "none"
                    log(f"[{self.role} service] no credential for the run store yet (NoCredentialsError; dynamic "
                        f"sources: {sources}): check this host's store credentials (the env file, ~/.aws, or "
                        "AWS_EC2_METADATA_DISABLED=true where there is no instance metadata); its ended marker is "
                        f"read again in {delay:.0f}s (try {tries}), nothing is started meanwhile")
                else:
                    log(f"[{self.role} service] the run store does not answer ({type(error).__name__}: {error}); its "
                        f"ended marker is read again in {delay:.0f}s (try {tries}), nothing is started meanwhile")
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
        """The ended marker, on this host first, then in the run store through a store client
        of its own with the heartbeat's short timeouts, the call cut off after
        :data:`~yeto.launch.fleet_syncer.HEARTBEAT_CALL_S` (the unit's stop budget). Neither
        write raises: a failure is said, and the service's exit stays its own."""
        from yeto.launch.fleet_syncer import HEARTBEAT_CALL_S  # noqa: PLC0415

        doc = {"schema": ENDED_SCHEMA, "run": self.run, "role": self.role, "exit_code": int(code),
               "reason": reason, "at": round(time.time(), 3), "node": socket.gethostname()}
        data = json.dumps(doc, sort_keys=True)
        path = self.ended_path
        stored = here = True
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(data + "\n")
            os.replace(tmp, path)
        except Exception as error:  # noqa: BLE001 - the run store's marker then holds alone; said
            here = False
            log(f"[{self.role} service] WARNING: this host's ended marker {path} is not written "
                f"({type(error).__name__}: {error}); the run store's keeps the run from starting again")
        try:
            objects = self._objects
            if objects is None:
                objects = heartbeat_objects(self.cfg.get("store"), self.env) or self.objects
            CallChannel(HEARTBEAT_CALL_S).call("ended marker", lambda: objects.write(self.ended_key, data.encode()))
        except Exception as error:  # noqa: BLE001 - this host's marker then holds alone; said
            stored = False
            log(f"[{self.role} service] WARNING: the ended marker is not in the run store ({type(error).__name__}: "
                f"{error})" + (f"; only this host's ({path}) keeps the run from starting again" if here
                               else "; NOTHING keeps the run from starting again"))
        where = " and ".join(([str(path)] if here else []) + ([f"{self.ended_key} in the run store"] if stored else []))
        log(f"[{self.role} service] run {self.run} ended (exit {code}: {reason}): ended marker "
            f"{where or 'not written'}; the unit does not start it again until `clear-ended`")
        return doc

    def mark_fresh(self, log: Callable[[str], None] = print) -> int:
        """``service-run --fresh``: the marker of :attr:`fresh_path`, only on the host whose service
        ran this run's syncer by the run store's syncer record (``service_node``); anywhere else,
        or without such a record, refused (exit 2), and no marker is written."""
        from yeto.rl.island_launches import read_syncer_record  # noqa: PLC0415

        node, name = socket.gethostname(), trainer_launch_name(self.cfg)
        try:
            last = read_syncer_record(self.objects)
        except Exception as error:  # noqa: BLE001 - nothing is marked
            log(f"[syncer service] --fresh refused: the run store's syncer record does not read "
                f"({type(error).__name__}: {error}); nothing marked")
            return 2
        if last is None or last.get("run") != name:
            log(f"[syncer service] --fresh refused: the run store has no syncer record of run {name!r}"
                + ("" if last is None else f" (it is run {last.get('run')!r}'s)")
                + ": the syncer starts from the record as it is, nothing to start fresh; nothing marked")
            return 2
        if last.get("service_node") != node:
            log(f"[syncer service] --fresh refused: the run store's syncer record is the service's on node "
                f"{last.get('service_node')!r}, not this one ({node!r}): run it there (`yeto fleet service fresh "
                f"<that host> ...`); nothing marked")
            return 2
        self.fresh_path.parent.mkdir(parents=True, exist_ok=True)
        self.fresh_path.write_text(json.dumps({"at": round(time.time(), 3), "epoch": last.get("syncer_epoch")}) + "\n")
        log(f"[syncer service] run {self.run}: its next start begins the syncer at the next term (after epoch "
            f"{last.get('syncer_epoch')}; the islands rejoin it), marker {self.fresh_path}; start the unit")
        return 0

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


_CREDENTIAL_ERRORS = ("PartialCredentialsError", "ProfileNotFound")


def _credential_error(error: BaseException, environ: Any) -> bool:
    """A store client error that says this host has no usable credential: a configuration
    error. ``NoCredentialsError`` only when no dynamic source (a container endpoint, a web
    identity, the instance metadata) could still give one (those may come up later: asked
    again, as a store that does not answer); botocore's partial key and missing profile; the
    store's own refusal of a profile that is not configured here."""
    if type(error).__name__ == "NoCredentialsError":
        return not credential_sources(environ)["dynamic"]
    if type(error).__name__ in _CREDENTIAL_ERRORS:
        return True
    from yeto.rl.engine.object_store import ObjectStoreError  # noqa: PLC0415

    return isinstance(error, ObjectStoreError) and ("no store credentials" in str(error) or "AWS profile" in str(error))


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
            return onsite.mark_fresh(log)
        try:
            problem = onsite.store_problem()
            if problem:
                raise ServiceError(problem)
            ended = onsite.read_ended(log=log)
        except ServiceError as error:
            log(f"[{args.role} service] refused: {error} (a configuration error: exit {CONFIG_EXIT}, not restarted)")
            return CONFIG_EXIT
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

def launch_args(argv: list[str], environ: dict[str, str] | None = None) -> Any:
    """``yeto launch <argv>`` parsed and prepared as the launch does (the syncer's profile and
    command are built from these). ``environ``: the environment its store clients read (run
    up's; ``store_env.bind_environ``, never an attribute of the args), never written to this
    process's."""
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
    if environ is not None:
        from yeto.launch.store_env import bind_environ  # noqa: PLC0415

        bind_environ(args, environ)
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


def service_lock_ttl_s(stop_grace_s: float | None, beat_s: float | None = None, *, poll_s: float | None = None,
                       call_s: float | None = None) -> float:
    """The service lock's lease: long enough that a service whose renewals fail (each call
    taking up to ``call_s``, :data:`~yeto.launch.fleet_syncer.HEARTBEAT_CALL_S`) fences itself
    (:meth:`Heartbeat.renew`) and still has its syncer stopped (its checkpoint written within
    ``stop_grace_s``) more than the lock's margin before the lease ends. The lease is
    :attr:`Heartbeat.fence_s` plus two heartbeat rounds as slow as they can be before the
    fence (``2 * (beat + 2 * call)``), the margin included, which is ``min(60, ttl / 10)``
    (``StoreLock.margin_s``); at least :data:`~yeto.launch.fleet_syncer.SERVICE_LOCK_TTL_S`."""
    from yeto.launch import fleet_syncer  # noqa: PLC0415

    beat = fleet_syncer.SERVICE_BEAT_S if beat_s is None else float(beat_s)
    poll = fleet_syncer.POLL_S if poll_s is None else float(poll_s)
    call = fleet_syncer.HEARTBEAT_CALL_S if call_s is None else float(call_s)
    rest = float(stop_grace_s or 0.0) + poll + beat + 3 * call + 2 * (beat + 2 * call)
    return max(fleet_syncer.SERVICE_LOCK_TTL_S, rest + min(60.0, rest / 9))


# the files of a profile that hold, or lead to, a credential (botocore's names)
_PROFILE_CREDENTIAL_KEYS = ("aws_access_key_id", "credential_process", "role_arn", "web_identity_token_file",
                            "sso_start_url", "sso_session", "login_session", "credential_source")


def credential_sources(environ: Any) -> dict[str, list[str]]:
    """Where an S3 client of ``environ`` may find a credential, from the environment and the
    local AWS files only (no request, no instance metadata asked): ``static`` (the store's key
    pair, the ``AWS_*`` keys, a profile's keys, ``credential_process``, a role or SSO in its
    config) and ``dynamic`` (a container credential endpoint, a web identity token, the
    instance metadata unless ``AWS_EC2_METADATA_DISABLED``)."""
    from yeto.cloud import s3_store  # noqa: PLC0415

    static: list[str] = []
    dynamic: list[str] = []
    if s3_store.store_credentials(environ) is not None:
        static.append("the store's key pair")
    if environ.get("AWS_ACCESS_KEY_ID") and environ.get("AWS_SECRET_ACCESS_KEY"):
        static.append("AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY")
    for name in ("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "AWS_CONTAINER_CREDENTIALS_FULL_URI"):
        if environ.get(name):
            dynamic.append(name)
    if environ.get("AWS_WEB_IDENTITY_TOKEN_FILE"):
        dynamic.append("AWS_WEB_IDENTITY_TOKEN_FILE")
    if str(environ.get("AWS_EC2_METADATA_DISABLED") or "").lower() != "true":
        dynamic.append("the instance metadata service")
    profile = s3_store.store_profile(environ) or "default"
    home = environ.get("HOME") or os.path.expanduser("~")

    def path(name: str, default: str) -> str:
        value = environ.get(name) or os.path.join(home, ".aws", default)
        return os.path.expanduser(value.replace("~", home, 1) if value.startswith("~") else value)

    try:
        from botocore import configloader  # noqa: PLC0415

        files = [(path("AWS_SHARED_CREDENTIALS_FILE", "credentials"), profile),
                 (path("AWS_CONFIG_FILE", "config"), profile if profile == "default" else f"profile {profile}")]
        for file, section in files:
            if not os.path.isfile(file):
                continue
            config = configloader.raw_config_parse(file, parse_subsections=False)
            entry = config.get(section) or (config.get(profile) if file == files[1][0] else None) or {}
            keys = [k for k in _PROFILE_CREDENTIAL_KEYS if entry.get(k)]
            if keys:
                static.append(f"profile {profile!r} in {file} ({', '.join(keys)})")
    except Exception as error:  # noqa: BLE001 - a file botocore cannot parse: the read decides
        static.append(f"the AWS files (not parsed here: {type(error).__name__})")
    return {"static": static, "dynamic": dynamic}


def store_credential_problem(uri: str | None, environ: Any) -> str | None:
    """Why a client of the run store ``uri`` can have no credential on this host, decided
    locally (the environment and the AWS files, no request, never the instance metadata):
    the store's AWS profile (``YETO_RL_STORE_S3_PROFILE``) is not configured here, or no
    credential source exists at all, static or dynamic (:func:`credential_sources`); an
    ``oss://`` store without its endpoint or key (and no relay). None otherwise: a source
    that yields nothing shows at the first read, where only a static one is refused."""
    from yeto.cloud import s3_store  # noqa: PLC0415

    text = str(uri or "")
    if text.startswith("oss://"):
        from yeto.cloud import oss_relay, oss_store  # noqa: PLC0415

        if oss_relay.relay_from_env(environ) is not None:
            return None
        try:
            oss_store.island_envs(text, environ)
        except ValueError as error:
            return str(error)
        return None
    if not text.startswith(("s3://", "r2://")):
        return None
    named = environ.get(s3_store.PROFILE_ENV)
    if named and s3_store.store_credentials(environ) is None and s3_store.profile_config(environ).get("_missing"):
        return (f"s3 store: AWS profile {named!r} ({s3_store.PROFILE_ENV}) is not configured on this host "
                "(~/.aws/config, ~/.aws/credentials)")
    sources = credential_sources(environ)
    if not sources["static"] and not sources["dynamic"]:
        return s3_store.missing_credentials_message() + " (and no AWS credential source here at all)"
    return None


def heartbeat_objects(uri: str | None, environ: dict[str, str]) -> Any:
    """The heartbeat's own client of an ``s3://`` / ``r2://`` run store: connect timeout
    :data:`~yeto.launch.fleet_syncer.HEARTBEAT_CONNECT_TIMEOUT_S`, read timeout
    :data:`~yeto.launch.fleet_syncer.HEARTBEAT_READ_TIMEOUT_S`, at most
    :data:`~yeto.launch.fleet_syncer.HEARTBEAT_MAX_ATTEMPTS` requests per call, shared with
    nothing on the main path. None for another store (the service's client is used; the
    heartbeat's own cut-off still bounds each call)."""
    from yeto.cloud import s3_store  # noqa: PLC0415
    from yeto.launch import fleet_syncer  # noqa: PLC0415

    text = str(uri or "")
    if not text.startswith(("s3://", "r2://")):
        return None
    env = {**environ, s3_store.CONNECT_TIMEOUT_ENV: f"{fleet_syncer.HEARTBEAT_CONNECT_TIMEOUT_S:g}",
           s3_store.READ_TIMEOUT_ENV: f"{fleet_syncer.HEARTBEAT_READ_TIMEOUT_S:g}",
           s3_store.MAX_ATTEMPTS_ENV: str(fleet_syncer.HEARTBEAT_MAX_ATTEMPTS),
           s3_store.RETRY_MODE_ENV: "standard"}  # never AWS_RETRY_MODE's: its backoff is the budget's
    return s3_store.head_objects(text.replace("r2://", "s3://", 1), environ=env)


class QueuedLog:
    """A log that never blocks its caller (the heartbeat thread): each line goes through a
    bounded queue to a writer thread of its own. A full queue drops the line and counts it;
    the next line written says how many were dropped. A line the log cannot take (a closed
    pipe) is dropped too."""

    def __init__(self, log: Callable[[str], None], size: int = 256) -> None:
        self._log = log
        self._queue: queue.Queue = queue.Queue(size)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.dropped = 0

    def __call__(self, message: str) -> None:
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="syncer-service-log", daemon=True)
                self._thread.start()
        try:
            self._queue.put_nowait(message)
        except queue.Full:
            with self._lock:
                self.dropped += 1

    def _run(self) -> None:
        while True:
            message = self._queue.get()
            try:
                with self._lock:
                    dropped, self.dropped = self.dropped, 0
                if dropped:
                    self._log(f"[syncer service] WARNING: {dropped} log line(s) dropped (the log did not keep up)")
                self._log(message)
            except Exception:  # noqa: BLE001 - a log that cannot be written changes nothing
                pass
            finally:
                self._queue.task_done()

    def flush(self, timeout_s: float) -> bool:
        """Wait up to ``timeout_s`` for the lines queued so far to be written."""
        deadline = time.monotonic() + max(0.0, timeout_s)
        while self._queue.unfinished_tasks:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.005)
        return True


class CallChannel:
    """Store calls one at a time, each waited for at most ``call_s`` (a thread per call): past
    that a ``TimeoutError``, and the thread ends on its client's own timeouts. A call made while
    the one before still runs fails at once (never two at a time, never queued)."""

    def __init__(self, call_s: float) -> None:
        self.call_s = float(call_s)
        self.pending: threading.Thread | None = None
        self._lock = threading.Lock()

    def busy(self) -> bool:
        pending = self.pending
        return pending is not None and pending.is_alive()

    def call(self, what: str, call: Callable[[], Any], *, wait_s: float = 0.0) -> Any:
        """``call()`` through the channel. ``wait_s``: how long a call still in flight is waited
        for first (0: it fails this one at once)."""
        out: dict[str, Any] = {}

        def run() -> None:
            try:
                out["value"] = call()
            except BaseException as error:  # noqa: BLE001 - handed to the caller
                out["error"] = error

        deadline = time.monotonic() + float(wait_s)
        while True:
            with self._lock:
                pending = self.pending
                if not self.busy():
                    thread = threading.Thread(target=run, name=what, daemon=True)
                    self.pending = thread
                    thread.start()
                    break
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"the {pending.name} call before this one still waits for the store")
            pending.join(left)
        thread.join(self.call_s)
        if thread.is_alive():
            raise TimeoutError(f"the store did not answer the {what} within {self.call_s:.0f}s")
        if "error" in out:
            raise out["error"]
        return out.get("value")

    def drain(self, wait_s: float) -> bool:
        """Wait up to ``wait_s`` for the call in flight; True when none is left."""
        pending = self.pending
        if pending is not None:
            pending.join(wait_s)
        return not self.busy()


class RecordRefused(RuntimeError):
    """The syncer record was not written: the service is fenced, or the record in the run store
    is a newer service's."""


class BoundedStore:
    """An object store whose every call goes through a :class:`CallChannel` (the syncer
    service's main path after it started: its record writes and its STOP reads never hold the
    service past the channel's ``call_s``). With ``token``, the syncer record (which
    ``write_syncer_record`` writes by ``write_exclusive``, always there on this store) is written
    only by :meth:`write_record`: never once ``lost()`` says the service is fenced or its lease
    is short, and only over a record whose ``service_generation`` is at most ``token`` (a
    conditional write)."""

    def __init__(self, objects: Any, channel: CallChannel, *, token: int | None = None,
                 lost: Callable[[], str | None] | None = None) -> None:
        self._objects, self._channel = objects, channel
        self._token, self._lost = token, lost or (lambda: None)

    def write(self, key: str, data: bytes) -> None:
        return self._channel.call("store write", lambda: self._objects.write(key, data))

    def write_exclusive(self, key: str, data: bytes) -> None:
        if self._token is not None and key == _syncer_record_key():
            return self.write_record(data)
        exclusive = getattr(self._objects, "write_exclusive", None)
        call = (lambda: exclusive(key, data)) if exclusive is not None else (lambda: self._objects.write(key, data))
        return self._channel.call("store write", call)

    def write_record(self, data: bytes) -> None:
        """The syncer record, conditionally: read it, refuse a newer service's
        (``service_generation`` above this service's token), and replace it only while it still
        holds the bytes read (``replace_if``: S3 ``If-Match``, a directory under its lock; a
        missing record by an exclusive create). A store with neither is read again right before
        a plain write (the window ``fleet_syncer._rerecord`` documents). Refused once fenced,
        before and inside the call."""
        key = _syncer_record_key()

        def check() -> None:
            why = self._lost()
            if why:
                raise RecordRefused(f"the syncer record is not written: {why}")

        def write() -> None:
            for _ in range(3):
                check()
                raw = self._objects.read(key)
                generation = _record_generation(raw)
                if generation is not None and generation > int(self._token):
                    raise RecordRefused(f"the syncer record is service generation {generation}'s, newer than this "
                                        f"service's ({self._token}): not overwritten")
                check()
                if raw is None:
                    if self._objects.create(key, data):
                        return
                    continue
                replace = getattr(self._objects, "replace_if", None)
                done = replace(key, data, raw) if replace is not None else None
                if done is None:  # no conditional write here: read again right before a plain write
                    if self._objects.read(key) != raw:
                        continue
                    self._objects.write(key, data)
                    return
                if done:
                    return
            raise RecordRefused("the syncer record changed under every try: not written")

        check()
        self._channel.call("syncer record", write)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._objects, name)
        if not callable(attr):
            return attr

        def bounded(*args: Any, **kwargs: Any) -> Any:
            return self._channel.call(f"store {name}", lambda: attr(*args, **kwargs))

        return bounded


def _syncer_record_key() -> str:
    from yeto.rl.island_launches import SYNCER_RECORD  # noqa: PLC0415

    return SYNCER_RECORD


def _record_generation(raw: bytes | None) -> int | None:
    if raw is None:
        return None
    try:
        generation = json.loads(raw).get("service_generation")
        return None if generation is None else int(generation)
    except (ValueError, TypeError, AttributeError):
        return None


class Heartbeat:
    """The service's heartbeat in the run store every :data:`SERVICE_BEAT_S` (a thread),
    which also renews the service lock. ``starting`` until :meth:`set` says otherwise;
    ``lost``: why the lock is no longer this service's (the beats stop), ``lost_code`` the
    service's exit for it. ``objects``: the heartbeat's own store client
    (:func:`heartbeat_objects`); every heartbeat write and lock call goes through one
    :class:`CallChannel` (``call_s``). ``on_fence``: called in the beat thread once the lock
    is lost or the service fenced itself (it stops the syncer there, whatever the main thread
    is waiting for)."""

    def __init__(self, objects: Any, host: str, key: str | None, *, lock: Any = None, generation: int | None = None,
                 node: str | None = None, every_s: float | None = None, log: Callable[[str], None] = print,
                 stop_grace_s: float | None = None, poll_s: float | None = None, call_s: float | None = None,
                 on_fence: Callable[[], None] | None = None) -> None:
        from yeto.launch.fleet_syncer import HEARTBEAT_CALL_S, POLL_S, SERVICE_BEAT_S, SERVICE_STARTING  # noqa: PLC0415

        self.objects, self.host, self.key, self.lock = objects, host, key, lock
        # never blocks the beat thread: a journal pipe that is full or hangs holds no fence up
        self.log = log if isinstance(log, QueuedLog) else QueuedLog(log)
        self.generation, self.node = generation, node
        self.state = SERVICE_STARTING
        self.lost: str | None = None
        self.lost_code: int | None = None
        self.poll_s = POLL_S if poll_s is None else float(poll_s)
        self.call_s = HEARTBEAT_CALL_S if call_s is None else float(call_s)
        self.stop_grace_s = float(stop_grace_s or 0.0)
        self.every_s = SERVICE_BEAT_S if every_s is None else float(every_s)
        self.channel = CallChannel(self.call_s)
        self.final_wait_s = 2 * self.call_s  # how long a final heartbeat waits behind a late call (finish)
        self.on_fence = on_fence
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="syncer-service-beat", daemon=True)

    @property
    def _pending(self) -> threading.Thread | None:
        return self.channel.pending

    def _bounded(self, what: str, call: Callable[[], Any]) -> Any:
        return self.channel.call(what, call)

    def _write(self, state: str | None, reason: str | None) -> None:
        from yeto.launch.fleet_syncer import write_service_status  # noqa: PLC0415

        write_service_status(self.objects, host=self.host, state=state or self.state, key=self.key,
                             reason=reason, generation=self.generation, node=self.node)

    def beat(self, state: str | None = None, reason: str | None = None, *, wait: bool = False) -> None:
        """One heartbeat through the channel. ``wait``: a call in flight (the beat thread's) is
        waited for first, up to :attr:`call_s` (a state change is never dropped for it)."""
        if self.lost is not None:
            return
        try:
            self.channel.call("heartbeat", lambda: self._write(state, reason), wait_s=self.call_s if wait else 0.0)
        except Exception as error:  # noqa: BLE001 - tried again at the next beat
            self._say(f"[syncer service] WARNING: heartbeat not written ({type(error).__name__}: {error})")

    def set(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.beat(wait=True)

    @property
    def fence_s(self) -> float:
        """Lease left before a renewal at or below which a service whose renewal failed fences
        itself: the lock's margin, a poll until serve sees it, the syncer's stop, and the beat
        thread's worst case until it could tell otherwise (this renewal, the heartbeat after it,
        a beat, the next renewal: ``every_s + 3 * call_s``)."""
        return float(self.lock.margin_s) + self.every_s + self.poll_s + self.stop_grace_s + 3 * self.call_s

    def _say(self, message: str) -> None:
        try:
            self.log(message)
        except Exception:  # noqa: BLE001 - a log that cannot be written (a closed pipe) changes nothing
            pass

    def _fenced(self) -> None:
        if self.on_fence is None:
            return
        try:
            self.on_fence()
        except Exception as error:  # noqa: BLE001 - the main thread stops it again at its end
            self._say(f"[syncer service] ERROR: the syncer was not stopped at the fence ({type(error).__name__}: "
                      f"{error})")

    def renew(self) -> None:
        """Renew the service lock (cut off after :attr:`call_s`). Another holder's token
        (``LockLost``): lost, exit :data:`LOCK_LOST_EXIT`. A renewal that fails (the store
        does not answer, or not in time): tried again at the next beat, unless the lease
        left when the renewal began is within :attr:`fence_s`; then the service fences itself
        (exit :data:`SELF_FENCED_EXIT`, restarted). Either way ``on_fence`` stops the syncer
        at once, here, before the lease ends."""
        from yeto.launch.pool.store_lock import LockLost  # noqa: PLC0415

        if self.lock is None or self.lost is not None:
            return
        left = self.lock.lease_left()  # before the call: a call that hangs has spent its time already
        try:
            self._bounded("renewal", self.lock.renew)
            return
        except LockLost as error:
            self.lost, self.lost_code = f"the syncer service lock is no longer this service's ({error})", LOCK_LOST_EXIT
            self._fenced()
            self._say(f"[syncer service] ERROR: {self.lost}: ending this syncer")
            return
        except Exception as error:  # noqa: BLE001 - a store blip: renewed at the next beat
            failed = error
        if left <= self.fence_s:  # stopped now (before any log line), the syncer is down before another service
            # may take the lock
            self.lost = (f"the syncer service lock was not renewed and its lease had {left:.0f}s left before this "
                         f"renewal (the syncer's stop needs up to {self.fence_s:.0f}s of it): fenced by itself")
            self.lost_code = SELF_FENCED_EXIT
            self._fenced()
            self._say(f"[syncer service] WARNING: lock not renewed ({type(failed).__name__}: {failed})")
            self._say(f"[syncer service] ERROR: {self.lost}: ending this syncer (exit {SELF_FENCED_EXIT}: restarted)")
            return
        self._say(f"[syncer service] WARNING: lock not renewed ({type(failed).__name__}: {failed})")

    def start(self) -> None:
        self.beat()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def finish(self, final: str | None, reason: str | None, *, release: bool) -> None:
        """The end of the service (after :meth:`stop`): the beat thread and the call in flight
        are waited for, each up to :attr:`call_s`. With none left, the final heartbeat (if any)
        and then the lock's release (if ``release``), each through the channel. With a call
        still in flight (a late heartbeat or renewal could land after them), the lock is not
        released (its lease runs out), and the final heartbeat is written after that call, if
        it ends within :attr:`final_wait_s` more (``2 * call_s``; a thread the interpreter
        waits for)."""
        if self._thread.is_alive():
            self._thread.join(self.call_s)
        if self.channel.drain(self.call_s):
            if final is not None and self.lost is None:
                self.beat(final, reason, wait=True)
            if release and self.lock is not None:
                try:
                    self._bounded("release", self.lock.release)
                except Exception as error:  # noqa: BLE001 - the lease then runs out on its own
                    self.log(f"[syncer service] WARNING: lock not released ({type(error).__name__}: {error}); its "
                              "lease runs out on its own")
            return
        self.log(f"[syncer service] WARNING: a store call of the heartbeat still runs after {2 * self.call_s:.0f}s: "
                 "the lock is not released (its lease runs out on its own)"
                 + ("; the final heartbeat is written after that call" if final is not None else ""))
        if final is None or self.lost is not None:
            return

        def after() -> None:
            if self.channel.drain(self.final_wait_s):
                self.beat(final, reason, wait=True)

        threading.Thread(target=after, name="syncer-service-final", daemon=False).start()

    def fence_here(self, why: str) -> None:
        """Fence the service now (exit :data:`SELF_FENCED_EXIT`) and stop the syncer: the beat
        thread failed, or the main thread found the lease short."""
        if self.lost is not None:
            return
        self.lost, self.lost_code = why, SELF_FENCED_EXIT
        self._fenced()
        self._say(f"[syncer service] ERROR: {why}: ending this syncer (exit {SELF_FENCED_EXIT}: restarted)")

    def _loop(self) -> None:
        try:
            while not self._stop.wait(self.every_s):
                self.renew()
                if self._stop.is_set():
                    return
                self.beat()
        except BaseException as error:  # noqa: BLE001 - the beat thread never ends without fencing
            self.fence_here(f"the heartbeat thread failed ({type(error).__name__}: {error})")


def run_syncer(config: str, *, binary: str | None = None, objects: Any = None,
               make_syncer: Callable[..., Any] | None = None, serve: Callable[..., int] | None = None,
               parse: Callable[[list[str]], Any] | None = None, node: str | None = None,
               stop_grace_s: float | None = SYNCER_STOP_GRACE_S, beat_s: float | None = None,
               workdir: str | None = None, log: Callable[[str], None] = print,
               fresh_path: Path | None = None, heartbeat_store: Any = None, poll_s: float | None = None,
               call_s: float | None = None) -> int:
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
    term), unless ``fresh_path`` exists (``service-run --fresh``): then the syncer starts at the
    next term, which the islands rejoin. The marker is removed only once this service's record
    at that term is in the run store (a start that fails before it is fresh again).

    ``heartbeat_store``: the heartbeat's store client (default :func:`heartbeat_objects`, or
    ``objects`` when given); ``poll_s`` / ``call_s``: the budget of :class:`Heartbeat`."""
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
    try:
        args = parse(argv) if parse is not None else launch_args(argv, env)
        if objects is None:
            store_env = store_client_environ(args, env)
            uri = getattr(args, "rl_checkpoint_store", None)
            problem = store_credential_problem(uri, store_env)
            if problem:
                raise ValueError(problem)
            objects = store_objects(uri, store_env)
            if objects is not None and heartbeat_store is None:
                heartbeat_store = heartbeat_objects(uri, store_env)
    except ValueError as error:  # the store's credentials, a launch flag: the env file or the config
        log(f"[syncer service] refused: {error} (a configuration error: exit {CONFIG_EXIT}, not restarted)")
        return CONFIG_EXIT
    if objects is None:
        log(f"[syncer service] refused: the run store {args.rl_checkpoint_store!r} has no object API here (a "
            f"configuration error: exit {CONFIG_EXIT}, not restarted)")
        return CONFIG_EXIT
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
    if fresh and not mine:  # a marker the record does not call for: never kept for a later start
        log(f"[syncer service] the --fresh marker {fresh_path} is not for this record (it is not this host's "
            "service's syncer of this run): removed, the syncer starts as the record says")
        Path(fresh_path).unlink(missing_ok=True)
        fresh = False
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
                                 ttl_s=service_lock_ttl_s(stop_grace_s, beat_s, poll_s=poll_s, call_s=call_s),
                                 reclaim=True)
    except StoreLockError as error:
        log(f"[syncer service] refused: the run store cannot hold the service lock ({error})")
        return 2
    if lock is None:
        other = StoreLock.holder_of(objects, fleet_syncer.SERVICE_LOCK) or {}
        log(f"[syncer service] refused: another syncer service of run {run.run} runs ({other.get('holder')}, its "
            f"lease {fleet_syncer.SERVICE_LOCK} runs out at {float(other.get('until') or 0):.0f}); one syncer at a "
            "time. Stop that unit first (or wait for its lease to run out if its host is gone)")
        return 2
    lock.objects = heartbeat_store if heartbeat_store is not None else objects  # renewed by the heartbeat only
    fenced, running = threading.Event(), {"syncer": None}
    beat_log = QueuedLog(log)  # the beat thread's lines: never blocking it (a full journal pipe holds no fence up)

    def halt(target: Any) -> None:
        (getattr(target, "terminate", None) or target.stop)()  # its process group: SIGTERM, SIGKILL after the grace

    def fence() -> None:  # the beat thread (or a lease the main thread found short): never waits for anything
        fenced.set()
        target = running["syncer"]
        if target is not None:
            halt(target)  # before any log line (and the line itself never blocks: queued)
            beat_log("[syncer service] the lock is no longer this service's: the syncer is stopped (SIGTERM, "
                     "SIGKILL after the grace)")

    beat = Heartbeat(lock.objects, host, kid, lock=lock, generation=lock.token, node=node, every_s=beat_s, log=beat_log,
                     stop_grace_s=stop_grace_s, poll_s=poll_s, call_s=call_s, on_fence=fence)
    # the main path from the start of the syncer on (its records, the STOP reads): the heartbeat's
    # client through a channel of its own, so no store call holds it past call_s after a fence
    def lease_floor_s() -> float:
        """The lease the main thread needs left to start, restart or record a syncer: the
        lock's margin, a poll until serve sees a fence, and the syncer's stop (the grace)."""
        return float(lock.margin_s) + beat.poll_s + beat.stop_grace_s

    def lost_now() -> str | None:
        """Why the main thread may no longer start, restart or record the syncer: the lock is
        lost, the service is fenced, or the lease left is at or below :func:`lease_floor_s`
        (the heartbeat is stuck: the main thread fences, and the syncer is stopped now)."""
        if beat.lost is not None:
            return beat.lost
        if fenced.is_set():
            return "the service is fenced"
        left = lock.lease_left()
        if left > lease_floor_s():
            return None
        beat.fence_here(f"the service lock's lease has {left:.0f}s left (the margin, a poll and the syncer's stop "
                        f"need {lease_floor_s():.0f}s) and the heartbeat did not fence: fenced by the main thread")
        return beat.lost or "the service is fenced"

    main_store = BoundedStore(lock.objects, CallChannel(beat.call_s), token=lock.token, lost=lost_now)
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
        if fresh:  # service-run --fresh: a new term, the islands rejoin it (the marker goes once it is recorded)
            here = False
        if last is not None and not here and last.get("syncer_epoch") is not None:
            args.rl_syncer_epoch = int(last["syncer_epoch"]) + 1
            log(f"[syncer service] the fleet's last syncer record (epoch {last['syncer_epoch']}) is "
                + ("this host's, started fresh (--fresh)" if fresh else "not this host's service's")
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
        restart = getattr(syncer, "restart", None)
        if restart is not None or getattr(syncer, "terminate", None) is not None:
            def guarded_restart() -> None:
                """The restart in two steps, the fence checked around each: once fenced, no syncer
                is started again, and one started meanwhile is stopped at once (the fence itself
                never waits for this)."""
                why = lost_now()
                if why:
                    raise RuntimeError(f"the syncer service is fenced ({why}): its syncer is not started again")
                if getattr(syncer, "terminate", None) is None:
                    restart()
                else:
                    syncer.terminate()
                    why = lost_now()
                    if why:
                        raise RuntimeError(f"the syncer service is fenced ({why}): its syncer is not started again")
                    syncer.start()
                why = lost_now()
                if why:  # fenced while it started: the fence may have stopped the old one only
                    halt(syncer)
                    raise RuntimeError(f"the syncer service was fenced while its syncer started ({why}): stopped "
                                       "again")

            syncer.restart = guarded_restart
        running["syncer"] = syncer
        beat.start()  # starting: no island dials it until its record is written
        if lost_now() is None:
            syncer.start()
        if lost_now() is not None:  # fenced before the syncer was up: never left running
            halt(syncer)
        for hook in ("start_log_forwarder", "start_store_upload"):
            method = getattr(syncer, hook, None)
            if method is not None:  # the tape upload: this service's store client, run up's environment
                method(args, objects) if hook == "start_store_upload" else method()
        log(f"[syncer service] the fleet syncer of run {run.run} runs on this host at {addr} (epoch "
            f"{args.rl_syncer_epoch}, generation {lock.token}, {num_learners} launched learner(s))")

        def recorded(record: dict[str, Any]) -> None:
            if record.get("service_generation") == lock.token:
                beat.set(fleet_syncer.SERVICE_RUNNING)
                if fresh and record.get("syncer_epoch") == args.rl_syncer_epoch:  # the new term is in the store
                    Path(fresh_path).unlink(missing_ok=True)

        def stopped() -> str | None:
            why = lost_now()  # the beat thread gone or stuck: the main thread fences before the floor
            if why:
                return why
            try:
                doc = fleet_syncer.read_onsite_stop(main_store)
            except Exception:  # noqa: BLE001 - read again at the next poll (cut off after call_s)
                return None
            return None if doc is None else f"the fleet's STOP in the run store ({doc.get('why')}, by {doc.get('by')})"

        code = (serve or fleet_syncer.serve)(
            args, addr, num_learners, clusters=[], head_mode=True, local_syncer=syncer,
            record_extra={"service": host, "service_node": node, "service_generation": lock.token, "key_id": kid},
            on_record=recorded, stopped=stopped, early_finish=True, objects=main_store)
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
        if syncer is not None:
            syncer.stop()
        # the final heartbeat, then the release (the next service need not wait for the lease), after
        # the calls in flight; another holder's lease is not ours to end
        beat.finish(final, reason, release=beat.lost_code != LOCK_LOST_EXIT)
        beat_log.flush(beat.call_s)  # the beat thread's lines before the end's (bounded: a log that hangs)
    log(f"[syncer service] ended: exit {code}" + (f" ({reason})" if reason else ""))
    return code
