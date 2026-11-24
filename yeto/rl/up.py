"""``yeto run up / status / down``: one run config, one command for an async GRPO run.

A thin layer over the existing commands; each step is the command a customer would
type, run as a subprocess with the run config's environment:

1. preflight   ``nc pool --dry-run`` (the sampler spec through the sampler's own checks,
               ``agentic.setup``), ``yeto sandbox status``, ``yeto launch ... --dry-run``,
               then every ``preflight.commands`` entry (e.g. the trainer's Miles argument
               check in its image). Any failure stops ``up`` before anything is billed.
2. sandbox     ``yeto sandbox prepare`` (idempotent; a ready host is left alone)
3. pool        ``python -m yeto.launch.nc pool`` as a detached process (it orders, sets up
               every node through ``agentic.setup``, replaces lost nodes)
4. first group waits until the store holds a sample-buffer object (``--trainer-now``: no
               wait, the trainer starts while the NC nodes set up)
5. trainer     ``yeto launch`` (detached; ``yeto down <run>`` stops it)
6. dashboard   ``yeto dashboard serve --store`` (reads the run store only)
7. monitor     budget, wall clock, the trainer launcher's exit (rc != 0 or finished: FAILED),
               and a backstop independent of the launcher: the trainer log / tapes not
               growing for max(4 x --rl-stall-timeout, budget.silence_min) (preflight
               refuses --rl-stall-timeout 0); trainer starvation (``yeto.rl.starvation``:
               no new buffer object for ``starvation.alert_min`` alerts; for ``stop_min``
               with no healthy sampler stops the trainer through the STOP flag); no new
               policy release after the first one (``yeto.rl.release_stall``: an alert,
               a stop with ``breakers.run.release_stall.action: stop``); the sandbox
               host's container count of this run is a warning only.

The NC sampler pool is never released by an automatic decision of ``up``: a breach (trainer
failure, budget, wall clock, first-group timeout) stops the trainer only
(``yeto down <run>``, which keeps NC leases) and prints which nodes still bill. The pool,
its nodes and the sandboxes are released by ``yeto run down`` (or ``pool-stop``), or by a
budget / wall-clock breach when the run config opts in with
``"budget": {"release_nc_on_breach": true}``. Without it the monitor stays up after a
budget / wall-clock breach, prints the billing banner every poll and forces ``pool-stop``
once the spend passes 1.2x ``budget.usd`` (or the wall time 1.2x ``wall_h``).

Run config (JSON; paths may use ``~``)::

    {"run": "<run>", "store": "s3://<bucket>/runs/<run>",
     "env": {"AWS_PROFILE": "...", "YETO_NC_ACCOUNTS": "5", ...},
     "preflight": {"commands": [["python", "check.py"]]},
     "sandbox": {"envs": "<envs.json>", "docker_host": "ssh://<sandbox host>", "max_run_containers": 40,
                 "max_containers_per_sampler": 20},  # the alert line: the larger of the two
     # AgentENV microVMs: "sandbox": {"provider": "agentenv"} (no docker host: no sandbox
     # prepare / status / teardown, no container count; the sampler spec's agentic.setup
     # names the gateway)
     # several sandbox hosts: "docker_host": ["ssh://h1", "ssh://h2"], or "envs" (every host
     # of the envs file); the containers of the run are counted and removed on each
     "pool": {"config": "<pool.json>", "interval_s": 30},
     # or several rollout pools (any provider; yeto.launch.rollout.pools):
     # "samplers": [{"name": "nc", "provider": "nc", "config": "<pool.json>"},
     #              {"name": "modal", "provider": {"kind": "modal", ...}, "target": 1, ...}],
     "trainer": {"argv": ["--training-mode", "rl", ...], "usd_per_h": 136},
     # or an island fleet (yeto.rl.fleet): "islands" (one trainer and its pools per island) and
     # "fleet" (the trainers' shared argv, the dispense mode) instead of trainer / samplers / pool;
     # load_config compiles them into the trainer and samplers sections and the FleetPlan
     "first_group": {"timeout_s": 5400, "poll_s": 60},
     "dashboard": {"port": 8795},
     "budget": {"usd": 450, "wall_h": 3.0, "nc_usd_per_h": 24,  # per NC node
                "silence_min": 30, "release_nc_on_breach": false},
     # the same limits in the unified form (yeto.breakers; the keys above stay its aliases):
     "breakers": {"run": {"budget": {"usd": 450}, "wall_clock": {"hours": 3.0}}}}

The monitor's limits are the run layer of the breaker registry (``yeto.breakers.layers.run``:
``run.budget``, ``run.wall_clock``, ``run.trainer_silence``, ``run.first_group``,
``run.starvation``, ``run.release_stall``, ...), resolved from the ``breakers:`` section and the
legacy keys; ``yeto breaker show --config <run.json>`` prints them. Each trip is an ``rl_breaker``
line in ``events.jsonl`` of the state directory, and a stop records its ``stop_reason``
(``breaker:<name>``) in ``state.json``; exit codes are unchanged.

State: ``~/.yeto/rl-up/<run>/state.json`` (pids, logs, step times). ``up`` resumes: a step
whose process is alive is not started again. One ``up`` per run at a time (``up.lock``, an ``flock``
that dies with its process; ``run down`` takes it as well, stopping a running ``up`` first and
going on without the lock after a grace period); an in-process ``up`` removes a STOP flag an earlier stop left.
Clocks: each ``up`` process appends its period to ``ups`` (``yeto.rl.up_segments``) and restarts
``t0`` (``budget.wall_h``, the starvation watch, the first-group wait count from this up; a trainer
restart inside the same up does not reset them); the spend (``budget.usd``) counts from the run's
first start (``run_t0``) over every up. ``run status`` (``uptime``) and the dashboard show the
cumulative runtime (the up periods summed, the stopped gaps excluded) and the cumulative spend.

``yeto run scale --config F (--samplers N | --add K | --remove TICKET) [--drain | --fence]``
changes the NC sampler count while the run trains (``pool-scale`` on the run's pool; the
trainer is not touched). Before the pool's own checks it refuses a scale up when the run's
breaker has fired or one more node hour per new node passes ``budget.usd``. The budget,
the breakers and ``status`` count the nodes the pool state bills, never the target.
With several pools, ``--pool NAME`` / ``--provider KIND`` picks the one to scale (``--remove T``
finds the pool holding T); ``yeto rollout scale`` is the same command.
"""

from __future__ import annotations

import atexit
import copy
import fcntl
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from yeto.rl import up_consistency, up_segments

# ``YETO_STATE_DIR`` moves it (read once, at import); the default is ``~/.yeto/rl-up``.
STATE_ROOT = Path(os.environ.get("YETO_STATE_DIR") or "~/.yeto/rl-up").expanduser()
# the launcher's last line (yeto.cli): ``[yeto] run '<run>' finished: <state> (exit code <n>)``
# a budget / wall-clock breach stops the trainer only; the NC pool is released by force
# once the spend (or the wall time) passes this multiple of the budget
FORCE_FACTOR = 1.2
_FORMAT_WARNED: set[str] = set()  # pool state files of a newer format already reported
DEFAULT_STALL_TIMEOUT_S = 900.0  # yeto.launch.fleet.DEFAULT_RL_STALL_TIMEOUT_S
# the defaults of the run layer's breakers (yeto.breakers.layers.run declares them from these)
DEFAULT_FIRST_GROUP_TIMEOUT_S = 5400.0
DEFAULT_SILENCE_MIN = 30.0  # the trainer silence limit is max(SILENCE_STALL_FACTOR x stall, this)
SILENCE_STALL_FACTOR = 4
# ``yeto launch --cluster-prefix`` default (yeto.cli): the trainer's run name without the flag
LAUNCH_DEFAULT_NAME = "yeto"
SEAL_OWNER = "run-up-data-end"  # the incarnation the data-end seal claims an owned shard as
FINISHED = re.compile(r"finished: (\S+).*?\(exit code (-?\d+)\)")


def _log_tail(path: str | os.PathLike, offset: int = 0, size: int = 200_000) -> str:
    """The last ``size`` bytes of a log at or after byte ``offset`` (the current launch's
    start in an appended log), read with a seek: a multi-day log is never read whole."""
    with open(path, "rb") as fh:
        end = fh.seek(0, os.SEEK_END)
        offset = int(offset or 0)
        if offset > end:  # truncated or replaced under us: the offset no longer marks this launch
            offset = 0
        start = max(offset, end - size, 0)
        fh.seek(start)
        return fh.read(end - start).decode("utf-8", "replace")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _utc_seconds(stamp: Any) -> float | None:
    """The epoch seconds of a :func:`_now` stamp, None when it is not one."""
    import datetime as _dt  # noqa: PLC0415

    try:
        return _dt.datetime.strptime(str(stamp), "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=_dt.timezone.utc).timestamp()
    except ValueError:
        return None


def _epoch(value: Any) -> float | None:
    """Finite epoch seconds of a pool-state stamp; None when missing or corrupt."""
    if isinstance(value, bool):
        return None
    try:
        t = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return t if math.isfinite(t) else None


def _argv_value(argv: list[str], flag: str) -> str | None:
    """The last value of ``flag`` in ``argv`` (``--flag V`` or ``--flag=V``; None: absent)."""
    value = None
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            value = argv[i + 1]
        elif a.startswith(flag + "="):
            value = a.split("=", 1)[1]
    return value


def _log(msg: str) -> None:
    print(f"{_now()} [rl up] {msg}", flush=True)


def _output_tail(proc: subprocess.CompletedProcess) -> str:
    """The last 3000 characters of a status tool's output for ``run status`` and the
    dashboard: its stdout, or else its stderr without credentials (a failing tool may
    print presigned URLs, ``key=value`` secrets or URL userinfo)."""
    if proc.stdout:
        return proc.stdout[-3000:]
    from yeto.rl.engine.init_from_release import redact_text  # noqa: PLC0415
    return redact_text(proc.stderr or "")[-3000:]


# yeto preflight --no-bench of an ssh: trainer: a first pull of the trainer image (~24 GB) is in it
CLUSTER_PREFLIGHT_TIMEOUT_S = 5400.0

class RunConfigError(ValueError):
    """The run config (or the run's state directory) cannot be used: one line for the user."""


class PoolConfigError(RunConfigError):
    """The rollout layer refused the samplers / pool section (``resolve``); ``open_run`` names the config."""


# the commands that order or change a run refuse a breakers section the argv disagrees with;
# every other command (down, status, pause, resume) only warns: it must never be blocked
BREAKERS_STRICT_COMMANDS = frozenset({"up", "scale"})


def load_config(path: str, *, strict_breakers: bool = True) -> dict[str, Any]:
    """The run config at ``path``, checked and compiled. ``strict_breakers``: a ``breakers:``
    value that disagrees with ``trainer.argv`` (or that no knob carries) raises; without it
    the value is only warned about and the argv is kept as written (``run down`` / ``status``
    must open any config ``run up`` once accepted, :data:`BREAKERS_STRICT_COMMANDS`)."""
    file = Path(path).expanduser()
    try:
        text = file.read_text()
    except FileNotFoundError:
        raise RunConfigError(f"run config {file}: not found") from None
    except OSError as error:
        raise RunConfigError(f"run config {file}: cannot be read ({error.strerror or error})") from None
    try:
        cfg = json.loads(text)
    except ValueError as error:
        raise RunConfigError(f"run config {file}: not valid JSON ({error})") from None
    if not isinstance(cfg, dict):
        raise RunConfigError(f"run config {file}: must be a JSON object")
    fleet = "islands" in cfg or "fleet" in cfg  # an island fleet (yeto.rl.fleet): no trainer / samplers
    if fleet:
        from yeto.rl import fleet as fleet_mod  # noqa: PLC0415

        fleet_mod.check_exclusive(cfg)
    for key in ("run", "store") if fleet else ("run", "store", "trainer"):
        if key not in cfg:
            raise ValueError(f"run config: missing {key!r}")
    if not fleet and "pool" not in cfg and "samplers" not in cfg:
        raise ValueError("run config: missing 'samplers' (the rollout pools) or 'pool' (one pool config)")
    known = {"run", "store", "env", "preflight", "sandbox", "pool", "samplers", "trainer", "first_group",
             "dashboard", "budget", "comment", "syncer", "controller", "rollout", "starvation", "breakers",
             "islands", "fleet"}
    unknown = set(cfg) - known
    if unknown:
        raise ValueError(f"run config: unknown keys {sorted(unknown)} (known: {sorted(known)})")
    if fleet:
        # one synced-group trainer launch and every island's pools, from here on a run like any other
        cfg, _plan = fleet_mod.compile_fleet(cfg, strict_breakers=strict_breakers)
    rollout_section(cfg)
    check_sections(cfg)
    if cfg.get("breakers") is not None:
        from yeto.breakers import BreakerConfigError, parse_section  # noqa: PLC0415

        try:
            parse_section(cfg["breakers"])
        except BreakerConfigError as error:
            raise ValueError(f"run config: {error}") from None
        # a fleet's compile passes each island's trainer values itself (yeto.rl.fleet)
        cfg = apply_argv_breakers(cfg, layers=("launch",) if fleet else ARGV_LAYERS, strict=strict_breakers)
    return cfg


ARGV_LAYERS = ("trainer", "launch")  # the layers whose knobs are yeto launch flags and island env
ARGV_SURFACES = ("trainer_flag", "trainer_env", "launch_flag")


def apply_argv_breakers(cfg: dict[str, Any], *, layers: tuple[str, ...] = ARGV_LAYERS,
                        strict: bool = True) -> dict[str, Any]:
    """``breakers.trainer.*`` and ``breakers.launch.*`` (and ``breakers.switches.cost_guard``)
    written into ``trainer.argv``. The trainer reads its flags (``--rl-sample-wait-s``,
    ``--rl-max-policy-age``) and its island env, the launcher its own flags
    (``--rl-stall-timeout``, ``--budget-usd``, ``--idle-warn-minutes``, ...), never the
    section: without this the preflight showed the section's value while the trainer and
    the launcher ran their defaults. The write-back is the one ``yeto launch --breakers``
    does (:func:`yeto.breakers.materialize`): a flag becomes its flag, an island env knob
    ``--island-env K=V``. A value the argv (or the run env) already sets is left as written;
    one that disagrees with the section is refused, as is a section value no knob carries.

    ``launch.budget.usd`` is written only when ``breakers.launch.budget.usd`` is (a number or
    ``"inherit"``): the value it inherits from ``run.budget.usd`` alone stays with run up's
    ``run.budget`` (:func:`yeto.breakers.rules_facts.launch_budget_inherited_only`). A config
    whose section sets none of these (or only values the argv already holds) is returned as
    it is, its argv unchanged byte for byte. Without ``strict`` a refusal is a warning and the
    config is returned as it is."""
    try:
        return _apply_argv_breakers(cfg, layers)
    except ValueError as error:
        if strict:
            raise
        _log(f"WARNING: {error}: not written into trainer.argv (the argv is read as written; "
             "`yeto run up` refuses this config)")
        return cfg


def _apply_argv_breakers(cfg: dict[str, Any], layers: tuple[str, ...]) -> dict[str, Any]:
    section = cfg.get("breakers")
    if not isinstance(section, dict) or not any(section.get(key) for key in (*layers, "switches")):
        return cfg
    from yeto import breakers  # noqa: PLC0415
    from yeto.breakers.rules_facts import launch_budget_inherited_only  # noqa: PLC0415

    if launch_budget_inherited_only(section):  # no implicit inheritance: run.budget is run up's
        budget = (section.get("launch") or {}).get("budget") or {}
        extra = sorted(set(budget) - {"enabled", "action"})
        if extra:  # a launch budget without its own usd is off: these would be dropped unseen
            raise ValueError(f"run config: breakers.launch.budget {', '.join(extra)} needs breakers.launch.budget.usd "
                             "(a number, or \"inherit\" for run.budget.usd): without it the launch has no budget "
                             "of its own (run.budget stops the run)")
        section = {key: value for key, value in section.items() if key != "run"}
    trainer = cfg["trainer"]
    sources = breakers.Sources.from_run_config(cfg, argv=list(trainer["argv"]), env=dict(cfg.get("env") or {}),
                                               section=section)
    try:
        resolved = breakers.resolve(sources, layers=layers)
        out = breakers.materialize(resolved, sources, surfaces=ARGV_SURFACES)
    except breakers.BreakerConfigError as error:
        raise ValueError(f"run config: {error}") from None
    if not out.changes:
        return cfg
    for change in out.changes:
        _log(f"run config: breakers: {change}")
    return {**cfg, "trainer": {**trainer, "argv": out.argv}}


# Keys an earlier ``run up`` read and run configs still carry: ignored (with a log line), not
# refused, so an existing run's config still loads for its next ``run up`` / ``run down``.
RETIRED_KEYS = {"budget": {"silence_grace_min"}}


def _section(cfg: dict[str, Any], name: str, keys: set[str]) -> dict[str, Any]:
    section = cfg.get(name)
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"run config: {name} must be an object")
    retired = set(section) & RETIRED_KEYS.get(name, set())
    if retired:
        _log(f"run config: {name} keys {sorted(retired)} are no longer read (ignored)")
    unknown = set(section) - keys - retired
    if unknown:
        raise ValueError(f"run config: unknown {name} keys {sorted(unknown)} (known: {sorted(keys)})")
    return section


def _number(section: dict[str, Any], field: str, key: str, *, positive: bool = False) -> None:
    """``section[key]``, when present, is a finite number (> 0 with ``positive``, else >= 0)."""
    if key not in section:
        return
    value = section[key]
    ok = (not isinstance(value, bool) and isinstance(value, (int, float)) and value == value
          and value != float("inf") and (value > 0 if positive else value >= 0))
    if not ok:
        raise ValueError(f"run config: {field}.{key} must be a number {'> 0' if positive else '>= 0'} "
                         f"(got {value!r})")


def check_sections(cfg: dict[str, Any]) -> None:
    """The ``first_group``, ``budget``, ``dashboard``, ``starvation`` sections and ``rollout.stall_alert_s``:
    a misspelled key, a null, a string or a negative value is refused here, before anything is
    ordered (else it crashes the run once the pools bill, or is silently read as the default).
    ``budget.usd`` / ``budget.wall_h`` (null allowed) are checked by the preflight."""
    fg = _section(cfg, "first_group", {"timeout_s", "poll_s"})
    _number(fg, "first_group", "timeout_s", positive=True)
    _number(fg, "first_group", "poll_s", positive=True)
    b = _section(cfg, "budget", {"usd", "wall_h", "nc_usd_per_h", "silence_min", "release_nc_on_breach",
                                 "release_nc_on_finish"})
    for key in ("nc_usd_per_h", "silence_min"):
        if b.get(key) is not None:  # null: the default (number_or_default)
            _number(b, "budget", key)
    for key in ("release_nc_on_breach", "release_nc_on_finish"):
        if key in b and not isinstance(b[key], bool):
            raise ValueError(f"run config: budget.{key} must be true or false (got {b[key]!r})")
    dash = _section(cfg, "dashboard", {"port", "prices"})
    if "port" in dash and (isinstance(dash["port"], bool) or not isinstance(dash["port"], int)
                           or not 0 < dash["port"] < 65536):
        raise ValueError(f"run config: dashboard.port must be a port number 1-65535 (got {dash['port']!r})")
    _check_run_sections(cfg)
    _number(rollout_section(cfg), "rollout", "stall_alert_s")
    if cfg.get("controller") is not None and cfg["controller"] not in CONTROLLERS:
        raise ValueError(f"run config: controller {cfg['controller']!r} is not 'subprocess' or 'in_process'")
    if cfg.get("starvation") is not None:
        from yeto.rl.starvation import StarvationConfig  # noqa: PLC0415

        if not isinstance(cfg["starvation"], dict):
            raise ValueError("run config: starvation must be an object")
        try:
            StarvationConfig.from_section(cfg["starvation"])
        except ValueError as error:
            raise ValueError(f"run config: {error}") from None


def _string_list(section: dict[str, Any], field: str, key: str) -> None:
    if key in section and not (isinstance(section[key], list) and all(isinstance(a, str) for a in section[key])):
        raise ValueError(f"run config: {field}.{key} must be a list of strings (got {section[key]!r})")


def _check_run_sections(cfg: dict[str, Any]) -> None:
    """The ``trainer``, ``sandbox``, ``pool`` and ``preflight`` sections: a misspelled key
    (``trainer.usd_per_hour``: the spend counts the trainer at $0/h and ``budget.usd`` never
    trips; ``sandbox.max_run_container``: the container alert is off) or a wrongly typed
    value is refused at load, not silently read as unset."""
    tr = _section(cfg, "trainer", {"argv", "usd_per_h", "progress_paths"})
    if "argv" not in tr:
        raise ValueError("run config: trainer.argv (the yeto launch arguments) is required")
    _string_list(tr, "trainer", "argv")
    _string_list(tr, "trainer", "progress_paths")
    if tr.get("usd_per_h") is not None:
        _number(tr, "trainer", "usd_per_h")
    sb = _section(cfg, "sandbox", {"envs", "docker_host", "max_run_containers", "max_containers_per_sampler",
                                   "provider", "boat"})
    provider = sb.get("provider", SANDBOX_REMOTE_DOCKER)
    if provider not in SANDBOX_PROVIDERS:
        raise ValueError(f"run config: sandbox.provider must be one of {list(SANDBOX_PROVIDERS)} (got {provider!r})")
    if provider == SANDBOX_AGENTENV and sb.get("docker_host"):
        raise ValueError("run config: sandbox.docker_host is for remote-docker; AgentENV microVMs have no docker "
                         "host (drop it with sandbox.provider agentenv)")
    if provider == SANDBOX_BOAT and (sb.get("docker_host") or sb.get("envs")):
        raise ValueError("run config: sandbox.docker_host / sandbox.envs are for remote-docker; boat sandboxes have "
                         "no docker host (drop them with sandbox.provider boat)")
    if "boat" in sb:
        boat_account(sb)  # the account's limits: whole numbers >= 1
    for key in ("max_run_containers", "max_containers_per_sampler"):
        if sb.get(key) is not None:
            _number(sb, "sandbox", key)
    for key in ("envs", "docker_host"):
        value = sb.get(key)
        if value is not None and not isinstance(value, str) and not (
                key == "docker_host" and isinstance(value, list) and all(isinstance(v, str) for v in value)):
            raise ValueError(f"run config: sandbox.{key} must be a string{' or a list of strings' if key == 'docker_host' else ''} "
                             f"(got {value!r})")
    if "pool" in cfg:
        pool = _section(cfg, "pool", {"config", "interval_s"})
        _number(pool, "pool", "interval_s", positive=True)
    pf = _section(cfg, "preflight", {"commands"})
    commands = pf.get("commands")
    if commands is not None and not (isinstance(commands, list) and all(
            isinstance(c, list) and all(isinstance(a, str) for a in c) for c in commands)):
        raise ValueError(f"run config: preflight.commands must be a list of argv lists (got {commands!r})")


# sandbox.provider: remote-docker (default; ``yeto sandbox`` prepares the docker hosts of sandbox.envs)
# or agentenv (E2B-compatible microVMs: nothing to prepare, the sampler's agentic.setup carries the gateway)
SANDBOX_REMOTE_DOCKER = "remote-docker"
SANDBOX_AGENTENV = "agentenv"
SANDBOX_BOAT = "boat"  # boat.dev cloud sandboxes: nothing to prepare; the account's quota is the run's (sandbox.boat)
SANDBOX_PROVIDERS = (SANDBOX_REMOTE_DOCKER, SANDBOX_AGENTENV, SANDBOX_BOAT)



def boat_account(sandbox: dict[str, Any] | None) -> dict[str, int | None]:
    """:func:`yeto.rl.fleet_sandbox.boat_account` (``sandbox.boat``)."""
    from yeto.rl.fleet_sandbox import boat_account as account  # noqa: PLC0415

    return account(sandbox)
ROLLOUT_KEYS = {"stall_alert_s", "drain_on_exit"}
CONTROLLERS = ("subprocess", "in_process")


def rollout_section(cfg: dict[str, Any]) -> dict[str, Any]:
    """The run config's ``rollout`` section (the in-process controller's knobs):
    ``stall_alert_s`` (default 1800; 0: off): every ready sampler silent that long --
    no group, no trainer-wait heartbeat -- is logged as a rollout stall (an alert only);
    ``drain_on_exit`` (default true): when this process exits (also on SIGTERM), the
    in-process controller drains every pool; false: the machines keep running, nothing
    drives them, and the next ``yeto run up`` of this config adopts them from the pool
    state files (for a supervised host: tmux, systemd ``Restart=on-failure``)."""
    section = cfg.get("rollout") or {}
    if not isinstance(section, dict):
        raise ValueError("run config: rollout must be an object")
    unknown = set(section) - ROLLOUT_KEYS
    if unknown:
        raise ValueError(f"run config: unknown rollout keys {sorted(unknown)} (known: {sorted(ROLLOUT_KEYS)})")
    if "drain_on_exit" in section and not isinstance(section["drain_on_exit"], bool):
        raise ValueError("run config: rollout.drain_on_exit must be true or false")
    return dict(section)


def _pythonpath_without_start_dir(env: dict[str, str]) -> dict[str, str]:
    """``PYTHONSAFEPATH`` only drops the implicit sys.path[0]; a ``PYTHONPATH`` of
    ``.`` (or an empty entry) puts the start directory right back. Resolve relative
    entries now and drop the ones naming the start directory, unless that directory
    is where this ``yeto`` package lives (children must import the same yeto)."""
    raw = env.get("PYTHONPATH")
    if raw is None:
        return {}
    start = Path.cwd().resolve()
    own_root = Path(__file__).resolve().parents[2]
    kept: list[str] = []
    for entry in raw.split(os.pathsep):
        path = Path(entry or ".")
        if not path.is_absolute():
            resolved = path.resolve()
            if resolved == start and resolved != own_root:
                continue
            entry = str(resolved)
        if entry not in kept:
            kept.append(entry)
    return {"PYTHONPATH": os.pathsep.join(kept)}


def _merge_state(base: Any, mine: Any, disk: Any) -> Any:
    """Three-way merge of ``state.json``: ``base`` is what this process last read or wrote,
    ``mine`` its copy now, ``disk`` the file now. Keys this process did not touch keep the
    disk value; dicts merge key by key; a list this process only appended to gets its new
    items after the disk list; any other change of this process wins."""
    if mine == base:
        return disk
    if isinstance(mine, dict) and isinstance(base, dict) and isinstance(disk, dict):
        out = dict(disk)
        for key in set(base) | set(mine):
            if key not in mine:
                if key in base and base[key] == disk.get(key):
                    out.pop(key, None)
                continue
            if key in base and key in disk:
                out[key] = _merge_state(base[key], mine[key], disk[key])
            elif mine.get(key) != base.get(key) or key not in disk:
                out[key] = mine[key]
        return out
    if (isinstance(mine, list) and isinstance(base, list) and isinstance(disk, list)
            and mine[:len(base)] == base):
        return disk + mine[len(base):]
    return mine



def _close_quietly(fh: Any) -> None:
    if fh is None:
        return
    try:
        fh.close()
    except OSError:  # the buffered tail did not fit on a full disk
        pass


def _killpg(proc: subprocess.Popen) -> None:
    """SIGKILL ``proc``'s process group (it leads its own session) and reap it."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()
    proc.wait()

class Run:
    def __init__(self, cfg: dict[str, Any], state_root: Path | None = None):
        self.cfg = cfg
        self.run = str(cfg["run"])
        self.dir = (state_root or STATE_ROOT) / self.run
        # Set when this run had no local state directory (``run down`` from another host):
        # down still stops what the config names, but cannot claim the run is verified down.
        self.no_local_state: Path | None = None
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.state: dict[str, Any] = {}
        self._state_base: dict[str, Any] = {}
        self.reload_state(strict=True)
        # read before this process marks anything (its preflight / sandbox stamps are now)
        self._legacy_stamp = self._legacy_last_stamp()
        self.env = dict(os.environ)
        self.env.update({k: os.path.expanduser(str(v)) for k, v in (cfg.get("env") or {}).items()})
        self.env.setdefault("YETO_RESOURCE_LEDGER", str(Path(f"~/.yeto/runs/{self.run}/resources.jsonl").expanduser()))
        # Every child is ``python -m ...`` (and spawns more, e.g. ``python -m modal``):
        # without this the directory ``up`` was started from is sys.path[0] of each,
        # and a stray ``h2.py`` / ``json.py`` there replaces a real module.
        self.env.setdefault("PYTHONSAFEPATH", "1")
        self.env.update(_pythonpath_without_start_dir(self.env))
        self.py = sys.executable
        from yeto.launch.rollout.pools import resolve  # noqa: PLC0415

        try:
            self.pools = resolve(cfg, self.dir)  # the rollout pools (samplers list, or the one pool)
        except RunConfigError:
            raise
        except ValueError as error:  # a samplers / pool section the rollout layer refuses
            raise PoolConfigError(str(error)) from None
        self.controller: Any = None  # controller: in_process -> the RolloutController of every pool
        self.auto_backlog = self._derive_auto_backlog()

    def _derive_auto_backlog(self) -> dict[str, Any] | None:
        """The cap of the pools' ``max_backlog_groups: "auto"`` samplers
        (:func:`yeto.rl.up_consistency.auto_backlog`), handed to every pool process
        in its environment (:data:`~yeto.rl.engine.sample_roles.AUTO_BACKLOG_ENV`);
        None (and no variable, not even an inherited one) when no spec says "auto"
        or the trainer is not a single buffer trainer: those samplers take the fallback."""
        from yeto.rl import up_consistency  # noqa: PLC0415
        from yeto.rl.engine import sample_roles  # noqa: PLC0415

        for key in (sample_roles.AUTO_BACKLOG_ENV, sample_roles.AUTO_BACKLOG_WHY_ENV):
            self.env.pop(key, None)
        try:
            pools = [(ref.load(), ref.data_path) for ref in self.pools]
        except (OSError, ValueError):
            return None  # the pool dry-run reports an unreadable pool config
        argv = [str(a) for a in (self.cfg.get("trainer") or {}).get("argv") or ()]
        found = up_consistency.auto_backlog(argv, pools)
        if found is not None:
            self.env[sample_roles.AUTO_BACKLOG_ENV] = str(found["groups"])
            self.env[sample_roles.AUTO_BACKLOG_WHY_ENV] = found["why"]
        return found

    # -- state
    def reload_state(self, *, strict: bool = False) -> bool:
        """Read ``state.json`` again into ``self.state`` and make it the merge base of
        :meth:`save` too (else what another process appended since our last read, e.g. a
        ``run scale``'s ``scales``, counts as our own change and is appended twice). An
        unreadable file (truncated JSON, I/O error) keeps the state in memory and returns
        False, unless ``strict`` (the first read: nothing in memory to fall back to)."""
        try:
            state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        except (ValueError, OSError) as exc:
            if strict:
                raise
            _log(f"{self.state_path}: unreadable ({type(exc).__name__}: {exc}); keeping the state in memory")
            return False
        self.state = state
        self._state_base = copy.deepcopy(state)
        return True

    def save(self) -> None:
        """Write ``state.json`` as a read-modify-write under ``state.lock`` (``flock``):
        several processes of a run write it (``run up``'s monitor, ``run scale``, ``run
        down``), each from its own copy. Only what this process changed since it last read
        or wrote the file is applied to what is on disk now (a list it appended to gets the
        appended items), so one writer no longer drops another's ``scales`` / ``steps`` /
        ``procs``. The write is atomic (temp file + rename): a reader never sees half a file."""
        tmp = self.state_path.with_name(f".state.json.{os.getpid()}.tmp")
        try:
            with open(self.dir / "state.lock", "a+") as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                try:
                    disk = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
                except ValueError:
                    disk = {}
                base = getattr(self, "_state_base", None)
                merged = _merge_state(base if base is not None else disk, self.state, disk)
                tmp.write_text(json.dumps(merged, indent=1, sort_keys=True))
                os.replace(tmp, self.state_path)
        except OSError as error:
            if not getattr(self, "soft_save", False):
                raise
            # the monitor (budget / wall-clock breaker, forced pool-stop) must outlive a full
            # or read-only disk: the state stays in memory (merged again at the next save)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            failure = f"{type(error).__name__}: {error}"
            if getattr(self, "save_error", None) != failure:
                _log(f"CRITICAL state_write_failed: {self.state_path}: {failure}; the monitor goes on "
                     "with the state in memory")
            self.save_error = failure
            return
        if getattr(self, "save_error", None):
            _log(f"state_write_recovered: {self.state_path} written again")
            self.save_error = None
        self.state.clear()  # in place: callers may hold self.state
        self.state.update(merged)
        self._state_base = copy.deepcopy(merged)

    def mark(self, step: str, **fields: Any) -> None:
        self.state.setdefault("steps", {})[step] = {"at": _now(), **fields}
        self.save()

    def lock_up(self, role: str = "up") -> int | None:
        """Hold ``up.lock`` of the run for the life of this process: a second ``yeto run up``
        of the same run while one still runs would start a second monitor (two breakers,
        two starvation stops) and both rewrite ``state.json`` from their own copy. An
        ``flock`` dies with its process, so a crashed ``up`` leaves no stale lock. None:
        the lock is ours; else the PID of the process that holds it (0: unknown), whose
        role (``up`` / ``down``) is then in ``self.lock_holder``. The file names the holder
        as JSON ``{"role", "pid", "start"}`` (``start``: the process start time, so a PID
        reused by another process is not taken for the holder)."""
        if getattr(self, "_up_lock", None) is not None:
            return None
        fh = open(self.dir / "up.lock", "a+")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.seek(0)
            text = fh.read().strip()
            fh.close()
            self.lock_holder = _parse_lock_holder(text)
            return self.lock_holder["pid"]
        fh.seek(0)
        fh.truncate()
        fh.write(json.dumps({"role": role, "pid": os.getpid(), "start": _proc_start(os.getpid())}))
        fh.flush()
        self._up_lock = fh
        return None

    # -- processes
    def _cmd(self, argv: list[str], *, log: str, timeout: float | None = None) -> int:
        path = self.dir / log
        fh: Any = None
        try:
            fh = open(path, "a")
            fh.write(f"\n### {_now()} {' '.join(argv)}\n")
            fh.flush()
        except OSError as error:
            # a full / read-only disk must not stop the command itself (a pool-stop, a down):
            # it runs without its log
            _log(f"{path}: log not writable ({type(error).__name__}: {error}); "
                 f"running {' '.join(argv[:6])} ... without it")
            _close_quietly(fh)
            fh = None
        try:
            # its own process group: a timeout kills the whole tree (a `yeto down` hangs in the
            # modal / ssh child it spawned; killing the direct child only would leave that one
            # running and still tearing down / billing after run down moved on)
            proc = subprocess.Popen(argv, env=self.env, stdout=fh if fh is not None else subprocess.DEVNULL,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # a hung step (dashboard export, Modal teardown) fails that step only: raised,
                # it ended run down before the pool-stop and the sampler nodes kept billing
                _killpg(proc)
                if fh is not None:
                    try:
                        fh.write(f"\n### {_now()} TIMED OUT after {timeout:.0f}s (process group killed)\n")
                    except OSError:
                        pass
                _log(f"{' '.join(argv[:6])} ... TIMED OUT after {timeout:.0f}s (log {path})")
                return 124
            except BaseException:  # Ctrl-C / SIGTERM of this process: its own session does not get it
                _killpg(proc)
                raise
        finally:
            _close_quietly(fh)
        _log(f"{' '.join(argv[:6])} ... rc={proc.returncode} (log {path})")
        return proc.returncode

    def _spawn(self, name: str, argv: list[str]) -> int:
        old = (self.state.get("procs") or {}).get(name)
        if old and _alive(old["pid"]):
            _log(f"{name}: already running (PID {old['pid']})")
            return int(old["pid"])
        path = self.dir / f"{name}.log"
        fh = open(path, "a")
        fh.write(f"\n### {_now()} {' '.join(argv)}\n")
        fh.flush()
        offset = fh.tell()
        proc = subprocess.Popen(argv, env=self.env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        self.state.setdefault("procs", {})[name] = {"pid": proc.pid, "log": str(path), "started": _now(),
                                                    "offset": offset}
        self.save()
        _log(f"{name}: started PID {proc.pid} (log {path})")
        return proc.pid

    def yeto(self, *args: str) -> list[str]:
        return [self.py, "-m", "yeto.cli", *args]

    def nc(self, *args: str) -> list[str]:
        return [self.py, "-m", "yeto.launch.nc", *args]

    @property
    def pool_config(self) -> str:
        """The first pool's config file (the only one of a run config with ``pool``)."""
        return self.pools[0].config

    @property
    def pool_configs(self) -> list[str]:
        return [p.config for p in self.pools]

    def _proc_name(self, i: int) -> str:
        return "pool" if i == 0 else f"pool-{self.pools[i].name}"

    def pool_index(self, *, pool: str | None = None, provider: str | None = None,
                   ticket: str | None = None) -> int | None:
        """The pool a command is for: by name / provider kind (exactly one match), else the
        pool whose state holds ``ticket``, else the only pool (None: ambiguous or none)."""
        if pool is not None or provider is not None:
            from yeto.launch.rollout.pools import select  # noqa: PLC0415

            chosen = select(self.pools, pool=pool, provider=provider)
            return self.pools.index(chosen[0]) if len(chosen) == 1 else None
        if ticket is not None:
            for i in range(len(self.pools)):
                if any(m.get("ticket") == ticket for m in (self.pool_doc(i) or {}).get("members") or ()):
                    return i
        return 0 if len(self.pools) == 1 else None

    @property
    def envs(self) -> str | None:
        """The docker-host envs file ``yeto sandbox`` drives; None for agentenv (no docker host:
        no sandbox status / prepare / teardown)."""
        sb = self.cfg.get("sandbox") or {}
        if sb.get("provider", SANDBOX_REMOTE_DOCKER) in (SANDBOX_AGENTENV, SANDBOX_BOAT):
            return None
        return os.path.expanduser(sb["envs"]) if sb.get("envs") else None

    # -- breakers (yeto.breakers; the run layer: yeto.breakers.layers.run)
    @property
    def breakers(self) -> Any:
        """The run config's resolved breakers: the ``breakers:`` section over the legacy keys
        (``budget.*``, ``starvation.*``, ...). Resolved again when the config changes; a
        conflict never raises here (the preflight refuses it, :meth:`breakers_problem`)."""
        from yeto import breakers  # noqa: PLC0415

        key = json.dumps(self.cfg, sort_keys=True, default=str)
        memo = getattr(self, "_breakers_memo", None)
        if memo is None or memo[0] != key:
            memo = self._breakers_memo = (key, breakers.resolve(breakers.Sources.from_run_config(self.cfg),
                                                                strict=False, layers=("run", "launch")))
        return memo[1]

    def breaker_param(self, name: str, param: str) -> Any:
        """``param`` of the resolved breaker ``name``; None when the breaker is off."""
        return self.breakers.get(f"{name}.{param}") if self.breakers.enabled(name) else None

    def breaker_event(self, name: str, *, value: float | None, limit: float | None, detail: str = "",
                      state: str = "trip", legacy_event: str | None = None) -> dict[str, Any]:
        """Report a run-layer breaker (``rl_breaker``) on the run's tape, ``events.jsonl`` in
        its state directory. A stop trip carries (and records) the run's ``stop_reason``;
        the caller still decides and acts as before. Never raises."""
        from yeto import breakers  # noqa: PLC0415

        try:
            entry = self.breakers.entry(name)
            action, source = entry.action, entry.source.location
        except Exception:  # noqa: BLE001 - a view only: the event still goes out
            action = source = None
        return breakers.trip(name, value, limit, action, state=state, emit=self._append_event, detail=detail,
                             legacy_event=legacy_event, config_source=source, run=self.run)

    def _append_event(self, event: dict[str, Any]) -> None:
        with open(self.dir / "events.jsonl", "a") as fh:
            fh.write(json.dumps(event, default=str) + "\n")

    # -- island fleet (yeto.rl.fleet): the run config compiled to one trainer launch and its pools
    def fleet_plan(self) -> Any:
        """The run's :class:`yeto.rl.fleet.FleetPlan`; None for a single-trainer run."""
        from yeto.rl import fleet  # noqa: PLC0415

        cfg = getattr(self, "cfg", None)  # a bare Run (pool helpers alone) has no config
        return None if cfg is None else fleet.plan_of(cfg)

    def log_fleet_plan(self) -> None:
        """Each island's learner id, store sub-directory, trainer cluster and price, and pools."""
        plan = self.fleet_plan()
        for line in plan.lines() if plan is not None else ():
            _log(line)

    def fleet_problem(self) -> str | None:
        """Why the fleet's training form cannot run (:func:`yeto.rl.fleet.form_problem`)."""
        from yeto.rl import fleet  # noqa: PLC0415

        plan = self.fleet_plan()
        return None if plan is None else fleet.form_problem(plan.sync)

    def write_fleet_plan(self) -> bool:
        """``<store>/fleet/plan.json`` (the island table the dashboard and the monitor read),
        written before anything is ordered; False (logged) when the store refuses it."""
        from yeto.rl import fleet  # noqa: PLC0415

        plan = self.fleet_plan()
        if plan is None:
            return True
        uri = fleet.substore_uri(self.cfg["store"], fleet.PLAN_OBJECT)
        try:
            self._objects().write(fleet.PLAN_OBJECT, plan.dumps().encode())
        except Exception as error:  # noqa: BLE001 - reported, nothing is ordered
            _log(f"CRITICAL: the fleet plan could not be written to {uri} ({type(error).__name__}: {error})"[:400])
            return False
        self.mark("fleet_plan", object=uri)
        return True

    def fleet_rt(self) -> Any:
        """The fleet's per-island runtime (:class:`yeto.rl.up_fleet_runtime.FleetRuntime`);
        call only when :meth:`fleet_plan` is set."""
        rt = getattr(self, "_fleet_rt", None)
        if rt is None:
            from yeto.rl.up_fleet_runtime import FleetRuntime  # noqa: PLC0415

            rt = self._fleet_rt = FleetRuntime(self)
        return rt

    def _pool_suffix(self, i: int) -> str:
        """Where a per-pool preflight line points: ``(pool <name>)`` with several pools;
        in a fleet ``(island k (name), pool i<k>-<name>)``."""
        plan = self.fleet_plan()
        if plan is not None:
            from yeto.rl import up_fleet  # noqa: PLC0415

            return f" ({up_fleet.pool_where(plan, self.pools[i].name)})"
        return f" (pool {self.pools[i].name})" if len(self.pools) > 1 else ""

    def _fleet_specs(self) -> list[tuple[str, dict[str, Any] | None]]:
        """``(pool name, rendered sampler spec)`` of every pool (None: no spec / unreadable)."""
        out = []
        for ref in self.pools:
            try:
                out.append((ref.name, up_consistency.effective_spec(ref.load())))
            except (OSError, ValueError):
                out.append((ref.name, None))  # the pool dry-run reports an unreadable pool config
        return out

    def fleet_preflight(self) -> str | None:
        """The checks only an island fleet has (:mod:`yeto.rl.up_fleet`), island by island:
        each island's pools record the same ``rollout_fields`` as its trainer trains on,
        their engines load the fleet's release form, each island's release cost and cadence, the
        syncer's region and price, and the fleet's budget estimate (logged). The reason
        to refuse the run (every island and pool named), else None. None at once without
        a fleet."""
        plan = self.fleet_plan()
        if plan is None:
            return None
        from yeto.rl import up_fleet  # noqa: PLC0415

        argv = [str(a) for a in self.cfg["trainer"]["argv"]]
        specs = self._fleet_specs()
        problems = up_fleet.breaker_pool_problems(plan, self.cfg.get("breakers"))
        problems += up_fleet.rollout_fields_problems(plan, argv, specs)
        refused, notes = up_fleet.release_form_lines(plan, specs, argv)
        problems += refused
        from yeto.rl import fleet  # noqa: PLC0415

        cadence = up_fleet.release_every_warnings(plan, argv, specs, step_s=fleet.island_step_s(self.cfg))
        for line in notes + cadence:
            _log(f"preflight: {line}")
        for line in up_fleet.syncer_lines(plan, argv, self.cfg.get("syncer")):
            _log(f"preflight: {line}")
        refused, notes = self.fleet_sandbox_findings()
        problems += refused
        for line in notes:
            _log(f"preflight: {line}")
        for line in self.fleet_budget_lines():
            _log(f"preflight: {line}")
        return "; ".join(problems) if problems else None

    def fleet_sandbox_pools(self) -> list[Any]:
        """The fleet's agentic pools as :mod:`yeto.rl.fleet_sandbox` sees them: the rendered
        spec, the env their harness sees, their samplers at the target and the elastic
        maximum, and their ``harness.sandbox_caps`` as each pool resolves them."""
        from yeto.breakers.rules_facts import pool_facts, pool_sources  # noqa: PLC0415
        from yeto.rl import fleet_sandbox, up_fleet  # noqa: PLC0415
        from yeto.rl.engine import sampler_breakers  # noqa: PLC0415

        plan = self.fleet_plan()
        out = []
        for i, ref in enumerate(self.pools):
            try:
                cfg = ref.load()
                spec = up_consistency.effective_spec(cfg)
                target = self.pool_target(None, i)
            except (OSError, ValueError):
                continue  # the pool dry-run reports an unreadable pool config
            if not isinstance((spec or {}).get("agentic"), dict):
                continue
            island = plan.island_of_pool(ref.name) if plan is not None else None
            elastic = cfg.get("elastic") if isinstance(cfg.get("elastic"), dict) else {}
            max_nodes = elastic.get("max_nodes") if isinstance(elastic.get("max_nodes"), int) else target
            env = dict(pool_sources(cfg, run_config=self.cfg).env or {})
            if isinstance(spec.get("breakers"), dict):
                # the sampler resolves its harness caps from its spec's breakers section
                # (sampler_breakers.apply_harness_env), where run up writes the pool's
                # share of a shared account (pools.with_boat_shares): read them the same way
                resolved = sampler_breakers.spec_harness_resolved(spec, env)
            else:
                resolved = pool_facts(ref.name, cfg, ref.data_path, run_config=self.cfg).resolved
            caps = {key: resolved.get(f"harness.sandbox_caps.{key}")
                    for key in ("max_sandboxes", "agentenv_max_active_n", "boat_max_active_n",
                                "boat_starts_per_minute_n")}
            row = fleet_sandbox.pool_of(
                ref.name, up_fleet.pool_where(plan, ref.name) if plan is not None else f"pool {ref.name}",
                None if island is None else island.index, spec, env=env,
                shards=[f"{cfg.get('shard_base')}-{n}" for n in range(max(0, int(target)))],
                max_nodes=int(max_nodes), caps=caps)
            if row is not None:
                out.append(row)
        return out

    def fleet_sandbox_findings(self) -> tuple[list[str], list[str]]:
        """``(problems, notes)`` of the fleet's agentic pools' sandboxes
        (:mod:`yeto.rl.fleet_sandbox`): providers, the caps of what they share, spend."""
        from yeto.breakers import fleet as breakers_fleet  # noqa: PLC0415
        from yeto.rl import fleet_sandbox  # noqa: PLC0415

        pools = self.fleet_sandbox_pools()
        if not pools:
            return [], []
        run_envs = None
        if self.envs:
            from yeto.sandbox.envs import EnvsError, load_envs  # noqa: PLC0415

            try:
                run_envs = load_envs(self.envs, run=self.run if fleet_sandbox.LABEL.fullmatch(self.run) else None)
            except (EnvsError, OSError, ValueError):
                run_envs = None  # yeto sandbox status (the preflight's next step) names why
        problems, notes = fleet_sandbox.provider_findings(pools, sandbox=self.cfg.get("sandbox"))
        refused, more = fleet_sandbox.capacity_findings(pools, run_envs=run_envs,
                                                        boat=boat_account(self.cfg.get("sandbox")))
        problems += refused + fleet_sandbox.port_findings(self.fleet_port_rows())
        return problems, notes + more + fleet_sandbox.spend_lines(pools, breakers_fleet.sandbox_price(self.cfg))

    def fleet_port_rows(self) -> list[tuple[str, Any, int, int]]:
        """``(where, port range, shard_offset, shard_limit)`` of each fleet pool whose
        samplers take a records relay / tunnel port by shard index (:mod:`yeto.rl.fleet_sandbox`)."""
        from yeto.launch.nc import cli as nc_cli  # noqa: PLC0415
        from yeto.rl import fleet, up_fleet  # noqa: PLC0415

        plan = self.fleet_plan()
        out = []
        for ref in self.pools:
            try:
                cfg = ref.load()
                spec = up_consistency.effective_spec(cfg)
                limit = nc_cli.pool_shard_limit(cfg)
            except (OSError, ValueError, TypeError):
                continue
            key = fleet.port_share_key(spec)
            if key is None or not limit:
                continue
            offset = (spec or {}).get("shard_offset")
            where = up_fleet.pool_where(plan, ref.name) if plan is not None else f"pool {ref.name}"
            out.append((where, key, offset if isinstance(offset, int) and not isinstance(offset, bool) else 0,
                        int(limit)))
        return out

    def fleet_budget_lines(self) -> list[str]:
        """The fleet's hourly estimate (each pool's target x node price, each island's
        trainer, the syncer) over the wall clock, against ``budget.usd``."""
        from yeto.breakers import fleet as breakers_fleet  # noqa: PLC0415
        from yeto.rl import up_fleet  # noqa: PLC0415

        rows = []
        for i, ref in enumerate(self.pools):
            try:
                target = self.pool_target(None, i)
            except Exception:  # noqa: BLE001 - an unreadable pool config: the pool dry-run says so
                target = 0
            rows.append((ref.name, target, self.node_usd_per_h(i)))
        return up_fleet.budget_lines(self.fleet_plan(), pools=rows,
                                     syncer_usd_per_h=breakers_fleet.syncer_price(self.cfg),
                                     usd=self.breaker_param("run.budget", "usd"),
                                     wall_h=self.breaker_param("run.wall_clock", "hours"),
                                     sandbox_usd_per_h=breakers_fleet.sandbox_price(self.cfg))

    # -- steps
    def preflight(self) -> bool:
        from yeto.launch_preflight import modal_sdk_problem  # noqa: PLC0415

        problem = self.fleet_problem()
        if problem is not None:
            _log(f"preflight: {problem}")
            self.mark("preflight", ok=False)
            return False
        problem = modal_sdk_problem(self.env) if "modal:" in " ".join(self.cfg["trainer"]["argv"]) else None
        if problem is not None:
            _log(f"preflight: {problem}")
            self.mark("preflight", ok=False)
            return False
        problem = self.budget_problem()
        if problem is not None:
            _log(f"preflight: {problem}")
            self.mark("preflight", ok=False)
            return False
        problem = self.starvation_problem()
        if problem is not None:
            _log(f"preflight: {problem}")
            self.mark("preflight", ok=False)
            return False
        problem = self.breakers_problem()
        if problem is not None:
            _log(f"preflight: {problem}")
            self.mark("preflight", ok=False)
            return False
        for line in self.breakers.deprecations():
            _log(f"preflight: deprecated: {line}")
        self.log_release_stall()
        problem = self.env_file_problem()
        if problem is not None:
            _log(f"preflight: {problem}")
            self.mark("preflight", ok=False)
            return False
        problem = self.rules_problem()
        if problem is not None:
            _log(f"preflight: {problem}")
            self.mark("preflight", ok=False)
            return False
        for i in range(len(self.pools)):
            problem = self.shard_problem(i)
            if problem is not None:
                _log(f"preflight: {problem}" + self._pool_suffix(i))
                self.mark("preflight", ok=False)
                return False
            problem = self.rollout_fields_problem(i) or self.consistency_problem(i)
            if problem is not None:
                _log(f"preflight: {problem}" + self._pool_suffix(i))
                self.mark("preflight", ok=False)
                return False
            self.log_compaction(i)
            if self.pools[i].spec_carried:  # breakers.sampler / .harness: no pool knob, the sampler spec has them
                _log(f"preflight: breakers: pool {self.pools[i].name}: applied in the sampler spec "
                     f"(sampler.breakers of {self.pools[i].config}): {', '.join(self.pools[i].spec_carried)}")
        problem = self.fleet_preflight()
        if problem is not None:
            _log(f"preflight: {problem}")
            self.mark("preflight", ok=False)
            return False
        ok = True
        for path in self.pool_configs:
            ok &= self._cmd(self.nc("pool", "--config", path, "--dry-run"), log="preflight.log") == 0
        if self.envs:
            ok &= self._cmd(self.yeto("sandbox", "status", "--envs", self.envs, "--store", self.cfg["store"]),
                            log="preflight.log") == 0
        if self.trainer_restart():
            _log("preflight: an earlier trainer launch of this run exists: a start from the base would hit "
                 "the rounds its ledger recorded on the island volume (LedgerError); the trainer starts "
                 "with --rl-ledger-set-aside-before, which sets that ledger aside (ledger.before-<ts>)")
        ok &= self._cmd(self.yeto("launch", *self.trainer_argv(), "--dry-run"), log="preflight.log") == 0
        for argv in (self.cfg.get("preflight") or {}).get("commands") or ():
            ok &= self._cmd([os.path.expanduser(a) for a in argv], log="preflight.log") == 0
        self.write_breakers_resolved()
        self.mark("preflight", ok=bool(ok))
        return bool(ok)

    def write_breakers_resolved(self) -> None:
        """``breakers.resolved.json`` in the state directory: every layer's settings as this
        run config resolves them, with the dependency findings (the ``yeto breaker show
        --json`` object; the dashboard's Breakers panel and ``run status`` read it). Never
        raises: a view only."""
        from yeto import breakers  # noqa: PLC0415
        from yeto.breakers.rules_facts import mark_run_enforced, run_view_sources  # noqa: PLC0415

        try:
            # the run's only pool fills the pool / sampler / harness rows (a view; Run.breakers,
            # which the monitor reads, stays the run config's own run / launch resolve)
            try:
                view = run_view_sources(self.cfg)
                sources, pool = view.sources, view.pool
            except ValueError as error:
                _log(f"preflight: breakers.resolved.json shows the run config alone ({error})")
                sources, pool = breakers.Sources.from_run_config(self.cfg), None
            resolved = breakers.resolve(sources, pool=pool, strict=False)
            doc = mark_run_enforced(resolved.to_json(breakers.preflight(resolved, run_config=self.cfg, sources=sources)),
                                    self.cfg)
            tmp = self.dir / "breakers.resolved.json.tmp"
            tmp.write_text(json.dumps(doc, indent=1, default=str))
            os.replace(tmp, self.dir / "breakers.resolved.json")
        except Exception as error:  # noqa: BLE001
            _log(f"preflight: breakers.resolved.json not written ({type(error).__name__}: {error})")

    # The trainer's cluster preflight (yeto.launch.preflight.gate): run before anything is
    # ordered; once it passed (or --skip-preflight skipped it) the trainer's own launch skips it.
    cluster_preflight_passed = False

    def cluster_preflight(self, *, dry_run: bool = False) -> bool:
        """``yeto preflight --no-bench`` of the trainer's ``ssh:`` islands (node checks and
        TCP ports): False on a FAIL; a dry run only logs the command. True without one."""
        from yeto.launch.preflight.gate import commands_for_launch_argv  # noqa: PLC0415

        commands = commands_for_launch_argv([str(a) for a in self.cfg["trainer"]["argv"]])
        if not commands:
            return True
        shown = "; ".join("yeto " + " ".join(c) for c in commands)
        if dry_run:
            _log(f"dry run: before ordering, run up would run the trainer's cluster preflight: {shown}")
            return True
        ok = all(self._cmd(self.yeto(*c), log="preflight.log", timeout=CLUSTER_PREFLIGHT_TIMEOUT_S) == 0
                 for c in commands)
        self.mark("cluster_preflight", ok=ok)
        if ok:
            self.cluster_preflight_passed = True
        else:
            _log(f"cluster preflight failed: {shown} (see {self.dir / 'preflight.log'} and the report in the "
                 "trainer's run dir; --skip-preflight skips it)")
        return ok

    def sandbox_prepare(self) -> bool:
        if not self.envs:
            return True
        ok = self._cmd(self.yeto("sandbox", "prepare", "--envs", self.envs, "--store", self.cfg["store"]),
                       log="sandbox.log") == 0
        self.mark("sandbox", ok=ok)
        return ok

    def pool(self) -> None:
        from yeto.launch.nc import pool_run  # noqa: PLC0415

        os.environ.update({k: v for k, v in self.env.items() if k.startswith(("YETO_NC_", "NC_"))})
        if self.fleet_plan() is not None and self.fleet_rt().all_stopped() is not None:
            stopped = "; ".join(f"{self.fleet_rt().label(i.index)}: {self.fleet_rt().island_stopped(i.index)}"
                                for i in self.fleet_plan().islands)
            raise RunUpRefused(f"every island of the fleet stays stopped by an earlier trip of its own ({stopped}): "
                               "nothing would sample. Remove fleet_island_stops from "
                               f"{self.state_path} for the islands to run again (or start a new run name)")
        if self.in_process:
            flag = self.dir / "STOP"
            if flag.exists():  # left by an earlier stop: the new controller would drain every pool at once
                _log(f"removing the STOP flag {flag} an earlier stop left: this run up drives the pools again")
                flag.unlink(missing_ok=True)
            self.start_controller()
            self.begin_up()
            self.mark("pool", controller="in_process", pools=[r.name for r in self.pools])
            return
        down = []
        for i, ref in enumerate(self.pools):
            proc = pool_run.pool_process(ref.load())
            if proc and proc.get("alive"):
                _log(f"pool {ref.name}: loop already running (PID {proc.get('pid')})")
            elif self.fleet_plan() is not None and self.fleet_rt().pool_restart_refusal(ref.name):
                _log(f"pool {ref.name}: {self.fleet_rt().pool_restart_refusal(ref.name)}")
            else:
                down.append((i, ref))
        # every stopped pool is checked before any is restarted or spawned: a refusal of a
        # later pool must not leave an earlier one's loop running (ordering nodes for a run
        # up that refused), as in_process restarts nothing until every pool passed
        self._restart_stopped_pool_states([ref for _, ref in down])
        for i, ref in down:
            self._spawn(self._proc_name(i), self.nc("pool", "--config", ref.config,
                                                    "--interval", str(int(ref.interval_s))))
        self.begin_up()
        self.mark("pool")

    def begin_up(self) -> None:
        """Open this ``run up`` process's period (``state["ups"]``, see
        :mod:`yeto.rl.up_segments`) and restart the per-up clocks: ``t0`` (``budget.wall_h``,
        the NC breaker's wall cap) becomes now. Once per process: a trainer restart inside
        the same ``run up`` keeps it. ``run_t0`` keeps the run's first start (the spend,
        ``budget.usd``, stays cumulative over every up)."""
        if getattr(self, "_up_period", None) is not None:
            return
        now = time.time()
        ups = self.state.setdefault("ups", [])
        if not ups and "t0" in self.state:  # a run started before periods were recorded
            start = float(self.state["t0"])
            stamp = getattr(self, "_legacy_stamp", None)
            if stamp is not None and not self._legacy_stopped():  # no down seen: it may have billed the gap
                stamp = now
            seen = min(now, max(start, stamp or start))
            ups.append({"pid": None, "start": start, "seen": seen, "end": None, "legacy": True})
        # an earlier up that ended without a down: closed at its last poll, or now while
        # the trainer it started still runs (a --no-monitor up, a killed up: the trainer
        # is a detached process and billed the whole gap; `run down` closes it now too)
        # (capped at the next period's start: periods never overlap, the spend is their sum)
        trainer_ran = self._trainer_alive()
        for i, period in enumerate(ups):
            if period.get("end") is None:
                end = now if trainer_ran else up_segments.period_end(period, now)
                if i + 1 < len(ups) and ups[i + 1].get("start") is not None:
                    end = max(float(period.get("start") or 0.0), min(end, float(ups[i + 1]["start"])))
                period["end"] = end
        self.state.setdefault("run_t0", float(self.state.get("t0", now)))
        self.state["t0"] = now
        period = {"pid": os.getpid(), "start": now, "seen": now, "end": None}
        ups.append(period)
        self._up_period = period
        self.save()

    def _legacy_last_stamp(self) -> float | None:
        """The last time a run from before the periods were recorded is known to have run:
        the newest stamp its ``state.json`` holds (``steps`` - ``down`` among them -,
        ``procs`` starts, ``trainer_exit_seen`` / ``trainer_finished``); None for a state
        with ``ups`` or without stamps. Read when the state is first loaded: the stamps this
        ``run up`` writes before :meth:`begin_up` (``preflight``, ``sandbox``) are not the
        old run's and would stretch its period over the stopped gap. :meth:`begin_up` keeps
        it within ``[t0, now]``. Without it the legacy period would end at its start: the
        hours that run billed before the upgrade would drop out of the spend."""
        if self.state.get("ups") or "t0" not in self.state:
            return None
        stamps = [step.get("at") for step in (self.state.get("steps") or {}).values() if isinstance(step, dict)]
        stamps += [p.get("started") for p in (self.state.get("procs") or {}).values() if isinstance(p, dict)]
        stamps += [self.state.get("trainer_exit_seen"), self.state.get("trainer_finished")]
        seconds = [s for s in (_utc_seconds(x) for x in stamps) if s is not None]
        return max(seconds) if seconds else None

    def _legacy_stopped(self) -> bool:
        """The run from before the periods were recorded is known to bill no trainer since
        its last stamp: a successful ``run down`` (``steps.down.ok``) at or after the trainer's
        last launch, or a ``trainer_finished``. Not a dead local pid: ``run status`` on another
        control machine, a killed launcher whose Modal app still runs, an old up that crashed
        all look dead here while the trainer bills; those keep billing to now (never less)."""
        if _utc_seconds(self.state.get("trainer_finished")) is not None:
            return True
        steps = self.state.get("steps") or {}
        down = steps.get("down") if isinstance(steps.get("down"), dict) else {}
        down_at = _utc_seconds(down.get("at")) if down.get("ok") is True else None
        if down_at is None:
            return False
        tr = steps.get("trainer") if isinstance(steps.get("trainer"), dict) else {}
        procs = self.state.get("procs") or {}
        trp = procs.get("trainer") if isinstance(procs.get("trainer"), dict) else {}
        launches = [s for s in (_utc_seconds(tr.get("at")), _utc_seconds(trp.get("started"))) if s is not None]
        return not launches or down_at >= max(launches)

    def _legacy_period(self, now: float) -> dict[str, Any] | None:
        """Before the first ``run up`` of an upgraded run (no ``ups`` yet): the period
        :meth:`begin_up` will record for its pre-upgrade ups, ``t0`` to the last stamp
        (``run status`` then bills no trainer over the stopped gap after it). None, and so
        the whole time since ``trainer_t0`` as before, when that stamp may be short: no stamp,
        or a trainer launcher or ``yeto launch`` worker that still runs."""
        stamp = getattr(self, "_legacy_stamp", None)
        if stamp is None or "t0" not in self.state or not self._legacy_stopped() or self._trainer_alive():
            return None
        if _worker_alive(self.trainer_launch_name()):
            return None
        start = float(self.state["t0"])
        return {"pid": None, "start": start, "seen": min(now, max(start, stamp)), "end": None, "legacy": True}

    def _trainer_alive(self) -> bool:
        tr = (self.state.get("procs") or {}).get("trainer")
        return bool(tr) and _alive(tr.get("pid"))

    def up_heartbeat(self) -> None:
        """The monitor's poll: this up's period was seen now; the spend estimate is kept
        for ``run status`` and the dashboard."""
        ups = self.state.get("ups") or []
        if not ups or ups[-1].get("pid") != os.getpid() or ups[-1].get("end") is not None:
            return
        ups[-1]["seen"] = time.time()
        self.state["spend_usd_estimate"] = round(self.spend_usd(), 2)
        self.save()

    def end_up(self) -> None:
        """``run down``: every open up period ends now."""
        now = time.time()
        for period in self.state.get("ups") or ():
            if period.get("end") is None:
                period["end"] = now
        self.state["spend_usd_estimate"] = round(self.spend_usd(), 2)

    def wall_elapsed_s(self, now: float | None = None) -> float:
        """Seconds of this up (since ``t0``) for ``budget.wall_h``: the wall clock, but never
        less than this process measured on the monotonic clock. A wall clock stepped back
        (NTP) shrank the elapsed time and delayed the wall-clock breaker and the forced
        pool-stop by the step."""
        now = time.time() if now is None else now
        t0 = float(self.state.get("t0", now))
        wall = now - t0
        mono = getattr(time, "monotonic", None)
        if mono is None:
            return wall
        anchor = getattr(self, "_wall_anchor", None)
        if anchor is None or anchor[0] != t0:  # first look at this up's t0: anchor it
            anchor = self._wall_anchor = (t0, mono(), wall)
        return max(wall, anchor[2] + (mono() - anchor[1]))

    def uptime(self, now: float | None = None) -> dict[str, Any]:
        """:func:`yeto.rl.up_segments.uptime_summary` with the live spend and the time left
        on this up's ``budget.wall_h``."""
        now = time.time() if now is None else now
        out = up_segments.uptime_summary(self.state, now)
        out["spend_usd_estimate"] = round(self.spend_usd(), 1)
        wall_h = self.breaker_param("run.wall_clock", "hours")
        out["wall_h"] = wall_h
        out["wall_h_left"] = None  # None: no wall-clock cap, or no open up
        if wall_h is not None and out["current_up_open"] and "t0" in self.state:
            out["wall_h_left"] = round(wall_h - self.wall_elapsed_s(now) / 3600, 4)
        return out

    @property
    def in_process(self) -> bool:
        """``controller: "in_process"``: every pool in one RolloutController in this
        process (default ``"subprocess"``: one ``nc pool`` loop per pool)."""
        mode = str(self.cfg.get("controller") or "subprocess")
        if mode not in CONTROLLERS:
            raise ValueError(f"run config: controller {mode!r} is not 'subprocess' or 'in_process'")
        return mode == "in_process"

    def start_controller(self, *, ops: dict[str, Any] | None = None, **kw: Any) -> Any:
        """One :class:`~yeto.launch.pool.controller.RolloutController` over every pool
        (each built like its ``nc pool`` loop: state, events, scale file, lock), started
        in a thread of this process. ``ops``: per pool name, for tests."""
        from yeto.launch.pool.controller import RolloutController, nc_pool  # noqa: PLC0415

        slots = {}
        try:
            for ref in self.pools:
                slots[ref.name] = nc_pool(ref.load(), ref.config, ops=(ops or {}).get(ref.name))
                self._restart_stopped_pool(ref.name, slots[ref.name].pool)
        except BaseException:
            for slot in slots.values():
                if slot.lock is not None:
                    slot.lock.close()
            raise
        interval = min(r.interval_s for r in self.pools)
        section = rollout_section(self.cfg)
        kw.setdefault("stall_s", float(self.breaker_param("run.rollout_stall_alert", "limit_s") or 0))
        kw.setdefault("drain_on_exit", bool(section.get("drain_on_exit", True)))
        self.controller = RolloutController(slots, stop_flag=self.dir / "STOP", interval=interval, log=_log, **kw)
        self.controller.start()
        if not self.controller.drain_on_exit:
            _log("rollout.drain_on_exit is false: if this process exits, the sampler machines keep running "
                 "(and billing) undriven until `yeto run up` of this config adopts them, or `yeto run down`")
            atexit.register(self._undriven_banner)
        _log(f"rollout controller: {len(slots)} pool(s) in this process ({', '.join(slots)})")
        return self.controller

    def _undriven_banner(self) -> None:
        """At exit with ``rollout.drain_on_exit`` false: what is left running, undriven."""
        controller = self.controller
        if controller is None or controller.stopped:
            return
        live = {name: [m.ticket for m in pool.live()] for name, pool in controller.pools.items()}
        live = {name: tickets for name, tickets in live.items() if tickets}
        if live:
            _log(f"exit without a drain (rollout.drain_on_exit false): STILL BILLING, undriven: {live}; "
                 "adopt them with `yeto run up --config <run.json>` or release them with `yeto run down`")

    def _objects(self):
        from yeto.cloud.s3_store import head_objects  # noqa: PLC0415

        return head_objects(self.cfg["store"], environ=self.env)

    def buffer_objects(self) -> int:
        return len(self.buffer_keys() or ())

    def wait_first_group(self) -> str | None:
        """None once the store holds a group; else why the wait ended (a breach, the timeout).
        A fleet waits island by island (:meth:`FleetRuntime.wait_first_group`)."""
        if self.fleet_plan() is not None:
            return self.fleet_rt().wait_first_group()
        fg = self.cfg.get("first_group") or {}
        timeout = self.breaker_param("run.first_group", "timeout_s")
        start = time.monotonic()  # a local interval: an NTP step must not end/stretch it
        deadline = math.inf if timeout is None else start + timeout
        while time.monotonic() < deadline:
            n = self.buffer_objects()
            if n:
                self.mark("first_group", objects=n)
                return None
            why = self.breach(trainer=False)
            if why:
                return why
            time.sleep(float(fg.get("poll_s", 60)))
        _log("first group: timed out")
        self.breaker_event("run.first_group", value=time.monotonic() - start, limit=timeout,
                           detail="no sample-buffer object reached the store")
        return "first group timed out"

    def trainer_restart(self) -> bool:
        """An earlier trainer launch of this run exists (this ``run up`` restarts it)."""
        return bool((self.state.get("procs") or {}).get("trainer") or (self.state.get("steps") or {}).get("trainer"))

    def trainer_argv(self, now: float | None = None) -> list[str]:
        """The trainer's ``yeto launch`` argv. On a restart it adds
        ``--rl-ledger-set-aside-before <now>``: a start from the base (no cut / release to
        continue) sets aside the ledger the earlier launch left on the island's volume
        (``ledger.before-<ts>``) instead of failing on it (LedgerError) after the start-up."""
        argv = [str(a) for a in self.cfg["trainer"]["argv"]]
        if self.trainer_restart() and not any(a.startswith("--rl-ledger-set-aside-before") for a in argv):
            argv += ["--rl-ledger-set-aside-before", f"{time.time() if now is None else now:.0f}"]
        if self.continue_from_release(argv):
            argv += ["--rl-init-from-release", "latest"]
        if self.cluster_preflight_passed and "--skip-preflight" not in argv:
            from yeto.launch.preflight.gate import commands_for_launch_argv  # noqa: PLC0415

            if commands_for_launch_argv(argv):  # an ssh: trainer: run up ran it before ordering anything
                argv.append("--skip-preflight")
        return argv

    def continue_from_release(self, argv: list[str]) -> bool:
        """A trainer restart under ``--rl-round-cuts off`` continues from the store's
        newest release (``--rl-init-from-release latest``): there is no round cut,
        and a start from the base would continue the release numbering with base
        weights. Not on the first launch, not when the store holds no release yet,
        not when the argv names a release itself."""
        if not self.trainer_restart() or _argv_value(argv, "--rl-round-cuts") != "off":
            return False
        if any(a == "--rl-init-from-release" or a.startswith("--rl-init-from-release=") for a in argv):
            return False
        if self.fleet_plan() is not None:
            return self._fleet_continue_from_release()
        if not self.store_has_release():
            _log("trainer restart with --rl-round-cuts off: the store holds no policy release yet, "
                 "so the trainer starts from the base checkpoint")
            return False
        if not getattr(self, "_init_release_logged", False):
            self._init_release_logged = True
            _log("trainer restart with --rl-round-cuts off: adding --rl-init-from-release latest "
                 "(the weights of the newest release; optimizer state, LR schedule position and "
                 "RNG start fresh)")
            self.mark("trainer_init_from_release", selector="latest")
        return True

    def _fleet_continue_from_release(self) -> bool:
        """A fleet's ``--rl-round-cuts off`` restart: each island's releases are in its own
        ``trainer<k>/``, but ``--rl-init-from-release`` makes every trainer node read the
        run store's root (``init_from_release``), which holds none, so the flag would fail
        every node. It is not added; the islands holding a release are logged (once)."""
        if getattr(self, "_init_release_logged", False):
            return False
        self._init_release_logged = True
        found = self.fleet_rt().releases()
        held = [self.fleet_rt().label(k) for k, has in sorted(found.items()) if has]
        if not held:
            _log("trainer restart with --rl-round-cuts off: no island's sub-directory holds a policy release yet, "
                 "so the trainers start from the base checkpoint")
            return False
        _log(f"WARNING: trainer restart with --rl-round-cuts off: {', '.join(held)} hold(s) policy releases in their "
             "trainer<k>/, but --rl-init-from-release reads the run store's root, not an island's sub-directory: "
             "it is not added, and the trainers start from the base checkpoint (their release numbering goes on)")
        self.mark("trainer_init_from_release", selector=None, islands_with_release=held)
        return False

    def store_has_release(self) -> bool:
        """The run store holds a trained policy release, judged as ``--rl-init-from-release
        latest`` and the trainer's resumed numbering judge it (``policy_release.
        trained_releases`` of every ``RELEASE.json`` and ``LATEST.json``): a record
        without LATEST counts, a torn or derived one does not. A failed listing or read
        raises: guessing would either start a restart from the base or name a release
        that is not there."""
        from yeto.rl.engine.policy_release import records_in_objects, trained_releases  # noqa: PLC0415

        try:
            records = records_in_objects(self._objects())
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(f"--rl-round-cuts off restart: cannot list the store's policy releases "
                               f"({type(error).__name__}: {error})"[:400]) from error
        return bool(trained_releases(records))

    def trainer(self) -> None:
        self._spawn("trainer", self.yeto("launch", *self.trainer_argv()))
        for key in ("trainer_exit_seen", "trainer_finished"):  # of an earlier launcher of this run
            self.state.pop(key, None)
        if self.fleet_plan() is not None:  # a fleet-wide stop of the launch this one replaces no longer holds
            self.fleet_rt().on_trainer_start()
        self.state.setdefault("trainer_t0", time.time())
        self.mark("trainer")

    def drain_controller(self) -> bool:
        """The exit drain of the in-process controller (``rollout.drain_on_exit``), run
        while ``main`` still runs. Left to the ``atexit`` hook it ran during interpreter
        shutdown, after the Modal client had closed: a tick in flight failed with
        ``ClientClosed``, the drain waited on it forever, and the process hung holding
        the pool with its sampler still billing.

        A SIGTERM during the drain is logged, not obeyed (it would cut the drain short).
        A drain that fails prints the billing banner and is not retried at exit (the
        controller leaves the active list either way). False: the drain failed."""
        controller = getattr(self, "controller", None)
        if controller is None or controller.stopped or not controller.drain_on_exit:
            return True
        from yeto.launch.pool import controller as ctl  # noqa: PLC0415

        restore = None
        if threading.current_thread() is threading.main_thread():
            restore = signal.signal(signal.SIGTERM, _log_sigterm_during_drain)
        try:
            ctl.drain_active("process exit")
            return True
        except Exception as error:  # noqa: BLE001 - reported with the billing banner, never re-raised
            _log(f"CRITICAL: the exit drain of the sampler pools FAILED ({error!r})"[:600])
            _log(f"CRITICAL: {self.pools_phrase()} may STILL BE BILLING: release it now with "
                 "`yeto run down --config <run.json>` and check that it reports verified")
            return False
        finally:
            with ctl._ACTIVE_LOCK:  # noqa: SLF001 - no second drain at interpreter exit
                if controller in ctl._ACTIVE:  # noqa: SLF001
                    ctl._ACTIVE.remove(controller)  # noqa: SLF001
            if restore is not None:
                signal.signal(signal.SIGTERM, restore)

    def _restart_stopped_pool(self, name: str, pool: Any) -> None:
        """A pool an earlier stop of this run left stopped (``yeto run down``, a process
        exit drain) and with no live member: this ``run up`` drives it again. Kept
        stopped, the controller skips it at every tick, so no sampler is ordered and the
        trainer waits for samples that never come. A pool with live members is left as
        it is (the stop has not finished), and so is one that finished its run, and (a
        fleet) one whose island an earlier trip of its own stopped."""
        if not pool.stopped or getattr(pool, "finished", None):
            return
        if self.fleet_plan() is not None:
            refused = self.fleet_rt().pool_restart_refusal(name)
            if refused:
                _log(f"pool {name}: {refused}")
                return
        released = getattr(pool, "manual_released", None) or {}
        # a member released by hand (`nc down` / `delete-ticket` with no loop running) is no
        # unfinished stop: the first tick reads its ticket ended and drops it
        live = [m for m in pool.live() if m.ticket not in released]
        if live:
            tickets = [m.ticket for m in live]
            raise RunUpRefused(
                f"pool {name} is stopped ({pool.stopped}) but still holds live member(s) {tickets}: an earlier "
                "stop has not finished releasing them, and a stopped pool orders nothing, so the trainer would "
                "wait for samples that never come. Run `yeto run down --config <run.json>` first (it waits "
                "for the release), then `yeto run up` again")
        budget = getattr(pool, "budget", None)
        refused = None if budget is None else budget.refusal(pool.ecfg, billed_nodes=0, target=pool.config.target,
                                                            hourly_ucr=pool.config.hourly())
        if refused:
            raise RunUpRefused(
                f"pool {name} would refuse every order: {refused}. elastic.max_orders and "
                "elastic.max_run_spend_ucr / max_run_spend_usd count over the WHOLE run (every earlier run up "
                "included), so a restarted run starts at the totals it already reached. Raise the limit in the "
                "pool config (or start a new run name), then `yeto run up` again")
        _log(f"pool {name}: its state says stopped ({pool.stopped}) by an earlier stop of this run and no "
             "member is live: this run up drives it again")
        pool.stopped = None
        pool._ended_done = False  # noqa: SLF001 - its next stop frees the pool's app again
        pool._event("pool_restarted", why="run up")  # noqa: SLF001

    def _restart_stopped_pool_states(self, refs: list[Any]) -> None:
        """``controller: subprocess``: :meth:`_restart_stopped_pool` on each pool's state
        file before its ``nc pool`` loop is spawned (that loop exits at once on a state that
        says stopped, so no sampler would be ordered). Every pool is checked under its pool
        lock first; the states are written only once none refused (a refusal raises with no
        state changed). A pool whose lock is held runs already and is left alone."""
        from yeto.launch.nc import cli, pool_run  # noqa: PLC0415
        from yeto.launch.pool.island_pool import IslandPool  # noqa: PLC0415

        held: list[tuple[Any, Any, Path]] = []
        locks = []
        try:
            for ref in refs:
                cfg = ref.load()
                state = cli.pool_state_path(cfg)
                if not state.exists():
                    continue
                lock = pool_run.acquire_lock(pool_run.lock_path(cfg), run=cli.run_id(cfg), role="run-up")
                if lock is None:
                    continue
                locks.append(lock)
                try:  # another run's state (same shard_base): archived when stopped, else refused
                    if pool_run.claim_state(cfg) is not None:
                        continue
                except pool_run.PoolStateOwnerError as error:
                    raise RunUpRefused(f"pool {ref.name}: {error}") from None
                pool = IslandPool.restore(state, cli.pool_config(cfg, allowed=None), None,  # type: ignore[arg-type]
                                          log=cli._log, events=pool_run.event_log(cfg))  # noqa: SLF001
                if not pool.stopped or getattr(pool, "finished", None):
                    continue
                self._restart_stopped_pool(ref.name, pool)  # raises before it changes anything
                held.append((ref, pool, state))
            for _, pool, state in held:
                pool.save(state)
        finally:
            for lock in locks:
                lock.close()

    def dashboard_args(self) -> list[str]:
        """``--store`` plus the run's local records (``--run``: the launcher's fleet record
        names the trainer's GPU, so its cost is priced), the cost bar's cap
        (``budget.usd``) and an optional price table (``dashboard.prices``)."""
        dash = self.cfg.get("dashboard") or {}
        # the launcher writes its records under ~/.yeto/runs/<its --cluster-prefix>
        argv = ["--store", self.cfg["store"], "--run", self.trainer_launch_name(),
                "--up-state", str(self.state_path)]
        usd = (self.cfg.get("budget") or {}).get("usd")
        if usd is None and self.breaker_param("run.budget", "usd") is not None:  # set by the breakers section
            usd = f"{self.breaker_param('run.budget', 'usd'):g}"
        if usd:
            argv += ["--budget", str(usd)]
        if dash.get("prices"):
            argv += ["--prices", os.path.expanduser(str(dash["prices"]))]
        return argv

    def dashboard(self) -> None:
        port = str((self.cfg.get("dashboard") or {}).get("port", 8795))
        self._spawn("dashboard", self.yeto("dashboard", "serve", *self.dashboard_args(), "--port", port))
        self.mark("dashboard", url=f"http://127.0.0.1:{port}/")

    # -- guards
    def spend_usd(self) -> float:
        if self.fleet_plan() is not None:  # pools + each island's trainer + the syncer
            return self.fleet_rt().spend().total_usd
        now = time.time()
        nc = sum(self.node_hours(i, now) * self.node_usd_per_h(i) for i in range(len(self.pools)))
        rate = up_consistency.number_or_default(self.cfg.get("trainer"), "usd_per_h", 0)
        return nc + self.trainer_hours(now) * rate

    def trainer_hours(self, now: float | None = None) -> float:
        """Billed trainer hours: from ``trainer_t0`` (its first launch) to now, counted only
        inside the ``run up`` periods (``ups``): the stopped gaps between a ``down`` and the
        next ``up`` bill no trainer. Inside one up: ``now - trainer_t0`` as before. A trainer that
        finished (``trainer_finished``, its launcher gone) bills up to that stamp only."""
        now = time.time() if now is None else now
        t0 = self.state.get("trainer_t0")
        if t0 is None:
            return 0.0
        t0 = float(t0)
        finished = _utc_seconds(self.state.get("trainer_finished"))
        if finished is not None and not self._trainer_alive():  # SUCCEEDED: its app bills no more
            now = min(now, max(finished, t0))
        ups = [p for p in self.state.get("ups") or () if p.get("start") is not None]
        if not ups:  # a run from before the periods were recorded
            legacy = self._legacy_period(now)
            if legacy is None:
                return max(0.0, now - t0) / 3600
            ups = [legacy]  # the period the next ``run up`` records (begin_up), not the stopped gap
        trainer_ran = None

        def end(i: int, p: dict[str, Any]) -> float:
            nonlocal trainer_ran
            e = up_segments.period_end(p, now)
            if i == len(ups) - 1 and p.get("end") is None:  # an up gone without a down
                if trainer_ran is None:  # (--no-monitor, killed): its detached trainer bills to now
                    trainer_ran = self._trainer_alive()
                if trainer_ran:
                    e = max(e, now)
            return min(e, now)  # a finished trainer: no period bills past its finish

        return sum(max(0.0, end(i, p) - max(float(p["start"]), t0)) for i, p in enumerate(ups)) / 3600

    def node_usd_per_h(self, i: int = 0) -> float:
        """One node hour of pool ``i``: its ``usd_per_h``, else the price its provider config
        states (``price_usd_h`` of a Modal / Daytona / Sky pool, the one the pool's own budget
        charges), else ``budget.nc_usd_per_h``."""
        price = self.pools[i].usd_per_h
        if price is None:
            try:
                from yeto.launch.nc.cli import pool_price  # noqa: PLC0415

                price, currency = pool_price(self.pools[i].load())
                if currency == "ucr":  # National Compute: its rate stays budget.nc_usd_per_h
                    price = None
            except Exception:  # noqa: BLE001 - an unreadable pool config: the NC rate
                price = None
        return up_consistency.number_or_default(self.cfg.get("budget"), "nc_usd_per_h", 24) if price is None else price

    def nc_node_hours(self, now: float | None = None) -> float:
        """Billed rollout node hours of this run, every pool (see :meth:`node_hours`)."""
        return sum(self.node_hours(i, now) for i in range(len(self.pools)))

    def node_hours(self, i: int = 0, now: float | None = None) -> float:
        """Billed node hours of pool ``i``: each pool state member from its grant (a ticket
        never granted bills nothing) to its release (``charged_to`` of a gone member) or now,
        counted from the run's first start (``run_t0``: the spend spans every ``run up``).
        Without a readable pool state: the run's cumulative up time (every up period; the
        whole time since ``run_t0`` before periods were recorded) times the pool target
        (the scaled one; every node assumed billed)."""
        now = time.time() if now is None else now
        t0 = float(self.state.get("run_t0", self.state.get("t0", now)))
        members = self._members(i)
        if members is None:  # no state yet / unreadable: the conservative estimate
            try:
                nodes = max(1, self.pool_target({}, i))
            except Exception:  # noqa: BLE001
                nodes = 1
            if self.state.get("ups"):
                return up_segments.uptime_summary(self.state, now)["cumulative_runtime_h"] * nodes
            return max(0.0, now - t0) / 3600 * nodes
        hours = 0.0
        for m in members:
            if m.get("granted") is None:
                continue
            # a corrupt stamp must not raise out of the breaker (it ended the monitor and the
            # budget guard) nor bill less: an unreadable grant bills from t0, an unreadable
            # release to now
            granted = _epoch(m["granted"])
            start = t0 if granted is None else max(granted, t0)
            charged = _epoch(m.get("charged_to")) if m.get("charged_to") else now
            end = (now if charged is None else charged) if m.get("state") == "gone" else now
            hours += max(0.0, end - start) / 3600
        return hours

    def _nc_dir(self) -> Path:
        return Path(self.env.get("YETO_NC_STATE_DIR", "~/.yeto/nc")).expanduser()  # tunnel.state_dir()

    def pool_doc(self, i: int = 0) -> dict[str, Any] | None:
        """The state file of pool ``i`` (None: no readable state)."""
        try:
            pool_cfg = self.pools[i].load()
            path = self._nc_dir() / f"pool-{pool_cfg['shard_base']}.json"
            doc = json.loads(path.read_text())
            from yeto.launch.nc.pool_run import foreign_owner  # noqa: PLC0415

            if foreign_owner(pool_cfg, doc) is not None:
                return None  # another run's state (same shard_base): not this run's members or spend
        except Exception:  # noqa: BLE001
            return None
        from yeto.launch.pool.island_pool import PoolStateFormatError, check_state_format  # noqa: PLC0415

        try:
            if isinstance(doc, dict):
                check_state_format(doc, path)
        except PoolStateFormatError as error:  # written by a newer yeto: not read, and said so once
            if str(path) not in _FORMAT_WARNED:
                _FORMAT_WARNED.add(str(path))
                print(f"[yeto run] WARNING: {error}; this pool's members, spend and target are not shown",
                      file=sys.stderr)
            return None
        return doc

    def _members(self, i: int) -> list[dict[str, Any]] | None:
        doc = self.pool_doc(i)
        try:
            return list(doc["members"]) if doc is not None else None
        except (KeyError, TypeError):
            return None

    def pool_members(self) -> list[dict[str, Any]] | None:
        """The members of every pool's state file (None: no pool state readable)."""
        found = [self._members(i) for i in range(len(self.pools))]
        if all(m is None for m in found):
            return None
        return [m for members in found if members for m in members]

    def billed_nodes(self) -> int | None:
        """Pool members billed now (granted, not gone), every pool; None without a pool state."""
        members = self.pool_members()
        if members is None:
            return None
        return sum(1 for m in members if m.get("state") != "gone" and m.get("granted") is not None)

    def pool_target(self, doc: dict[str, Any] | None, i: int = 0) -> int:
        """Pool ``i``'s wanted sampler count: its scale file (this run's), else the state, else the config."""
        pool_cfg = self.pools[i].load()
        try:
            scale = json.loads((self._nc_dir() / f"pool-{pool_cfg['shard_base']}.scale.json").read_text())
            if isinstance(scale, dict) and scale.get("run") == pool_cfg.get("run") and "target" in scale:
                return int(scale["target"])
        except (OSError, ValueError):
            pass
        return int((doc or {}).get("target", pool_cfg.get("target", 1)))

    def container_limit(self) -> int | None:
        """The sandbox container alert line: ``max_run_containers``, raised to
        ``max_containers_per_sampler`` x the pool's billed nodes (it follows a scale)."""
        fixed = self.breaker_param("run.sandbox_containers", "max_run_n") or None
        per = self.breaker_param("run.sandbox_containers", "max_per_sampler_n") or None
        if per is None:
            return fixed
        nodes = self.billed_nodes()
        if nodes is None:
            nodes = sum(self.pool_target(None, i) for i in range(len(self.pools)))
        scaled = per * max(1, nodes)
        return max(fixed or 0, scaled)

    def sandbox_hosts(self) -> list[str]:
        """The docker endpoints of the sandbox hosts (``sandbox.docker_host``: one, a list,
        or ``"envs"`` for every host of ``sandbox.envs``)."""
        value = (self.cfg.get("sandbox") or {}).get("docker_host")
        if not value:
            return []
        if value == "envs":
            if not self.envs:
                raise ValueError('sandbox.docker_host "envs" needs sandbox.envs')
            from yeto.sandbox.envs import load_envs  # noqa: PLC0415

            return [h.docker_host for h in load_envs(self.envs, run=self.run).hosts]
        return [str(v) for v in value] if isinstance(value, list) else [str(value)]

    def _host_containers(self, host: str, *, every: bool = False) -> list[str] | None:
        try:  # an ssh hang to the sandbox host must not end the monitor (and its guards)
            out = subprocess.run(["docker", "-H", host, "ps", "-aq" if every else "-q", "--filter",
                                  f"label=ai.yeto.run={self.run}"],
                                 env=self.env, capture_output=True, text=True, timeout=60, check=False)
        except (subprocess.TimeoutExpired, OSError) as error:
            _log(f"sandbox: docker ps on {host} failed: {type(error).__name__}")
            return None
        return out.stdout.split() if out.returncode == 0 else None

    def _sweep_trials(self, host: str) -> None:
        """Remove this run's trial containers and their compose networks on ``host``
        (``trial_labels.sweep_run``); a failure is logged, the count after it decides."""
        from yeto.sandbox.docker import DockerHost, subprocess_runner  # noqa: PLC0415
        from yeto.sandbox.trial_labels import sweep_run  # noqa: PLC0415

        base = subprocess_runner(600.0)
        env = {k: v for k, v in self.env.items() if k != "DOCKER_CONTEXT"}
        env["DOCKER_HOST"] = host
        dk = DockerHost(host, runner=lambda argv, _env, stdin: base(argv, env, stdin))
        try:
            done = sweep_run(dk, self.run)
        except Exception as error:  # noqa: BLE001 - the container count below verifies
            _log(f"sandbox: trial sweep on {host} failed: {type(error).__name__}: {error}")
            return
        _log(f"sandbox: {host}: removed {len(done['containers'])} trial container(s), "
             f"networks of {len(done['networks'])} trial(s)")

    def left_trial_containers(self, hosts: list[str]) -> tuple[int, list[str]]:
        """After the down's sweep: this run's trial containers still on ``hosts``, stopped
        ones too (they keep their compose networks), and the hosts that did not answer."""
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

        with ThreadPoolExecutor(max_workers=max(1, min(16, len(hosts)))) as pool:
            counts = list(pool.map(lambda h: self._host_containers(h, every=True), hosts))
        return (sum(len(c) for c in counts if c is not None),
                [h for h, c in zip(hosts, counts) if c is None])

    def run_containers(self) -> int | None:
        """This run's running trial containers on every sandbox host (asked at once);
        None when no host answered."""
        hosts = self.sandbox_hosts()
        if not hosts:
            return None
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

        with ThreadPoolExecutor(max_workers=min(16, len(hosts))) as pool:
            counts = list(pool.map(self._host_containers, hosts))
        known = [len(c) for c in counts if c is not None]
        return sum(known) if known else None

    def breach(self, *, trainer: bool = True) -> str | None:
        """Why the run must stop, else None. ``trainer=False`` (the first-group wait, before
        this invocation starts the trainer) skips the trainer log: on a resumed ``run up`` it is
        the log of the previous, stopped launcher, whose failure is already handled. Each
        reason is one run-layer breaker (``yeto.breakers.layers.run``), reported as an
        ``rl_breaker`` trip."""
        why = None
        tripped: tuple[str, float | None, float | None] | None = None  # breaker, value, limit
        controller = getattr(self, "controller", None)
        if controller is not None and controller.check():  # the in-process pools stopped being driven
            why = controller.check()
            self.breaker_event("run.controller_exit", value=None, limit=None, detail=why)
            return why
        wall_h = self.breaker_param("run.wall_clock", "hours")
        usd = self.breaker_param("run.budget", "usd")
        if wall_h is not None and "t0" in self.state and self.wall_elapsed_s() > wall_h * 3600:
            why, tripped = "wall clock", ("run.wall_clock", self.wall_elapsed_s() / 3600, wall_h)
        elif (spend := self.spend_usd()) > (up_consistency.NO_CAP_USD if usd is None else usd):
            why, tripped = f"budget ${spend:.2f}", ("run.budget", spend, usd)
        # the trainer: only its launcher's exit. An OOM or a fatal island error ends the
        # launcher; a hang is Miles' --rl-stall-timeout and the launcher's own business.
        tr = (self.state.get("procs") or {}).get("trainer") if trainer else None
        if why is None and tr and not _alive(tr["pid"]):
            try:
                tail = _log_tail(tr["log"], int(tr.get("offset", 0)), 20_000)
            except OSError:
                tail = ""
            done = FINISHED.findall(tail)
            if done and done[-1][0] == "SUCCEEDED" and done[-1][1] == "0":
                self.state["trainer_finished"] = self.state.get("trainer_finished") or _now()
            if not done and not _worker_alive(self.trainer_launch_name()):  # a killed log streamer: the worker may still run
                why, tripped = "trainer: launcher exited without a result", ("run.trainer_exit", None, None)
            elif done and (done[-1][0] == "FAILED" or done[-1][1] != "0"):
                why = f"trainer: launcher finished {done[-1][0]} (exit code {done[-1][1]})"
                tripped = ("run.trainer_exit", float(done[-1][1]), 0.0)
        if why is None and tr and _alive(tr["pid"]):
            why = self.silence(time.time())
            if why:
                tripped = ("run.trainer_silence", self._quiet_s, self.silence_limit_s())
        limit = n = None
        if why is None:
            # a view only: an envs file or pool config that does not read this poll must not
            # raise out of the breaker (it ended the monitor, and the budget / wall-clock guard)
            try:
                limit = self.container_limit()
                n = self.run_containers() if limit else None
            except Exception as error:  # noqa: BLE001 - counted again next poll
                _log(f"sandbox: container count failed: {type(error).__name__}: {error}"[:300])
                limit = None
        if why is None and limit:
            # an alert, not a breach: the containers belong to the NC samplers, so stopping the
            # trainer frees none of them; an evicted island's trials outlive it while its
            # replacement starts its own, so the count passes the limit in a normal replacement
            over = n is not None and n > int(limit)
            if over:
                _log(f"WARNING: {n} sandbox containers of this run > {limit} (not a breach)")
                self.state["sandbox_over"] = {"at": _now(), "containers": n}
                self.save()
            if over != getattr(self, "_containers_over", False):
                self.breaker_event("run.sandbox_containers", value=n, limit=limit, state="trip" if over else "clear",
                                   detail=f"{n} sandbox containers of this run")
            self._containers_over = over
        if why:
            _log(f"BREAKER: {why}")
            if self.fleet_plan() is not None and tripped:  # the fleet's one launch: every island stops
                event = (self.fleet_rt().budget_event(tripped[1], tripped[2], why) if tripped[0] == "run.budget"
                         else self.breaker_event(tripped[0], value=tripped[1], limit=tripped[2], detail=why))
                self.fleet_rt().record(event)
            else:
                event = self.breaker_event(tripped[0], value=tripped[1], limit=tripped[2], detail=why) if tripped else {}
            self.state["breaker"] = {"at": _now(), "why": why}
            if event.get("stop_reason"):
                self.state["breaker"]["stop_reason"] = event["stop_reason"]
            self.save()
        return why

    def shard_problem(self, pool: int = 0) -> str | None:
        """The trainer reads only the shards of its ``--rl-buffer-shards`` (a shard outside
        it is never trained on), so the pool may only order shards the trainer names: with
        that flag set, the pool needs a ``shard_limit`` and every ``<shard_base>-<i>`` below
        it in the trainer's list (each node and each lease-end replacement takes a new
        shard; plan the list for the largest scale plus the replacements). In a fleet the
        pool's island trainer reads its ``--rl-island-buffer-shards k:...``
        (:func:`yeto.rl.up_fleet.shard_problem`)."""
        argv = [str(a) for a in self.cfg["trainer"]["argv"]]
        plan = self.fleet_plan()
        if plan is not None:
            from yeto.rl import up_fleet  # noqa: PLC0415

            return up_fleet.shard_problem(plan, argv, self.pools[pool].name, self.pools[pool].load())
        named = None
        for i, a in enumerate(argv):
            if a == "--rl-buffer-shards" and i + 1 < len(argv):
                named = argv[i + 1]
            elif a.startswith("--rl-buffer-shards="):
                named = a.split("=", 1)[1]
        if named is None:
            return None
        shards = {x for x in named.split(",") if x}
        from yeto.launch.nc import cli as nc_cli  # noqa: PLC0415

        pool_cfg = self.pools[pool].load()
        limit = nc_cli.pool_shard_limit(pool_cfg)
        if limit is None:
            return (f"the trainer reads --rl-buffer-shards ({len(shards)} shards) but the pool config has no "
                    f"shard_limit: a scale up or a replacement past them writes groups nobody trains on")
        missing = [f"{pool_cfg['shard_base']}-{i}" for i in range(limit)
                   if f"{pool_cfg['shard_base']}-{i}" not in shards]
        if missing:
            return (f"pool shard_limit {limit} but the trainer's --rl-buffer-shards lacks {missing[:5]}"
                    f"{' ...' if len(missing) > 5 else ''}: those nodes' groups would never be trained on")
        return None

    def _trainer_flag(self, name: str) -> str | None:
        argv = [str(a) for a in self.cfg["trainer"]["argv"]]
        value = None
        for i, a in enumerate(argv):
            if a == name and i + 1 < len(argv):
                value = argv[i + 1]
            elif a.startswith(name + "="):
                value = a.split("=", 1)[1]
        return value

    def rollout_fields_problem(self, pool: int = 0) -> str | None:
        """A trainer with ``--rl-loss score_centering`` trains on the rollout top-K
        log-probs, which the sampler records only with ``rollout_fields.top_logprobs_num``
        (K, ``--rl-rollout-top-logprobs-num``, default 128) in its spec. Without it the
        dry-runs pass and the trainer dies on its first group (Miles: "Rollout top-k
        logprobs collection requires rollout_topk_token_ids") after the islands bill."""
        if self._trainer_flag("--rl-loss") != "score_centering":
            return None
        from yeto.rl.algos.score_centering import DEFAULT_TOP_LOGPROBS  # noqa: PLC0415

        want = int(self._trainer_flag("--rl-rollout-top-logprobs-num") or DEFAULT_TOP_LOGPROBS)
        pool_cfg = self.pools[pool].load()
        unread = (f"the trainer's --rl-loss score_centering needs the sampler spec's top-{want} log-probs but the "
                  f"pool's sampler spec {pool_cfg.get('spec') or '(none)'} cannot be read: nothing can be checked")
        if not pool_cfg.get("spec"):
            return unread  # fail closed
        try:
            spec = json.loads(Path(os.path.expanduser(str(pool_cfg["spec"]))).read_text())
        except (OSError, ValueError):
            return unread
        have = int(((spec.get("rollout_fields") or {}) if isinstance(spec, dict) else {}).get("top_logprobs_num") or 0)
        if have != want:
            return (f"the trainer's --rl-loss score_centering trains on the rollout top-{want} log-probs but the "
                    f"sampler spec {pool_cfg['spec']} has rollout_fields.top_logprobs_num {have or 'unset'}: "
                    f"set it to {want} (the trainer would fail on its first group)")
        return None

    def consistency_problem(self, pool: int = 0) -> str | None:
        """The values the trainer and this pool's samplers must share (group size,
        row widths, R3, expert selection, store, release directory;
        :mod:`yeto.rl.up_consistency`). Each side's own dry run passes a mismatch;
        it would surface after the trainer island loaded its weights, or never."""
        from yeto.rl import up_consistency  # noqa: PLC0415

        try:
            pool_cfg = self.pools[pool].load()
        except (OSError, ValueError):
            return None  # the pool dry-run reports an unreadable pool config
        found = up_consistency.problems(str(self.cfg["store"]), [str(a) for a in self.cfg["trainer"]["argv"]],
                                        pool_cfg, data_path=self.pools[pool].data_path)
        plan = self.fleet_plan()
        if plan is not None:  # a fleet pool writes its own island's trainer<k>, not any trainer<k>
            from yeto.rl import up_fleet  # noqa: PLC0415

            found += [p for p in (up_fleet.store_problem(plan, self.pools[pool].name, pool_cfg),) if p]
        return "; ".join(found) if found else None

    def env_file_problem(self) -> str | None:
        """A pool's ``agentic_env_file`` the samplers could not start with, or would read
        differently from the breaker views (:func:`yeto.breakers.rules_facts.env_file_problems`:
        unreadable, a line docker refuses, a quoted value a breaker reads); None when all are fine."""
        from yeto.breakers.rules_facts import env_file_problems  # noqa: PLC0415

        found = []
        for ref in self.pools:
            try:
                cfg = ref.load()
            except (OSError, ValueError):
                continue  # the pool dry-run reports an unreadable pool config
            found += [p + (f" (pool {ref.name})" if len(self.pools) > 1 else "") for p in env_file_problems(cfg)]
        return "; ".join(found) if found else None

    def rules_problem(self) -> str | None:
        """The dependency rules between the run's breaker thresholds
        (:mod:`yeto.breakers.rules`, D1-D21: the stall timer and the silence backstop,
        the samplers' backlog caps against the release cadence, dispense coverage,
        nested timeouts, ...): a warning or a note is a preflight line; an error
        refuses the run before anything is ordered. First the cap the pools'
        ``max_backlog_groups: "auto"`` samplers get (:attr:`auto_backlog`)."""
        from yeto import breakers  # noqa: PLC0415
        from yeto.breakers.rules import format_finding  # noqa: PLC0415

        self.log_auto_backlog()
        try:
            resolved = breakers.resolved(self.cfg, strict=False)  # conflicts: refused by their own check
        except ValueError as error:
            return str(error)
        errors = []
        for f in breakers.preflight(resolved, run_config=self.cfg):
            if f.level == "error":
                errors.append(format_finding(f))
            else:
                _log(f"preflight: {'WARNING' if f.level == 'warn' else 'note'}: {format_finding(f)}")
        return "; ".join(errors) if errors else None

    def log_auto_backlog(self) -> None:
        """The preflight line of the ``max_backlog_groups: "auto"`` cap (nothing without one)."""
        try:
            pools = [(ref.load(), ref.data_path) for ref in self.pools]
        except (OSError, ValueError):
            return  # the pool dry-run reports an unreadable pool config
        auto = self.auto_backlog
        if auto is not None:
            _log(f"preflight: max_backlog_groups auto -> {auto['groups']} per sampler ({auto['why']}; "
                 "set a number in the sampler spec to override)")
        elif any(up_consistency.is_auto_backlog(up_consistency.effective_spec(cfg)) for cfg, _path in pools):
            from yeto.rl.engine import sample_roles  # noqa: PLC0415

            _log(f"preflight: max_backlog_groups auto -> fallback {sample_roles.AUTO_BACKLOG_FALLBACK_GROUPS} per "
                 "sampler (the trainer is not a single buffer trainer, or no pool has a target)")

    def log_compaction(self, pool: int = 0) -> None:
        """Preflight lines of the pool's ``agentic.compaction`` (nothing without it) and a
        warning when one buffer group can exceed one object PUT
        (:func:`yeto.rl.up_consistency.payload_size_warning`, with or without compaction)."""
        from yeto.rl import up_consistency  # noqa: PLC0415

        plan = self.fleet_plan()
        island = plan.island_of_pool(self.pools[pool].name) if plan is not None else None
        try:
            lines = up_consistency.compaction_lines(self.pools[pool].load(),
                                                    [str(a) for a in self.cfg["trainer"]["argv"]], island=island)
        except (OSError, ValueError):
            return  # the pool dry-run reports an unreadable pool config
        warning = up_consistency.payload_size_warning(self.pools[pool].load(),
                                                       [str(a) for a in self.cfg["trainer"]["argv"]], self.env)
        if warning is not None:
            lines.append(f"WARNING: {warning}")
        lines += [f"WARNING: {w}" for w in up_consistency.score_centering_warnings(
            self.pools[pool].load(), [str(a) for a in self.cfg["trainer"]["argv"]])]
        for line in lines:
            _log(f"preflight: {line}" + (f" (pool {self.pools[pool].name})" if len(self.pools) > 1 else ""))

    def budget_problem(self) -> str | None:
        """A budget that would stop the run at once: ``budget.usd`` 0 (the guide's
        template value) with a price set, so the first estimated dollar is a breach.
        Logs the wall clock's implicit 3 h default (decision 2: shown, never silent)."""
        b = self.cfg.get("budget") or {}
        try:  # the legacy keys' own checks (their messages name the key)
            up_consistency.budget_capped(b)
            up_consistency.budget_wall_h(b)
        except ValueError as exc:
            return str(exc)
        usd = self.breaker_param("run.budget", "usd")
        wall = self.breakers.entry("run.wall_clock")
        wall_h = wall.params["hours"] if wall.enabled else None
        if "usd" in b and b["usd"] is None:
            _log("preflight: budget.usd is null: no spend cap (as if unset); the wall clock still applies")
        if wall.enabled and wall.sources["hours"].kind == "default":
            _log(f"preflight: WARNING: budget.wall_h is not set: the default {wall_h:g} h wall clock stops the "
                 f"trainer {wall_h:g} h after this run up starts the pools (and the breaker forces the sampler "
                 f"pools down at {wall_h * self.breakers.get('run.budget.force_pools_ratio'):g} h). For a long run "
                 "set budget.wall_h to its hours, or to null for no wall-clock cap")
        elif wall_h is None:
            off = "budget.wall_h is null" if "wall_h" in b else "breakers.run.wall_clock is off"
            _log(f"preflight: {off}: no wall-clock cap; only budget.usd (if set) stops the run")
        priced = (up_consistency.number_or_default(self.cfg["trainer"], "usd_per_h", 0) > 0
                  or up_consistency.number_or_default(b, "nc_usd_per_h", 24) > 0)
        if usd is not None and usd <= 0 and priced:
            shown = f"budget.usd is {b['usd']}" if b.get("usd") is not None else f"breakers.run.budget.usd is {usd:g}"
            return (f"{shown} while prices are set (trainer.usd_per_h / budget.nc_usd_per_h): "
                    "the first estimated dollar would stop the run; set the spend cap")
        if usd is not None:
            for ref in self.pools:
                try:
                    pool_cfg = ref.load()
                except (OSError, ValueError):
                    continue  # the pool dry-run reports an unreadable pool config
                problem = up_consistency.price_problem(ref.name, ref.usd_per_h, pool_cfg)
                if problem is not None:
                    return problem
        return None

    def breakers_problem(self) -> str | None:
        """The ``breakers:`` section against the legacy keys it replaces: a disagreement (or
        two legacy places that disagree) is refused, both locations named. Without the
        section the legacy keys are read as before: a disagreement is only logged."""
        conflicts = [c for c in self.breakers.conflicts if c.name.startswith("run.")]  # launch: its dry run
        if not conflicts:
            return None
        lines = [f"{c.message} ({', '.join(c.locations)})" for c in conflicts]
        if self.breakers.has_section:
            return "breakers: " + "; ".join(lines)
        for line in lines:
            _log(f"preflight: WARNING: breakers: {line} (refused once the run config has a breakers: section)")
        return None

    def stall_timeout_s(self) -> float:
        """The trainer's ``--rl-stall-timeout`` (seconds; the launcher's default when unset):
        the launch layer's ``launch.stall`` once it is registered, else the argv itself."""
        from yeto.breakers import default_registry  # noqa: PLC0415

        if "launch.stall" in default_registry():
            return float(self.breaker_param("launch.stall", "timeout_s") or 0)
        argv = [str(a) for a in self.cfg["trainer"]["argv"]]
        value = None
        for i, a in enumerate(argv):
            if a == "--rl-stall-timeout" and i + 1 < len(argv):
                value = argv[i + 1]
            elif a.startswith("--rl-stall-timeout="):
                value = a.split("=", 1)[1]
        try:
            return float(value) if value is not None else DEFAULT_STALL_TIMEOUT_S
        except ValueError:
            return DEFAULT_STALL_TIMEOUT_S

    def silence_limit_s(self) -> float:
        """``run.trainer_silence``: its ``limit_s``, else derived from the stall timeout
        (``max(stall_factor_n x --rl-stall-timeout, floor_s)``; ``budget.silence_min`` is the floor)."""
        entry = self.breakers.entry("run.trainer_silence")
        if entry.params["limit_s"] is not None:
            return float(entry.params["limit_s"])
        return max(entry.params["stall_factor_n"] * self.stall_timeout_s(), entry.params["floor_s"])

    def progress_paths(self) -> list[Path]:
        """What grows while the trainer works: its launcher log and any
        ``trainer.progress_paths`` (e.g. a local event tape)."""
        tr = (self.state.get("procs") or {}).get("trainer") or {}
        paths = [Path(tr["log"])] if tr.get("log") else []
        return paths + [Path(os.path.expanduser(str(p))) for p in self.cfg["trainer"].get("progress_paths") or ()]

    def silence(self, now: float) -> str | None:
        """A backstop independent of the launcher: the trainer log / tapes did not grow for
        ``max(4 x --rl-stall-timeout, budget.silence_min)`` while the launcher still runs
        (a launcher hung with its stall timer)."""
        sig = []
        for path in self.progress_paths():
            try:
                st = path.stat()
                sig.append((str(path), st.st_size, st.st_mtime))
            except OSError:
                sig.append((str(path), None, None))
        seen = getattr(self, "_progress", None)
        if seen is None or seen[0] != sig:
            self._progress = (sig, now)
            return None
        quiet = self._quiet_s = now - seen[1]
        if self.breakers.enabled("run.trainer_silence") and quiet > self.silence_limit_s():
            return f"trainer: log / tape silent {quiet / 60:.0f} min (limit {self.silence_limit_s() / 60:.0f} min)"
        return None

    @property
    def release_nc_on_breach(self) -> bool:
        """``run.budget.release_pools_on_trip`` (legacy ``budget.release_nc_on_breach``): a
        budget or wall-clock breach runs the full down."""
        return bool(self.breakers.get("run.budget.release_pools_on_trip"))

    @staticmethod
    def limit_breach(why: str) -> bool:
        """A budget or wall-clock breach (the pools may be released / forced down), not a
        trainer failure or a first-group timeout."""
        return why.startswith(("wall clock", "budget"))

    def trainer_down(self, why: str) -> None:
        """Stop the trainer only: the NC pool, its nodes and the sandboxes stay (a node setup
        and a model pull to rebuild); ``yeto down <run>`` keeps the run's NC leases."""
        tr = (self.state.get("procs") or {}).get("trainer")
        if tr and _alive(tr["pid"]) and _signal(tr["pid"], signal.SIGINT):
            time.sleep(10)
        self._cmd(self.yeto("down", "--missing-ok", self.trainer_launch_name()), log="down.log", timeout=900)
        self.mark("trainer_down", why=why)
        stops = " / ".join(f"python -m yeto.launch.nc pool-stop --config {path}" for path in self.pool_configs)
        _log(f"trainer stopped ({why}); {self.pools_phrase()} and the sandboxes are KEPT and still bill. "
             f"Release them with: yeto run down --config <run.json> (or {stops})")

    def on_breach(self, why: str) -> None:
        """Budget / wall clock with ``release_nc_on_breach``: the full ``down``; anything
        else (and by default): the trainer only."""
        if self.release_nc_on_breach and self.limit_breach(why):
            _log(f"{why}: run.budget.release_pools_on_trip (budget.release_nc_on_breach) is set: full down "
                 f"({self.pools_phrase()} included)")
            self.down()
            return
        self.trainer_down(why)

    def rollout_stall_alert(self) -> str | None:
        """The in-process controller's rollout stall view, logged once per silent stretch
        (an alert: nothing is stopped; a sampler waiting for the trainer is not stalled)."""
        controller = getattr(self, "controller", None)
        stalled = getattr(controller, "stalled", None)
        if stalled is None:
            return None
        try:
            why = stalled()
        except Exception:  # noqa: BLE001 - a view only
            return None
        warned = getattr(self, "_rollout_stall_warned", False)
        if why and not warned:
            _log(f"WARNING: rollout stalled: {why}; see yeto run status and the pool events")
            self.breaker_event("run.rollout_stall_alert", value=None, limit=getattr(controller, "stall_s", None),
                               detail=why)
        elif not why and warned:
            _log("rollout: sampler progress resumed")
            self.breaker_event("run.rollout_stall_alert", value=None, limit=getattr(controller, "stall_s", None),
                               state="clear", detail="sampler progress resumed")
        self._rollout_stall_warned = bool(why)
        return why

    # -- trainer starvation (yeto.rl.starvation)
    @property
    def starvation_config(self) -> Any:
        """The resolved ``run.starvation`` breaker (legacy: the ``starvation`` section)."""
        from yeto.rl.starvation import StarvationConfig  # noqa: PLC0415

        return StarvationConfig.from_resolved(self.breakers)

    def starvation_problem(self) -> str | None:
        from yeto.rl.starvation import StarvationConfig  # noqa: PLC0415

        try:
            StarvationConfig.from_section(self.cfg.get("starvation"))  # the legacy section, its own messages
            self.starvation_config  # noqa: B018 - validates the resolved breaker
        except ValueError as error:
            return str(error)
        return None

    def buffer_keys(self) -> set[str] | None:
        """The object keys under ``sample-buffer/shards/`` (None: the listing failed)."""
        try:
            return set(self._objects().list("sample-buffer/shards/"))
        except Exception as error:  # noqa: BLE001
            _log(f"store list failed: {type(error).__name__}: {error}"[:300])
            return None

    SHARD_SCAN_EVERY = 6  # incremental checks between two scans of ``shard-names/`` (new shards)

    def buffer_new_groups(self) -> int | None:
        """Objects written to the buffer shards since the previous call (None: a listing
        failed, or this was the first call: the baseline). Incremental: one page after
        the last key seen per known shard; every ``SHARD_SCAN_EVERY`` calls, the shard
        names, and a new shard's keys once."""
        calls = getattr(self, "_shard_calls", 0)
        self._shard_calls = calls + 1
        found, self._shard_last = new_buffer_groups(self._objects, getattr(self, "_shard_last", None), calls,
                                                    self.SHARD_SCAN_EVERY)
        return found

    def sampler_health(self, now: float | None = None, pools: Any = None) -> Any:
        """Ready, unpaused members of the pools whose loop runs (the pool state files and
        the pool lock / pid file; no provider call). A pool whose loop does not run counts
        no healthy member: nothing replaces or restarts its samplers. ``pools``: the pool
        indices to count (an island's pools; None: every pool)."""
        from yeto.launch.nc import pool_run  # noqa: PLC0415
        from yeto.rl.starvation import SamplerHealth  # noqa: PLC0415

        from yeto.launch.pool.status import ElasticConfig  # noqa: PLC0415

        now = time.time() if now is None else now
        healthy = paused = live = recovering = waiting = 0
        recover_s = 0.0
        readable = False
        unreadable = False
        notes = []
        controller = getattr(self, "controller", None)
        for i, ref in enumerate(self.pools):
            if pools is not None and i not in pools:
                continue
            members = self._members(i)
            if members is None:
                notes.append(f"{ref.name}: no state")
                unreadable = True
                continue
            readable = True
            if controller is not None:
                driven = not controller.stopped
            else:
                try:
                    proc = pool_run.pool_process(ref.load())
                except Exception:  # noqa: BLE001 - unknown: taken as running
                    proc = {"alive": True}
                driven = bool(proc and proc.get("alive"))
            alive = [m for m in members if m.get("state") != "gone"]
            live += len(alive)
            ready = [m for m in alive if m.get("state") == "ready"]
            paused += sum(1 for m in alive if m.get("paused"))  # an operator's hold, in any state
            if not driven:
                notes.append(f"{ref.name}: pool loop not running")
                continue
            healthy += sum(1 for m in ready if not m.get("paused"))
            waiting += sum(1 for m in ready if not m.get("paused") and m.get("trainer_wait"))
            try:
                timeout = float(ElasticConfig.from_dict(ref.load().get("elastic")).start_timeout_s)
            except Exception:  # noqa: BLE001
                timeout = float(ElasticConfig().start_timeout_s)
            starting = [m for m in alive if m.get("state") in ("provisioning", "starting") and not m.get("paused")
                        and not m.get("config_error") and now - float(m.get("created") or 0.0) <= timeout]
            if starting:
                recovering += len(starting)
                recover_s = max(recover_s, timeout)
            halted = [m for m in alive if m.get("config_error")]
            if halted:
                notes.append(f"{ref.name}: {len(halted)} member(s) with a config error")
        if not readable or (unreadable and healthy == 0):  # an unread pool may hold the healthy ones
            return SamplerHealth(None, paused=paused, live=live, detail="; ".join(notes))
        return SamplerHealth(healthy, paused=paused, live=live, detail="; ".join(notes),
                             recovering=recovering, recover_s=recover_s, waiting=waiting)

    def trainer_side_cause(self) -> str | None:
        """What the trainer log says about a pause of its own: the incarnation gate fails
        closed while a lease is unreadable (groups wait in the buffer, nothing trains), or
        it ended the trainer on a store refusal no retry changes (credentials, bucket)."""
        log = ((self.state.get("procs") or {}).get("trainer") or {}).get("log")
        try:
            offset = int(((self.state.get("procs") or {}).get("trainer") or {}).get("offset") or 0)
            tail = _log_tail(log, offset) if log else ""
        except OSError:
            return None
        from yeto.rl.engine.incarnation_lease import FATAL_MARK  # noqa: PLC0415

        paused = tail.rfind("incarnation gate: training paused")
        fatal = tail.rfind(FATAL_MARK)
        if fatal >= 0 and fatal > paused:
            line = tail[fatal:].splitlines()[0]
            return f"{line[:400]} (the trainer exits on it: not starvation, not a gate pause)"
        if paused < 0 or tail.rfind("incarnation gate: lease of", paused + 1) > paused:
            return None
        line = tail[paused:].splitlines()[0]
        return f"{line[:300]} (fail-closed pause, not starvation)"

    # -- the data end (every sampler pool finished)
    def pools_data_ended(self) -> bool:
        """Every sampler pool finished (a sampler exited 0 on the end of the data:
        ``IslandPool.finished``) and holds no member that is not gone: no sampler of
        this run runs or will start, so no shard gets another group. Only pools whose
        samplers write the run's store (``data_path`` r2, no other store): an end
        record the trainer does not read would only switch the starvation watch off.
        A fleet judges each island against its own sub-directory
        (:meth:`FleetRuntime.pools_finished`): its pools write ``trainer<k>``, not the run
        store, which the check below would refuse."""
        if self.fleet_plan() is not None:
            return self.fleet_rt().pools_data_ended()
        if not self.pools:
            return False
        from yeto.rl.up_consistency import _norm_store, pool_store  # noqa: PLC0415

        for i in range(len(self.pools)):
            if self.pools[i].data_path != "r2":
                return False  # its samplers write through the syncer relay: the trainer reads no store shard
            try:
                written = pool_store(self.pools[i].load())
            except Exception:  # noqa: BLE001
                return False
            if written is not None and written != _norm_store(self.cfg["store"]):
                return False  # its samplers write another store (a synced trainer's substore): not sealed here
            doc = self.pool_doc(i)
            if not doc or not doc.get("finished"):
                return False
            if any(m.get("state") != "gone" for m in doc.get("members") or ()):
                return False
        return True

    def data_end_shards(self) -> list[str]:
        """The shards whose end the trainer waits for: its ``--rl-buffer-shards`` (every
        shard index the pools may have used), else the shards the store names. A fleet:
        every island's ``--rl-island-buffer-shards``."""
        if self.fleet_plan() is not None:
            return self.fleet_rt().data_end_shards()
        named = self._trainer_flag("--rl-buffer-shards")
        if named is not None:
            return [s.strip() for s in named.split(",") if s.strip()]  # as the trainer reads it
        from yeto.rl.engine.sample_buffer import shard_names  # noqa: PLC0415

        return list(shard_names(self._objects()))

    def seal_data_end(self) -> bool:
        """Once every pool finished, end every shard the trainer reads that did not end.

        A sampler writes its shard's end record only when it ends on the data end (or
        hands its shard off); a shard whose sampler was fenced, lost or replaced, and a
        shard index of ``--rl-buffer-shards`` no node ever took, never ends, so the
        trainer's merged index never ends and the trainer waits for groups that cannot
        come (the starvation stop then ends it as a failure). No writer of these shards
        is left (every member is gone), so the end record (the format a sampler writes)
        is safe here. True once every shard ended (``state["data_ended"]``). A fleet seals
        island by island, each in its own sub-directory, and its data ended once every
        island's did (:meth:`FleetRuntime.seal_data_end`)."""
        if self.state.get("data_ended"):
            return True
        if self.fleet_plan() is not None:
            return self.fleet_rt().seal_data_end()
        if not self.pools_data_ended():
            return False
        if not self.dispense_settled():
            return False
        from yeto.rl.engine.sample_buffer import FileSampleBuffer, shard_index  # noqa: PLC0415

        sealed = []
        try:
            objects = self._objects()
            for shard in self.data_end_shards():
                writer = FileSampleBuffer(objects, batch_size=1, index=shard_index(shard))
                if not writer.data_ended:
                    if writer.owner is not None:
                        # a shard under ownership records takes only its owner's end
                        # (``sample_buffer`` module doc): no writer is left, so the seal
                        # takes the shard first (a fenced sampler's end record is void)
                        writer.claim_owner(SEAL_OWNER, by="operator")
                    writer.mark_data_end()
                    sealed.append(shard)
        except Exception as error:  # noqa: BLE001 - tried again next poll
            _log(f"data end: could not end the buffer shards: {type(error).__name__}: {error}"[:300])
            return False
        _log(f"data end: every sampler pool finished; ended {len(sealed)} shard(s) no sampler ended "
             f"({', '.join(sealed[:8])}{' ...' if len(sealed) > 8 else ''}); the trainer trains the full "
             "batches left and finishes")
        self.state["data_ended"] = {"at": _now(), "sealed": sealed}
        self.state.pop("starvation", None)
        self.save()
        return True

    def dispense_settled(self) -> bool:
        """The run's finite data is settled: every pool's store dispenser
        (``store_dispense.status``) counts every slot trained / exhausted / dropped
        (``totals["ended"]``), none queued, running, stale or unclaimed. A pool finished
        with tasks left (a sampler ended early, the rest went at their lease end) is no
        data end: a CRITICAL ``data_end_incomplete`` is logged (once per count) and the
        starvation watch stays on. A status that does not read: not settled. A fleet:
        every island settled (:meth:`FleetRuntime.settled`)."""
        if self.fleet_plan() is not None:
            return self.fleet_rt().dispense_settled()
        from yeto.rl.engine import store_dispense  # noqa: PLC0415

        left, pools = 0, []
        try:
            objects = store_dispense.ledger_objects(self._objects())
            seen: dict[tuple[str, str], dict[str, Any]] = {}  # pools sharing one ledger read it once
            for i in range(len(self.pools)):
                cfg = self.pools[i].load()
                store, ns = self.dispense_ledger(i, cfg)
                key = (up_consistency._norm_store(store), ns)
                if key not in seen:
                    ledger = objects if store == self.cfg["store"] else store_dispense.ledger_objects(
                        self._objects_of(store))
                    seen[key] = store_dispense.status(ledger, ns, details=False, settled=False)["totals"]
                totals = seen[key]
                if not totals.get("ended"):
                    n = sum(int(totals.get(k) or 0) for k in ("queued", "running", "stale", "unclaimed"))
                    left += n if n else 1  # no plan / no slots: not known to be settled
                    pools.append({"pool": self.pools[i].name, "namespace": ns, "totals": totals,
                                  "finished": (self.pool_doc(i) or {}).get("finished")})
        except Exception as error:  # noqa: BLE001 - asked again next poll
            _log(f"data end: the dispenser status did not read ({type(error).__name__}: {error}"[:300]
                 + "): not sealed")
            return False
        if not left:
            self.state.pop("data_end_incomplete", None)
            return True
        if (self.state.get("data_end_incomplete") or {}).get("left") != left:
            _log(f"CRITICAL data_end_incomplete: every sampler pool finished but {left} task(s) are not "
                 f"sampled ({json.dumps(pools, sort_keys=True)[:600]}): check why the pool finished; "
                 "`yeto run up` starts the pool again (the trainer's starvation watch stays on)")
            self.state["data_end_incomplete"] = {"at": _now(), "left": left, "pools": pools}
            self.save()
        return False

    def starvation_check(self, now: float) -> str | None:
        """Alert on a starving trainer; the reason to stop it, else None. A fleet watches
        each island (a stop stops that island; :meth:`fleet_all_stopped` ends the launch)."""
        if self.fleet_plan() is not None:
            self.fleet_rt().starvation_check(now)
            return None
        from yeto.rl.starvation import ALERT_EVENT, RECOVERED_EVENT, STOP, StarvationWatch  # noqa: PLC0415

        cfg = self.starvation_config
        if not cfg.enabled:
            return None
        watch = getattr(self, "_starvation", None)
        if watch is None:
            watch = self._starvation = StarvationWatch(cfg, now)
            self._starvation_listed: float | None = None
            self._starvation_list_ok = False
        def listing() -> int | None:
            self._starvation_listed = now
            found = self.buffer_new_groups()
            self._starvation_list_ok = found is not None
            return found

        listed_now = self._starvation_listed is None or now - self._starvation_listed >= cfg.check_s
        new_groups = listing() if listed_now else None
        health, cause = self.sampler_health(now), self.trainer_side_cause()
        verdict = watch.observe(now, new_groups=new_groups, health=health, cause=cause,
                                listed=getattr(self, "_starvation_list_ok", False))
        if verdict.state == STOP and not listed_now:  # a stop is decided on a fresh listing only
            verdict = watch.observe(now, new_groups=listing(), health=health, cause=cause,
                                    listed=self._starvation_list_ok)
        if verdict.recovered:
            _log(f"{RECOVERED_EVENT}: a new group reached the buffer")
            self.breaker_event("run.starvation", value=verdict.starved_s, limit=cfg.alert_s or None, state="clear",
                               detail="a new group reached the buffer", legacy_event=RECOVERED_EVENT)
            self.state.pop("starvation", None)
            self.save()
        if verdict.new_alert:
            _log(f"CRITICAL {ALERT_EVENT}: {verdict.reason}")
            self.breaker_event("run.starvation", value=verdict.starved_s, limit=cfg.alert_s or None, state="warn",
                               detail=verdict.reason, legacy_event=ALERT_EVENT)
            self.state["starvation"] = {"at": _now(), "event": ALERT_EVENT, "starved_min": round(verdict.starved_s / 60),
                                        "why": verdict.reason}
            self.save()
        self._starvation_trip = (verdict.starved_s, cfg.stop_s)  # what a stop reports
        return verdict.reason if verdict.state == STOP else None

    def trainer_launch_name(self) -> str:
        """The trainer launch's run name: its ``--cluster-prefix`` (``yeto launch`` names the
        run ``yeto`` without one). ``yeto down`` / the STOP flag / the worker check use it."""
        argv = [str(a) for a in self.cfg["trainer"]["argv"]]
        name = LAUNCH_DEFAULT_NAME
        for i, a in enumerate(argv):  # argparse: the last one wins
            if a == "--cluster-prefix" and i + 1 < len(argv):
                name = argv[i + 1]
            elif a.startswith("--cluster-prefix="):
                name = a.split("=", 1)[1]
        return name

    def starvation_stop(self, why: str, poll_s: float = 30.0) -> None:
        """Stop a starving trainer the way a budget stop does: the run's STOP flag (``yeto
        stop-run``: with the trainer scheduler the trainer drains with a cut; a fixed
        trainer island keeps its latest round cut), then, once the launcher exited or
        ``stop_grace_min`` passed, :meth:`trainer_down`. The sampler pools stay; ``yeto run
        up`` of this config resumes the trainer from its latest cut."""
        from yeto.rl.starvation import STOP_EVENT  # noqa: PLC0415

        value, limit = getattr(self, "_starvation_trip", (None, None))
        self.flag_stop("run.starvation", "starvation", STOP_EVENT, why, grace_s=self.starvation_config.stop_grace_s,
                       poll_s=poll_s, value=value, limit=limit,
                       after="once a sampler is healthy again, `yeto run up --config <run.json>` resumes "
                             "the trainer from its latest cut")

    def flag_stop(self, name: str, key: str, event: str, why: str, *, grace_s: float, poll_s: float = 30.0,
                  value: float | None = None, limit: float | None = None, after: str = "") -> None:
        """Stop the trainer for breaker ``name`` through the run's STOP flag, then
        :meth:`trainer_down` once the launcher exited or ``grace_s`` passed. ``state[key]``
        records the stop (``event``, ``why``, ``stop_reason``)."""
        from yeto import runs  # noqa: PLC0415

        _log(f"CRITICAL {event}: {why}; stopping the trainer (STOP flag, then yeto down)")
        trip = self.breaker_event(name, value=value, limit=limit, detail=why, legacy_event=event)
        self.state[key] = {"at": _now(), "event": event, "why": why}
        if trip.get("stop_reason"):
            self.state[key]["stop_reason"] = trip["stop_reason"]
        self.save()
        tr = (self.state.get("procs") or {}).get("trainer")
        deadline = time.monotonic() + grace_s
        try:
            flag = runs.request_stop(self.trainer_launch_name())
            _log(f"{event}: STOP flag written: {flag}")
        except OSError as error:  # no graceful drain to wait for: the yeto down below still stops it
            _log(f"{event}: STOP flag not written ({type(error).__name__}: {error}); stopping the trainer now")
            deadline = time.monotonic()
        while tr and _alive(tr["pid"]) and time.monotonic() < deadline:
            time.sleep(poll_s)
        self.trainer_down(f"{event}: {why}")
        if after:
            _log(f"{event}: {after}")

    # -- release stall (yeto.rl.release_stall)
    def latest_release(self, objects: Any = None) -> tuple[Any, bool]:
        """The identity of the store's LATEST policy release (None: no release yet) and
        whether it was read (False: a failed read, no evidence). ``objects``: another
        store's object API (an island's ``trainer<k>``; None: the run's store)."""
        from yeto.rl.quant.schemes import RELEASE_DIR  # noqa: PLC0415
        from yeto.rl.release_stall import LATEST, release_identity  # noqa: PLC0415

        try:
            raw = (self._objects() if objects is None else objects).read(f"{RELEASE_DIR}/{LATEST}")
            return (None if raw is None else release_identity(json.loads(raw))), True
        except Exception:  # noqa: BLE001 - checked again next poll
            return None, False

    def log_release_stall(self) -> None:
        """The preflight line of ``run.release_stall``: on by default, so it is stated."""
        from yeto.rl.release_stall import ALERT_EVENT, limit_phrase  # noqa: PLC0415

        entry = self.breakers.entry("run.release_stall")
        if not entry.enabled:
            _log(f"preflight: run.release_stall is off ({entry.sources['enabled'].location})")
            return
        what = "stops the trainer (STOP flag)" if entry.action == "stop" else f"logs {ALERT_EVENT} (no stop)"
        _log(f"preflight: run.release_stall is on: after the first policy release, no new release for "
             f"{limit_phrase(entry.params['limit_s'])} {what}. Set breakers.run.release_stall "
             '{"limit_s": N}, {"action": "stop"} to stop the trainer, or {"enabled": false} to turn it off')

    def release_stall_check(self, now: float) -> str | None:
        """Alert on a trainer that stopped publishing releases (``run.release_stall``);
        with ``action: stop`` the reason to stop it, else None. A fleet watches each
        island's ``trainer<k>/`` release (a stop stops that island)."""
        if self.fleet_plan() is not None:
            self.fleet_rt().release_stall_check(now)
            return None
        from yeto.rl.release_stall import ALERT_EVENT, RECOVERED_EVENT, STOP, ReleaseStallConfig, ReleaseStallWatch  # noqa: PLC0415

        entry = self.breakers.entry("run.release_stall")
        if not entry.enabled:
            return None
        watch = getattr(self, "_release_stall", None)
        if watch is None:
            watch = self._release_stall = ReleaseStallWatch(ReleaseStallConfig(
                limit_s=entry.params["limit_s"], check_s=entry.params["check_s"], stop=entry.action == "stop"))
            self._release_read: tuple[float, Any, bool] | None = None
        if self._release_read is None or now - self._release_read[0] >= watch.config.check_s:
            self._release_read = (now, *self.latest_release())
        _, release, read = self._release_read
        tr = (self.state.get("procs") or {}).get("trainer") or {}
        verdict = watch.observe(now, release=release, incarnation=(tr.get("pid"), tr.get("started")), read=read)
        if verdict.recovered:
            _log(f"{RECOVERED_EVENT}: a new policy release reached the store")
            self.breaker_event("run.release_stall", value=verdict.stalled_s, limit=verdict.limit_s, state="clear",
                               detail="a new policy release reached the store", legacy_event=RECOVERED_EVENT)
            self.state.pop("release_stall", None)
            self.save()
        if verdict.new_alert:
            _log(f"CRITICAL {ALERT_EVENT}: {verdict.reason}")
            if verdict.state != STOP:  # a stop reports its own trip (flag_stop)
                self.breaker_event("run.release_stall", value=verdict.stalled_s, limit=verdict.limit_s,
                                   detail=verdict.reason, legacy_event=ALERT_EVENT)
            self.state["release_stall"] = {"at": _now(), "event": ALERT_EVENT,
                                           "stalled_min": round(verdict.stalled_s / 60), "why": verdict.reason}
            self.save()
        self._release_stalled = verdict
        return verdict.reason if verdict.state == STOP else None

    def fleet_all_stopped(self) -> str | None:
        """A fleet whose every island a trip stopped: the fleet's stop reason (None: an
        island still runs, or no fleet)."""
        if self.fleet_plan() is None:
            return None
        return self.fleet_rt().all_stopped()

    def fleet_stop(self, reason: str, poll_s: float = 30.0) -> None:
        """No island of the fleet runs any more: stop the one trainer launch (STOP flag,
        then ``yeto down``), as a starvation stop of one trainer does."""
        from yeto.breakers.events import STOP_REASON_PREFIX  # noqa: PLC0415

        rows = (self.state.get("fleet_stops") or {}).get("islands") or ()
        why = "every island stopped (" + ", ".join(f"island {r['index']} ({r['name']}): {r['stop_reason']}"
                                                  for r in rows) + ")"
        self.flag_stop(reason.removeprefix(STOP_REASON_PREFIX), "fleet_stop", "fleet_all_islands_stopped", why,
                       grace_s=self.starvation_config.stop_grace_s, poll_s=poll_s,
                       after="`yeto run up --config <run.json>` resumes the trainers from their latest cut")

    def release_stall_stop(self, why: str, poll_s: float = 30.0) -> None:
        """``run.release_stall`` with ``action: stop``: the STOP flag, as a starvation stop."""
        from yeto.rl.release_stall import STOP_EVENT  # noqa: PLC0415

        verdict = getattr(self, "_release_stalled", None)
        self.flag_stop("run.release_stall", "release_stall", STOP_EVENT, why,
                       grace_s=self.breakers.get("run.release_stall.stop_grace_s"), poll_s=poll_s,
                       value=getattr(verdict, "stalled_s", None), limit=getattr(verdict, "limit_s", None),
                       after="`yeto run up --config <run.json>` resumes the trainer from its latest cut")

    def monitor(self, poll_s: float = 60.0) -> None:
        # from here a state.json write that fails (ENOSPC / EROFS / EIO) is logged and kept
        # in memory: raised, it ended the monitor (breaker, forced pool-stop) with nodes billing
        self.soft_save = True
        while True:
            self.up_heartbeat()
            self.rollout_stall_alert()
            why = self.breach()
            if why:
                self.on_breach(why)
                if self.limit_breach(why) and not self.release_nc_on_breach:
                    self.watch_nc_after_breach(why, poll_s)
                return
            tr = (self.state.get("procs") or {}).get("trainer")
            # the data ended: no new group is the normal end, not starvation (the trainer
            # finishes on the ended shards; its own stall timeout still bounds a hang)
            ended = self.seal_data_end()
            if tr and _alive(tr["pid"]) and not self.state.get("trainer_finished") and not ended:
                why = self.starvation_check(time.time())
                if why:
                    self.starvation_stop(why)
                    return
            if tr and _alive(tr["pid"]) and not self.state.get("trainer_finished"):
                why = self.release_stall_check(time.time())
                if why:
                    self.release_stall_stop(why)
                    return
                why = self.fleet_all_stopped()
                if why:
                    self.fleet_stop(why)
                    return
            if tr and not _alive(tr["pid"]) and not self.state.get("trainer_exit_seen"):
                _log(f"trainer launcher PID {tr['pid']} exited")
                self.state["trainer_exit_seen"] = _now()
                self.save()
            if self.state.get("trainer_finished") and self.on_trainer_finished():
                return
            time.sleep(poll_s)

    def pools_phrase(self) -> str:
        """The sampler pools as a banner names them: "the NC pool" for the one NC pool,
        else every pool with its provider ("the sampler pools nc (nc), modal (modal)")."""
        kinds = [ref.kind() for ref in self.pools]
        if len(kinds) == 1:
            return "the NC pool" if kinds[0] == "nc" else f"the {kinds[0]} sampler pool"
        return "the sampler pools " + ", ".join(f"{r.name} ({k})" for r, k in zip(self.pools, kinds))

    def on_trainer_finished(self) -> bool:
        """The trainer finished its rounds (``SUCCEEDED``). The samplers have nothing left
        to feed, but their nodes (every pool, every provider) and the sandboxes bill until
        released: with ``budget.release_nc_on_finish`` the full ``down`` runs now (returns
        True: the monitor ends); otherwise a banner says so (once), and the wall-clock guard stays."""
        if (self.cfg.get("budget") or {}).get("release_nc_on_finish"):
            _log("trainer finished: budget.release_nc_on_finish is set: full down (every sampler pool included)")
            self.down()
            return True
        if not self.state.get("finished_banner"):
            _log(f"trainer finished (SUCCEEDED); {self.pools_phrase()} and the sandboxes still bill. "
                 "Release them with: yeto run down --config <run.json> "
                 "(or set budget.release_nc_on_finish: true)")
            self.state["finished_banner"] = _now()
            self.save()
        return False

    def watch_nc_after_breach(self, why: str, poll_s: float = 60.0) -> None:
        """After a budget / wall-clock breach that stopped the trainer only, the NC pool
        still bills: stay up, print the billing banner every poll, and once the spend passes
        ``FORCE_FACTOR`` x ``budget.usd`` (or the wall time ``FORCE_FACTOR`` x ``wall_h``)
        release the pool (``pool-stop``). Returns when no member bills any more."""
        usd = self.breaker_param("run.budget", "usd")
        factor = self.breakers.get("run.budget.force_pools_ratio")
        usd_cap = (up_consistency.NO_CAP_USD if usd is None else usd) * factor
        wall_h = self.breaker_param("run.wall_clock", "hours")
        wall_cap = float("inf") if wall_h is None else wall_h * 3600 * factor
        forced: set[str] = set()
        failed = 0
        while True:
            members = self.pool_members()
            billed = None if members is None else [m for m in members if m.get("state") != "gone"]
            if billed is not None and not billed:
                _log(f"{self.pools_phrase()}: no machine bills any more; monitor done")
                return
            try:
                spend = self.spend_usd()
            except Exception as error:  # noqa: BLE001 - raised, it ended the watch with nodes billing
                _log(f"STILL BILLING after '{why}': spend unreadable ({type(error).__name__}: {error}); "
                     "checking again next poll"[:400])
                time.sleep(poll_s)
                continue
            elapsed = self.wall_elapsed_s()
            n = "?" if billed is None else len(billed)
            _log(f"STILL BILLING after '{why}': {n} machine(s) of {self.pools_phrase()}, spend ${spend:.2f}; "
                 f"release with: "
                 f"yeto run down --config <run.json> (forced at ${usd_cap:.2f}"
                 + (f" or {wall_cap / 3600:.1f} h)" if wall_h is not None else "; no wall-clock cap)"))
            if spend > usd_cap or elapsed > wall_cap:
                cap = f"spend ${spend:.2f} > ${usd_cap:.2f}" if spend > usd_cap else \
                    f"wall {elapsed / 3600:.1f} h > {wall_cap / 3600:.1f} h"
                _log(f"BREAKER: {cap} ({factor}x the budget): forcing pool-stop")
                if "nc_forced_stop" not in forced:
                    self.breaker_event("run.budget" if spend > usd_cap else "run.wall_clock", state="trip",
                                       value=spend if spend > usd_cap else elapsed / 3600,
                                       limit=usd_cap if spend > usd_cap else wall_cap / 3600,
                                       detail=f"{cap}: forced pool-stop of {self.pools_phrase()}")
                rc = max(self._cmd(self.nc("pool-stop", "--config", path), log="down.log", timeout=1800)
                         for path in self.pool_configs)
                if "nc_forced_stop" not in forced:
                    self.mark("nc_forced_stop", why=cap, rc=rc)
                    forced.add("nc_forced_stop")
                if rc == 0:
                    return
                # a pool-stop that failed or timed out (rc 124) has not released the machines:
                # keep watching (the loop ends once none bills) and try the stop again next poll
                failed += 1
                _log(f"BREAKER: pool-stop rc={rc}: the machines may still bill; retrying next poll")
                if failed >= 3:  # loud and recorded: `yeto run status` / the operator sees it
                    _log(f"CRITICAL: forced pool-stop failed {failed} times in a row (rc={rc}); "
                         f"{n} machine(s) still bill: release them by hand (yeto run down --config <run.json>)")
                    self.mark("nc_forced_stop_failed", why=cap, rc=rc, attempts=failed)
            time.sleep(poll_s)

    # -- down
    def lock_for_down(self, grace_s: float | None = None, poll_s: float = 1.0) -> bool:
        """``yeto run down`` takes ``up.lock`` too, so a ``run up`` of the run cannot order
        nodes or spawn a trainer while down stops them (and none starts during down). A
        running ``up`` holding the lock gets SIGTERM (its exit path runs); another ``run
        down`` holding it is only waited for. Down waits up to ``grace_s`` (default
        ``$YETO_RUN_DOWN_GRACE_S`` or 60 s) for the lock; past that down goes on without
        it: shutting the run down wins over the lock. Then the state is reloaded (the up
        may have spawned processes since this process read it). True: down holds the lock."""
        if grace_s is None:
            try:
                grace_s = float(os.environ.get("YETO_RUN_DOWN_GRACE_S", "") or DOWN_GRACE_S)
            except ValueError:
                grace_s = DOWN_GRACE_S
        holder = self.lock_up(role="down")
        if holder is not None:
            info = getattr(self, "lock_holder", None) or {"role": "up", "pid": holder, "start": None}
            role = info.get("role") or "up"
            same = holder and holder != os.getpid() and _alive(holder) and (
                not info.get("start") or info["start"] == _proc_start(holder))
            if role == "down":
                _log(f"down: another `yeto run down` of run {self.run} (PID {holder or '?'}) holds "
                     f"{self.dir / 'up.lock'}: waiting up to {grace_s:.0f}s for it")
            else:
                _log(f"down: `yeto run up` of run {self.run} is running (PID {holder or '?'}; "
                     f"{self.dir / 'up.lock'}): stopping it (SIGTERM), waiting up to {grace_s:.0f}s for its lock")
                if same:
                    try:
                        os.kill(holder, signal.SIGTERM)
                    except OSError:
                        pass
            for _ in range(max(1, int(grace_s / poll_s))):
                time.sleep(poll_s)
                holder = self.lock_up(role="down")
                if holder is None:
                    break
        if holder is not None:
            _log(f"down: PID {holder or '?'} still holds {self.dir / 'up.lock'} after "
                 f"{grace_s:.0f}s: going on without the lock (shutdown first); run `yeto run down` "
                 "again once it exited to stop what it started meanwhile")
        else:
            _log("down: holding up.lock (no run up of this run starts until down ends)")
        self.reload_state()
        return holder is None

    def down(self) -> bool:
        self.lock_for_down()
        failed: list[str] = []  # the checks that did not pass, named in the verdict
        self._cmd(self.yeto("dashboard", "export", *self.dashboard_args(), "-o",
                            str(self.dir / "dashboard.html")), log="down.log", timeout=600)  # read-only copy
        dash = (self.state.get("procs") or {}).get("dashboard")
        if dash and _alive(dash["pid"]):
            _signal(dash["pid"], signal.SIGTERM)
        tr = (self.state.get("procs") or {}).get("trainer")
        if tr and _alive(tr["pid"]) and _signal(tr["pid"], signal.SIGINT):
            time.sleep(10)
        if self._cmd(self.yeto("down", "--missing-ok", self.trainer_launch_name()), log="down.log", timeout=900) != 0:
            failed.append("trainer (yeto down: cloud/Modal teardown not confirmed)")
        for path in self.pool_configs:
            if self._cmd(self.nc("pool-stop", "--config", path), log="down.log", timeout=1800) != 0:
                failed.append(f"pool-stop {path}")
        if self.envs:
            if self._cmd(self.yeto("sandbox", "teardown", "--envs", self.envs, "--store", self.cfg["store"]),
                         log="down.log", timeout=600) != 0:
                failed.append("sandbox teardown")
        hosts = self.sandbox_hosts()
        if hosts:  # Harbor trial containers of this run (ai.yeto.run label) on every sandbox host
            for host in hosts:
                self._sweep_trials(host)
            left, unreachable = self.left_trial_containers(hosts)
            _log(f"sandbox: {left} container(s) of this run left" +
                 (f", hosts not answering: {' '.join(unreachable)}" if unreachable else ""))
            if left:
                failed.append(f"sandbox containers ({left} left)")
            if unreachable:
                failed.append(f"sandbox hosts not answering ({' '.join(unreachable)}): their containers unknown")
        for path in self.pool_configs:
            if self._cmd(self.nc("pool-status", "--config", path), log="down.log", timeout=300) != 0:
                failed.append(f"pool-status {path}")
        if self.no_local_state is not None:  # pids, step times, launch state of the run unknown here
            failed.append(f"no local state ({self.no_local_state} missing): only what the config names was stopped")
        ok = not failed
        self.end_up()
        self.mark("down", ok=ok)
        _log("down: " + ("verified" if ok else f"NOT verified ({'; '.join(failed)}), see down.log"))
        return bool(ok)

    def status(self) -> dict[str, Any]:
        keys = self.buffer_keys()  # None: the listing failed (no credentials, ...): unknown, not 0
        out: dict[str, Any] = {"run": self.run, "state": self.state,
                               "buffer_objects": None if keys is None else len(keys),
                               "spend_usd_estimate": round(self.spend_usd(), 1), "uptime": self.uptime(),
                               "procs": {k: dict(v, alive=_alive(v["pid"])) for k, v in
                                         (self.state.get("procs") or {}).items()}}
        out["sandbox_run_containers"] = self.run_containers()
        pools: list[dict[str, Any]] = []
        for ref in self.pools:
            p = subprocess.run(self.nc("pool-status", "--config", ref.config), env=self.env, capture_output=True,
                               text=True, timeout=300, check=False)
            row: dict[str, Any] = {"pool": ref.name, "provider": ref.kind(), "data_path": ref.data_path,
                                   "status": _output_tail(p)}
            try:
                doc = json.JSONDecoder().raw_decode((p.stdout or "").lstrip())[0]
                st = doc.get("state") or {}
                row["samplers"] = {"target": doc.get("target"), "nodes": doc.get("nodes"),
                                   "members": [{k: m.get(k) for k in ("ticket", "account", "shard", "state")}
                                               | {"retiring": m.get("retiring_since") is not None}
                                               for m in st.get("members") or () if m.get("state") != "gone"]}
            except (ValueError, TypeError, AttributeError):
                pass
            pools.append(row)
        out["pool"] = pools[0]["status"]  # the first (or only) pool, as before
        if "samplers" in pools[0]:
            out["samplers"] = pools[0]["samplers"] if len(pools) == 1 else {
                "target": sum(int((r.get("samplers") or {}).get("target") or 0) for r in pools),
                "nodes": sum(int((r.get("samplers") or {}).get("nodes") or 0) for r in pools),
                "members": [m for r in pools for m in (r.get("samplers") or {}).get("members") or ()]}
        if len(pools) > 1:
            out["pools"] = pools
        ledgers = []
        for store, ns, names in self.dispense_ledgers():
            p = subprocess.run([self.py, "-m", "yeto.rl.engine.store_dispense", "status", "--store", store,
                                "--namespace", ns, "--brief"],
                               env=self.env, capture_output=True, text=True, timeout=300, check=False)
            ledgers.append({"store": store, "namespace": ns, "pools": names, "status": _output_tail(p)})
        out["dispense"] = ledgers[0]["status"]  # the first (or only) ledger, as before
        if len(ledgers) > 1:  # pools with a ledger each (a fleet's islands): every one
            out["dispense_ledgers"] = ledgers
        out["trainer_buffer"] = self.trainer_buffer_counts()
        out["breakers"] = self.breaker_status()
        if self.fleet_plan() is not None:
            out["fleet"] = self.fleet_rt().status()
        return out

    def breaker_status(self) -> dict[str, Any]:
        """The run's breakers in short: the resolved settings (``breakers.resolved.json``
        in the state directory: implicit defaults, entries off, dependency findings) and
        the newest trips and near-limit gauges on this run's ``events.jsonl`` and the
        trainer's log (``rl_breaker`` / ``rl_breaker_status``)."""
        from yeto.dashboard.breakers import RESOLVED_FILE, summary  # noqa: PLC0415
        from yeto.rl.event_echo import PREFIX  # noqa: PLC0415

        doc = None
        try:
            doc = json.loads((self.dir / RESOLVED_FILE).read_text())
        except (OSError, ValueError):
            pass
        records: list[dict] = []
        log = ((self.state.get("procs") or {}).get("trainer") or {}).get("log")
        for path, prefix in ((self.dir / "events.jsonl", ""), (Path(log) if log else None, PREFIX)):
            if path is None or not path.is_file():
                continue
            with open(path, errors="replace") as fh:
                for line in fh:
                    if "rl_breaker" not in line:
                        continue
                    at = line.find(prefix) if prefix else 0
                    try:
                        record = json.loads(line[at + len(prefix):]) if at >= 0 else None
                    except ValueError:
                        continue
                    if isinstance(record, dict):
                        records.append(record)
        return summary(doc if isinstance(doc, dict) else None, records)

    def trainer_buffer_counts(self) -> dict[str, Any]:
        """What the trainer's buffer reader passed over, summed over the trainer's log
        (every launch of this run): groups without a policy identity after a resume
        (``rl_buffer_unidentified_policy_skipped``), groups older than the policy-age
        window (``rl_buffer_stale_skipped``), and the reader's last starvation warning
        (``rl_buffer_unidentified_policy_starved``: it reads groups and keeps none)."""
        from yeto.rl.event_echo import PREFIX  # noqa: PLC0415

        out: dict[str, Any] = {"unidentified_policy_groups": 0, "stale_groups": 0, "unidentified_starved": None}
        log = ((self.state.get("procs") or {}).get("trainer") or {}).get("log")
        if not log or not Path(log).is_file():
            return out
        wanted = ("rl_buffer_unidentified_policy_skipped", "rl_buffer_stale_skipped",
                  "rl_buffer_unidentified_policy_starved")
        with open(log, errors="replace") as fh:
            for line in fh:
                at = line.find(PREFIX)
                if at < 0 or not any(name in line for name in wanted):
                    continue
                try:
                    record = json.loads(line[at + len(PREFIX):])
                except ValueError:
                    continue
                event = record.get("event")
                if event == wanted[0]:
                    out["unidentified_policy_groups"] += int(record.get("unidentified_policy_groups") or 0)
                elif event == wanted[1]:
                    out["stale_groups"] += int(record.get("stale_groups") or 0)
                elif event == wanted[2]:
                    out["unidentified_starved"] = {k: record.get(k) for k in (
                        "time_unix", "rollout_id", "selections", "unidentified_policy_groups_total")}
        return out

    def scale_refusal(self, wanted: int, i: int = 0) -> str | None:
        """Why ``run scale`` refuses a scale up of pool ``i`` to ``wanted`` before the pool's checks."""
        doc = self.pool_doc(i) or {}
        members = [m for m in doc.get("members") or () if m.get("state") not in ("gone", "draining")
                   and m.get("retiring_since") is None]
        new = max(0, wanted - len(members))
        if not new:
            return None
        breaker = self.state.get("breaker")
        resumed = ((self.state.get("steps") or {}).get("trainer") or {}).get("at")
        if breaker and not (resumed and str(resumed) > str(breaker.get("at"))):  # a later run up resumed it
            return (f"the run's breaker fired ({breaker.get('why')} at {breaker.get('at')}): no node is added "
                    f"to a stopped run")
        usd = self.breaker_param("run.budget", "usd")
        cap = up_consistency.NO_CAP_USD if usd is None else usd
        spent = self.spend_usd()
        add = new * self.node_usd_per_h(i)
        if spent + add > cap:
            return (f"spend ${spent:.2f} + {new} new node hour(s) ${add:.2f} > budget.usd ${cap:.2f}")
        return None

    def scale(self, samplers: int | None = None, retire: list[str] | None = None, *, add: int = 0,
              remove: list[str] | None = None, mode: str | None = None, force_static: bool = False,
              dry_run: bool = False, source: str = "manual", pool: str | None = None,
              provider: str | None = None) -> int:
        """``yeto run scale`` / ``yeto rollout scale``: ``pool-scale`` on one pool of the run
        (``pool`` / ``provider``, else the pool holding a ``remove`` ticket, else the only
        pool), after the run's own budget and breaker check of a scale up."""
        i = self.pool_index(pool=pool, provider=provider, ticket=(remove or [None])[0])
        if i is None:
            listing = ", ".join(f"{r.name} ({r.kind()})" for r in self.pools)
            _log(f"scale: refused, nothing changed: name one pool with --pool (pools: {listing})")
            return 2
        # the run's budget check and the pool-scale it allows hold the run's scale lock
        # together: two concurrent scales each passing the check against the same target
        # (then both added by pool-scale) would pass budget.usd between them
        try:
            lock = open(self.dir / "scale.lock", "a+")  # noqa: SIM115 - closed below
        except OSError as error:  # a state directory this user cannot write
            _log(f"scale: refused, nothing changed: cannot open {self.dir / 'scale.lock'}: {error}")
            return 2
        with lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            # the state as it is now (merged with this process's own changes): a breaker
            # that fired while this scale waited for the lock refuses it
            self.save()
            return self._scale_locked(i, samplers, retire, add=add, remove=remove, mode=mode,
                                      force_static=force_static, dry_run=dry_run, source=source)

    def _scale_locked(self, i: int, samplers: int | None, retire: list[str] | None, *, add: int,
                      remove: list[str] | None, mode: str | None, force_static: bool, dry_run: bool,
                      source: str) -> int:
        """:meth:`scale` of pool ``i`` under the run's ``scale.lock``."""
        if samplers is None and add:
            wanted = self.pool_target(self.pool_doc(i), i) + int(add)
        elif samplers is None:
            wanted = self.pool_target(self.pool_doc(i), i) - len(remove or [])
        else:
            wanted = int(samplers)
        why = self.scale_refusal(wanted, i)
        if why is not None:
            _log(f"scale: refused, nothing changed: {why}")
            return 2
        if len(self.pools) > 1:
            _log(f"scale: pool {self.pools[i].name} ({self.pools[i].kind()})")
        argv = self.nc("pool-scale", "--config", self.pools[i].config)
        if samplers is not None:
            argv += ["--target", str(int(samplers))]
        elif add:
            argv += ["--add", str(int(add))]
        for ticket in remove or ():
            argv += ["--remove", ticket]
        for ticket in retire or ():
            argv += ["--retire", ticket]
        if mode:
            argv.append(f"--{mode}")
        if force_static:
            argv.append("--force-static")
        if dry_run:
            argv.append("--dry-run")
        if source != "manual":
            argv += ["--source", source]
        rc = subprocess.run(argv, env=self.env, check=False, timeout=300).returncode
        if rc == 0 and not dry_run:
            self.state.setdefault("scales", []).append({"at": _now(), "samplers": samplers, "add": add or None,
                                                        "remove": list(remove or []) or None, "mode": mode,
                                                        "pool": self.pools[i].name})
            self.save()
        return rc

    def _namespace(self) -> str:
        return self.dispense_ledger(0)[1]

    def dispense_ledger(self, i: int, cfg: dict[str, Any] | None = None) -> tuple[str, str]:
        """``(store, namespace)`` of pool ``i``'s store-dispenser ledger: the store its
        dispense config or spec names (``dispense_store``: a fleet's shared ledger,
        ``nc_cli.pool_dispense_store``), else the store its samplers write when that is
        not the run's (a trainer substore: an island's own ledger), else the run's
        store (the value of the run config, as before)."""
        from yeto.launch.nc import cli as nc_cli  # noqa: PLC0415

        cfg = self.pools[i].load() if cfg is None else cfg
        ns = str(((cfg.get("sampler") or {}).get("dispense") or {}).get("namespace") or "task-dispense")
        store = nc_cli.pool_dispense_store(cfg)
        if store is None:
            try:
                written = up_consistency.pool_store(cfg)
            except Exception:  # noqa: BLE001 - not known: the run's store
                written = None
            if written is not None and up_consistency._norm_store(written) != up_consistency._norm_store(
                    self.cfg["store"]):
                store = written
        return (self.cfg["store"] if store is None else str(store)), ns

    def dispense_ledgers(self) -> list[tuple[str, str, list[str]]]:
        """The distinct ledgers of the run's pools, ``(store, namespace, [pool names])`` in
        pool order (pools sharing one ledger are one entry); the run store's default
        ledger when the run has no pool."""
        out: dict[tuple[str, str], tuple[str, str, list[str]]] = {}
        for i, ref in enumerate(self.pools):
            store, ns = self.dispense_ledger(i)
            out.setdefault((up_consistency._norm_store(store), ns), (store, ns, []))[2].append(ref.name)
        return list(out.values()) or [(self.cfg["store"], "task-dispense", [])]

    def _objects_of(self, store: str):
        """The object API of a store other than the run's (a fleet's shared ledger)."""
        from yeto.cloud.s3_store import head_objects  # noqa: PLC0415

        if not str(store).startswith(("s3://", "r2://")):
            from yeto.rl.engine.store_dispense import DirObjects  # noqa: PLC0415

            return DirObjects(store)
        return head_objects(str(store).replace("r2://", "s3://", 1), environ=self.env)


def new_buffer_groups(objects_of: Any, last: dict[str, str] | None, calls: int,
                      scan_every: int) -> tuple[int | None, dict[str, str] | None]:
    """:meth:`Run.buffer_new_groups` over one store: ``(found, last)``. ``objects_of()``
    gives the store's object API; ``last`` maps each known shard to the last key seen
    (None: no baseline yet); ``calls`` counts the earlier calls. ``found`` is None on the
    baseline call or a failed listing; ``last`` comes back None when a failed baseline is
    to be taken again."""
    from yeto.rl.engine.sample_buffer import BUFFER_DIR, SHARD_NAMES, shard_prefix  # noqa: PLC0415

    baseline = last is None
    last = {} if last is None else last
    found = 0
    try:
        objects = objects_of()
        if baseline or calls % scan_every == 0:
            names_prefix = f"{BUFFER_DIR}/{SHARD_NAMES}/"
            for key in objects.list(names_prefix):
                shard = key[len(names_prefix):]
                if shard and shard not in last:
                    keys = objects.list(shard_prefix(shard))
                    last[shard] = max(keys) if keys else ""
                    found += 0 if baseline else len(keys)
        for shard, after in list(last.items()):
            prefix = shard_prefix(shard)
            if hasattr(objects, "list_page"):
                keys, _more = objects.list_page(prefix, start_after=after or "", limit=1000)
            else:
                keys = objects.list(prefix, start_after=after or "")
            if keys:
                last[shard] = max(keys)
                found += 0 if baseline else len(keys)
    except Exception as error:  # noqa: BLE001
        _log(f"store list failed: {type(error).__name__}: {error}"[:300])
        return None, (None if baseline else last)  # a failed baseline is taken again
    return (None if baseline else found), last


def _worker_alive(run: str) -> bool:
    """The detached ``yeto launch`` worker of ``run`` (the run registry) still runs."""
    from yeto import runs  # noqa: PLC0415

    try:
        return _alive((runs.load_run(run) or {}).get("pid"))
    except Exception:  # noqa: BLE001 - an unreadable registry: no evidence it runs
        return False


DOWN_GRACE_S = 60.0


def _parse_lock_holder(text: str) -> dict[str, Any]:
    """The holder named in ``up.lock``: JSON ``{"role", "pid", "start"}``, or a bare PID
    (written by an older yeto: an ``up``)."""
    try:
        doc = json.loads(text)
    except ValueError:
        doc = None
    if isinstance(doc, dict):
        pid = doc.get("pid")
        return {"role": str(doc.get("role") or "up"), "pid": pid if isinstance(pid, int) else 0,
                "start": doc.get("start")}
    return {"role": "up", "pid": int(text) if text.isdigit() else 0, "start": None}


def _proc_start(pid: int) -> str | None:
    """Start time of ``pid`` in clock ticks since boot (``/proc/<pid>/stat`` field 22);
    None: unknown (no ``/proc``, e.g. macOS: the PID alone is trusted). Two processes
    with the same PID have different start times."""
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text()
        return stat.rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError, ValueError):
        return None


def _signal(pid: Any, sig: int) -> bool:
    """Send ``sig`` to ``pid``; False when it exited since the ``_alive`` check (the stop
    goes on: a ProcessLookupError here must not skip ``yeto down``)."""
    try:
        os.kill(int(pid), sig)
    except ProcessLookupError:
        return False
    return True


def _alive(pid: Any) -> bool:
    """``pid`` runs. A child of this process that exited is reaped here first: ``run up``
    starts the trainer with Popen and never waits on it, so the finished launcher stays
    a zombie, which ``kill(pid, 0)`` reports alive (the trainer's end was never seen:
    no ``release_nc_on_finish`` down, the sampler islands kept billing)."""
    try:
        done, _status = os.waitpid(int(pid), os.WNOHANG)
        if done:
            return False
    except ChildProcessError:
        pass  # not a child of this process (an adopted run): kill 0 below
    except (ValueError, TypeError, OverflowError):
        return False
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True
    return True


POOL_KINDS = ("nc", "daytona", "sky", "modal")


def add_pool_args(p: Any) -> None:
    """``--pool`` / ``--provider``: which pool of a run config's ``samplers`` list."""
    p.add_argument("--pool", help="the pool's name in the run config's samplers list (several pools)")
    p.add_argument("--provider", choices=POOL_KINDS, help="pick the pool by provider kind")


def pause_resume(run: Run, command: str, tickets: list[str], *, hold_tasks: bool = False,
                 pool: str | None = None, provider: str | None = None) -> int:
    """``pool-pause`` / ``pool-resume`` on the pool of the tickets (or the named one)."""
    i = run.pool_index(pool=pool, provider=provider, ticket=tickets[0] if tickets else None)
    if i is None:
        _log(f"{command}: name the pool with --pool (pools: {', '.join(r.name for r in run.pools)})")
        return 2
    argv = run.nc(f"pool-{command}", "--config", run.pools[i].config)
    for ticket in tickets:
        argv += ["--ticket", ticket]
    if hold_tasks:
        argv.append("--hold-tasks")
    return subprocess.run(argv, env=run.env, check=False, timeout=300).returncode


def add_parser(sub: Any) -> None:
    p = sub.add_parser("run", help="one run config, one command: up / status / down of an async GRPO run "
                       "(yeto rl is the single-process CyberGym PPO path)")
    rsub = p.add_subparsers(dest="rl_command", metavar="{up,status,down,scale,pause,resume}")
    rsub.required = True
    up = rsub.add_parser("up", help="preflight, sandbox, pool, first group, trainer, dashboard, monitor")
    up.add_argument("--config", required=True)
    up.add_argument("--dry-run", action="store_true", help="preflight only; nothing is ordered")
    up.add_argument("--skip-preflight", action="store_true",
                    help="resume: preflight already passed (also skips the cluster preflight of an ssh: trainer)")
    up.add_argument("--no-monitor", action="store_true", help="return after the dashboard starts")
    up.add_argument("--allow-after-forced-stop", action="store_true",
                    help="run up again after a budget breaker forced the sampler pool down (nc_forced_stop)")
    up.add_argument("--trainer-now", action="store_true",
                    help="start the trainer without waiting for the first sample group (the samplers and the "
                    "trainer GPUs come up in parallel)")
    for name in ("status", "down"):
        q = rsub.add_parser(name)
        q.add_argument("--config", required=True)
    from yeto.launch.nc.scale_args import add_scale_args  # noqa: PLC0415 - no imports of its own

    sc = rsub.add_parser("scale", help="add or remove sampler nodes while the run trains (= yeto rollout scale)")
    sc.add_argument("--config", required=True)
    add_scale_args(sc, target_flag="--samplers")
    add_pool_args(sc)
    for name, what in (("pause", "manual maintenance: the pool takes no automatic action on these nodes"),
                       ("resume", "end a pause: the pool watches these nodes again")):
        q = rsub.add_parser(name, help=what)
        q.add_argument("--config", required=True)
        q.add_argument("--ticket", action="append", required=True, help="the node ticket (repeatable)")
        add_pool_args(q)
        if name == "pause":
            q.add_argument("--hold-tasks", action="store_true", help="its sampler claims no new task meanwhile")


def _exit_on_sigterm(signum: int, frame: Any) -> None:  # noqa: ARG001
    raise SystemExit(128 + int(signum))


def _log_sigterm_during_drain(signum: int, frame: Any) -> None:  # noqa: ARG001
    _log("SIGTERM during the exit drain of the sampler pools: ignored until the drain ends "
         "(cutting it short would leave machines billing)")


FORCED_EXIT_CODE = 128 + int(signal.SIGINT)  # a second Ctrl-C: the shell's own convention


class CtrlC:
    """SIGINT (Ctrl-C) of ``run up``: the SIGTERM exit, announced, with a forced second.

    Without it a ``run up`` started in the background of a non-interactive shell
    (``cmd &`` in a script, as a launcher script does) ignores SIGINT: POSIX starts
    that job with SIGINT set to SIG_IGN and Python keeps an inherited SIG_IGN, so
    ``kill -INT`` / Ctrl-C did nothing and only SIGTERM ended the monitor. In the
    foreground the default KeyboardInterrupt ended it with a traceback, and a second
    Ctrl-C cut the in-process exit drain short (only SIGTERM was held during it).

    The first SIGINT logs one line (what stops, what keeps running or is released)
    and then takes exactly what SIGTERM does at that moment: the handler installed
    for SIGTERM is called (in-process: ``SystemExit(143)`` through the exit drain;
    during the drain: logged and held), and with the default SIGTERM disposition
    (``controller: subprocess``) the process ends by SIGTERM, leaving the pool loops,
    the trainer and the dashboard running in their own sessions as SIGTERM does.
    A second SIGINT ends the process at once (exit code 130) without a traceback,
    saying what may still bill."""

    def __init__(self, run: Run):
        self.run = run
        self.stopping = False
        self._installed = False
        self._old: Any = None

    def install(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        try:
            self._old = signal.signal(signal.SIGINT, self.handle)
            self._installed = True
        except (ValueError, OSError):
            pass

    def restore(self) -> None:
        """Back to the earlier SIGINT handler, unless a Ctrl-C started the exit (then a
        second one must still force it)."""
        if self._installed and not self.stopping:
            try:
                signal.signal(signal.SIGINT, self._old if self._old is not None else signal.SIG_DFL)
            except (ValueError, OSError):
                pass
            self._installed = False

    def plan(self) -> str:
        """What the SIGTERM exit does with the machines, as the first Ctrl-C line says it."""
        run = self.run
        try:
            pools = run.pools_phrase()
        except Exception:  # noqa: BLE001 - a log line only
            pools = "the sampler pools"
        if run.in_process:
            controller = getattr(run, "controller", None)
            if controller is not None and controller.stopped:
                return (f"the in-process controller already stopped {pools}; the trainer and the dashboard keep "
                        "running in their own sessions")
            if bool(rollout_section(run.cfg).get("drain_on_exit", True)):
                return (f"draining and releasing {pools} before exit (rollout.drain_on_exit); the trainer and the "
                        "dashboard keep running in their own sessions. Ctrl-C again forces the exit now and "
                        "leaves the machines billing")
            return (f"rollout.drain_on_exit is false: leaving {pools} running and billing, undriven, and the "
                    "trainer and the dashboard running; release them with `yeto run down --config <run.json>`")
        return (f"leaving {pools} (the pool loops), the trainer and the dashboard running in their own sessions, "
                "still billing: release them with `yeto run down --config <run.json>`; a later `yeto run up` of "
                "this config adopts them")

    def handle(self, signum: int, frame: Any) -> None:
        if self.stopping:
            self._say("second Ctrl-C: forcing the exit now, the exit steps are cut short; "
                      + self._left_running()
                      + f" (exit code {FORCED_EXIT_CODE})")
            os._exit(FORCED_EXIT_CODE)
        self.stopping = True
        try:
            plan = self.plan()
        except Exception:  # noqa: BLE001 - a log line only: the exit itself must still happen
            plan = "the machines are kept or released as on SIGTERM"
        self._say(f"Ctrl-C: stopping the run up monitor, as on SIGTERM: {plan}")
        term = signal.getsignal(signal.SIGTERM)
        if callable(term):
            term(int(signal.SIGTERM), frame)
        elif term == signal.SIG_DFL:
            signal.raise_signal(signal.SIGTERM)
        else:  # SIGTERM ignored (inherited SIG_IGN): still the exit SIGTERM would take with a handler
            raise SystemExit(128 + int(signal.SIGTERM))

    def _left_running(self) -> str:
        try:
            pools = self.run.pools_phrase()
        except Exception:  # noqa: BLE001 - a log line only
            pools = "the sampler pools"
        return (f"{pools}, the trainer and the dashboard may STILL BE RUNNING and billing: check with "
                "`yeto run status --config <run.json>` and release them with `yeto run down --config <run.json>`")

    @staticmethod
    def _say(msg: str) -> None:
        # in a signal handler: a print that collides with the main thread's write
        # (reentrant buffered write) must not turn the stop into a traceback
        try:
            _log(msg)
        except (RuntimeError, OSError, ValueError):
            pass


class RunUpRefused(RuntimeError):
    """``run up`` must not go on; the message says why and what to do."""


def forced_stop_refusal(run: Run, allow: bool, *, record: bool = True) -> str | None:
    """A breaker of an earlier ``run up`` forced the NC pool down (``nc_forced_stop``,
    spend or wall time past ``FORCE_FACTOR`` x the budget): a new ``run up`` would order
    again past that cap. Refused unless ``--allow-after-forced-stop``, which is recorded
    (``nc_forced_stop_ack``) so later ``run up``s of the run go on until the next one.
    ``record`` False: the acknowledgement is left to the caller (:func:`ack_forced_stop`
    once the run up goes on to order; a dry run or a failed preflight does not record it)."""
    forced = unacknowledged_forced_stop(run)
    if forced is None:
        return None
    if allow:
        _log(f"--allow-after-forced-stop: the forced stop at {forced.get('at')} ({forced.get('why')}) is "
             "acknowledged; this run up orders samplers again")
        if record:
            ack_forced_stop(run)
        return None
    return (f"CRITICAL: an earlier run up of run {run.run} FORCED the sampler pool down at {forced.get('at')} "
            f"({forced.get('why')}, {FORCE_FACTOR}x the budget): nothing started. Check the spend, raise "
            "budget.usd / budget.wall_h if the run should go on, and rerun with --allow-after-forced-stop")


def unacknowledged_forced_stop(run: Run) -> dict[str, Any] | None:
    """The ``nc_forced_stop`` step no ``nc_forced_stop_ack`` at or after it covers (None: none)."""
    steps = run.state.get("steps") or {}
    forced = steps.get("nc_forced_stop")
    if not forced:
        return None
    ack = steps.get("nc_forced_stop_ack") or {}
    if ack.get("at") and str(ack["at"]) >= str(forced.get("at")):
        return None
    return forced


def ack_forced_stop(run: Run) -> None:
    forced = unacknowledged_forced_stop(run)
    if forced is not None:
        run.mark("nc_forced_stop_ack", of=forced.get("at"))


def open_run(config: str, create: bool = True, missing: str | None = None, *,
             strict_breakers: bool = True) -> Run:
    """``Run(load_config(config))``; a config or state directory that cannot be used raises
    :class:`RunConfigError` naming the file (the CLI prints it, without a traceback).
    ``create=False`` (read-only commands) refuses a run whose state directory does not exist
    instead of creating it: the run name is most likely misspelled. ``missing="temp"``
    (``run down``) neither refuses nor creates it: the run is opened on a fresh temporary
    state directory, so the stop steps the config drives (trainer down, pool-stop, sandbox
    teardown) still run on a control host that lost (or never had) the run's state."""
    try:
        cfg = load_config(config) if strict_breakers else load_config(config, strict_breakers=False)
    except RunConfigError:
        raise
    except ValueError as error:  # a check of the config's content
        raise RunConfigError(f"run config {Path(config).expanduser()}: {error}") from None
    state_dir = STATE_ROOT / str(cfg.get("run"))
    if missing == "temp" and "run" in cfg and not state_dir.is_dir():
        tmp = Path(tempfile.mkdtemp(prefix=f"yeto-run-down-{cfg.get('run')}-"))
        print(f"{_now()} [rl up] no local state for {cfg.get('run')} ({state_dir}): stopping what the config "
              f"names; logs in {tmp / str(cfg.get('run'))}", file=sys.stderr, flush=True)
        run = Run(cfg, state_root=tmp)
        run.no_local_state = state_dir
        return run
    if not create and "run" in cfg and not state_dir.is_dir():
        raise RunConfigError(f"run {cfg.get('run')}: no state directory {state_dir}; it was never brought up "
                             "here (`yeto run up`), or the run name is misspelled")
    try:
        if "run" in cfg:
            state_dir.mkdir(parents=True, exist_ok=True)  # what Run() does first
    except OSError as error:
        raise RunConfigError(f"run {cfg.get('run')}: cannot create the state directory {state_dir} "
                             f"({error.strerror or error}); HOME must be writable") from None
    try:
        return Run(cfg)
    except PoolConfigError as error:  # only resolve(); a corrupt state.json etc. propagates as on main
        raise RunConfigError(f"run config {Path(config).expanduser()}: {error}") from None


def main(args: Any) -> int:
    try:
        run = open_run(args.config, create=args.rl_command != "status",
                       missing="temp" if args.rl_command == "down" else None,
                       strict_breakers=args.rl_command in BREAKERS_STRICT_COMMANDS)
    except RunConfigError as error:
        print(f"{_now()} [rl up] {error}", file=sys.stderr, flush=True)
        return 2
    if args.rl_command == "status":
        print(json.dumps(run.status(), indent=1, default=str))
        return 0
    if args.rl_command == "down":
        return 0 if run.down() else 1
    if args.rl_command == "scale":
        return run.scale(args.target, args.retire, add=args.add, remove=args.remove, mode=args.mode,
                         force_static=args.force_static, dry_run=args.dry_run, source=args.source,
                         pool=getattr(args, "pool", None), provider=getattr(args, "provider", None))
    if args.rl_command in ("pause", "resume"):
        return pause_resume(run, args.rl_command, args.ticket, hold_tasks=getattr(args, "hold_tasks", False),
                            pool=getattr(args, "pool", None), provider=getattr(args, "provider", None))
    if run.in_process and args.no_monitor:
        _log("controller in_process: the pools run in this process, so it must stay up; drop --no-monitor "
             "(or set controller: subprocess)")
        return 2
    holder = run.lock_up()
    if holder is not None:
        role = (getattr(run, "lock_holder", None) or {}).get("role", "up")
        _log(f"another `yeto run {role}` of run {run.run} is running (PID {holder or '?'}; {run.dir / 'up.lock'}): "
             "nothing started" + ("; it already drives this run" if role == "up" else ""))
        return 2
    allow_forced = bool(getattr(args, "allow_after_forced_stop", False))
    refused = forced_stop_refusal(run, allow_forced, record=False)
    if refused:
        _log(refused)
        return 2
    run.log_fleet_plan()
    if not args.skip_preflight and not run.preflight():
        log = run.dir / "preflight.log"
        _log("preflight failed: nothing ordered" + (f" (see {log})" if log.exists() else
                                                     " (the reason is the preflight line above)"))
        return 2
    if args.skip_preflight:
        run.cluster_preflight_passed = True  # the escape hatch also reaches the trainer's own launch
    elif not run.cluster_preflight(dry_run=args.dry_run):
        _log("cluster preflight failed: nothing ordered")
        return 2
    if args.dry_run:
        _log("dry run: preflight ok, nothing ordered" + (
            "; a dry run does not acknowledge the forced stop" if allow_forced and unacknowledged_forced_stop(run)
            else "") + ("; the fleet plan is not written" if run.fleet_plan() is not None else ""))
        return 0
    if not run.write_fleet_plan():
        _log("fleet plan not written: nothing ordered")
        return 2
    if not run.sandbox_prepare():
        _log("sandbox prepare failed: nothing ordered")
        return 2
    if allow_forced:  # recorded once this run up goes on: not by a dry run or a failed preflight
        ack_forced_stop(run)
    if run.in_process:
        # SIGTERM (systemd stop, kill) ends the process through SystemExit, so the exit
        # path (the drain, or with drain_on_exit false the undriven banner) runs
        signal.signal(signal.SIGTERM, _exit_on_sigterm)
    # Ctrl-C from here on: the SIGTERM exit (CtrlC), announced; a second Ctrl-C forces it
    ctrl_c = CtrlC(run)
    ctrl_c.install()
    rc = 1
    try:
        # in the try: a SIGTERM while the pools start still drains what they ordered
        run.pool()
        rc = _drive(run, args)
    except RunUpRefused as error:
        _log(f"CRITICAL: run up refused: {error}")
        rc = 2
    finally:
        if run.in_process and not run.drain_controller():
            rc = rc or 1
        ctrl_c.restore()
    return rc


def _drive(run: Run, args) -> int:
    if getattr(args, "trainer_now", False):
        _log("--trainer-now: the trainer starts now, in parallel with the sampler setup")
    elif not (run.state.get("steps") or {}).get("first_group"):
        why = run.wait_first_group()
        if why is not None:
            run.on_breach(why)
            return 1
    run.trainer()
    run.dashboard()
    if args.no_monitor:
        return 0
    run.monitor()
    return 0
