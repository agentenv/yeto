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
               with no healthy sampler stops the trainer through the STOP flag); the
               sandbox host's container count of this run is a warning only.

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
     # several sandbox hosts: "docker_host": ["ssh://h1", "ssh://h2"], or "envs" (every host
     # of the envs file); the containers of the run are counted and removed on each
     "pool": {"config": "<pool.json>", "interval_s": 30},
     # or several rollout pools (any provider; yeto.launch.rollout.pools):
     # "samplers": [{"name": "nc", "provider": "nc", "config": "<pool.json>"},
     #              {"name": "modal", "provider": {"kind": "modal", ...}, "target": 1, ...}],
     "trainer": {"argv": ["--training-mode", "rl", ...], "usd_per_h": 136},
     "first_group": {"timeout_s": 5400, "poll_s": 60},
     "dashboard": {"port": 8795},
     "budget": {"usd": 450, "wall_h": 3.0, "nc_usd_per_h": 24,  # per NC node
                "silence_min": 30, "release_nc_on_breach": false}}

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
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from yeto.rl import up_consistency, up_segments

STATE_ROOT = Path("~/.yeto/rl-up").expanduser()
# the launcher's last line (yeto.cli): ``[yeto] run '<run>' finished: <state> (exit code <n>)``
# a budget / wall-clock breach stops the trainer only; the NC pool is released by force
# once the spend (or the wall time) passes this multiple of the budget
FORCE_FACTOR = 1.2
_FORMAT_WARNED: set[str] = set()  # pool state files of a newer format already reported
DEFAULT_STALL_TIMEOUT_S = 900.0  # yeto.launch.fleet.DEFAULT_RL_STALL_TIMEOUT_S
# ``yeto launch --cluster-prefix`` default (yeto.cli): the trainer's run name without the flag
LAUNCH_DEFAULT_NAME = "yeto"
SEAL_OWNER = "run-up-data-end"  # the incarnation the data-end seal claims an owned shard as
FINISHED = re.compile(r"finished: (\S+).*?\(exit code (-?\d+)\)")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def load_config(path: str) -> dict[str, Any]:
    cfg = json.loads(Path(path).expanduser().read_text())
    for key in ("run", "store", "trainer"):
        if key not in cfg:
            raise ValueError(f"run config: missing {key!r}")
    if "pool" not in cfg and "samplers" not in cfg:
        raise ValueError("run config: missing 'samplers' (the rollout pools) or 'pool' (one pool config)")
    known = {"run", "store", "env", "preflight", "sandbox", "pool", "samplers", "trainer", "first_group",
             "dashboard", "budget", "comment", "syncer", "controller", "rollout"}
    unknown = set(cfg) - known
    if unknown:
        raise ValueError(f"run config: unknown keys {sorted(unknown)} (known: {sorted(known)})")
    rollout_section(cfg)
    return cfg


ROLLOUT_KEYS = {"stall_alert_s", "drain_on_exit"}


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


class Run:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.run = str(cfg["run"])
        self.dir = STATE_ROOT / self.run
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.state: dict[str, Any] = {}
        self._state_base: dict[str, Any] = {}
        self.reload_state(strict=True)
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

        self.pools = resolve(cfg, self.dir)  # the rollout pools (samplers list, or the one pool)
        self.controller: Any = None  # controller: in_process -> the RolloutController of every pool

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
        with open(self.dir / "state.lock", "a+") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                disk = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
            except ValueError:
                disk = {}
            base = getattr(self, "_state_base", None)
            merged = _merge_state(base if base is not None else disk, self.state, disk)
            tmp = self.state_path.with_name(f".state.json.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(merged, indent=1, sort_keys=True))
            os.replace(tmp, self.state_path)
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
        with open(path, "a") as fh:
            fh.write(f"\n### {_now()} {' '.join(argv)}\n")
            fh.flush()
            proc = subprocess.run(argv, env=self.env, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout,
                                  check=False)
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
        sb = self.cfg.get("sandbox") or {}
        return os.path.expanduser(sb["envs"]) if sb.get("envs") else None

    # -- steps
    def preflight(self) -> bool:
        from yeto.launch_preflight import modal_sdk_problem  # noqa: PLC0415

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
        if self.stall_timeout_s() <= 0:
            _log("preflight: --rl-stall-timeout 0 disables the trainer's hang detection; run up refuses it "
                 "(the monitor's silence backstop is 4x this timeout)")
            self.mark("preflight", ok=False)
            return False
        for i in range(len(self.pools)):
            problem = self.shard_problem(i)
            if problem is not None:
                _log(f"preflight: {problem}" + (f" (pool {self.pools[i].name})" if len(self.pools) > 1 else ""))
                self.mark("preflight", ok=False)
                return False
            problem = self.rollout_fields_problem(i) or self.consistency_problem(i)
            if problem is not None:
                _log(f"preflight: {problem}" + (f" (pool {self.pools[i].name})" if len(self.pools) > 1 else ""))
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
        self.mark("preflight", ok=bool(ok))
        return bool(ok)

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
        if self.in_process:
            flag = self.dir / "STOP"
            if flag.exists():  # left by an earlier stop: the new controller would drain every pool at once
                _log(f"removing the STOP flag {flag} an earlier stop left: this run up drives the pools again")
                flag.unlink(missing_ok=True)
            self.start_controller()
            self.begin_up()
            self.mark("pool", controller="in_process", pools=[r.name for r in self.pools])
            return
        for i, ref in enumerate(self.pools):
            proc = pool_run.pool_process(ref.load())
            if proc and proc.get("alive"):
                _log(f"pool {ref.name}: loop already running (PID {proc.get('pid')})")
            else:
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
            ups.append({"pid": None, "start": float(self.state["t0"]), "seen": None, "end": None, "legacy": True})
        for period in ups:  # an earlier up that ended without a down: closed at its last poll
            if period.get("end") is None:
                period["end"] = up_segments.period_end(period, now)
        self.state.setdefault("run_t0", float(self.state.get("t0", now)))
        self.state["t0"] = now
        period = {"pid": os.getpid(), "start": now, "seen": now, "end": None}
        ups.append(period)
        self._up_period = period
        self.save()

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

    def uptime(self, now: float | None = None) -> dict[str, Any]:
        """:func:`yeto.rl.up_segments.uptime_summary` with the live spend and the time left
        on this up's ``budget.wall_h``."""
        now = time.time() if now is None else now
        out = up_segments.uptime_summary(self.state, now)
        out["spend_usd_estimate"] = round(self.spend_usd(), 1)
        wall_h = up_consistency.budget_wall_h(self.cfg.get("budget"))
        out["wall_h"] = wall_h
        out["wall_h_left"] = None  # None: no wall-clock cap, or no open up
        if wall_h is not None and out["current_up_open"] and "t0" in self.state:
            out["wall_h_left"] = round(wall_h - (now - float(self.state["t0"])) / 3600, 4)
        return out

    @property
    def in_process(self) -> bool:
        """``controller: "in_process"``: every pool in one RolloutController in this
        process (default ``"subprocess"``: one ``nc pool`` loop per pool)."""
        mode = str(self.cfg.get("controller") or "subprocess")
        if mode not in ("subprocess", "in_process"):
            raise ValueError(f"run config: controller {mode!r} is not 'subprocess' or 'in_process'")
        return mode == "in_process"

    def start_controller(self, *, ops: dict[str, Any] | None = None, **kw: Any) -> Any:
        """One :class:`~yeto.launch.pool.controller.RolloutController` over every pool
        (each built like its ``nc pool`` loop: state, events, scale file, lock), started
        in a thread of this process. ``ops``: per pool name, for tests."""
        from yeto.launch.pool.controller import DEFAULT_STALL_S, RolloutController, nc_pool  # noqa: PLC0415

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
        kw.setdefault("stall_s", float(section.get("stall_alert_s", DEFAULT_STALL_S)))
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
        """None once the store holds a group; else why the wait ended (a breach, the timeout)."""
        fg = self.cfg.get("first_group") or {}
        deadline = time.time() + float(fg.get("timeout_s", 5400))
        while time.time() < deadline:
            n = self.buffer_objects()
            if n:
                self.mark("first_group", objects=n)
                return None
            why = self.breach(trainer=False)
            if why:
                return why
            time.sleep(float(fg.get("poll_s", 60)))
        _log("first group: timed out")
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
        it is (the stop has not finished), and so is one that finished its run."""
        if not pool.stopped or getattr(pool, "finished", None):
            return
        if pool.live():
            tickets = [m.ticket for m in pool.live()]
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
        pool._event("pool_restarted", why="run up")  # noqa: SLF001

    def dashboard_args(self) -> list[str]:
        """``--store`` plus the run's local records (``--run``: the launcher's fleet record
        names the trainer's GPU, so its cost is priced), the cost bar's cap
        (``budget.usd``) and an optional price table (``dashboard.prices``)."""
        dash = self.cfg.get("dashboard") or {}
        # the launcher writes its records under ~/.yeto/runs/<its --cluster-prefix>
        argv = ["--store", self.cfg["store"], "--run", self.trainer_launch_name(),
                "--up-state", str(self.state_path)]
        usd = (self.cfg.get("budget") or {}).get("usd")
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
        now = time.time()
        nc = sum(self.node_hours(i, now) * self.node_usd_per_h(i) for i in range(len(self.pools)))
        rate = up_consistency.number_or_default(self.cfg.get("trainer"), "usd_per_h", 0)
        return nc + self.trainer_hours(now) * rate

    def trainer_hours(self, now: float | None = None) -> float:
        """Billed trainer hours: from ``trainer_t0`` (its first launch) to now, counted only
        inside the ``run up`` periods (``ups``): the stopped gaps between a ``down`` and the
        next ``up`` bill no trainer. Inside one up: ``now - trainer_t0`` as before."""
        now = time.time() if now is None else now
        t0 = self.state.get("trainer_t0")
        if t0 is None:
            return 0.0
        t0 = float(t0)
        ups = [p for p in self.state.get("ups") or () if p.get("start") is not None]
        if not ups:  # a run from before the periods were recorded
            return max(0.0, now - t0) / 3600
        return sum(max(0.0, up_segments.period_end(p, now) - max(float(p["start"]), t0)) for p in ups) / 3600

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
            start = max(float(m["granted"]), t0)
            end = float(m.get("charged_to") or now) if m.get("state") == "gone" else now
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
        sb = self.cfg.get("sandbox") or {}
        fixed = int(sb["max_run_containers"]) if sb.get("max_run_containers") else None
        per = int(sb["max_containers_per_sampler"]) if sb.get("max_containers_per_sampler") else None
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
        the log of the previous, stopped launcher, whose failure is already handled."""
        b = self.cfg.get("budget") or {}
        why = None
        controller = getattr(self, "controller", None)
        if controller is not None and controller.check():  # the in-process pools stopped being driven
            return controller.check()
        wall_h = up_consistency.budget_wall_h(b)
        if wall_h is not None and "t0" in self.state and time.time() - self.state["t0"] > wall_h * 3600:
            why = "wall clock"
        elif self.spend_usd() > up_consistency.budget_usd(b):
            why = f"budget ${self.spend_usd():.0f}"
        # the trainer: only its launcher's exit. An OOM or a fatal island error ends the
        # launcher; a hang is Miles' --rl-stall-timeout and the launcher's own business.
        tr = (self.state.get("procs") or {}).get("trainer") if trainer else None
        if why is None and tr and not _alive(tr["pid"]):
            try:
                tail = Path(tr["log"]).read_bytes()[int(tr.get("offset", 0)):][-20_000:].decode("utf-8", "replace")
            except OSError:
                tail = ""
            done = FINISHED.findall(tail)
            if done and done[-1][0] == "SUCCEEDED" and done[-1][1] == "0":
                self.state["trainer_finished"] = self.state.get("trainer_finished") or _now()
            if not done and not _worker_alive(self.trainer_launch_name()):  # a killed log streamer: the worker may still run
                why = "trainer: launcher exited without a result"
            elif done and (done[-1][0] == "FAILED" or done[-1][1] != "0"):
                why = f"trainer: launcher finished {done[-1][0]} (exit code {done[-1][1]})"
        if why is None and tr and _alive(tr["pid"]):
            why = self.silence(time.time())
        limit = self.container_limit() if why is None else None
        if why is None and limit:
            n = self.run_containers()
            # an alert, not a breach: the containers belong to the NC samplers, so stopping the
            # trainer frees none of them; an evicted island's trials outlive it while its
            # replacement starts its own, so the count passes the limit in a normal replacement
            if n is not None and n > int(limit):
                _log(f"WARNING: {n} sandbox containers of this run > {limit} (not a breach)")
                self.state["sandbox_over"] = {"at": _now(), "containers": n}
                self.save()
        if why:
            _log(f"BREAKER: {why}")
            self.state["breaker"] = {"at": _now(), "why": why}
            self.save()
        return why

    def shard_problem(self, pool: int = 0) -> str | None:
        """The trainer reads only the shards of its ``--rl-buffer-shards`` (a shard outside
        it is never trained on), so the pool may only order shards the trainer names: with
        that flag set, the pool needs a ``shard_limit`` and every ``<shard_base>-<i>`` below
        it in the trainer's list (each node and each lease-end replacement takes a new
        shard; plan the list for the largest scale plus the replacements)."""
        argv = [str(a) for a in self.cfg["trainer"]["argv"]]
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
        if not pool_cfg.get("spec"):
            return None
        try:
            spec = json.loads(Path(os.path.expanduser(str(pool_cfg["spec"]))).read_text())
        except (OSError, ValueError):
            return None  # the pool dry-run reports an unreadable spec
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
        return "; ".join(found) if found else None

    def budget_problem(self) -> str | None:
        """A budget that would stop the run at once: ``budget.usd`` 0 (the guide's
        template value) with a price set, so the first estimated dollar is a breach."""
        b = self.cfg.get("budget") or {}
        try:
            capped = up_consistency.budget_capped(b)
        except ValueError as exc:
            return str(exc)
        try:
            wall_h = up_consistency.budget_wall_h(b)
        except ValueError as exc:
            return str(exc)
        if "usd" in b and b["usd"] is None:
            _log("preflight: budget.usd is null: no spend cap (as if unset); the wall clock still applies")
        if "wall_h" not in b:
            _log(f"preflight: WARNING: budget.wall_h is not set: the default {wall_h:g} h wall clock stops the "
                 f"trainer {wall_h:g} h after this run up starts the pools (and the breaker forces the sampler "
                 f"pools down at {wall_h * FORCE_FACTOR:g} h). For a long run set budget.wall_h to its hours, "
                 "or to null for no wall-clock cap")
        elif wall_h is None:
            _log("preflight: budget.wall_h is null: no wall-clock cap; only budget.usd (if set) stops the run")
        priced = (up_consistency.number_or_default(self.cfg["trainer"], "usd_per_h", 0) > 0
                  or up_consistency.number_or_default(b, "nc_usd_per_h", 24) > 0)
        if capped and up_consistency.budget_usd(b) <= 0 and priced:
            return (f"budget.usd is {b['usd']} while prices are set (trainer.usd_per_h / budget.nc_usd_per_h): "
                    "the first estimated dollar would stop the run; set the spend cap")
        if capped:
            for ref in self.pools:
                try:
                    pool_cfg = ref.load()
                except (OSError, ValueError):
                    continue  # the pool dry-run reports an unreadable pool config
                problem = up_consistency.price_problem(ref.name, ref.usd_per_h, pool_cfg)
                if problem is not None:
                    return problem
        return None

    def stall_timeout_s(self) -> float:
        """The trainer's ``--rl-stall-timeout`` (seconds; the launcher's default when unset)."""
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
        b = self.cfg.get("budget") or {}
        return max(4 * self.stall_timeout_s(), up_consistency.number_or_default(b, "silence_min", 30) * 60)

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
        quiet = now - seen[1]
        if quiet > self.silence_limit_s():
            return f"trainer: log / tape silent {quiet / 60:.0f} min (limit {self.silence_limit_s() / 60:.0f} min)"
        return None

    @property
    def release_nc_on_breach(self) -> bool:
        return bool((self.cfg.get("budget") or {}).get("release_nc_on_breach", False))

    def trainer_down(self, why: str) -> None:
        """Stop the trainer only: the NC pool, its nodes and the sandboxes stay (a node setup
        and a model pull to rebuild); ``yeto down <run>`` keeps the run's NC leases."""
        tr = (self.state.get("procs") or {}).get("trainer")
        if tr and _alive(tr["pid"]):
            os.kill(int(tr["pid"]), signal.SIGINT)
            time.sleep(10)
        self._cmd(self.yeto("down", self.trainer_launch_name()), log="down.log", timeout=900)
        self.mark("trainer_down", why=why)
        stops = " / ".join(f"python -m yeto.launch.nc pool-stop --config {path}" for path in self.pool_configs)
        _log(f"trainer stopped ({why}); {self.pools_phrase()} and the sandboxes are KEPT and still bill. "
             f"Release them with: yeto run down --config <run.json> (or {stops})")

    def on_breach(self, why: str) -> None:
        """Budget / wall clock with ``release_nc_on_breach``: the full ``down``; anything
        else (and by default): the trainer only."""
        if self.release_nc_on_breach and why.startswith(("wall clock", "budget")):
            _log(f"{why}: budget.release_nc_on_breach is set: full down ({self.pools_phrase()} included)")
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
        elif not why and warned:
            _log("rollout: sampler progress resumed")
        self._rollout_stall_warned = bool(why)
        return why

    # -- trainer starvation (yeto.rl.starvation)
    @property
    def starvation_config(self) -> Any:
        from yeto.rl.starvation import StarvationConfig  # noqa: PLC0415

        return StarvationConfig.from_section(self.cfg.get("starvation"))

    def starvation_problem(self) -> str | None:
        try:
            self.starvation_config  # noqa: B018 - validates the section
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
        from yeto.rl.engine.sample_buffer import BUFFER_DIR, SHARD_NAMES, shard_prefix  # noqa: PLC0415

        last: dict[str, str] | None = getattr(self, "_shard_last", None)
        baseline = last is None
        last = {} if last is None else last
        calls = getattr(self, "_shard_calls", 0)
        self._shard_calls = calls + 1
        found = 0
        try:
            objects = self._objects()
            if baseline or calls % self.SHARD_SCAN_EVERY == 0:
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
            self._shard_last = None if baseline else last  # a failed baseline is taken again
            return None
        self._shard_last = last
        return None if baseline else found

    def sampler_health(self, now: float | None = None) -> Any:
        """Ready, unpaused members of the pools whose loop runs (the pool state files and
        the pool lock / pid file; no provider call). A pool whose loop does not run counts
        no healthy member: nothing replaces or restarts its samplers."""
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
            tail = Path(log).read_bytes()[-200_000:].decode("utf-8", "replace") if log else ""
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
        record the trainer does not read would only switch the starvation watch off."""
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
        shard index the pools may have used), else the shards the store names."""
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
        is safe here. True once every shard ended (``state["data_ended"]``)."""
        if self.state.get("data_ended"):
            return True
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
        starvation watch stays on. A status that does not read: not settled."""
        from yeto.rl.engine import store_dispense  # noqa: PLC0415

        left, pools = 0, []
        try:
            objects = store_dispense.ledger_objects(self._objects())
            for i in range(len(self.pools)):
                cfg = self.pools[i].load()
                ns = str(((cfg.get("sampler") or {}).get("dispense") or {}).get("namespace") or "task-dispense")
                totals = store_dispense.status(objects, ns, details=False, settled=False)["totals"]
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
        """Alert on a starving trainer; the reason to stop it, else None."""
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
            self.state.pop("starvation", None)
            self.save()
        if verdict.new_alert:
            _log(f"CRITICAL {ALERT_EVENT}: {verdict.reason}")
            self.state["starvation"] = {"at": _now(), "event": ALERT_EVENT, "starved_min": round(verdict.starved_s / 60),
                                        "why": verdict.reason}
            self.save()
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
        from yeto import runs  # noqa: PLC0415
        from yeto.rl.starvation import STOP_EVENT  # noqa: PLC0415

        _log(f"CRITICAL {STOP_EVENT}: {why}; stopping the trainer (STOP flag, then yeto down)")
        self.state["starvation"] = {"at": _now(), "event": STOP_EVENT, "why": why}
        self.save()
        flag = runs.request_stop(self.trainer_launch_name())
        _log(f"{STOP_EVENT}: STOP flag written: {flag}")
        tr = (self.state.get("procs") or {}).get("trainer")
        deadline = time.time() + self.starvation_config.stop_grace_s
        while tr and _alive(tr["pid"]) and time.time() < deadline:
            time.sleep(poll_s)
        self.trainer_down(f"{STOP_EVENT}: {why}")
        _log(f"{STOP_EVENT}: once a sampler is healthy again, `yeto run up --config <run.json>` resumes "
             "the trainer from its latest cut")

    def monitor(self, poll_s: float = 60.0) -> None:
        while True:
            self.up_heartbeat()
            self.rollout_stall_alert()
            why = self.breach()
            if why:
                self.on_breach(why)
                if why.startswith(("wall clock", "budget")) and not self.release_nc_on_breach:
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
        b = self.cfg.get("budget") or {}
        usd_cap = up_consistency.budget_usd(b) * FORCE_FACTOR
        wall_h = up_consistency.budget_wall_h(b)
        wall_cap = float("inf") if wall_h is None else wall_h * 3600 * FORCE_FACTOR
        while True:
            members = self.pool_members()
            billed = None if members is None else [m for m in members if m.get("state") != "gone"]
            if billed is not None and not billed:
                _log(f"{self.pools_phrase()}: no machine bills any more; monitor done")
                return
            spend = self.spend_usd()
            elapsed = time.time() - float(self.state.get("t0", time.time()))
            n = "?" if billed is None else len(billed)
            _log(f"STILL BILLING after '{why}': {n} machine(s) of {self.pools_phrase()}, spend ${spend:.0f}; "
                 f"release with: "
                 f"yeto run down --config <run.json> (forced at ${usd_cap:.0f}"
                 + (f" or {wall_cap / 3600:.1f} h)" if wall_h is not None else "; no wall-clock cap)"))
            if spend > usd_cap or elapsed > wall_cap:
                cap = f"spend ${spend:.0f} > ${usd_cap:.0f}" if spend > usd_cap else \
                    f"wall {elapsed / 3600:.1f} h > {wall_cap / 3600:.1f} h"
                _log(f"BREAKER: {cap} ({FORCE_FACTOR}x the budget): forcing pool-stop")
                rc = max(self._cmd(self.nc("pool-stop", "--config", path), log="down.log", timeout=1800)
                         for path in self.pool_configs)
                self.mark("nc_forced_stop", why=cap, rc=rc)
                return
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
            os.kill(int(dash["pid"]), signal.SIGTERM)
        tr = (self.state.get("procs") or {}).get("trainer")
        if tr and _alive(tr["pid"]):
            os.kill(int(tr["pid"]), signal.SIGINT)
            time.sleep(10)
        if self._cmd(self.yeto("down", self.trainer_launch_name()), log="down.log", timeout=900) != 0:
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
            left = self.run_containers()
            _log(f"sandbox: {left} container(s) of this run left")
            if left:
                failed.append(f"sandbox containers ({left} left)")
        for path in self.pool_configs:
            if self._cmd(self.nc("pool-status", "--config", path), log="down.log", timeout=300) != 0:
                failed.append(f"pool-status {path}")
        ok = not failed
        self.end_up()
        self.mark("down", ok=ok)
        _log("down: " + ("verified" if ok else f"NOT verified ({'; '.join(failed)}), see down.log"))
        return bool(ok)

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {"run": self.run, "state": self.state, "buffer_objects": self.buffer_objects(),
                               "spend_usd_estimate": round(self.spend_usd(), 1), "uptime": self.uptime(),
                               "procs": {k: dict(v, alive=_alive(v["pid"])) for k, v in
                                         (self.state.get("procs") or {}).items()}}
        out["sandbox_run_containers"] = self.run_containers()
        pools: list[dict[str, Any]] = []
        for ref in self.pools:
            p = subprocess.run(self.nc("pool-status", "--config", ref.config), env=self.env, capture_output=True,
                               text=True, timeout=300, check=False)
            row: dict[str, Any] = {"pool": ref.name, "provider": ref.kind(), "data_path": ref.data_path,
                                   "status": (p.stdout or p.stderr)[-3000:]}
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
        p = subprocess.run([self.py, "-m", "yeto.rl.engine.store_dispense", "status", "--store", self.cfg["store"],
                            "--namespace", self._namespace(), "--brief"],
                           env=self.env, capture_output=True, text=True, timeout=300, check=False)
        out["dispense"] = (p.stdout or p.stderr)[-3000:]
        out["trainer_buffer"] = self.trainer_buffer_counts()
        return out

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
        b = self.cfg.get("budget") or {}
        cap = up_consistency.budget_usd(b)
        spent = self.spend_usd()
        add = new * self.node_usd_per_h(i)
        if spent + add > cap:
            return (f"spend ${spent:.0f} + {new} new node hour(s) ${add:.0f} > budget.usd ${cap:.0f}")
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
        cfg = self.pools[0].load()
        return str(((cfg.get("sampler") or {}).get("dispense") or {}).get("namespace") or "task-dispense")


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
    up.add_argument("--skip-preflight", action="store_true", help="resume: preflight already passed")
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


class RunUpRefused(RuntimeError):
    """``run up`` must not go on; the message says why and what to do."""


def forced_stop_refusal(run: Run, allow: bool) -> str | None:
    """A breaker of an earlier ``run up`` forced the NC pool down (``nc_forced_stop``,
    spend or wall time past ``FORCE_FACTOR`` x the budget): a new ``run up`` would order
    again past that cap. Refused unless ``--allow-after-forced-stop``, which is recorded
    (``nc_forced_stop_ack``) so later ``run up``s of the run go on until the next one."""
    steps = run.state.get("steps") or {}
    forced = steps.get("nc_forced_stop")
    if not forced:
        return None
    ack = steps.get("nc_forced_stop_ack") or {}
    if ack.get("at") and str(ack["at"]) >= str(forced.get("at")):
        return None
    if allow:
        _log(f"--allow-after-forced-stop: the forced stop at {forced.get('at')} ({forced.get('why')}) is "
             "acknowledged; this run up orders samplers again")
        run.mark("nc_forced_stop_ack", of=forced.get("at"))
        return None
    return (f"CRITICAL: an earlier run up of run {run.run} FORCED the sampler pool down at {forced.get('at')} "
            f"({forced.get('why')}, {FORCE_FACTOR}x the budget): nothing started. Check the spend, raise "
            "budget.usd / budget.wall_h if the run should go on, and rerun with --allow-after-forced-stop")


def main(args: Any) -> int:
    run = Run(load_config(args.config))
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
    refused = forced_stop_refusal(run, bool(getattr(args, "allow_after_forced_stop", False)))
    if refused:
        _log(refused)
        return 2
    if not args.skip_preflight and not run.preflight():
        _log("preflight failed: nothing ordered (see preflight.log)")
        return 2
    if args.dry_run:
        _log("dry run: preflight ok, nothing ordered")
        return 0
    if not run.sandbox_prepare():
        _log("sandbox prepare failed: nothing ordered")
        return 2
    if run.in_process:
        # SIGTERM (systemd stop, kill) ends the process through SystemExit, so the exit
        # path (the drain, or with drain_on_exit false the undriven banner) runs
        signal.signal(signal.SIGTERM, _exit_on_sigterm)
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
