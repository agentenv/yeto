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
               refuses --rl-stall-timeout 0); the sandbox host's container count of this
               run is a warning only.

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
     "sandbox": {"envs": "<envs.json>", "docker_host": "ssh://<sandbox host>", "max_run_containers": 40},
     "pool": {"config": "<pool.json>", "interval_s": 30},
     "trainer": {"argv": ["--training-mode", "rl", ...], "usd_per_h": 136},
     "first_group": {"timeout_s": 5400, "poll_s": 60},
     "dashboard": {"port": 8795},
     "budget": {"usd": 450, "wall_h": 3.0, "nc_usd_per_h": 24,  # per NC node
                "silence_min": 30, "release_nc_on_breach": false}}

State: ``~/.yeto/rl-up/<run>/state.json`` (pids, logs, step times). ``up`` resumes: a step
whose process is alive is not started again.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

STATE_ROOT = Path("~/.yeto/rl-up").expanduser()
# the launcher's last line (yeto.cli): ``[yeto] run '<run>' finished: <state> (exit code <n>)``
# a budget / wall-clock breach stops the trainer only; the NC pool is released by force
# once the spend (or the wall time) passes this multiple of the budget
FORCE_FACTOR = 1.2
DEFAULT_STALL_TIMEOUT_S = 900.0  # yeto.launch.fleet.DEFAULT_RL_STALL_TIMEOUT_S
FINISHED = re.compile(r"finished: (\S+).*?\(exit code (-?\d+)\)")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _log(msg: str) -> None:
    print(f"{_now()} [rl up] {msg}", flush=True)


def load_config(path: str) -> dict[str, Any]:
    cfg = json.loads(Path(path).expanduser().read_text())
    for key in ("run", "store", "pool", "trainer"):
        if key not in cfg:
            raise ValueError(f"run config: missing {key!r}")
    known = {"run", "store", "env", "preflight", "sandbox", "pool", "trainer", "first_group", "dashboard",
             "budget", "comment"}
    unknown = set(cfg) - known
    if unknown:
        raise ValueError(f"run config: unknown keys {sorted(unknown)} (known: {sorted(known)})")
    return cfg


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


class Run:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.run = str(cfg["run"])
        self.dir = STATE_ROOT / self.run
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.state: dict[str, Any] = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.env = dict(os.environ)
        self.env.update({k: os.path.expanduser(str(v)) for k, v in (cfg.get("env") or {}).items()})
        self.env.setdefault("YETO_RESOURCE_LEDGER", str(Path(f"~/.yeto/runs/{self.run}/resources.jsonl").expanduser()))
        # Every child is ``python -m ...`` (and spawns more, e.g. ``python -m modal``):
        # without this the directory ``up`` was started from is sys.path[0] of each,
        # and a stray ``h2.py`` / ``json.py`` there replaces a real module.
        self.env.setdefault("PYTHONSAFEPATH", "1")
        self.env.update(_pythonpath_without_start_dir(self.env))
        self.py = sys.executable

    # -- state
    def save(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=1, sort_keys=True))

    def mark(self, step: str, **fields: Any) -> None:
        self.state.setdefault("steps", {})[step] = {"at": _now(), **fields}
        self.save()

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
        return os.path.expanduser(self.cfg["pool"]["config"])

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
        if self.stall_timeout_s() <= 0:
            _log("preflight: --rl-stall-timeout 0 disables the trainer's hang detection; run up refuses it "
                 "(the monitor's silence backstop is 4x this timeout)")
            self.mark("preflight", ok=False)
            return False
        ok = self._cmd(self.nc("pool", "--config", self.pool_config, "--dry-run"), log="preflight.log") == 0
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

        cfg = json.loads(Path(self.pool_config).read_text())
        os.environ.update({k: v for k, v in self.env.items() if k.startswith(("YETO_NC_", "NC_"))})
        proc = pool_run.pool_process(cfg)
        if proc and proc.get("alive"):
            _log(f"pool: loop already running (PID {proc.get('pid')})")
        else:
            self._spawn("pool", self.nc("pool", "--config", self.pool_config,
                                        "--interval", str(int(self.cfg["pool"].get("interval_s", 30)))))
        self.state.setdefault("t0", time.time())
        self.mark("pool")

    def _objects(self):
        from yeto.cloud.s3_store import head_objects  # noqa: PLC0415

        return head_objects(self.cfg["store"], environ=self.env)

    def buffer_objects(self) -> int:
        try:
            return sum(1 for _ in self._objects().list("sample-buffer/shards/"))
        except Exception as error:  # noqa: BLE001
            _log(f"store list failed: {type(error).__name__}: {error}"[:300])
            return 0

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
        return argv

    def trainer(self) -> None:
        self._spawn("trainer", self.yeto("launch", *self.trainer_argv()))
        for key in ("trainer_exit_seen", "trainer_finished"):  # of an earlier launcher of this run
            self.state.pop(key, None)
        self.state.setdefault("trainer_t0", time.time())
        self.mark("trainer")

    def dashboard_args(self) -> list[str]:
        """``--store`` plus the run's local records (``--run``: the launcher's fleet record
        names the trainer's GPU, so its cost is priced), the cost bar's cap
        (``budget.usd``) and an optional price table (``dashboard.prices``)."""
        dash = self.cfg.get("dashboard") or {}
        argv = ["--store", self.cfg["store"], "--run", self.run]
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
        b = self.cfg.get("budget") or {}
        now = time.time()
        nc = self.nc_node_hours(now) * float(b.get("nc_usd_per_h", 24))
        tr = (now - self.state.get("trainer_t0", now)) / 3600 * float(self.cfg["trainer"].get("usd_per_h", 0))
        return nc + tr

    def nc_node_hours(self, now: float | None = None) -> float:
        """Billed NC node hours of this run: each pool state member from its grant (a ticket
        never granted bills nothing) to its release (``charged_to`` of a gone member) or now,
        counted from this run's ``t0``. Without a readable pool state: the wall time since
        ``t0`` times the pool target (every node assumed billed)."""
        now = time.time() if now is None else now
        t0 = float(self.state.get("t0", now))
        members = self.pool_members()
        if members is None:  # no state yet / unreadable: the conservative estimate
            try:
                nodes = max(1, int(json.loads(Path(self.pool_config).read_text()).get("target", 1)))
            except Exception:  # noqa: BLE001
                nodes = 1
            return max(0.0, now - t0) / 3600 * nodes
        hours = 0.0
        for m in members:
            if m.get("granted") is None:
                continue
            start = max(float(m["granted"]), t0)
            end = float(m.get("charged_to") or now) if m.get("state") == "gone" else now
            hours += max(0.0, end - start) / 3600
        return hours

    def pool_members(self) -> list[dict[str, Any]] | None:
        """The members of the pool state file (None: no readable state)."""
        try:
            pool_cfg = json.loads(Path(self.pool_config).read_text())
            nc_dir = Path(self.env.get("YETO_NC_STATE_DIR", "~/.yeto/nc")).expanduser()  # tunnel.state_dir()
            doc = json.loads((nc_dir / f"pool-{pool_cfg['shard_base']}.json").read_text())
            return list(doc["members"])
        except Exception:  # noqa: BLE001
            return None

    def run_containers(self) -> int | None:
        sb = self.cfg.get("sandbox") or {}
        if not sb.get("docker_host"):
            return None
        try:  # an ssh hang to the sandbox host must not end the monitor (and its guards)
            out = subprocess.run(["docker", "-H", sb["docker_host"], "ps", "-q", "--filter",
                                  f"label=ai.yeto.run={self.run}"],
                                 env=self.env, capture_output=True, text=True, timeout=60, check=False)
        except (subprocess.TimeoutExpired, OSError) as error:
            _log(f"sandbox: docker ps failed: {type(error).__name__}")
            return None
        return len(out.stdout.split()) if out.returncode == 0 else None

    def breach(self, *, trainer: bool = True) -> str | None:
        """Why the run must stop, else None. ``trainer=False`` (the first-group wait, before
        this invocation starts the trainer) skips the trainer log: on a resumed ``run up`` it is
        the log of the previous, stopped launcher, whose failure is already handled."""
        b = self.cfg.get("budget") or {}
        why = None
        if "t0" in self.state and time.time() - self.state["t0"] > float(b.get("wall_h", 3)) * 3600:
            why = "wall clock"
        elif self.spend_usd() > float(b.get("usd", 1e9)):
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
            if not done and not _worker_alive(self.run):  # a killed log streamer: the worker may still run
                why = "trainer: launcher exited without a result"
            elif done and (done[-1][0] == "FAILED" or done[-1][1] != "0"):
                why = f"trainer: launcher finished {done[-1][0]} (exit code {done[-1][1]})"
        if why is None and tr and _alive(tr["pid"]):
            why = self.silence(time.time())
        limit = (self.cfg.get("sandbox") or {}).get("max_run_containers")
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

    def budget_problem(self) -> str | None:
        """A budget that would stop the run at once: ``budget.usd`` 0 (the guide's
        template value) with a price set, so the first estimated dollar is a breach."""
        b = self.cfg.get("budget") or {}
        priced = float(self.cfg["trainer"].get("usd_per_h", 0) or 0) > 0 or float(b.get("nc_usd_per_h", 24)) > 0
        if "usd" in b and float(b["usd"]) <= 0 and priced:
            return (f"budget.usd is {b['usd']} while prices are set (trainer.usd_per_h / budget.nc_usd_per_h): "
                    "the first estimated dollar would stop the run; set the spend cap")
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
        return max(4 * self.stall_timeout_s(), float(b.get("silence_min", 30)) * 60)

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
        self._cmd(self.yeto("down", self.run), log="down.log", timeout=900)
        self.mark("trainer_down", why=why)
        _log(f"trainer stopped ({why}); the NC pool and the sandboxes are KEPT and still bill. "
             f"Release them with: yeto run down --config <run.json> "
             f"(or python -m yeto.launch.nc pool-stop --config {self.pool_config})")

    def on_breach(self, why: str) -> None:
        """Budget / wall clock with ``release_nc_on_breach``: the full ``down``; anything
        else (and by default): the trainer only."""
        if self.release_nc_on_breach and why.startswith(("wall clock", "budget")):
            _log(f"{why}: budget.release_nc_on_breach is set: full down (NC pool included)")
            self.down()
            return
        self.trainer_down(why)

    def monitor(self, poll_s: float = 60.0) -> None:
        while True:
            why = self.breach()
            if why:
                self.on_breach(why)
                if why.startswith(("wall clock", "budget")) and not self.release_nc_on_breach:
                    self.watch_nc_after_breach(why, poll_s)
                return
            tr = (self.state.get("procs") or {}).get("trainer")
            if tr and not _alive(tr["pid"]) and not self.state.get("trainer_exit_seen"):
                _log(f"trainer launcher PID {tr['pid']} exited")
                self.state["trainer_exit_seen"] = _now()
                self.save()
            if self.state.get("trainer_finished") and self.on_trainer_finished():
                return
            time.sleep(poll_s)

    def on_trainer_finished(self) -> bool:
        """The trainer finished its rounds (``SUCCEEDED``). The samplers have nothing left
        to feed, but their nodes and the sandboxes bill until released: with
        ``budget.release_nc_on_finish`` the full ``down`` runs now (returns True: the
        monitor ends); otherwise a banner says so (once), and the wall-clock guard stays."""
        if (self.cfg.get("budget") or {}).get("release_nc_on_finish"):
            _log("trainer finished: budget.release_nc_on_finish is set: full down (NC pool included)")
            self.down()
            return True
        if not self.state.get("finished_banner"):
            _log("trainer finished (SUCCEEDED); the NC pool and the sandboxes still bill. "
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
        usd_cap = float(b.get("usd", 1e9)) * FORCE_FACTOR
        wall_cap = float(b.get("wall_h", 3)) * 3600 * FORCE_FACTOR
        while True:
            members = self.pool_members()
            billed = None if members is None else [m for m in members if m.get("state") != "gone"]
            if billed is not None and not billed:
                _log("NC pool: no node bills any more; monitor done")
                return
            spend = self.spend_usd()
            elapsed = time.time() - float(self.state.get("t0", time.time()))
            n = "?" if billed is None else len(billed)
            _log(f"NC STILL BILLING after '{why}': {n} node(s), spend ${spend:.0f}; release with: "
                 f"yeto run down --config <run.json> (forced at ${usd_cap:.0f} or {wall_cap / 3600:.1f} h)")
            if spend > usd_cap or elapsed > wall_cap:
                cap = f"spend ${spend:.0f} > ${usd_cap:.0f}" if spend > usd_cap else \
                    f"wall {elapsed / 3600:.1f} h > {wall_cap / 3600:.1f} h"
                _log(f"BREAKER: {cap} ({FORCE_FACTOR}x the budget): forcing pool-stop")
                rc = self._cmd(self.nc("pool-stop", "--config", self.pool_config), log="down.log", timeout=1800)
                self.mark("nc_forced_stop", why=cap, rc=rc)
                return
            time.sleep(poll_s)

    # -- down
    def down(self) -> bool:
        ok = True
        self._cmd(self.yeto("dashboard", "export", *self.dashboard_args(), "-o",
                            str(self.dir / "dashboard.html")), log="down.log", timeout=600)  # read-only copy
        dash = (self.state.get("procs") or {}).get("dashboard")
        if dash and _alive(dash["pid"]):
            os.kill(int(dash["pid"]), signal.SIGTERM)
        tr = (self.state.get("procs") or {}).get("trainer")
        if tr and _alive(tr["pid"]):
            os.kill(int(tr["pid"]), signal.SIGINT)
            time.sleep(10)
        ok &= self._cmd(self.yeto("down", self.run), log="down.log", timeout=900) == 0
        ok &= self._cmd(self.nc("pool-stop", "--config", self.pool_config), log="down.log", timeout=1800) == 0
        if self.envs:
            ok &= self._cmd(self.yeto("sandbox", "teardown", "--envs", self.envs, "--store", self.cfg["store"]),
                            log="down.log", timeout=600) == 0
        sb = self.cfg.get("sandbox") or {}
        if sb.get("docker_host"):  # Harbor trial containers of this run (ai.yeto.run label) and their networks
            host = sb["docker_host"]
            ids = subprocess.run(["docker", "-H", host, "ps", "-aq", "--filter", f"label=ai.yeto.run={self.run}"],
                                 env=self.env, capture_output=True, text=True, timeout=120, check=False).stdout.split()
            if ids:
                self._cmd(["docker", "-H", host, "rm", "-f", *ids], log="down.log", timeout=600)
            left = self.run_containers()
            _log(f"sandbox: {left} container(s) of this run left")
            ok &= left == 0
        ok &= self._cmd(self.nc("pool-status", "--config", self.pool_config), log="down.log", timeout=300) == 0
        self.mark("down", ok=bool(ok))
        _log("down: " + ("verified" if ok else "NOT verified, see down.log"))
        return bool(ok)

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {"run": self.run, "state": self.state, "buffer_objects": self.buffer_objects(),
                               "spend_usd_estimate": round(self.spend_usd(), 1),
                               "procs": {k: dict(v, alive=_alive(v["pid"])) for k, v in
                                         (self.state.get("procs") or {}).items()}}
        out["sandbox_run_containers"] = self.run_containers()
        for name, argv in (("pool", self.nc("pool-status", "--config", self.pool_config)),
                           ("dispense", [self.py, "-m", "yeto.rl.engine.store_dispense", "status", "--store",
                                         self.cfg["store"], "--namespace", self._namespace(), "--brief"])):
            p = subprocess.run(argv, env=self.env, capture_output=True, text=True, timeout=300, check=False)
            out[name] = (p.stdout or p.stderr)[-3000:]
            if name == "pool":
                try:
                    doc = json.JSONDecoder().raw_decode((p.stdout or "").lstrip())[0]
                    out["samplers"] = {"target": doc.get("target"), "nodes": doc.get("nodes")}
                except (ValueError, TypeError, AttributeError):
                    pass
        return out

    def scale(self, samplers: int, retire: list[str] | None = None) -> int:
        """``yeto run scale --samplers N``: ``pool-scale`` on the run's pool."""
        argv = self.nc("pool-scale", "--config", self.pool_config, "--target", str(int(samplers)))
        for ticket in retire or ():
            argv += ["--retire", ticket]
        return subprocess.run(argv, env=self.env, check=False, timeout=300).returncode

    def _namespace(self) -> str:
        cfg = json.loads(Path(self.pool_config).read_text())
        return str(((cfg.get("sampler") or {}).get("dispense") or {}).get("namespace") or "task-dispense")


def _worker_alive(run: str) -> bool:
    """The detached ``yeto launch`` worker of ``run`` (the run registry) still runs."""
    from yeto import runs  # noqa: PLC0415

    try:
        return _alive((runs.load_run(run) or {}).get("pid"))
    except Exception:  # noqa: BLE001 - an unreadable registry: no evidence it runs
        return False


def _alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True
    return True


def add_parser(sub: Any) -> None:
    p = sub.add_parser("run", help="one run config, one command: up / status / down of an async GRPO run "
                       "(yeto rl is the single-process CyberGym PPO path)")
    rsub = p.add_subparsers(dest="rl_command", metavar="{up,status,down,scale}")
    rsub.required = True
    up = rsub.add_parser("up", help="preflight, sandbox, pool, first group, trainer, dashboard, monitor")
    up.add_argument("--config", required=True)
    up.add_argument("--dry-run", action="store_true", help="preflight only; nothing is ordered")
    up.add_argument("--skip-preflight", action="store_true", help="resume: preflight already passed")
    up.add_argument("--no-monitor", action="store_true", help="return after the dashboard starts")
    up.add_argument("--trainer-now", action="store_true",
                    help="start the trainer without waiting for the first sample group (NC and the "
                    "trainer GPUs come up in parallel)")
    for name in ("status", "down"):
        q = rsub.add_parser(name)
        q.add_argument("--config", required=True)
    sc = rsub.add_parser("scale", help="add or remove NC sampler nodes while the run trains")
    sc.add_argument("--config", required=True)
    sc.add_argument("--samplers", type=int, required=True, help="sampler nodes wanted (0 or more)")
    sc.add_argument("--retire", action="append", default=[], metavar="TICKET",
                    help="on scale down, the node ticket to release (repeatable)")


def main(args: Any) -> int:
    run = Run(load_config(args.config))
    if args.rl_command == "status":
        print(json.dumps(run.status(), indent=1, default=str))
        return 0
    if args.rl_command == "down":
        return 0 if run.down() else 1
    if args.rl_command == "scale":
        return run.scale(args.samplers, args.retire)
    if not args.skip_preflight and not run.preflight():
        _log("preflight failed: nothing ordered (see preflight.log)")
        return 2
    if args.dry_run:
        _log("dry run: preflight ok, nothing ordered")
        return 0
    if not run.sandbox_prepare():
        _log("sandbox prepare failed: nothing ordered")
        return 2
    run.pool()
    if getattr(args, "trainer_now", False):
        _log("--trainer-now: the trainer starts now, in parallel with the NC node setup")
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
