"""The pool commands of ``python -m yeto.launch.nc``: ``pool`` (with ``--dry-run``),
``pool-stop`` and ``pool-status``.

A National Compute lease is **not** ended at ``lease_until``: the ticket stays
``active`` and billed until it is deleted. So every way out of the pool loop
either releases the nodes it created or prints, loudly, which nodes are still
billing and the command that releases them:

* ``pool-stop --config F`` asks a running pool to drain every member (a stop
  file the loop reads each tick) and waits for it; when no pool process is
  alive (it crashed, was killed, or its host went away) it releases the
  members of the state file and the open tickets the ledger holds for the run
  itself. It only ever deletes tickets this tool created for this run, by id
  (never ``--all``);
* SIGTERM, SIGINT and SIGHUP make the loop drain every member before it exits;
* an exit with members left (a drain that failed, an error) prints the
  billing banner; ``pool-status`` prints it too when the state file holds
  members and no pool process is alive.

The process files live next to the state file in the state dir
(``YETO_NC_STATE_DIR``): ``pool-<shard_base>.lock`` (an ``flock`` the loop holds
for its whole life: one pool instance per pool, released by the kernel when the
process dies, whatever the host name says), ``pool-<shard_base>.pid`` (the loop's
pid and host, for display), ``pool-<shard_base>.stop`` (a stop request) and
``pool-<shard_base>.scale.json`` (``pool-scale``'s target; it records the run id, a file of
another run is ignored, and ``pool-stop`` removes it once the pool is released).
``pool-stop`` releasing from the state file and ``yeto cleanup --include-pools``
take the same lock: neither releases a node while a pool loop runs.
"""

from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import shutil
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

from ..pool import status as island_status
from ..pool.island_pool import GONE, IslandPool, Member
from . import cli, elastic, ssh_sampler, tunnel
from .client import LEASE_HOURS, NcTools
from .provider import is_national_compute, provider_kind

LEASE_WARNING = ("National Compute does NOT end a lease at lease_until: the node stays active and billed "
                 "until its ticket is deleted. Keep this pool running, or release its nodes with "
                 "`python -m yeto.launch.nc pool-stop --config <file>`.")
SANDBOX_WARNING = ("Daytona sandboxes are created with auto-stop off: a sandbox stays billed until it is "
                   "deleted. Keep this pool running, or delete its sandboxes with "
                   "`python -m yeto.launch.nc pool-stop --config <file>`.")
DRAIN_ATTEMPTS = 3


CLOUD_WARNING = ("Cloud rollout islands ({kind}) bill until they are released: keep this pool running, "
                 "or release them with `python -m yeto.launch.nc pool-stop --config <file>`.")


SHARED_BASE_WARNING = ("only one pool may run shard_base {base!r}: the pool lock is an flock in this host's state "
                       "directory, so a second pool with the same shard_base on another machine is NOT detected; "
                       "both would order nodes, fence and continue the same shards of the same store")


def lease_warning(cfg: dict[str, Any], config_path: str) -> str:
    """The billing warning of the pool's provider (a National Compute lease, a sandbox, a cloud island)."""
    kind = provider_kind(cfg)
    text = LEASE_WARNING if is_national_compute(cfg) else SANDBOX_WARNING if kind == "daytona" else \
        CLOUD_WARNING.format(kind=kind)
    return text.replace("<file>", config_path)
STOP_SIGNALS = ("SIGTERM", "SIGINT", "SIGHUP")


def _say(msg: str) -> None:
    cli._log(msg)


# -- process files -------------------------------------------------------------

def pid_path(cfg: dict[str, Any]) -> Path:
    return tunnel.state_dir() / f"pool-{cfg['shard_base']}.pid"


def lock_path(cfg: dict[str, Any]) -> Path:
    return tunnel.state_dir() / f"pool-{cfg['shard_base']}.lock"


def acquire_lock(path: Path, *, run: str | None = None, role: str = "pool"):
    """The open lock file holding ``flock(LOCK_EX | LOCK_NB)`` on ``path`` (it records
    the run, pid and role), or None when another process holds it. Closing the file
    (or the process ending) releases it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    fh.seek(0)
    fh.truncate()
    fh.write(json.dumps({"run": run, "pid": os.getpid(), "role": role, "since": time.time()}))
    fh.flush()
    return fh


def lock_holder(path: Path) -> dict[str, Any] | None:
    """What the lock file records when a live process holds the lock; None when free."""
    if not path.exists():
        return None
    fh = acquire_lock_probe(path)
    if fh is not None:
        fh.close()
        return None
    try:
        return json.loads(path.read_text() or "{}")
    except (OSError, ValueError):
        return {}


def acquire_lock_probe(path: Path):
    """Try the lock without rewriting the file (a probe); the open file or None."""
    try:
        fh = open(path, "r")
    except OSError:
        return None
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


def run_pool_locks(run: str) -> list[Path]:
    """The pool lock files of the state dir that record ``run``."""
    out = []
    for path in sorted(tunnel.state_dir().glob("pool-*.lock")):
        try:
            doc = json.loads(path.read_text() or "{}")
        except (OSError, ValueError):
            continue
        if doc.get("run") == run:
            out.append(path)
    return out


def stop_path(cfg: dict[str, Any]) -> Path:
    return tunnel.state_dir() / f"pool-{cfg['shard_base']}.stop"


def retry_path(cfg: dict[str, Any]) -> Path:
    return tunnel.state_dir() / f"pool-{cfg['shard_base']}.retry"


def scale_path(cfg: dict[str, Any]) -> Path:
    """``pool-scale``'s control file: the target the loop holds (it outlives restarts)."""
    return tunnel.state_dir() / f"pool-{cfg['shard_base']}.scale.json"


def scale_lock_path(cfg: dict[str, Any]) -> Path:
    """Serializes ``pool-scale`` commands with each other (never the pool lock)."""
    return tunnel.state_dir() / f"pool-{cfg['shard_base']}.scale.lock"


def read_scale(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """The scale file of this run; one another run (same shard_base) left behind, or one
    without a run id, is ignored."""
    try:
        doc = json.loads(scale_path(cfg).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get("run") != cli.run_id(cfg):
        return None
    return doc


def check_sampler_spec(cfg: dict[str, Any], pc=None) -> dict[str, Any]:
    """Render the sampler spec of the pool's first member and build, locally, every
    part the sampler builds at start before it touches the engine: the island config
    (required keys, sampling params), the engine adapter from its section (unknown
    section keys, server fields or engine options are refused; no process starts),
    the rollout fields the engine must return, and the reward. Raises ValueError: a spec that fails here fails on
    every node, after its setup, so nothing is ordered for it."""
    from yeto.rl.engine.engine_sampler_entry import (  # noqa: PLC0415
        apply_rollout_quant,
        build_engine,
        build_reward,
        island_config,
    )
    from yeto.rl.engine.rollout_data_path import normalize_spec  # noqa: PLC0415

    pc = pc or cli.pool_config(cfg, allowed=None)
    member = Member(pc.accounts[0], "<ticket>", f"{pc.shard_base}-0",
                    data_slot=f"0/{pc.data_slots}" if pc.data_slots else None)
    try:
        spec = cli.render_spec(cfg, member, [f"{pc.shard_base}-{i}" for i in range(pc.target)])
        from yeto.launch.agentic_setup import strip  # noqa: PLC0415

        # the data path's defaults first, as the sampler applies them at start (syncer:
        # no round limit, a member store under work_dir)
        checked = apply_rollout_quant(normalize_spec(strip(spec)))
        island_config(checked)  # required keys, sampling parameters
        build_engine(checked)  # engine section, server + options dataclasses, rollout_fields (no process)
        build_reward(checked)  # reward_kind and the reward function import
    except Exception as error:  # noqa: BLE001 - every refusal is a config error here
        raise ValueError(f"sampler spec {cfg.get('spec')!r} is refused by the sampler: "
                         f"{type(error).__name__}: {error}"[:500]) from None
    return spec


def pool_process(cfg: dict[str, Any]) -> dict[str, Any] | None:
    """The pid file of the pool's loop with ``alive`` (None: no pid file and no lock
    holder). The pool lock decides on this host: held means alive. Without a lock
    file, a pid file of another host counts as alive (this host cannot tell)."""
    holder = lock_holder(lock_path(cfg))
    try:
        doc = json.loads(pid_path(cfg).read_text())
    except (OSError, ValueError):
        doc = None
    if holder is not None:
        return dict(doc or {}, pid=(doc or {}).get("pid", holder.get("pid")), alive=True, lock=holder)
    if doc is None:
        return None
    if lock_path(cfg).exists():  # the lock is free: no loop of this pool runs on this host
        return dict(doc, alive=False)
    alive = True
    if doc.get("host") == socket.gethostname():
        try:
            os.kill(int(doc["pid"]), 0)
        except ProcessLookupError:
            alive = False
        except PermissionError:
            alive = True
        except (KeyError, TypeError, ValueError):
            alive = False
    return dict(doc, alive=alive)


def _write_pid(cfg: dict[str, Any], config_path: str) -> None:
    path = pid_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "started": time.time(),
                                "config": str(Path(config_path).resolve())}))


def _remove(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


# -- billing -------------------------------------------------------------------

def billed_members(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Members of a pool state document that may still hold a ticket."""
    return [m for m in doc.get("members", []) if m.get("state") != GONE]


def _until(value: float | None) -> str:
    return "-" if value is None else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(value))


def stop_command(config_path: str) -> str:
    return f"python -m yeto.launch.nc pool-stop --config {config_path}"


WORDING = {  # provider kind -> how messages name its machines and their billing
    "nc": {"unit": "National Compute node", "units": "National Compute node(s)",
           "billing": "A lease is NOT ended at lease_until: the node stays active and billed until it is deleted."},
    "daytona": {"unit": "sandbox", "units": "sandbox(es)",
                "billing": "A sandbox stays billed until it is deleted (auto-stop is off)."},
    "sky": {"unit": "sky cluster", "units": "sky cluster(s)",
            "billing": "A cluster stays billed until it is taken down (sky down)."},
    "modal": {"unit": "Modal island", "units": "Modal island(s)",
              "billing": "A Modal function call bills until it is cancelled or reaches its timeout."},
}
GENERIC_WORDING = {"unit": "node", "units": "node(s)", "billing": "A node stays billed until it is released."}


def wording(cfg: dict[str, Any] | None = None, *, kind: str | None = None) -> dict[str, str]:
    """The names a message uses for the machines of a pool's provider."""
    if kind is None:
        kind = provider_kind(cfg or {})
    return WORDING.get(kind, GENERIC_WORDING)


def billing_banner(config_path: str, members: list[dict[str, Any]], *, why: str = "", national: bool = True,
                   kind: str | None = None) -> str:
    bar = "!" * 78
    w = wording(kind=kind or ("nc" if national else "daytona"))
    head = "NODES STILL BILLING" if national else "STILL BILLING"
    lines = [bar, f"{head}: {len(members)} {w['units']} of this pool may still be "
                  "active" + (f" ({why})" if why else "") + ".", w["billing"]]
    now = time.time()
    for m in members:
        late = m.get("lease_until") is not None and now > float(m["lease_until"])
        until = m.get("lease_until")
        lines.append(f"  {'account' if national else 'slot'} {m.get('account')} ticket {m.get('ticket')} "
                     f"shard {m.get('shard')} state {m.get('state')}"
                     + (f" lease_until {_until(until)}" if national or until is not None else "")
                     + ("  PAST lease_until, STILL BILLED" if late else ""))
    lines += ["Release them with:", f"  {stop_command(config_path)}"]
    if national:  # ``down`` is the National Compute ticket call
        lines.append("or one by one: python -m yeto.launch.nc down --account <account> --ticket <ticket>")
    lines.append(bar)
    return "\n".join(lines)


def _print_banner(text: str) -> None:
    print(text, file=sys.stderr, flush=True)
    print(text, flush=True)


# -- dry run -------------------------------------------------------------------

def daytona_problems(cfg: dict[str, Any]) -> list[str]:
    """Offline checks of a ``provider.kind: daytona`` pool (no Daytona call)."""
    from .daytona_provider import from_config  # noqa: PLC0415

    problems = []
    try:
        provider = from_config(cfg, cli.run_id(cfg), sdk=object(), client=object())
    except (ValueError, TypeError) as error:
        return [str(error)]
    if not os.environ.get(provider.api_key_env):
        problems.append(f"{provider.api_key_env} is not set (the Daytona API key)")
    if importlib.util.find_spec("daytona") is None:
        problems.append("the 'daytona' Python package is not installed")
    return problems


def model_problems(uri: str, env_file: str | os.PathLike[str]) -> list[str]:
    """What is missing of the model directory ``uri`` in the bucket, read with the
    nodes' credentials from the store env file: ``config.json`` and every weight
    shard ``model.safetensors.index.json`` names (or ``model.safetensors``). One
    paginated LIST of the prefix plus the index; no weight file is read."""
    from yeto.launch import model_check  # noqa: PLC0415

    return model_check.store_problems(elastic.s3_fence_store(uri, env_file).objects, uri)


def store_layout_problems(cfg: dict[str, Any], env_file: str | os.PathLike[str], bucket: str) -> list[str]:
    """Store-layout checks of ``pool --dry-run``: ``run_prefix`` does not repeat
    ``runs/``, the model is in the bucket (a missing upload otherwise surfaces only
    after a billed node was ordered and set up) and ``dest`` is the directory the
    sampler container mounts at the spec's model path."""
    problems = []
    prefix = str(cfg.get("run_prefix") or "").strip("/")
    if prefix == "runs" or prefix.startswith("runs/"):
        problems.append(f"run_prefix {cfg['run_prefix']!r}: the samplers write to s3://{bucket}/runs/<run_prefix>, "
                        f"so this one lands in s3://{bucket}/runs/{prefix}; drop the leading 'runs/'")
    pull = cfg.get("model_pull")
    if not pull:
        return problems
    dest = str(pull.get("dest", ssh_sampler.MODEL_HOST)).rstrip("/")
    if dest != ssh_sampler.MODEL_HOST:
        problems.append(f"model_pull.dest {dest!r}: the sampler container mounts only {ssh_sampler.MODEL_HOST} "
                        f"(at {ssh_sampler.MOUNTS[ssh_sampler.MODEL_HOST]}); leave dest unset")
    uri = f"s3://{bucket}/{str(pull['src_key']).strip('/')}"
    try:
        missing = model_problems(uri, env_file)
    except Exception as error:  # noqa: BLE001 - reported, the operator decides
        problems.append(f"model_pull: cannot list {uri} with the store env file: {error!r}"[:400])
    else:
        problems += [f"model_pull: {m}: upload the model to the bucket first "
                     "(docs/KIMI_K31_ASYNC_RL_GUIDE.md, Model weights)" for m in missing]
    return problems


def dry_run(cfg: dict[str, Any], config_path: str) -> int:
    """Validate the config and print the orders and the spend the pool would make;
    orders nothing and calls no National Compute API."""
    external = not is_national_compute(cfg)
    problems: list[str] = []
    try:
        pc = cli.pool_config(cfg, allowed=None if external else cli.allowed_accounts())
    except (ValueError, TypeError) as error:
        print(f"pool --dry-run: INVALID: {error}")
        return 2
    notes = list(cli.pool_config_warnings(cfg))
    if cfg.get("route", "ssh") == "ssh":
        try:
            spec = check_sampler_spec(cfg, pc)
            print(f"sampler spec: {cfg['spec']} passes the sampler's checks (sampler_id {spec.get('sampler_id')}, "
                  f"buffer_shards {spec.get('buffer_shards')})")
            from yeto.launch import agentic_setup  # noqa: PLC0415

            from yeto.rl.engine.agentic_round import drain_settings, drain_timeout_warning  # noqa: PLC0415

            if isinstance(spec.get("agentic"), dict):
                drain_settings(spec, drain=True)  # a bad drain_timeout_s / abandon_grace_s: a problem here
            warning = drain_timeout_warning(spec)  # none for a spec that resolves to the live switch
            if warning is not None:
                notes.append(warning)
            setup_lines, setup_problems = agentic_setup.dry_run_report(spec, base_dir=Path(cfg["spec"]).parent)
            for line in setup_lines:
                print(f"agentic.setup: {line}")
            problems += [f"agentic.setup: {p}" for p in setup_problems]
        except ValueError as error:
            problems.append(str(error))
        env_file = Path(str(cfg["r2_env_file"])).expanduser()
        if not env_file.exists():
            problems.append(f"store env file {cfg['r2_env_file']!r} does not exist")
        else:
            if env_file.stat().st_mode & 0o077:
                problems.append(f"store env file {cfg['r2_env_file']!r} must be mode 0600")
            bucket_env = cfg.get("bucket_env", "R2_BUCKET")
            if bucket_env not in elastic.read_env_file(env_file):
                problems.append(f"store env file has no {bucket_env} (the bucket key; set 'bucket_env')")
            else:
                problems += store_layout_problems(cfg, env_file, elastic.read_env_file(env_file)[bucket_env])
    from ..pool import provider as registry  # noqa: PLC0415

    entry = registry.registered(provider_kind(cfg))
    check = registry.resolve(entry.offline_problems) if entry is not None else None
    if check is not None:  # the provider kind's own offline checks (registered)
        problems += list(check(cfg))
    elif external:
        module = str(cfg["provider"]["factory"]).partition(":")[0]
        if importlib.util.find_spec(module) is None:
            problems.append(f"provider.factory module {module!r} is not importable")
    else:
        section = {k: v for k, v in (cfg.get("provider") or {}).items() if k != "kind"}
        try:
            tools = NcTools.from_env(section)
            for kind in ("manage", "query"):
                argv = getattr(tools, kind)
                if not shutil.which(argv[0]):
                    problems.append(f"National Compute {kind} tool {argv[0]!r} is not on PATH "
                                    f"(NC_{kind.upper()} / provider.{kind})")
            if not tools.tool_dir:
                notes.append("NC_TOOL_DIR / provider.tool_dir is unset (the workspace relay needs it)")
        except ValueError as error:
            problems.append(str(error))
    ecfg = pc.elastic
    hourly = pc.hourly()
    w = wording(cfg)
    peak = min(len(pc.accounts), ecfg.max_nodes if ecfg.max_nodes is not None else 2 * pc.target)
    if not external:
        print(f"pool {cli.run_id(cfg)}: route {cfg.get('route', 'ssh')}, machine type {cli.machine_type(cfg)}, "
              f"target {pc.target} sampler(s) on {len(pc.accounts)} account(s) (one node per account)")
        print("orders at start:")
        for i in range(pc.target):
            slot = f" data_slot {i}/{pc.data_slots}" if pc.data_slots else ""
            print(f"  account {pc.accounts[i]}: 1 x {cli.machine_type(cfg)} node, shard {pc.shard_base}-{i}{slot}")
        spare = pc.accounts[pc.target:]
        print(f"replacement accounts (a lease is granted for up to {LEASE_HOURS:g} h and must be released; "
              f"a replacement is ordered "
              f"{pc.replace_lead_s:.0f} s before its end): {spare if spare else 'none (gaps at every lease end)'}")
    else:
        print(f"pool {cli.run_id(cfg)}: provider {provider_kind(cfg)}, data path {cfg.get('data_path', 'r2')}, "
              f"target {pc.target} sampler(s) on {len(pc.accounts)} slot(s) (one {w['unit']} per slot)")
        print("orders at start:")
        for i in range(pc.target):
            slot = f" data_slot {i}/{pc.data_slots}" if pc.data_slots else ""
            print(f"  slot {pc.accounts[i]}: 1 x {w['unit']}, shard {pc.shard_base}-{i}{slot}")
    if external and cli.pool_price(cfg)[0] is None:
        notes.append(f"the provider config sets no price_usd_h: the budget counts each {w['unit']} hour at the "
                     "National Compute MI355X rate; set price_usd_h to the provider's price")
    if cli.pool_price(cfg)[0] is None:  # the National Compute rate, in ucr
        print(f"estimated spend: {hourly:.0f} ucr/h per node ({hourly / 1e6:g} credits/h, the 8-GPU MI355X rate); "
              f"steady {pc.target * hourly:.0f} ucr/h ({pc.target * hourly / 1e6:g} credits/h); "
              f"up to {peak * hourly:.0f} ucr/h ({peak * hourly / 1e6:g} credits/h) while replacements overlap")
    else:
        print(f"estimated spend: {cli.pool_hourly_text(cfg, hourly)} per node hour; steady "
              f"{cli.pool_hourly_text(cfg, pc.target * hourly)}/h; up to "
              f"{cli.pool_hourly_text(cfg, peak * hourly)}/h while replacements overlap")
    if ecfg.max_run_spend_ucr is None and ecfg.max_run_spend_usd is None:
        notes.append("elastic.max_run_spend_ucr is unset (and max_run_spend_usd): the run's spend has no cap")
    if ecfg.max_run_spend_ucr is not None:
        print(f"run spend cap: {ecfg.max_run_spend_ucr} ucr ({ecfg.max_run_spend_ucr / hourly / pc.target:.1f} h "
              f"at the steady rate)")
    if ecfg.max_run_spend_usd is not None:
        from ..pool.money import to_ucr  # noqa: PLC0415

        cap = to_ucr(float(ecfg.max_run_spend_usd), "usd")
        print(f"run spend cap: ${float(ecfg.max_run_spend_usd):.2f} ({cap / hourly / pc.target:.1f} h at the steady "
              "rate)")
    if not external:
        print(f"stops ordering and drains below a balance of {pc.stop_below_ucr} ucr")
    print(f"WARNING: {lease_warning(cfg, config_path)}")
    print("The control loop runs in the foreground on this host: keep it up for the whole run "
          "(tmux / systemd on an always-on machine).")
    print(f"WARNING: {SHARED_BASE_WARNING.format(base=cfg['shard_base'])}")
    for note in notes:
        print(f"note: {note}")
    for problem in problems:
        print(f"problem: {problem}")
    print("pool --dry-run: " + ("ok, nothing ordered" if not problems else f"{len(problems)} problem(s), "
                                                                        "nothing ordered"))
    return 0 if not problems else 1


# -- the loop ------------------------------------------------------------------

class _Signals:
    """SIGTERM / SIGINT / SIGHUP: remember it and wake the loop's wait."""

    def __init__(self) -> None:
        self.name: str | None = None
        self.wake = threading.Event()
        self._old: dict[int, Any] = {}

    def install(self) -> None:
        for name in STOP_SIGNALS:
            sig = getattr(signal, name, None)
            if sig is not None:
                self._old[sig] = signal.signal(sig, self._handle)

    def restore(self) -> None:
        for sig, old in self._old.items():
            signal.signal(sig, old)

    def _handle(self, signum, frame) -> None:  # noqa: ARG002
        self.name = signal.Signals(signum).name
        self.wake.set()


def _drain_all(pool: IslandPool, why: str, *, sleep=None, wait_s: float = 10.0) -> list[Member]:
    """Stop the pool and retry failed drains a few times; returns the members left."""
    return pool.drain_all(why, attempts=DRAIN_ATTEMPTS, wait_s=wait_s, sleep=sleep or time.sleep)


def event_log(cfg: dict[str, Any]) -> island_status.EventLog:
    """The pool's event log (its events file, the run id, the ``event_command`` hook),
    the same for the foreground loop and a :class:`~yeto.launch.pool.controller.RolloutController`."""
    hook = island_status.command_hook(cfg["event_command"]) if cfg.get("event_command") else None
    return island_status.EventLog(path=cli.pool_events_path(cfg), run=cli.run_id(cfg), log=cli._log, hook=hook)


def run_pool(cfg: dict[str, Any], config_path: str, *, interval: float, pool: IslandPool | None = None,
             signals: _Signals | None = None) -> int:
    """The foreground control loop of a pool (``pool --config F``)."""
    external = not is_national_compute(cfg)
    pc = cli.pool_config(cfg, allowed=None if external else cli.allowed_accounts())
    for note in cli.pool_config_warnings(cfg):
        _say(f"WARNING: {note}")
    holder = lock_holder(lock_path(cfg))
    if holder is not None:
        raise SystemExit(f"pool {cfg['shard_base']}: another instance holds {lock_path(cfg)} "
                         f"(pid {holder.get('pid')}, {holder.get('role')}); only one pool loop may run. "
                         f"Watch it with pool-status, stop it with: {stop_command(config_path)}")
    proc = pool_process(cfg)
    if proc is not None and proc.get("alive") and proc.get("pid") != os.getpid():
        raise SystemExit(f"pool {cfg['shard_base']} already runs (pid {proc.get('pid')} on {proc.get('host')}); "
                         f"stop it with: {stop_command(config_path)}")
    lock = acquire_lock(lock_path(cfg), run=cli.run_id(cfg), role="pool")
    if lock is None:
        raise SystemExit(f"pool {cfg['shard_base']}: another instance took {lock_path(cfg)}; only one pool loop "
                         "may run")
    try:
        return _run_pool_locked(cfg, config_path, pc, interval=interval, pool=pool, signals=signals)
    finally:
        lock.close()


def _run_pool_locked(cfg: dict[str, Any], config_path: str, pc, *, interval: float, pool: IslandPool | None,
                     signals: _Signals | None) -> int:
    if cfg.get("route", "ssh") == "ssh":
        try:
            check_sampler_spec(cfg, pc)
        except ValueError as error:
            raise SystemExit(f"pool: CONFIGURATION ERROR, nothing ordered: {error}") from None
    state = cli.pool_state_path(cfg)
    if pool is None:
        pool = IslandPool.restore(state, pc, cli.pool_ops(cfg), log=cli._log, events=event_log(cfg))
    if pool.scale_path is None:
        pool.scale_path = scale_path(cfg)
        pool.scale_run = cli.run_id(cfg)
    pool.apply_scale()
    if pool.stopped:
        _say(f"pool state {state} says stopped ({pool.stopped}); move it away to start a new pool")
        left = [m.__dict__ for m in pool.live()]
        if left:
            _print_banner(billing_banner(config_path, left, why="stopped pool with members left",
                                         national=is_national_compute(cfg), kind=provider_kind(cfg)))
            return 1
        return 0
    signals = signals or _Signals()
    _write_pid(cfg, config_path)
    _remove(stop_path(cfg))
    _say(f"WARNING: {lease_warning(cfg, config_path)}")
    _say("the control loop must stay up for the whole run (an always-on host, tmux or systemd); "
         "SIGTERM / Ctrl-C drains every node before it exits")
    signals.install()
    clean = False
    try:
        while not pool.stopped:
            try:
                pool.tick()
            except Exception as error:  # noqa: BLE001 - one bad tick never ends a pool that holds billed nodes
                pool.events("tick_failed", error=repr(error)[:300])
            if retry_path(cfg).exists():
                _retry_in_loop(pool, config_path)
            pool.save(state)
            if signals.name is None and not stop_path(cfg).exists():
                pool.wait(signals.wake, interval)  # a pause / resume / scale ends it at once
            if signals.name is not None or stop_path(cfg).exists():
                why = f"signal {signals.name}" if signals.name else "pool-stop requested"
                _say(f"pool: {why}: draining every node")
                _drain_all(pool, why)
                pool.save(state)
                break
        clean = True
    finally:
        signals.restore()
        try:
            pool.save(state)
        except Exception as error:  # noqa: BLE001 - the banner below matters more
            _say(f"pool: could not save {state}: {error!r}")
        _remove(pid_path(cfg))
        _remove(stop_path(cfg))
        if pool.halted:
            _say(f"pool: CONFIGURATION ERROR left unresolved: {pool.halted}")
        left = [m.__dict__ for m in pool.live()]
        if left:
            _print_banner(billing_banner(config_path, left,
                                         why="the pool loop exited" + ("" if clean else " on an error"),
                                         national=is_national_compute(cfg), kind=provider_kind(cfg)))
    return 0 if not pool.live() else 1


def _retry_in_loop(pool: IslandPool, config_path: str) -> None:
    """A pool-retry request: re-read the fixed config, check it, restart in place."""
    _remove(retry_path(pool_cfg_of(config_path)))
    try:
        cfg = json.loads(Path(config_path).read_text())
        cli.pool_config(cfg, allowed=None)
        if cfg.get("route", "ssh") == "ssh":
            check_sampler_spec(cfg)
    except (OSError, ValueError, TypeError) as error:
        pool.events("retry_refused", error=str(error)[:300])
        return
    if hasattr(pool.ops, "cfg"):
        pool.ops.cfg = cfg
    try:
        done = pool.retry()
    except Exception as error:  # noqa: BLE001 - the pool stays halted; the next retry tries again
        pool.events("retry_failed", error=repr(error)[:300])
        return
    _say(f"pool: retried in place on {done}")


def pool_cfg_of(config_path: str) -> dict[str, Any]:
    return json.loads(Path(config_path).read_text())


def retry_pool(cfg: dict[str, Any], config_path: str) -> int:
    """``pool-retry --config F``: after a configuration error, restart the sampler on
    the same node(s) with the fixed config (checked here first)."""
    check_state_file(cfg)
    try:
        cli.pool_config(cfg, allowed=None)
        if cfg.get("route", "ssh") == "ssh":
            check_sampler_spec(cfg)
    except (ValueError, TypeError) as error:
        print(f"pool-retry: the config is still wrong, nothing retried: {error}")
        return 2
    state = cli.pool_state_path(cfg)
    doc = json.loads(state.read_text()) if state.exists() else {}
    kept = [m for m in billed_members(doc) if m.get("config_error")]
    if not kept:
        print("pool-retry: no member is halted by a configuration error")
        return 1
    retry_path(cfg).write_text(json.dumps({"requested": time.time()}))
    proc = pool_process(cfg)
    names = ", ".join(f"{m['ticket']} (shard {m['shard']})" for m in kept)
    if proc is not None and proc.get("alive"):
        print(f"pool-retry: the running pool restarts the sampler in place on {names} at its next tick")
    else:
        print(f"pool-retry: no pool loop is alive; start it again (python -m yeto.launch.nc pool --config "
              f"{config_path}) and it restarts the sampler in place on {names}")
    return 0


def downgrade_state_file(cfg: dict[str, Any], config_path: str, *, drop_owed_fences: bool = False) -> int:
    """``pool-state-downgrade --config F``: before going back to a yeto that reads only
    the older pool state format, write the state file in that format (the file as it was
    is kept next to it as ``<name>.format<N>.bak``). Refused while a pool loop is alive
    and while a member is in a state the older code does not know."""
    from ..pool import island_pool  # noqa: PLC0415

    state = cli.pool_state_path(cfg)
    if not state.exists():
        print(f"pool-state-downgrade: no state file {state}: nothing to do")
        return 0
    lock = acquire_lock(lock_path(cfg), run=cli.run_id(cfg), role="pool-state-downgrade")
    if lock is None:
        print("pool-state-downgrade: refused, nothing changed: a pool loop is alive and rewrites the file "
              f"every tick; stop it first (kill the loop, or pool-stop --config {config_path})")
        return 2
    try:
        text = state.read_text()
        doc = json.loads(text)
        found = int(doc.get("format", 1))
        if found <= 1:
            print(f"pool-state-downgrade: {state} is format 1 already: nothing to do")
            return 0
        island_pool.check_state_format(doc, state)
        new, notes = island_pool.downgrade_state(doc, drop_owed_fences=drop_owed_fences)
        if new is None:
            print("pool-state-downgrade: refused, nothing changed:")
            for note in notes:
                print(f"  - {note}")
            return 2
        backup = state.with_name(f"{state.name}.format{found}.bak")
        backup.write_text(text)
        tmp = state.with_suffix(state.suffix + ".tmp")
        tmp.write_text(json.dumps(new, indent=1))
        tmp.replace(state)
    finally:
        lock.close()
    for note in notes:
        print(f"pool-state-downgrade: WARNING: {note}")
    print(f"pool-state-downgrade: {state} written as format 1 ({len(new['members'])} member(s)); the format "
          f"{found} file is kept as {backup}")
    return 0


# -- scale ---------------------------------------------------------------------

def _counted(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """State-file members that count against the target (IslandPool.counted)."""
    return [m for m in billed_members(doc) if m.get("state") != "draining" and m.get("retiring_since") is None
            and not (m.get("replaced_by") and m.get("state") == "ready")]


def current_target(cfg: dict[str, Any], doc: dict[str, Any]) -> int:
    scale = read_scale(cfg)
    if scale is not None:
        return int(scale["target"])
    return int(doc.get("target", cfg["target"]))


def scale_refusal(cfg: dict[str, Any], doc: dict[str, Any], target: int, retire: list[str], *,
                  force_static: bool = False) -> str | None:
    """Why ``pool-scale --target`` is refused before anything is ordered (None: allowed).
    ``doc`` is the pool's state file. Checked: the task source (a store dispenser, or
    static data slots only with ``force_static``), accounts (one node each, inside
    ``YETO_NC_ACCOUNTS``), data_slots, max_nodes, max_orders, a tripped breaker,
    max_run_spend_ucr (one node hour per new node), shard_limit (the trainer's buffer
    shards / the tunnel ports) and the --retire tickets."""
    pc = cli.pool_config(cfg, allowed=None)
    ecfg = pc.elastic
    if target < 0:
        return f"target {target} < 0"
    if is_national_compute(cfg):
        allowed = cli.allowed_accounts()
        usable = [a for a in pc.accounts if a in allowed]
        if target > len(usable):
            return (f"target {target} > {len(usable)} usable accounts {usable} (a National Compute account "
                    f"holds one node; the pool's accounts {pc.accounts} inside YETO_NC_ACCOUNTS "
                    f"{sorted(allowed)}; add accounts to the pool config and restart the pool)")
    elif target > len(pc.accounts):
        w = wording(cfg)
        return (f"target {target} > {len(pc.accounts)} slots (one {w['unit']} per slot: the pool config's "
                f"'slots'; add slots and restart the pool)")
    before = current_target(cfg, doc)
    if target != before and not cli.pool_has_dispenser(cfg):
        if pc.data_slots is None and target > before:  # removing one repeats nothing: allowed
            return ("the samplers have no dispenser (prompts.dispense.backend store, or the syncer's "
                    "dispenser on data_path syncer): every sampler reads every prompt, so another node "
                    "repeats every task; set sampler.dispense in the pool config (an elastic run needs it)")
        if pc.data_slots is not None and not force_static:
            return (f"static data_slots {pc.data_slots}: a removed node's slot pauses (its prompts wait "
                    f"until a scale up continues the slot) and no scale up goes past {pc.data_slots} nodes; "
                    f"use a store dispenser (sampler.dispense) for an elastic run, or pass --force-static")
    if pc.data_slots is not None and target > int(pc.data_slots):
        return f"target {target} > data_slots {pc.data_slots} (each member holds one slot)"
    if ecfg.max_nodes is not None and target > ecfg.max_nodes:
        return f"target {target} > elastic.max_nodes {ecfg.max_nodes}"
    counted = _counted(doc)
    new = max(0, target - len(counted))
    if new:
        if (doc.get("breaker") or {}).get("tripped"):
            return "the pool's order breaker is tripped (provisioning failed repeatedly): no new node is ordered"
        budget = doc.get("budget") or {}
        orders = int(budget.get("orders", 0))
        if ecfg.max_orders is not None and orders + new > ecfg.max_orders:
            return f"{orders} orders placed + {new} new > elastic.max_orders {ecfg.max_orders}"
        spent = float(budget.get("spent_ucr", 0.0))
        hourly = pc.hourly()  # ucr-equivalents (a USD provider price converted)
        if ecfg.max_run_spend_ucr is not None and spent + new * hourly > ecfg.max_run_spend_ucr:
            return (f"spend {spent:.0f} + {new} new node hour(s) x {hourly:.0f} > elastic.max_run_spend_ucr "
                    f"{ecfg.max_run_spend_ucr}")
        if ecfg.max_run_spend_usd is not None:
            from ..pool.money import to_usd  # noqa: PLC0415

            usd, hour = to_usd(spent, "ucr"), to_usd(hourly, "ucr")
            if usd + new * hour > float(ecfg.max_run_spend_usd):
                return (f"spend ${usd:.2f} + {new} new node hour(s) x ${hour:.2f} > elastic.max_run_spend_usd "
                        f"${float(ecfg.max_run_spend_usd):.2f}")
        next_shard = int(doc.get("next_shard", 0))
        if pc.shard_limit is not None and next_shard + new > pc.shard_limit:
            return (f"{new} new node(s) need shards {pc.shard_base}-{next_shard}..{pc.shard_base}-"
                    f"{next_shard + new - 1}, past shard_limit {pc.shard_limit} (the trainer's "
                    f"--rl-buffer-shards / the tunnel port range; shard indexes are not reused)")
    if retire:
        tickets = {m.get("ticket") for m in counted}
        unknown = [t for t in retire if t not in tickets]
        if unknown:
            return f"--retire {unknown}: not a live member of this pool (live: {sorted(tickets)})"
        if len(retire) > len(counted) - target:
            return (f"--retire {len(retire)} ticket(s) but the target {target} only drops "
                    f"{max(0, len(counted) - target)} of {len(counted)} member(s)")
    return None


def scale_pool(cfg: dict[str, Any], config_path: str, target: int | None = None, *,
               retire: list[str] | None = None, add: int = 0, remove: list[str] | None = None,
               mode: str | None = None, force_static: bool = False, dry_run: bool = False,
               source: str = "manual", extra: dict[str, Any] | None = None) -> int:
    """``pool-scale --config F (--target N | --add K | --remove T) [--retire T] [--drain|--fence]``:
    record the wanted target in the state dir. ``--add`` / ``--remove`` are relative to the
    target recorded when this command holds the scale lock (two concurrent ``--add 1``
    make +2); ``--remove T`` retires ticket T and lowers the target by one. A running loop
    applies it at its next tick (it keeps the pool lock: this command never takes it); a
    loop started later reads it at start. ``mode``: the scale-down mode of this request
    ("drain": the victims finish their trials in flight first; "fence": at once; None:
    the pool config's ``elastic.scale_down``). ``source``: "manual" (a person, the
    default) or "auto" (an automatic scaler); an automatic request is refused for
    ``elastic.manual_hold_s`` after a manual one (manual operations take priority).
    The control file keeps the manual entries (paused tickets, released tickets) and
    the tickets of an earlier request the loop has not applied yet."""
    check_state_file(cfg)
    retire = list(retire or []) + list(remove or [])
    if mode is not None and mode not in island_status.SCALE_DOWN_MODES:
        print(f"pool-scale: refused: mode {mode!r} is not one of {list(island_status.SCALE_DOWN_MODES)}")
        return 2
    state = cli.pool_state_path(cfg)
    path = scale_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = open(scale_lock_path(cfg), "a+")  # noqa: SIM115 - held for the check and the read-modify-write
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            doc = json.loads(state.read_text()) if state.exists() else {}
        except ValueError:
            doc = {}
        if doc.get("stopped"):
            print(f"pool-scale: refused: the pool is stopped ({doc['stopped']})")
            return 2
        old = read_scale(cfg) or {}
        before = current_target(cfg, doc)
        wanted = target if target is not None else before + int(add) - len(remove or [])
        hold = cli.pool_config(cfg, allowed=None).elastic.manual_hold_s
        last_manual = old.get("manual_at")
        if source != "manual" and last_manual is not None and time.time() - float(last_manual) < hold:
            print(f"pool-scale: refused, nothing changed: a manual change {time.time() - float(last_manual):.0f} s "
                  f"ago takes priority over automatic scaling for elastic.manual_hold_s {hold:.0f} s")
            return 2
        if int(old.get("seq", 0)) > int(doc.get("scale_seq", 0)):  # not applied yet: keep its tickets
            live = {m.get("ticket") for m in _counted(doc)}
            retire += [t for t in old.get("retire") or () if t not in retire and t in live]
        why = scale_refusal(cfg, doc, wanted, retire, force_static=force_static)
        if why is not None:
            print(f"pool-scale: refused, nothing changed: {why}")
            return 2
        change = "scale up" if wanted > before else "scale down" if wanted < before else "no change"
        if dry_run:
            print(f"pool-scale: dry run: target {before} -> {wanted} ({change}) would be accepted; nothing changed")
            return 0
        seq = int(old.get("seq", 0)) + 1
        new = {**{k: v for k, v in old.items() if k in MANUAL_KEYS},
               "run": cli.run_id(cfg), "target": wanted, "seq": seq, "retire": retire, "mode": mode,
               "requested": time.time(), "previous": before, "source": source, **(extra or {})}
        if source == "manual":
            new["manual_at"] = new["requested"]
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(new, indent=1))
        tmp.replace(path)
    finally:
        lock.close()
    actual = len(billed_members(doc))
    proc = pool_process(cfg)
    alive = proc is not None and proc.get("alive")
    pc = cli.pool_config(cfg, allowed=None)
    how = mode or pc.elastic.scale_down
    print(f"pool-scale: target {before} -> {wanted} ({change}); {actual} node(s) billed now"
          + (f"; retire {', '.join(retire)}" if retire else ""))
    if wanted < before:
        if pc.data_slots is not None:
            print("pool-scale: static data slots: the removed node's slot pauses until a scale up continues it")
        elif how == "drain":
            print(f"pool-scale: scale down by drain: the removed node(s) claim no new task, finish their "
                  f"trials in flight (at most {pc.elastic.scale_drain_timeout_s:.0f} s), then are fenced and "
                  f"released; they bill until then")
        else:
            print("pool-scale: scale down by fence: the removed node(s) are fenced now (their tasks go back to "
                  "the other nodes, their trials in flight are lost) and released")
    if alive:
        print(f"pool-scale: the running pool (pid {proc.get('pid')}) applies it at its next tick")
    else:
        print(f"pool-scale: no pool loop is alive; it applies when the pool starts (python -m yeto.launch.nc "
              f"pool --config {config_path})")
    return 0


# -- manual operations ---------------------------------------------------------

MANUAL_KEYS = ("paused", "hold_tasks", "manual_released", "manual_at")


def pause_pool(cfg: dict[str, Any], config_path: str, tickets: list[str], *, resume: bool = False,
               hold_tasks: bool = False) -> int:
    """``pool-pause --ticket T [--hold-tasks]`` / ``pool-resume --ticket T``: recorded in the
    control file; the running loop applies it within about a second (and after a restart). A paused
    member gets no automatic action and counts toward the target."""
    check_state_file(cfg)
    name = "pool-resume" if resume else "pool-pause"
    state = cli.pool_state_path(cfg)
    path = scale_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(scale_lock_path(cfg), "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            doc = json.loads(state.read_text()) if state.exists() else {}
        except ValueError:
            doc = {}
        live = {m.get("ticket") for m in billed_members(doc)}
        unknown = [t for t in tickets if t not in live]
        if unknown and not resume:
            print(f"{name}: refused, nothing changed: {unknown} not a live member of this pool "
                  f"(live: {sorted(t for t in live if t)})")
            return 2
        old = read_scale(cfg) or {}
        if resume:  # a mistyped ticket must not read as "resumed" while the paused node stays paused
            # paused in the control file, or held paused by the pool (its state outlives a lost control file)
            paused_now = list(old.get("paused") or ()) + sorted(
                m.get("ticket") for m in billed_members(doc) if m.get("paused") and m.get("ticket"))
            not_paused = [t for t in tickets if t not in paused_now]
            if not_paused:
                print(f"{name}: refused, nothing changed: {not_paused} not paused "
                      f"(paused now: {sorted(set(paused_now)) or 'none'})")
                return 2
        paused = [t for t in old.get("paused") or () if t in live or resume]
        hold = [t for t in old.get("hold_tasks") or () if t in paused]
        if resume:
            paused = [t for t in paused if t not in tickets]
            hold = [t for t in hold if t not in tickets]
        else:
            paused += [t for t in tickets if t not in paused]
            if hold_tasks:
                hold += [t for t in tickets if t not in hold]
        target = current_target(cfg, doc)
        new = {**{k: v for k, v in old.items() if k in MANUAL_KEYS}, "run": cli.run_id(cfg), "target": target,
               "seq": int(old.get("seq", 0)) + 1, "retire": [], "mode": None, "requested": time.time(),
               "previous": target, "source": "manual", "manual_at": time.time(), "paused": paused,
               "hold_tasks": hold}
        if int(old.get("seq", 0)) > int(doc.get("scale_seq", 0)):  # an unapplied request's tickets
            new["retire"] = [t for t in old.get("retire") or () if t in live]
            new["mode"] = old.get("mode")
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(new, indent=1))
        tmp.replace(path)
    proc = pool_process(cfg)
    verb = "resumed" if resume else "paused" + (" (no new tasks)" if hold_tasks else "")
    print(f"{name}: {', '.join(tickets)} {verb}; paused now: {paused or 'none'}")
    if not resume:
        print(f"{name}: the pool takes no automatic action on it (no restart, retire, replacement or lease-end "
              f"drain) and does not replace it; it keeps billing. Resume: python -m yeto.launch.nc pool-resume "
              f"--config {config_path} --ticket <ticket>")
    if not (proc is not None and proc.get("alive")):
        print(f"{name}: no pool loop is alive; it applies when the pool starts")
    return 0


def pool_of_ticket(ticket: str) -> tuple[dict[str, Any], str, dict[str, Any]] | None:
    """The pool (config, config path, member) whose state holds ``ticket`` as a live
    member, from the pid files of the state dir (they name the pool config). Raises
    ``PoolStateFormatError`` on a pool state of a newer format."""
    for pid in sorted(tunnel.state_dir().glob("pool-*.pid")):
        try:
            config_path = json.loads(pid.read_text())["config"]
            cfg = json.loads(Path(config_path).read_text())
            doc = json.loads(cli.pool_state_path(cfg).read_text())
        except (OSError, ValueError, KeyError, TypeError):
            continue
        check_state_file(cfg)  # a newer format: whether the ticket is a member cannot be told
        for m in billed_members(doc):
            if m.get("ticket") == ticket:
                return cfg, config_path, m
    return None


def record_manual_release(cfg: dict[str, Any], ticket: str, *, lower: bool = True) -> int:
    """Write a manual release of pool member ``ticket`` to the pool's control file
    (``manual_released[ticket]``, and target - 1 when ``lower``), for a delete sent while
    no loop runs: a later start does not lower the target again, and ``run up`` does not
    take the member for a live one of an unfinished stop. ``lower`` False: a leftover of
    a stopped pool; the next start fills the pool's target as before. Returns the target."""
    with open(scale_lock_path(cfg), "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            doc = json.loads(cli.pool_state_path(cfg).read_text())
        except (OSError, ValueError):
            doc = {}
        old = read_scale(cfg) or {}
        released = dict(old.get("manual_released") or {})
        # once per ticket: a rerun (`delete-ticket` again after a crash, `nc down` twice)
        # finds it recorded and does not lower the target a second time
        again = ticket in released
        target = max(0, current_target(cfg, doc) - (1 if lower and not again else 0))
        released[ticket] = released.get(ticket) or {"target_lowered": True, "at": time.time()}
        new = {**{k: v for k, v in old.items() if k in MANUAL_KEYS}, "run": cli.run_id(cfg), "target": target,
               "seq": int(old.get("seq", 0)) + 1, "retire": [], "mode": None, "requested": time.time(),
               "previous": current_target(cfg, doc), "source": "manual", "manual_at": time.time(),
               "manual_released": released}
        path = scale_path(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(new, indent=1))
        tmp.replace(path)
    return target


def down_member(ticket: str, *, mode: str | None = "fence") -> int | None:
    """``nc down --ticket T`` for a member of a pool: a manual removal through the pool's
    control file (target - 1, retire T), so the pool does not order a node in its place.
    None: ``T`` is no pool member (the caller deletes it itself)."""
    found = pool_of_ticket(ticket)
    if found is None:
        return None
    cfg, config_path, member = found
    proc = pool_process(cfg)
    if proc is not None and proc.get("alive"):
        print(f"down: ticket {ticket} is shard {member.get('shard')} of the running pool {config_path}: asking the "
              f"pool to remove it (pool-scale --remove {ticket}) instead of deleting it behind the pool's back, "
              f"which it would replace; the pool releases it and confirms the ticket ended")
        return scale_pool(cfg, config_path, remove=[ticket], mode=mode, force_static=True)
    # no loop: delete here, and record it so a later start keeps the lowered target
    target = record_manual_release(cfg, ticket)
    print(f"down: ticket {ticket} belongs to pool {config_path}, whose loop is not running: deleting it here; "
          f"the pool target is lowered to {target}, so a restarted pool does not order a node in its place")
    return None


# -- stop ----------------------------------------------------------------------

def stop_pool(cfg: dict[str, Any], config_path: str, *, wait_s: float = 900.0, poll_s: float = 5.0,
              sleep=None, ops=None) -> int:
    """Release every node the pool created (``pool-stop --config F``)."""
    check_state_file(cfg)
    sleep = sleep or time.sleep
    cli.pool_config(cfg, allowed=None)  # releasing needs no ordering rights
    state = cli.pool_state_path(cfg)
    proc = pool_process(cfg)
    if proc is not None and proc.get("alive"):
        stop_path(cfg).parent.mkdir(parents=True, exist_ok=True)
        stop_path(cfg).write_text(json.dumps({"requested": time.time()}))
        _say(f"pool-stop: asked the running pool (pid {proc.get('pid')} on {proc.get('host')}) to drain; "
             f"waiting up to {wait_s:.0f} s")
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            proc = pool_process(cfg)
            if proc is None or not proc.get("alive"):
                break
            sleep(poll_s)
        else:
            doc = json.loads(state.read_text()) if state.exists() else {}
            _print_banner(billing_banner(config_path, billed_members(doc),
                                         why=f"the running pool did not stop in {wait_s:.0f} s",
                                         national=is_national_compute(cfg), kind=provider_kind(cfg)))
            return 1
        if proc is None:  # the loop exited and removed its files: read what it left
            doc = json.loads(state.read_text()) if state.exists() else {}
            left = billed_members(doc)
            if not left:
                _remove(scale_path(cfg))
                _say("pool-stop: the pool drained every node")
                return 0
    # no live loop (crashed, killed, host gone) or it left members: release them from here,
    # holding the pool lock (a loop started meanwhile, or a cleanup, waits for us)
    lock = acquire_lock(lock_path(cfg), run=cli.run_id(cfg), role="pool-stop")
    if lock is None:
        _say(f"pool-stop: a pool loop holds {lock_path(cfg)} again; rerun pool-stop to ask it to drain")
        return 1
    try:
        return _release_from_state(cfg, config_path, proc, state, sleep=sleep, ops=ops)
    finally:
        lock.close()


def _release_from_state(cfg, config_path, proc, state, *, sleep, ops) -> int:
    if proc is not None and not proc.get("alive"):
        _say(f"pool-stop: no pool process alive (stale pid {proc.get('pid')}); releasing from the state file")
    pool = IslandPool.restore(state, cli.pool_config(cfg, allowed=None), ops or cli.pool_ops(cfg), log=cli._log,
                              events=island_status.EventLog(path=cli.pool_events_path(cfg), run=cli.run_id(cfg),
                                                            log=cli._log))
    left = _drain_all(pool, "pool-stop", sleep=sleep)
    pool.save(state)
    _remove(pid_path(cfg))
    _remove(stop_path(cfg))
    if left:
        _print_banner(billing_banner(config_path, [m.__dict__ for m in left], why="a release failed",
                                     national=is_national_compute(cfg), kind=provider_kind(cfg)))
        return 1
    _remove(scale_path(cfg))
    _say("pool-stop: every node of the pool is released")
    return 0


# -- status --------------------------------------------------------------------

def check_state_file(cfg: dict[str, Any]) -> None:
    """Raise ``PoolStateFormatError`` when the pool's state file has a format newer than
    this code reads (every pool command reads the file; none may guess at it)."""
    from ..pool.island_pool import check_state_format  # noqa: PLC0415

    state = cli.pool_state_path(cfg)
    try:
        doc = json.loads(state.read_text()) if state.exists() else {}
    except (OSError, ValueError):
        return  # unreadable: each command reports that in its own way
    if isinstance(doc, dict):
        check_state_format(doc, state)


def account_attention(cfg: dict[str, Any], *, book: Any = None) -> list[dict[str, Any]]:
    """What an operator must look at on the pool's National Compute accounts, from the
    ticket ledger (read only, no call to National Compute): a halted account, a delete not
    confirmed yet, an order without a recorded outcome whose call is over (the account may
    hold a ticket no ledger row knows). [] for another provider."""
    if not is_national_compute(cfg):
        return []
    try:
        book = book if book is not None else cli.ledger()
        accounts = [int(a) for a in cfg.get("accounts") or ()]
    except Exception as error:  # noqa: BLE001 - reported, never fatal to a status
        return [{"kind": "ledger_unreadable", "error": repr(error)[:300]}]
    out: list[dict[str, Any]] = []
    for account in accounts:
        try:
            halt, delete = book.halted(account), book.pending_delete(account, sent_only=True)
            orders = book.pending_orders(account, in_flight=False)
        except Exception as error:  # noqa: BLE001
            out.append({"kind": "ledger_unreadable", "account": account, "error": repr(error)[:300]})
            continue
        if halt is not None:
            out.append({"kind": "halted", "account": account, "halt_kind": halt.get("kind"),
                        "reason": halt.get("halt"),
                        "since": _until(halt.get("halted_at")), "expected_ticket": halt.get("expected_ticket"),
                        "released_ticket": halt.get("released_ticket"),
                        "action": f"check the account, then: python -m yeto.launch.nc clear-halt --account "
                                  f"{account} --yes (the pool orders and releases nothing on it until then)"})
        if delete is not None:
            out.append({"kind": "delete_unconfirmed", "account": account, "ticket": delete.get("pending_delete"),
                        "sent": _until(delete.get("sent_at")), "attempts": delete.get("attempts"),
                        "action": "no other write on the account until that ticket reads ended"})
        for row in orders:
            out.append({"kind": "order_without_outcome", "account": account,
                        "since": _until(row.get("since") or row.get("created_at")),
                        "observed_ticket": row.get("observed_ticket"),
                        "action": "an order was sent and its outcome never recorded: the account may hold a "
                                  "billed ticket no ledger row knows; check it (python -m yeto.launch.nc status "
                                  f"--account {account})"})
    return out


def print_attention(notes: list[dict[str, Any]]) -> None:
    for note in notes:
        what = " ".join(f"{k}={v}" for k, v in note.items() if k not in ("kind", "action") and v is not None)
        print(f"ATTENTION: {str(note.get('kind')).upper()}: {what}" + (f" -> {note['action']}" if note.get("action")
                                                                       else ""), file=sys.stderr)


def pool_status(cfg: dict[str, Any], config_path: str, *, events: int = 20) -> int:
    check_state_file(cfg)
    state = cli.pool_state_path(cfg)
    doc = json.loads(state.read_text()) if state.exists() else {}
    path = cli.pool_events_path(cfg)
    lines = path.read_text().splitlines() if path.exists() else []
    proc = pool_process(cfg)
    left = billed_members(doc)
    now = time.time()
    label = cli.slot_name(cfg)  # "account" on National Compute, else "slot"
    billed = [{label: m.get("account"), "ticket": m.get("ticket"), "shard": m.get("shard"),
               "state": m.get("state"), "lease_until": _until(m.get("lease_until")),
               "past_lease_until": m.get("lease_until") is not None and now > float(m["lease_until"])}
              for m in left]
    scale = read_scale(cfg)
    target = int(scale["target"]) if scale else int(doc.get("target", cfg["target"]))
    for row, m in zip(billed, left):
        row["paused"] = bool(m.get("paused"))
        row["retiring"] = m.get("retiring_since") is not None
    manual = {"target_source": (scale or {}).get("source", "config") if scale else "config",
              "target_set_manually": bool(scale) and (scale or {}).get("source", "manual") != "auto",
              "manual_at": _until((scale or {}).get("manual_at")),
              "paused": list((scale or {}).get("paused") or []),
              "hold_tasks": list((scale or {}).get("hold_tasks") or []),
              "released_by_hand": sorted((scale or {}).get("manual_released") or {})}
    attention = account_attention(cfg)
    print(json.dumps({"process": proc, "target": target, "nodes": len(left),
                      "samplers": f"{len(left)}/{target}", "accounts_attention": attention,
                      "manual": manual, "scale": scale, "billed": billed,
                      "state": doc,
                      "events": [json.loads(x) for x in lines[-events:]]}, indent=1))
    print_attention(attention)
    if left and (proc is None or not proc.get("alive")):
        _print_banner(billing_banner(config_path, left, why="no pool process is alive to drain them",
                                     national=is_national_compute(cfg), kind=provider_kind(cfg)))
        return 1
    if left:
        print(f"note: {lease_warning(cfg, config_path)}", file=sys.stderr)
    return 0


# -- delete one ticket of this run -------------------------------------------------

PROTECTED_NC_ACCOUNTS = frozenset({2, 3})  # production accounts: never deleted on, whatever the config says
RELEASED_STATES = frozenset({"ended", "released"})


def protected_accounts(cfg: dict[str, Any]) -> frozenset[int]:
    """The accounts ``delete-ticket`` refuses: :data:`PROTECTED_NC_ACCOUNTS` plus the
    config's ``protected_accounts`` (it can only add to the built-in set)."""
    return PROTECTED_NC_ACCOUNTS | {int(a) for a in cfg.get("protected_accounts") or ()}


def _pool_state_doc(cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        doc = json.loads(cli.pool_state_path(cfg).read_text())
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def _ticket_evidence(cfg: dict[str, Any], ticket: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The ledger rows of this run and the pool state members that name ``ticket``."""
    run = cli.run_id(cfg)
    rows = [r for r in cli.ledger().load() if r.get("ticket") == ticket and r.get("run") == run]
    members = [m for m in _pool_state_doc(cfg).get("members", []) if m.get("ticket") == ticket]
    return rows, members


def delete_own_ticket(cfg: dict[str, Any], config_path: str, ticket: str, *, yes: bool = False) -> int:
    """``delete-ticket --config F --ticket T [--yes]``: release one ticket this run
    ordered. Refused (exit 2, nothing sent) unless ``T`` is in this run's ledger rows
    or pool state, on an account of the config, not a protected account, and no pool
    loop of the config is alive. Without ``--yes`` it only prints what it would delete.
    No pattern, list or ``--all``: one exact ticket id per call."""
    check_state_file(cfg)
    cli.pool_config(cfg, allowed=None)

    def refuse(why: str) -> int:
        print(f"delete-ticket: refused, nothing sent: {why}", file=sys.stderr)
        return 2

    if not is_national_compute(cfg):
        return refuse("the pool config is not a National Compute pool")
    if not ticket or any(c in ticket for c in "*?[],; \t") or ticket.lower() in ("all", "--all"):
        return refuse(f"ticket {ticket!r} is not one exact ticket id (no patterns, lists or 'all')")
    rows, members = _ticket_evidence(cfg, ticket)
    run = cli.run_id(cfg)
    if not rows and not members:
        return refuse(f"ticket {ticket} is in neither the ledger rows of run {run!r} nor the pool state "
                      f"{cli.pool_state_path(cfg)}: this run did not order it")
    accounts = {int(r["account"]) for r in rows} | {int(m["account"]) for m in members if "account" in m}
    if len(accounts) != 1:
        return refuse(f"ticket {ticket}: the records disagree on its account ({sorted(accounts)})")
    account = accounts.pop()
    pool_accounts = [int(a) for a in cfg.get("accounts") or ()]
    if account not in pool_accounts:
        return refuse(f"ticket {ticket} is on account {account}, not one of the pool's accounts {pool_accounts}")
    if account in protected_accounts(cfg):
        return refuse(f"account {account} is protected ({sorted(protected_accounts(cfg))}: production); "
                      f"release it in the National Compute console if you must")
    proc = pool_process(cfg)
    if proc is not None and proc.get("alive"):
        return refuse(f"the pool loop of {config_path} is running (pid {proc.get('pid')}): use "
                      f"pool-scale --remove {ticket} so the pool does not replace it")
    where = ", ".join(([f"ledger ({len(rows)} row)"] if rows else []) + (["pool state"] if members else []))
    print(f"delete-ticket: run {run}, account {account}, ticket {ticket} (recorded in: {where})")
    if not yes:
        print("delete-ticket: dry run, nothing sent; pass --yes to delete this one ticket")
        return 0
    events = event_log(cfg)
    try:
        result = cli.client(account).delete(ticket)
    except Exception as error:
        events("ticket_delete_failed", ticket=ticket, account=account, error=repr(error)[:300], source="manual")
        print(f"delete-ticket: ticket {ticket}: delete failed: {error}", file=sys.stderr)
        return 1
    state, reason = result.get("state"), result.get("end_reason")
    if state not in RELEASED_STATES and reason not in RELEASED_STATES:
        events("ticket_delete_unconfirmed", ticket=ticket, account=account, state=state, source="manual")
        print(f"delete-ticket: ticket {ticket}: the delete returned state={state} end_reason={reason}, "
              f"not ended/released; check the National Compute console", file=sys.stderr)
        return 1
    events("ticket_deleted", ticket=ticket, account=account, state=state, end_reason=reason,
           already=bool(result.get("already")), source="manual")
    if any(m.get("state") != "gone" for m in members):  # a pool member: the pool sees a manual release
        stopped = bool(_pool_state_doc(cfg).get("stopped"))  # a leftover of a stopped pool: target kept
        target = record_manual_release(cfg, ticket, lower=not stopped)
        print(f"delete-ticket: ticket {ticket} was a member of the pool state: recorded as released by hand "
              f"(pool target {target}" + ("; the pool is stopped, its next start fills the target)" if stopped
                                          else ", no node is ordered in its place)"))
    print(f"delete-ticket: ticket {ticket} on account {account}: state={state} end_reason={reason}")
    return 0
