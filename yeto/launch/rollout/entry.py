"""``yeto rollout scale`` / ``yeto rollout status``: one entry for the rollout
islands of a run, whatever provider runs them (National Compute, Daytona, sky,
Modal).

* ``yeto rollout scale (N | --add K | --remove TICKET) --config run.json [--pool NAME]
  [--provider KIND] [--retire TICKET]... [--drain | --fence] [--force-static] [--dry-run]
  [--source manual|auto]``: the flags and semantics of ``yeto run scale`` (one command:
  :meth:`yeto.rl.up.Run.scale`, the run's budget / breaker check of a scale up, then
  ``pool-scale`` on the pool). With one pool no selector is needed; with several,
  ``--pool`` (or a ``--provider`` that matches one pool, or the pool holding a
  ``--remove`` ticket) picks it. ``--pool-config pool.json`` names a pool config file
  directly (``pool-scale`` alone, no run-level check).
* ``yeto rollout status --config run.json``: every pool of the run (provider, data
  path, target, billed islands, loop alive) and the totals; ``--detail`` adds each
  pool's full ``pool-status``. Exit 1 when an island bills with no pool loop alive.

The older entries stay: ``yeto run scale --samplers N`` (the same code path) and
``python -m yeto.launch.nc pool-scale --config F --target N`` (what both call).
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from typing import Any

from . import pools as rollout_pools


@contextlib.contextmanager
def run_env(env: dict[str, str] | None):
    """The run config's environment (``YETO_NC_STATE_DIR``, ...) for an in-process
    read of the pools' state; restored afterwards."""
    old = dict(os.environ)
    os.environ.update({k: os.path.expanduser(str(v)) for k, v in (env or {}).items()})
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old)


def pool_scale_argv(path: str, args: Any) -> list[str]:
    """The ``pool-scale`` argv of a parsed scale request (the flags of :mod:`yeto.launch.nc.scale_args`)."""
    argv = ["pool-scale", "--config", path]
    if args.target is not None:
        argv += ["--target", str(int(args.target))]
    elif args.add:
        argv += ["--add", str(int(args.add))]
    for ticket in args.remove or ():
        argv += ["--remove", ticket]
    for ticket in args.retire or ():
        argv += ["--retire", ticket]
    if args.mode:
        argv.append(f"--{args.mode}")
    if args.force_static:
        argv.append("--force-static")
    if args.dry_run:
        argv.append("--dry-run")
    if args.source != "manual":
        argv += ["--source", args.source]
    return argv


def scale(args: Any) -> int:
    if args.pool_config:
        ref = rollout_pools.PoolRef("pool", os.path.expanduser(args.pool_config))
        if args.provider is not None and ref.kind() != args.provider:
            print(f"rollout scale: refused, nothing changed: {ref.config} runs on {ref.kind()}, "
                  f"not {args.provider}")
            return 2
        from yeto.launch.nc import cli as nc_cli  # noqa: PLC0415

        return nc_cli.main(pool_scale_argv(ref.config, args))
    from yeto.rl import up  # noqa: PLC0415

    run = up.Run(up.load_config(args.config))
    return run.scale(args.target, args.retire, add=args.add, remove=args.remove, mode=args.mode,
                     force_static=args.force_static, dry_run=args.dry_run, source=args.source, pool=args.pool,
                     provider=args.provider)


def pool_summary(ref: rollout_pools.PoolRef) -> dict[str, Any]:
    from yeto.launch.nc import cli, pool_run  # noqa: PLC0415

    out: dict[str, Any] = {"pool": ref.name, "config": ref.config, "provider": ref.kind(),
                           "data_path": ref.data_path}
    try:
        cfg = ref.load()
    except (OSError, ValueError) as error:
        return dict(out, error=f"pool config unreadable: {error}")
    state = cli.pool_state_path(cfg)
    try:
        doc = json.loads(state.read_text()) if state.exists() else {}
    except ValueError:
        doc = {}
    try:
        pool_run.check_state_file(cfg)
    except RuntimeError as error:  # PoolStateFormatError: written by a newer yeto, not read here
        return dict(out, error=str(error))
    scale_doc = pool_run.read_scale(cfg)
    target = int(scale_doc["target"]) if scale_doc else int(doc.get("target", cfg.get("target", 0)))
    billed = pool_run.billed_members(doc)
    proc = pool_run.pool_process(cfg)
    alive = bool(proc and proc.get("alive"))
    out.update(target=target, islands=len(billed), ready=sum(1 for m in billed if m.get("state") == "ready"),
               loop_alive=alive, stopped=doc.get("stopped"),
               slot_name=cli.slot_name(cfg),
               billed=[{cli.slot_name(cfg): m.get("account"), "ticket": m.get("ticket"), "shard": m.get("shard"), "state": m.get("state"),
                        "retiring": m.get("retiring_since") is not None} for m in billed])
    if billed and not alive:
        out["warning"] = (f"{len(billed)} island(s) bill with no pool loop alive: "
                          f"python -m yeto.launch.nc pool-stop --config {ref.config}")
    attention = pool_run.account_attention(cfg)
    if attention:  # halted accounts, unconfirmed deletes, orders without an outcome (the NC ledger)
        out["accounts_attention"] = attention
        kinds = sorted({str(a.get("kind")) for a in attention})
        note = f"{len(attention)} account matter(s) need an operator ({', '.join(kinds)}): see accounts_attention"
        out["warning"] = f"{out['warning']}; {note}" if out.get("warning") else note
    return out


class _NoOps:
    """The pool ops of a read-only view: restoring a pool for its status never calls a provider."""

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(f"read-only pool view: {name} is not available")


def view(refs: list[rollout_pools.PoolRef]) -> Any:
    """A :class:`~yeto.launch.pool.controller.RolloutController` over the saved state of
    ``refs`` (restored read-only: no lock, no pid, no scale file consumed, never started),
    for its ``status()``. Pools whose config does not load are left out."""
    from yeto.launch.nc import cli  # noqa: PLC0415
    from yeto.launch.pool.controller import RolloutController  # noqa: PLC0415
    from yeto.launch.pool.island_pool import IslandPool  # noqa: PLC0415

    pools = {}
    for ref in refs:
        try:
            cfg = ref.load()
            pc = cli.pool_config(cfg, allowed=None)
            pools[ref.name] = IslandPool.restore(cli.pool_state_path(cfg), pc, _NoOps(), log=lambda _m: None)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:  # incl. a newer state format
            cli._log(f"rollout status: pool {ref.name}: {error}")  # noqa: SLF001
    return RolloutController(pools, drain_on_exit=False, log=lambda _m: None)


def status(refs: list[rollout_pools.PoolRef], *, pool: str | None = None, provider: str | None = None,
           env: dict[str, str] | None = None, detail: bool = False, events: int = 20) -> int:
    chosen = rollout_pools.select(refs, pool=pool, provider=provider)
    if not chosen:
        print(json.dumps({"pools": [], "note": "no pool matches"}))
        return 2
    with run_env(env):
        rows = [pool_summary(r) for r in chosen]
        controlled = view(chosen).status()["pools"]
        for row in rows:
            if row["pool"] in controlled:
                row["controller"] = controlled[row["pool"]]
        print(json.dumps({"pools": rows, "total": {"target": sum(r.get("target", 0) for r in rows),
                                                   "islands": sum(r.get("islands", 0) for r in rows)}},
                         indent=1, default=str))
        rc = 1 if any(r.get("warning") or r.get("error") for r in rows) else 0
        if detail:
            from yeto.launch.nc import pool_run  # noqa: PLC0415

            for ref in chosen:
                print(f"--- pool {ref.name}")
                rc = max(rc, pool_run.pool_status(ref.load(), ref.config, events=events))
    return rc


def _status_refs(args) -> tuple[list[rollout_pools.PoolRef], dict[str, str]]:
    if args.pool_config:
        return [rollout_pools.PoolRef("pool", os.path.expanduser(args.pool_config))], {}
    from yeto.rl.up import STATE_ROOT, load_config  # noqa: PLC0415

    cfg = load_config(args.config)
    state_dir = STATE_ROOT / str(cfg["run"])
    state_dir.mkdir(parents=True, exist_ok=True)
    return rollout_pools.resolve(cfg, state_dir), dict(cfg.get("env") or {})


def add_parser(sub: Any) -> None:
    from yeto.launch.nc.scale_args import add_scale_args  # noqa: PLC0415 - no imports of its own

    p = sub.add_parser("rollout", help="rollout islands of a run on any provider: scale / status / plan")
    rsub = p.add_subparsers(dest="rollout_command", metavar="{scale,status,plan}")
    rsub.required = True

    def source(q) -> None:
        g = q.add_mutually_exclusive_group(required=True)
        g.add_argument("--config", help="the run config (yeto run up): its samplers / pool")
        g.add_argument("--pool-config", help="one pool config file (python -m yeto.launch.nc pool --config)")
        q.add_argument("--pool", help="the pool's name in the run config's samplers list")
        q.add_argument("--provider", choices=rollout_pools.PROVIDER_KINDS,
                       help="pick the pool by provider kind (and check it)")

    sc = rsub.add_parser("scale", help="change one pool's island count while the run trains (= yeto run scale)",
                         description="Record the wanted island count of one pool; its loop applies it at the "
                         "next tick. Refused before anything is ordered past the run budget, the provider's "
                         "slots, the pool's limits or the shard limit.")
    add_scale_args(sc, target_flag="N")
    source(sc)
    st = rsub.add_parser("status", help="every pool of the run: provider, target, billed islands (read-only)")
    source(st)
    st.add_argument("--detail", action="store_true", help="each pool's full pool-status too")
    st.add_argument("--events", type=int, default=20, help="with --detail: last events per pool")
    from .plan import add_args as _plan_args  # noqa: PLC0415

    pl = rsub.add_parser("plan", help="offline capacity estimate of a sampler fleet: cold start, release "
                         "fan-out, group balance, sandbox hosts, cost (no API call)")
    _plan_args(pl)


def main(args: Any) -> int:
    if args.rollout_command == "plan":
        from .plan import main as plan_main  # noqa: PLC0415

        return plan_main(args)
    try:
        if args.rollout_command == "scale":
            if args.pool_config and args.pool:
                print("rollout scale: --pool names a pool of a run config (--config)", file=sys.stderr)
                return 2
            return scale(args)
        refs, env = _status_refs(args)
    except (ValueError, OSError) as error:
        print(f"rollout {args.rollout_command}: {error}", file=sys.stderr)
        return 2
    return status(refs, pool=args.pool, provider=args.provider, env=env, detail=args.detail, events=args.events)
