"""``yeto cleanup --run <id>``: release exactly the resources the run's ledger lists.

Rules:

* Only ledger entries are touched, each by its exact provider id (Modal
  FunctionCall id, Modal app id, NC ticket + account, sky cluster name). No
  name pattern, no prefix match, no ``--all``.
* An ``intent`` entry (crash between "about to create" and "created") is
  resolved only when the provider lists exactly one app whose name EQUALS the
  recorded name and that was created after the intent; otherwise it is
  reported unresolved and left alone.
* A Modal app marked ``shared`` (reused by other runs) is not stopped; only this
  run's FunctionCalls in it are cancelled.
* NC leases go through the NC lease manager's own guarded release
  (``yeto.launch.nc.cli.down``: refuses tickets it did not create). The ledger
  kinds of island providers (``nc_lease``) are dispatched through the provider
  registry (``yeto.launch.pool.provider``): each registers its release.
* Order: FunctionCalls, then apps, then NC leases, then sky clusters. Each
  success is recorded as ``release`` in the ledger (a rerun skips it); each
  failure as ``release_failed`` (a rerun retries it).
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Protocol

from yeto.launch.pool import provider as providers

from .ledger import Resource, ResourceLedger

ORDER = {"modal_call": 0, "modal_app": 1, "sky_cluster": 3,
         **{kind: e.cleanup_order for kind, e in providers.ledger_kinds().items()}}


class CleanupOps(Protocol):
    def modal_cancel_call(self, call_id: str) -> None: ...
    def modal_stop_app(self, app_id: str) -> None: ...
    def modal_app_ids_named(self, name: str, created_after: float) -> list[str] | None: ...
    def nc_release(self, account: int, ticket: str) -> dict[str, Any]: ...
    def sky_down(self, cluster: str) -> bool: ...


class ProductionCleanupOps:
    """Modal CLI/SDK by exact id, the NC lease manager's guarded down, sky down."""

    def modal_cancel_call(self, call_id: str) -> None:
        import modal

        modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)

    def modal_stop_app(self, app_id: str) -> None:
        if not str(app_id).startswith("ap-"):
            raise ValueError(f"refusing to stop {app_id!r}: not a Modal app id")
        proc = subprocess.run([sys.executable, "-m", "modal", "app", "stop", "--yes", app_id],
                              capture_output=True, text=True, timeout=120)
        out = (proc.stdout + proc.stderr).strip()
        if proc.returncode != 0 and "already" not in out.lower() and "stopped" not in out.lower():
            raise RuntimeError(f"modal app stop {app_id}: {out or proc.returncode}")

    def modal_app_ids_named(self, name: str, created_after: float) -> list[str] | None:
        proc = subprocess.run([sys.executable, "-m", "modal", "app", "list", "--json"],
                              capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            return None
        from .billing import _ts

        out = []
        for app in json.loads(proc.stdout or "[]"):
            if app.get("description") != name:  # exact equality, never a pattern
                continue
            if str(app.get("state", "")).lower() == "stopped":
                continue
            try:
                created = _ts(app.get("created_at"))
            except ValueError:
                created = None
            if created is not None and created >= created_after - 5:
                out.append(str(app.get("app_id")))
        return out

    def nc_release(self, account: int, ticket: str) -> dict[str, Any]:
        from ..launch.nc import cli as nc_cli

        return nc_cli.down(int(account), str(ticket))

    def sky_down(self, cluster: str) -> bool:
        import sky

        sky.stream_and_get(sky.down(cluster))
        return True


@dataclass
class Action:
    key: str
    action: str
    ok: bool
    detail: str = ""


def _release_one(r: Resource, ops: CleanupOps, ledger: ResourceLedger, dry_run: bool,
                 resolved_apps: dict[str, str]) -> Action:
    if r.kind == "modal_call":
        if dry_run:
            return Action(r.key, f"would cancel FunctionCall {r.id}", True)
        ops.modal_cancel_call(r.id)
        ledger.released("modal_call", r.id, r.name, by="cleanup")
        return Action(r.key, f"cancelled FunctionCall {r.id}", True)
    if r.kind == "modal_app":
        if r.attrs.get("shared"):
            return Action(r.key, "shared app left running (its calls of this run were cancelled)", True)
        app_id = r.id
        if app_id is None:
            ids = ops.modal_app_ids_named(r.name, r.created_at)
            if ids is None:
                return Action(r.key, "intent unresolved: app list unavailable", False)
            if not ids:
                if not dry_run:
                    ledger.released("modal_app", None, r.name, by="cleanup", never_created=True)
                return Action(r.key, "intent never became an app: nothing to stop", True)
            if len(ids) > 1:
                return Action(r.key, f"intent unresolved: {len(ids)} live apps named exactly "
                              f"{r.name!r} created after the intent; left alone", False)
            app_id = ids[0]
            resolved_apps[r.key] = app_id
        if dry_run:
            return Action(r.key, f"would stop app {app_id}", True)
        ops.modal_stop_app(app_id)
        if r.id is None:
            ledger.created("modal_app", app_id, r.name, resolved_by="cleanup")
        ledger.released("modal_app", app_id, r.name, by="cleanup")
        return Action(r.key, f"stopped app {app_id}", True)
    entry = providers.ledger_kinds().get(r.kind)
    if entry is not None and entry.release_resource is not None:  # an island provider's machine
        ok, action, res = providers.resolve(entry.release_resource)(r, ops, dry_run)
        if ok and not dry_run:
            ledger.released(r.kind, r.id, r.name, by="cleanup", result=res)
        return Action(r.key, action, ok)
    if r.kind == "sky_cluster":
        if dry_run:
            return Action(r.key, f"would sky down {r.id}", True)
        ok = ops.sky_down(r.id)
        if ok:
            ledger.released("sky_cluster", r.id, r.name, by="cleanup")
        return Action(r.key, f"sky down {r.id}", bool(ok), "" if ok else "not confirmed")
    return Action(r.key, "unknown kind", False)


def cleanup(ledger: ResourceLedger, ops: CleanupOps, *, dry_run: bool = False,
            kinds: tuple[str, ...] | None = None,
            keys: tuple[str, ...] | None = None) -> list[Action]:
    actions = []
    resolved: dict[str, str] = {}
    pending = sorted(ledger.open_resources(), key=lambda r: (ORDER.get(r.kind, 9), r.created_at))
    for r in pending:
        if (kinds is not None and r.kind not in kinds) or (keys is not None and r.key not in keys):
            continue
        try:
            actions.append(_release_one(r, ops, ledger, dry_run, resolved))
        except Exception as exc:  # noqa: BLE001 - every other resource still gets its release
            if not dry_run:
                ledger.release_failed(r.kind, r.id, r.name, f"{type(exc).__name__}: {exc}")
            actions.append(Action(r.key, "release failed", False, f"{type(exc).__name__}: {exc}"))
    return actions


# Sampler pool machines (NC leases, Daytona sandboxes, Modal sampler calls, Sky sampler
# clusters) belong to their pool (python -m yeto.launch.nc pool / pool-stop / yeto run down):
# a trainer launch that fails or exits must never release them (2026-11-10: a failed trainer
# launch's backstop cleanup released a healthy, set-up sampler node). Only an explicit
# `yeto cleanup --run R --include-pools` (or pool-stop) releases them.
POOL_OWNED_KINDS = providers.pool_owned_ledger_kinds()  # every row of these kinds: {"nc_lease", "daytona_sandbox"}


def pool_owned(r: Any) -> bool:
    """A ledger row of a sampler pool machine. Modal calls and Sky clusters share their
    kind with the trainer's, so a pool row is the one its provider marked (``pool``),
    or, for rows written before the mark, a Modal call with a pool ``ticket`` / a Sky
    cluster with a pool ``slot``."""
    attrs = getattr(r, "attrs", None) or {}
    if r.kind in POOL_OWNED_KINDS or attrs.get("pool"):
        return True
    return (r.kind == "modal_call" and bool(attrs.get("ticket"))) or (r.kind == "sky_cluster" and "slot" in attrs)


def cleanup_run(run: str, ops: CleanupOps | None = None, *, dry_run: bool = False,
                log=print, include_pools: bool = False, include_nc: bool | None = None) -> bool:
    """Release the run's open resources except its sampler pools' machines
    (``include_pools``; ``include_nc`` is its earlier name); True when nothing it may
    release is left open."""
    include_pools = bool(include_pools or include_nc)
    ledger = ResourceLedger.for_run(run)
    if not ledger.path.exists():
        log(f"[cleanup] run {run}: no resource ledger at {ledger.path}; nothing to release")
        return True
    locks: list = []
    refused = False
    if include_pools and any(pool_owned(r) for r in ledger.open_resources()):
        held = _take_pool_locks(run, dry_run=dry_run, log=log)
        if held is None:  # a pool loop of this run runs: it owns its machines (pool-stop asks it)
            include_pools, refused = False, True
        else:
            locks = held
    try:
        return _cleanup_run(run, ledger, ops, dry_run=dry_run, log=log, include_pools=include_pools) \
            and not refused
    finally:
        for fh in locks:
            fh.close()


def _take_pool_locks(run: str, *, dry_run: bool, log) -> list | None:
    """The pool locks (``pool_run.lock_path``) recording ``run``, all taken; None when a
    live pool loop (or a pool-stop) holds one: pool machines are then not released here."""
    try:
        from yeto.launch.nc.pool_run import acquire_lock, run_pool_locks  # noqa: PLC0415
    except Exception as error:  # noqa: BLE001 - a tree without the NC pool: nothing to share
        log(f"[cleanup] NC pool lock unavailable ({type(error).__name__}); releasing without it")
        return []
    held = []
    for path in run_pool_locks(run):
        fh = acquire_lock(path, run=run, role="cleanup")
        if fh is None:
            for other in held:
                other.close()
            log(f"[cleanup] a pool loop of run {run} holds {path}: its machines are left to it "
                f"(stop it with python -m yeto.launch.nc pool-stop --config <pool.json>)")
            return None
        held.append(fh)
    return held


def _cleanup_run(run: str, ledger: ResourceLedger, ops, *, dry_run: bool, log, include_pools: bool) -> bool:
    skipped = [] if include_pools else [r for r in ledger.open_resources() if pool_owned(r)]
    for r in skipped:
        log(f"[cleanup] keep {r.key}: a sampler pool machine (release it with pool-stop or yeto run down, "
            f"or yeto cleanup --run {run} --include-pools)")
    keys = None if include_pools else tuple(r.key for r in ledger.open_resources() if not pool_owned(r))
    actions = cleanup(ledger, ops or ProductionCleanupOps(), dry_run=dry_run, keys=keys)
    if not actions:
        log(f"[cleanup] run {run}: every ledger resource is already released")
    for a in actions:
        log(f"[cleanup] {'ok  ' if a.ok else 'FAIL'} {a.key}: {a.action}"
            + (f" ({a.detail})" if a.detail else ""))
    remaining = [r for r in ledger.open_resources()
                 if not (r.kind == "modal_app" and r.attrs.get("shared"))
                 and (include_pools or not pool_owned(r))]
    return all(a.ok for a in actions) and (dry_run or not remaining)

