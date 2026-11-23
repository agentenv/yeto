"""Owner metadata on AgentENV trial microVMs, and the stale-incarnation sweep.

A sampler process killed with SIGKILL (a crash, the stop grace running out) runs no
teardown, so the microVMs of its running trials stayed up until their TTL (three
hours before the TTL followed the trial wall clock). Every microVM yeto creates
therefore names its owner in the create's ``metadata`` (E2B-compatible, listed by
``GET /sandboxes``):

* ``yeto_run``: the run (``YETO_SANDBOX_RUN_LABEL``, else ``YETO_NC_RUN_PREFIX``);
* ``yeto_shard``: the sampler id (``YETO_SANDBOX_SHARD``, set by the sampler);
* ``yeto_inc``: the sampler process incarnation (``YETO_SAMPLER_INCARNATION``).

:func:`sweep_stale` deletes the microVMs whose run and shard equal this process's
and whose incarnation is set and differs from its own; the sampler calls it once its
incarnation lease is taken (no earlier incarnation of the shard still writes). The
list may be filtered by the gateway (``?metadata=``), but every row is checked again
here: a microVM without the three keys, of another run or shard, or of the current
incarnation is never deleted. An AgentENV gateway is often a shared machine; nothing
else on it is touched.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

RUN_LABEL_ENV = "YETO_SANDBOX_RUN_LABEL"  # yeto.sandbox.trial_labels.RUN_LABEL_ENV
RUN_PREFIX_ENV = "YETO_NC_RUN_PREFIX"     # the pool's run prefix: the same on every incarnation
SHARD_ENV = "YETO_SANDBOX_SHARD"          # yeto.sandbox.trial_labels.SHARD_ENV
INCARNATION_ENV = "YETO_SAMPLER_INCARNATION"

RUN_KEY, SHARD_KEY, INC_KEY = "yeto_run", "yeto_shard", "yeto_inc"


@dataclass(frozen=True)
class Owner:
    run: str
    shard: str
    inc: str

    def metadata(self) -> dict[str, str]:
        out = {RUN_KEY: self.run, SHARD_KEY: self.shard}
        if self.inc:
            out[INC_KEY] = self.inc
        return out

    def is_stale(self, metadata: Mapping[str, Any] | None) -> bool:
        """``metadata`` names this run and shard and another (set) incarnation."""
        if not isinstance(metadata, Mapping) or not self.run or not self.shard or not self.inc:
            return False
        inc = str(metadata.get(INC_KEY) or "")
        return (str(metadata.get(RUN_KEY) or "") == self.run and str(metadata.get(SHARD_KEY) or "") == self.shard
                and bool(inc) and inc != self.inc)


def owner_from_env(environ: Mapping[str, str] | None = None) -> Owner | None:
    """The owner of this process's microVMs; None without a run or a shard."""
    env = os.environ if environ is None else environ
    run = str(env.get(RUN_LABEL_ENV) or env.get(RUN_PREFIX_ENV) or "").strip().strip("/")
    shard = str(env.get(SHARD_ENV) or "").strip()
    if not run or not shard:
        return None
    return Owner(run=run, shard=shard, inc=str(env.get(INCARNATION_ENV) or "").strip())


def owner_metadata(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The owner keys a create carries (empty without an owner)."""
    owner = owner_from_env(environ)
    return owner.metadata() if owner is not None else {}


@dataclass(frozen=True)
class Listed:
    sandbox_id: str
    metadata: Mapping[str, Any]
    state: str = ""


def parse_listing(rows: Any) -> list[Listed]:
    """``GET /sandboxes`` rows (a list, or ``{"sandboxes": [...]}``) as :class:`Listed`;
    a row without an id or a metadata object is dropped (never a sweep candidate)."""
    if isinstance(rows, Mapping):
        rows = rows.get("sandboxes") or rows.get("items") or []
    out = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, Mapping):
            continue
        sid = row.get("sandboxID") or row.get("sandbox_id") or row.get("id")
        meta = row.get("metadata")
        if sid and isinstance(meta, Mapping):
            out.append(Listed(str(sid), dict(meta), str(row.get("state") or "")))
    return out


def stale(rows: Iterable[Listed], owner: Owner) -> list[Listed]:
    return [r for r in rows if owner.is_stale(r.metadata)]


def sweep_stale(client: Any, owner: Owner | None, *, emit: Callable[..., None] | None = None) -> list[str]:
    """Delete ``owner``'s shard's microVMs of other incarnations; returns their ids.
    Needs ``client.list_sandboxes(metadata)``; best effort: a failed list or delete is
    logged (an orphan then waits for its TTL), never raised."""
    if owner is None or not owner.inc:
        return []
    lister = getattr(client, "list_sandboxes", None)
    if lister is None:
        return []
    try:
        rows = stale(lister({RUN_KEY: owner.run, SHARD_KEY: owner.shard}), owner)
    except Exception as error:  # noqa: BLE001 - orphans only cost room until their TTL
        logger.warning("agentenv orphan sweep: listing failed (%s); orphans wait for their TTL",
                       type(error).__name__)
        return []
    deleted = []
    for row in rows:
        try:
            client.delete(row.sandbox_id)
            deleted.append(row.sandbox_id)
        except Exception as error:  # noqa: BLE001
            logger.warning("agentenv orphan sweep: delete of %s failed (%s)", row.sandbox_id, type(error).__name__)
    if rows:
        incs = sorted({str(r.metadata.get(INC_KEY)) for r in rows})
        logger.warning("agentenv: deleted %d of %d microVM(s) of run %s shard %s left by earlier incarnations %s",
                       len(deleted), len(rows), owner.run, owner.shard, incs)
        if emit is not None:
            emit("rl_sandbox_orphans_removed", provider="agentenv", shard=owner.shard, removed=len(deleted),
                 found=len(rows), incarnations=incs)
    return deleted
