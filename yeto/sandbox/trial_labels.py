"""Owner labels on Harbor trial containers and the stale-incarnation sweep.

A sampler process that is killed (exit 137, an in-place restart after a stall)
runs no teardown, so the trial containers it started on a remote sandbox host
(``provider: remote-docker``) keep running. The host's free room
(:class:`yeto.sandbox.gate.SandboxGate`) counts every running container, so
those orphans starve the next incarnation of the same shard.

Every trial container therefore carries its owner (``remote_docker_env``):

* ``ai.yeto.run``: the run (``YETO_SANDBOX_RUN_LABEL``);
* ``ai.yeto.shard``: the sampler id (``YETO_SANDBOX_SHARD``, set by the sampler);
* ``ai.yeto.inc``: the sampler process incarnation (``YETO_SAMPLER_INCARNATION``);
* ``ai.yeto.trial``: the trial's compose project (one per trial attempt).

:func:`sweep_stale` removes the containers (and their compose networks) whose
run and shard equal ours and whose incarnation is set and differs from ours.
A container of another run, another shard, without an incarnation label, or of
the current incarnation is never touched.

A replacement node continues a fenced shard in a new shard id (an NC pool:
``continues_shard``), so no later incarnation of the fenced shard ever starts.
:func:`sweep_shard` removes every container of such a predecessor shard; the
replacement sampler calls it once the predecessor left or was fenced.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

RUN_LABEL_ENV = "YETO_SANDBOX_RUN_LABEL"
SHARD_ENV = "YETO_SANDBOX_SHARD"
INCARNATION_ENV = "YETO_SAMPLER_INCARNATION"  # task_dispense.SAMPLER_INCARNATION_ENV

RUN_KEY = "ai.yeto.run"
SHARD_KEY = "ai.yeto.shard"
INC_KEY = "ai.yeto.inc"
TRIAL_KEY = "ai.yeto.trial"
PROJECT_KEY = "com.docker.compose.project"
OWNER_KEYS = (RUN_KEY, SHARD_KEY, INC_KEY, PROJECT_KEY)


@dataclass(frozen=True)
class Owner:
    run: str
    shard: str
    inc: str

    def is_stale(self, labels: Mapping[str, str]) -> bool:
        """``labels`` are a container of this run and shard, of another incarnation."""
        inc = str(labels.get(INC_KEY) or "")
        return (bool(self.run) and bool(self.shard) and labels.get(RUN_KEY) == self.run
                and labels.get(SHARD_KEY) == self.shard and bool(inc) and inc != self.inc)


def owner_from_env(environ: Mapping[str, str] | None = None) -> Owner | None:
    env = os.environ if environ is None else environ
    run = str(env.get(RUN_LABEL_ENV, "") or "").strip()
    shard = str(env.get(SHARD_ENV, "") or "").strip()
    if not run or not shard:
        return None
    return Owner(run=run, shard=shard, inc=str(env.get(INCARNATION_ENV, "") or "").strip())


def trial_labels(session: str | None = None, environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The labels of one trial container (empty without a run label)."""
    env = os.environ if environ is None else environ
    run = str(env.get(RUN_LABEL_ENV, "") or "").strip()
    if not run:
        return {}
    labels = {RUN_KEY: run}
    for key, name in ((SHARD_KEY, SHARD_ENV), (INC_KEY, INCARNATION_ENV)):
        value = str(env.get(name, "") or "").strip()
        if value:
            labels[key] = value
    if session:
        labels[TRIAL_KEY] = str(session)
    return labels


def _remove(docker: Any, rows: list[dict[str, Any]]) -> int:
    docker.remove([r["id"] for r in rows])
    projects = sorted({r["labels"].get(PROJECT_KEY) for r in rows} - {None, ""})
    for project in projects:
        docker.remove_networks(f"label={PROJECT_KEY}={project}")
    return len(rows)


def sweep_shard(docker: Any, run: str, shard: str) -> int:
    """Remove every trial container (any incarnation) of ``run``'s ``shard`` and
    their compose networks: the orphans of a predecessor shard that left or was
    fenced (module doc); the number of containers removed."""
    run, shard = str(run or "").strip(), str(shard or "").strip()
    if not run or not shard:
        return 0
    rows = docker.labelled_containers(OWNER_KEYS, f"label={RUN_KEY}={run}", f"label={SHARD_KEY}={shard}",
                                      all_=True)
    rows = [r for r in rows if r["labels"].get(RUN_KEY) == run and r["labels"].get(SHARD_KEY) == shard]
    if not rows:
        return 0
    n = _remove(docker, rows)
    logger.warning("sandbox: removed %d trial container(s) of run %s predecessor shard %s", n, run, shard)
    return n


def sweep_stale(docker: Any, owner: Owner | None) -> int:
    """Remove ``owner``'s shard's containers of other incarnations and their
    compose networks; the number of containers removed."""
    if owner is None or not owner.inc:
        return 0
    rows = docker.labelled_containers(OWNER_KEYS, f"label={RUN_KEY}={owner.run}",
                                      f"label={SHARD_KEY}={owner.shard}", all_=True)
    stale = [r for r in rows if owner.is_stale(r["labels"])]
    if not stale:
        return 0
    _remove(docker, stale)
    logger.warning("sandbox: removed %d trial container(s) of run %s shard %s from earlier incarnations %s",
                   len(stale), owner.run, owner.shard, sorted({r["labels"].get(INC_KEY) for r in stale}))
    return len(stale)
