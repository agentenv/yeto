"""Eviction-tolerant ("elastic") policy pieces of the National Compute sampler pool.

:class:`~yeto.launch.pool.island_pool.IslandPool` keeps ``target`` samplers writing one buffer
across 4 h leases. This module holds what it needs to survive a lease that goes
away *early* or *late*, and nothing that talks to a cloud:

What National Compute does (protocol reference section 9.3 plus measurements):

* a ticket is ``held``/``queued`` (free) -> ``provisioning`` -> ``active``
  (billed per GPU hour) -> ``releasing`` -> ``ended`` with ``end_reason`` one of
  ``released`` (we deleted it), ``preempted``, ``lost`` (supply failure, also
  before a grant), ``balance`` (organisation balance ran out), ``admin``;
* ``preempt_at`` is set on an active ticket the platform will take back;
* **a lease is not ended at ``lease_until``**: a ticket read ``active`` with
  ``ended_at=null`` 19.5 minutes past its ``lease_until`` and ended only when it
  was deleted (``end_reason=released``). So the pool never relies on the platform
  to end a lease: it drains before ``lease_until`` and a patrol deletes any ticket
  of the run found ``active`` past it;
* a node can drop off the network for good while its ticket stays ``active``
  (billed): reachability is the sampler pool's job, not the platform's;
* there is no renewal (no extend call; ``max_lease_hours`` 4): "renewing" is a
  replacement order on another account placed ``replace_lead_s`` before the end.

Pieces:

* :func:`classify` - one member's ticket status + node probe -> a
  :class:`Verdict` (healthy, network blip, unreachable/lost, sampler crashed,
  evicted, expired, overdue, balance, ...);
* :class:`Breaker` - consecutive provisioning failures open it (exponential
  cooldown); too many trips stop ordering for good;
* :class:`Budget` - run spend cap, concurrent node cap, order cap;
* :class:`EventLog` - JSONL events (``nc_elastic_*``) plus an optional hook;
* :func:`fence_shard` - isolate the shard of a node taken for gone: a handoff
  request (a live sampler stops at its next round) and a claim of the shard's
  incarnation lease (``yeto.rl.engine.incarnation_lease``) by the pool, so a
  node that comes back cannot add groups the trainer would take (the trainer's
  ``IncarnationGate`` drops groups of a superseded incarnation appended after
  the claim).

The provider-neutral pieces (:class:`LeaseStatus`, :class:`Probe`,
:func:`classify`, :class:`ElasticConfig`, :class:`Breaker`, :class:`Budget`,
:class:`EventLog`, the verdicts) live in :mod:`yeto.launch.pool.status`; this
module keeps the node probe script and the shard fencing.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


def probe_script(container: str, tape_glob: str, join_pattern: str = "rl_sampler_join",
                 metrics_port: int | None = None, *, state_script: str | None = None,
                 logs_script: str | None = None) -> str:
    """Node shell: container state, exit code, JOIN seen, age of the newest tape,
    whether its last line is a trainer-wait heartbeat, and (``metrics_port``) the
    engine's generation-token counter and running / waiting requests from vLLM's
    ``/metrics`` on the node (the sampler runs ``--network host``). One line
    ``YETO_NC_PROBE status=... exit=... joined=0|1 age=... tw=0|1 tok=... run=... wq=...``;
    read-only. ``state_script`` / ``logs_script``: shell printing ``<status> <exit>`` and
    the sampler's log, for a sampler that is a process rather than a container."""
    metrics = "t=- r=- w=-; "
    if metrics_port is not None:
        metrics = (f"set -- $(curl -s -m 5 http://127.0.0.1:{int(metrics_port)}/metrics 2>/dev/null | awk "
                   "'/^vllm:generation_tokens_total/{t+=$NF;f=1} /^vllm:num_requests_running/{r+=$NF} "
                   "/^vllm:num_requests_waiting/{w+=$NF} END{if(f) printf \"%.0f %d %d\", t, r, w; "
                   "else print \"- - -\"}'); t=${1:--} r=${2:--} w=${3:--}; ")
    state = state_script or (f"sudo docker inspect -f '{{{{.State.Status}}}} {{{{.State.ExitCode}}}}' {container} "
                             "2>/dev/null || echo 'absent -'")
    logs = logs_script or f"sudo docker logs --tail 2000 {container} 2>&1"
    return (f"s=$({state}); set -- $s; "
            f"j=0; ls {tape_glob} >/dev/null 2>&1 && grep -qs {join_pattern} {tape_glob} && j=1; "
            f"[ $j = 1 ] || {{ {logs} | grep -qs {join_pattern} && j=1; }}; "
            f"n=$(ls -t {tape_glob} 2>/dev/null | head -1); "
            "a=-; [ -n \"$n\" ] && a=$(( $(date +%s) - $(stat -c %Y \"$n\") )); "
            "tw=0; [ -n \"$n\" ] && tail -n 1 \"$n\" | grep -qs rl_sampler_waiting && tw=1; "
            "st=$1 ex=$2; " + metrics +
            "echo \"YETO_NC_PROBE status=$st exit=$ex joined=$j age=$a tw=$tw tok=$t run=$r wq=$w\"")


# -- fencing ------------------------------------------------------------------

FENCE_PREFIX = "pool-fence"


def fence_incarnation(store: Any, shard: str, *, by: str, clock: Callable[[], float] = time.time) -> dict[str, Any]:
    """Supersede the running incarnation of ``shard`` without asking the shard to stop
    (no handoff record): the sampler restarted in place on the same shard claims the
    lease anew at its join. Used for a stalled engine: its claims are released at once."""
    from yeto.rl.engine.incarnation_lease import IncarnationLease  # noqa: PLC0415

    return IncarnationLease(store, shard, f"{FENCE_PREFIX}:{by}", clock=clock).claim()


def fence_shard(store: Any, shard: str, *, by: str, clock: Callable[[], float] = time.time) -> dict[str, Any]:
    """Isolate ``shard`` (its node is taken for gone): ask its sampler to stop
    (handoff record) and claim its incarnation lease as ``pool-fence:<by>``. A
    sampler of the shard that is still running (a node back from a network
    partition) is fenced at its next lease read and writes no further group; a
    group it wrote anyway after the claim (minus the gate's clock margin) is a
    stale incarnation to the trainer. Groups written before stay (they are whole:
    a group is one object). Returns the lease record."""
    from yeto.rl.engine.incarnation_lease import IncarnationLease  # noqa: PLC0415
    from yeto.rl.engine.sampler_resume import request_handoff  # noqa: PLC0415

    request_handoff(store, shard, f"{FENCE_PREFIX}:{by}", clock=clock)
    return IncarnationLease(store, shard, f"{FENCE_PREFIX}:{by}", clock=clock).claim()


class ObjectRecords:
    """The record view ``fence_shard`` needs over an object API (``read``/``write``/``list``)."""

    def __init__(self, objects: Any):
        self.objects = objects


def s3_fence_store(uri: str, env_file: str | os.PathLike[str] | None,
                   environ: Mapping[str, str] | None = None) -> ObjectRecords:
    """The run's store as records for :func:`fence_shard`, from the launcher: the
    buffer URI and an env file in the sampler's format (``AWS_ACCESS_KEY_ID``,
    ``AWS_SECRET_ACCESS_KEY``, ``AWS_ENDPOINT_URL``)."""
    from yeto.cloud import s3_store  # noqa: PLC0415

    env = dict(os.environ if environ is None else environ)
    if env_file:
        env.update(read_env_file(env_file))
    env.setdefault(s3_store.CREDENTIALS_ENV, s3_store.CREDENTIALS_AWS_ENV)
    if env.get(s3_store.AWS_ENDPOINT_ENV):
        env.setdefault(s3_store.ENDPOINT_ENV, env[s3_store.AWS_ENDPOINT_ENV])
    env.setdefault(s3_store.REGION_ENV, "auto")
    return ObjectRecords(s3_store.head_objects(uri, environ=env))


def read_env_file(path: str | os.PathLike[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.removeprefix("export ").strip()] = value.strip().strip("'\"")
    return out
