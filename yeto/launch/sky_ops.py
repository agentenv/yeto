"""SkyPilot operations: cluster naming, the SDK wrapper and job log streaming."""

from __future__ import annotations

import sys
import time

from ..gpu_spec import ClusterSpec
from .teardown import teardown_island


def sky_cluster_name(name: str) -> str:
    """Every cluster name yeto hands sky is lower case.

    Verda hostnames are lower case, and sky 0.13's Verda status query
    matches the *display* name against them (its query_instances takes
    its arguments one slot off); an upper-case region such as FIN-03 in
    the name then reads as "no instances", sky drops the record, and the
    same-name relaunch that follows deletes the running node. Lower-case
    names break that chain at the first link (design D1)."""
    return name.lower()


def learner_cluster_names(prefix: str, specs: list[ClusterSpec]) -> list[str]:
    """Deterministic learner cluster names for a run: computable from the
    launch args alone, so the CLI can record them before provisioning.
    Modal islands always end in `-modal` (whatever their region hint) so
    every consumer — controller, registry, `yeto down` — can route them
    by name."""
    from ..modal_runner import modal_island_name

    return [
        modal_island_name(prefix, m)
        if spec.cloud == "modal"
        else sky_cluster_name(f"{prefix}-l{m}-{spec.region or spec.cloud}")
        for m, spec in enumerate(specs)
    ]


def maybe_install_nebius_reclaim_hook(task, launched, name: str) -> bool | None:
    """rl-spot-cost-saving (S19 real spot): a droppable island on Nebius gets the host
    unit that forwards the VM stop SIGTERM to the learner (yeto.cloud.nebius_reclaim).
    None when the island does not need it; never raises."""
    from ..cloud import nebius_reclaim

    if not nebius_reclaim.wants_host_hook(getattr(task, "envs", None)):
        return None
    handle = launched[1] if isinstance(launched, tuple) and len(launched) > 1 else None
    if handle is None:
        print(f"[launcher] {name}: Nebius reclaim hook NOT installed (no cluster handle)")
        return False
    return nebius_reclaim.install_host_hook(handle, name=name)


def _tail(cluster: str, job_id: int, prefix: str, collector=None) -> int:
    """Stream a sky job's log into the launcher output (and the tape collector).

    Root cause of "sky.tail_logs delivers the setup lines but no run-phase
    line" (S1 2x1 L40S 2026-10-03/04; -5r1 r7 stream silent after the learner
    restart), read from skypilot 0.13.0: the API server's ``log_streamer``
    appends a ``<heartbeat></heartbeat>`` control payload after every 30 s
    without a new log line (sky/server/stream_utils.py, ``_HEARTBEAT_INTERVAL``),
    and the client's ``decode_rich_status`` (sky/utils/rich_utils.py) yields
    ``None`` for EVERY control payload when it runs off the main thread --
    which this tail thread does. The sdk docstring says ``None`` means "the log
    has been completely streamed", so the old loop broke on the first quiet
    30 s (model load, generation) and the thread ended silently while the job
    ran on. The end of the stream is the iterator being exhausted; ``None`` is
    a control message and is skipped."""
    import sky

    while True:
        try:
            it = sky.tail_logs(cluster, job_id, follow=True, preload_content=False)
            for line in it:
                if line is None:  # rich-status/heartbeat control payload, not the end
                    continue
                print(f"[{prefix}] {line.rstrip()}", flush=True)
                if collector is not None:
                    collector.feed(line)
            return 0
        except Exception as e:  # transient stream drops: reconnect
            print(f"[{prefix}] log stream error: {e}; retrying", flush=True)
            time.sleep(5)


# The syncer refused this island (a deterministic configuration error, never
# fixed by a relaunch): elastic JOIN identity / strict HELLO contract mismatch.
SYNCER_REFUSAL_MARKERS = ("backend identity mismatch, JOIN refused",
                          "session mismatch (HELLO refused")


class SkySDKOps:
    """Thin adapter over the sky SDK: the only surface FleetController needs.

    Tests inject a fake with the same methods. Any sky call here may raise
    (e.g. the cluster no longer exists); the controller treats exceptions
    from job_status/cluster_up as "cluster gone".
    """

    def job_status(self, cluster: str, job_id: int):
        import sky

        return sky.get(sky.job_status(cluster, [job_id])).get(job_id)

    def job_alive(self, cluster: str, job_id: int) -> bool:
        """Whether `job_id` is still queued, setting up or running on `cluster`.

        `sky.queue` reads the cluster's job table without the UP gate that
        `sky.job_status` applies, so it still answers while the cluster's
        health reads INIT.  Any failure to read the queue counts as "not
        alive": the caller then falls back to its usual failure handling.
        """
        import sky

        try:
            records = sky.get(sky.queue(cluster))
        except Exception:
            return False
        for record in records:
            if isinstance(record, dict):
                rid, status = record.get("job_id"), record.get("status")
            else:
                rid = getattr(record, "job_id", None)
                status = getattr(record, "status", None)
            if rid != job_id:
                continue
            return status is not None and not status.is_terminal()
        return False

    def cluster_up(self, cluster: str) -> bool:
        import sky

        records = sky.get(sky.status(cluster_names=[cluster]))
        if not records:
            return False
        record = records[0]
        status = (
            record.get("status")
            if isinstance(record, dict)
            else getattr(record, "status", None)
        )
        return status == sky.ClusterStatus.UP

    def rl_strict_failure(self, cluster: str, job_id: int) -> str | None:
        """Return a strict RL event from a failed job's existing log."""

        import sky

        try:
            lines = sky.tail_logs(
                cluster, job_id, follow=False, preload_content=False
            )
            for line in lines:
                text = str(line).strip()
                if (
                    "[yeto-rl-strict-failure]" in text
                    or "RL strict failure " in text
                    or "StrictRlInvariantError:" in text
                    or any(marker in text for marker in SYNCER_REFUSAL_MARKERS)
                ):
                    return text
        except Exception:
            return None
        return None

    def relaunch(self, task, cluster: str):
        """Re-provision `cluster` (same spec) and submit `task` as a new job.

        Blocking; returns the new job id, or None if provisioning failed.
        """
        import sky

        try:
            job_id, _handle = sky.get(sky.launch(task, cluster_name=cluster))
            maybe_install_nebius_reclaim_hook(task, (job_id, _handle), cluster)
            return job_id
        except Exception as e:
            print(f"[launcher] relaunch of {cluster} failed: {e}", file=sys.stderr)
            return None

    def __init__(self, nodes_by_cluster=None):
        # rl-multinode-island D10: the controller's failure-path teardown (job FAILED ->
        # recovery refused -> _down) must verify EVERY node instance, like run()'s final
        # teardown; without the mapping a cluster is treated as single-node.
        self.nodes_by_cluster = dict(nodes_by_cluster or {})

    def down(self, cluster: str) -> None:
        import sky

        nodes = getattr(self, "nodes_by_cluster", None) or {}
        teardown_island(sky, cluster, num_nodes=nodes.get(cluster, 1))

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)
