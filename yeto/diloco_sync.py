"""Shared step-boundary orchestration for DiLoCo learner syncs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from .fragments import Fragment, FragmentLayout
from .protocol import (
    DTYPE_BF16,
    DTYPE_F32,
    DTYPE_Q4,
    FinalManifest,
    PullRequest,
    SyncerClient,
    bulk_dtype,
)
from .tensor_io import fragment_flat, pack_tensor, quantize_q4, unpack_fragment

# ``--wire-dtype`` names of the learner sessions.
WIRE_DTYPES = {"bf16": DTYPE_BF16, "f32": DTYPE_F32, "q4": DTYPE_Q4}


def wire_dtype_code(name: str) -> int:
    """The session dtype of a ``--wire-dtype`` name (``fp32`` is ``f32``)."""
    try:
        return WIRE_DTYPES["f32" if name == "fp32" else name]
    except KeyError:
        raise ValueError(f"wire dtype must be one of {tuple(WIRE_DTYPES)}, got {name!r}") from None


def connect_syncer(spec: str, learner_id: int, layout: FragmentLayout, wire_dtype: int,
                   wan_streams: int, **client_kwargs):
    """Start one island's syncer client: a ``SyncerClient`` for ``host:port``,
    a ``ShardedSyncerClient`` for a comma-separated shard list."""
    from .syncer_shards import ShardedSyncerClient, parse_syncer_addrs

    addrs = parse_syncer_addrs(spec)
    if len(addrs) > 1:
        # Fragment-sharded head: one connection per shard, every fragment
        # routed to its owner (yeto/syncer_shards.py).
        client = ShardedSyncerClient(addrs, learner_id, layout, wire_dtype, wan_streams, **client_kwargs)
    else:
        client = SyncerClient(addrs[0], learner_id, layout, wire_dtype, wan_streams, **client_kwargs)
    client.start()
    return client


def take_ready_pulls(
    state: DiLoCoSyncState,
    *,
    steps_total: int,
    units_total: int,
    anchored: Callable[[int], bool],
    min_steps: int = 1,
) -> list[tuple[PullRequest, int, int]]:
    """``(pull, c_steps, c_units)`` of every pending pull that may be answered
    now (the fragment trained ``min_steps`` steps since its last broadcast and
    has an anchor); the others stay pending, in order."""
    ready, still = [], []
    for pull in state.pending_pulls:
        fid = pull.fragment_id
        c_steps = steps_total - state.steps_at_reset[fid]
        if c_steps < max(1, min_steps) or not anchored(fid):
            still.append(pull)
            continue
        ready.append((pull, c_steps, units_total - state.units_at_reset[fid]))
    state.pending_pulls = still
    return ready


def encode_push_delta(client, delta: torch.Tensor) -> bytes:
    """PUSH wire bytes of one base-relative fragment delta."""
    if client.dtype == DTYPE_Q4:
        return quantize_q4(delta)
    return pack_tensor(delta, client.dtype)


def boundary_metrics(state: DiLoCoSyncState, *, merges: int, pushes: int, push_bytes: int,
                     push_delta_norm: float, steps_total: int) -> dict:
    """The observer row of one boundary that merged or pushed something."""
    # Per-fragment lag behind the newest global version this island has
    # seen: the single clearest signal that an island is falling behind
    # the fleet rather than merely training slowly.
    staleness = [state.global_step - v for v in state.fragment_versions]
    return {
        "sync/merges_applied": merges,
        "sync/pushes": pushes,
        "sync/push_bytes": push_bytes,
        "sync/push_delta_norm": push_delta_norm,
        "sync/pending_pulls": len(state.pending_pulls),
        "sync/staleness_max": max(staleness),
        "sync/staleness_mean": sum(staleness) / len(staleness),
        "local_step": steps_total,
        "global_step": state.global_step,
    }


@dataclass
class DiLoCoSyncState:
    """Persistent per-fragment state owned by a learner's sync boundary."""

    steps_at_reset: list[int]
    units_at_reset: list[int]
    fragment_versions: list[int]
    pending_pulls: list[PullRequest] = field(default_factory=list)
    anchors: list[torch.Tensor | None] | None = None
    global_step: int = 0
    shutdown: bool = False

    @classmethod
    def create(
        cls,
        num_fragments: int,
        *,
        track_anchors: bool,
    ) -> DiLoCoSyncState:
        return cls(
            steps_at_reset=[0] * num_fragments,
            units_at_reset=[0] * num_fragments,
            fragment_versions=[0] * num_fragments,
            anchors=[None] * num_fragments if track_anchors else None,
        )


def sync_diloco_boundary(
    client: SyncerClient | None,
    layout: FragmentLayout,
    state: DiLoCoSyncState,
    *,
    steps_total: int,
    units_total: int,
    merge_alpha: float,
    snapshot_params: Callable[[], dict[str, torch.Tensor]],
    apply_flat: Callable[[Fragment, torch.Tensor], None],
    finalize: Callable[[], FinalManifest] | None,
    rank: int = 0,
    world: int = 1,
    device: torch.device | None = None,
    reset_counters_on_replicas: bool = False,
    shutdown_after_pulls: bool = False,
    observer: Callable[[dict], None] | None = None,
) -> bool:
    """Run one learner's non-blocking DiLoCo synchronization boundary.

    Wire values are torch tensors for every backend. ``snapshot_params`` and
    ``apply_flat`` bridge those values to the learner's model representation;
    a snapshot is built lazily and independently for merge and push phases.
    Returns true only after authoritative finalization has completed.

    ``observer`` receives one metrics dict per boundary that actually
    merged or pushed something. Every backend routes its sync through this
    function, so an observer attached here covers all four of them; quiet
    boundaries are skipped because "no fragment moved" is not an event.
    """
    actions: list[tuple[int, int, torch.Tensor]] = []
    finalizing = False
    if rank == 0 and client is not None:
        client.check_health()
        finalizing = client.finalizing.is_set()
        if not finalizing:
            for update in client.drain_updates():
                fid = update.fragment_id
                flat = unpack_fragment(
                    layout.fragments[fid],
                    update.data,
                    bulk_dtype(client.dtype),
                )
                if state.anchors is not None:
                    # PUSH deltas are relative to the raw global, not the
                    # delayed-application blend installed in the model.
                    state.anchors[fid] = flat.clone()
                actions.append((fid, update.version, flat))
        if not shutdown_after_pulls:
            state.shutdown = client.shutdown.is_set()

    if world > 1:
        metadata = [(fid, version) for fid, version, _ in actions] if rank == 0 else None
        box = [metadata, state.shutdown, finalizing]
        dist.broadcast_object_list(box, src=0)
        metadata, state.shutdown, finalizing = box
        if rank != 0:
            actions = [
                (fid, version, torch.empty(layout.fragments[fid].numel))
                for fid, version in metadata
            ]

    if finalizing:
        if finalize is None:
            raise RuntimeError("DiLoCo finalization requested without a finalizer")
        manifest = finalize()
        state.global_step = max(state.global_step, manifest.global_step)
        state.shutdown = True
        return True

    # Apply broadcasts before answering pulls. A next-round pull can overtake
    # the previous broadcast on the control stream; applying first resets its
    # self-clock and ensures the later PUSH carries the fresh base version.
    merge_snapshot = None
    for fid, version, flat in actions:
        if device is not None:
            flat = flat.to(device)
        if world > 1:
            dist.broadcast(flat, src=0)
        if merge_alpha > 0:
            if merge_snapshot is None:
                merge_snapshot = snapshot_params()
            local = fragment_flat(layout.fragments[fid], merge_snapshot)
            flat = merge_alpha * local + (1.0 - merge_alpha) * flat
        apply_flat(layout.fragments[fid], flat)
        if rank == 0 or reset_counters_on_replicas:
            state.steps_at_reset[fid] = steps_total
            state.units_at_reset[fid] = units_total
            state.fragment_versions[fid] = version
        state.global_step = max(state.global_step, version)

    pushed = 0
    pushed_bytes = 0
    pushed_delta_norm = 0.0
    if rank == 0 and client is not None:
        state.pending_pulls.extend(client.drain_pulls())
        pull_snapshot = None
        for pull, c_steps, c_units in take_ready_pulls(
            state,
            steps_total=steps_total,
            units_total=units_total,
            anchored=lambda fid: state.anchors is not None and state.anchors[fid] is not None,
        ):
            fid = pull.fragment_id
            if pull_snapshot is None:
                pull_snapshot = snapshot_params()
            delta = fragment_flat(layout.fragments[fid], pull_snapshot).cpu() - state.anchors[fid]
            payload = encode_push_delta(client, delta)
            client.push_fragment(
                fid,
                pull.global_step,
                pull.round_attempt,
                state.fragment_versions[fid],
                steps_total,
                c_steps,
                c_units,
                payload,
            )
            if observer is not None:
                pushed += 1
                pushed_bytes += len(payload)
                pushed_delta_norm = max(pushed_delta_norm, float(delta.norm()))
        if shutdown_after_pulls:
            state.shutdown = client.shutdown.is_set()

    if observer is not None and rank == 0 and (actions or pushed):
        observer(
            boundary_metrics(
                state,
                merges=len(actions),
                pushes=pushed,
                push_bytes=pushed_bytes,
                push_delta_norm=pushed_delta_norm,
                steps_total=steps_total,
            )
        )

    return False
