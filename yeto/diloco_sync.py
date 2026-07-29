"""Shared step-boundary orchestration for DiLoCo learner syncs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.distributed as dist

from .fragments import Fragment, FragmentLayout
from .protocol import DTYPE_Q4, FinalManifest, PullRequest, SyncerClient, bulk_dtype
from .tensor_io import fragment_flat, pack_tensor, quantize_q4, unpack_fragment


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
) -> bool:
    """Run one learner's non-blocking DiLoCo synchronization boundary.

    Wire values are torch tensors for every backend. ``snapshot_params`` and
    ``apply_flat`` bridge those values to the learner's model representation;
    a snapshot is built lazily and independently for merge and push phases.
    Returns true only after authoritative finalization has completed.
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

    if rank == 0 and client is not None:
        state.pending_pulls.extend(client.drain_pulls())
        still_pending = []
        pull_snapshot = None
        for pull in state.pending_pulls:
            fid = pull.fragment_id
            c_steps = steps_total - state.steps_at_reset[fid]
            if c_steps < 1:
                still_pending.append(pull)
                continue
            c_units = units_total - state.units_at_reset[fid]
            if pull_snapshot is None:
                pull_snapshot = snapshot_params()
            anchor = state.anchors[fid] if state.anchors is not None else None
            if anchor is None:
                still_pending.append(pull)
                continue
            delta = fragment_flat(layout.fragments[fid], pull_snapshot).cpu() - anchor
            if client.dtype == DTYPE_Q4:
                payload = quantize_q4(delta)
            else:
                payload = pack_tensor(delta, client.dtype)
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
        state.pending_pulls = still_pending
        if shutdown_after_pulls:
            state.shutdown = client.shutdown.is_set()

    return False
