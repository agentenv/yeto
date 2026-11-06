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


def supersede_pull_attempts(pulls: list[PullRequest]) -> list[PullRequest]:
    """``pulls`` without any pull a newer attempt of the same round replaced.

    A round the syncer reopens after a quorum timeout sends a new pull with
    a higher ``round_attempt``; a push answering the old attempt would be
    rejected as out of round, so only the newest attempt is kept, in order.
    A repeat of the same attempt (resent after a redial) is kept once, so the
    fragment is not pushed twice for it.
    """
    newest: dict[tuple[int, int], int] = {}
    for pull in pulls:
        key = (int(pull.fragment_id), int(pull.global_step))
        newest[key] = max(newest.get(key, -1), int(pull.round_attempt))
    kept, seen = [], set()
    for pull in pulls:
        key = (int(pull.fragment_id), int(pull.global_step))
        attempt = int(pull.round_attempt)
        if attempt == newest[key] and (key, attempt) not in seen:
            seen.add((key, attempt))
            kept.append(pull)
    return kept


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
    for pull in supersede_pull_attempts(state.pending_pulls):
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

    def resume_record(self, *, anchored: list[bool], units_total: int,
                      spent: list[tuple[int, int] | None] | None = None) -> dict:
        """The per-fragment part of an island checkpoint (JSON-safe): which
        version each fragment's anchor is, and the H-gate / token counters
        measured from it.  The anchor values themselves are not included."""
        n = len(self.fragment_versions)
        spent = spent if spent is not None else [None] * n
        return {
            "versions": [int(v) for v in self.fragment_versions],
            "anchored": [bool(a) for a in anchored],
            "steps_at_reset": [int(s) for s in self.steps_at_reset],
            "units_at_reset": [int(u) for u in self.units_at_reset],
            "spent": [None if s is None else [int(s[0]), int(s[1])] for s in spent],
            "units_total": int(units_total),
            "global_step": int(self.global_step),
        }

    def restore_fragment(self, record: dict, fid: int) -> None:
        """Take fragment ``fid``'s counters from a ``resume_record`` (a resumed
        island that kept its local progress keeps its H-gate clock too)."""
        self.steps_at_reset[fid] = int(record["steps_at_reset"][fid])
        self.units_at_reset[fid] = int(record["units_at_reset"][fid])


# What a resumed island does with the first broadcast of a fragment.
RESUME_OVERWRITE = "overwrite"  # install the global (no usable record: a cold start)
RESUME_ANCHOR = "anchor"        # same version as the checkpoint's anchor: anchor only
RESUME_SHIFT = "shift"          # newer global: global_new + (local - anchor_old)


def resume_fragment_mode(record: dict | None, fid: int, version: int, *, have_anchor: bool) -> str:
    """How a resumed island installs the syncer's first broadcast of ``fid``.

    The checkpoint holds ``local``, trained from the anchor of
    ``record["versions"][fid]``.  The same version again means the broadcast
    *is* that anchor, so the local parameters stay as they are.  Any other
    version moves the island's local progress onto the new global, which
    needs the old anchor value (``have_anchor``); without it the fragment
    falls back to the cold-start overwrite."""
    if record is None or not record["anchored"][fid]:
        return RESUME_OVERWRITE
    if int(version) == int(record["versions"][fid]):
        return RESUME_ANCHOR
    return RESUME_SHIFT if have_anchor else RESUME_OVERWRITE


# What a running island does with a broadcast of a fragment it already
# holds an anchor of (``live_install_mode``); None = the ordinary install.
LIVE_CONFIRM = "confirm"  # a replay of its anchor: nothing to install
LIVE_REBASE = "rebase"    # a catch-up: global_new + (local - anchor)


def live_install_mode(*, anchored: bool, current_version: int, version: int, catch_up: bool,
                      pushed_from: int | None) -> str | None:
    """How a running island installs a broadcast of a fragment.

    A broadcast of the version the island is anchored at is a replay of
    that anchor (``LIVE_CONFIRM``): the local parameters stay bit for bit.
    The first broadcast after a redial (``catch_up``) is the syncer's
    current value, not the merge of a round the island answered: the
    island's progress since its anchor is moved onto it (``LIVE_REBASE``),
    also when the version went backwards (a syncer resumed from an older
    checkpoint) or kept its number with other values.  The exception is a
    newer catch-up of a fragment the island pushed from its current anchor
    (``pushed_from``): that delta may already be merged into it, so it is
    installed like any merge broadcast.  None: the ordinary install."""
    if not anchored:
        return None
    if not catch_up:
        return LIVE_CONFIRM if int(version) == int(current_version) else None
    if int(version) > int(current_version) and pushed_from is not None \
            and int(pushed_from) == int(current_version):
        return None
    return LIVE_REBASE


def shifted_resume_value(global_new: torch.Tensor, local: torch.Tensor, anchor_old: torch.Tensor) -> torch.Tensor:
    """``global_new + (local - anchor_old)`` in FP32: the island's progress since
    its checkpointed anchor, moved onto the newer global."""
    return global_new.to(torch.float32) + (local.to(torch.float32) - anchor_old.to(torch.float32))


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
