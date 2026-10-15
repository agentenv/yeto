"""Cloud binding of a convertible pool (rl-trainer-start-stop L7.5).

A launch whose ``--gpu`` islands are all declared ``--rl-node-class
islandM=convertible`` runs as a convertible pool: every island boots as a
rollout node, and the head's trainer loop (S1, :mod:`.trainer_loop`) converts the
best node to trainer when a wave of samples is waiting, then back
(:mod:`yeto.rl.engine.convertible_remote`). The declaration is the switch: the
trainer loop runs without ``--rl-trainer-scheduler``, no island is held back,
and ``T_s`` starts at :data:`DEFAULT_CONVERT_START_SECONDS` until a conversion is
measured.

The head reaches the islands only through the run's checkpoint store, read and
written with the Modal Volume SDK (``modal-volume://``); so the islands are Modal
islands. A sky island would need the head to read its bucket store: not bound yet.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from yeto.rl.engine.convertible import NodeProfile, NodeSpec, TrainerNeeds
from yeto.rl.engine.convertible_remote import ConvertibleGate, ConvertibleIslands
from yeto.rl.engine.sample_buffer import BUFFER_DIR, INDEX, CommittedBacklog, FileSampleBuffer, merge_shards
from yeto.rl.engine.sample_roles import convertible_islands, node_name
from yeto.rl.engine.trainer_remote import RemoteTrainerLifecycle, StoreCutLog, consumed_watermark

DEFAULT_CONVERT_START_SECONDS = 60.0
MIRROR_DIR = "buffer-mirror"
PARAMETER_SOURCE_DOMAIN = "modal"  # the modal-volume:// store holds the cuts


def enabled(args: Any) -> bool:
    from yeto.rl.engine.sample_roles import is_convertible_pool

    return is_convertible_pool(args)


def check(args: Any) -> None:
    """Refuse a convertible pool the head cannot drive (before any launch)."""
    from yeto.gpu_spec import parse_gpu_spec

    if getattr(args, "rl_trainer_island", None) is not None:
        raise SystemExit("--rl-trainer-island: a convertible pool holds no trainer island back; "
                         "a convertible node becomes the trainer")
    specs = parse_gpu_spec(args.gpu)
    islands = convertible_islands(args)
    if len(specs) < 2 or islands != list(range(len(specs))):
        raise SystemExit("--rl-node-class islandM=convertible: a convertible pool declares every "
                         f"--gpu island convertible, at least two (got {len(specs)} island(s), "
                         f"convertible {islands})")
    clouds = sorted({s.cloud for s in specs})
    if clouds != ["modal"]:
        raise SystemExit(f"a convertible pool needs Modal islands (got {clouds}): the head drives "
                         "the nodes through a modal-volume:// checkpoint store; sky is not bound yet")
    if not str(getattr(args, "rl_checkpoint_store", None) or "").startswith("modal-volume://"):
        raise SystemExit("a convertible pool needs a modal-volume:// --rl-checkpoint-store: the "
                         "buffer, the cuts and the node channels live there")


def node_profiles(args: Any, nodes: Sequence[NodeSpec]) -> list[NodeProfile]:
    """The declared profile of every convertible node, from its ``--rl-node-class``
    entry (``nodes``: ``trainer_loop.node_specs``) and the island's ``--gpu`` entry
    (GPU type, count, memory, network domain)."""
    from yeto.gpu_spec import parse_gpu_spec
    from yeto.memory_estimate import gpu_capacity_gib

    declared = {s.name: s for s in nodes}
    specs = parse_gpu_spec(args.gpu)
    out = []
    for m in convertible_islands(args):
        spec, node = specs[m], declared[node_name(m)]
        gpu = str(spec.gpu).rstrip("!")
        domain = spec.cloud if not getattr(spec, "region", None) else f"{spec.cloud}/{spec.region}"
        out.append(NodeProfile(node.name, spot=node.spot, precision=node.precision, gpu=gpu,
                               gpus=int(spec.gpus_per_node), gpu_memory_gb=gpu_capacity_gib(gpu),
                               domain=domain))
    return out


def trainer_needs(args: Any, precision: str) -> TrainerNeeds:
    """Precision, GPUs and memory a trainer node needs; the memory is the launch's
    per-GPU estimate (None when not estimated: level 1 then checks spot and
    precision only)."""
    from yeto.gpu_spec import parse_gpu_spec
    from yeto.memory_estimate import estimate_for_launch

    specs = parse_gpu_spec(args.gpu)
    peaks = []
    for spec in specs:
        try:
            est = estimate_for_launch(args, spec, margin=1.0)
        except Exception:  # noqa: BLE001 - an estimate is optional input of level 1
            est = {}
        if est.get("estimated") and est.get("peak_gib") is not None:
            peaks.append(float(est["peak_gib"]))
    gpus = int(getattr(args, "actor_num_gpus_per_node", 0) or specs[0].gpus_per_node)
    return TrainerNeeds(precision=precision, memory_gb=max(peaks) if peaks else None,
                        gpus=gpus, domains=(PARAMETER_SOURCE_DOMAIN,))


@dataclass
class MergedBacklog:
    """The head's buffer view of a pool: merge the node shards into the index (the
    head is its only writer), then count from the committed watermark.

    Only samples the trainer can train count (behavior version at most
    ``max_policy_age`` behind ``current_version``): a node that has not converted
    since keeps generating with the version it last served, and the trainer
    passes over its older samples (P0-S gate), so they must not start a trainer."""

    files: Any
    nodes: tuple[str, ...]
    inner: CommittedBacklog
    max_policy_age: int

    def backlog(self, current_version: int) -> Any:
        merge_shards(self.files, self.nodes)
        stats = self.inner.backlog(current_version)
        floor = int(current_version) - int(self.max_policy_age)
        fresh = sum(n for v, n in (stats.version_histogram or {}).items() if v >= floor)
        if fresh == stats.samples:
            return stats
        return replace(stats, samples=fresh, batches=fresh / self.inner.buffer.batch_size)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def binding(args: Any, *, journal_dir: Path, nodes: Sequence[NodeSpec], precision: str,
            files: Any = None) -> Any:
    """The :class:`~.trainer_island.CloudTrainerBinding` of a convertible pool.
    ``nodes``: the declared node classes; ``precision``: the trainer's; ``files``
    replaces the Modal Volume in tests."""
    from .storage import MODAL_CHECKPOINT_STORE_MOUNT, modal_checkpoint_store
    from .trainer_island import CloudTrainerBinding, MirrorStore, ModalVolumeFiles

    if files is None:
        volume, mount = modal_checkpoint_store(args)
        files = ModalVolumeFiles(volume, mount[len(MODAL_CHECKPOINT_STORE_MOUNT):])
    poll_s = float(getattr(args, "rl_trainer_command_poll_seconds", None) or 5.0)
    islands = ConvertibleIslands(files, node_profiles(args, nodes), trainer_needs(args, precision),
                                 poll_s=poll_s)
    cuts = StoreCutLog(files)
    mirror = MirrorStore(Path(journal_dir) / MIRROR_DIR, files, [f"{BUFFER_DIR}/{INDEX}"])
    lifecycles: list[RemoteTrainerLifecycle] = []

    def lifecycle(first: int) -> RemoteTrainerLifecycle:
        lifecycles.append(RemoteTrainerLifecycle(islands, files, first_incarnation=first,
                                                 poll_s=poll_s))
        return lifecycles[-1]

    def watermark() -> int:
        return consumed_watermark(cuts, lifecycles[-1] if lifecycles else None)

    def wrap(scheduler: Any, record: Callable[..., None]) -> ConvertibleGate:
        islands.record = record
        return ConvertibleGate(scheduler, islands)

    buffer = CommittedBacklog(FileSampleBuffer(mirror, batch_size=int(args.rollout_batch_size)),
                              watermark)
    return CloudTrainerBinding(
        lifecycle=lifecycle,
        buffer=MergedBacklog(files, tuple(n.name for n in islands.nodes), buffer,
                             max_policy_age=int(getattr(args, "rl_max_policy_age", 0) or 0)),
        cuts=cuts, release_orphan=islands.release_incarnation, data_ended=islands.data_ended,
        start_events=islands.start_events, wrap_scheduler=wrap, on_finished=islands.finish)
