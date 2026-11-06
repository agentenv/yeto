"""Full-parameter DiLoCo sync of an ms-swift Megatron island.

ms-swift trains the model with Megatron-Core (TP/EP/PP, DistributedOptimizer,
optionally CPU optimizer offload through ``HybridDeviceOptimizer``).  This
module lets such an island speak the torch learner's DiLoCo protocol to the
unchanged Rust syncer (``yeto/learner.py`` + ``yeto/diloco_sync.py``):

* the same fragment layout semantics (``yeto.fragments.build_layout`` over the
  island's *global* trainable parameter set: whole tensors, size-balanced or
  strided, embeddings in the direct-averaged fragment);
* the same PUSH: ``local - raw_anchor`` per fragment, where ``raw_anchor`` is
  the exact global value of the last accepted broadcast (before any
  ``merge_alpha`` blend), answered only once the fragment has been anchored
  and trained on;
* the same BCAST handling: broadcasts are applied before pulls are answered,
  reset the fragment's counters and adopt the syncer's version;
* the same INIT (learner 0 seeds every fragment), finalization (raw final cut,
  then FINAL_ACK) and HELLO (layout, HMAC key).

What differs is only how a fragment is read and written, because no rank holds
a whole tensor:

Parameter namespace
    Every trainable Megatron parameter is named by its *global* Megatron name:
    the pipeline-local layer index is replaced by the global one
    (``layer_number - 1``) and an expert's local index by its global index
    (``local_expert_indices``).  So the name set and every tensor's full shape
    are independent of PP and EP, and the TP shards of one parameter
    concatenate along ``partition_dim`` into the full tensor.  Parameters the
    trainer does not train (``trainable_filter``: frozen vision tower and
    aligner) and buffers (router ``expert_bias``) are not part of the layout.

    The names are Megatron's, not the HF checkpoint's (the syncer is
    name-agnostic).  ms-swift's own converter, ``mcore_bridge``
    (``model.config.bridge``, NemotronHBridge), is deliberately not used for
    the gather and write: ``export_weights`` is a whole-model pass with no
    per-parameter selection, gathers every tensor over TP/EP/PP into
    accelerator buffers on every rank (``_all_gather_tp`` /
    ``_broadcast_ep_pp``) before ``target_device`` moves it, and reads the
    bf16 model copy, which is not the FP32 master and is stale outside the
    owned range under ``overlap_param_gather``; ``load_weights`` reads only a
    safetensors directory and writes the bf16 copy, not the optimizer master.
    With ~0.4 GiB of HBM headroom on the production pp0 stage none of that
    fits.  Megatron-Bridge (``yeto.rl.megatron_hf_boundary``) is not installed
    with ms-swift.  Everything protocol-side is the torch learner's own code
    (``yeto.diloco_sync``: session state, pull readiness, push encoding,
    metrics, client connection; ``yeto.fragments``; ``SyncerClient``).  Fused
    Megatron tensors whose element order depends on the TP size (Mamba
    ``in_proj``, gated ``linear_fc1``) make the layout TP-specific, so the
    session contract hash binds the TP / expert-TP sizes and the syncer
    refuses an island of a different parallel shape.

Read (PUSH, INIT)
    Collective, one bounded chunk of tensors at a time: every rank gathers the
    FP32 optimizer masters of its local parameters over the data-parallel group
    (``yeto.rl.megatron_masters.full_masters``: DistributedOptimizer shards; with
    the precision-aware optimizer + CPU offload of the production recipe that is
    ``HybridDeviceOptimizer.param_to_fp32_param[bf16 shard]``, a pinned host
    tensor, looked up afresh on every call because HDO rebuilds its maps on
    ``load_state_dict``), subtracts its local anchor shard, all-gathers TP
    shards, and the lowest owning rank sends the full FP32 tensor
    point-to-point to global rank 0, which packs it into the wire buffer.  The
    bf16 model copy is never read: with ``overlap_param_gather`` its non-owned
    ranges are stale after a step.  The PUSH carries ``local - anchor``, the
    torch learner's sign (the syncer's outer gradient is its negation, ``base -
    local``).

Write (BCAST, final cut)
    Rank 0 broadcasts the chunk over the island in the wire dtype (bf16 by
    default, lossless for a bf16 session); every rank slices its TP shard,
    records it as its anchor, blends with its current master when
    ``merge_alpha > 0``, and writes the FP32 master shard *and* the bf16 model
    copy (``yeto.rl.megatron_masters.write_masters``: the HDO FP32 tensor, which
    under offload is also the host copy the CPU Adam steps, and the parameter's
    storage inside the DDP ``param_data`` buffer, from which the next forward's
    param all-gather reads).  The Adam moments are kept, exactly as the torch
    learner keeps its AdamW state when ``apply_fragment`` overwrites a fragment.

Redial (catch-up)
    After a redial the syncer sends every fragment's current value.  The
    client drops a replay of a version already held (same bytes), so an
    unchanged fragment stays bit for bit; a broadcast of the anchored
    version that does reach the sync is only confirmed.  The first broadcast
    of a fragment after the redial (``BcastFragment.catch_up``) moves the
    island's progress since its anchor onto it, ``global_new + (local -
    anchor)``, unless the island pushed from that anchor (the delta may be in
    the new version: the ordinary install).  A syncer that crashed and
    resumed from an older checkpoint rewinds; the client takes its catch-up
    (``accept_rewound_catch_up``) and the island rebases onto it, instead of
    keeping the lost versions as bases the syncer refuses forever.

Only global rank 0 talks to the syncer.  Every boundary starts with one
``broadcast_object_list`` of rank 0's plan (fragments to apply, fragments to
push, finalization, shutdown, error), so every rank runs the same collectives
in the same order and an error on rank 0 is raised on every rank.

Decoupled DiLoCo (no island waits for another)
    The island never waits for a merge or for another island: a boundary
    only drains what the client already received (broadcasts are applied at
    the next boundary after they arrive), the PUSH bytes are handed to a
    background sender thread on rank 0 (``_PushSender``; the client encodes
    int4/int8 with tensor blocks there), and while a PUSH is still streaming
    (slow link or syncer, a redial) pulls stay pending instead of blocking.
    One fragment is gathered per boundary.  A pull is not answered from an
    anchor already spent on another round (the syncer's late carry refuses
    that as ``anchor_reused``); a dropped PUSH un-spends its anchor.  The
    syncer side is the RL async full-parameter preset (quorum 1, adaptive
    grace, ``--max-base-lag 1``, tokens2-over-steps, two fragment rounds in
    flight, rda); ``wire_dtype="int4"`` is its int4 tensor-block push.

    What still blocks a step is the collective gather of one fragment's FP32
    masters (and the in-island broadcast of a received one): host copies,
    gloo all-reduce over DP and all-gather over TP, then point-to-point to
    rank 0.  For the 120B / 24-card recipe and P = 16 one fragment is
    ~7.5e9 elements, ~0.3e9 per rank (~1.2 GB FP32), 30 GB arriving at rank
    0: roughly 10-20 s over TCP gloo, against a ~570 s step, i.e. ~2-5 % at
    one push and one apply per step.  It is not overlapped with training: the
    HDO masters change under the next step, so overlapping needs a host
    snapshot of the local shards plus a second set of gloo groups driven by a
    background thread in the same order on every rank.

Memory (N = trainable parameters of the island, P = fragments, n_r = the
elements of rank r's local parameter shards, C = ``chunk_numel``, t = TP size):

* device: nothing with the default ``transport="cpu"`` (the collectives run
  on host buffers over gloo mirrors of the island's groups; the only device
  access is the in-place host-to-device copy into the existing bf16
  parameters).  The production pp0 stage has ~0.4 GiB of HBM headroom, so
  ``transport="device"`` (chunk buffers on the accelerator, peak about
  ``4 * C * (2 + t)`` bytes) is only for islands with room to spare;
* every rank, host: one anchor per local shard in the broadcast dtype (bf16
  for a bf16 session, FP32 for the int4 / int8 push sessions, whose
  broadcasts are FP32), ``2 * n_r`` or ``4 * n_r`` bytes, plus one chunk in
  flight, about
  ``4 * C * (2 + t)`` bytes (C = 2**28 -> ~6 GiB at t = 4; a single tensor
  larger than C is a chunk of its own).  For the TP4/EP8/PP3 (world 24)
  Nemotron-3.5-Super recipe (~120B, MoE-dominated, ETP 1) n_r is about
  120e9 / 24 = 5e9: ~10 GB (int4: ~20 GB) of anchors per rank, ~80 GB
  (int4: ~160 GB) plus ~50 GB of chunks
  per 8-card host, beside the ~450 GB the offloaded masters and Adam moments
  take of its ~2 TB;
* rank 0, host additionally: the wire bytes of one pushed fragment
  (``2 * N / P`` for bf16) and the client's received broadcasts (each
  ``2 * N / P``; at most the syncer's ``--pipeline`` rounds in flight).  For
  N = 120e9 and P = 16 that is ~15 GB per fragment, ~45-60 GB peak (int4:
  the FP32 PUSH parts and FP32 broadcasts, ~30 GB each, ~90-120 GB).  A
  full-payload final cut holds every fragment in FP32 (``4 * N`` = 480 GB) on
  rank 0's host: run the syncer with ``--final-cut-payload manifest`` and
  ``final_cut_payload="manifest"`` for a 120B island.

Torch, Megatron-Core and the syncer client are imported lazily by the
objects that need them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("swift-megatron-sync")


def _ensure_visible_logs() -> None:
    """ms-swift configures only its own ``swift`` logger, so with an
    unconfigured root this module's INFO lines (``resume: fragment ...``,
    per-fragment installs) were dropped and only WARNING reached stderr
    through logging's last resort.  Give the module a stderr handler at INFO
    unless the process configured logging itself."""
    if log.handlers or logging.getLogger().handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[swift-megatron-sync] %(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    if log.level == logging.NOTSET:
        log.setLevel(logging.INFO)

# Megatron's tied/untied output projection is an embedding-like tensor for the
# direct-averaged fragment, as ``lm_head`` is for the torch learner.
OUTPUT_LAYER_AVG_REGEX = r"(^|\.)output_layer\.weight$"
DEFAULT_CHUNK_NUMEL = 1 << 28
CONTRACT_DOMAIN = b"yeto-swift-megatron-native-v1\0"
FINAL_CUT_PAYLOADS = ("full", "manifest")
# Island checkpoint record of the sync (``save_state`` / ``resume_dir``), next
# to the trainer's checkpoint: the JSON on global rank 0, the optional anchor
# values one file per rank.
RESUME_STATE_FILE = "yeto_diloco_state.json"
RESUME_ANCHOR_DIR = "yeto_diloco_anchors"
RESUME_FORMAT = 1
# ``yeto.diloco_sync.resume_fragment_mode`` results (that module imports torch).
RESUME_OVERWRITE, RESUME_ANCHOR, RESUME_SHIFT = "overwrite", "anchor", "shift"
# ``yeto.diloco_sync.live_install_mode`` results.
LIVE_CONFIRM, LIVE_REBASE = "confirm", "rebase"

# ``spent_link`` of an anchor spent by the process the island resumed from.
_PREVIOUS_PROCESS = "previous-process"

_LAYER_MODULE_RE = re.compile(r"(^|\.)decoder\.layers\.(\d+)$")
_GROUPED_EXPERT_RE = re.compile(r"^(.*?)(weight|bias)(\d+)$")


class SwiftMegatronSyncError(RuntimeError):
    """The Megatron island cannot run the full-parameter DiLoCo sync as configured."""


# ----------------------------------------------------------------------------- parallel context
@dataclass(frozen=True)
class ParallelContext:
    """The process groups the sync needs, decoupled from ``parallel_state``
    so the sync runs on any group layout (tests build gloo subgroups).

    ``tp_*`` is the tensor-parallel group of dense parameters, ``etp_*`` the
    expert-tensor-parallel group of expert parameters (``param.allreduce is
    False``).  The data-parallel groups come from the optimizer
    (``data_parallel_group`` of each DistributedOptimizer leaf)."""

    rank: int
    world: int
    tp_group: Any
    tp_rank: int
    tp_size: int
    etp_group: Any
    etp_rank: int
    etp_size: int

    @classmethod
    def from_megatron(cls) -> ParallelContext:
        import torch.distributed as dist
        from megatron.core import parallel_state as mpu

        def expert_tp(name: str, fallback: Callable[[], Any]) -> Any:
            getter = getattr(mpu, name, None)
            return getter() if getter is not None else fallback()

        return cls(
            rank=int(dist.get_rank()),
            world=int(dist.get_world_size()),
            tp_group=mpu.get_tensor_model_parallel_group(),
            tp_rank=int(mpu.get_tensor_model_parallel_rank()),
            tp_size=int(mpu.get_tensor_model_parallel_world_size()),
            etp_group=expert_tp("get_expert_tensor_parallel_group", mpu.get_tensor_model_parallel_group),
            etp_rank=int(expert_tp("get_expert_tensor_parallel_rank", mpu.get_tensor_model_parallel_rank)),
            etp_size=int(expert_tp("get_expert_tensor_parallel_world_size",
                                   mpu.get_tensor_model_parallel_world_size)),
        )


# ----------------------------------------------------------------------------- global names
@dataclass
class LocalShard:
    """One local trainable parameter and where it sits in the global tensor."""

    name: str  # global Megatron name
    param: Any
    expert: bool
    part_dim: int  # TP partition dim, -1 when replicated over the TP group
    tp_rank: int
    tp_size: int
    tp_group: Any = field(repr=False, default=None)

    @property
    def local_shape(self) -> tuple[int, ...]:
        return tuple(int(d) for d in self.param.shape)

    def full_shape(self) -> tuple[int, ...]:
        shape = list(self.local_shape)
        if self.part_dim >= 0:
            shape[self.part_dim] *= self.tp_size
        return tuple(shape)


def unwrap_chunk(chunk: Any) -> Any:
    """The bare Megatron model of a (DDP / Float16Module wrapped) model chunk."""

    seen = 0
    while hasattr(chunk, "module") and seen < 8:
        chunk = chunk.module
        seen += 1
    return chunk


def _global_layer_prefixes(chunk: Any) -> list[tuple[str, str]]:
    """``(local prefix, global prefix)`` for every decoder layer of ``chunk``,
    longest first.  Megatron numbers ``decoder.layers`` per pipeline stage (and
    virtual stage); ``layer_number`` is the 1-based global index."""

    out = []
    for name, module in chunk.named_modules():
        match = _LAYER_MODULE_RE.search(name)
        if match is None or not hasattr(module, "layer_number"):
            continue
        global_name = name[: match.start(2)] + str(int(module.layer_number) - 1)
        out.append((name + ".", global_name + "."))
    return sorted(out, key=lambda item: -len(item[0]))


def _expert_modules(chunk: Any) -> list[tuple[str, list[int]]]:
    """``(prefix, local_expert_indices)`` of every expert container (the
    ``experts`` child of a MoE layer), longest first.

    Older Megatron-Core put ``local_expert_indices`` on the expert container
    itself; since 0.1x only the MoE layer (its parent, which also holds the
    router and shared experts) carries it, so fall back to the parent."""

    modules = dict(chunk.named_modules())
    out = []
    for name, module in modules.items():
        if name.rsplit(".", 1)[-1] != "experts":
            continue
        indices = getattr(module, "local_expert_indices", None)
        if indices is None and "." in name:
            indices = getattr(modules.get(name.rsplit(".", 1)[0]), "local_expert_indices", None)
        if indices is None:
            continue
        out.append((name + ".", [int(i) for i in indices]))
    return sorted(out, key=lambda item: -len(item[0]))


def _globalize_expert(rest: str, indices: Sequence[int], name: str) -> str:
    """The global form of ``rest`` (the name below an expert container)."""

    head, _, tail = rest.partition(".")
    if head == "local_experts":  # SequentialMLP: local_experts.<e>.<...>
        local, _, below = tail.partition(".")
        if not local.isdigit():
            raise SwiftMegatronSyncError(f"unrecognised expert parameter {name!r}")
        return f"local_experts.{indices[int(local)]}.{below}"
    match = _GROUPED_EXPERT_RE.match(rest)
    if match is not None:  # TEGroupedMLP: linear_fc1.weight<e>
        return f"{match.group(1)}{match.group(2)}{indices[int(match.group(3))]}"
    raise SwiftMegatronSyncError(
        f"expert parameter {name!r} stacks several experts in one tensor (legacy GroupedMLP); "
        "use TEGroupedMLP or SequentialMLP")


def global_parameter_name(local_name: str, layers: Sequence[tuple[str, str]],
                          experts: Sequence[tuple[str, list[int]]]) -> str:
    """``local_name`` with its layer and expert indices made global."""

    name = local_name
    for prefix, indices in experts:
        if name.startswith(prefix):
            name = prefix + _globalize_expert(name[len(prefix):], indices, local_name)
            break
    for local, global_prefix in layers:
        if name.startswith(local):
            name = global_prefix + name[len(local):]
            break
    return name


def default_trainable_filter(name: str, param: Any) -> bool:
    return bool(getattr(param, "requires_grad", False))


def local_shards(model_chunks: Sequence[Any], ctx: ParallelContext,
                 trainable_filter: Callable[[str, Any], bool] | None = None) -> list[LocalShard]:
    """This rank's trainable parameters under their global names (sorted)."""

    keep = trainable_filter or default_trainable_filter
    out: dict[str, LocalShard] = {}
    seen: set[int] = set()
    for wrapped in model_chunks:
        chunk = unwrap_chunk(wrapped)
        layers = _global_layer_prefixes(chunk)
        experts = _expert_modules(chunk)
        for local_name, param in chunk.named_parameters():
            if id(param) in seen or not keep(local_name, param):
                continue
            seen.add(id(param))
            name = global_parameter_name(local_name, layers, experts)
            if name in out:
                raise SwiftMegatronSyncError(f"two local parameters map to the global name {name!r}")
            expert = getattr(param, "allreduce", True) is False
            if expert and ".experts." in local_name and not any(
                    local_name.startswith(prefix) for prefix, _ in experts):
                # Without its global expert index the tensor would alias the
                # same-named tensor of every other expert-parallel rank.
                raise SwiftMegatronSyncError(
                    f"{local_name}: expert parameter without local_expert_indices on its container")
            tp_rank, tp_size, group = ((ctx.etp_rank, ctx.etp_size, ctx.etp_group) if expert
                                       else (ctx.tp_rank, ctx.tp_size, ctx.tp_group))
            part_dim = -1
            if getattr(param, "tensor_model_parallel", False) and tp_size > 1:
                if int(getattr(param, "partition_stride", 1)) != 1:
                    raise SwiftMegatronSyncError(f"{name}: strided TP partitions are not supported")
                part_dim = int(param.partition_dim)
            out[name] = LocalShard(name, param, expert, part_dim, tp_rank, tp_size, group)
    return [out[name] for name in sorted(out)]


@dataclass(frozen=True)
class GlobalTable:
    """Every global trainable tensor: its full shape and the rank that sends it
    (the lowest rank holding it at TP rank 0; pipeline copies of a tied
    embedding hold equal values, so any holder will do)."""

    shapes: dict[str, tuple[int, ...]]
    owners: dict[str, int]

    @classmethod
    def build(cls, shards: Sequence[LocalShard], group: Any = None) -> GlobalTable:
        import torch.distributed as dist

        rows = [(s.name, s.full_shape(), s.tp_rank) for s in shards]
        world = int(dist.get_world_size()) if dist.is_initialized() else 1
        if world > 1:
            gathered: list[Any] = [None] * world
            dist.all_gather_object(gathered, rows, group=group)
        else:
            gathered = [rows]
        shapes: dict[str, tuple[int, ...]] = {}
        owners: dict[str, int] = {}
        for holder, held in enumerate(gathered):
            for name, shape, tp_rank in held:
                shape = tuple(int(d) for d in shape)
                if shapes.setdefault(name, shape) != shape:
                    raise SwiftMegatronSyncError(
                        f"{name}: ranks disagree on the full shape ({shapes[name]} vs {shape})")
                if int(tp_rank) == 0 and (name not in owners or holder < owners[name]):
                    owners[name] = holder
        missing = sorted(set(shapes) - set(owners))
        if missing:
            raise SwiftMegatronSyncError(f"no TP-rank-0 holder for {missing[:4]}")
        return cls(shapes, owners)

    def numel(self, name: str) -> int:
        n = 1
        for d in self.shapes[name]:
            n *= int(d)
        return n


# ----------------------------------------------------------------------------- transport
TRANSPORTS = ("cpu", "device")


class Transport:
    """Where the sync's collectives run.

    ``"cpu"`` (default): host buffers over gloo mirrors of the island's groups
    (world, TP, expert-TP and every DistributedOptimizer leaf's data-parallel
    group), so the sync allocates nothing on the accelerator: the FP32
    masters already live on the host under CPU offload, and the only device
    traffic is the in-place host-to-device copy into the bf16 parameters.
    ``"device"``: the island's own (HCCL/NCCL) groups with accelerator buffers
    of at most one chunk, for islands with HBM to spare.

    Mirrors are created collectively (every rank creates every group, in one
    sorted order).  A gloo island needs no mirror."""

    def __init__(self, kind: str, ctx: ParallelContext, optimizer: Any, device: Any) -> None:
        import torch
        import torch.distributed as dist

        if kind not in TRANSPORTS:
            raise ValueError(f"transport must be one of {TRANSPORTS}, got {kind!r}")
        self.kind = kind
        self.buffer_device = torch.device("cpu") if kind == "cpu" else device
        self.mirrors: dict[tuple[int, ...], Any] | None = None
        self.world_ranks = tuple(range(ctx.world))
        if kind == "cpu" and dist.get_backend() != "gloo":
            from yeto.rl.megatron_masters import _optimizer_leaves

            groups = [None, ctx.tp_group, ctx.etp_group]
            groups += [getattr(leaf, "data_parallel_group", None) for leaf in _optimizer_leaves(optimizer)]
            mine = sorted({self._ranks(group) for group in groups})
            gathered: list[Any] = [None] * ctx.world
            dist.all_gather_object(gathered, mine)
            self.mirrors = {}
            for ranks in sorted({tuple(r) for held in gathered for r in held}):
                group = dist.new_group(list(ranks), backend="gloo")
                if ctx.rank in ranks:
                    self.mirrors[ranks] = group

    def _ranks(self, group: Any) -> tuple[int, ...]:
        import torch.distributed as dist

        if group is None or group is dist.GroupMember.WORLD:
            return self.world_ranks
        return tuple(int(r) for r in dist.get_process_group_ranks(group))

    def group(self, group: Any) -> Any:
        """The group to run a collective of ``group`` on."""

        if self.mirrors is None:
            return group
        return self.mirrors[self._ranks(group)]

    @property
    def world(self) -> Any:
        return self.group(None)

    def dp_reduce(self) -> Callable[[Any, Any], None] | None:
        """``full_masters``' data-parallel reduce on this transport (None: its own)."""

        if self.kind == "device":
            return None
        import torch.distributed as dist

        def reduce(flat: Any, leaf: Any) -> None:
            group = self.group(getattr(leaf, "data_parallel_group", None))
            if len(self._ranks(getattr(leaf, "data_parallel_group", None))) > 1:
                dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)

        return reduce


# ----------------------------------------------------------------------------- helpers
def _torch_wire(code: int) -> Any:
    import torch

    from yeto.protocol import DTYPE_BF16, DTYPE_F32

    return {DTYPE_F32: torch.float32, DTYPE_BF16: torch.bfloat16}[code]


def _syncer_spec(spec: str) -> str:
    """``host:port[,host:port...]`` without ``tcp://`` schemes (the hook's
    ``YETO_DILOCO_SYNCER`` may carry one)."""

    return ",".join(item.strip().split("://", 1)[-1] for item in spec.split(","))


def _read_hmac_key(path: str | None) -> bytes | None:
    """The island HMAC key file, with the encoding of ``YETO_ISLAND_HMAC_KEY``."""

    if path is None:
        return None
    with open(path, encoding="utf-8") as handle:
        value = handle.read().strip()
    if not value:
        raise SwiftMegatronSyncError(f"HMAC key file {path} is empty")
    return value.encode()


def chunk_names(names: Sequence[str], numel: Callable[[str], int], budget: int) -> list[list[str]]:
    """``names`` in order, cut into runs of at most ``budget`` elements (a
    larger tensor is a run of its own).  Pure, so every rank cuts alike."""

    out: list[list[str]] = []
    run: list[str] = []
    total = 0
    for name in names:
        n = numel(name)
        if run and total + n > budget:
            out.append(run)
            run, total = [], 0
        run.append(name)
        total += n
    if run:
        out.append(run)
    return out


@dataclass
class _Plan:
    """Rank 0's decision for one boundary, broadcast to every rank."""

    applies: list[tuple[int, int]] = field(default_factory=list)  # (fragment, version)
    modes: dict[int, str] = field(default_factory=dict)  # fragment -> LIVE_* (else ordinary)
    finalizing: bool = False
    shutdown: bool = False
    send_init: bool = False  # initial-cut wait of a resumed learner 0: the syncer needs INIT
    error: str | None = None


# ----------------------------------------------------------------------------- the sync
class MegatronDiLoCoSync:
    """DiLoCo full-parameter sync of one ms-swift Megatron island (module doc).

    Construct it on every rank after the model, optimizer and resume checkpoint
    are loaded; it connects rank 0 to the syncer, and learner 0 seeds every
    fragment from its current FP32 masters (INIT).  Then call
    :meth:`after_optimizer_step` on every rank after every ``optimizer.step()``.

    ``model_chunks``: the model chunks (DDP-wrapped or bare; several with
    virtual pipeline parallelism).  ``optimizer``: the Megatron optimizer
    (DistributedOptimizer, ChainedOptimizer of dense + expert, precision-aware,
    CPU offload).  ``trainable_filter(local_name, param)``: which parameters
    take part (default ``requires_grad``).

    ``sync_interval_steps`` (H): a pull for a fragment is answered once the
    island has taken H optimizer steps since that fragment was last merged
    (H = 1 is the torch learner's self-clock: at least one step).  The steps
    are counted from the absolute ``iteration`` the trainer passes, so a
    resumed run keeps its rounds.

    ``merge_alpha``: local weight of a broadcast, as the torch learner's
    ``--merge-alpha`` (0 overwrites; the anchor is always the raw global).
    ``final_cut_payload``: what the syncer's ``--final-cut-payload`` sends;
    ``"full"`` overwrites the island with the final global, ``"manifest"``
    only acknowledges it (the final parameters stay with the syncer).
    """

    def __init__(self, model_chunks, optimizer, *, syncer: str, learner_id: int,
                 num_learners: int, fragments: int, sync_interval_steps: int,
                 wire_dtype: str = "bf16", hmac_key_path: str | None = None,
                 trainable_filter=None, merge_alpha: float = 0.0,
                 fragment_pattern: str = "binpack", wan_streams: int = 4,
                 final_cut_payload: str = "full", chunk_numel: int = DEFAULT_CHUNK_NUMEL,
                 transport: str = "cpu", parallel_context: ParallelContext | None = None,
                 client_factory=None, observer: Callable[[dict], None] | None = None,
                 initial_cut_timeout: float = 0.0, resume_dir: str | None = None,
                 resume_iteration: int | None = None, init_probe_s: float = 1800.0,
                 heartbeat_s: float = 30.0, heartbeat_stall_s: float = 0.0) -> None:
        import torch
        import torch.distributed as dist

        from yeto.diloco_sync import DiLoCoSyncState
        from yeto.fragments import build_layout
        from yeto.protocol import bulk_dtype, layout_fingerprint

        if not 0 <= merge_alpha < 1:
            raise ValueError("merge_alpha must be in [0, 1)")
        if sync_interval_steps < 1 or fragments < 1 or num_learners < 1:
            raise ValueError("fragments, num_learners and sync_interval_steps must be >= 1")
        if not 0 <= learner_id < num_learners:
            raise ValueError(f"learner_id {learner_id} outside [0, {num_learners})")
        if final_cut_payload not in FINAL_CUT_PAYLOADS:
            raise ValueError(f"final_cut_payload must be one of {FINAL_CUT_PAYLOADS}")
        if "," in _syncer_spec(syncer):
            # Checked on every rank, before rank 0 connects: the chunked INIT
            # and push streaming are not routed per shard owner.
            raise ValueError("the ms-swift Megatron sync talks to one syncer; sharded syncers are not supported")
        if not dist.is_initialized():
            raise SwiftMegatronSyncError("torch.distributed must be initialized (Megatron does this)")
        _ensure_visible_logs()
        self.ctx = parallel_context or ParallelContext.from_megatron()
        self.rank, self.world = self.ctx.rank, self.ctx.world
        self.chunks = list(model_chunks)
        self.modules = [unwrap_chunk(chunk) for chunk in self.chunks]
        self.optimizer = optimizer
        self.learner_id = int(learner_id)
        self.interval = int(sync_interval_steps)
        self.merge_alpha = float(merge_alpha)
        self.final_cut_payload = final_cut_payload
        self.chunk_numel = int(chunk_numel)
        self.wire = session_dtype(wire_dtype)
        self.bulk = bulk_dtype(self.wire)
        self.bulk_torch = _torch_wire(self.bulk)
        self.anchor_dtype = self.bulk_torch  # the raw global is exact in the broadcast dtype

        self.shards = local_shards(self.chunks, self.ctx, trainable_filter)
        self.by_name: dict[str, list[LocalShard]] = {}
        for shard in self.shards:
            self.by_name.setdefault(shard.name, []).append(shard)
        devices = {shard.param.device for shard in self.shards}
        self.device = next(iter(devices)) if devices else torch.device("cpu")
        if dist.get_backend() != "gloo" and self.device.type == "cpu":
            # A rank with no trainable shard still joins device collectives.
            self.device = _current_accelerator()
        self.comm = Transport(transport, self.ctx, optimizer, self.device)
        self.table = GlobalTable.build(self.shards, self.comm.world)
        # Rank 0 scatters each applied chunk as per-rank slices (see _apply_chunk),
        # so it needs every rank's (name, TP partition) list, in that rank's order.
        self.peer_shards: list[list[tuple[str, int, int, int]]] = [[] for _ in range(self.world)]
        mine = [(s.name, s.part_dim, s.tp_rank, s.local_shape[s.part_dim] if s.part_dim >= 0 else 0)
                for s in self.shards]
        if self.world > 1:
            gathered: list[Any] = [None] * self.world
            dist.all_gather_object(gathered, mine, group=self.comm.world)
            self.peer_shards = [list(rows) for rows in gathered]
        else:
            self.peer_shards = [mine]
        names = sorted(self.table.shapes)
        if not names:
            raise SwiftMegatronSyncError("no trainable parameter to synchronize")
        self.layout = build_layout(
            [(name, self.table.numel(name)) for name in names], int(fragments), fragment_pattern,
            avg_name_regex=OUTPUT_LAYER_AVG_REGEX,
            named_shapes={name: self.table.shapes[name] for name in names})
        self.num_fragments = self.layout.num_fragments
        self.fragment_chunks = [
            chunk_names([name for name, _ in frag.tensors], self.table.numel, self.chunk_numel)
            for frag in self.layout.fragments]
        self.contract = hashlib.sha256(
            CONTRACT_DOMAIN + layout_fingerprint(self.layout)
            + f"tp={self.ctx.tp_size};etp={self.ctx.etp_size}".encode()).digest()

        self.anchors: dict[int, Any] = {}  # id(param) -> raw global shard (host)
        self.anchored = [False] * self.num_fragments
        # (base version, round step) of each fragment's last sent PUSH: a
        # decoupled syncer refuses a second delta from a spent anchor for
        # another round (late carry, ``anchor_reused``), so such a pull waits
        # for a newer broadcast instead of a gather that would be refused.  A
        # restarted attempt of the same round may reuse it.
        self.spent: list[tuple[int, int] | None] = [None] * self.num_fragments
        # The syncer link (connection generation) each spent PUSH went out on:
        # once the link is replaced, an undelivered delta is gone with the old
        # connection and a restarted syncer lost the rounds it was carried to,
        # so the guard is released (``_release_spent_after_reconnect``).
        self.spent_link: list[Any] = [None] * self.num_fragments
        # The anchor version of each fragment's last PUSH that may have been
        # delivered (a drop clears it; a redial does not): a newer catch-up
        # of that fragment may already hold the delta (``live_install_mode``).
        self.pushed_from: list[int | None] = [None] * self.num_fragments
        # The torch learner's per-fragment counters and versions, kept alike on
        # every rank (anchors live in ``self.anchors``, sharded, not here).
        self.state = DiLoCoSyncState.create(self.num_fragments, track_anchors=False)
        self._counters_started = False
        self.units_total = 0
        self.iteration = 0
        self.observer = observer
        self._updates: dict[int, Any] = {}  # rank 0: broadcasts drained for the current plan
        self._export_error: str | None = None  # rank 0: a non-finite value in the last export
        self.finished = False  # the final cut is installed and acknowledged
        self._closed = False
        self.client = None
        self.stats = {"applied": 0, "pushed": 0, "push_bytes": 0, "dropped_pushes": 0,
                      "deferred_pushes": 0, "apply_s": 0.0, "export_s": 0.0}
        self._observed_push_bytes = 0  # rank 0: push_bytes at the last observer report
        self.sender: _PushSender | None = None  # rank 0: background PUSH streaming
        # Rank 0: HEARTBEAT every ``heartbeat_s`` from a thread, so the syncer
        # can lease the island (``--island-lease-s``) across ~10 min steps,
        # INIT and checkpoint saves; ``_progress_at`` is the last sign of life
        # of the training loop (``heartbeat_stall_s``: none for that long and
        # the heartbeats stop, so a hung island loses its lease).
        self.heartbeat: _Heartbeat | None = None
        self._progress_at = time.monotonic()
        # (iteration, "apply" | "push", fragment, version | pull step), same on every rank.
        self.events: deque[tuple[int, str, int, int]] = deque(maxlen=4096)
        # Resume (``resume_dir``): the record ``save_state`` wrote with the
        # checkpoint the trainer resumed from, the old anchor of every local
        # shard when it was saved too, and how each fragment's first broadcast
        # after the restart was installed (fid -> mode, same on every rank).
        self.resume_record: dict | None = None
        self._resume_anchors: dict[int, Any] = {}  # id(param) -> old anchor (host)
        self._resume_has_anchors = False
        self.resumed: dict[int, str] = {}
        if resume_dir:
            self._load_resume(resume_dir, resume_iteration)

        if self.rank == 0:
            log.info("swift-megatron sync: %d trainable tensors (%.2f B elements) -> %d fragments",
                     len(names), sum(self.table.numel(n) for n in names) / 1e9, self.num_fragments)
        error = None
        if self.rank == 0:
            try:
                self.client = self._connect(syncer, hmac_key_path, wan_streams, client_factory)
                self.sender = _PushSender(self.client, self.stats, self._push_dropped, self._push_sent)
                if heartbeat_s > 0:
                    self.heartbeat = _Heartbeat(self.client, float(heartbeat_s), float(heartbeat_stall_s),
                                                lambda: self.iteration, lambda: self._progress_at)
            except Exception as exc:  # raised on every rank below
                error = f"{type(exc).__name__}: {exc}"
        self._raise_if(self._bcast_object(error))
        # A resumed learner 0 (it holds a record, so the syncer of this session
        # was initialized) first waits for the syncer's catch-up and sends
        # INIT_PARAMS (~4 bytes per parameter, 480 GB for 120B) only if nothing
        # arrives within ``init_probe_s``: an initialized syncer ignores INIT,
        # and receiving the catch-up while gathering INIT doubled rank 0's RSS.
        defer_init = self.learner_id == 0 and self.resume_record is not None and initial_cut_timeout > 0
        if self.learner_id == 0 and not defer_init:
            self._send_init()
        if initial_cut_timeout > 0:
            self._install_initial_cut(float(initial_cut_timeout),
                                      init_probe=float(init_probe_s) if defer_init else None)

    # ------------------------------------------------------------------ setup
    def _connect(self, syncer: str, hmac_key_path: str | None, wan_streams: int, client_factory):
        from yeto.diloco_sync import connect_syncer

        # A syncer resumed from an older checkpoint rewinds; its catch-up is
        # taken (and the island rebased onto it) instead of being dropped.
        kwargs = {"session_contract_hash": self.contract, "accept_rewound_catch_up": True}
        key = _read_hmac_key(hmac_key_path)
        if key is not None:
            kwargs["hmac_key"] = key
        connect = client_factory or connect_syncer
        client = connect(_syncer_spec(syncer), self.learner_id, self.layout, self.wire, wan_streams, **kwargs)
        log.info("connected to syncer at %s as learner %d", syncer, self.learner_id)
        return client

    def _send_init(self) -> None:
        """INIT_PARAMS of every fragment from this island's current masters.

        Streamed chunk by chunk: rank 0 hands each exported chunk to a
        background sender, which writes it into the fragment's INIT_PARAMS
        frame and drops it, while every rank gathers the next chunk.  At most
        a few chunks (``chunk_numel`` elements each) are on rank 0's host
        instead of whole fragments."""

        import queue
        import threading

        end = object()
        jobs: queue.Queue = queue.Queue(maxsize=1)
        failure: list[str] = []
        sender = None
        if self.rank == 0:
            def chunks_of(pending: queue.Queue, ended: list):
                while (item := pending.get()) is not end:
                    if isinstance(item, BaseException):
                        raise item
                    yield from item
                ended.append(True)

            def send() -> None:
                while (job := jobs.get()) is not None:
                    fid, pending = job
                    ended: list = []
                    try:
                        if not failure and self.client.send_init_parts(fid, chunks_of(pending, ended)) is False:
                            failure.append(f"INIT fragment {fid}: syncer link down")
                    except Exception as exc:
                        failure.append(f"INIT fragment {fid}: {type(exc).__name__}: {exc}")
                    # Drain what was not sent: the gathers go on until the error is broadcast.
                    while not ended and pending.get() is not end:
                        pass
                    del job, pending

            sender = threading.Thread(target=send, name="yeto-swift-init", daemon=True)
            sender.start()
        error = None
        try:
            for fid in range(self.num_fragments):
                pending = queue.Queue(maxsize=2) if self.rank == 0 else None
                if pending is not None:
                    jobs.put((fid, pending))
                for parts in self._export_fragment_chunks(fid, delta=False):
                    self._progress_at = time.monotonic()
                    if pending is None:
                        continue
                    if error is None and self._export_error is not None:
                        error = f"INIT fragment {fid}: {self._export_error}"
                        failure.append(error)  # the sender skips the fragments after it
                        pending.put(SwiftMegatronSyncError(error))
                    elif error is None:
                        pending.put(parts)
                    del parts
                if pending is not None:
                    pending.put(end)
                    if error is None and failure:
                        error = failure[0]
        finally:
            if sender is not None:
                jobs.put(None)
                sender.join()
        if self.rank == 0 and error is None and failure:
            error = failure[0]
        self._raise_if(self._bcast_object(error))
        if self.rank == 0:
            log.info("sent INIT_PARAMS for %d fragments", self.num_fragments)

    def _install_initial_cut(self, timeout: float, *, init_probe: float | None = None) -> None:
        """Apply the syncer's first broadcast of every fragment as it arrives,
        before training starts (collective: every rank loops together).

        Without this the whole initial cut (``4 * N`` bytes on an f32-bulk
        session, ~480 GB for 120B) queues in rank 0's client until the
        first boundary drains it, one optimizer step after INIT. Here each
        fragment is installed and freed as it lands. Gives up after
        ``timeout`` seconds (the rest is applied at the boundaries as
        before), and stops early at finalization or shutdown."""

        import time

        started = time.monotonic()
        init_pending = init_probe is not None  # resumed learner 0: INIT only if the syncer needs it
        while not all(self.anchored):
            self._progress_at = time.monotonic()
            plan = self._plan() if self.rank == 0 else None
            if self.rank == 0:
                waited = time.monotonic() - started
                if init_pending and not plan.applies and waited > init_probe:
                    plan.send_init = True
                elif not init_pending and waited > timeout and not plan.applies:
                    plan.shutdown = True  # only ends the wait; the client's flag is unchanged
            plan = self._bcast_object(plan)
            self._raise_if(plan.error)
            if init_pending and (plan.applies or plan.send_init):
                init_pending = False
                if plan.send_init:
                    if self.rank == 0:
                        log.warning("no catch-up from the syncer within %.0f s: sending INIT_PARAMS", init_probe)
                    self._send_init()
                elif self.rank == 0:
                    log.warning("the syncer is initialized (catch-up arriving): INIT_PARAMS skipped")
                started = time.monotonic()
            updates = self._take_updates(plan) if self.rank == 0 else {}
            for fid, version in plan.applies:
                data = updates.pop(fid) if self.rank == 0 else None
                live = plan.modes.get(fid)
                if live == LIVE_CONFIRM:
                    continue
                self._apply_first_or_blend(fid, data, version, live=live)
                del data
                self.state.fragment_versions[fid] = version
                self.state.global_step = max(self.state.global_step, version)
                self.anchored[fid] = True
                self.stats["applied"] += 1
                self.events.append((0, "apply", fid, version))
            if plan.finalizing or plan.shutdown:
                break
            if not plan.applies:
                time.sleep(0.2)
        if init_pending:  # the wait ended before the probe did: keep the old order's guarantee
            self._send_init()
        if self.rank == 0:
            log.warning("initial cut: %d of %d fragments installed before training in %.0f s",
                        sum(self.anchored), self.num_fragments, time.monotonic() - started)

    # ------------------------------------------------------------------ resume
    def _apply_first_or_blend(self, fid: int, data: Any, version: int, *, live: str | None = None) -> str | None:
        """Apply a broadcast; the first one of a fragment after a resume by
        ``resume_fragment_mode``, an anchored one by ``live`` (``_plan``:
        ``LIVE_REBASE`` or the ordinary install).  Returns the resume mode
        (None: not a resumed first install) and records it in
        ``self.resumed``."""

        mode = None
        if live == LIVE_REBASE and self.anchored[fid]:
            if self.rank == 0:
                log.info("fragment %d: catch-up version %d onto anchor %d -> rebase (local progress kept)",
                         fid, version, self.state.fragment_versions[fid])
            self._apply_fragment(fid, data, self.bulk_torch, blend=False, mode=LIVE_REBASE)
            self.stats["rebased"] = self.stats.get("rebased", 0) + 1
            return None
        if self.resume_record is not None and not self.anchored[fid] and fid not in self.resumed:
            from yeto.diloco_sync import resume_fragment_mode

            mode = resume_fragment_mode(self.resume_record, fid, version, have_anchor=self._resume_has_anchors)
            self.resumed[fid] = mode
            if self.rank == 0:
                log.info("resume: fragment %d version %d (checkpoint anchor %d) -> %s", fid, version,
                         self.resume_record["versions"][fid], mode)
        self._apply_fragment(fid, data, self.bulk_torch, blend=True, mode=mode)
        if mode is not None and len(self.resumed) == self.num_fragments:
            self._resume_anchors.clear()  # every old anchor is used or no longer needed
            if self.rank == 0:
                counts = {m: list(self.resumed.values()).count(m) for m in (RESUME_ANCHOR, RESUME_SHIFT, RESUME_OVERWRITE)}
                log.warning("resume: all %d fragments installed: anchor %d, shift %d, overwrite %d",
                            self.num_fragments, counts[RESUME_ANCHOR], counts[RESUME_SHIFT], counts[RESUME_OVERWRITE])
        return mode

    def _restore_counters(self, fid: int, mode: str) -> None:
        """A fragment whose checkpointed progress was kept (anchor / shift)
        keeps its H-gate and token counters, and an unchanged anchor its
        spent-anchor guard; an overwritten one starts afresh."""

        if mode == RESUME_OVERWRITE:
            return
        self.state.restore_fragment(self.resume_record, fid)
        if mode == RESUME_ANCHOR:
            spent = self.resume_record["spent"][fid]
            self.spent[fid] = None if spent is None else (int(spent[0]), int(spent[1]))
            # Pushed by the previous process, on a link that is gone: released
            # at the first boundary like any spent anchor after a reconnect.
            self.spent_link[fid] = None if spent is None else _PREVIOUS_PROCESS
            self.pushed_from[fid] = None if spent is None else int(spent[0])

    def _resume_paths(self, resume_dir: str) -> tuple[str, str]:
        return (os.path.join(resume_dir, RESUME_STATE_FILE),
                os.path.join(resume_dir, RESUME_ANCHOR_DIR, f"rank{self.rank:05d}.pt"))

    def _shard_keys(self) -> list[tuple[str, int]]:
        return [(s.name, s.tp_rank) for s in self.shards]

    def _load_resume(self, resume_dir: str, resume_iteration: int | None) -> None:
        """Read the record ``save_state`` left in ``resume_dir`` (collective).

        A missing or foreign record (another learner, layout, parallel shape
        or iteration) is a cold start: every fragment takes the global as
        before.  Old anchors are used only if every rank has its file."""

        import torch

        record, reason = None, None
        if self.rank == 0:
            path = self._resume_paths(resume_dir)[0]
            try:
                with open(path, encoding="utf-8") as handle:
                    record = json.load(handle)
            except FileNotFoundError:
                reason = f"no {RESUME_STATE_FILE} in {resume_dir}"
            except (OSError, ValueError) as exc:
                reason = f"unreadable {path}: {exc}"
            if record is not None:
                checks = [
                    (record.get("format") == RESUME_FORMAT, "format"),
                    (record.get("learner_id") == self.learner_id, "learner id"),
                    # A hand-written bridge record (an island checkpoint from before
                    # these records) may leave the contract out.
                    (record.get("contract", self.contract.hex()) == self.contract.hex(),
                     "layout / parallel shape"),
                    (len(record.get("versions", ())) == self.num_fragments, "fragment count"),
                    (resume_iteration is None or record.get("iteration") == int(resume_iteration), "iteration"),
                ]
                bad = [what for ok, what in checks if not ok]
                if bad:
                    reason = f"{path} does not match this island ({', '.join(bad)})"
                    record = None
            if record is None:
                log.warning("resume: %s; the syncer's cut replaces every fragment (cold start)", reason)
        record = self._bcast_object(record)
        if record is None:
            return
        loaded: dict[int, Any] = {}
        ok = bool(record.get("anchors"))
        if ok:
            try:
                saved = torch.load(self._resume_paths(resume_dir)[1], map_location="cpu", weights_only=True)
                if saved["keys"] != [list(k) for k in self._shard_keys()]:
                    raise ValueError("shard list differs")
                for shard, value in zip(self.shards, saved["values"], strict=True):
                    if tuple(value.shape) != tuple(shard.local_shape):
                        raise ValueError(f"{shard.name} has shape {tuple(value.shape)}")
                    loaded[id(shard.param)] = value
            except Exception as exc:  # agreed on below; a missing anchor file is not fatal
                log.warning("resume: rank %d cannot use its saved anchors: %s", self.rank, exc)
                ok = False
        ok = all(self._gather_object(ok))
        self.resume_record = record
        self._resume_has_anchors = ok
        self._resume_anchors = loaded if ok else {}
        self.units_total = int(record["units_total"])
        self.state.global_step = int(record["global_step"])
        if self.rank == 0:
            log.info("resume: island record of iteration %s (anchor versions %s..%s, old anchors %s)",
                     record.get("iteration"), min(record["versions"]), max(record["versions"]),
                     "loaded" if ok else "not saved: a newer fragment takes the global")

    def _gather_object(self, value: Any) -> list[Any]:
        import torch.distributed as dist

        if self.world == 1:
            return [value]
        out: list[Any] = [None] * self.world
        dist.all_gather_object(out, value, group=self.comm.world)
        return out

    def save_state(self, output_dir: str, *, iteration: int, save_anchors: bool = False) -> bool:
        """Record the sync's per-fragment state next to the trainer checkpoint
        ``output_dir`` of ``iteration`` (call it on every rank after the
        checkpoint is written; collective only with ``save_anchors``).

        Global rank 0 writes ``RESUME_STATE_FILE``: each fragment's anchor
        version, the H-gate / token counters and the spent-anchor guard.
        ``save_anchors`` also writes every rank's anchor shards (the
        broadcast dtype, ~``2 * n_r`` bytes per rank for bf16) to
        ``RESUME_ANCHOR_DIR``; only with them can a fragment that the syncer
        moved past the checkpoint's anchor keep the island's local progress
        (``global_new + (local - anchor_old)``).  Never raises on an I/O
        error (training goes on); returns whether rank 0 wrote the record
        (True on the other ranks)."""

        import torch

        anchors_ok = False
        if save_anchors:
            path = self._resume_paths(output_dir)[1]
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                values = [self.anchors.get(id(s.param)) for s in self.shards]
                if any(v is None for v in values):
                    raise ValueError("a local shard has no anchor yet")
                tmp = f"{path}.tmp"
                torch.save({"keys": [list(k) for k in self._shard_keys()], "values": values}, tmp)
                os.replace(tmp, path)
                anchors_ok = True
            except Exception as exc:
                log.warning("save_state: rank %d cannot write its anchors: %s", self.rank, exc)
        anchors_ok = all(self._gather_object(anchors_ok)) if save_anchors else False
        written = True
        if self.rank == 0:
            record = {"format": RESUME_FORMAT, "learner_id": self.learner_id, "iteration": int(iteration),
                      "contract": self.contract.hex(), "anchors": anchors_ok,
                      **self.state.resume_record(anchored=self.anchored, units_total=self.units_total,
                                                 spent=self.spent)}
            path = self._resume_paths(output_dir)[0]
            try:
                os.makedirs(output_dir, exist_ok=True)
                tmp = f"{path}.tmp"
                with open(tmp, "w", encoding="utf-8") as handle:
                    json.dump(record, handle)
                os.replace(tmp, path)
                log.info("save_state: %s (iteration %d, anchors %s)", path, iteration, anchors_ok)
            except OSError as exc:
                log.warning("save_state: cannot write %s: %s", path, exc)
                written = False
        return written

    # ------------------------------------------------------------------ collectives
    def _bcast_object(self, value: Any) -> Any:
        import torch.distributed as dist

        if self.world == 1:
            return value
        box = [value]
        dist.broadcast_object_list(box, src=0, group=self.comm.world)
        return box[0]

    @staticmethod
    def _raise_if(error: str | None) -> None:
        if error:
            raise SwiftMegatronSyncError(f"syncer side failed on rank 0: {error}")

    def _masters(self, shards: Sequence[LocalShard]) -> list[Any]:
        from yeto.rl.megatron_masters import full_masters

        return full_masters(self.optimizer, [s.param for s in shards], device=self.comm.buffer_device,
                            reduce=self.comm.dp_reduce())

    def _export_chunk(self, names: Sequence[str], *, delta: bool) -> list[Any] | None:
        """Full FP32 values (``delta``: minus the anchor) of ``names``; the CPU
        tensors on rank 0, None elsewhere.  Collective (module doc)."""

        import torch
        import torch.distributed as dist

        wanted = set(names)
        local = [s for s in self.shards if s.name in wanted]
        masters = self._masters(local)
        values: dict[int, Any] = {}
        for i, (shard, master) in enumerate(zip(local, masters, strict=True)):
            value = master.detach().to(dtype=torch.float32)
            if delta:
                anchor = self.anchors.get(id(shard.param))
                if anchor is None:
                    raise SwiftMegatronSyncError(f"{shard.name} has no anchor yet")
                value = value - anchor.to(device=value.device, dtype=torch.float32)
            values[i] = value
        # TP gathers: dense first, then expert, each in name order, so no two
        # ranks wait on two groups in opposite orders.
        full: dict[str, Any] = {}
        for expert in (False, True):
            for i, shard in enumerate(local):
                if shard.expert is not expert:
                    continue
                value = values[i]
                if shard.part_dim >= 0:
                    parts = [torch.empty_like(value) for _ in range(shard.tp_size)]
                    dist.all_gather(parts, value.contiguous(), group=self.comm.group(shard.tp_group))
                    value = torch.cat(parts, dim=shard.part_dim) if shard.tp_rank == 0 else None
                    del parts
                if value is not None and self.table.owners[shard.name] == self.rank:
                    full.setdefault(shard.name, value)
        del values, masters
        out: list[Any] | None = [] if self.rank == 0 else None
        for name in names:
            owner = self.table.owners[name]
            if self.rank == 0:
                if owner == 0:
                    tensor = full.pop(name)
                else:
                    tensor = torch.empty(self.table.shapes[name], dtype=torch.float32,
                                         device=self.comm.buffer_device)
                    dist.recv(tensor, src=owner, group=self.comm.world)
                out.append(tensor.detach().to("cpu"))
            elif owner == self.rank:
                dist.send(full.pop(name).contiguous(), dst=0, group=self.comm.world)
        return out

    def _export_fragment(self, fid: int, *, delta: bool) -> list[Any] | None:
        """The fragment's values on rank 0 as wire byte parts (INIT, or PUSH
        for bf16/f32 sessions) or, for a q4 push, one f32 tensor."""

        parts: list[Any] | None = [] if self.rank == 0 else None
        for chunk in self._export_fragment_chunks(fid, delta=delta):
            if parts is not None:
                parts.extend(chunk)
        return parts

    def _export_fragment_chunks(self, fid: int, *, delta: bool):
        """``_export_fragment`` one chunk at a time (collective): yields each
        chunk's parts on rank 0, None elsewhere; ``_export_error`` is set once
        a non-finite value was seen."""

        from yeto.protocol import DTYPE_Q4
        from yeto.tensor_io import pack_tensor

        self._export_error = None
        q4 = delta and self.wire == DTYPE_Q4
        for names in self.fragment_chunks[fid]:
            tensors = self._export_chunk(names, delta=delta)
            if tensors is None:
                yield None
                continue
            parts = []
            for tensor in tensors:
                if self._export_error is None and not bool(tensor.isfinite().all()):
                    # Raised on every rank by the caller's error broadcast.
                    self._export_error = f"fragment {fid} holds a non-finite value"
                parts.append(tensor.reshape(-1) if q4 else
                             pack_tensor(tensor.reshape(-1), self.bulk if not delta else self._push_part_dtype()))
            del tensors
            yield parts

    def _push_part_dtype(self) -> int:
        """The dtype of the PUSH parts handed to the client: f32 on the
        f32-bulk sessions (the client encodes bf16 / int8 / int4 itself, with
        tensor blocks), else the session's push wire dtype."""

        from yeto.protocol import DTYPE_F32, bulk_dtype, push_wire_dtype

        return DTYPE_F32 if bulk_dtype(self.wire) == DTYPE_F32 else push_wire_dtype(self.wire)

    def _apply_chunk(self, names: Sequence[str], flat: Any | None, dtype: Any, *, blend: bool,
                     mode: str | None = None) -> None:
        """Install the raw global values of ``names`` (``flat`` on rank 0, in the
        broadcast dtype) as every rank's anchor, master and model copy.

        A resumed island's first install of a fragment (``mode``,
        ``yeto.diloco_sync.resume_fragment_mode``): ``"anchor"`` only records
        the anchor and leaves the master and model copy untouched,
        ``"shift"`` writes ``global + (master - old anchor)``; ``"overwrite"``
        (or None) is the ordinary install."""

        import torch
        import torch.distributed as dist

        from yeto.rl.megatron_masters import has_complete_master, write_masters

        wanted = set(names)
        offsets, offset = {}, 0
        for name in names:
            offsets[name] = offset
            offset += self.table.numel(name)

        def piece(buffer, name, part_dim, tp_rank, size):
            whole = buffer[offsets[name]:offsets[name] + self.table.numel(name)].view(self.table.shapes[name])
            if part_dim >= 0:
                whole = whole.narrow(part_dim, tp_rank * size, size)
            return whole

        # Point-to-point scatter instead of a world broadcast: every rank gets
        # only its own TP slices of this chunk (about DP copies of the chunk
        # cross the wire in total, not world copies).
        local = [s for s in self.shards if s.name in wanted]
        pieces = []
        if self.rank == 0:
            buffer = flat.to(device=self.comm.buffer_device)
            pending = []
            for peer in range(1, self.world):
                rows = [row for row in self.peer_shards[peer] if row[0] in wanted]
                if not rows:
                    continue
                out = torch.cat([piece(buffer, *row).reshape(-1) for row in rows])
                pending.append((dist.isend(out, dst=peer, group=self.comm.world), out))
            for shard in local:
                pieces.append(piece(buffer, shard.name, shard.part_dim, shard.tp_rank,
                                    shard.local_shape[shard.part_dim] if shard.part_dim >= 0 else 0))
            for work, _ in pending:
                work.wait()
            del pending
        elif local:
            total = sum(int(s.param.numel()) for s in local)
            incoming = torch.empty(total, dtype=dtype, device=self.comm.buffer_device)
            dist.recv(incoming, src=0, group=self.comm.world)
            at = 0
            for shard in local:
                n = int(shard.param.numel())
                pieces.append(incoming[at:at + n].view(shard.local_shape))
                at += n
        targets, previous = [], []
        for shard, whole in zip(local, pieces, strict=True):
            raw = whole.to(dtype=torch.float32).contiguous()
            previous.append(self.anchors.get(id(shard.param)))
            self.anchors[id(shard.param)] = whole.to(device="cpu", dtype=self.anchor_dtype).contiguous().clone()
            targets.append(raw)
        if mode == RESUME_ANCHOR:
            return  # the broadcast is the checkpoint's own anchor: keep the local values
        if mode == LIVE_REBASE:
            # The island's progress since its anchor moves onto the catch-up;
            # a shard whose global did not change keeps its values bit for bit.
            from yeto.diloco_sync import shifted_resume_value

            moved = [i for i, (s, old) in enumerate(zip(local, previous, strict=True))
                     if old is None or not torch.equal(old, self.anchors[id(s.param)])]
            if not moved:
                return
            local = [local[i] for i in moved]
            masters = self._masters(local)
            targets = [targets[i] if previous[i] is None else shifted_resume_value(targets[i], m, previous[i])
                       for i, m in zip(moved, masters, strict=True)]
            del masters
        elif mode == RESUME_SHIFT:
            from yeto.diloco_sync import shifted_resume_value

            masters = self._masters(local)
            targets = [shifted_resume_value(t, m, self._resume_anchors.pop(id(s.param)))
                       for s, m, t in zip(local, masters, targets, strict=True)]
            del masters
        elif blend and self.merge_alpha > 0:
            masters = self._masters(local)
            targets = [self.merge_alpha * m.to(torch.float32) + (1.0 - self.merge_alpha) * t
                       for m, t in zip(masters, targets, strict=True)]
            del masters
        params = [s.param for s in local]
        with torch.no_grad():
            write_masters(self.optimizer, params, targets)
            for param, target in zip(params, targets, strict=True):
                if has_complete_master(param) and param.dtype != torch.float32:
                    param.data.copy_(target.to(dtype=param.dtype))

    def _check_payload(self, fid: int, data: Any, dtype: Any) -> None:
        """Rank 0: refuse a fragment payload of the wrong size (or none)."""

        import torch

        if data is None:
            raise SwiftMegatronSyncError(f"fragment {fid} has no payload")
        nbytes = len(memoryview(data).cast("B"))
        expected = self.layout.fragments[fid].numel * torch.empty((), dtype=dtype).element_size()
        if nbytes != expected:
            raise SwiftMegatronSyncError(f"fragment {fid} payload has {nbytes} bytes, expected {expected}")

    def _apply_fragment(self, fid: int, data: Any, dtype: Any, *, blend: bool,
                        mode: str | None = None) -> None:
        """Apply one fragment's payload (``data`` on rank 0) chunk by chunk
        (``mode``: a resumed island's first install, ``_apply_chunk``)."""

        import torch

        view = memoryview(data).cast("B") if self.rank == 0 else None
        itemsize = torch.empty((), dtype=dtype).element_size()
        offset = 0
        for names in self.fragment_chunks[fid]:
            nbytes = sum(self.table.numel(name) for name in names) * itemsize
            flat = None
            if view is not None:
                flat = torch.frombuffer(bytearray(view[offset:offset + nbytes]), dtype=torch.uint8).view(dtype)
            self._apply_chunk(names, flat, dtype, blend=blend, mode=mode)
            offset += nbytes

    # ------------------------------------------------------------------ boundary
    def after_optimizer_step(self, iteration: int, target_tokens: int, *,
                             update_successful: bool = True) -> None:
        """Run one non-blocking sync boundary.  Every rank calls it after every
        ``optimizer.step()``, with the same arguments.

        ``iteration``: the trainer's absolute 1-based iteration (continues on
        resume); the H-step gate counts it.  ``target_tokens``: the loss
        tokens of this step over the whole island (the PUSH ``c_tokens``
        weight).  A skipped step (loss-scale overflow / NaN,
        ``update_successful=False``, ``target_tokens`` 0) still counts as an
        inner step, so rounds stay aligned with ``iteration``, but adds no
        token weight; its parameters did not move, so the next push carries
        the same delta it would have carried anyway."""

        from yeto.diloco_sync import boundary_metrics

        if self.finished or self._closed:
            return
        self._progress_at = time.monotonic()
        state = self.state
        iteration = int(iteration)
        if not self._counters_started:  # count from the (possibly resumed) first step
            state.steps_at_reset = [iteration - 1] * self.num_fragments
            # A fragment that kept its checkpointed progress keeps its clock:
            # the H gate is not restarted by the island's own restart.
            for fid, mode in self.resumed.items():
                self._restore_counters(fid, mode)
            self._counters_started = True
        self.iteration = iteration
        if update_successful:
            self.units_total += max(0, int(target_tokens))
        plan = self._plan() if self.rank == 0 else None
        plan = self._bcast_object(plan)
        self._raise_if(plan.error)
        if plan.finalizing:
            self._finalize()
            return
        state.shutdown = plan.shutdown
        # Apply broadcasts before answering pulls (as sync_diloco_boundary):
        # a next-round pull can overtake the previous broadcast.
        updates = self._take_updates(plan) if self.rank == 0 else {}
        for fid, version in plan.applies:
            data = updates.get(fid) if self.rank == 0 else None
            live = plan.modes.get(fid)
            if live == LIVE_CONFIRM:  # a replay of the anchor: nothing changes
                self.stats["confirmed"] = self.stats.get("confirmed", 0) + 1
                self.events.append((iteration, "confirm", fid, version))
                continue
            started = time.perf_counter()
            mode = self._apply_first_or_blend(fid, data, version, live=live)
            self.stats["apply_s"] += time.perf_counter() - started
            if live != LIVE_REBASE:  # a rebased fragment keeps its local progress and clock
                state.steps_at_reset[fid] = iteration
                state.units_at_reset[fid] = self.units_total
            if mode is not None:
                self._restore_counters(fid, mode)
            state.fragment_versions[fid] = version
            self.anchored[fid] = True
            state.global_step = max(state.global_step, version)
            self.stats["applied"] += 1
            self.events.append((iteration, "apply", fid, version))
        pushes = self._ready_pushes() if self.rank == 0 else None
        pushes = self._bcast_object(pushes)
        if isinstance(pushes, str):
            self._raise_if(pushes)
        error = None
        for fid, step, attempt, c_steps, c_units in pushes:
            started = time.perf_counter()
            parts = self._export_fragment(fid, delta=True)
            self.stats["export_s"] += time.perf_counter() - started
            self.events.append((iteration, "push", fid, step))
            if self.rank == 0 and error is None:
                error = self._push(fid, step, attempt, c_steps, c_units, parts)
        if pushes:
            self._raise_if(self._bcast_object(error))
        if self.observer is not None and self.rank == 0 and (plan.applies or pushes):
            # push_bytes: what the sender finished since the last report (the
            # torch learner reports its boundary's bytes, not a running total)
            sent, self._observed_push_bytes = (self.stats["push_bytes"] - self._observed_push_bytes,
                                               self.stats["push_bytes"])
            metrics = boundary_metrics(state, merges=len(plan.applies), pushes=len(pushes),
                                       push_bytes=sent, push_delta_norm=0.0, steps_total=iteration)
            metrics.pop("sync/push_delta_norm", None)  # not computed: the delta is never materialized whole
            self.observer(metrics)

    @property
    def shutdown(self) -> bool:
        return self.state.shutdown

    @property
    def global_step(self) -> int:
        return self.state.global_step

    def _plan(self) -> _Plan:
        """Rank 0: drain the client's broadcasts and flags (never blocks)."""

        plan = _Plan()
        client = self.client
        try:
            client.check_health()
            if self.sender is not None:
                self.sender.raise_error()
            plan.finalizing = bool(client.finalizing.is_set())
            if plan.finalizing:
                return plan
            self._updates = {}
            catch_up: set[int] = set()
            for update in client.drain_updates():
                # A later broadcast of the same fragment supersedes an earlier
                # one; a catch-up among them is still the syncer's current value.
                self._updates[update.fragment_id] = update
                if getattr(update, "catch_up", False):
                    catch_up.add(update.fragment_id)
            for fid, update in self._updates.items():
                # Checked here, before the plan goes out: a bad payload found
                # while applying would fail rank 0 inside the chunk broadcasts.
                self._check_payload(fid, update.data, self.bulk_torch)
            plan.applies = sorted((fid, int(u.version)) for fid, u in self._updates.items())
            plan.modes = self._live_modes(plan.applies, catch_up)
            plan.shutdown = bool(client.shutdown.is_set())
        except Exception as exc:
            plan.error = f"{type(exc).__name__}: {exc}"
        return plan

    def _live_modes(self, applies: Sequence[tuple[int, int]], catch_up: set[int]) -> dict[int, str]:
        from yeto.diloco_sync import live_install_mode

        modes = {}
        for fid, version in applies:
            mode = live_install_mode(anchored=self.anchored[fid], current_version=self.state.fragment_versions[fid],
                                     version=version, catch_up=fid in catch_up, pushed_from=self.pushed_from[fid])
            if mode is not None:
                modes[fid] = mode
        return modes

    def _ready_pushes(self) -> list[tuple[int, int, int, int, int]] | str:
        """Rank 0: the pulls answered now, by the torch learner's rule
        (``take_ready_pulls``) with the H-step gate; one push per fragment
        per boundary, a second pull of it waits."""

        from yeto.diloco_sync import take_ready_pulls

        state = self.state
        try:
            state.pending_pulls.extend(self.client.drain_pulls())
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
        if self.sender is not None and self.sender.busy():
            # Decoupled: the previous PUSH is still streaming (slow link or
            # syncer); every pull waits rather than the training step.
            if state.pending_pulls:
                self.stats["deferred_pushes"] += 1
            return []
        self._release_spent_after_reconnect()
        ready = take_ready_pulls(state, steps_total=self.iteration, units_total=self.units_total,
                                 anchored=self.anchored.__getitem__, min_steps=self.interval)
        # One fragment per boundary: its PUSH streams in the background while
        # training goes on, and the gather of a second would only queue
        # behind it on the host.
        out = []
        for pull, c_steps, c_units in ready:
            if out or not self._pushable(pull):
                state.pending_pulls.append(pull)
                continue
            out.append((int(pull.fragment_id), int(pull.global_step), int(pull.round_attempt), c_steps, c_units))
        return out

    def link_of(self, fid: int) -> Any:
        """Rank 0: the connection generation fragment ``fid`` is pushed on
        (None for a client that does not expose it)."""
        client = self.client
        of = getattr(client, "connection_generation_of", None)
        if of is not None:
            return of(fid)
        return getattr(client, "connection_generation", None)

    def _release_spent_after_reconnect(self) -> None:
        """Rank 0, sender idle: free every spent anchor whose PUSH went out
        on a connection that has since been replaced.

        Without this an island that pushed just before a syncer restart (or
        whose PUSH was carried into a round the restarted syncer no longer
        has) refuses every pull of that fragment until a newer broadcast,
        which no syncer sends once every island waits like this.  Answering
        from the same anchor again cannot double-count it: a syncer that
        still holds the delta refuses the second one per island
        (``anchor_reused``, syncer ``LateCarry``), and one that lost it
        merges it once."""
        for fid, spent in enumerate(self.spent):
            if spent is None or self.spent_link[fid] is None:
                continue
            link = self.link_of(fid)
            if link is not None and link != self.spent_link[fid]:
                log.info("fragment %d: anchor %d released after a reconnect (push of step %d)",
                         fid, spent[0], spent[1])
                self.spent[fid] = None
                self.spent_link[fid] = None
                self.stats["released_anchors"] = self.stats.get("released_anchors", 0) + 1

    def _pushable(self, pull: Any) -> bool:
        fid = int(pull.fragment_id)
        spent = self.spent[fid]
        return spent is None or spent[0] != self.state.fragment_versions[fid] or spent[1] == int(pull.global_step)

    def _push_dropped(self, fid: int, base: int, step: int) -> None:
        """Sender thread: a dropped PUSH never reached the syncer, so its anchor is not spent."""
        if self.spent[fid] == (base, step):
            self.spent[fid] = None
            self.spent_link[fid] = None
        if self.pushed_from[fid] == base:
            self.pushed_from[fid] = None

    def _push_sent(self, fid: int) -> None:
        """Sender thread, before the PUSH is queued: the link it goes out on."""
        self.spent_link[fid] = self.link_of(fid)

    def _take_updates(self, plan: _Plan) -> dict[int, Any]:
        updates, self._updates = self._updates, {}
        return {fid: updates[fid].data for fid, _ in plan.applies}

    def _push(self, fid: int, step: int, attempt: int, c_steps: int, c_units: int,
              parts: list[Any]) -> str | None:
        """Rank 0: hand one PUSH_FRAGMENT to the background sender and return
        at once (an error only for a fragment that cannot be pushed at all)."""

        import torch

        from yeto.diloco_sync import encode_push_delta
        from yeto.protocol import DTYPE_Q4

        if self._export_error is not None:
            return f"PUSH fragment {fid}: {self._export_error}"
        head = (fid, step, attempt, self.state.fragment_versions[fid], self.iteration, c_steps, c_units)
        self.spent[fid] = (self.state.fragment_versions[fid], step)
        self.spent_link[fid] = None  # set by the sender once it goes out
        self.pushed_from[fid] = self.state.fragment_versions[fid]
        if self.wire == DTYPE_Q4:
            # q4 blocks span tensors: the shared encoder takes the whole delta.
            self.sender.submit(head, payload=encode_push_delta(self.client, torch.cat(parts)))
        else:
            self.sender.submit(head, parts=parts)
        return None

    def _finalize(self) -> None:
        """Install the coordinator's final cut raw (``full``) or only confirm it
        (``manifest``), then FINAL_ACK; every rank returns together."""

        import torch
        import torch.distributed as dist

        from yeto.protocol import FinalManifest

        error, manifest, finals = None, None, {}
        if self.rank == 0:
            try:
                if self.final_cut_payload == "full":
                    manifest, broadcasts = self.client.wait_for_final_fragments()
                    finals = {update.fragment_id: update for update in broadcasts}
                    # Checked before the outcome is broadcast, so a mismatch
                    # fails every rank instead of rank 0 alone.
                    for fid in range(self.num_fragments):
                        update = finals.get(fid)
                        if update is None or update.version != manifest.versions[fid]:
                            raise SwiftMegatronSyncError(
                                f"final fragment {fid} has version {None if update is None else update.version}, "
                                f"expected {manifest.versions[fid]}")
                        self._check_payload(fid, update.data, torch.float32)
                else:
                    manifest = self.client.wait_for_final_manifest()
            except Exception as exc:
                error = f"finalization: {type(exc).__name__}: {exc}"
        box = self._bcast_object((error, None if manifest is None else (manifest.global_step,
                                                                         tuple(manifest.versions))))
        self._raise_if(box[0])
        global_step, versions = box[1]
        manifest = FinalManifest(global_step, tuple(versions))
        if self.final_cut_payload == "full":
            for fid in range(self.num_fragments):
                update = finals.get(fid) if self.rank == 0 else None
                self._apply_fragment(fid, None if update is None else update.data, torch.float32, blend=False)
                self.state.fragment_versions[fid] = manifest.versions[fid]
        if self.world > 1:
            dist.barrier(group=self.comm.world)
        error = None
        if self.rank == 0:
            try:
                self.client.acknowledge_finalization(manifest)
            except Exception as exc:
                error = f"FINAL_ACK: {type(exc).__name__}: {exc}"
        self._raise_if(self._bcast_object(error))
        self.state.global_step = max(self.state.global_step, manifest.global_step)
        self.finished = True
        self.state.shutdown = True
        if self.rank == 0:
            log.info("final cut at global step %d installed and acknowledged", manifest.global_step)

    # ------------------------------------------------------------------ teardown
    def close(self) -> None:
        """Close the syncer connection (rank 0).  Idempotent, collective-free,
        safe from ``atexit`` after a crash."""

        if self._closed:
            return
        self._closed = True
        if self.heartbeat is not None:
            self.heartbeat.stop()
        if self.sender is not None:
            self.sender.stop()
        client, self.client = self.client, None
        if client is not None:
            try:
                client.close()
            except Exception as exc:  # teardown must not mask the original error
                log.warning("closing the syncer client failed: %s", exc)


def session_dtype(name: str) -> int:
    """The syncer session dtype of a wire name: the torch learner's
    ``--wire-dtype`` names (bf16, f32, q4), or the full-parameter push names
    of the RL island (``int4``, ``int8``, ``int4-log``, ``push-bf16``: f32
    INIT/BCAST, quantized PUSH deltas with blocks restarting at tensor
    boundaries, as the RL async full-parameter preset sends them)."""

    from yeto.diloco_sync import WIRE_DTYPES, wire_dtype_code
    from yeto.protocol import DTYPE_FLAG_TENSOR_BLOCKS, FULL_PARAMETER_PUSH_DTYPES

    if name in WIRE_DTYPES or name == "fp32":
        return wire_dtype_code(name)
    key = "bf16" if name == "push-bf16" else name
    if key not in FULL_PARAMETER_PUSH_DTYPES or key == "f32":
        raise ValueError(f"unknown wire dtype {name!r}: one of {sorted(WIRE_DTYPES)} or "
                         "int4, int8, int4-log, push-bf16")
    code = FULL_PARAMETER_PUSH_DTYPES[key]
    return code | DTYPE_FLAG_TENSOR_BLOCKS if key != "bf16" else code


class _Heartbeat:
    """Rank 0's HEARTBEAT thread (``client.heartbeat(local_step)`` every
    ``interval`` seconds): the syncer renews the island's lease on it, so a
    lease can be on although a step (or an INIT, a checkpoint save) takes far
    longer.  ``stall`` > 0: no heartbeat once the training loop showed no
    progress (``progress_at``) for that long.  A send during an outage is
    dropped by the client; the next one goes out after the redial."""

    def __init__(self, client: Any, interval: float, stall: float, step: Callable[[], int],
                 progress_at: Callable[[], float]) -> None:
        import threading

        self.client, self.interval, self.stall = client, interval, stall
        self.step, self.progress_at = step, progress_at
        self.sent = 0
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="yeto-swift-heartbeat", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            if self.stall > 0 and time.monotonic() - self.progress_at() > self.stall:
                continue
            try:
                self.client.heartbeat(int(self.step()))
                self.sent += 1
            except Exception as exc:  # never take the island down for a heartbeat
                log.debug("heartbeat not sent: %s", exc)

    def stop(self) -> None:
        self._stop.set()
        self.thread.join(timeout=5.0)


class _PushSender:
    """Rank 0's background PUSH streaming: one PUSH in flight at a time, so a
    slow link or syncer never holds a training step (decoupled DiLoCo, as the
    RL full-parameter island's background push).  A lost connection drops the
    push (the syncer re-requests it after the redial); any other failure is
    re-raised by the next boundary on every rank."""

    def __init__(self, client: Any, stats: dict, on_drop: Callable[[int, int, int], None],
                 on_sent: Callable[[int], None] | None = None) -> None:
        import queue
        import threading

        self.client, self.stats, self.on_drop, self.on_sent = client, stats, on_drop, on_sent
        self.jobs: queue.Queue = queue.Queue()
        self.idle = threading.Event()
        self.idle.set()
        self.error: str | None = None
        self.thread = threading.Thread(target=self._run, name="yeto-swift-push", daemon=True)
        self.thread.start()

    def busy(self) -> bool:
        return not self.idle.is_set()

    def submit(self, head: tuple, *, parts: list[Any] | None = None, payload: bytes | None = None) -> None:
        self.idle.clear()
        self.jobs.put((head, parts, payload))

    def raise_error(self) -> None:
        if self.error is not None:
            error, self.error = self.error, None
            raise SwiftMegatronSyncError(error)

    def wait_idle(self, timeout: float | None = None) -> bool:
        return self.idle.wait(timeout)

    def stop(self) -> None:
        self.jobs.put(None)
        self.thread.join(timeout=5.0)

    def _run(self) -> None:
        from yeto.protocol import PartialMessageGenerationLost

        # A torso on a dead connection generation is dropped by the syncer,
        # so all three mean the delta was never delivered.
        undelivered = (ConnectionError, OSError, PartialMessageGenerationLost)
        while True:
            job = self.jobs.get()
            if job is None:
                return
            head, parts, payload = job
            fid = head[0]
            try:
                if payload is not None:
                    parts = [payload]  # q4: already wire bytes, sent unchanged
                nbytes = sum(len(part) for part in parts)
                if self.on_sent is not None:
                    self.on_sent(fid)  # before the send: the link it is queued on
                if self.client.push_fragment_parts(*head, parts) is False:
                    # Nothing was queued: the link is down (an outage, a syncer
                    # restart).  The syncer never sees this delta.
                    raise ConnectionError("syncer link down, nothing queued")
                self.stats["pushed"] += 1
                self.stats["push_bytes"] += nbytes
            except undelivered as exc:  # the anchor is not spent
                self.stats["dropped_pushes"] += 1
                self.on_drop(fid, head[3], head[1])
                log.warning("push of fragment %d dropped (%s); the syncer re-requests it", fid, exc)
            except Exception as exc:
                self.error = f"PUSH fragment {fid}: {type(exc).__name__}: {exc}"
            finally:
                del parts, payload, job
                if self.jobs.empty():
                    self.idle.set()


def _current_accelerator() -> Any:
    import torch

    try:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            return torch.device("npu", torch.npu.current_device())
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


__all__: Iterable[str] = (
    "GlobalTable",
    "LocalShard",
    "MegatronDiLoCoSync",
    "ParallelContext",
    "SwiftMegatronSyncError",
    "chunk_names",
    "global_parameter_name",
    "local_shards",
)
