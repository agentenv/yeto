"""Preflight of ``--fragments`` (and, for verl, the layout hash) for a full-parameter run.

A Miles full-parameter island refuses a layout whose fragment count is not
``--fragments`` (``MilesChunkedFullParameterAdapter.initialize``), but it only
knows the count after it has built the model: minutes of paid boot per island.
The count follows from the model config, the trainer parallelism, the fragment
byte bound (``--rl-sync-fragment-max-bytes``, FP32) and the matrix merge, so
the launcher computes it here with the backend's offline planner (Miles:
``launch_flags.offline_full_parameter_layout``, the dense Qwen3 layout of
``models.dense_gpt_layout``) and refuses a mismatch before anything is ordered.

Only the model's ``config.json`` is read, from a local model directory or the
local Hugging Face cache; nothing is downloaded. Out of scope (no config in the
cache, a model family the offline layout does not list, shape-free layouts,
partial experts, CP or DP above 1, another backend), the check prints nothing
and the run goes on as before. At PP>1 the count is an estimate: a mismatch is
a warning, not a refusal.

verl (``launch_flags.OFFLINE_LAYOUT_CARD_FREE``): its layout names no card or shard,
so the planner (``adapters.verl.full_parameter_plan``: the HF model on the meta
device) gives one layout for every island shape, DP included; the verl island has no
``auto`` hash, so the launch also refuses a ``--rl-parameter-layout-sha256`` that is
not the planned hash (when the model revision is a commit and the hash can be
computed). A fleet island added or moved at run time runs the same check in its
launch's dry run.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def local_hf_config(model: str, revision: str | None) -> bytes | None:
    """The bytes of ``model``'s ``config.json`` from a local directory or the local
    HF cache (``revision``: a commit, branch or tag; None: main); None if not there."""
    try:
        path = Path(str(model)).expanduser()
        if path.is_dir():
            config = path / "config.json"
            return config.read_bytes() if config.is_file() else None
        from huggingface_hub import try_to_load_from_cache  # noqa: PLC0415

        found = try_to_load_from_cache(str(model), "config.json", revision=revision or None)
        return Path(found).read_bytes() if isinstance(found, str) else None
    except Exception:  # noqa: BLE001 - not a repo id, no huggingface_hub, unreadable
        return None


def _trainer_islands(args: Any, per_island: list, specs: list) -> list[tuple[Any, Any]]:
    from ..rl.engine.sample_roles import SAMPLER, island_roles

    roles = island_roles(args, len(specs))
    return [(iargs, spec) for m, (iargs, spec) in enumerate(zip(per_island, specs))
            if roles.get(m) != SAMPLER]


def _in_scope(args: Any, trainers: list[tuple[Any, Any]]) -> bool:
    from ..rl.engine.selection import FULL_PARAMETER_SYNC_PRESET
    from .topology import rl_trainer_shape

    if (getattr(args, "rl_sync_preset", None) != FULL_PARAMETER_SYNC_PRESET
            or getattr(args, "tuning", "lora") != "full"
            or getattr(args, "rl_sync_layout", "shape-bound") != "shape-bound"
            or getattr(args, "rl_model_recipe", "generic") != "generic"
            or getattr(args, "rl_expert_selection", None)
            or int(getattr(args, "context_parallel", 1) or 1) != 1
            or int(getattr(args, "expert_parallel", 1) or 1) != 1
            or not trainers):
        return False
    group = int(args.tensor_parallel or 1) * int(getattr(args, "pipeline_parallel", 1) or 1)
    card_free = _card_free(args)
    for iargs, spec in trainers:
        if getattr(iargs, "rl_backend", None) != getattr(args, "rl_backend", None):
            return False  # an island of another backend (--rl-island-engine)
        if card_free:  # one layout on every shape
            continue
        try:
            nodes, gpus = rl_trainer_shape(iargs, spec)
        except ValueError:
            return False
        if nodes * gpus != group:  # DP > 1 (or a shape the layout does not model)
            return False
    return True


def _card_free(args: Any) -> bool:
    """Whether the backend's layout names no card or shard (verl: module doc)."""
    from .rl_backend import _rl_backend_module

    return bool(getattr(_rl_backend_module(args, "launch_flags"), "OFFLINE_LAYOUT_CARD_FREE", False))


@dataclass(frozen=True)
class LayoutSummary:
    """The full-parameter layout an island of one trainer shape builds (offline planner)."""

    tp: int
    pp: int
    fragments: int
    ranks: int
    fragment_bytes: tuple[int, ...]
    layout_hash: str | None  # None without a commit revision (dense_gpt_layout.offline_layout)
    exact: bool

    @property
    def shape(self) -> str:
        return f"TP{self.tp} x PP{self.pp}"

    def digest(self) -> str:
        """Short digest of the layout: its hash, else the SHA-256 of the fragment sizes."""
        if self.layout_hash:
            return self.layout_hash[:16]
        return hashlib.sha256(json.dumps(self.fragment_bytes).encode()).hexdigest()[:16]

    def describe(self) -> str:
        return (f"{self.shape}: {self.fragments} fragments over {self.ranks} trainer rank(s), "
                f"layout {self.digest()}" + ("" if self.exact else " (PP>1 estimate)"))


def offline_summary(args: Any, *, tp: int, pp: int) -> LayoutSummary | None:
    """The layout of a trainer of shape TP x PP from the backend's offline planner and the
    model's local ``config.json``; None when the planner or the config is not there."""
    from .rl_backend import _rl_backend_module

    plan = getattr(_rl_backend_module(args, "launch_flags"), "offline_full_parameter_layout", None)
    if plan is None:
        return None
    raw = local_hf_config(args.model, getattr(args, "model_revision", None))
    if raw is None:
        return None
    try:
        config = json.loads(raw)
    except ValueError:
        return None
    try:
        layout = plan(
            config, tp=tp, pp=pp, minimum_fragments=int(args.fragments),
            max_fragment_bytes=int(getattr(args, "rl_sync_fragment_max_bytes", 2 << 30)),
            matrix_merge=getattr(args, "rl_sync_matrix_merge", "avg"),
            model_revision=str(getattr(args, "model_revision", None) or ""),
            config_sha256=hashlib.sha256(raw).hexdigest())
    except ImportError:  # the backend's planner needs a package the control host lacks
        return None
    if layout is None:
        return None
    return LayoutSummary(tp=tp, pp=pp, fragments=layout.fragments, ranks=layout.ranks,
                         fragment_bytes=tuple(layout.fragment_bytes), layout_hash=layout.layout_hash,
                         exact=layout.exact)


def check_fragment_layout(args: Any, per_island: list, specs: list) -> None:
    """Refuse (ValueError) a ``--fragments`` the islands' full-parameter layout
    cannot have; warn on stderr when the count is an estimate (PP>1). Silent when
    the count matches or the check is out of scope (module doc)."""
    from .rl_backend import _rl_backend_module

    # the backend's offline planner (Miles: launch_flags.offline_full_parameter_layout)
    plan = getattr(_rl_backend_module(args, "launch_flags"), "offline_full_parameter_layout", None)
    trainers = _trainer_islands(args, per_island, specs)
    if plan is None or not _in_scope(args, trainers):
        return
    tp = int(args.tensor_parallel or 1)
    pp = int(getattr(args, "pipeline_parallel", 1) or 1)
    layout = offline_summary(args, tp=tp, pp=pp)
    if layout is None:
        return
    card_free = _card_free(args)
    max_bytes = int(getattr(args, "rl_sync_fragment_max_bytes", 2 << 30))
    merge = getattr(args, "rl_sync_matrix_merge", "avg")
    backend = getattr(args, "rl_backend", None) or "miles"
    where = "on any island shape" if card_free else f"at TP{tp} x PP{pp}"
    if layout.fragments == int(args.fragments):
        pinned = str(getattr(args, "rl_parameter_layout_sha256", None) or "")
        if card_free and layout.layout_hash and re.fullmatch(r"[0-9a-f]{64}", pinned) and pinned != layout.layout_hash:
            raise ValueError(
                f"--rl-parameter-layout-sha256 {pinned[:16]}... is not the layout the trainer islands build: the "
                f"{backend} full-parameter layout of {args.model}@{args.model_revision} {where} (config.json from the "
                f"local model files), {layout.fragments} FP32 fragments of at most {max_bytes} bytes and matrix "
                f"merge {merge} has hash {layout.layout_hash}; every island would refuse it at its start "
                "(and the syncer session binds it). Set --rl-parameter-layout-sha256 " + layout.layout_hash)
        return
    name = "Miles" if backend == "miles" else backend
    detail = (f"the {name} full-parameter layout of {args.model} {where} "
              f"(config.json from the local model files), FP32 fragments of at most "
              f"{max_bytes} bytes (--rl-sync-fragment-max-bytes) and matrix merge {merge} "
              f"has {layout.fragments} fragments over {layout.ranks} trainer rank(s)")
    if not layout.exact:
        print(f"[launcher] WARNING: --fragments {args.fragments} may not match: by estimate "
              f"{detail} (PP>1: Megatron's stage split is estimated); an island refuses a "
              f"mismatched layout after it has built the model", file=sys.stderr, flush=True)
        return
    raise ValueError(
        f"--fragments {args.fragments} does not match the layout the trainer islands build: "
        f"{detail}; every island would refuse it after its boot. Set --fragments {layout.fragments}")


# The fields of a trainer shape (--rl-fleet-trainer-shape) and their value when a
# spec leaves them out: the parallel dims, the routed experts of the model, the
# partial-experts selection (its file's SHA-256) and the DeepSeek V4 expert-full count.
SHAPE_DEFAULTS: dict[str, str] = {"tp": "1", "pp": "1", "ep": "1", "cp": "1",
                                  "expert_selection": "none", "expert_full": "0"}
SHAPE_FIELDS = ("tp", "pp", "ep", "cp", "experts", "expert_selection", "expert_full")
# config.json keys of the routed expert count (Qwen3-MoE, DeepSeek, Mixtral)
_EXPERT_KEYS = ("num_experts", "n_routed_experts", "num_local_experts")


def parse_trainer_shape(value: str) -> dict[str, str]:
    """``--rl-fleet-trainer-shape``: ``TP``, ``TPxPP`` or ``key=value,...`` over
    :data:`SHAPE_FIELDS` (:func:`trainer_shape_spec` renders it). Fields left out
    take :data:`SHAPE_DEFAULTS`; ``experts`` left out is not compared."""
    text = str(value).strip().lower()
    if "=" not in text:
        parts = text.split("x")
        if len(parts) not in (1, 2) or not all(part.isdigit() and int(part) >= 1 for part in parts):
            raise ValueError(f"--rl-fleet-trainer-shape must be TP, TPxPP or key=value,... "
                             f"(keys {', '.join(SHAPE_FIELDS)}), got {value!r}")
        return {**SHAPE_DEFAULTS, "tp": parts[0], "pp": parts[1] if len(parts) == 2 else "1"}
    shape = dict(SHAPE_DEFAULTS)
    for item in text.split(","):
        key, _, raw = item.partition("=")
        key, raw = key.strip(), raw.strip()
        if key not in SHAPE_FIELDS or not raw:
            raise ValueError(f"--rl-fleet-trainer-shape: unknown or empty field {item!r} "
                             f"(keys {', '.join(SHAPE_FIELDS)})")
        if key != "expert_selection" and not (raw.isdigit() and (int(raw) >= 1 or key in ("experts", "expert_full"))):
            raise ValueError(f"--rl-fleet-trainer-shape: {key} must be a whole number, got {raw!r}")
        shape[key] = str(int(raw)) if key != "expert_selection" else raw
    return shape


def _model_experts(args: Any) -> str | None:
    """Routed experts of the model from its local config.json ("0": dense); None: unknown."""
    raw = local_hf_config(args.model, getattr(args, "model_revision", None))
    if raw is None:
        return None
    try:
        config = json.loads(raw)
    except ValueError:
        return None
    config = config.get("text_config", config) if isinstance(config.get("text_config"), dict) else config
    for key in _EXPERT_KEYS:
        if isinstance(config.get(key), int):
            return str(config[key])
    return "0"


def _selection_sha256(path: Any) -> str:
    if not path:
        return "none"
    try:
        return hashlib.sha256(Path(str(path)).expanduser().read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(f"--rl-expert-selection {path}: unreadable ({error}); the fleet shape "
                         "check compares its SHA-256") from None


def trainer_shape(args: Any) -> dict[str, str]:
    """This launch's trainer shape (:data:`SHAPE_FIELDS`); ``experts`` only when the
    model's config.json is local."""
    shape = {"tp": str(int(args.tensor_parallel or 1)),
             "pp": str(int(getattr(args, "pipeline_parallel", 1) or 1)),
             "ep": str(int(getattr(args, "expert_parallel", 1) or 1)),
             "cp": str(int(getattr(args, "context_parallel", 1) or 1)),
             "expert_selection": _selection_sha256(getattr(args, "rl_expert_selection", None)),
             "expert_full": str(int(getattr(args, "expert_full_count", 0) or 0))}
    experts = _model_experts(args)
    if experts is not None:
        shape["experts"] = experts
    return shape


def trainer_shape_spec(args: Any) -> str:
    """The ``--rl-fleet-trainer-shape`` value of a fleet whose islands launch with ``args``."""
    shape = trainer_shape(args)
    return ",".join(f"{key}={shape[key]}" for key in SHAPE_FIELDS if key in shape)


def _label(shape: Mapping[str, str]) -> str:
    out = f"TP{shape['tp']} x PP{shape['pp']} x EP{shape['ep']} x CP{shape['cp']}"
    if "experts" in shape:
        out += f", {shape['experts']} routed experts"
    if shape["expert_selection"] != "none":
        out += f", expert selection {shape['expert_selection'][:16]}"
    if shape["expert_full"] != "0":
        out += f", expert-full {shape['expert_full']}"
    return out


def check_fleet_shape(args: Any) -> None:
    """A new or migrated island of a running fleet (``--rl-fleet-trainer-shape``, the
    trainer shape of the fleet's islands): refuse (ValueError, so at dry-run, before
    anything is ordered) an island whose full-parameter layout differs from the fleet's.

    The shapes are compared field by field (TP, PP, EP, CP, the model's routed
    experts, the partial-experts selection, the expert-full count): a shape-bound
    layout binds the trainer ranks and the trained experts, so any difference is
    another layout. Where the ``--fragments`` preflight's offline planner models
    both shapes (dense models), the refusal also names both fragment counts and
    layout digests. A shape-free layout (``--rl-sync-layout shape-free``) spans the
    parallel dims; its trained experts are still compared. Unset: nothing is checked."""
    from ..rl.engine.selection import FULL_PARAMETER_SYNC_PRESET

    value = getattr(args, "rl_fleet_trainer_shape", None)
    if value is None:
        return
    fleet = parse_trainer_shape(value)
    if getattr(args, "rl_sync_preset", None) != FULL_PARAMETER_SYNC_PRESET:
        raise ValueError(f"--rl-fleet-trainer-shape applies to --rl-sync-preset {FULL_PARAMETER_SYNC_PRESET}")
    mine = trainer_shape(args)
    compared = [key for key in SHAPE_FIELDS if key in fleet and key in mine]
    if getattr(args, "rl_sync_layout", "shape-bound") != "shape-bound":
        compared = [key for key in compared if key not in ("tp", "pp", "ep", "cp")]
    differ = [key for key in compared if fleet[key] != mine[key]]
    if not differ:
        return
    detail = "; ".join(f"{key} {fleet[key][:16]} (fleet) vs {mine[key][:16]} (this island)" for key in differ)
    layouts = ""
    if set(differ) <= {"tp", "pp"}:
        theirs = offline_summary(args, tp=int(fleet["tp"]), pp=int(fleet["pp"]))
        own = offline_summary(args, tp=int(mine["tp"]), pp=int(mine["pp"]))
        if theirs is not None and own is not None:
            layouts = f" Layouts: fleet islands {theirs.describe()}; this island {own.describe()}."
    raise ValueError(
        f"this island's trainer shape differs from the fleet's: {detail}. Fleet islands "
        f"{_label(fleet)}; this island {_label(mine)}.{layouts} Islands whose full-parameter "
        "layouts differ cannot merge in one syncer session (the syncer would refuse the island "
        f"after its boot); launch it with the fleet's shape")
