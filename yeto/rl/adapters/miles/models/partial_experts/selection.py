"""``expert_selection.json`` generator and validator (INTERFACES 2.3).

Document (schema 1)::

    {"schema": 1, "model": "kimi_k31", "num_experts": 896, "num_trainable": 128,
     "ep_size": 8,
     "layers": {"1": {"trainable": [128 original ids, ascending],
                      "slot_order": [896 original ids: per EP rank the first
                                     num_trainable/ep_size local slots are
                                     trainable, the rest frozen]}, ...},
     "load": {"1": [896 routed-token counts]},        # optional, re-derives slot_order
     "source": {"method": "routing_frequency", "prompts_sha256": "...", "tokens": N},
     "sha256": "<sha256 of the canonical JSON of `layers`>",
     "trainable_sha256": "<sha256 of the canonical JSON of {layer: trainable}>"}

``trainable`` is the per-layer top-``num_trainable`` by routed-token count,
ties to the lower id. ``slot_order`` is the trainer's slot -> original id
permutation: the trainable experts are placed so that every power-of-two EP
deals the hot ones round-robin (the ``i``-th heaviest at trainable index
``bitrev(i)``), the frozen ones are dealt by load in snake order and sorted
within a rank. ``slot_order`` depends on the EP size; ``trainable`` does not,
so changing EP re-derives ``slot_order`` from ``load`` and keeps
``trainable_sha256``. ``sha256`` covers ``layers`` as a whole, as the
interface requires.

Inputs: group payloads of the sample buffer (``rollout_routed_experts``,
``uint16[T, L_moe, top_k]`` of original ids; S1), or routed-token counts of
the trainer-side router hook (``routing_plugin.drain``: ``{HF layer: [counts]}``).

CLI::

    python -m yeto.rl.adapters.miles.models.partial_experts.selection \\
        --model kimi_k31 --num-experts 896 --num-trainable 128 --ep-size 8 \\
        --moe-layer-offset 1 --payload group-*.json --output expert_selection.json

    ... --counts routed_counts.json ...           # routing_plugin counts
    ... --input expert_selection.json --ep-size 4  # re-derive slot_order
    ... --validate expert_selection.json
    ... --mode uniform --num-moe-layers 92 ...     # GPU smoke before any prompt set
    ... --mode random --seed 0 --num-moe-layers 92 ...

Selection is **per MoE layer**: every layer picks its own ``num_trainable``
experts (K3.1: 92 layers x 128); nothing is shared across layers.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .attest import canonical_sha256, require_sha256


def canonical_layers_sha256(layers: Mapping[str, Mapping[str, Sequence[int]]]) -> str:
    """The identity Miles checks (F2 ``configs/partial_experts.py``, copied verbatim)."""

    import hashlib

    canonical = {
        str(k): {"trainable": [int(x) for x in v["trainable"]], "slot_order": [int(x) for x in v["slot_order"]]}
        for k, v in layers.items()
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()

SCHEMA = 1
# A position without routing: 0xFFFF in the uint16 payload (S1), -1 in a signed one (Miles).
ROUTED_EXPERTS_PAD = 65535
METHOD_ROUTING_FREQUENCY = "routing_frequency"
METHOD_UNIFORM = "uniform"
METHOD_RANDOM = "random"
SYNTHETIC_METHODS = (METHOD_UNIFORM, METHOD_RANDOM)
ROUTED_EXPERTS_FIELD = "rollout_routed_experts"

Counts = dict[int, list[int]]  # HF layer -> per-original-expert routed-token count


class SelectionError(ValueError):
    """The selection document violates the interface."""


# --------------------------------------------------------------------------
# counting
# --------------------------------------------------------------------------


def counts_from_routed_experts(
    routed: Any,
    *,
    num_experts: int,
    moe_layer_offset: int,
    into: Counts | None = None,
) -> Counts:
    """Accumulate ``[T, L_moe, top_k]`` original ids into per-layer counts.

    MoE layer ``i`` of the array is HF layer ``i + moe_layer_offset`` (K3.1:
    layer 0 is dense, so the offset is 1).
    """

    import numpy

    array = numpy.asarray(routed)
    if array.ndim != 3:
        raise SelectionError(f"routed experts must be [T, L, top_k], got shape {array.shape}")
    if array.dtype.kind not in "iu":
        raise SelectionError(f"routed experts must be integers, got {array.dtype}")
    # pads (65535 in the uint16 payload, -1 in a signed one) carry no routing
    valid = (array != ROUTED_EXPERTS_PAD) & (array != -1) if array.size else array.astype(bool)
    if array.size and valid.any():
        ids = array[valid]
        if ids.min() < 0 or ids.max() >= num_experts:
            raise SelectionError(f"routed expert id outside [0, {num_experts})")
    counts = into if into is not None else {}
    for moe_layer in range(array.shape[1]):
        layer = moe_layer + moe_layer_offset
        row = counts.setdefault(layer, [0] * num_experts)
        ids = array[:, moe_layer, :][valid[:, moe_layer, :]].astype(numpy.int64)
        bins = numpy.bincount(ids, minlength=num_experts)
        for expert in numpy.nonzero(bins)[0]:
            row[int(expert)] += int(bins[expert])
    return counts


def _sample_routed_experts(sample: Mapping[str, Any]) -> Any:
    """The sample's routed experts through the buffer reader when it has one (S1:
    ``rollout_routed_experts(sample)`` normalizes uint8 / int32 / uint16 and pads)."""

    from yeto.rl.engine import buffer_payload

    reader = getattr(buffer_payload, "rollout_routed_experts", None)
    if callable(reader):
        return reader(sample)
    return sample.get(ROUTED_EXPERTS_FIELD)


def counts_from_payloads(
    paths: Iterable[str | Path],
    *,
    num_experts: int,
    moe_layer_offset: int,
) -> tuple[Counts, int]:
    """Counts and token total over the ``rollout_routed_experts`` of every sample."""

    from yeto.rl.engine.buffer_payload import decode_group

    counts: Counts = {}
    tokens = 0
    for path in paths:
        for sample in decode_group(Path(path).read_bytes()):
            routed = _sample_routed_experts(sample)
            if routed is None:
                continue
            counts_from_routed_experts(
                routed, num_experts=num_experts, moe_layer_offset=moe_layer_offset, into=counts
            )
            tokens += int(routed.shape[0])
    if not counts:
        raise SelectionError(f"no sample carried {ROUTED_EXPERTS_FIELD!r}")
    return counts, tokens


def counts_from_routing_plugin(value: Mapping[str, Sequence[int]], *, num_experts: int) -> Counts:
    """``routing_plugin.drain`` output (``{"<HF layer>": [counts]}``) as counts."""

    counts: Counts = {}
    for key, row in value.items():
        layer = int(key)
        if len(row) != num_experts:
            raise SelectionError(f"layer {layer} has {len(row)} counts, expected {num_experts}")
        counts[layer] = [int(x) for x in row]
    if not counts:
        raise SelectionError("routing counts are empty")
    return counts


def merge_counts(*parts: Counts) -> Counts:
    out: Counts = {}
    for part in parts:
        for layer, row in part.items():
            target = out.setdefault(layer, [0] * len(row))
            if len(target) != len(row):
                raise SelectionError(f"layer {layer}: expert counts differ in length")
            for i, value in enumerate(row):
                target[i] += int(value)
    return out


# --------------------------------------------------------------------------
# selection and slot order
# --------------------------------------------------------------------------


def select_top_k(row: Sequence[int | float], k: int) -> list[int]:
    """Top-``k`` ids by count, ties to the lower id, ascending."""

    if not 0 < k < len(row):
        raise SelectionError(f"num_trainable {k} must be in (0, {len(row)})")
    order = sorted(range(len(row)), key=lambda i: (-float(row[i]), i))
    return sorted(order[:k])


def _bit_reverse(value: int, bits: int) -> int:
    return int(f"{value:0{bits}b}"[::-1], 2) if bits else 0


def trainable_order(trainable: Sequence[int], load: Sequence[int | float] | None) -> list[int]:
    """EP-independent order of the trainable experts (trainable index -> id).

    The ``i``-th heaviest sits at ``bitrev(i)`` so every power-of-two EP, taking
    contiguous chunks, gets an equal share of the hot experts.
    """

    k = len(trainable)
    if k & (k - 1):
        raise SelectionError(f"num_trainable must be a power of two, got {k}")
    bits = k.bit_length() - 1
    weight = (lambda i: -float(load[i])) if load is not None else (lambda i: 0.0)
    by_load = sorted(trainable, key=lambda i: (weight(i), i))
    order = [0] * k
    for position, expert in enumerate(by_load):
        order[_bit_reverse(position, bits)] = expert
    return order


def _snake_assign(ids_by_load: Sequence[int], num_ranks: int) -> list[list[int]]:
    buckets: list[list[int]] = [[] for _ in range(num_ranks)]
    for position, expert in enumerate(ids_by_load):
        rnd, offset = divmod(position, num_ranks)
        buckets[offset if rnd % 2 == 0 else num_ranks - 1 - offset].append(expert)
    return buckets


def slot_order(
    trainable: Sequence[int],
    *,
    num_experts: int,
    ep_size: int,
    load: Sequence[int | float] | None,
) -> list[int]:
    """slot -> original id for one layer under ``ep_size``.

    Rank ``r`` owns slots ``[r*n_local, (r+1)*n_local)``; its first ``k_local``
    hold ``trainable_order[r*k_local:(r+1)*k_local]``, the rest the frozen
    experts dealt by load (snake), sorted by id.
    """

    k = len(trainable)
    if num_experts % ep_size or k % ep_size:
        raise SelectionError(f"EP {ep_size} must divide {num_experts} experts and {k} trainable")
    k_local = k // ep_size
    hot = trainable_order(trainable, load)
    frozen = sorted(set(range(num_experts)) - set(trainable),
                    key=(lambda i: (-float(load[i]), i)) if load is not None else (lambda i: i))
    cold = _snake_assign(frozen, ep_size)
    return [
        expert
        for rank in range(ep_size)
        for expert in (hot[rank * k_local:(rank + 1) * k_local] + sorted(cold[rank]))
    ]


# --------------------------------------------------------------------------
# document
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpertSelection:
    model: str
    num_experts: int
    num_trainable: int
    ep_size: int
    trainable: Mapping[int, tuple[int, ...]]  # HF layer -> ascending original ids
    slot_orders: Mapping[int, tuple[int, ...]]  # HF layer -> slot -> original id
    load: Mapping[int, tuple[int, ...]] | None = None
    source: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_counts(
        cls,
        counts: Counts,
        *,
        model: str,
        num_trainable: int,
        ep_size: int,
        source: Mapping[str, Any] | None = None,
        keep_load: bool = True,
    ) -> "ExpertSelection":
        if not counts:
            raise SelectionError("no routing counts")
        num_experts = {len(row) for row in counts.values()}
        if len(num_experts) != 1:
            raise SelectionError("layers disagree on the expert count")
        (n,) = num_experts
        trainable = {layer: tuple(select_top_k(row, num_trainable)) for layer, row in counts.items()}
        orders = {
            layer: tuple(slot_order(ids, num_experts=n, ep_size=ep_size, load=counts[layer]))
            for layer, ids in trainable.items()
        }
        load = {layer: tuple(int(x) for x in row) for layer, row in counts.items()} if keep_load else None
        src = {"method": METHOD_ROUTING_FREQUENCY, **dict(source or {})}
        src.setdefault("tokens", 0)
        return cls(model, n, num_trainable, int(ep_size), trainable, orders, load, src)

    @classmethod
    def synthetic(
        cls,
        mode: str,
        *,
        model: str,
        num_experts: int,
        num_trainable: int,
        ep_size: int,
        layers: Sequence[int],
        seed: int = 0,
    ) -> "ExpertSelection":
        """A selection without routing statistics, for GPU smoke runs.

        ``uniform``: ids spaced evenly over ``range(num_experts)`` (the same
        set on every layer); ``random``: an independent seeded sample per
        layer (``seed`` + layer). No ``load`` is stored, so ``slot_order``
        falls back to id order and ``coverage`` is undefined.
        """

        import random as _random

        if mode not in SYNTHETIC_METHODS:
            raise SelectionError(f"mode must be one of {SYNTHETIC_METHODS}, got {mode!r}")
        if not layers:
            raise SelectionError("synthetic selection needs at least one layer")
        if not 0 < num_trainable < num_experts:
            raise SelectionError(f"num_trainable {num_trainable} must be in (0, {num_experts})")
        trainable: dict[int, tuple[int, ...]] = {}
        for layer in layers:
            if mode == METHOD_UNIFORM:
                ids = sorted({(i * num_experts) // num_trainable for i in range(num_trainable)})
                if len(ids) != num_trainable:  # only when num_trainable > num_experts/2 rounding collides
                    ids = list(range(num_trainable))
            else:
                ids = sorted(_random.Random(seed * 1_000_003 + int(layer)).sample(range(num_experts), num_trainable))
            trainable[int(layer)] = tuple(ids)
        orders = {
            layer: tuple(slot_order(ids, num_experts=num_experts, ep_size=ep_size, load=None))
            for layer, ids in trainable.items()
        }
        source = {"method": mode, "tokens": 0, **({"seed": int(seed)} if mode == METHOD_RANDOM else {})}
        return cls(model, int(num_experts), int(num_trainable), int(ep_size), trainable, orders, None, source)

    def with_ep_size(self, ep_size: int) -> "ExpertSelection":
        """Same trainable set, ``slot_order`` re-derived for another EP."""

        orders = {
            layer: tuple(slot_order(
                ids, num_experts=self.num_experts, ep_size=int(ep_size),
                load=self.load[layer] if self.load is not None else None,
            ))
            for layer, ids in self.trainable.items()
        }
        return ExpertSelection(self.model, self.num_experts, self.num_trainable, int(ep_size),
                               self.trainable, orders, self.load, self.source)

    # -- identity ---------------------------------------------------------
    def layers_json(self) -> dict[str, dict[str, list[int]]]:
        return {
            str(layer): {"trainable": list(self.trainable[layer]),
                         "slot_order": list(self.slot_orders[layer])}
            for layer in sorted(self.trainable)
        }

    @property
    def sha256(self) -> str:
        return canonical_layers_sha256(self.layers_json())

    @property
    def trainable_sha256(self) -> str:
        return canonical_sha256({str(layer): list(ids) for layer, ids in sorted(self.trainable.items())})

    # -- serialization ----------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "schema": SCHEMA,
            "model": self.model,
            "num_experts": self.num_experts,
            "num_trainable": self.num_trainable,
            "ep_size": self.ep_size,
            "layers": self.layers_json(),
            "source": dict(self.source),
            "sha256": self.sha256,
            "trainable_sha256": self.trainable_sha256,
        }
        if self.load is not None:
            doc["load"] = {str(layer): list(row) for layer, row in sorted(self.load.items())}
        return doc

    def write(self, path: str | Path) -> str:
        Path(path).write_text(json.dumps(self.to_json(), indent=1, sort_keys=True) + "\n")
        return self.sha256

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> "ExpertSelection":
        validate_document(doc)
        load = doc.get("load")
        return cls(
            model=str(doc["model"]),
            num_experts=int(doc["num_experts"]),
            num_trainable=int(doc["num_trainable"]),
            ep_size=int(doc["ep_size"]),
            trainable={int(k): tuple(v["trainable"]) for k, v in doc["layers"].items()},
            slot_orders={int(k): tuple(v["slot_order"]) for k, v in doc["layers"].items()},
            load=None if load is None else {int(k): tuple(int(x) for x in v) for k, v in load.items()},
            source=dict(doc.get("source") or {}),
        )

    @classmethod
    def load_file(cls, path: str | Path, *, expected_sha256: str | None = None) -> "ExpertSelection":
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        selection = cls.from_json(doc)
        if expected_sha256 is not None:
            wanted = require_sha256(expected_sha256, "expected selection SHA256")
            if selection.sha256 != wanted:
                raise SelectionError(
                    f"expert selection SHA256 mismatch: expected {wanted}, got {selection.sha256}"
                )
        return selection

    def check_layers(self, expected: Iterable[int]) -> None:
        """The selection must name exactly the model's MoE layers (K3.1: 1..92)."""

        wanted = sorted(int(x) for x in expected)
        got = sorted(self.trainable)
        if got != wanted:
            missing = sorted(set(wanted) - set(got))
            extra = sorted(set(got) - set(wanted))
            raise SelectionError(
                f"selection layers {got[:3]}..{got[-1:]} != model MoE layers {wanted[:3]}..{wanted[-1:]} "
                f"(missing {missing[:8]}, extra {extra[:8]})"
            )

    # -- quality ----------------------------------------------------------
    def coverage(self, counts: Counts | None = None) -> dict[str, float]:
        """Fraction of routed tokens that land on a trainable expert.

        Per layer (keys are the layer numbers as strings) plus ``mean`` and
        ``min`` over layers. Uses ``counts`` (e.g. a held-out set) or the
        stored ``load``.
        """

        source = counts if counts is not None else self.load
        if source is None:
            raise SelectionError("coverage needs routing counts (no load stored)")
        out: dict[str, float] = {}
        for layer, ids in sorted(self.trainable.items()):
            row = source.get(layer)
            if row is None:
                continue
            total = float(sum(row))
            out[str(layer)] = (sum(float(row[i]) for i in ids) / total) if total else 0.0
        values = list(out.values())
        if not values:
            raise SelectionError("coverage: no layer of the selection has counts")
        out["mean"] = sum(values) / len(values)
        out["min"] = min(values)
        return out


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def validate_document(doc: Mapping[str, Any]) -> None:
    """Check a selection document against the interface; raise :class:`SelectionError`."""

    if not isinstance(doc, Mapping):
        raise SelectionError("selection must be a JSON object")
    if doc.get("schema") != SCHEMA:
        raise SelectionError(f"unsupported selection schema {doc.get('schema')!r}")
    for key in ("model", "num_experts", "num_trainable", "ep_size", "layers", "sha256"):
        if key not in doc:
            raise SelectionError(f"selection lacks {key!r}")
    n, k, ep = int(doc["num_experts"]), int(doc["num_trainable"]), int(doc["ep_size"])
    if n <= 0 or not 0 < k < n or ep <= 0:
        raise SelectionError("num_experts, num_trainable and ep_size must be positive, num_trainable < num_experts")
    if n % ep or k % ep:
        raise SelectionError(f"ep_size {ep} must divide num_experts {n} and num_trainable {k}")
    layers = doc["layers"]
    if not isinstance(layers, Mapping) or not layers:
        raise SelectionError("layers must be a non-empty object")
    n_local, k_local = n // ep, k // ep
    for key, layer in layers.items():
        try:
            int(key)
        except (TypeError, ValueError):
            raise SelectionError(f"layer key {key!r} is not an integer") from None
        if not isinstance(layer, Mapping) or "trainable" not in layer or "slot_order" not in layer:
            raise SelectionError(f"layer {key}: needs 'trainable' and 'slot_order'")
        trainable = list(layer["trainable"])
        order = list(layer["slot_order"])
        if len(trainable) != k or trainable != sorted(set(trainable)):
            raise SelectionError(f"layer {key}: trainable must be {k} unique ascending ids")
        if any(not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < n for i in trainable):
            raise SelectionError(f"layer {key}: trainable id outside [0, {n})")
        if len(order) != n or sorted(order) != list(range(n)):
            raise SelectionError(f"layer {key}: slot_order must be a permutation of range({n})")
        selected = set(trainable)
        placed = set()
        for rank in range(ep):
            local = order[rank * n_local:(rank + 1) * n_local]
            head, tail = local[:k_local], local[k_local:]
            if any(i not in selected for i in head):
                raise SelectionError(f"layer {key}: EP rank {rank} has a frozen expert in a trainable slot")
            if any(i in selected for i in tail):
                raise SelectionError(f"layer {key}: EP rank {rank} has a trainable expert in a frozen slot")
            placed.update(head)
        if placed != selected:
            raise SelectionError(f"layer {key}: trainable slots do not cover the trainable set")
    try:
        sha = require_sha256(doc["sha256"], "selection sha256")
        trainable_sha = (require_sha256(doc["trainable_sha256"], "trainable_sha256")
                         if "trainable_sha256" in doc else None)
    except ValueError as exc:
        raise SelectionError(str(exc)) from None
    actual = canonical_layers_sha256({str(int(key)): layer for key, layer in layers.items()})
    if sha != actual:
        raise SelectionError(f"selection sha256 {sha} does not match its layers ({actual})")
    if trainable_sha is not None:
        actual_t = canonical_sha256({str(int(key)): list(layer["trainable"]) for key, layer in layers.items()})
        if trainable_sha != actual_t:
            raise SelectionError("trainable_sha256 does not match the trainable sets")
    load = doc.get("load")
    if load is not None:
        if not isinstance(load, Mapping) or set(load) != set(layers):
            raise SelectionError("load must cover exactly the selection's layers")
        for key, row in load.items():
            if len(row) != n:
                raise SelectionError(f"load of layer {key} has {len(row)} entries, expected {n}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m yeto.rl.adapters.miles.models.partial_experts.selection")
    p.add_argument("--model", default="kimi_k31", help="selection 'model' field")
    p.add_argument("--num-experts", type=int, default=896)
    p.add_argument("--num-trainable", type=int, default=128)
    p.add_argument("--ep-size", type=int, default=8)
    p.add_argument("--moe-layer-offset", type=int, default=1,
                   help="HF layer of the first MoE layer in the routed-experts array (K3.1: 1)")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--payload", nargs="+", metavar="FILE",
                     help="sample-buffer group payloads carrying rollout_routed_experts")
    src.add_argument("--counts", metavar="JSON", help="routing_plugin counts {layer: [counts]}")
    src.add_argument("--input", metavar="JSON", help="existing selection; re-derive slot_order for --ep-size")
    src.add_argument("--validate", metavar="JSON", help="validate a selection file and exit")
    src.add_argument("--mode", choices=SYNTHETIC_METHODS,
                     help="synthetic selection without routing statistics (GPU smoke)")
    p.add_argument("--num-moe-layers", type=int, default=92,
                   help="with --mode: MoE layers, HF layers moe_layer_offset .. +num_moe_layers-1")
    p.add_argument("--seed", type=int, default=0, help="with --mode random")
    p.add_argument("--prompts-sha256", default=None, help="identity of the prompt set the counts came from")
    p.add_argument("--no-load", action="store_true", help="omit the per-layer counts from the output")
    p.add_argument("--skip-layer-check", action="store_true",
                   help="do not require the layers to equal the registered model's MoE layers")
    p.add_argument("--output", "-o", metavar="JSON", help="where to write the selection")
    return p


def _check_model_layers(selection: ExpertSelection, *, moe_layer_offset: int) -> None:
    """For a registered partial-experts model, the layers must be all its MoE layers."""

    from .config import known_model_configs

    cfg = known_model_configs().get(selection.model)
    if cfg is None:
        return
    if selection.num_experts != cfg.num_routed_experts:
        raise SelectionError(f"{selection.model}: num_experts {selection.num_experts} != {cfg.num_routed_experts}")
    selection.check_layers(range(moe_layer_offset, moe_layer_offset + cfg.num_moe_layers))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.validate:
        try:
            selection = ExpertSelection.load_file(args.validate)
        except SelectionError as exc:
            print(f"invalid: {exc}", file=sys.stderr)
            return 1
        print(json.dumps({"sha256": selection.sha256, "trainable_sha256": selection.trainable_sha256,
                          "layers": len(selection.trainable), "ep_size": selection.ep_size}))
        return 0
    if args.input:
        selection = ExpertSelection.load_file(args.input).with_ep_size(args.ep_size)
    elif args.mode:
        selection = ExpertSelection.synthetic(
            args.mode, model=args.model, num_experts=args.num_experts, num_trainable=args.num_trainable,
            ep_size=args.ep_size, seed=args.seed,
            layers=range(args.moe_layer_offset, args.moe_layer_offset + args.num_moe_layers),
        )
    else:
        source: dict[str, Any] = {}
        if args.prompts_sha256:
            source["prompts_sha256"] = require_sha256(args.prompts_sha256, "--prompts-sha256")
        if args.payload:
            counts, tokens = counts_from_payloads(
                args.payload, num_experts=args.num_experts, moe_layer_offset=args.moe_layer_offset
            )
            source["tokens"] = tokens
        elif args.counts:
            counts = counts_from_routing_plugin(
                json.loads(Path(args.counts).read_text()), num_experts=args.num_experts
            )
            source["tokens"] = int(max(sum(row) for row in counts.values()))
            source["counter"] = "routing_plugin"
        else:
            _parser().error("one of --payload, --counts, --input, --mode or --validate is required")
        selection = ExpertSelection.from_counts(
            counts, model=args.model, num_trainable=args.num_trainable, ep_size=args.ep_size,
            source=source, keep_load=not args.no_load,
        )
    if not args.skip_layer_check:
        _check_model_layers(selection, moe_layer_offset=args.moe_layer_offset)
    doc = selection.to_json()
    if args.output:
        selection.write(args.output)
    else:
        json.dump(doc, sys.stdout, indent=1, sort_keys=True)
        print()
    summary = {"sha256": selection.sha256, "trainable_sha256": selection.trainable_sha256,
               "layers": len(selection.trainable), "ep_size": selection.ep_size}
    if selection.load is not None:
        coverage = selection.coverage()
        summary["coverage_mean"] = round(coverage["mean"], 6)
        summary["coverage_min"] = round(coverage["min"], 6)
    print(json.dumps(summary), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
