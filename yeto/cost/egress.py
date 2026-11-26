"""Network egress of a Modal trainer: what it sends out of Modal, from its event tapes.

Modal bills every GiB a container sends out (``host.modal.egress_gib`` of the price
table) next to its GPU, CPU and memory. A trainer that publishes its weights to an
object store outside Modal (a buffer trainer's releases, read by samplers elsewhere)
or pushes fragments to a syncer sends a lot: on 1xL40S verl trainer apps publishing a
0.6B policy every round, egress was about as much as GPU, CPU and memory together.

The trainer's tape names those bytes: ``rl_publication`` / ``rl_member_publication``
(``sync/publication_payload_bytes``: the release it published) and ``rl_fragment_push``
(``wire_bytes``: the fragment on the wire to the syncer). The launcher records on the
trainer's Modal app (ledger attrs ``egress_events``, and ``egress_islands``: the island ids
its own records carry, :func:`tape_island_ids`) which of them leave Modal for its launch
(:func:`egress_events`): a publication only when the run's store is outside
Modal and the trainer publishes releases to it, a fragment push always (its syncer is
never in the trainer's container). :mod:`.billing` counts those bytes after the app's newest billed hour (the billed
hours carry the egress Modal billed).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

PUBLICATION_EVENTS = ("rl_publication", "rl_member_publication")
FRAGMENT_EVENTS = ("rl_fragment_push",)
BYTES_FIELD = {"rl_publication": "sync/publication_payload_bytes",
               "rl_member_publication": "sync/publication_payload_bytes",
               "rl_fragment_push": "wire_bytes"}
# a store URI that is inside Modal (no egress to it); a scheme-less path is the container's own disk
INSIDE_MODAL_SCHEMES = ("modal-volume://", "modal://")
GIB = float(1 << 30)
# Modal bills the bytes on the wire: TCP / TLS / HTTP framing on top of the payload a
# record names (the publication bytes of 1xL40S verl trainer apps came to 0.73-1.02x
# their billed egress line; 5 % keeps the estimate on the high side)
WIRE_OVERHEAD = 1.05


@dataclass(frozen=True)
class Sample:
    t: float
    event: str
    bytes: float | None  # None: the record named no size (the fallback prices it)
    island: int | None = None  # the record's island_id (None: it named none)


def store_outside_modal(store: Any) -> bool:
    """A store URI that is not on Modal: an object store (``s3://``, ``gs://``, ``r2://``,
    ``http(s)://`` ...); a Modal Volume or a plain path is not."""
    text = str(store or "").strip()
    return "://" in text and not text.startswith(INSIDE_MODAL_SCHEMES)


def egress_events(args: Any) -> list[str]:
    """The tape events whose bytes leave Modal for a Modal trainer launched with ``args``:
    publications when its ``--rl-checkpoint-store`` is outside Modal and it publishes
    releases there (``--rl-buffer-shards`` / ``--rl-island-buffer-shards``: samplers
    elsewhere read them); fragment pushes (to a syncer) always."""
    g = lambda name: getattr(args, name, None)  # noqa: E731
    out: list[str] = []
    if store_outside_modal(g("rl_checkpoint_store")) and (g("rl_buffer_shards") or g("rl_island_buffer_shards")):
        out += PUBLICATION_EVENTS
    # a fragment always goes to a syncer outside the trainer's container (counted even when
    # that syncer is a Modal container: high, never low)
    out += FRAGMENT_EVENTS
    return out


def tape_island_ids(args: Any, launch_islands: Iterable[int]) -> list[int]:
    """The ``island_id`` the tape records of a launch's Modal islands carry: a fleet island
    launch's ``--rl-fleet-learner-id`` (its one island), else each island's index in the
    launch (its ``--rl-learner-id``). The launch's other islands (on other clouds) write
    other ids: their bytes are not this app's egress."""
    from yeto.rl.engine.sample_roles import fleet_learner  # noqa: PLC0415

    fleet = fleet_learner(args)
    return [fleet[0]] if fleet is not None else sorted({int(m) for m in launch_islands})


def fallback_bytes(args: Any) -> float | None:
    """What a publication without a stated size counts: the model's parameters at 4 bytes
    each (fp32, the widest a publication carries; a bf16 release counts double: high,
    never low). None: the model's size is unknown."""
    class _Offline:  # the known-model table and --weights-gb only: never a Hub query at launch
        @staticmethod
        def get_or(key, fetch):
            raise LookupError(key)

    try:
        from yeto.shape.memory import model_weights_gb  # noqa: PLC0415

        bf16_gb = model_weights_gb(getattr(args, "model", None), getattr(args, "weights_gb", None), cache=_Offline())
    except Exception:  # noqa: BLE001 - an unknown model: no fallback
        return None
    return None if not bf16_gb else float(bf16_gb) * 1e9 * 2


@dataclass
class EgressMeter:
    """The egress-relevant records of a run's tapes, each once (a record mirrored to two
    tapes counts once)."""

    samples: list[Sample] = field(default_factory=list)
    _seen: set[tuple[Any, ...]] = field(default_factory=set)

    def observe(self, rec: dict[str, Any], t: float) -> None:
        event = rec.get("event")
        if event not in BYTES_FIELD:
            return
        raw = rec.get(BYTES_FIELD[event])
        size = float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw > 0 else None
        key = (event, rec.get("island_id"), rec.get("policy_version"), rec.get("fragment_id"),
               rec.get("sync/publication_payload_hash"), rec.get("time_unix"), size)
        if key in self._seen:
            return
        self._seen.add(key)
        island = rec.get("island_id")
        island = int(island) if isinstance(island, int) and not isinstance(island, bool) else None
        self.samples.append(Sample(float(t), str(event), size, island))

    def feed(self, records: Iterable[tuple[str, dict[str, Any]]]) -> "EgressMeter":
        from .guard import record_time  # noqa: PLC0415

        for _path, rec in records:
            self.observe(rec, record_time(rec, 0.0))
        return self


def bytes_in(samples: Sequence[Sample], events: Iterable[str], start: float, end: float,
             fallback: float | None, islands: Iterable[int] | None = None) -> tuple[dict[Sample, float], int]:
    """``({sample: bytes on the wire}, unsized)`` of the ``events`` samples in ``[start, end)``
    of ``islands`` (None: any; a record naming no island counts: high, never low), each size x
    :data:`WIRE_OVERHEAD`; a sample without a size counts the largest size seen of its event,
    else ``fallback``; ``unsized``: samples neither prices."""
    wanted = set(events)
    mine = None if islands is None else {int(i) for i in islands}
    largest: dict[str, float] = {}
    out: dict[Sample, float] = {}
    unsized = 0
    for s in sorted(samples, key=lambda s: s.t):
        if s.event not in wanted or (mine is not None and s.island is not None and s.island not in mine):
            continue
        if s.bytes is not None:
            largest[s.event] = max(largest.get(s.event, 0.0), s.bytes)
        size = s.bytes if s.bytes is not None else largest.get(s.event, fallback)
        if not start <= s.t < end:
            continue
        if size is None:
            unsized += 1
            continue
        out[s] = out.get(s, 0.0) + size * WIRE_OVERHEAD
    return out, unsized
