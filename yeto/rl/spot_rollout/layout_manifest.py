"""The trainer's fragment layout as a JSON record (S27 G-EA).

A subscriber must send the exact layout of the session (the syncer compares
it and its fingerprint) and needs the tensor names to turn fragment bytes
into weights. A sampler island has no Megatron model to build the layout
from, so training island 0 publishes it on a relay board
(:class:`~yeto.rl.spot_rollout.relay.RelayBoard`) and the sampler reads it.

Record (``yeto.rl.spot-layout/v1``)::

    {"schema", "layout_fingerprint": hex, "fragments": [
        {"merge_mode": int, "tensors": [[name, numel], ...],
         "shapes": {name: [rows, cols]} | null,
         "identity_shapes": {name: [dims...]} | null}, ...],
     "meta": {...}}

``meta`` carries what the materializer needs besides the layout (the stream
role, the trainer's parallel sizes); it is not part of the fingerprint.
"""

from __future__ import annotations

import json
import os
import threading
import time
import zlib
from collections.abc import Mapping
from typing import Any

from yeto.fragments import Fragment, FragmentLayout
from yeto.protocol import layout_fingerprint

SCHEMA = "yeto.rl.spot-layout/v1"
# Relay board that carries the layout of the actor stream.
LAYOUT_BOARD = 1


class LayoutManifestError(ValueError):
    pass


def layout_record(layout: FragmentLayout, meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
    fragments = []
    for fragment in layout.fragments:
        fragments.append({
            "merge_mode": int(fragment.merge_mode),
            "tensors": [[str(name), int(numel)] for name, numel in fragment.tensors],
            "shapes": None if fragment.shapes is None else
            {str(k): [int(d) for d in v] for k, v in fragment.shapes.items()},
            "identity_shapes": None if fragment.identity_shapes is None else
            {str(k): [int(d) for d in v] for k, v in fragment.identity_shapes.items()},
        })
    return {"schema": SCHEMA, "layout_fingerprint": layout_fingerprint(layout).hex(),
            "fragments": fragments, "meta": dict(meta or {})}


def layout_from_record(record: Mapping[str, Any]) -> FragmentLayout:
    if record.get("schema") != SCHEMA:
        raise LayoutManifestError(f"layout schema {record.get('schema')!r} != {SCHEMA}")
    fragments = []
    for item in record["fragments"]:
        shapes = item.get("shapes")
        identity = item.get("identity_shapes")
        fragments.append(Fragment(
            merge_mode=int(item["merge_mode"]),
            tensors=[(str(name), int(numel)) for name, numel in item["tensors"]],
            shapes=None if shapes is None else {k: tuple(int(d) for d in v) for k, v in shapes.items()},
            identity_shapes=None if identity is None else
            {k: tuple(int(d) for d in v) for k, v in identity.items()},
        ))
    layout = FragmentLayout(fragments)
    want = str(record.get("layout_fingerprint") or "")
    got = layout_fingerprint(layout).hex()
    if want != got:
        raise LayoutManifestError(f"layout fingerprint {got} != recorded {want}")
    return layout


def encode(layout: FragmentLayout, meta: Mapping[str, Any] | None = None) -> bytes:
    """Compressed record bytes (a board value)."""
    return zlib.compress(json.dumps(layout_record(layout, meta), sort_keys=True).encode(), 6)


def decode(data: bytes) -> tuple[FragmentLayout, dict[str, Any]]:
    try:
        record = json.loads(zlib.decompress(data))
    except (zlib.error, ValueError) as error:
        raise LayoutManifestError(f"layout record does not decode: {error}") from error
    return layout_from_record(record), dict(record.get("meta") or {})


def publish(board: Any, layout: FragmentLayout, meta: Mapping[str, Any] | None = None) -> int:
    """Set the layout board (``RelayBoard.set``); returns the board value id."""
    return int(board.set(encode(layout, meta)))


def fetch(board: Any, *, timeout_s: float, poll_s: float = 5.0) -> tuple[FragmentLayout, dict[str, Any]]:
    """Wait for and decode the layout board (``RelayBoard.wait``)."""
    return decode(board.wait(timeout_s=timeout_s, poll_s=poll_s).data)


PUBLISH_ENV = "YETO_SPOT_PUBLISH_LAYOUT"  # "1": training island 0 publishes its layout board


def maybe_publish(addr: tuple[str, int], learner_id: int, layout: FragmentLayout,
                  meta: Mapping[str, Any] | None = None, *, environ: Mapping[str, str] | None = None,
                  attempts: int = 30, retry_s: float = 10.0) -> threading.Thread | None:
    """Island 0 of a run with spot subscribers (:data:`PUBLISH_ENV`): set the
    layout board on a daemon thread (retrying while the relay is not
    reachable); never blocks or fails the training island. None: nothing to do."""
    env = os.environ if environ is None else environ
    if env.get(PUBLISH_ENV) != "1" or int(learner_id) != 0:
        return None
    data = encode(layout, meta)

    def run() -> None:
        from .relay import RelayBoard, RelayError, RelayRefused

        for attempt in range(1, int(attempts) + 1):
            board = RelayBoard(addr, LAYOUT_BOARD, int(learner_id))
            try:
                item = board.set(data)
                print(f"[yeto-spot] layout board set (id {item}, {layout.num_fragments} fragments, "
                      f"{len(data)} bytes)", flush=True)
                return
            except RelayRefused as error:
                print(f"[yeto-spot] layout board refused: {error}", flush=True)
                return
            except (RelayError, OSError) as error:
                print(f"[yeto-spot] layout board attempt {attempt} failed: {error}", flush=True)
                time.sleep(float(retry_s))
            finally:
                board.close()

    thread = threading.Thread(target=run, name="yeto-spot-layout", daemon=True)
    thread.start()
    return thread
