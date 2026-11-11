"""The trainer's fragment layout as a JSON record.

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
                  attempts: int = 30, retry_s: float = 10.0, watch_s: float = 30.0,
                  board_factory: Any = None, sleep: Any = time.sleep,
                  stop: threading.Event | None = None) -> threading.Thread | None:
    """Island 0 of a run with spot subscribers (:data:`PUBLISH_ENV`): set the
    layout board on a daemon thread (retrying while the relay is not
    reachable); never blocks or fails the training island. None: nothing to do.

    The board lives in the syncer's memory only: a restarted syncer (``--resume``)
    comes back with an empty board, and a sampler that starts after it waits for the
    layout forever while the trainer waits for its groups. So after the first set
    the thread reads the board every ``watch_s`` and sets it again when it is empty
    (``attempts`` bounds the consecutive failures, not the run)."""
    env = os.environ if environ is None else environ
    if env.get(PUBLISH_ENV) != "1" or int(learner_id) != 0:
        return None
    data = encode(layout, meta)
    stop = stop or threading.Event()

    def make_board() -> Any:
        if board_factory is not None:
            return board_factory()
        from .relay import RelayBoard

        return RelayBoard(addr, LAYOUT_BOARD, int(learner_id))

    def run() -> None:
        from .relay import RelayError, RelayRefused

        failures, published = 0, False
        while not stop.is_set() and failures < int(attempts):
            board = make_board()
            try:
                if published and board.get() is not None:
                    failures = 0
                    sleep(float(watch_s))
                    continue
                item = board.set(data)
                print(f"[yeto-spot] layout board {'set again (it was empty: a restarted syncer)' if published else 'set'} "
                      f"(id {item}, {layout.num_fragments} fragments, {len(data)} bytes)", flush=True)
                published, failures = True, 0
                sleep(float(watch_s))
            except RelayRefused as error:
                print(f"[yeto-spot] layout board refused: {error}", flush=True)
                return
            except (RelayError, OSError) as error:
                failures += 1
                print(f"[yeto-spot] layout board attempt {failures} failed: {error}", flush=True)
                sleep(float(retry_s))
            finally:
                board.close()

    thread = threading.Thread(target=run, name="yeto-spot-layout", daemon=True)
    thread.start()
    return thread
