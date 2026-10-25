"""Train-gap probe and compact R3 fill (S27 G-CT, ``SESSION27/TRAIN-GAP.md``).

Single-island Nano runs (s27-bb-b1, s27-ch-d1b, s27-ch-d0a) spend 12-15 s per
round in each trainer rank between Miles ``_train_actor`` start and its
``train`` timer; E1 (s27-cb-e1) spends 0.3-0.5 s there with the same Miles
commit, argv and batch shape.  That window holds ``get_data_iterator`` (no
work under the rollout-side schedule) and the routing-replay (R3)
``fill_replay_data``.  The logs do not say why the fill is slow.  In b1 the
gap fell to 0.4-0.55 s for the 5-7 rounds after each checkpoint cut, while
the cut writer processes kept the trainer node busy; the leading guess is the
wake-up cost of the many small multi-threaded (OpenMP) CPU tensor ops of the
legacy fill on an otherwise idle gVisor node (not verified).

This module runs in every Miles actor process (installed with the per-step
recorders, ``state_plugin.install_actor_recorders``).  It replaces the two
names in ``miles.backends.megatron_utils.actor`` with wrappers:

* ``get_data_iterator``: timed (``iter_s``);
* ``fill_replay_data``: before the fill, one read per memory page of every
  source replay array (``touch_s``: the first-touch cost of the zero-copy
  object-store arrays); then the fill, with the time inside the
  ``register_replay_list_func`` calls (``register_s``: the pinned copies per
  replay stream) and the rest (``prepare_s``: the per-sample and
  per-micro-batch pad/concatenate copies); ``getrusage`` deltas (CPU time,
  page faults, context switches) for the touch and for the fill.

Each fill prints one ``[yeto-train-gap] {json}`` line per rank (8 lines a
round on one 8-GPU trainer).  ``YETO_TRAIN_GAP_PROBE=0`` turns the probe off.

``YETO_REPLAY_FILL`` selects the fill itself:

* ``legacy`` (default): Miles' own ``fill_replay_data``;
* ``compact``: :func:`compact_fill_replay_data` -- per micro-batch one
  stream-major buffer (pinned when CUDA is available) with per-stream row
  copies from the source arrays and -1 in the padding rows only; each replay
  stream records a view of it.  No per-sample or per-micro-batch intermediate
  tensor is allocated, and every copy is below torch's parallel grain size
  (one thread, no OpenMP wake-up).  The recorded values, shapes and dtypes
  equal the legacy fill (tests pin this).  Used only for the case it covers
  (THD, context parallel 1, sequential stream registration, routing indices);
  any other call runs the legacy fill;
* ``serial``: the legacy fill with ``torch.set_num_threads(1)`` for the call
  (separates the OpenMP cost from the copy and allocation volume);
* ``ab``: ``compact`` on odd rollout ids, ``legacy`` on even ones;
* ``cycle``: ``legacy``, ``compact``, ``serial`` by rollout id modulo 3.

``ab`` and ``cycle`` measure the fills in one run under the same host.

``YETO_REPLAY_FILL_CHECK=K`` (default 0, needs the probe): the first K compact
fills of each process are checked on the real batch.  After the timed compact
fill, Miles' own fill runs on the same micro-batches into scratch replays, and
every recorded tensor is compared (dtype, shape, contiguity, values).  The
result goes into the record as ``check``; its time is not in ``fill_s``.  On a
mismatch (or a check error) the legacy tensors replace the compact ones for
that step and compact is off for the rest of the process.
"""

from __future__ import annotations

import functools
import json
import os
import resource
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

PROBE_ENV = "YETO_TRAIN_GAP_PROBE"
FILL_ENV = "YETO_REPLAY_FILL"
CHECK_ENV = "YETO_REPLAY_FILL_CHECK"
FILL_MODES = ("legacy", "compact", "serial", "ab", "cycle")
LOG_PREFIX = "[yeto-train-gap]"
PAGE_BYTES = 4096
SOURCE_MAP_EVERY = 25  # rounds between two /proc/self/maps lookups

_OFF = {"0", "false", "no", "off"}
_STATE: dict[str, Any] = {"installed": False, "actor": None, "mode": "legacy", "probe": True,
                          "iter_s": None, "fills": 0, "check": 0, "checked": 0, "compact_off": None}


class TrainGapConfigError(ValueError):
    """A bad ``YETO_REPLAY_FILL`` value (the run must not silently use another fill)."""


def probe_enabled(env: Mapping[str, str] = os.environ) -> bool:
    return env.get(PROBE_ENV, "1").strip().lower() not in _OFF


def fill_mode(env: Mapping[str, str] = os.environ) -> str:
    mode = (env.get(FILL_ENV, "") or "legacy").strip().lower()
    if mode not in FILL_MODES:
        raise TrainGapConfigError(f"{FILL_ENV}={mode!r} is not one of {', '.join(FILL_MODES)}")
    return mode


def check_count(env: Mapping[str, str] = os.environ) -> int:
    """How many compact fills of this process get the shadow check (``YETO_REPLAY_FILL_CHECK``)."""
    raw = (env.get(CHECK_ENV, "") or "0").strip()
    try:
        count = int(raw)
    except ValueError:
        count = -1
    if count < 0:
        raise TrainGapConfigError(f"{CHECK_ENV}={raw!r} is not a count >= 0")
    return count


def fill_for_round(mode: str, rollout_id: int | None) -> str:
    """The fill of one call: ``ab`` runs ``compact`` on odd rollout ids;
    ``cycle`` runs legacy, compact, serial by rollout id modulo 3."""
    if mode == "ab":
        return "compact" if rollout_id is not None and rollout_id % 2 == 1 else "legacy"
    if mode == "cycle":
        return ("legacy", "compact", "serial")[rollout_id % 3] if rollout_id is not None else "legacy"
    return mode


# --------------------------------------------------------------------------
# measurements
# --------------------------------------------------------------------------


def touch_pages(tensors: Sequence[Any]) -> tuple[int, int]:
    """Read one element per memory page of every contiguous CPU tensor.

    The first read of a page of a zero-copy object-store array maps it into
    this process; the time of this pass is that cost.  Returns
    ``(bytes, pages read)``; a non-contiguous or non-CPU tensor is skipped.
    """
    import torch

    total = pages = 0
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
            continue
        if tensor.numel() == 0 or not tensor.is_contiguous():
            continue
        step = max(1, PAGE_BYTES // tensor.element_size())
        sample = tensor.reshape(-1)[::step]
        sample.sum()
        total += tensor.numel() * tensor.element_size()
        pages += sample.numel()
    return total, pages


def _usage() -> tuple[float, float, int, int, int, int]:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return (r.ru_utime, r.ru_stime, r.ru_minflt, r.ru_majflt, r.ru_nvcsw, r.ru_nivcsw)


def usage_delta(before: Sequence[float], after: Sequence[float]) -> dict[str, float | int]:
    """``getrusage`` (this process, all threads) between two :func:`_usage` reads."""
    return {
        "utime_s": round(after[0] - before[0], 3),
        "stime_s": round(after[1] - before[1], 3),
        "minflt": int(after[2] - before[2]),
        "majflt": int(after[3] - before[3]),
        "nvcsw": int(after[4] - before[4]),
        "nivcsw": int(after[5] - before[5]),
    }


def node_memory_gib(path: str = "/proc/meminfo") -> dict[str, float | None]:
    """MemAvailable and Shmem of this node (GiB), or None when unreadable."""
    wanted = {"MemAvailable": "memavail_gib", "Shmem": "shmem_gib"}
    out: dict[str, float | None] = dict.fromkeys(wanted.values())
    try:
        with open(path) as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                if key in wanted:
                    out[wanted[key]] = round(int(rest.split()[0]) / (1 << 20), 2)
    except (OSError, ValueError, IndexError):
        pass
    return out


def mapping_of(address: int, path: str = "/proc/self/maps") -> str | None:
    """The ``/proc/self/maps`` path (or ``[anon]``) of the mapping that holds ``address``."""
    try:
        with open(path) as handle:
            for line in handle:
                fields = line.split(None, 5)
                start, _, end = fields[0].partition("-")
                if int(start, 16) <= address < int(end, 16):
                    name = fields[5].strip() if len(fields) > 5 else ""
                    return name or "[anon]"
    except (OSError, ValueError, IndexError):
        return None
    return None


def _rank() -> int | None:
    torch = sys.modules.get("torch")
    try:
        distributed = torch.distributed if torch is not None else None
        if distributed is not None and distributed.is_available() and distributed.is_initialized():
            return int(distributed.get_rank())
    except Exception:  # noqa: BLE001 - a probe never fails training
        return None
    return None


def _threads() -> int | None:
    torch = sys.modules.get("torch")
    try:
        return int(torch.get_num_threads()) if torch is not None else None
    except Exception:  # noqa: BLE001
        return None


class _TimedRegister:
    """``register_replay_list_func`` with its time, call count and bytes."""

    def __init__(self, inner: Callable[..., Any]) -> None:
        self.inner = inner
        self.seconds = 0.0
        self.calls = 0
        self.bytes = 0

    def __call__(self, replay_list: list, replay_data: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return self.inner(replay_list, replay_data, **kwargs)
        finally:
            self.seconds += time.perf_counter() - started
            self.calls += 1
            try:
                rows, _, topk = replay_data.shape
                self.bytes += int(rows) * int(topk) * len(replay_list) * replay_data.element_size()
            except (AttributeError, TypeError, ValueError):
                pass


# --------------------------------------------------------------------------
# compact fill
# --------------------------------------------------------------------------


def compact_supported(*, args: Any, parallel_state: Any, call: Mapping[str, Any],
                      sequential: Callable[..., Any] | None) -> str | None:
    """None when :func:`compact_fill_replay_data` covers this call, else why not."""
    if getattr(args, "qkv_format", "thd") != "thd":
        return "qkv_format is not thd"
    if int(parallel_state.cp.size) != 1:
        return "context parallel > 1"
    if call.get("indices_are_token_positions"):
        return "token-position indices"
    if sequential is None or call.get("register_replay_list_func") is not sequential:
        return "stream registration is not register_replay_list_sequential"
    return None


def _stream_indices(replay_list: Sequence[Any], num_streams: int) -> list[int]:
    streams = []
    for index, replay in enumerate(replay_list):
        stream = replay.stream_idx if getattr(replay, "stream_idx", None) is not None else index
        if not 0 <= stream < num_streams:
            raise AssertionError(
                f"replay stream_idx {stream} out of range (replay_data has {num_streams} streams)")
        streams.append(stream)
    return streams


def _append_recorded(replay: Any, view: Any) -> None:
    """Record ``view`` without a copy (Miles ``Replay`` keeps ``top_indices_list``)."""
    recorded = getattr(replay, "top_indices_list", None)
    if isinstance(recorded, list):
        recorded.append(view)
    else:
        replay.record(view)


def _record_micro_batch(samples: Sequence[Any], tokens: Sequence[Any], replay_list: Sequence[Any], *,
                        pad_size: int, sp: tuple[int, int] | None, pin: bool,
                        stats: dict[str, Any] | None) -> None:
    """One micro-batch: the legacy layout (each sample's rows, one -1 row, then
    -1 rows up to a multiple of ``pad_size``; the ``sp`` = (rank, size) window
    of it), written straight into one stream-major buffer."""
    import torch

    assert len(samples) == len(tokens)
    for sample, token in zip(samples, tokens, strict=True):
        assert sample.shape[0] == token.shape[0] - 1, f"{sample.shape}, {token.shape}"
    num_streams, topk = int(samples[0].shape[1]), int(samples[0].shape[2])
    streams = _stream_indices(replay_list, num_streams)
    rows = sum(int(sample.shape[0]) + 1 for sample in samples)
    padded = rows + (pad_size - rows % pad_size) % pad_size
    start, end = 0, padded
    if sp is not None:
        rank, size = sp
        assert padded % size == 0
        start, end = padded // size * rank, padded // size * (rank + 1)
    out = torch.empty((len(streams), end - start, topk), dtype=samples[0].dtype, pin_memory=pin)
    offset = 0
    for sample in samples:
        n = int(sample.shape[0])
        low, high = max(offset, start), min(offset + n, end)
        if low < high:
            for slot, stream in enumerate(streams):
                out[slot, low - start:high - start].copy_(sample[low - offset:high - offset, stream])
        if start <= offset + n < end:
            out[:, offset + n - start].fill_(-1)  # the sample's padding row
        offset += n + 1
    if max(rows, start) < end:
        out[:, max(rows, start) - start:].fill_(-1)  # the micro-batch padding rows
    started = time.perf_counter()
    for slot, replay in enumerate(replay_list):
        _append_recorded(replay, out[slot])
    if stats is not None:
        stats["register_s"] = stats.get("register_s", 0.0) + time.perf_counter() - started
        stats["mbs"] = stats.get("mbs", 0) + 1
        stats["rec_bytes"] = stats.get("rec_bytes", 0) + out.numel() * out.element_size()


def compact_fill_replay_data(*, args: Any, models: Any, data_iterator: Sequence[Any],
                             num_microbatches: Sequence[int], rollout_data: dict, data_key: str,
                             replay_list: list, register_replay_list_func: Any,
                             if_sp_region: bool = True, indices_are_token_positions: bool = False,
                             parallel_state: Any, pin: bool | None = None,
                             stats: dict[str, Any] | None = None) -> None:
    """Miles ``fill_replay_data`` for THD, context parallel 1, sequential
    stream registration and routing indices (:func:`compact_supported`).

    Same micro-batch order, padding (-1) and sequence-parallel window as the
    legacy fill; each replay stream records a contiguous view of one buffer per
    micro-batch instead of its own pinned copy.
    """
    import torch

    del models, register_replay_list_func
    if indices_are_token_positions:
        raise ValueError("compact replay fill does not rebase token-position indices")
    if data_key not in rollout_data:
        raise ValueError(f"{data_key} is required in rollout_data for replay.")
    for iterator in data_iterator:
        iterator.reset()
    pin = torch.cuda.is_available() if pin is None else pin
    tp_rank, tp_size = int(parallel_state.tp.rank), int(parallel_state.tp.size)
    pad_size = tp_size * int(args.data_pad_size_multiplier)
    sp = (tp_rank, tp_size) if getattr(args, "sequence_parallel", False) and if_sp_region else None
    for _ in range(sum(num_microbatches)):
        batch = data_iterator[0].get_next([data_key, "tokens", "max_seq_lens"])
        _record_micro_batch(batch[data_key], batch["tokens"], replay_list,
                            pad_size=pad_size, sp=sp, pin=pin, stats=stats)
    del rollout_data[data_key]
    for iterator in data_iterator:
        iterator.reset()


class _ShadowReplay:
    """A scratch replay stream with Miles ``Replay.record`` semantics (pinned copy)."""

    def __init__(self, stream_idx: int | None, pin: bool) -> None:
        self.stream_idx = stream_idx
        self.pin = pin
        self.top_indices_list: list[Any] = []

    def record(self, top_indices: Any) -> None:
        import torch

        buf = torch.empty_like(top_indices, device="cpu", pin_memory=self.pin)
        buf.copy_(top_indices)
        self.top_indices_list.append(buf)


def shadow_check(*, original: Callable[..., Any], sequential: Callable[..., Any], call: Mapping[str, Any],
                 before: Sequence[int], value: Any) -> dict[str, Any]:
    """Run Miles' fill (``original``) on the batch the compact fill just recorded,
    into scratch replays, and compare every recorded tensor.

    ``before`` is the length of each replay's ``top_indices_list`` before the
    compact fill; ``value`` is ``rollout_data[data_key]`` before the fill (both
    fills delete that key; it is put back for the shadow and deleted again).
    On a mismatch the legacy tensors replace the compact ones and compact is
    turned off for the rest of the process.
    """
    import torch

    started = time.perf_counter()
    replay_list = call["replay_list"]
    rollout_data, key = call["rollout_data"], call["data_key"]
    shadows = [_ShadowReplay(getattr(replay, "stream_idx", None), torch.cuda.is_available())
               for replay in replay_list]
    rollout_data[key] = value
    try:
        original(**dict(call, replay_list=shadows, register_replay_list_func=sequential))
    finally:
        rollout_data.pop(key, None)
        for iterator in call["data_iterator"]:
            iterator.reset()
    mismatch, compared = None, 0
    for slot, (replay, shadow) in enumerate(zip(replay_list, shadows, strict=True)):
        got, want = replay.top_indices_list[before[slot]:], shadow.top_indices_list
        if len(got) != len(want):
            mismatch = f"slot {slot}: {len(got)} micro-batches, legacy {len(want)}"
            break
        for index, (a, b) in enumerate(zip(got, want, strict=True)):
            compared += 1
            if (a.dtype != b.dtype or a.shape != b.shape or a.is_contiguous() != b.is_contiguous()
                    or not torch.equal(a, b)):
                mismatch = (f"slot {slot} micro-batch {index}: compact {a.dtype}{tuple(a.shape)} "
                            f"legacy {b.dtype}{tuple(b.shape)}")
                break
        if mismatch is not None:
            break
    result: dict[str, Any] = {"equal": mismatch is None, "tensors": compared,
                              "s": round(time.perf_counter() - started, 3)}
    if mismatch is not None:
        result["mismatch"] = mismatch
        for slot, (replay, shadow) in enumerate(zip(replay_list, shadows, strict=True)):
            del replay.top_indices_list[before[slot]:]
            replay.top_indices_list.extend(shadow.top_indices_list)
        _STATE["compact_off"] = f"shadow check mismatch ({mismatch})"
        print(f"{LOG_PREFIX} CHECK MISMATCH: {mismatch}; legacy tensors used, compact off", flush=True)
    return result


# --------------------------------------------------------------------------
# wrappers and install
# --------------------------------------------------------------------------


def _timed_iterator(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def get_data_iterator(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            _STATE["iter_s"] = time.perf_counter() - started

    get_data_iterator.__yeto_train_gap__ = True  # type: ignore[attr-defined]
    return get_data_iterator


def _source_map(sources: Sequence[Any]) -> str | None:
    if not sources:
        return None
    try:
        return mapping_of(int(sources[0].data_ptr()))
    except Exception:  # noqa: BLE001
        return None


def _probed_fill(original: Callable[..., Any], *, sequential: Callable[..., Any] | None,
                 parallel_state_fn: Callable[[], Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def fill_replay_data(**call: Any) -> Any:
        rollout_id = getattr(_STATE["actor"], "_last_rollout_id", None)
        mode = fill_for_round(_STATE["mode"], rollout_id)
        probe = _STATE["probe"]
        fill, reason = original, None
        parallel_state = None
        if mode == "serial":
            fill = functools.partial(_serial, original)
        elif mode == "compact":
            reason = _STATE["compact_off"]
            if reason is None:
                parallel_state = parallel_state_fn()
                reason = compact_supported(args=call["args"], parallel_state=parallel_state,
                                           call=call, sequential=sequential)
            if reason is None:
                fill = functools.partial(compact_fill_replay_data, parallel_state=parallel_state)
            else:
                mode = "legacy"
        if not probe:
            return fill(**call)
        check = None
        if mode == "compact" and _STATE["checked"] < _STATE["check"]:
            _STATE["checked"] += 1
            check = functools.partial(shadow_check, original=original, sequential=sequential)
        return _measured(fill, call, mode=mode, reason=reason, rollout_id=rollout_id,
                         compact=mode == "compact", check=check)

    fill_replay_data.__yeto_train_gap__ = True  # type: ignore[attr-defined]
    return fill_replay_data


def _serial(original: Callable[..., Any], **call: Any) -> Any:
    """``original`` with one torch intra-op thread for the call."""
    import torch

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return original(**call)
    finally:
        torch.set_num_threads(threads)


def _measured(fill: Callable[..., Any], call: dict[str, Any], *, mode: str, reason: str | None,
              rollout_id: int | None, compact: bool, check: Callable[..., Any] | None = None) -> Any:
    sources = list(call["rollout_data"].get(call["data_key"]) or ())
    shadow = None
    if check is not None:
        shadow = {"call": dict(call), "value": call["rollout_data"].get(call["data_key"]),
                  "before": [len(getattr(replay, "top_indices_list", ())) for replay in call["replay_list"]]}
    _STATE["fills"] += 1
    source_map = _source_map(sources) if _STATE["fills"] % SOURCE_MAP_EVERY == 1 else None
    memory = node_memory_gib()
    stats: dict[str, Any] = {}
    register = None
    if compact:
        call = dict(call, stats=stats)
    else:
        register = _TimedRegister(call["register_replay_list_func"])
        call = dict(call, register_replay_list_func=register)
    u0, t0 = _usage(), time.perf_counter()
    src_bytes, pages = touch_pages(sources)
    u1, t1 = _usage(), time.perf_counter()
    failed = True
    try:
        result = fill(**call)
        failed = False
        return result
    finally:
        u2, t2 = _usage(), time.perf_counter()
        register_s = register.seconds if register is not None else stats.get("register_s", 0.0)
        record = {
            "rollout": rollout_id, "rank": _rank(), "key": call["data_key"], "mode": mode,
            "iter_s": None if _STATE["iter_s"] is None else round(_STATE["iter_s"], 3),
            "touch_s": round(t1 - t0, 3), "fill_s": round(t2 - t1, 3),
            "register_s": round(register_s, 3), "prepare_s": round(t2 - t1 - register_s, 3),
            "mbs": register.calls if register is not None else stats.get("mbs", 0),
            "samples": len(sources), "pages": pages,
            "src_mib": round(src_bytes / (1 << 20), 1),
            "rec_mib": round((register.bytes if register is not None else stats.get("rec_bytes", 0))
                             / (1 << 20), 1),
            "threads": 1 if mode == "serial" else _threads(),
            "touch": usage_delta(u0, u1), "fill": usage_delta(u1, u2),
            **memory,
        }
        if reason is not None:
            record["compact_refused"] = reason
        if source_map is not None:
            record["src_map"] = source_map
        if shadow is not None and not failed:
            try:
                record["check"] = check(**shadow)
            except Exception as error:  # noqa: BLE001 - compact goes off; training goes on
                _STATE["compact_off"] = f"shadow check error ({type(error).__name__})"
                record["check"] = {"error": f"{type(error).__name__}: {error}"[:300]}
        print(f"{LOG_PREFIX} {json.dumps(record, sort_keys=True)}", flush=True)


def _miles_targets() -> tuple[Any, Callable[..., Any] | None, Callable[[], Any]]:
    from miles.backends.megatron_utils import actor as actor_module
    from miles.backends.training_utils.parallel import get_parallel_state
    from miles.backends.training_utils.replay_data import register_replay_list_sequential

    return actor_module, register_replay_list_sequential, get_parallel_state


def install(actor: Any, *, env: Mapping[str, str] = os.environ,
            targets: Callable[[], tuple[Any, Any, Any]] = _miles_targets) -> bool:
    """Wrap ``get_data_iterator`` and ``fill_replay_data`` of the Miles actor
    module in this process (idempotent).  False when neither the probe nor a
    non-legacy fill is on.  A bad ``YETO_REPLAY_FILL`` raises
    :class:`TrainGapConfigError`."""
    mode = fill_mode(env)
    probe = probe_enabled(env)
    _STATE.update(actor=actor, mode=mode, probe=probe, check=check_count(env))
    if _STATE["installed"]:
        return True
    if not probe and mode == "legacy":
        return False
    actor_module, sequential, parallel_state_fn = targets()
    if not getattr(actor_module.get_data_iterator, "__yeto_train_gap__", False):
        actor_module.get_data_iterator = _timed_iterator(actor_module.get_data_iterator)
    if not getattr(actor_module.fill_replay_data, "__yeto_train_gap__", False):
        actor_module.fill_replay_data = _probed_fill(
            actor_module.fill_replay_data, sequential=sequential, parallel_state_fn=parallel_state_fn)
    _STATE["installed"] = True
    if probe:
        from yeto.rl.host_info import print_host_report

        print_host_report(f"actor train-gap rank {_rank()} fill {mode} check {_STATE['check']}")
    return True


def install_best_effort(actor: Any, **kwargs: Any) -> None:
    """:func:`install` from the step-recorder plugin: a config error stops the
    run; any other failure only prints why (the probe never fails training)."""
    try:
        install(actor, **kwargs)
    except TrainGapConfigError:
        raise
    except Exception as error:  # noqa: BLE001
        print(f"{LOG_PREFIX} not installed: {type(error).__name__}: {error}", flush=True)
