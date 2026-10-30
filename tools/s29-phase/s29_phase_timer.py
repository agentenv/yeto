"""S29 MFU-B2: wall-clock phase timer for the verl Megatron actor (update_actor / compute_log_prob).

Off by default. Enabled by YETO_S29_PHASE_TIMER=<level> or the flag file
/data/work/s29-mfu/PHASE_TIMER_ON (its content is the level). Level 1 = coarse phases
(device-synchronised at every boundary); level 2 additionally wraps Megatron's
forward_step / backward_step and every P2PCommunicator call (per micro-batch,
synchronised), so stage compute vs pipeline wait is separable.
Each call (update_actor / compute_log_prob) appends one JSON line per rank to
$HOME/phase/phase-<host>-r<rank>.jsonl.
"""
import json
import os
import socket
import time
from contextlib import contextmanager

import torch


def _levels():
    """'1' or a comma list like '1,2': the i-th update_actor call (and the compute_log_prob
    that precedes it) runs at levels[min(i, len-1)]."""
    v = os.environ.get("YETO_S29_PHASE_TIMER")
    if v is None:
        try:
            with open("/data/work/s29-mfu/PHASE_TIMER_ON") as f:
                v = f.read().strip() or "1"
        except OSError:
            v = "0"
    try:
        return [int(x) for x in v.split(",") if x.strip()] or [0]
    except ValueError:
        return [1]


def _threads():
    """YETO_S29_TORCH_THREADS or /data/work/s29-mfu/TORCH_THREADS_ON: comma list like the
    levels; the i-th update_actor call sets torch.set_num_threads(v) when v > 0."""
    v = os.environ.get("YETO_S29_TORCH_THREADS")
    if v is None:
        try:
            with open("/data/work/s29-mfu/TORCH_THREADS_ON") as f:
                v = f.read().strip()
        except OSError:
            v = ""
    try:
        return [int(x) for x in v.split(",") if x.strip()]
    except ValueError:
        return []


THREADS = _threads()
LEVELS = _levels()
LEVEL = LEVELS[0]
_ua_calls = 0
_acc = {}
_lists = {}
_patched = False


def _sync():
    try:
        torch.npu.synchronize()
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def _add(name, dt, keep_list=False):
    a = _acc.setdefault(name, [0.0, 0])
    a[0] += dt
    a[1] += 1
    if keep_list:
        _lists.setdefault(name, []).append(round(dt, 4))


@contextmanager
def phase(name, level=1, keep_list=False):
    if LEVEL < level:
        yield
        return
    _sync()
    t = time.perf_counter()
    try:
        yield
    finally:
        _sync()
        _add(name, time.perf_counter() - t, keep_list)


_clocks = {}


def lap_start(clock):
    if LEVEL >= 1:
        _sync()
        _clocks[clock] = time.perf_counter()


def lap(clock, name):
    """Record the time since the previous lap/lap_start on this clock under `name`."""
    if LEVEL < 1 or clock not in _clocks:
        return
    _sync()
    now = time.perf_counter()
    _add(name, now - _clocks[clock])
    _clocks[clock] = now


def wrap(fn, name, level=2, keep_list=True):
    def inner(*a, **k):
        if LEVEL < level:
            return fn(*a, **k)
        _sync()
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            _sync()
            _add(name, time.perf_counter() - t, keep_list)

    inner.__wrapped__ = fn
    return inner


def patch_megatron(call="update_actor"):
    """Select this call's level; at level>=2 wrap schedules.forward_step/backward_step and
    the P2PCommunicator methods (once; the wrappers check LEVEL at call time)."""
    global _patched, LEVEL, _ua_calls
    LEVEL = LEVELS[min(_ua_calls, len(LEVELS) - 1)]
    if THREADS:
        th = THREADS[min(_ua_calls, len(THREADS) - 1)]
        if th > 0 and torch.get_num_threads() != th:
            torch.set_num_threads(th)
    if call == "update_actor":
        _ua_calls += 1
    if max(LEVELS) < 2 or _patched:
        return
    _patched = True
    from megatron.core.pipeline_parallel import schedules
    schedules.forward_step = wrap(schedules.forward_step, "mb.forward_step")
    schedules.backward_step = wrap(schedules.backward_step, "mb.backward_step")
    try:
        from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
        for m in ("recv_forward", "recv_backward", "send_forward", "send_backward",
                  "send_forward_recv_backward", "send_backward_recv_forward"):
            if hasattr(P2PCommunicator, m):
                setattr(P2PCommunicator, m, wrap(getattr(P2PCommunicator, m), "p2p." + m))
    except Exception as e:  # pragma: no cover
        _add("patch_error:" + repr(e)[:80], 0.0)


def flush(call, extra=None):
    if LEVEL < 1:
        return
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    rec = {"t": time.time(), "call": call, "host": socket.gethostname(), "rank": rank,
           "level": LEVEL, "acc": {k: [round(v[0], 4), v[1]] for k, v in _acc.items()}}
    try:
        from megatron.core import parallel_state as mpu
        rec["pp"] = mpu.get_pipeline_model_parallel_rank()
        rec["tp"] = mpu.get_tensor_model_parallel_rank()
        rec["dp"] = mpu.get_data_parallel_rank()
    except Exception:
        pass
    rec["torch_threads"] = torch.get_num_threads()
    rec["omp_env"] = os.environ.get("OMP_NUM_THREADS")
    if _lists:
        rec["lists"] = _lists
    if extra:
        rec.update(extra)
    d = os.path.join(os.environ.get("HOME", "/tmp"), "phase")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"phase-{rec['host']}-r{rank}.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    _acc.clear()
    _lists.clear()
