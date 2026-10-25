"""Train-gap probe and compact R3 fill (S27 G-CT, ``yeto/rl/adapters/miles/train_gap_probe.py``).

CPU only, no Ray.  The Miles equivalence tests run when Miles is importable
(``PYTHONPATH=<miles worktree>``); the rest use a hand-built reference of the
legacy layout.
"""

from __future__ import annotations

import io
import json
import random
from contextlib import redirect_stdout
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from yeto.rl.adapters.miles import train_gap_probe as probe  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_state():
    saved = dict(probe._STATE)
    probe._STATE.update(installed=False, actor=None, mode="legacy", probe=True, iter_s=None, fills=0)
    yield
    probe._STATE.clear()
    probe._STATE.update(saved)


class _Iterator:
    """The part of Miles ``DataIterator`` the fill uses (indexed micro-batches)."""

    def __init__(self, rollout_data, micro_batch_indices):
        self.rollout_data = rollout_data
        self.micro_batch_indices = micro_batch_indices
        self.offset = 0

    def get_next(self, keys):
        indices = self.micro_batch_indices[self.offset]
        self.offset += 1
        return {k: (None if self.rollout_data.get(k) is None else [self.rollout_data[k][i] for i in indices])
                for k in keys}

    def reset(self):
        self.offset = 0
        return self


class _Replay:
    def __init__(self, stream_idx=None):
        self.stream_idx = stream_idx
        self.top_indices_list = []

    def record(self, top_indices):
        self.top_indices_list.append(top_indices.clone())


def _sequential(replay_list, replay_data, **_kwargs):
    """Same contract as Miles ``register_replay_list_sequential``."""
    for index, replay in enumerate(replay_list):
        stream = replay.stream_idx if replay.stream_idx is not None else index
        if not 0 <= stream < replay_data.shape[1]:
            raise AssertionError(f"replay stream_idx {stream} out of range")
        replay.record(replay_data[:, stream])


def _reference_fill(*, args, data_iterator, num_microbatches, rollout_data, data_key, replay_list,
                    register_replay_list_func, if_sp_region=True, parallel_state, **_unused):
    """The legacy Miles fill (7402a53d5 ``replay_data.fill_replay_data``) for THD, CP 1."""
    for iterator in data_iterator:
        iterator.reset()

    def pad(data, n):
        return torch.cat([data, torch.full((n, *data.shape[1:]), -1, dtype=data.dtype)], dim=0)

    for _ in range(sum(num_microbatches)):
        batch = data_iterator[0].get_next([data_key, "tokens", "max_seq_lens"])
        data = torch.cat([pad(r, 1) for r in batch[data_key]], dim=0)
        pad_size = parallel_state.tp.size * args.data_pad_size_multiplier
        extra = (pad_size - data.size(0) % pad_size) % pad_size
        if extra:
            data = pad(data, extra)
        if args.sequence_parallel and if_sp_region:
            size = data.size(0) // parallel_state.tp.size
            data = data[size * parallel_state.tp.rank:size * (parallel_state.tp.rank + 1)]
        register_replay_list_func(replay_list, data)
    del rollout_data[data_key]
    for iterator in data_iterator:
        iterator.reset()


def _case(*, seed, samples, streams, topk, micro_batches):
    rng = random.Random(seed)
    lengths = [rng.randint(1, 40) for _ in range(samples)]
    gen = torch.Generator().manual_seed(seed)
    replay = [torch.randint(0, 128, (n, streams, topk), generator=gen, dtype=torch.int32) for n in lengths]
    tokens = [torch.zeros(n + 1, dtype=torch.long) for n in lengths]
    order = list(range(samples))
    rng.shuffle(order)
    chunks = [order[i::micro_batches] for i in range(micro_batches)]
    return {"rollout_routed_experts": replay, "tokens": tokens}, chunks


def _args(*, multiplier, sp):
    return SimpleNamespace(qkv_format="thd", data_pad_size_multiplier=multiplier, sequence_parallel=sp,
                           allgather_cp=False)


def _state(*, tp_rank=0, tp_size=1, cp_size=1):
    return SimpleNamespace(tp=SimpleNamespace(rank=tp_rank, size=tp_size),
                           cp=SimpleNamespace(rank=0, size=cp_size))


def _run(fill, *, data, chunks, args, state, stream_idx, register=_sequential, **extra):
    rollout_data = {k: list(v) for k, v in data.items()}
    replays = [_Replay(s) for s in stream_idx]
    fill(args=args, models=None, data_iterator=[_Iterator(rollout_data, chunks)],
         num_microbatches=[len(chunks)], rollout_data=rollout_data, data_key="rollout_routed_experts",
         replay_list=replays, register_replay_list_func=register, parallel_state=state, **extra)
    assert "rollout_routed_experts" not in rollout_data
    return [r.top_indices_list for r in replays]


def _assert_same(left, right):
    assert len(left) == len(right)
    for a_list, b_list in zip(left, right, strict=True):
        assert len(a_list) == len(b_list)
        for a, b in zip(a_list, b_list, strict=True):
            assert a.dtype == b.dtype and a.shape == b.shape and a.is_contiguous() and b.is_contiguous()
            assert torch.equal(a, b)


# ---------------------------------------------------------------- modes and helpers


def test_fill_mode_parses_and_rejects_unknown_values():
    assert probe.fill_mode({}) == "legacy"
    assert probe.fill_mode({probe.FILL_ENV: " AB "}) == "ab"
    with pytest.raises(probe.TrainGapConfigError):
        probe.fill_mode({probe.FILL_ENV: "fast"})
    assert probe.probe_enabled({}) and not probe.probe_enabled({probe.PROBE_ENV: "0"})


def test_ab_runs_compact_on_odd_rollouts_only_and_cycle_rotates_three_fills():
    assert [probe.fill_for_round("ab", r) for r in (None, 0, 1, 2, 7)] == [
        "legacy", "legacy", "compact", "legacy", "compact"]
    assert [probe.fill_for_round("cycle", r) for r in (None, 0, 1, 2, 3, 5)] == [
        "legacy", "legacy", "compact", "serial", "legacy", "serial"]
    assert probe.fill_mode({probe.FILL_ENV: "serial"}) == "serial"
    assert probe.fill_for_round("compact", None) == "compact"
    assert probe.fill_for_round("legacy", 3) == "legacy"


def test_touch_pages_reads_one_element_per_page_and_skips_strided():
    dense = torch.arange(10_000, dtype=torch.int32)  # 40,000 bytes -> 10 pages of 1024 int32
    strided = torch.zeros(64, 64, dtype=torch.int32).t()
    assert probe.touch_pages([dense, strided, "not a tensor"]) == (40_000, 10)


def test_usage_delta_and_proc_readers(tmp_path):
    delta = probe.usage_delta((1.0, 2.0, 10, 1, 5, 6), (1.5, 2.25, 15, 1, 8, 6))
    assert delta == {"utime_s": 0.5, "stime_s": 0.25, "minflt": 5, "majflt": 0, "nvcsw": 3, "nivcsw": 0}
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 4194304 kB\nMemAvailable: 2097152 kB\nShmem: 1048576 kB\n")
    assert probe.node_memory_gib(str(meminfo)) == {"memavail_gib": 2.0, "shmem_gib": 1.0}
    maps = tmp_path / "maps"
    maps.write_text("1000-2000 rw-s 00000000 00:05 7 /dev/shm/plasma 1\n3000-4000 rw-p 00000000 00:00 0\n")
    assert probe.mapping_of(0x1800, str(maps)) == "/dev/shm/plasma 1"
    assert probe.mapping_of(0x3000, str(maps)) == "[anon]"
    assert probe.mapping_of(0x5000, str(maps)) is None


# ---------------------------------------------------------------- compact fill


def test_compact_layout_by_hand():
    a = torch.arange(3 * 4 * 2, dtype=torch.int32).reshape(3, 4, 2)
    b = 100 + torch.arange(2 * 4 * 2, dtype=torch.int32).reshape(2, 4, 2)
    data = {"rollout_routed_experts": [a, b], "tokens": [torch.zeros(4), torch.zeros(3)]}
    got = _run(probe.compact_fill_replay_data, data=data, chunks=[[0, 1]], args=_args(multiplier=4, sp=False),
               state=_state(), stream_idx=[3, None], pin=False)
    minus = torch.full((1, 2), -1, dtype=torch.int32)
    want_3 = torch.cat([a[:, 3], minus, b[:, 3], minus, minus])  # 7 rows padded to 8
    want_1 = torch.cat([a[:, 1], minus, b[:, 1], minus, minus])
    assert torch.equal(got[0][0], want_3) and torch.equal(got[1][0], want_1)


@pytest.mark.parametrize("multiplier,tp_rank,tp_size,sp", [
    (128, 0, 1, False), (4, 0, 1, False), (8, 1, 2, True), (8, 0, 2, True), (8, 1, 2, False)])
def test_compact_equals_legacy_reference(multiplier, tp_rank, tp_size, sp):
    data, chunks = _case(seed=multiplier + tp_rank + 10 * tp_size, samples=13, streams=7, topk=3,
                         micro_batches=4)
    kwargs = dict(data=data, chunks=chunks, args=_args(multiplier=multiplier, sp=sp),
                  state=_state(tp_rank=tp_rank, tp_size=tp_size), stream_idx=[None, 5, 2, 6])
    _assert_same(_run(_reference_fill, **kwargs), _run(probe.compact_fill_replay_data, pin=False, **kwargs))


def test_compact_writes_every_row_of_an_uninitialized_buffer(monkeypatch):
    real_empty = torch.empty

    def dirty_empty(*args, **kwargs):
        return real_empty(*args, **kwargs).fill_(7777)

    data, chunks = _case(seed=5, samples=9, streams=6, topk=2, micro_batches=2)
    kwargs = dict(data=data, chunks=chunks, args=_args(multiplier=16, sp=True),
                  state=_state(tp_rank=1, tp_size=4), stream_idx=[None, 4])
    want = _run(_reference_fill, **kwargs)
    monkeypatch.setattr(torch, "empty", dirty_empty)
    got = _run(probe.compact_fill_replay_data, pin=False, **kwargs)
    monkeypatch.setattr(torch, "empty", real_empty)
    _assert_same(want, got)
    assert all(not (r == 7777).any() for stream in got for r in stream)


def test_compact_rejects_out_of_range_stream_and_token_positions():
    data, chunks = _case(seed=1, samples=3, streams=4, topk=2, micro_batches=1)
    with pytest.raises(AssertionError, match="out of range"):
        _run(probe.compact_fill_replay_data, data=data, chunks=chunks, args=_args(multiplier=4, sp=False),
             state=_state(), stream_idx=[4], pin=False)
    with pytest.raises(ValueError, match="token-position"):
        _run(probe.compact_fill_replay_data, data=data, chunks=chunks, args=_args(multiplier=4, sp=False),
             state=_state(), stream_idx=[0], pin=False, indices_are_token_positions=True)


def test_compact_supported_names_the_refusal():
    call = {"register_replay_list_func": _sequential}
    ok = dict(args=_args(multiplier=4, sp=False), parallel_state=_state(), sequential=_sequential)
    assert probe.compact_supported(call=call, **ok) is None
    assert "thd" in probe.compact_supported(call=call, **{**ok, "args": SimpleNamespace(qkv_format="bshd")})
    assert "context" in probe.compact_supported(call=call, **{**ok, "parallel_state": _state(cp_size=2)})
    assert "token" in probe.compact_supported(call={**call, "indices_are_token_positions": True}, **ok)
    assert "sequential" in probe.compact_supported(call={"register_replay_list_func": print}, **ok)


# ---------------------------------------------------------------- wrappers and install


def _fake_module(calls):
    def get_data_iterator(*args, **kwargs):
        calls.append("iter")
        return "iterators", [2]

    def fill_replay_data(**kwargs):
        calls.append("legacy")
        return _reference_fill(**kwargs, parallel_state=_state())

    return SimpleNamespace(get_data_iterator=get_data_iterator, fill_replay_data=fill_replay_data)


def _install(module, env, rollout_id):
    actor = SimpleNamespace(_last_rollout_id=rollout_id)
    with redirect_stdout(io.StringIO()):
        return probe.install(actor, env=env, targets=lambda: (module, _sequential, _state))


def _call_fill(module, *, args=None):
    data, chunks = _case(seed=3, samples=6, streams=5, topk=2, micro_batches=2)
    rollout_data = {k: list(v) for k, v in data.items()}
    replays = [_Replay(None), _Replay(4)]
    out = io.StringIO()
    with redirect_stdout(out):
        module.get_data_iterator(None)
        module.fill_replay_data(args=args or _args(multiplier=4, sp=False), models=None,
                                data_iterator=[_Iterator(rollout_data, chunks)], num_microbatches=[2],
                                rollout_data=rollout_data, data_key="rollout_routed_experts",
                                replay_list=replays, register_replay_list_func=_sequential)
    lines = [line for line in out.getvalue().splitlines() if line.startswith(probe.LOG_PREFIX)]
    return [json.loads(line[len(probe.LOG_PREFIX):]) for line in lines], replays


def test_probe_prints_one_record_per_fill_and_keeps_legacy_result():
    calls = []
    module = _fake_module(calls)
    assert _install(module, {}, rollout_id=4)
    records, replays = _call_fill(module)
    assert calls == ["iter", "legacy"]
    (record,) = records
    assert record["mode"] == "legacy" and record["rollout"] == 4 and record["mbs"] == 2
    assert record["samples"] == 6 and record["pages"] > 0 and "src_map" in record  # first fill names it
    for key in ("iter_s", "touch_s", "fill_s", "register_s", "prepare_s", "touch", "fill", "threads"):
        assert key in record
    assert set(record["fill"]) == {"utime_s", "stime_s", "minflt", "majflt", "nvcsw", "nivcsw"}
    assert all(len(r.top_indices_list) == 2 for r in replays)


def test_ab_mode_uses_compact_on_odd_round_with_same_records():
    calls = []
    module = _fake_module(calls)
    assert _install(module, {probe.FILL_ENV: "ab"}, rollout_id=5)
    records, compact = _call_fill(module)
    assert calls == ["iter"] and records[0]["mode"] == "compact" and records[0]["mbs"] == 2
    probe._STATE["actor"]._last_rollout_id = 6
    records, legacy = _call_fill(module)
    assert calls == ["iter", "iter", "legacy"] and records[0]["mode"] == "legacy"
    _assert_same([r.top_indices_list for r in legacy], [r.top_indices_list for r in compact])


def test_serial_mode_runs_legacy_with_one_thread_and_restores_it():
    calls, seen = [], []
    module = _fake_module(calls)
    inner = module.fill_replay_data

    def fill_replay_data(**kwargs):
        seen.append(torch.get_num_threads())
        return inner(**kwargs)

    module.fill_replay_data = fill_replay_data
    before = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        assert _install(module, {probe.FILL_ENV: "cycle"}, rollout_id=2)
        records, replays = _call_fill(module)
        assert seen == [1] and torch.get_num_threads() == 2
        assert records[0]["mode"] == "serial" and records[0]["threads"] == 1 and records[0]["mbs"] == 2
        assert all(len(r.top_indices_list) == 2 for r in replays)
    finally:
        torch.set_num_threads(before)


def test_compact_refusal_runs_legacy_and_says_why():
    calls = []
    module = _fake_module(calls)
    state = _state(cp_size=2)
    with redirect_stdout(io.StringIO()):
        assert probe.install(SimpleNamespace(_last_rollout_id=1), env={probe.FILL_ENV: "compact"},
                             targets=lambda: (module, _sequential, lambda: state))
    records, _ = _call_fill(module)
    assert calls == ["iter", "legacy"]
    assert records[0]["mode"] == "legacy" and records[0]["compact_refused"] == "context parallel > 1"


def test_install_off_leaves_miles_untouched_and_is_idempotent():
    calls = []
    module = _fake_module(calls)
    original = module.fill_replay_data
    assert _install(module, {probe.PROBE_ENV: "0"}, rollout_id=0) is False
    assert module.fill_replay_data is original
    assert _install(module, {}, rollout_id=0) is True
    wrapped = module.fill_replay_data
    assert wrapped is not original
    assert _install(module, {}, rollout_id=1) is True
    assert module.fill_replay_data is wrapped and probe._STATE["actor"]._last_rollout_id == 1


def test_probe_off_with_compact_fill_prints_nothing():
    calls = []
    module = _fake_module(calls)
    assert _install(module, {probe.PROBE_ENV: "0", probe.FILL_ENV: "compact"}, rollout_id=0)
    records, replays = _call_fill(module)
    assert records == [] and calls == ["iter"] and all(len(r.top_indices_list) == 2 for r in replays)


def test_install_best_effort_reports_a_missing_miles_and_raises_on_bad_mode():
    def missing():
        raise ImportError("no miles here")

    out = io.StringIO()
    with redirect_stdout(out):
        probe.install_best_effort(SimpleNamespace(), env={}, targets=missing)
    assert "not installed: ImportError: no miles here" in out.getvalue()
    assert probe._STATE["installed"] is False
    with pytest.raises(probe.TrainGapConfigError):
        probe.install_best_effort(SimpleNamespace(), env={probe.FILL_ENV: "fast"}, targets=missing)


def test_install_actor_recorders_without_miles_still_returns_the_recorder_result(monkeypatch):
    from yeto.rl.adapters.miles import state_plugin

    monkeypatch.setattr(state_plugin, "install_grad_norm_recorder", lambda: True)
    out = io.StringIO()
    with redirect_stdout(out):
        assert state_plugin.install_actor_recorders(SimpleNamespace()) is True
    if probe._STATE["installed"] is False:
        assert "not installed" in out.getvalue()


# ---------------------------------------------------------------- against Miles itself


@pytest.mark.parametrize("multiplier,tp_rank,tp_size,sp", [(128, 0, 1, False), (8, 1, 2, True)])
def test_compact_equals_miles_fill_replay_data(multiplier, tp_rank, tp_size, sp):
    replay_data = pytest.importorskip("miles.backends.training_utils.replay_data")
    parallel = pytest.importorskip("miles.backends.training_utils.parallel")
    state = _state(tp_rank=tp_rank, tp_size=tp_size)
    parallel.set_parallel_state(state)
    try:
        data, chunks = _case(seed=7 + tp_size, samples=17, streams=52, topk=6, micro_batches=3)
        kwargs = dict(data=data, chunks=chunks, args=_args(multiplier=multiplier, sp=sp),
                      stream_idx=[None, 3, 51, 10], register=replay_data.register_replay_list_sequential)

        def miles_fill(*, parallel_state, **call):
            return replay_data.fill_replay_data(**call)

        legacy = _run(miles_fill, state=state, **kwargs)
        compact = _run(probe.compact_fill_replay_data, state=state, pin=False, **kwargs)
        _assert_same(legacy, compact)
        assert probe.compact_supported(args=kwargs["args"], parallel_state=state,
                                       call={"register_replay_list_func": replay_data.register_replay_list_sequential},
                                       sequential=replay_data.register_replay_list_sequential) is None
    finally:
        parallel.set_parallel_state(None)
