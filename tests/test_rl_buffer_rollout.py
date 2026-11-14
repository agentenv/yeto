"""CPU, no Ray, no Miles: the Miles rollout through
the durable sample buffer -- payload codec, sampler writes, trainer selection with
the version gate, the watermark in the round cut, crash and restart in processes."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy
import pytest
import torch
from test_rl_multinode_round_cut import NEXT, _checkpoint, _Driver, _Ledger, _trainer
from test_rl_reconfig_recovery import _ctl

from tests.buffer_rollout_procs import sample
from tests.rl_cut_fakes import make_rank, params, train_step
from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.adapters.miles.entry import _wire_sample_buffer
from yeto.rl.engine.recovery import CUT_KEY
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer

REPO = Path(__file__).resolve().parents[1]


def _buffer(root, batch_size=2):
    return FileSampleBuffer(CheckpointStore(root), batch_size=batch_size)


def _group(g, version):
    return [sample(2 * g, g, version), sample(2 * g + 1, g, version)]


# ---------------------------------------------------------------- argv mapping
def test_argv_selects_the_buffer_rollout_functions():
    assert br.rollout_argv(None, max_policy_age=0) == ()
    assert br.rollout_argv("sampler", max_policy_age=0) == (
        "--rollout-function-path", "yeto.rl.adapters.miles.buffer_rollout_miles.BufferWriterRolloutFn")
    assert br.rollout_argv("trainer", max_policy_age=2) == (
        "--rollout-function-path", "yeto.rl.adapters.miles.buffer_rollout_miles.BufferReaderRolloutFn")


@pytest.mark.parametrize("kwargs, message", [
    ({"role": "trainer", "max_policy_age": 0}, "max_policy_age >= 1"),
    ({"role": "sampler", "max_policy_age": 1, "fully_async": True}, "--fully-async"),
    ({"role": "trainer", "max_policy_age": 1, "rollout_function_path": "x.Y"}, "replaces"),
    ({"role": "both", "max_policy_age": 1}, "expected one of"),
])
def test_argv_refuses_unsupported_combinations(kwargs, message):
    with pytest.raises(ValueError, match=message):
        br.rollout_argv(**kwargs)


def test_rollout_function_paths_name_classes_of_the_plugin_module():
    """The Miles classes exist under the dotted paths Miles imports (the module needs
    Miles, so its source is parsed instead of imported)."""
    tree = ast.parse((REPO / "yeto/rl/adapters/miles/buffer_rollout_miles.py").read_text())
    classes = {node.name: [b.id for b in node.bases] for node in tree.body if isinstance(node, ast.ClassDef)}
    for path in br.ROLLOUT_FUNCTION_PATHS.values():
        module, name = path.rsplit(".", 1)
        assert module == "yeto.rl.adapters.miles.buffer_rollout_miles" and name in classes
    assert classes["BufferWriterRolloutFn"] == ["InferenceRolloutFn"]
    assert classes["BufferReaderRolloutFn"] == ["BaseRolloutFn"]
    assert classes["ConvertibleRolloutFn"] == ["BufferWriterRolloutFn"]


def test_convertible_rollout_switches_between_generating_and_reading(monkeypatch):
    """A convertible node generates into its shard (sampler mode) until it
    converts; in trainer mode the same function reads the buffer."""
    plugin = _import_plugin_with_stub_miles(monkeypatch, {})
    fn = plugin.ConvertibleRolloutFn.__new__(plugin.ConvertibleRolloutFn)
    fn._round = lambda input: ("read", input)
    monkeypatch.setattr(plugin.BufferWriterRolloutFn, "__call__",
                        lambda self, input: ("generated", input), raising=False)
    assert br.convertible_mode() == br.SAMPLER and fn(1) == ("generated", 1)
    assert br.executor_set_mode(None, br.TRAINER) == br.SAMPLER
    try:
        assert fn(2) == ("read", 2)
    finally:
        br.executor_set_mode(None, br.SAMPLER)
    with pytest.raises(br.SampleBufferRolloutError):
        br.executor_set_mode(None, "both")
    monkeypatch.delenv(br.SHARD_ENV, raising=False)
    with pytest.raises(br.SampleBufferRolloutError, match=br.SHARD_ENV):
        fn._open_writer_buffer()


def _import_plugin_with_stub_miles(monkeypatch, registry):
    """Import buffer_rollout_miles against minimal stand-ins for the Miles modules."""
    import importlib
    import types

    class _Base:
        def __init__(self, input):
            self.constructor_input = input

    stubs = {
        "miles.rollout.base_types": dict(BaseRolloutFn=_Base, RolloutFnConstructorInput=object,
                                         RolloutFnInput=object, RolloutFnTrainOutput=object),
        "miles.rollout.inference_rollout.inference_rollout_common": dict(InferenceRolloutFn=_Base),
        "miles.utils.function_registry": dict(load_function=lambda path: registry.get(path) if path else None),
        "miles.utils.types": dict(Sample=object),
    }
    for name, attrs in stubs.items():
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(**attrs))
    plugin = "yeto.rl.adapters.miles.buffer_rollout_miles"
    monkeypatch.setitem(sys.modules, plugin, None)  # teardown restores the state before the test
    del sys.modules[plugin]
    return importlib.import_module(plugin)


def test_reader_round_calls_the_metadata_hooks(monkeypatch):
    """A trainer that reads the buffer must still run Miles' end-of-round hooks,
    or the driver finds no rollout metadata for the round."""
    calls = []
    registry = {"filter": lambda args, data: calls.append(("filter", data)),
                "all": lambda args, samples, source: calls.append(("all", samples, source))}
    plugin = _import_plugin_with_stub_miles(monkeypatch, registry)
    reader = plugin._BufferRound.__new__(plugin._BufferRound)
    reader.args = args = type("A", (), {"rollout_sample_filter_path": "filter",
                                             "rollout_all_samples_process_path": "all"})()
    groups = [["s0", "s1"], ["s2", "s3"]]
    reader._run_round_hooks(groups)
    assert calls == [("filter", groups), ("all", groups, None)]
    args.rollout_sample_filter_path = args.rollout_all_samples_process_path = None
    reader._run_round_hooks(groups)
    assert len(calls) == 2


def test_island_env():
    assert br.island_env("sampler", store_root="/s", sampler_id="s0") == {
        br.BUFFER_ROOT_ENV: "/s", br.SAMPLER_ID_ENV: "s0"}
    assert br.island_env("trainer", store_root="/s", max_policy_age=2, wait_s=30) == {
        br.BUFFER_ROOT_ENV: "/s", br.MAX_POLICY_AGE_ENV: "2", br.WAIT_ENV: "30"}
    with pytest.raises(ValueError):
        br.island_env("sampler", store_root="/s")
    assert br.island_env("convertible", store_root="/s", sampler_id="island1", shard="island1",
                         max_policy_age=2) == {
        br.BUFFER_ROOT_ENV: "/s", br.SAMPLER_ID_ENV: "island1", br.SHARD_ENV: "island1",
        br.MAX_POLICY_AGE_ENV: "2"}
    with pytest.raises(ValueError, match="shard"):
        br.island_env("convertible", store_root="/s", sampler_id="i", max_policy_age=2)


# ---------------------------------------------------------------- payload and entry
def test_payload_keeps_per_token_logprobs_versions_and_arrays():
    s = sample(0, 0, 4)
    s["rollout_routed_experts"] = numpy.arange(6, dtype=numpy.int32).reshape(2, 3)
    s["weight_versions"] = [[{"version": "3", "abs_start": 2, "abs_end": 3},
                             {"version": "4", "abs_start": 3, "abs_end": 5}]]
    back = br.decode_group(br.encode_group([s]))[0]
    assert back["rollout_log_probs"] == s["rollout_log_probs"]
    assert back["weight_versions"] == s["weight_versions"]
    assert back["rollout_routed_experts"].dtype == numpy.int32
    assert numpy.array_equal(back["rollout_routed_experts"], s["rollout_routed_experts"])
    assert br.group_entry_fields([s, sample(1, 0, 5)]) == (3, 6)  # oldest token version


def test_entry_reads_miles_per_call_weight_version_dicts():
    """Miles Sample.to_dict() (k31 pin 63834eaa) writes one WeightVersionsPerCall
    dict per generate call, {"spans": [...], "prefill_spans": [...], ...}."""
    from yeto.rl.engine import buffer_payload

    s = sample(0, 0, 4)
    s["weight_versions"] = [{"spans": [{"version": "5", "abs_start": 0, "abs_end": 2},
                                       {"version": "6", "abs_start": 2, "abs_end": 4}],
                             "prefill_spans": [], "output_start": 0, "prompt_tokens": 3}]
    back = br.decode_group(br.encode_group([s]))[0]
    assert buffer_payload.behavior_versions(back) == [5, 6]
    assert br.group_entry_fields([back, sample(1, 0, 5)]) == (5, 6)


def test_payload_stores_integer_arrays_in_the_narrowest_dtype():
    """R3 expert ids (0..127, int32 on the engine) are stored as uint8 and read
    back as int32 with the same values; a payload without ``cast`` (written
    before it existed) decodes as before."""
    import base64
    import json

    experts = numpy.random.default_rng(0).integers(0, 128, size=(300, 52, 6), dtype=numpy.int32)
    cases = {"ids": experts, "neg": numpy.array([-3, 100], dtype=numpy.int64),
             "wide": numpy.array([0, 70000], dtype=numpy.int64),
             "int8_already": numpy.array([1, 2], dtype=numpy.uint8),
             "empty": numpy.zeros((0, 6), dtype=numpy.int32),
             "floats": numpy.array([0.5, 1.5], dtype=numpy.float32)}
    s = sample(0, 0, 4)
    s.update(cases)
    data = br.encode_group([s])
    stored = json.loads(data)["samples"][0]
    assert (stored["ids"]["dtype"], stored["ids"]["cast"]) == ("uint8", "int32")
    assert (stored["neg"]["dtype"], stored["wide"]["dtype"]) == ("int8", "uint32")
    assert "cast" not in stored["int8_already"] and "cast" not in stored["floats"]
    assert len(data) < experts.nbytes / 2  # base64 of uint8 ids: a third of int32 base64
    back = br.decode_group(data)[0]
    for name, array in cases.items():
        assert back[name].dtype == array.dtype and back[name].shape == array.shape, name
        assert numpy.array_equal(back[name], array), name
        assert back[name].flags.writeable, name
    legacy = {"schema": br.PAYLOAD_SCHEMA, "samples": [{**{k: v for k, v in s.items() if k not in cases},
              "ids": {"__ndarray__": base64.b64encode(experts.tobytes()).decode("ascii"),
                      "dtype": "int32", "shape": list(experts.shape)}}]}
    old = br.decode_group(json.dumps(legacy).encode())[0]["ids"]
    assert old.dtype == numpy.int32 and numpy.array_equal(old, experts)


def test_entry_needs_logprobs_and_a_numeric_version():
    bad = sample(0, 0, 1)
    bad["rollout_log_probs"] = None
    with pytest.raises(br.SampleBufferRolloutError, match="rollout_log_probs missing"):
        br.group_entry_fields([bad])
    bad = sample(0, 0, 1)
    bad["weight_versions"] = []
    with pytest.raises(br.SampleBufferRolloutError, match="numeric engine weight version"):
        br.group_entry_fields([bad])


def test_writer_writes_payload_then_entry(tmp_path):
    buffer = _buffer(tmp_path)
    entry = br.GroupWriter(buffer, "s0").write(_group(0, 7), group_key="r0-g0")
    assert (entry.sample_id, entry.behavior_version, entry.tokens) == ("s0-r0-g0", 7, 6)
    reopened = _buffer(tmp_path)
    assert len(reopened) == 1
    assert br.decode_group(reopened.read_payload(entry.payload_ref))[1]["index"] == 1


# ---------------------------------------------------------------- reader
def test_reader_passes_over_groups_older_than_the_limit(tmp_path):
    buffer = _buffer(tmp_path)
    writer = br.GroupWriter(buffer, "s0")
    for g, version in enumerate((1, 3, 2, 4)):
        writer.write(_group(g, version), group_key=f"g{g}")
    reader = br.BatchReader(_buffer(tmp_path), max_policy_age=2, wait_s=0.0, poll_s=0.0)
    groups, report = reader.take(0, 2, current_version=5)  # floor = 3
    assert [(pos, g[0]["group_index"]) for pos, g in groups] == [(1, 1), (3, 3)]
    assert (report.stale_groups, report.start, report.end, report.behavior_versions) == (2, 0, 4, {3: 1, 4: 1})


def test_reader_reads_the_next_round_ahead_and_never_refreshes_with_a_read_open(tmp_path):
    """After a take the reader reads the next round's payloads in the background
    (a sampler appends them while the trainer trains); the next take uses them.
    No refresh (a Volume reload) runs while one of its payload reads is open."""
    import threading

    buffer = _buffer(tmp_path)
    writer = br.GroupWriter(buffer, "s0")
    for g in range(2):
        writer.write(_group(g, 1), group_key=f"r0-g{g}")
    open_reads = [0]
    overlaps = []

    class Watched(FileSampleBuffer):
        def read_payload(self, ref):
            open_reads[0] += 1
            try:
                time.sleep(0.01)
                return super().read_payload(ref)
            finally:
                open_reads[0] -= 1

        def refresh(self):
            if open_reads[0]:
                overlaps.append(open_reads[0])
            return super().refresh()

    reader = br.BatchReader(Watched(CheckpointStore(tmp_path), batch_size=2), max_policy_age=2,
                            wait_s=30.0, poll_s=0.01, read_workers=4)
    first, report = reader.take(0, 2, current_version=1)
    assert [pos for pos, _ in first] == [0, 1] and report.prefetched == 0
    for g in range(2, 4):  # the next round arrives while the trainer trains
        writer.write(_group(g, 2), group_key=f"r1-g{g}")
    deadline = time.monotonic() + 10
    while len(reader._reads) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    second, report = reader.take(1, 2, current_version=2)
    assert [(pos, g[0]["group_index"]) for pos, g in second] == [(2, 2), (3, 3)]
    assert report.prefetched == 2 and report.read_s >= 0.0
    assert set(report.metrics()) >= {"sample_buffer/select_s", "sample_buffer/read_s",
                                     "sample_buffer/decode_s", "sample_buffer/prefetched_groups"}
    assert overlaps == []
    reader._stop_prefetching()


def test_reader_without_peek_or_workers_reads_inline(tmp_path):
    buffer = _buffer(tmp_path)
    writer = br.GroupWriter(buffer, "s0")
    for g in range(3):
        writer.write(_group(g, 1), group_key=f"g{g}")

    class NoPeek:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            if name == "peek":
                raise AttributeError(name)
            return getattr(self.inner, name)

    reader = br.BatchReader(NoPeek(_buffer(tmp_path)), max_policy_age=1, wait_s=0.0, poll_s=0.0,
                            read_workers=1)
    assert reader.prefetch is False
    groups, report = reader.take(0, 3, current_version=1)
    assert [pos for pos, _ in groups] == [0, 1, 2] and report.prefetched == 0


def test_reader_waits_for_samples_then_times_out(tmp_path):
    buffer = _buffer(tmp_path)
    writer = br.GroupWriter(buffer, "s0")
    now = [0.0]

    def sleep(dt):  # a sampler appends while the reader waits
        now[0] += 1.0
        if now[0] == 2.0:
            writer.write(_group(0, 1), group_key="g0")

    reader = br.BatchReader(_buffer(tmp_path), max_policy_age=1, wait_s=5.0, poll_s=1.0,
                            clock=lambda: now[0], sleep=sleep)
    groups, report = reader.take(0, 1, current_version=1)
    assert len(groups) == 1 and report.wait_s == 2.0
    with pytest.raises(br.SampleBufferRolloutError, match="0/1 groups after 5 s"):
        reader.take(1, 1, current_version=1)


def test_reader_returns_the_rest_after_the_data_end(tmp_path):
    buffer = _buffer(tmp_path)
    br.GroupWriter(buffer, "s0").write(_group(0, 1), group_key="g0")
    buffer.mark_data_end()
    reader = br.BatchReader(_buffer(tmp_path), max_policy_age=1, wait_s=0.0, poll_s=0.0)
    groups, _ = reader.take(0, 4, current_version=1)
    assert len(groups) == 1
    with pytest.raises(br.SampleBufferRolloutError, match="ended with no trainable group"):
        reader.take(1, 4, current_version=1)


def test_watermark_covers_only_rounds_below_the_cut(tmp_path):
    """A round selected ahead of the cut (prefetch) is not in the watermark."""
    buffer = _buffer(tmp_path)
    writer = br.GroupWriter(buffer, "s0")
    for g in range(6):
        writer.write(_group(g, 1), group_key=f"g{g}")
    reader = br.BatchReader(_buffer(tmp_path), max_policy_age=1, wait_s=0.0, poll_s=0.0)
    for rid in range(3):
        reader.take(rid, 2, current_version=1)
    assert reader.watermark_before(0) == 0
    assert reader.watermark_before(2) == 4  # rounds 0 and 1 trained, round 2 selected
    assert reader.buffer.watermark == 4
    assert reader.reset(2) == 2
    groups, _ = reader.take(1, 2, current_version=1)
    assert [pos for pos, _ in groups] == [2, 3]


# ---------------------------------------------------------------- round cut
def _reader_link(tmp_path):
    for g in range(6):
        br.GroupWriter(_buffer(tmp_path / "buf"), "s0").write(_group(g, 1), group_key=f"g{g}")
    reader = br.BatchReader(_buffer(tmp_path / "buf"), max_policy_age=1, wait_s=0.0, poll_s=0.0)
    br.register_reader(reader)
    pool = type("Pool", (), {"_in_executor": staticmethod(lambda fn, *a: fn(None, *a))})()
    return reader, pool


def test_round_cut_records_the_watermark_and_resume_continues_there(tmp_path):
    reader, pool = _reader_link(tmp_path)
    for rid in range(NEXT + 1):  # rounds 0..NEXT-1 trained, round NEXT selected ahead
        reader.take(rid, 2, current_version=1)
    store = tmp_path / "store"
    rank = make_rank(0)
    g = torch.Generator().manual_seed(3)
    for _ in range(NEXT):
        train_step(rank, torch.randn(3, 6, generator=g))
    ctl = _ctl(tmp_path / "a/state", {"t": 1000.0}, checkpoint_store=store)
    driver = _Driver(rank, published=NEXT, local_step=NEXT)
    cp = _checkpoint(ctl, driver, _trainer(rank), _Ledger(NEXT))
    args = type("Args", (), {"rollout_function_path": br.ROLLOUT_FUNCTION_PATHS["trainer"]})()
    _wire_sample_buffer(cp, driver=type("D", (), {"rollout": pool})(), miles_args=args)
    info = cp.save(driver, rollout_id=NEXT)
    assert info[CUT_KEY]["buffer_watermark"] == 2 * NEXT
    ctl.close()

    rank_b = make_rank(9)
    ctl_b = _ctl(tmp_path / "b/state", {"t": 2000.0}, checkpoint_store=store)
    reader_b, pool_b = _reader_link(tmp_path / "b")
    driver_b = _Driver(rank_b)
    cp_b = _checkpoint(ctl_b, driver_b, _trainer(rank_b), _Ledger(NEXT))
    _wire_sample_buffer(cp_b, driver=type("D", (), {"rollout": pool_b})(), miles_args=args)
    cp_b.resume(driver_b)
    assert reader_b.buffer.watermark == reader_b.buffer.read_cursor == 2 * NEXT
    resumed = [e for e, f in driver_b.events if e == "rl_resume"]
    assert resumed and dict(driver_b.events)["rl_resume"]["checks"]["buffer_watermark"] == 2 * NEXT
    ctl_b.close()


def test_wiring_needs_round_cuts_and_skips_other_rollouts():
    args = type("Args", (), {"rollout_function_path": br.ROLLOUT_FUNCTION_PATHS["trainer"]})()
    with pytest.raises(RuntimeError, match="needs round cuts"):
        _wire_sample_buffer(None, driver=None, miles_args=args)
    _wire_sample_buffer(None, driver=None, miles_args=type("Args", (), {"rollout_function_path": None})())


# ---------------------------------------------------------------- processes
def _run(*args, check=True):
    return subprocess.run([sys.executable, "-m", "tests.buffer_rollout_procs", *map(str, args)], cwd=REPO, check=check,
                          capture_output=True, text=True, timeout=120)


def test_sampler_and_trainer_processes_crash_and_continue_at_the_watermark(tmp_path):
    store, cut, log = tmp_path / "store", tmp_path / "cut.json", tmp_path / "trained.jsonl"
    _run("sampler", store, "s0", 6, 9)
    crashed = _run("trainer", store, cut, log, 3, 2, 1, check=False)
    assert crashed.returncode == 3, crashed.stderr
    assert json.loads(cut.read_text()) == {"next": 1, "watermark": 2}  # round 1 consumed, not cut
    _run("trainer", store, cut, log, 2, 2)
    _run("sampler", store, "s1", 2, 9)  # a restarted sampler appends after the existing index
    _run("trainer", store, cut, log, 1, 2)
    trained = [json.loads(line) for line in log.read_text().splitlines()]
    assert [(t["rollout_id"], t["groups"]) for t in trained] == [
        (0, [0, 1]), (1, [2, 3]), (2, [4, 5]), (3, [6, 7])]  # round 1 trained once, after the restart
    assert json.loads(cut.read_text()) == {"next": 4, "watermark": 8}
    index = [json.loads(line) for line in (store / "sample-buffer/index.jsonl").read_text().splitlines()]
    assert [e["sample_id"].split("-")[0] for e in index] == ["s0"] * 6 + ["s1"] * 2


# ---------------------------------------------------------------- full-parameter stop and restart
def _round_batch(rollout_id):
    return torch.randn(3, 6, generator=torch.Generator().manual_seed(100 + rollout_id))


def test_full_parameter_trainer_stops_at_a_scheduled_cut_and_continues_without_retraining(tmp_path):
    """Blocker: a full-parameter trainer stopped by the scheduler cuts
    (trigger SCHEDULED); the next incarnation restores it, continues the policy version
    and the buffer position, trains no group twice and ends bitwise equal to a run
    that never stopped."""
    from yeto.rl.engine.recovery import SCHEDULED

    rounds, stop_at = 4, NEXT
    torch.manual_seed(99)
    reference = make_rank(0, full=True)
    for rid in range(rounds):
        train_step(reference, _round_batch(rid))

    reader, pool = _reader_link(tmp_path)  # 6 groups of version 1, 2 per round
    writer = br.GroupWriter(_buffer(tmp_path / "buf"), "s1")
    for g in range(6, 2 * rounds):
        writer.write(_group(g, 1), group_key=f"g{g}")
    args = type("Args", (), {"rollout_function_path": br.ROLLOUT_FUNCTION_PATHS["trainer"]})()
    store, trained = tmp_path / "store", []
    torch.manual_seed(99)
    rank_a = make_rank(0, full=True)
    for rid in range(stop_at):
        groups, _ = reader.take(rid, 2, current_version=1)
        trained += [pos for pos, _ in groups]
        train_step(rank_a, _round_batch(rid))
    reader.take(stop_at, 2, current_version=1)  # selected ahead, never trained by A
    ctl_a = _ctl(tmp_path / "a/state", {"t": 1000.0}, checkpoint_store=store)
    driver_a = _Driver(rank_a, published=stop_at, local_step=stop_at)
    cp_a = _checkpoint(ctl_a, driver_a, _trainer(rank_a), _Ledger(stop_at))
    _wire_sample_buffer(cp_a, driver=type("D", (), {"rollout": pool})(), miles_args=args)
    info = cp_a.save(driver_a, rollout_id=stop_at, final=True, trigger=SCHEDULED)
    assert info[CUT_KEY]["trigger"] == SCHEDULED and info["final"]
    assert info[CUT_KEY]["buffer_watermark"] == 2 * stop_at
    ctl_a.close()

    torch.manual_seed(12345)  # the new incarnation's process starts with other RNG state
    rank_b = make_rank(9, full=True)  # and other initial weights
    ctl_b = _ctl(tmp_path / "b/state", {"t": 2000.0}, checkpoint_store=store)
    reader_b = br.BatchReader(_buffer(tmp_path / "buf"), max_policy_age=1, wait_s=0.0, poll_s=0.0)
    br.register_reader(reader_b)
    driver_b = _Driver(rank_b)
    cp_b = _checkpoint(ctl_b, driver_b, _trainer(rank_b), _Ledger(stop_at))
    _wire_sample_buffer(cp_b, driver=type("D", (), {"rollout": pool})(), miles_args=args)
    resumed = cp_b.resume(driver_b)
    assert resumed["next_rollout_id"] == stop_at and driver_b.local_step == stop_at
    event = dict(driver_b.events)["rl_resume"]
    assert event["policy_version"] == stop_at and event["checks"]["policy_hash"] == "same"
    for rid in range(stop_at, rounds):
        groups, _ = reader_b.take(rid, 2, current_version=1)
        trained += [pos for pos, _ in groups]
        train_step(rank_b, _round_batch(rid))
    ctl_b.close()

    assert trained == list(range(2 * rounds))  # every group once, in order: none trained twice
    for name, value in params(reference).items():
        assert torch.equal(value, params(rank_b)[name]), name


def test_reader_round_reports_its_stale_groups_as_round_metadata(monkeypatch):
    """The groups the reader passed over this round reach the driver as
    per-round metadata (driver event rl_buffer_stale_skipped), before the
    end-of-round hooks put the round's main record."""
    from types import SimpleNamespace

    calls = []
    registry = {"all": lambda args, samples, source: calls.append(("all",))}
    plugin = _import_plugin_with_stub_miles(monkeypatch, registry)
    monkeypatch.setattr(plugin, "record_round_metadata",
                        lambda args, rid, **counts: calls.append(("meta", rid, counts)))
    monkeypatch.setattr(plugin, "Sample", SimpleNamespace(from_dict=lambda d: d))
    monkeypatch.setattr(plugin, "RolloutFnTrainOutput", lambda **kw: kw)
    monkeypatch.setattr(plugin, "offpolicy_source_event", lambda *a, **kw: {})
    monkeypatch.setattr(plugin, "flat_metrics", lambda source: {})
    group = [sample(0, 0, 4), sample(1, 0, 4)]
    report = br.TakeReport(rollout_id=2, groups=1, stale_groups=3, start=0, end=4)
    reader = plugin._BufferRound.__new__(plugin._BufferRound)
    reader.args = SimpleNamespace(rollout_sample_filter_path=None, rollout_all_samples_process_path="all")
    reader.groups, reader.samples_per_group = 1, 2
    reader.reader = SimpleNamespace(take=lambda rid, n, current_version: ([(3, group)], report))
    out = reader(SimpleNamespace(evaluation=False, rollout_id=2, weight_version=4))
    fill = {"buffer_incomplete_groups": 0, "buffer_dropped_small_groups": 0,
            "buffer_carried_over_groups": 0, "buffer_padded_samples": 0}
    assert calls == [("meta", 2, {"buffer_stale_groups": 3, **fill}), ("all",)]
    assert out["metrics"]["sample_buffer/stale_groups"] == 3.0


def test_reader_round_pads_a_short_round_to_exactly_the_global_batch(monkeypatch):
    """A short group is trained; Miles still gets rollout_batch_size * n samples:
    the padding comes after the end-of-round hooks, each in a group of its own."""
    from types import SimpleNamespace

    calls = []
    registry = {"all": lambda args, samples, source: calls.append(("all", sum(len(g) for g in samples)))}
    plugin = _import_plugin_with_stub_miles(monkeypatch, registry)
    monkeypatch.setattr(plugin, "record_round_metadata",
                        lambda args, rid, **counts: calls.append(("meta", rid, counts)))
    monkeypatch.setattr(plugin, "Sample", SimpleNamespace(from_dict=lambda d: d))
    monkeypatch.setattr(plugin, "RolloutFnTrainOutput", lambda **kw: kw)
    monkeypatch.setattr(plugin, "offpolicy_source_event", lambda *a, **kw: {})
    monkeypatch.setattr(plugin, "flat_metrics", lambda source: {})
    groups = [(0, _short(0, 4, 4)), (1, _short(1, 4, 3))]
    report = br.TakeReport(rollout_id=2, groups=2, stale_groups=0, start=0, end=3, incomplete_groups=1,
                           carried_over_groups=1)
    reader = plugin._BufferRound.__new__(plugin._BufferRound)
    reader.args = SimpleNamespace(rollout_sample_filter_path=None, rollout_all_samples_process_path="all")
    reader.groups, reader.samples_per_group = 2, 4
    reader.reader = SimpleNamespace(take=lambda rid, n, current_version: (groups, report))
    out = reader(SimpleNamespace(evaluation=False, rollout_id=2, weight_version=4))
    assert calls[0][2]["buffer_padded_samples"] == 1 and calls[0][2]["buffer_carried_over_groups"] == 1
    assert calls[1] == ("all", 7)  # the hooks see the real samples only
    samples = out["samples"]
    assert [len(g) for g in samples] == [4, 3, 1] and sum(len(g) for g in samples) == 2 * 4
    assert samples[1][0]["group_index"] == 1 and samples[2][0]["remove_sample"]
    assert samples[2][0]["loss_mask"] == [0, 0, 0]
    assert out["metrics"]["sample_buffer/padded_samples"] == 1.0
    # F2: the real samples' gradient shrinks by this fraction (documented, not rescaled)
    assert out["metrics"]["sample_buffer/padded_fraction"] == 1 / 8


def test_engine_sampler_sample_gets_one_miles_version_span():
    """An engine-sampler sample (metadata.weight_version only) reaches Miles with one
    WeightVersionsPerCall covering its response tokens; Miles' own spans are kept and
    a mid-generation version change gets none."""
    from types import SimpleNamespace

    from yeto.rl.adapters.miles.carry_over import sample_versions
    from yeto.rl.engine import buffer_payload

    s = {"index": 0, "tokens": [1, 2, 3, 10, 11], "response_length": 2,
         "metadata": {"weight_version": "4"}}
    out = buffer_payload.miles_sample_dict(s)
    assert out["weight_versions"] == [{"spans": [{"version": "4", "abs_start": 3, "abs_end": 5}],
                                       "prefill_spans": [], "output_start": 3, "prompt_tokens": 3}]
    spans = [SimpleNamespace(spans=[SimpleNamespace(**sp) for sp in out["weight_versions"][0]["spans"]])]
    assert sample_versions(SimpleNamespace(tokens=out["tokens"], response_length=2, weight_versions=spans)) == [4]
    miles = {**s, "weight_versions": [{"spans": [{"version": "7", "abs_start": 3, "abs_end": 5}],
                                       "prefill_spans": [], "output_start": 3}]}
    assert buffer_payload.miles_sample_dict(miles)["weight_versions"] == miles["weight_versions"]
    moved = {**s, "metadata": {"weight_version": "4", "final_weight_version": "5"}}
    assert "weight_versions" not in buffer_payload.miles_sample_dict(moved)



def _buffer_island(tmp_path, store, name, *, trained=True):
    import test_rl_resume_from_checkpoint as rfc
    from test_rl_resume_from_checkpoint import _cp, _rctl, _trained_island

    reader, pool = _reader_link(tmp_path / name)
    if trained:
        for rid in range(NEXT + 1):  # rounds 0..NEXT-1 trained, round NEXT selected ahead
            reader.take(rid, 2, current_version=1)
        _rank, _ctl_, driver, cp = _trained_island(tmp_path, store, NEXT, name=name)
    else:  # a freshly built trainer of a new island
        rank = make_rank(9)
        driver = rfc._Driver(rank)
        cp = _cp(_rctl(tmp_path, name, store), driver, rfc._trainer(rank), rfc._Ledger(NEXT))
    args = type("Args", (), {"rollout_function_path": br.ROLLOUT_FUNCTION_PATHS["trainer"]})()
    _wire_sample_buffer(cp, driver=type("D", (), {"rollout": pool})(), miles_args=args)
    return reader, driver, cp


def test_a_buffer_reader_cuts_in_the_background_on_a_store_that_commits_beside_it(tmp_path, monkeypatch):
    """A staged object store (cut_commits_beside_buffer_reads): the buffer-reading
    trainer's periodic cut is deferred (the next rounds do not wait for its upload);
    wait_uploads_durable completes it with the cut's watermark, and a new island
    resumes there."""
    import yeto.rl.engine.resume as rs

    monkeypatch.setattr(CheckpointStore, "cut_commits_beside_buffer_reads", property(lambda self: True))
    store = CheckpointStore(tmp_path / "store")
    _reader, driver, cp = _buffer_island(tmp_path, store, "a")
    assert cp.background_commit
    info = cp.save(driver, rollout_id=NEXT)
    assert info["store_commit"] == "deferred"
    assert cp.pointer() is None and rs.read_latest(store.root) is None  # nothing committed yet
    done = cp.wait_uploads_durable(driver)
    assert done == {"in_flight": False, "ok": True, "cut_id": info["cut_id"],
                    "pointed_cut_id": info["cut_id"], "error": None}
    assert rs.read_latest(store.root)["cut_id"] == info["cut_id"]
    assert cp.pointer()[CUT_KEY]["buffer_watermark"] == 2 * NEXT
    assert cp.wait_uploads_durable(driver)["cut_id"] is None  # nothing left in flight

    reader_b, driver_b, cp_b = _buffer_island(tmp_path, store, "b", trained=False)
    cp_b.resume(driver_b)
    assert reader_b.buffer.watermark == reader_b.buffer.read_cursor == 2 * NEXT


def test_a_buffer_reader_still_cuts_inline_on_a_store_that_does_not(tmp_path):
    store = CheckpointStore(tmp_path / "store")
    _reader, driver, cp = _buffer_island(tmp_path, store, "a")
    assert cp.background_commit and not cp._defer(final=False)
    assert cp.save(driver, rollout_id=NEXT)["store_commit"] == "inline"


def _short(g, version, k, n=4):
    """Group ``g`` with ``k`` of its ``n`` samples (a sampler lost the rest)."""
    return [sample(n * g + j, g, version) for j in range(k)]


def _reader(tmp_path, **kwargs):
    return br.BatchReader(_buffer(tmp_path), max_policy_age=1, wait_s=0.0, poll_s=0.0, group_size=4, **kwargs)


def test_reader_trains_short_groups_and_skips_only_groups_without_spread(tmp_path):
    writer = br.GroupWriter(_buffer(tmp_path), "s0")
    for g, k in enumerate((4, 1, 3, 2, 4)):  # g1: one sample, no in-group spread
        writer.write(_short(g, 1, k), group_key=f"g{g}")
    reader = _reader(tmp_path)
    groups, report = reader.take(0, 2, current_version=1)
    # g0 + g2 = 7 of 8: g3 (2) overflows the missing 1, so it is carried over
    assert [(pos, len(g)) for pos, g in groups] == [(0, 4), (2, 3)]
    assert (report.incomplete_groups, report.dropped_small_groups) == (1, 1)
    assert (report.carried_over_groups, report.padded_samples, report.end) == (1, 1, 3)
    assert report.behavior_versions == {1: 2} and set(report.position_sources) == {0, 2}
    metrics = report.metrics()
    assert metrics["sample_buffer/carried_over_groups"] == 1.0 and metrics["sample_buffer/padded_samples"] == 1.0
    assert reader.watermark_before(1) == 3  # the carried group is not past the watermark
    groups, report = reader.take(1, 2, current_version=1)
    assert [(pos, len(g)) for pos, g in groups] == [(3, 2), (4, 4)]  # the carried group first, whole
    assert (report.start, report.end, report.carried_over_groups, report.padded_samples) == (3, 5, 0, 2)
    assert reader.watermark_before(2) == 5


def test_reader_tops_a_short_round_up_with_whole_groups_that_fit(tmp_path):
    writer = br.GroupWriter(_buffer(tmp_path), "s0")
    for g, k in enumerate((4, 2, 2, 4)):
        writer.write(_short(g, 1, k), group_key=f"g{g}")
    groups, report = _reader(tmp_path).take(0, 2, current_version=1)
    assert sum(len(g) for _, g in groups) == 8 and [pos for pos, _ in groups] == [0, 1, 2]
    assert (report.carried_over_groups, report.padded_samples, report.end, report.incomplete_groups) == (0, 0, 3, 2)


def test_a_reset_drops_the_carried_group_so_it_is_selected_again(tmp_path):
    writer = br.GroupWriter(_buffer(tmp_path), "s0")
    for g, k in enumerate((4, 3, 4)):
        writer.write(_short(g, 1, k), group_key=f"g{g}")
    reader = _reader(tmp_path)
    _groups, report = reader.take(0, 2, current_version=1)
    assert (report.end, report.carried_over_groups) == (2, 1)
    reader.reset(reader.watermark_before(1))
    groups, _report = reader.take(1, 1, current_version=1)
    assert [(pos, len(g)) for pos, g in groups] == [(2, 4)]


def test_a_stale_carried_group_is_not_trained(tmp_path):
    writer = br.GroupWriter(_buffer(tmp_path), "s0")
    for g, k in enumerate((4, 3, 4, 4, 4)):
        writer.write(_short(g, 1 if g < 3 else 3, k), group_key=f"g{g}")
    reader = _reader(tmp_path)
    _groups, report = reader.take(0, 2, current_version=1)
    assert report.carried_over_groups == 1
    groups, report = reader.take(1, 2, current_version=3)  # max_policy_age 1: version 1 is too old
    assert [pos for pos, _ in groups] == [3, 4] and report.stale_groups == 1


def test_padding_fills_the_global_batch_without_touching_real_advantages():
    groups = [_short(0, 1, 4), _short(1, 1, 3)]
    for g, rewards in zip(groups, ([1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0]), strict=True):
        for d, r in zip(g, rewards, strict=True):
            d["reward"] = r
    groups[1][2]["tokens"] = [1, 2, 3]  # the shortest sample is the one copied
    pads = br.padding_samples(groups, 1, samples_per_group=4)
    batch = [d for g in groups for d in g] + pads
    assert len(batch) == 2 * 4  # rollout_batch_size * n_samples_per_prompt
    (pad,) = pads
    assert pad["loss_mask"] == [0, 0, 0] and pad["remove_sample"] and pad["metadata"]["yeto_padding"]
    assert pad["tokens"] == [1, 2, 3] and pad["rollout_id"] is None
    assert pad["group_index"] not in {d["group_index"] for g in groups for d in g}
    # group-relative centering (Miles' reward_group_segments groups by group_index):
    # the real groups' means use real samples only; a padding group centers to 0
    by_group: dict[int, list[float]] = {}
    for d in batch:
        by_group.setdefault(d["group_index"], []).append(d["reward"])
    centered = {g: [r - sum(v) / len(v) for r in v] for g, v in by_group.items()}
    assert centered[1] == pytest.approx([2 / 3, -1 / 3, -1 / 3]) and centered[pad["group_index"]] == [0.0]
    assert br.padding_samples(groups, 0, samples_per_group=4) == []


def test_padding_matches_miles_reward_normalization():
    miles = pytest.importorskip("miles.ray.rollout.train_data_conversion")
    from types import SimpleNamespace

    from miles.utils.types import Sample

    groups = [_short(0, 1, 4), _short(1, 1, 3)]
    for g, rewards in zip(groups, ([1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0]), strict=True):
        for d, r in zip(g, rewards, strict=True):
            d["reward"] = r
    batch = [Sample.from_dict(d) for d in [x for g in groups for x in g] + br.padding_samples(
        groups, 1, samples_per_group=4)]
    args = SimpleNamespace(advantage_estimator="grpo", grpo_std_normalization=False, n_samples_per_prompt=4,
                           rollout_batch_size=2, reward_key=None)
    out = miles._normalize_rewards_by_rollout(args, batch, [s.reward for s in batch], None)
    assert out == pytest.approx([0.5, -0.5, -0.5, 0.5, 2 / 3, -1 / 3, -1 / 3, 0.0])


def test_the_reader_rollout_log_leaves_the_padding_samples_out(monkeypatch):
    """F1: padding samples (copies of the shortest real one) must not enter Miles'
    rollout metrics; the reader's custom rollout log calls Miles' default logging over
    the real samples only, with the custom hook off (no recursion)."""
    import types

    plugin = _import_plugin_with_stub_miles(monkeypatch, {})
    logged = []

    def default_log(rollout_id, args, samples, extra, rollout_time):
        logged.append((rollout_id, args.custom_rollout_log_function_path, args.seed, samples, extra))

    monkeypatch.setitem(sys.modules, "miles.ray.rollout",
                        types.SimpleNamespace(metrics=types.SimpleNamespace(log_rollout_data=default_log)))
    real = [types.SimpleNamespace(metadata={}, reward=1.0) for _ in range(3)]
    pads = [types.SimpleNamespace(metadata={br.PADDING_KEY: True}, reward=1.0)]
    args = types.SimpleNamespace(custom_rollout_log_function_path=br.ROLLOUT_LOG_FUNCTION, seed=7)
    assert plugin.log_rollout_data(4, args, real + pads, {"x": 1}, 2.0) is True
    (rid, hook, seed, samples, extra), = logged
    assert rid == 4 and hook is None and seed == 7 and samples == real
    assert extra == {"x": 1, "rollout/padded_samples": 1, "rollout/padded_fraction": 0.25}
    # nothing padded: Miles logs as usual
    assert plugin.log_rollout_data(5, args, real, None, 2.0) is False and len(logged) == 1
    assert br.is_padding({"metadata": {br.PADDING_KEY: True}}) and not br.is_padding({"metadata": None})


def test_groups_the_top_up_passes_over_count_as_stale(tmp_path):
    """F3: a too-old group the short-round top-up (_fill) consumes unread is stale,
    as in the main selection."""
    writer = br.GroupWriter(_buffer(tmp_path), "s0")
    for g, (version, k) in enumerate(((2, 4), (2, 3), (0, 4), (2, 1))):
        writer.write(_short(g, version, k), group_key=f"g{g}")
    groups, report = _reader(tmp_path).take(0, 2, current_version=2)
    assert [pos for pos, _ in groups] == [0, 1]
    assert (report.stale_groups, report.dropped_small_groups, report.padded_samples) == (1, 1, 1)
    assert report.metrics()["sample_buffer/stale_groups"] == 1.0


def test_a_carried_group_that_ages_out_counts_as_stale(tmp_path):
    writer = br.GroupWriter(_buffer(tmp_path), "s0")
    for g, (version, k) in enumerate(((1, 4), (1, 3), (1, 2), (3, 4), (3, 4))):
        writer.write(_short(g, version, k), group_key=f"g{g}")
    reader = _reader(tmp_path)
    _, report = reader.take(0, 2, current_version=1)
    assert report.carried_over_groups == 1 and report.stale_groups == 0
    groups, report = reader.take(1, 2, current_version=3)  # the carried group (v1) is now too old
    assert [pos for pos, _ in groups] == [3, 4] and report.stale_groups == 1


def test_a_group_with_a_non_finite_reward_is_consumed_not_trained(tmp_path):
    """One NaN (or missing) reward makes the group's centered advantages NaN in Miles
    (a non-finite loss), and an all-NaN group stops the driver (nonfinite_reward): the
    reader passes such a group over, in the main selection and in the top-up."""
    written = _buffer(tmp_path)
    writer = br.GroupWriter(written, "s0")
    for g, (k, bad) in enumerate(((4, None), (4, float("nan")), (4, None), (3, "none"), (1, None), (4, None))):
        group = _short(g, 1, k)
        for j, d in enumerate(group):
            d["reward"] = float(j % 2)
        if bad == "none":
            group[0]["reward"] = None
        elif bad is not None:
            group[1]["reward"] = bad
        writer.write(group, group_key=f"g{g}")
    written.mark_data_end()
    reader = _reader(tmp_path)
    groups, report = reader.take(0, 2, current_version=1)
    assert [pos for pos, _ in groups] == [0, 2] and report.nonfinite_reward_groups == 1
    groups, report = reader.take(1, 2, current_version=1)
    # g3 (None reward) is passed over in the selection, g4 (one sample) is too small
    assert [pos for pos, _ in groups] == [5] and report.nonfinite_reward_groups == 1
    assert report.dropped_small_groups == 1 and report.metrics()["sample_buffer/nonfinite_reward_groups"] == 1.0
    assert br.finite_rewards([{"reward": {"acc": 1.0}}, {"index": 0}]) is True
    assert br.finite_rewards([{"reward": float("inf")}]) is False


def test_the_reader_waits_while_groups_keep_arriving(tmp_path):
    """wait_s bounds the time without a new group, not the whole round: a round
    that fills slowly (a 1M-context agentic round) is waited for; a stalled one fails."""
    buffer = _buffer(tmp_path)
    writer = br.GroupWriter(buffer, "s0")
    now = [0.0]
    arrivals = iter(range(1, 4))

    def sleep(seconds):
        now[0] += seconds
        if now[0] % 40 == 0:  # one group every 40 s: the round needs 120 s > wait_s
            g = next(arrivals, None)
            if g is not None:
                writer.write(_short(g, 1, 4), group_key=f"g{g}")

    writer.write(_short(0, 1, 4), group_key="g0")
    reader = br.BatchReader(buffer, max_policy_age=1, wait_s=50.0, poll_s=10.0, group_size=4,
                            clock=lambda: now[0], sleep=sleep, prefetch=False)
    groups, report = reader.take(0, 4, current_version=1)
    assert [pos for pos, _ in groups] == [0, 1, 2, 3] and now[0] >= 120
    with pytest.raises(br.SampleBufferRolloutError, match="without a new group"):
        reader.take(1, 1, current_version=1)
    assert 50 < now[0] - 120 <= 70


def test_xhw_downweight_under_std_normalization_is_refused_at_the_start():
    """Not at the first round the guard trips (the trainer would stop mid-run)."""
    env = {"YETO_RL_XHW_GUARD": "action=downweight"}
    with pytest.raises(br.SampleBufferRolloutError, match="action=quarantine"):
        br.check_xhw_downweight(std_normalized=True, environ=env)
    br.check_xhw_downweight(std_normalized=False, environ=env)
    br.check_xhw_downweight(std_normalized=True, environ={"YETO_RL_XHW_GUARD": "1"})
    br.check_xhw_downweight(std_normalized=True, environ={})


def test_padding_is_never_a_copy_of_an_empty_response():
    groups = [_short(0, 1, 2)]
    groups[0][0].update(tokens=[1, 2], response_length=0, loss_mask=[])  # an engine that stopped at once
    (pad,) = br.padding_samples(groups, 1, samples_per_group=4)
    assert pad["response_length"] == 3 and pad["loss_mask"] == [0, 0, 0]
    groups[0][1].update(tokens=[1, 2], response_length=0, loss_mask=[])
    assert br.padding_samples(groups, 1, samples_per_group=4)[0]["response_length"] == 0  # nothing else


def test_a_group_that_does_not_convert_to_miles_samples_is_left_out():
    """One malformed R3 row (pads and ids mixed) leaves its group out of the round
    instead of stopping the trainer; a round where nothing converts still raises."""
    good, bad = _short(0, 1, 2), _short(1, 1, 2)
    routing = numpy.zeros((5, 3, 2), dtype=numpy.uint16)
    bad[1]["rollout_routed_experts"] = routing.copy()
    bad[1]["rollout_routed_experts"][2, 1, 0] = 65535  # one pad next to an id
    good[0]["rollout_routed_experts"] = routing.copy()
    kept, problems = br.miles_groups([(0, good), (1, bad)], samples_per_group=2, r3_num_experts=8)
    assert [pos for pos, _ in kept] == [0] and len(problems) == 1 and "position 1" in problems[0]
    assert [d["index"] for d in kept[0][1]] == [0, 1] and kept[0][1][0]["rollout_routed_experts"].dtype == numpy.int32
    assert br.miles_groups([(1, bad)], samples_per_group=2)[0] == []


def test_a_group_with_a_non_finite_log_prob_is_consumed_not_trained(tmp_path):
    """Miles' TIS / off-policy correction takes exp(old_log_probs - rollout_log_probs):
    one NaN/inf behavior log-prob makes the loss NaN. The reader passes such a group
    over, in the main selection and in the top-up, and counts it apart."""
    from yeto.rl.engine.buffer_payload import nonfinite_logprob

    written = _buffer(tmp_path)
    writer = br.GroupWriter(written, "s0")
    for g, bad in enumerate((None, float("nan"), None, float("-inf"), None, None)):
        group = _short(g, 1, 4)
        if bad is not None:
            group[2]["rollout_log_probs"] = [-0.1, bad, -0.3]
        writer.write(group, group_key=f"g{g}")
    written.mark_data_end()
    reader = _reader(tmp_path)
    groups, report = reader.take(0, 2, current_version=1)
    assert [pos for pos, _ in groups] == [0, 2] and report.nonfinite_logprob_groups == 1
    assert report.nonfinite_reward_groups == 0
    groups, report = reader.take(1, 2, current_version=1)
    assert [pos for pos, _ in groups] == [4, 5] and report.nonfinite_logprob_groups == 1
    assert report.metrics()["sample_buffer/nonfinite_logprob_groups"] == 1.0
    assert nonfinite_logprob([{"rollout_log_probs": [-1.0]}, {"index": 0}]) is None
    assert "sample 3" in nonfinite_logprob([{"index": 3, "rollout_log_probs": [float("nan")]}])


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), float("-inf"), "x"])
@pytest.mark.parametrize("kind", ["list", "numpy", "torch"])
def test_nonfinite_logprob_fast_path_on_a_long_sample(bad, kind):
    """A 1M-token sample is judged by one vectorized check / one sum, and a NaN,
    inf, -inf or non-numeric entry is still found and located."""
    from yeto.rl.engine.buffer_payload import nonfinite_logprob

    if kind != "list" and bad == "x":
        pytest.skip("an array holds no strings")
    values = [-0.5] * 1_000_000
    if bad is not None:
        values[777_777] = bad
    if kind == "numpy":
        values = numpy.asarray(values, dtype=numpy.float32)
    elif kind == "torch":
        values = torch.tensor(values, dtype=torch.float32)
    started = time.perf_counter()
    why = nonfinite_logprob([{"index": 0, "rollout_log_probs": [-1.0]}, {"index": 1, "rollout_log_probs": values}])
    assert time.perf_counter() - started < 1.0
    if bad is None:
        assert why is None
    else:
        assert why is not None and "sample 1 rollout_log_probs[777777]" in why
