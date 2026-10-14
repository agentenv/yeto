"""rl-trainer-start-stop L4.5 (CPU, no Ray, no Miles): the Miles rollout through
the durable sample buffer -- payload codec, sampler writes, trainer selection with
the version gate, the watermark in the round cut, crash and restart in processes."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy
import pytest
import torch
from test_rl_multinode_round_cut import NEXT, _checkpoint, _Driver, _Ledger, _trainer
from test_rl_reconfig_recovery import _ctl

from tests.buffer_rollout_procs import sample
from tests.rl_cut_fakes import make_rank, train_step
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


def test_island_env():
    assert br.island_env("sampler", store_root="/s", sampler_id="s0") == {
        br.BUFFER_ROOT_ENV: "/s", br.SAMPLER_ID_ENV: "s0"}
    assert br.island_env("trainer", store_root="/s", max_policy_age=2, wait_s=30) == {
        br.BUFFER_ROOT_ENV: "/s", br.MAX_POLICY_AGE_ENV: "2", br.WAIT_ENV: "30"}
    with pytest.raises(ValueError):
        br.island_env("sampler", store_root="/s")


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
    assert [g[0]["group_index"] for g in groups] == [1, 3]
    assert (report.stale_groups, report.start, report.end, report.behavior_versions) == (2, 0, 4, {3: 1, 4: 1})


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
    assert [g[0]["group_index"] for g in groups] == [2, 3]


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
