"""S27 G-DS: verl full-parameter islands take the Miles island features that the
unified syncer and the engine-neutral driver already carry.

* bf16 broadcast deltas (``--rl-sync-bcast-delta-full-every``): the
  ``SyncerClient`` decodes them before the verl inbound sink, so the island
  only stops refusing them;
* data positions (``--rl-data-position-store``): the driver records the verl
  data cursor and a relaunched process continues at its next row;
* HF exports (``--rl-export-hf-rounds``): the verl writer exports the FP32
  masters as an HF directory in the model's dtype;
* catch-up: a restarted verl island re-joins a running asynchronous pool,
  takes the syncer's fragments as its anchors and finishes;
* the syncer-only options (checkpoint interval) reach the syncer and leave the
  verl island argv unchanged.

CPU only; the real-syncer tests need cargo.
"""
from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from test_s27_verl_full_parameter_launch import MILES_ONLY, _flags, _island_argv, _prepared, _verl_args
from test_s27_verl_full_parameter_sync import (  # noqa: F401 - syncer_binary is a fixture
    MAX_FRAGMENT_BYTES, SHAPES, TOKENS, _Driver, _expected_average, _Island, _layout, _Model, _plan,
    _run_dict, _start_syncer, _step, needs_cargo, syncer_binary,
)
from yeto.rl.adapters.verl import full_parameter_stream as vfs
from yeto.rl.adapters.verl import island_entry
from yeto.rl.adapters.verl.hf_export import INDEX_FILE, VerlHfWriter, export_dtype_name, plan_shards
from yeto.rl.contracts import LocalStepReceipt as Receipt


@pytest.fixture
def model_dir(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text('{"model_type": "qwen3", "torch_dtype": "bfloat16"}')
    return root


def _section(parsed, model_dir, sync="strict"):
    return vfs.plan_section(parsed, _run_dict(), model_path=str(model_dir), sync=sync,
                            algorithm_spec_sha256="a" * 64)


# ----------------------------------------------------------------------------- bf16 broadcast deltas
@pytest.mark.parametrize("rounds", ["sweep", "async"])
def test_broadcast_deltas_reach_the_verl_stream(model_dir, rounds):
    from yeto import launcher

    args = _prepared(_verl_args("--rl-sync-full-parameter-rounds", rounds, "--rl-sync-bcast-delta-full-every", "4"))
    assert "--bcast-delta-full-every 4" in launcher.role_syncer_commands(args, 2, binary="yeto-syncer")["actor"]
    for learner_id in range(2):
        parsed, ignored = island_entry.parse_args(_island_argv(args, learner_id))
        assert _flags(ignored) <= MILES_ONLY and parsed.bcast_delta
        section = _section(parsed, model_dir)["full_parameter"]
        assert section["bcast_delta"] is True
        config = vfs.role_stream_config(section, syncer="h:1", learner_id=learner_id, model_revision="a" * 40)
        assert config.bcast_delta and config.wire_dtype_label.endswith("+bcast-bf16-delta")
        vfs.refuse_unsupported(config)  # offered now


def test_without_broadcast_deltas_the_plan_has_no_new_key(model_dir):
    args = _prepared(_verl_args())
    parsed, _ignored = island_entry.parse_args(_island_argv(args, 0))
    section = _section(parsed, model_dir)["full_parameter"]
    assert "bcast_delta" not in section and not parsed.bcast_delta
    assert not vfs.role_stream_config(section, syncer="h:1", learner_id=0, model_revision="a" * 40).bcast_delta


def _run_islands(binary, tmp_path, monkeypatch, *, rounds, fragments, async_rounds, extra, plan_extra,
                 matrix_merge="avg", pipeline=1, final_cut="download"):
    from test_s27_verl_full_parameter_sync import _run_island

    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "verl-parity-test-key")
    monkeypatch.setenv("TOKIO_WORKER_THREADS", "4")
    process, port, profile = _start_syncer(binary, tmp_path, fragments=fragments, rounds=rounds, extra=extra)
    results: dict = {}
    try:
        plans = [_plan(i, port, profile, rounds=rounds, fragments=fragments, async_rounds=async_rounds,
                       matrix_merge=matrix_merge, pipeline=pipeline, final_cut=final_cut) for i in range(2)]
        for plan in plans:
            plan[vfs.PLAN_KEY].update(plan_extra)
        threads = [threading.Thread(target=_run_island, args=(i, plans[i], 20 * rounds, results))
                   for i in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        assert process.wait(timeout=30) == 0
    finally:
        if process.poll() is None:
            process.terminate()
        log = process.communicate(timeout=10)[0]
    for learner_id in range(2):
        assert not isinstance(results.get(learner_id), BaseException), (results.get(learner_id), log[-3000:])
        assert learner_id in results, log[-3000:]
    return results, log


@needs_cargo
def test_two_verl_islands_take_bf16_broadcast_deltas_from_the_real_syncer(syncer_binary, tmp_path, monkeypatch):
    """Strict sweep, FULL every second broadcast and bf16 DELTA in between: the
    deltas reach the verl islands (counted at the decoder) and the downloaded
    final cut is the exact average merge, as without deltas."""
    from yeto import downlink

    kinds: list[int] = []
    begin = downlink.DownlinkAnchors.begin

    def counted(self, fragment_id, version, header, numel):
        kinds.append(downlink.HEADER.unpack(header)[0])
        return begin(self, fragment_id, version, header, numel)

    monkeypatch.setattr(downlink.DownlinkAnchors, "begin", counted)
    rounds, fragments = 4, len(_layout().fragments.fragments)
    results, _log = _run_islands(syncer_binary, tmp_path, monkeypatch, rounds=rounds, fragments=fragments,
                                 async_rounds=False, plan_extra={"bcast_delta": True},
                                 extra=("--rl-sync-full-parameter-rounds", "sweep",
                                        "--rl-sync-final-cut", "download",
                                        "--rl-sync-bcast-delta-full-every", "2"))
    assert downlink.KIND_DELTA in kinds and downlink.KIND_FULL in kinds, kinds
    expected = _expected_average(rounds)
    for learner_id in range(2):
        for name, value in expected.items():
            assert torch.allclose(results[learner_id]["values"][name], value, atol=1e-6), (learner_id, name)
        acks = [f for e, f in results[learner_id]["events"] if e == "rl_full_parameter_final_ack"]
        assert acks == [{"acknowledged": True}]


@needs_cargo
def test_bf16_broadcast_deltas_with_async_rounds_and_local_init(syncer_binary, tmp_path, monkeypatch):
    """The async defaults (quorum 1, two rounds in flight, RDA, local INIT) with
    deltas: no island uploads INIT, the first broadcast of each fragment sets the
    decoder's anchor, both islands ACK the same downloaded final cut."""
    from yeto import downlink

    kinds: list[int] = []
    begin = downlink.DownlinkAnchors.begin

    def counted(self, fragment_id, version, header, numel):
        kinds.append(downlink.HEADER.unpack(header)[0])
        return begin(self, fragment_id, version, header, numel)

    monkeypatch.setattr(downlink.DownlinkAnchors, "begin", counted)
    rounds, fragments = 4, len(_layout(matrix_merge="rda").fragments.fragments)
    results, log = _run_islands(syncer_binary, tmp_path, monkeypatch, rounds=rounds, fragments=fragments,
                                async_rounds=True, plan_extra={"bcast_delta": True}, matrix_merge="rda",
                                pipeline=2, extra=("--rl-sync-full-parameter-rounds", "async",
                                                   "--rl-sync-final-cut", "download",
                                                   "--rl-sync-bcast-delta-full-every", "2"))
    assert downlink.KIND_DELTA in kinds and downlink.KIND_FULL in kinds, kinds
    for learner_id in range(2):
        acks = [f for e, f in results[learner_id]["events"] if e == "rl_full_parameter_final_ack"]
        assert acks == [{"acknowledged": True}], (learner_id, log[-2000:])
    for name in SHAPES:
        assert torch.equal(results[0]["values"][name], results[1]["values"][name]), name


# ----------------------------------------------------------------------------- data positions and HF exports
def _parsed(extra: list[str], *, parameter_mode="full", sync_flag=()):
    argv = ["--model", "m", "--model-revision", "a" * 40, "--data", "d", "--learner-id", "1",
            "--reward-function", "gsm8k_reward:score", "--global-rounds", "3", "--groups-per-round", "2",
            "--samples-per-group", "2", "--rollout-max-response-len", "16", "--event-tape", "t",
            "--inner-lr", "1e-6", "--seq-len", "64", "--seed", "1", "--parameter-mode", parameter_mode,
            *sync_flag, *extra]
    return island_entry.parse_args(argv)[0]


def test_the_island_plan_carries_exports_and_data_positions_only_when_given(model_dir):
    assert vfs.full_parameter_extras(_parsed([]), sync="strict") == {}
    parsed = _parsed(["--rl-export-hf-rounds", "2,final", "--rl-export-hf-store", "/s/hf",
                      "--rl-data-position-store", "/s"])
    extras = {"hf_exports": {"rounds": "2,final", "store": "/s/hf/island1"}, "data_position_store": "/s"}
    assert vfs.full_parameter_extras(parsed, sync="strict") == extras
    # plan_section puts them at the plan's top level (island_entry merges it into the plan)
    no_sync = _parsed(["--rl-export-hf-rounds", "final", "--rl-export-hf-store", "/s/hf"])
    plan = vfs.plan_section(no_sync, _run_dict(), model_path=str(model_dir), sync="none",
                            algorithm_spec_sha256="a" * 64)
    assert plan["hf_exports"] == {"rounds": "final", "store": "/s/hf/island1"} and "full_parameter" not in plan
    plain = vfs.plan_section(_parsed([]), _run_dict(), model_path=str(model_dir), sync="none",
                             algorithm_spec_sha256="a" * 64)
    assert set(plain) == {"model_config_sha256"}


@pytest.mark.parametrize("extra, sync, match", [
    (["--rl-export-hf-rounds", "final"], "strict", "go together"),
    (["--rl-data-position-store", "/s"], "none", "synced full-parameter"),
])
def test_the_island_refuses_exports_and_data_positions_where_they_do_not_apply(extra, sync, match):
    with pytest.raises(ValueError, match=match):
        vfs.full_parameter_extras(_parsed(extra), sync=sync)


def test_the_export_flags_are_defined_once_when_the_island_parser_has_them():
    """The verl Megatron island (S27 G-DC) parses the export flags in its own
    parser; add_island_flags then adds only the data position store."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rl-export-hf-rounds", default=None)
    parser.add_argument("--rl-export-hf-store", default=None)
    vfs.add_island_flags(parser)
    args = parser.parse_args(["--rl-export-hf-rounds", "final", "--rl-data-position-store", "/s"])
    assert (args.rl_export_hf_rounds, args.rl_data_position_store) == ("final", "/s")


def test_the_driver_gets_the_data_position_log_and_the_exports(tmp_path, model_dir):
    from yeto.rl.adapters.verl.driver_extras import wire_full_parameter_extras
    from yeto.rl.engine.data_position import DataPositionLog
    from yeto.rl.engine.hf_export import HfRoundExports

    island = _Island(_Model())
    island.full_group = vfs.VerlFullParameterGroup(island)
    driver = SimpleNamespace(data_positions=None, hf_exports=None)
    wire_full_parameter_extras(driver, island, {"learner_id": 1, "data_slot": None, "model_path": str(model_dir)})
    assert driver.data_positions is None and driver.hf_exports is None
    plan = {"learner_id": 1, "data_slot": "1/2", "model_path": str(model_dir),
            "data_position_store": str(tmp_path / "store"),
            "hf_exports": {"rounds": "final", "store": str(tmp_path / "store" / "hf" / "island1")}}
    wire_full_parameter_extras(driver, island, plan)
    assert isinstance(driver.data_positions, DataPositionLog) and driver.data_positions.slot == "1/2"
    assert isinstance(driver.hf_exports, HfRoundExports) and str(driver.hf_exports.rounds) == "final"


def test_a_recorded_verl_data_cursor_is_where_a_new_process_seeks_to(tmp_path):
    """The verl pool's cursor (prompts drawn through ``_fetch_one_gen_batch``)
    goes into the log after a boundary and a new process lands on it."""
    from yeto.rl.adapters.verl.data_cursor import VerlDataCursor
    from yeto.rl.engine.data_position import data_position_log

    class _Trainer:
        def __init__(self):
            self.config = SimpleNamespace(data={"gen_batch_size": 4})
            self.fetched = 0

        def _fetch_one_gen_batch(self):
            self.fetched += 1
            return [object()] * 4

    first = VerlDataCursor(_Trainer())
    log = data_position_log(str(tmp_path / "store"), learner_id=1, slot="1/2", environ={})
    log.start(first.cursor())
    for round_id in range(3):
        first._draw()
        log.record(round_id, first.cursor())
    log.close()
    trainer = _Trainer()
    second = VerlDataCursor(trainer)
    record = data_position_log(str(tmp_path / "store"), learner_id=1, slot="1/2", environ={}).last()
    assert record.round == 2 and second.seek(record.cursor) == {"sample_offset": 12} and trainer.fetched == 3


def test_the_launcher_forwards_exports_and_data_positions_to_verl_islands(model_dir):
    store = "modal-volume://v/run"
    args = _prepared(_verl_args("--rl-checkpoint-store", store, "--rl-export-hf-rounds", "2,final"))
    assert args.rl_peer_recovery is False and args.rl_island_state_source == "catch-up"
    for learner_id in range(2):
        argv = _island_argv(args, learner_id)
        parsed, ignored = island_entry.parse_args(argv)
        assert _flags(ignored) <= MILES_ONLY, _flags(ignored) - MILES_ONLY
        extras = vfs.full_parameter_extras(parsed, sync="strict")
        assert extras["hf_exports"]["rounds"] == "2,final"
        assert extras["hf_exports"]["store"].endswith(f"/hf/island{learner_id}")
        assert extras["data_position_store"] == parsed.rl_data_position_store
        _section(parsed, model_dir)  # the sync section still plans


def test_a_catch_up_store_alone_keeps_data_positions():
    """No export, no cuts, no peer recovery: the store holds the islands' data
    positions (asynchronous rounds; a relaunched island takes the catch-up)."""
    args = _prepared(_verl_args("--rl-checkpoint-store", "modal-volume://v/run"))
    parsed, ignored = island_entry.parse_args(_island_argv(args, 1))
    assert _flags(ignored) <= MILES_ONLY and parsed.rl_data_position_store
    with pytest.raises(ValueError, match="--rl-checkpoint-store"):  # a strict sweep has no catch-up
        _prepared(_verl_args("--rl-sync-full-parameter-rounds", "sweep",
                             "--rl-checkpoint-store", "modal-volume://v/run"))


def test_a_verl_lora_launch_still_refuses_hf_exports():
    with pytest.raises(ValueError, match="verl: synced full-parameter islands"):
        _prepared(_launcher_args("ports", ("--rl-backend", "verl", "--rl-checkpoint-store", "modal-volume://v/run",
                                           "--rl-export-hf-rounds", "final"), gpu="modal:1xh200,modal:1xh200"))


# ----------------------------------------------------------------------------- the HF writer
def test_the_writer_exports_every_parameter_in_the_model_dtype(tmp_path, model_dir):
    from safetensors.torch import load_file

    (model_dir / "tokenizer.json").write_text("{}")
    (model_dir / "model.safetensors").write_text("not copied")
    (model_dir / "model.safetensors.index.json").write_text("{}")
    model = _Model(seed=3)
    island = _Island(model)
    group = vfs.VerlFullParameterGroup(island)
    # 64 bytes per file: the 12x4 embedding (96 bf16 bytes) is a file of its own
    writer = VerlHfWriter(group, model_dir, run=vfs._LoopRunner().run, shard_bytes=64)
    out = tmp_path / "round-3"
    writer(2, out)
    index = json.loads((out / INDEX_FILE).read_text())
    assert sorted(index["weight_map"]) == sorted(SHAPES)
    loaded = {}
    for file_name in sorted(set(index["weight_map"].values())):
        loaded.update(load_file(str(out / file_name)))
    for name, value in model.values().items():
        assert loaded[name].dtype == torch.bfloat16 and torch.equal(loaded[name], value.to(torch.bfloat16)), name
    assert index["metadata"]["total_size"] == sum(t.numel() * 2 for t in loaded.values())
    assert (out / "config.json").read_text() == (model_dir / "config.json").read_text()
    assert (out / "tokenizer.json").is_file() and (out / ".complete").is_file()
    assert (out / "model.safetensors").exists() is False


def test_one_file_when_the_model_fits_and_the_round_lands_in_the_store(tmp_path, model_dir):
    from safetensors.torch import load_file

    from yeto.rl.adapters.verl.hf_export import hf_round_exports

    island = _Island(_Model(seed=4))
    group = vfs.VerlFullParameterGroup(island)
    store = tmp_path / "store" / "hf" / "island0"
    exports = hf_round_exports({"rounds": "final", "store": str(store)}, group=group, model_path=model_dir,
                               run=vfs._LoopRunner().run, environ={})
    assert exports.due(5, final=True) and not exports.due(5, final=False)
    record = exports.export(5, policy_version=6, final=True)
    assert record["ok"] and record["round"] == 6, record
    landed = store / "round-6"
    assert (landed / ".complete").is_file() and not (landed / INDEX_FILE).exists()
    assert sorted(load_file(str(landed / "model.safetensors"))) == sorted(SHAPES)
    assert json.loads((store / "final.json").read_text())["round"] == 6


def test_the_export_dtype_follows_the_config(tmp_path):
    for text, dtype in (('{"torch_dtype": "float16"}', "float16"), ('{"dtype": "bfloat16"}', "bfloat16"),
                        ("{}", "bfloat16"), ('{"torch_dtype": "float32"}', "float32")):
        (tmp_path / "config.json").write_text(text)
        assert export_dtype_name(tmp_path) == dtype
    (tmp_path / "config.json").write_text('{"torch_dtype": "int8"}')
    with pytest.raises(ValueError, match="int8"):
        export_dtype_name(tmp_path)
    assert plan_shards({"b": (2,), "a": (3,), "c": (8,)}, itemsize=2, shard_bytes=10) == [["a", "b"], ["c"]]


# ----------------------------------------------------------------------------- catch-up
def _pool_plan(learner_id, port, profile, *, rounds, fragments):
    plan = _plan(learner_id, port, profile, rounds=rounds, fragments=fragments, async_rounds=True,
                 matrix_merge="rda", pipeline=2, final_cut="download")
    plan[vfs.PLAN_KEY]["island_pool"] = {"lease_s": 3.0, "syncer_epoch": 0}
    return plan


def _train_rounds(learner_id, plan, results, *, stop_after=None, pace_s=0.05, max_rounds=4000):
    """One verl island process: train and sync until the final cut, or die
    (close its links, no LEAVE) after ``stop_after`` rounds. ``pace_s`` per
    local round: quorum 1 lets an island run far ahead of the syncer's steps."""
    model = _Model(seed=0)
    island = _Island(model)
    island.full_group = vfs.VerlFullParameterGroup(island)
    sync, _policy_state, stream = vfs.build_full_parameter_sync(island, plan, backend_identity_sha256="c" * 64)
    driver = _Driver()
    record = {"model": model, "events": driver.events, "stream": stream}
    try:
        state = sync.start(driver).state
        record["joined"] = sync.joined_running_pool
        record["versions_at_start"] = list(stream.fragment_versions)
        record["anchors_equal_checkpoint"] = all(
            torch.equal(model.values()[name], value) for name, value in _Model(seed=0).values().items())
        for round_id in range(max_rounds):
            if stop_after is not None and round_id == stop_after:
                record["died"] = True
                return
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.add_(_step(learner_id, round_id)[name])
            ids = (f"l{learner_id}-p{id(record)}-r{round_id}",)
            batch_hash = hashlib.sha256(ids[0].encode()).hexdigest()
            policy_hash = state.policy_tensor_hash()
            island.full_group.record_round(
                Receipt(algorithm="grpo", learner_id=learner_id, learner_generation=0,
                        base_policy_version=round_id, base_policy_hash=policy_hash, input_batch_hash=batch_hash,
                        trajectory_ids=ids, trained_tokens=TOKENS, optimizer_steps=1,
                        optimizer_step_succeeded=True,
                        parameter_layout_hash=plan[vfs.PLAN_KEY]["parameter_layout_sha256"]),
                vfs.VerlRoundEvidence(round_id, policy_hash, batch_hash, TOKENS, ids))
            boundary = sync.boundary(driver, rollout_id=round_id, stats=None)
            state = boundary.state
            if boundary.stop:
                sync.published(driver, rollout_id=round_id, policy_hash=state.policy_tensor_hash())
                break
            if pace_s:
                time.sleep(pace_s)
        sync.finish(driver)
        record["values"] = model.values()
        record["final_ack"] = [f for e, f in driver.events if e == "rl_full_parameter_final_ack"]
    except BaseException as error:  # noqa: BLE001 - reported by the test
        record["error"] = error
    finally:
        sync.close()
        results.setdefault(learner_id, []).append(record)


@needs_cargo
def test_a_restarted_verl_island_catches_up_from_the_running_pool_and_finishes(syncer_binary, tmp_path,
                                                                               monkeypatch):
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "verl-parity-catch-up-key")
    monkeypatch.setenv("TOKIO_WORKER_THREADS", "4")
    rounds, fragments = 12, len(_layout(matrix_merge="rda").fragments.fragments)
    process, port, profile = _start_syncer(syncer_binary, tmp_path, fragments=fragments, rounds=rounds, extra=(
        "--rl-sync-full-parameter-rounds", "async", "--rl-sync-final-cut", "download",
        "--rl-sync-quorum", "1", "--rl-sync-island-lease-s", "3"))
    results: dict = {}
    try:
        plans = [_pool_plan(i, port, profile, rounds=rounds, fragments=fragments) for i in range(2)]
        steady = threading.Thread(target=_train_rounds, args=(0, plans[0], results))
        steady.start()
        first = threading.Thread(target=_train_rounds, args=(1, plans[1], results), kwargs={"stop_after": 3})
        first.start()
        first.join(timeout=60)
        assert results[1][0].get("died") and "error" not in results[1][0], results[1][0].get("error")
        again = threading.Thread(target=_train_rounds, args=(1, plans[1], results))
        again.start()
        for thread in (steady, again):
            thread.join(timeout=120)
        assert process.wait(timeout=30) == 0
    finally:
        if process.poll() is None:
            process.terminate()
        log = process.communicate(timeout=10)[0]
    restarted = results[1][1]
    assert "error" not in restarted, (restarted.get("error"), log[-3000:])
    assert "error" not in results[0][0], (results[0][0].get("error"), log[-3000:])
    # The new process joined a running pool: it took the syncer's fragments
    # (newer than version 0), not the checkpoint weights, as its anchors.
    assert restarted["joined"] is True and max(restarted["versions_at_start"]) >= 1, restarted["versions_at_start"]
    assert restarted["anchors_equal_checkpoint"] is False
    assert restarted["final_ack"] == [{"acknowledged": True}] == results[0][0]["final_ack"]
    for name in SHAPES:  # both downloaded the same final cut
        assert torch.equal(restarted["values"][name], results[0][0]["values"][name]), name
    records = [json.loads(line) for line in (tmp_path / "tape.jsonl").read_text().splitlines() if line]
    island_1 = [r for r in records if r.get("island_id") == 1]
    rejoin = next(i for i, r in enumerate(island_1) if r.get("kind") == "pool_join" and r.get("catch_up"))
    assert any(r["kind"] == "delta_accepted" for r in island_1[rejoin + 1:]), "the restarted island never contributed"


@needs_cargo
def test_the_lead_cap_holds_a_fast_verl_island_to_the_slow_one(syncer_binary, tmp_path, monkeypatch):
    """K = 1 with a fixed roster (quorum 2): the fast island waits at its
    boundaries for the slow island's pushes; the cap fields reach its tape."""
    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "verl-parity-lead-cap-key")
    monkeypatch.setenv("TOKIO_WORKER_THREADS", "4")
    rounds, fragments = 4, len(_layout(matrix_merge="rda").fragments.fragments)
    process, port, profile = _start_syncer(syncer_binary, tmp_path, fragments=fragments, rounds=rounds, extra=(
        "--rl-sync-full-parameter-rounds", "async", "--rl-sync-final-cut", "download",
        "--rl-sync-quorum", "2", "--rl-sync-max-local-rounds-ahead", "1"))
    results: dict = {}
    try:
        plans = []
        for i in range(2):
            plan = _plan(i, port, profile, rounds=rounds, fragments=fragments, async_rounds=True,
                         matrix_merge="rda", pipeline=2, final_cut="download")
            plan[vfs.PLAN_KEY]["max_local_rounds_ahead"] = 1
            plans.append(plan)
        threads = [threading.Thread(target=_train_rounds, args=(0, plans[0], results), kwargs={"pace_s": 0.0}),
                   threading.Thread(target=_train_rounds, args=(1, plans[1], results), kwargs={"pace_s": 0.3})]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=180)
        assert process.wait(timeout=30) == 0
    finally:
        if process.poll() is None:
            process.terminate()
        log = process.communicate(timeout=10)[0]
    fast, slow = results[0][0], results[1][0]
    assert "error" not in fast and "error" not in slow, (fast.get("error"), slow.get("error"), log[-2000:])
    fast_rounds = [f for e, f in fast["events"] if e == "rl_full_parameter_round"]
    slow_rounds = [f for e, f in slow["events"] if e == "rl_full_parameter_round"]
    assert all(f["sync/max_local_rounds_ahead"] == 1 for f in fast_rounds)
    assert fast_rounds[-1]["sync/lead_waits"] > 0, fast_rounds[-1]
    # Without the cap the unpaced island would run hundreds of local rounds meanwhile.
    assert len(fast_rounds) <= len(slow_rounds) + 2, (len(fast_rounds), len(slow_rounds))
    assert fast["final_ack"] == slow["final_ack"] == [{"acknowledged": True}]


# ----------------------------------------------------------------------------- syncer-only options
def test_syncer_checkpoint_options_reach_the_syncer_and_leave_the_verl_islands_unchanged():
    from yeto import launcher

    sweep = ("--rl-sync-full-parameter-rounds", "sweep")
    for base_flags, tuned_flags, needles in (
            ((), ("--rl-sync-checkpoint-every", "4"), ("--checkpoint-every 4",)),
            (sweep, (*sweep, "--rl-sync-checkpoint-every", "8", "--rl-sync-async-checkpoint"),
             ("--checkpoint-every 8", "--async-checkpoint"))):
        base, tuned = _prepared(_verl_args(*base_flags)), _prepared(_verl_args(*tuned_flags))
        command = launcher.role_syncer_commands(tuned, 2, binary="yeto-syncer")["actor"]
        before = launcher.role_syncer_commands(base, 2, binary="yeto-syncer")["actor"]
        for needle in needles:
            assert needle in command and needle not in before, needle
        for learner_id in range(2):  # the profile hash binds the interval; nothing else moves
            strip = lambda argv: [t for t in argv if not t.startswith(("--syncer-profile-sha256",))]  # noqa: E731
            tuned_argv, base_argv = _island_argv(tuned, learner_id), _island_argv(base, learner_id)
            i, j = tuned_argv.index("--syncer-profile-sha256"), base_argv.index("--syncer-profile-sha256")
            assert strip(tuned_argv[:i] + tuned_argv[i + 2:]) == strip(base_argv[:j] + base_argv[j + 2:])


# ----------------------------------------------------------------------------- nothing ignored
@pytest.mark.parametrize("extra", [
    ("--rl-verify-every-round",), ("--rl-lr-schedule", "constant"), ("--rl-max-tokens-per-gpu", "4096"),
    ("--rl-deterministic-trainer",), ("--rl-activation-recompute",), ("--rl-gradient-accumulation-fusion",),
    ("--rl-async-buffer-rounds", "2"), ("--rl-lora-dropout", "0.1"), ("--rl-observe-timeline",),
    ("--rl-profile-trainer",), ("--rl-debug-dump",), ("--rl-resource-sample-interval", "5"),
])
def test_options_a_verl_full_parameter_island_would_ignore_are_refused(extra):
    """Each of these reached the verl island and was dropped by parse_known_args."""
    with pytest.raises(ValueError, match="would ignore it"):
        _prepared(_verl_args(*extra))


def test_every_refused_option_is_one_the_verl_island_does_not_parse():
    from yeto.rl.adapters.verl import launch_flags

    parser_options = set(island_entry.parse_args(_parsed_argv())[0].__dict__)
    for dest, _off, _option in launch_flags.IGNORED_ON_VERL_FULL_PARAMETER:
        assert dest not in parser_options, dest


def _parsed_argv():
    return ["--model", "m", "--model-revision", "a" * 40, "--data", "d", "--learner-id", "0",
            "--reward-function", "r", "--global-rounds", "1", "--groups-per-round", "1",
            "--samples-per-group", "1", "--rollout-max-response-len", "1", "--event-tape", "t",
            "--inner-lr", "1", "--seq-len", "1", "--seed", "1"]
