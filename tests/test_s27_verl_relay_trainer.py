"""S27 G-EV: the verl relay training island (SPOT-ROLLOUT.md §12.9 item 1).

CPU only. The trainer runs ``trainer.run_fully_async_island`` itself (G-EW's
``verl_trainer`` fixture stubs verl's modules) over the relay ports
(:mod:`yeto.rl.adapters.verl.relay_trainer`) and a local sample buffer filled
the way the relay pump fills it (``relay_buffer.with_cut_metadata``: every
sample names the external merged cut of its engine version). The trainer actor
is G-EW's ``FullBufferActor``. Checks:

* the driver names every group as an external merged cut and counts its age in
  merged versions (``ExternalMergedCuts``): a cut newer than the trainer's is
  kept (age -1), one older than the limit is dropped by the collector, nothing
  is released;
* the island entry sets the relay variables only on a synced full-parameter
  buffer trainer, the layout board only on a synced full-parameter island, and
  refuses both elsewhere (the flags used to be ignored);
* the spot samplers' materializer writes a verl layout's HF names as they are.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from test_rl_verl_buffer_trainer import _sample
from test_s27_verl_sample_role import FullBufferActor, _full_base, _lines, _plan, verl_trainer  # noqa: F401
from yeto.rl.adapters.verl import buffer_feed as bf
from yeto.rl.adapters.verl import island_entry
from yeto.rl.adapters.verl.relay_trainer import RELAY_SECTION_KEY, install_relay_ports
from yeto.rl.engine.buffer_payload import encode_group
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer, SampleIndexEntry
from yeto.rl.spot_rollout.relay_buffer import with_cut_metadata

DIGEST = "ab" * 32


def _relay_group(buffer: FileSampleBuffer, position: int, version: int) -> None:
    """One group as the relay pump appends it (``RelayPump.step``)."""
    samples = [_sample(version, index=i, reward=float(i)) for i in range(2)]
    header = {"behavior_version": version, "cuts": {str(version): {"merged_version": version, "digest": DIGEST}}}
    payload = with_cut_metadata(header, encode_group(samples))
    sample_id = f"spot-{position}"
    ref = buffer.write_payload(sample_id, payload)
    buffer.append(SampleIndexEntry(sample_id=sample_id, behavior_version=version, payload_ref=ref, tokens=6,
                                   provenance={"sampler_id": "spot-a"}, sampler_round=None))


def test_a_relay_trainer_trains_external_merged_cuts_and_releases_nothing(tmp_path, verl_trainer):
    actor = FullBufferActor(seed=0)
    base = _full_base(tmp_path, actor.model)
    store = CheckpointStore(tmp_path / "relay")
    writer = FileSampleBuffer(store, batch_size=1)
    # round 0 (merged 0): v0, and v1 from a cut the trainer has not applied yet (age -1);
    # round 1 (merged 1): v0 (age 1), v1; round 2 (merged 2): v0 (age 2 > 1: dropped), v2, v1
    for position, version in enumerate((0, 1, 0, 1, 0, 2, 1)):
        _relay_group(writer, position, version)
    plan = _plan(tmp_path, base, rounds=3, groups=2)  # no syncer: the merged version is the policy version
    islands = []

    def ports(island):
        islands.append(island)
        return install_relay_ports(island, buffer=FileSampleBuffer(store, batch_size=1, create_index=False),
                                   shape=bf.RowShape(64, 16, 0), tokenize=lambda text: [7, 8], poll_s=0.01, wait_s=5)

    result = verl_trainer.run_fully_async_island(actor, None, plan, {}, buffer_ports=ports)
    assert result["rounds_completed"] == 3 and result["final_policy_version"] == 3
    events = _lines(tmp_path / "tape.jsonl")
    policies = [e for e in events if e["event"] == "rl_group_policies"]
    assert [p["kind"] for p in policies] == ["external-merged-cut"] * 3
    assert [(p["merged_version"], p["ages"]) for p in policies] == [
        (0, {"-1": 1, "0": 1}), (1, {"0": 1, "1": 1}), (2, {"0": 1, "1": 1})]
    # each group is named by the merged versions of its samples
    assert [[g[1] for g in p["groups"]] for p in policies] == [["0", "1"], ["0", "1"], ["2", "1"]]
    consumed = _lines(tmp_path / "out" / "verl-buffer-consumed-0.jsonl")
    assert [(r["position"], r["kept"]) for r in consumed] == [
        (0, True), (1, True), (2, True), (3, True), (4, False), (5, True), (6, True)]
    published = _lines(tmp_path / "out" / "verl-relay-publish-0.jsonl")
    assert [(r["policy_version"], r["released"], r["merged_version"]) for r in published] == [
        (v, False, v) for v in range(4)]
    assert not (tmp_path / "out" / "verl-buffer-release-0.jsonl").exists()
    assert islands[0].full_group.trained_rounds == 3
    assert [len(a) for a in actor.assembled] == [2, 2, 2]
    # the trainer rows carry the merged versions as verl's global steps
    start = next(e for e in events if e["event"] == "rl_verl_island_start")
    assert start["sample_role"] == "trainer" and start["parameter_mode"] == "full"


def test_the_relay_pool_refuses_a_round_without_a_merged_version(tmp_path, verl_trainer):
    from yeto.rl.adapters.verl import fully_async_full
    from yeto.rl.adapters.verl.fully_async_ports import FullyAsyncIsland

    actor = FullBufferActor(seed=0)
    base = _full_base(tmp_path, actor.model)
    plan = _plan(tmp_path, base, rounds=1, groups=1)
    layout_hash, identity = fully_async_full.full_parameter_hashes(plan)
    island = FullyAsyncIsland(actor, learner_id=0, limit=1, required=1, out_dir=str(tmp_path / "out"),
                              readback_dir=str(tmp_path / "rb"), base_model_revision=plan["model_revision"],
                              lora_config_hash=identity, layout_hash=layout_hash)
    pool, publisher, _placement = install_relay_ports(
        island, buffer=FileSampleBuffer(CheckpointStore(tmp_path / "r"), batch_size=1),
        shape=bf.RowShape(64, 16, 0))
    island.published = (0, "0" * 64)
    with pytest.raises(RuntimeError, match="no merged version recorded"):
        pool.generate(0)
    with pytest.raises(RuntimeError, match="before bind"):
        publisher.publish(SimpleNamespace(policy_version=0))


def _entry(**kw):
    values = dict(rl_spot_subscribers=0, rl_sample_relay=False, rl_sample_relay_prefetch=None,
                  parameter_mode="full", rl_single_island_no_sync=False, syncer="10.0.0.1:7000",
                  learner_id=1, groups_per_round=8)
    values.update(kw)
    return SimpleNamespace(**values)


def test_the_island_entry_sets_the_relay_only_on_a_synced_full_parameter_buffer_trainer():
    from yeto.rl.spot_rollout.layout_manifest import PUBLISH_ENV
    from yeto.rl.spot_rollout.relay_buffer import RELAY_ADDR_ENV, RELAY_CHANNEL_ENV, RELAY_PREFETCH_ENV

    assert island_entry.spot_rollout_env(_entry(), None) == {}
    section: dict = {"store": "s"}
    env = island_entry.spot_rollout_env(_entry(rl_sample_relay=True, rl_spot_subscribers=3), section)
    assert env == {PUBLISH_ENV: "1", RELAY_ADDR_ENV: "10.0.0.1:7000", RELAY_CHANNEL_ENV: "1",
                   RELAY_PREFETCH_ENV: "16"}
    assert section[RELAY_SECTION_KEY] == {"channel": 1}
    for change in (dict(parameter_mode="lora"), dict(rl_single_island_no_sync=True), dict(syncer="a:1,b:2")):
        with pytest.raises(SystemExit):
            island_entry.spot_rollout_env(_entry(rl_sample_relay=True, **change), {"store": "s"})
    with pytest.raises(SystemExit, match="缓冲训练岛"):
        island_entry.spot_rollout_env(_entry(rl_sample_relay=True), None)  # not a buffer trainer
    with pytest.raises(SystemExit, match="spot-subscribers"):
        island_entry.spot_rollout_env(_entry(rl_spot_subscribers=2, parameter_mode="lora"), None)
    parsed, ignored = island_entry.parse_args([
        "--model", "m", "--model-revision", "r", "--data", "d", "--learner-id", "0", "--reward-function", "f",
        "--global-rounds", "1", "--groups-per-round", "1", "--samples-per-group", "1",
        "--rollout-max-response-len", "8", "--event-tape", "t", "--inner-lr", "1e-6", "--seq-len", "16",
        "--seed", "1", "--rl-sample-relay", "--rl-sample-relay-prefetch", "4", "--rl-spot-subscribers", "2"])
    assert not ignored and (parsed.rl_sample_relay, parsed.rl_sample_relay_prefetch,
                            parsed.rl_spot_subscribers) == (True, 4, 2)


def test_the_miles_relay_variables_are_unchanged():
    from yeto.rl.adapters.miles.island_entry import sample_relay_env

    env = sample_relay_env(SimpleNamespace(syncer="10.0.0.1:7000", learner_id=2, groups_per_round=4,
                                           rl_sample_relay_prefetch=None), tmp_root="/tmp/gev-relay-test")
    assert env == {"YETO_SAMPLE_BUFFER_ROOT": "/tmp/gev-relay-test/yeto-relay-buffer/island2",
                   "YETO_SAMPLE_RELAY_ADDR": "10.0.0.1:7000", "YETO_SAMPLE_RELAY_CHANNEL": "2",
                   "YETO_SAMPLE_RELAY_PREFETCH": "8"}


def test_the_materializer_writes_a_verl_layouts_hf_names_as_they_are(tmp_path):
    from safetensors.torch import load_file, save_file

    from test_s27_verl_full_parameter_sync import SHAPES, _layout, _Model
    from yeto.rl.spot_rollout.materialize import hf_named_layout, materialize
    from yeto.rl.spot_rollout.subscriber import WeightCut

    layout = _layout(minimum_fragments=2).fragments  # the verl FSDP2 layout: actor::global::<HF name>
    assert hf_named_layout(layout)
    values = _Model(seed=3).values()

    def payload(fragment):
        parts = []
        for name, _numel in fragment.tensors:
            base, _, rows = name.split("::", 2)[2].partition("#rows=")
            tensor = values[base]
            if rows:
                start, stop = (int(x) for x in rows.split(":"))
                tensor = tensor[start:stop]
            parts.append(tensor.to(torch.bfloat16).flatten())
        return torch.cat(parts).view(torch.int16).numpy().tobytes()

    cut = WeightCut(tuple(1 for _ in layout.fragments), tuple(payload(f) for f in layout.fragments))
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    save_file({k: v.to(torch.bfloat16) for k, v in values.items()}, str(base / "model.safetensors"))
    out = materialize(layout, cut, base_model=base, out_root=tmp_path / "out")
    written = load_file(str(out / "model.safetensors"))
    assert sorted(written) == sorted(SHAPES)
    for name, value in values.items():
        assert torch.equal(written[name], value.to(torch.bfloat16))
