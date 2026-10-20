"""S25 C2: verl trainer island over the sample buffer (CPU, no verl/Ray).

The trainer actor is a fake reached through the island's ``call``; the buffer,
the release store, the yeto driver, the fully_async rollout port (version map,
policy-age judgement), the feed, the publisher and the merged checkpoint are
the real ones. The end-to-end test runs the real engine sampler island (fake
engine) in a thread against the same store.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from yeto.rl.adapters.sampler_engines.fake import FakeEngine
from yeto.rl.adapters.verl import buffer_feed as bf
from yeto.rl.adapters.verl import fully_async_round as far
from yeto.rl.adapters.verl.buffer_ports import (CONSUMED_LOG, RELEASE_LOG, BufferReleasePublisher,
                                                install_buffer_ports, is_base_adapter)
from yeto.rl.adapters.verl.fully_async_translate import VersionMap
from yeto.rl.adapters.verl.merged_release import MergeError, lora_pairs, write_merged_checkpoint
from yeto.rl.engine.algorithm import AlgorithmSpec, CorrectionSpec, ExecutionSpec
from yeto.rl.engine.buffer_payload import GroupWriter
from yeto.rl.engine.engine_sampler import EngineSamplerConfig, PromptItem, build_engine_sampler
from yeto.rl.engine.policy_release import PolicyReleaseStore
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer
from yeto.rl.engine.sampler_checks import PostLoadChecks
from yeto.rl.engine.sampler_engine import SamplingParams

TOLERANT = AlgorithmSpec(execution=ExecutionSpec(max_policy_staleness=1),
                         correction=CorrectionSpec(method="tis", tis_clip=2.0, tis_clip_low=0.0))
MODULE = "model.layers.0.q_proj"
NAMES = (f"base_model.model.{MODULE}.lora_A.weight", f"base_model.model.{MODULE}.lora_B.weight")
SHAPE = bf.RowShape(prompt_length=6, response_length=8, pad_token_id=0)


def _sample(version, *, prompt=(11, 12), response=(21, 22, 23), reward=1.0, known=True, index=0,
            provenance=None):
    tokens = (list(prompt) if known else []) + list(response)
    return {"index": index, "prompt": "p", "tokens": tokens, "response_length": len(response),
            "rollout_log_probs": [-0.5] * len(response), "reward": reward,
            "metadata": {"weight_version": str(version), "prompt_tokens_known": known,
                         "finish_reason": "stop", "provenance": provenance or {"engine": "fake"}}}


# ---------------------------------------------------------------------- rows
def test_group_rows_pad_like_the_agent_loop():
    rows = bf.group_rows([_sample(3, index=0), _sample(3, response=(31,), reward=0.0, index=1)],
                         uid="buf7", param_versions=[2, 2], shape=SHAPE)
    t, nt = rows["tensors"], rows["non_tensor"]
    assert t["prompts"][0] == [0, 0, 0, 0, 11, 12]
    assert t["responses"][0] == [21, 22, 23, 0, 0, 0, 0, 0]
    assert t["response_mask"][1] == [1, 0, 0, 0, 0, 0, 0, 0]
    assert t["input_ids"][0] == [0, 0, 0, 0, 11, 12, 21, 22, 23, 0, 0, 0, 0, 0]
    assert t["attention_mask"][0] == [0] * 4 + [1] * 5 + [0] * 5
    assert t["position_ids"][0] == [0, 0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4, 4, 4]
    assert t["rm_scores"][0][:4] == [0.0, 0.0, 1.0, 0.0] and sum(t["rm_scores"][1]) == 0.0
    assert t["rollout_log_probs"][0][:4] == [-0.5, -0.5, -0.5, 0.0]
    assert nt["uid"] == ["buf7", "buf7"] and nt["min_global_steps"] == [2, 2] == nt["max_global_steps"]
    assert nt["provenance"][0] == {"engine": "fake"} and nt["finish_reason"] == ["stop", "stop"]
    assert len(rows["meta_info"]["metrics"]) == 2


def test_group_rows_rebuild_unknown_prompt_ids_with_the_trainer_tokenizer():
    rows = bf.group_rows([_sample(0, known=False)], uid="u", param_versions=[0], shape=SHAPE,
                         tokenize=lambda text: [5, 6, 7])
    assert rows["tensors"]["prompts"][0] == [0, 0, 0, 5, 6, 7]
    with pytest.raises(Exception, match="no tokenizer"):
        bf.group_rows([_sample(0, known=False)], uid="u", param_versions=[0], shape=SHAPE)


def test_group_rows_refuse_samples_wider_than_the_rows():
    with pytest.raises(bf.BufferFeedError, match="exceed"):
        bf.group_rows([_sample(0, prompt=tuple(range(1, 8)))], uid="u", param_versions=[0], shape=SHAPE)
    with pytest.raises(bf.BufferFeedError, match="param versions"):
        bf.group_rows([_sample(0)], uid="u", param_versions=[], shape=SHAPE)


# ---------------------------------------------------------------------- feed
def _buffer(tmp_path):
    return FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1)


def test_feed_maps_versions_judges_and_records_each_position_once(tmp_path):
    writer = GroupWriter(_buffer(tmp_path), "s0")
    for key, version in (("a", 0), ("b", 1), ("c", 5)):
        writer.write([_sample(version, index=0), _sample(version, index=1)], group_key=key)
    reader, vmap, added, records = _buffer(tmp_path), VersionMap(), [], []
    vmap.record(0, 0)
    vmap.record(1, 1)
    feed = bf.BufferFeed(reader, vmap=lambda: vmap, shape=SHAPE, add_group=added.append,
                         record=records.append, sleep=lambda s: None)
    collector = far.RoundCollector(vmap, 1, 1, 3)
    for _ in range(3):
        meta = feed.pull()
        keep = collector.offer(far.SampleMeta(**meta))
        feed("yeto_keep_last" if keep else "yeto_drop_last")
    assert [r["position"] for r in records] == [0, 1, 2] == feed.pulled_positions
    assert [r["kept"] for r in records] == [True, True, False]  # v5 was never published here
    assert records[2]["param_versions"] == [None, None]
    assert [rows["non_tensor"]["uid"][0] for rows in added] == ["buf0", "buf1"]
    assert collector.discarded[0].unknown
    assert feed.read_cursor() == 3


def test_feed_returns_none_at_the_data_end_and_times_out_without_data(tmp_path):
    buffer = _buffer(tmp_path)
    clock = iter(range(0, 100, 10))
    feed = bf.BufferFeed(buffer, vmap=VersionMap, shape=SHAPE, add_group=print, record=print,
                         wait_s=15, clock=lambda: next(clock), sleep=lambda s: None)
    with pytest.raises(bf.BufferFeedError, match="no buffer group within 15 s"):
        feed.pull()
    buffer.mark_data_end()
    assert bf.BufferFeed(_buffer(tmp_path), vmap=VersionMap, shape=SHAPE, add_group=print,
                         record=print).pull() is None
    with pytest.raises(bf.BufferFeedError, match="without a pulled group"):
        feed.keep()


def test_buffer_call_routes_queue_calls_to_the_feed():
    seen = []
    call = bf.BufferCall(lambda name, *a: seen.append(("feed", name)), lambda name, *a, **k: seen.append(("actor", name)))
    call("yeto_pull_sample")
    call("yeto_train_step")
    assert seen == [("feed", "yeto_pull_sample"), ("actor", "yeto_train_step")]


# ---------------------------------------------------------------------- merge
def _base(tmp_path, *, dtype=torch.bfloat16):
    base = tmp_path / "base"
    base.mkdir()
    weights = {f"{MODULE}.weight": torch.randn(4, 4).to(dtype), "lm_head.weight": torch.randn(3, 4).to(dtype)}
    save_file(weights, str(base / "model.safetensors"))
    (base / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    (base / "tokenizer.json").write_text("{}")
    return base, weights


def test_merge_adds_b_times_a_and_keeps_the_layout(tmp_path):
    base, weights = _base(tmp_path, dtype=torch.float32)
    a, b = torch.randn(2, 4), torch.randn(4, 2)
    summary = write_merged_checkpoint(base, {NAMES[0]: a, NAMES[1]: b}, tmp_path / "out")
    out = load_file(str(tmp_path / "out" / "model.safetensors"))
    assert torch.allclose(out[f"{MODULE}.weight"], weights[f"{MODULE}.weight"] + b @ a, atol=1e-6)
    assert torch.equal(out["lm_head.weight"], weights["lm_head.weight"])
    assert summary["merged_modules"] == 1 and summary["max_relative_change"] > 0
    assert (tmp_path / "out" / "config.json").read_text() == (base / "config.json").read_text()
    assert (tmp_path / "out" / "tokenizer.json").is_file()


def test_merge_keeps_the_base_dtype_and_refuses_unknown_or_misshaped_modules(tmp_path):
    base, _ = _base(tmp_path)
    write_merged_checkpoint(base, {NAMES[0]: torch.ones(2, 4), NAMES[1]: torch.ones(4, 2)}, tmp_path / "o1")
    assert load_file(str(tmp_path / "o1" / "model.safetensors"))[f"{MODULE}.weight"].dtype == torch.bfloat16
    other = "base_model.model.model.layers.9.q_proj"
    with pytest.raises(MergeError, match="no weight for 1 LoRA modules"):
        write_merged_checkpoint(base, {f"{other}.lora_A.weight": torch.ones(2, 4),
                                       f"{other}.lora_B.weight": torch.ones(4, 2)}, tmp_path / "o2")
    with pytest.raises(MergeError, match="do not fit"):
        write_merged_checkpoint(base, {NAMES[0]: torch.ones(2, 3), NAMES[1]: torch.ones(4, 2)}, tmp_path / "o3")
    with pytest.raises(MergeError, match="only side"):
        lora_pairs({NAMES[0]: torch.ones(2, 4)})


def test_base_adapter_is_version_zero_with_zero_b():
    zero = {NAMES[0]: torch.ones(2, 4), NAMES[1]: torch.zeros(4, 2)}
    assert is_base_adapter(0, zero)
    assert not is_base_adapter(1, zero)
    assert not is_base_adapter(0, {**zero, NAMES[1]: torch.full((4, 2), 1e-3)})


# ---------------------------------------------------------------------- driver end to end
class FakeBufferActor:
    """Stands in for YetoBufferTrainer: kept rows, a LoRA pair, a param version."""

    def __init__(self):
        self.tensors = {NAMES[0]: torch.full((2, 4), 0.1), NAMES[1]: torch.zeros(4, 2)}
        self.param_version = 0
        self.kept, self.assembled = [], []

    def __call__(self, name, *args, timeout=None):
        return getattr(self, name)(*args)

    def yeto_configure(self, *_):
        return None

    def yeto_add_group(self, rows):
        self.kept.append(rows)
        return len(self.kept)

    def yeto_assemble(self):
        self.assembled.append([r["non_tensor"]["uid"][0] for r in self.kept])
        n, self.kept = len(self.kept), []
        return {"samples": n, "queue_size": None}

    def yeto_train_step(self):
        self.tensors[NAMES[1]] += 0.01
        self.param_version += 1
        return {"metrics": {"actor/grad_norm": 0.5, "actor/pg_loss": 0.1, "actor/lr": 1e-5},
                "criteria": {}, "param_version": self.param_version}

    def yeto_param_version(self):
        return self.param_version

    def yeto_export_lora(self):
        return {k: v.clone() for k, v in self.tensors.items()}, {"count": 2}

    def yeto_call_workers(self, fn, tensors, reset):
        for k in self.tensors:
            self.tensors[k] = tensors[k].clone().float()
        return [{"applied": len(tensors)}]


def _trainer(tmp_path, store_dir, actor, *, rounds, required, base, buffer=None):
    from yeto.rl.adapters.verl.entry import verl_capabilities
    from yeto.rl.adapters.verl.fully_async_ports import FullyAsyncIsland, FullyAsyncTrainerGroup
    from yeto.rl.adapters.verl.ports_impl import VerlPolicyState
    from yeto.rl.core import CanonicalTensorSpec, canonical_layout_hash
    from yeto.rl.engine.bridges import LocalOnlySync
    from yeto.rl.engine.driver import EventTape, IslandDriver

    specs = (CanonicalTensorSpec(NAMES[0], (2, 4), "float32", 8), CanonicalTensorSpec(NAMES[1], (4, 2), "float32", 8))
    island = FullyAsyncIsland(actor, learner_id=0, limit=1, required=required, out_dir=str(tmp_path / "out"),
                              readback_dir=str(tmp_path / "rb"), base_model_revision="a" * 40,
                              lora_config_hash="b" * 64, layout_hash=canonical_layout_hash(specs),
                              expected_specs=specs)
    tape = EventTape(tmp_path / "tape.jsonl", 0)
    island.emit = lambda event, **f: tape.append({"event": event, **f})
    store = CheckpointStore(store_dir)
    if buffer is None:  # not ``or``: an empty FileSampleBuffer is falsy (len 0)
        buffer = FileSampleBuffer(store, batch_size=1, create_index=False)
    rollout, publisher, placement = install_buffer_ports(
        island, buffer=buffer,
        releases=PolicyReleaseStore(store), shape=bf.RowShape(64, 16, 0), base_dir=base,
        tokenize=lambda text: [7, 8], poll_s=0.01, wait_s=30)
    profile = far.execution_profile_for(TOLERANT, 1, groups_per_round=required, samples_per_group=2, sync="none")
    driver = IslandDriver(
        learner_id=0, rollout=rollout, trainer=FullyAsyncTrainerGroup(island), policy_state=VerlPolicyState(island),
        publisher=publisher, placement=placement,
        capabilities=verl_capabilities("sha256:" + "0" * 64, fully_async_limit=1),
        algorithm=TOLERANT, sync=LocalOnlySync(rounds), events=tape, profile=profile)
    return driver, island, publisher


def _lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_sampler_island_and_verl_buffer_trainer_run_three_rounds(tmp_path):
    store_dir = tmp_path / "store"
    store = CheckpointStore(store_dir)
    base, weights = _base(tmp_path, dtype=torch.float32)
    sampler_events, done = [], threading.Event()
    engine = FakeEngine(max_model_len=64)
    config = EngineSamplerConfig(sampler_id="island0", num_rollouts=40, max_backlog_groups=2,
                                 sampling=SamplingParams(max_tokens=6, n=2), max_prompt_tokens=32, poll_s=0.01)
    sampler = build_engine_sampler(
        config, engine=engine, releases=PolicyReleaseStore(store), buffer=FileSampleBuffer(store, batch_size=1),
        emit=lambda n, **f: sampler_events.append((n, f)),
        prompts=lambda r: [PromptItem(f"p{r}", f"prompt {r}", label="x")], reward=lambda item, c: float(len(c.token_ids) % 2),
        checks=PostLoadChecks(expected_expert_matrices=None, fixed_prompts=(), probe_pairs=(("Q", " 7"),)),
        should_stop=done.is_set,
        sleep=lambda s: time.sleep(0.005))
    thread = threading.Thread(target=sampler.run, daemon=True)
    thread.start()
    actor = FakeBufferActor()
    driver, island, publisher = _trainer(tmp_path, store_dir, actor, rounds=3, required=2, base=base)
    try:
        state = driver.run()
    finally:
        done.set()
        thread.join(timeout=10)
    assert driver.rounds_completed == 3 and state.policy_version == 3
    assert island.vmap.to_dict() == {"0": 0, "1": 1, "2": 2, "3": 3}
    # no group consumed twice; trained groups are those handed to the actor
    consumed = _lines(tmp_path / "out" / CONSUMED_LOG.format(learner=0))
    positions = [r["position"] for r in consumed]
    assert positions == sorted(set(positions)) == list(range(len(positions)))
    kept = [r for r in consumed if r["kept"]]
    assert len(kept) == 6 and [len(a) for a in actor.assembled] == [2, 2, 2]
    assert sorted(u for a in actor.assembled for u in a) == sorted(f"buf{r['position']}" for r in kept)
    # versions: v0 (the base) is not released; v1..v3 are, in order
    releases = _lines(tmp_path / "out" / RELEASE_LOG.format(learner=0))
    assert [(r["policy_version"], r["released"]) for r in releases] == [(0, False), (1, True), (2, True), (3, True)]
    assert [r["seq"] for r in releases if r["released"]] == [1, 2, 3]
    latest = PolicyReleaseStore(store).latest()
    assert (latest.version, latest.weights, latest.buffer_selected) == (3, "full", releases[-1]["buffer_selected"])
    merged = load_file(str(store_dir / "policy-releases" / latest.path / "model.safetensors"))
    a, b = actor.tensors[NAMES[0]], actor.tensors[NAMES[1]]
    assert torch.allclose(merged[f"{MODULE}.weight"], weights[f"{MODULE}.weight"] + b @ a, atol=1e-6)
    # the sampler loaded the releases in order, and the trained groups respect the age limit
    loaded = [f["policy_version"] for n, f in sampler_events if n == "rl_sampler_policy_loaded"]
    assert loaded == sorted(set(loaded)) and set(loaded) <= {1, 2, 3}
    events = _lines(tmp_path / "tape.jsonl")
    carry = [e for e in events if e["event"] == "rl_rollout_carry_over"]
    assert [e["rollout_id"] for e in carry] == [0, 1, 2]
    rounds = [e for e in events if e["event"] == "rl_verl_fa_round"]
    assert sum(e["kept"] for e in rounds) == 6
    assert sum(e["pulled"] for e in rounds) == len(consumed)


def test_publisher_releases_the_base_when_its_adapter_is_not_zero(tmp_path):
    from yeto.rl.adapters.verl.fully_async_ports import FullyAsyncIsland

    actor = FakeBufferActor()
    actor.tensors[NAMES[1]] += 1.0
    island = FullyAsyncIsland(actor, learner_id=0, limit=1, required=1, out_dir=str(tmp_path / "out"),
                              readback_dir=str(tmp_path / "rb"))
    store = CheckpointStore(tmp_path / "store")
    feed = bf.BufferFeed(FileSampleBuffer(store, batch_size=1), vmap=lambda: island.vmap, shape=SHAPE,
                         add_group=print, record=print)

    class State:
        policy_version = 0

        @staticmethod
        def policy_tensor_hash():
            return "0" * 64

        @staticmethod
        def to_lora():
            return type("L", (), {"tensors": actor.tensors})

    calls = []
    publisher = BufferReleasePublisher(island, PolicyReleaseStore(store), feed, base_dir=tmp_path,
                                       merge=lambda base, tensors, d: calls.append(d) or {"merged_modules": 1})
    result = publisher.publish(State())
    assert len(calls) == 1 and island.published == (0, "0" * 64)
    assert result.manifest.target_policy_version == 0 and island.vmap.to_dict() == {"0": 0}


# ---------------------------------------------------------------------- island entry
def _entry_args(*extra):
    from yeto.rl.adapters.verl.island_entry import parse_args

    return parse_args(["--model", "m", "--model-revision", "r", "--data", "d", "--learner-id", "0",
                       "--reward-function", "gsm8k_reward:score", "--global-rounds", "3",
                       "--groups-per-round", "2", "--samples-per-group", "2",
                       "--rollout-max-response-len", "64", "--event-tape", "t", "--inner-lr", "1e-5",
                       "--seq-len", "512", "--seed", "1", *extra])[0]


def test_island_entry_plans_a_buffer_trainer_and_refuses_other_roles():
    from yeto.rl.adapters.verl.island_entry import sample_buffer_section

    assert sample_buffer_section(_entry_args()) is None
    section = sample_buffer_section(_entry_args("--rl-sample-role", "trainer", "--rl-resume-store", "/s",
                                                "--rl-single-island-no-sync",
                                                "--rl-max-policy-age", "1", "--rl-sample-wait-s", "60"))
    assert section == {"store": "/s", "wait_s": 60.0, "lane": None}
    # S26 E9.4: a sampler island runs the engine sampler (no buffer section); other roles stop
    assert sample_buffer_section(_entry_args("--rl-sample-role", "sampler", "--rl-resume-store", "/s")) is None
    with pytest.raises(SystemExit, match="engine sampler"):
        sample_buffer_section(_entry_args("--rl-sample-role", "convertible", "--rl-resume-store", "/s"))
    with pytest.raises(SystemExit, match="max-policy-age"):
        sample_buffer_section(_entry_args("--rl-sample-role", "trainer", "--rl-resume-store", "/s"))


def test_island_entry_plans_a_synced_trainer_lane_and_refuses_partial_flags():
    from yeto.rl.adapters.verl.island_entry import sample_buffer_section

    synced = ("--rl-sample-role", "trainer", "--syncer", "h:1", "--rl-max-policy-age", "2")
    section = sample_buffer_section(_entry_args(*synced, "--rl-sample-store", "/s", "--rl-sample-lanes", "2"))
    assert section == {"store": "/s", "wait_s": None, "lane": "0/2"}
    for extra, message in ((("--rl-sample-store", "/s"), "needs --rl-sample-store and --rl-sample-lanes"),
                           (("--rl-sample-store", "/s", "--rl-sample-lanes", "2", "--rl-resume-store", "/s"),
                            "not --rl-resume-store")):
        with pytest.raises(SystemExit, match=message):
            sample_buffer_section(_entry_args(*synced, *extra))
    with pytest.raises(SystemExit, match="need --rl-sample-role trainer with a syncer"):
        sample_buffer_section(_entry_args("--rl-sample-role", "trainer", "--rl-single-island-no-sync",
                                          "--rl-max-policy-age", "1", "--rl-resume-store", "/s",
                                          "--rl-sample-lanes", "2"))


# ---------------------------------------------------------------------- synced trainers (S25 C3)
class _State:
    """A published policy state: local version, hash, the LoRA tensors."""

    def __init__(self, version, tensors, label):
        import hashlib

        self.policy_version, self._tensors = version, tensors
        self.label, self._digest = label, hashlib.sha256(label.encode()).hexdigest()

    def policy_tensor_hash(self):
        return self._digest

    def to_lora(self):
        return type("L", (), {"tensors": self._tensors})


def _synced_publisher(tmp_path, store, lane, events):
    from yeto.rl.adapters.verl.fully_async_ports import FullyAsyncIsland
    from yeto.rl.engine.merged_release import BufferLane, lane_cursor_log

    actor = FakeBufferActor()
    island = FullyAsyncIsland(actor, learner_id=lane, limit=2, required=1,
                              out_dir=str(tmp_path / f"out{lane}"), readback_dir=str(tmp_path / "rb"))
    (tmp_path / f"out{lane}").mkdir(exist_ok=True)
    island.emit = lambda event, **f: events.append((event, f))
    buffer = FileSampleBuffer(store, batch_size=1, lane=BufferLane(lane, 2))
    _, publisher, _ = install_buffer_ports(
        island, buffer=buffer, releases=PolicyReleaseStore(store), shape=SHAPE, base_dir=tmp_path,
        cursor_log=lane_cursor_log(buffer))
    def merge(base, tensors, directory):
        (directory / "model.safetensors").write_bytes(b"w")
        return {}

    publisher.merge = merge
    driver = type("D", (), {"outer_version": None})()
    publisher.bind(driver)
    return actor, island, publisher, driver


def test_only_the_publisher_releases_each_new_merged_version_once(tmp_path):
    store = CheckpointStore(tmp_path / "store")
    nonzero = {NAMES[0]: torch.ones(2, 4), NAMES[1]: torch.ones(4, 2)}
    zero = {NAMES[0]: torch.ones(2, 4), NAMES[1]: torch.zeros(4, 2)}
    events = {0: [], 1: []}
    islands = {m: _synced_publisher(tmp_path, store, m, events[m]) for m in (0, 1)}
    # (local version, outer version of the applied base): base, a local round without a
    # merge, merges 1 and 2, then merge 4 (merge 3 missed: a gap)
    steps = [(0, 0, zero), (1, 0, nonzero), (2, 1, nonzero), (3, 2, nonzero), (4, 4, nonzero)]
    for local, outer, tensors in steps:
        for m, (actor, island, publisher, driver) in islands.items():
            actor.param_version = local
            driver.outer_version = outer
            publisher.publish(_State(local, tensors, f"h{outer}-{local}"))
    released = [f for e, f in events[0] if e == "rl_policy_released"]
    assert [(f["policy_version"], f["local_version"]) for f in released] == [(1, 2), (2, 3), (4, 4)]
    assert all(f["origin"] == {"kind": "merged", "learner_id": 0, "num_learners": 2,
                               "local_version": f["local_version"]} for f in released)
    assert [f for e, f in events[0] if e == "rl_policy_release_gap"] == [{"previous_version": 2, "policy_version": 4}]
    assert not [e for e, _ in events[1] if e == "rl_policy_released"]
    assert ("rl_policy_release_role", {"learner_id": 1, "num_learners": 2, "publisher": False}) in events[1]
    latest = PolicyReleaseStore(store).latest()
    assert (latest.seq, latest.version, latest.policy_hash) == (3, 4, _State(4, {}, "h4-4").policy_tensor_hash())
    # both islands pair each merged version with the first local version that applied it
    for _, island, publisher, _ in islands.values():
        assert publisher.feed.merged.to_dict() == {"0": 0, "1": 2, "2": 3, "4": 4}


def test_a_synced_feed_reads_its_lane_translates_merged_versions_and_logs_each_round(tmp_path):
    store = CheckpointStore(tmp_path / "store")
    writer = GroupWriter(FileSampleBuffer(store, batch_size=1), "s0")
    for i, version in enumerate((0, 0, 1, 1, 3, 3)):  # merged behavior versions
        writer.write([_sample(version, index=0)], group_key=f"g{i}")
    events = []
    actor, island, publisher, driver = _synced_publisher(tmp_path, store, 1, events)
    for local, outer in ((0, 0), (1, 1), (2, 1)):  # merge 1 applied at local 1; local 2 trains on
        actor.param_version = local
        driver.outer_version = outer
        publisher.publish(_State(local, {NAMES[1]: torch.zeros(4, 2)}, f"h{local}"))
    feed = publisher.feed
    pulled = []
    for _ in range(3):
        meta = feed.pull()
        pulled.append((meta["uid"], meta["min_steps"]))
        feed.keep() if None not in meta["min_steps"] else feed.drop()
    # lane 1 of 2: positions 1, 3, 5; merged v0 -> local 0, v1 -> local 1, v3 never applied here
    assert pulled == [("buf1", (0,)), ("buf3", (1,)), ("buf5", (None,))]
    island.call("yeto_assemble")  # the round's collection ended: one lane cursor line
    lane_file = tmp_path / "store" / "sample-buffer" / "lanes" / "1-of-2.jsonl"
    assert _lines(lane_file) == [{"rollout_id": 2, "positions": [1, 3], "cursor": 6}]
    # a restarted island continues after the recorded cursor
    again = _synced_publisher(tmp_path, store, 1, [])[2].feed
    assert again.buffer.read_cursor == 6


def test_two_shard_samplers_drain_their_backlog_through_the_verl_release(tmp_path):
    """S26 E9 G1 (e9-g1-f) root cause: the verl trainer released without the per-shard
    read positions, so a shard sampler never saw its backlog drain (stuck at the cap).
    Two shard samplers with a 2-group cap now feed three rounds of 2 groups."""
    from yeto.rl.engine.engine_sampler_entry import build_stores
    from yeto.rl.engine.sample_buffer import open_trainer_buffer

    store_dir = tmp_path / "store"
    store = CheckpointStore(store_dir)
    base, _weights = _base(tmp_path, dtype=torch.float32)
    done, threads, events = threading.Event(), [], []
    for sid in ("island0", "island1"):
        releases, buffer = build_stores({"store": str(store_dir), "sampler_id": sid,
                                         "buffer_shards": ["island0", "island1"]})
        config = EngineSamplerConfig(sampler_id=sid, num_rollouts=40, max_backlog_groups=2,
                                     sampling=SamplingParams(max_tokens=6, n=2), max_prompt_tokens=32, poll_s=0.01)
        sampler = build_engine_sampler(
            config, engine=FakeEngine(max_model_len=64), releases=releases, buffer=buffer,
            emit=lambda n, _sid=sid, **f: events.append((_sid, n, f)),
            prompts=lambda r, _sid=sid: [PromptItem(f"{_sid}-p{r}", f"prompt {r}", label="x")],
            reward=lambda item, c: float(len(c.token_ids) % 2),
            checks=PostLoadChecks(expected_expert_matrices=None, fixed_prompts=(), probe_pairs=(("Q", " 7"),)),
            should_stop=done.is_set, sleep=lambda s: time.sleep(0.005))
        def run(_sampler=sampler, _sid=sid):
            try:
                _sampler.run()
            except BaseException as error:  # noqa: BLE001 - surfaced by the assert below
                events.append((_sid, "crashed", {"error": repr(error)}))
                raise

        threads.append(threading.Thread(target=run, daemon=True))
        threads[-1].start()
    trainer_buffer = open_trainer_buffer(store, batch_size=1, create_index=False, shards=["island0", "island1"])
    driver, _island, _publisher = _trainer(tmp_path, store_dir, FakeBufferActor(), rounds=3, required=2,
                                           base=base, buffer=trainer_buffer)
    try:
        driver.run()
    except Exception:
        listing = sorted(str(p.relative_to(store_dir)) for p in (store_dir / "sample-buffer").rglob("*.jsonl"))
        raise AssertionError(f"buffer {len(trainer_buffer)} shards {trainer_buffer.index.shard_names()} "
                             f"files {listing}; "
                             f"sampler events: {[(sid, n) for sid, n, _f in events][-20:]}; "
                             f"crashes: {[f for _sid, n, f in events if n == 'crashed']}")
    finally:
        done.set()
        for t in threads:
            t.join(timeout=10)
    assert driver.rounds_completed == 3
    releases = _lines(tmp_path / "out" / RELEASE_LOG.format(learner=0))
    shards = [r["buffer_shards"] for r in releases if r["released"]]
    assert all(set(s) <= {"island0", "island1"} and sum(s.values()) == r["buffer_selected"]
               for s, r in zip(shards, [r for r in releases if r["released"]]))
    assert PolicyReleaseStore(store).latest().buffer_shards == shards[-1]
