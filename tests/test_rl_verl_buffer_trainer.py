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


def _trainer(tmp_path, store_dir, actor, *, rounds, required, base):
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
    rollout, publisher, placement = install_buffer_ports(
        island, buffer=FileSampleBuffer(store, batch_size=1, create_index=False),
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
                                                "--rl-max-policy-age", "1", "--rl-sample-wait-s", "60"))
    assert section == {"store": "/s", "wait_s": 60.0}
    with pytest.raises(SystemExit, match="engine sampler"):
        sample_buffer_section(_entry_args("--rl-sample-role", "sampler", "--rl-resume-store", "/s"))
    with pytest.raises(SystemExit, match="max-policy-age"):
        sample_buffer_section(_entry_args("--rl-sample-role", "trainer", "--rl-resume-store", "/s"))
