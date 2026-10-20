"""S26 E9.4: one launch starts a verl trainer island and its sampler islands (dry run).

``yeto launch --rl-backend verl --rl-island-sample-role ...`` builds every island
with the verl island entry; a sampler island runs the engine sampler on the verl
image's vLLM (:mod:`yeto.rl.adapters.verl.sampler_entry`). No cloud, no GPU."""

from __future__ import annotations

import shlex
import sys
import types

import pytest

from tests.test_rl_engine_selection import _cli
from yeto.rl.adapters.verl import island_entry, sampler_entry
from yeto.rl.engine import sample_roles

STORE = "modal-volume://yeto-e9/run"


def _tasks(monkeypatch, extra, count):
    from tests.test_rl_launcher import _Resources, _Storage, _StorageMode, _Task
    from yeto import launcher
    from yeto.gpu_spec import parse_gpu_spec
    from yeto.launch.rl_island_task import make_miles_island_task

    monkeypatch.setitem(sys.modules, "sky", types.SimpleNamespace(
        Task=_Task, Resources=_Resources, Storage=_Storage, StorageMode=_StorageMode))
    args = _cli(tuple(extra))
    launcher._prepare_rl_args(args)
    args.model_revision, args.data_revision = "a" * 40, "b" * 40
    args.source_sha256, args.reward_sha256 = "c" * 64, "d" * 64
    specs = parse_gpu_spec(args.gpu)
    out = []
    for m in range(count):
        task = launcher._island_task(make_miles_island_task, args, specs[m], m, count, "127.0.0.1:1", {})
        line = next(l for l in task.run.splitlines() if "-m yeto.rl.adapters.verl.island_entry" in l)
        argv = shlex.split(line.split("-m yeto.rl.adapters.verl.island_entry", 1)[1]
                           .replace("$LEARNER_ID", str(m)).replace("$SYNCER_ADDR", "127.0.0.1:1"))
        out.append(island_entry.parse_args(argv)[0])
    return args, out


def test_synced_verl_trainers_and_a_separate_sampler_launch(monkeypatch):
    """Multi-island sync (S25 C3V shape): N synced trainers read lanes of one store and
    learner 0 releases; a sampler launch over the same store runs the engine sampler.
    A relaunched publisher (learner 0) gets the same flags (the fleet controller re-runs
    the same task), so it re-joins as the publisher."""
    from yeto.launch.modal_islands import effective_recover_timeout
    from yeto.rl.engine.merged_release import PUBLISHER_LEARNER_ID

    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k" * 64)  # test key; elastic islands sign messages
    common = ["--rl-backend", "verl", "--rl-checkpoint-store", STORE, "--rl-max-policy-age", "2",
              "--reward-function", "gsm8k_reward:score", "--total-steps", "4"]
    targs, trainers = _tasks(monkeypatch, common + ["--gpu", "modal:1xa10g,modal:1xa10g",
                                                    "--rl-sample-role", "trainer",
                                                    "--rl-island-scheduling", "elastic"], 2)
    for m, t in enumerate(trainers):
        assert t.rl_sample_role == "trainer" and t.rl_sample_lanes == 2 and t.rl_sample_store
        assert t.rl_resume_store is None and t.learner_id == m
        assert island_entry.sample_buffer_section(t)["lane"] == f"{m}/2"
    assert PUBLISHER_LEARNER_ID == 0
    assert effective_recover_timeout(targs) > 0  # relaunch allowed: the publisher can be replaced
    _sargs, (sampler,) = _tasks(monkeypatch, common + ["--gpu", "modal:1xa10g", "--rl-sample-role", "sampler",
                                                       "--rl-single-island-no-sync"], 1)
    assert sampler.rl_sample_role == "sampler" and sampler.rl_resume_store == trainers[0].rl_sample_store
    spec = sampler_entry.sampler_spec(sampler, model_path="/m", prompts_path="/p", card_model="A10G",
                                      engine_version="v", event_tape="/t")
    assert spec["sampler_id"] == "island0" and spec["store"] == trainers[0].rl_sample_store


def _islands(monkeypatch, roles):
    from tests.test_rl_launcher import _Resources, _Storage, _StorageMode, _Task
    from yeto import launcher
    from yeto.gpu_spec import parse_gpu_spec
    from yeto.launch.rl_island_task import make_miles_island_task

    monkeypatch.setitem(sys.modules, "sky", types.SimpleNamespace(
        Task=_Task, Resources=_Resources, Storage=_Storage, StorageMode=_StorageMode))
    extra = ["--rl-backend", "verl", "--rl-single-island-no-sync", "--rl-checkpoint-store", STORE,
             "--rl-max-policy-age", "2", "--gpu", ",".join(["modal:1xa10g"] * len(roles)),
             "--reward-function", "gsm8k_reward:score", "--total-steps", "4"]
    for m, role in enumerate(roles):
        extra += ["--rl-island-sample-role", f"{m}:{role}"]
    args = _cli(tuple(extra))
    launcher._prepare_rl_args(args)
    args.model_revision, args.data_revision = "a" * 40, "b" * 40
    args.source_sha256, args.reward_sha256 = "c" * 64, "d" * 64
    specs = parse_gpu_spec(args.gpu)
    out = []
    for m in range(len(roles)):
        task = launcher._island_task(make_miles_island_task, args, specs[m], m, len(roles), "none", {})
        line = next(l for l in task.run.splitlines() if "-m yeto.rl.adapters.verl.island_entry" in l)
        argv = shlex.split(line.split("-m yeto.rl.adapters.verl.island_entry", 1)[1].replace("$LEARNER_ID", "0"))
        parsed, _ignored = island_entry.parse_args(argv)
        out.append((parsed, task.envs))
    return out


def test_one_launch_gives_one_sampler_and_one_trainer(monkeypatch):
    (sampler, s_env), (trainer, _t_env) = _islands(monkeypatch, ["sampler", "trainer"])
    assert (sampler.rl_sample_role, trainer.rl_sample_role) == ("sampler", "trainer")
    assert sampler.rl_sampler_node == "island0" and sampler.rl_resume_store == trainer.rl_resume_store
    assert island_entry.sample_buffer_section(sampler) is None  # main() runs the engine sampler
    section = island_entry.sample_buffer_section(trainer)
    assert section["store"] == trainer.rl_resume_store and "shards" not in section
    assert s_env.get("YETO_SPOT_RECLAIM_MARKER")  # the reclaim signal reaches the sampler
    spec = sampler_entry.sampler_spec(sampler, model_path="/m", prompts_path="/p.jsonl", card_model="A10G",
                                      engine_version="0.x", event_tape="/t/events.jsonl", python="python3")
    assert spec["sampler_id"] == "island0" and spec["store"] == sampler.rl_resume_store
    assert spec["island_id"] == 0  # the launcher reads the sampler tape as island tape
    assert spec["engine"]["adapter"] == "vllm-server" and spec["num_rollouts"] == 4
    assert spec["max_backlog_groups"] == 4 * sample_roles.DEFAULT_MAX_BACKLOG_ROUNDS
    assert spec["reward"] == "yeto.rl.gsm8k_reward:score" and "buffer_shards" not in spec
    from yeto.rl.engine.engine_sampler_entry import island_config

    assert island_config(spec).sampler_id == "island0"  # the entry accepts the spec
    from yeto.rl.adapters.sampler_engines import from_spec

    assert spec["engine"]["factory"] == from_spec.FACTORY


def test_two_samplers_get_shards_and_disjoint_data_slots(monkeypatch):
    (s0, _), (s1, _), (trainer, _) = _islands(monkeypatch, ["sampler", "sampler", "trainer"])
    assert (s0.rl_buffer_shards, s1.rl_buffer_shards) == (["island0", "island1"],) * 2
    assert (s0.rl_sampler_data_slot, s1.rl_sampler_data_slot) == ("0/2", "1/2")
    assert (s0.global_rounds, s1.global_rounds, trainer.global_rounds) == (2, 2, 4)  # the samplers share the run
    spec = sampler_entry.sampler_spec(s1, model_path="/m", prompts_path="/p.jsonl", card_model="A10G",
                                      engine_version="0.x", event_tape="/t/e.jsonl")
    assert spec["buffer_shards"] == ["island0", "island1"] and spec["prompts"]["data_slot"] == "1/2"
    assert island_entry.sample_buffer_section(trainer)["shards"] == ["island0", "island1"]


def test_sampler_spec_refuses_missing_flags_and_unmapped_rewards():
    base = dict(rl_sampler_node="island0", rl_resume_store="/s", learner_id=3, groups_per_round=2, rollout_max_response_len=8,
                samples_per_group=2, seed=1, global_rounds=2, reward_function="gsm8k_reward:score")
    kw = dict(model_path="/m", prompts_path="/p", card_model="c", engine_version="v", event_tape="/t")
    with pytest.raises(sampler_entry.SamplerIslandError, match="resume-store"):
        sampler_entry.sampler_spec(types.SimpleNamespace(**{**base, "rl_resume_store": None}), **kw)
    # a separate sampler launch (no pair): the Miles sampler's id
    assert sampler_entry.sampler_spec(types.SimpleNamespace(**{**base, "rl_sampler_node": None}),
                                      **kw)["sampler_id"] == "island3"
    with pytest.raises(sampler_entry.SamplerIslandError, match="no engine sampler reward"):
        sampler_entry.sampler_spec(types.SimpleNamespace(**{**base, "reward_function": "x:y"}), **kw)


def test_prompt_rows_render_messages():
    rows = [{"messages": [{"role": "user", "content": "1+1?", "extra": 0}], "label": 2}]
    out = sampler_entry.prompt_rows(rows, lambda m: "|".join(x["content"] for x in m))
    assert out == [{"text": "1+1?", "label": "2"}]
    with pytest.raises(sampler_entry.SamplerIslandError):
        sampler_entry.prompt_rows([], str)


def test_engine_sampler_writes_its_named_shard(tmp_path):
    from yeto.rl.engine.engine_sampler_entry import SamplerSpecError, build_stores

    _releases, buffer = build_stores({"store": str(tmp_path), "sampler_id": "island1",
                                      "buffer_shards": ["island0", "island1"]})
    assert buffer.shard == "island1"
    with pytest.raises(SamplerSpecError, match="not one of the buffer shards"):
        build_stores({"store": str(tmp_path), "sampler_id": "island9", "buffer_shards": ["island0"]})


def test_trainer_buffer_merges_named_shards(tmp_path):
    from yeto.rl.engine.resume import store_for
    from yeto.rl.engine.sample_buffer import open_trainer_buffer

    buffer = open_trainer_buffer(store_for(str(tmp_path)), batch_size=1, shards=["island0", "island1"])
    assert buffer._merge and set(buffer._expected_shards) == {"island0", "island1"}
    plain = open_trainer_buffer(store_for(str(tmp_path / "one")), batch_size=1)
    assert not plain._merge and plain._expected_shards is None


def _group(monkeypatch, gpus, samplers, total_steps=4):
    """One launch of synced trainers plus sampler islands (S26 E9.4 synced group)."""
    from tests.test_rl_launcher import _Resources, _Storage, _StorageMode, _Task
    from yeto import launcher
    from yeto.gpu_spec import parse_gpu_spec
    from yeto.launch.rl_island_task import make_miles_island_task

    monkeypatch.setenv("YETO_ISLAND_HMAC_KEY", "k" * 64)  # test key; elastic islands sign messages
    monkeypatch.setitem(sys.modules, "sky", types.SimpleNamespace(
        Task=_Task, Resources=_Resources, Storage=_Storage, StorageMode=_StorageMode))
    extra = ["--rl-backend", "verl", "--rl-checkpoint-store", STORE, "--rl-max-policy-age", "2",
             "--reward-function", "gsm8k_reward:score", "--total-steps", str(total_steps),
             "--gpu", ",".join(["modal:1xa10g"] * gpus), "--rl-sample-role", "trainer",
             "--rl-island-scheduling", "elastic"]
    for m in samplers:
        extra += ["--rl-island-sample-role", f"{m}:sampler"]
    args = _cli(tuple(extra))
    launcher._prepare_rl_args(args)
    args.model_revision, args.data_revision = "a" * 40, "b" * 40
    args.source_sha256, args.reward_sha256 = "c" * 64, "d" * 64
    specs = parse_gpu_spec(args.gpu)
    learners = sample_roles.syncer_learner_count(args, len(specs))
    out = []
    for m in range(len(specs)):
        task = launcher._island_task(make_miles_island_task, args, specs[m], m, learners, "127.0.0.1:1", {})
        line = next(l for l in task.run.splitlines() if "-m yeto.rl.adapters.verl.island_entry" in l)
        argv = shlex.split(line.split("-m yeto.rl.adapters.verl.island_entry", 1)[1]
                           .replace("$LEARNER_ID", task.envs["LEARNER_ID"]).replace("$SYNCER_ADDR", "127.0.0.1:1"))
        out.append((island_entry.parse_args(argv)[0], task, line))
    return args, learners, out


def test_one_launch_starts_synced_trainers_and_their_sampler(monkeypatch):
    """Known limit fixed: synced trainers and a sampler start from one command."""
    args, learners, ((t0, _, l0), (t1, _, l1), (s, s_task, s_line)) = _group(monkeypatch, 3, [2])
    assert sample_roles.is_synced_group(args) and not sample_roles.is_pair(args)
    assert learners == 2  # the syncer waits for the two trainers only
    for k, t in enumerate((t0, t1)):
        assert t.rl_sample_role == "trainer" and t.learner_id == k and t.rl_sample_lanes == 2
        assert t.rl_sample_store and t.rl_resume_store is None
        assert island_entry.sample_buffer_section(t)["lane"] == f"{k}/2"
        assert "shards" not in island_entry.sample_buffer_section(t)  # one sampler writes the index
    assert "--rl-island-scheduling elastic" in l0 and "--syncer" in l1
    assert s.rl_sample_role == "sampler" and s.rl_sampler_node == "island2"
    assert s.rl_resume_store == t0.rl_sample_store  # the same store
    assert "--rl-single-island-no-sync" in s_line and "--rl-island-scheduling" not in s_line
    assert s_task.envs["LEARNER_ID"] == "0" and s_task.envs.get("YETO_SPOT_RECLAIM_MARKER")
    spec = sampler_entry.sampler_spec(s, model_path="/m", prompts_path="/p", card_model="A10G",
                                      engine_version="v", event_tape="/t")
    assert spec["num_rollouts"] == 8  # the rounds of both trainers (2 x 4)


def test_synced_group_with_two_samplers_shards_and_slots(monkeypatch):
    _args, learners, islands = _group(monkeypatch, 4, [0, 3])
    (s0, _, _), (t0, _, _), (t1, _, _), (s1, _, _) = islands
    assert learners == 2 and (t0.learner_id, t1.learner_id) == (0, 1)
    assert island_entry.sample_buffer_section(t0)["shards"] == ["island0", "island3"]
    assert (s0.rl_sampler_data_slot, s1.rl_sampler_data_slot) == ("0/2", "1/2")
    for s in (s0, s1):
        spec = sampler_entry.sampler_spec(s, model_path="/m", prompts_path="/p", card_model="A10G",
                                          engine_version="v", event_tape="/t")
        assert spec["num_rollouts"] == 4  # 2 trainers x 4 rounds over 2 samplers


@pytest.mark.parametrize("gpus,samplers,steps,match", [
    (2, [1], 4, "at least two trainer"),
    (3, [2], 3, None),  # 2 x 3 rounds over one sampler: fine
    (4, [2, 3], 3, None),  # 2 x 3 = 6 over 2 samplers: fine
])
def test_synced_group_launch_checks(monkeypatch, gpus, samplers, steps, match):
    if match is None:
        _group(monkeypatch, gpus, samplers, total_steps=steps)
        return
    with pytest.raises((ValueError, SystemExit), match=match):
        _group(monkeypatch, gpus, samplers, total_steps=steps)


def test_synced_group_lists_samplers_only():
    args = types.SimpleNamespace(rl_island_sample_role=["0:trainer", "2:sampler"], rl_sample_role="trainer",
                                 rl_single_island_no_sync=False, tuning="lora", total_steps=4,
                                 rl_node_class=None)
    with pytest.raises(ValueError, match="sampler islands only"):
        sample_roles.check_synced_group(args, 3)


def test_verl_trainer_buffer_merges_shards_without_a_sampler_index(tmp_path):
    """G1 (e9-g1-d) root cause: the verl buffer trainer opens its buffer with
    create_index=False; with named shards no sampler writes index.jsonl, so the
    merging trainer must create it (it is the index's only writer)."""
    from tests.test_rl_spot_sampler_replace import _cursor, _round
    from yeto.rl.adapters.miles import buffer_rollout as br
    from yeto.rl.engine.resume import CheckpointStore
    from yeto.rl.engine.sample_buffer import FileSampleBuffer, open_trainer_buffer

    writer = br.SamplerRoundWriter(FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1,
                                                    index="shards/island0.jsonl"), "island0", incarnation="a")
    writer.write_round(0, _round(0, 3), _cursor(3))
    assert not (tmp_path / "sample-buffer/index.jsonl").exists()
    trainer = open_trainer_buffer(CheckpointStore(tmp_path), batch_size=1, create_index=False,
                                  shards=["island0", "island1"])
    assert len(trainer.select("t", 3)) == 3
    assert (tmp_path / "sample-buffer/index.jsonl").exists()
    # a plain (single-sampler) reader still never creates the index
    reader = open_trainer_buffer(CheckpointStore(tmp_path / "plain"), batch_size=1, create_index=False)
    assert not reader._merge and not (tmp_path / "plain/sample-buffer/index.jsonl").exists()
