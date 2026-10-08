"""S14-M1: per-record TITO session mismatches reach the tape as ``rl_harness_mismatch``
(observe only). No Ray: the hook is exercised on fake samples and the driver on FakeEngine."""
from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import pytest
import torch

from yeto.rl.engine import timeline as tl
from yeto.rl.engine.algorithm import AlgorithmSpec
from yeto.rl.engine.bridges import LocalOnlySync
from yeto.rl.engine.driver import EventTape, IslandDriver
from yeto.rl.engine.execution_profile import ExecutionProfile
from yeto.rl.engine.fake import FakeEngine, fake_capabilities
from yeto.rl.engine.miles_adapter import rollout_meta_hook as hook
from yeto.rl.engine.miles_adapter.rollout import handle_from_metadata

KEY = hook.TITO_SESSION_MISMATCH_KEY
RKEY = hook.TITO_SESSION_MISMATCH_RECORDS_KEY


def _sample(i, group, mism=None, **meta):
    md = dict(meta)
    if mism is not None:
        md[KEY] = mism
    return SimpleNamespace(index=i, group_index=group, rollout_id=0, metadata=md, reward=1.0,
                           status=SimpleNamespace(value="completed"), response_length=4,
                           weight_versions=None, remove_sample=False)


def _rec(kind="assistant_text", seg=3, exp="a b", act="a  b", detail=""):
    return {"type": kind, "segment_index": seg, "expected_text": exp, "actual_text": act, "detail": detail}


def test_hook_records_flatten_truncate_and_cap():
    long = "x" * 2000
    groups = [
        [_sample(0, 0, [_rec(), _rec("special_token_count", -1, long, long, detail="structure")]),
         _sample(1, 0, 3)],                       # int counter: no records
        [[_sample(10, 1, [_rec("non_assistant_text", 1)])]],
        [_sample(20, 2, None), SimpleNamespace(metadata=None)],
    ]
    recs = hook.harness_mismatch_records(groups, limit=64, text_limit=512)
    assert [(r["sample_index"], r["group_index"], r["record_index"], r["kind"], r["segment_index"]) for r in recs] == [
        (0, 0, 0, "assistant_text", 3), (0, 0, 1, "special_token_count", -1), (10, 1, 0, "non_assistant_text", 1)]
    assert recs[0] == {"sample_index": 0, "group_index": 0, "record_index": 0, "kind": "assistant_text",
                       "segment_index": 3, "expected_text": "a b", "actual_text": "a  b", "detail": "",
                       "truncated": False}
    assert recs[1]["expected_text"] == "x" * 512 and recs[1]["actual_text"] == "x" * 512
    assert recs[1]["truncated"] is True and recs[1]["detail"] == "structure"
    for r in recs:
        assert tl.validate_harness_mismatch({**r, "rollout_id": 0, "policy_version": 0}) == []
    # cap: stops at ``limit`` records across samples; 0 disables
    assert len(hook.harness_mismatch_records(groups, limit=2, text_limit=512)) == 2
    assert hook.harness_mismatch_records(groups, limit=0, text_limit=512) == []
    # an upstream Enum-valued ``type`` and a non-dict entry are tolerated
    odd = [[_sample(5, 5, [{"type": SimpleNamespace(value="special_token_type")}, "raw"])]]
    got = hook.harness_mismatch_records(odd, limit=64, text_limit=8)
    assert [r["kind"] for r in got] == ["special_token_type", ""] and got[1]["detail"] == "'raw'"


def test_hook_limits_from_args_or_env(monkeypatch):
    monkeypatch.delenv(hook.MISMATCH_RECORDS_MAX_ENV, raising=False)
    monkeypatch.delenv(hook.MISMATCH_TEXT_MAX_ENV, raising=False)
    assert hook.mismatch_records_limit(SimpleNamespace()) == tl.HARNESS_MISMATCH_MAX_PER_ROUND == 64
    assert hook.mismatch_text_limit(SimpleNamespace()) == tl.HARNESS_MISMATCH_TEXT_MAX == 512
    assert hook.mismatch_records_limit(SimpleNamespace(yeto_rl_mismatch_tape_max=5)) == 5
    monkeypatch.setenv(hook.MISMATCH_RECORDS_MAX_ENV, "7")
    monkeypatch.setenv(hook.MISMATCH_TEXT_MAX_ENV, "bad")
    assert hook.mismatch_records_limit(SimpleNamespace()) == 7
    assert hook.mismatch_text_limit(SimpleNamespace()) == 512


@pytest.mark.parametrize("observe", [False, True])
def test_build_metadata_carries_records_only_when_observing(observe):
    args = SimpleNamespace(yeto_rl_observe_timeline=observe, yeto_rl_mismatch_tape_max=2)
    data = [[_sample(0, 0, [_rec(), _rec(seg=4), _rec(seg=5)]), _sample(1, 0, [])],
            [_sample(10, 1, None)]]
    hook.record_trained_groups(args, data)
    meta = hook.build_metadata(args, data)
    assert meta[KEY] == 3  # the count is unchanged either way
    if not observe:
        assert RKEY not in meta
        return
    assert [r["segment_index"] for r in meta[RKEY]] == [3, 4]  # capped at 2
    json.dumps(meta)  # sink-serialisable
    handle = handle_from_metadata(meta, rollout_id=0, policy_version=0, policy_hash="h", data_pack=None)
    assert handle.tito_session_mismatch_records == tuple(meta[RKEY])


def test_build_metadata_without_mismatches_adds_no_key():
    args = SimpleNamespace(yeto_rl_observe_timeline=True)
    data = [[_sample(0, 0, []), _sample(1, 0, None)]]
    hook.record_trained_groups(args, data)
    meta = hook.build_metadata(args, data)
    assert KEY not in meta and RKEY not in meta
    assert handle_from_metadata(meta, rollout_id=0, policy_version=0, policy_hash="h",
                                data_pack=None).tito_session_mismatch_records is None


def _driver(engine, tmp_path, observe):
    profile = ExecutionProfile(name="p", execution_mode="partitioned-serial",
                               outer_protocol="none").bind_algorithm(AlgorithmSpec())
    return IslandDriver(learner_id=0, rollout=engine.rollout, trainer=engine.trainer,
                        policy_state=engine.policy_state, publisher=engine.publisher,
                        placement=engine.placement,
                        capabilities=fake_capabilities(execution_modes={"partitioned-serial"}),
                        algorithm=AlgorithmSpec(), sync=LocalOnlySync(2),
                        events=EventTape(tmp_path / "e.jsonl", 0), profile=profile, observe=observe)


@pytest.mark.parametrize("observe", [False, True])
def test_driver_tapes_one_event_per_record_only_when_observing(tmp_path, observe):
    engine = FakeEngine(tensors={"base_model.model.layer.lora_A.weight": torch.zeros(1, 2)},
                        step_delta=1.0, placement_kind="fixed-partition")
    recs = hook.harness_mismatch_records(
        [[_sample(7, 2, [_rec(), _rec("special_token_count", -1, "y" * 600, "z")])]], limit=64, text_limit=512)
    per_round = {0: tuple(recs), 1: None}
    original = engine.rollout.generate
    engine.rollout.generate = lambda r, **kw: dataclasses.replace(
        original(r, **kw), tito_session_mismatch_records=per_round[r])
    _driver(engine, tmp_path, observe).run()
    events = [json.loads(l) for l in (tmp_path / "e.jsonl").read_text().splitlines()]
    got = [e for e in events if e["event"] == tl.HARNESS_MISMATCH_EVENT]
    if not observe:
        assert got == []  # default path: byte-for-byte the old tape
        assert not any("expected_text" in e for e in events)
        return
    assert [(e["rollout_id"], e["sample_index"], e["record_index"], e["kind"], e["segment_index"]) for e in got] == [
        (0, 7, 0, "assistant_text", 3), (0, 7, 1, "special_token_count", -1)]
    assert got[1]["expected_text"] == "y" * 512 and got[1]["truncated"] is True
    for e in got:
        assert e["policy_version"] == 0 and e["profile_hash"] and "config_epoch" in e and "t" in e
        assert tl.validate_harness_mismatch(e) == []
    # the driver re-applies the per-round cap for engines that do not
    assert len(got) == 2
    # ordering: the mismatch events sit after the round's rl_round_labels
    names = [e["event"] for e in events]
    assert names.index("rl_round_labels") < names.index(tl.HARNESS_MISMATCH_EVENT)


def test_driver_caps_records_per_round(tmp_path):
    engine = FakeEngine(tensors={"base_model.model.layer.lora_A.weight": torch.zeros(1, 2)},
                        step_delta=1.0, placement_kind="fixed-partition")
    many = tuple({"sample_index": i, "group_index": 0, "record_index": 0, "kind": "assistant_text",
                  "segment_index": 1, "expected_text": "", "actual_text": "", "detail": "", "truncated": False}
                 for i in range(100))
    original = engine.rollout.generate
    engine.rollout.generate = lambda r, **kw: dataclasses.replace(original(r, **kw), tito_session_mismatch_records=many)
    _driver(engine, tmp_path, True).run()
    events = [json.loads(l) for l in (tmp_path / "e.jsonl").read_text().splitlines()]
    per_round = {}
    for e in events:
        if e["event"] == tl.HARNESS_MISMATCH_EVENT:
            per_round[e["rollout_id"]] = per_round.get(e["rollout_id"], 0) + 1
    assert per_round == {0: tl.HARNESS_MISMATCH_MAX_PER_ROUND, 1: tl.HARNESS_MISMATCH_MAX_PER_ROUND}


def test_schema_validator_flags_bad_payloads():
    base = {"rollout_id": 1, "policy_version": 1, "sample_index": 0, "group_index": 0, "record_index": 0,
            "kind": "assistant_text", "segment_index": 2, "expected_text": "", "actual_text": "", "detail": "",
            "truncated": False}
    assert tl.validate_harness_mismatch(base) == []
    assert tl.validate_harness_mismatch({**base, "segment_index": "2"}) == ["mismatch key 'segment_index' is str, expected int"]
    assert tl.validate_harness_mismatch({**base, "truncated": 1}) == ["mismatch key 'truncated' is int, expected bool"]
    assert tl.validate_harness_mismatch({**base, "expected_text": "x" * 513}) == ["mismatch key 'expected_text' longer than 512"]
    assert "missing mismatch key 'kind'" in tl.validate_harness_mismatch({k: v for k, v in base.items() if k != "kind"})


# ---------------------------------------------------------------- rl-fn-codex-rollout 1.0: per-trajectory rewards (observe only)

TRKEY = hook.TRAJECTORY_REWARDS_KEY


def _tsample(i, group, task, reward, success=None, trajectory=None, aborted=False):
    md = {"task_id": task}
    if success is not None:
        md["success"] = success
    if trajectory is not None:
        md["trajectory_id"] = trajectory
    return SimpleNamespace(index=i, group_index=group, rollout_id=0, metadata=md, reward=reward,
                           status=SimpleNamespace(value="aborted" if aborted else "completed"),
                           response_length=4, weight_versions=None, remove_sample=False)


def test_trajectory_reward_records_fields_and_cap():
    groups = [[_tsample(0, 0, "fix-git", 1.0, True, "t0"), _tsample(1, 0, "fix-git", 0.0, False)],
              [[_tsample(10, 1, "regex-log", float("nan"), aborted=True)]],
              [SimpleNamespace(index=20, group_index=2, metadata=None, reward=0.5, status="completed")]]
    recs = hook.trajectory_reward_records(None, groups, None, limit=256)
    assert recs[0] == {"sample_index": 0, "group_index": 0, "task_id": "fix-git", "trajectory_id": "t0",
                       "reward": 1.0, "success": True, "aborted": False}
    assert recs[1]["trajectory_id"] == "0" and recs[1]["success"] is False and recs[1]["reward"] == 0.0
    assert recs[2] == {"sample_index": 10, "group_index": 1, "task_id": "regex-log", "trajectory_id": "0",
                       "reward": None, "success": None, "aborted": True}
    assert recs[3]["task_id"] == "" and recs[3]["reward"] == 0.5
    for r in recs:
        assert tl.validate_trajectory_reward({**r, "rollout_id": 0, "policy_version": 0}) == []
    assert len(hook.trajectory_reward_records(None, groups, None, limit=2)) == 2
    assert hook.trajectory_reward_records(None, groups, None, limit=0) == []


@pytest.mark.parametrize("observe", [False, True])
def test_build_metadata_carries_trajectory_rewards_only_when_observing(observe):
    args = SimpleNamespace(yeto_rl_observe_timeline=observe)
    data = [[_tsample(0, 0, "fix-git", 1.0, True), _tsample(1, 0, "fix-git", 0.0, False)],
            [_tsample(10, 1, "regex-log", 0.0, False)]]
    hook.record_trained_groups(args, data[:1])  # the second group is filtered
    meta = hook.build_metadata(args, data)
    if not observe:
        assert TRKEY not in meta
        return
    assert [(r["task_id"], r["reward"]) for r in meta[TRKEY]] == [("fix-git", 1.0), ("fix-git", 0.0)]
    json.dumps(meta)
    handle = handle_from_metadata(meta, rollout_id=0, policy_version=0, policy_hash="h", data_pack=None)
    assert handle.trajectory_rewards == tuple(meta[TRKEY])


@pytest.mark.parametrize("observe", [False, True])
def test_driver_tapes_trajectory_rewards_only_when_observing(tmp_path, observe):
    engine = FakeEngine(tensors={"base_model.model.layer.lora_A.weight": torch.zeros(1, 2)},
                        step_delta=1.0, placement_kind="fixed-partition")
    recs = hook.trajectory_reward_records(
        None, [[_tsample(7, 2, "fix-git", 1.0, True), _tsample(8, 2, "fix-git", 0.0, False)]], None, limit=256)
    per_round = {0: tuple(recs), 1: None}
    original = engine.rollout.generate
    engine.rollout.generate = lambda r, **kw: dataclasses.replace(original(r, **kw), trajectory_rewards=per_round[r])
    _driver(engine, tmp_path, observe).run()
    events = [json.loads(l) for l in (tmp_path / "e.jsonl").read_text().splitlines()]
    got = [e for e in events if e["event"] == tl.TRAJECTORY_REWARD_EVENT]
    if not observe:
        assert got == [] and not any("task_id" in e for e in events)
        return
    assert [(e["rollout_id"], e["sample_index"], e["task_id"], e["reward"], e["success"]) for e in got] == [
        (0, 7, "fix-git", 1.0, True), (0, 8, "fix-git", 0.0, False)]
    for e in got:
        assert e["policy_version"] == 0 and e["profile_hash"] and "t" in e
        assert tl.validate_trajectory_reward(e) == []
    names = [e["event"] for e in events]
    assert names.index("rl_round_labels") < names.index(tl.TRAJECTORY_REWARD_EVENT)


def test_trajectory_reward_records_carry_codex_exit_diagnostics_when_present():
    s = _tsample(3, 1, "fix-git", 0.0, False, "t3")
    s.metadata.update({
        "exit_status": "max_seq_len",
        "agent_metrics": {"turns": 7, "terminal_calls": 6, "submit_calls": 0, "parse_failures": 1,
                          "max_seq_len_hit": 1, "timed_out": 0, "total_tool_time": 1.5, "x": True},
        "tbench_trusted_outcome": {"testsh_rc": 1, "passed": False},
    })
    rec = hook.trajectory_reward_records(None, [[s]], None, limit=8)[0]
    assert rec["exit_status"] == "max_seq_len" and rec["turns"] == 7 and rec["submit_calls"] == 0
    assert rec["max_seq_len_hit"] == 1 and rec["testsh_rc"] == 1
    assert "total_tool_time" not in rec and "x" not in rec
    assert tl.validate_trajectory_reward({**rec, "rollout_id": 0, "policy_version": 0}) == []
    assert tl.validate_trajectory_reward({**rec, "rollout_id": 0, "policy_version": 0, "turns": True})
    # timeout outcomes carry no verdict: testsh_rc None is kept, not dropped
    s.metadata["tbench_trusted_outcome"] = {"testsh_rc": None}
    assert hook.trajectory_reward_records(None, [[s]], None, limit=8)[0]["testsh_rc"] is None


def test_driver_tapes_optional_trajectory_diagnostics(tmp_path):
    engine = FakeEngine(tensors={"base_model.model.layer.lora_A.weight": torch.zeros(1, 2)},
                        step_delta=1.0, placement_kind="fixed-partition")
    rec = {**hook.trajectory_reward_records(None, [[_tsample(7, 2, "fix-git", 0.0, False)]], None, limit=8)[0],
           "exit_status": "max_turns", "turns": 12, "testsh_rc": 1}
    per_round = {0: (rec,), 1: None}
    original = engine.rollout.generate
    engine.rollout.generate = lambda r, **kw: dataclasses.replace(original(r, **kw), trajectory_rewards=per_round[r])
    _driver(engine, tmp_path, True).run()
    got = [json.loads(l) for l in (tmp_path / "e.jsonl").read_text().splitlines()]
    got = [e for e in got if e["event"] == tl.TRAJECTORY_REWARD_EVENT]
    assert [(e["exit_status"], e["turns"], e["testsh_rc"]) for e in got] == [("max_turns", 12, 1)]
    assert "terminal_calls" not in got[0]


def test_trajectory_reward_records_carry_end_reason_and_last_completion():
    s = _tsample(3, 1, "regex-log", 0.0, False, "t3")
    s.metadata.update({"exit_status": "max_seq_len", "agent_metrics": {
        "turns": 1, "end_reason": "CodexSequenceLimit: Miles returned a truncated sample",
        "last_completion": {"finish_reason": "length", "content_chars": 0, "reasoning_chars": 15000, "tool_calls": 0,
                            "tool_names": "", "completion_tokens": 4096, "content_head": "", "reasoning_tail": "x" * 500}}})
    rec = hook.trajectory_reward_records(None, [[s]], None, limit=8)[0]
    assert rec["end_reason"].startswith("CodexSequenceLimit") and rec["last_finish_reason"] == "length"
    assert rec["last_reasoning_chars"] == 15000 and rec["last_completion_tokens"] == 4096 and len(rec["last_reasoning_tail"]) == 200
    assert tl.validate_trajectory_reward({**rec, "rollout_id": 0, "policy_version": 0}) == []
