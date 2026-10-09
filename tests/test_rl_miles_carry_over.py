"""agentic-rollout-utilization 4.1: Miles single-turn carry-over (fake engine, CPU).

A fake partial-rollout engine mimics what Miles does with ``--partial-rollout``:
groups not finished at the cut-off go back to the data buffer with the tokens
they already have (stamped with the version that generated them), and the next
rollout continues them through yeto's ``--buffer-filter-path``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace

import pytest

from yeto.rl.adapters.miles import carry_over
from yeto.rl.adapters.miles import rollout_meta_hook as hook
from yeto.rl.adapters.miles.rollout import handle_from_metadata, policy_token
from yeto.rl.engine.policy_age import PolicyAgeError, check_batch_ages

HASHES = {v: f"{v:x}" * 64 for v in range(1, 8)}
HASHES = {v: h[:64] for v, h in HASHES.items()}
PROMPT = [101, 102, 103]


def tok(version: int) -> str:
    return policy_token(version, HASHES[version])


class Status(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    ABORTED = "aborted"


@dataclass
class Span:  # miles.utils.types.WeightVersionSpan
    version: str
    abs_start: int
    abs_end: int


@dataclass
class Call:  # miles.utils.types.WeightVersionsPerCall
    spans: list


@dataclass
class Sample:  # the upstream Sample fields the hooks read
    index: int
    group_index: int
    reward: float | None = None
    tokens: list = field(default_factory=lambda: list(PROMPT))
    response_length: int = 0
    weight_versions: list = field(default_factory=list)
    rollout_log_probs: list | None = None
    status: Status = Status.PENDING
    metadata: dict = field(default_factory=dict)

    def get_reward_value(self, args):
        return self.reward


class FakePartialEngine:
    """Generates ``n`` tokens of a sample under one version (one engine call)."""

    def generate(self, sample: Sample, version: int, n: int, *, finish: bool, reward: float = 1.0):
        start = len(sample.tokens)
        sample.tokens = sample.tokens + [7] * n
        sample.response_length += n
        sample.rollout_log_probs = (sample.rollout_log_probs or []) + [-0.5] * n
        sample.weight_versions.append(Call([Span(tok(version), start, start + n)]))
        sample.status = Status.COMPLETED if finish else Status.ABORTED
        if finish:
            sample.reward = reward


def new_group(gi: int, size: int = 2) -> list[Sample]:
    return [Sample(index=gi * 10 + i, group_index=gi) for i in range(size)]


@pytest.fixture
def sink(tmp_path, monkeypatch):
    spec = f"dir:{tmp_path}"
    monkeypatch.setenv(hook.META_SINK_ENV, spec)
    return spec


def round_metadata(args, kept, sink, scorer=None):
    hook.record_trained_groups(args, kept)
    payload = hook.build_metadata(args, kept, sink)
    if scorer is not None:  # replace the router scorer of build_metadata
        payload.update(hook.carry_over_fields(args, [s for g in kept for s in g], sink, scorer))
    args._yeto_trained_group_keys = None
    return payload


def test_response_token_versions_follow_the_engine_spans():
    s = Sample(index=0, group_index=0)
    engine = FakePartialEngine()
    engine.generate(s, 3, 2, finish=False)
    engine.generate(s, 4, 3, finish=True)
    assert carry_over.response_token_versions(s) == [3, 3, 4, 4, 4]
    assert carry_over.group_versions([s]) == ({3, 4}, False)
    s.weight_versions.append(Call([Span("default", 0, len(s.tokens))]))  # unknown version
    assert carry_over.group_versions([s])[1] is True


def test_fake_engine_carry_over_segments_window_and_over_limit_discard(sink):
    """Limit 1: a group cut off at v3 continues at v4 with segments [3, 4]; a group
    still unfinished at v5 (oldest v3, two versions old) is discarded and counted."""
    engine = FakePartialEngine()
    args = SimpleNamespace(yeto_rl_max_policy_age=1, n_samples_per_prompt=2)
    buffer: list = []

    # ---- round v3: g0 finishes, g1/g2 are cut off with 2 tokens each
    hook.put_policy_token(tok(3), sink)
    g0, g1, g2 = new_group(0), new_group(1), new_group(2)
    for s in g0:
        engine.generate(s, 3, 4, finish=True, reward=float(s.index % 2))
    for s in g1 + g2:
        engine.generate(s, 3, 2, finish=False)
    buffer.extend([g1, g2])  # Miles abort(): partial groups back into the buffer
    p3 = round_metadata(args, [g0], sink)
    assert p3["groups"][0]["policy_token"] == tok(3)
    assert p3["groups"][0]["policy_versions"] == [3] and p3["cross_version_tokens"] == 0

    # ---- round v4: the buffer filter hands both back (age 1 <= 1)
    hook.put_policy_token(tok(4), sink)
    selected = hook.policy_buffer_filter(args, None, buffer, 4)
    assert selected == [g1, g2] and buffer == []
    for s in g1:
        engine.generate(s, 4, 3, finish=True, reward=float(s.index % 2))
    for s in g2:
        engine.generate(s, 4, 1, finish=False)
    buffer.append(g2)
    g3 = new_group(3)
    for s in g3:
        engine.generate(s, 4, 4, finish=True, reward=float(s.index % 2))
    p4 = round_metadata(args, [g1, g3], sink)
    rec = {g["group_id"]: g for g in p4["groups"]}
    assert rec["g1"]["policy_token"] == tok(3)  # names the OLDEST version
    assert rec["g1"]["policy_versions"] == [3, 4]
    assert rec["g3"]["policy_token"] == tok(4) and rec["g3"]["policy_versions"] == [4]
    assert carry_over.response_token_versions(g1[0]) == [3, 3, 4, 4, 4]
    assert (p4["carried_in_groups"], p4["carried_in_trajectories"], p4["carried_in_tokens"]) == (2, 4, 8)
    assert p4["resubmitted_groups"] == 2 and p4["over_age_discarded_groups"] == 0
    assert (p4["cross_version_tokens"], p4["trained_response_tokens"]) == (4, 18)
    handle = handle_from_metadata(p4, rollout_id=4, policy_version=4, policy_hash=HASHES[4], data_pack=None)
    by_id = {g.group_id: g for g in handle.groups}
    assert by_id["g1"].policy_versions == (3, 4) and by_id["g3"].policy_versions == (4,)
    assert handle.carry_over["carried_in_tokens"] == 8 and handle.carry_over["max_policy_age"] == 1
    # the driver's window check accepts the batch at limit 1, refuses it at limit 0
    assert check_batch_ages(handle.groups, 4, 1, HASHES) == []
    assert check_batch_ages(handle.groups, 4, 0, HASHES)

    # ---- round v5: g2 (oldest v3) is two versions old -> discarded and reported
    hook.put_policy_token(tok(5), sink)
    assert hook.policy_buffer_filter(args, None, buffer, 4) == [] and buffer == []
    g4 = new_group(4)
    for s in g4:
        engine.generate(s, 5, 4, finish=True, reward=float(s.index % 2))
    p5 = round_metadata(args, [g4], sink)
    assert (p5["over_age_discarded_groups"], p5["over_age_discarded_trajectories"],
            p5["over_age_discarded_tokens"]) == (1, 2, 6)
    assert p5["carried_in_groups"] == 0 and p5["cross_version_tokens"] == 0


def test_unknown_token_versions_are_discarded_not_guessed(sink):
    args = SimpleNamespace(yeto_rl_max_policy_age=1)
    g = new_group(0)
    for s in g:
        s.tokens = PROMPT + [7, 7]
        s.response_length = 2  # tokens without any engine span
    hook.put_policy_token(tok(4), sink)
    assert hook.policy_buffer_filter(args, None, [g], 2) == []
    assert carry_over.take_carry_stats(args, 4)["unknown_version_discarded_groups"] == 1


def test_limit_zero_metadata_and_buffer_filter_are_unchanged(sink):
    """Limit 0 (default): the payload is byte-identical with or without the switch
    attribute, and the buffer filter keeps the old complete-current-group rule."""
    engine = FakePartialEngine()
    hook.put_policy_token(tok(3), sink)
    kept = [new_group(0), new_group(1)]
    for g in kept:
        for s in g:
            engine.generate(s, 3, 4, finish=True, reward=float(s.index % 2))
    a = round_metadata(SimpleNamespace(), kept, sink)
    b = round_metadata(SimpleNamespace(yeto_rl_max_policy_age=0), kept, sink)
    assert hook._jsonable(a) == hook._jsonable(b)
    assert "policy_versions" not in a["groups"][0] and "max_policy_age" not in a
    partial = new_group(2)
    for s in partial:
        engine.generate(s, 3, 2, finish=False)
    buffer = [partial, kept[0]]
    assert hook.policy_buffer_filter(SimpleNamespace(n_samples_per_prompt=2), None, buffer, 4) == [kept[0]]
    handle = handle_from_metadata(a, rollout_id=3, policy_version=3, policy_hash=HASHES[3], data_pack=None)
    assert handle.carry_over is None and handle.cross_version_truncated_fraction is None
    assert all(g.policy_versions is None for g in handle.groups)


def test_cross_version_truncation_estimate_with_known_ratios(sink):
    """Older tokens: ratio exp(current - generated); TIS bounds [0, 2] -> known fraction."""
    engine = FakePartialEngine()
    args = SimpleNamespace(yeto_rl_max_policy_age=1, tis_clip=2.0, tis_clip_low=0.0)
    s = Sample(index=0, group_index=0)
    engine.generate(s, 3, 4, finish=False)
    engine.generate(s, 4, 2, finish=True, reward=1.0)
    # generated -0.5 everywhere; current: ratios e^1 (truncated), e^0, e^0.5, e^0.9 (truncated)
    current = [0.5, -0.5, 0.0, 0.4, -0.5, -0.5]
    out = carry_over.estimate_cross_version_truncation(args, [s], 4, lambda _s: current)
    assert out["cross_version_scored_tokens"] == 4
    assert math.isclose(out["cross_version_truncated_fraction"], 0.5)
    assert carry_over.estimate_cross_version_truncation(args, [s], 4, lambda _s: None) == {
        "cross_version_truncated_fraction": None, "cross_version_scored_tokens": 0,
        "cross_version_unscored_samples": 1}
    hook.put_policy_token(tok(4), sink)
    payload = round_metadata(args, [[s]], sink, scorer=lambda _s: current)
    handle = handle_from_metadata(payload, rollout_id=4, policy_version=4, policy_hash=HASHES[4],
                                  data_pack=None)
    assert math.isclose(handle.cross_version_truncated_fraction, 0.5)


def test_governor_fallback_lowers_the_rollout_limit_through_the_sink(sink, tmp_path):
    from yeto.rl.adapters.miles.rollout import DirMetadataSource

    args = SimpleNamespace(yeto_rl_max_policy_age=1)
    assert carry_over.max_policy_age(args) == 1
    DirMetadataSource(tmp_path).set_max_policy_age(0)
    assert carry_over.max_policy_age(args) == 0
    assert carry_over.max_policy_age(SimpleNamespace()) == 0


def test_miles_stage_two_support_and_single_turn_check():
    from yeto.rl.adapters.miles.policy_age import SUPPORT, check_task, policy_age_argv

    # agentic-rollout-utilization 5 (stage 3) raised the declared stage; limit still <= 1
    assert (SUPPORT.stage, SUPPORT.max_policy_age) == (3, 1)
    SUPPORT.check(1)
    with pytest.raises(PolicyAgeError, match="at most 1"):
        SUPPORT.check(2)
    assert policy_age_argv(0) == () and policy_age_argv(1) == ("--partial-rollout",)
    check_task(1)
    check_task(0, custom_agent="x.y", custom_generate="x.z")
    with pytest.raises(PolicyAgeError, match="stage 3"):
        check_task(1, custom_generate="yeto.rl.harness.codex.codex_openenv_generate.generate")
    with pytest.raises(PolicyAgeError, match="recompute"):
        check_task(1, recompute_prefill=True)


def test_launcher_refuses_agentic_carry_over_before_launch():
    from tests.test_rl_launcher import _args
    from yeto.launcher import _prepare_rl_args

    args = _args(["--rl-max-policy-age", "1", "--custom-generate-function-path",
                  "yeto.rl.harness.codex.codex_openenv_generate.generate"])
    with pytest.raises(PolicyAgeError, match="stage 3"):
        _prepare_rl_args(args)


def test_translate_emits_partial_rollout_from_the_limit_only():
    from yeto.rl.adapters.miles.config import MilesConfigError
    from yeto.rl.engine.algorithm import AlgorithmSpec, CorrectionSpec, ExecutionSpec
    from tests.test_rl_miles_adapter_config import make_config as _config  # noqa: PLC0415

    tolerant = AlgorithmSpec(execution=ExecutionSpec(max_policy_staleness=1),
                             correction=CorrectionSpec(method="tis", tis_clip=2.0, tis_clip_low=0.0))
    from yeto.rl.adapters.miles.config import translate_run_config

    zero = translate_run_config(_config(), tolerant).argv
    one = translate_run_config(_config(), tolerant, max_policy_age=1).argv
    assert "--partial-rollout" not in zero and "--mask-offpolicy-in-partial-rollout" not in zero
    assert [a for a in one if a not in zero] == ["--partial-rollout"]
    with pytest.raises(MilesConfigError, match="max_policy_staleness >= 1"):
        translate_run_config(_config(), AlgorithmSpec(), max_policy_age=1)


def test_driver_reports_carry_over_and_governs_on_the_rollout_estimate(tmp_path):
    import dataclasses

    from tests.test_rl_policy_age import _driver

    driver = _driver(tmp_path, 1)
    original = driver.rollout.generate
    calls = []
    driver.rollout.set_max_policy_age = calls.append

    def gen(rid, **kw):
        h = original(rid, **kw)
        return dataclasses.replace(h, carry_over={"max_policy_age": 1, "carried_in_groups": 1},
                                   cross_version_truncated_fraction=0.9)

    driver.rollout.generate = gen
    driver.run()
    events = [json.loads(l) for l in (tmp_path / "e.jsonl").read_text().splitlines()]
    carry = [e for e in events if e["event"] == "rl_rollout_carry_over"]
    assert carry and carry[0]["carried_in_groups"] == 1
    assert carry[0]["cross_version_truncated_fraction"] == 0.9
    assert [e for e in events if e["event"] == "rl_policy_age_fallback"] and calls == [0]


def test_trajectory_records_carry_start_round_and_segments(sink):
    """Dashboard field names (#160): started_rollout_id + policy_versions; absent at limit 0."""
    from yeto.rl.engine.timeline import validate_trajectory_reward

    engine = FakePartialEngine()
    s = Sample(index=0, group_index=0, metadata={"start_rollout_id": 3})
    engine.generate(s, 3, 2, finish=False)
    engine.generate(s, 4, 3, finish=True, reward=1.0)
    fresh = Sample(index=1, group_index=0)
    engine.generate(fresh, 4, 2, finish=True, reward=0.0)
    hook.put_policy_token(tok(4), sink)
    args = SimpleNamespace(yeto_rl_max_policy_age=1)
    recs = hook.trajectory_reward_records(args, [[s, fresh]], limit=10)
    assert recs[0]["started_rollout_id"] == 3 and recs[0]["policy_versions"] == [[3, 0, 2], [4, 2, 5]]
    assert recs[1]["started_rollout_id"] == 4 and recs[1]["policy_versions"] == [[4, 0, 2]]
    zero = hook.trajectory_reward_records(SimpleNamespace(), [[s, fresh]], limit=10)
    assert all("started_rollout_id" not in r and "policy_versions" not in r for r in zero)
    for r in recs:
        assert not [p for p in validate_trajectory_reward({**r, "rollout_id": 4, "policy_version": 4})
                    if "started_rollout_id" in p or "policy_versions" in p]
