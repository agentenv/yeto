"""Trajectory evidence and policy identity on compaction segments (compaction-1118 C11).

With context compaction on, one trial is k buffer samples (one per segment) that
share ``group_index``, ``index`` and ``reward`` and carry ``metadata.compaction``
(segment_index / num_segments / trajectory_id ...). The evidence binds every
segment, names the policy of each segment (a reload between segments changes
it), and records enough to restore the trial's segments. Without compaction the
evidence file is byte-for-byte what main writes (golden digests below were
produced by main ff50e575 from these same fixtures).
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto.rl import tbench_outcome
from yeto.rl.adapters.miles.full_parameter import evidence as fp_evidence
from yeto.rl.engine.policy_versions import PublishedPolicies
from yeto.rl.trajectory_evidence import (
    build_trajectory_batch_evidence,
    read_trajectory_batch_evidence,
    write_trajectory_batch_evidence,
)

ROUND_HASH = "a" * 64
REWARD_HASH = "b" * 64
OLDER_HASH = "c" * 64
OLDEST_HASH = "d" * 64
PUBLISHED = PublishedPolicies.from_dict({"5": OLDEST_HASH, "6": OLDER_HASH, "7": ROUND_HASH})


def _spans(tokens: int, response: int, versions: tuple[int, ...]):
    start = tokens - response
    return [SimpleNamespace(spans=[SimpleNamespace(version=str(v), abs_start=start + i, abs_end=start + i + 1)])
            for i, v in enumerate(versions)]


def _sample(*, group_index, index, reward=1.0, prompt="2+2?", base=100, versions=(7, 7),
            status="completed", compaction=None, mask=None):
    response = len(versions)
    tokens = [base, base + 1, *(base + 10 + i for i in range(response))]
    metadata = {"untrusted": {"not": "persisted"}}
    if compaction is not None:
        metadata["compaction"] = compaction
    return SimpleNamespace(
        group_index=group_index, index=index, prompt=prompt, reward=reward,
        status=SimpleNamespace(value=status), metadata=metadata, tokens=tokens,
        response_length=response, loss_mask=list(mask or [1] * response),
        rollout_log_probs=[-0.25 - 0.01 * i for i in range(response)],
        weight_versions=_spans(len(tokens), response, versions),
    )


def _compaction(segment_index, num_segments, trajectory_id="trial-7", **overrides):
    value = {"schema": 1, "trajectory_id": trajectory_id, "segment_index": segment_index,
             "num_segments": num_segments, "context_budget": 262144, "summary_ok": True,
             "kept_steps": 2, "segment_tokens": 2, "tokens_after": 2 * (num_segments - segment_index - 1)}
    value.update(overrides)
    return value


def _trial(*, group_index, index, versions_per_segment, reward=1.0, trajectory_id=None, last_status="completed"):
    n = len(versions_per_segment)
    return [
        _sample(group_index=group_index, index=index, reward=reward, base=100 * (index + 1) + 10 * k,
                versions=versions, compaction=_compaction(k, n, trajectory_id or f"trial-{index}"),
                status=last_status if k == n - 1 else "completed")
        for k, versions in enumerate(versions_per_segment)
    ]


def _args(tmp_path, **overrides):
    values = {
        "yeto_rl_trajectory_evidence_dir": str(tmp_path / "evidence"),
        "yeto_rl_trajectory_evidence_kind": "reward-function",
        "yeto_rl_trajectory_evidence_schema_version": 1,
        "yeto_rl_reward_sha256": REWARD_HASH,
        "yeto_rl_max_policy_age": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _persist(tmp_path, groups, **overrides) -> Path:
    fp_evidence.persist_published_round_evidence(
        _args(tmp_path, **overrides), groups, f"yeto:7:{ROUND_HASH}", PUBLISHED)
    return tmp_path / "evidence" / "rollout-00000007.json"


def _build(groups, **kwargs):
    kwargs.setdefault("rollout_id", 7)
    kwargs.setdefault("behavior_policy_hash", ROUND_HASH)
    kwargs.setdefault("reward_contract_hash", REWARD_HASH)
    kwargs.setdefault("evidence_kind", "reward-function")
    return build_trajectory_batch_evidence(groups, **kwargs)


# --------------------------------------------------------------------------
# default (no compaction): evidence bytes equal main's


def _default_groups():
    return [
        [_sample(group_index=3, index=12, reward=1.0, versions=(6, 7)),
         _sample(group_index=3, index=13, reward=0.0, base=200)],
        [_sample(group_index=4, index=16, reward=0.5, prompt="3+3?", base=300, status="truncated")],
    ]


def _tbench_metadata():
    return {
        "reward": 1.0, "exit_status": "completed",
        **tbench_outcome.build_outcome_metadata(
            task_id="fix-git", sample_id="train:fix-git:r0", episode_id="openenv-0123456789abcdef",
            status="completed", reward=1.0, verifier=tbench_outcome.TEST_SH_VERIFIER, testsh_rc=0),
    }


def _legacy_segment(index):
    metadata = copy.deepcopy(_tbench_metadata())
    metadata.update({
        "compaction_schema_version": 1, "compaction_trajectory_id": "session-0123456789abcdef",
        "compaction_context_window": index // 2, "compaction_context_budget": 8192,
        "compaction_segment_index": index,
        "compaction_segment_type": "execution" if index % 2 == 0 else "summary",
    })
    return SimpleNamespace(group_index=3, index=7, status=SimpleNamespace(value="completed"),
                           metadata=metadata, reward=1.0, tokens=[101, 200 + index, 210 + index, 220 + index],
                           response_length=3, loss_mask=[1, 0, 1],
                           rollout_log_probs=[-0.1 - index, -0.2 - index, -0.3 - index])


GOLDEN = {
    "schema3-reward-function": "c40458a2eeb2056ac5847a0df28b4a4bc8b27042e4d45e08e4fa823b3f3c9bf0",
    "schema4-hook": "a98f6ec274e8f0cc91645b93f5b4c93a5d69a61af39b7b6748fb926ff3b8e663",
    "schema3-legacy-tbench-compaction": "b57d63ec9a63e1833d330b6478f834a8f6e95b06fd11bba9a49e15bc6f40a623",
}


def _golden_files(tmp_path) -> dict[str, bytes]:
    out = {}
    evidence = _build(_default_groups())
    out["schema3-reward-function"] = write_trajectory_batch_evidence(tmp_path / "s3", evidence).read_bytes()
    out["schema4-hook"] = _persist(tmp_path / "s4", _default_groups()).read_bytes()
    legacy = _build([[[_legacy_segment(i) for i in range(3)]]], rollout_id=5, evidence_kind="terminal-bench-2.1")
    out["schema3-legacy-tbench-compaction"] = write_trajectory_batch_evidence(tmp_path / "lg", legacy).read_bytes()
    return out


def test_default_evidence_files_are_byte_identical_to_main(tmp_path):
    files = _golden_files(tmp_path)
    assert {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()} == GOLDEN
    for raw in files.values():
        assert b"num_segments" not in raw and b'"compaction":' not in raw


# --------------------------------------------------------------------------
# segmented trials


def _segmented_groups():
    """Group 3: trial 12 in three segments (a reload v5 -> v6 -> v7 between them)
    beside an uncompacted trial 13; group 4: trial 16 in two segments, the last
    truncated (compactions used up)."""
    return [
        [*_trial(group_index=3, index=12, versions_per_segment=[(5, 5), (5, 6), (7, 7)]),
         _sample(group_index=3, index=13, reward=0.0, base=900)],
        _trial(group_index=4, index=16, reward=0.5, versions_per_segment=[(6, 6), (7, 7)],
               last_status="truncated"),
    ]


def test_samples_sharing_an_index_without_compaction_metadata_stay_refused():
    """Main took every segment of a compacted trial for its own logical sample:
    the second segment failed the whole round with this error. It now holds
    only for samples that are not compaction segments."""
    flat = [_sample(group_index=3, index=12), _sample(group_index=3, index=12, base=300)]
    with pytest.raises(ValueError, match="duplicate logical sample indexes"):
        _build([flat])


def test_segmented_trials_bind_every_segment_with_its_own_policy(tmp_path):
    path = _persist(tmp_path, _segmented_groups())
    evidence = read_trajectory_batch_evidence(path)
    assert evidence.schema_version == 4
    rows = [(e.sample_index, e.compaction_trajectory_id, e.compaction_segment_index,
             e.compaction_num_segments, e.behavior_policy_version) for e in evidence.envelopes]
    assert rows == [
        (12, "trial-12", 0, 3, 5), (12, "trial-12", 1, 3, 5), (12, "trial-12", 2, 3, 7),
        (13, None, None, None, 7),
        (16, "trial-16", 0, 2, 6), (16, "trial-16", 1, 2, 7),
    ]
    assert [e.behavior_policy_hash for e in evidence.envelopes] == [
        OLDEST_HASH, OLDEST_HASH, ROUND_HASH, ROUND_HASH, OLDER_HASH, ROUND_HASH]
    assert all(e.compaction_segment_type is None for e in evidence.envelopes)
    assert {e.compaction_context_budget for e in evidence.envelopes} == {262144, None}
    assert len(set(evidence.trajectory_ids)) == 6
    assert evidence.trained_tokens == 2 * 6
    raw = json.loads(path.read_text())
    # the segment count is written only on segment envelopes
    assert ["compaction_num_segments" in item for item in raw["envelopes"]] == [
        True, True, True, False, True, True]


def test_a_trial_is_restored_from_its_segment_envelopes(tmp_path):
    evidence = read_trajectory_batch_evidence(_persist(tmp_path, _segmented_groups()))
    trials: dict[int, list] = {}
    for item in evidence.envelopes:
        trials.setdefault(item.sample_index, []).append(item)
    assert {index: len(items) for index, items in trials.items()} == {12: 3, 13: 1, 16: 2}
    for items in trials.values():
        if items[0].compaction_num_segments is None:  # an uncompacted trial
            assert len(items) == 1 and items[0].compaction_trajectory_id is None
            continue
        assert [i.compaction_segment_index for i in items] == list(range(len(items)))
        assert {i.compaction_num_segments for i in items} == {len(items)}
        assert len({(i.reward, i.task_id, i.prompt_group_id, i.compaction_trajectory_id)
                    for i in items}) == 1


def test_segments_out_of_row_order_are_bound_in_segment_order():
    """The reader keeps the buffer's row order: a trial's segments need not be
    adjacent nor in order there; the evidence binds them in segment order."""
    groups = _segmented_groups()
    first, second, third, flat = groups[0]
    groups[0][:] = [third, first, flat, second]
    shuffled = _build(groups)
    assert shuffled == _build(_segmented_groups())
    assert [(e.sample_index, e.compaction_segment_index) for e in shuffled.envelopes][:4] == [
        (12, 0), (12, 1), (12, 2), (13, None)]


def test_identical_segments_keep_distinct_trajectory_ids():
    trial = _trial(group_index=0, index=0, versions_per_segment=[(7,), (7,)])
    trial[1].tokens = list(trial[0].tokens)
    evidence = _build([trial])
    assert len(set(evidence.trajectory_ids)) == 2


def test_segmented_trial_tampering_is_rejected(tmp_path):
    path = _persist(tmp_path, _segmented_groups())
    original = json.loads(path.read_text())

    def rewrite(mutate):
        value = copy.deepcopy(original)
        mutate(value)
        path.write_text(json.dumps(value))
        path.chmod(0o600)
        return path

    def drop_segment(value):
        del value["envelopes"][1]

    def null_count(value):
        value["envelopes"][3]["compaction_num_segments"] = None

    def recount(value):
        for item in value["envelopes"][:3]:
            item["compaction_num_segments"] = 4

    def relabel_type(value):
        value["envelopes"][0]["compaction_segment_type"] = "execution"

    for mutate, message in ((drop_segment, "every segment of a trial"),
                            (null_count, "fields are malformed"),
                            (recount, "every segment of a trial"),
                            (relabel_type, "has no segment type")):
        with pytest.raises(ValueError, match=message):
            read_trajectory_batch_evidence(rewrite(mutate))


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (lambda g: g[0][1].metadata["compaction"].update(segment_index=0), "not 0..n-1"),
        (lambda g: g[0].pop(1), "num_segments 3 != 2"),
        (lambda g: setattr(g[0][1], "reward", 0.0), "share one reward"),
        (lambda g: setattr(g[0][0].status, "value", "truncated"), "segment 0 has status 'truncated'"),
        (lambda g: setattr(g[0][2].status, "value", "aborted"), "segment 2 has status 'aborted'"),
        (lambda g: g[0][1].metadata["compaction"].update(trajectory_id="other"), "mix compaction trajectory"),
        (lambda g: g[0][1].metadata["compaction"].update(context_budget=4096), "context budget"),
        (lambda g: g[0][1].metadata["compaction"].update(schema=2), "unsupported schema"),
        (lambda g: g[0][1].metadata.pop("compaction"), "lack compaction metadata"),
        (lambda g: g[0][3].metadata.update(compaction=_compaction(0, 1, "trial-12")), "spans sample indexes"),
        (lambda g: setattr(g[0][1], "group_index", 9), "mixes group identities"),
    ),
)
def test_malformed_segmented_trial_fails_closed(mutate, message):
    groups = _segmented_groups()
    mutate(groups)
    with pytest.raises(ValueError, match=message):
        _build(groups, behavior_policy_of=lambda segments: (7, ROUND_HASH), max_policy_age=2)


def test_one_trial_split_across_two_indexes_is_refused():
    """A trainer that numbered each segment as its own sample (C3: renumber per
    trial, not per sample) would hand the evidence one trajectory under two
    indexes; the evidence refuses it instead of binding two half trials."""
    trial = _trial(group_index=0, index=0, versions_per_segment=[(7,), (7,)])
    trial[1].index = 1
    with pytest.raises(ValueError, match=r"spans sample indexes \[0, 1\]"):
        _build([trial])


@pytest.mark.parametrize(
    ("versions", "message"),
    (([(7,), (8,)], "v8 of a trained trajectory was not published"),
     ([(4,), (7,)], "not published"),
     ([(7,), (7, None)], "unknown policy version")),
)
def test_one_bad_segment_refuses_the_trial(tmp_path, versions, message):
    """Identity is checked per segment; any segment outside the published window
    (or of unknown version) refuses the round, so the reader gate must drop the
    whole trial (in practice its whole group) before training."""
    trial = _trial(group_index=0, index=0, versions_per_segment=versions)
    if None in versions[-1]:
        trial[-1].weight_versions = trial[-1].weight_versions[:1]
    with pytest.raises((RuntimeError, ValueError), match=message):
        _persist(tmp_path, [trial])


def test_segment_older_than_the_age_limit_refuses_the_trial(tmp_path):
    trial = _trial(group_index=0, index=0, versions_per_segment=[(5,), (7,)])
    with pytest.raises(ValueError, match="policy-age limit"):
        _persist(tmp_path, [trial], yeto_rl_max_policy_age=1)


def _segment_policy(segments):
    version = int(segments[0].weight_versions[0].spans[0].version)
    return version, {5: OLDEST_HASH, 6: OLDER_HASH, 7: ROUND_HASH, 8: "e" * 64}[version]


def test_a_segment_newer_than_its_round_refuses_the_trial():
    trial = _trial(group_index=0, index=0, versions_per_segment=[(7,), (8,)])
    with pytest.raises(ValueError, match="newer than its round"):
        _build([trial], behavior_policy_of=_segment_policy, max_policy_age=2)


def test_rollout_ahead_rebinding_checks_every_segment():
    from yeto.rl.trajectory_evidence import bind_round_policy

    older = _trial(group_index=0, index=0, versions_per_segment=[(5,), (6,)])
    evidence = _build([older], behavior_policy_of=_segment_policy, max_policy_age=2)
    assert bind_round_policy(evidence, "e" * 64).behavior_policy_hash == "e" * 64
    # one segment of the round's own version keeps the round policy it names
    current = _trial(group_index=0, index=0, versions_per_segment=[(6,), (7,)])
    evidence = _build([current], behavior_policy_of=_segment_policy, max_policy_age=2)
    with pytest.raises(ValueError, match="round's own version"):
        bind_round_policy(evidence, "e" * 64)


def test_schema3_and_relay_schema5_bind_segments_too(tmp_path):
    groups = _segmented_groups()
    schema3 = _build(groups)
    assert schema3.schema_version == 3
    assert [e.compaction_num_segments for e in schema3.envelopes] == [3, 3, 3, None, 2, 2]
    cut = "f" * 64
    schema5 = _build(_segmented_groups(), behavior_cut_of=lambda s: (4, cut), merged_version=5,
                     max_policy_age=1)
    assert schema5.schema_version == 5
    for evidence in (schema3, schema5):
        path = write_trajectory_batch_evidence(tmp_path / f"s{evidence.schema_version}", evidence)
        assert read_trajectory_batch_evidence(path) == evidence


def test_legacy_and_segmented_compaction_do_not_mix():
    legacy = [_legacy_segment(i) for i in range(3)]
    trial = _trial(group_index=3, index=8, versions_per_segment=[(7,), (7,)])
    with pytest.raises(ValueError, match="mixes legacy and segmented"):
        _build([[legacy, *trial]], evidence_kind="terminal-bench-2.1")
    legacy[0].metadata["compaction"] = _compaction(0, 1)
    with pytest.raises(ValueError, match="not checked as a trial"):
        _build([[legacy]], evidence_kind="terminal-bench-2.1")


def test_the_segmented_path_never_calls_the_legacy_ppo_check(monkeypatch):
    import yeto.rl.compaction as compaction

    def refuse(*_args, **_kwargs):
        raise AssertionError("validate_compaction_segments is the legacy PPO check")

    monkeypatch.setattr(compaction, "validate_compaction_segments", refuse)
    assert len(_build(_segmented_groups()).envelopes) == 6
    root = Path(compaction.__file__).resolve().parents[2]
    callers = sorted(str(p.relative_to(root)) for p in (root / "yeto").rglob("*.py")
                     if "validate_compaction_segments(" in p.read_text(encoding="utf-8"))
    assert callers == ["yeto/rl/compaction.py"]


# --------------------------------------------------------------------------
# contracts.validate_compaction_group on buffer sample dicts (C3's reader)


def _buffer_trial(index, n, *, reward=1.0, trajectory_id=None, statuses=None):
    statuses = statuses or ["completed"] * n
    return [{"group_index": 2, "index": index, "reward": reward, "status": statuses[k],
             "metadata": {"compaction": _compaction(k, n, trajectory_id or f"t{index}")}}
            for k in range(n)]


def test_validate_compaction_group_counts_trials_not_samples():
    from yeto.rl.contracts import CompactionTrial, validate_compaction_group

    group = [*_buffer_trial(0, 3), {"group_index": 2, "index": 1, "reward": 0.0, "status": "completed",
                                    "metadata": {}},
             *_buffer_trial(2, 2, reward=0.5, statuses=["completed", "truncated"])]
    trials = validate_compaction_group(group)
    assert trials == (CompactionTrial(0, (0, 1, 2), "t0"), CompactionTrial(1, (3,), None),
                      CompactionTrial(2, (4, 5), "t2"))
    # rows out of order: positions come back in segment order
    shuffled = [group[2], group[3], group[0], group[5], group[1], group[4]]
    assert [t.positions for t in validate_compaction_group(shuffled)] == [(2, 4, 0), (1,), (5, 3)]
    assert [t.num_segments for t in trials] == [3, 1, 2] and len(trials) == 3
    # the trainer's numbering (one rollout_id per trial, C3) passes
    keyed = [{**row, "rollout_id": 20 + row["index"]} for row in group]
    assert [t.positions for t in validate_compaction_group(keyed)] == [(0, 1, 2), (3,), (4, 5)]
    # an uncompacted group: one trial per sample
    flat = [{"group_index": 2, "index": i, "reward": 1.0, "status": "completed", "metadata": {}}
            for i in range(4)]
    assert [t.positions for t in validate_compaction_group(flat)] == [(0,), (1,), (2,), (3,)]


@pytest.mark.parametrize(
    ("group", "message"),
    (
        ([], "empty"),
        (_buffer_trial(0, 2, statuses=["truncated", "completed"]), "segment 0 has status"),
        ([*_buffer_trial(0, 2), *_buffer_trial(1, 1, trajectory_id="t0")], "spans sample indexes"),
        ([{"group_index": 2, "index": True, "reward": 1.0, "status": "completed", "metadata": {}}],
         "sample.index"),
        ([{**_buffer_trial(0, 1)[0], "reward": float("nan")}], "no finite reward"),
        ([{**row, "rollout_id": k} for k, row in enumerate(_buffer_trial(0, 2))], "carry rollout ids"),
        ([{**row, "rollout_id": 5} for row in [*_buffer_trial(0, 2), *_buffer_trial(1, 1)]],
         "rollout id 5 is also trial 0"),
        ([{**_buffer_trial(0, 1)[0], "metadata": {"compaction": "yes"}}], "not an object"),
        ([{**_buffer_trial(0, 1)[0], "metadata": {"compaction": {**_compaction(0, 1), "summary_ok": 1}}}],
         "summary_ok"),
        ([{**_buffer_trial(0, 1)[0], "metadata": {"compaction": {**_compaction(0, 1), "kept_steps": -1}}}],
         "kept_steps"),
    ),
)
def test_validate_compaction_group_refuses(group, message):
    from yeto.rl.contracts import validate_compaction_group

    with pytest.raises(ValueError, match=message):
        validate_compaction_group(group)


def test_segment_envelope_contract():
    from yeto.rl.contracts import TrajectoryEnvelope

    base = dict(trajectory_id="t", task_id="k", prompt_group_id="r1:g0", sample_index=0,
                behavior_policy_version=1, behavior_policy_hash=ROUND_HASH, token_ids=(1, 2),
                response_token_count=1, behavior_logprobs_hash=None, reward=1.0,
                reward_contract_hash=REWARD_HASH, cleanup_evidence_hash=OLDER_HASH,
                compaction_trajectory_id="trial-0", compaction_segment_index=1,
                compaction_context_budget=8, compaction_num_segments=2)
    TrajectoryEnvelope(**base)
    for overrides, message in (({"compaction_segment_index": 2}, "exceeds its count"),
                               ({"compaction_num_segments": 0}, "segment count"),
                               ({"compaction_segment_type": "summary"}, "no segment type"),
                               ({"compaction_trajectory_id": None}, "incomplete"),
                               ({"compaction_context_budget": 0}, "context budget")):
        with pytest.raises(ValueError, match=message):
            TrajectoryEnvelope(**{**base, **overrides})
