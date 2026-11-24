"""A live agentic trial served by two versions (an SGLang in-place switch
mid-trial, compacted or not) reached the full-parameter trajectory evidence with no
version span and stopped the island ("a trained trajectory has tokens of unknown
policy version") after the batch was taken. Now its tokens are named by its oldest
version, and a group whose tokens are still of unknown version is left out of the
round before it is trained instead of stopping the island."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from yeto.rl.adapters.miles import buffer_rollout as br
from yeto.rl.adapters.miles.carry_over import sample_versions
from yeto.rl.adapters.miles.full_parameter.evidence import _oldest_published_policy
from yeto.rl.engine.buffer_payload import miles_sample_dict, unknown_version_tokens


def _sample(index, *, version="4", final=None, compaction=None, prompt=3, response=4, spans=None):
    metadata = {"weight_version": version, "agent_turns": 3, "trajectory_id": f"t{index}"}
    if final is not None:
        metadata["final_weight_version"] = final
    if compaction is not None:
        metadata["compaction"] = compaction
    s = {"group_index": 0, "index": index, "tokens": list(range(prompt + response)),
         "response_length": response, "loss_mask": [1] * response,
         "rollout_log_probs": [-0.5] * response, "reward": 1.0, "metadata": metadata}
    if spans is not None:
        s["weight_versions"] = spans
    return s


def _as_miles(d):
    calls = [SimpleNamespace(spans=[SimpleNamespace(**sp) for sp in c["spans"]])
             for c in d.get("weight_versions") or ()]
    return SimpleNamespace(tokens=d["tokens"], response_length=d["response_length"], weight_versions=calls)


@pytest.mark.parametrize("compaction", [None, {"segment_index": 0, "num_segments": 4}])
def test_a_trial_served_by_two_versions_is_named_by_its_oldest(compaction):
    out = miles_sample_dict(_sample(0, version="4", final="6", compaction=compaction))
    assert out["weight_versions"][0]["spans"] == [{"version": "4", "abs_start": 3, "abs_end": 7}]
    assert unknown_version_tokens(out) == 0
    assert sample_versions(_as_miles(out)) == [4]


def test_the_evidence_names_the_oldest_published_policy_of_such_a_trial():
    published = SimpleNamespace(hash_of=lambda v: {4: "h4", 6: "h6"}.get(v))
    behavior = _oldest_published_policy(published)
    segments = [_as_miles(miles_sample_dict(_sample(0, version="4", final="6"))),
                _as_miles(miles_sample_dict(_sample(1, version="6", final="6")))]
    assert behavior(segments) == (4, "h4")


def test_an_unknown_final_version_leaves_the_tokens_unknown():
    out = miles_sample_dict(_sample(0, version="4", final="yeto:x"))
    assert "weight_versions" not in out and unknown_version_tokens(out) == 4


def test_a_partial_span_counts_the_uncovered_tokens():
    spans = [{"spans": [{"version": "5", "abs_start": 3, "abs_end": 5}], "prefill_spans": [], "output_start": 3}]
    assert unknown_version_tokens(miles_sample_dict(_sample(0, spans=spans))) == 2


def test_a_full_parameter_island_leaves_out_a_group_of_unknown_version_tokens(monkeypatch):
    trips = []
    monkeypatch.setattr(br.trainer_breakers, "trip", lambda name, emit, **fields: trips.append((name, fields)))
    good = [_sample(0), _sample(1, version="4", final="6")]
    bad = [_sample(2), _sample(3, version="4", final="yeto:x")]
    stats: dict[str, int] = {}
    kept, problems = br.miles_groups([(10, good), (11, bad)], samples_per_group=2, stats=stats,
                                     require_known_versions=True)
    assert [position for position, _ in kept] == [10]
    assert len(problems) == 1 and "unknown policy version" in problems[0] and "buffer position 11" in problems[0]
    assert stats["unknown_version_groups"] == 1
    assert stats["cross_version_samples"] == 1  # the good group's v4 -> v6 trial (the bad one has no number)
    assert trips and trips[0][0] == br.trainer_breakers.SAMPLE_ALIGNMENT


def test_other_trainers_keep_such_a_group_as_before(monkeypatch):
    monkeypatch.setattr(br.trainer_breakers, "trip", lambda *a, **k: None)
    bad = [_sample(2), _sample(3, version="4", final="yeto:x")]
    kept, problems = br.miles_groups([(11, bad)], samples_per_group=2)
    assert [position for position, _ in kept] == [11] and problems == []


def test_cross_version_tells_a_sample_served_by_two_versions():
    from yeto.rl.engine.buffer_payload import cross_version

    assert cross_version(_sample(0, version="4", final="6"))
    assert not cross_version(_sample(0, version="4", final="4"))
    assert not cross_version(_sample(0, version="4"))
    assert not cross_version(_sample(0, version="4", final="yeto:x"))


def test_the_reader_exports_the_new_counts_only_when_there_are_some():
    from pathlib import Path

    import yeto.rl.adapters.miles as miles_adapter
    source = (Path(miles_adapter.__file__).parent / "buffer_rollout_miles.py").read_text()  # imports Miles
    assert '("cross_version_samples", "unknown_version_groups")' in source and "if stats.get(key)" in source
