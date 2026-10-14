"""P0-S: plain engine weight versions, the yeto-side label map and the
rollout/training overlap wiring (Miles fully_async)."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import pytest

from yeto.rl.engine.policy_versions import (
    PolicyVersionConflict,
    PublishedPolicies,
    engine_version_matches,
    engine_version_number,
    engine_weight_version,
)

H5, H6 = "5" * 64, "6" * 64


def test_engine_versions_are_plain_numbers_and_old_labels_still_parse():
    assert engine_weight_version(7) == "7"
    with pytest.raises(ValueError):
        engine_weight_version(-1)
    assert engine_version_number("7") == 7
    assert engine_version_number(f"yeto:7:{H5}") == 7
    assert engine_version_number("default") is None
    assert engine_version_matches("7", f"yeto:7:{H5}")
    assert not engine_version_matches("6", f"yeto:7:{H5}")
    assert engine_version_matches(f"yeto:7:{H5}", f"yeto:7:{H5}")
    assert not engine_version_matches(f"yeto:7:{H6}", f"yeto:7:{H5}")


def test_published_policies_label_engine_versions_and_refuse_reused_versions():
    published = PublishedPolicies()
    published.record(5, H5)
    published.record(6, H6)
    published.record(6, H6)  # a re-publication of the same weights is fine
    with pytest.raises(PolicyVersionConflict, match="v6"):
        published.record(6, H5)
    assert published.label_of_engine_version("5") == f"yeto:5:{H5}"
    assert published.label_of_engine_version(f"yeto:6:{H6}") == f"yeto:6:{H6}"
    assert published.label_of_engine_version(f"yeto:6:{H5}") is None  # other weights
    assert published.label_of_engine_version("4") is None  # never published
    assert published.label_of_engine_version("mixed:5|6") is None
    published.retain_from(6)
    assert published.hashes() == {6: H6}
    assert PublishedPolicies.from_dict(published.to_dict()).hashes() == {6: H6}


def _numeric_driver(tmp_path, limit=0, version_of=lambda rid: str(rid)):
    """A fake island whose groups report the engine's plain weight version."""
    from tests.test_rl_policy_age import _driver

    driver = _driver(tmp_path, limit)
    original = driver.rollout.generate

    def generate(rid, **kw):
        handle = original(rid, **kw)
        groups = tuple(dataclasses.replace(g, policy_token=version_of(rid)) for g in handle.groups)
        return dataclasses.replace(handle, groups=groups)

    driver.rollout.generate = generate
    return driver


def test_driver_names_plain_engine_versions_by_their_published_label(tmp_path):
    _numeric_driver(tmp_path).run()
    events = [json.loads(line) for line in (tmp_path / "e.jsonl").read_text().splitlines()]
    assert len([e for e in events if e["event"] == "rl_round_trained"]) == 3


def test_driver_refuses_an_engine_version_no_publication_recorded(tmp_path):
    from yeto.rl.engine.driver import PolicyIdentityError

    with pytest.raises(PolicyIdentityError, match="not sampled from"):
        _numeric_driver(tmp_path, version_of=lambda rid: "99").run()


def test_driver_hands_the_window_to_the_rollout(tmp_path):
    driver = _numeric_driver(tmp_path, limit=1)
    seen = []
    driver.rollout.set_published_policies = lambda published: seen.append(published.hashes())
    for version, policy_hash in ((5, H5), (6, H6), (7, "7" * 64)):
        driver._record_published_policy(version, policy_hash)
    assert seen[-1] == {6: H6, 7: "7" * 64}  # limit 1: versions 6 and 7 stay


def _published_state(driver, version):
    state = driver.export_local()
    return dataclasses.replace(state, policy_version=version)


def test_driver_refuses_a_reused_version_before_publishing(tmp_path):
    from yeto.rl.engine.driver import PublicationError

    driver = _numeric_driver(tmp_path)
    state = _published_state(driver, 7)
    driver.published_policies.record(7, H5)
    calls = []
    driver.publisher = SimpleNamespace(publish=lambda s: calls.append(s))
    with pytest.raises(PublicationError, match="v7"):
        driver.publish(state, rollout_id=7)
    assert calls == []  # no engine was touched


def test_failed_publication_leaves_the_published_map_unchanged(tmp_path):
    driver = _numeric_driver(tmp_path)
    state = _published_state(driver, 7)

    def fail(_state):
        raise RuntimeError("broadcast failed")

    driver.publisher = SimpleNamespace(publish=fail)
    with pytest.raises(RuntimeError, match="broadcast failed"):
        driver.publish(state, rollout_id=7)
    assert driver.published_policies.hashes() == {}


# ---------------------------------------------------------------- Miles side


def _tolerant():
    from yeto.rl.engine.algorithm import AlgorithmSpec, CorrectionSpec, ExecutionSpec

    return AlgorithmSpec(execution=ExecutionSpec(max_policy_staleness=1),
                         correction=CorrectionSpec(method="tis", tis_clip=2.0, tis_clip_low=0.0))


def test_partitioned_limit_overlaps_rollout_through_miles_fully_async():
    from tests.test_rl_miles_adapter_config import make_config, sub
    from yeto.rl.adapters.miles import config as mc
    from yeto.rl.adapters.miles.policy_age import policy_age_argv

    assert policy_age_argv(1, overlap=True) == ("--fully-async", "--max-weight-staleness", "1")
    partitioned = sub(make_config(colocated=False), "serving", offload_train=False)
    strict = mc.translate_run_config(partitioned, _tolerant())
    overlap = mc.translate_run_config(partitioned, _tolerant(), max_policy_age=1)
    assert not strict.rollout_overlap and overlap.rollout_overlap
    assert [a for a in overlap.argv if a not in strict.argv] == [
        "--fully-async", "--max-weight-staleness"]  # "1" already occurs in the argv
    for flag in ("--rollout-all-samples-process-path", "--buffer-filter-path", "--partial-rollout"):
        assert flag not in overlap.argv
    assert "--rollout-sample-filter-path" in overlap.argv
    # colocated keeps the carry-over at the cut-off (no overlap)
    colocated = mc.translate_run_config(make_config(), _tolerant(), max_policy_age=1)
    assert not colocated.rollout_overlap and "--partial-rollout" in colocated.argv


def test_parsed_fully_async_is_accepted_only_for_an_overlapped_launch():
    from tests.test_rl_miles_adapter_config import make_config, sub
    from yeto.rl.adapters.miles import config as mc

    partitioned = sub(make_config(colocated=False), "serving", offload_train=False)
    overlap = mc.translate_run_config(partitioned, _tolerant(), max_policy_age=1)
    strict = mc.translate_run_config(partitioned, _tolerant())
    ns = dict(use_session_server=False, partial_rollout=False, indep_dp=False)
    with pytest.raises(mc.MilesConfigError, match="fault-tolerance"):
        mc.validate_parsed_args(SimpleNamespace(**ns, fully_async=True), strict,
                                num_cells=lambda _a: 1)
    with pytest.raises(mc.MilesConfigError, match="did not enable fully_async"):
        mc.validate_parsed_args(SimpleNamespace(**ns, fully_async=False), overlap,
                                num_cells=lambda _a: 1)
    # accepted: validation moves on to the placement checks (which need the
    # full parsed namespace this fixture does not build)
    with pytest.raises(AttributeError, match="actor_num_nodes"):
        mc.validate_parsed_args(SimpleNamespace(**ns, fully_async=True), overlap,
                                num_cells=lambda _a: 1)


def test_fully_async_trained_groups_hook_reports_the_round(monkeypatch):
    from yeto.rl.adapters.miles import rollout_meta_hook as hook

    monkeypatch.setattr("yeto.rl.algos.sample_filters.apply_sample_filters", lambda *_a: None)
    from yeto.rl.adapters.miles import fully_async_prompts

    ledger = SimpleNamespace(marked=[], mark_trained=lambda data: ledger.marked.append(data),
                             take_round_counts=lambda: {"submitted_groups": 1})
    monkeypatch.setattr(fully_async_prompts, "_LEDGER", ledger)
    reported = []
    monkeypatch.setattr(hook, "extract_rollout_metadata",
                        lambda args, data, round_counts: reported.append((args, data, round_counts)))
    data = [[SimpleNamespace(group_index=0, index=0, tokens=[1], response_length=1,
                             loss_mask=[1], metadata={}, status=None)]]
    args = SimpleNamespace(fully_async=True)
    hook.record_trained_groups(args, data)
    assert reported == [(args, data, {"submitted_groups": 1})] and ledger.marked == [data]
    hook.record_trained_groups(SimpleNamespace(), data)
    assert len(reported) == 1  # the all-samples hook reports otherwise


def test_rollout_side_matches_plain_versions_against_the_published_label():
    from yeto.rl.adapters.miles import carry_over
    from yeto.rl.adapters.miles.rollout_meta_hook import group_matches_policy
    from yeto.rl.harness.codex.codex_openenv_generate import window_problem

    def sample(version):
        span = SimpleNamespace(version=version, abs_start=1, abs_end=2)
        return SimpleNamespace(status=SimpleNamespace(value="completed"), tokens=[0, 1],
                               response_length=1, weight_versions=[SimpleNamespace(spans=[span])])

    label = f"yeto:6:{H6}"
    assert group_matches_policy([sample("6"), sample("6")], label, 2)
    assert not group_matches_policy([sample("6"), sample("5")], label, 2)
    assert carry_over.response_token_versions(sample("5")) == [5]
    assert window_problem(["5", "6"], label, 1) is None
    assert "behind" in window_problem(["4"], label, 1)
