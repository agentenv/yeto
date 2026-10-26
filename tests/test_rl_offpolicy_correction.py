"""rl-offpolicy-correction-passthrough (CPU): neutral correction fields,
hash (L6b.1), Miles mapping, decoupled PPO refusals, refusals (L6b.2),
offpolicy_gate, A_hard >= 2, start-version gap and observe-only bounds."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from yeto.rl.adapters.miles import carry_over
from yeto.rl.adapters.miles import offpolicy_correction as mo
from yeto.rl.adapters.miles.algorithm_flags import algorithm_argv
from yeto.rl.adapters.miles.config import MilesConfigError, check_extra_argv
from yeto.rl.algos import mismatch_correction as mc
from yeto.rl.engine import offpolicy_correction as oc
from yeto.rl.engine.algorithm import AlgorithmSpec, AlgorithmSpecError, CorrectionSpec
from yeto.rl.engine.island_ledger import StalenessPolicy
from yeto.rl.engine.policy_age import PolicyAgeError
from yeto.rl.engine.version_segments import TokenProvenance


def decoupled(upper: float = 2.0, **kw) -> AlgorithmSpec:
    """Decoupled PPO with C = upper (the example-config value is 2.0)."""
    return AlgorithmSpec(correction=CorrectionSpec(method="tis", lower=0.0, upper=upper, **kw))


# -- L6.1 neutral fields ------------------------------------------------------


def test_fields_written_in_full_with_defaults():
    group = CorrectionSpec(method="tis", lower=0.0, upper=2.0).to_dict()
    for name in ("method", "level", "lower", "upper", "veto", "batch_normalize", "ratio_mask"):
        assert name in group
    assert (group["level"], group["ratio_mask"], group["veto"], group["batch_normalize"]) == (
        "token", "none", None, False)


def test_old_bound_names_are_aliases():
    old = CorrectionSpec(method="tis", tis_clip=2.0, tis_clip_low=0.1)
    assert old == CorrectionSpec(method="tis", upper=2.0, lower=0.1)
    assert (old.tis_clip, old.tis_clip_low) == (2.0, 0.1)
    assert CorrectionSpec.from_dict({"method": "tis", "tis_clip": 2, "tis_clip_low": 0}) == \
        CorrectionSpec(method="tis", upper=2.0, lower=0.0)
    with pytest.raises(AlgorithmSpecError, match="old name of correction.upper"):
        CorrectionSpec(method="tis", tis_clip=2.0, upper=3.0, lower=0.0)


@pytest.mark.parametrize("kw, needle", [
    (dict(method="tis", upper=2.0), "correction.upper and correction.lower"),
    (dict(method="tis", lower=0.0), "correction.upper and correction.lower"),
    (dict(ratio_mask="icepop"), "requires correction.method='tis'"),
    (dict(method="tis", lower=0.0, upper=2.0, ratio_mask="bogus"), "correction.ratio_mask"),
    (dict(method="tis", lower=0.0, upper=2.0, level="turn"), "correction.level"),
    (dict(method="tis", lower=0.0, upper=2.0, level="geometric"), "ratio_mask='mis'"),
    (dict(veto=1e-4), "correction.veto"),
    (dict(batch_normalize=True), "correction.batch_normalize"),
    (dict(level="sequence"), "correction.level"),
    (dict(method="tis", lower=0.0, upper=2.0, veto=0.0), "correction.veto"),
    (dict(method="tis", lower=0.0, upper=2.0, batch_normalize=1), "correction.batch_normalize"),
])
def test_invalid_specs(kw, needle):
    with pytest.raises(AlgorithmSpecError, match=needle.replace(".", r"\.").replace("(", r"\(")):
        CorrectionSpec(**kw)


def test_valid_specs():
    CorrectionSpec(method="tis", lower=0.5, upper=5.0, ratio_mask="icepop")
    CorrectionSpec(method="tis", lower=0.5, upper=2.0, ratio_mask="mis", level="geometric",
                   veto=1e-4, batch_normalize=True)
    assert not decoupled().rejections()
    assert any("smaller" in p for p in
               AlgorithmSpec(correction=CorrectionSpec(method="tis", lower=3.0, upper=2.0))
               .rejections())


# -- L6.2 / L6b.1 hash and records -------------------------------------------


@pytest.mark.parametrize("correction", [
    CorrectionSpec(method="tis", lower=0.0, upper=2.0),
    CorrectionSpec(method="tis", lower=0.0, upper=5.0),
    CorrectionSpec(method="tis", lower=0.5, upper=5.0, ratio_mask="icepop"),
    CorrectionSpec(method="tis", lower=0.5, upper=2.0, ratio_mask="mis", veto=1e-4),
    CorrectionSpec(method="opsm", opsm_delta=1e-4),
])
def test_correction_group_changes_the_hash(correction):
    # L6b.1: the correction group is in the hash
    base = AlgorithmSpec(entropy_coef=0.01)
    spec = base.replace(correction=correction)
    assert spec.sha256() != base.sha256()
    assert AlgorithmSpec.from_dict(json.loads(spec.canonical_json())) == spec


def test_each_correction_field_changes_the_hash():
    tis = AlgorithmSpec(correction=CorrectionSpec(method="tis", lower=0.0, upper=2.0))
    variants = [tis.replace(correction=CorrectionSpec(method="tis", lower=0.0, upper=5.0)),
                tis.replace(correction=CorrectionSpec(method="tis", lower=0.5, upper=2.0)),
                tis.replace(correction=CorrectionSpec(method="tis", lower=0.0, upper=2.0,
                                                      level="sequence"))]
    assert len({tis.sha256(), *(v.sha256() for v in variants)}) == 4


def test_default_hashes_are_unchanged():
    # values from before the change (base c4ae38a6)
    assert AlgorithmSpec().sha256() == \
        "27df1133c924e7a337e90246c7e1e30699017f461e9f35b1dd165a0263e65bea"
    assert AlgorithmSpec(loss={"variant": "policy_loss", "eps_clip": 0.3}).sha256() == \
        "64f4876cb7a99d2650731844dfc7da29dbbe4b638dc3837f5916cf4e98c4f2f4"
    assert AlgorithmSpec(entropy_coef=0.1).sha256() == \
        "e5a86b389f80e46d9e36bf5942687fc9d9a14106aac34159c680952724251e82"


def test_custom_function_source_hash_stays_in_the_identity():
    ref = mc.icepop_ref()
    spec = AlgorithmSpec(correction={"method": "custom", "function": ref.to_dict(),
                                     "tis_clip": 5.0, "tis_clip_low": 0.5})
    other = AlgorithmSpec(correction={"method": "custom", "tis_clip": 5.0, "tis_clip_low": 0.5,
                                      "function": {"path": ref.path, "sha256": "1" * 64}})
    assert spec.sha256() != other.sha256() != AlgorithmSpec().sha256()


def test_manifest_section_records_group_mapping_and_anchor():
    spec = decoupled()
    section = oc.run_correction_manifest(spec, {"0": "miles", "1": "miles"})
    record = section["islands"]["0"]
    assert record["group"] == spec.correction.to_dict()
    assert record["mapped"] == {"--use-tis": True, "--tis-clip": "2.0", "--tis-clip-low": "0.0"}
    assert record["prox_anchor"] == "per_step"
    assert section["warnings"] == []


def test_backend_without_mapping_is_recorded_and_warned():
    from yeto.rl.engine import backends

    backends.register(backends.BackendEntry("fake", "x", {}))
    try:
        section = oc.run_correction_manifest(decoupled(), {"0": "fake"})
        assert section["islands"]["0"]["mapped"] is None
        assert [w["backend"] for w in section["warnings"]] == ["fake"]
        # a default group needs no mapping: no warning
        assert oc.run_correction_manifest(AlgorithmSpec(), {"0": "fake"})["warnings"] == []
    finally:
        backends.unregister("fake")


def test_island_tape_gets_record_and_warnings_then_refuses(tmp_path):
    # L6b.2: the tape keeps the record and the warning; the island start is refused
    from yeto.rl.adapters.miles import island_entry

    events = []
    island_entry_append = island_entry._append_ports_event
    island_entry._append_ports_event = lambda _a, _m, e: events.append(e)
    try:
        island_entry._append_correction_record(SimpleNamespace(), SimpleNamespace(),
                                               AlgorithmSpec())
        assert events == []  # default tapes unchanged
        spec = decoupled(level="sequence")
        with pytest.raises(oc.CorrectionConfigError, match="level on backend 'miles'"):
            island_entry._append_correction_record(SimpleNamespace(), SimpleNamespace(), spec)
    finally:
        island_entry._append_ports_event = island_entry_append
    assert events[0]["event"] == oc.CORRECTION_RECORD_EVENT
    assert events[0]["group"]["level"] == "sequence" and events[0]["prox_anchor"] == "per_step"
    assert [(e["event"], e["field"], e["backend"]) for e in events[1:]] == [
        (oc.CORRECTION_CONFIG_WARNING, "level", "miles")]


# -- L6.3 Miles mapping -------------------------------------------------------


def test_miles_tis_row():
    assert algorithm_argv(decoupled()) == ["--use-tis", "--tis-clip", "2.0", "--tis-clip-low", "0.0"]


def test_miles_icepop_row():
    spec = AlgorithmSpec(correction=CorrectionSpec(method="tis", lower=0.5, upper=5.0,
                                                   ratio_mask="icepop"))
    assert algorithm_argv(spec) == ["--use-tis", "--tis-clip", "5.0", "--tis-clip-low", "0.5",
                                    "--custom-tis-function-path", mc.ICEPOP_PATH]
    assert mc.selected_corrections(spec) == ("icepop",)
    assert spec.to_legacy_runtime_attrs() == AlgorithmSpec().to_legacy_runtime_attrs()


def test_miles_mis_row():
    spec = AlgorithmSpec(correction=CorrectionSpec(method="tis", lower=0.5, upper=2.0,
                                                   ratio_mask="mis", level="sequence",
                                                   veto=1e-4, batch_normalize=True))
    assert algorithm_argv(spec) == ["--use-tis", "--tis-clip", "2.0", "--tis-clip-low", "0.5",
                                    "--custom-tis-function-path", mc.MIS_PATH]
    attrs = {k: v for k, v in spec.to_legacy_runtime_attrs().items() if not k.startswith("yeto_")}
    assert attrs == {
        "rs_level": "sequence", "rs_lower_bound": 0.5, "rs_upper_bound": 2.0,
        "rs_veto_threshold": 1e-4, "tis_batch_normalize": True, "tis_level": "sequence",
        "tis_lower_bound": 0.5, "tis_mode": "mask", "tis_upper_bound": 2.0, "use_rs": True}
    assert mc.selected_corrections(spec) == ("mis_mask",)
    assert mo.correction_mapping(spec).warnings == ()


def test_miles_opsm_row():
    spec = AlgorithmSpec(correction=CorrectionSpec(method="opsm", opsm_delta=1e-4))
    assert algorithm_argv(spec) == ["--use-opsm", "--opsm-delta", "0.0001"]


@pytest.mark.parametrize("kw, field", [
    (dict(level="sequence"), "level"),
    (dict(veto=1e-4), "veto"),
    (dict(batch_normalize=True), "batch_normalize"),
])
def test_miles_unmappable_fields_warn(kw, field):
    mapping = mo.correction_mapping(decoupled(**kw))
    assert [w.field for w in mapping.warnings] == [field]
    assert mapping.warnings[0].backend == "miles"
    event = mapping.warnings[0].event()
    assert event["event"] == "correction_config_warning" and "ran_with" in event


# -- L6.7 decoupled PPO: refused Miles flags -----------------------------------


@pytest.mark.parametrize("flag", ["--use-rollout-logprobs", "--skip-actor-forward-only",
                                  "--mask-offpolicy-in-partial-rollout"])
def test_forbidden_miles_flag_is_refused_by_name(flag):
    with pytest.raises(MilesConfigError, match=flag):
        check_extra_argv([flag], decoupled())
    with pytest.raises(oc.CorrectionConfigError, match=flag):
        mo.refuse_forbidden_argv(["--use-tis", flag])


def test_rollout_logprobs_from_the_spec_is_refused_in_the_final_argv():
    spec = AlgorithmSpec(correction=CorrectionSpec(use_rollout_logprobs=True))
    assert "--use-rollout-logprobs" in mo.correction_mapping(spec).argv
    with pytest.raises(oc.CorrectionConfigError, match="--use-rollout-logprobs"):
        mo.refuse_forbidden_argv(algorithm_argv(spec))


def test_decoupled_ppo_records_per_step_anchor():
    mapping = mo.correction_mapping(decoupled())
    assert mapping.prox_anchor == oc.PROX_PER_STEP
    assert mapping.argv == ("--use-tis", "--tis-clip", "2.0", "--tis-clip-low", "0.0")


# -- L6.5 cross-island comparison ---------------------------------------------


def _record(upper: float, anchor: str) -> dict:
    mapping = oc.CorrectionMapping(backend="x", prox_anchor=anchor)
    return oc.correction_record(decoupled(upper).correction, mapping, backend="x")


def test_islands_with_different_upper_give_one_warning():
    event = oc.compare_island_records({"0": _record(2.0, "per_step"), "1": _record(5.0, "per_step")})
    assert event["event"] == "correction_config_warning"
    assert event["differences"] == {"upper": {"0": 2.0, "1": 5.0}}


def test_different_prox_anchor_is_listed():
    event = oc.compare_island_records({"miles": _record(2.0, "per_step"),
                                       "verl": _record(2.0, "first_local_step")})
    assert event["differences"] == {"prox_anchor": {"miles": "per_step",
                                                    "verl": "first_local_step"}}
    assert oc.compare_island_records({"0": _record(2.0, "per_step"),
                                      "1": _record(2.0, "per_step")}) is None


def test_launcher_manifest_section():
    from tests.test_rl_launcher import _args
    from yeto.launch.rl_ports import _offpolicy_correction_manifest

    args = _args([])
    section = _offpolicy_correction_manifest(args, decoupled())
    assert "0" in section["islands"] and section["warnings"] == []
    # L6b.2: an unmappable field refuses the launch
    with pytest.raises(ValueError, match="level on backend 'miles'"):
        _offpolicy_correction_manifest(args, decoupled(level="sequence"))


def test_refuse_names_every_warning():
    oc.refuse_correction_warnings([])
    cross = oc.compare_island_records({"0": _record(2.0, "per_step"),
                                       "1": _record(5.0, "per_step")})
    unmappable = oc.CorrectionWarning("veto", "verl", "no mapping").event()
    with pytest.raises(oc.CorrectionConfigError) as exc:
        oc.refuse_correction_warnings([unmappable, cross])
    assert "veto on backend 'verl': no mapping" in str(exc.value)
    assert "islands differ in ['upper']" in str(exc.value)


# -- L6.8 offpolicy_gate --------------------------------------------------------


def test_dropped_length_histogram_fixed_buckets():
    hist = oc.length_histogram([300, 5000])
    assert hist == {"0-255": 0, "256-1023": 1, "1024-4095": 0, "4096-16383": 1, "16384+": 0}


def test_gate_event_granularity():
    event = oc.offpolicy_gate_event(backend="verl", counts={"kept": 3}, dropped_lengths={},
                                    truncation_by_turn={0: (1, 2)}, granularity="trajectory")
    assert event["granularity"] == "trajectory"
    assert event["truncation_by_turn"] == {"0": {"truncated": 1, "total": 2, "ratio": 0.5}}
    with pytest.raises(ValueError):
        oc.offpolicy_gate_event(backend="x", counts={}, dropped_lengths={},
                                truncation_by_turn=None, granularity="turn")


class _Span:
    def __init__(self, version, start, end):
        self.version, self.abs_start, self.abs_end = str(version), start, end


class _Sample:
    def __init__(self, group, spans, length, prompt=3, **kw):
        self.group_index, self.response_length = group, length
        self.tokens = [0] * (prompt + length)
        self.weight_versions = [SimpleNamespace(spans=[_Span(v, prompt + a, prompt + b)
                                                       for v, a, b in spans])]
        self.__dict__.update(kw)


def test_miles_buffer_filter_drops_feed_the_gate():
    args = SimpleNamespace()
    old_a = [_Sample(0, [(1, 0, 300)], 300)]
    old_b = [_Sample(1, [(1, 0, 5000)], 5000)]
    fresh = [_Sample(2, [(4, 0, 10)], 10)]
    kept = carry_over.carry_over_buffer_filter(args, [old_a, old_b, fresh], 4,
                                               current_version=4, limit=2)
    assert kept == [fresh]
    fields = carry_over.take_carry_stats(args, 4)
    gate = oc.gate_from_carry_over(fields, kept=1, backend="miles")
    assert oc.GATE_DROPPED_LENGTHS not in fields
    assert gate["counts"] == {"group_dropped": 2, "kept": 1, "unknown_version_dropped": 0}
    hist = gate["dropped_length_histogram"]["group_dropped"]
    assert (hist["256-1023"], hist["4096-16383"]) == (1, 1)
    assert gate["granularity"] == "token"


def test_miles_truncation_by_turn_is_token_level():
    # two model turns (loss_mask runs) of 2 tokens, separated by one tool token
    sample = _Sample(0, [(3, 0, 5)], 5, loss_mask=[1, 1, 0, 1, 1],
                     rollout_log_probs=[-0.5] * 5)
    current = [0.5, -0.5, -0.5, 0.4, -0.5]  # ratios e^1, 1, (tool), e^0.9, 1
    args = SimpleNamespace(tis_clip=2.0, tis_clip_low=0.0)
    out = carry_over.estimate_cross_version_truncation(args, [sample], 4, lambda _s: current)
    assert out[carry_over.TRUNCATED_BY_TURN] == {0: (1, 2), 1: (1, 2)}


def test_driver_tapes_offpolicy_gate_as_its_own_event():
    from yeto.rl.engine.driver import IslandDriver

    emitted = []
    fake = SimpleNamespace(emit=lambda name, **kw: emitted.append((name, kw)),
                           clock=lambda: 0.0, _labels=lambda: {})
    carry = {"max_policy_age": 2, "over_age_discarded_groups": 1,
             oc.GATE_DROPPED_LENGTHS: {"group_dropped": [300]},
             oc.TRUNCATED_BY_TURN: {0: (1, 2)}}
    batch = SimpleNamespace(carry_over=carry, policy_version=3, groups=(object(),))
    IslandDriver._emit_carry_over(fake, 3, batch)
    assert [name for name, _ in emitted] == [oc.OFFPOLICY_GATE_EVENT, "rl_rollout_carry_over"]
    assert emitted[0][1]["counts"]["kept"] == 1
    assert emitted[0][1]["counts"]["group_dropped"] == 1
    assert emitted[0][1]["granularity"] == "token"
    assert oc.GATE_DROPPED_LENGTHS not in emitted[1][1]
    assert oc.TRUNCATED_BY_TURN not in emitted[1][1]



@pytest.mark.parametrize("backend, label", [("verl", "verl"), ("miles", "miles"),
                                            (None, "unknown")])
def test_driver_labels_offpolicy_gate_with_the_batch_backend(backend, label):
    from yeto.rl.engine.driver import IslandDriver
    from yeto.rl.engine.ports import RolloutBatchHandle

    emitted = []
    fake = SimpleNamespace(emit=lambda name, **kw: emitted.append((name, kw)),
                           clock=lambda: 0.0, _labels=lambda: {})
    batch = RolloutBatchHandle(
        rollout_id=1, policy_version=1, policy_hash="h", groups=(), completed=0, aborted=0,
        carry_over={oc.GATE_DROPPED_LENGTHS: {}, oc.GATE_GRANULARITY: oc.GRANULARITY_TRAJECTORY},
        backend=backend)
    IslandDriver._emit_carry_over(fake, 1, batch)
    assert emitted[0][0] == oc.OFFPOLICY_GATE_EVENT
    assert emitted[0][1]["backend"] == label
    assert emitted[0][1]["granularity"] == oc.GRANULARITY_TRAJECTORY

# -- L6.9 A_hard >= 2 ----------------------------------------------------------


def test_a_hard_two_passes_support_and_maps_to_both_backends():
    from yeto.rl.adapters.miles.policy_age import SUPPORT as MILES, policy_age_argv
    from yeto.rl.adapters.verl.policy_age import SUPPORT as VERL, policy_age_overrides

    MILES.check(2)
    VERL.check(2)
    assert policy_age_argv(2, overlap=True) == ("--fully-async", "--max-weight-staleness", "2")
    assert "async_training.staleness_threshold=2.0" in policy_age_overrides(2, groups_per_round=4)
    with pytest.raises(PolicyAgeError):
        MILES.check(MILES.max_policy_age + 1)


def test_miles_admission_mapping():
    assert mo.admission_argv(0) == ()
    assert mo.admission_argv(2) == ("--max-weight-staleness", "2")
    assert mo.admission_argv(2, admission_limit=1) == (
        "--max-weight-staleness", "2", "--async-data-buffer-capacity-factor", "1")
    with pytest.raises(ValueError, match="k=3"):
        mo.admission_argv(2, admission_limit=3)


# -- L6.12 start version --------------------------------------------------------


def test_continued_trajectory_gap_from_start_version():
    provenance = TokenProvenance((3, 3), (-0.1, -0.1)).extend(4, [-0.2]).extend(5, [-0.3, -0.3])
    assert oc.start_version_gap(6, provenance.versions) == 3
    assert oc.oldest_is_start(provenance.versions) is True
    assert 6 - provenance.oldest == oc.start_version_gap(6, provenance.versions)


def test_tape_segments_check():
    records = [{"policy_versions": [[3, 0, 2], [4, 2, 3], [5, 3, 5]]},
               {"policy_versions": [[5, 0, 2], [4, 2, 3]]},  # not a continuation order
               {"reward": 1.0}]
    assert oc.segments_start_check(records) == {
        "checked": 2, "oldest_is_start": 1, "oldest_before_start": 1}


# -- L6.13 cross_version_is: observe-only, from the spec ---------------------------


def test_cross_version_bounds_follow_the_spec():
    assert oc.cross_version_bounds(decoupled(5.0).correction) == (0.0, 5.0)
    icepop = CorrectionSpec(method="tis", lower=0.5, upper=5.0, ratio_mask="icepop")
    assert oc.cross_version_bounds(icepop) == (0.5, 5.0)
    assert oc.cross_version_bounds(CorrectionSpec()) == oc.OBSERVE_DEFAULT_BOUNDS


def test_staleness_policy_correction_is_derived():
    assert StalenessPolicy.for_algorithm(decoupled()).correction == "tis"
    icepop = AlgorithmSpec(correction=CorrectionSpec(method="tis", lower=0.5, upper=5.0,
                                                     ratio_mask="icepop"))
    assert StalenessPolicy.for_algorithm(icepop).correction == "icepop"
    assert StalenessPolicy.for_algorithm(AlgorithmSpec()).correction == "tis"


def test_miles_rollout_handle_keeps_the_gate_inputs():
    # L6.10 GPU: the Miles handle dropped the gate keys, so the driver taped no offpolicy_gate
    from yeto.rl.adapters.miles.rollout import _carry_over_fields

    payload = {"max_policy_age": 1, "over_age_discarded_groups": 1,
               oc.GATE_DROPPED_LENGTHS: {"group_dropped": [300], "unknown_version_dropped": []},
               oc.TRUNCATED_BY_TURN: {"0": [1, 2]}}
    fields = _carry_over_fields(payload)
    gate = oc.gate_from_carry_over(fields, kept=3, backend="miles")
    assert gate["counts"] == {"group_dropped": 1, "kept": 3, "unknown_version_dropped": 0}
    assert gate["truncation_by_turn"]["0"]["truncated"] == 1
    assert oc.GATE_DROPPED_LENGTHS not in fields


@pytest.mark.parametrize("stale, expected", [(2, 2), (0, 0), (None, None)])
def test_miles_buffer_stale_drops_count_apart_from_the_gate(stale, expected):
    # GOAL §6: Miles' own --fully-async buffer drops get their own reason code
    from yeto.rl.adapters.miles.rollout import _carry_over_fields

    payload = {"max_policy_age": 1, "over_age_discarded_groups": 1, "stale_dropped_groups": stale,
               oc.GATE_DROPPED_LENGTHS: {"group_dropped": [300], "unknown_version_dropped": []}}
    gate = oc.gate_from_carry_over(_carry_over_fields(payload), kept=3, backend="miles")
    assert gate["counts"]["group_dropped"] == 1
    assert gate["counts"].get(oc.GATE_BACKEND_STALE_DROPPED) == expected
    assert oc.GATE_BACKEND_STALE_DROPPED not in gate["dropped_length_histogram"]
