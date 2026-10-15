"""rl-offpolicy-correction-passthrough, verl side (L6.4, L6.6, L6.11 CPU part).

CPU only: no verl install, no Ray. The parity tests (L6.6) load the correction
functions straight from the verl fork and the Miles checkout on this machine
and skip when either is absent.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from yeto.rl.adapters.verl import fully_async_round as far
from yeto.rl.adapters.verl import offpolicy_correction as voc
from yeto.rl.adapters.verl.fully_async_translate import VersionMap
from yeto.rl.engine import offpolicy_correction as oc
from yeto.rl.engine.algorithm import AlgorithmSpec, CorrectionSpec, ExecutionSpec

K = voc.PREFIX
DECOUPLED = AlgorithmSpec(execution=ExecutionSpec(max_policy_staleness=2),
                          correction=CorrectionSpec(method="tis", lower=0.0, upper=2.0))


def _spec(**correction) -> AlgorithmSpec:
    return AlgorithmSpec(correction=CorrectionSpec(**correction))


def _fields(mapping) -> set[str]:
    return {w.field for w in mapping.warnings}


# ------------------------------------------------------------------ L6.4 / L6.7 mapping


def test_decoupled_ppo_writes_all_three_values_over_the_fully_async_defaults():
    m = voc.correction_mapping(DECOUPLED)
    assert m.backend == "verl" and m.prox_anchor == oc.PROX_FIRST_LOCAL_STEP
    assert m.values[K + "bypass_mode"] is False
    assert m.values[K + "rollout_is"] == "token"
    assert m.values[K + "rollout_is_threshold"] == 2.0
    assert {K + "bypass_mode=False", K + "rollout_is=token",
            K + "rollout_is_threshold=2.0", K + "rollout_rs=null"} <= set(m.overrides)
    assert m.warnings == ()


def test_tis_lower_warns_and_runs_without_a_lower_bound():
    m = voc.correction_mapping(_spec(method="tis", lower=0.5, upper=2.0))
    (w,) = m.warnings
    assert (w.field, w.backend, w.ran_with) == ("lower", "verl", 0.0)
    assert m.values[K + "rollout_is_threshold"] == 2.0


def test_icepop_maps_to_a_lower_upper_string_without_denominator_warning():
    m = voc.correction_mapping(_spec(method="tis", lower=0.5, upper=5.0, ratio_mask="icepop"))
    assert m.values[K + "rollout_is_threshold"] == "0.5_5.0"
    assert K + "rollout_is_threshold='0.5_5.0'" in m.overrides
    assert m.warnings == ()
    zero = voc.correction_mapping(_spec(method="tis", lower=0.0, upper=5.0, ratio_mask="icepop"))
    assert _fields(zero) == {"lower"} and zero.warnings[0].ran_with == voc.ICEPOP_MIN_LOWER


def test_mis_maps_to_rejection_sampling_and_warns_on_the_denominator():
    m = voc.correction_mapping(_spec(method="tis", lower=0.5, upper=2.0, ratio_mask="mis"))
    assert m.values[K + "rollout_rs"] == "token_k1"
    assert m.values[K + "rollout_rs_threshold"] == "0.5_2.0"
    assert m.values[K + "rollout_is"] is None
    assert _fields(m) == {"ratio_mask"}
    geo = voc.correction_mapping(_spec(method="tis", lower=0.5, upper=2.0, ratio_mask="mis",
                                       level="geometric"))
    assert geo.values[K + "rollout_rs"] == "seq_mean_k1" and _fields(geo) == {"ratio_mask"}
    seq = voc.correction_mapping(_spec(method="tis", lower=0.5, upper=2.0, ratio_mask="mis",
                                       level="sequence"))
    assert _fields(seq) == {"ratio_mask", "level"}


def test_unmappable_fields_warn():
    assert _fields(voc.correction_mapping(_spec(method="tis", lower=0.0, upper=2.0, veto=1e-4))) == {"veto"}
    assert _fields(voc.correction_mapping(_spec(method="opsm", opsm_delta=0.1))) == {"method"}
    norm = voc.correction_mapping(_spec(method="tis", lower=0.0, upper=2.0, batch_normalize=True))
    assert norm.values[K + "rollout_is_batch_normalize"] is True and norm.warnings == ()
    seq = voc.correction_mapping(_spec(method="tis", lower=0.0, upper=2.0, level="sequence"))
    assert seq.values[K + "rollout_is"] == "sequence" and seq.warnings == ()


def test_bypass_mode_true_is_refused_with_the_setting_named():
    with pytest.raises(oc.CorrectionConfigError, match="bypass_mode=True"):
        voc.refuse_forbidden_overrides([K + "bypass_mode=true"])
    m = voc.correction_mapping(_spec(method="tis", lower=0.0, upper=2.0, use_rollout_logprobs=True))
    with pytest.raises(oc.CorrectionConfigError, match="pi_prox"):
        voc.apply_to_overrides([], m)
    voc.refuse_forbidden_overrides([K + "bypass_mode=False"])


def test_apply_replaces_every_rollout_correction_entry_once():
    out = voc.apply_to_overrides(["a=1", K + "bypass_mode=True", K + "rollout_is=sequence",
                                  "+" + K + "rollout_is_batch_normalize=True"],
                                 voc.correction_mapping(DECOUPLED))
    keys = [o.split("=", 1)[0] for o in out]
    assert len(keys) == len(set(keys)) and "a=1" in out
    assert K + "bypass_mode=False" in out and K + "rollout_is=token" in out


def test_backend_role_feeds_the_run_manifest_and_the_cross_island_warning():
    assert oc.backend_mapping(DECOUPLED, "verl").prox_anchor == oc.PROX_FIRST_LOCAL_STEP
    manifest = oc.run_correction_manifest(DECOUPLED, {"i0": "miles", "i1": "verl"})
    assert manifest["islands"]["i1"]["mapped"][K + "bypass_mode"] is False
    (cross,) = [w for w in manifest["warnings"] if w.get("scope") == "cross_island"]
    assert cross["differences"]["prox_anchor"] == {"i0": oc.PROX_PER_STEP,
                                                   "i1": oc.PROX_FIRST_LOCAL_STEP}


# ------------------------------------------------------------------ L6.11 fully_async path


@pytest.mark.parametrize("limit", [2, 4])
def test_fully_async_plan_runs_the_correction_mapping(limit, tmp_path):
    import json

    from yeto.rl.adapters.verl.island_entry import append_correction_record, fully_async_plan

    import yeto.rl.adapters.verl.config as vconf

    run = vconf.VerlRunConfig(model_path="/m", train_file="/t", val_file="/v", out_dir="/o",
                              groups_per_round=8, samples_per_group=4)
    spec = AlgorithmSpec(execution=ExecutionSpec(max_policy_staleness=limit),
                         correction=CorrectionSpec(method="tis", lower=0.5, upper=2.0))
    tape = tmp_path / "tape.jsonl"
    args = SimpleNamespace(groups_per_round=8, samples_per_group=4, global_rounds=3,
                           event_tape=str(tape), learner_id=0)
    out, keys = fully_async_plan(limit, vconf.build_overrides(run), spec.to_dict(), args, "none")
    append_correction_record(args, spec.to_dict())
    append_correction_record(args, AlgorithmSpec().to_dict())  # default: tape unchanged
    # D8: the island tape records the group, the verl values and prox_anchor
    events = [json.loads(line) for line in tape.read_text().splitlines()]
    assert events[0]["event"] == oc.CORRECTION_RECORD_EVENT and events[0]["backend"] == "verl"
    assert events[0]["prox_anchor"] == oc.PROX_FIRST_LOCAL_STEP
    assert events[0]["group"]["upper"] == 2.0
    assert [(e["event"], e["field"]) for e in events[1:]] == [(oc.CORRECTION_CONFIG_WARNING, "lower")]
    assert f"async_training.staleness_threshold={float(limit)}" in out
    assert "async_training.partial_rollout=True" in out
    assert [o for o in out if o.startswith(K + "bypass_mode=")] == [K + "bypass_mode=False"]
    assert K + "rollout_is=token" in out and K + "rollout_is_threshold=2.0" in out
    assert K + "bypass_mode" in keys


def _vmap(n: int) -> VersionMap:
    vm = VersionMap()
    for v in range(n):
        vm.record(v, v)
    return vm


def _meta(uid, lo, hi, lengths):
    n = len(lengths)
    return far.sample_meta({"uid": [uid] * n, "min_global_steps": [lo] * n,
                            "max_global_steps": [hi] * n}, lengths)


def test_round_collection_drops_whole_groups_over_a_hard_and_feeds_the_gate():
    rc = far.RoundCollector(_vmap(6), current_outer=5, limit=2, required=2)
    assert rc.offer(_meta("fresh", 5, 5, [10, 20]))
    assert not rc.offer(_meta("old", 2, 5, [300, 5000]))  # oldest v2: gap 3 > A_hard 2
    assert not rc.offer(_meta("unk", None, None, [7, 7]))
    assert rc.offer(_meta("edge", 3, 4, [1, 1]))  # gap 2 == A_hard: kept
    fields = rc.carry_over_fields()
    assert fields["over_age_discarded_groups"] == 1
    # the driver builds offpolicy_gate from these keys and removes them
    gate = oc.gate_from_carry_over(fields, kept=len(rc.kept), backend="verl")
    assert oc.GATE_DROPPED_LENGTHS not in fields and oc.GATE_GRANULARITY not in fields
    assert gate["backend"] == "verl" and gate["granularity"] == oc.GRANULARITY_TRAJECTORY
    assert gate["counts"] == {oc.GATE_KEPT: 2, oc.GATE_GROUP_DROPPED: 1,
                              oc.GATE_UNKNOWN_VERSION_DROPPED: 1}
    assert gate["dropped_length_histogram"][oc.GATE_GROUP_DROPPED] == \
        oc.length_histogram([300, 5000])
    assert gate["truncation_by_turn"] is None


# ------------------------------------------------------------------ L6.6 CPU parity


VERL_SRC = Path(os.environ.get("YETO_VERL_SRC", "/home/michael/work/verl-fork-acad9875"))
MILES_SRC = Path(os.environ.get("YETO_MILES_SRC", "/home/michael/work/miles-next"))
VERL_HELPER = VERL_SRC / "verl/trainer/ppo/rollout_corr_helper.py"
MILES_CORRECTIONS = MILES_SRC / "miles/backends/training_utils/loss_hub/corrections.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def engines():
    if not (VERL_HELPER.is_file() and MILES_CORRECTIONS.is_file()):
        pytest.skip("verl fork or Miles checkout not on this machine")
    # rollout_corr_helper imports verl_F (masked_sum / masked_mean: copied below
    # from verl/utils/torch_functional.py at acad9875), DataProto and two config
    # classes (type hints only). Stubs replace the heavy package imports.
    stub = types.ModuleType("verl.utils.torch_functional")

    def masked_sum(values, mask, axis=None):
        return (torch.where(mask.bool(), values, 0.0) * mask).sum(axis=axis)

    def masked_mean(values, mask, axis=None):
        return masked_sum(values, mask, axis) / (mask.sum(axis=axis) + 1e-8)

    stub.masked_sum, stub.masked_mean = masked_sum, masked_mean

    class DataProto:
        @classmethod
        def from_dict(cls, tensors):
            obj = cls()
            obj.batch = tensors
            return obj

    names = {"verl": types.ModuleType("verl"), "verl.utils": types.ModuleType("verl.utils"),
             "verl.utils.torch_functional": stub,
             "verl.protocol": types.SimpleNamespace(DataProto=DataProto),
             "verl.trainer.config.algorithm": types.SimpleNamespace(RolloutCorrectionConfig=object),
             "verl.workers.config.actor": types.SimpleNamespace(PolicyLossConfig=object)}
    names["verl"].utils = names["verl.utils"]
    names["verl.utils"].torch_functional = stub
    saved = {k: sys.modules.get(k) for k in names}
    sys.modules.update(names)
    try:
        verl = _load("_yeto_parity_verl_rollout_corr", VERL_HELPER)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    miles = _load("_yeto_parity_miles_corrections", MILES_CORRECTIONS)
    return verl, miles


def _logprobs(seed: int = 3):
    g = torch.Generator().manual_seed(seed)
    rollout = -torch.rand(4, 16, generator=g, dtype=torch.float64) * 3
    train = rollout + torch.randn(4, 16, generator=g, dtype=torch.float64) * 0.8
    mask = torch.ones(4, 16, dtype=torch.float64)
    mask[1, 12:] = 0
    mask[3, 5:] = 0
    return train, rollout, mask


def _miles_weights(miles, fn, train, rollout, mask, lower, upper):
    rows = [m.bool() for m in mask]
    flat_t = [t[r] for t, r in zip(train, rows)]
    flat_r = [t[r] for t, r in zip(rollout, rows)]
    n = sum(int(r.sum()) for r in rows)
    args = SimpleNamespace(tis_clip=upper, tis_clip_low=lower)
    pg, masks, _ = getattr(miles, fn)(args, pg_loss=torch.ones(n, dtype=torch.float64),
                                      train_log_probs=flat_t, rollout_log_probs=flat_r,
                                      loss_masks=[r.double() for r in rows])
    return pg, masks


def _verl(verl, train, rollout, mask, **kw):
    weights, new_mask, _ = verl.compute_rollout_correction_and_rejection_mask(
        old_log_prob=train, rollout_log_prob=rollout, response_mask=mask, **kw)
    return (None if weights is None else weights.batch["rollout_is_weights"][mask.bool()]), new_mask


def test_parity_tis_with_lower_zero_matches(engines):
    verl, miles = engines
    train, rollout, mask = _logprobs()
    miles_w, _ = _miles_weights(miles, "vanilla_tis_function", train, rollout, mask, 0.0, 2.0)
    verl_w, new_mask = _verl(verl, train, rollout, mask, rollout_is="token", rollout_is_threshold=2.0)
    torch.testing.assert_close(verl_w, miles_w)
    assert torch.equal(new_mask, mask)


def test_parity_tis_lower_differs_as_warned(engines):
    verl, miles = engines
    train, rollout, mask = _logprobs()
    miles_w, _ = _miles_weights(miles, "vanilla_tis_function", train, rollout, mask, 0.5, 2.0)
    verl_w, _ = _verl(verl, train, rollout, mask, rollout_is="token", rollout_is_threshold=2.0)
    low = miles_w > verl_w
    assert low.any() and torch.all(miles_w[low] == 0.5)
    torch.testing.assert_close(verl_w[~low], miles_w[~low])


def test_parity_icepop_matches_and_keeps_the_denominator(engines):
    verl, miles = engines
    train, rollout, mask = _logprobs()
    miles_w, miles_masks = _miles_weights(miles, "icepop_function", train, rollout, mask, 0.5, 2.0)
    verl_w, new_mask = _verl(verl, train, rollout, mask, rollout_is="token",
                             rollout_is_threshold="0.5_2.0")
    torch.testing.assert_close(verl_w, miles_w)
    assert (verl_w == 0).any()
    assert torch.equal(new_mask, mask)  # verl IcePop zeroes weights; loss_mask unchanged
    assert sum(int(m.sum()) for m in miles_masks) == int(mask.sum())


def test_verl_rejection_sampling_shrinks_the_denominator(engines):
    verl, _ = engines
    train, rollout, mask = _logprobs()
    _, new_mask = _verl(verl, train, rollout, mask, rollout_rs="token_k1",
                        rollout_rs_threshold="0.5_2.0")
    assert 0 < new_mask.sum() < mask.sum()
    ratio = torch.exp(train - rollout)
    kept = mask.bool() & (ratio >= 0.5) & (ratio <= 2.0)
    assert torch.equal(new_mask.bool(), kept)


def test_verl_use_rollout_log_probs_is_not_read_by_any_code():
    """L6.6: ``actor.use_rollout_log_probs`` (fully_async yaml: True) has no
    reader in the fork, so with ``bypass_mode=False`` pi_prox is recomputed."""
    if not VERL_SRC.is_dir():
        pytest.skip("verl fork not on this machine")
    readers = [p for p in (VERL_SRC / "verl").rglob("*.py")
               if "use_rollout_log_probs" in p.read_text(errors="ignore")
               and p.name != "actor.py"]
    assert readers == []


# ------------------------------------------------------------------ example configs


@pytest.mark.parametrize("name,a_hard", [("decoupled-ppo-c2", 2), ("decoupled-ppo-c2-agentic", 4)])
def test_decoupled_ppo_examples_map_on_both_backends(name, a_hard):
    from yeto.rl.adapters.miles.offpolicy_correction import admission_argv
    from yeto.rl.adapters.miles.offpolicy_correction import correction_mapping as miles_mapping
    from yeto.rl.adapters.verl.policy_age import policy_age_overrides

    examples = Path(__file__).resolve().parents[1] / "examples" / "rl_algorithms"
    spec = AlgorithmSpec.from_json_file(str(examples / f"{name}.json"))
    assert spec.rejections() == [] and spec.execution.max_policy_staleness == a_hard
    assert miles_mapping(spec).argv == ("--use-tis", "--tis-clip", "2.0", "--tis-clip-low", "0.0")
    assert admission_argv(a_hard) == ("--max-weight-staleness", str(a_hard))
    verl = voc.correction_mapping(spec)
    assert verl.values[K + "rollout_is_threshold"] == 2.0 and verl.warnings == ()
    assert f"async_training.staleness_threshold={float(a_hard)}" in \
        policy_age_overrides(a_hard, groups_per_round=8)
