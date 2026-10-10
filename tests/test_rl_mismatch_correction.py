"""Mismatch correction specs, translation, rejections and capability gates
(change rl-algo-mismatch-correction, tasks 2.4, 3.1-3.4, 4.1-4.3, 5.3, 6.1, 6.2).

Pure yeto CPU tests (no Miles). Numeric checks against Miles sources live in
tests/test_rl_mismatch_observe.py (miles-next-venv).
"""

from __future__ import annotations

import dataclasses
import json
import math
from types import SimpleNamespace

import pytest

from yeto.rl.algos import mismatch_correction as mc
from yeto.rl.engine.algorithm import AlgorithmSpec, AlgorithmSpecError, PluginRef
from yeto.rl.engine.capabilities import CapabilityMismatch
from yeto.rl.engine.driver import TrainStepMetrics
from yeto.rl.engine.fake import fake_capabilities
from yeto.rl.adapters.miles.algorithm_flags import (
    AlgorithmFlagConflict,
    absorb_extra_argv,
    algorithm_argv,
)
from yeto.rl.adapters.miles.entry import miles_capabilities

FINGERPRINT = "sha256:" + "0" * 64
CHECK = dict(layout="lora", placement="colocated", execution_mode="colocated-serial")
HAS_MASKED_FRACTION = "masked_fraction" in {f.name for f in dataclasses.fields(TrainStepMetrics)}
HAS_OPSM_COMBINATION = False
try:  # 1a-shared.patch (OPSM combinable with tis/custom) applied?
    AlgorithmSpec(correction={"method": "tis", "tis_clip": 2.0, "tis_clip_low": 0.0,
                              "opsm_delta": 1e-4})
    HAS_OPSM_COMBINATION = True
except AlgorithmSpecError:
    pass
HAS_NAMED_CUSTOM = hasattr(__import__("yeto.rl.engine.algorithm", fromlist=["x"]),
                           "register_named_correction_function")


def spec(**correction):
    return AlgorithmSpec(correction=correction)


def tis(high=2.0, low=0.0, **kw):
    return spec(method="tis", tis_clip=high, tis_clip_low=low, **kw)


def icepop(low=0.5, high=5.0, **kw):
    kw.setdefault("mismatch_metrics", True)
    return spec(method="custom", function=mc.icepop_ref().to_dict(),
                tis_clip=high, tis_clip_low=low, **kw)


def opsm(delta=1e-4, source=None, **kw):
    if source is not None:
        kw["opsm_old_logprob_source"] = source
        kw.setdefault("use_rollout_logprobs", source == "rollout")
    return spec(method="opsm", opsm_delta=delta, **kw)


def mis(level="token", mode="truncate", low=None, high=2.0, **kw):
    values = dict(method="custom", function=mc.mis_ref().to_dict(), mis_level=level,
                  mis_mode=mode, mis_upper_bound=high, **kw)
    if low is not None:
        values["mis_lower_bound"] = low
    return spec(**values)


ALL = {
    "mismatch_observe": mc.observe_spec,
    "tis": tis,
    "icepop": icepop,
    "opsm_trainer": opsm,
    "opsm_rollout": lambda: opsm(source="rollout"),
    "mis": mis,
    "mis_mask": lambda: mis(mode="mask", low=0.5),
}


# -- registration / default unchanged (6.2) -----------------------------------


def test_extension_registered_and_default_hash_unchanged():
    from yeto.rl.algos import EXTENSION_MODULES

    assert "yeto.rl.algos.mismatch_correction" in EXTENSION_MODULES
    default = AlgorithmSpec()
    assert default.canonical_json() == (
        '{"advantage_estimator":"grpo","dynamic_sampling_filter":null,'
        '"dynamic_sampling_max_replacements":null,"kl_coef":null,'
        '"loss":"policy_loss","schema":"yeto-rl-algorithm-spec-v1"}'
    )
    assert algorithm_argv(default) == []
    assert default.rejections() == []
    assert not any(d == "corrections" and n != "none" for d, n in default.required_mechanisms())


@pytest.mark.parametrize("name", sorted(ALL))
def test_each_mechanism_is_detected_alone(name):
    s = ALL[name]()
    assert s.rejections() == []
    named = {n for d, n in s.required_mechanisms() if d == "corrections"} & set(ALL)
    assert named == {name}
    assert mc.selected_corrections(s) == (name,)


# -- 2.4 observe-only ---------------------------------------------------------


def test_observe_translation_and_identity():
    s = mc.observe_spec()
    assert algorithm_argv(s) == [
        "--use-tis", "--custom-tis-function-path", mc.OBSERVE_PATH, "--get-mismatch-metrics",
    ]
    ref = PluginRef.from_path(mc.OBSERVE_PATH)
    assert s.correction.function == ref
    assert ref in s.referenced_plugins()
    # The plugin source hash enters the identity.
    other = AlgorithmSpec(correction={"method": "custom", "mismatch_metrics": True,
                                      "function": {"path": mc.OBSERVE_PATH, "sha256": "1" * 64}})
    assert other.sha256() != s.sha256() != AlgorithmSpec().sha256()
    s.verify_plugins(import_callable=True)


@pytest.mark.parametrize(
    "build, needle",
    [
        (lambda: spec(method="custom", function=mc.observe_ref().to_dict()), "mismatch_metrics"),
        (lambda: spec(method="custom", function=mc.observe_ref().to_dict(),
                      mismatch_metrics=True, tis_clip=2.0), "tis_clip"),
        (lambda: spec(method="custom", function=mc.observe_ref().to_dict(),
                      mismatch_metrics=True, use_rollout_logprobs=True),
         "tis_with_rollout_logprobs"),
    ],
)
def test_observe_rejections(build, needle):
    problems = build().rejections()
    assert problems and needle in " ".join(problems)


# -- 3.1 TIS ------------------------------------------------------------------


@pytest.mark.parametrize(
    "build, needle",
    [
        (lambda: spec(method="tis", tis_clip=2.0), "tis_clip_low"),
        (lambda: spec(method="tis", tis_clip_low=0.0), "tis_clip"),
        (lambda: spec(method="tis", tis_clip=float("inf"), tis_clip_low=0.0), "finite"),
        (lambda: spec(method="tis", tis_clip=float("nan"), tis_clip_low=0.0), "finite"),
        (lambda: spec(method="tis", tis_clip=2.0, tis_clip_low=-0.1), ">= 0.0"),
    ],
)
def test_tis_field_errors(build, needle):
    with pytest.raises(AlgorithmSpecError, match=needle):
        build()


@pytest.mark.parametrize("low, high", [(2.0, 2.0), (3.0, 2.0)])
def test_tis_inverted_bounds_rejected(low, high):
    problems = tis(high=high, low=low).rejections()
    assert problems and str(low) in problems[0] and str(high) in problems[0]


def test_tis_translation_explicit_values():
    assert algorithm_argv(tis(high=2.0, low=0.0)) == [
        "--use-tis", "--tis-clip", "2.0", "--tis-clip-low", "0.0",
    ]


def test_tis_with_rollout_logprobs_still_rejected_by_p0():
    problems = tis(use_rollout_logprobs=True).rejections()
    assert any("tis_with_rollout_logprobs" in p for p in problems)


def test_tis_expects_gradient_is_default_grpo():
    s = tis(low=0.0)
    batch = SimpleNamespace(groups=[SimpleNamespace(reward_std=1.0)])
    # TIS masks nothing: even a reported full mask keeps the expectation.
    assert s.expects_gradient(batch, SimpleNamespace(masked_fraction=1.0)) is True


# -- 3.2 IcePop ---------------------------------------------------------------


def test_icepop_translation_and_hash():
    s = icepop()
    assert algorithm_argv(s) == [
        "--use-tis", "--tis-clip", "5.0", "--tis-clip-low", "0.5",
        "--custom-tis-function-path", mc.ICEPOP_PATH, "--get-mismatch-metrics",
    ]
    assert s.correction.function.sha256 == mc.ICEPOP_SOURCE_SHA256
    assert icepop(low=0.4).sha256() != s.sha256()


@pytest.mark.parametrize(
    "build, needle",
    [
        (lambda: spec(method="custom", function=mc.icepop_ref().to_dict(), tis_clip=5.0),
         "tis_clip_low"),
        (lambda: spec(method="custom", function=mc.icepop_ref().to_dict()), "tis_clip"),
        (lambda: icepop(low=5.0, high=0.5), "5.0"),
        (lambda: icepop(low=5.0, high=5.0), "smaller"),
    ],
)
def test_icepop_bounds_rejected(build, needle):
    problems = build().rejections()
    assert problems and needle in " ".join(problems)


# -- 3.3 at most one correction ----------------------------------------------


@pytest.mark.parametrize(
    "argv, flags",
    [
        (["--use-opsm", "--opsm-delta", "1e-4", "--use-tis"], ("--use-tis", "--use-opsm")),
        (["--use-opsm", "--opsm-delta", "1e-4", "--use-tis", "--custom-tis-function-path",
          mc.OBSERVE_PATH], ("--custom-tis-function-path", "--use-opsm")),
    ],
)
def test_two_corrections_in_extra_argv_conflict(argv, flags):
    with pytest.raises(AlgorithmFlagConflict) as info:
        absorb_extra_argv(AlgorithmSpec(), argv)
    for flag in flags:
        assert flag in str(info.value)


def test_tis_plus_custom_function_in_argv_is_one_custom_correction():
    # P0 absorption: --use-tis + --custom-tis-function-path = method custom;
    # observe-only then refuses the TIS clip values.
    spec, _, _ = absorb_extra_argv(AlgorithmSpec(), [
        "--use-tis", "--tis-clip", "2", "--tis-clip-low", "0", "--get-mismatch-metrics",
        "--custom-tis-function-path", mc.OBSERVE_PATH])
    assert "observe-only" in " ".join(spec.rejections())


@pytest.mark.parametrize("base, argv", [
    (tis, ["--use-tis", "--custom-tis-function-path", mc.OBSERVE_PATH]),
    (icepop, ["--use-opsm", "--opsm-delta", "0.1"]),
    (mc.observe_spec, ["--use-opsm", "--opsm-delta", "0.1"]),
])
def test_spec_correction_plus_argv_correction_conflict(base, argv):
    with pytest.raises(AlgorithmFlagConflict, match="correction.method"):
        absorb_extra_argv(base(), argv)


def test_custom_config_path_refused_on_ports():
    from yeto.rl.adapters.miles.config import MilesConfigError, check_extra_argv

    with pytest.raises(MilesConfigError, match="custom-config-path"):
        check_extra_argv(["--custom-config-path", "x.yaml"], AlgorithmSpec())


def test_conflict_fails_before_gpu_process():
    from yeto.rl.adapters.miles.config import MilesConfigError, check_extra_argv

    with pytest.raises(MilesConfigError, match="--use-tis"):
        check_extra_argv(["--use-opsm", "--opsm-delta", "1e-4", "--use-tis"], AlgorithmSpec())


@pytest.mark.skipif(not HAS_OPSM_COMBINATION, reason="needs 1a-shared.patch (OPSM + tis/custom)")
def test_opsm_combines_with_tis_but_not_with_observe():
    combined = tis(opsm_delta=1e-4)
    assert combined.rejections() == []
    assert set(mc.selected_corrections(combined)) == {"tis", "opsm_trainer"}
    assert "--use-opsm" in algorithm_argv(combined) and "--use-tis" in algorithm_argv(combined)
    observe = AlgorithmSpec(correction={"method": "custom", "mismatch_metrics": True,
                                        "function": mc.observe_ref().to_dict(),
                                        "opsm_delta": 1e-4})
    assert "OPSM" in " ".join(observe.rejections())


# -- 3.4 / 4.3 / 5.3 zero-gradient rule ----------------------------------------

BATCH = SimpleNamespace(groups=[SimpleNamespace(reward_std=0.5)])


@pytest.mark.parametrize("name", sorted(ALL))
def test_expects_gradient_per_mechanism(name):
    s = ALL[name]()
    masking = name in mc.MASKING_MECHANISMS
    assert s.expects_gradient(BATCH, SimpleNamespace(masked_fraction=1.0)) is (not masking)
    assert s.expects_gradient(BATCH, SimpleNamespace(masked_fraction=0.99)) is True
    assert s.expects_gradient(BATCH, SimpleNamespace(masked_fraction=None)) is True
    assert s.expects_gradient(BATCH, None) is True


def test_masked_fraction_from_miles_metrics():
    assert mc.masked_fraction_from_metrics(icepop(), {"train/tis_clipfrac": 1.0}) == 1.0
    assert mc.masked_fraction_from_metrics(icepop(), {"tis_clipfrac": 0.25}) == 0.25
    assert mc.masked_fraction_from_metrics(icepop(), {}) is None
    assert mc.masked_fraction_from_metrics(icepop(), {"tis_clipfrac": math.nan}) is None
    m = mis(mode="mask", low=0.5)
    assert mc.masked_fraction_from_metrics(
        m, {"mis_tis_mask_fraction_low": 0.25, "mis_tis_mask_fraction_high": 0.75}) == 1.0
    assert mc.masked_fraction_from_metrics(m, {"mis_tis_mask_fraction_low": 0.25}) is None
    # OPSM reports no token fraction; TIS / observe mask nothing.
    assert mc.masked_fraction_from_metrics(opsm(), {"opsm_clipfrac": 1.0}) is None
    assert mc.masked_fraction_from_metrics(tis(), {"tis_clipfrac": 1.0}) is None


# -- 4.1 / 4.2 OPSM -----------------------------------------------------------


def test_opsm_source_identity():
    trainer, rollout = opsm(), opsm(source="rollout")
    assert trainer.correction.opsm_old_logprob_source == "trainer"
    assert opsm(source="trainer").sha256() == trainer.sha256()
    assert trainer.sha256() != rollout.sha256()
    assert json.loads(rollout.canonical_json())["correction"]["opsm_old_logprob_source"] == "rollout"
    # Not selecting OPSM: the field (at its default) does not affect the hash.
    assert spec(opsm_old_logprob_source="trainer").sha256() == AlgorithmSpec().sha256()


def test_opsm_requires_delta_and_source_requires_opsm():
    with pytest.raises(AlgorithmSpecError, match="opsm_delta"):
        spec(method="opsm")
    problems = spec(opsm_old_logprob_source="rollout", use_rollout_logprobs=True).rejections()
    assert any("requires OPSM" in p for p in problems)
    with pytest.raises(AlgorithmSpecError, match="opsm_old_logprob_source"):
        spec(method="opsm", opsm_delta=0.1, opsm_old_logprob_source="behavior")


def test_opsm_translation():
    assert algorithm_argv(opsm(delta=1e-4)) == ["--use-opsm", "--opsm-delta", "0.0001"]
    assert algorithm_argv(opsm(delta=1e-4, source="rollout")) == [
        "--use-rollout-logprobs", "--use-opsm", "--opsm-delta", "0.0001",
    ]


@pytest.mark.parametrize("source, flag", [("rollout", False), ("trainer", True)])
def test_opsm_source_must_match_rollout_logprobs(source, flag):
    problems = opsm(source=source, use_rollout_logprobs=flag).rejections()
    assert problems and "PPO ratio" in problems[0]


@pytest.mark.skipif(not HAS_OPSM_COMBINATION, reason="needs 1a-shared.patch (OPSM + tis/custom)")
def test_opsm_rollout_with_tis_rejected_with_pi_old_note():
    s = tis(opsm_delta=1e-4, opsm_old_logprob_source="rollout", use_rollout_logprobs=True)
    text = " ".join(s.rejections())
    assert "PPO ratio" in text and "tis_with_rollout_logprobs" in text


# -- 5.3 MIS ------------------------------------------------------------------


def test_mis_translation():
    s = mis(level="geometric", mode="mask", low=0.999, high=1.001)
    assert algorithm_argv(s) == ["--use-tis", "--custom-tis-function-path", mc.MIS_PATH]
    attrs = s.to_legacy_runtime_attrs()
    assert {k: v for k, v in attrs.items() if not k.startswith("yeto_rl_")} == {
        "rs_level": "geometric", "rs_lower_bound": None, "rs_upper_bound": None,
        "rs_veto_threshold": None, "tis_batch_normalize": False, "tis_level": "geometric",
        "tis_lower_bound": 0.999, "tis_mode": "mask", "tis_upper_bound": 1.001, "use_rs": False,
    }
    assert s.correction.function.sha256 == PluginRef.from_path(mc.MIS_PATH).sha256
    assert mis(mode="clip", low=0.5).sha256() != mis(mode="mask", low=0.5).sha256()


def test_runtime_attrs_unchanged_without_mis():
    for build in (AlgorithmSpec, tis, icepop, mc.observe_spec, opsm):
        assert build().to_legacy_runtime_attrs() == AlgorithmSpec().to_legacy_runtime_attrs()


@pytest.mark.parametrize(
    "build, needle",
    [
        (lambda: spec(method="custom", function=mc.mis_ref().to_dict()), "mis_level is required"),
        (lambda: mis(high=None), "mis_upper_bound is required"),
        (lambda: mis(mode="clip"), "mis_lower_bound is required"),
        (lambda: mis(mode="mask"), "mis_lower_bound is required"),
        (lambda: mis(mode="truncate", low=0.5), "unused"),
        (lambda: mis(mode="clip", low=3.0, high=2.0), "smaller"),
        (lambda: mis(level="geometric", mis_batch_normalize=True), "token|sequence"),
        (lambda: mis(tis_clip=2.0), "tis_clip"),
        (lambda: spec(mis_mode="mask"), "require the MIS function"),
        (lambda: tis(mis_upper_bound=2.0), "require the MIS function"),
    ],
)
def test_mis_rejections(build, needle):
    problems = build().rejections()
    assert problems and needle in " ".join(problems)


@pytest.mark.parametrize("field, value", [("mis_mode", "reject"), ("mis_level", "seq"),
                                          ("mis_upper_bound", 0.0), ("mis_lower_bound", math.inf)])
def test_mis_field_errors(field, value):
    with pytest.raises(AlgorithmSpecError, match=f"correction.{field}"):
        spec(**{field: value})


# -- 6.1 capability declaration ---------------------------------------------------


def test_fake_declaration_admits_every_mechanism():
    caps = fake_capabilities(
        corrections={"none", "tis", "opsm", "custom", *ALL},
        features={"mismatch_metrics", "rollout_logprobs_as_old"},
    )
    for name, build in ALL.items():
        caps.check(**CHECK, algorithm=build())


MILES_DECLARED = {"none", "tis", "opsm", "opsm_trainer", "opsm_rollout", "mismatch_observe", "icepop", "mis_mask", "mis"}  # opsm_rollout: rl-algo-supplement 4.6 on 64b591a4b; G1 + effect evidence (7.3, integ-decl; mis 2026-10-08-mis-trigger)
# mismatch_metrics under a use_tis correction is claimed by that correction
# (P0 CORRECTION_COMPANIONS, alignment §7b), so icepop/observe are accepted.
ACCEPTED_BY_MILES = {"tis", "opsm_trainer", "opsm_rollout", "mis_mask", "mis", "icepop", "mismatch_observe"}


def test_miles_adapter_declares_exactly_g1_passed_corrections():
    assert miles_capabilities(FINGERPRINT).corrections == MILES_DECLARED


@pytest.mark.parametrize("name", sorted(ALL))
def test_miles_adapter_accepts_declared_rejects_others(name):
    caps = miles_capabilities(FINGERPRINT)
    if name in ACCEPTED_BY_MILES:
        caps.check(**CHECK, algorithm=ALL[name]())
        return
    with pytest.raises(CapabilityMismatch) as info:
        caps.check(**CHECK, algorithm=ALL[name]())
    text = str(info.value)
    if name in MILES_DECLARED:  # declared, refused on the undeclared mismatch_metrics
        assert "features mechanism 'mismatch_metrics' not supported" in text
    else:
        assert "not supported" in text and f"supported: {sorted(MILES_DECLARED)}" in text


def test_unverified_allowance_admits_single_island_smoke():
    caps = miles_capabilities(
        FINGERPRINT,
        unverified_mechanisms=("corrections:custom", "corrections:icepop",
                               "features:mismatch_metrics"),
    )
    caps.check(**CHECK, algorithm=icepop())


@pytest.mark.skipif(not HAS_MASKED_FRACTION, reason="needs p0-driver.patch "
                    "(TrainStepMetrics.masked_fraction, _check_gradient -> expects_gradient)")
@pytest.mark.parametrize("name", sorted(mc.MASKING_MECHANISMS))
def test_fake_driver_full_mask_round_is_not_a_failure(tmp_path, name):
    import torch

    from yeto.rl.core import StrictRlInvariantError
    from yeto.rl.engine.bridges import LocalOnlySync
    from yeto.rl.engine.driver import EventTape, IslandDriver
    from yeto.rl.engine.fake import FakeEngine

    caps = fake_capabilities(corrections={"none", "tis", "opsm", "custom", *ALL},
                             features={"mismatch_metrics", "rollout_logprobs_as_old"})

    def run(**kw):
        engine = FakeEngine(tensors={"base_model.model.layer.lora_A.weight": torch.zeros(1, 2)},
                            zero_grad_rounds={1}, **kw)
        return IslandDriver(
            learner_id=0, rollout=engine.rollout, trainer=engine.trainer,
            policy_state=engine.policy_state, publisher=engine.publisher,
            placement=engine.placement, algorithm=ALL[name](), sync=LocalOnlySync(3),
            events=EventTape(tmp_path / f"{len(kw)}.jsonl", 0), capabilities=caps,
        ).run()

    assert run(masked_fraction_rounds={1: 1.0}).policy_version == 3
    for fraction in ({1: 0.5}, {}):
        with pytest.raises(StrictRlInvariantError, match="grad_norm 0"):
            run(masked_fraction_rounds=fraction)


@pytest.mark.skipif(not HAS_NAMED_CUSTOM, reason="needs 1a-shared.patch (named custom functions)")
@pytest.mark.parametrize("name", ["mismatch_observe", "mis", "mis_mask"])
def test_named_function_does_not_require_generic_custom(name):
    pairs = ALL[name]().required_mechanisms()
    assert ("corrections", "custom") not in pairs and ("corrections", name) in pairs
    other = AlgorithmSpec(correction={"method": "custom", "function": {
        "path": "yeto.rl.engine.algorithm.plugin_source_sha256", "sha256": "2" * 64}})
    assert ("corrections", "custom") in other.required_mechanisms()


def test_icepop_is_claimed_by_its_own_mechanism_not_generic_custom():
    # integ-decl: P0 accepts miles. built-ins as named correction functions,
    # so IcePop is declared as ("corrections", "icepop") alone.
    pairs = icepop().required_mechanisms()
    assert ("corrections", "icepop") in pairs
    assert ("corrections", "custom") not in pairs


@pytest.mark.skipif(not HAS_OPSM_COMBINATION, reason="needs 1a-shared.patch (always_emit)")
def test_opsm_source_always_in_canonical_json():
    assert json.loads(opsm().canonical_json())["correction"]["opsm_old_logprob_source"] == "trainer"
    assert "opsm_old_logprob_source" not in json.loads(tis().canonical_json())["correction"]


# -- 6.3 documentation examples through the P0 dry run --------------------------


def _doc_examples():
    import pathlib
    import re

    text = (pathlib.Path(__file__).resolve().parents[1] / "docs/MILES_RL.md").read_text()
    pattern = re.compile(r"<!-- mismatch-example allow=([\w:,]+) -->\n```json\n(.*?)```", re.S)
    return [(allow.split(","), body) for allow, body in pattern.findall(text)]


def test_doc_has_every_example():
    assert len(_doc_examples()) == 6


@pytest.mark.parametrize("index", range(6))
def test_doc_example_dry_run(tmp_path, index):
    from yeto.rl.adapters.miles.algorithm_flags import dry_run

    allow, body = _doc_examples()[index]
    path = tmp_path / "spec.json"
    path.write_text(body)
    spec = AlgorithmSpec.from_json_file(str(path))
    assert spec.rejections() == []
    plain = dry_run(["--dry-run", "--rl-algorithm-spec", str(path)])
    name = {0: "mismatch_observe", 1: "tis", 2: "icepop", 3: "opsm_trainer", 4: "opsm_rollout",
            5: "mis_mask"}[index]
    assert name in mc.selected_corrections(spec)
    if name in ACCEPTED_BY_MILES:
        assert plain["verdict"] == "accepted", plain.get("error")
        assert plain["miles_argv"][2:] == algorithm_argv(spec)
    else:
        assert plain["verdict"] == "rejected" and "not supported" in plain["error"]
    argv = ["--dry-run", "--rl-algorithm-spec", str(path)]
    for name in allow:
        argv += ["--rl-allow-unverified-mechanism", name]
    accepted = dry_run(argv)
    assert accepted["verdict"] == "accepted", accepted.get("error")
    assert accepted["algorithm_spec_sha256"] == spec.sha256()
    assert accepted["miles_argv"][2:] == algorithm_argv(spec)


def test_doc_examples_pin_current_plugin_sources(tmp_path):
    for _, body in _doc_examples():
        path = tmp_path / "s.json"
        path.write_text(body)
        spec = AlgorithmSpec.from_json_file(str(path))
        for ref in spec.referenced_plugins():
            if ref.path.startswith("yeto."):
                ref.verify(import_callable=False)
            else:
                assert ref.sha256 == mc.ICEPOP_SOURCE_SHA256
