"""Score centering (--loss-type score_centering) through the algorithm spec and Miles argv."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from yeto.rl.adapters.miles import algorithm_flags as af
from yeto.rl.adapters.miles import config as mc
from yeto.rl.algos import score_centering as sc
from yeto.rl.engine.algorithm import AlgorithmSpec, AlgorithmSpecError, mechanism_names

from test_rl_algorithm_flags import make_config  # noqa: E402  (shared R0 configuration)

# INTERFACES 2.5: the Miles argv of --rl-loss score_centering (plus --advantage-estimator grpo in R0 position)
EXPECTED_FRAGMENT = [
    "--calculate-per-token-loss", "--disable-grpo-std-normalization", "--use-rollout-logprobs",
    "--loss-type", "score_centering",
    "--rollout-top-logprobs-num", "128", "--score-centering-is", "none",
    "--rollout-temperature", "1.0", "--rollout-top-p", "1.0", "--rollout-top-k", "-1",
]


def test_variant_and_mechanism_are_registered():
    assert "losses:score_centering" in mechanism_names()
    spec = sc.score_centering_spec()
    assert spec.loss.variant == "score_centering"
    assert ("losses", "score_centering") in spec.required_mechanisms()
    assert spec.rejections() == []
    assert not spec.is_v1_expressible()


def test_spec_fields_enter_the_hash_only_when_selected():
    default = AlgorithmSpec()
    assert "score_centering_is" not in default.loss.to_dict()
    spec = sc.score_centering_spec()
    loss = spec.loss.to_dict()
    assert loss["score_centering_is"] == "none" and loss["rollout_top_logprobs_num"] == 128
    assert loss["score_centering_tis_clip"] == 2.0
    assert sc.score_centering_spec(importance_sampling="tis").sha256() != spec.sha256()
    assert sc.score_centering_spec(top_logprobs=64).sha256() != spec.sha256()
    # the default GRPO identity is untouched by the extension
    assert default.sha256() == AlgorithmSpec.from_dict(default.to_dict()).sha256()
    assert default.to_dict()["schema"] == AlgorithmSpec().to_dict()["schema"]


def test_argv_snapshot_none_tis_mis():
    assert af.algorithm_argv(sc.score_centering_spec()) == EXPECTED_FRAGMENT
    tis = af.algorithm_argv(sc.score_centering_spec(importance_sampling="tis", tis_clip=3.0))
    assert tis[:9] == EXPECTED_FRAGMENT[:7] + ["--score-centering-is", "tis"]
    assert tis[9:11] == ["--score-centering-tis-clip", "3.0"]
    assert tis[11:] == list(sc.SAMPLING_ARGV)
    mis = af.algorithm_argv(sc.score_centering_spec(importance_sampling="mis", mis_low=0.25, mis_high=4.0))
    assert mis[9:13] == ["--score-centering-mis-low", "0.25", "--score-centering-mis-high", "4.0"]
    for argv in (tis, mis):
        assert "--use-tis" not in argv and "--use-opsm" not in argv


def test_translate_run_config_argv_snapshot_and_default_unchanged():
    base = mc.translate_run_config(make_config(), AlgorithmSpec()).argv
    launch = mc.translate_run_config(make_config(), sc.score_centering_spec())
    argv = list(launch.argv)
    assert argv.count("--loss-type") == 1 and argv[argv.index("--loss-type") + 1] == "score_centering"
    assert argv[argv.index("--advantage-estimator") + 1] == "grpo"
    for flag in ("--rollout-top-logprobs-num", "--score-centering-is", "--use-rollout-logprobs",
                 "--disable-grpo-std-normalization", "--calculate-per-token-loss", "--rollout-temperature"):
        assert argv.count(flag) == 1, flag
    # every token beyond the default GRPO argv is the score-centering fragment
    extra = list(argv)
    for token in base:
        extra.remove(token)
    assert extra == EXPECTED_FRAGMENT
    # default path: byte-identical (the golden test covers the recorded samples)
    assert mc.translate_run_config(make_config(), AlgorithmSpec()).argv == tuple(base)


def test_extra_argv_absorbs_and_round_trips():
    spec, remaining, absorbed = af.absorb_extra_argv(
        sc.score_centering_spec(), ["--score-centering-is", "none", "--rollout-top-logprobs-num", "128", "--lr", "1"])
    assert remaining == ("--lr", "1")
    assert absorbed == {"--score-centering-is": "none", "--rollout-top-logprobs-num": "128"}
    assert spec == sc.score_centering_spec()
    with pytest.raises(af.AlgorithmFlagConflict):
        af.absorb_extra_argv(sc.score_centering_spec(importance_sampling="tis"), ["--score-centering-is", "mis"])
    # the upstream recipe given as raw argv on a default spec is absorbed into the same spec
    recipe = ["--loss-type", "score_centering", "--rollout-top-logprobs-num", "128", "--score-centering-is", "none",
              "--use-rollout-logprobs", "--disable-grpo-std-normalization", "--calculate-per-token-loss"]
    absorbed_spec, _, _ = af.absorb_extra_argv(AlgorithmSpec(), recipe)
    assert absorbed_spec == sc.score_centering_spec()


def test_flags_are_adapter_owned_and_mapped():
    for flag in sc.FLAGS.values():
        assert flag in af.mapped_flags() and flag in af.EXTENSION_FLAGS
        assert flag in mc.ADAPTER_OWNED_FLAGS
    with pytest.raises(mc.MilesConfigError, match="--rollout-temperature"):
        mc.check_extra_argv(["--rollout-temperature", "1.0"])  # derived from the loss, never passed raw


@pytest.mark.parametrize(("payload", "match"), [
    (dict(advantage=dict(estimator="gspo")), "needs 'grpo'"),
    (dict(advantage=dict(std_normalization=True)), "std_normalization=True"),
    (dict(loss=dict(aggregation="default")), "needs 'token'"),
    (dict(correction=dict(use_rollout_logprobs=False)), "use_rollout_logprobs=False"),
    (dict(correction=dict(method="opsm", opsm_delta=0.2)), "centered importance weights replace"),
    (dict(correction=dict(method="tis", upper=2.0, lower=0.5, use_rollout_logprobs=False)),
     "centered importance weights replace|needs True"),
    (dict(loss=dict(eps_clip=0.2)), "have no effect"),
    (dict(loss=dict(eps_clip_c=3.0)), "have no effect"),
    (dict(loss=dict(policy_loss_variant="cispo", eps_clip=0.2, eps_clip_high=0.3)), "have no effect|replaces"),
    (dict(loss=dict(score_centering_tis_clip=3.0)), "need loss.score_centering_is='tis'"),
    (dict(loss=dict(score_centering_mis_low=0.1)), "need loss.score_centering_is='mis'"),
    (dict(loss=dict(score_centering_is="mis", score_centering_mis_low=5.0, score_centering_mis_high=5.0)),
     "must be below"),
])
def test_rejections(payload, match):
    base = sc.score_centering_spec().structured_dict()
    for group, values in payload.items():
        base.setdefault(group, {}).update(values)
    spec = AlgorithmSpec.from_dict(base)
    problems = spec.rejections()
    assert problems, payload
    assert any(__import__("re").search(match, p) for p in problems), problems


def test_params_without_variant_rejected_and_bad_values_refused():
    spec = AlgorithmSpec.from_dict({**AlgorithmSpec().structured_dict(), "loss": {"score_centering_is": "tis"}})
    assert any("only apply to loss.variant='score_centering'" in p for p in spec.rejections())
    with pytest.raises(AlgorithmSpecError, match="one of"):
        sc.score_centering_spec(importance_sampling="truncate")
    with pytest.raises(AlgorithmSpecError, match="positive"):
        sc.score_centering_spec(tis_clip=0.0)
    with pytest.raises(AlgorithmSpecError, match="positive integer"):
        sc.score_centering_spec(top_logprobs=0)


def test_launcher_helper_from_rl_loss_args():
    base = AlgorithmSpec()
    assert sc.spec_from_rl_loss_args(SimpleNamespace(), base) is base
    assert sc.spec_from_rl_loss_args(SimpleNamespace(rl_loss="policy_loss"), base) is base
    spec = sc.spec_from_rl_loss_args(SimpleNamespace(rl_loss="score_centering", rl_score_centering_is="tis",
                                                     rl_score_centering_tis_clip=1.5))
    assert spec.loss.score_centering_is == "tis" and spec.loss.score_centering_tis_clip == 1.5
    assert spec.rejections() == []
    with pytest.raises(AlgorithmSpecError, match="needs --rl-loss score_centering"):
        sc.spec_from_rl_loss_args(SimpleNamespace(rl_loss="policy_loss", rl_score_centering_is="tis"))
    with pytest.raises(AlgorithmSpecError, match="--rl-loss must be one of"):
        sc.spec_from_rl_loss_args(SimpleNamespace(rl_loss="sft"))


def test_dry_run_reports_the_unverified_mechanism(tmp_path):
    path = tmp_path / "sc.json"
    path.write_text(sc.score_centering_spec().canonical_json())
    result = af.dry_run(["--dry-run", "--rl-algorithm-spec", str(path)])
    assert "losses:score_centering" in result["required_mechanisms"]
    assert result["verdict"] == "rejected"  # not declared by the Miles adapter until GPU-verified
    allowed = af.dry_run(["--dry-run", "--rl-algorithm-spec", str(path),
                          "--rl-allow-unverified-mechanism", "losses:score_centering",
                          "--rl-allow-unverified-mechanism", "features:rollout_logprobs_as_old",
                          "--rl-allow-unverified-mechanism", "features:no_grpo_std_normalization",
                          "--rl-allow-unverified-mechanism", "loss_aggregations:token"])
    assert allowed["verdict"] == "accepted", allowed.get("error")
    assert allowed["miles_argv"][:2] == ["--advantage-estimator", "grpo"]
    assert allowed["miles_argv"][2:] == EXPECTED_FRAGMENT


def test_rollout_logprobs_switch_is_allowed_only_under_the_loss():
    from yeto.rl.adapters.miles import offpolicy_correction as mo
    from yeto.rl.engine.offpolicy_correction import CorrectionConfigError

    argv = af.algorithm_argv(sc.score_centering_spec())
    mo.refuse_forbidden_argv(argv)  # the loss has no PPO clip anchor to lose
    mo.refuse_forbidden_argv(["--loss-type=score_centering", "--use-rollout-logprobs"])
    with pytest.raises(CorrectionConfigError, match="--use-rollout-logprobs"):
        mo.refuse_forbidden_argv([t for t in argv if t not in ("--loss-type", "score_centering")])
    with pytest.raises(CorrectionConfigError, match="--skip-actor-forward-only"):
        mo.refuse_forbidden_argv(argv + ["--skip-actor-forward-only"])
