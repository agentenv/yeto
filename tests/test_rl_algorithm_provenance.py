"""User input, island consistency and provenance (rl-algorithm-capabilities 5.1-5.5)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from yeto.rl import learner as rl_learner
from yeto.rl.engine.algorithm import (
    BOUNDED_NONZERO_STD_FILTER,
    AlgorithmSpec,
    AlgorithmSpecError,
    LossSpec,
    resolve_ports_algorithm,
)
from yeto.rl.engine.miles_adapter import config as mc
from yeto.rl.engine.miles_adapter.entry import selection_event
from test_rl_engine_selection import _learner_argv
from test_rl_miles_adapter_config import make_config

V2 = AlgorithmSpec(loss=LossSpec(eps_clip_high=0.28))


def _spec_file(tmp_path, payload, name="spec.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return str(path)


# -- 5.1 --------------------------------------------------------------------------


def test_learner_spec_file_v1_v2_absent(tmp_path):
    v1 = AlgorithmSpec(dynamic_sampling_filter=BOUNDED_NONZERO_STD_FILTER,
                       dynamic_sampling_max_replacements=3)
    for spec in (v1, V2):
        args = rl_learner.parse_args(_learner_argv(
            ("--rl-algorithm-spec", _spec_file(tmp_path, json.loads(spec.canonical_json())))))
        assert resolve_ports_algorithm(args, rl_engine="ports") == spec
    args = rl_learner.parse_args(_learner_argv())
    # absent: identical to R0 (from_legacy_args)
    assert resolve_ports_algorithm(args, rl_engine="ports") == AlgorithmSpec.from_legacy_args(args)
    assert resolve_ports_algorithm(args, rl_engine="ports").sha256() == AlgorithmSpec().sha256()


@pytest.mark.parametrize("extra", [
    ("--rl-algorithm-spec", "x.json"),
    ("--rl-allow-unverified-mechanism", "clip_higher"),
    ("--rl-expected-algorithm-sha256", "a" * 64),
    ("--rl-placement", "fixed-partition"),
])
def test_legacy_with_ports_only_options_refused(extra, capsys):
    with pytest.raises(SystemExit):
        rl_learner.parse_args(_learner_argv(("--rl-engine", "legacy", *extra)))
    assert "only appl" in capsys.readouterr().err


def test_legacy_cli_disagreeing_with_spec_file(tmp_path):
    path = _spec_file(tmp_path, json.loads(V2.canonical_json()))
    args = SimpleNamespace(rl_algorithm_spec=path,
                           dynamic_sampling_filter_path=BOUNDED_NONZERO_STD_FILTER,
                           dynamic_sampling_max_replacements=2)
    with pytest.raises(AlgorithmSpecError, match="disagrees with --rl-algorithm-spec"):
        resolve_ports_algorithm(args, rl_engine="ports")


def test_learner_unverified_allowance_multi_island_refused(capsys):
    with pytest.raises(SystemExit):
        rl_learner.parse_args(_learner_argv(("--rl-allow-unverified-mechanism", "clip_higher",
                                             "--num-learners", "2")))
    assert "single-island" in capsys.readouterr().err
    args = rl_learner.parse_args(_learner_argv(("--rl-allow-unverified-mechanism",
                                                "clip_higher")))
    assert args.rl_allow_unverified_mechanism == ["clip_higher"]


# -- 5.2 learner check ----------------------------------------------------------------


def _launch(spec=V2, extra=()):
    return mc.translate_run_config(make_config(), spec, extra_argv=extra)


def _check(tmp_path, expected, launch=None, allow=None):
    tape = tmp_path / "tape.jsonl"
    args = SimpleNamespace(rl_expected_algorithm_sha256=expected, event_tape=str(tape),
                           learner_id=3, rl_allow_unverified_mechanism=allow)
    miles_args = SimpleNamespace()
    rl_learner.verify_ports_algorithm(args, miles_args, launch or _launch())
    return miles_args, tape


def _tape(tape):
    return [json.loads(x) for x in tape.read_text().splitlines()] if tape.exists() else []


def test_expected_hash_match(tmp_path):
    miles_args, tape = _check(tmp_path, V2.sha256())
    assert _tape(tape) == []
    assert miles_args.yeto_rl_unverified_mechanisms == ()


def test_expected_hash_mismatch_event_and_refusal(tmp_path):
    with pytest.raises(rl_learner.AlgorithmMismatchError, match="refusing to join"):
        _check(tmp_path, AlgorithmSpec().sha256())
    [event] = _tape(tmp_path / "tape.jsonl")
    assert event["event"] == "rl_algorithm_mismatch" and event["island_id"] == 3
    assert event["rl/algorithm_spec_sha256"] == V2.sha256()
    assert event["rl/expected_algorithm_spec_sha256"] == AlgorithmSpec().sha256()


def test_expected_hash_missing_warns(tmp_path):
    _, tape = _check(tmp_path, None)
    [event] = _tape(tape)
    assert event["event"] == "rl_algorithm_expected_hash_missing"
    assert event["level"] == "warning"


def test_learner_checks_before_the_bridge():
    import inspect

    source = inspect.getsource(rl_learner.run_miles)
    assert source.index("verify_ports_algorithm(") < source.index("_run_ports(")
    ports = inspect.getsource(rl_learner._run_ports)
    assert "verify_ports_algorithm" not in ports  # no bridge/sync before it


# -- 5.2 launcher --------------------------------------------------------------------


def _launcher_args(engine, extra=(), gpu="aws:1xa100@us-east-1"):
    from yeto.cli import parse_args

    args = parse_args([
        "--gpu", gpu, "--model", "org/model", "--data", "org/data",
        "--training-mode", "rl", "--total-steps", "3", "--rollout-batch-size", "4",
        "--n-samples-per-prompt", "2", "--rollout-max-response-len", "128",
        "--local-rl-rounds-per-sync", "1", "--reward-function", "pkg.reward:score",
        "--trust-remote-code", "--rl-engine", engine, *extra,
    ])
    args.model_revision = "a" * 40
    args.data_revision = "b" * 40
    args.source_sha256 = "c" * 64
    args.reward_sha256 = "d" * 64
    return args


@pytest.fixture(autouse=True)
def _fake_sky(monkeypatch):
    import sys
    import types

    from test_rl_launcher import _Resources, _Storage, _StorageMode, _Task

    monkeypatch.setitem(sys.modules, "sky", types.SimpleNamespace(
        Task=_Task, Resources=_Resources, Storage=_Storage, StorageMode=_StorageMode))


def _island_run(args):
    from yeto.gpu_spec import parse_gpu_spec
    from yeto.launcher import _prepare_rl_args, make_miles_island_task

    _prepare_rl_args(args)
    return make_miles_island_task(args, parse_gpu_spec(args.gpu)[0], 0, 1, "127.0.0.1:1").run


def test_launcher_sends_expected_hash_only_on_ports(tmp_path):
    run = _island_run(_launcher_args("ports"))
    assert f"--rl-expected-algorithm-sha256 {AlgorithmSpec().sha256()}" in run
    assert "--rl-algorithm-spec" not in run
    legacy = _island_run(_launcher_args("legacy"))
    assert "--rl-expected-algorithm-sha256" not in legacy
    assert "algorithm_spec.json" not in legacy


def test_launcher_ships_spec_file_and_hash(tmp_path):
    path = _spec_file(tmp_path, json.loads(V2.canonical_json()))
    run = _island_run(_launcher_args("ports", ("--rl-algorithm-spec", path)))
    assert f"--rl-expected-algorithm-sha256 {V2.sha256()}" in run
    assert "--rl-algorithm-spec ~/yeto-rl/algorithm_spec.json" in run
    assert V2.canonical_json() in run


def test_launcher_refusals(tmp_path):
    from yeto.launcher import _prepare_rl_args

    two = _launcher_args("ports", ("--rl-allow-unverified-mechanism", "clip_higher"),
                         gpu="aws:1xa100@us-east-1,aws:1xa100@us-west-2")
    with pytest.raises(ValueError, match="single-island"):
        _prepare_rl_args(two)
    rejected = _spec_file(tmp_path, {"advantage_estimator": "grpo", "kl_coef": 0.1})
    with pytest.raises(ValueError, match="placement='loss'"):
        _prepare_rl_args(_launcher_args("ports", ("--rl-algorithm-spec", rejected)))
    with pytest.raises(ValueError, match="only apply to --rl-engine ports"):
        _prepare_rl_args(_launcher_args("legacy", ("--rl-algorithm-spec", rejected)))
    single = _island_run(_launcher_args("ports", ("--rl-allow-unverified-mechanism",
                                                  "clip_higher")))
    assert "--rl-allow-unverified-mechanism clip_higher --num-learners 1" in single


# -- 5.3 export ------------------------------------------------------------------------


def test_export_records_algorithm_like_the_event(tmp_path):
    from test_rl_export import (
        MODEL_REVISION, _model, _write_checkpoint, canonical_layout_hash,
        derive_peft_lora_specs, export_rl_checkpoint,
    )
    import torch

    model_path, _ = _model(tmp_path)
    specs = derive_peft_lora_specs(str(model_path), None, rank=2, targets="all-linear")
    checkpoint = tmp_path / "state.ckpt"
    _write_checkpoint(checkpoint, torch.zeros(sum(s.numel for s in specs)),
                      canonical_layout_hash(specs), ledger_size=0)
    launch = _launch()
    event = selection_event(launch=launch, algorithm=launch.algorithm, miles_commit="x",
                            unverified_mechanisms=("clip_higher",))
    export_rl_checkpoint(checkpoint, tmp_path / "ports", model=str(model_path),
                         model_revision=MODEL_REVISION, rank=2, lora_targets="all-linear",
                         algorithm_spec=event["rl/algorithm_spec"],
                         unverified_mechanisms=event["rl/unverified_mechanisms"])
    prov = json.loads((tmp_path / "ports" / "yeto_rl_provenance.json").read_text())
    assert prov["algorithm_spec"] == event["rl/algorithm_spec"] == V2.canonical_json()
    assert prov["algorithm_spec_sha256"] == event["rl/algorithm_spec_sha256"] == V2.sha256()
    assert prov["rl/unverified_mechanisms"] == ["clip_higher"]
    assert prov["contains_unverified_mechanisms"] is True
    with pytest.raises(ValueError, match="only for rl_engine='ports'"):
        export_rl_checkpoint(checkpoint, tmp_path / "legacy", model=str(model_path),
                             model_revision=MODEL_REVISION, rank=2,
                             lora_targets="all-linear", rl_engine="legacy",
                             algorithm_spec=V2)


def test_export_cli_reads_spec_file(tmp_path, monkeypatch):
    from yeto.rl import export as rl_export

    seen = {}
    monkeypatch.setattr(rl_export, "export_rl_checkpoint",
                        lambda *a, **kw: seen.update(kw) or SimpleNamespace(policy_version=1))
    path = _spec_file(tmp_path, json.loads(V2.canonical_json()))
    rl_export.main(["--checkpoint", "c", "--model", "m", "--model-revision", "a" * 40,
                    "--lora-r", "2", "--output-dir", "o", "--rl-algorithm-spec", path,
                    "--rl-unverified-mechanism", "clip_higher"])
    assert AlgorithmSpec.from_dict(json.loads(seen["algorithm_spec"])) == V2
    assert seen["unverified_mechanisms"] == ["clip_higher"]


# -- 5.4 / 5.5 events ----------------------------------------------------------------


def test_selection_event_records_absorbed_flags_and_allowance():
    launch = _launch(AlgorithmSpec(), ("--eps-clip-high", "0.28", "--use-rollout-logprobs"))
    event = selection_event(launch=launch, algorithm=launch.algorithm, miles_commit="x",
                            unverified_mechanisms=("clip_higher",))
    assert event["rl/algorithm_absorbed_flags"] == {"--eps-clip-high": "0.28",
                                                    "--use-rollout-logprobs": True}
    assert event["rl/unverified_mechanisms"] == ["clip_higher"]
    assert event["rl/algorithm_spec_sha256"] == launch.algorithm.sha256()
    plain = selection_event(launch=_launch(AlgorithmSpec()), algorithm=AlgorithmSpec(),
                            miles_commit="x")
    assert plain["rl/algorithm_absorbed_flags"] == {}
    assert "rl/unverified_mechanisms" not in plain


def test_verify_records_allowance_on_namespace(tmp_path):
    miles_args, _ = _check(tmp_path, V2.sha256(), allow=["clip_higher", "clip_higher"])
    assert miles_args.yeto_rl_unverified_mechanisms == ("clip_higher",)
    assert miles_args.yeto_rl_algorithm_absorbed_flags == {}


def test_launch_check_refuses_translation():
    from yeto.rl.engine import algorithm as alg

    alg.register_launch_check("t_batch", lambda s, run: [f"batch {run['rollout_batch_size']}"]
                              if s.entropy_coef == 0.25 else [])
    try:
        with pytest.raises(mc.MilesConfigError, match=r"for this run: \[t_batch\] batch 4"):
            mc.translate_run_config(make_config(), AlgorithmSpec(entropy_coef=0.25))
        mc.translate_run_config(make_config(), AlgorithmSpec())
    finally:
        alg.unregister(launch_check="t_batch")


def test_island_check_refuses_before_outer_sync(tmp_path):
    from yeto.rl.engine import algorithm as alg

    alg.register_island_check("t_rev", lambda s, isl: [f"rev {isl['base_model_revision']}"])
    try:
        tape = tmp_path / "tape.jsonl"
        args = SimpleNamespace(rl_expected_algorithm_sha256=V2.sha256(), event_tape=str(tape),
                               learner_id=0, rl_allow_unverified_mechanism=None,
                               model_revision="a" * 40)
        with pytest.raises(rl_learner.AlgorithmMismatchError, match="rev a"):
            rl_learner.verify_ports_algorithm(args, SimpleNamespace(), _launch())
        [event] = _tape(tape)
        assert event["event"] == "rl_algorithm_island_rejected"
    finally:
        alg.unregister(island_check="t_rev")


def test_expected_hash_on_miles_args(tmp_path):
    miles_args, _ = _check(tmp_path, V2.sha256().upper())
    assert miles_args.yeto_rl_expected_algorithm_sha256 == V2.sha256()
    miles_args, _ = _check(tmp_path, None)
    assert miles_args.yeto_rl_expected_algorithm_sha256 is None


def test_fixed_partition_passes_selection_on_learner_and_launcher(capsys):
    import inspect

    from yeto import launcher

    args = rl_learner.parse_args(_learner_argv(("--rl-placement", "fixed-partition",
                                                "--rollout-num-gpus", "1")))
    assert args.rl_placement == "fixed-partition"
    with pytest.raises(SystemExit):  # without --rl-placement it is still refused
        rl_learner.parse_args(_learner_argv(("--rollout-num-gpus", "1")))
    assert 'placement=getattr(args, "rl_placement", "colocated")' in inspect.getsource(
        launcher._prepare_rl_args)
