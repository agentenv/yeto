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
# a non-default spec the Miles adapter declares (the launcher checks capabilities)
DECLARED = AlgorithmSpec(dynamic_sampling_filter=BOUNDED_NONZERO_STD_FILTER,
                         dynamic_sampling_max_replacements=2)


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


def test_learner_unverified_allowance_refused_with_outer_sync(capsys):
    # design D11 as written: the learner CLI always joins a syncer (outer
    # sync), so the allowance is refused even for one island (F9/F10).
    for extra in ((), ("--num-learners", "2")):
        with pytest.raises(SystemExit):
            rl_learner.parse_args(_learner_argv(
                ("--rl-allow-unverified-mechanism", "features:clip_higher", *extra)))
        assert "without outer sync" in capsys.readouterr().err
    args = rl_learner.parse_args(_learner_argv())
    rl_learner._check_ports_algorithm_options(
        argparse_ns(args, rl_allow_unverified_mechanism=["features:clip_higher"]),
        outer_sync=False)  # a no-outer-sync single island would be admitted


def argparse_ns(args, **changes):
    values = dict(vars(args))
    values.update(changes)
    return SimpleNamespace(**values)


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
    path = _spec_file(tmp_path, json.loads(DECLARED.canonical_json()))
    run = _island_run(_launcher_args("ports", ("--rl-algorithm-spec", path)))
    assert f"--rl-expected-algorithm-sha256 {DECLARED.sha256()}" in run
    assert "--rl-algorithm-spec ~/yeto-rl/algorithm_spec.json" in run
    assert DECLARED.canonical_json() in run


def test_launcher_refusals(tmp_path):
    from yeto.launcher import _prepare_rl_args

    for gpu in ("aws:1xa100@us-east-1,aws:1xa100@us-west-2", "aws:1xa100@us-east-1"):
        allowed = _launcher_args("ports", ("--rl-allow-unverified-mechanism",
                                           "features:clip_higher"), gpu=gpu)
        with pytest.raises(ValueError, match="without outer sync"):
            _prepare_rl_args(allowed)
    rejected = _spec_file(tmp_path, {"advantage_estimator": "grpo", "kl_coef": 0.1})
    with pytest.raises(ValueError, match="placement='loss'"):
        _prepare_rl_args(_launcher_args("ports", ("--rl-algorithm-spec", rejected)))
    with pytest.raises(ValueError, match="only apply to --rl-engine ports"):
        _prepare_rl_args(_launcher_args("legacy", ("--rl-algorithm-spec", rejected)))
    # F7: undeclared mechanisms and registered launch checks fail before any cloud work
    undeclared = _spec_file(tmp_path, json.loads(V2.canonical_json()), "v2.json")
    with pytest.raises(ValueError, match="'clip_higher' not supported"):
        _prepare_rl_args(_launcher_args("ports", ("--rl-algorithm-spec", undeclared)))
    from yeto.rl.engine import algorithm as alg

    alg.register_launch_check("t_launcher", lambda s, run: [f"batch {run['rollout_batch_size']}"])
    try:
        with pytest.raises(ValueError, match=r"\[t_launcher\] batch 4"):
            _prepare_rl_args(_launcher_args("ports"))
    finally:
        alg.unregister(launch_check="t_launcher")


def test_launcher_hash_equals_learner_hash_without_absorption(tmp_path):
    # F14: same spec source -> same hash; absorption on the island -> refused.
    path = _spec_file(tmp_path, json.loads(DECLARED.canonical_json()))
    args = _launcher_args("ports", ("--rl-algorithm-spec", path))
    from yeto.launcher import _prepare_rl_args

    _prepare_rl_args(args)
    learner_args = rl_learner.parse_args(_learner_argv(("--rl-algorithm-spec", path)))
    base = resolve_ports_algorithm(learner_args, rl_engine="ports")
    launch = mc.translate_run_config(make_config(), base)
    assert launch.algorithm_sha256 == args.rl_expected_algorithm_sha256
    _check(tmp_path, args.rl_expected_algorithm_sha256, launch=launch)
    absorbed = mc.translate_run_config(make_config(), base, extra_argv=("--eps-clip", "0.2"))
    assert absorbed.algorithm_sha256 != args.rl_expected_algorithm_sha256
    with pytest.raises(rl_learner.AlgorithmMismatchError):
        _check(tmp_path, args.rl_expected_algorithm_sha256, launch=absorbed)


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
                            unverified_mechanisms=("features:clip_higher",))
    export_rl_checkpoint(checkpoint, tmp_path / "ports", model=str(model_path),
                         model_revision=MODEL_REVISION, rank=2, lora_targets="all-linear",
                         algorithm_spec=event["rl/algorithm_spec"],
                         unverified_mechanisms=event["rl/unverified_mechanisms"])
    prov = json.loads((tmp_path / "ports" / "yeto_rl_provenance.json").read_text())
    assert prov["algorithm_spec"] == event["rl/algorithm_spec"] == V2.canonical_json()
    assert prov["algorithm_spec_sha256"] == event["rl/algorithm_spec_sha256"] == V2.sha256()
    assert prov["rl/unverified_mechanisms"] == ["features:clip_higher"]
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
                            unverified_mechanisms=("features:clip_higher",))
    assert event["rl/algorithm_absorbed_flags"] == {"--eps-clip-high": "0.28",
                                                    "--use-rollout-logprobs": True}
    assert event["rl/unverified_mechanisms"] == ["features:clip_higher"]
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


# -- single-island no-sync entry (main-agent decision, alignment §7b) -----------------


def _no_sync_argv(extra=()):
    argv = _learner_argv(("--rl-single-island-no-sync", *extra))
    i = argv.index("--syncer")
    del argv[i:i + 2]
    return argv


def test_no_sync_entry_admits_allowance_and_refuses_sync_combinations(capsys):
    args = rl_learner.parse_args(_no_sync_argv(("--rl-allow-unverified-mechanism",
                                                "features:clip_higher")))
    assert args.rl_single_island_no_sync and args.syncer is None
    for extra in (("--syncer", "127.0.0.1:1"), ("--num-learners", "2"),
                  ("--sync-preset", "decoupled"), ("--rl-engine", "legacy")):
        argv = _no_sync_argv(extra) if extra[0] != "--syncer" else \
            _learner_argv(("--rl-single-island-no-sync",))
        with pytest.raises(SystemExit):
            rl_learner.parse_args(argv)
        assert "cannot be combined" in capsys.readouterr().err or extra[0] == "--rl-engine"
    argv = _learner_argv()
    i = argv.index("--syncer")
    del argv[i:i + 2]
    with pytest.raises(SystemExit):
        rl_learner.parse_args(argv)
    assert "--syncer is required" in capsys.readouterr().err


def test_no_sync_marks_event_and_namespace(tmp_path):
    tape = tmp_path / "tape.jsonl"
    args = SimpleNamespace(rl_expected_algorithm_sha256=V2.sha256(), event_tape=str(tape),
                           learner_id=0, rl_allow_unverified_mechanism=["features:clip_higher"],
                           rl_single_island_no_sync=True)
    miles_args = SimpleNamespace()
    rl_learner.verify_ports_algorithm(args, miles_args, _launch())
    assert miles_args.yeto_rl_outer_sync is False
    event = selection_event(launch=_launch(), algorithm=V2, miles_commit="x",
                            unverified_mechanisms=miles_args.yeto_rl_unverified_mechanisms,
                            outer_sync=miles_args.yeto_rl_outer_sync)
    assert event["rl/contains_unverified_mechanisms"] is True
    assert event["rl/outer_sync"] is False


def test_main_runs_without_policy_sync_in_no_sync_mode():
    import inspect

    assert 'yeto_policy_sync=not getattr(args, "rl_single_island_no_sync", False)' in \
        inspect.getsource(rl_learner.main)


def test_launcher_no_sync_island_and_refusals():
    from yeto.launcher import _prepare_rl_args

    run = _island_run(_launcher_args("ports", ("--rl-single-island-no-sync",
                                               "--rl-allow-unverified-mechanism",
                                               "features:clip_higher")))
    assert "--rl-single-island-no-sync" in run and "--syncer" not in run
    assert "--rl-allow-unverified-mechanism features:clip_higher" in run
    with pytest.raises(ValueError, match="exactly one island"):
        _prepare_rl_args(_launcher_args("ports", ("--rl-single-island-no-sync",),
                                        gpu="aws:1xa100@us-east-1,aws:1xa100@us-west-2"))
    with pytest.raises(ValueError, match="only applies to --rl-engine ports"):
        _prepare_rl_args(_launcher_args("legacy", ("--rl-single-island-no-sync",)))
    normal = _island_run(_launcher_args("ports"))
    assert "--syncer $SYNCER_ADDR" in normal and "no-sync" not in normal


def test_launcher_run_skips_syncer_in_no_sync_mode():
    import inspect

    from yeto import launcher

    source = inspect.getsource(launcher.run)
    assert 'syncer_cluster = None if head_mode or no_sync else f"{prefix}-syncer"' in source
    assert "if no_sync:\n            syncer_task = syncer_job = None" in source


def test_no_sync_run_export_is_marked_from_its_event_tape(tmp_path, monkeypatch):
    # The event a --rl-single-island-no-sync island writes, carried into export.
    from yeto.rl import export as rl_export

    launch = _launch()
    event = selection_event(launch=launch, algorithm=launch.algorithm, miles_commit="x",
                            unverified_mechanisms=("features:clip_higher",), outer_sync=False)
    tape = tmp_path / "rl-island-0.jsonl"
    tape.write_text(json.dumps({"island_id": 0, **event}) + "\n"
                    + json.dumps({"event": "rl_local_round"}) + "\n")
    seen = {}
    monkeypatch.setattr(rl_export, "export_rl_checkpoint",
                        lambda *a, **kw: seen.update(kw) or SimpleNamespace(policy_version=1))
    rl_export.main(["--checkpoint", "c", "--model", "m", "--model-revision", "a" * 40,
                    "--lora-r", "2", "--output-dir", "o", "--rl-event-tape", str(tape)])
    assert seen["algorithm_spec"] == event["rl/algorithm_spec"]
    assert list(seen["unverified_mechanisms"]) == ["features:clip_higher"]
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({**event, "rl/algorithm_spec_sha256": "0" * 64}) + "\n")
    with pytest.raises(ValueError):
        rl_export.algorithm_from_event_tape(bad)


def test_no_sync_requires_rl_training_mode():
    from yeto import launcher

    args = _launcher_args("ports", ("--rl-single-island-no-sync",))
    args.training_mode = "sft"
    with pytest.raises(ValueError, match="requires --training-mode rl"):
        launcher.run(args)


def test_outer_sync_always_in_event():
    plain = selection_event(launch=_launch(AlgorithmSpec()), algorithm=AlgorithmSpec(),
                            miles_commit="x")
    assert plain["rl/outer_sync"] is True


def test_launcher_forwards_optimizer_steps():
    from yeto.launcher import _prepare_rl_args

    assert "--optimizer-steps 1 " in _island_run(_launcher_args("ports"))
    run = _island_run(_launcher_args("ports", ("--rl-single-island-no-sync",
                                               "--rl-optimizer-steps", "2")))
    assert "--optimizer-steps 2 " in run  # rollout 4 x 2 samples = 8, divisible
    for extra, match in ((("--rl-optimizer-steps", "3"), "must divide"),
                         (("--rl-optimizer-steps", "0"), "positive int")):
        with pytest.raises(ValueError, match=match):
            _prepare_rl_args(_launcher_args("ports", extra))
    with pytest.raises(ValueError, match="only applies to --rl-engine ports"):
        _prepare_rl_args(_launcher_args("legacy", ("--rl-optimizer-steps", "2")))


# -- yeto launch --dry-run ----------------------------------------------------------


def _plan(engine, extra=(), gpu="aws:1xa100@us-east-1"):
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    args = _launcher_args(engine, extra, gpu=gpu)
    args.controller = "local" if "--controller" not in extra else args.controller
    _prepare_rl_args(args)
    return dry_run_plan(args)


def test_dry_run_no_sync_single_gpu_no_syncer():
    plan = _plan("ports", ("--rl-single-island-no-sync", "--rl-allow-unverified-mechanism",
                           "features:clip_higher"))
    assert plan["islands"] == 1 and plan["total_gpus"] == 1
    assert plan["syncer"] is None and plan["outer_sync"] is False
    assert plan["algorithm_spec_sha256"] == AlgorithmSpec().sha256()
    assert plan["unverified_mechanisms"] == ["features:clip_higher"]
    [island] = plan["island_requests"]
    assert island["gpu"] == "A100" and island["total_gpus"] == 1
    assert "--rl-single-island-no-sync" in island["learner_command"]
    assert "--syncer" not in island["learner_command"]


def test_dry_run_default_two_islands_with_syncer():
    plan = _plan("ports", gpu="aws:1xa100@us-east-1,aws:1xa100@us-west-2")
    assert plan["islands"] == 2 and plan["total_gpus"] == 2 and plan["outer_sync"]
    assert plan["syncer"].endswith("-syncer")
    assert all("--syncer $SYNCER_ADDR" in i["learner_command"] for i in plan["island_requests"])


def test_dry_run_refusals():
    from yeto.launcher import _prepare_rl_args, dry_run_plan

    args = _launcher_args("ports", ("--rl-single-island-no-sync",))
    _prepare_rl_args(args)
    args.controller = "head"
    with pytest.raises(ValueError, match="--controller local"):
        dry_run_plan(args)


def test_cli_dry_run_creates_nothing(monkeypatch, capsys):
    import yeto.cli as cli
    import yeto.launcher as launcher

    calls = []
    monkeypatch.setattr(launcher, "prepare_launch_args", launcher._prepare_rl_args)
    monkeypatch.setattr(launcher, "run", lambda *a, **k: calls.append("run"))
    monkeypatch.setattr(cli, "_spawn_worker", lambda *a, **k: calls.append("spawn"))
    monkeypatch.setattr(cli.runs, "create_run", lambda *a, **k: calls.append("create"))
    args = _launcher_args("ports", ("--rl-single-island-no-sync", "--dry-run",
                                    "--controller", "local"))
    assert cli.cmd_launch(args) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["dry_run"] and plan["total_gpus"] == 1 and plan["syncer"] is None
    assert calls == []
