"""Which fleets ``yeto run up`` admits by backend (rl-fleet-verl-islands, admission):
every island on one backend, and that backend declaring the fleet's form as a
capability (``yeto.rl.engine.backends.capabilities``); a verl fleet with compacting
pools or CompactionRL is refused at the preflight; a verl trainer image of another
commit than the pin is refused naming its island. No cloud, no GPU."""

from __future__ import annotations

import json

import pytest

from tests.test_run_up_fleet import TRAINER_ARGV, _cfg, _main_args, _write, files, no_subprocess  # noqa: F401
from yeto.rl import fleet, up
from yeto.rl.engine import backend_identity as bi
from yeto.rl.engine import backends

# a verl trainer sizes its prompt rows within --seq-len next to a full response (prompt width preflight)
VERL_ARGV = TRAINER_ARGV + ["--rl-backend", "verl", "--rl-parameter-layout-sha256", "a" * 64,
                            "--seq-len", "4096", "--rollout-max-response-len", "2048"]


def _run(files, argv, **edit):
    cfg = _cfg(files, **edit)
    cfg["fleet"]["trainer_argv"] = list(argv)
    return up.Run(up.load_config(_write(files, cfg)))


# -- capabilities ----------------------------------------------------------------------------

def test_miles_and_verl_declare_what_they_train_in_a_fleet():
    assert {bi.FLEET_FULL_PARAMETER, bi.FLEET_PARTIAL_EXPERTS, bi.PER_LEARNER_RELEASE, bi.RELEASE_CADENCE,
            bi.COMPACTION_SEGMENTS} <= backends.capabilities("miles")
    verl = backends.capabilities("verl")
    # both of verl's: the fleet admission and its buffer trainer's per-learner releases
    assert {bi.FLEET_FULL_PARAMETER, bi.PER_LEARNER_RELEASE} <= verl
    assert not verl & {bi.FLEET_PARTIAL_EXPERTS, bi.RELEASE_CADENCE, bi.COMPACTION_SEGMENTS}
    # a registered backend without an identity declares nothing; an unknown name is refused
    backends.register(backends.BackendEntry("bare", "yeto.rl.adapters.bare", {}))
    try:
        assert backends.capabilities("bare") == frozenset()
    finally:
        backends._REGISTRY.pop("bare")  # noqa: SLF001
    with pytest.raises(backends.UnknownBackend):
        backends.capabilities("nope")
    # the identity hash does not carry the capabilities: an island's contract is unchanged
    from yeto.rl.adapters.miles import identity

    assert "CAPABILITIES" not in str(identity.backend_identity(compat_group="nvidia-h100"))


def test_a_backend_without_the_capability_is_refused_by_name(files, monkeypatch):
    real = backends.capabilities
    monkeypatch.setattr(backends, "capabilities",
                        lambda name=None: frozenset() if name == "verl" else real(name))
    problem = _run(files, VERL_ARGV).fleet_problem()
    assert f"declares {bi.FLEET_FULL_PARAMETER} (miles)" in problem and "--rl-backend verl does not" in problem


# -- verl fleets ---------------------------------------------------------------------------

def test_a_two_island_verl_fleet_passes_and_prints_its_backend(files, no_subprocess, capsys):
    run = _run(files, VERL_ARGV)
    assert run.fleet_problem() is None
    sync = run.fleet_plan().sync
    assert (sync["backend"], sync["preset"], sync["rounds"], sync["form"]) == (
        "verl", "full-parameter", "async", "full parameters")
    cfg = _cfg(files)
    cfg["fleet"]["trainer_argv"] = list(VERL_ARGV)
    assert up.main(_main_args(_write(files, cfg))) == 0
    out = capsys.readouterr().out
    assert ("fleet: island 0 east: learner 0, store trainer0/ (s3://b/runs/r/trainer0), trainer modal:1x2xh100 "
            "on verl at $8/h, release_every 2, 5 buffer shard(s), pools i0-vllm, i0-sglang") in out
    assert "fleet: island 1 west: learner 1, store trainer1/" in out and "on verl at $8/h" in out


def test_a_miles_fleets_plan_lines_name_no_backend(files):
    lines = _run(files, TRAINER_ARGV).fleet_plan().lines()
    assert not any(" on miles" in line or " on verl" in line for line in lines)


def test_miles_and_verl_islands_in_one_fleet_are_refused(files):
    # a Miles sampler island in a verl fleet
    cfg = _cfg(files)
    cfg["fleet"]["trainer_argv"] = list(VERL_ARGV)
    cfg["islands"][1]["samplers"].insert(0, {"kind": "miles", "name": "ms", "gpu": "modal:1x1xh100"})
    backends_by = fleet.fleet_backends(cfg)
    assert backends_by == {"island 0 (east)": "verl", "island 1 (west)": "verl",
                           "Miles sampler ms of island 1 (west)": "miles"}
    problem = fleet.form_problem(fleet.training_form(VERL_ARGV), backends=backends_by)
    assert "run one training backend" in problem and "Miles sampler ms of island 1 (west): miles" in problem
    assert "shape-free layout" in problem
    # the same check through the run (a Miles fleet with a Miles sampler: one backend)
    assert fleet.form_problem(fleet.training_form(TRAINER_ARGV), backends={"island 0 (east)": "miles",
                                                                           "Miles sampler ms": "miles"}) is None


def test_verl_partial_experts_are_refused(files, tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text("{}")
    problem = _run(files, VERL_ARGV + ["--rl-expert-selection", str(selection)]).fleet_problem()
    assert "partial-experts fleet" in problem and "--rl-backend verl does not train partial experts" in problem
    assert "--rl-backend miles" in problem
    # Miles still trains partial experts in a fleet
    assert _run(files, TRAINER_ARGV + ["--rl-expert-selection", str(selection)]).fleet_problem() is None


def test_a_verl_lora_fleet_is_refused(files):
    lora = [a for a in VERL_ARGV if a not in ("--tuning", "full")] + ["--tuning", "lora"]
    assert "LoRA fleet" in _run(files, lora).fleet_problem()


# -- CompactionRL --------------------------------------------------------------------------

def _compacting(files):
    spec = json.loads((files / "sglang.json").read_text())
    spec["agentic"] = {"compaction": {"context_budget": 8192, "t_comp": 1024, "keep_recent_steps": 2,
                                      "max_compactions": 3}}
    (files / "sglang.json").write_text(json.dumps(spec))


def test_a_verl_fleet_with_a_compacting_pool_is_refused_naming_island_and_pool(files, no_subprocess):
    _compacting(files)
    run = _run(files, VERL_ARGV)
    problem = run.fleet_problem()
    assert problem.startswith("island 0 (east) pool i0-sglang: its sampler spec sets agentic.compaction")
    assert "use --rl-backend miles for CompactionRL" in problem
    assert not run.preflight() and no_subprocess == []
    # a Miles fleet takes the same pools to the per-pool checks, as before
    assert _run(files, TRAINER_ARGV).fleet_problem() is None


def test_a_verl_fleet_running_compactionrl_is_refused(files, monkeypatch):
    from yeto.launch import harness_env

    monkeypatch.setattr(harness_env, "_launch_algorithm",
                        lambda args: {"advantage": {"gae_variant": "cross_segment_per_sample"}})
    problem = _run(files, VERL_ARGV).fleet_problem()
    assert "runs CompactionRL" in problem and "--rl-backend verl" in problem


def test_the_run_time_compaction_round_stop_stays_on_the_verl_trainer():
    """The backstop: a verl buffer trainer still stops on a group of compacted trials
    (trainer.compaction_round) when one reaches it past the preflight."""
    import inspect

    from yeto.rl.adapters.verl import buffer_feed
    from yeto.rl.engine import batch_reader
    from yeto.rl.engine.trainer_breakers import COMPACTION_ROUND

    assert buffer_feed.check_compaction_round is batch_reader.check_compaction_round
    source = inspect.getsource(buffer_feed)
    assert "check_compaction_round(" in source and "cannot train a trial whole" in source
    seg = {"tokens": [1], "metadata": {"compaction": {"segment": 0}}}
    with pytest.raises(Exception, match="compacted trials") as caught:
        batch_reader.check_compaction_round(
            [(0, [seg, seg])], seq_length=None,
            schedule_refusal=lambda positions: f"groups at buffer positions {positions} hold compacted trials")
    assert COMPACTION_ROUND in repr(caught.value) or "compacted trials" in str(caught.value)


# -- trainer image pin ---------------------------------------------------------------------

def test_a_verl_image_of_another_commit_is_refused_naming_the_island():
    from yeto.launch.trainer_image_check import image_pin_problems
    from yeto.rl.adapters.verl import launch_flags, pins

    other = "0" * 40
    assert launch_flags.image_pin_problem(pins.VERL_IMAGE) is None
    assert launch_flags.image_pin_problem("docker:r@sha256:" + "a" * 64) is None
    why = launch_flags.image_pin_problem(f"verl-build:{other}")
    assert f"verl commit {other[:12]}" in why and f"use verl-build:{pins.VERL_COMMIT}" in why
    assert f"use verl-megatron-build:{pins.VERL_COMMIT}" in launch_flags.image_pin_problem(
        f"verl-megatron-build:{other}")
    args = _launch_args(["--rl-image", f"verl-build:{other}"])
    (found,) = image_pin_problems(args)
    assert found.startswith("--gpu island 0 (modal:1x2xH100): trainer image verl-build:") and other[:12] in found
    # the launch refuses it with that message, not the generic image form
    with pytest.raises(ValueError, match=r"--gpu island 0 .*built from verl commit 000000000000"):
        _prepared(["--rl-image", f"verl-build:{other}"])


def test_the_pinned_verl_image_and_miles_images_pass_the_pin_check():
    from yeto.launch.trainer_image_check import image_pin_problems

    assert image_pin_problems(_launch_args([])) == []
    miles = _launch_args([], backend="miles")
    assert image_pin_problems(miles) == []


def test_an_island_image_of_another_verl_commit_names_its_island():
    from yeto.rl.engine import island_engine

    args = _launch_args(["--rl-island-engine", f"1:image=verl-build:{'1' * 40}"],
                        gpu="modal:1x2xh100,modal:1x2xh100")
    with pytest.raises(ValueError, match=r"island 1 image .*built from verl commit 111111111111"):
        island_engine.check(args, 2)


def _launch_args(extra, *, backend="verl", gpu="modal:1x2xh100"):
    from yeto.cli import parse_args

    return parse_args(["--training-mode", "rl", "--model", "Qwen/Qwen3-0.6B", "--data", "openai/gsm8k",
                       "--gpu", gpu, "--reward-function", "yeto.rl.gsm8k_reward:score", "--rl-backend", backend,
                       *extra])


def _prepared(extra):
    from yeto.launcher import _prepare_rl_args

    args = _launch_args(extra)
    _prepare_rl_args(args)
    return args


def test_a_verl_fleet_launch_never_reaches_the_two_cluster_island_refusal(files):
    """engine_cluster refuses verl only for --rl-engine-cluster, which the fleet compile
    never writes (a fleet trainer starts no engine; its samplers are the pools)."""
    from yeto.launch import engine_cluster
    from yeto.rl.up_consistency import trainer_args

    run = _run(files, VERL_ARGV)
    argv = [str(a) for a in run.cfg["trainer"]["argv"]]
    assert not any(a.split("=", 1)[0].startswith("--rl-engine-cluster") for a in argv)
    args = trainer_args(argv)
    assert args.rl_backend == "verl" and not engine_cluster.requested(args)
    assert engine_cluster.plan(args, [], []) == {}
    args.rl_engine_cluster = "nebius:1xh100"  # given by hand: still refused, with its reason
    with pytest.raises(ValueError, match="--rl-engine-cluster needs --rl-backend miles"):
        engine_cluster.plan(args, [], [])
