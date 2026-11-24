"""``yeto run up`` preflight: values the trainer and the samplers must share.

Each case is a run config whose own dry runs (``nc pool --dry-run``, ``yeto launch
--dry-run``) pass, and that would fail only after the trainer island loaded its
weights (or train on the wrong data). Preflight must refuse it before anything is
ordered."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from yeto import launch_preflight
from yeto.rl import up, up_consistency
from yeto.rl.adapters.miles.models.partial_experts import selection as selection_tool

STORE = "s3://bkt/runs/r1"


def _selection(path) -> types.SimpleNamespace:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("builtins.print", lambda *a, **k: None)
        assert selection_tool.main(["--model", "kimi_k31", "--num-experts", "16", "--num-trainable", "8",
                                    "--ep-size", "8", "--moe-layer-offset", "1", "--mode", "uniform",
                                    "--num-moe-layers", "3", "--skip-layer-check", "--output", str(path)]) == 0
    sel = selection_tool.ExpertSelection.load_file(path)
    return types.SimpleNamespace(path=str(path), sha256=sel.sha256, trainable_sha256=sel.trainable_sha256)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """A consistent K3.1-shaped run: preflight passes it."""
    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    monkeypatch.setattr(launch_preflight, "modal_sdk_problem", lambda env: None)
    sel = _selection(tmp_path / "expert_selection.json")
    env_file = tmp_path / "store.env"
    env_file.write_text("R2_BUCKET=bkt\n")
    spec = {"sampler_id": "<nc-node-id>", "rollout_fields": {"routed_experts": True, "top_logprobs_num": 128},
            "expert_selection_sha256": sel.sha256,
            "engine": {"server": {"max_model_len": 32768}},
            "sampling": {"n": 8, "max_tokens": 8192}, "max_prompt_tokens": 4096,
            "release_dir": "policy-releases"}
    pool = {"shard_base": "nc", "shard_limit": 2, "run_prefix": "r1", "r2_env_file": str(env_file),
            "spec": str(tmp_path / "spec.json")}
    argv = ["--training-mode", "rl", "--gpu", "modal:6x8xb300", "--model", "org/k31", "--data", "d",
            "--rl-model-recipe", "kimi_k31", "--rl-expert-selection", sel.path,
            "--rl-checkpoint-store", STORE, "--rl-buffer-shards", "nc-0,nc-1",
            "--n-samples-per-prompt", "8", "--rollout-max-response-len", "12288", "--seq-len", "16384"]
    ns = types.SimpleNamespace(tmp=tmp_path, spec=spec, pool=pool, argv=argv, sel=sel, env_file=env_file)

    def make():
        (tmp_path / "spec.json").write_text(json.dumps(ns.spec))
        (tmp_path / "pool.json").write_text(json.dumps(ns.pool))
        run = up.Run({"run": "r1", "store": STORE, "pool": {"config": str(tmp_path / "pool.json")},
                      "trainer": {"argv": list(ns.argv), "usd_per_h": 100}, "budget": {"usd": 500}})
        run.calls = []
        run._cmd = lambda argv, *, log, timeout=None: run.calls.append(argv) or 0
        return run

    ns.make = make
    return ns


def _refused(run, *needles):
    assert run.preflight() is False
    assert run.calls == [], "nothing may run after a refused preflight (no dry-run, no order)"
    problem = run.consistency_problem()
    for needle in needles:
        assert needle in problem, problem
    return problem


def test_a_consistent_run_passes(setup):
    run = setup.make()
    assert run.consistency_problem() is None
    assert run.preflight() is True and run.calls


def _set(argv, flag, value):
    i = argv.index(flag)
    argv[i + 1] = value


def test_group_size_mismatch_is_refused(setup):
    _set(setup.argv, "--n-samples-per-prompt", "4")  # sampler groups of 8 > 4: consumed, never trained
    _refused(setup.make(), "sampling.n 8", "--n-samples-per-prompt 4", "consumed without being trained")
    _set(setup.argv, "--n-samples-per-prompt", "16")  # every round half padding
    _refused(setup.make(), "padded with 8 loss-free samples")


def test_the_launcher_default_group_size_is_compared(setup):
    i = setup.argv.index("--n-samples-per-prompt")
    del setup.argv[i:i + 2]  # the launcher's default (4) against the spec's 8
    _refused(setup.make(), "(the launcher default)")


def test_a_pool_override_of_n_is_the_one_compared(setup):
    setup.pool["sampler"] = {"n": 4}
    _refused(setup.make(), "sampling.n 4")
    setup.pool["sampler"] = {"n": 8}
    assert setup.make().consistency_problem() is None


def test_a_response_wider_than_the_trainer_row_is_refused(setup):
    setup.spec["sampling"]["max_tokens"] = 16384
    setup.spec["engine"]["server"]["max_model_len"] = 65536
    _refused(setup.make(), "sampling.max_tokens 16384 > the trainer's --rollout-max-response-len 12288")


def test_a_plain_spec_max_prompt_tokens_is_no_row_bound(setup):
    """A plain spec's max_prompt_tokens cuts no prompt (only the engine capacity check
    reads it; often max_model_len - max_tokens): a large one is no row mismatch."""
    setup.spec["engine"]["server"]["max_model_len"] = 262144
    setup.spec["max_prompt_tokens"] = 262144 - 8192
    run = setup.make()
    assert run.consistency_problem() is None
    assert run.preflight() is True


def test_agentic_caps_are_compared_to_the_trainer(setup):
    setup.spec["agentic"] = {"max_response_len": 32768, "max_prompt_len": 4096}
    problem = _refused(setup.make(), "agentic.max_response_len 32768 > the trainer's --rollout-max-response-len")
    assert "--seq-len" in problem
    setup.spec["agentic"] = {"max_response_len": 12288, "max_prompt_len": 8192}
    _refused(setup.make(), "agentic.max_prompt_len 8192 + agentic.max_response_len 12288 = 20480 > "
                           "the trainer's --seq-len 16384")
    setup.spec["agentic"] = {"max_response_len": 12288, "max_prompt_len": 4096}
    assert setup.make().consistency_problem() is None


def test_r3_trainer_needs_routed_experts_from_the_sampler(setup):
    setup.spec["rollout_fields"] = {"top_logprobs_num": 128}
    _refused(setup.make(), "R3, the kimi_k31 recipe", "rollout_fields.routed_experts")
    # a generic trainer with the flag as well
    setup.argv[setup.argv.index("--rl-model-recipe") + 1] = "generic"
    i = setup.argv.index("--rl-expert-selection")
    del setup.argv[i:i + 2]
    setup.argv.append("--use-rollout-routing-replay")
    _refused(setup.make(), "--use-rollout-routing-replay")


def test_expert_selection_of_another_file_is_refused(setup):
    setup.spec["expert_selection_sha256"] = "a" * 64
    _refused(setup.make(), "expert_selection_sha256 aaaaaaaaaaaa..", setup.sel.sha256[:12])
    setup.spec["expert_selection_sha256"] = "<sha256 of expert_selection.json>"  # a placeholder: not checked
    assert setup.make().consistency_problem() is None
    setup.spec["expert_trainable_sha256"] = "b" * 64  # replaces the selection check on the sampler
    _refused(setup.make(), "expert_trainable_sha256")
    setup.spec["expert_trainable_sha256"] = setup.sel.trainable_sha256
    setup.spec["expert_selection_sha256"] = "c" * 64  # not what the sampler checks then
    assert setup.make().consistency_problem() is None


def test_a_generic_partial_experts_trainer_refuses_a_spec_bound_to_its_selection(setup):
    """A generic partial-experts release is a whole HF checkpoint without a manifest:
    a sampler spec with either selection sha256 refuses every release (even one of the
    same selection file), so the preflight refuses the spec and names the fix."""
    _set(setup.argv, "--rl-model-recipe", "generic")
    setup.argv += ["--tuning", "full"]
    assert setup.spec["expert_selection_sha256"] == setup.sel.sha256  # the trainer's own selection
    problem = _refused(setup.make(), "the sampler spec sets expert_selection_sha256",
                       "carry no expert selection", "remove expert_selection_sha256 from the sampler spec")
    assert "expert_trainable_sha256" not in problem
    del setup.spec["expert_selection_sha256"]
    setup.spec["expert_trainable_sha256"] = setup.sel.trainable_sha256
    _refused(setup.make(), "remove expert_trainable_sha256 from the sampler spec")
    setup.spec["expert_selection_sha256"] = setup.sel.sha256
    _refused(setup.make(), "remove expert_selection_sha256 / expert_trainable_sha256 from the sampler spec")
    # without the two fields the sampler loads every release
    del setup.spec["expert_selection_sha256"], setup.spec["expert_trainable_sha256"]
    assert setup.make().consistency_problem() is None
    # a path that is only on the trainer node: still refused (the release form, not the file, decides)
    _set(setup.argv, "--rl-expert-selection", "/remote/expert_selection.json")
    setup.spec["expert_selection_sha256"] = setup.sel.sha256
    assert "carry no expert selection" in setup.make().consistency_problem()


def test_a_pool_writing_to_another_store_is_refused(setup):
    setup.pool["run_prefix"] = "r1-typo"
    _refused(setup.make(), "s3://bkt/runs/r1-typo", f"trainer reads {STORE}")
    setup.pool["run_prefix"] = "r1"
    setup.env_file.write_text("R2_BUCKET=other\n")
    _refused(setup.make(), "s3://other/runs/r1")


def test_the_trainer_store_must_be_the_run_store(setup):
    _set(setup.argv, "--rl-checkpoint-store", "s3://bkt/runs/r0")
    problem = _refused(setup.make(), "--rl-checkpoint-store s3://bkt/runs/r0 != the run config's store")
    assert "trainer reads s3://bkt/runs/r0" in problem  # and the pool writes elsewhere too


def test_store_spelling_and_synced_substores_are_accepted(setup):
    _set(setup.argv, "--rl-checkpoint-store", "r2://bkt/runs/r1/")
    assert setup.make().consistency_problem() is None
    _set(setup.argv, "--rl-checkpoint-store", STORE)
    setup.pool["run_prefix"] = "r1/trainer1"  # samplers following trainer 1 of a synced group
    assert setup.make().consistency_problem() is None
    setup.pool["run_prefix"] = "r1/other"
    assert "r1/other" in setup.make().consistency_problem()


def test_a_syncer_data_path_pool_is_not_store_checked(setup):
    setup.pool["run_prefix"] = "elsewhere"
    assert up_consistency.problems(STORE, setup.argv, {**setup.pool, "spec": None}, data_path="syncer") == []


def test_an_unknown_release_dir_is_refused(setup):
    setup.spec["release_dir"] = "policy-release"
    _refused(setup.make(), "release_dir 'policy-release'")
    setup.spec["release_dir"] = "policy-releases-mxfp4-g32-rtn"
    assert setup.make().consistency_problem() is None


def test_an_argv_the_launcher_rejects_is_left_to_its_dry_run(setup):
    setup.argv[:] = ["--gpu", "modal:2x8xb300"]  # no --model: yeto launch --dry-run reports it
    assert setup.make().consistency_problem() is None


def _release_cadence(setup, *, samplers, cap, every=6, groups=8):
    """A single buffer trainer releasing every ``every``-th version (skip-busy staged
    uploads) over ``samplers`` samplers of ``cap`` max_backlog_groups, 2 groups a round."""
    setup.argv += ["--rl-sample-role", "trainer", "--rl-single-island-no-sync", "--tuning", "full",
                   "--rollout-batch-size", str(groups), "--rl-release-every", str(every), "--rl-release-skip-busy"]
    setup.pool["target"] = samplers
    setup.spec["max_backlog_groups"] = cap
    setup.spec["prompts"] = {"groups_per_round": 2}


def test_backlog_caps_below_the_release_cadence_warn_with_the_value_to_set(setup, capsys):
    """3 samplers x 16 against 8 groups a round released every 6th version: the trainer
    reads up to (6 + 3 + 1) x 8 + 8 = 88 groups between two releases while the samplers
    hold 48 past the newest release's read position. A warning names the cap to set:
    ceil(88 / 3) = 30, or ceil(88 / 2) = 44 with one sampler down."""
    _release_cadence(setup, samplers=3, cap=16)
    run = setup.make()
    assert run.consistency_problem() is None and run.rules_problem() is None
    out = capsys.readouterr().out
    assert "WARNING: D19 warn" in out and "3 x 16 max_backlog_groups = 48 groups" in out and "up to 88 groups" in out
    assert "at least 30 per sampler" in out and "44 to keep up with one of the 3 samplers down" in out
    assert run.preflight() is True and run.calls  # a warning orders as before
    assert "max_backlog_groups" in capsys.readouterr().out
    setup.spec["max_backlog_groups"] = 30
    assert setup.make().rules_problem() is None
    assert "max_backlog_groups" not in capsys.readouterr().out
    setup.spec["max_backlog_groups"] = "auto"  # sized from the release cadence: satisfied
    assert setup.make().rules_problem() is None
    assert "D19" not in capsys.readouterr().out


def test_a_cap_the_trainer_cannot_read_past_to_its_first_release_is_refused(setup):
    """1 sampler x 16 (+2 past its cap) before the first release, but v6 needs 48 groups:
    the trainer would wait for groups and never release. Refused before anything is
    ordered; an elastic pool that may run 3 nodes at once is not refused."""
    _release_cadence(setup, samplers=1, cap=16)
    run = setup.make()
    assert run.preflight() is False and run.calls == []
    problem = run.rules_problem()
    assert problem.startswith("D19 error") and "= 48 groups before it can release" in problem
    assert "never release" in problem and "at least 88 per sampler" in problem
    setup.pool["elastic"] = {"max_nodes": 3}
    assert setup.make().rules_problem() is None  # 3 x (16 + 2) >= 48: a warning only


def test_the_backlog_check_follows_the_pool_section_and_skips_other_trainers(setup):
    _release_cadence(setup, samplers=1, cap=16)
    setup.pool["backlog"] = {"synced": False, "explicit_rounds": 64}  # nc.cli: 64 rounds x 2 groups
    run = setup.make()
    assert run.rules_problem() is None
    shape = up_consistency.pool_backlog(run.pools[0].load())
    assert (shape["cap"], shape["target"], shape["over"]) == (128, 1, 2)
    del setup.pool["backlog"]
    i = setup.argv.index("--rl-single-island-no-sync")
    del setup.argv[i]  # a synced trainer: its samplers are sized by the pool's backlog section
    assert setup.make().rules_problem() is None
    setup.argv.append("--rl-single-island-no-sync")
    setup.pool["data_path"] = "syncer"  # relay pools have no cap
    assert setup.make().rules_problem() is None


def test_release_record_rounds_and_backlog_need():
    from yeto.rl.engine import sample_roles

    rounds = sample_roles.release_record_rounds
    assert rounds(release_every=6, staged=False, skip_busy=False) == 0  # inline: recorded at once
    assert rounds(release_every=6, staged=True, skip_busy=True) == 1 + sample_roles.ASSUMED_UPLOAD_ROUNDS
    assert rounds(release_every=1, staged=True, skip_busy=False) == 1  # the next release waits
    assert rounds(release_every=6, staged=True, skip_busy=True, upload_rounds=5) == 6
    need = sample_roles.release_backlog_need
    assert need(groups_per_round=8, release_every=6, record_rounds=3) == 88
    assert need(groups_per_round=8, release_every=1, record_rounds=3) == 48  # the 1123 e2e: 1 x 16 < 48


def test_an_auto_cap_is_derived_from_the_release_cadence(setup, capsys, monkeypatch):
    """``max_backlog_groups: "auto"``: run up derives ceil(88 / 3) = 30 for 3 samplers of a
    trainer releasing every 6th version of 8 groups (88 / 1 = 88 for one), hands it to the
    pool processes in their environment, prints it, and the backlog check passes."""
    from yeto.rl.engine import sample_roles

    monkeypatch.setenv(sample_roles.AUTO_BACKLOG_ENV, "7")  # an inherited value never reaches the pools
    _release_cadence(setup, samplers=3, cap="auto")
    run = setup.make()
    assert run.auto_backlog["groups"] == 30 and run.env[sample_roles.AUTO_BACKLOG_ENV] == "30"
    assert run.env[sample_roles.AUTO_BACKLOG_WHY_ENV] == run.auto_backlog["why"] == (
        "ceil(((release_every 6 + m 3 + 1) x 8 + 8) / samplers 3) = 30")
    assert run.rules_problem() is None
    out = capsys.readouterr().out
    assert "max_backlog_groups auto -> 30 per sampler" in out and "D19" not in out
    shape = up_consistency.pool_backlog(run.pools[0].load(), auto=30)
    assert shape["cap"] == 30
    setup.pool["target"] = 1
    run = setup.make()
    assert run.auto_backlog["groups"] == 88 and run.rules_problem() is None
    assert run.preflight() is True and run.calls


def test_an_auto_cap_without_a_single_buffer_trainer_takes_the_fallback(setup, capsys, monkeypatch):
    from yeto.rl.engine import sample_roles

    monkeypatch.setenv(sample_roles.AUTO_BACKLOG_ENV, "7")
    _release_cadence(setup, samplers=2, cap="auto")
    setup.argv.remove("--rl-single-island-no-sync")  # a synced trainer: the formula does not cover it
    run = setup.make()
    assert run.auto_backlog is None and sample_roles.AUTO_BACKLOG_ENV not in run.env
    assert run.rules_problem() is None
    assert f"auto -> fallback {sample_roles.AUTO_BACKLOG_FALLBACK_GROUPS} per sampler" in capsys.readouterr().out
    setup.argv.append("--rl-single-island-no-sync")
    setup.spec["max_backlog_groups"] = 16  # a number: no derivation, the check as before
    run = setup.make()
    assert run.auto_backlog is None and sample_roles.AUTO_BACKLOG_ENV not in run.env


def test_resolve_auto_backlog():
    from yeto.rl.engine import sample_roles

    resolve = sample_roles.resolve_auto_backlog
    assert resolve(16, {}) == (16, None) and resolve("16", {}) == ("16", None)  # numbers are left alone
    cap, note = resolve("auto", {})
    assert cap == sample_roles.AUTO_BACKLOG_FALLBACK_GROUPS and "fallback" in note
    env = {sample_roles.AUTO_BACKLOG_ENV: "30", sample_roles.AUTO_BACKLOG_WHY_ENV: "ceil(88 / 3) = 30"}
    assert resolve("AUTO", env) == (30, "auto: ceil(88 / 3) = 30")
    assert resolve("auto", {sample_roles.AUTO_BACKLOG_ENV: "0"})[0] == sample_roles.AUTO_BACKLOG_FALLBACK_GROUPS
    assert sample_roles.auto_backlog_groups(groups_per_round=8, release_every=1, record_rounds=3, samplers=1) == 48
    assert sample_roles.auto_backlog_groups(groups_per_round=8, release_every=6, record_rounds=3, samplers=3) == 30


EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "k31"


def _example_trainer_argv(tmp_path, env_overrides) -> list[str]:
    """The ``yeto launch`` argv of examples/k31/launch-trainer-modal-b300.sh (a stub
    ``yeto`` on PATH records it; nothing launches)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    out = tmp_path / "argv.json"
    stub = bin_dir / "yeto"
    stub.write_text(f"#!{sys.executable}\nimport json, sys\njson.dump(sys.argv[2:], open({str(out)!r}, 'w'))\n")
    stub.chmod(0o755)
    env = {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "MODEL": "org/k31", "MODEL_REVISION": "r",
           "DATA": "d", "STORE_URI": STORE, "STORE_ENDPOINT": "https://e", "STORE_SECRET": "s",
           "EXPERT_SELECTION": str(tmp_path / "absent-selection.json"), "RL_IMAGE": "docker:i@sha256:0",
           "PARAMETER_LAYOUT_SHA256": "0" * 64, **env_overrides}
    subprocess.run(["bash", str(EXAMPLES / "launch-trainer-modal-b300.sh")], env=env, check=True)
    return json.loads(out.read_text())  # after "launch"


@pytest.mark.parametrize("spec_name", ["sampler-spec.nc-mi355x.json", "sampler-spec.nc-mi355x-vllm.json"])
@pytest.mark.parametrize("env_overrides", [{"STAGE": "1"}, {"STAGE": "2"}, {"STAGE": "2", "SEQ_LEN": "262144"},
                                           {"STAGE": "2", "FORM": "synced-2"}],
                         ids=["stage1", "stage2", "stage2-262k", "stage2-synced2"])
def test_the_k31_examples_pass_the_consistency_checks(tmp_path, spec_name, env_overrides):
    """Every STAGE of the example trainer launcher with each example sampler spec it
    names: a customer who runs the examples as they are is not refused."""
    argv = _example_trainer_argv(tmp_path, env_overrides)
    env_file = tmp_path / "store.env"
    env_file.write_text("R2_BUCKET=bkt\n")
    pool = {"run_prefix": "r1", "r2_env_file": str(env_file), "spec": str(EXAMPLES / spec_name)}
    assert up_consistency.trainer_args(argv) is not None
    assert up_consistency.problems(STORE, argv, pool) == []


def test_a_help_argv_prints_nothing(capsys):
    assert up_consistency.trainer_args(["--help"]) is None
    assert capsys.readouterr().out == ""


# ------------------------------------------------------------------ score_centering
def _score_centering(setup, **sampling):
    setup.argv += ["--rl-loss", "score_centering"]
    setup.spec["rollout_fields"] = {"routed_experts": True, "top_logprobs_num": 128, "fp32_lm_head": True}
    setup.spec["sampling"].update({"temperature": 1.0, "top_p": 1.0, "top_k": -1, **sampling})


def test_score_centering_with_the_recipe_sampling_passes(setup):
    _score_centering(setup)
    run = setup.make()
    assert run.rollout_fields_problem() is None and run.consistency_problem() is None
    assert run.preflight() is True
    setup.spec["sampling"]["top_k"] = 0  # 0: no top-k limit, as -1
    del setup.spec["sampling"]["top_p"]  # unset: the default 1
    assert setup.make().consistency_problem() is None


@pytest.mark.parametrize("field,value", [("temperature", 0.7), ("top_p", 0.95), ("top_k", 50),
                                         ("presence_penalty", 0.5), ("frequency_penalty", 0.1),
                                         ("repetition_penalty", 1.05)])
def test_score_centering_refuses_sampling_its_log_probs_do_not_hold_at(setup, field, value):
    _score_centering(setup, **{field: value})
    _refused(setup.make(), "score_centering", f"sampling.{field} {value!r}", "temperature 1, top_p 1, top_k -1")


def test_a_policy_loss_trainer_takes_any_sampling(setup):
    setup.spec["sampling"].update({"temperature": 0.7, "top_p": 0.9, "top_k": 50})
    setup.spec["engine"]["server"]["extra_args"] = ["--logprobs-mode", "processed_logprobs"]
    assert setup.make().consistency_problem() is None


def test_score_centering_refuses_the_log_prob_mode_switches(setup):
    _score_centering(setup)
    for args in (["--logprobs-mode", "raw_logprobs"], ["--logprobs-mode=processed_logprobs"]):
        setup.spec["engine"]["server"]["extra_args"] = args
        _refused(setup.make(), "vLLM --logprobs-mode in extra_args")
    setup.spec["engine"]["server"]["extra_args"] = []
    setup.spec["engine"]["server"]["extra_env"] = {"SGLANG_RETURN_ORIGINAL_LOGPROB": "1"}
    _refused(setup.make(), "SGLANG_RETURN_ORIGINAL_LOGPROB=1")
    setup.spec["engine"]["server"]["extra_env"] = {"SGLANG_RETURN_ORIGINAL_LOGPROB": "0"}
    assert setup.make().consistency_problem() is None
    env = setup.tmp / "agentic.env"
    env.write_text("SGLANG_RETURN_ORIGINAL_LOGPROB=true\n")
    setup.spec["engine"]["server"]["extra_env"] = {}
    setup.pool["agentic_env_file"] = str(env)
    _refused(setup.make(), "SGLANG_RETURN_ORIGINAL_LOGPROB=true")


def test_score_centering_without_a_readable_spec_is_refused(setup):
    """Fail closed: the top-K, sampling and switch checks cannot run without the spec."""
    _score_centering(setup)
    setup.pool["spec"] = str(setup.tmp / "absent.json")
    run = setup.make()
    assert "cannot be read" in run.rollout_fields_problem()
    assert "cannot be read" in run.consistency_problem()
    assert run.preflight() is False and run.calls == []
    del setup.pool["spec"]
    assert "(none) cannot be read" in setup.make().rollout_fields_problem()
    # without score_centering an unreadable spec stays the pool dry-run's to report
    setup.argv.remove("--rl-loss")
    setup.argv.remove("score_centering")
    run = setup.make()
    assert run.rollout_fields_problem() is None and run.consistency_problem() is None


def test_an_algorithm_spec_that_selects_score_centering_is_checked(setup):
    from yeto.rl.algos.score_centering import score_centering_spec

    path = setup.tmp / "algorithm.json"
    path.write_text(json.dumps(score_centering_spec(top_logprobs=64).structured_dict()))
    setup.argv += ["--rl-algorithm-spec", str(path)]
    setup.spec["sampling"]["temperature"] = 0.8
    _refused(setup.make(), "sampling.temperature 0.8", "top-64")
    setup.spec["sampling"]["temperature"] = 1.0
    setup.spec["rollout_fields"]["top_logprobs_num"] = 64
    assert setup.make().consistency_problem() is None


def test_score_centering_without_the_fp32_head_only_warns(setup, capsys):
    _score_centering(setup)
    del setup.spec["rollout_fields"]["fp32_lm_head"]
    run = setup.make()
    assert run.consistency_problem() is None
    assert run.preflight() is True
    assert "no rollout_fields.fp32_lm_head" in capsys.readouterr().out
    setup.spec["rollout_fields"]["fp32_lm_head"] = True
    assert setup.make().preflight() is True
    assert "fp32_lm_head" not in capsys.readouterr().out
    pool = json.loads((setup.tmp / "pool.json").read_text())
    setup.spec["rollout_fields"]["fp32_lm_head"] = False  # a policy_loss trainer: no warning
    setup.make()
    plain = [a for a in setup.argv if a not in ("--rl-loss", "score_centering")]
    assert up_consistency.score_centering_warnings(pool, plain) == []
