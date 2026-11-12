"""``yeto run up`` preflight: values the trainer and the samplers must share.

Each case is a run config whose own dry runs (``nc pool --dry-run``, ``yeto launch
--dry-run``) pass, and that would fail only after the trainer island loaded its
weights (or train on the wrong data). Preflight must refuse it before anything is
ordered."""

from __future__ import annotations

import json
import types

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
