"""Recipe environment defaults (yeto.launch.recipe_env) and the kimi_k31 recipe's
declaration: the next9 trainer MFU switches, set on the trainer island by default for
``--rl-model-recipe kimi_k31 --tuning full`` on the next9 image; an explicit value wins;
``--rl-recipe-env-defaults off`` turns them off; every other image / recipe / tuning
and sampler islands get none. Both island paths: the Modal island config and the
``ssh:`` node environment."""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto import launcher
from yeto.cli import parse_args as parse_cli
from yeto.gpu_spec import parse_gpu_spec
from yeto.launch import k31_checks, recipe_env
from yeto.launch.ssh_trainer_island import SshTrainerSpec, node_env
from yeto.launcher import _prepare_rl_args
from yeto.rl.adapters.miles import pins
from yeto.rl.adapters.miles.models.kimi_k31 import mfu_switches

ROOT = Path(__file__).resolve().parents[1]
NEXT9 = f"docker:{pins.MILES_K31_NEXT9_IMAGE}"
SWITCHES = dict(mfu_switches.MFU_SWITCHES)


def _args(**kw):
    base = dict(training_mode="rl", rl_backend="miles", rl_model_recipe="kimi_k31", tuning="full",
                rl_image=NEXT9, rl_recipe_env_defaults="auto", island_env=[], modal_env=[],
                rl_sample_role="trainer")
    return SimpleNamespace(**(base | kw))


def _defaults(**kw) -> dict[str, str]:
    env = recipe_env.island_defaults(_args(**kw))
    env.pop(recipe_env.DEFAULTED_ENV, None)
    return env


# --- the list -------------------------------------------------------------------
def test_the_list_is_the_27_measured_switches():
    assert len(mfu_switches.MFU_SWITCHES) == 27 == len(SWITCHES)
    assert SWITCHES["MILES_K3_ATTN_RES_BANK_PAD_ROWS"] == "0"
    assert SWITCHES["MILES_K31_ROUTER_GEMM"] == "tf32"
    assert (SWITCHES["MILES_K31_DSA_QUERY_SPLIT"], SWITCHES["MILES_K31_KDA_OUT_PROJ_RS_CHUNKS"],
            SWITCHES["MILES_K31_KDA_FLA_CP_BLOCK"]) == ("64", "4", "32")
    assert {v for k, v in SWITCHES.items()
            if k not in ("MILES_K3_ATTN_RES_BANK_PAD_ROWS", "MILES_K31_ROUTER_GEMM", "MILES_K31_DSA_QUERY_SPLIT",
                         "MILES_K31_KDA_OUT_PROJ_RS_CHUNKS", "MILES_K31_KDA_FLA_CP_BLOCK")} == {"1"}
    assert mfu_switches.NEXT9_DIGEST.startswith("3a0c5445")


def test_the_guide_table_shows_the_list():
    text = (ROOT / "docs" / "KIMI_K31_ASYNC_RL_GUIDE.md").read_text()
    section = text[text.index("#### 9.1.1"):text.index("### 9.2")]
    rows = dict(re.findall(r"^\| `([A-Z0-9_]+)` \| `([^`]+)` \|", section, flags=re.M))
    assert rows == SWITCHES


def test_the_guide_shows_the_launch_line_as_printed():
    text = (ROOT / "docs" / "KIMI_K31_ASYNC_RL_GUIDE.md").read_text()
    assert recipe_env.launch_line(recipe_env.summary(_args())) in text


# --- when they apply --------------------------------------------------------------
def test_kimi_k31_full_on_next9_gets_every_switch():
    env = recipe_env.island_defaults(_args())
    assert env.pop(recipe_env.DEFAULTED_ENV).split(",") == sorted(SWITCHES)
    assert env == SWITCHES


def test_next9_by_digest_from_any_registry():
    digest = pins.MILES_K31_NEXT9_IMAGE.rpartition("@sha256:")[2]
    assert _defaults(rl_image=f"docker:mirror.example/org/yeto-miles-k31@sha256:{digest}") == SWITCHES


@pytest.mark.parametrize("kw", [
    {"rl_image": f"docker:{pins.MILES_K31_FULL_IMAGE}"},             # the previous full image
    {"rl_image": f"docker:{pins.MILES_K31_FULL_OVERLAYFIX_IMAGE}"},
    {"rl_image": f"docker:{pins.MILES_K31_IMAGE}"},
    {"rl_image": "docker:ghcr.io/x/y:latest"},
    {"rl_image": None},
    {"tuning": "lora"},
    {"rl_model_recipe": "generic"},
    {"rl_model_recipe": "glm5_next"},
    {"rl_model_recipe": "deepseek_v41"},
    {"rl_recipe_env_defaults": "off"},
    {"rl_sample_role": "sampler"},
    {"rl_backend": "verl"},                                          # no recipe_env role
    {"training_mode": "sft"},
], ids=lambda kw: "-".join(f"{k}={v}" for k, v in kw.items())[:60])
def test_nothing_outside_kimi_k31_full_on_next9(kw):
    assert recipe_env.island_defaults(_args(**kw)) == {}


def test_a_bad_mode_is_refused():
    with pytest.raises(ValueError, match="--rl-recipe-env-defaults"):
        recipe_env.island_defaults(_args(rl_recipe_env_defaults="on"))


# --- explicit values win ------------------------------------------------------------
def test_island_env_modal_env_and_the_task_env_win():
    env = recipe_env.island_defaults(
        _args(island_env=["MILES_K31_KDA_SAVE_INTERMEDIATES=0"],
              modal_env=["MILES_K31_INDEXER_RECOMPUTE_CACHE=0"]),
        {"MILES_K3_SITU_TRITON": "0"})
    for name in ("MILES_K31_KDA_SAVE_INTERMEDIATES", "MILES_K31_INDEXER_RECOMPUTE_CACHE", "MILES_K3_SITU_TRITON"):
        assert name not in env
        assert name not in env[recipe_env.DEFAULTED_ENV].split(",")
    assert len(env) == 27 - 3 + 1


def test_summary_line_names_the_set_the_count_and_the_switch_off():
    record = recipe_env.summary(_args(island_env=["MILES_K31_ROUTER_GEMM=fp32"]))
    assert (record["applies"], record["declared"], record["injected"]) == (True, 27, 26)
    assert record["user_set"] == ["MILES_K31_ROUTER_GEMM"]
    assert "MILES_K31_ROUTER_GEMM" not in record["env"]
    line = recipe_env.launch_line(record)
    assert "kimi_k31 next9 trainer MFU switches: 26 of 27" in line
    assert "--rl-recipe-env-defaults off" in line and "9.1.1" in line
    off = recipe_env.summary(_args(rl_recipe_env_defaults="off"))
    assert (off["applies"], off["injected"]) == (False, 0)
    assert "not set (--rl-recipe-env-defaults off)" in recipe_env.launch_line(off)
    old = recipe_env.summary(_args(rl_image=f"docker:{pins.MILES_K31_FULL_IMAGE}"))
    assert not old["applies"] and "not next9" in recipe_env.launch_line(old)
    assert recipe_env.summary(_args(tuning="lora")) is None
    assert recipe_env.summary(_args(rl_model_recipe="generic")) is None


# --- the island paths --------------------------------------------------------------
def _cli(gpu: str, extra=()):
    return parse_cli([
        "--gpu", gpu, "--cluster-prefix", "run", "--model", "org/model", "--data", "org/data",
        "--training-mode", "rl", "--total-steps", "3",
        "--rollout-batch-size", "4", "--n-samples-per-prompt", "2",
        "--rollout-max-response-len", "128", "--local-rl-rounds-per-sync", "1",
        "--reward-function", "pkg.reward:score", "--trust-remote-code", *extra,
    ])


def _k31_task(args, spec):
    # the recipe and its image as a kimi_k31 --tuning full launch resolves them
    args.rl_model_recipe, args.tuning, args.rl_image = "kimi_k31", "full", NEXT9
    factory = lambda *a, **k: SimpleNamespace(run="true", setup="true", envs={"YETO_X": "1"})  # noqa: E731
    return launcher._island_task(factory, args, spec, 0, 1, "1.2.3.4:5000", {})


def test_modal_island_gets_the_defaults_and_modal_env_wins():
    args = _cli("modal:1xh100", ("--modal-env", "MILES_K31_ROUTER_GEMM=fp32",
                                 "--island-env", "MILES_K3_SITU_TRITON=0"))
    _prepare_rl_args(args)
    (spec,) = parse_gpu_spec(args.gpu)
    task = _k31_task(args, spec)
    cfg = launcher.build_modal_island_config(args, spec, 0, task, "1.2.3.4:5000")
    assert cfg.envs["MILES_K31_ROUTER_GEMM"] == "fp32" and cfg.envs["MILES_K3_SITU_TRITON"] == "0"
    for name, value in SWITCHES.items():
        if name not in ("MILES_K31_ROUTER_GEMM", "MILES_K3_SITU_TRITON"):
            assert cfg.envs[name] == value, name
    assert cfg.envs["YETO_X"] == "1"

    args.rl_recipe_env_defaults = "off"
    task = _k31_task(args, spec)
    cfg = launcher.build_modal_island_config(args, spec, 0, task, "1.2.3.4:5000")
    assert not set(SWITCHES) - {"MILES_K31_ROUTER_GEMM", "MILES_K3_SITU_TRITON"} & set(cfg.envs)
    assert recipe_env.DEFAULTED_ENV not in cfg.envs


def test_lora_and_an_old_image_get_nothing_on_the_island():
    args = _cli("aws:1xa100@us-east-1")
    _prepare_rl_args(args)
    (spec,) = parse_gpu_spec(args.gpu)
    factory = lambda *a, **k: SimpleNamespace(run="true", setup="true", envs={})  # noqa: E731
    for recipe, tuning, image in (("kimi_k31", "lora", NEXT9), ("kimi_k31", "full", f"docker:{pins.MILES_K31_FULL_IMAGE}"),
                                  ("generic", "full", NEXT9)):
        args.rl_model_recipe, args.tuning, args.rl_image = recipe, tuning, image
        task = launcher._island_task(factory, args, spec, 0, 1, "1.2.3.4:5000", {})
        assert not set(SWITCHES) & set(task.envs) and recipe_env.DEFAULTED_ENV not in task.envs


def test_ssh_island_env_wins_over_the_defaults_and_the_inventory_takes_back_its_names():
    from tests.test_rl_ssh_resident import b300

    args = _cli("aws:1xa100@us-east-1", ("--island-env", "MILES_K3_SITU_TRITON=0"))
    _prepare_rl_args(args)
    (spec,) = parse_gpu_spec(args.gpu)
    task = _k31_task(args, spec)
    assert task.envs["MILES_K3_SITU_TRITON"] == "0"  # --island-env, after the defaults
    # what island_spec_from_inventory hands the nodes: the task's environment
    inventory = {"MILES_K31_INDEXER_RECOMPUTE_CACHE": "0", "MILES_K3_SITU_TRITON": "1", "NCCL_IB_HCA": "=mlx5_0"}
    ssh = SshTrainerSpec(nodes=(b300(0, env=inventory), b300(1)), run_id="run", workdir="/w",
                         run_script="true", env=dict(task.envs))
    first, second = node_env(ssh, 0, {}), node_env(ssh, 1, {})
    assert first["MILES_K31_INDEXER_RECOMPUTE_CACHE"] == "0"  # the inventory's own value
    assert first["MILES_K3_SITU_TRITON"] == "0"  # --island-env still wins over the inventory
    assert second["MILES_K31_INDEXER_RECOMPUTE_CACHE"] == "1"
    for name, value in SWITCHES.items():
        if name not in ("MILES_K31_INDEXER_RECOMPUTE_CACHE", "MILES_K3_SITU_TRITON"):
            assert first[name] == value == second[name], name


# --- memory estimate --------------------------------------------------------------
def _layout(**kw):
    return _args(**({"tensor_parallel": 8, "pipeline_parallel": 3, "expert_parallel": 16,
                      "context_parallel": 2, "seq_len": 262144, "rl_max_tokens_per_gpu": 131072,
                      "rollout_max_response_len": 245760, "rl_optimizer_offload_fraction": 1.0,
                      "rl_precision_aware_optimizer": False} | kw))


def test_the_memory_estimate_counts_the_switches_the_trainer_runs_with():
    on = k31_checks.memory_estimate(_layout(), num_trainable=128, moe_layers=92, gpu="B300")
    off = k31_checks.memory_estimate(_layout(rl_recipe_env_defaults="off"), num_trainable=128,
                                     moe_layers=92, gpu="B300")
    assert off["mfu_switches_gib"] == {} and off["parts_gib"]["switches"] == 0
    terms = on["mfu_switches_gib"]
    assert terms["indexer_topk_cache"] > 0 and terms["kda_intermediates"] == 3.6
    assert terms["situ_saved"] == -18.0 and terms["frozen_redequant"] < 0
    # stage 0 holds 7 DSA layers x 3 in-flight microbatches of 128 MiB top-k parts
    s0_on, s0_off = on["stages_gib"][0], off["stages_gib"][0]
    assert s0_on["switches"] == pytest.approx(7 * 3 * 0.125 + 3.6 - 18.0 - 2.95, abs=0.1)
    assert s0_on["peak_alloc"] == pytest.approx(s0_off["peak_alloc"] + s0_on["switches"], abs=0.1)
    # the recommended 31/31/31 layout stays inside the no-expandable-segments budget
    assert on["pipeline_split"] == [31, 31, 31]
    assert on["per_gpu_gib"] < on["budget_gib"]["no_expandable_segments"]
    # a switch the launch turns off is not counted
    one_off = k31_checks.memory_estimate(_layout(island_env=["MILES_K31_KDA_SAVE_INTERMEDIATES=0"]),
                                         num_trainable=128, moe_layers=92, gpu="B300")
    assert "kda_intermediates" not in one_off["mfu_switches_gib"]


def test_the_switch_terms_stay_above_the_measured_16_layer_peak():
    # mfu-16l a0b (no switches) 98.92 GiB and x9 (every switch) 73.65 GiB allocated, PP1, 16 layers
    on = k31_checks.memory_estimate(_layout(pipeline_parallel=1), num_trainable=128, moe_layers=15, gpu="B300")
    off = k31_checks.memory_estimate(_layout(pipeline_parallel=1, rl_recipe_env_defaults="off"),
                                     num_trainable=128, moe_layers=15, gpu="B300")
    assert off["per_gpu_gib"] == pytest.approx(98.92, rel=0.06)
    assert 73.65 <= on["per_gpu_gib"] <= 73.65 * 1.2
