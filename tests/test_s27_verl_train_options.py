"""S27 G-EC: the Miles training and observation options a verl full-parameter
island maps onto verl (``yeto.rl.adapters.verl.train_options``), and the ones
that stay refused before any cloud work, each with its reason.

CPU only.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from test_s27_verl_full_parameter_launch import MILES_ONLY, _flags, _island_argv, _prepared, _verl_args
from yeto.rl.adapters.verl import config as vconf
from yeto.rl.adapters.verl import full_parameter_stream as vfs
from yeto.rl.adapters.verl import island_entry, launch_flags
from yeto.rl.adapters.verl import train_options as to
from yeto.rl.adapters.verl.driver_extras import wire_full_parameter_extras

MAPPED = {
    ("--rl-lr-schedule", "constant"): {to.LR_SCHEDULER_KEY: "constant"},
    ("--rl-max-tokens-per-gpu", "4096"): {to.ACTOR_TOKENS_KEY: "4096", to.LOG_PROB_TOKENS_KEY: "4096"},
    ("--rl-activation-recompute",): {to.CHECKPOINTING_KEY: "True"},
}
OBSERVED = {
    ("--rl-observe-timeline",): {"observe_timeline": True, "heartbeat_interval_s": 30.0,
                                 "resource_sample_interval_s": 60.0},
    ("--rl-resource-sample-interval", "5"): {"resource_sample_interval_s": 5.0},
    ("--rl-observe-timeline", "--rl-resource-sample-interval", "0"): {"observe_timeline": True,
                                                                      "heartbeat_interval_s": 30.0},
}


def _run():
    return vconf.VerlRunConfig(model_path="/m", train_file="/t", val_file="/v", out_dir="/o",
                               parameter_mode="full", lora_rank=0, lora_alpha=0, max_prompt_len=512,
                               max_response_len=1024, correction="none")


def _island(extra) -> SimpleNamespace:
    """The verl island's parsed args for a launch with ``extra``; checks that
    nothing but the Miles-only options was ignored on the way."""
    argv = _island_argv(_prepared(_verl_args(*extra)), 0)
    parsed, ignored = island_entry.parse_args(argv)
    assert _flags(ignored) <= MILES_ONLY, _flags(ignored) - MILES_ONLY
    return parsed


def _override_map(overrides) -> dict[str, str]:
    return dict(item.split("=", 1) for item in overrides)


@pytest.mark.parametrize("extra", list(MAPPED), ids=lambda extra: extra[0])
def test_a_mapped_training_option_reaches_the_verl_overrides_and_is_asserted(extra):
    parsed = _island(extra)
    base = vconf.build_overrides(_run())
    overrides, asserted = to.apply_training_options(parsed, base, vconf.ASSERTED_KEYS, max_prompt_len=512,
                                                    max_response_len=1024)
    got = _override_map(overrides)
    for key, value in MAPPED[extra].items():
        assert got[key] == value and key in asserted, key
    # Only the mapped keys change; a key the island already set is replaced, not repeated.
    assert len(overrides) == len(base) + (extra[0] == "--rl-lr-schedule")
    assert {k: v for k, v in got.items() if k not in MAPPED[extra]} == {
        k: v for k, v in _override_map(base).items() if k not in MAPPED[extra]}


def test_without_the_options_nothing_changes():
    parsed = _island(())
    base = vconf.build_overrides(_run())
    assert to.apply_training_options(parsed, base, vconf.ASSERTED_KEYS, max_prompt_len=512,
                                     max_response_len=1024) == (base, tuple(vconf.ASSERTED_KEYS))
    assert to.observation_plan(parsed) == {}


@pytest.mark.parametrize("extra", list(OBSERVED), ids=lambda extra: "-".join(extra))
def test_an_observation_option_reaches_the_plan_and_the_driver(extra, tmp_path):
    parsed = _island(extra)
    (tmp_path / "config.json").write_text('{"model_type": "qwen3"}')
    plan = vfs.full_parameter_extras(parsed, sync="pool")
    assert {k: plan[k] for k in OBSERVED[extra]} == OBSERVED[extra]
    driver = SimpleNamespace(observe=False, heartbeat_interval_s=None, resource_sample_interval_s=None)
    wire_full_parameter_extras(driver, SimpleNamespace(full_group=None), {**plan, "learner_id": 0})
    assert driver.observe is bool(OBSERVED[extra].get("observe_timeline"))
    assert driver.heartbeat_interval_s == OBSERVED[extra].get("heartbeat_interval_s")
    assert driver.resource_sample_interval_s == OBSERVED[extra].get("resource_sample_interval_s")


def test_values_verl_cannot_run_are_refused():
    with pytest.raises(ValueError, match="constant and cosine"):
        _prepared(_verl_args("--rl-lr-schedule", "linear", "--rl-sync-full-parameter-rounds", "sweep"))
    with pytest.raises(ValueError, match="只接 --rl-lr-schedule constant"):
        to.apply_training_options(SimpleNamespace(rl_lr_schedule="linear"), [], (), max_prompt_len=1,
                                  max_response_len=1)
    with pytest.raises(ValueError, match="放不下一条样本"):
        to.apply_training_options(SimpleNamespace(rl_max_tokens_per_gpu=1000), [], (), max_prompt_len=512,
                                  max_response_len=1024)
    with pytest.raises(ValueError, match=">= 0"):
        to.observation_plan(SimpleNamespace(rl_resource_sample_interval=-1.0))


@pytest.mark.parametrize("extra", [
    ("--rl-verify-every-round",), ("--rl-deterministic-trainer",), ("--rl-gradient-accumulation-fusion",),
    ("--rl-async-buffer-rounds", "2"), ("--rl-lora-dropout", "0.1"), ("--rl-lora-expert-rank", "4"),
    ("--rl-profile-trainer",), ("--rl-debug-dump",),
], ids=lambda extra: extra[0])
def test_the_options_without_a_verl_mapping_stay_refused_with_their_reason(extra):
    with pytest.raises(ValueError, match="would ignore it") as error:
        _prepared(_verl_args(*extra))
    assert launch_flags.NOT_MAPPED_REASONS[extra[0]] in str(error.value)


def test_every_refused_option_has_a_reason_and_no_mapped_option_is_refused():
    refused = {option for _dest, _off, option in launch_flags.IGNORED_ON_VERL_FULL_PARAMETER}
    assert refused == set(launch_flags.NOT_MAPPED_REASONS)
    mapped = {extra[0] for extra in (*MAPPED, *OBSERVED)}
    assert not refused & mapped
