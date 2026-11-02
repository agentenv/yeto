"""The measurement-only environment switches of the verl Megatron island."""

from __future__ import annotations

import pytest

from yeto.rl.adapters.verl import megatron_engine as me

TCFG = "++actor_rollout_ref.actor.megatron.override_transformer_config"


def test_parse_etp_default_and_value():
    assert me.parse_etp({}) == 1
    assert me.parse_etp({me.ETP_ENV: " 2 "}) == 2


def test_parse_etp_refuses_a_non_integer_with_a_clear_message():
    with pytest.raises(me.ExperimentEnvError, match="YETO_VERL_MEGATRON_ETP must be an integer.*'two'"):
        me.parse_etp({me.ETP_ENV: "two"})


def test_switches_are_logged_and_returned():
    lines: list[str] = []
    env = {me.ALLOW_ETP_ENV: "1", me.TCFG_EXTRA_ENV: "a=1", "OTHER": "x"}
    found = me.check_experiment_env(releases=False, environ=env, log=lines.append)
    assert found == {me.ALLOW_ETP_ENV: "1", me.TCFG_EXTRA_ENV: "a=1"}
    assert len(lines) == 2 and all("WARNING" in line for line in lines)
    assert me.check_experiment_env(releases=True, environ={}, log=lines.append) == {}


@pytest.mark.parametrize("name", [me.ALLOW_ETP_ENV, me.TCFG_EXTRA_ENV])
def test_release_run_refuses_the_risky_switches(name):
    with pytest.raises(me.ExperimentEnvError, match="--rl-measure-no-release"):
        me.check_experiment_env(releases=True, environ={name: "1"}, log=lambda line: None)


def test_etp_alone_is_allowed_on_a_release_run():
    assert me.check_experiment_env(releases=True, environ={me.ETP_ENV: "1"},
                                   log=lambda line: None) == {me.ETP_ENV: "1"}


def test_extra_overrides_pass_through():
    env = {me.TCFG_EXTRA_ENV: "moe_pad_expert_input_to_capacity=True, moe_expert_capacity_factor=1.25"}
    assert me._extra_tcfg_overrides(TCFG, environ=env) == [
        f"{TCFG}.moe_pad_expert_input_to_capacity=True", f"{TCFG}.moe_expert_capacity_factor=1.25"]
    assert me._extra_tcfg_overrides(TCFG, environ={}) == []


@pytest.mark.parametrize("key", ["moe_router_bias_update_rate", "moe_aux_loss_coeff"])
def test_extra_overrides_refuse_rl_fixed_keys(key):
    with pytest.raises(me.ExperimentEnvError, match=f"must not override {key}"):
        me._extra_tcfg_overrides(TCFG, environ={me.TCFG_EXTRA_ENV: f"{key}=0.1"})


def test_extra_overrides_refuse_asserted_and_written_keys():
    asserted = [k for k in me.ASSERTED_KEYS if ".override_transformer_config." in k]
    assert asserted  # gradient_accumulation_fusion
    key = asserted[0].rpartition(".")[2]
    with pytest.raises(me.ExperimentEnvError, match=key):
        me._extra_tcfg_overrides(TCFG, environ={me.TCFG_EXTRA_ENV: f"{key}=True"})
    with pytest.raises(me.ExperimentEnvError, match="moe_router_dtype"):
        me._extra_tcfg_overrides(TCFG, written=[f"{TCFG}.moe_router_dtype=fp32"],
                                 environ={me.TCFG_EXTRA_ENV: "moe_router_dtype=fp16"})


@pytest.mark.parametrize("raw", ["recompute_modules=[core_attn,mlp]", "novalue", "a="])
def test_extra_overrides_refuse_a_comma_inside_a_value(raw):
    with pytest.raises(me.ExperimentEnvError, match="not key=value"):
        me._extra_tcfg_overrides(TCFG, environ={me.TCFG_EXTRA_ENV: raw})


@pytest.mark.parametrize("name", [me.ETP_ENV, me.ALLOW_ETP_ENV, me.TCFG_EXTRA_ENV])
def test_switches_have_new_names_and_a_deprecated_fallback(name):
    legacy = me.LEGACY_EXPERIMENT_ENVS[name]
    assert name.startswith("YETO_VERL_MEGATRON_") and legacy.startswith("YETO_S29_")
    with pytest.warns(DeprecationWarning, match=f"{legacy} is deprecated; set {name}"):
        assert me.experiment_env_value(name, {legacy: "1"}) == "1"
    with pytest.warns(DeprecationWarning):
        assert me.experiment_env({legacy: "1"}) == {name: "1"}  # recorded under the new name


def test_new_name_wins_over_the_deprecated_one_without_a_warning():
    import warnings

    env = {me.ETP_ENV: "2", me.LEGACY_EXPERIMENT_ENVS[me.ETP_ENV]: "4"}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert me.parse_etp(env) == 2
        assert me.experiment_env_value(me.TCFG_EXTRA_ENV, {}) == ""


def test_deprecated_names_reach_parse_etp_and_tcfg_extra():
    with pytest.warns(DeprecationWarning):
        assert me.parse_etp({"YETO_S29_ETP": "2"}) == 2
    with pytest.warns(DeprecationWarning):
        assert me._extra_tcfg_overrides(TCFG, environ={"YETO_S29_TCFG_EXTRA": "a=1"}) == [f"{TCFG}.a=1"]
    with pytest.warns(DeprecationWarning), pytest.raises(me.ExperimentEnvError, match="--rl-measure-no-release"):
        me.check_experiment_env(releases=True, environ={"YETO_S29_ALLOW_ETP": "1"}, log=lambda line: None)


def test_deprecated_allow_etp_still_admits_an_etp_shape(monkeypatch):
    monkeypatch.delenv(me.ALLOW_ETP_ENV, raising=False)
    monkeypatch.setenv("YETO_S29_ALLOW_ETP", "1")
    with pytest.warns(DeprecationWarning):
        assert me.experiment_env_value(me.ALLOW_ETP_ENV) == "1"
