"""S28 G-M2 (W2, W3): the vLLM engine environment of a verl island on Ascend."""

from __future__ import annotations

import configparser
import importlib
import json
import os

from yeto.rl.adapters.verl import ascend_env as ae


def test_defaults_and_plugins_for_a_dense_model():
    env = {}
    notes = ae.island_env(env, mamba2=False)
    assert env == {"VLLM_WORKER_MULTIPROC_METHOD": "spawn", "TASK_QUEUE_ENABLE": "1", "VLLM_PLUGINS": "ascend"}
    assert len(notes) == 3


def test_caller_values_are_kept_but_broken_ones_corrected():
    env = {"VLLM_WORKER_MULTIPROC_METHOD": "fork", "TASK_QUEUE_ENABLE": "2",
           "VLLM_PLUGINS": "ascend_service_profiling,other"}
    ae.island_env(env, mamba2=False)
    assert env["VLLM_WORKER_MULTIPROC_METHOD"] == "fork"
    assert env["TASK_QUEUE_ENABLE"] == "1"
    assert env["VLLM_PLUGINS"] == "other,ascend"


def test_mamba2_model_gets_the_plugin_entry_point(tmp_path):
    env = {"PYTHONPATH": "/a"}
    ae.island_env(env, mamba2=True, plugin_dir=tmp_path / "p")
    assert env["VLLM_PLUGINS"] == "ascend,yeto_ascend_mamba2"
    assert env["PYTHONPATH"] == os.pathsep.join(["/a", str(tmp_path / "p")])
    cp = configparser.ConfigParser()
    cp.read(tmp_path / "p" / ae.DIST_INFO / "entry_points.txt")
    mod, fn = cp["vllm.general_plugins"][ae.MAMBA2_PLUGIN].split(":")
    assert callable(getattr(importlib.import_module(mod), fn))
    ae.island_env(env, mamba2=True, plugin_dir=tmp_path / "p")  # idempotent
    assert env["PYTHONPATH"].count(str(tmp_path / "p")) == 1


def test_is_mamba2_model(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "nemotron_h", "hybrid_override_pattern": "MEM*E"}))
    assert ae.is_mamba2_model(str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    assert not ae.is_mamba2_model(str(tmp_path))
    assert not ae.is_mamba2_model(str(tmp_path / "missing"))


def test_safe_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir("/")
    assert ae.safe_cwd(tmp_path / "o") == str(tmp_path / "o")
    assert os.getcwd() == str(tmp_path / "o")
    assert ae.safe_cwd(tmp_path) is None


def test_verl_main_forwards_the_engine_names():
    src = open(os.path.join(os.path.dirname(ae.__file__), "verl_main.py")).read()
    for name in ae.FORWARD_NAMES:
        assert f'"{name}"' in src.split("FORWARD_ENV_NAMES = ", 1)[1].split("\n", 1)[0]


def test_bringup_reward_is_mapped_and_varies():
    from yeto.rl.adapters.verl import reward_fn

    name = reward_fn.function_for("yeto.rl.bringup_reward:score")
    f = getattr(reward_fn, name)
    assert f("d", "abc d") == 1.0 and f("d", "") == 0.0
    assert 0.0 < f("d", "ab12") < 1.0


def test_rollout_island_drops_expandable_segments_buffer_trainer_keeps_it():
    env = {"PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True"}
    ae.island_env(env, mamba2=False, rollout=True)
    assert env["PYTORCH_NPU_ALLOC_CONF"] == "expandable_segments:False"
    env = {"PYTORCH_NPU_ALLOC_CONF": "max_split_size_mb:256,expandable_segments:True"}
    ae.island_env(env, mamba2=False, rollout=True)
    assert env["PYTORCH_NPU_ALLOC_CONF"] == "max_split_size_mb:256"
    env = {"PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True"}
    ae.island_env(env, mamba2=False, rollout=False)
    assert env["PYTORCH_NPU_ALLOC_CONF"] == "expandable_segments:True"


def test_rollout_overrides_turn_prefix_caching_off_for_mamba2_on_ascend(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "nemotron_h", "hybrid_override_pattern": "M*E"}))
    assert ae.rollout_overrides(str(tmp_path), "ascend") == ["actor_rollout_ref.rollout.enable_prefix_caching=False"]
    assert ae.rollout_overrides(str(tmp_path), "nvidia") == []
