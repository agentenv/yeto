"""decoupling 4.9: neutral determinism table; Miles x NVIDIA unchanged."""

import pytest

from yeto.rl.engine.determinism import DeterminismNotCalibrated, determinism_env

# Value before the move (miles_adapter/entry.py, s16-decouple-p2).
PRE_MOVE = {"NCCL_ALGO": "Ring", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "NVIDIA_TF32_OVERRIDE": "0", "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0"}


def test_miles_nvidia_env_is_unchanged():
    from yeto.rl.adapters.miles.entry import DETERMINISM_ENV

    assert DETERMINISM_ENV == PRE_MOVE
    assert list(DETERMINISM_ENV) == list(PRE_MOVE)
    assert determinism_env() == PRE_MOVE


def test_unmeasured_rows_are_refused():
    for backend, family in (("miles", "ascend"), ("verl", "nvidia")):
        with pytest.raises(DeterminismNotCalibrated):
            determinism_env(backend, family)


def test_neutral_flag_is_an_alias():
    import argparse

    from yeto.cli import _add_launch_args

    parser = argparse.ArgumentParser()
    _add_launch_args(parser)
    for flag in ("--deterministic", "--rl-deterministic-trainer"):
        args = parser.parse_args(["--model", "qwen35-9b", "--data", "d", flag])
        assert args.rl_deterministic_trainer is True
    assert parser.parse_args(["--model", "qwen35-9b", "--data", "d"]).rl_deterministic_trainer is False
