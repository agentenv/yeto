"""Attested adapter from a Yeto reward callable to Miles' batch RM API."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import math
import os
from pathlib import Path

_FUNCTION_ENV = "YETO_RL_REWARD_FUNCTION"
_SHA_ENV = "YETO_RL_REWARD_SHA256"
_WORKDIR_ENV = "YETO_RL_WORKDIR"


def _load_callable(
    spec: str,
    expected_sha256: str,
    workdir: str | Path,
    *,
    label: str = "RL reward",
):
    from ..provenance import python_spec_path

    source = python_spec_path(spec, base_dir=Path(workdir))
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(
            f"{label} source SHA256 mismatch: expected {expected_sha256}, got {actual}"
        )
    module_name, separator, function_name = spec.partition(":")
    if not separator:
        raise ValueError(f"{label} callable must be package.module:function")
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"RL reward target {spec!r} is not callable")
    return function


def validate_callable_source(
    spec: str,
    expected_sha256: str,
    workdir: str | Path,
    *,
    label: str,
) -> None:
    _load_callable(spec, expected_sha256, workdir, label=label)


async def miles_reward(args, samples):
    """Invoke the configured callable and enforce Miles' batch reward contract."""

    spec = os.environ.get(_FUNCTION_ENV)
    expected = os.environ.get(_SHA_ENV)
    workdir = os.environ.get(_WORKDIR_ENV)
    if not spec or not expected or not workdir:
        raise RuntimeError("Yeto RL reward environment is incomplete")
    function = _load_callable(spec, expected, workdir)
    result = function(args, samples)
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, (list, tuple)) or len(result) != len(samples):
        raise RuntimeError(
            f"RL reward callable returned {len(result) if isinstance(result, (list, tuple)) else 0} "
            f"rewards for {len(samples)} samples"
        )
    values = [float(value) for value in result]
    if any(not math.isfinite(value) for value in values):
        raise RuntimeError("RL reward callable returned NaN or Inf")
    return values


def configure_reward_environment(
    spec: str, expected_sha256: str, workdir: str | Path
) -> None:
    workdir = Path(workdir).resolve()
    validate_callable_source(
        spec,
        expected_sha256,
        workdir,
        label="RL reward",
    )
    os.environ[_FUNCTION_ENV] = spec
    os.environ[_SHA_ENV] = expected_sha256
    os.environ[_WORKDIR_ENV] = str(workdir)
