"""Sampler engine adapters: implementations of the neutral
:class:`~yeto.rl.engine.sampler_engine.SamplerEngine` port (rl-hetero-sampler-engines D1).

* ``sglang-server`` (:mod:`.sglang_server`): an SGLang HTTP server, PPU stage 1;
* ``vllm-server`` (:mod:`.vllm_server`): a vLLM OpenAI-compatible server
  (vllm-ascend on 910B), stage 2;
* ``fake`` (:mod:`.fake`): deterministic in-memory engine for CPU tests.

Vendor facts (patches, environment variables, launch flags, weight-name maps)
live only here. This package does not import another backend adapter.
"""

from __future__ import annotations

import importlib
from typing import Any

# adapter name -> "module:class"; imported on use so that a host without httpx
# or the engine package can still import this package.
ENGINE_ADAPTERS = {
    "sglang-server": "yeto.rl.adapters.sampler_engines.sglang_server:SGLangServerEngine",
    "vllm-server": "yeto.rl.adapters.sampler_engines.vllm_server:VllmServerEngine",
    "fake": "yeto.rl.adapters.sampler_engines.fake:FakeEngine",
}


def engine_class(name: str) -> Any:
    """The adapter class registered under ``name``; unknown -> ValueError."""
    try:
        target = ENGINE_ADAPTERS[name]
    except KeyError:
        raise ValueError(f"unknown sampler engine {name!r} (known: {sorted(ENGINE_ADAPTERS)})") from None
    module, _, attr = target.partition(":")
    return getattr(importlib.import_module(module), attr)
