"""Engine adapters from a sampler spec's ``engine`` section (the engine sampler
process entry, :mod:`yeto.rl.engine.engine_sampler_entry`, calls :func:`build`
through ``engine.factory``).

Section forms:

* ``{"adapter": NAME, "server": {...}, "options": {...}}``: an HTTP-server
  adapter; ``server`` = :class:`~.server.ServerConfig` fields, ``options`` = the
  adapter's options dataclass (``SGLangOptions``, ``VllmOptions``);
* ``{"adapter": NAME, "kwargs": {...}}``: an adapter built from keywords (the
  CPU ``fake`` engine).

``NAME`` is a key of :data:`~yeto.rl.adapters.sampler_engines.ENGINE_ADAPTERS`.

``rollout_fields`` (the entry copies the spec's neutral section into the engine
section): ``{"routed_experts": bool, "top_logprobs_num": int, "fp32_lm_head": bool}``
becomes the adapter's own options (:func:`apply_rollout_fields`); an adapter
that cannot return a requested field is refused here, before any process starts.
``platform`` (``cuda`` / ``rocm`` / ``ppu`` / ``npu``, copied the same way) becomes
``options.platform`` of ``sglang-server`` (the ROCm environment preset and card
check) and of ``vllm-server`` (``cuda`` / ``rocm``: upstream vLLM main with R3, the
fp32 head and the ROCm preset; ``npu``: vllm-ascend; no ``ppu``). Without a
``platform`` a ``vllm-server`` keeps the legacy line (vllm-ascend / the verl image),
which returns neither R3 nor an fp32 head.
"""

from __future__ import annotations

import dataclasses
import typing
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import engine_class
from .server import ServerConfig

FACTORY = "yeto.rl.adapters.sampler_engines.from_spec:build"  # value of the spec's engine.factory
ROLLOUT_FIELDS = "rollout_fields"
PLATFORM = "platform"
SECTION_KEYS = frozenset({"factory", "adapter", "server", "options", "kwargs", ROLLOUT_FIELDS, PLATFORM})
DEFAULT_TOP_LOGPROBS_NUM = 128  # K of score_centering (--rollout-top-logprobs-num)
FAKE_ROUTED_EXPERTS = (896, 2, 8)  # the fake engine's R3 shape when the spec only says "true"
VLLM_PLATFORMS = ("cuda", "rocm", "npu")  # vllm_server.PLATFORMS (no import: adapters load lazily)
VLLM_UPSTREAM_PLATFORMS = ("cuda", "rocm")  # vllm_server.UPSTREAM_PLATFORMS


def rollout_fields(section: Mapping[str, Any] | None) -> dict[str, Any]:
    """The normalized ``rollout_fields`` section: every key present, K defaulting
    to :data:`DEFAULT_TOP_LOGPROBS_NUM` when the section exists. None / absent
    section -> all off (the engine's own options decide)."""
    if not section:
        return {"routed_experts": False, "top_logprobs_num": 0, "fp32_lm_head": False}
    unknown = set(section) - {"routed_experts", "top_logprobs_num", "fp32_lm_head"}
    if unknown:
        raise ValueError(f"{ROLLOUT_FIELDS}: unknown keys {sorted(unknown)}")
    k = section.get("top_logprobs_num", DEFAULT_TOP_LOGPROBS_NUM)
    if k is None:
        k = 0
    if not isinstance(k, int) or isinstance(k, bool) or k < 0:
        raise ValueError(f"{ROLLOUT_FIELDS}.top_logprobs_num must be an integer >= 0, got {k!r}")
    return {"routed_experts": bool(section.get("routed_experts", False)), "top_logprobs_num": int(k),
            "fp32_lm_head": bool(section.get("fp32_lm_head", False))}


def apply_rollout_fields(adapter: str, fields: Mapping[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    """``options`` (the adapter's options / kwargs mapping) with the rollout
    fields the adapter supports set; a field the adapter cannot return raises
    ValueError. An option the section already names explicitly must agree."""
    wants = rollout_fields(fields)
    if adapter == "sglang-server":
        mapping = {"routed_experts": "return_routed_experts", "top_logprobs_num": "top_logprobs_num",
                   "fp32_lm_head": "fp32_lm_head"}
    elif adapter == "vllm-server":
        upstream = options.get("platform") in VLLM_UPSTREAM_PLATFORMS
        if wants["routed_experts"] and not upstream:
            raise ValueError(f"{ROLLOUT_FIELDS}.routed_experts: vllm-server returns routed experts (R3) only on "
                             f"upstream vLLM (platform cuda / rocm), not on {options.get('platform')!r}")
        if wants["fp32_lm_head"] and not upstream:
            raise ValueError(f"{ROLLOUT_FIELDS}.fp32_lm_head: vllm-server has an fp32 lm-head switch only on "
                             f"upstream vLLM (platform cuda / rocm), not on {options.get('platform')!r}")
        mapping = {"top_logprobs_num": "top_logprobs_num"}
        if upstream:
            mapping = {"routed_experts": "return_routed_experts", "top_logprobs_num": "top_logprobs_num",
                       "fp32_lm_head": "fp32_lm_head"}
    elif adapter == "fake":
        mapping = {"top_logprobs_num": "top_logprobs_num", "fp32_lm_head": "fp32_lm_head"}
        if wants["routed_experts"] and options.get("routed_experts") is None:
            options = {**options, "routed_experts": FAKE_ROUTED_EXPERTS}
    else:
        if any(wants.values()):
            raise ValueError(f"{ROLLOUT_FIELDS}: adapter {adapter!r} does not take rollout fields")
        return dict(options)
    out = dict(options)
    for key, option in mapping.items():
        if option in out and out[option] != wants[key]:
            raise ValueError(f"{ROLLOUT_FIELDS}.{key} = {wants[key]!r} but options.{option} = {out[option]!r}")
        out[option] = wants[key]
    return out


def apply_platform(adapter: str, platform: Any, options: dict[str, Any]) -> dict[str, Any]:
    """``options`` with the spec's ``platform`` applied: ``sglang-server`` and
    ``vllm-server`` take it as ``options.platform`` (``vllm-server``: ``cuda`` /
    ``rocm`` upstream vLLM, ``npu`` vllm-ascend; no ``ppu``); the fake engine
    ignores it."""
    platform = str(platform)
    if adapter == "vllm-server" and platform not in VLLM_PLATFORMS:
        raise ValueError(f"{PLATFORM} = {platform!r}: vllm-server runs on {' / '.join(VLLM_PLATFORMS)} only "
                         "(cuda / rocm: upstream vLLM; npu: vllm-ascend)")
    if adapter in ("sglang-server", "vllm-server"):
        if options.get("platform") is not None and options["platform"] != platform:
            raise ValueError(f"{PLATFORM} = {platform!r} but options.platform = {options['platform']!r}")
        return {**options, "platform": platform}
    return dict(options)


def options_type(cls: type) -> type | None:
    """The options dataclass an adapter's ``__init__(config, options)`` takes."""
    hint = typing.get_type_hints(cls.__init__).get("options")
    found = [m for m in (typing.get_args(hint) or (hint,)) if dataclasses.is_dataclass(m)]
    return found[0] if found else None


def build(section: Mapping[str, Any]) -> Any:
    """The adapter of one ``engine`` section (module doc). Raises ValueError /
    KeyError / TypeError on a section the adapter does not accept."""
    unknown = set(section) - SECTION_KEYS
    if unknown:
        raise ValueError(f"engine section: unknown keys {sorted(unknown)} (known: {sorted(SECTION_KEYS)})")
    if "server" in section and "kwargs" in section:
        raise ValueError("engine section: 'kwargs' is for adapters without a server; use 'options'")
    if "server" not in section and "options" in section:
        raise ValueError("engine section: 'options' needs a 'server' section; a keyword adapter takes 'kwargs'")
    adapter = str(section["adapter"])
    cls = engine_class(adapter)
    fields = section.get(ROLLOUT_FIELDS)
    platform = section.get(PLATFORM)
    if "server" not in section:
        kwargs = dict(section.get("kwargs") or {})
        if platform is not None:
            kwargs = apply_platform(adapter, platform, kwargs)
        if fields is not None:
            kwargs = apply_rollout_fields(adapter, fields, kwargs)
            if isinstance(kwargs.get("routed_experts"), list):
                kwargs["routed_experts"] = tuple(kwargs["routed_experts"])
        return cls(**kwargs)
    server = dict(section["server"])
    server["base_dir"] = Path(server["base_dir"])
    server["extra_args"] = tuple(server.get("extra_args") or ())
    options = dict(section.get("options") or {})
    if platform is not None:
        options = apply_platform(adapter, platform, options)
    if fields is not None:
        options = apply_rollout_fields(adapter, fields, options)
    options_cls = options_type(cls)
    if options_cls is None:
        if options:
            raise ValueError(f"adapter {section['adapter']!r} takes no options, the section gives {sorted(options)}")
        return cls(ServerConfig(**server))
    # JSON has no tuples; the options dataclasses hold tuples
    options = {k: tuple(v) if isinstance(v, list) else v for k, v in options.items()}
    return cls(ServerConfig(**server), options_cls(**options))
