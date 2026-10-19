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


def options_type(cls: type) -> type | None:
    """The options dataclass an adapter's ``__init__(config, options)`` takes."""
    hint = typing.get_type_hints(cls.__init__).get("options")
    found = [m for m in (typing.get_args(hint) or (hint,)) if dataclasses.is_dataclass(m)]
    return found[0] if found else None


def build(section: Mapping[str, Any]) -> Any:
    """The adapter of one ``engine`` section (module doc). Raises ValueError /
    KeyError / TypeError on a section the adapter does not accept."""
    cls = engine_class(str(section["adapter"]))
    if "server" not in section:
        return cls(**dict(section.get("kwargs") or {}))
    server = dict(section["server"])
    server["base_dir"] = Path(server["base_dir"])
    server["extra_args"] = tuple(server.get("extra_args") or ())
    options = dict(section.get("options") or {})
    options_cls = options_type(cls)
    if options_cls is None:
        if options:
            raise ValueError(f"adapter {section['adapter']!r} takes no options, the section gives {sorted(options)}")
        return cls(ServerConfig(**server))
    # JSON has no tuples; the options dataclasses hold tuples
    options = {k: tuple(v) if isinstance(v, list) else v for k, v in options.items()}
    return cls(ServerConfig(**server), options_cls(**options))
