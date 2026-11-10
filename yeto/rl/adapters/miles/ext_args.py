"""yeto's extension context on Miles' configuration (``yeto_ext``): adapter entry point.

The implementation is engine-neutral (:mod:`yeto.rl.engine.ext_attrs`, so the
core may use it too); this module is the name the Miles-side workflows import:

    from yeto.rl.adapters.miles.ext_args import ext_attr, install_ext
    install_ext(all_config, {"rl_sample_buffer": ..., ...})   # cross-process keys, once, before slicing
    ext_attr(args, "rl_sample_buffer", default=None)           # yeto_ext[key] -> args.yeto_<key> -> default
"""

from __future__ import annotations

from yeto.rl.engine.ext_attrs import (
    ATTR_PREFIX,
    EXT_FIELD,
    WIRE_KEYS,
    attr_name,
    build_ext,
    ext_attr,
    ext_has,
    ext_key,
    install_ext,
    iter_ext_keys,
)

__all__ = [
    "ATTR_PREFIX",
    "EXT_FIELD",
    "WIRE_KEYS",
    "attr_name",
    "build_ext",
    "ext_attr",
    "ext_has",
    "ext_key",
    "install_ext",
    "iter_ext_keys",
]
