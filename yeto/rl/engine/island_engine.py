"""Per-island engine image and training backend (hetero-1114 A22).

``--rl-island-engine ISLAND:KEY=VALUE`` (repeatable) gives one trainer island
its own engine image or training backend; every other island keeps the
run-wide ``--rl-image`` / ``--rl-backend``.

===========  ==================  ==========================================
key          run-wide flag       value
===========  ==================  ==========================================
``image``    ``--rl-image``      the island's image (the forms --rl-image
                                 takes for that island's backend)
``backend``  ``--rl-backend``    a registered backend (miles | verl)
``platform`` (none)              the island's device family (cuda | rocm |
                                 ascend | ppu); without ``image`` the island
                                 gets that family's trainer image for its
                                 backend (``engine_images.DEFAULT_IMAGES``)
===========  ==================  ==========================================

An island whose ``backend`` differs from the run-wide one and that names no
``image`` gets that backend's default image (the same default an omitted
``--rl-image`` gets). A launch without the flag is unchanged: :func:`island_args`
returns ``args`` itself, so every island command, image and record stays
byte-identical.

An island of a different backend shares the syncer session only through a
layout both backends export the same way; the launch refuses it without
``--rl-sync-layout shape-free`` (the shape-free layout is the HF parameter
names and shapes). The syncer's identity check still refuses a pair that does
not match at connect time.

This module is pure (no launcher import, no cloud work).
"""

from __future__ import annotations

import copy
import re
import sys
from collections.abc import Iterable
from typing import Any

OVERRIDES_DEST = "rl_island_engine"
FLAG = "--rl-island-engine"
_DIGEST_IMAGE = re.compile(r"docker:[^\s@]+@sha256:[0-9a-fA-F]{64}")


def _backend(value: str) -> str:
    from . import backends

    value = value.strip()
    if value not in backends.registered():
        raise ValueError(f"must be one of {list(backends.registered())}, got {value!r}")
    return value


def _image(value: str) -> str:
    value = value.strip()
    if not value or any(c.isspace() for c in value):
        raise ValueError(f"must be a non-empty image reference, got {value!r}")
    return value


def _platform(value: str) -> str:
    from yeto.launch.engine_images import canonical_device_family

    return canonical_device_family(value)


PLATFORM_DEST = "rl_island_platform"
# override key -> (args dest of the run-wide flag, value parser)
KEYS = {
    "image": ("rl_image", _image),
    "backend": ("rl_backend", _backend),
    "platform": (PLATFORM_DEST, _platform),
}


def parse_overrides(items: Iterable[str] | None, num_islands: int) -> dict[int, dict[str, Any]]:
    """``ISLAND:KEY=VALUE`` items -> ``{island: {args dest: value}}``."""

    out: dict[int, dict[str, Any]] = {}
    for item in items or ():
        island_text, sep, assignment = str(item).partition(":")
        key, eq, value = assignment.partition("=")
        if not sep or not eq or not key:
            raise ValueError(f"{FLAG} {item!r}: expected ISLAND:KEY=VALUE")
        try:
            island = int(island_text)
        except ValueError:
            raise ValueError(f"{FLAG} {item!r}: ISLAND must be an integer") from None
        if not 0 <= island < num_islands:
            raise ValueError(f"{FLAG} {item!r}: island {island} is not in 0..{num_islands - 1}")
        key = key.strip()
        if key not in KEYS:
            raise ValueError(f"{FLAG} {item!r}: unknown key {key!r} (known: {sorted(KEYS)})")
        dest, parse = KEYS[key]
        if dest in out.get(island, {}):
            raise ValueError(f"{FLAG}: island {island} gives {key!r} twice")
        try:
            out.setdefault(island, {})[dest] = parse(value)
        except ValueError as error:
            raise ValueError(f"{FLAG} {item!r}: {key} {error}") from None
    return out


def requested(args: Any) -> bool:
    return bool(getattr(args, OVERRIDES_DEST, None))


def default_backend_image(args: Any, backend: str) -> str:
    """The image an omitted ``--rl-image`` gets for ``backend`` (launch_flags
    ``DEFAULT_RL_IMAGE``, else the Miles recipe default)."""

    from . import backends

    flags = backends.module("launch_flags", backend)
    if getattr(flags, "DEFAULT_RL_IMAGE", None):
        return flags.DEFAULT_RL_IMAGE
    from ...rl import default_rl_image

    return default_rl_image(recipe=getattr(args, "rl_model_recipe", None),
                            tuning=getattr(args, "tuning", None))


def platform_trainer_image(args: Any, platform: str, backend: str) -> str:
    """The trainer image of an island that names ``platform`` but no ``image``.

    A family without a pinned trainer image for ``backend`` raises
    ``engine_images.NoDefaultImage`` (name the image). A pinned family
    (cuda) takes the same recipe-aware default an omitted ``--rl-image`` gets
    (:func:`default_backend_image`), so ``--rl-model-recipe kimi_k31 --tuning
    full`` keeps the full-form image instead of the table's generic row."""

    from yeto.launch.engine_images import NoDefaultImage, default_image_entry

    row = default_image_entry(platform, "trainer", backend)
    if row.ref is None:
        raise NoDefaultImage(f"no pinned trainer image for {row.device_family}/{row.engine}: "
                             f"{row.reason}; name it with {FLAG} ISLAND:image=...")
    return default_backend_image(args, backend)


def _backend_images(backend: str) -> set[str]:
    """The pinned images a backend ships (its ``pins`` ``*_IMAGE`` constants
    and its launch_flags ``DEFAULT_RL_IMAGE``)."""

    from . import backends

    images: set[str] = set()
    try:
        pins = backends.module("pins", backend)
    except backends.UnknownBackend:
        pins = None
    if pins is not None:
        images |= {v for k, v in vars(pins).items() if k.endswith("_IMAGE") and isinstance(v, str)}
    default = getattr(backends.module("launch_flags", backend), "DEFAULT_RL_IMAGE", None)
    if default:
        images.add(default)
    return images


def infer_backend(image: str | None) -> str | None:
    """The one backend ``image`` belongs to (a pinned image of that backend, or
    its own image form such as ``verl-build:<commit>``); None when no backend
    or more than one claims it."""

    from . import backends

    if not image:
        return None
    owners = set()
    for name in backends.registered():
        check = getattr(backends.module("launch_flags", name), "rl_image_ok", None)
        if image in _backend_images(name) or (check is not None and check(image)):
            owners.add(name)
    return owners.pop() if len(owners) == 1 else None


def island_args(args: Any, island: int, num_islands: int) -> Any:
    """A copy of ``args`` with island ``island``'s image / backend (``args``
    itself without the flag or without an override for this island)."""

    if not requested(args):
        return args
    own = parse_overrides(getattr(args, OVERRIDES_DEST), num_islands).get(island)
    if not own:
        return args
    out = copy.copy(args)
    run_backend = getattr(args, "rl_backend", None) or "miles"
    for dest, value in own.items():
        setattr(out, dest, value)
    backend = own.get("rl_backend")
    platform = own.get(PLATFORM_DEST)
    if "rl_image" not in own and platform is not None:
        out.rl_image = platform_trainer_image(args, platform, backend or run_backend)
    elif backend is not None and backend != run_backend and "rl_image" not in own:
        out.rl_image = default_backend_image(args, backend)
    return out


def image_ok(iargs: Any) -> bool:
    """``--rl-image``'s rule, applied to an island copy: a digest-pinned docker
    reference, or the island backend's own image form (verl: ``verl-build:<commit>``)."""

    image = getattr(iargs, "rl_image", None) or ""
    if _DIGEST_IMAGE.fullmatch(image):
        return True
    from . import backends

    check = getattr(backends.module("launch_flags", getattr(iargs, "rl_backend", None) or "miles"),
                    "rl_image_ok", None)
    return bool(check and check(image))


def records(args: Any, num_islands: int) -> list[dict[str, Any]]:
    """Dry-run / manifest rows of the overridden islands (empty without the flag)."""

    if not requested(args):
        return []
    rows = []
    for m in sorted(parse_overrides(getattr(args, OVERRIDES_DEST), num_islands)):
        iargs = island_args(args, m, num_islands)
        row = {"island": m, "backend": getattr(iargs, "rl_backend", None) or "miles",
               "image": getattr(iargs, "rl_image", None)}
        if getattr(iargs, PLATFORM_DEST, None):
            row["platform"] = getattr(iargs, PLATFORM_DEST)
        rows.append(row)
    return rows


def check(args: Any, num_islands: int) -> None:
    """Launch checks before any cloud work: syntax, each island's image form,
    a backend change only with ``--rl-sync-layout shape-free``, no image /
    backend change under ``--rl-code-pack release`` (the release manifest is
    checked against the run-wide image only) and no backend change under
    ``--rl-engine-cluster`` (its miles-only check reads the run-wide backend)."""

    if not requested(args):
        return
    if getattr(args, "training_mode", "rl") != "rl":
        raise ValueError(f"{FLAG} needs --training-mode rl")
    overrides = parse_overrides(getattr(args, OVERRIDES_DEST), num_islands)
    run_backend = getattr(args, "rl_backend", None) or "miles"
    for m, own in sorted(overrides.items()):
        iargs = island_args(args, m, num_islands)
        if not image_ok(iargs):
            from . import backends

            pin = getattr(backends.module("launch_flags", iargs.rl_backend or "miles"), "image_pin_problem", None)
            why = pin(str(iargs.rl_image or "")) if pin else None
            raise ValueError(f"{FLAG}: island {m} image {iargs.rl_image!r} is not a "
                             "docker:<repository>@sha256:<64 hex> reference (nor the "
                             f"{iargs.rl_backend or 'miles'} backend's own image form)"
                             + (f": {why}" if why else ""))
        backend = own.get("rl_backend")
        if "rl_image" in own:
            # an image of another backend than the island trains with
            island_backend = backend or run_backend
            inferred = infer_backend(own["rl_image"])
            if inferred is not None and inferred != island_backend:
                raise ValueError(f"{FLAG}: island {m} image {own['rl_image']!r} is a {inferred!r} image "
                                 f"but the island trains with backend {island_backend!r}; "
                                 f"set {m}:backend={inferred} (with --rl-sync-layout shape-free) "
                                 f"or name a {island_backend} image")
            if inferred is None and backend is None:
                print(f"[launcher] WARNING: {FLAG}: island {m} image {own['rl_image']!r} is not a "
                      f"known pinned image of any backend; the island trains with the run-wide "
                      f"backend {run_backend!r} and only the syncer identity check catches a "
                      f"different backend. Name {m}:backend=... if it is not {run_backend!r}.",
                      file=sys.stderr)
        if (backend is not None and backend != run_backend
                and getattr(args, "rl_sync_layout", "shape-bound") != "shape-free"):
            raise ValueError(f"{FLAG}: island {m} trains with backend {backend!r} but the run "
                             f"uses {run_backend!r}; islands of different backends share a "
                             "syncer session only with --rl-sync-layout shape-free")
        if backend is not None and backend != run_backend and getattr(args, "rl_engine_cluster", None):
            raise ValueError(f"{FLAG}: island {m} trains with backend {backend!r} but "
                             f"--rl-engine-cluster checks only the run-wide backend {run_backend!r}; "
                             "drop the island backend or --rl-engine-cluster")
        changed = ((getattr(iargs, "rl_backend", None) or "miles") != run_backend
                   or getattr(iargs, "rl_image", None) != getattr(args, "rl_image", None))
        if changed and getattr(args, "rl_code_pack", None) == "release":
            raise ValueError(f"{FLAG}: island {m} changes its image or backend but --rl-code-pack "
                             "release checks the release manifest of the run-wide --rl-image only; "
                             "drop the island override or use another --rl-code-pack mode")
