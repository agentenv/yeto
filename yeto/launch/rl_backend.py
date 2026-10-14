"""RL training-backend selection for the launcher (decoupling 5.4).

The launcher never imports a backend adapter directly; these helpers resolve
the adapter module for a role of the selected ``--rl-backend``.
"""

from __future__ import annotations


def resolve_default_rl_image(args) -> None:
    """Fill an omitted ``--rl-image`` with the engine's pinned default."""

    if getattr(args, "training_mode", "sft") != "rl":
        return
    from ..rl.engine import backends

    try:  # decoupling 5.4: a backend without a registered adapter is refused here
        backends.get(getattr(args, "rl_backend", None) or backends.DEFAULT_BACKEND)
    except backends.UnknownBackend as exc:
        raise ValueError(str(exc)) from None
    if getattr(args, "rl_image", None) is None:
        flags = _rl_backend_module(args, "launch_flags")
        if getattr(flags, "DEFAULT_RL_IMAGE", None):  # rl-verl-backend: Modal-built engine image
            args.rl_image = flags.DEFAULT_RL_IMAGE
            return
        from ..rl import default_rl_image

        args.rl_image = default_rl_image()


def _rl_callable(value: str | None, flag: str, *, required: bool) -> None:
    if value is None and not required:
        return
    module, separator, function = (value or "").partition(":")
    if (
        not separator
        or not module
        or module.endswith(".py")
        or not function.isidentifier()
    ):
        raise ValueError(f"{flag} must be package.module:function")


def _rl_miles_function(
    value: str | None, flag: str = "--custom-generate-function-path"
) -> None:
    if value is None:
        return
    parts = value.split(".")
    if len(parts) < 2 or any(not part.isidentifier() for part in parts):
        raise ValueError(f"{flag} must be package.module.function")


def _rl_backend_image_ok(args) -> bool:
    """A backend may accept its own --rl-image form (verl: ``verl-build:<commit>``)."""
    check = getattr(_rl_backend_module(args, "launch_flags"), "rl_image_ok", None)
    return bool(check and check(getattr(args, "rl_image", None) or ""))


def _miles_overlay():
    """The Miles adapter's overlay module (source patches on the ~/miles checkout)."""
    from ..rl.engine import backends

    return backends.module("overlay", "miles")


def _rl_backend_module(args, role: str):
    """Adapter module for ``role`` of the selected training backend (decoupling 5.4).

    The launcher never imports an adapter directly; ``--rl-backend`` (default
    miles) picks the row of ``yeto.rl.engine.backends``.
    """
    from ..rl.engine import backends

    return backends.module(role, getattr(args, "rl_backend", None) or backends.DEFAULT_BACKEND)
