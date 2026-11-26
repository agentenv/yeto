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
        nodes = _site_catalog_trainer_nodes(args)
        if nodes is not None:  # an ssh: trainer island whose inventory names no image
            import sys

            from . import site_images

            try:
                choice = site_images.select(
                    None, getattr(args, "model", None), "trainer", nodes,
                    engine=getattr(args, "rl_backend", None) or backends.DEFAULT_BACKEND,
                    path=site_images.inventory_catalog_path(getattr(args, "rl_node_inventory", None)),
                    default=lambda: _builtin_rl_image(args))
            except site_images.SiteImagesError as exc:
                raise ValueError(str(exc)) from None
            args.rl_image = f"docker:{choice.ref}" if choice.source == site_images.SOURCE_SITE else choice.ref
            print(f"[launcher] {choice.line('ssh trainer')}", file=sys.stderr, flush=True)
            return
        args.rl_image = _builtin_rl_image(args)


def _builtin_rl_image(args) -> str:
    """The engine's pinned default ``--rl-image``."""
    flags = _rl_backend_module(args, "launch_flags")
    if getattr(flags, "DEFAULT_RL_IMAGE", None):  # rl-verl-backend: Modal-built engine image
        return flags.DEFAULT_RL_IMAGE
    from ..rl import default_rl_image

    return default_rl_image(recipe=getattr(args, "rl_model_recipe", None), tuning=getattr(args, "tuning", None))


def _site_catalog_trainer_nodes(args) -> list | None:
    """The trainer nodes of an ``ssh:`` island of ``--gpu`` whose ``--rl-node-inventory``
    names no image for them (the site image catalog may then give one,
    :mod:`yeto.launch.site_images`); None for any other launch (no ssh: island, no
    inventory, an inventory image: the configuration names its image)."""
    path = getattr(args, "rl_node_inventory", None)
    if not path or not getattr(args, "gpu", None):
        return None
    from ..gpu_spec import parse_gpu_spec
    from .node_inventory import InventoryError, island_nodes, load_inventory

    try:
        if not any(spec.cloud == "ssh" for spec in parse_gpu_spec(args.gpu)):
            return None
        nodes = island_nodes(load_inventory(path), "trainer")
    except (ValueError, InventoryError):  # the launch's own checks name a bad --gpu or inventory
        return None
    if not nodes or any(n.image for n in nodes):
        return None
    return nodes


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


# clouds whose trainer island does not pull --rl-image from a registry: Modal builds
# the backend's own image form, an ssh: island checks its docker nodes' image itself
# (ssh_trainer_island.run_image)
_NO_PULL_CLOUDS = ("modal", "ssh")


def pulled_rl_image(args, cloud: str | None) -> dict | None:
    """``{"image", "tag", "replaces"}`` when a trainer island on ``cloud`` (a sky cloud)
    pulls a registry image in place of the backend's Modal-built ``--rl-image`` (verl:
    ``verl-build:<commit>`` -> the registry image of the same pin and recipe); None
    otherwise (the Miles backend, a Modal or ssh: island, any other image)."""
    if getattr(args, "training_mode", "sft") != "rl" or cloud is None or cloud in _NO_PULL_CLOUDS:
        return None
    pulled = getattr(_rl_backend_module(args, "launch_flags"), "pulled_image", None)
    return pulled(getattr(args, "rl_image", None) or "") if pulled else None


def _rl_backend_module(args, role: str):
    """Adapter module for ``role`` of the selected training backend (decoupling 5.4).

    The launcher never imports an adapter directly; ``--rl-backend`` (default
    miles) picks the row of ``yeto.rl.engine.backends``.
    """
    from ..rl.engine import backends

    return backends.module(role, getattr(args, "rl_backend", None) or backends.DEFAULT_BACKEND)
