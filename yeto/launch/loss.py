"""Loss-function resolution and the content-addressed pickled-loss staging."""

from __future__ import annotations

import os
from pathlib import Path

from . import REPO_ROOT


PICKLED_LOSS_FILE = ".yeto_loss.pkl"
PICKLED_LOSS_PREFIX = ".yeto_loss."


def pickled_loss_path(spec: str) -> Path:
    """Resolve a pickle spec, including a staged workdir-relative payload."""

    if not spec.startswith("pickle:"):
        raise ValueError(f"not a pickle loss spec: {spec!r}")
    source_text = spec.split(":", 1)[1]
    source_name = Path(source_text).name
    staged = (
        source_text == source_name
        and (
            source_name == PICKLED_LOSS_FILE
            or (
                source_name.startswith(PICKLED_LOSS_PREFIX)
                and source_name.endswith(".pkl")
            )
        )
    )
    return REPO_ROOT / source_name if staged else Path(source_text).expanduser()


def _stage_pickled_loss(payload: bytes) -> str:
    """Content-address a legacy executable payload to avoid cross-run races."""

    import hashlib

    digest = hashlib.sha256(payload).hexdigest()
    filename = f"{PICKLED_LOSS_PREFIX}{digest}.pkl"
    destination = REPO_ROOT / filename
    if destination.is_symlink():
        raise RuntimeError(
            f"refusing symlink at content-addressed pickle path {destination}"
        )
    if destination.exists():
        if destination.read_bytes() != payload:
            raise RuntimeError(
                f"content-addressed pickle collision at {destination}"
            )
    else:
        temporary = destination.with_name(
            f"{PICKLED_LOSS_PREFIX}{digest}.tmp-{os.getpid()}.pkl"
        )
        temporary.write_bytes(payload)
        os.chmod(temporary, 0o600)
        try:
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return f"pickle:{filename}"


def resolve_loss_function(
    loss_function,
    *,
    allow_unsafe_pickled_loss: bool = False,
) -> str:
    """Return the --loss-function string to pass to learners.

    A callable or a ``custom:<file.py>`` spec can be shipped through the
    legacy pickle lane only after the explicit unsafe opt-in.  Named losses
    pass through. All pickle inputs are copied to a content-addressed,
    shell-neutral workdir path and attested before learner execution.
    """
    from ..losses import load_custom_loss

    if callable(loss_function):
        fn = loss_function
    elif isinstance(loss_function, str) and loss_function.startswith("custom:"):
        if not allow_unsafe_pickled_loss:
            raise PermissionError(
                "custom loss transport uses legacy pickle and requires "
                "--allow-unsafe-pickled-loss"
            )
        fn = load_custom_loss(loss_function)
    elif isinstance(loss_function, str) and loss_function.startswith("pickle:"):
        if not allow_unsafe_pickled_loss:
            raise PermissionError(
                "legacy pickle loss transport requires "
                "--allow-unsafe-pickled-loss"
            )
        source = pickled_loss_path(loss_function)
        if not source.is_file():
            raise FileNotFoundError(f"pickled loss {str(source)!r} does not exist")
        return _stage_pickled_loss(source.read_bytes())
    else:
        return loss_function
    if not allow_unsafe_pickled_loss:
        raise PermissionError(
            "callable/custom loss transport uses legacy pickle and requires "
            "--allow-unsafe-pickled-loss"
        )
    import cloudpickle

    return _stage_pickled_loss(cloudpickle.dumps(fn))
