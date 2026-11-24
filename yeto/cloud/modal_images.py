"""Modal images built from a registry engine image (vLLM / SGLang sampler images).

Modal runs a function with the image's own interpreter and reads its version
from ``python`` on ``PATH``. The upstream vLLM images (and the yeto vLLM sampler
image built on them) install deadsnakes Python as ``/usr/bin/python3`` only, with
no ``python``; Modal then refuses the function at deploy time with
``ConflictError: unable to determine the version of Python installed in the Image``.

``Image.from_registry(add_python=...)`` is not a fix for these images: it copies
a standalone interpreter into ``/usr/local`` ahead of ``/usr/bin`` on ``PATH``,
and that interpreter has none of the engine's packages (vLLM, boto3, ...).
Instead one small layer links the image's own ``python3`` as ``python`` when
the image has no ``python``. An image that already has ``python`` (the SGLang
sampler image) keeps it unchanged. The layer goes on top of the registry layer,
so a registry image Modal has already pulled is reused, not rebuilt.
"""

from __future__ import annotations

from typing import Any

# no-op when the image has ``python``, or has no ``python3`` either (Modal then reports that image as before)
PYTHON_ON_PATH = ('RUN command -v python >/dev/null 2>&1 || ! command -v python3 >/dev/null 2>&1 '
                  '|| ln -sf "$(command -v python3)" /usr/local/bin/python')


def engine_registry_image(modal: Any, ref: str) -> Any:
    """``modal.Image.from_registry(ref)`` for an engine image that brings its own
    interpreter, with ``python`` on ``PATH`` for Modal (see the module docstring)."""
    return modal.Image.from_registry(ref).dockerfile_commands([PYTHON_ON_PATH])
