"""The Harbor layer of the Miles and verl engine images (S26 E0.4).

Harbor is imported in-process by both backends (verl ``HarborAgentLoop``,
Miles' Harbor example and ``records_generate``), so it cannot live in its own
venv without a process bridge. It is installed into the engine's Python under
a uv constraint file that holds every base package; only the moves listed in
``docker/harbor-layer/versions.json`` are allowed. The build then checks the
result and writes the start lock ``/opt/yeto/image/versions.lock.json`` that
:func:`yeto.launch.image_check.check_image_at_start` asserts before training.
"""

from __future__ import annotations

from pathlib import Path

from .pins import HARBOR_COMMIT

LAYER_DIR = Path(__file__).resolve().parents[4] / "docker" / "harbor-layer"
ASSERT_SCRIPT = LAYER_DIR.parent / "image-assert" / "assert_image_versions.py"
IMAGE_LAYER_DIR = "/opt/yeto/harbor-layer"
IMAGE_DIR = "/opt/yeto/image"
START_LOCK = f"{IMAGE_DIR}/versions.lock.json"
BACKENDS = ("miles", "verl")


def layer_commands(backend: str, python: str, *, uv_version: str = "0.9.5",
                   harbor_requirement: str | None = None, boat_sdk: str = "1.10.0") -> list[str]:
    """Shell commands (one image layer) that install Harbor + boat-sdk into ``python``'s environment.

    ``python`` is the engine interpreter (verl: the venv's python; Miles: the
    image's ``python``). Any failed step fails the image build.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; known: {list(BACKENDS)}")
    requirement = harbor_requirement or f"harbor @ git+https://github.com/harbor-framework/harbor@{HARBOR_COMMIT}"
    tool = f"{python} {IMAGE_LAYER_DIR}/harbor_layer.py"
    before, after = f"{IMAGE_LAYER_DIR}/freeze-before.txt", f"{IMAGE_LAYER_DIR}/freeze-after.txt"
    return [
        f"command -v uv || python3 -m pip install uv=={uv_version}",
        f"uv pip freeze --python {python} > {before}",
        f"{tool} constraints --backend {backend} {before} > {IMAGE_LAYER_DIR}/constraints.txt",
        f"uv pip install --python {python} -c {IMAGE_LAYER_DIR}/constraints.txt '{requirement}' 'boat-sdk=={boat_sdk}'",
        f"uv pip freeze --python {python} > {after}",
        f"{tool} check --backend {backend} {before} {after}",
        f"{tool} lock --backend {backend} {after} --out {START_LOCK}",
        f"{python} {IMAGE_DIR}/assert_image_versions.py {START_LOCK} --report {IMAGE_LAYER_DIR}/build-assert.json",
    ]


def modal_harbor_layer(image, backend: str, python: str):
    """``image`` (a ``modal.Image``) plus the Harbor layer; Modal ignores the Docker entrypoint,
    so the start check runs from yeto's entry points instead (``check_image_at_start``)."""
    import json

    spec = json.loads((LAYER_DIR / "versions.json").read_text())
    if spec["harbor"]["commit"] != HARBOR_COMMIT:
        raise ValueError(f"versions.json pins Harbor {spec['harbor']['commit']}, pins.py {HARBOR_COMMIT}")
    return (image
            .add_local_dir(str(LAYER_DIR), IMAGE_LAYER_DIR, copy=True)
            .add_local_file(str(ASSERT_SCRIPT), f"{IMAGE_DIR}/assert_image_versions.py", copy=True)
            .run_commands(*layer_commands(backend, python, uv_version=spec["uv"],
                                          harbor_requirement=spec["harbor"]["requirement"],
                                          boat_sdk=spec["boat_sdk"])))
