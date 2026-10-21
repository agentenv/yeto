"""Modal image of the verl engine (rl-verl-backend 1.1, design D10).

Same recipe S16 measured (VERL-MODAL-CHECK-S16.md section 1): CUDA 13.0 devel
base + Python 3.12, the pinned fork checked out at ``VERL_COMMIT``, all
dependencies from the fork's own ``uv.lock`` (``uv sync --frozen
--all-packages --extra vllm --extra fsdp``), plus yeto's one-hook local patch
(``patch_verl``).  ``--rl-image verl-build:<commit>`` selects it; Modal caches
the layers, so only the first run builds. yeto's object-store client (boto3,
``STORE_CLIENT_REQUIREMENTS``) is added to the same venv.  yeto's own image default
(ghcr.io/michaellchung/yeto-miles-ports) is unchanged.
"""

from __future__ import annotations

from pathlib import Path

from .pins import (STORE_CLIENT_REQUIREMENTS, VERL_BASE_IMAGE, VERL_COMMIT, VERL_PYTHON, VERL_REPOSITORY,
                   VERL_ROOT, VERL_UV_VERSION)

VENV_BIN = f"{VERL_ROOT}/.venv/bin"


def build_commands(commit: str = VERL_COMMIT) -> list[str]:
    if commit != VERL_COMMIT:
        raise ValueError(f"verl image commit {commit} is not the pinned {VERL_COMMIT}")
    return [
        f"git clone {VERL_REPOSITORY} {VERL_ROOT} && cd {VERL_ROOT} && git checkout {commit}"
        f" && git rev-parse HEAD > /workspace/verl_sha.txt",
        f"cd {VERL_ROOT} && UV_PYTHON=/usr/local/bin/python{VERL_PYTHON} uv sync --frozen --all-packages"
        " --extra vllm --extra fsdp",
    ]


def store_client_commands() -> list[str]:
    """An island over an s3:// or oss:// store reads it with boto3 (S25 C3V); its own
    layer, so the cached verl build above stays valid."""
    return [f"uv pip install --python {VENV_BIN}/python " + " ".join(STORE_CLIENT_REQUIREMENTS)]


def image_env() -> dict[str, str]:
    return {
        "PATH": f"{VENV_BIN}:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:"
                "/usr/sbin:/usr/bin:/sbin:/bin",
        "VIRTUAL_ENV": f"{VERL_ROOT}/.venv",
        "VERL_USE_UV": "0",
    }


def modal_image(modal, commit: str = VERL_COMMIT, patched: bool = True):
    """The verl image; ``patched=False`` leaves the verl source as the fork ships it
    (the bare-verl arms of the efficiency evaluation, which must not import yeto)."""
    image = (
        modal.Image.from_registry(VERL_BASE_IMAGE, add_python=VERL_PYTHON)
        .apt_install("git", "build-essential", "curl", "libnuma1")
        .pip_install(f"uv=={VERL_UV_VERSION}")
        .run_commands(*build_commands(commit))
        .run_commands(*store_client_commands())
    )
    if patched:
        patch = Path(__file__).with_name("patch_verl.py")
        image = (image.add_local_file(str(patch), "/workspace/patch_verl.py", copy=True)
                 .run_commands(f"{VENV_BIN}/python /workspace/patch_verl.py {VERL_ROOT}"))
    return image.env(image_env())
