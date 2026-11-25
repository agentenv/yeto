"""Modal image of the verl engine (rl-verl-backend 1.1).

Recipe: CUDA 13.0 devel
base + Python 3.12, the pinned fork checked out at ``VERL_COMMIT``, all
dependencies from the fork's own ``uv.lock`` (``uv sync --frozen
--all-packages --extra vllm --extra fsdp``), plus yeto's one-hook local patch
(``patch_verl``).  ``--rl-image verl-build:<commit>`` selects it; Modal caches
the layers, so only the first run builds. yeto's object-store client (boto3,
``STORE_CLIENT_REQUIREMENTS``) is added to the same venv.  yeto's own image default
(yeto-miles-ports on the yeto.rl.image_registry registry) is unchanged.
"""

from __future__ import annotations

from pathlib import Path

from .pins import (STORE_CLIENT_REQUIREMENTS, VERL_BASE_IMAGE, VERL_COMMIT, VERL_PYTHON, VERL_REPOSITORY,
                   VERL_ROOT, VERL_UV_VERSION)

VENV_BIN = f"{VERL_ROOT}/.venv/bin"
# The interpreter the Harbor layer installs into (``image_layer``):
# verl runs out of its own venv, not the image's /usr/local python.
HARBOR_LAYER_PYTHON = f"{VENV_BIN}/python"


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
    """An island over an s3:// or oss:// store reads it with boto3; its own
    layer, so the cached verl build above stays valid."""
    return [f"uv pip install --python {VENV_BIN}/python " + " ".join(STORE_CLIENT_REQUIREMENTS)]


def image_env() -> dict[str, str]:
    return {
        "PATH": f"{VENV_BIN}:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:"
                "/usr/sbin:/usr/bin:/sbin:/bin",
        "VIRTUAL_ENV": f"{VERL_ROOT}/.venv",
        "VERL_USE_UV": "0",
    }


def megatron_commands(max_jobs: int = 16) -> list[str]:
    """The Megatron engine layers on top of :func:`build_commands` (variant
    ``megatron``): the fork's ``megatron`` extra (``--inexact``: the vLLM and FSDP
    packages stay), then causal-conv1d and mamba-ssm built for Hopper (the extra does
    not list them; mamba-ssm needs its CUDA selective-scan module, so the build
    cannot be skipped), then the flash-attn FA4 preview removed (it raises
    AttributeError against the locked cutlass-dsl, Megatron-Core only catches
    ImportError; Hopper does not use it). ``megatron_recipe.image_commands`` (the bare-verl
    smoke) returns these same steps."""
    from .pins import CAUSAL_CONV1D_VERSION, MAMBA_SSM_VERSION

    env = f"MAX_JOBS={max_jobs} TORCH_CUDA_ARCH_LIST=9.0 CC=gcc CXX=g++"
    return [
        f"cd {VERL_ROOT} && UV_PYTHON=/usr/local/bin/python{VERL_PYTHON} uv sync --frozen --all-packages"
        " --inexact --extra vllm --extra fsdp --extra megatron",
        f"uv pip install --python {VENV_BIN}/python ninja packaging wheel setuptools",
        f"{env} CAUSAL_CONV1D_FORCE_BUILD=TRUE uv pip install --python {VENV_BIN}/python --no-build-isolation"
        f" 'causal-conv1d=={CAUSAL_CONV1D_VERSION}'",
        f"{env} MAMBA_FORCE_BUILD=TRUE uv pip install --python {VENV_BIN}/python"
        f" --no-build-isolation --no-deps 'mamba-ssm=={MAMBA_SSM_VERSION}'",
        f"rm -rf {VERL_ROOT}/.venv/lib/python{VERL_PYTHON}/site-packages/flash_attn/cute",
    ]


VARIANTS = (None, "megatron")
# The Megatron island sends the weights from the trainer container to the
# rollout container over NCCL (verl's checkpoint engine, cupy NCCL). Without the
# verbs libraries NCCL cannot use the cluster's RDMA network and falls back to
# sockets: a measured first 63 GB update took 57.7 s (about 1.1 GB/s).
RDMA_PACKAGES = ("rdma-core", "ibverbs-providers")


APT_PACKAGES = ("git", "build-essential", "curl", "libnuma1")
PATCH_DEST = "/workspace/patch_verl.py"


def recipe_steps(commit: str = VERL_COMMIT, patched: bool = True, variant: str | None = None) -> list[tuple]:
    """The build steps on top of the base image, in order, one Modal call each:
    ``("apt", packages)``, ``("pip", requirements)``, ``("run", commands)``,
    ``("copy", source, destination)`` and ``("env", mapping)``. :func:`modal_image`
    and :func:`dockerfile` (the registry image) both read this one list, so the two
    images cannot drift apart: only how Python lands on the base differs."""
    if variant not in VARIANTS:
        raise ValueError(f"verl image variant {variant!r} is not one of {VARIANTS}")
    steps: list[tuple] = [
        ("apt", APT_PACKAGES),
        ("pip", (f"uv=={VERL_UV_VERSION}",)),
        ("run", tuple(build_commands(commit))),
        ("run", tuple(store_client_commands())),
    ]
    if variant == "megatron":
        steps += [("run", tuple(megatron_commands())), ("apt", RDMA_PACKAGES)]
    if patched:
        suffix = " --profile megatron" if variant == "megatron" else ""
        steps += [("copy", str(Path(__file__).with_name("patch_verl.py")), PATCH_DEST),
                  ("run", (f"{VENV_BIN}/python {PATCH_DEST} {VERL_ROOT}{suffix}",))]
    steps.append(("env", image_env()))
    return steps


def modal_image(modal, commit: str = VERL_COMMIT, patched: bool = True, variant: str | None = None):
    """The verl image; ``patched=False`` leaves the verl source as the fork ships it
    (the bare-verl arms of the efficiency evaluation, which must not import yeto).
    ``variant="megatron"`` (``--rl-image verl-megatron-build:<commit>``) adds
    the Megatron engine layers after the cached FSDP ones and applies the
    ``megatron`` patch profile (``patch_verl``)."""
    steps = recipe_steps(commit, patched, variant)  # an unknown variant is refused before any build call
    image = modal.Image.from_registry(VERL_BASE_IMAGE, add_python=VERL_PYTHON)
    for step in steps:
        kind = step[0]
        if kind == "apt":
            image = image.apt_install(*step[1])
        elif kind == "pip":
            image = image.pip_install(*step[1])
        elif kind == "run":
            image = image.run_commands(*step[1])
        elif kind == "copy":
            image = image.add_local_file(step[1], step[2], copy=True)
        else:
            image = image.env(step[1])
    return image


# --- the registry image (fleet-verl-trainer-islands 5.1) ----------------------
#
# The same recipe as :func:`modal_image` (FSDP variant, patched) written as a
# Dockerfile, for islands that pull an image instead of building on Modal (sky,
# ssh docker nodes). ``docker/verl-fsdp/Dockerfile`` is this text, checked in;
# ``docker/verl-fsdp/build.sh`` builds and pushes it; pins.VERL_REGISTRY_* names
# the pushed digest and the recipe hash it was built from (a test keeps the three equal).
# The only difference from Modal: Modal's ``add_python`` puts its own Python
# 3.12 at /usr/local/bin; here the distribution's 3.12 is linked there (the
# venv verl runs from is created by uv from that interpreter in both images),
# with rsync and sudo, which a sky node's container setup uses.
DOCKERFILE_PATCH_SOURCE = "patch_verl.py"  # in the build context (build.sh copies it)


def _sh_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


def _dockerfile_body(commit: str) -> list[str]:
    lines = [
        f"RUN apt-get update && apt-get install -y --no-install-recommends python{VERL_PYTHON} "
        f"python{VERL_PYTHON}-dev python{VERL_PYTHON}-venv python3-pip ca-certificates rsync sudo"
        f" && ln -sf /usr/bin/python{VERL_PYTHON} /usr/local/bin/python{VERL_PYTHON}"
        f" && ln -sf /usr/bin/python{VERL_PYTHON} /usr/local/bin/python3"
        f" && ln -sf /usr/bin/python{VERL_PYTHON} /usr/local/bin/python",
    ]
    for step in recipe_steps(commit):
        kind = step[0]
        if kind == "apt":
            lines.append("RUN apt-get update && apt-get install -y " + " ".join(step[1]))
        elif kind == "pip":
            lines.append(f"RUN python{VERL_PYTHON} -m pip install --break-system-packages "
                         + " ".join(_sh_quote(r) for r in step[1]))
        elif kind == "run":
            lines.append("RUN " + " && ".join(f"({c})" for c in step[1]))
        elif kind == "copy":
            lines.append(f"COPY {DOCKERFILE_PATCH_SOURCE} {step[2]}")
        else:
            lines.append("ENV " + " ".join(f"{k}={_sh_quote(v)}" for k, v in step[1].items()))
    # a login shell (an ssh: or sky node's command) keeps the venv first on PATH too
    env = image_env()
    lines.append("RUN printf '%s\\n' " + " ".join(_sh_quote(f"export {k}={env[k]}")
                                                  for k in ("PATH", "VIRTUAL_ENV", "VERL_USE_UV"))
                 + " > /etc/profile.d/yeto-verl.sh")
    return lines


def recipe_sha256(commit: str = VERL_COMMIT) -> str:
    """sha256 of the registry image's recipe: the base image, every Dockerfile step
    after it and the content of the patch file it copies in."""
    import hashlib

    patch = hashlib.sha256(Path(__file__).with_name("patch_verl.py").read_bytes()).hexdigest()
    text = "\n".join([VERL_BASE_IMAGE, *_dockerfile_body(commit), f"patch_verl.py sha256 {patch}"])
    return hashlib.sha256(text.encode()).hexdigest()


def dockerfile(commit: str = VERL_COMMIT) -> str:
    """``docker/verl-fsdp/Dockerfile``: :func:`recipe_steps` of the FSDP image, patched."""
    lines = [
        "# GENERATED by `python -m yeto.rl.adapters.verl.image --dockerfile` from",
        "# yeto/rl/adapters/verl/image.py:recipe_steps (the Modal image's recipe). Do not edit;",
        "# docker/verl-fsdp/build.sh builds and pushes it.",
        f"FROM {VERL_BASE_IMAGE}",
        "ENV DEBIAN_FRONTEND=noninteractive",
        f"LABEL ai.yeto.verl.commit={commit} ai.yeto.verl.recipe_sha256={recipe_sha256(commit)}",
        *_dockerfile_body(commit),
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--dockerfile"]:
        sys.stdout.write(dockerfile())
    elif sys.argv[1:] == ["--recipe-sha256"]:
        print(recipe_sha256())
    else:
        raise SystemExit("usage: python -m yeto.rl.adapters.verl.image --dockerfile | --recipe-sha256")
