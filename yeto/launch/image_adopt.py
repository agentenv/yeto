"""``yeto image adopt``: take an image built for one model into the site image catalog.

An operator who has an image optimized for a new model (a trainer image for a Miles
or verl island, a sampler image for a vLLM / SGLang pool) adopts it on an SSH node
inventory in five steps, each one an existing module of yeto:

1. **manifest**: the registry has the image (:mod:`.image_manifest`, a manifest HEAD
   with the registry login of the pull); a tag is pinned to its digest. Nothing is
   asked of a node before this step passed;
2. **probe**: one node, after its host checks (ssh, the host probe, room in its image
   root) and a check that its cards are idle (:func:`busy_script`), runs the image once
   (``<container engine> run --rm``, no cards: the built card architectures are read
   from torch's build flags) and prints what it carries: CPU architecture, Python,
   torch, CUDA / HIP, the card architectures its kernels were built for, the engine
   packages and the yeto image lock (``/opt/yeto/image``);
3. **fit**: the image against every node of the role: CPU architecture, device
   runtime, card architecture (the image's built architectures under the CUDA binary
   compatibility rules, :func:`card_arch_fit`; a trainer image's
   verified floor, :mod:`.trainer_image_check`; a catalog sampler image's record,
   :func:`.engine_images.image_mismatch`) and the engine the role runs;
4. **pull**: every node of the role pulls the image in parallel after its image root
   was checked for room (the cluster preflight's ssh, host probe, ``disk_images`` and
   ``image_pull`` checks, :meth:`.preflight.cluster.ClusterPreflight.pull`; a failed
   pull names the node and docker's line, :mod:`.image_pull`);
5. **smoke**: the role's smoke on one node (:mod:`.image_smoke`).

Every step that fails stops the adoption with its name and the reason
(:class:`AdoptError`, which names the nodes the image was already pulled on: it stays
there for a retry); the catalog is written only after the smoke passed, with the
versions the probe read and the evidence of the run (:mod:`.site_images`).
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import re
import shlex
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import engine_images, image_manifest, image_pull, site_images, trainer_image_check
from .image_smoke import MAX_LAYERS as MAX_SMOKE_LAYERS
from .image_smoke import StepResult
from .node_inventory import Node, card_arch_of

STEPS = ("manifest", "probe", "fit", "pull", "smoke")
FACT = "YETO_IMAGE_FACT"
LOCK_PATH = "/opt/yeto/image/versions.lock.json"
# interpreters a probe tries in the image, in order (the sampler images' own venvs first)
PYTHONS = ("/opt/runtime-venv/bin/python3", "/opt/sglang/bin/python3", "python3", "python")
# the packages a probe reads (importlib.metadata names)
PACKAGES = ("torch", "vllm", "vllm-ascend", "sglang", "miles", "megatron-core", "megatron-bridge",
            "transformer-engine", "verl", "flash-attn", "torch-npu", "ray", "transformers")
# the packages a role's engine needs in the image (any of them)
ENGINE_PACKAGES = {("trainer", "miles"): ("miles", "megatron-core"), ("trainer", "verl"): ("verl",),
                   ("sampler", "vllm"): ("vllm",), ("sampler", "sglang"): ("sglang",),
                   ("sampler", "vllm-ascend"): ("vllm-ascend",)}
PROBE_TIMEOUT_S = 3600.0  # the probe node pulls the image first
BUSY_MARK = "YETO_BUSY"
MINOR_RULE_NOTE = "compatible by CUDA minor-version rule; the smoke test is the final check"
PTX_RULE_NOTE = "compatible by PTX JIT; the smoke test is the final check"
CUDA_RULE = ("CUDA binary compatibility: a cubin sm_XY runs on compute capability X.Z with Z >= Y, sm_XYa only "
             "on exactly X.Y; PTX compute_XY is JIT-compiled on any X.Y or newer card, compute_XYa only on X.Y")


class AdoptError(RuntimeError):
    """An adoption step failed; ``step`` is one of :data:`STEPS`."""

    def __init__(self, step: str, reason: str, *, holding: Sequence[str] = ()):
        super().__init__(f"image adopt: step {step} failed: {reason}")
        self.step = step
        self.reason = reason
        self.holding = tuple(holding)


@dataclass(frozen=True)
class AdoptRequest:
    """What to adopt: ``ref`` (a tag or a digest), the ``model`` it is built for, the
    ``role`` (``trainer`` / ``sampler``), the inventory nodes (:func:`role_nodes`),
    the ``engine`` (trainer: the ``--rl-backend``; sampler: ``vllm`` / ``sglang`` /
    ``vllm-ascend``; None: the default backend, or the engine the probe finds), the node
    that probes and smokes (None: the first of the role), the catalog path (None: the
    default), the image root's free space a pull needs, the trainer's attention
    backend (the trainer floor of :mod:`.trainer_image_check`), ``strict_arch`` (the
    image must carry a cubin of exactly the node's card architecture: no minor-version
    or PTX compatibility), ``allow_busy`` (probe and smoke a node whose cards run
    something, with a warning) and ``container_sudo`` (the engine runs as ``sudo -n
    <engine>``, as an ssh sampler pool's ``container.sudo`` runs it: its rootful store is
    the one checked, pulled into and run from)."""

    ref: str
    model: str
    role: str
    nodes: tuple[Node, ...]
    engine: str | None = None
    node: str | None = None
    catalog: str | None = None
    min_image_free_gib: float = 120.0
    attention_backend: str = "auto"
    strict_arch: bool = False
    allow_busy: bool = False
    container_sudo: bool = False
    spec_engine: str | None = None  # sampler: the engine the pool's sampler spec runs (its engine.adapter)

    def __post_init__(self) -> None:
        if self.role not in site_images.ROLES:
            raise ValueError(f"--role {self.role!r} is unknown (known: {list(site_images.ROLES)})")
        if not self.nodes:
            raise ValueError(f"the inventory has no {self.role} node")
        if self.node is not None and self.node not in {n.name for n in self.nodes}:
            raise ValueError(f"--node {self.node} is not a {self.role} node of the inventory "
                             f"({[n.name for n in self.nodes]})")

    @property
    def probe_node(self) -> Node:
        return next(n for n in self.nodes if self.node in (None, n.name))


@dataclass(frozen=True)
class ImageFacts:
    """What the probe read inside the image."""

    cpu_arch: str | None
    python: str | None
    python_path: str | None
    torch: str | None
    cuda: str | None
    hip: str | None
    arch_list: tuple[str, ...]
    packages: Mapping[str, str]
    lock: Mapping[str, Any] | None = None
    arch_source: str | None = None  # "build_flags" (no card needed) / "cards"

    def versions(self) -> dict[str, str]:
        """The catalog record's ``versions``: the interpreter, torch, the device runtime and
        every package found."""
        out = {"python": self.python, "python_path": self.python_path, "torch": self.torch, "cuda": self.cuda,
               "hip": self.hip, **{k: v for k, v in self.packages.items() if k != "torch"}}
        if self.lock:
            out["image_lock"] = str(self.lock.get("image") or self.lock.get("name") or "present")
        return {k: str(v) for k, v in out.items() if v}


@dataclass
class AdoptReport:
    """The steps run so far; ``record`` once the catalog was written."""

    request: AdoptRequest
    steps: list[StepResult] = field(default_factory=list)
    pinned: str | None = None
    facts: ImageFacts | None = None
    record: site_images.SiteImage | None = None
    catalog: Path | None = None

    @property
    def ok(self) -> bool:
        return bool(self.steps) and all(s.ok for s in self.steps)

    def lines(self) -> list[str]:
        out = [f"[image adopt] {s.name}: {'ok' if s.ok else 'FAILED'} ({s.seconds:.1f} s) {s.detail}"
               for s in self.steps]
        if self.record is not None:
            out.append(f"[image adopt] recorded {self.record.ref} for {self.record.model} {self.record.role} "
                       f"({self.record.engine}, {'/'.join(self.record.card_archs)}) in {self.catalog}")
        return out


def role_nodes(nodes: Sequence[Node], role: str) -> list[Node]:
    """The inventory nodes of ``role``: no roles, the role itself, or a role of its kind
    (``trainer-spare``): every node that may run the image is pulled and checked."""
    return [n for n in nodes if not n.roles or any(r == role or r.startswith(role + "-") for r in n.roles)]


# --- step 1: manifest ----------------------------------------------------------------


def pin(ref: str, *, login: Mapping[str, str] | None = None, urlopen=None) -> str:
    """The digest-pinned reference of ``ref`` (no ``docker:`` prefix), or :class:`AdoptError`
    (step ``manifest``) when the registry has no such manifest or cannot be read. With
    ``YETO_IMAGE_MANIFEST_CHECK=off`` a digest is taken as it is and a tag is refused."""
    bare = ref.strip().removeprefix("docker:")
    if not site_images.valid_reference(bare, digest=False):
        raise AdoptError("manifest", f"{bare!r} is not a docker reference ([REGISTRY[:PORT]/]PATH[:TAG][@sha256:<64 hex>])")
    if not image_manifest.enabled():
        if not re.search(r"@sha256:[0-9a-f]{64}$", bare):
            raise AdoptError("manifest", f"{image_manifest.CHECK_ENV}=off: a tag cannot be pinned; give the "
                                         "image's digest (REGISTRY/IMAGE@sha256:<64 hex>)")
        return bare
    try:
        status, detail = image_manifest.pin_digest(bare, dict(login) if login else None, urlopen=urlopen)
    except ValueError as error:
        raise AdoptError("manifest", str(error)) from None
    if status == image_manifest.PRESENT:
        return detail
    if status == image_manifest.MISSING:
        raise AdoptError("manifest", f"the registry has no manifest for {bare} ({detail})")
    if status == image_manifest.DENIED:
        raise AdoptError("manifest", f"{detail}; give the registry login of the pull (SKYPILOT_DOCKER_USERNAME / "
                                     "SKYPILOT_DOCKER_PASSWORD / SKYPILOT_DOCKER_SERVER, or ~/.docker/config.json)")
    raise AdoptError("manifest", f"{detail}; {image_manifest.CHECK_ENV}=off skips the check for a digest")


def registry_login(ref: str) -> dict[str, str] | None:
    """The registry login a pull of ``ref`` would use (the ``SKYPILOT_DOCKER_*`` variables,
    else ``~/.docker/config.json``); never printed."""
    host = image_manifest.split_tagged_ref(ref)[0]
    return image_pull.fresh_registry_login(host)


# --- step 2: probe -------------------------------------------------------------------

_PROBE_PY = r"""
import json, platform, sys
from importlib import metadata
def fact(key, value):
    print("YETO_IMAGE_FACT", key, json.dumps(value))
fact("python", platform.python_version())
fact("python_path", sys.executable)
pkgs = {}
for name in PACKAGES:
    try:
        pkgs[name] = metadata.version(name)
    except Exception:
        pass
fact("packages", pkgs)
try:
    import torch
    fact("torch", torch.__version__)
    fact("cuda", getattr(torch.version, "cuda", None))
    fact("hip", getattr(torch.version, "hip", None))
    archs, source = [], None
    try:
        flags = torch._C._cuda_getArchFlags()
        if flags:
            archs, source = flags.replace(";", " ").split(), "build_flags"
    except Exception:
        pass
    if not archs:
        try:
            archs, source = list(torch.cuda.get_arch_list()), "cards"
        except Exception:
            pass
    fact("arch_list", archs)
    fact("arch_source", source)
except Exception as error:
    fact("torch_error", f"{type(error).__name__}: {error}")
""".replace("PACKAGES", repr(PACKAGES))


def engine_of(node: Node) -> str | None:
    """The container engine of ``node``'s images (docker / podman; None: it runs none)."""
    from .node_inventory import profile_of  # noqa: PLC0415

    engine = profile_of(node).container
    return engine if engine in ("docker", "podman") else None


def container_engine(node: Node) -> str:
    """The container engine the node runs images with (its profile's ``container``:
    ``docker`` / ``podman``); a node that runs no container is refused (step ``probe``)."""
    from .node_inventory import profile_of  # noqa: PLC0415

    engine = profile_of(node).container
    if engine not in ("docker", "podman"):
        raise AdoptError("probe", f"node {node.name} runs no container engine (container {engine!r}): an image is "
                                  "adopted on a docker or podman node")
    return engine


def probe_script(image: str, node: Node, *, cards: bool = False, sudo: bool = False) -> str:
    """The node's ``<engine> run --rm`` of ``image`` that prints the image's facts
    (:data:`FACT` lines): ``uname -m``, the image lock and, under the first interpreter of
    :data:`PYTHONS` that imports torch (else the first that runs), the Python facts.
    The built card architectures come from torch's build flags
    (``torch._C._cuda_getArchFlags``), which need no card: the container gets the
    node's device flags only with ``cards`` (the fallback for a torch whose flags are
    not readable; ``torch.cuda.get_arch_list`` is empty without cards)."""
    from .node_inventory import profile_of  # noqa: PLC0415

    engine = container_engine(node)
    devices = profile_of(node).devices(None, engine=engine).args if cards else ()
    inner = (f"echo {FACT} uname_m \\\"$(uname -m)\\\"; "
             f"if [ -f {LOCK_PATH} ]; then printf '%s lock ' {FACT}; tr -d '\\n' < {LOCK_PATH}; echo; fi; "
             "PY=; for p in " + " ".join(PYTHONS) + "; do "
             "if command -v \"$p\" >/dev/null 2>&1 && \"$p\" -c 'import torch' >/dev/null 2>&1; then PY=$p; break; fi; "
             "done; "
             "if [ -z \"$PY\" ]; then for p in " + " ".join(PYTHONS) + "; do "
             "if command -v \"$p\" >/dev/null 2>&1; then PY=$p; break; fi; done; fi; "
             f"if [ -z \"$PY\" ]; then echo {FACT} no_python true; exit 0; fi; "
             f"\"$PY\" -c {shlex.quote(_PROBE_PY)}")
    argv = [*(["sudo", "-n"] if sudo else []), engine, "run", "--rm", "--network", "host", *devices,
            "--entrypoint", "bash", image, "-c", inner]
    return shlex.join(argv)


# the card tool that lists the processes on the cards, per unified family
_BUSY_TOOLS = {
    "cuda": (("nvidia-smi",), "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader"),
    "rocm": (("amd-smi", "rocm-smi"), "if command -v amd-smi >/dev/null 2>&1; then amd-smi process --csv; "
                                      "else rocm-smi --showpids; fi"),
    "ascend": (("npu-smi",), "npu-smi info"),
}


def busy_script(node: Node, *, sudo: bool = False) -> str:
    """What runs on the node's cards: the family's card tool (:data:`_BUSY_TOOLS`: the
    compute processes of ``nvidia-smi``, ``amd-smi process`` / ``rocm-smi --showpids``,
    the process rows of ``npu-smi info``) and the node's running yeto containers. Each
    line of the output is tagged (``YETO_BUSY tool|apps|containers ...``)."""
    from .node_inventory import profile_of  # noqa: PLC0415

    profile = profile_of(node)
    found = _BUSY_TOOLS.get(profile.family)
    engine = profile.container if profile.container in ("docker", "podman") else "docker"
    parts = []
    if found:
        names, tool = found
        have = " || ".join(f"command -v {n} >/dev/null 2>&1" for n in names)
        parts.append(f"if {have}; then echo {BUSY_MARK} tool {profile.family}; "
                     f"( {tool} ) 2>&1 | sed 's/^/{BUSY_MARK} apps /'; else echo {BUSY_MARK} tool none; fi")
    else:
        parts.append(f"echo {BUSY_MARK} tool none")
    parts.append(f"{'sudo -n ' if sudo else ''}{engine} ps --filter name=^yeto- --format '{{{{.Names}}}} {{{{.Image}}}}' 2>/dev/null "
                 f"| sed 's/^/{BUSY_MARK} containers /'")
    return "; ".join(parts)


_NPU_PROCESS_ROW = re.compile(r"^\|\s*\d+\s+\d+\s*\|\s*(\d+)\s*\|\s*(\S+)")


def parse_busy(text: str, family: str) -> tuple[list[str], list[str], bool]:
    """``(processes, containers, readable)`` of a :func:`busy_script` output: the
    processes on the cards (``pid name``), the running yeto containers and whether a
    card tool was read at all."""
    apps: list[str] = []
    containers: list[str] = []
    readable = False
    for line in (text or "").splitlines():
        parts = line.rstrip().split(" ", 2)
        if len(parts) < 3 or parts[0] != BUSY_MARK:
            continue
        kind, value = parts[1], parts[2].strip()
        if kind == "tool":
            readable = value != "none"
        elif kind == "containers" and value:
            containers.append(value)
        elif kind == "apps" and value:
            apps += _busy_rows(value, family)
    return apps, containers, readable


def _busy_rows(line: str, family: str) -> list[str]:
    if family == "cuda":  # pid, process_name, used_memory
        cells = [c.strip() for c in line.split(",")]
        return [f"{cells[0]} {cells[1] if len(cells) > 1 else '?'}"] if cells[0].isdigit() else []
    if family == "ascend":
        m = _NPU_PROCESS_ROW.match(line)
        return [f"{m.group(1)} {m.group(2)}"] if m else []
    if family == "rocm":  # amd-smi process --csv: gpu,name,pid,...; rocm-smi --showpids: PID NAME GPU(s) ...
        cells = [c.strip() for c in line.split(",")]
        if len(cells) >= 3 and cells[0].isdigit() and cells[2].isdigit():
            return [f"{cells[2]} {cells[1]}"]
        words = line.split()
        return [f"{words[0]} {words[1]}"] if len(words) >= 2 and words[0].isdigit() and len(cells) == 1 else []
    return []


def parse_probe(text: str) -> ImageFacts:
    """The facts of a probe's output; a line that does not parse is skipped."""
    facts: dict[str, Any] = {}
    for line in (text or "").splitlines():
        parts = line.strip().split(" ", 2)
        if len(parts) != 3 or parts[0] != FACT:
            continue
        try:
            facts[parts[1]] = json.loads(parts[2])
        except ValueError:
            continue
    cpu = facts.get("uname_m")
    try:
        cpu = engine_images.canonical_cpu_arch(cpu) if cpu else None
    except engine_images.UnknownCpuArch:
        pass
    packages = facts.get("packages") if isinstance(facts.get("packages"), dict) else {}
    archs = facts.get("arch_list") if isinstance(facts.get("arch_list"), list) else []
    source = facts.get("arch_source") if isinstance(facts.get("arch_source"), str) else None
    lock = facts.get("lock") if isinstance(facts.get("lock"), dict) else None
    return ImageFacts(cpu_arch=cpu, python=facts.get("python"), python_path=facts.get("python_path"),
                      torch=facts.get("torch"), cuda=facts.get("cuda"), hip=facts.get("hip"),
                      arch_list=tuple(str(a).lower() for a in archs),
                      packages={str(k): str(v) for k, v in packages.items()}, lock=lock, arch_source=source)


# --- step 3: fit ---------------------------------------------------------------------


_CUDA_ARCH = re.compile(r"(sm|compute)_?(\d{2,3})([af]?)")


def _cc(arch: str) -> tuple[str, tuple[int, int], str] | None:
    """``(kind, (major, minor), suffix)`` of ``sm_103`` / ``sm103`` / ``compute_90a``."""
    m = _CUDA_ARCH.fullmatch(arch)
    if not m:
        return None
    digits = m.group(2)
    return m.group(1), (int(digits[:-1]), int(digits[-1])), m.group(3)


def _gfx(arch: str) -> str:
    """A ROCm target without its feature suffix (``gfx90a:sramecc+:xnack-`` -> ``gfx90a``)."""
    return arch.split(":", 1)[0]


def node_card_arch(node: Node) -> str | None:
    """The node's card architecture: its profile's ``card_arch`` (an inventory may set
    it), else the catalog's for its card."""
    from .node_inventory import profile_of  # noqa: PLC0415

    arch = profile_of(node).card_arch or card_arch_of(node.card)
    return str(arch).lower() if arch else None


# The trainer's flash attention (Megatron --attention-backend flash) trains on Hopper and
# newer only: on A10G (sm86) cards the Miles image's forward pass runs and the backward
# fails ("Unsupported compute capability"), while unfused trains (own-cluster rehearsal,
# 2026-11-26). A sampler's inference attention is not held to it.
FLASH_TRAINER_MIN_ARCH = "sm90"


def flash_trainer_problem(node: Node, attention_backend: str | None) -> str | None:
    """The problem of a trainer on ``node`` with the ``flash`` attention backend below
    :data:`FLASH_TRAINER_MIN_ARCH` (None: another backend, a card at or above it, or a
    card of no CUDA architecture known here)."""
    if (attention_backend or "auto") != "flash":
        return None
    arch = node_card_arch(node)
    have, need = _cc(arch or ""), _cc(FLASH_TRAINER_MIN_ARCH)
    if have is None or need is None or have[1] >= need[1]:
        return None
    return (f"node {node.name}: the trainer's flash attention needs {FLASH_TRAINER_MIN_ARCH}+ cards; on {node.card} "
            f"({arch}) its backward pass fails (\"Unsupported compute capability\"): pass --attention-backend "
            "unfused (the smoke trains with it; it holds the full score matrix, short sequences only)")


def card_arch_fit(facts: ImageFacts, node: Node, *, strict: bool = False) -> tuple[str | None, str | None]:
    """``(problem, note)`` of the image's built card architectures on ``node``'s cards
    (problem None: they serve them, or the node's architecture is unknown; note: how).

    CUDA, the binary compatibility rules (:data:`CUDA_RULE`): a cubin of exactly the
    node's ``sm_XY`` serves it; otherwise a cubin ``sm_XW`` (no ``a`` suffix) of the same
    major with W <= the node's minor (note :data:`MINOR_RULE_NOTE`), or a PTX
    ``compute_XY`` the node's capability reaches (``compute_XYa``: only exactly X.Y;
    note :data:`PTX_RULE_NOTE`). With ``strict`` only the exact cubin counts. ROCm:
    the node's ``gfx`` target is built (feature suffixes after ``:`` ignored). An image
    whose probe saw no architecture is refused: nothing says it runs there."""
    arch = node_card_arch(node)
    if not arch:
        return None, None
    built = facts.arch_list
    have = f"{node.card} ({arch})"
    if not built:
        return (f"the probe read no card architecture from the image (torch's build flags and "
                f"torch.cuda.get_arch_list() are empty: no device build of torch); node {node.name} has {have}"), None
    node_cc = _cc(arch)
    if node_cc is not None:
        want = node_cc[1]
        exact = minor = ptx = None
        for entry in built:
            cc = _cc(entry)
            if cc is None:
                continue
            kind, version, suffix = cc
            if kind == "sm":
                if version == want:
                    exact = exact or entry
                elif suffix != "a" and version[0] == want[0] and version[1] <= want[1]:
                    minor = minor or entry
            elif suffix == "a":
                ptx = ptx or (entry if version == want else None)
            elif suffix == "f":
                ptx = ptx or (entry if version[0] == want[0] and version[1] <= want[1] else None)
            elif version <= want:
                ptx = ptx or entry
        if exact:
            return None, None
        if strict:
            return (f"--strict-arch: the image is built for {'/'.join(built)}, no cubin of exactly {arch}; "
                    f"node {node.name} has {have}"), None
        if minor:
            return None, f"node {node.name} ({arch}) runs the {minor} cubin: {MINOR_RULE_NOTE}"
        if ptx:
            return None, f"node {node.name} ({arch}) JIT-compiles the {ptx} PTX: {PTX_RULE_NOTE}"
        return f"the image is built for {'/'.join(built)}; node {node.name} has {have} ({CUDA_RULE})", None
    if _gfx(arch) in {_gfx(a) for a in built}:
        return None, None
    return (f"the image is built for {'/'.join(built)}; node {node.name} has {have} (ROCm: the node's gfx target "
            "must be built)"), None


def card_arch_problem(facts: ImageFacts, node: Node, *, strict: bool = False) -> str | None:
    """The problem of :func:`card_arch_fit` (None: the image serves the node's cards)."""
    return card_arch_fit(facts, node, strict=strict)[0]


def fit_problems(request: AdoptRequest, ref: str, facts: ImageFacts,
                 engine: str) -> tuple[list[str], list[str], list[str]]:
    """``(problems, warnings, notes)`` of the image on every node of the role (each names
    the node; a note says a node passed by a compatibility rule, :func:`card_arch_fit`).
    The trainer floor and a catalog sampler image's record are the existing rules
    (:func:`.trainer_image_check.image_problem`, :func:`.engine_images.image_mismatch`)."""
    problems: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []
    if request.spec_engine and request.engine and request.spec_engine.lower() != request.engine.lower():
        problems.append(f"--engine {request.engine} but the sampler spec runs {request.spec_engine} (its "
                        "engine.adapter): the record would name an engine the smoke did not run")
    need = ENGINE_PACKAGES.get((request.role, engine), ())
    if need and not any(p in facts.packages for p in need):
        problems.append(f"the image has no {request.role} engine {engine} (none of {list(need)} is installed; "
                        f"found {sorted(facts.packages) or 'no package'})")
    for node in request.nodes:
        if facts.cpu_arch and facts.cpu_arch != node.cpu_arch:
            problems.append(f"the image is {facts.cpu_arch}; node {node.name} is {node.cpu_arch}")
            continue
        family = engine_images.canonical_device_family(node.device_family)
        if family == "cuda" and not facts.cuda:
            problems.append(f"the image's torch has no CUDA build (torch {facts.torch or 'missing'}); node "
                            f"{node.name} is a CUDA node")
            continue
        if family == "rocm" and not facts.hip:
            problems.append(f"the image's torch has no ROCm build (torch {facts.torch or 'missing'}); node "
                            f"{node.name} is a ROCm node")
            continue
        if family in ("cuda", "rocm"):
            why, note = card_arch_fit(facts, node, strict=request.strict_arch)
            if why:
                problems.append(why)
                continue
            if note:
                notes.append(note)
        if request.role == "trainer":
            flash = flash_trainer_problem(node, request.attention_backend)
            if flash:  # the backend's own floor, before the image's (which flash would lift)
                problems.append(flash)
                continue
            found = trainer_image_check.image_problem(ref, node.card, attention_backend=request.attention_backend)
            if found:
                (problems if found[1] else warnings).append(f"node {node.name}: {found[0]}")
        else:
            why = engine_images.image_mismatch(ref, cpu_arch=node.cpu_arch, device_family=node.device_family,
                                               card_arch=node_card_arch(node), model=request.model)
            if why:
                problems.append(f"node {node.name}: {why}")
    return problems, warnings, notes


def default_engine(request: AdoptRequest, facts: ImageFacts | None) -> str:
    """The engine of the record: the one asked for (``--engine``; for a trainer smoke
    the backend the model registry recommends, which ``yeto image adopt`` passes), else
    the sampler spec's (``engine.adapter``), else the default training backend (trainer)
    or the engine package the probe found (sampler: vllm-ascend, vllm, sglang)."""
    if request.engine:
        return request.engine.lower()
    if request.spec_engine:
        return request.spec_engine.lower()
    if request.role == "trainer":
        from ..rl.engine import backends  # noqa: PLC0415

        return backends.DEFAULT_BACKEND
    packages = facts.packages if facts else {}
    return next((e for e, pkg in (("vllm-ascend", "vllm-ascend"), ("vllm", "vllm"), ("sglang", "sglang"))
                 if pkg in packages), "vllm")


# --- the steps in order ----------------------------------------------------------------


SmokeFn = Callable[["Adopter", str, ImageFacts, str], StepResult]


class Adopter:
    """The five steps over ``ops`` (an ``SshOps``: ``run(node, script, timeout=)``; tests
    pass a fake). ``smoke``: step 5 (``(adopter, pinned ref, facts, engine) ->
    StepResult``); ``urlopen`` / ``login``: the registry access of step 1."""

    def __init__(self, request: AdoptRequest, *, ops: Any = None, smoke: SmokeFn | None = None,
                 urlopen=None, login: Mapping[str, str] | None | bool = True,
                 clock: Callable[[], float] = time.time, log: Callable[[str], None] | None = None) -> None:
        if ops is None:
            from .ssh_trainer_island import SshOps  # noqa: PLC0415

            ops = SshOps()
        self.request = request
        self.ops = ops
        self.smoke = smoke
        self.urlopen = urlopen
        self.login = login
        self.clock = clock
        self.log = log or (lambda m: print(m, file=sys.stderr, flush=True))
        self.report = AdoptReport(request)
        self.holding: list[str] = []  # the nodes the image was pulled on so far

    def _held(self, error: AdoptError) -> AdoptError:
        """``error`` naming the nodes that hold the image (:attr:`holding`): the image
        stays there, a retry reuses it, ``<engine> rmi`` frees the room."""
        if not self.holding or error.holding:
            return error
        from .preflight.cluster import engine_command  # noqa: PLC0415

        ref = self.report.pinned or self.request.ref
        by_name = {n.name: n for n in self.request.nodes}
        per: dict[str, list[str]] = {}  # each holding node's own engine (and sudo)
        for name in self.holding:
            node = by_name.get(name)
            engine = (engine_of(node) if node is not None else None) or "docker"
            per.setdefault(engine_command(engine, self.request.container_sudo), []).append(name)
        nodes = ", ".join(self.holding)
        remove = "; ".join(f"`{command} rmi {ref}` on {', '.join(names)}" for command, names in sorted(per.items()))
        return AdoptError(error.step, f"{error.reason}; the image stays pulled on {nodes} (a retry reuses it; "
                                      f"{remove} frees the room)", holding=self.holding)

    def _hold(self, *names: str) -> None:
        self.holding += [n for n in names if n not in self.holding]

    def _step(self, name: str, fn: Callable[[], tuple[str, dict[str, Any]]]) -> dict[str, Any]:
        self.log(f"[image adopt] {name} ...")
        t0 = self.clock()
        try:
            detail, data = fn()
        except AdoptError as error:
            self.report.steps.append(StepResult(name, False, error.reason, self.clock() - t0))
            raise self._held(error) from None
        self.report.steps.append(StepResult(name, True, detail, self.clock() - t0, data))
        return data

    def run(self) -> AdoptReport:
        """Every step in order; :class:`AdoptError` at the first that fails (the catalog is
        not touched). On success the catalog has the record (:attr:`AdoptReport.record`)."""
        req = self.request
        self.checks()
        pinned, facts = str(self.report.pinned), self.report.facts
        assert facts is not None
        engine = default_engine(req, facts)
        if self.smoke is None:
            raise self._held(AdoptError("smoke", f"no {req.role} smoke is wired for this adoption"))
        try:  # the pull may have taken long: the node is checked again right before the smoke
            self.check_idle(req.probe_node, "smoke")
        except AdoptError as error:
            self.report.steps.append(StepResult("smoke", False, error.reason, 0.0))
            raise self._held(error) from None
        self.log("[image adopt] smoke ...")
        result = self.smoke(self, pinned, facts, engine)
        self.report.steps.append(result)
        if not result.ok:
            raise self._held(AdoptError("smoke", result.detail))
        self.report.record = self._record(pinned, facts, engine, result)
        self.report.catalog = site_images.add(self.report.record, req.catalog)
        return self.report

    def checks(self) -> AdoptReport:
        """Steps 1 to 4 only (no smoke, nothing written)."""
        pinned = self._step("manifest", self._manifest)["ref"]
        facts = self._step("probe", lambda: self._probe(pinned))["facts"]
        engine = default_engine(self.request, facts)
        self._step("fit", lambda: self._fit(pinned, facts, engine))
        self._step("pull", lambda: self._pull(pinned))
        return self.report

    # -- steps
    def _manifest(self) -> tuple[str, dict[str, Any]]:
        login = registry_login(self.request.ref) if self.login is True else (self.login or None)
        pinned = pin(self.request.ref, login=login, urlopen=self.urlopen)
        self.report.pinned = pinned
        return pinned, {"ref": pinned}

    def _probe(self, ref: str) -> tuple[str, dict[str, Any]]:
        node = self.request.probe_node
        self._probe_host(node, ref)
        self.check_idle(node, "probe")
        facts = self._run_probe(node, ref, cards=False)
        family = engine_images.canonical_device_family(node.device_family)
        if not facts.arch_list and family in ("cuda", "rocm"):
            self.log(f"[image adopt] torch's build flags name no card architecture; probing {node.name} again "
                     "with its cards")
            facts = self._run_probe(node, ref, cards=True)
        self.report.facts = facts
        what = ", ".join(f"{k} {v}" for k, v in (("python", facts.python), ("torch", facts.torch),
                                                 ("cuda", facts.cuda), ("hip", facts.hip)) if v)
        archs = "/".join(facts.arch_list) or "no card architecture"
        return f"on {node.name}: {facts.cpu_arch or '?'}, {what}, built for {archs}", {"facts": facts}

    def _probe_host(self, node: Node, ref: str) -> None:
        """ssh, the host probe and room in the image root on the probe node, before the
        probe pulls the image there (the cluster preflight's checks)."""
        from .preflight.cluster import FAIL, ClusterPreflight, Requirements  # noqa: PLC0415

        req = Requirements(min_image_root_free_gib=float(self.request.min_image_free_gib))
        checks = ClusterPreflight([node], image=ref, ops=self.ops, req=req, log=self.log, clock=self.clock,
                                  engine=container_engine(node), sudo=self.request.container_sudo,
                                  by_engine=True).hosts()
        failed = [c for c in checks if c.status == FAIL and c.name in ("ssh", "host_probe", "disk_images")]
        if failed:
            raise AdoptError("probe", "; ".join(f"node {c.scope}: {c.name}: {c.detail}" + (f" ({c.fix})" if c.fix
                                                                                           else "")
                                                for c in failed))

    def check_idle(self, node: Node, step: str) -> None:
        """Refuse (step ``step``) a node whose cards run a process or that runs a yeto
        container (:func:`busy_script`); with ``allow_busy`` it is a warning."""
        from .node_inventory import profile_of  # noqa: PLC0415

        family = profile_of(node).family
        try:
            done = self.ops.run(node, busy_script(node, sudo=self.request.container_sudo), timeout=120.0)
            text = (getattr(done, "stdout", "") or "") + "\n" + (getattr(done, "stderr", "") or "")
        except Exception as error:  # noqa: BLE001 - a dead ssh is the step's failure
            raise AdoptError(step, f"node {node.name}: cannot read what runs on its cards: "
                                   f"{type(error).__name__}: {error}") from None
        apps, containers, readable = parse_busy(text, family)
        if not readable:
            self.log(f"[image adopt] WARNING: node {node.name}: no card tool to read its processes ({family}); "
                     "its cards are taken as idle")
        busy = [f"process {a}" for a in apps] + [f"container {c}" for c in containers]
        if not busy:
            return
        why = f"node {node.name} is in use: {', '.join(busy[:8])}"
        if not self.request.allow_busy:
            raise AdoptError(step, f"{why}; stop it, pick another node with --node, or pass --allow-busy")
        self.log(f"[image adopt] WARNING: {why}; --allow-busy: the {step} runs there anyway")

    def _run_probe(self, node: Node, ref: str, *, cards: bool) -> ImageFacts:
        try:
            done = self.ops.run(node, probe_script(ref, node, cards=cards, sudo=self.request.container_sudo),
                                timeout=PROBE_TIMEOUT_S)
        except AdoptError:
            raise
        except Exception as error:  # noqa: BLE001 - a dead ssh or a timeout is the step's failure
            raise AdoptError("probe", f"node {node.name}: {type(error).__name__}: {error}") from None
        text = (getattr(done, "stdout", "") or "") + "\n" + (getattr(done, "stderr", "") or "")
        if done.returncode != 0:
            failure = image_pull.pull_failure(f"docker pull\n{text}")
            why = failure.line if failure else text.strip()[-400:]
            raise AdoptError("probe", f"node {node.name}: {container_engine(node)} run of {ref} failed "
                                      f"(rc {done.returncode}): {why}")
        self._hold(node.name)
        facts = parse_probe(text)
        if not facts.python:
            raise AdoptError("probe", f"node {node.name}: the image has no Python interpreter yeto can run "
                                      f"(tried {list(PYTHONS)})")
        return facts

    def _fit(self, ref: str, facts: ImageFacts, engine: str) -> tuple[str, dict[str, Any]]:
        problems, warnings, notes = fit_problems(self.request, ref, facts, engine)
        node = self.request.probe_node
        if not node_card_arch(node):
            problems.append(f"node {node.name} has {node.card} cards of no known architecture: the catalog "
                            "matches records by card architecture")
        for warning in warnings:
            self.log(f"[image adopt] WARNING: {warning}")
        for note in notes:
            self.log(f"[image adopt] note: {note}")
        if problems:
            raise AdoptError("fit", "; ".join(problems))
        detail = f"{engine} {self.request.role} image fits {len(self.request.nodes)} node(s)"
        return detail + "".join(f"; {n}" for n in notes), {"warnings": warnings, "notes": notes}

    def _pull(self, ref: str) -> tuple[str, dict[str, Any]]:
        """Every node of the role pulls the image with its own container engine (its
        profile's ``container``: docker or podman), the image root of that engine checked
        first (docker's data root / containerd store, podman's graphroot)."""
        from .preflight.cluster import FAIL, PASS, ClusterPreflight, Requirements  # noqa: PLC0415

        groups: dict[str, list[Node]] = {}
        for n in self.request.nodes:
            engine = engine_of(n)
            if engine:
                groups.setdefault(engine, []).append(n)
        skipped = [n.name for n in self.request.nodes if not engine_of(n)]
        if not groups:
            raise AdoptError("pull", f"no {self.request.role} node runs a container engine (docker or podman): "
                                     f"{skipped}")
        req = Requirements(min_image_root_free_gib=float(self.request.min_image_free_gib))
        checks, pulled = [], True
        for engine, nodes in sorted(groups.items()):
            pf = ClusterPreflight(nodes, image=ref, ops=self.ops, req=req, log=self.log, clock=self.clock,
                                  engine=engine, sudo=self.request.container_sudo, by_engine=True)
            found, ok = pf.pull()
            checks += found
            pulled = pulled and ok
        self._hold(*(c.scope for c in checks if c.name == "image_pull" and c.status == PASS))
        failed = [c for c in checks if c.status == FAIL and c.name in ("ssh", "host_probe", "disk_images",
                                                                       "image_pull")]
        if failed or not pulled:
            raise AdoptError("pull", "; ".join(f"node {c.scope}: {c.name}: {c.detail}" + (f" ({c.fix})" if c.fix
                                                                                          else "")
                                               for c in failed) or "the pull did not run")
        took = {c.scope: c.detail for c in checks if c.name == "image_pull"}
        return (f"pulled on {len(took)} node(s)" + (f"; not pulled (no container engine): {skipped}" if skipped
                                                     else ""),
                {"pulled": sorted(took)})

    def _record(self, ref: str, facts: ImageFacts, engine: str, smoke: StepResult) -> site_images.SiteImage:
        node = self.request.probe_node
        steps = {s.name: round(s.seconds, 1) for s in self.report.steps}
        evidence = {"at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "node": node.name, "seconds": round(sum(steps.values()), 1), "result": "pass",
                    "steps": steps, "pulled": next((list(s.data.get("pulled", [])) for s in self.report.steps
                                                    if s.name == "pull"), []),
                    **({"smoke": smoke.data} if smoke.data else {})}
        notes = [n for s in self.report.steps for n in (s.data.get("notes") or [])]
        if notes:
            evidence["notes"] = notes
        arch = node_card_arch(node)
        return site_images.parse_record({
            "model": self.request.model, "role": self.request.role, "engine": engine, "ref": ref,
            "device_family": engine_images.canonical_device_family(node.device_family),
            "cpu_arch": node.cpu_arch, "card_archs": [arch] if arch else [], "versions": facts.versions(),
            "evidence": evidence, "status": site_images.VERIFIED})


# --- yeto image adopt ------------------------------------------------------------------


def add_parser(sub: Any) -> Any:
    p = sub.add_parser("image", help="site images: `yeto image adopt <ref> --model M --role trainer|sampler "
                                     "--inventory INV`")
    image_sub = p.add_subparsers(dest="image_command", metavar="{adopt}", required=True)
    a = image_sub.add_parser(
        "adopt", help="check, pre-pull and smoke an image built for a model; record it in the site image catalog",
        description="Adopt an image built for one model on the nodes of an SSH inventory: registry manifest and "
                    "digest, a probe of the image on one node, its fit to every node of the role, a parallel "
                    "pre-pull with the image root checked, and the role's smoke on one node. Only a passing "
                    "smoke writes the site image catalog (the inventory's site_images:, else "
                    f"{site_images.DEFAULT_PATH}); an island or pool that names no image then takes it.")
    a.add_argument("ref", help="the image: REGISTRY/IMAGE:TAG or REGISTRY/IMAGE@sha256:<digest>")
    a.add_argument("--model", required=True, help="the model the image is built for (an HF id)")
    a.add_argument("--role", required=True, choices=site_images.ROLES)
    a.add_argument("--inventory", required=True, help="the SSH node inventory (YAML / JSON)")
    a.add_argument("--engine", default=None,
                   help="trainer: the --rl-backend (default: the model registry's recommendation; the smoke runs "
                        "miles); sampler: vllm | sglang | vllm-ascend (default: the sampler spec's engine.adapter, "
                        "else the engine found in the image); a spec that runs another engine is refused")
    a.add_argument("--node", default=None, help="the node that probes and smokes (default: the first of the role)")
    a.add_argument("--catalog", default=None, help="the site image catalog (default: the inventory's site_images:)")
    a.add_argument("--min-image-free-gib", type=float, default=120.0,
                   help="free GiB the image root (docker data root / containerd store) of every node needs")
    a.add_argument("--attention-backend", default="auto", choices=["auto", "flash", "fused", "unfused"],
                   help="the trainer's attention backend: the fit holds the image floor and flash's sm90+ floor "
                        "to it, and the smoke trains with it (--rl-training-attention-backend; default auto: the "
                        "recipe's)")
    a.add_argument("--strict-arch", action="store_true",
                   help="the image must carry a cubin of exactly the node's card architecture (no CUDA "
                        "minor-version or PTX compatibility)")
    a.add_argument("--container-sudo", action="store_true",
                   help="run the container engine as sudo -n <engine> (an ssh sampler pool's container.sudo: its "
                        "rootful store); default: --pool-config's container.sudo, else off")
    a.add_argument("--pool-config", default=None,
                   help="the ssh sampler pool's config: its provider.container.sudo decides --container-sudo")
    a.add_argument("--allow-busy", action="store_true",
                   help="probe and smoke a node whose cards run processes or that runs yeto containers (a warning)")
    a.add_argument("--keep-smoke-artifacts", action="store_true",
                   help="keep the smoke's cut model, checkpoint and files (printed with their size) instead of "
                        "deleting them")
    a.add_argument("--dry-run", action="store_true", help="print every step and command; reach no registry or node")
    t = a.add_argument_group("trainer smoke")
    t.add_argument("--model-config", default=None,
                   help="the model's config.json (default: a local dir / the HF cache, else fetched from the Hub)")
    t.add_argument("--model-revision", default=None, help="the HF revision the smoke cuts")
    t.add_argument("--model-path", default=None,
                   help="the model's checkpoint directory on the smoke node (instead of downloading it there)")
    t.add_argument("--smoke-store", default=None, help="s3://BUCKET/PREFIX: the smoke run's store (its checkpoint)")
    t.add_argument("--smoke-layers", type=int, default=None,
                   help=f"layers of the cut (default: the most, at most {MAX_SMOKE_LAYERS}, that fit one node)")
    t.add_argument("--smoke-arg", action="append", default=[], metavar="ARG",
                   help="an extra `yeto launch` argument of the smoke (repeat; --smoke-arg=--flag=value)")
    s = a.add_argument_group("sampler smoke")
    s.add_argument("--sampler-spec", default=None, help="the pool's sampler spec (JSON) whose engine is smoked")
    s.add_argument("--release", default=None,
                   help="the release directory the smoke reloads (container path; default: the engine's base weights)")
    return p


def _model_config(args: Any) -> dict[str, Any]:
    from ..rl.model_registry.match import load_config  # noqa: PLC0415

    if args.model_config:
        return load_config(args.model_config)[0]
    try:
        return load_config(args.model)[0]
    except FileNotFoundError:
        if args.dry_run:
            raise
    from ..rl.model_registry.cli import _fetch  # noqa: PLC0415 - config.json only, never weights

    with open(_fetch(args.model, "config.json", args.model_revision), encoding="utf-8") as handle:
        return json.load(handle)


def _plan(args: Any, request: AdoptRequest, run_id: str) -> Any:
    """The smoke plan, with the image as :data:`.image_smoke.REF` (the trainer's backend:
    the registry's, which has a smoke only for the default backend)."""
    from . import image_smoke  # noqa: PLC0415
    from .node_inventory import profile_of  # noqa: PLC0415

    workdir = Path(site_images.default_path()).parent / "image-adopt" / run_id
    if request.role == "trainer":
        if request.container_sudo:
            raise image_smoke.SmokeError("--container-sudo / a pool config's container.sudo is for sampler pool "
                                         "nodes: the ssh: trainer island runs docker as the ssh user")
        off = [f"{n.name} ({n.runtime}, container {engine_of(n) or profile_of(n).container})" for n in request.nodes
               if n.runtime != "docker" or engine_of(n) != "docker"]
        if off:  # every node of the role runs the island's image, not only the smoke node
            raise image_smoke.SmokeError(f"trainer nodes {off}: an ssh: trainer island starts docker containers "
                                         "only (inventory runtime: docker); podman and rootfs nodes take sampler "
                                         "images (--role sampler)")
        plan = image_smoke.trainer_plan(
            request.probe_node, model=request.model, config=_model_config(args), engine=request.engine,
            store=args.smoke_store, run_id=run_id, workdir=workdir, model_revision=args.model_revision,
            model_path=args.model_path, layers=args.smoke_layers, extra_args=tuple(args.smoke_arg),
            keep=args.keep_smoke_artifacts, attention_backend=request.attention_backend)
        return plan
    spec = _sampler_spec(args)
    return image_smoke.sampler_plan(request.probe_node, spec=spec, run_id=run_id, release=args.release,
                                    keep=args.keep_smoke_artifacts, sudo=request.container_sudo)


@dataclass(frozen=True)
class PoolView:
    """The ssh sampler pool a sampler image is adopted for (``--pool-config``): its nodes
    as the pool starts them (:func:`.ssh_sampler_pool.pool_profile`: the pool's
    ``container.engine``, whatever the inventory says), its ``container.sudo`` and
    ``container.pull``."""

    path: str
    nodes: tuple[Node, ...]
    sudo: bool
    pull: str


def pool_view(path: str, profiles: Sequence[Node]) -> PoolView:
    """The pool config at ``path`` read the way the pool reads it
    (:func:`.ssh_sampler_pool.parse_config` over the adoption's inventory ``profiles``:
    its ``nodes:``, its container section). Raises ValueError for a file that is not a
    pool config, a pool the pool itself refuses, and a ``runtime: venv`` pool (it runs
    no image)."""
    import yaml  # noqa: PLC0415

    from . import ssh_sampler_pool  # noqa: PLC0415

    try:
        cfg = yaml.safe_load(Path(path).expanduser().read_text())
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"--pool-config {path}: cannot be read ({type(error).__name__}: {error})"[:600]) from None
    if not isinstance(cfg, Mapping):
        raise ValueError(f"--pool-config {path}: not a pool config (a mapping with a provider section)")
    provider = cfg.get("provider")
    if not isinstance(provider, Mapping):
        raise ValueError(f"--pool-config {path}: provider is not a mapping")
    if provider.get("container") is not None and not isinstance(provider.get("container"), Mapping):
        raise ValueError(f"--pool-config {path}: provider.container is not a mapping")
    try:
        conf = ssh_sampler_pool.parse_config(dict(cfg), inventory=list(profiles))
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError(f"--pool-config {path}: {error}") from None
    if conf.runtime != "container" or conf.container is None:
        raise ValueError(f"--pool-config {path}: a runtime: {conf.runtime} pool runs its samplers from a venv on "
                         "the host, not from an image: there is no image to adopt for it")
    return PoolView(str(path), tuple(ssh_sampler_pool.pool_profile(n, conf) for n in conf.nodes),
                    bool(conf.container.sudo), str(conf.container.pull))


def container_sudo(args: Any, pool: PoolView | None = None) -> bool:
    """``--container-sudo``, else the pool config's ``container.sudo`` (the inventory has
    no such key), else off. The flag against a pool that runs its engine without sudo is
    a ValueError: the adoption would fill another store than the one the pool starts from."""
    flag = bool(getattr(args, "container_sudo", False))
    if pool is None:
        return flag
    if flag and not pool.sudo:
        raise ValueError(f"--container-sudo contradicts --pool-config {pool.path}: its container.sudo is off (the "
                         "pool runs the engine as the ssh user); drop the flag or fix the pool config")
    return pool.sudo


def _by_engine(nodes: Sequence[Node]) -> dict[str, list[str]]:
    """The node names per container engine (docker / podman; a node of none is left out)."""
    out: dict[str, list[str]] = {}
    for n in nodes:
        engine = engine_of(n)
        if engine:
            out.setdefault(engine, []).append(n.name)
    return dict(sorted(out.items()))


def _sampler_spec(args: Any) -> dict[str, Any]:
    from . import image_smoke  # noqa: PLC0415

    if not args.sampler_spec:
        raise image_smoke.SmokeError("the sampler smoke runs the pool's sampler spec: pass --sampler-spec")
    return json.loads(Path(args.sampler_spec).expanduser().read_text())


def main(args: Any, *, ops: Any = None, out: Any = None) -> int:
    """``yeto image adopt``: 0 recorded (or a dry run), 1 a step failed, 2 a bad request."""
    import yaml  # noqa: PLC0415

    from . import image_smoke  # noqa: PLC0415
    from .node_inventory import InventoryError, load_profiles  # noqa: PLC0415
    from .preflight.cluster import engine_command  # noqa: PLC0415

    out = out or sys.stdout
    try:
        # profiles: an inventory's container (podman) and the other node profile keys count here
        profiles = load_profiles(args.inventory)
        pool = None
        if getattr(args, "pool_config", None):
            if args.role != "sampler":
                raise ValueError("--pool-config is an ssh sampler pool's config: it is for --role sampler")
            pool = pool_view(args.pool_config, profiles)  # the pool's nodes, engine, sudo and pull
        nodes = list(pool.nodes) if pool else role_nodes(profiles, args.role)
        sudo = container_sudo(args, pool)
        catalog = args.catalog or str(site_images.inventory_catalog_path(args.inventory))
        spec_engine = engine_images.spec_needs(_sampler_spec(args))["engine_pref"] if args.role == "sampler" \
            and args.sampler_spec else None
        # trainer: the backend the fit checks is the one the smoke trains with (--attention-backend or a --smoke-arg)
        attention = image_smoke.smoke_attention(args.attention_backend, tuple(args.smoke_arg)) \
            if args.role == "trainer" else args.attention_backend
        request = AdoptRequest(ref=args.ref, model=args.model, role=args.role, nodes=tuple(nodes),
                               engine=args.engine, node=args.node, catalog=catalog,
                               min_image_free_gib=args.min_image_free_gib, attention_backend=attention,
                               strict_arch=args.strict_arch, allow_busy=args.allow_busy, spec_engine=spec_engine,
                               container_sudo=sudo)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S")
        slug = re.sub(r"[^a-z0-9]+", "-", args.model.rsplit("/", 1)[-1].lower()).strip("-")[:24] or "model"
        run_id = f"adopt-{slug}-{args.role}-{stamp}"
        plan = _plan(args, request, run_id)
        if args.role == "trainer" and not request.engine:  # the record names the backend the smoke ran
            request = dataclasses.replace(request, engine=str(plan.data["engine"]))
    except (InventoryError, site_images.SiteImagesError, image_smoke.SmokeError, FileNotFoundError, OSError,
            ValueError, TypeError, yaml.YAMLError) as error:
        print(f"yeto image adopt: {error}", file=sys.stderr)
        return 2
    if args.dry_run:
        node = request.probe_node
        role = [n.name for n in request.nodes]
        lines = [f"yeto image adopt (dry run): {args.ref} for {args.model} as {args.role} on {len(role)} node(s) "
                 f"{role}",
                 f"1. manifest: HEAD of {args.ref} in its registry; a tag is pinned to its digest "
                 f"(below: {image_smoke.REF})",
                 f"2. probe on {node.name}: {probe_script(image_smoke.REF, node, sudo=sudo)[:200]} ...",
                 f"3. fit: CPU, device runtime, built card architectures and the {args.role} engine on {role}"
                 + (f"; the trainer's {request.attention_backend} attention (flash: {FLASH_TRAINER_MIN_ARCH}+ cards)"
                    if args.role == "trainer" else ""),
                 f"4. pull: ssh + host probe (image root >= {args.min_image_free_gib:g} GiB free), then "
                 + "; ".join(f"{engine_command(engine, sudo)} pull {image_smoke.REF} on {names}"
                             for engine, names in _by_engine(request.nodes).items())
                 + " in parallel"
                 + "".join(f"; not pulled (no container engine): {n.name}" for n in request.nodes if not engine_of(n))
                 + (f"; the pool's container.pull is never (it starts only images already on its nodes): this "
                    "pull is what puts the image there" if pool and pool.pull == "never" else ""),
                 "5. " + "\n".join(plan.lines()),
                 f"then: a verified-smoke record in {catalog}"]
        print("\n".join(lines), file=out)
        return 0

    def smoke(adopter: Adopter, ref: str, facts: ImageFacts, engine: str) -> StepResult:
        return image_smoke.run_plan(plan, ops=adopter.ops, ref=ref, python=facts.python_path, log=adopter.log)

    adopter = Adopter(request, ops=ops, smoke=smoke)
    try:
        report = adopter.run()
    except AdoptError as error:
        print("\n".join(adopter.report.lines()), file=out)
        print(f"yeto image adopt: {error}; the site image catalog {catalog} is unchanged", file=sys.stderr)
        return 1
    print("\n".join(report.lines()), file=out)
    return 0
