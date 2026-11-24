"""Node preflight: one read-only shell for every device family, and its checks.

This joins the two preflights the routes had: the Daytona sandbox checks
(``/dev/kfd`` + ``/dev/dri``, ``/dev/shm``, memory, ``memlock``, free and writable
``/work`` + ``/models``, the sampler interpreter and its modules, the GPUs torch
sees and their architecture) and the SSH node checks (``uname -m`` against the
profile, the card tool against the expected card and count, the Ascend host
devices and driver mounts, cards busy with other processes, free GiB of the work
directory). Which of them apply follows the profile's device family and launch:
a container launch checks the interpreter inside the image, not on the host.

The script prints ``YETO_NODE_FACT <key> <value>`` lines, one ``YETO_NODE_PY
{json}`` line (process launch) and the card tool's output between
``YETO_NODE_CARDS_BEGIN`` / ``YETO_NODE_CARDS_END``; :func:`parse_facts` reads
them, :func:`evaluate` turns them into :class:`Check` rows. The facts are what
the ready record carries about the node.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from yeto.launch.node_inventory import NodeProfile

from .profile import is_container, node_dirs

DEFAULT_MODULES = ("boto3", "zstandard", "httpx", "numpy")

PREFLIGHT_PY = r"""
import importlib.util, json, sys
mods = [m for m in sys.argv[1].split(",") if m]
out = {"missing": [m for m in mods if importlib.util.find_spec(m) is None]}
try:
    import torch
    out["torch"] = torch.__version__
    out["hip"] = getattr(torch.version, "hip", None)
    out["available"] = bool(torch.cuda.is_available())
    n = torch.cuda.device_count() if out["available"] else 0
    out["count"] = n
    archs = set()
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        name = getattr(p, "gcnArchName", "") or ""
        archs.add(name.split(":")[0] if name else "sm%d%d" % (p.major, p.minor))
    out["arch"] = sorted(archs)
except Exception as exc:
    out["error"] = repr(exc)[:300]
print("YETO_NODE_PY " + json.dumps(out))
"""


@dataclass
class Requirements:
    min_free_gib: float = 20.0
    shm_gib: float = 0.0
    memory_gib: float = 0.0
    require_idle: bool = False
    python: str | None = None  # the sampler interpreter of a process launch (None: not checked)
    modules: tuple[str, ...] = DEFAULT_MODULES
    gpu_check: bool = False  # the interpreter must see the profile's cards (torch)
    store_endpoint: str | None = None  # an URL the node must reach (None: not checked)
    sudo: bool = False  # the container engine runs under sudo

    @classmethod
    def from_doc(cls, doc: Mapping[str, Any] | None) -> "Requirements":
        values = dict(doc or {})
        unknown = sorted(set(values) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"preflight: unknown requirement(s) {unknown}")
        if "modules" in values:
            values["modules"] = tuple(values["modules"])
        return cls(**values)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    warn: bool = False  # recorded, not refused


@dataclass
class Facts:
    facts: dict[str, str] = field(default_factory=dict)
    py: dict[str, Any] = field(default_factory=dict)
    cards_text: str = ""
    host_paths: list[str] = field(default_factory=list)

    def record(self) -> dict[str, Any]:
        out = {k: v for k, v in self.facts.items() if not k.startswith("write:")}
        if self.py:
            out["python"] = {k: self.py[k] for k in ("torch", "hip", "count", "arch", "missing") if k in self.py}
        if self.host_paths:
            out["host_paths"] = list(self.host_paths)
        return out


def preflight_script(profile: NodeProfile, req: Requirements) -> str:
    family, dirs = profile.family, node_dirs(profile)
    lines = ["set -u", 'f() { echo "YETO_NODE_FACT $1 ${2:-}"; }',
             f"mkdir -p {shlex.quote(dirs['work'])} 2>/dev/null",
             'f arch "$(uname -m)"', 'f user "$(id -un) $(id -Gn | tr " " ,)"']
    if family == "rocm":
        lines += ['if [ -e /dev/kfd ] && [ -r /dev/kfd ] && [ -w /dev/kfd ]; then f kfd rw; '
                  'elif [ -e /dev/kfd ]; then f kfd noaccess; else f kfd absent; fi',
                  "f dri $(ls /dev/dri 2>/dev/null | grep -c '^renderD')",
                  'f driver "$(cat /sys/module/amdgpu/version 2>/dev/null || echo -) $(uname -r)"']
    elif family in ("cuda", "ppu"):
        lines += ["f gpu_dev $(ls /dev 2>/dev/null | grep -c '^nvidia[0-9]')",
                  "f driver \"$(cat /proc/driver/nvidia/version 2>/dev/null | head -1 | awk '{print $8}') "
                  "$(uname -r)\""]
    elif family == "ascend":
        lines += ["f davinci $(ls /dev 2>/dev/null | grep -c '^davinci[0-9]')",
                  f"for p in {' '.join(profile.host_paths())}; do [ -e $p ] && echo YETO_NODE_HOSTPATH $p; done"]
    lines += ["f shm_kib $(df -Pk /dev/shm 2>/dev/null | awk 'NR==2{print $2}')",
              "f mem_kib $(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null)",
              'f memlock "$(ulimit -l)"']
    for d in dict.fromkeys((dirs["work"], dirs["models"])):
        q = shlex.quote(d)
        lines.append(f"f free_kib:{d} $(df -Pk {q} 2>/dev/null | awk 'NR==2{{print $4}}'); "
                     f"t={q}/.yeto-write-probe; if (: > $t) 2>/dev/null; then rm -f $t; f write:{d} ok; "
                     f"else f write:{d} no; fi")
    if is_container(profile):
        engine = ("sudo -n " if req.sudo else "") + profile.container
        lines.append(f"command -v {profile.container} >/dev/null && "
                     f"f container_engine \"$({engine} --version 2>&1 | head -1)\" || f container_engine absent")
    if req.python:
        py = shlex.quote(req.python)
        lines += [f"[ -x {py} ] || command -v {py} >/dev/null && f python ok || f python absent",
                  f"f vllm \"$({py} -c 'import vllm; print(vllm.__version__)' 2>/dev/null | tail -1)\"",
                  f"cd /tmp && {py} -c {shlex.quote(PREFLIGHT_PY)} {shlex.quote(','.join(req.modules))} 2>&1 | "
                  "grep '^YETO_NODE_PY' || echo 'YETO_NODE_PY {\"error\": \"interpreter failed\"}'"]
    if req.store_endpoint:
        lines.append(f"c=$(curl -s -o /dev/null -m 15 -w '%{{http_code}}' {shlex.quote(req.store_endpoint)} "
                     "2>/dev/null); f store \"${c:-000}\"")
    try:
        from yeto.hw import card_tools  # noqa: PLC0415

        commands = list(card_tools.preflight_commands(profile.device_family))
        if family == "rocm":
            commands += list(getattr(card_tools, "ROCM_SMI_FALLBACK", ()))
    except ValueError:
        commands = []
    # rocm: amd-smi, else rocm-smi (older images); the others: the family's first tool
    # (npu-smi info: its process table is what the idle check reads)
    tools = " || ".join(shlex.join(c) for c in (commands if family == "rocm" else commands[:1])) or "true"
    lines.append(f"echo YETO_NODE_CARDS_BEGIN; {{ {tools}; }} 2>&1; echo YETO_NODE_CARDS_END rc=$?")
    return "\n".join(lines) + "\n"


def parse_facts(text: str) -> Facts:
    out = Facts()
    out.cards_text = (text or "").partition("YETO_NODE_CARDS_BEGIN")[2].partition("YETO_NODE_CARDS_END")[0]
    for line in (text or "").splitlines():
        if line.startswith("YETO_NODE_FACT "):
            _, key, *rest = line.split(" ", 2)
            out.facts[key] = rest[0].strip() if rest else ""
        elif line.startswith("YETO_NODE_HOSTPATH "):
            out.host_paths.append(line.split(" ", 1)[1].strip())
        elif line.startswith("YETO_NODE_PY "):
            try:
                out.py = json.loads(line.split(" ", 1)[1])
            except ValueError:
                out.py = {"error": "unreadable interpreter report"}
    return out


def _int(value: str | None) -> int | None:
    try:
        return int(str(value).split()[0])
    except (TypeError, ValueError, IndexError):
        return None


def busy_npus(text: str) -> list[int]:
    """NPU ids with a process row in ``npu-smi info`` (the process table: ``| <npu> <chip> | <pid> | ...``)."""
    busy: set[int] = set()
    in_procs = False
    for line in text.splitlines():
        if "Process id" in line:
            in_procs = True
            continue
        if not in_procs or "No running processes" in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[1].isdigit():
            head = cells[0].split()
            if head and head[0].isdigit():
                busy.add(int(head[0]))
    return sorted(busy)


def evaluate(found: Facts, profile: NodeProfile, req: Requirements, *,
             cards: Sequence[int] | None = None) -> list[Check]:
    """The checks of a preflight report against the profile and ``req`` (pure)."""
    facts, py, out = found.facts, found.py, []
    dirs = node_dirs(profile)

    def add(name: str, ok: bool, detail: str, *, warn: bool = False) -> None:
        out.append(Check(name, ok, detail, warn))

    from yeto.launch import engine_images  # noqa: PLC0415

    try:
        arch = engine_images.canonical_cpu_arch(facts.get("arch", ""))
        add("cpu_arch", arch == profile.cpu_arch, f"uname -m {arch}, the profile says {profile.cpu_arch}")
    except engine_images.UnknownCpuArch:
        add("cpu_arch", False, f"uname -m reports {facts.get('arch', '')!r}")
    want = profile.cards_per_node if cards is None else len(cards)
    family = profile.family
    if family == "rocm":
        add("kfd", facts.get("kfd") == "rw", f"/dev/kfd {facts.get('kfd', 'unreported')}")
        dri = _int(facts.get("dri")) or 0
        add("dri", dri >= want, f"{dri} render nodes for {want} GPUs")
    elif family in ("cuda", "ppu"):
        devs = _int(facts.get("gpu_dev")) or 0
        add("gpu_dev", devs >= want, f"{devs} /dev/nvidia<i> for {want} GPUs", warn=family == "ppu")
    elif family == "ascend":
        devs = _int(facts.get("davinci")) or 0
        add("davinci", devs >= want, f"{devs} /dev/davinci<i> for {want} NPUs")
    if found.cards_text.strip() and profile.card:
        try:
            seen = profile.check_cards(found.cards_text)
            add("cards", len(seen) >= want, f"{len(seen)} x {profile.card}")
        except Exception as error:  # noqa: BLE001 - the card tool's own refusal, as a check
            add("cards", False, f"card check: {error}")
    else:
        add("cards", bool(found.cards_text.strip()), "card tool answered" if found.cards_text.strip()
            else "no card tool answered", warn=True)
    if req.require_idle and family == "ascend":
        mine = set(range(profile.cards_per_node) if cards is None else cards)
        busy = [b for b in busy_npus(found.cards_text) if b in mine]
        add("idle", not busy, f"cards {busy} run other processes" if busy else "no other process on the cards")
    if req.shm_gib:
        shm = (_int(facts.get("shm_kib")) or 0) / (1 << 20)
        add("shm", shm >= req.shm_gib, f"/dev/shm {shm:.0f} GiB (need {req.shm_gib:g})")
    if req.memory_gib:
        mem = (_int(facts.get("mem_kib")) or 0) / (1 << 20)
        add("memory", mem >= req.memory_gib, f"{mem:.0f} GiB (need {req.memory_gib:g})")
    for d in dict.fromkeys((dirs["work"], dirs["models"])):
        add(f"write:{d}", facts.get(f"write:{d}") == "ok", f"{d} writable={facts.get(f'write:{d}', 'unreported')}",
            warn=d == dirs["models"] and d != dirs["work"] and profile.model_source in ("volume", "preset", None))
    if req.min_free_gib:
        free = (_int(facts.get(f"free_kib:{dirs['work']}")) or 0) / (1 << 20)
        add("disk", free >= req.min_free_gib, f"{dirs['work']}: {free:.0f} GiB free (need {req.min_free_gib:g})")
    if is_container(profile):
        add("container_engine", facts.get("container_engine", "absent") != "absent",
            f"{profile.container}: {facts.get('container_engine', 'unreported')}")
    if req.python:
        add("python", facts.get("python") == "ok", f"{req.python} {facts.get('python', 'unreported')}")
        missing = list(py.get("missing") or [])
        add("modules", not missing, f"missing {missing}" if missing else "all importable")
        if req.gpu_check:
            count = int(py.get("count") or 0)
            add("gpus", count >= want, f"torch sees {count} of {want}"
                + (f" ({py['error']})" if py.get("error") else ""))
            if profile.card_arch:
                archs = list(py.get("arch") or [])
                add("card_arch", archs == [profile.card_arch], f"archs {archs or '-'} (profile {profile.card_arch})")
    if req.store_endpoint:
        code = _int(facts.get("store")) or 0
        add("store", 0 < code < 500, f"object store endpoint HTTP {code or 'no answer'}")
    return out


class PreflightError(RuntimeError):
    def __init__(self, node: str, failed: Sequence[Check]):
        self.failed = list(failed)
        super().__init__(f"node {node}: preflight refused: " + "; ".join(f"[{c.name}] {c.detail}" for c in self.failed))


def run_preflight(profile: NodeProfile, req: Requirements, *, cards: Sequence[int] | None = None,
                  run: Callable[..., Any] = subprocess.run,
                  environ: Mapping[str, str] | None = None) -> tuple[Facts, list[Check]]:
    """Run the preflight on this host (the node runtime); raises :class:`PreflightError`."""
    out = run(["bash", "-c", preflight_script(profile, req)], capture_output=True, text=True, timeout=600,
              check=False, **({"env": dict(environ)} if environ is not None else {}))
    found = parse_facts(out.stdout or "")
    if not found.facts:
        raise PreflightError(profile.name, [Check("exec", False, f"no report (exit {out.returncode}: "
                                                                 f"{(out.stderr or '').strip()[-300:]})")])
    checks = evaluate(found, profile, req, cards=cards)
    failed = [c for c in checks if not c.ok and not c.warn]
    if failed:
        raise PreflightError(profile.name, failed)
    return found, checks
