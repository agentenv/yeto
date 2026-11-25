"""Bare-metal cluster preflight for SSH trainer nodes (``yeto preflight``).

Before a multi-node trainer starts on nodes of the SSH node inventory
(:mod:`.node_inventory`), this finds what would make it fail or run slowly,
node by node and across nodes, and writes a PASS / WARN / FAIL table with the
fix for every finding into the run directory.

Per node (one SSH call runs every host probe; a second one runs the trainer
image with the inventory's ``docker_args``, as the trainer would):

- NVIDIA driver (>= 580 for the CUDA 13 trainer image), GPU count, model and
  HBM (``card_tools.check_cards``), NVSwitch fabric state, the NVLink matrix
  and the GPU -> NIC PCIe affinity (``nvidia-smi topo -m``);
- IB / RoCE HCAs (``ibstat``, else ``ibv_devinfo``): port state, link layer,
  rate and MTU of the HCAs NCCL will use (``NCCL_IB_HCA`` of the inventory
  ``env``, else every HCA on the fabric's link layer); RoCE v2 GIDs and
  ``NCCL_IB_GID_INDEX``;
- ``/dev/infiniband`` and the locked-memory limit on the host and inside the
  container (a container inherits dockerd's limit unless ``--ulimit memlock``);
- host memory, free space of the mounted host paths, the Docker data root and
  ``/var/lib/containerd``; Docker version, storage driver (overlay2 or the
  containerd snapshotter: images with a same-layer whiteout do not unpack
  there), NVIDIA runtime; the trainer image pull; clock sync and skew to the
  head; the rendezvous ``address`` on the node's interface.

Across nodes (inside the trainer container, the scripts of
:mod:`.bench` and ``ib_write_bw`` of the image):

- TCP reachability of the Ray / torch rendezvous ports between every pair;
- ``ib_write_bw`` between every pair of nodes, rail by rail (HCA i to HCA i),
  in GB/s;
- NCCL ``all_reduce`` / ``all_to_all`` / ``sendrecv`` on disjoint node pairs and
  on all nodes, with ``NCCL_DEBUG=INFO``: a collective that runs over
  ``NET/Socket`` instead of ``NET/IB`` is a FAIL, a missing GPUDirect RDMA a WARN.

Bandwidth limits are fractions of the HCAs' link rate (``Requirements``); they
flag a rail or a node far below its link, not a few percent of tuning.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ...hw import card_tools
from ..node_inventory import InventoryError, Node, island_nodes, load_inventory
from ..ssh_trainer_island import (
    CUDA_RDMA_DOCKER_ARGS,
    RDMA_FABRICS,
    SshIslandError,
    SshOps,
    cuda_docker_problems,
    inventory_trainer_nodes,
    rendezvous_address,
)
from ..topology import island_comm_env

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"
_ORDER = {FAIL: 0, WARN: 1, SKIP: 2, PASS: 3}
SECTION = "@@yeto-section"
RC = "@@yeto-rc"
BENCH = "@@yeto-bench"
IB_MARK = "@@yeto-ib"
CONTAINER_DIR = "/yeto-preflight"
BENCH_FILE = "preflight_bench.py"
_DELIM = "__YETO_PREFLIGHT_EOF__"
# yeto-miles-k31 images whose layer 115 whites out a directory and writes it again in the
# same layer: overlay appliers (moby overlay2, the containerd overlayfs snapshotter) refuse
# to unpack it ("not a directory"); docker/miles-ports/make_layer.py builds a fixed layer.
# The same set as the miles pins' OVERLAY_SAFE_DIGESTS keys plus OVERLAY_UNSAFE_UNREPACKED.
OVERLAY_UNSAFE_DIGESTS = (
    "63ad0d2a9e70a139221050f51214f073a5bc53f66c0c57090ae926012e439c04",
    "2de5d796de19d38f4e3165efce2a9528a7f7d3e60e3cb32fe29fd00c6abf9b5b",
    "028e505de58552e6ede8926c8bf8ab8451e8e749cfc35afc1eb70e8d362c65b3",
)
OVERLAY_DRIVERS = ("overlay2", "overlayfs", "overlay")
TOPO_RANK = {"PIX": 0, "PXB": 1, "PHB": 2, "NODE": 3, "SYS": 4}


@dataclass(frozen=True)
class Requirements:
    """Limits of the checks (the defaults: the K3.1 6 x 8 B300 trainer)."""

    min_driver_major: int = 580  # the trainer image is CUDA 13.0 (NVIDIA_REQUIRE_CUDA=cuda>=13.0)
    min_host_memory_gib: float = 1554.0  # PP3 x 48 cards x 262k, measured fit per node
    rec_host_memory_gib: float = 1800.0  # headroom below this is a WARN
    min_disk_free_gib: float = 2000.0  # weights ~1.5 TB + one cut stage ~0.9 TB
    rec_disk_free_gib: float = 6000.0  # weights, two cut stages, releases, Ray spill
    min_image_root_free_gib: float = 120.0  # an unpacked trainer image ~85-100 GB
    warn_clock_skew_s: float = 1.0
    fail_clock_skew_s: float = 60.0
    ib_pass_fraction: float = 0.8  # ib_write_bw of a rail vs its link rate
    ib_fail_fraction: float = 0.5
    all_reduce_fraction: float = 0.6  # busbw vs the node's HCA aggregate
    all_to_all_fraction: float = 0.5  # busbw vs the HCA bound of all_to_all
    sendrecv_fraction: float = 0.6  # busbw of one GPU vs one HCA
    min_mtu: int = 4096  # IB / RoCE active MTU
    # Ray head (GCS, dashboard, client), torch rendezvous, samples of the worker and
    # ephemeral ranges, and Ray's per-node agent ports (52365 fixed by Ray; 52366-52368
    # pinned on SSH nodes, rl_island_task.RAY_AGENT_PORTS)
    ports: tuple[int, ...] = (6379, 8265, 10001, 29500, 20000, 40000, 60000, 52365, 52366, 52367, 52368)


# Site overrides of the host memory / disk limits, read by ``yeto preflight`` and by the
# launch gate (which builds its own Requirements): a cluster whose fit was measured with
# less memory, or that copies only its pipeline stage's weights, sets them instead of
# skipping the whole preflight.
REQ_ENV = {"min_host_memory_gib": "YETO_PREFLIGHT_MIN_HOST_MEMORY_GIB",
           "min_disk_free_gib": "YETO_PREFLIGHT_MIN_DISK_FREE_GIB"}


def requirements_from_env(environ: Mapping[str, str] | None = None) -> Requirements:
    """:class:`Requirements` with the :data:`REQ_ENV` overrides that are set."""
    import os

    environ = os.environ if environ is None else environ
    kwargs: dict[str, float] = {}
    for name, var in REQ_ENV.items():
        raw = str(environ.get(var, "") or "").strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            raise ValueError(f"{var} must be a number of GiB, got {raw!r}") from None
        if not value >= 0:
            raise ValueError(f"{var} must be >= 0, got {raw!r}")
        kwargs[name] = value
    return Requirements(**kwargs)


@dataclass
class Check:
    scope: str  # node name, "a<->b", "cluster" or "inventory"
    name: str
    status: str
    detail: str
    fix: str = ""
    data: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ parsers
@dataclass(frozen=True)
class Section:
    rc: int | None
    text: str


def parse_sections(text: str) -> dict[str, Section]:
    """Output of a probe script (:func:`probe_script`) by section name."""
    out: dict[str, Section] = {}
    name, lines = None, []
    for line in (text or "").splitlines():
        if line.startswith(SECTION + " "):
            if name is not None:
                out[name] = Section(None, "\n".join(lines))
            name, lines = line[len(SECTION) + 1:].strip(), []
        elif line.startswith(RC + " ") and name is not None:
            try:
                rc = int(line[len(RC) + 1:].strip())
            except ValueError:
                rc = None
            out[name] = Section(rc, "\n".join(lines))
            name, lines = None, []
        elif name is not None:
            lines.append(line)
    if name is not None:
        out[name] = Section(None, "\n".join(lines))
    return out


def probe_script(sections: Sequence[tuple[str, str]]) -> str:
    """Bash that runs every ``(name, command)`` and marks its output and exit code."""
    body = []
    for name, command in sections:
        body.append(f"echo '{SECTION} {name}'\n( {command}\n) 2>&1 < /dev/null\necho \"{RC} $?\"\n")
    return "".join(body)


def driver_versions(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if re.fullmatch(r"\d+(\.\d+)+", line.strip())]


def driver_major(version: str) -> int:
    return int(version.split(".")[0])


def parse_fabric(text: str) -> list[tuple[str, str]]:
    """``(state, status)`` per GPU of the ``Fabric`` blocks of ``nvidia-smi -q``."""
    out, state = [], None
    for line in text.splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()
        if key == "state":
            state = value
        elif key == "status" and state is not None:
            out.append((state, value))
            state = None
    if state is not None:
        out.append((state, ""))
    return out


@dataclass
class Topo:
    gpus: list[str]
    nics: list[str]
    link: dict[tuple[str, str], str]
    nic_names: dict[str, str]  # NIC0 -> mlx5_0


def parse_topo(text: str) -> Topo:
    """The matrix of ``nvidia-smi topo -m`` (GPU and NIC rows) and its NIC legend."""
    lines = text.splitlines()
    header: list[str] | None = None
    link: dict[tuple[str, str], str] = {}
    names: dict[str, str] = {}
    for line in lines:
        legend = re.match(r"\s*(NIC\d+)\s*:\s*(\S+)\s*$", line)
        if legend:
            names[legend.group(1)] = legend.group(2)
            continue
        if header is None:
            tokens = line.split()
            if tokens and tokens[0] == "GPU0":
                header = [t for t in tokens if re.fullmatch(r"(GPU|NIC)\d+", t)]
            continue
        tokens = line.split()
        if not tokens or not re.fullmatch(r"(GPU|NIC)\d+", tokens[0]) or tokens[0] in {r for r, _ in link}:
            continue
        for col, value in zip(header, tokens[1:1 + len(header)]):
            link[(tokens[0], col)] = value
    header = header or []
    return Topo(gpus=[h for h in header if h.startswith("GPU")], nics=[h for h in header if h.startswith("NIC")],
                link=link, nic_names=names)


@dataclass
class IbPort:
    ca: str
    port: int
    state: str  # Active / Down / Init / Armed / Polling
    physical: str = ""
    rate_gbps: float | None = None
    link_layer: str = ""  # InfiniBand / Ethernet
    firmware: str = ""
    active_mtu: int | None = None

    @property
    def active(self) -> bool:
        return self.state.lower() == "active"


def parse_ibstat(text: str) -> list[IbPort]:
    ports: list[IbPort] = []
    ca, fw, cur = None, "", None
    for raw in text.splitlines():
        line = raw.strip()
        m = re.match(r"CA '([^']+)'", line)
        if m:
            ca, fw, cur = m.group(1), "", None
            continue
        m = re.match(r"Port (\d+):$", line)
        if m and ca:
            cur = IbPort(ca=ca, port=int(m.group(1)), state="", firmware=fw)
            ports.append(cur)
            continue
        key, _, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()
        if key == "firmware version":
            fw = value
        if cur is None:
            continue
        if key == "state":
            cur.state = value
        elif key == "physical state":
            cur.physical = value
        elif key == "rate":
            try:
                cur.rate_gbps = float(value.split()[0])
            except (ValueError, IndexError):
                pass
        elif key == "link layer":
            cur.link_layer = value
    return ports


_SPEED = re.compile(r"([\d.]+)\s*Gbps")
_WIDTH = re.compile(r"(\d+)X")


def parse_ibv_devinfo(text: str) -> list[IbPort]:
    ports: list[IbPort] = []
    ca, fw, cur = None, "", None
    width = speed = None

    def close() -> None:
        if cur is not None and width and speed and cur.rate_gbps is None:
            cur.rate_gbps = width * speed

    for raw in text.splitlines():
        line = raw.strip()
        key, _, value = line.partition(":")
        key, value = key.strip().lower(), value.strip()
        if key == "hca_id":
            close()
            ca, fw, cur, width, speed = value, "", None, None, None
        elif key == "fw_ver":
            fw = value
        elif key == "port" and ca:
            close()
            width = speed = None
            cur = IbPort(ca=ca, port=int(value.split()[0]), state="", firmware=fw)
            ports.append(cur)
        elif cur is None:
            continue
        elif key == "state":
            word = value.split()[0] if value else ""
            cur.state = {"PORT_ACTIVE": "Active", "PORT_DOWN": "Down", "PORT_INIT": "Initializing",
                         "PORT_ARMED": "Armed"}.get(word, word)
        elif key == "link_layer":
            cur.link_layer = value
        elif key == "active_mtu":
            try:
                cur.active_mtu = int(value.split()[0])
            except (ValueError, IndexError):
                pass
        elif key == "active_width":
            m = _WIDTH.search(value)
            width = int(m.group(1)) if m else None
        elif key == "active_speed":
            m = _SPEED.search(value)
            speed = float(m.group(1)) if m else None
    close()
    return ports


@dataclass(frozen=True)
class Gid:
    dev: str
    port: int
    index: int
    gid: str
    ndev: str
    type: str

    @property
    def ipv4(self) -> str | None:
        parts = self.gid.split(":")
        if len(parts) == 8 and all(p == "0000" for p in parts[:5]) and parts[5] == "ffff":
            hi, lo = int(parts[6], 16), int(parts[7], 16)
            return f"{hi >> 8}.{hi & 255}.{lo >> 8}.{lo & 255}"
        return None


def parse_gids(text: str) -> list[Gid]:
    out = []
    for line in text.splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) < 6 or not parts[1].isdigit() or not parts[2].isdigit():
            continue
        out.append(Gid(dev=parts[0], port=int(parts[1]), index=int(parts[2]), gid=parts[3], ndev=parts[4],
                       type=parts[5].strip()))
    return out


def roce_v2_index(gids: Iterable[Gid], dev: str) -> int | None:
    """The lowest RoCE v2 GID index with an IPv4 address on ``dev``."""
    found = sorted(g.index for g in gids if g.dev == dev and "v2" in g.type.lower() and g.ipv4)
    return found[0] if found else None


def parse_meminfo(text: str) -> tuple[float | None, float | None]:
    """``(MemTotal, MemAvailable)`` in GiB."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        m = re.match(r"(MemTotal|MemAvailable):\s+(\d+)\s*kB", line.strip())
        if m:
            values[m.group(1)] = int(m.group(2)) / (1 << 20)
    return values.get("MemTotal"), values.get("MemAvailable")


def memlock_unlimited(text: str) -> bool | None:
    words = text.split()
    if not words:
        return None
    return words[-1] == "unlimited"


@dataclass(frozen=True)
class Disk:
    path: str
    fstype: str
    device: str
    size_gib: float
    free_gib: float
    mountpoint: str


def parse_disks(text: str) -> tuple[list[Disk], list[str]]:
    """Rows ``path<TAB>fstype<TAB><df -P -k line>``; paths that do not exist come back apart."""
    disks, missing = [], []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        path, fstype, df = parts[0], parts[1], parts[2].split()
        if parts[2].strip() == "missing" or len(df) < 6:
            missing.append(path)
            continue
        try:
            size, free = int(df[1]) / (1 << 20), int(df[3]) / (1 << 20)
        except ValueError:
            missing.append(path)
            continue
        disks.append(Disk(path=path, fstype=fstype or "?", device=df[0], size_gib=size, free_gib=free,
                          mountpoint=df[5]))
    return disks, missing


def parse_docker_info(text: str) -> dict[str, Any] | None:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                return None
    return None


def docker_snapshotter(info: Mapping[str, Any]) -> bool:
    """True when docker stores images in the containerd image store (Docker 29 default)."""
    status = info.get("DriverStatus") or []
    return any(isinstance(row, (list, tuple)) and len(row) == 2 and row[0] == "driver-type"
               and "snapshotter" in str(row[1]) for row in status)


def parse_clock(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("ntp="):
            value = line[4:].strip().lower()
            out["ntp_synced"] = {"yes": True, "no": False}.get(value)
        elif line.startswith("System time"):
            m = re.search(r"([\d.]+) seconds (fast|slow)", line)
            if m:
                out["chrony_offset_s"] = float(m.group(1)) * (1 if m.group(2) == "fast" else -1)
        elif line.startswith("Leap status"):
            out["leap"] = line.partition(":")[2].strip()
    return out


def parse_ib_write_bw(text: str) -> tuple[float | None, str]:
    """``(BW average in Gb/s, error)`` of one ``ib_write_bw`` run."""
    unit_gbit = None
    col = None
    for line in text.splitlines():
        if "BW average" in line:
            heads = re.findall(r"#?[A-Za-z]+(?: [A-Za-z]+)?\[[^\]]+\]|#bytes|#iterations", line)
            for i, h in enumerate(heads):
                if h.startswith("BW average"):
                    col, unit_gbit = i, "Gb/sec" in h
            continue
        if col is not None:
            nums = line.split()
            if len(nums) > col and all(re.fullmatch(r"[\d.]+", n) for n in nums[:col + 1]):
                value = float(nums[col])
                return (value if unit_gbit else value * 8 * 1.048576 / 1000.0), ""
    errors = [ln.strip() for ln in text.splitlines()
              if re.search(r"(?i)couldn't|unable|failed|error|no such|not found|refused", ln)]
    return None, (errors[0] if errors else (text.strip().splitlines() or ["no output"])[-1])[:200]


@dataclass
class NcclTransport:
    ib: bool = False
    socket: bool = False
    gdr: bool = False
    no_ib_device: bool = False
    warnings: list[str] = field(default_factory=list)
    devices: list[str] = field(default_factory=list)


def parse_nccl_log(text: str) -> NcclTransport:
    t = NcclTransport()
    for line in text.splitlines():
        if "NCCL" not in line:
            continue
        if re.search(r"Using network IB|via NET/IB|NET/IB : Using", line):
            t.ib = True
        if re.search(r"Using network Socket|via NET/Socket", line):
            t.socket = True
        if "GDRDMA" in line:
            t.gdr = True
        if re.search(r"NET/IB : No device found", line):
            t.no_ib_device = True
        m = re.search(r"NET/IB : Using (.*)", line)
        if m:
            t.devices = re.findall(r"\[\d+\]([A-Za-z0-9_.]+:\d+)", m.group(1))
        if "NCCL WARN" in line and len(t.warnings) < 5:
            t.warnings.append(line.strip()[-240:])
    return t


def bench_records(text: str) -> list[dict[str, Any]]:
    out = []
    for line in text.splitlines():
        idx = line.find(BENCH + " ")
        if idx < 0:
            continue
        try:
            out.append(json.loads(line[idx + len(BENCH) + 1:]))
        except json.JSONDecodeError:
            continue
    return out


def ib_blocks(text: str) -> dict[int, str]:
    """``ib_write_bw`` output of every rail (``@@yeto-ib <rail> <hca>`` blocks)."""
    out: dict[int, str] = {}
    rail, lines = None, []
    for line in text.splitlines():
        if line.startswith(IB_MARK + " "):
            if rail is not None:
                out[rail] = "\n".join(lines)
            rail, lines = int(line.split()[1]), []
        elif rail is not None:
            lines.append(line)
    if rail is not None:
        out[rail] = "\n".join(lines)
    return out


def hca_filter(spec: str, ports: Sequence[IbPort]) -> tuple[list[IbPort], list[str]]:
    """The ports ``NCCL_IB_HCA`` selects (prefix match; ``=`` exact; ``^`` excludes),
    and the names of the list that match no port."""
    s = spec.strip()
    exact = s.startswith("=")
    s = s.lstrip("=")
    exclude = s.startswith("^")
    s = s.lstrip("^")
    exact = exact or s.startswith("=")
    s = s.lstrip("=")
    wanted = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, port = item.partition(":")
        wanted.append((name, int(port) if port.isdigit() else None))

    def match(p: IbPort, name: str, port: int | None) -> bool:
        ok = p.ca == name if exact else p.ca.startswith(name)
        return ok and (port is None or p.port == port)

    if exclude:
        return [p for p in ports if not any(match(p, n, q) for n, q in wanted)], []
    selected = [p for p in ports if any(match(p, n, q) for n, q in wanted)]
    missing = [n if q is None else f"{n}:{q}" for n, q in wanted if not any(match(p, n, q) for p in ports)]
    return selected, missing


def compute_ports(node: Node, ports: Sequence[IbPort]) -> tuple[list[IbPort], list[str], str]:
    """The HCA ports NCCL will use on ``node``: ``(ports, names missing, how chosen)``."""
    spec = node.env.get("NCCL_IB_HCA")
    if spec:
        selected, missing = hca_filter(spec, ports)
        return _natural(selected), missing, f"NCCL_IB_HCA={spec}"
    layer = "infiniband" if node.fabric == "ib" else "ethernet"
    return _natural([p for p in ports if p.link_layer.lower() == layer]), [], f"link layer {layer}"


def _natural(ports: Iterable[IbPort]) -> list[IbPort]:
    key = lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.ca)] + [p.port]
    return sorted(ports, key=key)


def round_robin(n: int) -> list[list[tuple[int, int]]]:
    """Rounds of disjoint pairs that cover every pair of ``n`` nodes once (circle method)."""
    ids = list(range(n)) + ([None] if n % 2 else [])
    m = len(ids)
    rounds = []
    for _ in range(m - 1):
        pairs = [(ids[i], ids[m - 1 - i]) for i in range(m // 2)]
        rounds.append([tuple(sorted(p)) for p in pairs if None not in p])
        ids = [ids[0], ids[-1], *ids[1:-1]]
    return [r for r in rounds if r]


def disjoint_pairs(n: int) -> list[list[tuple[int, int]]]:
    """``(0,1), (2,3), ...`` in one round; an odd last node pairs with node 0 in a second round."""
    first = [(i, i + 1) for i in range(0, n - 1, 2)]
    return [first] + ([[(0, n - 1)]] if n % 2 and n > 1 else [])


# --------------------------------------------------------------- node probe
HOST_SECTIONS_BASE = (
    ("uname", "uname -m; uname -r"),
    ("gpus", "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits"),
    ("driver", "nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
    ("fabric", "nvidia-smi -q | grep -A3 -E '^ *Fabric *$'"),
    ("fabricmanager", "systemctl is-active nvidia-fabricmanager"),
    ("topo", "nvidia-smi topo -m"),
    ("ibstat", "ibstat"),
    ("ibv_devinfo", "ibv_devinfo -v"),
    ("gids", ("for d in /sys/class/infiniband/*; do [ -d \"$d\" ] || continue; dev=${d##*/}; "
             "for p in \"$d\"/ports/*; do port=${p##*/}; for g in \"$p\"/gids/*; do i=${g##*/}; "
             "gid=$(cat \"$g\" 2>/dev/null) || continue; "
             "case $gid in 0000:0000:0000:0000:0000:0000:0000:0000|fe80:0000:0000:0000:0000:0000:0000:0000) continue;; esac; "
             "t=$(cat \"$p/gid_attrs/types/$i\" 2>/dev/null); n=$(cat \"$p/gid_attrs/ndevs/$i\" 2>/dev/null); "
             "echo \"$dev $port $i $gid ${n:--} ${t:-?}\"; done; done; done")),
    ("devib", "ls -1 /dev/infiniband"),
    ("memlock", "ulimit -l"),
    ("meminfo", "grep -E '^(MemTotal|MemAvailable):' /proc/meminfo"),
    ("addr", "ip -o -4 addr show"),
    ("peermem", "uname -r; lsmod | grep -E '^(nvidia_peermem|nv_peer_mem) ' || echo 'peermem: not loaded'"),
    ("docker", "docker info --format '{{json .}}'"),
    ("clock", ("echo ntp=$(timedatectl show -p NTPSynchronized --value 2>/dev/null); "
              "chronyc tracking 2>/dev/null | grep -E '^(System time|Leap status)'; true")),
)


def host_sections(node: Node) -> list[tuple[str, str]]:
    paths = ["/", "/var/lib/containerd", "$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)",
             *[shlex.quote(p) for p in sorted(node.mounts)]]
    disk = ("for p in " + " ".join(f'"{p}"' if p.startswith("$(") else p for p in paths) + "; do "
            "if [ -e \"$p\" ]; then printf '%s\\t%s\\t%s\\n' \"$p\" \"$(findmnt -n -o FSTYPE -T \"$p\" 2>/dev/null)\" "
            "\"$(df -P -k \"$p\" | tail -1)\"; else printf '%s\\t-\\tmissing\\n' \"$p\"; fi; done")
    return [*HOST_SECTIONS_BASE, ("disk", disk)]


CONTAINER_SECTIONS = (
    ("memlock", "ulimit -l"),
    ("devib", "ls -1 /dev/infiniband"),
    ("gpus", "nvidia-smi -L"),
    ("ibstat", "ibstat"),
    ("ibv_devinfo", "ibv_devinfo -v"),
    ("tools", "for t in ib_write_bw torchrun python3 python; do printf '%s=%s\\n' $t \"$(command -v $t)\"; done"),
    ("torch", ("$(command -v python3 || command -v python) -c \"import torch; "
              "print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), "
              "'devices', torch.cuda.device_count(), 'nccl', '.'.join(map(str, torch.cuda.nccl.version())))\"")),
)


@dataclass
class NodeState:
    node: Node
    reachable: bool = False
    host: dict[str, Section] = field(default_factory=dict)
    container: dict[str, Section] = field(default_factory=dict)
    container_ok: bool = False
    ports: list[IbPort] = field(default_factory=list)
    compute: list[IbPort] = field(default_factory=list)
    gids: list[Gid] = field(default_factory=list)
    tools: dict[str, str] = field(default_factory=dict)
    ports_from_container: bool = False
    pulled: bool = True

    @property
    def name(self) -> str:
        return self.node.name

    @property
    def hca_gbps(self) -> float | None:
        rates = [p.rate_gbps for p in self.compute if p.active and p.rate_gbps]
        return sum(rates) if rates else None

    @property
    def min_rail_gbps(self) -> float | None:
        rates = [p.rate_gbps for p in self.compute if p.active and p.rate_gbps]
        return min(rates) if rates else None


def ports_of(sections: Mapping[str, Section]) -> list[IbPort]:
    """The IB / RoCE ports of a probe: ``ibstat``, with the MTU of ``ibv_devinfo``; else ``ibv_devinfo``."""
    ibstat = parse_ibstat(sections["ibstat"].text) if _ok(sections.get("ibstat")) else []
    devinfo = parse_ibv_devinfo(sections["ibv_devinfo"].text) if _ok(sections.get("ibv_devinfo")) else []
    if not ibstat:
        return devinfo
    mtu = {(p.ca, p.port): p.active_mtu for p in devinfo}
    for p in ibstat:
        p.active_mtu = p.active_mtu or mtu.get((p.ca, p.port))
    return ibstat


def _ok(sec: Section | None) -> bool:
    return sec is not None and sec.rc == 0


def _tail(text: str, n: int = 300) -> str:
    text = " ".join((text or "").split())
    return text[-n:]


def _dialable(node: Node) -> bool:
    try:
        rendezvous_address(node)
    except SshIslandError:
        return False
    return True


def rdma_island(nodes: Sequence[Node]) -> bool:
    return len(nodes) > 1 and any(n.fabric in RDMA_FABRICS for n in nodes)


# --------------------------------------------------------- inventory checks
def inventory_checks(nodes: Sequence[Node], image: str | None) -> list[Check]:
    out: list[Check] = []
    rdma = rdma_island(nodes)
    for node in nodes:
        if node.runtime == "docker" and node.device_family == "cuda":
            missing = cuda_docker_problems(node.docker_args, rdma=rdma and node.fabric in RDMA_FABRICS)
            if missing:
                out.append(Check(node.name, "docker_args", FAIL, "docker_args lack " + "; ".join(missing),
                                 f"docker_args: {list(CUDA_RDMA_DOCKER_ARGS)}"))
            else:
                out.append(Check(node.name, "docker_args", PASS, " ".join(node.docker_args)))
        if len(nodes) > 1:
            try:
                rendezvous_address(node)
            except SshIslandError as exc:
                out.append(Check(node.name, "address", FAIL, str(exc), "set address: <cluster-network IP>"))
            if not node.interface:
                out.append(Check(node.name, "interface", WARN,
                                 "no interface: NCCL/Gloo pick the bootstrap interface themselves "
                                 "(may be docker0 or a management network)",
                                 "set interface: <cluster-network interface> (exported as NCCL_SOCKET_IFNAME)"))
        if rdma and node.fabric not in RDMA_FABRICS:
            out.append(Check(node.name, "fabric", FAIL, f"fabric {node.fabric}: NCCL runs over TCP sockets",
                             "set fabric: ib or roce"))
        env = node.env
        if env.get("NCCL_IB_DISABLE") == "1" and rdma:
            out.append(Check(node.name, "env", FAIL, "env sets NCCL_IB_DISABLE=1 on an RDMA island",
                             "remove NCCL_IB_DISABLE from the inventory env"))
        if node.fabric == "roce" and rdma and "NCCL_IB_GID_INDEX" not in env:
            out.append(Check(node.name, "env", WARN, "RoCE without NCCL_IB_GID_INDEX: NCCL picks a GID itself",
                             "set env NCCL_IB_GID_INDEX to the RoCE v2 IPv4 GID index (see the gid check)"))
        img = image or node.image or ""
        bad = [d for d in OVERLAY_UNSAFE_DIGESTS if d in img]
        if bad:
            out.append(Check(node.name, "image", WARN,
                             f"image {img[:80]} has a same-layer whiteout that overlay2 / containerd "
                             "overlayfs cannot unpack ('not a directory')",
                             "use an image rebuilt with docker/miles-ports/make_layer.py (or a repacked layer)"))
    return out


# -------------------------------------------------------------- host checks
def host_checks(state: NodeState, req: Requirements, *, multi: bool, image: str | None) -> list[Check]:
    node, h = state.node, state.host
    name = node.name
    out: list[Check] = []
    rdma = multi and node.fabric in RDMA_FABRICS

    def add(check: str, status: str, detail: str, fix: str = "", **data: Any) -> None:
        out.append(Check(name, check, status, detail, fix, data))

    # cpu arch
    uname = h.get("uname")
    if _ok(uname):
        lines = uname.text.split()
        try:
            node.check_cpu_arch(lines[0] if lines else "")
            add("cpu_arch", PASS, f"{lines[0]}, kernel {lines[1] if len(lines) > 1 else '?'}")
        except InventoryError as exc:
            add("cpu_arch", FAIL, str(exc), "fix cpu_arch in the inventory or use the right node")
    if node.device_family == "cuda":
        _cuda_checks(state, req, add, rdma=rdma)
    # /dev/infiniband and memlock on the host
    devib = h.get("devib")
    if rdma or node.fabric in RDMA_FABRICS:
        entries = devib.text.split() if _ok(devib) else []
        if any(e.startswith("uverbs") for e in entries):
            add("dev_infiniband", PASS, f"{sum(e.startswith('uverbs') for e in entries)} uverbs devices"
                + (", rdma_cm" if "rdma_cm" in entries else ""))
        else:
            add("dev_infiniband", FAIL if rdma else WARN, "/dev/infiniband has no uverbs devices",
                "load the RDMA stack (MLNX_OFED / DOCA host or rdma-core: ib_uverbs, mlx5_ib)")
    lock = memlock_unlimited(h["memlock"].text) if _ok(h.get("memlock")) else None
    if lock is not None:
        if lock:
            add("memlock_host", PASS, "unlimited")
        else:
            value = h["memlock"].text.strip()
            status = FAIL if (rdma and node.runtime != "docker") else (WARN if rdma else PASS)
            add("memlock_host", status,
                f"ulimit -l {value} KiB in the SSH session" + (
                    " (the container's limit comes from dockerd and --ulimit; see memlock_container)"
                    if node.runtime == "docker" else ""),
                "set '* - memlock unlimited' in /etc/security/limits.conf and LimitMEMLOCK=infinity "
                "for dockerd (systemctl edit docker)")
    # host memory
    total, avail = parse_meminfo(h["meminfo"].text) if _ok(h.get("meminfo")) else (None, None)
    if total is not None:
        detail = f"MemTotal {total:.0f} GiB, MemAvailable {avail or 0:.0f} GiB (inventory {node.host_memory_gib:g})"
        if total < req.min_host_memory_gib:
            add("host_memory", FAIL, detail + f" < {req.min_host_memory_gib:g} GiB needed",
                "use nodes with >= 1.6 TB of host memory (2 TB recommended)", total_gib=total)
        elif avail is not None and avail < req.min_host_memory_gib:
            add("host_memory", WARN, detail + f": less than {req.min_host_memory_gib:g} GiB available",
                "stop what holds host memory (page cache is fine; check for leftover trainers / ray)",
                total_gib=total)
        elif total < req.rec_host_memory_gib:
            add("host_memory", WARN, detail + f": little headroom (< {req.rec_host_memory_gib:g} GiB)",
                total_gib=total)
        elif node.host_memory_gib > total * 1.05:
            add("host_memory", WARN, detail + ": the inventory declares more than the node has",
                "set host_memory_gib to the node's MemTotal", total_gib=total)
        else:
            add("host_memory", PASS, detail, total_gib=total)
    # disks
    if h.get("disk") is not None:
        disks, missing = parse_disks(h["disk"].text)
        by_path = {d.path: d for d in disks}
        mounts = [d for p, d in by_path.items() if p in node.mounts]
        for p in sorted(node.mounts):
            if p in missing:
                add("disk", FAIL, f"mounted host path {p} does not exist", f"mkdir -p {p} on a local NVMe")
        if not node.mounts and node.runtime == "docker":
            add("disk", WARN, "no mounts: weights, cut stages and caches land in the container's writable layer",
                "mounts: {/mnt/nvme/yeto: /root/.cache, /mnt/nvme/models: /data/models}")
        seen: dict[str, Disk] = {}
        for d in mounts:
            seen.setdefault(d.device + d.mountpoint, d)
        if seen:
            free = sum(d.free_gib for d in seen.values())
            where = ", ".join(f"{d.mountpoint} ({d.fstype} {d.device}) {d.free_gib:.0f} GiB free" for d in seen.values())
            fs_bad = [d for d in seen.values() if d.fstype in ("tmpfs", "overlay")]
            if free < req.min_disk_free_gib:
                add("disk", FAIL, f"{free:.0f} GiB free on the mounted paths: {where}",
                    f"free or add local NVMe (>= {req.rec_disk_free_gib / 1000:g} TB recommended)", free_gib=free)
            elif free < req.rec_disk_free_gib or fs_bad:
                add("disk", WARN, f"{free:.0f} GiB free on the mounted paths: {where}"
                    + ("; tmpfs/overlay mount" if fs_bad else ""),
                    f">= {req.rec_disk_free_gib / 1000:g} TB local NVMe recommended", free_gib=free)
            else:
                add("disk", PASS, f"{free:.0f} GiB free: {where}", free_gib=free)
        info = parse_docker_info(h["docker"].text) if _ok(h.get("docker")) else None
        image_roots = {}
        if info is not None and node.runtime == "docker":
            # the containerd image store (Docker 29 default) keeps images and container layers
            # under /var/lib/containerd, which does not follow docker's data-root
            snap = docker_snapshotter(info)
            want = "/var/lib/containerd" if snap else str(info.get("DockerRootDir") or "/var/lib/docker")
            label = f"{want} ({'containerd image store' if snap else 'docker data root'})"
            image_roots = {label: d for d in disks if d.path == want}  # noqa: B035 - one path
        for label, d in image_roots.items():
            if d.free_gib < req.min_image_root_free_gib:
                add("disk_images", FAIL, f"{label} on {d.mountpoint}: {d.free_gib:.0f} GiB free",
                    "move it to NVMe (docker data-root; bind-mount /var/lib/containerd: it does not "
                    "follow data-root)", free_gib=d.free_gib)
            else:
                add("disk_images", PASS, f"{label} on {d.mountpoint}: {d.free_gib:.0f} GiB free")
    # docker
    if node.runtime == "docker":
        _docker_checks(state, req, add, image=image)
    # clock
    clock = parse_clock(h["clock"].text) if h.get("clock") is not None else {}
    if clock.get("ntp_synced") is False:
        add("clock_sync", WARN, "NTPSynchronized=no", "enable chrony / systemd-timesyncd")
    elif clock.get("ntp_synced"):
        offset = clock.get("chrony_offset_s")
        add("clock_sync", PASS, "NTP synchronized" + (f", chrony offset {offset * 1e3:.2f} ms"
                                                     if offset is not None else ""))
    # rendezvous address on the node
    if multi and _ok(h.get("addr")) and node.address:
        rows = h["addr"].text.splitlines()
        hit = [r for r in rows if re.search(rf"\binet {re.escape(node.address)}/", r)]
        if not hit:
            add("address", FAIL, f"address {node.address} is not assigned on the node",
                "set address to the node's IP on the cluster network")
        elif node.interface and not any(r.split()[1].rstrip(":").split("@")[0] == node.interface for r in hit):
            add("address", FAIL, f"address {node.address} is on {hit[0].split()[1]}, not interface {node.interface}",
                "fix interface / address in the inventory")
        else:
            add("address", PASS, f"{node.address} on {hit[0].split()[1]}")
    return out


def _cuda_checks(state: NodeState, req: Requirements, add: Callable[..., None], *, rdma: bool) -> None:
    node, h = state.node, state.host
    gpus = h.get("gpus")
    if gpus is None or gpus.rc != 0:
        add("gpus", FAIL, f"nvidia-smi failed: {_tail(gpus.text if gpus else 'no output')}",
            "install the NVIDIA driver (>= 580, open kernel module) and reboot")
        return
    try:
        records = node.check_cards(gpus.text)
        hbm = min((r.hbm_gib for r in records if r.hbm_gib), default=0.0)
        add("gpus", PASS, f"{len(records)} x {node.card}, {hbm:.1f} GiB HBM each")
    except (card_tools.CardMismatch, InventoryError, ValueError) as exc:
        add("gpus", FAIL, str(exc), "fix the inventory (card, cards_per_node, hbm_gib) or the node's GPUs")
    versions = driver_versions(h["driver"].text) if h.get("driver") is not None else []
    if versions:
        majors = {driver_major(v) for v in versions}
        if len(set(versions)) > 1:
            add("driver", WARN, f"mixed driver versions {sorted(set(versions))}", "reinstall one driver version")
        elif min(majors) < req.min_driver_major:
            add("driver", FAIL, f"driver {versions[0]} < {req.min_driver_major} (the image needs CUDA 13)",
                f"install NVIDIA driver >= {req.min_driver_major} (open kernel module)", version=versions[0])
        else:
            add("driver", PASS, f"driver {versions[0]}", version=versions[0])
    fabric = parse_fabric(h["fabric"].text) if h.get("fabric") is not None else []
    real = [(s, st) for s, st in fabric if s.upper() != "N/A"]
    fm = (h.get("fabricmanager").text.strip() if h.get("fabricmanager") is not None else "")
    if real:
        bad = [(s, st) for s, st in real if s.lower() != "completed" or (st and st.lower() != "success")]
        if bad:
            add("nvlink_fabric", FAIL, f"{len(bad)}/{len(real)} GPUs fabric state {bad[0][0]}/{bad[0][1]} "
                f"(fabric manager: {fm or '?'})",
                "start nvidia-fabricmanager (and NVLSM on Blackwell HGX), versions matching the driver; "
                "CUDA init fails or hangs until the fabric is Completed")
        else:
            add("nvlink_fabric", PASS, f"{len(real)} GPUs fabric Completed/Success (fabric manager: {fm or '?'})")
    topo = parse_topo(h["topo"].text) if _ok(h.get("topo")) else None
    if topo and topo.gpus:
        pairs = [(a, b, topo.link.get((a, b), "?")) for a in topo.gpus for b in topo.gpus if a < b]
        bad = [(a, b, v) for a, b, v in pairs if not v.startswith("NV")]
        kinds = sorted({v for _, _, v in pairs if v.startswith("NV")})
        if bad:
            add("nvlink", FAIL, f"{len(bad)} GPU pairs without NVLink, e.g. {bad[0][0]}-{bad[0][1]} {bad[0][2]}",
                "check NVLink / NVSwitch health (nvidia-smi nvlink -s, fabric manager)")
        else:
            add("nvlink", PASS, f"{len(topo.gpus)} GPUs, all pairs {'/'.join(kinds)}")
        compute_names = {p.ca for p in state.compute}
        nics = [n for n in topo.nics if not compute_names or topo.nic_names.get(n, n) in compute_names]
        if rdma and nics:
            far = []
            for g in topo.gpus:
                best = min((TOPO_RANK.get(topo.link.get((g, n), "SYS"), 4) for n in nics), default=4)
                if best >= TOPO_RANK["NODE"]:
                    far.append(g)
            if far:
                add("gpu_nic_affinity", WARN, f"{len(far)} GPUs have no NIC on their PCIe switch / host bridge "
                    f"({', '.join(far[:4])}): GPUDirect RDMA crosses the CPU",
                    "use a rail-aligned topology (one NIC per GPU under the same PCIe switch)")
            else:
                add("gpu_nic_affinity", PASS, f"every GPU has a NIC at PIX/PXB/PHB ({len(nics)} NICs)")
    peer = h.get("peermem")
    if rdma and peer is not None and peer.rc == 0:
        text = peer.text
        kernel = text.split()[0] if text.split() else "?"
        loaded = "not loaded" not in text
        m = re.match(r"(\d+)\.(\d+)", kernel)
        dmabuf = bool(m and (int(m.group(1)), int(m.group(2))) >= (5, 12))
        if loaded or dmabuf:
            add("gpudirect", PASS, ("nvidia_peermem loaded" if loaded else "") + (", " if loaded and dmabuf else "")
                + (f"kernel {kernel} (DMA-BUF)" if dmabuf else ""))
        else:
            add("gpudirect", WARN, f"kernel {kernel} < 5.12 and nvidia_peermem not loaded: no GPUDirect RDMA",
                "modprobe nvidia_peermem, or a kernel >= 5.12 with the open driver (DMA-BUF)")


def ib_checks(state: NodeState, req: Requirements, *, multi: bool) -> list[Check]:
    """The HCAs NCCL will use on the node (``state.ports``: from the host, else the container)."""
    out: list[Check] = []

    def add(check: str, status: str, detail: str, fix: str = "", **data: Any) -> None:
        out.append(Check(state.name, check, status, detail + (" (read in the container)" if state.ports_from_container
                                                              else ""), fix, data))

    _ib_checks(state, req, add, rdma=multi and state.node.fabric in RDMA_FABRICS)
    return out


def _ib_checks(state: NodeState, req: Requirements, add: Callable[..., None], *, rdma: bool) -> None:
    node = state.node
    if not rdma and node.fabric not in RDMA_FABRICS:
        return
    ports = state.ports
    if not ports:
        add("ib_ports", FAIL if rdma else WARN, "no IB/RoCE port found (ibstat / ibv_devinfo on the host and "
            "in the container)", "install the RDMA stack and check the HCAs (ibstat)")
        return
    compute, missing, how = compute_ports(node, ports)
    state.compute = compute
    if missing:
        add("ib_hca", FAIL, f"NCCL_IB_HCA names {missing}, which no port matches",
            f"set NCCL_IB_HCA to HCAs of the node: {sorted({p.ca for p in ports})}")
    if not compute:
        layers = sorted({p.link_layer for p in ports})
        add("ib_ports", FAIL, f"no HCA selected by {how} (ports: {len(ports)}, link layers {layers})",
            f"fabric {node.fabric} but the HCAs are {layers}: fix fabric or NCCL_IB_HCA")
        return
    down = [f"{p.ca}:{p.port} {p.state}" for p in compute if not p.active]
    active = [p for p in compute if p.active]
    rates = sorted({p.rate_gbps for p in active if p.rate_gbps})
    detail = (f"{len(active)}/{len(compute)} active ({how})" + (f", rate {'/'.join(f'{r:g}' for r in rates)} Gb/s"
                                                                if rates else ""))
    if down:
        add("ib_ports", FAIL if (rdma and not active) or node.env.get("NCCL_IB_HCA") else WARN,
            detail + f"; down: {', '.join(down[:8])}",
            "bring the ports up (cable, switch port, subnet manager for IB: opensm / UFM) or drop them from "
            "NCCL_IB_HCA")
    elif len(active) < node.cards_per_node and node.device_family == "cuda":
        add("ib_ports", WARN, detail + f": fewer HCAs than {node.cards_per_node} GPUs (GPUs share NICs)",
            "check that every compute HCA is on the fabric")
    elif len(rates) > 1:
        add("ib_ports", WARN, detail + ": mixed link rates (a frontend / storage NIC among them?)",
            "check cables / switch port speeds; set NCCL_IB_HCA to the compute HCAs only")
    elif not node.env.get("NCCL_IB_HCA") and len(active) > node.cards_per_node and node.device_family == "cuda":
        add("ib_ports", WARN, detail + f": more HCAs than {node.cards_per_node} GPUs and no NCCL_IB_HCA, NCCL "
            f"also uses {', '.join(p.ca for p in active[node.cards_per_node:])}",
            "set env NCCL_IB_HCA to the compute HCAs (e.g. =mlx5_0,...,mlx5_7)")
    else:
        add("ib_ports", PASS, detail, hca_gbps=state.hca_gbps)
    mtus = sorted({p.active_mtu for p in active if p.active_mtu})
    if mtus and mtus[0] < req.min_mtu:
        add("ib_mtu", WARN, f"active MTU {mtus[0]} < {req.min_mtu}",
            "raise the MTU (IB: subnet manager MTU 4096; RoCE: interface MTU 9000)")
    if node.fabric == "roce" and active:
        want = node.env.get("NCCL_IB_GID_INDEX")
        found = {p.ca: roce_v2_index(state.gids, p.ca) for p in active}
        if not state.gids:
            add("gid", WARN, "no GID table read from /sys/class/infiniband", "check the RoCE GIDs (show_gids)")
        elif want is not None:
            bad = [ca for ca in found if not any(g.dev == ca and g.index == int(want) and "v2" in g.type.lower()
                                                 and g.ipv4 for g in state.gids)]
            if bad:
                add("gid", FAIL, f"NCCL_IB_GID_INDEX={want} is not a RoCE v2 IPv4 GID on {bad[:4]}",
                    f"set NCCL_IB_GID_INDEX to {sorted({v for v in found.values() if v is not None})}")
            else:
                add("gid", PASS, f"GID index {want} is RoCE v2 IPv4 on {len(found)} HCAs")
        else:
            idx = sorted({v for v in found.values() if v is not None})
            none = [ca for ca, v in found.items() if v is None]
            if none:
                add("gid", FAIL, f"no RoCE v2 IPv4 GID on {none[:4]}",
                    "assign an IPv4 address to the RoCE interfaces")
            else:
                add("gid", WARN if len(idx) == 1 else FAIL, f"RoCE v2 IPv4 GID index {idx}",
                    f"set env NCCL_IB_GID_INDEX: {idx[0]}" if len(idx) == 1
                    else "make the GID index the same on every HCA")


def _docker_checks(state: NodeState, req: Requirements, add: Callable[..., None], *, image: str | None) -> None:
    sec = state.host.get("docker")
    if sec is None:
        return
    if sec.rc != 0:
        text = sec.text.lower()
        if "permission denied" in text:
            add("docker", FAIL, "the SSH user cannot reach the docker daemon (permission denied)",
                "usermod -aG docker <user> (and log in again)")
        elif "not found" in text:
            add("docker", FAIL, "docker is not installed", "install Docker Engine and nvidia-container-toolkit")
        else:
            add("docker", FAIL, f"docker info failed: {_tail(sec.text)}", "start dockerd")
        return
    info = parse_docker_info(sec.text) or {}
    version = str(info.get("ServerVersion", "?"))
    driver = str(info.get("Driver", "?"))
    snap = docker_snapshotter(info)
    runtimes = sorted((info.get("Runtimes") or {}).keys())
    detail = (f"Docker {version}, storage {driver}" + (" (containerd image store)" if snap else "")
              + f", root {info.get('DockerRootDir', '?')}, runtimes {runtimes}")
    img = image or state.node.image or ""
    bad = any(d in img for d in OVERLAY_UNSAFE_DIGESTS)
    if driver in OVERLAY_DRIVERS and bad:
        add("docker", FAIL, detail + f": the image cannot unpack on {driver}",
            "use the image rebuilt with docker/miles-ports/make_layer.py")
    elif driver in OVERLAY_DRIVERS:
        add("docker", PASS, detail + ": overlay storage refuses images with a same-layer whiteout "
            "('not a directory' on pull); the pull check below confirms the image")
    else:
        add("docker", PASS, detail)


def container_checks(state: NodeState, req: Requirements, *, multi: bool) -> list[Check]:
    node, c = state.node, state.container
    rdma = multi and node.fabric in RDMA_FABRICS
    out: list[Check] = []

    def add(check: str, status: str, detail: str, fix: str = "", **data: Any) -> None:
        out.append(Check(node.name, check, status, detail, fix, data))

    lock = memlock_unlimited(c["memlock"].text) if _ok(c.get("memlock")) else None
    if lock is not None:
        if lock:
            add("memlock_container", PASS, "unlimited")
        else:
            add("memlock_container", FAIL if rdma else WARN,
                f"ulimit -l {c['memlock'].text.strip()} KiB in the container: NCCL cannot register IB memory",
                "docker_args: --ulimit memlock=-1:-1 (--privileged does not lift it)")
    if rdma or node.fabric in RDMA_FABRICS:
        entries = c["devib"].text.split() if _ok(c.get("devib")) else []
        if any(e.startswith("uverbs") for e in entries):
            add("dev_infiniband_container", PASS, f"{sum(e.startswith('uverbs') for e in entries)} uverbs devices")
        else:
            add("dev_infiniband_container", FAIL if rdma else WARN,
                "no /dev/infiniband in the container: NCCL falls back to TCP sockets",
                "docker_args: --device /dev/infiniband")
    if node.device_family == "cuda":
        gpus = c.get("gpus")
        count = len([ln for ln in (gpus.text if gpus else "").splitlines() if ln.startswith("GPU ")])
        if not _ok(gpus) or count != node.cards_per_node:
            add("gpus_container", FAIL, f"{count} GPUs visible in the container: {_tail(gpus.text if gpus else '')[:200]}",
                "docker_args: --gpus all (nvidia-container-toolkit)")
        else:
            add("gpus_container", PASS, f"{count} GPUs")
        torch = c.get("torch")
        if torch is not None:
            m = re.search(r"torch (\S+) cuda (\S+) available (\S+) devices (\d+) nccl (\S+)", torch.text)
            if m and m.group(3) == "True" and int(m.group(4)) == node.cards_per_node:
                add("torch_cuda", PASS, f"torch {m.group(1)}, CUDA {m.group(2)}, NCCL {m.group(5)}, "
                    f"{m.group(4)} devices")
            else:
                add("torch_cuda", FAIL, f"torch cannot use the GPUs: {_tail(torch.text, 240)}",
                    "a driver older than the image's CUDA (driver >= 580 for CUDA 13) or missing --gpus")
    tools = state.tools
    if rdma and not tools.get("ib_write_bw"):
        add("tools", WARN, "no ib_write_bw in the image: the IB bandwidth check is skipped",
            "install perftest in the image")
    return out


# ------------------------------------------------------------- the executor
def container_job(node: Node, *, image: str | None, inner: str, env: Mapping[str, str], tag: str,
                  timeout_s: float, files: Mapping[str, str] | None = None) -> str:
    """Bash for one node: write ``files`` to a temp dir and run ``inner`` there, in the
    trainer image with the node's ``docker_args`` (docker nodes) or on the host (rootfs).
    ``$YETO_PF_DIR`` names the dir inside the job; the job ends after ``timeout_s``."""
    lines = ["set -u", "d=$(mktemp -d \"${TMPDIR:-/tmp}/yeto-preflight.XXXXXX\")"]
    name = re.sub(r"[^A-Za-z0-9_.-]", "-", f"yeto-preflight-{tag}-{node.name}")
    if node.runtime == "docker":
        lines.append(f"trap 'docker rm -f {name} >/dev/null 2>&1; rm -rf \"$d\"' EXIT")
    else:
        lines.append("trap 'rm -rf \"$d\"' EXIT")
    for fname, content in (files or {}).items():
        if _DELIM in content:
            raise ValueError(f"{fname} contains the heredoc delimiter")
        lines.append(f"cat > \"$d/{fname}\" <<'{_DELIM}'\n{content.rstrip()}\n{_DELIM}")
    t = int(timeout_s)
    if node.runtime == "docker":
        # --pull never: the pull check fetched the image; a missing one fails here at once
        argv = ["docker", "run", "--rm", "--pull", "never", "--name", name, "--network", "host", "--ipc", "host",
                *node.docker_args]
        for k, v in sorted({**env, "YETO_PF_DIR": CONTAINER_DIR}.items()):
            argv += ["-e", f"{k}={v}"]
        lines.append(f"docker rm -f {name} >/dev/null 2>&1 || true")
        lines.append(f"{shlex.join(argv)} -v \"$d:{CONTAINER_DIR}\" -w {CONTAINER_DIR} "
                     f"{shlex.quote(str(image or node.image))} timeout -k 10 {t} bash -c {shlex.quote(inner)}")
    else:
        lines.append("export YETO_PF_DIR=\"$d\"")
        lines += [f"export {k}={shlex.quote(str(v))}" for k, v in sorted(env.items())]
        lines.append(f"cd \"$d\" && timeout -k 10 {t} bash -c {shlex.quote(inner)}")
    return "\n".join(lines) + "\n"


def bench_source() -> str:
    """The benchmark script (:mod:`.bench`), shipped to the nodes as ``BENCH_FILE``."""
    return Path(__file__).with_name("bench.py").read_text()


PY = "PY=$(command -v python3 || command -v python); "


@dataclass
class Report:
    checks: list[Check]
    out_dir: Path | None = None

    @property
    def counts(self) -> dict[str, int]:
        out = {PASS: 0, WARN: 0, FAIL: 0, SKIP: 0}
        for c in self.checks:
            out[c.status] += 1
        return out

    @property
    def ok(self) -> bool:
        return not any(c.status == FAIL for c in self.checks)

    def table(self) -> str:
        rows = [("SCOPE", "CHECK", "STATUS", "DETAIL")] + [(c.scope, c.name, c.status, c.detail)
                                                          for c in self.checks]
        w = [max(len(r[i]) for r in rows) for i in range(3)]
        lines = [f"{r[0]:<{w[0]}}  {r[1]:<{w[1]}}  {r[2]:<{w[2]}}  {r[3]}" for r in rows]
        lines.insert(1, "-" * min(160, sum(w) + 6 + 40))
        fixes = [c for c in sorted(self.checks, key=lambda c: _ORDER[c.status]) if c.status in (FAIL, WARN) and c.fix]
        if fixes:
            lines += ["", "FIXES"]
            lines += [f"[{c.status}] {c.scope} {c.name}: {c.fix}" for c in fixes]
        n = self.counts
        verdict = FAIL if not self.ok else (WARN if n[WARN] else PASS)
        lines += ["", f"RESULT: {verdict} ({n[PASS]} pass, {n[WARN]} warn, {n[FAIL]} fail, {n[SKIP]} skip)"]
        return "\n".join(lines) + "\n"

    def to_json(self) -> dict[str, Any]:
        return {"ok": self.ok, "counts": self.counts, "checks": [asdict(c) for c in self.checks]}


@dataclass(frozen=True)
class Steps:
    pull: bool = True
    container: bool = True
    ports: bool = True
    ib: bool = True
    nccl: bool = True
    ib_pairs: str = "all"  # all | ring
    ib_seconds: int = 3
    nccl_sizes: str = "16M,128M,1G"
    nccl_iters: int = 20
    nccl_all: bool = True  # also run the collectives on every node at once
    master_port: int = 29611
    ib_port: int = 18515


class ClusterPreflight:
    """Runs every check over ``ops`` (an :class:`SshOps`; tests pass a fake)."""

    def __init__(self, nodes: Sequence[Node], *, image: str | None = None, ops: SshOps | None = None,
                 req: Requirements | None = None, steps: Steps | None = None, out_dir: Path | None = None,
                 log: Callable[[str], None] = lambda m: print(m, file=sys.stderr, flush=True),
                 clock: Callable[[], float] = time.time) -> None:
        self.nodes = list(nodes)
        self.image = image
        self.ops = ops or SshOps()
        self.req = req or Requirements()
        self.steps = steps or Steps()
        self.out_dir = Path(out_dir) if out_dir else None
        self.log = log
        self.clock = clock
        self.multi = len(self.nodes) > 1
        self.states = [NodeState(n) for n in self.nodes]

    # -- helpers
    def _raw(self, step: str, name: str, text: str) -> None:
        if self.out_dir is None:
            return
        path = self.out_dir / "raw" / step / f"{name}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text or "")

    def _run(self, node: Node, script: str, timeout: float, step: str) -> tuple[int, str]:
        try:
            done = self.ops.run(node, script, timeout=timeout)
            text = (done.stdout or "") + (("\n" + done.stderr) if getattr(done, "stderr", "") else "")
            rc = done.returncode
        except Exception as exc:  # noqa: BLE001 - a timeout or a dead ssh is a finding, not a crash
            rc, text = 255, f"{type(exc).__name__}: {exc}"
        self._raw(step, node.name, text)
        return rc, text

    def _parallel(self, fn: Callable[[Any], Any], items: Sequence[Any]) -> list[Any]:
        if not items:
            return []
        with ThreadPoolExecutor(max_workers=max(1, min(32, len(items)))) as pool:
            return list(pool.map(fn, items))

    def _env(self, node: Node, n: int, **extra: str) -> dict[str, str]:
        return {**island_comm_env(node.device_family, node.fabric, node.interface, num_nodes=n), **node.env,
                **extra}

    # -- steps
    def run(self) -> Report:
        checks = inventory_checks(self.nodes, self.image)
        checks += self._reach()
        live = [s for s in self.states if s.reachable]
        checks += self._host(live)
        if self.steps.pull and any(s.node.runtime == "docker" for s in live):
            checks += self._pull(live)
        if self.steps.container:
            checks += self._container([s for s in live if s.pulled])
        for state in live:
            checks += ib_checks(state, self.req, multi=self.multi)
        checks += self._cluster(live)
        ready = [s for s in live if (s.container_ok or (s.node.runtime != "docker" and self.steps.container))
                 and _dialable(s.node)]  # a node without a rendezvous address failed the inventory check
        if self.multi and self.steps.container:
            if self.steps.ports:
                checks += self._ports(ready)
            if self.steps.ib and rdma_island(self.nodes):
                checks += self._ib(ready)
            if self.steps.nccl and self.nodes[0].device_family == "cuda":
                checks += self._nccl(ready)
        report = Report(checks, self.out_dir)
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / "preflight.json").write_text(json.dumps(report.to_json(), indent=2) + "\n")
            (self.out_dir / "preflight.txt").write_text(report.table())
        return report

    def _reach(self) -> list[Check]:
        self.log(f"[preflight] ssh + clock on {len(self.states)} node(s)")

        def one(state: NodeState) -> Check:
            t0 = self.clock()
            rc, text = self._run(state.node, "date +%s.%N", 60.0, "ssh")
            t1 = self.clock()
            if rc != 0:
                return Check(state.name, "ssh", FAIL, f"ssh {state.node.host} failed (rc {rc}): {_tail(text, 200)}",
                             "key login with BatchMode (ssh-copy-id), known_hosts written, proxy_jump if needed")
            state.reachable = True
            try:
                remote = float(text.split()[0])
            except (ValueError, IndexError):
                return Check(state.name, "ssh", PASS, f"reachable ({t1 - t0:.2f} s)")
            # the remote clock was read between t0 and t1: the skew is known to +-(t1 - t0) / 2
            skew, slack = remote - (t0 + t1) / 2, (t1 - t0) / 2
            beyond = abs(skew) - slack
            if beyond > self.req.fail_clock_skew_s:
                return Check(state.name, "ssh", FAIL, f"reachable; clock {skew:+.1f} s off the head",
                             "sync the clock (chrony); object-store request signatures and timeouts break",
                             {"skew_s": skew})
            if beyond > self.req.warn_clock_skew_s:
                return Check(state.name, "ssh", WARN, f"reachable; clock {skew:+.2f} s off the head (+-{slack:.2f})",
                             "sync the clock (chrony)", {"skew_s": skew})
            return Check(state.name, "ssh", PASS, f"reachable ({t1 - t0:.2f} s), clock {skew:+.2f} s +-{slack:.2f}",
                         data={"skew_s": skew})

        return self._parallel(one, self.states)

    def _host(self, live: Sequence[NodeState]) -> list[Check]:
        self.log(f"[preflight] host probes on {len(live)} node(s)")

        def one(state: NodeState) -> list[Check]:
            _, text = self._run(state.node, probe_script(host_sections(state.node)), 300.0, "host")
            state.host = parse_sections(text)
            if not state.host:
                return [Check(state.name, "host_probe", FAIL, f"the host probe printed nothing: {_tail(text, 200)}",
                              "check that bash runs on the node")]
            state.ports = ports_of(state.host)
            state.compute = compute_ports(state.node, state.ports)[0]
            state.gids = parse_gids(state.host["gids"].text) if state.host.get("gids") else []
            return host_checks(state, self.req, multi=self.multi, image=self.image)

        return [c for cs in self._parallel(one, live) for c in cs]

    def _pull(self, live: Sequence[NodeState]) -> list[Check]:
        docker = [s for s in live if s.node.runtime == "docker"]
        self.log(f"[preflight] docker pull on {len(docker)} node(s)")

        def one(state: NodeState) -> Check:
            image = str(self.image or state.node.image)
            t0 = self.clock()
            rc, text = self._run(state.node, f"docker pull -q {shlex.quote(image)}", 3600.0, "pull")
            if rc == 0:
                return Check(state.name, "image_pull", PASS, f"{image[:90]} ({self.clock() - t0:.0f} s)")
            state.pulled = False
            fix = "check registry access (ghcr.io), credentials and disk space"
            if "not a directory" in text:
                fix = ("the image has a layer that whites out a path and writes it again: rebuild it with "
                       "docker/miles-ports/make_layer.py")
            elif "no space left" in text.lower():
                fix = "free the docker data root / /var/lib/containerd (put them on NVMe)"
            return Check(state.name, "image_pull", FAIL, f"docker pull failed (rc {rc}): {_tail(text, 240)}", fix)

        return self._parallel(one, docker)

    def _container(self, live: Sequence[NodeState]) -> list[Check]:
        docker = [s for s in live if s.node.runtime == "docker"]
        self.log(f"[preflight] container probes on {len(docker)} node(s)")

        def one(state: NodeState) -> list[Check]:
            job = container_job(state.node, image=self.image, inner=probe_script(CONTAINER_SECTIONS),
                                env=self._env(state.node, len(self.nodes)), tag="probe", timeout_s=240)
            rc, text = self._run(state.node, job, 600.0, "container")
            state.container = parse_sections(text)
            if not state.container:
                fix = "check docker_args, the image and nvidia-container-toolkit"
                if "No such image" in text or "pull access denied" in text:
                    fix = "the image is not on the node: run without --no-pull"
                return [Check(state.name, "container", FAIL, f"the trainer image did not start (rc {rc}): "
                              f"{_tail(text, 300)}", fix)]
            state.container_ok = True
            if _ok(state.container.get("tools")):
                for line in state.container["tools"].text.splitlines():
                    k, _, v = line.partition("=")
                    state.tools[k.strip()] = v.strip()
            if not state.ports:  # no ibstat / ibv_devinfo on the host: the image has them
                state.ports = ports_of(state.container)
                state.ports_from_container = bool(state.ports)
                state.compute = compute_ports(state.node, state.ports)[0]
            return container_checks(state, self.req, multi=self.multi)

        return [c for cs in self._parallel(one, docker) for c in cs]

    def _cluster(self, live: Sequence[NodeState]) -> list[Check]:
        out: list[Check] = []
        drivers = {}
        for s in live:
            v = driver_versions(s.host["driver"].text) if s.host.get("driver") else []
            if v:
                drivers[s.name] = v[0]
        if len(set(drivers.values())) > 1:
            out.append(Check("cluster", "driver", WARN, f"driver versions differ: {drivers}",
                             "install one driver version on every node"))
        fws = {s.name: sorted({p.firmware for p in s.compute if p.firmware}) for s in live if s.compute}
        if len({tuple(v) for v in fws.values()}) > 1:
            out.append(Check("cluster", "hca_firmware", WARN, f"HCA firmware differs: {fws}",
                             "flash one firmware version (mlxfwmanager)"))
        return out

    def _ports(self, ready: Sequence[NodeState]) -> list[Check]:
        if len(ready) < 2:
            return [Check("cluster", "tcp_ports", SKIP, "fewer than two nodes can run the container")]
        ports = ",".join(str(p) for p in self.req.ports)
        addresses = {s.name: rendezvous_address(s.node) for s in ready}
        self.log(f"[preflight] TCP ports {ports} between {len(ready)} node(s)")
        files = {BENCH_FILE: bench_source()}

        def one(state: NodeState) -> tuple[NodeState, int, str]:
            peers = ",".join(a for n, a in addresses.items() if n != state.name)
            inner = (PY + f"$PY \"$YETO_PF_DIR/{BENCH_FILE}\" ports --ports {ports} --peers {peers} "
                     f"--me {shlex.quote(state.name)} --wait 10 --hold 30 --dial-timeout 3")
            job = container_job(state.node, image=self.image, inner=inner, env={}, tag="ports", timeout_s=120,
                                files=files)
            rc, text = self._run(state.node, job, 300.0, "ports")
            return state, rc, text

        results = self._parallel(one, ready)
        by_addr = {a: n for n, a in addresses.items()}
        out: list[Check] = []
        bad: list[str] = []
        listen: dict[str, dict[str, str]] = {}
        for state, rc, text in results:
            recs = [r for r in bench_records(text) if r.get("mode") == "ports"]
            if not recs:
                out.append(Check(state.name, "tcp_ports", FAIL, f"the port probe did not run (rc {rc}): "
                                 f"{_tail(text, 200)}", "check that python3 runs in the image"))
                continue
            rec = recs[-1]
            listen[state.name] = rec.get("listen", {})
            busy = [p for p, v in rec.get("listen", {}).items() if str(v).startswith("busy")]
            if busy:
                out.append(Check(state.name, "tcp_ports", WARN, f"ports {busy} already in use on the node",
                                 "stop leftover ray / trainers (ray stop; docker ps) before the launch"))
            for addr, dials in (rec.get("dial") or {}).items():
                for port, result in dials.items():
                    if result != "ok":
                        bad.append(f"{state.name}->{by_addr.get(addr, addr)}:{port} {result}")
        if bad:
            out.append(Check("cluster", "tcp_ports", FAIL, f"{len(bad)} blocked: {', '.join(bad[:8])}",
                             "open all TCP ports between the nodes on the cluster network (Ray, NCCL bootstrap, "
                             "torch rendezvous use random ports)", data={"blocked": bad}))
        elif len(listen) == len(ready):
            out.append(Check("cluster", "tcp_ports", PASS,
                             f"ports {ports} open between all {len(ready)} nodes"))
        return out

    def _ib(self, ready: Sequence[NodeState]) -> list[Check]:
        usable = [s for s in ready if s.compute and any(p.active for p in s.compute)
                  and (s.node.runtime != "docker" or s.tools.get("ib_write_bw"))]
        if len(usable) < 2:
            return [Check("cluster", "ib_write_bw", SKIP, "fewer than two nodes with active HCAs and ib_write_bw")]
        rounds = round_robin(len(usable)) if self.steps.ib_pairs == "all" else (
            [[(i, (i + 1) % len(usable))] for i in range(len(usable))] if len(usable) > 2 else [[(0, 1)]])
        self.log(f"[preflight] ib_write_bw: {sum(len(r) for r in rounds)} pair(s) in {len(rounds)} round(s)")
        out: list[Check] = []
        for pairs in rounds:
            out += [c for cs in self._parallel(lambda p: self._ib_pair(usable[p[0]], usable[p[1]]), pairs)
                    for c in cs]
        return out

    def _ib_cmd(self, state: NodeState, rails: Sequence[IbPort], server: str | None) -> str:
        parts = []
        for i, p in enumerate(rails):
            gid = ""
            if state.node.fabric == "roce":
                idx = state.node.env.get("NCCL_IB_GID_INDEX")
                idx = idx if idx is not None else roce_v2_index(state.gids, p.ca)
                gid = f" -x {idx}" if idx is not None else ""
            cmd = (f"ib_write_bw -d {p.ca} -i {p.port} -p {self.steps.ib_port + i} -D {self.steps.ib_seconds} "
                   f"-s 1048576 -q 4 -F --report_gbits{gid}")
            if server:  # the client retries while the server's container starts
                parts.append(f"( for _ in $(seq 15); do {cmd} {server} > /tmp/yeto-ib-{i}.log 2>&1 && break; "
                             f"sleep 2; done ) &")
            else:
                parts.append(f"{cmd} > /tmp/yeto-ib-{i}.log 2>&1 &")
        body = ("sleep 3\n" if server else "") + "\n".join(parts) + "\nwait\n"
        body += "".join(f"echo '{IB_MARK} {i} {p.ca}'; cat /tmp/yeto-ib-{i}.log\n" for i, p in enumerate(rails))
        return body

    def _ib_pair(self, a: NodeState, b: NodeState) -> list[Check]:
        rails = min(len([p for p in a.compute if p.active]), len([p for p in b.compute if p.active]))
        ra = [p for p in a.compute if p.active][:rails]
        rb = [p for p in b.compute if p.active][:rails]
        scope = f"{a.name}<->{b.name}"
        t = self.steps.ib_seconds + 60
        jobs = [(a, container_job(a.node, image=self.image, inner=self._ib_cmd(a, ra, None), env={},
                                  tag=f"ibs-{b.name}", timeout_s=t)),
                (b, container_job(b.node, image=self.image, inner=self._ib_cmd(b, rb, rendezvous_address(a.node)),
                                  env={}, tag=f"ibc-{a.name}", timeout_s=t))]
        results = self._parallel(lambda j: self._run(j[0].node, j[1], t + 120, f"ib/{scope}"), jobs)
        client_text = results[1][1]
        blocks = ib_blocks(client_text)
        values, errors = [], []
        for i in range(rails):
            gbps, err = parse_ib_write_bw(blocks.get(i, ""))
            if gbps is None:
                errors.append(f"{ra[i].ca}->{rb[i].ca}: {err or 'no result'}")
            values.append(gbps)
        link = [min(x.rate_gbps or 0, y.rate_gbps or 0) for x, y in zip(ra, rb)]
        got = [v for v in values if v is not None]
        detail_rails = ", ".join(f"{ra[i].ca}:{(values[i] or 0) / 8:.1f}" for i in range(rails))
        total = sum(got) / 8
        data = {"rails_gbps": values, "link_gbps": link, "total_GBps": total}
        if errors:
            return [Check(scope, "ib_write_bw", FAIL, f"{len(errors)}/{rails} rails failed: {errors[0]}",
                          "check the HCA ports, GID index (RoCE), the subnet / VLAN and that the peer is reachable",
                          data)]
        fr = [v / lk for v, lk in zip(values, link) if lk]
        worst = min(fr) if fr else None
        detail = f"{rails} rails, {total:.1f} GB/s total; GB/s per rail: {detail_rails}"
        if worst is None:
            return [Check(scope, "ib_write_bw", PASS, detail + " (link rate unknown)", data=data)]
        status = PASS if worst >= self.req.ib_pass_fraction else (WARN if worst >= self.req.ib_fail_fraction else FAIL)
        return [Check(scope, "ib_write_bw", status, detail + f"; worst rail {worst:.0%} of link",
                      "" if status == PASS else "check that rail: cable, switch port, PCIe width (lspci -vv), "
                      "congestion control / PFC (RoCE)", data)]

    def _nccl(self, ready: Sequence[NodeState]) -> list[Check]:
        if len(ready) < 2:
            return [Check("cluster", "nccl", SKIP, "fewer than two nodes can run the container")]
        groups: list[list[list[NodeState]]] = [[[ready[i], ready[j]] for i, j in r] for r in disjoint_pairs(len(ready))]
        if self.steps.nccl_all and len(ready) > 2:
            groups.append([list(ready)])
        files = {BENCH_FILE: bench_source()}
        out: list[Check] = []
        for k, round_groups in enumerate(groups):
            self.log(f"[preflight] NCCL collectives: {', '.join('+'.join(s.name for s in g) for g in round_groups)}")
            def one(gi: tuple[int, list[NodeState]], k: int = k) -> list[Check]:
                return self._nccl_group(gi[1], files, port_off=gi[0] + k * 10)

            out += [c for cs in self._parallel(one, list(enumerate(round_groups))) for c in cs]
        return out

    def _nccl_group(self, group: Sequence[NodeState], files: Mapping[str, str], *, port_off: int) -> list[Check]:
        n = len(group)
        master = rendezvous_address(group[0].node)
        port = self.steps.master_port + port_off
        scope = "+".join(s.name for s in group) if n <= 2 else f"all {n} nodes"
        timeout = 900

        def job(rank: int) -> tuple[int, str]:
            s = group[rank]
            inner = (PY + f"$PY -m torch.distributed.run --nnodes={n} --node-rank={rank} "
                     f"--nproc-per-node={s.node.cards_per_node} --master-addr={master} --master-port={port} "
                     f"\"$YETO_PF_DIR/{BENCH_FILE}\" collectives --sizes {self.steps.nccl_sizes} "
                     f"--iters {self.steps.nccl_iters}")
            env = self._env(s.node, n, NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,NET")
            script = container_job(s.node, image=self.image, inner=inner, env=env, tag=f"nccl{port}", timeout_s=timeout,
                                   files=files)
            return self._run(s.node, script, timeout + 120, f"nccl/{scope.replace(' ', '-')}")

        results = self._parallel(job, list(range(n)))
        text = "\n".join(t for _, t in results)
        recs = bench_records(results[0][1])
        transport = parse_nccl_log(text)
        out: list[Check] = []
        done = [r for r in recs if r.get("event") == "done"]
        res = [r for r in recs if r.get("event") == "result"]
        rails = [s.min_rail_gbps for s in group if s.min_rail_gbps]
        aggs = [s.hca_gbps for s in group if s.hca_gbps]
        agg_GBps = min(aggs) / 8 if aggs else None
        rail_GBps = min(rails) / 8 if rails else None
        local = group[0].node.cards_per_node
        world = local * n
        # all_to_all: a rank sends (world - local) / world of its bytes off the node, so the
        # HCAs bound busbw at agg x (world - 1) / (local x (world - local))
        a2a_GBps = (agg_GBps * (world - 1) / (local * (world - local))
                    if agg_GBps and world > local else None)
        rdma = any(s.node.fabric in RDMA_FABRICS for s in group)
        if not done or any(rc != 0 for rc, _ in results):
            bad = [group[i].name for i, (rc, _) in enumerate(results) if rc != 0]
            out.append(Check(scope, "nccl", FAIL, f"the collectives did not finish (nodes {bad or '?'}): "
                             f"{'; '.join(transport.warnings[:2]) or _tail(text, 240)}",
                             "read raw/nccl logs; check the tcp_ports and ib checks first"))
        # transport
        if transport.socket or (rdma and transport.no_ib_device) or (rdma and not transport.ib and res):
            out.append(Check(scope, "nccl_transport", FAIL if rdma else WARN,
                             "NCCL runs over NET/Socket (TCP), not NET/IB",
                             "docker_args --device /dev/infiniband --ulimit memlock=-1:-1; NCCL_IB_HCA matching the "
                             "HCAs; NCCL_IB_GID_INDEX on RoCE; NCCL_IB_DISABLE unset"))
        elif transport.ib:
            if transport.gdr:
                out.append(Check(scope, "nccl_transport", PASS, "NET/IB with GPUDirect RDMA"
                                 + (f" ({', '.join(transport.devices[:8])})" if transport.devices else "")))
            else:
                out.append(Check(scope, "nccl_transport", WARN, "NET/IB without GPUDirect RDMA (GDRDMA): host bounce",
                                 "load nvidia_peermem or use DMA-BUF (kernel >= 5.12, open driver); check "
                                 "gpu_nic_affinity"))
        for op, frac, base, ref in (("all_reduce", self.req.all_reduce_fraction, agg_GBps, "per-node HCA aggregate"),
                                    ("all_to_all", self.req.all_to_all_fraction, a2a_GBps, "HCA bound"),
                                    ("sendrecv", self.req.sendrecv_fraction, rail_GBps, "one HCA")):
            rows = sorted((r for r in res if r.get("op") == op), key=lambda r: r.get("bytes", 0))
            if not rows:
                continue
            top = rows[-1]
            sizes = ", ".join(f"{_size(r['bytes'])}:{r['busbw_gbps']:.1f}" for r in rows)
            detail = f"busbw GB/s {sizes}"
            if not all(r.get("correct", True) for r in rows):
                out.append(Check(scope, f"nccl_{op}", FAIL, detail + ": wrong results",
                                 "a broken GPU / NIC / driver: rerun per pair to isolate the node"))
                continue
            if base is None or not rdma:
                out.append(Check(scope, f"nccl_{op}", PASS, detail, data={"rows": rows}))
                continue
            ratio = top["busbw_gbps"] / base if base else 0
            status = PASS if ratio >= frac else WARN
            out.append(Check(scope, f"nccl_{op}", status, detail + f" ({ratio:.0%} of {base:.0f} GB/s {ref})",
                             "" if status == PASS else "compare the ib_write_bw pairs; check NCCL_IB_HCA covers "
                             "every rail and the GPU-NIC affinity", data={"rows": rows}))
        init = [r for r in recs if r.get("event") == "init"]
        if init:
            out.append(Check(scope, "nccl_init", PASS, f"NCCL {init[0].get('nccl')}, {init[0].get('world')} ranks, "
                             f"init {init[0].get('init_s')} s"))
        return out


def _size(nbytes: int) -> str:
    for unit, scale in (("G", 1 << 30), ("M", 1 << 20), ("K", 1 << 10)):
        if nbytes >= scale:
            return f"{nbytes / scale:g}{unit}"
    return str(nbytes)


# --------------------------------------------------------------------- CLI
def resolve_nodes(gpu: str, inventory: str | None) -> list[Node]:
    """The trainer nodes of ``--gpu ssh:<NxMxCARD>`` with ``--rl-node-inventory``, or of
    ``--gpu ssh:<inventory file>``."""
    if not gpu.startswith("ssh:"):
        raise InventoryError(f"--gpu {gpu!r}: preflight runs on ssh: nodes (ssh:NxMxCARD or ssh:<inventory>)")
    rest = gpu[4:]
    path = Path(rest).expanduser()
    if rest.endswith((".yaml", ".yml", ".json")) or path.is_file():
        if inventory and Path(inventory).expanduser() != path:
            raise InventoryError("give the inventory once: --gpu ssh:<inventory> or --rl-node-inventory")
        return island_nodes(load_inventory(path), "trainer")
    if not inventory:
        raise InventoryError(f"--gpu {gpu} needs --rl-node-inventory (the nodes to check)")
    from ...gpu_spec import parse_gpu_spec

    (island,) = parse_gpu_spec(gpu)
    args = argparse.Namespace(rl_node_inventory_nodes=load_inventory(inventory))
    try:
        return list(inventory_trainer_nodes(args, island))
    except SshIslandError as exc:
        raise InventoryError(str(exc)) from None


def add_parser(sub: Any) -> argparse.ArgumentParser:
    p = sub.add_parser(
        "preflight", help="check SSH bare-metal trainer nodes before a launch (drivers, GPUs, IB, NCCL)",
        description=(__doc__ or "").split("\n\n")[0] + " Exit 1 when a check fails.")
    p.add_argument("--gpu", required=True, help="ssh:NxMxCARD (with --rl-node-inventory) or ssh:<inventory file>")
    p.add_argument("--rl-node-inventory", default=None, help="the SSH node inventory (YAML / JSON)")
    p.add_argument("--image", default=None, help="trainer image (default: the inventory's)")
    p.add_argument("--cluster-prefix", default="baremetal",
                   help="run name: the report goes to <runs dir>/<name>/preflight/<time>")
    p.add_argument("--run-dir", default=None, help="write the report here instead")
    p.add_argument("--no-pull", action="store_true", help="do not docker pull the image")
    p.add_argument("--no-container", action="store_true", help="host checks only (no container, no benchmarks)")
    p.add_argument("--no-bench", action="store_true",
                   help="skip the bandwidth benchmarks (ib_write_bw, NCCL); the TCP port check still runs "
                   "(what a launch of an ssh: trainer island runs first)")
    p.add_argument("--no-ports", action="store_true", help="skip the cross-node TCP port check")
    p.add_argument("--no-nccl", action="store_true", help="skip the NCCL collectives")
    p.add_argument("--no-ib", action="store_true", help="skip ib_write_bw")
    p.add_argument("--ib-pairs", choices=("all", "ring"), default="all", help="ib_write_bw: every pair or a ring")
    p.add_argument("--nccl-sizes", default="16M,128M,1G", help="message sizes of the collectives")
    p.add_argument("--nccl-iters", type=int, default=20)
    p.add_argument("--nccl-pairs-only", action="store_true", help="skip the all-node collectives run")
    p.add_argument("--min-driver", type=int, default=Requirements.min_driver_major)
    p.add_argument("--min-host-memory-gib", type=float, default=None,
                   help=f"default {Requirements.min_host_memory_gib:g}, or ${REQ_ENV['min_host_memory_gib']}")
    p.add_argument("--min-disk-free-gib", type=float, default=None,
                   help=f"default {Requirements.min_disk_free_gib:g}, or ${REQ_ENV['min_disk_free_gib']}")
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    return p


def trainer_image(image: str | None, nodes: Sequence[Node]) -> str | None:
    """The image the launcher would run on these nodes (``ssh_trainer_island.run_image``):
    ``--image`` else the inventory's, a pinned image that overlay appliers refuse mapped
    to its overlay-safe repack, so the pull and the probes check what the trainer runs."""
    from types import SimpleNamespace

    from ..ssh_trainer_island import run_image

    return run_image(SimpleNamespace(rl_image=image), nodes) or image


def main(args: argparse.Namespace, *, ops: SshOps | None = None) -> int:
    try:
        nodes = resolve_nodes(args.gpu, args.rl_node_inventory)
        image = trainer_image(args.image, nodes)
    except (InventoryError, SshIslandError, ValueError) as exc:
        print(f"[yeto] preflight: {exc}", file=sys.stderr)
        return 2
    if args.run_dir:
        out_dir = Path(args.run_dir).expanduser()
    else:
        from ... import runs

        out_dir = runs.run_dir(args.cluster_prefix) / "preflight" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    bench = not (args.no_bench or args.no_container)
    steps = Steps(pull=not args.no_pull, container=not args.no_container,
                  ports=not (getattr(args, "no_ports", False) or args.no_container), ib=bench and not args.no_ib,
                  nccl=bench and not args.no_nccl, ib_pairs=args.ib_pairs, nccl_sizes=args.nccl_sizes,
                  nccl_iters=args.nccl_iters, nccl_all=not args.nccl_pairs_only)
    try:
        env_req = requirements_from_env()
    except ValueError as exc:
        print(f"[yeto] preflight: {exc}", file=sys.stderr)
        return 2
    memory = args.min_host_memory_gib
    disk = args.min_disk_free_gib
    req = Requirements(min_driver_major=args.min_driver,
                       min_host_memory_gib=env_req.min_host_memory_gib if memory is None else memory,
                       min_disk_free_gib=env_req.min_disk_free_gib if disk is None else disk)
    pf = ClusterPreflight(nodes, image=image, ops=ops, req=req, steps=steps, out_dir=out_dir)
    report = pf.run()
    print(json.dumps(report.to_json(), indent=2) if args.json else report.table(), end="" if not args.json else "\n")
    print(f"[yeto] preflight report: {out_dir}", file=sys.stderr)
    return 0 if report.ok else 1
