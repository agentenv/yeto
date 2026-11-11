"""Per-card telemetry collectors for ``rl_resource_sample`` (
rl-hetero-sampler-engines "Telemetry by device family").

One collector per device family, one card-sample shape:

* ``cuda``: NVML (``pynvml``), unchanged from fleet-dashboard 2.2;
* ``npu``: ``npu-smi info`` (Ascend);
* ``ppu``: ``ppu-smi --query-gpu=... --format=csv,noheader,nounits``;
* ``rocm``: ``amd-smi metric --json`` (ROCm 6.x+), else ``rocm-smi --json``
  (AMD MI300 / MI355X), else ``rocm-smi --showmeminfo vram --showuse --json``
  (fewer flags: older / trimmed rocm-smi builds), else the amdgpu sysfs files
  ``/sys/class/drm/card*/device/mem_info_vram_{used,total}`` and
  ``gpu_busy_percent`` (images that ship neither amd-smi nor rocm-smi);
* ``cuda`` falls back to the ``rocm`` chain when NVML cannot open (an AMD node
  whose launcher left the device family unset).

A card sample has the keys of :data:`CARD_KEYS`; a value the tool does not
print is None. NVML samples keep their historical six keys
(``telemetry.GPU_SAMPLE_KEYS``); ``npu-smi`` and ``ppu-smi`` samples add
``temp_c`` and ``health``.

Verification state: the ``npu-smi info`` table shape is the one already
recorded in ``yeto/gpu_spec.py`` and ``card_tools`` (910B4); the power and
temperature columns are read from the same rows and are NOT checked against a
driver 25.x machine. ``ppu-smi`` is assumed to accept the nvidia-smi query
form (T-PPU-gaps.md 2.2 says the stack is NVML compatible); NOT verified on a
PPU. The ``amd-smi`` / ``rocm-smi`` JSON shapes are the documented ones of
ROCm 6.2-7.0 and are NOT verified on an MI355X by this code. Every parser
raises :class:`CardTelemetryError` on output it cannot read, so a format
change shows up as the island's single ``available: false`` record with the
reason, never as silent zeros.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Sequence
from typing import Any, Protocol

CARD_KEYS = ("index", "uuid", "util_pct", "mem_used_mb", "mem_total_mb", "power_w", "temp_c",
             "health")

Run = Callable[[Sequence[str]], str]  # argv -> stdout; raises on a failed command


class CardTelemetryError(RuntimeError):
    """The card tool is missing or its output cannot be read."""


class CardCollector(Protocol):
    """``source``: the tool name in the record; ``open()`` raises when the tool is
    not usable; ``sample()`` returns one dict per card."""

    source: str

    def open(self) -> None: ...

    def sample(self) -> list[dict[str, Any]]: ...


def run_command(argv: Sequence[str], timeout_s: float = 20.0) -> str:
    try:
        result = subprocess.run(list(argv), capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.SubprocessError) as exc:
        raise CardTelemetryError(f"{argv[0]}: {type(exc).__name__}: {exc}") from None
    if result.returncode != 0:
        raise CardTelemetryError(f"{' '.join(argv)} exited {result.returncode}: "
                                 f"{result.stderr.strip()[:200]}")
    return result.stdout


def _num(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


# --- NVML ---------------------------------------------------------------------

def _safe(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception:  # noqa: BLE001 - one unreadable field is None, not a failed sample
        return None


class NvmlCollector:
    """NVML through ``pynvml``; ``loader`` returns the initialised module."""

    source = "nvml"

    def __init__(self, loader: Callable[[], Any]) -> None:
        self._loader = loader
        self._nvml: Any = None

    def open(self) -> None:
        self._nvml = self._loader()

    def sample(self) -> list[dict[str, Any]]:
        nv = self._nvml
        cards = []
        for index in range(int(nv.nvmlDeviceGetCount())):
            handle = nv.nvmlDeviceGetHandleByIndex(index)
            mem = _safe(lambda: nv.nvmlDeviceGetMemoryInfo(handle))
            util = _safe(lambda: nv.nvmlDeviceGetUtilizationRates(handle))
            power = _safe(lambda: nv.nvmlDeviceGetPowerUsage(handle))
            uuid = _safe(lambda: nv.nvmlDeviceGetUUID(handle))
            cards.append({
                "index": index,
                "uuid": uuid.decode() if isinstance(uuid, bytes) else uuid,
                "util_pct": None if util is None else int(util.gpu),
                "mem_used_mb": None if mem is None else int(mem.used) / 2**20,
                "mem_total_mb": None if mem is None else int(mem.total) / 2**20,
                "power_w": None if power is None else float(power) / 1000.0,
            })
        return cards


# --- npu-smi ------------------------------------------------------------------
#
# ``npu-smi info`` prints two rows per chip (card_tools has the same sample):
#
#   | NPU   Name       | Health | Power(W)    Temp(C)     Hugepages-Usage(page) |
#   | Chip             | Bus-Id | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)   |
#   | 0     910B4      | OK     | 92.5        47          0    / 0              |
#   | 0                | 0000:C1:00.0 | 0     0    / 0    3161 / 65536         |
#
# Name row: NPU id, card, health, then power and temperature. Usage row: chip
# id, bus id, then AICore %, memory used / total, HBM used / total (the last
# pair). A card with two chips (910C) has two usage rows; each chip is one sample.

_CELL_SPLIT = re.compile(r"\s*\|\s*")
_NUMBERS = re.compile(r"-?\d+(?:\.\d+)?")


def _cells(line: str) -> list[str]:
    return [c for c in _CELL_SPLIT.split(line.strip().strip("|")) if c != ""] if line.strip().startswith("|") else []


def parse_npu_smi_usage(text: str) -> list[dict[str, Any]]:
    """Card samples from ``npu-smi info``; raises when no chip row is found."""
    samples: list[dict[str, Any]] = []
    card: dict[str, Any] | None = None
    chips = 0
    for line in text.splitlines():
        cells = _cells(line)
        if len(cells) < 3:
            continue
        head = cells[0].split()
        if not head or not head[0].isdigit():
            continue  # header rows ("NPU Name", "Chip") and process tables
        if len(head) >= 2 and re.fullmatch(r"[A-Za-z]+", cells[1]):  # name row
            numbers = [_num(n) for n in _NUMBERS.findall(cells[2])]
            card = {"index": int(head[0]), "card": head[1], "health": cells[1],
                    "power_w": numbers[0] if numbers else None,
                    "temp_c": numbers[1] if len(numbers) > 1 else None}
            chips = 0
            continue
        if card is None or len(head) != 1 or ":" not in cells[1]:  # usage row has a bus id
            continue
        numbers = [_num(n) for n in _NUMBERS.findall(cells[2])]
        if len(numbers) < 3:
            raise CardTelemetryError(f"npu-smi usage row has no HBM column: {line.strip()[:160]!r}")
        samples.append({
            "index": card["index"] if chips == 0 else f"{card['index']}.{chips}",
            "uuid": cells[1],  # bus id: stable per chip, npu-smi prints no uuid here
            "util_pct": numbers[0],
            "mem_used_mb": numbers[-2],
            "mem_total_mb": numbers[-1],
            "power_w": card["power_w"],
            "temp_c": card["temp_c"],
            "health": card["health"],
        })
        chips += 1
    if not samples:
        raise CardTelemetryError(f"npu-smi info printed no chip usage row: {text.strip()[:200]!r}")
    return samples


class NpuSmiCollector:
    source = "npu-smi"
    ARGV = ("npu-smi", "info")

    def __init__(self, run: Run = run_command) -> None:
        self._run = run

    def open(self) -> None:
        parse_npu_smi_usage(self._run(self.ARGV))

    def sample(self) -> list[dict[str, Any]]:
        return parse_npu_smi_usage(self._run(self.ARGV))


# --- ppu-smi (nvidia-smi query form) ------------------------------------------

PPU_QUERY = ("index", "uuid", "utilization.gpu", "memory.used", "memory.total", "power.draw",
             "temperature.gpu")


def parse_smi_query_csv(text: str, tool: str = "ppu-smi") -> list[dict[str, Any]]:
    """Rows of ``--query-gpu=<PPU_QUERY> --format=csv,noheader,nounits``."""
    samples = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(PPU_QUERY) or not parts[0].isdigit():
            continue
        values = [None if p in ("", "[N/A]", "N/A", "[Not Supported]") else p for p in parts]
        samples.append({
            "index": int(parts[0]), "uuid": values[1],
            "util_pct": None if values[2] is None else _num(values[2]),
            "mem_used_mb": None if values[3] is None else _num(values[3]),
            "mem_total_mb": None if values[4] is None else _num(values[4]),
            "power_w": None if values[5] is None else _num(values[5]),
            "temp_c": None if values[6] is None else _num(values[6]),
            "health": None,  # the query form has no health field
        })
    if not samples:
        raise CardTelemetryError(f"{tool} printed no card row: {text.strip()[:200]!r}")
    return samples


class PpuSmiCollector:
    source = "ppu-smi"
    ARGV = ("ppu-smi", f"--query-gpu={','.join(PPU_QUERY)}", "--format=csv,noheader,nounits")

    def __init__(self, run: Run = run_command) -> None:
        self._run = run

    def open(self) -> None:
        parse_smi_query_csv(self._run(self.ARGV))

    def sample(self) -> list[dict[str, Any]]:
        return parse_smi_query_csv(self._run(self.ARGV))


# --- amd-smi / rocm-smi (ROCm) -------------------------------------------------
#
# ``amd-smi metric --json`` prints a list, one object per GPU:
#   [{"gpu": 0, "usage": {"gpu_activity": {"value": 12, "unit": "%"}, ...},
#     "power": {"socket_power": {"value": 650, "unit": "W"}, ...},
#     "temperature": {"edge": {"value": 45, "unit": "C"}, "hotspot": {...}},
#     "mem_usage": {"total_vram": {"value": 294912, "unit": "MB"}, "used_vram": {...}}}]
# A value the tool cannot read is "N/A". ``rocm-smi --showuse --showmemuse --showpower
# --showtemp --showmeminfo vram --showuniqueid --json`` prints one object keyed "card0" ...
# with flat "<label>": "<string>" entries whose labels changed across ROCm releases, so
# they are matched by substring.

import json as _json  # noqa: E402 - keeps the NVML / npu / ppu sections above unchanged


def _metric(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("value")
    if value in (None, "N/A", ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mb(entry: Any) -> float | None:
    """A memory metric in MB (``{"value", "unit"}`` of amd-smi; plain bytes from rocm-smi)."""
    if isinstance(entry, dict):
        value = _metric(entry)
        unit = str(entry.get("unit") or "MB").upper()
        if value is None:
            return None
        return value * {"B": 1 / 2**20, "KB": 1 / 1024, "MB": 1.0, "GB": 1024.0}.get(unit, 1.0)
    value = _metric(entry)
    return None if value is None else value / 2**20  # rocm-smi prints bytes


def parse_amd_smi_metric(text: str) -> list[dict[str, Any]]:
    """Card samples from ``amd-smi metric --json``; raises on output without a GPU list."""
    try:
        doc = _json.loads(text)
    except ValueError as exc:
        raise CardTelemetryError(f"amd-smi printed no JSON: {exc}; {text.strip()[:160]!r}") from None
    if isinstance(doc, dict):  # some releases wrap the list
        doc = doc.get("gpus") or doc.get("gpu_data") or [doc]
    samples = []
    for card in doc if isinstance(doc, list) else []:
        if not isinstance(card, dict) or "gpu" not in card:
            continue
        usage, power = card.get("usage") or {}, card.get("power") or {}
        temp, mem = card.get("temperature") or {}, card.get("mem_usage") or {}
        samples.append({
            "index": int(card["gpu"]), "uuid": card.get("uuid") or card.get("bdf"),
            "util_pct": _metric(usage.get("gpu_activity")),
            "mem_used_mb": _mb(mem.get("used_vram")), "mem_total_mb": _mb(mem.get("total_vram")),
            "power_w": _metric(power.get("socket_power") or power.get("current_socket_power")),
            "temp_c": _metric(temp.get("hotspot") or temp.get("edge")),
            "health": None,
        })
    if not samples:
        raise CardTelemetryError(f"amd-smi metric printed no GPU entry: {text.strip()[:200]!r}")
    return samples


def _rocm_field(card: dict[str, Any], *needles: str) -> Any:
    for key, value in card.items():
        label = key.lower()
        if all(n in label for n in needles):
            return value
    return None


def parse_rocm_smi_json(text: str) -> list[dict[str, Any]]:
    """Card samples from ``rocm-smi ... --json`` (``{"card0": {...}}``)."""
    try:
        doc = _json.loads(text)
    except ValueError as exc:
        raise CardTelemetryError(f"rocm-smi printed no JSON: {exc}; {text.strip()[:160]!r}") from None
    samples = []
    for key, card in (doc.items() if isinstance(doc, dict) else ()):
        match = re.fullmatch(r"card(\d+)", str(key))
        if match is None or not isinstance(card, dict):
            continue
        total = _mb(_rocm_field(card, "vram total memory"))
        used = _mb(_rocm_field(card, "vram total used"))
        samples.append({
            "index": int(match.group(1)), "uuid": _rocm_field(card, "unique id") or _rocm_field(card, "pci bus"),
            "util_pct": _metric(_rocm_field(card, "gpu use")),
            "mem_used_mb": used, "mem_total_mb": total,
            "power_w": _metric(_rocm_field(card, "power") if "power" in " ".join(card).lower() else None),
            "temp_c": _metric(_rocm_field(card, "temperature", "junction") or _rocm_field(card, "temperature", "edge")),
            "health": None,
        })
    if not samples:
        raise CardTelemetryError(f"rocm-smi printed no card entry: {text.strip()[:200]!r}")
    return samples


SYSFS_DRM = "/sys/class/drm"


def read_sysfs_vram(root: str = SYSFS_DRM) -> list[dict[str, Any]]:
    """Card samples from the amdgpu sysfs files (bytes / percent) under ``root``."""
    from pathlib import Path  # noqa: PLC0415

    def num(p: Path) -> float | None:
        try:
            return float(p.read_text().strip())
        except (OSError, ValueError):
            return None

    samples = []
    for card in sorted(Path(root).glob("card*")):
        match = re.fullmatch(r"card(\d+)", card.name)
        dev = card / "device"
        used = num(dev / "mem_info_vram_used")
        if match is None or used is None:
            continue  # connectors (card0-DP-1) and non-amdgpu cards
        total = num(dev / "mem_info_vram_total")
        busy = num(dev / "gpu_busy_percent")
        uid = None
        try:
            uid = (dev / "unique_id").read_text().strip() or None
        except OSError:
            pass
        samples.append({"index": int(match.group(1)), "uuid": uid, "util_pct": busy,
                        "mem_used_mb": used / 2**20, "mem_total_mb": None if total is None else total / 2**20,
                        "power_w": None, "temp_c": None, "health": None})
    if not samples:
        raise CardTelemetryError(f"no amdgpu mem_info_vram_used under {root}")
    return samples


class SysfsAmdCollector:
    source = "sysfs-amdgpu"

    def __init__(self, root: str = SYSFS_DRM) -> None:
        self.root = root

    def open(self) -> None:
        read_sysfs_vram(self.root)

    def sample(self) -> list[dict[str, Any]]:
        return read_sysfs_vram(self.root)


class FallbackCollector:
    """The first collector of ``chain`` that opens; ``source`` is its source. When
    none opens, the error names every attempt."""

    def __init__(self, *chain: Any, report_first: bool = False) -> None:
        self.chain = chain
        self.report_first = report_first  # none opens: re-raise the first error unchanged
        self.active = chain[0]

    @property
    def source(self) -> str:
        return self.active.source

    def open(self) -> None:
        errors, first = [], None
        for c in self.chain:
            try:
                c.open()
            except Exception as error:  # noqa: BLE001 - ImportError, tool missing, parse error ...
                first = first or error
                errors.append(f"{getattr(c, 'source', type(c).__name__)}: {type(error).__name__}: {error}")
                continue
            self.active = c
            return
        if self.report_first and first is not None:
            raise first
        raise CardTelemetryError("; ".join(errors)[:600])

    def sample(self) -> list[dict[str, Any]]:
        return self.active.sample()


class RocmSmiCollector:
    """``amd-smi metric --json`` when it works, else ``rocm-smi --json``; the one
    that opened is the sample source (``source`` names it)."""

    AMD_SMI_ARGV = ("amd-smi", "metric", "--json")
    ROCM_SMI_MIN_ARGV = ("rocm-smi", "--showmeminfo", "vram", "--showuse", "--json")
    ROCM_SMI_ARGV = ("rocm-smi", "--showuse", "--showmemuse", "--showpower", "--showtemp", "--showmeminfo", "vram",
                     "--showuniqueid", "--json")

    def __init__(self, run: Run = run_command) -> None:
        self._run = run
        self.source = "amd-smi"
        self._argv: tuple[str, ...] = self.AMD_SMI_ARGV
        self._parse: Callable[[str], list[dict[str, Any]]] = parse_amd_smi_metric

    def open(self) -> None:
        try:
            parse_amd_smi_metric(self._run(self.AMD_SMI_ARGV))
        except CardTelemetryError as first:
            for argv in (self.ROCM_SMI_ARGV, self.ROCM_SMI_MIN_ARGV):
                try:
                    parse_rocm_smi_json(self._run(argv))
                except CardTelemetryError as second:
                    first = CardTelemetryError(f"{first}; {second}")
                    continue
                self.source, self._argv, self._parse = "rocm-smi", argv, parse_rocm_smi_json
                return
            raise first

    def sample(self) -> list[dict[str, Any]]:
        return self._parse(self._run(self._argv))


def rocm_collector(run: Run = run_command, sysfs_root: str = SYSFS_DRM) -> CardCollector:
    """amd-smi -> rocm-smi -> rocm-smi (vram + use only) -> amdgpu sysfs."""
    return FallbackCollector(RocmSmiCollector(run), SysfsAmdCollector(sysfs_root))


def collector_for(family: str, *, nvml_loader: Callable[[], Any] | None = None,
                  run: Run = run_command) -> CardCollector:
    """The collector of a launcher device family (``cuda``, ``npu``, ``ppu``, ``rocm``)."""
    if family == "cuda":
        if nvml_loader is None:
            raise ValueError("cuda collector needs an NVML loader")
        # the NVML record stays as before when no AMD tool answers either
        return FallbackCollector(NvmlCollector(nvml_loader), rocm_collector(run), report_first=True)
    if family == "npu":
        return NpuSmiCollector(run)
    if family == "ppu":
        return PpuSmiCollector(run)
    if family == "rocm":
        return rocm_collector(run)
    raise ValueError(f"unknown device family {family!r} (known: cuda, npu, ppu, rocm)")
