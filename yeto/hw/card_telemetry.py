"""Per-card telemetry collectors for ``rl_resource_sample`` (S25 N2.2,
rl-hetero-sampler-engines "Telemetry by device family").

One collector per device family, one card-sample shape:

* ``cuda``: NVML (``pynvml``), unchanged from fleet-dashboard 2.2;
* ``npu``: ``npu-smi info`` (Ascend);
* ``ppu``: ``ppu-smi --query-gpu=... --format=csv,noheader,nounits``.

A card sample has the keys of :data:`CARD_KEYS`; a value the tool does not
print is None. NVML samples keep their historical six keys
(``telemetry.GPU_SAMPLE_KEYS``); ``npu-smi`` and ``ppu-smi`` samples add
``temp_c`` and ``health``.

Verification state: the ``npu-smi info`` table shape is the one already
recorded in ``yeto/gpu_spec.py`` and ``card_tools`` (910B4); the power and
temperature columns are read from the same rows and are NOT checked against a
driver 25.x machine. ``ppu-smi`` is assumed to accept the nvidia-smi query
form (T-PPU-gaps.md 2.2 says the stack is NVML compatible); NOT verified on a
PPU. Both parsers raise :class:`CardTelemetryError` on output they cannot read,
so a format change shows up as the island's single ``available: false`` record
with the reason, never as silent zeros.
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


def collector_for(family: str, *, nvml_loader: Callable[[], Any] | None = None,
                  run: Run = run_command) -> CardCollector:
    """The collector of a launcher device family (``cuda``, ``npu``, ``ppu``)."""
    if family == "cuda":
        if nvml_loader is None:
            raise ValueError("cuda collector needs an NVML loader")
        return NvmlCollector(nvml_loader)
    if family == "npu":
        return NpuSmiCollector(run)
    if family == "ppu":
        return PpuSmiCollector(run)
    raise ValueError(f"unknown device family {family!r} (known: cuda, npu, ppu)")
