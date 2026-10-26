"""Card-tool adapters: one record shape for ``nvidia-smi``, ``npu-smi`` and
``ppu-smi`` (rl-ssh-node-pool design D3).

Every family has a parser that turns the tool output into
:class:`CardRecord` values (index, card name, HBM GiB, health). The launcher
compares those records against the node inventory before any accelerator
process starts, and refuses the launch on a card, count or HBM difference
(:func:`check_cards`).

Verification state of the output formats:

- ``nvidia-smi --query-gpu=...``: verified, this CSV form is already used in
  ``yeto/rl/ssh_harness.py`` and ``yeto/launch/remote_setup.py``.
- ``npu-smi info`` / ``npu-smi info -t memory``: the row shape comes from the
  sample recorded in ``yeto/gpu_spec.py`` (``_NPU_SMI_ROW_RE``, 910B4) and
  from ``yeto/hw/catalog.py``, which already reads ``npu-smi info -l`` and
  ``-t board``. The driver version of that sample is not recorded, and
  driver 25.x is not checked.
- ``ppu-smi``: NOT verified. ``infra-drafts/SESSION24/T-PPU-gaps.md`` 2.2 says
  the PPU stack offers NVML-compatible APIs, so the parser accepts the
  NVIDIA CSV form and a ``key: value`` form. A real machine must confirm it
  (task N1.2, GPU half).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from ..gpu_spec import DEVICE_FAMILIES, UnknownCard, canonical_card, device_family_of

# Driver-reserved HBM makes the reported total smaller than the nominal size
# (an H200 reports 143771 MiB = 140.4 GiB for a 141 GiB card; a 910B4 64 GiB
# card reports 65536 MiB). The comparison keeps this much slack, so a 32 GiB
# card can never pass as a 64 GiB card.
HBM_TOLERANCE = 0.08

MIB_PER_GIB = 1024.0


class CardToolError(RuntimeError):
    """The card tool output could not be parsed."""


class CardMismatch(RuntimeError):
    """The cards on the node differ from the ones the inventory declares."""


@dataclass(frozen=True)
class CardRecord:
    """One accelerator as its card tool reports it."""

    index: int
    card: str  # canonical catalog name, e.g. "910B4"
    hbm_gib: float | None
    health: str | None = None


# Command every family runs for the preflight. The parsers accept the output
# of all of a family's commands concatenated, so the executor may send them
# as one shell line (``cmd_a; cmd_b``).
PREFLIGHT_COMMANDS: dict[str, tuple[tuple[str, ...], ...]] = {
    "cuda": (("nvidia-smi", "--query-gpu=index,name,memory.total",
              "--format=csv,noheader,nounits"),),
    "npu": (("npu-smi", "info"), ("npu-smi", "info", "-t", "memory")),
    "ppu": (("ppu-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"),),
}


def preflight_commands(family: str) -> tuple[tuple[str, ...], ...]:
    """The card-tool commands of a device family; unknown family -> error."""
    try:
        return PREFLIGHT_COMMANDS[family]
    except KeyError:
        raise ValueError(f"unknown device family {family!r} "
                         f"(known: {DEVICE_FAMILIES})") from None


# --- nvidia-smi / ppu-smi (NVML CSV) ----------------------------------------

def _card_from_product_name(text: str) -> str:
    """Catalog card name inside a product name such as ``NVIDIA H200 NVL``.

    Every word of the product name is tried against the catalog; the first
    known card wins. Nothing known -> :class:`UnknownCard` with the whole
    product name, so the launch refusal shows what the tool reported.
    """
    for word in re.split(r"[\s/,]+", text.strip()):
        if not word:
            continue
        try:
            return canonical_card(word)
        except UnknownCard:
            continue
    raise UnknownCard(f"card tool reports {text.strip()!r}, which is not in the "
                      "yeto card catalog; refusing the launch (an unknown card "
                      "is never treated as cuda)")


def parse_nvml_csv(text: str) -> list[CardRecord]:
    """``index, name, memory.total`` CSV rows (``nvidia-smi``, ``ppu-smi``)."""
    records: list[CardRecord] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "+", "|", "=")):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3 or not fields[0].isdigit():
            continue
        total_mib = fields[2].split()[0] if fields[2] else ""
        records.append(CardRecord(
            index=int(fields[0]),
            card=_card_from_product_name(fields[1]),
            hbm_gib=(float(total_mib) / MIB_PER_GIB if _is_number(total_mib) else None),
        ))
    return records


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


# --- npu-smi ----------------------------------------------------------------
#
# ``npu-smi info`` prints two rows per chip:
#
#   | 0     910B4                   | OK              | 92.5  47  0 / 0     |
#   | 0                             | 0000:C1:00.0    | 0     0 / 0  3161 / 65536 |
#
# The first row carries the card name and the health column, the second the
# HBM usage in MB. ``npu-smi info -t memory`` prints ``HBM Capacity(MB)``,
# which wins when both are present (it is the capacity, not the usage total).

_NPU_NAME_ROW = re.compile(r"^\|\s*(\d+)\s+(\S+)\s*\|\s*([A-Za-z]+)\s*\|")
# The usage row's first column is the chip id inside the card, not the NPU id,
# so the row belongs to the card of the name row above it.
_NPU_USAGE_ROW = re.compile(r"^\|\s*\d+\s+\|.*?(\d+)\s*/\s*(\d+)\s*\|?\s*$")
_NPU_CAPACITY_LINE = re.compile(r"HBM\s+Capacity\s*\(MB\)\s*:\s*(\d+)", re.IGNORECASE)
_NPU_ID_LINE = re.compile(r"NPU\s+ID\s*:\s*(\d+)", re.IGNORECASE)


def parse_npu_smi(text: str) -> list[CardRecord]:
    """``npu-smi info`` (plus an optional ``-t memory``) as card records."""
    names: dict[int, str] = {}
    health: dict[int, str] = {}
    totals_mb: dict[int, int] = {}
    capacities_mb: dict[int, int] = {}
    npu_id: int | None = None
    last_index: int | None = None
    for raw in text.splitlines():
        line = raw.strip()
        name_row = _NPU_NAME_ROW.match(line)
        if name_row is not None and ":" not in name_row.group(2):
            index, name, state = int(name_row.group(1)), name_row.group(2), name_row.group(3)
            if name.lower() not in ("name", "chip"):
                names.setdefault(index, name)
                health.setdefault(index, state)
                last_index = index
            continue
        usage_row = _NPU_USAGE_ROW.match(line)
        if usage_row is not None and last_index is not None:
            totals_mb.setdefault(last_index, int(usage_row.group(2)))
            continue
        id_line = _NPU_ID_LINE.search(line)
        if id_line is not None:
            npu_id = int(id_line.group(1))
            continue
        capacity_line = _NPU_CAPACITY_LINE.search(line)
        if capacity_line is not None:
            capacities_mb[npu_id if npu_id is not None else 0] = int(capacity_line.group(1))
    if not names:
        raise CardToolError("npu-smi info reported no card row; cannot check the card "
                           f"type. Output was: {text.strip()[:200]!r}")
    records = []
    for index in sorted(names):
        # npu-smi adds a "-1" / "-2" chip-variant suffix; a 910B4-1 is a 910B4.
        card = _card_from_product_name(names[index].split("-")[0])
        mb = capacities_mb.get(index, totals_mb.get(index))
        records.append(CardRecord(index=index, card=card,
                                  hbm_gib=(mb / MIB_PER_GIB if mb else None),
                                  health=health.get(index)))
    return records


# --- ppu-smi ----------------------------------------------------------------

_PPU_KEY_VALUE = re.compile(r"^(Device|Index|Name|Product Name|Memory|HBM Capacity)"
                            r"[^:]*:\s*(.+?)\s*$", re.IGNORECASE)


def parse_ppu_smi(text: str) -> list[CardRecord]:
    """``ppu-smi`` output (NVML CSV form, or a ``key: value`` block per card).

    The format is NOT verified against a real PPU machine (T-PPU-gaps.md 2.2);
    both forms are accepted so that the first real run only has to confirm
    one of them.
    """
    records = parse_nvml_csv(text)
    if records:
        return records
    blocks: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for raw in text.splitlines():
        match = _PPU_KEY_VALUE.match(raw.strip())
        if match is None:
            continue
        key, value = match.group(1).lower(), match.group(2)
        if key in ("device", "index") and current:
            blocks.append(current)
            current = {}
        current[key] = value
    if current:
        blocks.append(current)
    if not blocks:
        raise CardToolError("ppu-smi reported no card; cannot check the card type. "
                           f"Output was: {text.strip()[:200]!r}")
    for position, block in enumerate(blocks):
        index = block.get("index") or block.get("device") or str(position)
        name = block.get("name") or block.get("product name")
        if name is None:
            raise CardToolError(f"ppu-smi card {index} has no name field")
        memory = block.get("hbm capacity") or block.get("memory") or ""
        mib = re.search(r"(\d+)", memory)
        records.append(CardRecord(
            index=int(re.sub(r"\D", "", index) or position),
            card=_card_from_product_name(name),
            hbm_gib=(int(mib.group(1)) / MIB_PER_GIB if mib else None),
        ))
    return records


_PARSERS = {"cuda": parse_nvml_csv, "npu": parse_npu_smi, "ppu": parse_ppu_smi}


def parse_card_tool(family: str, text: str) -> list[CardRecord]:
    """Card records from the output of a device family's card tool."""
    try:
        parser = _PARSERS[family]
    except KeyError:
        raise ValueError(f"unknown device family {family!r} "
                         f"(known: {DEVICE_FAMILIES})") from None
    records = parser(text)
    if not records:
        raise CardToolError(f"the {family} card tool reported no card; cannot check the "
                           f"card type. Output was: {text.strip()[:200]!r}")
    return records


def hbm_matches(declared_gib: float, reported_gib: float) -> bool:
    """True when a reported HBM total is the declared card size.

    The tools report the usable total, which is below the nominal size by the
    driver reservation; :data:`HBM_TOLERANCE` is the slack.
    """
    return reported_gib >= declared_gib * (1.0 - HBM_TOLERANCE)


def check_cards(family: str, text: str, *, card: str, count: int,
                hbm_gib: float | None = None,
                node: str | None = None) -> list[CardRecord]:
    """Refuse the launch when the node's cards differ from the inventory.

    Checks, in this order: the card tool output parses, every card name is in
    the catalog and equals ``card``, the card count equals ``count``, and the
    reported HBM total is the declared ``hbm_gib`` (when given). Returns the
    records so that the caller can record them.
    """
    where = f" on node {node}" if node else ""
    expected = canonical_card(card)
    if device_family_of(expected) != family:
        raise CardMismatch(f"card {expected} is not a {family} card{where}")
    try:
        records = parse_card_tool(family, text)
    except UnknownCard as exc:
        raise CardMismatch(f"{exc}{where}") from None
    wrong = sorted({record.card for record in records if record.card != expected})
    if wrong:
        raise CardMismatch(
            f"the {family} card tool reports card(s) {wrong}{where} but the inventory "
            f"declares {expected}; refusing to start (the card type is a numerics "
            "boundary)")
    if len(records) != count:
        raise CardMismatch(
            f"the {family} card tool reports {len(records)} x {expected}{where}, the "
            f"inventory declares {count}; refusing to start")
    if hbm_gib is not None:
        bad = [record for record in records
               if record.hbm_gib is not None and not hbm_matches(hbm_gib, record.hbm_gib)]
        if bad:
            raise CardMismatch(
                f"the inventory declares {hbm_gib:g} GiB HBM per {expected}{where} but "
                f"the card tool reports {_format_hbm(bad)}; refusing to start")
        if any(record.hbm_gib is None for record in records):
            raise CardMismatch(
                f"the {family} card tool reported no HBM capacity{where}; the inventory "
                f"declares {hbm_gib:g} GiB and the check cannot be skipped")
    return records


def _format_hbm(records: Sequence[CardRecord]) -> str:
    return ", ".join(f"card {record.index}: {record.hbm_gib:.1f} GiB" for record in records)
