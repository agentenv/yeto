"""Launcher-side card check of ROCm nodes (amd-smi / rocm-smi JSON, CPU fakes):
an MI355X node of the SSH inventory is a ``rocm`` device-family node whose cards
are checked before any start, like the cuda / npu / ppu families."""

from __future__ import annotations

import json
import shlex

import pytest

from yeto.gpu_spec import DEVICE_FAMILIES, canonical_card, device_family_of, device_type_of
from yeto.hw import card_tools

AMD_SMI = json.dumps([
    {"gpu": i, "asic": {"market_name": "AMD Instinct MI355X", "vendor_id": "0x1002", "device_id": "0x75a0"},
     "bus": {"bdf": f"0000:{0x10 + i:02x}:00.0"},
     "vram": {"type": "HBM3E", "vendor": "N/A", "size": {"value": 294912, "unit": "MB"}}}
    for i in range(8)])
ROCM_SMI = json.dumps({f"card{i}": {"Card Series": "AMD Instinct MI355X", "Card Model": "0x75a0",
                                     "Card Vendor": "Advanced Micro Devices, Inc. [AMD/ATI]",
                                     "VRAM Total Memory (B)": "309237645312",
                                     "VRAM Total Used Memory (B)": "1234567"} for i in range(8)})


def test_mi355x_is_a_rocm_card_with_torch_device_type_cuda():
    assert "rocm" in DEVICE_FAMILIES
    assert canonical_card("mi355x") == "MI355X" and device_family_of("MI355X") == "rocm"
    assert device_type_of("MI355X") == "cuda"  # torch on ROCm reports cuda
    assert device_family_of("mi300x") == "rocm" and device_family_of("H200") == "cuda"


def test_preflight_runs_amd_smi_then_rocm_smi():
    # card_tools lists amd-smi; the launcher's inventory node appends the rocm-smi fallback
    assert [c[0] for c in card_tools.preflight_commands("rocm")] == ["amd-smi"]
    assert shlex.join(card_tools.ROCM_SMI_FALLBACK[0]) == "rocm-smi --showproductname --showmeminfo vram --json"
    from tests.test_rl_ssh_node_pool import _inventory_node
    from yeto.launch.node_inventory import load_inventory

    node = _node(_inventory_node, load_inventory)
    assert [c[0] for c in node.preflight_commands()] == ["amd-smi", "rocm-smi"]
    assert all("--json" in c for c in node.preflight_commands())


def _node(_inventory_node, load_inventory, tmp=None):
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp())
    document = {"nodes": [_inventory_node(name="mi00", host="user@mi00", device_family="rocm", cpu_arch="x86_64",
                                          card="MI355X", cards_per_node=8, hbm_gib=288,
                                          image="registry.example/sglang-rocm:1")]}
    path = tmp / "inventory.json"
    path.write_text(json.dumps(document))
    (node,) = load_inventory(path)
    return node


def test_amd_smi_and_rocm_smi_json_give_the_same_records():
    for text in (AMD_SMI, ROCM_SMI):
        records = card_tools.parse_card_tool("rocm", text)
        assert [r.index for r in records] == list(range(8))
        assert {r.card for r in records} == {"MI355X"}
        assert all(r.hbm_gib == pytest.approx(288.0, abs=0.5) for r in records)
    # a GB-unit size
    alt = json.dumps([{"gpu": 0, "asic": {"market_name": "AMD Instinct MI300X"},
                       "vram": {"size": {"value": 192, "unit": "GB"}}}])
    (record,) = card_tools.parse_card_tool("rocm", alt)
    assert record.card == "MI300X" and record.hbm_gib == pytest.approx(192.0)


def test_check_cards_refuses_the_wrong_card_count_or_hbm():
    assert len(card_tools.check_cards("rocm", AMD_SMI, card="MI355X", count=8, hbm_gib=288, node="mi00")) == 8
    with pytest.raises(card_tools.CardMismatch, match="reports 8 x MI355X on node mi00, the inventory declares 4"):
        card_tools.check_cards("rocm", AMD_SMI, card="MI355X", count=4, node="mi00")
    with pytest.raises(card_tools.CardMismatch, match="declares MI300X"):
        card_tools.check_cards("rocm", ROCM_SMI, card="MI300X", count=8)
    with pytest.raises(card_tools.CardMismatch, match="is not a rocm card"):
        card_tools.check_cards("rocm", AMD_SMI, card="H200", count=8)
    with pytest.raises(card_tools.CardMismatch, match="declares 512 GiB"):
        card_tools.check_cards("rocm", AMD_SMI, card="MI355X", count=8, hbm_gib=512)
    with pytest.raises(card_tools.CardToolError, match="printed no JSON"):
        card_tools.parse_card_tool("rocm", "rocm-smi: command not found")
    # an unknown AMD card is refused, never taken for the declared one (the parser
    # keeps its MI<nnn> token so the refusal names what the tool reported)
    unknown = json.dumps([{"gpu": 0, "asic": {"market_name": "AMD Instinct MI999Z"}, "vram": {"size": {"value": 1, "unit": "MB"}}}])
    with pytest.raises(card_tools.CardMismatch, match=r"MI999Z.*declares MI355X"):
        card_tools.check_cards("rocm", unknown, card="MI355X", count=1)


def test_inventory_node_of_a_rocm_sampler(tmp_path):
    from tests.test_rl_ssh_node_pool import _inventory_node
    from yeto.launch.node_inventory import load_inventory

    document = {"nodes": [_inventory_node(name="mi00", host="user@mi00", device_family="rocm", cpu_arch="x86_64",
                                          card="MI355X", cards_per_node=8, hbm_gib=288,
                                          image="registry.example/sglang-rocm:1")]}
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(document))
    (node,) = load_inventory(path)
    assert node.device_family == "rocm" and [c[0] for c in node.preflight_commands()] == ["amd-smi", "rocm-smi"]
    assert len(node.check_cards(AMD_SMI)) == 8
