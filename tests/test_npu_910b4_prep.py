"""Ascend 910B4 pre-arrival unit tests (rl-verl-backend 3.10, 3.12, 3.14, 3.15).

No NPU is needed: ``torch_npu`` is a stub module, ``npu-smi`` output is a
captured string, and the accelerator family is monkeypatched. Same approach as
tests/test_accel.py. What these tests cannot cover is listed in
infra-drafts/S19-NPU-910B4-BRINGUP.md ("to verify on hardware").
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from yeto import gpu_spec
from yeto.gpu_spec import (NpuCardMismatch, assert_npu_cards, device_type_of,
                           parse_gpu_spec, parse_npu_smi_names)
from yeto.hw import catalog
from yeto.rl.adapters.verl import identity as verl_identity
from yeto.rl.adapters.verl import pins
from yeto.rl.engine.backend_identity import BackendIdentityMismatch, check_identity_match

# Shape of ``npu-smi info`` on a 910B4 node (two chips shown).
NPU_SMI_INFO = """
+------------------------------------------------------------------------------+
| npu-smi 24.1.0                   Version: 24.1.0                             |
+-------------------------------+-----------------+----------------------------+
| NPU   Name                    | Health          | Power(W)   Temp(C)         |
| Chip                          | Bus-Id          | AICore(%)  Memory(MB)      |
+===============================+=================+============================+
| 0     910B4                   | OK              | 92.5       47              |
| 0                             | 0000:C1:00.0    | 0          0 / 0           |
+===============================+=================+============================+
| 1     910B4                   | OK              | 93.1       45              |
| 1                             | 0000:C2:00.0    | 0          0 / 0           |
+===============================+=================+============================+
"""


@pytest.fixture
def fake_torch_npu(monkeypatch):
    """Install a fake ``torch_npu`` module, as the real extension would be."""
    stub = SimpleNamespace(__version__="2.10.0.post4", __name__="torch_npu")
    monkeypatch.setitem(sys.modules, "torch_npu", stub)
    return stub


@pytest.fixture
def ascend_node(monkeypatch, fake_torch_npu):
    """Make family auto-detection report an Ascend node."""
    from yeto import accel

    monkeypatch.setattr(accel, "available_type", lambda: "npu")
    return fake_torch_npu


# --- 3.15: the launcher grammar knows the card ------------------------------


def test_gpu_spec_parses_a_910b4_island():
    spec = parse_gpu_spec("ssh:1x8x910b4")[0]
    assert (spec.gpu, spec.gpus_per_node, spec.total_gpus) == ("910B4", 8, 8)
    assert spec.accelerators == "910B4:8"


def test_card_device_type_separates_npu_from_gpu():
    assert device_type_of("910B4") == "npu"
    assert device_type_of("910B") == "npu"
    assert device_type_of("H100") == "cuda"


def test_an_unknown_card_is_still_refused():
    with pytest.raises(ValueError, match="unknown GPU"):
        parse_gpu_spec("ssh:1x8x910c9")


def test_every_launchable_card_has_a_compat_group():
    for card in gpu_spec._GPU_CANONICAL.values():
        vendor = "ascend" if device_type_of(card) == "npu" else "nvidia"
        assert catalog.compat_group(card) == f"{vendor}-{card.lower()}"


# --- 3.9 (unit half): npu-smi card-name assertion ---------------------------


def test_npu_smi_rows_give_the_card_names():
    assert parse_npu_smi_names(NPU_SMI_INFO) == ["910B4", "910B4"]


def test_assert_npu_cards_accepts_the_expected_card_and_count():
    assert assert_npu_cards(NPU_SMI_INFO, "910B4", 2) == ["910B4", "910B4"]
    # npu-smi prints a chip variant suffix on some firmware; same card type.
    assert assert_npu_cards(NPU_SMI_INFO.replace("910B4", "910B4-1"), "910b4") == [
        "910B4-1", "910B4-1"]


def test_assert_npu_cards_refuses_another_card_type():
    with pytest.raises(NpuCardMismatch, match="expects 910B4"):
        assert_npu_cards(NPU_SMI_INFO.replace("910B4", "910B2"), "910B4")


def test_assert_npu_cards_refuses_a_wrong_card_count():
    with pytest.raises(NpuCardMismatch, match="asked for 8"):
        assert_npu_cards(NPU_SMI_INFO, "910B4", 8)


def test_assert_npu_cards_refuses_empty_output():
    with pytest.raises(NpuCardMismatch, match="no card row"):
        assert_npu_cards("npu-smi: command not found", "910B4")


# --- 3.12 / identity: device family comes from the device -------------------


def test_device_family_maps_the_torch_device_type():
    assert catalog.device_family("cuda") == "nvidia"
    assert catalog.device_family("npu") == "ascend"
    assert catalog.device_family("cpu") == "nvidia"  # historical default


def test_device_family_reads_the_node_when_not_named(ascend_node):
    assert catalog.device_family() == "ascend"


def test_verl_identity_declares_ascend_on_an_npu_node(ascend_node, monkeypatch):
    monkeypatch.setenv(catalog.COMPAT_GROUP_ENV, "ascend-910b4")
    ident = verl_identity.backend_identity()
    assert ident.device_family == "ascend"
    assert ident.compat_group == "ascend-910b4"


def test_runtime_versions_adds_the_npu_fields(ascend_node):
    versions = catalog.runtime_versions("ascend")
    assert set(versions) == {"driver_version", "cuda_version", "npu_driver_version",
                             "cann_version", "torch_npu_version"}
    assert versions["torch_npu_version"] == "2.10.0.post4"


def test_runtime_versions_keeps_the_nvidia_shape():
    assert set(catalog.runtime_versions("nvidia")) == {"driver_version", "cuda_version"}


# --- 3.10: an NPU island and a GPU island must not merge --------------------


def _identity(family: str, group: str):
    return verl_identity.backend_identity(device_family=family, compat_group=group)


def test_npu_island_and_gpu_island_are_refused():
    npu = _identity("ascend", "ascend-910b4")
    gpu = _identity("nvidia", "nvidia-h100")
    with pytest.raises(Exception) as err:  # CompatGroupMismatch (read first)
        check_identity_match(npu, gpu)
    assert "ascend-910b4" in str(err.value) and "nvidia-h100" in str(err.value)


def test_same_card_but_different_device_family_is_still_refused():
    """Defence in depth: even if someone forced the same compat group, the
    device family is part of the identity hash, so the merge is refused."""
    npu = _identity("ascend", "ascend-910b4")
    mislabelled = _identity("nvidia", "ascend-910b4")
    with pytest.raises(BackendIdentityMismatch):
        check_identity_match(npu, mislabelled)
    assert npu.sha256() != mislabelled.sha256()


def test_two_npu_islands_of_the_same_card_may_merge():
    check_identity_match(_identity("ascend", "ascend-910b4"),
                         _identity("ascend", "ascend-910b4"))


# --- 3.14: version pins branch per device family ----------------------------


def test_expected_versions_per_family():
    assert pins.expected_versions("nvidia")["vllm"] == "0.29.0"
    npu = pins.expected_versions("ascend")
    assert npu["vllm"] == "0.23.0"
    assert npu["torch"] == "2.10.0"
    assert npu["torch_npu"] == "2.10.0.post4"


def test_expected_versions_refuses_an_unknown_family():
    with pytest.raises(ValueError, match="no verl version pins"):
        pins.expected_versions("amd")


def test_the_cuda_pins_would_all_fail_on_an_npu_island():
    """Why 3.14 exists: the two stacks share no version."""
    cuda, npu = pins.expected_versions("nvidia"), pins.expected_versions("ascend")
    assert all(cuda[k] != npu[k] for k in ("torch", "vllm", "transformers"))


def test_npu_sleep_level_is_one():
    """vllm-ascend 0.23 does not support sleep_mode=2 (verl fork acad9875)."""
    assert pins.VERL_NPU_VLLM_SLEEP_LEVEL == 1


def test_npu_image_pins_name_their_sources():
    assert pins.VERL_NPU_CANN_VERSION == "9.1.0"
    assert pins.VERL_NPU_BASE_IMAGE.endswith("cann:9.1.0-910b-ubuntu22.04-py3.12")
    assert pins.VERL_NPU_VLLM_ASCEND_BRANCH == "releases/v0.23.0"


# --- 3.6: the island manifest on an NPU node --------------------------------


def test_runtime_manifest_uses_npu_smi_and_the_npu_pins(ascend_node, monkeypatch):
    from yeto.rl.adapters.verl import island_entry

    monkeypatch.setattr(island_entry.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(stdout=NPU_SMI_INFO, returncode=0))
    manifest = island_entry.runtime_manifest(0, device_family="ascend")
    assert manifest["device_family"] == "ascend"
    assert manifest["expected_versions"]["vllm"] == "0.23.0"
    assert parse_npu_smi_names(manifest["npu_smi"]) == ["910B4", "910B4"]
    assert manifest["versions"]["torch_npu"] == "2.10.0.post4"
    # No NPU image is installed here, so the manifest must report problems.
    assert any("vllm" in p for p in manifest["problems"])


def test_threshold_key_separates_npu_from_gpu():
    from yeto.rl.adapters.verl.island_entry import _threshold_card

    assert _threshold_card({"device_family": "ascend", "npu_smi": NPU_SMI_INFO}) == "910B4"
    assert _threshold_card({"device_family": "nvidia", "nvidia_smi": "NVIDIA H100 80GB"}) == "H100"
