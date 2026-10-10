"""yeto-framework-decoupling 7.7a-c: card-type compatibility group.

User decision 2026-10-09: islands with different card types are refused
(legacy HELLO and elastic JOIN), the error names both card types, H100 and
H200 are different types; driver and CUDA versions are recorded only.
The real-syncer refusals are in test_rl_strict_session_reject.py (HELLO) and
syncer/src/elastic_server.rs (JOIN)."""

from __future__ import annotations

import struct

import pytest

from yeto.gpu_spec import _GPU_CANONICAL, device_type_of, parse_gpu_spec
from yeto.hw import catalog
from yeto.hw.catalog import (CompatGroupMismatch, UnknownCardType, check_compat_group,
                             compat_group, compat_refusal, island_compat_group)
from yeto.protocol import HELLO_COMPAT_MAGIC, encode_hello
from yeto.rl.adapters.miles import identity as miles_identity
from yeto.rl.adapters.verl import identity as verl_identity
from yeto.rl.engine.backend_identity import BackendIdentityMismatch, check_identity_match

from test_rl_integration import _layout


def test_compat_group_values_from_the_catalog():
    assert compat_group("H100") == "nvidia-h100"
    assert compat_group("h200") == "nvidia-h200"
    assert compat_group("910B") == "ascend-910b"
    assert compat_group(parse_gpu_spec("aws:8xh100@us-east-2")[0].gpu) == "nvidia-h100"
    assert compat_group("910b4") == "ascend-910b4"
    # every card the launcher accepts has a group (no launch reaches the island without one)
    for card in _GPU_CANONICAL.values():
        vendor = "ascend" if device_type_of(card) == "npu" else "nvidia"
        assert compat_group(card).startswith(vendor + "-")


@pytest.mark.parametrize("card", ["", "h300", "mi300x"])
def test_unknown_card_type_is_an_error(card):
    with pytest.raises(UnknownCardType):
        compat_group(card)


def test_missing_island_group_is_an_error_before_start(monkeypatch):
    monkeypatch.delenv(catalog.COMPAT_GROUP_ENV, raising=False)
    with pytest.raises(UnknownCardType, match="not set"):
        island_compat_group()
    with pytest.raises(UnknownCardType):
        miles_identity.backend_identity("ports")
    with pytest.raises(UnknownCardType):
        verl_identity.backend_identity()
    assert miles_identity.backend_identity("ports", compat_group="nvidia-h200").compat_group == "nvidia-h200"
    monkeypatch.setenv(catalog.COMPAT_GROUP_ENV, "nvidia-h200")
    assert island_compat_group() == "nvidia-h200"
    with pytest.raises(ValueError):
        island_compat_group("H100")  # must be the lower-case "<vendor>-<card>" form


def test_compat_group_is_in_the_identity_hash():
    h100 = miles_identity.backend_identity("ports", compat_group="nvidia-h100")
    h200 = miles_identity.backend_identity("ports", compat_group="nvidia-h200")
    assert h100.sha256() != h200.sha256()
    assert h100.to_dict()["compat_group"] == "nvidia-h100"
    assert type(h100).from_dict(h100.to_dict()) == h100


def test_comparison_rules():
    assert compat_refusal("nvidia-h200", "nvidia-h200") is None
    reason = compat_refusal("nvidia-h100", "nvidia-h200")
    assert reason.startswith("兼容组不同：nvidia-h100 对 nvidia-h200，容差未标定")
    assert "容差未标定" in compat_refusal("nvidia-h100", "ascend-910b")
    assert "未声明" in compat_refusal("nvidia-h100", None)
    assert catalog.CARD_PAIR_TOLERANCE == {}  # phase 1: table key exists, empty
    with pytest.raises(CompatGroupMismatch):
        check_compat_group("nvidia-h100", "nvidia-h200")


def test_identity_check_names_both_card_types():
    h100 = miles_identity.backend_identity("ports", compat_group="nvidia-h100")
    with pytest.raises(CompatGroupMismatch, match="nvidia-h100 对 nvidia-h200"):
        check_identity_match(h100, miles_identity.backend_identity("ports", compat_group="nvidia-h200"))
    with pytest.raises(CompatGroupMismatch, match="nvidia-h100 对 ascend-910b"):
        check_identity_match(h100, verl_identity.backend_identity(compat_group="ascend-910b"))
    check_identity_match(miles_identity.backend_identity("ports", compat_group="nvidia-h200"),
                         miles_identity.backend_identity("ports", compat_group="nvidia-h200"))
    with pytest.raises(BackendIdentityMismatch):  # same card, other backend: old rule
        check_identity_match(h100, verl_identity.backend_identity(compat_group="nvidia-h100"))


def test_hello_compat_trailer_layout():
    layout = _layout()
    plain = encode_hello(0, 0, layout, 2, 1, bytes(32))
    tagged = encode_hello(0, 0, layout, 2, 1, bytes(32), compat_group="nvidia-h100")
    assert tagged == plain + HELLO_COMPAT_MAGIC + struct.pack("<I", 11) + b"nvidia-h100"


def test_runtime_versions_never_raise():
    versions = catalog.runtime_versions()
    assert {"driver_version", "cuda_version"} <= set(versions)


def test_dashboard_shows_hardware_record():
    from yeto.dashboard.reducer import Reducer

    red = Reducer()
    red.feed({"event": catalog.HARDWARE_EVENT, "island_id": 0, "time_unix": 1.0,
              "compat_group": "nvidia-h100", "driver_version": "550.54", "cuda_version": "12.8"})
    view = red.page_view(now=2.0)
    assert view["islands"]["0"]["hardware"] == {"compat_group": "nvidia-h100",
                                                "driver_version": "550.54", "cuda_version": "12.8"}
