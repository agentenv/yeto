"""S26 E3 sync options: bf16 push transcoding, session binding, preset scoping. CPU only."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from test_rl_full_parameter_sync import _role_stream_config
from yeto.protocol import (
    DTYPE_BF16,
    DTYPE_F32,
    DTYPE_F32_PUSH_BF16,
    _f32_parts_to_bf16,
    bulk_dtype,
    f32_bytes_to_bf16,
    push_wire_dtype,
)


def test_push_bf16_session_keeps_bulk_payloads_f32():
    assert bulk_dtype(DTYPE_F32_PUSH_BF16) == DTYPE_F32
    assert push_wire_dtype(DTYPE_F32_PUSH_BF16) == DTYPE_BF16
    assert push_wire_dtype(DTYPE_F32) == DTYPE_F32


def test_f32_to_bf16_matches_torch_round_to_nearest_even_across_split_parts():
    values = torch.randn(50_001) * torch.logspace(-20, 20, 50_001)
    values[:3] = torch.tensor([0.0, -0.0, float("inf")])
    raw = values.numpy().tobytes()
    want = values.to(torch.bfloat16).view(torch.int16).numpy().astype("<i2").tobytes()
    assert f32_bytes_to_bf16(raw) == want
    parts = [raw[:5], raw[5:4099], raw[4099:4100], raw[4100:]]  # split mid-value
    assert b"".join(_f32_parts_to_bf16(parts)) == want
    with pytest.raises(ValueError, match="mid-value"):
        list(_f32_parts_to_bf16([raw[:6]]))


def test_push_dtype_is_bound_into_the_role_stream_session_contract():
    from yeto.rl.adapters.miles.full_parameter.role_stream import (
        role_stream_session_contract_hash,
    )

    exact = _role_stream_config("actor", 30100, sweep=True)
    rounded = replace(exact, push_bf16=True)
    assert (role_stream_session_contract_hash("1" * 64, exact)
            != role_stream_session_contract_hash("1" * 64, rounded))


@pytest.mark.parametrize(("flag", "value"), (
    ("--rl-sync-fragment-pipeline", "2"),
    ("--rl-sync-push-dtype", "bf16"),
    ("--rl-sync-checkpoint-every", "4"),
))
def test_e3_options_only_apply_to_the_full_parameter_preset(flag, value):
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", (flag, value), gpu="modal:2xh100")
    args.controller = "local"
    with pytest.raises(ValueError, match=f"{flag} only applies to --rl-sync-preset full-parameter"):
        _prepare_rl_args(args)
