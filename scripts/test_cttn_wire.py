"""Python proxy test for the Rust <-> action-probe CTTN wire contract."""

from __future__ import annotations

import hashlib

import torch

from yeto.action_probe import (
    _slice_f32,
    build_cttn_request_frame,
    build_cttn_result_frame,
    decode_frame,
    parse_cttn_request,
)


def check(name: str, condition: bool) -> bool:
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    return condition


def main() -> int:
    digest = lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
    current_state = {
        "layer.z": torch.tensor([[5.0, 6.0]], dtype=torch.float32),
        "layer.a": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
    }
    fragment_names = ("layer.z", "layer.a")
    g = torch.tensor([0.5, -0.25, 1.0, 2.0, -3.0, 4.0], dtype=torch.float32)
    b = torch.tensor([-1.0, 0.0, 0.5, -0.5, 2.0, 3.0], dtype=torch.float32)
    encoded = build_cttn_request_frame(
        request_id="cttn-wire-test",
        run_uuid="run-wire-test",
        step=9,
        fragment_id=0,
        base_version=8,
        state_epoch=3,
        fragment_versions=(8,),
        layout_hash=digest("layout"),
        anchor_manifest_sha256=digest("anchor"),
        probe_config_sha256=digest("config"),
        current_state=current_state,
        fragment_names=fragment_names,
        g=g,
        b=b,
        mu=0.9,
        rho=0.1,
        block_steps=4,
    )
    frame = decode_frame(encoded)
    request = parse_cttn_request(frame)

    ok = True
    print("CTTN request wire round-trip:")
    ok &= check("header type is cttn_step", frame.header["type"] == "cttn_step")
    ok &= check(
        "payload layout is state tensors then g then b",
        frame.header["cttn"]["g"]["offset"]
        == sum(spec["nbytes"] for spec in frame.header["state"]["tensors"])
        and frame.header["cttn"]["b"]["offset"]
        == frame.header["cttn"]["g"]["offset"]
        + frame.header["cttn"]["g"]["nbytes"],
    )
    ok &= check("g round-trips", torch.equal(request.g, g))
    ok &= check("b round-trips", torch.equal(request.b, b))
    ok &= check("mu round-trips", request.mu == 0.9)
    ok &= check("rho round-trips", request.rho == 0.1)
    ok &= check("fragment order round-trips", request.fragment_names == fragment_names)
    ok &= check(
        "g and b digests round-trip",
        request.g_digest == frame.header["cttn"]["g"]["sha256"]
        and request.b_digest == frame.header["cttn"]["b"]["sha256"],
    )

    d = torch.tensor([0.4, -0.2, 0.9, 1.8, -2.7, 3.6], dtype=torch.float32)
    b_new = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float32)
    diagnostics = {
        "bind": True,
        "tau": 1.25,
        "retention": 0.4,
        "e_before": 5.0,
        "e_after": 0.5,
        "budget": 0.75,
        "n_modes_90": 2,
        "ritz_max": 7.0,
        "loss": 1.5,
    }
    response = decode_frame(
        build_cttn_result_frame(
            request,
            d,
            b_new,
            diagnostics,
            anchor_tensors_sha256=digest("anchor-tensors"),
        )
    )
    parsed_d, offset, _ = _slice_f32(response.payload, response.header["d"], 0)
    parsed_b_new, offset, _ = _slice_f32(
        response.payload, response.header["b_new"], offset
    )
    print("CTTN result wire round-trip:")
    ok &= check("header type is cttn_result", response.header["type"] == "cttn_result")
    ok &= check("d round-trips", torch.equal(parsed_d, d))
    ok &= check("b_new round-trips", torch.equal(parsed_b_new, b_new))
    ok &= check("result payload is fully claimed", offset == len(response.payload))
    ok &= check("diagnostics round-trip", response.header["diagnostics"] == diagnostics)

    zero_budget_diagnostics = dict(diagnostics, tau=float("inf"), budget=0.0, e_after=0.0)
    zero_budget_response = decode_frame(
        build_cttn_result_frame(
            request,
            d,
            b_new,
            zero_budget_diagnostics,
            anchor_tensors_sha256=digest("anchor-tensors"),
        )
    )
    ok &= check(
        "zero-budget tau=+infinity is encoded as JSON null",
        zero_budget_response.header["diagnostics"]["tau"] is None,
    )

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
