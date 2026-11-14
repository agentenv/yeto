"""New-model onboarding: preflight messages and suggestions (text only, no behaviour change)."""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from yeto.gpu_spec import parse_gpu_spec
from yeto.launch.rl_args import _model_revision_hint, _warn_backend_device
from yeto.launch_preflight import print_dry_run_memory
from yeto.memory_estimate import MemoryRequest, ModelArch, estimate, smallest_parallel_shape, suggestions
from yeto.rl.adapters.sampler_engines import from_spec

DENSE_100B = ModelArch(params=110e9, layers=80, hidden=10240, vocab=151936, intermediate=32768,
                       heads=80, kv_heads=8, head_dim=128)


def _server(**drop):
    server = {"base_dir": "/models/m", "card_model": "L4", "device_family": "cuda",
              "engine_version": "vllm@0.29.0", "precision": "bf16", "quant_scheme": "bf16",
              "topology": "tp1", "max_model_len": 4096}
    for key in drop:
        server.pop(key)
    return server


def test_server_section_missing_fields_are_named_with_their_meaning():
    with pytest.raises(TypeError) as err:
        from_spec.build({"adapter": "vllm-server", "server": _server(engine_version=1, quant_scheme=1)})
    text = str(err.value)
    assert "engine_version (" in text and "quant_scheme (" in text and "bf16" in text


def test_server_section_unknown_field_lists_the_known_ones():
    with pytest.raises(TypeError, match=r"unknown field\(s\) \['max_len'\].*max_model_len"):
        from_spec.build({"adapter": "vllm-server", "server": {**_server(), "max_len": 1}})


def test_complete_server_section_still_builds():
    assert from_spec.build({"adapter": "vllm-server", "server": _server()}) is not None


def test_large_model_gets_a_parallel_shape_that_fits():
    req = MemoryRequest(arch=DENSE_100B, gpu="H100", gpu_mem_gib=78.4, tokens=4096)
    shape = smallest_parallel_shape(req, 0.9)
    assert shape is not None and shape["tensor_parallel"] == 8
    after = estimate(MemoryRequest(arch=DENSE_100B, gpu="H100", gpu_mem_gib=78.4, tokens=4096,
                                   tp=shape["tensor_parallel"], pp=shape["pipeline_parallel"]))
    assert after["peak_gib"] <= 0.9 * 78.4
    assert any("model parallel" in s["change"] for s in suggestions(req, 0.9))


def test_model_revision_hint_names_the_local_path_and_the_fix():
    text = _model_revision_hint({"source": "local", "requested_identifier": "/w/m"})
    assert "/w/m" in text and "private" in text and "--model-revision" in text
    assert _model_revision_hint({"source": "huggingface"}) == "pass --model-revision <40-character commit>"


def test_miles_on_ascend_warns_and_verl_does_not(capsys):
    specs = parse_gpu_spec("ssh:1x8x910b4")
    _warn_backend_device(SimpleNamespace(rl_backend="miles"), specs)
    assert "no Ascend NPU support" in capsys.readouterr().err
    _warn_backend_device(SimpleNamespace(rl_backend="verl"), specs)
    _warn_backend_device(SimpleNamespace(rl_backend="miles"), parse_gpu_spec("modal:1xl4"))
    assert capsys.readouterr().err == ""


def test_dry_run_memory_lines():
    out = io.StringIO()
    print_dry_run_memory([{"learner_id": 0, "estimated": False, "reason": "GPU '910B4' memory unknown"},
                          {"learner_id": 1, "estimated": True, "fits": True}], out=out)
    assert out.getvalue().count("\n") == 1 and "910B4" in out.getvalue()
    print_dry_run_memory(None, out=out)
