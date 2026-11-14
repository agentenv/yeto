"""Provenance identity of the records proxies built inside a trainer's rollout process
(Miles ``records_generate``, verl ``rollout_engine``): the device family and precision
come from the run, not a hard-coded ``cuda`` / ``bf16``. CPU only, nothing started."""
from __future__ import annotations

import pytest

from yeto.rl.adapters.miles import records_generate as rg
from yeto.rl.adapters.verl import rollout_engine as vre
from yeto.rl.harness.chat_templates import ChatTemplate
from yeto.rl.harness.gateway import proxy as proxy_mod
from yeto.rl.harness.gateway import records_env as renv

BUILDERS = {"miles": rg.records_proxy_from_env, "verl": vre.records_proxy_from_env}


@pytest.fixture
def build(monkeypatch):
    for name in ("YETO_DEVICE_FAMILY", renv.CARD_ENV, renv.PRECISION_ENV, renv.QUANT_SCHEME_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(renv.TEMPLATE_ENV, "t")
    monkeypatch.setenv(renv.TOKENIZER_ENV, "/tok")
    monkeypatch.setenv(renv.MODEL_ENV, "m")
    monkeypatch.setenv(renv.SAMPLING_ENV, "{}")
    monkeypatch.setenv(renv.MAX_MODEL_LEN_ENV, "8192")
    monkeypatch.setenv(rg.ROUTER_URL_ENV, "http://router")
    monkeypatch.setattr(ChatTemplate, "load", classmethod(lambda cls, *a, **k: object()))
    monkeypatch.setattr(proxy_mod, "RecordsProxy", lambda engine, *a, **k: engine)
    monkeypatch.setattr(renv, "_torch_device_family", lambda: None)

    def run(backend, **env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        return BUILDERS[backend]().identity
    return run


@pytest.mark.parametrize("backend", sorted(BUILDERS))
def test_rocm_trainer_is_not_labelled_cuda(build, backend):
    ident = build(backend, YETO_DEVICE_FAMILY="rocm", YETO_RECORDS_CARD="MI355X")
    assert (ident.device_family, ident.card_model) == ("rocm", "MI355X")
    assert build(backend, YETO_RECORDS_CARD="MI355X").device_family == "rocm"


@pytest.mark.parametrize("backend", sorted(BUILDERS))
def test_b300_trainer_stays_cuda_bf16(build, backend):
    ident = build(backend, YETO_DEVICE_FAMILY="cuda", YETO_RECORDS_CARD="B300")
    assert (ident.device_family, ident.card_model, ident.precision, ident.quant_scheme) == \
        ("cuda", "B300", "bf16", "bf16")
    assert build(backend, YETO_RECORDS_CARD="B300").device_family == "cuda"


@pytest.mark.parametrize("backend", sorted(BUILDERS))
def test_nothing_known_reads_unknown_not_cuda(build, backend, monkeypatch):
    assert build(backend).device_family == "unknown"
    monkeypatch.setattr(renv, "_torch_device_family", lambda: "rocm")
    assert build(backend).device_family == "rocm"


@pytest.mark.parametrize("backend", sorted(BUILDERS))
def test_precision_is_declared_not_hard_coded(build, backend):
    ident = build(backend, YETO_RECORDS_PRECISION="fp8")
    assert (ident.precision, ident.quant_scheme) == ("fp8", "fp8")
    assert build(backend, YETO_RECORDS_QUANT_SCHEME="w4a8").quant_scheme == "w4a8"


@pytest.mark.parametrize("backend", sorted(BUILDERS))
def test_cuda_family_on_a_rocm_card_reads_rocm(build, backend):
    # yeto.accel has no rocm family: a ROCm box legitimately sets YETO_DEVICE_FAMILY=cuda
    assert build(backend, YETO_DEVICE_FAMILY="CUDA", YETO_RECORDS_CARD="MI355X").device_family == "rocm"
    assert build(backend, YETO_DEVICE_FAMILY="ROCm", YETO_RECORDS_CARD="MI355X").device_family == "rocm"


@pytest.mark.parametrize("backend", sorted(BUILDERS))
def test_family_contradicting_the_card_is_refused(build, backend):
    with pytest.raises(Exception, match="MI355X"):
        build(backend, YETO_DEVICE_FAMILY="npu", YETO_RECORDS_CARD="MI355X")
