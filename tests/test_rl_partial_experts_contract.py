"""The model-independent partial-experts contract and its DSV4 delegation."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from yeto.rl.adapters.miles.models.partial_experts import attest, config, export, freeze, hooks, optimizer


# -- config -------------------------------------------------------------------


def _config(**overrides):
    values = dict(
        model_type="toy", num_routed_experts=64, num_moe_layers=4,
        trainable_strategy=config.IN_PLACE_PERMUTED, ep_size_default=4,
        allow_distributed_optimizer=True, requires_complete_master=False,
        trainable_param_attr="partial_expert", needs_engine_patch=False,
        num_trainable_per_layer=16,
    )
    values.update(overrides)
    return config.PartialExpertModelConfig(**values)


def test_config_refuses_contradictions():
    with pytest.raises(ValueError, match="exclusive"):
        _config(allow_distributed_optimizer=True, requires_complete_master=True)
    with pytest.raises(ValueError, match="never patches"):
        _config(needs_engine_patch=True)
    with pytest.raises(ValueError, match="one of"):
        _config(trainable_strategy="mystery")


def test_config_ep_check():
    cfg = _config()
    assert cfg.check_ep_size(4) == 4
    with pytest.raises(ValueError, match="divide 64"):
        cfg.check_ep_size(3)
    with pytest.raises(ValueError, match="trainable"):
        cfg.check_ep_size(32)


def test_registry_knows_dsv4_and_k31():
    known = config.known_model_configs()
    assert known["deepseek_v4"].trainable_strategy == config.CLONE_PREFIX
    assert known["deepseek_v4"].allow_distributed_optimizer is False
    assert known["kimi_k31"].trainable_strategy == config.IN_PLACE_PERMUTED
    assert known["kimi_k31"].allow_distributed_optimizer is True
    assert config.model_config("kimi_k31").num_trainable_per_layer == 128
    with pytest.raises(KeyError):
        config.model_config("nope")


# -- attest -------------------------------------------------------------------


def test_attest_file_and_canonical(tmp_path):
    path = tmp_path / "s.json"
    path.write_bytes(b'{"a": 1}')
    digest = attest.sha256_file(path)
    assert attest.verify_file_sha256(path, digest) == digest
    with pytest.raises(ValueError, match="mismatch"):
        attest.verify_file_sha256(path, "0" * 64)
    with pytest.raises(ValueError, match="lowercase SHA256"):
        attest.require_sha256("xyz", "thing")
    assert attest.canonical_sha256({"b": 1, "a": [1, 2]}) == attest.canonical_sha256({"a": [1, 2], "b": 1})


# -- hooks --------------------------------------------------------------------


def test_install_or_defer_patches_now_or_on_import(monkeypatch):
    name = "yeto_test_partial_experts_hook_target"
    seen = []
    module = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    hooks.install_or_defer(name, lambda m: seen.append(("now", m)))
    assert seen == [("now", module)]

    deferred = "yeto_test_partial_experts_hook_missing"
    monkeypatch.delitem(sys.modules, deferred, raising=False)
    hooks.install_or_defer(deferred, lambda m: seen.append(("later", m)))
    assert deferred in hooks.armed_module_names()
    finder = next(f for f in hooks._FINDERS if f.fullname == deferred)
    try:
        assert finder.find_spec("something.else") is None
        assert finder.find_spec(deferred) is None  # not importable: no spec, hook stays armed
    finally:
        sys.meta_path.remove(finder)
        hooks._FINDERS.remove(finder)


def test_install_once_marker_is_namespaced():
    module = ModuleType("m")
    calls = []

    def _patch(mod):
        calls.append(mod)

    hooks.install_once(module, _patch, namespace="a")
    hooks.install_once(module, _patch, namespace="a")
    hooks.install_once(module, _patch, namespace="b")
    assert len(calls) == 2
    assert module._a__patch__installed and module._b__patch__installed


# -- freeze -------------------------------------------------------------------


class _Chunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Module()
        self.decoder.experts = torch.nn.Module()
        self.decoder.experts.weight0 = torch.nn.Parameter(torch.zeros(2, 2))
        self.decoder.experts.weight1 = torch.nn.Parameter(torch.zeros(2, 2))
        self.decoder.router = torch.nn.Linear(2, 2)


def test_apply_expert_freeze_marks_and_refuses_double_configuration():
    chunk = _Chunk()

    def classify(name, _parameter):
        if ".experts.weight" not in name:
            return None
        return freeze.ExpertParamDecision(name.endswith("weight1"), {"tag": name[-1]})

    records = freeze.apply_expert_freeze([chunk], classify, configured_attr="cfg", trainable_attr="trainable")
    assert [(r.parameter_name.split(".")[-1], r.trainable) for r in records] == [("weight0", False), ("weight1", True)]
    assert chunk.decoder.experts.weight0.requires_grad is False
    assert chunk.decoder.experts.weight1.requires_grad is True
    assert chunk.decoder.experts.weight1.trainable is True and chunk.decoder.experts.weight1.tag == "1"
    assert not hasattr(chunk.decoder.experts.weight0, "trainable")
    assert chunk.decoder.router.weight.requires_grad is True  # untouched
    with pytest.raises(RuntimeError, match="configured twice"):
        freeze.apply_expert_freeze([chunk], classify, configured_attr="cfg", trainable_attr="trainable")


def test_freeze_all_but_fragment():
    chunk = _Chunk()
    trainable, frozen = freeze.freeze_all_but_fragment(chunk, ".experts.", trainable_attr="partial_expert")
    assert (trainable, frozen) == (2, 2)
    assert chunk.decoder.router.weight.requires_grad is False
    assert chunk.decoder.experts.weight0.partial_expert is True
    with pytest.raises(RuntimeError, match="nothing would train"):
        freeze.freeze_all_but_fragment(chunk, ".absent.", trainable_attr="x")


# -- optimizer ----------------------------------------------------------------


def _fixture(*, sharded=False, owned=True, complete=True, dtype=torch.float32):
    parameter = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
    main = torch.nn.Parameter(torch.zeros(4 if complete else 2, dtype=dtype))
    parameter.main_param = main
    parameter.main_param_sharded = sharded
    chunk = torch.nn.Module()
    chunk.register_parameter("weight", parameter)
    held = main if owned else torch.nn.Parameter(torch.zeros(4))
    return chunk, SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{"params": [held]}]))


def test_master_validation_complete_vs_sharded():
    chunk, opt = _fixture()
    optimizer.validate_optimizer_masters([chunk], opt, require_complete=True, label="t")
    chunk, opt = _fixture(sharded=True)
    with pytest.raises(RuntimeError, match="complete, optimizer-owned"):
        optimizer.validate_optimizer_masters([chunk], opt, require_complete=True, label="t")
    # a DP-sharded master is fine when completeness is not required ...
    optimizer.validate_optimizer_masters([chunk], opt, require_complete=False, label="t")
    chunk, opt = _fixture(complete=False)
    optimizer.validate_optimizer_masters([chunk], opt, require_complete=False, label="t")
    # ... but it must still be FP32 and the optimizer's
    chunk, opt = _fixture(owned=False)
    with pytest.raises(RuntimeError, match="optimizer-owned"):
        optimizer.validate_optimizer_masters([chunk], opt, require_complete=False, label="t")
    chunk, opt = _fixture(dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="FP32"):
        optimizer.validate_optimizer_masters([chunk], opt, require_complete=False, label="t")


def _megatron_stubs(monkeypatch):
    opt_module = ModuleType("megatron.core.optimizer")
    opt_module.get_standard_config_overrides = lambda config: {"std": True}
    cfg_module = ModuleType("megatron.core.optimizer.optimizer_config")
    cfg_module.ParamKey = lambda *, attr: ("ParamKey", attr)
    monkeypatch.setitem(sys.modules, "megatron", ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", ModuleType("megatron.core"))
    monkeypatch.setitem(sys.modules, opt_module.__name__, opt_module)
    monkeypatch.setitem(sys.modules, cfg_module.__name__, cfg_module)


def test_wrap_megatron_optimizer_respects_model_policy(monkeypatch):
    _megatron_stubs(monkeypatch)
    calls = []

    def original(**kwargs):
        calls.append(kwargs)
        return "opt"

    validated = []
    strict = optimizer.wrap_megatron_optimizer(
        original, label="t", allow_distributed_optimizer=False, param_attr="pe",
        lr=lambda: 1e-5, validate=lambda chunks, opt: validated.append((chunks, opt)),
    )
    with pytest.raises(RuntimeError, match="replicated FP32"):
        strict(config=SimpleNamespace(use_distributed_optimizer=True), model_chunks=[])
    assert strict(config=SimpleNamespace(use_distributed_optimizer=False), model_chunks=["c"]) == "opt"
    assert calls[-1]["config_overrides"] == {"std": True, ("ParamKey", "pe"): {"max_lr": 1e-5, "min_lr": 1e-5}}
    assert validated == [(["c"], "opt")]

    relaxed = optimizer.wrap_megatron_optimizer(
        original, label="t", allow_distributed_optimizer=True, param_attr="pe",
        lr=lambda: None, validate=lambda chunks, opt: None,
    )
    assert relaxed(config=SimpleNamespace(use_distributed_optimizer=True), model_chunks=[]) == "opt"
    assert calls[-1]["config_overrides"] == {"std": True}  # no LR group without an LR


# -- export -------------------------------------------------------------------


def test_filter_tasks_requires_all_names_accepted():
    tasks = [
        SimpleNamespace(mapping=SimpleNamespace(hf_param={"g": "experts.1.gate", "u": "experts.1.up"})),
        SimpleNamespace(mapping=SimpleNamespace(hf_param={"g": "experts.1.gate", "u": "experts.9.up"})),
        SimpleNamespace(mapping=SimpleNamespace(hf_param="experts.1.down")),
        SimpleNamespace(mapping=SimpleNamespace(hf_param=None)),
        None,
    ]
    selected = export.filter_tasks(tasks, lambda name: ".1." in name)
    assert selected == [tasks[0], tasks[2]]
    assert export.mapping_hf_names(tasks[3]) == ()
    assert list(export.select_named_weights([("a", 1), ("b", 2)], lambda n: n == "b")) == [("b", 2)]


# -- DSV4 delegates to the generic layer -------------------------------------


def test_dsv4_runtime_uses_generic_layer():
    from yeto.rl.adapters.miles.models.deepseek_v4 import expert_full_runtime as rt
    from yeto.rl.adapters.miles.models.deepseek_v4 import ports_runtime

    assert rt.install_or_defer is hooks.install_or_defer
    assert ports_runtime.install_or_defer is hooks.install_or_defer
    assert rt.MODEL_CONFIG is config.model_config("deepseek_v4")
    assert rt.MODEL_CONFIG.requires_complete_master and not rt.MODEL_CONFIG.allow_distributed_optimizer
    assert rt._mapping_hf_names is export.mapping_hf_names
