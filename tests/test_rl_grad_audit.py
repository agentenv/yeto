"""Pre-optimizer LoRA gradient audit (rl-engine-ports tier 2, design D12)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from yeto.rl import grad_audit as ga  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "rl_engine_equivalence_ga", Path(__file__).resolve().parents[1] / "scripts" / "rl_engine_equivalence.py")
eq = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = eq
_spec.loader.exec_module(eq)

ON = {ga.GRAD_AUDIT_ENV: "1"}


class Lora(torch.nn.Module):
    """Tiny stand-in for one LoRA-wrapped linear: A (r x in), B (out x r)."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.A = torch.nn.Parameter(torch.randn(2, 3, generator=g))
        self.B = torch.nn.Parameter(torch.randn(4, 2, generator=g))

    def forward(self, x):
        return x @ self.A.T @ self.B.T


class ClippingOptimizer:
    """Megatron-like: step() clips the grads in place, then applies SGD."""

    def __init__(self, params, clip: float) -> None:
        self.params, self.clip = list(params), clip

    def get_loss_scale(self):
        return torch.tensor(1.0)

    def step(self):
        norm = torch.nn.utils.clip_grad_norm_(self.params, self.clip)
        with torch.no_grad():
            for p in self.params:
                p -= 0.1 * p.grad
        return True, norm, 0


def _backward(model, seed=0):
    x = torch.randn(5, 3, generator=torch.Generator().manual_seed(seed))
    model(x).pow(2).sum().backward()


def _bindings(model, *, prefix="", transpose=False):
    to_hf = (lambda t: t.T) if transpose else (lambda t: t)
    return [ga.GradBinding(ga.canonical_grad_name(f"{prefix}layers.0.q_proj.lora_A.weight"), model.A, to_hf),
            ga.GradBinding(ga.canonical_grad_name(f"{prefix}layers.0.q_proj.lora_B.weight"), model.B, to_hf)]


def test_canonical_names_align_legacy_and_ports_spellings():
    assert ga.canonical_grad_name("model.layers.0.q_proj.lora_A.weight") == \
        ga.canonical_grad_name("base_model.model.model.layers.0.q_proj.lora_A.default.weight") == \
        "base_model.model.model.layers.0.q_proj.lora_A.weight"
    with pytest.raises(ga.GradAuditError):
        ga.canonical_grad_name("model.layers.0.q_proj.weight")
    assert ga.stem(0, 0) == "round-00000001" and ga.stem(0, 2) == "round-00000001.step-002"


def test_collect_prefers_main_grad_and_removes_loss_scale():
    m = Lora(0)
    _backward(m)
    m.A.main_grad = m.A.grad.detach() * 8  # Megatron's FP32 DDP buffer, scaled
    m.B.main_grad = m.B.grad.detach() * 8
    tensors, prov = ga.collect(_bindings(m), loss_scale=8.0)
    assert prov["grad_source"] == ["main_grad"] and prov["grad_dtype"] == ["float32"]
    assert torch.allclose(tensors["base_model.model.layers.0.q_proj.lora_A.weight"], m.A.grad)
    m2 = Lora(0)
    with pytest.raises(ga.GradAuditError):
        ga.collect(_bindings(m2))  # no gradient at all


def test_arm_captures_pre_clip_gradient_once_and_records_clip(tmp_path):
    m = Lora(1)
    _backward(m)
    raw = {"A": m.A.grad.clone(), "B": m.B.grad.clone()}
    opt = ClippingOptimizer(m.parameters(), clip=1e-3)  # forces clipping
    args = SimpleNamespace(yeto_rl_grad_audit_dir=str(tmp_path), clip_grad=1e-3)
    original = opt.step
    assert ga.arm(args, 0, 0, opt, lambda: _bindings(m), engine="ports", environ=ON, dp_world=1)
    assert ga.arm(args, 0, 0, opt, lambda: _bindings(m), engine="ports", environ=ON, dp_world=1)  # idempotent
    opt.step()
    assert opt.step == original  # one shot: restored
    assert not torch.allclose(m.A.grad, raw["A"])  # the optimizer did clip in place
    index, flat = ga.read(tmp_path / "round-00000001.grad.f32")
    t = eq.grad_tensors(index, flat)
    assert torch.allclose(torch.from_numpy(t["base_model.model.layers.0.q_proj.lora_A.weight"]),
                          raw["A"].flatten())
    norm = float(torch.cat([raw["A"].flatten(), raw["B"].flatten()]).norm())
    assert index["grad_norm_l2"] == pytest.approx(norm, rel=1e-6)
    assert index["clip_coefficient"] == pytest.approx(1e-3 / (norm + 1e-6))
    assert index["optimizer_grad_norm"] == pytest.approx(norm, rel=1e-5)
    assert index["engine"] == "ports" and index["semantics"].startswith("pre-clip")
    assert [s["name"] for s in index["specs"]] == sorted(s["name"] for s in index["specs"])


def test_arm_disabled_non_writer_and_unsupported_layouts(tmp_path):
    m = Lora(2)
    opt = ClippingOptimizer(m.parameters(), clip=1.0)
    args = SimpleNamespace(yeto_rl_grad_audit_dir=str(tmp_path), clip_grad=1.0)
    assert not ga.arm(args, 0, 0, opt, lambda: _bindings(m), engine="x", environ={})
    _backward(m)
    assert ga.arm(args, 0, 0, opt, lambda: _bindings(m), engine="x", environ=ON,
                  is_writer=lambda: False, dp_world=1)
    opt.step()
    assert not list(tmp_path.iterdir())
    with pytest.raises(ga.GradAuditError, match="tensor_model_parallel_size"):
        ga.arm(SimpleNamespace(**vars(args), tensor_model_parallel_size=2), 0, 0, opt,
               lambda: [], engine="x", environ=ON, dp_world=1)
    with pytest.raises(ga.GradAuditError, match="distributed-optimizer"):
        ga.arm(SimpleNamespace(**vars(args), use_distributed_optimizer=True), 0, 0, opt,
               lambda: [], engine="x", environ=ON, dp_world=2)
    with pytest.raises(ga.GradAuditError, match="yeto_rl_grad_audit_dir"):
        ga.arm(SimpleNamespace(), 0, 0, opt, lambda: [], engine="x", environ=ON, dp_world=1)


def test_legacy_and_ports_layouts_align_to_the_same_gradient(tmp_path):
    """Different Megatron layouts (ports stores A transposed) -> same canonical file."""

    legacy, ports = Lora(3), Lora(3)
    _backward(legacy)
    with torch.no_grad():  # ports keeps its tensors transposed; to_hf undoes it
        ports.A = torch.nn.Parameter(legacy.A.detach().T.clone())
        ports.B = torch.nn.Parameter(legacy.B.detach().T.clone())
    ports.A.grad = legacy.A.grad.T.clone()
    ports.B.grad = legacy.B.grad.T.clone()
    args = lambda d: SimpleNamespace(yeto_rl_grad_audit_dir=str(d), clip_grad=1.0)  # noqa: E731
    for d, model, prefix, t in ((tmp_path / "l", legacy, "base_model.model.", False),
                                (tmp_path / "p", ports, "", True)):
        opt = ClippingOptimizer(model.parameters(), clip=1.0)
        ga.arm(args(d), 0, 0, opt, lambda m=model, p=prefix, tt=t: _bindings(m, prefix=p, transpose=tt),
               engine=d.name, environ=ON, dp_world=1)
        opt.step()
    g = eq.grad_compare(tmp_path / "l" / "round-00000001.grad.f32", tmp_path / "p" / "round-00000001.grad.f32")
    assert g["rel_l2"] == 0.0 and g["cosine"] == pytest.approx(1.0) and g["tensors"] == 2


def test_ports_recorder_arms_from_train_one_step(tmp_path, monkeypatch):
    from yeto.rl.adapters.miles import state_plugin as sp

    m = Lora(4)
    _backward(m)
    opt = ClippingOptimizer(m.parameters(), clip=1.0)
    chunks = [m]
    monkeypatch.setenv(ga.GRAD_AUDIT_ENV, "1")
    monkeypatch.setattr(ga, "_world", lambda _name: 1)
    monkeypatch.setitem(sp._GRAD_BINDINGS, id(chunks), tuple(_bindings(m)))
    args = SimpleNamespace(yeto_rl_grad_audit_dir=str(tmp_path), clip_grad=1.0)

    def train_one_step(args, rollout_id, step_id, data_iterator, model, optimizer, opt_param_scheduler,
                       num_microbatches):
        return {}, optimizer.step()[1], None

    assert sp._arm_grad_audit(train_one_step, (args, 2, 0, None, chunks), {"optimizer": opt,
                                                                            "opt_param_scheduler": None})
    train_one_step(args, 2, 0, None, chunks, opt, None, 1)
    index = json.loads((tmp_path / "round-00000003.grad.json").read_text())
    assert index["engine"] == "ports" and index["rollout_id"] == 2
    monkeypatch.delenv(ga.GRAD_AUDIT_ENV)
    assert not sp._arm_grad_audit(train_one_step, (args, 2, 0, None, chunks), {"optimizer": opt})


def test_learner_wires_legacy_hook_and_ports_directory(tmp_path, monkeypatch):
    from yeto.rl import learner

    args = SimpleNamespace(audit_dir=str(tmp_path / "audit"))
    monkeypatch.delenv(ga.GRAD_AUDIT_ENV, raising=False)
    assert not learner._configure_grad_audit(args, SimpleNamespace(), "legacy")
    monkeypatch.setenv(ga.GRAD_AUDIT_ENV, "1")
    legacy = SimpleNamespace(custom_megatron_before_train_step_hook_path=None)
    assert learner._configure_grad_audit(args, legacy, "legacy")
    assert legacy.custom_megatron_before_train_step_hook_path == ga.HOOK_PATH
    assert legacy.yeto_rl_grad_audit_dir == str(tmp_path / "audit")
    ports = SimpleNamespace()
    assert learner._configure_grad_audit(args, ports, "ports")
    assert not hasattr(ports, "custom_megatron_before_train_step_hook_path")
    with pytest.raises(ValueError, match="conflicts"):
        learner._configure_grad_audit(
            args, SimpleNamespace(custom_megatron_before_train_step_hook_path="x.y"), "legacy")
    with pytest.raises(ValueError, match="audit_dir"):
        learner._configure_grad_audit(SimpleNamespace(audit_dir=None), SimpleNamespace(), "legacy")
