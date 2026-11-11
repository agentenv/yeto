"""yeto/dual_clipfrac: the fork's dual_clipfrac metric (ports-v1 2be012421), which the
Miles k31/b8 port drops, reported by the yeto state plugin instead.

The bitwise check runs the real k31/b8 loss (``policy_loss_function`` on the
loss/hub layout) with the yeto wrappers installed, in a subprocess so the Miles
checkout and the stubs of its absent heavy dependencies never leak into this
process. The Miles checkout comes from ``YETO_TEST_MILES_K31_B8`` and must be at
``pins.MILES_K31_B8_COMMIT``; without it (unset or no checkout there), or without
torch, the test is skipped.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from yeto.rl.adapters.miles import pins

torch = pytest.importorskip("torch")

from yeto.rl.adapters.miles import state_plugin as sp  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
# The fork's reported dual_clipfrac for this input (Miles 28bc6cc24 loss snapshot
# dual_clip_b3: grpo, policy_loss, eps_clip_c 3.0, seed 42, responses 10/48/32).
FORK_DUAL_CLIPFRAC = 0.03125

_SCRIPT = r'''
import importlib.abc, importlib.machinery, importlib.util, json, sys, tempfile, types
from unittest import mock

b8, repo = sys.argv[1], sys.argv[2]
sys.path[:0] = [b8, repo]

HEAVY = ("ray", "megatron", "sglang", "wandb", "pybase64", "einops", "transformer_engine", "mbridge",
         "omegaconf", "flash_attn", "apex", "deep_ep", "tensorboard", "sglang_router", "zmq", "pylatexenc",
         "math_verify", "polars", "uvloop", "aiohttp", "nvtx", "pynvml", "torch_memory_saver", "flashinfer",
         "datasets")
MISSING = {name for name in HEAVY if importlib.util.find_spec(name) is None}


def _any_schema(cls, source, handler):
    from pydantic_core import core_schema
    return core_schema.any_schema()


class _StubMeta(type):
    def __getattr__(cls, item):
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        return mock.MagicMock(name=f"{cls.__name__}.{item}")


class _StubModule(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__path__, self.__file__ = [], f"<stub {name}>"

    def __getattr__(self, item):
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        value = _StubMeta(item, (mock.MagicMock,), {"__get_pydantic_core_schema__": classmethod(_any_schema)})
        setattr(self, item, value)
        return value


class _Stubs(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Only the heavy dependencies this interpreter lacks; none of them is on the loss path."""

    def find_spec(self, fullname, path, target=None):
        if fullname.split(".")[0] in MISSING:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, _Stubs())

import torch
import torch.distributed as dist

from miles.backends.training_utils.data.context_parallel import get_sum_of_sample_mean
from miles.backends.training_utils.loss.hub import losses
from miles.backends.training_utils.loss.hub.math_utils import _safe_exp_neg_ppo_kl
from tests.fast.backends.training_utils.loss import loss_test_utils as U

dist.init_process_group("gloo", init_method=f"file://{tempfile.mkdtemp()}/pg", rank=0, world_size=1)
CASES = {
    "dual_clip_b3": dict(eps_clip_c=3.0),
    "dual_clip_tis_b3": dict(eps_clip_c=3.0, use_tis=True, get_mismatch_metrics=True),
    "no_dual_clip_b3": dict(),
}


def run(overrides):
    args = U.make_args(global_batch_size=3, advantage_estimator="grpo", loss_type="policy_loss", **overrides)
    U.make_parallel_state()
    inputs = U.make_inputs(seed=42, batch_size=3, prompt_lens=[20, 64, 40], response_lens=[10, 48, 32],
                           vocab_size=128, args=args)
    batch = U.make_batch(inputs, "policy_loss")
    logits = U.deep_clone(inputs["policy_logits"]).requires_grad_(True)
    som = get_sum_of_sample_mean(batch["total_lengths"], batch["response_lengths"], batch["loss_masks"],
                                 args.backend.calculate_per_token_loss, args.qkv_format, batch.get("max_seq_lens"))
    loss, metrics = losses.policy_loss_function(args, batch, logits, som)
    loss.backward()
    return args, batch, som, loss.detach(), {k: v.detach() for k, v in metrics.items()}, logits.grad.clone()


def reference(overrides):
    """The fork's definition, computed apart from yeto: compute_dual_clipfrac's ops on
    the inputs of compute_policy_loss, reduced like the fork reduced it (its final
    sum_of_sample_mean: rebuilt with the TIS / RS masks when TIS is on)."""
    seen = {}
    cpl, tis = losses.compute_policy_loss, losses.vanilla_tis_function

    def capture_cpl(ppo_kl, advantages, eps_clip, eps_clip_high, eps_clip_c=None):
        seen["inputs"] = (ppo_kl.detach().clone(), advantages.detach().clone(), eps_clip, eps_clip_high, eps_clip_c)
        return cpl(ppo_kl, advantages, eps_clip, eps_clip_high, eps_clip_c)

    def capture_tis(**kwargs):
        out = tis(**kwargs)
        seen["masks"] = out[1]
        return out

    losses.compute_policy_loss, losses.vanilla_tis_function = capture_cpl, capture_tis
    try:
        args, batch, som, loss, metrics, grad = run(overrides)
    finally:
        losses.compute_policy_loss, losses.vanilla_tis_function = cpl, tis
    ppo_kl, adv, eps, eps_high, c = seen["inputs"]
    if c is None:
        return None, (loss, metrics, grad)
    with torch.no_grad():  # fork 2be012421 math_utils.compute_dual_clipfrac
        ratio = _safe_exp_neg_ppo_kl(ppo_kl)
        pg_losses1 = -ratio * adv
        pg_losses2 = -ratio.clamp(1 - eps, 1 + eps_high) * adv
        clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
        pg_losses3 = -c * adv
        indicator = ((adv < 0) & (pg_losses3 < clip_pg_losses1)).float()
    if "masks" in seen:
        som = get_sum_of_sample_mean(batch["total_lengths"], batch["response_lengths"], seen["masks"],
                                     args.backend.calculate_per_token_loss, args.qkv_format,
                                     batch.get("max_seq_lens"), denominators=batch.get("rollout_mask_sums"))
    return som(indicator).clone().detach(), (loss, metrics, grad)


def bits(t):
    return t.detach().float().reshape(-1).view(torch.int32).tolist()


plain = {name: reference(case) for name, case in CASES.items()}
from yeto.rl.adapters.miles import state_plugin

installed = state_plugin.install_dual_clipfrac_recorder()
out = {"installed": installed, "stubbed": sorted(MISSING), "cases": {}}
for name, case in CASES.items():
    want, (loss0, metrics0, grad0) = plain[name]
    _, _, _, loss, metrics, grad = run(case)
    got = metrics.pop(state_plugin.DUAL_CLIPFRAC_KEY, None)
    out["cases"][name] = {
        "yeto": None if got is None else got.item(),
        "yeto_bits": None if got is None else bits(got),
        "fork_definition": None if want is None else want.item(),
        "fork_definition_bits": None if want is None else bits(want),
        "dtype": None if got is None else str(got.dtype),
        "loss_bitwise_unchanged": torch.equal(loss, loss0),
        "grad_bitwise_unchanged": torch.equal(grad, grad0),
        "other_metrics_bitwise_unchanged": sorted(metrics) == sorted(metrics0)
        and all(torch.equal(metrics[k], metrics0[k]) for k in metrics0),
    }
print("RESULT " + json.dumps(out))
'''


def _miles_b8() -> Path:
    value = os.environ.get("YETO_TEST_MILES_K31_B8")
    if not value:
        pytest.skip("set YETO_TEST_MILES_K31_B8 to a Miles k31/b8 checkout")
    path = Path(value).expanduser()
    if not (path / "miles" / "backends" / "training_utils" / "loss" / "hub" / "losses.py").is_file():
        pytest.skip(f"no Miles k31/b8 checkout at {path}")
    try:
        head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"cannot read the Miles checkout's commit: {exc!r}")
    if head != pins.MILES_K31_B8_COMMIT:
        pytest.skip(f"Miles checkout at {head[:9]}, not k31/b8 {pins.MILES_K31_B8_COMMIT[:9]}")
    return path


def test_yeto_dual_clipfrac_is_bitwise_the_fork_metric_on_k31_b8():
    b8 = _miles_b8()
    proc = subprocess.run([sys.executable, "-I", "-c", _SCRIPT, str(b8), str(REPO)],
                          capture_output=True, text=True, timeout=900)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stderr[-4000:]
    out = json.loads(lines[-1][len("RESULT "):])
    assert out["installed"] is True
    dual = out["cases"]["dual_clip_b3"]
    # the fork's own number for this input, bit for bit (0.03125 = 2**-5 in float32)
    assert dual["dtype"] == "torch.float32"
    assert dual["yeto"] == FORK_DUAL_CLIPFRAC
    assert dual["yeto_bits"] == torch.tensor([FORK_DUAL_CLIPFRAC]).view(torch.int32).tolist()
    for name in ("dual_clip_b3", "dual_clip_tis_b3"):
        case = out["cases"][name]
        assert case["yeto_bits"] is not None and case["yeto_bits"] == case["fork_definition_bits"], (name, case)
    # only when dual-clip is on
    assert out["cases"]["no_dual_clip_b3"]["yeto"] is None
    # observation only: loss, gradient and every other metric bitwise unchanged
    for name, case in out["cases"].items():
        assert case["loss_bitwise_unchanged"] and case["grad_bitwise_unchanged"], name
        assert case["other_metrics_bitwise_unchanged"], name


# -- CPU, fake Miles: wiring and the pin gate ---------------------------------------


@pytest.fixture
def fake_loss_hub(monkeypatch):
    """A fake loss/hub: policy_loss_function calls compute_policy_loss and
    response_layout through the module globals, as the k31 loss does."""
    losses = types.ModuleType("miles.backends.training_utils.loss.hub.losses")
    math_utils = types.ModuleType("miles.backends.training_utils.loss.hub.math_utils")
    math_utils._safe_exp_neg_ppo_kl = lambda ppo_kl: torch.exp(-ppo_kl)

    def compute_policy_loss(ppo_kl, clean_advantages, eps_clip, eps_clip_high, eps_clip_c=None):
        return -torch.exp(-ppo_kl) * clean_advantages, torch.zeros_like(ppo_kl)

    class Layout:
        def __init__(self, masks):
            self.masks = masks

        def sum_of_sample_mean(self, x):
            return (x * self.masks).sum() / self.masks.sum()

        def reducer(self, masks):
            return lambda x: (x * masks).sum() / self.masks.sum()

    def response_layout(args, batch, logits, sum_of_sample_mean):
        return Layout(batch["mask"])

    def policy_loss_function(args, batch, logits, sum_of_sample_mean):
        layout = losses.response_layout(args, batch, logits, sum_of_sample_mean)
        reduce = layout.sum_of_sample_mean
        pg, clipfrac = losses.compute_policy_loss(batch["ppo_kl"], batch["adv"], args.eps_clip,
                                                  args.eps_clip_high, args.eps_clip_c)
        if getattr(args, "use_tis", False):
            reduce = layout.reducer(batch["rs_mask"])
        return reduce(pg), {"pg_loss": reduce(pg).detach(), "pg_clipfrac": reduce(clipfrac)}

    losses.compute_policy_loss, losses.response_layout = compute_policy_loss, response_layout
    losses.policy_loss_function = policy_loss_function
    names = ("miles", "miles.backends", "miles.backends.training_utils", "miles.backends.training_utils.loss",
             "miles.backends.training_utils.loss.hub")
    for name in names:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, losses.__name__, losses)
    monkeypatch.setitem(sys.modules, math_utils.__name__, math_utils)
    monkeypatch.setattr(sp, "_DUAL_CLIPFRAC_INSTALLED", False)
    return losses


def _batch():
    # ratios 4, 2, 4, 0.5; A -1, -1, 1, -1; c = 3: only token 0 takes the dual bound
    return {"ppo_kl": -torch.log(torch.tensor([4.0, 2.0, 4.0, 0.5])), "adv": torch.tensor([-1.0, -1.0, 1.0, -1.0]),
            "mask": torch.tensor([1.0, 1.0, 1.0, 1.0]), "rs_mask": torch.tensor([0.0, 1.0, 1.0, 1.0])}


def test_reported_only_with_dual_clip_and_reduced_like_the_loss(fake_loss_hub):
    from types import SimpleNamespace

    assert sp.install_dual_clipfrac_recorder() and sp.install_dual_clipfrac_recorder()  # idempotent
    args = SimpleNamespace(eps_clip=0.2, eps_clip_high=0.2, eps_clip_c=3.0)
    _, metrics = fake_loss_hub.policy_loss_function(args, _batch(), None, None)
    assert metrics[sp.DUAL_CLIPFRAC_KEY].item() == 0.25
    # TIS / RS: the reducer the loss rebuilt with the modified masks (token 0 dropped)
    _, metrics = fake_loss_hub.policy_loss_function(SimpleNamespace(**vars(args), use_tis=True), _batch(), None, None)
    assert metrics[sp.DUAL_CLIPFRAC_KEY].item() == 0.0
    _, metrics = fake_loss_hub.policy_loss_function(SimpleNamespace(eps_clip=0.2, eps_clip_high=0.2, eps_clip_c=None),
                                                    _batch(), None, None)
    assert sp.DUAL_CLIPFRAC_KEY not in metrics
    # the layout view is handed out only inside a dual-clip call
    assert type(fake_loss_hub.response_layout(args, _batch(), None, None)).__name__ == "Layout"


def test_installed_by_the_step_recorder_on_k31_pins_only(monkeypatch):
    calls = []
    monkeypatch.setattr(sp, "install_dual_clipfrac_recorder", lambda: calls.append(1) or True)
    model = types.ModuleType("miles.backends.megatron_utils.model")
    model.train_one_step = lambda *a, **k: ({}, 0.0)
    for name in ("miles", "miles.backends", "miles.backends.megatron_utils"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["miles.backends.megatron_utils"].model = model
    monkeypatch.setitem(sys.modules, model.__name__, model)
    monkeypatch.delenv("YETO_POLICY_METRICS", raising=False)
    for commit, expected in ((pins.MILES_K31_B8_COMMIT, [1]), (pins.MILES_PORTS_V1_COMMIT, [])):
        calls.clear()
        monkeypatch.setattr(sp, "_RECORDER_INSTALLED", False)
        monkeypatch.setattr(pins, "MILES_NEXT_COMMIT", commit)
        assert sp.install_grad_norm_recorder()
        assert calls == expected, commit  # ports-v1's fork reports dual_clipfrac itself
