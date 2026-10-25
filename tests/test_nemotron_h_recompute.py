"""yeto.rl.nemotron_h_recompute: the recompute hook this island installs on
transformers' ``NemotronHBlock`` (CPU, a stand-in block class).

The checks that matter for the island (S27 G-EQ4):
1. the wrapped forward gives the same values and the same gradients;
2. it really recomputes -- the block's forward runs a second time during the
   backward -- and the graph it leaves behind holds fewer nodes;
3. the module tree and ``state_dict`` keys do not change -- the full-parameter
   export reads those names;
4. an eval / no-grad / cache-filling forward is not checkpointed;
5. the device-family gate and the env override.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from yeto.rl import nemotron_h_recompute as rc  # noqa: E402


class _Block(torch.nn.Module):
    """Stands in for ``NemotronHBlock``: the same call shape (hidden states in,
    hidden states out, keyword arguments passed through), and it counts its calls."""

    calls: int = 0

    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.norm = torch.nn.Linear(width, width)
        self.mixer = torch.nn.Linear(width, width)

    def forward(self, hidden_states, attention_mask=None, use_cache=False, past_key_values=None, **kwargs):
        type(self).calls += 1
        out = torch.tanh(self.norm(hidden_states))
        out = torch.tanh(self.mixer(out))
        if past_key_values is not None:
            past_key_values.append(out.detach())
        return out + hidden_states


def _saved_elements(call):
    """Run ``call`` and return (result, how many tensor elements the forward kept
    for the backward). This is the quantity the 64 GiB card ran out of."""
    kept: list[int] = []

    def pack(tensor):
        kept.append(tensor.numel())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = call()
    return out, sum(kept)


def _fresh():
    """A fresh stand-in module and block class, so each test patches its own.

    The module is a namespace, not a class: ``apply`` reads ``module.__name__``,
    and a class's ``__name__`` is always its own class name.
    """
    from types import SimpleNamespace

    class Block(_Block):
        calls = 0

    return SimpleNamespace(NemotronHBlock=Block, __name__="test.modeling_nemotron_h"), Block


def test_same_values_and_gradients_and_it_recomputes():
    module, block_cls = _fresh()
    torch.manual_seed(0)
    block = block_cls().double()
    x = torch.randn(2, 5, 16, dtype=torch.double, requires_grad=True)

    block.train()
    plain, plain_kept = _saved_elements(lambda: block(x))
    plain.sum().backward()
    plain_calls = block_cls.calls  # 1: a plain forward is never run again
    plain_grad = x.grad.clone()
    x.grad = None

    info = rc.apply(module=module, enabled=True)
    assert info == {"already": False, "enabled": True, "module": "test.modeling_nemotron_h",
                    "cls": "NemotronHBlock", "use_reentrant": False}
    block_cls.calls = 0
    checkpointed, checkpointed_kept = _saved_elements(lambda: block(x))
    assert block_cls.calls == 1  # not yet recomputed
    checkpointed.sum().backward()

    assert torch.allclose(plain, checkpointed)
    assert torch.allclose(plain_grad, x.grad)
    assert plain_calls == 1
    assert block_cls.calls == 2  # the backward ran the block again: that is the recompute
    # and the forward kept fewer activations: that is the memory this island needed
    assert checkpointed_kept < plain_kept

    assert rc.apply(module=module, enabled=True)["already"] is True  # idempotent


def test_state_dict_keys_and_module_tree_are_unchanged():
    """The full-parameter export's rename table reads these names (module doc)."""
    module, block_cls = _fresh()
    block = block_cls()
    before = list(block.state_dict().keys()), [n for n, _ in block.named_modules()]
    rc.apply(module=module, enabled=True)
    after = list(block.state_dict().keys()), [n for n, _ in block.named_modules()]
    assert before == after
    assert before[0] == ["norm.weight", "norm.bias", "mixer.weight", "mixer.bias"]


@pytest.mark.parametrize("case", ["eval", "no_grad", "cache_flag", "cache_object"])
def test_a_forward_that_must_not_run_twice_is_not_checkpointed(case):
    module, block_cls = _fresh()
    block = block_cls().double()
    rc.apply(module=module, enabled=True)
    x = torch.randn(2, 5, 16, dtype=torch.double, requires_grad=True)
    block.train()
    cache: list = []

    if case == "eval":
        block.eval()
        out = block(x)
    elif case == "no_grad":
        with torch.no_grad():
            out = block(x)
    elif case == "cache_flag":
        out = block(x, use_cache=True)
    else:
        out = block(x, past_key_values=cache)

    if out.requires_grad:
        out.sum().backward()
    assert block_cls.calls == 1  # ran once, and the backward did not run it again
    if case == "cache_object":
        assert len(cache) == 1  # appended once, not twice


def test_the_gate_follows_the_device_family_and_the_env_override(monkeypatch):
    monkeypatch.delenv(rc.ENV, raising=False)
    assert rc.wanted(family="ascend") is True
    assert rc.wanted(family="nvidia") is False
    monkeypatch.setenv(rc.ENV, "1")
    assert rc.wanted(family="nvidia") is True
    monkeypatch.setenv(rc.ENV, "0")
    assert rc.wanted(family="ascend") is False
    # a disabled call changes nothing and says why
    module, block_cls = _fresh()
    before = block_cls.forward
    out = rc.apply(module=module)
    assert out["enabled"] is False and "YETO_NEMOTRONH_RECOMPUTE" in out["reason"]
    assert block_cls.forward is before
