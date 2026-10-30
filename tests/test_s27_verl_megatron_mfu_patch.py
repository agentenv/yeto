"""B16: verl's old-log-prob pass reads an MFU entry its Megatron engine never fills.

The patch (``yeto/rl/adapters/verl/verl_megatron_patches.py``) is the only place
where yeto changes verl's behaviour instead of its own: the defect is inside verl
and no configuration of ours avoids it (S27 G-FT, VERL-MEGATRON-PP §7.13.7).

These tests stand in for verl with a module pair of the same shape -- a reader
that does ``tu.get(output, "metrics")["mfu"]`` without a default, and an engine
output whose ``metrics`` is the per-rank list of empties the Megatron engine
produces.  **The point of the file is that the path raises without the patch and
does not raise with it**: a test that only exercised the patched path could not
tell the patch was doing anything.
"""

from __future__ import annotations

import sys
import types

import pytest

from yeto.rl.adapters.verl import verl_megatron_patches as patches


# ----------------------------------------------------------------------------- a stand-in verl
def _fake_verl(monkeypatch) -> tuple[types.ModuleType, type]:
    """A ``verl`` whose reader and engine have the shapes the real ones have."""

    tensordict_utils = types.ModuleType("verl.utils.tensordict_utils")

    def get(tensordict, key, default=None):
        return tensordict.get(key, default)

    tensordict_utils.get = get

    ray_trainer = types.ModuleType("verl.trainer.ppo.ray_trainer")

    class RayPPOTrainer:
        def _compute_old_log_prob(self, output):
            # verl/trainer/ppo/ray_trainer.py:1311, copied in shape, not in text
            return tensordict_utils.get(output, "metrics")["mfu"]

        def _update_actor(self, output):
            # ray_trainer.py:1370/:1398 read the same field outside the patched call
            return tensordict_utils.get(output, "metrics")

    ray_trainer.RayPPOTrainer = RayPPOTrainer

    for name, module in (("verl", types.ModuleType("verl")),
                         ("verl.utils", types.ModuleType("verl.utils")),
                         ("verl.trainer", types.ModuleType("verl.trainer")),
                         ("verl.trainer.ppo", types.ModuleType("verl.trainer.ppo")),
                         ("verl.utils.tensordict_utils", tensordict_utils),
                         ("verl.trainer.ppo.ray_trainer", ray_trainer)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(patches, "_INSTALLED", False, raising=False)
    return tensordict_utils, RayPPOTrainer


MEGATRON_FORWARD_ONLY = {"metrics": [{}, {}, {}, {}, {}, {}, {}, {}]}  # eight ranks, each {}
FSDP_LIKE = {"metrics": {"mfu": 12.5, "flops": 3.0}}


def test_the_megatron_shaped_output_raises_without_the_patch(monkeypatch):
    """The failure run ft-2 hit, reproduced: TypeError, not a missing key."""

    _tu, trainer_cls = _fake_verl(monkeypatch)
    with pytest.raises(TypeError, match="list indices must be integers"):
        trainer_cls()._compute_old_log_prob(MEGATRON_FORWARD_ONLY)


def test_the_patch_makes_that_read_give_none(monkeypatch):
    _tu, trainer_cls = _fake_verl(monkeypatch)
    assert patches.install() is True
    assert patches.install() is False, "idempotent"
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer  # the patched attribute

    assert RayPPOTrainer()._compute_old_log_prob(MEGATRON_FORWARD_ONLY) is None


def test_the_patch_does_not_invent_an_mfu_that_was_reported(monkeypatch):
    _tu, _trainer_cls = _fake_verl(monkeypatch)
    patches.install()
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    assert RayPPOTrainer()._compute_old_log_prob(FSDP_LIKE) == 12.5


def test_the_patch_is_scoped_to_that_one_call(monkeypatch):
    """The other metrics reads in the same verl file stay the original function."""

    tensordict_utils, _trainer_cls = _fake_verl(monkeypatch)
    plain = tensordict_utils.get
    patches.install()
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    assert tensordict_utils.get is plain, "nothing is replaced outside the call"
    trainer = RayPPOTrainer()
    assert trainer._update_actor(MEGATRON_FORWARD_ONLY) == [{}] * 8, "untouched by the patch"
    assert trainer._compute_old_log_prob(MEGATRON_FORWARD_ONLY) is None
    assert tensordict_utils.get is plain, "restored after the call"
    # and a non-metrics key is never rewritten, inside the call or outside it
    assert trainer._update_actor({"other": [1, 2]}) is None


def test_the_patch_restores_the_reader_when_the_call_raises(monkeypatch):
    tensordict_utils, _trainer_cls = _fake_verl(monkeypatch)
    plain = tensordict_utils.get
    patches.install()
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    with pytest.raises(AttributeError):  # not a mapping at all: the read itself raises
        RayPPOTrainer()._compute_old_log_prob(None)
    assert tensordict_utils.get is plain


def test_a_verl_without_the_symbols_is_not_patched(monkeypatch):
    """An upstream that moved on is left alone instead of patched blindly."""

    monkeypatch.setattr(patches, "_INSTALLED", False, raising=False)
    for name in ("verl", "verl.utils", "verl.trainer", "verl.trainer.ppo",
                 "verl.utils.tensordict_utils", "verl.trainer.ppo.ray_trainer"):
        monkeypatch.setitem(sys.modules, name, None)
    assert patches.install() is False


# ----------------------------------------------------------------------------- the normalizer alone
@pytest.mark.parametrize("value,expect", [
    ([{}, {}], None),
    ([{"mfu": 7.0}, {}], 7.0),
    ([{"other": 1}], None),
    ({}, None),
    ({"mfu": 3.0}, 3.0),
])
def test_the_normalizer_keeps_a_real_mfu_and_invents_none_otherwise(value, expect):
    assert patches.normalize_metrics(value)["mfu"] == expect


def test_the_normalizer_leaves_other_missing_keys_raising():
    normalized = patches.normalize_metrics([{}])
    with pytest.raises(KeyError):
        normalized["flops"]


def test_the_normalizer_passes_through_what_is_not_a_metrics_shape():
    assert patches.normalize_metrics(None) is None
    assert patches.normalize_metrics(7) == 7


def test_the_upstream_report_names_both_verl_sites():
    report = patches.upstream_report()
    assert "ray_trainer.py:1311" in report
    assert "transformer_impl.py:1400" in report


# --------------------------------------------- S28 G-120C: old-log-prob entropy switch
def _fake_verl_with_entropy(monkeypatch):
    """A verl whose old-log-prob pass has the real one's shape around the entropy:
    assign calculate_entropy=True into the batch, call the engine, read "entropy" and
    pad it (a None raises, as no_padding_2_padding(None, ...) does), read "log_probs".
    The engine computes entropy only when asked, like _lm_head_logits_processor."""
    import torch

    tensordict_utils = types.ModuleType("verl.utils.tensordict_utils")
    tensordict_utils.get = lambda td, key, default=None: td.get(key, default)

    def assign_non_tensor(td, **fields):
        td.update(fields)

    tensordict_utils.assign_non_tensor = assign_non_tensor
    calls = {"engine_entropy": 0}

    def engine(batch):
        logits = torch.tensor([[0.1, 0.7, 0.2], [0.5, 0.2, 0.3]])
        out = {"log_probs": torch.log_softmax(logits, dim=-1)[:, 0], "metrics": [{}]}
        if batch.get("calculate_entropy"):
            calls["engine_entropy"] += 1
            p = torch.softmax(logits, dim=-1)
            out["entropy"] = -(p * p.log()).sum(-1)
        return out

    def pad(t):
        if t is None:
            raise TypeError("'NoneType' object is not subscriptable")  # no_padding_2_padding(None)
        return t.float()

    ray_trainer = types.ModuleType("verl.trainer.ppo.ray_trainer")

    class RayPPOTrainer:
        def __init__(self, config=None):
            self.config = config or {}

        def _compute_old_log_prob(self, batch):
            tensordict_utils.assign_non_tensor(batch, calculate_entropy=True, compute_loss=False)
            output = engine(batch)
            entropy = tensordict_utils.get(output, "entropy")
            log_probs = tensordict_utils.get(output, "log_probs")
            _mfu = tensordict_utils.get(output, "metrics")["mfu"]
            return {"old_log_probs": pad(log_probs), "entropys": pad(entropy)}

    ray_trainer.RayPPOTrainer = RayPPOTrainer
    for name, module in (("verl", types.ModuleType("verl")),
                         ("verl.utils", types.ModuleType("verl.utils")),
                         ("verl.trainer", types.ModuleType("verl.trainer")),
                         ("verl.trainer.ppo", types.ModuleType("verl.trainer.ppo")),
                         ("verl.utils.tensordict_utils", tensordict_utils),
                         ("verl.trainer.ppo.ray_trainer", ray_trainer)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(patches, "_INSTALLED", False, raising=False)
    monkeypatch.setattr(patches, "_ENTROPY_OFF_CALLS", 0, raising=False)
    return RayPPOTrainer, calls


OFF = {"yeto_trainer": {"old_logprob_entropy": False}}


def test_without_the_switch_a_missing_entropy_raises_where_verl_pads_it(monkeypatch):
    """The point of the placeholder: turning calculate_entropy off alone breaks verl."""
    trainer_cls, _calls = _fake_verl_with_entropy(monkeypatch)
    with pytest.raises(TypeError):
        trainer_cls(OFF)._compute_old_log_prob({"calculate_entropy": False})  # what a bare override would do


def test_entropy_off_skips_the_engine_entropy_and_keeps_the_log_probs_bit_for_bit(monkeypatch):
    trainer_cls, calls = _fake_verl_with_entropy(monkeypatch)
    assert patches.install() is True
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    on = RayPPOTrainer({})._compute_old_log_prob({})  # key absent: verl's behaviour
    assert calls["engine_entropy"] == 1
    off = RayPPOTrainer(OFF)._compute_old_log_prob({})
    assert calls["engine_entropy"] == 1, "the engine was asked for entropy with the switch off"
    assert off["old_log_probs"].equal(on["old_log_probs"])  # the log-probs do not depend on it
    assert off["entropys"].shape == on["entropys"].shape and not off["entropys"].any()
    assert on["entropys"].any()
    assert patches._ENTROPY_OFF_CALLS == 1


def test_the_switch_reads_the_resolved_config_and_defaults_to_verl(monkeypatch):
    assert patches.old_logprob_entropy(None) is True
    assert patches.old_logprob_entropy({}) is True
    assert patches.old_logprob_entropy({"yeto_trainer": {}}) is True
    assert patches.old_logprob_entropy(OFF) is False
    assert patches.old_logprob_entropy({"yeto_trainer": {"old_logprob_entropy": "False"}}) is False
    assert patches.old_logprob_entropy({"yeto_trainer": {"old_logprob_entropy": True}}) is True


def test_the_engine_writes_and_asserts_the_switch():
    from yeto.rl.adapters.verl import megatron_engine as me
    from yeto.rl.adapters.verl.island_entry import _override_value
    import dataclasses

    from tests.test_rl_verl_megatron import D0, _run  # the shared fixtures of that file

    on = me.build_overrides(_run(), D0)
    off = me.build_overrides(_run(), dataclasses.replace(D0, old_logprob_entropy=False))
    assert _override_value(on, me.OLD_LOGPROB_ENTROPY_KEY) == "True"
    assert _override_value(off, me.OLD_LOGPROB_ENTROPY_KEY) == "False"
    assert [o for o in off if me.OLD_LOGPROB_ENTROPY_KEY in o] == [f"+{me.OLD_LOGPROB_ENTROPY_KEY}=False"]
    assert me.OLD_LOGPROB_ENTROPY_KEY in me.ASSERTED_KEYS  # want is never None: a real assertion
