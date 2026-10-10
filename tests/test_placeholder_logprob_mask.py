"""S19 #8: stop tokens with an exact 0.0 placeholder generation logprob get loss_mask 0
and are counted (payload -> handle -> rl_round_labels)."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from yeto.rl.adapters.miles import rollout_meta_hook as hook
from yeto.rl.adapters.miles.rollout import handle_from_metadata
from yeto.rl.engine.driver import IslandDriver
from yeto.rl.harness import placeholder_logprob_mask as plm

from tests.test_rl_miles_adapter_rollout import H, group

EOT, IM_END, PAD_UNK = 248044, 248046, 0


class Tok:
    eos_token_id = IM_END
    pad_token_id = EOT
    unk_token_id = PAD_UNK
    vocab = {"<|endoftext|>": EOT, "<|im_end|>": IM_END}

    def convert_tokens_to_ids(self, text):
        return self.vocab.get(text, self.unk_token_id)


def _sample():
    """prompt [1 2] + response: a EOT(0.0, grammar-forced) obs obs b EOT(-0.3) c IM_END(0.0)."""
    response = [5, EOT, 7, 7, 6, EOT, 8, IM_END]
    return SimpleNamespace(
        tokens=[1, 2] + response,
        response_length=len(response),
        loss_mask=[1, 1, 0, 0, 1, 1, 1, 1],
        rollout_log_probs=[0.0, 0.0, 0.0, 0.0, -0.2, -0.3, -0.1, 0.0],
        metadata={},
    )


def test_stop_token_ids_from_tokenizer():
    assert plm.stop_token_ids(Tok()) == {EOT, IM_END}


def test_masks_only_stop_tokens_with_exact_zero_logprob():
    s = _sample()
    changed = plm.mask_placeholder_in_sample(s, frozenset({EOT, IM_END}))
    # index 1 (EOT, 0.0) and 7 (IM_END, 0.0) masked; non-stop 0.0 at index 0 kept;
    # observation (mask 0) untouched; EOT with a real logprob (-0.3) kept.
    assert changed == 2
    assert s.loss_mask == [1, 0, 0, 0, 1, 1, 1, 0]
    assert s.rollout_log_probs[1] == 0.0 and len(s.tokens) == 10  # nothing else changed


def test_apply_to_groups_counts_and_is_idempotent():
    s = _sample()
    args = SimpleNamespace()
    assert plm.apply_to_groups([[s]], tokenizer_loader=lambda a: Tok(), args=args) == 2
    assert s.metadata[plm.COUNT_KEY] == 2
    assert plm.apply_to_groups([[s]], tokenizer_loader=lambda a: Tok(), args=args) == 0
    assert s.metadata[plm.COUNT_KEY] == 2


def test_no_candidate_does_not_load_tokenizer():
    s = _sample()
    s.rollout_log_probs = [-0.1] * s.response_length

    def boom(_args):
        raise AssertionError("tokenizer must not load")

    assert plm.apply_to_groups([[s]], tokenizer_loader=boom, args=SimpleNamespace()) == 0
    assert plm.COUNT_KEY not in s.metadata


def test_length_mismatch_raises():
    s = _sample()
    s.rollout_log_probs = s.rollout_log_probs[:-1]
    with pytest.raises(ValueError):
        plm.mask_placeholder_in_sample(s, frozenset({EOT}))


@pytest.mark.parametrize("forced", [True, False])
def test_hook_payload_and_handle(tmp_path, monkeypatch, forced):
    kept = [group(0, [1.0, 0.0])]
    for s in kept[0]:
        s.response_length = 3
        s.tokens = [1, 9, EOT, 9]
        s.loss_mask = [1, 1, 1]
        s.rollout_log_probs = [-0.5, 0.0 if forced else -2.0, -0.5]
    monkeypatch.setattr(hook, "_load_miles_tokenizer", lambda a: Tok())
    args = SimpleNamespace()
    hook.record_trained_groups(args, kept)
    os.environ[hook.META_SINK_ENV] = f"dir:{tmp_path}"
    try:
        hook.extract_rollout_metadata(args, kept, SimpleNamespace(sample_group_index=6))
    finally:
        os.environ.pop(hook.META_SINK_ENV)
    payload = json.loads((tmp_path / "rollout-3.json").read_text())
    h = handle_from_metadata(payload, rollout_id=3, policy_version=3, policy_hash=H, data_pack=None)
    if forced:
        assert [s.loss_mask for s in kept[0]] == [[1, 0, 1], [1, 0, 1]]
        assert payload[plm.COUNT_KEY] == 2 and h.placeholder_logprob_tokens == 2
    else:
        assert [s.loss_mask for s in kept[0]] == [[1, 1, 1], [1, 1, 1]]
        assert plm.COUNT_KEY not in payload and h.placeholder_logprob_tokens is None
    assert getattr(args, hook._PLACEHOLDER_ATTR) is None  # reset per rollout


@pytest.mark.parametrize("value", [3, None])
def test_driver_round_labels(value):
    events = []
    stub = SimpleNamespace(
        emit=lambda name, **kw: events.append((name, kw)), clock=lambda: 1.0,
        _labels=lambda: {}, trainer=SimpleNamespace())
    batch = SimpleNamespace(groups=(), aborted=0, filtered=None, carried_over=None,
                            placeholder_logprob_tokens=value)
    metrics = SimpleNamespace(clip_fraction=0.0, mean_kl=0.0, ess_ratio=1.0)
    IslandDriver._emit_round_labels(stub, 0, batch, metrics)
    (name, kw), = events
    assert name == "rl_round_labels"
    if value is None:
        assert "rl/placeholder_logprob_tokens" not in kw
    else:
        assert kw["rl/placeholder_logprob_tokens"] == 3
