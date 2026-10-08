"""S17 WP6: "reasoning tokens do not count toward the loss" switch (CPU only).

Covers: default is a strict no-op (samples, metadata and contract hashes
bit-identical); masking rules; idempotence (only 1 -> 0, never 0 -> 1);
alignment contract still holds; the hook is installed for colocated and
disaggregated placements; and, when a Miles checkout is available, both Miles
rollout paths call the hook before the samples are converted to train data.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
from types import SimpleNamespace

import pytest

from yeto.rl import codex_backend as cb
from yeto.rl.harness import reasoning_loss_mask as rlm
from yeto.rl.harness.codex.alignment import assert_sample_alignment

OPEN, CLOSE = 900, 901
IM_START, ASSISTANT, NL, IM_END, TOOL = 10, 11, 12, 13, 14


class Tok:
    unk_token_id = 0
    vocab = {"<think>": OPEN, "</think>": CLOSE}

    def convert_tokens_to_ids(self, text):
        return self.vocab.get(text, self.unk_token_id)


def _sample():
    """Merged two-turn sample, Qwen layout: the generation prompt opens <think>.

    prompt: [IM_START ASSISTANT NL OPEN NL]
    turn 1 (generated, mask 1): r r CLOSE a a IM_END
    tool + next generation prompt (mask 0): TOOL TOOL IM_START ASSISTANT NL OPEN NL
    turn 2 (generated, mask 1): r CLOSE a IM_END
    """
    prompt = [IM_START, ASSISTANT, NL, OPEN, NL]
    turn1 = [21, 22, CLOSE, 31, 32, IM_END]
    middle = [TOOL, TOOL, IM_START, ASSISTANT, NL, OPEN, NL]
    turn2 = [23, CLOSE, 33, IM_END]
    response = turn1 + middle + turn2
    mask = [1] * len(turn1) + [0] * len(middle) + [1] * len(turn2)
    return SimpleNamespace(
        tokens=prompt + response,
        response_length=len(response),
        loss_mask=mask,
        rollout_log_probs=[-0.5] * len(response),
        metadata={"k": "v"},
        weight_versions=[],
    )


def _args(**extra):
    return SimpleNamespace(**extra)


def _exclude_args():
    return _args(yeto_rl_codex_reasoning_in_loss=False, yeto_rl_codex_reasoning_markers=["<think>", "</think>"])


def _loader(_args):
    return Tok()


# --- default: strict no-op ----------------------------------------------------

@pytest.mark.parametrize("args", [_args(), _args(yeto_rl_codex_reasoning_in_loss=True)])
def test_default_touches_nothing(args):
    data = [[_sample(), _sample()]]
    before = copy.deepcopy(data)

    def boom(_):
        raise AssertionError("default must not load a tokenizer")

    assert rlm.apply_from_args(args, data, tokenizer_loader=boom) is None
    assert [[vars(s) for s in g] for g in data] == [[vars(s) for s in g] for g in before]
    assert not hasattr(args, "_yeto_reasoning_marker_ids")


@pytest.mark.parametrize("profile", sorted(cb._PROFILES))
def test_default_contract_is_bit_identical(profile):
    old = cb.stock_codex_backend_contract(profile, 4096)
    new = cb.stock_codex_backend_contract_with_reasoning_policy(profile, 4096)
    dump = lambda c: hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()  # noqa: E731
    assert new == old and dump(new) == dump(old)


def test_record_trained_groups_default_unchanged(monkeypatch):
    from yeto.rl.engine.miles_adapter import rollout_meta_hook as hook

    monkeypatch.setattr("yeto.rl.algos.sample_filters.apply_sample_filters", lambda a, d: None)
    data = [[_sample()]]
    before = copy.deepcopy(data)
    hook.record_trained_groups(_args(), data)
    assert vars(data[0][0]) == vars(before[0][0])


# --- opt-out: masking rules ---------------------------------------------------

def test_masks_reasoning_keeps_close_answers_and_tool():
    s = _sample()
    changed = rlm.apply_from_args(_exclude_args(), [[s]], tokenizer_loader=_loader)
    # turn1: r r masked, CLOSE a a IM_END kept; middle stays 0; turn2: r masked, rest kept
    assert s.loss_mask == [0, 0, 1, 1, 1, 1] + [0] * 7 + [0, 1, 1, 1]
    assert changed == 3
    assert s.metadata == {"k": "v", "reasoning_in_loss": False, "reasoning_tokens_masked": 3}
    assert len(s.rollout_log_probs) == s.response_length  # nothing else changed
    assert_sample_alignment(s)


def test_generated_open_marker_is_masked_and_truncation_masks_to_end():
    prompt = [IM_START, ASSISTANT, NL]
    response = [OPEN, 21, 22, 23]  # model opened <think> itself and was cut off
    s = SimpleNamespace(tokens=prompt + response, response_length=4, loss_mask=[1, 1, 1, 1],
                        rollout_log_probs=[-1.0] * 4, metadata={}, weight_versions=[])
    rlm.apply_from_args(_exclude_args(), [s], tokenizer_loader=_loader)
    assert s.loss_mask == [0, 0, 0, 0]


def test_full_length_mask_layout():
    s = _sample()
    prompt_len = len(s.tokens) - s.response_length
    s.loss_mask = [0] * prompt_len + list(s.loss_mask)
    rlm.apply_from_args(_exclude_args(), [s], tokenizer_loader=_loader)
    assert s.loss_mask[prompt_len:] == [0, 0, 1, 1, 1, 1] + [0] * 7 + [0, 1, 1, 1]
    assert s.loss_mask[:prompt_len] == [0] * prompt_len


def test_idempotent_only_sets_zero():
    s = _sample()
    args = _exclude_args()
    rlm.apply_from_args(args, [[s]], tokenizer_loader=_loader)
    first = list(s.loss_mask)
    assert rlm.apply_from_args(args, [[s]], tokenizer_loader=_loader) == 0
    assert s.loss_mask == first
    assert s.metadata["reasoning_tokens_masked"] == 3
    original = _sample().loss_mask
    assert all(new <= old for new, old in zip(first, original))  # never 0 -> 1


def test_record_trained_groups_applies_policy(monkeypatch):
    from yeto.rl.engine.miles_adapter import rollout_meta_hook as hook

    monkeypatch.setattr("yeto.rl.algos.sample_filters.apply_sample_filters", lambda a, d: None)
    monkeypatch.setattr(rlm, "_load_tokenizer", _loader)
    s = _sample()
    hook.record_trained_groups(_exclude_args(), [[s]])
    assert s.loss_mask[:2] == [0, 0] and s.metadata["reasoning_in_loss"] is False


# --- fail closed --------------------------------------------------------------

def test_missing_markers_fail_closed():
    with pytest.raises(rlm.ReasoningMarkerError):
        rlm.apply_from_args(_args(yeto_rl_codex_reasoning_in_loss=False), [[_sample()]], tokenizer_loader=_loader)


def test_marker_not_single_token_fails_closed():
    args = _args(yeto_rl_codex_reasoning_in_loss=False, yeto_rl_codex_reasoning_markers=["<thinking>", "</think>"])
    with pytest.raises(rlm.ReasoningMarkerError):
        rlm.apply_from_args(args, [[_sample()]], tokenizer_loader=_loader)


def test_contract_records_opt_out_and_requires_markers():
    c = cb.stock_codex_backend_contract_with_reasoning_policy("qwen38_next", 4096, False)
    assert c["reasoning_in_loss"] is False and c["reasoning_markers"] == ["<think>", "</think>"]
    assert c != cb.stock_codex_backend_contract("qwen38_next", 4096)
    with pytest.raises(ValueError, match="no reasoning_markers"):
        cb.stock_codex_backend_contract_with_reasoning_policy("deepseekv4", 4096, False)


# --- hook installation: both placements, both Miles rollout paths -------------

@pytest.mark.parametrize("colocated", [True, False])
def test_hook_installed_for_both_placements(colocated):
    from tests.test_rl_miles_adapter_config import flag_value, make_config  # noqa: PLC0415
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.miles_adapter import config as mc

    cfg = make_config(colocated=colocated)
    if not colocated:
        import dataclasses

        cfg = dataclasses.replace(cfg, serving=dataclasses.replace(cfg.serving, offload_train=False))
    argv = list(mc.translate_run_config(cfg, AlgorithmSpec()).argv)
    assert flag_value(argv, "--rollout-sample-filter-path") == mc.TRAINED_GROUPS_HOOK_PATH


def _miles_root() -> pathlib.Path | None:
    for candidate in (os.environ.get("YETO_MILES_CHECKOUT"), "/home/michael/work/s16-raw-lora-disagg-miles",
                      "/home/michael/work/miles-next"):
        if candidate and (pathlib.Path(candidate) / "miles/rollout/sglang_rollout.py").is_file():
            return pathlib.Path(candidate)
    return None


@pytest.mark.parametrize("rel", ["miles/rollout/sglang_rollout.py",
                                 "miles/rollout/inference_rollout/inference_rollout_train.py"])
def test_miles_calls_hook_before_returning_train_samples(rel):
    root = _miles_root()
    if root is None:
        pytest.skip("no Miles checkout")
    text = (root / rel).read_text()
    hook = text.index("rollout_sample_filter_path")
    assert hook < text.index("return RolloutFnTrainOutput(samples=data", hook)
    # train-data conversion happens after the rollout function returned
    executor = (root / "miles/ray/rollout/rollout_executor.py").read_text()
    assert "_convert_samples_to_train_data" in executor or "convert_samples_to_train_data" in executor


# --- flag plumbing: CLI -> launcher -> learner argv; default argv unchanged ----
from test_rl_launcher_codex_bundle import bundle  # noqa: E402,F401  (pytest fixture)

def test_launcher_forwards_flag_only_when_set(bundle):
    import test_rl_launcher_codex_bundle as tb  # noqa: PLC0415
    from yeto.gpu_spec import parse_gpu_spec
    from yeto.launcher import _prepare_rl_args, make_miles_island_task

    args = tb._codex_args()
    spec = parse_gpu_spec(args.gpu)[0]
    default_run = make_miles_island_task(args, spec, 0, 1, "127.0.0.1:29400").run
    assert "--codex-exclude-reasoning-from-loss" not in default_run

    args.codex_exclude_reasoning_from_loss = True
    _prepare_rl_args(args)
    run = make_miles_island_task(args, spec, 0, 1, "127.0.0.1:29400").run
    assert " --codex-exclude-reasoning-from-loss" in run
    assert run.replace(" --codex-exclude-reasoning-from-loss", "") == default_run


def test_launcher_rejects_flag_without_signed_codex_agent():
    from test_rl_engine_selection import _cli  # noqa: PLC0415
    from yeto.launcher import _prepare_rl_args

    args = _cli(())
    args.codex_exclude_reasoning_from_loss = True
    with pytest.raises(ValueError, match="signed Codex agent"):
        _prepare_rl_args(args)


def test_cli_flag_defaults_off():
    from test_rl_engine_selection import _cli  # noqa: PLC0415

    assert _cli(()).codex_exclude_reasoning_from_loss is False
    assert _cli(("--codex-exclude-reasoning-from-loss",)).codex_exclude_reasoning_from_loss is True
