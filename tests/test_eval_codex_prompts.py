from types import SimpleNamespace

import pytest
import torch

from scripts import eval_codex_prompts as evaluation


def test_model_class_selects_multimodal_auto_model_for_qwen35(monkeypatch):
    monkeypatch.setattr(
        evaluation.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(
            model_type="qwen3_5", architectures=["Qwen3_5ForConditionalGeneration"]
        ),
    )

    assert (
        evaluation._model_class("Qwen/Qwen3.6-27B")
        is evaluation.AutoModelForImageTextToText
    )


def test_heldout_loss_uses_only_weighted_assistant_targets(monkeypatch):
    dataset = [
        (
            torch.tensor([10, 11, 12, 13]),
            torch.tensor([0.0, 1.0, 1.0, 0.0]),
        )
    ]
    seen = {}

    class Model:
        def __call__(self, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(loss=torch.tensor(2.0))

    monkeypatch.setattr(evaluation, "build_packed_dataset", lambda *a, **k: dataset)

    metrics = evaluation.heldout_loss(
        tokenizer=object(),
        model=Model(),
        data_path="eval.jsonl",
        seq_len=4,
        max_blocks=8,
        device="cpu",
    )

    assert seen["labels"].tolist() == [[-100, 11, 12, -100]]
    assert metrics["loss"] == 2.0
    assert metrics["target_tokens"] == 2
    assert metrics["blocks"] == 1
    assert metrics["perplexity"] == pytest.approx(7.389056, rel=1e-5)


def test_render_prompt_uses_training_system_message():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages[0] == {
                "role": "system",
                "content": evaluation.SYSTEM_PROMPT,
            }
            assert messages[1] == {"role": "user", "content": "prompt"}
            assert kwargs["add_generation_prompt"] is True
            assert kwargs["enable_thinking"] is False
            return "rendered"

    assert evaluation.render_prompt(Tokenizer(), "prompt") == "rendered"
