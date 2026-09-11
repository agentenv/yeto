"""Focused regressions for the deferred CP first-update guard.

These tests run in the pinned NeMo image.  The repository's lightweight host
test environment intentionally skips them when PyTorch is unavailable.
"""
from contextlib import contextmanager
from types import MethodType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from nemo_automodel.components.distributed.context_parallel.sharder import (
    ContextParallelSharder,
    ShardLayout,
)
from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components.models.qwen3_5 import model as qwen35_model
from nemo_automodel.recipes.vlm import finetune

from . import masked_train


class _Mesh:
    def size(self):
        return 8

    def get_local_rank(self):
        return 0

    def get_group(self):
        return None


def _rank_zero_indices(device):
    return torch.tensor([0, 15], dtype=torch.long, device=device)


def _sharder(*, aux_only, mutate_outer_input=False, corrupt_deferred_labels=False):
    mesh = _Mesh()

    @contextmanager
    def lazy_context(labels, inputs):
        indices = _rank_zero_indices(labels.device)
        local_labels = labels.index_select(1, indices)
        if corrupt_deferred_labels:
            local_labels = local_labels.clone()
            local_labels[0, 0] += 1
        labels.resize_(local_labels.shape).copy_(local_labels)
        if not aux_only:
            local_inputs = inputs.index_select(1, indices)
            inputs.resize_(local_inputs.shape).copy_(local_inputs)
        yield

    def fake_batch(cp_mesh, tp_mesh, batch, *, loss_mask=None, padding_token_id=0):
        del cp_mesh, tp_mesh, loss_mask, padding_token_id
        if mutate_outer_input:
            batch["input_ids"].add_(1)
        # Real torch CP captures buffer objects when the context is created;
        # the recipe then pops the labels mapping entry before entering it.
        labels, inputs = batch["labels"], batch["input_ids"]
        return lambda: lazy_context(labels, inputs), batch, ShardLayout(
            original_seq_len=16, padded_seq_len=16)

    fake_batch.__module__ = "nemo_automodel.components.distributed.context_parallel.sharder"
    fake_batch.__name__ = fake_batch.__qualname__ = (
        "shard_batch_aux_only" if aux_only else "shard_batch_load_balanced")
    sharder = object.__new__(ContextParallelSharder)
    sharder.shard_batch = fake_batch
    sharder.local_token_global_indices = (
        lambda cp_mesh, padded_seq_len, device: _rank_zero_indices(device))
    sharder.shard_layout = None
    sharder._cp_mesh = mesh
    sharder._tp_mesh = None
    sharder._loss_mask = None
    sharder._padding_token_id = 248044

    def fake_gather(self, tensor, seq_dim=1, trim=False, fill=None):
        del self, tensor, seq_dim
        assert trim is True
        if fill == -100:
            return torch.arange(16, dtype=torch.long).reshape(1, 16)
        if fill == 248044:
            return torch.arange(16, dtype=torch.long).reshape(1, 16)
        raise AssertionError("unexpected gather fill")

    sharder.gather_token_tensor = MethodType(fake_gather, sharder)
    return sharder, mesh


def _run_guard(tmp_path, monkeypatch, *, aux_only=True, mutate_outer_input=False,
               enter_context=True, corrupt_deferred_labels=False,
               enter_context_twice=False, run_second_after_complete=False):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n")
    run = tmp_path / "run"
    run.mkdir()
    config = {
        "checkpoint": {"checkpoint_dir": str(run / "checkpoints")},
        "dataset": {"path_or_dataset": str(manifest)},
    }
    original_primary_alias = qwen35_model.shard_sequence_for_cp_round_robin
    observed = {}

    def fake_loss(self, hidden_states, labels, lm_weight, num_label_tokens=None,
                  grad_reduce_group=None):
        del self, hidden_states, labels, lm_weight, num_label_tokens, grad_reduce_group
        return torch.tensor(0.0)

    def fake_step(trainer, batches, max_grad_norm=None):
        del max_grad_norm
        sharder, mesh = _sharder(
            aux_only=aux_only, mutate_outer_input=mutate_outer_input,
            corrupt_deferred_labels=corrupt_deferred_labels)
        context, batch = ContextParallelSharder.shard(sharder, batches[0])
        labels = batch.pop("labels")
        if enter_context:
            with context():
                if aux_only:
                    # Exercise the exact module-global function that Qwen3.5
                    # forward calls after full-sequence embedding/splicing.
                    embeds = torch.arange(16 * 4, dtype=torch.float32).reshape(1, 16, 4)
                    local, indices, padded = qwen35_model.shard_sequence_for_cp_round_robin(
                        mesh, embeds, seq_dim=1)
                    expected = _rank_zero_indices(embeds.device)
                    assert padded == 16
                    assert torch.equal(indices, expected)
                    assert torch.equal(local, embeds.index_select(1, expected))
                    observed.update(alias_called=True, indices=indices.tolist())
                FusedLinearCrossEntropy.forward(
                    None, None, labels, None, num_label_tokens=16)
            if enter_context_twice:
                with context():
                    pass
        return SimpleNamespace(metrics={"num_label_tokens": 16})

    monkeypatch.setattr(
        finetune.FinetuneRecipeForVLM, "_run_train_optim_step", fake_step)
    monkeypatch.setattr(FusedLinearCrossEntropy, "forward", fake_loss)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 8)
    monkeypatch.setattr(torch.distributed, "all_gather",
                        lambda outputs, value: [out.copy_(value) for out in outputs])
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda value: value.mul_(8))
    trainer = SimpleNamespace(
        _get_cp_group_size=lambda: 8,
        _get_dp_group_size=lambda: 1,
        dist_env=SimpleNamespace(device=torch.device("cpu"), is_main=True),
        step_scheduler=SimpleNamespace(step=1),
    )
    batch = {
        "input_ids": torch.arange(16, dtype=torch.long).reshape(1, 16),
        "labels": torch.arange(16, dtype=torch.long).reshape(1, 16),
    }
    with masked_train.live_cp_data_semantics_guard(config, "train"):
        result = finetune.FinetuneRecipeForVLM._run_train_optim_step(
            trainer, [batch], None)
        if run_second_after_complete:
            # Once update 1 has passed, every production wrapper must be a
            # permanent pass-through.  A now-invalid global size proves the
            # startup-only topology gate is no longer evaluated.
            monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
            second_batch = {
                "input_ids": torch.arange(16, dtype=torch.long).reshape(1, 16),
                "labels": torch.arange(16, dtype=torch.long).reshape(1, 16),
            }
            result = finetune.FinetuneRecipeForVLM._run_train_optim_step(
                trainer, [second_batch], None)
    assert qwen35_model.shard_sequence_for_cp_round_robin is original_primary_alias
    return result, observed, run


def test_aux_only_guard_waits_for_lazy_context_and_keeps_full_primary(tmp_path, monkeypatch):
    """This exact valid shape failed under the old eager/sharded-input assertion."""
    result, observed, run = _run_guard(tmp_path, monkeypatch, aux_only=True)
    assert result.metrics["num_label_tokens"] == 16
    assert observed == {"alias_called": True, "indices": [0, 15]}
    receipt = __import__("json").loads((run / "first-update-data-semantics.json").read_text())
    micro = receipt["microbatches"][0]
    assert micro["cp_primary_stream_mode"] == "model_owned_aux_only_full_then_in_forward_shard"
    assert micro["outer_input_ids_full_length"] is True
    assert micro["outer_input_tokens_per_rank"] == 16
    assert micro["expected_model_local_input_tokens_per_rank"] == 2
    assert micro["cp_partition_label_tokens"] == micro["full_label_tokens"] == 16


def test_aux_only_guard_rejects_outer_primary_mutation(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="changed the full primary stream"):
        _run_guard(tmp_path, monkeypatch, aux_only=True, mutate_outer_input=True)


def test_guard_rejects_context_that_is_never_entered(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="Optimizer metric does not count"):
        _run_guard(tmp_path, monkeypatch, aux_only=True, enter_context=False)


def test_guard_rejects_wrong_deferred_label_shard(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="deferred CP label shard differs"):
        _run_guard(
            tmp_path, monkeypatch, aux_only=True, corrupt_deferred_labels=True)


def test_guard_rejects_second_context_entry(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="entered more than once"):
        _run_guard(tmp_path, monkeypatch, aux_only=True, enter_context_twice=True)


def test_standard_deferred_primary_shard_remains_supported(tmp_path, monkeypatch):
    result, observed, run = _run_guard(tmp_path, monkeypatch, aux_only=False)
    assert result.metrics["num_label_tokens"] == 16
    assert observed == {}
    receipt = __import__("json").loads((run / "first-update-data-semantics.json").read_text())
    micro = receipt["microbatches"][0]
    assert micro["cp_primary_stream_mode"] == "outer_context_sharded"
    assert micro["outer_input_ids_full_length"] is False


def test_guard_is_permanent_pass_through_after_first_update(tmp_path, monkeypatch):
    result, observed, run = _run_guard(
        tmp_path, monkeypatch, aux_only=True, run_second_after_complete=True)
    assert result.metrics["num_label_tokens"] == 16
    assert observed == {"alias_called": True, "indices": [0, 15]}
    assert (run / "first-update-data-semantics.json").is_file()
