"""Unit tests for learner helpers that run without GPUs or a process group."""

import torch

from yeto.learner import allreduce_trainable_grads, normalize_param_name


# --- normalize_param_name -------------------------------------------------


def test_clean_names_pass_through():
    name = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
    assert normalize_param_name(name) == name


def test_strips_fsdp_prefix():
    assert (
        normalize_param_name("_fsdp_wrapped_module.base_model.model.lm_head.weight")
        == "base_model.model.lm_head.weight"
    )


def test_strips_nested_fsdp_prefixes():
    # Nested FSDP wrapping (auto_wrap_policy) inserts the segment at every
    # wrapped level.
    name = (
        "_fsdp_wrapped_module.base_model.model.model.layers.0."
        "_fsdp_wrapped_module.self_attn.q_proj.lora_B.default.weight"
    )
    assert (
        normalize_param_name(name)
        == "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"
    )


def test_strips_checkpoint_wrapper_prefix():
    name = (
        "_fsdp_wrapped_module._checkpoint_wrapped_module.layers.0.lora_A.default.weight"
    )
    assert normalize_param_name(name) == "layers.0.lora_A.default.weight"


def test_normalized_names_match_unwrapped_layout_names():
    # Fragment layouts are keyed by parameter name, so an fsdp-lora learner
    # must expose the exact names a ddp/single-GPU learner would.
    unwrapped = [
        "base_model.model.model.embed_tokens.weight",
        "base_model.model.model.layers.1.mlp.up_proj.lora_A.default.weight",
    ]
    wrapped = ["_fsdp_wrapped_module." + n for n in unwrapped]
    assert [normalize_param_name(n) for n in wrapped] == unwrapped


# --- allreduce_trainable_grads --------------------------------------------


def _param(grad):
    p = torch.nn.Parameter(torch.zeros(3))
    p.grad = grad
    return p


def test_allreduce_noop_when_world_is_one(monkeypatch):
    import yeto.learner as learner

    def boom(*a, **k):
        raise AssertionError("dist.all_reduce must not be called for world == 1")

    monkeypatch.setattr(learner.dist, "all_reduce", boom)
    p = _param(torch.ones(3))
    allreduce_trainable_grads([p], world=1)
    assert torch.equal(p.grad, torch.ones(3))


def test_allreduce_divides_by_world(monkeypatch):
    import yeto.learner as learner

    world = 4

    def fake_all_reduce(t, op=None):
        # Every rank holds the same grad, so SUM yields world * grad.
        t.mul_(world)

    monkeypatch.setattr(learner.dist, "all_reduce", fake_all_reduce)
    g = torch.tensor([1.0, -2.0, 0.5])
    p = _param(g.clone())
    allreduce_trainable_grads([p], world=world)
    # SUM over identical ranks then /world == the original grad (DDP mean).
    assert torch.allclose(p.grad, g)


def test_allreduce_skips_none_grads(monkeypatch):
    import yeto.learner as learner

    calls = []

    def fake_all_reduce(t, op=None):
        calls.append(t)

    monkeypatch.setattr(learner.dist, "all_reduce", fake_all_reduce)
    with_grad = _param(torch.full((3,), 2.0))
    without_grad = _param(None)
    allreduce_trainable_grads([with_grad, without_grad], world=2)
    assert len(calls) == 1
    assert without_grad.grad is None
    assert torch.allclose(with_grad.grad, torch.ones(3))  # 2.0 (sum stub is id) / 2


def test_lora_targets_resolution():
    from types import SimpleNamespace

    from yeto.learner import _ATTENTION_TARGETS, is_moe_config, resolve_lora_targets

    dense = SimpleNamespace()
    moe = SimpleNamespace(n_routed_experts=256)
    assert not is_moe_config(dense) and is_moe_config(moe)
    # auto: attention for MoE, all-linear for dense.
    assert resolve_lora_targets("auto", moe) == _ATTENTION_TARGETS
    assert resolve_lora_targets("auto", dense) == "all-linear"
    assert resolve_lora_targets("attention", dense) == _ATTENTION_TARGETS
    assert resolve_lora_targets("all-linear", moe) == "all-linear"  # warned, honored


def test_attention_target_regex_matches_common_archs():
    import re

    from yeto.learner import _ATTENTION_TARGETS

    matching = [
        "model.layers.3.self_attn.q_proj",
        "model.layers.3.self_attn.o_proj",
        "model.layers.9.self_attn.kv_a_proj_with_mqa",  # DeepSeek MLA
        "model.layers.9.self_attn.q_b_proj",
    ]
    frozen = [
        "model.layers.3.mlp.experts.17.up_proj",  # routed expert
        "model.layers.3.mlp.gate",  # router
        "lm_head",
    ]
    for name in matching:
        assert re.fullmatch(_ATTENTION_TARGETS, name), name
    for name in frozen:
        assert not re.fullmatch(_ATTENTION_TARGETS, name), name


def test_offline_first_uses_cache_hit():
    from yeto.learner import _from_pretrained_offline_first

    calls = []

    class Factory:
        @staticmethod
        def from_pretrained(model_id, **kw):
            calls.append(kw)
            if not kw.get("local_files_only"):
                raise AssertionError("went online despite cache hit")
            return "cached-model"

    assert _from_pretrained_offline_first(Factory, "org/model", trust_remote_code=True) == "cached-model"
    assert calls == [{"local_files_only": True, "trust_remote_code": True}]


def test_offline_first_falls_back_online_on_cold_cache():
    from yeto.learner import _from_pretrained_offline_first

    calls = []

    class Factory:
        @staticmethod
        def from_pretrained(model_id, **kw):
            calls.append(kw)
            if kw.get("local_files_only"):
                raise OSError("not cached")
            return "downloaded-model"

    assert _from_pretrained_offline_first(Factory, "org/model") == "downloaded-model"
    assert [c.get("local_files_only") for c in calls] == [True, None]


def test_learner_accepts_explicit_training_seed():
    from yeto.learner import parse_args

    args = parse_args(
        [
            "--model",
            "org/model",
            "--data",
            "rows.jsonl",
            "--syncer",
            "none",
            "--learner-id",
            "0",
            "--num-learners",
            "1",
            "--seed",
            "29",
        ]
    )
    assert args.seed == 29


def test_benchmark_seed_pairs_matching_global_rank():
    from yeto.learner import _derived_training_seed, _stream_seed

    root = 17
    learners = 4
    ranks_per_learner = 2
    for learner_id in range(learners):
        for rank in range(ranks_per_learner):
            baseline_rank = learner_id + learners * rank
            assert _derived_training_seed(root, learner_id, learners, rank) == (
                _derived_training_seed(root, 0, 1, baseline_rank)
            )
            diloco_stream_seed = _stream_seed(
                root, learner_id, learners, rank, workers=0
            ) + rank
            baseline_stream_seed = _stream_seed(
                root, 0, 1, baseline_rank, workers=0
            ) + baseline_rank
            assert diloco_stream_seed == baseline_stream_seed


def test_lm_row_shards_pair_with_matching_baseline_rank():
    from yeto.data import _learner_rows

    row_count = 101
    learners = 4
    ranks_per_learner = 2
    baseline_rows = _learner_rows(row_count, 0, 1, None)
    for learner_id in range(learners):
        island_rows = _learner_rows(row_count, learner_id, learners, None)
        for rank in range(ranks_per_learner):
            baseline_rank = learner_id + learners * rank
            assert island_rows[rank::ranks_per_learner] == baseline_rows[
                baseline_rank :: learners * ranks_per_learner
            ]
