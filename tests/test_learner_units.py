"""Unit tests for learner helpers that run without GPUs or a process group."""

from types import SimpleNamespace

import pytest
import torch

from yeto.learner import (
    _loss_metric_dtype,
    allreduce_trainable_grads,
    load_model_and_tokenizer,
    normalize_param_name,
    run_inner_loop,
    setup_distributed,
)
from yeto.losses import sft_loss


def test_isolated_fused_loss_is_bound_before_peft(monkeypatch):
    import peft
    import transformers
    import yeto.learner as learner

    events = []
    config = SimpleNamespace(model_type="qwen2")
    tokenizer = object()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config

        def to(self, device):
            events.append(("to", device.type))
            return self

    base_model = Model()

    def fake_from_pretrained(factory, _model_id, **_kwargs):
        if factory is transformers.AutoConfig:
            events.append("config")
            return config
        if factory is transformers.AutoTokenizer:
            events.append("tokenizer")
            return tokenizer
        assert factory is transformers.AutoModelForCausalLM
        events.append("model")
        return base_model

    monkeypatch.setattr(learner, "_from_pretrained_offline_first", fake_from_pretrained)
    monkeypatch.setattr(
        learner, "validate_kernel_request", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(learner, "attention_load_kwargs", lambda *args: {})
    monkeypatch.setattr(
        learner,
        "require_liger_model_support",
        lambda actual_config: events.append("support") or actual_config.model_type,
    )
    monkeypatch.setattr(
        learner,
        "apply_liger_fused_linear_ce",
        lambda model: events.append("bind")
        or {
            "layer_backend": "transformers-native",
            "loss_implementation": "liger-fused-linear-cross-entropy",
        },
    )
    monkeypatch.setattr(
        learner,
        "resolved_attention_backend",
        lambda model, requested: events.append("attention") or requested,
    )
    monkeypatch.setattr(learner, "resolve_lora_targets", lambda *args: "all-linear")
    monkeypatch.setattr(peft, "LoraConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        learner,
        "validate_lora_production_envelope",
        lambda model: events.append("envelope"),
    )

    def fake_get_peft_model(model, _config):
        assert model is base_model
        assert "bind" in events
        events.append("peft")
        return model

    monkeypatch.setattr(peft, "get_peft_model", fake_get_peft_model)
    args = SimpleNamespace(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        base_quantization="none",
        tuning="lora",
        shard="ddp",
        kernel_backend="liger",
        attention_backend="sdpa",
        loss_function="cross_entropy",
        lora_r=16,
        lora_alpha=32,
        lora_targets="auto",
    )

    model, loaded_tokenizer = load_model_and_tokenizer(args, torch.device("cpu"))

    assert model is base_model
    assert loaded_tokenizer is tokenizer
    assert events == [
        "config",
        "support",
        "tokenizer",
        "model",
        "bind",
        "attention",
        "peft",
        ("to", "cpu"),
        "envelope",
    ]


def test_parent_adapter_is_loaded_trainable_instead_of_reinitialized(monkeypatch):
    import peft
    import transformers
    import yeto.learner as learner

    config = SimpleNamespace(model_type="qwen2")
    tokenizer = object()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config

    base_model = Model()
    loaded = {}

    def fake_from_pretrained(factory, _model_id, **_kwargs):
        if factory is transformers.AutoTokenizer:
            return tokenizer
        assert factory is transformers.AutoModelForCausalLM
        return base_model

    monkeypatch.setattr(learner, "_from_pretrained_offline_first", fake_from_pretrained)
    monkeypatch.setattr(
        learner, "validate_kernel_request", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(learner, "attention_load_kwargs", lambda *args: {})
    monkeypatch.setattr(
        learner, "resolved_attention_backend", lambda model, requested: requested
    )
    monkeypatch.setattr(
        peft.PeftModel,
        "from_pretrained",
        lambda model, source, **kwargs: loaded.update(
            model=model, source=source, kwargs=kwargs
        )
        or model,
    )
    monkeypatch.setattr(
        peft,
        "get_peft_model",
        lambda *_args, **_kwargs: pytest.fail("new adapter must not be initialized"),
    )
    args = SimpleNamespace(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        base_quantization="none",
        tuning="lora",
        shard="ddp",
        kernel_backend="native",
        attention_backend="auto",
        loss_function="cross_entropy",
        lora_r=16,
        lora_alpha=32,
        lora_targets="auto",
        resume_from=None,
        branch_from="/tmp/parent-adapter",
    )

    model, loaded_tokenizer = load_model_and_tokenizer(args, torch.device("cpu"))

    assert model is base_model
    assert loaded_tokenizer is tokenizer
    assert loaded == {
        "model": base_model,
        "source": "/tmp/parent-adapter",
        "kwargs": {"is_trainable": True},
    }


@pytest.mark.parametrize(
    ("kernel_backend", "should_reject"),
    [("liger", True), ("native", False)],
)
def test_learner_rejects_peft_output_head_drift_only_in_fused_lane(
    monkeypatch,
    kernel_backend,
    should_reject,
):
    import peft
    import transformers
    import yeto.learner as learner

    config = SimpleNamespace(model_type="qwen2")
    tokenizer = object()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.lm_head = torch.nn.Linear(2, 2, bias=False)
            self.adapter = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))

        def get_output_embeddings(self):
            return self.lm_head

    base_model = Model()

    def fake_from_pretrained(factory, _model_id, **_kwargs):
        if factory is transformers.AutoConfig:
            return config
        if factory is transformers.AutoTokenizer:
            return tokenizer
        assert factory is transformers.AutoModelForCausalLM
        return base_model

    def fake_get_peft_model(model, _config):
        model.lm_head.requires_grad_(False)
        model.lm_head.lora_A = torch.nn.ModuleDict(
            {"default": torch.nn.Linear(2, 1, bias=False)}
        )
        return model

    monkeypatch.setattr(learner, "_from_pretrained_offline_first", fake_from_pretrained)
    monkeypatch.setattr(
        learner, "validate_kernel_request", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(learner, "attention_load_kwargs", lambda *args: {})
    monkeypatch.setattr(learner, "require_liger_model_support", lambda config: "qwen2")
    monkeypatch.setattr(
        learner,
        "apply_liger_fused_linear_ce",
        lambda model: {"loss_implementation": "fused"},
    )
    monkeypatch.setattr(
        learner,
        "resolved_attention_backend",
        lambda model, requested: requested,
    )
    monkeypatch.setattr(learner, "resolve_lora_targets", lambda *args: "all-linear")
    monkeypatch.setattr(peft, "LoraConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(peft, "get_peft_model", fake_get_peft_model)
    args = SimpleNamespace(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        base_quantization="none",
        tuning="lora",
        shard="ddp",
        kernel_backend=kernel_backend,
        attention_backend="auto",
        loss_function="cross_entropy",
        lora_r=16,
        lora_alpha=32,
        lora_targets="auto",
    )

    if should_reject:
        with pytest.raises(RuntimeError, match="frozen, unadapted lm_head"):
            load_model_and_tokenizer(args, torch.device("cpu"))
    else:
        model, loaded_tokenizer = load_model_and_tokenizer(
            args,
            torch.device("cpu"),
        )
        assert model is base_model
        assert loaded_tokenizer is tokenizer


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


def test_loss_metric_dtype_uses_hccl_supported_float32_on_npu():
    assert _loss_metric_dtype(SimpleNamespace(type="npu")) is torch.float32
    assert _loss_metric_dtype(torch.device("cuda")) is torch.float64
    assert _loss_metric_dtype(torch.device("cpu")) is torch.float64


def test_setup_distributed_uses_the_selected_device(monkeypatch):
    import yeto.learner as learner

    device = torch.device("cuda", 2)
    calls = []
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setattr(learner.accel, "dist_backend", lambda actual: "nccl")
    monkeypatch.setattr(
        learner.dist,
        "init_process_group",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(learner.dist, "get_rank", lambda: 2)
    monkeypatch.setattr(learner.dist, "get_world_size", lambda: 4)

    assert setup_distributed(device) == (2, 4)
    assert calls == [{"backend": "nccl", "device_id": device}]


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


def test_allreduce_keeps_globally_unused_grads_none(monkeypatch):
    import yeto.learner as learner

    calls = []

    def fake_all_reduce(t, op=None):
        calls.append(t)

    monkeypatch.setattr(learner.dist, "all_reduce", fake_all_reduce)
    with_grad = _param(torch.full((3,), 2.0))
    without_grad = _param(None)
    allreduce_trainable_grads([with_grad, without_grad], world=2)
    # One presence-vector reduction, then one reduction for the globally-used grad.
    assert len(calls) == 2
    assert without_grad.grad is None
    assert torch.allclose(with_grad.grad, torch.ones(3))  # 2.0 (sum stub is id) / 2


def test_allreduce_uses_zero_for_a_locally_unused_grad(monkeypatch):
    import yeto.learner as learner

    used_elsewhere = _param(None)
    calls = 0

    def fake_all_reduce(t, op=None):
        nonlocal calls
        calls += 1
        if t.dtype == torch.int32:  # another rank used the parameter
            t.fill_(1)
        else:  # its gradient contribution was [2, 4, 6]
            t.add_(torch.tensor([2.0, 4.0, 6.0]))

    monkeypatch.setattr(learner.dist, "all_reduce", fake_all_reduce)
    allreduce_trainable_grads([used_elsewhere], world=2)
    assert calls == 2
    assert torch.allclose(used_elsewhere.grad, torch.tensor([1.0, 2.0, 3.0]))


# --- exact target-token normalization -------------------------------------


class _TinyLM(torch.nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        initial = torch.tensor(
            [
                [0.10, -0.20, 0.30, 0.00],
                [-0.10, 0.25, 0.05, -0.20],
                [0.20, 0.10, -0.15, 0.05],
                [0.00, -0.05, 0.20, 0.15],
            ]
        )
        self.weight = torch.nn.Parameter(initial if weight is None else weight.clone())
        self.forward_calls = 0

    def forward(self, input_ids):
        self.forward_calls += 1
        one_hot = torch.nn.functional.one_hot(input_ids, num_classes=4).float()
        return SimpleNamespace(logits=one_hot @ self.weight)


class _Loader:
    sampler = None

    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)


class _RecordingOptimizer:
    def __init__(self, params):
        self.params = list(params)
        self.grads = None
        self.steps = 0

    def zero_grad(self, set_to_none=True):
        for parameter in self.params:
            parameter.grad = None if set_to_none else torch.zeros_like(parameter)

    def step(self):
        self.steps += 1
        self.grads = [parameter.grad.detach().clone() for parameter in self.params]


class _Scheduler:
    def __init__(self, lr=1e-4):
        self.steps = 0
        self.lr = lr

    def step(self):
        self.steps += 1

    def get_last_lr(self):
        return [self.lr]


def _loop_args(grad_accum=2):
    return SimpleNamespace(
        grad_accum=grad_accum,
        loss_function="cross_entropy",
        max_local_steps=1,
        merge_alpha=0.0,
        micro_batch_size=1,
        seq_len=4,
        shard="ddp",
        tuning="lora",
    )


def _layout():
    return SimpleNamespace(num_fragments=1, fragments=[SimpleNamespace(numel=16)])


def _batch(ids, weights):
    return torch.tensor([ids], dtype=torch.long), torch.tensor([weights], dtype=torch.float32)


def _reference_grad(initial_weight, batches):
    model = _TinyLM(initial_weight)
    loss_sum = torch.zeros(())
    targets = 0
    for input_ids, weights in batches:
        out = model(input_ids)
        loss, count = sft_loss(out.logits, input_ids, weights=weights)
        loss_sum = loss_sum + loss
        targets += int(count)
    (loss_sum / targets).backward()
    torch.nn.utils.clip_grad_norm_([model.weight], 1.0)
    return model.weight.grad.detach(), targets


def test_accumulation_gradient_matches_one_global_target_mean():
    # The microbatches have one and three targets. Averaging their individual
    # means would overweight the first; the loop must equal one concatenated
    # sum(loss) / sum(targets) objective.
    batches = [
        _batch([0, 1, 2, 3], [0, 1, 0, 0]),
        _batch([3, 2, 1, 0], [0, 1, 1, 1]),
    ]
    model = _TinyLM()
    expected, target_count = _reference_grad(model.weight.detach(), batches)
    opt = _RecordingOptimizer([model.weight])
    sched = _Scheduler()

    counters = run_inner_loop(
        _loop_args(),
        model,
        {"weight": model.weight},
        _layout(),
        opt,
        sched,
        _Loader(batches),
        None,
        rank=0,
        world=1,
        device=torch.device("cpu"),
    )

    assert opt.steps == sched.steps == 1
    assert torch.allclose(opt.grads[0], expected, atol=1e-7, rtol=1e-6)
    assert counters.raw_tokens == 8
    assert counters.target_tokens == target_count == 4


def test_mocked_two_rank_gradient_includes_zero_target_rank(monkeypatch):
    import yeto.learner as learner

    rank0_batches = [
        _batch([0, 1, 2, 3], [0, 0, 0, 0]),
        _batch([3, 2, 1, 0], [0, 0, 0, 0]),
    ]
    rank1_batches = [
        _batch([1, 2, 3, 0], [0, 1, 0, 0]),
        _batch([2, 3, 0, 1], [0, 1, 1, 0]),
    ]
    model = _TinyLM()
    initial = model.weight.detach().clone()
    expected, global_targets = _reference_grad(
        initial, rank0_batches + rank1_batches
    )

    # Precompute rank 1's local contribution after the loop's world/global
    # loss scaling. The fake all-reduce adds it to rank 0's local gradient;
    # allreduce_trainable_grads then divides the sum by world like DDP/FSDP.
    rank1_model = _TinyLM(initial)
    rank1_loss = torch.zeros(())
    for input_ids, weights in rank1_batches:
        out = rank1_model(input_ids)
        loss, _ = sft_loss(out.logits, input_ids, weights=weights)
        rank1_loss = rank1_loss + loss
    (rank1_loss * (2 / global_targets)).backward()
    rank1_scaled_grad = rank1_model.weight.grad.detach().clone()
    rank1_raw = sum(ids.numel() for ids, _ in rank1_batches)

    def fake_all_reduce(tensor, op=None):
        if tensor.ndim == 0 and tensor.dtype == torch.long:
            return  # common accumulation size
        if tensor.shape == (2,) and tensor.dtype == torch.long:
            tensor.add_(torch.tensor([global_targets, rank1_raw]))
            # rank 0 has zero targets, so adding global_targets is rank 1's count.
            return
        if tensor.ndim == 0 and tensor.dtype == torch.int32:
            return  # target-count contract is true on both ranks
        if tensor.shape == (1,) and tensor.dtype == torch.int32:
            tensor.add_(1)  # rank 1 also has this gradient
            return
        if tensor.shape == model.weight.shape:
            tensor.add_(rank1_scaled_grad)
            return
        raise AssertionError(f"unexpected all_reduce tensor {tensor.shape}/{tensor.dtype}")

    monkeypatch.setattr(learner.dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(learner.dist, "broadcast_object_list", lambda *a, **k: None)
    opt = _RecordingOptimizer([model.weight])
    counters = run_inner_loop(
        _loop_args(),
        model,
        {"weight": model.weight},
        _layout(),
        opt,
        _Scheduler(),
        _Loader(rank0_batches),
        None,
        rank=0,
        world=2,
        device=torch.device("cpu"),
    )

    assert torch.allclose(opt.grads[0], expected, atol=1e-7, rtol=1e-6)
    assert counters.raw_tokens == 16
    assert counters.target_tokens == global_targets == 3


def test_globally_zero_target_group_fails_before_forward():
    batches = [
        _batch([0, 1, 2, 3], [0, 0, 0, 0]),
        _batch([3, 2, 1, 0], [0, 0, 0, 0]),
    ]
    model = _TinyLM()
    opt = _RecordingOptimizer([model.weight])
    with pytest.raises(ValueError, match="zero positive"):
        run_inner_loop(
            _loop_args(),
            model,
            {"weight": model.weight},
            _layout(),
            opt,
            _Scheduler(),
            _Loader(batches),
            None,
            rank=0,
            world=1,
            device=torch.device("cpu"),
        )
    assert model.forward_calls == 0
    assert opt.steps == 0


def test_incomplete_accumulation_tail_is_not_an_optimizer_step():
    batches = [
        _batch([0, 1, 2, 3], [0, 1, 1, 1]),
        _batch([3, 2, 1, 0], [0, 1, 0, 0]),
        _batch([1, 3, 0, 2], [0, 1, 1, 0]),  # incomplete second group
    ]
    model = _TinyLM()
    opt = _RecordingOptimizer([model.weight])
    args = _loop_args()
    args.max_local_steps = 2
    counters = run_inner_loop(
        args,
        model,
        {"weight": model.weight},
        _layout(),
        opt,
        _Scheduler(),
        _Loader(batches),
        None,
        rank=0,
        world=1,
        device=torch.device("cpu"),
    )
    assert opt.steps == 2
    # The one-batch tail is dropped; the second step starts with a fresh full
    # group from the next epoch.
    assert model.forward_calls == 4
    assert counters.raw_tokens == 16
    assert counters.target_tokens == 8


def test_dataset_smaller_than_accumulation_group_fails_instead_of_spinning():
    batches = [_batch([0, 1, 2, 3], [0, 1, 1, 0])]
    model = _TinyLM()
    opt = _RecordingOptimizer([model.weight])

    with pytest.raises(ValueError, match="fewer microbatches than --grad-accum"):
        run_inner_loop(
            _loop_args(grad_accum=2),
            model,
            {"weight": model.weight},
            _layout(),
            opt,
            _Scheduler(),
            _Loader(batches),
            None,
            rank=0,
            world=1,
            device=torch.device("cpu"),
        )

    assert opt.steps == 0
    assert model.forward_calls == 0


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


def test_offline_first_falls_back_online_on_partial_cache():
    """A partial cache fails offline load with arbitrary exceptions, not just
    OSError — e.g. sentencepiece raising `TypeError: not a string` when
    tokenizer_config.json is cached without its vocab file (observed on the
    megatron island's first hardware run). Any offline failure must fall
    back to the online path."""
    from yeto.learner import _from_pretrained_offline_first

    calls = []

    class Factory:
        @staticmethod
        def from_pretrained(model_id, **kw):
            calls.append(kw)
            if kw.get("local_files_only"):
                raise TypeError("not a string")
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


def test_torch_and_mlx_learners_parse_data_format_and_assistant_mask_mode():
    from yeto.learner import parse_args as parse_torch_args
    from yeto.mlx.learner import parse_args as parse_mlx_args

    base = [
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
    ]
    for parse in (parse_torch_args, parse_mlx_args):
        assert parse(base).data_format == "auto"
        assert parse(base + ["--data-format", "sharegpt"]).data_format == "sharegpt"
        assert parse(base).assistant_mask_mode == "native"
        assert parse(base + ["--assistant-mask-mode", "legacy"]).assistant_mask_mode == "legacy"


def test_learner_defaults_to_shared_initialization_seed():
    from yeto.learner import parse_args

    args = parse_args(
        [
            "--model",
            "org/model",
            "--data",
            "org/data",
            "--syncer",
            "none",
            "--learner-id",
            "0",
            "--num-learners",
            "1",
        ]
    )
    assert args.seed == 0


def test_learner_quantization_default_is_unchanged():
    from yeto.learner import parse_args

    args = parse_args(
        [
            "--model", "org/model",
            "--data", "rows.jsonl",
            "--syncer", "none",
            "--learner-id", "0",
            "--num-learners", "1",
        ]
    )
    assert args.base_quantization == "none"


def test_nf4_load_uses_qlora_quantization_recipe(monkeypatch):
    import peft
    import transformers
    import yeto.learner as learner

    seen = {}
    tokenizer = object()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace()
            self.norm = torch.nn.LayerNorm(2, dtype=torch.bfloat16)
            self.proj = torch.nn.Linear(2, 2, dtype=torch.bfloat16)

    base_model = Model()

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            seen["quantization"] = kwargs

    def fake_from_pretrained(factory, _model_id, **kwargs):
        if factory is transformers.AutoTokenizer:
            return tokenizer
        assert factory is transformers.AutoModelForCausalLM
        seen["model_kwargs"] = kwargs
        return base_model

    monkeypatch.setattr(transformers, "BitsAndBytesConfig", FakeBitsAndBytesConfig)
    monkeypatch.setattr(learner, "_from_pretrained_offline_first", fake_from_pretrained)
    monkeypatch.setattr(learner, "attention_load_kwargs", lambda *args: {})
    monkeypatch.setattr(learner, "resolved_attention_backend", lambda *args: "sdpa")
    monkeypatch.setattr(peft, "LoraConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(peft, "get_peft_model", lambda model, _config: model)

    args = SimpleNamespace(
        model="org/model",
        base_quantization="nf4",
        tuning="lora",
        shard="ddp",
        kernel_backend="native",
        attention_backend="auto",
        loss_function="cross_entropy",
        lora_r=16,
        lora_alpha=32,
        lora_targets="auto",
    )

    model, loaded_tokenizer = load_model_and_tokenizer(
        args, torch.device("cuda", 0)
    )

    assert model is base_model and loaded_tokenizer is tokenizer
    assert seen["quantization"] == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16,
        "bnb_4bit_use_double_quant": True,
    }
    assert seen["model_kwargs"]["device_map"] == {"": 0}
    assert seen["model_kwargs"]["low_cpu_mem_usage"] is True
    assert all(not parameter.requires_grad for parameter in base_model.parameters())
    assert base_model.norm.weight.dtype == torch.float32


def test_nf4_preparation_freezes_base_and_only_casts_norms():
    from yeto.learner import _prepare_nf4_base_for_lora

    class RMSNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))

    model = torch.nn.ModuleDict(
        {
            "embed": torch.nn.Embedding(8, 4, dtype=torch.bfloat16),
            "norm": RMSNorm(),
            "head": torch.nn.Linear(4, 8, bias=False, dtype=torch.bfloat16),
        }
    )
    _prepare_nf4_base_for_lora(model)
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert model["norm"].weight.dtype == torch.float32
    assert model["embed"].weight.dtype == torch.bfloat16
    assert model["head"].weight.dtype == torch.bfloat16


def test_nf4_preparation_casts_fp32_final_norm_output_for_bf16_head():
    from yeto.learner import _prepare_nf4_base_for_lora

    class OutputHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
            self.input_dtype = None

        def forward(self, hidden_states):
            self.input_dtype = hidden_states.dtype
            return hidden_states

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.LayerNorm(4, dtype=torch.bfloat16)
            self.head = OutputHead()

        def get_output_embeddings(self):
            return self.head

    model = Model()
    _prepare_nf4_base_for_lora(model)

    output = model.head(torch.ones(2, 4, dtype=torch.float32))

    assert model.norm.weight.dtype == torch.float32
    assert model.head.weight.dtype == torch.bfloat16
    assert model.head.input_dtype == torch.bfloat16
    assert output.dtype == torch.bfloat16


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


class _RecordingRun:
    """Stands in for a live W&B run (yeto.wandb_logger.WandbRun)."""

    enabled = True

    def __init__(self):
        self.logged = []

    def log(self, metrics):
        self.logged.append(metrics)

    def summary(self, metrics):
        pass

    def finish(self, exit_code=0):
        pass


def _telemetry_loop(wandb_run, *, max_local_steps=10, batches=None):
    batches = batches or [_batch([0, 1, 2, 3], [0, 1, 1, 1])] * (2 * max_local_steps)
    model = _TinyLM()
    args = _loop_args()
    args.max_local_steps = max_local_steps
    return run_inner_loop(
        args,
        model,
        {"weight": model.weight},
        _layout(),
        _RecordingOptimizer([model.weight]),
        _Scheduler(),
        _Loader(batches),
        None,
        rank=0,
        world=1,
        device=torch.device("cpu"),
        wandb_run=wandb_run,
    )


def test_training_metrics_ride_the_existing_ten_step_log():
    run = _RecordingRun()
    counters = _telemetry_loop(run, max_local_steps=10)

    assert counters.local_steps == 10
    # One point per ten steps, on the same cadence as the log line.
    assert len(run.logged) == 1
    metrics = run.logged[0]
    assert metrics["local_step"] == 10
    assert metrics["global_step"] == 0
    assert metrics["train/loss_per_token"] > 0
    assert metrics["train/lr"] == 1e-4
    assert metrics["train/raw_tokens_total"] == counters.raw_tokens
    assert metrics["train/target_tokens_total"] == counters.target_tokens
    assert metrics["train/sec_per_step"] >= 0
    assert metrics["train/tokens_per_sec"] >= 0


def test_no_telemetry_before_the_first_ten_steps():
    run = _RecordingRun()
    _telemetry_loop(run, max_local_steps=5)
    assert run.logged == []


def test_the_loop_runs_unchanged_without_a_run_object():
    # The default path (no --wandb) must not require the caller to pass a
    # sink; every existing call site relies on this.
    counters = _telemetry_loop(None, max_local_steps=10)
    assert counters.local_steps == 10


def test_throughput_is_measured_per_window_not_cumulatively():
    run = _RecordingRun()
    counters = _telemetry_loop(run, max_local_steps=20)
    assert len(run.logged) == 2
    first, second = run.logged
    # Each window reports the tokens of that window only; the running total
    # is a separate series.
    assert second["train/raw_tokens_total"] == counters.raw_tokens
    assert first["train/raw_tokens_total"] == counters.raw_tokens // 2
    window_tokens = second["train/tokens_per_sec"] * second["train/sec_per_step"] * 10
    assert window_tokens == pytest.approx(counters.raw_tokens / 2, rel=1e-6)
