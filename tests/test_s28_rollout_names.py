"""S28 G-M2: the full-parameter vLLM weight stream goes out in the checkpoint's names."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("transformers.models.nemotron_h")

from yeto.rl.adapters.verl import rollout_names as rn  # noqa: E402

TINY = dict(vocab_size=512, hidden_size=64, intermediate_size=64, num_hidden_layers=3,
            hybrid_override_pattern="M*E", num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            mamba_num_heads=4, mamba_head_dim=16, ssm_state_size=8, n_groups=1, conv_kernel=4,
            n_routed_experts=3, n_shared_experts=1, moe_intermediate_size=32,
            moe_shared_expert_intermediate_size=32, num_experts_per_tok=2, tie_word_embeddings=False)


@pytest.fixture
def tiny(tmp_path):
    from transformers import NemotronHConfig, NemotronHForCausalLM

    torch.manual_seed(0)
    model = NemotronHForCausalLM(NemotronHConfig(**TINY))
    model.save_pretrained(tmp_path)  # writes the checkpoint (per-expert, backbone.) layout
    return tmp_path, model


def test_stream_comes_out_in_checkpoint_names_and_rows(tiny):
    from safetensors.torch import load_file

    path, model = tiny
    ckpt = load_file(str(path / "model.safetensors"))
    targets = rn.state_targets(path)
    assert targets is not None
    out = dict(rn.checkpoint_stream(model.state_dict().items(), targets))
    assert set(out) == set(ckpt)
    for name, tensor in ckpt.items():
        assert torch.equal(out[name], tensor), name
    # the actor may stream names without the "model." prefix
    short = {k.removeprefix("model."): v for k, v in model.state_dict().items()}
    assert set(dict(rn.checkpoint_stream(short.items(), targets))) == set(ckpt)


def test_identity_model_passes_through():
    stream = [("a.weight", torch.zeros(2))]
    assert list(rn.checkpoint_stream(stream, None)) == stream


def test_patch_is_in_the_fsdp_profile_only():
    from yeto.rl.adapters.verl import patch_verl as pv

    assert pv.CKPT_NAMES in pv.PROFILES["fsdp"] and pv.CKPT_NAMES not in pv.PROFILES["megatron"]
    text = "x\n" + pv.CKPT_NAMES_ANCHOR
    out = pv._patched(text, pv.CKPT_NAMES_ANCHOR, pv.CKPT_NAMES_HOOK, "before")
    assert out.index(pv.CKPT_NAMES_HOOK) < out.index(pv.CKPT_NAMES_ANCHOR)


def test_worker_stream_leaves_lora_alone_and_caches(tiny):
    from types import SimpleNamespace

    path, model = tiny
    worker = SimpleNamespace(config=SimpleNamespace(model=SimpleNamespace(path=str(path))))
    stream = [("x", torch.zeros(1))]
    assert rn.worker_stream(worker, stream, peft_config={"r": 8}) is stream
    out = dict(rn.worker_stream(worker, model.state_dict().items()))
    assert "backbone.embeddings.weight" in out or any(k.startswith("backbone.") for k in out)
    assert worker._yeto_ckpt_targets is not None
