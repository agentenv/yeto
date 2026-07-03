import sys
from types import SimpleNamespace

import pytest

from yeto.shape import memory
from yeto.shape.memory import fits, min_nodes, model_weights_gb


def test_deepseek_lora_on_a100_40_needs_three_nodes():
    # fits at n nodes <=> 568/(8n) + 8 <= 0.92*40 = 36.8
    # n=2: 35.5 + 8 = 43.5 > 36.8; n=3: 23.67 + 8 = 31.67 <= 36.8
    assert not fits(568, "lora", 40, total_gpus=16)
    assert fits(568, "lora", 40, total_gpus=24)
    assert min_nodes(568, "lora", 40, gpus_per_node=8) == 3


def test_gemma_lora_fits_single_node():
    assert min_nodes(66, "lora", 80, gpus_per_node=8) == 1
    # even on 40 GB cards: 66/8 = 8.25 + 8 = 16.25 <= 36.8
    assert min_nodes(66, "lora", 40, gpus_per_node=8) == 1


def test_full_tuning_multiplies_footprint():
    # full = 8x the bf16 weight bytes (fp32 master + grad + Adam m,v):
    # n=1: 66*8/8 + 8 = 74 > 0.92*80 = 73.6; n=2: 33 + 8 = 41 <= 73.6
    assert not fits(66, "full", 80, total_gpus=8)
    assert fits(66, "full", 80, total_gpus=16)
    assert min_nodes(66, "full", 80, gpus_per_node=8) == 2


def test_seq_len_scales_activation_overhead():
    # seq 4096 -> overhead 2 + 6*2 = 14 GB (vs 8 at 2048).
    # deepseek lora on 40 GB, 3 nodes: 23.67 + 14 = 37.67 > 36.8 now fails;
    # 4 nodes: 17.75 + 14 = 31.75 fits.
    assert fits(568, "lora", 40, total_gpus=24, seq_len=2048)
    assert not fits(568, "lora", 40, total_gpus=24, seq_len=4096)
    assert min_nodes(568, "lora", 40, gpus_per_node=8, seq_len=4096) == 4


def test_unknown_tuning_rejected():
    with pytest.raises(ValueError, match="tuning"):
        fits(66, "qlora", 80, total_gpus=8)


def _forbid_hub(monkeypatch):
    def boom(model_id):
        raise AssertionError(f"hub fetch called for {model_id}")

    monkeypatch.setattr(memory, "_fetch_hub_weights", boom)


def test_override_wins(monkeypatch):
    _forbid_hub(monkeypatch)
    assert model_weights_gb("some/unknown-model", override=123.0) == 123.0


def test_alias_hits_launcher_table_without_network(monkeypatch):
    _forbid_hub(monkeypatch)
    assert model_weights_gb("gemma4") == 66.0
    assert model_weights_gb("deepseek4flash") == 568.0
    # the resolved HF id maps back through the alias to the same entry
    assert model_weights_gb("google/gemma-4-12B-it") == 66.0


def test_hub_fetch_sums_safetensors_sizes(monkeypatch):
    siblings = [
        SimpleNamespace(rfilename="model-00001.safetensors", size=5_000_000_000),
        SimpleNamespace(rfilename="model-00002.safetensors", size=2_500_000_000),
        SimpleNamespace(rfilename="pytorch_model.bin", size=9_999_999_999),  # skipped
        SimpleNamespace(rfilename="model-00003.safetensors", size=None),  # skipped
    ]

    class FakeApi:
        def model_info(self, model_id, files_metadata=False):
            assert files_metadata
            return SimpleNamespace(siblings=siblings)

    fake_hub = SimpleNamespace(HfApi=FakeApi)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    # 7.5e9 bytes -> 7.5 GB, rounded up to 8
    assert model_weights_gb("acme/unknown-model") == 8.0


def test_unknown_model_uses_hub_fetch_and_cache(monkeypatch):
    calls = {"n": 0}

    def fake_fetch(model_id):
        calls["n"] += 1
        assert model_id == "acme/unknown-model"
        return 13.0

    monkeypatch.setattr(memory, "_fetch_hub_weights", fake_fetch)

    class FakeCache:
        def __init__(self):
            self.store = {}

        def get_or(self, key, fetch):
            if key not in self.store:
                self.store[key] = fetch()
            return self.store[key]

    cache = FakeCache()
    assert model_weights_gb("acme/unknown-model", cache=cache) == 13.0
    assert model_weights_gb("acme/unknown-model", cache=cache) == 13.0
    assert calls["n"] == 1
    assert list(cache.store) == ["hf-weights:acme/unknown-model"]


def test_hub_failure_points_at_weights_gb_flag(monkeypatch):
    def fake_fetch(model_id):
        raise OSError("401 gated repo")

    monkeypatch.setattr(memory, "_fetch_hub_weights", fake_fetch)
    with pytest.raises(ValueError, match="--weights-gb"):
        model_weights_gb("acme/gated-model")


def test_hub_zero_bytes_points_at_weights_gb_flag(monkeypatch):
    monkeypatch.setattr(memory, "_fetch_hub_weights", lambda model_id: 0.0)
    with pytest.raises(ValueError, match="--weights-gb"):
        model_weights_gb("acme/empty-model")
