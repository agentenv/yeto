"""Alias-table consistency: one source of truth for model sugar."""

from yeto.models import MODEL_ALIASES, MODEL_WEIGHT_GB, resolve


def test_every_weight_key_is_an_alias():
    # Aliases may omit a weight (planning falls back to Hub metadata), but a
    # weight entry without an alias is dead data.
    assert set(MODEL_WEIGHT_GB) <= set(MODEL_ALIASES)


def test_weights_are_plausible_bf16_footprints():
    for alias, gb in MODEL_WEIGHT_GB.items():
        assert 0.4 <= gb <= 3500, f"{alias}: {gb} GB looks wrong"


def test_resolve():
    assert resolve("qwen35-9b") == "Qwen/Qwen3.5-9B"
    assert resolve("org/custom-model") == "org/custom-model"


def test_learner_and_launcher_reexport_the_same_tables():
    from yeto import launcher

    assert launcher.MODEL_WEIGHT_GB is MODEL_WEIGHT_GB
    # learner imports torch; only check it lazily if available.
    try:
        from yeto import learner
    except ImportError:
        return
    assert learner.MODEL_ALIASES is MODEL_ALIASES


def test_shape_memory_uses_the_shared_aliases():
    from yeto.shape import memory

    assert memory.MODEL_ALIASES is MODEL_ALIASES


def test_readme_table_matches_alias_table():
    import pathlib
    import re

    readme = (pathlib.Path(__file__).resolve().parents[1] / "README.md").read_text()
    table_aliases = set(re.findall(r"^\| `([a-z0-9\-]+)` \| `", readme, flags=re.M))
    assert table_aliases == set(MODEL_ALIASES), (
        "README model table is out of sync with yeto/models.py — regenerate it"
    )


def test_resolve_variant_uses_published_checkpoints_only():
    import pytest

    from yeto.models import resolve_variant

    # bf16: plain alias resolution.
    assert resolve_variant("gemma4", "bf16") == "google/gemma-4-12B-it"
    # Known published variants (native fp8 / mxfp4 repos).
    assert resolve_variant("deepseek4flash", "fp8") == "deepseek-ai/DeepSeek-V4-Flash"
    assert resolve_variant("gptoss-120b", "fp4") == "openai/gpt-oss-120b"
    assert resolve_variant("ornith-397b", "fp8") == "deepreinforce-ai/Ornith-1.0-397B-FP8"
    # No published checkpoint known -> refuse rather than quantize.
    with pytest.raises(ValueError, match="does not quantize"):
        resolve_variant("gemma4", "fp4")
    # Raw HF ids pass through on the caller's word.
    assert resolve_variant("org/custom-fp4-repo", "fp4") == "org/custom-fp4-repo"
