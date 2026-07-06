"""Alias-table consistency: one source of truth for model sugar."""

from yeto.models import MODEL_ALIASES, MODEL_SPECS, MODEL_WEIGHT_GB, resolve


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
    lm_section = readme.split("### Language-model aliases", 1)[1].split("### Component aliases", 1)[0]
    table_aliases = set(re.findall(r"^\| `([a-z0-9\-]+)` \| `", lm_section, flags=re.M))
    assert table_aliases == set(MODEL_ALIASES), (
        "README model table is out of sync with yeto/models.py — regenerate it"
    )


def test_readme_lists_structured_model_specs():
    import pathlib

    readme = (pathlib.Path(__file__).resolve().parents[1] / "README.md").read_text()
    component_section = readme.split("### Component aliases", 1)[1].split("## Docs", 1)[0]
    for alias in MODEL_SPECS:
        assert f"`{alias}`" in component_section


def test_resolve_passes_aliases_and_raw_ids():
    from yeto.models import resolve

    assert resolve("gemma4") == "google/gemma-4-12B-it"
    assert resolve("org/custom-repo") == "org/custom-repo"
