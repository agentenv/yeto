"""Alias-table consistency: one source of truth for model sugar."""

from yeto.models import MODEL_ALIASES, MODEL_WEIGHT_GB, resolve


def test_every_weight_key_is_an_alias():
    # Aliases may omit a weight (planning falls back to Hub metadata), but a
    # weight entry without an alias is dead data.
    assert set(MODEL_WEIGHT_GB) <= set(MODEL_ALIASES)


def test_weights_are_plausible_bf16_footprints():
    for alias, gb in MODEL_WEIGHT_GB.items():
        assert 1 <= gb <= 3500, f"{alias}: {gb} GB looks wrong"


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
