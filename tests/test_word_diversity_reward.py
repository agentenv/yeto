import asyncio
import types

from yeto.rl.word_diversity_reward import diversity, score


def test_diversity_values():
    assert diversity("") == 0.0
    assert diversity("unload unload unload unload") == 0.25
    assert diversity("a b c d") == 1.0


def test_score_has_variance_on_truncated_degenerate_samples():
    texts = ["Ref unload unload unload", "adam unload unloadimers unloadimers"]
    values = [asyncio.run(score(None, types.SimpleNamespace(response=t))) for t in texts]
    assert values[0] != values[1]
    assert asyncio.run(score(None, types.SimpleNamespace(response=None))) == 0.0
