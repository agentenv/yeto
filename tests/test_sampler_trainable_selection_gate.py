"""The sampler's partial-experts gate by trainable expert sets (expert_trainable_sha256):
the same trainable experts under another EP layout (another slot order, another whole-selection
sha256) are accepted; other trainable sets, or a manifest without them, are refused."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto.rl.adapters.miles.models.partial_experts import selection as sel
from yeto.rl.engine import engine_sampler_entry as entry
from yeto.rl.engine import sampler_checks as checks
from yeto.rl.engine.policy_release import PolicyRelease


def _selection(ep_size: int, hot: int = 0) -> sel.ExpertSelection:
    counts = {layer: [(e * 7 + layer + (100 if e == hot else 0)) % 37 for e in range(32)] for layer in (1, 2)}
    return sel.ExpertSelection.from_counts(counts, model="fake", num_trainable=16, ep_size=ep_size)


def _manifest(selection: sel.ExpertSelection) -> dict:
    return {"expert_selection_sha256": selection.sha256,
            "partial_experts": {"trainable_sha256": selection.trainable_sha256}}


def _loader(monkeypatch, **expected):
    engine = SimpleNamespace(identity=SimpleNamespace(quant_scheme="bf16"), served_version=lambda: 0)
    loader = checks.PortReleaseLoader(engine, checks.PostLoadChecks(), lambda *a, **k: None,
                                      max_consecutive_failures=10, **expected)
    served: list[int] = []
    monkeypatch.setattr(checks, "overlaid_release_scheme", lambda *a: "bf16")
    monkeypatch.setattr(loader, "_serve", lambda release, directory, version, first: served.append(version))
    return loader, served


def _release(version: int, manifest) -> PolicyRelease:
    return PolicyRelease(seq=version, version=version, policy_hash="h", path=f"r{version}-v{version}",
                         manifest=manifest)


def test_same_trainable_experts_under_another_ep_layout_are_accepted(monkeypatch, tmp_path):
    ep8, ep16 = _selection(8), _selection(8).with_ep_size(16)
    assert ep8.sha256 != ep16.sha256 and ep8.trainable_sha256 == ep16.trainable_sha256
    loader, served = _loader(monkeypatch, expected_selection_sha256=ep8.sha256,
                             expected_trainable_sha256=ep8.trainable_sha256)
    loader(_release(1, _manifest(ep16)), Path(tmp_path))
    assert served == [1]


def test_other_trainable_experts_or_no_trainable_sha_are_refused(monkeypatch, tmp_path):
    ours, other = _selection(8), _selection(8, hot=5)
    assert ours.trainable_sha256 != other.trainable_sha256
    loader, served = _loader(monkeypatch, expected_trainable_sha256=ours.trainable_sha256)
    for version, manifest in ((1, _manifest(other)), (2, {"expert_selection_sha256": ours.sha256}), (3, None)):
        with pytest.raises(checks.ReleaseNotLoaded) as refused:
            loader(_release(version, manifest), Path(tmp_path))
        assert refused.value.reason == "release_refused:expert_selection"
    assert served == []


def test_without_the_trainable_sha_the_whole_selection_still_gates(monkeypatch, tmp_path):
    ep8, ep16 = _selection(8), _selection(8).with_ep_size(16)
    loader, served = _loader(monkeypatch, expected_selection_sha256=ep8.sha256)
    with pytest.raises(checks.ReleaseNotLoaded):
        loader(_release(1, _manifest(ep16)), Path(tmp_path))
    loader(_release(2, _manifest(ep8)), Path(tmp_path))
    assert served == [2]


def test_the_spec_field_reaches_the_sampler_config():
    spec = {"sampler_id": "s", "num_rollouts": 1, "max_backlog_groups": 1, "max_prompt_tokens": 8,
            "sampling": {"max_tokens": 4}, "expert_trainable_sha256": "a" * 64}
    assert entry.island_config(spec).expert_trainable_sha256 == "a" * 64
    assert entry.island_config({**spec, "expert_trainable_sha256": None}).expert_trainable_sha256 is None
