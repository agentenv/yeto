"""A cut of a partial-experts run binds its expert selection: no flag resumes it under another."""

from types import SimpleNamespace

import pytest

from yeto.rl.engine.resume import ResumeRefused, check_fingerprint


def _fingerprint(selection=None):
    from yeto.rl.adapters.miles.entry import run_fingerprint_of

    manifest = None if selection is None else {"expert_selection_sha256": selection}
    return run_fingerprint_of(SimpleNamespace(yeto_rl_release_manifest=manifest), SimpleNamespace(sha256=lambda: "s"))


def test_only_a_partial_experts_run_carries_the_selection():
    assert "expert_selection_sha256" not in _fingerprint()
    assert _fingerprint("a" * 64)["expert_selection_sha256"] == "a" * 64


def test_another_selection_is_refused_even_with_allow_config_change():
    saved, now = _fingerprint("a" * 64), _fingerprint("b" * 64)
    assert check_fingerprint(saved, _fingerprint("a" * 64), allow_change=False) == {}
    for allow in (False, True):
        with pytest.raises(ResumeRefused, match="expert_selection_sha256"):
            check_fingerprint(saved, now, allow_change=allow)
