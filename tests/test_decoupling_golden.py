"""yeto-framework-decoupling phase 0 (task 1.1): golden samples must not move.

Recomputes every sample in ``tests/golden/decoupling/`` with the current code
(CPU only, see ``tests/decoupling_golden.py``) and compares the text byte for
byte.  On a mismatch the failing fields are named.  A difference is only
acceptable when it is intended and recorded in
``openspec/changes/yeto-framework-decoupling/hash-migration.md``; then
regenerate with ``python tests/decoupling_golden.py --write``.
"""

from __future__ import annotations

import json
import sys

import pytest

import decoupling_golden as dg

NAMES = [f"{c.name}.json" for c in dg.CONFIGS] + ["fake_engine_tapes.json"]


def _diff_fields(old: dict, new: dict) -> list[str]:
    keys = sorted(set(old) | set(new))
    return [k for k in keys if old.get(k) != new.get(k)]


@pytest.fixture(scope="module")
def rendered():
    return dg.render_all()


def test_sample_set_is_complete(rendered):
    on_disk = sorted(p.name for p in dg.GOLDEN_DIR.glob("*.json"))
    assert on_disk == sorted(rendered), "golden files and configurations differ"
    assert 6 <= len(dg.CONFIGS) <= 8


@pytest.mark.parametrize("name", NAMES)
def test_golden_sample_is_byte_identical(rendered, name):
    expected = (dg.GOLDEN_DIR / name).read_text()
    actual = rendered[name]
    if actual != expected:
        fields = _diff_fields(json.loads(expected), json.loads(actual))
        pytest.fail(f"{name}: regenerated sample differs in {fields}; see hash-migration.md "
                    "before regenerating (python tests/decoupling_golden.py --write)")


def test_hashes_agree_across_launcher_learner_and_translation(rendered):
    for config in dg.CONFIGS:
        sample = json.loads(rendered[f"{config.name}.json"])
        sha = sample["algorithm_sha256"]
        assert sample["algorithm_sha256_launcher_expected"] == sha, config.name
        assert sample["algorithm_sha256_learner_ports_spec"] == sha, config.name
        assert sample["launch_algorithm_sha256"] == sha, config.name
        assert sample["execution_profile"]["profile"]["algorithm_spec_sha256"] == sha


def test_no_ray_and_no_miles_loaded(rendered):
    assert "ray" not in sys.modules
    assert not any(m == "miles" or m.startswith("miles.") for m in sys.modules)
