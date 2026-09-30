"""Snapshot guard for the build_miles_argv split (rl-engine-ports task 2.4).

Every ``build_miles_argv`` call made by the existing launcher tests is
recorded (argv or raised error) and compared with a golden digest captured
from the unsplit function on main (originally c40a32c; regenerated at
cb55aca after #65 (unchanged at e21a7ff) intentionally changed the GDN-provider argv to the
``qwen3_5`` recipe -- the only entry that changed).  Paths under ``tmp_path`` are
normalized so the digests are machine independent.

Regenerate (only when an argv change is intended):
    YETO_ARGV_SNAPSHOT_PRINT=1 python -m pytest -q -s tests/test_rl_argv_snapshot.py
"""

from __future__ import annotations

import hashlib
import json
import os

import pytest

from yeto.rl import learner as rl_learner

import test_rl_launcher as launcher_tests

# test name -> [(kind, sha256 of normalized payload), ...] in call order.
GOLDEN: dict[str, list[list[str]]] = {
    "test_miles_argv_preserves_mrope_provider_configuration": [
        ["argv", "db53a537bbe8244326013a0b06be402873e93b2655c1ade0397b65e84788df22"],
    ],
    "test_miles_argv_uses_provider_capabilities_without_model_family_branches": [
        ["argv", "f73a2ff3f820066f77baf9c83e94369570bb17e197875bdb66325845e19dff63"],
        ["argv", "e4e9e0b0b848e2106208259d1f434f69162e15a3b785d1a50d08d04786de1acd"],
        ["argv", "f450592ea51be5f667b1b312585fe94d1840a8462c1eefc6cac8497eb03f7744"],
        ["argv", "0dff34624fe9d7a6aae07f6c4088671d6aa5ff08a348c3fa9a9816b9c2c43e0b"],
        ["raise", "248821ecfaf8627a9e55bb9b7f52052194df4125eb15baf0c92d3af83d00658c"],
        ["argv", "ce80f088374ff0cdfcb620fe2156a289cced7eb6714097c01356761132f31f06"],
        ["argv", "2d3cab802025fdaed551cb4bb2d325d09e15ab72a14b4bfd848ac4020f7a794f"],
        ["argv", "feab2f2e72c969c52e1c6580c2561eadd2adf5df2e27f1cac87bff30006dadbf"],
        ["argv", "6068f265bfe7e8ca62b8552b2e7933ec687ccd93035afd65554dff33b8bcc3dd"],
        ["argv", "1ef2b89bd7017f7a25dd5d131de6048bf0d694ce6ffce752aff954e7a1f44a7e"],
        ["argv", "c7e2251804f97425fe324edf34a3460231e6b6379819ab34cfd99b51c11a114a"],  # #65 GDN -> qwen3_5
        ["raise", "5c181811a084f5208586d91c742273c9e0daa627da6a012ab5a3be6cfcdb6ce3"],
        ["raise", "726b9ea19fdfccb7dfd45d3ef9b4549f158559fa0b47fbf01f761e1d00737e58"],
    ],
    "test_miles_dense_full_parameter_argv_and_runtime_contract": [
        ["argv", "7e5f890e1356602e3dfe08bb5510c65d8b83e8ed9cab365e40070dafc57d639f"],
    ],
}

CASES = {
    "test_miles_argv_uses_provider_capabilities_without_model_family_branches": (
        "monkeypatch",
    ),
    "test_miles_dense_full_parameter_argv_and_runtime_contract": (
        "monkeypatch",
        "tmp_path",
    ),
    "test_miles_argv_preserves_mrope_provider_configuration": (),
}


def _digest(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize("name", sorted(CASES))
def test_legacy_argv_is_identical_to_pre_split_snapshot(name, tmp_path):
    real = rl_learner.build_miles_argv
    records: list[list[str]] = []
    tmp = str(tmp_path)

    def recording(*args, **kwargs):
        try:
            argv = real(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - errors are part of the contract
            records.append(["raise", _digest([type(exc).__name__, str(exc).replace(tmp, "<TMP>")])])
            raise
        records.append(["argv", _digest([item.replace(tmp, "<TMP>") for item in argv])])
        return argv

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(rl_learner, "build_miles_argv", recording)
        mp.setattr(launcher_tests, "build_miles_argv", recording)
        fixtures = {"tmp_path": tmp_path}
        with pytest.MonkeyPatch.context() as inner:
            fixtures["monkeypatch"] = inner
            getattr(launcher_tests, name)(**{k: fixtures[k] for k in CASES[name]})

    if os.environ.get("YETO_ARGV_SNAPSHOT_PRINT"):
        print(f"\nGOLDEN[{name!r}] = {json.dumps(records)}")
    assert records, "no build_miles_argv calls recorded"
    assert records == GOLDEN[name]
