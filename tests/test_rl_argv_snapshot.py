"""Snapshot guard for the build_miles_argv split (rl-engine-ports task 2.4).

Every ``build_miles_argv`` call made by the existing launcher tests is
recorded (argv or raised error) and compared with a golden digest captured
from the unsplit function on main (originally c40a32c; regenerated at
cb55aca after #65 (unchanged at e21a7ff) intentionally changed the GDN-provider argv to the
``qwen3_5`` recipe -- the only entry that changed).  Paths under ``tmp_path`` are
normalized so the digests are machine independent.

fix-decoupled-lr-schedule intentionally regenerated ``GOLDEN``: every training
argv gained the four explicit LR-schedule flags.  ``PRE_LR_FIX_GOLDEN`` keeps
the previous digests and every argv, with those four flags stripped, must still
match it item for item.

Regenerate (only when an argv change is intended):
    YETO_ARGV_SNAPSHOT_PRINT=1 python -m pytest -q -s tests/test_rl_argv_snapshot.py
"""

from __future__ import annotations

import hashlib
import json
import os

import pytest

from yeto.rl import learner as rl_learner
from yeto.rl.engine import run_config as rc
from yeto.rl.engine.algorithm import AlgorithmSpec
from yeto.rl.engine.miles_adapter import config as mc

import test_rl_launcher as launcher_tests

# test name -> [(kind, sha256 of normalized payload), ...] in call order.
# Pre-LR-fix golden (18695ae).  fix-decoupled-lr-schedule only adds the four
# explicit LR-schedule flags; with them stripped every argv must still match.
PRE_LR_FIX_GOLDEN: dict[str, list[list[str]]] = {
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

# Current golden (regenerated for fix-decoupled-lr-schedule 1.3).
GOLDEN: dict[str, list[list[str]]] = {
    "test_miles_argv_preserves_mrope_provider_configuration": [
        ["argv", "2cfe3a98ffb56f5d7baa198c0859d61b1713c18be22df0f369923d2d282e0329"],
    ],
    "test_miles_argv_uses_provider_capabilities_without_model_family_branches": [
        ["argv", "c0bad3de502339c1fdcc99cf6ae209c26153201cc8c221dbbe7b43f0096fe7e4"],
        ["argv", "e4e9e0b0b848e2106208259d1f434f69162e15a3b785d1a50d08d04786de1acd"],
        ["argv", "711345dc1dd05843e0f2123c2e4a2a93cbfc6055314880b1fd9bac7cc2b0ef36"],
        ["argv", "6707c826d4923886848a54eaf4d145d33c2bc0e451e2143ec1a219b36fbf340d"],
        ["raise", "248821ecfaf8627a9e55bb9b7f52052194df4125eb15baf0c92d3af83d00658c"],
        ["argv", "f54782056dbffc5fe1ba05cd694a3241ce449638200e76ba765ed990ca078cd1"],
        ["argv", "8588e3e4b2093acee75e2fcd1be7eff7b002ff73961694f04b8b80bdb6ef2d64"],
        ["argv", "e085a72320b7b835fadde444d285356bbbd2ab6c2a9fdd8af6506311bdf70908"],
        ["argv", "3bd2b1cd1571631981fd684ad8e9181c6cf758e58d8b0ecd948ebc97117ad00d"],
        ["argv", "b2e1ea3cfc2a405cf2718245f260f2a435e878da78e6d31ae2c80259f5df63a2"],
        ["argv", "dbc96fed6480e775af384f6e4c4b11bd7f1d8e9512803a457534f4997944fbcb"],
        ["raise", "5c181811a084f5208586d91c742273c9e0daa627da6a012ab5a3be6cfcdb6ce3"],
        ["raise", "726b9ea19fdfccb7dfd45d3ef9b4549f158559fa0b47fbf01f761e1d00737e58"],
    ],
    "test_miles_dense_full_parameter_argv_and_runtime_contract": [
        ["argv", "918e66056fadaebd6ddc68c0c0fae934680e98da965b1dc2ae0954b20b0221fc"],
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


def _strip_lr_schedule(argv: list[str]) -> list[str]:
    out, it = [], iter(argv)
    for item in it:
        if item in rc.LR_SCHEDULE_FLAGS:
            next(it)
            continue
        out.append(item)
    return out


def _lr_schedule_values(argv) -> dict[str, str]:
    argv = list(argv)
    return {
        flag: argv[argv.index(flag) + 1] for flag in rc.LR_SCHEDULE_FLAGS if flag in argv
    }


def _digest(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize("name", sorted(CASES))
def test_legacy_argv_is_identical_to_pre_split_snapshot(name, tmp_path):
    real = rl_learner.build_miles_argv
    records: list[list[str]] = []
    stripped: list[list[str]] = []
    compared: list[dict[str, str]] = []
    tmp = str(tmp_path)

    def recording(*args, **kwargs):
        try:
            argv = real(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - errors are part of the contract
            entry = ["raise", _digest([type(exc).__name__, str(exc).replace(tmp, "<TMP>")])]
            records.append(entry)
            stripped.append(entry)
            raise
        norm = [item.replace(tmp, "<TMP>") for item in argv]
        records.append(["argv", _digest(norm)])
        stripped.append(["argv", _digest(_strip_lr_schedule(norm))])
        legacy = _lr_schedule_values(argv)
        # Same RLRunConfig through the ports translation: identical schedule.
        config = rc.resolve_rl_run_config(*args[:1], **kwargs)
        assert legacy == _lr_schedule_values(rc.lr_schedule_argv(config.algorithm.lr_schedule))
        try:
            ports = mc.translate_run_config(config, AlgorithmSpec()).argv
        except mc.UnmappedConfigError:
            pass  # legacy-only config (custom agent, DeepSeek V4, full params, fork ports)
        else:
            assert _lr_schedule_values(ports) == legacy
            compared.append(legacy)
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
        print(f"compared legacy/ports schedules: {compared}")
    assert records, "no build_miles_argv calls recorded"
    # Only the four LR-schedule flags differ from the pre-fix argv.
    assert stripped == PRE_LR_FIX_GOLDEN[name]
    assert records == GOLDEN[name]
    for entry in compared:
        assert set(entry) == set(rc.LR_SCHEDULE_FLAGS)


@pytest.mark.parametrize(
    "overrides, expected",
    [
        (dict(sync_preset="strict-avg", global_rounds=3, optimizer_steps=1),
         ["linear", "3", "0", "0"]),
        (dict(sync_preset="strict-avg", global_rounds=3, optimizer_steps=2),
         ["linear", "6", "0", "0"]),
        (dict(sync_preset="decoupled", global_rounds=4, optimizer_steps=1),
         ["constant", "4", "0", "0"]),
        (dict(sync_preset="decoupled", global_rounds=4, optimizer_steps=2),
         ["constant", "8", "0", "0"]),
        (dict(sync_preset="strict-avg", eval_only=True), None),
    ],
)
def test_legacy_and_ports_emit_identical_lr_schedule(overrides, expected):
    """Same RLRunConfig -> identical four LR-schedule flags on both engines."""

    import argparse

    captured = []
    real = rl_learner.build_miles_argv

    def capture(*args, **kwargs):
        captured.append((args, kwargs))
        return real(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(launcher_tests, "build_miles_argv", capture)
        launcher_tests.test_miles_argv_preserves_mrope_provider_configuration()
    (args,), kwargs = captured[0]
    args = argparse.Namespace(**{**vars(args), **overrides})

    legacy = rl_learner.build_miles_argv(args, **kwargs)
    config = rc.resolve_rl_run_config(args, **kwargs)
    ports = mc.translate_run_config(config, AlgorithmSpec()).argv
    assert _lr_schedule_values(legacy) == _lr_schedule_values(ports)
    if expected is None:
        assert _lr_schedule_values(legacy) == {}
    else:
        assert [_lr_schedule_values(legacy)[f] for f in rc.LR_SCHEDULE_FLAGS] == expected


def _captured_args():
    captured = []
    real = rl_learner.build_miles_argv

    def capture(*args, **kwargs):
        captured.append((args, kwargs))
        return real(*args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(launcher_tests, "build_miles_argv", capture)
        launcher_tests.test_miles_argv_preserves_mrope_provider_configuration()
    return captured[0]


def test_standby_gpus_need_a_lora_fixed_partition():
    """rl-infra-spec 2.1 / review F7: standby is never silently dropped."""

    import argparse

    (args,), kwargs = _captured_args()
    base = vars(args)
    part = argparse.Namespace(**{**base, "rl_placement": "fixed-partition",
                                 "rollout_num_gpus": 2, "rl_standby_gpus": 2})
    config = rc.resolve_rl_run_config(part, **kwargs)
    assert config.parallel.standby_gpus == 2
    assert config.parallel.dedicated_rollout_gpus == 2
    colo = argparse.Namespace(**{**base, "rl_standby_gpus": 1})
    with pytest.raises(ValueError, match="rl-placement fixed-partition"):
        rc.resolve_rl_run_config(colo, **kwargs)
    full = argparse.Namespace(**{**base, "parameter_mode": "full", "rl_standby_gpus": 1})
    with pytest.raises(ValueError, match="not supported for full-parameter"):
        rc.resolve_rl_run_config(full, **kwargs)
