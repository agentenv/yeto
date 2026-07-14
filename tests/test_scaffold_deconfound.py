from array import array
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/scaffold_deconfound.py"
SPEC = importlib.util.spec_from_file_location("scaffold_deconfound", SCRIPT)
scaffold_deconfound = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scaffold_deconfound)


def _write_f32(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        array("f", values).tofile(handle)


def test_fit_aligns_fragment_version_and_computes_scale(tmp_path):
    lite_root = tmp_path / "lite"
    sgd_root = tmp_path / "sgd"
    lite_rows = []
    sgd_rows = []
    for fragment, version, lite, sgd in (
        (0, 1, [2.0, 4.0], [1.0, 2.0]),
        (1, 2, [6.0], [3.0]),
    ):
        lite_name = f"updates/lite-{fragment}-{version}.f32"
        sgd_name = f"updates/sgd-{fragment}-{version}.f32"
        _write_f32(lite_root / lite_name, lite)
        _write_f32(sgd_root / sgd_name, sgd)
        lite_rows.append(
            {"fragment": fragment, "version": version,
             "applied_update_f32": lite_name}
        )
        sgd_rows.append(
            {"fragment": fragment, "version": version,
             "applied_update_f32": sgd_name}
        )

    result = scaffold_deconfound.fit(
        lite_root, lite_rows, sgd_root, sgd_rows
    )
    assert result["aligned_commits"] == 2
    assert result["s_star"] == pytest.approx(2.0)
    assert result["eta_match"] == pytest.approx(0.56)
    assert result["r_E"] == pytest.approx(2.0)


def test_legacy_capture_is_rejected_without_fabricating_fit(tmp_path, capsys):
    arms = {
        "lite": (tmp_path, [{"schema": "syncer_probe_capture_v1"}]),
        "sgd": (tmp_path, [{"schema": "syncer_probe_capture_v1"}]),
    }
    with pytest.raises(SystemExit) as error:
        scaffold_deconfound._check_update_vectors(arms)
    assert error.value.code == 2
    assert "applied_update_f32" in capsys.readouterr().err
