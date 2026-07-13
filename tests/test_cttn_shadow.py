import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_cttn_shadow", ROOT / "scripts" / "analyze_cttn_shadow.py"
)
shadow = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(shadow)


def _samples(matrix, scalar, *, retention=0.1, bind=True):
    records = []
    for index, (matrix_value, scalar_value) in enumerate(zip(matrix, scalar)):
        records.append(
            {
                "cttn_shadow_sample_step": index + 1,
                "cttn_shadow_fragment": index % 4,
                "cttn_shadow_matrix_alignment": matrix_value,
                "cttn_shadow_scalar_alignment": scalar_value,
                "cttn_shadow_retention": retention,
                "cttn_shadow_bind": bind,
            }
        )
    return records


def test_predictive_alignment_is_magnitude_aware():
    assert shadow.predictive_alignment([1.0, 0.0], [3.0, 4.0], 2.0) == 0.3
    assert shadow.predictive_alignment([2.0, 0.0], [3.0, 4.0], 2.0) == 0.6
    assert shadow.predictive_alignment([1.0], [0.0], 2.0) is None


def test_no_go_gate():
    result = shadow.decide(_samples([-0.1] * 32, [-0.2] * 32))
    assert result["decision"] == "NO-GO"
    assert result["bind_count"] == 32
    assert result["nonpositive_fragment_means"] == 4


def test_trigger_gate_requires_paired_matrix_win():
    matrix = [0.3] * 24 + [-0.1] * 8
    scalar = [value - 0.05 for value in matrix]
    result = shadow.decide(_samples(matrix, scalar, retention=0.5, bind=False))
    assert result["decision"] == "TRIGGER"
    assert result["positive_matrix_samples"] == 24
    assert result["positive_fragment_means"] == 4
    assert result["paired_matrix_minus_scalar"] > 0.0

    losing_scalar = [value + 0.05 for value in matrix]
    result = shadow.decide(
        _samples(matrix, losing_scalar, retention=0.5, bind=False)
    )
    assert result["decision"] == "INCONCLUSIVE"
