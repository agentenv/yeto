from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
from typing import Callable

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_pti_online_pair.py"
SPEC = importlib.util.spec_from_file_location("validate_pti_online_pair", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _responder(learner: int, fragment_visit: int) -> dict[str, int | float]:
    c_steps = 16
    c_tokens = 128 + learner
    return {
        "id": learner,
        "base_version": fragment_visit,
        "c_steps": c_steps,
        "c_tokens": c_tokens,
        "weight": c_tokens**2 / c_steps,
    }


def _inactive_pti() -> dict[str, object]:
    return {
        "pti_shadow_score": None,
        "pti_score_count": 0,
        "pti_interlock_open": False,
        "pti_used_nonstock": False,
        "pti_state_cleared": False,
        "pti_reason": None,
        "pti_stock_sha256": None,
        "pti_previous_stock_sha256": None,
        "pti_candidate_sha256": None,
        "pti_action_sha256": None,
    }


def _rows(*, pti: bool, learners: int = 4) -> list[dict]:
    rows: list[dict] = []
    previous: dict[int, str] = {}
    for commit in range(1, 33):
        fragment = (commit - 1) % 4
        visit = (commit - 1) // 4
        row = {
            "step": commit,
            "fragment": fragment,
            "commit_seq": commit,
            "commit_elapsed_ns": commit * 1_000_000,
            "responders": [_responder(learner, visit) for learner in range(learners)],
        }
        if not pti:
            row.update(_inactive_pti())
            rows.append(row)
            continue
        stock = _digest(f"stock-{fragment}-{visit}")
        if visit == 0:
            row.update(
                {
                    "pti_shadow_score": None,
                    "pti_score_count": 0,
                    "pti_interlock_open": False,
                    "pti_used_nonstock": False,
                    "pti_state_cleared": False,
                    "pti_reason": "warmup",
                    "pti_stock_sha256": stock,
                    "pti_previous_stock_sha256": None,
                    "pti_candidate_sha256": None,
                    "pti_action_sha256": stock,
                }
            )
        else:
            candidate = _digest(f"candidate-{fragment}-{visit}")
            score = None if visit == 1 else 0.01 * visit
            score_count = min(3, max(0, visit - 1))
            selected = score_count == 3
            row.update(
                {
                    "pti_shadow_score": score,
                    "pti_score_count": score_count,
                    "pti_interlock_open": selected,
                    "pti_used_nonstock": selected,
                    "pti_state_cleared": False,
                    "pti_reason": (
                        "candidate_selected" if selected else "interlock_closed"
                    ),
                    "pti_stock_sha256": stock,
                    "pti_previous_stock_sha256": previous[fragment],
                    "pti_candidate_sha256": candidate,
                    "pti_action_sha256": candidate if selected else stock,
                }
            )
        previous[fragment] = stock
        rows.append(row)
    return rows


def _checkpoint(path: Path) -> None:
    # Only the authoritative header is needed by this validator; extra bytes
    # stand in for the fragment tensors, ledgers, and optional PTI extension.
    path.write_bytes(struct.pack("<IQI", 0xD1705A7E, 32, 4) + b"checkpoint-body")


def _fixture(tmp_path: Path) -> dict[str, Path | str]:
    stock = tmp_path / "work" / "pti_m4_stock"
    pti = tmp_path / "work" / "pti_m4_candidate"
    stock.mkdir(parents=True)
    pti.mkdir(parents=True)
    results = tmp_path / "report" / "results.jsonl"
    _write_jsonl(
        results,
        [
            {"arm": "base (untrained)", "m": 0, "wall_s": 0.0, "eval_loss": 2.0},
            {"arm": "pti_m4_stock", "m": 4, "wall_s": 20.0, "eval_loss": 1.25},
            {
                "arm": "pti_m4_candidate",
                "m": 4,
                "wall_s": 21.0,
                "eval_loss": 1.20,
            },
        ],
    )
    _write_jsonl(stock / "tape.jsonl", _rows(pti=False))
    _write_jsonl(pti / "tape.jsonl", _rows(pti=True))
    _checkpoint(stock / "state.ckpt")
    _checkpoint(pti / "state.ckpt")
    return {
        "results": results,
        "stock_arm_dir": stock,
        "pti_arm_dir": pti,
        "stock_arm": "pti_m4_stock",
        "pti_arm": "pti_m4_candidate",
        "output": tmp_path / "report" / "pti_online_validation.json",
    }


def _run(pair: dict[str, Path | str]) -> dict:
    return MOD.run_gate(**pair)


def _rewrite(path: Path, mutate: Callable[[list[dict]], None]) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(rows)
    _write_jsonl(path, rows)


def test_pass_seals_deterministic_non_crn_summary(tmp_path: Path) -> None:
    pair = _fixture(tmp_path)
    result = _run(pair)
    assert result["status"] == "PASS"
    assert result["claim_scope"] == "exploratory_online_non_crn"
    assert result["capture_v2_crn_gate_satisfied"] is False
    validation = result["validation"]
    assert validation["loss"] == {
        "stock": 1.25,
        "pti": 1.2,
        "pti_minus_stock": pytest.approx(-0.05),
        "stock_minus_pti_gain": pytest.approx(0.05),
        "pti_relative_to_stock": 0.96,
    }
    assert validation["actions"]["candidate_actions"] == 16
    assert validation["actions"]["action_fraction"] == 0.5
    assert set(validation["per_fragment"]) == {"0", "1", "2", "3"}
    assert all(
        fragment["commits"] == 8 for fragment in validation["per_fragment"].values()
    )

    output = pair["output"]
    assert isinstance(output, Path)
    first_raw = output.read_bytes()
    first_sidecar = output.with_name(output.name + ".sha256").read_bytes()
    repeated = _run(pair)
    assert repeated["artifact_sha256"] == result["artifact_sha256"]
    assert output.read_bytes() == first_raw
    assert output.with_name(output.name + ".sha256").read_bytes() == first_sidecar
    assert first_sidecar == (
        f"{hashlib.sha256(first_raw).hexdigest()}  {output.name}\n".encode()
    )


@pytest.mark.parametrize(
    ("target", "mutate", "error"),
    [
        (
            "pti",
            lambda rows: rows[16].__setitem__(
                "pti_action_sha256", rows[16]["pti_stock_sha256"]
            ),
            "selected action hash is not candidate hash",
        ),
        (
            "pti",
            lambda rows: rows[20].__setitem__("pti_interlock_open", False),
            "interlock False != three-positive rule True",
        ),
        (
            "stock",
            lambda rows: rows[0].__setitem__("pti_used_nonstock", True),
            "stock field pti_used_nonstock must be False",
        ),
        (
            "pti",
            lambda rows: rows[31].__setitem__("fragment", 0),
            "fragments are not exactly balanced",
        ),
        (
            "pti",
            lambda rows: rows[5]["responders"][0].__setitem__("c_tokens", 999),
            "weight",
        ),
    ],
)
def test_fail_closed_on_tape_corruption(
    tmp_path: Path,
    target: str,
    mutate: Callable[[list[dict]], None],
    error: str,
) -> None:
    pair = _fixture(tmp_path)
    arm_dir = pair["pti_arm_dir" if target == "pti" else "stock_arm_dir"]
    assert isinstance(arm_dir, Path)
    _rewrite(arm_dir / "tape.jsonl", mutate)
    result = _run(pair)
    assert result["status"] == "FAIL"
    assert error in result["errors"][0]
    output = pair["output"]
    assert isinstance(output, Path)
    durable = json.loads(output.read_text())
    assert durable["status"] == "FAIL"
    assert durable["validation"] is None
    assert output.with_name(output.name + ".sha256").is_file()


def test_rejects_no_nonstock_actions_nonfinite_loss_and_missing_checkpoint(
    tmp_path: Path,
) -> None:
    pair = _fixture(tmp_path)
    pti_dir = pair["pti_arm_dir"]
    assert isinstance(pti_dir, Path)

    def close_every_action(rows: list[dict]) -> None:
        for row in rows:
            if row["pti_used_nonstock"]:
                row["pti_used_nonstock"] = False
                row["pti_interlock_open"] = False
                row["pti_reason"] = "interlock_closed"
                row["pti_action_sha256"] = row["pti_stock_sha256"]
                # Make the causal rule genuinely closed rather than merely
                # lying in the boolean fields.
                row["pti_shadow_score"] = -abs(row["pti_shadow_score"])

    _rewrite(pti_dir / "tape.jsonl", close_every_action)
    first = _run(pair)
    assert first["status"] == "FAIL"
    assert "PTI tape selected no nonstock actions" in first["errors"][0]

    pair = _fixture(tmp_path / "nonfinite")
    results = pair["results"]
    assert isinstance(results, Path)
    raw = results.read_text().replace('"eval_loss": 1.2', '"eval_loss": 1e999')
    results.write_text(raw)
    second = _run(pair)
    assert second["status"] == "FAIL"
    assert "non-finite JSON number" in second["errors"][0]

    pair = _fixture(tmp_path / "missing")
    checkpoint = pair["pti_arm_dir"] / "state.ckpt"
    checkpoint.unlink()
    third = _run(pair)
    assert third["status"] == "FAIL"
    assert "missing PTI final checkpoint" in third["errors"][0]
