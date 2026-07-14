from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any, Callable

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "validate_cplg_online_pair.py"
FROZEN_CONFIG = (
    ROOT / "experiments" / "optimizer" / ("cplg-sgd-active-e1-r1-config.json")
)
SPEC = importlib.util.spec_from_file_location("validate_cplg_online_pair", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


SOURCE_COMMIT = "a" * 40
LAYOUT_SHA256 = "b" * 64
INITIAL_STATE_SHA256 = "c" * 64


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _publish(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest


def _write_jsonl(
    path: Path, rows: list[dict[str, Any]], *, canonical: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoder = (
        _canonical
        if canonical
        else lambda row: json.dumps(row, sort_keys=True).encode()
    )
    path.write_bytes(b"".join(encoder(row) + b"\n" for row in rows))


def _common_tape_row(index: int) -> dict[str, Any]:
    sequence = index + 1
    fragment = index % 4
    visit = index // 4
    return {
        "step": sequence,
        "fragment": fragment,
        "commit_seq": sequence,
        "commit_elapsed_ns": sequence * 1_000,
        "gnorm": 1.0,
        "ms": sequence,
        "responders": [
            {
                "id": 0,
                "base_version": visit,
                "c_steps": 4,
                "c_tokens": 512,
                "weight": 65_536.0,
            }
        ],
        "outer_step_norm": 0.25,
        "outer_direction_cosine": None if index == 0 else 0.5,
        "outer_history_current_ratio": None if index == 0 else 0.5,
        "outer_restarted": False,
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
        "policy": "token_weighted",
        "selected_action": "A0",
        "committed_action": "A0",
        "selected_multiplier": 1.0,
        "committed_multiplier": 1.0,
        "fallback": False,
        "fallback_reason": None,
        "probe_latency_ms": None,
        "selected_mass": 1.0,
        "norm_scale": 1.0,
        "step_ratio": 1.0,
        "request_digest": None,
    }


def _cplg_boundary(
    *,
    index: int,
    histories: dict[int, list[float]],
    active_fragments: set[int],
    one_action_per_fragment: bool,
) -> dict[str, Any]:
    fragment = index % 4
    visit = index // 4
    stock = _digest(f"stock-{fragment}-{visit}")
    previous_stock = None if visit == 0 else _digest(f"stock-{fragment}-{visit - 1}")
    if visit == 0:
        reason = "stock_warmup"
        shadow = None
        candidate = None
        previous_tangent = None
        transported_tangent = None
        rho = theta = previous_theta = coherence = phi = None
    elif visit == 1:
        reason = "phase_warmup"
        shadow = None
        candidate = None
        previous_tangent = None
        transported_tangent = None
        rho = theta = coherence = 0.5
        previous_theta = phi = None
    else:
        should_be_positive = fragment in active_fragments
        if one_action_per_fragment and visit >= 5:
            should_be_positive = False
        shadow = 0.25 if should_be_positive else -0.25
        history = histories[fragment]
        history.append(shadow)
        del history[:-3]
        open_interlock = len(history) == 3 and all(score > 0.0 for score in history)
        reason = "candidate_selected" if open_interlock else "interlock_closed"
        candidate = _digest(f"candidate-{fragment}-{visit}")
        previous_tangent = _digest(f"tangent-{fragment}-{visit - 1}")
        transported_tangent = _digest(f"transported-{fragment}-{visit}")
        rho = theta = previous_theta = coherence = phi = 0.5
    history = histories[fragment]
    interlock_open = len(history) == 3 and all(score > 0.0 for score in history)
    used = reason == "candidate_selected"
    return {
        "cplg_rho": rho,
        "cplg_theta": theta,
        "cplg_previous_theta": previous_theta,
        "cplg_coherence": coherence,
        "cplg_phi": phi,
        "cplg_shadow_score": shadow,
        "cplg_score_count": len(history),
        "cplg_interlock_open": interlock_open,
        "cplg_used_nonstock": used,
        "cplg_state_cleared": False,
        "cplg_reason": reason,
        "cplg_stock_sha256": stock,
        "cplg_previous_stock_sha256": previous_stock,
        "cplg_previous_tangent_sha256": previous_tangent,
        "cplg_transported_tangent_sha256": transported_tangent,
        "cplg_candidate_sha256": candidate,
        "cplg_action_sha256": candidate if used else stock,
    }


def _candidate_rows(
    *, active_fragments: set[int], one_action_per_fragment: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    histories = {fragment: [] for fragment in range(4)}
    tapes: list[dict[str, Any]] = []
    ledgers: list[dict[str, Any]] = []
    predecessor = "0" * 64
    for index in range(32):
        cplg = _cplg_boundary(
            index=index,
            histories=histories,
            active_fragments=active_fragments,
            one_action_per_fragment=one_action_per_fragment,
        )
        tape = _common_tape_row(index)
        tape.update(cplg)
        tapes.append(tape)
        fragment = index % 4
        visit = index // 4
        ledger = {
            "schema_version": 1,
            "row_index": index,
            "run_id": MOD.RUN_ID,
            "run_config_sha256": MOD.RUN_CONFIG_SHA256,
            "source_commit": SOURCE_COMMIT,
            "commit_sequence": index,
            "fragment": fragment,
            "fragment_version": visit,
            "responder_step": (visit + 1) * 4,
            "responder_tokens": (visit + 1) * 512,
            "weight_identity_sha256": _digest(f"weight-{index}"),
            "layout_sha256": LAYOUT_SHA256,
            "initial_state_sha256": INITIAL_STATE_SHA256,
            **cplg,
            "previous_row_sha256": predecessor,
        }
        row_digest = hashlib.sha256(_canonical(ledger)).hexdigest()
        ledger["row_sha256"] = row_digest
        predecessor = row_digest
        ledgers.append(ledger)
    return tapes, ledgers, predecessor


def _stock_checkpoint(path: Path) -> None:
    path.write_bytes(struct.pack("<IQI", MOD.CKPT_MAGIC, 32, 4) + b"stock-body")


def _candidate_checkpoint(path: Path, ledger_head: str) -> None:
    raw = bytearray(struct.pack("<IQI", MOD.CKPT_MAGIC, 32, 4))
    for _ in range(4):
        raw.extend(struct.pack("<QQff", 8, 1, 1.0, 0.0))
    raw.extend(struct.pack("<IIQQQ", 1, 0, 32, 128, 16_384))
    raw.extend(struct.pack("<I", 0))
    raw.extend(struct.pack("<II", MOD.CPLG_CKPT_EXTENSION_MAGIC, 4))
    for _ in range(4):
        for value in (1.0, 1.0, 1.0):
            raw.extend(struct.pack("<Qf", 1, value))
        raw.extend(struct.pack("<Bf", 1, 0.5))
        raw.extend(struct.pack("<Ifff", 3, 0.25, 0.25, 0.25))
    raw.extend(struct.pack("<Q", 32))
    raw.extend(bytes.fromhex(ledger_head))
    path.write_bytes(raw)


def _initial(arm: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": MOD.RUN_ID,
        "run_config_sha256": MOD.RUN_CONFIG_SHA256,
        "source_commit": SOURCE_COMMIT,
        "arm": arm,
        "outer_optimizer": "nesterov" if arm == MOD.STOCK_ARM else "cplg-sgd",
        "layout_sha256": LAYOUT_SHA256,
        "initial_state_sha256": INITIAL_STATE_SHA256,
        "fragments": 4,
        "expected_commits": 32,
    }


def _learner_completion() -> dict[str, Any]:
    return {
        "schema": MOD.LEARNER_COMPLETION_SCHEMA,
        "learner_id": 0,
        "local_step": 34,
        "raw_tokens": 4_352,
        "global_step": 32,
        "reconnect_count": 0,
        "terminal_status": "syncer_shutdown",
    }


def _completion(
    *, arm: str, tape: Path, checkpoint: Path, interval_ns: int, ledger_head: str
) -> dict[str, Any]:
    candidate = arm == MOD.CANDIDATE_ARM
    return {
        "schema_version": 1,
        "run_id": MOD.RUN_ID,
        "arm": arm,
        "terminal_local_steps": 34,
        "raw_training_tokens": 4_352,
        "final_global_step": 32,
        "commits_observed": 32,
        "commits_per_fragment": [8, 8, 8, 8],
        "interval_start_ns": 10_000,
        "interval_end_ns": 10_000 + interval_ns,
        "interval_ns": interval_ns,
        "event_tape_sha256": hashlib.sha256(tape.read_bytes()).hexdigest(),
        "final_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "ledger_head": ledger_head if candidate else None,
        "ledger_rows": 32 if candidate else None,
        "writer_dropped": 0,
        "writer_abandoned": 0,
        "writer_pending": 0,
        "writer_errors": 0,
    }


def _ledger_manifest(root: Path, *, ledger_head: str) -> dict[str, Any]:
    arm = root / "work" / MOD.CANDIDATE_ARM
    return {
        "schema_version": 1,
        "run_id": MOD.RUN_ID,
        "run_config_sha256": MOD.RUN_CONFIG_SHA256,
        "source_commit": SOURCE_COMMIT,
        "arm": MOD.CANDIDATE_ARM,
        "layout_sha256": LAYOUT_SHA256,
        "initial_state_sha256": INITIAL_STATE_SHA256,
        "ledger_rows": 32,
        "ledger_head": ledger_head,
        "final_checkpoint_sha256": hashlib.sha256(
            (arm / "state.ckpt").read_bytes()
        ).hexdigest(),
        "event_tape_sha256": hashlib.sha256(
            (arm / "tape.jsonl").read_bytes()
        ).hexdigest(),
        "expected_commits": 32,
        "fragments": 4,
        "outer_optimizer": "cplg-sgd",
        "unresolved_tail": 4,
        "writer_dropped": 0,
        "writer_abandoned": 0,
        "writer_pending": 0,
        "writer_errors": 0,
    }


def _refresh_manifest(root: Path) -> Path:
    report = root / "report"
    manifest_path = report / "acquisition_manifest.json"
    terminal_path = report / "acquisition_terminal.json"
    excluded = {
        manifest_path,
        Path(f"{manifest_path}.sha256"),
        terminal_path,
        Path(f"{terminal_path}.sha256"),
    }
    files = []
    for path in sorted(root.rglob("*")):
        if path in excluded or path.is_dir():
            continue
        metadata = path.lstat()
        assert stat_is_regular(metadata.st_mode)
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": metadata.st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    total_bytes = sum(item["bytes"] for item in files)
    manifest = {
        "schema": MOD.MANIFEST_SCHEMA,
        "status": "ACQUIRED",
        "run_id": MOD.RUN_ID,
        "source_commit": SOURCE_COMMIT,
        "run_config_sha256": MOD.RUN_CONFIG_SHA256,
        "arms": {"stock": MOD.STOCK_ARM, "candidate": MOD.CANDIDATE_ARM},
        "files": files,
        "file_count": len(files),
        "total_bytes": total_bytes,
    }
    manifest_digest = _publish(manifest_path, manifest)
    _publish(
        terminal_path,
        {
            "schema": MOD.ACQUISITION_TERMINAL_SCHEMA,
            "status": "GPU_ACQUISITION_COMPLETE",
            "run_id": MOD.RUN_ID,
            "source_commit": SOURCE_COMMIT,
            "run_config_sha256": MOD.RUN_CONFIG_SHA256,
            "acquisition_manifest": manifest_path.name,
            "acquisition_manifest_sha256": manifest_digest,
            "file_count": len(files),
            "total_bytes": total_bytes,
            "gpu_analysis_performed": False,
            "scientific_verdict": None,
            "next_action": "round_trip_verify_delete_gpu_then_cpu_validate",
        },
    )
    return manifest_path


def stat_is_regular(mode: int) -> bool:
    return (mode & 0o170000) == 0o100000


def _fixture(
    tmp_path: Path,
    *,
    stock_loss: float = 1.0,
    candidate_loss: float = 1.0,
    stock_interval: int = 100_000,
    candidate_interval: int = 102_000,
    active_fragments: set[int] | None = None,
    one_action_per_fragment: bool = False,
) -> dict[str, Path]:
    root = tmp_path / "acquisition"
    report = root / "report"
    work = root / "work"
    report.mkdir(parents=True)
    work.mkdir()
    if active_fragments is None:
        active_fragments = {0, 1, 2, 3}
    _write_jsonl(
        report / "results.jsonl",
        [
            {"arm": MOD.BASE_ARM, "m": 0, "wall_s": 0.0, "eval_loss": 2.0},
            {
                "arm": MOD.STOCK_ARM,
                "m": 1,
                "wall_s": 10.0,
                "eval_loss": stock_loss,
            },
            {
                "arm": MOD.CANDIDATE_ARM,
                "m": 1,
                "wall_s": 11.0,
                "eval_loss": candidate_loss,
            },
        ],
    )
    (report / "report.md").write_text("# frozen synthetic report\n")
    (report / "frozen_run_config.json").write_bytes(FROZEN_CONFIG.read_bytes())
    _write_jsonl(work / "train.jsonl", [{"row": index} for index in range(9)])
    _write_jsonl(work / "eval.jsonl", [{"row": index} for index in range(8)])

    candidate_tape, ledger_rows, ledger_head = _candidate_rows(
        active_fragments=active_fragments,
        one_action_per_fragment=one_action_per_fragment,
    )
    for arm in (MOD.STOCK_ARM, MOD.CANDIDATE_ARM):
        arm_dir = work / arm
        (arm_dir / "learner-0").mkdir(parents=True)
        (arm_dir / "export").mkdir()
        tape_rows = (
            [_common_tape_row(index) for index in range(32)]
            if arm == MOD.STOCK_ARM
            else candidate_tape
        )
        _write_jsonl(arm_dir / "tape.jsonl", tape_rows)
        (arm_dir / "syncer.log").write_text("closed\n")
        (arm_dir / "learner-0.log").write_text("local_step=34 tokens=4352\n")
        _publish(
            arm_dir / "learner-0" / "learner_completion.json",
            _learner_completion(),
        )
        _publish(arm_dir / "cplg_online_initial_state.json", _initial(arm))
        (arm_dir / "export" / "adapter_config.json").write_text("{}\n")
        (arm_dir / "export" / "adapter_model.safetensors").write_bytes(b"adapter")
    stock_dir = work / MOD.STOCK_ARM
    candidate_dir = work / MOD.CANDIDATE_ARM
    _write_jsonl(
        candidate_dir / "cplg_action_ledger.jsonl", ledger_rows, canonical=True
    )
    _stock_checkpoint(stock_dir / "state.ckpt")
    _candidate_checkpoint(candidate_dir / "state.ckpt", ledger_head)
    _publish(
        stock_dir / "cplg_online_completion.json",
        _completion(
            arm=MOD.STOCK_ARM,
            tape=stock_dir / "tape.jsonl",
            checkpoint=stock_dir / "state.ckpt",
            interval_ns=stock_interval,
            ledger_head=ledger_head,
        ),
    )
    _publish(
        candidate_dir / "cplg_online_completion.json",
        _completion(
            arm=MOD.CANDIDATE_ARM,
            tape=candidate_dir / "tape.jsonl",
            checkpoint=candidate_dir / "state.ckpt",
            interval_ns=candidate_interval,
            ledger_head=ledger_head,
        ),
    )
    _publish(
        candidate_dir / "cplg_action_ledger_manifest.json",
        _ledger_manifest(root, ledger_head=ledger_head),
    )
    manifest = _refresh_manifest(root)
    return {
        "root": root,
        "manifest": manifest,
        "analysis": tmp_path / "analysis",
        "results": report / "results.jsonl",
        "stock": stock_dir,
        "candidate": candidate_dir,
    }


def _run(pair: dict[str, Path]) -> dict[str, Any]:
    return MOD.run_validation(
        acquisition_manifest=pair["manifest"], analysis_dir=pair["analysis"]
    )


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _rewrite_jsonl(
    path: Path,
    mutate: Callable[[list[dict[str, Any]]], None],
    *,
    canonical: bool = False,
) -> None:
    values = _rows(path)
    mutate(values)
    _write_jsonl(path, values, canonical=canonical)


def _refresh_candidate_bindings(pair: dict[str, Path]) -> None:
    root = pair["root"]
    candidate = pair["candidate"]
    ledger_rows = _rows(candidate / "cplg_action_ledger.jsonl")
    predecessor = "0" * 64
    tape_rows = _rows(candidate / "tape.jsonl")
    for row, tape in zip(ledger_rows, tape_rows, strict=True):
        row["previous_row_sha256"] = predecessor
        for field in MOD.CPLG_FIELDS:
            tape[field] = row[field]
        digest_input = dict(row)
        digest_input.pop("row_sha256", None)
        predecessor = hashlib.sha256(_canonical(digest_input)).hexdigest()
        row["row_sha256"] = predecessor
    _write_jsonl(candidate / "cplg_action_ledger.jsonl", ledger_rows, canonical=True)
    _write_jsonl(candidate / "tape.jsonl", tape_rows)
    _candidate_checkpoint(candidate / "state.ckpt", predecessor)
    completion_path = candidate / "cplg_online_completion.json"
    completion = json.loads(completion_path.read_text())
    completion["event_tape_sha256"] = hashlib.sha256(
        (candidate / "tape.jsonl").read_bytes()
    ).hexdigest()
    completion["final_checkpoint_sha256"] = hashlib.sha256(
        (candidate / "state.ckpt").read_bytes()
    ).hexdigest()
    completion["ledger_head"] = predecessor
    _publish(completion_path, completion)
    _publish(
        candidate / "cplg_action_ledger_manifest.json",
        _ledger_manifest(root, ledger_head=predecessor),
    )
    _refresh_manifest(root)


def _refresh_tape_bindings(pair: dict[str, Path], arm: str) -> None:
    arm_dir = pair["stock"] if arm == MOD.STOCK_ARM else pair["candidate"]
    completion_path = arm_dir / "cplg_online_completion.json"
    completion = json.loads(completion_path.read_text())
    completion["event_tape_sha256"] = hashlib.sha256(
        (arm_dir / "tape.jsonl").read_bytes()
    ).hexdigest()
    _publish(completion_path, completion)
    if arm == MOD.CANDIDATE_ARM:
        manifest_path = arm_dir / "cplg_action_ledger_manifest.json"
        ledger_manifest = json.loads(manifest_path.read_text())
        ledger_manifest["event_tape_sha256"] = completion["event_tape_sha256"]
        _publish(manifest_path, ledger_manifest)
    _refresh_manifest(pair["root"])


def test_fully_valid_fixture_passes_and_seals_distinct_checksums(
    tmp_path: Path,
) -> None:
    pair = _fixture(tmp_path)
    result = _run(pair)
    assert result["verdict"] == "PASS"
    assert result["claim"] is not None
    assert result["superiority_claim"] is False
    assert result["bootstrap_performed"] is False
    analysis = pair["analysis"]
    for name in ("analysis_report.json", "terminal_verdict.json"):
        artifact = analysis / name
        raw = artifact.read_bytes()
        assert artifact.with_name(name + ".sha256").read_text() == (
            f"{hashlib.sha256(raw).hexdigest()}  {name}\n"
        )
    detail = json.loads((analysis / "analysis_report.json").read_text())["validation"]
    assert detail["ledger"]["rows"] == 32
    assert detail["ledger"]["valid_nonstock_actions"] == 16
    assert detail["gates"]["matched_interval_overhead"]["fraction"] == 0.02
    assert detail["gates"]["matched_observed_work"] == {
        "arms": [MOD.STOCK_ARM, MOD.CANDIDATE_ARM],
        "observed": {
            "learner_id": 0,
            "local_step": 34,
            "raw_tokens": 4_352,
            "global_step": 32,
            "reconnect_count": 0,
            "terminal_status": "syncer_shutdown",
        },
        "passed": True,
    }


def _learner_receipt(pair: dict[str, Path], arm: str) -> Path:
    arm_dir = pair["stock"] if arm == MOD.STOCK_ARM else pair["candidate"]
    return arm_dir / "learner-0" / "learner_completion.json"


def _mutate_learner_receipt(pair: dict[str, Path], arm: str, **changes: Any) -> None:
    path = _learner_receipt(pair, arm)
    value = json.loads(path.read_text())
    value.update(changes)
    _publish(path, value)


@pytest.mark.parametrize(
    ("field", "observed", "asserted", "learner_field"),
    [
        ("terminal_local_steps", 33, 34, "local_step"),
        ("raw_training_tokens", 4_351, 4_352, "raw_tokens"),
    ],
)
def test_rejects_syncer_work_not_corroborated_by_observed_learner_work(
    tmp_path: Path,
    field: str,
    observed: int,
    asserted: int,
    learner_field: str,
) -> None:
    pair = _fixture(tmp_path)
    _mutate_learner_receipt(pair, MOD.STOCK_ARM, **{learner_field: observed})
    completion = json.loads((pair["stock"] / "cplg_online_completion.json").read_text())
    assert completion[field] == asserted
    _refresh_manifest(pair["root"])

    result = _run(pair)

    assert result["verdict"] != "PASS"
    assert (
        f"completion {field} does not equal observed learner {learner_field}"
        in result["errors"][0]
    )


def test_rejects_arms_with_different_observed_learner_work(tmp_path: Path) -> None:
    pair = _fixture(tmp_path)
    _mutate_learner_receipt(pair, MOD.CANDIDATE_ARM, local_step=33)
    completion_path = pair["candidate"] / "cplg_online_completion.json"
    completion = json.loads(completion_path.read_text())
    completion["terminal_local_steps"] = 33
    _publish(completion_path, completion)
    _refresh_manifest(pair["root"])

    result = _run(pair)

    assert result["verdict"] != "PASS"
    assert "arms have unequal learner observed local_step" in result["errors"][0]


def test_rejects_missing_learner_completion_receipt(tmp_path: Path) -> None:
    pair = _fixture(tmp_path)
    receipt = _learner_receipt(pair, MOD.STOCK_ARM)
    receipt.unlink()
    receipt.with_name(receipt.name + ".sha256").unlink()
    _refresh_manifest(pair["root"])

    result = _run(pair)

    assert result["verdict"] != "PASS"
    assert "learner_completion.json" in result["errors"][0]


def test_rejects_corrupt_learner_completion_sidecar(tmp_path: Path) -> None:
    pair = _fixture(tmp_path)
    receipt = _learner_receipt(pair, MOD.CANDIDATE_ARM)
    sidecar = receipt.with_name(receipt.name + ".sha256")
    sidecar.write_text(f"{'0' * 64}  {receipt.name}\n")
    _refresh_manifest(pair["root"])

    result = _run(pair)

    assert result["verdict"] != "PASS"
    assert (
        "learner_completion.json checksum sidecar digest mismatch"
        in result["errors"][0]
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("terminal_status", "max_local_steps_reached"),
        ("reconnect_count", 1),
    ],
)
def test_rejects_invalid_observed_learner_termination(
    tmp_path: Path, field: str, value: Any
) -> None:
    pair = _fixture(tmp_path)
    for arm in (MOD.STOCK_ARM, MOD.CANDIDATE_ARM):
        _mutate_learner_receipt(pair, arm, **{field: value})
    _refresh_manifest(pair["root"])

    result = _run(pair)

    assert result["verdict"] != "PASS"
    assert f"learner observed {field}" in result["errors"][0]


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_order",
        "duplicate",
        "extra",
        "missing",
    ],
)
def test_rejects_nonclosed_result_rows(tmp_path: Path, mutation: str) -> None:
    pair = _fixture(tmp_path)

    def mutate(rows: list[dict[str, Any]]) -> None:
        if mutation == "wrong_order":
            rows[1], rows[2] = rows[2], rows[1]
        elif mutation == "duplicate":
            rows[2] = dict(rows[1])
        elif mutation == "extra":
            rows.append(dict(rows[2]))
        else:
            rows.pop()

    _rewrite_jsonl(pair["results"], mutate)
    _refresh_manifest(pair["root"])
    result = _run(pair)
    assert result["verdict"] != "PASS"
    assert "results.jsonl arm order" in result["errors"][0]


def test_rejects_nonfinite_binary64_loss(tmp_path: Path) -> None:
    pair = _fixture(tmp_path)
    raw = (
        pair["results"].read_text().replace('"eval_loss": 1.0', '"eval_loss": 1e999', 1)
    )
    pair["results"].write_text(raw)
    _refresh_manifest(pair["root"])
    result = _run(pair)
    assert result["verdict"] != "PASS"
    assert "non-finite JSON number" in result["errors"][0]


@pytest.mark.parametrize("drift", ["schedule", "work"])
def test_rejects_schedule_and_work_drift(tmp_path: Path, drift: str) -> None:
    pair = _fixture(tmp_path)
    tape = pair["candidate"] / "tape.jsonl"

    def mutate(rows: list[dict[str, Any]]) -> None:
        if drift == "schedule":
            rows[9]["fragment"] = 3
        else:
            rows[9]["responders"][0]["c_tokens"] = 513

    _rewrite_jsonl(tape, mutate)
    _refresh_tape_bindings(pair, MOD.CANDIDATE_ARM)
    result = _run(pair)
    assert result["verdict"] != "PASS"
    assert drift in result["errors"][0]


@pytest.mark.parametrize("row_index", range(32))
def test_rejects_ledger_chain_corruption_at_any_position(
    tmp_path: Path, row_index: int
) -> None:
    pair = _fixture(tmp_path)
    ledger = pair["candidate"] / "cplg_action_ledger.jsonl"
    _rewrite_jsonl(
        ledger,
        lambda rows: rows[row_index].__setitem__("previous_row_sha256", "f" * 64),
        canonical=True,
    )
    _refresh_manifest(pair["root"])
    result = _run(pair)
    assert result["verdict"] != "PASS"
    assert "previous_row_sha256 mismatch" in result["errors"][0]


def test_rejects_checkpoint_embedded_head_mismatch(tmp_path: Path) -> None:
    pair = _fixture(tmp_path)
    candidate = pair["candidate"]
    checkpoint = candidate / "state.ckpt"
    raw = bytearray(checkpoint.read_bytes())
    raw[-1] ^= 1
    checkpoint.write_bytes(raw)
    completion_path = candidate / "cplg_online_completion.json"
    completion = json.loads(completion_path.read_text())
    completion["final_checkpoint_sha256"] = hashlib.sha256(raw).hexdigest()
    _publish(completion_path, completion)
    ledger_manifest_path = candidate / "cplg_action_ledger_manifest.json"
    ledger_manifest = json.loads(ledger_manifest_path.read_text())
    ledger_manifest["final_checkpoint_sha256"] = hashlib.sha256(raw).hexdigest()
    _publish(ledger_manifest_path, ledger_manifest)
    _refresh_manifest(pair["root"])
    result = _run(pair)
    assert result["verdict"] != "PASS"
    assert "checkpoint ledger head/count mismatch" in result["errors"][0]


@pytest.mark.parametrize("corruption", ["bad_fallback", "false_action"])
def test_rejects_action_hash_contract_corruption(
    tmp_path: Path, corruption: str
) -> None:
    pair = _fixture(tmp_path)
    ledger_path = pair["candidate"] / "cplg_action_ledger.jsonl"
    rows = _rows(ledger_path)
    if corruption == "bad_fallback":
        row = rows[8]
        assert row["cplg_reason"] == "interlock_closed"
        row["cplg_action_sha256"] = row["cplg_candidate_sha256"]
    else:
        row = rows[16]
        assert row["cplg_reason"] == "candidate_selected"
        row["cplg_action_sha256"] = row["cplg_stock_sha256"]
    _write_jsonl(ledger_path, rows, canonical=True)
    _refresh_candidate_bindings(pair)
    result = _run(pair)
    assert result["verdict"] != "PASS"
    assert (
        "bad exact-stock fallback" in result["errors"][0]
        or "false or malformed candidate action" in result["errors"][0]
    )


def test_complete_valid_run_with_fewer_than_eight_actions_is_fail(
    tmp_path: Path,
) -> None:
    pair = _fixture(tmp_path, one_action_per_fragment=True)
    result = _run(pair)
    assert result["verdict"] == "FAIL"
    assert result["errors"] == []
    detail = json.loads((pair["analysis"] / "analysis_report.json").read_text())[
        "validation"
    ]
    gate = detail["gates"]["valid_nonstock_actions"]
    assert gate == {"minimum": 8, "observed": 4, "passed": False}


def test_complete_valid_run_with_fewer_than_three_active_fragments_is_fail(
    tmp_path: Path,
) -> None:
    pair = _fixture(tmp_path, active_fragments={0, 1})
    result = _run(pair)
    assert result["verdict"] == "FAIL"
    detail = json.loads((pair["analysis"] / "analysis_report.json").read_text())[
        "validation"
    ]
    assert detail["gates"]["valid_nonstock_actions"]["observed"] == 8
    assert detail["gates"]["active_fragments"] == {
        "fragment_ids": [0, 1],
        "minimum": 3,
        "observed": 2,
        "passed": False,
    }


@pytest.mark.parametrize(
    ("candidate_loss", "expected"),
    [
        (0.05, "PASS"),
        (math.nextafter(0.05, math.inf), "FAIL"),
    ],
)
def test_loss_regression_gate_is_exactly_inclusive(
    tmp_path: Path, candidate_loss: float, expected: str
) -> None:
    pair = _fixture(tmp_path, stock_loss=0.0, candidate_loss=candidate_loss)
    result = _run(pair)
    assert result["verdict"] == expected


@pytest.mark.parametrize(
    ("candidate_interval", "expected"),
    [(102_000, "PASS"), (102_001, "FAIL"), (99_000, "PASS")],
)
def test_integer_nanosecond_overhead_gate_is_inclusive_and_unclamped(
    tmp_path: Path, candidate_interval: int, expected: str
) -> None:
    pair = _fixture(
        tmp_path, stock_interval=100_000, candidate_interval=candidate_interval
    )
    result = _run(pair)
    assert result["verdict"] == expected
    detail = json.loads((pair["analysis"] / "analysis_report.json").read_text())[
        "validation"
    ]
    overhead = detail["gates"]["matched_interval_overhead"]
    assert overhead["numerator_ns"] == candidate_interval - 100_000
    if candidate_interval < 100_000:
        assert overhead["fraction"] < 0.0


def test_rejects_sidecar_corruption_even_when_manifest_binds_it(tmp_path: Path) -> None:
    pair = _fixture(tmp_path)
    sidecar = pair["candidate"] / "cplg_online_initial_state.json.sha256"
    sidecar.write_text(f"{'0' * 64}  cplg_online_initial_state.json\n")
    _refresh_manifest(pair["root"])
    result = _run(pair)
    assert result["verdict"] != "PASS"
    assert "sidecar digest mismatch" in result["errors"][0]


@pytest.mark.parametrize("escape", ["symlink", "unsafe_manifest_path"])
def test_rejects_symlink_and_manifest_path_escape(tmp_path: Path, escape: str) -> None:
    pair = _fixture(tmp_path)
    if escape == "symlink":
        outside = tmp_path / "outside-results.jsonl"
        outside.write_bytes(pair["results"].read_bytes())
        pair["results"].unlink()
        pair["results"].symlink_to(outside)
    else:
        manifest_path = pair["manifest"]
        manifest = json.loads(manifest_path.read_text())
        manifest["files"][0]["path"] = "../escape"
        digest = _publish(manifest_path, manifest)
        terminal_path = pair["root"] / "report" / "acquisition_terminal.json"
        terminal = json.loads(terminal_path.read_text())
        terminal["acquisition_manifest_sha256"] = digest
        _publish(terminal_path, terminal)
    result = _run(pair)
    assert result["verdict"] != "PASS"
    assert any(word in result["errors"][0] for word in ("symlink", "unsafe path"))


def test_rejects_unlisted_stale_file(tmp_path: Path) -> None:
    pair = _fixture(tmp_path)
    (pair["root"] / "work" / "stale.tmp").write_text("old run\n")
    result = _run(pair)
    assert result["verdict"] != "PASS"
    assert "stale=['work/stale.tmp']" in result["errors"][0]


@pytest.mark.parametrize("field", ["cplg_reason", "terminal_status"])
def test_rejects_unknown_reason_and_unknown_status(tmp_path: Path, field: str) -> None:
    pair = _fixture(tmp_path)
    if field == "cplg_reason":
        ledger_path = pair["candidate"] / "cplg_action_ledger.jsonl"
        rows = _rows(ledger_path)
        rows[8]["cplg_reason"] = "invented_reason"
        _write_jsonl(ledger_path, rows, canonical=True)
        _refresh_candidate_bindings(pair)
    else:
        terminal_path = pair["root"] / "report" / "acquisition_terminal.json"
        terminal = json.loads(terminal_path.read_text())
        terminal["status"] = "MAYBE_DONE"
        _publish(terminal_path, terminal)
    result = _run(pair)
    assert result["verdict"] != "PASS"
    assert "unknown" in result["errors"][0]


@pytest.mark.parametrize(
    ("receipt", "field"),
    [
        ("stock_completion", "writer_dropped"),
        ("candidate_completion", "writer_abandoned"),
        ("candidate_completion", "writer_pending"),
        ("candidate_completion", "writer_errors"),
        ("ledger_manifest", "writer_dropped"),
        ("ledger_manifest", "writer_abandoned"),
        ("ledger_manifest", "writer_pending"),
        ("ledger_manifest", "writer_errors"),
    ],
)
def test_rejects_every_writer_nonclosure_counter(
    tmp_path: Path, receipt: str, field: str
) -> None:
    pair = _fixture(tmp_path)
    if receipt == "stock_completion":
        path = pair["stock"] / "cplg_online_completion.json"
    elif receipt == "candidate_completion":
        path = pair["candidate"] / "cplg_online_completion.json"
    else:
        path = pair["candidate"] / "cplg_action_ledger_manifest.json"
    value = json.loads(path.read_text())
    value[field] = 1
    _publish(path, value)
    _refresh_manifest(pair["root"])
    result = _run(pair)
    assert result["verdict"] != "PASS"
    assert field in result["errors"][0]


def test_analysis_prefix_must_be_fresh_and_outside_acquisition(tmp_path: Path) -> None:
    pair = _fixture(tmp_path)
    with pytest.raises(MOD.ValidationError, match="distinct prefix"):
        MOD.run_validation(
            acquisition_manifest=pair["manifest"],
            analysis_dir=pair["root"] / "analysis",
        )


def test_validator_source_is_cpu_only_and_has_no_remote_execution_surface() -> None:
    source = SCRIPT.read_text()
    for forbidden in (
        "import torch",
        "import cuda",
        "subprocess",
        "paramiko",
        "google.cloud",
    ):
        assert forbidden not in source.lower()
