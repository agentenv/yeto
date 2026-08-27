import csv
import io
import json
from pathlib import Path

import pytest

from yeto.rl.loss_curve import (
    MAX_LOG_LINE_BYTES,
    iter_metric_stream,
    merge_rows,
    parse_metric_line,
    persist_summaries,
    scan_log,
)


def test_miles_step_extracts_only_allowlisted_scalars():
    line = (
        "2026-08-15T05:04:17.029054704Z \x1b[36m"
        "(MegatronTrainRayActor pid=24290)\x1b[0m "
        "[2026-08-15 05:04:17.003 actor_cell0_rank8] "
        "log_utils.py:460 - step 7: {'train/loss': 1.25, "
        "'train/pg_loss': -0.5, "
        "'train/grad_norm': tensor(2.0), 'train/ppo_kl': 3e-2, "
        "'train/ess_ratio': 0.875, 'train/pg_clipfrac': 0.125, "
        "'train/lr-pg_0': 2e-5, 'train/step': 7, "
        "'policy_hash': 'do-not-copy', 'prompt': 'private prompt', "
        "'response': 'private response'}"
    )
    row = parse_metric_line(line, "island-0")
    assert row is not None
    assert row.step == 7
    assert row.loss == pytest.approx(1.25)
    assert row.pg_loss == pytest.approx(-0.5)
    assert row.grad_norm == pytest.approx(2.0)
    assert row.kl == pytest.approx(0.03)
    assert row.ess == pytest.approx(0.875)
    assert row.clipfrac == pytest.approx(0.125)
    assert row.lr == pytest.approx(2e-5)
    assert "hash" not in row.__dict__
    assert "prompt" not in row.__dict__
    assert "response" not in row.__dict__


def test_round_event_reward_pass_rate_and_missing_nonfinite_values():
    line = json.dumps(
        {
            "event": "rl_local_round",
            "local_round_id": 3,
            "rl/reward_mean": 0.625,
            "rl/pass_rate": 0.4,
            "rl/current_vs_rollout_kl": float("nan"),
            "rl/ess_ratio": None,
            "rl/clip_fraction": 0.2,
            "rl/policy_hash": "secret",
        }
    )
    row = parse_metric_line(line, "island-1")
    assert row is not None
    assert row.step == 3
    assert row.reward == pytest.approx(0.625)
    assert row.pass_rate == pytest.approx(0.4)
    assert row.clipfrac == pytest.approx(0.2)
    assert row.kl is None
    assert row.ess is None

    assert (
        parse_metric_line("step 1: {'train/loss': nan, 'train/step': 1}", "island-1")
        is None
    )
    assert parse_metric_line("payload {'train/loss': 1.0}", "island-1") is None
    assert parse_metric_line("step 1: {'train/loss': 1.0}", "island-1") is None
    assert parse_metric_line("x" * (MAX_LOG_LINE_BYTES + 1), "island-1") is None


def test_miles_flaky100_eval_extracts_only_exact_safe_scalar_keys():
    line = (
        "[2026-08-17 12:00:00.000 actor] metrics.py:53 - eval 11: {"
        "'eval/private_counter': 999, "
        "'eval/private-truncated_ratio': 0.99, "
        "'eval/flaky100': 0.72, "
        "'eval/flaky100-pass@1': np.float64(0.625), "
        "'eval/flaky100-truncated_ratio': 0.03125, "
        "'eval/flaky100-pass@2': 0.8, "
        "'eval/flaky100-none_reward_ratio': 0.0, "
        "'eval/flaky100/response_len/mean': 123, "
        "'unknown': {'eval/fake': 1.0}, "
        "'prompt': \"contains eval 99: {'eval/fake': 1}\", "
        "'policy_hash': 'do-not-copy'}"
    )
    row = parse_metric_line(line, "eval-island-0")
    assert row is not None
    assert row.step == 11
    assert row.eval_score == pytest.approx(0.72)
    assert row.eval_pass_at_1 == pytest.approx(0.625)
    assert row.eval_truncated_ratio == pytest.approx(0.03125)
    assert row.loss is None
    assert set(row.__dict__) == {
        "source",
        "step",
        "loss",
        "pg_loss",
        "grad_norm",
        "kl",
        "ess",
        "clipfrac",
        "lr",
        "reward",
        "pass_rate",
        "eval_score",
        "eval_pass_at_1",
        "eval_truncated_ratio",
    }


def test_eval_accepts_collected_docker_timestamp_and_ansi_ray_prefix():
    line = (
        "2026-08-17T12:00:00.123456789Z \x1b[36m"
        "(RolloutActor pid=4821)\x1b[0m "
        "[2026-08-17 12:00:00.000 actor] metrics.py:53 - eval 11: {"
        "'eval/flaky100': 0.72, "
        "'eval/flaky100-pass@1': 0.625, "
        "'eval/flaky100-truncated_ratio': 0.03125}"
    )
    row = parse_metric_line(line, "eval-island-0")
    assert row is not None
    assert row.step == 11
    assert row.eval_score == pytest.approx(0.72)
    assert row.eval_pass_at_1 == pytest.approx(0.625)
    assert row.eval_truncated_ratio == pytest.approx(0.03125)


def test_eval_requires_exact_prefix_and_drops_nonfinite_values():
    assert parse_metric_line(
        "eval 4: {'eval/flaky100': nan, "
        "'eval/flaky100-pass@1': inf, "
        "'eval/flaky100-truncated_ratio': 0.25}",
        "eval-0",
    ).eval_truncated_ratio == pytest.approx(0.25)
    assert (
        parse_metric_line(
            "eval 4: {'eval/flaky100': nan, 'eval/flaky100-pass@1': inf}",
            "eval-0",
        )
        is None
    )
    record = "eval 4: {'eval/flaky100': 0.5}"
    assert parse_metric_line("reeval 4: {'eval/flaky100': 0.5}", "eval-0") is None
    assert parse_metric_line(f"payload says {record}", "eval-0") is None
    assert parse_metric_line("eval  4: {'eval/flaky100': 0.5}", "eval-0") is None
    assert parse_metric_line("eval 4: {'EVAL/flaky100': 0.5}", "eval-0") is None
    assert parse_metric_line('eval 4: {"eval/not/safe": 0.5}', "eval-0") is None


def test_payload_text_cannot_impersonate_train_or_round_metrics():
    assert (
        parse_metric_line(
            'INFO request payload="prompt says step 7: '
            "{'train/loss': 99, 'train/step': 7}\"",
            "island-0",
        )
        is None
    )
    assert (
        parse_metric_line(
            "INFO request payload=\"{'event': 'rl_local_round', "
            "'local_round_id': 7, 'rl/reward_mean': 99}\"",
            "island-0",
        )
        is None
    )


def test_eval_output_never_contains_secrets_or_unknown_keys(tmp_path: Path):
    csv_path = tmp_path / "eval.csv"
    json_path = tmp_path / "eval.json"
    row = parse_metric_line(
        "eval 1: {'eval/flaky100': 0.5, "
        "'eval/flaky100-pass@1': 0.25, "
        "'secret_token': 'never-write-me', 'unknown_scalar': 42}",
        "eval-0",
    )
    assert row is not None
    persist_summaries([row], csv_path, json_path)
    combined = csv_path.read_text(encoding="utf-8") + json_path.read_text(
        encoding="utf-8"
    )
    assert "never-write-me" not in combined
    assert "secret_token" not in combined
    assert "unknown_scalar" not in combined


def test_eval_dataset_binding_ignores_other_complete_metric_sets():
    row = parse_metric_line(
        "eval 3: {'eval/other': 0.99, 'eval/other-pass@1': 0.99, "
        "'eval/flaky100': 0.5, 'eval/flaky100-pass@1': 0.25}",
        "eval-0",
        eval_dataset="flaky100",
    )
    assert row is not None
    assert row.eval_score == pytest.approx(0.5)
    assert row.eval_pass_at_1 == pytest.approx(0.25)
    with pytest.raises(ValueError, match="eval dataset"):
        parse_metric_line(
            "eval 3: {'eval/flaky100': 0.5}", "eval-0", eval_dataset="x/y"
        )


def test_multiple_islands_and_duplicate_steps_merge_scalar_fields():
    first = parse_metric_line(
        "step 2: {'train/loss': 1.0, 'train/step': 2}", "island-0"
    )
    duplicate = parse_metric_line(
        "step 2: {'train/loss': 0.75, 'train/grad_norm': 4.0, 'train/step': 2}",
        "island-0",
    )
    other = parse_metric_line(
        "step 2: {'train/loss': 2.0, 'train/step': 2}", "island-1"
    )
    assert first is not None and duplicate is not None and other is not None
    rows = merge_rows([first, duplicate, other])
    assert len(rows) == 2
    assert rows[0].source == "island-0"
    assert rows[0].loss == pytest.approx(0.75)
    assert rows[0].grad_norm == pytest.approx(4.0)
    assert rows[1].source == "island-1"
    assert rows[1].loss == pytest.approx(2.0)


def test_scan_skips_oversized_lines_and_preserves_partial_follow_line(tmp_path: Path):
    log = tmp_path / "learner.log"
    log.write_bytes(
        b"x" * (MAX_LOG_LINE_BYTES + 20)
        + b"\nstep 0: {'train/loss': 3.0, 'train/step': 0}\n"
        + b"step 1: {'train/loss': 2.0, 'train/step': 1}"
    )
    rows, offset = scan_log(log, "island-0", complete_lines_only=True)
    assert [(row.step, row.loss) for row in rows] == [(0, 3.0)]
    assert offset < log.stat().st_size

    with log.open("ab") as handle:
        handle.write(b"\n")
    rows, new_offset = scan_log(
        log, "island-0", offset=offset, complete_lines_only=True
    )
    assert [(row.step, row.loss) for row in rows] == [(1, 2.0)]
    assert new_offset == log.stat().st_size


def test_partial_oversized_line_never_reinterprets_its_tail(tmp_path: Path):
    log = tmp_path / "oversized.log"
    log.write_bytes(
        b"x" * (MAX_LOG_LINE_BYTES + 1)
        + b" step 9: {'train/loss': 99, 'train/step': 9}"
    )
    rows, offset = scan_log(log, "island-0", complete_lines_only=True)
    assert rows == []
    assert offset == 0

    with log.open("ab") as handle:
        handle.write(b"\nstep 0: {'train/loss': 1, 'train/step': 0}\n")
    rows, offset = scan_log(log, "island-0", offset=offset, complete_lines_only=True)
    assert [(row.step, row.loss) for row in rows] == [(0, 1.0)]
    assert offset == log.stat().st_size


def test_seekless_stream_skips_payloads_and_yields_real_metrics():
    stream = io.BytesIO(
        b"payload says step 9: {'train/loss': 99, 'train/step': 9}\n"
        + b"x" * (MAX_LOG_LINE_BYTES + 1)
        + b" step 8: {'train/loss': 88, 'train/step': 8}\n"
        + b"step 1: {'train/loss': 2.5, 'train/step': 1}\n"
    )
    rows = list(iter_metric_stream(stream, "island-0"))
    assert [(row.step, row.loss) for row in rows] == [(1, 2.5)]


def test_follow_detects_same_inode_copytruncate_regrowth(tmp_path: Path):
    from scripts.monitor_rl_loss import SourceState, _scan_state

    log = tmp_path / "learner.log"
    log.write_text("step 0: {'train/loss': 3.0, 'train/step': 0}\n" + "x" * 500 + "\n")
    state = SourceState("island-0", log)
    first = _scan_state(state, follow=True)
    assert [(row.step, row.loss) for row in first] == [(0, 3.0)]
    old_offset = state.offset

    with log.open("w", encoding="utf-8") as handle:
        handle.write(
            "step 1: {'train/loss': 2.0, 'train/step': 1}\n"
            + "y" * (old_offset + 100)
            + "\n"
        )
    assert log.stat().st_size > old_offset
    second = _scan_state(state, follow=True)
    assert [(row.step, row.loss) for row in second] == [(1, 2.0)]


def test_persist_is_idempotent_and_outputs_no_unknown_data(tmp_path: Path):
    csv_path = tmp_path / "curve.csv"
    json_path = tmp_path / "curve.json"
    row = parse_metric_line(
        "step 5: {'train/loss': 1.5, 'train/step': 5, 'prompt': 'never-write-me'}",
        "island-0",
    )
    assert row is not None
    assert len(persist_summaries([row], csv_path, json_path)) == 1
    assert len(persist_summaries([row], csv_path, json_path)) == 1

    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    document = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(csv_rows) == 1
    assert len(document["rows"]) == 1
    assert document["sources"] == {"island-0": {"latest_step": 5, "points": 1}}
    assert "never-write-me" not in csv_path.read_text(encoding="utf-8")
    assert "never-write-me" not in json_path.read_text(encoding="utf-8")


def test_invalid_source_label_is_rejected():
    with pytest.raises(ValueError, match="source labels"):
        parse_metric_line("step 0: {'train/loss': 1}", "/private/path")
    with pytest.raises(ValueError, match="fingerprints"):
        parse_metric_line("step 0: {'train/loss': 1, 'train/step': 0}", "a" * 64)
