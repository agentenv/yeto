"""``yeto breaker``: health breakers over event tapes, with log-prob mismatch limits per
sampler hardware (yeto.rl.breaker). CPU only, over synthetic tapes."""
from __future__ import annotations

import json

import pytest

from yeto import cli
from yeto.rl import breaker
from yeto.rl.engine.xhw_guard import XhwGuard

AMD_SOURCE = "rocm/MI355X/vllm@0.11/mxfp4/tp8"
NV_SOURCE = "cuda/B300/sglang@0.5/bf16/tp8"


def _trained(rid, abs_diff, **metrics):
    return {"event": "rl_round_trained", "rollout_id": rid, "trained_groups": 32,
            "train_metrics": {"train/train_rollout_logprob_abs_diff": abs_diff, "sc_is_clipped": 0.001,
                              "sc_importance_weight": 1.0, **metrics}}


def _round(source, **extra):
    return {"event": "rl_sampler_engine_round", "rollout_id": 1, "groups": 4, "source": source, **extra}


def _write(path, records, prefix=""):
    path.write_text("".join(f"{prefix}{json.dumps(r)}\n" for r in records))
    return path


def _checks(out):
    return {c["name"]: c for c in out["checks"]}


def test_amd_defaults_are_the_in_run_guards_and_the_measured_mi355x_band_does_not_trip():
    guard = XhwGuard()
    assert breaker.LOGPROB_LIMITS["amd"] == (guard.warn.abs_diff, guard.act.abs_diff) == (0.10, 0.15)
    # MI355X vLLM -> B300 Megatron measured 0.076-0.099: the old fixed 0.052 tripped here
    out = breaker.evaluate([_trained(i, 0.09) for i in range(3)], [_round(AMD_SOURCE)])
    c = _checks(out)["logprob_diff"]
    assert out["hardware"] == "amd" and c["limit"] == 0.15 and c["warn"] == 0.10
    assert c["level"] == "ok" and not out["tripped"]


def test_amd_warns_above_010_and_trips_above_015():
    sampler = [_round(AMD_SOURCE)]
    assert _checks(breaker.evaluate([_trained(i, 0.12) for i in range(3)], sampler))["logprob_diff"]["level"] == "warn"
    out = breaker.evaluate([_trained(i, 0.16) for i in range(3)], sampler)
    assert out["tripped"] == ["logprob_diff"]


def test_nvidia_samplers_keep_the_b300_baseline():
    sampler = [_round(NV_SOURCE)]
    out = breaker.evaluate([_trained(i, 0.06) for i in range(3)], sampler)
    assert out["hardware"] == "nvidia" and out["tripped"] == ["logprob_diff"]
    assert _checks(breaker.evaluate([_trained(i, 0.03) for i in range(3)], sampler))["logprob_diff"]["level"] == "ok"


def test_a_mixed_fleet_is_judged_by_the_amd_limits():
    out = breaker.evaluate([_trained(i, 0.09) for i in range(3)], [_round(NV_SOURCE), _round(AMD_SOURCE)])
    assert out["hardware"] == "amd" and out["hardware_seen"] == ["amd", "nvidia"] and not out["tripped"]


def test_unknown_hardware_uses_amd_limits_and_says_so():
    out = breaker.evaluate([_trained(i, 0.09) for i in range(3)], [])
    c = _checks(out)["logprob_diff"]
    assert out["hardware"] == "amd" and "unknown" in c["note"] and c["level"] == "ok"


def test_hardware_from_the_xhw_rows_when_no_sampler_tape():
    xhw = {"event": "rl_xhw_mismatch", "rollout_id": 2,
           "sources": {"s0|sglang|0.5|B300": {"sampler_id": "s0", "card_model": "B300", "level": "ok"}}}
    out = breaker.evaluate([*[_trained(i, 0.06) for i in range(3)], xhw], [])
    assert out["hardware"] == "nvidia" and "logprob_diff" in out["tripped"]


def test_explicit_hardware_noise_floor_and_overrides():
    trained = [_trained(i, 0.06) for i in range(3)]
    assert not breaker.evaluate(trained, [_round(NV_SOURCE)], hardware="amd")["tripped"]
    out = breaker.evaluate(trained, [_round(NV_SOURCE)], noise_floor=0.05)  # act max(0.052, 0.075)
    assert _checks(out)["logprob_diff"]["limit"] == pytest.approx(0.075) and not out["tripped"]
    out = breaker.evaluate(trained, [_round(AMD_SOURCE)], logprob_warn=0.02, logprob_act=0.05)
    assert out["tripped"] == ["logprob_diff"]
    with pytest.raises(ValueError, match="warn"):
        breaker.evaluate(trained, [], logprob_warn=0.2, logprob_act=0.1)
    with pytest.raises(ValueError, match="hardware"):
        breaker.evaluate(trained, [], hardware="tpu")


def test_the_window_is_the_trailing_steps():
    trained = [_trained(0, 0.5), *[_trained(i, 0.05) for i in range(1, 4)]]
    assert not breaker.evaluate(trained, [_round(AMD_SOURCE)], window=3)["tripped"]
    assert breaker.evaluate(trained, [_round(AMD_SOURCE)], window=4)["tripped"] == ["logprob_diff"]


def test_other_breakers():
    trained = [_trained(i, 0.01, sc_is_clipped=0.2, sc_importance_weight=0.5) for i in range(3)]
    stale = [{"event": "rl_buffer_stale_skipped", "stale_groups": 64}]
    sampler = [_round(AMD_SOURCE, agentic={"trials": 10, "lost": {"timeout": 3}, "infra_retries": {"x": 1}}),
               {"event": "rl_sampler_error", "error": "routed_experts has 3 rows"}]
    out = breaker.evaluate([*trained, *stale], sampler, flips={"r1": {"changed": 5, "total": 100}})
    assert set(out["tripped"]) == {"is_clipped", "is_weight", "stale", "aborted", "zero_call", "flips", "r3_rows"}


def test_nan_skips_trip_by_streak_and_clear_after_a_trained_round():
    skips = [{"event": "rl_nan_skip", "rollout_id": 5, "nan_consecutive": 3}]
    assert "nan_skip_consecutive" in breaker.evaluate([_trained(4, 0.01), *skips], [])["tripped"]
    assert not breaker.evaluate([*skips, _trained(6, 0.01)], [])["tripped"]


def test_xhw_rows_of_the_in_run_guard_are_reported_and_act_trips():
    xhw = {"event": "rl_xhw_mismatch", "rollout_id": 2, "sources": {
        "a|vllm|0.11|MI355X": {"sampler_id": "a", "card_model": "MI355X", "abs_diff": 0.2, "level": "act",
                               "action": "quarantine"},
        "b|vllm|0.11|MI355X": {"sampler_id": "b", "card_model": "MI355X", "abs_diff": 0.08, "level": "ok"}}}
    out = breaker.evaluate([_trained(1, 0.09), xhw], [])
    assert out["tripped"] == ["xhw:a"] and [r["trip"] for r in out["xhw"]] == [True, False]


def test_hardware_class():
    assert breaker.source_hardware(AMD_SOURCE) == "amd"
    assert breaker.source_hardware(NV_SOURCE) == "nvidia"
    assert breaker.hardware_class(None, "MI300X") == "amd"
    assert breaker.hardware_class(None, "H100") == "nvidia"
    assert breaker.hardware_class("none", "none") is None


def test_cli_reads_launcher_logs_and_tape_dirs_and_exits_by_trip(tmp_path, capsys):
    log = _write(tmp_path / "launcher.log", [_trained(i, 0.09) for i in range(3)],
                 prefix="[island] YETO_RL_EVENT ")
    tapes = tmp_path / "tapes" / "node1"
    tapes.mkdir(parents=True)
    _write(tapes / "events.0.jsonl", [_round(AMD_SOURCE)])
    assert cli.main(["breaker", "--trainer-log", str(log), "--sampler-tapes", str(tmp_path / "tapes")]) == 0
    text = capsys.readouterr().out
    assert "ok   logprob_diff" in text and "amd sampler limits" in text
    assert cli.main(["breaker", "--trainer-log", str(log), "--sampler-tapes", str(tmp_path / "tapes"),
                     "--sampler-hardware", "nvidia", "--json"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["tripped"] == ["logprob_diff"] and out["logprob_limits"] == {"warn": 0.04, "act": 0.052}


def test_cli_missing_file_is_a_usage_error(tmp_path, capsys):
    assert cli.main(["breaker", "--trainer-log", str(tmp_path / "nope.log")]) == 2
    assert "[breaker]" in capsys.readouterr().err
