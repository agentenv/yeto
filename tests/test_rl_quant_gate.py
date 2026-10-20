"""Gate between rollout quantization tiers (S26 E6): statistics, verdict, collection, CLI."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from yeto.rl.engine.sampler_engine import FINISH_STOP, Completion, SamplingParams
from yeto.rl.quant import gate
from yeto.rl.quant.gate import GateCriteria, Pair


def test_statistics_on_hand_made_pairs():
    pairs = [Pair((-1.0, -2.0), (-1.0, -2.5)), Pair((-0.1,), (-0.1 + math.log(3.0),)), Pair((-1.0,), (-1.0, -1.0))]
    stats = gate.compare(pairs, GateCriteria(tis_upper=2.0))
    assert stats["completions"] == 3 and stats["unaligned_completions"] == 1 and stats["tokens"] == 3
    diffs = [0.0, 0.5, -math.log(3.0)]
    assert stats["mean_diff"] == pytest.approx(sum(diffs) / 3)
    assert stats["mean_abs_diff"] == pytest.approx(sum(map(abs, diffs)) / 3)
    assert stats["max_abs_diff"] == pytest.approx(math.log(3.0))
    assert stats["p50_abs_diff"] == pytest.approx(0.5)
    # TIS weight exp(bf16 - quant): 1, exp(-0.5), 3 -> one of three above 2
    assert stats["tis_clip_high_share"] == pytest.approx(1 / 3) and stats["tis_clip_low_share"] == 0
    low = gate.compare(pairs, GateCriteria(tis_upper=2.0, tis_lower=0.7))
    assert low["tis_clip_low_share"] == pytest.approx(1 / 3)  # exp(-0.5) = 0.61 < 0.7


def test_quantile_interpolates():
    assert gate.quantile([3.0, 1.0, 2.0, 4.0], 0.5) == 2.5
    assert gate.quantile([1.0], 0.99) == 1.0
    assert math.isnan(gate.quantile([], 0.5))


def test_verdict_applies_only_given_criteria_and_fails_on_missing_values():
    stats = {"tokens": 10, "mean_abs_diff": 0.02, "p99_abs_diff": 0.3, "tis_clip_share": 0.001, "unaligned_share": 0.0}
    ok = gate.verdict(stats, None, GateCriteria(max_mean_abs_diff=0.05, max_tis_clip_share=0.01))
    assert ok["passed"] and set(ok["checks"]) == {"max_mean_abs_diff", "max_tis_clip_share", "max_unaligned_share"}
    bad = gate.verdict(stats, None, GateCriteria(max_p99_abs_diff=0.1))
    assert not bad["passed"] and not bad["checks"]["max_p99_abs_diff"]["passed"]
    no_speed = gate.verdict(stats, None, GateCriteria(min_speedup=1.1))
    assert not no_speed["passed"]
    fast = gate.verdict(stats, {"speedup": 1.3}, GateCriteria(min_speedup=1.1))
    assert fast["passed"]
    empty = gate.verdict({"tokens": 0, "mean_abs_diff": math.nan, "unaligned_share": math.nan}, None,
                         GateCriteria(max_mean_abs_diff=1.0))
    assert not empty["passed"]


def test_criteria_file_rejects_unknown_keys_and_bad_bounds():
    with pytest.raises(ValueError, match="unknown gate criteria"):
        GateCriteria.from_mapping({"max_kl": 1})
    with pytest.raises(ValueError, match="TIS bounds"):
        GateCriteria.from_mapping({"tis_upper": 0.9})


class StubEngine:
    """Generates fixed completions; scores them with a fixed shift (the BF16 engine)."""

    def __init__(self, shift: float, tokens: int = 4, tick: float = 1.0):
        self.shift, self.tokens, self.tick, self.calls = shift, tokens, tick, []

    def start(self, release_dir, version):
        self.calls.append("start")

    def stop(self):
        self.calls.append("stop")

    def generate(self, prompts, sampling):
        self.calls.append("generate")
        return [[Completion(tuple(range(self.tokens)), tuple(-0.5 for _ in range(self.tokens)), f"t{p}",
                            FINISH_STOP) for _ in range(sampling.n)] for p in prompts]

    def score(self, prompt, continuation):
        return [-0.5 + self.shift] * self.tokens


def test_collect_runs_engines_one_after_the_other():
    now = iter([0.0, 2.0, 10.0, 14.0])
    quant, bf16 = StubEngine(0.0), StubEngine(-0.1)
    pairs, speed, raw = gate.collect(quant, bf16, ["a", "b"], SamplingParams(max_tokens=8, n=2), clock=lambda: next(now))
    assert quant.calls == ["start", "generate", "stop"] and bf16.calls == ["start", "generate", "stop"]
    assert len(pairs) == 4 and len(raw) == 4 and raw[0]["bf16_logprobs"] == [-0.6] * 4
    assert speed["quant"]["tokens_per_s"] == 16 / 2 and speed["bf16"]["tokens_per_s"] == 16 / 4
    assert speed["speedup"] == 2.0
    stats = gate.compare(pairs, GateCriteria())
    assert stats["mean_diff"] == pytest.approx(0.1)


def test_records_cli_writes_the_report_and_sets_the_exit_code(tmp_path: Path):
    records = tmp_path / "r.jsonl"
    records.write_text("\n".join(json.dumps({"rollout_log_probs": [-1.0, -2.0], "log_probs": [-1.01, -2.02]})
                                 for _ in range(3)))
    criteria = tmp_path / "c.json"
    criteria.write_text(json.dumps({"max_mean_abs_diff": 0.05, "max_tis_clip_share": 0.0}))
    out = tmp_path / "gate" / "report.json"
    assert gate.main(["records", "--records", str(records), "--criteria", str(criteria), "--out", str(out),
                      "--label", "tier0->tier1"]) == 0
    saved = json.loads(out.read_text())
    assert saved["schema"] == gate.SCHEMA and saved["verdict"]["passed"] and saved["meta"]["label"] == "tier0->tier1"
    assert saved["stats"]["tokens"] == 6
    criteria.write_text(json.dumps({"max_mean_abs_diff": 0.001}))
    assert gate.main(["records", "--records", str(records), "--criteria", str(criteria), "--out", str(out)]) == 1
    assert not json.loads(out.read_text())["verdict"]["passed"]
