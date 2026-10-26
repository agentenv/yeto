"""tools/eval/health_gr.py: the merge-boundary reward check reads the whole batch
(Miles rollout/raw_reward), not the island tape's trajectory records, which stop at
256 of 512 samples per round (s27-bb-a1 island 1 round 146: half batch 0.72 -> 0.31,
FAIL; whole batch 0.51 -> 0.34, GBB-CLOSING.md 5.4)."""

from __future__ import annotations

import json

import pytest

from tools.eval import health_gr as hg

RUN = "s27-test"


def _line(island: int, text: str) -> str:
    return f"[{RUN}-l{island}-modal] {text}"


def _event(island: int, **record) -> str:
    return _line(island, "YETO_RL_EVENT " + json.dumps({"island_id": island, **record}))


def _write_run(tmp_path, raw: dict[int, float | None], half: dict[int, float], *, merge_at: int = 3,
               rounds: int = 6):
    """One island: Miles rollout/step lines (raw_reward = the whole batch), 10 trajectory
    records per round with success fraction ``half[r]`` (the half batch), merges applied at
    the boundary that starts rollout ``merge_at``."""
    lines = []
    for r in range(rounds):
        rollout = {"rollout/log_probs": -0.5, "rollout/rollout_log_probs": -0.5}
        if raw.get(r) is not None:
            rollout["rollout/raw_reward"] = raw[r]
        lines.append(_line(0, f"log_utils.py:57 - rollout {r}: {rollout!r}"))
        lines.append(_line(0, f"log_utils.py:57 - step {r}: "
                              f"{{'train/loss': 0.0, 'train/grad_norm': 0.1, 'train/train_rollout_logprob_abs_diff': 0.01, "
                              f"'train/train_rollout_kl': 0.001, 'train/entropy_loss': 0.9}}"))
        wins = round(half[r] * 10)
        for k in range(10):
            lines.append(_event(0, event="rl_trajectory_reward", rollout_id=r, reward=float(k < wins),
                                success=k < wins, task_id=f"t{r}-{k}"))
        lines.append(_event(0, event="rl_full_parameter_round", policy_version=r + 1,
                            **{"sync/fragment_versions": [1 if r + 1 >= merge_at else 0, 0]}))
    run = tmp_path / RUN
    run.mkdir()
    (run / "launch.stream.log").write_text("\n".join(lines) + "\n")
    return run


def _merge_jump(run):
    parsed = hg.parse(str(run))
    checks = {c["id"]: c for c in hg.check(parsed)}
    return checks["merge_jump"], parsed["jumps"]


HALF = {0: 0.5, 1: 0.6, 2: 0.9, 3: 0.3, 4: 0.5, 5: 0.5}  # half batch: 0.9 -> 0.3 at the merge


def test_the_merge_boundary_check_uses_the_whole_batch_reward(tmp_path):
    raw = {0: 0.50, 1: 0.52, 2: 0.55, 3: 0.48, 4: 0.50, 5: 0.51}
    check, jumps = _merge_jump(_write_run(tmp_path, raw, HALF))
    (jump,) = jumps
    assert jump["rollout"] == 3 and jump["acc_source"] == "raw_reward"
    assert jump["acc"] == (0.55, 0.48) and jump["d_acc"] == pytest.approx(-0.07)
    assert jump["half_batch_acc"] == (0.9, 0.3)  # the old check failed here (drop 0.6 > 0.35)
    assert check["status"] == "PASS" and jump["status"] == "PASS"
    assert "全批次 raw_reward 1 个边界，半批次 acc 0 个边界" in check["value"]


def test_a_whole_batch_drop_past_the_threshold_still_fails(tmp_path):
    raw = {0: 0.50, 1: 0.52, 2: 0.80, 3: 0.40, 4: 0.50, 5: 0.51}
    check, (jump,) = _merge_jump(_write_run(tmp_path, raw, {r: 0.5 for r in HALF}))
    assert jump["d_acc"] == pytest.approx(-0.40) and check["status"] == "FAIL"


def test_without_raw_reward_the_boundary_falls_back_to_the_half_batch(tmp_path):
    raw = {0: 0.50, 1: 0.52, 2: None, 3: 0.48, 4: 0.50, 5: 0.51}
    check, (jump,) = _merge_jump(_write_run(tmp_path, raw, HALF))
    assert jump["acc_source"] == "acc" and jump["acc"] == (0.9, 0.3)
    assert check["status"] == "FAIL" and "半批次 acc 1 个边界" in check["value"]


def test_the_cli_writes_the_json_outside_the_run_dir(tmp_path, capsys):
    raw = {r: 0.5 for r in HALF}
    run = _write_run(tmp_path, raw, HALF)
    out = tmp_path / "health.json"
    hg.main([str(run), "--no-table", "--json", str(out)])
    doc = json.loads(out.read_text())
    assert doc["parse"]["jumps"][0]["acc_source"] == "raw_reward"
    assert "reward(raw) 0.50->0.50" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        hg.main([str(run), "--no-table", "--json", str(run / "inside.json")])
