"""CPU tests of the efficiency evaluation (tools/eval/efficiency, S26 E0.1).

The synthetic logs below follow the formats the parsers cite:

* verl FileLogger rows ``{"step": N, "data": {...}}`` -- fork acad9875
  ``verl/utils/tracking.py`` ``FileLogger.log``; console ``step:N - k:v`` --
  ``verl/utils/logger/aggregate_logger.py`` ``concat_dict_to_str``; keys as in the
  real T-EVAL file ``SESSION25/logs/T-EVAL/eval-e1-verl-c/.../verl-file-logger-0.jsonl``
  and ``metric_utils.compute_timing_metrics`` (``timing_s/*``);
* Miles ``perf N: {dict}`` lines -- copied in shape from the real T-EVAL log
  ``SESSION25/logs/T-EVAL/eval-e1-miles-a/raw/launch.ts.log`` (rollout side
  ``metrics.py:93``, train side ``train_metric_utils.py:56``);
* yeto tape events -- field names as in the T-EVAL tapes (``rl_round_wall``,
  ``rl_local_round``, ``rl_round_trained``, ``rl_timeline_span``);
* NVML CSV -- ``nvidia-smi --query-gpu=... --format=csv`` output and the
  ``miles_local.sh`` ``<container>,`` prefixed form.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.eval import __main__ as cli
from tools.eval.efficiency import eff_reward, metrics, parse, report, spec

SPEC = spec.get_spec("qwen3-0.6b")


# ====================================================================== spec
def test_arms_cover_both_systems_frameworks_and_modes():
    assert len(spec.ARMS) == 8
    for name in spec.ARMS:
        arm = spec.Arm.parse(name)
        assert spec.Arm.parse(arm.peer).system != arm.system
    with pytest.raises(ValueError):
        spec.Arm.parse("bare-verl-fast")


def test_every_arm_carries_the_shared_values():
    plans = {a: spec.plan(SPEC, a, "r1") for a in spec.ARMS}
    assert len({p["fingerprint"] for p in plans.values()}) == 1
    for name, p in plans.items():
        text = p.get("script") or " ".join(p.get("argv") or []) or p["launcher_args"]
        assert p["cards"]["total"] == (1 if p["mode"] == "sync" else 2), name
        if p["system"] == "bare":
            assert "Qwen3-0.6B" in text and "/root/eff-data/train" in text, name
        else:
            assert SPEC.model_revision in text and SPEC.data_revision in text, name
        for value in (SPEC.rounds, SPEC.groups_per_round, SPEC.samples_per_group, SPEC.max_response_len,
                      SPEC.lr, SPEC.lora_r, SPEC.seed):
            assert str(value) in text, (name, value)


def test_fingerprint_changes_with_a_shared_value_only():
    base = spec.control_fingerprint(SPEC)
    assert spec.control_fingerprint(spec.get_spec("qwen3-0.6b", repeats=5)) == base
    assert spec.control_fingerprint(spec.get_spec("qwen3-0.6b", lr=2e-5)) != base


def test_bare_entry_points_follow_the_official_recipes():
    sync = spec.bare_script(SPEC, spec.Arm.parse("bare-verl-sync"), "r1")
    async_ = spec.bare_script(SPEC, spec.Arm.parse("bare-verl-async"), "r1")
    assert "-m verl.trainer.main_ppo" in sync and "fully_async_main" in async_
    assert "async_training.staleness_threshold=1" in async_ and "rollout.n_gpus_per_node=1" in async_
    assert "trainer.logger=[console,file]" in sync and "VERL_FILE_LOGGER_PATH" in sync
    miles_sync = spec.bare_script(SPEC, spec.Arm.parse("bare-miles-sync"), "r1")
    miles_async = spec.bare_script(SPEC, spec.Arm.parse("bare-miles-async"), "r1")
    assert "/root/miles/train.py $MODEL_ARGS" in miles_sync and "--colocate" in miles_sync
    assert "/root/miles/train_async.py" in miles_async and "--fully-async --max-weight-staleness 1" in miles_async
    assert "scripts/models/qwen3-0.6B.py" in miles_sync  # the image's own model flags
    for script in (sync, async_, miles_sync, miles_async):
        assert "yeto" not in script.replace("yeto-effeval", "").replace("yeto/", "")


def test_yeto_arms_use_the_same_policy_age_and_partition():
    assert "--rl-max-policy-age 1" in " ".join(spec.yeto_verl_argv(SPEC, "async", "/t"))
    assert "--rl-max-policy-age" not in " ".join(spec.yeto_verl_argv(SPEC, "sync", "/t"))
    # async needs an algorithm spec that tolerates the policy age (the default allows 0)
    argv = spec.yeto_verl_argv(SPEC, "async", "/t")
    algo = argv[argv.index("--rl-algorithm-spec") + 1]
    assert algo.endswith(spec.ALGORITHM_SPEC) and (Path(__file__).parents[1] / spec.ALGORITHM_SPEC).is_file()
    args = spec.yeto_miles_launcher_args(SPEC, "async", "r1")
    assert "--gpu modal:2xa10g" in args and "--rl-placement fixed-partition --rl-rollout-gpus 1" in args
    assert "--gpu modal:1xa10g" in spec.yeto_miles_launcher_args(SPEC, "sync", "r1")


# overrides yeto's verl translation sets and the bare recipe leaves at the default (or the reverse);
# a new key here means the two arms drifted apart: decide, then extend the list
KNOWN_VERL_DIFF = {
    "actor_rollout_ref.model.lora.merge", "actor_rollout_ref.rollout.enforce_eager",
    "actor_rollout_ref.rollout.full_determinism", "actor_rollout_ref.rollout.load_format",
    "ray_kwargs.ray_init.num_cpus", "trainer.project_name", "trainer.use_v1", "trainer.v1.trainer_mode",
    "actor_rollout_ref.rollout.mode", "async_training.staleness_threshold", "data.return_raw_chat",
    "rollout.total_rollout_steps",
}


@pytest.mark.parametrize("mode", spec.MODES)
def test_verl_knob_diff_is_the_known_set(mode):
    diff = spec.verl_knob_diff(SPEC, mode)
    assert set(diff) <= KNOWN_VERL_DIFF, set(diff) - KNOWN_VERL_DIFF


def test_4b_preset_is_ready_for_e02():
    s4 = spec.get_spec("qwen3-4b-instruct-2507")
    assert s4.gpu == "H100!" and s4.repeats >= 2 and len(s4.model_revision) == 40
    assert "--gpu modal:2xh100" in spec.yeto_miles_launcher_args(s4, "async", "r")
    assert "qwen3-4B-Instruct-2507.py" in spec.bare_script(s4, spec.Arm.parse("bare-miles-sync"), "r")


# ====================================================================== reward
@pytest.mark.parametrize("response,label", [
    ("so \\boxed{42}", "#### 42"), ("the answer is 1,234", "x #### 1234"), ("\\boxed{3.5} then 7", "#### 3.5"),
    ("no number", "#### 1"), ("\\boxed{-2}", "#### 2"), ("", None),
])
def test_bare_reward_equals_yeto_reward(response, label):
    from yeto.rl.rewards.builtin import gsm8k_grade

    assert eff_reward.grade(response, label) == gsm8k_grade(response, label)
    assert eff_reward.compute_score("d", response, label) == gsm8k_grade(response, label)


# ====================================================================== parse: verl
def _verl_rows(n: int, *, async_: bool = False) -> str:
    rows = []
    for step in range(1, n + 1):
        d = {"timing_s/gen": 10.0 + step, "timing_s/old_log_prob": 1.0, "timing_s/update_actor": 3.0,
             "timing_s/step": 16.0 + step, "response_length/mean": 100.0, "perf/total_num_tokens": 9000,
             "critic/score/mean": 0.1 * step, "rollout_corr/rollout_is_ratio_fraction_high": 0.01}
        if async_:
            d.pop("timing_s/gen")
            d.update({"timing_s/param_sync": 0.5, "timing_s/wait_for_enough_samples": 2.0,
                      "fully_async/count/dropped_stale_samples": step, "fully_async/count/total_generated_samples":
                      32 * step})
        else:
            d["timing_s/update_weights"] = 0.7
        rows.append(json.dumps({"step": step, "data": d}))
    return "\n".join(rows) + "\n"


def _verl_console(n: int, t0: float = 1000.0, dt: float = 20.0) -> str:
    return "".join(f"{t0 + dt * s:.3f} (TaskRunner pid=1) step:{s} - timing_s/gen:{10 + s} - critic/score/mean:0.1\n"
                   for s in range(1, n + 1))


def test_parse_verl_sync_rounds_and_sources():
    p = parse.parse_verl(_verl_rows(3), _verl_console(3), samples_per_round=32, trainer_cards=1, mode="sync")
    r = p.rounds
    assert [x["round"] for x in r] == [1, 2, 3]
    assert r[1]["round_s"] == 20.0 and r[0]["round_s"] is None
    assert r[0]["gen_tokens"] == 3200 and r[0]["train_s"] == 4.0 and r[0]["weight_sync_s"] == 0.7
    assert p.sources["gen_tokens"].startswith("derived:") and p.sources["wait_s"] == parse.NA
    assert p.sources["sampler_update_s"] == parse.NA and p.sources["version_lag"] == parse.NA
    assert p.segments_raw[0]["timing_s/step"] == 17.0


def test_parse_verl_async_reads_fully_async_counters():
    p = parse.parse_verl(_verl_rows(2, async_=True), "", samples_per_round=32, trainer_cards=1, mode="async")
    assert p.rounds[0]["wait_s"] == 2.0 and p.rounds[0]["weight_sync_s"] == 0.5
    assert p.rounds[1]["dropped_samples"] == 2 and p.sources["gen_s"] == parse.MISSING
    assert p.sources["t_end_unix"] == parse.MISSING and p.sources["version_lag"] == parse.MISSING


# ====================================================================== parse: Miles
def _miles_log(n: int, *, async_: bool = False) -> str:
    out = []
    for rid in range(n):
        ro = {"rollout/raw_reward": 0.4 + 0.01 * rid, "perf/rollout_time": 12.0,
              "perf/tokens_per_gpu_per_sec": 500.0, "rollout/num_training_samples": 32}
        tr = {"perf/train_wait_time": 12.5, "perf/train_time": 20.0, "perf/actor_train_time": 15.0,
              "perf/actor_train_tok_per_s": 800.0, "perf/update_weights_time": 0.15, "perf/sleep_time": 1.7}
        out.append(f"2026-10-20T06:36:{10 + rid:02d}Z [p-l0-modal] \x1b[36m(RolloutExecutor pid=2765)\x1b[0m "
                   f"[2026-10-20 06:36:{10 + rid:02d}.055 rollout_executor] metrics.py:93 - perf {rid}: {ro!r}")
        out.append(f"2026-10-20T06:37:{rid * 30 % 60:02d}Z [p-l0-modal] (MegatronTrainRayActor pid=3190) "
                   f"[2026-10-20 06:{37 + rid // 2}:{rid * 30 % 60:02d}.147 actor_cell00000_rank00000] "
                   f"train_metric_utils.py:56 - perf {rid}: {tr!r}")
    return "\n".join(out) + "\nunrelated line\n"


def test_parse_miles_merges_rollout_and_train_lines():
    p = parse.parse_miles(_miles_log(3), rollout_cards=1, mode="sync")
    r = p.rounds
    assert [x["round"] for x in r] == [0, 1, 2]
    assert r[0]["gen_s"] == 12.0 and r[0]["train_s"] == 20.0 and r[0]["weight_sync_s"] == 0.15
    assert r[0]["gen_tokens"] == 6000.0 and r[0]["train_tokens"] == 12000.0 and r[0]["samples"] == 32
    assert r[1]["round_s"] == 30.0 and r[0]["reward_mean"] == 0.4
    assert p.sources["wait_s"] == parse.NA and p.sources["is_clip_frac"] == parse.MISSING


def test_parse_miles_async_keeps_wait():
    p = parse.parse_miles(_miles_log(2), rollout_cards=1, mode="async")
    assert p.rounds[0]["wait_s"] == 12.5 and p.sources["dropped_samples"] == parse.MISSING


# ====================================================================== parse: yeto tape
def _tape(n: int, t0: float = 2000.0) -> list[dict]:
    ev = []
    for rid in range(n):
        end = t0 + 25.0 * (rid + 1)
        ev += [{"event": "rl_timeline_span", "rollout_id": rid, "task": "generate", "kind": "compute",
                "start": 100.0 * rid, "end": 100.0 * rid + 11.0},
               {"event": "rl_timeline_span", "rollout_id": rid, "task": "train", "kind": "compute",
                "start": 100.0 * rid + 11.0, "end": 100.0 * rid + 21.0},
               {"event": "rl_timeline_span", "rollout_id": rid, "task": "publish", "kind": "transfer",
                "start": 100.0 * rid + 21.0, "end": 100.0 * rid + 22.0},
               {"event": "rl_timeline_span", "rollout_id": rid, "task": "offload", "kind": "transfer",
                "start": 100.0 * rid + 22.0, "end": 100.0 * rid + 24.0},
               {"event": "rl_round_wall", "rollout_id": rid, "round_seconds": 25.0, "time_unix": end},
               {"event": "rl_local_round", "action_tokens": 3000 + rid, "reward_mean": 0.5, "time_unix": end - 1},
               {"event": "rl_round_trained", "rollout_id": rid, "trained_samples": 32,
                "mismatch": {"tis_clipfrac": 0.0}, "time_unix": end}]
    return ev


def test_parse_yeto_tape_with_framework_tokens():
    fw = parse.parse_miles(_miles_log(3), rollout_cards=1, mode="sync")
    p = parse.parse_yeto(_tape(3), mode="sync", framework_parsed=fw)
    r = p.rounds
    assert r[0]["gen_s"] == 11.0 and r[0]["train_s"] == 10.0 and r[0]["weight_sync_s"] == 1.0
    assert r[1]["round_s"] == 25.0 and r[2]["gen_tokens"] == 3002 and r[0]["train_tokens"] == 12000.0
    assert p.sources["train_tokens"].startswith("framework log:") and p.segments_raw[0]["offload"] == 2.0


def test_parse_yeto_driver_phase_fallback():
    ev = [{"event": "rl_driver_phase", "phase": "generate", "rollout_id": 0, "time_unix": 10.0},
          {"event": "rl_driver_phase", "phase": "train", "time_unix": 18.0},
          {"event": "rl_driver_phase", "phase": "publish", "time_unix": 21.0},
          {"event": "rl_driver_phase", "phase": "generate", "rollout_id": 1, "time_unix": 22.0},
          {"event": "rl_round_wall", "rollout_id": 0, "time_unix": 22.0}]
    p = parse.parse_yeto(ev, mode="sync")
    assert p.rounds[0]["gen_s"] == 8.0 and p.rounds[0]["train_s"] == 3.0 and p.rounds[0]["weight_sync_s"] == 1.0
    assert "rl_driver_phase" in p.sources["gen_s"]


def test_parse_nvml_both_formats():
    own = ("timestamp, index, name, utilization.gpu [%], memory.used [MiB], memory.total [MiB], power.draw [W]\n"
           "2026/10/20 06:36:11.055, 0, NVIDIA A10, 45 %, 6077 MiB, 23028 MiB, 81.7 W\n"
           "2026/10/20 06:36:12.055, 1, NVIDIA A10, 0 %, 100 MiB, 23028 MiB, 30.1 W\n")
    polled = "ta-01ABC,2026/10/20 06:36:11.055, NVIDIA A10, 44 %, 6077 MiB, 81.76 W\n"
    a, b = parse.parse_nvml(own), parse.parse_nvml(polled)
    assert [x["util"] for x in a] == [45.0, 0.0] and a[1]["key"] == ":1" and a[0]["mem_total_mb"] == 23028.0
    assert b[0]["key"] == "ta-01ABC:0" and b[0]["mem_mb"] == 6077.0 and b[0]["power_w"] == 81.76


# ====================================================================== run dirs, metrics, report
def _nvml_csv(t0: float, n: int, util: int) -> str:
    import datetime as dt

    lines = ["timestamp, index, name, utilization.gpu [%], memory.used [MiB], memory.total [MiB], power.draw [W]"]
    for i in range(n):
        stamp = dt.datetime.fromtimestamp(t0 + i, dt.timezone.utc).strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
        lines.append(f"{stamp}, 0, NVIDIA A10, {util if i % 4 else 0} %, 7000 MiB, 23028 MiB, 80 W")
    return "\n".join(lines) + "\n"


def _run_dir(tmp: Path, arm: str, run_id: str, dt_round: float) -> Path:
    plan = spec.plan(SPEC, arm, run_id)
    d = tmp / run_id
    spec.write_plan(d / "arm.json", plan)
    raw = d / "raw" / run_id / "arm"
    raw.mkdir(parents=True)
    n, t0 = SPEC.rounds, 1000.0
    (raw / "meta.json").write_text(json.dumps({"rc": 0, "start_unix": t0 - 200, "end_unix": t0 + dt_round * n + 30}))
    (raw / "nvml.csv").write_text(_nvml_csv(t0, int(dt_round * n) + 5, 60))
    if plan["system"] == "bare":
        (raw / "main.log").write_text(_verl_console(n, t0, dt_round))
        (raw / "eff-out").mkdir()
        (raw / "eff-out" / "verl-metrics.jsonl").write_text(_verl_rows(n))
    else:
        out = raw / "yeto-output"
        out.mkdir()
        (out / "verl-file-logger-0.jsonl").write_text("".join(  # steps 1..n = rollout ids 0..n-1
            json.dumps({"step": k, "data": {"global_seqlen/mean": 9000}}) + "\n" for k in range(1, n + 1)))
        (out / "rl-island-0.jsonl").write_text("\n".join(json.dumps(e) for e in _tape(n, t0 - dt_round)
                                                         ) + "\n")
    return d


def test_parse_run_metrics_and_report(tmp_path, monkeypatch):
    dirs = [_run_dir(tmp_path, "bare-verl-sync", "b1", 20.0), _run_dir(tmp_path, "bare-verl-sync", "b2", 21.0),
            _run_dir(tmp_path, "yeto-verl-sync", "y1", 25.0), _run_dir(tmp_path, "yeto-verl-sync", "y2", 25.0)]
    for d in dirs:
        assert cli.main(["eff-parse", str(d)]) == 0
        assert (d / "parsed.json").is_file() and (d / "summary.json").is_file()
    s = json.loads((dirs[0] / "summary.json").read_text())
    assert s["rounds"] == SPEC.rounds and s["steady_wall_s"] == 20.0 * (SPEC.rounds - 1)
    assert s["gen_tok_per_card_s"] == round(3200 / 20.0, 3) and s["bubble_frac"] == pytest.approx(0.25, abs=0.02)
    assert s["usd_per_1k_samples_steady"] == round(2.0 * 20.0 * 7 / 3600 / (32 * 7) * 1000, 4)
    assert s["gpu_mem_peak_mb"] == 7000.0
    assert s["missing"] == [] and "wait_s_median" in s["not_applicable"] and "drop_frac" in s["not_applicable"]
    y = json.loads((dirs[2] / "summary.json").read_text())
    assert y["missing"] == [] and y["train_s_median"] == 10.0 and y["unaccounted_s_median"] == 3.0
    assert y["train_tok_per_card_s"] == round(9000 / 25.0, 3)
    out = tmp_path / "report.md"
    assert cli.main(["eff-report", *map(str, dirs), "--out", str(out)]) == 0
    text = out.read_text()
    assert "### verl sync (bare n=2, yeto n=2)" in text
    section = text.split("### verl sync")[1]
    row = next(line for line in section.splitlines() if line.startswith("| round time (s)"))
    assert "yeto worse" in row and "+4.5" in row
    assert "| update sampler weights (s/round) | n/a | n/a | n/a | not applicable |" in text
    assert "wait (s/round) | n/a" in text


def test_report_needs_two_repeats_and_one_spec(tmp_path):
    one = [_run_dir(tmp_path, "bare-verl-sync", "b1", 20.0), _run_dir(tmp_path, "yeto-verl-sync", "y1", 25.0)]
    for d in one:
        cli.main(["eff-parse", str(d)])
    text = report.markdown(report.load(one), title="t")
    assert "insufficient repeats (bare 1, yeto 1; need 2)" in text
    runs = report.load(one)
    runs[1]["fingerprint"] = "other"
    with pytest.raises(ValueError, match="different specs"):
        report.markdown(runs, title="t")


def test_verdict_rules():
    assert report.verdict([10, 11], [12, 13], True) == "yeto worse"
    assert report.verdict([10, 11], [8, 9], True) == "yeto better"
    assert report.verdict([10, 12], [11, 13], True) == "within run-to-run range"
    assert report.verdict([10, 11], [8, 9], False) == "yeto worse"
    assert report.verdict([10, None], [8, 9], True) == "missing value"


def test_price_comes_from_the_config(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"A10G": {"usd_per_card_hour": 3.5}}))
    assert metrics.price("A10G", path) == 3.5
    with pytest.raises(KeyError):
        metrics.price("H100!", path)


def test_cli_plan_and_unregistered_run(tmp_path, monkeypatch, capsys):
    assert cli.main(["eff-plan", "qwen3-0.6b", "--arm", "bare-verl-sync"]) == 0
    assert json.loads(capsys.readouterr().out)["arms"]["bare-verl-sync"]["cards"]["total"] == 1
    monkeypatch.setattr(cli, "LEDGER", tmp_path / "ledger.md")
    (tmp_path / "ledger.md").write_text("nothing registered\n")
    with pytest.raises(SystemExit, match="not pre-registered"):
        cli.main(["eff-run", "qwen3-0.6b", "--arm", "bare-verl-sync", "--run-id", "eff-x", "--out", str(tmp_path)])


def test_bare_verl_async_uses_the_fork_working_lora_sync():
    script = spec.bare_script(SPEC, spec.Arm.parse("bare-verl-async"), "r")
    assert "actor_rollout_ref.model.lora.merge=True" in script and "export NCCL_CUMEM_HOST_ENABLE=0" in script
    assert "NCCL_CUMEM_HOST_ENABLE" not in spec.bare_script(SPEC, spec.Arm.parse("bare-verl-sync"), "r")
