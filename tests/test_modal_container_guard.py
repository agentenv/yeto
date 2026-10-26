"""tools/modal_container_guard.py: the shared container guard and cost guard of the
Modal run scripts, run offline on made-up ``modal container list --json`` output.

s27-ch-d0a (2026-10-25): two extra containers of the island function appeared at
00:56:15Z while the two original containers trained (step 66 at 00:56:11Z); the old
shell rule ("more distinct containers than NODES") stopped the healthy run.

s27-ex-1 (2026-10-25): Modal built the image in the app first (two build containers, 12 min);
the guard took them for the island containers and stopped the app when the islands'
containers came (18:06:21Z). The islands' own container lines now name the island containers."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from tools.modal_container_guard import (COST, EXIT_STOPPED, RESCHED, ContainerGuard, GuardConfig,
                                         IslandEnds, ProgressCounter, main)

APP = "yeto-s27-ch-d0a"
# the ids of s27-ch-d0a: the two original containers and the two extra ones
A, B = "ta-01M5R4W5WHK9E8PRNKV39YPG3R", "ta-01M5R4W5YGC327G6WNV0BVDZZR"
C, D = "ta-01M5R8XCBTK74HQYEZT6NWTW2R", "ta-01M5R8XC9EK8226E1ZF5AXREER"
T_ORIG, T_EXTRA = "2026-10-24T23:45:50Z", "2026-10-25T00:56:08Z"


def _row(cid: str, start: str, app: str = APP) -> dict:
    return {"app_id": "ap-test", "app_name": app, "container_id": cid, "start_time": start}


def _island_lines(*slots: tuple[int, int, str]) -> str:
    """The launch log lines of islands that name their containers: (island, rank, container id)."""
    return "".join(f"[s27-test-l{i}-modal] [modal-island {i}] rank {r} container {cid}\n" for i, r, cid in slots)


def _guard(**kw) -> ContainerGuard:
    config = dict(nodes=2, rate=48.9, extra_max_s=600.0, extra_stall_s=300.0)
    config.update(kw)
    return ContainerGuard(GuardConfig(**config), started=0.0)


ORIGINALS = [_row(A, T_ORIG), _row(B, T_ORIG)]
ORIG_LOG = _island_lines((0, 0, A), (1, 0, B))  # the log of an s27-ch-d0a style run with two 1-container islands
WITH_EXTRAS = ORIGINALS + [_row(C, T_EXTRA), _row(D, T_EXTRA)]


# -- rules ----------------------------------------------------------------------------------------


def test_extra_containers_beside_training_originals_do_not_stop_the_run():
    guard = _guard()
    assert guard.observe(30.0, ORIGINALS, progress=1).stop is None
    seen = guard.observe(60.0, WITH_EXTRAS, progress=2)
    assert seen.stop is None  # the old rule stopped here (4 distinct > 2)
    assert guard.originals == [A, B] and sorted(guard.extras) == sorted([C, D])
    events = {e["container_id"]: e for e in seen.events}
    assert events[C]["event"] == "extra_seen" and events[C]["start_time"] == T_EXTRA
    assert events[C]["originals_present"] == [A, B] and events[C]["progress_lines"] == 1
    assert guard.observe(90.0, WITH_EXTRAS, progress=3).stop is None
    gone = guard.observe(120.0, ORIGINALS, progress=4)
    assert gone.stop is None and not guard.extras
    assert sorted((e["event"], e["container_id"], e["present_s"]) for e in gone.events) == sorted([
        ("extra_gone", C, 60.0), ("extra_gone", D, 60.0)])
    assert guard.observe(10_000.0, ORIGINALS, progress=5).stop is None


def test_an_original_gone_while_an_extra_is_present_stops_the_run():
    guard = _guard()
    guard.observe(30.0, ORIGINALS, progress=1)
    decision = guard.observe(60.0, [_row(B, T_ORIG), _row(C, T_EXTRA)], progress=2)
    assert decision.stop is not None and decision.stop[0] == RESCHED
    assert A in decision.stop[1] and C in decision.stop[1]
    assert {"event": "original_gone", "container_id": A}.items() <= next(
        e for e in decision.events if e["event"] == "original_gone").items()


# s27-ek-b (2026-10-25): both islands finalized; an idle container of island 1's function
# started at 14:43:25Z; island 1's container left at its normal end.
EK_I0, EK_I1, EK_X = ("ta-01M5SQHXHTE3G91GN7YZSGMPER", "ta-01M5SQHXNQ4FPENB16T90SWYTR",
                      "ta-01M5SR86VJ7924QF3BRVTR6AXR")
EK_T0, EK_TX = "2026-10-25 14:31:13+00:00", "2026-10-25 14:43:25+00:00"
EK_ORIG = [_row(EK_I0, EK_T0), _row(EK_I1, EK_T0)]
EK_LOG = (f"[s27-ek-b-l0-modal] [modal-island 0] rank 0 container {EK_I0}\n"
          f"[s27-ek-b-l1-modal] [modal-island 1] rank 0 container {EK_I1}\n"
          '[s27-ek-b-l1-modal] YETO_RL_EVENT {"event":"rl_learner_finalized","island_id":1,"time_unix":1.0}\n')


def test_an_original_of_a_finished_island_leaving_beside_an_extra_does_not_stop():
    guard = _guard()
    guard.observe(30.0, EK_ORIG, progress=1)
    guard.observe(60.0, EK_ORIG + [_row(EK_X, EK_TX)], progress=2, finished={EK_I1})
    decision = guard.observe(90.0, [_row(EK_I0, EK_T0), _row(EK_X, EK_TX)], progress=3, finished={EK_I1})
    assert decision.stop is None
    gone = next(e for e in decision.events if e["event"] == "original_gone")
    assert gone["container_id"] == EK_I1 and gone["island_finished"] is True
    # the same poll without the end record: a replacement island, as before
    other = _guard()
    other.observe(30.0, EK_ORIG, progress=1)
    stop = other.observe(60.0, [_row(EK_I0, EK_T0), _row(EK_X, EK_TX)], progress=2).stop
    assert stop is not None and stop[0] == RESCHED and EK_I1 in stop[1]


def test_island_ends_reads_container_lines_and_finalized_events(tmp_path):
    log = tmp_path / "launch.stream.log"
    ends = IslandEnds(log)
    assert ends.read() == set()  # no file yet
    lines = EK_LOG.splitlines(keepends=True)
    log.write_text(lines[0] + lines[1] + lines[2][:40])
    assert ends.read() == set() and ends.island_of == {EK_I0: "0", EK_I1: "1"}
    with open(log, "a") as fh:
        fh.write(lines[2][40:])
    assert ends.read() == {EK_I1}
    with open(log, "a") as fh:  # key order of the record does not matter
        fh.write('YETO_RL_EVENT {"island_id": 0, "event": "rl_learner_finalized"}\n')
    assert ends.read() == {EK_I0, EK_I1}


def test_a_replacement_after_both_originals_left_stops_the_run():
    guard = _guard()
    guard.observe(30.0, ORIGINALS, progress=1)
    assert guard.observe(60.0, [], progress=1).stop is None  # no extra: the launcher sees the failure
    decision = guard.observe(90.0, [_row(C, T_EXTRA), _row(D, T_EXTRA)], progress=1)
    assert decision.stop is not None and decision.stop[0] == RESCHED and "gone" in decision.stop[1]


def test_extra_containers_present_too_long_stop_the_run():
    guard = _guard(extra_max_s=600.0)
    guard.observe(30.0, ORIGINALS, progress=1)
    guard.observe(60.0, WITH_EXTRAS, progress=2)
    assert guard.observe(660.0, WITH_EXTRAS, progress=30).stop is None  # 600 s: at the limit
    decision = guard.observe(661.0, WITH_EXTRAS, progress=31)
    assert decision.stop is not None and decision.stop[0] == RESCHED and "601 s" in decision.stop[1]


def test_no_progress_while_extras_are_present_stops_the_run():
    guard = _guard(extra_stall_s=300.0, extra_max_s=10_000.0)
    guard.observe(30.0, ORIGINALS, progress=1)
    # no progress line since t=30 (e.g. an export upload), the extras appear at t=500:
    # the stall counts from their appearance, not from the last progress line
    assert guard.observe(500.0, WITH_EXTRAS, progress=1).stop is None
    assert guard.observe(800.0, WITH_EXTRAS, progress=1).stop is None
    decision = guard.observe(801.0, WITH_EXTRAS, progress=1)
    assert decision.stop is not None and "no progress line for 301 s" in decision.stop[1]


def test_progress_resets_the_stall_and_no_progress_source_turns_the_rule_off():
    guard = _guard(extra_stall_s=300.0, extra_max_s=10_000.0)
    guard.observe(30.0, ORIGINALS, progress=1)
    guard.observe(60.0, WITH_EXTRAS, progress=1)
    assert guard.observe(300.0, WITH_EXTRAS, progress=2).stop is None
    assert guard.observe(599.0, WITH_EXTRAS, progress=2).stop is None
    assert guard.observe(601.0, WITH_EXTRAS, progress=2).stop is not None
    quiet = _guard(extra_stall_s=300.0, extra_max_s=10_000.0)
    quiet.observe(30.0, ORIGINALS)
    quiet.observe(60.0, WITH_EXTRAS)
    assert quiet.observe(5_000.0, WITH_EXTRAS).stop is None


def test_the_originals_are_the_earliest_containers_when_the_first_poll_sees_more():
    guard = _guard()
    guard.observe(30.0, [_row(C, T_EXTRA), _row(A, T_ORIG), _row(D, T_EXTRA), _row(B, T_ORIG)], progress=1)
    assert guard.originals == [A, B] and sorted(guard.extras) == sorted([C, D])


def test_a_missing_original_without_extras_and_fewer_containers_never_stop():
    guard = _guard()
    assert guard.observe(30.0, [_row(A, T_ORIG)], progress=0).stop is None  # B not started yet
    assert guard.observe(60.0, ORIGINALS, progress=1).stop is None
    assert guard.originals == [A, B] and not guard.extras
    assert guard.observe(90.0, [_row(B, T_ORIG)], progress=1).stop is None  # the island ends


# -- cost guard -----------------------------------------------------------------------------------


# -- expected island relaunches (fault injection runs) ---------------------------------------------

# a 2-island run of 2 containers each: island 1 (B, C) is killed and relaunched as (E, F),
# later killed again and relaunched as (G, H)
C1, E, F, G, H, X = (f"ta-relaunch-{n}" for n in "cefghx")
FOUR = [_row(A, T_ORIG), _row(B, T_ORIG), _row(C1, T_ORIG), _row(D, T_ORIG)]


def _relaunched(*ids: str, keep=(A, D)) -> list[dict]:
    return [_row(c, T_ORIG) for c in keep] + [_row(c, T_EXTRA) for c in ids]


def test_expected_relaunches_keep_the_run_until_more_containers_than_allowed():
    guard = _guard(nodes=4, replacements=4)
    assert guard.observe(30.0, FOUR, progress=1).stop is None
    assert guard.observe(60.0, _relaunched(), progress=2).stop is None  # island 1 gone
    # the relaunch: originals gone, new containers present (rule 1 is off) ...
    assert guard.observe(90.0, _relaunched(E, F), progress=3).stop is None
    # ... and present for longer than extra_max_s (rule 3 is off)
    assert guard.observe(2_000.0, _relaunched(E, F), progress=4).stop is None
    # a second relaunch of island 1
    assert guard.observe(2_030.0, _relaunched(), progress=5).stop is None
    assert guard.observe(2_060.0, _relaunched(G, H), progress=6).stop is None
    assert guard.observe(4_000.0, _relaunched(G, H), progress=7).stop is None
    # a fifth extra container: more than the run expected, the usual rules apply again
    stop = guard.observe(4_030.0, _relaunched(G, H, X), progress=8).stop
    assert stop is not None and stop[0] == RESCHED and "gone" in stop[1]


def test_expected_relaunches_still_stop_on_no_progress_and_on_double_billing():
    stalled = _guard(nodes=4, replacements=2)
    stalled.observe(30.0, FOUR, progress=1)
    assert stalled.observe(60.0, _relaunched(E, F), progress=1).stop is None
    stop = stalled.observe(400.0, _relaunched(E, F), progress=1).stop
    assert stop is not None and "no progress line" in stop[1]

    double = _guard(nodes=4, replacements=2)
    double.observe(30.0, FOUR, progress=1)
    # the relaunched island starts before the old island's containers left
    assert double.observe(60.0, FOUR + [_row(E, T_EXTRA), _row(F, T_EXTRA)], progress=2).stop is None
    assert double.observe(600.0, FOUR + [_row(E, T_EXTRA), _row(F, T_EXTRA)], progress=3).stop is None
    stop = double.observe(700.0, FOUR + [_row(E, T_EXTRA), _row(F, T_EXTRA)], progress=4).stop
    assert stop is not None and stop[0] == RESCHED and "beside old ones" in stop[1]


def test_no_expected_relaunch_by_default():
    guard = _guard(nodes=4)
    guard.observe(30.0, FOUR, progress=1)
    stop = guard.observe(60.0, _relaunched(E, F), progress=2).stop
    assert stop is not None and "replacement island" in stop[1]
    with pytest.raises(ValueError):
        _guard(replacements=-1)


def test_the_cost_counts_every_container_and_the_head():
    guard = _guard(rate=36.0, head_rate=3.6, cost_cap=1000.0)
    assert guard.observe(100.0, ORIGINALS).cost == pytest.approx(2 * 1.0 + 0.1)
    # 100 s with 4 containers (extras billed too) + head
    assert guard.observe(200.0, WITH_EXTRAS).cost == pytest.approx(2.0 + 4 * 1.0 + 0.2)
    # a failed list counts the last known number of containers
    assert guard.observe(300.0, None).cost == pytest.approx(6.0 + 4 * 1.0 + 0.3)
    assert guard.observe(400.0, ORIGINALS).cost == pytest.approx(10.0 + 2 * 1.0 + 0.4)


def test_the_cost_alert_fires_once_and_the_cap_override_wins():
    guard = _guard(rate=36.0, cost_cap=100.0, cost_alert=3.0, extra_max_s=10_000.0)
    first = guard.observe(100.0, ORIGINALS)
    assert not first.alert and first.stop is None
    second = guard.observe(200.0, ORIGINALS)
    assert second.alert and second.stop is None
    assert not guard.observe(300.0, ORIGINALS).alert
    decision = guard.observe(400.0, ORIGINALS, cap_override=7.5)
    assert decision.stop == (COST, "estimated cost 8.0 >= 7.5")
    assert guard.observe(500.0, WITH_EXTRAS).stop is None  # stops once


def test_the_cost_cap_also_stops_a_run_with_extras():
    guard = _guard(rate=360.0, cost_cap=30.0, extra_max_s=10_000.0)
    guard.observe(100.0, ORIGINALS)
    decision = guard.observe(200.0, WITH_EXTRAS)
    assert decision.stop is not None and decision.stop[0] == COST


# -- progress lines -------------------------------------------------------------------------------


def test_progress_counter_reads_only_complete_new_lines(tmp_path):
    log = tmp_path / "launch.stream.log"
    pattern = r"'train/step'|\"event\":\"rl_heartbeat\".*\"phase\":\"finish\""
    counter = ProgressCounter(log, pattern)
    assert counter.read() == 0  # no file yet
    log.write_text("x 'train/step': 1\nnoise\n{\"event\":\"rl_heartbeat\",\"phase\":\"fin")
    assert counter.read() == 1
    with open(log, "a") as fh:
        fh.write("ish\"}\n'train/step' 2\n")
    assert counter.read() == 3
    assert counter.read() == 3


# -- the command line, with a fake container list -------------------------------------------------

FAKE_LIST = textwrap.dedent("""
    import json, sys
    from pathlib import Path
    steps, counter, progress_log = map(Path, sys.argv[1:4])
    n = int(counter.read_text()) if counter.exists() else 0
    counter.write_text(str(n + 1))
    plan = json.loads(steps.read_text())
    step = plan[min(n, len(plan) - 1)]
    if step.get("progress") or step.get("log"):
        with open(progress_log, "a") as fh:
            fh.write(step.get("log", ""))
            if step.get("progress"):
                fh.write("'train/step': %d\\n" % n)
    if step.get("fail"):
        sys.exit(1)
    print(json.dumps(step["rows"]))
""")


def _cli(tmp_path: Path, steps: list[dict], *extra: str) -> tuple[int, Path]:
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    fake = tmp_path / "fake_list.py"
    fake.write_text(FAKE_LIST)
    (tmp_path / "steps.json").write_text(json.dumps(steps))
    log = run / "launch.stream.log"
    list_cmd = f"{sys.executable} {fake} {tmp_path / 'steps.json'} {tmp_path / 'counter'} {log}"
    rc = main(["--run-dir", str(run), "--app", APP, "--nodes", "2", "--rate", "36", "--cost-cap", "500",
               "--list-cmd", list_cmd, "--list-cwd", str(tmp_path), "--interval-s", "0", "--no-diagnose",
               "--progress-log", str(log), "--progress-re", "'train/step'",
               "--stop-cmd", f"echo \"{{kind}} $GUARD_KIND\" >> {tmp_path / 'stops.txt'}", *extra])
    return rc, run


def test_cli_keeps_the_run_with_extras_and_records_them(tmp_path):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "launch.stream.log").write_text(ORIG_LOG)
    other = _row("ta-OTHER", T_ORIG, app="another-app")
    steps = [{"rows": ORIGINALS + [other], "progress": True},
             {"rows": WITH_EXTRAS + [other], "progress": True},
             {"fail": True, "progress": True},
             {"rows": WITH_EXTRAS, "progress": True},
             {"rows": ORIGINALS, "progress": True}]
    rc, run = _cli(tmp_path, steps, "--max-polls", "5")
    assert rc == 0 and not (tmp_path / "stops.txt").exists()
    lines = (run / "containers.log").read_text().splitlines()
    assert [line.split()[1] for line in lines] == ["n=2", "n=4", "n=4", "n=2"]  # the failed list: no line
    assert (run / "poll/seen.txt").read_text().split() == [A, B, D, C]  # extras by (start_time, id)
    assert (run / "poll/current.txt").read_text().split() == [A, B]
    events = [json.loads(line) for line in (run / "poll/extra-containers.jsonl").read_text().splitlines()]
    assert [(e["event"], e["container_id"]) for e in events] == [
        ("extra_seen", D), ("extra_seen", C), ("extra_gone", D), ("extra_gone", C)]
    assert "container list failed" in (run / "poll/guard.log").read_text()
    assert (run / "first_container_utc.txt").is_file() and "est_cost_usd=" in (run / "cost.now").read_text()
    state = json.loads((run / "poll/guard-state.json").read_text())
    assert state["originals"] == [A, B] and state["stopped"] is None


def test_cli_stops_once_when_an_original_is_replaced(tmp_path):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "launch.stream.log").write_text(ORIG_LOG)
    steps = [{"rows": ORIGINALS, "progress": True},
             {"rows": [_row(B, T_ORIG), _row(C, T_EXTRA)], "progress": True},
             {"rows": ORIGINALS, "progress": True}]
    rc, run = _cli(tmp_path, steps, "--max-polls", "10")
    assert rc == EXIT_STOPPED
    assert (tmp_path / "stops.txt").read_text() == "resched resched\n"
    assert "gone while extra container(s)" in (run / "RESCHED_FIRED").read_text()
    assert len((run / "stopped.log").read_text().splitlines()) == 1
    assert int((tmp_path / "counter").read_text()) == 2  # no poll after the stop


def test_cli_keeps_a_finished_run_whose_island_left_beside_an_extra(tmp_path):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "launch.stream.log").write_text(EK_LOG)
    steps = [{"rows": EK_ORIG, "progress": True},
             {"rows": EK_ORIG + [_row(EK_X, EK_TX)], "progress": True},
             {"rows": [_row(EK_I0, EK_T0), _row(EK_X, EK_TX)], "progress": True},
             {"rows": [], "progress": True}]
    rc, run = _cli(tmp_path, steps, "--max-polls", "4")
    assert rc == 0 and not (tmp_path / "stops.txt").exists() and not (run / "RESCHED_FIRED").exists()
    events = [json.loads(line) for line in (run / "poll/extra-containers.jsonl").read_text().splitlines()]
    assert [(e["event"], e["container_id"], e.get("island_finished")) for e in events] == [
        ("extra_seen", EK_X, None), ("original_gone", EK_I1, True),
        ("extra_gone", EK_X, None), ("original_gone", EK_I0, False)]


def test_extra_log_capture_names_the_function_in_a_time_window(tmp_path, monkeypatch):
    import tools.modal_container_guard as guard_mod

    commands = []

    class FakePopen:
        def __init__(self, command, **kwargs):
            commands.append(command)

    monkeypatch.setattr(guard_mod.subprocess, "Popen", FakePopen)
    args = guard_mod.parse_args(["--run-dir", str(tmp_path), "--app", APP, "--nodes", "2", "--rate", "1",
                                 "--max-polls", "1"])
    now = 1792939422.0  # 2026-10-25T14:43:42Z, the poll that saw the s27-ek-b extra container
    guard_mod._diagnose_extra(args, guard_mod.RunFiles(tmp_path), "ap-x", EK_X, EK_TX, now)
    guard_mod._diagnose_extra(args, guard_mod.RunFiles(tmp_path), None, EK_X, EK_TX, now)  # no app id
    assert len(commands) == 1
    command = commands[0]
    assert "--show-function-id" in command and command[command.index("--container") + 1] == EK_X
    assert command[command.index("--since") + 1] == "2026-10-25T14:42:25+00:00"  # start - 60 s
    assert command[command.index("--until") + 1] == "2026-10-25T14:44:42+00:00"  # poll + 60 s
    assert guard_mod._log_window("not a time", now)[1] == "2026-10-25T14:33:42+00:00"  # poll - 600 s


def test_cli_cost_cap_override_file_stops_the_run(tmp_path):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "COST_CAP_OVERRIDE").write_text("0\n")
    rc, run = _cli(tmp_path, [{"rows": ORIGINALS, "progress": True}], "--max-polls", "3")
    assert rc == EXIT_STOPPED and (tmp_path / "stops.txt").read_text() == "cost cost\n"
    assert ">= 0.0" in (run / "COST_FIRED").read_text()


def test_cli_needs_an_end_condition_and_paired_progress_flags(tmp_path):
    base = ["--run-dir", str(tmp_path), "--app", APP, "--nodes", "2", "--rate", "1"]
    with pytest.raises(SystemExit):
        main(base)
    with pytest.raises(SystemExit):
        main(base + ["--max-polls", "1", "--progress-log", str(tmp_path / "x")])


def test_queued_tasks_count_for_the_cost_but_not_as_containers():
    guard = _guard(rate=36.0)
    queued = {"app_name": APP, "container_id": "ta-QUEUED", "start_time": "Pending"}
    decision = guard.observe(100.0, ORIGINALS + [queued, {**queued, "container_id": "ta-Q2", "start_time": None}])
    assert decision.stop is None and guard.originals == [A, B] and not guard.extras
    assert decision.cost == pytest.approx(4 * 1.0)


def test_older_cli_column_names_are_read(tmp_path):
    from tools.modal_container_guard import list_containers

    rows = [{"Container ID": A, "App Name": APP, "Start Time": T_ORIG}, {"Container ID": "x", "App Name": "o"}]
    script = tmp_path / "list.py"
    script.write_text(f"import json; print(json.dumps({rows!r}))")
    found, why = list_containers([sys.executable, str(script)], APP)
    assert why == "" and found == [{"container_id": A, "app_name": APP, "start_time": T_ORIG}]
    failed, why = list_containers([sys.executable, "-c", "import sys; sys.exit(2)"], APP)
    assert failed is None and why.startswith("rc=2")


# -- the island containers are the ones the islands name (s27-ex-1) ---------------------------------

# s27-ex-1 (2 islands x 1 container, new image): "Built image ... in 723.62s" (12 min) and a second
# image "in 10.40s" ran in containers of the app. The guard listed B1 from 18:03:46Z, B2 from
# 18:05:50Z, the island containers I0, I1 (function calls at 18:05:58Z) at 18:06:21Z and stopped the
# app there: "original container(s) [B1, B2] gone while extra container(s) [I0, I1]". The ids and
# times are the run's (the start times of the build containers are the ULID creation times).
EX_B1, EX_B2 = "ta-01M5T34A3M3TMN7CN0JKWT9T3B", "ta-01M5T3TVWM8ZN4PH5FF13MY1AB"
EX_I0, EX_I1 = "ta-01M5T3V8D52Y7T6SCS9F1H6TYR", "ta-01M5T3V8HP9WWE4VS7JC3A4YRR"
EX_X = "ta-01M5T3XEXTRAIDLE0000000000"  # an idle extra container of island 0's function (made up)
EX_TB1, EX_TB2 = "2026-10-25 17:53:27+00:00", "2026-10-25 18:05:46+00:00"
EX_T0, EX_T1 = "2026-10-25 18:06:09+00:00", "2026-10-25 18:06:11+00:00"
EX_BUILD1, EX_BUILD2 = [_row(EX_B1, EX_TB1)], [_row(EX_B2, EX_TB2)]
EX_ISLANDS = [_row(EX_I0, EX_T0), _row(EX_I1, EX_T1)]
EX_LINES = _island_lines((0, 0, EX_I0), (1, 0, EX_I1))


def _names(owners=(), others=()):
    from tools.modal_container_guard import IslandNames

    return IslandNames(owners=tuple(owners), others=frozenset(others))


def test_build_containers_before_the_islands_do_not_stop_the_run():
    """s27-ex-1 on the guard's data: the build containers come and go, the islands' containers
    come, their lines reach the log one poll later."""
    guard = _guard(rate=11.5)
    events = []
    for t, rows, owners in ((0.0, EX_BUILD1, []), (30.0, EX_BUILD1, []), (60.0, EX_BUILD1, []),
                            (90.0, EX_BUILD2, []),  # B1 left, B2 came (the second image)
                            (120.0, EX_ISLANDS, []),  # B2 left, the islands' lines are not in the log yet
                            (150.0, EX_ISLANDS, [EX_I0, EX_I1]), (10_000.0, EX_ISLANDS, [EX_I0, EX_I1])):
        decision = guard.observe(t, rows, progress=0, names=_names(owners))
        assert decision.stop is None, (t, decision.stop)
        events += [(e["event"], e["container_id"]) for e in decision.events]
        if t == 120.0:
            assert guard.originals == [] and not guard.extras and sorted(guard.unnamed) == sorted([EX_I0, EX_I1])
    assert guard.originals == [EX_I0, EX_I1] and not guard.unnamed and not guard.extras
    assert not guard.originals_gone
    # the build containers are recorded and none of them is an extra container or a gone original
    assert events == [("unnamed_seen", EX_B1), ("unnamed_seen", EX_B2), ("unnamed_gone", EX_B1),
                      ("unnamed_seen", EX_I0), ("unnamed_seen", EX_I1), ("unnamed_gone", EX_B2),
                      ("unnamed_claimed", EX_I0), ("unnamed_claimed", EX_I1)]


def test_the_old_rule_without_names_stops_on_the_same_poll_sequence():
    """The positional rule (no names) still takes B1, B2 for the originals: what s27-ex-1 did."""
    guard = _guard(rate=11.5)
    for t, rows in ((0.0, EX_BUILD1), (30.0, EX_BUILD1), (60.0, EX_BUILD2)):
        assert guard.observe(t, rows, progress=0).stop is None
    stop = guard.observe(90.0, EX_ISLANDS, progress=0).stop
    assert stop is not None and stop[0] == RESCHED and EX_B1 in stop[1] and EX_I0 in stop[1]
    assert guard.originals == [EX_B1, EX_B2]


def test_a_container_left_unnamed_after_all_islands_are_named_is_an_extra_from_that_poll():
    guard = _guard(extra_max_s=600.0, extra_stall_s=10_000.0)
    owners = [EX_I0, EX_I1]
    # a build container (or an idle extra) is listed beside the islands for a long time while one
    # island has not printed its line (e.g. it waits for its GPU)
    both = [_row(EX_B2, EX_TB2)] + EX_ISLANDS
    assert guard.observe(0.0, both, progress=1, names=_names([EX_I0])).stop is None
    assert guard.observe(2_000.0, both, progress=2, names=_names([EX_I0])).stop is None  # no rule counts it
    assert sorted(guard.unnamed) == sorted([EX_B2, EX_I1]) and not guard.extras
    # all named at t=2000 s: the container is an extra from now on (the clock starts here, not at t=0)
    promoted = guard.observe(2_030.0, both, progress=3, names=_names(owners))
    assert promoted.stop is None and sorted(guard.extras) == [EX_B2] and guard.unnamed == {}
    seen = next(e for e in promoted.events if e["event"] == "extra_seen")
    assert seen["unnamed_s"] == 2_030.0 and seen["originals_present"] == owners
    assert guard.observe(2_630.0, both, progress=4, names=_names(owners)).stop is None  # 600 s: at the limit
    stop = guard.observe(2_631.0, both, progress=5, names=_names(owners)).stop
    assert stop is not None and stop[0] == RESCHED and "601 s" in stop[1]


def test_after_all_islands_are_named_the_rules_are_the_old_ones():
    guard = _guard(extra_max_s=600.0)
    owners = [EX_I0, EX_I1]
    guard.observe(0.0, EX_ISLANDS, progress=1, names=_names(owners))
    # an idle extra container beside the islands (G-EO): only a warning, then the 600 s limit
    extra = [_row(EX_X, "2026-10-25 18:30:00+00:00")]
    seen = guard.observe(60.0, EX_ISLANDS + extra, progress=2, names=_names(owners))
    assert seen.stop is None and [e["event"] for e in seen.events] == ["extra_seen"]
    assert guard.observe(660.0, EX_ISLANDS + extra, progress=3, names=_names(owners)).stop is None
    assert guard.observe(661.0, EX_ISLANDS + extra, progress=4, names=_names(owners)).stop is not None
    # an island that finished and left beside an extra container (s27-ek-b): not a replacement
    done = _guard()
    done.observe(0.0, EX_ISLANDS, progress=1, names=_names(owners))
    done.observe(30.0, EX_ISLANDS + extra, progress=2, names=_names(owners), finished={EX_I1})
    assert done.observe(60.0, [EX_ISLANDS[0]] + extra, progress=3, names=_names(owners),
                        finished={EX_I1}).stop is None
    # an island that left without finishing beside an extra container: a replacement island
    lost = _guard()
    lost.observe(0.0, EX_ISLANDS, progress=1, names=_names(owners))
    lost.observe(30.0, EX_ISLANDS + extra, progress=2, names=_names(owners))
    stop = lost.observe(60.0, [EX_ISLANDS[0]] + extra, progress=3, names=_names(owners)).stop
    assert stop is not None and stop[0] == RESCHED and EX_I1 in stop[1]


def test_a_container_that_runs_the_island_script_again_is_an_extra_at_once():
    """The log names a second container for a slot that has one: Modal ran the island again in a
    new container (s27-gr-h). That is a proof, so it counts before the other island is named."""
    guard = _guard()
    guard.observe(0.0, [EX_ISLANDS[0]], progress=1, names=_names([EX_I0]))
    replaced = guard.observe(30.0, [_row(EX_X, "2026-10-25 18:30:00+00:00")], progress=2,
                             names=_names([EX_I0], others=[EX_X]))
    assert replaced.stop is not None and replaced.stop[0] == RESCHED
    assert EX_I0 in replaced.stop[1] and EX_X in replaced.stop[1]
    assert [e["event"] for e in replaced.events] == ["extra_seen", "original_gone"]
    # a pending container that prints a taken slot's line later becomes an extra
    late = _guard()
    late.observe(0.0, [EX_ISLANDS[0], _row(EX_X, "2026-10-25 18:30:00+00:00")], progress=1,
                 names=_names([EX_I0]))
    assert late.unnamed.keys() == {EX_X}
    late.observe(30.0, [EX_ISLANDS[0], _row(EX_X, "2026-10-25 18:30:00+00:00")], progress=2,
                 names=_names([EX_I0], others=[EX_X]))
    assert late.extras.keys() == {EX_X} and not late.unnamed


def test_a_named_container_the_list_does_not_show_yet_is_not_gone():
    guard = _guard()
    # the line is in the log, the container is not in the list yet (the list was read first)
    assert guard.observe(0.0, [EX_ISLANDS[0]], progress=1, names=_names([EX_I0, EX_I1])).stop is None
    assert guard.originals == [EX_I0, EX_I1] and not guard.originals_gone
    shown = guard.observe(30.0, EX_ISLANDS, progress=2, names=_names([EX_I0, EX_I1]))
    assert shown.stop is None and not shown.events and not guard.unnamed
    gone = guard.observe(60.0, [EX_ISLANDS[0]], progress=3, names=_names([EX_I0, EX_I1]))
    assert [e["event"] for e in gone.events] == ["original_gone"] and gone.stop is None  # no extra: no stop


def test_pending_containers_do_not_count_for_the_double_billing_limit_of_relaunches():
    guard = _guard(nodes=4, replacements=2, extra_max_s=600.0)
    ids = [f"ta-island-{n}" for n in range(4)]
    four = [_row(cid, T_ORIG) for cid in ids]
    five = four + [_row("ta-build", T_EXTRA)]
    # 5 started containers, but the build container and the fourth island's container are pending
    guard.observe(0.0, five, progress=1, names=_names(ids[:3]))
    assert guard.over_since is None and sorted(guard.unnamed) == ["ta-build", "ta-island-3"]
    # all four islands are named: the build container is an extra container from this poll on
    guard.observe(30.0, five, progress=2, names=_names(ids))
    assert guard.over_since == 30.0 and list(guard.extras) == ["ta-build"]
    assert guard.observe(601.0, five, progress=3, names=_names(ids)).stop is None  # 571 s since t=30
    stop = guard.observe(631.0, five, progress=4, names=_names(ids)).stop
    assert stop is not None and "for 601 s" in stop[1] and "beside old ones" in stop[1]


def test_the_state_tells_the_islands_containers_from_the_pending_ones():
    """A run script that counts started containers (poll/guard-state.json) can count the islands' own."""
    guard = _guard()
    guard.observe(0.0, EX_BUILD1, names=_names())
    state = guard.state()
    assert state["present"] == [EX_B1] and state["originals_present"] == []
    assert list(state["unnamed_present"]) == [EX_B1]
    guard.observe(30.0, EX_BUILD1 + [EX_ISLANDS[0]], names=_names([EX_I0]))
    assert guard.state()["present"] == [EX_B1, EX_I0] and guard.state()["originals_present"] == [EX_I0]


def test_a_pending_container_gets_one_warning_after_the_limit():
    from tools.modal_container_guard import unnamed_to_warn

    guard, warned = _guard(), set()
    guard.observe(0.0, EX_BUILD1 + [EX_ISLANDS[0]], names=_names())
    assert unnamed_to_warn(guard, 100.0, warned, 120.0) == []  # an island container waits for its line
    guard.observe(110.0, EX_BUILD1, names=_names())
    assert sorted(unnamed_to_warn(guard, 121.0, warned, 120.0)) == [EX_B1]  # the island container left meanwhile
    assert unnamed_to_warn(guard, 500.0, warned, 120.0) == []  # once


def test_island_ends_names_the_first_container_of_each_slot(tmp_path):
    log = tmp_path / "launch.stream.log"
    ends = IslandEnds(log)
    assert ends.names().owners == () and ends.names().others == frozenset()
    log.write_text(_island_lines((0, 0, "ta-a"), (0, 1, "ta-b"), (1, 0, "ta-c")) + "[x] [modal-island 1] rank 1 con")
    ends.read()
    assert ends.names().owners == ("ta-a", "ta-b", "ta-c") and not ends.names().others
    with open(log, "a") as fh:  # the rest of the half line, a second container for slot (0, 1), a repeat
        fh.write("tainer ta-d\n" + _island_lines((0, 1, "ta-e"), (0, 0, "ta-a")))
    ends.read()
    names = ends.names()
    assert names.owners == ("ta-a", "ta-b", "ta-c", "ta-d") and names.others == frozenset({"ta-e"})


def test_cli_build_containers_before_the_islands_do_not_stop_the_run(tmp_path):
    """The s27-ex-1 poll sequence through the command line. With the positional rule of
    0024e104 this run ends with exit 3 and RESCHED_FIRED."""
    steps = ([{"rows": EX_BUILD1}] * 4 + [{"rows": EX_BUILD2}, {"rows": EX_ISLANDS},  # lines not in the log yet
              {"rows": EX_ISLANDS, "log": EX_LINES, "progress": True}] + [{"rows": EX_ISLANDS, "progress": True}] * 2)
    rc, run = _cli(tmp_path, steps, "--max-polls", str(len(steps)), "--unnamed-warn-s", "-1")
    assert rc == 0 and not (tmp_path / "stops.txt").exists() and not (run / "RESCHED_FIRED").exists()
    events = [json.loads(line) for line in (run / "poll/extra-containers.jsonl").read_text().splitlines()]
    assert [(e["event"], e["container_id"]) for e in events] == [
        ("unnamed_seen", EX_B1), ("unnamed_seen", EX_B2), ("unnamed_gone", EX_B1),
        ("unnamed_seen", EX_I0), ("unnamed_seen", EX_I1), ("unnamed_gone", EX_B2),
        ("unnamed_claimed", EX_I0), ("unnamed_claimed", EX_I1)]
    guard_log = (run / "poll/guard.log").read_text()
    assert f"WARNING: container {EX_B1} is not named" in guard_log and "originals=by name" in guard_log
    assert guard_log.count("WARNING: container") == 4  # each pending container once (the islands' too, at -1 s)
    state = json.loads((run / "poll/guard-state.json").read_text())
    assert state["originals"] == [EX_I0, EX_I1] and state["unnamed_present"] == {} and state["by_name"] is True
    assert state["present"] == state["originals_present"] == [EX_I0, EX_I1]
    assert [g["container_id"] for g in state["unnamed_gone"]] == [EX_B1, EX_B2]
    assert (run / "poll/seen.txt").read_text().split() == [EX_B1, EX_B2, EX_I0, EX_I1]


def test_cli_saves_the_log_of_a_container_that_left_unnamed(tmp_path, monkeypatch):
    import tools.modal_container_guard as guard_mod

    calls = []
    monkeypatch.setattr(guard_mod, "_diagnose_extra",
                        lambda args, files, app_id, cid, start_time=None, now=None: calls.append((app_id, cid)))
    steps = [{"rows": EX_BUILD1}, {"rows": EX_BUILD2}, {"rows": EX_ISLANDS},
             {"rows": EX_ISLANDS, "log": EX_LINES, "progress": True}]
    run = tmp_path / "run"
    run.mkdir()
    (tmp_path / "steps.json").write_text(json.dumps(steps))
    fake = tmp_path / "fake_list.py"
    fake.write_text(FAKE_LIST)
    log = run / "launch.stream.log"
    rc = main(["--run-dir", str(run), "--app", APP, "--nodes", "2", "--rate", "36", "--max-polls", "4",
               "--list-cmd", f"{sys.executable} {fake} {tmp_path / 'steps.json'} {tmp_path / 'counter'} {log}",
               "--list-cwd", str(tmp_path), "--interval-s", "0", "--progress-log", str(log),
               "--progress-re", "'train/step'"])
    # the build containers (left unnamed) are captured; an island container is not (it was unnamed for one poll)
    assert rc == 0 and calls == [("ap-test", EX_B1), ("ap-test", EX_B2)]


def test_cli_positional_originals_keeps_the_old_rule_for_a_log_without_island_lines(tmp_path):
    steps = [{"rows": ORIGINALS, "progress": True},
             {"rows": [_row(B, T_ORIG), _row(C, T_EXTRA)], "progress": True}]
    rc, run = _cli(tmp_path, steps, "--max-polls", "10", "--positional-originals")
    assert rc == EXIT_STOPPED and "gone while extra container(s)" in (run / "RESCHED_FIRED").read_text()
    assert "originals=first started" in (run / "poll/guard.log").read_text()
    # the same polls with names expected but none in the log: nothing is an island container, nothing stops
    other = tmp_path / "other"
    other.mkdir()
    rc, run = _cli(other, steps, "--max-polls", "3")
    assert rc == 0 and not (run / "RESCHED_FIRED").exists()
