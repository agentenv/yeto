"""tools/modal_container_guard.py: the shared container guard and cost guard of the
Modal run scripts, run offline on made-up ``modal container list --json`` output.

s27-ch-d0a (2026-10-25): two extra containers of the island function appeared at
00:56:15Z while the two original containers trained (step 66 at 00:56:11Z); the old
shell rule ("more distinct containers than NODES") stopped the healthy run."""

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


def _guard(**kw) -> ContainerGuard:
    config = dict(nodes=2, rate=48.9, extra_max_s=600.0, extra_stall_s=300.0)
    config.update(kw)
    return ContainerGuard(GuardConfig(**config), started=0.0)


ORIGINALS = [_row(A, T_ORIG), _row(B, T_ORIG)]
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
    if step.get("progress"):
        with open(progress_log, "a") as fh:
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
