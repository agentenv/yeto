#!/usr/bin/env python3
"""rl-spot-cost-saving 3.2: reclaim drill on the Modal eval island.

Runs the real eval-island Modal function (``yeto.cloud.modal_eval_island``:
store on a Modal Volume, unit log, the #180 reclaim handler, launcher restart)
with the drill loader/attempt (``yeto.rl.eval.drill``: no model, each unit
waits ``UNIT_S``). After ``SIGNAL_AFTER`` results are committed, the driver
sends SIGINT to the function's process inside the container
(``modal container exec``), which is what Modal sends on a preemption. Raw
data stays in ``--out``; the judgment is written to ``judgment.json``.

Pre-registered criteria (S19-SPOT-EVAL-DRILL-REVIEW.md):
  D1 exactly one ``spot_reclaim`` event; cloud=modal, role=eval, source=modal_signal,
     remaining_s=25, saved=true, outcome=saved
  D2 the handler finished within 25 s (handler_s <= 25) and the first container
     ended (launcher event ``preempted``) within 60 s of the signal
  D3 the launcher started a second container which finished (``finished``)
  D4 resume: final ``rl_eval`` has eval/units == planned units, eval/duplicate_results == 0;
     results file has every planned unit exactly once
  D5 eval/results_sha256 equals a local one-shot run of the same plan (no loss, no change)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import modal  # noqa: E402

from yeto.cloud import modal_eval_island as mei  # noqa: E402
from yeto.cloud.modal_eval_island import _Uploader  # noqa: E402
from yeto.rl.eval.store import EvalStore  # noqa: E402

TASKS = ["fix-ocaml-gc", "raman-fitting", "sanitize-git-repo", "winning-avg-corewars"]
TRIALS_V0 = 2
UNIT_S = 15.0
SIGNAL_AFTER = 3
SAMPLING = {"rollout_temperature": 1.0, "rollout_top_p": 1.0, "drill": True}
MODAL = "/home/michael/work/reward-env-venv/bin/modal"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--app", default="yeto-s19-spot-drill")
    ap.add_argument("--volume", required=True)
    ap.add_argument("--gpu", default="T4")
    ap.add_argument("--hard-s", type=int, default=1500)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "drill.log", "a")

    def say(*parts):
        line = time.strftime("%H:%M:%SZ ", time.gmtime()) + " ".join(str(p) for p in parts)
        print(line, file=log, flush=True)
        print(line, flush=True)

    events: list[dict] = []

    def emit(event, **f):
        events.append({"event": event, "ts": time.time(), **f})
        with open(out / "launcher-events.jsonl", "a") as fh:
            fh.write(json.dumps(events[-1], default=str) + "\n")

    # 1) seed version 0 (drill files) into the Volume through the staging uploader
    # NOTE: --volume must be a FRESH volume. A reused volume keeps the last
    # run's results/ and done markers: the island then ends in seconds without
    # work and the signaller fires before any container exists
    # (s19-spot-drill-20261009b, 14:52:46Z wall_s=1.7).
    vol = modal.Volume.from_name(a.volume, create_if_missing=True)
    stage = out / "stage"
    store = EvalStore(stage, commit=_Uploader(stage, vol))
    store.put_version(0, files={"policy.safetensors": b"drill", "policy.json": b"{}"},
                      policy_tensor_hash="d" * 64, policy_token="yeto:0:drill", sampling=SAMPLING)
    store.mark_training_finished({"final_version": 0, "note": "spot drill"})
    say("seeded", a.volume)

    empty = out / "empty"
    (empty / "codex").mkdir(parents=True, exist_ok=True)
    (empty / "tb2").mkdir(parents=True, exist_ok=True)
    spec = mei.EvalFunctionSpec(app_name=a.app, volume=a.volume, workdir=str(REPO), codex_dir=str(empty / "codex"),
                                tb2_dir=str(empty / "tb2"), gpu=a.gpu, gpus=1, cpu=2.0, memory_mib=8192,
                                timeout_s=a.hard_s, envs={})
    app, island_fn, _seed = mei.build_eval_app(spec)
    config = {"store": mei.EVAL_STORE_MOUNT, "volume": a.volume, "island_id": "eval-spot",
              "holdout": f"{mei.WORKDIR_MOUNT}/data/eval/tb2-holdout.json", "task_ids": TASKS,
              "trials_v0": TRIALS_V0, "trials": 1, "wait_for_training": True, "poll_s": 5.0,
              "loader": "yeto.rl.eval.drill:loader_factory", "attempt": "yeto.rl.eval.drill:attempt_factory",
              "drill_unit_s": UNIT_S, "region": "modal-auto", "sampling": SAMPLING}
    signal: dict = {}

    def results_count() -> int:
        try:
            data = b"".join(vol.read_file("results/v000000.jsonl"))
        except Exception:  # noqa: BLE001 - not written yet
            return 0
        return sum(1 for line in data.decode().splitlines() if '"kind": "result"' in line)

    def signaller(app_id: str) -> None:
        deadline = time.time() + a.hard_s
        while time.time() < deadline and results_count() < SIGNAL_AFTER:
            time.sleep(3)
        listing = subprocess.run([MODAL, "container", "list", "--json"], capture_output=True, text=True, timeout=60)
        rows = [r for r in json.loads(listing.stdout or "[]") if app_id in json.dumps(r)]
        (out / "containers-before-signal.json").write_text(json.dumps(rows, indent=1))
        if not rows:
            say("SIGNAL: no container found for", app_id)
            return
        cid = rows[0].get("Container ID") or rows[0].get("container_id") or next(iter(rows[0].values()))
        # dump every cmdline, then pick the modal entrypoint; drill 3 showed the
        # `_container_entrypoint` glob matched nothing, so keep a fallback chain
        # and leave the full dump in container-procs.txt for diagnosis.
        find = ("for p in /proc/[0-9]*; do c=$(tr '\\0' ' ' < $p/cmdline 2>/dev/null); "
                "[ -n \"$c\" ] && echo ${p#/proc/} $c; done")
        # `--` stops the modal CLI from parsing the container command's own
        # options: without it `bash -c` dies with "No such option '-c'"
        # (s19-spot-drill-20261009a, 12:19:39Z: signal never delivered).
        ps = subprocess.run([MODAL, "container", "exec", "--no-pty", cid, "--", "bash", "-c", find],
                            capture_output=True, text=True, timeout=120)
        (out / "container-procs.txt").write_text(ps.stdout + ps.stderr)
        lines = [line for line in ps.stdout.splitlines() if line.strip()]
        # target: the modal function process itself -- cmdline STARTS with
        # python and runs `-m modal._container_entrypoint`.  pid 1 is
        # dumb-init wrapping the same text; SIGINT to it never reaches the
        # function (s19-spot-drill-20261009d: kill_rc=0, no spot_reclaim).
        target = next((l for l in lines
                       if l.split(maxsplit=1)[1].startswith("python") and "_container_entrypoint" in l), None)
        if not target:
            say("SIGNAL: no entrypoint process", ps.stdout[-300:], ps.stderr[-300:])
            return
        pids = [target.split()[0]]
        signal.update(container=cid, pid=pids[0], target_cmd=target.split(maxsplit=1)[1][:300],
                      results_before=results_count(), t=time.time())
        kill = subprocess.run([MODAL, "container", "exec", "--no-pty", cid, "--", "kill", "-INT", pids[0]],
                              capture_output=True, text=True, timeout=120)
        signal.update(kill_rc=kill.returncode, kill_out=(kill.stdout + kill.stderr)[-500:])
        (out / "signal.json").write_text(json.dumps(signal, indent=1))
        say("SIGNAL sent", json.dumps(signal))

    def watchdog():
        time.sleep(a.hard_s + 300)
        say("WATCHDOG: hard limit passed, stopping app")
        os.system(f"{MODAL} app stop {a.app} >> {out}/watchdog.out 2>&1")
        os._exit(3)

    threading.Thread(target=watchdog, daemon=True).start()
    summary = None
    with modal.enable_output(), app.run():
        app_id = app.app_id
        (out / "app_id.txt").write_text(app_id + "\n")
        say("app", app_id)
        threading.Thread(target=signaller, args=(app_id,), daemon=True).start()
        launcher = mei.EvalIslandLauncher(client=_ReclaimedContainerClient(island_fn, signal, say, out),
                                          emit=emit, config=config,
                                          gpu=a.gpu, gpus=1, price_per_hour=None, max_starts=3)
        try:
            summary = launcher.run()
            say("island done", json.dumps(summary, default=str)[:1500])
        except Exception as exc:  # noqa: BLE001
            say("island FAILED", type(exc).__name__, str(exc)[:1500])
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))

    # raw data first: the whole Volume
    dump = out / "volume"
    # trial 6 (s19-spot-drill-20261009f): without an existing target dir and
    # with the repo as cwd the get failed ("[Errno 21] Is a directory"); the
    # same get into an existing dir from /tmp worked
    dump.mkdir(parents=True, exist_ok=True)
    got = subprocess.run([MODAL, "volume", "get", "--force", a.volume, "/", str(dump)],
                         capture_output=True, text=True, timeout=600, cwd="/tmp")
    (out / "volume-get.out").write_text(f"rc={got.returncode}\n{got.stdout[-2000:]}\n{got.stderr[-2000:]}")
    apps = subprocess.run([MODAL, "app", "list", "--json"], capture_output=True, text=True, timeout=120)
    (out / "app-list.json").write_text(apps.stdout + apps.stderr)
    judge(out, dump, events, signal)
    return 0


class _ReclaimedContainerClient(mei.ModalFunctionClient):
    """Drill-only: a real Modal reclaim ends the signalled container; here
    the container is only signalled, so it stays warm and Modal routes the
    next input back into it. Before every restart, stop the signalled
    container (``modal container stop --yes``) and wait until it is gone, so
    the restart lands in a new container the way it does after a real
    reclaim. Trial 5 (s19-spot-drill-20261009e) restarted into the signalled
    container; the app was then stopped ("user stopped from dashboard",
    cause not found; a CPU repro did not show it)."""

    def __init__(self, function, signal: dict, say, out: Path) -> None:
        super().__init__(function)
        self.signal, self.say, self.out = signal, say, out

    def start(self, config):
        cid = self.signal.get("container")
        if self.calls and cid and not self.signal.get("container_stopped"):
            t = time.time()
            r = subprocess.run([MODAL, "container", "stop", "--yes", cid], capture_output=True, text=True, timeout=120)
            gone = False
            for _ in range(40):
                rows = subprocess.run([MODAL, "container", "list", "--json"], capture_output=True, text=True,
                                      timeout=60).stdout
                if cid not in rows:
                    gone = True
                    break
                time.sleep(3)
            self.signal.update(container_stopped=True, stop_rc=r.returncode, stop_out=(r.stdout + r.stderr)[-300:],
                               container_gone=gone, stop_wait_s=round(time.time() - t, 2))
            (self.out / "signal.json").write_text(json.dumps(self.signal, indent=1))
            self.say("CONTAINER stopped before restart", json.dumps(self.signal))
        return super().start(config)


def judge(out: Path, dump: Path, launcher_events: list, signal: dict) -> None:
    from yeto.rl.eval.drill import attempt_factory, loader_factory
    from yeto.rl.eval.island import EvalIsland, EvalPlan

    root = dump if (dump / "results").is_dir() else next((p for p in dump.rglob("results") if p.is_dir()), dump).parent
    tape = []
    for f in sorted((root / "events").glob("*.jsonl")):
        tape += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    reclaim = [e for e in tape if e.get("event") == "spot_reclaim"]
    rl_eval = [e for e in tape if e.get("event") == "rl_eval"]
    starts = [e for e in launcher_events if e.get("event") == mei.ISLAND_EVENT]
    planned = len(TASKS) * TRIALS_V0
    rows = [json.loads(line) for line in (root / "results" / "v000000.jsonl").read_text().splitlines() if line.strip()] \
        if (root / "results" / "v000000.jsonl").is_file() else []
    keys = [(r["task_id"], r["trial"]) for r in rows if r.get("kind") == "result"]
    # local one-shot reference of the same plan (unit time 0)
    ref = EvalStore(out / "oneshot")
    if not ref.queued_versions():
        ref.put_version(0, files={"policy.safetensors": b"drill", "policy.json": b"{}"},
                        policy_tensor_hash="d" * 64, policy_token="yeto:0:drill", sampling=SAMPLING)
    hold = json.loads((REPO / "data/eval/tb2-holdout.json").read_text())
    plan = EvalPlan.from_holdout(hold, sampling=SAMPLING, task_ids=TASKS, trials_v0=TRIALS_V0, trials=1)
    EvalIsland(ref, plan, loader=loader_factory({}), attempt=attempt_factory({"drill_unit_s": 0}),
               emit=lambda e, **f: None, island_id="eval-spot").run()
    ref_sha = ref.results_sha256(0)
    ev = reclaim[0] if reclaim else {}
    pre = [s for s in starts if s.get("outcome") == "preempted"]
    pre_ts = pre[0]["ts"] if pre else None
    C = {
        "D1_one_reclaim_event_fields": len(reclaim) == 1 and ev.get("cloud") == "modal" and ev.get("role") == "eval"
        and ev.get("source") == "modal_signal" and ev.get("remaining_s") == 25.0 and ev.get("saved") is True
        and ev.get("outcome") == "saved",
        "D2_within_25s": bool(ev) and ev.get("handler_s") is not None and ev["handler_s"] <= 25.0
        and pre_ts is not None and signal.get("t") is not None and pre_ts - signal["t"] <= 60.0,
        "D3_second_container_finished": [s.get("outcome") for s in starts].count("finished") == 1 and len(pre) >= 1,
        "D4_no_dup_no_loss": bool(rl_eval) and rl_eval[-1].get("eval/units") == planned
        and rl_eval[-1].get("eval/duplicate_results") == 0 and len(keys) == planned and len(set(keys)) == planned,
        "D5_same_as_oneshot": bool(rl_eval) and rl_eval[-1].get("eval/results_sha256") == ref_sha,
    }
    facts = {"spot_reclaim": reclaim, "launcher_starts": starts, "signal": signal, "planned_units": planned,
             "result_rows": len(keys), "orphan_starts_metric": (rl_eval[-1].get("eval/preemptions") if rl_eval else None),
             "rl_eval_units": rl_eval[-1].get("eval/units") if rl_eval else None, "oneshot_sha256": ref_sha,
             "results_sha256": rl_eval[-1].get("eval/results_sha256") if rl_eval else None,
             "signal_to_preempted_s": round(pre_ts - signal["t"], 2) if pre_ts and signal.get("t") else None}
    verdict = "通过" if all(C.values()) else "失败"
    (out / "judgment.json").write_text(json.dumps({"verdict": verdict, "criteria": C, "facts": facts},
                                                  indent=1, ensure_ascii=False, default=str))
    print(json.dumps({"verdict": verdict, "criteria": C}, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
