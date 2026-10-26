"""S27 P2: the launcher keeps the task dispenser's log with the run.

With ``--rl-data-split dispense`` the actor syncer appends every dispenser
event to ``~/yeto-output/task-dispense.jsonl`` (``--dispense-log``) on the
machine it runs on. At the end of the run (``launcher.run``, also after a
failure):

* the launcher that runs next to the syncer (the head, or this machine with
  ``--controller local``) copies the log into the run directory and checks
  it offline (``tools.eval.dispense_check``): ``<run dir>/task-dispense.jsonl``,
  ``<run dir>/dispense-check.json`` and ``~/yeto-output/dispense-check.json``,
  plus one ``[launcher] task dispenser:`` line in the launch log;
* the submitting machine of a head run (``yeto launch --controller head``)
  copies both files from the head into its own run directory once the head
  job ended (:func:`pull_from_head`).

A syncer on a cluster of its own keeps its log in that cluster's teardown
diagnostics (``~/yeto-output/*.jsonl``). Every step is best effort: a
missing or unreadable log is reported, never fatal to the teardown.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

DISPENSE_LOG = "~/yeto-output/task-dispense.jsonl"
DISPENSE_CHECK = "~/yeto-output/dispense-check.json"
RUN_DIR_LOG = "task-dispense.jsonl"
RUN_DIR_CHECK = "dispense-check.json"
# a local log file collects every local run (append): records older than the
# run's start minus this slack belong to earlier runs
SINCE_SLACK_S = 600.0
PULL_TIMEOUT_S = 120.0


def dispense_planned(args) -> bool:
    try:
        from .data_split_plan import data_split_plan

        # a sampler launch of its own (P4) claims from another launch's dispenser
        return bool(data_split_plan(args).dispense) and not getattr(args, "rl_dispense_addr", None)
    except Exception:  # noqa: BLE001 - not an RL launch, or no plan: nothing to keep
        return False


def _records_since(lines: list[str], since_unix: float | None) -> list[str]:
    if since_unix is None:
        return lines
    out = []
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if float(record.get("time_unix") or 0.0) >= since_unix - SINCE_SLACK_S:
            out.append(line)
    return out


def check_summary(log: Path) -> dict[str, Any]:
    """``tools.eval.dispense_check`` of ``log`` (without the per-position lists)."""
    from tools.eval.dispense_check import check, read_records

    summary = check(read_records([log]))
    summary.pop("trained_positions_by_island", None)
    return summary


def collect_dispense_log(args, run_dir: str | os.PathLike[str], *,
                         since_unix: float | None = None,
                         source: str = DISPENSE_LOG) -> dict[str, Any] | None:
    """Copy and check the dispenser log of this run (module doc); None when the
    run does not use the dispenser. Never raises."""
    if not dispense_planned(args):
        return None
    try:
        src = Path(os.path.expanduser(source))
        if not src.is_file():
            print(f"[launcher] task dispenser: no log at {source} on this machine (a syncer "
                  "cluster of its own keeps it in its teardown diagnostics)", file=sys.stderr)
            return {"copied": False}
        lines = [line for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
        lines = _records_since(lines, since_unix)
        dest = Path(run_dir)
        dest.mkdir(parents=True, exist_ok=True)
        log = dest / RUN_DIR_LOG
        log.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
        out: dict[str, Any] = {"copied": True, "records": len(lines), "log": str(log)}
        try:
            summary = check_summary(log)
        except Exception as error:  # noqa: BLE001 - the copy is what matters
            out["check_error"] = f"{type(error).__name__}: {error}"
            print(f"[launcher] task dispenser: log kept at {log}; check failed: "
                  f"{out['check_error']}", file=sys.stderr)
            return out
        text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
        (dest / RUN_DIR_CHECK).write_text(text, encoding="utf-8")
        check_copy = Path(os.path.expanduser(DISPENSE_CHECK))
        if check_copy.parent.is_dir():
            check_copy.write_text(text, encoding="utf-8")
        islands = {k: {"granted": v.get("granted"), "trained": v.get("trained"),
                       "incarnations": v.get("incarnations")}
                   for k, v in (summary.get("islands") or {}).items()}
        print(f"[launcher] task dispenser: ok={summary.get('ok')} problems="
              f"{summary.get('problem_count')} given_out={summary.get('given_out')} "
              f"trained={summary.get('trained')} dropped={summary.get('dropped')} "
              f"queued_at_end={summary.get('queued_at_end')} held_at_end="
              f"{summary.get('held_at_end')} islands={json.dumps(islands, sort_keys=True)} "
              f"(log {log})", flush=True)
        out.update(ok=summary.get("ok"), check=str(dest / RUN_DIR_CHECK))
        return out
    except Exception as error:  # noqa: BLE001 - never fatal to the teardown
        print(f"[launcher] task dispenser: keeping the log failed: {error!r}", file=sys.stderr)
        return {"copied": False, "error": repr(error)}


def pull_from_head(args, head_cluster: str, run_dir: str | os.PathLike[str], *,
                   run=subprocess.run, timeout_s: float = PULL_TIMEOUT_S) -> dict[str, str] | None:
    """The submitting machine of a head run: copy the head's dispenser log and
    check into ``run_dir`` (ssh, bounded). None without the dispenser. Never raises."""
    if not dispense_planned(args):
        return None
    out: dict[str, str] = {}
    try:
        Path(run_dir).mkdir(parents=True, exist_ok=True)
    except OSError as error:
        return {"dir": f"failed: {error}"}
    for remote, name in ((DISPENSE_LOG, RUN_DIR_LOG), (DISPENSE_CHECK, RUN_DIR_CHECK)):
        argv = ["rsync", "-a", "-e", "ssh -o ConnectTimeout=20 -o BatchMode=yes",
                f"{head_cluster}:{remote}", str(Path(run_dir) / name)]
        try:
            proc = run(argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                       timeout=timeout_s, check=False)
            code = getattr(proc, "returncode", 0)
            out[name] = "ok" if code == 0 else f"failed: exit {code}"
        except subprocess.TimeoutExpired:
            out[name] = "timeout"
        except Exception as error:  # noqa: BLE001
            out[name] = f"failed: {error}"
    print(f"[yeto] task dispenser log from {head_cluster} -> {run_dir}: {out}",
          file=sys.stderr if any(v != "ok" for v in out.values()) else sys.stdout, flush=True)
    return out
