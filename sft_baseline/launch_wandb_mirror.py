"""Start a fresh local numeric W&B mirror from an actual training launch receipt."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from sft_baseline.mirror_wandb import source_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--source-run-id", default="")
    args = parser.parse_args()
    raw = args.launch_receipt.read_bytes()
    launch = json.loads(raw)
    source = source_identity(launch["container"], launch["root"] + "/checkpoints", args.source_run_id)
    folder = Path(__file__).resolve().parent
    tag = Path(launch["root"]).name
    state = folder / ("wandb-mirror-" + tag + "-state.json")
    log = folder / ("wandb-mirror-" + tag + ".log")
    receipt = folder / ("wandb-mirror-" + tag + "-launch.json")
    if any(path.exists() for path in (state, log, receipt)):
        raise SystemExit("Mirror artifacts already exist; inspect instead of launching a duplicate")
    command = [sys.executable, "-u", "-m", "sft_baseline.mirror_wandb",
               "--state", str(state), "--source-container", source["container"],
               "--source-directory", source["directory"], "--source-run-id", source["run_id"],
               "--expected-user", "walden-lee"]
    with log.open("xb") as out:
        process = subprocess.Popen(command, cwd=folder.parent, stdin=subprocess.DEVNULL,
                                   stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    result = {"pid": process.pid, "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "source": source, "state_path": str(state), "log_path": str(log),
              "launch_receipt_sha256": hashlib.sha256(raw).hexdigest(),
              "implementation_sha256": hashlib.sha256((folder / "mirror_wandb.py").read_bytes()).hexdigest(),
              "credentials": "existing verified local walden-lee W&B login",
              "entity": "yeta", "project": "yeto-h200"}
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
