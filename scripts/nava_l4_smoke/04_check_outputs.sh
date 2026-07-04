#!/usr/bin/env bash
# Check smoke-test artifacts after learners finish. Copy learner-1 output here first if needed.

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

python3 - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ.get("RUN_DIR", "/tmp/yeto-nava-l4-smoke"))
num_learners = int(os.environ.get("NUM_LEARNERS", "2"))
total_steps = int(os.environ.get("TOTAL_STEPS", "4"))
check_learner_id = os.environ.get("CHECK_LEARNER_ID")
check_syncer = os.environ.get("CHECK_SYNCER", "1") != "0"
required = [
    run_dir / "train.nava.jsonl",
    run_dir / "filter_report.json",
]
if check_syncer:
    required += [
        run_dir / "yeto-state.ckpt",
        run_dir / "yeto-tape.jsonl",
    ]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit("missing required artifacts:\n" + "\n".join(missing))

rows = sum(1 for _ in (run_dir / "train.nava.jsonl").open("r", encoding="utf-8"))
report = json.loads((run_dir / "filter_report.json").read_text(encoding="utf-8"))

print(f"data rows: {rows}; report output={report.get('output')}")
if check_syncer:
    tape_lines = [json.loads(line) for line in (run_dir / "yeto-tape.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(tape_lines) < total_steps:
        raise SystemExit(f"event tape has {len(tape_lines)} steps, expected at least {total_steps}")
    print(f"event tape steps: {len(tape_lines)}; last step={tape_lines[-1].get('step')}")

learner_ids = [int(check_learner_id)] if check_learner_id is not None else list(range(num_learners))
for learner_id in learner_ids:
    out = run_dir / "output" / f"learner-{learner_id}"
    checks = [out / "layout_manifest.json", out / "train_config.json", out / "learner_state" / "learner_state.pt"]
    adapter_dir = out / "adapter"
    adapter_ok = (adapter_dir / "adapter_config.json").exists() and ((adapter_dir / "adapter.safetensors").exists() or (adapter_dir / "adapter.pt").exists())
    missing = [str(p) for p in checks if not p.exists()]
    if missing or not adapter_ok:
        print(f"learner {learner_id}: incomplete or not copied to this host")
        for item in missing:
            print(f"  missing {item}")
        if not adapter_ok:
            print(f"  missing adapter files under {adapter_dir}")
    else:
        manifest = json.loads((out / "layout_manifest.json").read_text(encoding="utf-8"))
        print(f"learner {learner_id}: ok layout_hash={manifest.get('layout_hash')}")

export_meta = run_dir / "export" / "yeto_export_meta.json"
if export_meta.exists():
    meta = json.loads(export_meta.read_text(encoding="utf-8"))
    print(f"export: ok format={meta.get('format')} global_step={meta.get('global_step')}")
else:
    print("export: not found; run 03_export.sh if export validation is required")
PY
