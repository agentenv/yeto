#!/usr/bin/env python3
"""TB2 task lists on the Harbor path (S26 E9.5; module ``yeto.rl.harness.harbor.task_lists``).

    # hold-out + training split (data/eval/tb2-holdout.json, tb2-train.json shapes)
    python tools/reward_env/tb2_lists.py split --tasks ~/work/tb21-data --out-dir out/

    # curriculum stages of a training split
    python tools/reward_env/tb2_lists.py curriculum --tasks ~/work/tb21-data \
        --train out/tb2-train.json --out out/tb2-curriculum.json \
        [--reference pos-all.jsonl] [--pass-rates baseline.json --require-signal]

    # dataset rows (Miles jsonl + verl jsonl) of a training list or curriculum stages
    python tools/reward_env/tb2_lists.py rows --tasks ~/work/tb21-data \
        --list out/tb2-curriculum.json [--through s1-short] --out-prefix out/tb2.1-s1

    # sha256 manifest of written files, with their sources
    python tools/reward_env/tb2_lists.py manifest --out out/manifest.json --source k=v ... FILE...

``--tasks`` is the Terminal-Bench checkout (TB2 flat, or TB2.1 with ``tasks/``).
The output is deterministic: same checkout and inputs, same bytes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yeto.rl.harness.harbor import task_lists as tl  # noqa: E402
from yeto.rl.harness.reward_env import benchmark as bm  # noqa: E402


def _split(a: argparse.Namespace) -> int:
    tasks = tl.HarborTaskSet(a.tasks, version=a.version)
    unusable = {} if a.no_unusable else tl.load_unusable(a.unusable)
    holdout, train = tl.build_tb2_lists(tasks, unusable=unusable, exclude={} if a.no_smoke6 else None)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, doc in (("tb2-holdout.json", holdout), ("tb2-train.json", train)):
        (out / name).write_text(tl.dump_list(doc))
        print(name, len(doc["items"]), bm.holdout_sha256(doc))
    return 0


def _curriculum(a: argparse.Namespace) -> int:
    tasks = tl.HarborTaskSet(a.tasks, version=a.version)
    train = json.loads(Path(a.train).read_text())
    source = {"train": Path(a.train).name}
    reference = pass_rates = None
    if a.reference:
        reference = tl.load_reference_results(a.reference)
        source["reference"] = Path(a.reference).name
    if a.pass_rates:
        pass_rates = tl.load_pass_rates(a.pass_rates)
        source["pass_rates"] = Path(a.pass_rates).name
    doc = tl.build_curriculum(tasks, train, reference=reference, pass_rates=pass_rates,
                              require_signal=a.require_signal, source=source)
    Path(a.out).write_text(tl.dump_list(doc))
    for stage in doc["stages"]:
        print(stage["name"], len(stage["items"]))
    print("dropped", len(doc["dropped"]))
    return 0


def _rows(a: argparse.Namespace) -> int:
    tasks = tl.HarborTaskSet(a.tasks, version=a.version)
    doc = json.loads(Path(a.list).read_text())
    if doc.get("schema") == tl.CURRICULUM_SCHEMA:
        ids = tl.curriculum_task_ids(doc, through=a.through)
    elif doc.get("schema") == bm.TRAIN_SPLIT_SCHEMA:
        if a.through:
            raise SystemExit("--through needs a curriculum file")
        ids = [item["task_id"] for item in doc["items"]]
    else:
        raise SystemExit(f"{a.list}: neither a training list nor a curriculum")
    if a.holdout:
        tl.assert_no_holdout(ids, json.loads(Path(a.holdout).read_text()))
    for backend, rows in (("miles", tl.miles_rows(tasks, ids, agent=a.agent)),
                          ("verl", tl.verl_rows(tasks, ids, agent=a.agent))):
        path = Path(f"{a.out_prefix}-{backend}.jsonl")
        path.write_text(tl.dump_jsonl(rows))
        print(path.name, len(rows))
    return 0


def _manifest(a: argparse.Namespace) -> int:
    import hashlib

    sources = dict(item.split("=", 1) for item in a.source)
    files = {}
    for name in a.files:
        body = Path(name).read_bytes()
        files[Path(name).name] = {"sha256": hashlib.sha256(body).hexdigest(),
                                  "rows": body.count(b"\n") if name.endswith(".jsonl") else None}
    Path(a.out).write_text(json.dumps({"schema": "yeto-data-manifest/1", "sources": sources, "files": files},
                                      indent=1, sort_keys=True) + "\n")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("split", help="hold-out and training lists")
    s.add_argument("--tasks", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--version", help="benchmark_version override (default: from the checkout)")
    s.add_argument("--unusable", help="unusable-task table (default data/eval/tb2-unusable.json)")
    s.add_argument("--no-unusable", action="store_true")
    s.add_argument("--no-smoke6", action="store_true", help="keep the S15 smoke-6 in the hold-out pool")
    s.set_defaults(run=_split)
    c = sub.add_parser("curriculum", help="ordered stages of a training list")
    c.add_argument("--tasks", required=True)
    c.add_argument("--train", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--version")
    c.add_argument("--reference", help="reference-solution runs (jsonl: task, passed, solve_s)")
    c.add_argument("--pass-rates", help="baseline pass counts (json: {task_id: {n, passed}})")
    c.add_argument("--require-signal", action="store_true", help="drop tasks with pass rate 0 or 1")
    c.set_defaults(run=_curriculum)
    r = sub.add_parser("rows", help="Miles and verl dataset rows of a list")
    r.add_argument("--tasks", required=True)
    r.add_argument("--list", required=True, help="training list or curriculum")
    r.add_argument("--through", help="last curriculum stage to include")
    r.add_argument("--holdout", help="hold-out list the rows must not touch")
    r.add_argument("--agent", default="codex")
    r.add_argument("--out-prefix", required=True)
    r.add_argument("--version")
    r.set_defaults(run=_rows)
    m = sub.add_parser("manifest", help="sha256 manifest of files")
    m.add_argument("--out", required=True)
    m.add_argument("--source", action="append", default=[], help="key=value provenance line")
    m.add_argument("files", nargs="+")
    m.set_defaults(run=_manifest)
    a = p.parse_args(argv)
    return a.run(a)


if __name__ == "__main__":
    raise SystemExit(main())
