from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path

from .adapters import load_traces
from .core import canonical
from .exporting import as_messages
from .provider import FakeProvider, OpenAICompatibleProvider
from .server import serve
from .store import Store
from .batch import generate_bulk


def main(argv=None):
    parser = argparse.ArgumentParser(description="Review-first synthetic rationale filler; no implicit model calls")
    parser.add_argument("--db", default="cot-review.sqlite3", help="Local SQLite sidecar")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("import")
    ingest.add_argument("input")
    ingest.add_argument("--format", choices=["auto", "session"], default="auto")
    ingest.add_argument("--gap-policy", choices=["markers", "missing-assistant"], default="markers")
    generate = sub.add_parser("generate")
    generate.add_argument("--fake", action="store_true", help="Fixture-only demo, no model/network")
    generate.add_argument("--config", help="Explicit teacher JSON config")
    generate.add_argument("--inference-lock", help="Existing shared lock file used by other inference clients on this host")
    generate.add_argument("--limit", type=int, default=10)
    bulk = sub.add_parser("generate-bulk", help="Explicitly confirmed, resumable bounded corpus generation")
    bulk.add_argument("--config", required=True)
    bulk.add_argument("--inference-lock", required=True)
    bulk.add_argument("--max-gaps", type=int, required=True, help="Finite safety bound; no unbounded run")
    bulk.add_argument("--batch-size", type=int, default=25)
    bulk.add_argument("--checkpoint")
    bulk.add_argument("--confirm-bulk", action="store_true", help="Required acknowledgement of corpus-scale inference")
    external = sub.add_parser("import-candidates", help="Import finished external rationale artifacts; no model calls or approval")
    external.add_argument("input", help="cot.external-candidate.v1 JSON/JSONL file, at most 10 records")
    review = sub.add_parser("serve")
    review.add_argument("--port", type=int, default=8765)
    export = sub.add_parser("export")
    export.add_argument("--arm", choices=["visible", "masked", "none"], default="visible")
    export.add_argument("--output", required=True)
    export.add_argument("--format", choices=["events", "messages"], default="events")
    sub.add_parser("status")
    args = parser.parse_args(argv)
    if args.command == "serve":
        serve(args.db, args.port)
        return
    store = Store(args.db)
    lock = None
    try:
        if args.command == "import":
            count = 0
            for trace, provenance in load_traces(args.input, args.format):
                count += store.import_trace(trace, provenance, args.gap_policy)
            print(canonical({"eligible_gaps": count, "stored_gaps": len(store.list_gaps()), "inference_started": False}))
        elif args.command == "import-candidates":
            path = Path(args.input).resolve()
            if path.stat().st_size > 1024 * 1024:
                raise ValueError("External candidate file exceeds the 1 MiB pilot bound")
            raw = path.read_bytes()
            if path.suffix == ".jsonl":
                records = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
            else:
                value = json.loads(raw)
                records = value if isinstance(value, list) else [value]
            imported = store.import_candidates(records, {"path": str(path), "file_sha256": hashlib.sha256(raw).hexdigest()})
            print(canonical({"imported": len(imported), "candidates": imported, "inference_started": False, "automatic_approval": False}))
        elif args.command in {"generate", "generate-bulk"}:
            if args.command == "generate-bulk":
                if not args.confirm_bulk:
                    raise ValueError("Pass --confirm-bulk to acknowledge corpus-scale inference")
                lock = open(args.inference_lock, "r+")
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ValueError("Inference is reserved by another client; no request sent") from None
                provider = OpenAICompatibleProvider(json.loads(Path(args.config).read_text()))
                result = generate_bulk(store, provider, max_gaps=args.max_gaps, batch_size=args.batch_size, checkpoint=args.checkpoint, confirm=True)
                print(canonical(result))
                if result["failed"]:
                    raise SystemExit(1)
                return
            if not 1 <= args.limit <= 50:
                raise ValueError("Pilot limit must be between 1 and 50; bulk operation is intentionally unavailable")
            if args.fake:
                if args.config:
                    raise ValueError("Choose either --fake or --config")
                provider = FakeProvider()
            else:
                if not args.config or not args.inference_lock:
                    raise ValueError("Real generation needs explicit --config and the existing shared --inference-lock; never use a new local lock for a remote busy fleet")
                lock = open(args.inference_lock, "r+")  # Do not create a dummy lock.
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ValueError("Inference is reserved by another client; no request sent") from None
                provider = OpenAICompatibleProvider(json.loads(Path(args.config).read_text()))
            done = failures = 0
            for row in store.generation_queue()[:args.limit]:
                try:
                    store.verify_source(row["id"])
                    gap = store.detail(row["id"])["gap"]
                    text, metadata = provider.generate(gap)
                    store.add_candidate(row["id"], text, metadata, row["revision"])
                    done += 1
                except (ValueError, OSError) as exc:
                    store.record_failure(row["id"], type(exc).__name__ + ": " + str(exc))
                    failures += 1
            print(canonical({"generated": done, "failed": failures, "demo_only": args.fake}))
            if failures:
                raise SystemExit(1)
        elif args.command == "status":
            counts = {}
            for gap in store.list_gaps():
                counts[gap["status"]] = counts.get(gap["status"], 0) + 1
            print(canonical({"counts": counts, "queued_for_generation": len(store.generation_queue())}))
        elif args.command == "export":
            output = Path(args.output).resolve()
            source_paths = {json.loads(r[0]).get("path") for r in store.db.execute("SELECT provenance FROM traces")}
            if str(output) in source_paths or output == Path(args.db).resolve():
                raise ValueError("Export cannot overwrite original input or sidecar")
            # Exclusive create protects previous exports and source files.
            rows = list(store.export(args.arm))
            if args.format == "messages":
                rows = [as_messages(row) for row in rows]
            with output.open("x") as stream:
                for row in rows:
                    stream.write(canonical(row) + "\n")
            print(canonical({"exported_traces": len(rows), "training_ready": False, "output": str(output)}))
    except (KeyError, ValueError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    finally:
        if lock:
            lock.close()
        store.close()


if __name__ == "__main__":
    main()
