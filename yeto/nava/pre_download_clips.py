"""Pre-fetch selected clips onto a learner's local disk before training.

Writes each clip to the exact path the training resolver will look for —
``stable_cache_path(uri, cache_dir)`` (see ``yeto.nava.utils``) — so once this
finishes, training reads every clip from local disk (cache hit) and never streams
from S3. Run during learner setup, before torchrun:

    python -m yeto.nava.pre_download_clips \
        --uri-list s3://bucket/manifests/selected_clip_uris.txt \
        --cache-dir /local_nvme/yeto-nava-cache --workers 48

``--uri-list`` may itself be an s3:// object or a local file (one clip URI per
line). Idempotent: already-cached clips are skipped.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from .utils import download_s3, is_s3_uri, materialize_uri, stable_cache_path


def _read_uri_list(uri_list: str) -> list[str]:
    local = materialize_uri(uri_list, tempfile.gettempdir()) if is_s3_uri(uri_list) else uri_list
    with open(local, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Pre-download NAVA clips into the resolver cache")
    ap.add_argument("--uri-list", required=True, help="s3:// or local newline list of clip URIs")
    ap.add_argument("--cache-dir", required=True, help="must match --nava-data-cache")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args(argv)

    os.makedirs(args.cache_dir, exist_ok=True)
    uris = _read_uri_list(args.uri_list)

    def fetch(uri: str) -> str:
        dest = stable_cache_path(uri, args.cache_dir)
        if dest.exists() and dest.stat().st_size > 0:
            return "cached"
        download_s3(uri, dest)  # atomic (.tmp + rename), same helper the resolver uses
        return "downloaded"

    ok = cached = errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch, u): u for u in uris}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                cached += fut.result() == "cached"
                ok += 1
            except Exception as e:  # noqa: BLE001 - report and continue
                errors += 1
                print(f"ERR {futs[fut]}: {e}", file=sys.stderr)
            if i % 500 == 0 or i == len(uris):
                print(f"  pre-download {i}/{len(uris)} (cached={cached} err={errors})", flush=True)

    print(f"pre-download complete: {ok} ok ({cached} already cached), {errors} errors", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
