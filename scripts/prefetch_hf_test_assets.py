"""Download the tokenizer/config files the unit tests declare in tests/hf_assets.py.

Usage:
    python scripts/prefetch_hf_test_assets.py            # fetch into the HF cache
    python scripts/prefetch_hf_test_assets.py --cache-key  # print a key of the repo list

Gated repos are fetched only when HF_TOKEN is set. Weights are never fetched.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

ASSETS = Path(__file__).resolve().parent.parent / "tests" / "hf_assets.py"


def load_assets():
    spec = importlib.util.spec_from_file_location("hf_assets", ASSETS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cache_key(assets) -> str:
    payload = json.dumps({"repos": assets.HF_TEST_REPOS, "patterns": assets.HF_ALLOW_PATTERNS}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-key", action="store_true", help="print the cache key and exit")
    args = parser.parse_args(argv)
    assets = load_assets()
    if args.cache_key:
        print(cache_key(assets))
        return 0

    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or None
    for repo, revision in sorted(assets.HF_TEST_REPOS.items()):
        if repo in assets.HF_GATED_REPOS and token is None:
            print(f"skip {repo}: gated and HF_TOKEN is not set")
            continue
        path = snapshot_download(repo, revision=revision, allow_patterns=assets.HF_ALLOW_PATTERNS, token=token)
        size = sum(f.stat().st_size for f in Path(path).iterdir() if f.is_file())
        print(f"{repo}@{revision[:8]}: {size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
