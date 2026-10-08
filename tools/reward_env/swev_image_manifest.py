#!/usr/bin/env python3
"""Pin SWE-bench Verified instance images by digest and record their sizes.

Reads the pinned dataset (``SWE-bench/SWE-bench_Verified`` parquet) and asks the
Docker Hub registry API (metadata only, no image pull, no Modal) for each
instance image's ``latest`` tag: index digest, amd64 compressed size, last
update.  Output JSON maps instance_id -> ``docker.io/<repo>@<digest>`` for
``SwebenchVerified(images=…)`` plus a size summary for the storage estimate.

    python tools/reward_env/swev_image_manifest.py --data swev-78f471bf.parquet --out swev-images.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yeto.rl.harness.reward_env.swebench_verified import docker_hub_repo, load_rows  # noqa: E402

API = "https://hub.docker.com/v2/repositories/{repo}/tags/{tag}"


def fetch(repo: str, tag: str, retries: int = 4) -> dict:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(API.format(repo=repo, tag=tag), timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt + 1 < retries:
                time.sleep(30 * (attempt + 1))
                continue
            return {"error": f"HTTP {exc.code}"}
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt + 1 < retries:
                time.sleep(5)
                continue
            return {"error": str(exc)}
    return {"error": "retries exhausted"}


def entry(row: dict, info: dict) -> dict:
    repo, tag = docker_hub_repo(row["image"])
    if "error" in info or not info.get("digest"):
        return {"image": row["image"], "error": info.get("error", "no digest")}
    amd64 = [i for i in info.get("images", []) if i.get("architecture") == "amd64"]
    return {"image": row["image"], "pinned": f"docker.io/{repo}@{info['digest']}",
            "amd64_compressed_bytes": amd64[0].get("size") if amd64 else None,
            "last_updated": info.get("last_updated")}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--pause-s", type=float, default=0.2)
    a = p.parse_args(argv)
    rows = load_rows(a.data)[: a.limit]
    out: dict = {"source": "hub.docker.com v2 tags API", "queried_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "images": {}}
    for row in rows:
        out["images"][row["instance_id"]] = entry(row, fetch(*docker_hub_repo(row["image"])))
        time.sleep(a.pause_s)
    sizes = [v["amd64_compressed_bytes"] for v in out["images"].values() if v.get("amd64_compressed_bytes")]
    out["summary"] = {"instances": len(rows), "pinned": sum("pinned" in v for v in out["images"].values()),
                      "errors": sum("error" in v for v in out["images"].values()),
                      "compressed_bytes_sum": sum(sizes),
                      "compressed_bytes_min": min(sizes) if sizes else None,
                      "compressed_bytes_max": max(sizes) if sizes else None}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"]))
    return 0 if out["summary"]["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
