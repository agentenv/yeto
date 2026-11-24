"""Model preparation on a node, by the profile's model source.

* ``store``: a capped, resumable parallel ``aws s3 cp`` of ``s3://$R2_BUCKET/<key>``
  (:func:`pull_body`, the body the National Compute and Daytona routes run: one
  classic-client process tops out near 275 MB/s on R2, so ``jobs`` processes
  share the object list, each capped at ``max_mb_s / jobs``; an object already
  at ``dest`` with the listed size is skipped). A local directory store copies
  the files instead (one host, tests);
* ``volume`` / ``preset``: the model is already at ``dest`` (a mounted volume, a
  baked image): checked to be a non-empty directory;
* ``hf``: ``huggingface_hub.snapshot_download`` of ``repo`` at ``revision``;
* ``none``: nothing to do (an engine that fetches its own model).
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .records import ModelPlan

PULL_JOBS = 16  # aws-cli processes; one classic-client process tops out near 275 MB/s on R2
PULL_CAP_MB_S = 3000  # ~7 GB/s dropped an MI355X node off the network for good (GPU-B section 4.1)


class ModelError(RuntimeError):
    pass


def pull_body(src_key: str, dest: str, *, max_mb_s: int = 2500, include: str | None = None,
              jobs: int = PULL_JOBS) -> str:
    """Shell of a capped, resumable parallel ``aws s3 cp`` of ``s3://$R2_BUCKET/<src_key>``
    into ``dest``: a path where the shell runs (``/models/...`` in the aws-cli container,
    or the sandbox path of a no-docker sampler). Credentials come from the environment."""
    if max_mb_s > PULL_CAP_MB_S:
        raise ValueError(f"pull cap {max_mb_s} MB/s exceeds the node limit 3000 MB/s")
    jobs = max(1, int(jobs))
    src = src_key.strip("/")
    one = ('rel=${1#' + src + '/}; case "$rel" in $YETO_PULL_INCLUDE) ;; *) exit 0;; esac; '
           f'f={shlex.quote(dest)}/$rel; '
           '[ -f "$f" ] && [ "$(wc -c < "$f" | tr -d " ")" = "$2" ] && exit 0; mkdir -p "$(dirname "$f")"; '
           'aws s3 cp --only-show-errors "s3://$R2_BUCKET/$1" "$f" --endpoint-url "$AWS_ENDPOINT_URL" '
           '|| { echo "YETO_NC_PULL fail $1"; exit 255; }')
    return (f"export YETO_PULL_INCLUDE={shlex.quote(include or '*')} && "
            "aws configure set default.s3.max_concurrent_requests 16 && "
            f"aws configure set default.s3.max_bandwidth {max(1, int(max_mb_s) // jobs)}MB/s && "
            f'echo "YETO_NC_PULL start $(date -u +%FT%TZ) jobs={jobs}" && '
            # the listing is read before any copy: a refused or empty listing is a failed
            # pull, not a "done" with nothing copied (a pipeline exits with xargs' status)
            f'{{ L=$(aws s3 ls --recursive "s3://$R2_BUCKET/{src}/" --endpoint-url "$AWS_ENDPOINT_URL") '
            f'|| {{ echo "YETO_NC_PULL fail listing s3://$R2_BUCKET/{src}/"; exit 3; }}; }} && '
            f'{{ [ -n "$L" ] || {{ echo "YETO_NC_PULL fail empty listing s3://$R2_BUCKET/{src}/"; exit 3; }}; }} && '
            'printf "%s\\n" "$L" | awk \'{print $4, $3}\' | sort -k2 -n -r '
            f"| xargs -r -P {jobs} -L1 sh -c {shlex.quote(one)} pull "
            '&& echo "YETO_NC_PULL done $(date -u +%FT%TZ)"')


def model_dest(plan: ModelPlan, models_dir: str) -> str:
    """The node path of the model: ``dest`` (``/models/...`` read as ``models_dir``),
    else ``<models_dir>/<last part of key or repo>``."""
    if plan.dest:
        dest = str(plan.dest)
        if dest == "/models" or dest.startswith("/models/"):
            dest = models_dir.rstrip("/") + dest[len("/models"):]
        return dest
    name = (plan.key or plan.repo or "model").rstrip("/").rsplit("/", 1)[-1]
    return f"{models_dir.rstrip('/')}/{name}"


def _local_root(objects: Any) -> Path | None:
    root = getattr(objects, "root", None)
    return None if root is None or "://" in str(root) else Path(root)


def prepare_model(plan: ModelPlan, *, models_dir: str, objects: Any = None,
                  environ: Mapping[str, str] | None = None, run: Callable[..., Any] = subprocess.run,
                  log: Callable[[str], None] = print) -> dict[str, Any]:
    """Make the model of ``plan`` present on the node; a report of what was done."""
    source = plan.source or "none"
    if source == "none":
        return {"source": "none"}
    dest = model_dest(plan, models_dir)
    if source in ("volume", "preset"):
        path = Path(dest)
        if not path.is_dir() or not any(path.iterdir()):
            raise ModelError(f"model source {source}: {dest} is not a non-empty directory on this node")
        return {"source": source, "dest": dest}
    if source == "store":
        if not plan.key:
            raise ModelError("model source store needs the model's key in the store")
        local = _local_root(objects)
        if local is not None:  # a store directory on this host: copy what is missing
            src = local / plan.key.strip("/")
            if not src.is_dir():
                raise ModelError(f"model source store: {plan.key} is not in the store")
            copied = 0
            for path in sorted(src.rglob("*")):
                if not path.is_file():
                    continue
                target = Path(dest) / path.relative_to(src)
                if target.is_file() and target.stat().st_size == path.stat().st_size:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                copied += 1
            return {"source": "store", "dest": dest, "copied": copied}
        Path(dest).mkdir(parents=True, exist_ok=True)
        body = pull_body(plan.key, dest, max_mb_s=int(plan.max_mb_s), include=plan.include, jobs=int(plan.jobs))
        out = run(["bash", "-c", body], capture_output=True, text=True, timeout=float(plan.timeout_s), check=False,
                  env=dict(os.environ if environ is None else environ))
        text = (out.stdout or "") + (out.stderr or "")
        if out.returncode != 0 or "YETO_NC_PULL done" not in text:
            raise ModelError(f"model pull of {plan.key} failed (exit {out.returncode}): {text.strip()[-400:]}")
        return {"source": "store", "dest": dest, "pulled": True}
    if source == "hf":
        if not plan.repo:
            raise ModelError("model source hf needs repo")
        script = ("import sys; from huggingface_hub import snapshot_download; "
                  "snapshot_download(sys.argv[1], revision=sys.argv[2] or None, local_dir=sys.argv[3])")
        out = run([os.environ.get("YETO_NODE_PYTHON", "python3"), "-c", script, plan.repo, plan.revision or "", dest],
                  capture_output=True, text=True, timeout=float(plan.timeout_s), check=False,
                  env=dict(os.environ if environ is None else environ))
        if out.returncode != 0:
            raise ModelError(f"hf download of {plan.repo} failed: {(out.stderr or '').strip()[-400:]}")
        return {"source": "hf", "dest": dest, "repo": plan.repo}
    raise ModelError(f"model source {source!r} is unknown")
