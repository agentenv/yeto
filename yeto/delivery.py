"""--output delivery: where the fine-tuned model lands after a run.

The head controller fetches ~/yeto-output from the winning learner before
learner teardown (so auto-teardown never destroys the only copy), then:

  * remote destinations — s3:// via the AWS CLI, other object-store URIs
    SkyPilot supports (gs://, r2://, oci://, ...) via its Storage abstraction,
    or hf://org/repo via the Hugging Face Hub — are uploaded from the head,
    after which the head tears itself down with a detached `sky down`: a
    fully self-cleaning run;
  * a local path or no --output keeps the artifact on the head (or, in
    local-controller mode, rsyncs it to the path on this machine) and the
    head stays up so nothing is lost.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def kind(output: str | None) -> str:
    """"none" | "local" | "hf" | "store" (any sky-supported object store)."""
    if not output:
        return "none"
    if output.startswith("hf://"):
        return "hf"
    from .datasource import _is_cloud_url

    return "store" if _is_cloud_url(output) else "local"


def is_remote(output: str | None) -> bool:
    return kind(output) in ("store", "hf")


def fetch_cmd(source_cluster: str, dest_dir: str) -> list[str]:
    """rsync the winning learner's output onto this machine (the head or
    the local worker) via the ssh alias sky wrote when launching it."""
    return ["rsync", "-az", f"{source_cluster}:yeto-output/", f"{dest_dir}/"]


def deliver(output: str, src_dir: str) -> None:
    """Upload src_dir to a remote destination. Raises on any failure —
    callers must not tear anything down when delivery did not happen."""
    k = kind(output)
    src_dir = os.path.expanduser(src_dir)
    if k == "store":
        if output.startswith("s3://"):
            _upload_s3(output, src_dir)
        else:
            _upload_sky(output, src_dir)
    elif k == "hf":
        # sky has no Hub store; huggingface_hub ships with transformers on
        # the head, and HF_TOKEN is forwarded from the submitter.
        from huggingface_hub import HfApi

        repo_id = output.removeprefix("hf://")
        api = HfApi()
        api.create_repo(repo_id, exist_ok=True, private=True)
        api.upload_folder(repo_id=repo_id, folder_path=src_dir)
    else:
        raise ValueError(f"deliver() only handles remote outputs, got {output!r}")


def _upload_s3(output: str, src_dir: str) -> None:
    """Upload output artifacts to S3 through the run store's client
    (yeto.cloud.s3_store): its endpoint (YETO_RL_STORE_S3_ENDPOINT_URL, R2/MinIO),
    store credentials or profile, timeouts and retries. The aws CLI read none of
    these and wrote an R2 store's output to AWS."""
    from yeto.cloud import s3_store

    objects = s3_store.head_objects(output.rstrip("/"))
    root = Path(src_dir)
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        objects.write_file(path.relative_to(root).as_posix(), path)


def _upload_sky(output: str, src_dir: str) -> None:
    """One path for every object store sky supports: resolve the store type
    from the URI scheme via sky's own registry, then Storage(source=dir)
    creates the bucket if needed and syncs the directory (into the URI's
    prefix when one is given)."""
    from sky.data import Storage, StoreType

    store_type = next(
        (st for st in StoreType if output.startswith(st.store_prefix())), None
    )
    if store_type is None:
        raise ValueError(f"no sky store handles {output!r}")
    bucket, _, prefix = output.removeprefix(store_type.store_prefix()).partition("/")
    storage = Storage(
        name=bucket,
        source=src_dir,
        _bucket_sub_path=prefix or None,
    )
    storage.add_store(store_type)
    storage.sync_all_stores()


def self_terminate(head_cluster: str) -> None:
    """Tear the head down with sky's own teardown, detached from this
    process: `sky down` running in-process would die with the instance
    mid-call, but a detached invocation issues the cloud terminate before
    shutdown reaches it. Called only after successful remote delivery.
    The submitter's registry handles the already-gone cluster gracefully
    on a later `yeto down`.
    """
    print(f"[yeto] output delivered; tearing down {head_cluster} (self)", flush=True)
    subprocess.Popen(
        ["sky", "down", "-y", head_cluster],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
