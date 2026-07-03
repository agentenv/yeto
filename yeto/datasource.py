"""--data source resolution: HF dataset id, local path, or cloud object store.

Non-HF sources ride SkyPilot's file_mounts abstraction so learners only ever
read a plain local path and data.py needs no cloud SDKs:

  * cloud URIs (s3://, gs://, r2://, cos://, oci://) are fetched by sky onto
    every learner node at provision time;
  * local paths are rsynced by sky — one hop in local-controller mode, two
    hops in head mode (submitter -> head at HEAD_DATA_PATH, then the head's
    copy -> learners), mirroring how the pickled loss ships.

HF dataset ids pass through untouched and are streamed by the learners.
"""

from __future__ import annotations

import os

# Fallback scheme list for environments without sky (learner nodes import
# nothing from here, but belt-and-braces); with sky importable, detection
# delegates to sky's own registry and covers every store it supports.
CLOUD_SCHEMES = ("s3://", "gs://", "r2://", "cos://", "oci://", "azure://", "nebius://")
LEARNER_DATA_PATH = "~/yeto-data"
HEAD_DATA_PATH = "~/yeto-data-src"


def _is_cloud_url(data: str) -> bool:
    try:
        from sky.data import data_utils

        return bool(data_utils.is_cloud_store_url(data))
    except Exception:
        return data.startswith(CLOUD_SCHEMES)


def kind(data: str) -> str:
    """"hf" | "cloud" | "local". Cloud detection is sky's own predicate
    (any object store SkyPilot can mount). Anything path-shaped (or that
    exists on this machine) is local; HF ids never start with /, ./, ../
    or ~."""
    if _is_cloud_url(data):
        return "cloud"
    if data.startswith(("/", "./", "../", "~")) or os.path.exists(os.path.expanduser(data)):
        return "local"
    return "hf"


def learner_data_arg(data: str) -> str:
    """What the learner's --data should be once mounts are in place."""
    return data if kind(data) == "hf" else LEARNER_DATA_PATH


def learner_file_mounts(data: str) -> dict[str, str]:
    """file_mounts entries for a learner task (empty for HF ids)."""
    k = kind(data)
    if k == "hf":
        return {}
    if k == "cloud":
        return {LEARNER_DATA_PATH: data}
    return {LEARNER_DATA_PATH: os.path.expanduser(data)}


def head_stage(data: str) -> tuple[str, dict[str, str]]:
    """(rewritten --data for the head job, extra head-VM file_mounts).

    Only local paths need staging: the submitter's copy lands on the head at
    HEAD_DATA_PATH, and the rewritten --data makes the head's launcher treat
    it as a local source again for the second hop. Cloud URIs and HF ids are
    reachable from the head directly.
    """
    if kind(data) == "local":
        return HEAD_DATA_PATH, {HEAD_DATA_PATH: os.path.expanduser(data)}
    return data, {}
