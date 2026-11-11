"""Preflight checks of a model checkpoint: it exists, it is readable and its
weight files are complete. Runs before any order (``yeto launch --dry-run``, the
trainer side of ``yeto run up``, and ``python -m yeto.launch.nc pool --dry-run``
for the samplers' bucket copy).

Each source is LISTed once; only ``config.json`` presence and the small
``*.index.json`` are looked at, no weight file is downloaded. With an index, every
file its ``weight_map`` names must be in the listing; without one, a single-file
``model.safetensors`` (or ``pytorch_model.bin``) must be.

Sources: a local directory, an object-store URI (``s3://`` / ``r2://``), a Modal
volume snapshot (``--modal-model-volume``) and a Hugging Face Hub repository.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable

GUIDE = "docs/KIMI_K31_ASYNC_RL_GUIDE.md, Model weights"
CONFIG = "config.json"
INDEXES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
SINGLE_FILES = ("model.safetensors", "pytorch_model.bin")
SHOW_MISSING = 5
MODES = ("check", "warn", "off")
MODE_ENV = "YETO_PREFLIGHT_MODEL"
MODEL_VOLUME_MOUNT_DIR = "hf"  # the trainer serves hf/<name>/<revision[:8]>/ of the volume


class ModelCheckError(ValueError):
    """The trainer's model is missing, unreadable or incomplete."""


def weight_problems(names: Iterable[str], read: Callable[[str], bytes | None], where: str) -> list[str]:
    """Problems of one checkpoint directory from its file listing ``names`` (paths
    relative to the directory) and ``read`` (a small file's bytes, None if absent)."""
    have = set(names)
    problems = []
    if CONFIG not in have:
        problems.append(f"no {where}/{CONFIG}")
    index = next((i for i in INDEXES if i in have), None)
    if index is None:
        if not any(name in have for name in SINGLE_FILES):
            problems.append(f"no weights in {where}: neither {INDEXES[0]} nor {SINGLE_FILES[0]}")
        return problems
    try:
        doc = json.loads(read(index) or b"")
        shards = sorted(set((doc.get("weight_map") or {}).values()))
    except (ValueError, AttributeError) as error:
        return problems + [f"{where}/{index} is not a valid index: {error}"[:300]]
    if not shards:
        return problems + [f"{where}/{index} has an empty weight_map"]
    missing = [s for s in shards if s not in have]
    if missing:
        more = f" (+{len(missing) - SHOW_MISSING} more)" if len(missing) > SHOW_MISSING else ""
        problems.append(f"{where}: {len(missing)} of {len(shards)} weight shards named by {index} are missing: "
                        + ", ".join(missing[:SHOW_MISSING]) + more)
    return problems


def local_problems(path: str | os.PathLike[str]) -> list[str]:
    root = Path(path).expanduser()
    if not root.is_dir():
        return [f"{root} is not a directory"]
    names = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]

    def read(name: str) -> bytes | None:
        try:
            return (root / name).read_bytes()
        except OSError:
            return None

    return weight_problems(names, read, str(root))


def store_problems(objects: Any, where: str) -> list[str]:
    """``objects``: an object API over the model prefix (``list`` pages through the
    whole prefix, ``read``). One LIST; the index is the only object read."""
    return weight_problems(objects.list(""), objects.read, where)


def store_uri_problems(uri: str, environ: dict[str, str] | None = None) -> list[str]:
    from yeto.cloud import s3_store  # noqa: PLC0415

    return store_problems(s3_store.head_objects(uri.rstrip("/"), environ=environ), uri.rstrip("/"))


def hub_files(repo: str, revision: str | None) -> tuple[list[str], Callable[[str], bytes | None]]:
    """The repository's file list (one API call) and a reader of its small files."""
    from huggingface_hub import HfApi, hf_hub_download  # noqa: PLC0415

    info = HfApi().model_info(repo, revision=revision)
    names = [s.rfilename for s in (info.siblings or ())]

    def read(name: str) -> bytes | None:
        return Path(hf_hub_download(repo, name, revision=revision)).read_bytes()

    return names, read


def modal_volume_files(volume: str, path: str) -> tuple[list[str], Callable[[str], bytes | None]] | None:
    """The files under ``path`` of Modal volume ``volume`` (None: no such directory)."""
    import modal  # noqa: PLC0415

    vol = modal.Volume.from_name(volume)
    try:
        entries = vol.listdir(path, recursive=True)
    except (FileNotFoundError, getattr(modal.exception, "NotFoundError", FileNotFoundError)):
        return None
    prefix = path.strip("/") + "/"
    names = [str(e.path).lstrip("/").removeprefix(prefix) for e in entries
             if getattr(e, "type", None) is None or "DIRECTORY" not in str(e.type)]

    def read(name: str) -> bytes | None:
        return b"".join(vol.read_file(f"{path}/{name}"))

    return names, read


def _is_uri(value: str) -> bool:
    return "://" in value


def trainer_model_check(args, *, hub=None, volume=None, store=None) -> dict[str, Any]:
    """What the trainer of ``args`` will read, checked from the control host.

    Returns ``{"model", "source", "problems", "warnings"}``; the caller refuses the
    launch on problems (``--preflight-model check``, the default)."""
    hub, volume, store = hub or hub_files, volume or modal_volume_files, store or store_uri_problems
    model = str(getattr(args, "model", "") or "")
    revision = getattr(args, "model_revision", None)
    record: dict[str, Any] = {"model": model, "source": None, "problems": [], "warnings": []}
    if not model:
        return record
    try:
        from yeto.models import resolve  # noqa: PLC0415

        model = str(resolve(model))
    except Exception:  # noqa: BLE001 - an alias table miss leaves the name as given
        pass
    from yeto.provenance import is_local_reference  # noqa: PLC0415

    if _is_uri(model):
        record["source"] = "store"
        try:
            record["problems"] += store(model)
        except Exception as error:  # noqa: BLE001 - reported, the operator decides
            record["problems"].append(f"cannot list {model} with this host's store credentials: {error!r}"[:400])
        return record
    if is_local_reference(model):
        record["source"] = "local"
        if Path(model).expanduser().exists():
            record["problems"] += local_problems(model)
        else:
            record["warnings"].append(f"--model {model} is not on this host; it must exist on the trainer node")
        return record
    vol = getattr(args, "modal_model_volume", None)
    if vol and revision and len(str(revision)) == 40:
        snap = f"{MODEL_VOLUME_MOUNT_DIR}/{model.split('/')[-1]}/{str(revision)[:8]}"
        try:
            listing = volume(vol, snap)
        except Exception as error:  # noqa: BLE001
            listing = None
            record["warnings"].append(f"cannot list Modal volume {vol}: {error!r}"[:300])
        if listing is not None:
            names, read = listing
            where = f"modal volume {vol}:/{snap}"
            try:
                manifest = json.loads(read("MANIFEST.json") or b"") if "MANIFEST.json" in names else None
            except ValueError:
                manifest = None
            if manifest and manifest.get("repo") == model and manifest.get("revision") == str(revision) \
                    and manifest.get("all_ok"):
                record["source"] = "modal-volume"
                record["problems"] += weight_problems(names, read, where)
                return record
            record["warnings"].append(f"{where} has no matching MANIFEST.json: the trainer downloads from the Hub")
        elif not record["warnings"]:
            record["warnings"].append(f"modal volume {vol} has no {snap}: the trainer downloads from the Hub")
    record["source"] = "hub"
    where = f"hf://{model}" + (f"@{revision}" if revision else "")
    try:
        names, read = hub(model, revision)
    except Exception as error:  # noqa: BLE001 - classify the Hub's answer
        kind = type(error).__name__
        if kind in ("RepositoryNotFoundError", "GatedRepoError", "RevisionNotFoundError", "HfHubHTTPError"):
            record["problems"].append(f"{where} is not readable ({kind}): check the name, --model-revision and "
                                      "the Hub token (gated models need hf auth login)")
        else:  # offline control host: the trainer may still reach the Hub
            record["warnings"].append(f"could not reach the Hub for {where}: {error!r}"[:300])
        return record
    record["problems"] += weight_problems(names, read, where)
    return record


def model_mode(args) -> str:
    mode = getattr(args, "preflight_model", None) or os.environ.get(MODE_ENV) or "check"
    if mode not in MODES:
        raise ValueError(f"--preflight-model {mode!r}: expected one of {', '.join(MODES)}")
    return mode


def preflight_trainer_model(args, *, out=None, **probes) -> dict[str, Any] | None:
    """``--preflight-model`` check of the trainer's model: raises ModelCheckError on
    a problem (mode ``check``), prints warnings (and problems in mode ``warn``)."""
    import sys  # noqa: PLC0415

    mode = model_mode(args)
    if mode == "off":
        return None
    record = trainer_model_check(args, **probes)
    out = out or sys.stderr
    for message in record["warnings"]:
        print(f"[launcher] warning: model preflight: {message}", file=out)
    if record["problems"]:
        text = "; ".join(record["problems"]) + f" (see {GUIDE}; --preflight-model warn to continue anyway)"
        if mode == "check":
            raise ModelCheckError(f"trainer model: {text}")
        print(f"[launcher] warning: trainer model: {text}", file=out)
    return record


__all__ = ["ModelCheckError", "local_problems", "preflight_trainer_model", "store_problems",
           "store_uri_problems", "trainer_model_check", "weight_problems"]
