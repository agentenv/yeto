"""Fail-closed source resolution and artifact provenance.

Production entry points resolve every remote Hugging Face reference to a
commit before loading it.  Local paths deliberately remain local and do not
accept a misleading Hub revision.  The resulting record is attached to the
runtime arguments so config, tokenizer, model, dataset, and artifact writers
all use the same immutable identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROVENANCE_FILE = "yeto_provenance.json"
PROVENANCE_SCHEMA_VERSION = 1

_COMMIT_RE = re.compile(r"[0-9a-fA-F]{40}\Z")
_PATH_PREFIXES = ("/", "./", "../", "~")
_URI_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://")


class ProvenanceError(ValueError):
    """A source cannot be given an immutable, internally consistent identity."""


def is_immutable_commit(value: str | None) -> bool:
    return bool(value and _COMMIT_RE.fullmatch(value))


def is_local_reference(value: str) -> bool:
    """Return whether *value* is path-shaped or already exists locally.

    A missing explicit path remains local so a typo cannot be reinterpreted as
    a Hub repository name.  Bare Hub ids (including legacy one-component
    dataset ids) remain remote.
    """

    expanded = os.path.expanduser(value)
    return value.startswith(_PATH_PREFIXES) or os.path.exists(expanded)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def python_spec_path(spec: str, *, base_dir: str | Path | None = None) -> Path:
    """Locate the source selected by a ``module-or-file:factory`` spec."""
    target, separator, _factory = spec.partition(":")
    if not separator or not target:
        raise ProvenanceError(
            "Python adapter spec must be module:factory or file.py:factory"
        )
    if target.endswith(".py") or os.path.sep in target:
        path = Path(target).expanduser()
        if not path.is_absolute() and base_dir is not None:
            path = Path(base_dir) / path
    else:
        # importlib.util.find_spec("parent.child") imports ``parent`` as a
        # side effect. Walk filesystem specs directly so an untrusted package
        # initializer cannot execute before the selected source is attested.
        from importlib.machinery import PathFinder

        parts = target.split(".")
        if not all(part and part.isidentifier() for part in parts):
            raise ProvenanceError(f"invalid Python adapter module {target!r}")
        search_path = None
        module_spec = None
        module_name = ""
        for index, part in enumerate(parts):
            module_name = f"{module_name}.{part}" if module_name else part
            module_spec = PathFinder.find_spec(module_name, search_path)
            if module_spec is None:
                break
            if index < len(parts) - 1:
                locations = module_spec.submodule_search_locations
                if locations is None:
                    module_spec = None
                    break
                search_path = list(locations)
        if module_spec is None or not module_spec.origin:
            raise ProvenanceError(f"cannot locate Python adapter module {target!r}")
        path = Path(module_spec.origin)
    if path.suffix != ".py" or not path.is_file():
        raise ProvenanceError(
            f"Python adapter {target!r} has no attestable .py source file"
        )
    return path.resolve()


def python_spec_sha256(spec: str, *, base_dir: str | Path | None = None) -> str:
    """Hash the Python source selected by a ``module-or-file:factory`` spec."""

    return file_sha256(python_spec_path(spec, base_dir=base_dir))


def source_tree_sha256(package_root: str | Path | None = None) -> str:
    """Stable digest of the installed Yeto Python source tree."""

    root = Path(package_root) if package_root is not None else Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def verify_source_tree_sha256(expected: str | None = None) -> str:
    """Attest the installed Python tree, rejecting launcher/worker drift."""

    actual = source_tree_sha256()
    if expected is not None and actual != expected.lower():
        raise ProvenanceError(
            f"Yeto source SHA256 mismatch: launcher expected {expected.lower()}, "
            f"worker has {actual}"
        )
    return actual


def verify_distributed_source_tree_sha256(
    expected: str | None,
    *,
    rank: int,
    world: int,
) -> str:
    """Collectively attest code so one mismatched rank cannot strand peers."""

    return _verify_distributed_sha256(
        source_tree_sha256,
        expected,
        rank=rank,
        world=world,
        artifact="Yeto source",
    )


def _verify_distributed_sha256(
    compute_digest: Callable[[], str],
    expected: str | None,
    *,
    rank: int,
    world: int,
    artifact: str,
) -> str:
    """Collect a digest or read error before any rank can use an artifact."""

    expected_digest = expected.lower() if expected is not None else None
    try:
        actual = compute_digest()
    except Exception as exc:  # peers still have to enter the collective
        local: dict[str, Any] = {
            "rank": rank,
            "ok": False,
            "digest": None,
            "expected": expected_digest,
            "error": f"{type(exc).__name__}: {exc}",
        }
    else:
        local = {
            "rank": rank,
            "ok": True,
            "digest": actual,
            "expected": expected_digest,
            "error": None,
        }

    if world <= 1:
        records: list[dict[str, Any] | None] = [local]
    else:
        import torch.distributed as dist

        records = [None] * world
        dist.all_gather_object(records, local)

    valid = [record for record in records if record and record.get("ok")]
    digests = {record.get("digest") for record in valid}
    failed = [record for record in records if not record or not record.get("ok")]
    expected_values = {record.get("expected") for record in records if record}
    shared_expected = next(iter(expected_values)) if len(expected_values) == 1 else None
    wrong_expected = [
        record
        for record in valid
        if shared_expected is not None and record.get("digest") != shared_expected
    ]
    if (
        failed
        or wrong_expected
        or len(valid) != world
        or len(digests) != 1
        or len(expected_values) != 1
    ):
        detail = ", ".join(
            "missing rank record"
            if record is None
            else (
                f"rank {record.get('rank')}={record.get('digest')}"
                if record.get("ok")
                else f"rank {record.get('rank')} error={record.get('error')}"
            )
            for record in records
        )
        scope = "distributed " if world > 1 else ""
        raise ProvenanceError(
            f"{scope}{artifact} attestation failed "
            f"(expected_by_rank={sorted(map(str, expected_values))!r}; {detail})"
        )

    return str(valid[0]["digest"])


def verify_distributed_file_sha256(
    path: str | Path,
    expected: str | None,
    *,
    rank: int,
    world: int,
    artifact: str = "executable artifact",
) -> str:
    """Collectively attest one file before any rank executes its contents."""

    _payload, digest = read_distributed_file_bytes(
        path,
        expected,
        rank=rank,
        world=world,
        artifact=artifact,
    )
    return digest


def read_distributed_file_bytes(
    path: str | Path,
    expected: str | None,
    *,
    rank: int,
    world: int,
    artifact: str = "executable artifact",
) -> tuple[bytes, str]:
    """Read once, collectively attest, and return the exact executable bytes."""

    payload: bytes | None = None

    def compute_digest() -> str:
        nonlocal payload
        payload = Path(path).read_bytes()
        return hashlib.sha256(payload).hexdigest()

    digest = _verify_distributed_sha256(
        compute_digest,
        expected,
        rank=rank,
        world=world,
        artifact=artifact,
    )
    if payload is None:  # pragma: no cover - a successful attestation set it
        raise AssertionError(f"{artifact} attestation returned without local bytes")
    return payload, digest


def verify_distributed_python_spec_sha256(
    spec: str,
    expected: str | None,
    *,
    rank: int,
    world: int,
) -> str:
    """Collectively attest the source selected by a Python factory spec."""

    _path, _source, digest = read_distributed_python_spec_bytes(
        spec,
        expected,
        rank=rank,
        world=world,
    )
    return digest


def read_distributed_python_spec_bytes(
    spec: str,
    expected: str | None,
    *,
    rank: int,
    world: int,
) -> tuple[Path, bytes, str]:
    """Resolve and attest a factory source without rereading it before execution."""

    path: Path | None = None
    source: bytes | None = None

    def compute_digest() -> str:
        nonlocal path, source
        path = python_spec_path(spec)
        source = path.read_bytes()
        return hashlib.sha256(source).hexdigest()

    digest = _verify_distributed_sha256(
        compute_digest,
        expected,
        rank=rank,
        world=world,
        artifact="diffusion adapter",
    )
    if path is None or source is None:  # pragma: no cover - success set both
        raise AssertionError("diffusion adapter attestation returned without source bytes")
    return path, source, digest


def resolve_hub_revision(
    repo_id: str,
    requested_revision: str | None,
    *,
    repo_type: str,
    api=None,
) -> tuple[str, str]:
    """Return ``(requested, immutable_commit)`` for a Hub repository.

    An already immutable commit needs no control-plane lookup and remains
    usable from a warm offline cache.  Moving branches/tags are resolved via
    the Hub API and rejected unless the API returns a full Git commit.
    """

    requested = requested_revision or "main"
    if is_immutable_commit(requested):
        return requested, requested.lower()
    if repo_type not in {"model", "dataset"}:
        raise ValueError(f"unsupported Hub repo type {repo_type!r}")
    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ProvenanceError(
                "huggingface_hub is required to resolve moving Hub revisions; "
                "install the production dependencies or pass a 40-character commit"
            ) from exc
        api = HfApi()
    try:
        if repo_type == "model":
            info = api.model_info(repo_id, revision=requested)
        else:
            info = api.dataset_info(repo_id, revision=requested)
    except Exception as exc:
        raise ProvenanceError(
            f"could not resolve {repo_type} {repo_id!r} revision {requested!r} "
            "to an immutable Hugging Face commit"
        ) from exc
    commit = getattr(info, "sha", None)
    if not is_immutable_commit(commit):
        raise ProvenanceError(
            f"Hugging Face returned invalid commit {commit!r} for {repo_type} "
            f"{repo_id!r} revision {requested!r}"
        )
    return requested, commit.lower()


def resolve_reference(
    identifier: str,
    requested_revision: str | None,
    *,
    repo_type: str,
    original_identifier: str | None = None,
    api=None,
) -> dict[str, Any]:
    if _URI_RE.match(identifier):
        if requested_revision is not None:
            raise ProvenanceError(
                f"--{repo_type}-revision applies only to Hugging Face repositories; "
                f"{identifier!r} is an external URI"
            )
        return {
            "source": "external-uri",
            "requested_identifier": original_identifier or identifier,
            "resolved_identifier": identifier,
            "requested_revision": None,
            "resolved_revision": None,
        }
    if is_local_reference(identifier):
        if requested_revision is not None:
            raise ProvenanceError(
                f"--{repo_type}-revision applies only to Hugging Face repositories; "
                f"{identifier!r} is a local path"
            )
        return {
            "source": "local",
            "requested_identifier": original_identifier or identifier,
            "resolved_identifier": str(Path(identifier).expanduser().resolve(strict=False)),
            "requested_revision": None,
            "resolved_revision": None,
        }
    requested, commit = resolve_hub_revision(
        identifier,
        requested_revision,
        repo_type=repo_type,
        api=api,
    )
    return {
        "source": "huggingface",
        "requested_identifier": original_identifier or identifier,
        "resolved_identifier": identifier,
        "requested_revision": requested,
        "resolved_revision": commit,
    }


def pin_runtime_provenance(args, *, include_data: bool = True, api=None) -> dict[str, Any]:
    """Resolve remote runtime inputs and apply their commits back to *args*."""

    from .models import resolve

    requested_model = getattr(args, "model", None)
    if not requested_model:
        raise ProvenanceError("a model reference is required for provenance resolution")
    resolved_model = resolve(requested_model)
    model = resolve_reference(
        resolved_model,
        getattr(args, "model_revision", None),
        repo_type="model",
        original_identifier=getattr(
            args, "model_requested_identifier", None
        ) or requested_model,
        api=api,
    )
    if getattr(args, "model_requested_revision", None) is not None:
        model["requested_revision"] = args.model_requested_revision
    payload: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "model": model,
        "trust_remote_code": bool(getattr(args, "trust_remote_code", False)),
    }
    args.model_revision = model["resolved_revision"]

    data_value = getattr(args, "data", None)
    if include_data and isinstance(data_value, str):
        data = resolve_reference(
            data_value,
            getattr(args, "data_revision", None),
            repo_type="dataset",
            original_identifier=getattr(
                args, "data_requested_identifier", None
            ) or data_value,
            api=api,
        )
        if getattr(args, "data_requested_revision", None) is not None:
            data["requested_revision"] = args.data_requested_revision
        payload["dataset"] = data
        args.data_revision = data["resolved_revision"]
    elif include_data and data_value is not None:
        if getattr(args, "data_revision", None) is not None:
            raise ProvenanceError(
                "--data-revision cannot be used with an in-memory dataset"
            )
        payload["dataset"] = {
            "source": "in-memory",
            "requested_identifier": None,
            "resolved_identifier": None,
            "requested_revision": None,
            "resolved_revision": None,
        }

    args._provenance = payload
    return payload


def apply_runtime_provenance(args, payload: dict[str, Any]) -> None:
    """Apply a rank-zero provenance record to another distributed rank."""

    args.model_revision = payload["model"]["resolved_revision"]
    if "dataset" in payload:
        args.data_revision = payload["dataset"]["resolved_revision"]
    args.trust_remote_code = bool(payload.get("trust_remote_code", False))
    args._provenance = payload


def pin_distributed_runtime_provenance(
    args,
    *,
    rank: int,
    world: int,
    include_data: bool = True,
) -> dict[str, Any]:
    """Resolve on rank zero and broadcast one authoritative source record."""

    if world <= 1:
        return pin_runtime_provenance(args, include_data=include_data)
    import torch.distributed as dist

    box: list[dict[str, Any] | None] = [None]
    if rank == 0:
        try:
            box[0] = {
                "ok": True,
                "payload": pin_runtime_provenance(args, include_data=include_data),
            }
        except Exception as exc:  # all peers must leave the collective together
            box[0] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    dist.broadcast_object_list(box, src=0)
    result = box[0]
    if not result or not result.get("ok"):
        detail = result.get("error", "rank zero returned no provenance") if result else "missing result"
        raise ProvenanceError(f"distributed provenance resolution failed: {detail}")
    payload = result["payload"]
    apply_runtime_provenance(args, payload)
    return payload


def model_load_kwargs(args) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": bool(getattr(args, "trust_remote_code", False))
    }
    revision = getattr(args, "model_revision", None)
    if revision:
        if not is_immutable_commit(revision):
            raise ProvenanceError(
                f"model revision {revision!r} is not an immutable 40-character commit"
            )
        kwargs["revision"] = revision.lower()
    return kwargs


def materialize_pinned_model(args) -> str:
    """Return a local snapshot path for a custom loader's pinned base model."""

    from .models import resolve

    model_id = resolve(args.model)
    if is_local_reference(model_id):
        return str(Path(model_id).expanduser().resolve(strict=False))
    revision = getattr(args, "model_revision", None)
    if not is_immutable_commit(revision):
        raise ProvenanceError(
            "custom remote model loading requires a resolved immutable model revision"
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ProvenanceError(
            "huggingface_hub is required to materialize a pinned model snapshot"
        ) from exc
    try:
        return snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_files_only=True,
        )
    except OSError:
        return snapshot_download(repo_id=model_id, revision=revision)


def require_custom_loader_contract(adapter, args) -> None:
    """Reject custom remote loaders that do not attest pinned-source support."""

    model = (getattr(args, "_provenance", {}) or {}).get("model") or {}
    if model.get("source") != "huggingface":
        return
    if not bool(getattr(adapter, "supports_pinned_model_source", False)):
        raise ProvenanceError(
            "the selected diffusion adapter has a custom model loader but does not "
            "declare supports_pinned_model_source=True; update it to load "
            "materialize_pinned_model(args) (or pass revision/trust flags to every "
            "underlying loader) before production use"
        )


def provenance_metadata(args) -> dict[str, Any]:
    payload = dict(getattr(args, "_provenance", {}) or {})
    if not payload:
        # Artifact helpers used directly by tests/callers may not have gone
        # through a CLI main. Never perform a network lookup while saving and
        # never promote a caller-supplied moving revision into an attested
        # resolved revision.
        from .models import resolve

        model = getattr(args, "model", None)
        payload = {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "attestation_status": "unattested",
            "attestation_reason": (
                "runtime provenance was not pinned before artifact writing"
            ),
            "model": {
                "source": "unattested",
                "requested_identifier": model,
                "resolved_identifier": resolve(model) if model else None,
                "requested_revision": getattr(args, "model_revision", None),
                "resolved_revision": None,
                "attestation_status": "unattested",
            },
            "trust_remote_code": bool(getattr(args, "trust_remote_code", False)),
        }
        data = getattr(args, "data", None)
        if data is not None:
            payload["dataset"] = {
                "source": "unattested",
                "requested_identifier": data if isinstance(data, str) else None,
                "resolved_identifier": None,
                "requested_revision": getattr(args, "data_revision", None),
                "resolved_revision": None,
                "attestation_status": "unattested",
            }
    payload["yeto_source_sha256"] = verify_source_tree_sha256(
        getattr(args, "source_sha256", None)
    )
    loss_hash = getattr(args, "loss_sha256", None)
    if loss_hash:
        payload["loss_artifact"] = {
            "spec": getattr(args, "loss_function", None),
            "sha256": loss_hash,
            "unsafe_pickle_enabled": bool(
                getattr(args, "allow_unsafe_pickled_loss", False)
            ),
        }
    return payload


def write_provenance_manifest(
    output_dir: str | Path,
    args,
    *,
    artifact_kind: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    payload = provenance_metadata(args)
    payload["artifact_kind"] = artifact_kind
    if extra:
        payload["artifact"] = extra
    path = out / PROVENANCE_FILE
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path
