"""Production defaults for a test that hands a rendered argv to the real parser.

tests/conftest.py changes production defaults for every test: ``yeto.cli.DEFAULT_INNER_LR``
is a plain float (an RL launch without ``--inner-lr`` is not refused), and
``YETO_PREFLIGHT_MODEL=off``, ``YETO_IMAGE_MANIFEST_CHECK=off``, ``YETO_SKY_SERVER_CHECK=off``
and ``YETO_RL_COMPAT_GROUP=nvidia-h100`` are in the environment. An argv that yeto renders
for another process (``run up`` -> ``yeto launch``, a fleet island's launch, an island's
entry, a pool's node) runs there without any of them, so a missing flag passes the suite
and is refused on the real run.

:func:`apply` puts the production defaults back for one test (monkeypatch, undone at its
end). The checks they switch back on reach no network: the Hub, a Modal volume, the
store, the registry and the sky API server answer from stubs that say the model and the
image exist. Everything else of conftest (the ~/.yeto guard, no ssh, no Ray) stays.
"""

from __future__ import annotations

import contextlib
import io
from typing import Any, Callable, Sequence

# What tests/conftest.py sets for every test that a rendered argv's own process does not get.
CONFTEST_ENV = ("YETO_PREFLIGHT_MODEL", "YETO_IMAGE_MANIFEST_CHECK", "YETO_SKY_SERVER_CHECK", "YETO_RL_COMPAT_GROUP")
MODEL_FILES = ("config.json", "model.safetensors")
HUB_COMMIT = "e" * 40  # what the stub Hub resolves a moving revision (main) to


def _model_listing(*_args: Any, **_kwargs: Any) -> tuple[list[str], Callable[[str], bytes | None]]:
    """A complete single-file checkpoint: the model exists and is readable."""
    return list(MODEL_FILES), lambda name: b"{}" if name in MODEL_FILES else None


def apply(monkeypatch: Any) -> dict[str, list[str]]:
    """Production defaults for this test; returns the record of every stubbed probe asked
    (``hub``, ``volume``, ``store``, ``registry``), so a test can tell a check ran."""
    import yeto.cli as cli
    from yeto import provenance, sky_patches
    from yeto.launch import image_manifest, model_check
    from yeto.launch.nc import pool_run

    asked: dict[str, list[str]] = {"hub": [], "volume": [], "store": [], "registry": []}
    monkeypatch.setattr(cli, "DEFAULT_INNER_LR", cli._DefaultLR(3e-4))
    for name in CONFTEST_ENV:
        monkeypatch.delenv(name, raising=False)

    def hub(repo, revision):
        asked["hub"].append(f"{repo}@{revision}")
        return _model_listing()

    def volume(vol, path):
        asked["volume"].append(f"{vol}:{path}")
        return None  # no snapshot there: the trainer downloads from the Hub

    def store(uri, environ=None):  # noqa: ARG001
        asked["store"].append(uri)
        return []

    def store_source(source, environ=None):  # noqa: ARG001
        asked["store"].append(source)
        return _model_listing()

    def bucket(uri, env_file):  # noqa: ARG001
        asked["store"].append(uri)
        return []

    def registry(ref, login=None, **_kw):  # noqa: ARG001
        asked["registry"].append(ref)
        return image_manifest.PRESENT, "stub registry: present"

    monkeypatch.setattr(model_check, "hub_files", hub)
    monkeypatch.setattr(model_check, "modal_volume_files", volume)
    monkeypatch.setattr(model_check, "store_uri_problems", store)
    monkeypatch.setattr(model_check, "model_store_source_files", store_source)
    monkeypatch.setattr(pool_run, "model_problems", bucket)
    monkeypatch.setattr(image_manifest, "manifest_status", registry)
    monkeypatch.setattr(sky_patches, "api_server_version_problem", lambda *a, **k: None)  # one sky version
    real_resolve = provenance.resolve_hub_revision

    def resolve(repo_id, requested_revision, *, repo_type, api=None):
        if api is not None or provenance.is_immutable_commit(requested_revision or ""):
            return real_resolve(repo_id, requested_revision, repo_type=repo_type, api=api)
        asked["hub"].append(f"{repo_type}:{repo_id}@{requested_revision or 'main'}")
        return requested_revision or "main", HUB_COMMIT  # a moving revision: the Hub's commit

    monkeypatch.setattr(provenance, "resolve_hub_revision", resolve)
    monkeypatch.setenv("MODAL_DISABLE_API_PROXY", "1")  # Modal is not asked; no proxy extra needed
    return asked


def launch_dry_run(argv: Sequence[str]) -> tuple[int, str]:
    """``yeto launch <argv> --dry-run`` through the real ``yeto.cli`` (its parser and the
    launch's checks), as the process the argv is handed to runs it: (rc, stdout+stderr)."""
    import yeto.cli as cli

    argv = [str(a) for a in argv]
    if "--dry-run" not in argv:
        argv.append("--dry-run")
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        try:
            rc = int(cli.main(["launch", *argv]) or 0)
        except SystemExit as stop:  # argparse: a missing or bad flag
            rc = int(stop.code or 0) if not isinstance(stop.code, str) else 2
            out.write(f"\nSystemExit: {stop.code}\n")
        except Exception as error:  # noqa: BLE001 - a refusal of the launch's checks, shown to the test
            rc = 1
            out.write(f"\n{type(error).__name__}: {error}\n")
    return rc, out.getvalue()


def refusal(text: str, limit: int = 40) -> str:
    """The lines of a dry run that say why it failed (for an assertion message)."""
    import re

    marks = re.compile(r"INVALID|refus|rejected|required|the following arguments|error|Error|cannot|must |"
                       r"Traceback|SystemExit|inner-lr|preflight|problem", re.I)
    lines = [line for line in text.splitlines() if marks.search(line)]
    return "\n".join(lines[:limit]) or text[-3000:]

