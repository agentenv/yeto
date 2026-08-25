from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "probes"
    / "qwen35_conversion_manifest.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "qwen35_conversion_manifest", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
build = _MODULE.build
verify = _MODULE.verify


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_conversion_manifest_binds_model_checkpoint_and_empty_source_file(tmp_path):
    revision = "a" * 40
    image_digest = f"sha256:{'b' * 64}"
    model = tmp_path / "model"
    checkpoint = tmp_path / "checkpoint"
    source = tmp_path / "source"
    metadata = model / ".cache" / "huggingface" / "download"
    metadata.mkdir(parents=True)
    checkpoint.mkdir()
    source.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen3_5"}\n')
    (model / "model.safetensors").write_bytes(b"weights")
    (metadata / "config.json.metadata").write_text(f"{revision}\nobject\ntime\n")
    (metadata / "model.safetensors.metadata").write_text(f"{revision}\nobject\ntime\n")
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("release")
    release = checkpoint / "release"
    release.mkdir()
    (release / "shard.pt").write_bytes(b"checkpoint")
    (source / "__init__.py").write_bytes(b"")
    output = checkpoint / "conversion-manifest.json"

    build(
        SimpleNamespace(
            model=model,
            checkpoint=checkpoint,
            conversion_source=source,
            revision=revision,
            image_digest=image_digest,
            output=output,
        )
    )
    verify(
        SimpleNamespace(
            model=model,
            manifest=output,
            expected_sha256=_hash(output),
            expected_revision=revision,
            expected_config_sha256=_hash(model / "config.json"),
            expected_image_digest=image_digest,
        )
    )
