"""Fail-closed production source and executable-artifact policies."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto import cli, launcher, losses, provenance


class _Api:
    def __init__(self, model_sha="a" * 40, dataset_sha="b" * 40):
        self.model_sha = model_sha
        self.dataset_sha = dataset_sha
        self.calls = []

    def model_info(self, repo_id, revision):
        self.calls.append(("model", repo_id, revision))
        return SimpleNamespace(sha=self.model_sha)

    def dataset_info(self, repo_id, revision):
        self.calls.append(("dataset", repo_id, revision))
        return SimpleNamespace(sha=self.dataset_sha)


def test_moving_model_and_dataset_refs_resolve_to_commits_once():
    api = _Api()
    args = SimpleNamespace(
        model="qwen3-8b",
        model_revision="release",
        data="org/data",
        data_revision=None,
        trust_remote_code=False,
    )

    record = provenance.pin_runtime_provenance(args, api=api)

    assert args.model_revision == "a" * 40
    assert args.data_revision == "b" * 40
    assert record["model"] == {
        "source": "huggingface",
        "requested_identifier": "qwen3-8b",
        "resolved_identifier": "Qwen/Qwen3-8B",
        "requested_revision": "release",
        "resolved_revision": "a" * 40,
    }
    assert record["dataset"]["requested_revision"] == "main"
    assert api.calls == [
        ("model", "Qwen/Qwen3-8B", "release"),
        ("dataset", "org/data", "main"),
    ]


def test_forwarded_origin_preserves_user_tags_while_loading_commits():
    args = SimpleNamespace(
        model="Qwen/Qwen3-8B",
        model_revision="a" * 40,
        model_requested_identifier="qwen3-8b",
        model_requested_revision="release",
        data="org/data",
        data_revision="b" * 40,
        data_requested_identifier="org/data",
        data_requested_revision="data-v4",
        trust_remote_code=False,
    )
    record = provenance.pin_runtime_provenance(args)
    assert record["model"]["requested_identifier"] == "qwen3-8b"
    assert record["model"]["requested_revision"] == "release"
    assert record["model"]["resolved_revision"] == "a" * 40
    assert record["dataset"]["requested_revision"] == "data-v4"
    assert record["dataset"]["resolved_revision"] == "b" * 40


def test_already_immutable_commit_never_needs_the_hub():
    class NoCalls:
        def model_info(self, *args, **kwargs):
            raise AssertionError("immutable commit made a control-plane request")

    requested, resolved = provenance.resolve_hub_revision(
        "org/model", "ABCDEF0123456789ABCDEF0123456789ABCDEF01", repo_type="model", api=NoCalls()
    )
    assert requested == "ABCDEF0123456789ABCDEF0123456789ABCDEF01"
    assert resolved == requested.lower()


def test_invalid_hub_sha_and_local_revision_fail_closed(tmp_path):
    with pytest.raises(provenance.ProvenanceError, match="invalid commit"):
        provenance.resolve_hub_revision(
            "org/model", "main", repo_type="model", api=_Api(model_sha="branch")
        )

    local = tmp_path / "model"
    local.mkdir()
    with pytest.raises(provenance.ProvenanceError, match="local path"):
        provenance.resolve_reference(
            str(local), "a" * 40, repo_type="model"
        )


def test_ssh_remote_local_model_keeps_an_immutable_source_revision():
    record = provenance.resolve_reference(
        "/data/yeto-rl/models/deepseek-v4-flash-bf16",
        "a" * 40,
        repo_type="model",
        allow_remote_local_revision=True,
    )

    assert record == {
        "source": "remote-local",
        "requested_identifier": "/data/yeto-rl/models/deepseek-v4-flash-bf16",
        "resolved_identifier": "/data/yeto-rl/models/deepseek-v4-flash-bf16",
        "requested_revision": "a" * 40,
        "resolved_revision": "a" * 40,
    }


def test_local_and_external_data_remain_usable_without_fake_revisions(tmp_path):
    local = tmp_path / "rows.jsonl"
    local.write_text("{}\n", encoding="utf-8")
    local_record = provenance.resolve_reference(str(local), None, repo_type="dataset")
    uri_record = provenance.resolve_reference("s3://bucket/rows", None, repo_type="dataset")

    assert local_record["source"] == "local"
    assert local_record["resolved_identifier"] == str(local.resolve())
    assert uri_record["source"] == "external-uri"
    with pytest.raises(provenance.ProvenanceError, match="external URI"):
        provenance.resolve_reference(
            "s3://bucket/rows", "a" * 40, repo_type="dataset"
        )


def test_model_load_kwargs_default_to_no_remote_code_and_require_commit():
    args = SimpleNamespace(model_revision="A" * 40)
    assert provenance.model_load_kwargs(args) == {
        "trust_remote_code": False,
        "revision": "a" * 40,
    }
    with pytest.raises(provenance.ProvenanceError, match="not an immutable"):
        provenance.model_load_kwargs(SimpleNamespace(model_revision="main"))


def test_causal_tokenizer_and_model_receive_one_pinned_safe_recipe(monkeypatch):
    torch = pytest.importorskip("torch")
    calls = []

    class TokenizerFactory:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(("tokenizer", model_id, kwargs))
            return SimpleNamespace()

    class Model:
        config = SimpleNamespace(_attn_implementation="sdpa")

        def to(self, device):
            calls.append(("to", str(device)))
            return self

    class ModelFactory:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(("model", model_id, kwargs))
            return Model()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoConfig=object(),
            AutoModelForCausalLM=ModelFactory,
            AutoTokenizer=TokenizerFactory,
        ),
    )
    from yeto.learner import load_model_and_tokenizer

    args = SimpleNamespace(
        model="org/model",
        model_revision="a" * 40,
        trust_remote_code=False,
        base_quantization="none",
        tuning="full",
        shard="ddp",
        loss_function="cross_entropy",
        kernel_backend="native",
        attention_backend="auto",
    )
    load_model_and_tokenizer(args, torch.device("cpu"))

    tokenizer_kwargs = calls[0][2]
    model_kwargs = calls[1][2]
    for kwargs in (tokenizer_kwargs, model_kwargs):
        assert kwargs["revision"] == "a" * 40
        assert kwargs["trust_remote_code"] is False
        assert kwargs["local_files_only"] is True
    assert model_kwargs["use_safetensors"] is True


def test_generic_diffusion_pipeline_receives_pinned_safe_recipe(monkeypatch):
    torch = pytest.importorskip("torch")
    calls = []

    class Pipe:
        components = {}

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append((model_id, kwargs))
            return cls()

    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(DiffusionPipeline=Pipe))
    from yeto.diffusion.learner import load_pipeline

    args = SimpleNamespace(
        model="org/model",
        model_revision="a" * 40,
        trust_remote_code=False,
        seed=None,
        tuning="full",
    )
    with pytest.raises(RuntimeError, match="no trainable diffusion module"):
        load_pipeline(args, torch.device("cpu"))
    assert calls == [
        (
            "org/model",
            {
                "local_files_only": True,
                "torch_dtype": torch.float32,
                "use_safetensors": True,
                "trust_remote_code": False,
                "revision": "a" * 40,
            },
        )
    ]


def test_source_tree_attestation_detects_launcher_worker_drift():
    actual = provenance.verify_source_tree_sha256()
    assert len(actual) == 64
    assert provenance.verify_source_tree_sha256(actual.upper()) == actual
    with pytest.raises(provenance.ProvenanceError, match="source SHA256 mismatch"):
        provenance.verify_source_tree_sha256("0" * 64)


def test_distributed_artifact_attestation_adopts_one_shared_digest(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    artifact = tmp_path / "loss.py"
    artifact.write_bytes(b"shared executable bytes")
    digest = provenance.file_sha256(artifact)

    def all_gather_object(records, local):
        assert local == {
            "rank": 0,
            "ok": True,
            "digest": digest,
            "expected": None,
            "error": None,
        }
        records[:] = [local, {**local, "rank": 1}]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    assert provenance.verify_distributed_file_sha256(
        artifact,
        None,
        rank=0,
        world=2,
        artifact="custom loss",
    ) == digest


def test_distributed_artifact_attestation_rejects_rank_digest_drift(
    monkeypatch, tmp_path
):
    torch = pytest.importorskip("torch")
    artifact = tmp_path / "adapter.py"
    artifact.write_bytes(b"rank zero bytes")

    def all_gather_object(records, local):
        records[:] = [
            local,
            {
                "rank": 1,
                "ok": True,
                "digest": "1" * 64,
                "expected": None,
                "error": None,
            },
        ]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    with pytest.raises(
        provenance.ProvenanceError,
        match="distributed diffusion adapter attestation failed",
    ):
        provenance.verify_distributed_python_spec_sha256(
            f"{artifact}:make_adapter",
            None,
            rank=0,
            world=2,
        )


def test_distributed_artifact_attestation_rejects_rank_expected_digest_drift(
    monkeypatch, tmp_path
):
    torch = pytest.importorskip("torch")
    artifact = tmp_path / "loss.py"
    artifact.write_bytes(b"shared executable bytes")
    digest = provenance.file_sha256(artifact)

    def all_gather_object(records, local):
        records[:] = [local, {**local, "rank": 1, "expected": None}]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    with pytest.raises(provenance.ProvenanceError, match="expected_by_rank"):
        provenance.verify_distributed_file_sha256(
            artifact,
            digest,
            rank=0,
            world=2,
            artifact="custom loss",
        )


def test_distributed_artifact_attestation_gathers_missing_file_before_raising(
    monkeypatch, tmp_path
):
    torch = pytest.importorskip("torch")
    gathered = False

    def all_gather_object(records, local):
        nonlocal gathered
        gathered = True
        assert local["ok"] is False
        assert "FileNotFoundError" in local["error"]
        records[:] = [
            local,
            {
                "rank": 1,
                "ok": True,
                "digest": "2" * 64,
                "expected": None,
                "error": None,
            },
        ]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    with pytest.raises(provenance.ProvenanceError, match="FileNotFoundError"):
        provenance.verify_distributed_file_sha256(
            tmp_path / "missing.pkl",
            None,
            rank=0,
            world=2,
            artifact="pickled loss",
        )
    assert gathered


def test_distributed_artifact_attestation_gathers_read_error_before_raising(
    monkeypatch, tmp_path
):
    torch = pytest.importorskip("torch")
    artifact = tmp_path / "loss.py"
    artifact.write_bytes(b"unreadable")
    gathered = False

    def fail_read(_path):
        raise PermissionError("read denied")

    def all_gather_object(records, local):
        nonlocal gathered
        gathered = True
        assert local["ok"] is False
        assert local["error"] == "PermissionError: read denied"
        records[:] = [local, {**local, "rank": 1}]

    monkeypatch.setattr(Path, "read_bytes", fail_read)
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    with pytest.raises(provenance.ProvenanceError, match="PermissionError: read denied"):
        provenance.verify_distributed_file_sha256(
            artifact,
            "0" * 64,
            rank=0,
            world=2,
        )
    assert gathered


def test_pinned_dataset_load_never_switches_to_conversion_ref(monkeypatch):
    calls = []

    def load_dataset(name, trust_remote_code=None, **kwargs):
        kwargs["trust_remote_code"] = trust_remote_code
        calls.append((name, kwargs))
        raise OSError("schema mismatch")

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    from yeto.data import load_rows

    with pytest.raises(RuntimeError, match="refusing the moving"):
        load_rows("org/data", revision="b" * 40)
    assert calls == [
        (
            "org/data",
            {
                "split": "train",
                "trust_remote_code": False,
                "revision": "b" * 40,
            },
        )
    ]


def test_unpinned_legacy_dataset_helper_retains_compatibility_fallback(monkeypatch):
    calls = []

    def load_dataset(name, trust_remote_code=None, **kwargs):
        kwargs["trust_remote_code"] = trust_remote_code
        calls.append((name, kwargs))
        if len(calls) == 1:
            raise OSError("schema mismatch")
        return ["normalized"]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    from yeto.data import load_rows

    assert load_rows("org/data") == ["normalized"]
    assert calls[1][1]["revision"] == "refs/convert/parquet"


def test_custom_loss_executes_only_the_bytes_matching_its_hash(tmp_path):
    source = tmp_path / "loss.py"
    source.write_text(
        "def loss_fn(logits, input_ids, weights):\n"
        "    return logits.sum(), weights.sum()\n",
        encoding="utf-8",
    )
    digest = provenance.file_sha256(source)
    assert callable(
        losses.load_custom_loss(f"custom:{source}", expected_sha256=digest)
    )
    source.write_text("raise RuntimeError('changed')\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        losses.load_custom_loss(f"custom:{source}", expected_sha256=digest)


def test_custom_loss_executes_the_exact_collectively_attested_buffer(tmp_path):
    source = tmp_path / "loss.py"
    source.write_text(
        "def loss_fn(*args):\n    return 'attested'\n",
        encoding="utf-8",
    )
    payload, digest = provenance.read_distributed_file_bytes(
        source,
        None,
        rank=0,
        world=1,
        artifact="custom loss",
    )
    source.write_text(
        "def loss_fn(*args):\n    return 'replacement'\n",
        encoding="utf-8",
    )

    fn = losses.load_custom_loss(
        f"custom:{source}",
        expected_sha256=digest,
        source_bytes=payload,
    )
    assert fn() == "attested"


def test_pickle_is_opt_in_and_hash_attested(tmp_path):
    path = tmp_path / "loss.pkl"
    losses.dump_pickled_loss(lambda *args: (0, 0), path)
    digest = provenance.file_sha256(path)
    with pytest.raises(PermissionError, match="allow-unsafe-pickled-loss"):
        losses.load_pickled_loss(f"pickle:{path}", expected_sha256=digest)
    assert callable(
        losses.load_pickled_loss(
            f"pickle:{path}", allow_unsafe=True, expected_sha256=digest
        )
    )
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        losses.load_pickled_loss(
            f"pickle:{path}", allow_unsafe=True, expected_sha256="0" * 64
        )


def test_launcher_stages_unsafe_pickle_at_fixed_shell_neutral_path(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "REPO_ROOT", tmp_path)
    dangerous = tmp_path / "loss; touch SHOULD_NOT_EXIST.pkl"
    losses.dump_pickled_loss(lambda *args: (0, 0), dangerous)

    with pytest.raises(PermissionError, match="requires --allow-unsafe"):
        launcher.resolve_loss_function(f"pickle:{dangerous}")
    spec = launcher.resolve_loss_function(
        f"pickle:{dangerous}", allow_unsafe_pickled_loss=True
    )

    assert spec.startswith(f"pickle:{launcher.PICKLED_LOSS_PREFIX}")
    assert launcher.pickled_loss_path(spec).read_bytes() == dangerous.read_bytes()
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists()


def test_pickled_loss_staging_is_content_addressed_per_run(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "REPO_ROOT", tmp_path)
    first = tmp_path / "first.pkl"
    second = tmp_path / "second.pkl"
    first.write_bytes(b"first executable payload")
    second.write_bytes(b"second executable payload")

    first_spec = launcher.resolve_loss_function(
        f"pickle:{first}", allow_unsafe_pickled_loss=True
    )
    second_spec = launcher.resolve_loss_function(
        f"pickle:{second}", allow_unsafe_pickled_loss=True
    )

    assert first_spec != second_spec
    assert launcher.pickled_loss_path(first_spec).read_bytes() == first.read_bytes()
    assert launcher.pickled_loss_path(second_spec).read_bytes() == second.read_bytes()


def test_launcher_does_not_execute_custom_loss_before_unsafe_opt_in(tmp_path):
    marker = tmp_path / "EXECUTED"
    source = tmp_path / "loss.py"
    source.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n",
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="requires --allow-unsafe"):
        launcher.resolve_loss_function(f"custom:{source}")
    assert not marker.exists()


def test_launcher_attests_synced_source_and_diffusion_adapter(monkeypatch, tmp_path):
    monkeypatch.setattr(launcher, "REPO_ROOT", tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    data = tmp_path / "rows.jsonl"
    data.write_text("{}\n", encoding="utf-8")
    adapter = tmp_path / "adapter.py"
    adapter.write_text("def make_adapter():\n    return object()\n", encoding="utf-8")
    args = SimpleNamespace(
        model=str(model),
        model_revision=None,
        data=str(data),
        data_revision=None,
        trust_remote_code=False,
        loss_function="cross_entropy",
        allow_unsafe_pickled_loss=False,
        loss_sha256=None,
        source_sha256=None,
        diffusion_adapter=f"{adapter}:make_adapter",
        diffusion_adapter_sha256=None,
    )

    launcher.prepare_launch_args(args)

    assert args.source_sha256 == provenance.source_tree_sha256()
    assert args.diffusion_adapter_sha256 == provenance.file_sha256(adapter)
    assert args.diffusion_adapter == "adapter.py:make_adapter"
    adapter.write_text("def make_adapter():\n    return None\n", encoding="utf-8")
    with pytest.raises(ValueError, match="diffusion adapter SHA256 mismatch"):
        launcher.prepare_launch_args(args)


def test_cloud_sampler_preserves_and_checks_expected_adapter_digest(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(launcher, "REPO_ROOT", tmp_path)
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "def make_adapter():\n    return object()\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        source_sha256=None,
        diffusion_adapter=f"{adapter}:make_adapter",
        diffusion_adapter_sha256="0" * 64,
    )

    with pytest.raises(ValueError, match="expected sampler attestation"):
        launcher.run_diffusion_sample(args)


def test_dotted_adapter_discovery_does_not_execute_parent_package(
    monkeypatch, tmp_path
):
    package = tmp_path / "untrusted_adapter_package"
    package.mkdir()
    marker = tmp_path / "PARENT_EXECUTED"
    (package / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n",
        encoding="utf-8",
    )
    source = package / "hook.py"
    source.write_text(
        "def make_adapter():\n    return object()\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    assert provenance.python_spec_path(
        "untrusted_adapter_package.hook:make_adapter"
    ) == source.resolve()
    assert provenance.python_spec_sha256(
        "untrusted_adapter_package.hook:make_adapter"
    ) == provenance.file_sha256(source)
    from yeto.diffusion.learner import load_diffusion_adapter

    assert load_diffusion_adapter(
        "untrusted_adapter_package.hook:make_adapter",
        expected_sha256=provenance.file_sha256(source),
    ) is not None
    assert not marker.exists()


def test_provenance_manifest_records_resolved_inputs_source_and_loss(tmp_path):
    args = SimpleNamespace(
        model="org/model",
        model_revision="a" * 40,
        data="org/data",
        data_revision="b" * 40,
        trust_remote_code=False,
        loss_function="pickle:.yeto_loss.pkl",
        loss_sha256="c" * 64,
        allow_unsafe_pickled_loss=True,
        _provenance={
            "schema_version": 1,
            "model": {"resolved_revision": "a" * 40},
            "dataset": {"resolved_revision": "b" * 40},
            "trust_remote_code": False,
        },
    )
    path = provenance.write_provenance_manifest(
        tmp_path,
        args,
        artifact_kind="test-artifact",
        extra={"global_step": 7},
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["artifact_kind"] == "test-artifact"
    assert record["artifact"] == {"global_step": 7}
    assert record["model"]["resolved_revision"] == "a" * 40
    assert record["dataset"]["resolved_revision"] == "b" * 40
    assert record["loss_artifact"]["sha256"] == "c" * 64
    assert len(record["yeto_source_sha256"]) == 64


def test_programmatic_manifest_marks_missing_revision_unattested(tmp_path):
    path = provenance.write_provenance_manifest(
        tmp_path,
        SimpleNamespace(
            model="org/model",
            model_revision=None,
            data=None,
            trust_remote_code=False,
            source_sha256=None,
        ),
        artifact_kind="programmatic-test",
    )

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["attestation_status"] == "unattested"
    assert record["model"] == {
        "source": "unattested",
        "requested_identifier": "org/model",
        "resolved_identifier": "org/model",
        "requested_revision": None,
        "resolved_revision": None,
        "attestation_status": "unattested",
    }


def test_programmatic_diffusion_writer_never_claims_moving_revision(tmp_path):
    from yeto.diffusion.learner import (
        DIFFUSION_ADAPTER_METADATA_FILE,
        write_diffusion_adapter_metadata,
    )

    args = SimpleNamespace(
        model="org/model",
        model_revision="main",
        data=None,
        trust_remote_code=True,
        source_sha256=None,
        diffusion_adapter=None,
        tuning="full",
        loss_function="flow_matching",
    )
    write_diffusion_adapter_metadata(
        tmp_path,
        args,
        SimpleNamespace(components={}),
    )

    record = json.loads(
        (tmp_path / DIFFUSION_ADAPTER_METADATA_FILE).read_text(encoding="utf-8")
    )["provenance"]
    assert record["attestation_status"] == "unattested"
    assert record["model"]["requested_revision"] == "main"
    assert record["model"]["resolved_revision"] is None
    assert record["model"]["source"] == "unattested"


def test_public_cli_security_defaults_are_fail_closed():
    args = cli.parse_args(
        ["--gpu", "aws:1xa100", "--model", "org/model", "--data", "org/data"]
    )
    assert args.model_revision is None
    assert args.data_revision is None
    assert args.trust_remote_code is False
    assert args.allow_unsafe_pickled_loss is False


def test_production_tree_has_no_unsafe_torch_load_or_forced_remote_code():
    root = Path(__file__).parents[1] / "yeto"
    for path in root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "trust_remote_code=True" not in source, path
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "torch"
                and func.attr == "load"
            ):
                continue
            weights_only = next(
                (kw.value for kw in node.keywords if kw.arg == "weights_only"), None
            )
            assert isinstance(weights_only, ast.Constant) and weights_only.value is True, path


def test_parse_entrypoints_expose_revision_and_trust_controls():
    from yeto import learner
    from yeto.diffusion import learner as diffusion_learner
    from yeto.diffusion import sample
    from yeto.megatron import learner as megatron_learner
    from yeto.mlx import learner as mlx_learner

    causal = learner.parse_args(
        [
            "--model",
            "org/model",
            "--data",
            "org/data",
            "--syncer",
            "none",
            "--learner-id",
            "0",
            "--num-learners",
            "1",
        ]
    )
    diffusion = diffusion_learner.parse_args(
        [
            "--model",
            "org/model",
            "--data",
            "org/data",
            "--syncer",
            "none",
            "--learner-id",
            "0",
            "--num-learners",
            "1",
        ]
    )
    megatron = megatron_learner.parse_args(["--model", "org/model", "--data", "org/data"])
    mlx = mlx_learner.parse_args(
        ["--model", "org/model", "--data", "org/data", "--syncer", "none"]
    )
    sampler = sample.parse_args(
        ["--adapter-dir", "out", "--prompt", "p", "--output", "x.png"]
    )
    for parsed in (causal, diffusion, megatron, mlx, sampler):
        assert parsed.model_revision is None
        assert parsed.trust_remote_code is False
