import pytest

from yeto.gpu_spec import ClusterSpec, parse_gpu_spec


def test_single_node_entry():
    (c,) = parse_gpu_spec("aws:8xa100@us-east-2")
    assert c == ClusterSpec("aws", "us-east-2", 1, 8, "A100")
    assert c.accelerators == "A100:8"
    assert c.total_gpus == 8


def test_multi_node_entry():
    (c,) = parse_gpu_spec("aws:4x8xa100@us-east-2")
    assert c.num_nodes == 4
    assert c.gpus_per_node == 8
    assert c.total_gpus == 32


def test_multi_cluster():
    specs = parse_gpu_spec(
        "aws:8xa100@us-east-2,aws:8xa100@us-east-1,aws:8xa100@us-west-2"
    )
    assert [c.region for c in specs] == ["us-east-2", "us-east-1", "us-west-2"]


def test_no_region_and_case():
    (c,) = parse_gpu_spec("GCP:2x4xL4")
    assert c.cloud == "gcp"
    assert c.region is None
    assert c.accelerators == "L4:4"


@pytest.mark.parametrize("bad", ["", "aws:", "aws:8a100", "aws:8xfoo@x", "8xa100"])
def test_rejects_bad_entries(bad):
    with pytest.raises(ValueError):
        parse_gpu_spec(bad)


def test_parse_image_spec():
    import pytest

    from yeto.launcher import parse_image_spec

    assert parse_image_spec(None) is None
    assert parse_image_spec("skypilot:gpu-ubuntu-2204") == "skypilot:gpu-ubuntu-2204"
    assert parse_image_spec("ami-0abc") == "ami-0abc"
    assert parse_image_spec("us-east-2=ami-a,us-west-2=ami-b") == {
        "us-east-2": "ami-a",
        "us-west-2": "ami-b",
    }
    with pytest.raises(ValueError, match="region=image-id"):
        parse_image_spec("us-east-2=,broken")


def test_learner_image_selection_precedence(monkeypatch):
    from types import SimpleNamespace

    from yeto import launcher
    from yeto.gpu_spec import ClusterSpec

    b200 = ClusterSpec("aws", "us-east-2", 1, 8, "B200")
    a100 = ClusterSpec("aws", "us-east-2", 1, 8, "A100")
    monkeypatch.setitem(
        launcher.GPU_IMAGE_OVERRIDES, ("aws", "B200"), lambda region: f"ami-blackwell-{region}"
    )
    # Internal table applies for B200...
    args = SimpleNamespace(learner_image=None)
    assert launcher.learner_image_for(args, b200) == "ami-blackwell-us-east-2"
    # ...not for GPUs the provider default drives fine...
    assert launcher.learner_image_for(args, a100) is None
    # ...and an explicit flag always wins.
    args = SimpleNamespace(learner_image="ami-user")
    assert launcher.learner_image_for(args, b200) == "ami-user"
    # A failing resolver degrades to None (setup-time driver remediation).
    monkeypatch.setitem(launcher.GPU_IMAGE_OVERRIDES, ("aws", "B200"), lambda region: None)
    assert launcher.learner_image_for(SimpleNamespace(learner_image=None), b200) is None


def test_learner_task_mounts_hf_token_when_present(tmp_path, monkeypatch):
    """The launching machine's HF token rides onto every learner, and the
    setup/run shells copy it to wherever NVME_ENV points HF_HOME."""
    from yeto import launcher

    token = tmp_path / "token"
    token.write_text("hf_test")
    monkeypatch.setattr(
        launcher.os.path, "expanduser",
        lambda p: str(token) if p == launcher.HF_TOKEN_PATH else p,
    )
    assert launcher.HF_TOKEN_PATH == "~/.cache/huggingface/token"
    assert "cp -n ~/.cache/huggingface/token $HF_HOME/token" in launcher.HF_TOKEN_ENV
