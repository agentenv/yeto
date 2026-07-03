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
