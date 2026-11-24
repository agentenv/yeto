"""Engine images as capability records (unified-sampler-pool 2.2): the K3.1 pins in the
catalog, resolve per node profile, and the refusal of an incompatible combination before
anything is ordered."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yeto.launch import engine_images as images
from yeto.launch.node_inventory import NodeProfile, parse_node
from yeto.rl import sampler_pins as pins

K31_SPEC = Path(__file__).resolve().parents[1] / "examples" / "k31" / "sampler-spec.nc-mi355x-vllm.json"
K31_ROLLOUT = ("routed_experts", "top_logprobs", "fp32_lm_head")


def mi355x():
    return NodeProfile.for_card("nc-a", "MI355X", 8, hbm_gib=288)


def h200():
    return NodeProfile.for_card("sky-h200", "H200", 8, hbm_gib=141)


def b300():
    return NodeProfile.for_card("b300", "B300", 8, hbm_gib=288)


# -- the K3.1 pins moved into the catalog keep their exact references


def test_catalog_refs_are_the_pins_constants_byte_for_byte():
    by_name = {i.name: i for i in images.engine_images()}
    assert by_name["vllm-rocm-k31"].ref == pins.VLLM_SAMPLER_IMAGE_ROCM
    assert by_name["vllm-cuda-k31"].ref == pins.VLLM_SAMPLER_IMAGE_CUDA
    assert by_name["sglang-cuda-k31"].ref == pins.SGLANG_SAMPLER_IMAGE_CUDA
    assert by_name["sglang-rocm-k31"].ref == pins.SGLANG_SAMPLER_IMAGE_ROCM
    assert by_name["vllm-rocm-k31"].engine_version == pins.VLLM_SAMPLER_ROCM_VERSION
    assert by_name["vllm-cuda-k31"].engine_version == pins.VLLM_SAMPLER_CUDA_VERSION
    assert by_name["sglang-cuda-k31"].engine_version == pins.SGLANG_SAMPLER_ENGINE_VERSIONS["cuda"]
    for image in images.engine_images():
        assert image.family in images.DEVICE_FAMILIES and image.engine in images.ENGINES
        assert image.cpu_arch in images.CPU_ARCHES and image.recipes
        for recipe in image.recipes:
            assert set(recipe.loads) <= set(images.LOAD_MODES) and set(recipe.rollout) <= set(images.ROLLOUT_FIELDS)
            assert set(recipe.load_arches) <= set(recipe.loads)


def test_the_customer_k31_sampler_resolves_to_the_image_it_always_pulled():
    from yeto.launch.nc import ssh_sampler, tasks

    needs = images.spec_needs(json.loads(K31_SPEC.read_text()))
    assert needs == {"model": "kimi_k31", "engine_pref": "vllm", "load": "in_place", "precision": "mxfp4",
                     "rollout": K31_ROLLOUT}
    model, pref = needs.pop("model"), needs.pop("engine_pref")
    chosen = images.resolve(mi355x(), model, pref, **needs)
    assert chosen.ref == pins.VLLM_SAMPLER_IMAGE_ROCM
    assert chosen.image.pull_ref == ssh_sampler.IMAGE == tasks.SERVING_IMAGE  # the NC route's constants
    assert chosen.python == ssh_sampler.PYTHON and chosen.load == "in_place"
    assert chosen.recipe.verified_on == ("MI355X",)
    # no engine preference: the catalog order puts the same image first
    assert images.resolve(mi355x(), "moonshotai/Kimi-K3.1-Pro", load="in_place").ref == pins.VLLM_SAMPLER_IMAGE_ROCM


def test_default_images_are_unchanged():
    assert images.default_image("rocm", "sampler", "vllm") == pins.VLLM_SAMPLER_IMAGE_ROCM
    assert images.default_image("cuda", "sampler", "sglang") == pins.SGLANG_SAMPLER_IMAGE_CUDA
    assert images.canonical_device_family("npu") == "ascend" and images.canonical_device_family("gpu") == "cuda"
    with pytest.raises(ValueError, match="device family 'tpu' is unknown"):
        images.canonical_device_family("tpu")


# -- refusals


def test_k31_mxfp4_in_place_is_refused_on_a_cuda_h200_with_what_works_instead():
    with pytest.raises(images.ImageIncompatible) as caught:
        images.resolve(h200(), "kimi_k31", load="in_place", precision="mxfp4", rollout=K31_ROLLOUT)
    error = caught.value
    assert "cuda/sm90/x86_64" in str(error) and "kimi_k31 mxfp4 load in_place" in str(error)
    assert "vllm-rocm-k31" in error.reasons and "it is a rocm image" in error.reasons["vllm-rocm-k31"][0]
    assert any("load in_place needs sm100/sm103" in r for r in error.reasons["sglang-cuda-k31"])
    assert any("at least 2304 GiB HBM, the node has 1128" in r for r in error.reasons["sglang-cuda-k31"])
    assert any("runs on sm100/sm103" in r for r in error.reasons["vllm-cuda-k31"])
    assert "vllm (vllm-rocm-k31) on a rocm gfx950 node" in error.alternatives
    assert not any("on this node" in a for a in error.alternatives)  # 8 x H200 cannot hold K3.1 at all
    # a node that holds K3.1 but cannot reload it in place is told to restart
    with pytest.raises(images.ImageIncompatible) as caught:
        images.resolve(b300(), "kimi_k31", "vllm", load="in_place", rollout=K31_ROLLOUT)
    assert "vllm (vllm-cuda-k31) on this node with load restart" in caught.value.alternatives
    assert "sglang (sglang-cuda-k31) on this node with load in_place/restart" in caught.value.alternatives


def test_k31_in_place_on_blackwell_needs_sglang():
    with pytest.raises(images.ImageIncompatible, match="load in_place is not served"):
        images.resolve(b300(), "kimi_k31", "vllm", load="in_place")
    assert images.resolve(b300(), "kimi_k31", "vllm", load="restart").ref == pins.VLLM_SAMPLER_IMAGE_CUDA
    assert images.resolve(b300(), "kimi_k31", "sglang", load="in_place").ref == pins.SGLANG_SAMPLER_IMAGE_CUDA
    assert images.resolve(b300(), "kimi_k31").load == "restart"  # vLLM first; its K3.1 recipe restarts


def test_k31_is_refused_on_a_small_or_older_rocm_node():
    small = NodeProfile.for_card("mi300", "MI300X", 4, hbm_gib=192)
    with pytest.raises(images.ImageIncompatible) as caught:
        images.resolve(small, "kimi_k31", load="in_place")
    assert any("targets gfx950" in r for r in caught.value.reasons["vllm-rocm-k31"])


def test_ascend_has_no_pinned_image_but_takes_a_site_image():
    npu = NodeProfile.for_card("a", "910B4", 8, hbm_gib=64, cpu_arch="aarch64")
    with pytest.raises(images.ImageIncompatible, match="ascend"):
        images.resolve(npu, "Qwen/Qwen3-0.6B")
    site = images.resolve(npu, "Qwen/Qwen3-0.6B", image="quay.example/ascend/vllm-ascend@sha256:" + "e" * 64)
    assert site.image is None and site.recipe is None and site.ref.startswith("quay.example")


def test_a_named_catalog_image_is_checked_like_any_other():
    with pytest.raises(images.ImageIncompatible, match="it is a rocm image"):
        images.resolve(h200(), "Qwen/Qwen3-0.6B", image=pins.VLLM_SAMPLER_IMAGE_ROCM)
    ok = images.resolve(h200(), "Qwen/Qwen3-0.6B", image=pins.VLLM_SAMPLER_IMAGE_CUDA.removeprefix("docker:"))
    assert ok.image.name == "vllm-cuda-k31" and ok.recipe.model == images.ANY_MODEL and ok.load == "restart"


def test_image_mismatch_refuses_at_dry_run_and_keeps_its_old_answers():
    customer = pins.VLLM_SAMPLER_IMAGE_ROCM
    assert images.image_mismatch(customer, cpu_arch="x86_64", device_family="rocm") is None
    assert images.image_mismatch(customer, cpu_arch="x86_64", device_family="rocm", card_arch="gfx950",
                                 model="kimi_k31", load="in_place", precision="mxfp4",
                                 rollout=K31_ROLLOUT, hbm_gib_per_node=2304) is None
    why = images.image_mismatch(customer, cpu_arch="x86_64", device_family="cuda")
    assert why and "it is a rocm image, the node is cuda" in why
    why = images.image_mismatch(pins.VLLM_SAMPLER_IMAGE_CUDA, cpu_arch="x86_64", device_family="cuda",
                                card_arch="sm90", model="kimi_k31", load="in_place")
    assert why and "vllm-cuda-k31" in why
    # the catalog of built images: as before, and npu / ascend now spell one family
    ppu = "reg.example/yeto-ppu-train:1"
    assert images.image_mismatch(ppu, cpu_arch="x86_64", device_family="ppu") is None
    assert "the node is device_family npu" in images.image_mismatch(ppu, cpu_arch="x86_64", device_family="npu")
    assert "the node is aarch64" in images.image_mismatch(ppu, cpu_arch="aarch64", device_family="ppu")
    assert images.image_mismatch("site/image:1", cpu_arch="x86_64", device_family="cuda", model="kimi_k31") is None


def test_a_node_profile_carrying_a_catalog_image_is_held_to_it():
    node = parse_node({"name": "x", "host": "x", "device_family": "cuda", "cpu_arch": "x86_64", "card": "H200",
                       "cards_per_node": 8, "hbm_gib": 141, "host_memory_gib": 512,
                       "image": pins.SGLANG_SAMPLER_IMAGE_CUDA}, index=0)
    profile = NodeProfile.from_node(node)
    assert images.resolve(profile, "Qwen/Qwen3-0.6B", "vllm").image.name == "sglang-cuda-k31"  # its own wins
    with pytest.raises(images.ImageIncompatible, match="sglang-cuda-k31"):
        images.resolve(profile, "kimi_k31", load="restart")  # its own image cannot hold K3.1 on 8 x H200


# -- heterogeneous pools


def test_a_heterogeneous_pool_gets_one_image_per_node_and_one_rollout_config():
    nodes = [mi355x(), b300(), NodeProfile.for_card("b300-b", "B300", 8, hbm_gib=288)]
    chosen = images.resolve_pool(nodes, "kimi_k31", load="restart", precision="mxfp4", rollout=K31_ROLLOUT)
    assert chosen["nc-a"].ref == pins.VLLM_SAMPLER_IMAGE_ROCM
    assert chosen["b300"].ref == chosen["b300-b"].ref == pins.VLLM_SAMPLER_IMAGE_CUDA
    assert all(set(K31_ROLLOUT) <= set(r.recipe.rollout) for r in chosen.values())
    with pytest.raises(images.ImageIncompatible) as caught:  # in place: the CUDA nodes need SGLang
        images.resolve_pool(nodes, "kimi_k31", "vllm", load="in_place")
    assert set(caught.value.reasons) == {"b300", "b300-b"}
    mixed = images.resolve_pool(nodes, "kimi_k31", ("vllm", "sglang"), load="in_place")
    assert mixed["nc-a"].engine == "vllm" and mixed["b300"].engine == "sglang"


def test_model_keys_and_load_names():
    for name in ("moonshotai/Kimi-K3.1-Pro", "kimi-k31-pro", "KIMI_K31", "/models/kimi-k3.1-pro"):
        assert images.model_key(name) == "kimi_k31"
    assert images.model_key("Qwen/Qwen3-0.6B") == "qwen/qwen3-0.6b" and images.model_key(None) == images.ANY_MODEL
    assert images.canonical_load("hot") == "in_place" and images.canonical_load("in-place") == "in_place"
    with pytest.raises(ValueError, match="load mode"):
        images.canonical_load("teleport")
