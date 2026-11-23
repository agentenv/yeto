"""Dry-run / preflight checks found on an AgentENV end-to-end run: task images on a
registry AgentENV may not pull from, the cold-start disk default and its refusal hint,
and the memory estimate of a small buffer trainer with recompute and optimizer offload.
No network, no machine."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from tests.test_agentenv_run_up import _spec as _agentenv_spec
from tests.test_agentenv_secure import fake_e2b  # noqa: F401  (fixture: the SDK stubbed)
from yeto import memory_estimate as me
from yeto.launch import agentic_setup as su
from yeto.rl.harness.harbor import agentenv_client as ac


def _tasks(tmp_path, images):
    root = tmp_path / "image-tasks"
    for i, image in enumerate(images):
        d = root / f"t{i}"
        d.mkdir(parents=True)
        body = f'docker_image = "{image}"\n' if image is not None else ""
        d.joinpath("task.toml").write_text(f'[task]\nname = "t{i}"\n\n[environment]\n{body}cpus = 1\n')
    return root


def _setup(tmp_path, images):
    spec = _agentenv_spec(tmp_path)
    spec["agentic"]["setup"]["tasks"] = {"source": str(_tasks(tmp_path, images))}
    return spec


@pytest.mark.parametrize("image, concern", [
    ("10.0.0.5:5000/library/task:1", "private address"),
    ("localhost:5000/task:1", "loopback"),
    ("registry:5000/task:1", "single-label"),
    ("52.1.2.3:5000/task:1", "IP address"),
    ("reg.example.com:5000/task:1", "port 5000"),
])
def test_a_registry_agentenv_may_not_pull_from_warns_with_the_fix(tmp_path, image, concern):
    lines, problems = su.dry_run_report(_setup(tmp_path, [image, image]))
    assert problems == []
    warning = next(ln for ln in lines if "WARNING" in ln and "task images on registry" in ln)
    assert concern in warning and "2 task(s)" in warning
    assert "https" in warning and "regctl registry set" in warning and "--tls disabled" in warning


@pytest.mark.parametrize("image", ["ghcr.io/org/task@sha256:" + "a" * 64, "reg.example.com/task:1", "ubuntu:22.04"])
def test_a_public_https_registry_passes_quietly(tmp_path, image):
    lines, problems = su.dry_run_report(_setup(tmp_path, [image]))
    assert problems == [] and not any("task images on registry" in ln for ln in lines)


def test_a_url_scheme_is_a_problem(tmp_path):
    lines, problems = su.dry_run_report(_setup(tmp_path, ["http://10.0.0.5:5000/task:1", None]))
    assert any("URL scheme" in p and "https" in p for p in problems)
    # the node setup refuses the same (local_problems runs before any step)
    other = tmp_path / "b"
    other.mkdir()
    assert any("URL scheme" in p for p in su.local_problems(su.parse(_setup(other, ["http://r/t:1"]))))


def test_image_registry_parsing():
    assert su.image_registry("ttl.sh/x@sha256:1") == "ttl.sh"
    assert su.image_registry("library/ubuntu:22.04") == "docker.io"
    assert su.image_registry("[fd00::1]:5000/a") == "[fd00::1]:5000"
    assert "private" in su._registry_concern("[fd00::1]:5000")


# -- the cold-start disk ---------------------------------------------------------------

def test_the_cold_start_disk_default_is_64_gib(monkeypatch, fake_e2b):  # noqa: F811
    assert ac.DEFAULT_DISK_SIZE_MB == 65536
    sent = {}

    def post(url, *, json, headers, timeout, trust_env):
        sent.update(json)
        return httpx.Response(201, json={"sandboxID": "sb-1", "envdAccessToken": "t"},
                              request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.delenv(ac.DISK_SIZE_MB_ENV, raising=False)
    client = ac.SdkAgentenvClient(api_key="k", api_url="http://gw.test", sandbox_url="http://gw.test",
                                  cold_start=True)
    client._create_cold("img:1", 60, None, None)
    assert sent["diskSizeMB"] == 65536
    monkeypatch.setenv(ac.DISK_SIZE_MB_ENV, "131072")
    client._create_cold("img:1", 60, None, None)
    assert sent["diskSizeMB"] == 131072


@pytest.mark.parametrize("status, text, expect", [
    (500, "failed to allocate block device", "set YETO_AGENTENV_DISK_SIZE_MB lower"),
    (400, "disk size smaller than base virtual size; shrinking is disabled", "raise YETO_AGENTENV_DISK_SIZE_MB"),
])
def test_a_refused_cold_start_names_the_disk_and_the_variable(monkeypatch, fake_e2b, status, text, expect):  # noqa: F811
    def post(url, *, json, headers, timeout, trust_env):
        return httpx.Response(status, text=text, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setenv(ac.DISK_SIZE_MB_ENV, "131072")
    client = ac.SdkAgentenvClient(api_key="secret-key", api_url="http://gw.test", sandbox_url="http://gw.test",
                                  cold_start=True)
    with pytest.raises(ac.AgentenvError) as info:
        client._create_cold("img:1", 60, None, None)
    msg = str(info.value)
    assert f"HTTP {status}" in msg and "131072" in msg and expect in msg
    assert "secret-key" not in msg and text not in msg  # the gateway's text picks the hint, it is not echoed


def test_a_refused_key_gets_no_disk_hint(monkeypatch, fake_e2b):  # noqa: F811
    def post(url, *, json, headers, timeout, trust_env):
        return httpx.Response(401, text="unauthorized", request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    client = ac.SdkAgentenvClient(api_key="k", api_url="http://gw.test", sandbox_url="http://gw.test",
                                  cold_start=True)
    with pytest.raises(ac.AgentenvError) as info:
        client._create_cold("img:1", 60, None, None)
    assert "HTTP 401" in str(info.value) and "DISK_SIZE" not in str(info.value)


# -- the memory estimate of a buffer trainer ---------------------------------------------------

QWEN3_4B = {"num_hidden_layers": 36, "hidden_size": 2560, "intermediate_size": 9728, "vocab_size": 151936,
            "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "tie_word_embeddings": True}


def _trainer_args(**kw):
    base = dict(training_mode="rl", model="Qwen/Qwen3-4B-Instruct-2507", tuning="full", seq_len=32768,
                rollout_max_response_len=32768, tensor_parallel=1, pipeline_parallel=1, rl_rollout_gpus=0,
                sglang_mem_fraction_static=0.4, rl_activation_recompute=True, rl_optimizer_offload_fraction=1.0,
                rl_buffer_shards="q-0,q-1", rl_sample_role=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_a_4b_buffer_trainer_with_recompute_and_offload_fits_an_h100():
    """The dry run of a 4B buffer trainer at 32k (recompute, optimizer offload 1.0) gave
    410.9 GiB: recompute and the offload were ignored and a rollout engine was assumed."""
    arch = me.arch_from_hf_config(QWEN3_4B)
    spec = SimpleNamespace(gpu="H100", total_gpus=1)
    req = me.request_from_args(_trainer_args(), spec, arch)
    assert req.recompute and req.optimizer_offload_fraction == 1.0 and not req.colocated
    est = me.estimate(req)
    parts = est["parts_gib"]
    assert parts["weights"] == pytest.approx(2 * arch.params / me.GIB, rel=1e-3)
    assert parts["grads_optimizer"] == pytest.approx(4 * arch.params / me.GIB, rel=1e-3)  # fp32 grads only
    assert est["rollout_phase_gib"] == 0 and parts["inference"] == 0
    assert 40 < est["peak_gib"] < 0.9 * me.gpu_capacity_gib("H100")
    # the same island as the dry run computed it before: no recompute, no offload, an engine
    old = me.request_from_args(_trainer_args(rl_activation_recompute=False, rl_optimizer_offload_fraction=0.0,
                                             rl_buffer_shards=None), spec, arch)
    assert me.estimate(old)["peak_gib"] > 300


def test_a_trainer_role_flag_alone_also_drops_the_engine():
    arch = me.arch_from_hf_config(QWEN3_4B)
    req = me.request_from_args(_trainer_args(rl_buffer_shards=None, rl_sample_role="trainer"),
                               SimpleNamespace(gpu="H100", total_gpus=1), arch)
    assert not req.colocated
    both = me.request_from_args(_trainer_args(rl_buffer_shards=None, rl_sample_role="both"),
                                SimpleNamespace(gpu="H100", total_gpus=1), arch)
    assert both.colocated
