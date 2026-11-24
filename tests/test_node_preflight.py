"""The node preflight: one script per profile (the Daytona and SSH node checks joined),
judged from canned reports for CUDA, ROCm and Ascend nodes."""

from __future__ import annotations

from yeto.launch.node_inventory import NodePaths, NodeProfile
from yeto.node.preflight import Requirements, evaluate, parse_facts, preflight_script, run_preflight


def _profile(**over) -> NodeProfile:
    values = dict(name="n0", host="", device_family="cuda", cpu_arch="x86_64", card="H200", cards_per_node=8,
                  hbm_gib=141.0, host_memory_gib=0.0, container="docker", model_source="store",
                  paths=NodePaths(work="/mnt/nvme/work", models="/mnt/nvme/models"))
    values.update(over)
    return NodeProfile(**values)


def _report(facts: dict[str, str], cards: str = "", host_paths=(), py: str | None = None) -> str:
    lines = [f"YETO_NODE_FACT {k} {v}" for k, v in facts.items()]
    lines += [f"YETO_NODE_HOSTPATH {p}" for p in host_paths]
    if py:
        lines.append(f"YETO_NODE_PY {py}")
    return "\n".join([*lines, "YETO_NODE_CARDS_BEGIN", cards, "YETO_NODE_CARDS_END rc=0"])


DISK = {"free_kib:/mnt/nvme/work": str(500 << 20), "write:/mnt/nvme/work": "ok", "write:/mnt/nvme/models": "ok",
        "container_engine": "Docker version 29"}
H200 = "\n".join(f"{i}, NVIDIA H200, 143771" for i in range(8))


def _failed(checks):
    return sorted(c.name for c in checks if not c.ok and not c.warn)


def test_the_script_follows_the_family():
    cuda = preflight_script(_profile(), Requirements())
    assert "nvidia-smi" in cuda and "/dev/kfd" not in cuda and "docker --version" in cuda
    rocm = preflight_script(_profile(device_family="rocm", card="MI355X", hbm_gib=288.0), Requirements())
    assert "/dev/kfd" in rocm and "amd-smi" in rocm and "rocm-smi" in rocm
    npu = preflight_script(_profile(device_family="npu", card="910B4", hbm_gib=64.0, cpu_arch="aarch64",
                                    container="podman"), Requirements(sudo=True))
    assert "npu-smi info" in npu and "/dev/davinci_manager" in npu and "sudo -n podman --version" in npu


def test_a_cuda_node_with_its_cards_passes():
    found = parse_facts(_report({"arch": "x86_64", "gpu_dev": "8", **DISK}, H200))
    assert _failed(evaluate(found, _profile(), Requirements())) == []


def test_a_cuda_node_short_of_cards_or_disk_is_refused():
    found = parse_facts(_report({"arch": "aarch64", "gpu_dev": "4", "free_kib:/mnt/nvme/work": "1024",
                                 "write:/mnt/nvme/work": "ok", "write:/mnt/nvme/models": "ok",
                                 "container_engine": "absent"}, H200))
    assert _failed(evaluate(found, _profile(), Requirements())) == ["container_engine", "cpu_arch", "disk", "gpu_dev"]


def test_a_rocm_process_node_checks_kfd_and_the_interpreter():
    profile = _profile(device_family="rocm", card="MI355X", hbm_gib=288.0, container="process",
                       paths=NodePaths(work="/work", models="/models"))
    report = _report({"arch": "x86_64", "kfd": "noaccess", "dri": "8", "free_kib:/work": str(900 << 20),
                      "write:/work": "ok", "write:/models": "ok", "python": "ok"},
                     py='{"missing": ["aiter"], "count": 8, "arch": ["gfx942"]}')
    req = Requirements(python="/opt/venv/bin/python3", gpu_check=True)
    assert _failed(evaluate(parse_facts(report), profile, req)) == ["card_arch", "kfd", "modules"]


def test_an_ascend_node_with_busy_cards_is_refused_when_idle_is_required():
    profile = _profile(device_family="npu", card="910B4", hbm_gib=64.0, cpu_arch="aarch64", container="podman",
                       cards_per_node=2)
    cards = ("| Process id | Process name |\n| 1 0 | 4242 | python | 100 |\n")
    found = parse_facts(_report({"arch": "aarch64", "davinci": "2", **DISK}, cards, ["/dev/davinci_manager"]))
    checks = evaluate(found, profile, Requirements(require_idle=True))
    assert "idle" in _failed(checks) and found.host_paths == ["/dev/davinci_manager"]


def test_run_preflight_raises_with_every_failed_check():
    class Out:
        returncode, stderr = 0, ""
        stdout = _report({"arch": "x86_64", "gpu_dev": "0", **DISK}, H200)

    try:
        run_preflight(_profile(), Requirements(), run=lambda *a, **k: Out())
    except Exception as error:  # noqa: BLE001
        assert "gpu_dev" in str(error)
    else:
        raise AssertionError("refused expected")
