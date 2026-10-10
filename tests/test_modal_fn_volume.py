"""--modal-model-volume / --modal-memory-gib / --modal-cpu (S13 FN Modal smoke)."""
import json
import subprocess

from yeto.launcher import modal_volume_store_env
from yeto.modal_runner import HostMemSampler, ModalIslandConfig

REV = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
MODEL = "Qwen/Qwen3.8-Flash-Next"


def _cfg(**kw):
    base = dict(app_name="a", learner_id=0, training_mode="sft", gpu="h200", gpus_per_node=8,
                num_nodes=1, run_script="true")
    base.update(kw)
    return ModalIslandConfig(**base)


def test_overrides_and_defaults():
    assert _cfg().memory_request_mib == 32 * 8 * 1024 and _cfg().cpu_request == 32
    c = _cfg(memory_gib_override=1024, cpu_override=48)
    assert c.memory_request_mib == 1024 * 1024 and c.cpu_request == 48


def test_model_volume_pairing():
    import pytest
    with pytest.raises(ValueError):
        _cfg(model_volume_name="v").validate()
    with pytest.raises(ValueError):
        _cfg(model_volume_name="t", model_volume_mount="/m", tape_volume_name="t", tape_subdir="x").validate()


def test_store_env_hit_and_miss(tmp_path):
    snap = tmp_path / "hf" / "Qwen3.8-Flash-Next" / REV[:8]
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    cache = tmp_path / "hub"
    sh = modal_volume_store_env(MODEL, REV, mount=str(tmp_path), cache=str(cache))
    miss = subprocess.run(["bash", "-c", sh + '; echo "hit=${YETO_MODEL_STORE_HIT:-}"'], capture_output=True, text=True)
    assert "hit=\n" in miss.stdout and "WARNING" in miss.stderr
    (snap / "MANIFEST.json").write_text(json.dumps({"repo": MODEL, "revision": REV, "all_ok": True}))
    hit = subprocess.run(["bash", "-c", sh + '; echo "hit=${YETO_MODEL_STORE_HIT:-} $HF_HUB_CACHE"'],
                         capture_output=True, text=True)
    assert f"hit=1 {cache}" in hit.stdout
    link = cache / "models--Qwen--Qwen3.8-Flash-Next" / "snapshots" / REV / "config.json"
    assert link.read_text() == "{}"
    assert (cache / "models--Qwen--Qwen3.8-Flash-Next" / "refs" / "main").read_text().strip() == REV
    (snap / "MANIFEST.json").write_text(json.dumps({"repo": MODEL, "revision": "0" * 40, "all_ok": True}))
    bad = subprocess.run(["bash", "-c", sh + '; echo "hit=${YETO_MODEL_STORE_HIT:-}"'], capture_output=True, text=True)
    assert "hit=\n" in bad.stdout


def test_hostmem_sampler(tmp_path):
    s = HostMemSampler(1, str(tmp_path / "h.jsonl"))
    rec = s.sample()
    assert rec["event"] == "modal_host_sample" and rec["meminfo_used"] > 0 and s.peak_host > 0


def test_hostmem_cgroup_v2_and_v1_and_rss(tmp_path):
    """s19-compaction-g3-20261010b: cgroup_current/peak were null on Modal (gVisor);
    the sampler now finds the own cgroup (v2, else v1) and adds summed process RSS."""
    from yeto import modal_runner as mr

    v2 = tmp_path / "v2"; (v2 / "c").mkdir(parents=True)
    (v2 / "c" / "memory.current").write_text("1000\n")
    (v2 / "c" / "memory.peak").write_text("2000\n")
    (v2 / "c" / "memory.max").write_text("max\n")
    (v2 / "c" / "memory.events").write_text("low 0\nhigh 0\nmax 3\noom 1\noom_kill 1\n")
    pc = tmp_path / "pc2"; pc.write_text("0::/c\n")
    m = mr._cgroup_mem(str(v2), str(pc))
    assert m == {"cgroup_version": 2, "cgroup_current": 1000, "cgroup_peak": 2000, "cgroup_max": "max",
                 "cgroup_oom_kill": 1, "cgroup_oom": 1}
    v1 = tmp_path / "v1"; (v1 / "memory" / "x").mkdir(parents=True)
    d = v1 / "memory" / "x"
    (d / "memory.usage_in_bytes").write_text("300\n"); (d / "memory.max_usage_in_bytes").write_text("400\n")
    (d / "memory.limit_in_bytes").write_text("68719476736\n"); (d / "memory.failcnt").write_text("0\n")
    (d / "memory.oom_control").write_text("oom_kill_disable 0\nunder_oom 0\noom_kill 2\n")
    pc1 = tmp_path / "pc1"; pc1.write_text("4:memory:/x\n0::/\n")
    m1 = mr._cgroup_mem(str(v1), str(pc1))
    assert m1["cgroup_version"] == 1 and m1["cgroup_current"] == 300 and m1["cgroup_max"] == 68719476736
    assert m1["cgroup_oom_kill"] == 2
    none = mr._cgroup_mem(str(tmp_path / "missing"), str(tmp_path / "nope"))
    assert none["cgroup_version"] is None and none["cgroup_current"] is None
    proc = tmp_path / "proc"; (proc / "12").mkdir(parents=True); (proc / "self").mkdir()
    (proc / "12" / "status").write_text("Name:\tx\nVmRSS:\t   2048 kB\n")
    assert mr._proc_rss_sum(str(proc)) == 2048 * 1024
    rec = mr._host_mem_used_bytes()  # this machine: at least one memory source is non-null
    assert rec["proc_rss_sum"] and (rec["cgroup_current"] is not None or rec["meminfo_used"])
