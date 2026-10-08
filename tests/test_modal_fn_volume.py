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
