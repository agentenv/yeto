from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_nava_learner_script_uses_configured_gpu_count():
    script = (ROOT / "scripts" / "nava_l4_smoke" / "02_run_learner.sh").read_text()

    assert '--nproc_per_node="${NAVA_GPUS_PER_NODE}"' in script
    assert "CUDA_VISIBLE_DEVICES=\"$(seq -s, 0" in script
    assert "global_batch=$(nava_global_batch_size)" in script


def test_nava_prepare_env_persists_batch_topology():
    script = (ROOT / "scripts" / "nava_l4_smoke" / "00_prepare.sh").read_text()
    common = (ROOT / "scripts" / "nava_l4_smoke" / "common.sh").read_text()

    assert 'NAVA_GPUS_PER_NODE:=${NAVA_GPUS_PER_NODE}' in script
    assert 'NAVA_BATCH_SIZE:=${NAVA_BATCH_SIZE}' in script
    assert 'NAVA_GRAD_ACCUM:=${NAVA_GRAD_ACCUM}' in script
    assert 'NAVA_SMOKE_BACKEND:=mock' in common


def test_nava_smoke_script_can_run_without_text_encoder_or_vae():
    prepare = (ROOT / "scripts" / "nava_l4_smoke" / "00_prepare.sh").read_text()
    learner = (ROOT / "scripts" / "nava_l4_smoke" / "02_run_learner.sh").read_text()

    assert "Small NAVA-compatible smoke pipeline with no text encoder or VAE" in prepare
    assert "pipeline_smoke.py" in learner


def test_nava_initial_sync_seeds_resume_versions():
    learner = (ROOT / "yeto" / "nava" / "learner.py").read_text()

    assert "return seen_versions, global_step" in learner
    assert "initial_versions, initial_global_step = wait_for_initial_sync" in learner
    assert "fragment_versions = initial_versions" in learner
    assert "global_step = initial_global_step" in learner
