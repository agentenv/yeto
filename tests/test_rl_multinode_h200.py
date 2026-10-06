"""S11 H200 chain: 1 node x 8 H200, island allocated 4 GPUs (--rl-island-use-gpus-per-node 4), cfg resources-1x4-h200.json."""
import pytest

from test_rl_launcher_multinode import _GPU_DIR, _elastic_task, launcher


@pytest.mark.parametrize("cfg,ro,sb,shape", [("T2R1S1", 1, 1, (1, 2)), ("T2R2S0", 2, 0, (1, 2)), ("T1R3S0", 3, 0, (1, 1))])
def test_h200_single_node_alloc4(monkeypatch, tmp_path, cfg, ro, sb, shape):
    a, s, t = _elastic_task(monkeypatch, tmp_path, "nebius:8xh200", _GPU_DIR / "resources-1x4-h200.json", cfg,
                            rollout=ro, standby=sb, extra=("--rl-island-use-gpus-per-node", "4"))
    island = launcher.rl_island_spec(a, s)
    assert island.gpus_per_node == 4 and s.gpus_per_node == 8
    assert launcher.rl_trainer_shape(a, island) == shape
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3" in t.run
