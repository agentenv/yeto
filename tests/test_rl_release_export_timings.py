"""S27 G-CJ: the trainer ranks' export timings reach the release event. CPU only, no Ray."""

from __future__ import annotations

import numpy as np
import pytest

from yeto.rl.engine import release_delta as rd
from yeto.rl.engine.policy_release import DeltaChain, PolicyReleaseStore
from yeto.rl.engine.resume import CheckpointStore


def _words(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 1 << 16, size=n, dtype=np.uint16)


def test_export_profile_sums_the_ranks_reports():
    from yeto.rl.adapters.miles.release_plugin import export_profile

    assert export_profile(None) == {} and export_profile([{"writers": "all"}]) == {}
    reports = [{"export_s": 50.0, "save_s": 30.0, "save_bytes": 10_000_000_000},
               {"export_s": 55.5, "save_s": 4.0, "save_bytes": 5_000_000_000}, "not a report"]
    assert export_profile(reports) == {"rank_export_max_s": 55.5, "rank_save_max_s": 30.0,
                                       "rank_save_sum_s": 34.0, "rank_save_gb": 15.0}


def test_timed_saves_wraps_safetensors_and_restores_it(tmp_path):
    torch = pytest.importorskip("torch")
    import safetensors.torch as safetensors_torch

    from yeto.rl.adapters.miles.release_plugin import _timed_saves

    original = safetensors_torch.save_file
    with _timed_saves() as stats:
        from safetensors.torch import save_file  # what the bridge does when it writes a shard

        save_file({"w": torch.zeros(1000, dtype=torch.bfloat16)}, str(tmp_path / "a.safetensors"))
    assert safetensors_torch.save_file is original
    assert stats["save_files"] == 1 and stats["save_bytes"] == (tmp_path / "a.safetensors").stat().st_size
    assert stats["save_s"] >= 0


def test_the_release_event_carries_the_export_timings(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from yeto.rl.adapters.miles import sampler_island as si
    from yeto.rl.engine.policy_release import BackgroundReleaser

    monkeypatch.setattr(rd, "MIN_DELTA_BYTES", 1024)
    events = []

    def export(version, directory):
        (directory / "model.safetensors").write_bytes(_words(4000, version).tobytes())
        return [{"export_s": 2.0, "save_s": 1.5, "save_bytes": 8000}]

    class State:
        policy_version = 1

        @staticmethod
        def policy_tensor_hash():
            return "h1"

    publisher = si.ReleasingPublisher(
        SimpleNamespace(publish=lambda state: None), PolicyReleaseStore(CheckpointStore(tmp_path / "store")),
        export, read_position=lambda: 0, emit=lambda e, **f: events.append((e, f)),
        staging=tmp_path / "staging", releaser=BackgroundReleaser(),
        delta=DeltaChain(full_every=4, engine=rd.DeltaEngine(workers=2, chunk_words=4096)))
    publisher.publish(State())
    publisher.drain_releases()
    timings = [f for e, f in events if e == "rl_policy_released"][0]["release_timings"]
    assert timings["rank_export_max_s"] == 2.0 and timings["rank_save_gb"] == 0.0
    resident = publisher.delta.engine.resident  # the full release's digests filled it
    assert resident.path is not None and resident.size("model.safetensors") == 8000
