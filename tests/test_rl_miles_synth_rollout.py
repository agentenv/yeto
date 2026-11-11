import json

import numpy as np
import pytest

from yeto.rl.adapters.miles.synth_rollout import SynthSpec, synth_sample_dict, write_synth_dumps

pytest.importorskip("miles.utils.types")
torch = pytest.importorskip("torch")


def _spec(**kw):
    base = dict(lengths=[96, 64, 96], samples=2, prompt=16, topk=8, layers=5, router_topk=4, experts=32)
    base.update(kw)
    return SynthSpec.parse(json.dumps(base))


def test_sample_shapes_and_routing():
    spec = _spec()
    sample = synth_sample_dict(spec, 96, 1)
    assert len(sample["tokens"]) == 96 and sample["response_length"] == 80
    assert sample["reward"] == 0.0 and len(sample["loss_mask"]) == 80
    assert sample["rollout_topk_token_ids"].shape == (80, 8)
    assert (sample["rollout_topk_token_ids"][:, 0] == np.asarray(sample["tokens"][16:])).all()
    routed = sample["rollout_routed_experts"]
    assert routed.shape == (95, 5, 4) and routed.dtype == np.int32
    assert (routed[:, 0] == -1).all()
    moe = np.sort(routed[:, 1:], axis=-1)
    assert (np.diff(moe, axis=-1) > 0).all() and moe.max() < 32


def test_same_length_is_identical_and_files_are_reused(tmp_path):
    spec = _spec()
    a, b = synth_sample_dict(spec, 96, 0), synth_sample_dict(spec, 96, 0)
    assert a["tokens"] == b["tokens"]
    template = str(tmp_path / "{rollout_id}.pt")
    paths = write_synth_dumps(spec, template)
    assert len(paths) == 3
    first = torch.load(paths[0], weights_only=False)
    third = torch.load(paths[2], weights_only=False)
    assert first["samples"][1]["tokens"] == third["samples"][1]["tokens"]
    mtime = (tmp_path / "0.pt").stat().st_mtime_ns
    write_synth_dumps(spec, template)
    assert (tmp_path / "0.pt").stat().st_mtime_ns == mtime


@pytest.mark.parametrize("bad", [dict(lengths=[]), dict(lengths=[16]), dict(experts=30), dict(unknown=1)])
def test_bad_specs_are_refused(bad):
    with pytest.raises((ValueError, TypeError)):
        _spec(**bad)
