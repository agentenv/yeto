"""S28 G-120: a Megatron buffer trainer publishes its policy release.

Run g120-6 (npu1, 24 x 910B3, 120B, ``--rl-sample-role trainer --parameter-mode full``)
trained its first step and then died publishing the release:

    VerlMegatronSyncError: the Megatron island exports HF files with Megatron-Bridge
    (megatron_ports.remote_hf_export) and has no peer recovery

Cause: the buffer trainer's release writer was ``hf_export.VerlHfWriter``, which pulls
whole HF tensors from the trainer group name by name (``group.export_parameters``).
``MegatronFullParameterGroup`` has no such method on purpose -- the Megatron HF
boundary exports whole files through Megatron-Bridge -- so every release raised.

The fix dispatches on ``plan["engine"]``: a Megatron island writes its release with
``megatron_ports.MegatronReleaseWriter``, which calls the trainer ranks collectively.
The ``ReleaseWeights`` contract is unchanged, so the release store, the staged
publication and the bf16 delta chain hold as they do on FSDP2.

What these tests cover, without a GPU or an NPU:

* the writer calls every rank, takes rank 0's record, and leaves the files the driver
  can see (Bridge writes on rank 0 only with ``distributed_save=False``);
* it refuses -- rather than publishing an empty release -- when no rank wrote, when
  more than one rank claims to have written, or when the driver cannot see the files
  (trainer rank 0 on another node's filesystem);
* ``release_weights_for`` picks this writer for a Megatron plan and keeps
  ``VerlHfWriter`` for an FSDP2 plan;
* a real :class:`PolicyReleaseStore` accepts what the writer produced: version,
  policy hash and weight files land in the record, and a second release against the
  first goes out as a bf16 delta that names its base.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yeto.rl.adapters.verl.megatron_full_parameter import VerlMegatronSyncError
from yeto.rl.adapters.verl.megatron_ports import MegatronReleaseWriter

WEIGHT_FILE = "model.safetensors"


class _RankIsland:
    """The part of ``FullyAsyncIsland`` the release writer uses, with a world of
    ``ranks`` whose rank 0 writes files the way Megatron-Bridge does."""

    def __init__(self, *, ranks: int = 4, writers=(0,), payload: bytes = b"w" * 32,
                 visible: bool = True, host: str = "n1"):
        self.ranks = ranks
        self.writers = set(writers)
        self.payload = payload
        self.visible = visible
        self.host = host
        self.calls: list = []

    def call_workers(self, fn, *args):
        self.calls.append((fn.__name__, args))
        directory = Path(args[0])
        results = []
        for rank in range(self.ranks):
            if rank not in self.writers:
                results.append(None)  # exhausted the generator, wrote nothing
                continue
            names = {}
            if self.visible:  # rank 0 shares the driver's filesystem
                directory.mkdir(parents=True, exist_ok=True)
                (directory / WEIGHT_FILE).write_bytes(self.payload)
                (directory / "config.json").write_text(json.dumps({"hidden_size": 8}))
                names = {p.name: p.stat().st_size for p in sorted(directory.iterdir()) if p.is_file()}
            else:  # the files landed on another node: the driver sees nothing
                names = {WEIGHT_FILE: len(self.payload)}
            results.append({"host": self.host, "directory": str(directory), "files": names})
        return results


def test_the_writer_takes_rank_zeros_files_and_calls_every_rank(tmp_path, capsys):
    island = _RankIsland()
    writer = MegatronReleaseWriter(island, tmp_path / "model")
    out = tmp_path / "release"

    writer(0, out)

    name, args = island.calls[0]
    assert name == "remote_release_hf_export"
    assert args == (str(out), str(tmp_path / "model"))  # the driver chooses the directory
    assert (out / WEIGHT_FILE).is_file()
    assert writer.last_record["files"][WEIGHT_FILE] == 32
    assert "2 file(s)" in capsys.readouterr().out


@pytest.mark.parametrize(
    "island, expected",
    [(_RankIsland(writers=()), "exactly one writing rank, got 0"),
     (_RankIsland(writers=(0, 1)), "exactly one writing rank, got 2"),
     (_RankIsland(payload=b"", visible=False), "cannot see")],
)
def test_the_writer_refuses_instead_of_publishing_an_empty_release(tmp_path, island, expected):
    with pytest.raises(VerlMegatronSyncError, match=expected):
        MegatronReleaseWriter(island, tmp_path / "model")(0, tmp_path / "release")


def test_a_rank_that_wrote_no_file_is_refused(tmp_path):
    """A rank that reports itself as the writer but produced nothing: the release would
    be an empty directory the sampler could not load."""

    class _Empty(_RankIsland):
        def call_workers(self, fn, *args):
            Path(args[0]).mkdir(parents=True, exist_ok=True)
            return [{"host": "n1", "directory": str(args[0]), "files": {}}]

    with pytest.raises(VerlMegatronSyncError, match="wrote no file"):
        MegatronReleaseWriter(_Empty(), tmp_path / "model")(0, tmp_path / "release")


def test_release_weights_for_picks_the_engines_writer(tmp_path, monkeypatch):
    """``plan["engine"]`` decides; an FSDP2 plan keeps the HF writer it always had.

    Only the dispatch is under test here, so the group is a stand-in: which group a
    Megatron island installs is
    ``test_s27_verl_megatron_buffer.test_a_megatron_buffer_trainer_installs_the_megatron_trainer_group``."""
    from yeto.rl.adapters.verl import buffer_ports
    from yeto.rl.adapters.verl.hf_export import VerlHfWriter
    from tests.test_s27_verl_megatron_buffer import _buffer_plan

    seen = {}

    def _capture(writer):
        seen["writer"] = writer
        return lambda state: None

    monkeypatch.setattr(buffer_ports, "full_release_weights", _capture)
    for engine, expected in (("megatron", MegatronReleaseWriter), ("fsdp2", VerlHfWriter)):
        island = _RankIsland()
        island.full_group = object()  # installed by full_parameter_ports on the real path
        plan = _buffer_plan(tmp_path, engine)
        buffer_ports.release_weights_for(island, plan)
        assert isinstance(seen["writer"], expected), engine

    # a LoRA trainer still releases the merged adapter, not the trainer's own weights
    assert buffer_ports.release_weights_for(_RankIsland(), {"parameter_mode": "lora"}) is None


def test_the_release_store_accepts_the_writers_output_and_deltas_against_it(tmp_path):
    """The real store: what the Megatron writer wrote is publishable, and the second
    release goes out as a bf16 delta naming the first (what the cross-cloud sampler
    rebuilds from)."""
    import torch
    from safetensors.torch import save_file

    from yeto.rl.engine.policy_release import DeltaChain, PolicyReleaseStore
    from yeto.rl.engine.resume import store_for

    store = PolicyReleaseStore(store_for(str(tmp_path / "store")))
    staging = tmp_path / "staging"
    staging.mkdir()

    def write_release(name: str, tensor) -> Path:
        """One release's weights, written the way MegatronReleaseWriter leaves them.

        The tensor is deliberately over 1 MiB: ``release_delta`` copies a safetensors
        file under ``MIN_DELTA_BYTES`` (1 MiB) whole, and the store falls back to a full
        upload when the delta is over ``MAX_DELTA_RATIO`` (0.5) of the weights."""
        island = _RankIsland()
        out = staging / name

        def call_workers(fn, *args):  # real safetensors, so the delta can be computed
            directory = Path(args[0])
            directory.mkdir(parents=True, exist_ok=True)
            save_file({"w": tensor}, str(directory / WEIGHT_FILE), metadata={"format": "pt"})
            (directory / "config.json").write_text(json.dumps({"hidden_size": int(tensor.numel())}))
            return [{"host": "n1", "directory": str(directory),
                     "files": {p.name: p.stat().st_size for p in sorted(directory.iterdir())}}]

        island.call_workers = call_workers
        MegatronReleaseWriter(island, tmp_path / "model")(0, out)
        return out

    base_weights = (torch.arange(1 << 20, dtype=torch.float32)
                    .reshape(1024, 1024).to(torch.bfloat16))  # 2 MiB of bf16
    stepped = base_weights.clone()
    stepped[0, 0] += 0.5  # what one optimizer step moves: most words are unchanged

    first = store.release_staged(1, "h" * 64, write_release("v1", base_weights),
                                 position=lambda: (7, {"modal-0": 7}), digest_full=True)
    assert first is not None
    assert (first.version, first.policy_hash, first.seq) == (1, "h" * 64, 1)
    assert WEIGHT_FILE in first.files and first.delta is None
    assert first.buffer_selected == 7
    assert set(first.file_sha256) >= {WEIGHT_FILE}  # the follower checks these before loading

    chain = DeltaChain(4)  # every 4th release is full; the ones between are deltas
    chain.adopt(first, write_release("base", base_weights))
    second = store.release_staged(2, "i" * 64, write_release("v2", stepped),
                                  position=lambda: (9, {"modal-0": 9}),
                                  delta_base=chain.base())
    assert second is not None
    assert second.version == 2 and second.seq == 2
    assert second.delta is not None, "the second release should be a delta against the first"
    assert second.delta["base_version"] == 1
