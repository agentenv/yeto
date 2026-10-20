"""S25 N7.6: verl/FSDP2 cut save + restore with the resume assertions. CPU only.

A verl round is ``K`` optimizer steps (PPO mini-batches) and one LR-scheduler
step, as in verl's ``update_actor``. Checked:

* FSDP2 over gloo, 2 ranks: "3 rounds, cut, fresh trainer, restore, 2 rounds"
  equals "5 rounds" bit for bit (parameters and Adam state on every rank);
* a restored Adam step reset to 0 on one rank is refused on every rank;
* single process through :class:`VerlCutTrainer`: the data watermark is compared
  with the cursor that landed, the first round must confirm it and advance every
  Adam step by ``K`` and the scheduler by 1; a cut without a resume section, a
  non-fresh trainer and a DP change are refused.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
from types import SimpleNamespace

import pytest
import torch

from yeto.rl.adapters.verl import cut as vcut
from yeto.rl.adapters.verl.resume_check import TrainHandles, handles_rank_state
from yeto.rl.engine.cut import (
    MANIFEST_NAME,
    AlgorithmIdentity,
    CutContext,
    CutError,
    CutProgress,
    RestoreExpectation,
    canonical_json,
    commit_manifest,
    load_manifest,
)
from yeto.rl.engine.resume_manifest import ResumeAssertionError

TOTAL, CUT, K = 5, 3, 2  # rounds, cut after round CUT, optimizer steps per round
ALGO = AlgorithmIdentity("a" * 64)
FINGERPRINT = "verl@test"
PROMPTS_PER_ROUND = 4


def _model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))


def _handles(model) -> TrainHandles:
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
    return TrainHandles(model, opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0 / (1 + s)))


def _round(h: TrainHandles, r: int, rank: int = 0) -> None:
    for mini in range(K):
        g = torch.Generator().manual_seed(1000 * rank + K * r + mini)
        x, y = torch.randn(4, 8, generator=g), torch.randn(4, 1, generator=g)
        h.optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.mse_loss(h.module(x), y).backward()
        h.optimizer.step()
    h.scheduler.step()


def _context(root, rounds=CUT, cut_id="cut-v") -> CutContext:
    return CutContext(
        root=str(root), cut_id=cut_id, backend_fingerprint=FINGERPRINT,
        progress=CutProgress(local_step=rounds, scheduler_samples=rounds, global_batch_size=1,
                             next_rollout_id=rounds, policy_version=rounds, policy_hash="h"),
        algorithm=ALGO, data={"sample_offset": PROMPTS_PER_ROUND * rounds},
        ledger={"carried_over": 0, "ready_unconsumed": 0}, outer={"settled": True})


def _expect(world=1, rounds=CUT) -> RestoreExpectation:
    return RestoreExpectation(algorithm=ALGO, layout=vcut.layout_for(world), backend_fingerprint=FINGERPRINT,
                              local_step=rounds, policy_version=rounds, epoch=1)


def _local(t):
    return t.to_local() if hasattr(t, "to_local") else t


def _assert_same_train_state(a: TrainHandles, b: TrainHandles) -> None:
    for (name, p), (_, q) in zip(a.module.named_parameters(), b.module.named_parameters()):
        assert torch.equal(_local(p), _local(q)), name
    for p, q in zip(a.module.parameters(), b.module.parameters()):
        for key in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(_local(a.optimizer.state[p][key]), _local(b.optimizer.state[q][key])), key
    assert a.scheduler.last_epoch == b.scheduler.last_epoch


def _reset_steps(h: TrainHandles) -> None:
    for state in h.optimizer.state.values():
        state["step"] = torch.zeros_like(state["step"])


# ---------------------------------------------------------------- FSDP2, 2 ranks


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _fsdp2_worker(rank: int, world: int, port: int, root: str) -> None:
    import torch.distributed as dist
    from torch.distributed.fsdp import fully_shard

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        def build() -> TrainHandles:
            model = _model()
            for layer in model:
                if isinstance(layer, torch.nn.Linear):
                    fully_shard(layer)
            fully_shard(model)
            return _handles(model)

        def gathered(value):
            out = [None] * world
            dist.all_gather_object(out, value)
            return out

        def trainer_over(h: TrainHandles, *, restore=vcut.restore_rank) -> vcut.VerlCutTrainer:
            # call_ranks: every rank runs fn on its own handles, all see every result
            return vcut.VerlCutTrainer(
                lambda fn, *args: gathered(fn(h, *args)), optimizer_steps_per_round=K,
                fns={"restore": lambda hh, d, f, c: restore(hh, directory=d, files=f, cut_id=c),
                     "resume": vcut.resume_rank, "layout": lambda hh: vcut.remote_rank_layout()})

        continuous = build()
        for r in range(TOTAL):
            _round(continuous, r, rank)

        first = build()
        for r in range(CUT):
            _round(first, r, rank)
        context = _context(root)
        results = gathered(vcut.save_rank(first, directory=os.path.join(root, context.cut_id),
                                          cut_id=context.cut_id))
        if rank == 0:
            commit_manifest(root, vcut.build_manifest(context, results, epoch=1))
        dist.barrier()

        resumed = build()
        trainer = trainer_over(resumed)
        trainer.restore_cut(context.cut_id, epoch=1, root=root, expect=_expect(world))
        trainer.confirm_resumed_data_cursor({"sample_offset": PROMPTS_PER_ROUND * CUT})
        _round(resumed, CUT, rank)
        trainer.check_first_step(1)
        for r in range(CUT + 1, TOTAL):
            _round(resumed, r, rank)
        _assert_same_train_state(continuous, resumed)

        def resetting_on_rank0(h, *, directory, files, cut_id):
            result = vcut.restore_rank(h, directory=directory, files=files, cut_id=cut_id)
            if rank == 0:  # the PPU fault, on one rank only
                _reset_steps(h)
                result["resume"] = handles_rank_state(h).to_dict()
            return result

        with pytest.raises(ResumeAssertionError, match=r"rank0/group0: Adam step 0 != expected 6 \(Adam step was reset\)"):
            trainer_over(build(), restore=resetting_on_rank0).restore_cut(
                context.cut_id, epoch=1, root=root, expect=_expect(world))
        open(os.path.join(root, f"ok{rank}"), "w").close()
    finally:
        dist.destroy_process_group()


def test_fsdp2_gloo_two_ranks_cut_resume_equals_continuous(tmp_path):
    import torch.multiprocessing as mp

    pytest.importorskip("torch.distributed.fsdp")
    if not torch.distributed.is_gloo_available():
        pytest.skip("gloo backend not built")
    world = 2
    mp.spawn(_fsdp2_worker, args=(world, _free_port(), str(tmp_path)), nprocs=world, join=True)
    assert sorted(p for p in os.listdir(tmp_path) if p.startswith("ok")) == [f"ok{r}" for r in range(world)]
    manifest = load_manifest(tmp_path, "cut-v")
    assert [f.path for f in manifest.files] == ["rank00000.pt", "rank00001.pt"]
    assert manifest.runtime["layout"] == {"world": 2, "dp": 2}
    assert manifest.resume["global_step"] == CUT and manifest.resume["data_watermark"] == PROMPTS_PER_ROUND * CUT


# ---------------------------------------------------------------- single process, VerlCutTrainer


def _single(tmp_path):
    """(cut written after CUT rounds, a fresh trainer over a fresh worker, the fresh handles)."""
    trained = _handles(_model())
    for r in range(CUT):
        _round(trained, r)
    worker = SimpleNamespace(module=trained.module, optimizer=trained.optimizer, lr_scheduler=trained.scheduler)
    vcut.VerlCutTrainer(lambda fn, *a: [fn(worker, *a)], optimizer_steps_per_round=K).save_cut(
        epoch=1, context=_context(tmp_path))
    fresh = _handles(_model())
    fresh_worker = SimpleNamespace(module=fresh.module, optimizer=fresh.optimizer, lr_scheduler=fresh.scheduler)
    return vcut.VerlCutTrainer(lambda fn, *a: [fn(fresh_worker, *a)], optimizer_steps_per_round=K), fresh


def test_single_rank_resume_checks_watermark_and_first_round(tmp_path):
    trainer, fresh = _single(tmp_path)
    trainer.restore_cut("cut-v", epoch=1, root=str(tmp_path), expect=_expect())
    with pytest.raises(ResumeAssertionError, match="data watermark 8 != saved 12"):
        trainer.confirm_resumed_data_cursor({"sample_offset": 8})
    _round(fresh, CUT)
    with pytest.raises(ResumeAssertionError, match="never compared"):
        trainer.check_first_step(1)
    trainer.confirm_resumed_data_cursor({"sample_offset": 12})
    trainer.check_first_step(1)  # Adam 6 -> 8, scheduler 3 -> 4
    trainer.check_first_step(1)  # nothing pending any more


def test_single_rank_first_round_must_advance_by_k(tmp_path):
    trainer, fresh = _single(tmp_path)
    trainer.restore_cut("cut-v", epoch=1, root=str(tmp_path), expect=_expect())
    trainer.confirm_resumed_data_cursor({"sample_offset": 12})
    fresh.scheduler.step()  # the scheduler moved, no optimizer step was taken
    with pytest.raises(ResumeAssertionError, match="Adam step 6 != expected 8"):
        trainer.check_first_step(1)


def test_single_rank_refusals(tmp_path):
    trainer, fresh = _single(tmp_path)
    with pytest.raises(CutError, match="parallel layout"):
        trainer.restore_cut("cut-v", epoch=1, root=str(tmp_path), expect=_expect(world=2))
    trainer.restore_cut("cut-v", epoch=1, root=str(tmp_path), expect=_expect())
    with pytest.raises(CutError, match="freshly built"):  # the optimizer now has state
        trainer.restore_cut("cut-v", epoch=1, root=str(tmp_path), expect=_expect())

    path = tmp_path / "cut-v" / MANIFEST_NAME
    raw = json.loads(path.read_text())
    raw.pop("resume")
    body = {k: v for k, v in raw.items() if k != "manifest_sha256"}
    raw["manifest_sha256"] = hashlib.sha256(canonical_json(body).encode()).hexdigest()
    path.write_text(json.dumps(raw))
    again, _ = _single(tmp_path / "other")
    with pytest.raises(CutError, match="no resume manifest"):
        again.restore_cut("cut-v", epoch=1, root=str(tmp_path), expect=_expect())


def test_cut_refuses_wrong_progress_units(tmp_path):
    h = _handles(_model())
    _round(h, 0)
    worker = SimpleNamespace(module=h.module, optimizer=h.optimizer, lr_scheduler=h.scheduler)
    context = _context(tmp_path)
    bad = CutContext(**{**context.__dict__, "progress": CutProgress(
        local_step=2, scheduler_samples=8, global_batch_size=4, next_rollout_id=2, policy_version=2,
        policy_hash="h")})
    with pytest.raises(CutError, match="global_batch_size must be 1"):
        vcut.VerlCutTrainer(lambda fn, *a: [fn(worker, *a)], optimizer_steps_per_round=K).save_cut(
            epoch=1, context=bad)
    assert not (tmp_path / "cut-v").exists()


def test_optimizer_steps_per_round_from_config():
    config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(actor={"ppo_epochs": 2, "ppo_mini_batch_size": 4}),
                             data=SimpleNamespace(train_batch_size=16))
    config.actor_rollout_ref.actor = _Cfg(config.actor_rollout_ref.actor)
    assert vcut.optimizer_steps_per_round(config) == 8
    config.data.train_batch_size = 10
    with pytest.raises(CutError, match="whole number"):
        vcut.optimizer_steps_per_round(config)


class _Cfg(dict):
    """Attribute + ``get`` access, like an omegaconf node."""

    __getattr__ = dict.__getitem__


# ---------------------------------------------------------------- N7.6c: digest over the stepped state


def test_restore_whose_optimizer_steps_a_stale_snapshot_is_refused(tmp_path, monkeypatch):
    """Inject the T-G3 bug shape: after the load the module holds the cut's weights, but the
    optimizer steps a snapshot taken when the fresh trainer was built. The restore digest
    reads the stepped tensors, so the restore is refused (a module-side digest would match)."""
    import torch.distributed.checkpoint.state_dict as dcp_sd

    real = dcp_sd.set_state_dict

    def load_then_step_stale(module, optimizer, **kwargs):
        stale = {id(p): p.detach().clone() for p in module.parameters()}  # build-time values
        real(module, optimizer, **kwargs)
        for group in optimizer.param_groups:
            for i, param in enumerate(group["params"]):
                copy = torch.nn.Parameter(stale[id(param)])
                optimizer.state[copy] = optimizer.state.pop(param)
                group["params"][i] = copy

    monkeypatch.setattr(dcp_sd, "set_state_dict", load_then_step_stale)
    trainer, fresh = _single(tmp_path)
    saved = torch.load(tmp_path / "cut-v" / vcut.rank_file(0), weights_only=True)["model"]
    with pytest.raises(ResumeAssertionError, match="loaded train state differs from the cut"):
        trainer.restore_cut("cut-v", epoch=1, root=str(tmp_path), expect=_expect())
    for name, p in fresh.module.named_parameters():  # the module copy alone looks restored
        assert torch.equal(p.detach(), saved[name]), name


def test_digest_reads_the_tensors_the_optimizer_steps():
    h = _handles(_model())
    _round(h, 0)
    before = vcut._digest(h)
    first = h.optimizer.param_groups[0]["params"][0]
    with torch.no_grad():
        first.add_(1.0)
    assert vcut._digest(h) != before
