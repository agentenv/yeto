"""S25 N7.6: resume assertions (A-TRN report #8, a silently reset Adam step). CPU only.

* "train 3, save, resume, train 2" equals "train 5" bit for bit (plain torch and
  FSDP2 over gloo with two CPU ranks);
* a restored optimizer whose Adam step, scheduler or moments differ is refused,
  and a first step that does not advance the step by exactly one is refused;
* the Miles cut path and the verl worker binding raise on the same faults.
"""

from __future__ import annotations

import copy
import os
import socket
from types import SimpleNamespace

import pytest
import torch

from yeto.rl.engine.cut import CutError
from yeto.rl.engine.resume_manifest import (
    RESUME_MANIFEST_NAME,
    ResumeAssertionError,
    ResumeManifest,
    read_resume_manifest,
    require_first_step,
    require_restored,
    sample_names,
    torch_optimizer_state,
    write_resume_manifest,
)

TOTAL, CUT = 5, 3


# ---------------------------------------------------------------- tiny trainer


def _model() -> torch.nn.Module:
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Tanh(), torch.nn.Linear(16, 8),
                               torch.nn.Tanh(), torch.nn.Linear(8, 1))


def _optim(model):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0 / (1 + s))
    return opt, sched


def _batch(step: int, rank: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(1000 * rank + step)  # data watermark -> batch
    return torch.randn(4, 8, generator=g), torch.randn(4, 1, generator=g)


def _step(model, opt, sched, step: int, rank: int = 0) -> None:
    x, y = _batch(step, rank)
    opt.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(model(x), y).backward()
    opt.step()
    sched.step()


def _manifest(model, opt, sched, step: int, rank: str = "rank0") -> ResumeManifest:
    return ResumeManifest(global_step=step, data_watermark=4 * step, scheduler_increment=1,
                          ranks={rank: torch_optimizer_state(opt, model.named_parameters(), sched)})


def _save(directory, model, opt, sched, step: int) -> None:
    torch.save({"model": model.state_dict(), "optim": opt.state_dict(), "sched": sched.state_dict(),
                "step": step}, os.path.join(directory, "ckpt.pt"))
    write_resume_manifest(directory, _manifest(model, opt, sched, step))  # after the checkpoint


def _load(directory, *, tamper=None):
    model = _model()
    opt, sched = _optim(model)
    ckpt = torch.load(os.path.join(directory, "ckpt.pt"), weights_only=True)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optim"])
    sched.load_state_dict(ckpt["sched"])
    if tamper is not None:
        tamper(model, opt, sched)
    return model, opt, sched, int(ckpt["step"])


def _equal_state(a_model, a_opt, b_model, b_opt) -> None:
    for (name, p), (_, q) in zip(a_model.named_parameters(), b_model.named_parameters()):
        assert torch.equal(p, q), name
    for p, q in zip(a_model.parameters(), b_model.parameters()):
        sa, sb = a_opt.state[p], b_opt.state[q]
        assert set(sa) == set(sb)
        for key in sa:
            assert torch.equal(torch.as_tensor(sa[key]), torch.as_tensor(sb[key])), key


def _continuous():
    model = _model()
    opt, sched = _optim(model)
    for step in range(TOTAL):
        _step(model, opt, sched, step)
    return model, opt, sched


def _resumed(tmp_path, *, tamper=None):
    model = _model()
    opt, sched = _optim(model)
    for step in range(CUT):
        _step(model, opt, sched, step)
    _save(tmp_path, model, opt, sched, CUT)
    saved = read_resume_manifest(tmp_path)
    model, opt, sched, start = _load(tmp_path, tamper=tamper)
    require_restored(saved, _manifest(model, opt, sched, start))
    _step(model, opt, sched, start)
    require_first_step(saved, _manifest(model, opt, sched, start + 1))
    for step in range(start + 1, TOTAL):
        _step(model, opt, sched, step)
    return model, opt, sched


# ---------------------------------------------------------------- equivalence


def test_train3_save_resume2_equals_train5(tmp_path):
    a = _continuous()
    b = _resumed(tmp_path)
    _equal_state(a[0], a[1], b[0], b[1])
    assert a[2].last_epoch == b[2].last_epoch == TOTAL
    assert a[2].get_last_lr() == b[2].get_last_lr()
    assert all(int(s["step"]) == TOTAL for s in b[1].state.values())


def test_manifest_round_trips_and_samples_are_fixed(tmp_path):
    model, opt, sched = _continuous()
    manifest = _manifest(model, opt, sched, TOTAL)
    write_resume_manifest(tmp_path, manifest)
    assert read_resume_manifest(tmp_path) == manifest
    names = [n for n, _ in model.named_parameters()]
    assert sample_names(names, 4) == sample_names(reversed(names), 4)
    assert len(manifest.ranks["rank0"].moments) == 4
    assert manifest.ranks["rank0"].adam_steps["group0"].max == TOTAL


def test_missing_manifest_is_refused(tmp_path):
    with pytest.raises(ResumeAssertionError, match=RESUME_MANIFEST_NAME):
        read_resume_manifest(tmp_path)


# ---------------------------------------------------------------- simulated faults


def _reset_steps(model, opt, sched):  # what A-TRN report #8 describes on PPU
    for state in opt.state.values():
        state["step"] = torch.zeros_like(state["step"])


def _reset_one_step(model, opt, sched):
    first = next(iter(model.parameters()))
    opt.state[first]["step"] = torch.zeros_like(opt.state[first]["step"])


def _fresh_moments(model, opt, sched):
    for state in opt.state.values():
        state["exp_avg"].zero_()


def _reset_scheduler(model, opt, sched):
    sched.last_epoch = 0


@pytest.mark.parametrize("tamper, message", [
    (_reset_steps, "Adam step was reset"),
    (_reset_one_step, "Adam step differs across parameters"),
    (_fresh_moments, "exp_avg norm"),
    (_reset_scheduler, "scheduler was reset"),
])
def test_restore_faults_are_refused(tmp_path, tamper, message):
    with pytest.raises(ResumeAssertionError, match=message):
        _resumed(tmp_path, tamper=tamper)


def test_resume_assertion_is_a_cut_error():
    assert issubclass(ResumeAssertionError, CutError)


def test_first_step_that_does_not_advance_is_refused(tmp_path):
    model = _model()
    opt, sched = _optim(model)
    for step in range(CUT):
        _step(model, opt, sched, step)
    saved = _manifest(model, opt, sched, CUT)
    _step(model, opt, sched, CUT)
    _reset_steps(model, opt, sched)  # e.g. a lazy re-init on the first step after the load
    with pytest.raises(ResumeAssertionError, match="Adam step 0 != expected 4"):
        require_first_step(saved, _manifest(model, opt, sched, CUT + 1))


def test_first_step_lr_follows_the_schedule(tmp_path):
    model = _model()
    opt, sched = _optim(model)
    for step in range(CUT):
        _step(model, opt, sched, step)
    saved = _manifest(model, opt, sched, CUT)
    _step(model, opt, sched, CUT)
    now = _manifest(model, opt, sched, CUT + 1)
    require_first_step(saved, now, expected_lrs={"rank0": [1e-2 / (CUT + 2)]})
    with pytest.raises(ResumeAssertionError, match="scheduled"):
        require_first_step(saved, now, expected_lrs={"rank0": [1e-2]})


def test_data_watermark_and_global_step_are_checked():
    model, opt, sched = _continuous()
    saved = _manifest(model, opt, sched, TOTAL)
    moved = ResumeManifest(TOTAL, saved.data_watermark + 4, 1, saved.ranks)
    with pytest.raises(ResumeAssertionError, match="data watermark"):
        require_restored(saved, moved)
    with pytest.raises(ResumeAssertionError, match="global step"):
        require_restored(saved, ResumeManifest(0, saved.data_watermark, 1, saved.ranks))


# ---------------------------------------------------------------- verl / FSDP2 binding


def test_verl_worker_binding_refuses_a_reset_step(tmp_path):
    from yeto.rl.adapters.verl.resume_check import (
        require_worker_first_step,
        require_worker_restored,
        worker_manifest,
    )

    model = _model()
    opt, sched = _optim(model)
    for step in range(CUT):
        _step(model, opt, sched, step)
    worker = SimpleNamespace(actor_module_fsdp=model, actor_optimizer=opt, actor_lr_scheduler=sched)
    saved = worker_manifest(worker, rank=0, global_step=CUT, data_watermark=12)
    require_worker_restored(saved, worker, rank=0, global_step=CUT, data_watermark=12)
    _step(model, opt, sched, CUT)
    require_worker_first_step(saved, worker, rank=0, global_step=CUT + 1, data_watermark=16)
    _reset_steps(model, opt, sched)
    with pytest.raises(ResumeAssertionError, match="Adam step was reset"):
        require_worker_restored(saved, worker, rank=0, global_step=CUT, data_watermark=12)
    with pytest.raises(ResumeAssertionError, match="lacks"):
        worker_manifest(SimpleNamespace(), rank=0, global_step=0, data_watermark=0)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _fsdp2_worker(rank: int, world: int, port: int, out_dir: str) -> None:
    """Two CPU ranks, FSDP2 over gloo: continuous 5 steps vs 3 + save/resume + 2."""
    import torch.distributed as dist
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    from torch.distributed.fsdp import fully_shard

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        def build():
            model = _model()
            for layer in model:
                if isinstance(layer, torch.nn.Linear):
                    fully_shard(layer)
            fully_shard(model)
            return (model, *_optim(model))

        def run(model, opt, sched, steps):
            for step in steps:
                _step(model, opt, sched, step, rank)

        def label():
            return f"rank{rank}"

        a = build()
        run(*a, range(TOTAL))

        b = build()
        run(*b, range(CUT))
        model_sd, optim_sd = get_state_dict(b[0], b[1])
        saved = _manifest(b[0], b[1], b[2], CUT, label())
        snapshot = (copy.deepcopy(model_sd), copy.deepcopy(optim_sd), b[2].state_dict())

        c = build()
        set_state_dict(c[0], c[1], model_state_dict=snapshot[0], optim_state_dict=snapshot[1])
        c[2].load_state_dict(snapshot[2])
        require_restored(saved, _manifest(c[0], c[1], c[2], CUT, label()))
        run(*c, [CUT])
        require_first_step(saved, _manifest(c[0], c[1], c[2], CUT + 1, label()))
        run(*c, range(CUT + 1, TOTAL))
        for (name, p), (_, q) in zip(a[0].named_parameters(), c[0].named_parameters()):
            assert torch.equal(p.to_local(), q.to_local()), name
        for p, q in zip(a[0].parameters(), c[0].parameters()):
            for key in ("exp_avg", "exp_avg_sq", "step"):
                x, y = a[1].state[p][key], c[1].state[q][key]
                x = x.to_local() if hasattr(x, "to_local") else x
                y = y.to_local() if hasattr(y, "to_local") else y
                assert torch.equal(x, y), key

        _reset_steps(c[0], c[1], c[2])  # the PPU fault on this rank's shard
        try:
            require_restored(saved, _manifest(c[0], c[1], c[2], CUT, label()))
        except ResumeAssertionError as error:
            assert "Adam step was reset" in str(error)
        else:
            raise AssertionError("a reset Adam step was not caught")
        open(os.path.join(out_dir, f"ok{rank}"), "w").close()
    finally:
        dist.destroy_process_group()


def test_fsdp2_gloo_two_ranks_resume_equals_continuous(tmp_path):
    import torch.multiprocessing as mp

    pytest.importorskip("torch.distributed.fsdp")
    if not torch.distributed.is_gloo_available():
        pytest.skip("gloo backend not built")
    world = 2
    mp.spawn(_fsdp2_worker, args=(world, _free_port(), str(tmp_path)), nprocs=world, join=True)
    assert sorted(os.listdir(tmp_path)) == [f"ok{r}" for r in range(world)]


# ---------------------------------------------------------------- Miles cut path


def _miles():
    from tests.test_rl_trainer_cut import RankGroup, _context, _expect, _trained_rank, _trainer

    return RankGroup, _context, _expect, _trained_rank, _trainer


def test_miles_cut_records_and_checks_the_resume_manifest(tmp_path):
    from tests.rl_cut_fakes import GBS, make_rank, train_step

    RankGroup, _context, _expect, _trained_rank, _trainer = _miles()
    _trainer(RankGroup([_trained_rank()])).save_cut(epoch=1, context=_context(tmp_path))
    fresh = make_rank(9)
    trainer = _trainer(RankGroup([fresh]))
    manifest = trainer.restore_cut("cut-a", epoch=1, root=str(tmp_path), expect=_expect())
    resume = ResumeManifest.from_dict(manifest.resume)
    assert (resume.global_step, resume.data_watermark, resume.scheduler_increment) == (2, 4, GBS)
    state = next(iter(resume.ranks.values()))
    assert {s.max for s in state.adam_steps.values()} == {2} and state.moments
    train_step(fresh, torch.randn(3, 6))
    trainer._check_first_step_after_resume(1)  # step 2 -> 3, scheduler 2*GBS -> 3*GBS
    assert trainer._resume_reference is None


def test_miles_restore_refuses_a_reset_adam_step(tmp_path):
    from tests.rl_cut_fakes import make_rank

    RankGroup, _context, _expect, _trained_rank, _trainer = _miles()
    _trainer(RankGroup([_trained_rank()])).save_cut(epoch=1, context=_context(tmp_path))
    fresh = make_rank(9)
    real_load = fresh._yeto_cut_backend.load_optimizer

    def resetting(optimizer, named, merged):
        real_load(optimizer, named, merged)
        for state in optimizer.state.values():
            state["step"] = torch.zeros_like(state["step"])

    fresh._yeto_cut_backend.load_optimizer = resetting
    with pytest.raises(CutError, match="Adam step was reset"):
        _trainer(RankGroup([fresh])).restore_cut("cut-a", epoch=1, root=str(tmp_path), expect=_expect())


def test_miles_first_step_without_advance_is_refused(tmp_path):
    from tests.rl_cut_fakes import make_rank

    RankGroup, _context, _expect, _trained_rank, _trainer = _miles()
    _trainer(RankGroup([_trained_rank()])).save_cut(epoch=1, context=_context(tmp_path))
    fresh = make_rank(9)
    trainer = _trainer(RankGroup([fresh]))
    trainer.restore_cut("cut-a", epoch=1, root=str(tmp_path), expect=_expect())
    with pytest.raises(ResumeAssertionError, match="Adam step 2 != expected 3"):
        trainer._check_first_step_after_resume(1)  # no optimizer step was taken


def test_miles_cut_without_resume_section_is_refused(tmp_path):
    """A cut written before N7.6 (no resume section, hash still valid) is not resumed."""
    import hashlib
    import json

    from tests.rl_cut_fakes import make_rank
    from yeto.rl.engine.cut import MANIFEST_NAME, canonical_json

    RankGroup, _context, _expect, _trained_rank, _trainer = _miles()
    _trainer(RankGroup([_trained_rank()])).save_cut(epoch=1, context=_context(tmp_path))
    path = tmp_path / "cut-a" / MANIFEST_NAME
    raw = json.loads(path.read_text())
    raw.pop("resume")
    body = {k: v for k, v in raw.items() if k != "manifest_sha256"}
    raw["manifest_sha256"] = hashlib.sha256(canonical_json(body).encode()).hexdigest()
    path.write_text(json.dumps(raw))
    with pytest.raises(CutError, match="no resume manifest"):
        _trainer(RankGroup([make_rank(9)])).restore_cut("cut-a", epoch=1, root=str(tmp_path), expect=_expect())
