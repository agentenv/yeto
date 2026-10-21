"""S26 E9.2: round cuts on the verl fully_async path (CPU part).

The data cursor counts prompts the rounds took from the queue; on resume the
rollouter skips them before it starts; the trainer port cuts and restores
through the trainer actor's worker group like the sync path."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests import test_rl_verl_cut as vc
from tests import test_rl_verl_fully_async_64b as fa
from yeto.rl.adapters.verl import fully_async_cut as fac


class FakeRollouter:
    """The two fields of verl's FullyAsyncRollouter the skip touches."""

    def __init__(self, prompts, epochs=2):
        self.prompts, self.epochs, self.global_steps = list(prompts), epochs, 1

    def _create_continuous_iterator(self):
        for epoch in range(self.epochs):
            for p in self.prompts:
                yield epoch, p


def test_skip_draws_through_the_same_iterator_and_counts_steps():
    r = FakeRollouter("abcd")
    assert fac.skip_rollouter_prompts(r, 6) == {"sample_offset": 6}
    assert list(r._create_continuous_iterator()) == [(1, "c"), (1, "d")]  # across the epoch end
    assert r.global_steps == 7
    with pytest.raises(RuntimeError, match="already skipped"):
        fac.skip_rollouter_prompts(r, 1)
    short = FakeRollouter("ab", epochs=1)
    fac.skip_rollouter_prompts(short, 3)
    with pytest.raises(RuntimeError, match="data ended after 2 of 3"):
        list(short._create_continuous_iterator())


def test_data_cursor_seek_rules():
    seeks = []
    cursor = fac.FullyAsyncDataCursor(lambda n: seeks.append(n) or {"sample_offset": n})
    assert cursor.seek({"sample_offset": 0}) == {"sample_offset": 0} and seeks == []
    assert cursor.seek({"sample_offset": 12}) == {"sample_offset": 12} and seeks == [12]
    with pytest.raises(ValueError, match="cannot move back"):
        cursor.seek({"sample_offset": 4})
    with pytest.raises(ValueError, match="only"):
        cursor.seek({"sample_offset": 12, "epoch": 0})
    cursor.started = True
    with pytest.raises(ValueError, match="before the rollouter starts"):
        cursor.seek({"sample_offset": 20})
    with pytest.raises(ValueError, match="no seek hook"):
        fac.FullyAsyncDataCursor(None).seek({"sample_offset": 1})


def test_optimizer_steps_per_round():
    cfg = {"actor_rollout_ref": {"actor": {"ppo_epochs": 2, "ppo_mini_batch_size": 4}}}
    assert fac.optimizer_steps_per_round(cfg, groups_per_round=8) == 4
    with pytest.raises(ValueError, match="whole number"):
        fac.optimizer_steps_per_round(cfg, groups_per_round=6)


def test_pool_cursor_counts_every_pulled_prompt_kept_or_dropped(tmp_path):
    queue = [fa._sample("a", 0, 0, [(0, 4)]), fa._sample("b", 0, 0, [(0, 3)]),
             fa._sample("c", 0, 1, [(0, 2), (1, 3)]), fa._sample("d", 1, 1, [(1, 4)]),
             fa._sample("e", 0, 0, [(0, 5)]),  # round 2: over age, dropped -- still consumed
             fa._sample("f", 1, 2, [(1, 1), (2, 2)]), fa._sample("g", 2, 2, [(2, 6)])]
    actor = fa.FakeTrainerActor(tmp_path / "rb", queue)
    driver, island, _tape, _started = fa._driver(tmp_path, actor)
    driver.run()
    assert driver.rollout.data_cursor() == {"sample_offset": 7}
    with pytest.raises(ValueError, match="before the rollouter starts"):
        driver.rollout.seek_data_cursor({"sample_offset": 9})


def test_pool_seek_before_the_start_skips_the_rollouter(tmp_path):
    from yeto.rl.adapters.verl.fully_async_ports import FullyAsyncRolloutPool

    rollouter = FakeRollouter("abcdef")
    events = []
    island = SimpleNamespace(required=2, rollouter_started=False,
                             seek_rollouter=lambda n: fac.skip_rollouter_prompts(rollouter, n),
                             emit=lambda event, **f: events.append((event, f)))
    pool = FullyAsyncRolloutPool(island)
    assert pool.seek_data_cursor({"sample_offset": 4}) == {"sample_offset": 4}
    assert [p for _, p in rollouter._create_continuous_iterator()][:2] == ["e", "f"]
    assert events == [("rl_verl_data_seek", {"data_cursor": {"sample_offset": 4}, "fully_async": True})]


def test_trainer_group_cuts_and_restores_through_the_worker_group(tmp_path):
    """FullyAsyncTrainerGroup's cut verbs reach the ranks through call_workers (the
    trainer actor's yeto_call_workers) and keep the sync path's checks."""
    from yeto.rl.adapters.verl.fully_async_ports import FullyAsyncTrainerGroup

    def group(handles):
        worker = SimpleNamespace(module=handles.module, optimizer=handles.optimizer,
                                 lr_scheduler=handles.scheduler)
        island = SimpleNamespace(call_workers=lambda fn, *a: [fn(worker, *a)], optimizer_steps_per_round=vc.K,
                                 emit=lambda *a, **k: None)
        return FullyAsyncTrainerGroup(island)

    trained = vc._handles(vc._model())
    for r in range(vc.CUT):
        vc._round(trained, r)
    assert group(trained).save_cut(epoch=1, context=vc._context(tmp_path)) == "cut-v"
    fresh = vc._handles(vc._model())
    port = group(fresh)
    assert port.layout() == vc.vcut.layout_for(1)
    port.restore_cut("cut-v", epoch=1, root=str(tmp_path), expect=vc._expect())
    port.confirm_resumed_data_cursor({"sample_offset": vc.PROMPTS_PER_ROUND * vc.CUT})
    vc._round(fresh, vc.CUT)
    events = []
    port._check_first_step_after_resume(vc.CUT, lambda event, **f: events.append(event))
    assert events == ["rl_resume_check"]
    vc._round(trained, vc.CUT)
    vc._assert_same_train_state(trained, fresh)


def test_trainer_group_without_cuts_refuses_to_cut():
    from yeto.rl.adapters.verl.fully_async_ports import FullyAsyncTrainerGroup

    port = FullyAsyncTrainerGroup(SimpleNamespace(call_workers=lambda fn, *a: [], optimizer_steps_per_round=None))
    with pytest.raises(RuntimeError, match="without round cuts"):
        port._cuts()


def test_a_fully_async_round_cut_takes_the_pool_with_a_policy_age(tmp_path):
    """S26 E9 G2 (e9-g2-b): with --rl-max-policy-age > 0 the cut asks the rollout
    pool for its in-flight work; the fully_async pool had no export, so every cut
    was refused (RebuildRefused). It now exports nothing (queued groups are
    generated again after a resume) and the cut context is built."""
    from yeto.rl.adapters.verl.fully_async_ports import FullyAsyncRolloutPool
    from yeto.rl.engine.algorithm import AlgorithmSpec
    from yeto.rl.engine.round_cut import CutSource

    island = SimpleNamespace(required=2, rollouter_started=True, seek_rollouter=None,
                             emit=lambda event, **f: None)
    pool = FullyAsyncRolloutPool(island)
    pool._cursor.consume(5)
    state = SimpleNamespace(policy_version=3, policy_tensor_hash=lambda: "h" * 64)
    driver = SimpleNamespace(published_state=state, published_version=3, local_step=3,
                             profile=SimpleNamespace(max_policy_age=2), at_safe_point=True, expected_token="yeto:3:h")
    source = CutSource(driver=driver, trainer=None, rollout=pool, ledger=None, algorithm=AlgorithmSpec(),
                       backend_fingerprint="sha256:" + "0" * 64, cut_root="", global_batch_size=1)
    context = source.context("r000003-x")
    assert context.data == {"sample_offset": 5, "buffer_length": 0}
    assert tuple(context.in_flight) == () and context.ledger["max_policy_age"] == 2
