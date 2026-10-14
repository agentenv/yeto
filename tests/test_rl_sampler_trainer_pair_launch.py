"""L5.4 launch composition: one launch of a sampler island and a trainer island
over one checkpoint store (``--rl-island-sample-role``), the trainer island held
back for the head's scheduler. Dry-run / plan level only: no cloud, no Ray."""

from __future__ import annotations

import shlex
import threading
import types

import pytest

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401

PAIR_GPU = "modal:1xh100,modal:1xh100"
STORE = "modal-volume://yeto-l54/run-a"
SPEC = "examples/rl_algorithms/decoupled-ppo-c2.json"
PAIR = ("--rl-single-island-no-sync", "--tuning", "full",
        "--rl-island-sample-role", "0:sampler", "--rl-island-sample-role", "1:trainer",
        "--rl-max-policy-age", "2", "--rl-algorithm-spec", SPEC,
        "--rl-checkpoint-store", STORE, "--rl-cut-every", "1",
        "--rl-sampler-max-backlog-rounds", "3", "--rl-sample-wait-s", "900",
        "--rl-trainer-scheduler", "s1", "--rl-trainer-island", "1",
        "--cluster-prefix", "l54-pair")


def _args(extra=(), gpu=PAIR_GPU, base=PAIR):
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", tuple(base) + tuple(extra), gpu=gpu)
    args.controller = "local"
    _prepare_rl_args(args)
    return args


def _plan(extra=()):
    from yeto.launcher import dry_run_plan

    args = _args(extra)
    return args, dry_run_plan(args)


def _island_tokens(command: str) -> list[str]:
    tail = command.split("-m yeto.rl.adapters.miles.island_entry", 1)[1]
    return [{"$LEARNER_ID": "0"}.get(t, t) for t in shlex.split(tail)]


def _flag(tokens: list[str], flag: str) -> str:
    return tokens[tokens.index(flag) + 1]


def test_pair_plan_has_no_syncer_and_the_trainer_island_is_held():
    _, plan = _plan()
    assert (plan["syncer"], plan["outer_sync"], plan["role_syncers"]) == (None, False, [])
    assert plan["trainer_scheduler"] == {"mode": "s1", "island": 1, "interval_s": 30.0}
    sampler, trainer = plan["island_requests"]
    assert (sampler["sample_role"], trainer["sample_role"]) == ("sampler", "trainer")
    assert "held_by_trainer_scheduler" not in sampler
    assert trainer["held_by_trainer_scheduler"] is True


def test_pair_island_commands_carry_their_own_role_and_one_learner():
    _, plan = _plan()
    sampler, trainer = (_island_tokens(i["learner_command"]) for i in plan["island_requests"])
    for tokens in (sampler, trainer):
        assert "--rl-single-island-no-sync" in tokens and "--syncer" not in tokens
        assert _flag(tokens, "--learner-id") == "0"
        assert "--num-learners" not in tokens  # one learner: no sweep roster
        assert _flag(tokens, "--parameter-mode") == "full" and "--lora-r" not in tokens
        assert _flag(tokens, "--rl-resume-store") == "/root/yeto-checkpoint-store/run-a"
        assert _flag(tokens, "--rl-max-policy-age") == "2"
    assert _flag(sampler, "--rl-sample-role") == "sampler"
    assert _flag(sampler, "--rl-sampler-max-backlog-rounds") == "3"
    assert "--rl-sample-wait-s" not in sampler
    assert _flag(trainer, "--rl-sample-role") == "trainer"
    assert _flag(trainer, "--rl-sample-wait-s") == "900.0"
    assert "--rl-sampler-max-backlog-rounds" not in trainer
    assert _flag(trainer, "--rl-cut-every") == "1"


def test_pair_modal_islands_keep_their_island_index_but_run_as_learner_zero():
    """Function name, tape subdir: island index. LEARNER_ID / NUM_LEARNERS: 0 / 1."""
    from yeto.gpu_spec import parse_gpu_spec
    from yeto.launch.modal_islands import build_modal_island_config
    from yeto.launcher import _island_task, make_miles_island_task

    args = _args(("--modal-tape-volume", "tapes"))
    specs = parse_gpu_spec(args.gpu)
    for m, spec in enumerate(specs):
        task = _island_task(make_miles_island_task, args, spec, m, 2, "none", {})
        cfg = build_modal_island_config(args, spec, m, task, "none")
        assert cfg.learner_id == m and cfg.tape_subdir.endswith(f"/l{m}")
        assert cfg.envs["LEARNER_ID"] == "0"
        assert cfg.checkpoint_store_volume_name == "yeto-l54"  # one store for both


def test_pair_island_commands_pass_the_island_parser_and_no_sync_checks():
    from yeto.rl.adapters.miles import island_entry
    from yeto.rl.engine import sample_roles

    _, plan = _plan()
    for island, role in zip(plan["island_requests"], ("sampler", "trainer")):
        parsed = island_entry.parse_args(_island_tokens(island["learner_command"]))
        assert (parsed.parameter_mode, parsed.rl_sample_role) == ("full", role)
        island_entry._check_single_island_no_sync(parsed)
        assert island_entry._full_parameter_mode(parsed, parsed.sync_preset) is True
        config = sample_roles.resolve(parsed, groups_per_round=parsed.groups_per_round)
        assert (config.role, config.store) == (role, "/root/yeto-checkpoint-store/run-a")


def test_full_parameter_without_a_buffer_role_is_still_refused_on_the_island():
    from yeto.rl.adapters.miles import island_entry

    args = types.SimpleNamespace(parameter_mode="full", rl_single_island_no_sync=True,
                                 rl_sample_role="both")
    with pytest.raises(ValueError, match="full-parameter sync preset"):
        island_entry._full_parameter_mode(args, "strict-avg")


def _replace(base, old: tuple, new: tuple) -> list[str]:
    """``base`` with the consecutive tokens ``old`` replaced by ``new``."""
    base = list(base)
    for at in range(len(base) - len(old) + 1):
        if tuple(base[at:at + len(old)]) == old:
            return base[:at] + list(new) + base[at + len(old):]
    raise AssertionError(f"{old} not in {base}")


NO_SCHEDULER = _replace(PAIR, ("--rl-trainer-scheduler", "s1", "--rl-trainer-island", "1"), ())


@pytest.mark.parametrize("old, new, message", [
    (("--rl-single-island-no-sync",), (), "no-sync|single-island"),
    (("--tuning", "full"), ("--tuning", "full", "--rl-sample-role", "trainer"),
     "replaces --rl-sample-role"),
    (("1:trainer",), ("1:sampler",), "one sampler and one trainer"),
    (("--rl-max-policy-age", "2"), ("--rl-max-policy-age", "0"), "max-policy-age"),
    (("--rl-checkpoint-store", STORE, "--rl-cut-every", "1"), (), "checkpoint-store"),
])
def test_pair_refusals(old, new, message):
    with pytest.raises((ValueError, SystemExit), match=message):
        _args(base=_replace(NO_SCHEDULER, old, new))


def test_pair_needs_two_islands():
    with pytest.raises(ValueError, match="island"):
        _args(gpu="modal:1xh100")


def test_tuning_full_outside_buffer_islands_is_refused():
    with pytest.raises(ValueError, match="requires --tuning lora"):
        _args(("--tuning", "full"), gpu="modal:1xh100", base=())


def test_held_trainer_drains_the_buffer_after_the_fleet_ended(tmp_path, monkeypatch):
    """When run() returns (the sampler ended), the loop sees data_ended and keeps
    ticking until it finished; then run_with_trainer_loop returns."""
    from yeto.launch import trainer_loop as launch_loop

    monkeypatch.setenv("YETO_RUNS_DIR", str(tmp_path / "runs"))
    seen: list[bool] = []
    finished = threading.Event()

    class _Loop:
        finished = False

        def restore(self, records):
            return False

        def tick(self):
            ended = binding_box["binding"].data_ended()
            seen.append(ended)
            if ended:
                self.finished = True
                finished.set()

        def close(self, reason):
            pass

    binding_box: dict = {}

    def fake_build(args, binding, journal):
        binding_box["binding"] = binding
        return _Loop()

    monkeypatch.setattr(launch_loop, "build", fake_build)
    args = types.SimpleNamespace(rl_trainer_scheduler="s1", rl_trainer_scheduler_interval=0.01,
                                 rl_trainer_drain_seconds=30.0, cluster_prefix="drain-a")
    binding = launch_loop.CloudTrainerBinding(lifecycle=lambda first: None, buffer=None, cuts=None)
    assert launch_loop.run_with_trainer_loop(args, lambda: "codes",
                                             factory=lambda a, i: binding) == "codes"
    assert finished.is_set() and seen[-1] is True
