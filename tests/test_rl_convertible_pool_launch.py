"""L7.5 launch composition: a convertible pool (every island declared
``--rl-node-class islandM=convertible``). Dry-run / plan level and the head
binding over an in-memory store: no cloud, no Ray."""

from __future__ import annotations

import json
import shlex
import types

import pytest

from test_rl_algorithm_provenance import _fake_sky, _launcher_args, _no_modal_listing  # noqa: F401
from test_rl_convertible_remote import MemFiles

POOL_GPU = "modal:1xh100,modal:1xh100"
STORE = "modal-volume://yeto-conv/run-a"
SPEC = "examples/rl_algorithms/decoupled-ppo-c2.json"
POOL = ("--rl-single-island-no-sync", "--tuning", "lora", "--on-demand",
        "--rl-node-class", "island0=convertible", "--rl-node-class", "island1=convertible",
        "--rl-max-policy-age", "2", "--rl-algorithm-spec", SPEC,
        "--rl-checkpoint-store", STORE, "--rl-cut-every", "1", "--rl-sample-wait-s", "300",
        "--cluster-prefix", "conv-pool")


def _args(extra=(), gpu=POOL_GPU, base=POOL):
    from yeto.launcher import _prepare_rl_args

    args = _launcher_args("ports", tuple(base) + tuple(extra), gpu=gpu)
    args.controller = "local"
    _prepare_rl_args(args)
    return args


def _island_tokens(command: str) -> list[str]:
    tail = command.split("-m yeto.rl.adapters.miles.island_entry", 1)[1]
    return [{"$LEARNER_ID": "0"}.get(t, t) for t in shlex.split(tail)]


def _flag(tokens, flag):
    return tokens[tokens.index(flag) + 1]


def test_declaring_convertible_islands_turns_the_scheduler_on():
    from yeto.launch import trainer_loop
    from yeto.launcher import dry_run_plan

    args = _args()
    assert trainer_loop.enabled(args) and trainer_loop.convertible_pool(args)
    assert trainer_loop.held_island(args) is None  # nothing held back: a node converts
    assert trainer_loop.start_seconds(args) == 60.0  # T_s default = the conversion time
    plan = dry_run_plan(args)
    assert plan["trainer_scheduler"] == {"mode": "s1", "island": None, "interval_s": 30.0,
                                         "convertible_pool": True}
    assert [i["sample_role"] for i in plan["island_requests"]] == ["convertible"] * 2
    assert not any(i.get("held_by_trainer_scheduler") for i in plan["island_requests"])


def test_pool_island_commands_carry_role_and_node_name_and_parse_on_the_island():
    from yeto.launcher import dry_run_plan
    from yeto.rl.adapters.miles import island_entry
    from yeto.rl.adapters.miles.buffer_rollout import SHARD_ENV
    from yeto.rl.engine import sample_roles

    plan = dry_run_plan(_args())
    for m, island in enumerate(plan["island_requests"]):
        tokens = _island_tokens(island["learner_command"])
        assert _flag(tokens, "--rl-sample-role") == "convertible"
        assert _flag(tokens, "--rl-convertible-node") == f"island{m}"
        assert _flag(tokens, "--learner-id") == "0" and "--num-learners" not in tokens
        parsed = island_entry.parse_args(tokens)
        island_entry._check_single_island_no_sync(parsed)
        config = sample_roles.resolve(parsed, groups_per_round=parsed.groups_per_round)
        assert (config.role, config.store) == ("convertible", "/root/yeto-checkpoint-store/run-a")
        env = {}
        island_entry.apply_sample_role(parsed, types.SimpleNamespace(), env)
        assert env[SHARD_ENV] == f"island{m}"


def test_pool_miles_argv_selects_the_convertible_rollout_function():
    from yeto.rl.adapters.miles.config import sample_buffer_argv

    argv = sample_buffer_argv("convertible", max_policy_age=2)
    assert argv == ("--rollout-function-path",
                    "yeto.rl.adapters.miles.buffer_rollout_miles.ConvertibleRolloutFn")


def test_pool_refuses_an_island_that_is_not_convertible():
    base = [("island1=rollout" if t == "island1=convertible" else t) for t in POOL]
    with pytest.raises((ValueError, SystemExit), match="every --gpu island convertible"):
        _args(base=base)


def test_pool_refuses_a_held_trainer_island():
    from yeto.launch import trainer_loop

    args = _args(("--rl-trainer-scheduler", "s1", "--rl-trainer-island", "1"))
    with pytest.raises(SystemExit, match="holds no trainer island"):
        trainer_loop.check(args)


def test_pool_needs_lora_colocated_and_a_modal_volume_store():
    from yeto.launch import trainer_loop

    with pytest.raises((ValueError, SystemExit), match="lora|colocated"):
        _args(base=[("full" if t == "lora" else t) for t in POOL])
    args = _args()
    args.rl_checkpoint_store = "/tmp/store"
    with pytest.raises(SystemExit, match="modal-volume"):
        trainer_loop.check(args)


def test_one_island_is_not_a_pool():
    with pytest.raises((ValueError, SystemExit), match="at least two|island"):
        _args(gpu="modal:1xh100")


def test_binding_merges_shards_counts_fresh_samples_and_selects_by_rule(tmp_path):
    """The head's view: shard lines merged into the index; samples older than
    A_hard behind the trainer do not count; the node profiles come from --gpu."""
    from yeto.launch import convertible_pool
    from yeto.rl.engine.convertible_remote import node_path
    from yeto.rl.engine.sample_buffer import BUFFER_DIR, shard_index

    args = _args()
    files = MemFiles()
    binding = convertible_pool.binding(args, journal_dir=tmp_path, files=files)
    profiles = convertible_pool.node_profiles(args)
    assert [(p.name, p.gpu, p.gpus, p.gpu_memory_gb, p.domain) for p in profiles] == [
        ("island0", "H100", 1, 80, "modal"), ("island1", "H100", 1, 80, "modal")]

    def line(pos, version):
        return json.dumps({"schema": "yeto.rl.sample-buffer/v1", "pos": pos, "sample_id": f"s{pos}",
                           "behavior_version": version, "payload_ref": "", "tokens": 1,
                           "appended_at": float(pos)})

    batch = int(args.rollout_batch_size)
    files.write(f"{BUFFER_DIR}/{shard_index('island0')}",
                ("\n".join(line(i, 0) for i in range(batch)) + "\n").encode())
    files.write(f"{BUFFER_DIR}/{shard_index('island1')}",
                ("\n".join(line(i, 5) for i in range(batch)) + "\n").encode())
    assert binding.buffer.backlog(0).batches == 2
    assert binding.buffer.backlog(5).batches == 1  # version 0 is 5 > A_hard=2 behind
    for name, inflight in (("island0", 3), ("island1", 1)):
        files.write(node_path(name, "status.json"), json.dumps(
            {"node": name, "role": "rollout", "at": 9e18, "inflight": inflight,
             "capacity": 0.5}).encode())
    records = []
    from yeto.rl.engine.trainer_scheduler import S1Config, S1Scheduler

    gate = binding.wrap_scheduler(S1Scheduler(S1Config(a_hard=2, start_seconds=60.0)),
                                  lambda kind, tx_id=None, **f: records.append(f))
    assert gate.islands.plan().group == ("island1",)
