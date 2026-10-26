"""S27 G-FN: the verl Megatron island as a buffer trainer (step 1c), CPU checks.

A buffer trainer island (``--rl-sample-role trainer``) consumes samples from the
durable store and publishes its own weights there; it holds no rollout engine. Six
pieces of yeto's own wiring blocked the Megatron engine on that role
(``SESSION27/VERL-MEGATRON-PP.md`` section 7.10, blockers B1-B6), plus a seventh
found while implementing (B7, the trainer card count). What each test covers:

* B1/B2/B3: ``island_entry.megatron_island_plan`` accepts the role, the fully_async
  policy age and a colocated placement, and still refuses the combinations that are
  wrong for either role (the gate is relaxed, not removed);
* B3: ``separate.trainer_only_pools`` makes every island card a training card, and
  ``MegatronShape.pools`` refuses to invent a rollout pool;
* B6/B7: ``island_entry.fully_async_plan`` writes the trainer's real card count on
  the fully_async path, for the buffer trainer and for the separate pools alike
  (the two were independent changes and meet in one call);
* B5: ``fully_async_full.full_parameter_ports`` installs the Megatron trainer group,
  so each release crosses the HF boundary;
* B4: ``megatron_ports.island_megatron_checks`` runs the Megatron island's two
  start-up readbacks on a path that never calls ``run_megatron_island``.

No GPU or NPU is touched here; the on-card result is section 7.12 of that report.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from yeto.rl.adapters.verl import megatron_engine as me
from yeto.rl.adapters.verl import separate


# ----------------------------------------------------------------------------- island plan
def _args(tmp_path, *, buffer_role: bool, **kw):
    """Island arguments of a Megatron island; ``buffer_role`` spells the trainer role."""
    from yeto.rl.adapters.verl import island_entry

    argv = ["--model", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "--model-revision", "r",
            "--data", "~/yeto-data.jsonl", "--learner-id", "0",
            "--reward-function", "yeto.rl.numeric_answer_reward:score", "--global-rounds", "3",
            "--groups-per-round", "4", "--samples-per-group", "8", "--rollout-max-response-len", "2048",
            "--event-tape", str(tmp_path / "tape.jsonl"), "--inner-lr", "1e-06", "--seq-len", "2560",
            "--seed", "17", "--rl-single-island-no-sync", "--parameter-mode", "full",
            "--actor-num-nodes", "1", "--actor-num-gpus-per-node", "8", "--rl-island-gpus-per-node", "8",
            "--tensor-parallel", "1", "--pipeline-parallel", "2", "--expert-parallel", "4",
            "--rl-allow-pipeline-parallel-sync", "--rl-lr-schedule", "constant", "--trust-remote-code",
            "--sglang-mem-fraction-static", "0.8"]
    if buffer_role:
        argv += ["--rl-sample-role", "trainer", "--rl-max-policy-age", "4",
                 "--rl-resume-store", str(tmp_path / "store"), "--rl-placement", "colocated"]
    else:
        argv += ["--rl-max-policy-age", "0", "--rl-placement", "fixed-partition",
                 "--rollout-num-gpus", "8", "--rollout-num-gpus-per-engine", "2"]
    args, _ignored = island_entry.parse_args(argv + list(kw.pop("extra", [])))
    for key, value in kw.items():
        setattr(args, key, value)
    return island_entry, args


def _plan(tmp_path, *, buffer_role: bool = True, cards: int = 8, config=None, **kw):
    island_entry, args = _args(tmp_path, buffer_role=buffer_role, **kw)
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text(json.dumps(config or {"n_routed_experts": 128}))
    return island_entry.megatron_island_plan(
        args, model_path=str(model), train_file="/d/train.parquet", val_file="/d/test.parquet", out_dir="/o",
        algo={"correction": "tis", "tis_upper": 2.0, "tis_lower": 0.0, "eps_clip": None, "eps_clip_high": None},
        chat={}, ray_info={"after": {"GPU": cards + (8 if not buffer_role else 0)}}, reward_path="/r.py",
        reward_name="compute_score_numeric_answer", buffer_trainer=buffer_role)


def _value(overrides, key):
    for item in overrides:
        if item.startswith(key + "="):
            return item.split("=", 1)[1]
    return None


def test_a_megatron_buffer_trainer_takes_every_island_card_as_a_training_card(tmp_path):
    """B1-B3: the role, the fully_async age and the colocated placement are accepted,
    and all eight cards are training cards (TP=1 x PP=2 -> DP=4)."""
    plan = _plan(tmp_path, cards=8)
    shape = plan["shape"]
    assert shape["rollout_pool"] is False
    assert (shape["trainer_nodes"], shape["trainer_gpus_per_node"], shape["trainer_gpus"]) == (1, 8, 8)
    assert (shape["tp"], shape["pp"], shape["ep"], shape["data_parallel"]) == (1, 2, 4, 4)
    assert plan["run"].n_gpus == 8


def test_the_megatron_keys_are_written_and_the_separate_mode_block_is_not(tmp_path):
    """B6: the fully_async config tree includes ppo_trainer, so every Megatron key the
    island writes still resolves; what drops out is verl's separate mode, which exists
    to place a rollout pool."""
    plan = _plan(tmp_path)
    overrides, asserted = plan["overrides"], plan["asserted_keys"]
    assert "model_engine=megatron" in overrides
    meg = "actor_rollout_ref.actor.megatron"
    assert _value(overrides, f"{meg}.pipeline_model_parallel_size") == "2"
    assert _value(overrides, f"{meg}.tensor_model_parallel_size") == "1"
    assert _value(overrides, f"{meg}.expert_model_parallel_size") == "4"
    assert _value(overrides, f"{meg}.use_mbridge") == "True"
    assert f"++{meg}.override_ddp_config.check_for_nan_in_grad=True" in overrides
    # the separate mode's own keys: none of them, and none of them asserted
    for key in ("trainer.v1.trainer_mode", "trainer.v1.separate_async.parameter_sync_step",
                "trainer.v1.separate_async.num_warmup_batches", "actor_rollout_ref.rollout.nnodes",
                "actor_rollout_ref.rollout.n_gpus_per_node"):
        assert _value(overrides, key) is None
        assert key not in asserted
    # what the fully_async overrides do write stays asserted
    assert "actor_rollout_ref.hybrid_engine" in asserted
    assert "actor_rollout_ref.rollout.checkpoint_engine.backend" in asserted
    assert f"{meg}.tensor_model_parallel_size" in asserted


def test_a_megatron_island_with_a_rollout_pool_is_unchanged(tmp_path):
    """The separate-mode island keeps both pools, the separate block and all its
    asserted keys: relaxing the gate for one role changed nothing for the other."""
    plan = _plan(tmp_path, buffer_role=False, cards=8)
    assert plan["shape"]["rollout_pool"] is True
    assert (plan["shape"]["trainer_gpus"], plan["shape"]["rollout_gpus"]) == (8, 8)
    assert _value(plan["overrides"], "trainer.v1.trainer_mode") == separate.TRAINER_MODE
    assert _value(plan["overrides"], "actor_rollout_ref.rollout.n_gpus_per_node") == "8"
    assert plan["asserted_keys"] == me.ASSERTED_KEYS


@pytest.mark.parametrize("role,change,match", [
    # the buffer trainer role: what is still refused
    (True, dict(rl_placement="fixed-partition"), "--rl-placement colocated"),
    (True, dict(rl_max_policy_age=0), "--rl-max-policy-age >= 1"),
    (True, dict(rl_rollout_quant_tier=2), "no --rl-rollout-quant-tier"),
    (True, dict(parameter_mode="lora"), "--tuning full"),
    (True, dict(rollout_num_gpus=4), "--rollout-num-gpus 4 不接"),
    (True, dict(rl_lr_schedule="cosine"), "constant"),
    # the separate-mode role: every old refusal still stands
    (False, dict(rl_placement="colocated"), "fixed-partition"),
    (False, dict(rl_sample_role="trainer"), "no --rl-sample-role"),
    (False, dict(rl_max_policy_age=1), "--rl-max-policy-age 0"),
    (False, dict(rl_rollout_quant_tier=2), "no --rl-rollout-quant-tier"),
])
def test_the_relaxed_gate_still_refuses_the_wrong_combinations(tmp_path, role, change, match):
    with pytest.raises(SystemExit, match=match):
        _plan(tmp_path, buffer_role=role, **change)


def test_the_buffer_trainer_pools_must_cover_every_island_card(tmp_path):
    """``trainer_only_pools``: the island's cards are the trainer's cards, so a node
    split that does not account for all of them is refused before verl starts."""
    args = SimpleNamespace(actor_num_nodes=1, actor_num_gpus_per_node=4, rollout_num_gpus=0,
                           sglang_mem_fraction_static=0.8)
    with pytest.raises(SystemExit, match="!= 岛内 Ray 的 8 卡"):
        separate.trainer_only_pools(args, 8)
    pools = separate.trainer_only_pools(SimpleNamespace(actor_num_nodes=2, actor_num_gpus_per_node=8,
                                                        rollout_num_gpus=None), 16)
    assert (pools.trainer_nodes, pools.trainer_gpus_per_node, pools.trainer_gpus) == (2, 8, 16)


def test_a_shape_without_a_rollout_pool_refuses_to_invent_one():
    """B3: nothing may ask a buffer trainer shape for the two pools; the engine-neutral
    pool validation is skipped for it instead of being fed placeholder numbers."""
    shape = me.MegatronShape(trainer_nodes=1, trainer_gpus_per_node=8, tp=1, pp=2, ep=4, num_experts=128,
                             rollout_pool=False)
    me.validate(shape)  # does not raise although the rollout fields are defaults
    with pytest.raises(me.MegatronShapeError, match="没有 rollout 池"):
        shape.pools()
    assert me.asserted_keys(shape) != me.ASSERTED_KEYS
    assert me.build_overrides(_run_config(), shape)  # does not raise
    # an impossible training shape is still refused
    with pytest.raises(me.MegatronShapeError, match="不整除训练卡数"):
        me.validate(me.MegatronShape(trainer_gpus_per_node=8, tp=3, pp=2, ep=1, rollout_pool=False))


def _run_config():
    from yeto.rl.adapters.verl import config as vconf

    return vconf.VerlRunConfig(
        model_path="/m", train_file="/t.parquet", val_file="/v.parquet", out_dir="/o",
        parameter_mode="full", lora_rank=0, lora_alpha=0, lr=1e-6, groups_per_round=4,
        samples_per_group=8, max_prompt_len=512, max_response_len=2048, seed=17, total_rounds=3,
        correction="tis", tis_upper=2.0, max_model_len=2560, reward_path="/r.py",
        reward_name="compute_score_numeric_answer", n_gpus=8, experiment_name="yeto-verl-0")


# ----------------------------------------------------------------------------- B6 + B7 meeting in one call
def test_the_fully_async_overrides_carry_the_trainer_cards_of_both_paths(tmp_path):
    """B7 (G-EZ2's fix) and the separate pools (G-EV) are independent changes that meet
    in ``fully_async_plan``. The buffer trainer's card count wins when both are given:
    it is the role with no rollout pool, so its cards are all of the island's."""
    from yeto.rl.adapters.verl import island_entry
    from yeto.rl.engine.algorithm import AlgorithmSpec, CorrectionSpec, ExecutionSpec

    _entry, args = _args(tmp_path, buffer_role=True)
    base = _plan(tmp_path)["overrides"]
    spec = AlgorithmSpec(execution=ExecutionSpec(max_policy_staleness=4),
                         correction=CorrectionSpec(method="tis", lower=0.0, upper=2.0)).to_dict()

    out, asserted = island_entry.fully_async_plan(4, base, spec, args, "none", pools=None, trainer_cards=8)
    assert _value(out, "trainer.n_gpus_per_node") == "8" and _value(out, "trainer.nnodes") == "1"
    assert _value(out, "actor_rollout_ref.hybrid_engine") == "False"
    assert "trainer.n_gpus_per_node" in asserted and "trainer.nnodes" in asserted

    # two nodes: --actor-num-nodes splits the same cards
    args.actor_num_nodes, args.actor_num_gpus_per_node = 2, 8
    out2, _ = island_entry.fully_async_plan(4, base, spec, args, "none", pools=None, trainer_cards=16)
    assert _value(out2, "trainer.n_gpus_per_node") == "8" and _value(out2, "trainer.nnodes") == "2"

    # a card count the nodes do not cover is refused here, not inside verl
    with pytest.raises(SystemExit, match="buffer trainer"):
        island_entry.fully_async_plan(4, base, spec, args, "none", pools=None, trainer_cards=12)

    # the separate pools still route through when there is no buffer trainer
    args.actor_num_nodes, args.actor_num_gpus_per_node = 1, 8
    pools = separate.SeparatePools(trainer_nodes=1, trainer_gpus_per_node=6, rollout_nodes=1,
                                   rollout_gpus_per_node=2, rollout_tp=2)
    out3, _ = island_entry.fully_async_plan(4, base, spec, args, "none", pools=pools, trainer_cards=None)
    assert _value(out3, "trainer.n_gpus_per_node") == "6"
    assert _value(out3, "rollout.n_gpus_per_node") == "2"

    # both given: the buffer trainer's cards win and no rollout cards are claimed
    out4, _ = island_entry.fully_async_plan(4, base, spec, args, "none", pools=pools, trainer_cards=8)
    assert _value(out4, "trainer.n_gpus_per_node") == "8" and _value(out4, "trainer.nnodes") == "1"
    assert _value(out4, "rollout.n_gpus_per_node") == "1"  # the fully_async default; no rollouter runs


# ----------------------------------------------------------------------------- B5: which group is installed
SHAPES = {"model.embed_tokens.weight": (4, 8)}
# One rank's ``megatron_hf_boundary.manifest`` record over SHAPES: one task, FP32 master.
MANIFEST = {"rank": 0, "tasks": [[0, ["model.embed_tokens.weight"], "float32"]], "refused": []}


class _Island:
    """The parts of ``FullyAsyncIsland`` a full-parameter group uses."""

    def __init__(self):
        self.full_group = None
        self.layout_hash = None
        self.calls: list = []

    def call_workers(self, fn, *args):
        self.calls.append((fn.__name__, args))
        return [dict(MANIFEST)]

    def call_workers_shared(self, fn, *args):
        return self.call_workers(fn, *args)


def _buffer_plan(tmp_path, engine: str) -> dict:
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text(json.dumps({"hidden_size": 8, "num_hidden_layers": 1}))
    return {"engine": engine, "parameter_mode": "full", "sync": "none", "global_rounds": 3,
            "model_path": str(model), "model_revision": "c" * 40,
            "model_config_sha256": "a" * 64, "max_policy_age": 4}


def test_a_megatron_buffer_trainer_installs_the_megatron_trainer_group(tmp_path, monkeypatch):
    """B5: the one-line blocker. Without the group the releases go through the FSDP2
    reader and ``megatron_hf_boundary.export_hf`` is never called."""
    from yeto.rl.adapters.verl import fully_async_full
    from yeto.rl.adapters.verl.full_parameter_stream import VerlFullParameterGroup
    from yeto.rl.adapters.verl.megatron_full_parameter import MegatronFullParameterGroup

    monkeypatch.setattr("yeto.rl.adapters.verl.full_parameter_plan.model_shapes", lambda path: dict(SHAPES))
    island = _Island()
    fully_async_full.full_parameter_ports(island, _buffer_plan(tmp_path, "megatron"),
                                          backend_identity_sha256="b" * 64)
    assert isinstance(island.full_group, MegatronFullParameterGroup)
    assert island.full_group.shapes == SHAPES
    # the local layout came from this group's manifest, over the trainer workers
    assert [name for name, _ in island.calls] == ["remote_megatron_manifest"]
    assert island.layout_hash

    # an FSDP2 plan asks for no group here, so full_parameter_policy keeps its default
    assert fully_async_full.trainer_group(_Island(), _buffer_plan(tmp_path, "fsdp2")) is None
    assert VerlFullParameterGroup is not MegatronFullParameterGroup


def test_the_release_weights_of_a_megatron_buffer_trainer_read_that_group(tmp_path, monkeypatch):
    """The publication path: ``buffer_ports.release_weights_for`` writes with the HF
    writer over ``island.full_group``, so the group decided above is the one each
    release reads (``hf_export.VerlHfWriter`` -> ``group.manifests``)."""
    from yeto.rl.adapters.verl import buffer_ports, fully_async_full
    from yeto.rl.adapters.verl.megatron_full_parameter import MegatronFullParameterGroup

    monkeypatch.setattr("yeto.rl.adapters.verl.full_parameter_plan.model_shapes", lambda path: dict(SHAPES))
    island = _Island()
    plan = _buffer_plan(tmp_path, "megatron")
    fully_async_full.full_parameter_ports(island, plan, backend_identity_sha256="b" * 64)
    weights = buffer_ports.release_weights_for(island, plan)
    assert weights is not None
    assert isinstance(island.full_group, MegatronFullParameterGroup)


# ----------------------------------------------------------------------------- B4: the Megatron readbacks
class _Tape:
    def __init__(self):
        self.events: list = []

    def append(self, event):
        self.events.append(event)


def test_the_buffer_path_runs_the_megatron_start_up_readbacks(capsys):
    """B4: ``run_megatron_island`` is the synchronous driver, which this path never
    calls, so its two readbacks are wired here instead: the gradient non-finite check
    (a refusal) and the Ascend plugin stack (a log)."""
    from yeto.rl.adapters.verl import megatron_ports

    class _OneRank(_Island):
        def __init__(self, flag):
            super().__init__()
            self.flag = flag

        def call_workers(self, fn, *args):
            self.calls.append(fn.__name__)
            if fn is megatron_ports.remote_megatron_grad_checks:
                return [{"rank": 0, "wrapped": True, "check_for_nan_in_grad": self.flag}]
            return [{"rank": 0, "ran": True, "installed": ["a"], "failed": {}}]

    plan = {"megatron": {"grad_nan_check": True},
            "ascend_plugin_stack": {"root": "/p", "env": {"X": "1"}, "plugins": ["a"]}}
    tape = _Tape()
    island = _OneRank(True)
    megatron_ports.island_megatron_checks(island, plan, tape)
    assert island.calls == ["remote_megatron_grad_checks", "remote_ascend_plugin_report"]
    events = [event["event"] for event in tape.events]
    assert events == ["rl_megatron_grad_checks", "rl_ascend_plugin_stack"]
    assert "gradient NaN check wanted=True" in capsys.readouterr().out

    # a rank without the check stops the island
    with pytest.raises(RuntimeError):
        megatron_ports.island_megatron_checks(_OneRank(False), plan, _Tape())

    # no plugin stack section: only the gradient check runs
    quiet = _OneRank(True)
    megatron_ports.island_megatron_checks(quiet, {"megatron": {"grad_nan_check": True}}, _Tape())
    assert quiet.calls == ["remote_megatron_grad_checks"]


# ----------------------------------------------------------------------------- the optimizer-state memory
def test_the_optimizer_offload_fraction_writes_the_engine_keys_and_is_off_by_default(tmp_path):
    """S27 G-FN (B9): on 64 GB cards a PP=2 Nano island does not fit with the whole
    optimizer state on the card. The engine-neutral --rl-optimizer-offload-fraction is
    reused for it (the launcher already forwards that flag and Miles already honours it);
    it writes Megatron's own keys, because the engine's idle-time offload switch does not
    set them (VERL-ASCEND-MEGATRON.md conclusion B: the resolved config kept
    optimizer_cpu_offload False and the per-card peak did not move). Off by default, so a
    GPU island is unchanged."""
    off = _plan(tmp_path)
    assert off["shape"]["optimizer_offload_fraction"] == 0.0
    assert not [o for o in off["overrides"] if "override_optimizer_config" in o]
    assert not [o for o in off["overrides"] if "use_precision_aware_optimizer" in o]
    assert me.optimizer_memory_overrides(0.0) == {}

    on = _plan(tmp_path, extra=["--rl-optimizer-offload-fraction", "1.0"])
    assert on["shape"]["optimizer_offload_fraction"] == 1.0
    actor = "actor_rollout_ref.actor"
    assert f"{actor}.optim.use_precision_aware_optimizer=True" in on["overrides"]
    assert f"{actor}.optim.main_grads_dtype=bf16" in on["overrides"]
    assert f"{actor}.optim.exp_avg_dtype=bf16" in on["overrides"]
    assert f"{actor}.optim.exp_avg_sq_dtype=bf16" in on["overrides"]
    assert f"++{actor}.optim.override_optimizer_config.optimizer_cpu_offload=True" in on["overrides"]
    assert f"++{actor}.optim.override_optimizer_config.optimizer_offload_fraction=1.0" in on["overrides"]
    assert (f"++{actor}.optim.override_optimizer_config.use_torch_optimizer_for_cpu_offload=True"
            in on["overrides"])
    # the engine's own idle-time offload keys keep their values: this does not change
    # what they mean (G-FJ2 recommended against remapping them)
    assert f"{actor}.megatron.optimizer_offload=False" in on["overrides"]
    assert f"{actor}.megatron.param_offload=False" in on["overrides"]
    # a half fraction passes the value through
    half = me.optimizer_memory_overrides(0.5)
    assert half["++optim.override_optimizer_config.optimizer_offload_fraction"] == "0.5"
    # out of range is refused by the shape, not passed to hydra
    with pytest.raises(me.MegatronShapeError, match="optimizer_offload_fraction"):
        me.validate(me.MegatronShape(tp=1, pp=2, ep=4, rollout_pool=False,
                                     optimizer_offload_fraction=1.5))


def test_a_hybrid_model_without_mtp_layers_closes_the_bridge_mtp_branch(tmp_path):
    """S27 G-FN (B12): a three-way interaction, not an Ascend quirk (G-FJ2 traced it,
    VERL-ASCEND-MEGATRON.md conclusion C item 2). The bridge enters its MTP branch
    whenever mtp_hybrid_override_pattern is truthy and then multiplies the pattern by
    mtp_num_layers, which verl spells None for off; transformers supplies the truthy
    default for a checkpoint whose config.json has no MTP key. The island writes the
    pattern as null so the branch stays closed -- which is what the checkpoint asked
    for. A model that really has MTP layers keeps the bridge's own handling."""
    key = "++actor_rollout_ref.actor.megatron.override_transformer_config.mtp_hybrid_override_pattern"

    # a model that is not hybrid at all: nothing is written
    plain = _plan(tmp_path)
    assert plain["shape"]["close_mtp_pattern"] is False
    assert not [o for o in plain["overrides"] if "mtp_hybrid_override_pattern" in o]

    # Nano's shape: a hybrid pattern and no MTP key anywhere
    closed = _plan(tmp_path, config={"n_routed_experts": 128,
                                     "hybrid_override_pattern": "MEMEM*EMEMEM"})
    assert closed["shape"]["close_mtp_pattern"] is True
    assert f"{key}={me.MTP_OFF_PATTERN}" in closed["overrides"]

    # a hybrid model that does declare MTP layers is left alone
    for declared in ("mtp_num_layers", "num_nextn_predict_layers", "mtp_hybrid_override_pattern"):
        mtp = _plan(tmp_path, config={"n_routed_experts": 128,
                                      "hybrid_override_pattern": "MEMEM*EMEMEM", declared: 1})
        assert mtp["shape"]["close_mtp_pattern"] is False
        assert not [o for o in mtp["overrides"] if "mtp_hybrid_override_pattern" in o]


def test_the_island_writes_a_positive_total_training_steps(tmp_path):
    """S27 G-FN (B13): the Megatron engine builds Megatron's LR scheduler inside
    init_model, where Megatron asserts lr_decay_steps > 0; verl derives that from
    optim.total_training_steps, default -1, and fills it in only after the workers
    exist. The FSDP2 engine builds no such scheduler, so only a Megatron island trips
    it (run fn-6 died there in every worker). The island writes the count up front."""
    key = "actor_rollout_ref.actor.optim.total_training_steps"
    assert _value(_plan(tmp_path)["overrides"], key) == "3"  # --global-rounds 3 in _args
    # not asserted: a trainer that recomputes it later must be allowed to win
    assert key not in _plan(tmp_path)["asserted_keys"]
    # the separate-mode island writes it too
    assert _value(_plan(tmp_path, buffer_role=False)["overrides"], key) == "3"


def test_a_buffer_trainer_asks_for_the_naive_weight_transport(tmp_path):
    """S27 G-FN (B14): a buffer trainer starts no rollouter and no checkpoint engine, so
    it asks for the always-present 'naive' transport. Every trainer worker validates the
    configured backend while it builds its model, and the fully_async yaml defaults it to
    nccl, so leaving the key alone is not enough: runs fn-7 and fn-8 on npu1 died in
    every worker with 'Checkpoint engine nccl not registered, registered backends:
    [naive]'. An island that does have a rollouter still asks for nccl."""
    from yeto.rl.adapters.verl import island_entry
    from yeto.rl.adapters.verl.fully_async_round import fully_async_run_overrides
    from yeto.rl.engine.algorithm import AlgorithmSpec, CorrectionSpec, ExecutionSpec

    key = "actor_rollout_ref.rollout.checkpoint_engine.backend"
    _entry, args = _args(tmp_path, buffer_role=True)
    base = _plan(tmp_path)["overrides"]
    spec = AlgorithmSpec(execution=ExecutionSpec(max_policy_staleness=4),
                         correction=CorrectionSpec(method="tis", lower=0.0, upper=2.0)).to_dict()

    buffer, _ = island_entry.fully_async_plan(4, base, spec, args, "none", pools=None, trainer_cards=8)
    assert _value(buffer, key) == "naive"
    # the rollouter island (no trainer_cards) keeps it
    rollouter, _ = island_entry.fully_async_plan(4, base, spec, args, "none", pools=None, trainer_cards=None)
    assert _value(rollouter, key) == "nccl"
    # the engine-neutral default is unchanged
    assert key + "=nccl" in fully_async_run_overrides([], 4, groups_per_round=2, rounds=2)
    assert key + "=naive" in fully_async_run_overrides([], 4, groups_per_round=2, rounds=2, rollouter=False)


def test_bf16_moments_flag_writes_precision_aware_keys_with_fp32_masters(tmp_path):
    """S28 G-RCP: --rl-megatron-bf16-moments keeps the optimizer state on the card with
    bf16 Adam moments and an FP32 master the full-parameter boundary can read."""
    on = _plan(tmp_path, extra=["--rl-megatron-bf16-moments"])
    actor = "actor_rollout_ref.actor"
    assert on["shape"]["bf16_moments"] is True
    assert f"{actor}.optim.use_precision_aware_optimizer=True" in on["overrides"]
    assert f"{actor}.optim.exp_avg_dtype=bf16" in on["overrides"]
    assert f"{actor}.optim.exp_avg_sq_dtype=bf16" in on["overrides"]
    assert f"++{actor}.optim.override_optimizer_config.store_param_remainders=False" in on["overrides"]
    assert not [o for o in on["overrides"] if "cpu_offload" in o or "main_grads_dtype" in o]
    with pytest.raises(me.MegatronShapeError):
        _plan(tmp_path, extra=["--rl-megatron-bf16-moments", "--rl-optimizer-offload-fraction", "0.5"])


def test_s28_the_120b_recipe_shape_plans_uneven_stages_full_recompute_and_host_offload(tmp_path):
    """S28 G-120: 24 cards TP4/PP3/EP8, 29/30 first/last stage layers (88 layers),
    full-layer recompute and the whole optimizer state on the host."""
    plan = _plan(tmp_path, cards=24, config={"n_routed_experts": 512},
                 actor_num_nodes=3, actor_num_gpus_per_node=8, rl_island_gpus_per_node=8,
                 tensor_parallel=4, pipeline_parallel=3, expert_parallel=8,
                 extra=["--rl-megatron-first-stage-layers", "29", "--rl-megatron-last-stage-layers", "30",
                        "--rl-megatron-recompute-full-layers", "1", "--rl-optimizer-offload-fraction", "1.0"])
    shape, o = plan["shape"], plan["overrides"]
    assert (shape["tp"], shape["pp"], shape["ep"], shape["data_parallel"], shape["trainer_gpus"]) == (4, 3, 8, 2, 24)
    t = "++actor_rollout_ref.actor.megatron.override_transformer_config"
    assert _value(o, f"{t}.num_layers_in_first_pipeline_stage") == "29"
    assert _value(o, f"{t}.num_layers_in_last_pipeline_stage") == "30"
    assert (_value(o, f"{t}.recompute_granularity"), _value(o, f"{t}.recompute_method"),
            _value(o, f"{t}.recompute_num_layers")) == ("full", "uniform", "1")
    assert _value(o, f"{t}.recompute_modules") is None
    assert _value(o, "++actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload") == "True"


def test_s28_default_shape_writes_no_stage_keys_and_keeps_selective_recompute(tmp_path):
    o = _plan(tmp_path)["overrides"]
    t = "++actor_rollout_ref.actor.megatron.override_transformer_config"
    assert not any("pipeline_stage" in x for x in o)
    assert _value(o, f"{t}.recompute_granularity") == "selective"


def test_s28_stage_layers_need_pp():
    with pytest.raises(me.MegatronShapeError, match="PP >= 2"):
        me.validate(me.MegatronShape(rollout_pool=False, pp=1, first_stage_layers=29))
