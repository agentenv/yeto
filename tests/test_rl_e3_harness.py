"""E3 A8/DEV-GATHER harness: CPU dry-run of the plan-v3 arm sequence and the G1-G6 judgement.

Pure-torch stand-in ranks (tests/rl_reshard_fakes.py); protocol only, no
GPU claim. The Miles backend and the Modal launcher are exercised on GPU
(DEV-GATHER first).
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.rl_reshard_fakes import GBS, MBS, ShardedBackend, default_args, make_world, train_step
from yeto.rl.engine.miles_adapter import LoopRunner
from yeto.rl.engine.miles_adapter import e3_probe
from yeto.rl.engine.miles_adapter.reshard import scheduled_partitions
from yeto.rl.engine.miles_adapter.trainer import MilesTrainerGroup

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "probes" / "e3_reshard"
sys.path.insert(0, str(TOOLS))
harness = importlib.import_module("harness")
compare = importlib.import_module("compare")


def _spec():
    return SimpleNamespace(loss=SimpleNamespace(aggregation="default", reducer=None, custom_loss=None),
                           advantage=SimpleNamespace(estimator="grpo", whiten=False), sha256=lambda: "a" * 64)


FROZEN = [torch.randn(GBS, 6, generator=torch.Generator().manual_seed(100 + i)) for i in range(8)]


class ProbeGroup:
    """run_plugin over fake ranks; the probe install/drain are emulated (they wrap fork modules)."""

    def __init__(self, ranks, records):
        self.ranks, self.records = ranks, records

    async def run_plugin(self, fn_path, kwargs=None):
        if fn_path == e3_probe.INSTALL_PROBE:
            return [{"process": 1, "loss_function": 1, "get_loss_function": 1} for _ in self.ranks]
        if fn_path == e3_probe.DRAIN_PROBE:
            out = [list(r) for r in self.records]
            for r in self.records:
                r.clear()
            return out
        module, name = fn_path.rsplit(".", 1)
        fn = getattr(importlib.import_module(module), name)
        return [fn(rank, **(kwargs or {})) for rank in self.ranks]


class FakeBackend:
    algorithm = _spec()
    global_batch_size = GBS
    micro_batch_size = MBS
    fingerprint = "fp"

    def __init__(self, perturb=None):
        self.trainer = None
        self.perturb = perturb or {}

    def start_arm(self, dp):
        random.seed(0)  # a fresh process
        np.random.seed(0)
        self.ranks = make_world(dp, seed=7, args=default_args(dp))
        self.records = [[] for _ in range(dp)]
        self.group = ProbeGroup(self.ranks, self.records)
        self.runner = LoopRunner()
        self.trainer = MilesTrainerGroup(args=self.ranks[0].args, actor_model=self.group, learner_id=0,
                                         learner_generation=0, parameter_layout_hash=lambda: "L",
                                         runner=self.runner, spec=self.algorithm)
        self.dp = dp

    def plugin(self, fn_path, kwargs=None):
        return list(self.runner.run(self.group.run_plugin(fn_path, kwargs)))

    def train(self, rollout_id):
        batch = FROZEN[rollout_id] * self.perturb.get(rollout_id, 1.0)
        sched = scheduled_partitions(list(range(GBS)), dp=self.dp, global_batch_size=GBS, micro_batch_size=MBS)
        for r, rank in enumerate(self.ranks):
            part = sched["partitions"][r]
            self.records[r].append({"kind": "shard", "partition": part, "has_micro_batch_indices": True,
                                    "has_num_rollouts": True, "micro_batch_indices": [[i] for i in range(len(part))],
                                    "num_rollouts": [GBS], "num_microbatches": sched["num_microbatches"]})
            with torch.no_grad():
                for i in part:
                    self.records[r].append({"kind": "normalizer", "num_microbatches": GBS // MBS // self.dp,
                                            "num_rollouts": GBS, "loss_parallel_size": self.dp})
                    loss = rank.model[0](batch[i:i + 1]).pow(2).mean()
                    self.records[r].append({"kind": "loss", "sample_indices": [i], "loss_hex": float(loss).hex()})
        train_step(self.ranks, batch)
        norm = float(torch.cat([p.detach().reshape(-1) for p in self.ranks[0].model[0].parameters()]).norm())
        return {"grad_norm": norm}

    def stop_arm(self):
        self.trainer = None


def _merge(states):
    names = sorted(states[0]["entries"])
    return ShardedBackend({}).check_optimizer(None, [(n, None) for n in names], states)


def _load(path):
    from yeto.rl.engine.miles_adapter.cut_plugin import from_safe

    return from_safe(torch.load(path, map_location="cpu", weights_only=True))


def test_dry_run_of_all_arms_is_judged_go(tmp_path):
    dirs = harness.run_all(lambda arm: FakeBackend(), tmp_path)
    assert [d.name for d in dirs] == ["A1", "A2", "B1", "B1p", "B2", "RT"]
    result = compare.judge(tmp_path, gbs=GBS, mbs=MBS, merge=_merge, load=_load)
    assert {g: result[g]["pass"] for g in ("G1", "G2", "G3", "G4", "G5", "G6")} == dict.fromkeys(
        ("G1", "G2", "G3", "G4", "G5", "G6"), True), json.dumps(result, default=repr)[:2000]
    assert result["decision"] == "go"
    b1 = harness.read_events(tmp_path / "arms" / "B1")
    assert [e["kind"] for e in b1][:6] == ["start", "probe_installed", "rank_info", "restore", "rank_info", "dump"]
    assert (tmp_path / "cuts" / "C1p" / "manifest.json").is_file()


def test_nondeterministic_repeat_is_inconclusive(tmp_path):
    harness.run_all(lambda arm: FakeBackend({2: 1.001} if arm.name == "B1p" else None), tmp_path)
    result = compare.judge(tmp_path, gbs=GBS, mbs=MBS, merge=_merge, load=_load)
    assert not result["G3"]["pass"] and result["decision"] == "inconclusive"


def test_unscheduled_shard_is_a_g2_failure(tmp_path):
    harness.run_all(lambda arm: FakeBackend(), tmp_path)
    path = tmp_path / "arms" / "B1" / "events.jsonl"
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    for e in lines:
        if e["kind"] == "train" and e["step"] == 3:
            for rank in e["probe"]:
                for r in rank:
                    if r["kind"] == "shard":
                        r["has_micro_batch_indices"] = False
    path.write_text("\n".join(json.dumps(e) for e in lines) + "\n")
    result = compare.judge(tmp_path, gbs=GBS, mbs=MBS, merge=_merge, load=_load)
    assert not result["G2"]["pass"] and result["decision"] == "no-go"


def test_corrupted_restore_is_a_g1_failure(tmp_path):
    harness.run_all(lambda arm: FakeBackend(), tmp_path)
    b1 = [e for e in harness.read_events(tmp_path / "arms" / "B1") if e["kind"] == "dump" and e["tag"] == "restored"]
    state = torch.load(b1[0]["ranks"][0]["path"], weights_only=True)
    first = next(iter(state["optimizer_named"]["entries"].values()))
    first["tensors"]["exp_avg"].add_(1e-3)
    torch.save(state, b1[0]["ranks"][0]["path"])
    result = compare.judge(tmp_path, gbs=GBS, mbs=MBS, merge=_merge, load=_load)
    assert result["G1"]["B1_vs_C1"] and result["decision"] == "no-go"


def test_arm_error_is_recorded_and_the_trainer_stopped(tmp_path):
    class Boom(FakeBackend):
        def train(self, rollout_id):
            raise RuntimeError("engine died")

    backend = Boom()
    with pytest.raises(RuntimeError):
        harness.run_arm(harness.ARM_BY_NAME["A1"], backend, tmp_path)
    events = harness.read_events(tmp_path / "arms" / "A1")
    assert events[-1]["kind"] == "error" and backend.trainer is None


# ---------------------------------------------------------------- probe plugin


def test_probe_wrappers_record_and_pass_through():
    e3_probe._RECORDS.clear()
    w = e3_probe.make_wrappers(lambda: 2)
    shard = {"partition": [1, 3], "micro_batch_indices": [[0], [1]], "num_rollouts": [4], "sample_indices": [1, 3]}
    proc = w["process"](lambda *a: (shard, "store"))
    assert proc("args", "ref", 0, 2) == (shard, "store")
    lf = w["loss_function"](lambda args, batch, nmb, logits, scale=False, num_rollouts=None: ("loss", nmb))
    assert lf("args", {}, 2, "logits", True, num_rollouts=4) == ("loss", 2)
    glf = w["get_loss_function"](lambda args, fn=None: (lambda a, b, l, s: (torch.tensor(0.5), {})))
    loss, _ = glf("args")("args", {"sample_indices": [3]}, None, None)
    records = e3_probe.drain_probe(None)
    assert records[0]["kind"] == "shard" and records[0]["has_micro_batch_indices"]
    assert records[1] == {"kind": "normalizer", "num_microbatches": 2, "num_rollouts": 4, "loss_parallel_size": 2}
    assert records[2] == {"kind": "loss", "sample_indices": [3], "loss_hex": (0.5).hex()}


def test_patch_everywhere_replaces_imported_names():
    def orig():
        return 1

    mod_a, mod_b = SimpleNamespace(f=orig), SimpleNamespace(g=orig, h=len)
    assert e3_probe.patch_everywhere(orig, "X", {"a": mod_a, "b": mod_b}) == 2
    assert mod_a.f == "X" and mod_b.g == "X" and mod_b.h is len


def test_rank_info_and_dump_state_on_a_fake_rank(tmp_path):
    rank = make_world(2)[1]
    info = e3_probe.rank_info(rank)
    assert info["coord"]["dp"] == 1 and info["dropout"]["lora_dropout"] == 0.0
    dumped = e3_probe.dump_state(rank, directory=str(tmp_path), tag="s2")
    assert Path(dumped["path"]).name == "s2_tp0_pp0_dp1.pt"


# ---------------------------------------------------------------- launch helpers


def test_container_script_asserts_before_running_arms():
    modal_run = importlib.import_module("modal_run")
    script = modal_run.container_script("a8")
    order = [script.index(x) for x in ("nvidia-smi", "GPU assertion failed", "miles pin mismatch",
                                       "--phase dry", "--phase gen", "--arm A1", "--arm A2", "--arm B1 ",
                                       "--arm B1p", "--arm B2", "--arm RT", "compare.py")]
    assert order == sorted(order)
    assert "NCCL_ALGO=Ring" in script and "NVIDIA H100 80GB HBM3" in script
    dev = modal_run.container_script("dev-gather")
    assert "NCCL_ALGO" not in dev and "NVIDIA A10G" in dev
    assert modal_run.PROFILES["a8"]["gpu"] == "H100!:2"


def test_learner_flags_are_taken_from_the_dry_run_plan():
    build_flags = importlib.import_module("build_flags")
    plan = {"island_requests": [{"learner_command": "RAY_ADDRESS=x python3 -m yeto.rl.learner "
                                                     "--learner-id 0 --rl-single-island-no-sync"}]}
    assert build_flags.learner_flags(plan) == "--learner-id 0 --rl-single-island-no-sync"
    with pytest.raises(ValueError):
        build_flags.learner_flags({"island_requests": [{"learner_command": "python3 -m yeto.rl.learner --x"}]})


ARGV = ["--global-batch-size", str(GBS), "--micro-batch-size", "1", "--lora-dropout", "0"]
ZERO = {"hidden_dropout": 0.0, "attention_dropout": 0.0, "lora_dropout": 0.0}


def test_shim_dry_phase_refuses_a_profile_with_dropout(tmp_path):
    shim = importlib.import_module("learner_shim")
    ns = SimpleNamespace(work=str(tmp_path), phase="dry", arm=None, set=[])
    launch = SimpleNamespace(argv=ARGV)
    args = SimpleNamespace(**{**vars(default_args(1)), "lora_dropout": 0.05})
    with pytest.raises(SystemExit, match="refused"):
        shim.make_phase(ns)(args, launch, _spec())
    ns.set = [f"{k}=0.0" for k in ZERO]
    assert shim.make_phase(ns)(SimpleNamespace(**vars(default_args(1))), launch, _spec()) is None
    summary = json.loads((tmp_path / "miles_args.dry.json").read_text())
    assert summary["argv_reshard_problems"] == summary["parsed_reshard_problems"] == {"1->2": [], "2->1": []}
    assert summary["parity_mismatch"] == []


def test_argv_and_parsed_disagreement_is_refused(tmp_path):
    shim = importlib.import_module("learner_shim")
    parsed = SimpleNamespace(**{**vars(default_args(1)), "balance_data": True})
    summary = shim.phase_summary(parsed, SimpleNamespace(argv=ARGV), _spec(), ZERO)
    assert "balance_data" in summary["parity_mismatch"]
    assert any("disagree on balance_data" in p for p in shim.summary_problems(summary))


FLAGS = (Path(__file__).resolve().parent / "data_e3_learner_flags.txt").read_text()


def test_local_dry_run_reproduces_the_container_refusal_and_passes_with_the_profile(monkeypatch):
    local_dry = importlib.import_module("local_dry")
    modal_run = importlib.import_module("modal_run")
    launch = local_dry.build_launch(FLAGS)
    assert "--balance-data" in launch.argv  # hard-coded by the ports translation
    shim = importlib.import_module("learner_shim")
    old = shim.argv_check(list(launch.argv), launch.algorithm,
                          shim.parse_overrides([o for o in modal_run.OVERRIDES["dev-gather"]
                                                if not o.startswith("balance_data")]))
    assert any("balance-data" in p for p in shim.summary_problems(old))  # the first DEV-GATHER refusal
    summary = local_dry.local_dry("dev-gather", FLAGS)
    assert summary["problems"] == [], summary["problems"]
    assert summary["argv_profile"]["balance_data"] is False and summary["argv_profile"]["global_batch_size"] == 16
    assert local_dry.local_dry("a8", FLAGS)["problems"] == []


def test_profile_overrides_are_applied_and_recorded(tmp_path):
    shim = importlib.import_module("learner_shim")
    modal_run = importlib.import_module("modal_run")
    assert "--set hidden_dropout=0.0" in modal_run.container_script("dev-gather")
    assert "--set deterministic_mode=true" in modal_run.container_script("a8")
    assert shim.parse_overrides(["hidden_dropout=0.0", "deterministic_mode=true"]) == {
        "hidden_dropout": 0.0, "deterministic_mode": True}
    ns = SimpleNamespace(work=str(tmp_path), phase="dry", arm=None,
                         set=["lora_dropout=0.0", "hidden_dropout=0.0", "attention_dropout=0.0"])
    args = SimpleNamespace(**{**vars(default_args(1)), "lora_dropout": 0.05})
    assert shim.make_phase(ns)(args, SimpleNamespace(argv=ARGV), _spec()) is None
    assert json.loads((tmp_path / "miles_args.dry.json").read_text())["overrides"]["lora_dropout"] == 0.0
