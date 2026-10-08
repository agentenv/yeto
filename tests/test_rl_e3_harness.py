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
from yeto.rl.adapters.miles import LoopRunner
from yeto.rl.adapters.miles import e3_probe
from yeto.rl.adapters.miles.reshard import scheduled_partitions
from yeto.rl.adapters.miles.trainer import MilesTrainerGroup

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
    from yeto.rl.adapters.miles.cut_plugin import from_safe

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
    order = [script.index(x) for x in ("trap pack EXIT", "nvidia-smi", "GPU assertion failed", "miles pin mismatch",
                                       "run_phase dry", "run_phase gen", "run_phase A1 ", "run_phase A2 ",
                                       "run_phase B1 ", "run_phase B1p ", "run_phase B2 ", "run_phase RT ",
                                       "compare.py")]
    assert order == sorted(order)
    assert "NCCL_ALGO=Ring" in script and "^(NVIDIA H100 80GB HBM3)," in script
    dev = modal_run.container_script("dev-gather")
    assert "NCCL_ALGO" not in dev and "^(NVIDIA A10G|NVIDIA A10)," in dev
    assert modal_run.PROFILES["a8"]["gpu"] == "H100!:2"
    # per-phase stall and server-error watchdogs, not only the Sandbox timeout
    assert f"-gt {modal_run.STALL_MINUTES * 60} ]" in dev and "503 Service Unavailable" in dev
    # pins come from yeto/rl/__init__.py (never hard-coded in the harness or here)
    from yeto.rl import MILES_NEXT_COMMIT, MILES_NEXT_IMAGE

    assert modal_run.MILES_COMMIT == MILES_NEXT_COMMIT
    assert "docker:" + modal_run.IMAGE == MILES_NEXT_IMAGE


def test_stall_watchdog_kills_a_silent_phase(tmp_path):
    """Run the real run_phase shell function with a phase that never reports progress."""
    import subprocess

    modal_run = importlib.import_module("modal_run")
    script = modal_run.container_script("dev-gather", work=str(tmp_path), flags_file="/dev/null")
    head = script[:script.index("nvidia-smi --query")]
    head = head.replace("sleep 30", "sleep 1").replace("cd /yeto", "cd " + str(tmp_path))
    head = head.replace(f"-gt {modal_run.STALL_MINUTES * 60} ]", "-gt 2 ]")
    body = head
    # replace the phase command by a silent sleeper
    body = body.replace("bash -c \"python", "sleep 60; : \"python")
    proc = subprocess.run(["bash", "-c", body + 'run_phase gen "--phase gen"; echo NOT-REACHED'],
                          capture_output=True, text=True, timeout=60)
    assert "STALLED" in (tmp_path / "progress.log").read_text()
    assert "NOT-REACHED" not in proc.stdout and "=== EVIDENCE_B64 ===" in proc.stdout


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


@pytest.fixture
def reward_module(tmp_path, monkeypatch):
    (tmp_path / "gsm8k_reward.py").write_text("def score(*a, **k):\n    return 0.0\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield
    sys.modules.pop("gsm8k_reward", None)


def test_local_dry_resolves_every_hook_the_argv_names(reward_module):
    local_dry = importlib.import_module("local_dry")
    summary = local_dry.local_dry("dev-gather", FLAGS)
    flags = {c["flag"]: c for c in summary["callables"]}
    assert flags["--rollout-all-samples-process-path"]["ok"] is True
    assert flags["--custom-rm-path"]["ok"] is True and summary["problems"] == []


def test_local_dry_reports_a_missing_hook_module(monkeypatch):
    local_dry = importlib.import_module("local_dry")
    sys.modules.pop("gsm8k_reward", None)
    assert any("gsm8k_reward" in p for p in local_dry.local_dry("dev-gather", FLAGS)["problems"])


def test_local_dry_run_reproduces_the_container_refusal_and_passes_with_the_profile(reward_module):
    local_dry = importlib.import_module("local_dry")
    shim = importlib.import_module("learner_shim")
    modal_run = importlib.import_module("modal_run")
    from yeto.rl import learner
    from yeto.rl.engine.run_config import resolve_rl_run_config

    # Without the trainer-edge translation the argv carries --balance-data: the DEV-GATHER run 1 refusal.
    args = learner.parse_args(local_dry.learner_argv(FLAGS))
    plain = learner.build_ports_launch(args, resolve_rl_run_config(
        args, model_path="/m", prompt_path="/p", provider=SimpleNamespace(**local_dry.QWEN3_0_6B_PROVIDER),
        target_modules=list(local_dry.TARGETS), yeto_policy_sync=False))
    assert "--balance-data" in plain.argv
    old = shim.argv_check(list(plain.argv), plain.algorithm, shim.parse_overrides(modal_run.OVERRIDES["dev-gather"]))
    assert any("balance-data" in p for p in shim.summary_problems(old))
    # The harness uses the production trainer-edge translation: no --balance-data, no override needed.
    summary = local_dry.local_dry("dev-gather", FLAGS)
    assert summary["problems"] == [], summary["problems"]
    assert "--balance-data" not in summary["argv"] and summary["argv_profile"]["balance_data"] is False
    assert summary["argv_profile"]["global_batch_size"] == 16
    # A8 needs the deterministic trainer flag.
    assert "argv lacks --deterministic-mode" in local_dry.local_dry("a8", FLAGS)["problems"]
    assert local_dry.local_dry("a8", FLAGS + " --rl-deterministic-trainer")["problems"] == []


def test_profile_overrides_are_applied_and_recorded(tmp_path):
    shim = importlib.import_module("learner_shim")
    modal_run = importlib.import_module("modal_run")
    assert "--set hidden_dropout=0.0" in modal_run.container_script("dev-gather")
    assert "balance_data" not in modal_run.container_script("a8")
    assert shim.parse_overrides(["hidden_dropout=0.0", "deterministic_mode=true"]) == {
        "hidden_dropout": 0.0, "deterministic_mode": True}
    ns = SimpleNamespace(work=str(tmp_path), phase="dry", arm=None,
                         set=["lora_dropout=0.0", "hidden_dropout=0.0", "attention_dropout=0.0"])
    args = SimpleNamespace(**{**vars(default_args(1)), "lora_dropout": 0.05})
    assert shim.make_phase(ns)(args, SimpleNamespace(argv=ARGV), _spec()) is None
    assert json.loads((tmp_path / "miles_args.dry.json").read_text())["overrides"]["lora_dropout"] == 0.0


def test_server_error_watchdog_kills_a_retrying_phase(tmp_path):
    import subprocess

    modal_run = importlib.import_module("modal_run")
    script = modal_run.container_script("dev-gather", work=str(tmp_path), flags_file="/dev/null",
                                        max_server_errors=5)
    head = script[:script.index("nvidia-smi --query")].replace("sleep 30", "sleep 1")
    head = head.replace("cd /yeto", "cd " + str(tmp_path))
    body = head.replace('bash -c "python', 'bash -c "for i in 1 2 3 4 5 6 7 8; do echo 503 Service Unavailable; '
                        'done; sleep 60; : python')
    proc = subprocess.run(["bash", "-c", body + 'run_phase gen "--phase gen"; echo NOT-REACHED'],
                          capture_output=True, text=True, timeout=60)
    assert "SERVER-ERRORS" in (tmp_path / "progress.log").read_text()
    assert "NOT-REACHED" not in proc.stdout


def test_generation_keeps_the_colocated_trainer_size():
    """DEV-GATHER run 4: a DP=1 trainer beside 2 colocated engines is refused by Miles' LoRA weight sync."""
    import ast

    src = (TOOLS / "miles_backend.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "generate_frozen")
    calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "_args"]
    assert calls and all("actor_num_gpus_per_node" not in [k.arg for k in c.keywords] for c in calls)


def test_generation_provides_the_rollout_metadata_sink():
    """DEV-GATHER run 5: the ports rollout hooks look up the named sink actor and the policy token."""
    src = (TOOLS / "miles_backend.py").read_text()
    body = src[src.index("def generate_frozen"):src.index("def start_arm")]
    assert "RayMetadataSink()" in body and "MilesRolloutPool(" in body and "metadata=sink" in body


# ---------------------------------------------------------------- production parity of the gen phase


class _Log(list):
    def rec(self, *item):
        self.append(item)


def _fake_miles_world(log, colocate=True):
    class Engine:
        def __init__(self, i):
            self.server_url, self.version = f"e{i}", None

        async def update_weight_version(self, token):
            log.rec("update_weight_version", token)
            self.version = token

        async def get_weight_version(self):
            return self.version

    engines = [Engine(0), Engine(1)]

    class Controller:
        async def onload_weights(self): log.rec("onload_weights")
        async def onload_kv(self): log.rec("onload_kv")
        async def offload_kv(self): log.rec("offload_kv")
        async def start_update_weights(self):
            log.rec("start_update_weights")
            return SimpleNamespace(rollout_engines=engines, snapshot_cell_id_to_hashes={"c0": 1, "c1": 2})
        async def end_update_weights(self, **_): log.rec("end_update_weights")
        async def abort_update_weights(self): log.rec("abort_update_weights")
        async def check_weights(self, action): log.rec("check_weights", action); return "raw"
        async def get_cell_statuses(self):
            return {"c0": SimpleNamespace(phase="Running"), "c1": SimpleNamespace(phase="Running")}
        async def prepare_rollout(self, rid): log.rec("prepare_rollout", rid)

    class Executor:
        async def get(self, rid): log.rec("executor.get", rid); return f"pack{rid}"

    class Actor:
        async def run_plugin(self, path, kwargs=None):
            log.rec("plugin", path.rsplit(".", 1)[1])
            return [None]
        async def onload(self): log.rec("trainer.onload")
        async def offload(self): log.rec("trainer.offload")
        async def clear_memory(self): log.rec("trainer.clear_memory")

    args = SimpleNamespace(colocate=colocate, offload_rollout=True, colocate_memory_peak_device="gpu",
                           offload_train=True, global_batch_size=GBS, micro_batch_size=1,
                           actor_num_nodes=1, actor_num_gpus_per_node=2)
    return Controller(), Executor(), Actor(), args


class _Sink:
    def __init__(self, log):
        self.log = log

    def set_policy_token(self, token):
        self.log.rec("sink.set_policy_token", token)

    def take(self, rid):
        self.log.rec("sink.take", rid)
        return {"rollout_id": rid}


def test_gen_phase_runs_the_production_round_components_in_driver_order(monkeypatch):
    miles_backend = importlib.import_module("miles_backend")
    from yeto.rl.adapters.miles import publish as publish_mod
    from yeto.rl.adapters.miles import rollout as rollout_mod
    from yeto.rl.adapters.miles import state as state_mod

    log = _Log()
    controller, executor, actor, args = _fake_miles_world(log)

    async def update_weights(a, act, ex, ctl):
        log.rec("update_weights")

    class State:
        def __init__(self, **kw): pass
        def export(self, policy_version=0):
            log.rec("export", policy_version)
            return SimpleNamespace(policy_version=policy_version, policy_tensor_hash=lambda: "a" * 64,
                                   lora=None)

    monkeypatch.setattr(publish_mod, "_default_update_weights", lambda: update_weights)
    monkeypatch.setattr(publish_mod, "_default_flatten", lambda: (lambda raw: [{"w": 1}, {"w": 1}]))
    monkeypatch.setattr(publish_mod, "payload_digest", lambda state: ("b" * 64, 1))
    monkeypatch.setattr(state_mod, "MilesPolicyState", State)
    monkeypatch.setattr(rollout_mod, "handle_from_metadata",
                        lambda payload, **k: SimpleNamespace(groups=[1, 2], payload=k["data_pack"],
                                                             data_cursor=None))
    backend = miles_backend.MilesBackend.__new__(miles_backend.MilesBackend)
    backend.base_args, backend.algorithm, backend.frozen_template = args, _spec(), "/f/r_{rollout_id}.pt"
    backend.runner = LoopRunner()
    backend.trainer = None

    def fake_open(a, *, trainer):
        backend._controller, backend._executor, backend._actor = controller, executor, actor
        return executor, actor

    backend._open = fake_open
    backend._close = lambda: log.rec("close")
    released = []
    monkeypatch.setattr(miles_backend, "_release_refs", lambda a, p: released.append(p), raising=False)
    from yeto.rl.adapters.miles import trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, "_default_release", lambda a, p: released.append(p))
    backend.generate_frozen(2, identity={"layout_hash": "L"}, metadata_sink=_Sink(log))
    names = [e[0] for e in log]
    first = names[:names.index("executor.get") + 1]
    assert first == ["export", "export", "update_weights", "onload_kv", "start_update_weights", "update_weight_version",
                     "update_weight_version", "end_update_weights", "check_weights", "trainer.offload",
                     "sink.set_policy_token", "prepare_rollout", "executor.get"]
    second = names[names.index("executor.get") + 1:]
    assert second[:5] == ["offload_kv", "sink.take", "trainer.onload", "export", "onload_weights"]
    assert ("update_weight_version", policy_token_for(1)) in log
    assert released == ["pack0", "pack1"] and names[-1] == "close"


def policy_token_for(rid):
    from yeto.rl.adapters.miles.rollout import policy_token

    return policy_token(rid, "a" * 64)


def test_arm_args_carry_every_parse_derivation_of_debug_train_only():
    """DEV-GATHER run 6: with --load-debug-rollout-data Miles parse also zeroes rollout_num_gpus and
    starts_inference_engines; setting only debug_train_only left engine cells beyond the placement group."""
    miles_backend = importlib.import_module("miles_backend")
    assert miles_backend.ARM_PARSE_DERIVED == {"debug_train_only": True, "rollout_num_gpus": 0,
                                               "starts_inference_engines": False}
    backend = miles_backend.MilesBackend.__new__(miles_backend.MilesBackend)
    backend.base_args = SimpleNamespace(rollout_num_gpus=2, starts_inference_engines=True, colocate=True,
                                        actor_num_gpus_per_node=2)
    backend.frozen_template, backend.algorithm, backend.runner = "/f/{rollout_id}", _spec(), LoopRunner()
    seen = {}
    backend._open = lambda a, *, trainer: (seen.update(vars(a)), (None, SimpleNamespace()))[1]
    backend.start_arm(1)
    assert seen["rollout_num_gpus"] == 0 and seen["starts_inference_engines"] is False
    assert seen["debug_train_only"] is True and seen["actor_num_gpus_per_node"] == 1
    assert backend.base_args.rollout_num_gpus == 2  # the launcher's args are not mutated


def test_g2_reads_the_fork_shard_sample_indices(tmp_path):
    """DEV-GATHER run 7: the fork's scheduled shard carries sample_indices, not partition."""
    harness.run_all(lambda arm: FakeBackend(), tmp_path)
    for arm in ("A1", "B1"):
        path = tmp_path / "arms" / arm / "events.jsonl"
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        for e in lines:
            for rank in e.get("probe") or []:
                for r in rank:
                    if r.get("kind") == "shard":
                        r["sample_indices"] = [100 + i for i in r.pop("partition")]
        path.write_text("\n".join(json.dumps(e) for e in lines) + "\n")
    dp = {"A1": 1, "A2": 2, "B1": 2, "B1p": 2, "B2": 1, "RT": 1}
    assert compare.g2(tmp_path, "A1", "B1", gbs=GBS, mbs=MBS, dp=dp) == []


def test_container_script_merges_stderr():
    modal_run = importlib.import_module("modal_run")
    script = modal_run.container_script("dev-gather")
    assert script.splitlines()[1] == "exec 2>&1" and "RESULT.json" in script


# ---------------------------------------------------------------- A8 data safety net


def test_packed_states_let_compare_run_offline(tmp_path):
    """pack_states in the container -> pulled files -> compare --offline gives the same G1-G6."""
    pack_states = importlib.import_module("pack_states")
    harness.run_all(lambda arm: FakeBackend(), tmp_path)
    online = compare.judge(tmp_path, gbs=GBS, mbs=MBS, merge=_merge, load=_load)
    index = pack_states.pack(tmp_path, _merge, _load)
    assert len(index["files"]) == 15 and not index.get("errors")
    assert index["fields"]["B1_restored"]["param"] == index["fields"]["A1_s2"]["param"]
    # offline: per-rank dumps gone, only events + packed states
    import shutil

    for arm in ("A1", "A2", "B1", "B1p", "B2", "RT"):
        shutil.rmtree(tmp_path / "arms" / arm / "state")

    def no_merge(_states):
        raise compare.Unavailable("offline")

    offline = compare.judge(tmp_path, gbs=GBS, mbs=MBS, merge=no_merge, load=_load)
    assert offline["decision"] == online["decision"] == "go" and offline["unavailable"] == []
    for g in ("G1", "G2", "G3", "G5"):
        assert offline[g] == online[g]


def test_offline_compare_on_events_only_reports_incomplete(tmp_path):
    harness.run_all(lambda arm: FakeBackend(), tmp_path)
    import shutil

    for arm in ("A1", "A2", "B1", "B1p", "B2", "RT"):
        shutil.rmtree(tmp_path / "arms" / arm / "state")
    res = compare.main([str(tmp_path), "--offline", "--gbs", str(GBS), "--mbs", str(MBS)])
    result = json.loads((tmp_path / "RESULT.json").read_text())
    assert res == 0 and result["decision"].startswith("incomplete") and result["G1"]["pass"] is None
    assert result["G2"]["pass"] and result["G3"]["pass"]


def test_pull_packed_copies_verifies_and_releases(tmp_path):
    modal_run = importlib.import_module("modal_run")
    pack_states = importlib.import_module("pack_states")
    work = tmp_path / "work"
    harness.run_all(lambda arm: FakeBackend(), work)
    pack_states.pack(work, _merge, _load)
    released = []

    import shutil as _sh

    class FakeFS:
        def _local(self, path):
            return str(path).replace("/work/e3", str(work))

        def read_text(self, path):
            return open(self._local(path)).read()

        def copy_to_local(self, remote, local):
            _sh.copyfile(self._local(remote), local)

        def write_text(self, data, path):
            assert path == modal_run.PULLED_FLAG
            released.append(path)

    class FakeSandbox:
        filesystem = FakeFS()

    import io

    log = io.StringIO()
    report = modal_run.pull_packed(FakeSandbox(), tmp_path / "out", touch=lambda: None, log=log)
    assert len(report["ok"]) == 15 and not report["bad"] and released
    assert "15 ok" in log.getvalue()


def test_resized_args_sets_world_size_with_the_trainer_size():
    from yeto.rl.adapters.miles.trainer_rebuild import resized_args

    base = SimpleNamespace(actor_num_nodes=1, actor_num_gpus_per_node=2, world_size=2, lr=1e-5)
    new = resized_args(base, 1)
    assert (new.actor_num_gpus_per_node, new.world_size) == (1, 1)
    assert (base.actor_num_gpus_per_node, base.world_size) == (2, 2) and new.lr == base.lr
    with pytest.raises(RuntimeError, match="single-node"):
        resized_args(SimpleNamespace(actor_num_nodes=2, actor_num_gpus_per_node=2), 1)


def test_a8_determinism_env_equals_production_and_precedes_ray():
    modal_run = importlib.import_module("modal_run")
    from yeto.rl.adapters.miles.entry import DETERMINISM_ENV

    assert modal_run.DETERMINISM_ENV == DETERMINISM_ENV
    script = modal_run.container_script("a8")
    assert script.index("export NCCL_ALGO=Ring") < script.index("ray start")
    assert "export NCCL_ALGO" not in modal_run.container_script("dev-gather")
