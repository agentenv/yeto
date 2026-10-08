"""rl-eval-difficulty-buckets 5.9/5.10/5.1/1.1/1.3: eval inference agent, Modal function, roles, data (CPU only)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto.rl.eval import codex_infer as ci
from yeto.rl.eval.export import StoreEvalExporter
from yeto.rl.eval.island import EvalTask
from yeto.rl.eval.store import EvalStore
from yeto.rl.eval.tb2_attempt import tb2_attempt

REPO = Path(__file__).resolve().parents[1]
MILES_SESSION_CONFIG = Path("/home/michael/work/s17-infra-fork-m/miles/rollout/session/config.py")


# --- PEFT writer pieces ---------------------------------------------------------------------


def test_rank_and_targets_from_tensor_shapes():
    shapes = {"base_model.model.layers.0.self_attn.o_proj.lora_A.weight": (16, 1024),
              "base_model.model.layers.0.self_attn.o_proj.lora_B.weight": (1024, 16),
              "base_model.model.layers.0.linear_attn.out_proj.lora_A.weight": (16, 2048),
              "base_model.model.layers.0.linear_attn.out_proj.lora_B.weight": (1024, 16)}
    assert ci.lora_rank_and_targets(shapes) == (16, ["o_proj", "out_proj"])
    with pytest.raises(ValueError, match="single LoRA rank"):
        ci.lora_rank_and_targets({**shapes, "x.q_proj.lora_A.weight": (8, 4)})


def test_peft_config_is_what_initial_adapter_accepts():
    cfg = ci.peft_adapter_config(rank=16, targets=["o_proj"], base_model="Qwen/Qwen3.5-0.8B", revision="r")
    assert cfg["peft_type"] == "LORA" and cfg["task_type"] == "CAUSAL_LM"
    assert cfg["lora_alpha"] == cfg["r"] == 16 and cfg["lora_dropout"] == 0.0 and cfg["bias"] == "none"
    assert cfg["rank_pattern"] == cfg["alpha_pattern"] == {} and not cfg["use_rslora"] and not cfg["use_dora"]


def test_write_peft_dir_checks_tensor_hash(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from yeto.rl.core import canonical_lora_config_hash, canonical_state
    from yeto.rl.engine.trainable_state import TrainableState
    from yeto.rl.eval.export import canonical_policy_files
    from yeto.rl.eval.store import EvalIntegrityError

    tensors = {"m.layers.0.self_attn.o_proj.lora_A.weight": torch.ones(4, 8),
               "m.layers.0.self_attn.o_proj.lora_B.weight": torch.zeros(8, 4)}
    state = TrainableState.from_lora(canonical_state(
        3, tensors, base_model_revision="a" * 40,
        lora_config_hash=canonical_lora_config_hash(rank=4, target_modules=["o_proj"])))
    store = EvalStore(tmp_path / "s")
    m = store.put_version(3, files=canonical_policy_files(state), policy_tensor_hash=state.policy_tensor_hash(),
                          policy_token="yeto:3:x", sampling={})
    info = ci.write_peft_dir(store.files_dir(3), m, tmp_path / "peft", base_model="B")
    assert info["rank"] == 4 and info["targets"] == ["o_proj"]
    assert json.loads((tmp_path / "peft/adapter_config.json").read_text())["revision"] == "a" * 40
    with pytest.raises(EvalIntegrityError):
        ci.write_peft_dir(store.files_dir(3), {**m, "policy_tensor_hash": "0" * 64}, tmp_path / "p2", base_model="B")


# --- SGLang / session server supervision ------------------------------------------------------


def test_sglang_argv_and_session_config_match_training():
    cfg = ci.InferConfig(base_model="/models/q", revision="r", sglang_extra_args=("--x", "1"))
    argv = ci.sglang_argv(cfg, rank=16, targets=["o_proj", "out_proj"], python="py")
    joined = " ".join(argv)
    assert "--enable-lora --max-lora-rank 16 --lora-target-modules o_proj out_proj" in joined
    assert "--context-length 8192" in joined and argv[-2:] == ["--x", "1"]
    assert "--revision" not in argv or not Path("/models/q").exists()
    sc = ci.session_server_config(cfg, rank=16)
    assert sc["tito_model"] == "qwen35" and sc["apply_chat_template_kwargs"] == {"clear_thinking": False}
    assert sc["lora_rank"] == 16 and sc["backend_url"] == "http://127.0.0.1:30000"
    assert sc["use_rollout_routing_replay"] is False and sc["session_message_matcher"] == "strict"
    if MILES_SESSION_CONFIG.is_file():  # every SessionServerConfig field, nothing else (strict model)
        text = MILES_SESSION_CONFIG.read_text().split("def compute_session_server_config")[0]
        fields = set(re.findall(r"^    (\w+): ", text, re.M))
        assert set(sc) == fields


class _Proc:
    def __init__(self):
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout=None):
        return 0


class _Http:
    def __init__(self, refuse_load=False):
        self.calls = []
        self.refuse_load = refuse_load

    def __call__(self, method, url, body=None, timeout=60.0):
        self.calls.append((method, url.split(":", 2)[-1], body))
        if url.endswith("/load_lora_adapter") and self.refuse_load:
            return 400, {"success": False, "message": "bad"}
        if url.endswith("/sessions") and method == "POST":
            return 200, {"session_id": "s1"}
        return 200, {}


def _loader(monkeypatch, tmp_path, *, rank=16, targets=("o_proj",), refuse=False):
    spawned = []

    def popen(argv, **kw):
        spawned.append(argv)
        return _Proc()

    def fake_write(files_dir, manifest, out_dir, *, base_model):
        return {"rank": rank, "targets": list(targets), "tensors": 2, "path": str(out_dir), "seconds": 0.1}

    monkeypatch.setattr(ci, "write_peft_dir", fake_write)
    events = []
    http = _Http(refuse_load=refuse)
    loader = ci.SglangSessionLoader(ci.InferConfig(base_model="B", work_dir=str(tmp_path)), popen=popen,
                                    http=http, sleep=lambda s: None,
                                    emit=lambda e, **f: events.append((e, f)))
    return loader, spawned, http, events


def test_loader_starts_once_then_swaps_adapter(monkeypatch, tmp_path):
    loader, spawned, http, events = _loader(monkeypatch, tmp_path)
    assert loader.load({"policy_version": 0, "rl/policy_token": "yeto:0:h"}, tmp_path) == "yeto:0:h"
    assert len(spawned) == 2 and "sglang.launch_server" in spawned[0]
    assert any(c[1].endswith("/health_generate") for c in http.calls)
    assert loader.load({"policy_version": 10, "rl/policy_token": "yeto:10:h"}, tmp_path) == "yeto:10:h"
    assert len(spawned) == 2  # no restart
    paths = [c[1] for c in http.calls if c[0] == "POST"]
    assert paths.count("30000/load_lora_adapter") == 2 and paths.count("30000/unload_lora_adapter") == 1
    load_bodies = [c[2] for c in http.calls if c[1].endswith("/load_lora_adapter")]
    assert {b["lora_name"] for b in load_bodies} == {ci.SERVED_LORA_NAME}
    assert [e for e, _ in events] == ["rl_eval_policy_load"] * 2 and "sglang_start_s" in events[0][1]
    loader.close()
    assert all(p is not None for p in loader.procs.values())


def test_loader_fails_closed(monkeypatch, tmp_path):
    loader, *_ = _loader(monkeypatch, tmp_path, refuse=True)
    with pytest.raises(RuntimeError, match="refused the adapter"):
        loader.load({"policy_version": 0, "rl/policy_token": "t"}, tmp_path)
    loader, *_ = _loader(monkeypatch, tmp_path)
    loader.load({"policy_version": 0, "rl/policy_token": "t"}, tmp_path)
    monkeypatch.setattr(ci, "write_peft_dir", lambda *a, **k: {"rank": 8, "targets": ["o_proj"], "path": "p",
                                                               "seconds": 0})
    with pytest.raises(RuntimeError, match="differs from the served"):
        loader.load({"policy_version": 1, "rl/policy_token": "t"}, tmp_path)


def test_loader_reports_a_dead_server(monkeypatch, tmp_path):
    loader, *_ = _loader(monkeypatch, tmp_path)

    def popen(argv, **kw):
        p = _Proc()
        p.returncode = 1
        return p

    loader.popen = popen
    with pytest.raises(RuntimeError, match="sglang exited"):
        loader.load({"policy_version": 0, "rl/policy_token": "t"}, tmp_path)


# --- the agent port -----------------------------------------------------------------------------


def _lease(tmp_path, instruction="Do the thing."):
    task = SimpleNamespace(instruction=instruction)
    return SimpleNamespace(env_url="http://127.0.0.1:9", env_token="tok", deadline_seconds=5.0,
                           worker_env={"W": "1"}, task=task)


def _agent(drive):
    http = _Http()
    return ci.CodexEvalAgent(ci.InferConfig(base_model="B", top_k=20), drive=drive, http=http), http


def test_agent_runs_the_training_worker_job(tmp_path):
    seen = {}

    async def drive(job, trajectory, board, worker_env):
        seen.update(job=job, board=board, worker_env=worker_env)
        return {"status": "max_turns", "metrics": {"turns": 12, "max_model_total_tokens": 7000},
                "episode_id": job["episode_id"]}

    agent, http = _agent(drive)
    out = asyncio.run(agent(_lease(tmp_path), EvalTask("a", "tb2-easy"), 0, policy_token="t"))
    job = seen["job"]
    assert job["base_url"] == "http://127.0.0.1:30100/sessions/s1" and job["prompt"] == "Do the thing."
    assert job["request_kwargs"] == {"temperature": 1.0, "top_p": 1.0, "max_tokens": 4096, "top_k": 20}
    assert job["max_seq_len"] == 8192 and job["env_token"] == "tok" and job["driver"] == "stock"
    assert seen["worker_env"] == {"W": "1"} and seen["board"] is None
    assert out["end_reason"] == "max_turns" and out["turns"] == 12 and out["tokens"] == 7000
    create = [c for c in http.calls if c[1].endswith("/sessions")][0]
    assert create[2]["evaluation"] is True and create[2]["top_k"] == 20
    assert http.calls[-1][0] == "DELETE"


def test_agent_maps_statuses_and_harness_errors(tmp_path):
    from yeto.rl.harness.codex import codex_harness_agent as harness

    async def timeout_drive(job, *a):
        return {"status": "timeout", "metrics": {}, "episode_id": job["episode_id"]}

    agent, _ = _agent(timeout_drive)
    assert asyncio.run(agent(_lease(tmp_path), EvalTask("a", "x"), 0, policy_token="t"))["end_reason"] == "timed_out"

    async def broken(job, *a):
        raise harness.CodexHarnessError("protocol violation")

    agent, http = _agent(broken)
    out = asyncio.run(agent(_lease(tmp_path), EvalTask("a", "x"), 0, policy_token="t"))
    assert out["end_reason"] == "infra_error" and "protocol" in out["error"]
    assert http.calls[-1][0] == "DELETE"  # session released on error too

    async def hang(job, *a):
        await asyncio.sleep(10)

    agent, _ = _agent(hang)
    lease = _lease(tmp_path)
    lease.deadline_seconds = 0.05
    assert asyncio.run(agent(lease, EvalTask("a", "x"), 0, policy_token="t"))["end_reason"] == "timed_out"


def test_tb2_attempt_records_timing_and_results_hash_ignores_it(tmp_path):
    class Verifier:
        async def evaluate(self, episode_id):
            return {"passed": True}

    class Provider:
        async def acquire(self, task_id, tid):
            async def destroy():
                return None
            return SimpleNamespace(verifier=Verifier(), destroy=destroy)

    async def agent(lease, task, trial, *, policy_token):
        return {"episode_id": "e", "end_reason": "completed", "turns": 2}

    out = tb2_attempt(Provider(), agent)(EvalTask("a", "x"), 0, policy_version=0, policy_token="t")
    assert set(out["timing"]) == {"sandbox_s", "agent_s", "judge_s", "destroy_s"}
    from yeto.rl.eval.island import EvalIsland

    rec = EvalIsland._record(SimpleNamespace(island_id="e"), EvalTask("a", "tb2-easy"), 0, 0, "t", out, 1.0)
    assert rec["timing"] == out["timing"]
    s1, s2 = EvalStore(tmp_path / "1"), EvalStore(tmp_path / "2")
    s1.append_unit(rec)
    s2.append_unit({**rec, "timing": {"agent_s": 99.0}})
    assert s1.results_sha256(0) == s2.results_sha256(0)


# --- training-finished marker (5.10) --------------------------------------------------------------


def test_final_export_marks_training_finished(tmp_path):
    store = EvalStore(tmp_path)
    exp = StoreEvalExporter(store, sampling={}, interval=10,
                            policy_files=lambda s: {"policy.safetensors": b"w"})
    state = SimpleNamespace(policy_tensor_hash=lambda: "h")
    exp(10, state, "yeto:10:h")
    assert not store.training_finished()
    exp(13, state, "yeto:13:h", final=True)
    assert store.training_finished()
    assert json.loads((tmp_path / "training-finished.json").read_text()) == {"final_version": 13}


# --- Modal function definition (fake modal) ------------------------------------------------------


def test_eval_function_spec_env_and_client():
    from yeto.cloud import modal_eval_island as mei

    spec = mei.EvalFunctionSpec(app_name="a", volume="vol", workdir=".", codex_dir="c", tb2_dir="t",
                                envs={"YETO_CODEX_VERSION": "x"})
    env = spec.container_env()
    assert env["YETO_RL_ISLAND_ROLE"] == "eval" and env["YETO_RL_EVAL_STORE"] == mei.EVAL_STORE_MOUNT
    assert env["YETO_RL_EVAL_STORE_MOUNTED"] == "1" and env["YETO_RL_EVAL_VOLUME"] == "vol"
    assert env["YETO_HARNESS_TB2_TASKS_DIR"] == mei.TB2_MOUNT and env["YETO_CODEX_VERSION"] == "x"
    assert spec.image_ref.startswith("ghcr.io/") and "@sha256:" in spec.image_ref

    class Call:
        def get(self):
            return {"evaluated": [0]}

    class Fn:
        def __init__(self):
            self.args = []

        def spawn(self, cfg):
            self.args.append(cfg)
            return Call()

    fn = Fn()
    events = []
    launcher = mei.EvalIslandLauncher(client=mei.ModalFunctionClient(fn), emit=lambda e, **f: events.append(e),
                                      config={"store": "s"}, gpu="H100!", gpus=1)
    assert launcher.run() == {"evaluated": [0]}
    assert fn.args == [{"store": "s"}] and events == ["rl_eval_island", "rl_eval_island"]


# --- roles: launcher + island ledger (5.1) ---------------------------------------------------------


def test_eval_role_never_joins_the_merge_pool():
    from yeto.launcher import ISLAND_ROLES, island_role_env
    from yeto.rl.engine.island_ledger import CrossIslandLedger, LedgerError
    from yeto.rl.eval.island import EVAL_ISLAND_ROLE

    assert EVAL_ISLAND_ROLE in ISLAND_ROLES and island_role_env("eval") == {"YETO_RL_ISLAND_ROLE": "eval"}
    with pytest.raises(ValueError):
        island_role_env("judge")
    ledger = CrossIslandLedger(mode="elastic")
    ledger.join("i0", now=0.0)
    with pytest.raises(LedgerError, match="cannot join the merge pool"):
        ledger.join("eval-0", now=0.0, role="eval")
    assert set(ledger.members) == {"i0"}


# --- eval / train jsonl (1.1 second half, 1.3) -------------------------------------------------------


def _fake_tb2(root: Path, spec):
    for tid, diff in spec:
        d = root / tid
        (d / "tests").mkdir(parents=True)
        (d / "tests/test.sh").write_text("echo ok\n")
        (d / "instruction.md").write_text(f"solve {tid}\n")
        (d / "task.toml").write_text(f'[metadata]\ndifficulty = "{diff}"\n[environment]\ndocker_image = "img/{tid}"\n')


def test_build_eval_data_split_and_determinism(tmp_path):
    import importlib.util

    from yeto.rl.harness.reward_env.benchmark import HOLDOUT_SCHEMA

    mod_spec = importlib.util.spec_from_file_location("build_eval_data", REPO / "tools/build_eval_data.py")
    tool = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(tool)
    _fake_tb2(tmp_path / "tb2", [("a", "easy"), ("b", "medium"), ("c", "hard"), ("d", "medium")])
    hold = {"schema": HOLDOUT_SCHEMA, "benchmark": "tb2", "benchmark_version": "tb2@unknown", "seed": 1,
            "rule": "t", "items": [{"task_id": "b", "difficulty": "medium", "eval_bucket": "tb2-medium"},
                                   {"task_id": "c", "difficulty": "hard", "eval_bucket": "tb2-hard"}]}
    out = tool.build(tmp_path / "tb2", hold)
    assert out == tool.build(tmp_path / "tb2", hold)  # same inputs, same bytes
    ev = [json.loads(x) for x in out[tool.EVAL_JSONL].decode().splitlines()]
    tr = [json.loads(x) for x in out[tool.TRAIN_JSONL].decode().splitlines()]
    assert [r["metadata"]["task_id"] for r in ev] == ["b", "c"] and [r["metadata"]["task_id"] for r in tr] == ["a", "d"]
    assert ev[1]["metadata"] == {"task_id": "c", "benchmark": "tb2", "benchmark_version": "tb2@unknown",
                                 "difficulty": "hard", "difficulty_source": "tb2-task.toml", "eval_bucket": "tb2-hard"}
    assert "eval_bucket" not in tr[0]["metadata"] and tr[0]["metadata"]["difficulty"] == "easy"
    sha = json.loads(out[tool.SHA_JSON])
    assert sha[tool.EVAL_JSONL]["sha256"] == hashlib.sha256(out[tool.EVAL_JSONL]).hexdigest()
    with pytest.raises(ValueError, match="checkout"):
        tool.build(tmp_path / "tb2", {**hold, "benchmark_version": "tb2@other"})


def test_shipped_eval_and_train_jsonl_match_their_sha_record():
    sha = json.loads((REPO / "data/eval/tb2-data.sha256.json").read_text())
    for rel, n in (("eval/tb2-holdout-eval.jsonl", 30), ("tb2/tb2-train.jsonl", 59)):
        data = (REPO / "data" / rel).read_bytes()
        assert sha[rel] == {"rows": n, "sha256": hashlib.sha256(data).hexdigest()}
    hold = json.loads((REPO / "data/eval/tb2-holdout.json").read_text())
    ev = {json.loads(x)["metadata"]["task_id"] for x in (REPO / "data/eval/tb2-holdout-eval.jsonl").read_text().splitlines()}
    tr = {json.loads(x)["metadata"]["task_id"] for x in (REPO / "data/tb2/tb2-train.jsonl").read_text().splitlines()}
    assert ev == {i["task_id"] for i in hold["items"]} and not ev & tr
    assert {e["task_id"] for e in hold["excluded"]} <= tr  # smoke-6: out of the eval pool, still trainable


def test_eval_island_body_keeps_logs_off_the_volume_until_the_end(tmp_path, monkeypatch):
    import subprocess

    from yeto.cloud import modal_eval_island as mei

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="NVIDIA H100 80GB HBM3\n"))
    seen = {}

    def fake_main(config):
        seen["infer"] = config["infer"]
        Path(config["infer"]["log_dir"]).mkdir(parents=True)
        (Path(config["infer"]["log_dir"]) / "sglang.log").write_text("up")
        return {"evaluated": [0], "loads": []}

    monkeypatch.setattr(mei, "island_main", fake_main)
    out = mei.eval_island_body({"store": str(tmp_path / "vol"), "island_id": "e1", "assert_gpu_name": "H100",
                                "local_run_dir": str(tmp_path / "local"), "infer": {"base_model": "B"}})
    assert out["evaluated"] == [0] and out["gpu_names"] == ["NVIDIA H100 80GB HBM3"]
    assert seen["infer"]["log_dir"].startswith(str(tmp_path / "local"))
    assert (tmp_path / "vol/runs/e1/logs/sglang.log").read_text() == "up"
    assert (tmp_path / "vol/runs/e1/body.json").is_file()
    with pytest.raises(RuntimeError, match="asked for A100"):
        mei.eval_island_body({"store": str(tmp_path / "vol"), "assert_gpu_name": "A100",
                              "local_run_dir": str(tmp_path / "l2")})
