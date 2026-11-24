"""The node runtime (``python -m yeto.node run``) end to end on a local store with the
fake engine: assignment by token, code by hash, model by source, spec paths from the
profile, the sampler as a child; the node records a control plane reads alone."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from yeto.launch.node_inventory import NodePaths, NodeProfile
from yeto.node import code as node_code
from yeto.node import records
from yeto.node.assign import assign_manual, assign_member, node_command
from yeto.node.paths import fill_spec, host_path
from yeto.node.profile import profile_doc, profile_from_doc
from yeto.node.runtime import EXIT_ASSIGNMENT, EXIT_PREFLIGHT, NodeRuntime
from yeto.node.__main__ import main as node_main
from yeto.rl.engine.object_store import DirectoryObjectStore
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer
from yeto.rl.engine.sampler_resume import MEMBERS_DIR

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def code_tar():
    return node_code.tree_code_tar(REPO)


def _profile(tmp_path: Path, **over) -> NodeProfile:
    values = dict(name="box0", host="", device_family="cuda", cpu_arch="x86_64", card="L4", cards_per_node=1,
                  hbm_gib=24.0, host_memory_gib=0.0, container="process", model_source="store",
                  paths=NodePaths(work=str(tmp_path / "work"), models=str(tmp_path / "models")))
    values.update(over)
    return NodeProfile(**values)


def _spec(shard: str, rounds: int = 2, **over) -> dict:
    spec = {
        "sampler_id": shard,
        "engine": {"factory": "yeto.rl.adapters.sampler_engines.from_spec:build", "adapter": "fake",
                   "kwargs": {"max_model_len": 64}},
        "sampling": {"max_tokens": 6, "n": 2},
        "num_rollouts": rounds, "max_backlog_groups": 100, "max_prompt_tokens": 32,
        "store": "__STORE__",
        "prompts": {"path": "/work/prompts.jsonl", "input_key": "q", "label_key": "a", "groups_per_round": 2},
        "reward": "yeto.rl.numeric_answer_reward:score", "reward_kind": "miles",
        "checks": {"require_fixed_answers": False},
        "event_tape": f"/work/sampler/events-{shard}.jsonl",
    }
    spec.update(over)
    return spec


def _setup(tmp_path: Path, code_tar, shard: str = "s0", **spec_over):
    store = tmp_path / "store"
    objects = DirectoryObjectStore(store)
    code = node_code.publish_code(objects, *code_tar)
    profile = _profile(tmp_path)
    work = Path(profile.paths.work)
    work.mkdir(parents=True)
    (work / "prompts.jsonl").write_text("".join(json.dumps({"q": f"question {i}", "a": str(i)}) + "\n"
                                                for i in range(6)))
    model_src = store / "models" / "tiny"
    model_src.mkdir(parents=True)
    (model_src / "config.json").write_text("{}")
    spec = _spec(shard, **spec_over)
    spec["store"] = str(store)
    return objects, code, profile, spec


def test_profile_round_trips_through_the_assignment_json(tmp_path):
    from yeto.launch.node_inventory import Reach
    profile = _profile(tmp_path, card_arch="sm89", reach=Reach("outbound"))
    doc = json.loads(json.dumps(profile_doc(profile)))
    back = profile_from_doc(doc)
    assert back == profile and back.reach.mode == "outbound" and back.family == "cuda"


def test_fill_spec_moves_container_paths_for_a_process_and_keeps_them_in_a_container(tmp_path):
    spec = {"prompts": {"path": "/work/p.jsonl"}, "engine": {"server": {"model": "/models/m"}},
            "event_tape": "__YETO_WORK__/sampler/e.jsonl", "store": "/srv/store", "x": "runs/__RUN_PREFIX__"}
    proc = fill_spec(spec, _profile(tmp_path), run_prefix="r1")
    assert proc["prompts"]["path"] == f"{tmp_path}/work/p.jsonl"
    assert proc["engine"]["server"]["model"] == f"{tmp_path}/models/m"
    assert proc["event_tape"] == f"{tmp_path}/work/sampler/e.jsonl" and proc["store"] == "/srv/store"
    assert proc["x"] == "runs/r1"
    box = fill_spec(spec, _profile(tmp_path, container="docker"))
    assert box["prompts"]["path"] == "/work/p.jsonl" and box["event_tape"] == "/work/sampler/e.jsonl"
    assert host_path("/work/sampler/s.json", _profile(tmp_path, container="docker")) == \
        f"{tmp_path}/work/sampler/s.json"


def test_publish_code_is_once_per_hash_and_ensure_code_checks_the_digest(tmp_path, code_tar):
    objects = DirectoryObjectStore(tmp_path / "store")
    sha, tar = code_tar
    first = node_code.publish_code(objects, sha, tar)
    assert node_code.publish_code(objects, sha, tar) == first
    tree = node_code.ensure_code(objects, first, str(tmp_path / "work"))
    assert (Path(tree) / "yeto" / "node" / "runtime.py").is_file()
    bad = records.CodeRef(sha="other", key=first.key, sha256="0" * 64)
    with pytest.raises(node_code.CodeError, match="sha256"):
        node_code.ensure_code(objects, bad, str(tmp_path / "work2"))


def test_a_wrong_token_reads_no_assignment(tmp_path, code_tar):
    objects, code, profile, spec = _setup(tmp_path, code_tar)
    assignment, token = records.assign(objects, ticket="t1", shard="s0", spec=spec, code=code)
    assert records.read_assignment(objects, "assignments/t1.json", token).shard == "s0"
    with pytest.raises(records.AssignmentError, match="token"):
        records.read_assignment(objects, "t1", "not-it")
    again, token2 = records.assign(objects, ticket="t1", shard="s0", spec=spec, code=code)
    assert again.generation == 2 and token2 != token


def test_node_runtime_end_to_end_with_the_fake_engine(tmp_path, code_tar):
    objects, code, profile, spec = _setup(tmp_path, code_tar)
    member = type("M", (), {"ticket": "t-e2e", "shard": "s0", "replaces_shard": None, "data_slot": None})()
    cfg = {"run_prefix": "run1", "model_pull": {"src_key": "models/tiny", "dest": "/models/tiny"}}
    assignment, token = assign_member(objects, cfg, member, ["s0"], profile, code, source="ssh", spec=spec)
    assert assignment.model.source == "store"
    environ = {k: v for k, v in __import__("os").environ.items() if not k.startswith("YETO_")}
    environ[records.TOKEN_ENV] = token
    rt = NodeRuntime(objects, records.read_assignment(objects, "t-e2e", token), profile, store_uri=None,
                     environ=environ, heartbeat_s=0.2, skip_preflight=True)
    assert rt.main() == 0, Path(tmp_path / "work" / "logs" / "s0.log").read_text()[-3000:]
    # code by hash, model copied by source, spec written with this node's paths
    assert (tmp_path / "work" / "yeto" / code.sha / "yeto").is_dir()
    assert (tmp_path / "models" / "tiny" / "config.json").is_file()
    written = json.loads((tmp_path / "work" / "sampler" / "s0.json").read_text())
    assert written["prompts"]["path"] == f"{tmp_path}/work/prompts.jsonl" and written["start_on_latest"] is True
    # the sampler wrote its groups and its member record names the machine
    buffer = FileSampleBuffer(CheckpointStore(tmp_path / "store"), batch_size=1)
    buffer.refresh()
    assert len(buffer) == 4
    [member_record] = [json.loads(p.read_text()) for p in (tmp_path / "store" / MEMBERS_DIR / "s0").glob("*.json")]
    assert member_record["node"]["ticket"] == "t-e2e" and member_record["node"]["source"] == "ssh"
    assert member_record["node"]["profile"]["card"] == "L4" and member_record["leave_reason"] == "FINISHED"
    # the store alone says ready, then exited with the reason
    status = records.node_status(objects, "t-e2e")
    assert status.state == "exited" and status.exit_code == 0 and status.reason == "finished"
    assert status.ready is not None and status.ready["shard"] == "s0"
    assert status.heartbeat["stage"] == "left" and status.heartbeat["leave_reason"] == "FINISHED"
    assert records.probe_of(status).sampler == "exited" and records.probe_of(status).joined


def test_the_cli_runs_from_a_token_in_the_environment(tmp_path, code_tar, monkeypatch):
    objects, code, profile, spec = _setup(tmp_path, code_tar, shard="s1", rounds=1)
    spec["num_rollouts"] = 1
    _, assignment, token = assign_manual(objects, {"run_prefix": "run1"}, shard="s1", shards=["s1"], profile=profile,
                                         code=code, run_id="run1", spec=spec)
    assert assignment.ticket.startswith("manual-run1-box0-") and assignment.source == "manual"
    for key in list(__import__("os").environ):
        if key.startswith("YETO_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv(records.TOKEN_ENV, "wrong")
    argv = ["run", "--assignment", assignment.ticket, "--store", str(tmp_path / "store"), "--skip-preflight"]
    assert node_main(argv) == EXIT_ASSIGNMENT
    monkeypatch.setenv(records.TOKEN_ENV, token)
    assert node_main([*argv, "--dry-run"]) == 0
    assert node_main(argv) == 0
    assert records.node_status(objects, assignment.ticket).reason == "finished"
    assert "--assignment" in node_command(assignment, "s3://b/runs/r") and token not in node_command(assignment, "x")


def test_a_refused_preflight_is_an_exit_record_with_the_reason(tmp_path, code_tar):
    objects, code, profile, spec = _setup(tmp_path, code_tar)
    assignment, token = records.assign(objects, ticket="t-pf", shard="s0", spec=spec, code=code,
                                       profile=profile_doc(profile), preflight={"min_free_gib": 10 ** 9})
    rt = NodeRuntime(objects, assignment, profile, store_uri=None)
    assert rt.main() == EXIT_PREFLIGHT
    status = records.node_status(objects, "t-pf")
    assert status.state == "exited" and status.reason == "preflight" and "disk" in status.exit["detail"]
    assert not (tmp_path / "work" / "yeto").exists()  # nothing fetched after a refusal


def test_dry_run_of_a_cuda_container_picks_the_catalog_image_and_cuda_flags(tmp_path, code_tar, capsys):
    objects, code, _, spec = _setup(tmp_path, code_tar)
    spec["engine"] = {"factory": "x:y", "adapter": "vllm-server", "server": {"port": 8000}}
    profile = _profile(tmp_path, container="docker", card="H200", cards_per_node=8, hbm_gib=141.0)
    assignment, token = records.assign(objects, ticket="t-dr", shard="s0", spec=spec, code=code,
                                       profile=profile_doc(profile))
    rt = NodeRuntime(objects, assignment, profile, store_uri="s3://bucket/runs/r1", environ={})
    assert rt.main(dry_run=True) == 0
    plan = json.loads(capsys.readouterr().out)
    argv = plan["argv"]
    assert argv[:4] == ["docker", "run", "--rm", "--name"] and "--gpus" in argv and "/dev/kfd" not in " ".join(argv)
    assert not any(a.startswith("HIP_") for a in argv) and "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" in argv
    assert f"{tmp_path}/work:/work" in argv and "YETO_RL_STORE_S3_URI=s3://bucket/runs/r1" in argv
    assert records.node_status(objects, "t-dr").state == "assigned"  # a dry run writes nothing


def test_an_incompatible_image_is_refused_before_the_sampler(tmp_path, code_tar):
    objects, code, _, spec = _setup(tmp_path, code_tar)
    spec["engine"] = {"factory": "x:y", "adapter": "vllm-server", "options": {"k31_preset": "x"},
                      "server": {"in_place_reload": True}}
    profile = _profile(tmp_path, container="docker", card="H200", cards_per_node=8, hbm_gib=141.0)
    assignment, _ = records.assign(objects, ticket="t-img", shard="s0", spec=spec, code=code)
    rt = NodeRuntime(objects, assignment, profile, store_uri=None, skip_preflight=True)
    assert rt.main() == 8
    assert records.node_status(objects, "t-img").reason == "image"


def test_python_m_yeto_node_has_a_run_command():
    out = subprocess.run([sys.executable, "-m", "yeto.node", "run", "--help"], capture_output=True, text=True,
                         check=True, cwd=REPO)
    assert "--assignment" in out.stdout and "--store" in out.stdout


def test_the_token_stays_out_of_the_samplers_environment(tmp_path, code_tar):
    objects, code, profile, spec = _setup(tmp_path, code_tar)
    assignment, token = records.assign(objects, ticket="t-tok", shard="s0", spec=spec, code=code)
    seen = {}

    class Child:
        def __init__(self, argv, env, **kw):
            seen.update(env)

        def wait(self):
            return 0

    rt = NodeRuntime(objects, assignment, profile, store_uri=None, skip_preflight=True, popen=Child,
                     environ={"YETO_NODE_TOKEN": token, "PATH": "/usr/bin"})
    assert rt.main() == 0
    assert "YETO_NODE_TOKEN" not in seen and seen["PATH"] == "/usr/bin"


def test_a_stop_during_the_boot_writes_the_exit_record(tmp_path, code_tar):
    import os
    import signal

    objects, code, profile, spec = _setup(tmp_path, code_tar)
    assignment, _ = records.assign(objects, ticket="t-stop", shard="s0", spec=spec, code=code)
    rt = NodeRuntime(objects, assignment, profile, store_uri=None, skip_preflight=True)
    rt.model = lambda: os.kill(os.getpid(), signal.SIGTERM) or {}  # SIGTERM while the model is prepared
    assert rt.main() == 128 + signal.SIGTERM
    status = records.node_status(objects, "t-stop")
    assert status.state == "exited" and status.reason == "stopped"
    assert signal.getsignal(signal.SIGTERM) is not None  # the handler before main is back
