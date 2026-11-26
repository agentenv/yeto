"""``yeto image adopt`` step 5 and the command (own-cluster-fleet-stability 7.3, 7.5):
the trainer smoke (a cut model, one round, a cut and a resume) and the sampler smoke
(the node runtime's container start of yeto.node.smoke: start, generate, reload,
generate) on CPU fakes; the catalog is written only after a passing smoke; the dry
run prints the smoke's commands; the documented commands run in a dry run."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests.test_cluster_preflight import FakeCluster, b300
from tests.test_image_adopt import PINNED, TAGGED, FakeNodes, probe_out, registry
from yeto.cli import build_parser
from yeto.launch import image_adopt as ia
from yeto.launch import image_smoke as sm
from yeto.launch import model_check
from yeto.launch import site_images as si
from yeto.node import smoke as node_smoke
from yeto.rl.adapters.sampler_engines.fake import FakeEngine
from yeto.rl.engine.object_store import DirectoryObjectStore

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_CONFIGS = ROOT / "tests" / "data" / "model_registry"
QWEN = json.loads((REGISTRY_CONFIGS / "qwen3_30b_a3b.json").read_text())
K31 = json.loads((REGISTRY_CONFIGS / "kimi_k31.json").read_text())
FAKE_SPEC = {"sampler_id": "adopt", "engine": {"factory": "yeto.rl.adapters.sampler_engines.from_spec:build",
                                               "adapter": "fake", "kwargs": {}}}
REV = "f" * 40
PY = "/usr/bin/python3"
RELEASE = "/models/r1"


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("YETO_IMAGE_MANIFEST_CHECK", raising=False)
    monkeypatch.delenv(si.PATH_ENV, raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)


def node(i=0, **kw):
    return b300(i, **{"scratch": "/mnt/nvme/yeto", "image": None, **kw})


# -- the node side of the sampler smoke ----------------------------------------------


class ReloadFails(FakeEngine):
    def load(self, release_dir, version):
        raise RuntimeError("update_weights_from_disk answered 500")


def test_the_node_smoke_starts_generates_reloads_and_generates(capsys):
    engine = FakeEngine()
    record = node_smoke.run_smoke(engine, release=None)
    assert record["ok"] and record["step"] is None and record["reload_mode"] == "in_place"
    assert set(record["seconds"]) == set(node_smoke.STEPS)
    assert engine.loads == [(None, 0), (None, 1)] and engine.stopped
    lines = [json.loads(line.split(" ", 1)[1]) for line in capsys.readouterr().out.splitlines()]
    assert [r["step"] for r in lines] == list(node_smoke.STEPS) and all(r["ok"] for r in lines)


def test_a_failed_reload_names_the_step_and_stops_the_engine():
    engine = ReloadFails()
    record = node_smoke.run_smoke(engine, release="/models/r1", out=open(os.devnull, "w"))
    assert (record["ok"], record["step"]) == (False, "reload")
    assert "update_weights_from_disk answered 500" in record["error"] and engine.stopped


def test_the_node_smoke_entry_reads_the_spec(tmp_path, capsys):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(FAKE_SPEC))
    assert node_smoke.main(["--spec", str(spec)]) == 0
    assert node_smoke.parse_result(capsys.readouterr().out)["ok"] is True
    spec.write_text(json.dumps({"engine": {"factory": "yeto.rl.adapters.sampler_engines.from_spec:build",
                                           "adapter": "nope"}}))
    assert node_smoke.main(["--spec", str(spec)]) == 1
    result = node_smoke.parse_result(capsys.readouterr().out)
    assert result["step"] == "start" and "unknown sampler engine 'nope'" in result["error"]


# -- the sampler smoke through the adoption -------------------------------------------


class SmokeNodes(FakeNodes):
    """The fake cluster, plus the sampler smoke: the upload is accepted, the engine
    step runs ``python -m yeto.node.smoke`` here on the spec it uploaded."""

    def __init__(self, nodes, tmp_path, *, pythonpath=(), **kw):
        super().__init__(nodes, probe=probe_out(packages={"vllm": "0.19.0", "torch": "2.9.1"}), **kw)
        self.tmp_path = tmp_path
        self.pythonpath = pythonpath

    def run(self, node, script, *, timeout=300.0):
        if script.startswith(("docker rm -f", "rm -rf", "du -sh")):
            self.scripts.append((node.name, "cleanup-" + script.split()[0]))
            return self.done(0, "")
        if "YETO_TREE_EOF" in script:
            self.scripts.append((node.name, "upload"))
            spec = re.search(r"<<'YETO_SPEC_EOF'\n(.*)\nYETO_SPEC_EOF", script, re.S).group(1)
            (self.tmp_path / "uploaded-spec.json").write_text(spec)
            return self.done(0, "")
        if "yeto.node.smoke" in script:
            self.scripts.append((node.name, "engine"))
            argv = shlex.split(script)
            assert PINNED in argv and "--rm" in argv and "--gpus" in argv
            env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), *map(str, self.pythonpath)])}
            done = subprocess.run([sys.executable, "-m", "yeto.node.smoke", "--spec",
                                   str(self.tmp_path / "uploaded-spec.json")], capture_output=True, text=True,
                                  env=env, timeout=120)
            return self.done(done.returncode, done.stdout + done.stderr)
        return super().run(node, script, timeout=timeout)


def _adopt(nodes, tmp_path, plan, ops, role, **kw):
    opener, _ = registry({"k99": 200})
    req = ia.AdoptRequest(ref=TAGGED, model="org/new-model", role=role, nodes=tuple(nodes),
                          catalog=str(tmp_path / "site-images.yaml"), engine=plan.data.get("engine"), **kw)
    store = tmp_path / "store"

    def smoke(adopter, ref, facts, engine):
        return sm.run_plan(plan, ops=adopter.ops, ref=ref, python=facts.python_path,
                           log=getattr(ops, "logs", []).append, run=getattr(ops, "launch", subprocess.run), clock=lambda: 0.0,
                           objects_for=lambda uri: DirectoryObjectStore(store), environ={})

    return ia.Adopter(req, ops=ops, smoke=smoke, urlopen=opener, login=None, log=lambda m: None,
                      clock=lambda: FakeCluster.NOW)


def test_a_sampler_image_whose_smoke_passes_is_recorded(tmp_path):
    nodes = [node(0), node(1)]
    plan = sm.sampler_plan(nodes[0], spec=FAKE_SPEC, run_id="adopt-t", release=RELEASE, code=("t000", b"tar"))
    ops = SmokeNodes(nodes, tmp_path)
    report = _adopt(nodes, tmp_path, plan, ops, "sampler").run()
    assert report.ok and report.steps[-1].detail.startswith("engine started, generated, reloaded")
    (rec,) = si.load(tmp_path / "site-images.yaml")
    assert (rec.role, rec.engine, rec.ref, rec.card_archs) == ("sampler", "vllm", si.parse_record(
        rec.to_doc()).ref, ("sm103",))
    assert rec.evidence["smoke"]["reload_mode"] == "in_place"
    assert set(rec.evidence["smoke"]["engine_steps"]) == set(node_smoke.STEPS)
    assert ops.kinds()[-2:] == ["cleanup-docker", "cleanup-rm"]  # the container, then the run's directory


def test_a_sampler_whose_reload_fails_is_not_recorded(tmp_path):
    (tmp_path / "adopt_fail_engine.py").write_text(
        "from tests.test_image_smoke import ReloadFails\n"
        "def build(section):\n    return ReloadFails()\n")
    spec = {"sampler_id": "adopt", "engine": {"factory": "adopt_fail_engine:build"}}
    nodes = [node(0)]
    plan = sm.sampler_plan(nodes[0], spec=spec, run_id="adopt-t", release=RELEASE, code=("t000", b"tar"))
    ops = SmokeNodes(nodes, tmp_path, pythonpath=[tmp_path])
    with pytest.raises(ia.AdoptError) as error:
        _adopt(nodes, tmp_path, plan, ops, "sampler").run()
    assert error.value.step == "smoke"
    assert ops.kinds()[-2:] == ["cleanup-docker", "cleanup-rm"]  # removed although the smoke failed
    assert error.value.reason.startswith("engine: reload failed: RuntimeError: update_weights_from_disk answered 500")
    assert not (tmp_path / "site-images.yaml").exists()


def test_the_sampler_smoke_starts_like_the_node_runtime():
    from yeto.launch.node_inventory import profile_of
    from yeto.node import launch as node_launch

    n = node(0)
    plan = sm.sampler_plan(n, spec=FAKE_SPEC, run_id="adopt-t", release=RELEASE, code=("t000", b"tar"))
    engine = shlex.split(plan.steps[1].render({sm.REF: PINNED, sm.PYTHON: PY}).script)
    runtime = node_launch.container_argv(profile_of(n), name="yeto-adopt-adopt-t", image=PINNED,
                                         env={"PYTHONPATH": "/work/image-adopt/yeto/t000", "PYTHONUNBUFFERED": "1",
                                              **profile_of(n).family_env()},
                                         python=PY, spec_path="/work/image-adopt/adopt-t/spec.json")
    assert engine[:-1] == runtime[:-1]  # the same flags, mounts and environment
    assert engine[-1].endswith(f"exec {PY} -m yeto.node.smoke --spec /work/image-adopt/adopt-t/spec.json "
                               f"--release {RELEASE}")
    assert runtime[-1].endswith(f"-m {node_launch.SAMPLER_MODULE} --spec /work/image-adopt/adopt-t/spec.json")
    with pytest.raises(sm.SmokeError, match="needs a container runtime"):
        sm.sampler_plan(node(0, runtime="rootfs"), spec=FAKE_SPEC, run_id="x", release=RELEASE, code=("t", b""))
    with pytest.raises(sm.SmokeError, match="pass --release"):  # no release, no base_dir: nothing to reload
        sm.sampler_plan(n, spec=FAKE_SPEC, run_id="x", code=("t", b""))
    based = {**FAKE_SPEC, "engine": {**FAKE_SPEC["engine"], "server": {"base_dir": "/models/base"}}}
    assert sm.sampler_plan(n, spec=based, run_id="x", code=("t", b"")).steps


# -- the trainer smoke ----------------------------------------------------------------


def test_the_cut_is_the_most_layers_that_fit_one_node():
    layout = sm.smoke_layout(QWEN, node(0), engine=None)
    assert layout["engine"] == "miles" and layout["layers"] == 16 and layout["plan"]["world"] <= 8
    assert sm.smoke_layout(QWEN, node(0), engine=None, layers=4)["layers"] == 4
    small = node(0, card="L4", hbm_gib=24.0, cards_per_node=1)
    assert sm.smoke_layout(QWEN, small, engine="miles")["layers"] < 16
    with pytest.raises(sm.SmokeError, match="the model has 48 layers"):
        sm.smoke_layout(QWEN, node(0), engine=None, layers=99)


def _trainer_plan(tmp_path, **kw):
    target = kw.pop("node", node(0))
    args = {"model": "Qwen/Qwen3-30B-A3B", "config": QWEN, "engine": None, "store": "s3://bucket/smoke",
            "run_id": "adopt-t", "workdir": tmp_path / "work", "python": "py", **kw}
    return sm.trainer_plan(target, **args)


def test_the_trainer_plan_cuts_trains_and_resumes_on_one_node(tmp_path):
    plan = _trainer_plan(tmp_path)
    assert [s.name for s in plan.steps] == ["prune", "train", "resume"]
    prune, train, resume = plan.steps
    script = prune.render({sm.REF: PINNED, sm.PYTHON: PY}, {}).script
    assert "--layers 16" in script and "hf://Qwen/Qwen3-30B-A3B" in script
    assert "base64 -d" in script and sm.REVISION_MARK in script
    assert f"--cache-dir {sm.HUB_IN_CONTAINER}/.dl-Qwen3-30B-A3B-16layer" in script  # not the docker root
    assert f"rm -rf {sm.HUB_IN_CONTAINER}/.dl-Qwen3-30B-A3B-16layer" in script
    assert f"the shards it reads need {round(sm.cut_gib(QWEN, 16) * 2.2 + 1, 1)} GiB" in script
    argv = list(train.argv)
    assert argv[:4] == ["py", "-m", "yeto.cli", "launch"]
    for flag, value in (("--rl-backend", "miles"), ("--gpu", "ssh:1x8xB300"), ("--rl-image", f"docker:{sm.REF}"),
                        ("--model", "yeto-local/Qwen3-30B-A3B-16layer"), ("--model-revision", sm.REVISION),
                        ("--model-store", f"local://{sm.HUB_IN_CONTAINER}"),
                        ("--rl-checkpoint-store", "s3://bucket/smoke/adopt-t"), ("--rl-stop-after-rounds", "1")):
        assert argv[argv.index(flag) + 1] == value
    assert argv.count("--model") == 1
    for flag in ("--rl-single-island-no-sync", "--rl-allow-local-data"):
        assert flag in argv
    assert sm.TRUST in argv and plan.data["trust_remote_code"] is True  # qwen3-moe: no auto_map, but Miles needs it
    assert argv[argv.index("--controller") + 1] == "local"
    assert argv[argv.index("--reward-function") + 1] == sm.REWARD
    prompts = Path(argv[argv.index("--data") + 1])
    assert [json.loads(line)["label"] for line in plan.files[str(prompts)].splitlines()] == ["4", "8"]
    assert list(resume.argv)[-2:] == ["--rl-stop-after-rounds", "2"]
    assert (train.check, resume.check, resume.expect) == ("round_cut", "resume", None)
    synth = json.loads(next(a for a in argv if a.startswith(sm.SYNTH_ENV + "=")).split("=", 1)[1])
    assert synth["layers"] == 16 and synth["experts"] == 128 and synth["router_topk"] == 8
    text = plan.files[argv[argv.index("--rl-node-inventory") + 1]]
    (entry,) = yaml.safe_load(text)["nodes"]
    assert entry["mounts"]["/mnt/nvme/yeto/image-adopt/hub"] == sm.HUB_IN_CONTAINER and "image" not in entry
    assert entry["mounts"]["/mnt/nvme/models"] == "/data/models"  # the node's own mounts stay
    assert entry["roles"] == ["trainer"] and entry["scratch"] == "/mnt/nvme/yeto"
    assert plan.data["layers"] == 16 and plan.data["engine"] == "miles"
    assert any("16 of 48 layers" in n for n in plan.notes)
    assert any("tuning lora (the registry's; --smoke-arg=--tuning=... overrides it)" in n for n in plan.notes)
    assert plan.data["notes"] == list(plan.notes) and plan.data["tuning"] == "lora"


def test_a_node_path_takes_the_checkpoint_from_the_node(tmp_path):
    def script(**kw):
        return _trainer_plan(tmp_path, **kw).steps[0].render({sm.REF: PINNED, sm.PYTHON: PY}, {}).script

    local = script(model_path="/mnt/models/qwen3", model_revision="abc")
    assert "/mnt/models/qwen3:/mnt/models/qwen3:ro" in local and "hf://" not in local
    assert "--revision" not in local and "--cache-dir" not in local
    assert "--revision abc --cache-dir" in script(model_revision="abc")
    assert "--revision" not in script()


@pytest.mark.parametrize("kw, message", [
    ({"node": node(0, scratch=None)}, "has no scratch"),
    ({"store": None}, "pass --smoke-store"),
    ({"node": node(0, runtime="rootfs")}, "runs rootfs"),
    ({"engine": "verl"}, "no trainer smoke for backend verl"),
])
def test_the_trainer_plan_refuses_what_it_cannot_run(tmp_path, kw, message):
    with pytest.raises(sm.SmokeError, match=message):
        _trainer_plan(tmp_path, **kw)


def test_a_registry_placeholder_needs_a_smoke_arg(tmp_path):
    with pytest.raises(sm.SmokeError, match=r"need \['<expert_selection.json>'\]"):
        _trainer_plan(tmp_path, model="moonshotai/Kimi-K3.1-Pro", config=K31)
    plan = _trainer_plan(tmp_path, model="moonshotai/Kimi-K3.1-Pro", config=K31,
                         extra_args=("--rl-expert-selection=/sel.json",))
    argv = list(plan.steps[1].argv)
    assert "--rl-expert-selection=/sel.json" in argv and "--rl-expert-selection" not in argv
    assert "<expert_selection.json>" not in argv


def event(name, **fields):
    record = {"event": name, "island_id": 0, "time_unix": 1.0, **fields}
    return "[cluster] YETO_RL_EVENT " + json.dumps(record, sort_keys=True, separators=(",", ":"))


CUT = "r000001-abc"


class Launches:
    """The launcher's subprocess for the train and resume steps (the train writes its
    cut's manifest into the store ``store``), and the cleanup's ``yeto down``."""

    def __init__(self, store=None, *, resume_out=None, train_out=None, rc=0, manifest=True, timeout=False):
        self.calls, self.downs, self.store, self.rc = [], [], store, rc
        self.resume_out = event("rl_resume", cut_id=CUT) if resume_out is None else resume_out
        self.train_out = event("rl_round_cut", ok=True, cut_id=CUT) if train_out is None else train_out
        self.manifest, self.timeout = manifest, timeout

    def __call__(self, argv, **kw):
        if "down" in argv:
            self.downs.append(argv)
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        self.calls.append(argv)
        if self.timeout:
            raise subprocess.TimeoutExpired(argv, kw.get("timeout"))
        resume = argv[-1] == "2"
        if not resume and self.manifest and self.store is not None:
            cut = Path(self.store) / "island-0" / "round-cuts" / CUT
            cut.mkdir(parents=True, exist_ok=True)
            (cut / "manifest.json").write_text("{}")
        return subprocess.CompletedProcess(argv, self.rc, stdout=self.resume_out if resume else self.train_out,
                                           stderr="")


class TrainerNodes(FakeNodes):
    def __init__(self, nodes, launches, **kw):
        super().__init__(nodes, **kw)
        self.launch = launches
        self.logs = []

    def run(self, node, script, *, timeout=300.0):
        if script.startswith(("docker ps -aq", "rm -rf", "du -sh")):
            self.scripts.append((node.name, "cleanup-" + script.split()[0]))
            return self.done(0, "4.0G\t/mnt/nvme/yeto/image-adopt/hub/models--x\n" if script.startswith("du") else "")
        if "YETO_PRUNE_EOF" in script:
            self.scripts.append((node.name, "prune"))
            assert PINNED in script and "/usr/bin/python3 /tmp/yeto-prune_layers.py" in script
            return self.done(0, f"{sm.REVISION_MARK} {REV}\n")
        return super().run(node, script, timeout=timeout)


def test_a_trainer_image_whose_smoke_passes_is_recorded(tmp_path):
    nodes = [node(0), node(1)]
    plan = _trainer_plan(tmp_path)
    launches = Launches(tmp_path / "store")
    ops = TrainerNodes(nodes, launches)
    report = _adopt(nodes, tmp_path, plan, ops, "trainer").run()
    assert report.ok
    train, resume = launches.calls
    assert train[train.index("--model-revision") + 1] == REV and resume[resume.index("--rl-image") + 1] == \
        f"docker:{PINNED}"
    assert train[train.index("--rl-node-inventory") + 1] in plan.files  # written for the run, deleted after
    (rec,) = si.load(tmp_path / "site-images.yaml")
    assert (rec.role, rec.engine, rec.model) == ("trainer", "miles", "org/new-model")
    assert rec.evidence["smoke"]["layers"] == 16 and rec.evidence["smoke"]["revision"] == REV
    assert set(rec.evidence["smoke"]["steps"]) == {"prune", "train", "resume"}
    assert rec.evidence["smoke"]["cut_id"] == CUT and rec.evidence["smoke"]["tuning"] == "lora"
    assert rec.evidence["notes"] == plan.data["notes"]
    # the cleanup: the run stopped, the containers and the cut removed, the store and launcher files deleted
    assert launches.downs and launches.downs[0][-3:] == ["down", "adopt-t", "--missing-ok"]
    assert ops.kinds()[-2:] == ["cleanup-docker", "cleanup-rm"]
    assert not list((tmp_path / "store").rglob("manifest.json")) and not (tmp_path / "work").exists()


@pytest.mark.parametrize("kw, message", [
    ({"resume_out": "rl_resume: resumed from cut round 1"}, "resume: no rl_resume / rl_round_cut_restored event"),
    ({"resume_out": event("rl_round_cut_restored", cut_id="r000009-old")},
     r"resume: it resumed from \['r000009-old'\], not the cut r000001-abc"),
    ({"train_out": "round 1 cut"}, "train: no rl_round_cut event with a cut_id"),
    ({"train_out": event("rl_round_cut", ok=False, error="disk")}, "train: no rl_round_cut event"),
    ({"train_out": event("rl_round_cut", ok=False, cut_id=CUT, error="disk")}, "train: no rl_round_cut event"),
    ({"manifest": False}, "train: the store has no round-cuts/r000001-abc/manifest.json"),
    ({"rc": 1}, "train: rc 1"),
    ({"timeout": True}, "train: TimeoutExpired"),
])
def test_a_trainer_smoke_that_does_not_resume_or_train_is_not_recorded_and_cleans_up(tmp_path, kw, message):
    nodes = [node(0)]
    launches = Launches(tmp_path / "store", **kw)
    ops = TrainerNodes(nodes, launches)
    with pytest.raises(ia.AdoptError, match=message):
        _adopt(nodes, tmp_path, _trainer_plan(tmp_path), ops, "trainer").run()
    assert not (tmp_path / "site-images.yaml").exists()
    assert launches.downs and ops.kinds()[-2:] == ["cleanup-docker", "cleanup-rm"]
    assert not list((tmp_path / "store").rglob("*.json")) and not (tmp_path / "work").exists()


def test_kept_smoke_artifacts_are_named_with_their_size(tmp_path):
    nodes = [node(0)]
    launches = Launches(tmp_path / "store")
    ops = TrainerNodes(nodes, launches)
    _adopt(nodes, tmp_path, _trainer_plan(tmp_path, keep=True), ops, "trainer").run()
    assert ops.kinds()[-2:] == ["cleanup-docker", "cleanup-du"] and launches.downs
    assert list((tmp_path / "store").rglob("manifest.json")) and (tmp_path / "work").is_dir()
    assert "[image adopt] kept on b300-0: 4.0G\t/mnt/nvme/yeto/image-adopt/hub/models--x" in ops.logs
    assert any(m.startswith("[image adopt] kept in the store: s3://bucket/smoke/adopt-t (1 object(s))")
               for m in ops.logs)
    assert any(m.startswith(f"[image adopt] kept on the launcher: {tmp_path / 'work'} (") for m in ops.logs)


def test_a_prune_short_of_room_names_the_room_it_needs(tmp_path):
    class Full(TrainerNodes):
        def run(self, node, script, *, timeout=300.0):
            if "YETO_PRUNE_EOF" in script:
                return self.done(7, f"{sm.SPACE_MARK} /mnt/nvme/yeto/image-adopt/hub has 3 GiB free; the cut (5.8 GiB) "
                                    "and the shards it reads need 13.8 GiB\n")
            return super().run(node, script, timeout=timeout)
    nodes = [node(0)]
    with pytest.raises(ia.AdoptError, match=r"step smoke failed: prune: /mnt/nvme/yeto/image-adopt/hub has 3 GiB "
                                            r"free; the cut \(5.8 GiB\) and the shards it reads need 13.8 GiB"):
        _adopt(nodes, tmp_path, _trainer_plan(tmp_path), Full(nodes, Launches(tmp_path / "store")), "trainer").run()


def test_a_prune_without_a_revision_stops_the_smoke(tmp_path):
    class NoRev(TrainerNodes):
        def run(self, node, script, *, timeout=300.0):
            if "YETO_PRUNE_EOF" in script:
                return self.done(0, "cut done\n")
            return super().run(node, script, timeout=timeout)
    nodes = [node(0)]
    with pytest.raises(ia.AdoptError, match="prune: the output has no 'YETO_SMOKE_REVISION'"):
        _adopt(nodes, tmp_path, _trainer_plan(tmp_path), NoRev(nodes, Launches(tmp_path / "store")), "trainer").run()


DOCKER_SHIM = r"""#!/usr/bin/env python3
import os, subprocess, sys
args = sys.argv[1:]
assert args[:2] == ["run", "--rm"], args
mounts, i = [], 2
while args[i] != "--entrypoint":
    if args[i] == "-v":
        host, inside = args[i + 1].split(":")[:2]
        mounts.append((inside, host))
        i += 2
    else:
        i += 1
inner = args[args.index("-c") + 1]
for inside, host in sorted(mounts, key=lambda m: -len(m[0])):
    inner = inner.replace(inside, host)
sys.exit(subprocess.run(["bash", "-c", inner]).returncode)
"""


def _safetensors(path: Path, names):
    import struct

    header = {n: {"dtype": "F32", "shape": [1], "data_offsets": [4 * i, 4 * i + 4]} for i, n in enumerate(names)}
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * (4 * len(names)))


def test_the_prune_script_on_the_node_writes_a_hub_snapshot(tmp_path):
    """The prune step's node script, with ``docker`` a shim that runs the container's
    bash here (the container paths mapped to their host directories)."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "config.json").write_text(json.dumps({**QWEN, "num_hidden_layers": 48}))
    _safetensors(src / "model.safetensors", ["model.embed_tokens.weight", *[f"model.layers.{i}.w" for i in range(6)],
                                             "lm_head.weight"])
    shim = tmp_path / "bin" / "docker"
    shim.parent.mkdir()
    shim.write_text(DOCKER_SHIM)
    shim.chmod(0o755)
    n = node(0, scratch=str(tmp_path / "scratch"))
    plan = _trainer_plan(tmp_path, node=n, model_path=str(src), layers=4)
    prune = plan.steps[0].render({sm.REF: PINNED, sm.PYTHON: sys.executable}, {})
    done = subprocess.run(["bash", "-c", prune.script], capture_output=True, text=True, timeout=120,
                          env={**os.environ, "PATH": f"{shim.parent}{os.pathsep}{os.environ['PATH']}"})
    assert done.returncode == 0, done.stdout + done.stderr
    rev = re.search(rf"{sm.REVISION_MARK} ([0-9a-f]{{40}})", done.stdout).group(1)
    hub = tmp_path / "scratch" / "image-adopt" / "hub" / "models--yeto-local--Qwen3-30B-A3B-4layer"
    snapshot = hub / "snapshots" / rev
    assert (hub / "refs" / "main").read_text() == rev
    assert json.loads((snapshot / "config.json").read_text())["num_hidden_layers"] == 4
    import hashlib

    assert hashlib.sha1((snapshot / "config.json").read_bytes()).hexdigest() == rev
    kept = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    assert sorted(kept) == sorted(["model.embed_tokens.weight", "lm_head.weight",
                                   *[f"model.layers.{i}.w" for i in range(4)]])
    assert not list((tmp_path / "scratch" / "image-adopt" / "hub").glob(".cut-*"))  # the temporary cut moved


def test_a_scratch_without_room_for_the_cut_stops_the_prune(tmp_path, monkeypatch):
    shim = tmp_path / "bin" / "docker"
    shim.parent.mkdir()
    shim.write_text("#!/bin/sh\necho docker ran; exit 0\n")
    shim.chmod(0o755)
    monkeypatch.setattr(sm, "cut_gib", lambda config, layers: 1e9)
    plan = _trainer_plan(tmp_path, node=node(0, scratch=str(tmp_path / "scratch")), layers=4)
    prune = plan.steps[0].render({sm.REF: PINNED, sm.PYTHON: PY}, {})
    done = subprocess.run(["bash", "-c", prune.script], capture_output=True, text=True, timeout=60,
                          env={**os.environ, "PATH": f"{shim.parent}{os.pathsep}{os.environ['PATH']}"})
    assert done.returncode == 7 and "docker ran" not in done.stdout
    assert f"{sm.SPACE_MARK} {tmp_path / 'scratch' / 'image-adopt' / 'hub'} has" in done.stdout
    assert "the cut (1000000000.0 GiB) and the shards it reads need" in done.stdout


# -- yeto image adopt -------------------------------------------------------------------


def _inventory(tmp_path, **node_kw):
    entry = {"name": "b300-0", "host": "ops@b300-0", "device_family": "cuda", "cpu_arch": "x86_64",
             "card": "B300", "cards_per_node": 8, "hbm_gib": 288, "host_memory_gib": 2048, "runtime": "docker",
             "docker_args": ["--gpus", "all"], "scratch": "/mnt/nvme/yeto", **node_kw}
    path = tmp_path / "cluster.yaml"
    path.write_text(yaml.safe_dump({"nodes": [entry]}))
    return path


def _cli(*argv):
    return build_parser().parse_args(["image", "adopt", *map(str, argv)])


def test_the_dry_run_prints_every_step_and_the_smoke_commands_and_writes_nothing(tmp_path, capsys):
    inv = _inventory(tmp_path)
    args = _cli(TAGGED, "--model", "Qwen/Qwen3-30B-A3B", "--role", "trainer", "--inventory", inv,
                "--model-config", REGISTRY_CONFIGS / "qwen3_30b_a3b.json", "--smoke-store", "s3://b/s", "--dry-run")
    assert ia.main(args, ops=SimpleNamespace(run=None)) == 0
    out = capsys.readouterr().out
    for want in ("1. manifest:", "2. probe on b300-0: docker run --rm", "3. fit:", "4. pull:",
                 "smoke (trainer) on node b300-0", "1. prune [node]", "2. train [launcher]", "3. resume [launcher]",
                 "--rl-stop-after-rounds 1", "finally: stop [launcher]", "finally: artifacts [node]: rm -rf",
                 f"then: a verified-smoke record in {tmp_path / 'home'}"):
        assert want in out
    assert not (tmp_path / "home" / ".yeto").exists()  # no catalog, no smoke inventory
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(FAKE_SPEC))
    args = _cli(TAGGED, "--model", "org/m", "--role", "sampler", "--inventory", inv, "--sampler-spec", spec,
                "--release", RELEASE, "--dry-run")
    assert ia.main(args) == 0
    out = capsys.readouterr().out
    assert "1. upload [node]" in out and "-m yeto.node.smoke --spec" in out


@pytest.mark.parametrize("extra, message", [
    (["--role", "sampler"], "pass --sampler-spec"),
    (["--role", "trainer", "--model-config", REGISTRY_CONFIGS / "qwen3_30b_a3b.json"], "pass --smoke-store"),
    (["--role", "trainer", "--node", "b300-9", "--smoke-store", "s3://b/s"], "--node b300-9 is not a trainer node"),
])
def test_a_bad_request_is_refused_before_anything(tmp_path, capsys, extra, message):
    args = _cli(TAGGED, "--model", "Qwen/Qwen3-30B-A3B", "--inventory", _inventory(tmp_path), "--dry-run", *extra)
    assert ia.main(args) == 2
    assert message in capsys.readouterr().err


def test_a_failed_step_exits_1_and_says_the_catalog_is_unchanged(tmp_path, capsys, monkeypatch):
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps(FAKE_SPEC))
    monkeypatch.setattr(ia, "registry_login", lambda ref: None)
    monkeypatch.setattr(ia.image_manifest, "_OPENER", SimpleNamespace(open=registry({})[0]))
    args = _cli(TAGGED, "--model", "org/m", "--role", "sampler", "--inventory", _inventory(tmp_path),
                "--sampler-spec", spec, "--release", RELEASE)
    assert ia.main(args, ops=FakeNodes([node(0)])) == 1
    err = capsys.readouterr().err
    assert "step manifest failed: the registry has no manifest" in err and "is unchanged" in err


def test_the_documented_commands_run_in_a_dry_run(tmp_path, capsys):
    doc = (ROOT / "docs" / "OWN_CLUSTERS.md").read_text()
    section = doc.split("## Onboarding a model image", 1)[1].split("\n## ", 1)[0]
    commands = [" ".join(c.replace("\\\n", " ").split()) for c in re.findall(r"```\n(yeto image adopt.*?)```", section,
                                                                            re.S)[0].strip().split("\n\n")]
    assert len(commands) == 2
    inv = _inventory(tmp_path)
    spec = tmp_path / "sampler-spec.json"
    spec.write_text(json.dumps({**FAKE_SPEC, "engine": {**FAKE_SPEC["engine"], "server": {"base_dir": "/models/b"}}}))
    for command in commands:
        argv = shlex.split(command)[1:]
        argv = [str(inv) if a == "cluster.yaml" else str(spec) if a == "sampler-spec.json" else
                str(REGISTRY_CONFIGS / "qwen3_30b_a3b.json") if a == "config.json" else a for a in argv]
        args = build_parser().parse_args(argv)
        assert args.dry_run and ia.main(args) == 0, capsys.readouterr().err


class StaysAtZero(FakeEngine):
    def load(self, release_dir, version):
        return super().load(release_dir, 0)


class Silent(FakeEngine):
    def generate(self, prompts, sampling):
        import dataclasses

        return [[dataclasses.replace(c, text="") for c in group] for group in super().generate(prompts, sampling)]


def test_a_reload_that_serves_the_old_version_or_an_empty_answer_fails():
    devnull = open(os.devnull, "w")
    record = node_smoke.run_smoke(StaysAtZero(), release=None, out=devnull)
    assert (record["ok"], record["step"]) == (False, "reload") and "serves version 0" in record["error"]
    record = node_smoke.run_smoke(Silent(), release=None, out=devnull)
    assert (record["ok"], record["step"]) == (False, "generate") and "empty completion" in record["error"]


def test_the_node_smoke_entry_exits_1_on_a_failed_step(tmp_path, monkeypatch, capsys):
    (tmp_path / "adopt_fail_engine2.py").write_text(
        "from tests.test_image_smoke import ReloadFails\n"
        "def build(section):\n    return ReloadFails()\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"engine": {"factory": "adopt_fail_engine2:build"}}))
    assert node_smoke.main(["--spec", str(spec)]) == 1
    assert node_smoke.parse_result(capsys.readouterr().out)["step"] == "reload"


def test_the_sampler_smoke_reloads_the_release_given():
    plan = sm.sampler_plan(node(0), spec=FAKE_SPEC, run_id="adopt-t", release="/models/r1", code=("t000", b"tar"))
    assert shlex.split(plan.steps[1].render({sm.REF: PINNED, sm.PYTHON: PY}).script)[-1].endswith(
        "-m yeto.node.smoke --spec /work/image-adopt/adopt-t/spec.json --release /models/r1")


# -- review fixes: a launch the real parser takes, no injection, the secret, busy, engine --------------


@pytest.mark.parametrize("step", ["train", "resume"])
def test_the_trainer_smoke_launch_passes_the_real_launch_checks_in_a_dry_run(tmp_path, capsys, monkeypatch, step):
    """The rendered ``yeto launch`` of the smoke through the real parser and the
    launch's checks (``--dry-run``: no order, no node): a flag the launch comes to
    need fails here first."""
    from yeto import cli

    monkeypatch.setattr(cli, "DEFAULT_INNER_LR", cli._DefaultLR(3e-4))  # the production default (conftest unmarks it)
    monkeypatch.delenv(model_check.MODE_ENV, raising=False)  # conftest turns the model preflight off for every test

    class RepositoryNotFoundError(Exception):
        pass

    def no_hub(*_args, **_kw):  # the cut exists only on the node: the Hub does not know yeto-local/...
        raise RepositoryNotFoundError("404 yeto-local/Qwen3-30B-A3B-16layer")

    monkeypatch.setattr(model_check, "hub_files", no_hub)
    plan = _trainer_plan(tmp_path)
    for path, text in plan.files.items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)
    rendered = next(s for s in plan.steps if s.name == step).render({sm.REF: PINNED, sm.PYTHON: PY,
                                                                     sm.REVISION: REV}, {})
    argv = list(rendered.argv)
    assert argv[1:3] == ["-m", "yeto.cli"]
    try:
        rc = cli.main([*argv[3:], "--dry-run"])
    except SystemExit as exit_:
        rc = exit_.code
    out = capsys.readouterr()
    assert rc == 0, out.err[-2000:]


@pytest.mark.parametrize("ref", ["reg.example/a/b:t;touch /tmp/x", "reg.example/a/b:$(id)", "reg.example/a/b c:t",
                                 "reg.example/a/b:t`id`"])
def test_a_reference_outside_the_docker_grammar_is_refused_before_the_registry(ref):
    seen = []
    with pytest.raises(ia.AdoptError, match="is not a docker reference"):
        ia.pin(ref, urlopen=lambda req, timeout=None: seen.append(req))
    assert seen == []


def test_rendered_values_are_one_shell_word_each():
    step = sm.SmokeStep("x", "node", node_argv=("docker", "run", sm.REF, "-c", f"{sm.PYTHON_SH} -V"))
    bad_ref, bad_py = "a;touch /tmp/pwn", "/usr/bin/python3; touch /tmp/pwn"
    argv = shlex.split(step.render({sm.REF: bad_ref, sm.PYTHON: bad_py}, {}).script)
    assert argv[:3] == ["docker", "run", bad_ref]
    assert shlex.split(argv[4]) == [bad_py, "-V"]  # the interpreter is one word inside the inner shell too
    plan = sm.sampler_plan(node(0), spec=FAKE_SPEC, run_id="adopt-t", release=RELEASE, code=("t000", b"tar"))
    engine = shlex.split(plan.steps[1].render({sm.REF: bad_ref, sm.PYTHON: bad_py}, {}).script)
    assert bad_ref in engine and shlex.split(engine[-1].split("; ", 1)[1])[:2] == ["exec", bad_py]


@pytest.mark.parametrize("python", ["/usr/bin/python3; touch /tmp/pwn", "python3", "/opt/py $(id)/bin/python"])
def test_an_image_interpreter_that_is_not_a_plain_path_is_refused(tmp_path, python):
    nodes = [node(0)]
    ops = TrainerNodes(nodes, Launches(tmp_path / "store"))
    result = sm.run_plan(_trainer_plan(tmp_path), ops=ops, ref=PINNED, python=python, run=ops.launch,
                         log=lambda m: None, objects_for=lambda uri: None, environ={})
    assert not result.ok and "is not a plain absolute path" in result.detail
    assert ops.scripts == [] and ops.launch.calls == []  # nothing ran


def test_the_hf_token_goes_to_the_node_from_the_launcher_and_is_never_printed(tmp_path):
    plan = _trainer_plan(tmp_path)
    prune = plan.steps[0]
    rendered = prune.render({sm.REF: PINNED, sm.PYTHON: PY}, {"HF_TOKEN": "tok-123", "OTHER": "x"})
    assert rendered.script.startswith("export HF_TOKEN=tok-123\n") and "OTHER" not in rendered.script
    assert "-e HF_TOKEN -e HF_ENDPOINT" in rendered.script  # docker reads the value from the exported variable
    assert "tok-123" not in rendered.text() and "tok-123" not in "\n".join(plan.lines())
    assert not prune.render({sm.REF: PINNED, sm.PYTHON: PY}, {}).script.startswith("export")


class TurnsBusy(TrainerNodes):
    """Idle for the probe, busy by the smoke."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.checks = 0

    def run(self, node, script, *, timeout=300.0):
        if ia.BUSY_MARK in script:
            self.checks += 1
            self.busy = f"{ia.BUSY_MARK} tool cuda\n" + (
                f"{ia.BUSY_MARK} apps 77, python, 1000 MiB\n" if self.checks > 1 else "")
        return super().run(node, script, timeout=timeout)


def test_a_node_that_turned_busy_before_the_smoke_is_refused(tmp_path):
    nodes = [node(0)]
    launches = Launches(tmp_path / "store")
    with pytest.raises(ia.AdoptError, match="step smoke failed: node b300-0 is in use: process 77 python"):
        _adopt(nodes, tmp_path, _trainer_plan(tmp_path), TurnsBusy(nodes, launches), "trainer").run()
    assert launches.calls == [] and not (tmp_path / "site-images.yaml").exists()


def test_the_sampler_record_takes_the_spec_engine_and_a_contradicting_engine_is_refused(tmp_path):
    from tests.test_image_adopt import passing_smoke, request

    facts = ia.parse_probe(probe_out(packages={"sglang": "0.5.0", "torch": "2.9.1"}))
    req = request([node(0)], tmp_path, role="sampler", spec_engine="sglang")
    assert ia.default_engine(req, facts) == "sglang"
    problems, _, _ = ia.fit_problems(dataclasses_replace(req, engine="vllm"), PINNED, facts, "vllm")
    assert any("--engine vllm but the sampler spec runs sglang" in p for p in problems)
    nodes = [node(0)]
    ops = FakeNodes(nodes, probe=probe_out(packages={"vllm": "0.19", "sglang": "0.5.0"}))
    opener, _ = registry({"k99": 200})
    ia.Adopter(req, ops=ops, smoke=passing_smoke(), urlopen=opener, login=None, log=lambda m: None,
               clock=lambda: FakeCluster.NOW).run()
    (rec,) = si.load(tmp_path / "site-images.yaml")
    assert rec.engine == "sglang"  # the spec's, although the image also has vllm


def dataclasses_replace(obj, **kw):
    import dataclasses

    return dataclasses.replace(obj, **kw)


def test_the_cli_passes_the_spec_engine_the_arch_busy_and_keep_flags(tmp_path, monkeypatch):
    seen = {}

    class Captured:
        def __init__(self, request, **kw):
            seen["request"] = request
            seen["smoke"] = kw["smoke"]
            self.report = ia.AdoptReport(request)

        def run(self):
            raise ia.AdoptError("manifest", "stopped by the test")

    monkeypatch.setattr(ia, "Adopter", Captured)
    real_plan = sm.trainer_plan

    def recording_plan(*a, **kw):
        seen["keep"] = kw["keep"]
        return real_plan(*a, **kw)

    monkeypatch.setattr(sm, "trainer_plan", recording_plan)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({**FAKE_SPEC, "engine": {**FAKE_SPEC["engine"], "adapter": "sglang-server"}}))
    args = _cli(TAGGED, "--model", "org/m", "--role", "sampler", "--inventory", _inventory(tmp_path),
                "--sampler-spec", spec, "--release", RELEASE, "--strict-arch", "--allow-busy",
                "--keep-smoke-artifacts")
    assert ia.main(args) == 1
    req = seen["request"]
    assert (req.spec_engine, req.strict_arch, req.allow_busy) == ("sglang", True, True)
    args = _cli(TAGGED, "--model", "Qwen/Qwen3-30B-A3B", "--role", "trainer", "--inventory", _inventory(tmp_path),
                "--model-config", REGISTRY_CONFIGS / "qwen3_30b_a3b.json", "--smoke-store", "s3://b/s",
                "--keep-smoke-artifacts")
    assert ia.main(args) == 1
    assert seen["request"].engine == "miles"  # the registry's backend, the one the smoke runs
    assert seen["keep"] is True


def test_a_tuning_override_and_a_dropped_mtp_are_in_the_notes(tmp_path):
    plan = _trainer_plan(tmp_path, extra_args=("--tuning=full",))
    argv = list(plan.steps[1].argv)
    assert "--tuning=full" in argv and "--tuning" not in argv and plan.data["tuning"] == "full"
    assert any("tuning full (--smoke-arg;" in n for n in plan.notes)
    with pytest.raises(sm.SmokeError, match="--tuning half: the recipe"):
        _trainer_plan(tmp_path, extra_args=("--tuning", "half"))
    mtp = _trainer_plan(tmp_path, config={**QWEN, "num_nextn_predict_layers": 1})
    assert any("drops the model's 1 MTP layer(s)" in n for n in mtp.notes)
    assert not any("MTP" in n for n in plan.notes)
    lr = _trainer_plan(tmp_path, extra_args=("--inner-lr=3e-6",))
    assert "--inner-lr=3e-6" in lr.steps[1].argv and "1e-6" not in lr.steps[1].argv


STUB_PYTHON = r"""#!/bin/bash
# the image's interpreter on the prune: records what reached it, writes the cut's config.json
out={record}
{{ printf '%s\0' "$@"; printf 'HF_TOKEN=%s\0' "${{HF_TOKEN-}}"; printf 'HF_ENDPOINT=%s\0' "${{HF_ENDPOINT-}}"; }} > "$out"
mkdir -p "$3" && printf '{{}}' > "$3/config.json"
"""


def test_shell_words_in_the_model_revision_and_hub_env_reach_the_node_verbatim_and_run_nothing(tmp_path):
    """The rendered prune script in a real bash (``docker`` a shim running the container's
    bash here, the image's interpreter a stub recording its argv and environment): a
    ``$()``, backticks or quotes in ``--model``, ``--model-revision``, ``HF_TOKEN`` or
    ``HF_ENDPOINT`` arrive as the same bytes and run nothing."""
    pwn = tmp_path / "pwn"

    def payload(tag):
        return f"x'\"$(touch {pwn}-{tag}-a)`touch {pwn}-{tag}-b`;touch {pwn}-{tag}-c;\"'y"

    model, revision = "org/m" + payload("model"), payload("rev")
    env = {"HF_TOKEN": payload("token"), "HF_ENDPOINT": "https://hub.example/" + payload("endpoint")}
    shim = tmp_path / "bin" / "docker"
    shim.parent.mkdir()
    shim.write_text(DOCKER_SHIM)
    shim.chmod(0o755)
    record = tmp_path / "record"
    stub = tmp_path / "stub-python"
    stub.write_text(STUB_PYTHON.format(record=shlex.quote(str(record))))
    stub.chmod(0o755)
    n = node(0, scratch=str(tmp_path / "scratch"))
    plan = _trainer_plan(tmp_path, node=n, model=model, model_revision=revision, layers=4)
    prune = plan.steps[0].render({sm.REF: PINNED, sm.PYTHON: str(stub)}, env)
    assert env["HF_TOKEN"] not in prune.text()
    done = subprocess.run(["bash", "-c", prune.script], capture_output=True, text=True, timeout=60,
                          env={"PATH": f"{shim.parent}{os.pathsep}{os.environ['PATH']}", "HOME": str(tmp_path)},
                          cwd=tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("pwn")) == []
    seen = record.read_text().split("\0")[:-1]
    argv, got = seen[:-2], dict(item.split("=", 1) for item in seen[-2:])
    assert argv[1] == f"hf://{model}" and argv[argv.index("--revision") + 1] == revision
    assert got == env
    assert re.search(rf"{sm.REVISION_MARK} [0-9a-f]{{40}}", done.stdout)


def test_a_sampler_reload_that_falls_back_to_a_restart_fails_and_names_the_reload_mode(capsys):
    from yeto.rl.engine.sampler_engine import RELOAD_IN_PLACE, RELOAD_RESTART

    assert node_smoke.IN_PLACE == RELOAD_IN_PLACE
    engine = FakeEngine(in_place=False)
    record = node_smoke.run_smoke(engine, release=None)
    assert (record["ok"], record["step"], record["reload_mode"]) == (False, "reload", RELOAD_RESTART)
    assert f"reload_mode {RELOAD_RESTART}" in record["error"] and engine.stopped
    lines = [json.loads(line.split(" ", 1)[1]) for line in capsys.readouterr().out.splitlines()]
    assert lines[-1]["step"] == "reload" and not lines[-1]["ok"] and lines[-1]["reload_mode"] == RELOAD_RESTART
    assert "generate_after_reload" not in record["seconds"]


def _with(config, **over):
    return {**json.loads(json.dumps(config)), **over}


@pytest.mark.parametrize("config, family, model, why", [
    (QWEN, "qwen3-moe", False, "has no auto_map"),
    (_with(QWEN, auto_map={"AutoModelForCausalLM": "modeling_x.X"}), "qwen3-moe", True, "auto_map names remote code"),
    (_with(QWEN, text_config={"auto_map": {"AutoConfig": "c.C"}}), "qwen3-moe", True, "auto_map names remote code"),
    (QWEN, "generic-moe", True, "does not know this model"),
    (QWEN, None, True, "does not know this model"),
])
def test_the_smoke_trusts_remote_code_as_the_model_needs(config, family, model, why):
    on, notes = sm.remote_code(config, family, "verl")  # a backend that leaves it to the model
    assert on is model and why in notes[0]
    on, notes = sm.remote_code(config, family, "miles")  # pinned Miles refuses a launch without it
    assert on is True and why in notes[0] and ("launch passes it anyway" in notes[0]) is (not model)


def test_the_smoke_arg_decides_the_remote_code_and_miles_refuses_it_off(tmp_path):
    assert sm.remote_code(QWEN, "qwen3-moe", "verl", [sm.TRUST]) == (True, [
        f"the model: {sm.TRUST} on (--smoke-arg={sm.TRUST})"])
    assert sm.remote_code(_with(QWEN, auto_map={"A": "b.C"}), "generic-moe", "verl", [sm.TRUST, sm.NO_TRUST])[0] is False
    with pytest.raises(sm.SmokeError, match="launch refuses to start without --trust-remote-code"):
        _trainer_plan(tmp_path, extra_args=[sm.NO_TRUST])
    plan = _trainer_plan(tmp_path, config=_with(QWEN, architectures=["SomethingNewForCausalLM"],
                                                model_type="something_new"))
    argv = list(plan.steps[1].argv)
    assert argv.count(sm.TRUST) == 1 and sm.NO_TRUST not in argv
    assert any("does not know this model" in n for n in plan.data["notes"])
    plan = _trainer_plan(tmp_path, extra_args=[sm.TRUST])
    assert list(plan.steps[1].argv).count(sm.TRUST) == 1


def test_the_remote_code_note_reaches_the_evidence(tmp_path):
    nodes = [node(0)]
    assert _adopt(nodes, tmp_path, _trainer_plan(tmp_path), TrainerNodes(nodes, Launches(tmp_path / "store")),
                  "trainer").run().ok
    (rec,) = si.load(tmp_path / "site-images.yaml")
    assert any(n.startswith(f"the model: {sm.TRUST} off") and "launch passes it anyway" in n
               for n in rec.evidence["notes"])
