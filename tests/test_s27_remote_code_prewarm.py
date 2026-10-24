"""S27 G-CK: the remote-code module cache is filled once per island node before engines start."""

from __future__ import annotations

import inspect
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from yeto.rl.adapters.miles import remote_code_prewarm as rcp

ROOT = Path(__file__).resolve().parents[1]

CONFIG_PY = '''
from transformers import PretrainedConfig

from .helpers_fake import HIDDEN


class FakeHConfig(PretrainedConfig):
    model_type = "fake_h"

    def __init__(self, hidden_size=HIDDEN, **kwargs):
        self.hidden_size = hidden_size
        super().__init__(**kwargs)
'''
HELPERS_PY = "HIDDEN = 7\n"
IMAGE_PY = '''
from transformers.image_processing_utils import BaseImageProcessor


class FakeOmniImageProcessor(BaseImageProcessor):
    model_input_names = ["pixel_values"]
'''


def _model_dir(root: Path, *, processor: bool = False, auto_map: bool = True) -> Path:
    """A local snapshot-like directory (base name = a commit hash, as in the HF cache)."""
    model = root / "snapshots" / "bf77c3174f68ad409e1c2aa60daeb46e32d1c606"
    model.mkdir(parents=True)
    config = {"model_type": "fake_h", "hidden_size": 7}
    if auto_map:
        config["auto_map"] = {"AutoConfig": "configuration_fake_h.FakeHConfig",
                              "AutoModelForCausalLM": "modeling_fake_h.FakeHForCausalLM"}
    (model / "config.json").write_text(json.dumps(config))
    (model / "configuration_fake_h.py").write_text(CONFIG_PY)
    (model / "helpers_fake.py").write_text(HELPERS_PY)
    if processor:
        (model / "image_processing_fake_omni.py").write_text(IMAGE_PY)
        (model / "processor_config.json").write_text(json.dumps(
            {"auto_map": {"AutoProcessor": "processing_fake_omni.FakeOmniProcessor"}}))
        (model / "preprocessor_config.json").write_text(json.dumps(
            {"auto_map": {"AutoImageProcessor": "image_processing_fake_omni.FakeOmniImageProcessor"}}))
    return model


# ---------------------------------------------------------------------------
# which loaders run
# ---------------------------------------------------------------------------
def test_loader_plan_covers_config_processor_and_image_processor(tmp_path):
    nano = _model_dir(tmp_path / "nano")
    assert rcp.auto_map_classes(nano) == {"AutoConfig", "AutoModelForCausalLM"}
    assert rcp.loader_plan(nano) == ("AutoConfig",)

    vl = _model_dir(tmp_path / "vl", processor=True)
    assert rcp.loader_plan(vl) == ("AutoConfig", "AutoProcessor", "AutoImageProcessor")

    (vl / "tokenizer_config.json").write_text(json.dumps({"auto_map": ["tokenization_x.T", None]}))
    assert rcp.loader_plan(vl) == ("AutoConfig", "AutoTokenizer", "AutoProcessor", "AutoImageProcessor")


def test_no_auto_map_means_no_prewarm(tmp_path):
    plain = _model_dir(tmp_path, auto_map=False)
    assert rcp.loader_plan(plain) == ()
    assert rcp.engine_model_paths(SimpleNamespace(hf_checkpoint=str(plain))) == []
    assert rcp.engine_model_paths(SimpleNamespace(hf_checkpoint=str(tmp_path / "missing"))) == []


def test_engine_model_paths_follow_miles_rollout_checkpoint(tmp_path):
    trainer = _model_dir(tmp_path / "bf16")
    rollout = _model_dir(tmp_path / "fp8")
    args = SimpleNamespace(hf_checkpoint=str(trainer), rollout_hf_checkpoint=str(rollout))
    assert rcp.engine_model_paths(args) == [str(rollout), str(trainer)]
    same = SimpleNamespace(hf_checkpoint=str(trainer), rollout_hf_checkpoint=None)
    assert rcp.engine_model_paths(same) == [str(trainer)]


# ---------------------------------------------------------------------------
# child: every *.py cached, loaders run, failures recorded (fakes)
# ---------------------------------------------------------------------------
def test_warm_model_dir_caches_every_module_and_keeps_going_after_a_failure(tmp_path):
    model = _model_dir(tmp_path, processor=True)
    calls = []

    class Dmu:
        @staticmethod
        def get_cached_module_file(path, name):
            calls.append(("cache", name))
            if name == "helpers_fake.py":
                raise ImportError("needs a package")

    class Loader:
        def __init__(self, name):
            self.name = name

        def from_pretrained(self, path, *, trust_remote_code):
            assert trust_remote_code is True and path == str(model)
            calls.append(("load", self.name))
            if self.name == "AutoProcessor":
                raise ValueError("no processor class")

    tf = SimpleNamespace(**{n: Loader(n) for n in ("AutoConfig", "AutoProcessor", "AutoImageProcessor")})
    out = rcp.warm_model_dir(str(model), transformers_module=tf, dynamic_module_utils=Dmu)
    assert [c for c in calls if c[0] == "cache"] == [
        ("cache", "configuration_fake_h.py"), ("cache", "helpers_fake.py"),
        ("cache", "image_processing_fake_omni.py")]
    assert out["cached"] == ["configuration_fake_h.py", "image_processing_fake_omni.py"]
    assert out["loaded"] == ["AutoConfig", "AutoImageProcessor"]
    assert [e["step"] for e in out["errors"]] == ["cache helpers_fake.py", "AutoProcessor"]


# ---------------------------------------------------------------------------
# child with the real transformers: after the prewarm, an engine only reads the cache
# ---------------------------------------------------------------------------
def _cache_state(cache: Path) -> dict[str, tuple[int, int, int]]:
    return {str(p.relative_to(cache)): (p.stat().st_ino, p.stat().st_mtime_ns, p.stat().st_size)
            for p in cache.rglob("*") if p.is_file() and "__pycache__" not in p.parts}


def _env(cache: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("HF_")}
    env["HF_MODULES_CACHE"] = str(cache)
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return env


ENGINE_LOADS = '''
import sys
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
path = sys.argv[1]
assert AutoConfig.from_pretrained(path, trust_remote_code=True).hidden_size == 7
assert get_class_from_dynamic_module("image_processing_fake_omni.FakeOmniImageProcessor", path).__name__ \
    == "FakeOmniImageProcessor"
'''


def test_real_transformers_engine_start_writes_nothing_after_the_prewarm(tmp_path):
    pytest.importorskip("transformers")
    model = _model_dir(tmp_path / "hub", processor=True)
    (model / "processor_config.json").unlink()  # no processing_*.py here; keep the image processor
    cache = tmp_path / "modules"
    env = _env(cache)

    warm = subprocess.run([sys.executable, "-m", rcp.MODULE, "--model-path", str(model)],
                          env=env, capture_output=True, text=True, timeout=300)
    report = rcp._parse_report(warm.stdout)
    assert report is not None, warm.stderr[-2000:]
    (result,) = report["models"]
    assert result["cached"] == ["configuration_fake_h.py", "helpers_fake.py",
                                "image_processing_fake_omni.py"]
    assert {"AutoConfig", "AutoProcessor"} <= set(result["loaded"])
    # transformers' AutoImageProcessor needs torchvision, which a CPU test venv may lack
    assert {e["step"] for e in result["errors"]} <= {"AutoImageProcessor"}
    assert (cache / rcp.LOCK_NAME).is_file()
    module_files = {Path(k).name for k in _cache_state(cache) if k.endswith(".py")}
    assert {"configuration_fake_h.py", "helpers_fake.py", "image_processing_fake_omni.py"} <= module_files

    before = _cache_state(cache)
    engine = subprocess.run([sys.executable, "-c", ENGINE_LOADS, str(model)],
                            env=env, capture_output=True, text=True, timeout=300)
    assert engine.returncode == 0, engine.stderr[-2000:]
    assert _cache_state(cache) == before  # no file created, rewritten or touched


# ---------------------------------------------------------------------------
# node side: child interpreter, no GPU, never raises
# ---------------------------------------------------------------------------
def test_run_on_this_node_uses_the_worker_interpreter_without_gpus():
    seen = {}

    def run(cmd, *, env, capture_output, text, timeout):
        seen.update(cmd=cmd, env=env, timeout=timeout)
        report = {"models": [{"path": "/m", "cached": ["a.py"], "loaded": ["AutoConfig"], "errors": []}]}
        return SimpleNamespace(returncode=0, stdout="noise\n" + rcp.RESULT_PREFIX + json.dumps(report),
                               stderr="")

    out = rcp.run_on_this_node(["/m"], 12.0, run=run)
    assert out["ok"] is True and out["report"]["models"][0]["cached"] == ["a.py"]
    assert seen["cmd"] == [sys.executable, "-m", rcp.MODULE, "--model-path", "/m"]
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert seen["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(ROOT)
    assert seen["timeout"] == 12.0


def test_run_on_this_node_reports_timeout_and_missing_report():
    def slow(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    assert rcp.run_on_this_node(["/m"], 5.0, run=slow)["ok"] is False

    def crashed(cmd, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="Traceback\nImportError: transformers")

    out = rcp.run_on_this_node(["/m"], 5.0, run=crashed)
    assert out["ok"] is False and "ImportError: transformers" in out["error"]


# ---------------------------------------------------------------------------
# driver side: one task per machine, pinned there; failures only warn
# ---------------------------------------------------------------------------
class FakeRay:
    def __init__(self, nodes, *, results=None, get_error=None):
        self._nodes = nodes
        self.results = results or {}
        self.get_error = get_error
        self.submitted = []
        self.cancelled = []
        self.remote_options = None

    def is_initialized(self):
        return True

    def nodes(self):
        return self._nodes

    def remote(self, **options):
        self.remote_options = options
        ray = self

        class Task:
            def __init__(self, fn):
                self.fn = fn

            def options(self, *, scheduling_strategy):
                fn = self.fn

                class Bound:
                    @staticmethod
                    def remote(paths, timeout_s):
                        ray.submitted.append((scheduling_strategy, list(paths), timeout_s, fn))
                        return scheduling_strategy["node_id"]

                return Bound

        return Task

    def get(self, ref, timeout):
        assert timeout > 0
        if self.get_error and ref in self.get_error:
            raise self.get_error[ref]
        return self.results.get(ref, {"host": f"h-{ref}", "ok": True, "report": {"models": []}})

    def cancel(self, ref, force):
        self.cancelled.append((ref, force))


def _node(node_id, address, gpus=8, alive=True):
    return {"NodeID": node_id, "NodeManagerAddress": address, "Alive": alive,
            "Resources": {"GPU": gpus, "CPU": 32} if gpus else {"CPU": 32}}


def _affinity(*, node_id, soft):
    return {"node_id": node_id, "soft": soft}


def test_prewarm_runs_once_per_machine_pinned_to_it(tmp_path):
    model = _model_dir(tmp_path)
    ray = FakeRay([
        _node("head", "10.100.0.1"),
        _node("eng-a", "10.100.0.2"),
        _node("eng-b", "10.100.0.2"),  # a second raylet on the same machine
        _node("cpu", "10.100.0.3", gpus=0),
        _node("dead", "10.100.0.4", alive=False),
    ])
    events = []
    out = io.StringIO()
    results = rcp.prewarm_remote_code(SimpleNamespace(hf_checkpoint=str(model)), ray_module=ray,
                                      affinity=_affinity, emit=events.append, out=out)
    assert sorted(s[0]["node_id"] for s in ray.submitted) == ["eng-a", "head"]
    assert all(s[0]["soft"] is False for s in ray.submitted)  # hard pin: runs there or not at all
    assert all(s[1] == [str(model)] and s[3] is rcp.node_task for s in ray.submitted)
    assert ray.remote_options == {"num_cpus": 0, "num_gpus": 0}
    assert sorted(r["node"] for r in results) == ["10.100.0.1", "10.100.0.2"]
    assert out.getvalue() == ""  # no warning
    (event,) = events
    assert event["event"] == rcp.EVENT and len(event["nodes"]) == 2


def test_prewarm_failure_only_warns(tmp_path):
    model = _model_dir(tmp_path)
    ray = FakeRay([_node("head", "10.100.0.1"), _node("eng", "10.100.0.2")],
                  results={"head": {"host": "h", "ok": False, "error": "timed out after 300 s"}},
                  get_error={"eng": TimeoutError("GetTimeoutError")})
    out = io.StringIO()
    results = rcp.prewarm_remote_code(SimpleNamespace(hf_checkpoint=str(model)), ray_module=ray,
                                      affinity=_affinity, out=out)
    assert [r["ok"] for r in results] == [False, False]
    assert ray.cancelled == [("eng", True)]
    text = out.getvalue()
    assert text.count("[rl] WARNING: remote-code prewarm on") == 2
    assert "timed out after 300 s" in text and "GetTimeoutError" in text


def test_prewarm_never_raises(tmp_path):
    model = _model_dir(tmp_path)

    class Broken(FakeRay):
        def nodes(self):
            raise ConnectionError("GCS unreachable")

    out = io.StringIO()
    assert rcp.prewarm_remote_code(SimpleNamespace(hf_checkpoint=str(model)),
                                   ray_module=Broken([]), affinity=_affinity, out=out) is None
    assert "remote-code prewarm did not run (ConnectionError: GCS unreachable)" in out.getvalue()

    out = io.StringIO()
    assert rcp.prewarm_remote_code(SimpleNamespace(hf_checkpoint=str(model)),
                                   ray_module=FakeRay([_node("c", "1.1.1.1", gpus=0)]),
                                   affinity=_affinity, out=out) is None
    assert "no alive Ray node with GPUs" in out.getvalue()


def test_prewarm_without_remote_code_dispatches_nothing(tmp_path):
    ray = FakeRay([_node("head", "10.100.0.1")])
    plain = _model_dir(tmp_path, auto_map=False)
    assert rcp.prewarm_remote_code(SimpleNamespace(hf_checkpoint=str(plain)), ray_module=ray,
                                   affinity=_affinity) is None
    assert ray.submitted == []


def test_prewarm_without_connected_ray_runs_on_this_machine(tmp_path):
    model = _model_dir(tmp_path)

    class Local(FakeRay):
        def is_initialized(self):
            return False

    ray = Local([_node("head", "10.100.0.1")])
    calls = []
    results = rcp.prewarm_remote_code(
        SimpleNamespace(hf_checkpoint=str(model)), ray_module=ray, affinity=_affinity,
        run_local=lambda paths: calls.append(list(paths)) or {"host": "me", "ok": True})
    assert calls == [[str(model)]] and ray.submitted == []
    assert results == [{"host": "me", "ok": True, "node": "local"}]


# ---------------------------------------------------------------------------
# order: after the island's Ray is connected, before Miles creates any engine
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("module, function", [
    ("yeto.rl.adapters.miles.entry", "run_ports_island"),
    ("yeto.rl.adapters.miles.sampler_island", "run_sampler_island"),
])
def test_prewarm_runs_between_ray_connect_and_engine_creation(module, function):
    import importlib

    source = inspect.getsource(getattr(importlib.import_module(module), function))
    connect = source.index("connect_island_ray(miles_args=miles_args)")
    prewarm = source.index("prewarm_remote_code(miles_args")
    engines = source.index("await create_rollout_components(miles_args)")
    assert connect < prewarm < engines
    assert source.count("prewarm_remote_code(") == 1
