"""yeto.modal_runner without the Modal SDK: config rules, the container
contract, and the launcher-facing ops against a fake `modal` module."""

from __future__ import annotations

import contextlib
import sys
import types

import pytest

from yeto import modal_runner as mr


def _cfg(**kw) -> mr.ModalIslandConfig:
    base = dict(
        app_name="yeto-run", learner_id=1, training_mode="sft", gpu="H100",
        gpus_per_node=8, num_nodes=1, run_script="echo hi", envs={"SYNCER_ADDR": "1.2.3.4:5000"},
    )
    base.update(kw)
    return mr.ModalIslandConfig(**base)


# --- shape rules --------------------------------------------------------------


def test_whole_node_rule_for_multi_container_islands():
    mr.validate_modal_shape("H100", 8, 2)
    mr.validate_modal_shape("H100", 4, 1)  # partial node is fine for one container
    with pytest.raises(ValueError, match="whole nodes: H100:8 per container, not H100:4"):
        mr.validate_modal_shape("H100", 4, 2)
    with pytest.raises(ValueError, match="whole-node GPU"):
        mr.validate_modal_shape("L4", 4, 2)
    with pytest.raises(ValueError, match="no V100"):
        mr.validate_modal_shape("V100", 8, 1)
    with pytest.raises(ValueError, match="1-8 GPUs"):
        mr.validate_modal_shape("H100", 16, 1)


def test_rl_config_requires_a_digest_pinned_image():
    with pytest.raises(ValueError, match="pin the Miles image by digest"):
        _cfg(training_mode="rl", image_ref="ghcr.io/x/miles:latest").validate()
    _cfg(training_mode="rl", image_ref="ghcr.io/x/miles@sha256:" + "a" * 64).validate()
    assert mr.image_ref_from_rl_image("docker:ghcr.io/x/miles@sha256:" + "b" * 64) == "ghcr.io/x/miles@sha256:" + "b" * 64
    with pytest.raises(ValueError, match="must pin a digest"):
        mr.image_ref_from_rl_image("docker:ghcr.io/x/miles:latest")


def test_config_round_trips_through_json():
    cfg = _cfg(pip_requirements=("torch", "peft"), volume_name="v", volume_mount="/root/ckpt")
    assert mr.ModalIslandConfig.from_json(cfg.to_json()) == cfg
    assert cfg.function_name == "island-1" and cfg.gpu_request == "H100:8"
    assert cfg.cpu_request == 32 and cfg.memory_request_mib == 32 * 8 * 1024


# --- the container contract -----------------------------------------------------


def test_container_sees_skypilot_variables_and_runs_the_sky_script():
    env = mr.skypilot_env(rank=1, container_ips=["10.0.0.1", "10.0.0.2"], gpus_per_node=8)
    assert env == {
        "SKYPILOT_NODE_IPS": "10.0.0.1\n10.0.0.2",
        "SKYPILOT_NUM_NODES": "2",
        "SKYPILOT_NODE_RANK": "1",
        "SKYPILOT_NUM_GPUS_PER_NODE": "8",
    }
    cmd = mr.container_command("MASTER_ADDR=$(echo \"$SKYPILOT_NODE_IPS\" | head -n1)\ntorchrun ...")
    assert cmd[:2] == ["bash", "-lc"] and cmd[2].startswith("cd /root/sky_workdir && ")


def test_island_main_runs_script_with_envs(monkeypatch):
    seen = {}

    def fake_call(cmd, env=None):
        seen["cmd"], seen["env"] = cmd, env
        return 0

    monkeypatch.setattr(mr.subprocess, "call", fake_call)
    cfg = _cfg(envs={"SYNCER_ADDR": "1.2.3.4:5000", "LEARNER_ID": "1"})
    assert mr.island_main(cfg.to_json()) == 0
    assert seen["env"]["SYNCER_ADDR"] == "1.2.3.4:5000"
    assert seen["env"]["SKYPILOT_NODE_RANK"] == "0" and seen["env"]["SKYPILOT_NUM_NODES"] == "1"
    assert seen["env"]["HOME"] == "/root"
    monkeypatch.setattr(mr.subprocess, "call", lambda cmd, env=None: 3)
    with pytest.raises(RuntimeError, match="exited with 3"):
        mr.island_main(cfg.to_json())


# --- reachability -------------------------------------------------------------------


def test_syncer_must_be_reachable_from_modal(monkeypatch):
    assert mr.is_public_address("8.8.8.8")
    assert not mr.is_public_address("10.1.2.3") and not mr.is_public_address("127.0.0.1")
    assert mr.resolve_syncer_for_modal("8.8.8.8:5000", None) == "8.8.8.8:5000"
    with pytest.raises(ValueError, match="not reachable from Modal.*--syncer-public-addr"):
        mr.resolve_syncer_for_modal("192.168.1.10:5000", None)
    assert mr.resolve_syncer_for_modal("192.168.1.10:5000", "tunnel.example.com:5000") == "tunnel.example.com:5000"


# --- fake modal SDK -------------------------------------------------------------------


class _Image:
    def __init__(self):
        self.calls = []

    def _rec(self, name, *a, **k):
        self.calls.append((name, a, k))
        return self

    def run_commands(self, *a, **k):
        return self._rec("run_commands", *a, **k)

    def pip_install(self, *a, **k):
        return self._rec("pip_install", *a, **k)

    def add_local_dir(self, *a, **k):
        return self._rec("add_local_dir", *a, **k)

    def env(self, *a, **k):
        return self._rec("env", *a, **k)


class _Call:
    def __init__(self, cid, outcome="running"):
        self.object_id = cid
        self.outcome = outcome
        self.cancelled = None
        self.logs = types.SimpleNamespace(
            tail=lambda entries=100: iter([types.SimpleNamespace(message="log line")]),
            stream=lambda: iter([types.SimpleNamespace(message="streamed")]),
        )

    def get(self, timeout=None, index=0):
        if self.outcome == "running":
            raise TimeoutError("still running")
        if self.outcome == "failed":
            raise RuntimeError("island exited with 1")
        return 0

    def cancel(self, terminate_containers=False):
        self.cancelled = terminate_containers


def fake_modal(monkeypatch):
    modal = types.ModuleType("modal")
    state = {"functions": {}, "calls": {}, "spawned": [], "deployed": [], "images": []}

    class App:
        def __init__(self, name):
            self.name = name
            state["app"] = self

        def function(self, **kwargs):
            def deco(fn):
                state["functions"][kwargs["name"]] = (fn, kwargs)
                return fn

            return deco

        def deploy(self, name=None):
            state["deployed"].append(name)

    class Image:
        @staticmethod
        def from_registry(tag, **kw):
            img = _Image()
            img.calls.append(("from_registry", (tag,), kw))
            state["images"].append(img)
            return img

        @staticmethod
        def debian_slim(python_version=None):
            img = _Image()
            img.calls.append(("debian_slim", (), {"python_version": python_version}))
            state["images"].append(img)
            return img

    class Function:
        @staticmethod
        def from_name(app_name, name):
            fn, kwargs = state["functions"][name]

            class _F:
                @staticmethod
                def spawn(payload):
                    cid = f"fc-{len(state['spawned'])}"
                    state["spawned"].append((app_name, name, payload))
                    state["calls"][cid] = _Call(cid)
                    return state["calls"][cid]

            return _F

    class FunctionCall:
        @staticmethod
        def from_id(cid):
            return state["calls"][cid]

    modal.App, modal.Image, modal.Function, modal.FunctionCall = App, Image, Function, FunctionCall
    modal.Retries = lambda **kw: ("retries", kw)
    modal.Secret = types.SimpleNamespace(from_dict=lambda d: ("secret", d))
    modal.Volume = types.SimpleNamespace(from_name=lambda n, create_if_missing=False: ("volume", n))
    modal.enable_output = contextlib.nullcontext
    experimental = types.ModuleType("modal.experimental")
    experimental.clustered = lambda size, rdma=False, **kw: (lambda fn: (state.__setitem__("clustered", (size, rdma)) or fn))
    experimental.get_cluster_info = lambda: types.SimpleNamespace(rank=0, container_ips=["10.0.0.1"])
    modal.experimental = experimental
    monkeypatch.setitem(sys.modules, "modal", modal)
    monkeypatch.setitem(sys.modules, "modal.experimental", experimental)
    return state


def test_define_builds_the_function_from_the_config(monkeypatch):
    state = fake_modal(monkeypatch)
    ops = mr.ModalOps("yeto-run")
    cfg = _cfg(num_nodes=2, region="us", pip_requirements=("torch",), volume_name="ck", volume_mount="/root/ck")
    ops.define(cfg)
    ops.define(cfg)  # idempotent
    fn, kwargs = state["functions"]["island-1"]
    assert fn is mr.island_main
    assert kwargs["gpu"] == "H100:8" and kwargs["region"] == "us"
    assert kwargs["cpu"] == 32 and kwargs["memory"] == 32 * 8 * 1024
    assert kwargs["retries"] == ("retries", {"max_retries": mr.DEFAULT_RETRIES, "initial_delay": 0.0})
    assert kwargs["secrets"] == [("secret", {"SYNCER_ADDR": "1.2.3.4:5000"})]
    assert kwargs["volumes"] == {"/root/ck": ("volume", "ck")}
    assert state["clustered"] == (2, True)
    (img,) = state["images"]
    names = [c[0] for c in img.calls]
    # add_local_dir(copy=False) must come last: Modal refuses any build
    # step after it (this order failed the first live Modal deploy).
    assert names == ["debian_slim", "pip_install", "env", "add_local_dir"]
    assert img.calls[names.index("add_local_dir")][1][1] == mr.CONTAINER_WORKDIR
    # Unpinned single-container island: no region kwarg, no clustered.
    state2 = fake_modal(monkeypatch)
    mr.ModalOps("yeto-run").define(_cfg(learner_id=2))
    assert "region" not in state2["functions"]["island-2"][1] and "clustered" not in state2


def test_rl_island_uses_the_digest_image(monkeypatch):
    state = fake_modal(monkeypatch)
    ref = "ghcr.io/x/miles@sha256:" + "c" * 64
    mr.ModalOps("yeto-run").define(_cfg(training_mode="rl", image_ref=ref, setup_script="pip install x"))
    (img,) = state["images"]
    assert img.calls[0] == ("from_registry", (ref,), {})
    # The setup reads the repo, which is only mounted at start-up, so it
    # must not become a build step (the first live RL deploy failed so).
    assert [c[0] for c in img.calls] == ["from_registry", "env", "add_local_dir"]


def test_rl_setup_runs_in_the_container_before_the_run_script(monkeypatch):
    seen = []

    def fake_call(cmd, env=None):
        seen.append(cmd[2])
        return 0

    monkeypatch.setattr(mr.subprocess, "call", fake_call)
    cfg = _cfg(num_nodes=1, training_mode="rl", setup_script="echo setup", run_script="echo run")
    assert mr.island_main(cfg.to_json()) == 0
    assert seen[0].endswith("echo setup") and seen[1].endswith("echo run")
    monkeypatch.setattr(mr.subprocess, "call", lambda cmd, env=None: 5)
    with pytest.raises(RuntimeError, match="setup exited with 5"):
        mr.island_main(cfg.to_json())


def test_spawn_status_cancel_and_logs(monkeypatch):
    state = fake_modal(monkeypatch)
    ops = mr.ModalOps("yeto-run")
    cfg = _cfg()
    ops.define(cfg)
    ops.deploy()
    assert state["deployed"] == ["yeto-run"]
    cid = ops.spawn(cfg)
    assert state["spawned"][0][:2] == ("yeto-run", "island-1")
    assert mr.ModalIslandConfig.from_json(state["spawned"][0][2]) == cfg
    assert ops.status(cid) == "RUNNING"
    state["calls"][cid].outcome = "done"
    assert ops.status(cid) == "SUCCEEDED"
    state["calls"][cid].outcome = "failed"
    assert ops.status(cid) == "FAILED"
    ops.cancel(cid)
    assert state["calls"][cid].cancelled is True
    assert list(ops.tail_logs(cid)) == ["log line"]


def test_island_ops_relaunch_keeps_the_learner_id_and_routing(monkeypatch):
    state = fake_modal(monkeypatch)
    ops = mr.ModalOps("yeto-run")
    cfg = _cfg(learner_id=3)
    ops.define(cfg)
    island_ops = mr.ModalIslandOps(ops)
    name = mr.modal_island_name("run", 3)
    assert name == "run-l3-modal" and mr.is_modal_island(name)
    first = island_ops.relaunch(cfg, name)
    state["calls"][first].outcome = "failed"
    assert str(island_ops.job_status(name, first)) == "FAILED"
    assert island_ops.job_status(name, first).is_terminal()
    second = island_ops.relaunch(cfg, name)
    assert second != first
    payloads = [mr.ModalIslandConfig.from_json(p).learner_id for _, _, p in state["spawned"]]
    assert payloads == [3, 3]  # same learner id both times
    assert island_ops.cluster_up(name)
    island_ops.down(name)
    assert state["calls"][second].cancelled is True

    class SkyOps:
        def job_status(self, n, j):
            return "sky-status"

        def cluster_up(self, n):
            return "sky-up"

        def relaunch(self, t, n):
            return "sky-job"

        def down(self, n):
            return "sky-down"

        def now(self):
            return 1.0

        def sleep(self, s):
            return None

    routing = mr.RoutingOps(SkyOps(), island_ops)
    assert routing.job_status("run-l0-us-east-1", 7) == "sky-status"
    assert str(routing.job_status(name, second)) == "RUNNING"
    assert routing.cluster_up("run-l0-us-east-1") == "sky-up" and routing.now() == 1.0


class _ImmediateThread:
    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self._target, self._args, self._kwargs = target, args, kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)


def test_preempted_island_is_relaunched_with_the_same_learner_id_then_abandoned(monkeypatch, capsys):
    # The launcher's supervisor treats a Modal island like any other: a
    # call that ends in failure (preempted, Modal's own retries exhausted)
    # is re-spawned with the SAME learner id; when Modal keeps refusing
    # past the recovery timeout the island is abandoned, its last call
    # cancelled, and the exit reported.
    from yeto.launcher import ABANDONED, RUNNING, FleetController

    state = fake_modal(monkeypatch)
    ops = mr.ModalOps("yeto-run")
    cfg = _cfg(learner_id=2)
    ops.define(cfg)
    island_ops = mr.ModalIslandOps(ops)
    name = mr.modal_island_name("run", 2)
    first = island_ops.relaunch(cfg, name)

    class Clock:
        t = 0.0

        def now(self):
            return self.t

        def sleep(self, s):
            self.t += s

    clock = Clock()
    routing = mr.RoutingOps(clock, island_ops)
    controller = FleetController(
        learners={name: (cfg, first)}, syncer=None, sky_ops=routing, poll_interval=1,
        recover_timeout=10, thread_cls=_ImmediateThread,
        syncer_probe=lambda: None, syncer_restart=lambda: None,
    )
    rec = controller.learners[name]
    controller._poll(rec, is_syncer=False)
    assert rec["state"] == RUNNING and rec["job_id"] == first
    state["calls"][first].outcome = "failed"  # preemption, retries exhausted
    controller._poll(rec, is_syncer=False)  # -> recovering, relaunch runs inline
    controller._poll(rec, is_syncer=False)  # -> recovered
    assert rec["state"] == RUNNING and rec["job_id"] != first
    learner_ids = [mr.ModalIslandConfig.from_json(p).learner_id for _, _, p in state["spawned"]]
    assert learner_ids == [2, 2]
    # Modal now refuses every spawn: past the timeout the island is
    # abandoned and its last call cancelled.
    second = rec["job_id"]
    state["calls"][second].outcome = "failed"
    monkeypatch.setattr(ops, "spawn", lambda c: (_ for _ in ()).throw(RuntimeError("no capacity")))
    controller._poll(rec, is_syncer=False)
    clock.t += 11
    controller._poll(rec, is_syncer=False)
    assert rec["state"] == ABANDONED
    assert state["calls"][second].cancelled is True
    assert "ABANDONED" in capsys.readouterr().err


def test_manual_cli_builds_a_config(tmp_path, monkeypatch):
    script = tmp_path / "run.sh"
    script.write_text("torchrun -m yeto.learner\n")
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    reqs = tmp_path / "req.txt"
    reqs.write_text("# comment\ntransformers==5.13.0\npeft>=0.14\n")
    ns = mr._parse_args([
        "--learner-id", "2", "--num-learners", "3", "--syncer-addr", "8.8.8.8:5000",
        "--cluster-prefix", "demo", "--gpu", "h100", "--gpus-per-node", "8",
        "--run-script", str(script), "--env", "WANDB_API_KEY=w", "--requirements", str(reqs),
    ])
    cfg = mr.config_from_cli(ns)
    assert cfg.app_name == "yeto-demo" and cfg.gpu == "H100" and cfg.gpus_per_node == 8
    assert cfg.envs["SYNCER_ADDR"] == "8.8.8.8:5000" and cfg.envs["LEARNER_ID"] == "2"
    assert cfg.envs["WANDB_API_KEY"] == "w" and cfg.envs["HF_TOKEN"] == "hf_x"
    assert cfg.pip_requirements == ("transformers==5.13.0", "peft>=0.14")
    assert cfg.run_script.startswith("torchrun")


def test_stop_app_confirms_and_surfaces_failures(monkeypatch):
    """Modal's `app stop` asks for confirmation and aborts without a TTY;
    the first live `yeto down` printed "stopped" while the app stayed
    deployed. Pass --yes and raise when Modal did not stop it."""
    import subprocess

    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout=fake_run.out, stderr="")

    fake_run.out = ""
    monkeypatch.setattr(mr.subprocess, "run", fake_run)
    ops = mr.ModalOps("yeto-run1")
    ops.stop_app()
    assert calls[-1][-3:] == ["stop", "--yes", "yeto-run1"]
    fake_run.out = "Aborted: no interactive terminal detected."
    with pytest.raises(RuntimeError, match="modal app stop yeto-run1 failed"):
        ops.stop_app()


def test_island_image_skips_agent_worktrees_and_nested_caches():
    """A live Modal deploy aborted because a file under .claude/worktrees
    changed mid-upload; nested __pycache__ dirs were uploaded too."""
    ignore = mr.MODAL_WORKDIR_IGNORE
    assert ".claude" in ignore and ".git" in ignore and ".venv" in ignore
    assert "**/__pycache__" in ignore and "**/*.pyc" in ignore
