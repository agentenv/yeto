"""The verl island Ray's CPU slots, on every cloud.

A verl island on an 8-vCPU sky VM (AWS g5.2xlarge, A10G) stopped after "Starting create
trainer": its Ray had 8 CPU slots, the trainer actor asked for 10, Ray left it pending
with no error (``Pending Demands: {'CPU': 10.0}``), the GPU idle. Only a Modal container
had its Ray restarted with 32 slots (``island_entry.ensure_ray_cpus``).

* the launcher starts a verl island's Ray with ``max(node vCPUs, 32)`` slots on every node,
  sky, ssh and Modal alike; a Miles island's ``ray start`` is unchanged;
* a Ray still short of the trainer's slots stops the island before verl starts, and a
  trainer actor or worker placement group Ray does not place stops it within the schedule
  timeout: a configuration fault, the deterministic stop (exit 11), never a silent hang;
* the dry run shows the node's vCPUs, the Ray slots and what the trainer asks for.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_k31_customer_plan_golden import _hermetic_env
from test_rl_algorithm_provenance import _fake_sky, _no_modal_listing  # noqa: F401  (fixtures)
from yeto.config_errors import DETERMINISTIC_STOP_ENV, ISLAND_STOP_EXIT, ISLAND_STOP_MARKER
from yeto.rl.adapters.verl import ray_resources as rr

ROOT = Path(__file__).resolve().parents[1]


def _argv(*extra, gpu="aws:1xa10g", backend="verl"):
    return ["--training-mode", "rl", "--rl-backend", backend, "--gpu", gpu,
            "--model", "Qwen/Qwen3-0.6B", "--model-revision", "a" * 40, "--data", "org/prompts",
            "--data-revision", "b" * 40, "--reward-function", "yeto.rl.numeric_answer_reward:score",
            "--trust-remote-code", "--rl-single-island-no-sync", "--controller", "local",
            "--total-steps", "3", "--rollout-batch-size", "16", "--n-samples-per-prompt", "2",
            "--rollout-max-response-len", "128", "--preflight-model", "off", *extra]


def _dry_run(tmp_path, monkeypatch, capsys, argv):
    import yeto.cli as cli

    if not (tmp_path / "home").exists():
        _hermetic_env(tmp_path, monkeypatch)
    monkeypatch.setenv("YETO_VERL_IMAGE_REGISTRY", "ghcr.io/example")  # the trainer image's registry
    from yeto.launch import trainer_image_check

    monkeypatch.setattr(trainer_image_check, "check_trainer_images", lambda args, specs: None)  # no registry read
    code = cli.cmd_launch(cli.parse_args([*argv, "--dry-run"]))
    captured = capsys.readouterr()
    assert code == 0, captured.err
    return json.loads(captured.out[:captured.out.rindex("}") + 1])


def _ray_starts(run: str) -> tuple[str, str]:
    head = run[run.index("ray start --head"):].split("\n", 1)[0]
    worker = run[run.index('ray start --address="$MASTER_ADDR:6379"'):].split(" && break", 1)[0]
    return head, worker


def _island_run(backend, gpu="aws:1xa10g"):
    from yeto.gpu_spec import parse_gpu_spec
    from yeto.launch.rl_island_task import make_miles_island_task
    from yeto.launcher import _prepare_rl_args
    import yeto.cli as cli

    args = cli.parse_args(_argv(gpu=gpu, backend=backend))
    args.source_sha256, args.reward_sha256 = "c" * 64, "d" * 64
    _prepare_rl_args(args)
    return make_miles_island_task(args, parse_gpu_spec(args.gpu)[0], 0, 1, "127.0.0.1:29400").run


# ---------------------------------------------------------------------- the launcher
FLAG = ' --num-cpus="$(n=$(nproc); [ "$n" -gt 32 ] && echo "$n" || echo 32)"'


@pytest.mark.parametrize("gpu", ["aws:1xa10g", "aws:2x1xa10g", "modal:1x1xl40s"])
def test_a_verl_island_ray_gets_max_vcpus_32_slots_on_every_node(tmp_path, monkeypatch, gpu):
    _hermetic_env(tmp_path, monkeypatch)
    head, worker = _ray_starts(_island_run("verl", gpu))
    assert FLAG in head and FLAG in worker


def test_a_miles_island_ray_start_is_unchanged(tmp_path, monkeypatch):
    _hermetic_env(tmp_path, monkeypatch)
    run = _island_run("miles", "aws:2x1xa10g")
    assert "--num-cpus" not in run


@pytest.mark.parametrize(("vcpus", "slots"), [(8, 32), (32, 32), (96, 96)])
def test_the_flag_gives_max_of_the_node_vcpus_and_32(vcpus, slots):
    out = subprocess.run(["bash", "-c", f"nproc() {{ echo {vcpus}; }}; echo{FLAG}"],
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out == f"--num-cpus={slots}"


def test_the_launcher_and_the_island_entry_share_one_floor():
    from yeto.rl.adapters.verl import island_entry, launch_flags

    assert island_entry.MIN_RAY_CPUS == launch_flags.ISLAND_RAY_MIN_CPUS == 32
    assert rr.trainer_cpus() == 11 <= launch_flags.ISLAND_RAY_MIN_CPUS


def test_the_trainer_actors_ask_for_the_slots_the_checks_count():
    for name in ("fully_async_runner.py", "buffer_trainer_runner.py"):
        source = (ROOT / "yeto/rl/adapters/verl" / name).read_text()
        assert "ray.remote(num_cpus=TRAINER_ACTOR_CPUS)" in source and "num_cpus=10" not in source
    runner = (ROOT / "yeto/rl/adapters/verl/fully_async_runner.py").read_text()
    assert "await_actor(trainer" in runner and "get_watching_placements(trainer.init_workers.remote())" in runner
    assert "ray.get(trainer.init_workers.remote())" not in runner


# ---------------------------------------------------------------------- the dry run
def test_the_dry_run_shows_the_vcpus_and_the_trainer_slots_of_a_verl_island(tmp_path, monkeypatch, capsys):
    plan = _dry_run(tmp_path, monkeypatch, capsys, _argv())
    (island,) = plan["island_requests"]
    assert island["island_ray"] == {"node_vcpus": "8+", "ray_cpu_slots": "max(node vCPUs, 32)",
                                    "trainer_ray_cpus": 11.0, "fits": True}
    typed = _dry_run(tmp_path, monkeypatch, capsys, _argv("--learner-instance-type", "g5.2xlarge"))
    assert typed["island_requests"][0]["island_ray"]["node_vcpus"] == "those of g5.2xlarge"
    miles = _dry_run(tmp_path, monkeypatch, capsys, _argv(backend="miles"))
    assert "island_ray" not in miles["island_requests"][0]


# ---------------------------------------------------------------------- the island
def test_a_ray_short_of_the_trainer_slots_is_refused_with_need_and_have():
    eight = {"before": {"CPU": 8.0, "GPU": 1.0, "accelerator_type:A10G": 1.0, "memory": 2e10}}
    problem = rr.cpu_shortfall(eight)
    assert problem.startswith("the island Ray has 8 CPU slots ({'CPU': 8, 'GPU': 1, 'accelerator_type:A10G': 1})"
                              " but the verl trainer needs 11 (trainer actor 10 + task runner 1)")
    assert "ray start --num-cpus" in problem
    assert rr.cpu_shortfall({"before": {"CPU": 8.0}, "after": {"CPU": 32.0}}) is None  # the restart's
    assert rr.cpu_shortfall({"before": {"CPU": 11.0}}) is None
    assert rr.cpu_shortfall({"skipped": "no RAY_ADDRESS"}) is None  # unknown: not refused


def test_the_island_entry_stops_with_the_deterministic_stop_exit(monkeypatch, capsys):
    from yeto.launch.island_lifecycle import CONFIG_ERROR, island_stop_decision
    from yeto.rl.adapters.verl.island_entry import stop_on_ray_cpu_shortfall

    short = {"before": {"CPU": 8.0, "GPU": 1.0}}
    monkeypatch.setenv(DETERMINISTIC_STOP_ENV, "1")
    with pytest.raises(SystemExit) as stopped:
        stop_on_ray_cpu_shortfall(short)
    assert stopped.value.code == ISLAND_STOP_EXIT == 11
    assert f"{ISLAND_STOP_MARKER} {CONFIG_ERROR}" in capsys.readouterr().err
    assert island_stop_decision(stopped.value.__cause__).cause == CONFIG_ERROR
    stop_on_ray_cpu_shortfall({"before": {"CPU": 32.0, "GPU": 1.0}})  # enough: nothing happens
    monkeypatch.delenv(DETERMINISTIC_STOP_ENV)
    with pytest.raises(rr.RayResourcesUnsatisfiable, match="needs 11"):  # switched off: raised (exit 1)
        stop_on_ray_cpu_shortfall(short)
    entry = (ROOT / "yeto/rl/adapters/verl/island_entry.py").read_text()
    assert entry.index("ray_info = ensure_ray_cpus()") < entry.index("stop_on_ray_cpu_shortfall(ray_info)")


class _FakeRay:
    """``ray.wait`` / ``ray.get`` / ``placement_group_table`` over a fake clock."""

    def __init__(self, *, ready_after=None, groups=None):
        self.now, self.ready_after = 0.0, ready_after
        self.groups = groups or (lambda now: {})
        self.util = SimpleNamespace(placement_group_table=lambda: self.groups(self.now))

    def wait(self, refs, timeout):
        assert self.now < 100_000, "never stopped"
        if self.ready_after is not None and self.now + timeout >= self.ready_after:
            self.now = max(self.now, self.ready_after)
            return refs, []
        self.now += timeout
        return [], refs

    def get(self, ref):
        return "initialized"

    def cluster_resources(self):
        return {"CPU": 8.0, "GPU": 1.0}

    def available_resources(self):
        return {"CPU": 7.0, "GPU": 1.0}


class _Actor:
    __ray_ready__ = SimpleNamespace(remote=lambda: "ready-ref")


def test_a_trainer_actor_ray_does_not_place_stops_after_the_timeout():
    ray = _FakeRay()
    with pytest.raises(rr.RayResourcesUnsatisfiable) as raised:
        rr.await_actor(_Actor(), demand={"CPU": 10.0}, timeout_s=300, ray_mod=ray)
    assert str(raised.value).startswith(
        "the verl trainer actor asked Ray for {'CPU': 10} and was not placed within 300 s; "
        "the island Ray has {'CPU': 8, 'GPU': 1} in all, {'CPU': 7, 'GPU': 1} free")
    assert ray.now == 300
    rr.await_actor(_Actor(), demand={"CPU": 10.0}, timeout_s=300, ray_mod=_FakeRay(ready_after=40))


def test_a_worker_placement_group_pending_past_the_timeout_stops_the_island():
    pending = {"pg1": {"state": "PENDING", "bundles": {0: {"GPU": 1.0, "CPU": 1.0}, 1: {"GPU": 1.0}}}}
    ray = _FakeRay(groups=lambda now: pending)
    with pytest.raises(rr.RayResourcesUnsatisfiable, match=r"stayed pending for more than 300 s, asking for "
                                                           r"\{'CPU': 1, 'GPU': 2\}") as raised:
        rr.get_watching_placements("init-ref", timeout_s=300, poll_s=10, ray_mod=ray, clock=lambda: ray.now)
    assert 300 < ray.now <= 320 and "{'CPU': 8, 'GPU': 1} in all" in str(raised.value)


def test_a_slow_init_whose_groups_get_placed_is_waited_for():
    """A model load takes long: a group pending for a while, then placed, is not a fault."""
    ray = _FakeRay(ready_after=2000, groups=lambda now: (
        {"pg1": {"state": "PENDING" if now < 200 else "CREATED", "bundles": {0: {"GPU": 1.0}}},
         "pg0": {"state": "CREATED", "bundles": {0: {"CPU": 1.0}}}}))
    assert rr.get_watching_placements("init-ref", timeout_s=300, poll_s=10, ray_mod=ray,
                                      clock=lambda: ray.now) == "initialized"
    assert ray.now == 2000


def test_the_schedule_timeout_is_read_from_the_environment():
    assert rr.schedule_timeout_s({}) == 300.0
    assert rr.schedule_timeout_s({rr.SCHEDULE_TIMEOUT_ENV: "60"}) == 60.0
    assert rr.schedule_timeout_s({rr.SCHEDULE_TIMEOUT_ENV: "x"}) == 300.0
    assert rr.schedule_timeout_s({rr.SCHEDULE_TIMEOUT_ENV: "0"}) == 300.0
