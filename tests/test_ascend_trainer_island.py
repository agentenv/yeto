"""An Ascend trainer island from the node inventory (unified-trainer-islands 4.4), and
the launch preflight's trainer image / card check (P5b-7).

Two 910B4 nodes of an inventory run one verl trainer island: ``yeto launch --dry-run``
passes and shows, per node, the command, the HCCL environment and the scripts the job
sources first (CANN's set_env.sh by default on a rootfs node, or the run's own venv
activation). A docker node cannot run an image yeto builds on Modal, and a trainer
image is refused on cards it was never verified on."""

from __future__ import annotations

import json

import pytest

from test_k31_customer_plan_golden import _hermetic_env
from test_rl_algorithm_provenance import _fake_sky, _no_modal_listing  # noqa: F401  (fixtures)
from yeto.gpu_spec import parse_gpu_spec
from yeto.launch import ssh_trainer_island as sti
from yeto.launch.node_inventory import load_inventory
from yeto.launch.trainer_image_check import check_trainer_images, image_problem
from yeto.rl.adapters.miles.pins import MILES_K31_IMAGE, MILES_NEXT_IMAGE

IFACE = "enp189s0f0"


def _node(i, **kw):
    fields = {"name": f"npu-{i}", "host": f"ops@192.0.2.{30 + i}", "address": f"198.51.100.{30 + i}",
              "interface": IFACE, "device_family": "npu", "cpu_arch": "aarch64", "card": "910B4",
              "cards_per_node": 8, "hbm_gib": 32, "host_memory_gib": 1024, "fabric": "roce", "runtime": "rootfs",
              **kw}
    return "  - {" + ", ".join(f"{k}: {json.dumps(v)}" for k, v in fields.items()) + "}\n"


def _inventory(tmp_path, n=2, **kw):
    path = tmp_path / "inventory.yaml"
    path.write_text("nodes:\n" + "".join(_node(i, **kw) for i in range(n)))
    return path


def _argv(tmp_path, *extra, gpu="ssh:2x8x910b4", inventory=None):
    return ["--training-mode", "rl", "--rl-backend", "verl", "--gpu", gpu,
            "--rl-node-inventory", str(inventory or _inventory(tmp_path)),
            "--model", "Qwen/Qwen3-0.6B", "--model-revision", "a" * 40, "--data", "org/prompts",
            "--data-revision", "b" * 40, "--reward-function", "yeto.rl.numeric_answer_reward:score",
            "--trust-remote-code", "--rl-single-island-no-sync", "--controller", "local",
            "--rl-ssh-trainer-workdir", "/opt/yeto", "--total-steps", "3", "--rollout-batch-size", "16",
            "--n-samples-per-prompt", "2", "--rollout-max-response-len", "128", "--preflight-model", "off",
            *extra]


def _dry_run(tmp_path, monkeypatch, capsys, argv):
    import yeto.cli as cli

    if not (tmp_path / "home").exists():
        _hermetic_env(tmp_path, monkeypatch)
    args = cli.parse_args([*argv, "--dry-run"])
    code = cli.cmd_launch(args)
    captured = capsys.readouterr()
    if code != 0:
        return code, captured.err
    return code, json.loads(captured.out[:captured.out.rindex("}") + 1])


def test_two_910b_nodes_dry_run_one_trainer_island(tmp_path, monkeypatch, capsys):
    code, plan = _dry_run(tmp_path, monkeypatch, capsys, _argv(tmp_path))
    assert code == 0, plan
    (island,) = plan["island_requests"]
    assert (island["gpu"], island["num_nodes"], island["total_gpus"]) == ("910B4", 2, 16)
    assert "yeto.rl.adapters.verl.island_entry" in island["learner_command"]
    ssh = island["ssh_island"]
    assert ssh["mode"] == "resident" and ssh["image"] is None and ssh["warnings"] == []
    assert ssh["activate"] == {"scripts": list(sti.DEFAULT_ACTIVATE["npu"]), "required": False}
    assert [n["name"] for n in ssh["nodes"]] == ["npu-0", "npu-1"]
    for node in ssh["nodes"]:
        assert node["runtime"] == "rootfs" and node["command"].startswith("cd /opt/yeto && bash -c <island script")
        assert {"HCCL_CONNECT_TIMEOUT", "HCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME", "YETO_DEVICE_FAMILY",
                "SKYPILOT_NODE_IPS", "SKYPILOT_NODE_RANK"} <= set(node["env"])
        assert not any(name.startswith("NCCL_") for name in node["env"])
    assert [n["global_ranks"] for n in ssh["rank_layout"]["nodes"]] == ["0-7", "8-15"]


def _spec(tmp_path, *, activate=(), n=2, **node):
    nodes = load_inventory(_inventory(tmp_path, n=n, **node))
    return sti.SshTrainerSpec(nodes=tuple(nodes), run_id="asc", workdir="/opt/yeto", run_script="python3 -m x",
                              activate=tuple(activate))


def test_the_node_environment_and_start_script_of_an_ascend_island(tmp_path):
    spec = _spec(tmp_path)
    paths = sti.remote_paths("asc", "asc-l0-ssh", 0)
    for rank in (0, 1):
        env = sti.node_env(spec, rank, {})
        assert {k: env[k] for k in ("YETO_DEVICE_FAMILY", "HCCL_CONNECT_TIMEOUT", "HCCL_SOCKET_IFNAME",
                                    "GLOO_SOCKET_IFNAME", "SKYPILOT_NODE_IPS", "SKYPILOT_NODE_RANK",
                                    "SKYPILOT_NUM_NODES", "SKYPILOT_NUM_GPUS_PER_NODE")} == {
            "YETO_DEVICE_FAMILY": "npu", "HCCL_CONNECT_TIMEOUT": "1800", "HCCL_SOCKET_IFNAME": IFACE,
            "GLOO_SOCKET_IFNAME": IFACE, "SKYPILOT_NODE_IPS": "198.51.100.30\n198.51.100.31",
            "SKYPILOT_NODE_RANK": str(rank), "SKYPILOT_NUM_NODES": "2", "SKYPILOT_NUM_GPUS_PER_NODE": "8"}
        assert "NCCL_IB_DISABLE" not in env
        script = sti.start_script(spec, rank, {}, paths)
        assert "export HCCL_SOCKET_IFNAME=enp189s0f0\n" in script and "docker run" not in script
        body = sti._trainer_body(spec, rank, "log")
        # CANN first (when the node has it), then the island script
        assert body.startswith("[ ! -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] || "
                               "source /usr/local/Ascend/ascend-toolkit/set_env.sh\n"
                               "[ ! -f /usr/local/Ascend/nnal/atb/set_env.sh ] || "
                               "source /usr/local/Ascend/nnal/atb/set_env.sh\n")
        assert body.index("set_env.sh") < body.index("python3 -m x")


@pytest.mark.parametrize("script", ["~/x'>/tmp/y", "/opt/a b/activate", "~/$(id)", "/opt/../etc/profile",
                                    "relative/activate", '/opt/"q"/activate'])
def test_an_activation_script_path_is_shell_inert(tmp_path, script):
    with pytest.raises(sti.SshIslandError, match="activation script"):
        _spec(tmp_path, activate=(script,))


def test_a_venv_activation_replaces_the_defaults_and_must_exist(tmp_path):
    spec = _spec(tmp_path, activate=("/usr/local/Ascend/ascend-toolkit/set_env.sh", "~/verl-npu/bin/activate"))
    body = sti._trainer_body(spec, 0, "log")
    assert body.startswith(
        "source /usr/local/Ascend/ascend-toolkit/set_env.sh || "
        "{ echo '[yeto-island] could not source /usr/local/Ascend/ascend-toolkit/set_env.sh' >&2; exit 1; }\n"
        "source \"$HOME\"/verl-npu/bin/activate || "
        "{ echo '[yeto-island] could not source ~/verl-npu/bin/activate' >&2; exit 1; }\n")
    assert "nnal/atb" not in body
    with pytest.raises(sti.SshIslandError, match="absolute path"):
        _spec(tmp_path, activate=("verl/bin/activate",))


def test_activation_on_docker_nodes_and_cuda_defaults(tmp_path):
    with pytest.raises(sti.SshIslandError, match="runtime: rootfs"):
        _spec(tmp_path, activate=("/opt/venv/bin/activate",), runtime="docker", image="reg/x@sha256:" + "1" * 64,
              docker_args=["--device", "/dev/davinci0"])
    docker = _spec(tmp_path, runtime="docker", image="reg/x@sha256:" + "1" * 64, docker_args=["--device", "/dev/davinci0"])
    assert sti.activation(docker) == ((), False) and "set_env" not in sti._trainer_body(docker, 0, "log")
    cuda = _spec(tmp_path, device_family="cuda", card="H100", hbm_gib=80, cpu_arch="x86_64", fabric="ib")
    assert sti.activation(cuda) == ((), False)  # nothing sourced by default off Ascend


def test_explicit_activation_reaches_the_dry_run(tmp_path, monkeypatch, capsys):
    argv = _argv(tmp_path, "--rl-ssh-trainer-activate", "/opt/verl-npu/bin/activate")
    code, plan = _dry_run(tmp_path, monkeypatch, capsys, argv)
    assert code == 0, plan
    assert plan["island_requests"][0]["ssh_island"]["activate"] == {
        "scripts": ["/opt/verl-npu/bin/activate"], "required": True}


def test_a_multi_node_ascend_island_without_interface_warns(tmp_path, monkeypatch, capsys):
    path = tmp_path / "inventory.yaml"
    path.write_text("nodes:\n" + "".join(_node(i).replace(f'"interface": "{IFACE}", ', "").replace(
        f"interface: \"{IFACE}\", ", "") for i in range(2)))
    assert IFACE not in path.read_text()
    code, plan = _dry_run(tmp_path, monkeypatch, capsys, _argv(tmp_path, inventory=path))
    assert code == 0, plan
    (warning,) = plan["island_requests"][0]["ssh_island"]["warnings"]
    assert "HCCL_SOCKET_IFNAME" in warning and "npu-0" in warning


def test_docker_nodes_refuse_an_image_built_on_modal(tmp_path, monkeypatch, capsys):
    inv = _inventory(tmp_path, runtime="docker", docker_args=["--device", "/dev/davinci0"])
    code, err = _dry_run(tmp_path, monkeypatch, capsys, _argv(tmp_path, inventory=inv))
    assert code == 1 and "an image yeto builds on Modal" in err and "runtime: rootfs" in err
    site = "docker:swr.example/ascend/verl@sha256:" + "2" * 64
    code, plan = _dry_run(tmp_path, monkeypatch, capsys, _argv(tmp_path, "--rl-image", site, inventory=inv))
    assert code == 0, plan
    assert plan["island_requests"][0]["ssh_island"]["image"] == site.removeprefix("docker:")


# -- P5b-7: a trainer image on cards it was not verified on ------------------------------


def test_the_miles_k31_images_need_sm90_cards():
    for image in (MILES_K31_IMAGE, MILES_NEXT_IMAGE):
        for gpu in ("A100", "A100-80GB", "L40S", "L4", "A10G"):
            message, refused = image_problem(image, gpu)
            assert refused and "No dot product attention backend" in message and "sm90" in message
        for gpu in ("H100", "H200", "B300"):  # the cards it was run on
            assert image_problem(image, gpu) is None
        for gpu in ("B200", "RTX-PRO-6000"):  # above the floor, never run: a warning only
            message, refused = image_problem(image, gpu)
            assert not refused and "above its sm90 floor but was never run" in message
    message, refused = image_problem(MILES_K31_IMAGE, "A100", attention_backend="unfused")
    assert not refused and "make sure it was run on this card" in message
    assert image_problem(MILES_K31_IMAGE, "A100", attention_backend="fused")[1]
    assert image_problem("docker:ghcr.io/x/y@sha256:" + "a" * 64, "A100") is None  # not a known image
    assert image_problem(MILES_K31_IMAGE, "910B4") is None  # not a CUDA card: another check's
    for gpu in ("MI355X", "MI300X"):  # ROCm cards: not sm architectures, never refused here
        assert image_problem(MILES_K31_IMAGE, gpu) is None
    assert image_problem(MILES_K31_IMAGE, "GB200") is None  # outside the card catalog: the card checks' job


def _miles_args(gpu, *extra):
    from test_rl_algorithm_provenance import _launcher_args

    return _launcher_args("ports", extra, gpu=gpu)


def test_the_check_reads_the_trainer_islands_only():
    from yeto.launch.rl_backend import resolve_default_rl_image

    args = _miles_args("modal:1xa100")
    resolve_default_rl_image(args)
    assert args.rl_image == MILES_NEXT_IMAGE  # the default Miles image
    with pytest.raises(ValueError, match="--gpu island 0 .*sm90"):
        check_trainer_images(args, parse_gpu_spec(args.gpu))
    ok = _miles_args("modal:1xh100")
    resolve_default_rl_image(ok)
    assert check_trainer_images(ok, parse_gpu_spec(ok.gpu)) == []
    lifted = _miles_args("modal:1xl40s", "--rl-training-attention-backend", "flash")
    resolve_default_rl_image(lifted)
    (warning,) = check_trainer_images(lifted, parse_gpu_spec(lifted.gpu))
    assert "L40S" in warning
    for gpu in ("modal:1xa100", "modal:1xl40s"):  # verl runs its own images: never refused here
        verl = _miles_args(gpu, "--rl-backend", "verl")
        verl.rl_image = MILES_NEXT_IMAGE
        assert check_trainer_images(verl, parse_gpu_spec(verl.gpu)) == []


@pytest.mark.parametrize("extra, backend, refused", [
    ((), "fused", True),  # the generic recipe's default: TE fused attention
    (("--rl-training-attention-backend", "fused"), "fused", True),
    (("--rl-model-recipe", "kimi_k31"), "auto", True),  # Megatron lets TE pick
    (("--rl-deterministic-trainer",), "unfused", False),  # the generic recipe takes unfused there
    (("--rl-training-attention-backend", "unfused"), "unfused", False),
    (("--rl-model-recipe", "deepseek-v4-flash"), "flash", False),
])
def test_the_attention_the_trainer_actually_runs_decides(extra, backend, refused):
    from yeto.launch.rl_backend import resolve_default_rl_image
    from yeto.launch.trainer_image_check import training_attention

    args = _miles_args("modal:1xa100", *extra)
    resolve_default_rl_image(args)
    assert training_attention(args)[0] == backend
    if refused:
        with pytest.raises(ValueError, match=f"sm90\\+.*training attention: {backend}"):
            check_trainer_images(args, parse_gpu_spec(args.gpu))
    else:
        (warning,) = check_trainer_images(args, parse_gpu_spec(args.gpu))
        assert f"training attention: {backend}" in warning and "make sure it was run on this card" in warning


def test_the_dry_run_refuses_the_p5b_pairing(tmp_path, monkeypatch, capsys):
    import yeto.cli as cli
    import yeto.modal_runner as modal_runner

    _hermetic_env(tmp_path, monkeypatch)
    monkeypatch.setattr(modal_runner, "modal_available", lambda: True)
    argv = ["--training-mode", "rl", "--gpu", "modal:1xa100-80gb", "--model", "Qwen/Qwen3-0.6B",
            "--model-revision", "a" * 40, "--data", "org/prompts", "--data-revision", "b" * 40,
            "--reward-function", "yeto.rl.numeric_answer_reward:score", "--trust-remote-code",
            "--rl-single-island-no-sync", "--controller", "local", "--total-steps", "3",
            "--rollout-batch-size", "4", "--n-samples-per-prompt", "2", "--rollout-max-response-len", "128",
            "--preflight-model", "off"]
    assert cli.cmd_launch(cli.parse_args([*argv, "--dry-run"])) == 1
    err = capsys.readouterr().err
    assert "dry run rejected" in err and "A100-80GB (sm80)" in err and "No dot product attention backend" in err
    assert "(sm90+) only" in err and "Use sm90+ cards (H100, H200, B300) for the trainer" in err
    h100 = [a if a != "modal:1xa100-80gb" else "modal:1xh100" for a in argv]
    assert cli.cmd_launch(cli.parse_args([*h100, "--dry-run"])) == 0


def test_the_launch_preflight_refuses_the_p5b_pairing_before_any_order(monkeypatch, tmp_path):
    """The real launch goes through launch_preflight.pre_cloud_checks (the CLI before its
    worker, and launcher.run): a Miles trainer on A100 is refused there, before any cloud
    work; on H100 it passes."""
    import yeto.launch_preflight as lp
    import yeto.launcher as launcher
    import yeto.modal_runner as modal_runner
    from yeto import runs

    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setenv("YETO_PREFLIGHT_MODEL", "off")
    no_sync = ("--rl-single-island-no-sync", "--controller", "local")
    a100 = _miles_args("modal:1xa100", *no_sync)
    launcher._prepare_rl_args(a100)
    with pytest.raises(ValueError, match="sm90\\+.*training attention: fused"):
        lp.pre_cloud_checks(a100, threads=False)
    h100 = _miles_args("modal:1xh100", *no_sync)
    launcher._prepare_rl_args(h100)
    lp.pre_cloud_checks(h100, threads=False)

    ordered = []
    monkeypatch.setattr(launcher, "prepare_launch_args", launcher._prepare_rl_args)
    monkeypatch.setattr(modal_runner, "ModalOps", lambda *a, **k: ordered.append(("modal", a)))
    with pytest.raises(ValueError, match="sm90\\+"):
        launcher.run(_miles_args("modal:1xa100", *no_sync))
    assert ordered == []  # refused in the launcher's preflight: no Modal app, no cluster


def test_the_explicit_environment_wins_over_the_activation_scripts(tmp_path):
    """CANN's set_env.sh sets LD_LIBRARY_PATH / PYTHONPATH / ASCEND_*: the job sources it
    first and exports the inventory / --island-env / island task environment after it."""
    spec = _spec(tmp_path)
    env = {"LD_LIBRARY_PATH": "/opt/site/lib", "ASCEND_RT_VISIBLE_DEVICES": "0,1"}
    body = sti._trainer_body(spec, 0, "/tmp/requant.log", env=env)
    sourced = body.index("ascend-toolkit/set_env.sh")
    assert sourced < body.index("export LD_LIBRARY_PATH=/opt/site/lib")
    assert sourced < body.index("export ASCEND_RT_VISIBLE_DEVICES=0,1")
    plain = sti._trainer_body(spec, 0, "/tmp/requant.log")  # no env given: nothing exported again
    assert "export LD_LIBRARY_PATH" not in plain
