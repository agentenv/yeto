"""fix-verda-provider C block: RL islands on clouds whose sky adapter has no
docker runtime (Verda) run the same setup/run inside `docker run` on a bare
VM; every other cloud's task is byte-for-byte what it was."""
from __future__ import annotations

import sys
import types

import pytest

from yeto import launcher
from yeto.gpu_spec import parse_gpu_spec


def _task_for(gpu: str, monkeypatch, *, env=None):
    from test_rl_launcher import _Resources, _Storage, _StorageMode, _Task, _args, _prepare_rl_args

    monkeypatch.setitem(sys.modules, "sky", types.SimpleNamespace(
        Task=_Task, Resources=_Resources, Storage=_Storage, StorageMode=_StorageMode))
    for k in ("SKYPILOT_DOCKER_USERNAME", "SKYPILOT_DOCKER_PASSWORD", "SKYPILOT_DOCKER_SERVER", "HF_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    args = _args(("--gpu", gpu, "--rl-engine", "ports", "--no-island-relaunch", "--total-steps", "1"))
    args.model_revision = "a" * 40
    args.data_revision = "b" * 40
    args.source_sha256 = "c" * 64
    args.reward_sha256 = "d" * 64
    _prepare_rl_args(args)
    return args, launcher.make_miles_island_task(args, parse_gpu_spec(args.gpu)[0], 0, 1, "127.0.0.1:29400")


LOGIN = {"SKYPILOT_DOCKER_USERNAME": "u", "SKYPILOT_DOCKER_PASSWORD": "sekrit-token", "SKYPILOT_DOCKER_SERVER": "ghcr.io",
         "HF_TOKEN": "hf_sekrit"}


def test_gpu_spec_knows_verda_cards():
    spec = parse_gpu_spec("verda:1xrtx-6000-ada@FIN-01")[0]
    assert (spec.cloud, spec.gpu, spec.region, spec.accelerators) == ("verda", "RTX-6000-Ada", "FIN-01", "RTX-6000-Ada:1")
    assert parse_gpu_spec("verda:1xrtx-pro-6000")[0].gpu == "RTX-PRO-6000"
    assert launcher.GPU_MEM_GB["RTX-6000-Ada"] == 48


def test_verda_island_runs_in_vm_docker(monkeypatch):
    args, task = _task_for("verda:1xrtx-6000-ada@FIN-01", monkeypatch, env=LOGIN)
    image = launcher.in_vm_docker_image(args.rl_image)
    assert image.startswith("ghcr.io/") and "@sha256:" in image and not image.startswith("docker:")
    # VM task: sky gets no docker image (its Verda adapter raises on one); the login stays a secret
    assert not hasattr(task.resources, "image_id")
    # secret-handling-hardening: HF token and island HMAC key ride as secrets too.
    assert {k: v for k, v in task.secrets.items() if k.startswith("SKYPILOT_DOCKER")} == {k: LOGIN[k] for k in ("SKYPILOT_DOCKER_USERNAME", "SKYPILOT_DOCKER_PASSWORD", "SKYPILOT_DOCKER_SERVER")}
    assert task.resources.accelerators == "RTX-6000-Ada:1" and task.resources.infra == "verda/FIN-01"
    # host setup: toolkit check, login via stdin from the exported secret, pull by digest, GPU probe
    assert "--password-stdin" in task.setup and f"$DOCKER pull -q {image}" in task.setup
    assert task.setup.index("DOCKER login") < task.setup.index("unset SKYPILOT_DOCKER_USERNAME") < task.setup.index("pull -q")
    assert f"--gpus all {image} nvidia-smi -L" in task.setup
    # host run: the ORIGINAL setup and run execute in one docker run with the island paths mounted
    assert f"exec $DOCKER run --rm --name yeto-island-0 --gpus all --net=host --ipc=host" in task.run
    assert "--entrypoint /bin/bash " + image + " -c " in task.run
    for m in ("sky_workdir", "yeto-output", "yeto-rl", ".cache/huggingface"):
        assert f'-v "$HOME/{m}:/root/{m}"' in task.run
    assert "python3 -m yeto.rl.adapters.miles.island_entry" in task.run and "ray start --head" in task.run  # original run, in run.sh
    assert "[yeto-island] in-VM docker setup done" in task.run
    assert task.run.index("<<'YETO_ISLAND_SETUP_EOF'") < task.run.index("YETO_ISLAND_SETUP_EOF\n", 40) < task.run.index("<<'YETO_ISLAND_RUN_EOF'")
    # the original setup (fork checkout) is in setup.sh
    assert "~/miles" in task.run.split("YETO_ISLAND_SETUP_EOF")[1]
    # environment is forwarded by NAME only: no value of any env/secret in either script
    for name in list(task.envs) + [k for k in task.secrets if not k.startswith("SKYPILOT_DOCKER")] + ["SKYPILOT_NODE_IPS", "SKYPILOT_NODE_RANK", "SKYPILOT_NUM_GPUS_PER_NODE"]:
        assert f"-e {name} " in task.run or f"-e {name}\n" in task.run or task.run.rstrip().endswith(f"-e {name}")
    for secret in ("sekrit-token", "hf_sekrit"):
        assert secret not in task.setup and secret not in task.run
    # secret-handling-hardening: a sky secret (sky exports it on the host, redacted in records)
    assert "HF_TOKEN" not in task.envs and task.secrets["HF_TOKEN"] == "hf_sekrit"
    # the task is launched through the live-stock path, which wants no retry_until_up
    assert launcher.effective_recover_timeout(args) == 0  # --no-island-relaunch: G0 never relaunches


def test_verda_island_without_registry_login_pulls_anonymously(monkeypatch):
    _, task = _task_for("verda:1xrtx-6000-ada", monkeypatch)
    assert not any(k.startswith("SKYPILOT_DOCKER") for k in getattr(task, "secrets", None) or {})
    assert "docker login" not in task.setup and "pull -q" in task.setup
    assert not hasattr(task.resources, "image_id")


@pytest.mark.parametrize("gpu", ["nebius:1xl40s@eu-north1", "aws:1xa10g@us-west-2"])
def test_other_clouds_are_untouched_by_the_in_vm_path(monkeypatch, gpu):
    """The feature is keyed on IN_VM_DOCKER_CLOUDS only: with the set emptied the
    Nebius/AWS task is identical, and the normal task carries no in-VM markers."""
    _, task = _task_for(gpu, monkeypatch, env=LOGIN)
    monkeypatch.setattr(launcher, "IN_VM_DOCKER_CLOUDS", frozenset())
    _, plain = _task_for(gpu, monkeypatch, env=LOGIN)
    for attr in ("setup", "run", "envs", "secrets", "num_nodes", "file_mounts"):
        assert getattr(task, attr) == getattr(plain, attr), attr
    assert vars(task.resources) == vars(plain.resources)
    image = task.resources.image_id
    if isinstance(image, dict):  # Nebius baked VM image (NEBIUS_BAKED_IMAGES): {region: vm, docker: image}
        image = image["docker"]
    assert image == launcher.in_vm_docker_image(image).join(["docker:", ""])
    for marker in ("docker run", "yeto-island/", "docker pull", "YETO_ISLAND", "in_vm", "--gpus all"):
        assert marker not in task.setup and marker not in task.run


def test_in_vm_docker_run_rejects_heredoc_collisions():
    with pytest.raises(ValueError, match="heredoc marker"):
        launcher.in_vm_docker_run("docker:x@sha256:0", "echo YETO_ISLAND_SETUP_EOF", "true", [], 0)


def test_verda_candidates_know_rtx_6000_ada():
    import json

    from yeto.shape.providers import verda_candidates

    types_ = json.load(open("tests/fixtures/verda_rca/instance_types.json"))
    types_ = types_ if isinstance(types_, list) else types_["instance_types"]
    cands = verda_candidates("RTX-6000-Ada", 1, types_, {"FIN-01": ["1RTX6000ADA.10V"], "FIN-02": []})
    assert [c["instance_type"] for c in cands] == ["1RTX6000ADA.10V"] and cands[0]["region"] == "FIN-01"


def test_island_pre_run_is_embedded_before_ray(monkeypatch, tmp_path):
    from test_rl_launcher import _Resources, _Storage, _StorageMode, _Task, _args, _prepare_rl_args

    monkeypatch.setitem(sys.modules, "sky", types.SimpleNamespace(
        Task=_Task, Resources=_Resources, Storage=_Storage, StorageMode=_StorageMode))
    for k in ("SKYPILOT_DOCKER_USERNAME", "SKYPILOT_DOCKER_PASSWORD", "SKYPILOT_DOCKER_SERVER", "HF_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    snippet = tmp_path / "pre.sh"
    snippet.write_text('echo "[g0-arch] probe"\nbash scripts/convert_qwen3_8_next.sh --variant 4layer')
    args = _args(("--gpu", "verda:1xrtx-pro-6000@FIN-01", "--rl-engine", "ports", "--no-island-relaunch",
                  "--total-steps", "1", "--rl-single-island-no-sync", "--rl-island-pre-run", str(snippet)))
    args.model_revision = "a" * 40; args.data_revision = "b" * 40
    args.source_sha256 = "c" * 64; args.reward_sha256 = "d" * 64
    _prepare_rl_args(args)
    task = launcher.make_miles_island_task(args, parse_gpu_spec(args.gpu)[0], 0, 1, "127.0.0.1:29400")
    run = task.run
    # the snippet runs inside the container run script, after cd, before the island Ray and the learner
    i_cd = run.index("cd ~/sky_workdir")
    i_start = run.index(launcher.ISLAND_PRE_RUN_START)
    i_body = run.index("convert_qwen3_8_next.sh --variant 4layer")
    i_done = run.index(launcher.ISLAND_PRE_RUN_DONE)
    assert i_cd < i_start < i_body < i_done < run.index("ray start --head") < run.index("-m yeto.rl.adapters.miles.island_entry")
    assert launcher.ISLAND_PRE_RUN_FAILED in run and "exit 1" in run[i_body:i_done]
    assert run.index(launcher.ISLAND_PRE_RUN_START) > run.index("YETO_ISLAND_RUN_EOF")  # inside the in-VM heredoc
    # the island resources/image are unchanged by the flag
    assert task.resources.accelerators == "RTX-PRO-6000:1" and task.resources.infra == "verda/FIN-01"


def test_island_pre_run_needs_no_sync_and_a_file(tmp_path):
    from test_rl_launcher import _args, _prepare_rl_args

    base = ("--gpu", "verda:1xrtx-pro-6000@FIN-01", "--rl-engine", "ports", "--total-steps", "1")
    args = _args(base + ("--rl-island-pre-run", str(tmp_path / "missing.sh"), "--rl-single-island-no-sync"))
    with pytest.raises(ValueError, match="not a file"):
        _prepare_rl_args(args)
    f = tmp_path / "pre.sh"; f.write_text("echo hi\n")
    args = _args(base + ("--rl-island-pre-run", str(f)))
    with pytest.raises(ValueError, match="rl-single-island-no-sync"):
        _prepare_rl_args(args)
    args = _args(base)
    _prepare_rl_args(args)
    assert launcher.island_pre_run_block(getattr(args, "rl_island_pre_run_script", None)) == ""
