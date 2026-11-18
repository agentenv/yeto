"""The bare-metal 6 x 8 B300 trainer kit of examples/k31 (docs/K31_BARE_METAL_B300.md):
the node inventory goes through ``load_inventory`` and the cuda docker checks of the SSH
trainer island, and the run config's trainer argv goes through ``yeto launch --dry-run``
with the shipped inventory, an EP16 selection and the shipped algorithm spec. Nothing is
ordered and no node is contacted."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from yeto.launch.node_inventory import load_inventory
from yeto.launch.ssh_trainer_island import cuda_docker_problems

ROOT = Path(__file__).resolve().parents[1]
K31 = ROOT / "examples" / "k31"
INVENTORY = K31 / "inventory.b300-6x8-baremetal.yaml"
RUN = K31 / "run.baremetal-b300-262k.json"
RUNBOOK = ROOT / "docs" / "K31_BARE_METAL_B300.md"


def _argv() -> list[str]:
    return [str(a) for a in json.loads(RUN.read_text())["trainer"]["argv"]]


def _flag(argv: list[str], name: str) -> str:
    values = [argv[i + 1] for i, a in enumerate(argv) if a == name]
    assert len(values) == 1, f"{name}: {values}"
    return values[0]


def test_inventory_lists_six_b300_nodes_with_distinct_cluster_addresses():
    nodes = load_inventory(INVENTORY)
    assert len(nodes) == 6
    assert {(n.device_family, n.card, n.cards_per_node, n.cpu_arch) for n in nodes} == {("cuda", "B300", 8, "x86_64")}
    assert len({n.address for n in nodes}) == 6 and all(n.address for n in nodes)
    assert {n.fabric for n in nodes} <= {"ib", "roce"}
    assert all(n.host_memory_gib >= 1600 for n in nodes)


def test_inventory_docker_args_pass_the_multi_node_rdma_checks():
    for node in load_inventory(INVENTORY):
        assert node.runtime == "docker"
        assert cuda_docker_problems(node.docker_args, rdma=True) == [], node.name


def _island_env(argv: list[str]) -> dict[str, str]:
    pairs = [argv[i + 1] for i, a in enumerate(argv) if a == "--island-env"]
    return dict(p.split("=", 1) for p in pairs)


def test_inventory_env_is_the_fabric_and_the_run_settings_ride_island_env():
    nodes = load_inventory(INVENTORY)
    envs = {json.dumps(dict(n.env), sort_keys=True) for n in nodes}
    assert len(envs) == 1, "the fabric settings are the same on every node"
    assert nodes[0].env["NCCL_IB_HCA"]
    env = _island_env(_argv())
    for key in ("MILES_MXFP4_QAT_TRITON", "MILES_K31_INDEXER_QUERY_CHUNK", "MILES_K3_ATTN_RES_TOKEN_CHUNK",
                "YETO_MILES_TRAIN_EXTRA_ARGV", "YETO_RL_NAN_SKIP"):
        assert key in env, key
        assert key not in nodes[0].env, f"{key}: one copy, in the run config"
    # 93 layers split 31/31/31 over PP3 without any split flag
    assert "pipeline-num-layers" not in env["YETO_MILES_TRAIN_EXTRA_ARGV"]
    assert "first_stage_grad.before_train_step" in env["YETO_MILES_TRAIN_EXTRA_ARGV"]


def test_inventory_image_is_the_trainer_image_of_the_run_config():
    from types import SimpleNamespace

    from yeto.launch.ssh_trainer_island import run_image

    image = _flag(_argv(), "--rl-image").removeprefix("docker:")
    nodes = load_inventory(INVENTORY)
    assert {n.image for n in nodes} <= {None, image}  # the run's --rl-image decides
    # the overlay-safe repack: run_image keeps it as it is
    assert run_image(SimpleNamespace(rl_image=f"docker:{image}"), nodes) == image


def test_nodes_keep_the_run_data_on_scratch_and_get_the_code_from_the_store():
    argv = _argv()
    assert "--rl-ssh-trainer-workdir" not in argv, "the code pack comes from the s3 store"
    assert _flag(argv, "--rl-checkpoint-store").startswith("s3://")
    assert _flag(argv, "--rl-island-ray-spill-dir").startswith("/yeto-scratch/")
    for node in load_inventory(INVENTORY):
        assert node.scratch and node.scratch.startswith("/"), node.name
        assert node.scratch_min_free_gib and node.scratch_min_free_gib >= 1500, node.name


def test_run_config_names_the_shipped_inventory_and_the_layout():
    argv = _argv()
    assert _flag(argv, "--gpu") == "ssh:6x8xb300"
    assert re.sub(r"[<>]", "", _flag(argv, "--rl-node-inventory")) == INVENTORY.name
    layout = {f: int(_flag(argv, f)) for f in ("--pipeline-parallel", "--tensor-parallel",
                                                "--context-parallel", "--expert-parallel")}
    assert layout == {"--pipeline-parallel": 3, "--tensor-parallel": 8, "--context-parallel": 2,
                      "--expert-parallel": 16}
    assert int(_flag(argv, "--rl-max-tokens-per-gpu")) == int(_flag(argv, "--seq-len")) // 2
    assert not any(a.startswith("--modal-") for a in argv), "an ssh island takes no Modal flags"


def test_the_runbook_links_the_kit_files():
    text = RUNBOOK.read_text()
    for name in (INVENTORY.name, RUN.name):
        assert name in text, name


def _filled_argv(run: Path, tmp_path: Path) -> list[str]:
    from yeto.rl.adapters.miles.models.partial_experts.selection import ExpertSelection

    selection = ExpertSelection.synthetic("uniform", model="kimi_k31", num_experts=896, num_trainable=128,
                                          ep_size=16, layers=range(1, 93))
    selection_path = tmp_path / "expert_selection.ep16.json"
    selection.write(selection_path)
    fill = {
        "<run>": "bmkit", "<bucket>": "bmbucket", "<store-endpoint>": "https://store.example",
        "<inventory.b300-6x8-baremetal.yaml>": str(INVENTORY),
        "<expert_selection.ep16.json>": str(selection_path),
        "<algorithm-spec.async-age12.json>": str(K31 / "algorithm-spec.async-age12.json"),
        "<model-revision>": "a" * 40, "<data-revision>": "b" * 40, "<prompt-dataset>": "org/prompts",
        "<module:function>": "yeto.rl.numeric_answer_reward:score", "<total-steps>": "100",
        "<weights-volume>": "k31-weights", "<modal-secret>": "k31-store",
    }
    out = []
    for arg in json.loads(run.read_text())["trainer"]["argv"]:
        for key, value in fill.items():
            arg = arg.replace(key, value)
        assert "<" not in arg, f"unfilled placeholder in {arg!r}"
        out.append(arg)
    return out


@pytest.mark.parametrize("run", [RUN, K31 / "run.agentic-262k.json"], ids=lambda p: p.name)
def test_trainer_argv_passes_the_launch_dry_run(run, tmp_path, monkeypatch, capsys):
    """Both 262k kits (bare metal and Modal) pass the launcher's own dry run: the algorithm spec
    fits --rl-loss score_centering, and --rl-round-cuts off carries no cut-only flag."""
    import yeto.cli as cli
    import yeto.launcher as launcher

    for key, value in (("YETO_RL_STORE_S3_ACCESS_KEY_ID", "test"), ("YETO_RL_STORE_S3_SECRET_ACCESS_KEY", "test"),
                       ("HF_HUB_OFFLINE", "1")):
        monkeypatch.setenv(key, value)
    import yeto.modal_runner as modal_runner

    monkeypatch.setattr(modal_runner, "modal_available", lambda: True)  # CI has no Modal token
    calls = []
    monkeypatch.setattr(launcher, "run", lambda *a, **k: calls.append("run"))
    monkeypatch.setattr(cli, "_spawn_worker", lambda *a, **k: calls.append("spawn"))
    monkeypatch.setattr(cli.runs, "create_run", lambda *a, **k: calls.append("create"))
    args = cli.parse_args([*_filled_argv(run, tmp_path), "--dry-run", "--preflight-model", "off"])
    assert cli.cmd_launch(args) == 0
    out = capsys.readouterr().out
    plan = json.loads(out[:out.rindex("}") + 1])
    assert plan["dry_run"] and plan["total_gpus"] == 48 and calls == []
    (island,) = plan["island_requests"]
    cloud = "ssh" if run == RUN else "modal"
    assert (island["cloud"], island["gpu"], island["num_nodes"], island["gpus_per_node"]) == (cloud, "B300", 6, 8)
    assert "--rl-sample-role trainer" in island["learner_command"]


def test_both_262k_kits_split_31_31_31_and_the_modal_kit_keeps_a_45_min_idle_window():
    """93 layers split evenly over PP3 (no split flag); the Modal kit keeps idle shutdown on
    (paid by the hour) with a window above a 262k step of about 25-35 min."""
    modal = [str(a) for a in json.loads((K31 / "run.agentic-262k.json").read_text())["trainer"]["argv"]]
    extra = [a for a in modal if a.startswith("YETO_MILES_TRAIN_EXTRA_ARGV=")]
    assert len(extra) == 1 and "pipeline-num-layers" not in extra[0]
    assert "first_stage_grad.before_train_step" in extra[0]
    assert not any("pipeline-num-layers" in a for a in _argv())
    assert (_flag(modal, "--idle-warn-minutes"), _flag(modal, "--idle-stop-minutes")) == ("45", "45")
    assert _flag(_argv(), "--idle-warn-minutes") == "0"
