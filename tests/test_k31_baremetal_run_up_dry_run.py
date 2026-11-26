"""The customer's bare-metal K3.1 run (examples/k31/run.baremetal-b300-262k.json) through
``yeto run up`` as the customer runs it: credentials set, the six trainer nodes reachable
over ssh (faked, every endpoint answers), the National Compute pool dry run and the
sandbox status faked to pass, the trainer's ``yeto launch --dry-run`` run in this process.

* ``--dry-run``: the normalized output is the one recorded on the code before the own-node
  egress check (tests/fixtures/own_cluster/k31_baremetal_dry_run.main.txt), line for
  line, and no node is contacted.
* ``--dry-run --check-nodes``: that output plus only the egress check's lines, listed here
  one by one.
* the real ``run up``: the egress check runs after the trainer's cluster preflight and
  before anything is ordered; a node that does not reach the store orders nothing."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
K31 = ROOT / "examples" / "k31"
RUN = "run.baremetal-b300-262k.json"
GOLDEN = ROOT / "tests" / "fixtures" / "own_cluster" / "k31_baremetal_dry_run.main.txt"
HOME = str(Path.home())  # read at import, before a test moves HOME: some paths are fixed when yeto loads
RECORD_ENV = "YETO_RECORD_K31_BAREMETAL_DRY_RUN"  # set: write GOLDEN from this tree (the code before the change)
NODES = [f"b300-{i}" for i in range(6)]
STORE, REGISTRY = "https://store.example", "https://ghcr.io/"
# the only lines the egress check adds to the customer's dry run
EGRESS_LINES = [  # with --check-nodes
    "dry run: own nodes " + ", ".join(NODES) + f": checking over ssh that they reach store {STORE}, "
    f"registry {REGISTRY}",
    *[line for node in NODES for line in (
        f"dry run:   [PASS] {node} reach_store: the store endpoint {STORE} (s3://bmbucket/runs/bmkit): http 403",
        f"dry run:   [PASS] {node} reach_registry: the registry endpoint {REGISTRY} "
        "(ghcr.io/agentenv/yeto-miles-k31@sha256:<SHA256>): http 200")],
    "dry run:   18 pass, 0 warn, 0 fail, 0 skip",
]


class ReachableNodes:
    """The ssh of the preflight: every node answers, every endpoint gives an HTTP answer
    (except the store from the nodes in ``no_store``)."""

    calls: list[str] = []
    no_store: set[str] = set()

    def run(self, node, script, *, timeout=300.0):  # noqa: ARG002
        ReachableNodes.calls.append(node.name)
        if script.startswith("date"):
            return subprocess.CompletedProcess([], 0, f"{time.time():.3f}\n", "")
        out = []
        for name, body in re.findall(r"echo '@@yeto-section (\S+)'\n(.*?)\necho \"@@yeto-rc", script, re.S):
            if "docker info" not in body and node.name in ReachableNodes.no_store:
                out += [f"@@yeto-section {name}", "http 000", "curl: (7) Failed to connect", "@@yeto-rc 7"]
                continue
            code = "200" if "docker info" in body else "403"
            out += [f"@@yeto-section {name}", f"http {code}", "@@yeto-rc 0"]
        return subprocess.CompletedProcess([], 0, "\n".join(out) + "\n", "")


def _fill(tmp: Path) -> Path:
    from yeto.rl.adapters.miles.models.partial_experts.selection import ExpertSelection

    sel = tmp / "expert_selection.ep16.json"
    ExpertSelection.synthetic("uniform", model="kimi_k31", num_experts=896, num_trainable=128, ep_size=16,
                              layers=range(1, 93)).write(sel)
    (tmp / "envs.json").write_text("{}")
    (tmp / "store.env").write_text("R2_BUCKET=bmbucket\nAWS_ENDPOINT_URL=https://store.example\n")
    (tmp / "store.env").chmod(0o600)
    (tmp / "agentic.env").write_text("X=1\n")
    (tmp / "prompts.jsonl").write_text('{"prompt": "1+1?", "metadata": "2"}\n')
    for name in ("dial.key", "tunnel.key", "known_hosts"):
        (tmp / name).write_text("x\n")
        (tmp / name).chmod(0o600)
    for task in ("example_task_1", "example_task_2"):
        (tmp / "tasks" / task).mkdir(parents=True)
        (tmp / "tasks" / task / "task.toml").write_text("x = 1\n")
    (tmp / "nc-state").mkdir()
    (tmp / "nc-tool").mkdir()
    fill = {
        "<run>": "bmkit", "<bucket>": "bmbucket", "<store-endpoint>": STORE, "<profile>": "bmprofile",
        "<store-profile>": "bmprofile", "<nc-state-dir>": str(tmp / "nc-state"), "<nc-tool-dir>": str(tmp / "nc-tool"),
        "<envs.json>": str(tmp / "envs.json"), "<sandbox-user>": "u", "<sandbox-host>": "sandbox.example",
        "<pool.nc-mi355x-agentic-262k.json>": str(tmp / "pool.nc-mi355x-agentic-262k.json"),
        "<sampler-spec.nc-mi355x-vllm-agentic-262k.json>": str(tmp / "sampler-spec.nc-mi355x-vllm-agentic-262k.json"),
        "<inventory.b300-6x8-baremetal.yaml>": str(tmp / "inventory.b300-6x8-baremetal.yaml"),
        "<expert_selection.ep16.json>": str(sel),
        "<algorithm-spec.async-age12.json>": str(tmp / "algorithm-spec.async-age12.json"),
        "<model-revision>": "a" * 40, "<data-revision>": "b" * 40, "<prompt-dataset>": "org/prompts",
        "<module:function>": "yeto.rl.numeric_answer_reward:score", "<total-steps>": "100",
        "<weights-volume>": "k31-weights", "<modal-secret>": "k31-store", "<store-env-file>": str(tmp / "store.env"), "<agentic-env-file>": str(tmp / "agentic.env"),
        "<prompts.jsonl>": str(tmp / "prompts.jsonl"), "<nc-node-id>": "node1", "<peer-node-ip>": "192.0.2.2",
        "<models/prefix of Kimi-K3.1-Pro>": "models/Kimi-K3.1-Pro",
        "<sandbox dial key>": str(tmp / "dial.key"), "<sandbox tunnel key>": str(tmp / "tunnel.key"),
        "<sandbox known_hosts>": str(tmp / "known_hosts"), "<staged task directory>": str(tmp / "tasks"),
        "<low | high | max>": "high", "<sha256 of expert_selection.json>": ExpertSelection.load_file(sel).sha256,
    }
    for src in K31.iterdir():
        if src.suffix not in (".json", ".yaml"):
            continue
        text = src.read_text()
        accounts = iter(range(101, 200))
        text = re.sub(r"<account-id>", lambda _m: str(next(accounts)), text)
        for key, value in fill.items():
            text = text.replace(key, value)
        if src.name.startswith("pool.") and src.suffix == ".json":
            doc = json.loads(text)
            doc["accounts"] = [int(a) for a in doc.get("accounts", [])]
            text = json.dumps(doc, indent=1)
        (tmp / src.name).write_text(text)
    return tmp / RUN


def _normalize(text: str, tmp: Path) -> list[str]:
    text = text.replace(str(tmp), "<TMP>").replace(str(ROOT), "<ROOT>").replace(HOME, "<HOME>")
    # the island script carries the source tree's digest, which differs between any two trees
    text = re.sub(r"(island script: \d+ bytes, sha256 )[0-9a-f]{16}", r"\1<ISH>", text)
    text = re.sub(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z", "<TS>", text)
    text = re.sub(r"\b\d{8}T\d{6}Z\b", "<TS>", text)
    text = re.sub(r"\b[0-9a-f]{64}\b", "<SHA256>", text)
    text = re.sub(r"\b[0-9a-f]{40}\b", "<SHA1>", text)
    text = re.sub(r"\b[0-9a-f]{12}\b", "<SHA12>", text)
    lines = [re.sub(r"^<TS> \[rl up\] ", "", line) for line in text.splitlines()]
    return [line for line in lines if line.strip()]


def _run_up(tmp_path, monkeypatch, capsys, *, dry_run: bool = True, check_nodes: bool = False,
            rc_expected: int = 0, no_store: tuple[str, ...] = ()) -> list[str]:
    import yeto.cli as cli
    import yeto.modal_runner as modal_runner
    from yeto.cloud import s3_store
    from yeto.launch.preflight import cluster
    from yeto.rl import up

    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for key, value in (("YETO_RL_STORE_S3_ACCESS_KEY_ID", "test"), ("YETO_RL_STORE_S3_SECRET_ACCESS_KEY", "test"),
                       ("HF_HUB_OFFLINE", "1"), ("YETO_PREFLIGHT_MODEL", "off"), ("MODAL_DISABLE_API_PROXY", "1"),
                       ("YETO_NC_ACCOUNTS", "101,102,103,104")):
        monkeypatch.setenv(key, value)
    for key in (s3_store.ENDPOINT_ENV, s3_store.REGION_ENV, s3_store.PROFILE_ENV, "AWS_ENDPOINT_URL", "AWS_PROFILE",
                "YETO_NC_STATE_DIR", "NC_TOOL_DIR"):
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    monkeypatch.setattr(modal_runner, "modal_available", lambda: True)
    monkeypatch.setattr(cluster, "SshOps", ReachableNodes)
    ReachableNodes.calls, ReachableNodes.no_store = [], set(no_store)
    commands = []

    def run_cmd(self, argv, *, log, timeout=None):  # noqa: ARG001
        argv = [str(a) for a in argv]
        commands.append(argv)
        if argv[2] == "yeto.launch.nc" or "sandbox" in argv:  # the NC pool dry run, the sandbox status: pass
            print(f"(fake) {' '.join(argv[2:4])}: ok")
            return 0
        if argv[3] == "preflight":  # the trainer's cluster preflight (a real run up): passes
            return 0
        assert "--dry-run" in argv, f"run up ordered: {argv}"
        try:
            return int(cli.main(argv[3:]) or 0)
        except SystemExit as stop:
            return int(stop.code or 0)

    monkeypatch.setattr(up.Run, "_cmd", run_cmd)
    config = _fill(tmp_path)
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        rc = up.main(types.SimpleNamespace(config=str(config), rl_command="up", skip_preflight=False, dry_run=dry_run,
                                           check_nodes=check_nodes, no_monitor=True, trainer_now=False,
                                           allow_after_forced_stop=False))
    finally:
        os.chdir(cwd)
    captured = capsys.readouterr()
    assert rc == rc_expected, (captured.out + captured.err)[-4000:]
    assert all("--dry-run" in c or "status" in c or c[3] == "preflight" for c in commands)
    return _normalize(captured.out + captured.err, tmp_path)


def test_the_customer_dry_run_is_main_s_line_for_line_and_contacts_no_node(tmp_path, monkeypatch, capsys):
    lines = _run_up(tmp_path, monkeypatch, capsys)
    if os.environ.get(RECORD_ENV):
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text("\n".join(lines) + "\n")
    assert lines == GOLDEN.read_text().splitlines()
    assert ReachableNodes.calls == []
    assert "dry run: preflight ok, nothing ordered" in lines
    assert not any("spend estimate" in line for line in lines)  # an NC-pool run keeps its output


def test_the_customer_dry_run_with_check_nodes_adds_only_the_egress_lines(tmp_path, monkeypatch, capsys):
    lines = _run_up(tmp_path, monkeypatch, capsys, check_nodes=True)
    added = [line for line in lines if line in EGRESS_LINES]
    assert added == EGRESS_LINES  # every egress line, once, in this order
    assert [line for line in lines if line not in EGRESS_LINES] == GOLDEN.read_text().splitlines()
    assert sorted(set(ReachableNodes.calls)) == NODES


def test_the_customer_run_up_orders_nothing_when_a_node_does_not_reach_the_store(tmp_path, monkeypatch, capsys):
    from yeto.rl import up

    ordered = []
    for name in ("write_fleet_plan", "sandbox_prepare", "pool"):
        monkeypatch.setattr(up.Run, name, lambda self, _n=name: ordered.append(_n) or False)
    lines = _run_up(tmp_path, monkeypatch, capsys, dry_run=False, rc_expected=2, no_store=("b300-3",))
    assert ("cluster preflight:   [FAIL] b300-3 reach_store: b300-3 cannot reach the store endpoint "
            f"{STORE} (s3://bmbucket/runs/bmkit) (rc 7): http 000 curl: (7) Failed to connect -> open the "
            "node's egress to it (firewall, proxy: https_proxy in the node's environment, DNS)") in lines
    assert "cluster preflight:   17 pass, 0 warn, 1 fail, 0 skip" in lines  # 6 ssh + 12 endpoint checks
    assert "cluster preflight failed: nothing ordered" in lines and ordered == []
    # reached: the PASS lines, then run up goes on to its first order (stopped there)
    (tmp_path / "again").mkdir()
    lines = _run_up(tmp_path / "again", monkeypatch, capsys, dry_run=False, rc_expected=2)
    assert [line.replace("cluster preflight: ", "dry run: ", 1) for line in lines if "reach_" in line] == \
        [line for line in EGRESS_LINES if "reach_" in line]
    assert ordered == ["write_fleet_plan"]
