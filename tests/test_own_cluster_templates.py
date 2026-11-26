"""The own-cluster templates of examples/own-cluster (docs/OWN_CLUSTERS.md): an own
trainer with sky spot or Modal samplers, an own sampler pool, and a fleet of one own and
one cloud island. Each passes ``yeto run up --dry-run`` with its pool dry runs and the
trainer's launch dry run run in this process (no cloud, no sky API server, no Modal
call), its own nodes reach the store and the registry over a fake ssh, and the spend
counts only the cloud parts. A node that cannot reach its store refuses the run up,
naming the node and the endpoint, and nothing is ordered."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "examples" / "own-cluster"
TEMPLATES = ("run.own-trainer-sky-sampler.json", "run.own-trainer-modal-sampler.json", "run.own-sampler-pool.json",
             "run.fleet-own-and-cloud.json")
ENDPOINT = "https://acct.r2.example"
SAMPLER_IMAGE = "ghcr.io/example/vllm-qwen3@sha256:" + "0" * 64


class FakeSsh:
    """The ssh of the preflight (``SshOps``): the clock read and the egress probe, which
    reaches every endpoint except the ``(node, url)`` pairs in ``unreachable``."""

    unreachable: set[tuple[str, str]] = set()
    calls: list[tuple[str, str]] = []

    def run(self, node, script, *, timeout=300.0):  # noqa: ARG002
        FakeSsh.calls.append((node.name, script))
        if script.startswith("date"):
            return subprocess.CompletedProcess([], 0, f"{time.time():.3f}\n", "")
        out = []
        for name, body in re.findall(r"echo '@@yeto-section (\S+)'\n(.*?)\necho \"@@yeto-rc", script, re.S):
            url = re.search(r"(https?://[^'\s;]+)", body).group(1)
            if (node.name, url) in FakeSsh.unreachable:
                out += [f"@@yeto-section {name}", "curl: (6) Could not resolve host", "@@yeto-rc 6"]
            else:
                out += [f"@@yeto-section {name}", "http 403", "@@yeto-rc 0"]
        return subprocess.CompletedProcess([], 0, "\n".join(out) + "\n", "")


def _fill(tmp_path: Path) -> dict[str, Path]:
    (tmp_path / "store.env").write_text(f"R2_BUCKET=ocbucket\nAWS_ENDPOINT_URL={ENDPOINT}\n")
    (tmp_path / "store.env").chmod(0o600)
    (tmp_path / "sky-key").write_text("key\n")
    (tmp_path / "prompts.jsonl").write_text("".join(json.dumps({"question": f"{i}+1?", "answer": str(i + 1)}) + "\n"
                                                    for i in range(1000)))
    fill = {"<run>": "ocrun", "<bucket>": "ocbucket", "<sky-ssh-key>": str(tmp_path / "sky-key"),
            "<store.env>": str(tmp_path / "store.env"), "<store-endpoint>": ENDPOINT,
            "<model-revision>": "a" * 40, "<prompts.jsonl>": str(tmp_path / "prompts.jsonl"),
            "<algorithm-spec.async-age2-tis.json>": str(ROOT / "examples/k31/algorithm-spec.async-age2-tis.json"),
            "<modal-secret>": "oc-store", "<volume>": "oc-weights", "<sampler-image>": SAMPLER_IMAGE}
    for path in KIT.iterdir():
        if path.suffix == ".yaml" or path.name.startswith("sampler-spec"):
            fill[f"<{path.name}>"] = str(tmp_path / path.name)
    out = {}
    for path in KIT.iterdir():
        text = path.read_text()
        for key, value in sorted(fill.items(), key=lambda kv: -len(kv[0])):
            text = text.replace(key, value)
        if path.suffix == ".json":
            assert "<" not in json.dumps(json.loads(text)), f"an unfilled placeholder in {path.name}"
        (tmp_path / path.name).write_text(text)
        out[path.name] = tmp_path / path.name
    return out


@pytest.fixture
def kit(tmp_path, monkeypatch):
    """The filled templates; run up's subprocesses (pool and launch dry runs) run in this
    process; the ssh is :class:`FakeSsh`; nothing reaches a cloud."""
    import yeto.cli as cli
    import yeto.launcher as launcher
    import yeto.modal_runner as modal_runner
    from yeto import launch_preflight
    from yeto.cloud import s3_store
    from yeto.launch.nc import cli as nc_cli
    from yeto.launch.nc import pool_run
    from yeto.launch.preflight import cluster
    from yeto.rl import up

    monkeypatch.setattr(up, "STATE_ROOT", tmp_path / "rl-up")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for key, value in (("YETO_RL_STORE_S3_ACCESS_KEY_ID", "test"), ("YETO_RL_STORE_S3_SECRET_ACCESS_KEY", "test"),
                       ("HF_HUB_OFFLINE", "1"), ("YETO_PREFLIGHT_MODEL", "off"), ("MODAL_TOKEN_ID", "x"),
                       ("MODAL_TOKEN_SECRET", "y")):
        monkeypatch.setenv(key, value)
    for key in (s3_store.ENDPOINT_ENV, s3_store.REGION_ENV, s3_store.PROFILE_ENV, "AWS_ENDPOINT_URL", "AWS_PROFILE"):
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    monkeypatch.setattr(modal_runner, "modal_available", lambda: True)
    monkeypatch.setattr(launch_preflight, "modal_sdk_problem", lambda *a, **k: None)
    monkeypatch.setattr(pool_run, "model_problems", lambda uri, env_file: [])  # no bucket LIST
    monkeypatch.setattr(cluster, "SshOps", FakeSsh)
    FakeSsh.unreachable, FakeSsh.calls = set(), []
    ordered = []
    monkeypatch.setattr(launcher, "run", lambda *a, **k: ordered.append("launcher.run"))
    monkeypatch.setattr(cli, "_spawn_worker", lambda *a, **k: ordered.append("spawn"))
    commands = []

    def in_process(self, argv, *, log, timeout=None):  # noqa: ARG001
        argv = [str(a) for a in argv]
        commands.append(argv)
        assert "--dry-run" in argv, f"a dry run ran a command that is not a dry run: {argv}"
        module, rest = argv[2], argv[3:]
        main = nc_cli.main if module == "yeto.launch.nc" else cli.main
        try:
            return int(main(rest) or 0)
        except SystemExit as stop:
            return int(stop.code or 0)

    monkeypatch.setattr(up.Run, "_cmd", in_process)
    files = _fill(tmp_path)
    return types.SimpleNamespace(files=files, commands=commands, ordered=ordered, up=up)


def _dry_run(kit, name: str, capsys) -> tuple[int, str]:
    args = types.SimpleNamespace(config=str(kit.files[name]), rl_command="up", skip_preflight=False, dry_run=True,
                                 no_monitor=True, trainer_now=True, allow_after_forced_stop=False)
    rc = kit.up.main(args)
    captured = capsys.readouterr()
    return rc, captured.out + captured.err


@pytest.mark.parametrize("name", TEMPLATES)
def test_every_template_passes_the_run_up_dry_run(kit, name, capsys):
    rc, out = _dry_run(kit, name, capsys)
    assert rc == 0, out[-4000:]
    assert "dry run: preflight ok, nothing ordered" in out
    assert kit.ordered == []
    assert all("--dry-run" in c for c in kit.commands) and kit.commands



def _cfg(kit, name: str) -> dict:
    return json.loads(kit.files[name].read_text())


def _reach_calls() -> dict[str, list[str]]:
    """The egress probes FakeSsh ran, by node: the endpoint URLs of each."""
    out: dict[str, list[str]] = {}
    for node, script in FakeSsh.calls:
        if "@@yeto-section reach" in script:
            out.setdefault(node, []).extend(re.findall(r"(https?://[^'\s;]+)", script))
    return {node: sorted(set(urls)) for node, urls in out.items()}


TRAINER_NODES = ["own-b300-0", "own-b300-1"]
SAMPLER_NODES = ["own-smp-0", "own-smp-1", "own-smp-2"]
REGISTRY = "https://ghcr.io/"  # the registry of the trainer image and of the sampler image
REACHED = sorted([ENDPOINT, REGISTRY])


@pytest.mark.parametrize("name, nodes", [
    ("run.own-trainer-sky-sampler.json", TRAINER_NODES),
    ("run.own-trainer-modal-sampler.json", TRAINER_NODES),
    ("run.own-sampler-pool.json", SAMPLER_NODES),
    ("run.fleet-own-and-cloud.json", TRAINER_NODES),
])
def test_the_dry_run_checks_every_own_node_reaches_the_store_and_the_registry(kit, name, nodes, capsys):
    rc, out = _dry_run(kit, name, capsys)
    assert rc == 0, out[-3000:]
    assert _reach_calls() == {node: REACHED for node in nodes}
    assert f"dry run: own nodes {', '.join(nodes)}: checking over ssh that they reach store {ENDPOINT}" in out


def test_own_trainer_with_sky_spot_samplers_spends_only_on_the_pool(kit, capsys):
    cfg = _cfg(kit, "run.own-trainer-sky-sampler.json")
    assert cfg["trainer"]["usd_per_h"] == 0 and "ssh:2x8xb300" in cfg["trainer"]["argv"]
    (pool,) = cfg["samplers"]
    offer = pool["provider"]["offers"][0]["resources"]
    assert pool["provider"]["kind"] == "sky" and offer["use_spot"] is True and offer["accelerators"] == "L40S:1"
    rc, out = _dry_run(kit, "run.own-trainer-sky-sampler.json", capsys)
    assert rc == 0, out[-3000:]
    assert ("dry run: spend estimate $1.50/h = pools $1.50/h (sky-l40s (sky) 2 x $0.7492) + trainer $0/h "
            "(own nodes: ssh:2x8xb300); over the 6 h wall clock $8.99 (budget.usd $30)") in out
    assert "sky offer L40S:1@nebius/eu-north1/spot" in out and "estimated spend: $0.75 per node hour" in out


def test_own_trainer_with_modal_samplers_spends_only_on_the_pool(kit, capsys):
    rc, out = _dry_run(kit, "run.own-trainer-modal-sampler.json", capsys)
    assert rc == 0, out[-3000:]
    line = next(line for line in out.splitlines() if "dry run: spend estimate" in line)
    assert "pools $5.68/h (modal-l40s (modal) 2 x $2.83877) + trainer $0/h (own nodes: ssh:2x8xb300)" in line
    assert line.split("dry run: spend estimate ")[1].startswith("$5.68/h")


def test_an_own_sampler_pool_bills_nothing_and_runs_on_the_node_runtime(kit, capsys):
    cfg = _cfg(kit, "run.own-sampler-pool.json")
    (pool,) = cfg["samplers"]
    assert pool["provider"]["kind"] == "ssh" and len(pool["provider"]["nodes"]) == pool["target"] + 1  # one spare
    from yeto.launch import ssh_sampler_pool

    assert "node_runtime" not in pool["provider"] and "node_runtime" not in pool  # the ssh pool's default:
    assert ssh_sampler_pool.parse_config(pool).node_runtime is True
    rc, out = _dry_run(kit, "run.own-sampler-pool.json", capsys)
    assert rc == 0, out[-3000:]
    assert ("dry run: spend estimate $8.00/h = pools $0.00/h (own-h100 (ssh) 2 x $0) + trainer $8/h; "
            "over the 6 h wall clock $48.00 (budget.usd $60)") in out
    assert "estimated spend: $0.00 per node hour" in out


def test_a_fleet_of_an_own_and_a_cloud_island_counts_the_own_trainer_at_zero(kit, capsys):
    rc, out = _dry_run(kit, "run.fleet-own-and-cloud.json", capsys)
    assert rc == 0, out[-3000:]
    assert "fleet: island 0 own: learner 0, store trainer0/" in out and "trainer ssh:2x8xb300 at $0/h" in out
    assert "fleet: island 1 cloud: learner 1" in out and "trainer modal:1x2xb300 at $17/h" in out
    assert ("fleet: budget estimate $21.72/h = pools $4.34/h (i0-sky-l40s 2 x $0.7492, i1-modal-l40s 1 x $2.83877) "
            "+ trainers $17.00/h (own $0, cloud $17) + syncer $0.38/h") in out
    assert "dry run: spend estimate" not in out  # the fleet's own estimate, not a second one
    launches = [c for c in kit.commands if "launch" in c]
    assert launches and all(c[c.index("--gpu") + 1] == "ssh:2x8xb300,modal:1x2xb300" for c in launches)


@pytest.mark.parametrize("name, node, url", [
    ("run.own-trainer-sky-sampler.json", "own-b300-1", ENDPOINT),
    ("run.fleet-own-and-cloud.json", "own-b300-0", REGISTRY),
    ("run.own-sampler-pool.json", "own-smp-2", ENDPOINT),
])
def test_a_node_that_cannot_reach_an_endpoint_refuses_the_run_up_and_names_both(kit, name, node, url, capsys):
    FakeSsh.unreachable = {(node, url)}
    rc, out = _dry_run(kit, name, capsys)
    assert rc == 2, out[-3000:]
    role = "store" if url == ENDPOINT else "registry"
    (line,) = [line for line in out.splitlines() if "[FAIL]" in line]
    assert f"[FAIL] {node} reach_{role}: {node} cannot reach the {role} endpoint {url}" in line
    assert "Could not resolve host" in line
    assert "cluster preflight failed: nothing ordered" in out and "dry run: preflight ok" not in out
    assert kit.ordered == [] and all("--dry-run" in c for c in kit.commands)


def test_a_node_that_takes_no_ssh_refuses_the_run_up(kit, monkeypatch, capsys):
    reached = FakeSsh.run

    def no_ssh(self, node, script, *, timeout=300.0):
        if node.name == "own-b300-0":
            return subprocess.CompletedProcess([], 255, "", "ssh: connect to host port 22: Connection timed out")
        return reached(self, node, script, timeout=timeout)

    monkeypatch.setattr(FakeSsh, "run", no_ssh)
    rc, out = _dry_run(kit, "run.own-trainer-sky-sampler.json", capsys)
    assert rc == 2
    assert "[FAIL] own-b300-0 ssh: ssh ops@192.0.2.30 failed (rc 255)" in out
    assert "own-b300-1" in _reach_calls() and "own-b300-0" not in _reach_calls()


def test_a_cloud_only_trainer_contacts_no_node(kit, capsys):
    path = kit.files["run.own-trainer-sky-sampler.json"]
    cfg = json.loads(path.read_text())
    argv = cfg["trainer"]["argv"]
    argv[argv.index("--gpu") + 1] = "modal:1x2xh100"
    i = argv.index("--rl-node-inventory")
    del argv[i:i + 2]
    cfg["trainer"]["usd_per_h"] = 8.0
    path.write_text(json.dumps(cfg))
    rc, out = _dry_run(kit, "run.own-trainer-sky-sampler.json", capsys)
    assert rc == 0, out[-3000:]
    assert FakeSsh.calls == [] and "own nodes" not in out and "dry run: spend estimate" not in out


def test_the_templates_run_on_the_shipped_inventory_and_the_docs_name_them():
    from yeto.launch.node_inventory import island_nodes, load_inventory

    nodes = load_inventory(KIT / "inventory.own-nodes.yaml")
    assert [n.name for n in island_nodes(nodes, "trainer")] == TRAINER_NODES
    assert [n.name for n in nodes if "sampler" in n.roles] == SAMPLER_NODES
    docs = (ROOT / "docs" / "OWN_CLUSTERS.md").read_text()
    for name in (*TEMPLATES, "inventory.own-nodes.yaml"):
        assert name in docs, name
    for heading in ("## Trainer with cloud samplers", "## Own sampler pools"):
        assert heading in docs


# -- the endpoints -------------------------------------------------------------------------

@pytest.mark.parametrize("image, url", [
    ("docker:ghcr.io/agentenv/yeto-miles@sha256:" + "1" * 64, REGISTRY),
    ("registry.example:5000/team/img:tag", "https://registry.example:5000/"),
    ("vllm/vllm-openai:latest", "https://registry-1.docker.io/"),
    ("ubuntu", "https://registry-1.docker.io/"),
    ("localhost/img", "https://localhost/"),
])
def test_the_registry_of_an_image(image, url):
    from yeto.launch.preflight.gate import registry_endpoint

    assert registry_endpoint(image).url == url


def test_no_registry_for_no_image_or_a_placeholder():
    from yeto.launch.preflight.gate import registry_endpoint

    assert registry_endpoint(None) is None and registry_endpoint("<sampler-image>") is None


@pytest.mark.parametrize("store, env, url", [
    ("s3://b/runs/r", {"YETO_RL_STORE_S3_ENDPOINT_URL": "https://x.r2.example"}, "https://x.r2.example"),
    ("s3://b/runs/r", {"AWS_ENDPOINT_URL": "https://minio.site:9000"}, "https://minio.site:9000"),
    ("s3://b/runs/r", {"AWS_REGION": "eu-west-1"}, "https://s3.eu-west-1.amazonaws.com"),
    ("s3://b/runs/r", {}, "https://s3.amazonaws.com"),
    ("oss://b/runs/r", {"YETO_RL_STORE_OSS_ENDPOINT_URL": "oss-cn-hangzhou.aliyuncs.com"},
     "https://oss-cn-hangzhou.aliyuncs.com"),
])
def test_the_store_endpoint(store, env, url, monkeypatch):
    from yeto.launch.preflight.gate import store_endpoint

    for key in ("AWS_REGION", "AWS_DEFAULT_REGION", "AWS_PROFILE", "YETO_RL_STORE_S3_PROFILE",
                "YETO_RL_STORE_S3_REGION"):
        monkeypatch.delenv(key, raising=False)
    endpoint = store_endpoint(store, env)
    assert endpoint.url == url and endpoint.role == "store" and endpoint.source == store


@pytest.mark.parametrize("store", ["modal-volume://v", "/shared/store", "oss://b/r", None])
def test_no_store_endpoint_without_a_network_store(store):
    from yeto.launch.preflight.gate import store_endpoint

    assert store_endpoint(store, {}) is None


def test_the_launch_endpoints_take_the_argv_endpoint_and_the_default_image():
    from yeto.launch.preflight.gate import launch_endpoints
    from yeto.rl.adapters.miles import pins

    argv = ["--rl-checkpoint-store", "s3://b/runs/r", "--rl-object-store-endpoint", "https://e.example"]
    store, registry = launch_endpoints(argv, {})
    assert (store.url, store.source) == ("https://e.example", "s3://b/runs/r")
    assert registry.source == pins.default_rl_image().removeprefix("docker:")
    (_, registry) = launch_endpoints([*argv, "--rl-image", "docker:quay.io/x/y:1"], {})
    assert registry.url == "https://quay.io/"


def test_the_probe_marks_any_http_answer_reached_and_a_failed_connect_not():
    from yeto.launch.preflight import cluster

    endpoints = [cluster.Endpoint("store", ENDPOINT, "s3://b/r"),
                 cluster.Endpoint("registry", "https://registry.example:5000/")]
    script = cluster.probe_script(cluster.reach_sections(endpoints))
    assert "/dev/tcp/registry.example/5000" in script
    text = "\n".join(["@@yeto-section reach0", "http 403", "@@yeto-rc 0",
                      "@@yeto-section reach1", "curl: (28) Connection timed out", "@@yeto-rc 28"])
    store, registry = cluster.reach_checks("n0", endpoints, cluster.parse_sections(text))
    assert store.status == cluster.PASS and registry.status == cluster.FAIL
    assert "n0 cannot reach the registry endpoint https://registry.example:5000/ (rc 28)" in registry.detail
    (missing,) = cluster.reach_checks("n0", endpoints[:1], {})
    assert missing.status == cluster.FAIL and "printed nothing" in missing.detail


# -- the probe itself, in bash, behind proxies -----------------------------------------------
# A fake curl stands for the node's network: FAKE_DIRECT_BLOCKED lets only a proxied
# request out, FAKE_TLS_INTERCEPT fails certificate checks (a proxy that re-signs TLS),
# FAKE_CODE is the HTTP status that comes back. A fake docker answers ``docker info`` with
# the daemon's proxy. PATH holds these and nothing else.
FAKE_CURL = r"""#!/bin/sh
printf '%s ' curl "$@" >> "$FAKE_LOG"; printf '\n' >> "$FAKE_LOG"
k=; f=; proxy=${HTTPS_PROXY:-$https_proxy}
while [ $# -gt 0 ]; do
  case "$1" in -k) k=1;; -f|--fail) f=1;; --proxy) shift; proxy=$1;; -w|-o|-m|--connect-timeout) shift;; esac; shift
done
code=${FAKE_CODE:-200}
if [ -z "$proxy" ] && [ -n "$FAKE_DIRECT_BLOCKED" ]; then
  echo "http 000"; echo "curl: (7) Failed to connect" >&2; exit 7
fi
if [ -n "$FAKE_TLS_INTERCEPT" ] && [ -z "$k" ]; then
  echo "http 000"; echo "curl: (60) SSL certificate problem: self-signed certificate in certificate chain" >&2; exit 60
fi
if [ -n "$f" ] && [ "$code" -ge 400 ]; then
  echo "http $code"; echo "curl: (22) The requested URL returned error: $code" >&2; exit 22
fi
echo "http $code"
"""
FAKE_DOCKER = r"""#!/bin/sh
printf '%s ' docker "$@" >> "$FAKE_LOG"; printf '\n' >> "$FAKE_LOG"
[ "$1" = info ] && printf '%s\n' "$FAKE_DOCKER_PROXY"
"""
PROXY = "http://proxy.example:3128"


class LocalShell:
    """``SshOps`` that runs the node's script in a local bash with only the fake tools."""

    def __init__(self, tmp: Path, *, curl: bool = True, **fake: str):
        tmp = Path(tempfile.mkdtemp(dir=tmp))
        self.bin = tmp / "bin"
        self.bin.mkdir()
        self.log = tmp / "calls.log"
        for name, body in (("curl", FAKE_CURL), ("docker", FAKE_DOCKER))[0 if curl else 1:]:
            (self.bin / name).write_text(body)
            (self.bin / name).chmod(0o755)
        self.env = {"PATH": str(self.bin), "FAKE_LOG": str(self.log), **fake}

    def run(self, node, script, *, timeout=300.0):  # noqa: ARG002
        if script.startswith("date"):
            return subprocess.CompletedProcess([], 0, f"{time.time():.3f}\n", "")
        done = subprocess.run(["/bin/bash", "-c", script], env=self.env, capture_output=True, text=True, timeout=60)
        return subprocess.CompletedProcess([], done.returncode, done.stdout + done.stderr, "")

    def calls(self) -> list[str]:
        return self.log.read_text().splitlines() if self.log.exists() else []


def _egress(ops, env=None):
    from yeto.launch.node_inventory import Node
    from yeto.launch.preflight import cluster

    node = Node(name="n0", host="ops@192.0.2.9", device_family="cuda", cpu_arch="x86_64", card="B300",
                cards_per_node=8, hbm_gib=288, host_memory_gib=2048, env=env or {})
    endpoints = [cluster.Endpoint("store", "https://store.example/", "s3://b/r"),
                 cluster.Endpoint("registry", "https://registry.example:5000/", "registry.example:5000/img:1")]
    report = cluster.egress_preflight([node], endpoints, ops=ops, log=lambda _m: None)
    return {c.name: c for c in report.checks if c.name.startswith("reach_")}, report


@pytest.mark.parametrize("code", ["200", "401", "403", "404"])
def test_behind_a_tls_intercepting_proxy_any_http_answer_is_reached(tmp_path, code):
    from yeto.launch.preflight import cluster, gate

    ops = LocalShell(tmp_path, FAKE_TLS_INTERCEPT="1", FAKE_CODE=code)
    checks, report = _egress(ops)
    assert {c.status for c in checks.values()} == {cluster.PASS}
    assert all(c.detail.endswith(f"http {code}") for c in checks.values())
    curls = [c for c in ops.calls() if c.startswith("curl ")]
    assert len(curls) == 2 and all(" -k " in c for c in curls)  # the certificate is not what is checked
    assert gate.egress_lines(report) == [
        f"  [PASS] n0 reach_store: the store endpoint https://store.example/ (s3://b/r): http {code}",
        "  [PASS] n0 reach_registry: the registry endpoint https://registry.example:5000/ "
        f"(registry.example:5000/img:1): http {code}",
        "  3 pass, 0 warn, 0 fail, 0 skip"]  # with the ssh check, printed only when it does not pass


def test_the_registry_goes_by_the_docker_daemon_proxy_and_the_store_by_the_shell_one(tmp_path):
    from yeto.launch.preflight import cluster

    ops = LocalShell(tmp_path, FAKE_DIRECT_BLOCKED="1", FAKE_DOCKER_PROXY=PROXY)
    checks, report = _egress(ops)
    assert checks["reach_registry"].status == cluster.PASS  # what docker pull would use
    assert any(c.startswith("curl ") and f"--proxy {PROXY} https://registry.example:5000/" in c for c in ops.calls())
    assert checks["reach_store"].status == cluster.FAIL  # the node's shell has no proxy: the store is not reached
    assert "cannot reach the store endpoint https://store.example/ (s3://b/r) (rc 7)" in checks["reach_store"].detail
    assert not report.ok


def test_the_inventory_env_proxy_reaches_both_endpoints(tmp_path):
    from yeto.launch.preflight import cluster

    blocked, _ = _egress(LocalShell(tmp_path, FAKE_DIRECT_BLOCKED="1"))
    assert {c.status for c in blocked.values()} == {cluster.FAIL}
    ops = LocalShell(tmp_path, FAKE_DIRECT_BLOCKED="1")
    checks, report = _egress(ops, env={"HTTPS_PROXY": PROXY, "NCCL_IB_HCA": "mlx5"})
    assert {c.status for c in checks.values()} == {cluster.PASS} and report.ok
    assert not any(c.startswith("docker ") for c in ops.calls())  # the shell's proxy is the registry's too


def test_no_curl_behind_a_proxy_warns_and_does_not_fail(tmp_path):
    from yeto.launch.preflight import cluster, gate

    checks, report = _egress(LocalShell(tmp_path, curl=False), env={"https_proxy": PROXY})
    assert {c.status for c in checks.values()} == {cluster.WARN} and report.ok
    assert "no curl on the node and a proxy is set" in checks["reach_store"].detail
    assert gate.egress_lines(report)[-1] == "  1 pass, 2 warn, 0 fail, 0 skip"


# -- what the templates needed of the pools -------------------------------------------------

def test_a_sky_pool_on_the_node_runtime_may_pull_the_model_to_any_models_path(tmp_path, monkeypatch):
    from yeto.launch.nc import pool_run

    monkeypatch.setattr(pool_run, "model_problems", lambda uri, env_file: [])
    sky = {"provider": {"kind": "sky", "resources": {"accelerators": "L40S:1"}}, "run_prefix": "r",
           "model_pull": {"src_key": "models/Qwen3-4B", "dest": "/models/Qwen3-4B"}}
    assert pool_run.store_layout_problems(sky, tmp_path / "env", "b") == []
    bad = dict(sky, model_pull={"src_key": "m", "dest": "/opt/models/x"})
    (problem,) = pool_run.store_layout_problems(bad, tmp_path / "env", "b")
    assert "a path under /models/" in problem
    old_route = dict(sky, node_runtime=False)  # the sky route's own start mounts the K3.1 directory only
    assert "mounts only" in pool_run.store_layout_problems(old_route, tmp_path / "env", "b")[0]
    nc = {"accounts": [1], "run_prefix": "r", "model_pull": {"src_key": "m", "dest": "/models/Qwen3-4B"}}
    assert "mounts only" in pool_run.store_layout_problems(nc, tmp_path / "env", "b")[0]


def test_an_own_ssh_pool_needs_no_price_under_a_spend_cap():
    from yeto.rl import up_consistency

    ssh = {"provider": {"kind": "ssh", "inventory": "x.yaml"}}
    assert up_consistency.price_problem("own", None, ssh) is None
    assert up_consistency.price_problem("own", 0.0, ssh) is None
    modal = {"provider": {"kind": "modal", "gpu": "L40S", "price_usd_h": 0}}
    assert "priced in USD" in up_consistency.price_problem("m", None, modal)
