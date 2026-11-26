"""``yeto image adopt`` steps 1-4 (own-cluster-fleet-stability 7.2): manifest and digest,
the probe on one node, the fit to every node, the parallel pre-pull with the image root
check. Each failure stops at its step, names the reason and leaves the catalog alone."""

from __future__ import annotations

import dataclasses
import json
import shlex
import subprocess
import urllib.error

import pytest

from tests.test_cluster_preflight import Behavior, FakeCluster, b300, disk_line
from yeto.launch import image_adopt as ia
from yeto.launch import image_manifest as im
from yeto.launch import site_images as si

DIGEST = "sha256:" + "d" * 64
TAGGED = "reg.example/site/miles-new:k99"
PINNED = f"reg.example/site/miles-new@{DIGEST}"
MILES_FACTS = {"uname_m": "x86_64", "python": "3.12.3", "python_path": "/usr/bin/python3", "torch": "2.9.1+cu130",
               "cuda": "13.0", "hip": None, "arch_list": ["sm_90", "sm_100", "sm_103", "compute_103"],
               "packages": {"torch": "2.9.1+cu130", "megatron-core": "0.15.0", "miles": "0.3.0", "ray": "2.52.0"},
               "lock": {"image": "miles-new", "versions": {"torch": "2.9.1"}}}


class _Resp:
    def __init__(self, status, headers=None, body=b""):
        self.status, self.headers, self._body = status, headers or {}, body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def registry(manifests: dict[str, int], digest: str = DIGEST):
    """A registry that answers a manifest HEAD by tag or digest (anonymous)."""
    seen = []

    def urlopen(req, timeout=None):
        seen.append(req.full_url)
        reference = req.full_url.rsplit("/manifests/", 1)[1]
        code = manifests.get(reference, 404)
        if code == 200:
            return _Resp(200, {"Docker-Content-Digest": digest})
        raise urllib.error.HTTPError(req.full_url, code, "x", {}, None)

    return urlopen, seen


def probe_out(facts=None, **over):
    facts = {**MILES_FACTS, **(facts or {}), **over}
    return "\n".join(f"{ia.FACT} {k} {json.dumps(v)}" for k, v in facts.items() if v is not None) + "\n"


class FakeNodes(FakeCluster):
    """The preflight's fake B300 cluster plus the adoption probe."""

    def __init__(self, nodes, *, probe=None, probe_rc=0, **per_node):
        super().__init__(nodes, **per_node)
        self.probe = probe_out() if probe is None else probe
        self.probe_rc = probe_rc

    def run(self, node, script, *, timeout=300.0):
        if ia.FACT in script:
            self.scripts.append((node.name, "probe"))
            return self.done(self.probe_rc, self.probe)
        return super().run(node, script, timeout=timeout)

    def kinds(self):
        return [("probe" if s == "probe" else s.split()[0] + " " + s.split()[1] if s.startswith("docker")
                 else "host" if "@@yeto-section" in s else s.split()[0]) for _, s in self.scripts]


def request(nodes, tmp_path, **kw):
    return ia.AdoptRequest(ref=kw.pop("ref", TAGGED), model="moonshotai/Kimi-K3.1-Pro", role=kw.pop("role", "trainer"),
                           nodes=tuple(nodes), catalog=str(tmp_path / "site-images.yaml"), **kw)


def adopter(req, ops, *, manifests=None, smoke=None):
    opener, seen = registry(manifests or {"k99": 200, DIGEST: 200})
    a = ia.Adopter(req, ops=ops, smoke=smoke, urlopen=opener, login=None, log=lambda m: None,
                   clock=lambda: FakeCluster.NOW)
    a.seen = seen
    return a


def passing_smoke(calls=None):
    def smoke(adopter, ref, facts, engine):
        if calls is not None:
            calls.append((ref, engine))
        return ia.StepResult("smoke", True, "one step trained, checkpoint saved and loaded", 1.0,
                             {"layers": 4})
    return smoke


@pytest.fixture(autouse=True)
def _registry_on(monkeypatch):
    monkeypatch.delenv(im.CHECK_ENV, raising=False)  # tests here pass their own fake registry


@pytest.fixture
def catalog_before(tmp_path):
    path = tmp_path / "site-images.yaml"
    path.write_text("version: 1\nimages: []\n")
    return path


# -- step 1: manifest -----------------------------------------------------------------


def test_a_tag_is_pinned_to_the_registry_digest():
    opener, seen = registry({"k99": 200})
    assert ia.pin(TAGGED, urlopen=opener) == PINNED
    assert seen == ["https://reg.example/v2/site/miles-new/manifests/k99"]
    opener, _ = registry({DIGEST: 200})
    assert ia.pin("docker:" + PINNED, urlopen=opener) == PINNED
    assert im.pin_digest("reg.example:5000/a/b:v1", urlopen=registry({"v1": 200})[0]) == \
        (im.PRESENT, f"reg.example:5000/a/b@{DIGEST}")


def test_a_registry_without_a_digest_header_is_not_pinned():
    def urlopen(req, timeout=None):
        return _Resp(200, {})
    assert im.pin_digest(TAGGED, urlopen=urlopen)[0] == im.UNKNOWN


def test_without_the_registry_check_only_a_digest_is_taken(monkeypatch):
    monkeypatch.setenv(im.CHECK_ENV, "off")
    assert ia.pin(PINNED) == PINNED
    with pytest.raises(ia.AdoptError, match="a tag cannot be pinned"):
        ia.pin(TAGGED)


def test_a_missing_manifest_stops_before_any_node(tmp_path, catalog_before):
    nodes = [b300(i) for i in range(2)]
    ops = FakeNodes(nodes)
    a = adopter(request(nodes, tmp_path), ops, manifests={"k99": 404}, smoke=passing_smoke())
    with pytest.raises(ia.AdoptError) as error:
        a.run()
    assert error.value.step == "manifest"
    assert "the registry has no manifest for reg.example/site/miles-new:k99" in str(error.value)
    assert ops.scripts == []  # no node was asked anything
    assert catalog_before.read_text() == "version: 1\nimages: []\n"
    assert [(s.name, s.ok) for s in a.report.steps] == [("manifest", False)]


def test_a_denied_manifest_names_the_registry_login(tmp_path):
    with pytest.raises(ia.AdoptError, match="give the registry login"):
        ia.pin(TAGGED, urlopen=registry({"k99": 403})[0])


# -- step 2: probe --------------------------------------------------------------------


def test_the_probe_runs_the_image_with_the_node_cards_and_reads_its_facts():
    node = b300(0)
    script = ia.probe_script(PINNED, node)
    argv = shlex.split(script)
    assert argv[:4] == ["docker", "run", "--rm", "--network"] and PINNED in argv
    assert list(node.docker_args) == argv[5:5 + len(node.docker_args)]  # the node's device flags
    facts = ia.parse_probe(probe_out() + "noise line\nYETO_IMAGE_FACT bad {not json\n")
    assert (facts.cpu_arch, facts.python, facts.cuda, facts.arch_list) == (
        "x86_64", "3.12.3", "13.0", ("sm_90", "sm_100", "sm_103", "compute_103"))
    assert facts.versions() == {"python": "3.12.3", "python_path": "/usr/bin/python3", "torch": "2.9.1+cu130",
                                "cuda": "13.0", "megatron-core": "0.15.0", "miles": "0.3.0", "ray": "2.52.0",
                                "image_lock": "miles-new"}


def test_the_probe_body_runs_under_bash_without_torch(tmp_path):
    """The script inside the container: uname, no lock, a Python without torch."""
    argv = shlex.split(ia.probe_script(PINNED, b300(0)))
    inner = argv[argv.index("-c") + 1]
    done = subprocess.run(["bash", "-c", inner], capture_output=True, text=True, timeout=60)
    facts = ia.parse_probe(done.stdout)
    assert done.returncode == 0, done.stderr
    assert facts.python and facts.cpu_arch in ("x86_64", "aarch64", "arm64")


def test_a_failed_probe_names_the_node_and_docker_line(tmp_path, catalog_before):
    nodes = [b300(0)]
    ops = FakeNodes(nodes, probe="docker: Error response from daemon: manifest unknown: manifest unknown\n",
                    probe_rc=125)
    with pytest.raises(ia.AdoptError, match=r"step probe failed: node b300-0: docker run of .* failed \(rc 125\): "
                                            r"docker: Error response from daemon: manifest unknown"):
        adopter(request(nodes, tmp_path), ops, smoke=passing_smoke()).run()
    assert catalog_before.read_text() == "version: 1\nimages: []\n"


# -- step 3: fit ----------------------------------------------------------------------


def test_an_image_built_above_the_nodes_card_stops_at_fit(tmp_path, catalog_before):
    a100 = [b300(i, card="A100-80GB", hbm_gib=80.0) for i in range(2)]
    ops = FakeNodes(a100, probe=probe_out(arch_list=["sm_90", "sm_100"]))
    a = adopter(request(a100, tmp_path), ops, smoke=passing_smoke())
    with pytest.raises(ia.AdoptError) as error:
        a.run()
    assert error.value.step == "fit"
    assert "the image is built for sm_90/sm_100 (lowest sm90); node b300-0 has A100-80GB (sm80)" in error.value.reason
    assert "node b300-1 has A100-80GB (sm80)" in error.value.reason
    assert ops.kinds() == ["probe"]  # nothing pulled
    assert catalog_before.read_text() == "version: 1\nimages: []\n"


@pytest.mark.parametrize("archs, card, ok", [
    (["sm_90", "compute_90"], "B300", True),      # PTX of sm90 compiles on sm103
    (["sm_100"], "B300", False),                  # cubin of sm100 only: not sm103
    (["sm_80", "sm_90"], "H100", True),
    (["gfx942"], "MI355X", False),
    (["gfx942", "gfx950"], "MI355X", True),
    ([], "B300", False),                          # no cards seen in the probe
])
def test_card_arch_rules(archs, card, ok):
    facts = ia.parse_probe(probe_out(arch_list=archs))
    family = "rocm" if card.startswith("MI") else "cuda"
    node = b300(0, card=card, device_family=family)
    assert (ia.card_arch_problem(facts, node) is None) is ok


def test_fit_refuses_another_cpu_a_missing_engine_and_a_cpu_only_torch(tmp_path):
    node = b300(0)
    req = request([node], tmp_path)
    arm = ia.parse_probe(probe_out(uname_m="aarch64"))
    assert ia.fit_problems(req, PINNED, arm, "miles")[0] == ["the image is aarch64; node b300-0 is x86_64"]
    bare = ia.parse_probe(probe_out(packages={"torch": "2.9.1"}))
    assert "the image has no trainer engine miles" in ia.fit_problems(req, PINNED, bare, "miles")[0][0]
    cpu = ia.parse_probe(probe_out(cuda=None))
    assert "has no CUDA build" in ia.fit_problems(req, PINNED, cpu, "miles")[0][0]
    vllm = request([node], tmp_path, role="sampler")
    assert "the image has no sampler engine vllm" in ia.fit_problems(vllm, PINNED, bare, "vllm")[0][0]


def test_the_trainer_floor_still_applies_to_a_known_trainer_image(tmp_path):
    a100 = b300(0, card="A100-80GB", hbm_gib=80.0)
    ref = "ghcr.io/owner/yeto-miles-k31@sha256:" + "ab" * 32
    facts = ia.parse_probe(probe_out(arch_list=["sm_80", "sm_90"]))
    problems, _ = ia.fit_problems(request([a100], tmp_path), ref, facts, "miles")
    assert len(problems) == 1 and "node b300-0: trainer image" in problems[0] and "sm90+" in problems[0]
    lifted = dataclasses.replace(request([a100], tmp_path), attention_backend="flash")
    problems, warnings = ia.fit_problems(lifted, ref, facts, "miles")
    assert problems == [] and "the flash attention backend takes another path" in warnings[0]


def test_a_catalog_sampler_image_is_held_to_its_record(tmp_path):
    from yeto.rl import sampler_pins as pins

    ref = pins.VLLM_SAMPLER_IMAGE_ROCM.removeprefix("docker:")
    facts = ia.parse_probe(probe_out(packages={"vllm": "0.19"}))
    problems, _ = ia.fit_problems(request([b300(0)], tmp_path, role="sampler"), ref, facts, "vllm")
    assert len(problems) == 1 and "it is a rocm image" in problems[0]


# -- step 4: pull ---------------------------------------------------------------------


def test_a_node_short_of_image_room_stops_the_pull_everywhere(tmp_path, catalog_before):
    nodes = [b300(i) for i in range(3)]
    low = Behavior(host={"disk": (0, "\n".join([disk_line("/", 80, 200, dev="/dev/sda2", mount="/"),
                                                disk_line("/var/lib/containerd", 40, mount="/"),
                                                disk_line("/mnt/nvme/docker", 3000, mount="/mnt/nvme")]))})
    ops = FakeNodes(nodes, **{"b300-2": low})
    a = adopter(request(nodes, tmp_path), ops, smoke=passing_smoke())
    with pytest.raises(ia.AdoptError) as error:
        a.run()
    assert error.value.step == "pull"
    assert "node b300-2: disk_images: /var/lib/containerd (containerd image store) on /: 40 GiB free" \
        in error.value.reason
    assert "b300-0" not in error.value.reason
    assert "docker pull" not in ops.kinds()  # no node pulled
    assert catalog_before.read_text() == "version: 1\nimages: []\n"


def test_a_failed_pull_names_the_node(tmp_path, catalog_before):
    nodes = [b300(i) for i in range(2)]
    bad = Behavior(pull=(1, "Error response from daemon: no space left on device"))
    ops = FakeNodes(nodes, **{"b300-1": bad})
    with pytest.raises(ia.AdoptError, match=r"step pull failed: node b300-1: image_pull: docker pull failed \(rc 1\)"):
        adopter(request(nodes, tmp_path), ops, smoke=passing_smoke()).run()
    assert ops.kinds().count("docker pull") == 2  # every node was pulled in parallel
    assert catalog_before.read_text() == "version: 1\nimages: []\n"


def test_an_unreachable_node_stops_the_pull(tmp_path):
    nodes = [b300(i) for i in range(2)]
    ops = FakeNodes(nodes, **{"b300-1": Behavior(ssh_rc=255)})
    with pytest.raises(ia.AdoptError, match="node b300-1: ssh: ssh ops@b300-1 failed"):
        adopter(request(nodes, tmp_path, node="b300-0"), ops, smoke=passing_smoke()).run()


# -- the whole run ----------------------------------------------------------------------


def test_every_step_passes_and_the_catalog_gets_the_record(tmp_path):
    nodes = [b300(i) for i in range(2)]
    ops = FakeNodes(nodes)
    calls = []
    report = adopter(request(nodes, tmp_path), ops, smoke=passing_smoke(calls)).run()
    assert [s.name for s in report.steps] == list(ia.STEPS) and report.ok
    assert calls == [(PINNED, "miles")]
    (rec,) = si.load(tmp_path / "site-images.yaml")
    assert (rec.ref, rec.role, rec.engine, rec.cpu_arch, rec.card_archs, rec.device_family, rec.status) == (
        PINNED, "trainer", "miles", "x86_64", ("sm103",), "cuda", "verified-smoke")
    assert rec.model == "moonshotai/Kimi-K3.1-Pro" and rec.versions["torch"] == "2.9.1+cu130"
    assert rec.evidence["node"] == "b300-0" and rec.evidence["result"] == "pass"
    assert rec.evidence["pulled"] == ["b300-0", "b300-1"] and rec.evidence["smoke"] == {"layers": 4}
    assert set(rec.evidence["steps"]) == set(ia.STEPS)
    assert si.match("kimi_k31", "trainer", nodes, path=tmp_path / "site-images.yaml") == [rec]
    assert any("recorded" in line for line in report.lines())


def test_a_failed_smoke_writes_nothing(tmp_path, catalog_before):
    nodes = [b300(0)]

    def smoke(adopter, ref, facts, engine):
        return ia.StepResult("smoke", False, "hot reload: the engine answered 500", 2.0)

    with pytest.raises(ia.AdoptError, match="step smoke failed: hot reload"):
        adopter_ = adopter(request(nodes, tmp_path, role="sampler"), FakeNodes(
            nodes, probe=probe_out(packages={"vllm": "0.19.0"})), smoke=smoke)
        adopter_.run()
    assert catalog_before.read_text() == "version: 1\nimages: []\n"


def test_no_smoke_wired_writes_nothing(tmp_path, catalog_before):
    nodes = [b300(0)]
    with pytest.raises(ia.AdoptError, match="step smoke failed: no trainer smoke"):
        adopter(request(nodes, tmp_path), FakeNodes(nodes)).run()
    assert catalog_before.read_text() == "version: 1\nimages: []\n"


def test_the_sampler_engine_comes_from_the_image_unless_named(tmp_path):
    req = request([b300(0)], tmp_path, role="sampler")
    assert ia.default_engine(req, ia.parse_probe(probe_out(packages={"sglang": "0.5"}))) == "sglang"
    assert ia.default_engine(req, ia.parse_probe(probe_out(packages={"vllm": "0.19", "vllm-ascend": "0.19"}))) \
        == "vllm-ascend"
    assert ia.default_engine(dataclasses.replace(req, engine="SGLang"), None) == "sglang"
    assert ia.default_engine(request([b300(0)], tmp_path), None) == "miles"


def test_role_nodes_take_spares_and_unroled_nodes():
    nodes = [b300(0, roles=("trainer",)), b300(1, roles=("trainer-spare",)), b300(2, roles=("sampler",)), b300(3)]
    assert [n.name for n in ia.role_nodes(nodes, "trainer")] == ["b300-0", "b300-1", "b300-3"]
    assert [n.name for n in ia.role_nodes(nodes, "sampler")] == ["b300-2", "b300-3"]
    with pytest.raises(ValueError, match="--node b300-9 is not a trainer node"):
        ia.AdoptRequest(ref=PINNED, model="m", role="trainer", nodes=tuple(nodes), node="b300-9")


def test_an_image_whose_probe_saw_no_card_architecture_is_refused():
    facts = ia.parse_probe(probe_out(arch_list=[]))
    assert ia.card_arch_problem(facts, b300(0)).startswith("the probe read no card architecture from the image")


def test_a_tag_behind_the_token_flow_is_pinned_from_the_authenticated_answer():
    def urlopen(req, timeout=None):
        if "/token?" in req.full_url:
            return _Resp(200, body=b'{"token": "t0k"}')
        if req.headers.get("Authorization") == "Bearer t0k":
            return _Resp(200, {"Docker-Content-Digest": DIGEST})
        raise urllib.error.HTTPError(req.full_url, 401, "x", {"WWW-Authenticate": (
            'Bearer realm="https://reg.example/token",service="reg.example",scope="repository:site/miles-new:pull"')},
            None)
    assert ia.pin(TAGGED, urlopen=urlopen) == PINNED


def test_the_preflight_pull_reports_a_failed_pull():
    from yeto.launch.preflight import cluster as cp

    nodes = [b300(i) for i in range(2)]
    ops = FakeNodes(nodes, **{"b300-1": Behavior(pull=(1, "Error response from daemon: manifest unknown"))})
    pf = cp.ClusterPreflight(nodes, image=PINNED, ops=ops, log=lambda m: None, clock=lambda: FakeCluster.NOW)
    checks, pulled = pf.pull()
    assert pulled is False and {c.scope for c in checks if c.name == "image_pull" and c.status == cp.FAIL} == {"b300-1"}
    ok_ops = FakeNodes(nodes)
    assert cp.ClusterPreflight(nodes, image=PINNED, ops=ok_ops, log=lambda m: None,
                               clock=lambda: FakeCluster.NOW).pull()[1] is True
