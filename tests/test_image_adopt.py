"""``yeto image adopt`` steps 1-4 (own-cluster-fleet-stability 7.2): manifest and digest,
the probe on one node, the fit to every node, the parallel pre-pull with the image root
check. Each failure stops at its step, names the reason and leaves the catalog alone."""

from __future__ import annotations

import dataclasses
import json
import shlex
import subprocess
import urllib.error
import urllib.parse

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

    def __init__(self, nodes, *, probe=None, probe_rc=0, bare=None, busy=None, **per_node):
        super().__init__(nodes, **per_node)
        self.probe = probe_out() if probe is None else probe
        self.bare = bare  # the probe's output without cards (None: the same as with them)
        self.probe_rc = probe_rc
        self.busy = f"{ia.BUSY_MARK} tool cuda\n" if busy is None else busy

    def run(self, node, script, *, timeout=300.0):
        if ia.BUSY_MARK in script:
            self.scripts.append((node.name, "busy"))
            return self.done(0, self.busy)
        if script.startswith(("podman pull", "sudo -n podman pull")):
            self.scripts.append((node.name, script))
            return self.done(*self.b[node.name].pull)
        if ia.FACT in script:
            cards = "--gpus" in shlex.split(script)
            self.scripts.append((node.name, "probe+cards" if cards else "probe"))
            return self.done(self.probe_rc, self.bare if self.bare is not None and not cards else self.probe)
        return super().run(node, script, timeout=timeout)

    def kinds(self):
        return [(s if s in ("probe", "probe+cards", "busy") else s.split()[0] + " " + s.split()[1]
                 if s.startswith("docker") else "host" if "@@yeto-section" in s else s.split()[0])
                for _, s in self.scripts]


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


def test_the_probe_runs_the_image_without_cards_and_reads_its_facts():
    node = b300(0)
    argv = shlex.split(ia.probe_script(PINNED, node))
    assert argv[:6] == ["docker", "run", "--rm", "--network", "host", "--entrypoint"] and PINNED in argv
    assert "torch._C._cuda_getArchFlags()" in argv[-1]  # the build flags: no card needed
    with_cards = shlex.split(ia.probe_script(PINNED, node, cards=True))
    assert list(node.docker_args) == with_cards[5:5 + len(node.docker_args)]  # the fallback: the node's flags
    from yeto.launch.node_inventory import NodeProfile

    podman = NodeProfile.from_node(b300(0, docker_args=()), container="podman")
    argv = shlex.split(ia.probe_script(PINNED, podman, cards=True))
    assert argv[:3] == ["podman", "run", "--rm"] and ["--device", "nvidia.com/gpu=all"] == argv[5:7]
    with pytest.raises(ia.AdoptError, match="node b300-0 runs no container engine"):
        ia.probe_script(PINNED, b300(0, runtime="rootfs"))
    facts = ia.parse_probe(probe_out() + "noise line\nYETO_IMAGE_FACT bad {not json\n")
    assert (facts.cpu_arch, facts.python, facts.cuda, facts.arch_list) == (
        "x86_64", "3.12.3", "13.0", ("sm_90", "sm_100", "sm_103", "compute_103"))
    assert facts.arch_source is None and ia.parse_probe(probe_out(arch_source="build_flags")).arch_source == \
        "build_flags"
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
    assert "the image is built for sm_90/sm_100; node b300-0 has A100-80GB (sm80) (CUDA binary compatibility: " \
        "a cubin sm_XY runs on compute capability X.Z with Z >= Y" in error.value.reason
    assert "node b300-1 has A100-80GB (sm80)" in error.value.reason
    assert ops.kinds() == ["date", "host", "busy", "probe"]  # the probe node's checks, nothing pulled
    assert catalog_before.read_text() == "version: 1\nimages: []\n"


@pytest.mark.parametrize("archs, card, ok, note", [
    (["sm_90", "compute_90"], "B300", True, ia.PTX_RULE_NOTE),  # PTX of sm90 compiles on sm103
    (["sm_100"], "B300", True, ia.MINOR_RULE_NOTE),              # a 10.0 cubin runs on 10.3
    (["sm_80"], "A10G", True, ia.MINOR_RULE_NOTE),               # an 8.0 cubin runs on 8.6
    (["sm_86"], "A100", False, None),                            # not on a lower minor
    (["sm_100a"], "B300", False, None),                          # an arch-specific cubin: only 10.0
    (["sm_90a"], "H100", True, None),
    (["sm_120"], "B300", False, None),                           # another major
    (["sm_90", "compute_90a"], "B300", False, None),             # arch-specific PTX: no forward JIT
    (["compute_103a"], "B300", True, ia.PTX_RULE_NOTE),
    (["sm_90", "compute_120"], "B300", False, None),             # PTX of a newer card: no backward JIT
    (["sm_100f"], "B300", True, ia.MINOR_RULE_NOTE),             # a family cubin of 10.x
    (["sm_90", "compute_100f"], "B300", True, ia.PTX_RULE_NOTE),  # family PTX within 10.x
    (["sm_90", "compute_100f"], "RTX-PRO-6000", False, None),    # not across majors (sm120)
    (["sm_90", "compute_103f"], "B200", False, None),            # not to a lower minor
    (["sm_80", "sm_90"], "H100", True, None),
    (["sm_90", "sm_100", "sm_103"], "B300", True, None),
    (["gfx942"], "MI355X", False, None),
    (["gfx942", "gfx950"], "MI355X", True, None),
    (["gfx950:sramecc+:xnack-"], "MI355X", True, None),          # a target's feature suffix
    ([], "B300", False, None),                                   # no cards seen in the probe
])
def test_card_arch_rules(archs, card, ok, note):
    facts = ia.parse_probe(probe_out(arch_list=archs))
    family = "rocm" if card.startswith("MI") else "cuda"
    node = b300(0, card=card, device_family=family)
    problem, how = ia.card_arch_fit(facts, node)
    assert (problem is None) is ok, problem
    assert (note in how) if note else how is None
    assert (ia.card_arch_problem(facts, node) is None) is ok


@pytest.mark.parametrize("archs", [["sm_100"], ["sm_90", "compute_90"], ["compute_103"]])
def test_strict_arch_wants_the_exact_cubin(archs):
    facts = ia.parse_probe(probe_out(arch_list=archs))
    problem, _ = ia.card_arch_fit(facts, b300(0), strict=True)
    assert problem.startswith("--strict-arch: the image is built for") and "no cubin of exactly sm103" in problem
    assert ia.card_arch_fit(ia.parse_probe(probe_out(arch_list=["sm_103a"])), b300(0), strict=True) == (None, None)


def test_the_node_profile_card_arch_wins_over_the_card():
    from yeto.launch.node_inventory import NodeProfile

    node = NodeProfile.from_node(b300(0), card_arch="sm100")
    facts = ia.parse_probe(probe_out(arch_list=["sm_100"]))
    assert ia.card_arch_fit(facts, node) == (None, None) and ia.node_card_arch(node) == "sm100"


def test_a_minor_rule_pass_is_noted_in_the_fit_and_the_evidence(tmp_path):
    nodes = [b300(i) for i in range(2)]
    ops = FakeNodes(nodes, probe=probe_out(arch_list=["sm_90", "sm_100"]))
    report = adopter(request(nodes, tmp_path), ops, smoke=passing_smoke()).run()
    fit = next(s for s in report.steps if s.name == "fit")
    assert ia.MINOR_RULE_NOTE in fit.detail and len(fit.data["notes"]) == 2
    (rec,) = si.load(tmp_path / "site-images.yaml")
    assert rec.evidence["notes"] == fit.data["notes"]
    assert rec.evidence["notes"][0] == f"node b300-0 (sm103) runs the sm_100 cubin: {ia.MINOR_RULE_NOTE}"
    strict = request(nodes, tmp_path, strict_arch=True)
    with pytest.raises(ia.AdoptError, match="step fit failed: --strict-arch"):
        adopter(strict, FakeNodes(nodes, probe=probe_out(arch_list=["sm_90", "sm_100"])), smoke=passing_smoke()).run()


def test_a_node_of_no_known_card_architecture_is_refused_at_fit(tmp_path):
    node = b300(0, card="M890", device_family="ppu")
    a = adopter(request([node], tmp_path), FakeNodes([node]))
    with pytest.raises(ia.AdoptError, match="node b300-0 has M890 cards of no known architecture"):
        a._fit(PINNED, ia.parse_probe(probe_out()), "miles")


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
    problems, _, _ = ia.fit_problems(request([a100], tmp_path), ref, facts, "miles")
    assert len(problems) == 1 and "node b300-0: trainer image" in problems[0] and "sm90+" in problems[0]
    lifted = dataclasses.replace(request([a100], tmp_path), attention_backend="unfused")
    problems, warnings, _ = ia.fit_problems(lifted, ref, facts, "miles")
    assert problems == [] and "the unfused attention backend takes another path" in warnings[0]
    flash = dataclasses.replace(request([a100], tmp_path), attention_backend="flash")
    problems, warnings, _ = ia.fit_problems(flash, ref, facts, "miles")  # flash's own floor, not the lift
    assert len(problems) == 1 and "flash attention needs sm90+ cards; on A100-80GB (sm80)" in problems[0]
    assert warnings == []


A10G_FACTS = probe_out(arch_list=["sm_80", "sm_86", "sm_90", "sm_100", "sm_103"],
                       packages={**MILES_FACTS["packages"], "vllm": "0.11.0"})


@pytest.mark.parametrize("card, backend, role, refused", [
    ("A10G", "flash", "trainer", True),      # sm86: flash's backward fails (the rehearsal)
    ("L40S", "flash", "trainer", True),      # sm89: still below sm90
    ("H100", "flash", "trainer", False),     # sm90: the floor itself
    ("B300", "flash", "trainer", False),     # sm103
    ("A10G", "unfused", "trainer", False),   # unfused trains on sm86
    ("A10G", "fused", "trainer", False),     # the floor is flash's only
    ("A10G", "auto", "trainer", False),
    ("A10G", "flash", "sampler", False),     # inference attention: no such floor
])
def test_the_trainer_s_flash_attention_needs_sm90(tmp_path, card, backend, role, refused):
    node = b300(0, card=card, hbm_gib=24.0 if card == "A10G" else 80.0)
    req = request([node], tmp_path, role=role, attention_backend=backend)
    problems, _, _ = ia.fit_problems(req, PINNED, ia.parse_probe(A10G_FACTS), "miles" if role == "trainer" else "vllm")
    if refused:
        assert problems == [f"node b300-0: the trainer's flash attention needs sm90+ cards; on {card} "
                            f"({ia.node_card_arch(node)}) its backward pass fails (\"Unsupported compute "
                            "capability\"): pass --attention-backend unfused (the smoke trains with it; it holds "
                            "the full score matrix, short sequences only)"]
    else:
        assert problems == []


def test_flash_on_sm86_stops_the_adoption_at_fit_on_any_node_of_the_role(tmp_path):
    nodes = [b300(0, card="H100", hbm_gib=80.0), b300(1, card="A10G", hbm_gib=24.0)]
    req = request(nodes, tmp_path, attention_backend="flash")
    with pytest.raises(ia.AdoptError, match=r"step fit failed: node b300-1: the trainer's flash attention needs "
                                            r"sm90\+ cards; on A10G \(sm86\)") as raised:
        adopter(req, FakeNodes(nodes, probe=A10G_FACTS), smoke=passing_smoke()).run()
    assert "node b300-0:" not in str(raised.value)  # the H100 node passes the floor
    assert not (tmp_path / "site-images.yaml").exists()
    ok = request(nodes, tmp_path, attention_backend="unfused")
    adopter(ok, FakeNodes(nodes, probe=A10G_FACTS), smoke=passing_smoke()).run()
    assert [r.ref for r in si.load(tmp_path / "site-images.yaml")] == [PINNED]


@pytest.mark.parametrize("arch, refused", [("sm86", True), ("sm90", False), ("sm100", False), ("gfx950", False),
                                           (None, False)])
def test_the_flash_floor_reads_the_node_s_architecture(arch, refused, monkeypatch):
    node = b300(0)
    monkeypatch.setattr(ia, "node_card_arch", lambda n: arch)
    assert (ia.flash_trainer_problem(node, "flash") is not None) is refused
    assert ia.flash_trainer_problem(node, "unfused") is None and ia.flash_trainer_problem(node, None) is None


def test_a_catalog_sampler_image_is_held_to_its_record(tmp_path):
    from yeto.rl import sampler_pins as pins

    ref = pins.VLLM_SAMPLER_IMAGE_ROCM.removeprefix("docker:")
    facts = ia.parse_probe(probe_out(packages={"vllm": "0.19"}))
    problems, _, _ = ia.fit_problems(request([b300(0)], tmp_path, role="sampler"), ref, facts, "vllm")
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
    assert "node b300-0:" not in error.value.reason
    assert "the image stays pulled on b300-0 (a retry reuses it" in error.value.reason  # the probe pulled it
    assert error.value.holding == ("b300-0",)
    assert "docker pull" not in ops.kinds()  # no node pulled
    assert catalog_before.read_text() == "version: 1\nimages: []\n"


def test_a_failed_pull_names_the_node(tmp_path, catalog_before):
    nodes = [b300(i) for i in range(3)]
    bad = Behavior(pull=(1, "Error response from daemon: no space left on device"))
    ops = FakeNodes(nodes, **{"b300-2": bad})
    with pytest.raises(ia.AdoptError, match=r"step pull failed: node b300-2: image_pull: docker pull failed \(rc 1\)") \
            as error:
        adopter(request(nodes, tmp_path), ops, smoke=passing_smoke()).run()
    assert error.value.holding == ("b300-0", "b300-1")  # the probe node, then the nodes whose pull passed
    assert f"the image stays pulled on b300-0, b300-1 (a retry reuses it; `docker rmi {PINNED}` on b300-0, b300-1 " \
        "frees the room)" in error.value.reason
    assert ops.kinds().count("docker pull") == 3  # every node was pulled in parallel
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

    with pytest.raises(ia.AdoptError, match="step smoke failed: hot reload") as error:
        adopter_ = adopter(request(nodes, tmp_path, role="sampler"), FakeNodes(
            nodes, probe=probe_out(packages={"vllm": "0.19.0"})), smoke=smoke)
        adopter_.run()
    assert error.value.holding == ("b300-0",) and "the image stays pulled on b300-0" in error.value.reason
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


# -- the probe node's checks (host, idle cards) and the probe fallback ----------------------


def test_the_probe_node_is_checked_for_room_before_the_probe_pulls(tmp_path, catalog_before):
    nodes = [b300(i) for i in range(2)]
    low = Behavior(host={"disk": (0, "\n".join([disk_line("/", 80, 200, dev="/dev/sda2", mount="/"),
                                                disk_line("/var/lib/containerd", 40, mount="/")]))})
    ops = FakeNodes(nodes, **{"b300-0": low})
    with pytest.raises(ia.AdoptError, match=r"step probe failed: node b300-0: disk_images: ") as error:
        adopter(request(nodes, tmp_path), ops, smoke=passing_smoke()).run()
    assert error.value.holding == () and "stays pulled" not in error.value.reason
    assert ops.kinds() == ["date", "host"]  # no probe ran


@pytest.mark.parametrize("family, card, out, apps", [
    ("cuda", "B300", "YETO_BUSY tool cuda\nYETO_BUSY apps 4242, python3, 81000 MiB\n", ["4242 python3"]),
    ("cuda", "B300", "YETO_BUSY tool cuda\nYETO_BUSY apps No devices were found\n", []),
    ("npu", "910B3", "YETO_BUSY tool ascend\n"
     "YETO_BUSY apps | 0     910B3               | OK            | 93.1        39                0    / 0   |\n"
     "YETO_BUSY apps | 0                         | 0000:C1:00.0  | 0           0    / 0          3404 / 65536 |\n"
     "YETO_BUSY apps | 0       0                 | 2938420       | python                   | 61307 |\n"
     "YETO_BUSY apps | No running processes found in NPU 1                            |\n", ["2938420 python"]),
    ("rocm", "MI355X", "YETO_BUSY tool rocm\nYETO_BUSY apps gpu,name,pid,mem\nYETO_BUSY apps 0,vllm,777,1024\n",
     ["777 vllm"]),
    ("rocm", "MI355X", "YETO_BUSY tool rocm\nYETO_BUSY apps KFD process information:\n"
     "YETO_BUSY apps PID  PROCESS NAME  GPU(s)  VRAM USED\nYETO_BUSY apps 3131 python3 1 0\n", ["3131 python3"]),
])
def test_busy_parsing_per_family(family, card, out, apps):
    from yeto.launch.node_inventory import profile_of

    node = b300(0, card=card, device_family=family)
    found, containers, readable = ia.parse_busy(out, profile_of(node).family)
    assert (found, containers, readable) == (apps, [], True)
    assert ia.BUSY_MARK in ia.busy_script(node)


def test_the_busy_script_runs_under_bash(tmp_path):
    done = subprocess.run(["bash", "-c", ia.busy_script(b300(0))], capture_output=True, text=True, timeout=60,
                          env={"PATH": "/usr/bin:/bin"})
    assert ia.parse_busy(done.stdout, "cuda")[2] is False  # no nvidia-smi here: unreadable, not busy
    assert f"{ia.BUSY_MARK} tool none" in done.stdout


def test_a_busy_probe_node_is_refused_unless_allowed(tmp_path, catalog_before):
    nodes = [b300(0)]
    busy = "YETO_BUSY tool cuda\nYETO_BUSY apps 4242, python3, 81000 MiB\nYETO_BUSY containers yeto-r1-i0 img\n"
    with pytest.raises(ia.AdoptError, match=r"step probe failed: node b300-0 is in use: process 4242 python3, "
                                            r"container yeto-r1-i0 img; .* --allow-busy"):
        adopter(request(nodes, tmp_path), FakeNodes(nodes, busy=busy), smoke=passing_smoke()).run()
    assert catalog_before.read_text() == "version: 1\nimages: []\n"
    logs = []
    a = adopter(request(nodes, tmp_path, allow_busy=True), FakeNodes(nodes, busy=busy), smoke=passing_smoke())
    a.log = logs.append
    assert a.run().ok
    assert any("WARNING: node b300-0 is in use" in m and "--allow-busy" in m for m in logs)


def test_an_unreadable_card_tool_warns_and_goes_on(tmp_path):
    nodes = [b300(0)]
    logs = []
    a = adopter(request(nodes, tmp_path), FakeNodes(nodes, busy="YETO_BUSY tool none\n"), smoke=passing_smoke())
    a.log = logs.append
    assert a.run().ok and any("no card tool to read its processes" in m for m in logs)


def test_the_probe_falls_back_to_the_cards_when_the_build_flags_are_empty(tmp_path):
    nodes = [b300(0)]
    ops = FakeNodes(nodes, bare=probe_out(arch_list=[]))
    report = adopter(request(nodes, tmp_path), ops, smoke=passing_smoke()).run()
    assert ops.kinds()[:5] == ["date", "host", "busy", "probe", "probe+cards"]
    assert report.facts.arch_list == ("sm_90", "sm_100", "sm_103", "compute_103")
    flags = FakeNodes(nodes)
    adopter(request(nodes, tmp_path), flags, smoke=passing_smoke()).run()
    assert "probe+cards" not in flags.kinds()  # the build flags were enough


def _podman(i, **kw):
    from yeto.launch.node_inventory import NodeProfile

    return NodeProfile.from_node(b300(i, docker_args=(), **kw), container="podman")


def _graphroot(root="/mnt/nvme/podman", free=3000):
    """A podman node's host probe: ``podman info`` names ``root``, which has ``free`` GiB."""
    return Behavior(host={"podman_root": (0, root),
                          "disk": (0, "\n".join([disk_line("/", 80, 200, dev="/dev/sda2", mount="/"),
                                                 disk_line(root, free, mount="/mnt/nvme")]))})


def test_podman_nodes_are_probed_checked_and_pulled_with_podman(tmp_path):
    nodes = [_podman(0, runtime="rootfs"), b300(1)]  # podman on a node of no docker runtime
    ops = FakeNodes(nodes, **{"b300-0": _graphroot()})
    report = adopter(request(nodes, tmp_path), ops, smoke=passing_smoke()).run()
    assert report.ok
    pulls = sorted(s.split(" pull")[0] + "@" + n for n, s in ops.scripts if " pull -q " in s)
    assert pulls == ["docker@b300-1", "podman@b300-0"]  # each node with its own engine
    host = [s for n, s in ops.scripts if n == "b300-0" and "@@yeto-section" in s]
    assert host and all("podman info --format" in s for s in host)
    assert not any("podman" in s for n, s in ops.scripts if n == "b300-1" and "@@yeto-section" in s)
    (rec,) = si.load(tmp_path / "site-images.yaml")
    assert rec.evidence["pulled"] == ["b300-0", "b300-1"]


@pytest.mark.parametrize(("dev", "fix"), [
    ("/dev/sda2", "move it to NVMe (graphroot in /etc/containers/storage.conf)"),
    ("/dev/nvme0n1p2", "/ is already on NVMe (/dev/nvme0n1p2): free space there"),
])
def test_a_podman_graphroot_short_of_room_stops_the_pull(tmp_path, dev, fix):
    """The NVMe or not of the fix comes from the node's probe (its df device), given here."""
    nodes = [_podman(0), _podman(1)]
    low = Behavior(host={"podman_root": (0, "/var/lib/containers/storage"),
                         "disk": (0, "\n".join([disk_line("/", 80, 200, dev=dev, mount="/"),
                                                disk_line("/var/lib/containers/storage", 30, dev=dev, mount="/")]))})
    ops = FakeNodes(nodes, **{"b300-0": _graphroot(), "b300-1": low})
    with pytest.raises(ia.AdoptError) as error:
        adopter(request(nodes, tmp_path, node="b300-0"), ops, smoke=passing_smoke()).run()
    assert error.value.step == "pull"
    assert ("node b300-1: disk_images: /var/lib/containers/storage (podman graphroot) on /: 30 GiB free "
            f"({fix}") in error.value.reason
    assert ("move it to NVMe" in error.value.reason) == (dev == "/dev/sda2")
    assert not any(" pull -q " in s for _, s in ops.scripts)


def test_the_pull_needs_a_container_engine(tmp_path):
    nodes = [b300(i, runtime="rootfs") for i in range(2)]
    a = adopter(request(nodes, tmp_path), FakeNodes(nodes))
    with pytest.raises(ia.AdoptError, match=r"step pull failed: no trainer node runs a container engine "
                                            r"\(docker or podman\): \['b300-0', 'b300-1'\]"):
        a._pull(PINNED)


def test_the_preflight_host_probe_of_a_docker_node_is_unchanged():
    from yeto.launch.preflight import cluster as cp

    node = b300(0)
    assert cp.host_sections(node) == cp.host_sections(node, engine="docker")
    assert "podman" not in str(cp.host_sections(node)) and "podman" in str(cp.host_sections(node, engine="podman"))
    assert [n for n, _ in cp.host_sections(node, engine="podman")][-2:] == ["podman_root", "disk"]


def test_a_fit_failure_names_the_probe_node_that_holds_the_image(tmp_path):
    nodes = [b300(0)]
    ops = FakeNodes(nodes, probe=probe_out(uname_m="aarch64"))
    with pytest.raises(ia.AdoptError, match="step fit failed") as error:
        adopter(request(nodes, tmp_path), ops, smoke=passing_smoke()).run()
    assert error.value.holding == ("b300-0",)


def test_a_tag_next_to_a_digest_is_left_out_of_the_repository_and_the_token_scope():
    assert im.split_tagged_ref(f"reg.example/org/img:v1@{DIGEST}") == ("reg.example", "org/img", DIGEST)
    assert im.split_tagged_ref(f"reg.example:5000/org/img:v1@{DIGEST}") == ("reg.example:5000", "org/img", DIGEST)
    seen = []

    def urlopen(req, timeout=None):  # a registry: the challenge's scope is the repository the URL names
        seen.append(req.full_url)
        if "/token?" in req.full_url:
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(req.full_url).query)
            ok = query.get("scope") == ["repository:org/img:pull"]
            return _Resp(200, body=b'{"token": "t0k"}' if ok else b'{"token": "wrong-scope"}')
        if req.headers.get("Authorization") == "Bearer t0k":
            return _Resp(200, {"Docker-Content-Digest": DIGEST})
        repo = req.full_url.split("/v2/", 1)[1].rsplit("/manifests/", 1)[0]
        raise urllib.error.HTTPError(req.full_url, 401, "x", {"WWW-Authenticate": (
            f'Bearer realm="https://reg.example/token",service="reg.example",scope="repository:{repo}:pull"')}, None)
    assert ia.pin(f"reg.example/org/img:v1@{DIGEST}", urlopen=urlopen) == f"reg.example/org/img@{DIGEST}"
    assert seen[0] == f"https://reg.example/v2/org/img/manifests/{DIGEST}"


@pytest.mark.parametrize("root, why", [
    ("", "`podman info` printed no storage graphroot"),
    ("/srv/containers", "the graphroot /srv/containers that `podman info` printed is not on the node"),
])
def test_an_image_store_whose_disk_cannot_be_confirmed_fails(tmp_path, root, why):
    nodes = [_podman(0)]
    ops = FakeNodes(nodes, **{"b300-0": Behavior(host={"podman_root": (0, root)})})
    with pytest.raises(ia.AdoptError) as error:
        adopter(request(nodes, tmp_path), ops, smoke=passing_smoke()).run()
    assert error.value.step == "probe"
    assert f"disk_images: cannot confirm the disk of the image store: {why}" in error.value.reason
    assert ops.kinds()[-1] == "host"  # stopped before the probe ran the image


def test_a_sudo_pool_engine_reads_its_rootful_store_and_pulls_through_sudo(tmp_path):
    class Raw(FakeNodes):
        def run(self, node, script, *, timeout=300.0):
            self.raw = [*getattr(self, "raw", []), script]
            return super().run(node, script, timeout=timeout)

    nodes = [_podman(0), _podman(1)]
    ops = Raw(nodes, **{n.name: _graphroot() for n in nodes})
    report = adopter(request(nodes, tmp_path, container_sudo=True), ops, smoke=passing_smoke()).run()
    assert report.ok
    sent = [s for s in ops.raw if ia.FACT in s or ia.BUSY_MARK in s]
    assert sent and all(s.startswith("sudo -n podman run ") for s in sent if ia.FACT in s)
    assert all("sudo -n podman ps --filter name=^yeto-" in s for s in sent if ia.BUSY_MARK in s)
    host = [s for _, s in ops.scripts if "@@yeto-section" in s]
    assert host and all("sudo -n podman info --format '{{.Store.GraphRoot}}'" in s for s in host)
    assert all(s.count("podman info") == s.count("sudo -n podman info") == 2 for s in host)  # section and disk
    assert sorted(n for n, s in ops.scripts if s.startswith("sudo -n podman pull -q ")) == ["b300-0", "b300-1"]
    assert shlex.split(ia.probe_script(PINNED, nodes[0], sudo=True))[:4] == ["sudo", "-n", "podman", "run"]
    assert "sudo -n podman ps --filter name=^yeto-" in ia.busy_script(nodes[0], sudo=True)
    assert "sudo" not in ia.probe_script(PINNED, nodes[0]) and "sudo" not in ia.busy_script(nodes[0])


def test_a_docker_node_with_sudo_reads_docker_info_through_sudo_and_without_it_is_unchanged():
    from yeto.launch.preflight import cluster as cp

    node = b300(0)
    plain = dict(cp.host_sections(node))
    sudo = dict(cp.host_sections(node, sudo=True))
    assert plain == dict(cp.host_sections(node, engine="docker", sudo=False)) and "sudo" not in str(plain)
    assert sudo["docker"] == "sudo -n docker info --format '{{json .}}'"
    assert "$(sudo -n docker info -f '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)" in sudo["disk"]
    assert {k: v for k, v in sudo.items() if k not in ("docker", "disk")} == \
        {k: v for k, v in plain.items() if k not in ("docker", "disk")}


def test_the_docker_host_probe_is_byte_for_byte_main_s():
    """The preflight's host probe of a docker node (no engine, no sudo given): the digest
    of main's script (recorded from main d9fb37915), so the customer path cannot drift."""
    import hashlib

    from yeto.launch.preflight import cluster as cp

    for node, want in ((b300(0), "3e692f2760ba7ad24840de395a33dd4b106b8abcc3fe645004aad4a2b52fcd8e"),
                       (b300(1, mounts={"/mnt/nvme/x": "/x"}),
                        "958b24e9b3fed68eeb27d8221ae0357653a5009b8e016a3a9ba270211cfce5de")):
        assert hashlib.sha256(cp.probe_script(cp.host_sections(node)).encode()).hexdigest() == want


@pytest.mark.parametrize("sudo", [False, True])
def test_the_removal_hint_names_each_holding_node_s_engine(tmp_path, sudo):
    nodes = [_podman(0), b300(1), _podman(2)]
    ops = FakeNodes(nodes, **{"b300-0": _graphroot(), "b300-2": _graphroot()})
    a = adopter(request(nodes, tmp_path, container_sudo=sudo), ops, smoke=passing_smoke())
    a.holding = ["b300-0", "b300-1", "b300-2"]
    a.report.pinned = PINNED
    reason = a._held(ia.AdoptError("smoke", "failed")).reason
    pre = "sudo -n " if sudo else ""
    assert (f"(a retry reuses it; `{pre}docker rmi {PINNED}` on b300-1; `{pre}podman rmi {PINNED}` on b300-0, "
            "b300-2 frees the room)") in reason
    assert ("sudo -n" in reason) is sudo


def test_a_rootfs_node_with_docker_is_pulled_and_a_rootfs_node_without_an_engine_is_listed(tmp_path):
    """The adoption picks nodes by their profile's container, not by runtime: a
    ``runtime: rootfs`` node whose inventory says ``container: docker`` is pulled and its
    docker image root checked; one of no container engine is named as not pulled."""
    from yeto.launch.node_inventory import NodeProfile

    docker_rootfs = NodeProfile.from_node(b300(1, runtime="rootfs"), container="docker")
    nodes = [b300(0), docker_rootfs, b300(2, runtime="rootfs")]
    ops = FakeNodes(nodes)
    report = adopter(request(nodes, tmp_path), ops, smoke=passing_smoke()).run()
    pull = next(s for s in report.steps if s.name == "pull")
    assert sorted(n for n, s in ops.scripts if s.startswith("docker pull -q ")) == ["b300-0", "b300-1"]
    assert "not pulled (no container engine): ['b300-2']" in pull.detail
    from yeto.launch.preflight import cluster as cp

    pf = cp.ClusterPreflight([docker_rootfs], image=PINNED, ops=FakeNodes([docker_rootfs]), log=lambda m: None,
                             clock=lambda: FakeCluster.NOW, by_engine=True)
    roots = [c for c in pf.hosts() if c.name == "disk_images"]
    assert roots and roots[0].scope == "b300-1"  # its docker image root is checked
    plain = cp.ClusterPreflight([docker_rootfs], image=PINNED, ops=FakeNodes([docker_rootfs]), log=lambda m: None,
                                clock=lambda: FakeCluster.NOW)
    assert not [c for c in plain.hosts() if c.name == "disk_images"]  # the preflight's own rule: runtime docker


def test_a_rootfs_probe_node_with_docker_has_its_image_root_checked_before_the_probe(tmp_path):
    from yeto.launch.node_inventory import NodeProfile

    probe = NodeProfile.from_node(b300(0, runtime="rootfs"), container="docker")
    low = Behavior(host={"disk": (0, "\n".join([disk_line("/", 80, 200, dev="/dev/sda2", mount="/"),
                                                disk_line("/var/lib/containerd", 40, mount="/")]))})
    ops = FakeNodes([probe, b300(1)], **{"b300-0": low})
    with pytest.raises(ia.AdoptError, match=r"step probe failed: node b300-0: disk_images: /var/lib/containerd"):
        adopter(request([probe, b300(1)], tmp_path), ops, smoke=passing_smoke()).run()
    assert ops.kinds() == ["date", "host"]  # no probe ran
