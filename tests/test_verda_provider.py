"""fix-verda-provider: provider (1.x), mis-delete guards (2.x), teardown (3.x).

No network, no cloud. Verda responses come from real captures
(tests/fixtures/verda_rca/, 2026-09-29) or hand-written fixtures; sky's
real Verda provisioner is exercised in a subprocess when a SkyPilot 0.13
Python is available (tests/verda_sky_scenarios.py).
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import types
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from yeto.shape import providers
from yeto.shape.providers import (
    VERDA_LOCATIONS_FALLBACK,
    VerdaCapacityExhausted,
    launch_with_verda_candidates,
    verda_any_of,
    verda_candidates,
    verda_credentials,
    verda_sky_catalog_rows,
    write_verda_sky_catalog,
)

REPO = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures" / "verda_rca"


def _load(name):
    return json.loads((FIX / name).read_text())


def _avail(name="availability_ondemand.json"):
    return {e["location_code"]: e["availabilities"] for e in _load(name)}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("VERDA_CLIENT_ID", raising=False)
    monkeypatch.delenv("VERDA_CLIENT_SECRET", raising=False)
    (tmp_path / ".verda").mkdir()
    return tmp_path


# --- 1.1 credentials, responses, tables ---------------------------------------


def test_credentials_from_env_json_and_ini(home, monkeypatch):
    assert verda_credentials() is None
    (home / ".verda" / "credentials").write_text(
        "[default]\nverda_client_id = ini-id\nverda_client_secret = ini-secret\n"
    )
    assert verda_credentials() == ("ini-id", "ini-secret")
    (home / ".verda" / "config.json").write_text('{"client_id": "j-id", "client_secret": "j-s"}')
    assert verda_credentials() == ("j-id", "j-s")  # JSON wins over INI
    monkeypatch.setenv("VERDA_CLIENT_ID", "e-id")
    monkeypatch.setenv("VERDA_CLIENT_SECRET", "e-s")
    assert verda_credentials() == ("e-id", "e-s")  # env wins over both


def test_credentials_ini_any_profile_and_broken_json_falls_through(home):
    (home / ".verda" / "config.json").write_text("{not json")
    (home / ".verda" / "credentials").write_text(
        "[work]\nclient_id = w-id\nclient_secret = w-s\n"
    )
    assert verda_credentials() == ("w-id", "w-s")


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _urlopen_returning(body: bytes):
    return lambda req, timeout=30: _Resp(body)


@pytest.mark.parametrize(
    "body,expected",
    [
        ((FIX / "instance_types.json").read_bytes(), "list"),
        (b"3f2a9c1e-0000-4000-8000-000000000001", "3f2a9c1e-0000-4000-8000-000000000001"),
        (b"", None),
        (b"   \n", None),
    ],
)
def test_request_handles_json_text_and_empty(monkeypatch, body, expected):
    monkeypatch.setattr("urllib.request.urlopen", _urlopen_returning(body))
    got = providers._verda_request("GET", "/x")
    if expected == "list":
        assert isinstance(got, list) and len(got) == 70
    else:
        assert got == expected


def test_http_error_keeps_body_and_hides_credentials(home, monkeypatch):
    (home / ".verda" / "config.json").write_text('{"client_id": "cid-123", "client_secret": "TOPSECRET"}')

    def boom(req, timeout=30):
        body = b'{"code":"service_unavailable","message":"no capacity; echo TOPSECRET Bearer abc.def"}'
        raise urllib.error.HTTPError(req.full_url, 503, "Service Unavailable", {}, io.BytesIO(body))

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(RuntimeError) as ei:
        providers._verda_request("POST", "/instances", token="tok", body={"x": 1})
    msg = str(ei.value)
    assert "HTTP 503" in msg and "no capacity" in msg
    assert "TOPSECRET" not in msg and "abc.def" not in msg
    assert getattr(ei.value, "status", None) == 503


def test_locations_and_models():
    assert "ICL-01" not in VERDA_LOCATIONS_FALLBACK
    sig = providers.VerdaSignals(cache=None)
    sig._fetch_types = lambda: _load("instance_types.json")
    sig._locations = lambda: list(VERDA_LOCATIONS_FALLBACK)
    rows = sig.offerings(None, None, None)
    gpus = {(o.gpu, o.instance_type) for o in rows}
    assert ("RTX-PRO-6000", "1RTXPRO6000.30V") in gpus
    assert ("A100", "1A100.40S.22V") in gpus
    assert ("L40S", "1L40S.20V") in gpus
    assert not any(o.instance_type.endswith(".CC") for o in rows)


# --- 1.2 local sky catalog ----------------------------------------------------


def test_catalog_rows_cover_every_type_and_location():
    rows = verda_sky_catalog_rows(_load("instance_types.json"), ["FIN-01", "FIN-02", "FIN-03"])
    by = {(r["InstanceType"], r["Region"]): r for r in rows}
    assert len({r["InstanceType"] for r in rows}) == 70 - 15  # the 15 ".CC" types are skipped
    assert by[("1L40S.20V", "FIN-02")]["AcceleratorName"] == "L40S"
    assert by[("1RTXPRO6000.30V", "FIN-03")]["AcceleratorName"] == "RTX-PRO-6000"
    assert by[("1A100.40S.22V", "FIN-01")]["AcceleratorName"] == "A100"
    assert by[("1A100.22V", "FIN-01")]["AcceleratorName"] == "A100-80GB"
    assert by[("1L40S.20V", "FIN-01")]["Price"] == 1.543
    assert by[("CPU.4V.16G", "FIN-02")]["AcceleratorName"] == ""


# A test process that has imported torch makes a child's numpy import die
# with SIGINT unless its thread pools are pinned (seen with this machine's
# SkyPilot venv); pin them for every sky subprocess.
SKY_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def _sky_python() -> str | None:
    cand = os.environ.get("YETO_SKY_PYTHON") or "/home/michael/work/gpu-head/venv/bin/python"
    if cand and os.path.exists(cand):
        r = subprocess.run([cand, "-c", "import sky"], capture_output=True, text=True,
                           env=dict(os.environ, **SKY_ENV))
        if r.returncode == 0:
            return cand
        print("sky python unusable:", r.stderr[-800:])
    return None


def test_generated_catalog_is_what_sky_reads(tmp_path):
    path = write_verda_sky_catalog(
        _load("instance_types.json"), ["FIN-01", "FIN-02", "FIN-03"], runtime_dir=str(tmp_path)
    )
    assert path == str(tmp_path / ".sky" / "catalogs" / "v8" / "verda" / "vms.csv")
    py = _sky_python()
    if py is None:
        pytest.skip("no SkyPilot 0.13 Python (set YETO_SKY_PYTHON)")
    code = (
        "import json\n"
        "from sky.catalog import verda_catalog as c\n"
        "accs = c.list_accelerators(True, None, None, None)\n"
        "print(json.dumps({'l40s': c.instance_type_exists('1L40S.20V'),"
        " 'rtxpro': c.instance_type_exists('1RTXPRO6000.30V'),"
        " 'accs': sorted(accs), 'l40s_fin01': c.get_hourly_cost('1L40S.20V', region='FIN-01'),"
        " 'regions': sorted(set(c._df['Region']))}))\n"
    )
    env = dict(os.environ, SKY_RUNTIME_DIR=str(tmp_path), HOME=str(tmp_path), **SKY_ENV)
    out = subprocess.run([py, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["l40s"] and got["rtxpro"]
    assert {"L40S", "RTX-PRO-6000", "A100", "A100-80GB", "H100"} <= set(got["accs"])
    assert got["l40s_fin01"] == pytest.approx(1.543)
    assert got["regions"] == ["FIN-01", "FIN-02", "FIN-03"]


def test_head_bootstrap_writes_the_catalog_for_verda_fleets(monkeypatch):
    from yeto.cli import _make_head_task
    from yeto.shape.providers import VERDA_HEAD_CATALOG_STEP

    task = _head_task(monkeypatch, "verda:1xl40s@FIN-01")
    assert VERDA_HEAD_CATALOG_STEP in task.setup
    task = _head_task(monkeypatch, "aws:1xa100@us-east-2")
    assert VERDA_HEAD_CATALOG_STEP not in task.setup
    assert _make_head_task  # imported for the helper below


def _head_task(monkeypatch, gpu):
    import yeto.cli as cli
    import yeto.launcher as launcher

    fake_sky = types.ModuleType("sky")

    class Task:
        def __init__(self, **kw):
            self.__dict__.update(kw)

        def set_resources(self, r):
            self.resources = r

    fake_sky.Task = Task
    fake_sky.Resources = lambda **kw: SimpleNamespace(kwargs=kw)
    monkeypatch.setitem(sys.modules, "sky", fake_sky)
    monkeypatch.setattr(launcher, "head_cloud_credentials", lambda clouds: ({}, {}))
    args = SimpleNamespace(
        gpu=gpu, syncer_region="nebius/eu-north1", syncer_memory="32", loss_function="ce",
        wandb=False, controller="head",
    )
    return cli._make_head_task(args)


# --- 1.3 candidates -------------------------------------------------------------


TYPES = _load("instance_types.json")


def test_candidates_follow_live_stock_and_price():
    avail = {"FIN-01": ["1L40S.20V", "1A100.22V"], "FIN-02": ["1L40S.20V"], "FIN-03": []}
    c = verda_candidates("L40S", 1, TYPES, avail)
    assert [(x["instance_type"], x["region"]) for x in c] == [("1L40S.20V", "FIN-01"), ("1L40S.20V", "FIN-02")]
    assert verda_candidates("L40S", 1, TYPES, avail, regions=["FIN-02"])[0]["region"] == "FIN-02"
    assert verda_candidates("H100", 1, TYPES, avail) == []
    # Real capture: A100 80GB on demand is in stock in FIN-01 only.
    real = verda_candidates("A100-80GB", 1, TYPES, _avail())
    assert [(x["instance_type"], x["region"]) for x in real] == [("1A100.22V", "FIN-01")]


def test_candidates_demote_capacity_failures_and_cap():
    avail = {loc: ["1L40S.20V"] for loc in ("FIN-01", "FIN-02", "FIN-03")}
    c = verda_candidates("L40S", 1, TYPES, avail, demoted={("1L40S.20V", "FIN-01"): 1}, limit=2)
    assert [x["region"] for x in c] == ["FIN-02", "FIN-03"]
    assert verda_any_of(c, "L40S", 1) == [
        {"infra": "verda/FIN-02", "instance_type": "1L40S.20V", "accelerators": "L40S:1"},
        {"infra": "verda/FIN-03", "instance_type": "1L40S.20V", "accelerators": "L40S:1"},
    ]


def test_launch_retries_with_fresh_stock_and_capped_backoff():
    fetches, launches, sleeps = [], [], []
    stock = [{"FIN-01": ["1L40S.20V"], "FIN-02": ["1L40S.20V"]}, {"FIN-02": ["1L40S.20V"]}]

    def fetch():
        fetches.append(1)
        return stock[min(len(fetches) - 1, 1)]

    def build(avail, demoted):
        return verda_candidates("L40S", 1, TYPES, avail, demoted=demoted)

    def launch(cands):
        launches.append([c["region"] for c in cands])
        if len(launches) == 1:
            raise RuntimeError("ResourcesUnavailableError: 503 no capacity")
        return "job-7"

    got = launch_with_verda_candidates(launch, fetch, build, base_delay=30, max_delay=45, sleep=sleeps.append)
    assert got == "job-7"
    assert launches == [["FIN-01", "FIN-02"], ["FIN-02"]]  # refreshed stock, not the cached one
    assert len(fetches) == 2 and sleeps == [30]


def test_launch_gives_up_with_a_per_candidate_report():
    sleeps = []

    def launch(cands):
        raise RuntimeError("Resources are currently unavailable on Verda (503)")

    build = lambda avail, demoted: verda_candidates("L40S", 1, TYPES, avail, demoted=demoted)  # noqa: E731
    fetch = lambda: {"FIN-01": ["1L40S.20V"], "FIN-03": ["1L40S.20V"]}  # noqa: E731
    with pytest.raises(VerdaCapacityExhausted) as ei:
        launch_with_verda_candidates(launch, fetch, build, max_attempts=4, base_delay=30, max_delay=60,
                                     sleep=sleeps.append)
    assert sleeps == [30, 60, 60]  # doubling, capped
    report = "\n".join(ei.value.report)
    assert "1L40S.20V@FIN-01: attempt 4" in report and "1L40S.20V@FIN-03: attempt 4" in report


def test_launch_non_capacity_error_is_not_retried():
    def launch(cands):
        raise ValueError("bad task spec")

    with pytest.raises(ValueError):
        launch_with_verda_candidates(
            launch, lambda: {"FIN-01": ["1L40S.20V"]},
            lambda a, d: verda_candidates("L40S", 1, TYPES, a, demoted=d), sleep=lambda s: None,
        )


def test_launch_reports_when_nothing_is_in_stock():
    with pytest.raises(VerdaCapacityExhausted, match="no in-stock candidate"):
        launch_with_verda_candidates(
            lambda c: "never", lambda: {"FIN-01": []},
            lambda a, d: verda_candidates("L40S", 1, TYPES, a, demoted=d),
            max_attempts=2, sleep=lambda s: None,
        )


# --- 2.1 lower-case cluster names -----------------------------------------------


def test_cluster_names_are_lower_case():
    from yeto.gpu_spec import parse_gpu_spec
    from yeto.launcher import learner_cluster_names, sky_cluster_name

    names = learner_cluster_names("Run1", parse_gpu_spec("verda:1xl40s@FIN-03,aws:8xa100@us-east-2"))
    assert names == ["run1-l0-fin-03", "run1-l1-us-east-2"]
    assert all(n == n.lower() for n in names)
    assert sky_cluster_name("R-Head") == "r-head"


def test_down_uses_recorded_names_verbatim(tmp_path, monkeypatch):
    import yeto.cli as cli
    import yeto.runs as runs

    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    ns = cli.parse_args(["--gpu", "verda:1xl40s@FIN-03", "--model", "gemma4", "--model-revision", "a" * 40,
                         "--data", "org/ds", "--data-revision", "b" * 40, "--controller", "local",
                         "--cluster-prefix", "old"])
    runs.create_run("old", vars(ns))
    # A run from before the lower-casing: its names are what sky knows.
    runs.update_run("old", pid=None, clusters=["old-syncer", "old-l0-FIN-03"])
    downed = []
    monkeypatch.setattr(cli, "_down_and_verify", lambda c: downed.append(c) or True)
    assert cli.main(["down", "old"]) == 0
    assert sorted(downed) == ["old-l0-FIN-03", "old-syncer"]


# --- 2.2 the sky Verda patch (fake module; real sky in a subprocess below) ---------


class _CS:
    INIT, UP, STOPPED = "INIT", "UP", "STOPPED"


def _fake_module(instances, *, create_fails=False, list_empty=False):
    """Stand-in for sky.provision.verda.instance with an in-memory Verda."""
    from tests.verda_sky_scenarios import FakeInstance, FakeVerda

    fake = FakeVerda(instances, create_fails=create_fails, list_empty=list_empty)
    m = types.ModuleType("fake_verda_instance")
    m.verda = fake
    m.InstanceStatus = SimpleNamespace(RUNNING="running", PROVISIONING="provisioning", ORDERED="ordered",
                                       OFFLINE="offline", ERROR="error")
    m.status_lib = SimpleNamespace(ClusterStatus=_CS)
    m.logger = SimpleNamespace(info=lambda *a: None, warning=lambda *a: None)
    m.time = SimpleNamespace(sleep=lambda s: None)
    m.MAX_POLLS_FOR_UP_OR_TERMINATE = 3
    m.POLL_INTERVAL = 0

    def _filter_instances(name, status_filters=None):  # sky's substring matcher
        return {i.instance_id: i for i in fake.instances_get() if name in i.hostname
                and (status_filters is None or i.status in status_filters)}

    def run_instances(region, cluster_name, name, config):
        # sky's shape: look up 'ACTIVE' nodes, create the rest, fail on 503.
        have = m._filter_instances(name, ["ACTIVE"])
        for _ in range(config.count - len(have)):
            fake.instance_create({"hostname": f"{name}-head" if not have else f"{name}-worker"})
            have = m._filter_instances(name, ["ACTIVE"])
        if len(m._filter_instances(name, ["running"])) != config.count:
            raise RuntimeError("Timed out waiting")
        return "record"

    m._filter_instances = _filter_instances
    m.run_instances = run_instances
    m.terminate_instances = lambda *a, **k: None
    m.query_instances = lambda *a, **k: {}
    m.get_cluster_info = lambda *a, **k: None
    return m, fake


@pytest.fixture
def ids_file(tmp_path, monkeypatch):
    monkeypatch.setenv("YETO_VERDA_IDS_PATH", str(tmp_path / "ids.json"))


def _patched(instances, **kw):
    from yeto.sky_patches import verda as vp

    m, fake = _fake_module(instances, **kw)
    ok, why = vp.apply(m, version="0.13.0")
    assert ok, why
    return m, fake


def test_patch_sibling_prefix_is_never_touched(ids_file):
    m, fake = _patched([
        {"id": "a", "status": "running", "hostname": "vfix-x-head"},
        {"id": "b", "status": "running", "hostname": "vfix-xb-head"},
        {"id": "c", "status": "running", "hostname": "vfix-x-worker-1"},
    ])
    assert sorted(m.query_instances("VFIX-X", "vfix-x", {}, True)) == ["a", "c"]
    m.terminate_instances("vfix-x", {})
    assert sorted(fake.deleted) == ["a", "c"]
    assert fake.instances["b"]["status"] == "running"


def test_patch_failed_create_keeps_the_running_node(ids_file):
    m, fake = _patched([{"id": "k1", "status": "running", "hostname": "vfix-k-head"}], create_fails=True)
    with pytest.raises(RuntimeError):
        m.run_instances("FIN-01", "vfix-k", "vfix-k", SimpleNamespace(count=2))
    m.terminate_instances("vfix-k", {})  # the provisioner's failure teardown
    assert fake.deleted == [] and fake.instances["k1"]["status"] == "running"
    # A later, deliberate down still deletes it.
    m.terminate_instances("vfix-k", {})
    assert fake.deleted == ["k1"]


def test_patch_failed_launch_deletes_only_what_it_created(ids_file):
    m, fake = _patched([{"id": "k1", "status": "running", "hostname": "vfix-n-head"}])
    orig_create = fake.instance_create

    def create_then_fail(cfg):
        orig_create(cfg)
        fake.instances[fake.created[-1]]["status"] = "provisioning"  # never comes up

    fake.instance_create = create_then_fail
    with pytest.raises(RuntimeError):
        m.run_instances("FIN-01", "vfix-n", "vfix-n", SimpleNamespace(count=2))
    m.terminate_instances("vfix-n", {})
    assert fake.deleted == ["new-1"] and fake.instances["k1"]["status"] == "running"


def test_patch_running_node_is_reused_not_recreated(ids_file):
    m, fake = _patched([{"id": "r1", "status": "running", "hostname": "vfix-r-head"}], create_fails=True)
    assert m.run_instances("FIN-01", "vfix-r", "vfix-r", SimpleNamespace(count=1)) == "record"
    assert fake.created == [] and fake.deleted == []


def test_patch_status_map_and_unknown_is_init(ids_file):
    m, _ = _patched([
        {"id": "u1", "status": "ordered", "hostname": "vfix-u-head"},
        {"id": "u2", "status": "brand_new_state", "hostname": "vfix-u-worker"},
        {"id": "u3", "status": "deleted", "hostname": "vfix-u-worker-2"},
        {"id": "u4", "status": "offline", "hostname": "vfix-u-worker-3"},
    ])
    assert m.query_instances("x", "vfix-u", {}, True) == {"u1": ("INIT", None), "u2": ("INIT", None),
                                                          "u4": ("STOPPED", None)}
    assert m.query_instances("x", "vfix-u", {}, False)["u3"] == (None, None)


def test_patch_empty_listing_rechecks_recorded_ids(ids_file):
    from yeto.sky_patches import verda as vp

    vp.remember_ids("vfix-e", ["e1", "gone-id"])
    m, _ = _patched([{"id": "e1", "status": "running", "hostname": "vfix-e-head"}], list_empty=True)
    assert m.query_instances("x", "vfix-e", {}, True) == {"e1": ("UP", None)}


def test_patch_wait_accepts_more_running_than_requested(ids_file):
    m, fake = _patched([
        {"id": "h", "status": "running", "hostname": "vfix-w-head"},
        {"id": "s", "status": "running", "hostname": "vfix-w-worker"},  # straggler
    ])
    assert m.run_instances("FIN-01", "vfix-w", "vfix-w", SimpleNamespace(count=1)) == "record"
    assert fake.created == []


def test_patch_version_guard_leaves_sky_alone():
    from yeto.sky_patches import verda as vp

    m, _ = _fake_module([])
    before = (m.run_instances, m.query_instances, m.terminate_instances, m._filter_instances)
    ok, why = vp.apply(m, version="0.14.0")
    assert not ok and "0.14.0" in why
    assert (m.run_instances, m.query_instances, m.terminate_instances, m._filter_instances) == before
    assert not vp.is_patched(m)


def test_patch_against_real_sky_provisioner(tmp_path):
    py = _sky_python()
    if py is None:
        pytest.skip("no SkyPilot 0.13 Python (set YETO_SKY_PYTHON)")
    env = dict(os.environ, HOME=str(tmp_path), SKY_RUNTIME_DIR=str(tmp_path), **SKY_ENV)
    out = subprocess.run([py, str(Path(__file__).parent / "verda_sky_scenarios.py"), str(REPO)],
                         capture_output=True, text=True, env=env, timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    r = json.loads(out.stdout.strip().splitlines()[-1])
    assert r["sky_version"] == "0.13.0" and r["patched"] and r["package_reexports_patched"]
    assert r["guard_rejects_other_version"]
    assert r["sibling_prefix"] == {"query": ["a"], "deleted": ["a"]}
    assert r["relaunch_running"] == {"result": "ok", "created": [], "deleted": [], "r1": "running"}
    assert r["create_fails_keeps_running"]["deleted"] == [] and r["create_fails_keeps_running"]["k1"] == "running"
    assert set(r["unknown_status"].values()) == {"ClusterStatus.INIT"}
    assert r["empty_recheck"] == {"e1": "ClusterStatus.UP"}
    assert r["dispatcher"] == {"d1": "ClusterStatus.UP"}
    assert r["fresh_launch"]["created"] == ["new-1"] and r["fresh_launch"]["deleted"] == []
    rn = r["real_naming"]
    assert rn["on_cloud"].startswith("vfix-l0-fin-01-") and rn["on_cloud"] != "vfix-l0-fin-01"
    assert rn["hostnames"] == [rn["on_cloud"] + "-head"]  # sky's real node name
    assert rn["display_name_matches"] == [] and rn["on_cloud_matches"] == ["new-1"]
    assert rn["verify_after_down"] == [True, []] and rn["live_after_down"] == []
    assert rn["verify_without_ids"][0] is False


# --- 2.3 local + head activation ----------------------------------------------------


def test_head_bootstrap_pins_sky_and_writes_pth(monkeypatch):
    from yeto.cli import HEAD_SKYPILOT_VERSION
    from yeto.sky_patches import verda as vp

    task = _head_task(monkeypatch, "verda:1xl40s@FIN-01")
    assert f'skypilot[aws,gcp,runpod,nebius,verda]=={HEAD_SKYPILOT_VERSION}"' in task.setup
    assert HEAD_SKYPILOT_VERSION in vp.VERIFIED_SKY_VERSIONS
    assert "from yeto.sky_patches import PTH_NAME, pth_line" in task.setup
    # The .pth step runs after pip (sky present) and before the ready marker.
    assert task.setup.index("skypilot[") < task.setup.index("pth_line") < task.setup.index("touch ~/.yeto_head_ready")


def test_pth_line_installs_the_hook_in_a_fresh_interpreter(tmp_path):
    from yeto.sky_patches import PTH_NAME, pth_line

    site_dir = tmp_path / "site"
    site_dir.mkdir()
    (site_dir / PTH_NAME).write_text(pth_line(str(REPO)))
    code = (
        f"import site; site.addsitedir({str(site_dir)!r})\n"
        "import sys\n"
        "lazy = any(type(f).__name__ == '_YetoLazy' for f in sys.meta_path)\n"
        "print(lazy, 'yeto' in sys.modules)\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "import yeto.sky_patches as p\n"
        "print(p.status())\n"
    )
    out = subprocess.run([sys.executable, "-S", "-c", code], capture_output=True, text=True,
                         cwd=str(tmp_path), env={"PATH": os.environ.get("PATH", "")})
    assert out.returncode == 0, out.stderr
    # Lazy: the hook is armed but yeto is not imported by unrelated processes.
    assert out.stdout.splitlines()[0] == "True False" and out.stderr == ""
    from yeto.sky_patches import pth_repo

    assert pth_repo(str(site_dir / PTH_NAME)) == str(REPO)
    # A hook pointing at a missing worktree stays silent (no startup noise).
    (site_dir / PTH_NAME).write_text(pth_line(str(tmp_path / "gone")))
    out = subprocess.run([sys.executable, "-S", "-c", f"import site; site.addsitedir({str(site_dir)!r})"],
                         capture_output=True, text=True, env={"PATH": os.environ.get("PATH", "")})
    assert out.returncode == 0 and out.stderr == ""


def test_install_patches_on_import_of_the_target(monkeypatch, ids_file):
    import yeto.sky_patches as patches
    from yeto.sky_patches import verda as vp

    monkeypatch.setattr(patches, "_STATUS", {})
    m, _ = _fake_module([])
    monkeypatch.setitem(sys.modules, "sky", SimpleNamespace(__version__="0.13.0"))
    monkeypatch.setitem(sys.modules, vp.TARGET_MODULE, m)
    assert patches.install()[vp.TARGET_MODULE] == "applied"
    assert vp.is_patched(m)


# --- 2.4 recovery checks Verda by id ---------------------------------------------------


class FakeApi:
    def __init__(self, instances=None, volumes=None, trash=None):
        self.instances = {i["id"]: dict(i) for i in (instances or [])}
        self.volumes = {v["id"]: dict(v) for v in (volumes or [])}
        self.trash = {v["id"]: dict(v) for v in (trash or [])}
        self.deleted, self.purged = [], []

    def list_instances(self):
        return list(self.instances.values())

    def get_instance(self, iid):
        return self.instances.get(iid)

    def delete_instance(self, iid, volume_ids=None):
        self.deleted.append(iid)
        if self.instances[iid].get("sticky"):
            return  # sky/verda said deleted, the node keeps running
        self.instances[iid]["status"] = "deleted"

    def list_volumes(self):
        return list(self.volumes.values())

    def list_trash(self):
        return list(self.trash.values())

    def delete_volume_permanently(self, vid):
        self.purged.append(vid)
        self.trash[vid]["is_permanently_deleted"] = True


# sky names Verda nodes `<cluster_name_on_cloud>-head`, where the name on
# cloud is make_cluster_name_on_cloud(display, 120) = display.lower() + '-'
# + user hash (8 hex). The real function is exercised in
# verda_sky_scenarios.py; here the same shape with a fixed hash.
USER_HASH = "2ea485ea"
ON_CLOUD = f"r-l0-fin-01-{USER_HASH}"


@pytest.mark.parametrize(
    "state,expect",
    [("running", "alive"), ("provisioning", "alive"), ("deleted", "gone"), (None, "gone")],
)
def test_guard_by_instance_id(state, expect):
    from yeto.verda_ops import VerdaInstanceGuard

    api = FakeApi([{"id": "i1", "status": "running", "hostname": f"{ON_CLOUD}-head"}])
    guard = VerdaInstanceGuard(api, ["r-l0-fin-01"], resolve_on_cloud=lambda c: None)
    assert guard.record("r-l0-fin-01") == []  # no name on cloud -> nothing guessed
    assert guard.check("r-l0-fin-01")[0] == "unknown"  # and no blind relaunch
    assert guard.record("r-l0-fin-01", "r-l0-fin-01") == []  # display name never matches
    assert guard.record("r-l0-fin-01", ON_CLOUD) == ["i1"]
    if state is None:
        del api.instances["i1"]
    else:
        api.instances["i1"]["status"] = state
    kind, detail = guard.check("r-l0-fin-01")
    assert kind == expect
    if expect == "gone":
        assert detail == "r-l0-fin-01-r1" and guard.check("other") is None
        assert "r-l0-fin-01-r1" in guard.clusters
    else:
        assert "i1" in detail


def test_next_cluster_name():
    from yeto.verda_ops import next_cluster_name

    assert next_cluster_name("a-l0-fin-01") == "a-l0-fin-01-r1"
    assert next_cluster_name("a-l0-fin-01-r1") == "a-l0-fin-01-r2"


def _controller(ops, guard=None, no_recover=(), renames=None):
    from tests.test_controller import SYNCER, ImmediateThread, RUNNING
    from yeto.launcher import FleetController

    ops.status_seq.setdefault(SYNCER, [RUNNING])
    return FleetController(
        learners={"l0": ("task-l0", 1), "l1": ("task-l1", 1)},
        syncer=(SYNCER, "task-syncer", 1), sky_ops=ops, poll_interval=30, recover_timeout=100,
        thread_cls=ImmediateThread, instance_guard=guard, no_recover=no_recover,
        on_rename=(lambda o, n: renames.append((o, n))) if renames is not None else None,
    )


def test_controller_does_not_relaunch_a_node_verda_still_runs():
    from tests.test_controller import FakeOps, RUNNING, SUCCEEDED

    ops = FakeOps()
    ops.status_seq["l0"] = [RUNNING]
    ops.up["l0"] = False  # sky lost it...
    ops.status_seq["l1"] = [SUCCEEDED]
    guard = SimpleNamespace(check=lambda n: ("alive", "Verda still has instance(s) {'i1': 'running'}")
                            if n == "l0" else None)
    ctl = _controller(ops, guard)
    codes = ctl.run()
    assert ops.relaunch_calls == []  # ...but Verda says it runs: no relaunch
    assert codes["l0"].startswith("ABANDONED") and ctl.learners["l0"]["state"] == "abandoned"


def test_controller_relaunches_a_gone_node_under_a_new_name():
    from tests.test_controller import FakeOps, RUNNING, SUCCEEDED

    ops = FakeOps()
    ops.status_seq["l0"] = [RUNNING]
    ops.up["l0"] = False
    ops.status_seq["l1"] = [SUCCEEDED]
    ops.relaunch_results["l0-r1"] = [55]
    ops.after_relaunch["l0-r1"] = [SUCCEEDED]
    renames = []
    guard = SimpleNamespace(check=lambda n: ("gone", "l0-r1") if n == "l0" else None)
    ctl = _controller(ops, guard, renames=renames)
    codes = ctl.run()
    assert ops.relaunch_calls == ["l0-r1"] and renames == [("l0", "l0-r1")]
    assert ctl.learners["l0"]["name"] == "l0-r1" and "SUCCEEDED" in codes["l0"]


def test_controller_defers_when_verda_cannot_be_asked():
    from tests.test_controller import FakeOps, RUNNING, SUCCEEDED

    ops = FakeOps()
    ops.status_seq["l0"] = [RUNNING]
    ops.up["l0"] = False
    ops.status_seq["l1"] = [SUCCEEDED]
    guard = SimpleNamespace(check=lambda n: ("unknown", "Verda HTTP 503") if n == "l0" else None)
    ctl = _controller(ops, guard)
    ctl.run()
    assert ops.relaunch_calls == []  # never a blind same-name relaunch
    assert ctl.learners["l0"]["state"] == "abandoned"  # after recover_timeout


def test_worker_saves_verda_ids_in_the_run_record(tmp_path, monkeypatch):
    import yeto.cli as cli
    import yeto.launcher as launcher
    import yeto.runs as runs

    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    runs.create_run("w", {"gpu": "verda:1xl40s@FIN-01"})

    def fake_run(args, on_clusters=None, local_syncer=None, on_instance_ids=None):
        on_clusters(["w-syncer", "w-l0-fin-01"])
        on_instance_ids("w-l0-fin-01", ["i-1"])
        return 0

    monkeypatch.setattr(launcher, "run", fake_run)
    assert cli.cmd_worker("w") == 0
    assert runs.load_run("w")["verda_instance_ids"] == {"w-l0-fin-01": ["i-1"]}


# --- 2.5 unpatched sky: no Verda auto-recovery -----------------------------------------


def test_unverified_sky_disables_verda_recovery_only(monkeypatch, capsys):
    import yeto.launcher as launcher
    from yeto.sky_patches import verda as vp
    from tests.test_controller import FakeOps, RUNNING, SUCCEEDED

    monkeypatch.setattr(vp, "verified_for_current_sky", lambda: (False, "sky 0.14.0 not in verified ['0.13.0']"))
    monkeypatch.setattr("yeto.shape.providers.write_verda_sky_catalog", lambda: "/tmp/x.csv")
    prep = launcher.prepare_verda_islands(["l0"])
    err = capsys.readouterr().err
    assert prep["no_recover"] == {"l0"} and "NOT patched" in err and "recover_timeout=0" in err
    monkeypatch.setattr(vp, "verified_for_current_sky", lambda: (True, ""))
    import yeto.sky_patches as patches

    monkeypatch.setattr(patches, "ensure_local_pth", lambda repo: ("/x/yeto_sky_patches.pth", False))
    monkeypatch.setattr(launcher, "remote_sky_api_server", lambda: None)
    assert launcher.prepare_verda_islands(["l0"])["no_recover"] == set()
    # Hook only now written: a running server may be unpatched -> no recovery.
    monkeypatch.setattr(patches, "ensure_local_pth", lambda repo: ("/x/yeto_sky_patches.pth", True))
    prep = launcher.prepare_verda_islands(["l0"])
    assert prep["no_recover"] == {"l0"} and prep["pth_written"] == "/x/yeto_sky_patches.pth"
    # Remote API server: cannot be patched from here.
    monkeypatch.setattr(patches, "ensure_local_pth", lambda repo: ("/x/yeto_sky_patches.pth", False))
    monkeypatch.setattr(launcher, "remote_sky_api_server", lambda: "https://sky.example.com")
    prep = launcher.prepare_verda_islands(["l0"])
    assert prep["no_recover"] == {"l0"} and "remote" in capsys.readouterr().err

    ops = FakeOps()
    for n in ("l0", "l1"):
        ops.status_seq[n] = [RUNNING]
        ops.up[n] = False
    ops.relaunch_results["l1"] = [9]
    ops.after_relaunch["l1"] = [SUCCEEDED]
    ctl = _controller(ops, no_recover={"l0"})
    codes = ctl.run()
    assert ops.relaunch_calls == ["l1"]  # the other cloud still recovers
    assert codes["l0"].startswith("ABANDONED") and "SUCCEEDED" in codes["l1"]
    assert "auto-recovery disabled" in capsys.readouterr().err


# --- 3.1 diagnostics before teardown ----------------------------------------------------


@pytest.mark.parametrize("mode", ["ok", "timeout", "fail"])
def test_diagnostics_never_block_teardown(tmp_path, monkeypatch, mode):
    import yeto.launcher as launcher

    calls = []

    def run(argv, stdout=None, stderr=None, timeout=None, check=False):
        calls.append(argv[:2])
        if mode == "timeout":
            raise subprocess.TimeoutExpired(argv, timeout)
        if mode == "fail":
            return SimpleNamespace(returncode=255)
        if stdout not in (None, subprocess.DEVNULL):
            stdout.write(b"log line\n")
        return SimpleNamespace(returncode=0)

    got = launcher.collect_teardown_diagnostics("c1", str(tmp_path), head=True, run=run, timeout=1)
    want = {"ok": "ok", "timeout": "timeout", "fail": "failed: exit 255"}[mode]
    assert set(got) == {"job_log", "event_tape", "sky_server_log", "cluster_events"}
    assert set(got.values()) == {want}
    assert ["sky", "logs"] in calls

    monkeypatch.setenv("YETO_TEARDOWN_DIAG", "1")
    downs = []
    monkeypatch.setattr(launcher, "terminate_and_verify", lambda sky, c, **kw: downs.append(c) or True)
    collect = lambda c, head=False: launcher.collect_teardown_diagnostics(  # noqa: E731
        c, str(tmp_path), head=head, run=run, timeout=1)
    assert launcher.teardown_island(None, "c1", collect=collect) is True
    assert downs == ["c1"]

    def explode(c, head=False):
        raise OSError("disk full")

    assert launcher.teardown_island(None, "c2", collect=explode) is True and downs[-1] == "c2"


# --- 3.2 teardown proven at Verda -------------------------------------------------------


def test_teardown_reports_a_node_sky_called_deleted(monkeypatch):
    from yeto.verda_ops import verify_teardown

    api = FakeApi([{"id": "i1", "status": "running", "hostname": "t-l0-fin-01-head", "sticky": True,
                    "os_volume_id": "v1", "volume_ids": ["v1"]}])
    ok, left = verify_teardown(api, ["i1"], "t-l0-fin-01", attempts=3, sleep_fn=lambda s: None)
    assert not ok and "i1" in left[0] and "running" in left[0]
    assert api.deleted == ["i1", "i1", "i1"]  # re-deleted each round, by id


def test_teardown_purges_volumes_from_the_trash():
    from yeto.verda_ops import verify_teardown

    api = FakeApi(
        [{"id": "i1", "status": "running", "hostname": "t-head", "os_volume_id": "v1", "volume_ids": ["v1"]}],
        trash=[{"id": "v1", "is_permanently_deleted": False}],
    )
    ok, left = verify_teardown(api, ["i1"], "t", attempts=3, sleep_fn=lambda s: None)
    assert ok and left == [] and api.purged == ["v1"]

    api = FakeApi([], volumes=[{"id": "v2", "status": "attached"}])
    api.instances["i2"] = {"id": "i2", "status": "deleted", "hostname": "u-head", "os_volume_id": "v2"}
    ok, left = verify_teardown(api, ["i2"], "u", attempts=2, sleep_fn=lambda s: None)
    assert not ok and "v2" in left[0]


def test_teardown_never_succeeds_without_ids_and_never_deletes_strangers():
    from yeto.verda_ops import verify_teardown

    api = FakeApi([{"id": "x9", "status": "running", "hostname": f"{ON_CLOUD}-head", "os_volume_id": "vx"},
                   {"id": "i1", "status": "running", "hostname": f"{ON_CLOUD}-head", "os_volume_id": "v1",
                    "volume_ids": ["v1", "data-vol"]}],
                  trash=[{"id": "v1", "is_permanently_deleted": False},
                         {"id": "data-vol", "is_permanently_deleted": False}])
    ok, left = verify_teardown(api, [], ON_CLOUD, sleep_fn=lambda s: None)
    assert not ok and "no Verda instance ids" in left[0] and api.deleted == []
    ok, left = verify_teardown(api, ["i1"], ON_CLOUD, attempts=2, sleep_fn=lambda s: None)
    assert api.deleted == ["i1"]  # the stranger x9 is reported, never deleted
    assert not ok and any("suspected leftover" in x and "x9" in x for x in left)
    assert api.purged == ["v1"]  # only the recorded instance's OS volume


def test_verda_teardown_check_falls_back_without_ids(monkeypatch):
    import yeto.launcher as launcher
    from yeto.verda_ops import VerdaInstanceGuard

    guard = VerdaInstanceGuard(FakeApi(), ["c"], resolve_on_cloud=lambda c: None)
    monkeypatch.setitem(launcher.VERDA_ISLANDS, "c", guard)
    assert launcher._verda_teardown_check("c") is None  # sky's cloud probe decides instead
    guard.ids["c"], guard.on_cloud["c"] = ["i1"], ON_CLOUD
    assert callable(launcher._verda_teardown_check("c"))


def test_verda_api_refreshes_expired_token_and_retries_401(home):
    from yeto.verda_ops import VerdaApi

    (home / ".verda" / "config.json").write_text('{"client_id": "a", "client_secret": "b"}')
    now = [0.0]
    tokens, calls = [], []

    def request(method, path, token=None, body=None, params=None):
        if path == "/oauth2/token":
            tokens.append(len(tokens) + 1)
            return {"access_token": f"t{len(tokens)}", "expires_in": 3600}
        calls.append(token)
        if token == "t2" and len([c for c in calls if c == "t2"]) == 1:
            err = RuntimeError("HTTP 401")
            err.status = 401
            raise err
        return []

    api = VerdaApi(request=request, clock=lambda: now[0])
    api.list_instances()
    now[0] = 3550.0  # within the refresh margin of expiry
    api.list_instances()
    assert calls == ["t1", "t2", "t3"] and tokens == [1, 2, 3]  # refreshed, then 401 -> refreshed once more


def test_terminate_and_verify_uses_the_verda_answer():
    from yeto.launcher import terminate_and_verify

    downs = []
    assert terminate_and_verify(None, "c", down=lambda: downs.append(1), verda_check=lambda: (True, [])) is True
    assert terminate_and_verify(None, "c", down=lambda: downs.append(1),
                                verda_check=lambda: (False, ["instance i1 running"])) is False

    def sky_down_raises():
        raise RuntimeError("cluster does not exist")

    # sky no longer knows the cluster: Verda's answer still decides.
    assert terminate_and_verify(None, "c", down=sky_down_raises, verda_check=lambda: (False, ["i1"])) is False


def test_local_pth_for_the_sky_api_server(tmp_path, monkeypatch, capsys):
    import yeto.launcher as launcher
    import yeto.sky_patches as patches
    from yeto.sky_patches import PTH_NAME, ensure_local_pth, verda as vp

    path, fresh = ensure_local_pth(str(REPO), site_dir=str(tmp_path))
    assert fresh and path == str(tmp_path / PTH_NAME)
    assert ensure_local_pth(str(REPO), site_dir=str(tmp_path)) == (path, False)
    from yeto.sky_patches import remove_local_pth, uninstall

    # Another worktree takes over with a warning; ours is then not removed.
    ensure_local_pth("/other/worktree", site_dir=str(tmp_path))
    assert "another worktree" in capsys.readouterr().err
    assert remove_local_pth(path, str(REPO)) is False and os.path.exists(path)
    assert remove_local_pth(path, "/other/worktree") is True and not os.path.exists(path)
    ensure_local_pth(str(REPO), site_dir=str(tmp_path))
    assert uninstall(site_dir=str(tmp_path)) == path and not os.path.exists(path)
    monkeypatch.setattr(launcher, "remote_sky_api_server", lambda: None)

    monkeypatch.setattr(vp, "verified_for_current_sky", lambda: (True, ""))
    monkeypatch.setattr("yeto.shape.providers.write_verda_sky_catalog", lambda: "/tmp/x.csv")

    def unwritable(repo, site_dir=None):
        raise PermissionError("read-only site-packages")

    monkeypatch.setattr(patches, "ensure_local_pth", unwritable)
    prep = launcher.prepare_verda_islands(["l0"])
    assert prep["no_recover"] == {"l0"}  # unpatched server -> no auto-recovery
    assert "cannot install the patch hook" in capsys.readouterr().err


def test_launch_verda_island_sends_ordered_any_of(monkeypatch):
    import yeto.launcher as launcher
    from yeto.gpu_spec import parse_gpu_spec

    class Res:
        def __init__(self, **kw):
            self.kw = kw

        def copy(self, **kw):
            return Res(**{**self.kw, **kw})

    task = SimpleNamespace(resources={Res(accelerators="L40S:1", disk_size=200, image_id="x")})
    task.set_resources = lambda r: setattr(task, "resources", r)
    sent = []
    fake_sky = SimpleNamespace(
        launch=lambda t, cluster_name, retry_until_up: sent.append((cluster_name, [r.kw for r in t.resources])) or "rid",
        stream_and_get=lambda rid: (3, None),
        CLOUD_REGISTRY=SimpleNamespace(from_str=lambda c: c.upper()),
    )
    monkeypatch.setattr(providers.VerdaSignals, "_fetch_types", lambda self: TYPES)
    monkeypatch.setattr(providers.VerdaSignals, "_fetch_availability",
                        lambda self: {"FIN-01": ["1L40S.20V"], "FIN-03": ["1L40S.20V"]})
    spec = parse_gpu_spec("verda:1xl40s")[0]
    got = launcher.launch_verda_island(fake_sky, task, "r-l0-verda", spec, SimpleNamespace(spot=False))
    assert got == (3, None)
    (name, res), = sent
    assert name == "r-l0-verda"
    assert [(r["cloud"], r["region"], r["instance_type"], r["disk_size"]) for r in res] == [
        ("VERDA", "FIN-01", "1L40S.20V", 200), ("VERDA", "FIN-03", "1L40S.20V", 200)]
    assert all("infra" not in r for r in res)


def test_verda_copy_override_works_on_real_sky_resources():
    # Regression: base.copy(infra=...) on a Resources that already has
    # cloud/region raised "Cannot specify both infra and cloud, region, or zone".
    sky = pytest.importorskip("sky")
    import yeto.launcher as launcher

    base = sky.Resources(infra="verda/FIN-01", accelerators="RTX6000Ada:1", disk_size=200)
    r = base.copy(**launcher._verda_copy_override(
        sky, {"infra": "verda/FIN-02", "instance_type": "1RTX6000ADA.10V", "accelerators": "RTX6000Ada:1"}))
    assert (str(r.cloud), r.region, r.instance_type) == ("Verda", "FIN-02", "1RTX6000ADA.10V")


# --- 5.1 Verda head / syncer: no ports, in-VM firewall, external probe ------------------


def test_verda_syncer_gets_no_ports_and_a_ufw_setup(monkeypatch):
    import yeto.launcher as launcher

    assert launcher.syncer_ports(SimpleNamespace(syncer_region="verda/FIN-01")) is None
    assert launcher.syncer_ports(SimpleNamespace(syncer_region="nebius/eu-north1")) == [launcher.SYNCER_PORT]
    assert launcher.syncer_ports(SimpleNamespace(syncer_region="us-east-1")) == [launcher.SYNCER_PORT]
    rules = launcher.ufw_setup(29400)
    assert "ufw default deny incoming" in rules and "ufw allow 22/tcp" in rules
    assert "ufw allow 29400/tcp" in rules and "ufw --force enable" in rules
    assert rules.count("ufw allow") == 2  # SSH and the syncer, nothing else

    task = _head_task(monkeypatch, "verda:1xl40s@FIN-01")
    task_verda = _head_task_on(monkeypatch, "verda/FIN-01")
    assert task.resources.kwargs["ports"] == [launcher.SYNCER_PORT]
    assert task_verda.resources.kwargs["ports"] is None
    assert task_verda.setup.index("ufw --force enable") < task_verda.setup.index("pip install")


def _head_task_on(monkeypatch, syncer_region):
    import yeto.cli as cli
    import yeto.launcher as launcher

    _head_task(monkeypatch, "verda:1xl40s@FIN-01")  # installs the fake sky
    monkeypatch.setattr(launcher, "head_cloud_credentials", lambda clouds: ({}, {}))
    args = SimpleNamespace(gpu="verda:1xl40s@FIN-01", syncer_region=syncer_region, syncer_memory="32",
                           loss_function="ce", wandb=False, controller="head")
    return cli._make_head_task(args)


class _Conn:
    def __init__(self, greeting=b"yeto-probe"):
        self.greeting = greeting

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def settimeout(self, t):
        pass

    def recv(self, n):
        return self.greeting[:n]


def test_tcp_probe_success_and_failure():
    import yeto.launcher as launcher

    tries = []

    def flaky(addr, timeout=None):
        tries.append(addr)
        if len(tries) < 3:
            raise ConnectionRefusedError("refused")
        return _Conn()

    ok, detail = launcher.tcp_probe("203.0.113.9", 29400, expect=launcher.PROBE_BANNER, connect=flaky,
                                    sleep=lambda s: None)
    assert ok and "attempt 3" in detail and tries[0] == ("203.0.113.9", 29400)

    def never(addr, timeout=None):
        raise TimeoutError("timed out")

    ok, detail = launcher.tcp_probe("203.0.113.9", 29400, attempts=4, connect=never, sleep=lambda s: None)
    assert not ok and "after 4 attempt(s)" in detail and "timed out" in detail
    ok, _ = launcher.tcp_probe("h", 1, expect=b"yeto-probe", attempts=1, connect=lambda a, timeout=None: _Conn(b"SSH-2.0"))
    assert not ok  # something else answered: not our listener
    assert "29400" in launcher.probe_listener_command(29400)


@pytest.mark.parametrize("reachable", [True, False])
def test_head_launch_on_verda_probes_before_islands(tmp_path, monkeypatch, reachable):
    import yeto.cli as cli
    import yeto.runs as runs

    monkeypatch.setattr(runs, "RUNS_DIR", tmp_path / "runs")
    events = []
    monkeypatch.setattr(cli, "_make_head_task", lambda args, mounts=None: "head-task")
    monkeypatch.setattr(cli, "_sky_launch_head", lambda task, c: events.append(("launch", c)) or SimpleNamespace(head_ip="203.0.113.5"))
    monkeypatch.setattr(cli, "_probe_head_port", lambda c, ip: events.append(("probe", c, ip)) or reachable)
    monkeypatch.setattr(cli, "_down_and_verify", lambda c: events.append(("down", c)) or True)
    monkeypatch.setattr(cli, "_sky_exec_head", lambda task, c: events.append(("exec", c)) or 7)
    monkeypatch.setattr(cli, "_stream_head_logs", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_record_head_result", lambda *a, **k: None)
    import yeto.launcher as launcher

    monkeypatch.setattr(launcher, "head_cloud_credentials", lambda clouds: ({}, {}))
    fake_sky = types.ModuleType("sky")
    fake_sky.Task = lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "sky", fake_sky)
    rc = cli.main(["launch", "--gpu", "verda:1xl40s@FIN-01", "--model", "gemma4", "--model-revision", "a" * 40,
                   "--data", "org/ds", "--data-revision", "b" * 40, "--controller", "head",
                   "--syncer-region", "verda/FIN-01", "--cluster-prefix", "vh"])
    assert events[:2] == [("launch", "vh-head"), ("probe", "vh-head", "203.0.113.5")]
    if reachable:
        assert ("exec", "vh-head") in events and ("down", "vh-head") not in events
    else:
        assert rc == 1 and events[2:] == [("down", "vh-head")]  # never exec'd, torn down


def test_remote_api_server_detection(monkeypatch):
    import yeto.launcher as launcher

    monkeypatch.setitem(sys.modules, "sky.server", None)  # sky absent -> env var decides
    monkeypatch.delenv("SKYPILOT_API_SERVER_ENDPOINT", raising=False)
    assert launcher.remote_sky_api_server() is None
    monkeypatch.setenv("SKYPILOT_API_SERVER_ENDPOINT", "http://127.0.0.1:46580")
    assert launcher.remote_sky_api_server() is None
    monkeypatch.setenv("SKYPILOT_API_SERVER_ENDPOINT", "https://sky.example.com")
    assert launcher.remote_sky_api_server() == "https://sky.example.com"


def test_failed_launch_marker_expires_and_retries_failed_deletes(ids_file, monkeypatch):
    from yeto.sky_patches import verda as vp

    m, fake = _patched([{"id": "k1", "status": "running", "hostname": "vfix-t-head"}])
    orig_create = fake.instance_create

    def create_then_hang(cfg):
        orig_create(cfg)
        fake.instances[fake.created[-1]]["status"] = "provisioning"

    fake.instance_create = create_then_hang
    orig_action = fake.instance_action
    fails = [True]

    def flaky_delete(instance_id, action):
        if fails[0]:
            fails[0] = False
            raise RuntimeError("HTTP 500")
        orig_action(instance_id, action)

    fake.instance_action = flaky_delete
    with pytest.raises(RuntimeError):
        m.run_instances("FIN-01", "vfix-t", "vfix-t", SimpleNamespace(count=2))
    assert fake.deleted == []  # the in-launch delete failed
    m.terminate_instances("vfix-t", {})  # provisioner teardown retries it
    assert fake.deleted == ["new-1"] and fake.instances["k1"]["status"] == "running"
    assert "vfix-t" not in vp._FAILED_LAUNCH
    # A stale marker (past the TTL) does not shield a later deliberate down.
    vp._FAILED_LAUNCH["vfix-t"] = ([], 0.0)
    monkeypatch.setattr(vp, "FAILED_LAUNCH_TTL_S", 1.0)
    m.terminate_instances("vfix-t", {})
    assert "k1" in fake.deleted


def test_parallel_diagnostics_are_bounded(monkeypatch):
    import threading as th

    import yeto.launcher as launcher

    monkeypatch.setenv("YETO_TEARDOWN_DIAG", "1")
    release = th.Event()
    seen = []

    def collect(c, head=False):
        seen.append(c)
        if c == "slow":
            release.wait(5)
        return {"job_log": "ok"}

    got = launcher.collect_diagnostics_parallel(["a", "slow", "b"], total_timeout=0.3, collect=collect)
    release.set()
    assert sorted(seen) == ["a", "b", "slow"] and set(got) == {"a", "b"}


def test_probe_runs_while_the_listener_job_is_still_submitting(monkeypatch):
    import threading as th

    import yeto.cli as cli
    import yeto.launcher as launcher

    job_done = th.Event()
    fake_sky = types.ModuleType("sky")
    fake_sky.Task = lambda **kw: SimpleNamespace(**kw)
    fake_sky.exec = lambda task, cluster_name: "rid"
    fake_sky.stream_and_get = lambda rid: job_done.wait(5)  # blocks until the listener exits
    monkeypatch.setitem(sys.modules, "sky", fake_sky)
    monkeypatch.setattr(cli, "PROBE_SUBMIT_JOIN_S", 1.0)

    def probe(host, port, expect=None):
        job_done.set()  # our connection is what ends the listener job
        return True, "reachable"

    monkeypatch.setattr(launcher, "tcp_probe", probe)
    assert cli._probe_head_port("vh-head", "203.0.113.5") is True


def test_pth_line_hook_works_when_executed_inside_a_function(tmp_path):
    # site.addpackage exec()s each .pth line in a function scope; the hook's
    # class must still see _R (regression: NameError "_R is not defined").
    import subprocess
    import sys as _sys

    from yeto.sky_patches import pth_line

    line = pth_line("/nonexistent-yeto-repo").splitlines()[1]
    prog = (
        "import sys\n"
        "def addpackage(line):\n"
        "    exec(line)\n"
        f"addpackage({line!r})\n"
        "hook = sys.meta_path[0]\n"
        "assert type(hook).__name__ == '_YetoLazy'\n"
        "sys.modules.pop('yeto', None); sys.modules.pop('yeto.sky_patches', None)\n"
        "sys.path[:] = [p for p in sys.path if 'yeto' not in p and p not in ('', '.')]\n"
        "assert hook.find_spec('sky.provision.verda.instance') is None\n"
    )
    r = subprocess.run([_sys.executable, "-c", prog], capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "not applied" not in r.stderr, r.stderr
