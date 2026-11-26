"""Node health while a run goes on (``yeto.hw.node_health``): the cuda checker over the
recorded tool outputs of ``yeto preflight``'s tests plus made-up faults (XID 79, ECC
growth, an unfinished NVLink fabric, ``mlx5_3`` down), the verdict of two bad checks in a
row, and the null checker of every other device family. The IB / fabric parsing is the
preflight's own (one implementation, ``yeto.hw.host_probe``)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from test_cluster_preflight import _sections_out, devinfo_text, ibstat_text  # tests/ is on sys.path

from yeto.hw import host_probe, node_health
from yeto.hw.node_health import HealthTracker, HealthWatch, checker_for, hcas_of
from yeto.launch.preflight import cluster as cp

REPO = Path(__file__).resolve().parents[1]
BUSES = ("18", "2a", "3a", "5d", "9a", "ab", "ba", "db")
HCA = "=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7"  # the customer inventory's NCCL_IB_HCA
FABRIC_OK = "\n".join(["    Fabric", "        State                             : Completed",
                       "        Status                            : Success",
                       "        CliqueId                          : 32766"] * 8)
FABRIC_IN_PROGRESS = "    Fabric\n        State : In Progress\n        Status : N/A\n" * 8  # the preflight's sample
BOOT = "0b6f8c1e-2d4a-4f7e-9a51-3c2d1e0f9a88"


def gpus_text(n=8, mib=275040, ecc=None, gone=()) -> str:
    ecc = ecc or {}
    rows = []
    for i in range(n):
        if i in gone:
            rows.append(f"Unable to determine the device handle for GPU0000:{BUSES[i].upper()}:00.0: Unknown Error")
            continue
        rows.append(f"{i}, 00000000:{BUSES[i].upper()}:00.0, NVIDIA B300 SXM6 AC, {mib}, {ecc.get(i, 0)}")
    return "\n".join(rows)


def xid_line(ts: float, gpu: int, code: int, what: str = "GPU has fallen off the bus.") -> str:
    return f"[{ts:12.6f}] NVRM: Xid (PCI:0000:{BUSES[gpu]}:00): {code}, pid=4242, name=python3, {what}"


def xid_text(*lines: str, boot=BOOT, source="dmesg") -> str:
    return "\n".join([f"boot={boot}", f"source={source}", *lines])


def node_text(**over) -> str:
    """What the cuda check script prints on a healthy 8 x B300 IB node (the preflight's samples)."""
    sections = {"gpus": (0, gpus_text()), "fabric": (0, FABRIC_OK), "fabricmanager": (0, "active"),
                "xid": (0, xid_text()), "ibstat": (0, ibstat_text()), "ibv_devinfo": (3, "")}
    sections.update(over)
    script = node_health.checker_for("cuda").script(hcas_of(HCA))
    return _sections_out(script, sections)


def check(text, prev=None, hcas=None, **kw):
    return checker_for("cuda", **kw).parse(text, hcas_of(HCA) if hcas is None else hcas, prev)


def test_a_healthy_node_passes_for_both_jobs_and_sets_its_baselines():
    r = check(node_text())
    assert r.checked and r.ok_single is True and r.ok_multi is True and r.reasons == []
    assert r.facts["gpu_count"] == 8 and r.facts["fabric"] == "8/8 completed" and r.facts["fm"] == "active"
    assert r.facts["ib_rate"]["mlx5_0"] == 400 and r.facts["ports"]["mlx5_3"] == "Active 400"
    assert r.facts["xid"] == {"boot": BOOT, "ts": None, "count": 0} and r.facts["xid_source"] == "dmesg"


def test_xid_79_since_the_previous_check_fails_the_gpu():
    first = check(node_text(xid=(0, xid_text(xid_line(100.5, 2, 13, "Graphics Exception")))))
    assert first.ok_single  # what the log held before the first check is history
    second = check(node_text(xid=(0, xid_text(xid_line(100.5, 2, 13, "Graphics Exception"),
                                              xid_line(3600.25, 5, 79)))), first.facts)
    assert second.ok_single is False and second.ok_multi is False
    assert second.reasons == ["GPU 5 XID 79"] and second.facts["gpu_reasons"] == ["GPU 5 XID 79"]
    third = check(node_text(xid=(0, xid_text(xid_line(100.5, 2, 13, "Graphics Exception"),
                                             xid_line(3600.25, 5, 79)))), second.facts)
    assert third.ok_single  # the same line again is not a new XID
    old = check(node_text(xid=(0, xid_text(xid_line(7.0, 4, 79)))))
    assert old.ok_single and old.facts["xid"]["ts"] == 7.0  # a hardware XID before the first check: history


def test_an_application_xid_is_a_fact_not_a_fault():
    first = check(node_text())
    r = check(node_text(xid=(0, xid_text(xid_line(50.0, 1, 13, "Graphics Exception")))), first.facts)
    assert r.ok_single and r.facts["xid_other"] == ["GPU 1 XID 13"]


def test_xids_after_a_reboot_or_without_timestamps_are_counted_from_the_cursor():
    first = check(node_text(xid=(0, xid_text(xid_line(10.0, 0, 13)))))
    rebooted = check(node_text(xid=(0, xid_text(xid_line(5.0, 0, 48), boot="other-boot"))), first.facts)
    assert rebooted.reasons == ["GPU 0 XID 48"]
    plain = lambda g, c: f"NVRM: Xid (PCI:0000:{BUSES[g]}:00): {c}, pid=1, name=x"  # noqa: E731 - printk.time=0
    a = check(node_text(xid=(0, xid_text(plain(0, 13)))))
    b = check(node_text(xid=(0, xid_text(plain(0, 13), plain(3, 95)))), a.facts)
    assert b.reasons == ["GPU 3 XID 95"]


def test_uncorrectable_ecc_growth_fails_the_gpu():
    first = check(node_text(gpus=(0, gpus_text(ecc={3: 1}))))
    assert first.ok_single  # a count already there at the first check is the baseline
    grown = check(node_text(gpus=(0, gpus_text(ecc={3: 3}))), first.facts)
    assert grown.ok_single is False and grown.reasons == ["GPU 3 uncorrectable ECC +2"]
    same = check(node_text(gpus=(0, gpus_text(ecc={3: 3}))), grown.facts)
    assert same.ok_single


def test_an_unfinished_nvlink_fabric_fails_the_node():
    r = check(node_text(fabric=(0, FABRIC_IN_PROGRESS)))
    assert r.ok_single is False
    assert r.reasons == ["NVLink fabric In Progress/N/A on 8/8 GPUs (fabric manager: active)"]
    stopped = check(node_text(fabricmanager=(3, "failed")), check(node_text()).facts)
    assert stopped.reasons == ["nvidia-fabricmanager failed (was active)"]


def test_mlx5_3_down_fails_a_multi_node_job_only():
    r = check(node_text(ibstat=(0, ibstat_text(down=(3,)))))
    assert r.checked and r.ok_single is True and r.ok_multi is False
    assert r.reasons == ["mlx5_3 port DOWN"] and r.facts["ib_reasons"] == ["mlx5_3 port DOWN"]
    assert r.facts["ports"]["mlx5_3"] == "Down 400"
    assert r.job_reasons(multi=False) == [] and r.job_reasons(multi=True) == ["mlx5_3 port DOWN"]


def test_ports_outside_nccl_ib_hca_and_a_job_without_hcas_are_not_judged():
    storage_down = ibstat_text(storage=True).replace("CA 'mlx5_8'\n\tCA type: MT4125\n\tFirmware version: "
                                                     "22.41.1000\n\tPort 1:\n\t\tState: Active",
                                                     "CA 'mlx5_8'\n\tCA type: MT4125\n\tFirmware version: "
                                                     "22.41.1000\n\tPort 1:\n\t\tState: Down")
    r = check(node_text(ibstat=(0, storage_down)))
    assert r.ok_multi is True and r.facts["ports"]["mlx5_8"].startswith("Down")
    no_hca = check(node_text(ibstat=(0, ibstat_text(down=(3,)))), hcas=[])
    assert no_hca.ok_multi is True and no_hca.reasons == []


def test_a_slower_port_a_missing_hca_and_unread_ib_tools():
    first = check(node_text())
    slow = check(node_text(ibstat=(0, ibstat_text().replace("Rate: 400", "Rate: 200", 4))), first.facts)
    assert slow.ok_multi is False and slow.ok_single is True
    assert slow.reasons == [f"mlx5_{i} rate 200 < 400 Gb/s" for i in range(4)]
    declared = check(node_text(), None, min_ib_gbps=800)
    assert declared.reasons[0] == "mlx5_0 rate 400 < 800 Gb/s"
    missing = check(node_text(ibstat=(0, ibstat_text(n=3, storage=False))))
    assert missing.ok_multi is False and missing.reasons == [f"mlx5_{i} missing" for i in range(3, 8)]
    unread = check(node_text(ibstat=(127, "bash: ibstat: command not found"),
                             ibv_devinfo=(127, "bash: ibv_devinfo: command not found")))
    assert unread.checked and unread.ok_single is True and unread.ok_multi is None


def test_ibv_devinfo_is_read_when_ibstat_is_missing():
    devinfo = devinfo_text().replace("PORT_ACTIVE", "PORT_DOWN", 1)
    r = check(node_text(ibstat=(127, "bash: ibstat: command not found"), ibv_devinfo=(0, devinfo)))
    assert r.reasons == ["mlx5_0 port DOWN"] and r.facts["ports"]["mlx5_1"] == "Active 400"


def test_a_gpu_that_fell_off_the_bus_and_a_missing_gpu():
    first = check(node_text())
    gone = check(node_text(gpus=(15, gpus_text(gone=(5,)))), first.facts)
    assert gone.ok_single is False and gone.reasons == ["GPU 5 unreachable (Unknown Error)"]
    fewer = check(node_text(gpus=(0, gpus_text(n=7))), first.facts)
    assert fewer.reasons == ["7 of 8 GPUs visible"]
    expected = check(node_text(gpus=(0, gpus_text(n=7))), None, expect_gpus=8)
    assert expected.reasons == ["7 of 8 GPUs visible"]
    none = check(node_text(gpus=(6, "No devices were found")), first.facts)
    assert none.checked and none.reasons == ["0 of 8 GPUs visible"]
    shrunk = check(node_text(gpus=(0, gpus_text(mib=200000))), first.facts)
    assert shrunk.reasons[0] == "GPU 0 memory 200000 MiB < 275040 MiB" and len(shrunk.reasons) == 8


def test_nvidia_smi_that_times_out_is_a_read_failure_that_keeps_the_baselines():
    first = check(node_text(xid=(0, xid_text(xid_line(1.0, 0, 13)))))
    r = check(node_text(gpus=(124, "")), first.facts)
    assert not r.checked and r.ok_single is None and r.ok_multi is None
    assert r.reasons == ["nvidia-smi: timed out after 30s"]
    assert all(r.facts[k] == first.facts[k] for k in ("gpu_count", "bus", "ecc", "xid", "ib_rate"))
    assert not check("").checked and check("").reasons == ["nvidia-smi: no output (the check did not run)"]


def test_two_bad_checks_in_a_row_make_a_node_lost_and_a_read_failure_counts_neither_way():
    ok, bad = check(node_text()), check(node_text(fabric=(0, FABRIC_IN_PROGRESS)))
    failed = check(node_text(gpus=(124, "")))
    t = HealthTracker()
    assert [t.update(x) for x in (ok, bad, ok, bad, bad, bad, ok)] == \
        ["ok", "bad_once", "ok", "bad_once", "lost", "lost", "ok"]
    t = HealthTracker()
    assert [t.update(x) for x in (bad, failed, bad)] == ["bad_once", "read_failed", "lost"]
    assert [HealthTracker().update(failed) for _ in range(3)] == ["read_failed"] * 3
    port_down = check(node_text(ibstat=(0, ibstat_text(down=(3,)))))
    single, multi = HealthTracker(multi=False), HealthTracker(multi=True)
    assert [single.update(port_down) for _ in range(2)] == ["ok", "ok"]
    assert [multi.update(port_down) for _ in range(2)] == ["bad_once", "lost"]
    unread_ib = check(node_text(ibstat=(127, ""), ibv_devinfo=(127, "")))
    assert HealthTracker(multi=True).update(unread_ib) == "read_failed"


def test_a_watch_asks_for_one_event_per_change_and_records_the_heartbeat_field():
    texts = iter([node_text(), node_text(gpus=(124, "")), node_text(gpus=(124, "")), node_text(),
                  node_text(ibstat=(0, ibstat_text(down=(3,)))), node_text(ibstat=(0, ibstat_text(down=(3,)))),
                  node_text(ibstat=(0, ibstat_text(down=(3,))))])
    w = HealthWatch("cuda", hcas=hcas_of(HCA), multi=True, clock=lambda: 1000.0)
    got = [(r.verdict, r.event) for r in (w.observe(next(texts)) for _ in range(7))]
    assert got == [("ok", None), ("read_failed", "node_health_read_failed"), ("read_failed", None), ("ok", None),
                   ("bad_once", None), ("lost", "node_health"), ("lost", None)]
    doc = w.record()
    assert doc["verdict"] == "lost" and doc["reasons"] == ["mlx5_3 port DOWN"] and doc["bad_in_row"] == 3
    assert doc["ok"] is False and doc["multi"] is True and doc["facts"]["ports"]["mlx5_3"] == "Down 400"
    single = HealthWatch("cuda", hcas=hcas_of(HCA))
    single.observe(node_text(ibstat=(0, ibstat_text(down=(3,)))))
    doc = single.record()
    assert doc["verdict"] == "ok" and doc["reasons"] == [] and doc["notes"] == ["mlx5_3 port DOWN"]


@pytest.mark.parametrize("family", ["rocm", "npu", "ppu", "ascend", "", "metal"])
def test_every_other_family_has_the_null_checker(family):
    c = checker_for(family)
    assert not c.enabled and c.script(["mlx5_0"]) == ""
    r = c.parse(node_text(), ["mlx5_0"], None)
    assert not r.checked and r.ok_single is None and r.ok_multi is None and r.reasons == []
    w = HealthWatch(family)
    assert not w.enabled and w.record() is None


def test_a_family_checker_can_be_registered(monkeypatch):
    class Fake(node_health.NullChecker):
        enabled = True

    monkeypatch.setitem(node_health.CHECKERS, "rocm", lambda **kw: Fake("rocm"))
    assert checker_for("rocm").enabled and isinstance(checker_for("rocm"), Fake)


def test_one_ib_and_fabric_parser_shared_with_the_preflight():
    for name in ("parse_ibstat", "parse_ibv_devinfo", "parse_fabric", "parse_sections", "probe_script",
                 "hca_filter", "ports_of", "IbPort", "Section", "SECTION", "RC"):
        assert getattr(cp, name) is getattr(host_probe, name), name
    sources = [p for p in (REPO / "yeto").rglob("*.py") if "def parse_ibstat" in p.read_text(errors="replace")
               or "def parse_ibv_devinfo" in p.read_text(errors="replace")]
    assert sources == [REPO / "yeto" / "hw" / "host_probe.py"]
    assert "ibstat" in node_health.CudaChecker().script([]) and "def parse_ib" not in \
        (REPO / "yeto" / "hw" / "node_health.py").read_text()


def test_the_cli_dry_run_prints_the_script():
    out = subprocess.run([sys.executable, "-m", "yeto.hw.node_health", "--hca", HCA, "--multi", "--dry-run"],
                         cwd=REPO, capture_output=True, text=True, timeout=60, check=True).stdout
    assert out == node_health.CudaChecker().script(hcas_of(HCA)) and "nvidia-smi --query-gpu" in out
    other = subprocess.run([sys.executable, "-m", "yeto.hw.node_health", "--family", "npu", "--dry-run"],
                           cwd=REPO, capture_output=True, text=True, timeout=60, check=True).stdout
    assert "no health checker" in other


def test_hcas_of_keeps_the_nccl_ib_hca_value():
    assert ",".join(hcas_of(HCA)) == HCA and hcas_of(None) == [] and hcas_of(" mlx5_0 , mlx5_1 ") == ["mlx5_0",
                                                                                                    "mlx5_1"]
