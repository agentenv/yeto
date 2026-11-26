"""Node health while a run goes on (``yeto.hw.node_health``): the cuda checker over the
recorded tool outputs of ``yeto preflight``'s tests plus made-up faults (XID 79, ECC
growth, an unfinished NVLink fabric, ``mlx5_3`` down), the verdict of two bad checks in a
row, and the null checker of every other device family. The IB / fabric parsing is the
preflight's own (one implementation, ``yeto.hw.host_probe``)."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from test_cluster_preflight import _sections_out, devinfo_text, ibstat_text  # tests/ is on sys.path

from yeto.hw import host_probe, node_health
from yeto.hw.host_probe import probe_script
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


def PORT_DOWN_TEXT():  # noqa: N802 - a sample, like the constants above
    return node_text(ibstat=(0, ibstat_text(down=(3,))))


def check(text, prev=None, hcas=None, **kw):
    return checker_for("cuda", **kw).parse(text, hcas_of(HCA) if hcas is None else hcas, prev)


def test_a_healthy_node_passes_for_both_jobs_and_sets_its_baselines():
    r = check(node_text())
    assert r.checked and r.ok_single is True and r.ok_multi is True and r.reasons == []
    assert r.facts["gpu_count"] == 8 and r.facts["fabric"] == "8/8 completed" and r.facts["fm"] == "active"
    assert r.facts["ib_rate"]["mlx5_0"] == 400 and r.facts["ports"]["mlx5_3"] == "Active 400"
    assert r.facts["xid"] == {"boot": BOOT, "ts": None, "count": 0, "last": None, "last_n": 0, "run": 0,
                              "before": None}
    assert r.facts["xid_source"] == "dmesg"


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
    assert unread.job_reasons(multi=True) == [node_health.IB_UNREAD] and unread.job_reasons(multi=False) == []
    sick_unread = check(node_text(fabric=(0, FABRIC_IN_PROGRESS), ibstat=(127, ""), ibv_devinfo=(127, "")))
    assert sick_unread.ok_single is False and sick_unread.ok_multi is False  # the GPUs decide first
    assert check(node_text(fabric=(0, FABRIC_IN_PROGRESS))).ok_multi is False


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
    assert none.checked and none.reasons == ["no GPU visible (nvidia-smi: No devices were found)"]
    # a cuda node with no GPU is bad from its first check on, whatever the baseline
    blank = check(node_text(gpus=(6, "No devices were found")), None)
    assert blank.checked and blank.ok_single is False and blank.ok_multi is False
    shrunk = check(node_text(gpus=(0, gpus_text(mib=200000))), first.facts)
    assert shrunk.reasons[0] == "GPU 0 memory 200000 MiB < 275040 MiB" and len(shrunk.reasons) == 8


def test_nvidia_smi_that_times_out_is_a_read_failure_that_keeps_the_baselines():
    first = check(node_text(xid=(0, xid_text(xid_line(1.0, 0, 13)))))
    r = check(node_text(gpus=(124, ""), xid=(0, xid_text(xid_line(1.0, 0, 13)))), first.facts)
    assert not r.checked and r.ok_single is None and r.ok_multi is None
    assert r.reasons == [f"nvidia-smi: timed out after {node_health.SECTION_TIMEOUT_S}s"]
    # killed by its section's timeout after it printed some GPUs: those rows are not the answer
    partial = check(node_text(gpus=(124, gpus_text(n=4))), first.facts)
    assert partial.checked is False and partial.ok_single is None and partial.facts["gpu_count"] == 8
    assert all(r.facts[k] == first.facts[k] for k in ("gpu_count", "bus", "ecc", "xid", "ib_rate"))
    assert not check("").checked and check("").reasons == ["nvidia-smi: no output (the check did not run)"]


KILLED = f"node health check timed out after {node_health.HEALTH_LIMIT_S}s (killed)"


def _cut(text: str, before: str) -> str:
    """``text`` cut off where section ``before`` starts, then the kill's mark (a killed check)."""
    return text[:text.index(f"@@yeto-section {before}")] + f"{node_health.TIMEOUT_MARK} 45\n"


def test_a_check_killed_past_its_limit_is_a_read_failure_that_keeps_the_baselines():
    first = check(node_text())
    # killed while nvidia-smi ran: the GPUs it printed so far are not read as the whole answer
    partial = _cut(node_text(gpus=(0, gpus_text(n=3))), "fabric").replace("@@yeto-rc 0\n@@yeto-section fabric", "")
    partial = partial[:partial.rindex("@@yeto-rc")] + f"{node_health.TIMEOUT_MARK} 45\n"
    killed = check(partial, first.facts)
    assert not killed.checked and killed.reasons == [KILLED] and killed.facts["gpu_count"] == 8
    assert killed.facts["xid"] == first.facts["xid"]  # nothing new in a completed XID section
    whole = check(node_text(gpus=(0, gpus_text(n=3))) + f"{node_health.TIMEOUT_MARK} 45\n", first.facts)
    assert not whole.checked and whole.reasons == [KILLED]  # even all of its sections: not judged but XIDs
    # a hardware XID in the completed XID section counts: bad, not a read failure
    xid_79 = check(_cut(node_text(xid=(0, xid_text(xid_line(9000.0, 5, 79)))), "gpus"), first.facts)
    assert xid_79.checked and xid_79.ok_single is False and xid_79.ok_multi is False
    assert xid_79.reasons == ["GPU 5 XID 79"] and xid_79.facts["read_error"] == KILLED
    # an XID section cut off in its middle is not read
    torn = node_text(xid=(0, xid_text(xid_line(9000.0, 5, 79))))
    torn = torn[:torn.index("@@yeto-rc")] + f"{node_health.TIMEOUT_MARK} 45\n"
    cut = check(torn, first.facts)
    assert not cut.checked and cut.facts["xid"] == first.facts["xid"]
    again = check(node_text(xid=(0, xid_text(xid_line(9000.0, 5, 79)))), cut.facts)
    assert again.reasons == ["GPU 5 XID 79"]  # the next whole check reads it


def test_nvidia_smi_without_its_driver_is_bad_and_a_hardware_xid_counts_while_it_does_not_answer():
    first = check(node_text())
    down = check(node_text(gpus=(9, "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA "
                                    "driver. Make sure that the latest NVIDIA driver is installed and running.")),
                 first.facts)
    assert down.checked and down.ok_single is False and down.ok_multi is False
    assert down.reasons == ["nvidia-smi: couldn't communicate with the NVIDIA driver"]
    hung = check(node_text(gpus=(124, ""), xid=(0, xid_text(xid_line(9000.0, 5, 79)))), first.facts)
    assert hung.checked and hung.ok_single is False and hung.reasons == ["GPU 5 XID 79"]
    assert hung.facts["read_error"] == f"nvidia-smi: timed out after {node_health.SECTION_TIMEOUT_S}s"
    quiet = check(node_text(gpus=(124, ""), xid=(0, xid_text(xid_line(9000.0, 5, 13)))), first.facts)
    assert not quiet.checked  # an application XID only: still a read failure
    assert quiet.facts["xid"]["count"] == 1 and quiet.facts["xid_other"] == ["GPU 5 XID 13"]  # its cursor moved
    assert quiet.facts["read_error"] == f"nvidia-smi: timed out after {node_health.SECTION_TIMEOUT_S}s"
    t = HealthTracker(multi=True)
    assert [t.update(r) for r in (hung, check(node_text(gpus=(124, ""), xid=(0, xid_text(
        xid_line(9000.0, 5, 79), xid_line(9100.0, 5, 79)))), hung.facts))] == ["bad_once", "lost"]


def test_a_sick_gpu_with_the_ib_tools_unread_is_lost_for_a_multi_node_job():
    blind = dict(ibstat=(127, ""), ibv_devinfo=(127, ""))
    first = check(node_text(**blind))
    one = check(node_text(xid=(0, xid_text(xid_line(9000.0, 5, 79))), **blind), first.facts)
    two = check(node_text(xid=(0, xid_text(xid_line(9000.0, 5, 79), xid_line(9100.0, 5, 79))), **blind), one.facts)
    assert one.ok_single is False and one.ok_multi is False  # not None: the GPUs failed already
    t = HealthTracker(multi=True)
    assert [t.update(r) for r in (first, one, two)] == ["read_failed", "bad_once", "lost"]


def test_the_check_is_bounded_on_the_node_and_prints_what_it_read(tmp_path):
    import time  # noqa: PLC0415

    t0 = time.time()
    out = subprocess.run(["bash", "-c", node_health.bounded("echo partial; sleep 30; echo never", 1)],
                         capture_output=True, text=True, timeout=20).stdout
    assert time.time() - t0 < 10
    assert out.startswith(f"{node_health.TIMEOUT_MARK} 1\n") and "partial" in out and "never" not in out
    pid_file = tmp_path / "pid"
    subprocess.run(["bash", "-c", node_health.bounded(f"echo $$ > {pid_file}; sleep 30", 1)],
                   capture_output=True, text=True, timeout=20)
    gone = subprocess.run(["bash", "-c", f"kill -0 {pid_file.read_text().strip()}"], capture_output=True)
    assert gone.returncode != 0  # the check's shell was killed, not left behind
    out = subprocess.run(["bash", "-c", "echo state-line; " + node_health.bounded("echo fast", 5)],
                         capture_output=True, text=True, timeout=20).stdout
    assert out == "state-line\nfast\n"
    script = node_health.CudaChecker().script([])
    assert script.startswith("__yeto_hs=") and node_health.TIMEOUT_MARK in script  # the cuda check is bounded
    assert script.index("@@yeto-section xid") < script.index("@@yeto-section gpus")  # XIDs before nvidia-smi
    assert script == node_health.bounded(node_health.TOOL_WRAPPER + probe_script(node_health.CUDA_SECTIONS))
    body = probe_script(node_health.CUDA_SECTIONS)
    for name, cmd in node_health._SECTIONS:  # every section under timeout -k 5 (the preflight's: none)
        assert f"( __yeto_t {node_health.SECTION_TIMEOUT_S} bash -c {shlex.quote(cmd)}\n)" in body, name
    assert "__yeto_t" not in probe_script(cp.HOST_SECTIONS_BASE + cp.CONTAINER_SECTIONS)
    assert "set -o pipefail; nvidia-smi -q" in dict(node_health._SECTIONS)["fabric"]
    with_tool = subprocess.run(["bash", "-c", node_health.TOOL_WRAPPER + "__yeto_t 1 sleep 5; echo rc=$?"],
                               capture_output=True, text=True, timeout=20).stdout
    assert with_tool.strip() in ("rc=124", "rc=0")  # rc=0: a host without timeout(1) runs the tool as is
    stubborn = subprocess.run(["bash", "-c", node_health.TOOL_WRAPPER +
                               "__yeto_t 1 bash -c \"trap '' TERM; sleep 30\"; echo rc=$?"],
                              capture_output=True, text=True, timeout=20).stdout
    assert stubborn.strip() in ("rc=137", "rc=124", "rc=0")  # a section deaf to TERM: killed 5 s later


def test_two_bad_checks_in_a_row_make_a_node_lost_and_a_read_failure_counts_neither_way():
    ok, bad = check(node_text()), check(node_text(fabric=(0, FABRIC_IN_PROGRESS)))
    failed = check(node_text(gpus=(124, "")))
    t = HealthTracker(multi=False)
    assert [t.update(x) for x in (ok, bad, ok, bad, bad, bad, ok)] == \
        ["ok", "bad_once", "ok", "bad_once", "lost", "lost", "ok"]
    t = HealthTracker(multi=False)
    assert [t.update(x) for x in (bad, failed, bad)] == ["bad_once", "read_failed", "lost"]
    assert [HealthTracker(multi=False).update(failed) for _ in range(3)] == ["read_failed"] * 3
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


def test_a_reset_keeps_the_baselines_and_clears_the_streak():
    w = HealthWatch("cuda", hcas=hcas_of(HCA), multi=True)
    for text in (node_text(), PORT_DOWN_TEXT(), PORT_DOWN_TEXT()):
        reading = w.observe(text)
    assert reading.lost and w.prev["xid"]["boot"] == BOOT
    prev, before = dict(w.prev), w.record()
    w.reset()
    assert w.last is None and w.tracker.bad_in_row == 0 and w.prev == prev
    assert w.record() == {**before, "reset": True}  # the record stays until the next check
    assert w.observe(PORT_DOWN_TEXT()).verdict == "bad_once"  # a new job: one bad check is not lost
    assert "reset" not in w.record() and w.record()["bad_in_row"] == 1
    assert HealthWatch("cuda").record() is None and (HealthWatch("cuda").reset() or True)


def test_what_a_check_cannot_judge_is_said_once_per_node():
    w = HealthWatch("cuda", hcas=hcas_of(HCA), multi=True)
    blind = node_text(xid=(0, xid_text(source="none")), ibstat=(127, ""), ibv_devinfo=(127, ""))
    first, second = w.observe(blind), w.observe(blind)
    assert first.degraded == ["XID unread (no dmesg / journalctl -k access)", node_health.IB_UNREAD]
    assert second.degraded is None and w.record()["degraded"] == first.degraded
    assert first.verdict == "read_failed" and w.record()["reasons"] == [node_health.IB_UNREAD]
    no_hca = HealthWatch("cuda", multi=True).observe(node_text())
    assert no_hca.degraded == ["no NCCL_IB_HCA: the ports are not judged"]
    assert HealthWatch("cuda", hcas=hcas_of(HCA)).observe(node_text()).degraded is None  # single, all read
    assert HealthWatch("cuda").observe(node_text()).degraded is None  # a single-node job needs no HCA
    assert HealthWatch("cuda").observe(node_text(gpus=(124, ""))).degraded is None  # nothing read


def test_a_record_keeps_twenty_reasons_and_thirty_two_ports_and_ages_out():
    w = HealthWatch("cuda", hcas=hcas_of(",".join(f"mlx5_{i}" for i in range(40))), multi=True,
                    clock=lambda: 1000.0, every_s=60)
    w.observe(node_text(ibstat=(0, ibstat_text(n=40, down=range(40), storage=False))))
    doc = w.record()
    assert len(w.last.result.reasons) == 40 and len(doc["reasons"]) == 20 and len(doc["facts"]["ports"]) == 32
    assert doc["at"] == 1000.0 and doc["every_s"] == 60
    assert node_health.fresh(doc, 1000.0 + 180) is doc
    stale = node_health.fresh(doc, 1000.0 + 181)
    assert stale["verdict"] == "read_failed" and stale["stale"] and stale["reasons"] == [
        "health check stale: the last one 181s ago"]
    assert node_health.fresh(None, 0) is None and node_health.fresh({"verdict": "lost"}, 1e12)["verdict"] == "lost"
    single = HealthWatch("cuda", hcas=hcas_of(",".join(f"mlx5_{i}" for i in range(40))))
    single.observe(node_text(ibstat=(0, ibstat_text(n=40, down=range(40), storage=False))))
    assert single.record()["reasons"] == [] and len(single.record()["notes"]) == 20


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


# ------------------------------------------------------------------ baselines and cursors
def test_an_ecc_count_that_went_down_is_the_new_baseline():
    first = check(node_text(gpus=(0, gpus_text(ecc={2: 5}))))
    lower = check(node_text(gpus=(0, gpus_text(ecc={2: 2}))), first.facts)
    assert lower.ok_single and lower.facts["ecc"]["0000:3a:00"] == 2  # a reset counter: no error
    up = check(node_text(gpus=(0, gpus_text(ecc={2: 3}))), lower.facts)
    assert up.reasons == ["GPU 2 uncorrectable ECC +1"]


def test_memory_and_ecc_baselines_follow_the_pci_address_not_the_index():
    def two(*rows):
        return "\n".join(f"{i}, 00000000:{bus}:00.0, NVIDIA B300 SXM6 AC, {mib}, {ecc}" for i, (bus, mib, ecc)
                         in enumerate(rows))
    first = check(node_text(gpus=(0, two(("18", 100000, 0), ("2A", 275040, 4)))), expect_gpus=2)
    swapped = check(node_text(gpus=(0, two(("2A", 275040, 4), ("18", 100000, 0)))), first.facts, expect_gpus=2)
    assert swapped.ok_single and swapped.reasons == []  # the same GPUs, other indexes
    shrunk = check(node_text(gpus=(0, two(("2A", 275040, 4), ("18", 90000, 0)))), swapped.facts, expect_gpus=2)
    assert shrunk.reasons == ["GPU 1 memory 90000 MiB < 100000 MiB"]


def test_a_node_without_nvswitch_passes_with_its_fabric_manager_inactive_and_no_fabric():
    pcie = dict(fabricmanager=(3, "inactive"), fabric=(1, ""))
    first = check(node_text(**pcie))
    second = check(node_text(**pcie), first.facts)
    assert first.ok_single and second.ok_single and second.facts["fm"] == "inactive"
    assert second.facts["fabric"] == "none"
    na = "    Fabric\n        State : N/A\n        Status : N/A\n" * 8  # a PCIe card: no NVLink fabric
    r = check(node_text(fabric=(0, na), fabricmanager=(3, "inactive")), first.facts)
    assert r.ok_single and r.facts["fabric"] == "none"
    stopped = check(node_text(fabricmanager=(3, "inactive")), check(node_text()).facts)
    assert stopped.reasons == ["nvidia-fabricmanager inactive (was active)"]  # one that ran and stopped
    # nvidia-smi -q cut off by its section's timeout: what it printed is not the fabric's state
    hung = check(node_text(fabric=(124, FABRIC_IN_PROGRESS)), first.facts)
    assert hung.ok_single and hung.reasons == []


def _plain_xid(gpu: int, code: int) -> str:  # a kernel log without timestamps (dmesg -t, a syslog copy)
    return f"NVRM: Xid (PCI:0000:{BUSES[gpu]}:00): {code}, pid=4242, name=python3, GPU has fallen off the bus."


def test_without_timestamps_the_xid_cursor_is_the_last_line_and_its_count():
    a, b = _plain_xid(5, 79), _plain_xid(3, 79)
    first = check(node_text(xid=(0, xid_text(a))))
    assert first.ok_single and first.facts["xid"]["last"] == a and first.facts["xid"]["last_n"] == 1
    again = check(node_text(xid=(0, xid_text(a, a))), first.facts)
    assert again.reasons == ["GPU 5 XID 79"]  # the same text once more: a new XID
    same = check(node_text(xid=(0, xid_text(a, a))), again.facts)
    assert same.ok_single
    # the log's head rotated out: fewer copies of the cursor's line than it counted, none new
    assert check(node_text(xid=(0, xid_text(a))), same.facts).ok_single
    other = check(node_text(xid=(0, xid_text(a, a, b))), same.facts)
    assert other.reasons == ["GPU 3 XID 79"]
    head_gone = check(node_text(xid=(0, xid_text(a, b))), other.facts)  # the log's head rotated out
    assert head_gone.ok_single
    assert f"tail -c {node_health.XID_TAIL_BYTES}" in node_health.XID_SCRIPT and "tail -n" not in node_health.XID_SCRIPT
    assert node_health.XID_TAIL_BYTES == 1 << 20


def test_ibv_devinfo_runs_only_when_ibstat_fails(tmp_path):
    cmd = dict(node_health._SECTIONS)["ibv_devinfo"]
    for name, rc in (("ibstat", 1), ("ibv_devinfo", 0)):
        (tmp_path / name).write_text(f"#!/bin/sh\necho {name}-ran\nexit {rc}\n")
        (tmp_path / name).chmod(0o755)
    env = {"PATH": f"{tmp_path}:/usr/bin:/bin"}
    out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, env=env, timeout=20)
    assert out.returncode == 0 and out.stdout == "ibv_devinfo-ran\n"  # ibstat failed: the fallback reads
    (tmp_path / "ibstat").write_text("#!/bin/sh\nexit 0\n")
    out = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, env=env, timeout=20)
    assert out.returncode == 3 and out.stdout == ""  # ibstat read: no second tool


# ------------------------------------------------------------------ events
def test_the_lost_event_is_written_once_per_streak_a_read_failure_inside_does_not_restart_it():
    w = HealthWatch("cuda", hcas=hcas_of(HCA), multi=True)
    texts = [node_text(), PORT_DOWN_TEXT(), PORT_DOWN_TEXT(), node_text(gpus=(124, "")), PORT_DOWN_TEXT(),
             node_text(), PORT_DOWN_TEXT(), PORT_DOWN_TEXT()]
    readings = [w.observe(t) for t in texts]
    assert [r.verdict for r in readings] == ["ok", "bad_once", "lost", "read_failed", "lost", "ok", "bad_once",
                                             "lost"]
    assert [r.event for r in readings] == [None, None, node_health.EVENT_LOST, node_health.EVENT_READ_FAILED,
                                           None, None, None, node_health.EVENT_LOST]


def test_read_failures_in_a_row_write_a_degraded_event_once_per_streak_and_lose_nothing():
    w = HealthWatch("cuda", hcas=hcas_of(HCA), multi=True)
    w.observe(node_text())
    hung = node_text(gpus=(124, ""))
    readings = [w.observe(hung) for _ in range(node_health.READ_FAILED_DEGRADE + 2)]
    assert all(r.verdict == "read_failed" for r in readings)
    marks = [r.degraded for r in readings]
    k = node_health.READ_FAILED_DEGRADE
    assert marks[:k - 1] == [None] * (k - 1) and marks[k:] == [None, None]
    assert marks[k - 1] == [f"read failed {k} checks in a row: nvidia-smi: timed out after "
                            f"{node_health.SECTION_TIMEOUT_S}s"]
    assert w.record()["read_failed_in_row"] == k + 2 and w.tracker.bad_in_row == 0
    w.observe(node_text())
    assert "read_failed_in_row" not in w.record()
    again = [w.observe(hung) for _ in range(k)]
    assert again[-1].degraded and all(r.degraded is None for r in again[:-1])  # a new streak: said again


def test_the_tracker_takes_the_job_kind_explicitly():
    with pytest.raises(TypeError):
        HealthTracker()  # noqa: the job's kind (multi) is not guessed


def test_an_empty_or_cut_off_output_is_a_read_failure_never_healthy():
    first = check(node_text())
    rc_only = f"{host_probe.RC} 0\n{host_probe.RC} 255\n"
    whole = node_text()
    cut_in_gpus = whole[:whole.index(f"{host_probe.RC}", whole.index(f"{host_probe.SECTION} gpus"))]
    no_ib = whole[:whole.index(f"{host_probe.SECTION} ibstat")]
    for label, text in (("empty", ""), ("rc lines only", rc_only), ("cut in the GPU section", cut_in_gpus),
                        ("cut before the IB sections", no_ib)):
        for prev in (None, first.facts):
            r = check(text, prev)
            assert not r.checked and r.ok_single is None and r.ok_multi is None, (label, prev is None)
            assert HealthTracker(multi=False).update(r) == "read_failed", label
    assert check("").reasons == ["nvidia-smi: no output (the check did not run)"]
    assert check(cut_in_gpus).reasons == ["node health output incomplete (no gpus, fabric, fabricmanager, ibstat, "
                                          "ibv_devinfo)"]
    assert check(no_ib).reasons == ["node health output incomplete (no ibstat, ibv_devinfo)"]
    # a cut-off output still shows a new hardware XID of its completed XID section
    sick = node_text(xid=(0, xid_text(xid_line(9000.0, 5, 79))))
    r = check(sick[:sick.index(f"{host_probe.SECTION} fabric")], first.facts)
    assert r.checked and r.ok_single is False and r.reasons == ["GPU 5 XID 79"]


def test_without_timestamps_a_repeat_is_seen_while_the_log_head_rotates_out():
    """The cursor's anchor (the line before its last run) keeps its place when one old copy
    of the repeated line leaves the head of the log."""
    a, b, c = _plain_xid(5, 79), _plain_xid(3, 13), _plain_xid(4, 79)
    first = check(node_text(xid=(0, xid_text(a, b, a, a))))
    assert first.facts["xid"]["run"] == 2 and first.facts["xid"]["before"] == b
    # the oldest a rotated out, a new a came: as many copies of a as before, one is new
    rotated = check(node_text(xid=(0, xid_text(b, a, a, a))), first.facts)
    assert rotated.reasons == ["GPU 5 XID 79"]
    assert check(node_text(xid=(0, xid_text(b, a, a, a))), rotated.facts).ok_single  # nothing new
    later = check(node_text(xid=(0, xid_text(a, a, a, c))), rotated.facts)  # the anchor rotated out too
    assert later.reasons == ["GPU 4 XID 79"]
    # the limit (documented): a log of one repeated line that rotates one copy out
    only = check(node_text(xid=(0, xid_text(a, a))))
    assert check(node_text(xid=(0, xid_text(a, a))), only.facts).ok_single


OLD = 1_000_000_000  # an old pid file: its check ran past the limit and was not killed


def _start_time(pid: int, env: dict | None = None) -> str:
    """The start time the check records for ``pid`` (node_health.START_TIME_FN)."""
    return subprocess.run(["bash", "-c", node_health.START_TIME_FN + f"__yeto_st {pid}"], capture_output=True,
                          text=True, env=env, timeout=20).stdout.strip()


def _bounded(pid_file) -> str:
    return subprocess.run(["bash", "-c", node_health.bounded("echo ran", 5, pid_file=str(pid_file))],
                          capture_output=True, text=True, timeout=20).stdout


def _pid_file(path, text: str, *, old: bool = True):
    import os  # noqa: PLC0415

    path.write_text(text)
    if old:
        os.utime(path, (OLD, OLD))


def test_a_check_still_stuck_on_the_node_makes_the_next_one_skip(tmp_path):
    pid_file = tmp_path / "health.pid"
    stuck = subprocess.Popen(["sleep", "30"])
    try:
        started = _start_time(stuck.pid)
        assert started
        _pid_file(pid_file, f"{stuck.pid} {started}\n")  # the same process, still there: busy
        out = _bounded(pid_file)
        assert out == f"{node_health.BUSY_MARK} {stuck.pid}\n"
        r = check(out, check(node_text()).facts)
        assert not r.checked and r.reasons == ["node health check skipped: the previous one still runs "
                                               "(a tool stuck on the node)"]
        _pid_file(pid_file, f"{stuck.pid} {started}\n", old=False)  # another caller's check, running now
        assert _bounded(pid_file) == "ran\n"
        # the pid now names another process (reused, or the node rebooted): stale, the check runs
        _pid_file(pid_file, f"{stuck.pid} 12345\n")
        assert _bounded(pid_file) == "ran\n" and not pid_file.exists()
        _pid_file(pid_file, f"{stuck.pid}\n")  # no start time recorded: not proof of the same process
        assert _bounded(pid_file) == "ran\n"
    finally:
        stuck.kill()
        stuck.wait()
    # a running check has its pid and its start time recorded before its first command (read
    # here as that first command)
    seen = subprocess.run(["bash", "-c", node_health.bounded(
        f"cat {pid_file}; " + node_health.START_TIME_FN.replace("\n", "; ") + "echo $$ $(__yeto_st $$)", 5,
        pid_file=str(pid_file))], capture_output=True, text=True, timeout=20).stdout.split()
    assert len(seen) == 4 and seen[:2] == seen[2:] and seen[1], seen  # the check's own pid and start time
    assert not list(tmp_path.glob("health.pid.*"))  # no temporary file left
    # a check that ended removes its pid; one that is gone leaves nothing to skip for
    assert _bounded(pid_file) == "ran\n" and not pid_file.exists()
    _pid_file(pid_file, f"{stuck.pid} {started}\n")
    assert _bounded(pid_file) == "ran\n" and not pid_file.exists()


def test_the_start_time_is_field_22_of_proc_stat_and_ps_without_proc(tmp_path):
    import os  # noqa: PLC0415

    proc = tmp_path / "proc" / "4242"
    proc.mkdir(parents=True)
    # a command name with spaces and a parenthesis: the fields count from after its last ")"
    fields = ["S", "1", "4242", "4242", "0", "-1", "4194560", "0", "0", "0", "0", "0", "0", "0", "0", "20", "0",
              "1", "0", "987654", "1000", "100"]
    (proc / "stat").write_text("4242 (my (odd) tool) " + " ".join(fields) + "\n")
    env = {**os.environ, "__YETO_PROC": str(tmp_path / "proc")}
    assert _start_time(4242, env) == "987654"
    # without /proc (another OS): ps's start time, the same for the same process
    env = {**os.environ, "__YETO_PROC": str(tmp_path / "none")}
    me = _start_time(os.getpid(), env)
    assert me and me == _start_time(os.getpid(), env) and " " not in me
