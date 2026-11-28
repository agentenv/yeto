"""Node health while a run goes on (``yeto.hw.node_health``): the cuda checker over the
recorded tool outputs of ``yeto preflight``'s tests plus made-up faults (XID 79, ECC
growth, an unfinished NVLink fabric, ``mlx5_3`` down), the verdict of two bad checks in a
row, and the null checker of every other device family. The IB / fabric parsing is the
preflight's own (one implementation, ``yeto.hw.host_probe``)."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import time
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


def gpus_text(n=8, mib=275040, ecc=None, gone=(), name="NVIDIA B300 SXM6 AC") -> str:
    ecc = ecc or {}
    rows = []
    for i in range(n):
        if i in gone:
            rows.append(f"Unable to determine the device handle for GPU0000:{BUSES[i].upper()}:00.0: Unknown Error")
            continue
        rows.append(f"{i}, 00000000:{BUSES[i].upper()}:00.0, {name}, {mib}, {ecc.get(i, 0)}")
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
    stopped = check(node_text(fabricmanager=(3, fm_out("failed"))), check(node_text()).facts)
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
    pcie = dict(fabricmanager=(3, fm_out("inactive")), fabric=(1, ""))
    first = check(node_text(**pcie))
    second = check(node_text(**pcie), first.facts)
    assert first.ok_single and second.ok_single and second.facts["fm"] == "inactive"
    assert second.facts["fabric"] == "none"
    na = "    Fabric\n        State : N/A\n        Status : N/A\n" * 8  # a PCIe card: no NVLink fabric
    r = check(node_text(fabric=(0, na), fabricmanager=(3, fm_out("inactive"))), first.facts)
    assert r.ok_single and r.facts["fabric"] == "none"
    stopped = check(node_text(fabricmanager=(3, fm_out("inactive"))), check(node_text()).facts)
    assert stopped.reasons == ["nvidia-fabricmanager inactive (was active)"]  # one that ran and stopped
    # nvidia-smi -q cut off by its section's timeout: what it printed is not the fabric's state
    hung = check(node_text(fabric=(124, FABRIC_IN_PROGRESS)), first.facts)
    assert hung.ok_single and hung.reasons == []


# -- the fabric manager over a boot (a fabric manager that stays down reaches lost) ----------

# What the fabricmanager section shows (FM_SCRIPT). The default is an HGX B300 node as the
# ssh user (not root) sees it: its NVSwitches sit behind a ConnectX bridge, so no procfs
# entry, no device file and no NVIDIA PCI bridge; ten Mellanox devices whose VPD does not
# read; GPUs "NVIDIA B300 SXM6 AC" with a Completed fabric.
B300_ROOT = dict(vpd=10, sw=1)  # as root: the VPD of the switch management port names SW_MNG
B300_ROOT_NO_SW = dict(vpd=10, sw=0)  # every VPD read, none names it: the GPUs still say NVSwitch
VPD_NO_SW = dict(name="NVIDIA H100 80GB HBM3", vpd=10, sw=0)  # a fabric, every VPD read without the port
UNIT_GONE = dict(load="not-found", fm_proc=0)  # no unit, no nv-fabricmanager process (a shared NVSwitch VM)
UNIT_IN_CONTAINER = dict(load="not-found", fm_proc=1)  # no unit, its process from a driver container
HIDEPID = dict(load="not-found", proc1=0)  # no unit, the processes not seen
HOPPER = dict(name="NVIDIA H100 80GB HBM3", proc=4, pci=4)  # Hopper HGX: NVSwitch on the host's PCIe
HIDDEN = dict(name="NVIDIA H100 80GB HBM3")  # a fabric, no NVSwitch shown, VPD unread: not determined
TRAY = dict(name="NVIDIA GB300")  # an NVL72-class compute tray: a Completed fabric, switches elsewhere
PCIE = dict(name="NVIDIA L40S", mlx=2, fabric=(1, ""))  # no Fabric block in nvidia-smi -q
PCIE_NA = dict(name="NVIDIA H100 PCIe", mlx=2,
               fabric=(0, "    Fabric\n        State : N/A\n        Status : N/A\n" * 8))
NO_PROBE = HIDDEN


ENTERED = 41_000_000  # the unit's ActiveEnterTimestampMonotonic: it was active in this boot


def fm_out(state: str | None, *, load="loaded", entered=ENTERED, fm_proc=None, fm_ids=None, proc1=1, container=0,
           proc=None, dev=0, pci=0, bridge=None, mlx=10, vpd=0, sw=0, restarts=0) -> str:
    """The fabricmanager section's output; ``load`` the unit's LoadState, ``entered`` its
    ActiveEnterTimestampMonotonic (0: not active in this boot; None: not printed; a unit
    not found has 0); ``fm_proc`` the nv-fabricmanager processes (None: not counted),
    ``fm_ids`` their pid:starttime; ``proc1`` 0: the processes not seen (hidepid); ``proc`` /
    ``pci`` / ``vpd`` None: no procfs dir / no lspci / no timeout (the VPD not read);
    ``bridge``: another NVIDIA class 0680 bridge (None: not printed)."""
    entered = 0 if load == "not-found" and entered == ENTERED else entered
    lines = ([state] if state else []) + [f"loadstate={load}"]
    lines += [] if entered is None else [f"active_enter={entered}"]
    lines += [f"container={container}", f"proc1={proc1}"]
    if fm_proc is not None:
        ids = fm_ids if fm_ids is not None else " ".join(f"{4242 + n}:777" for n in range(fm_proc))
        lines += [f"fm_proc={fm_proc}", f"fm_ids={ids}"]
    lines += [f"nrestarts={'' if restarts is None else restarts}"]
    lines += ([] if proc is None else [f"nvswitch_proc={proc}"]) + [f"nvswitch_dev={dev}"]
    lines += ([] if pci is None else [f"nvswitch_pci={pci}"]) + ([] if bridge is None else [f"nvidia_bridge={bridge}"])
    lines += [f"mlx={mlx}"]
    lines += [] if vpd is None else [f"mlx_vpd={vpd}", f"sw_mng={sw}"]
    return "\n".join(lines)


def fm_text(state: str | None, boot=BOOT, *, name="NVIDIA B300 SXM6 AC", restarts=0, **over) -> str:
    """A healthy 8-GPU node (default: HGX B300, as above) whose fabric manager reads
    ``state`` in ``boot`` (None: the section timed out, nothing printed)."""
    keys = ("load", "entered", "fm_proc", "fm_ids", "proc1", "container", "proc", "dev", "pci", "bridge", "mlx",
            "vpd", "sw")
    probe = {k: over.pop(k) for k in keys if k in over}
    fm = (124, "") if state is None else (0 if state == "active" else 3, fm_out(state, restarts=restarts, **probe))
    over.setdefault("gpus", (0, gpus_text(name=name)))
    return node_text(fabricmanager=fm, xid=(0, xid_text(boot=boot)), **over)


def watch_through(*texts, multi=False):
    w = HealthWatch("cuda", hcas=hcas_of(HCA), multi=multi)
    return w, [w.observe(t) for t in texts]


def test_a_fabric_manager_that_stays_failed_after_it_was_active_is_lost():
    w, rs = watch_through(fm_text("active"), fm_text("failed"), fm_text("failed"), fm_text("failed"))
    assert [r.verdict for r in rs] == ["ok", "bad_once", "lost", "lost"]
    assert [r.event for r in rs] == [None, None, "node_health", None]  # the loss is said once
    assert all(r.result.reasons == ["nvidia-fabricmanager failed (was active)"] for r in rs[1:])
    assert all(r.result.facts["fabric"] == "8/8 completed" for r in rs)  # the fabric still reads Completed
    assert rs[-1].result.facts["fm_boot"] == BOOT and "fm_why" not in rs[-1].result.facts
    assert w.record()["reasons"] == ["nvidia-fabricmanager failed (was active)"]


@pytest.mark.parametrize("state", ["inactive", "activating", "deactivating"])
def test_any_state_but_active_after_active_in_the_boot_stays_a_reason(state):
    _, rs = watch_through(fm_text("active"), fm_text(state), fm_text(state))
    assert [r.verdict for r in rs] == ["ok", "bad_once", "lost"]
    assert rs[2].result.reasons == [f"nvidia-fabricmanager {state} (was active)"]


def test_an_active_fabric_manager_clears_the_streak_and_an_unread_one_judges_nothing():
    _, rs = watch_through(fm_text("active"), fm_text("failed"), fm_text("active"), fm_text("failed"),
                          fm_text(None), fm_text("failed"))
    # systemctl that timed out: that check passes (the streak goes), the memory stays
    assert [r.verdict for r in rs] == ["ok", "bad_once", "ok", "bad_once", "ok", "bad_once"]
    assert rs[4].result.reasons == [] and rs[4].result.facts["fm_boot"] == BOOT
    assert rs[5].result.reasons == ["nvidia-fabricmanager failed (was active)"]


def test_a_read_failure_keeps_the_fabric_manager_memory():
    _, rs = watch_through(fm_text("active"), fm_text("failed"), node_text(gpus=(124, "")), fm_text("failed"))
    assert [r.verdict for r in rs] == ["ok", "bad_once", "read_failed", "lost"]
    assert rs[2].result.facts["fm_boot"] == BOOT


@pytest.mark.parametrize("node", [PCIE, PCIE_NA, TRAY, VPD_NO_SW], ids=["pcie", "pcie-na", "gb-tray", "vpd-no-sw"])
@pytest.mark.parametrize("state", ["inactive", "failed", "activating"])
def test_a_node_without_nvswitch_of_its_own_is_never_judged_on_its_fabric_manager(node, state):
    _, rs = watch_through(*[fm_text(state, restarts=n, **node) for n in range(4)])
    assert [r.verdict for r in rs] == ["ok"] * 4 and all(r.result.reasons == [] for r in rs)
    assert all("fm_boot" not in r.result.facts and r.result.facts["nvswitch"] == "no" for r in rs)
    # REV-905 M1: not even after it was active in the boot (a GB tray's unit may run, then fail)
    _, rs = watch_through(fm_text("active", **node), fm_text(state, **node), fm_text(state, restarts=3, **node))
    assert [r.verdict for r in rs] == ["ok"] * 3


def test_whether_the_node_has_nvswitch_of_its_own():
    of = lambda **kw: (lambda f: (f["nvswitch"], f["nvswitch_by"]))(check(fm_text("active", **kw)).facts)  # noqa: E731
    assert of() == ("yes", "B200/B300 GPU with an NVLink fabric")  # HGX B300, not root
    assert of(name="NVIDIA B200") == ("yes", "B200/B300 GPU with an NVLink fabric")
    assert of(**B300_ROOT) == ("yes", "ConnectX VPD SW_MNG")
    # REV-905c M: a VPD without the port does not overrule B200 / B300 GPUs with a fabric
    assert of(**B300_ROOT_NO_SW) == ("yes", "B200/B300 GPU with an NVLink fabric")
    assert of(**VPD_NO_SW) == ("no", "ConnectX VPD without SW_MNG")
    assert of(name="NVIDIA H100 80GB HBM3", vpd=4, sw=0) == ("unread", "NVSwitch not determined")  # some unread
    assert of(name="NVIDIA H100 80GB HBM3", vpd=None) == ("unread", "NVSwitch not determined")  # no timeout
    assert of(name="NVIDIA H100 80GB HBM3", dev=1) == ("yes", "device files")  # a single NVSwitch
    assert of(name="NVIDIA B40") == ("unread", "NVSwitch not determined")  # B-like, not B200 / B300
    assert of(**HOPPER) == ("yes", "procfs")
    assert of(name="NVIDIA H100 80GB HBM3", dev=8) == ("yes", "device files")
    assert of(name="NVIDIA H100 80GB HBM3", pci=4) == ("yes", "lspci")
    assert of(**HIDDEN) == ("unread", "NVSwitch not determined")
    assert of(**TRAY) == ("no", "compute tray (GB GPU)")
    assert of(**TRAY, **B300_ROOT) == ("no", "compute tray (GB GPU)")
    assert of(**PCIE) == ("no", "no NVLink fabric") and of(**PCIE_NA) == ("no", "no NVLink fabric")
    assert of(sw=1, **PCIE) == ("no", "no NVLink fabric")  # no fabric first: a PCIe node's ConnectX VPD
    assert of(name="NVIDIA B300 PCIe") == ("unread", "NVSwitch not determined")
    # the fabric unread (nvidia-smi -q timed out or failed): by the GPU names no more
    assert of(fabric=(124, FABRIC_IN_PROGRESS)) == ("unread", "NVSwitch not determined")
    assert of(fabric=(9, "")) == ("unread", "NVSwitch not determined")
    assert of(fabric=(124, ""), **B300_ROOT) == ("yes", "ConnectX VPD SW_MNG")
    assert of(fabric=(124, ""), **TRAY) == ("no", "compute tray (GB GPU)")
    # mixed GPUs: all must be B200 / B300
    mixed = gpus_text().replace("NVIDIA B300 SXM6 AC", "NVIDIA H100 80GB HBM3", 1)
    assert check(fm_text("active", gpus=(0, mixed))).facts["nvswitch"] == "unread"
    assert node_health.nvswitch_of({}, [], True) == ("unread", "NVSwitch not determined")
    # REV-905e: another NVIDIA class 0680 bridge is soft evidence (yes, not hard); an NVSwitch id is hard
    h100 = dict(name="NVIDIA H100 80GB HBM3")
    assert of(bridge=1, **h100) == ("yes", "lspci NVIDIA bridge (no known NVSwitch id)")
    assert check(fm_text("active", bridge=1, **h100)).facts["nvswitch_hard"] is False
    assert check(fm_text("active", pci=1, **h100)).facts["nvswitch_hard"] is True
    assert of(bridge=1, **PCIE) == ("no", "no NVLink fabric") and of(bridge=1, **TRAY)[0] == "no"
    assert of(bridge=1, vpd=10, sw=0, **h100)[0] == "yes"  # before the VPD rule
    assert of(bridge=0, **h100) == ("unread", "NVSwitch not determined")


def test_a_new_boot_forgets_the_fabric_manager_as_before():
    # active in the old boot, never in the new one: one reason when it stops (as before), then none
    _, rs = watch_through(fm_text("active"), fm_text("failed", boot="boot-2"), fm_text("failed", boot="boot-2"),
                          fm_text("failed", boot="boot-2"))
    assert [r.verdict for r in rs] == ["ok", "bad_once", "ok", "ok"]
    assert "fm_boot" not in rs[1].result.facts and "fm_boot" not in rs[3].result.facts
    # failed before the reboot and after it, never active since: nothing, as before
    _, rs = watch_through(fm_text("active"), fm_text("failed"), fm_text("failed", boot="boot-2"),
                          fm_text("failed", boot="boot-2"))
    assert [r.verdict for r in rs] == ["ok", "bad_once", "ok", "ok"]
    # NVSwitch not determined: not even the stop across the reboot
    _, rs = watch_through(fm_text("active", **HIDDEN), fm_text("failed", boot="boot-2", **HIDDEN))
    assert [r.verdict for r in rs] == ["ok", "ok"]
    # active again in the new boot: that boot's memory
    _, rs = watch_through(fm_text("active"), fm_text("active", boot="boot-2"), fm_text("failed", boot="boot-2"),
                          fm_text("failed", boot="boot-2"))
    assert [r.verdict for r in rs] == ["ok", "ok", "bad_once", "lost"]
    assert rs[3].result.facts["fm_boot"] == "boot-2"


@pytest.mark.parametrize("node", [{}, B300_ROOT, HOPPER], ids=["b300", "b300-root", "hopper"])
@pytest.mark.parametrize("state", ["failed", "inactive", "activating", "deactivating"])
def test_a_new_launcher_reports_a_down_fabric_manager_on_an_nvswitch_node_at_once(state, node):
    _, rs = watch_through(fm_text(state, **node), fm_text(state, **node), fm_text(state, **node))
    assert [r.verdict for r in rs] == ["bad_once", "lost", "lost"]
    assert all(r.result.reasons == [f"nvidia-fabricmanager {state} (NVSwitch present)"] for r in rs)
    assert rs[0].result.facts["fm_boot"] == BOOT and rs[0].result.facts["fm_why"] == "nvswitch"
    # it comes back: the reason goes; down again in the boot: the usual reason
    back = check(fm_text("active", **node), rs[-1].result.facts)
    assert back.ok_single and "fm_why" not in back.facts
    assert check(fm_text("failed", **node), back.facts).reasons == ["nvidia-fabricmanager failed (was active)"]
    # a new boot forgets it too
    assert check(fm_text(state, boot="boot-2", **node), rs[-1].result.facts).ok_single


def test_a_first_read_that_cannot_judge_is_not_the_first_judged_one():
    # REV-905 M1: the first check cannot judge (the fabric timed out: NVSwitch not determined),
    # the next three read: they are judged as the first read of a new launcher
    unread = fm_text("failed", fabric=(124, FABRIC_IN_PROGRESS))
    _, rs = watch_through(unread, fm_text("failed"), fm_text("failed"), fm_text("failed"))
    assert [r.verdict for r in rs] == ["ok", "bad_once", "lost", "lost"]
    assert rs[0].result.facts["nvswitch"] == "unread" and rs[0].result.facts["fm_judged"] is False
    # the fabric manager section timed out first: the same
    _, rs = watch_through(fm_text(None), fm_text("failed"), fm_text("failed"))
    assert [r.verdict for r in rs] == ["ok", "bad_once", "lost"]
    # a whole read failure first: the same
    _, rs = watch_through(node_text(gpus=(124, "")), fm_text("failed"), fm_text("failed"))
    assert [r.verdict for r in rs] == ["read_failed", "bad_once", "lost"]
    # an unjudged read, then a read failure (it carries "not judged yet"), then a judged one
    _, rs = watch_through(fm_text("failed", **NO_PROBE), node_text(gpus=(124, "")), fm_text("failed"))
    assert [r.verdict for r in rs] == ["ok", "read_failed", "bad_once"]
    assert rs[1].result.facts["fm_judged"] is False
    # NVSwitch never determined: never judged bad on a fabric manager never active
    _, rs = watch_through(*[fm_text("failed", **NO_PROBE)] * 3)
    assert [r.verdict for r in rs] == ["ok"] * 3
    # once active in the boot, the memory judges it while NVSwitch is not determined
    _, rs = watch_through(fm_text("active", **NO_PROBE), fm_text("failed", **NO_PROBE), fm_text("failed", **NO_PROBE))
    assert [r.verdict for r in rs] == ["ok", "bad_once", "lost"]


def test_inactive_at_the_first_read_follows_nvswitch_and_the_fabric():
    _, rs = watch_through(*[fm_text("inactive")] * 2)  # NVSwitch: the unit must run
    assert [r.verdict for r in rs] == ["bad_once", "lost"]
    _, rs = watch_through(*[fm_text("inactive", **TRAY)] * 3)  # no NVSwitch, a Completed fabric: nothing
    assert [r.verdict for r in rs] == ["ok"] * 3
    stuck = fm_text("inactive", fabric=(0, FABRIC_IN_PROGRESS), **TRAY)  # a fabric not completed: its own reason
    _, rs = watch_through(stuck, stuck)
    assert [r.verdict for r in rs] == ["bad_once", "lost"]
    assert all(r.result.reasons == ["NVLink fabric In Progress/N/A on 8/8 GPUs (fabric manager: inactive)"]
               for r in rs)


def test_a_fabric_manager_that_restarts_is_bad_only_while_it_goes_on_restarting():
    # REV-905e: restarts that came back active are no reason: a node_health_fm_restarted event
    # at every check of the streak (its count), never node_health_degraded
    w, rs = watch_through(*[fm_text("active", restarts=n) for n in (0, 1, 2)])
    assert [r.verdict for r in rs] == ["ok"] * 3 and all(r.result.reasons == [] for r in rs)
    assert all(r.degraded is None for r in rs) and "degraded" not in w.record()
    assert rs[0].fm_restarted is None and rs[0].fm_restarted_fields() == {}
    assert [r.fm_restarted_fields() for r in rs[1:]] == [
        {"streak": n, "nrestarts": n, "reasons": ["nvidia-fabricmanager restarted 1x since the previous check"]}
        for n in (1, 2)]
    assert w.record()["fm_restarted"] == rs[2].fm_restarted_fields()
    assert rs[2].result.facts["fm_restart_streak"] == {"boot": BOOT, "n": 2}
    assert "fm_restarted" not in w.observe(fm_text("active", restarts=2)).result.facts
    assert "fm_restarted" not in w.record()  # the record holds the last check's restarts only
    # FM_RESTART_STREAK (3) checks in a row restarting: a reason, and lost at the next one
    _, rs = watch_through(*[fm_text("active", restarts=n) for n in (0, 1, 2, 3, 4)])
    assert [r.verdict for r in rs] == ["ok", "ok", "ok", "bad_once", "lost"]
    assert rs[3].result.reasons == ["nvidia-fabricmanager restarted 1x since the previous check"]
    assert [r.fm_restarted["streak"] for r in rs[1:]] == [1, 2, 3, 4]  # the event goes on with it
    assert rs[4].result.facts["fm_restarts"] == {"boot": BOOT, "n": 4}
    # a check without a restart ends the streak
    _, rs = watch_through(*[fm_text("active", restarts=n) for n in (0, 1, 2, 2, 3, 4)])
    assert [r.verdict for r in rs] == ["ok"] * 6
    assert rs[3].result.facts["fm_restart_streak"]["n"] == 0 and rs[3].fm_restarted is None
    assert rs[5].fm_restarted["streak"] == 2
    # down while restarting: bad by its state at once (the restart itself not counted yet)
    _, rs = watch_through(fm_text("active", restarts=0), fm_text("activating", restarts=1))
    assert [r.verdict for r in rs] == ["ok", "bad_once"]
    assert rs[1].result.reasons == ["nvidia-fabricmanager activating (was active)"]
    # NRestarts unread: the baseline and the streak stay; a new boot: new ones
    _, rs = watch_through(*[fm_text("active", restarts=n) for n in (0, 1, 2)], node_text(gpus=(124, "")),
                          fm_text("active", restarts=3))
    assert [r.verdict for r in rs] == ["ok", "ok", "ok", "read_failed", "bad_once"]  # a read failure keeps it
    assert rs[3].fm_restarted is None and rs[4].fm_restarted["streak"] == 3
    # NRestarts unread at a check: no growth there, the streak is zeroed (the baseline stays)
    _, rs = watch_through(*[fm_text("active", restarts=n) for n in (0, 1, 2, None, 3, 4)])
    assert [r.verdict for r in rs] == ["ok"] * 6
    assert rs[3].result.facts["fm_restart_streak"]["n"] == 0 and rs[4].fm_restarted["streak"] == 1
    _, rs = watch_through(*[fm_text("active", restarts=n) for n in (0, 1, 2)],
                          fm_text("active", boot="boot-2", restarts=0), fm_text("active", boot="boot-2", restarts=1))
    assert [r.verdict for r in rs] == ["ok"] * 5 and rs[4].result.facts["fm_restart_streak"]["n"] == 1
    _, rs = watch_through(fm_text("active", restarts=1), fm_text("active", boot="boot-2", restarts=3))
    assert [r.verdict for r in rs] == ["ok", "ok"]  # more than the old boot's count: still a new baseline
    assert rs[1].fm_restarted is None and rs[1].result.facts["fm_restarts"] == {"boot": "boot-2", "n": 3}
    # NVSwitch not determined, never active: nothing; once active, the memory counts the restarts
    _, rs = watch_through(*[fm_text("activating", restarts=n, **HIDDEN) for n in range(4)])
    assert [r.verdict for r in rs] == ["ok"] * 4 and all(r.degraded is None for r in rs)
    _, rs = watch_through(*[fm_text("active", restarts=n, **HIDDEN) for n in range(4)])
    assert [r.verdict for r in rs] == ["ok", "ok", "ok", "bad_once"]


def test_the_fabric_manager_absolute_rule_judges_one_check_alone():
    absolute = node_health.fabric_manager_absolute
    assert absolute(check(fm_text("active")).facts) == ("ok", "nvidia-fabricmanager active")
    assert absolute(check(fm_text("failed")).facts) == ("bad", "nvidia-fabricmanager failed (NVSwitch present)")
    assert absolute(check(fm_text("inactive", **B300_ROOT)).facts)[0] == "bad"
    assert absolute(check(fm_text("failed", **TRAY)).facts) == ("ok", "no NVSwitch on the node (compute tray (GB GPU))")
    assert absolute(check(fm_text("failed", **PCIE)).facts) == ("ok", "no NVSwitch on the node (no NVLink fabric)")
    # REV-905 H1: the HGX B300 sample (no NVSwitch device on the host) is never ok when down
    b300 = check(fm_text("failed", fabric=(124, FABRIC_OK)))
    assert absolute(b300.facts) == ("unread", "NVSwitch not determined")
    assert absolute(check(fm_text("failed", **HIDDEN)).facts) == ("unread", "NVSwitch not determined")
    assert absolute(check(fm_text(None)).facts) == ("unread", "nvidia-fabricmanager state unread")
    assert absolute(check(fm_text(None, **HIDDEN)).facts) == ("unread", "FM section timed out")
    # REV-905c L2: cut off after the state line (the VPD reads): not "state unread"
    cut = node_text(fabricmanager=(124, "failed\nloadstate=loaded\nnrestarts=0\nnvswitch_dev=0"),
                    gpus=(0, gpus_text(name="NVIDIA H100 80GB HBM3")))
    assert absolute(check(cut).facts) == ("unread", "FM section timed out after reading state")
    assert check(cut).facts["fm_state"] == "failed"
    assert absolute(check(node_text(gpus=(124, ""))).facts)[0] == "unread"  # nvidia-smi did not answer
    assert absolute({})[0] == "unread" and absolute({"nvswitch": 4, "fm_state": "failed"})[0] == "unread"
    # REV-905e H1: a loaded unit stopped, no NVSwitch device seen (B300 as the ssh user, another
    # NVIDIA bridge), every fabric Completed: by whether it was active in this boot
    for soft in ({}, dict(name="NVIDIA H100 80GB HBM3", bridge=1)):
        # REV-905f M1: activating / deactivating too (a Restart= gap, a hung NVL5 precheck)
        for state in ("inactive", "failed", "activating", "deactivating"):
            assert absolute(check(fm_text(state, entered=0, **soft)).facts) == ("ok", node_health.OFF_HOST)
            assert absolute(check(fm_text(state, **soft)).facts) == (
                "bad", f"nvidia-fabricmanager {state} (NVSwitch present)")
            assert absolute(check(fm_text(state, entered=None, **soft)).facts) == (
                "unread", f"nvidia-fabricmanager {state}, fabric Completed, its active time unread")
            # the fabric not Completed (or unread): bad as before
            assert absolute(check(fm_text(state, entered=0, fabric=(0, FABRIC_IN_PROGRESS), **soft)).facts)[0] == "bad"
    for hard in (B300_ROOT, HOPPER, dict(name="NVIDIA H100 80GB HBM3", pci=1)):  # a device of its own: bad
        assert absolute(check(fm_text("inactive", entered=0, **hard)).facts)[0] == "bad"
    # its own read, never carried: an unread check after a bad one is unread
    bad = check(fm_text("failed"))
    assert absolute(check(node_text(gpus=(124, "")), bad.facts).facts)[0] == "unread"
    assert absolute(check(fm_text("failed", fabric=(124, "")), bad.facts).facts)[0] == "unread"


def test_an_off_host_fabric_manager_restarting_is_never_lost():
    # REV-905f M1: soft NVSwitch evidence, fabric Completed, never active in this boot: a unit
    # in its Restart= gap (activating / deactivating) and then failed stays ok
    for soft in ({}, dict(name="NVIDIA H100 80GB HBM3", bridge=1)):
        for mid in ("activating", "deactivating"):
            _, rs = watch_through(fm_text(mid, entered=0, **soft), fm_text("failed", entered=0, **soft),
                                  fm_text("failed", entered=0, **soft))
            assert [r.verdict for r in rs] == ["ok", "ok", "ok"]
            # it ran here (ActiveEnter > 0): bad, then lost
            _, rs = watch_through(*[fm_text(mid, **soft) for _ in range(3)])
            assert [r.verdict for r in rs] == ["bad_once", "lost", "lost"]


def test_the_fabric_manager_section_reads_systemctl_and_the_nvswitch_probe(tmp_path):
    read = node_health.read_fabric_manager
    every = dict(active_enter=ENTERED, nvswitch_dev=0, nvswitch_pci=2, nvidia_bridge=1, mlx=10, mlx_vpd=0, sw_mng=0)
    assert read(host_probe.Section(3, fm_out("failed", pci=2, bridge=1, restarts=7))) == ("failed", 7, every)
    assert read(host_probe.Section(0, fm_out("active", proc=4, pci=None, entered=None)))[2] == dict(
        nvswitch_proc=4, nvswitch_dev=0, mlx=10, mlx_vpd=0, sw_mng=0)
    assert read(host_probe.Section(3, fm_out("unknown")))[0] == ""  # not a state
    assert read(host_probe.Section(3, fm_out("failed", vpd=None, entered=0)))[2] == dict(
        active_enter=0, nvswitch_dev=0, nvswitch_pci=0, mlx=10)
    # REV-905e: hidden processes; a loaded unit that was active in this boot and failed stays failed
    hid = dict(load="loaded", proc1=0)
    assert read(host_probe.Section(3, fm_out("failed", **hid)))[0] == "failed"
    assert "fm_hidden" not in read(host_probe.Section(3, fm_out("failed", **hid)))[2]
    for state, entered in (("failed", 0), ("failed", None), ("inactive", ENTERED)):
        got = read(host_probe.Section(3, fm_out(state, entered=entered, **hid)))
        assert got[0] == "" and got[2]["fm_hidden"] == "hidepid", (state, entered)
    assert read(host_probe.Section(3, fm_out("failed", load="masked", proc1=0)))[0] == ""
    assert read(host_probe.Section(3, fm_out("activating", **hid)))[0] == "activating"
    assert "fm_hidden" not in read(host_probe.Section(3, fm_out("activating", **hid)))[2]
    assert "fm_hidden" not in read(host_probe.Section(124, "loadstate=not-found\nproc1=0"))[2]  # no state: no note
    assert read(host_probe.Section(127, "nrestarts=\nnvswitch_dev=0")) == ("", None, {"nvswitch_dev": 0})
    assert read(host_probe.Section(0, "active")) == ("active", None, {})  # an older script's output
    assert read(host_probe.Section(3, "failed")) == ("", None, {})  # no LoadState: no unit known to have failed
    assert read(host_probe.Section(127, "active")) == ("", None, {})  # systemctl not found: no state
    assert read(None) == ("", None, {})
    # REV-905c H: the LoadState decides what inactive / failed mean
    assert read(host_probe.Section(3, fm_out("inactive", load="not-found", fm_proc=0)))[0] == "not running"
    assert read(host_probe.Section(3, fm_out("inactive", load="not-found", fm_proc=2)))[0] == "active"
    assert read(host_probe.Section(3, fm_out("inactive", load="not-found")))[0] == ""  # no process count
    assert read(host_probe.Section(3, fm_out("inactive", load="masked")))[0] == ""
    assert read(host_probe.Section(3, fm_out("activating", load="masked")))[0] == "activating"
    assert read(host_probe.Section(124, fm_out("failed")))[2]["timed_out"] == 2
    assert read(host_probe.Section(124, ""))[2] == {"timed_out": 1}
    # REV-905d L2: rc 137 (killed) is a timeout too; lines without the state line are "before"
    assert read(host_probe.Section(137, fm_out("failed")))[2]["timed_out"] == 2
    assert read(host_probe.Section(124, "loadstate=loaded\nproc1=1"))[2]["timed_out"] == 1
    assert read(host_probe.Section(124, fm_out("unknown")))[2]["timed_out"] == 1  # no state: before

    def run(path, **env):  # the section as the check runs it (its stderr merged), on a host of fakes
        out = subprocess.run(["bash", "-c", probe_script([("fabricmanager", node_health.FM_SCRIPT)])],
                             capture_output=True, text=True, env={"PATH": path, **env}, check=False)
        return host_probe.parse_sections(out.stdout)["fabricmanager"]

    fakes, tools, paths = _fm_host(tmp_path)
    path = f"{fakes}:{tools}"
    unit, entered, proc = fakes / "unit", fakes / "entered", Path(paths["__YETO_PROC"])
    sec = run(path, **paths)
    # REV-905d: systemd < 230 has no --value (the fake refuses it): LoadState still reads
    assert sec.text.splitlines()[:5] == ["failed", "loadstate=loaded", "active_enter=0", "container=0",
                                         "proc1=1"]  # the state first
    base = dict(active_enter=0, nvswitch_dev=0, nvswitch_pci=2, nvidia_bridge=1, mlx=4, mlx_vpd=3, sw_mng=1)
    assert sec.rc == 3 and read(sec) == ("failed", 5, dict(base, fm_proc=0))
    _fm_process(proc, 4242, 777)  # a stale failed host unit, a driver container's fabric manager
    assert read(run(path, **paths)) == ("active", 5, dict(base, fm_proc=1, fm_ids="4242:777"))
    # in a container (REV-905e): by systemd-detect-virt -c's output (none: not one), else
    # /run/systemd/container, /.dockerenv, /run/.containerenv, else pid 1's environ
    virt = fakes / "systemd-detect-virt"
    for out, rc, want in (("docker", 0, "1"), ("lxc", 0, "1"), ("none", 1, "0"), ("", 1, "0")):
        virt.write_text(f'#!/bin/bash\n[ "$1" = -c ] && {{ [ -n "{out}" ] && echo {out}; exit {rc}; }}; exit 2\n')
        virt.chmod(0o755)
        sec = run(path, **paths)
        assert f"container={want}" in sec.text, out
    virt.write_text('#!/bin/bash\n[ "$1" = -c ] && { echo docker; exit 0; }; exit 2\n')
    sec = run(path, **paths)
    assert "proc1=0" in sec.text and read(sec)[0] == ""
    assert read(sec)[2]["fm_hidden"] == "container" and "fm_proc" not in read(sec)[2]
    virt.write_text('#!/bin/bash\necho none; exit 1\n')  # there, says none: the fallbacks are not read
    (Path(paths["__YETO_ROOT"]) / ".dockerenv").write_text("")
    assert "container=0" in run(path, **paths).text
    virt.unlink()
    assert "container=1" in run(path, **paths).text  # /.dockerenv
    (Path(paths["__YETO_ROOT"]) / ".dockerenv").unlink()
    (Path(paths["__YETO_RUN"]) / ".containerenv").write_text("")
    assert "container=1" in run(path, **paths).text  # podman
    (Path(paths["__YETO_RUN"]) / ".containerenv").unlink()
    (Path(paths["__YETO_RUN"]) / "systemd").mkdir()
    (Path(paths["__YETO_RUN"]) / "systemd" / "container").write_text("")  # empty: not one
    assert "container=0" in run(path, **paths).text
    (Path(paths["__YETO_RUN"]) / "systemd" / "container").write_text("lxc\n")
    assert "container=1" in run(path, **paths).text
    (Path(paths["__YETO_RUN"]) / "systemd" / "container").unlink()
    (proc / "1" / "environ").write_bytes(b"PATH=/bin\x00container=lxc\x00")
    assert "container=1" in run(path, **paths).text
    (proc / "1" / "environ").write_bytes(b"PATH=/bin\x00")
    assert "container=0" in run(path, **paths).text
    (proc / "1" / "environ").chmod(0o000)  # not readable: not a container, no error
    sec = run(path, **paths)
    assert "container=0" in sec.text and "environ" not in sec.text
    (proc / "1" / "environ").chmod(0o600)
    unit.write_text("not-found")  # no unit: the process (comm cut at 15) says it runs
    assert read(run(path, **paths))[0] == "active"
    (proc / "4242" / "comm").write_text("bash\n")
    assert read(run(path, **paths))[0] == "not running"
    (proc / "1" / "comm").unlink()  # the processes not seen (hidepid): nothing counted
    sec = run(path, **paths)
    assert "proc1=0" in sec.text and read(sec)[0] == "" and "fm_proc" not in read(sec)[2]
    unit.write_text("loaded")
    nvsw, dev = Path(paths["__YETO_NVSW_PROC"]), Path(paths["__YETO_DEV"])
    nvsw.mkdir()
    for n in range(4):
        (nvsw / f"0000:0{n}:00.0").write_text("")
        (dev / f"nvidia-nvswitch{n}").write_text("")
    (dev / "nvidia-nvswitchctl").write_text("")  # the control device is not a switch
    (fakes / "lspci").write_text("#!/bin/bash\nexit 1\n")  # lspci there, failing: no count (L1)
    assert read(run(path, **paths))[2] == dict(active_enter=0, nvswitch_proc=4, nvswitch_dev=4, mlx=4, mlx_vpd=3,
                                               sw_mng=1, fm_hidden="hidepid")  # /proc/1/comm still gone
    entered.write_text("ActiveEnterTimestampMonotonic=41000000")  # active in this boot: failed stays failed
    assert read(run(path, **paths))[0] == "failed" and "fm_hidden" not in read(run(path, **paths))[2]
    entered.write_text("ActiveEnterTimestampMonotonic=0")
    (fakes / "timeout").unlink()  # no timeout: no VPD read (it could hang the section)
    assert read(run(path, **paths))[2] == dict(active_enter=0, nvswitch_proc=4, nvswitch_dev=4, mlx=4,
                                               fm_hidden="hidepid")
    (fakes / "systemctl").unlink()
    (fakes / "lspci").unlink()
    sec = run(path, **paths)  # no systemctl, no lspci
    assert sec.rc == 127 and read(sec)[:2] == ("", None) and "nvswitch_pci" not in read(sec)[2]


def _fm_process(proc, pid, start, comm="nv-fabricmanage"):
    (proc / str(pid)).mkdir(parents=True, exist_ok=True)
    (proc / str(pid) / "comm").write_text(comm + "\n")
    (proc / str(pid) / "stat").write_text(f"{pid} ({comm}) S " + " ".join(["0"] * 18) + f" {start} 0 0\n")


def _fm_host(tmp_path, *, real_timeout=False):
    """Fakes of a node for the fabricmanager section: systemctl (its LoadState in ``unit``;
    it refuses --value, as systemd < 230), lspci, timeout (the real one with
    ``real_timeout``), /proc, the NVSwitch procfs dir, /dev and the PCI sysfs."""
    fakes, tools = tmp_path / "bin", tmp_path / "tools"  # tools: only what the section needs
    fakes.mkdir()
    tools.mkdir()
    for tool in ("bash", "ls", "wc", "tr", "grep", "head", "printf", "sed", "awk", "sleep"):
        if shutil.which(tool):
            (tools / tool).symlink_to(shutil.which(tool))
    unit, entered = fakes / "unit", fakes / "entered"
    unit.write_text("loaded")
    entered.write_text("ActiveEnterTimestampMonotonic=0")
    (fakes / "systemctl").write_text(
        f'#!/bin/bash\nread -r load < {shlex.quote(str(unit))}\nread -r ae < {shlex.quote(str(entered))}\n'
        'for a in "$@"; do [ "$a" = --value ] && { echo "Unknown option --value" >&2; exit 1; }; done\n'
        'if [ "$1" = is-active ]; then if [ "$load" = loaded ]; then echo failed; else echo inactive; fi; exit 3; fi\n'
        'if [ "$1" = show ]; then case "$3" in LoadState) echo "LoadState=$load";; NRestarts) echo NRestarts=5;; '
        'ActiveEnterTimestampMonotonic) echo "$ae";; esac; fi\n')
    # two NVSwitches (GH100 ids), another NVIDIA 0680 bridge (soft), a GPU
    (fakes / "lspci").write_text("#!/bin/bash\necho '05:00.0 0680: 10de:22a3 (rev a1)'\n"
                                 "echo '06:00.0 0680: 10de:22a3 (rev a1)'\necho '07:00.0 0680: 10de:1111 (rev a1)'\n"
                                 "echo '18:00.0 0302: 10de:2901 (rev a1)'\n")
    if real_timeout:
        (tools / "timeout").symlink_to(shutil.which("timeout"))
    else:
        (fakes / "timeout").write_text('#!/bin/bash\nshift\nexec "$@"\n')
    for f in fakes.iterdir():
        f.chmod(0o755)
    proc, dev, sys_pci = tmp_path / "proc", tmp_path / "dev", tmp_path / "pci"
    dev.mkdir()
    _fm_process(proc, 1, 1, comm="systemd")
    _fm_process(proc, 4243, 5, comm="nv-fabricmanagerx")
    # SW_MNG on a .1 function only (REV-905c M): every function is read
    for name, vendor, vpd in (("0000:01:00.0", "0x15b3", b"PN\x00MCX75\x00"),
                              ("0000:01:00.1", "0x15b3", b"PN\x00MCX75\x00V3SMDL=SW_MNG\x00"),
                              ("0000:02:00.0", "0x15b3", b""), ("0000:03:00.0", "0x15b3", None),
                              ("0000:18:00.0", "0x10de", b"")):
        (sys_pci / name).mkdir(parents=True)
        (sys_pci / name / "vendor").write_text(vendor + "\n")
        if vpd is not None:
            (sys_pci / name / "vpd").write_bytes(vpd)
    (tmp_path / "run").mkdir()
    (tmp_path / "root").mkdir()
    paths = dict(__YETO_PROC=str(proc), __YETO_NVSW_PROC=str(tmp_path / "proc-nvswitch"), __YETO_DEV=str(dev),
                 __YETO_SYS_PCI=str(sys_pci), __YETO_RUN=str(tmp_path / "run"), __YETO_ROOT=str(tmp_path / "root"))
    return fakes, tools, paths


@pytest.mark.skipif(shutil.which("timeout") is None, reason="no timeout command here")
@pytest.mark.parametrize("fifos", [1, 8])
def test_a_vpd_that_does_not_answer_does_not_hold_the_section(tmp_path, fifos):
    # REV-905d L1: a VPD read is cut at 1 s, all of them at 3 s (the rest not read)
    fakes, tools, paths = _fm_host(tmp_path, real_timeout=True)
    for n in range(fifos):
        d = Path(paths["__YETO_SYS_PCI"]) / f"0000:9{n}:00.0"
        d.mkdir()
        (d / "vendor").write_text("0x15b3\n")
        os.mkfifo(d / "vpd")  # opens, never answers
    start = time.monotonic()
    out = subprocess.run(["bash", "-c", probe_script([("fabricmanager", node_health.FM_SCRIPT)])],
                         capture_output=True, text=True, env={"PATH": f"{fakes}:{tools}", **paths}, check=False,
                         timeout=60)
    took = time.monotonic() - start
    probe = node_health.read_fabric_manager(host_probe.parse_sections(out.stdout)["fabricmanager"])[2]
    assert probe["mlx"] == 4 + fifos and probe["mlx_vpd"] < probe["mlx"]
    if fifos == 1:
        assert took < 2.5 and probe["mlx_vpd"] == 3  # the others read
    else:
        assert took < fifos - 1  # unbounded, it takes one second per FIFO


def read_state(text, rc=3):
    return node_health.read_fabric_manager(host_probe.Section(rc, text))[0]


def test_a_fabric_manager_without_its_unit_is_judged_by_its_process():
    # REV-905c H: no unit on the host (a GPU Operator driver container runs nv-fabricmanager;
    # a shared NVSwitch VM runs it on its service VM): systemctl says inactive, nothing stopped
    for node in (dict(HOPPER), {}, dict(B300_ROOT_NO_SW), dict(B300_ROOT)):
        _, rs = watch_through(*[fm_text("inactive", **UNIT_IN_CONTAINER, **node)] * 3)
        assert [r.verdict for r in rs] == ["ok"] * 3 and rs[0].result.facts["fm"] == "active"
    # REV-905d: no unit and no process. Off-host (no NVSwitch device of its own, every
    # fabric Completed: a shared NVSwitch guest, B300 seen only by its GPU names): ok
    for node in ({}, dict(B300_ROOT_NO_SW), dict(HIDDEN)):
        _, rs = watch_through(*[fm_text("inactive", **UNIT_GONE, **node)] * 3)
        assert [r.verdict for r in rs] == ["ok"] * 3 and all(r.result.reasons == [] for r in rs)
        assert node_health.fabric_manager_absolute(rs[0].result.facts) == ("ok", node_health.OFF_HOST)
        assert rs[0].result.facts["fm"] == rs[0].result.facts["fm_state"] == "not running"
    # an NVSwitch device on the node, its processes seen: the fabric manager is gone, at once
    for node in (dict(HOPPER), dict(B300_ROOT), dict(name="NVIDIA H100 80GB HBM3", dev=1)):
        _, rs = watch_through(*[fm_text("inactive", **UNIT_GONE, **node)] * 3)
        assert [r.verdict for r in rs] == ["bad_once", "lost", "lost"]
        assert rs[0].result.reasons == ["nv-fabricmanager not running (NVSwitch present)"]
    # the processes not seen (hidepid): unread, never bad, never ok
    _, rs = watch_through(*[fm_text("inactive", **HIDEPID, **HOPPER)] * 3)
    assert [r.verdict for r in rs] == ["ok"] * 3
    assert node_health.fabric_manager_absolute(rs[0].result.facts) == ("unread", "nvidia-fabricmanager state unread")
    assert read_state(fm_out("inactive", load="not-found", proc1=0, fm_proc=0)) == ""  # a count not seen
    # nvidia-smi -q failed (no fabric state): not off-host
    failed_q = check(fm_text("inactive", fabric=(9, ""), **UNIT_GONE, **HIDDEN))
    assert node_health.fabric_manager_absolute(failed_q.facts)[0] == "unread"
    # the fabric not completed: bad by its own reason, as before
    stuck = fm_text("inactive", fabric=(0, FABRIC_IN_PROGRESS), **UNIT_GONE)
    _, rs = watch_through(stuck, stuck)
    assert [r.verdict for r in rs] == ["bad_once", "lost"]
    assert rs[0].result.reasons == ["NVLink fabric In Progress/N/A on 8/8 GPUs (fabric manager: not running)"]
    assert node_health.fabric_manager_absolute(rs[0].result.facts)[0] == "unread"
    # its process seen in this boot counts as active: gone later, it must run (and reaches lost)
    _, rs = watch_through(fm_text("inactive", **UNIT_IN_CONTAINER), fm_text("inactive", **UNIT_GONE),
                          fm_text("inactive", **UNIT_GONE))
    assert [r.verdict for r in rs] == ["ok", "bad_once", "lost"]
    assert rs[1].result.reasons == ["nv-fabricmanager not running (was active)"]
    # a new boot forgets it; a GB tray is never judged
    _, rs = watch_through(fm_text("inactive", **UNIT_IN_CONTAINER), fm_text("inactive", boot="boot-2", **UNIT_GONE))
    assert [r.verdict for r in rs] == ["ok", "ok"]
    _, rs = watch_through(fm_text("inactive", **UNIT_IN_CONTAINER, **TRAY), fm_text("inactive", **UNIT_GONE, **TRAY))
    assert [r.verdict for r in rs] == ["ok", "ok"]
    # off-host is a judged read: the next one is not a first read
    _, rs = watch_through(fm_text("inactive", **UNIT_GONE), fm_text("failed"), fm_text("failed"))
    assert [r.verdict for r in rs] == ["ok", "ok", "ok"]
    # an unjudged read (hidepid): the next judged read is still the first
    _, rs = watch_through(fm_text("inactive", **HIDEPID), fm_text("failed"), fm_text("failed"))
    assert [r.verdict for r in rs] == ["ok", "bad_once", "lost"]


# REV-905d scenarios: (LoadState, is-active, nv-fabricmanager processes) on an NVSwitch node
SCENARIOS = {
    "18 masked unit, the fabric manager crashed": (dict(load="masked", fm_proc=0, **HOPPER), "inactive", "lost"),
    "19 masked unit, its process runs": (dict(load="masked", fm_proc=1, **HOPPER), "inactive", "ok"),
    "20 stale disabled host unit, a driver container runs it": (dict(load="loaded", fm_proc=1), "inactive", "ok"),
    "20f stale failed host unit, a driver container runs it": (dict(load="loaded", fm_proc=1), "failed", "ok"),
    "20b failed host unit, nothing runs it (Hopper)": (dict(load="loaded", fm_proc=0, **HOPPER), "failed", "lost"),
    "21 failed host unit, nothing runs it (B300 as root)": (dict(load="loaded", fm_proc=0, **B300_ROOT), "failed",
                                                            "lost"),
    # REV-905e H1: HGX B300 as the ssh user (no NVSwitch device seen, a Completed fabric): the
    # unit's ActiveEnterTimestampMonotonic says whether it ran in this boot
    "22 B300 non-root, loaded unit never active (off-host)": (dict(load="loaded", fm_proc=0, entered=0), "inactive",
                                                              "ok"),
    "22f B300 non-root, loaded unit failed before it was active": (dict(load="loaded", fm_proc=0, entered=0),
                                                                   "failed", "ok"),
    "23 B300 non-root, loaded unit active before, failed": (dict(load="loaded", fm_proc=0), "failed", "lost"),
    "23i B300 non-root, loaded unit active before, stopped": (dict(load="loaded", fm_proc=0), "inactive", "lost"),
    "24 B300 non-root, its active time unread": (dict(load="loaded", fm_proc=0, entered=None), "failed", "unread"),
    # REV-905e L1: lspci alone
    "25 only lspci 10de:22a3, no unit, nothing runs it": (dict(UNIT_GONE, name="NVIDIA H100 80GB HBM3", pci=1),
                                                         "inactive", "lost"),
    "26 only another 10de 0680 bridge, no unit, nothing runs it": (dict(UNIT_GONE, name="NVIDIA H100 80GB HBM3",
                                                                        pci=0, bridge=1), "inactive", "ok"),
    "26u only another 10de 0680 bridge, no unit, fabric not read": (dict(UNIT_GONE, name="NVIDIA H100 80GB HBM3",
                                                                         pci=0, bridge=1, fabric=(124, "")),
                                                                    "inactive", "unread"),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_the_fabric_manager_scenarios(name):
    node, state, want = SCENARIOS[name]
    _, rs = watch_through(*[fm_text(state, **node)] * 3)
    assert [r.verdict for r in rs] == (["bad_once", "lost", "lost"] if want == "lost" else ["ok"] * 3)
    assert node_health.fabric_manager_absolute(rs[0].result.facts)[0] == {"lost": "bad"}.get(want, want)
    if want == "ok" and name[:2] in ("19", "20"):
        assert rs[0].result.facts["fm"] == "active"


def test_a_fabric_manager_without_its_unit_that_restarts_counts_its_processes():
    run = lambda ids, **kw: fm_text("inactive", **UNIT_IN_CONTAINER, fm_ids=ids, **kw)  # noqa: E731
    _, rs = watch_through(run("4242:777"), run("4242:777"), run("5151:901"), run("5151:901"), run("6000:950"))
    assert [r.verdict for r in rs] == ["ok"] * 5 and all(r.degraded is None for r in rs)
    assert [r.fm_restarted is not None for r in rs] == [False, False, True, False, True]
    assert rs[2].fm_restarted_fields() == {"streak": 1, "nrestarts": 0, "fm_ids": "5151:901", "reasons": [
        "nv-fabricmanager restarted (another process since the previous check)"]}  # no unit: NRestarts stays 0
    assert rs[4].result.facts["fm_ids"] == {"boot": BOOT, "ids": "6000:950"}
    # REV-905e M1: a process that comes and goes next to the one that stays is no restart
    _, rs = watch_through(run("1:1"), run("1:1 9:9"), run("1:1"), run("1:1 9:9"), run("1:1"))
    assert [r.verdict for r in rs] == ["ok"] * 5 and all(r.fm_restarted is None for r in rs)
    assert rs[4].result.facts["fm_restart_streak"]["n"] == 0
    # every process of the previous check replaced: one restart
    _, rs = watch_through(run("1:1 2:2"), run("3:3 4:4"))
    assert rs[1].fm_restarted["streak"] == 1
    _, rs = watch_through(run("1:1 2:2"), run("2:2 3:3"))  # one of them stays: none
    assert rs[1].fm_restarted is None
    # another process at every check: the same streak as NRestarts
    _, rs = watch_through(*[run(f"{4242 + n}:777") for n in range(5)])
    assert [r.verdict for r in rs] == ["ok", "ok", "ok", "bad_once", "lost"]
    assert rs[3].result.reasons == ["nv-fabricmanager restarted (another process since the previous check)"]
    # the same pid with another start time is another process; a new boot is a new baseline
    _, rs = watch_through(*[run(f"4242:{777 + n}") for n in range(4)], run("4242:1", boot="boot-2"))
    assert [r.verdict for r in rs] == ["ok", "ok", "ok", "bad_once", "ok"]
    # gone between two checks: bad by its state; back as the same process: ok
    _, rs = watch_through(run("4242:777"), fm_text("inactive", **UNIT_GONE), run("4242:777"))
    assert [r.verdict for r in rs] == ["ok", "bad_once", "ok"]
    _, rs = watch_through(*[run(f"{4242 + n}:777", **TRAY) for n in range(5)])
    assert [r.verdict for r in rs] == ["ok"] * 5  # a GB tray is never judged
    # the baseline and the streak are carried over a read failure; processes in another order are the same
    _, rs = watch_through(run("1:7"), run("2:7"), node_text(gpus=(124, "")), run("3:7"), run("4:7"))
    assert [r.verdict for r in rs] == ["ok", "ok", "read_failed", "ok", "bad_once"]
    _, rs = watch_through(*[run("5151:901 4242:777"), run("4242:777 5151:901")] * 2)
    assert [r.verdict for r in rs] == ["ok"] * 4 and all(r.fm_restarted is None for r in rs)
    assert rs[0].result.facts["fm_ids"]["ids"] == "4242:777 5151:901"  # sorted: the event's fm_ids are stable


def test_hidden_processes_leave_a_stopped_unit_unjudged():
    # REV-905e: hidepid on a GPU Operator host, its stale host unit inactive or failed (20 / 20f):
    # the driver container's fabric manager would not show
    for state, entered in (("inactive", ENTERED), ("inactive", 0), ("failed", 0), ("failed", None)):
        w, rs = watch_through(*[fm_text(state, load="loaded", proc1=0, entered=entered, **HOPPER)] * 3)
        assert [r.verdict for r in rs] == ["ok"] * 3, (state, entered)
        assert rs[0].degraded == [node_health.FM_HIDDEN_NOTE] and rs[1].degraded is None
        assert w.record()["degraded"] == [node_health.FM_HIDDEN_NOTE]  # it stays in the record
        assert node_health.fabric_manager_absolute(rs[0].result.facts) == ("unread",
                                                                           "nvidia-fabricmanager state unread")
    # REV-905e: its loaded unit was active in this boot and failed: bad, hidden or not, no note
    for node in ({}, HOPPER):
        _, rs = watch_through(*[fm_text("failed", load="loaded", proc1=0, **node)] * 3)
        assert [r.verdict for r in rs] == ["bad_once", "lost", "lost"]
        assert rs[0].result.reasons == ["nvidia-fabricmanager failed (NVSwitch present)"] and rs[0].degraded is None
    # the fabric not completed: bad by its own reason
    stuck = fm_text("failed", load="loaded", proc1=0, entered=0, fabric=(0, FABRIC_IN_PROGRESS))
    _, rs = watch_through(stuck, stuck)
    assert [r.verdict for r in rs] == ["bad_once", "lost"]
    # active stays active; once seen active, the memory judges a later visible stop
    _, rs = watch_through(fm_text("active", proc1=0), fm_text("failed", load="loaded", proc1=0, entered=0),
                          fm_text("failed"))
    assert [r.verdict for r in rs] == ["ok", "ok", "bad_once"]
    # a system container (its own pid 1): the same, with its own note
    _, rs = watch_through(*[fm_text("inactive", load="not-found", proc1=0, container=1, **HOPPER)] * 3)
    assert [r.verdict for r in rs] == ["ok"] * 3 and rs[0].degraded == [node_health.FM_CONTAINER_NOTE]


def test_a_gb_tray_is_never_an_nvswitch_node_whatever_it_shows():
    # REV-905e: the GB rule comes before the devices
    for probe in (dict(pci=4), dict(dev=2), dict(proc=2), dict(sw=1, vpd=10)):
        _, rs = watch_through(*[fm_text("failed", **TRAY, **probe)] * 3)
        assert [r.verdict for r in rs] == ["ok"] * 3
        assert rs[0].result.facts["nvswitch"] == "no" and rs[0].result.facts["nvswitch_by"] == "compute tray (GB GPU)"


def test_without_a_boot_id_the_fabric_manager_is_judged_as_before():
    no_boot = lambda state: node_text(fabricmanager=(0 if state == "active" else 3, fm_out(state)),  # noqa: E731
                                      xid=(0, "source=dmesg"))
    _, rs = watch_through(no_boot("active"), no_boot("failed"), no_boot("failed"))
    assert [r.verdict for r in rs] == ["ok", "bad_once", "ok"]
    # remembered in a boot, then the boot id stops reading: the memory stays and judges nothing
    _, rs = watch_through(fm_text("active"), no_boot("failed"), no_boot("failed"), fm_text("failed"))
    assert [r.verdict for r in rs] == ["ok", "bad_once", "ok", "bad_once"]
    assert rs[2].result.facts["fm_boot"] == BOOT
    first = check(no_boot("failed"))  # a new launcher: the reason, no memory
    assert first.reasons == ["nvidia-fabricmanager failed (NVSwitch present)"] and "fm_boot" not in first.facts


def test_an_older_prev_without_the_fabric_manager_memory_is_read():
    old = {k: v for k, v in check(fm_text("active")).facts.items()
           if k not in ("fm_boot", "fm_why", "fm_judged", "fm_restarts", "fm_state", "nvswitch")}
    assert old["fm"] == "active" and old["xid"]["boot"] == BOOT
    a = check(fm_text("failed"), old)
    b = check(fm_text("failed"), a.facts)
    assert a.reasons == b.reasons == ["nvidia-fabricmanager failed (was active)"] and b.facts["fm_boot"] == BOOT
    # its XID cursor names another boot: a reboot since (one reason, as before, then none)
    rebooted = check(fm_text("failed", boot="boot-2"), old)
    assert rebooted.reasons == ["nvidia-fabricmanager failed (was active)"]
    assert check(fm_text("failed", boot="boot-2"), rebooted.facts).ok_single
    # no XID cursor at all (no kernel log access): this boot
    bare = {k: v for k, v in old.items() if k != "xid"}
    c = check(fm_text("failed"), bare)
    assert c.reasons == check(fm_text("failed"), c.facts).reasons == ["nvidia-fabricmanager failed (was active)"]
    # an older prev whose fabric manager was not active: it was judged then, nothing remembered
    assert check(fm_text("failed"), {**old, "fm": "failed"}).ok_single
    # an older prev of a check without the fabric manager: the next read is the first judged one
    assert check(fm_text("failed"), {k: v for k, v in old.items() if k != "fm"}).reasons == [
        "nvidia-fabricmanager failed (NVSwitch present)"]


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


def test_a_journal_without_a_kernel_line_of_this_boot_is_no_xid_source(tmp_path):
    """Inside a container without the host's journal (an ssh pool's sampler on the node
    runtime), dmesg is refused and journalctl -k exits 0 with no entry: no XID is read, so
    the source is none (the degraded event says so), not an empty journal."""
    refused = "echo 'dmesg: read kernel buffer failed: Operation not permitted' >&2\nexit 1"
    (tmp_path / "dmesg").write_text(f"#!/bin/sh\n{refused}\n")
    # no journal: -q drops the "-- No entries --" line, the exit code stays 0
    empty = ("echo 'No journal files were found.' >&2\n"
             "case \" $* \" in *' -q '*) ;; *) echo '-- No entries --';; esac\nexit 0")
    (tmp_path / "journalctl").write_text(f"#!/bin/sh\n{empty}\n")
    for name in ("dmesg", "journalctl"):
        (tmp_path / name).chmod(0o755)
    env = {"PATH": f"{tmp_path}:/usr/bin:/bin"}
    out = subprocess.run(["bash", "-c", node_health.XID_SCRIPT], capture_output=True, text=True, env=env, timeout=20)
    assert "source=none" in out.stdout and "source=journal" not in out.stdout
    (tmp_path / "journalctl").write_text("#!/bin/sh\necho '[ 1.0] kernel: Linux version 6.8.0'\nexit 0\n")
    out = subprocess.run(["bash", "-c", node_health.XID_SCRIPT], capture_output=True, text=True, env=env, timeout=20)
    assert "source=journal" in out.stdout


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
