"""S17 N7 A5: deliver reconfiguration requests from THIS machine (no in-container resident process).
Follows the head job's streamed log for island 0's `rl_driver_phase` events; when a trigger's (phase, rollout_id)
appears, writes <state>/inbox/<req>.request.json into island 0's container with `modal container exec` (atomic
rename), reads it back, then waits until the island has taken it (request file gone or <req>.status.json present).
If a write/readback fails or the island does not take a request within --confirm-timeout-s, it tears the run down
at once (no blind rounds). Also snapshots island 0's elastic-state (tar.gz) after each confirmation and every 60 s.
usage: python3 s17-a5-deliver.py <run dir> <app name> '<triggers json>' [--confirm-timeout-s 420]"""
import argparse, base64, json, os, subprocess, sys, time

PY = "/home/michael/work/gpu-head/venv/bin/python"
STATE = "/root/yeto-rl/elastic-state"
ENV = {**os.environ, "HOME": "/home/michael"}


def log(R, **kw):
    kw["wall"] = time.time()
    with open(f"{R}/deliver.jsonl", "a") as f:
        f.write(json.dumps(kw) + "\n")
    print(json.dumps(kw), flush=True)


def modal(*args, timeout=120):
    t0 = time.time()
    p = subprocess.run([PY, "-m", "modal", *args], capture_output=True, text=True, timeout=timeout, cwd="/tmp", env=ENV)
    return p.returncode, p.stdout, p.stderr, time.time() - t0


def containers(app):
    rc, out, _, _ = modal("container", "list", "--json", timeout=60)
    if rc:
        return []
    return [c.get("Container ID") or c.get("container_id") for c in json.loads(out) if app in json.dumps(c)]


def find_island0(app, R, deadline):
    while time.time() < deadline:
        for cid in containers(app):
            rc, out, _, _ = modal("container", "exec", cid, "--", "sh", "-c",
                                  "ls /root/yeto-output/rl-island-*.jsonl 2>/dev/null", timeout=60)
            if rc == 0 and "rl-island-0.jsonl" in out:
                return cid
        time.sleep(15)
    return None


def snapshot(cid, R, tag):
    rc, out, _, dt = modal("container", "exec", cid, "--", "sh", "-c",
                           f"tar czf - -C /root/yeto-rl elastic-state 2>/dev/null | base64 -w0", timeout=120)
    if rc == 0 and out.strip():
        os.makedirs(f"{R}/es-snapshots", exist_ok=True)
        open(f"{R}/es-snapshots/{int(time.time())}-{tag}.tgz", "wb").write(base64.b64decode(out.strip()))
    return rc, dt


def phases(cid, seen):
    """New island-0 rl_driver_phase events, read from the island's own tape via exec (the head stream of island 0
    went silent in run c, so it is not used)."""
    rc, out, _, _ = modal("container", "exec", cid, "--", "sh", "-c",
                          "grep '\"rl_driver_phase\"' /root/yeto-output/rl-island-0.jsonl 2>/dev/null | tail -n 20", timeout=60)
    new = []
    if rc != 0:
        return new, rc
    for line in out.splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        k = (e.get("phase"), e.get("rollout_id"), e.get("time_unix"))
        if e.get("event") == "rl_driver_phase" and k not in seen:
            seen.add(k); new.append(e)
    return new, rc


def teardown(R, why):
    log(R, action="teardown", reason=why)
    subprocess.run(["bash", f"{R}/teardown.sh", "deliver-abort"], env=ENV)
    open(f"{R}/DELIVER_ABORT", "w").write(why + "\n")
    sys.exit(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("R"); ap.add_argument("app"); ap.add_argument("triggers")
    ap.add_argument("--confirm-timeout-s", type=float, default=420)
    ap.add_argument("--boot-timeout-s", type=float, default=2400)
    a = ap.parse_args()
    R, trig = a.R, json.loads(a.triggers)["triggers"]
    if not trig:
        return
    cid = find_island0(a.app, R, time.time() + a.boot_timeout_s)
    if cid is None:
        teardown(R, "island 0 container not found")
    log(R, action="island0", container=cid)
    seen, done, last_snap, fails = set(), set(), 0.0, 0
    while len(done) < len(trig):
        if os.path.exists(f"{R}/submit_rc.txt"):
            log(R, action="run_ended_before_all_delivered", delivered=sorted(done))
            break
        evs, rc = phases(cid, seen)
        fails = fails + 1 if rc != 0 else 0
        if fails >= 10:
            teardown(R, "cannot read island 0 tape via exec (10 consecutive failures)")
        for e in evs:
            for i, t in enumerate(trig):
                ph, rid, req, body = t[:4]
                if i in done or e.get("phase") != ph or e.get("rollout_id") != rid:
                    continue
                b = base64.b64encode(json.dumps(body).encode()).decode()
                rc, _, err, dtw = modal("container", "exec", cid, "--", "sh", "-c",
                                        f"d={STATE}/inbox; mkdir -p $d && echo {b} | base64 -d > $d/.{req}.tmp && mv $d/.{req}.tmp $d/{req}.request.json")
                rc2, got, _, dtr = modal("container", "exec", cid, "--", "sh", "-c",
                                         f"cat {STATE}/inbox/{req}.request.json 2>/dev/null || cat {STATE}/inbox/{req}.status.json")
                log(R, action="write", request=req, trigger=[ph, rid], event_time=e.get("time_unix"),
                    write_rc=rc, write_s=round(dtw, 2), read_rc=rc2, read_s=round(dtr, 2), readback=got.strip()[:300], err=err[-300:])
                if rc != 0 or rc2 != 0 or not got.strip():
                    teardown(R, f"write/readback failed for {req}")
                t0 = time.time(); taken = None
                while time.time() - t0 < a.confirm_timeout_s:
                    rc3, out, _, _ = modal("container", "exec", cid, "--", "sh", "-c",
                                           f"ls {STATE}/inbox; cat {STATE}/inbox/{req}.status.json 2>/dev/null", timeout=60)
                    if rc3 == 0 and (f"{req}.status.json" in out or f"{req}.request.json" not in out):
                        taken = out.strip()[-600:]; break
                    time.sleep(10)
                log(R, action="confirm", request=req, taken=taken is not None, after_s=round(time.time() - t0, 1), inbox=taken)
                if taken is None:
                    teardown(R, f"island did not take {req} within {a.confirm_timeout_s} s")
                snapshot(cid, R, f"after-{req}")
                done.add(i)
        if time.time() - last_snap > 60:
            snapshot(cid, R, "periodic"); last_snap = time.time()
        time.sleep(3)
    # keep snapshotting until the run ends (finalization records come at the stop boundary)
    while not os.path.exists(f"{R}/submit_rc.txt"):
        rc, _ = snapshot(cid, R, "tail")
        if rc != 0:
            break
        time.sleep(30)
    log(R, action="done", delivered=sorted(done))


if __name__ == "__main__":
    main()
