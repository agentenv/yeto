"""Cluster teardown with verification and teardown diagnostics."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

from .verda import VERDA_ISLANDS


def _cloud_live_instances_probe(cluster: str):
    """A zero-arg callable that queries the CLOUD (not sky's state DB) for a
    cluster's non-terminated instances, returning their ids. Captured BEFORE
    sky.down, because down deletes the cluster record we need to build the
    query. Returns None when the cluster can't be cloud-verified — never
    provisioned, or a cloud whose sky implementation predates the provision
    status API — in which case callers trust sky.down. Cloud-agnostic: sky's
    provision.query_instances routes to each cloud's own implementation.
    """
    try:
        from sky import clouds, global_user_state
        from sky import provision as provision_lib

        record = global_user_state.get_cluster_from_name(cluster)
        if record is None or record.get("handle") is None:
            return None
        handle = record["handle"]
        cloud = handle.launched_resources.cloud
        # sky's StatusVersion enum defines only ``>=``; ``<`` raises, which
        # used to trip the except below and silently disable verification
        # for every cloud ("cannot set up cloud verification ... '<' not
        # supported").
        if cloud is None or not (cloud.STATUS_VERSION >= clouds.StatusVersion.SKYPILOT):
            return None
        cloud_name = repr(cloud)
        name = handle.cluster_name
        name_on_cloud = handle.cluster_name_on_cloud
        provider_config = global_user_state.get_cluster_yaml_dict(handle.cluster_yaml)["provider"]

        def live():
            found = provision_lib.query_instances(
                cloud_name, name, name_on_cloud, provider_config,
                non_terminated_only=True,
            )
            # A None status means terminated/terminating; a real status means
            # the instance is still up (or stopped) — i.e. an orphan.
            return [iid for iid, (st, _reason) in found.items() if st is not None]

        return live
    except Exception as e:  # any sky-internals drift -> fall back to trusting down
        print(f"[launcher] cannot set up cloud verification for {cluster}: {e}", file=sys.stderr)
        return None


# Diagnostics fetched before an island is torn down: (label, remote glob or
# None for `sky logs`, local file name). Best effort, bounded.
TEARDOWN_DIAG_DIR = "~/yeto-diag"
TEARDOWN_DIAG_TIMEOUT_S = 120


def teardown_diagnostic_commands(cluster: str, dest: str, head: bool = False) -> list[tuple[str, list[str], str | None]]:
    """(label, argv, stdout file or None) for everything worth keeping
    before `cluster` goes away: its job log and event tape(s); on the head
    also sky's API server log and the cluster's recorded events."""
    cmds: list[tuple[str, list[str], str | None]] = [
        ("job_log", ["sky", "logs", cluster, "--no-follow"], os.path.join(dest, "job.log")),
        (
            "event_tape",
            ["rsync", "-a", "-e", "ssh -o ConnectTimeout=20 -o BatchMode=yes",
             f"{cluster}:~/yeto-output/*.jsonl", dest + "/"],
            None,
        ),
    ]
    if head:
        cmds.append(("sky_server_log", ["cp", os.path.expanduser("~/.sky/api_server/server.log"), dest + "/"], None))
        cmds.append(
            (
                "cluster_events",
                ["python3", "-c",
                 "import json,sys\n"
                 "from sky import global_user_state as g\n"
                 "f=getattr(g,'get_cluster_events',None)\n"
                 "print(json.dumps(f(sys.argv[1], None) if f else 'unsupported', default=str, indent=1))",
                 cluster],
                os.path.join(dest, "cluster_events.json"),
            )
        )
    return cmds


def collect_teardown_diagnostics(
    cluster: str, dest_root: str | None = None, *, head: bool = False,
    run=subprocess.run, timeout: float = TEARDOWN_DIAG_TIMEOUT_S,
) -> dict[str, str]:
    """Best-effort pull of an island's logs before its teardown.

    Never raises: every command gets `timeout` seconds and a failure or
    timeout is only a warning, so the teardown that follows always runs.
    Returns {label: "ok" | "timeout" | "failed: ..."}."""
    dest = os.path.join(os.path.expanduser(dest_root or TEARDOWN_DIAG_DIR), cluster)
    out: dict[str, str] = {}
    try:
        os.makedirs(dest, exist_ok=True)
    except OSError as e:
        print(f"[launcher] WARNING: no diagnostics dir for {cluster}: {e}", file=sys.stderr)
        return {"dir": f"failed: {e}"}
    for label, argv, stdout_file in teardown_diagnostic_commands(cluster, dest, head=head):
        fh = None
        try:
            fh = open(stdout_file, "wb") if stdout_file else None
            proc = run(argv, stdout=fh or subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=timeout, check=False)
            code = getattr(proc, "returncode", 0)
            out[label] = "ok" if code == 0 else f"failed: exit {code}"
        except subprocess.TimeoutExpired:
            out[label] = "timeout"
        except Exception as e:  # noqa: BLE001 - diagnostics must never block teardown
            out[label] = f"failed: {e}"
        finally:
            if fh is not None:
                fh.close()
        if out[label] != "ok":
            print(f"[launcher] WARNING: {cluster}: could not fetch {label} before teardown ({out[label]})",
                  file=sys.stderr)
    print(f"[launcher] {cluster}: pre-teardown diagnostics in {dest}: {out}")
    return out


def _verda_teardown_check(cluster: str):
    """Verda-by-id verification for a Verda island whose ids are on record;
    None (sky's own cloud probe decides) when there are none — an empty id
    set must never read as "confirmed gone"."""
    guard = VERDA_ISLANDS.get(cluster)
    if guard is None:
        return None
    ids = guard.ids.get(cluster)
    if not ids:
        print(f"[launcher] WARNING: {cluster}: no Verda instance ids on record; "
              "falling back to sky's cloud probe", file=sys.stderr)
        return None
    from ..verda_ops import verify_teardown

    return lambda: verify_teardown(guard.api, ids, guard.on_cloud.get(cluster))


def collect_diagnostics_parallel(clusters, *, head: bool = False, total_timeout: float = 300.0,
                                 collect=None) -> dict:
    """Pre-teardown diagnostics for every island at once, bounded in total:
    islands still collecting after `total_timeout` are left behind (their
    threads are daemons) and the teardown proceeds."""
    if not clusters or os.environ.get("YETO_TEARDOWN_DIAG", "1") == "0":
        return {}
    collect = collect or collect_teardown_diagnostics
    out: dict = {}

    def one(c):
        try:
            out[c] = collect(c, head=head)
        except Exception as e:  # noqa: BLE001
            out[c] = {"error": str(e)}

    threads = [threading.Thread(target=one, args=(c,), daemon=True) for c in clusters]
    for t in threads:
        t.start()
    deadline = time.monotonic() + total_timeout
    for t in threads:
        t.join(max(0.0, deadline - time.monotonic()))
    late = [c for c in clusters if c not in out]
    if late:
        print(f"[launcher] WARNING: diagnostics still running for {late} after {total_timeout:.0f}s; "
              "tearing down anyway", file=sys.stderr)
    return out


def teardown_island(sky, cluster: str, *, head: bool = False, collect=collect_teardown_diagnostics,
                    num_nodes: int = 1) -> bool:
    """The launcher's teardown of one of its own clusters: diagnostics
    first, then terminate_and_verify (Verda-verified for Verda islands)."""
    if collect is not None and os.environ.get("YETO_TEARDOWN_DIAG", "1") != "0":
        try:
            collect(cluster, head=head)
        except Exception as e:  # noqa: BLE001
            print(f"[launcher] WARNING: diagnostics for {cluster} failed: {e}", file=sys.stderr)
    return terminate_and_verify(sky, cluster, num_nodes=num_nodes, verda_check=_verda_teardown_check(cluster))


def terminate_and_verify(
    sky, cluster, *, probe="auto", attempts=4, sleep_fn=time.sleep, down=None, num_nodes=1,
    verda_check=None,
) -> bool:
    """sky.down a cluster and CONFIRM at the cloud level that no instance
    survives, retrying the down while the cloud still reports live ones.

    sky.down has been observed to report success while a spot instance
    lingers (state DB and cloud diverge). In head-controller mode the head's
    sky is the ONLY thing that can reach the learner clusters, so a silent
    orphan is unrecoverable once the head is gone — hence verify here, before
    the head relinquishes control. Returns True iff the cluster is confirmed
    gone. When the cloud can't be queried we fall back to sky.down's own
    result: a clean down, or "does not exist" (sky never had it), counts;
    any other down error does not — that is exactly the case that used to
    print "teardown failed" and then claim the run was down.

    `down` overrides the sky.down call (the CLI routes it through its own
    patchable hook); `probe` is captured before the first down because
    sky.down deletes the record the probe is built from.

    `verda_check` (Verda islands) replaces sky's probe with Verda's own
    answer by instance id, including the OS volumes (trash included): it
    returns (ok, remaining) and the teardown counts only when ok.
    """
    if verda_check is not None:
        try:
            down = down or (lambda: sky.get(sky.down(cluster)))
            down()
        except Exception as e:
            print(f"[launcher] sky.down({cluster}) error: {e}; verifying at Verda", file=sys.stderr)
        try:
            ok, remaining = verda_check()
        except Exception as e:  # noqa: BLE001
            ok, remaining = False, [f"Verda verification failed: {e}"]
        if not ok:
            print(
                f"[launcher] {cluster}: teardown NOT complete at Verda; remaining: {remaining}",
                file=sys.stderr,
            )
        else:
            print(f"[launcher] {cluster}: Verda confirms instances and volumes deleted")
        return ok
    if probe == "auto":
        probe = _cloud_live_instances_probe(cluster)
    down = down or (lambda: sky.get(sky.down(cluster)))
    if int(num_nodes or 1) > 1:
        return _terminate_and_verify_nodes(cluster, probe=probe, attempts=attempts,
                                           sleep_fn=sleep_fn, down=down, num_nodes=int(num_nodes))

    def _down():
        try:
            down()
            return None
        except Exception as e:
            print(f"[launcher] sky.down({cluster}) error: {e}", file=sys.stderr)
            return e

    err = _down()
    if probe is None:
        return err is None or "does not exist" in str(err)
    for i in range(attempts):
        try:
            live = probe()
        except Exception as e:
            print(f"[launcher] cloud verify of {cluster} failed ({e}); trusting sky.down",
                  file=sys.stderr)
            return True
        if not live:
            return True
        print(
            f"[launcher] {cluster}: {len(live)} instance(s) still live after down "
            f"({','.join(map(str, live))}); retrying teardown ({i + 1}/{attempts})",
            file=sys.stderr,
        )
        sleep_fn(min(30, 5 * (i + 1)))
        _down()
    try:
        return not probe()
    except Exception:
        return True


def _terminate_and_verify_nodes(cluster, *, probe, attempts, sleep_fn, down, num_nodes) -> bool:
    """rl-multinode-island D10: a multi-node island is confirmed gone only when the
    cloud confirms EVERY node instance. The instance ids are captured before the
    first down; no cloud probe, a failing probe or any instance still live after
    the retries -> False, with the unconfirmed instance ids listed. sky.down's own
    result is never trusted for more than one node."""
    if probe is None:
        print(f"[launcher] {cluster}: {num_nodes}-node island cannot be cloud-verified; "
              "NOT confirmed down (check the cloud console)", file=sys.stderr)
        return False
    try:
        before = list(probe())
    except Exception as e:  # noqa: BLE001 - verification must not rely on sky's DB
        print(f"[launcher] {cluster}: cloud probe before down failed ({e}); NOT confirmed",
              file=sys.stderr)
        return False
    if len(before) != num_nodes:
        print(f"[launcher] {cluster}: cloud shows {len(before)} instance(s) for a {num_nodes}-node "
              f"island ({','.join(map(str, before))}); verifying what the cloud reports", file=sys.stderr)
    live = before
    for i in range(attempts + 1):
        try:
            down()
        except Exception as e:  # noqa: BLE001
            print(f"[launcher] sky.down({cluster}) error: {e}", file=sys.stderr)
        try:
            live = list(probe())
        except Exception as e:  # noqa: BLE001
            print(f"[launcher] {cluster}: cloud verify failed ({e}); NOT confirmed", file=sys.stderr)
            return False
        if not live:
            for iid in before:
                print(f"[launcher] {cluster}: node instance {iid} confirmed terminated")
            if not before:
                print(f"[launcher] {cluster}: cloud reports no instances (confirmed)")
            return True
        if i < attempts:
            print(f"[launcher] {cluster}: {len(live)} node instance(s) still live "
                  f"({','.join(map(str, live))}); retrying teardown ({i + 1}/{attempts})", file=sys.stderr)
            sleep_fn(min(30, 5 * (i + 1)))
    print(f"[launcher] {cluster}: UNCONFIRMED node instance(s) {','.join(map(str, live))} "
          "still live after teardown; delete them in the cloud console or rerun `yeto down`",
          file=sys.stderr)
    return False
