"""fleet-dashboard 4.4: ssh tape mirror reconnects without losing or repeating records.

The ssh hop is simulated with ``sh -c 'tail -c +N file'``: each connection
ends (a dropped link) after streaming what exists; the remote tape grows
while "disconnected".
"""

import json

from yeto.dashboard.mirror import SshTapeMirror, ssh_tail_command
from yeto.dashboard.reducer import Reducer
from yeto.dashboard.sources import load_all


def line(i):
    return json.dumps({"event": "rl_local_round", "island_id": 7, "local_round_id": i,
                       "time_unix": 1000.0 + i}) + "\n"


def test_reconnect_resumes_at_the_mirrored_offset(tmp_path):
    remote = tmp_path / "remote.jsonl"
    local = tmp_path / "mirror" / "rl-island-7.jsonl"
    remote.write_text(line(1) + line(2))
    sleeps = []
    clock = iter(range(100, 200))

    def grow(delay):
        sleeps.append(delay)
        if len(sleeps) == 1:
            with open(remote, "a") as fh:  # written while the link is down
                fh.write(line(3) + '{"event": "rl_local_round", "isl')  # torn tail too
        elif len(sleeps) == 2:
            with open(remote, "a") as fh:
                fh.write('and_id": 7, "local_round_id": 4, "time_unix": 1004.0}\n')

    m = SshTapeMirror("host", str(remote), local, island="7",
                      command=lambda off: ["sh", "-c", f"tail -c +{off + 1} {remote}"],
                      sleep=grow, clock=lambda: float(next(clock)))
    m.run_forever(max_connections=4)
    assert local.read_text() == remote.read_text()  # no loss, no duplicates
    side = [json.loads(x) for x in m.sidecar.read_text().splitlines()]
    assert [x["event"] for x in side][:3] == ["dashboard_source_lost", "dashboard_source_restored",
                                              "dashboard_source_lost"]
    assert all(x["island_id"] == "7" for x in side)
    assert sleeps[0] == 1.0  # backoff reset after a productive connection
    r = Reducer()
    load_all(r, [str(local.parent)])
    assert r.islands["7"]["round"] == 4 and r.counts["learner"] == 4 + len(side)


def test_backoff_grows_while_the_source_stays_down(tmp_path):
    sleeps = []
    m = SshTapeMirror("h", "/nonexistent", tmp_path / "m.jsonl",
                      command=lambda off: ["sh", "-c", "echo 'ssh: connect refused' >&2; exit 255"],
                      sleep=sleeps.append, backoff_max=8)
    m.run_forever(max_connections=6)
    assert sleeps == [2, 4, 8, 8, 8]
    (lost,) = [json.loads(x) for x in m.sidecar.read_text().splitlines()]
    assert lost["event"] == "dashboard_source_lost" and "refused" in lost["error"]


def test_lost_source_is_shown_on_the_island():
    r = Reducer()
    r.feed({"event": "dashboard_source_lost", "island_id": "7", "time_unix": 5.0, "error": "x"})
    assert [a["rule"] for a in r.overview(now=6.0)["alerts"]] == ["source_lost"]
    r.feed({"event": "dashboard_source_restored", "island_id": "7", "time_unix": 7.0})
    assert r.overview(now=8.0)["alerts"] == []


def test_ssh_command_quotes_the_remote_path():
    cmd = ssh_tail_command("u@h", "/tmp/a b.jsonl", 10)
    assert cmd[0] == "ssh" and cmd[-2] == "u@h" and cmd[-1] == "tail -c +11 -F '/tmp/a b.jsonl'"
