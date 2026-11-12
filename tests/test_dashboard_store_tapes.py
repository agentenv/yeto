"""Island tapes through the run store: a sampler uploads its tape in chunks
(tapes/<island>/<incarnation>/<seq>.jsonl); serve / export --store read every
island from there, a replacement node (new incarnation) appears by itself."""

from __future__ import annotations

import json

from yeto.dashboard.cli import main as _unused  # noqa: F401 - the CLI module imports cleanly
from yeto.dashboard.reducer import Reducer
from yeto.dashboard.serve import DashboardState
from yeto.dashboard.store_tapes import StoreTapes, open_objects
from yeto.rl.engine.object_store import DirectoryObjectStore
from yeto.rl.engine.tape_upload import TapeUploader, chunk_key, store_objects, upload_interval_s

T0 = 1_790_000_000.0


def _append(path, *records):
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _round(sid, rid, t):
    return {"event": "rl_sampler_round", "sampler_id": sid, "rollout_id": rid, "groups": 4,
            "generate_s": 30.0, "policy_version": 0, "version_lag": 0, "time_unix": t}


def test_uploader_writes_complete_lines_in_numbered_chunks(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "store")
    tape = tmp_path / "events.a1.jsonl"
    up = TapeUploader(objects, tape, island="nc/s1", incarnation="a1", interval_s=999)
    assert up.flush() == 0  # no file yet
    _append(tape, _round("s1", 0, T0))
    with open(tape, "a") as f:
        f.write('{"event": "rl_sampler_ro')  # torn last line: waits
    assert up.flush() > 0
    _append(tape, {"x": 1})  # completes the torn line... as garbage + a record
    up.flush()
    keys = objects.list("tapes/")
    assert keys == [chunk_key("nc/s1", "a1", 0), chunk_key("nc/s1", "a1", 1)]
    assert keys[0] == "tapes/nc_s1/a1/00000000.jsonl"
    assert b"".join(objects.read(k) for k in keys) == tape.read_bytes()


def test_uploader_failure_never_raises_and_retries(tmp_path):
    class Flaky:
        def __init__(self):
            self.calls, self.data = 0, {}

        def write(self, key, data):
            self.calls += 1
            if self.calls == 1:
                raise OSError("store down")
            self.data[key] = data

    tape = tmp_path / "t.jsonl"
    _append(tape, _round("s1", 0, T0))
    obj, logs = Flaky(), []
    up = TapeUploader(obj, tape, island="s1", incarnation="a", log=logs.append)
    assert up.flush() == 0 and up.failures == 1 and logs
    assert up.flush() > 0 and list(obj.data) == [chunk_key("s1", "a", 0)]


def test_uploader_thread_and_final_flush(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "store")
    tape = tmp_path / "t.jsonl"
    up = TapeUploader(objects, tape, island="s1", incarnation="a", interval_s=0.01).start()
    _append(tape, _round("s1", 0, T0), _round("s1", 1, T0 + 30))
    up.stop()
    assert b"".join(objects.read(k) for k in objects.list("tapes/")) == tape.read_bytes()


def test_upload_interval_env_and_off():
    assert upload_interval_s(None, {}) == 30.0
    assert upload_interval_s(5, {}) == 5.0
    assert upload_interval_s(5, {"YETO_RL_TAPE_UPLOAD_S": "0"}) is None


def test_store_objects_of_a_directory_store(tmp_path):
    class S:
        root = tmp_path

    assert isinstance(store_objects(S()), DirectoryObjectStore)
    sentinel = object()

    class O:
        objects = sentinel

    assert store_objects(O()) is sentinel


def test_store_tapes_follow_islands_and_replacements(tmp_path):
    store = tmp_path / "store"
    objects = DirectoryObjectStore(store)
    a = tmp_path / "a.jsonl"
    up_a = TapeUploader(objects, a, island="nc-s1", incarnation="inc1")
    _append(a, {"event": "rl_sampler_start", "sampler_id": "nc-s1", "time_unix": T0}, _round("nc-s1", 0, T0 + 30))
    up_a.flush()
    r = Reducer()
    state = DashboardState(r, [], store=StoreTapes(open_objects(str(store))))
    state.poll()
    assert set(r.islands) == {"nc-s1"}
    # the node is replaced: a new incarnation of the same sampler, plus a second sampler
    b = tmp_path / "b.jsonl"
    up_b = TapeUploader(objects, b, island="nc-s1", incarnation="inc2")
    _append(b, _round("nc-s1", 1, T0 + 90))
    up_b.flush()
    c = tmp_path / "c.jsonl"
    up_c = TapeUploader(objects, c, island="nc-s2", incarnation="x")
    _append(c, _round("nc-s2", 0, T0 + 95))
    up_c.flush()
    _append(a, _round("nc-s1", 2, T0 + 100))  # late chunk of the old incarnation
    up_a.flush()
    state.poll()
    state.poll()  # re-polling never double counts
    assert set(r.islands) == {"nc-s1", "nc-s2"}
    rounds = r.page_view()["async"]["samplers"]["nc-s1"]["rounds"]
    assert [x["rollout_id"] for x in rounds] == [0, 1, 2]
    assert r.page_view()["async"]["samplers"]["nc-s1"]["groups"] == 12
    assert state.store.sources == {"store:nc-s1/inc1", "store:nc-s1/inc2", "store:nc-s2/x"}


def test_store_read_failure_keeps_chunk_order(tmp_path):
    objects = DirectoryObjectStore(tmp_path / "store")
    tape = tmp_path / "t.jsonl"
    up = TapeUploader(objects, tape, island="s1", incarnation="a")
    _append(tape, _round("s1", 0, T0)); up.flush()  # noqa: E702
    _append(tape, _round("s1", 1, T0 + 30)); up.flush()  # noqa: E702

    class Once:
        failed = False

        def list(self, prefix):
            return objects.list(prefix)

        def read(self, key):
            if key.endswith("00000000.jsonl") and not self.failed:
                self.failed = True
                raise OSError("timeout")
            return objects.read(key)

    r = Reducer()
    st = StoreTapes(Once())
    assert st.pump(r) == 0 and st.errors == 1  # chunk 1 waits for chunk 0
    assert st.pump(r) == 2
    assert [x["rollout_id"] for x in r.page_view()["async"]["samplers"]["s1"]["rounds"]] == [0, 1]


def test_export_with_store(tmp_path, capsys):
    from yeto.cli import main

    objects = DirectoryObjectStore(tmp_path / "store")
    tape = tmp_path / "t.jsonl"
    _append(tape, _round("s1", 0, T0))
    TapeUploader(objects, tape, island="s1", incarnation="a").flush()
    out = tmp_path / "p.html"
    assert main(["dashboard", "export", "--store", str(tmp_path / "store"), "-o", str(out)]) == 0
    assert "store:s1/a" in out.read_text() or "1 record" in capsys.readouterr().out
    assert '"s1"' in out.read_text()


def test_engine_sampler_uploads_its_tape_and_the_dashboard_reads_it(tmp_path, ports_v1_pin):
    """A real (fake-engine) sampler run: its whole tape lands under tapes/ in the
    store, and the dashboard shows the sampler with its rounds from there alone."""
    from tests.test_stage1_gaps import _sampler_spec
    from yeto.rl.engine import engine_sampler_entry as entry

    assert entry.run(_sampler_spec(tmp_path), resource_sampler=lambda emit: None) == 2
    [tape] = (tmp_path / "tape").glob("events.*.jsonl")
    objects = DirectoryObjectStore(tmp_path / "store")
    keys = objects.list("tapes/ppu4/")
    assert keys and b"".join(objects.read(k) for k in keys) == tape.read_bytes()
    r = Reducer()
    StoreTapes(open_objects(str(tmp_path / "store"))).pump(r)
    s = r.page_view()["async"]["samplers"]["ppu4"]
    assert s["n_rounds"] == 2 and s["samples"] == 8 and s["reward_mean_last"] == 0.5


def test_engine_sampler_upload_off(tmp_path, ports_v1_pin, monkeypatch):
    from tests.test_stage1_gaps import _sampler_spec
    from yeto.rl.engine import engine_sampler_entry as entry

    monkeypatch.setenv("YETO_RL_TAPE_UPLOAD_S", "0")
    assert entry.run(_sampler_spec(tmp_path), resource_sampler=lambda emit: None) == 2
    assert DirectoryObjectStore(tmp_path / "store").list("tapes/") == []


# ------------------------------------------------------------ request budget
class _PagedTapes:
    """An object store with S3's listing: sorted keys, ``limit`` a request;
    ``lists`` / ``reads`` count the requests."""

    def __init__(self, paged=True):
        self.data: dict[str, bytes] = {}
        self.lists = self.reads = 0
        if not paged:
            self.list_page = None

    def add(self, island, inc, seq, *records):
        self.data[chunk_key(island, inc, seq)] = "".join(json.dumps(r) + "\n" for r in records).encode()

    def read(self, key):
        self.reads += 1
        return self.data.get(key)

    def list(self, prefix, *, start_after=""):
        self.lists += max(1, -(-sum(1 for k in self.data if k.startswith(prefix)) // 1000))
        return sorted(k for k in self.data if k.startswith(prefix) and k > start_after)

    def list_page(self, prefix, *, start_after="", limit=1000):
        self.lists += 1
        keys = sorted(k for k in self.data if k.startswith(prefix) and k > start_after)
        return keys[:limit], (keys[limit - 1] if len(keys) > limit else None)


class _Feed:
    """A reducer that records what it is fed (the real one's offset rule)."""

    def __init__(self):
        self.records, self.offsets = [], {}

    def feed(self, rec, *, source, offset):
        if offset <= self.offsets.get(source, -1):
            return False
        self.offsets[source] = offset
        self.records.append((source, offset, rec))
        return True


def test_a_poll_lists_from_each_tape_last_chunk_not_the_whole_prefix():
    """A day of a 16-sampler run: 2880 chunks an island. main listed all of tapes/
    on every poll (one request per 1000 keys, growing for as long as the run)."""
    objects = _PagedTapes()
    islands = [f"nc-s{i:02d}" for i in range(16)] + ["trainer"]
    for island in islands:
        for seq in range(2880):
            objects.data[chunk_key(island, "inc1", seq)] = b""
    objects.add("nc-s03", "inc0", 0, _round("nc-s03", 0, T0))  # a short-lived earlier incarnation
    tapes, feed = StoreTapes(objects), _Feed()
    tapes.pump(feed)
    assert objects.lists == 17 * 2880 // 1000 + 1 and objects.reads == 17 * 2880 + 1  # the first poll reads everything
    objects.lists = objects.reads = 0
    assert tapes.pump(feed) == 0
    assert objects.lists == 17 + 1 and objects.reads == 0  # one jump per long tape; the short one rides along
    # every island adds a chunk, a node is replaced (new incarnation) and a new island joins
    for island in islands:
        objects.add(island, "inc1", 2880, _round(island, 2880, T0))
    objects.add("nc-s05", "inc2", 0, _round("nc-s05", 0, T0 + 1))
    objects.add("nc-s99", "a", 0, _round("nc-s99", 0, T0 + 2))
    objects.lists = objects.reads = 0
    assert tapes.pump(feed) == 19
    assert objects.lists == 17 + 1 and objects.reads == 19
    assert {"store:nc-s05/inc2", "store:nc-s99/a", "store:trainer/inc1"} <= tapes.sources
    assert len(tapes.cursors) == 20 and tapes.requests == 49 + 18 + 18
    assert tapes.pump(feed) == 0


def test_incremental_listing_feeds_what_a_full_listing_feeds_on_random_uploads():
    import random

    for seed in range(8):
        rng = random.Random(seed)
        paged, plain = _PagedTapes(), _PagedTapes(paged=False)
        plain.data = paged.data
        readers = [(StoreTapes(paged, page_keys=rng.choice([1, 2, 5, 50])), _Feed()), (StoreTapes(plain), _Feed())]
        next_seq: dict[tuple[str, str], int] = {}
        for step in range(300):
            tape = (f"s{rng.randrange(5)}", f"i{rng.randrange(3)}")
            if rng.random() < 0.7:
                seq = next_seq.get(tape, 0)
                next_seq[tape] = seq + 1
                paged.add(*tape, seq, {"event": "x", "tape": tape, "seq": seq}, {"event": "y", "seq": seq})
            if rng.random() < 0.05:
                paged.data[f"tapes/stray-{step}.txt"] = b"not a chunk"
                paged.data[f"tapes/s0/i0/deep/{step:08d}.jsonl"] = b"{}\n"
            if rng.random() < 0.4:
                assert len({tapes.pump(feed) for tapes, feed in readers}) == 1
        for tapes, feed in readers:
            tapes.pump(feed)
        assert readers[0][1].records == readers[1][1].records and len(readers[0][1].records) == 2 * sum(next_seq.values())
        assert readers[0][0].sources == readers[1][0].sources and readers[0][0].errors == 0


def test_a_failed_chunk_read_is_listed_again_from_the_tape_cursor():
    objects = _PagedTapes()
    for seq in range(4):
        objects.add("s1", "a", seq, _round("s1", seq, T0 + 30 * seq))
    read, fail = objects.read, {chunk_key("s1", "a", 2)}

    def flaky(key):
        if key in fail:
            fail.clear()
            raise OSError("timeout")
        return read(key)

    objects.read = flaky
    tapes, feed = StoreTapes(objects, page_keys=2), _Feed()
    assert tapes.pump(feed) == 2 and tapes.errors == 1  # chunks 0 and 1; chunk 3 waits for chunk 2
    assert tapes.cursors == {"tapes/s1/a/": chunk_key("s1", "a", 1)}
    assert tapes.pump(feed) == 2 and tapes.cursors == {"tapes/s1/a/": chunk_key("s1", "a", 3)}
    assert [rec["rollout_id"] for _s, _o, rec in feed.records] == [0, 1, 2, 3]


def test_store_polls_are_spaced_by_the_store_interval():
    objects = _PagedTapes()
    objects.add("s1", "a", 0, _round("s1", 0, T0))
    now = [100.0]
    tapes, feed = StoreTapes(objects, interval_s=10.0, clock=lambda: now[0]), _Feed()
    assert tapes.pump(feed) == 1 and objects.lists == 1
    objects.add("s1", "a", 1, _round("s1", 1, T0 + 30))
    now[0] += 2.0
    assert tapes.pump(feed) == 0 and objects.lists == 1  # the page polls every 2 s; the store is not asked
    now[0] += 8.0
    assert tapes.pump(feed) == 1 and objects.lists == 2


def test_directory_and_s3_stores_page_a_listing_the_same_way(tmp_path):
    from tests.test_rl_object_store import URI, FakeS3
    from yeto.cloud import s3_store

    class Paged(FakeS3):
        def list_objects_v2(self, Bucket, Prefix, MaxKeys, StartAfter=""):  # noqa: N803
            keys = sorted(k for b, k in self.objects if b == Bucket and k.startswith(Prefix) and k > StartAfter)
            return {"Contents": [{"Key": k} for k in keys[:MaxKeys]], "IsTruncated": len(keys) > MaxKeys}

    for store in (DirectoryObjectStore(tmp_path / "dir"), s3_store.S3ObjectStore(URI, client=Paged())):
        for seq in range(5):
            store.write(chunk_key("s1", "a", seq), b"{}\n")
        first, after = store.list_page("tapes/", limit=2)
        assert first == [chunk_key("s1", "a", 0), chunk_key("s1", "a", 1)] and after == first[-1]
        assert store.list_page("tapes/", start_after=chunk_key("s1", "a", 2), limit=2) == (
            [chunk_key("s1", "a", 3), chunk_key("s1", "a", 4)], None)
        assert store.list_page("tapes/", start_after=chunk_key("s1", "a", 4)) == ([], None)
