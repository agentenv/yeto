"""The site image catalog (own-cluster-fleet-stability 7.1) and the image selection
order of an ssh: trainer island and an ssh sampler pool (7.4): explicit, then the
site catalog, then the built-in default."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import yaml

from yeto.launch import site_images as si
from yeto.launch import ssh_sampler_pool as sp
from yeto.launch.node_inventory import InventoryError, parse_inventory, parse_node

D1 = "a" * 64
D2 = "b" * 64
D3 = "c" * 64
REF1 = f"registry.example/site/miles-new@sha256:{D1}"
REF2 = f"registry.example/site/miles-new@sha256:{D2}"
REF3 = f"registry.example/site/vllm-new@sha256:{D3}"
K31_SPEC = Path(__file__).resolve().parents[1] / "examples" / "k31" / "sampler-spec.nc-mi355x-vllm.json"


def record(**kw):
    raw = {"model": "moonshotai/Kimi-K3.1-Pro", "role": "trainer", "engine": "miles", "ref": REF1,
           "device_family": "cuda", "cpu_arch": "x86_64", "card_archs": ["sm103"],
           "versions": {"python": "3.12.3", "torch": "2.9.0", "cuda": "13.0"},
           "evidence": {"at": "2026-11-26T10:00:00Z", "node": "b300-0", "seconds": 812.4, "result": "pass"},
           "status": "verified-smoke", **kw}
    return {k: v for k, v in raw.items() if v is not None}


def write_catalog(path: Path, *records) -> Path:
    path.write_text(yaml.safe_dump({"version": 1, "images": list(records)}))
    return path


def b300(name="b300-0", **kw):
    raw = {"name": name, "host": name, "device_family": "cuda", "cpu_arch": "x86_64", "card": "B300",
           "cards_per_node": 8, "hbm_gib": 288, "host_memory_gib": 2048, **kw}
    return parse_node(raw, index=0)


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    """No developer catalog leaks in: the default path is under a fresh home."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv(si.PATH_ENV, raising=False)


# -- schema (7.1) --------------------------------------------------------------------


def test_a_record_round_trips_and_normalizes():
    r = si.parse_record(record(ref="docker:" + REF1, card_archs=["SM103"], cpu_arch="amd64", device_family="nvidia"))
    assert (r.ref, r.card_archs, r.cpu_arch, r.device_family, r.model_key) == (REF1, ("sm103",), "x86_64", "cuda",
                                                                             "kimi_k31")
    assert si.parse_record(r.to_doc()) == r


@pytest.mark.parametrize("change, message", [
    ({"role": "syncer"}, "role 'syncer' is unknown"),
    ({"engine": "deepspeed"}, "trainer engine 'deepspeed' is unknown"),
    ({"role": "sampler", "engine": "miles"}, "sampler engine 'miles' is unknown"),
    ({"ref": "registry.example/site/miles-new:latest"}, "must be a docker reference pinned by digest"),
    ({"ref": "Registry/Site/miles@sha256:" + D1}, "must be a docker reference"),     # upper-case path
    ({"ref": "Registry/site/miles@sha256:" + D1}, "must be a docker reference"),     # a dotless first part is path
    ({"ref": "registry.example/site/miles new@sha256:" + D1}, "must be a docker reference"),
    ({"ref": "registry.example/site/miles@sha256:" + "A" * 64}, "must be a docker reference"),
    ({"ref": "registry.example/-bad/miles@sha256:" + D1}, "must be a docker reference"),
    ({"card_archs": ["sm999"]}, r"card_archs \['sm999'\] unknown \(known: \["),
    ({"evidence": {"at": "2026-11-26T10:00:00", "node": "n", "seconds": 1, "result": "pass"}}, "has no time zone"),
    ({"evidence": {"at": "2026-11-26", "node": "n", "seconds": 1, "result": "pass"}}, "is not an ISO-8601 time"),
    ({"evidence": {"at": "yesterday", "node": "n", "seconds": 1, "result": "pass"}}, "is not an ISO-8601 time"),
    ({"evidence": {"at": "2026-13-40T10:00:00Z", "node": "n", "seconds": 1, "result": "pass"}},
     "is not an ISO-8601 time"),
    ({"cpu_arch": "riscv64"}, "cpu_arch 'riscv64' is unknown"),
    ({"card_archs": []}, "card_archs must be a non-empty list"),
    ({"card_archs": "sm103"}, "card_archs must be a non-empty list"),
    ({"status": "smoked"}, "status 'smoked' is unknown"),
    ({"evidence": {"at": "x", "node": "n"}}, r"needs evidence \['seconds', 'result'\]"),
    ({"evidence": {"at": "x", "node": "n", "seconds": 1, "result": "fail"}}, "needs evidence.result pass"),
    ({"model": None}, r"missing key\(s\) \['model'\]"),
    ({"device_family": "quantum"}, "device family 'quantum' is unknown"),
])
def test_bad_records_are_refused_by_key(change, message):
    raw = record(**{k: v for k, v in change.items() if v is not None})
    if change.get("model", 1) is None:
        raw.pop("model")
    with pytest.raises(si.SiteImagesError, match=message):
        si.parse_record(raw, where="images[0]")


def test_an_unknown_key_is_a_warning_and_the_record_is_read(capsys):
    r = si.parse_record(record(colour="red"), where="images[0]")
    assert r.ref == REF1
    assert "[yeto] site image catalog: WARNING: images[0]: unknown key(s) ['colour'] ignored" in capsys.readouterr().err


@pytest.mark.parametrize("ref", [
    REF1, f"localhost:5000/miles@sha256:{D1}", f"miles@sha256:{D1}", f"reg.example:443/a/b_c.d-e:t1.2@sha256:{D1}",
    f"ghcr.io/owner/yeto__x@sha256:{D1}",
])
def test_docker_references_pinned_by_digest_are_taken(ref):
    assert si.parse_record(record(ref=ref)).ref == ref
    assert si.valid_reference(ref) and not si.valid_reference(ref.split("@")[0])
    assert si.valid_reference(ref.split("@")[0], digest=False)


def test_evidence_at_is_read_as_utc_and_records_order_by_the_time(tmp_path):
    east = si.parse_record(record(evidence={"at": "2026-11-26T18:00:00+08:00", "node": "n", "seconds": 1,
                                            "result": "pass"}))
    assert east.at == "2026-11-26T10:00:00Z" and east.at_time.isoformat() == "2026-11-26T10:00:00+00:00"
    # 09:30 UTC is earlier than 10:00Z although its text sorts later
    later_text = record(ref=REF2, evidence={"at": "2026-11-26T11:30:00+02:00", "node": "n", "seconds": 1,
                                            "result": "pass"})
    path = write_catalog(tmp_path / "c.yaml", record(ref=REF1), later_text)
    assert [r.ref for r in si.match("kimi_k31", "trainer", b300(), path=path)] == [REF1, REF2]
    # half a second later is newer, although "10:00:00.500000Z" sorts before "10:00:00Z" as text
    half = record(ref=REF2, evidence={"at": "2026-11-26T10:00:00.5Z", "node": "n", "seconds": 1, "result": "pass"})
    path = write_catalog(tmp_path / "c2.yaml", half, record(ref=REF1))
    assert si.load(path)[0].at == "2026-11-26T10:00:00.500000Z"
    assert [r.ref for r in si.match("kimi_k31", "trainer", b300(), path=path)] == [REF2, REF1]
    bad = write_catalog(tmp_path / "bad.yaml", record(), record(evidence={"at": "2026-11-26T10:00:00", "node": "n",
                                                                          "seconds": 1, "result": "pass"}))
    with pytest.raises(si.SiteImagesError, match=rf"^\[yeto\] site image catalog: {bad}: images\[1\]: evidence.at "
                                                 r"'2026-11-26T10:00:00' has no time zone"):
        si.load(bad)


def test_a_retired_record_needs_no_evidence():
    assert si.parse_record(record(status="retired", evidence=None)).status == "retired"


def test_a_missing_catalog_is_empty_and_a_broken_one_names_its_path(tmp_path):
    assert si.load(tmp_path / "none.yaml") == []
    bad = tmp_path / "bad.yaml"
    bad.write_text("images: [ {model: x\n")
    with pytest.raises(si.SiteImagesError, match=rf"^\[yeto\] site image catalog: {bad} is not valid YAML"):
        si.load(bad)
    bad.write_text(yaml.safe_dump({"version": 1, "images": [record(), record(role="pilot")]}))
    with pytest.raises(si.SiteImagesError, match=rf"^\[yeto\] site image catalog: {bad}: images\[1\]: role 'pilot'"):
        si.load(bad)
    bad.write_text(yaml.safe_dump(["not", "a", "mapping"]))
    with pytest.raises(si.SiteImagesError, match="must be a mapping"):
        si.load(bad)
    for version in (0, "1", True):
        bad.write_text(yaml.safe_dump({"version": version, "images": []}))
        with pytest.raises(si.SiteImagesError, match="version must be a positive integer"):
            si.load(bad)


def test_a_newer_catalog_and_unknown_top_level_keys_are_read_with_a_warning(tmp_path, capsys):
    path = write_catalog(tmp_path / "c.yaml", record())
    path.write_text(yaml.safe_dump({"version": 2, "images": [record()], "extra": 1}))
    assert [r.ref for r in si.load(path)] == [REF1]
    err = capsys.readouterr().err
    assert f"WARNING: {path}: version 2 is newer than this reader's 1; read as far as it goes" in err
    assert f"WARNING: {path}: unknown top-level key(s) ['extra'] ignored" in err


def test_one_image_adopted_on_two_kinds_of_card_keeps_two_records(tmp_path):
    path = tmp_path / "site-images.yaml"
    si.add(record(), path)
    si.add(record(card_archs=["sm100"]), path)
    si.add(record(cpu_arch="aarch64"), path)
    si.add(record(card_archs=["sm100"], evidence={"at": "2026-11-27T00:00:00Z", "node": "b200-0", "seconds": 2,
                                                  "result": "pass"}), path)
    assert [(r.card_archs, r.cpu_arch, r.evidence["node"]) for r in si.load(path)] == [
        (("sm103",), "x86_64", "b300-0"), (("sm103",), "aarch64", "b300-0"), (("sm100",), "x86_64", "b200-0")]


def test_the_catalog_keeps_its_mode_and_a_new_one_is_0644(tmp_path):
    import stat

    path = tmp_path / "site-images.yaml"
    si.add(record(), path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    path.chmod(0o640)
    si.add(record(ref=REF2), path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


def test_the_environment_moves_the_default_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv(si.PATH_ENV, str(tmp_path / "elsewhere" / "c.yaml"))
    assert si.default_path() == tmp_path / "elsewhere" / "c.yaml"
    si.add(record())
    assert [r.ref for r in si.load()] == [REF1] and (tmp_path / "elsewhere" / "c.yaml").exists()
    monkeypatch.setenv(si.PATH_ENV, " ")
    assert si.default_path() == tmp_path / "home" / ".yeto" / "site-images.yaml"


def test_add_replaces_the_same_image_keeps_the_others_and_never_overwrites_a_broken_catalog(tmp_path):
    path = tmp_path / "sub" / "site-images.yaml"
    si.add(record(), path)
    si.add(record(ref=REF2), path)
    si.add(record(evidence={"at": "2026-11-27T00:00:00Z", "node": "b300-1", "seconds": 9, "result": "pass"}), path)
    records = si.load(path)
    assert [(r.ref, r.evidence["node"]) for r in records] == [(REF2, "b300-0"), (REF1, "b300-1")]
    with pytest.raises(si.SiteImagesError):
        si.add(record(status="verified-smoke", evidence={}), path)
    assert si.load(path) == records  # an invalid record writes nothing
    path.write_text("images: [")
    with pytest.raises(si.SiteImagesError, match="not valid YAML"):
        si.add(record(ref=REF3), path)
    assert path.read_text() == "images: ["
    leftovers = [p.name for p in path.parent.iterdir() if p.name.startswith(".")]
    assert leftovers == [si.lock_path(path).name]  # no temporary file left, only the lock


def _add_one(path: str, index: int) -> None:
    si.add(record(ref=f"registry.example/site/miles-new@sha256:{index:064x}"), path)


def test_concurrent_adds_from_separate_processes_keep_every_record(tmp_path):
    import multiprocessing  # noqa: PLC0415

    path = tmp_path / "site-images.yaml"
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=_add_one, args=(str(path), i)) for i in range(8)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(60)
    assert [proc.exitcode for proc in procs] == [0] * 8
    assert sorted(r.ref for r in si.load(path)) == sorted(
        f"registry.example/site/miles-new@sha256:{i:064x}" for i in range(8))


def test_an_add_waits_for_the_catalog_lock(tmp_path):
    import threading  # noqa: PLC0415

    path = tmp_path / "site-images.yaml"
    si.add(record(), path)
    done = threading.Event()

    def writer():
        si.add(record(ref=REF2), path)
        done.set()

    with si.locked(path) as held:
        assert held == path
        thread = threading.Thread(target=writer)
        thread.start()
        assert not done.wait(0.5)  # blocked on the lock another holder has
        assert [r.ref for r in si.load(path)] == [REF1]
    thread.join(10)
    assert done.is_set() and [r.ref for r in si.load(path)] == [REF1, REF2]


def test_save_flushes_the_file_and_its_directory(tmp_path, monkeypatch):
    synced = []
    real = si.os.fsync
    monkeypatch.setattr(si.os, "fsync", lambda fd: (synced.append(fd), real(fd))[1])
    si.save([si.parse_record(record())], tmp_path / "site-images.yaml")
    assert len(synced) == 2  # the temporary file before the rename, the directory after


# -- matching (7.1) ------------------------------------------------------------------


def test_match_takes_the_newest_verified_record_that_fits_every_node(tmp_path):
    older = record(ref=REF1, evidence={"at": "2026-11-20T00:00:00Z", "node": "b300-0", "seconds": 1,
                                       "result": "pass"})
    newer = record(ref=REF2, evidence={"at": "2026-11-26T00:00:00Z", "node": "b300-0", "seconds": 1,
                                       "result": "pass"})
    path = write_catalog(tmp_path / "c.yaml", newer, older,
                         record(ref=REF3, status="retired", evidence={"at": "2027-01-01T00:00:00Z"}),
                         record(ref=REF3, model="Qwen/Qwen3-8B"),
                         record(ref=REF3, card_archs=["sm90"]),
                         record(ref=REF3, cpu_arch="aarch64"),
                         record(ref=REF3, role="sampler", engine="vllm"))
    assert [r.ref for r in si.match("kimi-k3.1-pro", "trainer", b300(), path=path)] == [REF2, REF1]
    assert [r.ref for r in si.match("moonshotai/Kimi-K3.1-Pro", "trainer", [b300(), b300("b300-1")],
                                    path=path)] == [REF2, REF1]
    # one node of another card: no record fits the whole island
    h200 = b300("h200-0", card="H200", hbm_gib=141)
    assert si.match("kimi_k31", "trainer", [b300(), h200], path=path) == []
    assert si.match("kimi_k31", "trainer", b300(), engine="verl", path=path) == []
    assert si.match(None, "trainer", b300(), path=path) == []
    assert si.match("kimi_k31", "trainer", [], path=path) == []  # no node: nothing to fit
    assert [r.ref for r in si.match("kimi_k31", "sampler", b300(), path=path)] == [REF3]
    assert [r.ref for r in si.match("kimi-k31", "trainer", b300(), path=path)] == [REF2, REF1]  # an alias
    assert [r.ref for r in si.match("qwen3-8b", "trainer", b300(), path=path)] == [REF3]  # yeto.models alias


def test_records_of_one_date_prefer_the_later_entry(tmp_path):
    path = write_catalog(tmp_path / "c.yaml", record(ref=REF1), record(ref=REF2))
    assert [r.ref for r in si.match("kimi_k31", "trainer", b300(), path=path)] == [REF2, REF1]


def test_a_record_names_why_it_does_not_fit():
    r = si.parse_record(record(card_archs=["sm90"], cpu_arch="aarch64"))
    assert r.problems(b300()) == ["it is a aarch64 image, node b300-0 is x86_64",
                                  "it was smoked on sm90 cards, node b300-0 has sm103"]


# -- selection order (7.4) -----------------------------------------------------------


def test_select_order_explicit_then_site_then_default(tmp_path):
    path = write_catalog(tmp_path / "c.yaml", record())
    ref, source = si.select("docker:mine@sha256:" + D3, "kimi_k31", "trainer", b300(), path=path, default="d")
    assert (ref, source) == ("docker:mine@sha256:" + D3, si.SOURCE_EXPLICIT)
    choice = si.select(None, "kimi_k31", "trainer", b300(), path=path, default="d")
    assert tuple(choice) == (REF1, si.SOURCE_SITE)
    assert choice.line("trainer") == (f"trainer image {REF1} (adopted 2026-11-26T10:00:00Z on b300-0) "
                                      f"(source: site image catalog {path})")
    calls = []
    choice = si.select(None, "Qwen/Qwen3-8B", "trainer", b300(), path=path,
                       default=lambda: calls.append(1) or "builtin")
    assert tuple(choice) == ("builtin", si.SOURCE_DEFAULT) and calls == [1]
    assert choice.line("trainer") == (f"trainer image builtin (source: built-in default; no verified-smoke "
                                      f"record in {path} for qwen/qwen3-8b trainer on these nodes)")
    missing = si.select(None, "kimi_k31", "trainer", b300(), path=tmp_path / "none.yaml", default="d")
    assert missing.source == si.SOURCE_DEFAULT and "no catalog at" in missing.why


def test_a_fallback_names_the_closest_records_and_why_they_do_not_fit(tmp_path):
    path = write_catalog(tmp_path / "c.yaml", record(card_archs=["sm90"]),
                         record(ref=REF2, cpu_arch="aarch64",
                                evidence={"at": "2026-11-27T00:00:00Z", "node": "g", "seconds": 1, "result": "pass"}),
                         record(ref=REF3, model="Qwen/Qwen3-8B"))
    choice = si.select(None, "kimi_k31", "trainer", b300(), engine="miles", path=path, default="d")
    assert choice.source == si.SOURCE_DEFAULT
    assert choice.why.endswith(f"(closest: {REF2}: it is a aarch64 image, node b300-0 is x86_64 | {REF1}: it was "
                               "smoked on sm90 cards, node b300-0 has sm103)")
    other = si.select(None, "kimi_k31", "trainer", b300(), engine="verl", path=path, default="d")
    assert f"{REF1}: it is a miles image, not verl; it was smoked on sm90" in other.why


def test_an_explicit_image_never_reads_the_catalog(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("images: [")
    assert tuple(si.select("x@sha256:" + D1, "kimi_k31", "trainer", b300(), path=bad)) == \
        ("x@sha256:" + D1, si.SOURCE_EXPLICIT)
    with pytest.raises(si.SiteImagesError):
        si.select(None, "kimi_k31", "trainer", b300(), path=bad)


def test_the_default_path_is_under_home_and_the_inventory_may_move_it(tmp_path):
    assert si.default_path() == tmp_path / "home" / ".yeto" / "site-images.yaml"
    inv = tmp_path / "cluster" / "inventory.yaml"
    inv.parent.mkdir()
    inv.write_text(yaml.safe_dump({"site_images": "images.yaml", "nodes": []}))
    assert si.inventory_catalog_path(inv) == tmp_path / "cluster" / "images.yaml"
    inv.write_text(yaml.safe_dump({"site_images": "~/elsewhere.yaml", "nodes": []}))
    assert si.inventory_catalog_path(inv) == tmp_path / "home" / "elsewhere.yaml"
    inv.write_text(yaml.safe_dump({"nodes": []}))
    assert si.inventory_catalog_path(inv) == si.default_path()
    assert si.inventory_catalog_path(None) == si.default_path()


def test_the_inventory_accepts_site_images_and_refuses_a_non_path():
    node = {"name": "b300-0", "host": "b300-0", "device_family": "cuda", "cpu_arch": "x86_64", "card": "B300",
            "cards_per_node": 8, "hbm_gib": 288, "host_memory_gib": 2048}
    assert [n.name for n in parse_inventory({"site_images": "/etc/yeto/images.yaml", "nodes": [node]})] == ["b300-0"]
    with pytest.raises(InventoryError, match="site_images: must be a file path"):
        parse_inventory({"site_images": ["a"], "nodes": [node]})
    with pytest.raises(InventoryError,
                       match=r"known: \['images', 'lease_store', 'node_health', 'nodes', 'site_images'\]"):
        parse_inventory({"lease": "x", "nodes": [node]})


# -- the ssh: trainer island (7.4) ---------------------------------------------------


def _inventory(tmp_path, *, catalog=None, image=None, nodes=1):
    entries = []
    for i in range(nodes):
        entry = {"name": f"b300-{i}", "host": f"b300-{i}", "device_family": "cuda", "cpu_arch": "x86_64",
                 "card": "B300", "cards_per_node": 8, "hbm_gib": 288, "host_memory_gib": 2048,
                 "runtime": "docker", "docker_args": ["--gpus", "all"]}
        if image:
            entry["image"] = image
        entries.append(entry)
    doc = {"nodes": entries}
    if catalog:
        doc["site_images"] = str(catalog)
    path = tmp_path / "inventory.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def _args(inventory, **kw):
    base = {"training_mode": "rl", "rl_backend": None, "rl_image": None, "gpu": "ssh:2x8xB300",
            "rl_node_inventory": str(inventory), "model": "moonshotai/Kimi-K3.1-Pro", "rl_model_recipe": None,
            "tuning": None}
    return argparse.Namespace(**{**base, **kw})


def test_an_ssh_trainer_island_without_an_image_takes_the_site_record(tmp_path, capsys):
    from yeto.launch.rl_backend import resolve_default_rl_image

    catalog = write_catalog(tmp_path / "c.yaml", record())
    args = _args(_inventory(tmp_path, catalog=catalog, nodes=2))
    resolve_default_rl_image(args)
    assert args.rl_image == "docker:" + REF1
    err = capsys.readouterr().err
    assert f"[launcher] ssh trainer image {REF1} (adopted 2026-11-26T10:00:00Z on b300-0) (source: site image " \
           f"catalog {catalog})" in err


def test_no_site_record_falls_back_to_the_built_in_default_and_says_so(tmp_path, capsys):
    from yeto.launch.rl_backend import _builtin_rl_image, resolve_default_rl_image

    catalog = write_catalog(tmp_path / "c.yaml", record(model="Qwen/Qwen3-8B"))
    args = _args(_inventory(tmp_path, catalog=catalog))
    resolve_default_rl_image(args)
    assert args.rl_image == _builtin_rl_image(_args(tmp_path))
    assert "(source: built-in default; no verified-smoke record in" in capsys.readouterr().err


def test_a_mixed_fleet_takes_the_built_in_default_and_says_why(tmp_path, capsys):
    from yeto.launch.rl_backend import MIXED_FLEET, _builtin_rl_image, resolve_default_rl_image

    catalog = write_catalog(tmp_path / "c.yaml", record())  # it would match the ssh: nodes
    args = _args(_inventory(tmp_path, catalog=catalog), gpu="ssh:2x8xB300,modal:1x8xB300")
    resolve_default_rl_image(args)
    assert args.rl_image == _builtin_rl_image(args)
    assert capsys.readouterr().err == (f"[launcher] trainer image {args.rl_image} (source: built-in default; "
                                       f"{MIXED_FLEET})\n")
    both = _args(_inventory(tmp_path, catalog=catalog), gpu="ssh:1x8xB300,ssh:1x8xB300")
    resolve_default_rl_image(both)
    assert both.rl_image == "docker:" + REF1  # every entry ssh: the catalog is read


@pytest.mark.parametrize("kw", [
    {"rl_image": "docker:ghcr.io/x/explicit@sha256:" + D3},  # --rl-image (the customer's configs)
    {"gpu": "modal:1x8xB300"},                              # not an ssh: island
    {"rl_node_inventory": None},
])
def test_an_explicit_image_or_another_island_is_unchanged_and_reads_no_catalog(tmp_path, capsys, kw):
    from yeto.launch.rl_backend import _builtin_rl_image, resolve_default_rl_image

    bad = tmp_path / "bad.yaml"
    bad.write_text("images: [")  # would refuse the launch if it were read
    args = _args(_inventory(tmp_path, catalog=bad), **kw)
    expect = kw.get("rl_image") or _builtin_rl_image(args)
    resolve_default_rl_image(args)
    assert args.rl_image == expect
    assert capsys.readouterr().err == ""


def test_an_inventory_image_is_explicit(tmp_path, capsys):
    from yeto.launch.rl_backend import _builtin_rl_image, resolve_default_rl_image

    bad = tmp_path / "bad.yaml"
    bad.write_text("images: [")
    args = _args(_inventory(tmp_path, catalog=bad, image="ghcr.io/x/inv@sha256:" + D3))
    resolve_default_rl_image(args)
    assert args.rl_image == _builtin_rl_image(args)  # unchanged: run_image compares the two as before
    assert capsys.readouterr().err == ""


def test_a_broken_catalog_refuses_an_island_that_reads_it(tmp_path):
    from yeto.launch.rl_backend import resolve_default_rl_image

    bad = tmp_path / "bad.yaml"
    bad.write_text("images: [")
    with pytest.raises(ValueError, match="not valid YAML"):
        resolve_default_rl_image(_args(_inventory(tmp_path, catalog=bad)))


# -- the ssh sampler pool (7.4) ------------------------------------------------------


def _pool(catalog, *, image="auto", node_image=None):
    node = b300(**({"image": node_image} if node_image else {}))
    sec = {"kind": "ssh", "nodes": ["b300-0"], "workdir": "/srv/yeto/run0", "runtime": "container",
           "container": {"engine": "docker", "image": image}, "store": "/srv/yeto/run0/store"}
    conf = sp.parse_config({"provider": sec}, inventory=[node])
    return sp.SshPoolConf(**{**conf.__dict__, "site_images": str(catalog)})


def test_an_auto_pool_node_takes_the_site_sampler_record_and_the_dry_run_names_it(tmp_path, capsys):
    spec = json.loads(K31_SPEC.read_text())
    catalog = write_catalog(tmp_path / "c.yaml", record(role="sampler", engine="vllm", ref=REF3))
    conf = _pool(catalog)
    assert sp.node_image(conf, conf.nodes[0], spec) == REF3
    assert sp.image_problems(conf, spec) == []  # a site image is the site's
    assert f"ssh provider: node b300-0: sampler image {REF3} (adopted" in capsys.readouterr().out


def test_an_auto_pool_without_a_site_record_keeps_the_engine_catalog_choice(tmp_path, capsys):
    from yeto.rl import sampler_pins as pins

    spec = json.loads(K31_SPEC.read_text())
    restart = dict(spec, engine=dict(spec["engine"], server=dict(spec["engine"]["server"], in_place_reload=False)))
    conf = _pool(tmp_path / "none.yaml")
    assert sp.node_image(conf, conf.nodes[0], restart) == pins.VLLM_SAMPLER_IMAGE_CUDA.removeprefix("docker:")
    assert sp.image_problems(conf, restart) == []
    assert "(source: built-in default; no catalog at" in capsys.readouterr().out


def test_a_named_pool_image_reads_no_catalog_and_prints_nothing_new(tmp_path, capsys):
    spec = json.loads(K31_SPEC.read_text())
    bad = tmp_path / "bad.yaml"
    bad.write_text("images: [")
    for conf in (_pool(bad, image="ghcr.io/x/pool@sha256:" + D1),
                 _pool(bad, node_image="ghcr.io/x/node@sha256:" + D2)):
        assert sp.image_problems(conf, spec) == []
    assert sp.node_image(_pool(bad, image="ghcr.io/x/pool@sha256:" + D1), b300(), spec) == "ghcr.io/x/pool@sha256:" + D1
    assert capsys.readouterr().out == ""
    problems = sp.image_problems(_pool(bad), spec)
    assert len(problems) == 1 and "not valid YAML" in problems[0]


def test_the_pool_matches_the_site_record_against_all_its_nodes(tmp_path, capsys):
    spec = json.loads(K31_SPEC.read_text())
    spec = dict(spec, engine=dict(spec["engine"], server=dict(spec["engine"]["server"], in_place_reload=False)))
    catalog = write_catalog(tmp_path / "c.yaml", record(role="sampler", engine="vllm", ref=REF3))
    nodes = [b300(), b300("h200-0", card="H200", hbm_gib=141)]
    sec = {"kind": "ssh", "nodes": ["b300-0", "h200-0"], "workdir": "/srv/yeto/run0", "runtime": "container",
           "container": {"engine": "docker", "image": "auto"}, "store": "/srv/yeto/run0/store"}
    conf = sp.SshPoolConf(**{**sp.parse_config({"provider": sec}, inventory=nodes).__dict__,
                             "site_images": str(catalog)})
    choices = sp.pool_image_choices(conf, spec)
    # the record fits b300-0 only: a spare of the other card would start another image, so none takes it
    assert choices["b300-0"].source == si.SOURCE_DEFAULT
    assert sp.node_image_choice(conf, nodes[0], spec).source == si.SOURCE_DEFAULT
    alone = sp.SshPoolConf(**{**conf.__dict__, "nodes": (nodes[0],)})
    assert sp.node_image(alone, nodes[0], spec) == REF3


def test_the_pool_fixes_its_images_at_the_first_start_and_keeps_them(tmp_path, monkeypatch):
    spec = json.loads(K31_SPEC.read_text())
    catalog = write_catalog(tmp_path / "c.yaml", record(role="sampler", engine="vllm", ref=REF3))
    conf = _pool(catalog)
    monkeypatch.setattr(sp, "spec_of", lambda cfg: spec)

    def ops():
        return _pool_ops(conf, tmp_path)

    first = ops()
    assert first.image_for(conf.nodes[0]) == REF3
    newer = record(role="sampler", engine="vllm", ref=f"registry.example/site/vllm-newer@sha256:{D1}",
                   evidence={"at": "2027-01-01T00:00:00Z", "node": "b300-0", "seconds": 1, "result": "pass"})
    write_catalog(catalog, record(role="sampler", engine="vllm", ref=REF3), newer)
    assert first.image_for(conf.nodes[0]) == REF3  # not read again in the run
    kept = json.loads(first.images_path().read_text())
    assert kept["images"] == {"b300-0": {"ref": REF3, "source": si.SOURCE_SITE}}
    logs = []
    monkeypatch.setattr(sp, "_log", logs.append)
    assert ops().image_for(conf.nodes[0]) == REF3  # a restarted controller starts the same image
    assert logs == [f"ssh pool: node b300-0: sampler image {REF3} (source: site image catalog; kept from the "
                    "pool's start (ssh-pool-run1-images.json))"]
    first.images_path().unlink()
    assert ops().image_for(conf.nodes[0]) == newer["ref"]  # a new pool takes the newest record
    first.images_path().unlink()
    assert first.image_for(conf.nodes[0]) == REF3  # chosen once per pool process, whatever the state file says


def test_the_pool_reads_the_catalog_its_inventory_names(tmp_path, monkeypatch):
    monkeypatch.setattr(sp, "load_profiles", lambda _p: [b300()])
    inv = tmp_path / "inventory.yaml"
    inv.write_text(yaml.safe_dump({"site_images": "images.yaml", "nodes": []}))
    sec = {"kind": "ssh", "inventory": str(inv), "nodes": ["b300-0"], "workdir": "/srv/yeto/run0",
           "runtime": "container", "container": {"engine": "docker", "image": "auto"}}
    assert sp.parse_config({"provider": sec}).site_images == str(tmp_path / "images.yaml")


def test_an_unquoted_yaml_timestamp_reads_as_its_iso_text(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "images": [record()]}).replace(
        "'2026-11-26T10:00:00Z'", "2026-11-26T10:00:00Z"))
    assert "at: 2026-11-26T10:00:00Z" in path.read_text()
    (r,) = si.load(path)
    assert r.at == "2026-11-26T10:00:00Z"


def _pool_ops(conf, tmp_path):
    o = sp.SshSamplerPoolOps.__new__(sp.SshSamplerPoolOps)
    o.cfg, o.conf, o.run_id, o.state_dir, o._images = {}, conf, "run1", tmp_path / "state", None
    import threading

    o._images_lock = threading.Lock()
    return o


def test_parallel_starts_choose_the_pool_images_once(tmp_path, monkeypatch):
    import threading
    import time as _time

    spec = json.loads(K31_SPEC.read_text())
    monkeypatch.setattr(sp, "spec_of", lambda cfg: spec)
    conf = _pool(write_catalog(tmp_path / "c.yaml", record(role="sampler", engine="vllm", ref=REF3)))
    calls = []

    def choices(conf_, spec_, *, only=None):  # each call would see a newer catalog (an adopt in between)
        calls.append(1)
        _time.sleep(0.2)
        ref = f"registry.example/site/vllm-{len(calls)}@sha256:{D1}"
        return {"b300-0": si.Choice(ref, si.SOURCE_SITE, why="test")}

    monkeypatch.setattr(sp, "pool_image_choices", choices)
    ops = _pool_ops(conf, tmp_path)
    barrier, got, errors = threading.Barrier(6), [], []

    def start():
        barrier.wait()
        try:
            got.append(ops.image_for(conf.nodes[0]))
        except Exception as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=start) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert errors == [] and len(calls) == 1 and set(got) == {f"registry.example/site/vllm-1@sha256:{D1}"}
    kept = json.loads(ops.images_path().read_text())
    assert kept["images"]["b300-0"]["ref"] == got[0]  # the state file names the image the nodes run
    assert [p.name for p in ops.images_path().parent.iterdir()] == [ops.images_path().name]  # no temporary file


def _kept(ops, spec, images, **key):
    from yeto.launch import engine_images

    needs = engine_images.spec_needs(spec)
    doc = {"key": {"model": needs.get("model"), "engine": needs.get("engine_pref"), "catalog": ops.conf.site_images,
                   **key}, "images": images}
    ops.images_path().parent.mkdir(parents=True, exist_ok=True)
    ops.images_path().write_text(json.dumps(doc))


def test_a_kept_image_never_replaces_a_named_one_or_another_model(tmp_path, monkeypatch):
    spec = json.loads(K31_SPEC.read_text())
    monkeypatch.setattr(sp, "spec_of", lambda cfg: spec)
    catalog = write_catalog(tmp_path / "c.yaml", record(role="sampler", engine="vllm", ref=REF3))
    kept = {"b300-0": {"ref": f"registry.example/site/kept@sha256:{D2}", "source": si.SOURCE_SITE}}
    named = _pool_ops(_pool(catalog, image=f"ghcr.io/x/pool@sha256:{D1}"), tmp_path)
    _kept(named, spec, kept)
    assert named.image_for(named.conf.nodes[0]) == f"ghcr.io/x/pool@sha256:{D1}"  # the configuration names it
    auto = _pool_ops(_pool(catalog), tmp_path / "b")
    _kept(auto, spec, kept)
    assert auto.image_for(auto.conf.nodes[0]) == kept["b300-0"]["ref"]  # the same pool: its start's image
    other = _pool_ops(_pool(catalog), tmp_path / "c")
    _kept(other, spec, kept, model="Qwen/Qwen3-8B")
    assert other.image_for(other.conf.nodes[0]) == REF3  # kept for another model: chosen again


def test_a_record_of_another_device_family_does_not_fit():
    from types import SimpleNamespace

    r = si.parse_record(record(device_family="cuda"))
    ppu = SimpleNamespace(name="p0", family="ppu", cpu_arch="x86_64", card_arch="sm103")
    assert r.problems(ppu) == ["it is a cuda image, node p0 is ppu"]
    assert si.parse_record(record(device_family=None)).problems(ppu) == []


def test_the_trainer_match_takes_the_spares_too(tmp_path, capsys):
    from yeto.launch.rl_backend import _builtin_rl_image, resolve_default_rl_image

    inv = _inventory(tmp_path, catalog=write_catalog(tmp_path / "c.yaml", record()), nodes=2)
    doc = yaml.safe_load(inv.read_text())
    doc["nodes"][0]["roles"] = ["trainer"]
    doc["nodes"][1].update(roles=["trainer-spare"], card="H200", hbm_gib=141)  # the record does not fit the spare
    inv.write_text(yaml.safe_dump(doc))
    args = _args(inv)
    resolve_default_rl_image(args)
    assert args.rl_image == _builtin_rl_image(args)
    assert "b300-1 has sm90" in capsys.readouterr().err


def test_the_kept_file_is_written_through_a_temporary_file_of_its_own(tmp_path, monkeypatch):
    spec = json.loads(K31_SPEC.read_text())
    monkeypatch.setattr(sp, "spec_of", lambda cfg: spec)
    ops = _pool_ops(_pool(write_catalog(tmp_path / "c.yaml", record(role="sampler", engine="vllm", ref=REF3))),
                    tmp_path)
    ops.images_path().with_suffix(".tmp").mkdir(parents=True)  # another writer's fixed temporary path
    assert ops.image_for(ops.conf.nodes[0]) == REF3
    assert json.loads(ops.images_path().read_text())["images"]["b300-0"]["ref"] == REF3


def test_a_spare_naming_its_own_image_does_not_turn_the_catalog_off(tmp_path, capsys):
    from yeto.launch.rl_backend import resolve_default_rl_image

    inv = _inventory(tmp_path, catalog=write_catalog(tmp_path / "c.yaml", record()), nodes=2)
    doc = yaml.safe_load(inv.read_text())
    doc["nodes"][0]["roles"] = ["trainer"]
    doc["nodes"][1].update(roles=["trainer-spare"], image="registry.example/site/spare@sha256:" + "e" * 64)
    inv.write_text(yaml.safe_dump(doc))
    args = _args(inv)
    resolve_default_rl_image(args)
    assert args.rl_image == "docker:" + record()["ref"]  # the island's node names no image: the site record
    doc["nodes"][0]["image"] = doc["nodes"][1].pop("image")  # the island's own node names one: no catalog
    inv.write_text(yaml.safe_dump(doc))
    args = _args(inv)
    resolve_default_rl_image(args)
    assert args.rl_image != "docker:" + record()["ref"]
