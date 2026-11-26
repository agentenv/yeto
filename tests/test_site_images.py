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
    ({"ref": "registry.example/site/miles-new:latest"}, "must be pinned by digest"),
    ({"cpu_arch": "riscv64"}, "cpu_arch 'riscv64' is unknown"),
    ({"card_archs": []}, "card_archs must be a non-empty list"),
    ({"card_archs": "sm103"}, "card_archs must be a non-empty list"),
    ({"status": "smoked"}, "status 'smoked' is unknown"),
    ({"evidence": {"at": "x", "node": "n"}}, r"needs evidence \['seconds', 'result'\]"),
    ({"evidence": {"at": "x", "node": "n", "seconds": 1, "result": "fail"}}, "needs evidence.result pass"),
    ({"colour": "red"}, r"unknown key\(s\) \['colour'\]"),
    ({"model": None}, r"missing key\(s\) \['model'\]"),
    ({"device_family": "quantum"}, "device family 'quantum' is unknown"),
])
def test_bad_records_are_refused_by_key(change, message):
    raw = record(**{k: v for k, v in change.items() if v is not None})
    if change.get("model", 1) is None:
        raw.pop("model")
    with pytest.raises(si.SiteImagesError, match=message):
        si.parse_record(raw, where="images[0]")


def test_a_retired_record_needs_no_evidence():
    assert si.parse_record(record(status="retired", evidence=None)).status == "retired"


def test_a_missing_catalog_is_empty_and_a_broken_one_names_its_path(tmp_path):
    assert si.load(tmp_path / "none.yaml") == []
    bad = tmp_path / "bad.yaml"
    bad.write_text("images: [ {model: x\n")
    with pytest.raises(si.SiteImagesError, match=f"{bad} is not valid YAML"):
        si.load(bad)
    bad.write_text(yaml.safe_dump({"version": 1, "images": [record(), record(role="pilot")]}))
    with pytest.raises(si.SiteImagesError, match=rf"{bad}: images\[1\]: role 'pilot'"):
        si.load(bad)
    bad.write_text(yaml.safe_dump({"version": 2, "images": []}))
    with pytest.raises(si.SiteImagesError, match="version 2 is not 1"):
        si.load(bad)
    bad.write_text(yaml.safe_dump(["not", "a", "mapping"]))
    with pytest.raises(si.SiteImagesError, match="must be a mapping"):
        si.load(bad)
    bad.write_text(yaml.safe_dump({"version": 1, "images": [], "extra": 1}))
    with pytest.raises(si.SiteImagesError, match=r"unknown top-level key\(s\) \['extra'\]"):
        si.load(bad)


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
    assert [r.ref for r in si.match("kimi_k31", "sampler", b300(), path=path)] == [REF3]


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
    with pytest.raises(InventoryError, match=r"known: \['images', 'node_health', 'nodes', 'site_images'\]"):
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
