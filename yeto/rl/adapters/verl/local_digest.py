"""Light policy digest of a verl Megatron full-parameter island (S28 G-LHASH).

The full check of a no-sync full-parameter island
(:meth:`.full_parameter_stream.VerlChunkedFullParameterAdapter.fragment_digests`)
rebuilds every HF tensor: per fragment it copies the FP32 masters to the
card, gathers the TP/EP/PP/DP shards with HCCL, converts them to HF and
hashes the fragment on rank 0, one fragment after another.  For the 120B
model that is about 250 fragments, about 27.7 min per check
(infra-drafts/SESSION28/G-120C.md section 30).

The light digest reads the same FP32 masters but moves nothing between
ranks: every rank hashes the master shards it owns, on the host, in a thread
pool, and returns one small row per shard.  The island driver sorts the rows
of all ranks and hashes them into one global digest.

Definition (``LIGHT_DIGEST_VERSION``):

* shard digest = ``H(prefix, dtype, numel, H(block 0), H(block 1), ...)``,
  blocks of ``BLOCK_BYTES`` raw FP32 bytes (fixed, so the value does not
  depend on the thread count);
* row = ``[key, start, stop, numel_of_parameter, shard digest]``; ``key`` is
  ``"<chunk>:<megatron name>@tp<t>.pp<p>[.ep<e>]"`` (the
  :func:`.megatron_ports.owned_parameters` key), ``[start, stop)`` the
  flat range of the parameter that the shard holds;
* global digest = ``SHA-256(prefix, algorithm, topology, sorted rows)`` (64 hex).

Invariants (tested in ``tests/test_s28_light_digest.py``):

1. Same weights, same topology: same digest (rows are sorted, so the order
   of ranks and of threads does not matter).
2. One changed FP32 element changes the digest (it changes one block, so
   one shard digest, so the row list).
3. The rows of one parameter key tile ``[0, numel)`` exactly; the combine
   step refuses a gap, an overlap or two different digests for one range
   (a replicated master that diverged).
4. The topology (TP/PP/EP/DP/CP sizes, model chunk count) is part of the
   digest.  So the light digest of one model under two different splits is
   different by construction: it is an identity *within one run's
   topology*, not across splits.  The HF-canonical hash (the full check)
   stays the identity across splits and engines; the driver still runs it
   on the first round, on every round that keeps a round cut and on the last
   round (:func:`yeto.rl.engine.driver.verify_publication`).

Hash: BLAKE3 when the ``blake3`` module imports, else XXH3-128 (``xxhash``),
else SHA-256; the algorithm name is in the digest, so two runs that use
different algorithms never compare equal by accident.  Switch:
``YETO_RL_LIGHT_DIGEST`` (default on; ``0`` keeps the full check on every
round), ``YETO_RL_LIGHT_DIGEST_THREADS`` (default 16).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

LIGHT_DIGEST_VERSION = "yeto-verl-megatron-light-digest-v1"
BLOCK_BYTES = 64 << 20
DEFAULT_THREADS = 16
HOST_COPY_BUDGET = 2 << 30  # bytes of card-resident shards copied to the host at once


class LightDigestError(RuntimeError):
    """The per-rank rows do not combine into one digest (gap, overlap, divergence)."""


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    raw = (os.environ if environ is None else environ).get("YETO_RL_LIGHT_DIGEST", "1")
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def thread_count(environ: Mapping[str, str] | None = None) -> int:
    raw = (os.environ if environ is None else environ).get("YETO_RL_LIGHT_DIGEST_THREADS", "")
    return max(1, int(raw)) if raw.strip() else DEFAULT_THREADS


# ----------------------------------------------------------------------------- hashing
def algorithm(prefer: Sequence[str] = ("blake3", "xxh3_128", "sha256")) -> str:
    """The first hash in ``prefer`` whose module imports."""

    for name in prefer:
        if name == "blake3":
            try:
                import blake3  # noqa: F401
            except ImportError:
                continue
            return name
        if name == "xxh3_128":
            try:
                import xxhash  # noqa: F401
            except ImportError:
                continue
            return name
        if name == "sha256":
            return name
    raise ValueError(f"no usable hash in {tuple(prefer)}")


def _hasher(name: str):
    if name == "blake3":
        import blake3

        return blake3.blake3(max_threads=1)
    if name == "xxh3_128":
        import xxhash

        return xxhash.xxh3_128()
    if name == "sha256":
        return hashlib.sha256()
    raise ValueError(f"unknown hash {name!r}")


def hash_bytes(name: str, data) -> bytes:
    h = _hasher(name)
    h.update(data)
    return h.digest()


def block_views(raw: memoryview) -> list[memoryview]:
    """``raw`` (bytes) cut into ``BLOCK_BYTES`` blocks; one empty block for empty input."""

    if len(raw) == 0:
        return [raw]
    return [raw[i:i + BLOCK_BYTES] for i in range(0, len(raw), BLOCK_BYTES)]


def shard_digest(name: str, dtype: str, numel: int, block_digests: Iterable[bytes]) -> str:
    h = _hasher(name)
    h.update(f"{LIGHT_DIGEST_VERSION}\0shard\0{dtype}\0{int(numel)}\0".encode())
    for block in block_digests:
        h.update(block)
    return h.hexdigest()


def hash_shards(name: str, shards: Sequence[tuple[str, int, memoryview]], *, threads: int) -> list[str]:
    """Shard digests of ``(dtype, numel, raw bytes)``, blocks hashed in ``threads`` threads
    (hashlib, blake3 and xxhash release the GIL on large buffers)."""

    from concurrent.futures import ThreadPoolExecutor

    jobs = [(index, block) for index, (_d, _n, raw) in enumerate(shards) for block in block_views(raw)]
    if threads <= 1 or len(jobs) <= 1:
        results = [hash_bytes(name, block) for _i, block in jobs]
    else:
        with ThreadPoolExecutor(max_workers=threads, thread_name_prefix="yeto-light-hash") as pool:
            results = list(pool.map(lambda job: hash_bytes(name, job[1]), jobs))
    per_shard: list[list[bytes]] = [[] for _ in shards]
    for (index, _block), digest in zip(jobs, results, strict=True):
        per_shard[index].append(digest)
    return [shard_digest(name, dtype, numel, per_shard[i]) for i, (dtype, numel, _r) in enumerate(shards)]


# ----------------------------------------------------------------------------- combine (island driver)
def combine(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One global digest from every rank's record of :func:`remote_light_digest`.

    Returns ``{"digest", "algorithm", "topology", "rows", "bytes", "seconds"}``.
    Raises :class:`LightDigestError` when the ranks disagree on the algorithm or
    the topology, when one range has two digests, or when a parameter's ranges
    do not tile it."""

    if not records:
        raise LightDigestError("no rank returned a light digest")
    algorithms = {str(r["algorithm"]) for r in records}
    topologies = {json.dumps(r["topology"], sort_keys=True) for r in records}
    if len(algorithms) != 1:
        raise LightDigestError(f"ranks hash with different algorithms: {sorted(algorithms)}")
    if len(topologies) != 1:
        raise LightDigestError(f"ranks report different topologies: {sorted(topologies)}")
    unique: dict[tuple[str, int, int], tuple[int, str]] = {}
    for record in records:
        for key, start, stop, numel, digest in record["rows"]:
            slot = (str(key), int(start), int(stop))
            seen = unique.get(slot)
            if seen is not None and seen[1] != digest:
                raise LightDigestError(f"{key}[{start}:{stop}] has two digests: replicas diverged")
            unique[slot] = (int(numel), str(digest))
    rows = sorted([key, start, stop, numel, digest] for (key, start, stop), (numel, digest) in unique.items())
    _check_tiling(rows)
    name = algorithms.pop()
    topology = json.loads(topologies.pop())
    h = hashlib.sha256()  # small input; always 64 hex, whatever hashes the blocks
    h.update(f"{LIGHT_DIGEST_VERSION}\0global\0{name}\0".encode())
    h.update(json.dumps(topology, sort_keys=True, separators=(",", ":")).encode())
    for row in rows:
        h.update(b"\n" + json.dumps(row, separators=(",", ":")).encode())
    digest = h.hexdigest()
    return {"digest": digest, "algorithm": name, "topology": topology, "rows": len(rows),
            "bytes": sum(int(r.get("bytes", 0)) for r in records),
            "seconds": max((float(r.get("seconds", 0.0)) for r in records), default=0.0)}


def _check_tiling(rows: Sequence[Sequence]) -> None:
    cursor: dict[str, int] = {}
    numels: dict[str, int] = {}
    for key, start, stop, numel, _digest in rows:  # sorted by key, then start
        if numels.setdefault(key, numel) != numel:
            raise LightDigestError(f"{key} has two sizes ({numels[key]}, {numel})")
        if start != cursor.get(key, 0) or stop < start:
            raise LightDigestError(f"{key}: range [{start}:{stop}] leaves a gap or overlaps "
                                   f"(expected start {cursor.get(key, 0)})")
        cursor[key] = stop
    for key, stop in cursor.items():
        if stop != numels[key]:
            raise LightDigestError(f"{key}: ranges cover [0:{stop}] of {numels[key]} element(s)")


# ----------------------------------------------------------------------------- worker side
def _topology(chunks: int) -> dict[str, int]:
    from megatron.core import parallel_state as mpu

    def size(name: str, **kwargs) -> int:
        fn = getattr(mpu, name, None)
        return int(fn(**kwargs)) if callable(fn) else 1

    return {"tp": size("get_tensor_model_parallel_world_size"),
            "pp": size("get_pipeline_model_parallel_world_size"),
            "ep": size("get_expert_model_parallel_world_size"),
            "cp": size("get_context_parallel_world_size"),
            "dp": size("get_data_parallel_world_size", with_context_parallel=True),
            "chunks": int(chunks)}


def local_shards(chunks: Sequence[Any], optimizer: Any, coords: Mapping[str, int]):
    """``(key, start, stop, numel, fp32 master shard)`` of every master shard this rank
    owns.  A DistributedOptimizer shard is owned by the rank that steps it; a complete
    master (no DP sharding) by data-parallel rank 0 (expert: expert-DP rank 0), as in
    :func:`.megatron_ports.owned_parameters`."""

    from yeto.rl import megatron_masters as mm

    from .megatron_ports import is_expert_param

    has_dist, owned = mm.distributed_ranges(optimizer)
    for index, chunk in enumerate(chunks):
        for name, param in chunk.named_parameters():
            expert = is_expert_param(name, param)
            where = f"tp{coords['tp']}.pp{coords['pp']}" + (f".ep{coords['ep']}" if expert else "")
            key = f"{index}:{name}@{where}"
            numel = int(param.numel())
            if id(param) in owned:
                leaf, _p, start, stop = owned[id(param)]
                yield key, int(start), int(stop), numel, mm._main_shard(leaf, param)
                continue
            if has_dist and not mm.has_complete_master(param):
                continue  # another DP rank owns every range of it
            if coords["edp" if expert else "dp"] != 0:
                continue
            yield key, 0, numel, numel, mm.master_of(param)


def remote_light_digest(worker_dict, threads: int = DEFAULT_THREADS, prefer: Sequence[str] | None = None) -> dict:
    """This rank's light-digest rows (no collective, no tensor leaves the process).

    Each FP32 master shard is copied to the host if it is not there yet (under the
    CPU-offloaded optimizer it is), checked finite and hashed in ``threads`` threads."""

    import torch

    from .megatron_full_parameter import _engine, _rank
    from .megatron_ports import _parallel_coordinates

    started = time.perf_counter()
    _bridge, chunks, optimizer = _engine(worker_dict)
    coords = _parallel_coordinates()
    name = algorithm(tuple(prefer) if prefer else ("blake3", "xxh3_128", "sha256"))
    meta, shards, digests, total, copied = [], [], [], 0, 0
    with torch.no_grad():
        for key, start, stop, numel, master in local_shards(chunks, optimizer, coords):
            host = master.detach().reshape(-1)
            if host.dtype != torch.float32 or int(host.numel()) != stop - start:
                raise LightDigestError(f"{key}: master shard is {host.dtype} x {host.numel()}, "
                                       f"range [{start}:{stop}]")
            if host.device.type != "cpu":
                copied += int(host.numel()) * 4
            host = host.cpu().contiguous()
            if not bool(torch.isfinite(host).all().item()):
                raise LightDigestError(f"{key}[{start}:{stop}] has NaN/Inf")
            raw = memoryview(host.view(torch.uint8).numpy())
            total += len(raw)
            meta.append((key, start, stop, numel))
            shards.append(("float32", stop - start, raw))
            if copied >= HOST_COPY_BUDGET:  # bound the host copies of card-resident shards
                digests += hash_shards(name, shards, threads=int(threads))
                shards, copied = [], 0
        digests += hash_shards(name, shards, threads=int(threads))
    rows = [[key, start, stop, numel, digest] for (key, start, stop, numel), digest in zip(meta, digests)]
    return {"rank": _rank(), "algorithm": name, "topology": _topology(len(chunks)), "rows": rows,
            "bytes": total, "seconds": round(time.perf_counter() - started, 3)}
