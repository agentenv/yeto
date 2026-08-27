#!/usr/bin/env python3
"""Select DeepSeek-V4 routed experts from an audited always-solved corpus.

The profiler uses SGLang's native routed-expert capture for learned-router
layers.  DeepSeek-V4's first ``n_hash_layers`` deliberately bypass that capture,
so their exact routes are reconstructed from the checkpoint's immutable
``tid2eid`` tensors and the same input token IDs.

No text from the profiling corpus is written to the result.  The durable output
contains task IDs, content/token hashes, per-task counts, aggregate statistics,
and a deterministic per-layer hot-expert clone map.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import re
import struct
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from safetensors import safe_open


_TASK_ID = re.compile(r"CVE-\d{4}-\d{4,}")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--solve-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--miles-revision", required=True)
    parser.add_argument("--sglang-revision", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--top-n", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=32)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--mem-fraction-static", type=float, default=0.55)
    parser.add_argument("--chunked-prefill-size", type=int, default=4096)
    parser.add_argument("--page-size", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_npz(path: Path, **values: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npz")
    np.savez_compressed(temporary, **values)
    os.replace(temporary, path)


def _hash_token_ids(token_ids: Iterable[int]) -> str:
    digest = hashlib.sha256()
    digest.update(b"yeto-dsv4-token-ids-v1\0")
    for token_id in token_ids:
        digest.update(struct.pack("<I", int(token_id)))
    return digest.hexdigest()


def _longest_common_prefix(left: list[int], right: list[int]) -> int:
    index = 0
    while index < min(len(left), len(right)) and left[index] == right[index]:
        index += 1
    return index


def _eligible_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for row in rows:
        if row.get("tier") != "always":
            continue
        try:
            valid = (
                int(row["rounds_total"]) == 3
                and int(row["rounds_real"]) == 3
                and int(row["rounds_solved"]) == 3
                and int(row["env_failures"]) == 0
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid solve CSV always row") from exc
        if not valid:
            continue
        task_id = row.get("cve", "")
        if not _TASK_ID.fullmatch(task_id):
            raise ValueError(f"invalid always task ID {task_id!r}")
        selected.append(row)
    selected.sort(key=lambda row: row["cve"])
    if len(selected) != 654 or len({row["cve"] for row in selected}) != len(selected):
        raise ValueError(f"expected 654 unique fail-closed always tasks, got {len(selected)}")
    return selected


def _prepare_inputs(args, tokenizer, system_prompt: str, tools: list[dict]):
    from miles.utils.chat_template_utils import apply_chat_template

    rows = _eligible_rows(args.solve_csv)
    inputs: list[list[int]] = []
    task_starts: list[int] = []
    records = []
    corpus_digest = hashlib.sha256()
    corpus_digest.update(b"yeto-dsv4-always-routing-corpus-v1\0")
    missing_solutions = []

    for row in rows:
        task_id = row["cve"]
        task_root = args.corpus_root / "envs" / task_id
        prompt_path = task_root / "prompt_l2.md"
        solution_path = task_root / "solution.md"
        if not prompt_path.is_file():
            raise ValueError(f"always task {task_id} has no L2 prompt")
        prompt = prompt_path.read_text(encoding="utf-8")
        if not prompt.strip():
            raise ValueError(f"always task {task_id} has an empty L2 prompt")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        empty_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ""},
        ]
        files = [("prompt_l2.md", prompt_path)]
        add_generation_prompt = True
        if solution_path.is_file():
            solution = solution_path.read_text(encoding="utf-8")
            if not solution.strip():
                raise ValueError(f"always task {task_id} has an empty solution")
            # Canonical solutions stand in for successful reasoning trajectories;
            # they select a parameter subspace only and are never training rows.
            assistant = {
                "role": "assistant",
                "reasoning_content": solution,
                "content": "",
            }
            messages.append(assistant)
            empty_messages.append(dict(assistant))
            files.append(("solution.md", solution_path))
            add_generation_prompt = False
        else:
            missing_solutions.append(task_id)

        token_ids = apply_chat_template(
            messages,
            tokenizer=tokenizer,
            tools=tools,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
        empty_ids = apply_chat_template(
            empty_messages,
            tokenizer=tokenizer,
            tools=tools,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
        if not isinstance(token_ids, list) or not all(
            isinstance(token, int) for token in token_ids
        ):
            raise TypeError("DeepSeek V4 TITO renderer did not return integer IDs")
        task_start = _longest_common_prefix(token_ids, empty_ids)
        if task_start <= 0 or task_start >= len(token_ids):
            raise ValueError(f"cannot isolate task tokens for {task_id}")
        if len(token_ids) > 32767:
            raise ValueError(f"always routing input exceeds context limit for {task_id}")

        file_records = []
        for name, path in files:
            relative = f"envs/{task_id}/{name}"
            payload = path.read_bytes()
            encoded = relative.encode("utf-8")
            corpus_digest.update(struct.pack("<I", len(encoded)))
            corpus_digest.update(encoded)
            corpus_digest.update(struct.pack("<Q", len(payload)))
            corpus_digest.update(payload)
            file_records.append(
                {
                    "path": relative,
                    "bytes": len(payload),
                    "sha256": _sha256_bytes(payload),
                }
            )

        inputs.append(token_ids)
        task_starts.append(task_start)
        records.append(
            {
                "task_id": task_id,
                "files": file_records,
                "input_tokens": len(token_ids),
                "task_tokens": len(token_ids) - task_start,
                "task_start": task_start,
                "input_ids_sha256": _hash_token_ids(token_ids),
                "task_input_ids_sha256": _hash_token_ids(token_ids[task_start:]),
            }
        )

    if missing_solutions != ["CVE-2015-8103"]:
        raise ValueError(f"unexpected missing always solutions: {missing_solutions}")
    return rows, inputs, task_starts, records, corpus_digest.hexdigest(), missing_solutions


def _load_model_contract(model_path: Path):
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    contract = {
        "num_layers": int(config["num_hidden_layers"]),
        "num_experts": int(config["n_routed_experts"]),
        "topk": int(config["num_experts_per_tok"]),
        "num_hash_layers": int(config.get("n_hash_layers", 3)),
    }
    if contract != {
        "num_layers": 43,
        "num_experts": 256,
        "topk": 6,
        "num_hash_layers": 3,
    }:
        raise ValueError(f"unexpected DeepSeek V4 routing contract: {contract}")
    return config, contract


def _load_hash_routes(model_path: Path, num_hash_layers: int, topk: int):
    index = json.loads(
        (model_path / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    routes = []
    records = []
    for layer in range(num_hash_layers):
        candidates = (
            f"model.layers.{layer}.mlp.topk.tid2eid",
            f"layers.{layer}.ffn.gate.tid2eid",
        )
        found = [(name, index[name]) for name in candidates if name in index]
        if len(found) != 1 or not isinstance(found[0][1], str):
            raise ValueError(
                f"checkpoint must contain exactly one hash route key for layer "
                f"{layer}: candidates={candidates}, found={found}"
            )
        name, shard = found[0]
        with safe_open(model_path / shard, framework="pt", device="cpu") as handle:
            value = handle.get_tensor(name)
        if value.ndim != 2 or value.shape[1] != topk:
            raise ValueError(f"invalid {name} shape {tuple(value.shape)}")
        value = value.to(dtype=torch.int64).contiguous().numpy()
        if value.min() < 0 or value.max() >= 256:
            raise ValueError(f"invalid expert ID in {name}")
        routes.append(value)
        records.append(
            {
                "layer": layer,
                "checkpoint_key": name,
                "shard": shard,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "tensor_sha256": _sha256_bytes(value.tobytes(order="C")),
            }
        )
    return routes, records


def _decode_capture(value: Any, expected: int) -> np.ndarray:
    if not isinstance(value, str):
        raise TypeError(f"routed-expert capture is not base64 text: {type(value)!r}")
    try:
        payload = base64.b64decode(value.encode("ascii"), validate=True)
    except Exception as exc:
        raise ValueError("invalid routed-expert base64") from exc
    # ``np.frombuffer`` aliases the immutable ``bytes`` returned by b64decode,
    # so its result is read-only.  The profiler deliberately replaces
    # SGLang's uncaptured hash-router slots below; own the buffer before that
    # substitution.
    routed = np.frombuffer(payload, dtype=np.int32).copy()
    if routed.size != expected:
        raise ValueError(f"routed-expert capture has {routed.size} values, expected {expected}")
    return routed


def _contract(args, records, corpus_sha256, solve_sha256, system_prompt, tools):
    model_path = Path(args.model)
    tokenizer_path = Path(args.tokenizer)
    return {
        "schema": 1,
        "method": "task-normalized-mean-route-share",
        "bucket": "always",
        "eligibility": {
            "rounds_total": 3,
            "rounds_real": 3,
            "rounds_solved": 3,
            "env_failures": 0,
            "prompt_tier": "l2",
        },
        "task_count": len(records),
        "corpus_sha256": corpus_sha256,
        "solve_csv_sha256": solve_sha256,
        "source_revision": args.source_revision,
        "model_revision": args.model_revision,
        "model_manifest_sha256": args.model_manifest_sha256,
        "model_config_sha256": _sha256_file(model_path / "config.json"),
        "model_index_sha256": _sha256_file(
            model_path / "model.safetensors.index.json"
        ),
        "tokenizer_json_sha256": _sha256_file(tokenizer_path / "tokenizer.json"),
        "tokenizer_config_sha256": _sha256_file(
            tokenizer_path / "tokenizer_config.json"
        ),
        "miles_revision": args.miles_revision,
        "sglang_revision": args.sglang_revision,
        "image_digest": args.image_digest,
        "profiler_sha256": _sha256_file(Path(__file__)),
        "system_prompt_sha256": _sha256_bytes(system_prompt.encode("utf-8")),
        "tools_sha256": _sha256_bytes(_canonical_json(tools)),
        "top_n": args.top_n,
        "engine": {
            "tp_size": args.tp_size,
            "ep_size": args.ep_size,
            "mem_fraction_static": args.mem_fraction_static,
            "chunked_prefill_size": args.chunked_prefill_size,
            "page_size": args.page_size,
            "attention_backend": "dsv4",
            "temperature": 0.0,
            "max_new_tokens": 1,
        },
    }


def _selection(counts: np.ndarray, token_counts: np.ndarray, top_n: int, ep_size: int):
    if counts.ndim != 3 or counts.shape[1:] != (43, 256):
        raise ValueError(f"unexpected counts shape {counts.shape}")
    denominators = token_counts.astype(np.float64)[:, None, None] * 6.0
    shares = counts.astype(np.float64) / denominators
    mean = shares.mean(axis=0)
    median = np.median(shares, axis=0)
    p10 = np.quantile(shares, 0.10, axis=0)
    coverage = (counts > 0).mean(axis=0)

    layers = []
    selected_ids = np.empty((43, top_n), dtype=np.int32)
    experts_per_ep_rank = 256 // ep_size
    for layer in range(43):
        order = sorted(
            range(256),
            key=lambda expert: (
                -mean[layer, expert],
                -median[layer, expert],
                -p10[layer, expert],
                -coverage[layer, expert],
                expert,
            ),
        )
        chosen = order[:top_n]
        selected_ids[layer] = chosen
        shard_counts = [0] * ep_size
        entries = []
        for clone_offset, expert in enumerate(chosen):
            shard = expert // experts_per_ep_rank
            shard_counts[shard] += 1
            entries.append(
                {
                    "clone_expert_id": 256 + clone_offset,
                    "source_expert_id": expert,
                    "ep_rank": shard,
                    "mean_route_share": float(mean[layer, expert]),
                    "median_route_share": float(median[layer, expert]),
                    "p10_route_share": float(p10[layer, expert]),
                    "task_coverage": float(coverage[layer, expert]),
                }
            )
        layers.append(
            {
                "layer": layer,
                "selected": entries,
                "source_ep_rank_histogram": shard_counts,
            }
        )

    global_mean = mean.mean(axis=0)
    global_median = median.mean(axis=0)
    global_order = sorted(
        range(256),
        key=lambda expert: (-global_mean[expert], -global_median[expert], expert),
    )
    global_top = [
        {
            "source_expert_id": expert,
            "mean_route_share_across_layers": float(global_mean[expert]),
            "mean_median_route_share_across_layers": float(global_median[expert]),
        }
        for expert in global_order[:top_n]
    ]
    return layers, global_top, selected_ids, mean, median, p10, coverage


def main() -> None:
    args = _args()
    if args.top_n != 32:
        raise ValueError("this recipe requires exactly 32 cloned experts")
    if args.tp_size != 8 or args.ep_size != 8:
        raise ValueError("this recipe requires TP8/EP8 profiling")
    if args.batch_size <= 0 or args.checkpoint_every <= 0:
        raise ValueError("batch/checkpoint sizes must be positive")
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_revision):
        raise ValueError("source revision must be an immutable Git commit")
    for label in ("model_revision", "miles_revision", "sglang_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", getattr(args, label)):
            raise ValueError(f"{label} must be an immutable Git commit")
    if not re.fullmatch(r"[0-9a-f]{64}", args.model_manifest_sha256):
        raise ValueError("model manifest hash must be SHA256")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image_digest):
        raise ValueError("image digest must be an immutable SHA256 digest")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    complete_path = args.output_dir / "complete.json"
    if complete_path.exists():
        raise FileExistsError(f"profile is already complete: {complete_path}")

    from transformers import AutoTokenizer
    from yeto_miles_secrlenv.agent import SYSTEM_PROMPT, TOOLS

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rows, input_ids, task_starts, records, corpus_sha256, missing_solutions = _prepare_inputs(
        args, tokenizer, SYSTEM_PROMPT, TOOLS
    )
    solve_sha256 = _sha256_file(args.solve_csv)
    profile_contract = _contract(
        args,
        records,
        corpus_sha256,
        solve_sha256,
        SYSTEM_PROMPT,
        TOOLS,
    )
    contract_sha256 = _sha256_bytes(_canonical_json(profile_contract))
    _atomic_bytes(
        args.output_dir / "corpus.json",
        _canonical_json(
            {
                **profile_contract,
                "contract_sha256": contract_sha256,
                "missing_solutions": missing_solutions,
                "tasks": records,
            }
        ),
    )

    model_path = Path(args.model)
    _config, model_contract = _load_model_contract(model_path)
    hash_routes, hash_route_records = _load_hash_routes(
        model_path,
        model_contract["num_hash_layers"],
        model_contract["topk"],
    )
    routing_contract_path = args.output_dir / "routing_contract.json"
    _atomic_bytes(
        routing_contract_path,
        _canonical_json(
            {
                "model": model_contract,
                "hash_routes": hash_route_records,
            }
        ),
    )
    task_ids = np.asarray([row["cve"] for row in rows])
    token_counts = np.asarray(
        [len(tokens) - start for tokens, start in zip(input_ids, task_starts, strict=True)],
        dtype=np.int32,
    )
    counts = np.zeros((len(rows), 43, 256), dtype=np.uint32)
    processed = 0
    partial_path = args.output_dir / "partial.npz"
    partial_meta_path = args.output_dir / "partial.json"
    if args.resume:
        if not partial_path.is_file() or not partial_meta_path.is_file():
            raise FileNotFoundError("--resume requires partial.npz and partial.json")
        meta = json.loads(partial_meta_path.read_text(encoding="utf-8"))
        if meta.get("contract_sha256") != contract_sha256:
            raise ValueError("partial profile contract mismatch")
        with np.load(partial_path, allow_pickle=False) as saved:
            counts = saved["counts"]
            processed = int(saved["processed"])
            if not np.array_equal(saved["task_ids"], task_ids):
                raise ValueError("partial task order mismatch")
        if counts.shape != (len(rows), 43, 256) or not 0 <= processed < len(rows):
            raise ValueError("invalid partial profile state")
    elif partial_path.exists() or partial_meta_path.exists():
        raise FileExistsError("partial profile exists; pass --resume or use a new output")

    from sglang import Engine

    started = time.monotonic()
    engine = Engine(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        trust_remote_code=True,
        model_impl="sglang",
        tp_size=args.tp_size,
        dp_size=1,
        ep_size=args.ep_size,
        attention_backend="dsv4",
        mem_fraction_static=args.mem_fraction_static,
        max_running_requests=args.batch_size,
        chunked_prefill_size=args.chunked_prefill_size,
        page_size=args.page_size,
        enable_return_routed_experts=True,
        skip_server_warmup=True,
        random_seed=0,
        log_level="warning",
    )
    print(
        json.dumps(
            {
                "event": "engine_ready",
                "seconds": time.monotonic() - started,
                "remaining_tasks": len(rows) - processed,
                "remaining_tokens": int(sum(len(ids) for ids in input_ids[processed:])),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        while processed < len(rows):
            end = min(processed + args.batch_size, len(rows))
            batch = input_ids[processed:end]
            result = engine.generate(
                input_ids=batch,
                sampling_params={
                    "temperature": 0.0,
                    "max_new_tokens": 1,
                    "ignore_eos": True,
                },
                return_routed_experts=True,
                routed_experts_start_len=0,
            )
            results = result if isinstance(result, list) else [result]
            if len(results) != len(batch):
                raise ValueError(f"engine returned {len(results)} results for {len(batch)} inputs")

            for local_index, (tokens, response) in enumerate(
                zip(batch, results, strict=True)
            ):
                task_index = processed + local_index
                meta = response.get("meta_info") if isinstance(response, dict) else None
                if not isinstance(meta, dict):
                    raise TypeError("SGLang response has no meta_info")
                if int(meta.get("prompt_tokens", -1)) != len(tokens):
                    raise ValueError("SGLang prompt token count mismatch")
                routed = _decode_capture(
                    meta.get("routed_experts"),
                    len(tokens) * 43 * 6,
                ).reshape(len(tokens), 43, 6)
                learned = routed[:, model_contract["num_hash_layers"] :, :]
                if learned.size and (learned.min() < 0 or learned.max() >= 256):
                    raise ValueError("learned router returned an invalid logical expert ID")
                token_array = np.asarray(tokens, dtype=np.int64)
                for layer, table in enumerate(hash_routes):
                    if token_array.max(initial=0) >= table.shape[0]:
                        raise ValueError("token ID exceeds hash router table")
                    routed[:, layer, :] = table[token_array]

                task_routes = routed[task_starts[task_index] :]
                if task_routes.shape[0] != token_counts[task_index]:
                    raise ValueError("task routing row count mismatch")
                for layer in range(43):
                    layer_counts = np.bincount(
                        task_routes[:, layer, :].reshape(-1),
                        minlength=256,
                    )
                    if layer_counts.shape != (256,) or int(layer_counts.sum()) != int(
                        token_counts[task_index]
                    ) * 6:
                        raise ValueError("invalid per-layer routed-expert count")
                    counts[task_index, layer] = layer_counts.astype(np.uint32)

            processed = end
            if processed % args.checkpoint_every < args.batch_size or processed == len(rows):
                _atomic_npz(
                    partial_path,
                    contract_sha256=np.asarray(contract_sha256),
                    processed=np.asarray(processed, dtype=np.int32),
                    task_ids=task_ids,
                    counts=counts,
                )
                _atomic_bytes(
                    partial_meta_path,
                    _canonical_json(
                        {
                            "contract_sha256": contract_sha256,
                            "processed": processed,
                            "total": len(rows),
                        }
                    ),
                )
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "processed": processed,
                        "total": len(rows),
                        "elapsed_seconds": time.monotonic() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        engine.shutdown()

    layers, global_top, selected_ids, mean, median, p10, coverage = _selection(
        counts, token_counts, args.top_n, args.ep_size
    )
    counts_path = args.output_dir / "counts.npz"
    scores_path = args.output_dir / "scores.npz"
    selection_path = args.output_dir / "selection.json"
    _atomic_npz(
        counts_path,
        contract_sha256=np.asarray(contract_sha256),
        task_ids=task_ids,
        token_counts=token_counts,
        counts=counts,
    )
    _atomic_npz(
        scores_path,
        contract_sha256=np.asarray(contract_sha256),
        selected_source_expert_ids=selected_ids,
        mean_route_share=mean,
        median_route_share=median,
        p10_route_share=p10,
        task_coverage=coverage,
    )
    selection = {
        **profile_contract,
        "contract_sha256": contract_sha256,
        "selection_scope": "32 independent source experts per decoder layer",
        "clone_id_contract": "source rank 0..31 maps to logical clone IDs 256..287",
        "ranking_tiebreak": [
            "mean_route_share_desc",
            "median_route_share_desc",
            "p10_route_share_desc",
            "task_coverage_desc",
            "source_expert_id_asc",
        ],
        "layers": layers,
        "global_top32_diagnostic_only": global_top,
        "counts_sha256": _sha256_file(counts_path),
        "scores_sha256": _sha256_file(scores_path),
        "routing_contract_sha256": _sha256_file(routing_contract_path),
    }
    _atomic_bytes(selection_path, _canonical_json(selection))
    complete = {
        "status": "complete",
        "contract_sha256": contract_sha256,
        "corpus_sha256": corpus_sha256,
        "selection_sha256": _sha256_file(selection_path),
        "counts_sha256": _sha256_file(counts_path),
        "scores_sha256": _sha256_file(scores_path),
        "tasks": len(rows),
        "task_tokens": int(token_counts.sum()),
        "total_seconds": time.monotonic() - started,
    }
    _atomic_bytes(complete_path, _canonical_json(complete))
    print(json.dumps(complete, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
