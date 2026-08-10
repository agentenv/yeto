#!/usr/bin/env python3
"""Load and execute the expanded FP8 checkpoint through pinned SGLang."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--adapter-positive")
    parser.add_argument("--adapter-negative")
    parser.add_argument(
        "--adapter-transport",
        choices=("path", "tensors"),
        default="path",
        help="exercise disk-path loading or Engine/TokenizerManager tensor loading",
    )
    return parser.parse_args()


def _response_signature(response: dict) -> str:
    meta = response.get("meta_info")
    if not isinstance(meta, dict):
        raise ValueError("SGLang response has no metadata")
    payload = {
        "text": response.get("text"),
        "output_token_logprobs": meta.get("output_token_logprobs"),
        "output_token_ids": meta.get("output_token_ids"),
        "completion_tokens": meta.get("completion_tokens"),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _generate(engine, input_ids: list[int], lora_path: str | None) -> dict:
    response = engine.generate(
        input_ids=input_ids,
        sampling_params={
            "temperature": 0.0,
            "max_new_tokens": 2,
            "ignore_eos": True,
        },
        lora_path=lora_path,
        return_logprob=True,
        logprob_start_len=len(input_ids),
        return_routed_experts=True,
        routed_experts_start_len=0,
    )
    if not isinstance(response, dict):
        raise TypeError(f"unexpected SGLang response type {type(response)!r}")
    return response


def _load_adapter(
    engine,
    path: str,
    *,
    transport: str,
    config: dict,
    name: str,
) -> str:
    if transport == "path":
        lora_name = path
        result = engine.load_lora_adapter(
            lora_name=lora_name,
            lora_path=path,
            pinned=False,
        )
    elif transport == "tensors":
        from safetensors.torch import load_file
        import torch.multiprocessing as torch_mp

        from sglang.srt.utils import MultiprocessingSerializer
        from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

        lora_name = name
        tensors = load_file(
            str(Path(path) / "adapter_model.safetensors"),
            device="cpu",
        )
        # Miles' colocated policy push sends one flattened bucket per TP rank,
        # not tens of thousands of independent tensor storages.  Exercise that
        # same SGLang load_format here.  The fixture is CPU-backed, so use the
        # filename sharing strategy: the scheduler broadcasts one serialized
        # payload to all TP ranks and every rank must be able to attach to the
        # same storage.  The default file-descriptor reducer is one-shot and
        # fails after the first consumer with resource_sharer EOF/KeyError.
        torch_mp.set_sharing_strategy("file_system")
        bucket = FlattenedTensorBucket(named_tensors=list(tensors.items()))
        flattened = bucket.get_flattened_tensor()
        serialized = MultiprocessingSerializer.serialize(
            {
                "flattened_tensor": flattened,
                "metadata": bucket.get_metadata(),
            },
            output_str=True,
        )
        try:
            # This reaches TokenizerManager.load_lora_adapter_from_tensors and
            # TP-worker FlattenedTensorBucket reconstruction, the same backend
            # and wire format used by Miles' colocated adapter update.
            result = engine.load_lora_adapter_from_tensors(
                lora_name=lora_name,
                tensors=serialized,
                config_dict=config,
                load_format="flattened_bucket",
            )
        finally:
            del serialized
            del flattened
            del bucket
            del tensors
    else:  # argparse constrains this, but keep the runtime fail closed.
        raise ValueError(f"unsupported adapter transport {transport!r}")
    success = (
        result.get("success")
        if isinstance(result, dict)
        else getattr(result, "success", None)
    )
    if success is not True:
        error = (
            result.get("error_message")
            if isinstance(result, dict)
            else getattr(result, "error_message", repr(result))
        )
        raise RuntimeError(f"failed to load LoRA fixture {path!r}: {error}")
    return lora_name


def main() -> None:
    args = _args()
    if (args.adapter_positive is None) != (args.adapter_negative is None):
        raise ValueError("both positive and negative adapter fixtures are required")
    from transformers import AutoConfig

    from yeto.rl.deepseek_v4_expert_clone import contract_from_config
    from yeto.rl.sglang_deepseek_v4_clone import install

    contract = contract_from_config(
        AutoConfig.from_pretrained(
            args.model,
            trust_remote_code=True,
            local_files_only=True,
        )
    )
    if contract is None:
        raise ValueError("SGLang clone validation requires an expanded checkpoint")
    install()

    adapter_targets = None
    if args.adapter_positive is not None:
        configs = []
        manifests = []
        for path in (args.adapter_positive, args.adapter_negative):
            config_path = Path(path) / "adapter_config.json"
            manifest_path = Path(path) / "yeto_fixture_manifest.json"
            if not config_path.is_file() or not manifest_path.is_file():
                raise FileNotFoundError(f"incomplete adapter fixture {path!r}")
            config = json.loads(config_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            attention_tensors = int(manifest.get("attention_tensors", -1))
            original_expert_tensors = int(
                manifest.get("original_expert_tensors", -1)
            )
            clone_expert_tensors = int(manifest.get("clone_expert_tensors", -1))
            counts = (
                attention_tensors,
                original_expert_tensors,
                clone_expert_tensors,
            )
            if (
                config.get("r") != 8
                or config.get("lora_alpha") != 8
                or counts
                not in {
                    (214, 66_048, 8_256),
                    (0, 66_048, 8_256),
                    (214, 0, 0),
                }
                or manifest.get("tensor_count") != sum(counts)
            ):
                raise ValueError(f"invalid E288 adapter fixture contract at {path!r}")
            configs.append(config)
            manifests.append(manifest)
        count_fields = (
            "attention_tensors",
            "original_expert_tensors",
            "clone_expert_tensors",
        )
        if any(manifests[0].get(key) != manifests[1].get(key) for key in count_fields):
            raise ValueError("adapter fixtures disagree on physical tensor scope")
        positive_targets = set(configs[0].get("target_modules") or ())
        negative_targets = set(configs[1].get("target_modules") or ())
        if not positive_targets or positive_targets != negative_targets:
            raise ValueError("adapter fixtures disagree on target modules")
        expected_targets = set()
        if manifests[0]["clone_expert_tensors"]:
            expected_targets.update({"gate_proj", "up_proj", "down_proj"})
        if manifests[0]["attention_tensors"]:
            expected_targets.update({"q_a_proj", "q_b_proj"})
        if positive_targets != expected_targets:
            raise ValueError(
                "adapter fixture targets do not match its physical tensor scope"
            )
        adapter_targets = sorted(positive_targets)
        if "q_a_proj" in positive_targets:
            fuse = os.environ.get("SGLANG_OPT_FUSE_WQA_WKV")
            if fuse not in (None, "0"):
                raise ValueError(
                    "attention LoRA requires SGLANG_OPT_FUSE_WQA_WKV=0"
                )
            os.environ["SGLANG_OPT_FUSE_WQA_WKV"] = "0"

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
        moe_runner_backend="triton",
        disable_shared_experts_fusion=True,
        enable_eplb=False,
        mem_fraction_static=0.55,
        max_running_requests=1,
        chunked_prefill_size=4096,
        page_size=256,
        disable_cuda_graph=True,
        enable_return_routed_experts=True,
        skip_server_warmup=True,
        random_seed=0,
        log_level="warning",
        enable_lora=adapter_targets is not None,
        max_lora_rank=8 if adapter_targets is not None else None,
        max_loras_per_batch=2,
        max_loaded_loras=2 if adapter_targets is not None else None,
        lora_target_modules=adapter_targets,
        lora_use_virtual_experts=False,
        lora_strict_loading=True,
        experts_shared_outer_loras=False,
    )
    ready_seconds = time.monotonic() - started
    try:
        # Low ordinary IDs keep the smoke independent of special-token
        # aliases while exercising prefill and decode through every layer.
        input_ids = list(range(1, 65))
        response = _generate(engine, input_ids, None)
        meta = response.get("meta_info")
        if not isinstance(meta, dict) or int(meta.get("prompt_tokens", -1)) != len(
            input_ids
        ):
            raise ValueError("SGLang clone response has invalid prompt metadata")
        text = response.get("text")
        capture = meta.get("routed_experts")
        if not isinstance(capture, str) or not capture:
            raise ValueError("SGLang clone response has no routed-expert capture")
        base_signature = _response_signature(response)

        adapter_results = None
        if args.adapter_positive is not None:
            load_started = time.monotonic()
            positive_name = _load_adapter(
                engine,
                args.adapter_positive,
                transport=args.adapter_transport,
                config=configs[0],
                name="yeto-tensor-positive",
            )
            positive_load_seconds = time.monotonic() - load_started
            load_started = time.monotonic()
            negative_name = _load_adapter(
                engine,
                args.adapter_negative,
                transport=args.adapter_transport,
                config=configs[1],
                name="yeto-tensor-negative",
            )
            negative_load_seconds = time.monotonic() - load_started
            positive_started = time.monotonic()
            positive = _generate(engine, input_ids, positive_name)
            positive_seconds = time.monotonic() - positive_started
            negative_started = time.monotonic()
            negative = _generate(engine, input_ids, negative_name)
            negative_seconds = time.monotonic() - negative_started
            positive_signature = _response_signature(positive)
            negative_signature = _response_signature(negative)
            if base_signature == positive_signature:
                raise RuntimeError("full expert LoRA fixture had no rollout effect")
            if positive_signature == negative_signature:
                raise RuntimeError("dynamic expert LoRA replacement had no rollout effect")
            adapter_results = {
                "transport": args.adapter_transport,
                "target_modules": adapter_targets,
                "base_signature": base_signature,
                "positive_signature": positive_signature,
                "negative_signature": negative_signature,
                "positive_request_seconds": positive_seconds,
                "negative_request_seconds": negative_seconds,
                "positive_load_seconds": positive_load_seconds,
                "negative_load_seconds": negative_load_seconds,
                "positive_text_bytes": len(
                    (positive.get("text") or "").encode("utf-8")
                ),
                "negative_text_bytes": len(
                    (negative.get("text") or "").encode("utf-8")
                ),
            }
        print(
            json.dumps(
                {
                    "status": "ok",
                    "ready_seconds": ready_seconds,
                    "prompt_tokens": meta["prompt_tokens"],
                    "completion_tokens": meta.get("completion_tokens"),
                    "finish_reason": meta.get("finish_reason"),
                    "text_bytes": len((text or "").encode("utf-8")),
                    "capture_base64_bytes": len(capture),
                    "selection_sha256": contract.selection_sha256,
                    "tp_size": args.tp_size,
                    "ep_size": args.ep_size,
                    "adapter_results": adapter_results,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        active_exception = sys.exc_info()[0] is not None
        try:
            engine.shutdown()
        except Exception:
            # Do not let a secondary shutdown failure hide the actual adapter
            # or scheduler error that made the validation leave its body.
            if not active_exception:
                raise


if __name__ == "__main__":
    main()
