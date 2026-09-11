"""Full-run entry point for user-authorized generated-CoT masked Qwen SFT.

This delegates the qualified NeMo lifecycle without changing any baseline file.
Only this Python process temporarily binds the new dataset validators/identity.
Run with torchrun in the pinned image after separately reserving an idle host.
Explicit same-contract checkpoint resume is supported; no service management is performed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import threading

from training.qwen38_no_cot import nemo_train as baseline
from . import masked_data as data, masked_recipe as recipe

VERSION = "qwen38-native-gap-masked-runner/v4"
BASE_MODEL_DIR = Path("/data/sft_baseline_20260908/models/Qwen3.8-27B")
MODEL_CONFIG_SHA256 = "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
MODEL_INDEX_SHA256 = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
RUNTIME_IMAGE = "sha256:b0bd0e50b29cb0a3ec69c3cbbb49652d3c1aabedbfe22ee0fa6d053b6bb33aee"
BASELINE_RUNTIME_SHA256 = {
    "nemo_train.py": "2be1d6d89e0922be5520faa82bcad14a1cb5ad2f277e100d0095b41068a2e5df",
    "nemo_data.py": "f1687c09078503f20614b93a0d161187156c3c6714458c4c3b417db35c1d976a",
    "nemo_recipe.py": "fddafa0bcf46370dfcca2c9dda6adc1f3f779deefc670a338175de8ab1dda783",
    "nemo_checkpoint_probe.py": "8baa2d93a7d3575a0510f5251a5f36d097736abcf1b99f72334bdfa8537c5a7d",
    "nemo_model_probe.py": "8a55a8b219aaedd31fff1e1f02020ad096b90163bc31fb773ea6604f80ae468c",
    "nemo_cp_memory.py": "c27fb867ac19b5766d455de0fc07ed50a244b7ff9f6464b3499ed8e944e97295",
    "nemo_block_checkpoint.py": "9e796ca0e45dee313abf8a216be6a95f86ad09266b7f20169d7b2b2efd971cfa",
    "nemo_checkpoint.py": "69374ead7037d987d4389824064b44a3dc083da68b4d1ae24623de8209c8aad5",
    "render.py": "3e5ef75c849acb4158f6c873e333c16af3c6fe1ea452215990be16ab91ace18a",
    "upstream_manager/render.py": "e2eacc680cbea4343458ad76d57ad1d36e84b2a324a1b8cc9d90692c5ddab1e8",
}
PHASES = {"train"}
_ORIGINAL_REQUIRE = baseline.require_recipe
_ORIGINAL_IDENTITY = baseline.contract_identity
_DELEGATION_LOCK = threading.Lock()
_BASE_VERIFICATION_CACHE = {}


def _tensor_sha256(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode())
    digest.update(str(value.dtype).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def live_cp_data_semantics_guard(config, mode):
    """Fail before update 1 unless the actual CP8 batch/mask is exact.

    The loader batch must be identical on all eight CP ranks.  The resolved CP
    sharder must reproduce and gather every token exactly once, and the fused
    loss must receive the once-shifted global label count rather than an 8x
    count.  A rank-0 receipt is published only after the first update returns.
    """
    if mode != "train":
        yield
        return
    import torch
    import torch.distributed as dist
    from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
    from nemo_automodel.recipes.vlm import finetune

    original_step = finetune.FinetuneRecipeForVLM._run_train_optim_step
    original_shard = finetune.ContextParallelSharder.shard
    original_loss = FusedLinearCrossEntropy.forward
    state = {"active": False, "complete": False, "expected_labels": None,
             "microbatches": [], "rank_batch_hashes": [], "loss_calls": 0}

    def gather_digest(value, device):
        raw = bytes.fromhex(value)
        local = torch.tensor(list(raw), dtype=torch.uint8, device=device)
        gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, local)
        return [bytes(item.cpu().tolist()).hex() for item in gathered]

    def guarded_shard(sharder, batch):
        if not state["active"] or state["complete"] or "labels" not in batch:
            return original_shard(sharder, batch)
        full_labels = batch["labels"].detach().clone()
        full_inputs = batch["input_ids"].detach().clone()
        context, local = original_shard(sharder, batch)
        if "labels" not in local or "input_ids" not in local:
            raise RuntimeError("CP sharder removed a required token-aligned tensor")

        # torch.distributed.tensor.experimental.context_parallel is lazy: its
        # buffers are not resized to this rank's shard until train_ctx() is
        # entered.  Keep references here (the recipe pops ``labels`` from the
        # mapping before entering the context) and validate them only inside the
        # entered context.
        local_labels = local["labels"]
        local_inputs = local["input_ids"]
        expected_labels = sharder.shard_token_tensor(
            full_labels.to(local_labels.device), seq_dim=1, fill=-100)
        expected_inputs = sharder.shard_token_tensor(
            full_inputs.to(local_inputs.device), seq_dim=1, fill=248044)
        shard_batch = sharder.shard_batch
        backend = (getattr(shard_batch, "__module__", "") + "."
                   + getattr(shard_batch, "__qualname__", getattr(shard_batch, "__name__", "")))
        aux_only = backend == (
            "nemo_automodel.components.distributed.context_parallel.sharder."
            "shard_batch_aux_only")
        if aux_only and not torch.equal(local_inputs, full_inputs.to(local_inputs.device)):
            raise RuntimeError("Aux-only CP sharder changed the full primary stream before model forward")
        entered = False

        @contextmanager
        def audited_context():
            nonlocal entered
            if entered:
                raise RuntimeError("CP train context was entered more than once for one microbatch")
            entered = True
            with context():
                # Auxiliary-only Qwen CP intentionally keeps input_ids full so
                # the model can embed/splice them and shard embeddings inside
                # forward.  Other CP backends shard input_ids lazily together
                # with labels when this context is entered.
                if not torch.equal(local_labels, expected_labels):
                    raise RuntimeError("Actual deferred CP label shard differs from the token-index contract")
                if aux_only:
                    if not torch.equal(local_inputs, full_inputs.to(local_inputs.device)):
                        raise RuntimeError("Aux-only CP primary stream changed before in-model sharding")
                    audited_local_inputs = expected_inputs
                    primary_mode = "model_owned_aux_only_full_then_in_forward_shard"
                else:
                    if not torch.equal(local_inputs, expected_inputs):
                        raise RuntimeError("Actual deferred CP input shard differs from the token-index contract")
                    audited_local_inputs = local_inputs
                    primary_mode = "outer_context_sharded"
                gathered_labels = sharder.gather_token_tensor(
                    local_labels, seq_dim=1, trim=True, fill=-100)
                gathered_inputs = sharder.gather_token_tensor(
                    audited_local_inputs, seq_dim=1, trim=True, fill=248044)
                if (not torch.equal(gathered_labels.cpu(), full_labels.cpu())
                        or not torch.equal(gathered_inputs.cpu(), full_inputs.cpu())):
                    raise RuntimeError(
                        "CP8 local partitions do not collectively reconstruct one full sample")
                local_count = int((local_labels != -100).sum().item())
                total_count = torch.tensor(
                    local_count, dtype=torch.long, device=local_labels.device)
                dist.all_reduce(total_count)
                full_count = int((full_labels != -100).sum().item())
                if int(total_count.item()) != full_count:
                    raise RuntimeError("CP8 partitions duplicate or omit supervised targets")
                state["microbatches"].append({
                    "full_input_sha256": _tensor_sha256(full_inputs),
                    "full_labels_sha256": _tensor_sha256(full_labels),
                    "cp_primary_stream_mode": primary_mode,
                    "outer_input_ids_full_length": bool(aux_only),
                    "outer_input_tokens_per_rank": int(local_inputs.numel()),
                    "expected_model_local_input_tokens_per_rank": int(
                        audited_local_inputs.numel()),
                    "expected_model_local_input_ids_sha256_by_rank": gather_digest(
                        _tensor_sha256(audited_local_inputs), audited_local_inputs.device),
                    "local_labels_sha256_by_rank": gather_digest(
                        _tensor_sha256(local_labels), local_labels.device),
                    "full_label_tokens": full_count,
                    "cp_partition_label_tokens": int(total_count.item()),
                    "full_tokens": int(full_inputs.numel()),
                    "local_tokens_per_rank": int(audited_local_inputs.numel()),
                    "in_forward_primary_shard": (
                        "qualified_by_exact_pinned_qwen_aux_only_prelaunch_probe"
                        if aux_only else "not_applicable"),
                })
                yield

        return audited_context, local

    def guarded_loss(loss, hidden_states, labels, lm_weight, num_label_tokens=None,
                     grad_reduce_group=None):
        if state["active"] and not state["complete"]:
            local = torch.tensor(int((labels != -100).sum().item()), dtype=torch.long,
                                 device=labels.device)
            dist.all_reduce(local)
            index = state["loss_calls"]
            if index >= len(state["microbatches"]):
                raise RuntimeError("Fused CE ran without a matching audited CP microbatch")
            if (num_label_tokens != state["expected_labels"]
                    or int(local.item()) != state["microbatches"][index]["full_label_tokens"]):
                raise RuntimeError("Fused CE label count is shifted, duplicated or CP8-miscounted")
            state["loss_calls"] += 1
        return original_loss(loss, hidden_states, labels, lm_weight,
                             num_label_tokens=num_label_tokens,
                             grad_reduce_group=grad_reduce_group)

    def guarded_step(trainer, batches, max_grad_norm=None):
        if state["complete"]:
            return original_step(trainer, batches, max_grad_norm)
        if (dist.get_world_size() != 8 or trainer._get_cp_group_size() != 8
                or trainer._get_dp_group_size() != 1):
            raise RuntimeError("Live data semantics require exact one-host CP8")
        signatures = []
        expected = 0
        for batch in batches:
            signatures.append(hashlib.sha256((
                _tensor_sha256(batch["input_ids"]) + _tensor_sha256(batch["labels"])
            ).encode()).hexdigest())
            expected += int((batch["labels"] != -100).sum().item())
        combined = hashlib.sha256(json.dumps(signatures, separators=(",", ":")).encode()).hexdigest()
        rank_hashes = gather_digest(combined, trainer.dist_env.device)
        if len(set(rank_hashes)) != 1:
            raise RuntimeError("StatefulDataLoader selected different CP samples or order across ranks")
        state.update(active=True, expected_labels=expected, rank_batch_hashes=rank_hashes,
                     microbatches=[], loss_calls=0)
        result = original_step(trainer, batches, max_grad_norm)
        if (len(state["microbatches"]) != len(batches)
                or state["loss_calls"] != len(batches)
                or int(result.metrics["num_label_tokens"]) != expected):
            raise RuntimeError("Optimizer metric does not count the exact once-shifted targets")
        state.update(active=False, complete=True)
        if trainer.dist_env.is_main:
            destination = Path(config["checkpoint"]["checkpoint_dir"]).parent / "first-update-data-semantics.json"
            payload = {
                "schema": "qwen38-native-gap-v4-first-update-data-semantics/v1",
                "status": "passed", "world_size": 8, "cp_size": 8, "dp_size": 1,
                "pre_cp_batch_sha256_by_rank": rank_hashes,
                "pre_cp_batches_identical": True,
                "microbatches": state["microbatches"],
                "expected_once_shifted_label_tokens": expected,
                "reported_num_label_tokens": int(result.metrics["num_label_tokens"]),
                "training_step": int(trainer.step_scheduler.step),
                "loss_internal_shift": False,
                "manifest_sha256": data.digest_file(config["dataset"]["path_or_dataset"]),
                "config_canonical_sha256": hashlib.sha256(json.dumps(
                    config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
            }
            pending = destination.with_suffix(".tmp")
            with pending.open("x") as stream:
                stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
                stream.flush(); os.fsync(stream.fileno())
            pending.replace(destination)
        return result

    finetune.FinetuneRecipeForVLM._run_train_optim_step = guarded_step
    finetune.ContextParallelSharder.shard = guarded_shard
    FusedLinearCrossEntropy.forward = guarded_loss
    try:
        yield
    finally:
        finetune.FinetuneRecipeForVLM._run_train_optim_step = original_step
        finetune.ContextParallelSharder.shard = original_shard
        FusedLinearCrossEntropy.forward = original_loss


def _stable_signature(path):
    stat = Path(path).stat()
    return tuple(getattr(stat, key) for key in
                 ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))


def _path_on_read_only_mount(path):
    """Require the model's most-specific Linux mount to be read-only."""
    mountinfo = Path("/proc/self/mountinfo")
    if not mountinfo.is_file():
        return False
    target = str(Path(path).resolve(strict=True))
    matches = []
    for line in mountinfo.read_text().splitlines():
        before, separator, after = line.partition(" - ")
        if not separator:
            continue
        fields = before.split()
        if len(fields) < 6:
            continue
        mount = fields[4].replace("\\040", " ").replace("\\134", "\\")
        if target == mount or target.startswith(mount.rstrip("/") + "/"):
            matches.append((len(mount), set(fields[5].split(","))))
    return bool(matches) and "ro" in max(matches, key=lambda item: item[0])[1]


def verify_base_assets(config):
    """Rehash the exact full-byte base receipt before any model construction."""
    model = config["model"]
    path = Path(model["pretrained_model_name_or_path"])
    if (not path.is_absolute() or path != BASE_MODEL_DIR or path.resolve(strict=True) != BASE_MODEL_DIR
            or model.get("_target_") != "nemo_automodel.NeMoAutoModelForImageTextToText.from_pretrained"
            or model.get("local_files_only") is not True or model.get("trust_remote_code") is not False
            or any(key in model for key in ("state_dict", "from_tf", "from_flax", "adapter_name_or_path"))):
        raise ValueError("This fresh-base runner only loads the pinned original Qwen model directory")
    if not _path_on_read_only_mount(path):
        raise ValueError("The pinned base-model directory must be a dedicated read-only launch mount")
    binding = config.get("dataset_contract", {}).get("base_asset_receipt")
    if binding != data.BASE_ASSET_RECEIPT:
        raise ValueError("Training config lacks the exact durable base-model receipt identity")
    receipt_path = Path(binding["path"])
    if (not receipt_path.is_absolute() or receipt_path.is_symlink()
            or receipt_path.resolve(strict=True) != receipt_path
            or data.digest_file(receipt_path) != binding["sha256"]):
        raise ValueError("Durable full-byte base-model receipt changed")
    if not _path_on_read_only_mount(receipt_path):
        raise ValueError("The base-model verification receipt must be a read-only launch mount")
    receipt = data._strict_json(receipt_path.read_bytes())
    expected_receipt = {
        "schema": binding["schema"], "status": "passed", "root": str(BASE_MODEL_DIR),
        "repository": data.NATIVE_ASSET_IDENTITY["repository"],
        "revision": data.NATIVE_ASSET_IDENTITY["revision"],
        "source_manifest_sha256": binding["source_manifest_sha256"],
        "file_count": binding["file_count"], "weight_shards": binding["weight_shards"],
        "total_bytes": binding["total_bytes"], "exact_file_membership": True,
        "full_weight_bytes_rehashed": True, "config_sha256": MODEL_CONFIG_SHA256,
        "weight_index_sha256": MODEL_INDEX_SHA256,
        "tokenizer_sha256": data.NATIVE_ASSET_IDENTITY["tokenizer_sha256"],
        "tokenizer_config_sha256": data.NATIVE_ASSET_IDENTITY["tokenizer_config_sha256"],
        "chat_template_sha256": data.NATIVE_ASSET_IDENTITY["official_template_sha256"],
    }
    if any(receipt.get(key) != value for key, value in expected_receipt.items()):
        raise ValueError("Full-byte base-model receipt has the wrong source identity")
    records = receipt.get("files")
    if not isinstance(records, list) or len(records) != binding["file_count"]:
        raise ValueError("Full-byte receipt file membership is incomplete")
    files, folded = {}, set()
    for record in records:
        if (not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}
                or not isinstance(record["path"], str) or Path(record["path"]).name != record["path"]
                or record["path"].casefold() in folded or type(record["bytes"]) is not int
                or record["bytes"] < 1 or not data._hash(record["sha256"])):
            raise ValueError("Full-byte receipt contains an unsafe or colliding path")
        files[record["path"]] = record
        folded.add(record["path"].casefold())
    children = list(path.iterdir())
    if any(child.is_symlink() or not child.is_file() for child in children):
        raise ValueError("Every base-model directory member must be a regular top-level file")
    actual_names = {child.name for child in children}
    if actual_names != set(files) or len({name.casefold() for name in actual_names}) != len(actual_names):
        raise ValueError("Base-model files differ from the exact full-byte receipt membership")
    cache_key = (str(path), binding["sha256"], _stable_signature(receipt_path),
                 tuple((name, _stable_signature(path / name)) for name in sorted(files)))
    if cache_key not in _BASE_VERIFICATION_CACHE:
        total = 0
        for name in sorted(files):
            file_path, record = path / name, files[name]
            before = _stable_signature(file_path)
            actual = data.digest_file(file_path)
            after = _stable_signature(file_path)
            if before != after or after[2] != record["bytes"] or actual != record["sha256"]:
                raise ValueError("A base-model file changed or differs from its full-byte receipt")
            total += after[2]
        source_manifest = "".join(
            record["sha256"] + "  ./" + name + "\n" for name, record in sorted(files.items()))
        if (total != binding["total_bytes"]
                or hashlib.sha256(source_manifest.encode()).hexdigest() != binding["source_manifest_sha256"]
                or data.digest_file(receipt_path) != binding["sha256"]):
            raise ValueError("Base-model full-byte aggregate or receipt changed during verification")
        _BASE_VERIFICATION_CACHE.clear()
        _BASE_VERIFICATION_CACHE[cache_key] = True
    if (data.digest_file(path / "config.json") != MODEL_CONFIG_SHA256
            or data.digest_file(path / "model.safetensors.index.json") != MODEL_INDEX_SHA256):
        raise ValueError("Pinned base model config or weight-index identity changed")
    index = data._strict_json((path / "model.safetensors.index.json").read_bytes())
    mapping = index.get("weight_map")
    if not isinstance(mapping, dict) or not mapping or any(not isinstance(v, str) for v in mapping.values()):
        raise ValueError("Pinned base model index has no valid weight map")
    shards = sorted(set(mapping.values()))
    actual_shards = sorted(name for name in files if name.endswith(".safetensors"))
    if shards != actual_shards or len(shards) != binding["weight_shards"]:
        raise ValueError("Weight-index membership differs from the full-byte receipt")
    for name in shards:
        shard = path / name
        if (Path(name).name != name or not name.endswith(".safetensors")
                or not shard.is_file() or shard.resolve(strict=True) != shard
                or shard.stat().st_size < 1):
            raise ValueError("Every original base weight shard must exist inside the pinned model directory")
    tokenizer = data._strict_json((path / "tokenizer.json").read_bytes())
    vocab = tokenizer.get("model", {}).get("vocab", {})
    added = tokenizer.get("added_tokens", [])
    token_ids = list(vocab.values()) + [item.get("id") for item in added]
    controls = {item.get("content"): item.get("id") for item in added}
    if (any(type(token) is not int for token in token_ids)
            or len(token_ids) != data.TOKENIZER_ID_COUNT
            or len(set(token_ids)) != len(token_ids)
            or len(controls) != len(added)
            or set(token_ids) != set(range(data.TOKENIZER_ID_COUNT))
            or max(token_ids) != data.MAX_TOKEN_ID
            or any(controls.get(token) != token_id
                   for token, token_id in data.PINNED_CONTROL_TOKEN_IDS.items())):
        raise ValueError("Pinned tokenizer ID range or native control-token mapping changed")
    if os.environ.get("YETA_TRAINING_IMAGE") != RUNTIME_IMAGE:
        raise ValueError("The separate runner requires the exact qualified NeMo image identity")
    baseline_root = Path(baseline.__file__).parent
    actual_runtime = {name: data.digest_file(baseline_root / name)
                      for name in BASELINE_RUNTIME_SHA256}
    if actual_runtime != BASELINE_RUNTIME_SHA256:
        raise ValueError("The complete qualified baseline runtime changed; requalify it first")
    return {"model_config_sha256": MODEL_CONFIG_SHA256,
            "model_weight_index_sha256": MODEL_INDEX_SHA256,
            "runtime_image": RUNTIME_IMAGE, "base_weight_shards": len(shards),
            "base_asset_receipt_sha256": binding["sha256"],
            "base_source_manifest_sha256": binding["source_manifest_sha256"],
            "weight_validation": "exact-membership-stable-full-byte-rehash/v2",
            "qualified_baseline_runtime_sha256": actual_runtime,
            "full_weight_bytes_rehashed": True}


def verify_read_only_runtime_inputs(config):
    """Close post-admission TOCTOU for code and the exact portable export."""
    project_code = Path(__file__).resolve().parents[2]
    required = {"project code": project_code}
    for dataset_name in ("dataset", "validation_dataset"):
        item = config.get(dataset_name)
        if item is None:
            continue
        manifest_supplied = Path(item["path_or_dataset"])
        manifest = data.read_manifest(manifest_supplied)
        if manifest_supplied.is_symlink():
            raise ValueError("Dataset manifest cannot be a symlink")
        export_dir = manifest_supplied.resolve(strict=True).parent
        required[dataset_name + " export directory"] = export_dir
        for entry in manifest["splits"][item["split"]]:
            required[dataset_name + " shard " + entry["path"]] = export_dir / entry["path"]
        for key, label in (("path_or_dataset", "manifest"), ("index_path", "index"),
                           ("complete_path", "completion marker")):
            supplied = Path(item[key])
            if supplied.is_symlink():
                raise ValueError(dataset_name + " " + label + " cannot be a symlink")
            path = supplied.resolve(strict=True)
            required[dataset_name + " " + label] = path
    failures = [name for name, path in required.items()
                if (path.is_symlink()
                    or (not path.is_dir() if name == "project code" or name.endswith("export directory")
                        else not path.is_file())
                    or not _path_on_read_only_mount(path))]
    if failures:
        raise ValueError("Training code and complete dataset tree require read-only launch mounts: "
                         + ", ".join(sorted(failures)))
    return {"verified": True, "schema": "qwen38-read-only-runtime-inputs/v1",
            "paths": {name: str(path) for name, path in sorted(required.items())}}


def assert_output(config, config_path, resume=None):
    checkpoint = Path(config["checkpoint"]["checkpoint_dir"])
    if not checkpoint.is_absolute() or checkpoint.name != "checkpoints":
        raise ValueError("Use an absolute new run/checkpoints directory")
    root = checkpoint.parent
    if root == BASE_MODEL_DIR or root in BASE_MODEL_DIR.parents or BASE_MODEL_DIR in root.parents:
        raise ValueError("Training output must be separate from the original model")
    if root.exists() and (not root.is_dir() or root.resolve() != root):
        raise ValueError("Training output cannot alias an existing location")
    if Path(config["wandb"]["dir"]) != root:
        raise ValueError("Metrics and checkpoints must belong to the same run")
    if resume is not None:
        if checkpoint.is_symlink() or not checkpoint.is_dir() or checkpoint.resolve() != checkpoint:
            raise ValueError("Explicit resume requires this run's canonical checkpoint directory")
        return  # Exact dataset/model/runtime identity and cursor are checked before GPU work.
    if checkpoint.is_symlink() or (checkpoint.exists() and (not checkpoint.is_dir() or any(checkpoint.iterdir()))):
        raise ValueError("Fresh-base training refuses an occupied checkpoint directory")
    if Path(config["wandb"]["dir"]) != root:
        raise ValueError("Metrics and checkpoints must belong to the same fresh run")
    # A launcher may already have copied this config, its launch receipt and a
    # compilation cache. Existing training artifacts do not qualify as fresh.
    allowed = {"checkpoints", "launch.json", "cpu-preflight.json", "triton-cache"}
    config_path = Path(config_path).resolve()
    if config_path.parent == root:
        allowed.add(config_path.name)
    if root.exists() and any(path.name not in allowed or path.is_symlink() for path in root.iterdir()):
        raise ValueError("Existing run artifacts require a different fresh output directory")


def _check_phase(config, mode):
    if mode not in {"preflight", "train"} or config.get("experiment_phase") != "train":
        raise ValueError("This experiment uses one full shuffled training pass")
    return mode


def prepare_run(config, *, mode, config_path, resume=None):
    recipe.require_recipe(config)
    forwarded_mode = _check_phase(config, mode)
    assert_output(config, config_path, resume)
    read_only_inputs = verify_read_only_runtime_inputs(config)
    assets = verify_base_assets(config)
    bound = recipe.contract_identity(config)
    counts = {}
    runtime_counts = {}
    for name in ("dataset", "validation_dataset"):
        if name not in config:
            continue
        item = config[name]
        # Check complete supplied manifest membership, without smoke filters.
        checked = data.GeneratedMaskedTokenDataset(item["path_or_dataset"], index_path=item["index_path"],
            index_sha256=item["index_sha256"], complete_path=item["complete_path"],
            complete_sha256=item["complete_sha256"], split=item["split"], seq_len=item["seq_len"],
            require_provenance=item["require_provenance"])
        counts[item["split"]] = len(checked)
        # Reuse the validated index lengths instead of rehashing every shard
        # a second time merely to report the validation subset size.
        retained = sum(1 for ref in checked.refs if item.get("max_input_tokens") is None
                       or ref[4] <= item["max_input_tokens"])
        runtime_counts[item["split"]] = min(retained, item.get("max_samples") or retained)
    batch = config["step_scheduler"]["global_batch_size"]
    rows = counts.get("train", 0)
    if rows < 1:
        raise ValueError("The full run needs a nonempty training split")
    resume_receipt = None
    if resume is not None:
        from training.qwen38_no_cot.nemo_checkpoint import resume_identity, resolve_explicit_resume
        resume_receipt = resolve_explicit_resume(config["checkpoint"]["checkpoint_dir"], resume,
                                                resume_identity(config, contract_identity(config)))
    return {"schema": VERSION, "mode": mode, "forwarded_mode": forwarded_mode,
            "dataset_rows": counts,
            "runtime_dataset_rows": runtime_counts,
            "expected_optimizer_updates": (rows + batch - 1) // batch,
            "final_partial_window_microbatches": rows % batch,
            "manifest_sha256": bound["manifest_sha256"], "base_assets": assets,
            "read_only_runtime_inputs": read_only_inputs,
            "training_contract": data.TRAINING_CONTRACT,
            "semantic_quality_qualified": False, "unreviewed_cot_included": True,
            "resume_requested": resume is not None, "resume_receipt": resume_receipt,
            "gpu_runtime_qualified": False, "training_started": False}


def contract_identity(config):
    """Keep the qualified lifecycle identity and replace all no-CoT semantics."""
    bound = recipe.contract_identity(config)
    assets = verify_base_assets(config)
    result = _ORIGINAL_IDENTITY(config)
    project = Path(__file__).resolve().parents[2]
    files = ["training/qwen38_native_gap_v3/" + name for name in (
        "masked_train.py", "masked_recipe.py", "masked_data.py", "masked.py", "normalize.py",
        "source_adapters.py", "prepare_masked_full.py", "masked_replay.py")]
    files += ["training/qwen38_cot_experimental/" + name for name in (
        "train.py", "recipe.py", "data.py", "render.py", "prepare_full.py", "generation_contract.py", "replay.py")]
    files += ["cot_filler/" + name for name in (
        "core.py", "corpus_worker.py", "regeneration.py", "grounding_review.py",
        "grounding_review_ids.py", "grounding_review_v3.py", "grounding_review_v4.py", "grounding_review_fast.py")]
    result.update(training_contract=data.TRAINING_CONTRACT, runner_version=VERSION,
                  mask_policy=data.MASK_POLICY, sequence_policy=data.SEQUENCE_POLICY,
                  thinking_mode="xhigh", reasoning_loss=0, initialization="pinned_base_or_explicit_same_contract_resume",
                  filled_cot_contract=bound, base_assets=assets,
                  read_only_runtime_inputs=verify_read_only_runtime_inputs(config),
                  filled_cot_code_sha256={name: data.digest_file(project / name) for name in files})
    return result


@contextmanager
def bound_baseline(config, config_path, mode, resume=None):
    """Temporarily bind only this process; reject nested/concurrent delegation."""
    if not _DELEGATION_LOCK.acquire(blocking=False):
        raise RuntimeError("A baseline delegation is already active in this process")
    old_require, old_identity = baseline.require_recipe, baseline.contract_identity
    expected = None

    def require_bound(actual):
        if json.dumps(actual, sort_keys=True, separators=(",", ":"), allow_nan=False) != expected:
            raise ValueError("Training configuration changed after wrapper validation")
        recipe.require_recipe(actual)
        _check_phase(actual, mode)
        assert_output(actual, config_path, resume)

    try:
        expected = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if (old_require is not _ORIGINAL_REQUIRE or old_identity is not _ORIGINAL_IDENTITY
                or recipe.require_baseline_runtime is not _ORIGINAL_REQUIRE):
            raise RuntimeError("Delegation requires the original captured baseline guards")
        baseline.require_recipe = require_bound
        baseline.contract_identity = contract_identity
        yield
    finally:
        baseline.require_recipe, baseline.contract_identity = old_require, old_identity
        _DELEGATION_LOCK.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("preflight", "train"), default="preflight")
    parser.add_argument("--short-receipt")
    parser.add_argument("--long-receipt")
    parser.add_argument("--resident-long-receipt")
    parser.add_argument("--validate-memory-repair-in-production", action="store_true")
    parser.add_argument("--resume", help="Explicit LATEST or same-run checkpoint; exact identity and cursor must match")
    args = parser.parse_args(argv)
    config_path = Path(args.config).resolve(strict=True)
    config = json.loads(config_path.read_bytes())
    prepared = prepare_run(config, mode=args.mode, config_path=config_path,
                           resume=args.resume)
    forwarded_mode = prepared["forwarded_mode"]
    if forwarded_mode == "train" and os.environ.get("PYTORCH_ALLOC_CONF") != "expandable_segments:True":
        parser.error("Full training requires the qualified expandable-segments allocator setting")
    if forwarded_mode == "train" and not (args.resident_long_receipt or args.validate_memory_repair_in_production):
        parser.error("Train requires its matching resident receipt or explicit production memory validation")
    forwarded = [str(Path(__file__).resolve()), "--config", str(config_path), "--mode", forwarded_mode]
    for name in ("short_receipt", "long_receipt", "resident_long_receipt"):
        value = getattr(args, name)
        if value:
            forwarded.extend(("--" + name.replace("_", "-"), value))
    if args.validate_memory_repair_in_production:
        forwarded.append("--validate-memory-repair-in-production")
    if args.resume is not None:
        forwarded.extend(("--resume", args.resume))
    print(json.dumps(prepared, sort_keys=True), flush=True)
    old_argv = sys.argv
    try:
        with bound_baseline(config, config_path, args.mode, args.resume), \
                live_cp_data_semantics_guard(config, args.mode):
            sys.argv = forwarded
            return baseline.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
