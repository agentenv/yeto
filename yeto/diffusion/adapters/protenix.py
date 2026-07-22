"""Yeto adapter boundary for Protenix structure diffusion.

Protenix is an AlphaFold3-style biomolecular structure predictor. Its training
contract is not a Diffusers image/video contract: Protenix owns feature
preprocessing, MSA/template handling, recycling, coordinate diffusion, auxiliary
heads, and the native loss. This adapter therefore exposes Protenix to Yeto as
a full-step external adapter and keeps orchestration/synchronization in Yeto.
"""

from __future__ import annotations

import importlib
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Mapping

from yeto.diffusion.adapters.base import DiffusionAdapter

if TYPE_CHECKING:
    import torch


def _torch():
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError("Protenix adapter requires torch to train or save parameters") from exc


def _deep_update(base: dict, update: Mapping) -> dict:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


class _ConfigNamespace(SimpleNamespace):
    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and hasattr(self, key)

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __iter__(self):
        return iter(vars(self))

    def __len__(self) -> int:
        return len(vars(self))

    def keys(self):
        return vars(self).keys()

    def items(self):
        return vars(self).items()

    def values(self):
        return vars(self).values()


def _namespace(value):
    if isinstance(value, Mapping):
        ns = _ConfigNamespace()
        for key, item in value.items():
            setattr(ns, key, _namespace(item))
        return ns
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _to_plain_config(value):
    if isinstance(value, Mapping):
        return {key: _to_plain_config(item) for key, item in value.items()}
    if isinstance(value, SimpleNamespace):
        return {
            key: _to_plain_config(item)
            for key, item in vars(value).items()
        }
    if isinstance(value, list):
        return [_to_plain_config(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_plain_config(item) for item in value)
    return value


def _config_get(configs, key: str, default=None):
    if isinstance(configs, Mapping):
        return configs.get(key, default)
    return getattr(configs, key, default)


def _import_attr(module_name: str, attr: str):
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _resolve_path(value: str | os.PathLike, base_dir: str | os.PathLike | None = None) -> Path:
    path = Path(os.path.expanduser(str(value)))
    if not path.is_absolute() and base_dir:
        path = Path(base_dir) / path
    return path


def _model_name_from_alias(model: str | None) -> str:
    if model == "protenix-v2":
        return "protenix-v2"
    return "protenix_base_default_v1.0.0"


def _dtype_from_args(args, device) -> str:
    dtype = getattr(args, "dtype", "bf16")
    if dtype and dtype != "auto":
        return dtype
    if getattr(device, "type", str(device)) == "cpu":
        return "f32"
    return "bf16"


def _tensorize(value, device):
    torch = _torch()
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {key: _tensorize(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tensorize(item, device) for item in value)
    if isinstance(value, list):
        if value and all(isinstance(item, (int, float, bool)) for item in value):
            return torch.tensor(value, device=device)
        return [_tensorize(item, device) for item in value]
    return value


class NativeProtenixPipeline:
    """Small wrapper around Protenix model/loss for prebuilt feature batches."""

    def __init__(
        self,
        *,
        configs,
        model,
        loss,
        symmetric_permutation,
        device,
    ) -> None:
        self.configs = configs
        self.model = model
        self.loss = loss
        self.symmetric_permutation = symmetric_permutation
        self.device = device
        self.step = 0

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    def build_batch(self, rows, args, device):
        if len(rows) != 1:
            raise RuntimeError(
                "Native Protenix batches must be pre-collated; use micro-batch 1 "
                "or store a complete Protenix batch per Yeto row."
            )
        row = rows[0]
        if "protenix_batch" in row:
            batch = row["protenix_batch"]
        elif {"input_feature_dict", "label_dict", "label_full_dict"} <= set(row):
            batch = {
                "input_feature_dict": row["input_feature_dict"],
                "label_dict": row["label_dict"],
                "label_full_dict": row["label_full_dict"],
            }
        else:
            batch_path = row.get("protenix_batch_path") or row.get("batch_path")
            if not batch_path:
                raise RuntimeError(
                    "Protenix rows need 'protenix_batch', Protenix batch keys, "
                    "or 'protenix_batch_path'/'batch_path'."
                )
            torch = _torch()
            path = _resolve_path(batch_path, row.get("__yeto_data_root__"))
            batch = torch.load(path, map_location="cpu", weights_only=False)
        del args
        return _tensorize(batch, device)

    def training_step(self, batch, global_step=0):
        self.step = global_step
        pred_dict, label_dict, log_dict = self.model(
            input_feature_dict=batch["input_feature_dict"],
            label_dict=batch["label_dict"],
            label_full_dict=batch["label_full_dict"],
            mode="train",
            current_step=global_step,
            symmetric_permutation=self.symmetric_permutation,
            mc_dropout_apply_rate=0,
        )
        del log_dict
        loss, _loss_dict = self.loss(
            feat_dict=batch["input_feature_dict"],
            pred_dict=pred_dict,
            label_dict=label_dict,
            mode="train",
        )
        return loss

    compute_loss = training_step


class ProtenixAdapter(DiffusionAdapter):
    """Full-step adapter for Protenix or a Protenix-compatible wrapper.

    The concrete Protenix Python APIs have changed across releases, so Yeto
    supports two integration modes:

    - install ``protenix`` and pass prebuilt Protenix batches in Yeto rows;
    - pass an object from tests or a local wrapper via ``backend``;
    - provide a wrapper module with ``load_pipeline`` for custom data/model APIs.

    The wrapper object should expose ``model`` or ``named_parameters()``, plus
    either ``training_step`` or ``compute_loss``.
    """

    def __init__(
        self,
        *,
        model_name: str | None = None,
        checkpoint_path: str | None = None,
        backend: Any | None = None,
        wrapper_module: str | None = None,
        config_args: str | None = None,
    ) -> None:
        self.model_name = model_name or os.environ.get("YETO_PROTENIX_MODEL_NAME")
        self.checkpoint_path = checkpoint_path or os.environ.get("YETO_PROTENIX_CHECKPOINT")
        self.backend = backend
        self.wrapper_module = wrapper_module or os.environ.get("YETO_PROTENIX_WRAPPER")
        self.config_args = config_args or os.environ.get("YETO_PROTENIX_CONFIG_ARGS", "")

    def load_pipeline(self, args, device):
        if self.backend is not None:
            return self.backend
        if not self.wrapper_module:
            return self._load_native_pipeline(args, device)
        module = importlib.import_module(self.wrapper_module)
        load = getattr(module, "load_pipeline", None)
        if load is None:
            raise RuntimeError(f"{self.wrapper_module} has no load_pipeline() function")
        return load(
            args,
            device,
            model_name=self.model_name or getattr(args, "model", None),
            checkpoint_path=self.checkpoint_path,
        )

    def _load_native_pipeline(self, args, device):
        torch = _torch()
        self._require_protenix()
        configs = self._build_native_configs(args, device)
        model_cls = _import_attr("protenix.model.protenix", "Protenix")
        loss_cls = _import_attr("protenix.model.loss", "ProtenixLoss")
        permutation_cls = _import_attr(
            "protenix.utils.permutation.permutation",
            "SymmetricPermutation",
        )
        model = model_cls(configs).to(device)
        loss = loss_cls(configs)
        error_dir = os.environ.get("YETO_PROTENIX_ERROR_DIR", ".")
        symmetric_permutation = permutation_cls(configs, error_dir=error_dir)
        pipe = NativeProtenixPipeline(
            configs=configs,
            model=model,
            loss=loss,
            symmetric_permutation=symmetric_permutation,
            device=device,
        )
        checkpoint_path = self.checkpoint_path
        if checkpoint_path:
            checkpoint = torch.load(
                Path(os.path.expanduser(checkpoint_path)),
                map_location=device,
                weights_only=False,
            )
            state = checkpoint.get("model", checkpoint)
            cleaned = {
                key.removeprefix("module."): value
                for key, value in state.items()
            }
            strict = bool(_config_get(configs, "load_strict", False))
            pipe.model.load_state_dict(cleaned, strict=strict)
        return pipe

    def _build_native_configs(self, args, device=None):
        configs_base = _import_attr("configs.configs_base", "configs")
        data_configs = _import_attr("configs.configs_data", "data_configs")
        model_configs = _import_attr("configs.configs_model_type", "model_configs")
        model_name = self.model_name or _model_name_from_alias(getattr(args, "model", None))
        if model_name not in model_configs:
            raise RuntimeError(
                f"unknown Protenix model_name {model_name!r}; "
                f"available models include {sorted(model_configs)[:8]}"
            )
        base = deepcopy(dict(configs_base))
        base["data"] = data_configs
        _deep_update(base, model_configs[model_name])
        base["model_name"] = model_name
        base["dtype"] = _dtype_from_args(args, device)
        if self.checkpoint_path:
            base["load_checkpoint_path"] = self.checkpoint_path
        base.setdefault("load_strict", False)
        parse_configs = _import_attr("protenix.config.config", "parse_configs")
        arg_str = f"--model_name {model_name} --dtype {base['dtype']}"
        if self.config_args:
            arg_str = f"{arg_str} {self.config_args}"
        return parse_configs(
            configs=base,
            arg_str=arg_str,
            fill_required_with_null=True,
        )

    def prepare_model(self, pipe, args, device):
        del args
        if hasattr(pipe, "to"):
            pipe.to(device)
        model = getattr(pipe, "model", pipe)
        if hasattr(model, "train"):
            model.train()
        return pipe

    def trainable_params(self, pipe) -> dict[str, torch.Tensor]:
        custom = getattr(pipe, "trainable_params", None)
        if custom is not None:
            return dict(custom())
        model = getattr(pipe, "model", pipe)
        if not hasattr(model, "named_parameters"):
            raise RuntimeError(
                "Protenix wrapper must expose model.named_parameters() or trainable_params()"
            )
        return {
            f"protenix.{name}": param
            for name, param in model.named_parameters()
            if getattr(param, "requires_grad", False)
        }

    def trainable_module_items(self, pipe):
        torch = _torch()
        model = getattr(pipe, "model", pipe)
        if isinstance(model, torch.nn.Module):
            return [("protenix", model)]
        return []

    def training_step(self, pipe, rows, args, device, global_step: int = 0):
        for name in ("training_step", "compute_loss"):
            fn = getattr(pipe, name, None)
            if fn is None:
                continue
            out = fn(self.build_batch(pipe, rows, args, device), global_step=global_step)
            return self._normalize_loss(out, device)
        raise RuntimeError(
            "Protenix wrapper must expose training_step(batch, global_step=...) or compute_loss(...)"
        )

    compute_loss = training_step

    def build_batch(self, pipe, rows, args, device):
        build = getattr(pipe, "build_batch", None)
        if build is not None:
            return build(rows, args=args, device=device)
        return {
            "rows": rows,
            "model_name": self.model_name or getattr(args, "model", None),
            "checkpoint_path": self.checkpoint_path,
        }

    def save_adapters(self, pipe, output_dir):
        torch = _torch()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        save = getattr(pipe, "save_adapters", None) or getattr(pipe, "save", None)
        if save is not None:
            save(out)
            return
        state = {name: p.detach().cpu() for name, p in self.trainable_params(pipe).items()}
        torch.save(state, out / "trainable_state.pt")

    save = save_adapters

    def load_adapters(self, pipe, adapter_dir, meta, args):
        torch = _torch()
        del meta, args
        load = getattr(pipe, "load_adapters", None) or getattr(pipe, "load", None)
        if load is not None:
            load(Path(adapter_dir))
            return pipe
        state_path = Path(adapter_dir) / "trainable_state.pt"
        state = torch.load(state_path, map_location="cpu")
        current = self.trainable_params(pipe)
        for name, tensor in state.items():
            if name in current:
                current[name].data.copy_(tensor.to(device=current[name].device, dtype=current[name].dtype))
        return pipe

    def sample(self, pipe, args, meta):
        del meta
        for name in ("sample", "predict", "infer"):
            fn = getattr(pipe, name, None)
            if fn is not None:
                return fn(args)
        raise RuntimeError("Protenix wrapper must expose sample(args), predict(args), or infer(args)")

    @staticmethod
    def _normalize_loss(out, device):
        torch = _torch()
        if isinstance(out, tuple) and len(out) == 2:
            return out
        if isinstance(out, dict):
            loss = out.get("loss")
            denom = out.get("denominator", out.get("denom"))
            if loss is None:
                raise RuntimeError("Protenix loss dict must contain a 'loss' key")
            if denom is None:
                denom = torch.ones((), device=device)
            return loss, denom
        return out, torch.ones((), device=device)

    @staticmethod
    def _require_protenix() -> None:
        try:
            importlib.import_module("protenix")
        except ImportError as exc:
            raise RuntimeError(
                "Protenix is not installed. Install Yeto with the diffusion-protenix "
                "extra or install protenix separately, then provide a wrapper via "
                "YETO_PROTENIX_WRAPPER."
            ) from exc


def make_adapter(**kwargs) -> ProtenixAdapter:
    return ProtenixAdapter(**kwargs)
