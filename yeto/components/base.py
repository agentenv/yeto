"""Component interfaces used by the generic diffusion backend."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    import torch


class DiffusionComponent:
    """Adapter for one diffusion-family training stack.

    The diffusion backend owns provisioning and the generic DiLoCo sync loop;
    a component supplies model construction, data iteration, and artifact
    export for a concrete framework such as NAVA.
    """

    name: str = ""
    default_config: str | None = None
    default_lora_targets: str = "all-linear"
    output_dir: str = "yeto-diffusion-output"

    def add_launch_cli_args(self, parser: argparse.ArgumentParser) -> None:
        """Optional component-specific launch flags."""

    def add_export_cli_args(self, parser: argparse.ArgumentParser) -> None:
        """Optional component-specific export flags."""

    def validate(self, args) -> list[str]:
        return []

    def warnings(self, args) -> list[str]:
        return []

    def head_file_mounts(self, args) -> dict[str, str]:
        return {}

    def rewrite_for_head(self, args) -> None:
        pass

    def build_learner_file_mounts(self, args) -> dict[str, str]:
        return {}

    def learner_env(self, args) -> dict[str, str]:
        return {}

    def setup_steps(self, args) -> list[str]:
        return []

    def warn_if_wont_fit(self, args, specs) -> None:
        pass

    def resolve_component_config(self, args) -> str:
        config = args.component_config or self.default_config
        if not config:
            raise ValueError(f"component {self.name!r} needs --component-config")
        return config

    # Learner/runtime hooks. Heavy component imports belong inside these.
    def resolve_paths(self, args) -> None:
        pass

    def load_config(self, args) -> dict:
        raise NotImplementedError

    def build_pipeline(self, args, cfg: dict, device: "torch.device"):
        raise NotImplementedError

    def configure_trainables(self, runtime, args):
        raise NotImplementedError

    def get_model(self, runtime):
        return runtime.model

    def set_model(self, runtime, model) -> None:
        runtime.model = model

    def trainable_params(self, runtime) -> Mapping[str, "torch.Tensor"]:
        raise NotImplementedError

    def build_dataloader(self, args, cfg: dict, runtime, rank: int, world: int):
        raise NotImplementedError

    def training_step(self, runtime, batch, global_step: int):
        raise NotImplementedError

    def build_scheduler(self, optimizer, cfg: dict):
        """Return the LR scheduler used by this component."""
        import torch

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)

    def batch_units(self, batch, world: int) -> int:
        return max(1, world)

    def save_artifact(self, runtime, args, output_dir: Path, params, adapter_config, metadata: dict) -> None:
        raise NotImplementedError

    def export(self, args, ckpt) -> None:
        raise NotImplementedError
