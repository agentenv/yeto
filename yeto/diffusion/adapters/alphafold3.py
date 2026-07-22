"""Guarded AlphaFold3 inference adapter for Yeto.

Official AlphaFold3 is license-gated and non-commercial. This adapter never
downloads or bundles model parameters. It only invokes a local official
AlphaFold3 checkout with a user-provided model-parameters directory.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from yeto.diffusion.adapters.base import DiffusionAdapter

ALPHAFOLD3_MODEL_ID = "alphafold3"
ALPHAFOLD3_LICENSE_MESSAGE = (
    "Official AlphaFold3 is licensed for non-commercial use and its model "
    "parameters must be received directly from Google. Set YETO_ALPHAFOLD3_ROOT "
    "to an official checkout and YETO_ALPHAFOLD3_MODEL_PARAMETERS_DIR to your "
    "authorized local parameters directory."
)


def _expand(value: str | os.PathLike | None) -> Path | None:
    if not value:
        return None
    return Path(os.path.expanduser(str(value))).resolve()


def _require_dir(path: Path | None, label: str) -> Path:
    if path is None or not path.exists() or not path.is_dir():
        raise RuntimeError(f"{label} is required. {ALPHAFOLD3_LICENSE_MESSAGE}")
    return path


class AlphaFold3Runtime:
    """Thin subprocess wrapper around official ``run_alphafold.py``."""

    def __init__(
        self,
        *,
        root: Path,
        model_parameters_dir: Path,
        databases_dir: Path | None = None,
        output_root: Path | None = None,
        python_executable: str | None = None,
        extra_args: str | None = None,
    ) -> None:
        self.root = root
        self.model_parameters_dir = model_parameters_dir
        self.databases_dir = databases_dir
        self.output_root = output_root or Path.cwd() / "alphafold3-yeto-output"
        self.python_executable = python_executable or sys.executable
        self.extra_args = extra_args or ""

    def sample(self, args) -> dict:
        json_path = self._input_json(args)
        run_dir = self.output_root / json_path.stem
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.python_executable,
            str(self.root / "run_alphafold.py"),
            "--json_path",
            str(json_path),
            "--model_dir",
            str(self.model_parameters_dir),
            "--output_dir",
            str(run_dir),
        ]
        if self.databases_dir is not None:
            cmd += ["--db_dir", str(self.databases_dir)]
        if self.extra_args:
            import shlex

            cmd += shlex.split(self.extra_args)
        subprocess.run(cmd, cwd=self.root, check=True)
        return {"paths": [run_dir]}

    @staticmethod
    def _input_json(args) -> Path:
        value = getattr(args, "input_json", None) or getattr(args, "prompt", None)
        if value is None:
            raise RuntimeError("AlphaFold3 sampling expects --prompt to be an input JSON path")
        path = _expand(value)
        if path is None or not path.exists():
            raise FileNotFoundError(f"AlphaFold3 input JSON not found: {value}")
        return path


class AlphaFold3Adapter(DiffusionAdapter):
    """License-gated adapter for official AlphaFold3 inference."""

    def __init__(
        self,
        *,
        root: str | None = None,
        model_parameters_dir: str | None = None,
        databases_dir: str | None = None,
        output_root: str | None = None,
        python_executable: str | None = None,
        extra_args: str | None = None,
        backend: Any | None = None,
    ) -> None:
        self.root = _expand(root or os.environ.get("YETO_ALPHAFOLD3_ROOT"))
        self.model_parameters_dir = _expand(
            model_parameters_dir
            or os.environ.get("YETO_ALPHAFOLD3_MODEL_PARAMETERS_DIR")
            or os.environ.get("YETO_ALPHAFOLD3_MODEL_DIR")
        )
        self.databases_dir = _expand(databases_dir or os.environ.get("YETO_ALPHAFOLD3_DATABASES_DIR"))
        self.output_root = _expand(output_root or os.environ.get("YETO_ALPHAFOLD3_OUTPUT_DIR"))
        self.python_executable = python_executable or os.environ.get("YETO_ALPHAFOLD3_PYTHON")
        self.extra_args = extra_args or os.environ.get("YETO_ALPHAFOLD3_ARGS", "")
        self.backend = backend

    def load_pipeline(self, args, device):
        del args, device
        raise RuntimeError(
            "Official AlphaFold3 support is inference-only in Yeto. "
            "Use the adapter for sampling/prediction, not DiLoCo training. "
            + ALPHAFOLD3_LICENSE_MESSAGE
        )

    def load_sample_pipeline(self, adapter_dir, meta, args, device):
        del adapter_dir, meta, args, device
        if self.backend is not None:
            return self.backend
        root = _require_dir(self.root, "YETO_ALPHAFOLD3_ROOT")
        model_parameters_dir = _require_dir(
            self.model_parameters_dir,
            "YETO_ALPHAFOLD3_MODEL_PARAMETERS_DIR",
        )
        run_script = root / "run_alphafold.py"
        if not run_script.exists():
            raise RuntimeError(f"{root} does not look like an AlphaFold3 checkout: missing run_alphafold.py")
        if self.databases_dir is not None:
            _require_dir(self.databases_dir, "YETO_ALPHAFOLD3_DATABASES_DIR")
        return AlphaFold3Runtime(
            root=root,
            model_parameters_dir=model_parameters_dir,
            databases_dir=self.databases_dir,
            output_root=self.output_root,
            python_executable=self.python_executable,
            extra_args=self.extra_args,
        )

    load_pipeline_for_sampling = load_sample_pipeline

    def sample(self, pipe, args, meta):
        del meta
        if hasattr(pipe, "sample"):
            return pipe.sample(args)
        raise RuntimeError("AlphaFold3 runtime object must expose sample(args)")

    def training_step(self, pipe, rows, args, device, global_step: int = 0):
        del pipe, rows, args, device, global_step
        raise RuntimeError("Official AlphaFold3 training is not supported by this guarded adapter")

    compute_loss = training_step

    def trainable_params(self, pipe) -> dict:
        del pipe
        return {}

    def save_adapters(self, pipe, output_dir):
        del pipe
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "ALPHAFOLD3_LICENSE_NOTICE.txt").write_text(ALPHAFOLD3_LICENSE_MESSAGE + "\n", encoding="utf-8")


def make_adapter(**kwargs) -> AlphaFold3Adapter:
    return AlphaFold3Adapter(**kwargs)
