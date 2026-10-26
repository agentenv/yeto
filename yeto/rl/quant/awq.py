"""Fixed AWQ smoothing for re-quantization (option R2) and its no-AWQ fallback (R1).

AWQ (activation-aware weight quantization) moves a per-input-channel factor
``s`` out of a norm and into the linears that read the norm's output:

* the smooth layer (a norm weight, 1-D) is divided by ``s``;
* each balance layer (a linear weight ``[out, in]``) has column ``j``
  multiplied by ``s[j]``.

The product of the two is unchanged in exact arithmetic, but the balance
weights are better suited to int4. R2 keeps the factors of the original
artefact for every training round: the mappings come from its ``recipe.yaml``
(``RecordedAWQ.mappings``) and the factors from ``awq-scales/<smooth_layer>.pt``
(FP64, one value per hidden channel). The arithmetic is done in FP64 and
rounded once to the target dtype, which reproduces the original expert
tensors bit for bit on the sampled expert (see :mod:`.int4`).

R1 (:meth:`AwqSmoothing.none`) applies no factor. Its output is a consistent
model, but it differs from the original artefact also in the norms, routers
and shared experts; its quality loss is not measured.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml

WEIGHT_SUFFIX = ".weight"


class _RecipeLoader(yaml.SafeLoader):
    """SafeLoader that tolerates the python object tags llm-compressor writes (e.g. ``torch.device``)."""


_RecipeLoader.add_multi_constructor(
    "tag:yaml.org,2002:python/", lambda loader, suffix, node: None
)


@dataclass(frozen=True)
class AwqMapping:
    smooth_layer: str
    balance_exact: frozenset[str]
    balance_patterns: tuple[re.Pattern[str], ...]

    def is_balance(self, module: str) -> bool:
        return module in self.balance_exact or any(p.fullmatch(module) for p in self.balance_patterns)


@dataclass
class AwqSmoothing:
    """Map from module name to its smoothing role and factor. Empty = R1 (no smoothing)."""

    mappings: tuple[AwqMapping, ...] = ()
    factors: Mapping[str, torch.Tensor] = field(default_factory=dict)  # smooth_layer -> s (FP64)
    source: str = "none"

    @classmethod
    def none(cls) -> "AwqSmoothing":
        return cls()

    @classmethod
    def from_artefact(cls, recipe_path: Path, scales_dir: Path) -> "AwqSmoothing":
        """Read the mappings from ``recipe.yaml`` and the factors from ``scales_dir``; fail on any gap."""
        recipe = yaml.load(Path(recipe_path).read_text(encoding="utf-8"), Loader=_RecipeLoader)
        found = list(_find_key(recipe, "RecordedAWQ")) + list(_find_key(recipe, "AWQModifier"))
        if len(found) != 1:
            raise ValueError(f"{recipe_path}: expected exactly one AWQ modifier, found {len(found)}")
        mappings: list[AwqMapping] = []
        factors: dict[str, torch.Tensor] = {}
        for m in found[0].get("mappings") or []:
            smooth = m["smooth_layer"]
            exact = frozenset(b for b in m["balance_layers"] if not b.startswith("re:"))
            patterns = tuple(re.compile(b[3:]) for b in m["balance_layers"] if b.startswith("re:"))
            mappings.append(AwqMapping(smooth, exact, patterns))
            path = Path(scales_dir) / f"{smooth}.pt"
            if not path.is_file():
                raise FileNotFoundError(f"AWQ factor file missing for {smooth}: {path}")
            s = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(s, torch.Tensor) or s.dim() != 1 or not torch.isfinite(s).all() or (s <= 0).any():
                raise ValueError(f"{path}: AWQ factor must be a finite positive 1-D tensor")
            factors[smooth] = s.to(torch.float64)
        if not mappings:
            raise ValueError(f"{recipe_path}: AWQ modifier has no mappings")
        return cls(tuple(mappings), factors, source=f"{recipe_path} + {scales_dir}")

    @property
    def enabled(self) -> bool:
        return bool(self.mappings)

    def role(self, tensor_name: str) -> tuple[str, torch.Tensor] | None:
        """``("smooth" | "balance", s)`` for a ``*.weight`` tensor that AWQ changes, else ``None``."""
        if not tensor_name.endswith(WEIGHT_SUFFIX):
            return None
        module = tensor_name[: -len(WEIGHT_SUFFIX)]
        hits = [("smooth", m) for m in self.mappings if m.smooth_layer == module]
        hits += [("balance", m) for m in self.mappings if m.is_balance(module)]
        if len(hits) > 1:
            raise ValueError(f"{tensor_name}: matched by {len(hits)} AWQ mappings; recipe is ambiguous")
        if not hits:
            return None
        kind, m = hits[0]
        return kind, self.factors[m.smooth_layer]

    def apply(self, tensor_name: str, tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Return ``tensor`` smoothed (FP64 arithmetic, one rounding) and cast to ``dtype``.

        Works on any row slice of a balance weight, because the factor scales columns.
        """
        hit = self.role(tensor_name)
        if hit is None:
            return tensor.to(dtype)
        kind, s = hit
        if tensor.shape[-1] != s.numel():
            raise ValueError(f"{tensor_name}: last dim {tensor.shape[-1]} != AWQ factor size {s.numel()}")
        if kind == "smooth":
            if tensor.dim() != 1:
                raise ValueError(f"{tensor_name}: smooth layer weight must be 1-D, got {tuple(tensor.shape)}")
            return (tensor.double() / s).to(dtype)
        if tensor.dim() != 2:
            raise ValueError(f"{tensor_name}: balance layer weight must be 2-D, got {tuple(tensor.shape)}")
        return (tensor.double() * s.unsqueeze(0)).to(dtype)


def _find_key(node: object, key: str):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key:
                yield v
            else:
                yield from _find_key(v, key)
    elif isinstance(node, list):
        for v in node:
            yield from _find_key(v, key)
