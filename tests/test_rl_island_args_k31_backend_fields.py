"""The k31 pin keeps some native fields in two places (the trainer trait and the
Megatron argparse namespace); an override must land in both or Miles' custom
function view refuses the config (GPU-A run13/run15)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from yeto.rl.adapters.miles.island_args import _mirror_megatron_base_args, _split_backend_updates


class _RawMegatron(BaseModel):
    model_config = ConfigDict(frozen=True)
    base_args: dict = {}


class _Backend(BaseModel):
    wandb_project: str | None = None


class _Top(BaseModel):
    model_config = ConfigDict(extra="allow")
    raw_megatron: _RawMegatron = _RawMegatron(base_args={"wandb_project": None, "lr": 1e-6})
    backend: _Backend = _Backend()
    wandb_project: str | None = None
    hf_checkpoint: str | None = None


def test_megatron_base_args_mirror_the_shared_fields_only():
    top = _Top()
    out = _mirror_megatron_base_args(top, {"wandb_project": "yeto", "wandb_mode": "online", "hf_checkpoint": "x"})
    assert set(out) == {"raw_megatron"}
    assert out["raw_megatron"].base_args == {"wandb_project": "yeto", "lr": 1e-6}
    assert _mirror_megatron_base_args(top, {"hf_checkpoint": "x"}) == {}
    assert _mirror_megatron_base_args(object(), {"wandb_project": "yeto"}) == {}


def test_backend_only_fields_route_to_the_backend():
    class _TopNoField(BaseModel):
        model_config = ConfigDict(extra="allow")
        backend: _Backend = _Backend()

    top, nested = _split_backend_updates(_TopNoField(), {"wandb_project": "yeto", "other": 1})
    assert nested == {"wandb_project": "yeto"} and top == {"other": 1}
    # a field the top config declares itself stays at the top
    top, nested = _split_backend_updates(_Top(), {"wandb_project": "yeto"})
    assert top == {"wandb_project": "yeto"} and nested == {}
