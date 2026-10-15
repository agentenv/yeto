"""refresh_actor_backup: refresh an existing pinned "actor" copy, never create one.

S24 T-27B: the cut restore called ``weights_backuper.backup("actor")`` on a Miles
backuper that had skipped the actor backup (colocated LoRA / fixed partition), which
allocated a pinned host copy of the whole frozen base per rank (+133 GB on 27B).
"""
from types import SimpleNamespace

import torch

from yeto.rl.adapters.miles.state_plugin import refresh_actor_backup


class _NormalBackuper:
    """Same allocation rule as Miles' _TensorBackuperNormal.backup (CPU only, no pinning)."""

    def __init__(self, weights):
        self._weights, self._backups = weights, {}

    @property
    def backup_tags(self):
        return list(self._backups)

    def backup(self, tag):
        dst = self._backups.setdefault(tag, {})
        for name, w in self._weights.items():
            if name not in dst:
                dst[name] = torch.empty_like(w)
            dst[name].copy_(w)

    def host_bytes(self):
        return sum(t.numel() * t.element_size() for d in self._backups.values() for t in d.values())


def _weights():
    return {"base.w": torch.ones(256, 256), "lora.a": torch.ones(4, 256)}


def test_skipped_backup_is_not_created():
    b = _NormalBackuper(_weights())
    refresh_actor_backup(SimpleNamespace(weights_backuper=b))
    assert b.backup_tags == [] and b.host_bytes() == 0


def test_existing_backup_is_refreshed_in_place():
    w = _weights()
    b = _NormalBackuper(w)
    b.backup("actor")
    before = b.host_bytes()
    w["lora.a"].fill_(3.0)
    refresh_actor_backup(SimpleNamespace(weights_backuper=b))
    assert b.host_bytes() == before
    assert torch.equal(b._backups["actor"]["lora.a"], w["lora.a"])


def test_no_backuper_and_legacy_backuper():
    refresh_actor_backup(SimpleNamespace())
    calls = []
    refresh_actor_backup(SimpleNamespace(weights_backuper=SimpleNamespace(backup=calls.append)))
    assert calls == ["actor"]
