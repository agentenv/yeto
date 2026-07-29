"""Apply the coordinator's authoritative final cut on torch islands."""

from __future__ import annotations

import torch
import torch.distributed as dist

from .protocol import DTYPE_F32, FinalManifest, SyncerClient
from .tensor_io import apply_fragment, unpack_fragment


def finalize_torch_island(
    client: SyncerClient | None,
    layout,
    params,
    *,
    rank: int,
    world: int,
    device: torch.device,
) -> FinalManifest:
    """Overwrite every trainable fragment with the exact manifested global.

    Rank 0 owns the WAN client. Once its control stream sees a final
    manifest, all island ranks enter this blocking terminal phase together.
    Raw coordinator values are broadcast within the island and applied with
    no normal ``merge_alpha`` blend. The ACK is emitted only after every rank
    has installed the complete cut.
    """
    if rank == 0:
        assert client is not None
        manifest, broadcasts = client.wait_for_final_fragments()
        actions = [
            (
                update.fragment_id,
                update.version,
                unpack_fragment(
                    layout.fragments[update.fragment_id],
                    update.data,
                    DTYPE_F32,
                ),
            )
            for update in broadcasts
        ]
        manifest_data = (manifest.global_step, manifest.versions)
    else:
        manifest = None
        actions = []
        manifest_data = None

    if world > 1:
        box = [manifest_data]
        dist.broadcast_object_list(box, src=0)
        global_step, versions = box[0]
        manifest = FinalManifest(global_step, tuple(versions))
        if rank != 0:
            actions = [
                (fid, version, torch.empty(layout.fragments[fid].numel))
                for fid, version in enumerate(manifest.versions)
            ]

    assert manifest is not None
    for fid, version, flat in actions:
        expected = manifest.versions[fid]
        if version != expected:
            raise RuntimeError(
                f"final fragment {fid} has version {version}, expected {expected}"
            )
        flat = flat.to(device)
        if world > 1:
            dist.broadcast(flat, src=0)
        # Terminal overwrite: delayed-application blending is intentionally
        # disabled so the save path observes the coordinator's raw value.
        apply_fragment(layout.fragments[fid], flat, params)

    if world > 1:
        dist.barrier()
    if rank == 0:
        assert client is not None
        client.acknowledge_finalization(manifest)
    if world > 1:
        dist.barrier()
    return manifest
