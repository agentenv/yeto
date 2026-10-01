"""yeto GPU ids <-> bundles of the placement group Miles created at startup (E3 4.7 glue).

A pool GPU id (``ElasticWiring.pool_gpus``, the ids the resources manifest's
``placement`` lists) is the LOGICAL bundle position of the fork-M1 placement
map: ``pool_gpus[p]`` is logical bundle ``p``. The startup views
(``RayWorkerManager.get_pg_view`` for ``actor``/``rollout``/``standby``,
captured once before any view is re-pointed) give each logical position its
reordered bundle index and GPU id:

* with a placement map, role view ``r`` is ``full[pm[r]]`` (fork
  ``_create_placement_groups_from_map``), so position ``p`` is entry
  ``pm[r].index(p)`` of view ``r``;
* without one (offset layout) the ``actor`` view is the whole group, so
  position ``p`` is its entry ``p``.

:meth:`view_for` builds the ``PlacementGroupInfo`` slice for ``set_pg_view``
(trainer rebuild view / rollout cell rebind); :meth:`gpus_for_bundles` maps a
cell's bundles (``get_cell_bundles``) back to pool GPU ids.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

ROLE_VIEWS = {"trainer": "actor", "rollout": "rollout", "standby": "standby"}


class BundleMapError(RuntimeError):
    pass


class StartupBundles:
    def __init__(self, *, pool_gpus: Sequence[str], views: Mapping[str, Any],
                 placement_map: Mapping[str, Sequence[int]] | None) -> None:
        self.pool_gpus = tuple(str(g) for g in pool_gpus)
        if len(set(self.pool_gpus)) != len(self.pool_gpus):
            raise BundleMapError(f"pool GPU ids repeat: {self.pool_gpus}")
        self._entry: dict[str, tuple[Any, Any, int, int]] = {}  # gpu -> (info type, pg, bundle, gpu id)
        for p, gpu in enumerate(self.pool_gpus):
            if placement_map is None:
                role, index = "trainer", p
            else:
                owners = [r for r, positions in placement_map.items() if p in list(positions)]
                if len(owners) != 1:
                    raise BundleMapError(f"logical bundle {p} ({gpu}) belongs to roles {owners}")
                role = owners[0]
                index = list(placement_map[role]).index(p)
            info = views.get(ROLE_VIEWS.get(role, role))
            if info is None:
                raise BundleMapError(f"no startup view for role {role!r}")
            bundles, gpus = list(info.pg_reordered_bundle_indices), list(info.pg_reordered_gpu_ids)
            if index >= len(bundles):
                raise BundleMapError(f"{gpu}: position {index} outside the {role} view ({len(bundles)})")
            self._entry[gpu] = (type(info), info.pg, bundles[index], gpus[index])
        self._by_bundle = {e[2]: g for g, e in self._entry.items()}

    def view_for(self, gpus: Sequence[str]) -> Any:
        gpus = [str(g) for g in gpus]
        missing = [g for g in gpus if g not in self._entry]
        if missing or not gpus:
            raise BundleMapError(f"GPUs {missing or gpus} are not pool GPUs {self.pool_gpus}")
        entries = [self._entry[g] for g in gpus]
        info_type, pg = entries[0][0], entries[0][1]
        if any(e[1] is not pg for e in entries):
            raise BundleMapError("GPUs span more than one placement group")
        return info_type(pg, [e[2] for e in entries], [e[3] for e in entries])

    def gpus_for_bundles(self, bundles: Sequence[int]) -> tuple[str, ...]:
        unknown = [b for b in bundles if b not in self._by_bundle]
        if unknown:
            raise BundleMapError(f"bundles {unknown} are not in the startup placement group")
        return tuple(self._by_bundle[b] for b in bundles)
