"""Driver-side facade over the parsed Miles configuration (``IslandArgs``).

The island driver used to hang its own context on Miles' parsed namespace
(``miles_args.yeto_rl_x = ...``). Miles' configuration objects are immutable
now, so :class:`IslandArgs` keeps yeto's names in its own context and routes
Miles' names to the wrapped config. The driver code keeps its assignment
syntax; the facade decides where a name lives:

* ``yeto_*`` / ``_yeto_*`` names and :data:`YETO_PLAIN_NAMES`: yeto's context.
  The keys in :data:`ext_args.WIRE_KEYS` are copied into the config's
  ``yeto_ext`` by :meth:`IslandArgs.freeze_for_slicing`; everything else is
  driver-only and never reaches a Miles process.
* Miles' own names are read from the config (then from
  ``raw_megatron.base_args``) and refused as writes; a native field changes
  through :meth:`IslandArgs.override_before_slice` (re-validated) before the
  component configs are sliced.

On the pinned Miles the config is a mutable ``argparse.Namespace``; the
facade then also mirrors every write onto it, so the processes and the code
that still read ``args.yeto_*`` directly see exactly what they saw before.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from yeto.rl.engine.ext_attrs import ATTR_PREFIX, EXT_FIELD, WIRE_KEYS, build_ext, ext_key, install_ext

# yeto-owned names without the ``yeto_`` prefix (W&B of the island's own run)
YETO_PLAIN_NAMES: frozenset[str] = frozenset({"wandb", "wandb_entity"})

# Driver-only values (B0 class D): read by the island driver process only.
DRIVER_KEYS: frozenset[str] = frozenset({
    "rl_allow_cross_node_engine_tp",
    "rl_allow_cross_node_tp",
    "rl_base_model_revision",
    "rl_bridge_config",
    "rl_code_pack_digest",
    "rl_code_pack_mode",
    "rl_completed_groups_path",
    "rl_critic_init_sha256",
    "rl_critic_syncer_addr",
    "rl_cut_auto",
    "rl_data",
    "rl_data_dispense",
    "rl_data_position_store",
    "rl_data_revision",
    "rl_data_slot",
    "rl_edge_costs_path",
    "rl_elastic",
    "rl_elastic_window_s",
    "rl_expert_selection",
    "rl_full_parameter_config",
    "rl_heartbeat_interval_s",
    "rl_hf_exports",
    "rl_init_from_release_dir",
    "rl_init_release",
    "rl_initial_adapter",
    "rl_initial_adapter_sha256",
    "rl_island_lease_s",
    "rl_island_pool",
    "rl_island_tasks",
    "rl_layout_hash",
    "rl_learner_budget_steps",
    "rl_local_horizon",
    "rl_lora_config_hash",
    "rl_model",
    "rl_num_fragments",
    "rl_outer_sync",
    "rl_overlap_eval",
    "rl_parameter_mode",
    "rl_pipeline",
    "rl_recommend_mode",
    "rl_release_manifest",
    "rl_resource_sample_interval_s",
    "rl_resume",
    "rl_resume_every",
    "rl_rollout_model_revision",
    "rl_round_cuts",
    "rl_sample_buffer",
    "rl_sample_lane",
    "rl_source_sha256",
    "rl_sync_layout_fingerprint",
    "rl_sync_preset",
    "rl_syncer_epoch",
    "rl_total_fragment_steps",
    "rl_total_sweeps",
    "rl_unverified_mechanisms",
    "rl_verify_every",
    "rl_verify_every_round",
    "rl_ledger_set_aside_before",
    "rollout_quant_fp8_source",
    "rollout_quant_tier",
    # elastic_hook.TUNING (``--rl-auto-*`` / ``--rl-recommend-*``)
    "rl_auto_k_windows",
    "rl_auto_safety_margin_s",
    "rl_auto_horizon_s",
    "rl_auto_min_dwell_s",
    "rl_auto_cooldown_s",
    "rl_auto_max_switches",
    "rl_auto_switch_window_s",
    "rl_recommend_ttl_s",
    "rl_recommend_min_windows",
    "rl_recommend_efficiency_lower",
})

# Driver-only values that change after the island started (never cross).
DRIVER_MUTABLE_KEYS: frozenset[str] = frozenset({
    "rl_cell_id",
    "rl_eval_policy_version",
    "rl_policy_token",
})

ALL_KEYS: frozenset[str] = WIRE_KEYS | DRIVER_KEYS | DRIVER_MUTABLE_KEYS

_MISSING = object()


def is_yeto_name(name: str) -> bool:
    """A name yeto owns on the facade (``yeto_ext`` itself is Miles' field)."""
    if name == EXT_FIELD:
        return False
    return name.startswith(("yeto_", "_yeto_")) or name in YETO_PLAIN_NAMES


def _is_pydantic(obj: Any) -> bool:
    return hasattr(type(obj), "model_fields") and hasattr(obj, "model_copy")


class IslandArgs:
    """Driver-side facade: yeto names -> its own context, Miles names -> the config."""

    __slots__ = ("_miles", "_ctx", "_frozen", "_mirror")

    def __init__(self, miles: Any) -> None:
        object.__setattr__(self, "_miles", miles)
        object.__setattr__(self, "_ctx", {})
        object.__setattr__(self, "_frozen", False)
        # a mutable namespace (pinned Miles, fakes) also carries every write
        object.__setattr__(self, "_mirror", not _is_pydantic(miles))

    @property
    def miles(self) -> Any:
        """The Miles configuration (what Miles' own APIs take)."""
        return self._miles

    @property
    def frozen(self) -> bool:
        return self._frozen

    def context(self) -> dict[str, Any]:
        """A copy of yeto's context (attribute names)."""
        return dict(self._ctx)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        if is_yeto_name(name):
            try:
                return self._ctx[name]
            except KeyError:
                if self._mirror:  # a value another writer put on the namespace
                    return getattr(self._miles, name)
                raise AttributeError(name) from None
        try:
            return getattr(self._miles, name)
        except AttributeError:
            raw = getattr(self._miles, "raw_megatron", None)
            base = getattr(raw, "base_args", None) or {}
            if name in base:
                return base[name]
            raise

    def __setattr__(self, name: str, value: Any) -> None:
        if not is_yeto_name(name):
            raise AttributeError(
                f"{name} belongs to Miles: pass it in argv, set it in preprocess_args or call "
                "override_before_slice()"
            )
        if self._frozen and ext_key(name) in WIRE_KEYS and self._ctx.get(name, _MISSING) != value:
            raise AttributeError(f"{name} already crossed into Miles processes; it is frozen")
        self._ctx[name] = value
        if self._mirror:
            setattr(self._miles, name, value)

    def __delattr__(self, name: str) -> None:
        if not is_yeto_name(name):
            raise AttributeError(f"{name} belongs to Miles")
        self._ctx.pop(name, None)
        if self._mirror and hasattr(self._miles, name):
            delattr(self._miles, name)

    def __repr__(self) -> str:
        return f"IslandArgs({type(self._miles).__name__}, {len(self._ctx)} yeto names)"

    def override_before_slice(self, **updates: Any) -> None:
        """Change native Miles fields before the component configs are sliced.

        A pydantic config is re-validated as a whole; a mutable namespace is
        updated in place.
        """
        if self._frozen:
            raise RuntimeError("component configs were sliced; Miles fields are frozen")
        if _is_pydantic(self._miles):
            cls = type(self._miles)
            top, backend_updates = _split_backend_updates(self._miles, updates)
            if backend_updates:
                backend = self._miles.backend
                top["backend"] = type(backend).model_validate(dict(backend) | backend_updates)
            top.update(_mirror_megatron_base_args(self._miles, updates))
            object.__setattr__(self, "_miles", cls.model_validate(dict(self._miles) | top))
            return
        for name, value in updates.items():
            setattr(self._miles, name, value)


    def wire(self) -> dict[str, Any]:
        """The ``yeto_ext`` payload the Miles processes will see."""
        return build_ext(self._ctx)

    def freeze_for_slicing(self) -> Any:
        """Install the cross-process keys on the config and freeze them.

        Call once, right before Miles slices the component configs
        (``ArgvOrchestratorStartupInfo.create`` / ``init_orchestration_script``).
        Returns the config to pass to Miles. Driver-only names stay writable.
        """
        if not self._frozen:
            object.__setattr__(self, "_miles", install_ext(self._miles, self.wire()))
            object.__setattr__(self, "_frozen", True)
        return self._miles

    def handoff(self) -> Any:
        """The object the orchestration entry points take after freezing.

        On the pinned Miles (mutable namespace, every write mirrored) that is
        the namespace itself, exactly as before; with an immutable config the
        driver code needs the facade for its own names and ``.miles`` for Miles.
        """
        self.freeze_for_slicing()
        return self._miles if self._mirror else self


def as_island(miles_args: Any) -> IslandArgs:
    """Wrap a parsed Miles config (idempotent)."""
    return miles_args if isinstance(miles_args, IslandArgs) else IslandArgs(miles_args)


def miles_config(args: Any) -> Any:
    """The Miles configuration behind a facade (or ``args`` itself)."""
    return args.miles if isinstance(args, IslandArgs) else args


def override_native(miles_args: Any, **updates: Any) -> None:
    """Set native Miles fields on a facade (re-validated) or on a plain namespace."""
    if isinstance(miles_args, IslandArgs):
        miles_args.override_before_slice(**updates)
        return
    for name, value in updates.items():
        setattr(miles_args, name, value)


def declares_field(config: Any, name: str) -> bool:
    """Whether the Miles configuration declares native field ``name``. A
    pydantic config answers from its fields (``extra="forbid"`` refuses the
    rest); a mutable namespace (the pinned ports-v1 Miles, test fakes) takes any
    name."""
    config = miles_config(config)
    if _is_pydantic(config):
        return name in type(config).model_fields
    return True


def override_fork_field(miles_args: Any, **updates: Any) -> dict[str, str]:
    """Set fork-only Miles fields (the ports-v1 ``external_policy_*`` flags).

    A Miles whose config declares the field gets it natively
    (:func:`override_native`); one that does not (the k31 lineage, frozen and
    ``extra="forbid"``) carries the value as yeto's ``yeto_rl_<name>`` in
    ``yeto_ext`` instead, where the Miles-side readers of that lineage find it
    (``ext_attr(args, "rl_<name>")``). Returns ``{name: "native" | "ext"}``.
    """
    native = {name: value for name, value in updates.items() if declares_field(miles_args, name)}
    if native:
        override_native(miles_args, **native)
    carried = {}
    for name, value in updates.items():
        if name in native:
            carried[name] = "native"
        else:
            setattr(miles_args, ATTR_PREFIX + "rl_" + name, value)
            carried[name] = "ext"
    return carried


__all__ = [
    "ALL_KEYS",
    "DRIVER_KEYS",
    "DRIVER_MUTABLE_KEYS",
    "EXT_FIELD",
    "IslandArgs",
    "YETO_PLAIN_NAMES",
    "as_island",
    "is_yeto_name",
    "declares_field",
    "miles_config",
    "override_fork_field",
    "override_native",
]


def _split_backend_updates(miles: Any, updates: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(top-level updates, backend updates)``: the k31 pin moved some native
    fields (``wandb_project``, ``global_batch_size``, ``micro_batch_size`` ...) into
    ``config.backend`` and exposes them on the config as read-only properties. Set
    at the top they would land in the pydantic extras and clash with the backend's
    value in Miles' custom-function view (``_merge_dicts``: "conflicting values")."""
    fields = set(getattr(type(miles), "model_fields", {}) or {})
    backend = getattr(miles, "backend", None)
    backend_fields = set(getattr(type(backend), "model_fields", {}) or {}) if backend is not None else set()
    top, nested = {}, {}
    for name, value in updates.items():
        if name not in fields and name in backend_fields:
            nested[name] = value
        else:
            top[name] = value
    return top, nested


def _mirror_megatron_base_args(miles: Any, updates: Mapping[str, Any]) -> dict[str, Any]:
    """``{"raw_megatron": <re-validated>}`` when an update names a Megatron argument
    the k31 pin also keeps in ``config.raw_megatron.base_args`` (the trainer trait's
    ``Arg(reset=True)`` fields such as ``wandb_project``: the CLI writes both, and
    Miles' custom-function view refuses the two to differ -- "conflicting values");
    ``{}`` otherwise."""
    raw = getattr(miles, "raw_megatron", None)
    base_args = getattr(raw, "base_args", None)
    if not isinstance(base_args, Mapping):
        return {}
    shared = {name: value for name, value in updates.items() if name in base_args}
    if not shared:
        return {}
    return {"raw_megatron": type(raw).model_validate(dict(raw) | {"base_args": {**base_args, **shared}})}
