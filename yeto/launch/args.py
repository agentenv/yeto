"""Launch-argument preparation shared by every run and the cloud prerequisite checks."""

from __future__ import annotations

import os

from ..gpu_spec import ClusterSpec, parse_gpu_spec
from . import REPO_ROOT
from .rl_backend import resolve_default_rl_image
from .loss import pickled_loss_path, resolve_loss_function
from .storage import modal_checkpoint_store
from .rl_args import _prepare_rl_args
from .codex import modal_sandbox_secret_name


def prepare_launch_args(
    args,
    *,
    allow_local_rl_data: bool = False,
    allow_remote_rl_model: bool = False,
) -> None:
    """Resolve immutable inputs and executable artifacts before cloud spend."""
    # --rl-allow-local-data (ports) is honoured by every caller (CLI, worker,
    # head second hop), not only the one that parsed the flag.
    allow_local_rl_data = allow_local_rl_data or bool(
        getattr(args, "rl_allow_local_data", False)
    )

    from ..provenance import (
        file_sha256,
        pin_runtime_provenance,
        python_spec_path,
        python_spec_sha256,
        verify_source_tree_sha256,
    )

    args.source_sha256 = verify_source_tree_sha256(
        getattr(args, "source_sha256", None)
    )
    payload = pin_runtime_provenance(
        args, allow_remote_model=allow_remote_rl_model
    )
    args.model_requested_identifier = payload["model"]["requested_identifier"]
    args.model_requested_revision = payload["model"]["requested_revision"]
    if "dataset" in payload:
        args.data_requested_identifier = payload["dataset"]["requested_identifier"]
        args.data_requested_revision = payload["dataset"]["requested_revision"]
    from ..adapter_lifecycle import (
        inspect_parent_adapter,
        prepare_parent_source,
        selected_parent,
    )

    prepare_parent_source(args)
    expected_loss_sha256 = getattr(args, "loss_sha256", None)
    args.loss_function = resolve_loss_function(
        args.loss_function,
        allow_unsafe_pickled_loss=bool(
            getattr(args, "allow_unsafe_pickled_loss", False)
        ),
    )
    if args.loss_function.startswith("pickle:"):
        actual_loss_sha256 = file_sha256(pickled_loss_path(args.loss_function))
        if (
            expected_loss_sha256 is not None
            and actual_loss_sha256 != expected_loss_sha256.lower()
        ):
            raise ValueError(
                "pickled loss SHA256 mismatch: expected "
                f"{expected_loss_sha256.lower()}, got {actual_loss_sha256}"
            )
        args.loss_sha256 = actual_loss_sha256
    elif args.loss_function.startswith("custom:"):
        args.loss_sha256 = file_sha256(args.loss_function.split(":", 2)[1])
    else:
        args.loss_sha256 = None
    _parent_mode, parent_source = selected_parent(args)
    if parent_source is not None:
        from ..datasource import kind

        if kind(parent_source) == "local":
            # Fail on malformed artifacts and recipe drift on the submitter
            # (and again on every learner), before provisioning GPU capacity.
            inspect_parent_adapter(
                args,
                expected_sha256=args.adapter_sha256,
            )
    adapter_spec = getattr(args, "diffusion_adapter", None)
    if adapter_spec:
        adapter_path = python_spec_path(adapter_spec, base_dir=REPO_ROOT)
        try:
            relative_adapter_path = adapter_path.relative_to(REPO_ROOT.resolve())
        except ValueError as exc:
            raise ValueError(
                f"diffusion adapter source {adapter_path} is outside the synced "
                "Yeto workdir; copy it into the repository before launch"
            ) from exc
        target, separator, factory_name = adapter_spec.partition(":")
        if target.endswith(".py") or os.path.sep in target:
            args.diffusion_adapter = (
                f"{relative_adapter_path.as_posix()}{separator}{factory_name}"
            )
            adapter_sha256 = file_sha256(adapter_path)
        else:
            adapter_sha256 = python_spec_sha256(adapter_spec)
        expected_adapter_sha256 = getattr(args, "diffusion_adapter_sha256", None)
        if (
            expected_adapter_sha256 is not None
            and adapter_sha256 != expected_adapter_sha256.lower()
        ):
            raise ValueError(
                "diffusion adapter SHA256 mismatch: expected "
                f"{expected_adapter_sha256.lower()}, got {adapter_sha256}"
            )
        args.diffusion_adapter_sha256 = adapter_sha256
    resolve_default_rl_image(args)
    if getattr(args, "gpu", None):
        check_cloud_prerequisites(parse_gpu_spec(args.gpu), args=args)
    _prepare_rl_args(
        args,
        allow_local_data=allow_local_rl_data,
        allow_remote_model=allow_remote_rl_model,
    )


def check_cloud_prerequisites(
    specs: list[ClusterSpec],
    project_ids: dict[str, str] | None = None,
    args=None,
    modal_ok: bool | None = None,
) -> None:
    """Per-cloud facts that must hold before any cloud spend.

    Nebius binds one project to one region, and sky reads the project for
    a region from `nebius.region_configs.<region>.project_id` in its
    config; a fleet naming a Nebius region without one fails at provision
    time, after the head VM is up. Fail here instead, naming every region
    that lacks a project. `project_ids` is injectable for tests.

    Modal islands have scheduling rules Modal only enforces at launch
    (whole nodes for multi-container groups), need a token on this
    machine, cannot mount object-store data, and (for RL) need the Miles
    image pinned by digest. `modal_ok` overrides the token check in tests.
    """
    if args is not None and getattr(args, "modal_sandbox_secret", None):
        for s in specs:
            modal_sandbox_secret_name(args, s)  # non-Modal islands: refused
    nebius_regions = sorted({s.region for s in specs if s.cloud == "nebius" and s.region})
    if nebius_regions:
        from ..shape.providers import SKY_CONFIG_PATH, nebius_project_ids

        have = project_ids if project_ids is not None else nebius_project_ids()
        missing = [r for r in nebius_regions if r not in have]
        if missing:
            raise ValueError(
                "Nebius needs a project per region: no project_id for "
                f"{', '.join(missing)} under nebius.region_configs in "
                f"{SKY_CONFIG_PATH} (one Nebius project is bound to one region)"
            )
    modal_specs = [s for s in specs if s.cloud == "modal"]
    store = str(getattr(args, "rl_checkpoint_store", None) or "") if args is not None else ""
    if store:
        # rl-resume-from-checkpoint 1.2: no silent drop; each cloud gets a store it can mount
        if modal_specs:
            modal_checkpoint_store(args)  # raises on a non-modal-volume store
        if store.startswith("modal-volume://") and len(modal_specs) != len(specs):
            raise ValueError("--rl-checkpoint-store modal-volume://... only mounts on Modal islands; "
                             "use a bucket URI or a shared path for the other clouds")
    if modal_specs:
        from ..modal_runner import (
            image_ref_from_rl_image,
            modal_available,
            modal_credential_hint,
            validate_modal_shape,
        )

        for spec in modal_specs:
            validate_modal_shape(spec.gpu, spec.gpus_per_node, spec.num_nodes)
        if not (modal_ok if modal_ok is not None else modal_available()):
            raise ValueError(f"modal islands need a Modal token: {modal_credential_hint()}")
        if args is not None:
            if getattr(args, "training_mode", "sft") == "rl":
                resolve_default_rl_image(args)
                image_ref_from_rl_image(getattr(args, "rl_image", "") or "")
            data = getattr(args, "data", None)
            if data:
                from ..datasource import kind as data_kind

                if data_kind(data) == "cloud":
                    raise ValueError(
                        f"modal islands cannot mount object-store data ({data}); "
                        "pass an HF dataset id or a local path"
                    )
            override = getattr(args, "syncer_public_addr", None)
            if override and ":" not in override:
                raise ValueError("--syncer-public-addr must be HOST:PORT")
