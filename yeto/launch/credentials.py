"""Cloud credentials, secret env splitting and container-registry logins."""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..gpu_spec import parse_gpu_spec


# Where each cloud's own tooling stores credentials on the submitting
# machine, and the env vars that stand in for the file. The head VM gets
# the files mounted (or the vars forwarded) for every cloud its fleet
# touches — and nothing for clouds it does not.
CLOUD_CREDENTIAL_PATHS: dict[str, tuple[str, ...]] = {
    "aws": ("~/.aws",),
    "gcp": ("~/.config/gcloud",),
    "runpod": ("~/.runpod",),
    "nebius": ("~/.nebius",),
    "verda": ("~/.verda",),
    "modal": ("~/.modal.toml",),
}
CLOUD_CREDENTIAL_ENV: dict[str, tuple[str, ...]] = {
    "aws": ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
    "runpod": ("RUNPOD_API_KEY",),
    "nebius": ("NEBIUS_IAM_TOKEN", "NEBIUS_TENANT_ID"),
    "verda": ("VERDA_CLIENT_ID", "VERDA_CLIENT_SECRET"),
    "modal": ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"),
}
# Everything a learner island may legitimately receive; anything in
# CLOUD_CREDENTIAL_ENV or CLOUD_CREDENTIAL_PATHS must never reach one.
CLOUD_CREDENTIAL_ENV_NAMES = frozenset(v for vals in CLOUD_CREDENTIAL_ENV.values() for v in vals)

# secret-handling-hardening: env values that are secrets. They go to sky
# ``secrets=`` (redacted in sky's records), never to plain ``envs=``.
SECRET_ENV_NAMES = frozenset({
    "HF_TOKEN", "WANDB_API_KEY", "CYBERGYM_API_KEY",
    "YETO_SANDBOX_MODAL_TOKEN_ID", "YETO_SANDBOX_MODAL_TOKEN_SECRET", "YETO_ISLAND_HMAC_KEY",
    # yeto.cloud.s3_store.SECRET_ENVS: the s3 store credential (a test keeps them equal)
    "YETO_RL_STORE_S3_SECRET_ACCESS_KEY", "YETO_RL_STORE_S3_SESSION_TOKEN",
})


def _secret_value(value) -> str:
    """Plain text of a task secret: sky.Task stores pydantic SecretStr."""
    getter = getattr(value, "get_secret_value", None)
    return str(getter() if callable(getter) else value)


def split_secret_envs(envs) -> tuple[dict[str, str], dict[str, str]]:
    """(plain envs, secret envs): every name in SECRET_ENV_NAMES moves to secrets."""
    envs = dict(envs or {})
    secrets = {k: envs.pop(k) for k in list(envs) if k in SECRET_ENV_NAMES}
    return envs, secrets


def head_cloud(args) -> str:
    """The cloud the head/syncer VM is placed on (`--syncer-region` is
    'region' for AWS or 'cloud/region')."""
    region = getattr(args, "syncer_region", None) or ""
    return region.split("/", 1)[0].lower() if "/" in region else "aws"


def fleet_clouds(args) -> list[str]:
    """Every cloud this launch touches: the learner islands' plus the head's."""
    clouds = [head_cloud(args)] if getattr(args, "controller", "local") == "head" else []
    if getattr(args, "gpu", None):
        clouds += [s.cloud for s in parse_gpu_spec(args.gpu)]
    return list(dict.fromkeys(clouds))


def head_cloud_credentials(clouds: list[str], environ=None) -> tuple[dict[str, str], dict[str, str]]:
    """(file_mounts, envs) that carry each cloud's credentials onto the
    head. A cloud with neither its file(s) nor its env vars present is a
    ValueError naming the expected path — before the head is submitted.
    GCP is only ever optional (gs:// output uploads)."""
    environ = os.environ if environ is None else environ
    mounts: dict[str, str] = {}
    envs: dict[str, str] = {}
    for cloud in clouds:
        paths = CLOUD_CREDENTIAL_PATHS.get(cloud)
        if paths is None:
            continue  # a cloud sky handles with no local files we know of
        present = [p for p in paths if os.path.exists(os.path.expanduser(p))]
        for p in present:
            mounts[p] = os.path.expanduser(p)
        if present:
            continue
        env_names = CLOUD_CREDENTIAL_ENV.get(cloud, ())
        if env_names and all(environ.get(v) for v in env_names):
            for v in env_names:
                envs[v] = environ[v]
            continue
        if cloud == "gcp":
            continue
        hint = f" or env {' + '.join(env_names)}" if env_names else ""
        raise ValueError(
            f"{cloud} credentials not found at {', '.join(paths)}{hint}; the head VM "
            f"needs them to launch and tear down {cloud} islands (see docs/CLOUDS.md)"
        )
    gcloud = os.path.expanduser("~/.config/gcloud")
    if os.path.isdir(gcloud):
        mounts.setdefault("~/.config/gcloud", gcloud)  # enables gs:// --output
    if "nebius" in clouds:
        # Nebius binds one project to one region and sky reads the project
        # per region from its config; the head re-runs the project check and
        # provisions Nebius islands itself, so it needs the same file.
        # (Presence is checked before submission by check_cloud_prerequisites.)
        from ..shape.providers import SKY_CONFIG_PATH

        sky_config = os.path.expanduser(SKY_CONFIG_PATH)
        if os.path.isfile(sky_config):
            mounts[SKY_CONFIG_PATH] = sky_config
    return mounts, envs


def docker_config_login(image_ref: str | None, path: str | Path | None = None) -> dict[str, str] | None:
    """The ``~/.docker/config.json`` ``auths`` entry for ``image_ref``'s
    registry as a SKYPILOT_DOCKER_* triple, or None when the file or the
    entry is missing (credential helpers are not consulted)."""
    from ..modal_runner import DOCKER_LOGIN_ENV_VARS, registry_host

    if not image_ref:
        return None
    path = Path(path) if path else Path.home() / ".docker" / "config.json"
    try:
        auths = json.loads(path.read_text(encoding="utf-8")).get("auths") or {}
    except (OSError, ValueError):
        return None
    host = registry_host(image_ref)
    entry = next((auths[k] for k in (host, f"https://{host}", f"https://{host}/v1/") if isinstance(auths.get(k), dict)), None)
    if not entry or not entry.get("auth"):
        return None
    import base64

    try:
        user, password = base64.b64decode(entry["auth"]).decode().split(":", 1)
    except (ValueError, UnicodeDecodeError):
        return None
    if not user or not password:
        return None
    return dict(zip(DOCKER_LOGIN_ENV_VARS, (user, password, host)))


def registry_login_for(args, environ=None) -> dict[str, str] | None:
    """The private-registry login a launch injects for ``--rl-image``, or
    None (the default: a public image, nothing injected anywhere -- not the
    island task secrets, not the Modal pull secret, not the head job).

    Injection happens only when (a) the SKYPILOT_DOCKER_* variables for
    the image's registry are in ``environ`` (ports engine only, as before),
    or (b) ``--rl-image-private`` was given: then the environment login is
    used if present, else the registry's ``~/.docker/config.json`` entry;
    neither being available is an error rather than a silent public pull."""
    from ..modal_runner import registry_credentials

    from ..rl import default_rl_image

    environ = os.environ if environ is None else environ
    # before prepare_launch_args (the head CLI) --rl-image may still be unset
    image = getattr(args, "rl_image", None) or default_rl_image()
    explicit = bool(getattr(args, "rl_image_private", False))
    login = registry_credentials(image, environ)
    if login is None and explicit:
        login = docker_config_login(image)
        if login is None:
            raise ValueError(
                f"--rl-image-private: no registry login for {image!r}: set "
                "SKYPILOT_DOCKER_USERNAME/PASSWORD/SERVER or add the registry to ~/.docker/config.json"
            )
    return login


def _sky_docker_login_config(login: dict[str, str]):
    """SkyPilot's DockerLoginConfig for a SKYPILOT_DOCKER_* triple."""
    from sky.provision.docker_utils import DockerLoginConfig

    return DockerLoginConfig.from_env_vars(login)
