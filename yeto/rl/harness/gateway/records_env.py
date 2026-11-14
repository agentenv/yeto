"""Environment names of a records proxy built inside a trainer's rollout process.

Shared by the backend adapters (verl ``rollout_engine``, Miles
``records_generate``); kept here so adapters do not import each other.
"""

TEMPLATE_ENV = "YETO_RECORDS_TEMPLATE"          # chat template name (``chat_templates.TEMPLATES``)
TOKENIZER_ENV = "YETO_RECORDS_TOKENIZER"        # tokenizer directory (the checkpoint)
MODEL_ENV = "YETO_RECORDS_MODEL"                # model name the agent sends
SAMPLING_ENV = "YETO_RECORDS_SAMPLING"          # JSON signed sampling
MAX_MODEL_LEN_ENV = "YETO_RECORDS_MAX_MODEL_LEN"  # context length of the rollout engine
CARD_ENV = "YETO_RECORDS_CARD"                  # GPU name (provenance)
PRECISION_ENV = "YETO_RECORDS_PRECISION"        # weight precision the rollout serves (provenance; default bf16)
QUANT_SCHEME_ENV = "YETO_RECORDS_QUANT_SCHEME"  # release scheme the rollout serves (provenance; default: the precision)
TRUST_REMOTE_CODE_ENV = "YETO_RECORDS_TRUST_REMOTE_CODE"  # "1": run the checkpoint's tokenizer code (opt-in)

# the proxy's own server names. They live here, not in ``recorder``,
# so the launch layer can name the tunnelled port without importing aiohttp
# (``recorder`` imports ``.proxy``); ``recorder`` re-exports them.
PROXY_ENV = "YETO_RECORDS_PROXY"                  # module:callable returning a RecordsProxy
PROXY_HOST_ENV = "YETO_RECORDS_PROXY_HOST"        # bind address
PROXY_PORT_ENV = "YETO_RECORDS_PROXY_PORT"        # bind port (0: a free port)
PROXY_PUBLIC_HOST_ENV = "YETO_RECORDS_PROXY_PUBLIC_HOST"
PROXY_PUBLIC_BASE_ENV = "YETO_RECORDS_PROXY_PUBLIC_BASE"   # full URL the sandbox dials (a tunnel)
PROXY_SECRET_FILE_ENV = "YETO_RECORDS_PROXY_SECRET_FILE"   # file holding the path secret


def _torch_device_family() -> str | None:
    """``rocm`` / ``cuda`` of the torch build in this process, None without torch or a GPU build."""
    try:
        import torch
    except ImportError:
        return None
    version = getattr(torch, "version", None)
    if getattr(version, "hip", None):
        return "rocm"
    if getattr(version, "cuda", None):
        return "cuda"
    return None


def records_device_family(environ=None) -> str:
    """Device family of the trainer's rollout (provenance): ``$YETO_DEVICE_FAMILY`` (set by
    the island launcher), else the card's catalog family, else the torch build, else
    ``unknown`` -- never a guessed ``cuda``. ``cuda`` with a ROCm card reads ``rocm``; any
    other family that contradicts the card is refused."""
    import os

    from yeto.gpu_spec import DEVICE_FAMILY_ENV, UnknownCard, device_family_of

    env = os.environ if environ is None else environ
    family = (env.get(DEVICE_FAMILY_ENV) or "").strip().lower() or None
    card = (env.get(CARD_ENV) or "").strip()
    card_family = None
    if card:
        try:
            card_family = device_family_of(card)
        except UnknownCard:
            card_family = None
    if family == "cuda" and card_family == "rocm":
        family = "rocm"  # a ROCm torch build runs as the cuda device type (yeto.accel has no rocm)
    if family and card_family and family != card_family:
        raise ValueError(f"{DEVICE_FAMILY_ENV}={family} but {CARD_ENV}={card} is a {card_family} card")
    return family or card_family or _torch_device_family() or "unknown"


def records_precision(environ=None) -> str:
    """Weight precision of the trainer's rollout (provenance): ``$YETO_RECORDS_PRECISION``,
    else ``bf16`` (the trained weights a colocated rollout serves)."""
    import os

    env = os.environ if environ is None else environ
    return (env.get(PRECISION_ENV) or "").strip() or "bf16"


def records_quant_scheme(environ=None) -> str:
    """Release scheme of the trainer's rollout (provenance): ``$YETO_RECORDS_QUANT_SCHEME``,
    else the precision (:func:`records_precision`), so the two never disagree by default."""
    import os

    env = os.environ if environ is None else environ
    return (env.get(QUANT_SCHEME_ENV) or "").strip() or records_precision(env)
