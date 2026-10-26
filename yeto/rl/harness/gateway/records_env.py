"""Environment names of a records proxy built inside a trainer's rollout process (S25 N10.9).

Shared by the backend adapters (verl ``rollout_engine``, Miles
``records_generate``); kept here so adapters do not import each other.
"""

TEMPLATE_ENV = "YETO_RECORDS_TEMPLATE"          # chat template name (``chat_templates.TEMPLATES``)
TOKENIZER_ENV = "YETO_RECORDS_TOKENIZER"        # tokenizer directory (the checkpoint)
MODEL_ENV = "YETO_RECORDS_MODEL"                # model name the agent sends
SAMPLING_ENV = "YETO_RECORDS_SAMPLING"          # JSON signed sampling
MAX_MODEL_LEN_ENV = "YETO_RECORDS_MAX_MODEL_LEN"  # context length of the rollout engine
CARD_ENV = "YETO_RECORDS_CARD"                  # GPU name (provenance)

# S27 G-FS: the proxy's own server names. They live here, not in ``recorder``,
# so the launch layer can name the tunnelled port without importing aiohttp
# (``recorder`` imports ``.proxy``); ``recorder`` re-exports them.
PROXY_ENV = "YETO_RECORDS_PROXY"                  # module:callable returning a RecordsProxy
PROXY_HOST_ENV = "YETO_RECORDS_PROXY_HOST"        # bind address
PROXY_PORT_ENV = "YETO_RECORDS_PROXY_PORT"        # bind port (0: a free port)
PROXY_PUBLIC_HOST_ENV = "YETO_RECORDS_PROXY_PUBLIC_HOST"
PROXY_PUBLIC_BASE_ENV = "YETO_RECORDS_PROXY_PUBLIC_BASE"   # full URL the sandbox dials (a tunnel)
PROXY_SECRET_FILE_ENV = "YETO_RECORDS_PROXY_SECRET_FILE"   # file holding the path secret
