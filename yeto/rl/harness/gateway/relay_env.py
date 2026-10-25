"""Environment names of the Harbor agent's gate relay (S26 E9.5; S27 G-DA).

A dependency-free module so the launch layer (Modal islands: the relay tunnel)
and the Miles adapter (``harbor_agent.relay_settings``) share one spelling
without the launch layer importing an adapter.
"""

RELAY_PORT_ENV = "YETO_HARBOR_GATE_RELAY_PORT"
RELAY_HOST_ENV = "YETO_HARBOR_GATE_RELAY_HOST"
RELAY_PUBLIC_HOST_ENV = "YETO_HARBOR_GATE_RELAY_PUBLIC_HOST"
RELAY_PUBLIC_BASE_ENV = "YETO_HARBOR_GATE_RELAY_PUBLIC_BASE"
RELAY_SECRET_FILE_ENV = "YETO_HARBOR_GATE_RELAY_SECRET_FILE"
