# Daytona snapshot of the K3.1 vLLM ROCm sampler (MI355X, gfx950).
#
# The pinned sampler image (yeto/rl/sampler_pins.py VLLM_SAMPLER_IMAGE_ROCM) plus what the
# no-docker sampler route (yeto/launch/daytona_sampler.py) needs inside the sandbox: the
# sampler's pure-Python dependencies in the runtime venv, awscli (base model pull, store
# task sync), openssh-client (sandbox docker over SSH, reverse tunnel), the docker CLI
# (client only: an agentic sampler drives the sandbox host's daemon over SSH), and /work +
# /models writable by the sandbox user. The yeto tree is not baked in: every start uploads
# the launcher's commit to /work/yeto/<sha>.
#
# Build for linux/amd64, push to a registry Daytona can pull, then register a GPU snapshot
# from the pushed digest with 8 x MI355X, memory and disk as in docs/DAYTONA_SAMPLER.md
# (a sandbox created from a snapshot takes the snapshot's resources).
FROM docker.io/vllm/vllm-openai-rocm@sha256:5f47cc1004d9a7ac65f8a913725edb20c62c99a07e31cbf4ba21741bd0b4f9b1

ARG DOCKER_CLI_VERSION=29.0.2
ARG COMPOSE_VERSION=2.40.3

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      openssh-client curl ca-certificates procps iproute2 tar util-linux \
 && rm -rf /var/lib/apt/lists/*

# the sampler's imports the base may lack, and awscli, in the runtime venv
RUN /opt/runtime-venv/bin/python3 -m pip install --no-cache-dir \
      "boto3>=1.34" "safetensors>=0.4" "zstandard>=0.22" "httpx>=0.27" "awscli>=1.33" \
 && ln -sf /opt/runtime-venv/bin/aws /usr/local/bin/aws

RUN curl -fsSL "https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_CLI_VERSION}.tgz" \
      | tar xz --strip-components=1 -C /usr/local/bin docker/docker \
 && mkdir -p /usr/local/lib/docker/cli-plugins \
 && curl -fsSL -o /usr/local/lib/docker/cli-plugins/docker-compose \
      "https://github.com/docker/compose/releases/download/v${COMPOSE_VERSION}/docker-compose-linux-x86_64" \
 && chmod 755 /usr/local/bin/docker /usr/local/lib/docker/cli-plugins/docker-compose

RUN mkdir -p /work /models /tunnel && chmod 1777 /work /models && chmod 700 /tunnel

ENV PYTHONUNBUFFERED=1
ENTRYPOINT []
CMD ["sleep", "infinity"]
