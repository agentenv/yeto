# bm-sim node image: one simulated bare-metal node (sshd, Ray, the yeto runtime
# deps on CPU). The yeto checkout is not baked in: the harness mounts it
# read-only at the trainer workdir, as a customer node holds the code pack.
FROM python:3.12-slim-bookworm
RUN apt-get update \
 && apt-get install -y --no-install-recommends openssh-server procps iproute2 util-linux coreutils \
      ca-certificates git \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.8.0" \
 && pip install --no-cache-dir "ray[default]==2.48.0" "boto3>=1.34" pyyaml numpy cloudpickle \
      "safetensors>=0.4" "huggingface-hub>=0.34" "zstandard>=0.22" protobuf
RUN useradd -m -s /bin/bash yeto && usermod -p "*" yeto \
 && mkdir -p /run/sshd /etc/bm-sim /tmp/bm-sim-mounts && chmod 1777 /tmp/bm-sim-mounts \
 && ln -s /tmp/bm-sim-mounts/yeto-island-state /yeto-island-state \
 && printf 'PasswordAuthentication no\nKbdInteractiveAuthentication no\nUsePAM no\nPermitRootLogin no\nAllowUsers yeto\n' \
      > /etc/ssh/sshd_config.d/bm-sim.conf
COPY node_init.sh /usr/local/bin/bm-sim-node-init
COPY nvidia-smi /usr/local/bin/nvidia-smi
COPY docker-shim /usr/local/lib/bm-sim/docker
RUN chmod 755 /usr/local/bin/bm-sim-node-init /usr/local/bin/nvidia-smi /usr/local/lib/bm-sim/docker
CMD ["/usr/local/bin/bm-sim-node-init"]
