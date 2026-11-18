#!/bin/bash
# Node entrypoint: the head's public key, host keys, the docker shim when the
# node simulates a `runtime: docker` node (BM_SIM_DOCKER=1), then sshd.
set -eu
install -d -m 700 -o yeto -g yeto /home/yeto/.ssh
install -m 600 -o yeto -g yeto /etc/bm-sim/authorized_keys /home/yeto/.ssh/authorized_keys
if [ "${BM_SIM_DOCKER:-0}" = 1 ]; then ln -sf /usr/local/lib/bm-sim/docker /usr/local/bin/docker; fi
# the card tool reports what this node pretends to have (inventory: card, count, HBM)
printf '%s\n' "${BM_SIM_CARDS:-2}" "${BM_SIM_CARD_NAME:-NVIDIA B300 SXM6 AC}" "${BM_SIM_CARD_MIB:-275040}" \
  > /etc/bm-sim/cards
ssh-keygen -A >/dev/null
exec /usr/sbin/sshd -D -e
