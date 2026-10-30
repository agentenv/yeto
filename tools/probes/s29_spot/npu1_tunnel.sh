#!/bin/bash
# S29 spot: keep a reverse SSH tunnel from npu1 to one spot VM so the VM's 127.0.0.1:29400 is the
# npu1 syncer (npu1 accepts no inbound connection except its own SSH port; the sky container shares
# the VM host's network). Reconnects until killed; a preempted and restarted VM keeps its IP or is
# re-run with the new one.
#   npu1_tunnel.sh <vm-public-ip> <ssh-user> <key-file>
IP=${1:?vm ip}
USER_=${2:?ssh user}
KEY=${3:?key file}
while true; do
  ssh -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes \
      -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o ExitOnForwardFailure=yes \
      -N -R 127.0.0.1:29400:127.0.0.1:29400 "$USER_@$IP"
  echo "[tunnel $IP] ssh exited $? at $(date -u +%FT%TZ); retry in 10 s"
  sleep 10
done
