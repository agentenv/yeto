#!/bin/bash
# hard cost cap: terminate lrfix sandboxes 1900s after creation
cd /tmp/lrfix/ctl; P=/tmp/modal-venv/bin/python
for i in $(seq 400); do
  for t in p2 l2; do
    [ -f $t.t_create ] && [ ! -f $t.t_down ] && [ $(( $(date -u +%s) - $(cat $t.t_create) )) -gt 1900 ] && [ -f $t.sid ] && { $P sbx.py kill $(cat $t.sid); echo "WATCHDOG killed $t"; touch $t.killed; }
  done
  [ -f p2.t_down ] && [ -f l2.t_down ] && exit 0
  sleep 15
done
