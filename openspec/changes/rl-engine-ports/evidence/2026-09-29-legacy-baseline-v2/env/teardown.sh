#!/bin/bash
# kill every sandbox in app yeto-legacy-baseline and stop the app
cd /home/michael/work/gpu-legacy/ctl2; P=/tmp/modal-venv/bin/python
for s in $($P lsbx.py list 2>/dev/null | grep '^sb-'); do $P lsbx.py kill $s 2>&1 | grep -v -i deprec; done
/tmp/modal-venv/bin/modal app stop yeto-legacy-baseline -y 2>&1 | tail -1 || true
