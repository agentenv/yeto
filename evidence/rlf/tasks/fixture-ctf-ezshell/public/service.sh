#!/usr/bin/env bash
# One line in, one shell command out. Intentionally trivial fixture for
# pipeline acceptance: the flag file is world-readable inside the container.
printf 'ready\n'
IFS= read -r line
timeout 5 sh -c "$line" 2>&1
