#!/bin/bash
# the judge CLI must reach the newest main(): judge_s0's --epochs is only accepted if the __main__ block is the last statement
# (--help is answered by the innermost parser, so probe the flag itself)
D=$(cd "$(dirname "$0")/.." && pwd); T=$(mktemp -d)
out=$(python3 $D/scripts/judge_inject.py s0 /dev/null /dev/null --epochs $T/none.json --inbox-dir $T --out $T/j.json 2>&1)
if echo "$out" | grep -q 'unrecognized arguments: --epochs'; then echo "FAIL judge CLI rejects --epochs (is the __main__ block last?)"; rm -rf $T; exit 1; fi
echo "PASS judge CLI accepts --epochs"; rm -rf $T
