#!/bin/bash
# runs on the island: pick a python that can import ray, then run fork_probe.py with the args given
cands=""
p=$(pgrep -f 'ray::' | head -1); [ -n "$p" ] && cands="$cands $(readlink /proc/$p/exe 2>/dev/null)"
r=$(which ray 2>/dev/null); [ -n "$r" ] && cands="$cands $(head -1 "$r" </dev/null | sed 's/^#!//')"
cands="$cands /opt/sglang/bin/python3 $(ls /opt/*/bin/python3 2>/dev/null) $(which -a python3 python 2>/dev/null)"
exe=""
for c in $cands; do
  if [ -x "$c" ] && $c -c 'import ray,sys; sys.exit(0 if ray.__version__ != "2.9.3" else 1)' </dev/null 2>/dev/null; then exe=$c; break; fi
done
lp=$(pgrep -f '[y]eto.rl.learner' | head -1); ra=""
[ -n "$lp" ] && ra=$(tr '\0' '\n' < /proc/$lp/environ 2>/dev/null | grep '^RAY_ADDRESS=' | cut -d= -f2-)
[ -n "$ra" ] || { for h in $(hostname -I | awk '{print $1}') 127.0.0.1; do (exec 3<>/dev/tcp/$h/6379) 2>/dev/null && { ra="$h:6379"; break; }; done; }
[ -n "$ra" ] || ra="127.0.0.1:6379"
export RAY_ADDRESS=$ra
echo "RAY_ADDRESS=$ra" 1>&2
echo "EXE=$exe candidates=$cands" 1>&2
[ -n "$exe" ] || exit 9
exec $exe ~/yeto-rl/fork_probe.py "$@"
