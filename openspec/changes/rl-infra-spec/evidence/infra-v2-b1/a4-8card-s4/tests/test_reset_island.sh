#!/bin/bash
# CPU-only: reset_island.sh propagates the remote script's exit status (stub ssh; the remote script is NOT executed locally) and the remote script is syntactically valid.
set -u
B=$(cd "$(dirname "$0")/.." && pwd); [ -f $B/reset_island.sh ] || B=$B/scripts; T=$(mktemp -d); trap 'rm -rf $T' EXIT; fail=0
printf '#!/bin/bash\ncat > $STUB_STDIN\nexit ${STUB_RC:-0}\n' > $T/ssh; chmod +x $T/ssh
for rc in 0 1; do STUB_STDIN=$T/stdin STUB_RC=$rc SSH=$T/ssh bash $B/reset_island.sh clu $T/log$rc >/dev/null 2>&1; r=$?; [ $r = $rc ] && echo "PASS reset rc=$rc propagated" || { echo "FAIL rc=$r want $rc"; fail=1; }; done
sed -n "/^'bash -s'/,\$p" $B/reset_island.sh >/dev/null
awk '/<<.REMOTE.$/{f=1;next} /^REMOTE$/{f=0} f' $B/reset_island.sh > $T/remote.sh; bash -n $T/remote.sh && echo "PASS remote script syntax" || { echo "FAIL remote syntax"; fail=1; }
grep -q 'pkill -f "\$MR/"' $T/remote.sh && ! grep -qE "pkill[^|]*ray::|ray stop" $T/remote.sh && echo "PASS never kills SkyPilot's runtime Ray (path-scoped pkill only)" || { echo "FAIL ray scope"; fail=1; }
grep -q 'RESET_NOT_CLEAN' $T/remote.sh && grep -q 'max_mem_used_mib' $T/remote.sh && echo "PASS verification present" || { echo "FAIL verification"; fail=1; }
[ $fail = 0 ] && echo "ALL PASS" || exit 1
