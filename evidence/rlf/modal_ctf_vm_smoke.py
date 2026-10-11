import modal, json
app=modal.App.lookup('yeto-rlf-ctf-vm-smoke2', create_if_missing=True)
image=modal.Image.from_registry('docker:27-dind').add_local_dir('/home/michael/work/rlenvfactory/tests/fixtures/ctf-ret2win',remote_path='/opt/task')
script=r'''set +e
rm -rf /var/lib/docker /run/docker; mkdir -p /run/docker
(dockerd --host=unix:///run/docker/docker.sock --iptables=false --bridge=none >/tmp/dockerd.log 2>&1 &)
for i in $(seq 1 40); do docker -H unix:///run/docker/docker.sock info >/tmp/docker.info 2>&1 && break; sleep 1; done
D='docker -H unix:///run/docker/docker.sock'
$D info >/tmp/docker.info 2>&1; echo DOCKER_INFO_RC:$?
$D build --progress=plain -t rlf-ctf /opt/task >/tmp/build.log 2>&1; b=$?; echo BUILD_RC:$b; tail -30 /tmp/build.log
if [ $b -eq 0 ]; then $D run -d --network none -e FLAG='flag{vm-smoke}' rlf-ctf >/tmp/cid; c=$(cat /tmp/cid); sleep 3; echo CONTAINER_ID:$c; $D ps -a; $D logs $c; $D rm -f $c; echo CLEAN_RC:$?; fi
echo DOCKER_LOG; tail -20 /tmp/dockerd.log
'''
sb=modal.Sandbox.create('sh','-lc',script,app=app,image=image,runtime='vm',cpu=2,memory=4096,timeout=600,idle_timeout=60,tags={'yeto-purpose':'rlf-ctf-vm-smoke'})
try: print(json.dumps({'sandbox_id':sb.object_id,'stdout':''.join(list(sb.stdout)),'stderr':''.join(list(sb.stderr))}))
finally: sb.terminate()
