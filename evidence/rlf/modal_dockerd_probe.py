import modal, json
app=modal.App.lookup('yeto-rlf-dockerd-probe', create_if_missing=True)
image=modal.Image.from_registry('docker:27-dind')
script=r'''set -eu
rm -rf /var/lib/docker /run/docker
mkdir -p /run/docker
(dockerd --host=unix:///run/docker/docker.sock --iptables=false --bridge=none > /tmp/dockerd.log 2>&1 & echo $! > /tmp/dockerd.pid)
for i in $(seq 1 40); do docker -H unix:///run/docker/docker.sock info >/tmp/docker.info 2>&1 && break; sleep 1; done
if docker -H unix:///run/docker/docker.sock info >/tmp/docker.info 2>&1; then echo RLF_DOCKER_READY; cat /tmp/docker.info | grep -E 'Server Version|Storage Driver' || true; else echo RLF_DOCKER_FAILED; tail -40 /tmp/dockerd.log; exit 1; fi
'''
sb=modal.Sandbox.create('sh','-lc',script,app=app,image=image,runtime='vm',cpu=2,memory=4096,timeout=180,idle_timeout=60,tags={'yeto-purpose':'rlf-dockerd-probe'})
try: print(json.dumps({'sandbox_id':sb.object_id,'stdout':''.join(list(sb.stdout)),'stderr':''.join(list(sb.stderr))}))
finally: sb.terminate()
