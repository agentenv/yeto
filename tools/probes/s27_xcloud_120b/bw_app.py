"""S27 G-ES: npu1 -> Modal throughput (CPU container, iperf3 server behind a Modal TCP tunnel).

The container starts ``iperf3 -s`` on port 5201, opens ``modal.forward(5201,
unencrypted=True)`` and prints ``[bw] tunnel HOST PORT REGION``; the client on
npu1 then runs ``iperf3 -c HOST -p PORT -P N -t T`` (npu1 sends) and ``-R``
(Modal sends). The function ends after ``seconds``.

    YETO_ES_APP=yeto-s27-es-bw1 modal run --detach tools/probes/s27_xcloud_120b/bw_app.py --seconds 900
"""
from __future__ import annotations

import os
import subprocess
import time

import modal

app = modal.App(os.environ.get("YETO_ES_APP", "yeto-s27-es-bw"))
image = modal.Image.debian_slim().apt_install("iperf3")


@app.function(image=image, cpu=8, memory=8192, timeout=3600, region=os.environ.get("YETO_ES_REGION") or None)
def sink(seconds: int = 900) -> None:
    server = subprocess.Popen(["iperf3", "-s", "-p", "5201"])
    with modal.forward(5201, unencrypted=True) as tunnel:
        host, port = tunnel.tcp_socket
        print(f"[bw] tunnel {host} {port} region={os.environ.get('MODAL_REGION', '?')} "
              f"cloud={os.environ.get('MODAL_CLOUD_PROVIDER', '?')}", flush=True)
        time.sleep(seconds)
    server.terminate()


@app.local_entrypoint()
def main(seconds: int = 900) -> None:
    call = sink.spawn(seconds)
    print(f"spawned sink {call.object_id}")
