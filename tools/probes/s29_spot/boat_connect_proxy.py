"""S29 D17: a minimal HTTP CONNECT proxy for the boat API only (boat.dev:443).

The trainer nodes (n4) have no direct internet; npu1 does. Run this on npu1 on its
intranet address (n4's sshd refuses remote forwards); the trainer then sets
``YETO_BOAT_PROXY=http://192.168.17.3:18781`` so its boat orphan
watchdog (``boat_labels.OrphanWatchdog``) can list and delete sandboxes.

    python3 boat_connect_proxy.py --host 192.168.17.3 [--port 18781]
"""
from __future__ import annotations

import argparse
import asyncio

ALLOWED = {("boat.dev", 443)}


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except (ConnectionError, OSError):
        pass
    finally:
        writer.close()


async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 30)
        method, target, _ = head.split(b"\r\n", 1)[0].decode().split(" ", 2)
        host, _, port = target.rpartition(":")
        if method != "CONNECT" or (host, int(port)) not in ALLOWED:
            writer.write(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            await writer.drain()
            writer.close()
            return
        up_r, up_w = await asyncio.wait_for(asyncio.open_connection(host, int(port)), 30)
        writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
        await writer.drain()
        await asyncio.gather(_pipe(reader, up_w), _pipe(up_r, writer))
    except (asyncio.IncompleteReadError, asyncio.TimeoutError, ValueError, OSError):
        writer.close()


async def main(host: str, port: int) -> None:
    server = await asyncio.start_server(_handle, host, port)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18781)
    a = p.parse_args()
    asyncio.run(main(a.host, a.port))
