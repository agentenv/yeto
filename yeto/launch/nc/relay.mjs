// WebSocket -> TCP relay that runs inside a National Compute workspace (node, no deps).
// The workspace gateway (port.open + gardengate_port cookie) forwards HTTP upgrades to
// this port; each connection asks for ?target=HOST:PORT and is piped to that TCP address.
// Only targets listed (one HOST:PORT per line) in the allow file are dialed; the file is
// re-read per connection so the controller can add a node without restarting the relay.
//   RELAY_PORT=8791 RELAY_ALLOW=<relay-dir>/relay-allow node relay.mjs   (<relay-dir>: YETO_NC_RELAY_DIR)
import crypto from "node:crypto";
import fs from "node:fs";
import http from "node:http";
import net from "node:net";

const port = Number(process.env.RELAY_PORT ?? 8791);
const allowFile = process.env.RELAY_ALLOW ?? "/data/home/.yeto-nc/relay-allow";
const GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

function allowed(target) {
  try {
    return fs.readFileSync(allowFile, "utf8").split("\n").map((l) => l.trim()).includes(target);
  } catch { return false; }
}

function frame(opcode, payload) {
  const n = payload.length;
  const head = n < 126 ? Buffer.from([0x80 | opcode, n])
    : n < 65536 ? Buffer.from([0x80 | opcode, 126, n >> 8, n & 255])
    : (() => { const b = Buffer.alloc(10); b[0] = 0x80 | opcode; b[1] = 127; b.writeBigUInt64BE(BigInt(n), 2); return b; })();
  return Buffer.concat([head, payload]);
}

const server = http.createServer((req, res) => { res.writeHead(200); res.end("yeto-nc-relay\n"); });
server.on("upgrade", (req, socket) => {
  const target = new URL(req.url, "http://x").searchParams.get("target") ?? "";
  const m = /^([0-9A-Za-z.\-]+):(\d+)$/.exec(target);
  const key = req.headers["sec-websocket-key"];
  if (!m || !key || !allowed(target)) {
    socket.end("HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n");
    return;
  }
  const accept = crypto.createHash("sha1").update(key + GUID).digest("base64");
  const tcp = net.connect(Number(m[2]), m[1]);
  tcp.setNoDelay(true);
  let opened = false;
  tcp.on("connect", () => {
    opened = true;
    socket.write(`HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ${accept}\r\n\r\n`);
  });
  tcp.on("data", (d) => socket.write(frame(2, d)));
  const closeAll = () => { tcp.destroy(); socket.destroy(); };
  tcp.on("error", () => { if (!opened) socket.end("HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\n\r\n"); closeAll(); });
  tcp.on("close", () => { if (opened) socket.end(frame(8, Buffer.alloc(0))); });
  let buf = Buffer.alloc(0);
  socket.on("data", (chunk) => {
    buf = Buffer.concat([buf, chunk]);
    for (;;) {
      if (buf.length < 2) return;
      const op = buf[0] & 15; const masked = buf[1] & 128; let len = buf[1] & 127; let off = 2;
      if (len === 126) { if (buf.length < 4) return; len = buf.readUInt16BE(2); off = 4; }
      else if (len === 127) { if (buf.length < 10) return; len = Number(buf.readBigUInt64BE(2)); off = 10; }
      const maskOff = off; if (masked) off += 4;
      if (buf.length < off + len) return;
      const data = Buffer.from(buf.subarray(off, off + len));
      if (masked) for (let i = 0; i < len; i += 1) data[i] ^= buf[maskOff + (i & 3)];
      buf = buf.subarray(off + len);
      if (op === 8) { closeAll(); return; }
      if (op === 9) { socket.write(frame(10, data)); continue; }
      if (op === 0 || op === 1 || op === 2) tcp.write(data);
    }
  });
  socket.on("error", closeAll);
  socket.on("close", () => tcp.destroy());
});
server.listen(port, "127.0.0.1", () => console.log(`yeto-nc-relay listening 127.0.0.1:${port}`));
