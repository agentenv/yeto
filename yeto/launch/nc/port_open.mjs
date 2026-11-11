// Open a workspace port through the National Compute controller and print
// {"base_url", "cookie", "expires_at"} as JSON on stdout (the caller stores it 0600).
//   node port_open.mjs --tool-dir ~/.yeta-tools/national --account 1 --port 8791
import path from "node:path";
import { pathToFileURL } from "node:url";

const argv = process.argv.slice(2);
const opt = (name, dflt) => { const i = argv.indexOf(`--${name}`); return i >= 0 ? argv[i + 1] : dflt; };
const toolDir = opt("tool-dir", process.env.NC_TOOL_DIR);
if (!toolDir) { console.error("--tool-dir or NC_TOOL_DIR is required"); process.exit(64); }
const port = Number(opt("port", 8791));
const core = await import(pathToFileURL(path.join(toolDir, "tool", "core.mjs")).href);
process.chdir(toolDir);
const account = await core.getAccount(Number(opt("account", 1)));
const controller = await core.openController(account, { allowLogin: false });
try {
  let opened = null;
  for (let attempt = 0; attempt < 10; attempt += 1) {
    opened = await controller.rpc.call("port.open", { port });
    if (opened.listening) break;
    await new Promise((r) => setTimeout(r, 500));
  }
  if (!opened?.listening) throw new Error(`port.open ${port}: not listening`);
  const redeemed = await fetch(opened.url, { redirect: "manual", signal: AbortSignal.timeout(15_000) });
  const location = redeemed.headers.get("location");
  const setCookie = redeemed.headers.get("set-cookie");
  if (!location || !setCookie) throw new Error(`ticket redeem failed: ${redeemed.status}`);
  const baseUrl = new URL(location, opened.url).toString().replace(/\?.*$/, "").replace(/\/$/, "");
  process.stdout.write(JSON.stringify({ base_url: baseUrl, cookie: setCookie.split(";", 1)[0], expires_at: opened.expiresAt ?? null, port }) + "\n");
} finally {
  controller.rpc.close();
}
