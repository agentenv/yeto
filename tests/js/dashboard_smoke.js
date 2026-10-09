// Minimal DOM stub: runs the dashboard page script (section 9 redesign) against an
// exported HTML file: render, layout switch, metric tab, linked hover, keyboard.
// Usage: node dashboard_smoke.js page.html  -> prints JSON with rendered HTML.
const fs = require("fs");
const html = fs.readFileSync(process.argv[2], "utf8");
const dataM = html.match(/<script type="application\/json" id="yeto-data">([\s\S]*?)<\/script>/);
const appM = html.match(/<script>\n([\s\S]*?)\n<\/script>/);
const els = {};
function node(id) {
  const n = { id, innerHTML: "", textContent: "", hidden: false, style: {}, dataset: {}, children: [], attrs: {},
    listeners: {}, offsetWidth: 200, offsetHeight: 120,
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    setAttribute(k, v) { this.attrs[k] = String(v); }, removeAttribute(k) { delete this.attrs[k]; },
    appendChild(c) { this.children.push(c); return c; },
    addEventListener(t, fn) { (this.listeners[t] = this.listeners[t] || []).push(fn); },
    getBoundingClientRect() { return { left: 0, top: 0, width: 560, height: 400 }; },
    querySelectorAll() { return []; } };
  return n;
}
function mk(id) {
  if (!els[id]) { els[id] = node(id); if (id === "yeto-data" && dataM) els[id].textContent = dataM[1]; }
  return els[id];
}
const svgNodes = [];
global.document = {
  documentElement: node("html"),
  getElementById: (id) => (id === "yeto-data" && !dataM ? null : mk(id)),
  createElementNS: () => { const n = node(null); svgNodes.push(n); return n; },
  createElement: () => node(null),
  querySelectorAll: () => [],
  title: "",
};
global.navigator = {};
global.localStorage = { getItem() { return null; }, setItem() {} };
global.setInterval = () => 0;
global.setTimeout = () => 0;
eval(appM[1]);
const out = { title: els.title.innerHTML, kpis: els.kpis.innerHTML, islands: els.islands.innerHTML,
  cost: els.costBox.innerHTML, events: els.evs.innerHTML, folds: els.foldBox.innerHTML,
  metric: els.cMetric.innerHTML, svg_nodes: svgNodes.length, wall_hidden: els.islandWall.hidden };
// agentic-rollout-utilization 7: generation-stage utilization panel
out.util = { empty_hidden: mk("utilEmpty").hidden, body_hidden: mk("utilBody").hidden, ctl: mk("utilCtl").innerHTML,
  cut: mk("uCut").innerHTML, carry_hidden: mk("uCarryBox").hidden, carry: mk("uCarry").innerHTML,
  done_svg: svgNodes.filter((n) => n.attrs["class"] === "udone").length,
  cut_lines: svgNodes.filter((n) => n.attrs["class"] === "ucut").length };
// hover a timeline/duration segment with a mousemove handler -> tooltip
const hov = svgNodes.find((n) => n.listeners.mousemove && n.attrs["class"] === "tseg");
if (hov) { hov.listeners.mousemove[0]({ clientX: 10, clientY: 10 }); out.tip = els.tip.innerHTML; }
// keyboard: right arrow on the rounds panel
const kd = (els.rounds.listeners.keydown || [])[0];
if (kd) { kd({ key: "ArrowRight", preventDefault() {} }); out.tip_key = els.tip.innerHTML; }
els.layoutBtn.onclick();
out.title_after_switch = els.title.innerHTML; out.wall_hidden_after_switch = els.islandWall.hidden; out.wall = els.wall.innerHTML;
process.stdout.write(JSON.stringify(out));
