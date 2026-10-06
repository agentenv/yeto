// Minimal DOM stub: runs the dashboard page script against an exported HTML
// file and exercises render, alert focus, island drill-down and metric tabs.
// Usage: node dashboard_smoke.js page.html  -> prints JSON with rendered HTML.
const fs = require("fs");
const html = fs.readFileSync(process.argv[2], "utf8");
const dataM = html.match(/<script type="application\/json" id="yeto-data">([\s\S]*?)<\/script>/);
const appM = html.match(/<script>\n([\s\S]*?)\n<\/script>/);
const els = {};
function mk(id) {
  return els[id] || (els[id] = { id, innerHTML: "", textContent: dataM && id === "yeto-data" ? dataM[1] : "",
    checked: false, dataset: {}, classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    scrollIntoView() {}, offsetTop: 0, scrollTop: 0, onclick: null, onchange: null });
}
let clickHandler = null;
global.document = {
  getElementById: (id) => (id === "yeto-data" && !dataM ? null : mk(id)),
  querySelectorAll: () => [],
  addEventListener: (t, fn) => { if (t === "click") clickHandler = fn; },
};
global.navigator = {};
global.setInterval = () => 0;
global.setTimeout = () => 0;
function target(ds) { return { dataset: ds, closest() { return this; }, classList: { contains() { return false; } } }; }
eval(appM[1]);
setImmediate(async () => {
  const out = { header: els.hdr.innerHTML, alerts: els.alerts.innerHTML, chart: els.big.innerHTML,
    cards: els.cards.innerHTML, eff: els.eff.innerHTML, rounds: els.rounds.innerHTML };
  if (els.alerts.innerHTML.includes('data-a="0"')) clickHandler({ target: target({ a: "0" }) });
  const card = els.cards.innerHTML.match(/data-k="([^"]+)"/);
  if (card) clickHandler({ target: target({ k: card[1] }) });
  clickHandler({ target: target({ m: "grad_norm" }) });
  await new Promise((r) => setImmediate(r));
  out.drill = els.drill.innerHTML; out.chart_grad = els.big.innerHTML;
  process.stdout.write(JSON.stringify(out));
});
