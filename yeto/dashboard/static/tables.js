// yeto dashboard: long tables and lists. Every table that grows with the run (one row per round,
// release, merge, event...) shows its newest rows only (default 10) with "+50" / "all" buttons;
// the choice per table is kept in localStorage. Sparklines above a cut table show its key
// columns over every row the page holds. Loaded first: sft.js and app.js call YetoTbl.
var YetoTbl=(function(){
"use strict";
var KEY="yeto-tables-v1",STEP=50,ST={},onChange=null;
try{ST=JSON.parse(localStorage.getItem(KEY)||"{}")||{};if(typeof ST!=="object"||Array.isArray(ST))ST={}}catch(e){ST={}}
function save(){try{localStorage.setItem(KEY,JSON.stringify(ST))}catch(e){}}
function esc(s){return String(s==null?"":s).replace(/[&<>"']/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]})}
function fin(v){return typeof v==="number"&&isFinite(v)}
/* rows wanted for a table: its default, or what the viewer opened (a number, or "all") */
function want(key,def){var v=ST[key];return v==="all"?Infinity:(fin(v)&&v>def?v:def)}
/* cap(key, rows, {def, total, tail, unit}): rows in display order. Without tail the newest row is
   first (rows[0]) and the first n are kept; with tail the newest row is last and the last n are kept.
   total: rows the run has when the server sent fewer (it keeps the newest). Returns {rows, bar}. */
function cap(key,rows,o){o=o||{};rows=rows||[];var def=o.def||10,n=rows.length,tot=fin(o.total)&&o.total>n?o.total:n,
    w=want(key,def),k=Math.min(n,w),shown=o.tail?rows.slice(n-k):rows.slice(0,k),unit=o.unit||"rows";
  return {rows:shown,n:k,total:tot,bar:bar(key,def,k,n,tot,w,unit)}}
function btn(key,def,act,label,title){return '<button class="btn" data-k="'+esc(key)+'" data-d="'+def+'" data-a="'+act+'" title="'+esc(title)+'" onclick="YetoTbl.click(this)">'+esc(label)+'</button>'}
function bar(key,def,k,n,tot,w,unit){
  if(!n)return "";
  var txt=k<tot?"latest "+k+" of "+tot+" "+unit:tot+" "+unit,b=[];
  if(tot>n)txt+=" (the page holds the newest "+n+")";
  if(k<n){b.push(btn(key,def,"more","+"+Math.min(STEP,n-k),"show "+STEP+" more rows"));b.push(btn(key,def,"all","all","show all "+n+" rows"))}
  if(w>def&&n>def)b.push(btn(key,def,"less","latest "+def,"back to the newest "+def+" rows"));
  return '<div class="tcap"><span>'+esc(txt)+'</span>'+b.join("")+'</div>'}
function click(b){var key=b.getAttribute("data-k"),def=+b.getAttribute("data-d")||10,a=b.getAttribute("data-a"),cur=want(key,def);
  if(a==="more")ST[key]=cur===Infinity?"all":cur+STEP;else if(a==="all")ST[key]="all";else delete ST[key];
  save();if(onChange)onChange()}
/* a cut table in its own horizontally scrolling box; td cells stay on one line (class cpt) */
function table(head,rowsHtml,cls){return '<div class="tw"><table class="c cpt'+(cls?" "+cls:"")+'">'+head+rowsHtml+'</table></div>'}
/* sparklines: [{label, vals (oldest first), fmt, title}] -> one compact row, each over every value */
function spark(items){
  var out=(items||[]).map(function(it){var v=(it.vals||[]).map(function(x){return fin(x)?x:null}),pts=[];v.forEach(function(x,i){if(x!=null)pts.push([i,x])});
    if(pts.length<2)return "";
    var lo=Infinity,hi=-Infinity;pts.forEach(function(p){lo=Math.min(lo,p[1]);hi=Math.max(hi,p[1])});
    var W=84,H=18,n=Math.max(1,v.length-1),X=function(i){return (i*W/n).toFixed(1)},Y=function(x){return (hi===lo?H/2:H-1-(x-lo)*(H-2)/(hi-lo)).toFixed(1)};
    var fmt=it.fmt||function(x){return String(+x.toPrecision(3))},last=pts[pts.length-1];
    var d=pts.map(function(p,i){return (i?"L":"M")+X(p[0])+","+Y(p[1])}).join("");
    return '<span class="si" title="'+esc((it.title||it.label)+": "+pts.length+" points, min "+fmt(lo)+", max "+fmt(hi)+", last "+fmt(last[1]))+'"><span>'+esc(it.label)+'</span>'+
      '<svg class="sp" viewBox="0 0 '+W+' '+H+'" preserveAspectRatio="none" aria-hidden="true"><path d="'+d+'" fill="none" stroke="var(--accent)" stroke-width="1.3" vector-effect="non-scaling-stroke"/>'+
      '<circle cx="'+X(last[0])+'" cy="'+Y(last[1])+'" r="1.8" fill="var(--accent)"/></svg><b class="num">'+esc(fmt(last[1]))+'</b></span>'}).filter(Boolean);
  return out.length?'<div class="spk">'+out.join("")+'</div>':""}
/* a table wider than its box scrolls inside it (.tw); while more columns sit to the right, the box's right
   edge fades (class "more") so a narrow screen shows the table goes on. A box that fits is left untouched */
function edge(w){var more=w.scrollWidth>w.clientWidth+1&&w.scrollLeft+w.clientWidth<w.scrollWidth-1;
  // the horizontal scrollbar's own height stays unmasked (0 for an overlay scrollbar: no unfaded strip)
  if(more&&w.style&&w.style.setProperty&&fin(w.offsetHeight))w.style.setProperty("--sb",Math.max(0,w.offsetHeight-w.clientHeight)+"px");
  if(w.classList&&w.classList.contains("more")!==more)w.classList.toggle("more",more)}
var hooked=false;
function hint(root){root=root||(typeof document!=="undefined"?document:null);if(!root||typeof root.querySelectorAll!=="function")return;
  Array.prototype.forEach.call(root.querySelectorAll(".tw"),edge);
  if(!hooked&&typeof document!=="undefined"&&document.addEventListener){hooked=true;
    document.addEventListener("scroll",function(e){var t=e.target;if(t&&t.classList&&t.classList.contains("tw"))edge(t)},true)}}
return {cap:cap,click:click,table:table,spark:spark,hint:hint,want:want,setRender:function(fn){onChange=fn},_state:function(){return ST}};
})();
