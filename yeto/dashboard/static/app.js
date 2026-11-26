// yeto dashboard (section 9 redesign). No build step, no external requests.
// Live: polls /api/view every 5 s. Offline export: reads the inlined full view (its "page" key).
// Every number comes from the reducer; anything missing is shown as "n/a" (never interpolated).
(function(){
"use strict";
var node=document.getElementById("yeto-data"),INLINE=null;
if(node){try{INLINE=JSON.parse(node.textContent)}catch(e){INLINE=null}}
var OFFLINE=!!INLINE,POLL_MS=5000,NS="http:"+"//www.w3.org/2000/svg"; // SVG namespace id, not a request
var V=null,O=null,RS=[],SEL=null,cur=null,layout="auto",metric="reward",themeI=2,pinned=false,alertBound=false;
var PH=[["R","rollout"],["T","training"],["S","post-train sync"],["P","publish"]];
var METRICS=[["reward","reward"],["trunc","truncation"],["logprob_diff","logprob diff"],["kl","PPO KL"],["kl_tr","train-rollout KL"],["tis_clip","TIS clipfrac"],["grad_norm","grad_norm"],
  ["clip","clip"],["entropy","entropy"],["resp_len","response length"],["tok_s","tok/s"]];
function $(id){return document.getElementById(id)}
(function(){var nj=document.getElementById("nojs");if(nj)nj.style.display="none"})();
function esc(s){return String(s==null?"":s).replace(/[&<>"']/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]})}
function fin(v){return typeof v==="number"&&isFinite(v)}
// a trainer releasing every N rounds runs ahead of its last release: name both, never two bare numbers
function pvTxt(v,rel){if(!fin(v))return fin(rel)?"released v"+rel:"n/a";return fin(rel)&&rel!==v?"local v"+v+" · released v"+rel:"v"+v}
function f(v,d){if(!fin(v))return "n/a";if(d==null)d=3;return Math.abs(v)>=100?v.toFixed(0):v.toFixed(d)}
function pct(v){return fin(v)?(v*100).toFixed(1)+"%":"n/a"}
function hm(s){return !fin(s)?"n/a":(s>=3600?(s/3600).toFixed(1)+" h":(s/60).toFixed(0)+" min")}
function clock(t){if(!fin(t))return "n/a";var d=new Date(t*1000);function p(n){return (n<10?"0":"")+n}return p(d.getHours())+":"+p(d.getMinutes())+":"+p(d.getSeconds())}
function el(tag,a,parent){var e=document.createElementNS(NS,tag);for(var k in a)e.setAttribute(k,a[k]);
  /* a fixed viewBox stretched to a full-width panel scales its text 3-4x: cap the drawn width at 1.4x the viewBox */
  if(tag==="svg"&&a&&a.viewBox){var vw=+String(a.viewBox).split(" ")[2];if(vw>0)e.style.maxWidth=Math.round(vw*1.4)+"px"}
  if(parent)parent.appendChild(e);return e}
/* draw at the chart's pixel width (1 viewBox unit = 1 px, so 11px text stays 11px); def when not laid out.
   Measured on the parent: an emptied chart box is display:none (#cTl:empty ...) while it is redrawn */
function boxW(box,def){var p=box&&box.parentNode,w=p&&p.clientWidth;return w>=240?Math.floor(w):def}
// launch-preflight-guards 3.5: negative-test island badge (rl_island_override)
function negBadge(c){return c&&c.negative_test?' <span class="st wn" title="'+esc(c.negative_test.label)+'"><i></i>'+esc(c.negative_test.label)+'</span> ':""}
/* an island's colour mark and name as Islands now draws them (sft.js rlCards: PAL by tape order, "sampler 2" / "trainer 0") */
var PAL=["#3b82f6","#f97316","#14b8a6","#8b5cf6","#ec4899","#eab308","#22c55e","#06b6d4","#ef4444","#a3e635"];
function cardOf(id){var s=String(id);return (O.islands||[]).filter(function(c){return String(c.id)===s})[0]||null}
function islColor(id){var i=(O.islands||[]).map(function(c){return String(c.id)}).indexOf(String(id));return i<0?"var(--ink2)":PAL[i%PAL.length]}
function lbl(id){var c=cardOf(id),r=c&&c.role;return (r==="sampler"?"sampler ":r==="trainer"?"trainer ":"island ")+id}
function nameCell(id){var c=cardOf(id)||{};return '<span class="nn nc"><i class="sq" style="background:'+islColor(id)+'"></i><span title="'+esc(String(id)+(c.name?" ("+c.name+")":"")+(c.role?" · "+c.role:""))+'">'+esc(lbl(id))+'</span></span>'}
/* the island switch above the details: a fleet's islands + "Fleet", an async run's trainers, a synchronous run's
   islands; none for a run with one trainer / one island. PICK survives every redraw (poll, resize) and a reload */
var FLEET="fleet",PICK=null,FOLDS=[];
function fleetOn(){return !!(V.async&&V.async.islands&&V.async.islands.length)}
function fleetView(){return fleetOn()&&PICK===FLEET}
/* a fleet or a run with a syncer (trainer islands): a value its tapes do not carry reads as a dash with the reason
   in its title, a KPI without data as a sentence; a run with one trainer (no syncer, no fleet) keeps "n/a" */
function multiOn(){var d=V&&V.topology&&V.topology.diloco;return fleetOn()||!!(d&&(d.kind==="island"||d.kind==="synced_group"))}
function naDash(h,why){return multiOn()?String(h).replace(/(<[^>]*>)|\bn\/a\b/g,function(m,tag){return tag||'<span class="note" title="'+esc(why)+'">—</span>'}):h}
function pickItems(){var A=V.async,cards=O.islands||[],one=function(c){return {key:String(c.id),label:lbl(c.id),title:c.name||"",color:islColor(c.id),tid:String(c.id)}};
  if(fleetOn())return A.islands.map(function(x){var t=x.trainer!=null?String(x.trainer):null;
    return {key:"i"+x.island,label:"Island "+x.island,title:[x.name,t!=null?"trainer "+t:"trainer not on the tapes yet"].filter(Boolean).join(" · "),color:t!=null&&cardOf(t)?islColor(t):"var(--ink2)",tid:t}}).concat(
    [{key:FLEET,label:"Fleet",title:"every island side by side: trainer metrics, syncer merges, task ledger, pool events"}]);
  var trs=cards.filter(function(c){return c.role==="trainer"});
  if(A&&trs.length>1)return trs.map(one);
  if(!A&&cards.length>1)return cards.map(one);
  return []}
function pickStore(){return "yeto-pick-v1:"+(O.run||"")}
function resolvePick(){var it=pickItems(),keys=it.map(function(x){return x.key}),ids=(O.islands||[]).map(function(c){return String(c.id)});
  if(keys.indexOf(PICK)<0){var s=null;try{s=localStorage.getItem(pickStore())}catch(e){}PICK=keys.indexOf(s)>=0?s:(keys[0]||null)}
  var cur=it.filter(function(x){return x.key===PICK})[0];
  if(cur&&cur.tid!==undefined)SEL=cur.tid!=null&&ids.indexOf(cur.tid)>=0?cur.tid:null;
  else if(PICK===FLEET){var best=null,bn=-1;(V.async.islands||[]).forEach(function(x){var t=x.trainer!=null?String(x.trainer):null,n=t!=null&&ids.indexOf(t)>=0?((V.islands[t]||{}).rounds||[]).length:-1;if(n>bn){bn=n;best=t}});
    SEL=best!=null&&bn>=0?best:(ids[0]||null)} // the trainer with the most rounds anchors the hover tips
  else{var tr=(O.islands||[]).filter(function(c){return c.role==="trainer"})[0];SEL=ids.indexOf(String(SEL))>=0?String(SEL):(V.async&&tr?String(tr.id):(ids[0]||null))}
  return it}
function picker(){
  var it=resolvePick(),sw=$("islSw"),fl=fleetOn();sw.hidden=it.length<2;
  sw.innerHTML=it.map(function(x){return '<button type="button" role="tab" aria-selected="'+(x.key===PICK)+'"'+(x.key===PICK?' class="on"':"")+' data-k="'+esc(x.key)+'" title="'+esc(x.title||"")+'">'+
    (x.color?'<i class="sq" style="background:'+x.color+'"></i>':"")+esc(x.label)+'</button>'}).join("");
  Array.prototype.forEach.call(sw.querySelectorAll("button"),function(b){b.onclick=function(){PICK=b.dataset.k;try{localStorage.setItem(pickStore(),PICK)}catch(e){}cur=null;pinned=false;$("tip").style.display="none";render()}});
  $("detailsCs").textContent=fl?(fleetView()?"every island of the fleet: per-island summary, trainer metrics side by side, syncer merges, task ledger, pool events":
    "the picked island: trainer rounds, its releases, samplers and pools, backlog, release timeline, per-round drill-down; Fleet = run-wide ledgers"):
    V.async?"trainer rounds, releases, samplers, backlog, release timeline, per-round drill-down, events":it.length?"per-round charts of the picked island, island health, events":"per-round charts, island health, events";
}
function kind(){return layout==="auto"?O.run_kind:layout}
/* why an island left: "left: evicted (preempted) at 05:09, replaced by <ticket>" */
function leftText(s,l){return (s==="lost"?"lost":"left")+": "+(l.reason||"?")+(fin(l.ts)?" at "+clock(l.ts).slice(0,5):"")+(l.replaced_by?", replaced by "+l.replaced_by:"")}
function stBadge(s,c){var m={ok:["ok","healthy"],starting:["wn","starting"],stopped:["mu","stopped"],stale:["bd","stale"],waiting:["wn","waiting"],lost:["bd","lost"],
  left:["mu","left the run"],recovery:["bd","needs recovery"],done:["mu","finished"],unknown:["mu","no data"]}[s]||["mu",s];var t=m[1];
  if(c){if(s==="starting"&&c.startup_slow)t="starting (slow)";if(s==="waiting"&&c.waiting_reason)t="waiting: "+c.waiting_reason;if((s==="left"||s==="lost")&&c.left)t=leftText(s,c.left);
    // its pool removed / retired it but it still runs: draining, or a release that failed (still billed)
    if(c.pool_removed&&!c.left){var pr=c.pool_removed;if(pr.released){m=["bd"];t="active after release"}else if(s==="ok"||s==="waiting"||s==="starting"){m=["wn"];t="draining ("+(pr.reason||"retired")+")"}else t+=" · "+(pr.reason||"retired")+", not released"}}
  return '<span class="st '+m[0]+'" title="'+esc(c&&c.left&&c.left.detail&&(s==="left"||s==="lost")?c.left.detail:c&&c.pool_removed&&!c.left?c.pool_removed.detail||"":c&&c.role?c.role:"")+'"><i></i>'+esc(t)+'</span>'}

/* theme + layout */
var THEMES=["auto","light","dark"];
try{var saved=localStorage.getItem("yeto-theme-v2");if(saved)themeI=Math.max(0,THEMES.indexOf(saved))}catch(e){}
function applyTheme(){var t=THEMES[themeI];if(t==="auto")document.documentElement.removeAttribute("data-theme");else document.documentElement.setAttribute("data-theme",t);
  $("themeBtn").textContent="Theme: "+{auto:"system",light:"light",dark:"dark"}[t];try{localStorage.setItem("yeto-theme-v2",t)}catch(e){}}
$("themeBtn").onclick=function(){themeI=(themeI+1)%3;applyTheme()};applyTheme();
$("layoutBtn").onclick=function(){layout=layout==="auto"?(O.run_kind==="single_island"?"multi_island":"single_island"):"auto";render()};

function header(){
  var k=kind();
  $("title").innerHTML=esc(k==="single_island"?"Single-island overview":"Multi-island overview")+' · <span class="mono">'+esc(O.run||"unnamed run")+'</span>'+
    (O.run_inferred?'<span class="tag">name inferred from tapes</span>':"")+'<span class="tag">'+esc(O.run_kind)+'</span>';
  document.title="yeto dashboard · "+(O.run||"unnamed run");
  var c=(O.islands||[])[0]||{};
  $("sub").textContent=(OFFLINE?"offline export":"live")+" · data as of "+clock(O.data_ts)+(c.cloud?" · "+c.cloud+" "+(c.gpu||"")+" × "+(c.gpus==null?"n/a":c.gpus):"");
  $("layoutBtn").textContent="Layout: "+(layout==="auto"?"auto ("+(O.run_kind==="single_island"?"single island":"multi-island")+")":(layout==="single_island"?"single island (temporary)":"multi-island (temporary)"));
  $("navSub").textContent=(O.global_status||{}).text||"";
}
function kpis(){
  var gs=O.global_status||{},c=O.cost||{},ru=c.run_up||null,isl=(O.islands||[]).find(function(x){return x.id===SEL})||{},last=RS[RS.length-1]||{};
  var rate=(c.islands||[]).reduce(function(a,r){return a+(fin(r.rate_usd_h)?r.rate_usd_h:0)},0);
  var dur=fin(O.data_ts)&&fin(O.first_ts)?O.data_ts-O.first_ts:null;
  var pubs=RS.map(function(r){return r.published_version}).filter(fin);
  var lvl={ok:"ok",warn:"wn",bad:"bd",muted:"mu"}[gs.level]||"mu";
  var cards=[["Status",'<span class="st '+lvl+'"><i></i>'+esc(gs.text||"n/a")+'</span>',(O.alerts||[]).length+" alerts",true],
    ["Estimated cost",fin(c.total_usd)?"$"+f(c.total_usd,1):"n/a",(rate?"$"+f(rate,2):"n/a")+"/h ("+((c.gpu_only||[]).length?"some GPU only":"GPU+CPU+memory")+")"+(fin(c.budget_pct)?" · budget "+f(c.budget_pct,0)+"%":"")],
    ["Rounds",String(RS.length),pubs.length?"published up to v"+Math.max.apply(null,pubs):"nothing published yet"],
    ["Duration",hm(dur),isl.starting?"starting for "+hm(isl.startup_s):"first to last event"],
    ["Run uptime",ru&&fin(ru.cumulative_runtime_h)?hm(ru.cumulative_runtime_h*3600):"n/a",ru?(ru.ups+" up(s)"+(fin(ru.current_up_runtime_h)?" · this up "+hm(ru.current_up_runtime_h*3600):"")+(fin(ru.spend_usd_estimate)?" · spend $"+f(ru.spend_usd_estimate,1):"")):"no run up state"],
    ["Throughput",f(last.tok_s,0),"tok/s, last round"]];
  $("kpis").innerHTML=cards.map(function(x){return '<div class="kpi'+(x[3]?" main":"")+'"><div class="k">'+x[0]+'</div><div class="v">'+x[1]+'</div><div class="s">'+esc(x[2])+'</div></div>'}).join("");
  // breaker trips and loud dependency checks stay up here, each linked to the Breakers panel at the bottom
  var al=O.alerts||[],brk=typeof YetoBrk!=="undefined"&&V&&V.breakers,extra=brk?YetoBrk.alertRows(V):"";$("alertBox").hidden=!al.length&&!extra;
  $("alertBox").innerHTML="<h2>Alerts</h2>"+al.map(function(a){return '<div><span class="st '+(a.sev===0?"bd":a.sev===1?"wn":"mu")+'"><i></i>'+["critical","warning","info"][a.sev]+'</span> <b>'+esc(a.title)+'</b> <span class="note">'+esc(a.detail)+'</span>'+(brk&&(a.rule==="breaker"||a.rule==="guard_drops")?" "+YetoBrk.link():"")+'</div>'}).join("")+extra;
  if(brk&&!alertBound){alertBound=true;$("alertBox").addEventListener("click",function(e){var t=e.target;if(t&&t.getAttribute&&t.getAttribute("data-brk"))YetoBrk.open()})}
}
/* a sampler without a fleet record: its platform from rl_sampler_start provenance (engine on device
   family; the cloud is not on its tape) and its reward from its last round with groups */
function spOf(c){return c&&c.role==="sampler"?(((V.async||{}).samplers||{})[c.id]||null):null}
function platform(c){if(c.cloud)return c.cloud;var p=(spOf(c)||{}).provenance||{};
  return p.engine||p.device_family?[p.engine,p.device_family].filter(Boolean).join(" on ")+" (cloud not on the tape)":"n/a"}
function islReward(c){if(c.reward!=null)return c.reward;var sp=spOf(c);return sp?sp.reward_mean_last:null}
/* a sampler writes no rl_heartbeat phase: its latest event says what it is doing */
function islPhase(c){if(c.phase)return c.phase;var sp=spOf(c);if(!sp)return null;if(sp.ended||/^(done|stopped|left|lost|finished)$/.test(String(c.status||"")))return "stopped";var lt=String(sp.last_type||"");
  if(/drain/.test(lt))return "draining";if(/policy_load|load_checked|start_on_latest/.test(lt))return "loading policy";
  if(/backlog_full|_waiting$/.test(lt))return "waiting (backlog full)";if(/_start$|_join$|_lease$/.test(lt))return "starting";return "sampling"}
/* syncer merges (a run with a syncer): a fleet shows them under "Fleet" */
function syncerPanel(show){
  var on=show&&kind()==="multi_island"&&!!V.usage.syncer;$("syncerBox").hidden=!on;if(!on)return;
  var all=V.rounds_syncer||[],sc=YetoTbl.cap("syncer-merges",all.slice().reverse(),{def:10,total:(V.table_totals||{}).rounds_syncer,unit:"merges"});
  $("syncer").innerHTML=YetoTbl.spark([["quorum ms","quorum_ms"],["merge ms","merge_ms"],["responded","responded"]].map(function(x){return {label:x[0],vals:all.map(function(r){return r[x[1]]}),fmt:function(v){return f(v,0)}}}))+sc.bar+
    YetoTbl.table('<tr><th>global round</th><th>expected / responded</th><th>missed</th><th>quorum ms</th><th>merge ms</th><th>resends</th></tr>',sc.rows.map(function(r){
    return '<tr><td class="num">'+esc(r.round)+'</td><td class="num">'+f(r.expected,0)+' / '+f(r.responded,0)+'</td><td>'+(r.missed.length?'<span class="bd">'+esc(r.missed.map(function(m){return cardOf(m)?lbl(m):m}).join(", "))+'</span>':'<span class="note">none</span>')+'</td><td class="num">'+f(r.quorum_ms,0)+'</td><td class="num">'+f(r.merge_ms,0)+'</td><td class="num">'+esc(r.resend)+'</td></tr>'}).join(""))
}
/* chart 1: metric per round; reward gets the p10-p90 band; < 5 points: dots only, no line */
var xOf=null;
var metricWant="reward";
/* the series the metric chart draws: the picked trainer; under "Fleet" every trainer of the fleet; a synchronous
   multi-island run every island (they train the same rounds). Never a sampler: it has no trained rounds */
function metricIds(){var A=V.async;
  if(fleetView())return (A.islands||[]).map(function(x){return x.trainer!=null?String(x.trainer):null}).filter(function(t){return t!=null&&cardOf(t)});
  if(!A&&kind()==="multi_island")return (O.islands||[]).map(function(c){return c.id});
  return SEL!=null?[SEL]:[]}
function hasMetric(k,ids){return ids.some(function(id){return k==="trunc"||k==="logprob_diff"?((V.islands[id]||{}).rounds||[]).some(function(r){return fin(r[k])}):(((O.series||{})[id]||{})[k]||[]).length})}
function chartMetric(){
  var ids=metricIds(),multi=ids.length>1,avail=METRICS.filter(function(m){return hasMetric(m[0],ids)});
  var hasK=function(k){return avail.some(function(m){return m[0]===k})}; // the viewer's metric when this trainer has it, else the first it has
  metric=hasK(metricWant)?metricWant:avail.length?avail[0][0]:metricWant;
  // only the metrics this trainer reports: a tab per metric of the algorithm list was mostly empty charts
  $("tabs").innerHTML=avail.map(function(m){return '<button class="btn'+(m[0]===metric?" on":"")+'" data-m="'+m[0]+'">'+m[1]+'</button>'}).join("");
  Array.prototype.forEach.call($("tabs").querySelectorAll("button"),function(b){b.onclick=function(){metric=metricWant=b.dataset.m;chartMetric();highlight()}});
  var series=ids.map(function(id){return {id:id,pts:((O.series||{})[id]||{})[metric==="trunc"||metric==="logprob_diff"?"_":metric]||[]}});
  if(metric==="trunc"||metric==="logprob_diff"){series=ids.map(function(id){var rr=(V.islands[id]||{}).rounds||[];return {id:id,pts:rr.filter(function(r){return fin(r[metric])}).map(function(r){return [r.round+1,r[metric]]})}})}
  var all=[];series.forEach(function(s){s.pts.forEach(function(p){all.push(p)})});
  var band=metric==="reward"&&!multi?RS.filter(function(r){return fin(r.p10)&&fin(r.p90)}):[];
  var box=$("cMetric");box.innerHTML="";var W=boxW(box,560),H=250,m={l:52,r:16,t:12,b:26};
  var mname=(METRICS.filter(function(x){return x[0]===metric})[0]||[0,metric])[1];
  $("legMetric").innerHTML=(multi?ids.map(function(id){return '<span><i class="sw" style="background:'+islColor(id)+'"></i>'+esc(lbl(id))+'</span>'}).join("")+'<span>'+esc(mname)+'</span>':
    '<span><i class="sw" style="background:var(--T)"></i>'+esc(mname)+(ids.length?" · "+esc(lbl(ids[0])):"")+'</span>')+
    (band.length?'<span><i class="sw" style="background:var(--band)"></i>p10–p90</span>':"");
  if(!all.length){box.innerHTML='<div class="nodata">'+(ids.length?(avail.length?"no data":"no per-round metric on the tapes of "+esc(ids.map(lbl).join(", "))):"no trainer on the tapes yet")+'</div>';xOf=null;return}
  var xs=all.map(function(p){return p[0]}),x0=Math.min.apply(null,xs),x1=Math.max.apply(null,xs);
  var vs=all.map(function(p){return p[1]});band.forEach(function(r){vs.push(r.p10,r.p90)});
  var lo=Math.min.apply(null,vs),hi=Math.max.apply(null,vs);if(metric==="reward"||metric==="trunc"){lo=Math.min(lo,0);hi=Math.max(hi,1)}if(hi===lo){hi+=1;lo-=1}
  var x=function(r){return m.l+(x1===x0?(W-m.l-m.r)/2:(r-x0)*(W-m.l-m.r)/(x1-x0))},y=function(v){return m.t+(hi-v)/(hi-lo)*(H-m.t-m.b)};
  xOf=function(round){return x(round+1)};
  var s=el("svg",{viewBox:"0 0 "+W+" "+H,role:"img","aria-label":"Per-round metrics"},box),g=el("g",{"class":"grid"},s);
  for(var j=0;j<=4;j++){var v=lo+(hi-lo)*j/4;el("line",{x1:m.l,x2:W-m.r,y1:y(v),y2:y(v)},g);el("text",{x:m.l-6,y:y(v)+4,"text-anchor":"end"},s).textContent=+v.toPrecision(3)}
  // round numbers on the axis, as many as fit at ~7.3 px per digit (the caption "round" sits under the value axis)
  var ticks={};xs.forEach(function(r){ticks[r]=1});var tk=Object.keys(ticks).map(Number).sort(function(a,b){return a-b});
  var lw=7.3*String(Math.max.apply(null,tk)-1).length+16,fit=Math.max(2,Math.floor((W-m.l-m.r)/lw)),every=Math.max(1,Math.ceil(tk.length/fit));
  el("text",{x:m.l-6,y:H-8,"text-anchor":"end"},s).textContent="round";
  tk.forEach(function(r,i){if(i%every===0)el("text",{x:x(r),y:H-8,"text-anchor":"middle"},s).textContent=String(r-1)});
  var connect=series.some(function(sr){return sr.pts.length>=5});
  if(band.length>=2&&connect)el("path",{d:"M"+band.map(function(r){return x(r.round+1)+","+y(r.p90)}).join("L")+"L"+band.slice().reverse().map(function(r){return x(r.round+1)+","+y(r.p10)}).join("L")+"Z",fill:"var(--band)"},s);
  else band.forEach(function(r){el("line",{x1:x(r.round+1),x2:x(r.round+1),y1:y(r.p10),y2:y(r.p90),stroke:"var(--band)","stroke-width":10,"stroke-linecap":"round"},s)});
  series.forEach(function(sr){var col=multi?islColor(sr.id):"var(--T)";
    if(sr.pts.length>=5)el("path",{d:"M"+sr.pts.map(function(p){return x(p[0])+","+y(p[1])}).join("L"),fill:"none",stroke:col,"stroke-width":2},s);
    sr.pts.forEach(function(p){el("circle",{cx:x(p[0]),cy:y(p[1]),r:sr.pts.length>60?2.5:4,fill:col,stroke:"var(--panel)","stroke-width":2,"class":"rdot","data-r":p[0]-1},s)})});
  if(!connect)el("text",{x:W-m.r,y:m.t+10,"text-anchor":"end"},s).textContent="fewer than 5 points: dots only, no lines";
  el("line",{id:"vline",y1:m.t,y2:H-m.b,stroke:"var(--ink2)","stroke-dasharray":"3 3",visibility:"hidden"},s);
  var hit=el("rect",{x:m.l-10,y:0,width:W-m.l-m.r+20,height:H,fill:"transparent"},s);
  function pick(e){var b=s.getBoundingClientRect(),vx=(e.clientX-b.left)*W/b.width,best=null,bd=1e9;
    tk.forEach(function(r){var d=Math.abs(x(r)-vx);if(d<bd){bd=d;best=r-1}});setCur(best,e)}
  hit.addEventListener("mousemove",pick);hit.addEventListener("click",function(e){pinned=!pinned;pick(e)});hit.addEventListener("mouseleave",hideTip);
}
/* chart 2: per-round time split (stacked R/T/S/P) */
function chartDur(){
  $("legDur").innerHTML=PH.map(function(p){return '<span><i class="sw" style="background:var(--'+p[0]+')"></i>'+p[1]+'</span>'}).join("");
  var box=$("cDur");box.innerHTML="";
  if(!RS.some(function(r){return Object.keys(r.dur).length})){box.innerHTML='<div class="note">no phase data (no rl_timeline_span in the tapes)</div>';return}
  var dc=YetoTbl.cap("round-time",RS,{def:10,tail:true,unit:"rounds"}),rows=dc.rows;box.innerHTML=dc.bar;
  var W=boxW(box,420),rowH=24,m={l:64,r:60,t:2,b:2},H=m.t+m.b+rowH*rows.length;
  var s=el("svg",{viewBox:"0 0 "+W+" "+H,role:"img","aria-label":"Per-round time breakdown"},box);
  var tot=rows.map(function(r){return PH.reduce(function(a,p){return a+(r.dur[p[0]]||0)},0)}),mx=Math.max.apply(null,tot.concat([1]));
  rows.forEach(function(r,i){var y0=m.t+i*rowH,x0=m.l;
    el("rect",{x:0,y:y0,width:W,height:rowH,fill:"transparent","class":"drow","data-r":r.round},s);
    el("text",{x:m.l-8,y:y0+rowH/2+4,"text-anchor":"end"},s).textContent="round "+r.round;
    PH.forEach(function(p){var w=(r.dur[p[0]]||0)*(W-m.l-m.r)/mx;if(w>0){el("rect",{x:x0,y:y0+4,width:Math.max(w-2,1),height:rowH-8,rx:3,fill:"var(--"+p[0]+")"},s);x0+=w}});
    el("text",{x:x0+6,y:y0+rowH/2+4},s).textContent=(tot[i]/60).toFixed(1)+"m";
    var hit=el("rect",{x:0,y:y0,width:W,height:rowH,fill:"transparent"},s);
    hit.addEventListener("mousemove",function(e){setCur(r.round,e)});hit.addEventListener("mouseleave",hideTip);hit.addEventListener("click",function(e){pinned=!pinned;setCur(r.round,e)})});
}
/* chart 3: phase timeline on the wall clock from the first round's first phase; the startup before it is a note,
   not a grey block (it was most of the axis on a run whose rounds took minutes after a long boot) */
function durTxt(s){return !fin(s)?"n/a":s<90?f(s,0)+" s":s<5400?(s/60).toFixed(1)+" min":(s/3600).toFixed(1)+" h"}
function chartTl(){
  var ex=V.islands[SEL]||{},box=$("cTl");box.innerHTML="";
  var su=fin(ex.first_ts)&&fin(ex.driver_start_ts)&&ex.driver_start_ts>ex.first_ts?ex.driver_start_ts-ex.first_ts:null;
  $("legTl").innerHTML=PH.map(function(p){return '<span><i class="sw" style="background:var(--'+p[0]+')"></i>'+p[1]+'</span>'}).join("")+
    (su!=null?'<span title="first event of the island to its driver start: load weights, form the Ray cluster, start the inference engine">startup before the first round: '+durTxt(su)+'</span>':"");
  var segs=[];RS.forEach(function(r){Object.keys(r.phases).forEach(function(k){segs.push([k,r.phases[k][0],r.phases[k][1],r.round])})});
  if(!segs.length){box.innerHTML='<div class="nodata">no phase data (no rl_timeline_span / rl_driver_phase in the tapes)</div>';return}
  var t0=Math.min.apply(null,segs.map(function(q){return q[1]})),t1=Math.max.apply(null,segs.map(function(q){return q[2]}));
  var W=boxW(box,1000),LH=26,m={l:70,r:16,t:4,b:22},H=m.t+2*LH+m.b,sx=function(t){return m.l+(t-t0)*(W-m.l-m.r)/((t1-t0)||1)};
  var s=el("svg",{viewBox:"0 0 "+W+" "+H,role:"img","aria-label":"Phase timeline"},box);
  el("text",{x:m.l-8,y:m.t+LH/2+4,"text-anchor":"end"},s).textContent="rollout";el("text",{x:m.l-8,y:m.t+LH*1.5+4,"text-anchor":"end"},s).textContent="training";
  segs.forEach(function(q){var rc=el("rect",{x:sx(q[1]),y:q[0]==="R"?m.t:m.t+LH,width:Math.max(sx(q[2])-sx(q[1])-1,1),height:LH-6,rx:3,fill:"var(--"+q[0]+")","class":"tseg","data-r":q[3]},s);
    rc.addEventListener("mousemove",function(e){setCur(q[3],e)});rc.addEventListener("mouseleave",hideTip);rc.addEventListener("click",function(e){pinned=!pinned;setCur(q[3],e)})});
  var nT=Math.max(2,Math.min(8,Math.floor((W-m.l-m.r)/110)));for(var j=0;j<=nT;j++){var t=t0+(t1-t0)*j/nT;el("text",{x:sx(t),y:H-6,"text-anchor":j===0?"start":j===nT?"end":"middle"},s).textContent=clock(t)}
}
/* linked hover state */
function setCur(r,e){cur=r;highlight();if(e)showTip(e)}
function highlight(){
  Array.prototype.forEach.call(document.querySelectorAll(".rdot"),function(d){d.setAttribute("stroke-width",+d.dataset.r===cur?4:2)});
  var v=$("vline");if(v&&cur!=null&&xOf){v.setAttribute("x1",xOf(cur));v.setAttribute("x2",xOf(cur));v.setAttribute("visibility","visible")}
  Array.prototype.forEach.call(document.querySelectorAll(".drow"),function(d){d.setAttribute("fill",+d.dataset.r===cur?"var(--band)":"transparent")});
  Array.prototype.forEach.call(document.querySelectorAll(".tseg"),function(d){d.setAttribute("opacity",cur==null||+d.dataset.r===cur?1:.35)});
}
function row(k,v){return '<div class="row"><span>'+k+'</span><span>'+v+'</span></div>'}
function tipHtml(r){
  var h='<div class="h">Round '+r.round+' · trained on policy v'+r.policy_version+' → published '+(r.published_version==null?"n/a":"v"+r.published_version)+'</div>';
  var ids=metricIds(),asy=!!V.async;
  if(ids.length>1){return '<div class="h">Round '+r.round+'</div>'+ids.map(function(id){var sw='<i class="sw" style="background:'+islColor(id)+'"></i>';
    var rr=((V.islands[id]||{}).rounds||[]).filter(function(x){return x.round===r.round})[0];
    if(!rr&&asy)return row(sw+esc(lbl(id)),'<span class="note">no round '+r.round+'</span>');
    return row(sw+esc(lbl(id)),rr?f(rr.reward)+" · truncated "+pct(rr.trunc):"missed")}).join("")}
  // async buffer mode: what the trainer took from each sampler this round (samplers keep their own round numbers)
  var bs=asy?Object.keys(r.by_sampler||{}).map(function(k){var b=r.by_sampler[k],vs=b.versions||[];
    return row('<i class="sw" style="background:'+islColor(k)+'"></i>from '+esc(lbl(k)),(b.groups!=null?f(b.groups,0)+" groups":f(b.samples,0)+" samples")+(vs.length?" · v"+vs[0]+(vs.length>1?"–v"+vs[vs.length-1]:""):"")+
      (fin(b.age_mean)?" (age "+f(b.age_mean,1)+")":"")+(fin(b.reward_mean)?" · reward "+f(b.reward_mean,3):""))}).join(""):"";
  return h+row("reward mean",f(r.reward))+row("train-rollout KL / TIS clipfrac",f(r.kl_tr,5)+" / "+f(r.tis_clip,4))+row("p10 / p50 / p90",f(r.p10,2)+" / "+f(r.p50,2)+" / "+f(r.p90,2))+row("truncation",pct(r.trunc))+
    row("logprob diff",f(r.logprob_diff,4))+row("response length mean / p95",f(r.resp_mean,0)+" / "+f(r.resp_p95,0))+row("grad_norm",f(r.grad_norm,4))+
    PH.map(function(p){return row('<i class="sw" style="background:var(--'+p[0]+')"></i>'+p[1],fin(r.dur[p[0]])?f(r.dur[p[0]],0)+" s":"n/a")}).join("")+bs}
/* the round under the cursor: the anchor island's, else the first drawn series that has it (a fleet's trainers
   count their own rounds) */
function roundOf(rd){var r=RS.filter(function(x){return x.round===rd})[0];if(r)return r;var ids=metricIds();
  for(var i=0;i<ids.length;i++){r=((V.islands[ids[i]]||{}).rounds||[]).filter(function(x){return x.round===rd})[0];if(r)return r}return null}
function showTip(e){var r=roundOf(cur);if(!r)return;var t=$("tip"),p=$("rounds").getBoundingClientRect();
  t.innerHTML=tipHtml(r);t.style.display="block";var x=e.clientX-p.left+14,y=e.clientY-p.top+14,w=t.offsetWidth,h=t.offsetHeight;
  if(x+w>p.width-8)x=e.clientX-p.left-w-14;if(y+h>p.height-8)y=Math.max(8,e.clientY-p.top-h-14);t.style.left=Math.max(8,x)+"px";t.style.top=y+"px"}
function hideTip(){if(!pinned)$("tip").style.display="none"}
$("rounds").addEventListener("keydown",function(e){if(!RS.length)return;var ids=RS.map(function(r){return r.round}),i=ids.indexOf(cur);
  if(e.key==="ArrowRight")i=Math.min(ids.length-1,i+1);else if(e.key==="ArrowLeft")i=Math.max(0,i<0?ids.length-1:i-1);else if(e.key==="Escape"){pinned=false;cur=null;$("tip").style.display="none";highlight();return}else return;
  e.preventDefault();cur=ids[i];highlight();var t=$("tip");t.innerHTML=tipHtml(RS[i]);t.style.display="block";t.style.left="16px";t.style.top="48px"});
/* generation-stage utilization. Missing values -> "unknown"; no data -> hidden / "no data". */
var uRound="all",uMode="side";
var UPH=[["gen","generation","var(--R)"],["tool","tool execution","var(--i2)"],["judge","judging","var(--T)"],["sandbox","sandbox startup","var(--S)"]];
var RUN_COLORS=["var(--i0)","var(--i1)","var(--i2)","var(--i3)"];
function unk(v,d){return fin(v)?f(v,d==null?0:d):"unknown"}
function utilRuns(){
  var runs=[{label:V.label||(V.compare?"A":""),run:O.run,util:(V.islands[SEL]||{}).util}];
  (V.compare||[]).forEach(function(c){var u=c.util||{},k=u[SEL]?SEL:Object.keys(u)[0];runs.push({label:c.label,run:c.run,util:k!=null?u[k]:null})});
  return runs}
function runHead(r,i,n){return n>1?'<div class="rl"><i class="sw" style="background:'+RUN_COLORS[i]+'"></i>'+esc(r.label)+' · <span class="mono">'+esc(r.run||"unnamed run")+'</span></div>':""}
/* which utilization charts have data: a chart without any is not drawn, it is named under "Panels not used in this run" */
var USEC=[["uCutBox","Cutoff drop / carry-over","no rl_rollout_cutoff on the tapes",function(u){return (u.rounds||[]).some(function(x){return x.cutoff})}],
  ["uDoneBox","Trajectory completion curve","no trajectory start / end times (rl_trajectory_reward timing fields)",function(u){return (u.rounds||[]).some(function(x){return (x.done||[]).length})}],
  ["uPhBox","Four-part timing","trajectory events lack the generation / tool / judge / sandbox timing fields",function(u){return (u.rounds||[]).some(function(x){return UPH.some(function(p){return fin((x.phases||{})[p[0]])})})}],
  ["uTrBox","Trajectories of a round","no per-trajectory timing on the tapes",function(u){return (u.trajectories||[]).some(function(t){return UPH.some(function(p){return fin(t[p[0]])})})}],
  ["uLoadBox","KV usage and queueing","no rl_load_sample (engine load probe)",function(u){return (u.load||[]).some(function(p){return fin(p[0])})}]];
function util(show){
  var runs=utilRuns(),secs=USEC.map(function(c){return runs.some(function(r){return r.util&&c[3](r.util)})}),has=secs.some(Boolean);
  USEC.forEach(function(c,i){$(c[0]).hidden=!secs[i];if(!secs[i])FOLDS.push(["Generation-phase utilization · "+c[1],c[2]])});
  $("util").hidden=!show||!has;$("utilEmpty").hidden=true;$("utilBody").hidden=!has;
  $("utilIsl").textContent=SEL!=null&&(O.islands||[]).length>1?"· "+lbl(SEL):"";
  if(!has||!show){$("uCarryBox").hidden=true;return}
  var rids={};runs.forEach(function(r){((r.util||{}).rounds||[]).forEach(function(x){rids[x.round]=1})});
  var ids=Object.keys(rids).map(Number).sort(function(a,b){return a-b});
  if(uRound!=="all"&&ids.indexOf(uRound)<0)uRound="all";
  // one button for all rounds, one menu for a single round (newest first): a button per round grew with the run
  var ctl='<button class="btn'+(uRound==="all"?" on":"")+'" data-u="all">all rounds overlaid</button>'+
    '<select id="utilRound" aria-label="pick a round"><option value="">'+(uRound==="all"?"pick a round…":"all rounds")+'</option>'+
    ids.slice().reverse().map(function(i){return '<option value="'+i+'"'+(i===uRound?" selected":"")+'>round '+i+'</option>'}).join("")+'</select>';
  if(runs.length>1)ctl+=' <button class="btn'+(uMode==="side"?" on":"")+'" data-mode="side">side by side</button><button class="btn'+(uMode==="overlay"?" on":"")+'" data-mode="overlay">overlay</button>';
  $("utilCtl").innerHTML=ctl;
  Array.prototype.forEach.call($("utilCtl").querySelectorAll("button"),function(b){b.onclick=function(){
    if(b.dataset.mode)uMode=b.dataset.mode;else uRound="all";util()}});
  var us=$("utilRound");if(us&&us.addEventListener)us.onchange=function(){uRound=us.value===""?"all":+us.value;util()};
  uCut(runs);uDone(runs);uPh(runs);uTr(runs);uLoad(runs);uCarry(runs);
}
function uCut(runs){
  $("uCut").innerHTML=runs.map(function(r,i){var rr=((r.util||{}).rounds||[]);
    if(!rr.some(function(x){return x.cutoff}))return '<div>'+runHead(r,i,runs.length)+'<div class="note">no data (no rl_rollout_cutoff)</div></div>';
    // newest rounds first; a picked round stays in the table even when it is older than the shown ones
    var tc=YetoTbl.cap("cutoff-rounds",rr.slice().reverse(),{def:10,total:(r.util||{}).rounds_total,unit:"rounds"}),shown=tc.rows.slice();
    if(uRound!=="all"&&shown.every(function(x){return x.round!==uRound}))shown=shown.concat(rr.filter(function(x){return x.round===uRound}));
    return '<div>'+runHead(r,i,runs.length)+YetoTbl.spark([["generation s",function(x){return x.gen_s}],["dropped groups",function(x){return (x.cutoff||{}).discarded_groups}],["dropped trajectories",function(x){return (x.cutoff||{}).discarded_trajectories}]].map(function(s){
        return {label:s[0],vals:rr.map(s[1]),fmt:function(v){return unk(v)}}}))+tc.bar+
      YetoTbl.table('<tr><th>round</th><th>generation s</th><th>submitted groups</th><th>target groups</th><th>dropped groups</th><th>dropped trajectories</th><th>dropped tokens</th><th>filtered groups</th></tr>',
      shown.map(function(x){var c=x.cutoff||{};return '<tr'+(x.round===uRound?' class="hl"':"")+'><td class="num">'+x.round+'</td><td class="num">'+unk(x.gen_s)+'</td><td class="num">'+unk(c.submitted_groups)+'</td><td class="num">'+unk(c.target_groups)+
        '</td><td class="num">'+unk(c.discarded_groups)+'</td><td class="num">'+unk(c.discarded_trajectories)+'</td><td class="num">'+unk(c.discarded_tokens)+'</td><td class="num">'+unk(c.filtered_groups)+'</td></tr>'}).join(""))+'</div>'}).join("");
}
function uSel(rr){return uRound==="all"?rr:rr.filter(function(x){return x.round===uRound})}
function uDone(runs){
  var box=$("uDone");box.innerHTML="";
  var groups=uMode==="overlay"&&runs.length>1?[runs.map(function(r,i){return [r,i]})]:runs.map(function(r,i){return [[r,i]]});
  $("legDone").innerHTML=runs.length>1?runs.map(function(r,i){return '<span><i class="sw" style="background:'+RUN_COLORS[i]+'"></i>'+esc(r.label)+'</span>'}).join(""):(uRound==="all"?'<span>one line per round, darker is later</span>':"");
  groups.forEach(function(g){
    var div=document.createElement?document.createElement("div"):null,host=div||box;if(div)box.appendChild(div);
    var lines=[];g.forEach(function(ri){uSel(((ri[0].util||{}).rounds||[])).forEach(function(x,k,arr){if(x.done.length||fin(x.gen_s))lines.push({run:ri[1],round:x.round,done:x.done,cut:x.gen_s,k:k,n:arr.length})})});
    var head=g.length===1?runHead(g[0][0],g[0][1],runs.length):"";
    if(!lines.length){host.innerHTML=head+'<div class="note">no data (no trajectory start/end times)</div>';return}
    if(div)div.innerHTML=head;
    var W=560,H=210,m={l:40,r:12,t:10,b:26};
    var xm=Math.max.apply(null,lines.map(function(l){return Math.max(fin(l.cut)?l.cut:0,l.done.length?l.done[l.done.length-1]:0)}).concat([1]));
    var ym=Math.max.apply(null,lines.map(function(l){return l.done.length}).concat([1]));
    var sx=function(v){return m.l+v*(W-m.l-m.r)/xm},sy=function(v){return m.t+(1-v/ym)*(H-m.t-m.b)};
    var s=el("svg",{viewBox:"0 0 "+W+" "+H,role:"img","aria-label":"Trajectory completion curve"},host),gg=el("g",{"class":"grid"},s);
    [0,.5,1].forEach(function(q){el("line",{x1:m.l,x2:W-m.r,y1:sy(ym*q),y2:sy(ym*q)},gg);el("text",{x:m.l-6,y:sy(ym*q)+4,"text-anchor":"end"},s).textContent=Math.round(ym*q)});
    [0,.25,.5,.75,1].forEach(function(q){el("text",{x:sx(xm*q),y:H-8,"text-anchor":"middle"},s).textContent=Math.round(xm*q)+"s"});
    lines.forEach(function(l){var col=runs.length>1||uRound!=="all"?RUN_COLORS[l.run]:"var(--R)",op=uRound==="all"&&l.n>1?.35+.65*l.k/(l.n-1):1;
      var d="M"+sx(0)+","+sy(0);l.done.forEach(function(t,j){d+="L"+sx(t)+","+sy(j)+"L"+sx(t)+","+sy(j+1)});
      if(fin(l.cut))d+="L"+sx(Math.max(l.cut,l.done.length?l.done[l.done.length-1]:0))+","+sy(l.done.length);
      el("path",{d:d,fill:"none",stroke:col,"stroke-width":2,opacity:op,"class":"udone","data-r":l.round},s);
      if(fin(l.cut))el("line",{x1:sx(l.cut),x2:sx(l.cut),y1:m.t,y2:H-m.b,stroke:col,"stroke-dasharray":"4 3",opacity:op,"class":"ucut"},s)});
  });
}
function phBar(s,y0,h,ph,W,m,mx){var x0=m.l;UPH.forEach(function(p){var v=ph[p[0]];if(fin(v)&&v>0){var w=v*(W-m.l-m.r)/mx;el("rect",{x:x0,y:y0,width:Math.max(w-1,1),height:h,rx:2,fill:p[2]},s);x0+=w}});return x0}
function uPh(runs){
  $("legPh").innerHTML=UPH.map(function(p){return '<span><i class="sw" style="background:'+p[2]+'"></i>'+p[1]+'</span>'}).join("");
  var box=$("uPh");box.innerHTML="";
  runs.forEach(function(r,i){var div=document.createElement?document.createElement("div"):box;if(div!==box)box.appendChild(div);
    var rr=((r.util||{}).rounds||[]).filter(function(x){return UPH.some(function(p){return fin(x.phases[p[0]])})});
    if(!rr.length){div.innerHTML=runHead(r,i,runs.length)+'<div class="note">no data (trajectory events lack the four-part timing fields)</div>';return}
    var pc=YetoTbl.cap("timing-rounds",rr,{def:10,tail:true,unit:"rounds"});rr=pc.rows;
    div.innerHTML=runHead(r,i,runs.length)+pc.bar;
    var W=480,rowH=24,m={l:52,r:110,t:4,b:4},H=m.t+m.b+rowH*rr.length;
    var mx=Math.max.apply(null,rr.map(function(x){return UPH.reduce(function(a,p){return a+(x.phases[p[0]]||0)},0)}).concat([1]));
    var s=el("svg",{viewBox:"0 0 "+W+" "+H,role:"img","aria-label":"Four-part timing"},div);
    rr.forEach(function(x,k){var y0=m.t+k*rowH;if(x.round===uRound)el("rect",{x:0,y:y0,width:W,height:rowH,fill:"var(--band)"},s);
      el("text",{x:m.l-8,y:y0+rowH/2+4,"text-anchor":"end"},s).textContent="round "+x.round;
      var xe=phBar(s,y0+5,rowH-10,x.phases,W,m,mx);
      el("text",{x:xe+6,y:y0+rowH/2+4},s).textContent="tool "+(fin(x.tool_share)?(x.tool_share*100).toFixed(1)+"%":"unknown")+" · "+x.trajectories+" trajectories"});
  });
}
function uTr(runs){
  var box=$("uTr");box.innerHTML="";
  if(uRound==="all"){$("uTrNote").textContent="(pick a round above first)";return}
  $("uTrNote").textContent="(round "+uRound+", sorted by total time; bar length = sum of the four parts)";
  runs.forEach(function(r,i){var div=document.createElement?document.createElement("div"):box;if(div!==box)box.appendChild(div);
    var tr=((r.util||{}).trajectories||[]).filter(function(t){return t.rid===uRound}).sort(function(a,b){return (b.dur||0)-(a.dur||0)});
    if(!tr.length){div.innerHTML=runHead(r,i,runs.length)+'<div class="note">no data</div>';return}
    var tcap=YetoTbl.cap("round-trajectories",tr,{def:20,unit:"trajectories"});tr=tcap.rows;
    div.innerHTML=runHead(r,i,runs.length)+tcap.bar.replace("latest ","longest ");
    var W=480,rowH=14,m={l:8,r:70,t:2,b:2},H=m.t+m.b+rowH*tr.length;
    var mx=Math.max.apply(null,tr.map(function(t){return UPH.reduce(function(a,p){return a+(t[p[0]]||0)},0)}).concat([1]));
    var s=el("svg",{viewBox:"0 0 "+W+" "+H,role:"img","aria-label":"Four-part timing per trajectory"},div);
    tr.forEach(function(t,k){var y0=m.t+k*rowH,xe=phBar(s,y0+2,rowH-4,t,W,m,mx);
      var tl=el("title",{},el("rect",{x:0,y:y0,width:W,height:rowH,fill:"transparent"},s));
      tl.textContent=(t.task||"")+" · generation "+unk(t.gen,1)+" s · tool "+unk(t.tool,1)+" s · judging "+unk(t.judge,1)+" s · sandbox "+unk(t.sandbox,1)+" s · reward "+unk(t.reward,2);
      el("text",{x:xe+4,y:y0+rowH-3},s).textContent=fin(t.tool)&&fin(t.gen)?"tool "+(t.tool/((t.gen||0)+(t.tool||0)+(t.judge||0)+(t.sandbox||0))*100).toFixed(0)+"%":"unknown"});
  });
}
function uLoad(runs){
  var box=$("uLoad");box.innerHTML="";
  $("legLoad").innerHTML='<span><i class="sw" style="background:var(--T)"></i>KV usage %</span><span><i class="sw" style="background:var(--R)"></i>queued requests</span><span><i class="sw" style="background:var(--band)"></i>generation span</span>';
  runs.forEach(function(r,i){var div=document.createElement?document.createElement("div"):box;if(div!==box)box.appendChild(div);
    var u=r.util||{},ld=(u.load||[]).filter(function(p){return fin(p[0])});
    if(!ld.length){div.innerHTML=runHead(r,i,runs.length)+'<div class="note">no data (no rl_load_sample)</div>';return}
    div.innerHTML=runHead(r,i,runs.length)+(u.load_has_kv?"":'<div class="note">KV: not in this tape (SGLang did not report kv_used_tokens)</div>');
    var t0=ld[0][0],t1=ld[ld.length-1][0];
    var spans=(u.rounds||[]).filter(function(x){return x.gen}).map(function(x){return [x.gen[0],x.gen[1],x.round]});
    if(uRound!=="all"){var sp=spans.filter(function(q){return q[2]===uRound})[0];if(sp){var pad=(sp[1]-sp[0])*.1;t0=sp[0]-pad;t1=sp[1]+pad;ld=ld.filter(function(p){return p[0]>=t0&&p[0]<=t1})}}
    var kvm=Math.max.apply(null,ld.map(function(p){return fin(p[1])?p[1]:0}).concat([.01])),qm=Math.max.apply(null,ld.map(function(p){return fin(p[2])?p[2]:0}).concat([1]));
    var W=560,H=180,m={l:40,r:36,t:8,b:22},sx=function(t){return m.l+(t-t0)*(W-m.l-m.r)/((t1-t0)||1)};
    var syk=function(v){return m.t+(1-v/kvm)*(H-m.t-m.b)},syq=function(v){return m.t+(1-v/qm)*(H-m.t-m.b)};
    var s=el("svg",{viewBox:"0 0 "+W+" "+H,role:"img","aria-label":"KV usage and queueing"},div),g=el("g",{"class":"grid"},s);
    spans.forEach(function(q){var a=Math.max(q[0],t0),b=Math.min(q[1],t1);if(b>a)el("rect",{x:sx(a),y:m.t,width:sx(b)-sx(a),height:H-m.t-m.b,fill:"var(--band)"},s)});
    [0,.5,1].forEach(function(q){el("line",{x1:m.l,x2:W-m.r,y1:syk(kvm*q),y2:syk(kvm*q)},g);el("text",{x:m.l-6,y:syk(kvm*q)+4,"text-anchor":"end"},s).textContent=(kvm*q*100).toFixed(0)+"%";
      el("text",{x:W-m.r+6,y:syq(qm*q)+4},s).textContent=Math.round(qm*q)});
    var kp=ld.filter(function(p){return fin(p[1])});if(kp.length)el("path",{d:"M"+kp.map(function(p){return sx(p[0])+","+syk(p[1])}).join("L"),fill:"none",stroke:"var(--T)","stroke-width":2},s);
    var qp=ld.filter(function(p){return fin(p[2])});if(qp.length)el("path",{d:"M"+qp.map(function(p){return sx(p[0])+","+syq(p[2])}).join("L"),fill:"none",stroke:"var(--R)","stroke-width":1.5,"stroke-dasharray":"2 2"},s);
    [t0,(t0+t1)/2,t1].forEach(function(t){el("text",{x:sx(t),y:H-6,"text-anchor":"middle"},s).textContent=clock(t)});
  });
}
function uCarry(runs){
  var any=runs.some(function(r){return (r.util||{}).carry});$("uCarryBox").hidden=!any;if(!any){$("uCarry").innerHTML="";return}
  $("uCarry").innerHTML=runs.map(function(r,i){var c=(r.util||{}).carry||[],cc=YetoTbl.cap("carried-trajectories",c.slice().reverse(),{def:10,total:(r.util||{}).carry_total,unit:"trajectories"});
    return '<div>'+runHead(r,i,runs.length)+(c.length?cc.bar+'<table class="c"><tr><th>trajectory</th><th>task</th><th>start round → training round</th><th>version spans</th></tr>'+cc.rows.map(function(x){
      return '<tr><td class="mono">'+esc(String(x.id||"").slice(-10))+'</td><td>'+esc(x.task||"")+'</td><td class="num">'+(x.from==null?"unknown":x.from)+' → '+(x.to==null?"unknown":x.to)+'</td><td class="mono">'+esc((x.versions||[]).map(function(v){return Array.isArray(v)?"v"+v[0]+"["+v[1]+","+v[2]+")":"v"+v}).join(" "))+'</td></tr>'}).join("")+'</table>':'<div class="note">no carried-over trajectories</div>')+'</div>'}).join("");
}
/* nodes */
function chartNodes(){
  var ex=V.islands[SEL]||{},ns=ex.node_series||{},ks=Object.keys(ns).sort(),box=$("cNodes");box.innerHTML="";$("legNodes").innerHTML="";
  if(!ks.length){box.innerHTML='<div class="note">'+esc(ex.gpu_mem_note||"not sampled (no host probe data)")+'</div>';$("nodesNote").textContent="";return}
  var col=function(k){return k==="0"?"var(--T)":k==="1"?"var(--R)":"var(--i2)"};
  $("legNodes").innerHTML=ks.map(function(k){return '<span><i class="sw" style="background:'+col(k)+'"></i>node '+esc(k)+(k==="0"?" (driver)":"")+'</span>'}).join("");
  var all=[];ks.forEach(function(k){ns[k].forEach(function(p){all.push(p)})});
  var t0=Math.min.apply(null,all.map(function(p){return p[0]})),t1=Math.max.apply(null,all.map(function(p){return p[0]})),vm=Math.max.apply(null,all.map(function(p){return p[1]}).concat([1]));
  var W=boxW(box,560),H=210,m={l:44,r:16,t:8,b:26},sx=function(t){return m.l+(t-t0)*(W-m.l-m.r)/((t1-t0)||1)},sy=function(v){return m.t+(1-v/vm)*(H-m.t-m.b)};
  var s=el("svg",{viewBox:"0 0 "+W+" "+H,role:"img","aria-label":"GPU memory per node"},box),g=el("g",{"class":"grid"},s);
  [0,.5,1].forEach(function(q){el("line",{x1:m.l,x2:W-m.r,y1:sy(vm*q),y2:sy(vm*q)},g);el("text",{x:m.l-6,y:sy(vm*q)+4,"text-anchor":"end"},s).textContent=(vm*q).toFixed(0)});
  // no line across a gap (3x the usual spacing): an island down or restarted draws two pieces, not a diagonal
  ks.forEach(function(k){var P=ns[k].slice().sort(function(a,b){return a[0]-b[0]});
    el("path",{d:YetoExt.segments(P,P).map(function(G){return "M"+G.map(function(p){return sx(p[0])+","+sy(p[1])}).join("L")}).join(""),fill:"none",stroke:col(k),"stroke-width":2},s)});
  tAxis(s,t0,t1,sx,H);
  var hasU=ks.some(function(k){return ns[k].some(function(p){return fin(p[2])})});
  $("nodesNote").textContent="Source: per-node host probe or the island's NVML samples (rl_resource_sample)."+(hasU?"":" GPU utilization: not sampled in this tape (older probes record memory only).");
}
/* island health of a synchronous run; an async run has it in its island block (header, Samplers & pools) */
function islands(){
  $("nodes").hidden=!!V.async;if(V.async)return;
  $("islands").innerHTML=(O.islands||[]).map(function(c){var ex=V.islands[c.id]||{},re=ex.ray_embed||{};
    return '<details class="isl"'+(String(c.id)===String(SEL)?" open":"")+'><summary><span><b>'+nameCell(c.id)+'</b> <span class="note">'+esc(c.name||"")+'</span></span>'+negBadge(c)+stBadge(c.status,c)+'</summary>'+
    naDash('<div class="kv"><span>cloud / GPU</span><span class="num">'+esc(platform(c))+' · '+esc(c.gpu||"n/a")+' × '+esc(c.gpus==null?"n/a":c.gpus)+'</span>'+
    '<span>phase / policy</span><span class="num">'+esc(islPhase(c)||"n/a")+' · '+(c.policy_version==null?"n/a":"v"+esc(c.policy_version))+'</span>'+
    '<span>last event</span><span class="num">'+(fin(c.last_event_age_s)?f(c.last_event_age_s,0)+" s ago":"n/a")+'</span>',"not reported by this island yet")+
    (c.starting?'<span>startup elapsed</span><span class="num">'+hm(c.startup_s)+'</span>':"")+
    (Object.keys(c.startup_steps||{}).length?'<span>startup steps</span><span class="num">'+
      Object.keys(c.startup_steps).map(function(k){var v=c.startup_steps[k];
        return esc({ray_connected:"Ray cluster formed",engine_ready:"inference engine ready",weights_loaded:"training weights loaded"}[k]||k)+" "+(fin(v.step_s)?f(v.step_s,0)+" s":"?")}).join(" · ")+
      (c.starting&&fin(c.startup_step_age_s)?" · current step waiting "+f(c.startup_step_age_s,0)+" s":"")+'</span>':"")+
    (c.stopped_by_us?'<span>stop</span><span>stopped by us ('+esc((O.operator_stop||{}).cause)+', marker '+esc((O.operator_stop||{}).marker)+')</span>':"")+
    (c.nodes||[]).map(function(n){var pk=Math.max.apply(null,(n.gpu_mem_used_mib_peak||[0]).concat([0]));
      return '<span>node '+esc(n.node)+'</span><span class="num">peak memory '+(pk/1024).toFixed(1)+' GiB · utilization '+(fin(n.gpu_util_pct)?f(n.gpu_util_pct,0)+"%":"not sampled")+'</span>'}).join("")+
    (ex.hardware?'<span>GPU type / driver / CUDA</span><span>'+esc(ex.hardware.compat_group||"not declared")+' / '+esc(ex.hardware.driver_version||"not recorded")+' / '+esc(ex.hardware.cuda_version||"not recorded")+'</span>':"")+
    '<span>cells / transactions</span><span class="num">'+esc(ex.cells||0)+' / '+esc(ex.transactions||0)+'</span>'+
    '<span>Ray dashboard</span><span>'+esc(re.note||(re.command?re.command:"localhost :"+re.port))+'</span></div></details>'}).join("")||'<div class="note">no data</div>';
}
function cost(){
  var c=O.cost||{},ci=c.islands||[];
  var ni=(O.islands||[]).length;
  if(!ci.length){$("costBox").innerHTML='<div class="note">'+(ni?'no cost data for the '+ni+' island(s): cost needs island_ready in fleet.jsonl or the cloud and GPU type on the tapes':'no island has reported yet, so nothing is priced')+
    (fin(c.budget_usd)?' · budget $'+f(c.budget_usd,0):'')+'. Prices: '+esc(c.prices_source||"")+'</div>';return}
  $("costBox").innerHTML='<div class="tw"><table class="c cpt"><tr><th></th><th>GPU/h</th><th>CPU+memory/h</th><th>total/h</th><th>duration</th><th>cumulative</th></tr>'+ci.map(function(r){
    // a fleet's rows (cost.fleet_rows): each island's trainer, its pools' samplers and the syncer
    var lab=r.kind==="syncer"?"Syncer":r.kind==="pool sampler"?"Island "+esc(r.island)+" pool "+esc(r.pool)+' <span class="note">'+esc(r.id)+'</span>':r.kind==="trainer"?"Island "+esc(r.island)+" trainer":"Island "+esc(r.id);
    var rs=r.rate_source?' title="'+esc(r.rate_source+(r.billed_from?"; hours: "+r.billed_from:""))+'"':"";
    if(r.kind)return '<tr><td'+rs+'>'+lab+'</td><td class="num">'+(fin(r.gpu_rate_usd_h)?"$"+f(r.gpu_rate_usd_h,2):r.kind==="syncer"?"—":"unpriced")+'</td><td class="num">'+(fin(r.host_rate_usd_h)?"$"+f(r.host_rate_usd_h,2):fin(r.rate_usd_h)&&r.rate_source?"in total":r.kind==="syncer"?"—":"GPU only")+'</td><td class="num">'+(fin(r.rate_usd_h)?"$"+f(r.rate_usd_h,2):"unpriced")+'</td><td class="num">'+f(r.hours,2)+' h</td><td class="num">'+(fin(r.cost_usd)?"$"+f(r.cost_usd,2):fin(r.rate_usd_h)?"no hours yet":"unpriced")+'</td></tr>';
    return '<tr><td>Island '+esc(r.id)+'</td><td class="num">'+(fin(r.gpu_rate_usd_h)?"$"+f(r.gpu_rate_usd_h,2):"unpriced")+'</td><td class="num">'+(fin(r.host_rate_usd_h)?"$"+f(r.host_rate_usd_h,2):"GPU only")+'</td><td class="num">'+(fin(r.rate_usd_h)?"$"+f(r.rate_usd_h,2):"n/a")+'</td><td class="num">'+f(r.hours,2)+' h</td><td class="num">'+(fin(r.cost_usd)?"$"+f(r.cost_usd,1):"n/a")+'</td></tr>'}).join("")+'</table></div>'+
    '<div class="note" style="margin-top:6px">'+(c.fleet?esc("fleet spend = pools $"+f(c.fleet.pools_usd,2)+" + trainers "+(fin(c.fleet.trainers_usd)?"$"+f(c.fleet.trainers_usd,2):"not priced ("+(c.fleet.trainers_note||"no trainer row")+")")+" + syncer "+(fin(c.fleet.syncer_usd)?"$"+f(c.fleet.syncer_usd,2):"unpriced")+" (hover a row for its price and hours source). "):"")+esc(c.note||"")+(function(){var sp=ci.filter(function(r){return !r.kind||r.cpus!=null});return sp.length?'. Specs: '+sp.map(function(r){return "island "+esc(r.id)+" "+(r.cpus==null?"n/a":r.cpus)+" cores, "+(r.memory_gib==null?"n/a":r.memory_gib)+" GiB"}).join("; "):""})()+'. Prices: '+esc(c.prices_source||"")+'</div>';
}
var EV_KEEP={rl_engine_selected:1,rl_driver_start:1,rl_driver_phase:1,rl_publication:1,rl_round_trained:1,rl_reconfiguration:1,rl_learner_finalized:1,
  island_ready:1,island_lost:1,island_stop:1,island_done:1,island_leave:1,operator_stop:1,dashboard_source_lost:1,rl_elastic_recommendation:1,
  rl_policy_released:1,rl_sampler_start:1,rl_sampler_join:1,rl_sampler_policy_loaded:1,rl_sampler_backlog_full:1,rl_sampler_admitted:1,rl_sampler_stopped:1};
var POOL_BAD=/evicted|node_lost|crash_loop|failed|breaker|timeout|overdue|orphan|refused|manual_release/;
function keepEv(t){return EV_KEEP[t]||/^nc_elastic_/.test(t)}
var EVENTS=[],EV_CURSOR=0;
function evLevel(t,rec){if(t==="operator_stop")return ["mu","stop"];
  if(/^nc_elastic_/.test(t))return POOL_BAD.test(t)?["bd","pool"]:["wn","pool"];if(t==="island_leave")return ["wn","left"];if(t==="island_lost"||(t==="rl_reconfiguration"&&rec.result==="RECOVERY_REQUIRED"))return ["bd","failure"];
  if(t==="rl_publication"||t==="rl_round_trained")return ["ok","done"];if(t==="rl_reconfiguration"||t==="dashboard_source_lost")return ["wn","change"];return ["mu","info"]}
function evText(e){var r=e.record||{},keys=["phase","rollout_id","policy_version","result","cause","recommendation","marker","reason_name","backlog_groups","buffer_selected","ticket","shard","why","detail"];
  return keys.filter(function(k){return r[k]!=null}).map(function(k){return k+"="+String(r[k]).slice(0,80)}).join(" ")}
/* newest first by event time; the tapes are read one after another, so arrival (seq) order is not time order.
   One total-order key per event: its time (an event without one takes the time of the last timed event that
   arrived before it, -inf before any) and its arrival index; ties keep the arrival order, newest arrival first */
function evOrder(evs){var t=-Infinity;return evs.map(function(e,i){if(fin(e.ts))t=e.ts;return [e,fin(e.ts)?e.ts:t,i]})
  .sort(function(a,b){return a[1]!==b[1]?(b[1]>a[1]?1:-1):b[2]-a[2]}).map(function(x){return x[0]})}
function events(){
  var ec=YetoTbl.cap("event-stream",evOrder(EVENTS.filter(function(e){return keepEv(e.type)})),{def:15,unit:"events"}),list=ec.rows;
  $("evs").innerHTML=ec.bar+list.map(function(e){var l=evLevel(e.type,e.record||{});return '<div><span class="t">'+(fleetOn()&&!fin(e.ts)?'<span title="no time on the record">—</span>':clock(e.ts))+'</span><span class="st '+l[0]+'" title="'+l[1]+'"><i></i>'+l[1]+'</span><span class="x"><b>'+esc(e.type.replace(/^rl_/,""))+'</b>'+(e.island!=null?" "+esc(cardOf(e.island)?lbl(e.island):"island "+e.island):"")+' '+esc(evText(e))+'</span></div>'}).join("")||'<div class="note">none</div>';
}
function cmds(){
  var run=O.run||"<run>",cm=O.commands||{};
  var list=[["Local dashboard",cm.serve||"yeto dashboard serve --run "+run+" --port 8787"],["Export offline page",cm.export||"yeto dashboard export --run "+run+" -o "+run+".html"],
    ["Run status","yeto status "+run]];
  if(((O.islands||[])[0]||{}).cloud==="modal")list.push(["Modal logs","modal app logs yeto-"+run],["Stop app","modal app stop --yes yeto-"+run]);
  $("cmdBox").innerHTML=list.map(function(c,i){return '<div class="cmd"><span class="l">'+esc(c[0])+'</span><code id="cmd'+i+'">'+esc(c[1])+'</code><button class="btn" data-c="'+i+'">Copy</button></div>'}).join("");
  Array.prototype.forEach.call($("cmdBox").querySelectorAll("button"),function(b){b.onclick=function(){var t=$("cmd"+b.dataset.c).textContent;
    var ok=function(){b.textContent="Copied";setTimeout(function(){b.textContent="Copy"},1200)},bad=function(){b.textContent="Copy manually"};
    if(navigator.clipboard&&navigator.clipboard.writeText)navigator.clipboard.writeText(t).then(ok,bad);else bad()}});
}
function folds(){
  var u=V.usage||{},multi=(O.islands||[]).length>1,items=[];
  if(!u.syncer)items.push(["Syncer merge timeline / round table","no syncer tape"]);
  if(!u.transactions)items.push(["Reconfiguration transactions","no journal transactions"]);
  if(!multi)items.push(["Inter-island elasticity / leases","only one island"]);
  if((O.islands||[]).every(function(c){return (c.cloud||"")==="modal"}))items.push(["Ray dashboard embed","Modal containers have no inbound port"]);
  items=items.concat(FOLDS);
  // one line, opened on demand: each panel and the tape event it would need
  $("foldBox").innerHTML=items.length?'<details class="ufold"><summary><b>Panels not used in this run</b> · '+items.length+': '+esc(items.map(function(x){return x[0].replace(/^Generation-phase utilization · /,"")}).join(" · "))+'</summary>'+
    items.map(function(x){return '<div class="fold"><span>'+esc(x[0])+'</span><span>not used in this run ('+esc(x[1])+')</span></div>'}).join("")+'</details>':'<div class="note">Panels not used in this run: none</div>';
}
function spot(){
  var rows=(V.spot||[]).slice().reverse();$("spotBox").hidden=!rows.length;if(!rows.length)return;
  var sc=YetoTbl.cap("spot",rows,{def:10,unit:"events"});
  $("spot").innerHTML=sc.bar+YetoTbl.table('<tr><th>event</th><th>island</th><th>cloud / region</th><th>source</th><th>seconds left</th><th>saved</th><th>outcome</th><th>candidates/rejected</th></tr>',sc.rows.map(function(r){
    return '<tr><td>'+esc(r.event)+'</td><td>'+esc(r.island)+'</td><td class="mono">'+esc(r.cloud||"n/a")+' / '+esc(r.region||"n/a")+'</td><td>'+esc(r.source||"")+'</td><td class="num">'+f(r.remaining_s,1)+'</td><td>'+(r.saved==null?"":(r.saved?"yes":"no"))+'</td><td>'+esc(r.outcome||(r.auto_launch===false?"advice only":""))+'</td><td class="num">'+(r.candidates==null?"":esc(r.candidates)+' / '+esc(r.rejected))+'</td></tr>'}).join(""))}
/* async GRPO: trainer rounds, samplers, buffer backlog, releases, NC pool */
// trainer rounds table: [header, title, cell(r, a), has(r, a) or null = always shown]; a column no shown
// round has a value for (an algorithm that does not report it, e.g. PPO KL with score centering) is hidden
// and named under the table
var isClip=function(r){return fin(r.tis_clip)?r.tis_clip:r.is_clip};
var TR_COLS=[["round","",function(r){return r.round},null],
  ["reward mean ± std","",function(r){return f(r.reward)+(fin(r.reward_std)?" ± "+f(r.reward_std):"")},null],
  ["p10 · p50 · p90","reward quantiles of the round's samples",function(r){return cq(r.p10)+" · "+cq(r.p50)+" · "+cq(r.p90)},function(r){return fin(r.p50)}],
  ["train-rollout KL","train_metrics.train_rollout_kl: trainer vs sampler logprobs",function(r){return f(r.kl_tr,5)},function(r){return fin(r.kl_tr)}],
  ["PPO KL","train_metrics.ppo_kl: trainer policy vs the old logprobs of the batch",function(r){return fin(r.kl)&&Math.abs(r.kl)<1e-3&&r.kl!==0?r.kl.toExponential(2):f(r.kl,5)},function(r){return fin(r.kl)}],
  ["logprob |diff|","",function(r){return f(r.logprob_diff,4)},function(r){return fin(r.logprob_diff)}],
  ["TIS / IS clipped","mismatch.tis_clipfrac (TIS); with score centering train_metrics.sc_is_clipped (share of tokens whose IS weight was clipped)",function(r){return f(isClip(r),4)},function(r){return fin(isClip(r))}],
  ["grad_norm","",function(r){return f(r.grad_norm,4)},function(r){return fin(r.grad_norm)}],
  ["generated tokens mean / p95","model-generated tokens per trained sample (loss mask); full = whole response incl. tool / env tokens, when the tape has it",
   function(r){return f(r.resp_mean,0)+(r.resp_derived?"*":"")+" / "+f(r.resp_p95,0)+(fin(r.resp_full_mean)?' <span class="note">· full '+f(r.resp_full_mean,0)+'</span>':"")},function(r){return fin(r.resp_mean)}],
  ["truncated","",function(r){return pct(r.trunc)},function(r){return fin(r.trunc)}],["tok/s","",function(r){return f(r.tok_s,0)},function(r){return fin(r.tok_s)}],
  ["groups","",function(r,a){return f(a.trained_groups,0)},null],
  ["zero-variance","rl_local_round.zero_variance_group_ratio: groups whose rewards were all equal (trained with zero advantage)",function(r,a){return fin(r.zero_var_ratio)?pct(r.zero_var_ratio)+(fin(a.trained_groups)?" ("+f(Math.round(r.zero_var_ratio*a.trained_groups),0)+")":""):"n/a"},function(r){return fin(r.zero_var_ratio)}],
  ["stale dropped","",function(r,a){return f(a.stale_groups,0)},null],
  ["policy ages","rl_round_trained.policy_age_groups: trained groups by policy age (age:groups)",function(r,a){return histS(a.ages)+(fin(a.policy_age_mean)?' <span class="note">· mean '+f(a.policy_age_mean,2)+'</span>':"")},null]];
/* a quantile in a one-line cell: 0, 1, 0.5 (two decimals at most) */
function cq(v){return fin(v)?String(+v.toFixed(2)):"n/a"}
/* mean policy age of a round's trained groups: its age histogram, else the trainer's own mean */
function ageMean(a){var h=a.ages||{},n=0,s=0;Object.keys(h).forEach(function(x){if(fin(h[x])&&!isNaN(+x)){n+=h[x];s+=h[x]*(+x)}});return n?s/n:fin(a.policy_age_mean)?a.policy_age_mean:null}
/* rows: every trained round, oldest first; the table shows the newest (YetoTbl), the sparklines all of them */
function trTable(rows,ar,key){var cols=TR_COLS.filter(function(c){return !c[3]||rows.some(function(r){return c[3](r,ar[r.round]||{})})}),gone=TR_COLS.filter(function(c){return cols.indexOf(c)<0});
  var A=function(r){return ar[r.round]||{}},sp=[["reward",function(r){return r.reward},3],["train-rollout KL",function(r){return r.kl_tr},5],["logprob |diff|",function(r){return r.logprob_diff},4],
    ["grad_norm",function(r){return r.grad_norm},3],["TIS / IS clipped",isClip,4],["tok/s",function(r){return r.tok_s},0],["policy age",function(r){return ageMean(A(r))},2]];
  var c=YetoTbl.cap(key,rows.slice().reverse(),{def:10,unit:"rounds"});
  return YetoTbl.spark(sp.map(function(x){return {label:x[0],vals:rows.map(x[1]),fmt:function(v){return f(v,x[2])},title:x[0]+" per round, whole run"}}))+c.bar+
    YetoTbl.table('<tr>'+cols.map(function(c){return '<th'+(c[1]?' title="'+esc(c[1])+'"':"")+'>'+esc(c[0])+'</th>'}).join("")+'</tr>',
    c.rows.map(function(r){var a=A(r);return '<tr>'+cols.map(function(c,i){return '<td'+(i===cols.length-1?"":' class="num"')+'>'+c[2](r,a)+'</td>'}).join("")+'</tr>'}).join(""))+
    (gone.length?'<div class="note">hidden: '+gone.map(function(c){return esc(c[0])}).join(", ")+' (not reported by this algorithm / run)</div>':"")}
function hist(h){var ks=Object.keys(h||{});return ks.length?ks.map(function(k){return "age "+esc(k)+": "+f(h[k],0)}).join(" · "):"n/a"}
/* a policy-age histogram in a one-line cell: "age:groups", e.g. "11:2 12:6" */
function histS(h){var ks=Object.keys(h||{});return ks.length?ks.map(function(k){return esc(k)+":"+f(h[k],0)}).join(" "):"n/a"}
/* the picked island of an async run: a fleet island (its pools' samplers, its releases and backlog), else the picked
   trainer of a buffer run with every sampler of the run (they share the buffer). One drawing path for both */
function asyncCtx(){var A=V.async,cards=O.islands||[];
  if(fleetOn()){var x=A.islands.filter(function(i){return "i"+i.island===PICK})[0]||A.islands[0],sids=[];
    x.pools.forEach(function(p){p.samplers.forEach(function(id){sids.push(String(id))})});
    return {fleet:true,x:x,tid:x.trainer!=null?String(x.trainer):null,sids:sids,pools:x.pools,rel:x.releases||[],relTotal:((V.table_totals||{}).island_releases||{})[x.island],backlog:x.backlog,key:"-island-"+x.island}}
  var trs=cards.filter(function(c){return c.role==="trainer"}),tid=trs.some(function(c){return String(c.id)===String(SEL)})?String(SEL):trs.length?String(trs[0].id):null;
  var sids=cards.filter(function(c){return c.role==="sampler"}).map(function(c){return String(c.id)});
  Object.keys(A.samplers||{}).forEach(function(k){if(sids.indexOf(k)<0)sids.push(k)});
  var rel=A.releases||[],multi=trs.length>1;if(multi)rel=rel.filter(function(r){return r.trainer==null||String(r.trainer)===tid});
  return {fleet:false,tid:tid,sids:sids,pools:null,rel:rel,relTotal:multi?null:(V.table_totals||{}).releases,backlog:A.backlog,key:multi?"-trainer-"+tid:""}}
var ENG={vllm:"vLLM",sglang:"SGLang"},ENG_S={vllm:"vLLM",sglang:"SGL"};
function engName(e){return e?(ENG[String(e).toLowerCase()]||String(e)):null}
function poolOf(ctx){var o={};(ctx.pools||[]).forEach(function(p){p.samplers.forEach(function(s){o[String(s)]=p})});return o}
/* a sampler's short tag in the release table and timeline: its engine ("vLLM", "SGL", "Miles"), numbered when two
   samplers of the island share one ("SGL 0", "SGL 1"); without an engine the tail of its id */
function samplerTags(ctx){var S=V.async.samplers||{},pool=poolOf(ctx),out={},cnt={};
  ctx.sids.forEach(function(id){var p=pool[id],pr=(S[id]||{}).provenance||{},e=p&&p.source==="miles"?"Miles":(p&&(p.engine||(p.provenance||{}).engine))||pr.engine;
    e=e?(ENG_S[String(e).toLowerCase()]||String(e)):null;out[id]=e;if(e)cnt[e]=(cnt[e]||0)+1});
  // samplers sharing an engine: numbered by their shard number when those differ, else 1, 2, ... in table order
  var tail=function(id){var m=/-(\d+)$/.exec(id);return m?m[1]:null},grp={};
  ctx.sids.forEach(function(id){var e=out[id];if(e&&cnt[e]>1)(grp[e]=grp[e]||[]).push(id)});
  Object.keys(grp).forEach(function(e){var t=grp[e].map(tail),uniq=t.every(function(x,i){return x!=null&&t.indexOf(x)===i});
    grp[e].forEach(function(id,i){out[id]=e+" "+(uniq?t[i]:i+1)})});
  ctx.sids.forEach(function(id){if(!out[id])out[id]=/^\d+$/.test(id)?"s"+id:id.length>10?id.split("-").slice(-2).join("-"):id});
  var seen={};ctx.sids.forEach(function(id){if(seen[out[id]])out[id]=id;seen[out[id]]=1});return out}
/* a timeline lane: the tag and the id, unless the tag is made from the id */
function laneName(tag,id){return tag===id||tag==="s"+id||id.slice(-tag.length)===tag?lbl(id):tag+" · "+id}
function loadTag(ld,lab,id,v){var who=lbl(id);
  if(!ld)return '<span class="lt no" title="'+esc(who+" never loaded v"+v)+'">'+esc(lab)+'</span>';
  var t=who+" loaded v"+v+" at "+clock(ld.loaded_at)+(ld.at_start?" (started on it)":fin(ld.latency_s)?", "+f(ld.latency_s,0)+" s after the release":"")+
    [["resolve",ld.resolve_s],["drain",ld.drain_s],["load",ld.load_s],["pause",ld.pause_s]].filter(function(x){return fin(x[1])}).map(function(x){return " · "+x[0]+" "+f(x[1],1)+" s"}).join("")+
    (fin(ld.drain_in_flight)?" · "+ld.drain_in_flight+" trials in flight at the drain":"")+(ld.drain_cut?" · "+ld.drain_cut+" cut by the drain":"");
  return '<span class="lt'+(ld.at_start?" st":"")+'" title="'+esc(t)+'">'+esc(lab)+" "+(ld.at_start?"at start":fin(ld.latency_s)?f(ld.latency_s,0)+"s":clock(ld.loaded_at))+(fin(ld.drain_s)?" · drain "+f(ld.drain_s,0)+"s":"")+(ld.drain_cut?" · "+ld.drain_cut+" cut":"")+'</span>'}
/* releases of the picked island: version, release time, upload, then one load tag per sampler of the island */
function relTable(rows,sids,tags,key,total){
  var c=YetoTbl.cap(key,rows.slice().reverse(),{def:10,total:total,unit:"releases"});
  var leg=sids.length?'<div class="ltk">'+sids.map(function(id){return '<span><span class="lt">'+esc(tags[id])+'</span>'+esc(lbl(id))+'</span>'}).join("")+'<span><span class="lt no">not loaded</span></span></div>':"";
  return leg+c.bar+YetoTbl.table('<tr><th class="l">policy</th><th>released</th><th>upload</th><th class="l" title="seconds from the release to the sampler\'s load done; hover a tag for resolve / drain / load / pause">loaded by (s after release)</th></tr>',
    // a sampler starting on the base weights makes a v0 row with no trainer release: "base weights"
    c.rows.map(function(r){var base=r.trainer==null&&!fin(r.released_at);
      return '<tr><td class="l num">v'+esc(r.policy_version)+'</td><td class="num">'+(base?'<span class="note">base weights</span>':clock(r.released_at))+'</td><td class="num">'+(base||!fin(r.upload_s)?"—":f(r.upload_s,1)+" s")+'</td><td class="l">'+
        (sids.length?'<div class="lts">'+sids.map(function(id){return loadTag((r.loads||{})[id],tags[id],id,r.policy_version)}).join("")+'</div>':'<span class="note">no sampler</span>')+'</td></tr>'}).join(""))}
function mmCell(m){return m?f(m.value,4)+' <span class="note" title="'+esc(m.method)+'">'+m.rounds+" round"+(m.rounds===1?"":"s")+"</span>":'<span class="note">not reported</span>'}
var SRC_TAG={miles:["Miles native","var(--delta)"],external:["external pool","var(--traj)"]};
function srcTag(s){var t=SRC_TAG[s];return t?' <span class="tag" style="border-color:'+t[1]+';color:'+t[1]+';font-weight:600;white-space:nowrap">'+t[0]+'</span>':""}
function poolEngine(p){var pr=p.provenance||{},prec=String(p.rollout_quant||pr.precision||"").toUpperCase();
  // a Miles pool: the cards of its --gpu cluster when its sampler reports no provenance
  var card=pr.card_model||(p.card?p.card+(fin(p.gpus)?" × "+p.gpus:""):null);
  return [engName(p.engine||pr.engine),prec,card,pr.topology,p.provider].filter(Boolean).join(" · ")||"not reported"}
function samplerEngine(id){var pr=((V.async.samplers||{})[id]||{}).provenance||{},c=cardOf(id)||{};
  return [engName(pr.engine),String(pr.precision||"").toUpperCase(),pr.card_model||c.gpu?(pr.card_model||c.gpu)+(fin(c.gpus)?" × "+c.gpus:""):null,pr.topology,c.cloud].filter(Boolean).join(" · ")||"not reported"}
/* the engine's device family and cloud as the sampler's start record names them, on hover */
function platTitle(p,id){var pr=(p&&p.provenance&&p.provenance.device_family?p.provenance:((V.async.samplers||{})[id]||{}).provenance)||{},c=cardOf(id)||{};
  var t=[pr.engine||(p&&p.engine),pr.device_family?"on "+pr.device_family:null,c.cloud?"· "+c.cloud:"(cloud not on the tape)"].filter(Boolean).join(" ");return t?' title="'+esc(t)+'"':""}
/* zero-variance groups the trainers saw (rl_local_round ratio × trained groups), for samplers that report none */
function zeroVarTrainer(){var A=V.async,zg=0,zt=0;Object.keys(A.trainers||{}).forEach(function(i){var tg={};((A.trainers[i]||{}).rounds||[]).forEach(function(x){tg[x.rollout_id]=x.trained_groups});
  ((V.islands[i]||{}).rounds||[]).forEach(function(r){if(fin(r.zero_var_ratio)&&fin(tg[r.round])){zg+=r.zero_var_ratio*tg[r.round];zt+=tg[r.round]}})});return [zg,zt]}
/* Samplers & pools: one row per sampler of the island, grouped by pool (pool, engine and the pool's mismatch
   metrics span its rows). A column no sampler reports is hidden and named under the table */
function spTable(ctx){
  var S=V.async.samplers||{},zv=zeroVarTrainer(),fl=!!ctx.pools;
  var groups=fl?ctx.pools.map(function(p){return {p:p,s:p.samplers.map(String)}}):ctx.sids.map(function(id){return {p:null,s:[id]}});
  if(!ctx.sids.length)return '<div class="nodata">no sampler on the tapes'+(fl?" for this island yet":"")+'</div>';
  var zvCell=function(x){if(fl&&fin(x.zero_variance_seen)&&!fin(x.zero_variance_dropped))return f(x.zero_variance_seen,0)+' / <span title="the pool keeps zero-variance groups (drop_zero_variance_groups off): they train with zero advantage">0</span>';
    return fin(x.zero_variance_seen)||fin(x.zero_variance_dropped)?f(x.zero_variance_seen,0)+" / "+f(x.zero_variance_dropped,0):
    zv[1]?'<span class="note" title="the sampler tape has no per-trial rewards; rl_local_round.zero_variance_group_ratio of the trainer">trainer: '+pct(zv[0]/zv[1])+' of '+f(zv[1],0)+'</span>':"n/a"};
  var COLS=[["policy","version the sampler generates with",function(x){return x.policy_version==null?"n/a":"v"+esc(x.policy_version)},function(x){return x.policy_version!=null}],
    ["age p50 / max","policy age of its rounds; hover for the histogram",function(x){return '<span title="'+esc("rounds by policy age: "+hist(x.lag_hist).replace(/&#39;/g,"'"))+'">'+f(x.lag_p50,1)+" / "+f(x.lag_max,0)+'</span>'},function(x){return fin(x.lag_p50)||fin(x.lag_max)}],
    ["rounds","",function(x){return f(x.n_rounds,0)},function(x){return fin(x.n_rounds)}],
    ["generate s p50 / max","",function(x){return f(x.generate_p50_s,0)+" / "+f(x.generate_max_s,0)},function(x){return fin(x.generate_p50_s)}],
    ["groups written","",function(x){return f(x.groups,0)},function(x){return fin(x.groups)}],
    ["zero-var seen / dropped","groups whose rewards were all equal",zvCell,function(x){return fin(x.zero_variance_seen)||fin(x.zero_variance_dropped)||zv[1]>0}],
    ["truncated","truncated samples (share of the samples it wrote)",function(x){return f(x.truncated,0)+(fin(x.samples)&&fin(x.truncated)&&x.samples?' <span class="note">'+pct(x.truncated/x.samples)+'</span>':"")},function(x){return fin(x.truncated)}],
    ["dropped groups","",function(x){return f(x.dropped_groups,0)},function(x){return fin(x.dropped_groups)}],
    ["trials retried / lost","",function(x){return f(x.retried_trials,0)+" / "+f(x.lost_trials,0)},function(x){return fin(x.retried_trials)||fin(x.lost_trials)}],
    ["reward","mean reward of its last round with groups",function(x){return f(x.reward_mean_last)},function(x){return fin(x.reward_mean_last)}],
    ["backlog cap","max_backlog_groups",function(x){return f(x.max_backlog_groups,0)},function(x){return fin(x.max_backlog_groups)}]];
  var cols=COLS.filter(function(c){return ctx.sids.some(function(id){return c[3](S[id]||{})})}),gone=COLS.filter(function(c){return cols.indexOf(c)<0});
  var mm=fl&&ctx.pools.some(function(p){var m=p.mismatch||{};return m.logprob_diff||m.tis_clip});
  var head='<tr><th class="l">sampler</th>'+(fl?'<th class="l">pool</th>':"")+'<th class="l">engine · cards</th><th class="l">status</th>'+cols.map(function(c){return '<th'+(c[1]?' title="'+esc(c[1])+'"':"")+'>'+esc(c[0])+'</th>'}).join("")+
    (mm?'<th title="train_metrics.train_rollout_logprob_abs_diff of the steps this pool fed">logprob |diff|</th><th title="train_metrics.tis_clipfrac (mismatch.tis_clipfrac) of the steps this pool fed">TIS clipfrac</th>':"")+'</tr>';
  var body=groups.map(function(g,gi){var p=g.p,n=g.s.length,rs=n>1?' rowspan="'+n+'"':"",mmv=p&&p.mismatch||{};
    if(!n)return '<tr'+(gi?' class="pg"':"")+'><td class="l note">none yet</td>'+(fl?'<td class="l">'+esc(p.name)+srcTag(p.source)+'</td>':"")+'<td class="l">'+esc(poolEngine(p))+'</td><td></td>'+cols.map(function(){return "<td></td>"}).join("")+(mm?"<td></td><td></td>":"")+'</tr>';
    return g.s.map(function(id,i){var x=S[id]||{},c=cardOf(id)||{};
      return '<tr'+(gi&&!i?' class="pg"':"")+'><td class="l">'+nameCell(id)+'</td>'+
        (i===0&&fl?'<td class="l"'+rs+'>'+esc(p.name)+(p.source_name&&p.source_name!==p.name&&String(p.name).slice(-String(p.source_name).length-1)!=="-"+p.source_name?' <span class="note">'+esc(p.source_name)+'</span>':"")+srcTag(p.source)+'</td>':"")+
        (i===0?'<td class="l"'+rs+platTitle(p,id)+'>'+esc(p?poolEngine(p):samplerEngine(id))+'</td>':"")+
        '<td class="l" title="'+esc(fin(c.last_event_age_s)?"last event "+f(c.last_event_age_s,0)+" s ago":"")+'">'+(c.status?stBadge(c.status,c):'<span class="note">not on the tapes</span>')+'</td>'+
        cols.map(function(cl){var v=cl[2](x);return '<td class="num">'+(fl?naText(v):v)+'</td>'}).join("")+
        (mm&&i===0?'<td class="num"'+rs+'>'+mmCell(mmv.logprob_diff)+'</td><td class="num"'+rs+'>'+mmCell(mmv.tis_clip)+'</td>':"")+'</tr>'}).join("")}).join("");
  var notes=[];
  if(gone.length)notes.push("hidden: "+gone.map(function(c){return c[0]}).join(", ")+" (no sampler of this island reports it)");
  if(fl&&!mm)notes.push("logprob |diff| / TIS clipfrac per pool: not reported (the trainer steps carry no per-pool mismatch split)");
  if(ctx.sids.some(function(id){var x=S[id]||{};return !fin(x.samples)}))notes.push(fl?"—: a sampler without rl_sampler_engine_round (a Miles native sampler) reports no zero-variance, truncated, dropped or reward counts":"a sampler without rl_sampler_engine_round (Miles-generating samplers, older yeto versions) has no truncated / reward counts; zero-variance then comes from the trainer");
  return YetoTbl.table(head,body)+(notes.length?'<div class="note">'+esc(notes.join(" · "))+'</div>':"")}
/* a fleet sampler cell: every "n/a" in its text (not inside a tag's attributes) is a dash */
function naText(h){return String(h).replace(/(<[^>]*>)|\bn\/a\b/g,function(m,tag){return tag||'<span class="note" title="not on this sampler\'s tape">—</span>'})}
/* trainer totals as one row of chips: zero and missing counts are named once at the end, not one chip each */
function trChips(T,trained){
  var it=[["trained groups",T.trained_groups,""],["dropped as stale",T.stale_groups,"older than max_policy_age"],["no policy identity",T.unidentified_policy_groups,"dropped after a resume"],
    ["off-policy gate dropped",T.gate_dropped,""],["short groups trained",T.incomplete_groups,""],["too small dropped",T.dropped_small_groups,""],["carried over",T.carried_over_groups,""],["padded samples",T.padded_samples,""]];
  var show=[],zero=[],na=[];it.forEach(function(x,i){if(!fin(x[1]))na.push(x[0]);else if(x[1]===0&&i>0)zero.push(x[0]);else show.push(x)});
  // a fleet island that never trained (its caller knows: no trainer round, no trained group): its missing counts and
  // policy ages say so once (a run with one trainer keeps n/a)
  var noAges=fleetOn()&&!Object.keys(T.age_hist||{}).length,never=noAges&&!trained,fold=never&&na.length>0;
  if(fold)na.splice(1,0,"policy ages");
  return '<div class="chips">'+show.map(function(x){return '<span'+(x[2]?' title="'+esc(x[2])+'"':"")+'>'+esc(x[0])+' <b>'+f(x[1],0)+'</b></span>'}).join("")+
    (T.unidentified_starved?'<span class="wn">reader starved <b>'+f(T.unidentified_starved.selections,0)+' selections kept no group</b></span>':"")+
    (fold?"":'<span title="policy ages of the trained groups, whole run (age:groups)">policy ages <b>'+(never?"not trained yet":noAges?"not on the tapes":histS(T.age_hist))+'</b></span>')+
    (zero.length?'<span>0: '+esc(zero.join(", "))+'</span>':"")+(na.length?'<span>'+(never?"not trained yet":fleetOn()?"not reported":"n/a")+': '+esc(na.join(", "))+'</span>':"")+'</div>'}
function asyncPanel(){
  var A=V.async,show=!!A&&!fleetView();$("async").hidden=!show;$("navAsync").style.display=A?"":"none";if(!A)return;
  if(!show){aComp(A,null);return}
  var ctx=asyncCtx(),tid=ctx.tid,tc=tid!=null?cardOf(tid)||{}:{},x=ctx.x,tags=samplerTags(ctx);
  // header: the island (fleet) or the trainer, its machine and lane, its status
  var meta=(ctx.fleet?[tid!=null?lbl(tid):"trainer not on the tapes yet",x.gpu,"R2 lane "+(x.substore||"trainer"+x.island)+"/",fin(x.release_every)?"release every "+x.release_every:null,fin(x.max_policy_age)?"max policy age "+x.max_policy_age:null]:
    [tid!=null?lbl(tid):"no trainer on the tapes yet",tc.cloud?tc.cloud+" "+(tc.gpu||"")+(fin(tc.gpus)?" × "+tc.gpus:""):null]).filter(Boolean);
  if(fin(tc.policy_version)||fin(tc.released_version))meta.push("policy "+pvTxt(tc.policy_version,tc.released_version));
  // what island health showed for the trainer: phase, hardware, cells / transactions, stop marker, Ray dashboard
  var ex=tid!=null?V.islands[tid]||{}:{},hw=ex.hardware,re=ex.ray_embed||{},ph=tid!=null&&tc.id!=null?islPhase(tc):null;
  if(ph)meta.push("phase "+ph);
  if(hw)meta.push("GPU type / driver / CUDA "+(hw.compat_group||"not declared")+" / "+(hw.driver_version||"not recorded")+" / "+(hw.cuda_version||"not recorded"));
  if(ex.cells||ex.transactions)meta.push("cells / transactions "+(ex.cells||0)+" / "+(ex.transactions||0));
  if(tc.stopped_by_us)meta.push("stopped by us ("+((O.operator_stop||{}).cause||"?")+", marker "+((O.operator_stop||{}).marker||"?")+")");
  if(!re.note&&(re.command||re.port))meta.push("Ray dashboard "+(re.command||"localhost :"+re.port));
  var su=Object.keys(tc.startup_steps||{}).map(function(k){var v=tc.startup_steps[k];return ({ray_connected:"Ray",engine_ready:"engine",weights_loaded:"weights"}[k]||k)+" "+(fin(v.step_s)?f(v.step_s,0)+" s":"?")});
  if(su.length)meta.push("startup: "+su.join(" · "));
  $("aHead").innerHTML=(ctx.fleet?"Island "+esc(x.island)+(x.name?" · "+esc(x.name):""):"Async GRPO")+' <span class="note">'+esc(meta.join(" · "))+'</span>';
  var nodes=(tc.nodes||[]).map(function(n){var pk=Math.max.apply(null,(n.gpu_mem_used_mib_peak||[0]).concat([0]));return "node "+n.node+": peak memory "+(pk/1024).toFixed(1)+" GiB, utilization "+(fin(n.gpu_util_pct)?f(n.gpu_util_pct,0)+"%":"not sampled")});
  if(re.note)nodes.push("Ray dashboard: "+re.note);if(!(ex.cells||ex.transactions)&&tid!=null)nodes.push("cells / transactions 0 / 0");
  $("aMeta").innerHTML=((tc.nodes||[]).length||nodes.length?'<span class="note" title="'+esc(nodes.join("\n"))+'">'+((tc.nodes||[]).length?(tc.nodes.length+" node"+(tc.nodes.length>1?"s":"")):"health details")+'</span>':"")+(tc.status?negBadge(tc)+stBadge(tc.status,tc):"")+(fin(tc.last_event_age_s)?'<span>last event '+esc(hm(tc.last_event_age_s))+' ago</span>':"");
  // trainer rounds
  var rr=((V.islands[tid]||{}).rounds||[]).filter(function(r){return fin(r.reward)||fin(r.kl_tr)||fin(r.kl)||fin(r.tok_s)}),ar={};
  (((A.trainers||{})[tid]||{}).rounds||[]).forEach(function(q){ar[q.rollout_id]=q});
  var T=(A.trainers||{})[tid]||{};
  $("aTrNote").textContent=tid!=null?lbl(tid):"";
  $("aTrainer").innerHTML=!rr.length?'<div class="nodata">no trained rounds on '+(tid!=null?esc(lbl(tid)):"this island")+' yet</div>'+(tid!=null?trChips(T,T.trained_groups>0):""):
    trChips(T,true)+trTable(rr,ar,"trainer-rounds"+ctx.key)+
    (rr.some(function(r){return r.resp_derived})?'<div class="note">* mean response length = trained response tokens ÷ trained samples (rl_rollout_carry_over), the tape has no batch summary</div>':"");
  // releases of this island
  var nRel=fin(ctx.relTotal)?ctx.relTotal:ctx.rel.length;
  $("aRelNote").textContent=ctx.fleet?(fin(x.latest)?"latest v"+x.latest+" · ":"")+nRel+" on this island":nRel?nRel+" in total":"";
  $("aRelTable").innerHTML=ctx.rel.length?relTable(ctx.rel,ctx.sids,tags,"releases"+ctx.key,ctx.relTotal):'<div class="nodata">no release yet</div>';
  $("aSamplers").innerHTML=spTable(ctx);
  blChart($("aBacklog"),$("legBacklog"),ctx.backlog);
  aRelChart(ctx,tags);aComp(A,ctx);
}
/* CompactionRL: per island totals and the samplers' recent rounds (hidden without compaction data) */
function segHist(h){var ks=Object.keys(h||{});return ks.length?ks.map(function(k){return esc(k)+" seg: "+f(h[k],0)}).join(" · "):"n/a"}
// without a single compaction the summary / after-compaction columns have nothing to measure: "—"
function compCells(c,short){var no=!(c.compacted_trials>0||c.compactions>0||c.summaries>0),NO='<span class="note" title="no trial was compacted">'+(short?"—":"— (no compaction)")+'</span>';
  return '<td class="num">'+f(c.trials,0)+'</td><td class="num">'+f(c.samples,0)+" ("+f(c.samples_per_trial,2)+'/trial)</td><td class="num">'+f(c.compacted_trials,0)+" ("+pct(c.compacted_frac)+')</td><td class="num">'+f(c.compactions_mean,2)+" (max "+f(c.compactions_max,0)+(fin(c.max_compactions)?" / limit "+f(c.max_compactions,0):"")+')</td><td>'+segHist(c.segments_hist)+
  '</td><td class="num">'+(no?NO:pct(c.summary_ok_frac)+" of "+f(c.summaries,0)+(fin(c.summary_truncated)&&c.summary_truncated>0?" · cut "+pct(c.summary_truncated_frac):""))+'</td><td class="num">'+(no?NO:f(c.truncated_compacted_trials,0)+" ("+pct(c.truncated_compacted_frac)+")"+(fin(c.exhausted_truncated_trials)?" · used up "+f(c.exhausted_truncated_trials,0):""))+'</td><td class="num">'+(no?NO:pct(c.summary_token_frac))+'</td><td class="num">'+(no?NO:f(c.fork_lost_trials,0)+" ("+pct(c.fork_lost_frac)+")")+'</td><td class="num">'+(c.retry_replays==null?(no?NO:"n/a"):f(c.retry_replays,0)+" in "+f(c.retry_replay_trials,0)+" trials")+'</td>'}
function reasons(o){var ks=Object.keys(o||{});return ks.length?ks.map(function(k){return esc(k)+" "+f(o[k],0)}).join(" · "):"0"}
// stale (trainer): a trainer row without stale_by_segments shows 0 when the trainer dropped no stale group
// at all (rl_buffer_stale_skipped.stale_groups); a sampler never drops stale groups itself ("—")
function fateRows(rows,A){var out=[];rows.forEach(function(x){var by=x[2].by_segments||{},ts=x[0]==="trainer"?(((A||{}).trainers||{})[x[1]]||{}).stale_groups:null;Object.keys(by).forEach(function(b){var v=by[b];
  var st=v.stale!=null?f(v.stale,0):x[0]!=="trainer"?"—":ts===0?"0":fin(ts)?'<span class="note">'+f(ts,0)+' in all (not by segment)</span>':"n/a";
  out.push('<tr><td>'+esc(x[0])+" "+esc(x[1])+'</td><td>'+esc(b)+'</td><td class="num">'+f(v.written,0)+'</td><td>'+reasons(v.lost)+'</td><td>'+reasons(v.dropped)+'</td><td class="num">'+st+'</td><td class="num">'+pct(v.not_written_frac)+'</td></tr>')})});return out}
var COMP_HEAD='<th>trials</th><th>samples</th><th>compacted trials</th><th>compactions / trial</th><th>segments per trial</th><th>summary_ok</th><th>truncated after compaction</th><th>summary tokens / trainable</th><th>lost to forks after compaction</th><th>retry replays</th>';
/* ctx: the picked island (a fleet island: only its trainer and samplers); null = not drawn */
function aComp(A,ctx){
  var rows=[],recent=[],inS=function(k){return !ctx||!ctx.fleet||ctx.sids.indexOf(String(k))>=0},inT=function(k){return !ctx||!ctx.fleet||String(k)===ctx.tid};
  if(ctx){Object.keys(A.samplers||{}).sort().forEach(function(k){var x=A.samplers[k];if(!inS(k))return;if(x.compaction)rows.push(["sampler",k,x.compaction]);
    (x.rounds||[]).forEach(function(r){if(r.compaction)recent.push([k,r])})});
  Object.keys(A.trainers||{}).sort().forEach(function(k){var x=A.trainers[k];if(x.compaction&&inT(k))rows.push(["trainer",k,x.compaction])})}
  $("aCompBox").hidden=!rows.length;if(!rows.length){$("aComp").innerHTML="";return}
  recent.sort(function(a,b){return (a[1].ts||0)-(b[1].ts||0)});
  var fr=fateRows(rows,A);
  $("aComp").innerHTML='<div class="tw"><table class="c"><tr><th>island</th><th>rounds</th>'+COMP_HEAD+'<th>reward mean per trial / per segment (last round)</th><th>xhw downweighted groups (max trial-mean shift)</th></tr>'+rows.map(function(x){var c=x[2];return '<tr><td>'+esc(x[0])+" "+esc(x[1])+'</td><td class="num">'+f(c.rounds,0)+'</td>'+compCells(c)+'<td class="num">'+f(c.reward_mean_trial)+" / "+f(c.reward_mean_segment)+'</td><td class="num">'+(c.xhw_downweighted_groups==null?"n/a":f(c.xhw_downweighted_groups,0)+" ("+f(c.xhw_trial_mean_shift_max,4)+")")+'</td></tr>'}).join("")+'</table></div>'+
    rows.filter(function(x){return x[2].drained_by_segments}).map(function(x){return '<div class="note">'+esc(x[0])+" "+esc(x[1])+': trials cut by drains, by segment count: '+reasons(x[2].drained_by_segments)+'</div>'}).join("")+
    (fr.length?'<div class="note">trials by segment count: written (trainer: trained), lost before the group was written (drain timeout, other version, aborted), dropped with their group, stale-dropped by the trainer; a much higher share for long trials biases the GRPO baseline to short ones</div><div class="tw"><table class="c"><tr><th>island</th><th>segments</th><th>written</th><th>lost</th><th>dropped with group</th><th>stale (trainer)</th><th>not written</th></tr>'+fr.join("")+'</table></div>':"")+
    (recent.length?'<div class="note">recent sampler rounds</div>'+(function(){var c=YetoTbl.cap("compaction-rounds",recent.slice().reverse(),{def:10,unit:"sampler rounds"});
      return c.bar+YetoTbl.table('<tr><th>sampler</th><th>round</th><th>time</th>'+COMP_HEAD+'</tr>',c.rows.map(function(x){return '<tr><td>'+esc(x[0])+'</td><td class="num">'+esc(x[1].rollout_id)+'</td><td class="num">'+clock(x[1].ts)+'</td>'+compCells(x[1].compaction,true)+'</tr>'}).join(""))})():"")+
    '<div class="note">alert cards fire on a low summary_ok share and on many compacted trials ending truncated (docs/DASHBOARD.md, compaction_* thresholds); summary tokens n/a = the samples do not report summary_tokens; — = no trial was compacted</div>';
}
function tAxis(s,t0,t1,sx,H){
  if(fleetOn()&&!(t1-t0>=1)){el("text",{x:sx(t0),y:H-6,"text-anchor":"start"},s).textContent=clock(t0);return}  // one instant (a fleet island's single release): one label, not three on top of each other
  [t0,(t0+t1)/2,t1].forEach(function(t,i){el("text",{x:sx(t),y:H-6,"text-anchor":["start","middle","end"][i]},s).textContent=clock(t)})}
function blChart(box,leg,b){
  box.innerHTML="";leg.innerHTML="";
  if(!b||!(b.points||[]).length){box.innerHTML='<div class="nodata">no data (no rl_sampler_round with groups)</div>';return}
  var pts=b.points,rep=b.reported||{},rk=Object.keys(rep),all=pts.map(function(p){return p[0]});rk.forEach(function(k){rep[k].forEach(function(p){all.push(p[0])})});
  var t0=Math.min.apply(null,all),t1=Math.max.apply(null,all),vm=Math.max.apply(null,pts.map(function(p){return fin(p[3])?p[3]:0}).concat([1]));
  rk.forEach(function(k){rep[k].forEach(function(p){vm=Math.max(vm,p[1])})});
  var W=boxW(box,560),H=210,m={l:44,r:16,t:10,b:22},sx=function(t){return m.l+(t-t0)*(W-m.l-m.r)/((t1-t0)||1)},sy=function(v){return m.t+(1-v/vm)*(H-m.t-m.b)};
  var s=el("svg",{viewBox:"0 0 "+W+" "+H,role:"img","aria-label":"Buffer backlog"},box),g=el("g",{"class":"grid"},s);
  [0,.5,1].forEach(function(q){el("line",{x1:m.l,x2:W-m.r,y1:sy(vm*q),y2:sy(vm*q)},g);el("text",{x:m.l-6,y:sy(vm*q)+4,"text-anchor":"end"},s).textContent=Math.round(vm*q)});
  var bp=pts.filter(function(p){return fin(p[3])});
  if(bp.length)el("path",{d:"M"+bp.map(function(p){return sx(p[0])+","+sy(p[3])}).join("L"),fill:"none",stroke:"var(--T)","stroke-width":2},s);
  // what each sampler reported about its own backlog: full (it stopped writing) or admitted (it resumed)
  var nf=0,na=0;rk.forEach(function(k){rep[k].forEach(function(p){if(p[2])nf++;else na++;
    el("title",{},el("circle",{cx:sx(p[0]),cy:sy(p[1]),r:3.5,fill:p[2]?"var(--bad)":"var(--good)"},s)).textContent=lbl(k)+": "+(p[2]?"backlog_full":"admitted")+" at "+clock(p[0])+" ("+f(p[1],0)+" groups)"})});
  tAxis(s,t0,t1,sx,H);
  leg.innerHTML='<span><i class="sw" style="background:var(--T)"></i>backlog (written '+f(b.written,0)+' · '+(fleetOn()&&!fin(b.read)?"not read yet":'read position '+f(b.read,0))+')</span>'+(nf?'<span><i class="sw" style="background:var(--bad);border-radius:50%"></i>sampler reported backlog_full ('+nf+')</span>':"")+(na?'<span><i class="sw" style="background:var(--good);border-radius:50%"></i>admitted ('+na+')</span>':"")+
    (b.has_reads&&b.read_source&&b.read_source!=="buffer_selected"?'<span>read position = '+esc(b.read_source)+' (no rl_policy_released.buffer_selected yet)</span>':"")+(b.has_reads?"":'<span>no read position on the tapes (no rl_policy_released.buffer_selected): backlog unknown</span>');
}
/* the task dispense ledger (rl_dispense_*): slots by state, claims still held and who holds them. A sampler hands
   back the slots it did not write as "release" events on every reload: counted, not listed */
function aDisp(A,cards,show){
  var D=A&&A.dispense,box=$("aDispBox");box.hidden=!D||!show;if(!D||!show)return;
  var st={};(cards||[]).forEach(function(c){st[c.id]=c.status});
  var dead=function(id){return ["stale","lost","left","stopped","done","recovery"].indexOf(st[id])>=0};
  var kv=function(o){var k=Object.keys(o||{});return k.length?k.map(function(x){return esc(x)+" "+o[x]}).join(" · "):"0"};
  var now=fin(O.now)?O.now:Date.now()/1e3,held=D.held||[],by={};held.forEach(function(h){(by[h.by]=by[h.by]||[]).push(h)});
  var evs=(D.events||[]).filter(function(e){return e.kind!=="drain"}),odd=evs.filter(function(e){return e.kind!=="release"}),nrel=evs.length-odd.length;
  var html='<div class="chips"><span>slots seen <b>'+D.slots+'</b></span><span>settled <b>'+kv(D.settled)+'</b></span><span title="expired = the lease ran out and the slot is claimed again; release = handed back unwritten">ended unsettled <b>'+kv(D.ends)+'</b></span><span>re-claimed slots <b>'+D.reclaimed_slots+'</b></span><span>held now <b>'+held.length+'</b></span></div>';
  html+=Object.keys(by).sort().map(function(k){var hs=by[k],bad=dead(k);
    return '<div class="note">'+(bad?'<span class="st bd"><i></i>'+esc(st[k])+'</span> ':"")+esc(lbl(k))+' holds '+hs.length+': '+hs.map(function(h){return "slot "+h.slot+(h.attempt?" (attempt "+h.attempt+")":"")+" claimed "+clock(h.claim_ts)+(fin(h.claim_ts)?" ("+f((now-h.claim_ts)/60,0)+" min ago)":"")}).join(", ")+
      (bad?' — <b>its island is '+esc(st[k])+': these slots wait for their lease to expire, then another sampler re-claims them</b>':"")+'</div>'}).join("");
  var ev=YetoTbl.cap("dispense-events",odd.slice().reverse(),{def:10,unit:"events"});
  html+='<div class="note">'+(odd.length?"":"no recover / expired event")+(nrel?(odd.length?"":" · ")+nrel+" unwritten slot"+(nrel===1?" was":"s were")+" handed back on a reload (release): counted above, not listed":"")+'</div>';
  var det=function(e){return [e.reason,e.state,e.superseded_by?"superseded by "+e.superseded_by:null,fin(e.seen)?"last renewed "+clock(e.seen):null].filter(Boolean).join(" · ")},hasDet=ev.rows.some(function(e){return det(e)});
  if(ev.rows.length)html+=ev.bar+YetoTbl.table('<tr><th>time</th><th class="l">event</th><th>slot</th><th>attempt</th><th class="l">noticed by</th>'+(hasDet?'<th class="l">detail</th>':"")+'</tr>',ev.rows.map(function(e){
    return '<tr><td class="num">'+clock(e.ts)+'</td><td class="l">'+esc(e.kind)+'</td><td class="num">'+(e.slot==null?"—":e.slot)+'</td><td class="num">'+(e.attempt==null?"—":e.attempt)+'</td><td class="l">'+esc(e.by?(cardOf(e.by)?lbl(e.by):e.by):"")+'</td>'+(hasDet?'<td class="l note wrap">'+esc(det(e))+'</td>':"")+'</tr>'}).join(""));
  $("aDisp").innerHTML=html;
}
/* NC pool events (nc_elastic_*): run-wide, a fleet shows them under "Fleet" */
function aPool(A,show){
  var pc=YetoTbl.cap("pool-events",((A&&A.pool)||[]).slice().reverse(),{def:10,unit:"pool events"}),pl=pc.rows;$("aPoolBox").hidden=!pl.length||!show;if(!pl.length||!show)return;
  var sc=A.pool_scale;$("aPoolScale").textContent=sc?("samplers: "+sc.ready.length+" ready"+(sc.draining.length?", "+sc.draining.length+" draining ("+sc.draining.join(", ")+")":"")+(sc.paused&&sc.paused.length?", "+sc.paused.length+" paused by hand ("+sc.paused.join(", ")+")":"")+(sc.target!=null?" · target "+sc.target+(sc.manual?" (set by hand)":""):"")+(sc.ready.length?" · shards "+sc.ready.join(", "):"")):"";
  $("aPool").innerHTML=pc.bar+pl.map(function(e){var bad=POOL_BAD.test(e.event||"");return '<div><span class="t">'+clock(e.ts)+'</span><span class="st '+(bad?"bd":"wn")+'"><i></i>'+(/^nc_elastic_manual_/.test(e.event||"")?"manual":"pool")+'</span><span class="x"><b>'+esc(String(e.event).replace(/^nc_elastic_/,""))+'</b> '+
    esc(["ticket","slot","shard","why","reason","detail","error","old","new","outcome","waited_s","source","by"].filter(function(k){return k=="slot"?(e.slot!=null||e.account!=null):e[k]!=null}).map(function(k){var v=e[k],n=k;if(k=="slot"){n=e.slot_name||"account";v=e.slot!=null?e.slot:e.account}return n+"="+String(v).slice(0,80)}).join(" "))+'</span></div>'}).join("");
}
/* release timeline of the picked island: the trainer's releases (version labels on one row, every step-th version
   with the smallest step 1/2/5/10/... whose labels keep apart, and the latest; every diamond keeps a tooltip) and
   each sampler's load done */
// the release labels to draw: {v, x} of every step-th version (steps 1, 2, 5, 10, 20, ...: the smallest whose labels
// stay 12 px apart on one row) and the latest (it replaces a label it would overlap)
function relLabels(pts,W){var GAP=12,wd=function(v){return 7.3*("v"+v).length};
  function fit(c){var out=[],right=-1e9;c.forEach(function(q){if(q.x-wd(q.v)/2>right+GAP&&q.x+wd(q.v)/2<=W-2){out.push(q);right=q.x+wd(q.v)/2}});return out}
  var vs=pts.filter(function(q){return fin(q.v)});if(!vs.length)return [];var last=vs[vs.length-1];
  for(var i=0,steps=[1,2,5];;i++){var st=steps[i%3]*Math.pow(10,Math.floor(i/3));
    var c=vs.filter(function(q){return q.v%st===0}),got=fit(c);
    if(got.length===c.length||st>1e6){var lw=wd(last.v)/2;got=got.filter(function(q){return q===last||q.x+wd(q.v)/2+GAP<last.x-lw});
      if(got.indexOf(last)<0&&last.x+lw<=W-2)got.push(last);return got}}}
function aRelChart(ctx,tags){
  var box=$("aRelChart"),rows=ctx.rel,sids=ctx.sids,A=V.async;box.innerHTML="";
  var pool=(A.pool||[]).filter(function(e){return fin(e.ts)&&(!ctx.fleet||sids.indexOf(String(e.shard))>=0)});
  var ts=[];rows.forEach(function(r){if(fin(r.released_at))ts.push(r.released_at);sids.forEach(function(k){var ld=(r.loads||{})[k];if(ld&&fin(ld.loaded_at))ts.push(ld.loaded_at)})});pool.forEach(function(e){ts.push(e.ts)});
  if(!ts.length){box.innerHTML='<div class="nodata">no data (no rl_policy_released / rl_sampler_policy_loaded'+(ctx.fleet?" on this island":" / nc_elastic_* events")+')</div>';$("legRel").innerHTML="";return}
  var W=boxW(box,560),long=function(k){return laneName(tags[k],k)},full=sids.map(long).concat(["trainer "+ctx.tid]);
  var CW=7.4,useLong=18+CW*Math.max.apply(null,full.map(function(n){return n.length}))<=W*0.36; // 12px mono
  var lanes=[ctx.tid!=null?lbl(ctx.tid):"trainer"].concat(sids.map(function(k){return useLong?long(k):tags[k]}));
  var t0=Math.min.apply(null,ts),t1=Math.max.apply(null,ts),rowH=30,ml=Math.min(Math.round(W*0.4),Math.max(64,18+CW*Math.max.apply(null,lanes.map(function(n){return n.length})))),
    m={l:ml,r:18,t:38,b:22},H=m.t+m.b+rowH*lanes.length,sx=function(t){return m.l+(t-t0)*(W-m.l-m.r)/((t1-t0)||1)},ly=function(i){return m.t+rowH*i+rowH/2-6};
  var s=el("svg",{viewBox:"0 0 "+W+" "+H,role:"img","aria-label":"Policy release timeline"},box);
  lanes.forEach(function(n,i){var tx=el("text",{x:m.l-10,y:ly(i)+4,"text-anchor":"end"},s);tx.textContent=n;if(i)el("title",{},tx).textContent=lbl(sids[i-1]);el("line",{x1:m.l,x2:W-m.r,y1:ly(i),y2:ly(i),stroke:"var(--line)"},s)});
  pool.forEach(function(e){var x=sx(e.ts),bad=POOL_BAD.test(e.event||"");el("line",{x1:x,x2:x,y1:m.t-6,y2:H-m.b,stroke:bad?"var(--bad)":"var(--warn)","stroke-dasharray":"3 3"},s);
    el("title",{},el("rect",{x:x-3,y:m.t,width:6,height:H-m.t-m.b,fill:"transparent"},s)).textContent=e.event+" "+clock(e.ts)});
  var rel=rows.filter(function(r){return fin(r.released_at)}).slice().sort(function(a,b){return a.released_at-b.released_at});
  var labs=relLabels(rel.map(function(r){return {v:r.policy_version,x:sx(r.released_at)}}),W);
  labs.forEach(function(q){var t=el("text",{x:q.x,y:ly(0)-11,"text-anchor":"middle"},s);t.textContent="v"+q.v});
  rel.forEach(function(r){var x0=sx(r.released_at);
    el("title",{},el("rect",{x:x0-4,y:ly(0)-4,width:8,height:8,fill:"var(--P)",transform:"rotate(45 "+x0+" "+ly(0)+")"},s)).textContent="v"+r.policy_version+" released "+clock(r.released_at);
    sids.forEach(function(k,i){var ld=(r.loads||{})[k];if(!ld||!fin(ld.loaded_at))return;var x1=sx(ld.loaded_at);
      el("line",{x1:x0,y1:ly(0),x2:x1,y2:ly(i+1),stroke:"var(--P)","stroke-opacity":.45},s);
      el("title",{},el("circle",{cx:x1,cy:ly(i+1),r:4,fill:"var(--P)"},s)).textContent=lbl(k)+" loaded v"+r.policy_version+" at "+clock(ld.loaded_at)+(fin(ld.latency_s)?" ("+f(ld.latency_s,0)+" s)":"")})});
  // a sampler that started on a version (no release on the chart): a hollow dot at its load
  rows.forEach(function(r){sids.forEach(function(k,i){var ld=(r.loads||{})[k];if(ld&&ld.at_start&&fin(ld.loaded_at)&&!fin(r.released_at))
    el("title",{},el("circle",{cx:sx(ld.loaded_at),cy:ly(i+1),r:4,fill:"var(--panel)",stroke:"var(--P)","stroke-width":1.5},s)).textContent=lbl(k)+" started on v"+r.policy_version+" at "+clock(ld.loaded_at)})});
  tAxis(s,t0,t1,sx,H);
  $("legRel").innerHTML='<span><i class="sw" style="background:var(--P)"></i>release → load done</span>'+(rows.some(function(r){return !fin(r.released_at)})?'<span><i class="sw" style="background:transparent;border:1.5px solid var(--P);border-radius:50%"></i>started on the version</span>':"")+
    (pool.length?'<span><i class="sw" style="background:var(--warn)"></i>NC pool event</span><span><i class="sw" style="background:var(--bad)"></i>NC pool failure (evicted / lost / crash loop)</span>':"");
}
/* "Fleet": one row per island of the fleet (the summary each island block opens with); a setting the island
   does not override and no tape names is the fleet's shared trainer flag */
function fleetUnset(x){return x.plan?'<span class="note" title="the fleet plan sets no per-island value: the shared trainer argv value">shared</span>':
  '<span class="note" title="no fleet plan and no tape names it">—</span>'}
function fleetBox(){
  var on=fleetView();$("fleetBox").hidden=!on;if(!on)return;
  var I=V.async.islands||[];
  $("fleetTbl").innerHTML=YetoTbl.table('<tr><th class="l">island</th><th class="l">trainer</th><th class="l">pools</th><th>latest release</th><th>backlog</th><th>trained groups</th><th>stale dropped</th><th title="policy age of the trained groups, last round">policy age (last round)</th><th>release every</th><th>max policy age</th></tr>',
    I.map(function(x){var b=x.backlog||{},pts=b.points||[],last=pts.length?pts[pts.length-1]:null,pa=x.policy_age||{},t=x.trainer!=null?String(x.trainer):null,tc=t!=null?cardOf(t):null;
      return '<tr><td class="l"><b>Island '+esc(x.island)+'</b>'+(x.name?' <span class="note">'+esc(x.name)+'</span>':"")+'</td><td class="l">'+(tc?nameCell(t)+" "+stBadge(tc.status,tc):'<span class="note">not on the tapes yet</span>')+'</td><td class="l">'+
        x.pools.map(function(p){return esc(p.name)+' <span class="note">'+p.samplers.length+'</span>'}).join(", ")+'</td><td class="num">'+(fin(x.latest)?"v"+x.latest:"none")+'</td><td class="num">'+(last&&fin(last[3])?f(last[3],0):pts.length?'<span class="note" title="its trainer has not read the lane yet">'+(fin(b.written)?f(b.written,0)+' written, ':'')+'not read yet</span>':'<span class="note">no rounds</span>')+'</td><td class="num">'+f(x.trained_groups,0)+
        '</td><td class="num">'+f(pa.stale_groups,0)+'</td><td class="num">'+(fin(pa.mean_last)?"mean "+f(pa.mean_last,2)+" · max "+f(pa.max_last,0):'<span class="note">no trained round</span>')+'</td><td class="num">'+(fin(x.release_every)?x.release_every:fleetUnset(x))+'</td><td class="num">'+(fin(x.max_policy_age)?x.max_policy_age:fleetUnset(x))+'</td></tr>'}).join(""));
}
/* the run-wide ledgers: two to a row, one alone takes the row */
function runWide(){
  var A=V.async,show=!fleetOn()||fleetView();
  syncerPanel(show);aDisp(A,O.islands||[],show);aPool(A,show);
  var n=["syncerBox","aDispBox","aPoolBox"].filter(function(id){return !$(id).hidden}).length;
  $("runWide").className="dg2"+(n<2?" one":"");$("runWide").hidden=!n;
}
/* the per-round drill-down follows the picked island; "Fleet" keeps only the trainers' metric chart */
function roundsPanel(){var fv=fleetView();
  $("rDurBox").hidden=fv;$("rNodesBox").hidden=fv;$("rTlBox").hidden=fv;$("rMetricBox").className=fv?"full":"";
  $("roundsIsl").textContent=fv?"· every trainer of the fleet":SEL!=null&&(O.islands||[]).length>1?"· "+lbl(SEL):"";
  chartMetric();if(!fv){chartDur();chartTl();chartNodes()}highlight()}
function render(){
  if(!V)return;O=V.overview;FOLDS=[];
  picker();
  RS=SEL!=null?((V.islands[SEL]||{}).rounds||[]):[];
  header();kpis();fleetBox();asyncPanel();roundsPanel();util(!fleetView());islands();runWide();cost();events();cmds();folds();spot();
  if(typeof YetoExt!=="undefined")YetoExt.render(V,OFFLINE);
  if(typeof YetoTopo!=="undefined")YetoTopo.render(V,OFFLINE);
  if(typeof YetoBrk!=="undefined")YetoBrk.render(V);
  if(typeof YetoTbl!=="undefined"&&YetoTbl.hint)YetoTbl.hint();
}
// the server runs newer page code than this tab (redeployed): reload once per build, else say so
function stale(b){if(!b||typeof YETO_BUILD!=="string"||b===YETO_BUILD)return;var done=null;
  try{done=sessionStorage.getItem("yeto-build-reload")}catch(e){}
  if(done!==b){try{sessionStorage.setItem("yeto-build-reload",b)}catch(e){}location.reload();return}
  var t=document.title;if(t.indexOf("[reload]")<0)document.title="[reload] "+t}
function load(){
  if(OFFLINE){V=INLINE.page;EVENTS=(INLINE.events||{}).events||[];render();return}
  Promise.all([fetch("/api/view",{cache:"no-store"}).then(function(r){stale(r.headers&&r.headers.get&&r.headers.get("X-Yeto-Build"));return r.json()}),
    fetch("/api/events?after="+EV_CURSOR+"&limit=1000",{cache:"no-store"}).then(function(r){return r.json()})]).then(function(x){
    V=x[0];EVENTS=EVENTS.concat(x[1].events||[]).slice(-3000);EV_CURSOR=x[1].cursor||EV_CURSOR;render()}).catch(function(e){$("sub").textContent="fetch failed: "+e})
}
if(typeof YetoExt!=="undefined")YetoExt.setReload(load);
if(typeof YetoTbl!=="undefined")YetoTbl.setRender(render); // "+50" / "all" under a long table: draw again from the data in hand
var resizeT=null;if(typeof window!=="undefined"&&window.addEventListener)window.addEventListener("resize",function(){clearTimeout(resizeT);resizeT=setTimeout(render,250)}); // charts drawn at the box width
load();if(!OFFLINE)setInterval(load,POLL_MS);
})();
