// yeto dashboard (section 9 redesign). No build step, no external requests.
// Live: polls /api/view every 5 s. Offline export: reads the inlined full view (its "page" key).
// Every number comes from the reducer; anything missing is shown as "n/a" (never interpolated).
(function(){
"use strict";
var node=document.getElementById("yeto-data"),INLINE=null;
if(node){try{INLINE=JSON.parse(node.textContent)}catch(e){INLINE=null}}
var OFFLINE=!!INLINE,POLL_MS=5000,NS="http:"+"//www.w3.org/2000/svg"; // SVG namespace id, not a request
var V=null,O=null,RS=[],SEL=null,cur=null,layout="auto",metric="reward",themeI=2,pinned=false;
var PH=[["R","rollout"],["T","training"],["S","post-train sync"],["P","publish"]];
var ISL_COLORS=["var(--i0)","var(--i1)","var(--i2)"];
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
/* short island label (s0 / s1 / trainer 0) with the full id as tooltip: shard ids are long run names */
function islShort(c){var id=String(c.id),m=/-(\d+)$/.exec(id),async=((V||{}).async||{}).samplers;
  if(c.role==="sampler")return m?"s"+m[1]:id;if(c.role==="trainer"&&async&&Object.keys(async).length)return "trainer "+(m?m[1]:id);return m&&id.length>12?"island "+m[1]:"Island "+id}
function islName(c){return '<span title="'+esc(String(c.id)+(c.name?" ("+c.name+")":""))+'">'+esc(islShort(c))+'</span>'}
function el(tag,a,parent){var e=document.createElementNS(NS,tag);for(var k in a)e.setAttribute(k,a[k]);
  /* a fixed viewBox stretched to a full-width panel scales its text 3-4x: cap the drawn width at 1.4x the viewBox */
  if(tag==="svg"&&a&&a.viewBox){var vw=+String(a.viewBox).split(" ")[2];if(vw>0)e.style.maxWidth=Math.round(vw*1.4)+"px"}
  if(parent)parent.appendChild(e);return e}
/* draw at the chart's pixel width (1 viewBox unit = 1 px, so 11px text stays 11px); def when not laid out.
   Measured on the parent: an emptied chart box is display:none (#cTl:empty ...) while it is redrawn */
function boxW(box,def){var p=box&&box.parentNode,w=p&&p.clientWidth;return w>=240?Math.floor(w):def}
// launch-preflight-guards 3.5: negative-test island badge (rl_island_override)
function negBadge(c){return c&&c.negative_test?' <span class="st wn" title="'+esc(c.negative_test.label)+'"><i></i>'+esc(c.negative_test.label)+'</span> ':""}
function islColor(id){var ids=(O.islands||[]).map(function(c){return c.id}),i=ids.indexOf(id);return i>=0&&i<3?ISL_COLORS[i]:"var(--i3)"}
function kind(){return layout==="auto"?O.run_kind:layout}
/* why an island left: "left: evicted (preempted) at 05:09, replaced by <ticket>" */
function leftText(s,l){return (s==="lost"?"lost":"left")+": "+(l.reason||"?")+(fin(l.ts)?" at "+clock(l.ts).slice(0,5):"")+(l.replaced_by?", replaced by "+l.replaced_by:"")}
function stBadge(s,c){var m={ok:["ok","healthy"],starting:["wn","starting"],stopped:["mu","stopped"],stale:["bd","stale"],waiting:["wn","waiting"],lost:["bd","lost"],
  left:["mu","left the run"],recovery:["bd","needs recovery"],done:["mu","finished"],unknown:["mu","no data"]}[s]||["mu",s];var t=m[1];
  if(c){if(s==="starting"&&c.startup_slow)t="starting (slow)";if(s==="waiting"&&c.waiting_reason)t="waiting: "+c.waiting_reason;if((s==="left"||s==="lost")&&c.left)t=leftText(s,c.left)}
  return '<span class="st '+m[0]+'" title="'+esc(c&&c.role?c.role:"")+'"><i></i>'+esc(t)+'</span>'}

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
  var al=O.alerts||[];$("alertBox").hidden=!al.length;
  $("alertBox").innerHTML="<h2>Alerts</h2>"+al.map(function(a){return '<div><span class="st '+(a.sev===0?"bd":a.sev===1?"wn":"mu")+'"><i></i>'+["critical","warning","info"][a.sev]+'</span> <b>'+esc(a.title)+'</b> <span class="note">'+esc(a.detail)+'</span></div>'}).join("");
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
function wall(){
  var multi=kind()==="multi_island";$("islandWall").hidden=!multi;$("navWall").style.display=multi?"":"none";
  $("syncerBox").hidden=!(multi&&V.usage.syncer);
  if(multi)$("wall").innerHTML=(O.islands||[]).map(function(c){return '<div class="card'+(c.id===SEL?" sel":"")+'" data-id="'+esc(c.id)+'" tabindex="0">'+
    '<div style="display:flex;justify-content:space-between"><b><i class="sw" style="background:'+islColor(c.id)+'"></i>'+islName(c)+'</b>'+negBadge(c)+stBadge(c.status,c)+'</div>'+
    '<div class="note mono">'+esc(platform(c))+' · '+esc(c.gpu||"")+' × '+esc(c.gpus==null?"n/a":c.gpus)+'</div>'+
    '<div class="kv"><span>policy version</span><span class="num">'+esc(pvTxt(c.policy_version,c.released_version))+'</span><span>reward</span><span class="num">'+f(islReward(c))+'</span><span>phase</span><span class="num">'+esc(islPhase(c)||"n/a")+'</span><span>role</span><span class="num">'+esc(c.role||"n/a")+'</span></div></div>'}).join("");
  Array.prototype.forEach.call(document.querySelectorAll(".card"),function(d){d.onclick=d.onkeydown=function(e){if(e.type==="keydown"&&e.key!=="Enter")return;SEL=d.dataset.id;cur=null;render()}});
  if(multi&&V.usage.syncer){var rows=(V.rounds_syncer||[]).slice(-30).reverse();
    $("syncer").innerHTML='<table class="c"><tr><th>global round</th><th>expected/responded</th><th>missed</th><th>quorum ms</th><th>merge ms</th><th>resends</th></tr>'+rows.map(function(r){
      return '<tr><td class="num">'+esc(r.round)+'</td><td class="num">'+f(r.expected,0)+' / '+f(r.responded,0)+'</td><td>'+(r.missed.length?'<span class="bd">'+esc(r.missed.join(","))+'</span>':"n/a")+'</td><td class="num">'+f(r.quorum_ms,0)+'</td><td class="num">'+f(r.merge_ms,0)+'</td><td class="num">'+esc(r.resend)+'</td></tr>'}).join("")+'</table>'}
}
/* chart 1: metric per round; reward gets the p10-p90 band; < 5 points: dots only, no line */
var xOf=null;
var metricPicked=false;
function hasMetric(k,ids){return ids.some(function(id){return k==="trunc"||k==="logprob_diff"?((V.islands[id]||{}).rounds||[]).some(function(r){return fin(r[k])}):(((O.series||{})[id]||{})[k]||[]).length})}
function chartMetric(){
  var ids0=kind()==="multi_island"?(O.islands||[]).map(function(c){return c.id}):[SEL];
  if(!metricPicked&&!hasMetric(metric,ids0)){var alt=METRICS.filter(function(m){return hasMetric(m[0],ids0)})[0];if(alt)metric=alt[0]}
  $("tabs").innerHTML=METRICS.map(function(m){return '<button class="btn'+(m[0]===metric?" on":"")+'" data-m="'+m[0]+'">'+m[1]+'</button>'}).join("");
  Array.prototype.forEach.call($("tabs").querySelectorAll("button"),function(b){b.onclick=function(){metric=b.dataset.m;metricPicked=true;chartMetric();highlight()}});
  var multi=kind()==="multi_island",ids=multi?(O.islands||[]).map(function(c){return c.id}):[SEL];
  var series=ids.map(function(id){return {id:id,pts:((O.series||{})[id]||{})[metric==="trunc"||metric==="logprob_diff"?"_":metric]||[]}});
  if(metric==="trunc"||metric==="logprob_diff"){series=ids.map(function(id){var rr=(V.islands[id]||{}).rounds||[];return {id:id,pts:rr.filter(function(r){return fin(r[metric])}).map(function(r){return [r.round+1,r[metric]]})}})}
  var all=[];series.forEach(function(s){s.pts.forEach(function(p){all.push(p)})});
  var band=metric==="reward"&&!multi?RS.filter(function(r){return fin(r.p10)&&fin(r.p90)}):[];
  var box=$("cMetric");box.innerHTML="";var W=boxW(box,560),H=250,m={l:52,r:14,t:12,b:26};
  $("legMetric").innerHTML=(multi?ids.map(function(id){return '<span><i class="sw" style="background:'+islColor(id)+'"></i>Island '+esc(id)+'</span>'}).join(""):'<span><i class="sw" style="background:var(--T)"></i>'+esc(METRICS.find(function(x){return x[0]===metric})[1])+'</span>')+
    (band.length?'<span><i class="sw" style="background:var(--band)"></i>p10–p90</span>':"");
  if(!all.length){box.innerHTML='<div class="note">no data</div>';xOf=null;return}
  var xs=all.map(function(p){return p[0]}),x0=Math.min.apply(null,xs),x1=Math.max.apply(null,xs);
  var vs=all.map(function(p){return p[1]});band.forEach(function(r){vs.push(r.p10,r.p90)});
  var lo=Math.min.apply(null,vs),hi=Math.max.apply(null,vs);if(metric==="reward"||metric==="trunc"){lo=Math.min(lo,0);hi=Math.max(hi,1)}if(hi===lo){hi+=1;lo-=1}
  var x=function(r){return m.l+(x1===x0?(W-m.l-m.r)/2:(r-x0)*(W-m.l-m.r)/(x1-x0))},y=function(v){return m.t+(hi-v)/(hi-lo)*(H-m.t-m.b)};
  xOf=function(round){return x(round+1)};
  var s=el("svg",{viewBox:"0 0 "+W+" "+H,role:"img","aria-label":"Per-round metrics"},box),g=el("g",{"class":"grid"},s);
  for(var j=0;j<=4;j++){var v=lo+(hi-lo)*j/4;el("line",{x1:m.l,x2:W-m.r,y1:y(v),y2:y(v)},g);el("text",{x:m.l-6,y:y(v)+4,"text-anchor":"end"},s).textContent=+v.toPrecision(3)}
  var ticks={};xs.forEach(function(r){ticks[r]=1});var tk=Object.keys(ticks).map(Number),every=Math.ceil(tk.length/10);
  tk.forEach(function(r,i){if(i%every===0)el("text",{x:x(r),y:H-8,"text-anchor":"middle"},s).textContent="round "+(r-1)});
  var connect=series.some(function(sr){return sr.pts.length>=5});
  if(band.length>=2&&connect)el("path",{d:"M"+band.map(function(r){return x(r.round+1)+","+y(r.p90)}).join("L")+"L"+band.slice().reverse().map(function(r){return x(r.round+1)+","+y(r.p10)}).join("L")+"Z",fill:"var(--band)"},s);
  else band.forEach(function(r){el("line",{x1:x(r.round+1),x2:x(r.round+1),y1:y(r.p10),y2:y(r.p90),stroke:"var(--band)","stroke-width":10,"stroke-linecap":"round"},s)});
  series.forEach(function(sr){var col=multi?islColor(sr.id):"var(--T)";
    if(sr.pts.length>=5)el("path",{d:"M"+sr.pts.map(function(p){return x(p[0])+","+y(p[1])}).join("L"),fill:"none",stroke:col,"stroke-width":2},s);
    sr.pts.forEach(function(p){el("circle",{cx:x(p[0]),cy:y(p[1]),r:sr.pts.length>60?2.5:5,fill:col,stroke:"var(--panel)","stroke-width":2,"class":"rdot","data-r":p[0]-1},s)})});
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
  var box=$("cDur");box.innerHTML="";var rows=RS.slice(-20);
  if(!rows.some(function(r){return Object.keys(r.dur).length})){box.innerHTML='<div class="note">no phase data (no rl_timeline_span in the tapes)</div>';return}
  var W=boxW(box,420),rowH=30,m={l:64,r:60,t:4,b:4},H=m.t+m.b+rowH*rows.length;
  var s=el("svg",{viewBox:"0 0 "+W+" "+H,role:"img","aria-label":"Per-round time breakdown"},box);
  var tot=rows.map(function(r){return PH.reduce(function(a,p){return a+(r.dur[p[0]]||0)},0)}),mx=Math.max.apply(null,tot.concat([1]));
  rows.forEach(function(r,i){var y0=m.t+i*rowH,x0=m.l;
    el("rect",{x:0,y:y0,width:W,height:rowH,fill:"transparent","class":"drow","data-r":r.round},s);
    el("text",{x:m.l-8,y:y0+rowH/2+4,"text-anchor":"end"},s).textContent="round "+r.round;
    PH.forEach(function(p){var w=(r.dur[p[0]]||0)*(W-m.l-m.r)/mx;if(w>0){el("rect",{x:x0,y:y0+5,width:Math.max(w-2,1),height:rowH-10,rx:3,fill:"var(--"+p[0]+")"},s);x0+=w}});
    el("text",{x:x0+6,y:y0+rowH/2+4},s).textContent=(tot[i]/60).toFixed(1)+"m";
    var hit=el("rect",{x:0,y:y0,width:W,height:rowH,fill:"transparent"},s);
    hit.addEventListener("mousemove",function(e){setCur(r.round,e)});hit.addEventListener("mouseleave",hideTip);hit.addEventListener("click",function(e){pinned=!pinned;setCur(r.round,e)})});
}
/* chart 3: phase timeline on the wall clock, startup segment first */
function chartTl(){
  var ex=V.islands[SEL]||{},box=$("cTl");box.innerHTML="";
  $("legTl").innerHTML=PH.map(function(p){return '<span><i class="sw" style="background:var(--'+p[0]+')"></i>'+p[1]+'</span>'}).join("")+'<span><i class="sw" style="background:var(--line)"></i>startup (load weights, form Ray cluster, start inference engine)</span>';
  var segs=[];RS.forEach(function(r){Object.keys(r.phases).forEach(function(k){segs.push([k,r.phases[k][0],r.phases[k][1],r.round])})});
  if(!segs.length){box.innerHTML='<div class="note">no phase data</div>';return}
  var t0=Math.min.apply(null,segs.map(function(q){return q[1]}).concat(fin(ex.first_ts)?[ex.first_ts]:[])),t1=Math.max.apply(null,segs.map(function(q){return q[2]}));
  var W=boxW(box,1000),LH=30,H=4+2*LH+24,m={l:70,r:16,t:4,b:24},sx=function(t){return m.l+(t-t0)*(W-m.l-m.r)/((t1-t0)||1)};
  var s=el("svg",{viewBox:"0 0 "+W+" "+H,role:"img","aria-label":"Phase timeline"},box);
  el("text",{x:m.l-8,y:m.t+LH/2+3,"text-anchor":"end"},s).textContent="rollout";el("text",{x:m.l-8,y:m.t+LH*1.5+3,"text-anchor":"end"},s).textContent="training";
  if(fin(ex.first_ts)&&fin(ex.driver_start_ts))el("rect",{x:sx(ex.first_ts),y:m.t,width:Math.max(sx(ex.driver_start_ts)-sx(ex.first_ts),1),height:2*LH-4,fill:"var(--line)",rx:3},s);
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
  // async buffer mode: samplers keep their own round numbers and never take part in a trainer round,
  // so a sampler row says what the trainer took from it this round (no "missed"); nothing taken, no row
  var asy=Object.keys((V.async||{}).samplers||{}).length>0;
  if(kind()==="multi_island"){return h+(O.islands||[]).map(function(c){var sw='<i class="sw" style="background:'+islColor(c.id)+'"></i>';
    if(asy&&c.role==="sampler"){var b=(r.by_sampler||{})[c.id];if(!b)return "";var vs=b.versions||[];
      return row(sw+"sampler "+esc(c.id),(b.groups!=null?f(b.groups,0)+" groups":f(b.samples,0)+" samples")+(vs.length?" · v"+vs[0]+(vs.length>1?"–v"+vs[vs.length-1]:""):"")+
        (fin(b.age_mean)?" (age "+f(b.age_mean,1)+")":"")+(fin(b.reward_mean)?" · reward "+f(b.reward_mean,3):""))}
    var rr=((V.islands[c.id]||{}).rounds||[]).find(function(x){return x.round===r.round});
    if(!rr&&asy)return "";
    return row(sw+"Island "+esc(c.id),rr?f(rr.reward)+" · truncated "+pct(rr.trunc):"missed")}).join("")}
  return h+row("reward mean",f(r.reward))+row("train-rollout KL / TIS clipfrac",f(r.kl_tr,5)+" / "+f(r.tis_clip,4))+row("p10 / p50 / p90",f(r.p10,2)+" / "+f(r.p50,2)+" / "+f(r.p90,2))+row("truncation",pct(r.trunc))+
    row("logprob diff",f(r.logprob_diff,4))+row("response length mean / p95",f(r.resp_mean,0)+" / "+f(r.resp_p95,0))+row("grad_norm",f(r.grad_norm,4))+
    PH.map(function(p){return row('<i class="sw" style="background:var(--'+p[0]+')"></i>'+p[1],fin(r.dur[p[0]])?f(r.dur[p[0]],0)+" s":"n/a")}).join("")}
function showTip(e){var r=RS.find(function(x){return x.round===cur});if(!r)return;var t=$("tip"),p=$("rounds").getBoundingClientRect();
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
function util(){
  var runs=utilRuns(),has=runs.some(function(r){return r.util});
  $("util").hidden=false;$("utilEmpty").hidden=has;$("utilBody").hidden=!has;
  $("utilIsl").textContent=SEL!=null&&(O.islands||[]).length>1?"· island "+SEL:"";
  if(!has){$("uCarryBox").hidden=true;return}
  var rids={};runs.forEach(function(r){((r.util||{}).rounds||[]).forEach(function(x){rids[x.round]=1})});
  var ids=Object.keys(rids).map(Number).sort(function(a,b){return a-b});
  if(uRound!=="all"&&ids.indexOf(uRound)<0)uRound="all";
  var btns=[["all","all rounds overlaid"]].concat(ids.map(function(i){return [i,"round "+i]}));
  var ctl=btns.map(function(b){return '<button class="btn'+(b[0]===uRound?" on":"")+'" data-u="'+b[0]+'">'+b[1]+'</button>'}).join("");
  if(runs.length>1)ctl+=' <button class="btn'+(uMode==="side"?" on":"")+'" data-mode="side">side by side</button><button class="btn'+(uMode==="overlay"?" on":"")+'" data-mode="overlay">overlay</button>';
  $("utilCtl").innerHTML=ctl;
  Array.prototype.forEach.call($("utilCtl").querySelectorAll("button"),function(b){b.onclick=function(){
    if(b.dataset.mode)uMode=b.dataset.mode;else uRound=b.dataset.u==="all"?"all":+b.dataset.u;util()}});
  uCut(runs);uDone(runs);uPh(runs);uTr(runs);uLoad(runs);uCarry(runs);
}
function uCut(runs){
  $("uCut").innerHTML=runs.map(function(r,i){var rr=((r.util||{}).rounds||[]);
    if(!rr.some(function(x){return x.cutoff}))return '<div>'+runHead(r,i,runs.length)+'<div class="note">no data (no rl_rollout_cutoff)</div></div>';
    return '<div>'+runHead(r,i,runs.length)+'<table class="c"><tr><th>round</th><th>generation s</th><th>submitted groups</th><th>target groups</th><th>dropped groups</th><th>dropped trajectories</th><th>dropped tokens</th><th>filtered groups</th></tr>'+
      rr.map(function(x){var c=x.cutoff||{};return '<tr'+(x.round===uRound?' class="hl"':"")+'><td class="num">'+x.round+'</td><td class="num">'+unk(x.gen_s)+'</td><td class="num">'+unk(c.submitted_groups)+'</td><td class="num">'+unk(c.target_groups)+
        '</td><td class="num">'+unk(c.discarded_groups)+'</td><td class="num">'+unk(c.discarded_trajectories)+'</td><td class="num">'+unk(c.discarded_tokens)+'</td><td class="num">'+unk(c.filtered_groups)+'</td></tr>'}).join("")+'</table></div>'}).join("");
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
    div.innerHTML=runHead(r,i,runs.length);
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
    div.innerHTML=runHead(r,i,runs.length);
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
  $("uCarry").innerHTML=runs.map(function(r,i){var c=(r.util||{}).carry||[];
    return '<div>'+runHead(r,i,runs.length)+(c.length?'<table class="c"><tr><th>trajectory</th><th>task</th><th>start round → training round</th><th>version spans</th></tr>'+c.slice(-50).map(function(x){
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
  var W=boxW(box,560),H=210,m={l:44,r:14,t:8,b:22},sx=function(t){return m.l+(t-t0)*(W-m.l-m.r)/((t1-t0)||1)},sy=function(v){return m.t+(1-v/vm)*(H-m.t-m.b)};
  var s=el("svg",{viewBox:"0 0 "+W+" "+H,role:"img","aria-label":"GPU memory per node"},box),g=el("g",{"class":"grid"},s);
  [0,.5,1].forEach(function(q){el("line",{x1:m.l,x2:W-m.r,y1:sy(vm*q),y2:sy(vm*q)},g);el("text",{x:m.l-6,y:sy(vm*q)+4,"text-anchor":"end"},s).textContent=(vm*q).toFixed(0)});
  // no line across a gap (3x the usual spacing): an island down or restarted draws two pieces, not a diagonal
  ks.forEach(function(k){var P=ns[k].slice().sort(function(a,b){return a[0]-b[0]});
    el("path",{d:YetoExt.segments(P,P).map(function(G){return "M"+G.map(function(p){return sx(p[0])+","+sy(p[1])}).join("L")}).join(""),fill:"none",stroke:col(k),"stroke-width":2},s)});
  [t0,(t0+t1)/2,t1].forEach(function(t){el("text",{x:sx(t),y:H-6,"text-anchor":"middle"},s).textContent=clock(t)});
  var hasU=ks.some(function(k){return ns[k].some(function(p){return fin(p[2])})});
  $("nodesNote").textContent="Source: per-node host probe or the island's NVML samples (rl_resource_sample)."+(hasU?"":" GPU utilization: not sampled in this tape (older probes record memory only).");
}
function islands(){
  $("islands").innerHTML=(O.islands||[]).map(function(c){var ex=V.islands[c.id]||{},re=ex.ray_embed||{};
    return '<details class="isl"'+(c.id===SEL?" open":"")+'><summary><span><b>'+islName(c)+'</b> <span class="note">'+esc(c.name||"")+'</span></span>'+negBadge(c)+stBadge(c.status,c)+'</summary>'+
    '<div class="kv"><span>cloud / GPU</span><span class="num">'+esc(platform(c))+' · '+esc(c.gpu||"n/a")+' × '+esc(c.gpus==null?"n/a":c.gpus)+'</span>'+
    '<span>phase / policy</span><span class="num">'+esc(islPhase(c)||"n/a")+' · '+(c.policy_version==null?"n/a":"v"+esc(c.policy_version))+'</span>'+
    '<span>last event</span><span class="num">'+(fin(c.last_event_age_s)?f(c.last_event_age_s,0)+" s ago":"n/a")+'</span>'+
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
  $("costBox").innerHTML='<table class="c"><tr><th></th><th>GPU/h</th><th>CPU+memory/h</th><th>total/h</th><th>duration</th><th>cumulative</th></tr>'+ci.map(function(r){
    return '<tr><td>Island '+esc(r.id)+'</td><td class="num">'+(fin(r.gpu_rate_usd_h)?"$"+f(r.gpu_rate_usd_h,2):"unpriced")+'</td><td class="num">'+(fin(r.host_rate_usd_h)?"$"+f(r.host_rate_usd_h,2):"GPU only")+'</td><td class="num">'+(fin(r.rate_usd_h)?"$"+f(r.rate_usd_h,2):"n/a")+'</td><td class="num">'+f(r.hours,2)+' h</td><td class="num">'+(fin(r.cost_usd)?"$"+f(r.cost_usd,1):"n/a")+'</td></tr>'}).join("")+'</table>'+
    '<div class="note" style="margin-top:6px">'+esc(c.note||"")+'. Specs: '+ci.map(function(r){return "island "+esc(r.id)+" "+(r.cpus==null?"n/a":r.cpus)+" cores, "+(r.memory_gib==null?"n/a":r.memory_gib)+" GiB"}).join("; ")+'. Prices: '+esc(c.prices_source||"")+'</div>';
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
function events(){
  var list=EVENTS.filter(function(e){return keepEv(e.type)}).slice(-200).reverse();
  $("evs").innerHTML=list.map(function(e){var l=evLevel(e.type,e.record||{});return '<div><span class="t">'+clock(e.ts)+'</span><span class="st '+l[0]+'" title="'+l[1]+'"><i></i>'+l[1]+'</span><span class="x"><b>'+esc(e.type.replace(/^rl_/,""))+'</b>'+(e.island!=null?" island "+esc(e.island):"")+' '+esc(evText(e))+'</span></div>'}).join("")||'<div class="note">none</div>';
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
  $("foldBox").innerHTML=items.map(function(x){return '<div class="fold"><span>'+x[0]+'</span><span>not used in this run ('+x[1]+')</span></div>'}).join("")||'<div class="note">none</div>';
}
function spot(){
  var rows=(V.spot||[]).slice(-50).reverse();$("spotBox").hidden=!rows.length;if(!rows.length)return;
  $("spot").innerHTML='<table class="c"><tr><th>event</th><th>island</th><th>cloud / region</th><th>source</th><th>seconds left</th><th>saved</th><th>outcome</th><th>candidates/rejected</th></tr>'+rows.map(function(r){
    return '<tr><td>'+esc(r.event)+'</td><td>'+esc(r.island)+'</td><td class="mono">'+esc(r.cloud||"n/a")+' / '+esc(r.region||"n/a")+'</td><td>'+esc(r.source||"")+'</td><td class="num">'+f(r.remaining_s,1)+'</td><td>'+(r.saved==null?"":(r.saved?"yes":"no"))+'</td><td>'+esc(r.outcome||(r.auto_launch===false?"advice only":""))+'</td><td class="num">'+(r.candidates==null?"":esc(r.candidates)+' / '+esc(r.rejected))+'</td></tr>'}).join("")+'</table>'}
/* async GRPO: trainer rounds, samplers, buffer backlog, releases, NC pool */
// trainer rounds table: [header, title, cell(r, a), has(r, a) or null = always shown]; a column no shown
// round has a value for (an algorithm that does not report it, e.g. PPO KL with score centering) is hidden
// and named under the table
var isClip=function(r){return fin(r.tis_clip)?r.tis_clip:r.is_clip};
var TR_COLS=[["round","",function(r){return r.round},null],
  ["reward mean ± std","",function(r){return f(r.reward)+(fin(r.reward_std)?" ± "+f(r.reward_std):"")},null],
  ["p10 / p50 / p90","",function(r){return f(r.p10,2)+" / "+f(r.p50,2)+" / "+f(r.p90,2)},function(r){return fin(r.p50)}],
  ["train-rollout KL","train_metrics.train_rollout_kl: trainer vs sampler logprobs",function(r){return f(r.kl_tr,5)},function(r){return fin(r.kl_tr)}],
  ["PPO KL","train_metrics.ppo_kl: trainer policy vs the old logprobs of the batch",function(r){return fin(r.kl)&&Math.abs(r.kl)<1e-3&&r.kl!==0?r.kl.toExponential(2):f(r.kl,5)},function(r){return fin(r.kl)}],
  ["logprob |diff|","",function(r){return f(r.logprob_diff,4)},function(r){return fin(r.logprob_diff)}],
  ["TIS / IS clipped","mismatch.tis_clipfrac (TIS); with score centering train_metrics.sc_is_clipped (share of tokens whose IS weight was clipped)",function(r){return f(isClip(r),4)},function(r){return fin(isClip(r))}],
  ["grad_norm","",function(r){return f(r.grad_norm,4)},function(r){return fin(r.grad_norm)}],
  ["generated tokens mean / p95","model-generated tokens per trained sample (loss mask); full = whole response incl. tool / env tokens, when the tape has it",
   function(r){return f(r.resp_mean,0)+(r.resp_derived?"*":"")+" / "+f(r.resp_p95,0)+(fin(r.resp_full_mean)?'<div class="note">full '+f(r.resp_full_mean,0)+'</div>':"")},function(r){return fin(r.resp_mean)}],
  ["truncated","",function(r){return pct(r.trunc)},function(r){return fin(r.trunc)}],["tok/s","",function(r){return f(r.tok_s,0)},function(r){return fin(r.tok_s)}],
  ["groups","",function(r,a){return f(a.trained_groups,0)},null],
  ["zero-variance","rl_local_round.zero_variance_group_ratio: groups whose rewards were all equal (trained with zero advantage)",function(r,a){return fin(r.zero_var_ratio)?pct(r.zero_var_ratio)+(fin(a.trained_groups)?" ("+f(Math.round(r.zero_var_ratio*a.trained_groups),0)+")":""):"n/a"},function(r){return fin(r.zero_var_ratio)}],
  ["stale dropped","",function(r,a){return f(a.stale_groups,0)},null],
  ["policy ages","rl_round_trained.policy_age_groups: trained groups by policy age",function(r,a){return hist(a.ages)+(fin(a.policy_age_mean)?' <span class="note">mean '+f(a.policy_age_mean,2)+'</span>':"")},null]];
function trTable(rows,ar){var cols=TR_COLS.filter(function(c){return !c[3]||rows.some(function(r){return c[3](r,ar[r.round]||{})})}),gone=TR_COLS.filter(function(c){return cols.indexOf(c)<0});
  return '<table class="c"><tr>'+cols.map(function(c){return '<th'+(c[1]?' title="'+esc(c[1])+'"':"")+'>'+esc(c[0])+'</th>'}).join("")+'</tr>'+
    rows.map(function(r){var a=ar[r.round]||{};return '<tr>'+cols.map(function(c,i){return '<td'+(i===cols.length-1?"":' class="num"')+'>'+c[2](r,a)+'</td>'}).join("")+'</tr>'}).join("")+'</table>'+
    (gone.length?'<div class="note">hidden: '+gone.map(function(c){return esc(c[0])}).join(", ")+' (not reported by this algorithm / run)</div>':"")}
function hist(h){var ks=Object.keys(h||{});return ks.length?ks.map(function(k){return "age "+esc(k)+": "+f(h[k],0)}).join(" · "):"n/a"}
function asyncPanel(){
  var A=V.async;$("async").hidden=!A;$("navAsync").style.display=A?"":"none";if(!A)return;
  var cards=O.islands||[],trs=cards.filter(function(c){return c.role==="trainer"}),sps=cards.filter(function(c){return c.role==="sampler"});
  var tid=trs.some(function(c){return c.id===SEL})?SEL:(trs[0]||{}).id;$("aTrNote").textContent=tid!=null?"· island "+tid+(trs.length>1?" (pick another trainer on the island wall)":""):"";
  var rr=((V.islands[tid]||{}).rounds||[]).filter(function(r){return fin(r.reward)||fin(r.kl_tr)||fin(r.kl)||fin(r.tok_s)}),ar={};
  (((A.trainers||{})[tid]||{}).rounds||[]).forEach(function(x){ar[x.rollout_id]=x});
  var T=(A.trainers||{})[tid]||{};
  $("aTrainer").innerHTML=!rr.length?'<div class="note">no trained rounds on this trainer</div>':
    '<div class="note">totals: trained groups '+f(T.trained_groups,0)+' · dropped as stale (older than max_policy_age) '+f(T.stale_groups,0)+' · no policy identity (dropped after a resume) '+f(T.unidentified_policy_groups,0)+(T.unidentified_starved?' <b>[reader starved: '+f(T.unidentified_starved.selections,0)+' selections kept no group]</b>':'')+' · off-policy gate dropped '+f(T.gate_dropped,0)+' · short groups trained '+f(T.incomplete_groups,0)+' · too small dropped '+f(T.dropped_small_groups,0)+' · carried over '+f(T.carried_over_groups,0)+' · padded samples '+f(T.padded_samples,0)+' · policy ages of trained groups: '+hist(T.age_hist)+'</div>'+
    trTable(rr.slice(-40).reverse(),ar)+
    (rr.some(function(r){return r.resp_derived})?'<div class="note">* mean response length = trained response tokens ÷ trained samples (rl_rollout_carry_over), the tape has no batch summary</div>':"");
  var S=A.samplers||{};
  // zero-variance groups the trainer saw (rl_local_round ratio x trained groups), for samplers that report none
  var zg=0,zt=0;Object.keys(A.trainers||{}).forEach(function(i){var tg={};((A.trainers[i]||{}).rounds||[]).forEach(function(x){tg[x.rollout_id]=x.trained_groups});
    ((V.islands[i]||{}).rounds||[]).forEach(function(r){if(fin(r.zero_var_ratio)&&fin(tg[r.round])){zg+=r.zero_var_ratio*tg[r.round];zt+=tg[r.round]}})});
  var zvCell=function(x){return fin(x.zero_variance_seen)||fin(x.zero_variance_dropped)?f(x.zero_variance_seen,0)+" / "+f(x.zero_variance_dropped,0):
    zt?'<span class="note" title="the sampler tape has no per-trial rewards; rl_local_round.zero_variance_group_ratio of the trainer">trainer: '+pct(zg/zt)+' of '+f(zt,0)+' trained</span>':"not reported"};
  $("aSamplers").innerHTML=!sps.length?'<div class="note">no sampler islands on the tapes</div>':'<table class="c"><tr><th>sampler</th><th>status</th><th>policy</th><th>rounds</th><th>policy age p50 / max</th><th>age histogram (rounds)</th><th>generate s p50 / max</th><th>groups written</th><th>zero-variance groups seen / dropped</th><th>truncated samples</th><th>dropped groups</th><th>trials retried / lost</th><th>reward (last round)</th><th>backlog cap</th></tr>'+
    sps.map(function(c){var x=S[c.id]||{};return '<tr><td>Island '+esc(c.id)+(c.name?' <span class="note">'+esc(c.name)+'</span>':"")+'</td><td>'+stBadge(c.status,c)+'</td><td class="num">'+(x.policy_version==null?"n/a":"v"+esc(x.policy_version))+'</td><td class="num">'+f(x.n_rounds,0)+
      '</td><td class="num">'+f(x.lag_p50,1)+" / "+f(x.lag_max,0)+'</td><td>'+hist(x.lag_hist)+'</td><td class="num">'+f(x.generate_p50_s,0)+" / "+f(x.generate_max_s,0)+'</td><td class="num">'+f(x.groups,0)+'</td><td class="num">'+zvCell(x)+
      '</td><td class="num">'+f(x.truncated,0)+(fin(x.samples)&&fin(x.truncated)?" ("+pct(x.truncated/x.samples)+")":"")+'</td><td class="num">'+f(x.dropped_groups,0)+'</td><td class="num">'+f(x.retried_trials,0)+" / "+f(x.lost_trials,0)+'</td><td class="num">'+f(x.reward_mean_last)+'</td><td class="num">'+f(x.max_backlog_groups,0)+'</td></tr>'}).join("")+'</table>'+
    (sps.some(function(c){var x=S[c.id]||{};return !fin(x.samples)})?'<div class="note">n/a in truncated / reward: the sampler did not write rl_sampler_engine_round with these fields (Miles-generating samplers and older yeto versions do not); zero-variance then comes from the trainer</div>':"");
  aBacklog(A);aRel(A);aComp(A);aDisp(A,cards);aFleet(A);
  var pl=(A.pool||[]).slice(-100).reverse();$("aPoolBox").hidden=!pl.length;
  var sc=A.pool_scale;$("aPoolScale").textContent=sc?("samplers: "+sc.ready.length+" ready"+(sc.draining.length?", "+sc.draining.length+" draining ("+sc.draining.join(", ")+")":"")+(sc.paused&&sc.paused.length?", "+sc.paused.length+" paused by hand ("+sc.paused.join(", ")+")":"")+(sc.target!=null?" · target "+sc.target+(sc.manual?" (set by hand)":""):"")+(sc.ready.length?" · shards "+sc.ready.join(", "):"")):"";
  $("aPool").innerHTML=pl.map(function(e){var bad=POOL_BAD.test(e.event||"");return '<div><span class="t">'+clock(e.ts)+'</span><span class="st '+(bad?"bd":"wn")+'"><i></i>'+(/^nc_elastic_manual_/.test(e.event||"")?"manual":"pool")+'</span><span class="x"><b>'+esc(String(e.event).replace(/^nc_elastic_/,""))+'</b> '+
    esc(["ticket","slot","shard","why","reason","detail","error","old","new","outcome","waited_s","source","by"].filter(function(k){return k=="slot"?(e.slot!=null||e.account!=null):e[k]!=null}).map(function(k){var v=e[k],n=k;if(k=="slot"){n=e.slot_name||"account";v=e.slot!=null?e.slot:e.account}return n+"="+String(v).slice(0,80)}).join(" "))+'</span></div>'}).join("");
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
function aComp(A){
  var rows=[],recent=[];
  Object.keys(A.samplers||{}).sort().forEach(function(k){var x=A.samplers[k];if(x.compaction)rows.push(["sampler",k,x.compaction]);
    (x.rounds||[]).forEach(function(r){if(r.compaction)recent.push([k,r])})});
  Object.keys(A.trainers||{}).sort().forEach(function(k){var x=A.trainers[k];if(x.compaction)rows.push(["trainer",k,x.compaction])});
  $("aCompBox").hidden=!rows.length;if(!rows.length){$("aComp").innerHTML="";return}
  recent.sort(function(a,b){return (a[1].ts||0)-(b[1].ts||0)});
  var fr=fateRows(rows,A);
  $("aComp").innerHTML='<table class="c"><tr><th>island</th><th>rounds</th>'+COMP_HEAD+'<th>reward mean per trial / per segment (last round)</th><th>xhw downweighted groups (max trial-mean shift)</th></tr>'+rows.map(function(x){var c=x[2];return '<tr><td>'+esc(x[0])+" "+esc(x[1])+'</td><td class="num">'+f(c.rounds,0)+'</td>'+compCells(c)+'<td class="num">'+f(c.reward_mean_trial)+" / "+f(c.reward_mean_segment)+'</td><td class="num">'+(c.xhw_downweighted_groups==null?"n/a":f(c.xhw_downweighted_groups,0)+" ("+f(c.xhw_trial_mean_shift_max,4)+")")+'</td></tr>'}).join("")+'</table>'+
    rows.filter(function(x){return x[2].drained_by_segments}).map(function(x){return '<div class="note">'+esc(x[0])+" "+esc(x[1])+': trials cut by drains, by segment count: '+reasons(x[2].drained_by_segments)+'</div>'}).join("")+
    (fr.length?'<div class="note">trials by segment count: written (trainer: trained), lost before the group was written (drain timeout, other version, aborted), dropped with their group, stale-dropped by the trainer; a much higher share for long trials biases the GRPO baseline to short ones</div><table class="c"><tr><th>island</th><th>segments</th><th>written</th><th>lost</th><th>dropped with group</th><th>stale (trainer)</th><th>not written</th></tr>'+fr.join("")+'</table>':"")+
    (recent.length?'<div class="note">recent sampler rounds</div><table class="c"><tr><th>sampler</th><th>round</th><th>time</th>'+COMP_HEAD+'</tr>'+recent.slice(-20).reverse().map(function(x){return '<tr><td>'+esc(x[0])+'</td><td class="num">'+esc(x[1].rollout_id)+'</td><td class="num">'+clock(x[1].ts)+'</td>'+compCells(x[1].compaction,true)+'</tr>'}).join("")+'</table>':"")+
    '<div class="note">alert cards fire on a low summary_ok share and on many compacted trials ending truncated (docs/DASHBOARD.md, compaction_* thresholds); summary tokens n/a = the samples do not report summary_tokens; — = no trial was compacted</div>';
}
function tAxis(s,t0,t1,sx,H){[t0,(t0+t1)/2,t1].forEach(function(t,i){el("text",{x:sx(t),y:H-6,"text-anchor":["start","middle","end"][i]},s).textContent=clock(t)})}
function aBacklog(A){
  if(A.islands){$("aBacklog").innerHTML='<div class="note">one backlog per island (each trainer reads its own store lane): see Islands above</div>';$("legBacklog").innerHTML="";return}
  blChart($("aBacklog"),$("legBacklog"),A.backlog)}
function blChart(box,leg,b){
  box.innerHTML="";leg.innerHTML="";
  if(!b||!(b.points||[]).length){box.innerHTML='<div class="note">no data (no rl_sampler_round with groups)</div>';return}
  var pts=b.points,rep=b.reported||{},rk=Object.keys(rep),all=pts.map(function(p){return p[0]});rk.forEach(function(k){rep[k].forEach(function(p){all.push(p[0])})});
  var t0=Math.min.apply(null,all),t1=Math.max.apply(null,all),vm=Math.max.apply(null,pts.map(function(p){return fin(p[3])?p[3]:0}).concat([1]));
  rk.forEach(function(k){rep[k].forEach(function(p){vm=Math.max(vm,p[1])})});
  var W=boxW(box,560),H=220,m={l:44,r:14,t:10,b:22},sx=function(t){return m.l+(t-t0)*(W-m.l-m.r)/((t1-t0)||1)},sy=function(v){return m.t+(1-v/vm)*(H-m.t-m.b)};
  var s=el("svg",{viewBox:"0 0 "+W+" "+H,role:"img","aria-label":"Buffer backlog"},box),g=el("g",{"class":"grid"},s);
  [0,.5,1].forEach(function(q){el("line",{x1:m.l,x2:W-m.r,y1:sy(vm*q),y2:sy(vm*q)},g);el("text",{x:m.l-6,y:sy(vm*q)+4,"text-anchor":"end"},s).textContent=Math.round(vm*q)});
  var bp=pts.filter(function(p){return fin(p[3])});
  if(bp.length)el("path",{d:"M"+bp.map(function(p){return sx(p[0])+","+sy(p[3])}).join("L"),fill:"none",stroke:"var(--T)","stroke-width":2},s);
  rk.forEach(function(k,i){rep[k].forEach(function(p){el("circle",{cx:sx(p[0]),cy:sy(p[1]),r:3.5,fill:p[2]?"var(--bad)":RUN_COLORS[(i+1)%4]},s)})});
  tAxis(s,t0,t1,sx,H);
  leg.innerHTML='<span><i class="sw" style="background:var(--T)"></i>backlog (written '+f(b.written,0)+' · read position '+f(b.read,0)+')</span>'+(rk.length?'<span><i class="sw" style="background:var(--bad)"></i>sampler reported: backlog full</span><span><i class="sw" style="background:var(--i1)"></i>sampler reported: admitted</span>':"")+
    (b.has_reads&&b.read_source&&b.read_source!=="buffer_selected"?'<span>read position = '+esc(b.read_source)+' (no rl_policy_released.buffer_selected yet)</span>':"")+(b.has_reads?"":'<span>no read position on the tapes (no rl_policy_released.buffer_selected): backlog unknown</span>');
}
function aDisp(A,cards){
  // task dispense ledger (rl_dispense_*): slots by state, claims still held and who holds them
  var D=A.dispense,box=$("aDispBox");box.hidden=!D;if(!D)return;
  var st={};(cards||[]).forEach(function(c){st[c.id]=c.status});
  var dead=function(id){return ["stale","lost","left","stopped","done","recovery"].indexOf(st[id])>=0};
  var kv=function(o){var k=Object.keys(o||{});return k.length?k.map(function(x){return esc(x)+" "+o[x]}).join(" · "):"0"};
  var now=fin(O.now)?O.now:Date.now()/1e3,held=D.held||[],by={};held.forEach(function(h){(by[h.by]=by[h.by]||[]).push(h)});
  var html='<div class="note">slots seen '+D.slots+' · settled: '+kv(D.settled)+' · attempts ended unsettled: '+kv(D.ends)+' (expired = lease ran out, the slot is claimed again) · re-claimed slots '+D.reclaimed_slots+' · held now '+held.length+'</div>';
  html+=Object.keys(by).sort().map(function(k){var hs=by[k],bad=dead(k);
    return '<div class="note">'+(bad?'<span class="st bd"><i></i>'+esc(st[k])+'</span> ':"")+'sampler '+esc(k)+' holds '+hs.length+': '+hs.map(function(h){return "slot "+h.slot+(h.attempt?" (attempt "+h.attempt+")":"")+" claimed "+clock(h.claim_ts)+(fin(h.claim_ts)?" ("+f((now-h.claim_ts)/60,0)+" min ago)":"")}).join(", ")+
      (bad?' — <b>its island is '+esc(st[k])+': these slots wait for their lease to expire, then another sampler re-claims them</b>':"")+'</div>'}).join("");
  var ev=(D.events||[]).filter(function(e){return e.kind!=="drain"}).slice(-12).reverse();
  if(ev.length)html+='<table class="c"><tr><th>time</th><th>event</th><th>slot</th><th>attempt</th><th>noticed by</th><th>detail</th></tr>'+ev.map(function(e){
    return '<tr><td class="num">'+clock(e.ts)+'</td><td>'+esc(e.kind)+'</td><td class="num">'+(e.slot==null?"—":e.slot)+'</td><td class="num">'+(e.attempt==null?"—":e.attempt)+'</td><td>'+esc(e.by)+'</td><td class="note">'+esc([e.reason,e.state,e.superseded_by?"superseded by "+e.superseded_by:null,fin(e.seen)?"last renewed "+clock(e.seen):null].filter(Boolean).join(" · "))+'</td></tr>'}).join("")+'</table>';
  $("aDisp").innerHTML=html;
}
function aRel(A){
  var box=$("aRelChart"),rows=A.releases||[],pool=(A.pool||[]).filter(function(e){return fin(e.ts)}),box2=$("aRelTable");box.innerHTML="";box2.innerHTML="";
  var sids=[];rows.forEach(function(r){Object.keys(r.loads||{}).forEach(function(k){if(sids.indexOf(k)<0)sids.push(k)})});sids.sort();
  var ts=[];rows.forEach(function(r){if(fin(r.released_at))ts.push(r.released_at);Object.keys(r.loads||{}).forEach(function(k){if(fin(r.loads[k].loaded_at))ts.push(r.loads[k].loaded_at)})});pool.forEach(function(e){ts.push(e.ts)});
  if(!ts.length){box.innerHTML='<div class="note">no data (no rl_policy_released / rl_sampler_policy_loaded / nc_elastic_* events)</div>';$("legRel").innerHTML="";return}
  var t0=Math.min.apply(null,ts),t1=Math.max.apply(null,ts),lanes=["trainer release"].concat(sids.map(function(k){return "sampler "+k}));
  var W=boxW(box,560),rowH=34,ml=Math.min(Math.round(W*0.4),Math.max(86,14+6.7*Math.max.apply(null,lanes.map(function(n){return n.length})))),m={l:ml,r:16,t:14,b:22},H=m.t+m.b+rowH*lanes.length,sx=function(t){return m.l+(t-t0)*(W-m.l-m.r)/((t1-t0)||1)},ly=function(i){return m.t+rowH*i+rowH/2};
  var s=el("svg",{viewBox:"0 0 "+W+" "+H,role:"img","aria-label":"Policy release timeline"},box);
  lanes.forEach(function(n,i){el("text",{x:m.l-8,y:ly(i)+4,"text-anchor":"end"},s).textContent=n;el("line",{x1:m.l,x2:W-m.r,y1:ly(i),y2:ly(i),stroke:"var(--line)"},s)});
  pool.forEach(function(e){var x=sx(e.ts),bad=POOL_BAD.test(e.event||"");el("line",{x1:x,x2:x,y1:m.t,y2:H-m.b,stroke:bad?"var(--bad)":"var(--warn)","stroke-dasharray":"3 3"},s);
    el("title",{},el("rect",{x:x-3,y:m.t,width:6,height:H-m.t-m.b,fill:"transparent"},s)).textContent=e.event+" "+clock(e.ts)});
  rows.forEach(function(r){if(!fin(r.released_at))return;var x0=sx(r.released_at);
    el("rect",{x:x0-4,y:ly(0)-4,width:8,height:8,fill:"var(--P)",transform:"rotate(45 "+x0+" "+ly(0)+")"},s);
    el("text",{x:x0,y:ly(0)-8,"text-anchor":"middle"},s).textContent="v"+r.policy_version;
    sids.forEach(function(k,i){var ld=(r.loads||{})[k];if(!ld||!fin(ld.loaded_at))return;var x1=sx(ld.loaded_at);
      el("line",{x1:x0,y1:ly(0),x2:x1,y2:ly(i+1),stroke:"var(--P)","stroke-opacity":.5},s);el("circle",{cx:x1,cy:ly(i+1),r:4,fill:"var(--P)"},s)})});
  tAxis(s,t0,t1,sx,H);
  $("legRel").innerHTML='<span><i class="sw" style="background:var(--P)"></i>release → load done</span>'+(pool.length?'<span><i class="sw" style="background:var(--warn)"></i>NC pool event</span><span><i class="sw" style="background:var(--bad)"></i>NC pool failure (evicted / lost / crash loop)</span>':"");
  if(rows.length)box2.innerHTML=relTable(rows,fleetSamplers(A),milesSamplers(A));
}
// a fleet's samplers -> their island: a release row of another island is no load the sampler missed
function fleetSamplers(A){var o={};(A.islands||[]).forEach(function(x){x.pools.forEach(function(p){p.samplers.forEach(function(s){o[s]=x.island})})});return o}
// a fleet's Miles sampler islands of the launch: their release-load column says so (an external pool's keeps its id)
function milesSamplers(A){var o={};(A.islands||[]).forEach(function(x){x.pools.forEach(function(p){if(p.source==="miles")p.samplers.forEach(function(s){o[s]=1})})});return o}
function relTable(rows,sIsl,ml){ml=ml||{};
  var sids=[];rows.forEach(function(r){Object.keys(r.loads||{}).forEach(function(k){if(sids.indexOf(k)<0)sids.push(k)})});sids.sort();
  // a fleet: one row per (island, version), both islands may release v4
  var il=rows.some(function(r){return r.island!=null});sIsl=sIsl||{};
  return '<table class="c"><tr>'+(il?'<th>island</th>':"")+'<th>policy</th><th>trainer</th><th>released at</th><th>upload s</th>'+sids.map(function(k){return '<th>'+(ml[k]?"Miles sampler island ":"sampler ")+esc(k)+' loaded (latency)</th>'}).join("")+'</tr>'+
    // a sampler starting on the base weights makes a v0 row with no trainer release: "base"
    rows.slice(-30).reverse().map(function(r){var base=r.trainer==null&&!fin(r.released_at);return '<tr>'+(il?'<td class="num">'+esc(r.island==null?"—":r.island)+'</td>':"")+'<td class="num">v'+esc(r.policy_version)+'</td><td>'+(base?'<span class="note">base</span>':esc(r.trainer==null?"n/a":r.trainer))+'</td><td class="num">'+(base?'<span class="note">base weights</span>':clock(r.released_at))+'</td><td class="num">'+(base?"—":f(r.upload_s,1))+'</td>'+
      sids.map(function(k){var ld=(r.loads||{})[k];if(!ld&&il&&sIsl[k]!=null&&sIsl[k]!==r.island)return '<td class="num note">—</td>';return '<td class="num">'+(ld?clock(ld.loaded_at)+(ld.at_start?" (at start)":" ("+f(ld.latency_s,0)+" s)"+(fin(ld.drain_s)?'<div class="note">drain '+f(ld.drain_s,0)+" s"+(ld.drain_cut?", "+f(ld.drain_cut,0)+" cut":"")+'</div>':"")):"not loaded")+'</td>'}).join("")+'</tr>'}).join("")+'</table>'}
/* a fleet of async islands (async.islands, fleet_islands.py): per island its pools, backlog, policy ages and releases */
var ENG={vllm:"vLLM",sglang:"SGLang"};
function engName(e){return e?(ENG[String(e).toLowerCase()]||String(e)):null}
function mmCell(m){return m?f(m.value,4)+'<div class="note">'+esc(m.method)+" · "+m.rounds+" round"+(m.rounds===1?"":"s")+"</div>":'<span class="note">not reported</span>'}
// a fleet with Miles sampler islands of the launch: each sampler row's source
var SRC_TAG={miles:["Miles native","var(--delta)"],external:["external pool","var(--traj)"]};
function srcTag(s){var t=SRC_TAG[s];return t?' <span class="tag" style="border-color:'+t[1]+';color:'+t[1]+';font-weight:600;white-space:nowrap">'+t[0]+'</span>':""}
function poolTable(x){
  return '<table class="c"><tr><th>pool</th><th>engine · precision</th><th>cards · provider</th><th>samplers</th><th>policy</th><th>policy age p50 / max</th><th>groups written</th>'+
    '<th title="train_metrics.train_rollout_logprob_abs_diff of the steps this pool fed">logprob |diff|</th><th title="train_metrics.tis_clipfrac (mismatch.tis_clipfrac) of the steps this pool fed">TIS clipfrac</th></tr>'+
    x.pools.map(function(p){var pr=p.provenance||{},mm=p.mismatch||{},prec=String(p.rollout_quant||pr.precision||"").toUpperCase(),ml=p.source==="miles";
      // a Miles row: cards of its --gpu cluster when its sampler reports no provenance
      var card=pr.card_model||(p.card?p.card+(fin(p.gpus)?" × "+p.gpus:""):null);
      return '<tr><td>'+esc(p.name)+(p.source_name&&p.source_name!==p.name?' <span class="note">'+esc(p.source_name)+'</span>':"")+srcTag(p.source)+'</td><td>'+esc([engName(p.engine||pr.engine),prec].filter(Boolean).join(" · ")||"not reported")+
        '</td><td>'+esc([card,pr.topology,p.provider].filter(Boolean).join(" · ")||"not reported")+'</td><td>'+esc(p.samplers.map(function(s){return ml?"launch island "+s:s}).join(", ")||"none yet")+'</td><td class="num">'+(fin(p.policy_version)?"v"+esc(p.policy_version):"n/a")+
        '</td><td class="num">'+f(p.lag_p50,1)+" / "+f(p.lag_max,0)+'</td><td class="num">'+f(p.groups,0)+'</td><td class="num">'+mmCell(mm.logprob_diff)+'</td><td class="num">'+mmCell(mm.tis_clip)+'</td></tr>'}).join("")+'</table>'}
function aFleet(A){
  var box=$("aFleetBox"),I=A.islands;if(!box)return;box.hidden=!I;if(!I)return;
  $("aFleet").innerHTML=I.map(function(x,i){var pa=x.policy_age||{},b=x.backlog||{},pts=b.points||[],last=pts.length?pts[pts.length-1]:null;
    return '<h3>Island '+esc(x.island)+(x.name?" · "+esc(x.name):"")+' <span class="note">'+esc([x.trainer!=null?"trainer island "+x.trainer:"trainer not on the tapes yet",x.gpu,"R2 lane "+(x.substore||"trainer"+x.island)+"/",
      fin(x.release_every)?"release every "+x.release_every:null,fin(x.max_policy_age)?"max policy age "+x.max_policy_age:null].filter(Boolean).join(" · "))+'</span></h3>'+
      '<div class="note">backlog '+(last&&fin(last[3])?f(last[3],0)+" groups":"n/a")+" (written "+f(b.written,0)+" · read position "+f(b.read,0)+") · latest release "+(fin(x.latest)?"v"+x.latest:"none")+
      " · trained groups "+f(x.trained_groups,0)+" · stale dropped "+f(pa.stale_groups,0)+" · policy age of trained groups: "+hist(pa.hist)+(fin(pa.mean_last)?" (last round mean "+f(pa.mean_last,2)+", max "+f(pa.max_last,0)+")":"")+'</div>'+
      poolTable(x)+'<div style="max-width:760px"><div class="legend" id="legFl'+i+'"></div><div id="aFl'+i+'"></div></div>'+((x.releases||[]).length?relTable(x.releases,null,milesSamplers({islands:[x]})):"")}).join("");
  I.forEach(function(x,i){blChart($("aFl"+i),$("legFl"+i),x.backlog)})}
function render(){
  if(!V)return;O=V.overview;
  var ids=(O.islands||[]).map(function(c){return c.id});if(ids.indexOf(SEL)<0)SEL=ids[0]||null;
  RS=SEL!=null?((V.islands[SEL]||{}).rounds||[]):[];
  $("roundsIsl").textContent=SEL!=null&&ids.length>1?"· island "+SEL:"";
  header();kpis();wall();chartMetric();chartDur();chartTl();highlight();asyncPanel();util();chartNodes();islands();cost();events();cmds();folds();spot();
  if(typeof YetoExt!=="undefined")YetoExt.render(V,OFFLINE);
  if(typeof YetoTopo!=="undefined")YetoTopo.render(V,OFFLINE);
  if(typeof YetoBrk!=="undefined")YetoBrk.render(V);
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
var resizeT=null;if(typeof window!=="undefined"&&window.addEventListener)window.addEventListener("resize",function(){clearTimeout(resizeT);resizeT=setTimeout(render,250)}); // charts drawn at the box width
load();if(!OFFLINE)setInterval(load,POLL_MS);
})();
