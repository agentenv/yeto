// yeto dashboard (section 9 redesign). No build step, no external requests.
// Live: polls /api/view every 5 s. Offline export: reads the inlined full view (its "page" key).
// Every number comes from the reducer; anything missing is shown as "无" (never interpolated).
(function(){
"use strict";
var node=document.getElementById("yeto-data"),INLINE=null;
if(node){try{INLINE=JSON.parse(node.textContent)}catch(e){INLINE=null}}
var OFFLINE=!!INLINE,POLL_MS=5000,NS="http:"+"//www.w3.org/2000/svg"; // SVG namespace id, not a request
var V=null,O=null,RS=[],SEL=null,cur=null,layout="auto",metric="reward",themeI=0,pinned=false;
var PH=[["R","推理生成"],["T","训练"],["S","训练后同步"],["P","发布"]];
var ISL_COLORS=["var(--i0)","var(--i1)","var(--i2)"];
var METRICS=[["reward","reward"],["trunc","截断率"],["logprob_diff","logprob 差"],["kl","KL"],["grad_norm","grad_norm"],
  ["clip","clip"],["entropy","entropy"],["resp_len","回答长度"],["tok_s","tok/s"]];
function $(id){return document.getElementById(id)}
function esc(s){return String(s==null?"":s).replace(/[&<>"']/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]})}
function fin(v){return typeof v==="number"&&isFinite(v)}
function f(v,d){if(!fin(v))return "无";if(d==null)d=3;return Math.abs(v)>=100?v.toFixed(0):v.toFixed(d)}
function pct(v){return fin(v)?(v*100).toFixed(1)+"%":"无"}
function hm(s){return !fin(s)?"无":(s>=3600?(s/3600).toFixed(1)+" h":(s/60).toFixed(0)+" min")}
function clock(t){if(!fin(t))return "无";var d=new Date(t*1000);function p(n){return (n<10?"0":"")+n}return p(d.getHours())+":"+p(d.getMinutes())+":"+p(d.getSeconds())}
function el(tag,a,parent){var e=document.createElementNS(NS,tag);for(var k in a)e.setAttribute(k,a[k]);if(parent)parent.appendChild(e);return e}
// launch-preflight-guards 3.5: negative-test island badge (rl_island_override)
function negBadge(c){return c&&c.negative_test?' <span class="st wn" title="'+esc(c.negative_test.label)+'"><i></i>'+esc(c.negative_test.label)+'</span> ':""}
function islColor(id){var ids=(O.islands||[]).map(function(c){return c.id}),i=ids.indexOf(id);return i>=0&&i<3?ISL_COLORS[i]:"var(--i3)"}
function kind(){return layout==="auto"?O.run_kind:layout}
function stBadge(s){var m={ok:["ok","健康"],starting:["wn","启动中"],stopped:["mu","已停止"],stale:["bd","掉线疑似"],lost:["bd","已丢失"],
  recovery:["bd","需恢复"],done:["mu","已结束"],unknown:["mu","无数据"]}[s]||["mu",s];return '<span class="st '+m[0]+'"><i></i>'+esc(m[1])+'</span>'}

/* theme + layout */
var THEMES=["auto","light","dark"];
try{var saved=localStorage.getItem("yeto-theme");if(saved)themeI=Math.max(0,THEMES.indexOf(saved))}catch(e){}
function applyTheme(){var t=THEMES[themeI];if(t==="auto")document.documentElement.removeAttribute("data-theme");else document.documentElement.setAttribute("data-theme",t);
  $("themeBtn").textContent="主题："+{auto:"跟随系统",light:"浅色",dark:"深色"}[t];try{localStorage.setItem("yeto-theme",t)}catch(e){}}
$("themeBtn").onclick=function(){themeI=(themeI+1)%3;applyTheme()};applyTheme();
$("layoutBtn").onclick=function(){layout=layout==="auto"?(O.run_kind==="single_island"?"multi_island":"single_island"):"auto";render()};

function header(){
  var k=kind();
  $("title").innerHTML=esc(k==="single_island"?"单岛总览":"多岛总览")+' · <span class="mono">'+esc(O.run||"未命名运行")+'</span>'+
    (O.run_inferred?'<span class="tag">名称由磁带推断</span>':"")+'<span class="tag">'+esc(O.run_kind)+'</span>';
  document.title="yeto 看板 · "+(O.run||"未命名运行");
  var c=(O.islands||[])[0]||{};
  $("sub").textContent=(OFFLINE?"离线导出":"实时")+" · 数据截至 "+clock(O.data_ts)+(c.cloud?" · "+c.cloud+" "+(c.gpu||"")+" × "+(c.gpus==null?"无":c.gpus):"");
  $("layoutBtn").textContent="布局："+(layout==="auto"?"自动（"+(O.run_kind==="single_island"?"单岛":"多岛")+"）":(layout==="single_island"?"单岛（临时）":"多岛（临时）"));
  $("navSub").textContent=(O.global_status||{}).text||"";
}
function kpis(){
  var gs=O.global_status||{},c=O.cost||{},isl=(O.islands||[]).find(function(x){return x.id===SEL})||{},last=RS[RS.length-1]||{};
  var rate=(c.islands||[]).reduce(function(a,r){return a+(fin(r.rate_usd_h)?r.rate_usd_h:0)},0);
  var dur=fin(O.data_ts)&&fin(O.first_ts)?O.data_ts-O.first_ts:null;
  var pubs=RS.map(function(r){return r.published_version}).filter(fin);
  var lvl={ok:"ok",warn:"wn",bad:"bd",muted:"mu"}[gs.level]||"mu";
  var cards=[["状态",'<span class="st '+lvl+'"><i></i>'+esc(gs.text||"无")+'</span>',(O.alerts||[]).length+" 条告警",true],
    ["估算成本",fin(c.total_usd)?"$"+f(c.total_usd,1):"无",(rate?"$"+f(rate,2):"无")+"/h（"+((c.gpu_only||[]).length?"部分仅 GPU":"GPU+CPU+内存")+"）"+(fin(c.budget_pct)?" · 预算 "+f(c.budget_pct,0)+"%":"")],
    ["轮次",String(RS.length),pubs.length?"已发布到 v"+Math.max.apply(null,pubs):"尚无发布"],
    ["时长",hm(dur),isl.starting?"启动中 "+hm(isl.startup_s):"首个到最后事件"],
    ["吞吐",f(last.tok_s,0),"tok/s，最后一轮"]];
  $("kpis").innerHTML=cards.map(function(x){return '<div class="kpi'+(x[3]?" main":"")+'"><div class="k">'+x[0]+'</div><div class="v">'+x[1]+'</div><div class="s">'+esc(x[2])+'</div></div>'}).join("");
  var al=O.alerts||[];$("alertBox").hidden=!al.length;
  $("alertBox").innerHTML="<h2>告警</h2>"+al.map(function(a){return '<div><span class="st '+(a.sev===0?"bd":a.sev===1?"wn":"mu")+'"><i></i>'+["严重","警告","提示"][a.sev]+'</span> <b>'+esc(a.title)+'</b> <span class="note">'+esc(a.detail)+'</span></div>'}).join("");
}
function wall(){
  var multi=kind()==="multi_island";$("islandWall").hidden=!multi;$("navWall").style.display=multi?"":"none";
  $("syncerBox").hidden=!(multi&&V.usage.syncer);
  if(multi)$("wall").innerHTML=(O.islands||[]).map(function(c){return '<div class="card'+(c.id===SEL?" sel":"")+'" data-id="'+esc(c.id)+'" tabindex="0">'+
    '<div style="display:flex;justify-content:space-between"><b><i class="sw" style="background:'+islColor(c.id)+'"></i>岛 '+esc(c.id)+'</b>'+negBadge(c)+stBadge(c.status)+'</div>'+
    '<div class="note mono">'+esc(c.cloud||"无")+' · '+esc(c.gpu||"")+' × '+esc(c.gpus==null?"无":c.gpus)+'</div>'+
    '<div class="kv"><span>策略版本</span><span class="num">'+(c.policy_version==null?"无":"v"+esc(c.policy_version))+'</span><span>reward</span><span class="num">'+f(c.reward)+'</span><span>阶段</span><span class="num">'+esc(c.phase||"无")+'</span></div></div>'}).join("");
  Array.prototype.forEach.call(document.querySelectorAll(".card"),function(d){d.onclick=d.onkeydown=function(e){if(e.type==="keydown"&&e.key!=="Enter")return;SEL=d.dataset.id;cur=null;render()}});
  if(multi&&V.usage.syncer){var rows=(V.rounds_syncer||[]).slice(-30).reverse();
    $("syncer").innerHTML='<table class="c"><tr><th>全局轮</th><th>应到/实到</th><th>缺席</th><th>quorum ms</th><th>合并 ms</th><th>重发</th></tr>'+rows.map(function(r){
      return '<tr><td class="num">'+esc(r.round)+'</td><td class="num">'+f(r.expected,0)+' / '+f(r.responded,0)+'</td><td>'+(r.missed.length?'<span class="bd">'+esc(r.missed.join(","))+'</span>':"无")+'</td><td class="num">'+f(r.quorum_ms,0)+'</td><td class="num">'+f(r.merge_ms,0)+'</td><td class="num">'+esc(r.resend)+'</td></tr>'}).join("")+'</table>'}
}
/* chart 1: metric per round; reward gets the p10-p90 band; < 5 points: dots only, no line */
var xOf=null;
function chartMetric(){
  $("tabs").innerHTML=METRICS.map(function(m){return '<button class="btn'+(m[0]===metric?" on":"")+'" data-m="'+m[0]+'">'+m[1]+'</button>'}).join("");
  Array.prototype.forEach.call($("tabs").querySelectorAll("button"),function(b){b.onclick=function(){metric=b.dataset.m;chartMetric();highlight()}});
  var multi=kind()==="multi_island",ids=multi?(O.islands||[]).map(function(c){return c.id}):[SEL];
  var series=ids.map(function(id){return {id:id,pts:((O.series||{})[id]||{})[metric==="trunc"||metric==="logprob_diff"?"_":metric]||[]}});
  if(metric==="trunc"||metric==="logprob_diff"){series=ids.map(function(id){var rr=(V.islands[id]||{}).rounds||[];return {id:id,pts:rr.filter(function(r){return fin(r[metric])}).map(function(r){return [r.round+1,r[metric]]})}})}
  var all=[];series.forEach(function(s){s.pts.forEach(function(p){all.push(p)})});
  var band=metric==="reward"&&!multi?RS.filter(function(r){return fin(r.p10)&&fin(r.p90)}):[];
  var W=560,H=230,m={l:46,r:12,t:10,b:26},box=$("cMetric");box.innerHTML="";
  $("legMetric").innerHTML=(multi?ids.map(function(id){return '<span><i class="sw" style="background:'+islColor(id)+'"></i>岛 '+esc(id)+'</span>'}).join(""):'<span><i class="sw" style="background:var(--T)"></i>'+esc(METRICS.find(function(x){return x[0]===metric})[1])+'</span>')+
    (band.length?'<span><i class="sw" style="background:var(--band)"></i>p10–p90</span>':"");
  if(!all.length){box.innerHTML='<div class="note">无数据</div>';xOf=null;return}
  var xs=all.map(function(p){return p[0]}),x0=Math.min.apply(null,xs),x1=Math.max.apply(null,xs);
  var vs=all.map(function(p){return p[1]});band.forEach(function(r){vs.push(r.p10,r.p90)});
  var lo=Math.min.apply(null,vs),hi=Math.max.apply(null,vs);if(metric==="reward"||metric==="trunc"){lo=Math.min(lo,0);hi=Math.max(hi,1)}if(hi===lo){hi+=1;lo-=1}
  var x=function(r){return m.l+(x1===x0?(W-m.l-m.r)/2:(r-x0)*(W-m.l-m.r)/(x1-x0))},y=function(v){return m.t+(hi-v)/(hi-lo)*(H-m.t-m.b)};
  xOf=function(round){return x(round+1)};
  var s=el("svg",{viewBox:"0 0 "+W+" "+H,role:"img","aria-label":"每轮指标"},box),g=el("g",{"class":"grid"},s);
  for(var j=0;j<=4;j++){var v=lo+(hi-lo)*j/4;el("line",{x1:m.l,x2:W-m.r,y1:y(v),y2:y(v)},g);el("text",{x:m.l-6,y:y(v)+4,"text-anchor":"end"},s).textContent=+v.toPrecision(3)}
  var ticks={};xs.forEach(function(r){ticks[r]=1});var tk=Object.keys(ticks).map(Number),every=Math.ceil(tk.length/10);
  tk.forEach(function(r,i){if(i%every===0)el("text",{x:x(r),y:H-8,"text-anchor":"middle"},s).textContent="第"+(r-1)+"轮"});
  var connect=series.some(function(sr){return sr.pts.length>=5});
  if(band.length>=2&&connect)el("path",{d:"M"+band.map(function(r){return x(r.round+1)+","+y(r.p90)}).join("L")+"L"+band.slice().reverse().map(function(r){return x(r.round+1)+","+y(r.p10)}).join("L")+"Z",fill:"var(--band)"},s);
  else band.forEach(function(r){el("line",{x1:x(r.round+1),x2:x(r.round+1),y1:y(r.p10),y2:y(r.p90),stroke:"var(--band)","stroke-width":10,"stroke-linecap":"round"},s)});
  series.forEach(function(sr){var col=multi?islColor(sr.id):"var(--T)";
    if(sr.pts.length>=5)el("path",{d:"M"+sr.pts.map(function(p){return x(p[0])+","+y(p[1])}).join("L"),fill:"none",stroke:col,"stroke-width":2},s);
    sr.pts.forEach(function(p){el("circle",{cx:x(p[0]),cy:y(p[1]),r:sr.pts.length>60?2.5:5,fill:col,stroke:"var(--panel)","stroke-width":2,"class":"rdot","data-r":p[0]-1},s)})});
  if(!connect)el("text",{x:W-m.r,y:m.t+10,"text-anchor":"end"},s).textContent="点少于 5 个：只画点不连线";
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
  if(!rows.some(function(r){return Object.keys(r.dur).length})){box.innerHTML='<div class="note">无阶段数据（磁带没有 rl_timeline_span）</div>';return}
  var W=420,rowH=26,m={l:56,r:56,t:4,b:4},H=m.t+m.b+rowH*rows.length;
  var s=el("svg",{viewBox:"0 0 "+W+" "+H,role:"img","aria-label":"每轮耗时拆分"},box);
  var tot=rows.map(function(r){return PH.reduce(function(a,p){return a+(r.dur[p[0]]||0)},0)}),mx=Math.max.apply(null,tot.concat([1]));
  rows.forEach(function(r,i){var y0=m.t+i*rowH,x0=m.l;
    el("rect",{x:0,y:y0,width:W,height:rowH,fill:"transparent","class":"drow","data-r":r.round},s);
    el("text",{x:m.l-8,y:y0+rowH/2+4,"text-anchor":"end"},s).textContent="第"+r.round+"轮";
    PH.forEach(function(p){var w=(r.dur[p[0]]||0)*(W-m.l-m.r)/mx;if(w>0){el("rect",{x:x0,y:y0+5,width:Math.max(w-2,1),height:rowH-10,rx:3,fill:"var(--"+p[0]+")"},s);x0+=w}});
    el("text",{x:x0+6,y:y0+rowH/2+4},s).textContent=(tot[i]/60).toFixed(1)+"m";
    var hit=el("rect",{x:0,y:y0,width:W,height:rowH,fill:"transparent"},s);
    hit.addEventListener("mousemove",function(e){setCur(r.round,e)});hit.addEventListener("mouseleave",hideTip);hit.addEventListener("click",function(e){pinned=!pinned;setCur(r.round,e)})});
}
/* chart 3: phase timeline on the wall clock, startup segment first */
function chartTl(){
  var ex=V.islands[SEL]||{},box=$("cTl");box.innerHTML="";
  $("legTl").innerHTML=PH.map(function(p){return '<span><i class="sw" style="background:var(--'+p[0]+')"></i>'+p[1]+'</span>'}).join("")+'<span><i class="sw" style="background:var(--line)"></i>启动（加载权重、组 Ray、起推理引擎）</span>';
  var segs=[];RS.forEach(function(r){Object.keys(r.phases).forEach(function(k){segs.push([k,r.phases[k][0],r.phases[k][1],r.round])})});
  if(!segs.length){box.innerHTML='<div class="note">无阶段数据</div>';return}
  var t0=Math.min.apply(null,segs.map(function(q){return q[1]}).concat(fin(ex.first_ts)?[ex.first_ts]:[])),t1=Math.max.apply(null,segs.map(function(q){return q[2]}));
  var W=1000,H=84,m={l:56,r:10,t:4,b:22},sx=function(t){return m.l+(t-t0)*(W-m.l-m.r)/((t1-t0)||1)};
  var s=el("svg",{viewBox:"0 0 "+W+" "+H,role:"img","aria-label":"阶段时间线"},box);
  el("text",{x:m.l-8,y:m.t+16,"text-anchor":"end"},s).textContent="推理";el("text",{x:m.l-8,y:m.t+42,"text-anchor":"end"},s).textContent="训练侧";
  if(fin(ex.first_ts)&&fin(ex.driver_start_ts))el("rect",{x:sx(ex.first_ts),y:m.t,width:Math.max(sx(ex.driver_start_ts)-sx(ex.first_ts),1),height:48,fill:"var(--line)",rx:3},s);
  segs.forEach(function(q){var rc=el("rect",{x:sx(q[1]),y:q[0]==="R"?m.t:m.t+26,width:Math.max(sx(q[2])-sx(q[1])-1,1),height:22,rx:3,fill:"var(--"+q[0]+")","class":"tseg","data-r":q[3]},s);
    rc.addEventListener("mousemove",function(e){setCur(q[3],e)});rc.addEventListener("mouseleave",hideTip);rc.addEventListener("click",function(e){pinned=!pinned;setCur(q[3],e)})});
  for(var j=0;j<=6;j++){var t=t0+(t1-t0)*j/6;el("text",{x:sx(t),y:H-6,"text-anchor":"middle"},s).textContent=clock(t)}
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
  var h='<div class="h">第 '+r.round+' 轮 · 训练用策略 v'+r.policy_version+' → 发布 '+(r.published_version==null?"无":"v"+r.published_version)+'</div>';
  if(kind()==="multi_island"){return h+(O.islands||[]).map(function(c){var rr=((V.islands[c.id]||{}).rounds||[]).find(function(x){return x.round===r.round});
    return row('<i class="sw" style="background:'+islColor(c.id)+'"></i>岛 '+esc(c.id),rr?f(rr.reward)+" · 截断 "+pct(rr.trunc):"缺席")}).join("")}
  return h+row("reward 均值",f(r.reward))+row("p10 / p50 / p90",f(r.p10,2)+" / "+f(r.p50,2)+" / "+f(r.p90,2))+row("截断率",pct(r.trunc))+
    row("logprob 差",f(r.logprob_diff,4))+row("回答长度 均值 / p95",f(r.resp_mean,0)+" / "+f(r.resp_p95,0))+row("grad_norm",f(r.grad_norm,4))+
    PH.map(function(p){return row('<i class="sw" style="background:var(--'+p[0]+')"></i>'+p[1],fin(r.dur[p[0]])?f(r.dur[p[0]],0)+" s":"无")}).join("")}
function showTip(e){var r=RS.find(function(x){return x.round===cur});if(!r)return;var t=$("tip"),p=$("rounds").getBoundingClientRect();
  t.innerHTML=tipHtml(r);t.style.display="block";var x=e.clientX-p.left+14,y=e.clientY-p.top+14,w=t.offsetWidth,h=t.offsetHeight;
  if(x+w>p.width-8)x=e.clientX-p.left-w-14;if(y+h>p.height-8)y=Math.max(8,e.clientY-p.top-h-14);t.style.left=Math.max(8,x)+"px";t.style.top=y+"px"}
function hideTip(){if(!pinned)$("tip").style.display="none"}
$("rounds").addEventListener("keydown",function(e){if(!RS.length)return;var ids=RS.map(function(r){return r.round}),i=ids.indexOf(cur);
  if(e.key==="ArrowRight")i=Math.min(ids.length-1,i+1);else if(e.key==="ArrowLeft")i=Math.max(0,i<0?ids.length-1:i-1);else if(e.key==="Escape"){pinned=false;cur=null;$("tip").style.display="none";highlight();return}else return;
  e.preventDefault();cur=ids[i];highlight();var t=$("tip");t.innerHTML=tipHtml(RS[i]);t.style.display="block";t.style.left="16px";t.style.top="48px"});
/* generation-stage utilization (agentic-rollout-utilization 7). Missing values -> "未知"; no data -> hidden / "无数据". */
var uRound="all",uMode="side";
var UPH=[["gen","模型生成","var(--R)"],["tool","工具执行","var(--i2)"],["judge","判分","var(--T)"],["sandbox","沙箱启动","var(--S)"]];
var RUN_COLORS=["var(--i0)","var(--i1)","var(--i2)","var(--i3)"];
function unk(v,d){return fin(v)?f(v,d==null?0:d):"未知"}
function utilRuns(){
  var runs=[{label:V.label||(V.compare?"A":""),run:O.run,util:(V.islands[SEL]||{}).util}];
  (V.compare||[]).forEach(function(c){var u=c.util||{},k=u[SEL]?SEL:Object.keys(u)[0];runs.push({label:c.label,run:c.run,util:k!=null?u[k]:null})});
  return runs}
function runHead(r,i,n){return n>1?'<div class="rl"><i class="sw" style="background:'+RUN_COLORS[i]+'"></i>'+esc(r.label)+' · <span class="mono">'+esc(r.run||"未命名运行")+'</span></div>':""}
function util(){
  var runs=utilRuns(),has=runs.some(function(r){return r.util});
  $("util").hidden=false;$("utilEmpty").hidden=has;$("utilBody").hidden=!has;
  $("utilIsl").textContent=SEL!=null&&(O.islands||[]).length>1?"· 岛 "+SEL:"";
  if(!has){$("uCarryBox").hidden=true;return}
  var rids={};runs.forEach(function(r){((r.util||{}).rounds||[]).forEach(function(x){rids[x.round]=1})});
  var ids=Object.keys(rids).map(Number).sort(function(a,b){return a-b});
  if(uRound!=="all"&&ids.indexOf(uRound)<0)uRound="all";
  var btns=[["all","全部轮叠加"]].concat(ids.map(function(i){return [i,"第"+i+"轮"]}));
  var ctl=btns.map(function(b){return '<button class="btn'+(b[0]===uRound?" on":"")+'" data-u="'+b[0]+'">'+b[1]+'</button>'}).join("");
  if(runs.length>1)ctl+=' <button class="btn'+(uMode==="side"?" on":"")+'" data-mode="side">并排</button><button class="btn'+(uMode==="overlay"?" on":"")+'" data-mode="overlay">叠加</button>';
  $("utilCtl").innerHTML=ctl;
  Array.prototype.forEach.call($("utilCtl").querySelectorAll("button"),function(b){b.onclick=function(){
    if(b.dataset.mode)uMode=b.dataset.mode;else uRound=b.dataset.u==="all"?"all":+b.dataset.u;util()}});
  uCut(runs);uDone(runs);uPh(runs);uTr(runs);uLoad(runs);uCarry(runs);
}
function uCut(runs){
  $("uCut").innerHTML=runs.map(function(r,i){var rr=((r.util||{}).rounds||[]);
    if(!rr.some(function(x){return x.cutoff}))return '<div>'+runHead(r,i,runs.length)+'<div class="note">无数据（没有 rl_rollout_cutoff）</div></div>';
    return '<div>'+runHead(r,i,runs.length)+'<table class="c"><tr><th>轮</th><th>生成段 s</th><th>提交组</th><th>目标组</th><th>丢弃组</th><th>丢弃条</th><th>丢弃 token</th><th>过滤组</th></tr>'+
      rr.map(function(x){var c=x.cutoff||{};return '<tr'+(x.round===uRound?' class="hl"':"")+'><td class="num">'+x.round+'</td><td class="num">'+unk(x.gen_s)+'</td><td class="num">'+unk(c.submitted_groups)+'</td><td class="num">'+unk(c.target_groups)+
        '</td><td class="num">'+unk(c.discarded_groups)+'</td><td class="num">'+unk(c.discarded_trajectories)+'</td><td class="num">'+unk(c.discarded_tokens)+'</td><td class="num">'+unk(c.filtered_groups)+'</td></tr>'}).join("")+'</table></div>'}).join("");
}
function uSel(rr){return uRound==="all"?rr:rr.filter(function(x){return x.round===uRound})}
function uDone(runs){
  var box=$("uDone");box.innerHTML="";
  var groups=uMode==="overlay"&&runs.length>1?[runs.map(function(r,i){return [r,i]})]:runs.map(function(r,i){return [[r,i]]});
  $("legDone").innerHTML=runs.length>1?runs.map(function(r,i){return '<span><i class="sw" style="background:'+RUN_COLORS[i]+'"></i>'+esc(r.label)+'</span>'}).join(""):(uRound==="all"?'<span>每条线一轮，越深越晚</span>':"");
  groups.forEach(function(g){
    var div=document.createElement?document.createElement("div"):null,host=div||box;if(div)box.appendChild(div);
    var lines=[];g.forEach(function(ri){uSel(((ri[0].util||{}).rounds||[])).forEach(function(x,k,arr){if(x.done.length||fin(x.gen_s))lines.push({run:ri[1],round:x.round,done:x.done,cut:x.gen_s,k:k,n:arr.length})})});
    var head=g.length===1?runHead(g[0][0],g[0][1],runs.length):"";
    if(!lines.length){host.innerHTML=head+'<div class="note">无数据（没有轨迹起止时间）</div>';return}
    if(div)div.innerHTML=head;
    var W=560,H=210,m={l:40,r:12,t:10,b:26};
    var xm=Math.max.apply(null,lines.map(function(l){return Math.max(fin(l.cut)?l.cut:0,l.done.length?l.done[l.done.length-1]:0)}).concat([1]));
    var ym=Math.max.apply(null,lines.map(function(l){return l.done.length}).concat([1]));
    var sx=function(v){return m.l+v*(W-m.l-m.r)/xm},sy=function(v){return m.t+(1-v/ym)*(H-m.t-m.b)};
    var s=el("svg",{viewBox:"0 0 "+W+" "+H,role:"img","aria-label":"轨迹完成曲线"},host),gg=el("g",{"class":"grid"},s);
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
    if(!rr.length){div.innerHTML=runHead(r,i,runs.length)+'<div class="note">无数据（轨迹事件没有四段耗时字段）</div>';return}
    div.innerHTML=runHead(r,i,runs.length);
    var W=480,rowH=24,m={l:52,r:110,t:4,b:4},H=m.t+m.b+rowH*rr.length;
    var mx=Math.max.apply(null,rr.map(function(x){return UPH.reduce(function(a,p){return a+(x.phases[p[0]]||0)},0)}).concat([1]));
    var s=el("svg",{viewBox:"0 0 "+W+" "+H,role:"img","aria-label":"四段耗时"},div);
    rr.forEach(function(x,k){var y0=m.t+k*rowH;if(x.round===uRound)el("rect",{x:0,y:y0,width:W,height:rowH,fill:"var(--band)"},s);
      el("text",{x:m.l-8,y:y0+rowH/2+4,"text-anchor":"end"},s).textContent="第"+x.round+"轮";
      var xe=phBar(s,y0+5,rowH-10,x.phases,W,m,mx);
      el("text",{x:xe+6,y:y0+rowH/2+4},s).textContent="工具 "+(fin(x.tool_share)?(x.tool_share*100).toFixed(1)+"%":"未知")+" · "+x.trajectories+" 条"});
  });
}
function uTr(runs){
  var box=$("uTr");box.innerHTML="";
  if(uRound==="all"){$("uTrNote").textContent="（先在上方选一轮）";return}
  $("uTrNote").textContent="（第 "+uRound+" 轮，按总时长排序；横条长度 = 四段耗时之和）";
  runs.forEach(function(r,i){var div=document.createElement?document.createElement("div"):box;if(div!==box)box.appendChild(div);
    var tr=((r.util||{}).trajectories||[]).filter(function(t){return t.rid===uRound}).sort(function(a,b){return (b.dur||0)-(a.dur||0)});
    if(!tr.length){div.innerHTML=runHead(r,i,runs.length)+'<div class="note">无数据</div>';return}
    div.innerHTML=runHead(r,i,runs.length);
    var W=480,rowH=14,m={l:8,r:70,t:2,b:2},H=m.t+m.b+rowH*tr.length;
    var mx=Math.max.apply(null,tr.map(function(t){return UPH.reduce(function(a,p){return a+(t[p[0]]||0)},0)}).concat([1]));
    var s=el("svg",{viewBox:"0 0 "+W+" "+H,role:"img","aria-label":"每条轨迹四段耗时"},div);
    tr.forEach(function(t,k){var y0=m.t+k*rowH,xe=phBar(s,y0+2,rowH-4,t,W,m,mx);
      var tl=el("title",{},el("rect",{x:0,y:y0,width:W,height:rowH,fill:"transparent"},s));
      tl.textContent=(t.task||"")+" · 生成 "+unk(t.gen,1)+" s · 工具 "+unk(t.tool,1)+" s · 判分 "+unk(t.judge,1)+" s · 沙箱 "+unk(t.sandbox,1)+" s · reward "+unk(t.reward,2);
      el("text",{x:xe+4,y:y0+rowH-3},s).textContent=fin(t.tool)&&fin(t.gen)?"工具 "+(t.tool/((t.gen||0)+(t.tool||0)+(t.judge||0)+(t.sandbox||0))*100).toFixed(0)+"%":"未知"});
  });
}
function uLoad(runs){
  var box=$("uLoad");box.innerHTML="";
  $("legLoad").innerHTML='<span><i class="sw" style="background:var(--T)"></i>KV 占用 %</span><span><i class="sw" style="background:var(--R)"></i>排队请求</span><span><i class="sw" style="background:var(--band)"></i>生成段</span>';
  runs.forEach(function(r,i){var div=document.createElement?document.createElement("div"):box;if(div!==box)box.appendChild(div);
    var u=r.util||{},ld=(u.load||[]).filter(function(p){return fin(p[0])});
    if(!ld.length){div.innerHTML=runHead(r,i,runs.length)+'<div class="note">无数据（没有 rl_load_sample）</div>';return}
    div.innerHTML=runHead(r,i,runs.length)+(u.load_has_kv?"":'<div class="note">KV：这份磁带未报（SGLang 没报 kv_used_tokens）</div>');
    var t0=ld[0][0],t1=ld[ld.length-1][0];
    var spans=(u.rounds||[]).filter(function(x){return x.gen}).map(function(x){return [x.gen[0],x.gen[1],x.round]});
    if(uRound!=="all"){var sp=spans.filter(function(q){return q[2]===uRound})[0];if(sp){var pad=(sp[1]-sp[0])*.1;t0=sp[0]-pad;t1=sp[1]+pad;ld=ld.filter(function(p){return p[0]>=t0&&p[0]<=t1})}}
    var kvm=Math.max.apply(null,ld.map(function(p){return fin(p[1])?p[1]:0}).concat([.01])),qm=Math.max.apply(null,ld.map(function(p){return fin(p[2])?p[2]:0}).concat([1]));
    var W=560,H=180,m={l:40,r:36,t:8,b:22},sx=function(t){return m.l+(t-t0)*(W-m.l-m.r)/((t1-t0)||1)};
    var syk=function(v){return m.t+(1-v/kvm)*(H-m.t-m.b)},syq=function(v){return m.t+(1-v/qm)*(H-m.t-m.b)};
    var s=el("svg",{viewBox:"0 0 "+W+" "+H,role:"img","aria-label":"KV 占用与排队"},div),g=el("g",{"class":"grid"},s);
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
    return '<div>'+runHead(r,i,runs.length)+(c.length?'<table class="c"><tr><th>轨迹</th><th>题目</th><th>开始轮 → 训练轮</th><th>版本段</th></tr>'+c.slice(-50).map(function(x){
      return '<tr><td class="mono">'+esc(String(x.id||"").slice(-10))+'</td><td>'+esc(x.task||"")+'</td><td class="num">'+(x.from==null?"未知":x.from)+' → '+(x.to==null?"未知":x.to)+'</td><td class="mono">'+esc((x.versions||[]).map(function(v){return Array.isArray(v)?"v"+v[0]+"["+v[1]+","+v[2]+")":"v"+v}).join(" "))+'</td></tr>'}).join("")+'</table>':'<div class="note">无续跑轨迹</div>')+'</div>'}).join("");
}
/* nodes */
function chartNodes(){
  var ex=V.islands[SEL]||{},ns=ex.node_series||{},ks=Object.keys(ns).sort(),box=$("cNodes");box.innerHTML="";$("legNodes").innerHTML="";
  if(!ks.length){box.innerHTML='<div class="note">未采样（没有主机探针数据）</div>';$("nodesNote").textContent="";return}
  var col=function(k){return k==="0"?"var(--T)":k==="1"?"var(--R)":"var(--i2)"};
  $("legNodes").innerHTML=ks.map(function(k){return '<span><i class="sw" style="background:'+col(k)+'"></i>节点 '+esc(k)+(k==="0"?"（驱动所在）":"")+'</span>'}).join("");
  var all=[];ks.forEach(function(k){ns[k].forEach(function(p){all.push(p)})});
  var t0=Math.min.apply(null,all.map(function(p){return p[0]})),t1=Math.max.apply(null,all.map(function(p){return p[0]})),vm=Math.max.apply(null,all.map(function(p){return p[1]}).concat([1]));
  var W=560,H=190,m={l:40,r:10,t:8,b:22},sx=function(t){return m.l+(t-t0)*(W-m.l-m.r)/((t1-t0)||1)},sy=function(v){return m.t+(1-v/vm)*(H-m.t-m.b)};
  var s=el("svg",{viewBox:"0 0 "+W+" "+H,role:"img","aria-label":"每节点显存"},box),g=el("g",{"class":"grid"},s);
  [0,.5,1].forEach(function(q){el("line",{x1:m.l,x2:W-m.r,y1:sy(vm*q),y2:sy(vm*q)},g);el("text",{x:m.l-6,y:sy(vm*q)+4,"text-anchor":"end"},s).textContent=(vm*q).toFixed(0)});
  ks.forEach(function(k){el("path",{d:"M"+ns[k].map(function(p){return sx(p[0])+","+sy(p[1])}).join("L"),fill:"none",stroke:col(k),"stroke-width":2},s)});
  [t0,(t0+t1)/2,t1].forEach(function(t){el("text",{x:sx(t),y:H-6,"text-anchor":"middle"},s).textContent=clock(t)});
  var hasU=ks.some(function(k){return ns[k].some(function(p){return fin(p[2])})});
  $("nodesNote").textContent="来源：每节点主机探针。"+(hasU?"":"GPU 利用率：这份磁带未采样（旧探针只记显存）。");
}
function islands(){
  $("islands").innerHTML=(O.islands||[]).map(function(c){var ex=V.islands[c.id]||{},re=ex.ray_embed||{};
    return '<details class="isl"'+(c.id===SEL?" open":"")+'><summary><span><b>岛 '+esc(c.id)+'</b> <span class="note">'+esc(c.name||"")+'</span></span>'+negBadge(c)+stBadge(c.status)+'</summary>'+
    '<div class="kv"><span>云 / 卡</span><span class="num">'+esc(c.cloud||"无")+' · '+esc(c.gpu||"无")+' × '+esc(c.gpus==null?"无":c.gpus)+'</span>'+
    '<span>阶段 / 策略</span><span class="num">'+esc(c.phase||"无")+' · '+(c.policy_version==null?"无":"v"+esc(c.policy_version))+'</span>'+
    '<span>最后事件</span><span class="num">'+(fin(c.last_event_age_s)?f(c.last_event_age_s,0)+" s 前":"无")+'</span>'+
    (c.starting?'<span>启动已用</span><span class="num">'+hm(c.startup_s)+'</span>':"")+
    (Object.keys(c.startup_steps||{}).length?'<span>启动子步骤</span><span class="num">'+
      Object.keys(c.startup_steps).map(function(k){var v=c.startup_steps[k];
        return esc({ray_connected:"Ray 集群组好",engine_ready:"推理引擎就绪",weights_loaded:"训练权重加载完"}[k]||k)+" "+(fin(v.step_s)?f(v.step_s,0)+" s":"?")}).join(" · ")+
      (c.starting&&fin(c.startup_step_age_s)?" · 当前步已等 "+f(c.startup_step_age_s,0)+" s":"")+'</span>':"")+
    (c.stopped_by_us?'<span>停机</span><span>我方停机（'+esc((O.operator_stop||{}).cause)+'，依据 '+esc((O.operator_stop||{}).marker)+'）</span>':"")+
    (c.nodes||[]).map(function(n){var pk=Math.max.apply(null,(n.gpu_mem_used_mib_peak||[0]).concat([0]));
      return '<span>节点 '+esc(n.node)+'</span><span class="num">显存峰值 '+(pk/1024).toFixed(1)+' GiB · 利用率 '+(fin(n.gpu_util_pct)?f(n.gpu_util_pct,0)+"%":"未采样")+'</span>'}).join("")+
    (ex.hardware?'<span>卡型 / 驱动 / CUDA</span><span>'+esc(ex.hardware.compat_group||"未声明")+' / '+esc(ex.hardware.driver_version||"未记录")+' / '+esc(ex.hardware.cuda_version||"未记录")+'</span>':"")+
    '<span>cell / 事务</span><span class="num">'+esc(ex.cells||0)+' / '+esc(ex.transactions||0)+'</span>'+
    '<span>Ray 面板</span><span>'+esc(re.note||(re.command?re.command:"本机 :"+re.port))+'</span></div></details>'}).join("")||'<div class="note">无数据</div>';
}
function cost(){
  var c=O.cost||{};
  $("costBox").innerHTML='<table class="c"><tr><th></th><th>GPU/h</th><th>CPU+内存/h</th><th>合计/h</th><th>时长</th><th>累计</th></tr>'+(c.islands||[]).map(function(r){
    return '<tr><td>岛 '+esc(r.id)+'</td><td class="num">'+(fin(r.gpu_rate_usd_h)?"$"+f(r.gpu_rate_usd_h,2):"未定价")+'</td><td class="num">'+(fin(r.host_rate_usd_h)?"$"+f(r.host_rate_usd_h,2):"仅 GPU")+'</td><td class="num">'+(fin(r.rate_usd_h)?"$"+f(r.rate_usd_h,2):"无")+'</td><td class="num">'+f(r.hours,2)+' h</td><td class="num">'+(fin(r.cost_usd)?"$"+f(r.cost_usd,1):"无")+'</td></tr>'}).join("")+'</table>'+
    '<div class="note" style="margin-top:6px">'+esc(c.note||"")+'。规格：'+(c.islands||[]).map(function(r){return "岛 "+esc(r.id)+" "+(r.cpus==null?"无":r.cpus)+" 核、"+(r.memory_gib==null?"无":r.memory_gib)+" GiB"}).join("；")+'。价目：'+esc(c.prices_source||"")+'</div>';
}
var EV_KEEP={rl_engine_selected:1,rl_driver_start:1,rl_driver_phase:1,rl_publication:1,rl_round_trained:1,rl_reconfiguration:1,rl_learner_finalized:1,
  island_ready:1,island_lost:1,island_stop:1,operator_stop:1,dashboard_source_lost:1,rl_elastic_recommendation:1};
var EVENTS=[],EV_CURSOR=0;
function evLevel(t,rec){if(t==="operator_stop")return ["mu","停机"];if(t==="island_lost"||(t==="rl_reconfiguration"&&rec.result==="RECOVERY_REQUIRED"))return ["bd","故障"];
  if(t==="rl_publication"||t==="rl_round_trained")return ["ok","完成"];if(t==="rl_reconfiguration"||t==="dashboard_source_lost")return ["wn","变更"];return ["mu","信息"]}
function evText(e){var r=e.record||{},keys=["phase","rollout_id","policy_version","result","cause","recommendation","marker"];
  return keys.filter(function(k){return r[k]!=null}).map(function(k){return k+"="+String(r[k]).slice(0,80)}).join(" ")}
function events(){
  var list=EVENTS.filter(function(e){return EV_KEEP[e.type]}).slice(-200).reverse();
  $("evs").innerHTML=list.map(function(e){var l=evLevel(e.type,e.record||{});return '<div><span class="t">'+clock(e.ts)+'</span><span class="st '+l[0]+'" title="'+l[1]+'"><i></i>'+l[1]+'</span><span class="x"><b>'+esc(e.type.replace(/^rl_/,""))+'</b>'+(e.island!=null?" 岛 "+esc(e.island):"")+' '+esc(evText(e))+'</span></div>'}).join("")||'<div class="note">无</div>';
}
function cmds(){
  var run=O.run||"<run>",list=[["本地看板","yeto dashboard serve --run "+run+" --port 8787"],["导出离线页","yeto dashboard export --run "+run+" -o "+run+".html"],
    ["运行状态","yeto status "+run]];
  if(((O.islands||[])[0]||{}).cloud==="modal")list.push(["Modal 日志","modal app logs yeto-"+run],["停止应用","modal app stop --yes yeto-"+run]);
  $("cmdBox").innerHTML=list.map(function(c,i){return '<div class="cmd"><span class="l">'+esc(c[0])+'</span><code id="cmd'+i+'">'+esc(c[1])+'</code><button class="btn" data-c="'+i+'">复制</button></div>'}).join("");
  Array.prototype.forEach.call($("cmdBox").querySelectorAll("button"),function(b){b.onclick=function(){var t=$("cmd"+b.dataset.c).textContent;
    var ok=function(){b.textContent="已复制";setTimeout(function(){b.textContent="复制"},1200)},bad=function(){b.textContent="请手动复制"};
    if(navigator.clipboard&&navigator.clipboard.writeText)navigator.clipboard.writeText(t).then(ok,bad);else bad()}});
}
function folds(){
  var u=V.usage||{},multi=(O.islands||[]).length>1,items=[];
  if(!u.syncer)items.push(["Syncer 合并时间线 / round 表","无 syncer 磁带"]);
  if(!u.transactions)items.push(["E1 重配置事务","无 journal 事务"]);
  if(!multi)items.push(["岛间弹性 / 租约","只有一个岛"]);
  if((O.islands||[]).every(function(c){return (c.cloud||"")==="modal"}))items.push(["Ray 面板嵌入","Modal 容器无入站端口"]);
  $("foldBox").innerHTML=items.map(function(x){return '<div class="fold"><span>'+x[0]+'</span><span>本次运行未使用（'+x[1]+'）</span></div>'}).join("")||'<div class="note">无</div>';
}
function spot(){
  var rows=(V.spot||[]).slice(-50).reverse();$("spotBox").hidden=!rows.length;if(!rows.length)return;
  $("spot").innerHTML='<table class="c"><tr><th>事件</th><th>岛</th><th>云 / 区域</th><th>来源</th><th>剩余秒</th><th>已保存</th><th>结果</th><th>候选/拒绝</th></tr>'+rows.map(function(r){
    return '<tr><td>'+esc(r.event)+'</td><td>'+esc(r.island)+'</td><td class="mono">'+esc(r.cloud||"无")+' / '+esc(r.region||"无")+'</td><td>'+esc(r.source||"")+'</td><td class="num">'+f(r.remaining_s,1)+'</td><td>'+(r.saved==null?"":(r.saved?"是":"否"))+'</td><td>'+esc(r.outcome||(r.auto_launch===false?"只出建议":""))+'</td><td class="num">'+(r.candidates==null?"":esc(r.candidates)+' / '+esc(r.rejected))+'</td></tr>'}).join("")+'</table>'}
function render(){
  if(!V)return;O=V.overview;
  var ids=(O.islands||[]).map(function(c){return c.id});if(ids.indexOf(SEL)<0)SEL=ids[0]||null;
  RS=SEL!=null?((V.islands[SEL]||{}).rounds||[]):[];
  $("roundsIsl").textContent=SEL!=null&&ids.length>1?"· 岛 "+SEL:"";
  header();kpis();wall();chartMetric();chartDur();chartTl();highlight();util();chartNodes();islands();cost();events();cmds();folds();spot();
}
function load(){
  if(OFFLINE){V=INLINE.page;EVENTS=(INLINE.events||{}).events||[];render();return}
  Promise.all([fetch("/api/view",{cache:"no-store"}).then(function(r){return r.json()}),
    fetch("/api/events?after="+EV_CURSOR+"&limit=1000",{cache:"no-store"}).then(function(r){return r.json()})]).then(function(x){
    V=x[0];EVENTS=EVENTS.concat(x[1].events||[]).slice(-3000);EV_CURSOR=x[1].cursor||EV_CURSOR;render()}).catch(function(e){$("sub").textContent="拉取失败："+e})
}
load();if(!OFFLINE)setInterval(load,POLL_MS);
})();
