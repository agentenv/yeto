// Breakers panel (page view key "breakers", breakers.py): per layer, each breaker's state, effective
// value and source, its headroom from rl_breaker_status, recent rl_breaker trips with their stop_reason,
// and the dependency-rule findings of breakers.resolved.json. Rows wrap instead of scrolling sideways.
var YetoBrk=(function(){
"use strict";
var LAYER_DOC={run:"run up monitor",launch:"launcher / cost guard",trainer:"trainer island",sampler:"samplers",
  harness:"trial harness",pool:"sampler pools",monitor:"dashboard / health alerts"};
var STATE_LABEL={armed:"armed",implicit:"implicit",off:"off",invariant:"invariant"};
var STATE_CLS={armed:"ok",implicit:"wn",off:"mu",invariant:"mu"};
var TRIP_CLS={trip:"bd",warn:"wn",config_mismatch:"wn",shadow:"mu",clear:"ok"};
var open={};  // layer -> its collapsed rows shown (kept across the 5 s redraws)

function $(id){return document.getElementById(id)}
function esc(s){return String(s==null?"":s).replace(/[&<>"']/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]})}
function fin(x){return typeof x==="number"&&isFinite(x)}
function clock(t){if(!fin(t))return "n/a";var d=new Date(t*1000);function p(n){return (n<10?"0":"")+n}return p(d.getHours())+":"+p(d.getMinutes())+":"+p(d.getSeconds())}
function dur(s){if(!fin(s))return "n/a";if(s<120)return (Math.round(s*10)/10)+" s";if(s<7200)return Math.round(s/60)+" min";return (Math.round(s/360)/10)+" h"}
/* one value in its unit: seconds as a duration, dollars, ratios as percent, counts as integers */
function fmt(v,unit){if(v===null||v===undefined)return "none";if(typeof v==="boolean")return v?"on":"off";if(!fin(v))return esc(v);
  if(unit==="s")return dur(v);if(unit==="usd")return "$"+(Math.abs(v)>=100?v.toFixed(0):v.toFixed(2));if(unit==="ratio")return Math.round(v*1000)/10+"%";
  if(unit==="h")return v+" h";return String(Math.round(v*1000)/1000)}
function unitOf(name){return /_s$/.test(name)?"s":/usd$/.test(name)?"usd":/ratio$/.test(name)?"ratio":/_h$|^hours$/.test(name)?"h":"n"}
function params(r){var ks=Object.keys(r.params||{});if(!ks.length)return "";
  return ks.map(function(k){var u=(r.units||{})[k]||unitOf(k);return '<span class="bp"><span class="bk">'+esc(k)+'</span> '+esc(fmt(r.params[k],u))+'</span>'}).join(" ")}
function source(s){if(!s)return '<span class="note">source n/a</span>';
  var kind=s.kind==="legacy"?"alias":s.kind==="breakers"?"breakers section":s.kind;
  return '<span class="bsrc bs-'+esc(s.kind)+'" title="'+esc(s.location)+'">'+esc(kind)+'</span> <span class="note bl" title="'+esc(s.location)+'">'+esc(s.location)+'</span>'}
function bar(g,tripped){if(!g||!fin(g.ratio))return "";var r=Math.max(0,g.ratio),w=Math.min(100,r*100);
  var cls=tripped||r>=0.9?"r":r>=0.7?"y":"g";
  return '<div class="bbar" title="'+esc(fmt(g.value,g.unit)+" of "+fmt(g.limit,g.unit))+'"><div class="bbf '+cls+'" style="width:'+w.toFixed(1)+'%"></div></div>'}
function gauge(r){var g=r.gauge;if(!g)return '<span class="note">no live value</span>';
  var txt=fin(g.limit)?fmt(g.value,g.unit)+" / "+fmt(g.limit,g.unit)+(fin(g.ratio)?" ("+Math.round(g.ratio*100)+"%)":""):fmt(g.value,g.unit)+(g.gauge==="count"?" counted":"");
  return '<span class="num">'+esc(txt)+'</span>'+(g.scope?' <span class="note">'+esc(g.scope)+(g.scopes>1?" (worst of "+g.scopes+")":"")+'</span>':"")+(g.detail?' <span class="note">· '+esc(g.detail)+'</span>':"")}
function last(t){if(!t)return "";var cls=TRIP_CLS[t.state]||"mu";
  return '<span class="st '+cls+'"><i></i>'+esc(t.legacy?"legacy "+t.legacy_event:t.state)+'</span> <span class="note">'+esc(clock(t.ts))+(t.scope?" · "+esc(t.scope):"")+(t.stop_reason?" · "+esc(t.stop_reason):"")+'</span>'}
/* a guard that drops data acted as designed: amber count on its card; red once it drops too much */
function acted(r){var a=r.acted;if(!a)return "";var n=a.acts||0,rt=a.rate,cls=r.tripped?"bd":"wn";
  var parts=[];if(n)parts.push("acted "+n+" time"+(n===1?"":"s")+" · "+fmt(a.dropped,"n")+" "+esc(a.unit)+(fin(a.last_ts)?" · last at "+clock(a.last_ts):"")+(a.scope?" ("+esc(a.scope)+")":""));
  if(rt)parts.push(Math.round(rt.ratio*1000)/10+"% of "+fmt(rt.total,"n")+" groups over the last "+rt.rounds+" rounds"+(rt.scope?" ("+esc(rt.scope)+")":""));
  if(!n&&!r.tripped)return parts.length?'<div class="bt2"><span class="note">no drops · '+parts.join(" · ")+'</span></div>':"";
  return '<div class="bt2"><span class="st '+cls+'"><i></i>'+(r.tripped?"dropping too much":"acted")+'</span> <span class="note">'+parts.join(" · ")+(a.last_detail?" · "+esc(a.last_detail):"")+'</span></div>'}
function row(r){
  var st=r.state,mismatch=r.runtime_params?' <span class="st wn" title="'+esc(JSON.stringify(r.runtime_params))+'"><i></i>runs with other values</span>':"";
  return '<div class="brow'+(r.tripped?" btrip":"")+'">'+
    '<div class="bn"><span class="mono">'+esc(r.name)+'</span> <span class="st '+STATE_CLS[st]+'"><i></i>'+STATE_LABEL[st]+'</span>'+
      (r.action?' <span class="tag">'+esc(r.action)+'</span>':"")+(r.kind&&r.kind!=="breaker"?' <span class="note">'+esc(r.kind.replace("_"," "))+'</span>':"")+mismatch+'</div>'+
    '<div class="bv">'+(params(r)||'<span class="note">'+(st==="invariant"?"cannot be disabled":"no parameters")+'</span>')+'</div>'+
    '<div class="bg">'+bar(r.gauge,r.tripped)+gauge(r)+'</div>'+
    '<div class="bs">'+source(r.source)+(r.redundant&&r.redundant.length?' <span class="note">(repeated at '+esc(r.redundant.join(", "))+')</span>':"")+'</div>'+
    (r.last?'<div class="bt2">last: '+last(r.last)+(r.last.detail?' <span class="note">'+esc(r.last.detail)+'</span>':"")+'</div>':"")+acted(r)+
  '</div>'}
function layer(L){
  var main=L.rows.filter(function(r){return r.state==="armed"||r.state==="implicit"||r.tripped}),rest=L.rows.filter(function(r){return main.indexOf(r)<0});
  var c=L.counts,head='<div class="blh"><b>'+esc(L.layer)+'</b> <span class="note">'+esc(LAYER_DOC[L.layer]||"")+' · '+
    ["armed","implicit","off","invariant"].filter(function(k){return c[k]}).map(function(k){return c[k]+" "+k}).join(" · ")+'</span></div>';
  var more="";if(rest.length){var on=!!open[L.layer];
    more='<button class="btn bmore" data-layer="'+esc(L.layer)+'" aria-expanded="'+on+'">'+(on?"hide":"show")+" "+rest.length+" off / invariant</button>"+(on?rest.map(row).join(""):"")}
  return '<div class="blay">'+head+main.map(row).join("")+more+'</div>'}
function trips(B){var t=B.trips||[],g=B.acted_total?'<div class="note">guards that drop data (groups, trials, claims) acted '+B.acted_total+' time'+(B.acted_total===1?"":"s")+': counted on their cards, not listed here</div>':"";
  if(!t.length)return g+'<div class="note">no trips on the tapes</div>';
  return g+'<div class="btrips">'+t.slice(0,30).map(function(x){var cls=TRIP_CLS[x.state]||"mu";
    return '<div><span class="t">'+esc(clock(x.ts))+'</span><span class="x"><span class="st '+cls+'"><i></i>'+esc(x.state)+'</span> <span class="mono">'+esc(x.name)+'</span>'+
      (x.action?' <span class="tag">'+esc(x.action)+'</span>':"")+(x.scope?' <span class="note">'+esc(x.scope)+'</span>':"")+
      (fin(x.value)&&fin(x.limit)?' <span class="num note">'+esc(fmt(x.value,x.unit))+" / "+esc(fmt(x.limit,x.unit))+(x.count>1?" ×"+x.count:"")+'</span>':"")+
      (x.stop_reason?'<div class="mono bsr">stop_reason='+esc(x.stop_reason)+'</div>':"")+
      (x.detail?'<div class="note">'+esc(x.detail)+'</div>':"")+(x.legacy?'<div class="note">legacy event '+esc(x.legacy_event)+' (no rl_breaker on this tape)</div>':"")+'</span></div>'}).join("")+'</div>'+
    (B.trip_total>30?'<div class="note">'+(B.trip_total-30)+' older not shown</div>':"")}
function findings(B){var f=B.findings||[];if(!f.length)return "";
  return '<h3>Dependency checks</h3><div class="bfind">'+f.map(function(x){var cls=x.level==="error"?"bd":x.level==="warn"?"wn":"mu";
    return '<div><span class="st '+cls+'"><i></i>'+esc(x.level)+'</span> <b>'+esc(x.rule)+'</b>'+(x.overridden?' <span class="tag">overridden: '+esc(x.overridden)+'</span>':"")+
      ' <span>'+esc(x.message)+'</span>'+((x.items||[]).length?' <span class="note mono">'+esc(x.items.join(", "))+'</span>':"")+'</div>'}).join("")+'</div>'}
function banners(B){var out=[];
  (B.conflicts||[]).forEach(function(c){out.push('<div class="banner bbad">conflict: '+esc(c.message)+((c.locations||[]).length?" ("+esc(c.locations.join(" vs "))+")":"")+'</div>')});
  if(B.budget_mismatch)out.push('<div class="banner">--budget $'+esc(B.budget_mismatch.cli)+' differs from the run config\'s run.budget.usd $'+esc(B.budget_mismatch.resolved)+'; the page uses --budget</div>');
  return out.join("")}
function switches(B){var s=B.switches||{},ks=Object.keys(s);if(!ks.length)return "";
  return '<div class="note bsw">switches: '+ks.map(function(k){var g=s[k]||{};return '<span class="mono">'+esc(k)+'</span> '+(g.on===false?'<b>off</b>':"on")+' ('+esc((g.members||[]).join(", "))+')'}).join(" · ")+'</div>'}
/* the dependency checks a viewer must not miss: warn and error level */
function loud(B){return (B.findings||[]).filter(function(x){return x.level==="warn"||x.level==="error"})}
/* the one line the collapsed panel shows: entry states, the newest trip, loud dependency checks */
function summary(B){var c=B.counts||{},t=(B.trips||[])[0],f=loud(B),ne=f.filter(function(x){return x.level==="error"}).length;
  var states=["armed","implicit","off","invariant"].filter(function(k){return c[k]}).map(function(k){return c[k]+" "+k}).join(" · ")||"no entries";
  var trip=t?'<span class="st '+(TRIP_CLS[t.state]||"mu")+'"><i></i>'+esc(B.trip_total)+" trip"+(B.trip_total===1?"":"s")+'</span> <span class="note">newest '+esc(t.name)+" "+esc(t.state)+" at "+esc(clock(t.ts))+'</span>':'<span class="note">no trips</span>';
  var dep=f.length?'<span class="st '+(ne?"bd":"wn")+'"><i></i>'+(ne?ne+" failing":"")+(ne&&f.length>ne?" · ":"")+(f.length>ne?(f.length-ne)+" warn":"")+" dependency check"+(f.length===1?"":"s")+'</span>':'<span class="note">no warn / fail dependency checks</span>';
  return '<b>'+esc(states)+'</b> · '+trip+' · '+dep}
/* Alerts rows for the top of the page: the loud dependency checks (trips are alerts already),
   each with a link down to this panel (it sits at the bottom, collapsed) */
function alertRows(V){var B=V&&V.breakers;if(!B)return "";
  return loud(B).map(function(x){return '<div><span class="st '+(x.level==="error"?"bd":"wn")+'"><i></i>'+(x.level==="error"?"critical":"warning")+'</span> <b>breaker dependency check '+esc(x.rule)+'</b> <span class="note">'+esc(x.message)+'</span> '+link()+'</div>'}).join("")}
function link(){return '<a href="#brkBox" class="brkjump" data-brk="1">Breakers ↓</a>'}
var detOpen=false;  // the viewer opened the panel (kept across the 5 s redraws)
function openPanel(){detOpen=true;var d=$("brkDet");if(d&&!d.open)d.open=true}
function render(V){
  var box=$("brkBox"),B=V&&V.breakers;if(!box)return;box.hidden=!B;var nav=$("navBrk");if(nav)nav.style.display=B?"":"none";if(!B)return;
  var c=B.counts||{};
  $("brkNote").textContent=["armed","implicit","off","invariant"].filter(function(k){return c[k]}).map(function(k){return c[k]+" "+k}).join(" · ")+
    (B.trip_total?" · "+B.trip_total+" trips":"")+(B.acted_total?" · "+B.acted_total+" guard acts":"")+" · "+(B.resolved_from?"settings from "+B.resolved_from.split("/").slice(-2).join("/"):B.status_scopes?"settings from "+B.status_scopes+" process report(s)":"events only");
  if(typeof location!=="undefined"&&location&&location.hash==="#brkBox")detOpen=true;
  $("brk").innerHTML='<details class="bdet" id="brkDet"'+(detOpen?" open":"")+'><summary class="bsum">'+summary(B)+' <span class="note">(click for every entry)</span></summary>'+
    banners(B)+switches(B)+'<div class="bgrid"><div>'+(B.layers||[]).map(layer).join("")+'</div><div><h3>Recent trips</h3>'+trips(B)+findings(B)+'</div></div></details>';
  var d=$("brkDet");if(d&&d.addEventListener)d.addEventListener("toggle",function(){detOpen=!!d.open});
  var bs=box.querySelectorAll?box.querySelectorAll(".bmore"):[];
  for(var i=0;i<bs.length;i++)bs[i].addEventListener("click",function(e){var k=e.currentTarget.getAttribute("data-layer");open[k]=!open[k];render(V)});
}
return {render:render,alertRows:alertRows,link:link,open:openPanel};
})();
