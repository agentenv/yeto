// Live topology of a run with a syncer (page view key topology.diloco, topology_diloco.py).
// One double lane per trainer / syncer shard pair: the upper (violet) lane carries delta
// PUSHes, the lower (teal) one fragment BCASTs, each particle labelled fN and replayed at its
// tape window DELAY_S behind the newest data, as in topology.js. kind "island" draws a group
// box per island (machine labels, rollout and trainer sub-cards, the in-island groups /
// snapshot lanes); kind "synced_group" draws the samplers and the store left of the trainers.
// A fleet of async islands (diloco.fleet, fleet_islands.py) draws one box per island: its trainer,
// one card per external pool (engine, rollout precision, card, provider) and its R2 lane trainer<k>/.
// The fragment matrix (P columns x syncer + trainers) sits under the graph.
var YetoTopoD=(function(){
"use strict";
var DELAY_S=45,BCAST_WARN_S=90,BCAST_BAD_S=300,ROUND_WARN_S=120;
var U=null,T=null,D=null,V=null,L=null,C=null,recvAt=0,raf=null,lastTick=null,acc=0,bound=false,v1Legend=null,fmKey="";

function $(id){return document.getElementById(id)}
function cfgBits(c){return c?[fin(c.tau)?"τ "+c.tau:null,fin(c.T)?"T "+c.T:null,fin(c.lr)?"lr "+sig(c.lr):null]:[]}
// the syncer config's float32 lr (0.6999999) at 4 significant figures (0.7)
function sig(x){return String(Number(x.toPrecision(4)))}
function fin(x){return typeof x==="number"&&isFinite(x)}
function dur(s){return U.dur(s)}
function mb(b){return U.bytes(b)}
function esc(s){return U.esc(s)}
function lr(x){return fin(x)?(x===0?"0":x.toExponential(0)):null}
function kn(v){return !fin(v)?"n/a":v>=1e9?(v/1e9).toFixed(1)+"B":v>=1e6?(v/1e6).toFixed(1)+"M":v>=1e4?(v/1e3).toFixed(1)+"k":v.toFixed(0)}
function now(){return T?T.now+(U.clock()-recvAt)/1000:0}
function colors(){var cs=typeof getComputedStyle==="function"?getComputedStyle(document.documentElement):null,o={},g=function(n){return cs&&cs.getPropertyValue?String(cs.getPropertyValue(n)||"").trim():""};
  ["ink","ink2","line","good","warn","bad","traj","wt","delta","bg"].forEach(function(k){o[k]=g("--"+k)});o.wire=g("--topo-wire");return o}
function arch(a){return a?"var(--arch-"+String(a).toLowerCase()+")":"var(--ink2)"}
function el(tag,attrs,parent,text){return U.el(tag,attrs,parent,text)}
function clip(s,n){s=String(s==null?"":s);return s.length>n?s.slice(0,Math.max(1,n-1))+"…":s}
// [[text, colour]] lines folded to maxW at their " · " seams (the continuation indented), at most
// n lines (the rest is cut by fitText on the last one)
function wrapLines(lines,maxW,n){var cw=6.3,cap=Math.max(8,Math.floor(maxW/cw)),out=[];
  lines.forEach(function(ln){var s=ln[0];
    while(s.length>cap){var cut=s.lastIndexOf(" · ",cap);if(cut<=2)break;out.push([s.slice(0,cut),ln[1]]);s="  "+s.slice(cut+3)}
    out.push([s,ln[1]])});
  if(out.length<=n)return out;
  // too many: keep each source line whole on one row first (fitText cuts it), in order
  return lines.slice(0,n)}
// a card line that never leaves its card: cut to maxW (svg units) with an ellipsis, the whole
// line on hover (<title>). Measured where the browser lays the text out, else by the monospace
// advance (10.5 px mono ≈ 6.3, the bold 12 px value font ≈ 7.3)
function fitText(e,s,maxW){s=String(s==null?"":s);if(e._full===s&&e._fitW===maxW)return;e._full=s;e._fitW=maxW;
  e.textContent=s;if(!s||!(maxW>0))return;
  var cw=/\btpv\b/.test(e.getAttribute&&e.getAttribute("class")||"")?7.3:6.3,len=null;
  try{if(typeof e.getComputedTextLength==="function")len=e.getComputedTextLength()}catch(_){len=null}
  if(!fin(len)||len<=0)len=s.length*cw;
  if(len<=maxW)return;
  var n=Math.max(1,Math.floor(s.length*maxW/len));
  for(var k=0;k<12&&n>1;k++){e.textContent=s.slice(0,n-1)+"…";var l2=null;
    try{if(typeof e.getComputedTextLength==="function")l2=e.getComputedTextLength()}catch(_){l2=null}
    if(!fin(l2)||l2<=0||l2<=maxW)break;n--}
  if(typeof document!=="undefined"&&document.createElementNS&&e.namespaceURI){var t=document.createElementNS(e.namespaceURI,"title");t.textContent=s;e.appendChild(t)}}
// rich text in an svg <text>: tspans in a browser, the plain text where innerHTML does nothing
function rich(e,parts){e.textContent=parts.map(function(p){return p[0]}).join("");
  if(typeof document.createElementNS==="function"&&e.namespaceURI)e.innerHTML=parts.map(function(p){return p[1]?'<tspan style="'+p[1]+'">'+esc(p[0])+'</tspan>':esc(p[0])}).join("")}

// -- reading the view --------------------------------------------------------------
function trainerState(tr,tn,isl){
  var age=fin(tr.phase_since)?tn-tr.phase_since:null,a=age!=null?" · "+dur(age):"",ph=tr.phase_raw,co=isl&&isl.colocated;
  if(tr.failure)return["failed: "+tr.failure.metric,"bad"];
  if(sft()){var q=fin(tr.last_event_ts)?dur(tn-tr.last_event_ts)+" ago":"n/a";
    return tr.live?["training · last step "+q,"good"]:tr.status==="done"?["finished","mute"]:tr.status==="stopped"?["stopped or restarting · last step "+q,"mute"]:["no step yet","mute"]}
  if(!tr.live){if(tr.status==="done"||tr.status==="stopped"||tr.finalized)return["finished","mute"];
    return[(tr.status||"lost")+(tr.reason?": "+tr.reason:"")+(fin(tr.last_event_ts)?" · no event for "+dur(tn-tr.last_event_ts):""),"bad"]}
  if(ph==="train")return["training"+a,"good"];
  if(ph==="sync"||ph==="apply"||ph==="drain_bcast"||ph==="drain_pull")return["safe boundary · apply + push","wt"];
  if(ph==="publish")return[isl?"publishing snapshot → rollout":"publishing","wt"];
  if(ph==="generate"){if(!isl)return["waiting for data"+a,age>30?"warn":"mute"];return[co?"offloaded · rollout on the GPU"+a:"idle · waiting for rollout"+a,"mute"]}
  if(ph)return[tr.phase+a,"mute"];
  return[tr.status==="stale"?"quiet":"running",tr.status==="stale"?"bad":"good"]}
function rolloutState(tr,tn){
  var r=tr.island,age=fin(tr.phase_since)?tn-tr.phase_since:null,ph=tr.phase_raw;
  if(!tr.live)return[tr.status==="done"||tr.finalized?"finished":"down with its island",tr.status==="done"||tr.finalized?"mute":"bad"];
  if(ph==="generate"){var est=fin(r.rollout_p50_s)&&age!=null?Math.min(99,Math.round(100*age/r.rollout_p50_s)):null;
    return["generating"+(est!=null?" · ~"+est+"% (estimate)":"")+(age!=null?" · "+dur(age):""),"good",est]}
  if(ph==="publish")return["loading the new snapshot"+(age!=null?" · "+dur(age):""),"wt"];
  if(ph==="onload")return["onloading weights","wt"];
  return[r.colocated?"offloaded · GPU is training":"idle · trainer step (serial)","mute"]}
function col(k){return{good:C.good,warn:C.warn,bad:C.bad,wt:C.wt,mute:C.ink2,delta:C.delta}[k]||C.ink2}
function shardOf(f){var sh=D.shards;for(var i=0;i<sh.length;i++)if(f>=sh[i].lo&&f<sh[i].hi)return i;return 0}
function behind(tr,f){return !D.done&&tr.versions&&fin(tr.versions[f])&&D.versions[f]>tr.versions[f]}
function lag(tr,f){return (tr.pending||{})[String(f)]||0}

// -- layout ------------------------------------------------------------------------
// the samplers that feed trainer tr through its own store lane (--rl-sampler-follow trainer)
function follow(){return D.kind==="synced_group"&&D.follow}
// decoupled DiLoCo SFT (topology_sft.py): learner islands, no rollout, no fragment-version vector
function sft(){return D.kind==="sft"}
function nh(){return sft()?62:86} // machine label pitch
function mine(tr){if(fleet()){var fi=fisland(tr);return fi?fi.pools.map(poolCard):[]}
  var ids=((D.lanes||{})[tr.id]||{}).samplers||[];return (T.samplers||[]).filter(function(s){return ids.indexOf(s.id)>=0})}
function snode(id){return (D.sampler_nodes||[]).filter(function(n){return n.id===id})[0]||{id:id}}
function laneName(tr){return (((D.lanes||{})[tr.id]||{}).substore||"trainer"+tr.id)+"/"}
// a fleet of async islands: each island's external pools, one card per pool (its samplers pooled)
function fleet(){return follow()&&D.fleet?D.fleet:null}
// a fleet with Miles sampler islands of the launch (source "miles") next to external pools (source "external")
var SRC={miles:["Miles native","var(--delta)"],external:["external pool","var(--traj)"]};
function fisland(tr){var F=fleet();return F?F.islands.filter(function(x){return x.trainer===tr.id})[0]||null:null}
function fpool(name){var F=fleet(),hit=null;if(F)F.islands.forEach(function(x){x.pools.forEach(function(p){if(p.name===name)hit=p})});return hit}
function poolCard(p){var ss=(T.samplers||[]).filter(function(s){return p.samplers.indexOf(s.id)>=0}),lv=ss.filter(function(s){return s.live}),f0=ss[0]||{};
  var vs=lv.map(function(s){return s.policy_version}).filter(fin),lg=lv.map(function(s){return s.lag}).filter(fin);
  var st=lv.some(function(s){return s.state==="sampling"})?"sampling":lv.length?(lv[0].state||"starting"):null;
  return{id:p.name,pool:p,ids:p.samplers,source:p.source||null,card:p.card||f0.card||null,precision:p.precision||f0.precision||null,engine:p.engine||f0.engine||null,arch:p.arch||f0.arch||null,
    gpus:f0.gpus||p.gpus||null,live:lv.length>0,state:st,status:ss.length?(f0.status||"ended"):"not started",policy_version:vs.length?Math.max.apply(null,vs):null,
    lag:lg.length?Math.max.apply(null,lg):null,rate_gpm:ss.reduce(function(a,x){return a+(x.rate_gpm||0)},0),n:p.samplers.length,nlive:lv.length}}
// the sampler (or pool) a card shows, from the newest view (a card is built once per layout)
function fresh(s){if(s.pool){var p=fpool(s.id);return p?poolCard(p):s}return (T.samplers||[]).filter(function(x){return x.id===s.id})[0]||s}
function layout(W){
  var narrow=W<640,trs=D.trainers,isl=D.kind==="island",fw=follow(),samp=D.kind==="synced_group"&&!fw?(T.samplers||[]):[],N=D.shards.length||1;
  var g={narrow:narrow,VW:narrow?400:1480,boxes:[],syn:[],samp:[],store:null};
  if(!narrow&&sft()&&trs.length>4)hub(g,trs);
  else if(!narrow){
    var x0=samp.length?420:10,lw=124,rx=x0+lw+26,rw=isl?262:0,tx=isl?rx+rw+124:fw?x0+lw+26+196+112:x0+lw+26,tw=isl?330:sft()||fw?340:290,th=sft()?170:212,y=14;
    trs.forEach(function(tr){var ms=fw?mine(tr):[],nl=((tr.nodes||[]).length||1)+ms.length,h=Math.max(th,nl*nh()-8,ms.length*64)+44,B;
      B={x:x0,y:y,w:tx+tw-x0+12,h:h,lx:x0+12,ly:y+34,lw:lw,rx:rx,ry:y+34,rw:rw,rh:h-44,tx:tx,ty:y+34,tw:tw,th:h-44,
        co:!!(tr.island&&tr.island.colocated),roll:isl&&!!tr.island,samp:[]};
      if(fw){var sy0=B.ty+(B.th-ms.length*64+8)/2;ms.forEach(function(sp,j){B.samp.push({x:rx,y:sy0+j*64,w:196,h:56,s:sp})});B.lane={cx:rx+196+56,cy:B.ty+B.th/2,r:30}}
      g.boxes.push(B);y+=h+16});
    var H=y,yw=336,yx=g.VW-yw-10,yh=N===1?318:250,yy=Math.max(14,(H-(N*(yh+16)-16))/2);
    for(var s=0;s<N;s++)g.syn.push({x:yx,y:yy+s*(yh+16),w:yw,h:yh});
    if(samp.length){var sh=62,sy=Math.max(14,H/2-(samp.length*(sh+10))/2);samp.forEach(function(sp,k){g.samp.push({x:10,y:sy+k*(sh+10),w:228,h:sh})});
      g.store={cx:330,cy:H/2,r:46};H=Math.max(H,sy+samp.length*(sh+10)+20)}
    g.VH=Math.max(H,yy+N*(yh+16)+44)}
  else{var yN=6;
    if(samp.length){samp.forEach(function(sp,k){g.samp.push({x:4,y:yN+k*62,w:250,h:56})});g.store={cx:330,cy:yN+Math.max(56,samp.length*62)/2,r:34};yN+=Math.max(110,samp.length*62+14)}
    trs.forEach(function(tr){var ms=fw?mine(tr):[],nl=((tr.nodes||[]).length||1)+ms.length,roll=isl&&!!tr.island,sh=ms.length*64,h=30+nl*62+(roll?182:0)+sh+(fw?20:0)+240,B;
      var ry=yN+28+nl*62;
      B={x:4,y:yN,w:392,h:h,lx:12,ly:yN+28,lw:376,rx:12,ry:ry,rw:376,rh:172,tx:12,ty:ry+(roll?182:0)+sh+(fw?20:0),tw:376,th:sft()?170:196,
        co:!!(tr.island&&tr.island.colocated),roll:roll,samp:[]};
      if(fw){ms.forEach(function(sp,j){B.samp.push({x:12,y:ry+j*64,w:300,h:56,s:sp})});B.lane={cx:358,cy:ry+Math.max(sh,56)/2,r:27}}
      g.boxes.push(B);yN+=h+12});
    for(var q=0;q<N;q++){g.syn.push({x:4,y:yN+30,w:392,h:N===1?300:236});yN+=(N===1?300:236)+36}
    g.VH=yN+36}
  return g}
// many SFT learner islands: the syncer in the middle, islands left and right of it (the right
// column mirrored: card first, machine labels outside), lane labels folded into the cards
function hub(g,trs){var lw=124,tw=300,cw=12+lw+26+tw+12,yw=336,xr=g.VW-10-cw,y=14,N=D.shards.length||1;
  for(var i=0;i<trs.length;i+=2){var h=0;[trs[i],trs[i+1]].forEach(function(tr){if(tr)h=Math.max(h,Math.max(170,((tr.nodes||[]).length||1)*nh()-8)+44)});
    [0,1].forEach(function(c){if(!trs[i+c])return;var x=c?xr:10,tx=c?x+12:x+12+lw+26;
      g.boxes.push({x:x,y:y,w:cw,h:h,lx:c?tx+tw+26:x+12,ly:y+34,lw:lw,rx:tx,ry:y+34,rw:0,rh:h-44,tx:tx,ty:y+34,tw:tw,th:h-44,co:false,roll:false,samp:[],mirror:!!c,hub:true})});
    y+=h+16}
  var yh=N===1?318:250,yx=(g.VW-yw)/2,yy=Math.max(14,(y-(N*(yh+16)-16))/2);
  for(var s=0;s<N;s++)g.syn.push({x:yx,y:yy+s*(yh+16),w:yw,h:yh});
  g.VH=Math.max(y,yy+N*(yh+16)+44)}
function shape(){return D.kind+"|"+D.trainers.map(function(t){return t.id+":"+(t.nodes||[]).length+":"+!!t.island}).join(",")+"|"+D.shards.length+"|"+(D.kind==="synced_group"?(T.samplers||[]).map(function(s){return s.id}).join(","):"")+"|"+D.P+"|"+!!D.follow+
  (D.fleet?"|"+D.fleet.islands.map(function(x){return x.trainer+":"+x.pools.map(function(p){return p.name}).join("+")}).join(","):"")}

// -- build ---------------------------------------------------------------------------
function build(host){
  var W=host.clientWidth||800,g=layout(W),isl=D.kind==="island";
  host.innerHTML="";
  var wrap=document.createElement("div");wrap.className="tpw";host.appendChild(wrap);
  var svg=el("svg",{viewBox:"0 0 "+g.VW+" "+g.VH,role:"img","aria-label":isl?"islands (trainer + in-island rollout) to syncer":sft()?"learner islands to syncer":"samplers to store to trainers to syncer"},wrap);
  var wires=el("g",{},svg),labels=el("g",{},svg),nd=el("g",{},svg);
  var lay={host:host,svg:svg,g:g,W:W,boxes:[],syn:[],pair:[],inl:[],samp:[],trunk:[],store:null,shape:shape()};
  D.trainers.forEach(function(tr,k){lay.boxes.push(islandBox(nd,wires,labels,g.boxes[k],tr,g.narrow))});
  g.syn.forEach(function(B,s){lay.syn.push(syncerCard(nd,B,D.shards[s]))});
  syncerLabel(nd,g.syn[g.syn.length-1],g.narrow);
  // trainer <-> syncer shard lanes: upper PUSH (->), lower BCAST (<-)
  D.trainers.forEach(function(tr,k){var B=g.boxes[k],row=[];
    g.syn.forEach(function(S,s){var base,a,b;
      if(!g.narrow){var x0=B.mirror?B.tx:B.tx+B.tw,y0=B.ty+B.th/2,x1=B.mirror?S.x+S.w:S.x,y1=S.y+S.h/2+(k-(D.trainers.length-1)/2)*24,bend=(x1-x0)*0.5;
        base=U.bez([x0,y0],[x0+bend,y0],[x1-bend,y1],[x1,y1],40)}
      else{var xr=g.VW-8-k*10-s*4,yt=B.ty+B.th/2;base=U.poly([[B.tx+B.tw,yt],[xr,yt],[xr,S.y-8-s*6],[S.x+S.w*0.5+k*12,S.y-8-s*6],[S.x+S.w*0.5+k*12,S.y]])}
      a=U.offset(base,-3.2);b=U.offset(base,3.2);
      var ea=el("path",{d:U.dpath(a),"class":"tpwire tpbold"},wires),eb=el("path",{d:U.dpath(b),"class":"tpwire tpbold"},wires);
      var m=U.at(base,0.5),x=el("text",{x:m[0],y:m[1]+4,"text-anchor":"middle","class":"tpv",display:"none"},labels,"✕");
      el("path",{d:U.dpath(base),"class":"tphit","data-td":"pair:"+k+":"+s},wires);
      var cnt=el("text",{x:m[0],y:m[1]-8,"text-anchor":"middle","class":"tpc",display:"none"},labels,"");
      row.push({push:a,bcast:U.rev(b),e:[ea,eb],x:x,cnt:cnt})});
    // the lane labels sit in the gap between the card and the syncer, on the side the lanes
    // leave free (above when they bend down to the syncer, below when they bend up), cut to the gap
    var Sy0=g.syn[0],lx=B.tx+B.tw+12,y0=B.ty+B.th/2,y1=Sy0?Sy0.y+Sy0.h/2+(k-(D.trainers.length-1)/2)*24:y0,
      yA=g.narrow?B.ty+B.th+12:(y1>=y0?y0-66:y0+28),t=[],labW=g.narrow?B.tw-8:(Sy0?Sy0.x:g.VW)-lx-10;
    for(var i=0;i<4;i++)t.push(el("text",{x:g.narrow?B.tx+4:lx,y:yA+i*13,"class":"tpm "+(i<2?"tpdl":"tpwtc"),display:B.hub||g.narrow&&i%2?"none":"inline"},labels,""));
    t.w=labW;lay.pair.push({row:row,lab:t})});
  if(g.store){lay.store=store(nd,g.store);
    (T.samplers||[]).forEach(function(sp,k){var B=g.samp[k],sx=B.x+B.w,sy=B.y+B.h/2,ex=g.store.cx-g.store.r-8,ey=g.store.cy+(k-((T.samplers.length-1)/2))*6,bend=(ex-sx)*0.55;
      var p=U.bez([sx,sy],[sx+bend,sy],[ex-bend,ey],[ex,ey],30);el("path",{d:U.dpath(U.offset(p,-2.4)),"class":"tpwire"},wires);
      lay.samp.push({lane:U.offset(p,-2.4),card:samplerCard(nd,B,sp)})});
    D.trainers.forEach(function(tr,k){var B=g.boxes[k],p;
      if(!g.narrow){var y=B.y+B.h/2;p=U.bez([g.store.cx+g.store.r+8,g.store.cy],[g.store.cx+g.store.r+40,g.store.cy],[B.x-30,y],[B.x,y],30)}
      else p=U.poly([[g.store.cx,g.store.cy+g.store.r+8],[g.store.cx,B.y-4],[B.tx+B.tw*0.6,B.y-4],[B.tx+B.tw*0.6,B.ty]]);
      var a=U.offset(p,-3),b=U.offset(p,3);el("path",{d:U.dpath(a),"class":"tpwire tpbold"},wires);el("path",{d:U.dpath(b),"class":"tpwire tpbold"},wires);
      lay.trunk.push({read:a,up:U.rev(b)})})}
  var cv=document.createElement("canvas");cv.className="tpcv";wrap.appendChild(cv);
  var fm=document.createElement("div");fm.className="fmw";host.appendChild(fm);
  lay.cv=cv;lay.ctx=cv.getContext?cv.getContext("2d"):null;lay.fm=fm;L=lay;fmKey="";fit()}
function fit(){if(!L||!L.svg.getBoundingClientRect)return;var r=L.svg.getBoundingClientRect(),dpr=Math.min(2,(typeof window!=="undefined"&&window.devicePixelRatio)||1);
  L.cv.style.width=r.width+"px";L.cv.style.height=r.height+"px";L.cv.width=Math.max(1,Math.round(r.width*dpr));L.cv.height=Math.max(1,Math.round(r.height*dpr));
  L.scale=r.width/L.g.VW;L.dpr=dpr}

function islandBox(nd,wires,labels,B,tr,narrow){
  var isl=D.kind==="island",r=tr.island,o={B:B};
  el("rect",{x:B.x,y:B.y,width:B.w,height:B.h,rx:14,"class":"tpgrp"},nd);
  var short=isl||sft()||!!fleet();
  el("text",{x:B.x+14,y:B.y+20,"class":"tpt"},labels,(short?"Island ":"Trainer island ")+tr.id);
  el("text",{x:B.x+(short?78:120),y:B.y+20,"class":"tpm"},labels,fleet()?clip(boxTitle(tr,narrow),Math.floor((B.w-92)/6.3)):boxTitle(tr,narrow));
  o.nodes=(tr.nodes||[]).map(function(n,j){return nodeLabel(nd,B.lx,B.ly+j*(narrow?62:nh()),narrow?Math.min(B.lw,250):B.lw,narrow?54:nh()-8,n,tr,j)});
  // --rl-sampler-follow trainer: this trainer's samplers, its lane of the store (trainer<k>/) and the V1 flows inside the box
  o.samp=[];
  if(B.lane){var j0=o.nodes.length,LN=B.lane;
    B.samp.forEach(function(S,j){var sn=snode(S.s.id),pl=S.s.pool;
      o.nodes.push(nodeLabel(nd,B.lx,B.ly+(j0+j)*(narrow?62:86),narrow?Math.min(B.lw,250):B.lw,narrow?54:78,
        pl?{id:(pl.source==="miles"?"m":"p")+j,machine:pl.source_name||pl.name,machine_full:pl.name,gpu:S.s.card||"GPU",gpus:S.s.gpus,cloud:pl.provider,
            roles:pl.samplers.slice(0,6).map(function(){return "R"}),incarnation:1,pool:pl.name}:
        {id:"s"+j,machine:sn.machine_full?String(sn.machine_full).split("-").slice(-2).join("-"):S.s.id,machine_full:sn.machine_full,gpu:sn.gpu||S.s.card,gpus:sn.gpus,roles:["R"],
         cloud:sn.cloud,incarnation:1,sampler:S.s.id},tr,"s"+j));
      var p=narrow?U.poly([[S.x+S.w,S.y+S.h/2],[LN.cx-LN.r-4,LN.cy]]):U.bez([S.x+S.w,S.y+S.h/2],[S.x+S.w+30,S.y+S.h/2],[LN.cx-LN.r-30,LN.cy],[LN.cx-LN.r-4,LN.cy],24);
      el("path",{d:U.dpath(p),"class":"tpwire"},wires);o.samp.push({lane:p,card:samplerCard(nd,S,S.s,sn)})});
    var tl=narrow?[LN.cx,LN.cy+LN.r+4]:[LN.cx+LN.r+4,LN.cy],tt=narrow?[LN.cx,B.ty]:[B.tx,B.ty+B.th/2],lp=U.poly([tl,tt]);
    o.read=U.offset(lp,-3);o.up=U.rev(U.offset(lp,3));
    el("path",{d:U.dpath(o.read),"class":"tpwire tpbold"},wires);el("path",{d:U.dpath(U.offset(lp,3)),"class":"tpwire tpbold"},wires);
    var lg=el("g",{"data-td":"lane:"+tr.id},nd);el("circle",{cx:LN.cx,cy:LN.cy,r:LN.r,"class":"tpbox"},lg);
    el("text",{x:LN.cx,y:LN.cy-3,"text-anchor":"middle","class":"tpm",style:"font-size:9.5px"},lg,laneName(tr));
    o.lat=el("text",{x:LN.cx,y:LN.cy+10,"text-anchor":"middle","class":"tpv tpwtc",style:"font-size:10px"},lg,"");
    o.buf=el("text",{x:LN.cx,y:LN.cy+LN.r+14,"text-anchor":"middle","class":"tpm"},labels,"")}  // under the circle, not in it
  if(B.co){o.merged=el("rect",{x:B.rx,y:B.ry,width:B.tx+B.tw-B.rx,height:B.th,rx:10,"class":"tpbox"},nd);el("rect",{x:B.rx,y:B.ry,width:4,height:B.th,rx:2,fill:arch(tr.arch)},nd);
    if(!narrow)el("line",{x1:B.tx-14,y1:B.ry+12,x2:B.tx-14,y2:B.ry+B.th-12,stroke:"var(--line)","stroke-dasharray":"3 3"},nd)}
  if(B.roll){o.ro=rolloutCard(nd,B.rx,B.ry,B.rw,B.rh,tr,B.co);
    var x0=B.rx+B.rw+4,x1=B.tx-(B.co?18:4),yc=B.ry+B.rh/2,gp,sp;
    if(!narrow){gp=U.poly([[x0,yc-9],[x1,yc-9]]);sp=U.poly([[x1,yc+9],[x0,yc+9]])}
    else{var xc=B.rx+B.rw-30;gp=U.poly([[xc-8,B.ry+B.rh],[xc-8,B.ty]]);sp=U.poly([[xc+8,B.ty],[xc+8,B.ry+B.rh]])}
    var eg=el("path",{d:U.dpath(gp),"class":"tpwire tpbold"},wires),es=el("path",{d:U.dpath(sp),"class":"tpwire tpbold"},wires);
    if(!narrow){el("text",{x:(x0+x1)/2,y:yc-18,"text-anchor":"middle","class":"tpm tptraj"},labels,"groups →");
      o.sl=el("text",{x:(x0+x1)/2,y:yc+30,"text-anchor":"middle","class":"tpm tpwtc"},labels,"← snapshot");
      o.sl2=el("text",{x:(x0+x1)/2,y:yc+43,"text-anchor":"middle","class":"tpm tpwtc"},labels,"")}
    el("path",{d:U.dpath(U.poly([[x0,yc],[x1,yc]])),"class":"tphit","data-td":"rol:"+tr.id},wires);
    o.grp=gp;o.pub=sp;o.e=[eg,es]}
  o.tc=trainerCard(nd,B.tx,B.ty,B.tw,B.th,tr,B.co);
  // machine label -> sub-card: a thin line per role cell (T to the trainer, R to the rollout)
  if(!narrow&&!B.mirror)o.nodes.forEach(function(n){n.cells.forEach(function(c){var to=c.role==="R"?(B.roll?[B.rx,B.ry+B.rh*0.3]:null):[B.co?B.rx:B.tx,B.ty+B.th*(c.role==="T+R"?0.5:0.3)];
    if(to)el("path",{d:"M"+(c.x+c.w)+","+c.y+"C"+(c.x+c.w+14)+","+c.y+" "+(to[0]-14)+","+to[1]+" "+to[0]+","+to[1],fill:"none",stroke:c.role==="R"?"var(--traj)":"var(--delta)","stroke-width":1,opacity:0.45,"stroke-dasharray":"2 3"},wires)})});
  return o}
function boxTitle(tr,narrow){var r=tr.island,n=tr.gpus,g=tr.gpu||"GPU",pl=r&&r.placement,eng=r&&fin(r.engines)?r.engines+" rollout engine"+(r.engines>1?"s":""):null;
  if(sft()){var ln=tr.learner||{};return [tr.name,ln.layout,fin(n)?n+" cards":null,!narrow&&ln.dataset?"data "+ln.dataset:null].filter(Boolean).join(" · ")}
  if(!r&&fleet()){var fi=fisland(tr)||{},ps=fi.pools||[],nm=ps.filter(function(p){return p.source==="miles"}).length,np=ps.length-nm;
    var what=nm?(narrow?nm+" Miles + "+np+" pool(s)":nm+" Miles sampler(s) + "+np+" external pool(s)"):narrow?np+" pool(s)":np+" external pool(s)";
    return [fi.name,fin(n)?n+" × "+g:fi.gpu||g,narrow?what:what+" · R2 lane "+laneName(tr)].filter(Boolean).join(" · ")}
  if(!r){var ms=follow()?mine(tr):[];return [fin(n)?n+" × "+g:g,tr.arch,ms.length?ms.length+" sampler(s) on their own store lane "+laneName(tr):null].filter(Boolean).join(" · ")}
  var parts=[pl,fin(n)?n+" × "+g:g];
  if(r.colocated)parts.push(narrow?"shared GPU":"rollout and train time-share the GPU");else if(eng)parts.push(narrow?eng:"trainer + "+eng+" (split not on the tape)");
  return parts.filter(Boolean).join(" · ")}

function nodeLabel(nd,x,y,w,h,n,tr,j){
  var g=el("g",{transform:"translate("+x+","+y+")","data-td":"node:"+tr.id+":"+j},nd),old=!!n.previous;
  var box=el("rect",{width:w,height:h,rx:8,"class":"tpbox"},g);
  el("text",{x:9,y:15,"class":"tpv",style:"font-size:10.5px"},g,clip(n.id+" · "+(n.machine||"—"),Math.floor((w-(n.incarnation>1||old?40:18))/6.4)));
  if(n.incarnation>1||old)el("text",{x:w-8,y:15,"text-anchor":"end","class":"tpm"},g,"#"+n.incarnation);
  el("text",{x:9,y:30,"class":"tpm"},g,clip((n.gpu||(sft()?"card":"GPU"))+(fin(n.gpus)?" ×"+n.gpus:"")+(n.cloud?" · "+n.cloud:""),Math.floor((w-14)/6.3)));
  var cells=[],roles=n.roles||[],cw=roles.length&&roles[0]==="T+R"?34:20,cx=9;
  roles.forEach(function(rl){var fill=rl==="R"?"var(--traj)":rl==="S"?"var(--ink2)":"var(--delta)";
    el("rect",{x:cx,y:37,width:cw,height:15,rx:3,fill:fill,opacity:old?0.3:0.22,stroke:fill},g);el("text",{x:cx+cw/2,y:48,"text-anchor":"middle","class":"tpm",style:"font-size:9.5px;font-weight:600"},g,rl+(n.inferred?"*":""));
    cells.push({role:rl,x:x+cx,y:y+44,w:cw});cx+=cw+4});
  if(!roles.length)fitText(el("text",{x:9,y:48,"class":"tpm"},g,""),"roles not on the tape",w-26);
  // a short label (narrow page): the info line beside the role cells, not across them
  var ix=h<70?cx+6:9,dot=el("circle",{cx:w-12,cy:44,r:3.5},g),info=el("text",{x:ix,y:h-8,"class":"tpm"},g,"");info._w=w-ix-(h<70?22:10);
  if(old)g.setAttribute("opacity",0.45);
  return{g:g,box:box,dot:dot,info:info,n:n,old:old,cells:cells}}
function syncerLabel(nd,S,narrow){if(!S)return;var y=S.y+S.h+8,w=S.w;
  var g=el("g",{transform:"translate("+S.x+","+y+")","data-td":"snode"},nd);el("rect",{width:w,height:28,rx:7,"class":"tpbox"},g);
  var cx=8;D.shards.forEach(function(s){var lab="S"+(D.shards.length>1?s.index:"");el("rect",{x:cx,y:7,width:22,height:14,rx:3,fill:"var(--ink2)",opacity:0.22,stroke:"var(--ink2)"},g);
    el("text",{x:cx+11,y:17.5,"text-anchor":"middle","class":"tpm",style:"font-size:9.5px;font-weight:600"},g,lab);cx+=26});
  fitText(el("text",{x:cx+4,y:18,"class":"tpm"},g,""),"syncer · machine not on the tape",w-cx-12)}

function trainerCard(nd,x,y,w,h,tr,co){
  var g=el("g",{transform:"translate("+x+","+y+")","data-td":"trn:"+tr.id},nd);
  var box=el("rect",{width:w,height:h,rx:10,"class":"tpbox",visibility:co?"hidden":"visible"},g);if(!co)el("rect",{x:0,y:0,width:4,height:h,rx:2,fill:arch(tr.arch)},g);
  el("text",{x:14,y:20,"class":"tpt"},g,clip((sft()?"Learner ":"Trainer ")+tr.id+(tr.name?" · "+tr.name:""),Math.floor((w-70)/7)));
  el("text",{x:14,y:36,"class":"tpm"},g,sft()?["ms-swift Megatron",(tr.learner||{}).layout].filter(Boolean).join(" · "):co?"Megatron · same GPU as the rollout":D.kind==="island"?"Megatron · "+(tr.gpu||"GPU"):[tr.gpus?tr.gpus+" × "+tr.gpu:tr.gpu,tr.arch].filter(Boolean).join(" · "));
  var o={g:g,box:box,co:co};
  o.badge=el("text",{x:w-12,y:20,"text-anchor":"end","class":"tpv"},g,"");
  o.step=el("text",{x:14,y:66,"class":"tpbig2"},g,"");o.outer=el("text",{x:w-12,y:66,"text-anchor":"end","class":"tpm"},g,"");
  o.dot=el("circle",{cx:18,cy:86,r:3.5},g);o.ph=el("text",{x:27,y:90},g,"");
  o.l=[];for(var i=0;i<5;i++){var ly=114+i*(h>200?18:16);o.l.push(el("text",{x:14,y:ly,"class":"tpm",display:ly>h-4?"none":"inline"},g,""))}
  o.lw=w-26;o.phw=w-36;
  o.flash=el("rect",{width:w,height:h,rx:10,fill:"none","stroke-width":1.5,opacity:0},g);
  return o}
function rolloutCard(nd,x,y,w,h,tr,co){
  var g=el("g",{transform:"translate("+x+","+y+")","data-td":"rol:"+tr.id},nd),r=tr.island;
  var box=co?null:el("rect",{width:w,height:h,rx:10,"class":"tpbox"},g);if(!co)el("rect",{x:0,y:0,width:4,height:h,rx:2,fill:arch(tr.arch)},g);
  el("text",{x:14,y:20,"class":"tpt"},g,"Rollout"+(fin(r.engines)?" · "+r.engines+" engine"+(r.engines>1?"s":""):""));
  var o={g:g,box:box,co:co};
  o.sub=el("text",{x:14,y:36,"class":"tpm"},g,co?"":(tr.gpu||"GPU")+" · own GPU(s) · "+(r.exec_mode||r.placement||""));
  o.big=el("text",{x:14,y:64,"class":"tpbig2"},g,"");
  el("rect",{x:14,y:71,width:w-28,height:3,rx:1.5,fill:"var(--line)"},g);o.barf=el("rect",{x:14,y:71,width:0,height:3,rx:1.5,fill:"var(--traj)"},g);o.barW=w-28;
  o.dot=el("circle",{cx:18,cy:88,r:3.5},g);o.st=el("text",{x:27,y:92},g,"");
  o.l=[];for(var i=0;i<4;i++)o.l.push(el("text",{x:14,y:114+i*(h>180?20:16),"class":"tpm"},g,""));o.lw=w-26;
  o.flash=el("rect",{width:w,height:h,rx:10,fill:"none","stroke-width":1.5,opacity:0},g);
  return o}
function syncerCard(nd,B,sh){
  var g=el("g",{transform:"translate("+B.x+","+B.y+")","data-td":"syn:"+sh.index},nd),w=B.w,h=B.h,N=D.shards.length;
  var box=el("rect",{width:w,height:h,rx:10,"class":"tpbox"},g);el("rect",{x:0,y:0,width:4,height:h,rx:2,fill:"var(--ink2)"},g);
  el("text",{x:14,y:20,"class":"tpt"},g,N>1?"Syncer shard "+sh.index+" · fragments "+sh.lo+"–"+(sh.hi-1):"Syncer");
  fitText(el("text",{x:14,y:36,"class":"tpm"},g,""),(N>1?"steps t ≡ "+(sh.index+1)+" (mod "+N+") · ":"")+["P "+D.P,D.quorum!=null?"quorum "+D.quorum+" of "+D.M:"M "+D.M,D.H?(D.H_exact?"H ":"H ≤ ")+D.H:null].concat(cfgBits(D.config)).filter(Boolean).join(" · "),w-26);
  var o={g:g,box:box,sh:sh,lw:w-26};
  o.roster=el("text",{x:w-12,y:20,"text-anchor":"end","class":"tpv"},g,"");
  o.step=el("text",{x:14,y:66,"class":"tpbig2"},g,"");o.of=el("text",{x:w-12,y:66,"text-anchor":"end","class":"tpm"},g,"");
  el("rect",{x:14,y:74,width:w-28,height:4,rx:2,fill:"var(--line)"},g);o.barf=el("rect",{x:14,y:74,width:0,height:4,rx:2,fill:"var(--wt)"},g);o.barW=w-28;
  o.dot=el("circle",{cx:18,cy:96,r:3.5},g);o.st=el("text",{x:27,y:100},g,"");
  o.ln=[];for(var i=0;i<(N>1?5:7);i++)o.ln.push(el("text",{x:14,y:122+i*16,"class":"tpm"},g,""));
  var n=sh.hi-sh.lo,cw=Math.min(40,(w-28-(n-1)*3)/Math.max(n,1)),sy=h-32;o.cells=[];
  var gap=cw<8?1:3;cw=Math.min(40,(w-28-(n-1)*gap)/Math.max(n,1)); // many fragments (SFT: 64): a strip of ticks, the matrix has the numbers
  for(var f=sh.lo;f<sh.hi;f++){var cx=14+(f-sh.lo)*(cw+gap);o.cells.push({r:el("rect",{x:cx,y:sy,width:cw,height:22,rx:cw<8?1:4,fill:"var(--bg)",stroke:"var(--line)"},g),
    t:el("text",{x:cx+cw/2,y:sy+15,"text-anchor":"middle","class":"tpm",style:"font-size:9.5px",display:cw<30?"none":"inline"},g,"")})}
  return o}
function samplerCard(nd,B,s,sn){
  var g=el("g",{transform:"translate("+B.x+","+B.y+")","data-td":"smp:"+s.id},nd);sn=sn||snode(s.id);
  el("rect",{width:B.w,height:B.h,rx:9,"class":"tpbox"},g);el("rect",{x:0,y:0,width:4,height:B.h,rx:2,fill:arch(s.arch)},g);
  el("text",{x:13,y:18,"class":"tpt"},g,clip([s.card,s.precision,s.engine].filter(Boolean).join(" · ")||s.id,Math.floor((B.w-50)/6.5)));
  var src=s.pool&&SRC[s.source],x2=13;
  // a fleet with Miles sampler islands: a source tag (Miles native / external pool) before the row's name
  if(src){var tw=src[0].length*5.9+10;el("rect",{x:13,y:23.5,width:tw,height:13,rx:6.5,fill:src[1],"fill-opacity":0.2,stroke:src[1]},g);
    el("text",{x:13+tw/2,y:33,"text-anchor":"middle","class":"tpm",style:"font-size:9px;font-weight:700;letter-spacing:.3px"},g,src[0].toUpperCase());x2=13+tw+6}
  el("text",{x:x2,y:33,"class":"tpm"},g,clip(src?s.id:s.pool?s.id+" · "+s.n+" sampler"+(s.n===1?"":"s"):
    ["sampler "+s.id,s.gpus?s.gpus+" GPU":null,sn.cloud].filter(Boolean).join(" · "),Math.floor((B.w-x2-7)/6.3)));
  return{ver:el("text",{x:B.w-10,y:18,"text-anchor":"end","class":"tpv"},g,""),dot:el("circle",{cx:16.5,cy:B.h-12.5,r:3.5},g),
    st:el("text",{x:25,y:B.h-9,"class":"tps2"},g,""),rate:el("text",{x:B.w-10,y:B.h-9,"text-anchor":"end","class":"tpm"},g,""),s:s}}
function store(nd,R){var g=el("g",{"data-td":"store"},nd);
  el("circle",{cx:R.cx,cy:R.cy,r:R.r,"class":"tpbox"},g);el("text",{x:R.cx,y:R.cy-4,"text-anchor":"middle","class":"tpt tpbig"},g,"Store");
  // the buffer and publisher lines sit under the circle: labels of the graph, not of the circle
  return{lat:el("text",{x:R.cx,y:R.cy+14,"text-anchor":"middle","class":"tpv tpwtc"},g,""),info:el("text",{x:R.cx,y:R.cy+R.r+18,"text-anchor":"middle","class":"tpm"},nd,""),
    pub:el("text",{x:R.cx,y:R.cy+R.r+32,"text-anchor":"middle","class":"tpm tpwtc"},nd,"")}}

// -- paint (4 Hz) ------------------------------------------------------------------
function lane(e,k){var c={delta:C.delta,wt:C.wt,warn:C.warn,bad:C.bad}[k];e.style.stroke=c||"";e.style.strokeDasharray=k==="warn"?"6 4":"";e.setAttribute("opacity",c?0.85:1)}
function active(kk,iid,sh,td){return T.live&&D.flows.some(function(f){return f.k===kk&&f.i===iid&&(sh==null||f.sh===sh)&&f.t0<=td&&Math.max(f.t1,f.t0+1.3)>=td})}
function paint(){
  var tn=now(),td=tn-DELAY_S,red=U.reduced();
  D.trainers.forEach(function(tr,k){var o=L.boxes[k],tc=o.tc,st=trainerState(tr,tn,tr.island),c=col(st[1]),dead=!tr.live&&!(tr.status==="done"||tr.finalized);
    tc.step.textContent=(fin(tr.step)?"step "+tr.step:"step n/a")+(sft()&&fin(tr.steps_total)?" / "+tr.steps_total:"");tc.outer.textContent=fin(tr.outer)?"outer v"+tr.outer:"";
    fitText(tc.ph,st[0],tc.phw);tc.ph.style.fill=st[1]==="good"?C.ink:c;tc.dot.setAttribute("fill",c);
    tc.badge.textContent=tr.failure?(/hash/.test(tr.failure.metric)?"hash ✕":/lr/.test(tr.failure.metric)?"lr 0":"failed"):tr.lr_zero&&tr.live?"lr 0":dead?"lost":tr.syncer_link&&tr.syncer_link.state==="disconnected"&&!D.done?"link ✕":"";tc.badge.style.fill=C.bad;
    [tc.co?o.merged:tc.box].forEach(function(b){if(b){b.style.stroke=dead||tr.failure?C.bad:"";b.style.strokeDasharray=dead?"5 4":""}});
    o.tc.g.setAttribute("opacity",dead?0.6:1);
    var L0=tc.l,lw=tc.lw;
    fitText(L0[0],"snapshot "+(tr.hash||"n/a")+(lr(tr.lr)!=null?" · lr "+lr(tr.lr):""),lw);
    if(tr.versions){var parts=[["frag [",""]];tr.versions.forEach(function(v,f){var b=behind(tr,f);parts.push([String(v),b?"fill:"+C.warn+";font-weight:700":""]);if(f<tr.versions.length-1)parts.push([" ",""])});parts.push(["]",""]);rich(L0[1],parts)}
    else L0[1].textContent="fragment versions not on the tape";
    var old=fin(tr.pending_oldest_ts)?tn-tr.pending_oldest_ts:0;
    fitText(L0[2],(tr.pending_n&&D.done?"↓ run ended · last "+tr.pending_n+" merge(s) → final cut":tr.pending_n?"↓ BCAST pending "+tr.pending_n+" · oldest "+dur(old):"↓ every merged fragment applied")+(fin(tr.apply_s)?" · apply "+tr.apply_s.toFixed(2)+"s":""),lw);
    L0[2].style.fill=sft()||D.done?C.ink2:tr.live&&old>BCAST_BAD_S?C.bad:tr.live&&old>BCAST_WARN_S?C.warn:C.ink2;
    var up=Object.keys(tr.pushed||{});
    fitText(L0[3],tr.full_push?"↑ "+(tr.full_push.count||0)+" PUSH last round · "+(tr.full_push.total||0)+" in run"+(tr.full_push.overwritten?" · "+tr.full_push.overwritten+" overwritten":fin(tr.full_push.inflight)&&tr.full_push.inflight?" · in flight "+tr.full_push.inflight:""):
      "↑ unmerged PUSH "+up.length+(up.length?" ("+up.slice(0,3).map(function(f){return "f"+f+" t"+tr.pushed[f]}).join(", ")+")":""),lw);
    fitText(L0[4],fin(tr.gpu_util)?"GPU "+Math.round(tr.gpu_util)+"%"+(fin(tr.mem_pct)?" · HBM "+Math.round(tr.mem_pct)+"%":""):"",lw);
    var ap=fin(tr.apply_ts)&&T.live&&td-tr.apply_ts>=0&&td-tr.apply_ts<3;tc.flash.style.stroke=C.wt;tc.flash.setAttribute("opacity",ap?0.9:0);
    if(o.ro)paintRollout(tr,o,tn,td);
    if(o.lat){var ln=(D.lanes||{})[tr.id]||{};o.lat.textContent=fin(ln.latest)?"v"+ln.latest:"no release";o.buf.textContent=fin(ln.backlog)?"buffer "+Math.round(ln.backlog):"written "+Math.round(ln.written||0);
      o.samp.forEach(function(x){x.card.s=fresh(x.card.s);paintSampler(x.card)})}
    o.nodes.forEach(function(n){var pl=n.n.pool?fpool(n.n.pool):null,sp=pl?poolCard(pl):n.n.sampler?snode(n.n.sampler):null,nc=n.old?C.ink2:sp?(sp.live?C.good:C.ink2):dead||tr.failure?C.bad:tr.live?C.good:C.ink2;n.dot.setAttribute("fill",nc);
      fitText(n.info,n.old?"replaced"+(n.n.reason?": "+String(n.n.reason).slice(0,18):""):fin(n.n.util)?"GPU "+Math.round(n.n.util)+"%"+(fin(n.n.mem_pct)?" · HBM "+Math.round(n.n.mem_pct)+"%":""):"",n.info._w);
      n.box.style.stroke=n.old?"":dead?C.bad:"";n.box.style.strokeDasharray=n.old?"4 3":""});
    // lanes to each shard
    var pr=L.pair[k],failP=dead||(tr.failure&&/lr/.test(tr.failure.metric))||(tr.live&&tr.lr_zero),failB=dead||(tr.failure&&/hash/.test(tr.failure.metric));
    pr.row.forEach(function(ln,s){var blockedP=D.rounds.some(function(r){return r.shard===s&&r.blocked&&r.missing.indexOf(tr.id)>=0}),
        blockedB=!sft()&&tr.live&&old>BCAST_WARN_S&&Object.keys(tr.pending||{}).some(function(f){return shardOf(+f)===s});
      lane(ln.e[0],failP?"bad":blockedP?"warn":active("p",tr.id,s,td)?"delta":"");
      lane(ln.e[1],failB?"bad":blockedB?"warn":active("b",tr.id,s,td)?"wt":"");
      ln.x.setAttribute("display",failP||failB?"inline":"none");ln.x.style.fill=C.bad;
      ln.cnt.setAttribute("display",red&&T.live?"inline":"none");
      if(red&&T.live)ln.cnt.textContent=D.flows.filter(function(f){return f.i===tr.id&&f.sh===s&&f.k==="p"&&f.t1>tn-300}).length+" PUSH / 5 min"});
    var lp=tr.last_push,lb=tr.last_bcast;
    var lw2=pr.lab.w;
    fitText(pr.lab[0],lp?"↑ f"+lp.frag+" t"+lp.step+" base v"+lp.base+" · "+mb(lp.bytes)+(lp.wire?" wire":""):tr.full_push?"↑ full-parameter PUSH (aggregate only)":"↑ no PUSH yet",lw2);
    fitText(pr.lab[1],lp?"  PULL→PUSH "+dur(lp.p2p_s)+(lp.h?" · H "+lp.h:""):"",lw2);
    fitText(pr.lab[2],lb?"↓ f"+lb.frag+" v"+lb.v+" · "+mb(lb.bytes)+(lb.overwrite?" · overwrite":lb.catch_up?" · catch-up":""):tr.full_push?"↓ BCAST applied in the round (aggregate)":"↓ no BCAST applied yet",lw2);
    fitText(pr.lab[3],lb?"  queue "+dur(lb.queue_s)+(fin(tr.apply_s)?" · apply "+tr.apply_s.toFixed(2)+"s":""):"",lw2);
    if(sft())paintLearner(tr,tc.l,pr.lab,tn,tc.lw)});
  L.syn.forEach(function(o,s){var sh=D.shards[s],mine=D.rounds.filter(function(r){return r.shard===s}),st=sh.state,last=sh.last;
    o.step.textContent="step "+sh.step;o.of.textContent=(fin(D.total_steps)?"of "+D.total_steps+" · ":"")+sh.merges+" merges · sweep "+(D.P?Math.floor(Math.max(0,sh.step-1)/D.P)+1:"n/a");
    o.barf.setAttribute("width",D.P?(o.barW*(((sh.step-1)%D.P+1)/D.P)).toFixed(1):0);
    o.roster.textContent=D.done?"run done · roster "+D.M:"roster "+D.roster+"/"+D.M;o.roster.style.fill=D.done?C.ink2:D.roster<D.M?C.bad:C.good;
    var c=st==="done"?C.ink2:st==="waiting for quorum"?(sh.attempt_max>1?C.bad:C.warn):st==="merging"?C.wt:st==="gathering"?C.good:C.ink2;
    fitText(o.st,(st==="done"?"done (final cut)":st)+" · in flight "+mine.length+(D.behind?" · syncer tape behind":""),o.lw-14);o.st.style.fill=c===C.good?C.ink:c;o.dot.setAttribute("fill",c);
    o.box.style.stroke=c===C.bad?C.bad:"";
    var lines=mine.slice(0,3).map(function(r){return["t"+r.step+" f"+r.frag+": "+(r.state==="merging"?"merging":r.state==="pull"?"PULL out":"PUSH "+r.pushed.length+"/"+D.M)+" · "+dur(tn-r.since)+(r.attempt>1?" · attempt "+r.attempt:"")+(r.timeouts?" · "+r.timeouts+" timeout"+(r.timeouts>1?"s":""):""),r.blocked||r.timeouts?C.warn:C.ink2]});
    lines.push([last?"last merge t"+last.step+" f"+last.frag+" · quorum "+dur(last.quorum_s)+" · merge "+(fin(last.merge_s)?last.merge_s.toFixed(2)+"s":"n/a"):"no merge yet",C.ink2]);
    if(last)lines.push(["checkpoint step "+last.step+(fin(last.checkpoint_s)?" · "+last.checkpoint_s.toFixed(2)+"s":"")+" · "+dur(tn-last.ts)+" ago (before BCAST)",C.ink2]);
    if(last&&last.missed.length)lines.push(["last merge without "+(sft()?"island ":"trainer ")+last.missed.join(", ")+" (quorum "+last.quorum+")",C.ink2]);
    if(D.shards.length===1)lines.push(["5 min: ↑ "+D.traffic.push+" PUSH"+(fin(D.traffic.push_bytes)?" "+mb(D.traffic.push_bytes):"")+" · ↓ "+D.traffic.bcast+" applied",C.ink2]);
    if(D.behind)lines.push(["syncer tape behind the islands · no negative lag drawn",C.ink2]);
    lines=wrapLines(lines,o.lw,o.ln.length);
    o.ln.forEach(function(e,i){fitText(e,lines[i]?lines[i][0]:"",o.lw);e.style.fill=lines[i]?lines[i][1]:C.ink2});
    o.cells.forEach(function(cl,i){var f=sh.lo+i,r=mine.filter(function(x){return x.frag===f})[0],stuck=D.stuck.indexOf(f)>=0;
      cl.t.textContent="f"+f+" "+D.versions[f];cl.r.style.stroke=stuck?C.bad:r?(r.state==="merging"?C.wt:C.delta):"";cl.r.style.strokeWidth=r||stuck?1.6:1})});
  if(L.store){var b=T.buffer||{};L.store.lat.textContent=fin(T.latest)?"LATEST v"+T.latest:"no release";
    L.store.info.textContent=fin(b.backlog)?"buffer "+Math.round(b.backlog)+" groups":"buffer n/a";
    L.store.pub.textContent=D.publisher!=null?"samplers load trainer "+D.publisher+"'s weights":"";
    L.samp.forEach(function(x){x.card.s=fresh(x.card.s);paintSampler(x.card)})}
  matrix(tn)}
// an SFT learner: loss / tokens per step instead of policy and rollout; PUSH from the merge records' responders
function paintLearner(tr,l,lab,tn,lw){var ln=tr.learner||{},lp=tr.last_push,la=tr.last_apply;
  fitText(l[0],[fin(ln.loss)?"loss "+ln.loss.toFixed(4):"loss n/a",lr(tr.lr)!=null?"lr "+lr(tr.lr):null].filter(Boolean).join(" · "),lw);
  fitText(l[1],"step "+(fin(ln.step_s)?dur(ln.step_s):"n/a")+" · "+kn(ln.tps)+" tok/s",lw);
  fitText(l[2],la?"↓ "+tr.pending_n+" merged since its apply at step "+la.step:"↓ no broadcast applied yet",lw);
  fitText(l[3],lp?"↑ f"+lp.frag+" → t"+lp.step+" · base v"+lp.base+(fin(lp.export_s)?" · export "+lp.export_s.toFixed(1)+"s":""):"↑ no PUSH on the syncer tape",lw);
  fitText(l[4],ln.dataset?"data "+ln.dataset:"",lw);
  fitText(lab[0],lp?"↑ f"+lp.frag+" t"+lp.step+" base v"+lp.base+" · "+kn(lp.tokens)+" tok":"↑ no PUSH yet",lab.w);
  fitText(lab[1],lp?"  recv "+dur(lp.recv_s)+(fin(lp.staleness)?" · staleness "+lp.staleness:""):"",lab.w);
  fitText(lab[2],la?"↓ applied at step "+la.step+(fin(la.apply_s)?" · "+la.apply_s.toFixed(1)+"s":""):"↓ no apply yet",lab.w);
  fitText(lab[3],la?"  "+dur(tn-la.ts)+" ago":"",lab.w)}
function paintSampler(o){var s=o.s;o.ver.textContent=fin(s.policy_version)?"v"+s.policy_version+(s.lag>0?" −"+s.lag:""):s.pool&&!s.live?"":"v?";
  o.ver.style.fill=s.lag>=2?C.bad:s.lag===1?C.warn:C.ink;var c=s.live?(s.state==="sampling"?C.good:s.state==="waiting"?C.warn:C.wt):C.ink2;
  o.dot.setAttribute("fill",c);o.st.textContent=(s.live?s.state:(s.status||"ended"))+(s.pool&&s.n?" · "+s.nlive+"/"+s.n+" up":"");o.rate.textContent=(s.rate_gpm||0).toFixed(1)+" g/min"}
function paintRollout(tr,o,tn,td){var r=tr.island,ro=o.ro,st=rolloutState(tr,tn),c=col(st[1]);
  ro.big.textContent=fin(r.round)?"round "+r.round:"round n/a";fitText(ro.st,st[0],ro.lw-14);ro.st.style.fill=st[1]==="good"?C.ink:c;ro.dot.setAttribute("fill",c);
  ro.barf.setAttribute("width",st[2]!=null?(ro.barW*st[2]/100).toFixed(1):0);
  if(ro.co){var on=tr.phase_raw==="generate"||tr.phase_raw==="publish",live=tr.live;
    rich(ro.sub,[["GPU phase: ",""],["rollout",on&&live?"fill:"+C.traj+";font-weight:700":""],[" ⇄ ",""],["train",!on&&live&&tr.phase_raw==="train"?"fill:"+C.delta+";font-weight:700":""]])}
  var sv=r.served;
  fitText(ro.l[0],sv?"serving yeto:"+sv.v+":"+sv.hash:"serving the initial weights",ro.lw);
  fitText(ro.l[1],r.groups_src==="trained"?(fin(r.groups)?r.groups+" groups / round (trainer side)":"groups not reported by the rollout"):
    fin(r.groups)?r.groups+" groups · "+(r.traj||0)+" traj · "+(fin(r.tokens)?(r.tokens/1000).toFixed(1)+"k":"n/a")+" tok / round":"no round yet",ro.lw);
  fitText(ro.l[2],fin(r.p50_s)?"group p50 "+r.p50_s.toFixed(1)+"s · p95 "+(fin(r.p95_s)?r.p95_s.toFixed(1)+"s":"—"):fin(r.rollout_p50_s)?"rollout "+r.rollout_p50_s.toFixed(1)+"s (median of last rounds)":"",ro.lw);
  fitText(ro.l[3],[fin(r.reward)?"reward "+r.reward.toFixed(2):null,fin(r.zero_var)?"zero-var "+Math.round(r.zero_var*100)+"%":null,T.live?(r.groups_5m/5).toFixed(1)+" g/min":null].filter(Boolean).join(" · "),ro.lw);
  var ld=sv&&fin(sv.ts)&&T.live&&td-sv.ts>=0&&td-sv.ts<3;ro.flash.style.stroke=C.wt;ro.flash.setAttribute("opacity",ld&&!ro.co?0.9:0);
  if(ro.box){ro.box.style.stroke=!tr.live&&tr.status!=="done"&&!tr.finalized?C.bad:""}
  var gq=active("q",tr.id,null,td);o.e[0].style.stroke=gq?C.traj:"";o.e[0].setAttribute("opacity",gq?0.85:1);
  lane(o.e[1],active("s",tr.id,null,td)?"wt":"");
  if(o.sl){o.sl.textContent=sv?"← snapshot v"+sv.v:"← snapshot";o.sl2.textContent=sv&&fin(sv.publish_s)?"publish "+sv.publish_s.toFixed(1)+"s":""}}

// -- fragment matrix (HTML) ------------------------------------------------------------
function matrix(tn){
  var P=D.P||0,H=D.H,h='<div class="fmh"><b>Fragments</b><span>P '+P+" × M "+D.M+(H?(D.H_exact?" · H ":" · H ≤ ")+H:"")+(fin(D.period_s)?" · fragment period ≈ "+dur(D.period_s):"")+" · column header: age of the fragment's last merge"+(T.live?"":", as of the snapshot")+(D.behind?" · syncer tape behind":"")+'</span></div><table class="fm"><tr><th class="rh">fragment</th>';
  for(var f=0;f<P;f++){var st=D.stuck.indexOf(f)>=0,lt=D.last_merge_ts[f];h+='<th class="'+(st?"stuck":"")+'" title="'+esc("fragment "+f+(D.shards.length>1?", shard "+shardOf(f):"")+(fin(lt)?", last merged "+dur(tn-lt)+" ago":", not merged yet"))+'">f'+f+(D.shards.length>1?" · s"+shardOf(f):"")+'<span class="fa">'+(st?"stuck ":"")+(fin(lt)?"merged "+dur(tn-lt)+" ago":"not merged")+"</span></th>"}
  h+='</tr><tr><th class="rh">Syncer<span class="ns">committed version · round</span></th>';
  for(f=0;f<P;f++){var r=D.rounds.filter(function(x){return x.frag===f})[0],cls="",s2="idle",lt2=D.last_merge_ts[f];
    if(r){cls=r.blocked?"sy-stuck":r.state==="merging"?"sy-merge":"sy-gather";s2=(r.state==="merging"?"merging":r.state==="pull"?"PULL out":"PUSH "+r.pushed.length+"/"+D.M)+" t"+r.step}
    var fresh=T.live&&fin(lt2)&&tn-DELAY_S-lt2>=0&&tn-DELAY_S-lt2<4;
    h+='<td class="'+cls+(fresh?" flash":"")+'" title="'+esc("committed v"+D.versions[f]+" (global step of the last merge)"+(r?"; step "+r.step+" open "+dur(r.age_s)+", pushed by "+(r.pushed.join(", ")||"none"):""))+'"><div class="v">v'+D.versions[f]+'</div><div class="s">'+s2+"</div></td>"}
  h+="</tr>";
  var finalOnSyncer=false;
  if(sft())h+=learnerRows(tn);else D.trainers.forEach(function(tr){var off=!tr.live&&!(tr.status==="done"||tr.finalized);
    h+='<tr><th class="rh">Trainer '+esc(tr.id)+'<span class="ns">'+esc((tr.gpu||"")+" · step "+(fin(tr.step)?tr.step:"n/a")+(off?" · "+(tr.status||"lost"):tr.failure?" · failed":""))+"</span></th>";
    for(var f=0;f<P;f++){var v=tr.versions?tr.versions[f]:null,lg=D.done?0:lag(tr,f),s=tr.since?tr.since[f]:null,pushed=(tr.pushed||{})[String(f)]!=null;
      var cls=off||tr.failure?"off":lg>=2?"lag2":lg===1?"lag1":"",ready=fin(s)&&H&&s>=H;
      var mk=pushed?'<span class="mk dl">↑</span>':lg&&tr.live?'<span class="mk wt">↓'+lg+"</span>":"";
      var ahead=fin(v)&&v>D.versions[f],fin_=D.done&&fin(v)&&fin(D.versions[f])&&v<D.versions[f];if(fin_)finalOnSyncer=true;
      h+='<td class="'+cls+'" title="'+esc("anchor v"+(fin(v)?v:"n/a")+(fin_?": the island's last applied version; the final version v"+D.versions[f]+" is on the syncer (final cut)":"")+(lg?", "+lg+" merge(s) not applied":"")+(ahead?", ahead of the syncer tape":"")+(fin(s)?", "+s+" local step(s) since the anchor":"")+(pushed?", pushed t"+tr.pushed[String(f)]+", waiting for the merge":""))+'">'+mk+'<div class="v">'+(fin(v)?"v"+v:"—")+(lg?' <span class="'+(lg>=2?"bd":"wn")+'">−'+lg+"</span>":"")+"</div>"+
        (fin_?'<div class="s">last applied · final v'+D.versions[f]+' on the syncer</div>':"")+
        (fin(s)?'<div class="s">'+s+(H?"/"+H:"")+' since anchor</div><div class="b"><i class="'+(ready?"rdy":"")+'" style="width:'+Math.min(100,H?100*s/H:0)+'%"></i></div>':(tr.since?'<div class="s">since anchor —</div>':""))+"</td>"}
    h+="</tr>"});
  h+='</table><div class="fmlg">'+(sft()?'<span><b>v41</b> committed version (syncer) = global step of the fragment\'s last merge</span><span><b>t203</b> the island\'s last contribution: the merge its delta went into, base v = the version it was computed on</span><span class="dl">↑ PUSH accepted, waiting for the merge</span><span class="wt">↓n merged after the island\'s last applying step</span>':
    '<span><b>v41</b> committed (syncer) / anchor (trainer) version = global step of the fragment\'s last merge</span>'+(D.trainers.some(function(t){return t.since})?'<span>bar = local steps since anchor / H (green: ≥ H)</span>':"")+'<span class="dl">↑ pushed, waiting for the merge</span><span class="wt">↓n merged, not applied yet (on the wire or queued for the safe boundary)</span><span>tint: 1 / ≥ 2 merges behind</span>'+(finalOnSyncer?'<span><b>last applied</b> the island\'s last applied version; the final version is on the syncer (final cut)</span>':""))+'</div>';
  if(h!==fmKey){L.fm.innerHTML=h;fmKey=h}}

// an SFT learner has no anchor vector: its last contribution per fragment (responders[] of the merges)
function learnerRows(tn){var P=D.P||0,h="";
  D.trainers.forEach(function(tr){h+='<tr><th class="rh">Island '+esc(tr.id)+'<span class="ns">'+esc([tr.name,"step "+(fin(tr.step)?tr.step:"n/a"),tr.live?null:tr.status].filter(Boolean).join(" · "))+"</span></th>";
    for(var f=0;f<P;f++){var c=(tr.contrib||[])[f],lg=lag(tr,f),pu=(tr.pushed||{})[String(f)];
      var mk=pu!=null?'<span class="mk dl">↑</span>':lg&&tr.live?'<span class="mk wt">↓'+lg+"</span>":"";
      h+='<td class="'+(tr.live?"":"off")+'" title="'+esc((c?"last contribution: merge t"+c.step+", delta on base v"+c.base+(fin(c.staleness)?", staleness "+c.staleness:"")+", arrived "+dur(tn-c.ts)+" ago":"no contribution on the syncer tape")+
        (lg?"; "+lg+" merge(s) after the island's last applying step":"")+(pu!=null?"; PUSH accepted for t"+pu:""))+'">'+mk+'<div class="v">'+(c?"t"+c.step:"—")+'</div><div class="s">'+(c?"base v"+c.base+(c.staleness?" · stale "+c.staleness:""):"no PUSH")+"</div></td>"}
    h+="</tr>"});
  return h}

// -- particles -------------------------------------------------------------------------
function pathOf(f){var k=-1;D.trainers.forEach(function(t,i){if(t.id===f.i)k=i});if(k<0)return null;
  if(f.k==="p")return L.pair[k].row[f.sh]&&L.pair[k].row[f.sh].push;if(f.k==="b")return L.pair[k].row[f.sh]&&L.pair[k].row[f.sh].bcast;
  if(f.k==="q")return L.boxes[k].grp;if(f.k==="s")return L.boxes[k].pub;return null}
// the sample-buffer flows of a synced group (topology.py flows): groups written (sampler -> store),
// read (store -> trainer) and release uploads (trainer -> store) on the shared store or the island's lane
function bufferPath(f){var k=-1;
  if(f.k==="g"){if(L.store){(T.samplers||[]).forEach(function(s,i){if(s.id===f.s)k=i});return k>=0&&L.samp[k]?L.samp[k].lane:null}
    var hit=null;L.boxes.forEach(function(o){(o.samp||[]).forEach(function(x){if(x.card.s.id===f.s||(x.card.s.ids||[]).indexOf(f.s)>=0)hit=x.lane})});return hit}
  D.trainers.forEach(function(t,i){if(t.id===f.i)k=i});if(k<0)return null;
  var tk=L.store?L.trunk[k]:L.boxes[k];return tk?(f.k==="r"?tk.read:f.k==="u"?tk.up:null):null}
function bufferParticles(ctx,td){(T.flows||[]).forEach(function(f){var t0=f.k==="u"&&fin(f.u0)?f.u0:f.t,d=f.k==="u"&&fin(f.u0)?Math.max(1.2,f.t-f.u0):1.8,u=(td-t0)/d;
  if(u<0||u>1||"gru".indexOf(f.k)<0)return;var p=bufferPath(f);if(!p)return;var pt=U.at(p,f.k==="u"?(u*d%1.2)/1.2:u*u*(3-2*u)),c=f.k==="u"?C.wt:C.traj;
  ctx.fillStyle=c;ctx.beginPath();ctx.arc(pt[0],pt[1],f.k==="u"?2.6:2,0,6.2832);ctx.fill()})}
function particles(ctx,td){
  ctx.setTransform(L.dpr*L.scale,0,0,L.dpr*L.scale,0,0);ctx.font="600 9.5px ui-monospace,monospace";
  if(D.kind==="synced_group")bufferParticles(ctx,td);
  for(var i=0;i<D.flows.length;i++){var f=D.flows[i];if(f.t0>td)break;var d=Math.max(f.t1-f.t0,1.3),u=(td-f.t0)/d;if(u>1)continue;
    var p=pathOf(f);if(!p)continue;var c={p:C.delta,b:C.wt,q:C.traj,s:C.wt}[f.k],r=fin(f.b)?1.6+Math.max(0,Math.log10(f.b/1e6))*0.9:2.1;
    var e=u*u*(3-2*u),pt=U.at(p,e),bk=U.at(p,Math.max(0,e-0.05));
    ctx.strokeStyle=c;ctx.fillStyle=c;ctx.lineWidth=r*1.3;ctx.lineCap="round";ctx.globalAlpha=f.a?0.25:0.45;ctx.beginPath();ctx.moveTo(bk[0],bk[1]);ctx.lineTo(pt[0],pt[1]);ctx.stroke();
    ctx.globalAlpha=f.a?0.6:1;ctx.beginPath();ctx.arc(pt[0],pt[1],r,0,6.2832);ctx.fill();ctx.globalAlpha=1;
    if(f.k==="p"||f.k==="b")ctx.fillText("f"+f.f,pt[0]+5,pt[1]-5)}}
function frame(ts){raf=null;if(!L||!T)return;
  var dt=lastTick==null?0:Math.max(0,Math.min(0.1,(ts-lastTick)/1000));lastTick=ts;acc+=dt;if(acc>=0.25){acc=0;paint()}
  var ctx=L.ctx;if(!ctx)return;ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,L.cv.width,L.cv.height);
  if(T.live&&!U.reduced())particles(ctx,now()-DELAY_S);
  if(!document.hidden)raf=requestAnimationFrame(frame)}
function kick(){if(raf==null&&!document.hidden&&typeof requestAnimationFrame==="function"&&T&&T.live){lastTick=null;raf=requestAnimationFrame(frame)}}

// -- tooltips ------------------------------------------------------------------------
function row(k,v){return '<div class="row"><span>'+esc(k)+'</span><span>'+esc(v==null?"n/a":v)+"</span></div>"}
function tipHtml(key){var a=key.split(":"),tn=now(),tr=null;D.trainers.forEach(function(t,i){if(t.id===a[1]||(a[0]==="pair"&&String(i)===a[1]))tr=t});
  if(a[0]==="trn"&&tr&&sft()){var ln=tr.learner||{},lp=tr.last_push,la=tr.last_apply;return '<div class="h">Learner '+esc(tr.id+(tr.name?" · "+tr.name:""))+"</div>"+row("island status",tr.status)+row("step",(fin(tr.step)?tr.step:"n/a")+" / "+(tr.steps_total||"n/a"))+
    row("loss",fin(ln.loss)?ln.loss.toFixed(4):null)+row("learning rate",lr(tr.lr))+row("tokens / s",kn(ln.tps))+row("step time",fin(ln.step_s)?dur(ln.step_s):null)+row("parallel layout",ln.layout)+row("data",ln.dataset)+
    row("last PUSH",lp?"f"+lp.frag+" into t"+lp.step+" · base v"+lp.base+" · "+kn(lp.tokens)+" tok":null)+row("last apply",la?"step "+la.step+(fin(la.apply_s)?" · "+la.apply_s.toFixed(1)+"s":""):null)+row("merges since its last apply",tr.pending_n)+
    row("source","logging.jsonl · yeto_steps.jsonl · bench_metrics.jsonl · syncer tape responders")+'<div class="ns">an SFT learner writes no fragment-version vector and does not name the fragments it applied</div>'}
  if(a[0]==="trn"&&tr)return '<div class="h">Trainer '+esc(tr.id)+"</div>"+row("island status",tr.status)+row("local step",tr.step)+row("outer version",tr.outer)+row("policy hash",tr.hash)+
    row("fragment versions",tr.versions?"["+tr.versions.join(", ")+"]":null)+row("merged, not applied",tr.pending_n)+row("applied lr",lr(tr.lr))+
    row("last PUSH",tr.last_push?"f"+tr.last_push.frag+" t"+tr.last_push.step+" · "+mb(tr.last_push.bytes)+(tr.last_push.wire?" on the wire ("+mb(tr.last_push.payload_bytes)+" f32)":""):null)+
    row("BCAST overwrite / catch-up",tr.bcast_overwrites||tr.bcast_catch_ups?(tr.bcast_overwrites||0)+" / "+(tr.bcast_catch_ups||0):null)+row("last event",fin(tr.last_event_ts)?dur(tn-tr.last_event_ts)+" ago":null)+
    row("source",tr.full_push?"rl_full_parameter_round · rl_fragment_*":"rl_local_round · rl_policy_snapshot · rl_fragment_*")+'<div class="ns">PULL permits held and live BCAST queueing are not on the tapes yet</div>';
  if(a[0]==="rol"&&tr&&tr.island){var r=tr.island;return '<div class="h">Island '+esc(tr.id)+" rollout ("+esc(r.placement||"placement n/a")+")</div>"+row("execution mode",r.exec_mode)+row("engines",r.engines)+row("rollout round",r.round)+
    row("serving",r.served?r.served.token:null)+row("groups / traj / tokens",[r.groups,r.traj,r.tokens].join(" / "))+row("rollout seconds",fin(r.rollout_s)?r.rollout_s.toFixed(1):null)+
    row("group p50 / p95",fin(r.p50_s)?r.p50_s.toFixed(1)+"s / "+(fin(r.p95_s)?r.p95_s.toFixed(1)+"s":"—"):"not timed")+row("snapshot publish",r.served&&fin(r.served.publish_s)?r.served.publish_s.toFixed(1)+"s (publish phase)":null)+
    row("source","rl_local_round rl/* · rl_publication · rl_driver_phase")+'<div class="ns">trainer / rollout GPU split and generation progress are not on the tapes yet</div>'}
  if(a[0]==="node"&&tr&&a[2].charAt(0)==="s"&&fleet()&&((mine(tr)[+a[2].slice(1)]||{}).pool||{}).source==="miles"){var mc=mine(tr)[+a[2].slice(1)],mp=mc.pool;
    return '<div class="h">Miles native sampler '+esc(mc.id)+"</div>"+row("source","Miles sampler island of the launch (--rl-island-sample-role)")+row("in run.json",mp.source_name||"no fleet plan")+
      row("launch island",fin(mp.launch_island)?mp.launch_island+" · buffer shard "+(mp.shard||"island"+mp.launch_island):"not on the tape")+row("engine",mc.engine||"not on the tape")+row("cards",(mc.card||"GPU")+(fin(mc.gpus)?" × "+mc.gpus:""))+row("cloud",mp.provider||"not on the tape")+
      row("status",mc.n?(mc.live?mc.state:mc.status):"not started")+row("serving",fin(mc.policy_version)?"v"+mc.policy_version:"no release loaded")+row("feeds","trainer "+tr.id+" (R2 lane "+laneName(tr)+")")+
      row("source of the row",(D.fleet.source==="plan"?"fleet/plan.json miles_samplers · ":"")+"launch tape rl_sampler_start.store")}
  if(a[0]==="node"&&tr&&a[2].charAt(0)==="s"&&fleet()){var pc=mine(tr)[+a[2].slice(1)]||{},pp=pc.pool||{};return '<div class="h">'+(pp.source?"External pool ":"Pool ")+esc(pc.id||"")+"</div>"+
    row("pool in run.json",pp.source_name)+row("provider",pp.provider)+row("engine",pc.engine)+row("rollout precision",pc.precision)+row("cards",(pc.card||"GPU")+(fin(pc.gpus)?" × "+pc.gpus:""))+
    row("samplers (up / all)",pc.nlive+" / "+pc.n+(pp.samplers&&pp.samplers.length?" · "+pp.samplers.join(", "):""))+row("serving",fin(pc.policy_version)?"v"+pc.policy_version:"no release loaded")+
    row("feeds","trainer "+tr.id+" (R2 lane "+laneName(tr)+")")+row("source",(D.fleet.source==="plan"?"fleet/plan.json · ":"")+"rl_sampler_start.island / .store / .provenance")}
  if(a[0]==="node"&&tr&&a[2].charAt(0)==="s"){var sp=mine(tr)[+a[2].slice(1)]||{},sn=snode(sp.id);return '<div class="h">'+esc(a[2]+" · "+(sn.machine_full||"machine not on the tape"))+"</div>"+
    row("sampler",sp.id)+row("cards",(sn.gpu||sp.card||"GPU")+(fin(sn.gpus)?" × "+sn.gpus:""))+row("cloud",sn.cloud)+row("feeds","trainer "+tr.id+" (store lane "+laneName(tr)+")")+row("status",sn.status)}
  if(a[0]==="lane"&&tr){var ln=(D.lanes||{})[tr.id]||{};return '<div class="h">Store lane '+esc(laneName(tr))+"</div>"+row("samplers",(ln.samplers||[]).join(", "))+row("groups written",ln.written)+
    row("trainer read position",ln.read)+row("buffer",ln.backlog)+row("latest release","v"+ln.latest)+row("source","rl_sampler_start.store · rl_sampler_round · rl_policy_released")}
  if(a[0]==="node"&&tr){var n=(tr.nodes||[])[+a[2]]||{};return '<div class="h">'+esc(n.id+" · "+(n.machine_full||"machine not on the tape"))+"</div>"+row("cards",(n.gpu||"GPU")+(fin(n.gpus)?" × "+n.gpus:""))+
    row("roles",(n.roles||[]).join(" ")||"not on the tape")+(n.inferred?'<div class="ns">* inferred: a fixed-partition machine with 2 cards runs one trainer and one rollout card</div>':"")+row("incarnation",n.incarnation)+
    (n.previous?row("replaced",(n.reason||"")+" "+dur(tn-n.lost_ts)+" ago"):"")}
  if(a[0]==="snode")return '<div class="h">Syncer process</div>'+row("shards",D.shards.length)+'<div class="ns">the syncer\'s machine is not on the tapes yet</div>';
  if(a[0]==="syn"){var sh=D.shards[+a[1]],l=sh.last||{};return '<div class="h">'+(D.shards.length>1?"Syncer shard "+sh.index:"Syncer")+"</div>"+row("committed step",sh.step)+row("fragments",sh.lo+"–"+(sh.hi-1))+row("merges",sh.merges)+
    row("last merge","t"+l.step+" f"+l.frag+" base v"+l.base)+row("responded / expected",(l.responded||[]).join(",")+" / "+(l.expected||[]).join(","))+row("quorum wait",fin(l.quorum_s)?dur(l.quorum_s):null)+row("merge / checkpoint",(fin(l.merge_s)?l.merge_s.toFixed(2)+"s":"n/a")+" / "+(fin(l.checkpoint_s)?l.checkpoint_s.toFixed(2)+"s":"n/a"))+
    row("quorum wait p50 (recent)",fin(D.quorum_p50_s)?dur(D.quorum_p50_s):null)+row("source",D.config?"syncer tape merge records and syncer_config / round_open":"syncer tape merge records")+(D.config?row("config",cfgBits(D.config).join(" · ")):'<div class="ns">rounds whose PULL got no PUSH yet, T and τ are not on the tapes yet</div>')}
  if(a[0]==="pair"&&tr){var ps=D.flows.filter(function(f){return f.i===tr.id&&f.k==="p"}).slice(-4).reverse(),bs=D.flows.filter(function(f){return f.i===tr.id&&f.k==="b"}).slice(-4).reverse();
    return '<div class="h">'+(sft()?"Island ":"Trainer ")+esc(tr.id)+" ↔ syncer"+(D.shards.length>1?" shard "+a[2]:"")+'</div><div class="h">last PUSHes (wire window on the merge record)</div>'+ps.map(function(f){return row("f"+f.f,mb(f.b)+" · "+dur(f.t1-f.t0)+" · "+dur(tn-f.t1)+" ago")}).join("")+
      '<div class="h">last BCASTs</div>'+bs.map(function(f){return row("f"+f.f,(f.a?(sft()?dur(f.t1-f.t0)+" merge→next applying step":"applied time n/a"):dur(f.t1-f.t0)+" merge→received")+" · "+dur(tn-f.t0)+" ago")}).join("")}
  return ""}
function tip(e){var tp=$("topoTip");if(!tp||!L||!T)return;var g=e.target&&e.target.closest&&e.target.closest("[data-td]");
  if(!g){tp.style.display="none";return}var h=tipHtml(g.getAttribute("data-td"));if(!h){tp.style.display="none";return}
  tp.innerHTML=h;tp.style.display="block";var hr=L.host.getBoundingClientRect(),x=e.clientX-hr.left+14,y=e.clientY-hr.top+14;
  if(x+tp.offsetWidth>hr.width-4)x=e.clientX-hr.left-tp.offsetWidth-14;tp.style.left=Math.max(4,x)+"px";tp.style.top=y+"px"}

// -- legend --------------------------------------------------------------------------------
function legend(on){var lg=$("topoLg");if(!lg)return;if(v1Legend==null)v1Legend=lg.innerHTML;
  if(!on){if(lg.innerHTML!==v1Legend)lg.innerHTML=v1Legend;return}
  var isl=D.kind==="island";
  if(sft()){lg.innerHTML='<span style="color:var(--delta)"><i class="ln"></i>Fragment delta PUSH (learner → syncer)</span><span style="color:var(--wt)"><i class="ln"></i>Fragment BCAST (syncer → learner, applied at its next step boundary)</span>'+
    '<span><i style="background:var(--delta)"></i>T learner cards</span><span><i class="cd" style="background:var(--good)"></i>training</span><span><i class="cd" style="background:var(--ink2)"></i>stopped / restarting / done</span><span><i class="cd" style="background:var(--warn)"></i>waiting for quorum</span>';return}
  lg.innerHTML='<span style="color:var(--traj)"><i class="ln"></i>'+(isl?"Rollout groups (rollout → trainer, inside the island)":"Trajectories + log-probabilities (sampler → store → trainer)")+"</span>"+
    '<span style="color:var(--delta)"><i class="ln"></i>Fragment delta PUSH (trainer → syncer)</span>'+
    '<span style="color:var(--wt)"><i class="ln"></i>New weights: fragment BCAST (syncer → trainer), '+(isl?"snapshot publish (trainer → rollout)":"release (trainer → store → sampler)")+"</span>"+
    '<span><i style="background:var(--delta)"></i>T trainer card</span><span><i style="background:var(--traj)"></i>R rollout card</span>'+
    '<span><i class="cd" style="background:var(--good)"></i>running</span><span><i class="cd" style="background:var(--wt)"></i>applying / publishing</span><span><i class="cd" style="background:var(--warn)"></i>waiting / blocked</span><span><i class="cd" style="background:var(--bad)"></i>lost / failed</span>'+
    '<span style="color:var(--warn)"><i class="lnd"></i>blocked lane</span><span style="color:var(--bad)">✕ failed lane</span>'}

// -- entry ---------------------------------------------------------------------------------
// true when it draws the panel (a run with a syncer); otherwise topology.js draws V1 and the
// layout decision only goes into the panel's hover title
function render(view,offline){
  var box=$("topoBox"),host=$("topo");if(!box||!host)return false;
  var t=view&&view.topology,lay=(view&&view.layout)||{};
  if(!t||!t.diloco){if(T){T=null;D=null;L=null;legend(false)}if(lay.kind)box.setAttribute("title","layout: "+(lay.kind==="async"?"async-grpo":lay.kind)+" ("+lay.reason+")");return false}
  box.removeAttribute("title");
  U=YetoTopo.util;box.hidden=false;
  t.live=!offline&&((view.overview||{}).mode==="live");T=t;D=t.diloco;V=view;recvAt=U.clock();C=colors();
  // a fleet drawn from its plan before any merge record (P unknown): one empty syncer card
  if(!(D.shards||[]).length)D.shards=[{index:0,lo:0,hi:0,step:0,merges:0,state:"idle",attempt_max:1,last:null}];
  legend(true);
  var W=host.clientWidth||800;
  if(!L||L.shape!==shape()||(L.W<640)!==(W<640)||(host.contains&&!host.contains(L.svg)))build(host);
  var live=D.trainers.filter(function(x){return x.live}).length,rolls=D.trainers.filter(function(x){return x.island}).length;
  $("topoNote").textContent="layout: "+D.kind.replace("_"," ")+" ("+D.layout_reason+") · "+(sft()?D.trainers.length+" learner island(s), no rollout":D.kind==="island"?rolls+" island(s): trainer + in-island rollout":(t.samplers||[]).filter(function(s){return s.live}).length+" live sampler(s)"+(D.fleet?" in "+D.fleet.islands.length+" island(s)":""))+" · "+live+"/"+D.M+(sft()?" learner(s)":" trainer(s)")+" live · syncer "+(D.shards.length>1?D.shards.length+" shards":"1 process")+
    (t.live?" · replayed "+DELAY_S+" s behind the newest tape data":" · offline snapshot");
  paint();kick();
  if(!bound){bound=true;host.addEventListener("mousemove",tip);host.addEventListener("mouseleave",function(){var tp=$("topoTip");if(tp)tp.style.display="none"});
    if(document.addEventListener)document.addEventListener("visibilitychange",kick);
    if(typeof MutationObserver!=="undefined"){var mo=new MutationObserver(function(){C=colors();if(T&&L)paint()});mo.observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]})}
    var rt=null;if(typeof window!=="undefined"&&window.addEventListener)window.addEventListener("resize",function(){clearTimeout(rt);rt=setTimeout(function(){if(!T||!L)return;var w=host.clientWidth;if((w<640)!==(L.W<640))build(host);else fit();paint()},150)})}
  return true}

// -- the page's top cards and "Islands now" table for a DiLoCo run (sft.js) ----------------------
function ago(t,tn){return fin(t)?dur(tn-t)+" ago":"n/a"}
function kpis(view,kp){
  var t=view&&view.topology,d=t&&t.diloco;if(!d)return kp;U=YetoTopo.util;
  var tn=(view.overview||{}).now||t.now,trs=d.trainers,isl=d.kind==="island",live=trs.filter(function(x){return x.live}).length,rolls=trs.filter(function(x){return x.island});
  var eng=rolls.reduce(function(a,x){return a+(x.island.engines||0)},0),pls={};rolls.forEach(function(x){if(x.island.placement)pls[x.island.placement]=1});
  var pl=Object.keys(pls),co=rolls.some(function(x){return x.island.colocated});
  var pct=function(x){return fin(x)?Math.round(100*x)+"%":"—"},sh=d.shards[0]||{},open=d.rounds.length,pend=trs.filter(function(x){return x.live}).reduce(function(a,x){return a+x.pending_n},0);
  var oldest=trs.filter(function(x){return x.live&&fin(x.pending_oldest_ts)}).map(function(x){return x.pending_oldest_ts});
  var out=[];
  kp.forEach(function(c){var k=c[0];
    if(k==="Islands active / total"){if(isl)c=[k,c[1],"trainers "+live+"/"+trs.length+" · in-island rollout engines "+(eng||rolls.length)+" ("+(pl.join(", ")||"placement n/a")+(co?", shared GPU":"")+")"];out.push(c)}
    else if(k==="Groups trained / written"){if(isl){var gt=trs.reduce(function(a,x){return a+(x.trained_groups||0)},0),nr=rolls.some(function(x){return x.island.groups_src==="trained"}),gr=rolls.reduce(function(a,x){return a+(x.island.groups_total||0)},0);
      out.push(["Groups trained / rolled out",gt+" / "+(nr?"—":gr),nr?"in-island rollout does not report its groups (full parameter): every trained group came from it · "+trs.map(function(x){return "island "+x.id+" trained "+(x.trained_groups||0)}).join(" · "):
        trs.map(function(x){return "island "+x.id+" "+(x.trained_groups||0)+"/"+(x.island?x.island.groups_total:0)}).join(" · ")+" · in-island rollout, whole run"])}else out.push(c)}
    else if(k==="Policy version / releases"){var steps=trs.map(function(x){return fin(x.step)?x.step:"—"}).join(" · "),outs={};trs.forEach(function(x){if(fin(x.outer))outs[x.outer]=1});
      out.push(["Syncer step / local steps",d.shards.map(function(s){return s.step}).join("+")+" / "+steps,"outer "+(Object.keys(outs).length?"v"+Object.keys(outs).join(", v"):"n/a")+" · "+trs.map(function(x){return "trainer "+x.id+" local "+(fin(x.step)?x.step:"—")}).join(" · ")+" (local steps are per island, not comparable)"])}
    else if(k.indexOf("Rejected")===0){if(isl){var zv=rolls.map(function(x){return x.island.zero_var_recent}).filter(fin),tr=trs.map(function(x){return x.truncated}).filter(fin);
      out.push(["In-island rollout: zero-var / truncated",pct(zv.length?zv.reduce(function(a,b){return a+b},0)/zv.length:null)+" / "+(tr.length?pct(Math.max.apply(null,tr)):"—"),
        rolls.map(function(x){return "island "+x.id+" zero-var "+pct(x.island.zero_var_recent)}).join(" · ")+" (median of the last rounds)"+(tr.length?"":" · truncation not reported by the trainers")])}else out.push(c)}
    else if(k==="Buffer backlog"){var bb=["Fragment rounds in flight / BCAST pending",open+" / "+pend,(oldest.length?"oldest merged-not-applied "+dur(tn-Math.min.apply(null,oldest)):"every merged fragment applied")+(open?" · "+d.rounds.slice(0,3).map(function(r){return "t"+r.step+" f"+r.frag}).join(", "):"")+(d.config?"":" · PULL-only rounds not on the tapes")];
      if(isl)out.push(bb);else{out.push(c);out.push(bb)}}
    else if(k==="Syncer outer step")out.push(["Syncer",d.done?"done":"roster "+d.roster+"/"+d.M,(d.done?"every trainer finished · roster "+d.M+" · ":"")+"last merge "+ago(d.last_merge_ts_any,tn)+" · quorum wait p50 "+(fin(d.quorum_p50_s)?dur(d.quorum_p50_s):"—")+" · "+(sh.merges||0)+" merges · P "+(d.P||"—")+(d.behind?" · syncer tape behind the islands":"")]);
    else if(k==="Syncer RSS"){}  // the RL syncer's memory is not on its tape: no card rather than n/a
    else out.push(c)});
  return out}
function table(view){var d=view&&view.topology&&view.topology.diloco;if(!d)return null;U=YetoTopo.util;var by={};d.trainers.forEach(function(t){by[t.id]=t});
  return{trainers:by,P:d.P,
    step:function(c){var t=by[c.id];return t?(fin(t.step)?t.step:"—")+" / "+(fin(t.outer)?"outer v"+t.outer:"outer —"):null},
    push:function(c,tn){var t=by[c.id];if(!t)return null;var p=t.last_push,b=t.last_bcast;
      return (p?"↑ f"+p.frag+" t"+p.step+" · "+dur(tn-p.ts)+" ago":t.full_push?"↑ "+(t.full_push.count||0)+" PUSH last round, "+(t.full_push.total||0)+" in the run"+(t.full_push.overwritten?" · overwritten "+t.full_push.overwritten:""):"↑ none")+'<div class="ns">'+(b?"↓ f"+b.frag+" v"+b.v+" applied"+(fin(t.apply_s)?" · apply "+t.apply_s.toFixed(2)+"s":""):t.full_push?"↓ applied in the round (no per-fragment record yet)":"↓ no BCAST applied yet")+"</div>"+
        '<div class="ns">frag ['+(t.versions||[]).join(" ")+"]</div>"}}}

return{render:render,kpis:kpis,table:table};
})();
