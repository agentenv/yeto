// Live topology panel: samplers -> run store -> trainers (page view key "topology", topology.py).
// Groups written / read and policy releases / loads are tape events replayed at their own
// time_unix, DELAY_S behind the newest data, so the motion keeps the real order and pace
// instead of jumping every tape upload (30 s) and store poll (10 s).
var YetoTopo=(function(){
"use strict";
var NS="http:"+"//www.w3.org/2000/svg"; // SVG namespace id, not a request
var DELAY_S=45,MAX_CARDS=12,MAX_PARTICLES=900;
var ARCHS=["Ampere","Ada","Hopper","Blackwell","CDNA3","CDNA4","Ascend"];
var bound=false,T=null,V=null,L=null,P=[],C=null,recvAt=0,lastTd=null,raf=null,visible=true,io=null,lastTick=null,acc=0,reloadTick={},uploadTick={};

function $(id){return document.getElementById(id)}
function esc(s){return String(s==null?"":s).replace(/[&<>"]/g,function(c){return{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]})}
function fin(x){return typeof x==="number"&&isFinite(x)}
function el(tag,attrs,parent,text){var e=document.createElementNS(NS,tag);if(attrs)for(var k in attrs)e.setAttribute(k,attrs[k]);if(text!=null)e.textContent=text;if(parent)parent.appendChild(e);return e}
function dur(s){if(!fin(s))return "n/a";s=Math.max(0,Math.round(s));if(s<60)return s+"s";var m=Math.floor(s/60),r=s%60;return m<60?m+"m"+(r<10?"0":"")+r+"s":Math.floor(m/60)+"h"+(m%60)+"m"}
function bytes(b){return !fin(b)?"n/a":b>=1e9?(b/1e9).toFixed(2)+" GB":(b/1e6).toFixed(0)+" MB"}
function reduced(){return !!(typeof matchMedia==="function"&&matchMedia("(prefers-reduced-motion: reduce)").matches)}
function colors(){var cs=typeof getComputedStyle==="function"?getComputedStyle(document.documentElement):null,o={},g=function(n){return cs&&cs.getPropertyValue?String(cs.getPropertyValue(n)||"").trim():""};
  ["ink","ink2","line","good","warn","bad","traj","wt","wire","panel"].forEach(function(k){o[k]=g("--"+(k==="wire"?"topo-wire":k))});
  o.arch={};ARCHS.forEach(function(a){o.arch[a]=g("--arch-"+a.toLowerCase())});return o}
function archColor(a){return (C.arch[a]||C.ink2)}
function clock(){return typeof performance!=="undefined"&&performance.now?performance.now():Date.now()}
function now(){return T?T.now+(clock()-recvAt)/1000:0}

// -- geometry -------------------------------------------------------------------
function bez(p0,p1,p2,p3,n){var pts=[],len=[0],Lr=0;for(var i=0;i<=n;i++){var u=i/n,a=1-u;
  var x=a*a*a*p0[0]+3*a*a*u*p1[0]+3*a*u*u*p2[0]+u*u*u*p3[0],y=a*a*a*p0[1]+3*a*a*u*p1[1]+3*a*u*u*p2[1]+u*u*u*p3[1];
  if(i){var q=pts[i-1];Lr+=Math.hypot(x-q[0],y-q[1]);len.push(Lr)}pts.push([x,y])}return{pts:pts,len:len,L:Lr}}
function poly(pts){var len=[0],Lr=0;for(var i=1;i<pts.length;i++){Lr+=Math.hypot(pts[i][0]-pts[i-1][0],pts[i][1]-pts[i-1][1]);len.push(Lr)}return{pts:pts,len:len,L:Lr}}
function offset(p,d){var o=[],pts=p.pts;for(var i=0;i<pts.length;i++){var a=pts[Math.max(0,i-1)],b=pts[Math.min(pts.length-1,i+1)],dx=b[0]-a[0],dy=b[1]-a[1],l=Math.hypot(dx,dy)||1;o.push([pts[i][0]-dy/l*d,pts[i][1]+dx/l*d])}return poly(o)}
function rev(p){return poly(p.pts.slice().reverse())}
function dpath(p){return "M"+p.pts.map(function(q){return q[0].toFixed(1)+","+q[1].toFixed(1)}).join("L")}
function at(p,u){var t=Math.max(0,Math.min(1,u))*p.L,len=p.len,lo=0,hi=len.length-1;while(hi-lo>1){var m=(lo+hi)>>1;if(len[m]<t)lo=m;else hi=m}
  var f=(t-len[lo])/((len[hi]-len[lo])||1),a=p.pts[lo],b=p.pts[hi];return[a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f]}

// -- nodes: live samplers as cards (folded by card / precision / engine past MAX_CARDS) -----
function label(s){var p=[s.card,s.precision,s.engine].filter(Boolean);return p.length?p.join(" · "):s.id}
function nodes(t){
  var live=t.samplers.filter(function(s){return s.live}),ended=t.samplers.filter(function(s){return !s.live});
  if(live.length<=MAX_CARDS)return{cards:live.map(function(s){return{one:s,members:[s],key:s.id}}),ended:ended};
  var by={},order=[];
  live.forEach(function(s){var k=label(s);if(!by[k]){by[k]={members:[],key:k};order.push(k)}by[k].members.push(s)});
  order.sort(function(a,b){return by[b].members.length-by[a].members.length});
  var cards=order.slice(0,MAX_CARDS-1).map(function(k){return by[k]});
  if(order.length>MAX_CARDS-1){var rest={members:[],key:"other"};order.slice(MAX_CARDS-1).forEach(function(k){rest.members=rest.members.concat(by[k].members)});cards.push(rest)}
  cards.forEach(function(c){if(c.members.length===1)c.one=c.members[0]});
  return{cards:cards,ended:ended}}

function stText(s,tn){
  if(s.state==="reloading")return "reloading v"+s.reload.target+" · "+dur(tn-s.reload.since);
  if(s.state==="loading")return "loading v"+s.reload.target+" · "+dur(tn-s.reload.since);
  if(s.state==="waiting")return "backlog full";
  if(s.state==="starting")return "starting";
  if(s.state==="offline")return (s.status||"offline")+(s.reason?": "+s.reason:"");
  return s.status==="stale"?"sampling (quiet)":"sampling"}
function stColor(st){return{sampling:C.good,reloading:C.wt,loading:C.wt,waiting:C.warn,starting:C.warn,offline:C.bad}[st]||C.ink2}

// -- build the svg (on data shape / width change) -------------------------------------
function build(host){
  var t=T,N=nodes(t),n=N.cards.length,W=host.clientWidth||800,narrow=W<640;
  var trs=t.trainers.filter(function(x){return x.live});if(!trs.length&&t.trainers.length)trs=[t.trainers[0]];
  var VW,VH,cw,ch,gap,xs=[],ys=[],R2,TRS=[];
  if(!narrow){VW=1200;cw=272;ch=62;gap=10;VH=Math.max(250,24+n*(ch+gap),trs.length*160+(trs.length===1?100:20));
    for(var i=0;i<n;i++){xs.push(10+(n>2?40*Math.sin(Math.PI*(i+0.5)/n):0));ys.push(VH/2-(n*(ch+gap)-gap)/2+i*(ch+gap))}
    R2={cx:650,cy:VH/2-(n<=2?4:16),r:54};
    trs.forEach(function(x,k){TRS.push({x:VW-302,y:(trs.length===1?Math.max(10,VH/2-120):VH/2-(trs.length*160-10)/2)+k*160,w:292,h:150})})}
  else{VW=400;cw=200;ch=58;gap=8;var colH=Math.max(n*(ch+gap),200);
    for(var j=0;j<n;j++){xs.push(4+(n>2?12*Math.sin(Math.PI*(j+0.5)/n):0));ys.push(6+j*(ch+gap))}
    R2={cx:326,cy:Math.min(colH/2-26,170),r:40};
    trs.forEach(function(x,k){TRS.push({x:4,y:colH+60+k*160,w:VW-8,h:150})});VH=(TRS.length?TRS[TRS.length-1].y+158:colH+60)}
  host.innerHTML="";
  var wrap=document.createElement("div");wrap.className="tpw";host.appendChild(wrap);
  var svg=el("svg",{viewBox:"0 0 "+VW+" "+VH,role:"img","aria-label":"samplers to run store to trainers"},wrap);
  var wires=el("g",{},svg),labels=el("g",{},svg),nd=el("g",{},svg);
  var traj=[],wt=[],lanes=[],cards=[],counters=[],idx={};
  N.cards.forEach(function(c,k){c.members.forEach(function(s){idx[s.id]=k});
    var sx=xs[k]+cw,sy=ys[k]+ch/2,ex=R2.cx-R2.r-10,ey=R2.cy+(k-(n-1)/2)*(narrow?3:Math.min(7,90/Math.max(n,1))),bend=(ex-sx)*0.55;
    var base=bez([sx,sy],[sx+bend,sy],[ex-bend,ey],[ex,ey],40),a=offset(base,-2.6),b=offset(base,2.6);
    traj.push(a);wt.push(rev(b));
    lanes.push([el("path",{d:dpath(a),"class":"tpwire"},wires),el("path",{d:dpath(b),"class":"tpwt",opacity:0},wires)]);
    var m=at(base,0.5);counters.push(el("text",{x:m[0],y:m[1]-5,"text-anchor":"middle","class":"tpc",display:"none"},labels,""));
    cards.push(card(nd,xs[k],ys[k],cw,ch,c))});
  var trunks=[];
  TRS.forEach(function(B,k){var p;
    if(!narrow){var y=B.y+B.h/2;p=bez([R2.cx+R2.r+10,R2.cy],[R2.cx+R2.r+90,R2.cy],[B.x-90,y],[B.x,y],30)}
    else p=poly([[R2.cx,R2.cy+R2.r+64],[R2.cx,R2.cy+R2.r+64+(B.y-R2.cy-R2.r-64)*0.5],[B.x+B.w*0.75,B.y]]);
    var a=offset(p,narrow?3.5:-3.5),b=offset(p,narrow?-3.5:3.5);
    el("path",{d:dpath(a),"class":"tpwire tpbold"},wires);el("path",{d:dpath(b),"class":"tpwire tpbold"},wires);
    trunks.push({read:a,up:rev(b)})});
  if(!narrow&&TRS.length){var mx=(R2.cx+R2.r+TRS[0].x)/2,y0=Math.min(R2.cy,TRS[0].y+TRS[0].h/2);
    el("text",{x:mx,y:y0-12,"text-anchor":"middle","class":"tpc tptraj"},labels,"Trajectories + log-probs →");
    el("text",{x:mx,y:Math.max(R2.cy,TRS[0].y+TRS[0].h/2)+22,"text-anchor":"middle","class":"tpc tpwtc"},labels,"← New weights")}
  var r2=store(nd,R2,narrow),trc=TRS.map(function(B,k){return trainer(nd,B,trs[k])});
  var stats=[];
  if(!narrow&&TRS.length===1){var sg=el("g",{transform:"translate("+(TRS[0].x+14)+","+(TRS[0].y+TRS[0].h+28)+")"},nd);
    el("text",{x:0,y:0,"class":"tpt"},sg,"Last 5 min");for(var q=0;q<3;q++)stats.push(el("text",{x:0,y:18+q*16,"class":"tpm"},sg,""))}
  var cv=document.createElement("canvas");cv.className="tpcv";wrap.appendChild(cv);
  var foot=document.createElement("div");foot.className="tpfoot";host.appendChild(foot);
  L={host:host,svg:svg,cv:cv,ctx:cv.getContext?cv.getContext("2d"):null,VW:VW,VH:VH,narrow:narrow,W:W,N:N,idx:idx,traj:traj,wt:wt,lanes:lanes,cards:cards,counters:counters,
     trunks:trunks,trs:trs,r2:r2,trc:trc,stats:stats,foot:foot,shape:shape(t)};
  fit();P=[]}
function fit(){if(!L||!L.svg.getBoundingClientRect)return;var r=L.svg.getBoundingClientRect(),dpr=Math.min(2,(typeof window!=="undefined"&&window.devicePixelRatio)||1);
  L.cv.style.width=r.width+"px";L.cv.style.height=r.height+"px";L.cv.width=Math.max(1,Math.round(r.width*dpr));L.cv.height=Math.max(1,Math.round(r.height*dpr));
  L.scale=r.width/L.VW;L.dpr=dpr}
function shape(t){var N=nodes(t),trs=t.trainers.filter(function(x){return x.live});
  return N.cards.map(function(c){return c.key+":"+c.members.length}).join("|")+"#"+(trs.length?trs:t.trainers.slice(0,1)).map(function(x){return x.id}).join(",")}

function card(parent,x,y,w,h,c){
  var compact=w<230,g=el("g",{transform:"translate("+x+","+y+")","data-tp":c.key},parent),one=c.one,s0=c.members[0];
  var box=el("rect",{width:w,height:h,rx:9,"class":"tpbox"},g);
  var archs={};c.members.forEach(function(s){archs[s.arch||"?"]=1});var al=Object.keys(archs);
  al.forEach(function(a,i){el("rect",{x:0,y:i*h/al.length,width:4,height:h/al.length,rx:2,fill:archColor(a)},g)});
  var title=c.key==="other"?"Other samplers":(compact?[s0.card,s0.precision].filter(Boolean).join(" · ")||s0.id:label(s0));
  el("text",{x:13,y:18,"class":compact?"tpt tps":"tpt"},g,title+(c.members.length>1?" ×"+c.members.length:""));
  var sub=one?(compact?[one.engine,one.arch].filter(Boolean).join(" · ")||one.id:[one.id,one.arch,one.gpus?one.gpus+" GPU":null].filter(Boolean).join(" · "))
    :c.members.length+" samplers · "+al.filter(function(a){return a!=="?"}).join(" / ");
  el("text",{x:13,y:33,"class":"tpm"},g,sub);
  var ver=el("text",{x:w-10,y:18,"text-anchor":"end","class":"tpv"},g,"");
  var dot=el("circle",{cx:16.5,cy:h-12.5,r:3.5},g),st=el("text",{x:25,y:h-9,"class":"tps2"},g,""),rate=el("text",{x:w-10,y:h-9,"text-anchor":"end","class":"tpm"},g,"");
  var flash=el("rect",{width:w,height:h,rx:9,fill:"none","stroke-width":1.5,opacity:0},g);
  return{g:g,box:box,ver:ver,dot:dot,st:st,rate:rate,flash:flash,c:c,compact:compact}}

function store(parent,R,narrow){
  var g=el("g",{},parent),cx=R.cx,cy=R.cy,r=R.r;
  el("circle",{cx:cx,cy:cy,r:r+7,fill:"none","class":"tpring"},g);
  var ring=el("circle",{cx:cx,cy:cy,r:r+7,fill:"none","stroke-width":5,"stroke-linecap":"round",transform:"rotate(-90 "+cx+" "+cy+")"},g);
  el("circle",{cx:cx,cy:cy,r:r,"class":"tpbox"},g);
  var gw=r*0.56,gh=r*0.3,gx=cx-gw/2,gy=cy-r*0.66;
  el("path",{d:"M"+gx+","+(gy+5)+"v"+gh+"a"+gw/2+",5 0 0 0 "+gw+",0v-"+gh,"class":"tpglyph"},g);
  el("ellipse",{cx:cx,cy:gy+5,rx:gw/2,ry:5,"class":"tpglyph"},g);
  el("text",{x:cx,y:cy+r*0.24,"text-anchor":"middle","class":"tpt tpbig"},g,"Store");
  var lat=el("text",{x:cx,y:cy+r*0.5,"text-anchor":"middle","class":"tpv tpwtc"},g,"");
  var info=[],iy=cy+r+28;for(var i=0;i<3;i++)info.push(el("text",{x:cx,y:iy+i*15,"text-anchor":"middle","class":i?"tpm":"tpv"},g,""));
  return{ring:ring,C2:2*Math.PI*(r+7),lat:lat,info:info,narrow:narrow}}

function trainer(parent,B,tr){
  var g=el("g",{transform:"translate("+B.x+","+B.y+")"},parent),w=B.w,h=B.h;
  var box=el("rect",{width:w,height:h,rx:10,"class":"tpbox"},g);
  el("rect",{x:0,y:0,width:4,height:h,rx:2,fill:archColor(tr&&tr.arch)},g);
  el("text",{x:14,y:20,"class":"tpt"},g,"Trainer"+(tr&&tr.name?" · "+tr.name:""));
  el("text",{x:14,y:36,"class":"tpm"},g,tr&&tr.gpu?(tr.gpus?tr.gpus+" × ":"")+tr.gpu:(tr?tr.id:""));
  var step=el("text",{x:14,y:66,"class":"tpbig2"},g,""),ver=el("text",{x:w-12,y:66,"text-anchor":"end","class":"tpm"},g,"");
  var dot=el("circle",{cx:18,cy:86,r:3.5},g),ph=el("text",{x:27,y:90,"class":"tps2"},g,"");
  var up=el("text",{x:14,y:116,"class":"tpm"},g,""),up2=el("text",{x:14,y:h-12,"class":"tpm"},g,"");
  return{g:g,box:box,step:step,ver:ver,dot:dot,ph:ph,up:up,up2:up2}}

// -- per-tick text / colors (4 Hz) ---------------------------------------------------
function paint(){
  var t=T,tn=now(),red=reduced();
  L.cards.forEach(function(o,k){var c=o.c,ms=c.members,one=c.one;
    var lags=ms.map(function(s){return s.lag}).filter(fin),lag=lags.length?Math.max.apply(null,lags):0;
    var vs=ms.map(function(s){return s.policy_version}).filter(fin),vmin=vs.length?Math.min.apply(null,vs):null,vmax=vs.length?Math.max.apply(null,vs):null;
    o.ver.textContent=vmin==null?"v?":(vmin===vmax?"v"+vmin:"v"+vmin+"–"+vmax)+(lag>0?" −"+lag:"");
    o.ver.style.fill=lag>=2?C.bad:lag===1?C.warn:C.ink;
    var st,col;
    if(one){st=stText(one,tn);col=stColor(one.state)}
    else{var cnt={};ms.forEach(function(s){cnt[s.state]=(cnt[s.state]||0)+1});
      var ord=["sampling","reloading","loading","waiting","starting"];st=ord.filter(function(k){return cnt[k]}).map(function(k){return cnt[k]+" "+k}).join(" · ");
      col=cnt.reloading||cnt.loading?C.wt:cnt.waiting||cnt.starting?C.warn:C.good}
    if(o.compact&&st.length>19)st=st.replace("reloading","reload").replace(" · "," ");
    o.st.textContent=st;o.st.style.fill=(one?one.state==="sampling":col===C.good)?C.ink:col;
    o.dot.setAttribute("fill",col);
    var rate=ms.reduce(function(a,s){return a+(s.rate_gpm||0)},0);o.rate.textContent=rate.toFixed(1)+" g/min";
    var stale=ms.some(function(s){return s.status==="stale"});
    o.box.style.stroke=stale?C.bad:lag>=2?C.warn:"";o.box.style.strokeDasharray=stale?"4 3":"";
    var lg=Math.max.apply(null,ms.map(function(s){return fin(s.last_group_ts)?s.last_group_ts:-1e18})),age=tn-DELAY_S-lg;
    o.flash.style.stroke=C.traj;o.flash.setAttribute("opacity",lg>0&&age>=0?Math.max(0,1-age/8)*0.9:0);
    var on=ms.some(function(s){return s.state==="reloading"});
    L.lanes[k][1].setAttribute("opacity",on?0.75:0);L.lanes[k][1].style.stroke=C.wt;
    var cn=L.counters[k];cn.setAttribute("display",red?"inline":"none");
    if(red){var g5=t.flows.filter(function(f){return f.k==="g"&&L.idx[f.s]===k&&f.t>t.now-300}).reduce(function(a,f){return a+(f.m||1)},0);cn.textContent=g5+" groups / 5 min"}});
  var b=t.buffer||{},rl=t.last_release,f=fin(b.backlog)?Math.min(1,b.backlog/Math.max(24,b.backlog)):0;
  L.r2.ring.style.stroke=C.traj;L.r2.ring.setAttribute("stroke-dasharray",(L.r2.C2*f).toFixed(1)+" "+L.r2.C2.toFixed(1));
  L.r2.lat.textContent=fin(t.latest)?"LATEST v"+t.latest:"no release";
  L.r2.info[0].textContent=fin(b.backlog)?"buffer "+Math.round(b.backlog)+" groups":"buffer n/a";
  L.r2.info[1].textContent=fin(b.written)?"written "+Math.round(b.written)+(fin(b.read)?" − read "+Math.round(b.read):""):"";
  L.r2.info[2].textContent=rl?(L.r2.narrow?"":"last release ")+(rl.kind||"")+" "+bytes(rl.bytes)+" · "+dur(tn-rl.ts)+" ago":"no release yet";
  L.trc.forEach(function(o,k){var tr=L.trs[k];
    // the round the "Islands now" table shows: the trainer's last round record
    var rr=(((V||{}).islands||{})[tr.id]||{}).rounds||[],lr=null;for(var i=rr.length-1;i>=0;i--)if(fin(rr[i].ts)&&rr[i].round!=null){lr=rr[i].round;break}
    o.step.textContent=lr!=null?"round "+lr:fin(tr.step)?"round "+tr.step:"round n/a";o.ver.textContent=fin(tr.policy_version)?"policy v"+tr.policy_version:"";
    var txt,col;
    if(!tr.live){txt=(tr.status||"ended")+(tr.reason?": "+tr.reason:"");col=C.ink2}
    else if(tr.starting||tr.status==="starting"){txt="starting";col=C.warn}
    else if(tr.phase){var age=fin(tr.phase_since)?tn-tr.phase_since:null;txt=tr.phase+(age!=null?" · "+dur(age):"");col=tr.phase==="training"?C.good:tr.phase==="waiting for data"&&age>30?C.warn:C.ink2}
    else{txt=tr.status==="stale"?"quiet":"running";col=tr.status==="stale"?C.bad:C.good}
    o.ph.textContent=txt;o.ph.style.fill=col===C.good?C.ink:col;o.dot.setAttribute("fill",col);
    o.g.setAttribute("opacity",tr.live?1:0.6);
    var nb=fin(b.backlog)?Math.round(b.backlog):null;
    o.up.textContent=nb==null?"":"buffer "+nb+" groups ready";
    o.up2.textContent=rl&&fin(rl.upload_s)?"last upload "+dur(rl.upload_s)+" ("+(rl.kind||"")+" "+bytes(rl.bytes)+")":""});
  if(L.stats.length){var g5=0,rd=0,ld=[],tc=t.now-300;t.flows.forEach(function(x){if(x.t<=tc)return;if(x.k==="g")g5+=x.m||1;else if(x.k==="r")rd+=x.m||0;else if(x.k==="l"&&fin(x.seen))ld.push(x.t-x.seen)});
    ld.sort(function(a,b){return a-b});
    L.stats[0].textContent=g5+" groups written · "+rd+" read";
    L.stats[1].textContent=ld.length+" sampler loads"+(ld.length?" · seen→loaded p50 "+dur(ld[ld.length>>1]):"");
    L.stats[2].textContent=T.live?"motion replays the tapes "+DELAY_S+" s behind":"offline: last state, no motion"}
  var en=L.N.ended;
  L.foot.textContent=en.length?en.length+" sampler(s) ended (left / lost / done / stopped): "+en.slice(0,8).map(function(s){return s.id+" "+s.status}).join(", ")+(en.length>8?", …":""):""}

// -- particles ------------------------------------------------------------------
function add(path,o){if(!path)return;if(P.length>=MAX_PARTICLES)P.shift();P.push({p:path,age:-(o.delay||0),dur:o.dur||1.5,r:o.r||2,c:o.c,tail:o.tail||0})}
function spawn(f){
  var k=L.idx[f.s];
  if(f.k==="g"&&k!=null){for(var i=0;i<Math.min(f.m||1,6);i++)add(L.traj[k],{delay:i*0.15,dur:1.8,r:2,c:C.traj,tail:0.012*(f.n||8)})}
  else if(f.k==="r"&&L.trunks.length){var tk=L.trunks[Math.max(0,L.trs.map(function(x){return x.id}).indexOf(f.i))]||L.trunks[0];
    for(var j=0;j<Math.min(f.m||0,32);j++)add(tk.read,{delay:j*0.15,dur:1.2,r:2.4,c:C.traj,tail:0.05})}
  else if(f.k==="l"&&k!=null){var o=L.cards[k];if(o){o.flash.style.stroke=C.wt;o.flash.setAttribute("opacity",0.9)}}}
function step(dt,ctx){var keep=[];for(var i=0;i<P.length;i++){var q=P[i];q.age+=dt;if(q.age<0){keep.push(q);continue}var u=q.age/q.dur;if(u>=1)continue;keep.push(q);
  var e=u*u*(3-2*u),pt=at(q.p,e);ctx.fillStyle=q.c;
  if(q.tail){var b=at(q.p,Math.max(0,e-q.tail));ctx.strokeStyle=q.c;ctx.lineWidth=q.r*1.3;ctx.lineCap="round";ctx.globalAlpha=0.45;ctx.beginPath();ctx.moveTo(b[0],b[1]);ctx.lineTo(pt[0],pt[1]);ctx.stroke();ctx.globalAlpha=1}
  ctx.beginPath();ctx.arc(pt[0],pt[1],q.r,0,6.2832);ctx.fill()}P=keep}

function frame(ts){
  raf=null;if(!L||!T)return;
  var dt=lastTick==null?0:Math.max(0,Math.min(0.1,(ts-lastTick)/1000));lastTick=ts;acc+=dt;
  if(acc>=0.25){acc=0;paint()}
  var ctx=L.ctx;if(!ctx)return;ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,L.cv.width,L.cv.height);
  if(T.live&&!reduced()){
    ctx.setTransform(L.dpr*L.scale,0,0,L.dpr*L.scale,0,0);
    var td=now()-DELAY_S;
    if(lastTd==null||td<lastTd||td-lastTd>30)lastTd=td-0.001;
    for(var i=0;i<T.flows.length;i++){var f=T.flows[i];if(f.t>lastTd&&f.t<=td)spawn(f)}
    // weights: the release's upload window (trainer -> store), and every sampler still
    // reloading a release (store -> sampler; its exact fetch window is only known once loaded)
    T.flows.forEach(function(f){if(f.k==="u"&&fin(f.u0)&&td>=f.u0&&td<=f.t&&L.trunks.length){var key=f.i+":"+f.v;
      if(td-(uploadTick[key]||-1e18)>=2.5){uploadTick[key]=td;var tk=L.trunks[Math.max(0,L.trs.map(function(x){return x.id}).indexOf(f.i))]||L.trunks[0];
        add(tk.up,{dur:1.2,r:f.kind==="full"?3:1.8,c:C.wt,tail:f.kind==="full"?0.09:0.035})}}});
    L.cards.forEach(function(o,k){if(!o.c.members.some(function(s){return s.state==="reloading"}))return;
      if(td-(reloadTick[k]||-1e18)>=2){reloadTick[k]=td;add(L.wt[k],{dur:1.6,r:2.2,c:C.wt,tail:0.05})}});
    lastTd=td;step(dt,ctx)}
  if(visible&&!document.hidden)raf=requestAnimationFrame(frame)}
function kick(){if(raf==null&&visible&&!document.hidden&&typeof requestAnimationFrame==="function"&&T&&T.live){lastTick=null;raf=requestAnimationFrame(frame)}}

function tip(e){
  var tp=$("topoTip");if(!tp||!L)return;var g=e.target.closest&&e.target.closest("[data-tp]");
  if(!g){tp.style.display="none";return}
  var c=L.cards.filter(function(o){return o.c.key===g.getAttribute("data-tp")})[0];if(!c){tp.style.display="none";return}
  var tn=now(),row=function(k,v){return '<div class="row"><span>'+esc(k)+'</span><span>'+esc(v)+'</span></div>'};
  var h=c.c.members.slice(0,10).map(function(s){return '<div class="h">'+esc(label(s))+'</div>'+row("sampler",s.id)+row("island status",s.status)+row("state",stText(s,tn))+
    row("policy",fin(s.policy_version)?"v"+s.policy_version+(s.lag>0?" ("+s.lag+" behind)":""):"n/a")+row("groups / min (5 min)",(s.rate_gpm||0).toFixed(2))+
    row("last group",fin(s.last_group_ts)?dur(T.now-s.last_group_ts)+" ago":"n/a")}).join("")+(c.c.members.length>10?'<div class="ns">+'+(c.c.members.length-10)+' more</div>':"");
  tp.innerHTML=h;tp.style.display="block";
  var hr=L.host.getBoundingClientRect(),x=e.clientX-hr.left+14,y=e.clientY-hr.top+14;
  if(x+tp.offsetWidth>hr.width-4)x=e.clientX-hr.left-tp.offsetWidth-14;tp.style.left=Math.max(4,x)+"px";tp.style.top=y+"px"}

function render(view,offline){
  var box=$("topoBox"),host=$("topo");if(!box||!host)return;
  var t=view&&view.topology;
  if(!t||!t.samplers||!t.samplers.length){box.hidden=true;T=null;return}
  box.hidden=false;
  t.live=!offline&&((view.overview||{}).mode==="live");
  var old=T;T=t;V=view;recvAt=clock();C=colors();
  var W=host.clientWidth||800;
  if(!L||L.shape!==shape(t)||(L.W<640)!==(W<640)||(host.contains&&!host.contains(L.svg)))build(host);
  $("topoNote").textContent=t.samplers.filter(function(s){return s.live}).length+" live sampler(s) · "+t.trainers.filter(function(x){return x.live}).length+" trainer(s)"+
    (t.live?" · groups and releases replayed "+DELAY_S+" s behind the newest tape data":" · offline snapshot");
  paint();
  if(!io&&typeof IntersectionObserver!=="undefined"){io=new IntersectionObserver(function(es){visible=es[0].isIntersecting;kick()});io.observe(box)}
  kick();
  if(!old&&!bound){bound=true;host.addEventListener("mousemove",tip);host.addEventListener("mouseleave",function(){var tp=$("topoTip");if(tp)tp.style.display="none"});
    if(document.addEventListener)document.addEventListener("visibilitychange",kick);
    if(typeof MutationObserver!=="undefined"){var mo=new MutationObserver(function(){C=colors();if(T)paint()});mo.observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]})}
    if(typeof matchMedia==="function"){var mq=matchMedia("(prefers-color-scheme: light)");if(mq&&mq.addEventListener)mq.addEventListener("change",function(){C=colors();if(T)paint()})}
    var rt=null;if(typeof window!=="undefined"&&window.addEventListener)window.addEventListener("resize",function(){clearTimeout(rt);rt=setTimeout(function(){if(!T)return;var w=host.clientWidth;if((w<640)!==(L.W<640))build(host);else fit();paint()},150)})}}

return{render:render,_nodes:nodes,_label:label,_particles:function(){return P.length}};
})();
