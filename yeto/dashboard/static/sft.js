// yeto dashboard: metrics layout (dark ops style) for SFT runs, plus the shared header
// controls (time range, reload) and the RL distribution panel. Loaded before app.js in the
// same <script>; app.js calls YetoExt.render(V, OFFLINE) after its own render.
// Hand-written SVG; no external requests. Every number comes from the reducer view.
var YetoExt=(function(){
"use strict";
var PAL=["#3b82f6","#f97316","#14b8a6","#8b5cf6","#ec4899","#eab308","#22c55e","#06b6d4","#ef4444","#a3e635"];
var GREY="#9ca3af",RANGES=[["run",0],["1h",1],["6h",6],["24h",24],["7d",168],["30d",720]];
// "run" (the RL default; SFT keeps 24 h): the window is the run itself, first event -> now, so a one-hour run fills the charts
var RANGE_H=24,RANGE_RUN=null,CH={},V=null,OFF=false,NOW=0,onReload=null,EXPAND={};
// END: the right edge of every time window. NOW while the run is live; once it ended (runEnded), its last
// event plus a little room, so a finished run is not squeezed into the left edge of a window that runs to now
var END=0,ENDED=false,RUN_SPAN=null;
try{var sr=localStorage.getItem("yeto-range");if(sr&&sr!=="run"&&+sr>0){RANGE_H=+sr;RANGE_RUN=false}}catch(e){}
try{var hm=/range=(\d+|run)/.exec(location.hash||"");if(hm&&hm[1]!=="run"){RANGE_H=+hm[1];RANGE_RUN=false}else if(hm)RANGE_RUN=true}catch(e){}
var RANGE_SET=RANGE_RUN!==null;
function $(id){return document.getElementById(id)}
function esc(s){return String(s==null?"":s).replace(/[&<>"']/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]})}
function fin(v){return typeof v==="number"&&isFinite(v)}
function k(v,d){if(!fin(v))return "n/a";var a=Math.abs(v);if(a>=1e12)return (v/1e12).toFixed(2)+"T";if(a>=1e9)return (v/1e9).toFixed(2)+"B";if(a>=1e6)return (v/1e6).toFixed(1)+"M";if(a>=1e4)return (v/1e3).toFixed(1)+"k";
  if(a>=100)return v.toFixed(0);if(a>=1)return v.toFixed(d==null?2:d);if(a===0)return "0";return v.toPrecision(3)}
function dur(s){if(!fin(s))return "n/a";var a=Math.abs(s);if(a<1)return (s*1000).toFixed(0)+" ms";if(a<100)return s.toFixed(a<10?2:1)+" s";if(a<3600)return a<600?s.toFixed(0)+" s":(s/60).toFixed(1)+" min";if(a<86400*2)return (s/3600).toFixed(1)+" h";return (s/86400).toFixed(1)+" d"}
function gib(b){return fin(b)?(b/1073741824):null}
function bytes(b){if(!fin(b))return "n/a";if(b>=1e12)return (b/1e12).toFixed(2)+" TB";return (b/1073741824).toFixed(0)+" GiB"}
function p2(n){return (n<10?"0":"")+n}
function clock(t,sec){if(!fin(t))return "n/a";var d=new Date(t*1000);return p2(d.getHours())+":"+p2(d.getMinutes())+(sec?":"+p2(d.getSeconds()):"")}
function day(t){if(!fin(t))return "n/a";var d=new Date(t*1000);return (d.getMonth()+1)+"/"+d.getDate()+" "+clock(t)}
function ago(t){return fin(t)?dur(Math.max(0,NOW-t))+" ago":"n/a"}
function pctl(sorted,q){return rank(sorted,q)} // one quantile rule for the whole page (nearest rank)
function sum(a){var s=0;a.forEach(function(v){if(fin(v))s+=v});return s}
function nice(lo,hi,n){if(!(hi>lo)){hi=lo+1}var step=Math.pow(10,Math.floor(Math.log10((hi-lo)/n))),err=(hi-lo)/n/step;
  step*=err>=7.5?10:err>=3.5?5:err>=1.5?2:1;var a=Math.floor(lo/step)*step,b=Math.ceil(hi/step)*step,t=[];for(var v=a;v<=b+step/2;v+=step)t.push(+v.toPrecision(12));return t}

/* duration axis labels: one unit for every tick of an axis (dur() switches unit per value) */
function durUnit(m){m=Math.abs(m);return m<1?[0.001,"ms"]:m<=600?[1,"s"]:m<3*3600?[60,"min"]:m<2*86400?[3600,"h"]:[86400,"d"]}
function durTicks(lo,hi,n){var u=durUnit(Math.max(Math.abs(lo),Math.abs(hi)))[0];return nice(lo/u,hi/u,n).map(function(v){return +(v*u).toPrecision(12)})} // round numbers in the label unit
function durAxis(ticks){var m=Math.max.apply(null,ticks.map(Math.abs)),u=durUnit(m),st=ticks.length>1?Math.abs(ticks[1]-ticks[0]):m||1,q=st/u[0],
    dec=q>=0.999?0:Math.min(3,Math.ceil(-Math.log10(q)-1e-9));
  return function(v){return v===0?"0":(v/u[0]).toFixed(dec)+" "+u[1]}}
/* numeric axis labels: one decimal count for every tick (from the tick step), scientific
   notation below 1e-3 so 0.000200 does not overflow the left margin */
function numAxis(ticks){var m=Math.max.apply(null,ticks.map(Math.abs)),st=ticks.length>1?Math.abs(ticks[1]-ticks[0]):(m||1);
  if(m>=1e4)return function(v){return k(v)};
  if(m>0&&m<1e-3){var e=Math.floor(Math.log10(m)),dd=Math.max(0,Math.min(3,Math.ceil(-Math.log10(st/Math.pow(10,e))-1e-9)));
    return function(v){return v===0?"0":(v/Math.pow(10,e)).toFixed(dd)+"e"+e}}
  var dec=Math.max(0,Math.min(6,Math.ceil(-Math.log10(st)-1e-9)));return function(v){return v.toFixed(dec)}}
var SPARSE=4; // a series with this many points or fewer is drawn as dots with a faint connector
/* stacked series on one time grid: each series is linearly interpolated between its own samples
   (islands step at different moments, so summing raw step points saw-tooths). Samples outside the
   range anchor the edges. Inside a gap longer than brk the series holds for one step after its last
   sample and from one step before its next one, else 0 (island down). Returns per grid point
   [t, top, bottom, own value]. */
function interpStack(list,grid,fallbackGap,minBrk){var acc=grid.map(function(){return 0});
  return list.map(function(P){var gap=[];for(var q=1;q<P.length;q++)gap.push(P[q][0]-P[q-1][0]);gap.sort(function(a,b){return a-b});
    var med=gap.length?gap[Math.floor(gap.length/2)]:fallbackGap,brk=Math.max(minBrk||0,2.5*med),j=-1;
    return grid.map(function(x,gi){while(j+1<P.length&&P[j+1][0]<=x)j++;var v=0,a=j>=0?P[j]:null,b=j+1<P.length?P[j+1]:null;
      if(a&&b&&b[0]-a[0]<=brk)v=a[1]+(b[1]-a[1])*(x-a[0])/(b[0]-a[0]||1);
      else if(a&&x-a[0]<=(b?1.5*med:brk))v=a[1];
      else if(b&&b[0]-x<=med)v=b[1];
      var lo=acc[gi];acc[gi]=lo+v;return [x,lo+v,lo,v]})})}

/* ---------- header controls ---------- */
function controls(offline,reload){
  onReload=reload;var box=$("rangeBox");if(!box)return;
  box.innerHTML='<div class="seg">'+RANGES.map(function(r){var on=r[1]===0?RANGE_RUN:!RANGE_RUN&&r[1]===RANGE_H;
    return '<button class="'+(on?"on":"")+'" data-h="'+r[1]+'"'+(r[1]===0?' title="the whole run: first event to now"':"")+'>'+r[0]+'</button>'}).join("")+'</div>'+
    '<button class="btn rl" id="reloadBtn" title="'+(offline?"offline export: data is fixed":"fetch now")+'"'+(offline?" disabled":"")+'>&#x21bb; Reload</button>';
  Array.prototype.forEach.call(box.querySelectorAll?box.querySelectorAll(".seg button"):[],function(b){b.onclick=function(){var h=+b.dataset.h;RANGE_RUN=h===0;if(h)RANGE_H=h;RANGE_SET=true;try{localStorage.setItem("yeto-range",h?h:"run")}catch(e){}render(V,OFF)}});
  var rb=$("reloadBtn");if(rb)rb.onclick=function(){if(onReload)onReload()};
}
/* the window as a phrase after "over" / "in": a live run's last N hours; a finished run's whole span */
function overLabel(){if(!ENDED)return "the last "+rangeLabel();
  return RANGE_RUN?"the whole run ("+(fin(RUN_SPAN)?spanTxt(RUN_SPAN/3600):"")+")":"the last "+rangeLabel()+" of the run"}
function noPrev(){return ENDED&&RANGE_RUN?"nothing before the run":"no previous "+rangeLabel()}
function spanTxt(h){var m=h*60;return m<120?Math.round(m)+" min":m<2880?h.toFixed(1)+" h":(h/24).toFixed(1)+" days"}
/* a run that ended: it says so (overview.run_end: finished or stopped by us), the syncer reached its final cut, or
   every trainer has an explicit end (done, stopped by us, or left with a clean reason). A stale / lost / quiet
   run is not ended: its window keeps running to now and its alerts stay */
var CLEAN_LEAVE=["STOPPED","DONE","FINISHED","COMPLETED","DRAINED"];
function runEnded(V){var O=V.overview||{},d=(V.topology||{}).diloco;
  if(O.run_end||(d&&d.done))return true;
  var tr=(O.islands||[]).filter(function(c){return c.role==="trainer"});
  return tr.length>0&&tr.every(function(c){return c.status==="done"||c.status==="stopped"||
    (c.status==="left"&&c.left&&CLEAN_LEAVE.indexOf(String(c.left.reason||"").toUpperCase())>=0)})}
function rangeLabel(){if(RANGE_RUN){var m=RANGE_H*60;return (m<120?Math.round(m)+" min":m<2880?(RANGE_H).toFixed(1)+" h":(RANGE_H/24).toFixed(1)+" days")+" (whole run)"}
  return RANGE_H>=48?(RANGE_H/24)+" days":RANGE_H===1?"hour":RANGE_H+" hours"}

/* ---------- charts ---------- */
var W=380,H=170,M={l:46,r:10,t:10,b:22};
function axisX(t0,t1){var span=t1-t0,steps=[60,300,600,900,1800,3600,7200,10800,21600,43200,86400,172800,604800],st=steps[steps.length-1];
  for(var i=0;i<steps.length;i++)if(span/steps[i]<=5){st=steps[i];break}
  var off=new Date().getTimezoneOffset()*60,out=[];for(var t=Math.ceil((t0-off)/st)*st+off;t<=t1;t+=st)out.push(t);return {ticks:out,fmt:st>=86400?function(t){var d=new Date(t*1000);return (d.getMonth()+1)+"/"+d.getDate()}:function(t){return clock(t)}}}
/* restart / pool marks -> clusters for the marker strip above the plot: marks closer than 2 min
   (or than 6 px at this range) merge into one, whatever island they belong to; the labels go to
   the hover tooltip, never into the plot */
var MARK_MERGE_S=120,MARK_STRIP=12,REW_WIN=8,SAMPLER_LINES=2;
function markClusters(list,t0,t1,pxS){var out=[],gap=Math.max(MARK_MERGE_S,6*(pxS||0));
  (list||[]).filter(function(m){return fin(m.t)&&m.t>=t0&&m.t<=t1}).sort(function(a,b){return a.t-b.t}).forEach(function(m){var g=out[out.length-1];
    if(g&&m.t-g.last<=gap){g.items.push(m);g.last=m.t;if(g.colors.indexOf(m.color||GREY)<0)g.colors.push(m.color||GREY)}
    else out.push({t:m.t,last:m.t,items:[m],colors:[m.color||GREY]})});return out}
/* a line series splits where two neighbours are further apart than 3x its median spacing, or
   where the incarnation (third element of a point, when given) changes: no line across a gap */
/* a gap breaks the line when it is over 3x the usual spacing AROUND it (the median of up to 4
   gaps on each side): a run whose cadence changes (a sampler gone, slower steps) keeps its line
   after the change instead of turning into dots; a real hole is long next to its neighbours */
function segments(P,all){var T=all.map(function(p){return p[0]}),g=[];for(var i=1;i<T.length;i++)g.push(T[i]-T[i-1]);
  var mid=function(v){v=v.slice().sort(function(a,b){return a-b});return v.length?v[Math.floor(v.length/2)]:0},glob=g.length>=2?mid(g):0;
  var near=function(t){var lo=0,hi=T.length-1;while(lo<hi){var m=(lo+hi)>>1;if(T[m]<t)lo=m+1;else hi=m}if(T[lo]!==t||lo<1)return glob;
    var j=lo-1,w=g.slice(Math.max(0,j-4),j).concat(g.slice(j+1,j+5));return w.length>=2?mid(w):glob};
  var out=[],cur=[];
  P.forEach(function(p,i){var q=P[i-1],med=q&&g.length>=2?near(p[0]):0;
    if(q&&((med>0&&p[0]-q[0]>3*med)||(typeof p[2]==="string"&&typeof q[2]==="string"&&p[2]!==q[2]))){out.push(cur);cur=[]}cur.push(p)});
  if(cur.length)out.push(cur);return out}
/* opts: series [{name,color,pts:[[t,v(,incarnation)]],dash,width,dots,faint}], kind line|stack|bars, refs [{y,label}], yfmt, y0 (force 0 baseline),
   bucket (bars, s), vmarks [{t,color,label}] (marker strip above the plot) */
function chart(id,opts){return chartTs(id,opts)}
function chartTs(id,opts){
  var box=$(id);if(!box)return;var t1=END,t0=END-RANGE_H*3600;
  // opts.extra: per-island series behind an aggregate, drawn when the legend toggle is open
  var xs0=opts.extra||[],xon=xs0.length&&(EXPAND[id]!=null?EXPAND[id]:!!opts.extraOpen);
  var ser=(opts.series||[]).concat(xon?xs0:[]).map(function(s){var all=(s.pts||[]).filter(function(p){return fin(p[0])&&fin(p[1])}).sort(function(a,b){return a[0]-b[0]});
    return {name:s.name,color:s.color,dash:s.dash,width:s.width,dots:s.dots,band:s.band,empty:s.empty,legend:s.legend!==false,all:all,
    pts:all.filter(function(p){return p[0]>=t0&&p[0]<=t1})}});
  var leg='<div class="lg">'+ser.filter(function(s){return s.legend}).map(function(s){return s.pts.length?'<span><i style="background:'+s.color+'"></i>'+esc(s.name)+'</span>':
    '<span class="nd"><i style="background:#4b4b52"></i>'+esc(s.name)+' ('+esc(s.empty||"no data")+')</span>'}).join("")+
    (opts.refs||[]).filter(function(r){return r.legend}).map(function(r){return '<span><i style="background:'+(r.color||GREY)+'"></i>'+esc(r.legend)+'</span>'}).join("")+
    (xs0.length?'<span class="lgx" role="button" tabindex="0">'+(xon?"\u25be hide ":"\u25b8 show ")+esc(opts.extraLabel||"each island")+' ('+xs0.length+')</span>':"")+'</div>';
  var any=ser.some(function(s){return s.pts.length});
  ser=ser.filter(function(s){return s.pts.length}); // series without data are neither stacked nor drawn
  var bindX=function(){var b=box.querySelector&&box.querySelector(".lgx");if(b)b.onclick=b.onkeydown=function(e){if(e&&e.type==="keydown"&&e.key!=="Enter")return;EXPAND[id]=!xon;chartTs(id,opts)}};
  if(!any){box.innerHTML='<div class="empty">no data in '+overLabel()+'</div>'+leg;CH[id]=null;bindX();return}
  var stack=opts.kind==="stack"||opts.kind==="bars",span=t1-t0,bk=opts.bucket||Math.max(60,span/72);
  if(span/bk<4)bk=span<=3600?300:600; // range shorter than a few buckets: 5-10 min buckets
  if(opts.kind==="stack"&&opts.interp){ // interpolate every series on one grid, then stack
    var gs=Math.max(30,span/180),gx=[];for(var gt=Math.ceil(t0/gs)*gs;gt<t1;gt+=gs)gx.push(gt);gx.push(t1);
    var lo0=Math.min.apply(null,ser.map(function(s){return s.pts[0][0]})),first=Math.max(t0,lo0-opts.interp);
    gx=gx.filter(function(x){return x>=first}); // no zero ramp before the first island has data
    var stk=interpStack(ser.map(function(s){return s.all}),gx,opts.interp,2*gs);ser.forEach(function(s,i){s.pts=stk[i]});
  }else if(stack){ // bucket every series on a shared grid
    var keys={};ser.forEach(function(s){var b={};s.pts.forEach(function(p){var x=Math.floor(p[0]/bk)*bk;b[x]=(b[x]||0)+p[1];keys[x]=1});s.b=b});
    var xs=Object.keys(keys).map(Number).sort(function(a,b){return a-b}),acc={};
    ser.forEach(function(s){s.pts=xs.map(function(x){var lo=acc[x]||0,v=s.b[x]||0;acc[x]=lo+v;return [Math.min(t1,Math.max(t0,x+bk/2)),lo+v,lo,v]})});var NB=xs.length;
  }
  var vals=[];ser.forEach(function(s){s.pts.forEach(function(p){vals.push(p[1]);if((stack||s.band)&&p.length>2)vals.push(p[2])})});(opts.refs||[]).forEach(function(r){if(fin(r.y))vals.push(r.y)});
  var off=null;if(opts.clip&&!stack&&vals.length>=10){var sv=vals.slice().sort(function(a,b){return a-b}),q9=rank(sv,0.9),mx=sv[sv.length-1];
    if(q9>0&&mx>4*q9){var cap=1.5*q9;off={cap:cap,max:mx,n:0};vals=vals.filter(function(v){return v<=cap});vals.push(cap)}} // a few huge values (8 h first round) would flatten the rest
  var lo=Math.min.apply(null,vals),hi=Math.max.apply(null,vals),dlo=lo,dhi=hi;if(opts.y0!==false&&(stack||opts.y0||lo>=0&&lo<hi*0.35))lo=Math.min(0,lo);
  if(!stack&&hi>lo&&(lo>0||hi<0)){var pd=(hi-lo)*0.12;lo=(lo>0&&lo-pd<0)?0:lo-pd;hi=(hi<0&&hi+pd>0)?0:hi+pd} // headroom: no point on the frame
  var zoom=!stack&&(lo>0||hi<0)&&dhi>dlo&&(dhi-dlo)<0.2*Math.max(Math.abs(dhi),Math.abs(dlo))?"axis not from 0 \u00b7 spread "+(100*(dhi-dlo)/Math.max(Math.abs(dhi),Math.abs(dlo))).toFixed(1)+"% of max":"";if(fin(opts.ymin))lo=Math.min(lo,opts.ymin);if(hi-lo<=Math.abs(hi)*0.02||hi===lo){var c=(hi+lo)/2,dd=Math.abs(c)*0.1||1;hi=c+dd;lo=(c>=0&&c-dd<0)?0:c-dd}
  var ticks=(opts.yfmt===dur?durTicks:nice)(lo,hi,4);lo=ticks[0];hi=ticks[ticks.length-1];
  if(opts.yint&&ticks.length>1&&ticks[1]-ticks[0]<1){var ia=Math.floor(lo),ib=Math.max(ia+1,Math.ceil(hi)),ist=Math.max(1,Math.ceil((ib-ia)/5));ticks=[];
    for(var iv=ia;iv<ib+ist;iv+=ist)ticks.push(iv);lo=ticks[0];hi=ticks[ticks.length-1]} // counts / policy ages: whole numbers only
  var yf=opts.yfmt||function(v){return k(v)},yl=yf===dur?durAxis(ticks):(opts.yaxis||!opts.yfmt||opts.yfmt===k)?numAxis(ticks):yf;
  var ML=Math.max(30,Math.min(96,Math.ceil(8+5.4*Math.max.apply(null,ticks.map(function(v){return String(yl(v)).length})))));
  var MK=markClusters(opts.vmarks,t0,t1,(t1-t0)/(W-ML-M.r)),MT=M.t+(MK.length?MARK_STRIP:0); // marks: a strip above the plot
  var x=function(t){return ML+(t-t0)*(W-ML-M.r)/(t1-t0)},y=function(v){return MT+(hi-Math.min(v,off?hi:v))/(hi-lo)*(H-MT-M.b)};
  var o=['<svg viewBox="0 0 '+W+' '+H+'" class="ch" role="img" aria-label="'+esc(opts.title||id)+'"><defs><clipPath id="cp-'+id+'"><rect x="'+(ML-4)+'" y="'+(MT-4)+'" width="'+(W-ML-M.r+8)+'" height="'+(H-MT-M.b+8)+'"/></clipPath></defs>'];
  ticks.forEach(function(v){o.push('<line class="gl" x1="'+ML+'" x2="'+(W-M.r)+'" y1="'+y(v).toFixed(1)+'" y2="'+y(v).toFixed(1)+'"/><text x="'+(ML-6)+'" y="'+(y(v)+3.5).toFixed(1)+'" text-anchor="end">'+esc(yl(v))+'</text>')});
  var ax=axisX(t0,t1);ax.ticks.forEach(function(t){o.push('<text x="'+x(t).toFixed(1)+'" y="'+(H-6)+'" text-anchor="middle">'+esc(ax.fmt(t))+'</text>')});
  var PW=W-ML-M.r,bw=Math.max(1,Math.min(PW*bk/span*0.8,PW/Math.max(1,NB||1)*0.8));
  if(MK.length){o.push('<g class="mks"><line class="gl" x1="'+ML+'" x2="'+(W-M.r)+'" y1="'+(MARK_STRIP+1)+'" y2="'+(MARK_STRIP+1)+'"/>'); // restarts / joins / exits: one small triangle per island color
    MK.forEach(function(g){var mx0=x(g.t),cols=g.colors.slice(0,3);o.push('<g class="mk"><title>'+esc(g.items.map(function(m){return clock(m.t,true)+" "+(m.label||"")}).join("\n"))+'</title>'+
      '<rect x="'+(mx0-6).toFixed(1)+'" y="1" width="12" height="'+MARK_STRIP+'" fill="transparent"/>'+cols.map(function(cl,j){var cx0=mx0+(j-(cols.length-1)/2)*5;
        return '<path d="M'+(cx0-3).toFixed(1)+',4L'+(cx0+3).toFixed(1)+',4L'+cx0.toFixed(1)+','+(MARK_STRIP-1)+'Z" fill="'+cl+'"/>'}).join("")+
      (g.items.length>1?'<text x="'+(mx0+3+2.5*cols.length).toFixed(1)+'" y="'+(MARK_STRIP-2)+'" font-size="8" fill="'+GREY+'">'+g.items.length+'</text>':"")+'</g>')});o.push('</g>')}
  o.push('<g clip-path="url(#cp-'+id+')">');
  ser.forEach(function(s){var P=s.pts;if(!P.length)return;var SG=opts.kind?[P]:segments(P,s.all);
    if(s.band){SG.forEach(function(G){if(G.length<2)return;o.push('<path d="'+G.map(function(p,i){return (i?"L":"M")+x(p[0]).toFixed(1)+","+y(p[1]).toFixed(1)}).join("")+
      G.slice().reverse().map(function(p){return "L"+x(p[0]).toFixed(1)+","+y(p[2]).toFixed(1)}).join("")+'Z" fill="'+s.color+'" fill-opacity=".16"/>')});return} // p10-p90 band
    if(opts.kind==="bars"){P.forEach(function(p){if(p[3]>0)o.push('<rect x="'+Math.min(W-M.r-bw,Math.max(ML,x(p[0])-bw/2)).toFixed(1)+'" y="'+y(p[1]).toFixed(1)+'" width="'+bw.toFixed(1)+'" height="'+Math.max(0.5,y(p[2])-y(p[1])).toFixed(1)+'" fill="'+s.color+'"/>')});return}
    var line=SG.map(function(G){return G.map(function(p,i){return (i?"L":"M")+x(p[0]).toFixed(1)+","+y(p[1]).toFixed(1)}).join("")}).join("");
    if(opts.kind==="stack"){var base=P.slice().reverse().map(function(p){return "L"+x(p[0]).toFixed(1)+","+y(p[2]).toFixed(1)}).join("");
      o.push('<path d="'+line+base+'Z" fill="'+s.color+'" fill-opacity=".32"/>')}
    var sparse=P.length<=SPARSE&&opts.kind!=="stack",dots=opts.dots||s.dots;
    if(sparse&&P.length>1&&!dots)o.push('<path d="'+line+'" fill="none" stroke="'+s.color+'" stroke-opacity=".35" stroke-width="1"'+(s.dash?' stroke-dasharray="'+(typeof s.dash==="string"?s.dash:"3 3")+'"':"")+'/>');
    if(sparse||dots)P.forEach(function(p){if(off&&p[1]>hi){off.n++;var cx0=x(p[0]),cy0=M.t+1;
        o.push('<path class="off" d="M'+(cx0-3.5).toFixed(1)+','+(cy0+6)+'L'+(cx0+3.5).toFixed(1)+','+(cy0+6)+'L'+cx0.toFixed(1)+','+cy0+'Z" fill="'+s.color+'"/>');return}
      o.push('<circle cx="'+x(p[0]).toFixed(1)+'" cy="'+y(p[1]).toFixed(1)+'" r="'+(sparse?(s.dash?2.4:3):1.6)+'" fill="'+(s.dash&&sparse?"var(--panel)":s.color)+'" stroke="'+s.color+'" stroke-width="'+(s.dash&&sparse?1.4:0)+'"/>')});
    if(!sparse&&P.length>1&&!dots){o.push('<path d="'+line+'" fill="none" stroke="'+s.color+'" stroke-width="'+(s.width||1.5)+'"'+(s.dash?' stroke-dasharray="'+(typeof s.dash==="string"?s.dash:"3 3")+'"':"")+' stroke-linejoin="round"/>');
      SG.forEach(function(G){if(G.length===1)o.push('<circle cx="'+x(G[0][0]).toFixed(1)+'" cy="'+y(G[0][1]).toFixed(1)+'" r="1.8" fill="'+s.color+'"/>')})}}); // a lone point between gaps
  o.push('</g>');
  if(off&&off.n)o.push('<text x="'+(ML+4)+'" y="'+(MT+16)+'" class="rt offn">\u25b2 '+off.n+' off scale \u00b7 max '+esc(yf(off.max))+'</text>');
  if(zoom)o.push('<text x="'+(W-M.r-2)+'" y="'+(MT+8)+'" text-anchor="end" class="rt">'+esc(zoom)+'</text>');
  (opts.refs||[]).forEach(function(r){if(!fin(r.y)||r.y<lo||r.y>hi)return;o.push('<line x1="'+ML+'" x2="'+(W-M.r)+'" y1="'+y(r.y).toFixed(1)+'" y2="'+y(r.y).toFixed(1)+'" stroke="'+(r.color||GREY)+'" stroke-dasharray="4 3" stroke-width="1"/>'+
    (r.label?'<text x="'+(W-M.r-2)+'" y="'+(y(r.y)-4).toFixed(1)+'" text-anchor="end" class="rt">'+esc(r.label)+'</text>':""))});
  o.push('<line class="cx" x1="0" x2="0" y1="'+MT+'" y2="'+(H-M.b)+'" visibility="hidden"/></svg>');
  box.innerHTML='<div class="cw">'+o.join("")+'<div class="tt"></div></div>'+leg;bindX();
  CH[id]={ser:ser,x:x,t0:t0,t1:t1,yf:yf,stack:stack,ml:ML,mk:MK};
  box.onmousemove=function(e){hover(id,box,e)};box.onmouseleave=function(){var t=box.querySelector&&box.querySelector(".tt");if(t)t.style.display="none";var c=box.querySelector&&box.querySelector(".cx");if(c)c.setAttribute("visibility","hidden")};
}
function hover(id,box,e){var c=CH[id];if(!c||!box.querySelector)return;var svg=box.querySelector("svg"),tt=box.querySelector(".tt"),cx=box.querySelector(".cx");if(!svg||!tt)return;
  var b=svg.getBoundingClientRect(),vx=(e.clientX-b.left)*W/b.width;var t=c.t0+(vx-c.ml)*(c.t1-c.t0)/(W-c.ml-M.r);if(t<c.t0||t>c.t1){tt.style.display="none";return}
  var rows=[],bestT=null;c.ser.forEach(function(s){var best=null,bd=1e18;s.pts.forEach(function(p){var d=Math.abs(p[0]-t);if(d<bd){bd=d;best=p}});
    if(best&&bd<(c.t1-c.t0)/20){bestT=best[0];rows.push('<div><i style="background:'+s.color+'"></i>'+esc(s.name)+'<b>'+esc(s.band?c.yf(best[2])+" \u2013 "+c.yf(best[1]):c.yf(c.stack?best[3]:best[1]))+'</b>'+
      (fin(best.raw)?'<b class="rw">round '+esc(c.yf(best.raw))+'</b>':"")+'</div>')}}); // a smoothed point also names its round's own value
  // marks within 8 px of the cursor: their labels (the plot itself carries none)
  var mg=null,md=8;(c.mk||[]).forEach(function(g){var d=Math.abs(c.x(g.t)-vx);if(d<=md){md=d;mg=g}});
  if(mg){if(bestT==null)bestT=mg.t;rows=mg.items.map(function(m){return '<div class="mkr"><i style="background:'+(m.color||GREY)+'"></i>'+esc(clock(m.t,true)+" "+(m.label||""))+'</div>'}).concat(rows)}
  if(!rows.length){tt.style.display="none";return}
  cx.setAttribute("x1",c.x(bestT));cx.setAttribute("x2",c.x(bestT));cx.setAttribute("visibility","visible");
  tt.innerHTML='<div class="th">'+esc(day(bestT))+'</div>'+rows.join("");tt.style.display="block";
  var bx=box.getBoundingClientRect(),px=e.clientX-bx.left;tt.style.left=(px>bx.width/2?px-tt.offsetWidth-12:px+12)+"px";tt.style.top="8px"}
/* held-out eval per syncer checkpoint: x = syncer outer step (not time, so the range buttons do not apply),
   y = held-out loss; every vs_<name> comparison of a row is a reference drawn at loss - dloss (vs_prev, the
   previous row, is only listed in the tooltip). Error bars: +-1 se of the paired per-sample difference vs the
   primary reference (the first one that is not "base"), drawn around the checkpoint's loss. */
var EV_C={loss:PAL[0]},EV_PAL=[PAL[1],PAL[6],PAL[3],PAL[4],PAL[5]];
function sgn(v,d){return fin(v)?(v>=0?"+":"")+v.toFixed(d==null?4:d):"n/a"}
function evalCmp(c){if(!c||!fin(c.dloss))return "n/a";return "\u0394 "+sgn(c.dloss)+(fin(c.mean)?" (per-sample "+sgn(c.mean)+(fin(c.se)?" \u00b1 "+c.se.toFixed(4):"")+")":"")}
function evalRefs(R){var seen={};R.forEach(function(r){Object.keys(r).forEach(function(k){if(k.indexOf("vs_")===0&&k!=="vs_prev"&&r[k]&&typeof r[k]==="object")seen[k.slice(3)]=1})});
  return Object.keys(seen).sort(function(a,b){return (a==="base")-(b==="base")||(a<b?-1:a>b?1:0)})} // base last: the primary reference is the first other one
function evalColor(names,n){return EV_PAL[Math.max(0,names.indexOf(n))%EV_PAL.length]}
function evalTip(r,names){names=names||evalRefs([r]);return '<div class="th">step '+esc(r.step)+(r.ckpt_written?" \u00b7 "+esc(String(r.ckpt_written).replace("T"," ").slice(0,16))+" UTC":"")+'</div>'+
  '<div><i style="background:'+EV_C.loss+'"></i>loss<b>'+(fin(r.loss)?r.loss.toFixed(4):"n/a")+'</b></div>'+
  names.map(function(n){return '<div><i style="background:'+evalColor(names,n)+'"></i>vs '+esc(n)+'<b>'+esc(evalCmp(r["vs_"+n]))+'</b></div>'}).join("")}
function evalChart(id,E){
  var box=$(id);if(!box)return;var R=((E&&E.rows)||[]).filter(function(r){return r&&fin(r.step)&&fin(r.loss)});
  var names=evalRefs(R),primary=names.filter(function(n){return n!=="base"})[0]||names[0]||null;
  var ref=function(r,n){var c=r["vs_"+n];return c&&fin(c.ref_loss)?c.ref_loss:null};
  var pts={};names.forEach(function(n){pts[n]=R.filter(function(r){return fin(ref(r,n))})});
  var leg=function(notes){notes=notes||{};return '<div class="lg"><span><i style="background:'+EV_C.loss+'"></i>held-out loss</span>'+
    names.map(function(n){return pts[n].length?'<span><i style="background:'+evalColor(names,n)+'"></i>'+esc(n)+(n===primary?" (loss \u2212 \u0394)":"")+esc(notes[n]||"")+'</span>':
      '<span class="nd"><i style="background:#4b4b52"></i>'+esc(n)+' (no data)</span>'}).join("")+'</div>'};
  if(!R.length){box.innerHTML='<div class="empty">no held-out eval yet</div>'+leg();box.onmousemove=null;return}
  var se=function(r){var c=primary?r["vs_"+primary]:null;return c&&fin(c.se)?c.se:0};
  var vals=[];R.forEach(function(r){vals.push(r.loss-se(r),r.loss+se(r))});if(primary)pts[primary].forEach(function(r){vals.push(ref(r,primary))});
  var lo=Math.min.apply(null,vals),hi=Math.max.apply(null,vals),span=Math.max(hi-lo,Math.abs(hi)*0.002,1e-6);
  // a reference far from the checkpoints (a base model ~0.08 above a ~0.005 spread) stays on the axis only when it does not flatten the detail
  var on={};names.forEach(function(n){if(n===primary){on[n]=true;return}var v=pts[n].map(function(r){return ref(r,n)});
    on[n]=v.length&&Math.max.apply(null,v)<=hi+3*span&&Math.min.apply(null,v)>=lo-3*span});
  names.forEach(function(n){if(on[n]&&n!==primary)pts[n].forEach(function(r){vals.push(ref(r,n))})});
  lo=Math.min.apply(null,vals);hi=Math.max.apply(null,vals);span=Math.max(hi-lo,Math.abs(hi)*0.002,1e-6);
  var yt=nice(lo-span*0.08,hi+span*0.08,4);lo=yt[0];hi=yt[yt.length-1];var td=Math.max(3,Math.ceil(-Math.log10(yt[1]-yt[0])-1e-9));
  var s0=R[0].step,s1=R[R.length-1].step;if(s1-s0<2){s0-=1;s1+=1}var xt=nice(s0,s1,5).filter(function(v){return v>=s0&&v<=s1&&v===Math.round(v)});
  var x=function(v){return M.l+8+(v-s0)*(W-M.l-M.r-16)/(s1-s0)},y=function(v){return M.t+(hi-v)/(hi-lo)*(H-M.t-M.b)};
  var o=['<svg viewBox="0 0 '+W+' '+H+'" class="ch" role="img" aria-label="Held-out eval loss">'];
  yt.forEach(function(v){o.push('<line class="gl" x1="'+M.l+'" x2="'+(W-M.r)+'" y1="'+y(v).toFixed(1)+'" y2="'+y(v).toFixed(1)+'"/><text x="'+(M.l-6)+'" y="'+(y(v)+3.5).toFixed(1)+'" text-anchor="end">'+esc(v.toFixed(td))+'</text>')});
  xt.forEach(function(v){o.push('<text x="'+x(v).toFixed(1)+'" y="'+(H-6)+'" text-anchor="middle">'+esc(v)+'</text>')});
  var path=function(L,k){return L.map(function(r,i){return (i?"L":"M")+x(r.step).toFixed(1)+","+y(ref(r,k)).toFixed(1)}).join("")};
  var flat=function(L,k){return L.length===1?"M"+(M.l)+","+y(ref(L[0],k)).toFixed(1)+"L"+(W-M.r)+","+y(ref(L[0],k)).toFixed(1):path(L,k)};
  names.forEach(function(n){if(on[n]&&pts[n].length)o.push('<path class="evref" data-ref="'+esc(n)+'" d="'+flat(pts[n],n)+'" fill="none" stroke="'+evalColor(names,n)+'" stroke-width="1.2" stroke-dasharray="4 3"/>')});
  R.forEach(function(r){var e=se(r);if(e>0)o.push('<line class="eveb" x1="'+x(r.step).toFixed(1)+'" x2="'+x(r.step).toFixed(1)+'" y1="'+y(r.loss-e).toFixed(1)+'" y2="'+y(r.loss+e).toFixed(1)+'" stroke="'+EV_C.loss+'" stroke-opacity=".6" stroke-width="1"/>')});
  if(R.length>1)o.push('<path class="evloss" d="'+R.map(function(r,i){return (i?"L":"M")+x(r.step).toFixed(1)+","+y(r.loss).toFixed(1)}).join("")+'" fill="none" stroke="'+EV_C.loss+'" stroke-width="1.5" stroke-linejoin="round"/>');
  R.forEach(function(r){o.push('<circle cx="'+x(r.step).toFixed(1)+'" cy="'+y(r.loss).toFixed(1)+'" r="2.5" fill="'+EV_C.loss+'"/>')});
  var notes={},row=0;names.forEach(function(n){if(on[n]||!pts[n].length)return;var lv=ref(pts[n][pts[n].length-1],n);notes[n]=" "+lv.toFixed(4)+" (off scale)";
    o.push('<text x="'+(W-M.r-2)+'" y="'+(M.t+9+12*row++)+'" text-anchor="end" class="rt" fill="'+evalColor(names,n)+'">'+esc(n)+' '+esc(lv.toFixed(4))+(lv>hi?" \u2191":" \u2193")+'</text>')});
  o.push('<line class="cx" x1="0" x2="0" y1="'+M.t+'" y2="'+(H-M.b)+'" visibility="hidden"/></svg>');
  box.innerHTML='<div class="cw">'+o.join("")+'<div class="tt"></div></div>'+leg(notes);
  box.onmousemove=function(e){if(!box.querySelector)return;var svg=box.querySelector("svg"),tt=box.querySelector(".tt"),cx=box.querySelector(".cx");if(!svg||!tt)return;
    var b=svg.getBoundingClientRect(),vx=(e.clientX-b.left)*W/b.width,best=null,bd=1e18;
    R.forEach(function(r){var d=Math.abs(x(r.step)-vx);if(d<bd){bd=d;best=r}});if(!best||bd>40){tt.style.display="none";return}
    if(cx){cx.setAttribute("x1",x(best.step));cx.setAttribute("x2",x(best.step));cx.setAttribute("visibility","visible")}
    tt.innerHTML=evalTip(best,names);tt.style.display="block";var bx=box.getBoundingClientRect(),px=e.clientX-bx.left;tt.style.left=(px>bx.width/2?px-tt.offsetWidth-12:px+12)+"px";tt.style.top="8px"};
  box.onmouseleave=function(){var t=box.querySelector&&box.querySelector(".tt");if(t)t.style.display="none";var c=box.querySelector&&box.querySelector(".cx");if(c)c.setAttribute("visibility","hidden")};
}
function card(id,title,sub,wide){return '<div class="cc'+(wide?" wide":"")+'"><div class="ct">'+esc(title)+'</div><div class="cs" title="'+esc(sub||"")+'">'+esc(sub||"")+'</div><div id="'+id+'"></div></div>'}
/* columns of a card grid: up to maxN cards of at least minW px per row; one fewer when that
   would leave a single card alone on the last row */
function balanceGrid(id,minW,maxN){var g=$(id);if(!g||!g.style||!g.children)return;var n=g.children.length,w=g.clientWidth||0,gap=14;
  var c=Math.max(1,Math.min(maxN,Math.floor((w+gap)/(minW+gap))));if(c>2&&n>c&&n%c===1)c--;
  g.style.gridTemplateColumns=c>1?"repeat("+c+",minmax(0,1fr))":""}
/* cards whose chart had no data in the range: drop them from the grid, list them in one line */
function foldEmpty(ids,lineId){var gone=[];ids.forEach(function(x){if(CH[x[0]])return;var b=$(x[0]),cc=b&&b.parentNode;if(cc&&cc.parentNode){cc.parentNode.removeChild(cc);gone.push(x)}});
  var ln=$(lineId);if(!ln)return;ln.hidden=!gone.length;ln.innerHTML=gone.length?'No data in '+esc(overLabel())+', card hidden: '+gone.map(function(x){return '<b>'+esc(x[1])+'</b>'+(x[2]?' ('+esc(x[2])+')':"")}).join(" \u00b7 "):""}

/* histogram on a log axis, dashed p10/p90/p99, solid p50, outer 0.2% tails not drawn */
function rank(sorted,q){if(!sorted.length)return null;return sorted[Math.min(sorted.length-1,Math.max(0,Math.ceil(q*sorted.length)-1))]} // nearest rank: always a sample
/* log-axis bins: tails (outer 0.2%) are only cut with >= 500 samples; every kept sample lands in a bin
   (last bin closed); pct sums to 100% of the kept samples. */
function bins(vals){
  var v=vals.filter(function(x){return fin(x)&&x>0}).sort(function(a,b){return a-b});if(!v.length)return null;
  var kept=v.length>=500?v.filter(function(x){return x>=rank(v,0.002)&&x<=rank(v,0.998)}):v;
  var lo=kept[0],hi=kept[kept.length-1];if(!(hi>lo)){lo=lo*0.95;hi=hi*1.05}
  var L0=Math.log(lo),L1=Math.log(hi),nb=Math.max(6,Math.min(24,Math.round(Math.sqrt(kept.length)*2))),cnt=[],edges=[];
  for(var i=0;i<nb;i++){cnt.push(0)}for(i=0;i<=nb;i++)edges.push(Math.exp(L0+(L1-L0)*i/nb));edges[0]=lo;edges[nb]=hi;
  var idx=function(x){return Math.min(nb-1,Math.max(0,Math.floor((Math.log(x)-L0)/(L1-L0)*nb)))};
  kept.forEach(function(x){cnt[idx(x)]++});
  var q={};[["p10",0.1],["p25",0.25],["p50",0.5],["p75",0.75],["p90",0.9],["p99",0.99]].forEach(function(p){var t=rank(v,p[1]);q[p[0]]={v:t,bin:t>=lo&&t<=hi?idx(t):null}});
  return {sorted:v,kept:kept.length,lo:lo,hi:hi,L0:L0,L1:L1,edges:edges,pct:cnt.map(function(c){return c/kept.length}),q:q}}
function hist(vals,o){
  var B=bins(vals);
  if(!B||B.sorted.length<2)return '<div class="empty">'+(B?"one sample: "+esc(o.fmt(B.sorted[0])):"no data")+'</div>';
  var nb=B.pct.length,w=300,h=118,m={l:30,r:8,t:26,b:18},mx=Math.max.apply(null,B.pct),yt=nice(0,mx,3),ym=yt[yt.length-1];
  var X=function(x){return m.l+(Math.log(x)-B.L0)/(B.L1-B.L0)*(w-m.l-m.r)},Y=function(f){return m.t+(1-f/ym)*(h-m.t-m.b)};
  var s=['<svg viewBox="0 0 '+w+' '+h+'" class="hs">'];
  yt.forEach(function(f){s.push('<line class="gl" x1="'+m.l+'" x2="'+(w-m.r)+'" y1="'+Y(f).toFixed(1)+'" y2="'+Y(f).toFixed(1)+'"/><text x="'+(m.l-4)+'" y="'+(Y(f)+3).toFixed(1)+'" text-anchor="end">'+(f*100).toFixed(0)+'%</text>')});
  B.pct.forEach(function(f,i){if(!f)return;var x0=X(B.edges[i]),x1=X(B.edges[i+1]);
    s.push('<rect class="hb" x="'+(x0+0.5).toFixed(1)+'" y="'+Y(f).toFixed(1)+'" width="'+Math.max(0.8,x1-x0-1).toFixed(1)+'" height="'+(Y(0)-Y(f)).toFixed(1)+'" fill="'+o.color+'"/>')});
  var tk=[],dec=Math.pow(10,Math.floor(Math.log10(B.lo)));for(var d=dec;d<=B.hi*1.01;d*=10)[1,2,5].forEach(function(mm){var t=d*mm;if(t>=B.lo&&t<=B.hi)tk.push(t)});
  if(tk.length<2)tk=[B.lo,B.hi];var lastX=-99;tk.forEach(function(t){var xx=X(t);if(xx-lastX<34)return;lastX=xx;s.push('<text x="'+xx.toFixed(1)+'" y="'+(h-5)+'" text-anchor="middle">'+esc(o.fmt(t))+'</text>')});
  var rows=[[],[]],order=[["p50",0],["p10",1],["p90",1],["p99",1]];
  order.forEach(function(p){var t=B.q[p[0]];if(!t||t.bin==null)return;var xx=X(t.v);
    s.push('<line x1="'+xx.toFixed(1)+'" x2="'+xx.toFixed(1)+'" y1="'+(m.t-2)+'" y2="'+Y(0)+'" stroke="'+(p[1]?"#d1d5db":"#f3f4f6")+'"'+(p[1]?' stroke-dasharray="3 2"':' stroke-width="1.2"')+'/>');
    for(var r=0;r<2;r++){if(rows[r].every(function(px){return Math.abs(px-xx)>=18})){rows[r].push(xx);
      s.push('<text x="'+xx.toFixed(1)+'" y="'+(r?m.t-14:m.t-4)+'" text-anchor="middle" class="rt">'+p[0]+'</text>');break}}});
  s.push('</svg>');return s.join("")}
function qrow(lists){ // [[label, sorted vals, fmt, slowText]]
  var qs=[["p10",0.1],["p25",0.25],["p50",0.5],["p75",0.75],["p90",0.9],["p99",0.99]];
  return '<table class="qt"><tr><th></th>'+lists.map(function(l){return '<th>'+esc(l[0])+'</th>'}).join("")+'</tr>'+
    qs.map(function(q){return '<tr'+(q[0]==="p50"?' class="b"':"")+'><td>'+q[0]+'</td>'+lists.map(function(l){return '<td>'+(l[1].length?esc(l[2](rank(l[1],q[1]))):'<span class="note" title="no samples in the window yet">—</span>')+'</td>'}).join("")+'</tr>'}).join("")+
    '<tr class="m"><td>'+esc(lists[0][4]||"Samples")+'</td>'+lists.map(function(l){return '<td>'+k(l[1].length,0)+'</td>'}).join("")+'</tr>'+
    '<tr class="m"><td>Slow</td>'+lists.map(function(l){return '<td>'+esc(l[3]||"")+'</td>'}).join("")+'</tr></table>'}
function distCard(color,title,spec,parts){ // parts: [{title, vals, fmt, slow:[thr,text]}]
  var lists=parts.map(function(p){var s=p.vals.filter(fin).sort(function(a,b){return a-b});
    var slow=p.slow&&s.length?(100*s.filter(function(x){return p.slow[2]?x<p.slow[0]:x>p.slow[0]}).length/s.length).toFixed(1)+"% "+p.slow[1]:"";return [p.short||p.title,s,p.fmt,slow,p.count]});
  return '<div class="dc"><div class="dh"><span><i style="background:'+color+'"></i>'+esc(title)+'</span><span class="cs">'+esc(spec||"")+'</span></div>'+
    '<div class="dg">'+parts.map(function(p){return '<div><div class="dt">'+esc(p.title)+'</div>'+hist(p.vals,{fmt:p.fmt,color:color})+'</div>'}).join("")+'</div>'+qrow(lists)+'</div>'}

/* ---------- SFT page ---------- */
function col(S,name){return S.cols.indexOf(name)}
function inRange(t,a,b){return fin(t)&&t>a&&t<=b}
function rowsIn(isl,a,b){return isl.rows.filter(function(r){return inRange(r[1],a,b)})}
function islColor(i){return PAL[i%PAL.length]}
var STATUS_TEXT={ok:"running",stopped:"stopped / restarting",done:"done",unknown:"no data"};
function statusText(s){return STATUS_TEXT[s]||s}
function statusDot(s){return '<i class="dot '+({ok:"g",done:"m",warn:"y",stale:"r",stopped:"m",unknown:"m"}[s]||"m")+'" title="'+esc(statusText(s))+'"></i>'}
function spec(I){return [Array.isArray(I.nodes)&&I.nodes.length?I.nodes.join("+"):null,I.layout,fin(I.cards)?I.cards+" cards":null].filter(Boolean).join(" · ")}
function series(S,isls,c,fn){var ci=col(S,c);return isls.map(function(I,i){return {name:I.name,color:islColor(i),pts:I.rows.map(function(r){return [r[1],fn?fn(r):r[ci]]})}})}

function lossCell(row,c,cur,d,fmt){if(!row)return "n/a";var v=fmt?fmt(row[c]):k(row[c],d);return row[0]!=null&&cur!=null&&row[0]!==cur?v+'<div class="ns">step '+row[0]+'</div>':v}
function sft(S){
  var isls=S.islands||[],sy=S.syncer||{},refs=S.refs||{},C=function(n){return col(S,n)};
  var t1=END,t0=END-RANGE_H*3600,tp=t0-RANGE_H*3600;
  var active=isls.filter(function(I){return I.status==="ok"});  // a stopped / restarting island shows its last step, not live numbers
  var fast=isls[0]||null,last=function(I){return I&&I.rows.length?I.rows[I.rows.length-1]:null};
  var lastWith=function(I,c){if(!I)return null;for(var j=I.rows.length-1;j>=0;j--){if(fin(I.rows[j][c]))return I.rows[j]}return null};
  // KPIs
  var tps=sum(active.map(function(I){var r=lastWith(I,C("tps"));return r?r[C("tps")]:null})),mfu=active.map(function(I){var r=lastWith(I,C("mfu"));return r?r[C("mfu")]:null}).filter(fin);
  var fs=fast?rowsIn(fast,t0,t1).map(function(r){return r[C("step_s")]}).filter(fin).sort(function(a,b){return a-b}):[],fp50=pctl(fs,0.5),ref=refs.sft_ref_step_s;
  var ev=(sy.events||[]),hr=ev.filter(function(e){return inRange(e[0],NOW-3600,NOW)}),cnt=function(c){return hr.filter(function(e){return e[1]===c}).length};
  var rss=(sy.rss||[]),rl=rss.length?rss[rss.length-1]:null,cap=refs.sft_syncer_rss_limit_bytes;
  var losses=isls.map(function(I){var r=lastWith(I,C("loss"));return r&&fin(r[C("loss")])?I.name.replace("island ","")+" "+r[C("loss")].toFixed(4):null}).filter(Boolean);
  // run totals from the reducer (S.tokens): one row per island iteration, a step re-run after a restart counts once
  var TK=S.tokens||{},totTok=fin(TK.input)?TK.input:0,totTgt=fin(TK.loss)?TK.loss:0;
  var TIP_IN="Sum over all islands of each step's packed input tokens (bench tokens, padding excluded), including context tokens that carry no loss. One value per island iteration: a step re-run after a restart counts once.";
  var TIP_LOSS="Sum over all islands of each step's loss-bearing tokens (yeto plugin target_tokens). One value per island iteration: a step re-run after a restart counts once.";
  var kp=[["Islands active / total",active.length+" / "+isls.length,isls.map(function(I){return I.name+" "+statusText(I.status)}).join(" · ")],
    ["Input tokens",k(totTok),"packed, no padding · whole run, all islands",TIP_IN],
    ["Loss tokens",k(totTgt),(totTok&&totTgt?(100*totTgt/totTok).toFixed(1)+"% of input · ":"")+"whole run, all islands",TIP_LOSS],
    ["Fleet tokens / s",k(tps),mfu.length?"MFU "+(sum(mfu)/mfu.length).toFixed(1)+"% (6N), last step":"no bench data"],
    [(fast?fast.name:"Island")+" step p50",dur(fp50),fin(fp50)&&fin(ref)?((fp50/ref-1)*100>=0?"+":"")+((fp50/ref-1)*100).toFixed(1)+"% vs reference "+ref.toFixed(0)+" s · "+fs.length+" steps":fs.length+" steps"],
    ["Latest loss",lastWith(fast,C("loss"))?k(lastWith(fast,C("loss"))[C("loss")],4):"n/a",losses.join(" · ")||"no data"],
    ["Syncer outer step",(sy.outer_step==null?"0":sy.outer_step)+" / "+(sy.total_steps==null?"n/a":sy.total_steps),sy.eta_ts?"ETA final cut "+day(sy.eta_ts):(sy.outer_step==null?"waiting for the first outer step":"ETA n/a")],
    ["Rejected / timeout / not carried",cnt("rejected")+" / "+cnt("timeout")+" / "+cnt("not_carried"),"last hour, from "+(sy.log_seen?"syncer.log":"the syncer tape")],
    ["Syncer RSS",rl?bytes(rl[1]):"n/a",rl?(fin(cap)?(100*rl[1]/cap).toFixed(0)+"% of the "+bytes(cap)+" cap · ":"")+"sampled "+ago(rl[0]):OFF?"not sampled in this offline snapshot (no collector RSS samples)":"not sampled (no collector RSS samples)"]];
  $("sKpis").innerHTML=kp.map(function(x){return '<div class="kc"'+(x[3]?' title="'+esc(x[3])+'"':'')+'><div class="kt">'+esc(x[0])+'</div><div class="kv">'+esc(x[1])+'</div><div class="ks">'+esc(x[2])+'</div></div>'}).join("");
  // islands now
  var hbmLim=refs.sft_hbm_limit_gib,hasLim=fin(hbmLim),hbmMax=Math.max(64,hasLim?hbmLim*1.15:0); // no limit configured: no limit marker
  $("sNodesNote").textContent="last step "+ago(Math.max.apply(null,isls.map(function(I){return I.last_ts||0}).concat([0])))+" · syncer outer step "+(sy.outer_step==null?"none yet":sy.outer_step);
  $("sNodes").innerHTML='<table class="nt"><tr><th>Island</th><th>Step</th><th>Loss</th><th>grad_norm</th><th>Step time</th><th>Tokens/s</th><th class="bar">HBM peak'+(hasLim?' vs '+hbmLim+' GiB':'')+'</th><th>Last push (export)</th><th>Last apply</th><th>Data</th></tr>'+
    isls.map(function(I,i){var r=last(I)||[],hr=lastWith(I,C("hbm_gib")),h=hr?hr[C("hbm_gib")]:null,pc=fin(h)?Math.min(100,100*h/hbmMax):0,lim=hasLim?100*hbmLim/hbmMax:null,live=I.status==="ok";
      return '<tr'+(live?'':' class="off"')+'><td><div class="nn">'+statusDot(I.status)+'<i class="sq" style="background:'+islColor(i)+'"></i>'+esc(I.name)+'</div><div class="ns">'+esc(spec(I)||"n/a")+(live?'':' · '+esc(statusText(I.status)))+'</div></td>'+
      '<td>'+(r[0]==null?"n/a":r[0])+' / '+esc(I.steps_total||"n/a")+'<div class="ns">'+ago(I.last_ts)+'</div></td><td>'+lossCell(lastWith(I,C("loss")),C("loss"),r[0],4)+'</td><td>'+lossCell(lastWith(I,C("grad_norm")),C("grad_norm"),r[0],3)+'</td><td>'+lossCell(lastWith(I,C("step_s")),C("step_s"),r[0],null,dur)+'</td><td>'+lossCell(lastWith(I,C("tps")),C("tps"),r[0],null,k)+'</td>'+
      '<td class="bar"><div class="bl">'+(fin(h)?h.toFixed(1)+" GiB"+(live?"":" · last step"+(hr[0]!=null?" "+hr[0]:"")):"n/a")+'</div><div class="bt"><div class="bf'+(!live?" idle":fin(h)&&hasLim&&h>=hbmLim?" over":"")+'" style="width:'+pc.toFixed(1)+'%"></div>'+(lim===null?'':'<div class="bm" style="left:'+lim.toFixed(1)+'%"></div>')+'</div></td>'+
      '<td>'+(I.last_push?dur(I.last_push.export_s)+'<div class="ns">step '+I.last_push.step+' · '+ago(I.last_push.ts)+'</div>':"none yet")+'</td>'+
      '<td>'+(I.last_apply?dur(I.last_apply.apply_s)+'<div class="ns">step '+I.last_apply.step+' · '+ago(I.last_apply.ts)+'</div>':"none yet")+'</td><td>'+esc(I.dataset||"n/a")+'</td></tr>'}).join("")+'</table>';
  // fleet charts
  $("sFleetH").textContent="Fleet over "+overLabel();
  var g=[["fEval","Held-out eval loss","per syncer checkpoint, x = outer step (all checkpoints, not the time range); dashed: each vs_<name> reference (loss \u2212 \u0394); bars \u00b11 se vs the first reference other than base"],
    ["fLoss","Loss","per island, ms-swift logging.jsonl"],["fGn","grad_norm","per island, before clipping"],["fLr","Learning rate","per island schedule"],
    ["fStep","Step time","bench wall_s per step"+(fin(ref)?"; dashed: reference "+dur(ref):"")],["fTps","Tokens / s","stacked by island; each island's bench tokens_per_s interpolated onto one time grid, then summed"],
    ["fHbm","HBM peak","max reserved over ranks per step"+(hasLim?"; dashed: "+hbmLim+" GiB line":"")],["fHost","Host available memory","MemAvailable per node, sampled by the collector"],
    ["fSync","Sync overhead per step","export_s (solid) on steps that exported a fragment, apply_s (dashed) on steps that applied a broadcast"],
    ["fMerge","Syncer merge time","sync_ms (sync/merge_seconds) per round: the merge itself, waiting excluded"],["fRound","Syncer round wait","per round: quorum wait (quorum_ms) and grace wait (ms \u2212 quorum_ms \u2212 sync_ms); merge excluded"],["fGnorm","Outer gnorm by fragment","global delta norm per merge, colour by fragment"],
    ["fRss","Syncer RSS","VmRSS of the syncer"+(fin(cap)?"; dashed: "+bytes(cap)+" cap":"")],["fEv","Syncer problems","rejected / timeout / not carried / stale / disconnect per bucket"]];
  $("sFleet").innerHTML=g.map(function(x){return card(x[0],x[1],x[2])}).join("");
  var total={name:"Fleet total",color:GREY,dash:true};
  evalChart("fEval",S.eval);
  chart("fLoss",{series:series(S,isls,"loss"),yfmt:function(v){return v.toFixed(3)},y0:false});
  chart("fGn",{series:series(S,isls,"grad_norm"),yfmt:function(v){return v.toFixed(2)}});
  chart("fLr",{series:series(S,isls,"lr"),yfmt:function(v){return v.toExponential(1)}});
  chart("fStep",{series:series(S,isls,"step_s"),refs:fin(ref)?[{y:ref,label:"reference "+dur(ref),legend:"Reference"}]:[],yfmt:dur});
  chart("fTps",{kind:"stack",interp:bucketFor(isls),series:series(S,isls,"tps"),yfmt:k});
  chart("fHbm",{series:series(S,isls,"hbm_gib"),refs:hasLim?[{y:hbmLim,label:hbmLim+" GiB",legend:"Limit",color:"#eab308"}]:[],yfmt:function(v){return v.toFixed(0)+" G"},y0:false,ymin:hasLim?0.8*hbmLim:null});
  var hs=[];isls.forEach(function(I,i){Object.keys(I.host||{}).forEach(function(n,j){hs.push({name:n,color:PAL[(i*3+j)%PAL.length],pts:I.host[n].map(function(p){return [p[0],gib(p[1])]})})})});
  chart("fHost",{series:hs,yfmt:function(v){return v.toFixed(0)+" G"}});
  var ss=[];isls.forEach(function(I,i){ss.push({name:I.name+" export",color:islColor(i),pts:I.rows.filter(function(r){return r[C("export_s")]>0}).map(function(r){return [r[1],r[C("export_s")]]})},
    {name:I.name+" apply",color:islColor(i),dash:true,pts:I.rows.filter(function(r){return r[C("applies")]||r[C("apply_s")]>0}).map(function(r){return [r[1],r[C("apply_s")]]})})});
  chart("fSync",{series:ss,yfmt:dur,dots:false});
  var MG=sy.merges||[],mc=function(n){var i=(sy.merge_cols||[]).indexOf(n);return i<0?{merge_s:3,round_s:5,quorum_s:6,grace_s:7,wait_s:8}[n]:i};
  var mser=function(n){var i=mc(n);return MG.map(function(m){return [m[0],m[i]]})};
  chart("fMerge",{dots:true,series:[{name:"merge (sync_ms)",color:PAL[0],pts:mser("merge_s")}],yfmt:dur});
  chart("fRound",{dots:true,clip:true,series:[{name:"quorum wait",color:PAL[1],pts:mser("quorum_s")},{name:"grace wait",color:PAL[5],pts:mser("grace_s")}],yfmt:dur});
  var fr={};(sy.merges||[]).forEach(function(m){var f=m[2]==null?"?":m[2];(fr[f]=fr[f]||[]).push([m[0],m[4]])});
  var fk=Object.keys(fr);chart("fGnorm",{dots:true,series:fk.map(function(f,i){return {name:"fragment "+f,color:PAL[(+f||i)%PAL.length],pts:fr[f],legend:fk.length<=8}}),yfmt:function(v){return v.toFixed(2)}});
  chart("fRss",{series:[{name:"syncer RSS",color:PAL[3],pts:rss.map(function(p){return [p[0],p[1]]})}],refs:fin(cap)?[{y:cap,label:bytes(cap)+" cap",legend:"Cap"}]:[],yfmt:bytes});
  var cats=[["rejected",PAL[8]],["timeout",PAL[1]],["not_carried",PAL[5]],["stale",PAL[3]],["disconnect",GREY]];
  chart("fEv",{kind:"bars",series:cats.map(function(c){return {name:c[0].replace("_"," "),color:c[1],pts:ev.filter(function(e){return e[1]===c[0]}).map(function(e){return [e[0],1]})}}),yfmt:function(v){return +v.toFixed(1)+""}});
  // distribution
  $("sDistH").textContent="Distribution over "+overLabel();
  var slowT=fin(ref)?ref*(refs.sft_slow_factor||1.1):null,dc=isls.map(function(I,i){var R=rowsIn(I,t0,t1);
    return distCard(islColor(i),I.name,spec(I),[{title:"Step time",short:"Step time",vals:R.map(function(r){return r[C("step_s")]}),fmt:dur,slow:fin(slowT)?[slowT,"over "+dur(slowT)]:null,count:"Steps"},
      {title:"Sync blocking per step (sync_s)",short:"sync_s",vals:R.map(function(r){return r[C("sync_s")]}),fmt:dur,slow:fin(ref)?[ref*0.05,"over 5% of a step"]:null}])});
  var mr=MG.filter(function(m){return inRange(m[0],t0,t1)}),mg=mr.map(function(m){return m[mc("merge_s")]}),ms=mg.filter(fin).sort(function(a,b){return a-b}),mp=pctl(ms,0.5);
  var os=mr.map(function(m){return m[mc("wait_s")]}); // ms - sync_ms: quorum + grace wait (outer step_seconds is the quorum wait only)
  dc.push(distCard(GREY,"syncer",(sy.meta&&sy.meta.learners!=null?"learners "+sy.meta.learners+" · ":"")+"H="+((sy.meta||{}).sync_interval||"n/a"),
    [{title:"Merge time per round",short:"merge",vals:mg,fmt:dur,slow:fin(mp)?[2*mp,"over 2x p50"]:null,count:"Rounds"},{title:"Wait per round (round \u2212 merge)",short:"wait",vals:os,fmt:dur,slow:null}]));
  $("sDist").innerHTML=dc.join("");
  // cost / efficiency
  $("sCostH").textContent="Cost / efficiency over "+overLabel();
  function agg(a,b){var o={tok:0,tgt:0,ch:0,sync:0,wall:0,mfuw:0,mfus:0};isls.forEach(function(I){var cards=fin(I.cards)?I.cards:0;rowsIn(I,a,b).forEach(function(r){var w=r[C("step_s")];
      o.tok+=fin(r[C("tokens")])?r[C("tokens")]:0;o.tgt+=fin(r[C("target_tokens")])?r[C("target_tokens")]:0;
      if(fin(w)){o.wall+=w;o.ch+=w*cards/3600;o.sync+=fin(r[C("sync_s")])?r[C("sync_s")]:0;if(fin(r[C("mfu")])){o.mfuw+=w*r[C("mfu")];o.mfus+=w}}})});
    o.tpc=o.ch?o.tok/(o.ch*3600):null;o.mfu=o.mfus?o.mfuw/o.mfus:null;o.share=o.wall?o.sync/o.wall:null;return o}
  var A=agg(t0,t1),B=agg(tp,t0);
  function delta(a,b,fmt){if(!fin(a)||!fin(b)||!b)return '<span class="dl">'+noPrev()+'</span>';var d=a-b,p=100*d/b;
    return '<span class="dl '+(d>=0?"up":"dn")+'">'+(d>=0?"&uarr; +":"&darr; ")+esc(fmt(d))+' · '+(p>=0?"+":"")+p.toFixed(0)+'% vs previous '+rangeLabel()+'</span>'}
  var etas=isls.map(function(I){var r=last(I),R=I.rows.slice(-20).map(function(x){return x[C("step_s")]}).filter(fin).sort(function(a,b){return a-b}),p=pctl(R,0.5);
    return r&&I.steps_total&&fin(p)&&fin(I.last_ts)?I.last_ts+(I.steps_total-r[0])*p:null}).filter(fin),eta=etas.length?Math.max.apply(null,etas):null;
  var ck=[["Input tokens",k(A.tok),"packed, no padding, all islands",delta(A.tok,B.tok,k)],["Loss tokens",k(A.tgt),"loss-bearing (target_tokens), all islands",delta(A.tgt,B.tgt,k)],
    ["Card-hours",k(A.ch,1),"cards × step wall time",delta(A.ch,B.ch,function(v){return k(v,1)})],["Tokens / card·s",k(A.tpc),"tokens ÷ card-seconds",delta(A.tpc,B.tpc,k)],
    ["MFU (6N)",fin(A.mfu)?A.mfu.toFixed(1)+"%":"n/a","step-time weighted",delta(A.mfu,B.mfu,function(v){return v.toFixed(2)+" pt"})],
    ["Sync overhead",fin(A.share)?(100*A.share).toFixed(2)+"%":"n/a","sync_s ÷ step time",delta(A.share,B.share,function(v){return (100*v).toFixed(2)+" pt"})],
    ["Projected finish",eta?day(eta):"n/a",eta?"last island at its p50 step time · in "+dur(eta-NOW):"needs steps_total and step times",""]];
  $("sCostK").innerHTML=ck.map(function(x){return '<div class="kc"><div class="kt">'+esc(x[0])+'</div><div class="kv">'+esc(x[1])+'</div><div class="ks">'+esc(x[2])+'</div>'+x[3]+'</div>'}).join("");
  $("sCost").innerHTML=card("cTok","Tokens per hour","stacked by island")+card("cCh","Card-hours per hour","all card time vs effective training (minus sync blocking)")+
    card("cSync","Sync overhead per hour","sync_s summed per hour, by island")+card("cDs","Tokens by dataset","per hour, stacked; v1 / v2 from the island's cached_dataset");
  chart("cTok",{kind:"bars",bucket:3600,series:series(S,isls,"tokens"),yfmt:k});
  var chs=isls.map(function(I){return I.rows.map(function(r){return [r[1],fin(r[C("step_s")])&&fin(I.cards)?r[C("step_s")]*I.cards/3600:null]})});
  var flat=function(L){var o=[];L.forEach(function(a){a.forEach(function(p){o.push(p)})});return o};
  var HB=RANGE_H<=1?600:3600,hourly=function(pts){var b={};pts.forEach(function(p){if(fin(p[0])&&fin(p[1])){var x=Math.floor(p[0]/HB)*HB;b[x]=(b[x]||0)+p[1]}});return Object.keys(b).map(Number).sort(function(a,b){return a-b}).map(function(x){return [x+HB/2,b[x]]})};
  var eff=isls.map(function(I){return I.rows.map(function(r){var w=r[C("step_s")];return [r[1],fin(w)&&fin(I.cards)?(w-(fin(r[C("sync_s")])?r[C("sync_s")]:0))*I.cards/3600:null]})});
  chart("cCh",{series:[{name:"Card-hours",color:PAL[0],pts:hourly(flat(chs))},{name:"Effective training",color:PAL[2],pts:hourly(flat(eff))}],yfmt:function(v){return k(v,2)}});
  chart("cSync",{kind:"bars",bucket:3600,series:series(S,isls,"sync_s"),yfmt:dur});
  var dsets={};isls.forEach(function(I){var d=I.dataset||"unknown";(dsets[d]=dsets[d]||[]).push(I)});
  chart("cDs",{kind:"stack",bucket:3600,series:Object.keys(dsets).sort().map(function(d,i){return {name:d,color:PAL[[0,1,2,3][i%4]],pts:flat(dsets[d].map(function(I){return I.rows.map(function(r){return [r[1],r[C("tokens")]]})}))}}),yfmt:k});
  // syncer events
  var pc=YetoTbl.cap("syncer-events",ev.filter(function(e){return e[1]!=="joined"}).reverse(),{def:10,unit:"events"}),pe=pc.rows;
  $("sEvents").innerHTML=pe.length?pc.bar+'<table class="nt ev"><tr><th>Time</th><th>Category</th><th>Learner</th><th>Source</th></tr>'+pe.map(function(e){return '<tr><td>'+esc(day(e[0]))+'</td><td>'+esc(e[1])+'</td><td>'+(e[2]==null?"":esc(e[2]))+'</td><td>'+esc(e[3])+'</td></tr>'}).join("")+'</table>':'<div class="empty">no syncer events</div>';
}
function bucketFor(isls){var s=[];isls.forEach(function(I){I.rows.forEach(function(r){if(fin(r[5]))s.push(r[5])})});s.sort(function(a,b){return a-b});var p=pctl(s,0.5);return Math.max(fin(p)?p*1.05:600,RANGE_H*3600/120)}

/* ---------- RL: distribution of round phase durations ---------- */
function rlDist(V){
  var box=$("rlDist");if(!box)return;var O=V.overview||{},ids=(O.islands||[]).map(function(c){return c.id}),t0=END-RANGE_H*3600;
  var cards=ids.map(function(id,i){var c0=(O.islands||[])[i]||{},sp=((V.async||{}).samplers||{})[id];
    if(c0.role==="sampler"&&sp){var sr=(sp.rounds||[]).filter(function(r){return !fin(r.ts)||r.ts>=t0}),G=sr.map(function(r){return r.generate_s}),pg=G.filter(fin).sort(function(a,b){return a-b});
      return distCard(PAL[i%PAL.length],"Island "+id+" (sampler)",[c0.cloud,c0.gpu&&c0.gpus!=null?c0.gpu+" × "+c0.gpus:c0.gpu].filter(Boolean).join(" · "),
        [{title:"Generate time per round",short:"generate",vals:G,fmt:dur,slow:pg.length?[2*pctl(pg,0.5),"over 2x p50"]:null,count:"Rounds"},
         {title:"Groups written per round",short:"groups",vals:sr.map(function(r){return r.groups}),fmt:function(v){return fin(v)?v.toFixed(0):"n/a"}}])}
    var rr=((V.islands||{})[id]||{}).rounds||[],sel=rr.filter(function(r){var p=r.phases&&(r.phases.R||r.phases.T);return !p||!fin(p[1])||p[1]>=t0});
    var R=sel.map(function(r){return (r.dur||{}).R}),T=sel.map(function(r){return (r.dur||{}).T}),c=(O.islands||[])[i]||{};
    var pr=R.filter(fin).sort(function(a,b){return a-b}),pt=T.filter(fin).sort(function(a,b){return a-b});
    return distCard(PAL[i%PAL.length],"Island "+id,[c.cloud,c.gpu&&c.gpus!=null?c.gpu+" × "+c.gpus:c.gpu].filter(Boolean).join(" · "),
      [{title:"Rollout time per round",short:"rollout",vals:R,fmt:dur,slow:pr.length?[2*pctl(pr,0.5),"over 2x p50"]:null,count:"Rounds"},
       {title:"Train time per round",short:"train",vals:T,fmt:dur,slow:pt.length?[2*pctl(pt,0.5),"over 2x p50"]:null}])});
  $("rlDistH").textContent="Distribution over "+overLabel();
  box.innerHTML=cards.join("")||'<div class="empty">no data</div>';
}

/* ---------- RL metrics layout (same components as the SFT page) ---------- */
function rlLive(s){return s==="ok"||s==="waiting"||s==="starting"}
function rlDot(s){return '<i class="dot '+({ok:"g",waiting:"y",starting:"y",done:"m",left:"m",stopped:"m",stale:"r",lost:"r",recovery:"r"}[s]||"m")+'" title="'+esc(s)+'"></i>'}
function lastW(rows,key){for(var j=rows.length-1;j>=0;j--)if(fin(rows[j][key]))return rows[j];return null}
function inR(t,a,b){return fin(t)&&t>a&&t<=b}
function rlCards(V){var O=V.overview||{},cs=(O.islands||[]).map(function(c,i){var o={};for(var key in c)o[key]=c[key];o.i=i;o.color=PAL[i%PAL.length];
  o.label=({sampler:"sampler ",trainer:"trainer ",unrecognized:"unrecognized tape "}[o.role]||"island ")+o.id;var sp=((V.async||{}).samplers||{})[o.id],topo=sp&&sp.provenance&&sp.provenance.topology;
  o.cards=fin(c.gpus)?c.gpus:(topo&&/tp(\d+)/.test(topo)?+/tp(\d+)/.exec(topo)[1]:1);return o});return cs}
/* billed hours: fleet ready -> stop wall clock (cost view) when the island is in fleet.jsonl, else
   (a sampler outside the fleet ledger: NC lease, Modal / Daytona sandbox) the union of its
   incarnations' first -> last event, or the island's first -> last event without incarnations.
   A lower bound: boot before an incarnation's first event is not on the tapes. */
function spanUnion(sp){var a=sp.filter(function(x){return fin(x[0])&&fin(x[1])&&x[1]>=x[0]}).sort(function(p,q){return p[0]-q[0]}),out=[];
  a.forEach(function(x){var l=out[out.length-1];if(l&&x[0]<=l[1])l[1]=Math.max(l[1],x[1]);else out.push([x[0],x[1]])});return out}
function rlBilled(V,c){var ci=(((V.overview||{}).cost||{}).islands||[]).filter(function(r){return String(r.id)===String(c.id)})[0],ex=(V.islands||{})[c.id]||{};
  if(ci&&fin(ci.hours)&&ci.hours>0)return {h:ci.hours,src:"fleet"};var sp=((V.async||{}).samplers||{})[c.id]||{};
  var spans=spanUnion((sp.incarnations||[]).map(function(x){return [x.first_ts,x.last_ts]}));
  if(!spans.length)spans=spanUnion([[ex.first_ts,ex.last_ts]]);
  return spans.length?{h:sum(spans.map(function(x){return x[1]-x[0]}))/3600,src:"events",spans:spans}:{h:null,src:"none"}}
function spanOverlap(spans,a,b){return sum(spans.map(function(x){return Math.max(0,Math.min(b,x[1])-Math.max(a,x[0]))}))}
/* the trainer round in progress: a live trainer whose last round ended (train / sync / publish
   done) and that has not read a new batch since is waiting on the buffer right now. Completed
   rounds alone hide a round stuck for many minutes. */
function curWait(V,c){if(!rlLive(c.status)||c.status==="starting"||c.starting)return null;/* a launch still starting is not waiting on the buffer */var rr=(((V.islands||{})[c.id]||{}).rounds||[]),l=rr[rr.length-1];if(!l)return null;
  var end=fin(l.ts)?l.ts:null;Object.keys(l.phases||{}).forEach(function(p){var e=(l.phases[p]||[])[1];if(fin(e)&&(end==null||e>end))end=e});if(end==null)return null;
  var tr=((((V.async||{}).trainers||{})[c.id]||{}).rounds||[]),rd=tr.length?tr[tr.length-1].read_ts:null;if(fin(rd)&&rd>end)return null;
  return NOW>end?{since:end,s:NOW-end,round:fin(l.round)?l.round+1:null}:null}
// a card row in rows of equal length: as many columns as fit (cards >= 150 px), then the
// cards spread evenly over the rows, so a row of 8 never ends with one card alone on a line
function evenGrid(box,n){if(!box||!n)return;var w=box.clientWidth||box.getBoundingClientRect().width||0;if(!w)return;
  var fit=Math.max(1,Math.floor((w+12)/(150+12))),rows=Math.ceil(n/fit),cols=Math.ceil(n/rows);
  box.style.gridTemplateColumns="repeat("+cols+",minmax(0,1fr))"}
// a hot reload's parts: seen after the release, resolve (fetch / merge), drain (in-flight trials), load, pause
function loadParts(l){var p=[["resolve",l.resolve_s],["drain",l.drain_s],["load",l.load_s],["pause",l.pause_s]].filter(function(x){return fin(x[1])});
  return p.length?'<div class="ns" title="sampler follower timings (rl_sampler_policy_loaded) and the drain before the switch (rl_agentic_drain)'+(fin(l.drain_in_flight)?": "+l.drain_in_flight+" trials in flight":"")+(l.drain_cut?", "+l.drain_cut+" cut":"")+'">'+
    p.map(function(x){return x[0]+" "+dur(x[1])}).join(" · ")+(l.drain_cut?' · <b>'+l.drain_cut+' cut</b>':"")+'</div>':""}
/* a fleet of async islands (async.islands): one buffer per island lane (the run-wide async.backlog is null
   there), and pools that keep zero-variance groups report them as seen, not dropped */
function fleetKpis(A,S,sps,kp,ssum,zvTot,caps){
  var I=A.islands,last=function(x){var p=((x.backlog||{}).points||[]);return p.length?p[p.length-1]:null};
  var ib=I.map(function(x){var p=last(x);return {x:x,b:x.backlog,v:p&&fin(p[3])?p[3]:null}}),have=ib.filter(function(o){return fin(o.v)}),any=ib.some(function(o){return o.b});
  var one=function(o){return fin(o.v)?k(o.v,0):o.b?(fin(o.b.written)?k(o.b.written,0)+" written, ":"")+"not read yet":"no rounds"};
  var bl=["Buffer backlog",have.length?k(sum(have.map(function(o){return o.v})),0):any?"not read yet":"no rounds",
    any?ib.map(function(o){return "island "+o.x.island+": "+one(o)}).join(" · ")+
      " · written "+k(sum(ib.map(function(o){return (o.b||{}).written})),0)+" · read "+k(sum(ib.map(function(o){return (o.b||{}).read})),0)+(caps.length?" · cap "+caps.join("+"):""):"no island has a sampler round yet"];
  // zero-variance: a pool with drop_zero_variance_groups off reports the groups as seen and writes no dropped count
  var keeps=sps.filter(function(c){var x=S[c.id]||{};return fin(x.zero_variance_seen)&&!fin(x.zero_variance_dropped)});
  var kept=!fin(ssum("zero_variance_dropped"))&&!fin(zvTot)&&keeps.length>0;
  var quiet=sps.filter(function(c){return !fin((S[c.id]||{}).zero_variance_seen)&&!fin((S[c.id]||{}).zero_variance_dropped)});
  var kz=function(){return k(sum(keeps.map(function(x){return S[x.id].zero_variance_seen})),0)+" of "+k(sum(keeps.map(function(x){return S[x.id].groups})),0)};
  var NA=/\bn\/a\b/g,dash=function(v){return String(v).replace(NA,"—")},why=" · —: not on the tapes";
  return kp.map(function(c){
    if(c[0]==="Buffer backlog")return bl;
    var rej=c[0].indexOf("Rejected")===0,v=String(c[1]),sub=String(c[2]);
    if(rej){var p=v.split(" / ");if(kept)p[1]="0";v=p.join(" / ");
      if(kept)sub+=" · zero-var: "+kz()+" groups all-equal, kept by the pools (drop_zero_variance_groups off)";
      else if(fin(ssum("zero_variance_dropped"))&&keeps.length)sub+=" · zero-var kept (not dropped) by "+keeps.map(function(x){return x.label}).join(", ")+": "+kz()+" groups";
      if(quiet.length)sub+=" · "+quiet.map(function(x){return x.label}).join(", ")+": no zero-variance count"}
    var miss=/\bn\/a\b/.test(v+" "+sub);
    // a count the tapes lack is a dash, and the card says so
    return [c[0],dash(v),dash(sub)+(miss&&sub.indexOf(why)<0?why:"")]});
}
/* a fleet or a run with a syncer (trainer islands, app.js multiOn): a KPI without data reads as a sentence and an
   island value its tapes do not carry as a dash with the reason; a run with one trainer keeps "n/a" */
function rlMulti(V){var d=V.topology&&V.topology.diloco;return !!((V.async||{}).islands||[]).length||!!(d&&(d.kind==="island"||d.kind==="synced_group"))}
var NA_KPI={"MFU":"not on the tapes","Projected finish":"no total round count","Buffer backlog":"no sampler rounds",
  "Syncer RSS":"not collected","Tokens / card·s":"no trained tokens yet","Latest reward":"not trained yet"};
// a KPI card [title, value, sub, ...]: a value of "n/a" becomes a sentence, an "n/a" inside a value or a sub a dash
function naKpi(x){var t=String(x[0]),v=String(x[1]),y=x.slice();
  y[1]=v==="n/a"?(NA_KPI[t]||(/step p50$/.test(t)?"no steps yet":"not on the tapes yet")):v.replace(/\bn\/a\b/g,"—");
  y[2]=String(x[2]==null?"":x[2]).replace(/\bn\/a\b/g,"—");return y}
function rlm(V){
  var O=V.overview||{},A=V.async||{},S=A.samplers||{},TR=A.trainers||{},X=V.islands||{},cs=rlCards(V);
  var t1=END,t0=END-RANGE_H*3600,tp=t0-RANGE_H*3600;
  var trs=cs.filter(function(c){return c.role!=="sampler"&&c.role!=="unrecognized"}),sps=cs.filter(function(c){return c.role==="sampler"});
  var unk=cs.filter(function(c){return c.role==="unrecognized"}),act=cs.filter(function(c){return c.role!=="unrecognized"});
  var RR=function(c){return ((X[c.id]||{}).rounds||[]).filter(function(r){return fin(r.ts)})},SR=function(c){return ((S[c.id]||{}).rounds||[]).filter(function(r){return fin(r.ts)})};
  var tsum=function(key){var v=0,n=0;Object.keys(TR).forEach(function(i){if(fin(TR[i][key])){v+=TR[i][key];n++}});return n?v:null};
  var ssum=function(key){var v=0,n=0;Object.keys(S).forEach(function(i){if(fin(S[i][key])){v+=S[i][key];n++}});return n?v:null};
  // zero-variance groups the trainer saw: rl_local_round.zero_variance_group_ratio × the round's trained groups
  var zvT=function(c){var tg={};((TR[c.id]||{}).rounds||[]).forEach(function(r){tg[r.rollout_id]=r.trained_groups});
    return RR(c).filter(function(r){return fin(r.zero_var_ratio)&&fin(tg[r.round])}).map(function(r){return [r.ts,Math.round(r.zero_var_ratio*tg[r.round])]})};
  var zvTot=null,zvTr=0;trs.forEach(function(c){zvT(c).forEach(function(p){zvTot=(zvTot||0)+p[1]})});zvTr=tsum("trained_groups")||0;
  // KPIs
  var liveTr=trs.filter(function(c){return rlLive(c.status)}),tokS=sum((liveTr.length?liveTr:trs).map(function(c){var r=lastW(RR(c),"tok_s");return r?r.tok_s:null}));
  var f0=trs[0],fs=f0?RR(f0).filter(function(r){return inR(r.ts,t0,t1)}).map(function(r){return fin(r.step_s)?r.step_s:null}).filter(fin).sort(function(a,b){return a-b}):[];
  var fr=f0?RR(f0).filter(function(r){return inR(r.ts,t0,t1)}).map(function(r){return r.round_s}).filter(fin).sort(function(a,b){return a-b}):[];
  var rew=cs.map(function(c){var sp=S[c.id]||{},v=c.role==="sampler"?sp.reward_mean_last:(lastW(RR(c),"reward")||{}).reward;return fin(v)?(c.role==="sampler"?"sampler ":"trainer ")+c.id+" "+v.toFixed(3)+(c.role==="sampler"&&fin(sp.reward_last_ts)?" ("+ago(sp.reward_last_ts)+")":""):null}).filter(Boolean);
  var trained=tsum("trained_groups")||trs.some(function(c){return RR(c).length>0}); // a trainer round without a reward: not recorded, not "not trained"
  var lastRew=f0?lastW(RR(f0),"reward"):null,pvs=cs.map(function(c){return c.policy_version}).filter(fin),rel=A.releases||[],lr=rel.length?rel[rel.length-1]:null;
  var spv=sps.map(function(c){return c.policy_version}).filter(fin); // the versions the samplers generate with
  var trel=rel.filter(function(r){return r.trainer!=null||fin(r.released_at)}),ltr=trel.length?trel[trel.length-1]:null;
  var toks=sum(trs.map(function(c){return sum(RR(c).map(function(r){return r.tokens}))})),bl=A.backlog,blp=bl&&bl.points&&bl.points.length?bl.points[bl.points.length-1]:null;
  var caps=sps.map(function(c){return (S[c.id]||{}).max_backlog_groups}).filter(fin);
  var kp=[["Islands active / total",act.filter(function(c){return rlLive(c.status)}).length+" / "+act.length,["trainers "+liveTr.length+"/"+trs.length,"samplers "+sps.filter(function(c){return rlLive(c.status)}).length+"/"+sps.length].concat(act.map(function(c){return c.label+" "+c.status}),unk.length?[unk.length+" unrecognized tape(s) not counted"]:[]).join(" · ")],
    ["Groups trained / written",k(tsum("trained_groups")||0,0)+" / "+k(ssum("groups")||0,0),(tsum("trained_groups")?"trainers trained "+k(tsum("trained_groups"),0)+" groups ("+k(toks)+" tokens)":"trainer has trained 0 groups yet")+" · samplers wrote "+k(ssum("groups")||0,0)+" groups, whole run"],
    ["Fleet tokens / s",k(tokS),(liveTr.length?"live trainers, last round":"last round of each trainer")+" · MFU not on the tapes"],
    [(f0?f0.label:"Trainer")+" step p50",!fs.length&&!trained&&(A.islands||[]).length?"not trained yet":dur(pctl(fs,0.5)),"round p50 "+dur(pctl(fr,0.5))+" · "+fs.length+" steps in range"],
    ["Latest reward",lastRew?k(lastRew.reward,3):trained?"not recorded":"not trained yet",(lastRew?(lastRew.reward_src==="trajectories"?"trainer round: mean of its trajectory rewards (no batch summary on the tape)":"trainer batch"):trained?"reward not recorded on this run (trainer batch summary off: --rl-observe-timeline)":"trainer has no trained batch yet")+(rew.length?" · "+rew.join(" · "):lastRew||!trained?(cs.length?" · no reward reported yet (trainers report it per trained batch, samplers per round)":" · no island has reported yet"):"")],
    // releases = versions a trainer released; a sampler starting on the base weights (v0, no
    // trainer, no release time) makes a release row for its load but is not a release
    ["Policy version / releases",(pvs.length?(trs.some(function(c){return fin(c.released_version)&&c.released_version!==c.policy_version})?"local v":"v")+Math.max.apply(null,pvs):"n/a")+" / "+(fin((V.table_totals||{}).releases_trainer)?V.table_totals.releases_trainer:trel.length),ltr?"last release v"+ltr.policy_version+" "+ago(ltr.released_at):"no trainer release yet"+(spv.length?" · samplers on v"+Math.max.apply(null,spv)+(Math.max.apply(null,spv)===0?" (base weights)":""):"")],
    ["Rejected: stale / zero-var / trunc / infra",[tsum("stale_groups"),fin(ssum("zero_variance_dropped"))?ssum("zero_variance_dropped"):zvTot,ssum("truncated"),ssum("dropped_groups")].map(function(v){return fin(v)?k(v,0):"n/a"}).join(" / "),"groups / groups / samples / groups, whole run"+
      (!fin(ssum("zero_variance_dropped"))&&fin(zvTot)?" · zero-var: "+k(zvTot,0)+" of "+k(zvTr,0)+" trained groups ("+(zvTr?(100*zvTot/zvTr).toFixed(0):"?")+"%) had all-equal rewards, trained with zero advantage (trainer)":"")+
      // trials that never formed a group (aborted on infrastructure) are not in the group counts
      (fin(ssum("trials_aborted"))||fin(ssum("trials_ok"))?" · trials aborted "+k(ssum("trials_aborted")||0,0)+" of "+k((ssum("trials_aborted")||0)+(ssum("trials_ok")||0)+(ssum("trials_policy_limit")||0),0)+" ended":"")],
    ["Buffer backlog",blp&&fin(blp[3])?k(blp[3],0):"n/a",bl?"written "+k(bl.written,0)+" · read "+k(bl.read,0)+(caps.length?" · cap "+caps.join("+"):""):"no sampler rounds"]];
  if((V.usage||{}).syncer){var rs=V.rounds_syncer||[];kp.push(["Syncer outer step",rs.length?String(rs[rs.length-1].round):"0","merges on the syncer tape"],["Syncer RSS","n/a","syncer.log not collected for RL runs"])}
  // a run with a syncer: its own cards instead of the sample-buffer ones (topology_diloco.js)
  if(typeof YetoTopoD!=="undefined")kp=YetoTopoD.kpis(V,kp);
  if(A.islands&&A.islands.length)kp=fleetKpis(A,S,sps,kp,ssum,zvTot,caps);
  // a long value (the four rejected counts) wraps in a smaller font instead of being cut off
  if(rlMulti(V))kp=kp.map(naKpi);
  $("rKpis").innerHTML=kp.map(function(x){return '<div class="kc"><div class="kt">'+esc(x[0])+'</div><div class="kv'+(String(x[1]).length>12?" long":"")+'" title="'+esc(x[1])+'">'+esc(x[1])+'</div><div class="ks">'+esc(x[2])+'</div></div>'}).join("");
  evenGrid($("rKpis"),kp.length);
  // islands now
  var pool=A.pool||[],pl=pool.length?pool[pool.length-1]:null;
  var lts=cs.map(function(c){return (X[c.id]||{}).last_ts}).filter(function(t){return fin(t)&&t>0});
  var st=O.store,stx=!st?"":st.failing?" · run store: read failing ("+st.failing+"×)":!st.polled?" · run store: not read yet":" · run store: "+st.sources+" tape(s)"+(fin(st.last_ok_at)?", read "+ago(st.last_ok_at):"");
  $("rNodesNote").textContent=(lts.length?"last event "+ago(Math.max.apply(null,lts)):"no events yet")+(pl?" · NC pool: "+pl.event.replace("nc_elastic_","")+" "+ago(pl.ts):"")+stx;
  var hbm=function(c){var ns=(X[c.id]||{}).node_series||{},pk=null;Object.keys(ns).forEach(function(n){ns[n].forEach(function(p){if(fin(p[1]))pk=pk==null?p[1]:Math.max(pk,p[1])})});return pk};
  // a run with a syncer: local step / outer version, last PUSH / BCAST; no HBM column when nothing sampled it
  var dl=typeof YetoTopoD!=="undefined"?YetoTopoD.table(V):null,noHbm=!!dl&&!cs.some(function(c){return fin(hbm(c))});
  var head='<tr><th>Island</th><th>'+(dl?'<span title="local step: this island\'s own steps; outer sweep: every fragment merged at least that many times">Local step / outer sweep</span>':"Round / policy")+'</th><th>Reward</th><th>train-rollout KL</th><th>logprob |diff|</th><th>grad_norm</th><th>Step / round time</th><th>Tokens/s</th>'+(noHbm?"":'<th class="bar">HBM peak</th>')+'<th>'+(dl?"Last PUSH / BCAST applied":"Last release / load")+'</th><th>Shard · incarnation</th></tr>';
  var flNa=rlMulti(V);
  var rowOf=function(c){var ex=X[c.id]||{},s=S[c.id],rr=RR(c),r=rr[rr.length-1]||{},liv=rlLive(c.status),h=hbm(c),lim=ex.hbm_total_gib,mx=Math.max(64,fin(lim)?lim*1.15:0),pc=fin(h)?Math.min(100,100*h/mx):0;
      var row='<tr'+(liv?'':' class="off"')+'><td><div class="nn">'+rlDot(c.status)+'<i class="sq" style="background:'+c.color+'"></i>'+esc(c.label)+'</div><div class="ns">'+esc([c.name,c.gpu?c.gpu+" × "+c.cards:null,c.backend_label].filter(Boolean).join(" · ")||"n/a")+' · '+esc(c.status+(c.waiting_reason?": "+c.waiting_reason:"")+(c.left?" ("+(c.left.reason||"")+")":"")+(c.starting&&c.startup_stage?": "+c.startup_stage.label+(fin(c.startup_stage.age_s)?" "+dur(c.startup_stage.age_s):"")+(fin(c.startup_s)?" · start "+dur(c.startup_s):""):"")+((c.launches||[]).length>1?" · launch "+(c.launches.length)+" ("+(c.launches.length-1)+" superseded)":"")+
        // a trainer island the syncer has merged no push of for N merge cycles (its record's mark, else its fleet event)
        ((c.island_now&&"straggler" in c.island_now?c.island_now.straggler:c.straggler)&&liv?" · straggler":""))+'</div></td>';
      var hb='<td class="bar"><div class="bl">'+(fin(h)?h.toFixed(1)+" GiB"+(fin(lim)?" of "+lim+" GiB":""):"not sampled")+'</div><div class="bt"><div class="bf'+(liv?"":" idle")+'" style="width:'+pc.toFixed(1)+'%"></div>'+(fin(lim)?'<div class="bm" style="left:'+(100*lim/mx).toFixed(1)+'%"></div>':"")+'</div>'+(fin(h)?"":'<div class="ns" title="'+esc(ex.gpu_mem_note||"")+'">'+(/unavailable/.test(ex.gpu_mem_note||"")?"telemetry library missing":"enable --rl-resource-sample-interval")+'</div>')+'</td>';
      if(noHbm)hb="";
      if(c.role==="sampler"&&s){var sr=s.rounds||[],l=s.last_round||sr[sr.length-1]||{},ld=(A.releases||[]).map(function(x){return (x.loads||{})[c.id]?[x.policy_version,x.loads[c.id]]:null}).filter(Boolean).pop();
        return row+'<td title="'+esc(k(s.n_rounds-(s.empty_rounds||0),0)+" rounds wrote groups"+(s.empty_rounds?", "+k(s.empty_rounds,0)+" wrote none (backlog full, all trials lost or drained)":"")+"; generating with policy "+(s.policy_version==null?"n/a":"v"+s.policy_version))+'">'+
          k(s.n_rounds-(s.empty_rounds||0),0)+(s.empty_rounds?' <span class="ns">(+'+k(s.empty_rounds,0)+' empty)</span>':'')+' / '+(s.policy_version==null?"n/a":"v"+s.policy_version)+'<div class="ns">'+ago(l.ts)+' · age p50 '+k(s.lag_p50,1)+'</div></td><td>'+(dl&&!fin(s.reward_mean_last)?'—<div class="ns">not on its tape</div>':k(s.reward_mean_last,3))+(fin(s.reward_last_ts)?'<div class="ns">round '+esc(s.reward_last_rollout_id)+' · '+ago(s.reward_last_ts)+'</div>':"")+'</td>'+'<td title="samplers do not train">—</td>'.repeat(3)+'<td>'+dur(l.generate_s)+'<div class="ns">generate, p50 '+dur(s.generate_p50_s)+'</div></td><td>'+(fin(l.response_tokens)&&l.generate_s?k(l.response_tokens/l.generate_s):dl?'—<div class="ns">no tokens on its rounds</div>':"n/a")+'</td>'+hb+
          '<td>'+(ld?(ld[1].at_start?"at start":dur(ld[1].load_wall_s))+'<div class="ns">load v'+ld[0]+' · '+ago(ld[1].loaded_at)+' · '+(ld[1].at_start?"loaded when the sampler started":dur(ld[1].latency_s)+' after release')+'</div>'+(ld[1].at_start?"":loadParts(ld[1])):"none yet")+'</td><td>'+esc(s.shard||c.id)+'<div class="ns">'+esc(s.incarnation||"n/a")+(pl?" · pool "+esc(pl.event.replace("nc_elastic_",""))+(pl.ticket?" "+esc(pl.ticket):""):"")+'</div></td></tr>'}
      var mr=(A.releases||[]).filter(function(x){return String(x.trainer)===String(c.id)}).pop(),kl=lastW(rr,"kl_tr")||lastW(rr,"kl")||{},cw=curWait(V,c),un=c.unreported||{};
      var gn=lastW(rr,"grad_norm")||{},dstep=dl&&dl.step(c);
      return row+'<td>'+(dstep!=null?esc(dstep):(r.round==null?"n/a":r.round)+' / '+(c.policy_version==null?'<span title="no policy version reported by the current launch yet">—</span>':(fin(c.released_version)&&c.released_version!==c.policy_version?"local v":"v")+c.policy_version))+'<div class="ns">'+(dstep==null&&fin(c.released_version)&&c.released_version!==c.policy_version?"released v"+c.released_version+" · ":"")+ago(r.ts)+'</div></td><td>'+k((lastW(rr,"reward")||{}).reward,3)+'</td><td>'+(fin(kl.kl_tr)?k(kl.kl_tr,4)+(fin(kl.kl)?'<div class="ns">PPO '+k(kl.kl,4)+'</div>':un.ppo_kl?'<div class="ns">PPO '+esc(un.ppo_kl)+'</div>':""):fin(kl.kl)?(un.train_rollout_kl?'<span title="'+esc(un.train_rollout_kl)+'">—</span>':'n/a')+'<div class="ns">PPO '+k(kl.kl,4)+'</div>':un.train_rollout_kl?'—<div class="ns">'+esc(un.train_rollout_kl)+'</div>':"n/a")+'</td><td>'+k((lastW(rr,"logprob_diff")||{}).logprob_diff,4)+'</td><td>'+k(gn.grad_norm,3)+(gn.grad_norm===0&&gn.reward_std===0?'<div class="ns">all rewards equal</div>':"")+'</td>'+
        '<td>'+dur(r.step_s)+(fin(r.round_s)||!flNa?'<div class="ns">round '+dur(r.round_s)+'</div>':'')+(cw&&cw.s>=60?'<div class="ns wt">current round'+(cw.round!=null?" "+cw.round:"")+' waiting for data '+dur(cw.s)+'</div>':"")+'</td><td>'+k(r.tok_s)+'</td>'+hb+'<td>'+(dl&&dl.trainers[c.id]?dl.push(c,NOW):mr?dur(mr.upload_s)+'<div class="ns">release v'+mr.policy_version+' · '+ago(mr.released_at)+'</div>':"none"+(c.status==="left"?"":" by this trainer"))+'</td><td>trainer '+esc(c.id)+'<div class="ns">'+esc((c.left||{}).reason||"")+'</div></td></tr>'};
  var unkRow=function(c){var ex=X[c.id]||{};return '<tr class="off"><td><div class="nn">'+rlDot("unknown")+'<i class="sq" style="background:'+c.color+'"></i>'+esc(c.label)+'</div><div class="ns">no trainer or sampler events in this tape; not counted</div></td><td colspan="'+(noHbm?9:10)+'">'+
    esc(c.source_lost?"tape source lost: "+String(c.source_lost.error||"").slice(0,120):"")+'</td></tr>'};
  var liveC=act.filter(function(c){return rlLive(c.status)||c.status==="stale"}),endC=act.filter(function(c){return liveC.indexOf(c)<0});
  // a fleet: a value its tapes do not carry is a dash that says so, never a bare n/a (tags and attributes kept)
  // a fleet / syncer run: an island that ended before its first round is one line saying how it ended
  var never=function(c){return flNa&&!rlLive(c.status)&&c.status!=="stale"&&!RR(c).length&&!SR(c).length};
  var neverRow=function(c){var why=c.left&&(c.left.reason||c.left.detail),st=c.status==="left"?"left":c.status;
    return '<tr class="off"><td><div class="nn">'+rlDot(c.status)+'<i class="sq" style="background:'+c.color+'"></i>'+esc(c.label)+'</div><div class="ns">'+esc([c.name,c.gpu?c.gpu+" × "+c.cards:null].filter(Boolean).join(" · "))+'</div></td>'+
      '<td colspan="'+(noHbm?9:10)+'" class="ns" title="'+esc((c.left&&c.left.detail)||"")+'">'+esc(st+(why?": "+why:"")+" · ended before its first round")+'</td></tr>'};
  var rowNa=function(c){if(never(c))return neverRow(c);var h=rowOf(c);return flNa?h.replace(/(<[^>]*>)|\bn\/a\b/g,function(m,tag){return tag||'<span class="note" title="not on this island\'s tapes">—</span>'}):h};
  $("rNodes").innerHTML=(liveC.length?'<table class="nt">'+head+liveC.map(rowNa).join("")+'</table>':'<div class="empty">no active island</div>')+
    (endC.length?'<details class="ended"'+(liveC.length?"":" open")+'><summary class="ns">'+endC.length+' island(s) that ended (left / lost / finished / stopped): '+esc(endC.map(function(c){return c.label+" "+c.status}).join(", "))+'</summary><table class="nt">'+head+endC.map(rowNa).join("")+'</table></details>':"")+
    (unk.length?'<details class="ended"><summary class="ns">'+unk.length+' unrecognized tape(s)</summary><table class="nt">'+unk.map(unkRow).join("")+'</table></details>':"");
  // fleet charts
  $("rFleetH").textContent="Fleet over "+overLabel();
  // a trainer point carries the launch that trained it (the newest launch started before it): a relaunch breaks the line
  var lOf=function(c,t){var ls=(c.launches||[]).filter(function(x){return fin(x.first_ts)&&x.first_ts<=t}).sort(function(a,b){return a.first_ts-b.first_ts});return ls.length?ls[ls.length-1].launch:"-"};
  var ser=function(L,key,f){return L.map(function(c){return {name:c.label,color:c.color,pts:RR(c).map(function(r){return [r.ts,f?f(r):r[key],lOf(c,r.ts)]})}})};
  var sser=function(key,f){return sps.map(function(c){return {name:c.label,color:c.color,dash:true,pts:SR(c).map(function(r){return [r.ts,f?f(r):r[key],r.incarnation]})}})};
  // sampler reward: group-weighted rolling mean over the last n groups (whole rounds, newest first);
  // each point keeps its own round's mean (.raw) for the tooltip
  var roll=function(rows,n,inc){var w=[],out=[];rows.forEach(function(r){if(!fin(r.reward_mean)||!(r.groups>0))return;w.push(r);var g=0,v=0;
    for(var j=w.length-1;j>=0&&g<n;j--){g+=w[j].groups;v+=w[j].groups*w[j].reward_mean}var p=inc?[r.ts,v/g,r.incarnation]:[r.ts,v/g];p.raw=r.reward_mean;out.push(p)});return out};
  // many samplers (more than SAMPLER_LINES): one aggregate per chart, each sampler's own line behind
  // the legend toggle; up to SAMPLER_LINES they are drawn one by one
  var MANY=sps.length>SAMPLER_LINES,SC="#fb923c",XL="each sampler";
  var pooled=function(get){var b={},bk=Math.max(120,RANGE_H*3600/40);sps.forEach(function(c){get(c).forEach(function(p){if(fin(p[0])&&fin(p[1])){var x=Math.floor(p[0]/bk)*bk;(b[x]=b[x]||[]).push(p[1])}})});
    var xs=Object.keys(b).map(Number).sort(function(a,c){return a-c}),mid=[],band=[];
    xs.forEach(function(x){var v=b[x].sort(function(a,c){return a-c});mid.push([x+bk/2,pctl(v,0.5)]);band.push([x+bk/2,pctl(v,0.9),pctl(v,0.1)])});return {mid:mid,band:band}};
  // sampler series of a chart: [aggregate series], [per-sampler series]
  var spSeries=function(name,get,o){o=o||{};var each=sps.map(function(c){return {name:c.label+(o.suffix||""),color:c.color,dash:o.dash,pts:get(c)}});
    if(!MANY)return {series:each,extra:[]};var q=pooled(get);
    return {series:[{name:"samplers p50 ("+sps.length+")"+(o.suffix||""),color:SC,dash:o.dash,pts:q.mid}].concat(o.noBand?[]:[{name:"samplers p10–p90",color:SC,band:true,legend:false,pts:q.band}]),extra:each}};
  var evs=cs.filter(function(c){return ((X[c.id]||{}).evals||[]).length});
  var g=[];if(evs.length)g.push(["rfEval","Held-out eval","rl_eval: "+esc(((X[evs[0].id].evals[0])||[])[2]||"")]);
  g=g.concat([["rfRew","Reward","trainer: batch mean per trained step (solid). Samplers: group-weighted mean of the last "+(sps.length>1?REW_WIN+"×"+sps.length+" groups over all samplers (dashed; each sampler's own line from the legend toggle"+(MANY?")":", open by default)"):REW_WIN+" groups (dashed)")+"; hover shows the round's own mean. Triangles above every fleet chart: restarts and pool events (merged within 2 min), hover for details. Lines break at gaps over 3× the usual spacing and at a new incarnation"],["rfKl","train-rollout KL","trainer vs sampler logprobs (train_metrics.train_rollout_kl) per trainer round"],["rfPkl","PPO KL","trainer policy vs the old logprobs of the batch (train_metrics.ppo_kl); near 0 with one optimizer step per batch"],
    ["rfLp","logprob |diff|","train vs rollout logprob abs diff per round"],["rfTis","TIS / IS clipped","share of tokens whose IS weight was clipped: mismatch.tis_clipfrac, with score centering train_metrics.sc_is_clipped"],["rfGn","grad_norm","per trainer round"],
    ["rfLr","Learning rate","rl_round_trained.applied_lrs[0] (applied_lr): the lr the trainer reports per round; the engine's own log line (e.g. Miles lr-pg_0) is not on the tapes"],["rfStep","Trainer step time","train step seconds (solid); round wall time (dashed)"],["rfTps","Tokens / s","stacked by trainer; tok_per_s interpolated onto one time grid, then summed"],
    ["rfHbm","HBM","mean used GiB over each island's GPUs (NVML / host probe)"+(MANY?"; samplers: p50 and p10–p90 band":"")],["rfHost","Host memory","per node, from the Modal host probe (YETO_MODAL_HOSTMEM_SAMPLE_S)"],
    ["rfRound","Sampler round time","generate seconds per sampler round"+(MANY?"; p50 and p10–p90 band over samplers":"")],["rfAge","Group policy age","trained groups' mean policy age per trainer round (rl_round_trained.policy_age_groups); sampler version lag per round"],
    ["rfResp","Response length","mean tokens per sample. Trainer (solid): model-generated tokens only (loss mask); trainer full (dotted, newer tapes) and sampler (dashed): whole response incl. tool / env tokens"],["rfBl","Buffer backlog","groups written − trainers' read position (solid); sampler-reported (dots)"],
    ["rfRel","Release → load latency","per sampler: load done − trainer release, one point per policy version"],["rfDrop","Rejected groups / samples","counts per time bucket: groups (stale, zero-variance, infra drops) and samples (truncated); hover for each count. Zero-variance from the trainer (rl_local_round ratio × trained groups) is trained with zero advantage, from samplers it is dropped"+(A.islands&&A.islands.length?" when the pool drops it (drop_zero_variance_groups), else kept":"")]]);
  $("rFleet").innerHTML=g.map(function(x){return card(x[0],x[1],x[2])}).join("");
  var axs=function(o){o.yaxis=true;return o},mag=function(o){o.yaxis=true;o.y0=true;return o}; // magnitudes / fractions: the axis starts at 0
  var f3=function(v){return k(v,3)},f4=function(v){return v.toPrecision(3)};
  if(evs.length)chart("rfEval",{series:evs.map(function(c){return {name:c.label,color:c.color,pts:X[c.id].evals.map(function(e){return [e[0],e[1]]})}}),yfmt:f3,y0:false,dots:true});
  var shN=function(x){var m=/-(\d+)$/.exec(String(x==null?"":x));return m?"s"+m[1]:String(x==null?"?":x)};
  var vm=[];sps.forEach(function(c){((S[c.id]||{}).restarts||[]).forEach(function(x){vm.push({t:x.ts,color:c.color,label:"↻ "+c.label+" restarted (incarnation "+x.incarnation+")"})})});
  trs.forEach(function(c){(c.launches||[]).slice(1).forEach(function(x){vm.push({t:x.first_ts,color:c.color,label:"↻ "+c.label+" relaunched ("+x.launch+")"})})});
  var PMK={member_ready:["+","ready"],sampler_restart:["−","restart"],released:["−","released"],released_externally:["−","released"],shard_fenced:["−","fenced"],
    evicted:["−","evicted"],expired:["−","expired"],node_lost:["−","node lost"],sampler_crash_loop:["−","crash loop"],scaled_down:["−","scaled down"],manual_release_detected:["−","released by hand"],manual_pause:["−","paused by hand"],manual_resume:["+","resumed by hand"],scale_draining:["−","draining (scale down)"],start_failed:["−","start failed"],start_retry:["↻","start retry"],sampler_vram_high:["!","VRAM high"],stale_handoff_set_aside:["+","stale stop request set aside"],node_reachable_again:["+","reachable"]};
  (A.pool||[]).forEach(function(p){var e=String(p.event||"").replace("nc_elastic_",""),m=PMK[e];if(!m)return;var why=p.detail||p.why||p.reason||p.error||(p.max_frac!=null?"max "+(100*p.max_frac).toFixed(1)+"% of a GPU":"");
    var sc=sps.filter(function(c){return String(c.id)===String(p.shard)})[0];vm.push({t:p.ts,color:sc?sc.color:null,
      label:m[0]+shN(p.shard)+" NC pool: sampler shard "+(p.shard==null?"?":p.shard)+" "+m[1]+(p.ticket?" · ticket "+p.ticket:"")+(fin(p.restarts)?" · restarts "+p.restarts:"")+(why?" · "+String(why).slice(0,160):"")})});
  var fc=function(id,o){o.vmarks=vm;return chartTs(id,o)}; // every fleet time chart carries the same marker strip
  var each=sps.map(function(c){return {name:c.label+" (mean of last "+REW_WIN+" groups)",color:c.color,dash:true,pts:roll(SR(c),REW_WIN,true)}});
  var allR=[];sps.forEach(function(c){allR=allR.concat(SR(c))});allR.sort(function(a,b){return a.ts-b.ts});
  // two or more samplers: the pooled mean is the line to read; each sampler's own line is open by default up to SAMPLER_LINES
  var POOL=sps.length>1;
  fc("rfRew",{series:ser(trs,"reward").concat(POOL?[{name:"all "+sps.length+" samplers (mean of last "+REW_WIN*sps.length+" groups)",color:SC,dash:true,width:2,pts:roll(allR,REW_WIN*sps.length,false)}]:each),
    extra:POOL?each:[],extraOpen:!MANY,extraLabel:XL,yfmt:f3,y0:false});
  fc("rfKl",mag({series:ser(trs,"kl_tr"),yfmt:f4}));fc("rfPkl",mag({series:ser(trs,"kl"),yfmt:f4}));
  fc("rfLp",mag({series:ser(trs,"logprob_diff"),yfmt:f4}));fc("rfTis",mag({series:ser(trs,"tis_clip",function(r){return fin(r.tis_clip)?r.tis_clip:r.is_clip}),yfmt:f4}));fc("rfGn",mag({series:ser(trs,"grad_norm"),yfmt:f3}));
  fc("rfLr",axs({series:ser(trs,"lr"),yfmt:function(v){return v.toExponential(2)}}));
  fc("rfStep",{series:ser(trs,"step_s").concat(trs.map(function(c){return {name:c.label+" round",color:c.color,dash:true,legend:false,pts:RR(c).map(function(r){return [r.ts,r.round_s,lOf(c,r.ts)]})}})),yfmt:dur});
  fc("rfTps",{kind:"stack",interp:600,series:ser(trs,"tok_s"),yfmt:k});
  var hbmPts=function(c){var ns=(X[c.id]||{}).node_series||{},n=Object.keys(ns)[0];return n?ns[n].map(function(p){return [p[0],p[1]]}):[]};
  var hs=spSeries("HBM",hbmPts);
  fc("rfHbm",{series:cs.filter(function(c){return c.role!=="sampler"}).map(function(c){return {name:c.label,color:c.color,pts:hbmPts(c)}}).concat(hs.series),extra:hs.extra,extraLabel:XL,yfmt:function(v){return v.toFixed(0)+" G"}});
  fc("rfHost",{series:[]});
  var rs=spSeries("round",function(c){return SR(c).map(function(r){return [r.ts,r.generate_s,r.incarnation]})});
  fc("rfRound",{series:rs.series,extra:rs.extra,extraLabel:XL,yfmt:dur});
  // trained groups' mean policy age: the round's age histogram, else the trainer's own mean (rl_round_trained.policy_age_mean; newer tapes carry no histogram)
  var ageMean=function(r){var h=r.ages||{},n=0,s=0;Object.keys(h).forEach(function(x){if(fin(h[x])&&!isNaN(+x)){n+=h[x];s+=h[x]*(+x)}});return n?s/n:fin(r.policy_age_mean)?r.policy_age_mean:null};
  var ls=spSeries("lag",function(c){return SR(c).map(function(r){return [r.ts,r.version_lag,r.incarnation]})},{suffix:" version lag",noBand:true});
  fc("rfAge",{series:trs.map(function(c){return {name:c.label+" trained groups",color:c.color,pts:((TR[c.id]||{}).rounds||[]).map(function(r){return [r.ts,ageMean(r)]})}}).concat(ls.series),extra:ls.extra,extraLabel:XL,
    yfmt:function(v){return v.toFixed(1)},yaxis:true,yint:true,y0:true,dots:true});
  var rsp=spSeries("resp",function(c){return SR(c).map(function(r){return [r.ts,r.resp_len_mean,r.incarnation]})},{suffix:" full",dash:true});
  fc("rfResp",mag({series:trs.map(function(c){return {name:c.label+" generated (loss mask)",color:c.color,pts:RR(c).map(function(r){return [r.ts,r.resp_mean,lOf(c,r.ts)]})}})
    .concat(trs.map(function(c){return {name:c.label+" full",color:c.color,dash:"1 3",pts:RR(c).map(function(r){return [r.ts,r.resp_full_mean,lOf(c,r.ts)]})}}).filter(function(x){return x.pts.some(function(p){return fin(p[1])})}))
    .concat(rsp.series),extra:rsp.extra,extraLabel:XL,yfmt:k}));
  var bls=sps.map(function(c){return {name:c.label+" reported",color:c.color,pts:((S[c.id]||{}).backlog||[]).map(function(p){return [p[0],p[1]]}),empty:"never full in range"}}); // it reports only when its backlog is full
  // a fleet: one backlog line per island lane (async.backlog is null there), in its trainer's colour, without
  // the sum of every lane's cap (no lane can reach it)
  var FLT=!!(A.islands&&A.islands.length);
  var blMain=FLT?A.islands.map(function(x,i){var tc=cs.filter(function(c){return x.trainer!=null&&String(c.id)===String(x.trainer)})[0];
      return {name:"island "+x.island+" backlog",color:tc?tc.color:PAL[(3+i)%PAL.length],pts:((x.backlog||{}).points||[]).map(function(p){return [p[0],p[3]]})}}):
    [{name:"backlog",color:PAL[3],pts:bl?bl.points.map(function(p){return [p[0],p[3]]}):[]}];
  fc("rfBl",{series:blMain.concat(MANY?[]:bls),extra:MANY?bls:[],extraLabel:"each sampler's own count",yfmt:k,
    refs:caps.length&&!FLT?[{y:sum(caps),label:"caps "+sum(caps),legend:"sum of backlog caps",color:"#eab308"}]:[]});
  var rel=sps.map(function(c){return {name:c.label,color:c.color,pts:(A.releases||[]).map(function(x){var l=(x.loads||{})[c.id];return l&&!l.at_start?[l.loaded_at,l.latency_s]:null}).filter(Boolean)}});
  fc("rfRel",{dots:true,series:MANY?[{name:"all samplers",color:SC,pts:[].concat.apply([],rel.map(function(x){return x.pts}))}]:rel,extra:MANY?rel:[],extraLabel:XL,yfmt:dur});
  var tb=function(key){var o=[];trs.forEach(function(c){((TR[c.id]||{}).rounds||[]).forEach(function(r){if(fin(r.ts)&&r[key]>0)o.push([r.ts,r[key]])})});return o};
  var sb=function(key){var o=[];sps.forEach(function(c){SR(c).forEach(function(r){if(r[key]>0)o.push([r.ts,r[key]])})});return o};
  // the source reports the count (zeros included) in range: an empty series is "0 so far", not "no data"
  var tz=function(key){return trs.some(function(c){return ((TR[c.id]||{}).rounds||[]).some(function(r){return fin(r[key])&&inR(r.ts,t0,t1)})})?"0 so far":null};
  var sz=function(key){return sps.some(function(c){return SR(c).some(function(r){return fin(r[key])&&inR(r.ts,t0,t1)})})?"0 so far":null};
  var zvs=sb("zero_variance_dropped").length?sb("zero_variance_dropped"):sb("zero_variance_seen"),zvt=[].concat.apply([],trs.map(zvT)).filter(function(p){return p[1]>0});
  fc("rfDrop",{kind:"bars",series:[{name:"stale groups",color:PAL[3],pts:tb("stale_groups"),empty:tz("stale_groups")},
    {name:zvs.length?(FLT&&!sb("zero_variance_dropped").length?"zero-variance groups (samplers, kept: zero advantage)":"zero-variance groups (samplers, dropped)"):"zero-variance groups (trainer, zero advantage)",color:PAL[5],pts:zvs.length?zvs:zvt},
    {name:"truncated samples",color:PAL[1],pts:sb("truncated"),empty:sz("truncated")},{name:"infra-dropped groups",color:PAL[8],pts:sb("dropped_groups"),empty:sz("dropped_groups")}],yfmt:function(v){return +v.toFixed(1)+""},yaxis:true,yint:true});
  foldEmpty(g,"rFleetNone");balanceGrid("rFleet",340,4);
  rlCost(V,cs,trs,sps,RR,SR,t0,t1,tp);
}
/* a fleet of async islands: the syncer's total steps (syncer_config.total_steps on its tape) give the finish;
   the tapes carry no FLOP or parameter counts, so there is no MFU card rather than an "n/a" one */
/* a fleet of async islands: the syncer's total steps (syncer_config.total_steps on its tape) give the finish,
   paced from the step the syncer (re)started at (syncer_config.global_step: a resumed syncer starts past 0) */
function fleetCost(V,ck){
  var d=(V.topology||{}).diloco||null,cf=d&&d.config||{},T=cf.T,steps=d?(d.shards||[]).map(function(s){return s.step}).filter(fin):[],st=steps.length?Math.min.apply(null,steps):null;
  var s0=fin(cf.step0)?cf.step0:cf.resumed?null:0,lm=d&&fin(d.last_merge_ts_any)?d.last_merge_ts_any:null,t0=fin(cf.ts)?cf.ts:null;
  var fin2=function(){
    if(!fin(T))return ["Projected finish","—","the syncer tape has no total step count yet (syncer_config)",""];
    if(d.done||(fin(st)&&st>=T))return ["Finished",lm?day(lm):"step "+T,"syncer reached step "+(fin(st)?st:T)+" of "+T+" (final cut)"+(lm&&t0?" · "+dur(lm-t0)+" after its "+(cf.resumed?"resume":"config"):""),""];
    if(!fin(s0))return ["Projected finish","—","syncer step "+(fin(st)?st:0)+" of "+T+"; resumed at a step not on its tape, so no pace",""];
    // a sharded syncer resumes each shard at its own step: shard 0's start against the slowest shard is no pace
    if(cf.resumed&&cf.shards>1)return ["Projected finish","—","syncer step "+(fin(st)?st:0)+" of "+T+"; a sharded syncer resumed per shard, so no pace",""];
    var n=fin(st)?st-s0:0;
    if(n<=0||!lm||!t0||lm<=t0)return ["Projected finish","—","syncer step "+(fin(st)?st:0)+" of "+T+"; needs a merge"+(cf.resumed?" after its resume":"")+" to pace the rest",""];
    var pace=(lm-t0)/n,eta=lm+(T-st)*pace;
    return ["Projected finish",day(eta),"syncer step "+st+" of "+T+" at its mean merge pace "+dur(pace)+(cf.resumed?" since its resume at step "+s0:"")+
      (eta>=NOW?" · in "+dur(eta-NOW):" · overdue by "+dur(NOW-eta)+", last merge "+ago(lm)),""]};
  var NA=/\bn\/a\b/g;
  return ck.map(function(x){
    if(x[0]==="Projected finish")return fin2();
    if(x[0]==="MFU")return ["MFU","—","not on the tapes (no FLOP counts)",""];
    var sub=String(x[2]).replace(/ · completed rounds only n\/a/,""),miss=/\bn\/a\b/.test(String(x[1])+" "+sub);
    // any other count the tapes lack is a dash, and the card says so
    return [x[0],String(x[1]).replace(NA,"—"),sub.replace(NA,"—")+(miss?" · —: not on the tapes":""),x[3]]})}
function rlCost(V,cs,trs,sps,RR,SR,t0,t1,tp){
  $("rCostH").textContent="Cost / efficiency over "+overLabel();
  var X=V.islands||{};
  var waitS=function(c,a,b){return sum(((X[c.id]||{}).rounds||[]).filter(function(r){return r.phases&&r.phases.R&&inR(r.phases.R[1],a,b)}).map(function(r){return r.dur.R}))};
  var CW={};trs.forEach(function(c){var w=curWait(V,c);if(w)CW[c.id]=w});
  function agg(a,b){var o={groups:0,tok:0,billT:0,billS:0,effT:0,effS:0,wait:0,wall:0,cur:0};
    cs.forEach(function(c){var bh=rlBilled(V,c),span=bh.src==="events"?null:[((X[c.id]||{}).ready_ts)||(X[c.id]||{}).first_ts,(X[c.id]||{}).last_ts];
      var ov=span?(fin(span[0])&&fin(span[1])?Math.max(0,Math.min(b,span[1])-Math.max(a,span[0])):0):spanOverlap(bh.spans||[],a,b),cards=c.cards;
      if(c.role==="sampler"){o.billS+=ov*cards;SR(c).forEach(function(r){if(inR(r.ts,a,b)){o.effS+=(r.generate_s||0)*cards;o.groups+=r.groups||0}})}
      else{o.billT+=ov*cards;RR(c).forEach(function(r){if(inR(r.ts,a,b)){o.effT+=(r.step_s||0)*cards;o.tok+=r.tokens||0;o.wall+=r.round_s||0}});o.wait+=waitS(c,a,b);
        var w=CW[c.id];if(w)o.cur+=spanOverlap([[w.since,NOW]],a,b)}}); // the round in progress, still waiting
    o.bill=o.billT+o.billS;o.eff=o.effT+o.effS;o.util=o.bill?o.eff/o.bill:null;o.tpc=o.billT?o.tok/o.billT:null;o.wsDone=o.wall?o.wait/o.wall:null;o.ws=o.wall+o.cur?(o.wait+o.cur)/(o.wall+o.cur):null;return o}
  var A=agg(t0,t1),B=agg(tp,t0);
  function delta(a,b,fmt){if(!fin(a)||!fin(b)||!b)return '<span class="dl">'+noPrev()+'</span>';var d=a-b,p=100*d/b;
    return '<span class="dl '+(d>=0?"up":"dn")+'">'+(d>=0?"&uarr; +":"&darr; ")+esc(fmt(d))+' · '+(p>=0?"+":"")+p.toFixed(0)+'% vs previous '+rangeLabel()+'</span>'}
  var pc=function(v){return fin(v)?(100*v).toFixed(1)+"%":"n/a"},ch=function(s){return k(s/3600,1)};
  var nc=sps.some(function(c){return rlBilled(V,c).src==="events"});
  var ck=[["Groups / tokens",k(A.groups,0)+" / "+k(A.tok),"groups written by samplers / tokens trained",delta(A.groups,B.groups,function(v){return k(v,0)})],
    ["Card-hours",ch(A.billT)+" + "+ch(A.billS),"trainers + samplers"+(nc?"; samplers outside the fleet ledger: union of their incarnations' first → last event (a lower bound: boot before the first event is not on the tapes)":""),delta(A.bill,B.bill,ch)],
    ["Tokens / card·s",k(A.tpc),"trained tokens ÷ trainer card-seconds",delta(A.tpc,B.tpc,k)],
    ["MFU","n/a","the RL tapes carry no FLOP counts",""],
    ["GPU effective utilization",pc(A.util),"(train step + generate time) × cards ÷ billed card time · trainers "+pc(A.billT?A.effT/A.billT:null)+" · samplers "+pc(A.billS?A.effS/A.billS:null),delta(A.util,B.util,function(v){return (100*v).toFixed(1)+" pt"})],
    ["Trainer waiting for data",pc(A.ws),"rollout phase (waiting on the buffer) ÷ round wall time, including the round in progress"+
      (Object.keys(CW).length?" · now: "+Object.keys(CW).map(function(i){return "trainer "+i+" waiting "+dur(CW[i].s)}).join(", ")+" · completed rounds only "+pc(A.wsDone):""),delta(A.ws,B.ws,function(v){return (100*v).toFixed(1)+" pt"})],
    ["Projected finish","n/a","the tapes carry no total round count",""]];
  if(((V.async||{}).islands||[]).length)ck=fleetCost(V,ck);
  if(rlMulti(V))ck=ck.map(naKpi);
  $("rCostK").innerHTML=ck.map(function(x){return '<div class="kc"><div class="kt">'+esc(x[0])+'</div><div class="kv">'+esc(x[1])+'</div><div class="ks">'+esc(x[2])+'</div>'+x[3]+'</div>'}).join("");
  $("rCost").innerHTML=card("rcG","Groups per hour","stacked by sampler")+card("rcCh","Card-hours per hour","billed card time vs effective (train step / generate)")+
    card("rcW","Trainer waiting per hour","rollout-phase seconds (waiting on the buffer) per trainer; yellow: the round still waiting now")+card("rcPv","Groups by policy version","per hour, stacked by the version the sampler generated with (no task labels on the tapes)");
  chart("rcG",{kind:"bars",bucket:3600,series:sps.map(function(c){return {name:c.label,color:c.color,pts:SR(c).map(function(r){return [r.ts,r.groups]})}}),yfmt:k});
  var HB=RANGE_H<=1?600:3600,hrs=[];for(var h=Math.floor(t0/HB)*HB;h<t1;h+=HB)hrs.push(h);
  var bill=hrs.map(function(h){var a=agg(h,h+HB);return [[h+HB/2,a.bill/3600],[h+HB/2,a.eff/3600]]});
  chart("rcCh",{series:[{name:"Card-hours",color:PAL[0],pts:bill.map(function(x){return x[0]}).filter(function(p){return p[1]>0})},{name:"Effective",color:PAL[2],pts:bill.map(function(x){return x[1]}).filter(function(p,i){return bill[i][0][1]>0})}],yfmt:function(v){return k(v,2)}});
  // the round in progress: its waiting so far, split per hour bucket, in a lighter bar on top
  var cwPts=function(c){var w=CW[c.id],o=[];if(!w)return o;for(var h=Math.floor(w.since/3600)*3600;h<NOW;h+=3600){var v=spanOverlap([[w.since,NOW]],h,h+3600);if(v>0)o.push([Math.min(NOW,Math.max(w.since,h+1)),v])}return o};
  chart("rcW",{kind:"bars",bucket:3600,series:trs.map(function(c){return {name:c.label,color:c.color,pts:((X[c.id]||{}).rounds||[]).filter(function(r){return r.phases&&r.phases.R}).map(function(r){return [r.phases.R[1],r.dur.R]})}})
    .concat(trs.filter(function(c){return CW[c.id]}).map(function(c){return {name:c.label+" current round (in progress)",color:"#eab308",pts:cwPts(c)}})),yfmt:dur});
  var pv={};sps.forEach(function(c){SR(c).forEach(function(r){var v=r.policy_version==null?"?":"v"+r.policy_version;(pv[v]=pv[v]||[]).push([r.ts,r.groups])})});
  chart("rcPv",{kind:"bars",bucket:3600,series:Object.keys(pv).sort(function(a,b){return (parseInt(a.slice(1),10)||-1)-(parseInt(b.slice(1),10)||-1)}).map(function(v,i){return {name:v,color:PAL[i%PAL.length],pts:pv[v]}}),yfmt:k});
  foldEmpty([["rcG","Groups per hour","stacked by sampler"],["rcCh","Card-hours per hour",""],["rcW","Trainer waiting per hour",""],["rcPv","Groups by policy version",""]],"rCostNone");balanceGrid("rCost",340,4);
}
function render(view,offline){
  V=view;OFF=offline;if(!V)return;var S=V.sft;
  NOW=offline?((S&&S.now)||(V.overview||{}).now||Date.now()/1000):Date.now()/1000;
  // until the viewer picks a range, show the whole run in the smallest window that holds it:
  // an export once (its data is fixed), the live page on every render (1h, then 6h ... as the
  // run grows), so a run younger than an hour is not squeezed into the right edge of 24h
  if(RANGE_RUN===null)RANGE_RUN=!S;
  // a run that ended: every window ends at its last event (plus 2% of the run, at least a minute), not at now
  var oe=V.overview||{};ENDED=runEnded(V);RUN_SPAN=fin(oe.data_ts)&&fin(oe.first_ts)?Math.max(0,oe.data_ts-oe.first_ts):null;
  END=ENDED&&fin(oe.data_ts)?Math.min(NOW,oe.data_ts+Math.max(60,0.02*(RUN_SPAN||0))):NOW;
  if(RANGE_RUN){var o0=oe,end=ENDED?END:offline?(o0.data_ts||NOW):NOW,sp=fin(end)&&fin(o0.first_ts)?(end-o0.first_ts)/3600:null;
    if(fin(sp))RANGE_H=Math.max(0.25,sp*1.03+2/60)} // a little room before the first event
  var sroot=$("sftRoot"),rroot=$("rlRoot");
  var RL=!S&&!!$("rlmRoot");
  if(document.body&&document.body.classList)document.body.classList.toggle("sftmode",!!S||RL);
  if(sroot)sroot.hidden=!S;if(rroot)rroot.hidden=!!S;
  // one topology panel: under the SFT page's islands for a DiLoCo SFT run, else in the RL page
  var tb=$("topoBox"),slot=$(S?"sTopoSlot":"rTopoSlot");if(tb&&slot&&tb.parentNode!==slot)slot.appendChild(tb);
  controls(offline,onReload);
  var run=(V.overview||{}).run;
  if(S){$("title").innerHTML='yeto metrics <span class="tg">'+esc(run||"unnamed run")+'</span><span class="tg">SFT · decoupled DiLoCo</span>';
    document.title="yeto metrics · "+(run||"SFT");
    $("sub").textContent="updated "+clock(offline?NOW:Date.now()/1000,true)+" · times in your local zone"+(offline?" · offline export, data as of "+day(NOW):" · live");
    sft(S)}
  else{rlDist(V);if(RL){var lay=String((($("title")||{}).innerHTML)||"").replace(/<[^>]*>/g,"").split(" · ")[0],lk=(V.layout||{}).kind,tg=(lk==="island"||lk==="synced_group"?"RL · decoupled DiLoCo ("+lk.replace("_"," ")+")":V.async?"RL · async GRPO":"RL")+(lay?" · "+lay:"");$("title").innerHTML='yeto metrics <span class="tg">'+esc(run||"unnamed run")+'</span><span class="tg">'+tg+'</span>';
    // a run that ended says how: finished on its own (rl_run_end) or stopped by us (reducer.run_end_status)
    var ov=V.overview||{},ended=ov.run_end&&(ov.global_status||{}).text?" · run "+ov.global_status.text:"";
    document.title="yeto metrics · "+(run||"RL");$("sub").textContent="updated "+clock(offline?NOW:Date.now()/1000,true)+" · times in your local zone"+(offline?" · offline export, data as of "+day(NOW):" · live")+ended;rlm(V)}}
}
return {markClusters:markClusters,segments:segments,spanUnion:spanUnion,spanOverlap:spanOverlap,bins:bins,evalTip:evalTip,durAxis:durAxis,durTicks:durTicks,numAxis:numAxis,nice:nice,interpStack:interpStack,render:render,setReload:function(fn){onReload=fn},rangeH:function(){return RANGE_H},
  window:function(){return {t0:END-RANGE_H*3600,t1:END,ended:ENDED,now:NOW}}};
})();
