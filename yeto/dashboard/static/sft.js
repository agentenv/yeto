// yeto dashboard: metrics layout (dark ops style) for SFT runs, plus the shared header
// controls (time range, reload) and the RL distribution panel. Loaded before app.js in the
// same <script>; app.js calls YetoExt.render(V, OFFLINE) after its own render.
// Hand-written SVG; no external requests. Every number comes from the reducer view.
var YetoExt=(function(){
"use strict";
var PAL=["#3b82f6","#f97316","#14b8a6","#8b5cf6","#ec4899","#eab308","#22c55e","#06b6d4","#ef4444","#a3e635"];
var GREY="#9ca3af",RANGES=[["1h",1],["6h",6],["24h",24],["7d",168],["30d",720]];
var RANGE_H=24,CH={},V=null,OFF=false,NOW=0,onReload=null;
try{var sr=localStorage.getItem("yeto-range");if(sr&&+sr>0)RANGE_H=+sr}catch(e){}
try{var hm=/range=(\d+)/.exec(location.hash||"");if(hm)RANGE_H=+hm[1]}catch(e){}
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

/* ---------- header controls ---------- */
function controls(offline,reload){
  onReload=reload;var box=$("rangeBox");if(!box)return;
  box.innerHTML='<div class="seg">'+RANGES.map(function(r){return '<button class="'+(r[1]===RANGE_H?"on":"")+'" data-h="'+r[1]+'">'+r[0]+'</button>'}).join("")+'</div>'+
    '<button class="btn rl" id="reloadBtn" title="'+(offline?"offline export: data is fixed":"fetch now")+'"'+(offline?" disabled":"")+'>&#x21bb; Reload</button>';
  Array.prototype.forEach.call(box.querySelectorAll?box.querySelectorAll(".seg button"):[],function(b){b.onclick=function(){RANGE_H=+b.dataset.h;try{localStorage.setItem("yeto-range",RANGE_H)}catch(e){}render(V,OFF)}});
  var rb=$("reloadBtn");if(rb)rb.onclick=function(){if(onReload)onReload()};
}
function rangeLabel(){return RANGE_H>=48?(RANGE_H/24)+" days":RANGE_H===1?"hour":RANGE_H+" hours"}

/* ---------- charts ---------- */
var W=380,H=170,M={l:46,r:10,t:10,b:22};
function axisX(t0,t1){var span=t1-t0,steps=[60,300,600,900,1800,3600,7200,10800,21600,43200,86400,172800,604800],st=steps[steps.length-1];
  for(var i=0;i<steps.length;i++)if(span/steps[i]<=5){st=steps[i];break}
  var off=new Date().getTimezoneOffset()*60,out=[];for(var t=Math.ceil((t0-off)/st)*st+off;t<=t1;t+=st)out.push(t);return {ticks:out,fmt:st>=86400?function(t){var d=new Date(t*1000);return (d.getMonth()+1)+"/"+d.getDate()}:function(t){return clock(t)}}}
/* opts: series [{name,color,pts:[[t,v]],dash,width}], kind line|stack|bars, refs [{y,label}], yfmt, y0 (force 0 baseline), bucket (bars, s) */
function chart(id,opts){
  var box=$(id);if(!box)return;var t1=NOW,t0=NOW-RANGE_H*3600;
  var ser=(opts.series||[]).map(function(s){return {name:s.name,color:s.color,dash:s.dash,width:s.width,legend:s.legend!==false,
    pts:(s.pts||[]).filter(function(p){return fin(p[0])&&fin(p[1])&&p[0]>=t0&&p[0]<=t1}).sort(function(a,b){return a[0]-b[0]})}});
  var leg='<div class="lg">'+ser.filter(function(s){return s.legend}).map(function(s){return s.pts.length?'<span><i style="background:'+s.color+'"></i>'+esc(s.name)+'</span>':
    '<span class="nd"><i style="background:#4b4b52"></i>'+esc(s.name)+' (no data)</span>'}).join("")+
    (opts.refs||[]).filter(function(r){return r.legend}).map(function(r){return '<span><i style="background:'+(r.color||GREY)+'"></i>'+esc(r.legend)+'</span>'}).join("")+'</div>';
  var any=ser.some(function(s){return s.pts.length});
  ser=ser.filter(function(s){return s.pts.length}); // series without data are neither stacked nor drawn
  if(!any){box.innerHTML='<div class="empty">no data in the last '+rangeLabel()+'</div>'+leg;CH[id]=null;return}
  var stack=opts.kind==="stack"||opts.kind==="bars",span=t1-t0,bk=opts.bucket||Math.max(60,span/72);
  if(span/bk<4)bk=span<=3600?300:600; // range shorter than a few buckets: 5-10 min buckets
  if(opts.kind==="stack"&&opts.hold){ // step-hold resample on a shared grid: each series takes its last sample
    var gs=Math.max(30,span/180),gx=[],lo0=Math.min.apply(null,ser.map(function(s){return s.pts[0][0]}));
    for(var gt=Math.ceil(Math.max(t0,lo0)/gs)*gs;gt<=t1;gt+=gs)gx.push(gt);
    var acc2=gx.map(function(){return 0});
    ser.forEach(function(s){var P=s.pts,j=-1,gap=[];for(var q=1;q<P.length;q++)gap.push(P[q][0]-P[q-1][0]);gap.sort(function(a,b){return a-b});
      var age=Math.max(2*gs,3*(gap.length?gap[Math.floor(gap.length/2)]:opts.hold));
      s.pts=gx.map(function(x,gi){while(j+1<P.length&&P[j+1][0]<=x)j++;var v=(j>=0&&x-P[j][0]<=age)?P[j][1]:0,lo=acc2[gi];acc2[gi]=lo+v;return [x,lo+v,lo,v]})});
  }else if(stack){ // bucket every series on a shared grid
    var keys={};ser.forEach(function(s){var b={};s.pts.forEach(function(p){var x=Math.floor(p[0]/bk)*bk;b[x]=(b[x]||0)+p[1];keys[x]=1});s.b=b});
    var xs=Object.keys(keys).map(Number).sort(function(a,b){return a-b}),acc={};
    ser.forEach(function(s){s.pts=xs.map(function(x){var lo=acc[x]||0,v=s.b[x]||0;acc[x]=lo+v;return [Math.min(t1,Math.max(t0,x+bk/2)),lo+v,lo,v]})});var NB=xs.length;
  }
  var vals=[];ser.forEach(function(s){s.pts.forEach(function(p){vals.push(p[1]);if(p.length>2)vals.push(p[2])})});(opts.refs||[]).forEach(function(r){if(fin(r.y))vals.push(r.y)});
  var lo=Math.min.apply(null,vals),hi=Math.max.apply(null,vals);if(opts.y0!==false&&(stack||opts.y0||lo>=0&&lo<hi*0.35))lo=Math.min(0,lo);if(fin(opts.ymin))lo=Math.min(lo,opts.ymin);if(hi-lo<=Math.abs(hi)*0.02||hi===lo){var c=(hi+lo)/2,dd=Math.abs(c)*0.1||1;hi=c+dd;lo=(c>=0&&c-dd<0)?0:c-dd}
  var ticks=nice(lo,hi,4);lo=ticks[0];hi=ticks[ticks.length-1];
  var x=function(t){return M.l+(t-t0)*(W-M.l-M.r)/(t1-t0)},y=function(v){return M.t+(hi-v)/(hi-lo)*(H-M.t-M.b)};
  var yf=opts.yfmt||function(v){return k(v)},o=['<svg viewBox="0 0 '+W+' '+H+'" class="ch" role="img" aria-label="'+esc(opts.title||id)+'"><defs><clipPath id="cp-'+id+'"><rect x="'+M.l+'" y="'+M.t+'" width="'+(W-M.l-M.r)+'" height="'+(H-M.t-M.b)+'"/></clipPath></defs>'];
  ticks.forEach(function(v){o.push('<line class="gl" x1="'+M.l+'" x2="'+(W-M.r)+'" y1="'+y(v).toFixed(1)+'" y2="'+y(v).toFixed(1)+'"/><text x="'+(M.l-6)+'" y="'+(y(v)+3.5).toFixed(1)+'" text-anchor="end">'+esc(yf(v))+'</text>')});
  var ax=axisX(t0,t1);ax.ticks.forEach(function(t){o.push('<text x="'+x(t).toFixed(1)+'" y="'+(H-6)+'" text-anchor="middle">'+esc(ax.fmt(t))+'</text>')});
  var PW=W-M.l-M.r,bw=Math.max(1,Math.min(PW*bk/span*0.8,PW/Math.max(1,NB||1)*0.8));
  o.push('<g clip-path="url(#cp-'+id+')">');
  ser.forEach(function(s){var P=s.pts;if(!P.length)return;
    if(opts.kind==="bars"){P.forEach(function(p){if(p[3]>0)o.push('<rect x="'+Math.min(W-M.r-bw,Math.max(M.l,x(p[0])-bw/2)).toFixed(1)+'" y="'+y(p[1]).toFixed(1)+'" width="'+bw.toFixed(1)+'" height="'+Math.max(0.5,y(p[2])-y(p[1])).toFixed(1)+'" fill="'+s.color+'"/>')});return}
    var line=P.map(function(p,i){return (i?"L":"M")+x(p[0]).toFixed(1)+","+y(p[1]).toFixed(1)}).join("");
    if(opts.kind==="stack"){var base=P.slice().reverse().map(function(p){return "L"+x(p[0]).toFixed(1)+","+y(p[2]).toFixed(1)}).join("");
      o.push('<path d="'+line+base+'Z" fill="'+s.color+'" fill-opacity=".32"/>')}
    if(P.length===1||opts.dots)P.forEach(function(p){o.push('<circle cx="'+x(p[0]).toFixed(1)+'" cy="'+y(p[1]).toFixed(1)+'" r="'+(opts.dots?1.6:2.5)+'" fill="'+s.color+'"/>')});
    if(P.length>1&&!opts.dots)o.push('<path d="'+line+'" fill="none" stroke="'+s.color+'" stroke-width="'+(s.width||1.5)+'"'+(s.dash?' stroke-dasharray="3 3"':"")+' stroke-linejoin="round"/>')});
  o.push('</g>');
  (opts.refs||[]).forEach(function(r){if(!fin(r.y)||r.y<lo||r.y>hi)return;o.push('<line x1="'+M.l+'" x2="'+(W-M.r)+'" y1="'+y(r.y).toFixed(1)+'" y2="'+y(r.y).toFixed(1)+'" stroke="'+(r.color||GREY)+'" stroke-dasharray="4 3" stroke-width="1"/>'+
    (r.label?'<text x="'+(W-M.r-2)+'" y="'+(y(r.y)-4).toFixed(1)+'" text-anchor="end" class="rt">'+esc(r.label)+'</text>':""))});
  o.push('<line class="cx" x1="0" x2="0" y1="'+M.t+'" y2="'+(H-M.b)+'" visibility="hidden"/></svg>');
  box.innerHTML='<div class="cw">'+o.join("")+'<div class="tt"></div></div>'+leg;
  CH[id]={ser:ser,x:x,t0:t0,t1:t1,yf:yf,stack:stack};
  box.onmousemove=function(e){hover(id,box,e)};box.onmouseleave=function(){var t=box.querySelector&&box.querySelector(".tt");if(t)t.style.display="none";var c=box.querySelector&&box.querySelector(".cx");if(c)c.setAttribute("visibility","hidden")};
}
function hover(id,box,e){var c=CH[id];if(!c||!box.querySelector)return;var svg=box.querySelector("svg"),tt=box.querySelector(".tt"),cx=box.querySelector(".cx");if(!svg||!tt)return;
  var b=svg.getBoundingClientRect(),vx=(e.clientX-b.left)*W/b.width;var t=c.t0+(vx-M.l)*(c.t1-c.t0)/(W-M.l-M.r);if(t<c.t0||t>c.t1){tt.style.display="none";return}
  var rows=[],bestT=null;c.ser.forEach(function(s){var best=null,bd=1e18;s.pts.forEach(function(p){var d=Math.abs(p[0]-t);if(d<bd){bd=d;best=p}});
    if(best&&bd<(c.t1-c.t0)/20){bestT=best[0];rows.push('<div><i style="background:'+s.color+'"></i>'+esc(s.name)+'<b>'+esc(c.yf(c.stack?best[3]:best[1]))+'</b></div>')}});
  if(!rows.length){tt.style.display="none";return}
  cx.setAttribute("x1",c.x(bestT));cx.setAttribute("x2",c.x(bestT));cx.setAttribute("visibility","visible");
  tt.innerHTML='<div class="th">'+esc(day(bestT))+'</div>'+rows.join("");tt.style.display="block";
  var bx=box.getBoundingClientRect(),px=e.clientX-bx.left;tt.style.left=(px>bx.width/2?px-tt.offsetWidth-12:px+12)+"px";tt.style.top="8px"}
function card(id,title,sub,wide){return '<div class="cc'+(wide?" wide":"")+'"><div class="ct">'+esc(title)+'</div><div class="cs">'+esc(sub||"")+'</div><div id="'+id+'"></div></div>'}

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
    qs.map(function(q){return '<tr'+(q[0]==="p50"?' class="b"':"")+'><td>'+q[0]+'</td>'+lists.map(function(l){return '<td>'+(l[1].length?esc(l[2](rank(l[1],q[1]))):"n/a")+'</td>'}).join("")+'</tr>'}).join("")+
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
function statusDot(s){return '<i class="dot '+({ok:"g",done:"m",warn:"y",stale:"r",unknown:"m"}[s]||"m")+'" title="'+esc(s)+'"></i>'}
function spec(I){return [I.nodes&&I.nodes.length?I.nodes.join("+"):null,I.layout,fin(I.cards)?I.cards+" cards":null].filter(Boolean).join(" · ")}
function series(S,isls,c,fn){var ci=col(S,c);return isls.map(function(I,i){return {name:I.name,color:islColor(i),pts:I.rows.map(function(r){return [r[1],fn?fn(r):r[ci]]})}})}

function sft(S){
  var isls=S.islands||[],sy=S.syncer||{},refs=S.refs||{},C=function(n){return col(S,n)};
  var t1=NOW,t0=NOW-RANGE_H*3600,tp=t0-RANGE_H*3600;
  var active=isls.filter(function(I){return I.status==="ok"||I.status==="warn"});
  var fast=isls[0]||null,last=function(I){return I&&I.rows.length?I.rows[I.rows.length-1]:null};
  // KPIs
  var tps=sum(active.map(function(I){var r=last(I);return r?r[C("tps")]:null})),mfu=active.map(function(I){var r=last(I);return r?r[C("mfu")]:null}).filter(fin);
  var fs=fast?rowsIn(fast,t0,t1).map(function(r){return r[C("step_s")]}).filter(fin).sort(function(a,b){return a-b}):[],fp50=pctl(fs,0.5),ref=refs.sft_ref_step_s;
  var ev=(sy.events||[]),hr=ev.filter(function(e){return inRange(e[0],NOW-3600,NOW)}),cnt=function(c){return hr.filter(function(e){return e[1]===c}).length};
  var rss=(sy.rss||[]),rl=rss.length?rss[rss.length-1]:null,cap=refs.sft_syncer_rss_limit_bytes;
  var losses=isls.map(function(I){var r=last(I);return r&&fin(r[C("loss")])?I.name.replace("island ","")+" "+r[C("loss")].toFixed(4):null}).filter(Boolean);
  var kp=[["Islands active / total",active.length+" / "+isls.length,isls.map(function(I){return I.name+" "+I.status}).join(" · ")],
    ["Fleet tokens / s",k(tps),mfu.length?"MFU "+(sum(mfu)/mfu.length).toFixed(1)+"% (6N), last step":"no bench data"],
    ["Fast island step p50",dur(fp50),fin(fp50)&&ref?((fp50/ref-1)*100>=0?"+":"")+((fp50/ref-1)*100).toFixed(1)+"% vs production "+ref.toFixed(0)+" s · "+fs.length+" steps":"production "+dur(ref)],
    ["Latest loss",fast&&last(fast)?k(last(fast)[C("loss")],4):"n/a",losses.join(" · ")||"no data"],
    ["Syncer outer step",(sy.outer_step==null?"0":sy.outer_step)+" / "+(sy.total_steps==null?"n/a":sy.total_steps),sy.eta_ts?"ETA final cut "+day(sy.eta_ts):(sy.outer_step==null?"waiting for the first outer step":"ETA n/a")],
    ["Rejected / timeout / not carried",cnt("rejected")+" / "+cnt("timeout")+" / "+cnt("not_carried"),"last hour, from "+(sy.log_seen?"syncer.log":"the syncer tape")],
    ["Syncer RSS",rl?bytes(rl[1]):"n/a",rl&&cap?(100*rl[1]/cap).toFixed(0)+"% of the "+bytes(cap)+" cap · sampled "+ago(rl[0]):"not sampled"]];
  $("sKpis").innerHTML=kp.map(function(x){return '<div class="kc"><div class="kt">'+esc(x[0])+'</div><div class="kv">'+esc(x[1])+'</div><div class="ks">'+esc(x[2])+'</div></div>'}).join("");
  // islands now
  var hbmLim=refs.sft_hbm_limit_gib,hbmMax=Math.max(64,hbmLim*1.15);
  $("sNodesNote").textContent="last step "+ago(Math.max.apply(null,isls.map(function(I){return I.last_ts||0}).concat([0])))+" · syncer outer step "+(sy.outer_step==null?"none yet":sy.outer_step);
  $("sNodes").innerHTML='<table class="nt"><tr><th>Island</th><th>Step</th><th>Loss</th><th>grad_norm</th><th>Step time</th><th>Tokens/s</th><th class="bar">HBM peak vs '+hbmLim+' GiB</th><th>Last push (export)</th><th>Last apply</th><th>Data</th></tr>'+
    isls.map(function(I,i){var r=last(I)||[],h=r[C("hbm_gib")],pc=fin(h)?Math.min(100,100*h/hbmMax):0,lim=100*hbmLim/hbmMax;
      return '<tr><td><div class="nn">'+statusDot(I.status)+'<i class="sq" style="background:'+islColor(i)+'"></i>'+esc(I.name)+'</div><div class="ns">'+esc(spec(I)||"n/a")+'</div></td>'+
      '<td>'+(r[0]==null?"n/a":r[0])+' / '+(I.steps_total||"n/a")+'<div class="ns">'+ago(I.last_ts)+'</div></td><td>'+k(r[C("loss")],4)+'</td><td>'+k(r[C("grad_norm")],3)+'</td><td>'+dur(r[C("step_s")])+'</td><td>'+k(r[C("tps")])+'</td>'+
      '<td class="bar"><div class="bl">'+(fin(h)?h.toFixed(1)+" GiB":"n/a")+'</div><div class="bt"><div class="bf'+(fin(h)&&h>=hbmLim?" over":"")+'" style="width:'+pc.toFixed(1)+'%"></div><div class="bm" style="left:'+lim.toFixed(1)+'%"></div></div></td>'+
      '<td>'+(I.last_push?dur(I.last_push.export_s)+'<div class="ns">step '+I.last_push.step+' · '+ago(I.last_push.ts)+'</div>':"none yet")+'</td>'+
      '<td>'+(I.last_apply?dur(I.last_apply.apply_s)+'<div class="ns">step '+I.last_apply.step+' · '+ago(I.last_apply.ts)+'</div>':"none yet")+'</td><td>'+esc(I.dataset||"n/a")+'</td></tr>'}).join("")+'</table>';
  // fleet charts
  $("sFleetH").textContent="Fleet over the last "+rangeLabel();
  var g=[["fLoss","Loss","per island, ms-swift logging.jsonl"],["fGn","grad_norm","per island, before clipping"],["fLr","Learning rate","per island schedule"],
    ["fStep","Step time","bench wall_s per step; dashed: production "+dur(ref)],["fTps","Tokens / s","stacked by island; bench tokens_per_s held from each step on a shared time grid"],
    ["fHbm","HBM peak","max reserved over ranks per step; dashed: "+hbmLim+" GiB line"],["fHost","Host available memory","MemAvailable per node, sampled by the collector"],
    ["fSync","Sync overhead per step","export_s (solid) on steps that exported a fragment, apply_s (dashed) on steps that applied a broadcast"],
    ["fMerge","Syncer merge time","sync/merge_seconds (sync_ms) per merge: the merge itself, no waiting"],["fRound","Syncer round time","per merge: quorum wait (quorum_ms) and round open-to-done total (ms)"],["fGnorm","Outer gnorm by fragment","global delta norm per merge, colour by fragment"],
    ["fRss","Syncer RSS","VmRSS of the syncer; dashed: "+bytes(cap)+" cap"],["fEv","Syncer problems","rejected / timeout / not carried / stale / disconnect per bucket"]];
  $("sFleet").innerHTML=g.map(function(x){return card(x[0],x[1],x[2])}).join("");
  var total={name:"Fleet total",color:GREY,dash:true};
  chart("fLoss",{series:series(S,isls,"loss"),yfmt:function(v){return v.toFixed(3)},y0:false});
  chart("fGn",{series:series(S,isls,"grad_norm"),yfmt:function(v){return v.toFixed(2)}});
  chart("fLr",{series:series(S,isls,"lr"),yfmt:function(v){return v.toExponential(1)}});
  chart("fStep",{series:series(S,isls,"step_s"),refs:[{y:ref,label:"production "+dur(ref),legend:"Production"}],yfmt:dur});
  chart("fTps",{kind:"stack",hold:bucketFor(isls),series:series(S,isls,"tps"),yfmt:k});
  chart("fHbm",{series:series(S,isls,"hbm_gib"),refs:[{y:hbmLim,label:hbmLim+" GiB",legend:"Limit",color:"#eab308"}],yfmt:function(v){return v.toFixed(0)+" G"},y0:false,ymin:0.8*hbmLim});
  var hs=[];isls.forEach(function(I,i){Object.keys(I.host||{}).forEach(function(n,j){hs.push({name:n,color:PAL[(i*3+j)%PAL.length],pts:I.host[n].map(function(p){return [p[0],gib(p[1])]})})})});
  chart("fHost",{series:hs,yfmt:function(v){return v.toFixed(0)+" G"}});
  var ss=[];isls.forEach(function(I,i){ss.push({name:I.name+" export",color:islColor(i),pts:I.rows.filter(function(r){return r[C("export_s")]>0}).map(function(r){return [r[1],r[C("export_s")]]})},
    {name:I.name+" apply",color:islColor(i),dash:true,pts:I.rows.filter(function(r){return r[C("applies")]||r[C("apply_s")]>0}).map(function(r){return [r[1],r[C("apply_s")]]})})});
  chart("fSync",{series:ss,yfmt:dur,dots:false});
  chart("fMerge",{dots:true,series:[{name:"merge (sync_ms)",color:PAL[0],pts:(sy.merges||[]).map(function(m){return [m[0],m[3]]})}],yfmt:dur});
  chart("fRound",{dots:true,series:[{name:"quorum wait",color:PAL[1],pts:(sy.merges||[]).map(function(m){return [m[0],m[6]]})},{name:"round total (ms)",color:PAL[2],dash:true,pts:(sy.merges||[]).map(function(m){return [m[0],m[5]]})}],yfmt:dur});
  var fr={};(sy.merges||[]).forEach(function(m){var f=m[2]==null?"?":m[2];(fr[f]=fr[f]||[]).push([m[0],m[4]])});
  var fk=Object.keys(fr);chart("fGnorm",{dots:true,series:fk.map(function(f,i){return {name:"fragment "+f,color:PAL[(+f||i)%PAL.length],pts:fr[f],legend:fk.length<=8}}),yfmt:function(v){return v.toFixed(2)}});
  chart("fRss",{series:[{name:"syncer RSS",color:PAL[3],pts:rss.map(function(p){return [p[0],p[1]]})}],refs:[{y:cap,label:bytes(cap)+" cap",legend:"Cap"}],yfmt:bytes});
  var cats=[["rejected",PAL[8]],["timeout",PAL[1]],["not_carried",PAL[5]],["stale",PAL[3]],["disconnect",GREY]];
  chart("fEv",{kind:"bars",series:cats.map(function(c){return {name:c[0].replace("_"," "),color:c[1],pts:ev.filter(function(e){return e[1]===c[0]}).map(function(e){return [e[0],1]})}}),yfmt:function(v){return +v.toFixed(1)+""}});
  // distribution
  $("sDistH").textContent="Distribution over the last "+rangeLabel();
  var slowT=ref*(refs.sft_slow_factor||1.1),dc=isls.map(function(I,i){var R=rowsIn(I,t0,t1);
    return distCard(islColor(i),I.name,spec(I),[{title:"Step time",short:"Step time",vals:R.map(function(r){return r[C("step_s")]}),fmt:dur,slow:[slowT,"over "+dur(slowT)],count:"Steps"},
      {title:"Sync blocking per step (sync_s)",short:"sync_s",vals:R.map(function(r){return r[C("sync_s")]}),fmt:dur,slow:[ref*0.05,"over 5% of a step"]}])});
  var mg=(sy.merges||[]).filter(function(m){return inRange(m[0],t0,t1)}).map(function(m){return m[3]}),ms=mg.filter(fin).sort(function(a,b){return a-b}),mp=pctl(ms,0.5);
  var os=(sy.outer||[]).filter(function(o){return inRange(o[0],t0,t1)}).map(function(o){return o[3]});
  dc.push(distCard(GREY,"syncer",(sy.meta&&sy.meta.learners!=null?"learners "+sy.meta.learners+" · ":"")+"H="+((sy.meta||{}).sync_interval||"n/a"),
    [{title:"Merge time per round",short:"merge",vals:mg,fmt:dur,slow:fin(mp)?[2*mp,"over 2x p50"]:null,count:"Rounds"},{title:"Outer step time",short:"outer step",vals:os,fmt:dur,slow:null}]));
  $("sDist").innerHTML=dc.join("");
  // cost / efficiency
  $("sCostH").textContent="Cost / efficiency over the last "+rangeLabel();
  function agg(a,b){var o={tok:0,tgt:0,ch:0,sync:0,wall:0,mfuw:0,mfus:0};isls.forEach(function(I){var cards=fin(I.cards)?I.cards:0;rowsIn(I,a,b).forEach(function(r){var w=r[C("step_s")];
      o.tok+=fin(r[C("tokens")])?r[C("tokens")]:0;o.tgt+=fin(r[C("target_tokens")])?r[C("target_tokens")]:0;
      if(fin(w)){o.wall+=w;o.ch+=w*cards/3600;o.sync+=fin(r[C("sync_s")])?r[C("sync_s")]:0;if(fin(r[C("mfu")])){o.mfuw+=w*r[C("mfu")];o.mfus+=w}}})});
    o.tpc=o.ch?o.tok/(o.ch*3600):null;o.mfu=o.mfus?o.mfuw/o.mfus:null;o.share=o.wall?o.sync/o.wall:null;return o}
  var A=agg(t0,t1),B=agg(tp,t0);
  function delta(a,b,fmt){if(!fin(a)||!fin(b)||!b)return '<span class="dl">no previous '+rangeLabel()+'</span>';var d=a-b,p=100*d/b;
    return '<span class="dl '+(d>=0?"up":"dn")+'">'+(d>=0?"&uarr; +":"&darr; ")+esc(fmt(d))+' · '+(p>=0?"+":"")+p.toFixed(0)+'% vs previous '+rangeLabel()+'</span>'}
  var etas=isls.map(function(I){var r=last(I),R=I.rows.slice(-20).map(function(x){return x[C("step_s")]}).filter(fin).sort(function(a,b){return a-b}),p=pctl(R,0.5);
    return r&&I.steps_total&&fin(p)&&fin(I.last_ts)?I.last_ts+(I.steps_total-r[0])*p:null}).filter(fin),eta=etas.length?Math.max.apply(null,etas):null;
  var ck=[["Tokens trained",k(A.tok),"bench tokens, all islands",delta(A.tok,B.tok,k)],["Target tokens",k(A.tgt),"loss-bearing tokens (yeto plugin)",delta(A.tgt,B.tgt,k)],
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
  var pe=ev.filter(function(e){return e[1]!=="joined"}).slice(-40).reverse();
  $("sEvents").innerHTML=pe.length?'<table class="nt ev"><tr><th>Time</th><th>Category</th><th>Learner</th><th>Source</th></tr>'+pe.map(function(e){return '<tr><td>'+esc(day(e[0]))+'</td><td>'+esc(e[1])+'</td><td>'+(e[2]==null?"":esc(e[2]))+'</td><td>'+esc(e[3])+'</td></tr>'}).join("")+'</table>':'<div class="empty">no syncer events</div>';
}
function bucketFor(isls){var s=[];isls.forEach(function(I){I.rows.forEach(function(r){if(fin(r[5]))s.push(r[5])})});s.sort(function(a,b){return a-b});var p=pctl(s,0.5);return Math.max(fin(p)?p*1.05:600,RANGE_H*3600/120)}

/* ---------- RL: distribution of round phase durations ---------- */
function rlDist(V){
  var box=$("rlDist");if(!box)return;var O=V.overview||{},ids=(O.islands||[]).map(function(c){return c.id}),t0=NOW-RANGE_H*3600;
  var cards=ids.map(function(id,i){var rr=((V.islands||{})[id]||{}).rounds||[],sel=rr.filter(function(r){var p=r.phases&&(r.phases.R||r.phases.T);return !p||!fin(p[1])||p[1]>=t0});
    var R=sel.map(function(r){return (r.dur||{}).R}),T=sel.map(function(r){return (r.dur||{}).T}),c=(O.islands||[])[i]||{};
    var pr=R.filter(fin).sort(function(a,b){return a-b}),pt=T.filter(fin).sort(function(a,b){return a-b});
    return distCard(PAL[i%PAL.length],"Island "+id,[c.cloud,c.gpu&&c.gpus!=null?c.gpu+" × "+c.gpus:c.gpu].filter(Boolean).join(" · "),
      [{title:"Rollout time per round",short:"rollout",vals:R,fmt:dur,slow:pr.length?[2*pctl(pr,0.5),"over 2x p50"]:null,count:"Rounds"},
       {title:"Train time per round",short:"train",vals:T,fmt:dur,slow:pt.length?[2*pctl(pt,0.5),"over 2x p50"]:null}])});
  $("rlDistH").textContent="Distribution over the last "+rangeLabel();
  box.innerHTML=cards.join("")||'<div class="empty">no data</div>';
}

function render(view,offline){
  V=view;OFF=offline;if(!V)return;var S=V.sft;
  NOW=offline?((S&&S.now)||(V.overview||{}).now||Date.now()/1000):Date.now()/1000;
  var sroot=$("sftRoot"),rroot=$("rlRoot");
  if(document.body&&document.body.classList)document.body.classList.toggle("sftmode",!!S);
  if(sroot)sroot.hidden=!S;if(rroot)rroot.hidden=!!S;
  controls(offline,onReload);
  var run=(V.overview||{}).run;
  if(S){$("title").innerHTML='yeto metrics <span class="tg">'+esc(run||"unnamed run")+'</span><span class="tg">SFT · decoupled DiLoCo</span>';
    document.title="yeto metrics · "+(run||"SFT");
    $("sub").textContent="updated "+clock(offline?NOW:Date.now()/1000,true)+" · times in your local zone"+(offline?" · offline export, data as of "+day(NOW):" · live");
    sft(S)}
  else rlDist(V);
}
return {bins:bins,render:render,setReload:function(fn){onReload=fn},rangeH:function(){return RANGE_H}};
})();
