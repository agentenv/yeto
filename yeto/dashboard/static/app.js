// yeto fleet dashboard (v7). No build step, no external requests.
// Live mode polls /api/overview every 5 s; offline export reads the inlined JSON.
(function(){
"use strict";
var DATA=null,node=document.getElementById("yeto-data");
if(node){try{DATA=JSON.parse(node.textContent)}catch(e){DATA=null}}
var OFFLINE=!!DATA, POLL_MS=5000;
var COLORS=["var(--i0)","var(--i1)","var(--i2)","var(--i3)","#0f9bb0","#b0457a","#6b8e23","#7a6cf0"];
var SEV=[["bad","严重"],["warn","警告"],["muted","提示"]];
var OV=null,ROUNDS=[],cur={m:"reward",isl:null,round:null},DRILL=null;
function el(id){return document.getElementById(id)}
function esc(s){return String(s==null?"":s).replace(/[&<>"']/g,function(c){return {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]})}
var ND='<span class="nd">无数据</span>';
function f(x,d){if(x==null||x!==x)return ND;if(typeof x!="number")return esc(x);var a=Math.abs(x);
 if(d!=null)return x.toFixed(d);return a!==0&&(a<1e-3||a>=1e5)?x.toExponential(2):(+x.toPrecision(4)).toString()}
function api(path){
 if(OFFLINE){var p=path.split("?")[0],o=null;
  if(p=="/api/overview")o=DATA.overview;else if(p=="/api/rounds")o={rounds:DATA.rounds,derived:true};
  else if(p=="/api/fleet")o=DATA.fleet;else if(p=="/api/events")o=DATA.events;
  else if(p.indexOf("/api/islands/")==0)o=DATA.islands[decodeURIComponent(p.slice(13))];
  return Promise.resolve(o)}
 return fetch(path,{cache:"no-store"}).then(function(r){if(!r.ok)throw new Error(r.status);return r.json()})}
function idx(id){if(!OV)return -1;for(var i=0;i<OV.islands.length;i++)if(OV.islands[i].id===id)return i;return -1}
function color(id){var i=idx(id);return COLORS[(i<0?0:i)%COLORS.length]}
function ts(t){if(t==null)return "无数据";var d=new Date(t*1000);function p(n){return (n<10?"0":"")+n}
 return d.getFullYear()+"-"+p(d.getMonth()+1)+"-"+p(d.getDate())+" "+p(d.getHours())+":"+p(d.getMinutes())+":"+p(d.getSeconds())}
function bar(pct,cls){return '<span class="bar" style="display:block;margin-top:2px"><i style="width:'+Math.min(100,pct||0)+'%;background:var(--'+cls+')"></i></span>'}
function costCls(p){return p==null?"muted":p>=80?"bad":p>=60?"warn":"ok"}
function copyBtn(cmd){return '<button class="cp" data-copy="'+esc(cmd)+'" title="复制到剪贴板">复制：'+esc(cmd)+'</button>'}

function header(){var o=OV,c=o.cost,gs=o.global_status||{};
 var mode=OFFLINE?'<span class="chip off">离线导出 · 生成于 '+ts(DATA.generated_at)+'</span>':'<span class="chip live">实时 · SSH 隧道 · 5s 刷新</span>';
 var cost;if(c.total_usd==null)cost='<span class="k">成本 </span>'+ND;
 else{var cls=costCls(c.budget_pct);cost='<span style="min-width:170px"><span class="k">成本（估算） </span><span class="num '+cls+'">$'+c.total_usd.toFixed(1)+(c.budget_usd?' / $'+c.budget_usd.toFixed(0):'')+'</span>'+(c.budget_pct!=null?bar(c.budget_pct,cls):'')+'</span>'}
 var age=o.data_ts!=null?Math.max(0,o.now-o.data_ts):null;
 el("title").textContent=(o.run_kind=="single_island"?"单岛总览 · ":"Syncer 总览 · ")+(o.run||"未命名运行")+(o.run_inferred?"（由磁带推断）":"");
 el("hdr").innerHTML='<span class="mono">run: '+esc(o.run||"-")+'</span> '+mode+
  ' <span class="chip" style="background:var(--'+(gs.level=="muted"?"chip":gs.level)+');color:'+(gs.level=="muted"?"var(--muted)":"var(--panel)")+'">全局：'+esc(gs.text)+'</span> '+cost+
  ' <span class="k">数据更新于 '+ts(o.data_ts)+(age!=null?'（'+Math.round(age)+'s 前）':'')+'</span>'}

function alerts(){var a=OV.alerts;
 if(!a.length){el("alerts").innerHTML='<div class="panel k">无告警</div>';return}
 el("alerts").innerHTML=a.map(function(x,i){var c=SEV[x.sev];
  var loc=(x.island!=null?'定位 岛 '+esc(x.island):'')+(x.round!=null?' · r'+esc(x.round):'')+(x.island==null&&x.round==null?'查看成本效率':'');
  return '<button class="al" data-a="'+i+'" style="border-left-color:var(--'+(c[0]=="muted"?"line":c[0])+')"><span class="sev '+c[0]+'">'+c[1]+'</span><div class="t">'+esc(x.title)+'</div><div class="k">'+esc(x.detail)+'</div><div class="k" style="color:var(--accent)">'+loc+'</div></button>'}).join("")}

function legend(){el("lg").innerHTML=OV.islands.map(function(s){return '<span class="chip"><span class="dot" style="background:'+color(s.id)+'"></span>岛 '+esc(s.id)+'</span>'}).join(" ")}
function tabs(){el("tabs").innerHTML=OV.metrics.map(function(m){return '<button data-m="'+m[0]+'" aria-pressed="'+(m[0]==cur.m)+'">'+esc(m[1])+'</button>'}).join("")}

function chart(key){var W=720,H=260,L=48,B=20,T=10,ser=OV.series,ids=OV.islands.map(function(s){return s.id});
 var all=[],xs=[];ids.forEach(function(id){(ser[id]&&ser[id][key]||[]).forEach(function(p){xs.push(p[0]);all.push(p[1])})});
 var label=(OV.metrics.filter(function(m){return m[0]==key})[0]||[key,key])[1];
 if(!all.length)return '<div class="empty">'+esc(label)+'：无数据（该指标未出现在磁带中；旧磁带或尚未发射）</div>';
 var x0=Math.min.apply(0,xs),x1=Math.max.apply(0,xs);if(x1==x0)x1=x0+1;
 var mn=Math.min.apply(0,all),mx=Math.max.apply(0,all),sp=mx-mn||Math.abs(mx)||1;if(mx==mn){mn-=sp/2;mx+=sp/2;sp=mx-mn}
 function X(x){return L+(x-x0)/(x1-x0)*(W-L-6)}function Y(v){return T+(mx-v)/sp*(H-T-B)}
 var o='<svg viewBox="0 0 '+W+' '+H+'" width="100%" role="img" aria-label="'+esc(label)+' 各岛叠加曲线">';
 (OV.round_marks||[]).forEach(function(r){if(r.missed&&r.round>=x0&&r.round<=x1)o+='<rect x="'+(X(r.round)-3).toFixed(1)+'" y="'+T+'" width="6" height="'+(H-T-B)+'" style="fill:var(--warn);opacity:.13"/>'});
 for(var g=0;g<=3;g++){var y=T+(H-T-B)*g/3;o+='<line class="ax" x1="'+L+'" x2="'+W+'" y1="'+y+'" y2="'+y+'"/><text x="2" y="'+(y+4)+'">'+(mx-sp*g/3).toPrecision(3)+'</text>'}
 var step=Math.max(1,Math.ceil((x1-x0)/8));
 (OV.round_marks||[]).forEach(function(r){if(r.round>=x0&&r.round<=x1&&(r.round-x0)%step==0)o+='<line x1="'+X(r.round)+'" x2="'+X(r.round)+'" y1="'+T+'" y2="'+(H-B)+'" style="stroke:var(--muted)" stroke-dasharray="3 3" opacity=".5"/>'});
 for(var t=x0;t<=x1;t+=step)o+='<text x="'+(X(t)+2)+'" y="'+(H-6)+'">r'+t+'</text>';
 if(cur.round!=null&&cur.round>=x0&&cur.round<=x1)o+='<line x1="'+X(cur.round)+'" x2="'+X(cur.round)+'" y1="'+T+'" y2="'+(H-B)+'" style="stroke:var(--accent)" stroke-width="2"/>';
 ids.forEach(function(id){var v=ser[id]&&ser[id][key]||[];if(!v.length)return;var dim=cur.isl!=null&&cur.isl!==id;
  if(v.length==1)o+='<circle cx="'+X(v[0][0]).toFixed(1)+'" cy="'+Y(v[0][1]).toFixed(1)+'" r="3" style="fill:'+color(id)+';opacity:'+(dim?.3:1)+'"/>';
  else o+='<polyline fill="none" style="stroke:'+color(id)+';opacity:'+(dim?.3:1)+'" stroke-width="'+(dim?1.2:1.8)+'" points="'+v.map(function(p){return X(p[0]).toFixed(1)+","+Y(p[1]).toFixed(1)}).join(" ")+'"/>'});
 OV.alerts.forEach(function(a){if(a.island==null||a.round==null)return;var v=ser[a.island]&&ser[a.island][key]||[];
  var p=null;v.forEach(function(q){if(q[0]<=a.round)p=q});if(!p&&v.length)p=v[0];if(!p)return;
  o+='<circle cx="'+X(p[0]).toFixed(1)+'" cy="'+Y(p[1]).toFixed(1)+'" r="5" style="fill:var(--'+SEV[a.sev][0]+');stroke:var(--panel)" stroke-width="2"><title>'+esc(a.title)+'</title></circle>'});
 return o+'</svg>'}
function draw(){el("big").innerHTML=chart(cur.m);[].forEach.call(document.querySelectorAll("#tabs button"),function(b){b.setAttribute("aria-pressed",b.dataset.m==cur.m)})}

function status(s){return {ok:'<span class="ok">健康</span>',stale:'<span class="bad">掉线疑似</span>',lost:'<span class="bad">已丢失</span>',recovery:'<span class="bad">RECOVERY_REQUIRED</span>',done:'<span class="muted">已结束</span>',starting:'<span class="warn">启动中</span>',stopped:'<span class="muted">已停止</span>',unknown:ND}[s.status]||esc(s.status)}
function cards(){if(!OV.islands.length){el("cards").innerHTML='<div class="empty">无数据</div>';return}
 el("cards").innerHTML=OV.islands.map(function(s){var bad=s.status=="stale"||s.status=="lost"||s.status=="recovery";
  var hb=s.heartbeat_seen?(s.heartbeat_age_s!=null?Math.round(s.heartbeat_age_s)+'s 前':ND):'<span class="nd">无数据（磁带无 rl_heartbeat）</span>';
  return '<div class="hc'+(cur.isl===s.id?' sel':'')+'" data-k="'+esc(s.id)+'" tabindex="0"><div style="display:flex;justify-content:space-between"><span><span class="dot" style="background:'+color(s.id)+'"></span><b>岛 '+esc(s.id)+'</b>'+(s.name?' <span class="k">'+esc(s.name)+'</span>':'')+'</span>'+status(s)+'</div>'+
  '<div class="hr"><span class="k">last event</span><span class="num '+(bad?'bad':'')+'">'+(s.last_event_age_s!=null?Math.round(s.last_event_age_s)+'s':ND)+'</span></div>'+
  '<div class="hr"><span class="k">心跳</span><span>'+hb+'</span></div>'+
  '<div class="hr"><span class="k">round / policy</span><span class="num">'+f(s.round)+' / '+f(s.policy_version)+'</span></div>'+
  '<div class="hr"><span class="k">staleness / 贡献</span><span class="num">'+f(s.staleness)+' / '+f(s.contribution)+'</span></div>'+
  '<div class="hr"><span class="k">GPU util / 显存</span><span class="num">'+(s.gpu_util_pct!=null?s.gpu_util_pct+'%':ND)+' / '+(s.mem_pct!=null?s.mem_pct+'%':ND)+'</span></div>'+
  (s.host_mem_peak_bytes!=null?'<div class="hr"><span class="k">主机内存 峰值 / 当前</span><span class="num">'+(s.host_mem_peak_bytes/1073741824).toFixed(1)+' / '+(s.host_mem_current_bytes!=null?(s.host_mem_current_bytes/1073741824).toFixed(1):ND)+' GiB</span></div>':'')+
  (s.gpu_util_pct!=null?'<div class="bar" style="margin-top:4px"><i style="width:'+s.gpu_util_pct+'%;background:'+color(s.id)+'"></i></div>':'')+'</div>'}).join("")}

function eff(){var c=OV.cost;
 if(!c.islands.length){el("eff").innerHTML='<div class="empty">无数据（无 fleet.jsonl：成本需要岛生命周期与价目表）</div><div class="k">'+esc(c.note)+'</div>';return}
 var h='';if(c.total_usd!=null){var cls=costCls(c.budget_pct);h+='<div class="k">累计估算'+(c.budget_usd?' / 预算上限':'')+'</div><div class="v '+cls+'">$'+c.total_usd.toFixed(1)+(c.budget_usd?' / $'+c.budget_usd.toFixed(0):'')+'</div>'+(c.budget_pct!=null?bar(c.budget_pct,cls):'')+
  '<div class="k" style="margin-top:6px">当前速率 '+(c.burn_usd_h!=null?'$'+c.burn_usd_h.toFixed(2)+'/h':ND)+(c.hours_to_cap!=null?'，预计 '+c.hours_to_cap.toFixed(1)+' h 后触达上限':'')+'</div>'}
 h+='<div class="scroll"><table><tr><th>岛</th><th>$/h</th><th>时长 h</th><th>累计 $</th><th>$/1M tok</th><th>reward/$</th></tr>'+c.islands.map(function(r){
  return '<tr><td><span class="dot" style="background:'+color(r.id)+'"></span>'+esc(r.id)+'</td><td class="num">'+(r.priced?f(r.rate_usd_h,2):'<span class="nd">未定价</span>')+'</td><td class="num">'+f(r.hours,2)+'</td><td class="num">'+(r.priced?f(r.cost_usd,2):'<span class="nd">未定价</span>')+'</td><td class="num">'+(r.local?'0（自有）':f(r.usd_per_1m_tok,2))+'</td><td class="num">'+(r.ranked?f(r.reward_per_usd,4):'<span class="muted">-</span>')+'</td></tr>'}).join("")+'</table></div>'+
  '<div class="k" style="margin-top:6px">'+esc(c.note)+'；价目表：'+esc(c.prices_source||"-")+'；reward/$ = (末轮 − 首轮 reward) / 累计 $；本地岛按 $0 计，不参与排名。</div>';
 el("eff").innerHTML=h}

function rounds(){var only=el("onlybad").checked;
 var rows=ROUNDS.filter(function(r){return !only||r.bad}).slice().reverse();
 if(!ROUNDS.length){el("rounds").innerHTML='<div class="empty">无数据（未提供 syncer 磁带，或单岛 no-sync 运行）</div>';return}
 el("rounds").innerHTML='<div class="scroll"><table><tr><th>round</th><th>fragment</th><th>responded/expected</th><th>missed</th><th>quorum ms</th><th>grace ms</th><th>sync ms</th><th>merge ms</th><th>gnorm</th><th>重发</th></tr>'+rows.map(function(r){
  var miss=r.missed.length;return '<tr id="r'+esc(r.round)+'" class="'+(miss?'miss ':'')+(cur.round==r.round?'hl':'')+'"><td class="num">'+f(r.round)+'</td><td class="num">'+f(r.fragment)+'</td><td class="num '+(r.responded<r.expected?'warn':'')+'">'+f(r.responded)+'/'+f(r.expected)+'</td><td>'+(miss?esc(r.missed.map(function(x){return "岛 "+x}).join(",")):'<span class="muted">-</span>')+'</td><td class="num '+(r.quorum_ms>10000?'bad':'')+'">'+f(r.quorum_ms,0)+'</td><td class="num">'+f(r.grace_ms,0)+'</td><td class="num">'+f(r.sync_ms,0)+'</td><td class="num">'+f(r.merge_ms,0)+'</td><td class="num">'+f(r.gnorm)+'</td><td class="num '+(r.resend?'warn':'')+'">'+(r.resend||'')+'</td></tr>'}).join("")+'</table></div>'}

function ray(v,card){var e=v.ray_embed||{mode:"unknown"},c=card;
 if(e.mode=="none"||e.mode=="unknown"){var r=v.resource||{};var tok=c.tok_s;
  return '<div class="panel" style="background:var(--panel2)"><b>Ray dashboard：'+(e.mode=="none"?'不可嵌入':'无数据')+'</b><div class="k">'+esc(e.note||"")+'；退化为资源 / 事件面板</div><div class="stat" style="margin-top:6px"><div><div class="k">GPU util</div><div class="v">'+(c.gpu_util_pct!=null?c.gpu_util_pct+'%':ND)+'</div></div><div><div class="k">显存</div><div class="v">'+(c.mem_pct!=null?c.mem_pct+'%':ND)+'</div></div><div><div class="k">tok/s</div><div class="v">'+(tok!=null?Math.round(tok):ND)+'</div></div></div>'+(r.available===false?'<div class="k">NVML 不可用（available:false）</div>':'')+'</div>'}
 var cmd=e.mode=="tunnel"?e.command:null;
 var btn=OFFLINE?'<div class="k">离线导出不加载 iframe</div>':'<button data-ray="'+e.port+'">加载 Ray dashboard（端口 '+e.port+'）</button>';
 return '<div class="ray" id="raybox"><div><b>Ray dashboard 嵌入位 - 岛 '+esc(c.id)+'</b><br>'+(e.mode=="direct"?'本机直连可嵌入':'经 SSH 隧道可嵌入（请先自行建立隧道）')+'<br>'+(cmd?'<span class="mono">'+esc(cmd)+'</span><br>'+copyBtn(cmd)+'<br>':'')+btn+'</div></div>'}
function cellTable(v){if(!v.cells||!v.cells.length)return '<div class="nd">无数据：待 infra 就绪（rl_cell_snapshot 未发射，journal 无 gpu_pool 记录）</div>';
 return '<div class="k">来源：'+esc(v.cells_source)+'</div><div class="scroll"><table><tr><th>cell</th><th>角色</th><th>GPU</th><th>状态</th></tr>'+v.cells.map(function(c){var st=String(c.state||"");
  return '<tr><td class="mono" title="'+esc(c.cell)+'">'+esc(String(c.cell).slice(0,16))+'</td><td>'+esc(c.role)+'</td><td class="num">'+f(c.gpus)+'</td><td class="'+(/reject|unreach|unbound|lost/.test(st)?'bad':'ok')+'">'+esc(st||"-")+(c.note?' <span class="k">'+esc(c.note).slice(0,60)+'</span>':'')+'</td></tr>'}).join("")+'</table></div>'}
function txTable(v){if(!v.transactions.length)return '<div class="nd">无数据（controller journal 无事务记录）</div>';
 return '<div class="scroll"><table><tr><th>txn</th><th>类型</th><th>阶段</th><th>结果</th></tr>'+v.transactions.slice(-12).map(function(t){var res=t.result;
  var cls=res=="RECOVERY_REQUIRED"?'bad':(res=="COMMITTED"||res=="SUCCEEDED")?'ok':res?'warn':'muted';
  return '<tr><td class="mono">'+esc(t.tx_id)+'</td><td>'+esc(t.kind||t.scope||"-")+'</td><td style="white-space:normal;text-align:left">'+esc(t.phases.join(" → ")||"-")+'</td><td class="'+cls+'" title="'+esc(t.error||"")+'">'+esc(res||"进行中")+'</td></tr>'}).join("")+'</table></div>'}
function drill(id){DRILL=id;api("/api/islands/"+encodeURIComponent(id)).then(function(v){if(!v||DRILL!==id)return;var s=v.card,d=el("drill");
 var ev=v.recent_events.slice(-12).map(function(e){return esc(e.type)+' '+esc(e.summary)}).join("\n")||"无数据";
 var run=OV.run||"<run>";
 d.innerHTML='<div style="display:flex;justify-content:space-between;align-items:center"><h2 style="margin:0"><span class="dot" style="background:'+color(id)+'"></span>岛明细：'+esc(id)+'</h2><button id="cls">收起</button></div>'+
 '<div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(240px,1fr));margin-top:8px"><div><h2>进度 / policy</h2><div class="mono">round '+f(s.round)+' · rollout '+f(s.rollout_id)+' · policy '+f(s.policy_version)+'</div><div class="k">阶段 '+f(s.phase)+' · '+f(s.cloud)+' · '+f(s.region)+' · '+(s.gpus!=null?s.gpus+'x ':'')+f(s.gpu)+'</div>'+
 '<h2 style="margin-top:10px">cell 表</h2>'+cellTable(v)+'</div>'+
 '<div><h2>E1 事务</h2>'+txTable(v)+
 '<h2 style="margin-top:10px">事件流（最新）</h2><pre class="mono" style="font-size:11px;white-space:pre-wrap;margin:0;max-height:220px;overflow:auto">'+ev+'</pre>'+
 '<div style="margin-top:6px">'+copyBtn("yeto logs "+run)+copyBtn("yeto dashboard export --run "+run+" -o "+run+".html")+'</div></div><div>'+ray(v,s)+'</div></div>';
 d.classList.add("open");el("cls").onclick=function(){d.classList.remove("open");DRILL=null;cur.isl=null;cards();draw()}})}

function focus(a){cur.isl=a.island;cur.round=a.round;if(a.metric)cur.m=a.metric;draw();rounds();cards();
 if(a.island!=null)drill(a.island);
 if(a.round!=null){var tr=el("r"+a.round);if(tr){var box=el("rounds");box.scrollTop=tr.offsetTop-box.offsetTop-30}}
 (a.island==null&&a.round==null?el("eff"):el("big")).scrollIntoView({behavior:"smooth",block:"center"})}
document.addEventListener("click",function(e){var t=e.target.closest("[data-copy],[data-ray],[data-a],[data-m],.hc");if(!t)return;
 if(t.dataset.copy!=null){var txt=t.dataset.copy;if(navigator.clipboard)navigator.clipboard.writeText(txt).catch(function(){});t.textContent="已复制";setTimeout(function(){t.textContent="复制："+txt},1200);return}
 if(t.dataset.ray!=null){var box=el("raybox");box.innerHTML='<iframe title="ray" style="width:100%;min-height:420px;border:0" src="//127.0.0.1:'+(+t.dataset.ray)+'/"></iframe>';return}
 [].forEach.call(document.querySelectorAll(".al"),function(x){x.classList.toggle("sel",x===t)});
 if(t.dataset.a!=null)focus(OV.alerts[+t.dataset.a]);else if(t.dataset.m){cur.m=t.dataset.m;draw()}
 else{cur.isl=t.dataset.k;cards();draw();drill(cur.isl);el("drill").scrollIntoView({behavior:"smooth",block:"start"})}});
document.addEventListener("keydown",function(e){if(e.key=="Enter"&&e.target.classList&&e.target.classList.contains("hc"))e.target.click()});
el("onlybad").onchange=rounds;

function render(){header();alerts();legend();tabs();draw();cards();eff();rounds()}
function refresh(){return Promise.all([api("/api/overview"),api("/api/rounds")]).then(function(r){OV=r[0];ROUNDS=r[1].rounds||[];render();if(DRILL!=null)drill(DRILL)})
 .catch(function(err){el("hdr").innerHTML='<span class="chip off">刷新失败：'+esc(err.message)+'（隧道断开？）</span>'})}
refresh();if(!OFFLINE)setInterval(refresh,POLL_MS);
})();
