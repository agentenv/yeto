#!/usr/bin/env python3
"""FN(G6) 续跑前缀重算代价估算。只读 G6 数据。"""
import json, re, statistics as st, datetime as dt
R='/home/michael/work/s1-runs/s17-fncodex-r3-20261008a/'
OUT='/home/michael/work/s1-runs/s18-aru3-fn-prefix/estimate.json'
def pct(a,p):
    a=sorted(a); 
    if not a: return None
    k=(len(a)-1)*p/100; f=int(k); c=min(f+1,len(a)-1); return a[f]+(a[c]-a[f])*(k-f)
def summ(a): return dict(n=len(a),p50=pct(a,50),p90=pct(a,90),max=max(a) if a else None,mean=st.mean(a) if a else None)
an=json.load(open(R+'analysis.json'))
tr=an['per_trajectory']
turns=[t['turns'] for t in tr]
ctx_all=[c for t in tr for c in t['turn_context_tokens']]
final_ctx=[t['turn_context_tokens'][-1] for t in tr if t['turn_context_tokens']]
multi=[t for t in tr if t['turns']>=5]            # 长轨迹(尾部候选)
multi_final=[t['turn_context_tokens'][-1] for t in multi]
tape_ctx_sum=sum(ctx_all)
# SGLang Prefill 行
rx=re.compile(r'\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0.*?Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+).*?#running-req: (\d+), #queue-req: (\d+), #pending-token: (\d+).*?input throughput \(token/s\): ([\d.]+)')
rd=re.compile(r'\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) TP0.*?Decode batch, #running-req: (\d+)')
pf=[];dec=[]
for l in open(R+'launch.log',errors='replace'):
    m=rx.search(l)
    if m:
        pf.append(dict(ts=m[1],seq=int(m[2]),new=int(m[3]),cached=int(m[4]),run=int(m[5]),q=int(m[6]),pend=int(m[7]),thr=float(m[8]))); continue
    m=rd.search(l)
    if m: dec.append((m[1],int(m[2])))
utc=lambda s: dt.datetime.strptime(s,'%Y-%m-%d %H:%M:%S').replace(tzinfo=dt.timezone.utc).timestamp()
tape=[json.loads(l) for l in open(R+'tape-direct/yeto-s17-fncodex-r3-20261008a/l0/rank0/rl-island-0.jsonl')]
ph={(r['phase'],r.get('rollout_id'),r.get('policy_version')):r['time_unix'] for r in tape if r.get('event')=='rl_driver_phase'}
gen_start=ph[('generate',0,0)]; gen_end=ph[('onload',0,None)]
spans=[r for r in an['timeline_spans']]
gen_s=[s['seconds'] for s in spans if s['task']=='generate'][0]
train_s=an['round_trained'][0]['step_seconds']
# 预填充
big=[p for p in pf if p['new']>1]
cached_sum=sum(p['cached'] for p in pf)
new_sum_gen=sum(p['new'] for p in pf if gen_start-3<=utc(p['ts'])<=gen_end+3)
# 吞吐：该字段是"上一日志间隔内 token/间隔秒"，1 秒时间戳粒度下同秒批次会出现荒谬大值。
# 取"续批(前一批 pending>0 时的下一批, 即连续分块预填充)且 new>=6000"的值里 1e4~1e5 之间的作为连续预填充速率
sane=[p['thr'] for p in big if p['new']>=6000 and 1e4<=p['thr']<=1e5]
allthr=[p['thr'] for p in big]
# 总体下界：生成段内总预填充 token / 有预填充活动的墙钟
thr_lo=new_sum_gen/gen_s
rate=pct(sane,50) if sane else None
# 在途并发：Decode 行 running-req 时序
dsec=[(utc(t)-gen_start,n) for t,n in dec if gen_start<=utc(t)<=gen_end+3]
peak=max(n for _,n in dsec)
# 尾部：running<=2(24条的最慢~10%)起至生成段结束
tail_start=None
for i,(t,n) in enumerate(dsec):
    if all(m<=2 for _,m in dsec[i:]): tail_start=t;break
t_run_le=lambda k: next((t for i,(t,n) in enumerate(dsec) if all(m<=k for _,m in dsec[i:])),None)
saved_10=gen_s-t_run_le(2) if t_run_le(2) is not None else None
saved_25=gen_s-t_run_le(5) if t_run_le(5) is not None else None
# 额外预填充估算
def secs(tok): return tok/rate if rate else None
n_inflight_tail=3   # 24条的最慢10%(取整上界)
est={}
for name,pool in (('all_turn_ctx',ctx_all),('long_traj_final_ctx',multi_final),('all_traj_final_ctx',final_ctx)):
    p50,p90=pct(pool,50),pct(pool,90)
    est[name]=dict(ctx_p50=p50,ctx_p90=p90,sec_p50=secs(p50),sec_p90=secs(p90))
out=dict(
 source=dict(tape=R+'tape-direct/.../rl-island-0.jsonl',analysis=R+'analysis.json',sglang_log=R+'launch.log'),
 sample_size=dict(rounds=1,trajectories=len(tr),note='只有 1 轮 24 条(6 题×4)，p90 仅 2–3 条支撑'),
 turns_per_traj=summ(turns),turns_dist=an['turns']['dist'],
 turn_input_ctx_tokens=summ(ctx_all),final_turn_ctx_tokens=summ(final_ctx),
 long_traj_ge5_turns=dict(n=len(multi),final_ctx=summ(multi_final)),
 prefill_log=dict(n_lines=len(pf),cached_token_sum=cached_sum,
   gen_window_new_tokens_sum=new_sum_gen,tape_turn_ctx_sum=tape_ctx_sum,
   ratio_log_over_tape=new_sum_gen/tape_ctx_sum,
   rate_tok_s_median_of_sane_lines=rate,rate_n=len(sane),
   rate_lower_bound_whole_gen_window=thr_lo,
   input_throughput_field_all=summ(allthr),
   note='input throughput 字段=间隔均值，1s时间戳下同秒批次出现荒谬大值(最高~1e6)，不可直接用'),
 timing=dict(generate_s=gen_s,train_s=train_s,
   publish_s=[s['seconds'] for s in spans if s['task']=='publish'],
   outer_sync_s=[s['seconds'] for s in spans if s['task']=='outer_sync'],
   round_total_s=ph[('finish',1,None)]-gen_start,
   rollout_cutoff_events=0,peak_running_req=peak,
   t_running_le2_from_gen_start=t_run_le(2),t_running_le5_from_gen_start=t_run_le(5)),
 saved_wait_s=dict(drop_slowest_10pct_proxy=saved_10,drop_slowest_25pct_proxy=saved_25,
   note='用 Decode 行 running-req<=k 的尾段时长近似；running-req 不含在工具等待中的轨迹，偏低估尾长'),
 extra_prefill=dict(rate_used_tok_s=rate,per_traj=est,
   tail10pct_inflight_n=n_inflight_tail,
   total_sec_tail_p50=n_inflight_tail*est['long_traj_final_ctx']['sec_p50'] if rate else None,
   total_sec_tail_p90=n_inflight_tail*est['long_traj_final_ctx']['sec_p90'] if rate else None,
   total_sec_all24_upper=24*est['all_turn_ctx']['sec_p50'] if rate else None),
)
json.dump(out,open(OUT,'w'),ensure_ascii=False,indent=1)
print(json.dumps(out,ensure_ascii=False,indent=1))
