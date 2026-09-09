from pathlib import Path
import fcntl,hashlib,json,os,signal,sqlite3,sys,threading,time,urllib.request
BASE=Path('/mnt/lvm_data/sft_analysis/sft_baseline_20260908')
STAGE=BASE/'cot-code-v5-accuracy-bulk'
RUN=BASE/'cot-joint-accuracy-v4-bulk'
SOURCE=BASE/'cot-source-v4-projection-v2.jsonl'
SOURCE_SHA='ed833db681fd8fd1e3c2315d96503016e8753bf7b4036ed6fbda4f88981c8bed'
INDEX=BASE/'cot-bulk-v5-review-v3-thinking.sqlite3'
LOCK=Path('/mnt/lvm_data/sft_analysis/extracted/trace_labeling/runs/fleet_medium/inference.lock')
sys.path.insert(0,str(STAGE))
from cot_filler.core import canonical
from cot_filler.corpus_source import file_sha256
from cot_filler.corpus_worker import Journal,PrefixReviewProvider,_code_identity,run_corpus,now
from cot_filler.provider import OpenAICompatibleProvider
from cot_filler.derived_index import DerivedIndexJournal
from cot_filler.grounding_review_v4 import REVIEW_VERSION,response_format
from cot_filler.review_comparison import NoGeneration,import_candidates,snapshot_candidates,comparison_gate
from cot_filler.review_recovery import restrict_retry
STOP=threading.Event()

def save(name,value):
 p=RUN/name;fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'w') as f:f.write(canonical(value)+'\n')

def readurl(url):
 with urllib.request.urlopen(url,timeout=20) as r:return json.load(r)

def review_config(url):
 cfg=json.loads((BASE/'cot-review-router-v3-thinking32k.json').read_text())
 cfg.update(base_url=url,review_version=REVIEW_VERSION,max_output_tokens=49152,custom_params={'thinking_budget':32768},response_format=response_format(),require_strict_thinking_server=True)
 cfg.pop('strict_thinking_server_verified',None)
 return cfg

def identity(gen,rev):
 return {'generator_config':gen,'reviewer_config':rev,'implementation_sha256':_code_identity(),'supervisor_sha256':file_sha256(__file__),'activation_policy':'bulk-generation-only-until-independent-accuracy-and-all-six-live-strict-verification/v1'}

class NoReview:
 def review(self,*a,**kw):raise RuntimeError('Bulk approval/review is prohibited before accuracy and fleet activation')

def bulk(gen,rev):
 try:
  generator=OpenAICompatibleProvider(gen)
  old=sqlite3.connect(INDEX.as_uri()+'?mode=ro',uri=True)
  parent=json.loads(old.execute("SELECT value FROM meta WHERE key='identity'").fetchone()[0]);old.close()
  expected=parent['implementation_sha256']['corpus_worker.py']
  matches=[p for p in BASE.glob('cot-code-*/cot_filler/corpus_worker.py') if file_sha256(p)==expected]
  if not matches:raise ValueError('Pinned original index builder is unavailable')
  p=BASE/'cot-bulk-v5-review-v4-accuracy.sqlite3'
  with open(str(p)+'.worker.lock','a+') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   j=DerivedIndexJournal(p,SOURCE,SOURCE_SHA,{**identity(gen,rev),'generator_tokenizer':generator.tokenizer.identity},index_template=INDEX,template_worker_path=matches[0],expected_sources=14146,expected_gaps=643513)
   try:
    save('bulk-started.json',{'pid':os.getpid(),'journal':str(p),'source_sha256':SOURCE_SHA,'states_before':j.summary(),'workers':48,'token_budget':6291456,'max_gaps':643513,'mode':'generation_only','reviewer_qualification_pending':True,'shared_lock_owner':os.getpid(),'started':now()})
    result=run_corpus(j,generator,NoReview(),max_gaps=643513,workers=48,token_budget=6291456,retries=1,stop=STOP,generation_only=True,progress=lambda x:print(canonical({'stream':'bulk',**x}),flush=True))
    save('bulk-completed.json',result);print(canonical({'stream':'bulk',**result}),flush=True)
   finally:j.close()
 except Exception as exc:
  save('bulk-failed.json',{'type':type(exc).__name__,'detail':str(exc)[:300],'at':now()});print(canonical({'stream':'bulk','failed':type(exc).__name__}),flush=True)

def review_saved(name,original_name,count,ordinals,config):
 snapshot=snapshot_candidates(BASE/original_name,expected_count=count)
 provider=PrefixReviewProvider(config)
 p=BASE/name
 with open(str(p)+'.worker.lock','a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  ident={'comparison_version':'cot.accuracy-first-persisted-review/v1','original_journal':snapshot['original_journal'],'original_snapshot_sha256':snapshot['snapshot_sha256'],'selected_ordinals':ordinals,'reviewer_config':config,'reviewer_tokenizer':provider.tokenizer.identity,'implementation_sha256':_code_identity(),'supervisor_sha256':file_sha256(__file__),'live_server_verification_sha256':file_sha256(RUN/'n5-server-verified.json')}
  j=Journal(p,snapshot['original_identity']['source_path'],snapshot['original_identity']['source_sha256'],ident)
  try:
   import_candidates(j,snapshot)
   if ordinals:restrict_retry(j,ordinals)
   save(name+'-started.json',{'pid':os.getpid(),'journal':str(p),'selected_ordinals':ordinals or 'all','max_gaps':len(ordinals) or count,'thinking_budget':32768,'total_output_budget':49152,'shared_lock_owner':os.getpid(),'started':now()})
   result=run_corpus(j,NoGeneration(),provider,max_gaps=len(ordinals) or count,workers=8,token_budget=2097152,retries=1,stop=STOP,progress=lambda x:print(canonical({'stream':name,**x}),flush=True))
   if count==48:result.update(comparison_gate(j))
   result.update(automatic_bulk_approval=False)
   save(name+'-completed.json',result);print(canonical({'stream':name,**result}),flush=True)
  finally:j.close()

def main():
 for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:STOP.set())
 save('waiting-for-lock.json',{'pid':os.getpid(),'startticks':Path('/proc/self/stat').read_text().split()[21],'at':now()})
 with open(LOCK,'r+') as shared:
  fcntl.flock(shared,fcntl.LOCK_EX)
  if STOP.is_set():return
  save('lock-acquired.json',{'pid':os.getpid(),'startticks':Path('/proc/self/stat').read_text().split()[21],'lock':str(LOCK),'at':now()})
  server=readurl('http://127.0.0.1:33122/get_server_info');args=server.get('server_args',server)
  if args.get('enable_strict_thinking') is not True or args.get('served_model_name')!='DeepSeek-V4-Flash' or args.get('context_length')!=262144:raise ValueError('Direct n5 strict configuration mismatch')
  if args.get('model_path')!='/data/models/DeepSeek-V4-Flash-0731' or args.get('grammar_backend')!='xgrammar':raise ValueError('Direct n5 model/grammar identity mismatch')
  workers=readurl('http://127.0.0.1:30100/workers')['workers']
  good=[w for w in workers if w['is_healthy']]
  if len(good)!=6 or any('208.64.254.181' in w['url'] for w in good):raise ValueError('Expected exactly six inference engines, excluding n7')
  if any(w.get('load')!=0 for w in workers):raise ValueError('Unexpected existing inference workload')
  save('n5-server-verified.json',{'server_args_sha256':hashlib.sha256(canonical(args).encode()).hexdigest(),'strict_thinking_verified':True,'expected_n5_pid':4129424,'expected_n5_startticks':'33904940','direct_url':'http://127.0.0.1:33122','model_path':args['model_path'],'context_length':args['context_length'],'verified_at':now(),'fleet_review_not_yet_verified':True})
  gen=json.loads((BASE/'cot-generation-router-v5-bulk.json').read_text());fleet=review_config('http://127.0.0.1:30100/v1');direct=review_config('http://127.0.0.1:33122/v1')
  save('generator-config.json',gen);save('fleet-review-config.json',fleet);save('direct-review-config.json',direct)
  thread=threading.Thread(target=bulk,args=(gen,fleet),name='bulk-generation-coordinator');thread.start()
  try:
   review_saved('cot-review-v5-v4-strict32k-48.sqlite3','cot-pilot-v5-diverse48.sqlite3',48,[],direct)
   if not STOP.is_set():review_saved('cot-review-v5-v4-strict32k-regression2.sqlite3','cot-pilot-v5-diverse128-thinking32k.sqlite3',128,[13,64],direct)
  except Exception as exc:
   save('bounded-review-failed.json',{'type':type(exc).__name__,'detail':str(exc)[:300],'at':now()});print(canonical({'stream':'bounded-review','failed':type(exc).__name__}),flush=True)
  thread.join()

if __name__=='__main__':
 try:main()
 except Exception as exc:
  save('supervisor-failed.json',{'type':type(exc).__name__,'detail':str(exc)[:300],'at':now()});raise
