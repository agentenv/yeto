from pathlib import Path
import fcntl,json,os,signal,sqlite3,sys,threading
BASE=Path('/mnt/lvm_data/sft_analysis/sft_baseline_20260908')
RUN=BASE/'cot-generation-only-20260909'
STAGE=BASE/'cot-code-v5-accuracy-bulk'
JOURNAL=BASE/'cot-bulk-v5-review-v4-accuracy.sqlite3'
LOCK=Path('/mnt/lvm_data/sft_analysis/extracted/trace_labeling/runs/fleet_medium/inference.lock')
sys.path.insert(0,str(STAGE))
from cot_filler.core import canonical
from cot_filler.corpus_worker import Journal,_code_identity,run_corpus,now
from cot_filler.corpus_source import file_sha256
from cot_filler.provider import OpenAICompatibleProvider
STOP=threading.Event()
class NoReview:
 def review(self,*a,**k):raise RuntimeError('Reviewer is explicitly paused by the user')
def save(name,value):
 fd=os.open(RUN/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 with os.fdopen(fd,'w') as f:f.write(canonical(value)+'\n')
def main():
 for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:STOP.set())
 save('waiting-for-reservation.json',{'pid':os.getpid(),'startticks':Path('/proc/self/stat').read_text().split()[21],'at':now()})
 with open(str(JOURNAL)+'.worker.lock','a+') as own,open(LOCK,'r+') as shared:
  fcntl.flock(own,fcntl.LOCK_EX);fcntl.flock(shared,fcntl.LOCK_EX)
  if STOP.is_set():return
  old=sqlite3.connect(JOURNAL.as_uri()+'?mode=ro',uri=True)
  identity=json.loads(old.execute("SELECT value FROM meta WHERE key='identity'").fetchone()[0])
  before=dict(old.execute('SELECT state,count(*) FROM gaps GROUP BY state'))
  uncommitted=[r[0] for r in old.execute("SELECT id FROM gaps WHERE state='generating'")]
  counts={'candidates':old.execute('SELECT count(*) FROM candidates').fetchone()[0],'responses':old.execute('SELECT count(*) FROM responses').fetchone()[0],'reviews':old.execute('SELECT count(*) FROM reviews').fetchone()[0]};old.close()
  if identity['implementation_sha256']!=_code_identity():raise ValueError('Pinned generator/worker identity changed')
  save('reservation-acquired.json',{'pid':os.getpid(),'startticks':Path('/proc/self/stat').read_text().split()[21],'at':now(),'states_before_resume':before,'counts_before_resume':counts,'only_uncommitted_generations_requeued':uncommitted,'reviewer_paused_by_user':True,'supervisor_sha256':file_sha256(__file__)})
  provider=OpenAICompatibleProvider(identity['generator_config'])
  journal=Journal(JOURNAL,identity['source_path'],identity['source_sha256'],identity)
  try:
   save('generation-started.json',{'pid':os.getpid(),'journal':str(JOURNAL),'states':journal.summary(),'started':now(),'workers':48,'token_budget':6291456,'max_gaps':643513,'mode':'generation_only','reviewer_paused_by_user':True})
   result=run_corpus(journal,provider,NoReview(),max_gaps=643513,workers=48,token_budget=6291456,retries=1,stop=STOP,generation_only=True,progress=lambda x:print(canonical(x),flush=True))
   save('completed.json',result);print(canonical(result),flush=True)
  finally:journal.close()
if __name__=='__main__':
 try:main()
 except Exception as exc:
  save('failed.json',{'type':type(exc).__name__,'detail':str(exc)[:300],'at':now()});raise
