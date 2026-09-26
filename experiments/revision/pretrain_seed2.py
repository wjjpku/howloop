"""Train one pending matched pair in an isolated directory, then install it only if the main queue has not started it."""
import os,sys,json,time,subprocess,hashlib
from pathlib import Path
R=Path('/data/paperexperiment/reviewer_revision_20260926');S=R/'matched';W=S/'prefetch_seed2';W.mkdir(exist_ok=True)
(W/'manifest.json').write_text(json.dumps({'seed':2,'purpose':'same registered backbone protocol; prefetch only, map fitting remains in original GPU6 queue','pid':os.getpid(),'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'started':time.time(),'code_sha256':hashlib.sha256((R/'code/matched_train.py').read_bytes()).hexdigest()},indent=2))
for regime in ['final','stepwise']:
 dest=W/f'{regime}_seed2'
 if not (dest/'summary.json').exists():
  args=[sys.executable,'-u',str(R/'code/matched_train.py'),'--out-dir',str(dest),'--selection-lock',str(S/'selection.pt'),'--final-test-lock',str(S/'test.pt'),'--seed','2','--loops','8','--depth-mode','fixed','--steps','20000','--batch-size','512','--eval-every','1000','--amp','--supervision',regime]
  print(json.dumps({'event':'prefetch_train','regime':regime,'command':args}),flush=True);subprocess.run(args,check=True)
aa=json.load(open(W/'final_seed2/summary.json'));bb=json.load(open(W/'stepwise_seed2/summary.json'))
assert aa['initial_state_sha256']==bb['initial_state_sha256'] and aa['training_tokens_sha256']==bb['training_tokens_sha256']
installed=[]
for regime in ['final','stepwise']:
 src=W/f'{regime}_seed2';dst=S/'runs'/src.name
 if not dst.exists():src.rename(dst);installed.append(str(dst))
 else:print('Main queue already owns '+str(dst)+'; keeping prefetch separate',flush=True)
(W/'complete.json').write_text(json.dumps({'paired_hashes_match':True,'installed':installed,'time':time.time()},indent=2));print('Prefetch complete',flush=True)
