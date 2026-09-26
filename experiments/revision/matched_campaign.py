import os,sys,json,time,subprocess,hashlib
from pathlib import Path
R=Path('/data/paperexperiment/reviewer_revision_20260926');B=Path('/data/paperexperiment/n10_migration_20260923');sys.path.insert(0,str(B/'code'))
import torch
from reasoning_loop.paper2027_graph_g4_protocol import create_unique_lock
S=R/'matched';S.mkdir(parents=True,exist_ok=True)
a=create_unique_lock(path=S/'selection.pt',node_count=10,max_depth=8,permutations=512,seed=2026092611,role='selection')
b=create_unique_lock(path=S/'test.pt',node_count=10,max_depth=8,permutations=512,seed=2026092612,role='final_test',excluded_codes=a['permutation_codes'])
smoke='--smoke' in sys.argv
out=S/('smoke' if smoke else 'runs');out.mkdir(exist_ok=True)
manifest={'pid':os.getpid(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'started':time.time(),'paired_seeds':list(range(5)),'steps':5 if smoke else 20000,'batch_size':512,'losses':['CE8','CE8 + mean(CE1..CE7)'],'fixed_depth':8,'frozen_map_budget':8000,'map_fit_seeds':[1,2],'map_targets':['stay','one','two'],'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'code').glob('*.py')}}
(out/('manifest_gpu'+os.environ['CUDA_VISIBLE_DEVICES']+'.json')).write_text(json.dumps(manifest,indent=2))
selected_seeds=[0] if smoke else ([int(x) for x in sys.argv[sys.argv.index('--seeds')+1:]] if '--seeds' in sys.argv else list(range(5)))
for seed in selected_seeds:
 for regime in ['final','stepwise']:
  dest=out/f'{regime}_seed{seed}'
  if (dest/'summary.json').exists():continue
  args=[sys.executable,'-u',str(R/'code/matched_train.py'),'--out-dir',str(dest),'--selection-lock',str(S/'selection.pt'),'--final-test-lock',str(S/'test.pt'),'--seed',str(seed),'--loops','8','--depth-mode','fixed','--steps',str(manifest['steps']),'--batch-size','512','--eval-every','1000','--amp','--supervision',regime]
  print(json.dumps({'event':'start_backbone','seed':seed,'regime':regime,'command':args}),flush=True);subprocess.run(args,check=True)
 aa=json.load(open(out/f'final_seed{seed}/summary.json'));bb=json.load(open(out/f'stepwise_seed{seed}/summary.json'))
 assert aa['initial_state_sha256']==bb['initial_state_sha256'],'initialization mismatch'
 assert aa['training_tokens_sha256']==bb['training_tokens_sha256'],'sample-stream mismatch'
 (out/f'pair{seed}_validated.json').write_text(json.dumps({'initial_state_sha256':aa['initial_state_sha256'],'training_tokens_sha256':aa['training_tokens_sha256'],'matched':True},indent=2))
 if not smoke:
  for regime in ['final','stepwise']:
   subprocess.run([sys.executable,'-u',str(R/'code/matched_maps.py'),'--backbone',str(out/f'{regime}_seed{seed}/final.pt'),'--output',str(out/f'{regime}_seed{seed}/maps')],check=True)
(out/('complete_gpu'+os.environ['CUDA_VISIBLE_DEVICES']+'.json')).write_text(json.dumps({'complete':True,'time':time.time()}))
