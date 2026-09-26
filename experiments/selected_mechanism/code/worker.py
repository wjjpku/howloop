import os,sys,json,time,subprocess,hashlib,gzip,argparse
from pathlib import Path
import torch
P=Path(__file__).resolve().parents[1];B=Path('/data/paperexperiment/n10_migration_20260923');E=Path('/data/paperexperiment/reviewer_revision_20260926/d8l6_readout_extension');C=P/'code'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2))
ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);a=ap.parse_args();gpu=os.environ['CUDA_VISIBLE_DEVICES'];torch.set_num_threads(4)
status=P/f'status_{a.shard}.json'
def run(cmd,seed,phase):
 cmd=list(map(str,cmd));print(json.dumps({'event':'launch','seed':seed,'phase':phase,'command':cmd}),flush=True);p=subprocess.Popen(cmd,cwd=B/'code');save(status,dict(state='running',seed=seed,phase=phase,pid=p.pid,worker_pid=os.getpid(),gpu=gpu,time=time.time(),command=cmd))
 while p.poll() is None:
  time.sleep(30)
  free,total=map(int,subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(','))
  with (P/f'resources_{a.shard}.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),gpu=gpu,free_mib=free,seed=seed,phase=phase))+'\n')
  if free<max(16384,total*.2):p.terminate();p.wait();raise RuntimeError('Reserve breached; stopped own subprocess')
 if p.returncode:raise RuntimeError(f'{phase} exited {p.returncode}')
try:
 for seed in json.loads((P/'cohort.json').read_text())['seeds']:
  if (seed-8)%2!=a.shard:continue
  out=P/'runs'/f'seed{seed}';cp=E/'backbones'/f'L6_seed{seed}'/'best.pt';out.mkdir(parents=True,exist_ok=True)
  if (out/'complete.json').exists():continue
  while not (cp.parent/'summary.json').exists():
   save(status,dict(state='waiting_for_backbone',seed=seed,worker_pid=os.getpid(),gpu=gpu,time=time.time()));time.sleep(30)
  summary=json.loads((cp.parent/'summary.json').read_text());assert sha(B/'locks/donors.pt') in summary['extra_excluded_locks'].values()
  step=torch.load(cp,map_location='cpu',weights_only=False)['step']
  for hop in [1,2]:
   for fit in [1,2]:
    sub=out/f'hop{hop}_seed{fit}'
    if not (sub/'summary.json').exists():
     run([sys.executable,'-u','-m','reasoning_loop.paper2027_d8l6_s1','train','--checkpoint',cp,'--out-dir',sub,'--target-hop',hop,'--seed',fit,'--rank',48,'--validation-seed',10000+fit,'--validation-lock',B/'locks/selection.pt','--train-exclude-locks',*sorted((B/'locks').glob('*.pt')),'--validation-batch-size',500],seed,f'train_hop{hop}_fit{fit}')
  if not (out/'export_validation.json').exists():run([sys.executable,'-u',B/'code/export_maps.py','--checkpoint',cp,'--runs',out],seed,'export_validate')
  def battery(head,panel):
   dest=out/f'head{head}';folder=dest/panel/panel
   if not (folder/'manifest.json').exists():run([sys.executable,'-u',C/'evaluate_battery.py','--looplus',B/'code','--checkpoint',cp,'--checkpoint-sha256',sha(cp),'--checkpoint-step',step,'--controllers',out/'controllers.pt','--controller-sha256',sha(out/'controllers.pt'),'--datasets',P/f'{panel}_datasets.json','--family','D8L6_section5_extension','--backbone-name',f'seed{seed}','--head',head,'--panel',panel,'--seeds',1,2,'--out',dest/panel],seed,f'{panel}_head{head}')
   return dest/panel/panel
  scores=[]
  for head in range(4):
   folder=battery(head,'discovery');totals={fit:[0,0] for fit in [1,2]}
   with gzip.open(folder/'events.jsonl.gz','rt') as f:
    for line in f:
     r=json.loads(line)
     if r['kind']=='cross_graph_address' and r['receiver']=='J_one' and r['condition']==f'J_one_pat_H{head}':
      t=totals[r['controller_seed']];t[0]+=sum(int(ok and y==z) for ok,y,z in zip(r['eligible'],r['prediction'],r['target']));t[1]+=sum(r['eligible'])
   score=sum(k/n for k,n in totals.values())/2 if all(n for k,n in totals.values()) else -1
   scores.append(dict(head=head,score=score,counts=totals))
  head=max(scores,key=lambda r:(r['score'],-r['head']))['head'];save(out/'head_selection.json',dict(locked_head=head,rule='two-fit mean eligible cross-graph pattern accuracy on discovery; absent eligibility=-1; tie smallest head',scores=scores))
  for h in range(4):battery(h,'confirmation')
  if not (P/'exchange'/f'seed{seed}_fit2.json').exists():run([sys.executable,'-u',C/'target_exchange.py','--models',str(seed)],seed,'target_exchange')
  save(out/'complete.json',dict(state='complete',seed=seed,checkpoint_sha256=sha(cp),controllers_sha256=sha(out/'controllers.pt'),head=head,time=time.time()))
 save(status,dict(state='complete',gpu=gpu,time=time.time()))
except Exception as e:
 save(status,dict(state='failed',error=str(e),gpu=gpu,time=time.time()));raise
