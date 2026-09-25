import argparse,json,os,subprocess,sys,time,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1]; C=R/'code'
p=argparse.ArgumentParser();p.add_argument('--shard',type=int,choices=[0,1],required=True);a=p.parse_args()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((R/'MANIFEST.json').read_text()); jobs=[]
# Prioritize the main-paper cohort; no selection based on outcome.
for family in ['local_control','trajectory','continuation']:
 cfg=manifest['backbones'][family]
 for seed in cfg['seeds']:jobs.append((family,cfg['loops'],seed,cfg['depth_mode']))
for idx,(family,loops,seed,mode) in enumerate(jobs):
 if idx%2!=a.shard:continue
 out=R/'backbones'/f'{family}_L{loops}_seed{seed}'
 if (out/'summary.json').exists():continue
 if out.exists():raise RuntimeError(f'Incomplete run requires explicit diagnosis: {out}')
 cmd=[sys.executable,'-u','-m','reasoning_loop.paper2027_graph_g4_backbone','--out-dir',str(out),'--selection-lock',str(R/'locks/selection.pt'),'--final-test-lock',str(R/'locks/confirmation.pt'),'--extra-exclude-locks',*[str(R/'locks'/f'{x}.pt') for x in ['discovery','rings','smoke','donors']],'--seed',str(seed),'--loops',str(loops),'--depth-mode',mode,'--steps','20000','--eval-every','1000','--batch-size','512','--eval-batch-size','500','--amp']
 status={'status':'running','pid':os.getpid(),'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'family':family,'seed':seed,'command':cmd,'started':time.time(),'manifest_sha256':sha(R/'MANIFEST.json')}
 (R/f'worker_backbones_{a.shard}.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
 subprocess.run(cmd,cwd=C,check=True)
 print(json.dumps({'event':'backbone_complete','run':str(out),'best_sha256':sha(out/'best.pt')}),flush=True)
(R/f'worker_backbones_{a.shard}.json').write_text(json.dumps({'status':'complete','pid':os.getpid(),'gpu':os.environ['CUDA_VISIBLE_DEVICES']}))
