import os,sys,time,json,subprocess,hashlib
from pathlib import Path
import numpy as np
B=Path('/data/paperexperiment/n10_migration_20260923');R=Path('/data/paperexperiment/reviewer_revision_20260926/d8l6_readout_extension');R.mkdir(parents=True,exist_ok=True)
gpu=os.environ['CUDA_VISIBLE_DEVICES'];seeds=list(range(8,20))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest={'seeds':seeds,'gpu':gpu,'pid':os.getpid(),'started':time.time(),'base_protocol':'paper2027.graph.n10.disjoint_backbone.v1','loops':6,'depth_mode':'uniform','steps':20000,'checkpoint':'best.pt selected only by original endpoint validation','selection_motivation':'Readout-first search for variable semantic increments, hypothesizing more accessible one-hop computation than regular two-hop backbones. No mechanism outcomes used for readout screening.','inventory':'Retain every seed including regular, ambiguous, or unsuccessful runs. Training finishes full fixed seed list independent of candidate count.','training_source_sha256':sha(B/'code/reasoning_loop/paper2027_graph_g4_backbone.py'),'evaluation_source_sha256':sha(B/'code/evaluate_native.py')}
(R/'manifest.json').write_text(json.dumps(manifest,indent=2))
def run(cmd,seed,phase):
 p=subprocess.Popen(cmd,cwd=B/'code');(R/'status.json').write_text(json.dumps({'state':'running','phase':phase,'seed':seed,'pid':p.pid,'worker_pid':os.getpid(),'gpu':gpu,'command':cmd,'time':time.time()},indent=2));print(json.dumps({'event':phase,'seed':seed,'pid':p.pid}),flush=True)
 while p.poll() is None:
  time.sleep(15)
  free,total=map(int,subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(','))
  with (R/'resources.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'free_mib':free,'gpu':gpu,'seed':seed,'phase':phase})+'\n')
  if free<max(16384,int(total*.2)):
   p.terminate();p.wait();raise RuntimeError('Own job stopped: reserve breached')
 if p.returncode:raise RuntimeError(f'{phase} failed {p.returncode}')
rows=[]
for seed in seeds:
 out=R/'backbones'/f'L6_seed{seed}';traj=R/'trajectories'/f'L6_seed{seed}'
 if not (out/'summary.json').exists():
  if out.exists():raise RuntimeError(f'Incomplete output needs diagnosis: {out}')
  cmd=[sys.executable,'-u','-m','reasoning_loop.paper2027_graph_g4_backbone','--out-dir',str(out),'--selection-lock',str(B/'locks/selection.pt'),'--final-test-lock',str(B/'locks/confirmation.pt'),'--extra-exclude-locks',*[str(B/'locks'/f'{k}.pt') for k in ['discovery','rings','smoke','donors']],'--seed',str(seed),'--loops','6','--depth-mode','uniform','--steps','20000','--eval-every','1000','--batch-size','512','--eval-batch-size','500','--amp']
  run(cmd,seed,'training')
 if not (traj/'summary.json').exists():run([sys.executable,'-u',str(B/'code/evaluate_native.py'),'--checkpoint',str(out/'best.pt'),'--out',str(traj)],seed,'readout')
 s=json.loads((traj/'summary.json').read_text());a=np.array(s['splits']['rings']['category_match']).T;modes=[]
 for v in a:
  k=np.flatnonzero(np.isclose(v,v.max(),rtol=0,atol=1e-8));modes.append(int(k[0]) if len(k)==1 else None)
 increments=[(modes[t]-modes[t-1])%10 if modes[t] is not None and modes[t-1] is not None else None for t in range(1,7)]
 progress=[i for i in increments if i not in [None,0]]
 rows.append({'seed':seed,'modes_0_to_16':modes,'increments_1_to_6':increments,'nonzero_increment_types':sorted(set(progress)),'variable_nonzero_increments':len(set(progress))>1,'has_one_and_two':1 in progress and 2 in progress,'mode_support_0_to_16':a.max(1).tolist(),'endpoint_accuracy_at_6':s['splits']['rings']['endpoint_accuracy_by_call'][6],'checkpoint_sha256':s['checkpoint_sha256']})
 (R/'readout_inventory.json').write_text(json.dumps({'scope':'Unique cohort modal increments modulo ten, ties excluded; not proof of internal operations. Retention zero excluded from variable-progress flag. Flags describe prospective screen, not historical selection thresholds.','rows':rows},indent=2));print(json.dumps({'event':'readout_complete',**rows[-1]}),flush=True)
(R/'status.json').write_text(json.dumps({'state':'complete','seeds':seeds,'time':time.time()}))
