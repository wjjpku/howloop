import os,sys,json,time,subprocess,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
BASE=R.parent
shard=int(sys.argv[1]); gpu=int(os.environ['CUDA_VISIBLE_DEVICES'])
seeds=[2,3,4,5,7,9,10,11][shard::2]
status=R/f'worker_{shard}.json'
def write(x):
 x.update(time=time.time(),worker_pid=os.getpid(),physical_gpu=gpu);status.write_text(json.dumps(x,indent=2));print(json.dumps(x),flush=True)
def run(cmd,seed,phase):
 free,total=map(int,subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(','))
 if free < 40960:raise RuntimeError('Insufficient free memory; do not launch')
 p=subprocess.Popen(cmd,cwd=R/'code',start_new_session=True)
 write(dict(state='running',seed=seed,phase=phase,child_pid=p.pid,command=cmd))
 while p.poll() is None:
  time.sleep(30)
  free,total=map(int,subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(','))
  with (R/f'resources_{shard}.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),gpu=gpu,free_mib=free))+'\n')
  if free<max(16384,int(total*.2)):
   p.terminate();raise RuntimeError('Stopped own child: GPU memory reserve breached')
 if p.returncode:raise RuntimeError(f'{phase} failed: {p.returncode}')
try:
 for seed in seeds:
  out=R/'backbones'/f'trajectory_L8_seed{seed}'
  cmd=[sys.executable,'-u','-m','reasoning_loop.paper2027_graph_g4_backbone','--out-dir',str(out),'--selection-lock',str(BASE/'locks/selection.pt'),'--final-test-lock',str(BASE/'locks/confirmation.pt'),'--extra-exclude-locks',*[str(BASE/'locks'/f'{x}.pt') for x in ['discovery','rings','smoke','donors']],'--seed',str(seed),'--loops','8','--depth-mode','uniform','--steps','20000','--eval-every','1000','--batch-size','512','--eval-batch-size','500','--amp']
  run(cmd,seed,'training')
  run([sys.executable,'-u','evaluate_native.py','--checkpoint',str(out/'best.pt'),'--out',str(R/'trajectories'/f'L8_seed{seed}')],seed,'trajectory_evaluation')
 write(dict(state='complete',seeds=seeds))
except Exception as e:
 write(dict(state='failed',error=repr(e)));raise
