"""Sequential stage executor; does not launch until its own backbone queue ends."""
import argparse,json,os,subprocess,sys,time,signal
from pathlib import Path
R=Path(__file__).resolve().parents[1];C=R/'code'
p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);a=p.parse_args();gpu=os.environ['CUDA_VISIBLE_DEVICES'];status=R/f'after_backbones_{a.shard}.json'
def write(x):status.write_text(json.dumps(x,indent=2))
def memory():
 values=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(',');return [int(x) for x in values]
def guarded(cmd,label):
 free,total=memory();reserve=max(16384,int(.2*total))
 if free<reserve+8192:raise RuntimeError('insufficient memory headroom for phase')
 proc=subprocess.Popen(list(map(str,cmd)),cwd=C,start_new_session=True)
 write(dict(status='running',phase=label,pid=os.getpid(),child_pid=proc.pid,gpu=gpu,command=list(map(str,cmd))))
 while proc.poll() is None:
  free,total=memory()
  if free<reserve:
   os.killpg(proc.pid,signal.SIGTERM)
   write(dict(status='stopped_resource_guard',phase=label,child_pid=proc.pid,free_mib=free));raise RuntimeError('GPU reserve breached; stopped only own stage process group')
  print(json.dumps(dict(event='resource_check',phase=label,free_mib=free,time=time.time())),flush=True);time.sleep(30)
 if proc.returncode:raise RuntimeError(f'{label} failed with {proc.returncode}')
try:
 while True:
  f=R/f'worker_backbones_{a.shard}.json'
  if f.exists() and json.loads(f.read_text()).get('status')=='complete':break
  write(dict(status='waiting_for_backbones',pid=os.getpid(),gpu=gpu));time.sleep(30)
 # The first controller workload is unmeasured on N10: wait for an empty card,
 # then perform the full-batch smoke. Do not co-locate an unmeasured workload.
 while subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
  write(dict(status='waiting_for_empty_gpu_for_controller_smoke',pid=os.getpid(),gpu=gpu));time.sleep(30)
 py=sys.executable
 # The continuation smoke is the longest unroll and therefore precedes all J work.
 smoke=R/f'smoke_g3_gpu{gpu}'
 if not (smoke/'summary.json').exists():
  guarded([py,'-u','-m','reasoning_loop.paper2027_graph_g3_controller','train','--checkpoint',R/'smoke_L8/best.pt','--out-dir',smoke,'--seed',1,'--updates',2,'--batch-size',128,'--validation-every',2,'--validation-seed',10001,'--validation-batch-size',250,'--validation-lock',R/'locks/selection.pt','--train-exclude-locks',*sorted((R/'locks').glob('*.pt'))],'continuation_gpu_smoke')
 for script,label in [('worker_local.py','local_control_and_mechanism'),('restricted.py','restricted_controllers'),('worker_continuation.py','continuation')]:
  guarded([py,'-u',script,'--shard',a.shard],label)
 # Native reads of all four descriptive D8L8 models.
 for i,seed in enumerate([0,1,6,8]):
  if i%2==a.shard:guarded([py,'evaluate_native.py','--checkpoint',R/'backbones'/f'trajectory_L8_seed{seed}'/'best.pt','--out',R/'trajectories'/f'L8_seed{seed}'],'native_trajectory')
 if a.shard == 0:
  guarded([py,'-u','worker_sensitivity.py'],'regularization_sensitivity')
 guarded([py,'-u','worker_supervision.py','--shard',a.shard],'matched_supervision')
 if a.shard == 0:
  guarded([py,'-u','single_h64.py'],'single_backbone_h64')
 write(dict(status='experiments_complete',gpu=gpu,pid=os.getpid(),remaining=['aggregate raw predictions','review results','paper figures and compilation']))
except Exception as exc:
 write(dict(status="failed",pid=os.getpid(),gpu=gpu,error=repr(exc)))
 raise
