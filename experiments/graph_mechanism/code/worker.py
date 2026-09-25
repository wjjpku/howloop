import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];CODE=ROOT/'code'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2))
def verify_code():
 for r in json.loads((ROOT/'code_lock.json').read_text()):assert sha(ROOT/r['path'])==r['sha256'],r
ap=argparse.ArgumentParser();ap.add_argument('--names',nargs='+',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
env=os.environ.copy();env['PYTHONPATH']=str(CODE);env['OMP_NUM_THREADS']='3';env['MKL_NUM_THREADS']='3';env['OPENBLAS_NUM_THREADS']='1'
lock=json.loads((ROOT/'backbones.json').read_text());worker=ROOT/f'worker_gpu{env["CUDA_VISIBLE_DEVICES"]}{"_smoke" if a.smoke else ""}.json'
for name in a.names:
 d=next(x for x in lock if x['name']==name);assert sha(Path(d['checkpoint']))==d['checkpoint_sha256'];out=ROOT/('smoke' if a.smoke else 'runs')/name;out.mkdir(parents=True,exist_ok=False)
 for hop,seed in ([(1,1),(1,1)] if a.smoke else [(h,s) for h in (1,2) for s in (1,2)]):
  suffix=f'hop{hop}_seed{seed}'
  if a.smoke:suffix+=f'_repeat{len(list(out.iterdir()))}'
  sub=out/suffix;verify_code()
  cmd=[sys.executable,'-u',str(CODE/'reasoning_loop/paper2027_d8l6_s1.py'),'train','--checkpoint',d['checkpoint'],'--out-dir',str(sub),'--target-hop',str(hop),'--seed',str(seed),'--rank','48','--updates',str(2 if a.smoke else 8000),'--batch-size','128','--learning-rate','0.0001','--grad-clip','1','--validation-every',str(2 if a.smoke else 400),'--validation-seed',str(seed+10000),'--validation-batches',str(1 if a.smoke else 8),'--validation-batch-size','256','--device','cuda']
  write(worker,{'status':'running','worker_pid':os.getpid(),'gpu':env['CUDA_VISIBLE_DEVICES'],'backbone':name,'fit':suffix,'command':cmd,'protocol_sha256':sha(ROOT/'PROTOCOL.md')});print(json.dumps({'event':'launch','backbone':name,'fit':suffix,'command':cmd}),flush=True)
  subprocess.run(cmd,env=env,check=True)
 if not a.smoke:
  verify_code();subprocess.run([sys.executable,'-u',str(CODE/'export_validate.py'),'--name',name],env=env,check=True)
  heads=dict(A=0,B=1,C=0,D=2,E=3)
  target=out/'controllers.pt';verify_code()
  cmd=[sys.executable,'-u',str(CODE/'evaluate_battery.py'),'--looplus',str(CODE),'--checkpoint',d['checkpoint'],'--checkpoint-sha256',d['checkpoint_sha256'],'--checkpoint-step',str(d['checkpoint_step']),'--controllers',str(target),'--controller-sha256',sha(target),'--datasets',str(ROOT/'datasets.json'),'--family','unified_pure_ce','--backbone-name',name,'--head',str(heads[name]),'--panel','confirmation','--seeds','1','2','--out',str(out/'evaluation')]
  write(worker,{'status':'evaluating','worker_pid':os.getpid(),'gpu':env['CUDA_VISIBLE_DEVICES'],'backbone':name,'command':cmd});subprocess.run(cmd,env=env,check=True)
  write(out/'complete.json',{'status':'complete','backbone':name,'controller_sha256':sha(target),'protocol_sha256':sha(ROOT/'PROTOCOL.md')})
write(worker,{'status':'complete','worker_pid':os.getpid(),'gpu':env['CUDA_VISIBLE_DEVICES'],'names':a.names})
