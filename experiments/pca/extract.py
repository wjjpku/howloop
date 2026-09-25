from pathlib import Path
import sys,os,json,hashlib,time
import numpy as np,torch
B=Path('/data/wujiaju/n10_migration_20260923');R=Path(__file__).resolve().parent;sys.path.insert(0,str(B/'code'))
from reasoning_loop.paper2027_d8l6_s1 import load_controller,terminal_h6
from reasoning_loop.graph_path_depth_circuit import load_checkpoint,fixed_depth_batch
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
torch.set_num_threads(3);device=torch.device('cuda');cp=B/'backbones/local_control_L6_seed6/best.pt';assert sha(cp)=='faca3e18f01122fa395e0ba08fc953e840c87b5f4c82cfb61603dab294640291'
m,cfg,ck=load_checkpoint(cp,device);m.eval();m.requires_grad_(False);js={};hashes={}
for fit in [1,2]:
 for hop in [1,2]:
  f=B/f'local/A/hop{hop}_seed{fit}/best_controller.pt';j,z=load_controller(f,cp,device);j.eval();js[fit,hop]=j;hashes[f'{fit}_{hop}']=sha(f)
graphs=json.loads((R/'datasets.json').read_text())['confirmation'];assert len(graphs)==512
manifest={'state':'running','pid':os.getpid(),'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'checkpoint':str(cp),'checkpoint_sha256':sha(cp),'maps':hashes,'dataset_sha256':sha(R/'datasets.json'),'graphs':512,'starts':10,'code_sha256':sha(__file__),'states':'raw h6, J(h6), F(J(h6)); answer token before final norm and token mean; F(h6) baseline','all_samples':True};(R/'manifest.json').write_text(json.dumps(manifest,indent=2));out={};start=time.time()
def put(k,v):out.setdefault(k,[]).append(v.detach().cpu().numpy())
def state(k,h):
 put(k+'_answer',h[:,-1]);put(k+'_mean',h.mean(1));put(k+'_pred',logits_from_raw_state(m,h).argmax(-1))
with torch.inference_mode():
 for first in range(0,512,16):
  g=torch.tensor(graphs[first:first+16],device=device).repeat_interleave(10,0);u=torch.arange(10,device=device).repeat(len(g)//10);s=u.clone();inv=g.argsort(-1)
  for _ in range(cfg.max_depth):s=inv.gather(1,s[:,None])[:,0]
  tok,*_=fixed_depth_batch(cfg,len(g),device,path_positions=10,successors=g,start=s);h=terminal_h6(m,tok,cfg);assert cfg.max_loops==6
  state('raw',h);state('rawF',m.apply_loop(h,loop_index=6));put('graph_id',torch.arange(first,first+len(g)//10,device=device).repeat_interleave(10));put('current',u)
  for hop in [1,2]:
   u=g.gather(1,u[:,None])[:,0];put(f'target{hop}',u)
  for fit in [1,2]:
   for hop in [1,2]:
    a=js[fit,hop](h);state(f'fit{fit}_hop{hop}_J',a);state(f'fit{fit}_hop{hop}_JF',m.apply_loop(a,loop_index=6))
  print(json.dumps({'graphs':first+len(g)//10,'seconds':time.time()-start}),flush=True)
np.savez_compressed(R/'states.npz',**{k:np.concatenate(v) for k,v in out.items()});manifest.update(state='complete',elapsed=time.time()-start,peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,hidden_dim=int(h.shape[-1]),tokens=int(h.shape[1]));(R/'manifest.json').write_text(json.dumps(manifest,indent=2))
