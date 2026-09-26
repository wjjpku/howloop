import argparse,sys,os,json,time,hashlib,copy
from pathlib import Path
import torch,numpy as np
import torch.nn.functional as F
B=Path('/data/paperexperiment/n10_migration_20260923');sys.path.insert(0,str(B/'code'))
from reasoning_loop.graph_path_depth_circuit import load_checkpoint,fixed_depth_batch
from reasoning_loop.paper2027_graph_g4_protocol import permutation_codes
R=Path('/data/paperexperiment/reviewer_revision_20260926/matched')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Map(torch.nn.Module):
 def __init__(self):
  super().__init__();self.D=torch.nn.Parameter(torch.ones(256));self.A=torch.nn.Parameter(torch.randn(256,48)*.02);self.B=torch.nn.Parameter(torch.zeros(48,256));self.b=torch.nn.Parameter(torch.zeros(256))
 def forward(self,h):return h*self.D+(h@self.A)@self.B+self.b

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--backbone',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);a.output.mkdir(parents=True,exist_ok=True)
 m,cfg,_=load_checkpoint(a.backbone,torch.device('cuda'));m.eval().requires_grad_(False);versions=[p._version for p in m.parameters()];assert cfg.max_loops==8
 selection=torch.load(R/'selection.pt',weights_only=False)['successors'];test=torch.load(R/'test.pt',weights_only=False)['successors'];forbidden=set(permutation_codes(torch.cat([selection,test])).tolist());gen=torch.Generator().manual_seed(2026092613);rows=[];seen=set(forbidden)
 while len(rows)<2048:
  gs=torch.rand(4096,10,generator=gen).argsort(-1)
  for g,c in zip(gs,permutation_codes(gs).tolist()):
   if c not in seen:rows.append(g);seen.add(c)
   if len(rows)==2048:break
 train=torch.stack(rows);assert not set(permutation_codes(train).tolist())&forbidden
 np.savez_compressed(a.output/'graphs.npz',train=train.numpy(),selection=selection.numpy(),test=test.numpy())
 def read(h):return m.unembed(m.ln_final(h[:,-1]))[:,:10]
 @torch.no_grad()
 def cache(gs):
  hs=[];ys=[];native=[]
  for off in range(0,len(gs),16):
   g=gs[off:off+16].to('cuda').repeat_interleave(10,0);start=torch.arange(10,device='cuda').repeat(len(g)//10);tok,lab,*_=fixed_depth_batch(cfg,len(g),torch.device('cuda'),path_positions=10,successors=g,start=start);h=m.token_embed(tok)+m.pos_embed[None]
   for t in range(8):h=m.apply_loop(h,loop_index=t)
   hs.append(h.cpu());ys.append(lab[:,7:10].cpu());native.append(read(m.apply_loop(h,loop_index=8)).argmax(-1).cpu())
  return torch.cat(hs),torch.cat(ys),torch.cat(native)
 pools={k:cache(g[:2] if a.smoke else g) for k,g in [('train',train),('selection',selection),('test',test)]};print(json.dumps({'event':'cached_states','backbone':str(a.backbone),'counts':{k:len(v[0]) for k,v in pools.items()}}),flush=True)
 def mask(y):return (y[:,0]!=y[:,1])&(y[:,0]!=y[:,2])&(y[:,1]!=y[:,2])
 @torch.no_grad()
 def evaluate(j,pool):
  h,y,raw=pool;pred=[];pre=[]
  for off in range(0,len(h),160):
   x=j(h[off:off+160].to('cuda'));pre.append(read(x).argmax(-1).cpu());pred.append(read(m.apply_loop(x,loop_index=8)).argmax(-1).cpu())
  return torch.cat(pred),torch.cat(pre)
 for fit in [1,2]:
  for hop,name in enumerate(['stay','one','two']):
   dest=a.output/f'{name}_fit{fit}.json'
   if dest.exists():continue
   torch.manual_seed(1000+fit);torch.cuda.manual_seed_all(1000+fit);j=Map().cuda();opt=torch.optim.AdamW(j.parameters(),lr=1e-4,weight_decay=0);h,y,_=pools['train'];steps=5 if a.smoke else 8000;best=-1;beststate=None;history=[];tic=time.time()
   with torch.no_grad():assert torch.equal(j(h[:2].cuda()),h[:2].cuda())
   for step in range(1,steps+1):
    idx=torch.randint(0,len(h),(128,));x=h[idx].cuda();target=y[idx,hop].cuda();opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=F.cross_entropy(read(m.apply_loop(j(x),loop_index=8)),target)
    loss.backward();torch.nn.utils.clip_grad_norm_(j.parameters(),1.);opt.step()
    if step%400==0 or step==steps:
     pred,_=evaluate(j,pools['selection']);mm=mask(pools['selection'][1]);val=float((pred[mm]==pools['selection'][1][mm,hop]).float().mean());history.append({'step':step,'selection':val,'loss':float(loss.detach()),'elapsed':time.time()-tic})
     if val>best:best=val;beststate=copy.deepcopy(j.state_dict());beststep=step
     print(json.dumps({'event':'fit_map','target':name,'fit':fit,**history[-1]}),flush=True)
     if torch.cuda.mem_get_info()[0]<16*2**30:raise RuntimeError('GPU reserve breached')
   j.load_state_dict(beststate);pred,pre=evaluate(j,pools['test']);hh,yy,raw=pools['test'];mm=mask(yy);np.savez_compressed(dest.with_suffix('.npz'),pred=pred.numpy(),pre=pre.numpy(),native=raw.numpy(),labels=yy.numpy(),distinct=mm.numpy());torch.save(beststate,dest.with_suffix('.pt'))
   assert versions==[p._version for p in m.parameters()] and all(p.grad is None for p in m.parameters())
   dest.write_text(json.dumps({'target':name,'fit':fit,'best_step':beststep,'selection_accuracy':best,'test_accuracy_all':float((pred==yy[:,hop]).float().mean()),'test_accuracy_distinct':float((pred[mm]==yy[mm,hop]).float().mean()),'test_n_distinct':int(mm.sum()),'backbone_sha256':sha(a.backbone),'map_sha256':sha(dest.with_suffix('.pt')),'code_sha256':sha(__file__),'graph_sha256':sha(a.output/'graphs.npz'),'history':history,'weights_frozen':True,'pid':os.getpid(),'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'peak_mib':torch.cuda.max_memory_allocated()/2**20},indent=2))
   if a.smoke:return
 (a.output/'complete.json').write_text(json.dumps({'complete':True,'time':time.time()}))
if __name__=='__main__':main()
