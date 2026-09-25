"""Five-backbone graph mechanism replication, matched training and lifetime audit."""
import argparse, copy, csv, dataclasses, hashlib, itertools, json, os, random, sys, time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as TF
sys.path.insert(0,str(Path(__file__).resolve().parent))
from reasoning_loop.graph_path_depth_circuit import load_checkpoint,fixed_depth_batch
from reasoning_loop.graph_path_functional_circuit import run_instrumented_state,FunctionalIntervention
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state
from reasoning_loop.graph_path_jump_controller_causal_switch import _load_controller
from reasoning_loop.graph_path_jump_controller import apply_vector_map

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def adv(g,c,n):
 for _ in range(n):c=g.gather(1,c[:,None])[:,0]
 return c
def get_h(m,c,g,v,shuffle=False):
 tok,*_=fixed_depth_batch(c,len(v),g.device,path_positions=10,successors=g,start=adv(g.argsort(-1),v,8))
 if shuffle:
  gen=torch.Generator(device=g.device).manual_seed(928341)
  order=torch.rand((len(v),10),generator=gen,device=g.device).argsort(-1)
  tok[:,1:31]=tok[:,1:31].reshape(-1,10,3).gather(1,order[:,:,None].expand(-1,-1,3)).reshape(-1,30)
 h=m.token_embed(tok)+m.pos_embed[None]
 for t in range(6):h=m.apply_loop(h,loop_index=t)
 return h
def loop(m,h):return m.apply_loop(h,loop_index=6)
def instrument(m,h,parts=(),source=None,heads=(0,1,2,3),differentiable=False):
 iv=tuple(FunctionalIntervention(site=1,component=p,mode='patch',heads=heads,positions=(34,) if p in ('q','attention_pattern','head_context') else None) for p in parts)
 runner=run_instrumented_state.__wrapped__ if differentiable else run_instrumented_state
 return runner(m,h,loop_indices=(6,),interventions=iv,donor_trace=source)

class Map(nn.Module):
 def __init__(self,kind):
  super().__init__();self.kind=kind;self.diag=nn.Parameter(torch.ones(256));self.bias=nn.Parameter(torch.zeros(256))
  if kind=='random_Q':
   self.A=nn.Parameter(torch.zeros(256,16));self.register_buffer('B',torch.linalg.qr(torch.randn(256,16)).Q.T.contiguous())
  else:self.A=nn.Parameter(torch.randn(256,8)*.02);self.B=nn.Parameter(torch.zeros(8,256))
 def forward(self,x):return x*self.diag+(x@self.A)@self.B+self.bias

def controlled(m,h,j):
 if j.kind=='all':out=loop(m,j(h))
 elif j.kind=='answer':
  hj=h.clone();hj[:,34]=j(h[:,34]);out=loop(m,hj)
 else:
  with torch.no_grad():_,tr=instrument(m,h)
  q=tr.sites[1].q.clone();flat=q[:,:,34,:].reshape(-1,256).clone();q[:,:,34,:]=j(flat).reshape(-1,4,64)
  donor=dataclasses.replace(tr,sites=[tr.sites[0],dataclasses.replace(tr.sites[1],q=q)])
  logits,trace=instrument(m,h,('q',),donor,differentiable=True)
  return logits,trace.sites[-1].hidden_out
 return logits_from_raw_state(m,out),out

@torch.no_grad()
def cache(m,c,graphs):
 gs=[];vs=[];hs=[]
 for first in range(0,len(graphs),16):
  g=torch.tensor(graphs[first:first+16],device='cuda').repeat_interleave(10,0);v=torch.arange(10,device='cuda').repeat(len(g)//10)
  gs.append(g);vs.append(v);hs.append(get_h(m,c,g,v))
 g=torch.cat(gs);v=torch.cat(vs);return torch.cat(hs),g,v,adv(g,v,1)

@torch.no_grad()
def evaluate(m,h,j):
 preds=[];pre=[]
 for first in range(0,len(h),128):
  x=h[first:first+128];logits,_=controlled(m,x,j);preds.extend(logits.argmax(-1).tolist())
  if j.kind in ('all','answer'):
   xx=j(x) if j.kind=='all' else torch.cat((x[:,:34],j(x[:,34:35])),dim=1)
   pre.extend(logits_from_raw_state(m,xx).argmax(-1).tolist())
 return preds,pre


def main():
 from reasoning_loop.paper2027_graph_g4_protocol import merged_forbidden_codes,sample_training_permutations
 ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);a=ap.parse_args();torch.set_num_threads(4)
 R=Path(__file__).resolve().parents[1];root=R/'restricted';root.mkdir(exist_ok=True)
 forbidden=merged_forbidden_codes(sorted((R/'locks').glob('*.pt')),node_count=10)
 torch.manual_seed(2026092317);train_rows=[];seen=set()
 while len(train_rows)<2048:
  proposal=sample_training_permutations(batch_size=2048,node_count=10,forbidden_codes=forbidden,device=torch.device('cpu'))
  for row in proposal.tolist():
   if tuple(row) not in seen:seen.add(tuple(row));train_rows.append(row)
   if len(train_rows)==2048:break
 ds=json.loads((R/'datasets.json').read_text());split={'train':train_rows,'validation':ds['selection'],'test':ds['confirmation'][:256]}
 splitfile=root/f'split_shard{a.shard}.json';save(splitfile,split)
 for ix,seed in enumerate([6,3,4,5,7]):
  if ix%2!=a.shard:continue
  label='ABCDE'[ix];out=root/label;out.mkdir(exist_ok=True)
  ck=R/'backbones'/f'local_control_L6_seed{seed}'/'best.pt';m,c,_=load_checkpoint(ck,torch.device('cuda'));m.eval().requires_grad_(False);versions=[p._version for p in m.parameters()]
  train,_,_,yt=cache(m,c,split['train']);val,_,_,yv=cache(m,c,split['validation']);test,_,_,ye=cache(m,c,split['test'])
  for fit in range(3):
   for kind in ('all','answer','Q','random_Q'):
    sub=out/f'{kind}_seed{fit}'
    if (sub/'evaluation.json').exists():continue
    torch.manual_seed(20260922+fit);j=Map(kind).cuda();assert sum(p.numel() for p in j.parameters())==4608
    with torch.no_grad():
     ll,_=controlled(m,test[:16],j);base=logits_from_raw_state(m,loop(m,test[:16]));err=float((ll-base).abs().max());assert err<2e-4,err
    opt=torch.optim.AdamW(j.parameters(),lr=.001,weight_decay=0);gen=torch.Generator(device='cuda').manual_seed(7300+fit);history=[]
    for step in range(1,801):
     idx=torch.randint(len(train),(64,),generator=gen,device='cuda');ll,_=controlled(m,train[idx],j);loss=TF.cross_entropy(ll,yt[idx]);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(j.parameters(),1.0);opt.step()
     if step%100==0:history.append(dict(step=step,ce=float(loss.detach())));print(json.dumps(dict(backbone=label,kind=kind,fit=fit,**history[-1])),flush=True)
    pred,pre=evaluate(m,test,j);vp,_=evaluate(m,val,j);sub.mkdir(exist_ok=True)
    torch.save(dict(state_dict=j.state_dict(),kind=kind,seed=fit,steps=800,backbone_sha=sha(ck)),sub/'final.pt')
    # Reload, independently rerun predictions before reporting the fit.
    check=torch.load(sub/'final.pt',weights_only=False);j.load_state_dict(check['state_dict']);replayed,_=evaluate(m,test,j);assert pred==replayed
    save(sub/'evaluation.json',dict(predictions=pred,targets=ye.tolist(),pre_predictions=pre,accuracy=float(torch.tensor(pred,device='cuda').eq(ye).float().mean()),validation_accuracy=float(torch.tensor(vp,device='cuda').eq(yv).float().mean()),replay_exact=True,identity_error=err,parameters=4608,history=history))
  assert versions==[p._version for p in m.parameters()];save(out/'complete.json',dict(status='complete',backbone_sha256=sha(ck),split_sha256=sha(splitfile)));del m,train,val,test;torch.cuda.empty_cache()
if __name__=='__main__':main()
