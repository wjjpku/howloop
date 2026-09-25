import json,subprocess,sys,hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
from reasoning_loop.graph_path_depth_circuit import load_checkpoint,fixed_depth_batch
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state
from restricted import Map,controlled,get_h,loop
from train_supervision_J import prepare_first_loop,run_seven
from supervision_utils import AffineJ
R=Path(__file__).resolve().parents[1];torch.set_num_threads(4);device=torch.device('cpu');ck=R/'smoke_snapshot/best.pt';m,c,_=load_checkpoint(ck,device);m.requires_grad_(False)
g=torch.tensor(json.loads((R/'datasets.json').read_text())['smoke'][:1]).repeat_interleave(10,0);v=torch.arange(10)
with torch.no_grad():h=get_h(m,c,g,v);base=logits_from_raw_state(m,loop(m,h))
errors={}
for kind in ['all','answer','Q','random_Q']:
 j=Map(kind);logits,_=controlled(m,h,j);err=float((logits-base).abs().max());assert err<2e-4;loss=F.cross_entropy(logits,g.gather(1,v[:,None])[:,0]);loss.backward();assert all(torch.isfinite(p.grad).all() for p in j.parameters() if p.grad is not None);assert any(p.grad.abs().sum()>0 for p in j.parameters() if p.grad is not None);errors[kind]=err
m,c,_=load_checkpoint(R/'smoke_L8/best.pt',device);m.requires_grad_(False);graphs=json.loads((R/'datasets.json').read_text())['smoke'][:1];data={'graphs':graphs,'order':[list(range(9,-1,-1))]};x,path,tokens=prepare_first_loop(m,c,data,device);j=AffineJ(c.d_model);out,pre=run_seven(m,j,x);assert out.shape==(10,35,256);F.cross_entropy(logits_from_raw_state(m,out),path[:,15]).backward();assert j.delta.weight.grad.abs().sum()>0
result={'status':'passed','restricted_identity_logit_errors':errors,'restricted_gradients':True,'fixed8_supervision_J_shape':[10,35,256],'fixed8_supervision_J_gradients':True};(R/'supplement_implementation_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
