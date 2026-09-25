import argparse,json,hashlib
from pathlib import Path
import torch
from reasoning_loop.paper2027_d8l6_s1 import load_controller,validate,terminal_h6
from reasoning_loop.graph_path_depth_circuit import load_checkpoint,fixed_depth_batch
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state
ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--runs',type=Path,required=True);ap.add_argument('--device',default='cuda');a=ap.parse_args();torch.set_num_threads(4);R=Path(__file__).resolve().parents[1];device=torch.device(a.device);m,c,p=load_checkpoint(a.checkpoint,device);m.requires_grad_(False);maps={};checks=[]
graphs=json.loads((R/'datasets.json').read_text())['confirmation']
with torch.no_grad():
 for hop in (1,2):
  for seed in (1,2):
   cp=a.runs/f'hop{hop}_seed{seed}/best_controller.pt';j,item=load_controller(cp,a.checkpoint,device)
   score=validate(m,c,j,hop=hop,device=device,lock_path=R/'locks/selection.pt',batch_size=500);assert score==item['validation_post_executor_accuracy']
   sd=j.state_dict();w=torch.diag(sd['diagonal'])+sd['A']@sd['B'];b=sd['bias'];name=f'seed{seed}_J_{"one" if hop==1 else "two"}_beh_rank48';maps[name]={'weight':w.cpu(),'bias':b.cpu()}
   mismatches=0;err=0.;n=0
   for first in range(0,len(graphs),16):
    g=torch.tensor(graphs[first:first+16],device=device).repeat_interleave(c.node_count,0);current=torch.arange(c.node_count,device=device).repeat(len(g)//c.node_count);start=current.clone();inv=g.argsort(-1)
    for _ in range(c.max_depth):start=inv.gather(1,start[:,None])[:,0]
    tok,*_=fixed_depth_batch(c,len(g),device,path_positions=10,successors=g,start=start);h=terminal_h6(m,tok,c);factor=j(h);dense=h@w+b
    for x,y in [(factor,dense),(m.apply_loop(factor,loop_index=6),m.apply_loop(dense,loop_index=6))]:
     la=logits_from_raw_state(m,x);lb=logits_from_raw_state(m,y);mismatches+=int((la.argmax(-1)!=lb.argmax(-1)).sum());err=max(err,float((la-lb).abs().max()))
    n+=len(g)
   assert mismatches==0,dict(name=name,mismatches=mismatches,max_logit_error=err)
   checks.append(dict(name=name,examples=n,prediction_mismatches=mismatches,max_logit_error=err,validation_replayed=score,selected_update=item['update'],checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest()))
torch.save(maps,a.runs/'controllers.pt');(a.runs/'export_validation.json').write_text(json.dumps(checks,indent=2));print(json.dumps({'event':'export_validated','checks':checks}),flush=True)
