import argparse,hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
from reasoning_loop.paper2027_d8l6_s1 import load_controller,validate,terminal_h6,target_for_hop
from reasoning_loop.graph_path_depth_circuit import load_checkpoint,fixed_depth_batch
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state
ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);args=ap.parse_args();torch.set_num_threads(3)
d=next(x for x in json.loads((ROOT/'backbones.json').read_text()) if x['name']==args.name);path=Path(d['checkpoint']);device=torch.device('cuda');model,cfg,p=load_checkpoint(path,device);model.requires_grad_(False);assert p['step']==d['checkpoint_step'];out=ROOT/'runs'/args.name;maps={};checks=[]
graphs=json.loads((ROOT/'datasets.json').read_text())['confirmation']
for hop in (1,2):
 for seed in (1,2):
  cp=out/f'hop{hop}_seed{seed}/best_controller.pt';j,item=load_controller(cp,path,device)
  score=validate(model,cfg,j,hop=hop,device=device,seed=10000+seed,batches=8,batch_size=256);assert score==item['validation_post_executor_accuracy']
  sd=j.state_dict();w=torch.diag(sd['diagonal'])+sd['A']@sd['B'];b=sd['bias'];name=f'seed{seed}_J_{"one" if hop==1 else "two"}_beh_rank48';maps[name]={'weight':w.cpu(),'bias':b.cpu()}
  mismatches=0;err=0.;n=0
  with torch.no_grad():
   for first in range(0,len(graphs),16):
    g=torch.tensor(graphs[first:first+16],device=device).repeat_interleave(8,0);current=torch.arange(8,device=device).repeat(len(g)//8);s=current.clone();inv=g.argsort(-1)
    for _ in range(8):s=inv.gather(1,s[:,None])[:,0]
    tok,*_=fixed_depth_batch(cfg,len(g),device,path_positions=10,successors=g,start=s);h=terminal_h6(model,tok,cfg);a=j(h);v=h@w+b
    for x,y in [(a,v),(model.apply_loop(a,loop_index=6),model.apply_loop(v,loop_index=6))]:
     la=logits_from_raw_state(model,x);lb=logits_from_raw_state(model,y);mismatches+=int((la.argmax(-1)!=lb.argmax(-1)).sum());err=max(err,float((la-lb).abs().max()))
    n+=len(g)
  # Floating-point dense re-association may change argmax near ties. Do not silently
  # accept those: fail the campaign and switch evaluator to exact factors if needed.
  assert mismatches==0,{'name':name,'mismatches':mismatches,'max_logit_error':err}
  checks.append({'name':name,'n':n,'prediction_mismatches':mismatches,'max_logit_error':err,'validation_replayed':score,'selected_update':item['update'],'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest()})
torch.save(maps,out/'controllers.pt');(out/'export_validation.json').write_text(json.dumps(checks,indent=2));print(json.dumps({'event':'export_validated','name':args.name,'checks':checks}),flush=True)
