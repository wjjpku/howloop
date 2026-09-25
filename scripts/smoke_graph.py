#!/usr/bin/env python3
"""CPU/GPU replay checks for the main N10 backbone and one one-hop controller."""
from pathlib import Path
import argparse,sys,hashlib,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments/n10/code'))
import torch
from reasoning_loop.graph_path_depth_circuit import load_checkpoint,fixed_depth_batch
from reasoning_loop.paper2027_d8l6_s1 import load_controller,terminal_h6
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--controller',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--output',type=Path,default=ROOT/'outputs/graph_smoke.json');a=p.parse_args()
torch.set_num_threads(2)
m,cfg,ck=load_checkpoint(a.checkpoint,torch.device(a.device));m.eval().requires_grad_(False)
j,jpay=load_controller(a.controller,a.checkpoint,torch.device(a.device));j.eval()
assert cfg.node_count==10 and cfg.max_loops==6 and cfg.max_depth==8
lock=torch.load(ROOT/'experiments/n10/locks/confirmation.pt',map_location='cpu',weights_only=False)
succ=lock['successors'][:2].to(a.device).repeat_interleave(10,0);starts=torch.arange(10,device=a.device).repeat(2)
with torch.no_grad():
 tok,tgt,_,_=fixed_depth_batch(cfg,20,torch.device(a.device),path_positions=10,successors=succ,start=starts)
 h=terminal_h6(m,tok,cfg);pre=logits_from_raw_state(m,j(h));post=logits_from_raw_state(m,m.apply_loop(j(h),loop_index=6))
 assert torch.isfinite(pre).all() and torch.isfinite(post).all()
 split=torch.cat([logits_from_raw_state(m,m.apply_loop(j(h[i:i+1]),loop_index=6)) for i in range(20)])
 assert torch.equal(post.argmax(-1),split.argmax(-1))
 # Verify the computational path stays differentiable through frozen F to J.
j.train();j.requires_grad_(True);j.zero_grad();loss=torch.nn.functional.cross_entropy(logits_from_raw_state(m,m.apply_loop(j(h.detach()),loop_index=6)),tgt[:,8]);loss.backward()
assert all(p.grad is None for p in m.parameters())
grads=[p.grad for p in j.parameters() if p.requires_grad];assert any(g is not None and g.abs().sum()>0 for g in grads);assert all(g is None or torch.isfinite(g).all() for g in grads)
report={'examples':20,'device':a.device,'node_count':cfg.node_count,'loops':cfg.max_loops,'finite':True,'batch_predictions_equal':True,'frozen_backbone_no_grad':True,'controller_gradient':True,'one_hop_correct':int(post.argmax(-1).eq(tgt[:,8]).sum()),'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'controller_sha256':hashlib.sha256(a.controller.read_bytes()).hexdigest(),'scope':'smoke check only, not a full statistical reproduction'}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
