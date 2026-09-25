"""Redesigned N10 covered-horizon case, not a replica of legacy phase-aligned fitting."""
import json,hashlib,time,argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from reasoning_loop.graph_path_depth_circuit import load_checkpoint,fixed_depth_batch
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state
from reasoning_loop.paper2027_graph_g3_controller import DiagonalLowRankGraphController
from reasoning_loop.paper2027_graph_g4_protocol import load_unique_lock,merged_forbidden_codes,sample_training_permutations
R=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda');ap.add_argument('--out',type=Path,default=R/'single_h64');ap.add_argument('--checkpoint',type=Path,default=R/'backbones/trajectory_L8_seed0/best.pt');ap.add_argument('--updates-per-stage',type=int,default=1000);ap.add_argument('--horizons',type=int,nargs='+',default=[1,2,4,8,16,32,64]);ap.add_argument('--batch-size',type=int,default=32);ap.add_argument('--skip-evaluation',action='store_true');args=ap.parse_args()
out=args.out;out.mkdir(exist_ok=True);torch.set_num_threads(4);device=torch.device(args.device);ck=args.checkpoint;m,c,cp=load_checkpoint(ck,device);m.requires_grad_(False);torch.manual_seed(211001);j=DiagonalLowRankGraphController(256,48).to(device);opt=torch.optim.AdamW(j.parameters(),lr=1e-4,weight_decay=0);locks=sorted((R/'locks').glob('*.pt'));forbidden=merged_forbidden_codes(locks,node_count=10);before={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
def write(name,x):(out/name).write_text(json.dumps(x,indent=2))
@torch.no_grad()
def initial(graphs):
 tokens,path,*_=fixed_depth_batch(c,len(graphs),device,path_positions=136,successors=graphs)
 h=m.token_embed(tokens)+m.pos_embed[None]
 for t in range(8):h=m.apply_loop(h,loop_index=t)
 return h,path
@torch.no_grad()
def evaluate(horizon,modes,split):
 lock=load_unique_lock(R/'locks'/f'{split}.pt');counts={mode:torch.zeros(horizon,dtype=torch.long,device=device) for mode in modes};n=0
 for lo in range(0,len(lock['successors']),10):
  gs=lock['successors'][lo:lo+10].repeat_interleave(10,0).to(device);starts=torch.arange(10,device=device).repeat(len(gs)//10)
  tokens,path,*_=fixed_depth_batch(c,len(gs),device,path_positions=8+horizon,successors=gs,start=starts);h=m.token_embed(tokens)+m.pos_embed[None]
  for t in range(8):h=m.apply_loop(h,loop_index=t)
  for mode in modes:
   x=h.clone()
   for k in range(1,horizon+1):
    z=j(x) if mode=='always' or (mode=='first16' and k<=16) else x
    x=m.apply_loop(z,loop_index=7+k);counts[mode][k-1]+=logits_from_raw_state(m,x).argmax(-1).eq(path[:,7+k]).sum()
  n+=len(gs)
 return {mode:{'accuracy':(count/n).tolist(),'n':n} for mode,count in counts.items()}
write('manifest.json',dict(status='running',node_count=10,checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),seed=211001,horizons=args.horizons,updates_per_stage=args.updates_per_stage,batch_size=args.batch_size,parameterization='D+AB+b rank48',objective='mean successor CE over native-h8 continuation',lr=1e-4,change_from_legacy='New explicitly specified curriculum from native h8, no historical affine initialization or decoded phase alignment. Do not claim matching legacy budget.',started=time.time()))
for horizon in args.horizons:
 save=out/f'h{horizon}.pt'
 if save.exists():
  item=torch.load(save,weights_only=False);j.load_state_dict(item['J']);opt.load_state_dict(item['optimizer']);torch.set_rng_state(item['cpu_rng']);
  if device.type=='cuda':torch.cuda.set_rng_state(item['cuda_rng'])
  continue
 for step in range(1,args.updates_per_stage+1):
  graphs=sample_training_permutations(batch_size=args.batch_size,node_count=10,forbidden_codes=forbidden,device=device);h,path=initial(graphs);losses=[]
  for k in range(1,horizon+1):
   h=m.apply_loop(j(h),loop_index=7+k);losses.append(F.cross_entropy(logits_from_raw_state(m,h),path[:,7+k]))
  loss=torch.stack(losses).mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(j.parameters(),1);opt.step()
  if step%100==0:print(json.dumps(dict(horizon=horizon,step=step,loss=float(loss.detach()))),flush=True)
 torch.save(dict(J=j.state_dict(),optimizer=opt.state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state() if device.type=='cuda' else None),save)
 if not args.skip_evaluation:write(f'validation_h{horizon}.json',evaluate(horizon,['always'],'selection'))
if not args.skip_evaluation:
 for split in ['confirmation','rings']:write(f'evaluation_{split}.json',evaluate(128,['always','first16','raw'],split))
assert all(torch.equal(before[k],v.cpu()) for k,v in m.state_dict().items());write('complete.json',dict(status='implementation_fixture' if args.skip_evaluation else 'complete',backbone_unchanged=True,peak_memory_gib=torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else 0))
