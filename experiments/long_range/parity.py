import os,sys,time,json,hashlib,argparse
from pathlib import Path
import torch,numpy as np
sys.path.insert(0,'/data/wujiaju/parity_input_once_20260811/evaluation_code')
from reasoning_loop.paper_length_telomere import load_backbone,load_controller,ControllerView,generate_paper_batch
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=2);p.add_argument('--smoke',action='store_true');a=p.parse_args()
O=Path(a.out);O.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
B=Path('/data/wujiaju/parity_input_once_20260811/backbones/parity_input_once_seed2/best.pt');J=Path('/data/wujiaju/parity_input_once_20260811/controllers/seed2/extension20to40/controller.pt')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();m,s,pay=load_backbone(B,device=torch.device('cuda'));j,jpay=load_controller(J,device=torch.device('cuda'));j=ControllerView(j,mode='full').eval();m.eval()
assert jpay['anchor_step']==1 and m.config.token_embedding_injection=='initial_only' and m.config.position_embedding=='none'
lengths=([1000] if a.smoke else list(range(500,1001,5))[a.shard::a.shards]);nex=8 if a.smoke else 128;bs=8 if a.smoke else 32
manifest={'status':'running','pid':os.getpid(),'gpu':os.getenv('CUDA_VISIBLE_DEVICES'),'checkpoint':str(B),'controller':str(J),'checkpoint_sha256':sha(B),'controller_sha256':sha(J),'code_sha256':sha(__file__),'lengths':lengths,'examples_per_length':nex,'batch_size':bs,'seed':2026092401,'schedule':'J before calls 2..n; input once; t=n','metric':'whole answer exact match; all samples retained','cuda_graph':True}
(O/'manifest.json').write_text(json.dumps(manifest,indent=2));start=time.time()
@torch.inference_mode()
def run():
 for n in lengths:
  gen=torch.Generator().manual_seed(2026092401+1009*n);tot=[0,0];preds=[];tic=time.time()
  for off in range(0,nex,bs):
   batch=generate_paper_batch(s,batch_size=bs,min_length=n,max_length=n,fixed_length=n,generator=gen).to('cuda');emb=m.input_embeddings(batch.inputs,step_index=1);state0=m.recurrent_step(torch.zeros_like(emb),emb);state=torch.cat([state0,state0]).contiguous();zero=torch.zeros_like(state)
   def step():
    prepared=torch.cat([state[:bs],j(state[bs:])]);state.copy_(m.recurrent_step(prepared,zero))
   # Capture on a side stream after warm-up; restore the true call-1 state.
   stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
   with torch.cuda.stream(stream):
    for _ in range(3):step()
   torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
   cg=torch.cuda.CUDAGraph()
   with torch.cuda.graph(cg):step()
   state.copy_(torch.cat([state0,state0]))
   for _ in range(1,n):cg.replay()
   logits=m.read_out(state[:,n:,:]).float();target=batch.targets[:,n:];pr=logits.argmax(-1);correct=pr.eq(target.repeat(2,1)).all(-1).reshape(2,bs);tot=[tot[k]+int(correct[k].sum()) for k in range(2)]
   if a.smoke:
    ref=torch.cat([state0,state0]);z=m.input_embeddings(batch.inputs,step_index=2);z=torch.cat([z,z])
    for _ in range(1,n):ref=m.recurrent_step(torch.cat([ref[:bs],j(ref[bs:])]),z)
    ref_logits=m.read_out(ref[:,n:,:]);delta=(ref_logits-logits).abs().max().item();same=bool(ref_logits.argmax(-1).eq(pr).all());assert same
    manifest['eager_comparison']={'max_logit_error':delta,'all_predictions_identical':same};print('AUDIT',manifest['eager_comparison'],flush=True)
   preds.append(correct.cpu().numpy());del cg,state,logits,batch;torch.cuda.empty_cache()
  for k,name in enumerate(['raw','J']):
   row={'variant':name,'length':n,'loop':n,'examples':nex,'correct':tot[k],'exact_match':tot[k]/nex,'elapsed':time.time()-tic}
   with (O/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  np.savez_compressed(O/f'length_{n}.npz',correct=np.concatenate(preds,axis=1))
  print(json.dumps({'length':n,'correct':tot,'examples':nex,'seconds':round(time.time()-tic,2),'elapsed':round(time.time()-start,2),'peak_mib':torch.cuda.max_memory_allocated()/2**20}),flush=True)
run();manifest.update(status='complete',elapsed=time.time()-start,peak_mib=torch.cuda.max_memory_allocated()/2**20);(O/'manifest.json').write_text(json.dumps(manifest,indent=2))
