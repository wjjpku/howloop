import os,sys,json,time,hashlib
from pathlib import Path
import torch,numpy as np
B=Path('/data/paperexperiment/n10_migration_20260923');R=Path('/data/paperexperiment/reviewer_revision_20260926/matched');sys.path.insert(0,str(B/'code'))
from reasoning_loop.graph_path_depth_circuit import load_checkpoint,fixed_depth_batch
@torch.no_grad()
def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);graphs=torch.load(R/'test.pt',weights_only=False)['successors'];device=torch.device('cpu')
 for d in sorted((R/'runs').glob('*_seed*')):
  cp=d/'final.pt';dest=d/'native_test.json'
  if not cp.exists() or dest.exists():continue
  tic=time.time();m,cfg,_=load_checkpoint(cp,device);m.eval().requires_grad_(False);pred=[];lab=[]
  for off in range(0,len(graphs),16):
   g=graphs[off:off+16].repeat_interleave(10,0);start=torch.arange(10).repeat(len(g)//10);tok,y,*_=fixed_depth_batch(cfg,len(g),device,path_positions=16,successors=g,start=start);h=m.token_embed(tok)+m.pos_embed[None];rr=[]
   def read(h):return m.unembed(m.ln_final(h[:,-1]))[:,:10].argmax(-1)
   rr.append(read(h))
   for t in range(16):h=m.apply_loop(h,loop_index=t);rr.append(read(h))
   pred.append(torch.stack(rr,-1).numpy());lab.append(torch.cat([start[:,None],y],1).numpy())
  pred=np.concatenate(pred);lab=np.concatenate(lab);np.savez_compressed(d/'native_test.npz',predictions=pred,labels=lab,graphs=graphs.numpy());mask=(lab[:,8]!=lab[:,9])&(lab[:,8]!=lab[:,10])&(lab[:,9]!=lab[:,10])
  # CPU kernels may differ numerically from the GPU cache near ties; report disagreement rather than silently conflating them.
  matches={}
  for f in (d/'maps').glob('one_fit*.npz'):
   z=np.load(f);assert np.array_equal(z['labels'],lab[:,8:11]);matches[f.name]=int((z['native']!=pred[:,9]).sum())
  summary={'backbone':d.name,'endpoint_test_accuracy':float((pred[:,8]==lab[:,8]).mean()),'endpoint_test_correct':int((pred[:,8]==lab[:,8]).sum()),'n_all':len(pred),'n_distinct':int(mask.sum()),'one_hop_schedule_accuracy':[float((pred[:,t]==lab[:,t]).mean()) for t in range(1,17)],'requested_endpoint_accuracy_by_loop':[float((pred[:,t]==lab[:,8]).mean()) for t in range(17)],'native_F9_distinct':{str(k):float((pred[mask,9]==lab[mask,8+k]).mean()) for k in [0,1,2]},'cpu_gpu_native_prediction_disagreements':matches,'backbone_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'test_lock_sha256':hashlib.sha256((R/'test.pt').read_bytes()).hexdigest(),'seconds':time.time()-tic}
  dest.write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
