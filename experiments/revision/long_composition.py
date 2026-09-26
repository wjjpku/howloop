import sys,os,json,time,argparse,hashlib
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,'/data/paperexperiment/paper_strengthening_20260925/code')
from graph_matrix import B,load_checkpoint,fixed_depth_batch,apply_vector_map,_load_controller,sha
O=Path('/data/paperexperiment/reviewer_revision_20260926/long_composition')
@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);O.mkdir(parents=True,exist_ok=True)
 graphs=json.load(open('/data/paperexperiment/paper_strengthening_20260925/graph_data.json'))['confirmation']; rng=np.random.default_rng(2026092608)
 mixed=[]
 while len(mixed)<32:
  x=tuple(rng.integers(1,3,8).tolist())
  if len(set(x))==2 and x not in mixed:mixed.append(x)
 seq=np.array([[1]*8,[2]*8]+mixed);np.save(O/'sequences.npy',seq)
 for name,seed in [('A',6)]:
  cp=B/f'backbones/local_control_L6_seed{seed}/best.pt';jp=B/f'local/{name}/controllers.pt';m,cfg,_=load_checkpoint(cp,torch.device('cuda'));m.eval().requires_grad_(False);versions=[p._version for p in m.parameters()]
  for fit in [1,2]:
   dest=O/(f'{name}_fit{fit}'+('_smoke' if a.smoke else '')+'.npz')
   if dest.exists():continue
   maps={k:_load_controller(path=jp,name=f'seed{fit}_J_{s}_beh_rank48',device=torch.device('cuda')) for k,s in [(1,'one'),(2,'two')]};preds=[];pres=[];ys=[];raws=[];tic=time.time()
   def read(h):return m.unembed(m.ln_final(h[:,-1]))[:,:10].argmax(-1)
   for off in range(0,2 if a.smoke else len(graphs),16):
    batch=graphs[off:off+(2 if a.smoke else 16)];g=torch.tensor(batch,device='cuda').repeat_interleave(10,0);cur=torch.arange(10,device='cuda').repeat(len(batch));start=cur.clone()
    for _ in range(8):start=g.argsort(-1).gather(1,start[:,None])[:,0]
    tok,*_=fixed_depth_batch(cfg,len(g),torch.device('cuda'),path_positions=10,successors=g,start=start);h=m.token_embed(tok)+m.pos_embed[None]
    for t in range(6):h=m.apply_loop(h,loop_index=t)
    labels=[cur]
    for _ in range(16):labels.append(g.gather(1,labels[-1][:,None])[:,0])
    ys.append(torch.stack(labels,-1).cpu().numpy());pp=[];pre=[];cache={():h};outcache={};precache={}
    for sequence in seq:
     run=[];bef=[]
     for t in range(8):
      prefix=tuple(sequence[:t+1]);prev=prefix[:-1]
      if prefix not in cache:
       x=apply_vector_map(cache[prev],positions=tuple(range(cfg.seq_len)),controller=maps[prefix[-1]]);precache[prefix]=read(x).cpu().numpy();cache[prefix]=m.apply_loop(x,loop_index=6+t);outcache[prefix]=read(cache[prefix]).cpu().numpy()
      run.append(outcache[prefix]);bef.append(precache[prefix])
     pp.append(np.stack(run,-1));pre.append(np.stack(bef,-1))
    # All four first-two prefixes are checked against the archived two-step run.
    old=np.load(f'/data/paperexperiment/continuous_composition_20260925/{name}_fit{fit}.npz')
    for aa in [1,2]:
     for bb in [1,2]:
      prefix=(aa,bb);idx=list(old['sequences']).index(('one' if aa==1 else 'two')+'_'+('one' if bb==1 else 'two'))
      assert np.array_equal(outcache[prefix],old['second'][off*10:(off+len(batch))*10,idx]),'two-step replay mismatch'
    preds.append(np.stack(pp,1));pres.append(np.stack(pre,1));rr=[];x=h
    for t in range(8):x=m.apply_loop(x,loop_index=6+t);rr.append(read(x).cpu().numpy())
    raws.append(np.stack(rr,-1));del cache,outcache,precache
    print(json.dumps({'model':name,'fit':fit,'graphs':off+len(batch),'seconds':time.time()-tic,'peak_mib':torch.cuda.max_memory_allocated()/2**20}),flush=True)
    if torch.cuda.mem_get_info()[0]<16*2**30:raise RuntimeError('GPU reserve breached')
   assert versions==[p._version for p in m.parameters()]
   np.savez_compressed(dest,predictions=np.concatenate(preds),pre=np.concatenate(pres),labels=np.concatenate(ys),native=np.concatenate(raws),sequences=seq,graphs=np.array(graphs[:2] if a.smoke else graphs))
   dest.with_suffix('.json').write_text(json.dumps({'complete':True,'seed':seed,'fit':fit,'pid':os.getpid(),'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'seconds':time.time()-tic,'backbone_sha256':sha(cp),'controllers_sha256':sha(jp),'code_sha256':sha(__file__),'two_step_replay_exact':True,'weights_unchanged':True,'peak_mib':torch.cuda.max_memory_allocated()/2**20},indent=2))
   if a.smoke:return
  del m;torch.cuda.empty_cache()
 (O/'complete.json').write_text(json.dumps({'complete':True,'time':time.time()}))
if __name__=='__main__':main()
