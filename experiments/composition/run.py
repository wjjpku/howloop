import sys,os,time,json,itertools,hashlib
from pathlib import Path
import torch,numpy as np
ROOT=Path('/data/paperexperiment/paper_strengthening_20260925');sys.path.insert(0,str(ROOT/'code'))
from graph_matrix import B,load_checkpoint,fixed_depth_batch,apply_vector_map,_load_controller,sha
O=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 graphs=json.loads((ROOT/'graph_data.json').read_text())['confirmation']; device=torch.device('cuda');names=['raw','one','two'];seqs=list(itertools.product(names,repeat=2))
 for name,seed in [('A',6),('C',4),('B',3),('D',5),('E',7)]:
  cp=B/f'backbones/local_control_L6_seed{seed}/best.pt';jp=B/f'local/{name}/controllers.pt';m,cfg,_=load_checkpoint(cp,device);m.eval().requires_grad_(False);vers=[p._version for p in m.parameters()]
  for fit in [1,2]:
   tic=time.time();maps={k:_load_controller(path=jp,name=f'seed{fit}_J_{k}_beh_rank48',device=device) for k in ['one','two']};labels=[];first=[];second=[];pre1=[];pre2=[]
   def read(h):return m.unembed(m.ln_final(h[:,-1]))[:,:10].argmax(-1)
   def steer(h,k):return h if k=='raw' else apply_vector_map(h,positions=tuple(range(cfg.seq_len)),controller=maps[k])
   for off in range(0,len(graphs),16):
    g=torch.tensor(graphs[off:off+16],device=device).repeat_interleave(10,0);cur=torch.arange(10,device=device).repeat(len(g)//10);start=cur.clone()
    for _ in range(8):start=g.argsort(-1).gather(1,start[:,None])[:,0]
    tok,*_=fixed_depth_batch(cfg,len(g),device,path_positions=10,successors=g,start=start)
    h=m.token_embed(tok)+m.pos_embed[None]
    for t in range(6):h=m.apply_loop(h,loop_index=t)
    ys=[cur]
    for _ in range(4):ys.append(g.gather(1,ys[-1][:,None])[:,0])
    labels.append(torch.stack(ys,-1).cpu().numpy());hs={k:steer(h,k) for k in names};pre1.append(torch.stack([read(hs[k]) for k in names],-1).cpu().numpy());hs={k:m.apply_loop(v,loop_index=6) for k,v in hs.items()};first.append(torch.stack([read(hs[k]) for k in names],-1).cpu().numpy())
    ss=[];pp=[]
    for a,b in seqs:
     x=steer(hs[a],b);pp.append(read(x));ss.append(read(m.apply_loop(x,loop_index=7)))
    second.append(torch.stack(ss,-1).cpu().numpy());pre2.append(torch.stack(pp,-1).cpu().numpy())
   first=np.concatenate(first);old=np.load(ROOT/f'graph/{name}_fit{fit}.npz');assert np.array_equal(first,old['native']), 'first step mismatch'
   assert vers==[p._version for p in m.parameters()]
   np.savez_compressed(O/f'{name}_fit{fit}.npz',first=first,second=np.concatenate(second),pre1=np.concatenate(pre1),pre2=np.concatenate(pre2),labels=np.concatenate(labels),graphs=graphs,names=names,sequences=['_'.join(x) for x in seqs])
   meta={'model':name,'seed':seed,'fit':fit,'seconds':time.time()-tic,'first_step_matches_previous':True,'weights_unchanged':True,'backbone_sha256':sha(cp),'controllers_sha256':sha(jp),'code_sha256':sha(__file__),'data_sha256':sha(ROOT/'graph_data.json'),'pid':os.getpid(),'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'peak_mib':torch.cuda.max_memory_allocated()/2**20}
   (O/f'{name}_fit{fit}.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
  del m;torch.cuda.empty_cache()
 (O/'complete.json').write_text(json.dumps({'complete':True,'time':time.time()}))
if __name__=='__main__':main()
