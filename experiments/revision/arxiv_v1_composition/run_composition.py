import sys,os,json,time,argparse,random
from pathlib import Path
import numpy as np,torch
ORIGINAL_ROOT=Path(os.environ['HOWLOOP_ORIGINAL_ROOT']).expanduser().resolve()
sys.path.insert(0,str(ORIGINAL_ROOT/'paper_strengthening_20260925/code'))
from graph_matrix import B,load_checkpoint,fixed_depth_batch,apply_vector_map,_load_controller,sha
O=Path(os.environ['HOWLOOP_COMPOSITION_OUTPUT']).expanduser().resolve()
@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--device',default='cuda');a=ap.parse_args();device=torch.device(a.device)
 torch.set_num_threads(4);torch.set_num_interop_threads(1);O.mkdir(parents=True,exist_ok=True)
 used=set()
 for p in [B/'datasets.json',ORIGINAL_ROOT/'n10_fig4_fresh_20260924/datasets.json',ORIGINAL_ROOT/'n10_selected_mechanism_20260924/datasets.json',ORIGINAL_ROOT/'paper_strengthening_20260925/graph_data.json']:
  for gs in json.loads(p.read_text()).values():
   if isinstance(gs,list):used.update(tuple(g) for g in gs if isinstance(g,list))
 pool=sorted(set(map(tuple,torch.load(B/'locks/donors.pt',weights_only=False)['successors'].tolist()))-used)
 random.Random(2026093001).shuffle(pool);assert len(pool)>=512
 graphs=pool[:2 if a.smoke else 512]
 seq=np.asarray(json.loads((Path(__file__).parent/'results/data.json').read_text())['sequences'],dtype=np.int64)
 (O/('smoke_data.json' if a.smoke else 'data.json')).write_text(json.dumps({'graphs':graphs,'sequences':seq.tolist(),'seed':2026093001,'excluded_graph_count':len(used),'remaining_locked_pool':len(pool)}))
 for name,seed in ([('A',6)] if a.smoke else [('A',6),('B',3),('C',4),('D',5),('E',7)]):
  cp=B/f'backbones/local_control_L6_seed{seed}/best.pt';jp=B/f'local/{name}/controllers.pt'
  assert sha(B/'locks/donors.pt') in json.loads((cp.parent/'summary.json').read_text())['extra_excluded_locks'].values()
  m,cfg,_=load_checkpoint(cp,device);m.eval().requires_grad_(False);versions=[p._version for p in m.parameters()]
  for fit in ([1] if a.smoke else [1,2]):
   dest=O/(f'{name}_fit{fit}'+('_smoke' if a.smoke else '')+'.npz')
   maps={k:_load_controller(path=jp,name=f'seed{fit}_J_{s}_beh_rank48',device=device) for k,s in [(1,'one'),(2,'two')]}
   for k in [1,2]:
    ident=json.loads((B/f'local/{name}/hop{k}_seed{fit}/training_identity.json').read_text());assert str(B/'locks/donors.pt') in str(ident['arguments']['train_exclude_locks'])
   records={k:[] for k in ['predictions','pre','labels','native','first_only','omit_current']};tic=time.time()
   def read(h):return m.unembed(m.ln_final(h[:,-1]))[:,:10].argmax(-1).cpu().numpy()
   for off in range(0,len(graphs),16):
    batch=graphs[off:off+16];g=torch.tensor(batch,device=device).repeat_interleave(10,0);cur=torch.arange(10,device=device).repeat(len(batch));start=cur.clone()
    for _ in range(8):start=g.argsort(-1).gather(1,start[:,None])[:,0]
    tok,*_=fixed_depth_batch(cfg,len(g),device,path_positions=10,successors=g,start=start);h=m.token_embed(tok)+m.pos_embed[None]
    for t in range(6):h=m.apply_loop(h,loop_index=t)
    labels=[cur]
    for _ in range(16):labels.append(g.gather(1,labels[-1][:,None])[:,0])
    records['labels'].append(torch.stack(labels,-1).cpu().numpy());cache={():h};pred={};pre={};omit={}
    for sequence in seq:
     for t in range(8):
      prefix=tuple(sequence[:t+1]);prev=prefix[:-1]
      if prefix not in cache:
       if prev not in omit:omit[prev]=read(m.apply_loop(cache[prev],loop_index=6+t))
       x=apply_vector_map(cache[prev],positions=tuple(range(cfg.seq_len)),controller=maps[prefix[-1]]);pre[prefix]=read(x);cache[prefix]=m.apply_loop(x,loop_index=6+t);pred[prefix]=read(cache[prefix])
    for k,dic,shift in [('predictions',pred,1),('pre',pre,1),('omit_current',omit,0)]:
     records[k].append(np.stack([np.stack([dic[tuple(s[:t+shift])] for t in range(8)],-1) for s in seq],1))
    raw=[];x=h
    for t in range(8):x=m.apply_loop(x,loop_index=6+t);raw.append(read(x))
    records['native'].append(np.stack(raw,-1));first=[]
    for k in [1,2]:
     x=cache[(k,)];r=[read(x)]
     for t in range(1,8):x=m.apply_loop(x,loop_index=6+t);r.append(read(x))
     first.append(np.stack(r,-1))
    records['first_only'].append(np.stack(first,1));del cache
    print(json.dumps({'model':name,'fit':fit,'graphs_done':off+len(batch),'seconds':time.time()-tic,'peak_mib':(torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else 0)}),flush=True)
   assert versions==[p._version for p in m.parameters()]
   np.savez_compressed(dest,**{k:np.concatenate(v) for k,v in records.items()},sequences=seq,graphs=np.array(graphs))
   dest.with_suffix('.json').write_text(json.dumps({'complete':True,'seed':seed,'fit':fit,'pid':os.getpid(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES','cpu'),'seconds':time.time()-tic,'backbone_sha256':sha(cp),'controllers_sha256':sha(jp),'code_sha256':sha(__file__),'weights_unchanged':True,'locked_graphs_excluded_from_backbone_and_controller_training':True,'peak_mib':(torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else 0)},indent=2))
  del m;torch.cuda.empty_cache()
 (O/('smoke_complete.json' if a.smoke else 'complete.json')).write_text(json.dumps({'complete':True,'time':time.time()}))
if __name__=='__main__':main()
