import os,sys,json,time,hashlib,random,argparse,itertools
from pathlib import Path
import torch,numpy as np
B=Path('/data/wujiaju/n10_migration_20260923')
sys.path.insert(0,str(B/'code'))
from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch,load_checkpoint
from reasoning_loop.graph_path_functional_circuit import FunctionalIntervention as I,run_instrumented_state
from reasoning_loop.graph_path_jump_controller import apply_vector_map
from reasoning_loop.graph_path_jump_controller_causal_switch import _load_controller
P=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2))
def prepare():
 used=set()
 for p in [B/'datasets.json',Path('/data/wujiaju/n10_fig4_fresh_20260924/datasets.json'),Path('/data/wujiaju/n10_selected_mechanism_20260924/datasets.json')]:
  d=json.loads(p.read_text())
  for gs in d.values():used.update(map(tuple,gs))
 # Exclude all previously generated semantic donor variants as in fresh confirmation.
 d=json.loads((B/'datasets.json').read_text())
 for panel in ['discovery','confirmation','smoke']:
  for g in d[panel]:
   for c in range(10):
    wrong=(c+3)%10
    for x,y in [(i,(i+1)%10) for i in range(10)]+[(g[wrong],(g[wrong]+1)%10 if (g[wrong]+1)%10!=wrong else (g[wrong]+2)%10)]:
     h=g.copy();h[x],h[y]=h[y],h[x];used.add(tuple(h))
 pool=sorted(set(map(tuple,torch.load(B/'locks/donors.pt',weights_only=False)['successors'].tolist()))-used)
 random.Random(2026092501).shuffle(pool)
 save(P/'graph_data.json',{'smoke':pool[:2],'confirmation':pool[2:514],'seed':2026092501,'remaining_pool':len(pool)})
 return pool
@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--models',nargs='+',default=['A','C','B','D','E']);a=ap.parse_args()
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 if not (P/'graph_data.json').exists():prepare()
 data=json.loads((P/'graph_data.json').read_text());graphs=data['smoke' if a.smoke else 'confirmation'];o=P/('graph_smoke' if a.smoke else 'graph');o.mkdir(exist_ok=True)
 comps={'pattern':['attention_pattern'],'value':['v'],'pattern_value':['attention_pattern','v'],'query':['q'],'key':['k'],'query_key':['q','k'],'attention_output':['attention_out'],'mlp':['mlp_out'],'block_input':['block_input']}
 conditions=[]
 for src,dst in itertools.permutations(['raw','one','two'],2):
  for lname,ls in [('L1',[0]),('L2',[1]),('L12',[0,1])]:
   for cname,cs in comps.items():conditions.append((src,dst,lname,cname,tuple(I(site=l,component=c,mode='patch') for l in ls for c in cs)))
  conditions.append((src,dst,'L2_answer','pattern',(I(site=1,component='attention_pattern',mode='patch',positions=(34,)),)))
 names=[f'{s}_to_{d}/{l}/{c}' for s,d,l,c,_ in conditions]
 for modelname in a.models:
  seed={'A':6,'B':3,'C':4,'D':5,'E':7}[modelname];cp=B/f'backbones/local_control_L6_seed{seed}/best.pt';jp=P/f'local/{modelname}/controllers.pt'
  m,cfg,pay=load_checkpoint(cp,torch.device('cuda'));m.eval().requires_grad_(False);versions=[p._version for p in m.parameters()]
  assert sha(B/'locks/donors.pt') in json.loads((cp.parent/'summary.json').read_text())['extra_excluded_locks'].values()
  for fit in [1,2]:
   dest=o/f'{modelname}_fit{fit}.npz'
   if dest.exists():continue
   tic=time.time();maps={k:_load_controller(path=jp,name=f'seed{fit}_J_{k}_beh_rank256',device=torch.device('cuda')) for k in ['one','two']}
   for k in [1,2]:
    ident=json.loads((B/f'local/{modelname}/hop{k}_seed{fit}/training_identity.json').read_text());assert str(B/'locks/donors.pt') in str(ident['arguments']['train_exclude_locks'])
   records=[];bases=[];pres=[];labels=[];checks=[]
   for off in range(0,len(graphs),16):
    g=torch.tensor(graphs[off:off+16],device='cuda').repeat_interleave(10,0);cur=torch.arange(10,device='cuda').repeat(len(g)//10);start=cur.clone()
    for _ in range(8):start=g.argsort(-1).gather(1,start[:,None])[:,0]
    tok,_,_,_=fixed_depth_batch(cfg,len(g),torch.device('cuda'),path_positions=10,successors=g,start=start)
    h=m.token_embed(tok)+m.pos_embed[None]
    for t in range(6):h=m.apply_loop(h,loop_index=t)
    states={'raw':h,**{k:apply_vector_map(h,positions=tuple(range(cfg.seq_len)),controller=j) for k,j in maps.items()}}
    runs={k:run_instrumented_state(m,x,loop_indices=(6,)) for k,x in states.items()}
    target1=g.gather(1,cur[:,None])[:,0];target2=g.gather(1,target1[:,None])[:,0];labels.append(torch.stack([cur,target1,target2],-1).cpu().numpy())
    bases.append(torch.stack([runs[k][0].argmax(-1) for k in states],-1).cpu().numpy());pres.append(torch.stack([m.unembed(m.ln_final(x[:,-1]))[:,:10].argmax(-1) for x in states.values()],-1).cpu().numpy())
    if off==0:
     for k,x in states.items():
      direct=m.unembed(m.ln_final(m.apply_loop(x,loop_index=6)[:,-1]))[:,:10];assert torch.equal(direct.argmax(-1),runs[k][0].argmax(-1))
      for cname,cs in comps.items():
       iv=tuple(I(site=l,component=c,mode='patch') for l in [0,1] for c in cs)
       y,_=run_instrumented_state(m,x,loop_indices=(6,),interventions=iv,donor_trace=runs[k][1]);err=float((y-runs[k][0]).abs().max());assert torch.equal(y.argmax(-1),runs[k][0].argmax(-1));checks.append({'state':k,'component':cname,'max_logit_error':err})
    preds=[]
    for src,dst,l,c,iv in conditions:
     y,_=run_instrumented_state(m,states[dst],loop_indices=(6,),interventions=iv,donor_trace=runs[src][1]);preds.append(y.argmax(-1).cpu().numpy())
    records.append(np.stack(preds));print(json.dumps({'model':modelname,'fit':fit,'graphs_done':min(off+16,len(graphs)),'seconds':time.time()-tic,'peak_mib':torch.cuda.max_memory_allocated()/2**20}),flush=True)
    if torch.cuda.mem_get_info()[0]<16*2**30:raise RuntimeError('reserve breached')
   assert versions==[p._version for p in m.parameters()]
   np.savez_compressed(dest,predictions=np.concatenate(records,axis=1),native=np.concatenate(bases),pre=np.concatenate(pres),labels=np.concatenate(labels),conditions=names,graphs=np.array(graphs))
   save(dest.with_suffix('.json'),{'status':'complete','pid':os.getpid(),'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'backbone':str(cp),'checkpoint_sha256':sha(cp),'controllers_sha256':sha(jp),'data_sha256':sha(P/'graph_data.json'),'code_sha256':sha(__file__),'fit':fit,'n':len(graphs)*10,'checks':checks,'weights_unchanged':True,'seconds':time.time()-tic,'peak_mib':torch.cuda.max_memory_allocated()/2**20})
  del m;torch.cuda.empty_cache()
if __name__=='__main__':main()
