import os,sys,time,json,hashlib,itertools,argparse
from pathlib import Path
import torch,numpy as np
ROOT=Path('/data/wujiaju/paper2027_confirmatory/graph_g4_disjoint_v3');sys.path.insert(0,str(ROOT/'analysis_code'))
from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch,load_checkpoint
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state
from reasoning_loop.paper2027_graph_g3_controller import _load_controller,_initial
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();O=Path(a.out);O.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rng=np.random.default_rng(2026092402);universe=np.array(list(itertools.permutations(range(8))),dtype=np.int64);ids=rng.choice(len(universe),size=8192,replace=False);graphs=universe[ids];np.save(O/'successors.npy',graphs)
seeds=[100] if a.smoke else list(range(100,112));ng=32 if a.smoke else 8192;graphs=graphs[:ng];bs=256;calls=40
man={'status':'running','pid':os.getpid(),'gpu':os.getenv('CUDA_VISIBLE_DEVICES'),'code_sha256':sha(__file__),'graph_seed':2026092402,'graphs':ng,'unique_population':40320,'sampling':'uniform without replacement, no topology or prediction filters','graph_sha256':sha(O/'successors.npy'),'backbone_seeds':seeds,'J_seeds':[1,2],'max_call':calls,'starts':8,'schedule':'Raw from h0; J before every F call from h0, matching archived G4 evaluation','metric':'ordinary fraction of all node predictions correct; no collision exclusions','training_graph_disjoint':False,'models':[]}
(O/'manifest.json').write_text(json.dumps(man,indent=2));start=time.time()
@torch.inference_mode()
def run():
 for seed in seeds:
  cp=ROOT/f'backbones/seed{seed}/graphpath_N8_D8_d256_B2_L8_seed{seed}/final.pt';model,cfg,payload=load_checkpoint(cp,torch.device('cuda'));model.eval();js=[];record={'seed':seed,'checkpoint':str(cp),'checkpoint_sha256':sha(cp),'controllers':[]}
  for rep in [1,2]:
   jp=ROOT/f'g4_controllers/seed{seed}/rank48_seed{rep}/best_controller.pt';j,jpay=_load_controller(path=jp,checkpoint=cp,device=torch.device('cuda'));js.append(j);record['controllers'].append({'path':str(jp),'sha256':sha(jp)})
  counts=np.zeros((3,calls,ng),dtype=np.uint8)
  for off in range(0,ng,bs//8):
   subset=torch.tensor(graphs[off:off+bs//8],device='cuda');num=len(subset);succ=subset.repeat_interleave(8,dim=0);starts=torch.arange(8,device='cuda').repeat(num)
   tokens,targets,_,_=fixed_depth_batch(cfg,num*8,torch.device('cuda'),path_positions=calls,successors=succ,start=starts);s0=_initial(model,tokens);states=[s0.clone() for _ in range(3)];tally=[]
   for t in range(calls):
    prepared=torch.cat([states[0],js[0](states[1]),js[1](states[2])]);joined=model.apply_loop(prepared,loop_index=t);states=list(joined.split(num*8));pred=logits_from_raw_state(model,joined).argmax(-1).reshape(3,num,8);gold=targets[:,t].reshape(num,8);tally.append(pred.eq(gold[None]).sum(-1))
   counts[:,:,off:off+num]=torch.stack(tally,dim=1).cpu().numpy().astype(np.uint8)
   if a.smoke:
    errs=[]
    for mode in range(3):
     state=s0.clone();ref=[]
     for t in range(calls):
      if mode:state=js[mode-1](state)
      state=model.apply_loop(state,loop_index=t);pred=logits_from_raw_state(model,state).argmax(-1).reshape(num,8);ref.append(pred.eq(targets[:,t].reshape(num,8)).sum(-1))
     rr=torch.stack(ref).cpu().numpy();assert np.array_equal(rr,counts[mode,:,:num]);errs.append(True)
    man['separate_branch_check']=errs
   if off%(32*16)==0:print(json.dumps({'seed':seed,'graphs_done':off+num,'elapsed':round(time.time()-start,1),'peak_mib':torch.cuda.max_memory_allocated()/2**20}),flush=True)
  np.savez_compressed(O/f'seed{seed}.npz',correct_counts=counts)
  man['models'].append(record);(O/'manifest.json').write_text(json.dumps(man,indent=2));print(json.dumps({'seed_done':seed,'means_9_16':counts[:,8:16].mean(axis=(1,2)).tolist(),'means_17_32':counts[:,16:32].mean(axis=(1,2)).tolist()}),flush=True)
  del model,js,states,joined;torch.cuda.empty_cache()
run();man.update(status='complete',elapsed=time.time()-start,peak_mib=torch.cuda.max_memory_allocated()/2**20);(O/'manifest.json').write_text(json.dumps(man,indent=2))
