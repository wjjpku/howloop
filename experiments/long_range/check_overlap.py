import sys,json,os
from pathlib import Path
import numpy as np,torch
R=Path('/data/wujiaju/paper2027_confirmatory/graph_g4_disjoint_v3');sys.path.insert(0,str(R/'analysis_code'))
from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch,load_checkpoint
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state
from reasoning_loop.paper2027_graph_g3_controller import _load_controller,_initial
D=Path('/data/wujiaju/fig6_retest_20260924');cases=json.loads((D/'overlap_differences.json').read_text());graphs=np.load(D/'graph/successors.npy');old=np.array(json.loads((D/'old_lock_successors.json').read_text()));result=[];torch.set_num_threads(2)
@torch.inference_mode()
def run():
 for case in cases:
  seed=case['seed'];rep=int(case['mode'].split('seed')[1].split('/')[0]);cp=R/f'backbones/seed{seed}/graphpath_N8_D8_d256_B2_L8_seed{seed}/final.pt';model,cfg,_=load_checkpoint(cp,torch.device('cuda'));js=[_load_controller(path=R/f'g4_controllers/seed{seed}/rank48_seed{k}/best_controller.pt',checkpoint=cp,device=torch.device('cuda'))[0] for k in [1,2]]
  preds={};logits={}
  for name,array,key,batched in [('new_joint',graphs,'new_graph',True),('new_separate',graphs,'new_graph',False),('old_separate',old,'old_graph',False)]:
   ix=case[key];first=ix//32*32;subset=torch.tensor(array[first:first+32],device='cuda');tokens,targets,_,_=fixed_depth_batch(cfg,256,torch.device('cuda'),path_positions=40,successors=subset.repeat_interleave(8,0),start=torch.arange(8,device='cuda').repeat(32));state=_initial(model,tokens)
   if batched:state=torch.cat([state,state,state])
   for t in range(case['call']):
    if batched:prepared=torch.cat([state[:256],js[0](state[256:512]),js[1](state[512:])])
    else:prepared=js[rep-1](state)
    state=model.apply_loop(prepared,loop_index=t)
   if batched:state=state[rep*256:(rep+1)*256]
   l=logits_from_raw_state(model,state)[(ix-first)*8:(ix-first+1)*8];gold=targets[(ix-first)*8:(ix-first+1)*8,case['call']-1];preds[name]={'correct':int(l.argmax(-1).eq(gold).sum()),'predictions':l.argmax(-1).tolist(),'targets':gold.tolist()};logits[name]=l
  result.append({**case,'recheck':preds,'max_logit_difference':float((logits['new_joint']-logits['old_separate']).abs().max()),'old_new_separate_max_error':float((logits['new_separate']-logits['old_separate']).abs().max())});print(json.dumps(result[-1]),flush=True)
run();(D/'overlap_recheck.json').write_text(json.dumps(result,indent=2))
