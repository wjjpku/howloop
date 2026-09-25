import argparse,json,hashlib
from pathlib import Path
import torch
from reasoning_loop.graph_path_depth_circuit import load_checkpoint,fixed_depth_batch
from reasoning_loop.graph_path_temporal_intervention import logits_from_raw_state
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda');a=p.parse_args();torch.set_num_threads(4)
R=Path(__file__).resolve().parents[1]; d=torch.device(a.device);m,c,cp=load_checkpoint(a.checkpoint,d);m.eval();assert c.node_count==10;datasets=json.loads((R/'datasets.json').read_text());outputs={}
with torch.inference_mode():
 for split in ['confirmation','rings']:
  predictions=[];probs=[];paths=[]
  for lo in range(0,len(datasets[split]),20):
   g=torch.tensor(datasets[split][lo:lo+20],device=d).repeat_interleave(10,0);start=torch.arange(10,device=d).repeat(len(g)//10)
   tokens,path,*_=fixed_depth_batch(c,len(g),d,path_positions=16,successors=g,start=start)
   h=m.token_embed(tokens)+m.pos_embed[None]; pp=[]; qq=[]
   for call in range(17):
    if call:h=m.apply_loop(h,loop_index=call-1)
    logits=logits_from_raw_state(m,h);pp.append(logits.argmax(-1).cpu());qq.append(logits.softmax(-1).cpu())
   predictions.append(torch.stack(pp,1));probs.append(torch.stack(qq,1));paths.append(torch.cat([start[:,None],path],1).cpu())
  outputs[split]={'predictions':torch.cat(predictions),'probabilities':torch.cat(probs),'targets':torch.cat(paths)}
a.out.mkdir(parents=True,exist_ok=True);torch.save(outputs,a.out/'native.pt')
summary={'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'step':cp['step'],'node_count':10,'loops':c.max_loops,'splits':{}}
for split,v in outputs.items():
 p,t=v['predictions'],v['targets'];summary['splits'][split]={'endpoint_accuracy_by_call':p.eq(t[:,8,None]).float().mean(0).tolist(),'category_match':[p.eq(t[:,k,None]).float().mean(0).tolist() for k in range(10)],'examples':len(p)}
(a.out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps({'event':'native_complete','out':str(a.out)}))
