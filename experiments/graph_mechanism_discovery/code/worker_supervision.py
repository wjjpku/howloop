import argparse,json,sys,subprocess,random
from pathlib import Path
import torch
from reasoning_loop.paper2027_graph_g4_protocol import sample_training_permutations,merged_forbidden_codes
R=Path(__file__).resolve().parents[1];C=R/'code';p=argparse.ArgumentParser();p.add_argument('--shard',type=int,required=True);a=p.parse_args();torch.set_num_threads(4);sup=R/'supervision';sup.mkdir(exist_ok=True)
# Each shard independently computes identical immutable data; atomic replace.
torch.manual_seed(2026092321);forbidden=merged_forbidden_codes(sorted((R/'locks').glob('*.pt')),node_count=10);pool=[];seen=set()
while len(pool)<512:
 for row in sample_training_permutations(batch_size=512,node_count=10,forbidden_codes=forbidden,device=torch.device('cpu')).tolist():
  if tuple(row) not in seen:pool.append(row);seen.add(tuple(row))
  if len(pool)==512:break
datasets=json.loads((R/'datasets.json').read_text());ds={};rng=random.Random(2026092322)
for name,graphs in [('train',pool),('validation',datasets['selection']),('test',datasets['confirmation']),('cycle10',datasets['rings'][:128])]:
 orders=[]
 for _ in graphs:
  order=list(range(10));rng.shuffle(order);orders.append(order)
 ds[name]={'graphs':graphs,'order':orders}
f=sup/f'datasets_{a.shard}.tmp';f.write_text(json.dumps(ds));f.replace(sup/'datasets.json')
def run(cmd):print(json.dumps(list(map(str,cmd))),flush=True);subprocess.run(list(map(str,cmd)),cwd=C,check=True)
mode=['final_only','per_loop'][a.shard];out=sup/mode
if not (out/'summary.json').exists():
 run([sys.executable,'-u','-m','reasoning_loop.n10_supervision_backbone','--supervision',mode,'--out-dir',out,'--selection-lock',R/'locks/selection.pt','--final-test-lock',R/'locks/confirmation.pt','--extra-exclude-locks',*[R/'locks'/f'{x}.pt' for x in ['discovery','rings','smoke','donors']],'--seed',3,'--loops',8,'--depth-mode','fixed','--amp'])
for target in [9,10,12,15]:
 for fit in [0,1,2]:
  name=f'{mode}_n{target}_seed{fit}';dest=sup/'fixed8_runs'/name
  if (dest/'results.json').exists():continue
  run([sys.executable,'-u','train_supervision_J.py','--checkpoint',out/'final.pt','--seed',fit,'--target',target,'--steps',5000,'--run-name',name])
