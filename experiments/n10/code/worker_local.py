import argparse,json,os,subprocess,sys,hashlib,gzip
from pathlib import Path
R=Path(__file__).resolve().parents[1];C=R/'code';ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);a=ap.parse_args()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(cmd):print(json.dumps({'event':'launch','command':list(map(str,cmd))}),flush=True);subprocess.run(list(map(str,cmd)),cwd=C,check=True)
py=sys.executable
for ix,seed in enumerate([6,3,4,5,7]):
 if ix%2!=a.shard:continue
 name='ABCDE'[ix];checkpoint=R/'backbones'/f'local_control_L6_seed{seed}'/'best.pt';out=R/'local'/name;out.mkdir(parents=True,exist_ok=True)
 if (out/'complete.json').exists():continue
 cp=__import__('torch').load(checkpoint,map_location='cpu',weights_only=False)
 run([py,'evaluate_native.py','--checkpoint',checkpoint,'--out',out/'native'])
 for hop in [1,2]:
  for fit in [1,2]:
   sub=out/f'hop{hop}_seed{fit}'
   if not (sub/'summary.json').exists():
    run([py,'-u','-m','reasoning_loop.paper2027_d8l6_s1','train','--checkpoint',checkpoint,'--out-dir',sub,'--target-hop',hop,'--seed',fit,'--validation-seed',10000+fit,'--validation-lock',R/'locks/selection.pt','--train-exclude-locks',*sorted((R/'locks').glob('*.pt')),'--validation-batch-size','500'])
   run([py,'-m','reasoning_loop.paper2027_d8l6_s1','evaluate','--checkpoint',checkpoint,'--controller',sub/'best_controller.pt','--locked-test',R/'locks/confirmation.pt','--out-dir',sub/'evaluation','--test-seed',20260923,'--batch-size',250])
 run([py,'export_maps.py','--checkpoint',checkpoint,'--runs',out])
 def battery(head,panel):
  dest=out/f'head{head}'
  run([py,'-u','evaluate_battery.py','--looplus',C,'--checkpoint',checkpoint,'--checkpoint-sha256',sha(checkpoint),'--checkpoint-step',cp['step'],'--controllers',out/'controllers.pt','--controller-sha256',sha(out/'controllers.pt'),'--datasets',R/'datasets.json','--family','N10_pure_CE','--backbone-name',name,'--head',head,'--panel',panel,'--seeds',1,2,'--out',dest])
  return dest/panel
 scores=[]
 for head in range(4):
  folder=battery(head,'discovery');totals={f:[0,0] for f in [1,2]}
  with gzip.open(folder/'events.jsonl.gz','rt') as f:
   for line in f:
    row=json.loads(line)
    if row['kind']=='cross_graph_address' and row['receiver']=='J_one' and row['condition']==f'J_one_pat_H{head}':
     fit=row['controller_seed'];totals[fit][0]+=sum(int(ok and pred==target) for ok,pred,target in zip(row['eligible'],row['prediction'],row['target']));totals[fit][1]+=sum(row['eligible'])
  values=[k/n if n else None for k,n in totals.values()];score=sum(v for v in values if v is not None)/len(values) if all(v is not None for v in values) else -1
  scores.append({'head':head,'score':score,'counts':totals})
 head=max(scores,key=lambda x:(x['score'],-x['head']))['head']
 (out/'head_selection.json').write_text(json.dumps({'locked_head':head,'rule':'mean eligible J_one pattern CF across two fits on discovery; missing fit => -1; tie smallest head','scores':scores},indent=2))
 battery(head,'confirmation')
 (out/'complete.json').write_text(json.dumps({'status':'complete','checkpoint_sha256':sha(checkpoint),'head':head}))
