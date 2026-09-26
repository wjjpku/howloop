import argparse,json,random,re,hashlib
from pathlib import Path
from transformers import AutoTokenizer
N='Alice Bob Carol David Emma Frank Grace Henry Iris Jack'.split()
INTRO='Whenever someone receives the letter, they pass it to their designated recipient. The rules never change.'
def walk(g,s,k):
 for _ in range(k):s=g[s]
 return s
def cycle(r):
 n=N.copy();r.shuffle(n);return dict(zip(n,n[1:]+n[:1]))
def enc(tok,g,start,order):
 p=INTRO+'\n'+'\n'.join(f'{x} passes the letter to {g[x]}.' for x in order)+f'\nThe letter starts with {start}. After exactly 08 transfers, who has it? Answer directly with only the person\'s name.'
 text=tok.apply_chat_template([dict(role='user',content=p)],tokenize=False,add_generation_prompt=True)
 e=tok(text,add_special_tokens=False,return_offsets_mapping=True);slots=[]
 for x in order:
  rule=f'{x} passes the letter to {g[x]}.';i=text.index(rule);j=i+len(x)+len(' passes the letter to ')
  for a,b in [(i,i+len(x)),(j,j+len(g[x]))]:slots.append([k for k,(u,v) in enumerate(e['offset_mapping']) if v>a and u<b])
 a=text.index('The letter starts with ')+len('The letter starts with ');slots.append([k for k,(u,v) in enumerate(e['offset_mapping']) if v>a and u<a+len(start)])
 return dict(prompt=p,ids=e['input_ids'],slots=slots)
a=argparse.ArgumentParser();a.add_argument('--out',required=True);a.add_argument('--seed',type=int,default=2026092301);a.add_argument('--pairs',type=int,default=8);a.add_argument('--exclude');args=a.parse_args()
tok=AutoTokenizer.from_pretrained('/data/paperexperiment/models/Ouro-2.6B',local_files_only=True);r=random.Random(args.seed);out=[];seen=set();attempt=0
if args.exclude:
 for x in json.loads(Path(args.exclude).read_text())['pairs']:
  for field in ['base_graph','source_graph']:seen.add(tuple(x[field][n] for n in N))
while len(out)<args.pairs:
 attempt+=1;b=cycle(r);s=cycle(r);sb=r.choice(N);ss=r.choice(N);yb=walk(b,sb,8);ys=walk(s,ss,8);last=walk(s,ss,7);cf=b[last]
 if len({yb,ys,cf})<3:continue
 sig=lambda g:tuple(g[n] for n in N)
 if sig(b) in seen or sig(s) in seen or sig(b)==sig(s):continue
 order=N.copy();r.shuffle(order);eb=enc(tok,b,sb,order);es=enc(tok,s,ss,order)
 if len(eb['ids'])!=len(es['ids']) or eb['slots']!=es['slots']:continue
 tids=[tok.encode(x,add_special_tokens=False)[0] for x in [yb,ys,cf]]
 if len(set(tids))<3:continue
 seen.update([sig(b),sig(s)]);out.append(dict(index=len(out),base=eb,source=es,base_graph=b,source_graph=s,base_start=sb,source_start=ss,base_answer=yb,source_answer=ys,counterfactual=cf,source_last_holder=last,secondary_start_counterfactual=walk(b,ss,8),target_ids=tids))
p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(dict(seed=args.seed,attempts=attempt,pairs=out),indent=2));print(json.dumps(dict(pairs=len(out),attempts=attempt,lengths=[len(x['base']['ids']) for x in out])))
