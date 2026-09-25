import argparse,csv,gzip,json,hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--names',default='A');args=ap.parse_args();names=args.names
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'analysis';OUT.mkdir(exist_ok=True)
rng=np.random.default_rng(2026092317);weights=rng.multinomial(512,np.full(512,1/512),size=5000)
def stats(v,m,g):
 n=np.bincount(g,weights=m.astype(float),minlength=512);x=np.bincount(g,weights=v*m,minlength=512)
 if n.sum()==0:return {'n':0,'graphs':0,'mean':None,'ci95':None}
 den=weights@n;num=weights@x;ok=den>0;bs=num[ok]/den[ok]
 return {'n':int(n.sum()),'graphs':int((n>0).sum()),'mean':float(x.sum()/n.sum()),'ci95':np.quantile(bs,[.025,.975]).tolist(),'valid_draws':int(ok.sum())}
results={};flat=[];integrity=[]
for name in names:
 head=2
 run=ROOT;mf=json.loads((run/'evaluation/confirmation/manifest.json').read_text());assert mf['status']=='complete';assert mf['head']==head;assert mf['seeds']==[1,2];assert mf['graphs']==512
 p=run/'evaluation/confirmation/events.jsonl.gz';assert hashlib.sha256(p.read_bytes()).hexdigest()==mf['predictions_sha256'];groups=defaultdict(lambda:defaultdict(list))
 with gzip.open(p,'rt') as f:
  for line in f:
   d=json.loads(line);kind=d['kind'];rec=d['receiver'];cond=d['condition']
   keep=(kind=='baseline' and rec in ['identity','J_one','J_two']) or (kind=='cross_graph_address' and rec=='J_one' and cond in [f'J_one_pat_H{head}',f'J_one_ctx_H{head}']) or (kind=='unconditional_swap' and rec=='J_one' and cond in ['rescue_H0123','damage_H0123']) or (kind=='rescue' and rec=='J_one')
   if not keep:continue
   key=(d['controller_seed'],kind,rec,cond)
   for k,v in d.items():
    if isinstance(v,list):groups[key][k].extend(v)
 data={k:{n:np.array(v) for n,v in d.items()} for k,d in groups.items()}
 def get(s,k,r,c):return data[(s,k,r,c)]
 def cross(s,c):return get(s,'cross_graph_address','J_one',f'J_one_{c}_H{head}')
 common=cross(1,'pat')['eligible']&cross(2,'pat')['eligible'];results[name]={}
 for seed in (1,2):
  b={};r={'behavior':{},'cross_graph':{},'restoration':{}}
  for rec in ['identity','J_one','J_two']:
   d=get(seed,'baseline',rec,'clean');g=d['graph_ids'];cur=d['current'];one=d['one_target'];two=d['two_target'];assert len(cur)==5120;assert np.array_equal(g,np.repeat(np.arange(512),10));mask=(cur!=one)&(cur!=two)&(one!=two);b[rec]=d
   r['behavior'][rec]={}
   for cohort,m in [('all',np.ones(len(cur),dtype=bool)),('distinct_labels',mask)]:
    r['behavior'][rec][cohort]={}
    for rd,pred in [('pre_F',d['pre_prediction']),('post_F',d['prediction'])]:
     for target,t in [('current',cur),('one',one),('two',two)]:r['behavior'][rec][cohort][rd+'_'+target]=stats(pred==t,m,g)
  p=cross(seed,'pat');c=cross(seed,'ctx');assert np.array_equal(p['eligible'],c['eligible']);assert np.array_equal(p['graph_ids'],c['graph_ids'])
  pc=(p['prediction']==p['target']).astype(float);cc=(c['prediction']==c['target']).astype(float);pd=(p['prediction']==p['donor_target']).astype(float);cd=(c['prediction']==c['donor_target']).astype(float)
  for label,m in [('own_eligible',p['eligible']),('common_two_fits',common),('labels_only',p['labels_distinct'])]:
   r['cross_graph'][label]={k:stats(v,m,p['graph_ids']) for k,v in [('pattern_cf',pc),('context_cf',cc),('pattern_donor',pd),('context_donor',cd),('cf_difference',pc-cc),('donor_difference',cd-pd)]}
  mask=np.ones(len(pc),dtype=bool);flow={'candidates':len(pc)}
  for key in ['labels_distinct','base_pre_correct','base_post_correct','donor_pre_correct','donor_post_correct']:mask &=p[key];flow[key]=int(mask.sum())
  assert np.array_equal(mask,p['eligible']);r['filter_counts']=flow
  raw=b['identity'];j=b['J_one'];ids=j['graph_ids'];cur=j['current'];one=j['one_target'];two=j['two_target'];mask=(cur!=one)&(cur!=two)&(one!=two)
  rawhit=(raw['prediction']==one).astype(float);full=(j['prediction']==one).astype(float);swap=get(seed,'unconditional_swap','J_one','rescue_H0123');damage=get(seed,'unconditional_swap','J_one','damage_H0123');rescue=(swap['prediction']==one).astype(float);dam=(damage['prediction']==one).astype(float)
  nums=np.bincount(ids,weights=(rescue-rawhit)*mask,minlength=512);dens=np.bincount(ids,weights=(full-rawhit)*mask,minlength=512);den=weights@dens;num=weights@nums;ok=den>0
  r['pattern_swap']={k:stats(v,mask,ids) for k,v in [('native',rawhit),('full_J',full),('rescue',rescue),('damage',dam),('gain',rescue-rawhit)]};r['pattern_swap']['fraction_of_gain']={'mean':float(nums.sum()/dens.sum()) if dens.sum()>0 else None,'ci95':np.quantile(num[ok]/den[ok],[.025,.975]).tolist() if ok.any() else None,'valid_draws':int(ok.sum())}
  for condition in ['blocked']+[f'clean_context_H{i}' for i in range(4)]+['clean_context_H0123']:
   d=get(seed,'rescue','J_one',condition);m=d['eligible']&d['broken'];r['restoration'][condition]=stats(d['prediction']==d['target'],m,d['graph_ids'])
  results[name][str(seed)]=r
 integrity.append({'backbone':name,'events_sha256':mf['predictions_sha256'],'dataset_sha256':mf['dataset_sha256'],'controller_sha256':mf['controller_sha256']})
(OUT/'summary.json').write_text(json.dumps({'protocol':'n10_fresh_reserved_pairs','results':results,'bootstrap_draws':5000,'integrity':integrity},indent=2))
print(json.dumps(results))
