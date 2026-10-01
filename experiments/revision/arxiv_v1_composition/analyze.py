from pathlib import Path
import numpy as np,json
P=Path(__file__).parent;out={};boot=np.random.default_rng(2026093002).multinomial(512,np.ones(512)/512,size=3000)
old=np.load(P.parent/'long_composition/A_fit1.npz')['graphs']
new=json.loads((P/'results/data.json').read_text())
assert not set(map(tuple,old)) & set(map(tuple,new['graphs']))
for name,seed in zip('ABCDE',[6,3,4,5,7]):
 if not (P/'results'/f'{name}_fit2.npz').exists():continue
 ds=[np.load(P/'results'/f'{name}_fit{k}.npz') for k in [1,2]];d=ds[0];l=d['labels'];seq=d['sequences'];N=len(l);assert N==5120
 assert np.array_equal(ds[0]['labels'],ds[1]['labels'])
 assert np.array_equal(ds[0]['sequences'],ds[1]['sequences'])
 assert np.array_equal(ds[0]['graphs'],ds[1]['graphs'])
 assert np.array_equal(d['graphs'],np.asarray(new['graphs']))
 assert np.array_equal(seq,np.asarray(new['sequences']))
 mask=(np.diff(np.sort(l[:,:5],axis=1),axis=1)!=0).all(1);target=l[:,seq.cumsum(1)]
 metrics={}
 for kind in ['predictions','pre','omit_current','native','first_only','copy_u']:
  xs=[]
  for z in ds:
   pred=(np.broadcast_to(z['native'][:,None,:],target.shape) if kind=='native' else z['first_only'][:,seq[:,0]-1,:] if kind=='first_only' else np.broadcast_to(l[:,0,None,None],target.shape) if kind=='copy_u' else z[kind]);xs.append(pred==target)
  ok=np.stack(xs);groups={}
  for group,sl in [('one',slice(0,1)),('two',slice(1,2)),('mixed',slice(2,None))]:
   q=ok[:,:,sl,:].mean((0,2));prefix=np.cumprod(ok[:,:,sl,:],-1).mean((0,2));num=(q*mask[:,None]).reshape(512,10,8).sum(1);den=mask.reshape(512,10).sum(1);bs=(boot@num)/(boot@den)[:,None]
   nonreturn=(target[:,sl,:]!=l[:,0,None,None]) & mask[:,None,None]
   strict=(ok[:,:,sl,:]*nonreturn[None]).sum((0,1,2))/(nonreturn.sum((0,1))*2)
   groups[group]={'endpoint_pct':(q[mask].mean(0)*100).tolist(),'prefix_pct':(prefix[mask].mean(0)*100).tolist(),'ci95_pct':(np.quantile(bs,[.025,.975],axis=0).T*100).tolist(),'nonreturn_pct':(strict*100).tolist(),'nonreturn_n':nonreturn.sum((0,1)).tolist(),'all_endpoint_pct':(q.mean(0)*100).tolist()}
  metrics[kind]=groups
 two=[]
 for a,b in [(1,1),(1,2),(2,1),(2,2)]:
  j=np.flatnonzero((seq[:,:2]==[a,b]).all(1))[0];t=target[:,j,1]
  two.append({'order':f'{a}-{b}','accuracy':float(np.mean([(z['predictions'][mask,j,1]==t[mask]).mean() for z in ds])*100),'omit_second':float(np.mean([(z['omit_current'][mask,j,1]==t[mask]).mean() for z in ds])*100),'pre_F':float(np.mean([(z['pre'][mask,j,1]==t[mask]).mean() for z in ds])*100)})
 out[name]={'seed':seed,'n':int(mask.sum()),'graphs':512,'fits':2,'metrics':metrics,'two_step':two}
expected=json.loads((P/'summary.json').read_text())
def compare(a,b):
 if isinstance(a,dict):
  assert a.keys()==b.keys()
  for key in a:compare(a[key],b[key])
 elif isinstance(a,list):
  assert len(a)==len(b)
  for x,y in zip(a,b):compare(x,y)
 elif isinstance(a,(int,float)) and not isinstance(a,bool):
  assert np.isclose(a,b,rtol=0,atol=1e-8),(a,b)
 else:assert a==b,(a,b)
compare(out,expected)
print(json.dumps({k:{'seed':v['seed'],'n':v['n'],'two':v['two_step'],'end8':{g:round(v['metrics']['predictions'][g]['endpoint_pct'][-1],2) for g in ['one','two','mixed']},'prefix8':{g:v['metrics']['predictions'][g]['prefix_pct'][-1] for g in ['one','two','mixed']}} for k,v in out.items()},indent=2))
