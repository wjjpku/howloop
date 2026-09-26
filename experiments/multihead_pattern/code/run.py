import core as R
from core import torch,np,json,os,time,dataclasses
from pathlib import Path
import argparse,itertools
P=Path(__file__).resolve().parents[1]
def specs():
 return [dict(id=f'{scope}_{scale}_{mask:03d}',scope=scope,scale=scale,mask=mask) for scope,scale,mask in itertools.product(['answer','all'],[0,1],range(256))]
def ivs(s,setting):
 iv=[]
 for layer in [0,1]:
  hs=tuple(i for i in range(4) if s['mask']&(1<<(4*layer+i)))
  if hs:iv.append(R.I(site=layer,component='attention_pattern',mode='patch',positions=(34,) if s['scope']=='answer' else None,heads=hs))
 if s['scale']:iv.append(R.I(site=setting['site'],component='residual_skip',mode='patch',positions=(34,)))
 return iv
@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--phase',required=True);a=ap.parse_args();torch.set_num_threads(4);n=a.model;phase=a.phase;out=P/phase;out.mkdir(exist_ok=True,parents=True)
 cp=R.B/f'backbones/local_control_L6_seed{dict(B=3,D=5,E=7)[n]}/best.pt';jp=R.D/f'local/{n}/controllers.pt';m,cfg,_=R.load_checkpoint(cp,torch.device('cuda'));m.eval().requires_grad_(False);ver=[p._version for p in m.parameters()]
 hashes={k:R.sha(v) for k,v in dict(checkpoint=cp,controllers=jp,data=P/'datasets.json',code=Path(__file__),core=P/'code/core.py',evaluator=P/'code/circuit_extended.py',prior_selection=P/'prior_selection.json').items()};prior=json.loads((R.D/f'graph/{n}_fit1.json').read_text());assert hashes['checkpoint']==prior['checkpoint_sha256'] and hashes['controllers']==prior['controllers_sha256']
 setting=json.loads((P/'prior_selection.json').read_text())[n]['targets']['attention_pattern'];ss=specs()
 if phase=='confirmation':
  select=json.loads((P/'selection.json').read_text())[n];ss=[s for s in ss if s['id'] in select['ids']];hashes['selection']=R.sha(P/'selection.json')
 if phase=='smoke':ss=[s for s in ss if s['mask'] in [0,1,16,17,15,240,255]]
 gs=json.loads((P/'datasets.json').read_text())[phase]
 for fit in ([1,2] if phase=='confirmation' else [1]):
  j=R._load_controller(path=jp,name=f'seed{fit}_J_one_beh_rank256',device=torch.device('cuda'));preds=[];wrong=[];labs=[];baselines=[];tic=time.time();R.save(out/f'{n}_{fit}_running.json',dict(pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],hashes=hashes,phase=phase,conditions=len(ss)))
  for off in range(0,len(gs),16):
   h,labels=R.batch(m,cfg,gs[off:off+16]);z=R.apply_vector_map(h,positions=tuple(range(35)),controller=j);yr,r=R.run(m,h,loop_indices=(6,));yj,jt=R.run(m,z,loop_indices=(6,));labs.append(labels.cpu().numpy());baselines.append(torch.stack([yr.argmax(-1),yj.argmax(-1)]).cpu().numpy())
   site=setting['site'];v=r.sites[site].hidden_in;scaled=v+(setting['alpha']-1)*R.center(v);don=R.changed_trace(jt,site,'hidden_in',scaled)
   idx=torch.arange(len(h),device='cuda').reshape(-1,10).roll(1,1).reshape(-1)
   wt=dataclasses.replace(jt,sites=[dataclasses.replace(t,attention_pattern=t.attention_pattern[idx]) for t in jt.sites]);wd=R.changed_trace(wt,site,'hidden_in',scaled)
   pp=[];ww=[]
   if off==0:
    assert torch.equal(m.unembed(m.ln_final(m.apply_loop(h,loop_index=6)[:,34]))[:,:10].argmax(-1),yr.argmax(-1))
    for scope in ['answer','all']:
     test=dict(scope=scope,scale=0,mask=255);ys,_=R.run(m,h,loop_indices=(6,),interventions=ivs(test,setting),donor_trace=r);assert torch.equal(ys,yr)
   for s in ss:
    yy,_=R.run(m,h,loop_indices=(6,),interventions=ivs(s,setting),donor_trace=don if s['scale'] else jt);pp.append(yy.argmax(-1).cpu().numpy())
    if phase=='confirmation':
     yy,_=R.run(m,h,loop_indices=(6,),interventions=ivs(s,setting),donor_trace=wd if s['scale'] else wt);ww.append(yy.argmax(-1).cpu().numpy())
   preds.append(np.array(pp))
   if ww:wrong.append(np.array(ww))
   print(json.dumps(dict(model=n,phase=phase,fit=fit,graphs=off+len(h)//10,seconds=round(time.time()-tic,1),peak_mib=torch.cuda.max_memory_allocated()/2**20)),flush=True)
   assert torch.cuda.mem_get_info()[0]>16*2**30
  assert ver==[p._version for p in m.parameters()] and hashes['checkpoint']==R.sha(cp) and hashes['controllers']==R.sha(jp)
  arr=dict(predictions=np.concatenate(preds,1),labels=np.concatenate(labs),baseline=np.concatenate(baselines,1),graphs=np.array(gs),conditions=np.array([s['id'] for s in ss]))
  if wrong:arr['wrong_current']=np.concatenate(wrong,1)
  np.savez_compressed(out/f'{n}_{fit}.npz',**arr);R.save(out/f'{n}_{fit}.json',dict(status='complete',hashes=hashes,settings=setting,seconds=time.time()-tic,same_run_exact=True,weights_unchanged=True,peak_mib=torch.cuda.max_memory_allocated()/2**20))
if __name__=='__main__':main()
