import os,sys,json,time,hashlib,argparse,dataclasses
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parents[1];B=Path('/data/wujiaju/n10_migration_20260923');D=Path('/data/wujiaju/dense_affine_mechanism_20260925')
sys.path.insert(0,str(B/'code'))
from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch,load_checkpoint
from reasoning_loop.graph_path_functional_circuit import run_instrumented_state as original
from reasoning_loop.graph_path_jump_controller import apply_vector_map
from reasoning_loop.graph_path_jump_controller_causal_switch import _load_controller
import circuit_extended as C
I=C.FunctionalIntervention
old_token=C._apply_token_tensor;old_head=C._apply_head_tensor;old_pattern=C._apply_attention_pattern

def fast_token(value,*,site,component,interventions,donor):
 chosen=[i for i in interventions if i.site==site and i.component==component]
 if not chosen:return value
 result=value.clone()
 for i in chosen:
  assert i.mode in ['patch','zero'];pos=slice(None) if i.positions is None else list(i.positions)
  result[:,pos]=0 if i.mode=='zero' else donor[:,pos]
 return result

def fast_head(value,*,site,component,interventions,donor):
 chosen=[i for i in interventions if i.site==site and i.component==component]
 if not chosen:return value
 result=value.clone()
 for i in chosen:
  heads=list(range(value.shape[1])) if i.heads is None else list(i.heads);pos=list(range(value.shape[2])) if i.positions is None else list(i.positions)
  hi=torch.tensor(heads,device=value.device)[:,None];pi=torch.tensor(pos,device=value.device)[None,:]
  result[:,hi,pi,:]=0 if i.mode=='zero' else donor[:,hi,pi,:]
 return result

def fast_pattern(value,*,site,interventions,donor):
 for i in interventions:
  if i.site==site and i.component=='attention_pattern':assert i.source_positions is None and i.dynamic_source_positions is None
 return fast_head(value,site=site,component='attention_pattern',interventions=interventions,donor=donor)
C._apply_token_tensor=fast_token;C._apply_head_tensor=fast_head;C._apply_attention_pattern=fast_pattern
run=C.run_instrumented_state
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2))
def center(x):return x-x.mean(-1,keepdim=True)
def changed_trace(trace,site,key,value):
 sites=list(trace.sites);sites[site]=dataclasses.replace(sites[site],**{key:value});return dataclasses.replace(trace,sites=sites)
def rolled(trace):
 return dataclasses.replace(trace,sites=[dataclasses.replace(s,**{f.name:getattr(s,f.name).roll(10,0) for f in dataclasses.fields(s) if torch.is_tensor(getattr(s,f.name))}) for s in trace.sites])
def groups():
 g=C.explicit_depth_position_groups(10)
 return dict(all=tuple(range(35)),answer=(34,),nonanswer=tuple(range(34)),graph=g['graph'],source=g['source'],destination=g['destination'],metadata=g['query_metadata'])
def specifications():
 ss=[]
 def add(family,**kw):ss.append(dict(family=family,**kw))
 for base in ['raw','J','full_skip','pattern']:add('control',base=base)
 for site in [-1,0,1]:
  for group in (['all','answer','nonanswer'] if site==-1 else list(groups())):
   for alpha in [0,.05,.1,.2,.4,.7]:add('scalar',site=site,group=group,alpha=alpha)
 for site in [0,1]:
  for comp in ['attention_pattern','q','k','v','head_context']:
   for head in [0,1,2,3,-1]:add('head',site=site,component=comp,head=head)
 for site in [0,1]:
  for comp in ['k','v']:
   for group in ['graph','all']:
    for head in [0,1,2,3,-1]:add('head',site=site,component=comp,head=head,group=group)
 for site in [0,1]:
  for alpha in [0,.05,.1,.2,.4,.7,1]:
   for comp in ['attention_pattern','head_context']:
    for head in [0,1,2,3,-1]:add('scalar_head',site=site,alpha=alpha,component=comp,head=head)
 for pos in range(34):add('positions',positions=[34,pos])
 for group in ['graph','source','destination','metadata']:add('positions',positions=sorted(set([34,*groups()[group]])))
 for rank in [1,2,4,8,16,32]:
  for group in ['answer','all']:
   for site in [-1,0]:
    for pat in [False,True]:add('lowrank',rank=rank,group=group,site=site,pattern=pat)
 for site in [0,1]:
  for direction in ['current','node_span']:
   for strength in [.5,1,2,4]:
    for pat in [False,True]:add('readout',site=site,direction=direction,strength=strength,pattern=pat)
 for i,s in enumerate(ss):s['id']=f'c{i:03d}'
 return ss

def batch(m,cfg,gs):
 g=torch.tensor(gs,device='cuda').repeat_interleave(10,0);u=torch.arange(10,device='cuda').repeat(len(gs));start=u.clone()
 for _ in range(8):start=g.argsort(-1).gather(1,start[:,None])[:,0]
 tok,*_=fixed_depth_batch(cfg,len(g),torch.device('cuda'),path_positions=10,successors=g,start=start)
 h=m.token_embed(tok)+m.pos_embed[None]
 for t in range(6):h=m.apply_loop(h,loop_index=t)
 one=g.gather(1,u[:,None])[:,0];two=g.gather(1,one[:,None])[:,0]
 return h,torch.stack([u,one,two],-1)

def patch(m,h,z,r,jtrace,basis,nodebasis,s,shuffle=False):
 donor=rolled(jtrace) if shuffle else jtrace
 delta=(z-h).roll(10,0) if shuffle else z-h
 x=h;iv=[];fam=s['family'];site=s.get('site',0)
 if fam=='control':
  if s['base']=='J':x=z
  elif s['base']=='full_skip':iv=[I(site=0,component='residual_skip',mode='patch')]
  elif s['base']=='pattern':iv=[I(site=l,component='attention_pattern',mode='patch') for l in [0,1]]
 elif fam in ['scalar','scalar_head']:
  group='answer' if fam=='scalar_head' else s['group'];pos=groups()[group]
  v=h if site==-1 else r.sites[site].hidden_in
  replacement=v+(s['alpha']-1)*center(v)
  if site==-1:x=h.clone();x[:,pos]=replacement[:,pos]
  else:
   donor=changed_trace(donor,site,'hidden_in',replacement);iv.append(I(site=site,component='residual_skip',mode='patch',positions=pos))
  if fam=='scalar_head':iv.append(I(site=1,component=s['component'],mode='patch',positions=(34,),heads=None if s['head']==-1 else (s['head'],)))
 elif fam=='head':iv.append(I(site=site,component=s['component'],mode='patch',positions=groups()[s.get('group','answer')],heads=None if s['head']==-1 else (s['head'],)))
 elif fam=='positions':
  x=h.clone();pos=s['positions'];x[:,pos]=h[:,pos]+delta[:,pos]
 elif fam=='lowrank':
  v=basis[:,:s['rank']];dd=delta@v@v.T;pos=groups()[s['group']]
  if site==-1:x=h.clone();x[:,pos]=h[:,pos]+dd[:,pos]
  else:
   donor=changed_trace(donor,0,'hidden_in',h+dd);iv.append(I(site=0,component='residual_skip',mode='patch',positions=pos))
  if s['pattern']:iv.append(I(site=1,component='attention_pattern',mode='patch',positions=(34,)))
 elif fam=='readout':
  v=r.sites[site].hidden_in;vc=center(v)
  if s['direction']=='node_span':dd=vc@nodebasis@nodebasis.T
  else:
   pred=m.unembed(m.ln_final(h[:,-1]))[:,:10].argmax(-1)
   w=m.unembed.weight[:10]-m.unembed.weight[:10].mean(0,keepdim=True);w=center(w)[pred][:,None,:]
   dd=(vc*w).sum(-1,keepdim=True)/w.square().sum(-1,keepdim=True).clamp_min(1e-20)*w
  donor=changed_trace(donor,site,'hidden_in',v-s['strength']*dd);iv.append(I(site=site,component='residual_skip',mode='patch',positions=(34,)))
  if s['pattern']:iv.append(I(site=1,component='attention_pattern',mode='patch',positions=(34,)))
 else:raise ValueError(fam)
 return x,iv,donor

@torch.no_grad()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['B','D','E'],required=True);ap.add_argument('--phase',choices=['smoke','discovery','confirmation'],required=True);args=ap.parse_args()
 torch.set_num_threads(4);torch.set_num_interop_threads(1);phase=args.phase;name=args.model
 cp=B/f'backbones/local_control_L6_seed{dict(B=3,D=5,E=7)[name]}/best.pt';jp=D/f'local/{name}/controllers.pt';hashes=dict(checkpoint=sha(cp),controllers=sha(jp),data=sha(P/'datasets.json'),code=sha(__file__),evaluator=sha(P/'code/circuit_extended.py'))
 prior=json.loads((D/f'graph/{name}_fit1.json').read_text());assert hashes['checkpoint']==prior['checkpoint_sha256'] and hashes['controllers']==prior['controllers_sha256']
 data=json.loads((P/'datasets.json').read_text());gs=data[phase]
 specs=specifications();save(P/f'specifications_{name}.json',specs)
 if phase=='confirmation':
  selected=json.loads((P/'selection.json').read_text())[name]['ids'];specs=[s for s in specs if s['id'] in selected];hashes['selection']=sha(P/'selection.json')
 out=P/phase;out.mkdir(exist_ok=True,parents=True)
 m,cfg,_=load_checkpoint(cp,torch.device('cuda'));m.eval().requires_grad_(False);versions=[p._version for p in m.parameters()]
 assert len(m.blocks)==2 and cfg.seq_len==35 and cfg.n_heads==4
 for block in m.blocks:assert block.residual_projector is None
 w=center(m.unembed.weight[:10]-m.unembed.weight[:10].mean(0,keepdim=True));_,sv,vh=torch.linalg.svd(w,full_matrices=False);nodebasis=vh[:9].T
 assert sv[-1]<1e-4
 if phase=='confirmation':basis=torch.load(P/f'basis/{name}.pt',weights_only=True,map_location='cuda')['basis']
 else:
  jf=_load_controller(path=jp,name='seed1_J_one_beh_rank256',device=torch.device('cuda'));ds=[]
  for off in range(0,len(gs),16):
   h,_=batch(m,cfg,gs[off:off+16]);z=apply_vector_map(h,positions=tuple(range(35)),controller=jf);ds.append((z-h)[:,34])
  dm=torch.cat(ds);_,svals,vh=torch.linalg.svd(dm,full_matrices=True);basis=vh.T
  assert torch.allclose(dm@basis@basis.T,dm,atol=2e-4,rtol=2e-4)
  path=P/('smoke_basis' if phase=='smoke' else 'basis');path.mkdir(exist_ok=True);torch.save(dict(basis=basis.cpu(),singular_values=svals.cpu()),path/f'{name}.pt')
 hashes['basis']=sha(P/('smoke_basis' if phase=='smoke' else 'basis')/f'{name}.pt')
 fits=[1,2] if phase=='confirmation' else [1]
 for fit in fits:
  tic=time.time();j=_load_controller(path=jp,name=f'seed{fit}_J_one_beh_rank256',device=torch.device('cuda'))
  save(out/f'{name}_fit{fit}_running.json',dict(pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],hashes=hashes,phase=phase,conditions=len(specs),graphs=len(gs),fit=fit,started=time.time()))
  results=[];shuffled=[];prob=[];margin=[];lab=[];checks=[]
  for off in range(0,len(gs),16):
   h,labels=batch(m,cfg,gs[off:off+16]);z=apply_vector_map(h,positions=tuple(range(35)),controller=j)
   yr,r=run(m,h,loop_indices=(6,));yj,jt=run(m,z,loop_indices=(6,));lab.append(labels.cpu().numpy())
   if off==0:
    for x,yy in [(h,yr),(z,yj)]:assert torch.allclose(original(m,x,loop_indices=(6,))[0],yy,atol=2e-4,rtol=2e-4)
    for site in [0,1]:
     for pos in [None,(34,),tuple(range(34))]:
      ii=[I(site=site,component='block_input',mode='patch',positions=pos)]
      kw=dict(site=site,component='block_input',interventions=ii,donor=jt.sites[site].hidden_in)
      assert torch.equal(fast_token(r.sites[site].hidden_in,**kw),old_token(r.sites[site].hidden_in,**kw))
      for comp in ['q','k','v','head_context','attention_pattern']:
       for head in [None,(0,),(3,)]:
        ii=[I(site=site,component=comp,mode='patch',positions=pos,heads=head)];kw=dict(site=site,interventions=ii,donor=getattr(jt.sites[site],comp));val=getattr(r.sites[site],comp)
        if comp=='attention_pattern':assert torch.equal(fast_pattern(val,**kw),old_pattern(val,**kw))
        else:assert torch.equal(fast_head(val,component=comp,**kw),old_head(val,component=comp,**kw))
    checks.append('baseline and vectorized helper equivalence')
   pp=[];ss=[];qq=[];mm=[]
   for s in specs:
    x,iv,don=patch(m,h,z,r,jt,basis,nodebasis,s)
    y,t=run(m,x,loop_indices=(6,),interventions=iv,donor_trace=don)
    if off==0 and s['family']=='scalar_head' and s['alpha']==1:
     base=I(site=1,component=s['component'],mode='patch',positions=(34,),heads=None if s['head']==-1 else (s['head'],))
     yy,_=run(m,h,loop_indices=(6,),interventions=[base],donor_trace=jt);assert torch.equal(yy,y)
    if off==0 and s['family']=='scalar' and s['site']>=0:
     site=s['site'];assert torch.equal(t.sites[site].q,r.sites[site].q) and torch.equal(t.sites[site].attention_out,r.sites[site].attention_out)
    pp.append(y.argmax(-1).cpu().numpy());qq.append(y.softmax(-1).gather(1,labels[:,1,None])[:,0].cpu().numpy());mm.append((y.gather(1,labels[:,1,None])-y.gather(1,labels[:,0,None]))[:,0].cpu().numpy())
    if phase=='confirmation':
     xx,ii,dd=patch(m,h,z,r,jt,basis,nodebasis,s,shuffle=True);ys,_=run(m,xx,loop_indices=(6,),interventions=ii,donor_trace=dd);ss.append(ys.argmax(-1).cpu().numpy())
   results.append(np.array(pp));prob.append(np.array(qq));margin.append(np.array(mm))
   if ss:shuffled.append(np.array(ss))
   print(json.dumps(dict(model=name,phase=phase,fit=fit,graphs_done=off+len(h)//10,seconds=round(time.time()-tic,1),peak_mib=torch.cuda.max_memory_allocated()/2**20)),flush=True)
   if torch.cuda.mem_get_info()[0]<16*2**30:raise RuntimeError('reserve breached')
  arrays=dict(predictions=np.concatenate(results,1),probability=np.concatenate(prob,1),margin=np.concatenate(margin,1),labels=np.concatenate(lab),graphs=np.array(gs),conditions=np.array([s['id'] for s in specs]))
  if shuffled:arrays['shuffled']=np.concatenate(shuffled,1)
  assert versions==[p._version for p in m.parameters()] and sha(cp)==hashes['checkpoint'] and sha(jp)==hashes['controllers']
  np.savez_compressed(out/f'{name}_fit{fit}.npz',**arrays);save(out/f'{name}_fit{fit}.json',dict(status='complete',phase=phase,model=name,fit=fit,hashes=hashes,checks=checks,weights_unchanged=True,seconds=time.time()-tic,conditions=len(specs),pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],peak_mib=torch.cuda.max_memory_allocated()/2**20))
if __name__=='__main__':main()
