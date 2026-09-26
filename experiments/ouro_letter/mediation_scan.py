import argparse,hashlib,importlib,json,os,sys,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
sys.path.insert(0,'/data/paperexperiment/letter_walk_native_20260914/code')
from train_full import MODEL
from train_affine_pair import Affine,CK_SHA

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(8*1024**2),b''):h.update(c)
 return h.hexdigest()
def emit(v):print(json.dumps(v),flush=True)
a=argparse.ArgumentParser();a.add_argument('--pairs',required=True);a.add_argument('--out',required=True);a.add_argument('--smoke',action='store_true');args=a.parse_args()
O=Path(args.out);O.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(2026092301)
BASE='/data/paperexperiment/ouro26_letter_full_20260915/run/checkpoint.pt';JP='/data/paperexperiment/ouro26_affine_pair_20260915/L4/checkpoint.pt'
assert sha(BASE)==CK_SHA
jsha=sha(JP);assert jsha=='15b55ed1255915fe785891735a6a83a9c3baf851bc42c6a73e2eabe6c058c898'
manifest=dict(status='loading',pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],base_sha=CK_SHA,j_sha=jsha,code_sha=sha(__file__),pairs_sha=sha(args.pairs),smoke=args.smoke,started=time.time())
(O/'manifest.json').write_text(json.dumps(manifest,indent=2));emit(manifest)
model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True,torch_dtype=torch.float32,attn_implementation='sdpa')
ck=torch.load(BASE,map_location='cpu',weights_only=False,mmap=True);model.load_state_dict(ck['model'],strict=True);del ck
model=model.to('cuda').eval().requires_grad_(False);assert model.model.total_ut_steps==4
j=Affine(model.config.hidden_size);ck=torch.load(JP,map_location='cpu',weights_only=False,mmap=True);j.load_state_dict(ck['affine']);del ck;j=j.to('cuda').eval().requires_grad_(False)
versions=[p._version for p in model.parameters()]
state=dict(loop=0,enabled=True,capture=None,scope=set(),kind=None);banks={}
def jhook(m,args,kw):
 if state['enabled'] and kw.get('current_ut',0)>0:return (j(args[0]),)+args[1:],kw
 return args,kw
model.model.layers[0].register_forward_pre_hook(jhook,with_kwargs=True)
def stage(m,args,kw):state['loop']=int(kw['current_ut'])
for b in model.model.layers:b.self_attn.register_forward_pre_hook(stage,with_kwargs=True)
mod=importlib.import_module(model.model.layers[34].self_attn.__class__.__module__);registry=mod.ALL_ATTENTION_FUNCTIONS;original=registry['sdpa']
def attention(m,q,k,v,mask,**kw):
 out,w=original(m,q,k,v,mask,**kw);site=(state['loop'],m.layer_idx)
 capture=state['capture'] is not None and site[0]>0
 patch=site[0]>0 and site[1] in state['scope']
 if not(capture or patch):return out,w
 if patch:out=out.clone()
 # Sequential head processing matches the measured single-head GPU tensor workload.
 # All persistent activation banks are CPU tensors; no sixteen-head score allocation.
 for head in range(q.shape[1]):
  with torch.autocast('cuda',enabled=False):
   kk=mod.repeat_kv(k,m.num_key_value_groups)[:,head].float();vv=mod.repeat_kv(v,m.num_key_value_groups)[:,head].float();scores=q[:,head].float()@kk.transpose(-2,-1)*kw['scaling']
   if mask is None:scores=scores.masked_fill(torch.ones(scores.shape[-2:],device=q.device,dtype=torch.bool).triu(1),float('-inf'))
   else:
    mm=mask[:,0,:,:k.shape[-2]];scores=scores.masked_fill(~mm,float('-inf')) if mm.dtype==torch.bool else scores+mm.float()
   prob=scores.softmax(-1);assert torch.isfinite(prob).all()
  if capture:banks[state['capture'],site,head]={'pattern':prob.cpu(),'output':out[:,:,head].cpu(),'value':vv.to(out.dtype).cpu()}
  if patch:
   kind=state['kind'];source='native' if kind.startswith('self') or kind=='damage_pattern' else 'J';record=banks[source,site,head]
   if kind.endswith('output'):new=record['output'].to(out.device)
   else:
    with torch.autocast('cuda',enabled=False):
     pp=record['pattern'].to(q.device) if kind.endswith('pattern') else prob
     vs=record['value'].to(q.device).float() if kind.endswith('value') else vv
     new=(pp@vs).to(out.dtype)
    del pp,vs
   out[:,:,head]=new
   del new
  del kk,vv,scores,prob
 return out,w
registry.register('sdpa',attention)
def run(ids):
 x=torch.tensor([ids],device='cuda');_,hs,_=model.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False);return model.lm_head(hs[-1][:,-1]).float()[0]
pairs=json.loads(Path(args.pairs).read_text())['pairs'];pairs=pairs[:1] if args.smoke else pairs
scopes={'all':list(range(48)),'early':list(range(16)),'middle':list(range(16,32)),'late':list(range(32,48)),'local':[33,34]}
manifest.update(status='running',scopes=scopes);(O/'manifest.json').write_text(json.dumps(manifest,indent=2));start=time.time()
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 for pair in pairs:
  banks={};state.update(scope=set(),kind=None,capture='native',enabled=False);zn=run(pair['base']['ids']);state.update(capture='J',enabled=True);zj=run(pair['base']['ids']);state['capture']=None;target=pair['target_ids'][0]
  def save(kind,z,scope=None):
   row=dict(pair=pair['index'],condition=kind,scope=scope,prediction=int(z.argmax()),target=target,correct=int(z.argmax())==target,probability=float(z.softmax(-1)[target]),native_correct=int(zn.argmax())==target,J_correct=int(zj.argmax())==target)
   if kind.startswith('self'):
    row['max_logit_error']=float((z-zn).abs().max())
    if kind=='self_output':assert row['max_logit_error']==0,row
   with (O/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  save('native',zn);save('J',zj)
  for name,layers in scopes.items():
   state['scope']=set(layers)
   for kind in ['self_output','self_pattern','rescue_pattern','rescue_output','rescue_value','damage_pattern']:
    state.update(enabled=kind=='damage_pattern',kind=kind);z=run(pair['base']['ids']);save(kind,z,name);del z
   emit(dict(event='scope_complete',pair=pair['index'],scope=name,elapsed=time.time()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30))
   if torch.cuda.mem_get_info()[0]<16*2**30:raise RuntimeError('GPU reserve')
 assert versions==[p._version for p in model.parameters()]
manifest.update(status='complete',elapsed=time.time()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30,parameters_unchanged=True);(O/'manifest.json').write_text(json.dumps(manifest,indent=2));emit(manifest)
