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
state=dict(loop=0,capture=None,site=None,kind=None);banks={};layers=[8,16,24,32,33,34,35,40,44,47]
if args.smoke:layers=[34]
def jhook(m,args,kw):
 if kw.get('current_ut',0)>0:return (j(args[0]),)+args[1:],kw
 return args,kw
model.model.layers[0].register_forward_pre_hook(jhook,with_kwargs=True)
def stage(m,args,kw):state['loop']=int(kw['current_ut'])
for b in model.model.layers:b.self_attn.register_forward_pre_hook(stage,with_kwargs=True)
mod=importlib.import_module(model.model.layers[34].self_attn.__class__.__module__);registry=mod.ALL_ATTENTION_FUNCTIONS;original=registry['sdpa']
def attention(m,q,k,v,mask,**kw):
 out,w=original(m,q,k,v,mask,**kw);site=(state['loop'],m.layer_idx)
 capture=state['capture'] is not None and site[0] in [1,2,3] and site[1] in layers
 patch=state['site']==site
 if not(capture or patch):return out,w
 # CPU banks keep persistent GPU footprint close to prior batch-one evaluation.
 if capture or state['kind'].endswith('pattern'):
  with torch.autocast('cuda',enabled=False):
   kk=mod.repeat_kv(k,m.num_key_value_groups)[:,5].float();vv=mod.repeat_kv(v,m.num_key_value_groups)[:,5].float();scores=q[:,5].float()@kk.transpose(-2,-1)*kw['scaling']
   if mask is None:scores=scores.masked_fill(torch.ones(scores.shape[-2:],device=q.device,dtype=torch.bool).triu(1),float('-inf'))
   else:
    mm=mask[:,0,:,:k.shape[-2]];scores=scores.masked_fill(~mm,float('-inf')) if mm.dtype==torch.bool else scores+mm.float()
   prob=scores.softmax(-1);assert torch.isfinite(prob).all()
 if capture:banks[state['capture'],site]={'pattern':prob.cpu(),'output':out[:,:,5].cpu()}
 if patch:
  source='base' if state['kind'].startswith('self') else 'source';record=banks[source,site]
  if state['kind'].endswith('output'):new=record['output'].to(out.device)
  else:
   with torch.autocast('cuda',enabled=False):new=(record['pattern'].to(q.device)@vv).to(out.dtype)
  out=out.clone();out[:,:,5]=new
 return out,w
registry.register('sdpa',attention)
def run(ids):
 x=torch.tensor([ids],device='cuda')
 _,hs,_=model.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False)
 z=model.lm_head(hs[-1][:,-1]).float()[0];return z
pairs=json.loads(Path(args.pairs).read_text())['pairs'];pairs=pairs[:1] if args.smoke else pairs
start=time.time();manifest['status']='running';(O/'manifest.json').write_text(json.dumps(manifest,indent=2))
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 for pair in pairs:
  banks={};state.update(site=None,kind=None,capture='source');zs=run(pair['source']['ids']);state['capture']='base';zb=run(pair['base']['ids']);state['capture']=None
  def save(kind,z,site=None):
   row=dict(pair=pair['index'],condition=kind,site=site,prediction=int(z.argmax()),target_ids=pair['target_ids'],target_probabilities=z.softmax(-1)[pair['target_ids']].tolist(),target_logits=z[pair['target_ids']].tolist(),base_correct=int(zb.argmax())==pair['target_ids'][0],source_correct=int(zs.argmax())==pair['target_ids'][1])
   if kind=='self_output':
    row['max_logit_error']=float((z-zb).abs().max());assert row['max_logit_error']==0,row
   if kind=='self_pattern':row['max_logit_error']=float((z-zb).abs().max())
   with (O/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  save('base',zb);save('source',zs)
  for layer in layers:
   for loop in [1,2,3]:
    site=(loop,layer);state['site']=site
    for kind in ['self_output','self_pattern','source_pattern','source_output']:
     state['kind']=kind;z=run(pair['base']['ids']);save(kind,z,site);del z
    emit(dict(event='site_complete',pair=pair['index'],site=site,elapsed=round(time.time()-start,2),peak_gib=torch.cuda.max_memory_allocated()/2**30))
    if torch.cuda.mem_get_info()[0]<16*2**30:raise RuntimeError('free GPU reserve below16GiB')
  assert versions==[p._version for p in model.parameters()]
manifest.update(status='complete',elapsed=time.time()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30,parameters_unchanged=True);(O/'manifest.json').write_text(json.dumps(manifest,indent=2));emit(manifest)
