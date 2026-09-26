import argparse,hashlib,importlib,json,os,sys,time
from pathlib import Path
import subprocess
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
sys.path.insert(0,'/data/paperexperiment/letter_walk_native_20260914/code')
from train_full import MODEL
from train_affine_pair import Affine,CK_SHA

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(8*1024**2),b''):h.update(c)
 return h.hexdigest()
def emit(v):print(json.dumps(v),flush=True)
a=argparse.ArgumentParser();a.add_argument('--pairs',required=True);a.add_argument('--out',required=True);a.add_argument('--config',required=True);a.add_argument('--smoke',action='store_true');a.add_argument('--omit-call',type=int,default=0);a.add_argument('--patch-call',type=int,default=0);args=a.parse_args()
O=Path(args.out);O.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(2026092301)
BASE='/data/paperexperiment/ouro26_letter_full_20260915/run/checkpoint.pt';JP='/data/paperexperiment/ouro26_affine_pair_20260915/L4/checkpoint.pt'
assert sha(BASE)==CK_SHA
jsha=sha(JP);assert jsha=='15b55ed1255915fe785891735a6a83a9c3baf851bc42c6a73e2eabe6c058c898'
manifest=dict(status='loading',pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],base_sha=CK_SHA,j_sha=jsha,code_sha=sha(__file__),pairs_sha=sha(args.pairs),smoke=args.smoke,omit_call=args.omit_call,patch_call=args.patch_call,started=time.time())
(O/'manifest.json').write_text(json.dumps(manifest,indent=2));emit(manifest)
subprocess.Popen([sys.executable,str(Path(__file__).with_name('watch.py')),'--pid',str(os.getpid()),'--gpu',os.environ['CUDA_VISIBLE_DEVICES'],'--out',str(O/'resources.jsonl')],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True,torch_dtype=torch.float32,attn_implementation='sdpa')
ck=torch.load(BASE,map_location='cpu',weights_only=False,mmap=True);model.load_state_dict(ck['model'],strict=True);del ck
model=model.to('cuda').eval().requires_grad_(False);assert model.model.total_ut_steps==4
j=Affine(model.config.hidden_size);ck=torch.load(JP,map_location='cpu',weights_only=False,mmap=True);j.load_state_dict(ck['affine']);del ck;j=j.to('cuda').eval().requires_grad_(False)
versions=[p._version for p in model.parameters()]
tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
state=dict(loop=0,enabled=True,capture=None,scope=set(),kind=None,prefill=True,length=150);banks={}
def reset(m,aa,kw):
 ids=kw.get('input_ids',aa[0] if aa else None);state['prefill']=ids.shape[1]==state['length']
model.model.register_forward_pre_hook(reset,with_kwargs=True)
def jhook(m,aa,kw):
 if state['enabled'] and kw.get('current_ut',0)>0:return (j(aa[0]),)+aa[1:],kw
 return aa,kw
model.model.layers[0].register_forward_pre_hook(jhook,with_kwargs=True)
def stage(m,aa,kw):state['loop']=int(kw['current_ut'])
for b in model.model.layers:b.self_attn.register_forward_pre_hook(stage,with_kwargs=True)
mod=importlib.import_module(model.model.layers[34].self_attn.__class__.__module__);registry=mod.ALL_ATTENTION_FUNCTIONS;original=registry['sdpa']
def attention(m,q,k,v,mask,**kw):
 out,w=original(m,q,k,v,mask,**kw);site=(state['loop'],m.layer_idx)
 if not state['prefill']:return out,w
 capture=state['capture'] is not None and site[0] in [1,3] and site[1] in [41,43,47]
 patch=site[0] in state.get('calls',[1,2,3]) and site[1] in state['scope']
 if not(capture or patch):return out,w
 def pattern(head):
  with torch.autocast('cuda',enabled=False):
   kk=mod.repeat_kv(k,m.num_key_value_groups)[:,head].float();scores=q[:,head].float()@kk.transpose(-2,-1)*kw['scaling']
   if mask is None:scores=scores.masked_fill(torch.ones(scores.shape[-2:],device=q.device,dtype=torch.bool).triu(1),float('-inf'))
   else:
    mm=mask[:,0,:,:k.shape[-2]];scores=scores.masked_fill(~mm,float('-inf')) if mm.dtype==torch.bool else scores+mm.float()
   return scores.softmax(-1)
 if capture:
  ps=torch.stack([pattern(h) for h in range(q.shape[1])],dim=1);assert torch.isfinite(ps).all()
  banks[state['capture'],site]={'pattern':ps.cpu(),'output':out.cpu(),'value':mod.repeat_kv(v,m.num_key_value_groups).to(out.dtype).cpu()}
 if patch:
  kind=state['kind'];source='base' if 'self' in kind else 'source'
  source_site=((site[0]%3)+1,site[1]) if 'wrong_call' in kind else site;record=banks[source,source_site]
  if kind.endswith('output'):
   out=out.clone();heads=state['scope'][site[1]];out[:,:,heads]=record['output'].to(out.device)[:,:,heads];return out,w
  source_pattern=record['pattern'].to(q.device) if not kind.endswith('value') else None
  source_value=record['value'].to(q.device) if kind.endswith('value') else None
  out=out.clone()
  for head in state['scope'][site[1]]:
   with torch.autocast('cuda',enabled=False):
    pp=pattern(head) if kind.endswith('value') else source_pattern[:,head]
    vv=source_value[:,head].float() if kind.endswith('value') else mod.repeat_kv(v,m.num_key_value_groups)[:,head].float()
    out[:,:,head]=(pp@vv).to(out.dtype)
 return out,w
registry.register('sdpa',attention)
def run(ids):
 x=torch.tensor([ids],device='cuda');_,hs,_=model.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False);return model.lm_head(hs[-1][:,-1]).float()[0]
config=json.loads(Path(args.config).read_text());conditions=config['conditions'];generate=set(config.get('generate',[]))
pairs=json.loads(Path(args.pairs).read_text())['pairs'];pairs=pairs[:1] if args.smoke else pairs
manifest.update(status='running',conditions=conditions,config_sha=sha(args.config),generate=sorted(generate),protocol_sha=sha(Path(__file__).with_name('PARALLEL_PROTOCOL.md')));(O/'manifest.json').write_text(json.dumps(manifest,indent=2));start=time.time()
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 for pair in pairs:
  state['length']=len(pair['base']['ids']);banks={};state.update(scope=set(),kind=None,capture='base',enabled=True);zb=run(pair['base']['ids']);state['capture']='source';zs=run(pair['source']['ids']);state['capture']=None;zc=run(pair['counterfactual_run']['ids'])
  names=[pair['base_answer'],pair['source_answer'],pair['counterfactual']]
  def save(kind,z,ids_data=None):
   row=dict(pair=pair['index'],condition=kind,prediction=int(z.argmax()),target_ids=pair['target_ids'],target_probabilities=z.softmax(-1)[pair['target_ids']].tolist(),base_correct=int(zb.argmax())==pair['target_ids'][0],source_correct=int(zs.argmax())==pair['target_ids'][1],counterfactual_run_correct=int(zc.argmax())==pair['target_ids'][2])
   if 'self' in kind:
    row['max_logit_error']=float((z-zb).abs().max())
    if kind.endswith('output'):assert row['max_logit_error']==0,row
   if kind in generate:
    ids=torch.tensor([ids_data if ids_data is not None else pair['base']['ids']],device='cuda');state['capture']=None
    output=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=16,do_sample=False,use_cache=True,exit_at_step=3,pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    generated=output[0,ids.shape[1]:];text=tokenizer.decode(generated,skip_special_tokens=True).strip();clean=text.strip('.!\n ').lower()
    row.update(generated=text,generation_matches=[clean==name.lower() for name in names],truncated=len(generated)==16,first_generated_token=int(generated[0]));assert row['first_generated_token']==row['prediction'],row
   with (O/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  state.update(scope=set(),kind=None,enabled=True);save('base',zb);save('source',zs,pair['source']['ids']);save('counterfactual',zc,pair['counterfactual_run']['ids'])
  for kind,scope in conditions.items():
   state.update(scope={int(l):hs for l,hs in scope['heads'].items()},calls=scope.get('calls',[1,2,3]),kind=kind,enabled=True);z=run(pair['base']['ids']);save(kind,z);del z
  emit(dict(event='pair_complete',pair=pair['index'],elapsed=time.time()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30))
  if torch.cuda.mem_get_info()[0]<16*2**30:raise RuntimeError('GPU reserve')
 assert versions==[p._version for p in model.parameters()]
manifest.update(status='complete',elapsed=time.time()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30,parameters_unchanged=True);(O/'manifest.json').write_text(json.dumps(manifest,indent=2));emit(manifest)
