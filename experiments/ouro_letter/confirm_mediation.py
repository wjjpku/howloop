import argparse,hashlib,importlib,json,os,sys,time
from pathlib import Path
import subprocess
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
sys.path.insert(0,'/data/wujiaju/letter_walk_native_20260914/code')
from train_full import MODEL
from train_affine_pair import Affine,CK_SHA

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(8*1024**2),b''):h.update(c)
 return h.hexdigest()
def emit(v):print(json.dumps(v),flush=True)
a=argparse.ArgumentParser();a.add_argument('--pairs',required=True);a.add_argument('--out',required=True);a.add_argument('--smoke',action='store_true');a.add_argument('--omit-call',type=int,default=0);a.add_argument('--patch-call',type=int,default=0);args=a.parse_args()
O=Path(args.out);O.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(2026092301)
BASE='/data/wujiaju/ouro26_letter_full_20260915/run/checkpoint.pt';JP='/data/wujiaju/ouro26_affine_pair_20260915/L4/checkpoint.pt'
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
 capture=state['capture'] is not None and site[0]>0
 patch=site[0]>0 and site[1] in state['scope']
 if not(capture or patch):return out,w
 if patch:out=out.clone()
 for head in range(q.shape[1]):
  with torch.autocast('cuda',enabled=False):
   kk=mod.repeat_kv(k,m.num_key_value_groups)[:,head].float();vv=mod.repeat_kv(v,m.num_key_value_groups)[:,head].float();scores=q[:,head].float()@kk.transpose(-2,-1)*kw['scaling']
   if mask is None:scores=scores.masked_fill(torch.ones(scores.shape[-2:],device=q.device,dtype=torch.bool).triu(1),float('-inf'))
   else:
    mm=mask[:,0,:,:k.shape[-2]];scores=scores.masked_fill(~mm,float('-inf')) if mm.dtype==torch.bool else scores+mm.float()
   prob=scores.softmax(-1);assert torch.isfinite(prob).all()
  if capture:banks[state['capture'],site,head]={'pattern':prob.cpu(),'output':out[:,:,head].cpu(),'value':vv.to(out.dtype).cpu()}
  if patch:
   kind=state['kind'];source='native' if 'self' in kind or 'damage' in kind else 'unrelated' if 'unrelated' in kind else 'J'
   source_site=((site[0]%3)+1,site[1]) if 'wrong_call' in kind else site
   record=banks[source,source_site,head]
   if kind.endswith('output'):new=record['output'].to(out.device)
   else:
    with torch.autocast('cuda',enabled=False):
     pp=prob if kind.endswith('value') else record['pattern'].to(q.device)
     vs=record['value'].to(q.device).float() if kind.endswith('value') else vv
     new=(pp@vs).to(out.dtype)
    del pp,vs
   out[:,:,head]=new;del new
  del kk,vv,scores,prob
 return out,w
registry.register('sdpa',attention)
def run(ids):
 x=torch.tensor([ids],device='cuda');_,hs,_=model.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False);return model.lm_head(hs[-1][:,-1]).float()[0]
L=set(range(32,48));E=set(range(16));M=set(range(16,32));Q={33,34}
conditions={'late_self_output':L,'late_self_pattern':L,'late_rescue_pattern':L,'late_rescue_output':L,'late_rescue_value':L,'late_damage_pattern':L,'early_rescue_pattern':E,'middle_rescue_pattern':M,'local_rescue_pattern':Q,'late_unrelated_pattern':L,'late_wrong_call_pattern':L}
generate={'native','J','late_rescue_pattern','late_rescue_output','late_rescue_value','late_damage_pattern','late_unrelated_pattern','late_wrong_call_pattern'}
pairs=json.loads(Path(args.pairs).read_text())['pairs'];pairs=pairs[:1] if args.smoke else pairs
manifest.update(status='running',conditions={k:sorted(v) for k,v in conditions.items()},generate=sorted(generate),protocol_sha=sha(Path(__file__).with_name('CONFIRMATION_PROTOCOL.md')));(O/'manifest.json').write_text(json.dumps(manifest,indent=2));start=time.time()
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 for pair in pairs:
  state['length']=len(pair['base']['ids']);banks={};state.update(scope=set(),kind=None,capture='native',enabled=False);zn=run(pair['base']['ids']);state.update(capture='J',enabled=True);zj=run(pair['base']['ids']);state['capture']='unrelated';run(pair['source']['ids']);state['capture']=None;target=pair['target_ids'][0]
  def save(kind,z):
   row=dict(pair=pair['index'],condition=kind,prediction=int(z.argmax()),target=target,correct=int(z.argmax())==target,probability=float(z.softmax(-1)[target]),native_correct=int(zn.argmax())==target,J_correct=int(zj.argmax())==target)
   if 'self' in kind:
    row['max_logit_error']=float((z-zn).abs().max())
    if kind.endswith('output'):assert row['max_logit_error']==0,row
   if kind in generate:
    ids=torch.tensor([pair['base']['ids']],device='cuda');state['capture']=None
    output=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=16,do_sample=False,use_cache=True,exit_at_step=3,pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    generated=output[0,ids.shape[1]:];text=tokenizer.decode(generated,skip_special_tokens=True).strip();row.update(generated=text,generation_correct=text.strip('.!\n ').lower()==pair['base_answer'].lower(),truncated=len(generated)==16,first_generated_token=int(generated[0]))
    assert row['first_generated_token']==row['prediction'],row
   with (O/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  state.update(scope=set(),kind=None,enabled=False);save('native',zn);state['enabled']=True;save('J',zj)
  for kind,scope in conditions.items():
   state.update(scope=scope,kind=kind,enabled='damage' in kind);z=run(pair['base']['ids']);save(kind,z);del z
  emit(dict(event='pair_complete',pair=pair['index'],elapsed=time.time()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30))
  if torch.cuda.mem_get_info()[0]<16*2**30:raise RuntimeError('GPU reserve')
 assert versions==[p._version for p in model.parameters()]
manifest.update(status='complete',elapsed=time.time()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30,parameters_unchanged=True);(O/'manifest.json').write_text(json.dumps(manifest,indent=2));emit(manifest)
