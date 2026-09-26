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
a=argparse.ArgumentParser();a.add_argument('--pairs',required=True);a.add_argument('--out',required=True);a.add_argument('--smoke',action='store_true');a.add_argument('--omit-call',type=int,default=0);a.add_argument('--patch-call',type=int,default=0);args=a.parse_args()
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
SELECTED={47: [4, 13, 0, 3, 2, 15, 7], 41: [10, 12, 2, 15], 43: [15, 11, 13, 6, 1]};NEIGHBOR={47: [1, 5, 6, 8, 9, 10, 11], 41: [0, 1, 3, 4], 43: [0, 2, 3, 4, 5]}
state=dict(loop=0,capture=None,arm='base',length=150,prefill=True);banks={}
def reset(m,aa,kw):
 ids=kw.get('input_ids',aa[0] if aa else None);state['prefill']=ids.shape[1]==state['length']
model.model.register_forward_pre_hook(reset,with_kwargs=True)
def jhook(m,aa,kw):
 if kw.get('current_ut',0)>0:return (j(aa[0]),)+aa[1:],kw
 return aa,kw
model.model.layers[0].register_forward_pre_hook(jhook,with_kwargs=True)
def stage(m,aa,kw):state['loop']=int(kw['current_ut'])
for b in model.model.layers:b.self_attn.register_forward_pre_hook(stage,with_kwargs=True)
mod=importlib.import_module(model.model.layers[34].self_attn.__class__.__module__);registry=mod.ALL_ATTENTION_FUNCTIONS;original=registry['sdpa']
def attention(m,q,k,v,mask,**kw):
 site=state['prefill'] and m.layer_idx in SELECTED and state['loop']==3;arm=state['arm'];loop=state['loop']
 qq=banks['alternative',loop,m.layer_idx]['q'].to(q.device) if site and arm not in ['base','alternative'] else q
 out,w=original(m,qq,k,v,mask,**kw)
 if site and state['capture'] is not None:banks[state['capture'],loop,m.layer_idx]={'q':q.cpu(),'out':out.cpu()}
 if site and arm.startswith('restore'):
  clean=banks['base',loop,m.layer_idx]['out'].to(out.device)
  if arm=='restore_all':out=clean
  else:
   hs=NEIGHBOR[m.layer_idx] if arm=='restore_neighbor' else SELECTED[m.layer_idx] if arm=='restore_selected' else SELECTED[m.layer_idx] if arm==f'restore_layer{m.layer_idx}' else []
   out=out.clone();out[:,:,hs]=clean[:,:,hs]
 return out,w
registry.register('sdpa',attention)
def run(ids):
 x=torch.tensor([ids],device='cuda');_,hs,_=model.model(input_ids=x,attention_mask=torch.ones_like(x),use_cache=False);return model.lm_head(hs[-1][:,-1]).float()[0]
pairs=json.loads(Path(args.pairs).read_text())['pairs'];pairs=pairs[:1] if args.smoke else pairs
conditions=['corrupt','restore_selected','restore_neighbor','restore_layer41','restore_layer43','restore_layer47','restore_all'];gen={'base','corrupt','restore_selected','restore_neighbor','restore_all'}
manifest.update(status='running',conditions=conditions,selected_heads=SELECTED,neighbor_heads=NEIGHBOR,generate=sorted(gen),protocol_sha=sha(Path(__file__).with_name('PARALLEL_RESTORE_PROTOCOL.md')));(O/'manifest.json').write_text(json.dumps(manifest,indent=2));start=time.time()
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 for pair in pairs:
  banks={};state.update(length=len(pair['base']['ids']),arm='base',capture='base');zb=run(pair['base']['ids']);state.update(arm='alternative',capture='alternative');zc=run(pair['counterfactual_run']['ids']);state['capture']=None;target=pair['target_ids'][0]
  def save(kind,z):
   row=dict(pair=pair['index'],condition=kind,prediction=int(z.argmax()),target=target,correct=int(z.argmax())==target,probability=float(z.softmax(-1)[target]),alternative_id=pair['target_ids'][2],base_correct=int(zb.argmax())==target,alternative_run_correct=int(zc.argmax())==pair['target_ids'][2])
   if kind=='restore_all':row['max_logit_error']=float((z-zb).abs().max());assert row['max_logit_error']==0,row
   if kind in gen:
    ids=torch.tensor([pair['base']['ids']],device='cuda');output=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=16,do_sample=False,use_cache=True,exit_at_step=3,pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id);tokens=output[0,ids.shape[1]:];text=tokenizer.decode(tokens,skip_special_tokens=True).strip()
    row.update(generated=text,generation_correct=text.strip('.!\n ').lower()==pair['base_answer'].lower(),truncated=len(tokens)==16,first_generated_token=int(tokens[0]));assert row['first_generated_token']==row['prediction'],row
   with (O/'results.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  state['arm']='base';save('base',zb);save('alternative',zc)
  for kind in conditions:
   state['arm']=kind;z=run(pair['base']['ids']);save(kind,z);del z
  emit(dict(event='pair_complete',pair=pair['index'],elapsed=time.time()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30))
  if torch.cuda.mem_get_info()[0]<16*2**30:raise RuntimeError('GPU reserve')
 assert versions==[p._version for p in model.parameters()]
manifest.update(status='complete',elapsed=time.time()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30,parameters_unchanged=True);(O/'manifest.json').write_text(json.dumps(manifest,indent=2));emit(manifest)
