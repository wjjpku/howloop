"""Ouro antonym SFT: native four loops, final-only loss, frozen unused gate."""
import argparse, hashlib, json, math, os, random, shutil, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import Adafactor

MODEL='/data/paperexperiment/models/Ouro-2.6B'
from task import make_sequence, trace, OPPOSITE, bank
import re
from score_ouro_content import parse as parse_answer

def topology(words):
    labels={}; result=[]
    for w in words:
        pair=tuple(sorted((w,OPPOSITE[w])))
        if pair not in labels: labels[pair]=len(labels)
        result.append(labels[pair])
    return tuple(result)

def partition(words):
    return int(hashlib.sha256(bytes(topology(words))).hexdigest(),16)%100

HISTORICAL={topology(r['words']) for r in bank() if r['kind']=='cancellation'}

def task(seed,k,template):
    # Validation and training are disjoint even after arbitrary pair relabeling.
    rng=random.Random(seed)
    held=9100000<=seed<9200000
    while True:
        words=make_sequence(rng); steps=trace(words)
        bucket=partition(words)
        if (80<=bucket<90 if held else bucket<80) and topology(words) not in HISTORICAL and max(x['dependency_depth'] for x in steps[:8])>=3:
            break
    instructions=[
        'Repeatedly delete the leftmost adjacent pair of antonyms, then join the remaining words without changing their order.',
        'Scan from left to right, remove the first adjacent opposite-meaning pair, close the gap, and repeat.',
        'At each step erase only the leftmost neighboring antonyms. Preserve all other words in order.',
        'Play antonym cancellation: delete the earliest adjacent opposite pair and concatenate the remaining words.',
        'Each deletion removes the leftmost adjacent pair of antonyms in the current list. Repeat on the shortened list.',
        'Keep canceling neighboring antonyms, always choosing the leftmost available pair and closing its gap.']
    prompt='Word sequence: '+' '.join(words)+'. '+instructions[template]+f' Which two words are deleted on deletion number {k}? Answer with only those two words in their original left-to-right order.'
    return prompt,' '.join(steps[k-1]['removed']),topology(words)

def encode_task(tok,seed,k,template):
    prompt,answer,graph=task(seed,k,template)
    prefix=tok.apply_chat_template([dict(role='user',content=prompt)],add_generation_prompt=True,tokenize=True)
    full=tok.apply_chat_template([dict(role='user',content=prompt),dict(role='assistant',content=answer)],tokenize=True)
    assert full[:len(prefix)]==prefix
    return full,[-100]*len(prefix)+full[len(prefix):],graph

def prepare(root,tok):
    import pyarrow.parquet as pq
    root.mkdir(parents=True,exist_ok=True)
    pools={s:{src:[] for src in ['documents','stories','code']} for s in ['train','validation']}
    seen=set(); provenance=[]
    def add(text,source,identity):
        h=hashlib.sha256(text.encode()).hexdigest()
        if h in seen:return
        seen.add(h); split='validation' if int(h[:8],16)%10==0 else 'train'
        if len(pools[split][source])>= (600 if split=='train' else 40):return
        ids=tok(text,add_special_tokens=False).input_ids
        for i in range(0,min(len(ids)-512,4096)+1,512):
            chunk=ids[i:i+512]
            if len(chunk)==512:pools[split][source].append(chunk)
        provenance.append(dict(source=source,path=identity,sha256=h,split=split))
    for p in sorted(Path('/data/datasets/LooGLE/data').glob('*.jsonl')):
        for line in p.open():
            obj=json.loads(line);text=obj.get('context','')
            if len(text)>1000:add(text,'documents',str(p))
    for p in sorted(Path('/data/paperexperiment/loop-attnres-tinystories-20260729/raw-parquet').glob('*.parquet')):
        for batch in pq.ParquetFile(p).iter_batches(batch_size=256,columns=['text']):
            for text in batch.column(0).to_pylist():add(text,'stories',str(p))
            if all(len(pools[s]['stories'])>= (600 if s=='train' else 40) for s in pools):break
        if all(len(pools[s]['stories'])>= (600 if s=='train' else 40) for s in pools):break
    for p in sorted(Path('/usr/lib/python3.10').glob('*.py')):
        add(p.read_text(errors='replace'),'code',str(p))
    counts={}
    for s in pools:
        for source,rows in pools[s].items():
            assert len(rows)>=20,(s,source,len(rows))
            arr=np.asarray(rows,dtype=np.int32);np.save(root/f'{s}_{source}.npy',arr);counts[f'{s}_{source}']=int(arr.size)
    (root/'manifest.json').write_text(json.dumps(dict(tokenizer=MODEL,counts=counts,document_disjoint=True,source_mix=[.7,.2,.1],provenance=provenance),indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--mode',choices=['prepare','smoke','train'],required=True);p.add_argument('--steps',type=int,default=500);p.add_argument('--resume',action='store_true');a=p.parse_args()
    torch.set_num_threads(4);tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    data=Path('/data/paperexperiment/ouro26_letter_full_20260915/data');a.root.mkdir(exist_ok=True,parents=True)
    if a.mode=='prepare':raise ValueError('Replay is reused read-only; do not regenerate shared data')
    if shutil.disk_usage(a.root).free < 24_000_000_000:
        raise RuntimeError('Need 24 GB free for safe FP32 checkpoint replacement; no model loaded')
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==1
    general={s:{src:np.load(data/f'{s}_{src}.npy') for src in ['documents','stories','code']} for s in ['train','validation']}
    torch.manual_seed(20260915)
    model=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa').to('cuda')
    model.requires_grad_(True);model.config.use_cache=False
    model.model.early_exit_gate.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    assert model.config.total_ut_steps==4
    opt=Adafactor((p for p in model.parameters() if p.requires_grad),lr=1e-5,scale_parameter=False,relative_step=False,warmup_init=False,beta1=None,weight_decay=0.)
    mode=a.mode; output=a.root/'run';output.mkdir(exist_ok=a.resume)
    # Refuse to begin a run that cannot safely retain and atomically replace FP32 weights.
    if shutil.disk_usage(a.root).free < 2*sum(p.numel() for p in model.parameters())*4 + 2*2**30:
        raise RuntimeError('Need approximately 24 GB free for two FP32 checkpoints and reserve; no training started')
    total_params=sum(p.numel() for p in model.parameters())
    manifest=dict(pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],mode=mode,model=MODEL,parameters=total_params,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),frozen_names=[n for n,p in model.named_parameters() if not p.requires_grad],
      loops=4,final_only=True,tbptt=False,optimizer='Adafactor FP32 factored states, no first moment',master_weights='FP32',autocast='BF16',
      lr=1e-5,warmup=50,steps=a.steps,task_examples_per_update=8,k_uniform=[1,2,3,4],templates=5,
      general_input_token_fraction=.2,loss='0.8 mean task answer CE + 0.2 general token CE',general_source_mix=[.7,.2,.1],
      validation_guard_nll_delta=None,checkpoint_retention=1,task='antonym cancellation only; no lexical classification',split='topology-disjoint train/validation; historical bank topologies excluded',gate='explicitly frozen; unused in final-only loss; no auxiliary gate training',
      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/f'manifest_{mode}.json').write_text(json.dumps(manifest,indent=2))
    def emit(x):
        print(json.dumps(x),flush=True)
        with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
    def batch(rows):
        n=max(len(x[0]) for x in rows);pad=tok.pad_token_id or tok.eos_token_id
        ids=torch.tensor([x[0]+[pad]*(n-len(x[0])) for x in rows],device='cuda')
        labels=torch.tensor([x[1]+[-100]*(n-len(x[1])) for x in rows],device='cuda')
        mask=torch.tensor([[1]*len(x[0])+[0]*(n-len(x[0])) for x in rows],device='cuda')
        return ids,labels,mask
    def loss(rows,audit=False):
        ids,labels,mask=batch(rows)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            outputs,hidden,gates=model.model(input_ids=ids,attention_mask=mask,use_cache=False)
            if audit:
                for h in hidden:h.retain_grad()
            logits=model.lm_head(hidden[3][:,:-1,:])
            value=torch.nn.functional.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),labels[:,1:].reshape(-1),ignore_index=-100)
        return value,hidden if audit else None
    held_graphs={task(9100000+i,k,t)[2] for i in range(16) for k in range(1,9) for t in [0,5]}
    def evaluate(step):
        model.eval(); vals={}
        with torch.no_grad():
            for src in general['validation']:
                values=[]
                for arr in general['validation'][src][:4]:values.append(float(loss([(arr.tolist(),arr.tolist())])[0]))
                vals[src]=sum(values)/len(values)
        model.train();nll=.7*vals['documents']+.2*vals['stories']+.1*vals['code'];emit(dict(event='general_validation',step=step,nll=nll,by_source=vals));return nll
    def task_eval(step,ks=range(1,5),templates=(0,)):
        model.eval();model.gradient_checkpointing_disable(); records=[]
        with torch.inference_mode():
            for template in templates:
                for k in ks:
                    for i in range(16):
                        prompt,answer,_=task(9100000+i,k,template)
                        ids=tok.apply_chat_template([dict(role='user',content=prompt)],add_generation_prompt=True,return_tensors='pt').to('cuda')
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            out=model.generate(ids,max_new_tokens=64,do_sample=False,use_cache=True,exit_at_step=3,pad_token_id=tok.pad_token_id or tok.eos_token_id)
                        text=tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True).strip()
                        records.append(dict(step=step,k=k,template=template,sequence=i,expected=answer,generated=text,correct=' '.join(re.findall(r'[a-z]+', text.lower()))==answer,content_correct=parse_answer(dict(kind='cancellation',generated=text))==answer))
        with (output/'eval_outputs.jsonl').open('a') as f:
            for row in records:f.write(json.dumps(row)+'\n')
        emit(dict(event='task_eval',step=step,acc={f't{t}_k{k}':sum(r['correct'] for r in records if r['k']==k and r['template']==t)/16 for t in templates for k in ks},content_acc={f't{t}_k{k}':sum(r['content_correct'] for r in records if r['k']==k and r['template']==t)/16 for t in templates for k in ks}))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train()
    def save(step,baseline,counts):
        required=total_params*4+256*2**20
        if shutil.disk_usage(a.root).free<required+2*2**30:raise RuntimeError('Insufficient space for atomic full checkpoint; existing checkpoint preserved')
        tmp=output/'checkpoint.tmp';dest=output/'checkpoint.pt'
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,baseline_nll=baseline,counts=counts,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),tmp)
        os.replace(tmp,dest);emit(dict(event='checkpoint',step=step,bytes=dest.stat().st_size))
    counts=dict(task_input=0,general_input=0,task_supervised=0,general_supervised=0);start=0
    if a.resume:
        ck=torch.load(output/'checkpoint.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);start=ck['step'];baseline=ck['baseline_nll'];counts=ck['counts'];torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state(ck['cuda_rng']);del ck
    else:baseline=evaluate(0)
    if not a.resume and mode!='smoke':task_eval(0)
    for step in range(start+1,(2 if mode=='smoke' else a.steps)+1):
        begin=time.time();opt.zero_grad(set_to_none=True);rows=[]
        for j in range(8):
            seed=100000000+step*100+j
            row=encode_task(tok,seed,1+j%4,(step*8+j)%5)
            while row[2] in held_graphs:seed+=100000000;row=encode_task(tok,seed,1+j%4,(step*8+j)%5)
            rows.append(row)
        nt=sum(len(r[0]) for r in rows);ng=round(nt/4);remaining=ng;r=random.Random(8000000+step);general_rows=[]
        while remaining:
            src=r.choices(['documents','stories','code'],[.7,.2,.1])[0];arr=r.choice(general['train'][src]);length=min(512,remaining)
            if length<2:break
            tokens=arr[:length].tolist();general_rows.append((tokens,tokens));remaining-=length
        ng=sum(len(x[0]) for x in general_rows); task_loss=0.;gen_loss=0.;stage_grads=None
        for j in range(0,8,2):
            val,hidden=loss(rows[j:j+2],audit=step==1 and j==0)
            (val*.8/4).backward();task_loss+=float(val.detach())/4
            if hidden is not None:
                stage_grads=[None if h.grad is None else float(h.grad.float().norm()) for h in hidden]
                assert all(x is not None and x>0 and math.isfinite(x) for x in stage_grads),stage_grads
        total_general_labels=sum(len(x[0])-1 for x in general_rows)
        for row in general_rows:
            val,_=loss([row]);weight=(len(row[0])-1)/total_general_labels
            (val*.2*weight).backward();gen_loss+=float(val.detach())*weight
        missing=[n for n,p in model.named_parameters() if p.grad is None]
        assert all('early_exit_gate' in n for n in missing),missing
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True))
        lr=1e-5*min(step/50,1.)*(1 if step<=50 else .1+.9*.5*(1+math.cos(math.pi*(step-50)/max(a.steps-50,1))))
        for group in opt.param_groups:group['lr']=lr
        opt.step()
        counts['task_input']+=nt;counts['general_input']+=ng;counts['task_supervised']+=sum(sum(v!=-100 for v in row[1][1:]) for row in rows);counts['general_supervised']+=total_general_labels
        emit(dict(event='train',step=step,task_loss=task_loss,general_loss=gen_loss,lr=lr,grad_norm=norm,stage_grads=stage_grads,missing_grad=missing if step==1 else None,counts=counts,seconds=time.time()-begin,peak_gib=torch.cuda.max_memory_allocated()/2**30))
        if torch.cuda.mem_get_info()[0]<16*2**30:raise RuntimeError('GPU reserve breached')
        if mode=='train' and (step%100==0 or step==a.steps):
            nll=evaluate(step)
            if nll>baseline+.3:emit(dict(event='general_regression_warning',step=step,baseline=baseline,nll=nll))
            save(step,baseline,counts);task_eval(step)
    if mode=='smoke':
        save(2,baseline,counts)
        ck=torch.load(output/'checkpoint.pt',map_location='cpu',weights_only=False);assert ck['step']==2 and len(ck['optimizer']['state'])>0;del ck
        emit(dict(event='smoke_passed_checkpoint_readback'))
    else:task_eval(a.steps,ks=range(1,9),templates=(5,))
    emit(dict(event='complete',mode=mode))

if __name__=='__main__':main()
