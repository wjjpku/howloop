"""Paired lexical transfer, no training; run after the authorized J fit finishes."""
import os,json,time,hashlib,subprocess,random
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from train_ouro_full import MODEL
from ouro_stepwise_pair_task import example
from ouro_eval_panel import EVAL_SEEDS
from task import PAIRS
from train_ouro_stepwise_pair_j import Affine

BASE=Path('/data/wujiaju/ouro26_stepwise_pair_control_20260915')
ROOT=BASE/'vocab_transfer_20260916'
A=[('big','small'),('tall','short'),('strong','weak'),('left','right'),('up','down'),('day','night'),('black','white'),('true','false'),('hard','soft'),('thick','thin'),('loud','quiet'),('smooth','rough')]
B=[('good','bad'),('wide','narrow'),('deep','shallow'),('early','late'),('love','hate'),('win','lose'),('buy','sell'),('rise','fall'),('accept','reject'),('begin','end'),('increase','decrease'),('push','pull')]
assert len({w for p in PAIRS+A+B for w in p})==72

def case(seed,k,condition):
    prompt,answer,sig,_=example(seed,k,k,0)
    pairs=list(PAIRS);rng=random.Random(seed+20260916)
    if condition!='original':
        pool=list(A if condition!='new_B' else B);rng.shuffle(pool)
        chosen=list(range(12));rng.shuffle(chosen)
        for i,j in enumerate(chosen[:6] if condition=='half_A' else chosen):pairs[j]=pool[i]
    mapping={w:v for old,new in zip(PAIRS,pairs) for w,v in zip(old,new)}
    sequence,rest=prompt.split('. ',1)
    words=sequence.removeprefix('Word sequence: ').split()
    newwords=[mapping[w] for w in words]
    expected=' '.join(mapping[w] for w in answer.split())
    # Independently replay the relabeled oracle to verify the same deletion position.
    opposites={w:v for x,y in pairs for w,v in [(x,y),(y,x)]}
    live=newwords[:];history=[]
    for t in range(12):
        idx=next(i for i in range(len(live)-1) if opposites[live[i]]==live[i+1])
        history.append(' '.join(live[idx:idx+2]));del live[idx:idx+2]
    assert expected==history[k-1] and len(newwords)==24 and len(set(newwords))==24
    return 'Word sequence: '+' '.join(newwords)+'. '+rest,expected,history

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    ROOT.mkdir(exist_ok=False)
    trainlog=Path('/data/wujiaju/logs/ouro_stepwise_pair_j_3gpu.log')
    deadline=time.time()+3600
    while '"event": "complete", "smoke": false' not in trainlog.read_text():
        if time.time()>deadline:raise RuntimeError('Training dependency timeout; no GPU allocated')
        print('Waiting for final J checkpoint/evaluation; no GPU allocated',flush=True);time.sleep(30)
    # Do not co-locate this unmeasured evaluation workload.
    for attempt in range(12):
        used=int(subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if used<100:break
        time.sleep(10)
    else:raise RuntimeError('Selected GPU not empty; no evaluation launched')
    torch.set_num_threads(4);torch.manual_seed(20260916)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tok.padding_side='left'
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa')
    ck=torch.load(BASE/'checkpoint.pt',map_location='cpu',weights_only=False,mmap=True)
    assert ck['step']==500;model.load_state_dict(ck['model']);del ck
    model=model.to('cuda').eval().requires_grad_(False);model.model.total_ut_steps=4;model.config.total_ut_steps=4
    jck=torch.load(BASE/'shared_j_fit_k18/checkpoint.pt',map_location='cpu',weights_only=False)
    assert jck['step']==500 and jck['backbone_sha256']==sha(BASE/'checkpoint.pt')
    J=Affine(model.config.hidden_size).to('cuda');J.load_state_dict(jck['affine']);J.eval().requires_grad_(False)
    enabled=False;counter=0
    def reset(m,args,kwargs):
        nonlocal counter
        counter=0
    def hook(m,args,out):
        nonlocal counter
        counter+=1;assert counter<=4
        return J(out) if enabled else out
    model.model.register_forward_pre_hook(reset,with_kwargs=True);model.model.norm.register_forward_hook(hook)
    versions=[p._version for p in list(model.parameters())+list(J.parameters())]
    manifest=dict(pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],training=False,loops=4,
        baseline_sha256=jck['backbone_sha256'],J_sha256=sha(BASE/'shared_j_fit_k18/checkpoint.pt'),J_step=500,
        words_A=A,words_B=B,n_per_k=64,seeds=EVAL_SEEDS,template=0,max_new_tokens=160,batch=8,
        conditions=['original','half_A','new_A','new_B'],pair_relabeling_seed='sequence seed + 20260916',
        caveat='Unseen in task training, not guaranteed absent from pretraining/general replay. Natural antonym ambiguity and tokenizer changes remain possible.')
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
    records=[];start=time.time()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for arm in ['J','no_J']:
            enabled=arm=='J'
            for condition in manifest['conditions']:
                for k in range(1,9):
                    for off in range(0,64,8):
                        if torch.cuda.mem_get_info()[0]<16*2**30:raise RuntimeError('Memory reserve breached')
                        seeds=EVAL_SEEDS[off:off+8];rows=[case(s,k,condition) for s in seeds]
                        prompts=[tok.apply_chat_template([dict(role='user',content=r[0])],tokenize=True,add_generation_prompt=True) for r in rows]
                        batch=tok.pad(dict(input_ids=prompts),padding=True,return_tensors='pt').to('cuda')
                        out=model.generate(**batch,exit_at_step=3,max_new_tokens=160,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
                        assert counter==4
                        for seed,row,ids,p in zip(seeds,rows,out[:,batch.input_ids.shape[1]:],prompts):
                            generated=tok.decode(ids,skip_special_tokens=True).strip();parsed=' '.join(generated.lower().split())
                            rec=dict(arm=arm,condition=condition,k=k,seed=seed,prompt=row[0],expected=row[1],generated=generated,
                                exact=parsed==row[1],matching_steps=[i+1 for i,v in enumerate(row[2]) if parsed==v],
                                prompt_tokens=len(p),answer_tokens=len(tok.encode(row[1],add_special_tokens=False)))
                            records.append(rec)
                            with (ROOT/'outputs.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
                    rr=[r for r in records if r['arm']==arm and r['condition']==condition and r['k']==k]
                    print(json.dumps(dict(event='cell',arm=arm,condition=condition,k=k,n=len(rr),correct=sum(r['exact'] for r in rr),seconds=time.time()-start)),flush=True)
    assert versions==[p._version for p in list(model.parameters())+list(J.parameters())]
    cells=[dict(arm=a,condition=c,k=k,n=64,correct=sum(r['exact'] for r in records if r['arm']==a and r['condition']==c and r['k']==k)) for a in ['J','no_J'] for c in manifest['conditions'] for k in range(1,9)]
    result=dict(status='complete',cells=cells,parameters_unchanged=True,seconds=time.time()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30)
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
