"""Paired leave-one-J-out and keep-one-J-only evaluation; no training."""
import json,os,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from diagnose_ouro_heads import BASE,JPATH,JSHA,digest
from train_ouro_four_j import Affine,CK_SHA
from train_ouro_full import MODEL,task
from ouro_eval_panel import EVAL_SEEDS
from score_ouro_content import parse

ROOT=Path('/data/wujiaju/ouro26_j_positions_20260915')
CONDITIONS={'all':[0,1,2,3],'none':[]}
CONDITIONS.update({f'drop_{i+1}':[j for j in range(4) if j!=i] for i in range(4)})
CONDITIONS.update({f'only_{i+1}':[i] for i in range(4)})

def main():
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==1
    ROOT.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(20260915)
    assert digest(BASE)==CK_SHA and digest(JPATH)==JSHA
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tok.padding_side='left'
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True,
        torch_dtype=torch.float32,attn_implementation='sdpa')
    ck=torch.load(BASE,map_location='cpu',weights_only=False,mmap=True)
    assert ck['step']==500;model.load_state_dict(ck['model'],strict=True);del ck
    model=model.to('cuda').eval().requires_grad_(False)
    j=Affine(2048);ck=torch.load(JPATH,map_location='cpu',weights_only=False)
    assert ck['step']==500;j.load_state_dict(ck['affine']);del ck
    j=j.to('cuda').eval().requires_grad_(False)
    versions=[x._version for x in model.parameters()]+[x._version for x in j.parameters()]
    active=set();stage=[0];trace=[]
    def reset(m,args,kwargs):stage[0]=0
    def hook(m,args,h):
        s=stage[0];stage[0]+=1;assert s<4
        if s in active:trace.append(s);return j(h)
        return h
    model.model.register_forward_pre_hook(reset,with_kwargs=True)
    model.model.norm.register_forward_hook(hook)
    assert model.model.total_ut_steps==4
    manifest=dict(pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],backbone_sha256=CK_SHA,
        j_sha256=JSHA,script_sha256=digest(Path(__file__)),physical_loops=4,seeds=EVAL_SEEDS,
        conditions=CONDITIONS,k=[5,6,7,8],n_per_cell=64,batch=16,max_new_tokens=64,
        training=False,j_shared=True,intervention='bypass J, never zero hidden state',
        positions='J1 after loop1 RMSNorm through J4 after loop4 RMSNorm before LM head',
        device_sharing='GPU6 empty at launch; >=16GiB reserve monitored')
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
    cache={}
    for k in range(5,9):
        rr=[]
        for i,seed in enumerate(EVAL_SEEDS):
            prompt,answer,_=task(seed,k,0)
            ids=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,add_generation_prompt=True)
            rr.append((i,seed,ids,answer))
        cache[k]=rr
    results=[];started=time.time()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for name,sites in CONDITIONS.items():
            active.clear();active.update(sites)
            for k in range(5,9):
                cell=[]
                for off in range(0,64,16):
                    if torch.cuda.mem_get_info()[0]<16*2**30:raise RuntimeError('Shared GPU reserve breached')
                    rr=cache[k][off:off+16]
                    b=tok.pad(dict(input_ids=[r[2] for r in rr]),padding=True,return_tensors='pt').to('cuda')
                    trace.clear()
                    out=model.generate(**b,exit_at_step=3,max_new_tokens=64,do_sample=False,
                        use_cache=True,pad_token_id=tok.pad_token_id)
                    if sites:
                        assert len(trace)%len(sites)==0 and trace==sites*(len(trace)//len(sites))
                    else:assert not trace
                    for (i,seed,_,answer),ids in zip(rr,out[:,b.input_ids.shape[1]:]):
                        text=tok.decode(ids,skip_special_tokens=True).strip()
                        parsed=parse(dict(kind='cancellation',generated=text))
                        r=dict(condition=name,k=k,i=i,seed=seed,answer=answer,generated=text,
                            parsed=parsed,correct=parsed==answer)
                        cell.append(r);results.append(r)
                        with (ROOT/'outputs.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
                print(json.dumps(dict(condition=name,k=k,correct=sum(r['correct'] for r in cell),n=64,
                    elapsed_seconds=time.time()-started)),flush=True)
    assert versions==[x._version for x in model.parameters()]+[x._version for x in j.parameters()]
    index={(r['condition'],r['k'],r['i']):r for r in results};assert len(index)==2560
    summary={}
    for name in CONDITIONS:
        summary[name]={}
        for k in range(5,9):
            pairs=[(index['all',k,i],index[name,k,i]) for i in range(64)]
            correct=sum(b['correct'] for a,b in pairs)
            summary[name][str(k)]=dict(correct=correct,n=64,accuracy=correct/64,
                breaks=sum(a['correct'] and not b['correct'] for a,b in pairs),
                repairs=sum(not a['correct'] and b['correct'] for a,b in pairs))
    report=dict(status='complete',results=summary,seconds=time.time()-started,
        peak_gib=torch.cuda.max_memory_allocated()/2**30,parameters_unchanged=True)
    (ROOT/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
