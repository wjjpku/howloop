"""Exhaustive single-head causal search, followed by independent confirmation.

Every one of the 768 weight-shared heads is zeroed in ALL four loops.
Selection uses greedy answer accuracy loss, not activation magnitude.
Two pinned independent workers split heads, not model parameters.
"""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from diagnose_ouro_heads import BASE,JPATH,JSHA,digest
from train_ouro_four_j import Affine,CK_SHA
from train_ouro_full import MODEL,task
from ouro_eval_panel import EVAL_SEEDS
from score_ouro_content import parse

ROOT=Path('/data/wujiaju/ouro26_causal_head_scan_20260915')
ARMS={'plain4':(False,4),'j4':(True,4),'j8':(True,8)}

def select_candidates(root):
    rows=[]
    for worker in [0,1]:
        assert (root/f'worker{worker}'/'screen_complete.json').exists()
        rows += [json.loads(l) for l in (root/f'worker{worker}'/'screen.jsonl').read_text().splitlines()]
    selected=set(); rankings={}
    for arm in ARMS:
        rr=[r for r in rows if r['arm']==arm and r['head_id']>=0]
        assert len(rr)==768 and len({r['head_id'] for r in rr})==768
        # Positive net accuracy loss is required: never label a zero-effect head critical.
        rr.sort(key=lambda r:(r['accuracy_drop_pp'],r['breaks'],r['first_token_logp_drop']),reverse=True)
        positive=[r for r in rr if r['accuracy_drop_pp']>0]
        rankings[arm]=positive
        selected.update(r['head_id'] for r in positive[:8])
    return sorted(selected),rankings

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int,choices=[0,1],required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==1
    out=ROOT/(f'smoke{a.worker}' if a.smoke else f'worker{a.worker}')
    out.mkdir(parents=True,exist_ok=a.resume)
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
    assert model.model.total_ut_steps==4 and len(model.model.layers)==48
    use_j=[False];head_id=[-1];hits=[0]
    model.model.norm.register_forward_hook(lambda m,args,h:j(h) if use_j[0] else h)
    def hook(layer):
        def apply(m,args):
            if head_id[0]<0 or head_id[0]//16!=layer:return
            h=args[0];v=h.reshape(*h.shape[:-1],16,128).clone()
            v[:,:,head_id[0]%16,:]=0;hits[0]+=1
            return (v.reshape_as(h),)
        return apply
    for layer,block in enumerate(model.model.layers):
        block.self_attn.o_proj.register_forward_pre_hook(hook(layer))
    versions=[x._version for x in model.parameters()]
    manifest=dict(pid=os.getpid(),worker=a.worker,gpu=os.environ['CUDA_VISIBLE_DEVICES'],
        backbone_sha256=CK_SHA,j_sha256=JSHA,script_sha256=digest(Path(__file__)),
        head_definition='layer*16+head, zero pre-o_proj at all positions in all four loops',
        physical_loops=4,training=False,discovery_seeds=EVAL_SEEDS[:16],confirmation_seeds=EVAL_SEEDS[16:],
        rank='greedy parsed answer accuracy drop, then breaks, then first-answer-token logp drop',
        selection='positive accuracy-drop heads only, top8 per arm; union cross-tested in all arms',
        generation=dict(max_new_tokens=64,do_sample=False,batch=16),
        metrics='answer match, not token accuracy or teacher-forced accuracy; all trials retained',
        limitations='single-head zero ablation; redundancy and post-attention normalization remain')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(event='loaded',**manifest)),flush=True)
    cache={}
    for arm,(_,k) in ARMS.items():
        rows=[]
        for seed in EVAL_SEEDS:
            prompt,answer,_=task(seed,k,0)
            ids=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,add_generation_prompt=True)
            rows.append((ids,answer))
        cache[arm]=rows
    def evaluate(arm,hid,indices):
        use_j[0]=ARMS[arm][0];head_id[0]=hid;hits[0]=0;res=[];started=time.time()
        for off in range(0,len(indices),16):
            if torch.cuda.mem_get_info()[0]<16*2**30:raise RuntimeError('GPU reserve breached')
            ii=indices[off:off+16];rr=[cache[arm][i] for i in ii]
            batch=tok.pad(dict(input_ids=[r[0] for r in rr]),padding=True,return_tensors='pt').to('cuda')
            result=model.generate(**batch,exit_at_step=3,max_new_tokens=64,do_sample=False,
                use_cache=True,pad_token_id=tok.pad_token_id,return_dict_in_generate=True,output_scores=True)
            first=result.scores[0].float().log_softmax(-1)
            for n,(i,(_,answer),ids) in enumerate(zip(ii,rr,result.sequences[:,batch.input_ids.shape[1]:])):
                text=tok.decode(ids,skip_special_tokens=True).strip()
                parsed=parse(dict(kind='cancellation',generated=text))
                target=tok.encode(answer,add_special_tokens=False)[0]
                res.append(dict(i=i,answer=answer,generated=text,parsed=parsed,correct=parsed==answer,
                    first_target_logp=first[n,target].item()))
            del result,first
        assert hid<0 or (hits[0]>=4 and hits[0]%4==0),hits
        return dict(arm=arm,head_id=hid,layer=hid//16,head=hid%16,n=len(res),
            correct=sum(r['correct'] for r in res),rows=res,seconds=time.time()-started,hook_calls=hits[0])
    def score(row,base):
        old={r['i']:r for r in base['rows']}
        row['accuracy_drop_pp']=100*(base['correct']-row['correct'])/row['n']
        row['breaks']=sum(old[r['i']]['correct'] and not r['correct'] for r in row['rows'])
        row['repairs']=sum(not old[r['i']]['correct'] and r['correct'] for r in row['rows'])
        row['first_token_logp_drop']=float(np.mean([old[r['i']]['first_target_logp']-r['first_target_logp'] for r in row['rows']]))
        return row
    def save(file,row):
        with (out/file).open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='rows'}),flush=True)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        baselines={arm:evaluate(arm,-1,list(range(16))) for arm in ARMS}
        for row in baselines.values():save('baseline.jsonl',row)
        # Explicit zero-head smoke checks that interventions are actually applied.
        if a.smoke:
            for arm in ARMS:
                for hid in [0,767]:save('screen.jsonl',score(evaluate(arm,hid,list(range(16))),baselines[arm]))
            summary=dict(peak_gib=torch.cuda.max_memory_allocated()/2**30,status='smoke_complete')
            (out/'complete.json').write_text(json.dumps(summary));print(json.dumps(summary),flush=True);return
        completed=set()
        if a.resume and (out/'screen.jsonl').exists():
            completed={(r['arm'],r['head_id']) for r in [json.loads(l) for l in (out/'screen.jsonl').read_text().splitlines()]}
        order=np.random.default_rng(20260915).permutation(768).tolist()[a.worker::2]
        started=time.time()
        for idx,hid in enumerate(order):
            for arm in ARMS:
                if (arm,hid) in completed:continue
                save('screen.jsonl',score(evaluate(arm,hid,list(range(16))),baselines[arm]))
            if idx%8==0:print(json.dumps(dict(event='progress',heads=idx+1,total=384,seconds=time.time()-started)),flush=True)
        (out/'screen_complete.json').write_text(json.dumps(dict(seconds=time.time()-started)))
        deadline=time.time()+4*3600
        while not all((ROOT/f'worker{w}'/'screen_complete.json').exists() for w in [0,1]):
            if time.time()>deadline:raise RuntimeError('Peer scan did not finish within bounded wait')
            time.sleep(15)
        candidates,rankings=select_candidates(ROOT)
        (out/'selected.json').write_text(json.dumps(dict(candidates=candidates,ranked_ids={a:[r['head_id'] for r in rr] for a,rr in rankings.items()}),indent=2))
        confirmation={arm:evaluate(arm,-1,list(range(16,64))) for arm in ARMS}
        for row in confirmation.values():save('confirmation_baseline.jsonl',row)
        for hid in candidates[a.worker::2]:
            for arm in ARMS:save('confirmation.jsonl',score(evaluate(arm,hid,list(range(16,64))),confirmation[arm]))
        assert versions==[x._version for x in model.parameters()]
        (out/'complete.json').write_text(json.dumps(dict(status='complete',candidates=candidates,
            peak_gib=torch.cuda.max_memory_allocated()/2**30,backbone_unchanged=True)))
        print('COMPLETE',flush=True)

if __name__=='__main__':main()
