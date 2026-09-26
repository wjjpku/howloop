"""Localize the established head by physical loop, positions and generation phase."""
import json,os,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from diagnose_ouro_heads import BASE,JPATH,JSHA,digest
from train_ouro_four_j import Affine,CK_SHA
from train_ouro_full import MODEL,task
from ouro_eval_panel import EVAL_SEEDS
from score_ouro_content import parse

ROOT=Path('/data/paperexperiment/ouro_l8h5_native_signals_v2_20260916')
ARMS={'plain4':(False,4),'j4':(True,4),'j8':(True,8)}

def main():
    ROOT.mkdir(exist_ok=False);torch.set_num_threads(4);torch.manual_seed(20260916)
    assert digest(BASE)==CK_SHA and digest(JPATH)==JSHA
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tok.padding_side='left'
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True,torch_dtype=torch.float32,attn_implementation='sdpa')
    ck=torch.load(BASE,map_location='cpu',weights_only=False,mmap=True);model.load_state_dict(ck['model']);del ck
    model=model.to('cuda').eval().requires_grad_(False);model.model.total_ut_steps=4;model.config.total_ut_steps=4
    j=Affine(2048);ck=torch.load(JPATH,map_location='cpu',weights_only=False);j.load_state_dict(ck['affine']);del ck
    j=j.to('cuda').eval().requires_grad_(False)
    state=dict(loop=0,prefill=True,condition=None,use_j=False,hits=0,zeroed=0,mask=None)
    def reset(m,args,kwargs):
        state['loop']=0
        ids=kwargs.get('input_ids',args[0] if args else None)
        assert ids is not None
        state['prefill']=ids.shape[1]>1
    def norm(m,args,h):
        state['loop']+=1
        return j(h) if state['use_j'] else h
    def intervention(m,args):
        c=state['condition'];loop=state['loop']+1
        if c is None or c['loop'] not in [0,loop]:return
        phase='prefill' if state['prefill'] else 'decode'
        if c['phase'] not in ['both',phase]:return
        h=args[0];v=h.reshape(*h.shape[:-1],16,128).clone()
        mask=state['mask'][c['region']] if state['prefill'] else torch.ones(h.shape[:2],device=h.device,dtype=torch.bool)
        assert mask.shape==h.shape[:2]
        v[:,:,4,:].masked_fill_(mask[:,:,None],0)
        state['hits']+=1;state['zeroed']+=int(mask.sum())
        return (v.reshape_as(h),)
    model.model.register_forward_pre_hook(reset,with_kwargs=True)
    model.model.norm.register_forward_hook(norm)
    model.model.layers[7].self_attn.o_proj.register_forward_pre_hook(intervention)
    versions=[p._version for p in list(model.parameters())+list(j.parameters())]
    cache={}
    for arm,(_,k) in ARMS.items():
        rr=[]
        for seed in EVAL_SEEDS:
            prompt,answer,_=task(seed,k,0)
            text=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True)
            enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
            ids=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,add_generation_prompt=True)
            assert ids==enc.input_ids
            start=text.index('Word sequence: ')+len('Word sequence: ');end=text.index('.',start)
            words=[i for i,(a,b) in enumerate(enc.offset_mapping) if a<end and b>start]
            assert words and max(words)<len(ids)-1
            rr.append(dict(seed=seed,ids=ids,words=words,answer=answer,prompt=prompt))
        cache[arm]=rr
    manifest=dict(pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],backbone_sha256=CK_SHA,J_sha256=JSHA,
        script_sha256=digest(Path(__file__)),head='L8.H5 one-based',head_slice=[512,640],intervention='zero head output before o_proj',
        discovery_seeds=EVAL_SEEDS[:16],confirmation_seeds=EVAL_SEEDS[16:],arms=ARMS,loops=4,batch=8,max_new_tokens=64,
        phase_definition='prefill includes first answer-token decision; decode is subsequent cached answer-token computation',
        positions='word-span token offsets; final chat prompt token; remaining non-padding prompt tokens',
        selection='top two positive accuracy-drop localized conditions per arm, union; independent confirmation of union',
        semantic_claim='none: physical loop is not assumed equal to algorithm deletion step',training=False)
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
    def evaluate(arm,c,indices):
        state.update(use_j=ARMS[arm][0],condition=c,hits=0,zeroed=0);outrows=[]
        for off in range(0,len(indices),8):
            if torch.cuda.mem_get_info()[0]<16*2**30:raise RuntimeError('GPU reserve breached')
            rr=[cache[arm][i] for i in indices[off:off+8]]
            batch=tok.pad(dict(input_ids=[r['ids'] for r in rr]),padding=True,return_tensors='pt').to('cuda')
            valid=batch.attention_mask.bool();words=torch.zeros_like(valid);last=torch.zeros_like(valid)
            for n,r in enumerate(rr):words[n,[i+valid.shape[1]-len(r['ids']) for i in r['words']]]=True
            last[:,-1]=True
            state['mask']=dict(words=words,last=last,other=valid&~words&~last,all=valid)
            result=model.generate(**batch,exit_at_step=3,max_new_tokens=64,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
            for r,ids in zip(rr,result[:,batch.input_ids.shape[1]:]):
                text=tok.decode(ids,skip_special_tokens=True).strip();parsed=parse(dict(kind='cancellation',generated=text))
                outrows.append(dict(seed=r['seed'],expected=r['answer'],generated=text,parsed=parsed,correct=parsed==r['answer']))
        if c is not None:assert state['hits']>0 and state['zeroed']>0
        return dict(arm=arm,condition=c,n=len(indices),correct=sum(r['correct'] for r in outrows),rows=outrows,hits=state['hits'],zeroed_positions=state['zeroed'])
    def score(row,base):
        old={r['seed']:r['correct'] for r in base['rows']}
        row.update(drop_pp=100*(base['correct']-row['correct'])/row['n'],breaks=sum(old[r['seed']] and not r['correct'] for r in row['rows']),repairs=sum(not old[r['seed']] and r['correct'] for r in row['rows']),baseline_correct=base['correct'])
        return row
    def save(file,row):
        with (ROOT/file).open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='rows'}),flush=True)

    # Collect descriptive candidate signals only. Returning the original SDPA output
    # preserves the computation; attention statistics do not establish causal use.
    import importlib
    from task import trace as oracle_trace,OPPOSITE
    mod=importlib.import_module(type(model.model.layers[7].self_attn).__module__)
    original=mod.ALL_ATTENTION_FUNCTIONS['sdpa']
    context={};signals=[]
    def probe(module,q,k,v,attention_mask,**kwargs):
        result=original(module,q,k,v,attention_mask,**kwargs)
        if module.layer_idx!=7:return result
        loop=state['loop']+1
        kh=k.repeat_interleave(q.shape[1]//k.shape[1],dim=1)
        logits=(q[:,4].float()@kh[:,4].float().transpose(-1,-2))*kwargs['scaling']
        n=logits.shape[-1]
        if attention_mask is None:
            logits.masked_fill_(torch.ones(n,n,device=q.device,dtype=torch.bool).triu(1),-float('inf'))
        elif attention_mask.dtype==torch.bool:logits.masked_fill_(~attention_mask[:,0],-float('inf'))
        else:logits+=attention_mask[:,0,:,:n].float()
        prob=torch.softmax(logits,dim=-1)[0].cpu()
        positions=context['positions'];words=context['words']
        for wi,token in enumerate(positions):
            past=[j for j in range(wi+1) if words[j]==OPPOSITE[words[wi]]]
            rec=dict(arm=context['arm'],seed=context['seed'],loop=loop,word_index=wi,word=words[wi],
                self_mass=prob[token,token].item(),
                previous_word_mass=prob[token,positions[wi-1]].item() if wi else None,
                partner_mass=prob[token,positions[past[0]]].item() if past else None,
                past_partner_exists=bool(past),
                most_attended_word=int(prob[token,positions].argmax()),
                algorithm_deletion_step=context['deletion_steps'][wi],
                removed_mass_by_stage=[sum(prob[token,positions[j]].item() for j in range(24) if context['deletion_steps'][j]<=s) for s in range(1,9)])
            signals.append(rec)
        return result
    mod.ALL_ATTENTION_FUNCTIONS.register('ouro_semantic_probe',probe)
    model.config._attn_implementation='ouro_semantic_probe'
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for arm in ARMS:
            state.update(use_j=ARMS[arm][0],condition=None)
            for i,r in enumerate(cache[arm]):
                words=r['prompt'].split('Word sequence: ',1)[1].split('.',1)[0].split()
                # Existing word list has exactly one token per word in these prompts.
                assert len(r['words'])==len(words)==24
                live=list(range(24));stages=[None]*24
                for step in range(1,13):
                    pair_index=next(index for index in range(len(live)-1) if OPPOSITE[words[live[index]]]==words[live[index+1]])
                    for ix in live[pair_index:pair_index+2]:stages[ix]=step
                    del live[pair_index:pair_index+2]
                context.update(arm=arm,seed=r['seed'],positions=r['words'],words=words,deletion_steps=stages)
                ids=torch.tensor([r['ids']],device='cuda')
                _,hs,_=model.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
                assert len(hs)==4
                if (i+1)%16==0:print(json.dumps(dict(event='native_signal_progress',arm=arm,done=i+1)),flush=True)
    mod.ALL_ATTENTION_FUNCTIONS.register('sdpa',original)
    assert versions==[p._version for p in list(model.parameters())+list(j.parameters())]
    with (ROOT/'signals.jsonl').open('w') as f:
        for row in signals:f.write(json.dumps(row)+'\n')
    summary=dict(status='complete',n_records=len(signals),parameters_unchanged=True,
        evidence_level='descriptive candidates only; no semantic substitution or native rescue established')
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
