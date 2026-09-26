"""Coarse downstream rescue localization, then locked fresh confirmation."""
import json, os, time, sys
from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from diagnose_ouro_heads import BASE,JPATH,JSHA,digest
from train_ouro_four_j import Affine,CK_SHA
from train_ouro_full import MODEL,task
from score_ouro_content import parse

FINE='--fine' in sys.argv
ROOT=Path('/data/paperexperiment/ouro_rescue_fine_20260916' if FINE else '/data/paperexperiment/ouro_rescue_localization_20260916')

def main():
    ROOT.mkdir(exist_ok=False);torch.set_num_threads(4)
    assert digest(BASE)==CK_SHA and digest(JPATH)==JSHA
    prior=json.loads(Path('/data/paperexperiment/ouro_native_causal_20260916/manifest.json').read_text())
    discovery=[p['recipient'] for p in prior['pairs'][:16]]
    seen={task(p[key],4,0)[2] for p in prior['pairs'] for key in ('recipient','donor')}
    if FINE:
        coarse=json.loads(Path('/data/paperexperiment/ouro_rescue_localization_20260916/manifest.json').read_text())
        seen.update(task(s,8,0)[2] for s in coarse['confirmation'])
    confirmation=[]
    for seed in range(9160000 if FINE else 9140000,9199000):
        sig=task(seed,8,0)[2]
        if sig not in seen:
            seen.add(sig);confirmation.append(seed)
        if len(confirmation)==48:break
    conditions=[]
    for skip in (2,3):
        for phase in ('prefill','decode','both'):
            conditions.append(dict(name=f'J{skip}_skip_{phase}',skip=skip,phase=phase,kind=None))
        for region in ('words','last','all'):
            for layer in (8,24,48):
                conditions.append(dict(name=f'J{skip}_residual_L{layer}_{region}',skip=skip,phase='both',kind='residual',layers=[layer],region=region))
            for kind in ('attention','mlp'):
                for lo,hi in ((1,16),(17,32),(33,48)):
                    conditions.append(dict(name=f'J{skip}_{kind}_L{lo}-{hi}_{region}',skip=skip,phase='both',kind=kind,layers=list(range(lo,hi+1)),region=region))
    donors={}
    if FINE:
        conditions=[c for c in conditions if c['kind'] is None]
        for skip,start,region in ((2,33,'words'),(3,1,'last')):
            for width in (1,4,8,16):
                for lo in range(start,start+16,width):
                    hi=lo+width-1
                    conditions.append(dict(name=f'J{skip}_attention_L{lo}-{hi}_{region}',skip=skip,phase='both',kind='attention',layers=list(range(lo,hi+1)),region=region))
        pool=iter(range(9170000,9199000))
        for seed in confirmation:
            receiver_answer=task(seed,8,0)[1]
            for other in pool:
                _,answer,sig=task(other,8,0)
                if sig not in seen and answer!=receiver_answer:
                    seen.add(sig);donors[seed]=other;break
        assert len(donors)==48
    manifest=dict(pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],backbone_sha256=CK_SHA,J_sha256=JSHA,
        script_sha256=digest(Path(__file__)),discovery=discovery,confirmation=confirmation,conditions=conditions,
        loops=4,k=8,training=False,source='same-input normal J prefill',
        scope='Only physical loop immediately after bypassed J; prefill patches only; no J output patch',
        selection='Per J, best positive net-rescue residual and best positive net-rescue component; ties prefer fewer positions then fewer layers then name',
        limitation='Whole residual restoration is coarse localization, not semantic reuse; no head scan')
    if FINE:
        manifest.update(stage='fine attention localization',wrong_donors=donors,
            selection='Per J select fewest layers retaining >=80% of best discovery net gain; tie higher accuracy then layer index. Also retain full 16-layer anchor.',
            source_tests='Normal J same-input k8; wrong-input J k8; same-input no-J k8; same-input no-J k4',
            native_caveat='No-J k8 may fail. No-J k4 changes query; at last prompt position it is a target-mismatch control, NOT a semantically matched operation. Word-position causal prefix is unchanged.')
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2));print('manifest ready',flush=True)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True,torch_dtype=torch.float32,attn_implementation='sdpa')
    ck=torch.load(BASE,map_location='cpu',weights_only=False,mmap=True);model.load_state_dict(ck['model']);del ck
    model.to('cuda').eval().requires_grad_(False);model.model.total_ut_steps=model.config.total_ut_steps=4
    j=Affine(2048);ck=torch.load(JPATH,map_location='cpu',weights_only=False);j.load_state_dict(ck['affine']);del ck
    j.to('cuda').eval().requires_grad_(False)
    versions=[p._version for p in list(model.parameters())+list(j.parameters())]
    state={}
    def reset(m,args,kwargs):
        ids=kwargs.get('input_ids',args[0] if args else None)
        state.update(loop=0,prefill=ids.shape[1]>1)
    def norm(m,args,h):
        state['loop']+=1;c=state['condition']
        phase='prefill' if state['prefill'] else 'decode'
        skip=c and c['skip']==state['loop'] and c['phase'] in ('both',phase)
        return h if skip or not state['use_j'] else j(h)
    model.model.register_forward_pre_hook(reset,with_kwargs=True);model.model.norm.register_forward_hook(norm)
    def hook(kind,layer):
        def apply(m,args,h):
            loop=state['loop']+1
            if not state['prefill'] or loop not in (3,4):return
            key=(loop,kind,layer)
            if state['collect']:
                state['cache'][key]=h.detach().clone()
            c=state['condition']
            if not c or c['kind']!=kind or loop!=c['skip']+1 or layer not in c['layers']:return
            out=h.clone();positions=state['regions'][c['region']]
            assert state['source'][key].shape==h.shape, 'donor token alignment mismatch'
            out[:,positions]=state['source'][key][:,positions]
            state['hits']+=1
            return out
        return apply
    for idx,layer in enumerate(model.model.layers,1):
        # Patch normalized branch contributions, immediately before residual addition.
        layer.input_layernorm_2.register_forward_hook(hook('attention',idx))
        layer.post_attention_layernorm_2.register_forward_hook(hook('mlp',idx))
        if idx in (8,24,48):layer.register_forward_hook(hook('residual',idx))
    def run(seed,c=None,collect=False,source=None,use_j=True,k=8):
        if torch.cuda.mem_get_info()[0]<16*2**30:raise RuntimeError('GPU reserve breached')
        prompt,answer,sig=task(seed,k,0)
        text=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True)
        enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
        ids=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,add_generation_prompt=True)
        assert ids==enc.input_ids
        start=text.index('Word sequence: ')+len('Word sequence: ');end=text.index('.',start)
        positions=[i for i,(a,b) in enumerate(enc.offset_mapping) if a<end and b>start];assert len(positions)==24
        state.update(condition=c,collect=collect,source=source,cache={},hits=0,use_j=use_j,
            regions=dict(words=positions,last=[len(ids)-1],all=list(range(len(ids)))))
        ids=torch.tensor([ids],device='cuda')
        result=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),exit_at_step=3,max_new_tokens=64,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
        generated=tok.decode(result[0,ids.shape[1]:],skip_special_tokens=True).strip()
        if c and c['kind']:assert state['hits']==len(c['layers'])
        parsed=parse(dict(kind='cancellation',generated=generated))
        return dict(seed=seed,expected=answer,generated=generated,parsed=parsed,correct=parsed==answer,hits=state['hits'],tokens=result.shape[1]-ids.shape[1]),state['cache']
    rows=[];start=time.time()
    def save(split,name,row):
        r=dict(split=split,condition=name,**row);rows.append(r)
        with (ROOT/'records.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
    def panel(split,seeds,selected):
        for index,seed in enumerate(seeds):
            normal,source=run(seed,collect=True);save(split,'normal',normal)
            # Exact identity control at a broad boundary, without bypassing any real J.
            c=dict(skip=0,phase='both',kind='residual',layers=[8],region='all')
            # Use same active loop as J2 downstream, but disable bypass via phase sentinel.
            c.update(skip=2,phase='none')
            identity,_=run(seed,c,source=source);assert identity['generated']==normal['generated']
            save(split,'self_residual_L8',identity)
            for c in selected:
                row,_=run(seed,c,source=source);save(split,c['name'],row)
            if FINE and split=='confirmation':
                for label,donor_seed,use_j,k in [('wrong_J_k8',donors[seed],True,8),('native_k8',seed,False,8),('native_k4',seed,False,4)]:
                    donor,activities=run(donor_seed,collect=True,use_j=use_j,k=k)
                    save(split,'donor_'+label,dict(donor,recipient_seed=seed))
                    for c in selected:
                        if not c['kind']:continue
                        row,_=run(seed,c,source=activities)
                        row.update(donor_seed=donor_seed,donor_expected=donor['expected'],donor_generated=donor['generated'],donor_correct=donor['correct'])
                        save(split,c['name']+'_'+label,row)
                    del activities
            del source
            print(json.dumps(dict(event='sample_complete',split=split,done=index+1,total=len(seeds),elapsed_s=time.time()-start,peak_GiB=torch.cuda.max_memory_allocated()/2**30)),flush=True)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        panel('discovery',discovery,conditions)
        counts=defaultdict(int)
        for r in rows:counts[r['condition']]+=r['correct']
        chosen=[]
        for skip in (2,3):
            chosen.extend(c for c in conditions if c['skip']==skip and c['kind'] is None)
            baseline=counts[f'J{skip}_skip_both']
            if FINE:
                candidates=[c for c in conditions if c['skip']==skip and c['kind']]
                gain=max(counts[c['name']]-baseline for c in candidates)
                eligible=[c for c in candidates if gain>0 and counts[c['name']]-baseline>=0.8*gain]
                eligible.sort(key=lambda c:(len(c['layers']),-counts[c['name']],c['layers'][0]))
                if eligible:chosen.append(eligible[0])
                anchor=next(c for c in candidates if len(c['layers'])==16)
                if anchor not in chosen:chosen.append(anchor)
                continue
            for family in ('residual','component'):
                candidates=[c for c in conditions if c['skip']==skip and c['kind'] and (c['kind']=='residual')==(family=='residual') and counts[c['name']]>baseline]
                candidates.sort(key=lambda c:(-counts[c['name']],{'last':1,'words':24,'all':1000}[c['region']],len(c['layers']),c['name']))
                if candidates:chosen.append(candidates[0])
        (ROOT/'locked_selection.json').write_text(json.dumps(dict(selected=chosen,discovery_counts=counts),indent=2))
        print(json.dumps(dict(event='selection_locked',selected=chosen)),flush=True)
        panel('confirmation',confirmation,chosen)
    assert versions==[p._version for p in list(model.parameters())+list(j.parameters())]
    totals=defaultdict(lambda:dict(n=0,correct=0))
    for r in rows:
        t=totals[r['split']+'|'+r['condition']];t['n']+=1;t['correct']+=r['correct']
    summary=dict(status='complete',parameters_unchanged=True,elapsed_s=time.time()-start,totals=totals,selected=chosen)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
