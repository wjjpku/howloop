"""Original Ouro-2.6B: paired loop readouts across eight task families; no training."""
import argparse,hashlib,json,os,random,re,subprocess,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM

MODEL='/data/wujiaju/models/Ouro-2.6B'
ROOT=Path('/data/wujiaju/ouro_raw_early_answers_20260916')
DEPENDENCY=Path('/data/wujiaju/ouro26_stepwise_pair_control_20260915/vocab_transfer_20260916/summary.json')

def dataset():
    rng=random.Random(20260916);rows=[]
    def add(family,level,prompt,answer):
        rows.append(dict(id=len(rows),family=family,level=level,prompt=prompt+' Answer directly with only the answer. Do not explain.',answer=str(answer)))
    days=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    directions=['north','east','south','west'];names=['Alice','Bob','Carol','David','Emma','Frank','Grace','Henry','Iris','Jack']
    for level in [1,2,3]:
        for rep in range(16):
            a,b,c=[rng.randint(2,9 if level==1 else 49 if level==2 else 199) for _ in range(3)]
            add('arithmetic',level,f'Calculate {a} + {b}.' if level==1 else f'Calculate ({a} + {b}) - {c}.' if level==2 else f'Calculate ({a} + {b}) * {c}.',a+b if level==1 else a+b-c if level==2 else (a+b)*c)
            d=rng.randrange(7);n=rng.randint(1,[6,28,180][level-1])
            add('weekday',level,f'Today is {days[d]}. What day of the week will it be after {n} days?',days[(d+n)%7])
            d=rng.randrange(4);turns=[rng.choice([-1,1]) for _ in range([1,4,8][level-1])]
            add('rotation',level,f'You face {directions[d]}. Turn '+', then '.join('90 degrees right' if t==1 else '90 degrees left' for t in turns)+'. Which direction do you now face?',directions[(d+sum(turns))%4])
            on=rng.randrange(2);n=rng.randint(1,[3,12,70][level-1])
            add('switch',level,f'A switch starts {"on" if on else "off"}. Each toggle reverses its state. After {n} toggles, is it on or off?','on' if (on+n)%2 else 'off')
            ns=rng.sample(names,[3,5,8][level-1]);facts=[f'{ns[i]} is taller than {ns[i+1]}.' for i in range(len(ns)-1)];rng.shuffle(facts)
            p,q=rng.sample(range(len(ns)),2)
            add('order_chain',level,' '.join(facts)+f' Who is taller, {ns[p]} or {ns[q]}?',ns[min(p,q)])
            symbols=rng.sample(list('ABCDEFGH'),8);mapping=dict(zip(symbols,symbols[1:]+symbols[:1]));start=rng.choice(symbols);n=[1,3,7][level-1];state=start
            for _ in range(n):state=mapping[state]
            rules=list(mapping.items());rng.shuffle(rules)
            add('symbol_updates',level,'Replacement rules: '+', '.join(f'{x}->{y}' for x,y in rules)+f'. Start with {start} and apply the rule {n} times. What symbol results?',state)
            items=rng.sample(list('ABCDEFGH'),6);live=items[:];swaps=[]
            for _ in range([1,3,6][level-1]):
                p,q=rng.sample(range(6),2);live[p],live[q]=live[q],live[p];swaps.append(f'{p+1} and {q+1}')
            pos=rng.randrange(6)
            add('position_swaps',level,'List: '+' '.join(items)+'. Positions are numbered from 1. In order, swap positions '+'; then '.join(swaps)+f'. What is at position {pos+1}?',live[pos])
            who=rng.sample(names,4);places=rng.sample(['kitchen','garden','office','bedroom','hallway','garage'],4);target=rng.randrange(4)
            facts=[f'{w} is in the {p}.' for w,p in zip(who,places)]
            if level==1:prompt=' '.join(facts)+f' Where is {who[target]}?';ans=places[target]
            elif level==2:prompt=' '.join(facts)+f' The key is with {who[target]}. In which room or place is the key?';ans=places[target]
            else:
                other=(target+1)%4;prompt=' '.join(facts)+f' The key is with {who[target]}. {who[target]} gives the key to {who[other]}. In which room or place is the key now?';ans=places[other]
            add('short_qa',level,prompt,ans)
    assert len(rows)==384 and all(n==48 for n in Counter(r['family'] for r in rows).values())
    return rows

def normalize(text):return re.sub(r'[.!?]+$','',text.strip().lower()).strip()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    rows=dataset()
    assert normalize(' Monday. ')==normalize('Monday') and normalize('Monday because')!='monday'
    if args.self_test:
        print(json.dumps(dict(n=len(rows),families=dict(Counter(r['family'] for r in rows)),examples=rows[:8])));return
    ROOT.mkdir(exist_ok=False)
    manifest=dict(status='queued',pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],dependency=str(DEPENDENCY),model=MODEL,
        original_pretrained=True,J=False,training=False,gate_used=False,loops=[1,2,3,4],n=384,n_per_family=48,
        levels=[1,2,3],seed=20260916,batch_size=8,max_new_tokens=64,do_sample=False,
        task_data_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        caveats=['Readout persistence is not a hidden-state fixed point or proof of no subsequent computation.',
                 'Same canonical answer prefix for teacher-forced scores; generated-prefix trajectories are separate.',
                 'Synthetic convenience sample, not a representative language benchmark; difficulty labels are family-specific.',
                 'Input lengths vary with task difficulty; this study is not fixed-token length generalization.'])
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2));(ROOT/'tasks.json').write_text(json.dumps(rows,indent=2))
    deadline=time.time()+7200
    while not (DEPENDENCY.exists() and json.loads(DEPENDENCY.read_text()).get('status')=='complete'):
        if time.time()>deadline:raise RuntimeError('Dependency timeout; did not allocate GPU')
        print('Queued behind vocabulary evaluation; no GPU allocated',flush=True);time.sleep(30)
    # Require two empty observations; do not interfere with another workload.
    empty_since=None
    while True:
        used=int(subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if used<100:
            if empty_since is None:empty_since=time.time()
            if time.time()-empty_since>=60:break
        else:empty_since=None
        if time.time()>deadline:raise RuntimeError('GPU never became safely available')
        time.sleep(15)
    torch.set_num_threads(4);torch.manual_seed(20260916)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tok.padding_side='left'
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
    manifest.update(status='running',model_config=model.config.to_dict(),model_files={p.name:p.stat().st_size for p in Path(MODEL).glob('*.safetensors')})
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2));print('Original non-thinking Ouro loaded; starting readout evaluation',flush=True)
    versions=[p._version for p in model.parameters()];records={r['id']:dict(**r,readouts=[],canonical_scores=[],prompt_state_change=[]) for r in rows}
    def save_record(rec):
        with (ROOT/'outputs.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
    started=time.time()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for depth in range(1,5):
            model.model.total_ut_steps=depth;model.config.total_ut_steps=depth
            for off in range(0,len(rows),8):
                if torch.cuda.mem_get_info()[0]<16*2**30:raise RuntimeError('Memory reserve breached')
                rr=rows[off:off+8];prompts=[tok.apply_chat_template([dict(role='user',content=r['prompt'])],tokenize=True,add_generation_prompt=True) for r in rr]
                batch=tok.pad(dict(input_ids=prompts),padding=True,return_tensors='pt').to('cuda')
                out=model.generate(**batch,exit_at_step=depth-1,max_new_tokens=64,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
                for r,ids,p in zip(rr,out[:,batch.input_ids.shape[1]:],prompts):
                    text=tok.decode(ids,skip_special_tokens=True).strip();readout=dict(loop=depth,generated=text,normalized=normalize(text),correct=normalize(text)==normalize(r['answer']),prompt_tokens=len(p),hit_cap=len(ids)==64 and tok.eos_token_id not in ids.tolist())
                    records[r['id']]['readouts'].append(readout);save_record(dict(event='generation',id=r['id'],**readout))
            print(json.dumps(dict(event='loop_complete',loop=depth,seconds=time.time()-started)),flush=True)
        model.model.total_ut_steps=4;model.config.total_ut_steps=4
        for r in rows:
            prefix=tok.apply_chat_template([dict(role='user',content=r['prompt'])],tokenize=True,add_generation_prompt=True)
            full=tok.apply_chat_template([dict(role='user',content=r['prompt']),dict(role='assistant',content=r['answer'])],tokenize=True)
            assert full[:len(prefix)]==prefix
            # Exclude the template suffix/EOS from canonical answer likelihood.
            answer_ids=tok.encode(r['answer'],add_special_tokens=False)
            assert full[len(prefix):len(prefix)+len(answer_ids)]==answer_ids
            ids=torch.tensor([prefix+answer_ids],device='cuda');_,states,_=model.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
            assert len(states)==4
            previous=None
            for i,h in enumerate(states):
                logits=model.lm_head(h[0,len(prefix)-1:len(prefix)+len(answer_ids)-1]).float()
                labels=torch.tensor(answer_ids,device='cuda');lp=torch.log_softmax(logits,dim=-1).gather(1,labels[:,None])[:,0]
                first=logits[0].clone();correct_logit=first[answer_ids[0]].item();first[answer_ids[0]]=-float('inf')
                score=dict(loop=i+1,answer_mean_logprob=lp.mean().item(),answer_sum_logprob=lp.sum().item(),first_token_margin=correct_logit-first.max().item())
                records[r['id']]['canonical_scores'].append(score)
                state=h[0,len(prefix)-1].float()
                if previous is not None:
                    records[r['id']]['prompt_state_change'].append(dict(from_loop=i,to_loop=i+1,relative_l2=((state-previous).norm()/previous.norm().clamp_min(1e-12)).item(),cosine=torch.nn.functional.cosine_similarity(state,previous,dim=0).item()))
                previous=state
            save_record(dict(event='canonical_probe',id=r['id'],scores=records[r['id']]['canonical_scores'],state_change=records[r['id']]['prompt_state_change']))
            if (r['id']+1)%48==0:print(json.dumps(dict(event='canonical_progress',done=r['id']+1)),flush=True)
    def summarize(rr):
        n=len(rr);result=dict(n=n,acc_by_loop=[sum(r['readouts'][i]['correct'] for r in rr)/n for i in range(4)],
            first_correct_histogram=dict(Counter(str(next((i+1 for i,x in enumerate(r['readouts']) if x['correct']),0)) for r in rr)),
            answer_change_rates=[sum(r['readouts'][i]['normalized']!=r['readouts'][i+1]['normalized'] for r in rr)/n for i in range(3)],
            early_correct_and_stays=sum(all(x['correct'] for x in r['readouts']) for r in rr),
            wrong_and_unchanged=sum(not any(x['correct'] for x in r['readouts']) and len({x['normalized'] for x in r['readouts']})==1 for r in rr),
            correct_then_lost=sum(any(x['correct'] for x in r['readouts'][:-1]) and not r['readouts'][-1]['correct'] for r in rr),
            mean_canonical_logprob=[float(np.mean([r['canonical_scores'][i]['answer_mean_logprob'] for r in rr])) for i in range(4)])
        for early in [1,2,3]:
            cohort=[r for r in rr if r['readouts'][early-1]['correct']]
            result[f'correct_at_{early}_retention']=dict(n=len(cohort),all_later_correct=sum(all(x['correct'] for x in r['readouts'][early:]) for r in cohort))
        return result
    rr=list(records.values());assert versions==[p._version for p in model.parameters()]
    result=dict(status='complete',overall=summarize(rr),by_family={f:summarize([r for r in rr if r['family']==f]) for f in sorted({r['family'] for r in rr})},by_family_level={f'{f}_L{l}':summarize([r for r in rr if r['family']==f and r['level']==l]) for f in sorted({r['family'] for r in rr}) for l in [1,2,3]},parameters_unchanged=True,seconds=time.time()-started,peak_gib=torch.cuda.max_memory_allocated()/2**30)
    (ROOT/'records.json').write_text(json.dumps(rr));(ROOT/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
