"""Native Ouro CoT boundary pilot; fixed graphs, final-answer scoring, no training."""
import json, os, random, re, time, hashlib, subprocess
from pathlib import Path

NAMES='Alice Bob Carol David Emma Frank Grace Henry Iris Jack'.split()
ROOT=Path('/data/paperexperiment/letter_walk_native_20260914/ouro26_cycle_boundary_v1')
MODEL='/data/paperexperiment/models/Ouro-2.6B-Thinking'
COARSE=[1,4,8,9,10,11,16,24,32,48,64,96,99]

def case(g,k):
    rng=random.Random(2026091400+g); cycle=NAMES.copy(); rng.shuffle(cycle)
    edges=dict(zip(cycle,cycle[1:]+cycle[:1])); order=NAMES.copy(); rng.shuffle(order)
    prompt='Whenever someone receives the letter, they pass it to their designated recipient. The rules never change.\n'
    prompt+='\n'.join(f'{x} passes the letter to {edges[x]}.' for x in order)
    prompt+=f'\nThe letter starts with {cycle[0]}. After exactly {k:02d} transfers, who has it? Answer directly with only the person\'s name.'
    return dict(graph=g,k=k,edges=edges,start=cycle[0],answer=cycle[k % len(cycle)],prompt=prompt)

def parse(text):
    # Only inspect final paragraph: mentioning the answer somewhere in CoT is insufficient.
    tail=text.strip().split('\n\n')[-1]
    boxed=re.findall(r'\\boxed\{([A-Za-z]+)\}',tail)
    if boxed and boxed[-1] in NAMES: return boxed[-1]
    names=re.findall(r'\b(?:'+'|'.join(NAMES)+r')\b',tail,flags=re.I)
    unique={n.lower() for n in names}
    if len(unique)==1:
        return next(n for n in NAMES if n.lower() in unique)
    return None

def dump(name,obj):
    (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2))

def main():
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    assert os.environ['CUDA_VISIBLE_DEVICES']=='6'
    ROOT.mkdir(exist_ok=False)
    manifest=dict(pid=os.getpid(),gpu=6,model=MODEL,native_steps=4,early_exit_threshold=1.0,training=False,
        graphs=4,nodes=10,cycle_length=10,coarse_k=COARSE,seed=2026091400,max_new_tokens=2048,truncation_retry=4096,
        score='unique name in final paragraph or final boxed name; truncated outputs never scored correct',
        refinement='first coarse k with fewer than 3/4 correct; test up to 7 intervening depths',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),status='waiting_gpu')
    dump('manifest.json',manifest)
    print('Waiting for GPU 6 to become empty; existing jobs will not be interrupted.',flush=True)
    deadline=time.time()+7200; stable=None
    while True:
        used=int(subprocess.check_output(['nvidia-smi','-i','6','--query-gpu=memory.used','--format=csv,noheader,nounits']).decode().strip())
        if used<100:
            if stable is None: stable=time.time()
            if time.time()-stable>=60: break
        else: stable=None
        if time.time()>deadline: raise RuntimeError('GPU wait exceeded two hours')
        time.sleep(30)
    torch.set_num_threads(4); torch.manual_seed(20260914)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    data=[case(g,k) for k in COARSE for g in range(4)]
    lengths={}
    for g in range(4):
        lengths[g]=sorted({len(tok.apply_chat_template([{'role':'user','content':case(g,k)['prompt']}],add_generation_prompt=True)) for k in range(1,100)})
    manifest['input_token_lengths_by_graph']=lengths
    dump('manifest.json',manifest)
    assert all(len(x)==1 for x in lengths.values()),'Input token length differs across k'
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16).to('cuda').eval().requires_grad_(False)
    assert model.config.total_ut_steps==4 and model.config.early_exit_threshold==1.0
    manifest['status']='running'; dump('manifest.json',manifest)
    print('Loaded model; fixed input lengths '+str(lengths),flush=True)
    results=[]
    def run(e):
        ids=tok.apply_chat_template([{'role':'user','content':e['prompt']}],add_generation_prompt=True,return_tensors='pt').to('cuda')
        for budget in [2048,4096]:
            torch.manual_seed(20260914); t=time.time()
            with torch.inference_mode():
                out=model.generate(ids,max_new_tokens=budget,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id or tok.eos_token_id)
            gen=out[0,ids.shape[1]:].tolist(); text=tok.decode(gen,skip_special_tokens=True)
            truncated=len(gen)==budget and gen[-1] not in ([model.generation_config.eos_token_id] if isinstance(model.generation_config.eos_token_id,int) else model.generation_config.eos_token_id or [])
            parsed=parse(text)
            row=dict(**e,budget=budget,input_tokens=ids.shape[1],generated_ids=gen,generated=text,raw=tok.decode(gen,skip_special_tokens=False),parsed=parsed,truncated=truncated,correct=not truncated and parsed==e['answer'],seconds=time.time()-t,peak_gib=torch.cuda.max_memory_allocated()/2**30)
            with (ROOT/'attempts.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            print(json.dumps({k:row[k] for k in ['graph','k','answer','parsed','correct','truncated','budget','seconds','peak_gib']}),flush=True)
            if torch.cuda.mem_get_info()[0]<16*2**30:raise RuntimeError('GPU reserve reached')
            if not truncated:break
        results.append(row)
        with (ROOT/'outputs.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        summary={str(k):dict(n=sum(r['k']==k for r in results),correct=sum(r['k']==k and r['correct'] for r in results),truncated=sum(r['k']==k and r['truncated'] for r in results),unparsed=sum(r['k']==k and r['parsed'] is None for r in results)) for k in sorted({r['k'] for r in results})}
        dump('summary.json',dict(status='partial',per_k=summary))
    for e in data:run(e)
    for i,k in enumerate(COARSE):
        if sum(r['correct'] for r in results if r['k']==k)<3:
            prev=COARSE[i-1] if i else 0
            for kk in list(range(prev+1,k))[:7]:
                for g in range(4):run(case(g,kk))
            break
    manifest['status']='complete'; dump('manifest.json',manifest)
    summary=json.loads((ROOT/'summary.json').read_text());summary['status']='complete';dump('summary.json',summary)
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
