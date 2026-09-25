"""Frozen, native-depth letter-transfer pilot. Preserve prompts and raw outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time

def examples():
    names=['Alice','Bob','Carol','David','Emma','Frank','Grace','Henry','Iris','Jack']
    out=[]
    for gid in range(2):
        rng=random.Random(2026091400+gid);cycle=names.copy();rng.shuffle(cycle)
        edges=dict(zip(cycle,cycle[1:]+cycle[:1]));order=names.copy();rng.shuffle(order)
        start=cycle[0]
        for k in range(1,9):
            answer=start
            for _ in range(k):answer=edges[answer]
            prompt='Whenever someone receives the letter, they pass it to their designated recipient. The rules never change.\n'
            prompt+='\n'.join(f'{x} passes the letter to {edges[x]}.' for x in order)
            prompt+=f'\nThe letter starts with {start}. After exactly {k:02d} transfers, who has it? Answer directly with only the person\'s name.'
            out.append(dict(id=f'g{gid}-k{k}',graph=gid,edges=edges,start=start,k=k,answer=answer,prompt=prompt,language='en'))
        k=3;answer=start
        for _ in range(k):answer=edges[answer]
        prompt='每个人收到信后，都会把信交给指定的人，规则始终不变。\n'
        prompt+='\n'.join(f'{x}把信交给{edges[x]}。' for x in order)
        prompt+=f'\n信最初在{start}手中。经过03次转交后，信在谁手中？直接回答姓名。'
        out.append(dict(id=f'g{gid}-k3-zh',graph=gid,edges=edges,start=start,k=k,answer=answer,prompt=prompt,language='zh'))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=18);a=p.parse_args()
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==1
    torch.set_num_threads(4);torch.manual_seed(20260914)
    a.output.mkdir(parents=True,exist_ok=False)
    modelpath=Path(a.model);huginn='huginn' in modelpath.name.lower()
    assert list(modelpath.glob('*.safetensors')), 'Missing weights'
    if huginn:assert (modelpath/'weights_verified.json').exists()
    tokenizer=AutoTokenizer.from_pretrained(modelpath,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(modelpath,torch_dtype=torch.bfloat16,trust_remote_code=True,local_files_only=True).to('cuda').eval().requires_grad_(False)
    data=examples()[:a.limit]
    manifest=dict(model=str(modelpath),pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],torch=torch.__version__,
        native_steps=32 if huginn else model.config.total_ut_steps,extra_wrapping=False,training=False,
        do_sample=False,max_new_tokens=64,use_cache=True,ouro_enable_thinking=False,
        seed=20260914,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    results=[]
    for e in data:
        torch.manual_seed(20260914)
        prompt=tokenizer.apply_chat_template([{'role':'user','content':e['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tokenizer(prompt,return_tensors='pt',add_special_tokens=False).input_ids.to('cuda')
        start=time.time()
        extra={'num_steps':32} if huginn else {}
        with torch.inference_mode():
            generated=model.generate(ids,max_new_tokens=64,do_sample=False,use_cache=True,pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,**extra)
        tokens=generated[0,ids.shape[1]:].tolist();raw=tokenizer.decode(tokens,skip_special_tokens=False);text=tokenizer.decode(tokens,skip_special_tokens=True)
        normalized=text.strip().strip('.!。！').strip()
        row=dict(**e,rendered_prompt=prompt,input_tokens=ids.shape[1],generated=text,raw=raw,generated_ids=tokens,
            exact=normalized.casefold()==e['answer'].casefold(),truncated=len(tokens)==64,seconds=time.time()-start,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        results.append(row)
        with (a.output/'outputs.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(json.dumps({k:row[k] for k in ['id','k','answer','generated','exact','truncated','seconds','peak_allocated_gib']},ensure_ascii=False),flush=True)
        if torch.cuda.mem_get_info()[0]<16*2**30:raise RuntimeError('GPU reserve reached')
    summary=dict(total=len(results),correct=sum(x['exact'] for x in results),
        per_k={str(k):dict(correct=sum(x['exact'] for x in results if x['k']==k and x['language']=='en'),total=sum(x['k']==k and x['language']=='en' for x in results)) for k in range(1,9)},
        prompt_lengths={str(g):sorted(set(x['input_tokens'] for x in results if x['graph']==g and x['language']=='en')) for g in range(2)})
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
