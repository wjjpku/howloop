"""Paired loop-3/loop-4 native readout of the frozen SFT baseline, no J."""
import hashlib,json,os,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from train_ouro_full import MODEL,task
from ouro_eval_panel import EVAL_SEEDS
from score_ouro_content import parse

ROOT=Path('/data/paperexperiment/ouro26_antonym_four_j_20260915/baseline_readout12')
CK=Path('/data/paperexperiment/ouro26_antonym_full_20260915/run/checkpoint.pt')

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='6'
    torch.set_num_threads(4);torch.manual_seed(20260915)
    ROOT.mkdir(parents=True,exist_ok=False)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tok.padding_side='left'
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa')
    ck=torch.load(CK,map_location='cpu',weights_only=False,mmap=True);assert ck['step']==500
    model.load_state_dict(ck['model'],strict=True);del ck
    model=model.to('cuda').eval().requires_grad_(False)
    assert model.config.total_ut_steps==4
    rows=[]
    for i,seed in enumerate(EVAL_SEEDS):
        for k in range(1,9):
            prompt,answer,_=task(seed,k,0)
            rows.append(dict(sequence=i,seed=seed,k=k,prompt=prompt,answer=answer))
    prompts=[tok.apply_chat_template([dict(role='user',content=r['prompt'])],tokenize=True,add_generation_prompt=True) for r in rows]
    manifest=dict(pid=os.getpid(),gpu=6,checkpoint=str(CK),checkpoint_step=500,loops_executed=4,readout_loops=[1,2],use_j=False,training=False,rows_per_readout=512,batch=4,max_new_tokens=64,template=0,seed=20260915,shared_with_J_training=True,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
    eos=model.generation_config.eos_token_id
    eos=[eos] if isinstance(eos,int) else eos or []
    results=[];start=time.time()
    for loop in [1,2]:
        for off in range(0,len(rows),4):
            if torch.cuda.mem_get_info()[0]<16*2**30:raise RuntimeError('Shared GPU reserve breached')
            b=tok.pad(dict(input_ids=prompts[off:off+4]),padding=True,return_tensors='pt').to('cuda')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                out=model.generate(**b,exit_at_step=loop-1,max_new_tokens=64,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
            for row,ids in zip(rows[off:off+4],out[:,b.input_ids.shape[1]:].tolist()):
                text=tok.decode(ids,skip_special_tokens=True).strip()
                parsed=parse(dict(kind='cancellation',generated=text))
                r=dict(**row,readout_loop=loop,generated=text,parsed=parsed,correct=parsed==row['answer'],eos_emitted=any(x in eos for x in ids),generated_ids=ids)
                results.append(r)
                with (ROOT/'outputs.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
            if off%32==0:print(json.dumps(dict(event='progress',readout=loop,done=off+4,seconds=time.time()-start)),flush=True)
    index={(r['sequence'],r['k'],r['readout_loop']):r for r in results}
    def summary(ks):
        pairs=[(index[i,k,1],index[i,k,2]) for i in range(64) for k in ks]
        n=len(pairs)
        return dict(n=n,loop1_correct=sum(a['correct'] for a,b in pairs),loop2_correct=sum(b['correct'] for a,b in pairs),answer_agreement=sum(a['parsed'] is not None and a['parsed']==b['parsed'] for a,b in pairs),loop2_repairs=sum(not a['correct'] and b['correct'] for a,b in pairs),loop2_breaks=sum(a['correct'] and not b['correct'] for a,b in pairs),loop1_unparsed=sum(a['parsed'] is None for a,b in pairs),loop2_unparsed=sum(b['parsed'] is None for a,b in pairs),loop1_without_eos=sum(not a['eos_emitted'] for a,b in pairs),loop2_without_eos=sum(not b['eos_emitted'] for a,b in pairs))
    report=dict(status='complete',per_k={str(k):summary([k]) for k in range(1,9)},k1_4=summary(range(1,5)),k5_8=summary(range(5,9)),seconds=time.time()-start)
    (ROOT/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()

