"""Frozen stepwise backbone: requested k4..8, always four physical loops, no J."""
import json,os,time,hashlib
from pathlib import Path
from collections import Counter
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from train_ouro_full import MODEL
from ouro_stepwise_task import example,parse
from ouro_eval_panel import EVAL_SEEDS

BASE=Path('/data/paperexperiment/ouro26_stepwise_control_20260915')
ROOT=BASE/'no_j_L4_k58'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))==1
    assert json.loads((BASE/'backbone_evaluation.json').read_text())['gate_passed']
    ROOT.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(20260915)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tok.padding_side='left'
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True,
        torch_dtype=torch.float32,attn_implementation='sdpa')
    digest=sha(BASE/'checkpoint.pt')
    ck=torch.load(BASE/'checkpoint.pt',map_location='cpu',weights_only=False,mmap=True)
    assert ck['step']==500;model.load_state_dict(ck['model'],strict=True);del ck
    model=model.to('cuda').eval().requires_grad_(False)
    assert model.model.total_ut_steps==4
    versions=[p._version for p in model.parameters()]
    manifest=dict(pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],checkpoint_sha256=digest,
        checkpoint_step=500,loops=4,J=False,requested_k=[4,5,6,7,8],n_per_k=64,
        seeds=EVAL_SEEDS,template=0,batch=16,max_new_tokens=160,greedy=True,
        script_sha256=sha(Path(__file__)),training=False,
        caveat='Backbone training requested-k text was always 4; changing k tests a previously constant prompt field, not established query control.')
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
    results=[];started=time.time()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for k in range(4,9):
            for off in range(0,64,16):
                if torch.cuda.mem_get_info()[0]<16*2**30:raise RuntimeError('GPU reserve breached')
                seeds=EVAL_SEEDS[off:off+16];rr=[example(seed,k,k,0) for seed in seeds]
                prompts=[tok.apply_chat_template([dict(role='user',content=r[0])],tokenize=True,add_generation_prompt=True) for r in rr]
                b=tok.pad(dict(input_ids=prompts),padding=True,return_tensors='pt').to('cuda')
                out=model.generate(**b,exit_at_step=3,max_new_tokens=160,do_sample=False,use_cache=True,pad_token_id=tok.pad_token_id)
                for i,(seed,r,ids) in enumerate(zip(seeds,rr,out[:,b.input_ids.shape[1]:])):
                    text=tok.decode(ids,skip_special_tokens=True).strip();parsed=parse(text)
                    references={s:example(seed,k,s,0)[3] for s in range(1,13)}
                    rec=dict(k=k,i=off+i,seed=seed,prompt=r[0],expected=r[3],generated=text,parsed=parsed,
                        field_correct=[parsed is not None and parsed[j]==r[3][j] for j in range(3)],
                        exact=parsed==r[3],matching_steps=[s for s,f in references.items() if parsed==f],
                        pair_matching_steps=[s for s,f in references.items() if parsed is not None and parsed[0]==f[0]],
                        step4_reference=references[4],prompt_tokens=len(prompts[i]),generated_tokens=len(ids))
                    results.append(rec)
                    with (ROOT/'outputs.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
                print(json.dumps(dict(event='progress',k=k,done=off+16,seconds=time.time()-started)),flush=True)
    summary={}
    for k in range(4,9):
        rr=[r for r in results if r['k']==k]
        baseline={r['seed']:r for r in results if r['k']==4}
        summary[str(k)]=dict(n=64,exact=sum(r['exact'] for r in rr),fields=[sum(r['field_correct'][j] for r in rr) for j in range(3)],
            parsed=sum(r['parsed'] is not None for r in rr),
            whole_record_step_histogram=dict(Counter(str(r['matching_steps'][0]) if r['matching_steps'] else 'no_exact_step' for r in rr)),
            pair_step_histogram=dict(Counter(str(r['pair_matching_steps'][0]) if r['pair_matching_steps'] else 'no_pair_step' for r in rr)),
            identical_text_to_k4=sum(r['generated']==baseline[r['seed']]['generated'] for r in rr))
    assert versions==[p._version for p in model.parameters()]
    report=dict(status='complete',results=summary,seconds=time.time()-started,parameters_unchanged=True,
        peak_gib=torch.cuda.max_memory_allocated()/2**30)
    (ROOT/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
