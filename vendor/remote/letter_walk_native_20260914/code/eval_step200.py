"""Frozen step-200 checkpoint: matched unseen graphs, k=4 control and k=5..8 OOD."""
import hashlib,json,os,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from train_full import MODEL,task

ROOT=Path('/data/paperexperiment/ouro26_letter_full_20260915')
OUT=ROOT/'eval_step200_k5_k8_v1'

def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='7'
    OUT.mkdir(exist_ok=False);torch.set_num_threads(4);torch.manual_seed(20260915)
    checkpoint=ROOT/'run/checkpoint.pt'
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    assert ck['step']==200,ck['step']
    digest=hashlib.sha256()
    with checkpoint.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):digest.update(b)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa')
    model.load_state_dict(ck['model'],strict=True);del ck
    model=model.to('cuda').eval().requires_grad_(False)
    assert model.config.total_ut_steps==4
    manifest=dict(pid=os.getpid(),gpu=7,checkpoint=str(checkpoint),step=200,checkpoint_sha256=digest.hexdigest(),
        loops=4,readout_loop=4,training=False,gate_used=False,ks=[4,5,6,7,8],graphs_per_k=16,template=0,
        graph_seeds=[9100000+i for i in range(16)],precision='FP32 weights, BF16 autocast',
        max_new_tokens=16,do_sample=False,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)
    rows=[]
    for k in range(4,9):
        for i in range(16):
            prompt,answer,graph=task(9100000+i,k,0)
            ids=tok.apply_chat_template([dict(role='user',content=prompt)],add_generation_prompt=True,return_tensors='pt').to('cuda')
            start=time.time()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                out=model.generate(ids,max_new_tokens=16,do_sample=False,use_cache=True,exit_at_step=3,pad_token_id=tok.pad_token_id or tok.eos_token_id)
            tokens=out[0,ids.shape[1]:].tolist();text=tok.decode(tokens,skip_special_tokens=True).strip()
            row=dict(k=k,graph=i,prompt=prompt,edges=graph,expected=answer,generated=text,raw=tok.decode(tokens,skip_special_tokens=False),
                generated_ids=tokens,input_tokens=ids.shape[1],correct=text.strip('.!\n ').lower()==answer.lower(),length_limit_reached=len(tokens)==16,seconds=time.time()-start)
            rows.append(row)
            with (OUT/'outputs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps({s:row[s] for s in ['k','graph','expected','generated','correct','length_limit_reached']}),flush=True)
            if torch.cuda.mem_get_info()[0]<16*2**30:raise RuntimeError('GPU reserve reached')
        summary={str(kk):dict(correct=sum(r['correct'] for r in rows if r['k']==kk),total=sum(r['k']==kk for r in rows),length_limit_reached=sum(r['length_limit_reached'] for r in rows if r['k']==kk)) for kk in range(4,k+1)}
        (OUT/'summary.json').write_text(json.dumps(dict(status='complete' if k==8 else 'partial',per_k=summary),indent=2))
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
