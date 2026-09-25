"""Original Ouro-2.6B base, native four loops, exact historical antonym bank."""
import argparse,hashlib,json,os,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from task import bank,score,summarize
from train_ouro_full import task as training_task
from ouro_eval_panel import EVAL_SEEDS

MODEL='/data/wujiaju/models/Ouro-2.6B'
ROOT=Path('/data/wujiaju/ouro26_antonym_four_j_20260915/baseline_depth_unique')

def main():
    global ROOT
    parser=argparse.ArgumentParser();parser.add_argument('--max-new-tokens',type=int,default=12);args=parser.parse_args();budget=args.max_new_tokens
    assert budget==64
    assert os.environ['CUDA_VISIBLE_DEVICES']=='5'
    torch.set_num_threads(4);torch.manual_seed(20260915)
    ROOT.mkdir(parents=True,exist_ok=False)
    rows=[]
    for i in range(64):
        for k in range(5,9):
            prompt,answer,_=training_task(EVAL_SEEDS[i],k,0)
            rows.append(dict(id=f'validation-{i}-{k}',kind='cancellation',sequence_id=i,k=k,template=0,prompt=prompt,answer=answer))
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);tok.padding_side='left'
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    prompts=[tok.apply_chat_template([dict(role='user',content=r['prompt'])],tokenize=True,add_generation_prompt=True) for r in rows]
    lengths={str(i):sorted({len(ids) for r,ids in zip(rows,prompts) if r.get('sequence_id')==i}) for i in range(64)}
    assert all(len(v)==1 for v in lengths.values())
    model=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa').to('cuda').eval().requires_grad_(False)
    ckpath=Path('/data/wujiaju/ouro26_antonym_full_20260915/run/checkpoint.pt')
    ck=torch.load(ckpath,map_location='cpu',weights_only=False,mmap=True)
    assert ck['step']==500
    model.load_state_dict(ck['model'],strict=True);del ck
    assert model.config.total_ut_steps==4
    manifest=dict(backbone_checkpoint=str(ckpath),backbone_step=500,pid=os.getpid(),gpu=5,model=MODEL,loops=4,readout_loop=4,use_j=False,training=False,trainable_parameters=0,
        system_prompt='You are a helpful assistant.',prompt_mode='historical zero-shot',seed=20260915,max_new_tokens=budget,batch_size=4,
        parameter_dtype='FP32',compute_dtype='BF16',rows=len(rows),input_lengths=lengths,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),task_sha256=hashlib.sha256(Path(__file__).with_name('task.py').read_bytes()).hexdigest())
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2));(ROOT/'questions.json').write_text(json.dumps(rows,indent=2));print(json.dumps(manifest),flush=True)
    results=[];start=time.time();eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos or []
    for offset in range(0,len(rows),4):
        batch=tok.pad({'input_ids':prompts[offset:offset+4]},padding=True,return_tensors='pt').to('cuda')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            out=model.generate(**batch,max_new_tokens=budget,do_sample=False,use_cache=True,exit_at_step=3,pad_token_id=tok.pad_token_id)
        for row,tokens in zip(rows[offset:offset+4],out[:,batch.input_ids.shape[1]:].tolist()):
            text=tok.decode(tokens,skip_special_tokens=True);ended=any(x in eos for x in tokens)
            result=dict(**row,generated=text,generated_token_ids=tokens,eos_emitted=ended,length_limit_reached=not ended and len(tokens)==budget,**score(row,text))
            results.append(result)
            with (ROOT/'results.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
        summary=dict(status='complete' if len(results)==len(rows) else 'partial',metrics=summarize(results),length_limit_reached=sum(r['length_limit_reached'] for r in results),seconds=time.time()-start)
        (ROOT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(dict(done=len(results),**summary)),flush=True)
        if torch.cuda.mem_get_info()[0]<16*2**30:raise RuntimeError('GPU reserve breached')
    print('COMPLETE',flush=True)

if __name__=='__main__':main()

