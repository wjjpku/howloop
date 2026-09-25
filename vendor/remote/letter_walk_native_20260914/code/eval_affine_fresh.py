"""Locked fresh-graph evaluation of completed affine fits; no training."""
import argparse,collections,hashlib,json,os,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from train_full import MODEL,task
from train_affine_pair_distributed_v5 import Affine,BASE,CK_SHA

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def fresh_graphs():
    oldheld={task(9100000+i,1,0)[2] for i in range(16)}
    forbidden=set(oldheld)
    for steps,batch,offset in [(200,8,100000000),(500,16,300000000)]:
        for step in range(1,steps+1):
            for j in range(batch):
                seed=offset+step*100+j;graph=task(seed,1,0)[2]
                while graph in oldheld:seed+=100000000;graph=task(seed,1,0)[2]
                forbidden.add(graph)
    seeds=[];seen=set();candidate=2026091500
    while len(seeds)<64:
        graph=task(candidate,1,0)[2]
        if graph not in forbidden and graph not in seen:seeds.append(candidate);seen.add(graph)
        candidate+=1
    assert len(seen)==64 and not(seen&forbidden)
    return seeds,forbidden

def main():
    p=argparse.ArgumentParser();p.add_argument('--loops',type=int,choices=[4,8],required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    seeds,forbidden=fresh_graphs()
    cases=[]
    for template in [0,5]:
        for k in range(1,9):
            for g,seed in enumerate(seeds):cases.append((True,template,k,g,seed,'fresh_graphs_k1_k8'))
    for k in range(9,17):
        for g,seed in enumerate(seeds):cases.append((True,0,k,g,seed,'untrained_k9_k16_cycle10'))
    for k in range(1,9):
        for g,seed in enumerate(seeds[:16]):cases.append((False,0,k,g,seed,'no_j_fresh_control'))
    dataset_sha=hashlib.sha256(json.dumps(cases).encode()).hexdigest()
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    lengths={}
    for t in [0,5]:
        lengths[t]=sorted({len(tok.apply_chat_template([dict(role='user',content=task(seeds[0],k,t)[0])],add_generation_prompt=True)) for k in range(1,17)})
    assert all(len(v)==1 for v in lengths.values()),lengths
    print(json.dumps(dict(event='dataset_validated',graphs=64,forbidden_graphs=len(forbidden),cases=len(cases),dataset_sha256=dataset_sha,lengths=lengths)),flush=True)
    if a.prepare_only:return
    assert os.environ['CUDA_VISIBLE_DEVICES']==('5' if a.loops==4 else '7')
    torch.set_num_threads(4);torch.manual_seed(20260915)
    root=Path('/data/wujiaju/ouro26_affine_fresh_eval_20260915')/f'L{a.loops}';root.mkdir(parents=True,exist_ok=False)
    backbone=BASE/'run/checkpoint.pt';assert sha(backbone)==CK_SHA
    ck=torch.load(backbone,map_location='cpu',weights_only=False,mmap=True);assert ck['step']==200
    model=AutoModelForCausalLM.from_pretrained(MODEL,trust_remote_code=True,local_files_only=True,torch_dtype=torch.float32,attn_implementation='sdpa')
    model.load_state_dict(ck['model'],strict=True);del ck
    model=model.to('cuda').eval().requires_grad_(False);model.config.total_ut_steps=a.loops;model.model.total_ut_steps=a.loops
    jpath=Path('/data/wujiaju/ouro26_affine_pair_20260915')/f'L{a.loops}'/'checkpoint.pt'
    jc=torch.load(jpath,map_location='cpu',weights_only=False);assert jc['step']==500 and jc['loops']==a.loops and jc['backbone_sha256']==CK_SHA
    affine=Affine(model.config.hidden_size);affine.load_state_dict(jc['affine']);del jc
    affine=affine.to('cuda').eval().requires_grad_(False)
    enabled=True
    def hook(module,args,kwargs):
        if enabled and kwargs.get('current_ut',0)>0:return (affine(args[0]),)+args[1:],kwargs
        return args,kwargs
    model.model.layers[0].register_forward_pre_hook(hook,with_kwargs=True)
    manifest=dict(pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],loops=a.loops,backbone_sha256=CK_SHA,j_checkpoint=str(jpath),j_sha256=sha(jpath),j_step=500,
        training=False,graphs=64,excluded_graphs=len(forbidden),graph_seeds=seeds,cases=len(cases),dataset_sha256=dataset_sha,
        no_j_controls=128,primary_score='normalized whole answer exact match',max_new_tokens=16,do_sample=False,
        interpretation='k9..16 untrained query values, cycle length10, not proof of sequential k-step execution',script_sha256=sha(Path(__file__)))
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(event='loaded',**manifest)),flush=True)
    rows=[]
    def summarize(complete=False):
        grouped={}
        for row in rows:
            key=f'{row["split"]}/t{row["template"]}/k{row["k"]}'
            group=grouped.setdefault(key,dict(total=0,correct=0,length_limit_reached=0,output_phase_counts={}))
            group['total']+=1;group['correct']+=row['correct'];group['length_limit_reached']+=row['length_limit_reached']
            phase=str(row['output_phase']);group['output_phase_counts'][phase]=group['output_phase_counts'].get(phase,0)+1
        all_k={}
        for template in [0,5]:
            completed=[g for g in range(64) if sum(r['split']=='fresh_graphs_k1_k8' and r['template']==template and r['graph']==g for r in rows)==8]
            all_k[str(template)]=dict(complete_graphs=len(completed),all_8_correct=sum(all(r['correct'] for r in rows if r['split']=='fresh_graphs_k1_k8' and r['template']==template and r['graph']==g) for g in completed))
        (root/'summary.json').write_text(json.dumps(dict(status='complete' if complete else 'partial',by_k=grouped,all_k_condition=all_k),indent=2))
    for index,(use_j,template,k,g,seed,split) in enumerate(cases):
        enabled=use_j;prompt,answer,graph=task(seed,k,template)
        ids=tok.apply_chat_template([dict(role='user',content=prompt)],add_generation_prompt=True,return_tensors='pt').to('cuda');start=time.time()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            out=model.generate(ids,max_new_tokens=16,do_sample=False,use_cache=True,exit_at_step=a.loops-1,pad_token_id=tok.pad_token_id or tok.eos_token_id)
        tokens=out[0,ids.shape[1]:].tolist();text=tok.decode(tokens,skip_special_tokens=True).strip();normalized=text.strip('.!\n ').lower()
        phase=next((j for j in range(10) if task(seed,j,template)[1].lower()==normalized),None)
        row=dict(split=split,use_j=use_j,template=template,k=k,graph=g,graph_seed=seed,prompt=prompt,edges=graph,expected=answer,generated=text,generated_ids=tokens,
            correct=normalized==answer.lower(),output_phase=phase,input_tokens=ids.shape[1],length_limit_reached=len(tokens)==16,seconds=time.time()-start)
        rows.append(row)
        with (root/'outputs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if (index+1)%16==0:
            summarize();print(json.dumps(dict(event='progress',done=index+1,total=len(cases),split=split,k=k,template=template,last16_correct=sum(x['correct'] for x in rows[-16:]),peak_gib=torch.cuda.max_memory_allocated()/2**30)),flush=True)
        if torch.cuda.mem_get_info()[0]<16*2**30:raise RuntimeError('GPU reserve breached')
    summarize(True);print('COMPLETE',flush=True)

if __name__=='__main__':main()
