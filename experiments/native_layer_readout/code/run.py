import core as R
from core import torch,np,json,time,os
from pathlib import Path
P=Path(__file__).resolve().parents[1]
@torch.no_grad()
def main():
 torch.set_num_threads(4);gs=json.loads((P/'datasets.json').read_text())['confirmation'];out=P/'raw';out.mkdir(exist_ok=True,parents=True)
 for name,seed in [('B',3),('D',5),('E',7),('C',4),('A',6)]:
  tic=time.time();cp=R.B/f'backbones/local_control_L6_seed{seed}/best.pt';m,cfg,_=R.load_checkpoint(cp,torch.device('cuda'));m.eval().requires_grad_(False);versions=[p._version for p in m.parameters()];assert m.outer_norm is None and all(m.active_block_indices(t)==(0,1) for t in range(10));sha=R.sha(cp);assert sha==json.loads((R.D/f'graph/{name}_fit1.json').read_text())['checkpoint_sha256'];pred=[];probs=[];paths=[];maxerr=0
  R.save(out/f'{name}_running.json',dict(pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],checkpoint=sha))
  for off in range(0,len(gs),16):
   g=torch.tensor(gs[off:off+16],device='cuda').repeat_interleave(10,0);s=torch.arange(10,device='cuda').repeat(len(g)//10);tok,path,*_=R.fixed_depth_batch(cfg,len(g),torch.device('cuda'),path_positions=12,successors=g,start=s);path=torch.cat([s[:,None],path],1);paths.append(path.cpu().numpy());h=m.token_embed(tok)+m.pos_embed[None];states=[h];boundaries=['embedding']
   for t in range(10):
    direct=m.apply_loop(h,loop_index=t);y,tr=R.run(m,h,loop_indices=(t,));err=float((direct-tr.sites[-1].hidden_out).abs().max());maxerr=max(err,maxerr);assert torch.allclose(direct,tr.sites[-1].hidden_out,atol=2e-4,rtol=2e-4)
    for layer,site in enumerate(tr.sites):
     states.extend([site.residual_mid,site.hidden_out]);boundaries.extend([f'loop{t+1}_L{layer+1}_attn',f'loop{t+1}_L{layer+1}_mlp'])
    h=direct
   logits=torch.stack([m.unembed(m.ln_final(x[:,34]))[:,:10] for x in states]);pred.append(logits.argmax(-1).cpu().numpy());probs.append(logits.softmax(-1).cpu().numpy());assert torch.cuda.mem_get_info()[0]>16*2**30
  arr=dict(predictions=np.concatenate(pred,1),probabilities=np.concatenate(probs,1),paths=np.concatenate(paths),graphs=np.array(gs),boundaries=boundaries);old=np.load(Path('/data/wujiaju/state_recombination_20260926/confirmation')/f'{name}.npz');assert np.array_equal(arr['predictions'][::4][:10],old['native']);assert versions==[p._version for p in m.parameters()] and sha==R.sha(cp)
  np.savez_compressed(out/f'{name}.npz',**arr);meta=dict(status='complete',seed=seed,n=len(gs)*10,checkpoint_sha256=sha,data_sha256=R.sha(P/'datasets.json'),code_sha256=R.sha(__file__),max_state_error=maxerr,native_loop_readout_exact=True,weights_unchanged=True,seconds=time.time()-tic,peak_mib=torch.cuda.max_memory_allocated()/2**20);R.save(out/f'{name}.json',meta);print(json.dumps(meta),flush=True);del m;torch.cuda.empty_cache()
if __name__=='__main__':main()
