import json
from pathlib import Path
import torch
from reasoning_loop.paper2027_graph_g4_protocol import load_unique_lock, merged_forbidden_codes, sample_training_permutations, permutation_codes
from reasoning_loop.graph_path_loop import GraphPathConfig
from reasoning_loop.graph_path_depth_circuit import fixed_depth_batch
R=Path(__file__).resolve().parents[1];torch.set_num_threads(4)
locks=list((R/'locks').glob('*.pt'));forbidden=merged_forbidden_codes(locks,node_count=10)
for p in locks:
 lock=load_unique_lock(p);assert torch.equal(lock['successors'].sort(-1).values,torch.arange(10)[None].expand(len(lock['successors']),-1))
g=sample_training_permutations(batch_size=10000,node_count=10,forbidden_codes=forbidden,device=torch.device('cpu'))
assert not torch.isin(permutation_codes(g),forbidden).any()
cfg=GraphPathConfig(node_count=10,max_depth=8,max_loops=6)
graph=(torch.arange(10)+1).remainder(10)[None].repeat(10,1);starts=torch.arange(10)
tok,targets,*_=fixed_depth_batch(cfg,10,torch.device('cpu'),path_positions=16,successors=graph,start=starts)
assert cfg.seq_len==35 and tok.shape==(10,35)
assert torch.equal(targets,(starts[:,None]+torch.arange(1,17)[None])%10)
assert (targets[:,7]!=targets[:,8]).all() and (targets[:,8]!=targets[:,9]).all() and (targets[:,7]!=targets[:,9]).all()
assert int(permutation_codes(torch.arange(9,-1,-1)[None]))==9876543210
result={'status':'passed','unique_disjoint_heldout_graphs':len(forbidden),'training_sample_exclusion_check':10000,'sequence_length':35,'target_indexing_cycle10':True,'codes_exceed_int32_checked':True}
(R/'protocol_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
