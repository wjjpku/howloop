"""Fixed fresh validation panel, deduplicated modulo antonym-pair relabeling."""
from train_ouro_full import task

def make_seeds():
    seen={task(9100000+i,1,0)[2] for i in range(16)}
    seeds=[]
    for seed in range(9101000,9200000):
        sig=task(seed,5,0)[2]
        if sig in seen:continue
        seen.add(sig);seeds.append(seed)
        if len(seeds)==64:return seeds
    raise RuntimeError('Could not construct 64 unique held-out topologies')

EVAL_SEEDS=make_seeds()
