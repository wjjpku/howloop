from pathlib import Path
import torch,json,random,hashlib
P=Path(__file__).resolve().parents[1];B=Path('/data/paperexperiment/n10_migration_20260923');torch.set_num_threads(2)
pool=sorted(set(map(tuple,torch.load(B/'locks/donors.pt',weights_only=False)['successors'].tolist())))
random.Random(2026092608).shuffle(pool)
def save(p,x):p.write_text(json.dumps(x,indent=2))
assert len(pool)>=1602
save(P/'discovery_datasets.json',{'discovery':pool[:32],'corrupted':pool[32:64]})
save(P/'confirmation_datasets.json',{'confirmation':pool[64:576],'corrupted':pool[576:1088]})
save(P/'graph_data.json',{'confirmation':pool[1088:1600],'smoke':pool[1600:1602]})
save(P/'MANIFEST.json',{'seeds':list(range(8,20)),'protocol':'latest submission Section 5, rank48 diagonal+lowrank affine; 8000 updates; two fits per target; frozen h6 + one F','head_selection':'independent 32 graph pairs; all 4 L2 heads retained on 512 pair confirmation; no backbone outcome filtering','datasets':'fixed all-backbone cohorts; 1602 distinct graphs drawn only from donors lock excluded from every backbone and map training','donors_sha256':hashlib.sha256((B/'locks/donors.pt').read_bytes()).hexdigest(),'additional':'512 graphs for whole-layer bidirectional target exchange; same-run patch validation; appendix query-corruption/output-restoration controls','gpus':[1,3],'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'code').glob('*.py')}})
print('prepared and locked')
