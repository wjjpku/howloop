from pathlib import Path
import subprocess,sys
P=Path(__file__).resolve().parents[1];B=Path('/data/wujiaju/n10_migration_20260923')
for n,seed in [('A',6),('B',3),('C',4),('D',5),('E',7)]:
 for hop in [1,2]:
  for fit in [1,2]:
   sub=P/f'local/{n}/hop{hop}_seed{fit}'
   if (sub/'evaluation/summary.json').exists():continue
   assert (sub/'summary.json').exists()
   subprocess.run(list(map(str,[sys.executable,'-u',P/'code/train_dense.py','evaluate','--checkpoint',B/f'backbones/local_control_L6_seed{seed}/best.pt','--controller',sub/'best_controller.pt','--locked-test',B/'locks/confirmation.pt','--out-dir',sub/'evaluation','--test-seed',20260923,'--batch-size',250])),check=True)
print('All 20 locked single-step evaluations completed.')
