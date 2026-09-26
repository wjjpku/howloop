#!/usr/bin/env python3
"""Explicit experiment commands for the isolated runtime (dry-run by default)."""
from pathlib import Path
import argparse,hashlib,json,os,shlex,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('stage',choices=['n10-backbone','n10-controller','n10-evaluate','n10-native','n10-export','graph-confirmation','composition','target-exchange','n8-backbone','n8-controller','graph-long','parity-backbone','parity-controller','parity-long','parity-diagnostics','ouro-letter-backbone','ouro-letter-controller','ouro-semantic','ouro-restore','ouro-final-controller','ouro-stepwise-controller','ouro-final-backbone','ouro-stepwise-backbone'])
p.add_argument('--work',type=Path,required=True);p.add_argument('--seed',type=int,default=6);p.add_argument('--loops',type=int,choices=[6,8],default=6);p.add_argument('--fit',type=int,choices=[1,2],default=1);p.add_argument('--hop',type=int,choices=[1,2],default=1);p.add_argument('--gpu',help='Physical GPU id, required for execution');p.add_argument('--execute',action='store_true')
a=p.parse_args();W=a.work.resolve()
if not (W/'runtime_manifest.json').exists():raise SystemExit('Run prepare_runtime.py first.')
N=W/'n10_migration_20260923';C=N/'code';G=W/'paper2027_confirmatory/graph_g4_disjoint_v3';P=W/'parity_input_once_20260811';py=sys.executable;cwd=C;env=dict(os.environ);cmd=[py,'-u'];stage=a.stage
if a.loops==6:
 if stage.startswith('n10') and a.seed not in [6,3,4,5,7]:raise SystemExit('Registered D8L6 seeds: 6,3,4,5,7')
 cp=N/f'backbones/local_control_L6_seed{a.seed}/best.pt'
 name=dict(zip([6,3,4,5,7],'ABCDE')).get(a.seed,'A');local=N/'local'/name
else:
 family=N if a.seed in [0,1,6,8] else N/'trajectory_seed_extension_20260924';cp=family/f'backbones/trajectory_L8_seed{a.seed}/best.pt';local=N/'trajectories'/f'L8_seed{a.seed}'
if stage=='n10-backbone':
 cmd+=['-m','reasoning_loop.paper2027_graph_g4_backbone','--out-dir',str(cp.parent),'--selection-lock',str(N/'locks/selection.pt'),'--final-test-lock',str(N/'locks/confirmation.pt'),'--extra-exclude-locks',*[str(N/'locks'/f'{x}.pt') for x in ['discovery','rings','smoke','donors']],'--seed',str(a.seed),'--loops',str(a.loops),'--depth-mode','uniform','--steps','20000','--eval-every','1000','--batch-size','512','--eval-batch-size','500','--amp']
elif stage in ['n10-controller','n10-evaluate']:
 sub=local/f'hop{a.hop}_seed{a.fit}';cmd+=['-m','reasoning_loop.paper2027_d8l6_s1','train' if stage=='n10-controller' else 'evaluate','--checkpoint',str(cp)]
 if stage=='n10-controller':cmd+=['--out-dir',str(sub),'--target-hop',str(a.hop),'--seed',str(a.fit),'--validation-seed',str(10000+a.fit),'--validation-lock',str(N/'locks/selection.pt'),'--train-exclude-locks',*[str(x) for x in sorted((N/'locks').glob('*.pt'))],'--validation-batch-size','500']
 else:cmd+=['--controller',str(sub/'best_controller.pt'),'--locked-test',str(N/'locks/confirmation.pt'),'--out-dir',str(sub/'evaluation'),'--test-seed','20260923','--batch-size','250']
elif stage=='n10-export':cmd += ['export_maps.py','--checkpoint',str(cp),'--runs',str(local)]
elif stage=='n10-native':cmd += ['evaluate_native.py','--checkpoint',str(cp),'--out',str(local/'native')]
elif stage=='graph-confirmation':
 if a.seed != 6:raise SystemExit('This submission registers seed 6 here; seeds 10 and 13 use experiments/selected_mechanism/code.')
 # Identity is read from the archived lock for exact original-weight replay.
 lock=json.loads((ROOT/'experiments/graph_mechanism/SELECTION_LOCK.json').read_text())['models']['A'];mf=lock['manifest'];name='A' if a.seed==6 else 'C';dest=W/'n10_selected_mechanism_20260924'
 cmd += [str(dest/'code/evaluate_battery.py'),'--looplus',str(dest/'code'),'--checkpoint',str(cp),'--checkpoint-sha256',mf['backbone_sha256'],'--checkpoint-step',str(mf['backbone_step']),'--controllers',str(local/'controllers.pt'),'--controller-sha256',mf['controller_sha256'],'--datasets',str(dest/'datasets.json'),'--family','N10_selected_fresh_confirmation','--backbone-name',name,'--head',str(lock['head']),'--panel','confirmation','--seeds','1','2','--out',str(dest/name/'evaluation')]
elif stage=='composition':
 cwd=W/'continuous_composition_20260925';cmd += ['run.py']
elif stage=='target-exchange':
 cwd=W/'paper_strengthening_20260925/code';cmd += ['graph_matrix.py']
elif stage.startswith('n8-'):
 if not 100<=a.seed<=111:raise SystemExit('N8 continuation seeds100..111 only.')
 cwd=G/('code' if stage=='n8-backbone' else 'analysis_code');cp=G/f'backbones/seed{a.seed}/graphpath_N8_D8_d256_B2_L8_seed{a.seed}/final.pt';sel=G/'locks/selection_permutations_512.pt';test=G/'locks/final_test_permutations_512.pt'
 if stage=='n8-backbone':cmd+=['-m','reasoning_loop.paper2027_graph_g4_backbone','--out-dir',str(cp.parent),'--selection-lock',str(sel),'--final-test-lock',str(test),'--seed',str(a.seed),'--steps','20000','--batch-size','512','--eval-batch-size','512','--eval-every','1000','--learning-rate','0.0003','--weight-decay','0.3','--warmup-steps','500','--grad-clip','1.0','--amp','--device','cuda']
 else:cmd+=['-m','reasoning_loop.paper2027_graph_g3_controller','train','--checkpoint',str(cp),'--out-dir',str(G/f'g4_controllers/seed{a.seed}/rank48_seed{a.fit}'),'--seed',str(2026095000+10*a.seed+a.fit),'--validation-seed',str(2026097000+10*a.seed+a.fit),'--validation-batches','8','--validation-batch-size','256','--validation-lock',str(sel),'--train-exclude-locks',str(sel),str(test),'--protocol-id','paper2027.graph.g4.disjoint_controller.v1']
elif stage in ['graph-long','parity-long']:cmd += [str(W/'fig6_retest_20260924'/('graph.py' if stage=='graph-long' else 'parity.py')),'--out',str(W/'recomputed'/stage)]+(['--shards','1'] if stage=='parity-long' else [])
elif stage.startswith('parity-'):
 if a.seed not in [0,1,2]:raise SystemExit('Parity seeds0,1,2 only.')
 env['PARITY_SEED']=str(a.seed);env['PYTHONPATH']=str(P/'evaluation_code');cwd=P/'evaluation_code'
 if stage=='parity-backbone':cmd=['bash',str(P/'code/run_backbone_multiseed.sh')]
 elif stage=='parity-controller':cmd=['bash',str(P/'evaluation_code/scripts/run_parity_input_once_controller_pair.sh')]
 else:cmd=['bash',str(P/'evaluation_code/scripts/run_parity_input_once_paper_figure_recheck.sh')]
elif stage.startswith('ouro-letter'):
 cwd=W/'letter_walk_native_20260914/code'
 if stage=='ouro-letter-backbone':cmd += ['train_full.py','--root',str(W/'ouro26_letter_full_20260915'),'--mode','train','--steps','500','--stop-after','200']
 else:cmd+=['train_affine_pair.py','--loops','4','--steps','500']
elif stage in ['ouro-semantic','ouro-restore']:
 cwd=W/'ouro_mechanism_20260923';d=cwd/'parallel_section';cfg=d/'config.json'
 if stage=='ouro-semantic':cmd += ['parallel_semantics.py','--pairs',str(d/'parallel_pairs64_with_cf.json'),'--config',str(cfg),'--out',str(W/'recomputed/ouro_semantic')]
 else:cmd += ['parallel_restore.py','--pairs',str(cwd/'restore_confirmation_with_cf.json'),'--out',str(W/'recomputed/ouro_restore')]
elif stage in ['ouro-final-backbone','ouro-stepwise-backbone']:
 if stage=='ouro-final-backbone':
  cwd=W/'ouro26_antonym_full_20260915/code';cmd += ['train_ouro_distributed.py','--root',str(W/'ouro26_antonym_full_20260915'),'--mode','train','--steps','500']
 else:
  cwd=W/'ouro26_antonym_four_j_20260915/code';cmd += ['train_ouro_stepwise_distributed.py','--pair-only','--steps','500']
elif stage in ['ouro-final-controller','ouro-stepwise-controller']:
 cwd=W/'ouro26_antonym_four_j_20260915/code';cmd += ['train_ouro_finalonly_pair_lora128.py' if stage=='ouro-final-controller' else 'train_ouro_stepwise_pair_lora128.py','--steps','500']
if not cwd.exists():raise SystemExit(f'Missing runtime code directory: {cwd}')
print('Working directory:',cwd);print('Command:',shlex.join(cmd))
if a.execute:
 if a.gpu is None:raise SystemExit('Specify --gpu after checking available GPU memory.')
 env.update(CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
 logs=W/'logs';logs.mkdir(exist_ok=True);tag=f'{stage}_seed{a.seed}_fit{a.fit}_hop{a.hop}'
 (logs/f'{tag}.command.json').write_text(json.dumps({'cwd':str(cwd),'command':cmd,'gpu':a.gpu},indent=2))
 with (logs/f'{tag}.log').open('a') as f:subprocess.run(cmd,cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
