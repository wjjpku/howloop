#!/usr/bin/env python3
from pathlib import Path
import json,subprocess,sys,os,argparse
import numpy as np
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=R/'outputs/parity_phase');a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);report=[]
for seed in range(3):
 src=R/f'experiments/parity_phase/seed{seed}';dest=out/f'seed{seed}'
 cmd=[sys.executable,str(R/'experiments/parity_phase/analyze_parity_ridge_slope.py'),'--metrics',str(src/'source/diagonal_band_metrics.csv'),'--out-dir',str(dest),'--minimum-length','41','--maximum-length','490','--period','4','--bootstrap-samples','5000','--bootstrap-block-length','20','--seed','2026092318']
 proc=subprocess.run(cmd,capture_output=True,text=True,env=dict(os.environ,MPLBACKEND='Agg'));(out/f'seed{seed}.log').write_text(proc.stdout+proc.stderr)
 if proc.returncode:raise RuntimeError(proc.stderr[-2000:])
 got=json.loads((dest/'parity_ridge_slope_analysis.json').read_text());ref=json.loads((src/'analysis/parity_ridge_slope_analysis.json').read_text())
 for variant in ['raw','J']:
  x=got['metrics']['exact_match'][variant];y=ref['metrics']['exact_match'][variant]
  assert np.isclose(x['phase_drift']['loops_per_100_tokens'],y['phase_drift']['loops_per_100_tokens'],atol=1e-10)
  assert np.allclose(x['bootstrap']['slope_ci95'],y['bootstrap']['slope_ci95'],atol=1e-10)
  report.append(dict(seed=seed,variant=variant,loops_per100=x['phase_drift']['loops_per_100_tokens'],ci95=(100*np.array(x['bootstrap']['slope_ci95'])).tolist(),mae=x['fit_quality']['mae_loops']))
(out/'verified_phase_table.json').write_text(json.dumps(report,indent=2));print('All 6 phase fits and bootstrap intervals match.')
