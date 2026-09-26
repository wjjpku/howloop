#!/usr/bin/env python3
"""CPU-only reproduction of the 2026-09-26 ICLR submission figures."""
from pathlib import Path
import argparse,os,shutil,subprocess,sys,json,hashlib,time
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--only',nargs='+',choices=['fig2','fig3_submission','fig4_submission','fig5','fig6','fig7_submission'])
p.add_argument('--output',type=Path,default=ROOT/'outputs')
a=p.parse_args();out=a.output.resolve();(out/'figures').mkdir(parents=True,exist_ok=True)
env=dict(os.environ,PAPEREXPERIMENT_OUTPUT=str(out),MPLBACKEND='Agg',PYTHONHASHSEED='0',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
shutil.copy2(ROOT/'paper/figures/fig01_overview.pdf',out/'figures/fig01_overview.pdf')
status=[]
for name in a.only or ['fig2','fig3_submission','fig4_submission','fig5','fig6','fig7_submission']:
 print('Reproducing',name,flush=True);t=time.time()
 proc=subprocess.run([sys.executable,str(ROOT/'plots'/f'{name}.py')],env=env,cwd=ROOT,capture_output=True,text=True)
 (out/f'{name}.log').write_text(proc.stdout+proc.stderr)
 status.append({'plot':name,'exit_code':proc.returncode,'seconds':round(time.time()-t,2)})
 if proc.returncode:print(proc.stdout+proc.stderr);raise SystemExit(proc.returncode)
for old,new in [('fig2a_readout.pdf','fig02a_readout.pdf'),('fig2b_readout.pdf','fig02b_readout.pdf'),('fig2c_readout.pdf','fig02c_readout.pdf'),('fig2d_readout.pdf','fig02d_readout.pdf'),('fig5a_parity.pdf','fig05a_parity.pdf'),('fig5b_graph.pdf','fig05b_graph.pdf'),('fig5_legend.pdf','fig05_legend.pdf'),('fig6_parity_timing.pdf','fig06_parity_timing.pdf')]:
 src=out/'figures'/old
 if src.exists():shutil.copy2(src,out/'figures'/new)
expected=[Path(x['paper_asset']).name for x in json.loads((ROOT/'provenance/figures.json').read_text())]
if not a.only:
 assert all((out/'figures'/name).is_file() for name in expected)
report={'paper_assets':expected,'figures':status,'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((out/'figures').glob('*.pdf'))},'conceptual_figure':'Fig. 1 is preserved author-supplied artwork, not experimental output.'}
(out/'figure_reproduction.json').write_text(json.dumps(report,indent=2))
if not a.only:
 subprocess.run([sys.executable,str(ROOT/'scripts/audit_submission.py')],env=env,cwd=ROOT,check=True)
print('Complete:',len(expected),'manuscript assets;',len(report['files']),'PDFs including intermediates in',out/'figures')
