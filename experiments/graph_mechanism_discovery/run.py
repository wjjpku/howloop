from pathlib import Path
import json,os,subprocess,sys,time
R=Path(__file__).resolve().parent;m=json.loads((R/'MANIFEST.json').read_text())
cmd=[sys.executable,'-u',str(R/'code/evaluate_battery.py'),'--looplus',str(R/'code'),'--checkpoint',m['checkpoint'],'--checkpoint-sha256',m['checkpoint_sha256'],'--checkpoint-step',str(m['checkpoint_step']),'--controllers',m['controllers'],'--controller-sha256',m['controller_sha256'],'--datasets',str(R/'datasets.json'),'--family','N10_fresh_reserved_pairs','--backbone-name','A','--head','2','--panel','confirmation','--seeds','1','2','--out',str(R/'evaluation')]
p=subprocess.Popen(cmd,cwd=R/'code');s=dict(pid=p.pid,worker_pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],command=cmd,start=time.time(),state='running');(R/'RUN.json').write_text(json.dumps(s,indent=2));rc=p.wait();s.update(state='complete' if rc==0 else 'failed',exit_code=rc);(R/'RUN.json').write_text(json.dumps(s,indent=2));sys.exit(rc)
