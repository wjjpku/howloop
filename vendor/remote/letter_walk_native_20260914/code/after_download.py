"""Bounded download-to-evaluation pipeline; never launch onto an occupied GPU."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT=Path('/data/wujiaju/letter_walk_native_20260914')
MODEL=Path('/data/wujiaju/models/huginn-0125')
PY='/data/wujiaju/.venvs/loopreasoner/bin/python'

def emit(event,**kw):
    row=dict(event=event,time=time.time(),pid=os.getpid(),**kw)
    (ROOT/'pipeline_status.json').write_text(json.dumps(row,indent=2))
    print(json.dumps(row),flush=True)

def main():
    start=time.time();emit('waiting_for_verified_weights',download_pid=766448)
    while not (MODEL/'weights_verified.json').exists():
        if time.time()-start>3600:raise RuntimeError('Download wait exceeded 1 hour')
        os.kill(766448,0)
        time.sleep(30)
    # Two fresh empty-card checks. Existing workloads remain untouched.
    for _ in range(2):
        used=int(subprocess.check_output(['nvidia-smi','-i','5','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if used>100:raise RuntimeError('GPU5 occupied; no automatic co-location')
        time.sleep(60 if _==0 else 0)
    if shutil.disk_usage(ROOT).free<2**30:raise RuntimeError('Insufficient output reserve')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='5',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME='/data/wujiaju/cache/huggingface',OMP_NUM_THREADS='4')
    cmd=[PY,'-u',str(ROOT/'code/probe.py'),'--model',str(MODEL),'--output',str(ROOT/'huginn_results_v1')]
    with open('/data/wujiaju/logs/letter_walk_huginn_20260914.log','x') as f:
        child=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        emit('evaluation_started',child_pid=child.pid,gpu=5,command=cmd)
        begun=time.time()
        try:
            while child.poll() is None:
                free=int(subprocess.check_output(['nvidia-smi','-i','5','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if free<16384 or time.time()-begun>1800:raise RuntimeError('Evaluation safety limit')
                print(json.dumps(dict(event='resource',child_pid=child.pid,free_mib=free,time=time.time())),flush=True)
                time.sleep(30)
        except BaseException:
            if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
            raise
    if child.returncode:raise RuntimeError(f'Evaluation exit {child.returncode}; inspect log')
    assert (ROOT/'huginn_results_v1/summary.json').exists()
    emit('complete',summary=str(ROOT/'huginn_results_v1/summary.json'))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        emit('failed',error=repr(e));raise
