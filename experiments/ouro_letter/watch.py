import subprocess,time,json,os,signal,argparse
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--out',required=True);a=p.parse_args()
for _ in range(11):
 try:os.kill(a.pid,0)
 except ProcessLookupError:break
 data=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
 row=next(l for l in data.splitlines() if int(l.split(',')[0])==a.gpu);free=int(row.split(',')[2]);entry={'time':time.time(),'pid':a.pid,'gpu':a.gpu,'gpu_row':row,'apps':apps}
 with open(a.out,'a') as f:f.write(json.dumps(entry)+'\n')
 if free<16384:os.kill(a.pid,signal.SIGTERM);break
 time.sleep(30)
