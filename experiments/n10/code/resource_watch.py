"""Watch only this campaign; terminate its own jobs if GPU reserve is breached."""
import json,os,signal,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];log=Path('/data/paperexperiment/logs/n10_migration_20260923/resource.jsonl')
def stop_own(pid):
 p=Path(f'/proc/{pid}')
 if not p.exists():return
 command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
 if p.stat().st_uid!=os.getuid():return
 try:cwd=(p/'cwd').resolve()
 except FileNotFoundError:return
 if str(cwd).startswith(str(R)) or 'n10_migration_20260923' in command:
  if os.getpgid(pid)==pid:os.killpg(pid,signal.SIGTERM)
  else:os.kill(pid,signal.SIGTERM)
while True:
 done=[]
 for shard,gpu in [(0,3),(1,6)]:
  free,total=map(int,subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(','));reserve=max(16384,int(total*.2));row=dict(time=time.time(),gpu=gpu,free_mib=free,reserve_mib=reserve)
  if free<reserve:
   pids=set()
   for name in [f'worker_backbones_{shard}.json',f'after_backbones_{shard}.json']:
    file=R/name
    if file.exists():
     status=json.loads(file.read_text())
     for key in ['pid','child_pid']:
      if key in status:pids.add(int(status[key]))
     command=status.get('command',[])
     if '--out-dir' in command:
      meta=Path(command[command.index('--out-dir')+1])/'metadata.json'
      if meta.exists():pids.add(int(json.loads(meta.read_text())['pid']))
   for pid in pids:stop_own(pid)
   row.update(status='stopped_own_campaign_reserve_breach',pids=sorted(pids));(R/'RESOURCE_STOP.json').write_text(json.dumps(row,indent=2))
  with log.open('a') as f:f.write(json.dumps(row)+'\n')
  status_file=R/f'after_backbones_{shard}.json';done.append(status_file.exists() and json.loads(status_file.read_text()).get('status') in ['experiments_complete','failed'])
 if all(done):break
 time.sleep(30)
