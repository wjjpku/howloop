"""End already-running legacy workers once they reach a cancelled seed; never stop active allowed work."""
import os,signal,time,json
from pathlib import Path
P=Path(__file__).resolve().parents[1];allowed=set(json.loads((P/'cohort.json').read_text())['seeds']);done=set()
while len(done)<2:
 for shard in [0,1]:
  if shard in done:continue
  f=P/f'status_{shard}.json'
  try:s=json.loads(f.read_text())
  except (ValueError,FileNotFoundError):continue
  if s['state'] in ['complete','failed']:done.add(shard);continue
  if s['state']=='waiting_for_backbone' and s['seed'] not in allowed:
   pid=s['worker_pid'];proc=Path(f'/proc/{pid}');cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
   assert str(P/'code/worker.py') in cmd and proc.stat().st_uid==os.getuid()
   expected=[i for i in sorted(allowed) if (i-8)%2==shard]
   assert all((P/f'runs/seed{i}/complete.json').exists() for i in expected)
   os.kill(pid,signal.SIGTERM)
   f.write_text(json.dumps(dict(state='complete',seeds=expected,gpu=s['gpu'],time=time.time(),reason='User limited cohort; skipped remaining backbone seeds'),indent=2));done.add(shard);print(f'Finished shard {shard}: {expected}',flush=True)
 time.sleep(15)
