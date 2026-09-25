import json,time
from pathlib import Path
R=Path(__file__).resolve().parents[1]
backbones=[]
for p in sorted((R/'backbones').glob('*/summary.json')):
 s=json.loads(p.read_text());backbones.append({'run':p.parent.name,'best_step':s['best_step'],'best_selection_accuracy':s['best_selection_endpoint_accuracy'],'elapsed_sec':s['elapsed_sec']})
workers={}
for name in ['worker_backbones_0','worker_backbones_1','after_backbones_0','after_backbones_1']:
 p=R/(name+'.json');workers[name]=json.loads(p.read_text()) if p.exists() else {'status':'not_started'}
report={'updated_unix':time.time(),'status':'in_progress','main_backbones_complete':len(backbones),'main_backbones_total':21,'completed_backbones':backbones,'workers':workers,'paper_recompiled_with_n10':False,'complete_controller_fits':len(list((R/'local').glob('*/hop*/summary.json')))+len(list((R/'continuation').glob('*/summary.json'))),'resource_stop':(R/'RESOURCE_STOP.json').exists()}
(R/'STATUS.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['workers','completed_backbones']},indent=2))
