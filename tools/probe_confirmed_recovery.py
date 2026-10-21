#!/usr/bin/env python3
"""Replay actual confirmed-object SAM recovery; no simulator object truth is read."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from teleop_assist.model_bridge import ModelProcess
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--recording',type=Path,default=ROOT/'runs/predictive_stack_acceptance_005')
parser.add_argument('--frames',type=int,nargs='+',default=[58,62,66,70])
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();out=args.output
if (out/'report.json').exists():parser.error('Use a new output directory to preserve evidence')
out.mkdir(parents=True,exist_ok=True)
worker=ModelProcess(**json.loads((ROOT/'configs/perception.interaction.json').read_text()))
records=[]
try:
 selected=None
 for number in args.frames:
  bundle={'frames':{}}
  for name,folder in (('head','rgbd'),('right_wrist','sensors/right_wrist'),('left_wrist','sensors/left_wrist')):
   path=args.recording/folder/f'{number:06d}.npz'
   if not path.exists() and name!='head':continue
   with np.load(path) as data:frame={k:data[k].copy() for k in data.files}
   frame['timestamp']=float(frame['timestamp']);frame['frame_id']=int(frame['frame_id']);frame['camera_id']=name
   bundle['frames'][name]=frame
  result=worker.process_frame(bundle['frames']['head'],bundle,selected)
  if selected is None:selected=next(t['id'] for t in result['targets'] if t['label']=='handle')
  np.savez(out/f'{number:06d}_masks.npz',**result.pop('_masks'));result.pop('_camera_observations')
  records.append({'frame':number,'selected':selected,'selected_present':any(t['id']==selected for t in result['targets']),'result':result})
  print(number,selected,records[-1]['selected_present'],result['diagnostics'].get('sensors',{}).get('confirmed_identity_recovery'),flush=True)
 (out/'report.json').write_text(json.dumps({'recording':str(args.recording.resolve()),'records':records,'scene_truth_used':False,'gates_relaxed':False},indent=2))
finally:worker.close()
