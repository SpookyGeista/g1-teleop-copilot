#!/usr/bin/env python3
"""Real pretrained pose latency on recorded images, with original timestamps."""
import argparse,copy,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from teleop_assist.model_bridge import ModelProcess
from teleop_assist.pose_bridge import PoseProcess,PoseRuntime
from teleop_assist.visual_tracker import RGBDMaskTracker
from teleop_assist.model_registry import sha256


def load(path):
    with np.load(path,allow_pickle=False) as data:frame={key:data[key].copy() for key in data.files}
    frame['frame_id']=int(frame['frame_id']);frame['timestamp']=float(frame['timestamp'])
    return frame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording',type=Path,default=ROOT/'runs/predictive_stack_acceptance_009')
    parser.add_argument('--start',type=int,default=210);parser.add_argument('--end',type=int,default=217)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'report.json').exists():parser.error('Use a new output directory')
    frame=load(args.recording/'rgbd'/f'{args.start:06d}.npz')
    model=ModelProcess(**json.loads((ROOT/'configs/perception.interaction.json').read_text()))
    try:initial=model.process_frame(frame,{'frames':{'head':frame}})
    finally:model.close()
    target=next(target for target in initial['targets'] if target['label']=='handle');identity=target['id']
    tracker=RGBDMaskTracker();tracker.initialize(frame,[target],initial['_masks'])
    runtime=PoseRuntime(PoseProcess(),interval=0.,current_frame_wait_s=.08)
    records=[]
    try:
        runtime.update(frame,[target],tracker.last_masks,identity)
        if runtime.future is not None:runtime.future.result(timeout=30.)
        runtime._complete_future()
        for number in range(args.start+1,args.end+1):
            frame=load(args.recording/'rgbd'/f'{number:06d}.npz')
            observed=tracker.track(frame)
            if len(observed['targets'])!=1:raise RuntimeError('Recorded target tracking failed')
            started=time.monotonic();runtime.update(frame,observed['targets'],tracker.last_masks,identity)
            elapsed=time.monotonic()-started
            if runtime.identity_confirmation:tracker.confirm_identity(runtime.identity_confirmation)
            target=observed['targets'][0]
            record={'frame_id':number,'acquisition_timestamp':frame['timestamp'],
                'input_sha256':sha256(args.recording/'rgbd'/f'{number:06d}.npz'),
                'runtime_update_ms':elapsed*1000,'pose_timestamp':target.get('pose_timestamp'),
                'pose_age_at_frame_s':frame['timestamp']-target.get('pose_timestamp',frame['timestamp']),
                'pose_valid':target.get('pose_valid',False),'diagnostics':copy.deepcopy(runtime.diagnostics)}
            records.append(record)
            print(json.dumps({key:value for key,value in record.items() if key not in ('diagnostics','input_sha256')}),flush=True)
        report={'kind':'Actual FoundationPose inference on actual recorded RGBD with newly inferred SAM initialization',
            'recording':str(args.recording),'selected_id':identity,'records':records,
            'same_frame_accepted':sum(r['diagnostics']['pose_result_consumed_in_same_frame'] and r['pose_valid'] for r in records),
            'warm_frame_count':len(records),'original_timestamps_preserved':True,'freshness_gate_s':.5,
            'limits':['No concurrent Isaac/DINO load during warm pose loop; full-stack validation remains required.']}
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:runtime.close()


if __name__=='__main__':main()
