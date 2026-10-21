#!/usr/bin/env python3
"""Real learned-mask / FoundationPose replay of a recorded mask contamination."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from teleop_assist.model_bridge import ModelProcess
from teleop_assist.foundation_pose import FoundationPoseEstimator
from teleop_assist.visual_tracker import RGBDMaskTracker
from teleop_assist.model_registry import sha256


def load(path):
    with np.load(path,allow_pickle=False) as data:frame={k:data[k].copy() for k in data.files}
    frame['frame_id']=int(frame['frame_id']);frame['timestamp']=float(frame['timestamp'])
    return frame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording',type=Path,default=ROOT/'runs/vision_grasp_acceptance_002')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'report.json').exists():parser.error('Use a new output directory')
    frames={i:load(args.recording/'rgbd'/f'{i:06d}.npz') for i in range(69,80)}
    measured={};identity=None
    with_models=ModelProcess(**json.loads((ROOT/'configs/perception.interaction.json').read_text()))
    try:
        for number in (69,72,75):
            frame=frames[number];bundle={'frames':{'head':frame}}
            for camera in ('right_wrist','left_wrist'):
                bundle['frames'][camera]=load(args.recording/'sensors'/camera/f'{number:06d}.npz')
            result=with_models.process_frame(frame,bundle,identity)
            if identity is None:identity=next(t['id'] for t in result['targets'] if t['label']=='handle')
            measured[number]=result
            np.savez(args.output/f'{number:06d}_semantic_masks.npz',**result['_masks'])
    finally:with_models.close()
    estimator=FoundationPoseEstimator(ROOT/'assets/cad/handle_pose/PullHandleGrasp.obj',
        config={'resolve_x_half_turn_with_world_up':True},debug_dir=args.output/'debug')
    tracker=RGBDMaskTracker();records=[]
    for number in range(69,80):
        source={69:69,74:72,78:75}.get(number)
        if source:
            result=measured[source]
            targets=[t for t in result['targets'] if t['id']==identity]
            if not targets:raise RuntimeError(f'No actual semantic handle in frame {source}')
            tracker.initialize(frames[source],targets,{identity:result['_masks'][identity]})
        frame=frames[number];observed=tracker.track(frame)
        if identity not in tracker.last_masks:raise RuntimeError(f'Observed mask lost in frame {number}: {observed}')
        mask=tracker.last_masks[identity]
        method=estimator.register if number==69 else estimator.track
        result=method(frame['rgb'],frame['depth'],frame['intrinsics'],mask,frame['T_world_camera'],timestamp=frame['timestamp'])
        result.update(frame_id=number,timestamp=frame['timestamp'],source_sha256=sha256(args.recording/'rgbd'/f'{number:06d}.npz'))
        records.append(result)
        target=observed['targets'][0]
        if result['accepted']:
            tracker.confirm_identity({'kind':'foundationpose_rgbd','accepted':True,'target_id':identity,
                'timestamp':frame['timestamp'],'frame_id':number,'camera_id':'head','center_world':target['center_world'],
                'quality':result['quality'],'pose_position_sigma_m':result['pose_position_sigma_m'],
                'pose_orientation_sigma_rad':result['pose_orientation_sigma_rad']})
        filtered=getattr(estimator,'last_mask',mask)
        np.savez(args.output/f'{number:06d}_pose_masks.npz',raw=mask,filtered=filtered,
                 rendered_depth=getattr(estimator,'last_rendered_depth',np.zeros(mask.shape)))
        overlay=frame['rgb'].copy().astype(float)
        overlay[filtered]=.6*overlay[filtered]+.4*np.array([0,255,0])
        overlay[mask & ~filtered]=.3*overlay[mask & ~filtered]+.7*np.array([255,0,255])
        Image.fromarray(overlay.astype(np.uint8)).save(args.output/f'{number:06d}_mask_overlay.png')
        print('POSE_RESULT',number,result['accepted'],result.get('quality'),result.get('mask_prior_filter'),flush=True)
    report={'kind':'Actual pretrained DINO/SAM and FoundationPose inference on recorded RGBD',
        'recording':str(args.recording),'frames':len(records),'accepted':sum(r['accepted'] for r in records),
        'pose_gates_unchanged':True,'freshness_limit_s':.5,'scene_truth_used':False,'records':records,
        'limits':['Freshly inferred masks; original live semantic worker memory is not reconstructed exactly.',
                  'Isolated replay; no concurrent Isaac load. CAD filter does not change general UI mask geometry.']}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('SUMMARY',report['accepted'],report['frames'],flush=True)


if __name__=='__main__':main()
