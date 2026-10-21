#!/usr/bin/env python3
"""Real-image tracker replay using archived, actually accepted CAD measurements.

DINO/SAM is run once on the exact starting image. FoundationPose results are
replayed from the recorded acceptance run, not recomputed or fabricated.
"""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from teleop_assist.learned_perception import LearnedCabinetPerception
from teleop_assist.visual_tracker import RGBDMaskTracker
from teleop_assist.model_registry import sha256


def load(path):
    with np.load(path,allow_pickle=False) as data:frame={key:data[key].copy() for key in data.files}
    frame['frame_id']=int(frame['frame_id']);frame['timestamp']=float(frame['timestamp'])
    return frame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording',type=Path,default=ROOT/'runs/predictive_stack_acceptance_006')
    parser.add_argument('--start',type=int,default=85)
    parser.add_argument('--end',type=int,default=93)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'report.json').exists():parser.error('Use a new output directory')
    rows={r['frame_id']:r for r in map(json.loads,(args.recording/'frames.jsonl').read_text().splitlines())}
    initial=load(args.recording/'rgbd'/f'{args.start:06d}.npz')
    pipeline=LearnedCabinetPerception(json.loads((ROOT/'configs/perception.interaction.json').read_text()))
    result=pipeline.process(initial['rgb'],initial['depth'],initial['intrinsics'],initial['T_world_camera'],initial['frame_id'],initial['timestamp'])
    target=next(t for t in result['targets'] if t['label']=='handle');identity=target['id'];mask=pipeline.last_masks[identity]
    np.savez(args.output/'initial_mask.npz',mask=mask)
    trackers={name:RGBDMaskTracker() for name in ('sam_expiry_only','accepted_cad_evidence')}
    for tracker in trackers.values():tracker.initialize(initial,[target],{identity:mask})
    records=[]
    for number in range(args.start+1,args.end+1):
        frame=load(args.recording/'rgbd'/f'{number:06d}.npz');record={'frame_id':number,'modes':{}}
        for mode,tracker in trackers.items():
            out=tracker.track(frame);evidence_used=False
            cad=rows[number]['diagnostics'].get('cad_pose',{});pose=cad.get('result',{})
            if mode=='accepted_cad_evidence' and pose.get('accepted'):
                if pose.get('target_id')!=identity:raise RuntimeError('Recorded identity differs from the observed starting instance')
                source=next(t for t in rows[pose['frame_id']]['targets'] if t['id']==identity)
                evidence={'kind':'foundationpose_rgbd','accepted':True,'target_id':identity,
                    'timestamp':pose['timestamp'],'frame_id':pose['frame_id'],'camera_id':cad.get('observation_camera','head'),
                    'center_world':source['center_world'],'quality':pose['quality'],
                    'pose_position_sigma_m':pose['pose_position_sigma_m'],
                    'pose_orientation_sigma_rad':pose['pose_orientation_sigma_rad']}
                evidence_used=tracker.confirm_identity(evidence)
            record['modes'][mode]={'visible':any(t['id']==identity for t in out['targets']),
                'cad_confirmation_consumed':evidence_used,'diagnostics':out['diagnostics'],
                'targets':out['targets']}
        records.append(record)
    report={'scope':'Actual archived RGBD plus newly inferred initial SAM mask; actual archived FoundationPose measurements replayed',
            'recording':str(args.recording),'recorded_events_sha256':sha256(args.recording/'frames.jsonl'),
            'start_frame':args.start,'end_frame':args.end,'selected_id':identity,'records':records,
            'sam_timestamp_changed':False,'rgbd_gates_changed':False,'scene_truth_used':False}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({name:{'visible_frames':[r['frame_id'] for r in records if r['modes'][name]['visible']],
                           'lost_frames':[r['frame_id'] for r in records if not r['modes'][name]['visible']]}
                       for name in trackers}),flush=True)


if __name__=='__main__':main()
